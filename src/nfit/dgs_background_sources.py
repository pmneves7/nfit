"""Bounded native DGS sources for directional background replay.

Raw sources use the same reduction, calibration and per-run cache as native
binning. The nine-column stream is an internal replay interface, not an MDE
file or a Mantid dependency.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np


def raw_source(group):
    return group.metadata.get("raw_dgs", {}).get("format") == "raw-direct-geometry-nexus"


@contextmanager
def prepared_raw_source(group, source):
    from . import raw_dgs as raw
    from .dgs_reduction_policy import resolved_dgs_reduction_policies
    from .raw_dgs_cache import cached_reduction, reduction_signature
    from .reduction_recipes import effective_reduction_config

    config = effective_reduction_config(group, source)
    signature = reduction_signature(source, config)
    cache = cached_reduction(source, signature) if config.get("cache_reduced_events", True) else None
    if cache is not None:
        with cache.open() as archive:
            header = json.loads(str(archive["header_json"].item()))
            info = raw._run_info_from_cache({**header["run_info"], "path": source.metadata["source_file"]})
            payload = {key: np.asarray(archive[key]) for key in ("detector_ids", "direction", "solid", "charge")}
            yield config, info, payload, archive, None, header.get("hyspec_preprocessing"), signature
    else:
        info = raw.inspect_raw_dgs_run(source.metadata["source_file"],
            monitor_variance_policy=resolved_dgs_reduction_policies(config)["monitor_variance_policy"],
            bad_pulse_threshold=config.get("bad_pulse_threshold", 95.))
        setup = None
        if info.instrument_name.upper() == "HYSPEC":
            import h5py
            with h5py.File(info.path, "r") as handle:
                setup = raw.resolved_hyspec_preprocessing(handle["entry"], config,
                    float(config.get("incident_energy_override") or info.incident_energy),
                    instrument_name=info.instrument_name)
        geometry = raw._detector_geometry(info.path, hyspec_preprocessing=setup)
        path = config.get("normalization_file")
        norm = raw.load_detector_normalization(path) if path else None
        mask = raw._combined_detector_mask(config)
        payload = raw._run_normalization_payload(info, geometry, norm, mask, config)
        yield config, info, payload, None, raw._masked_detector_geometry(geometry, norm, mask), setup, signature


def trajectory_payloads(group, runs, inverse_basis, symmetry=None, *, reference_energy=None, progress_callback=None):
    from . import mdevent, raw_dgs
    from .reduction_runtime import effective_trajectory_energies

    if not raw_source(group):
        return mdevent._trajectory_payloads(group, runs, inverse_basis, symmetry,
            reference_energy=reference_energy, progress_callback=progress_callback)
    energies = effective_trajectory_energies(group, runs,
        (run.metadata["incident_energy"] for run in runs), reference_energy=reference_energy)
    detectors, payloads = [], []
    for run, ei in zip(runs, energies, strict=True):
        with prepared_raw_source(group, run) as (config, info, norm, *_):
            direction = norm["direction"]
            detectors.append((norm["detector_ids"], np.arccos(np.clip(direction[:, 2], -1., 1.)),
                              np.arctan2(direction[:, 1], direction[:, 0]), norm["solid"]))
            canonical = np.linalg.inv(2*np.pi*np.asarray(config["ub_matrix"])) @ raw_dgs._goniometer(info.omega, info.phi, info.chi).T
            bounds = raw_dgs._energy_transfer_bounds(config, float(config.get("incident_energy_override") or info.incident_energy))
            payloads.extend((inverse_basis[:3, :3].T @ operation @ canonical, ei, np.asarray(bounds),
                             float(norm["charge"])*run.fit_weight, len(detectors)-1)
                            for operation in raw_dgs._symmetry_matrices(symmetry))
    return detectors, payloads


def sample_exposures(group, runs):
    """Use accepted pulse charge for raw sources, including beam rejection."""
    if not raw_source(group):
        return np.asarray([float(run.metadata["proton_charge"])*run.fit_weight for run in runs])
    values = []
    for run in runs:
        with prepared_raw_source(group, run) as (_config, _info, norm, *_):
            values.append(float(norm["charge"])*run.fit_weight)
    return np.asarray(values)


@contextmanager
def open_event_stream(group, source, *, rows):
    """Yield orientation, layout and ordered nine-column replay blocks."""
    if not raw_source(group):
        import h5py

        from .mdevent import _read_goniometer_matrix
        config = group.metadata["mdevent"]
        with h5py.File(source.metadata["source_file"], "r") as handle:
            workspace = handle[config["workspace_path"]]
            index = int(source.metadata["mdevent_experiment_index"])
            values = workspace["event_data/event_data"]
            yield _read_goniometer_matrix(workspace[f"experiment{index}"]), values.shape, values.dtype, (
                np.asarray(values[start:start+rows]) for start in range(0, values.shape[0], rows))
        return
    from scipy.spatial import cKDTree

    from .dgs_reduction_policy import ENERGY_TO_K2

    with reduced_event_stream(group, source, rows=rows) as (config, info, norm, chunks):
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        directions = norm["direction"]
        tree = cKDTree(directions)
        # Detector identities cannot be inferred from a ray shared by two pixels.
        if len(directions) > 1 and np.any(tree.query(directions, k=2)[0][:, 1] < 1e-12):
            raise ValueError("Directional raw replay requires distinct detector rays")

        def blocks():
            ki = np.sqrt(ei/ENERGY_TO_K2)
            for events, _raw_count in chunks:
                if not len(events):
                    continue
                kf = np.sqrt((ei-events[:, 3])/ENERGY_TO_K2)
                direction = np.column_stack((-events[:, 0], -events[:, 1], ki-events[:, 2]))/kf[:, None]
                distance, indices = tree.query(direction)
                if np.any(distance > 1e-8):
                    raise ValueError("Reduced background events do not match their detector geometry")
                block = np.zeros((len(events), 9))
                block[:, :2] = events[:, 4:6]
                block[:, 4] = norm["detector_ids"][indices]
                block[:, 5:8], block[:, 8] = events[:, :3], events[:, 3]
                yield block
        yield np.eye(3), (info.event_count, 9), np.dtype(float), blocks()



@contextmanager
def reduced_event_stream(group, source, *, rows):
    """Yield native configuration, run metadata, calibration and cached chunks."""
    from . import raw_dgs as raw
    from .raw_dgs_cache import cache_event_chunks, iter_cached_event_chunks

    with prepared_raw_source(group, source) as (config, info, norm, archive, geometry, setup, signature):
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        if archive is not None:
            chunks = iter_cached_event_chunks(archive, max(1, rows)*48)
        else:
            chunks = raw._iter_reduced_event_chunks(info, config, geometry, ei,
                raw._energy_transfer_bounds(config, ei), max(1, rows)*96, hyspec_preprocessing=setup)
            if config.get("cache_reduced_events", True):
                header = {"run_info": {**asdict(info), "path": str(info.path), "ub_matrix": info.ub_matrix.tolist()},
                          "hyspec_preprocessing": setup}
                chunks = cache_event_chunks(source, signature, header, norm, chunks)
        yield config, info, norm, chunks

def replay_group(source):
    """Restore a native event reader from a persisted covariance recipe."""
    from .pipeline import DatasetEntry, DatasetGroup
    if source.get("source_format") == "raw-direct-geometry-nexus":
        entry = DatasetEntry("Background source", None, kind="raw_dgs_nexus",
            metadata={"source_file": source["source_file"], **source["raw_run_metadata"]})
        config = {**source["raw_dgs_config"], "cache_reduced_events": False}
        return DatasetGroup("Replay source", datasets=[entry], metadata={"raw_dgs": config}), entry
    entry = DatasetEntry("Background source", None, metadata={"source_file": source["source_file"],
        "mdevent_experiment_index": source["experiment_index"]})
    return DatasetGroup("Replay source", metadata={"mdevent": {"workspace_path": source["workspace_path"]}}), entry
