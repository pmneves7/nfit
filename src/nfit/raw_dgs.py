"""Streaming TOF reduction for compatible direct-geometry spectrometer NeXus files.

The implementation deliberately reads one bank (and then one event chunk) at a
time. A compatible run can therefore be combined into an HKLE or radial powder
histogram without materialising its event table. This is an adapter for
direct-geometry spectrometers with the expected NeXus layout, not a generic
reducer for every direct-geometry NeXus file.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .dgs_event_accumulation import accumulate_projected_dgs_events
from .dgs_normalization import (
    DEFAULT_TRAJECTORY_ENERGY_POLICY,
    validated_trajectory_energy_policy,
)
from .dgs_reduction_policy import (
    DEFAULT_EVENT_PRECISION_POLICY,
    DEFAULT_MONITOR_VARIANCE_POLICY,
    DEFAULT_SYMMETRY_VARIANCE_POLICY,
    dgs_histogram_edges,
    dgs_powder_coordinates,
    dgs_uses_mantid_trajectory_grid,
    prepare_dgs_event_projector,
    resolved_dgs_reduction_policies,
    validated_monitor_variance_policy,
)
from .event_covariance import accumulate_copy_covariance
from .histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from .mdevent import (
    _MDEVENT_NUMBA,
    ENERGY_TO_K2,
    MDEVENT_TRAJECTORY_BATCH_TASKS,
    _accumulate_detector_trajectory,
    _accumulate_discrete_event_coordinates,
    _accumulate_powder_detector_trajectory,
    _dgs_grid_memory,
    _requested_edges,
    _symmetry_matrices,
    _trajectory_worker_count,
    _validated_minimum_samples,
    load_detector_normalization,
)
from .mdhisto import MDHistoAxis, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup
from .raw_dgs_cache import (
    RAW_DGS_REDUCTION_VERSION,
    cache_event_chunks,
    cached_reduction,
    iter_cached_event_chunks,
    reduction_signature,
)
from .raw_dgs_geometry import (
    evaluate_log_expression,
    location_transform,
    resolved_geometry_signature,
    resolved_idf_xml,
    source_distance,
)
from .raw_dgs_geometry_precision import (
    HE3_EFFICIENCY_EXPONENTIAL_CONSTANT as HE3_EFFICIENCY_EXPONENTIAL_CONSTANT,
)
from .raw_dgs_geometry_precision import (
    mantid_cylinder_radius,
    mantid_he3_exponent,
)
from .raw_dgs_goniometer import GONIOMETER_AVERAGING_CONVENTION, sample_rotation
from .raw_dgs_hyspec import (
    hyspec_default_detector_keep,
    raw_hyspec_tof_keep,
    resolved_hyspec_preprocessing,
)
from .raw_dgs_monitors import (
    TOF_US_PER_M_SQRT_MEV,
    _mantid_getei_v2_peak,
)
from .raw_dgs_monitors import (
    _mantid_getei_peak_region as _mantid_getei_peak_region,
)
from .raw_dgs_pulses import select_pulses
from .reduction_runtime import (
    acquisition_identity,
    effective_trajectory_energies,
    record_resolved_reduction,
)
from .resource_budget import memory_guard
from .source_lineage import source_lineage_metadata

# Mantid's parameter files select these formula-driven GetEi v2 paths instead
# of fitting two monitor peaks. The formulas are instrument definitions, not
# empirical corrections, and use incident energy in meV.
MANTID_T0_FORMULAS = {
    "CNCS": "288.595706061-110.833514059/sqrt(incidentEnergy)+89.6080314589/incidentEnergy-42.6999684563*sqrt(incidentEnergy)+1.89672170078*incidentEnergy",
    "HYSPEC": "4.0 + (107.0 / (1.0 + (incidentEnergy / 31.0)^3))",
}

_DGS_INSTRUMENT_NAMES = frozenset({"ARCS", "CNCS", "HYSPEC", "SEQUOIA"})
_NON_DGS_INSTRUMENT_NAMES = frozenset({
    "WAND", "WAND2", "WAND^2", "HB2C", "HB2A", "CORELLI", "MACS",
    "CG2", "CG3", "GP-SANS", "BIOSANS", "DMC", "D33", "SANS-I", "SANS1",
})


@dataclass(frozen=True)
class RawDGSRunInfo:
    path: Path
    run_number: str
    event_count: int
    incident_energy: float
    proton_charge: float
    omega: float
    phi: float
    chi: float
    l1: float
    t0: float
    ub_matrix: np.ndarray
    calibration_source: str = "unspecified"
    calibration_warning: str | None = None
    instrument_name: str = "unknown"
    geometry_signature: str | None = None
    goniometer_logs_vary: bool = False


_RUN_INFO_CACHE_MAXSIZE = 2048
_RUN_INFO_CACHE: OrderedDict[tuple, tuple[RawDGSRunInfo, bool]] = OrderedDict()
_RUN_INFO_CACHE_LOCK = RLock()


def is_raw_dgs_nexus_file(path: str | Path) -> bool:
    """Identify compatible DGS acquisitions rather than arbitrary event banks."""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            entry = handle.get("entry")
            if entry is None or not any(
                name.startswith("bank") and name.endswith("_events") and "event_id" in entry[name]
                for name in entry
            ):
                return False
            root = ET.fromstring(entry["instrument/instrument_xml/data"][()].tobytes())
            name = str(root.get("name", "")).upper()
            if name in _NON_DGS_INSTRUMENT_NAMES:
                return False
            # Classification precedes numerical validation: a malformed Ei
            # must fail in the DGS importer, rather than become ordinary data.
            energy = _log_value(entry, ("EnergyRequest", "Ei", "BL17:Det:TH:BL:Ei"), None)
            return name in _DGS_INSTRUMENT_NAMES or energy is not None
    except (ImportError, OSError, KeyError, TypeError, ValueError, ET.ParseError):
        return False


def inspect_raw_dgs_run(
    path: str | Path, *, monitor_variance_policy: str = DEFAULT_MONITOR_VARIANCE_POLICY,
    bad_pulse_threshold: float = 0.0,
) -> RawDGSRunInfo:
    """Read metadata, reusing successful calibration for an unchanged source.

    File identity is checked on every call. Returned UB matrices remain private,
    writable copies; the bounded cache retains no monitor or detector arrays.
    Failed calibration is retried so its warning behavior remains unchanged.
    Sample angles use time means after pause/bad-pulse filtering; changing the
    pulse threshold rereads only varying angle logs, without recalibration.
    """

    monitor_variance_policy = validated_monitor_variance_policy(monitor_variance_policy)
    bad_pulse_threshold = float(bad_pulse_threshold)
    if not np.isfinite(bad_pulse_threshold) or bad_pulse_threshold < 0.0:
        raise ValueError("bad_pulse_threshold must be finite and nonnegative")
    source = Path(path)
    resolved = source.resolve()
    stat = resolved.stat()
    key = (
        str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns, monitor_variance_policy,
    )
    with _RUN_INFO_CACHE_LOCK:
        cached = _RUN_INFO_CACHE.get(key)
        if cached is not None:
            _RUN_INFO_CACHE.move_to_end(key)
    if cached is None:
        info, filename_run_number = _inspect_raw_dgs_run_uncached(
            source, monitor_variance_policy=monitor_variance_policy,
        )
        if info.calibration_warning is None:
            ub_matrix = info.ub_matrix.copy()
            ub_matrix.setflags(write=False)
            retained = replace(info, path=resolved, ub_matrix=ub_matrix)
            with _RUN_INFO_CACHE_LOCK:
                _RUN_INFO_CACHE[key] = retained, filename_run_number
                _RUN_INFO_CACHE.move_to_end(key)
                while len(_RUN_INFO_CACHE) > _RUN_INFO_CACHE_MAXSIZE:
                    _RUN_INFO_CACHE.popitem(last=False)
    else:
        info, filename_run_number = cached
    if bad_pulse_threshold > 0.0 and info.goniometer_logs_vary:
        import h5py

        with h5py.File(source, "r") as handle:
            angles = sample_rotation(handle["entry"], bad_pulse_threshold=bad_pulse_threshold)
        info = replace(info, omega=angles.omega, phi=angles.phi, chi=angles.chi)
    return replace(
        info, path=source, ub_matrix=info.ub_matrix.copy(),
        run_number=source.stem if filename_run_number else info.run_number,
    )


def _inspect_raw_dgs_run_uncached(
    path: str | Path, *, monitor_variance_policy: str,
) -> tuple[RawDGSRunInfo, bool]:
    """Inspect and calibrate one run without retaining its HDF5 arrays."""

    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        entry = handle["entry"]
        event_count = sum(
            int(group["event_id"].shape[0])
            for name, group in entry.items()
            if name.startswith("bank") and name.endswith("_events") and "event_id" in group
        )
        requested_ei = _log_value(entry, ("EnergyRequest", "Ei", "BL17:Det:TH:BL:Ei"), 0.0)
        calibration = {}
        calibrated_ei, calibrated_t0 = _monitor_ei_t0(
            entry, requested_ei, diagnostics=calibration, variance_policy=monitor_variance_policy,
        )
        identity = acquisition_identity(entry)
        identity["geometry_signature"] = resolved_geometry_signature(entry)
        missing_run_number = object()
        run_number = _text_scalar(entry.get("run_number"), missing_run_number)
        filename_run_number = run_number is missing_run_number
        angles = sample_rotation(entry)
        return RawDGSRunInfo(
            path=source,
            run_number=source.stem if filename_run_number else run_number,
            event_count=event_count,
            incident_energy=calibrated_ei,
            proton_charge=_raw_proton_charge_uah(entry),
            omega=angles.omega,
            phi=angles.phi,
            chi=angles.chi,
            goniometer_logs_vary=angles.varying_logs,
            l1=_source_distance(entry),
            t0=calibrated_t0,
            ub_matrix=_ub_from_logs(entry),
            calibration_source=calibration["source"],
            calibration_warning=calibration.get("warning"),
            **identity,
        ), filename_run_number


def raw_dgs_dataset_group(
    paths: Iterable[str | Path],
    *,
    normalization_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    name: str | None = None,
    trajectory_energy_policy: str = DEFAULT_TRAJECTORY_ENERGY_POLICY,
    monitor_variance_policy: str = DEFAULT_MONITOR_VARIANCE_POLICY,
    event_precision_policy: str = DEFAULT_EVENT_PRECISION_POLICY,
    symmetry_variance_policy: str = DEFAULT_SYMMETRY_VARIANCE_POLICY,
    progress_callback: Any | None = None,
) -> DatasetGroup:
    """Create lightweight raw-run entries sharing reduction and sample setup."""

    trajectory_energy_policy = validated_trajectory_energy_policy(trajectory_energy_policy)
    policies = resolved_dgs_reduction_policies({
        "monitor_variance_policy": monitor_variance_policy,
        "event_precision_policy": event_precision_policy,
        "symmetry_variance_policy": symmetry_variance_policy,
    })
    resolved_paths = [Path(path) for path in paths]
    if resolved_paths:
        from .corelli import (
            corelli_dataset_group,
            inspect_corelli_run,
            is_corelli_raw_nexus_file,
        )

        if is_corelli_raw_nexus_file(resolved_paths[0]):
            first_corelli = inspect_corelli_run(resolved_paths[0])
            corelli_infos = [first_corelli]
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "raw_dgs_import",
                        "iteration": 1,
                        "total": len(resolved_paths),
                        "message": (
                            "reading CORELLI run metadata "
                            f"1/{len(resolved_paths):,}"
                        ),
                    }
                )
            for index, path in enumerate(resolved_paths[1:], start=2):
                try:
                    corelli_infos.append(inspect_corelli_run(path))
                except ValueError as error:
                    raise ValueError(
                        "CORELLI correlation runs cannot share a group with "
                        "fixed-Ei raw runs"
                    ) from error
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "raw_dgs_import",
                            "iteration": index,
                            "total": len(resolved_paths),
                            "message": (
                                "reading CORELLI run metadata "
                                f"{index:,}/{len(resolved_paths):,}"
                            ),
                        }
                    )
            return corelli_dataset_group(
                resolved_paths,
                solid_angle_path=normalization_path,
                mask_path=mask_path,
                name=name,
                progress_callback=progress_callback,
                _run_infos=corelli_infos,
            )

    infos = [inspect_raw_dgs_run(path, monitor_variance_policy=policies["monitor_variance_policy"])
             for path in resolved_paths]
    if not infos:
        raise ValueError("select at least one raw direct-geometry NeXus file")
    first = infos[0]
    energy_lower = min(-0.95 * info.incident_energy for info in infos)
    energy_upper = max(0.95 * info.incident_energy for info in infos)
    q_upper = max(
        math.sqrt(info.incident_energy / ENERGY_TO_K2)
        + math.sqrt((info.incident_energy - energy_lower) / ENERGY_TO_K2)
        for info in infos
    )
    hkl_transform = np.linalg.inv(2.0 * np.pi * first.ub_matrix)
    hkl_limits = q_upper * np.sum(np.abs(hkl_transform), axis=1)
    shared = {
        "format": "raw-direct-geometry-nexus",
        "source_files": [str(info.path) for info in infos],
        "event_count": sum(info.event_count for info in infos),
        "ub_matrix": first.ub_matrix.tolist(),
        "normalization_file": None if normalization_path is None else str(normalization_path),
        "mask_file": None if mask_path is None else str(mask_path),
        "incident_energy_override": None,
        "trajectory_energy_policy": trajectory_energy_policy,
        **policies,
        "t0_override": None,
        "hyspec_default_mask": True,
        "hyspec_tof_crop": True,
        "hyspec_tank_offset_override": None,
        "energy_min_fraction": -0.95,
        "energy_max_fraction": 0.95,
        "bad_pulse_threshold": 95.0,
        "ki_kf_normalization": True,
        "he3_detector_efficiency_correction": True,
        "normalization": "proton_charge_and_detector_trajectory",
        "dimensions": [
            {"name": name, "lower": -q_upper, "upper": q_upper}
            for name in ("Q_lab_x", "Q_lab_y", "Q_lab_z")
        ]
        + [{"name": "DeltaE", "lower": energy_lower, "upper": energy_upper}],
        "hkl_bounds": [
            (-float(limit), float(limit)) for limit in hkl_limits
        ]
        + [(energy_lower, energy_upper)],
        "q_modulus_bounds": [0.0, q_upper],
    }
    datasets = []
    for index, info in enumerate(infos, start=1):
        datasets.append(
            DatasetEntry(
                name=f"run {info.run_number}",
                data=None,
                kind="raw_dgs_nexus",
                data_type="single_crystal_inelastic",
                metadata={
                    "source_file": str(info.path),
                    "run_number": info.run_number,
                    "event_count": info.event_count,
                    "incident_energy": info.incident_energy,
                    "calibration_source": info.calibration_source,
                    "calibration_warning": info.calibration_warning,
                    "instrument_name": info.instrument_name,
                    "geometry_signature": info.geometry_signature,
                    "proton_charge": info.proton_charge,
                    "omega": info.omega,
                    "phi": info.phi,
                    "chi": info.chi,
                    "l1": info.l1,
                    "t0": info.t0,
                    "import_status": "pending",
                },
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "raw_dgs_import",
                    "iteration": index,
                    "total": len(infos),
                    "message": f"reading raw run metadata {index:,}/{len(infos):,}",
                }
            )
    group = DatasetGroup(
        name=name or first.path.stem, datasets=datasets, metadata={"raw_dgs": shared}
    )
    from .reduction_recipes import ensure_reduction_recipe

    ensure_reduction_recipe(group)
    return group


@memory_guard(_dgs_grid_memory, operation="Reducing and binning raw DGS events")
def bin_raw_dgs_group(
    group: DatasetGroup,
    *,
    lower: Iterable[float],
    upper: Iterable[float],
    num_bins: Iterable[int],
    step_size: Iterable[float] | None = None,
    bin_edges: Iterable[Iterable[float] | None] | None = None,
    minimum_samples: float = 0.0,
    datasets: Iterable[DatasetEntry] | None = None,
    vectors: Iterable[Iterable[float]] | None = None,
    axis_names: Iterable[str] | None = None,
    max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: Any | None = None,
    symmetry_operations: Iterable[Iterable[Iterable[float]]] | None = None,
    coordinate_mode: str = "hkle",
    fractional_axes: Iterable[bool] | None = None,
    metadata_dimensions: Iterable[Any] | None = None,
) -> MDHistoData:
    """Reduce raw direct-geometry event banks into an HKLE histogram.

    Accepted events carry Mantid's direct-geometry He-3 detector-efficiency
    and ki/kf corrections and are divided by an MDNorm-style detector-
    trajectory denominator. The retained proton charge is integrated from the
    raw pulse log in microampere-hours.
    Shiver's ``NormFilename`` convention is preserved here: non-positive
    detector values mask events, while positive values supply MDNorm's
    detector solid-angle weights.
    """

    config = group.metadata["raw_dgs"]
    if config.get("format") == "corelli-correlation-nexus":
        from .corelli import bin_corelli_group

        return bin_corelli_group(
            group,
            lower=lower,
            upper=upper,
            num_bins=num_bins,
            step_size=step_size,
            bin_edges=bin_edges,
            minimum_samples=minimum_samples,
            datasets=datasets,
            vectors=vectors,
            axis_names=axis_names,
            max_batch_bytes=max_batch_bytes,
            progress_callback=progress_callback,
            symmetry_operations=symmetry_operations,
            coordinate_mode=coordinate_mode,
            fractional_axes=fractional_axes,
            metadata_dimensions=metadata_dimensions,
        )
    policies = resolved_dgs_reduction_policies(config)
    selected = list(group.datasets if datasets is None else datasets)
    from .dataset_criteria import filter_dataset_criteria

    selected = filter_dataset_criteria(group, selected, compute=True,
                                       progress_callback=progress_callback)
    for run in selected:
        if not np.isfinite(run.scale_factor) or not np.isfinite(run.fit_weight) or run.fit_weight < 0:
            raise ValueError("raw-DGS scales must be finite and fit weights finite and nonnegative")
    selected = [run for run in selected if run.enabled and run.fit_weight > 0]
    if not selected:
        raise ValueError("raw-DGS binning requires an enabled run with positive fit weight")
    if coordinate_mode not in {"hkle", "powder"}:
        raise ValueError("raw direct-geometry coordinate mode must be 'hkle' or 'powder'")
    powder = coordinate_mode == "powder"
    expected_dimensions = 2 if powder else 4
    lo, hi, bins = (
        np.asarray(tuple(values), dtype=dtype)
        for values, dtype in ((lower, float), (upper, float), (num_bins, int))
    )
    expected_shape = (expected_dimensions,)
    if (
        lo.shape != expected_shape
        or hi.shape != expected_shape
        or bins.shape != expected_shape
        or np.any(bins <= 0)
    ):
        label = "|Q| and energy" if powder else "four"
        raise ValueError(
            f"raw direct-geometry {coordinate_mode} binning requires {label} positive bin counts"
        )
    if powder and lo[0] < 0.0:
        raise ValueError("powder |Q| lower bound must be nonnegative")
    requested_edges = _requested_edges(lo, hi, bins, step_size, bin_edges=bin_edges)
    edges = dgs_histogram_edges(requested_edges, policy=policies["event_precision_policy"])
    mantid_trajectory_precision = dgs_uses_mantid_trajectory_grid(
        requested_edges, policies["event_precision_policy"]
    )
    minimum_samples = _validated_minimum_samples(minimum_samples)
    shape = tuple(edge.size - 1 for edge in edges)
    basis = None
    basis_inverse = None
    symmetry = (np.eye(3),)
    names = ("|Q|", "DeltaE")
    if not powder:
        basis = (
            np.eye(4)
            if vectors is None
            else np.asarray(tuple(tuple(v) for v in vectors), dtype=float)
        )
        if basis.shape != (4, 4) or np.linalg.matrix_rank(basis) != 4:
            raise ValueError("raw direct-geometry coordinate axes must form an invertible 4D basis")
        if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1.0:
            raise ValueError("raw direct-geometry momentum axes cannot mix energy")
        basis_inverse = np.linalg.inv(basis)
        symmetry = _symmetry_matrices(symmetry_operations)
        names = tuple(
            axis_names or (_axis_name(row, index) for index, row in enumerate(basis))
        )
    data_sum = np.zeros(shape)
    variance_sum = np.zeros(shape)
    event_count = np.zeros(shape)
    # GenerateDGSMDE uses NormFilename to mask raw events, and MakeSlice passes
    # the same workspace to MDNorm as its SolidAngleWorkspace.  The event
    # numerator is therefore unscaled while the trajectory denominator carries
    # the positive processed-vanadium values.
    detector_norm = None
    detector_mask = None
    calibration_payloads = {}
    total = sum(int(dataset.metadata.get("event_count", 0)) for dataset in selected)
    processed = 0
    energy_bounds_by_dataset_id: dict[str, tuple[float, float]] = {}
    run_infos_by_dataset_id: dict[str, RawDGSRunInfo] = {}
    resolved_energy_windows = []
    normalization_payloads_by_dataset_id = {}
    cache_hits = 0
    cache_misses = 0
    copy_covariance = {"within_bin_pairs_corrected": 0, "cross_bin_pairs_unrepresented": 0, "cross_terms_added": 0.0}
    projectors = () if powder else tuple(
        prepare_dgs_event_projector(config["ub_matrix"], basis, operation, requested_edges,
            policy=policies["event_precision_policy"])
        for operation in symmetry
    )
    from .reduction_recipes import effective_reduction_config

    for dataset in selected:
        run_config = effective_reduction_config(group, dataset)
        run_policies = resolved_dgs_reduction_policies(run_config)
        source = Path(dataset.metadata["source_file"])
        use_cache = bool(run_config.get("cache_reduced_events", True))
        signature = reduction_signature(dataset, run_config) if use_cache else None
        cache = cached_reduction(dataset, signature) if use_cache else None
        with cache.open() if cache is not None else nullcontext() as archive:
            if cache is not None:
                header = json.loads(str(archive["header_json"].item()))
                info = _run_info_from_cache({**header["run_info"], "path": str(source)})
                hyspec_setup = header.get("hyspec_preprocessing")
                normalization_payload = {
                    key: np.asarray(archive[key])
                    for key in ("detector_ids", "direction", "solid", "charge")
                }
                cache_hits += 1
            else:
                calibration_key = (run_config.get("normalization_file"), run_config.get("mask_file"))
                if calibration_key not in calibration_payloads:
                    normalization_path = run_config.get("normalization_file")
                    calibration_payloads[calibration_key] = (
                        load_detector_normalization(normalization_path) if normalization_path else None,
                        _combined_detector_mask(run_config),
                    )
                detector_norm, detector_mask = calibration_payloads[calibration_key]
                info = inspect_raw_dgs_run(source,
                    monitor_variance_policy=run_policies["monitor_variance_policy"],
                    bad_pulse_threshold=run_config.get("bad_pulse_threshold", 95.0))
                hyspec_setup = None
                if str(info.instrument_name).upper() == "HYSPEC":
                    import h5py

                    with h5py.File(source, "r") as handle:
                        hyspec_setup = resolved_hyspec_preprocessing(
                            handle["entry"], run_config,
                            float(run_config.get("incident_energy_override") or info.incident_energy),
                            instrument_name=info.instrument_name,
                        )
                geometry = _detector_geometry(source, hyspec_preprocessing=hyspec_setup)
                normalization_payload = _run_normalization_payload(
                    info, geometry, detector_norm, detector_mask, run_config
                )
                cache_misses += 1
            run_infos_by_dataset_id[dataset.id] = info
            # Cache unweighted reduction; apply run weights only to this binning.
            normalization_payloads_by_dataset_id[dataset.id] = {
                **normalization_payload,
                "charge": np.asarray(normalization_payload["charge"]) * dataset.fit_weight,
            }
            ei = float(run_config.get("incident_energy_override") or info.incident_energy)
            if ei <= 0.0 or info.l1 <= 0.0:
                raise ValueError(f"{source.name} has no usable incident energy or source distance")
            energy_bounds = _energy_transfer_bounds(run_config, ei)
            record_resolved_reduction(dataset, run_config,
                automatic={"incident_energy_meV": info.incident_energy, "t0_microseconds": info.t0,
                    "omega_degrees": info.omega, "phi_degrees": info.phi, "chi_degrees": info.chi,
                    **({"hyspec_tank_offset_degrees": hyspec_setup["tank_offset_degrees"]}
                       if hyspec_setup is not None else {})},
                provenance={"algorithm_version": RAW_DGS_REDUCTION_VERSION, "calibration_source": info.calibration_source,
                    "calibration_warning": info.calibration_warning, "instrument_name": info.instrument_name,
                    "geometry_signature": info.geometry_signature, "reduction_signature": signature,
                    "goniometer_averaging": GONIOMETER_AVERAGING_CONVENTION,
                    **({"hyspec_preprocessing": hyspec_setup} if hyspec_setup is not None else {})})
            energy_bounds_by_dataset_id[dataset.id] = energy_bounds
            resolved_energy_windows.append(
                {
                    "dataset_id": dataset.id,
                    "run_number": info.run_number,
                    "incident_energy_meV": ei,
                    "minimum_meV": energy_bounds[0],
                    "maximum_meV": energy_bounds[1],
                }
            )
            gonio = _goniometer(info.omega, info.phi, info.chi)
            signal_factor = float(dataset.scale_factor) * float(dataset.fit_weight)

            def accumulate(chunks, *, gonio=gonio, cache=cache, signal_factor=signal_factor):
                nonlocal processed
                for events, raw_count in chunks:
                    q_lab, energy, weights = events[:, :3], events[:, 3], events[:, 4] * signal_factor
                    variances = events[:, 5] * signal_factor**2
                    if powder:
                        coordinate_blocks = (
                            dgs_powder_coordinates(np.linalg.norm(q_lab, axis=1), energy,
                                requested_edges, policy=policies["event_precision_policy"]),
                        )
                    else:
                        if policies["event_precision_policy"] == "mantid":
                            q_sample = np.zeros_like(q_lab)
                            for index in range(3):
                                q_sample += q_lab[:, index, None] * gonio[index]
                        else:
                            q_sample = q_lab @ gonio
                    copy_bins = np.full((len(symmetry), len(weights)), -1, dtype=np.int64) if (
                        not powder and len(symmetry) > 1
                        and policies["symmetry_variance_policy"] == "within_bin_covariance"
                    ) else None
                    if powder:
                        for coords, accumulation_edges in coordinate_blocks:
                            _accumulate_discrete_event_coordinates(
                                coords, weights, variances, accumulation_edges, shape,
                                data_sum, variance_sum, event_count,
                            )
                    else:
                        for copy_index, projector in enumerate(projectors):
                            accumulate_projected_dgs_events(
                                projector, q_sample, energy, weights, variances, shape,
                                data_sum, variance_sum, event_count,
                                fallback_accumulator=_accumulate_discrete_event_coordinates,
                                bin_indices=None if copy_bins is None else copy_bins[copy_index],
                            )
                    if copy_bins is not None:
                        terms = accumulate_copy_covariance(copy_bins, variances, variance_sum)
                        for key in copy_covariance:
                            copy_covariance[key] += terms[key]
                    processed += raw_count
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "stage": "raw_dgs_events",
                                "iteration": processed,
                                "total": total,
                                "message": (
                                    f"binning cached events {processed:,}/{total:,}"
                                    if cache is not None else
                                    f"reducing raw events {processed:,}/{total:,}"
                                ),
                            }
                        )

            if cache is not None:
                accumulate(iter_cached_event_chunks(archive, max_batch_bytes))
            else:
                chunks = _iter_reduced_event_chunks(
                    info, run_config, _masked_detector_geometry(geometry, detector_norm, detector_mask),
                    ei, energy_bounds, max_batch_bytes,
                    hyspec_preprocessing=hyspec_setup,
                )
                if use_cache:
                    header = {"run_info": {
                        **asdict(info), "path": str(info.path), "ub_matrix": info.ub_matrix.tolist(),
                    }, "hyspec_preprocessing": hyspec_setup}
                    chunks = cache_event_chunks(
                        dataset, signature, header, normalization_payload, chunks
                    )
                accumulate(chunks)
    if powder:
        normalization = _powder_trajectory_normalization(
            group,
            selected,
            edges,
            shape,
            detector_norm,
            detector_mask,
            energy_bounds_by_dataset_id,
            run_infos_by_dataset_id=run_infos_by_dataset_id,
            normalization_payloads_by_dataset_id=normalization_payloads_by_dataset_id,
            mantid_precision=mantid_trajectory_precision,
            **(
                {"progress_callback": progress_callback}
                if progress_callback is not None
                else {}
            ),
        )
    else:
        normalization = _trajectory_normalization(
            group,
            selected,
            edges,
            shape,
            basis_inverse,
            detector_norm,
            detector_mask,
            symmetry,
            energy_bounds_by_dataset_id,
            run_infos_by_dataset_id=run_infos_by_dataset_id,
            normalization_payloads_by_dataset_id=normalization_payloads_by_dataset_id,
            mantid_precision=mantid_trajectory_precision,
            **(
                {"progress_callback": progress_callback}
                if progress_callback is not None
                else {}
            ),
        )
    covered = normalization > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = data_sum / normalization
        errors = np.sqrt(variance_sum) / normalization
    total_events = float(event_count.sum())
    rms = float(np.sqrt(max(0.0, variance_sum.sum() - copy_covariance["cross_terms_added"]) / total_events)) if total_events else 1.0
    mask = ~covered | (event_count < minimum_samples)
    if powder:
        axes = (
            MDHistoAxis("|Q|", edges[0], "1/angstrom", "momentum", frame="Q modulus"),
            MDHistoAxis("DeltaE", edges[1], "meV", "energy", frame="General Frame"),
        )
    else:
        axes = tuple(
            MDHistoAxis(
                name,
                edge,
                "meV" if index == 3 else "r.l.u.",
                "energy" if index == 3 else "momentum",
                frame="General Frame" if index == 3 else "HKL",
            )
            for index, (name, edge) in enumerate(zip(names, edges, strict=True))
        )
    for output in (signal, errors, mask, event_count, normalization, data_sum, variance_sum):
        output.setflags(write=False)
    return MDHistoData(
        axes=axes,
        signal=signal,
        errors=errors,
        mask=mask,
        num_events=event_count,
        metadata={
            **source_lineage_metadata(selected, include_disabled=True),
            "raw_dgs": copy.deepcopy(config),
            "resolved_run_reductions": {dataset.id: copy.deepcopy(dataset.metadata.get("resolved_reduction")) for dataset in selected},
            "dgs_reduction_policies": policies,
            "reduced_event_cache": {"hits": cache_hits, "misses": cache_misses},
            "raw_dgs_energy_windows_meV": resolved_energy_windows,
            "raw_dgs_calibration": {
                dataset_id: {
                    "source": info.calibration_source,
                    "warning": info.calibration_warning,
                    "incident_energy_meV": info.incident_energy,
                    "t0_microseconds": info.t0,
                }
                for dataset_id, info in run_infos_by_dataset_id.items()
            },
            "rebin": {
                **({"vectors": basis.tolist()} if basis is not None else {}),
                "bin_edges": [edge.tolist() for edge in edges],
                "minimum_samples": minimum_samples,
            },
            "signal_semantics": "density",
            "signal_semantics_source": (
                "nfit_raw_tof_powder_reduction"
                if powder
                else "nfit_raw_tof_reduction"
            ),
            "normalization_denominator": normalization,
            "zero_event_bins_are_measured": True,
            "zero_count_error_model": "observed_event_variance",
            EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA),
            "symmetry_covariance": {"policy": policies["symmetry_variance_policy"], **copy_covariance},
            "event_weight_rms": rms,
            **(
                {
                    "powder_reduction": {
                        "coordinates": "|Q|,DeltaE",
                        "normalization": "proton_charge_and_detector_trajectory",
                    }
                }
                if powder
                else {
                    "symmetry_operations_hkl": [
                        operation.tolist() for operation in symmetry
                    ]
                }
            ),
            "ki_kf_normalization": _use_ki_kf_correction(config),
            "he3_detector_efficiency_correction": bool(
                config.get("he3_detector_efficiency_correction", True)
            ),
            "proton_charge_units": "microampere-hour (retained raw pulse charge in picocoulombs divided by 3.6e9)",
        },
        auxiliary_channels=event_statistics_channels(data_sum, variance_sum, normalization),
    )


def bin_raw_dgs_powder_group(
    group: DatasetGroup,
    *,
    lower: Iterable[float],
    upper: Iterable[float],
    num_bins: Iterable[int],
    step_size: Iterable[float] | None = None,
    bin_edges: Iterable[Iterable[float] | None] | None = None,
    minimum_samples: float = 0.0,
    datasets: Iterable[DatasetEntry] | None = None,
    max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: Any | None = None,
    fractional_axes: Iterable[bool] | None = None,
) -> MDHistoData:
    """Reduce raw direct-geometry detector events to ``|Q|, DeltaE``."""

    return bin_raw_dgs_group(
        group,
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        step_size=step_size,
        bin_edges=bin_edges,
        minimum_samples=minimum_samples,
        datasets=datasets,
        max_batch_bytes=max_batch_bytes,
        progress_callback=progress_callback,
        coordinate_mode="powder",
        fractional_axes=fractional_axes,
    )


def _run_info_from_cache(payload):
    return RawDGSRunInfo(
        **{**payload, "path": Path(payload["path"]), "ub_matrix": np.asarray(payload["ub_matrix"])},
    )


def _inspect_configured_run(dataset, group):
    from .reduction_recipes import effective_reduction_config

    config = effective_reduction_config(group, dataset)
    return inspect_raw_dgs_run(dataset.metadata["source_file"],
        monitor_variance_policy=resolved_dgs_reduction_policies(config)["monitor_variance_policy"],
        bad_pulse_threshold=config.get("bad_pulse_threshold", 95.0))


def _run_normalization_payload(info, geometry, detector_norm, detector_mask, config):
    direction = geometry.positions / np.linalg.norm(geometry.positions, axis=1)[:, None]
    solid = (
        np.ones(geometry.detector_ids.size)
        if detector_norm is None else detector_norm.value_for_ids(geometry.detector_ids)
    )
    if detector_mask is not None:
        solid[detector_mask.value_for_ids(geometry.detector_ids) <= 0.0] = 0.0
    solid[~hyspec_default_detector_keep(geometry.detector_ids, instrument_name=info.instrument_name,
        enabled=config.get("hyspec_default_mask", True))] = 0.0
    import h5py

    with h5py.File(info.path, "r") as handle:
        charge = _retained_proton_charge_uah(
            handle["entry"], float(config.get("bad_pulse_threshold", 95.0))
        )
    return {"detector_ids": geometry.detector_ids, "direction": direction,
            "solid": solid, "charge": np.asarray(charge)}


def _iter_reduced_event_chunks(info, config, geometry, ei, energy_bounds, max_batch_bytes,
                               *, hyspec_preprocessing=None):
    """Yield Q_lab (Å⁻¹), DeltaE (meV), corrected weight and its variance."""
    import h5py

    t0 = float(config.get("t0_override") if config.get("t0_override") is not None else info.t0)
    mantid_precision = resolved_dgs_reduction_policies(config)["event_precision_policy"] == "mantid"
    rows = max(1, int(max_batch_bytes) // 96)
    with h5py.File(info.path, "r") as handle:
        pulse_selection = select_pulses(handle["entry"], float(config.get("bad_pulse_threshold", 95.0)))
        for bank_name in handle["entry"]:
            if not (bank_name.startswith("bank") and bank_name.endswith("_events")):
                continue
            bank = handle["entry"][bank_name]
            if "event_id" not in bank:
                continue
            pulse_keep = pulse_selection.bank_keep(bank)
            ids, tofs = bank["event_id"], bank["event_time_offset"]
            event_index = np.asarray(bank["event_index"], dtype=np.int64) if pulse_keep is not None else None
            for start in range(0, ids.shape[0], rows):
                stop = min(start + rows, ids.shape[0])
                event_ids = np.asarray(ids[start:stop], dtype=np.int64)
                raw_tof = np.asarray(tofs[start:stop], dtype=float)
                event_tof = raw_tof - t0
                indices, exponents, valid = geometry.event_indices_for_ids(
                    event_ids, mantid_precision=config.get("he3_detector_efficiency_correction", True),
                )
                if str(info.instrument_name).upper() == "HYSPEC":
                    valid &= hyspec_default_detector_keep(event_ids, instrument_name=info.instrument_name,
                        enabled=config.get("hyspec_default_mask", True))
                if hyspec_preprocessing is not None:
                    valid &= raw_hyspec_tof_keep(raw_tof, hyspec_preprocessing)
                if pulse_keep is not None:
                    pulse_index = np.searchsorted(event_index, np.arange(start, stop), side="right") - 1
                    valid &= pulse_keep[np.clip(pulse_index, 0, pulse_keep.size - 1)]
                indices, exponents, tof = indices[valid], exponents[valid], event_tof[valid]
                exceptional_geometry = geometry.has_exceptional_positions
                l2 = (
                    np.linalg.norm(geometry.positions[indices], axis=1)
                    if exceptional_geometry else geometry.distances[indices]
                )
                final_tof = tof - TOF_US_PER_M_SQRT_MEV * info.l1 / math.sqrt(ei)
                good = final_tof > 0.0
                indices, exponents = indices[good], exponents[good]
                l2, final_tof = l2[good], final_tof[good]
                ef = (TOF_US_PER_M_SQRT_MEV * l2 / final_tof) ** 2
                energy = ei - ef
                kf_energy = ei - energy if mantid_precision else ef
                kf = np.sqrt(np.maximum(kf_energy, 0.0) / ENERGY_TO_K2)
                keep = (energy >= energy_bounds[0]) & (energy <= energy_bounds[1])
                direction = (
                    geometry.positions[indices[keep]] / l2[keep, None]
                    if exceptional_geometry else geometry.directions[indices[keep]]
                )
                kf, energy, exponents = kf[keep], energy[keep], exponents[keep]
                q_lab = np.column_stack((
                    -kf * direction[:, 0], -kf * direction[:, 1],
                    math.sqrt(ei / ENERGY_TO_K2) - kf * direction[:, 2],
                ))
                weights = np.ones(energy.size, dtype=np.float32 if mantid_precision else float)
                variances = np.ones_like(weights)
                if config.get("he3_detector_efficiency_correction", True):
                    correction = _he3_tube_efficiency_correction(kf, exponents, mantid_precision=True)
                    if mantid_precision:
                        correction = correction.astype(np.float32)
                    weights *= correction
                    variances *= correction * correction
                if _use_ki_kf_correction(config):
                    correction = np.sqrt(ei / (ei - energy))
                    if mantid_precision:
                        correction = correction.astype(np.float32)
                    weights *= correction
                    variances *= correction * correction
                yield np.column_stack((q_lab, energy, weights, variances)), stop - start


def _trajectory_normalization(
    group,
    datasets,
    edges,
    shape,
    basis_inverse,
    detector_norm,
    detector_mask,
    symmetry_operations=None,
    energy_bounds_by_dataset_id=None,
    *,
    progress_callback=None,
    run_infos_by_dataset_id=None,
    normalization_payloads_by_dataset_id=None,
    mantid_precision=False,
):
    """Native MDNorm-style detector trajectories for compatible direct-geometry runs."""
    config = group.metadata["raw_dgs"]
    result = np.zeros(shape)
    symmetry = _symmetry_matrices(symmetry_operations)
    payloads = []
    detector_payload = None
    shared_detector_geometry = True
    datasets = list(datasets)
    infos = run_infos_by_dataset_id if run_infos_by_dataset_id is not None else {
        dataset.id: _inspect_configured_run(dataset, group)
        for dataset in datasets
    }
    trajectory_energies = effective_trajectory_energies(
        group, datasets, (infos[dataset.id].incident_energy for dataset in datasets)
    )
    for dataset, ei in zip(datasets, trajectory_energies, strict=True):
        info = infos[dataset.id]
        from .reduction_recipes import effective_reduction_config

        run_config = effective_reduction_config(group, dataset)
        snapshot = (
            _run_normalization_payload(info, _detector_geometry(info.path), detector_norm, detector_mask, run_config)
            if normalization_payloads_by_dataset_id is None
            else normalization_payloads_by_dataset_id[dataset.id]
        )
        direction, solid = snapshot["direction"], snapshot["solid"]
        charge = float(snapshot["charge"])
        ub = np.asarray(config["ub_matrix"], dtype=float)
        canonical_inverse = (
            np.linalg.inv(2.0 * np.pi * ub) @ _goniometer(info.omega, info.phi, info.chi).T
        )
        inverses = [
            basis_inverse[:3, :3].T @ operation @ canonical_inverse for operation in symmetry
        ]
        energy_bounds = (
            _energy_transfer_bounds(run_config, float(
                run_config.get("incident_energy_override") or info.incident_energy
            ))
            if energy_bounds_by_dataset_id is None
            else energy_bounds_by_dataset_id[dataset.id]
        )
        theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
        phi = np.arctan2(direction[:, 1], direction[:, 0])
        current_detector_payload = (snapshot["detector_ids"], theta, phi, solid)
        if detector_payload is None:
            detector_payload = current_detector_payload
        else:
            shared_detector_geometry &= all(
                first.shape == current.shape and np.array_equal(first, current)
                for first, current in zip(
                    detector_payload,
                    current_detector_payload,
                    strict=True,
                )
            )
        payloads.extend(
            (inverse, ei, energy_bounds, charge, direction, solid)
            for inverse in inverses
        )
    accelerated = _MDEVENT_NUMBA is not None
    if accelerated and shared_detector_geometry:
        from .dgs_trajectory_tasks import pool_trajectory_tasks

        payloads = pool_trajectory_tasks(
            payloads, edges[3], checked_shared_geometry=True,
        )
    task_total = sum(int(np.count_nonzero(item[5] > 0.0)) for item in payloads)
    workers = _trajectory_worker_count(int(np.prod(shape))) if accelerated else 1
    completed = 0

    def report_progress():
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "raw_dgs_normalization",
                    "iteration": completed,
                    "total": task_total,
                    "message": (
                        f"integrated {completed:,}/{task_total:,} detector trajectories"
                    ),
                    "workers": workers,
                    "output_bins": int(np.prod(shape)),
                }
            )

    report_progress()
    if accelerated:
        edge_arrays = tuple(np.asarray(edge) for edge in edges)
        shape_array = np.asarray(shape, dtype=np.int64)
        factory = getattr(_MDEVENT_NUMBA, "trajectory_normalization_accumulator", None)
        accumulator = (
            factory(*edge_arrays, shape_array, workers=workers)
            if factory is not None else None
        )

        def accumulate_batch(theta, phi, solid, batch):
            args = (
                theta, phi, solid,
                np.asarray([item[0] for item in batch]),
                np.asarray([item[1] for item in batch]),
                np.asarray([item[2] for item in batch]),
                np.asarray([item[3] for item in batch]),
                *edge_arrays, shape_array, mantid_precision,
            )
            if accumulator is not None:
                accumulator.accumulate(*args)
            else:
                flat = _MDEVENT_NUMBA.run_trajectory_normalization(*args, workers=workers)
                result[:] += np.asarray(flat).reshape(shape)

        if detector_payload is not None and shared_detector_geometry:
            _, theta, phi, solid = detector_payload
            payload_batch = max(1, MDEVENT_TRAJECTORY_BATCH_TASKS // max(int(theta.size), 1))
            for start in range(0, len(payloads), payload_batch):
                batch = payloads[start:start + payload_batch]
                accumulate_batch(theta, phi, solid, batch)
                completed += sum(int(np.count_nonzero(item[5] > 0.0)) for item in batch)
                report_progress()
        else:
            # The accumulator receives each geometry's own angles and weights.
            # Worker grids depend only on the output grid, not the instrument.
            for payload in payloads:
                direction, solid = payload[4:6]
                theta = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
                phi = np.arctan2(direction[:, 1], direction[:, 0])
                accumulate_batch(theta, phi, solid, [payload])
                completed += int(np.count_nonzero(solid > 0.0))
                report_progress()
        if accumulator is not None:
            return np.asarray(accumulator.result()).reshape(shape)
        return result
    report_stride = max(task_total // 100, 1)
    for inverse, ei, energy_bounds, charge, direction, solid in payloads:
        for index in np.flatnonzero(solid > 0.0):
            _accumulate_detector_trajectory(
                result, edges, inverse, direction[index], ei, energy_bounds, charge * solid[index],
                mantid_precision=mantid_precision,
            )
            completed += 1
            if completed == task_total or completed % report_stride == 0:
                report_progress()
    return result


def _powder_trajectory_normalization(
    group,
    datasets,
    edges,
    shape,
    detector_norm,
    detector_mask,
    energy_bounds_by_dataset_id,
    *,
    progress_callback=None,
    run_infos_by_dataset_id=None,
    normalization_payloads_by_dataset_id=None,
    mantid_precision=False,
):
    """Accumulate the radial MDNorm-style denominator for raw runs."""

    result = np.zeros(shape)
    payloads = []
    datasets = list(datasets)
    infos = run_infos_by_dataset_id if run_infos_by_dataset_id is not None else {
        dataset.id: _inspect_configured_run(dataset, group)
        for dataset in datasets
    }
    trajectory_energies = effective_trajectory_energies(
        group, datasets, (infos[dataset.id].incident_energy for dataset in datasets)
    )
    for dataset, incident_energy in zip(datasets, trajectory_energies, strict=True):
        info = infos[dataset.id]
        from .reduction_recipes import effective_reduction_config

        run_config = effective_reduction_config(group, dataset)
        snapshot = (
            _run_normalization_payload(info, _detector_geometry(info.path), detector_norm, detector_mask, run_config)
            if normalization_payloads_by_dataset_id is None
            else normalization_payloads_by_dataset_id[dataset.id]
        )
        direction, solid = snapshot["direction"], snapshot["solid"]
        charge = float(snapshot["charge"])
        energy_bounds = energy_bounds_by_dataset_id[dataset.id]
        scattering_angles = np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
        payloads.append(
            (incident_energy, energy_bounds, charge, scattering_angles, solid)
        )
    task_total = sum(int(np.count_nonzero(item[4] > 0.0)) for item in payloads)
    accelerated = (
        _MDEVENT_NUMBA is not None
        and hasattr(_MDEVENT_NUMBA, "run_powder_trajectory_normalization")
    )
    workers = _trajectory_worker_count(int(np.prod(shape))) if accelerated else 1
    completed = 0

    def report_progress():
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "raw_dgs_normalization",
                    "iteration": completed,
                    "total": task_total,
                    "message": (
                        f"integrated {completed:,}/{task_total:,} "
                        "powder detector trajectories"
                    ),
                    "workers": workers,
                    "output_bins": int(np.prod(shape)),
                }
            )

    report_progress()
    if accelerated:
        for incident_energy, energy_bounds, charge, scattering_angles, solid in payloads:
            flat = _MDEVENT_NUMBA.run_powder_trajectory_normalization(
                scattering_angles,
                solid,
                np.asarray([incident_energy]),
                np.asarray([energy_bounds]),
                np.asarray([charge]),
                np.asarray(edges[0]),
                np.asarray(edges[1]),
                np.asarray(shape, dtype=np.int64),
                mantid_precision,
                workers=workers,
            )
            result += np.asarray(flat).reshape(shape)
            completed += int(np.count_nonzero(solid > 0.0))
            report_progress()
        return result
    report_stride = max(task_total // 100, 1)
    for incident_energy, energy_bounds, charge, scattering_angles, solid in payloads:
        for detector_index in np.flatnonzero(solid > 0.0):
            _accumulate_powder_detector_trajectory(
                result,
                edges,
                float(scattering_angles[detector_index]),
                incident_energy,
                energy_bounds,
                charge * float(solid[detector_index]),
                mantid_precision=mantid_precision,
            )
            completed += 1
            if completed == task_total or completed % report_stride == 0:
                report_progress()
    return result


def _combined_detector_mask(config):
    paths = [config.get("normalization_file"), config.get("mask_file")]
    masks = [
        load_detector_normalization(path) for path in dict.fromkeys(path for path in paths if path)
    ]
    if not masks:
        return None
    ids = np.unique(np.concatenate([mask.detector_ids for mask in masks]))
    values = np.ones(ids.size)
    for mask in masks:
        values[mask.value_for_ids(ids) <= 0.0] = 0.0
    return type(masks[0])(Path("combined_mask"), ids, values, np.zeros(ids.size))


def _energy_transfer_bounds(config, incident_energy):
    """Resolve and validate the per-run energy-transfer limits in meV."""

    ei = float(incident_energy)
    minimum_fraction = float(config.get("energy_min_fraction", -0.95))
    maximum_fraction = float(config.get("energy_max_fraction", 0.95))
    if not np.isfinite(ei) or ei <= 0.0:
        raise ValueError("incident energy must be positive and finite")
    if not np.isfinite(minimum_fraction) or not np.isfinite(maximum_fraction):
        raise ValueError("raw direct-geometry energy limits must be finite")
    if minimum_fraction >= maximum_fraction:
        raise ValueError("raw direct-geometry energy minimum must be below its maximum")
    if maximum_fraction >= 1.0:
        raise ValueError(
            "raw direct-geometry energy maximum must be below 1 Ei so final energy remains positive"
        )
    return minimum_fraction * ei, maximum_fraction * ei


def _good_pulses(entry, threshold):
    return select_pulses(entry, threshold).charge_keep


def _pulse_charge_values(entry):
    logs = entry.get("DASlogs")
    if logs is None:
        return None
    charge_log = logs.get("proton_charge")
    if charge_log is None or "value" not in charge_log:
        return None
    values = np.asarray(charge_log["value"], dtype=float).reshape(-1)
    return values if values.size else None


def _raw_proton_charge_uah(entry):
    """Return the unfiltered run charge in Mantid's microampere-hour units."""

    values = _pulse_charge_values(entry)
    if values is not None:
        return float(np.sum(values) / 3.6e9)
    if "proton_charge" in entry:
        raw_charge = np.asarray(entry["proton_charge"], dtype=float).reshape(-1)
        if raw_charge.size:
            return float(raw_charge[0] / 3.6e9)
    return _log_value(entry, ("gd_prtn_chrg",), 1.0)


def _retained_proton_charge_uah(entry, threshold):
    """Integrate the same pulse-charge series used to retain raw events."""

    values = _pulse_charge_values(entry)
    if values is None:
        return _raw_proton_charge_uah(entry)
    keep = _good_pulses(entry, threshold)
    if keep is not None:
        values = values[keep]
    return float(np.sum(values) / 3.6e9)


def _use_ki_kf_correction(config):
    """Read the corrected setting while accepting projects saved by nfit 0.4.x."""

    return bool(config.get("ki_kf_normalization", config.get("kf_ki_normalization", True)))


@dataclass(frozen=True)
class _DetectorGeometry:
    detector_ids: np.ndarray
    positions: np.ndarray
    he3_exponents: np.ndarray
    mantid_he3_exponents: np.ndarray | None = None

    def __post_init__(self):
        if self.mantid_he3_exponents is None:
            object.__setattr__(self, "mantid_he3_exponents", self.he3_exponents)
        for name, dtype in (("detector_ids", np.int64), ("positions", float),
                            ("he3_exponents", float), ("mantid_he3_exponents", float)):
            values = np.asarray(getattr(self, name), dtype=dtype)
            if values.flags.writeable:
                values = values.copy()
                values.setflags(write=False)
            object.__setattr__(self, name, values)

    @cached_property
    def _sorted_ids_and_order(self):
        order = np.argsort(self.detector_ids)
        sorted_ids = self.detector_ids[order]
        order.setflags(write=False)
        sorted_ids.setflags(write=False)
        return sorted_ids, order

    @cached_property
    def _event_geometry(self):
        """Precompute exact arithmetic without changing selected-event errors."""

        # An unused exceptional pixel must not raise or warn during lookup.
        # Those snapshots retain the original selected-event arithmetic below.
        exceptional = False
        with np.errstate(all="raise"):
            try:
                distances = np.linalg.norm(self.positions, axis=1)
                directions = self.positions / distances[:, None]
            except FloatingPointError:
                exceptional = True
                with np.errstate(all="ignore"):
                    distances = np.linalg.norm(self.positions, axis=1)
                    directions = self.positions / distances[:, None]
        distances.setflags(write=False)
        directions.setflags(write=False)
        exceptional |= bool(np.any(~np.isfinite(distances) | (distances == 0)))
        return distances, directions, exceptional

    @property
    def distances(self):
        """Sample-to-detector distances for this immutable geometry snapshot."""

        return self._event_geometry[0]

    @property
    def directions(self):
        """Detector unit directions, using the event converter's arithmetic."""

        return self._event_geometry[1]

    @property
    def has_exceptional_positions(self):
        """Whether selected events need the original geometry error handling."""

        return self._event_geometry[2]

    def event_indices_for_ids(
        self, ids: np.ndarray, *, mantid_precision=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve event membership and validate shape before event filtering.

        Indices address only this geometry's immutable rows. Complete resolved
        geometry and static detector selection determine the owning snapshot;
        detector IDs or instrument names never identify these cached values.
        """

        sorted_ids, order = self._sorted_ids_and_order
        found = np.searchsorted(sorted_ids, ids)
        valid = found < sorted_ids.size
        valid[valid] &= sorted_ids[found[valid]] == ids[valid]
        indices = np.zeros(ids.size, dtype=np.int64)
        exponents = np.zeros(ids.size, dtype=float)
        indices[valid] = order[found[valid]]
        source = self.mantid_he3_exponents if mantid_precision else self.he3_exponents
        exponents[valid] = source[indices[valid]]
        if mantid_precision and np.any(~np.isfinite(exponents[valid])):
            raise ValueError(
                "Mantid He-3 correction does not support this detector cylinder shape. "
                "Changing event precision does not change the supported geometry."
            )
        return indices, exponents, valid

    def event_geometry_for_ids(
        self, ids: np.ndarray, *, mantid_precision=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices, exponents, valid = self.event_indices_for_ids(
            ids, mantid_precision=mantid_precision,
        )
        positions = np.zeros((ids.size, 3), dtype=float)
        positions[valid] = self.positions[indices[valid]]
        return positions, exponents, valid


def _masked_detector_geometry(geometry, detector_norm, detector_mask):
    """Apply static detector selection once per run, before event lookups."""

    enabled = np.ones(geometry.detector_ids.size, dtype=bool)
    for selection in (detector_norm, detector_mask):
        if selection is not None:
            enabled &= selection.value_for_ids(geometry.detector_ids) > 0.0
    if np.all(enabled):
        return geometry
    return _DetectorGeometry(
        geometry.detector_ids[enabled], geometry.positions[enabled],
        geometry.he3_exponents[enabled], geometry.mantid_he3_exponents[enabled],
    )


def _detector_geometry(path: Path, *, hyspec_preprocessing=None) -> _DetectorGeometry:
    """Resolve run geometry and reuse it only for identical complete definitions."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            xml = resolved_idf_xml(handle["entry"])
        if hyspec_preprocessing is not None and hyspec_preprocessing["tank_offset_degrees"]:
            from .raw_dgs_hyspec import rotated_hyspec_idf_xml

            xml = rotated_hyspec_idf_xml(xml, hyspec_preprocessing["tank_offset_degrees"])
        return _detector_geometry_from_xml(xml)
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from error


@lru_cache(maxsize=8)
def _detector_geometry_from_xml(xml: bytes) -> _DetectorGeometry:
    """Keep a bounded set of immutable geometries, keyed by the full IDF."""

    root = ET.fromstring(xml)
    namespace = root.tag.split("}")[0] + "}"
    types = {item.get("name"): item for item in root.findall(f"{namespace}type")}
    idlists = {item.get("idname"): item for item in root.findall(f"{namespace}idlist")}
    he3_parameters = _idf_he3_parameters(root, namespace)
    ids: list[int] = []
    positions: list[np.ndarray] = []
    he3_exponents: list[float] = []
    mantid_exponents: list[float] = []
    cylinder_shapes = {}
    for component in root.findall(f"{namespace}component"):
        idname = component.get("idlist")
        if not idname or idname not in idlists:
            continue
        component_type = component.get("type")
        leaf_positions = []
        for location in component.findall(f"{namespace}location") or [None]:
            translation, rotation = _location_transform(location)
            leaf_positions.extend(_expand_type(
                component_type, types, rotation, translation, namespace,
                he3_parameters, he3_parameters.get(component.get("name") or component_type),
                cylinder_shapes,
            ))
        detector_ids = _expand_idlist(idlists[idname], namespace)
        if len(leaf_positions) != len(detector_ids):
            continue
        ids.extend(detector_ids)
        positions.extend(item[0] for item in leaf_positions)
        for position, axis, radius, parameters, ray_radius in leaf_positions:
            nominal_exponent = _he3_exponent(position, axis, radius, parameters)
            he3_exponents.append(nominal_exponent)
            if nominal_exponent == 0.0:
                mantid_exponents.append(0.0)
            elif ray_radius is None:
                mantid_exponents.append(float("nan"))
            else:
                mantid_exponents.append(mantid_he3_exponent(position, axis, ray_radius, parameters))
    if not ids:
        raise ValueError("instrument XML did not define detector pixel positions")
    geometry = _DetectorGeometry(
        np.asarray(ids, dtype=np.int64),
        np.asarray(positions, dtype=float),
        np.asarray(he3_exponents, dtype=float),
        np.asarray(mantid_exponents, dtype=float),
    )
    return geometry


def _expand_type(name, types, rotation, translation, ns, he3_parameters, inherited_he3,
                 cylinder_shapes):
    item = types.get(name)
    if item is None:
        return []
    leaves = []
    for component in item.findall(f"{ns}component"):
        child = component.get("type")
        child_he3 = he3_parameters.get(component.get("name") or child, inherited_he3)
        for location in component.findall(f"{ns}location") or [None]:
            local_translation, local_rotation = _location_transform(location)
            child_rotation = rotation @ local_rotation
            child_translation = translation + rotation @ local_translation
            if child in types and types[child].get("is") == "detector":
                if child not in cylinder_shapes:
                    cylinder_shapes[child] = _idf_detector_cylinder_precisions(types[child])
                axis, radius, ray_radius = cylinder_shapes[child]
                leaves.append((child_translation, child_rotation @ axis, radius, child_he3, ray_radius))
            else:
                leaves.extend(
                    _expand_type(
                        child,
                        types,
                        child_rotation,
                        child_translation,
                        ns,
                        he3_parameters,
                        child_he3,
                        cylinder_shapes,
                    )
                )
    return leaves


def _idf_he3_parameters(root, ns):
    """Return He-3 tube parameters keyed by IDF component-link name."""

    result = {}
    for link in root.findall(f"{ns}component-link"):
        values = {}
        for parameter in link.findall(f"{ns}parameter"):
            value = parameter.find(f"{ns}value")
            if value is not None and value.get("val") is not None:
                try:
                    values[parameter.get("name")] = float(value.get("val"))
                except ValueError:
                    continue
        required = ("tube_pressure", "tube_thickness", "tube_temperature")
        if link.get("name") and all(key in values for key in required):
            result[link.get("name")] = tuple(values[key] for key in required)
    return result


def _idf_detector_cylinder(detector_type):
    cylinder = detector_type.find("{*}cylinder")
    if cylinder is None:
        return np.array([0.0, 1.0, 0.0]), 0.0
    axis_element = cylinder.find("{*}axis")
    axis = (
        np.array([float(axis_element.get(key, 0.0)) for key in ("x", "y", "z")])
        if axis_element is not None
        else np.array([0.0, 1.0, 0.0])
    )
    length = np.linalg.norm(axis)
    if length <= 0.0:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis /= length
    radius = cylinder.find("{*}radius")
    return axis, float(radius.get("val", 0.0)) if radius is not None else 0.0


def _idf_detector_cylinder_precisions(detector_type):
    """Parse both nominal and compatibility radii once per detector type."""
    axis, radius = _idf_detector_cylinder(detector_type)
    cylinder = detector_type.find("{*}cylinder")
    if cylinder is None:
        return axis, radius, radius
    if (cylinder.find("{*}rotate") is not None
            or detector_type.find("{*}rotate-all") is not None):
        return axis, radius, None
    bottom = cylinder.find("{*}centre-of-bottom-base")
    if bottom is not None and float(bottom.get("r", 0.0)) != 0.0:
        return axis, radius, None
    height = cylinder.find("{*}height")
    # Historical/incomplete IDFs omit height. Keep their nominal-radius path.
    if height is None:
        return axis, radius, radius
    centre_bottom = ([float(bottom.get(key, 0.0)) for key in ("x", "y", "z")]
                     if bottom is not None else [0.0, 0.0, 0.0])
    return axis, radius, mantid_cylinder_radius(radius, axis, centre_bottom, height.get("val"))


def _he3_exponent(position, axis, radius, parameters):
    if parameters is None or radius <= 0.0:
        return 0.0
    pressure, thickness, temperature = parameters
    direction_length = np.linalg.norm(position)
    axis_length = np.linalg.norm(axis)
    straight_path = 2.0 * (radius - thickness)
    if (
        pressure <= 0.0
        or temperature <= 0.0
        or straight_path <= 0.0
        or direction_length <= 0.0
        or axis_length <= 0.0
    ):
        return 0.0
    cosine = float(np.dot(axis, position) / (axis_length * direction_length))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    if sine <= 1e-12:
        return 0.0
    return HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (pressure / temperature) * straight_path / sine


def _he3_tube_efficiency_correction(kf, exponents, *, mantid_precision=False):
    """Mantid He3TubeEfficiency correction at the final neutron wavelength."""

    correction = np.ones(np.asarray(kf).shape, dtype=float)
    active = np.isfinite(exponents) & (exponents > 0.0) & np.isfinite(kf) & (kf > 0.0)
    if np.any(active):
        wavelength = 2.0 * np.pi / np.asarray(kf)[active]
        alpha = np.asarray(exponents)[active] * wavelength
        denominator = 1.0 - np.exp(-alpha) if mantid_precision else -np.expm1(-alpha)
        correction[active] = 1.0 / denominator
    return correction


def _location_transform(location):
    """Compatibility wrapper for the authoritative IDF location service."""
    return location_transform(location)


def _expand_idlist(item, ns):
    values = []
    for element in item.findall(f"{ns}id"):
        if element.get("start") is None:
            continue
        start = int(element.get("start"))
        end = int(element.get("end", start))
        step = int(element.get("step", 1))
        values.extend(range(start, end + (1 if step > 0 else -1), step))
    return values


def _log_value(entry, names, default):
    logs = entry.get("DASlogs")
    if logs is None:
        return default
    for name in names:
        group = logs.get(name)
        if group is not None:
            data = group.get("average_value") or group.get("value")
            if data is not None:
                values = np.asarray(data[()]).reshape(-1)
                if values.size:
                    return float(values[0])
    return default


def _source_distance(entry):
    """Compatibility wrapper for the resolved moderator position service."""
    return source_distance(entry)


def _monitor_ei_t0(entry, energy_guess, *, diagnostics=None, variance_policy=DEFAULT_MONITOR_VARIANCE_POLICY):
    """Estimate Ei/T0 from an embedded-IDF direct-geometry monitor layout.

    This follows Shiver's Mantid ``GetEi`` route. The two monitor groups are
    discovered from the raw file's IDF, then Mantid's current v2 peak-width,
    rebinning, background, and first-moment calculation is reproduced locally.
    The first two usable monitor spectra in IDF order are used, matching
    Mantid's standard spectrum ordering.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["source"] = "requested_energy_no_usable_monitors"
    has_monitor_data = False

    def failed_calibration(reason, *, source="requested_energy_failed_monitor_calibration"):
        message = (
            f"Ei/T0 calibration failed for {entry.file.filename}: {reason}; "
            f"using requested Ei={energy_guess:g} meV and T0=0 microseconds. "
            "Set explicit Ei/T0 overrides before reducing this run."
        )
        diagnostics.update(source=source, warning=message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return float(energy_guess), 0.0

    try:
        if energy_guess <= 0.0:
            return float(energy_guess), 0.0
        xml_data = entry["instrument/instrument_xml/data"][()]
        root = ET.fromstring(xml_data.tobytes().decode())
        instrument_name = str(root.get("name", "")).upper()
        formula = _mantid_t0_formula(root, instrument_name)
        if formula is not None:
            try:
                t0 = _evaluate_mantid_t0_formula(formula, energy_guess)
            except (SyntaxError, TypeError, ValueError, ArithmeticError) as error:
                return failed_calibration(
                    f"instrument T0 formula could not be evaluated: {error}",
                    source="requested_energy_failed_t0_formula",
                )
            diagnostics["source"] = "instrument_t0_formula"
            return float(energy_guess), t0
        locations = []
        for component in root.findall(".//{*}component[@type='monitor']"):
            for location in component.findall("{*}location"):
                locations.append((location.get("name"), _idf_location(entry, location)))
        source_z = -_source_distance(entry)
        idf_names = {name for name, _ in locations if name}
        monitor_groups = sorted(
            (
                name
                for name, group in entry.items()
                if "event_time_offset" in group
                and (
                    getattr(group, "attrs", {}).get("NX_class", b"") in (b"NXmonitor", "NXmonitor")
                    or name in idf_names
                )
            ),
            key=_natural_sort_key,
        )
        monitor_data = []
        for index, (idf_name, position) in enumerate(locations):
            name = (
                idf_name
                if idf_name in monitor_groups
                else (monitor_groups[index] if index < len(monitor_groups) else None)
            )
            if name is None:
                continue
            values = np.asarray(entry[f"{name}/event_time_offset"], dtype=float)
            if values.size:
                source_to_monitor = np.linalg.norm(position - np.array([0.0, 0.0, source_z]))
                monitor_data.append((name, values, float(source_to_monitor)))
        if len(monitor_data) < 2:
            return float(energy_guess), 0.0
        has_monitor_data = True

        grid_start = min(float(np.min(values)) for _, values, _ in monitor_data)
        peaks = []
        for name, values, distance in monitor_data:
            peak_centre = _mantid_getei_v2_peak(values, distance, energy_guess,
                grid_start=grid_start, variance_policy=variance_policy)
            if peak_centre is None:
                continue
            peaks.append((name, distance, peak_centre))
        if len(peaks) < 2:
            return failed_calibration("fewer than two monitor peaks were found")
        _, left_distance, left_time = peaks[0]
        _, right_distance, right_time = peaks[1]
        elapsed = right_time - left_time
        if elapsed == 0.0:
            return failed_calibration("monitor peaks have coincident flight times")
        velocity = (right_distance - left_distance) * 1.0e6 / elapsed
        if not np.isfinite(velocity) or velocity <= 0.0:
            return failed_calibration("monitor peak positions imply an invalid incident velocity")
        energy = (velocity * TOF_US_PER_M_SQRT_MEV / 1.0e6) ** 2
        if not np.isfinite(energy) or energy <= 0.0:
            return failed_calibration("monitor peak positions imply an invalid incident energy")
        t0 = left_time - left_distance * 1.0e6 / velocity
        diagnostics["source"] = "monitor_peak_calibration"
        return float(energy), float(t0)
    except (KeyError, TypeError, ValueError, ET.ParseError) as error:
        if has_monitor_data:
            return failed_calibration(str(error))
        return float(energy_guess), 0.0


def _mantid_t0_formula(root, instrument_name):
    """Read a Mantid ``t0_formula`` from an IDF or supported parameter set."""

    for parameter in root.findall(".//{*}parameter[@name='t0_formula']"):
        value = parameter.find("{*}value")
        if value is not None and value.get("val"):
            return str(value.get("val"))
    return MANTID_T0_FORMULAS.get(instrument_name)


def _evaluate_mantid_t0_formula(formula, incident_energy):
    """Evaluate Mantid's arithmetic t0_formula, including muParser's power syntax."""

    tree = ast.parse(str(formula).replace("^", "**"), mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Call,
    )
    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(f"unsupported Mantid t0_formula: {formula!r}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {"incidentEnergy", "sqrt"}:
            raise ValueError(f"unsupported Mantid t0_formula name: {node.id!r}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id != "sqrt"
        ):
            raise ValueError(f"unsupported Mantid t0_formula call: {formula!r}")
    result = float(
        eval(
            compile(tree, "<mantid-t0-formula>", "eval"),
            {"__builtins__": {}},
            {
                "incidentEnergy": float(incident_energy),
                "sqrt": math.sqrt,
            },
        )
    )
    if not math.isfinite(result):
        raise ValueError(f"nonfinite Mantid t0_formula result: {formula!r}")
    return result




def _idf_location(entry, location):
    """Compatibility wrapper for static and run-log-driven IDF positions."""
    return location_transform(location, entry=entry)[0]


def _evaluate_idf_log_expression(expression, value):
    """Compatibility wrapper for supported IDF logfile arithmetic."""
    return evaluate_log_expression(expression, value)


def _natural_sort_key(name):
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(name))
    )


def _interpolated_half_height_time(histogram, edges, maximum, half_height, direction):
    """Return Mantid GetEi's interpolated half-height crossing for one side."""

    index = maximum
    while 0 < index < histogram.size - 1 and histogram[index] > half_height:
        index += direction
    if index <= 0 or index >= histogram.size - 1:
        return None
    centre = 0.5 * (edges[index] + edges[index + 1])
    previous = index - direction
    gradient = (histogram[index] - histogram[previous]) / (edges[index] - edges[previous])
    if gradient == 0.0:
        return centre
    return float(centre - (histogram[index] - half_height) / gradient)


def _ub_from_logs(entry):
    logs = entry.get("DASlogs")
    if logs is not None and "BL17:CS:CrystalAlign:UBMatrix" in logs:
        raw = _text_scalar(logs["BL17:CS:CrystalAlign:UBMatrix"].get("value"), "")
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        if len(numbers) >= 9:
            return np.asarray(numbers[:9], dtype=float).reshape(3, 3)
    return np.eye(3)


def _text_scalar(dataset, default):
    if dataset is None:
        return default
    value = np.asarray(dataset[()]).reshape(-1)
    if not value.size:
        return default
    first = value[0]
    return first.decode() if isinstance(first, bytes) else str(first)


def _goniometer(omega, phi, chi):
    # SEQUOIA's NeXus/Mantid convention has beam along lab +z and vertical +y.
    # Its recorded omega matrix is therefore a rotation around lab y.
    def rot(axis, degrees):
        angle = math.radians(degrees)
        c, s = math.cos(angle), math.sin(angle)
        if axis == 1:
            return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        if axis == 2:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])

    return rot(1, omega) @ rot(2, chi) @ rot(1, phi)


def _axis_name(vector, index):
    if index == 3:
        return "DeltaE"
    labels = ("H", "K", "L")
    return (
        "["
        + ",".join(
            "0" if value == 0 else label if value == 1 else f"{value:g}{label}"
            for value, label in zip(vector[:3], labels, strict=True)
        )
        + "]"
    )
