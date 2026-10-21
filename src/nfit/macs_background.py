"""Monitor-normalized MACS backgrounds replayed at measured sample angles.

Every background acquisition remains one counting primitive when projected to
multiple angles. Sparse sensitivities retain its covariance across output bins.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from .dgs_binning import _symmetry_matrices
from .event_bin_indices import flat_bin_indices
from .macs import import_macs_nexus
from .macs_reduction import MACS_DEFAULTS, macs_group_config
from .mdhisto import MDHistoChannel, MDHistoData
from .measurement_dependencies import (
    SourceDependencies,
    SourceReplayRequired,
    independent_source_dependencies,
)
from .project_masks import _point_data_with_nfit_masks
from .reduction_recipes import effective_reduction_config
from .source_lineage import (
    SOURCE_LINEAGE_KEY,
    SOURCE_SELECTION_LINEAGE_KEY,
    merge_source_lineage_metadata,
    source_identity,
)

DEFAULT_ENERGY_TOLERANCE_MEV = 0.02

_MATCH_SETTINGS = (
    "stream", "monitor_target", "monitor_response", "monitor_reference_k_inv_angstrom",
    "ki_kf_normalization", "higher_order_monitor_correction", "apply_detector_efficiency",
    "incident_energy_policy", "final_energy_policy",
    "vanadium_files", "vanadium_a4_mask_deg",
)


def _target_symmetry(target):
    """Resolve both native-event and point-histogram symmetry provenance."""
    operations = target.metadata.get("symmetry_operations_hkl")
    if operations is None:
        operations = target.metadata.get("rebin", {}).get("symmetry", {}).get("operations_hkl")
    return _symmetry_matrices(operations)


def prepare_macs_background_sample(target, *, source_signature):
    """Track independent sample bins for a nonfractional MACS subtraction.

    Original sample primitives cannot span multiple bins under nearest physical
    assignment without symmetry. Fractional or symmetry-derived bins require
    their original sample sensitivities and are deliberately rejected here.
    """
    if target.source_dependencies is not None:
        return target
    reduction = target.metadata.get("measurement_reduction", {})
    if reduction.get("assignment") != "nearest_physical_bin":
        raise SourceReplayRequired("MACS subtraction requires original sample sensitivities for fractional or untracked binning")
    operations = _target_symmetry(target)
    if len(operations) != 1 or not np.array_equal(operations[0], np.eye(3)):
        raise SourceReplayRequired("MACS subtraction requires original sample sensitivities for symmetry-expanded bins")
    namespace = hashlib.sha256(str(source_signature).encode()).hexdigest()
    variance = np.where(np.isfinite(target.errors), np.square(target.errors), 0.)
    dependencies = independent_source_dependencies(variance, f"macs-sample-bins:{namespace}")
    return target.with_updates(source_dependencies=dependencies,
        metadata={**target.metadata, "sample_uncertainty": "independent_bins_from_nonfractional_unsymmetrized_counts"})


def _rotation(angles):
    radians = np.deg2rad(90.0 - np.asarray(angles))
    matrix = np.zeros((len(radians), 3, 3))
    matrix[:, 0, 0] = matrix[:, 1, 1] = np.cos(radians)
    matrix[:, 0, 1] = -np.sin(radians)
    matrix[:, 1, 0] = np.sin(radians)
    matrix[:, 2, 2] = 1.
    return matrix


def _source(group, run, masks):
    if run.transforms or run.backgrounds:
        raise ValueError("MACS replay requires untransformed, unsubtracted source runs")
    config = effective_reduction_config(group, run)
    path = Path(run.metadata["source_file"])
    data = import_macs_nexus(path, {key: value for key, value in config.items()
                                   if key in MACS_DEFAULTS or key == "stream"})
    data = _point_data_with_nfit_masks(run, data, extra_masks=list(masks))
    with h5py.File(path, "r") as handle:
        logs = handle["entry/DAS_logs"]
        node = logs.get("sampleTheta/primaryNode")
        if node is None:
            node = logs["sampleTheta/softPosition"]
        angles = np.asarray(node[()], float).reshape(-1)
    rows = data.size // 20
    if len(angles) == 1:
        angles = np.repeat(angles, rows)
    if len(angles) != rows or np.any(~np.isfinite(angles)):
        raise ValueError("MACS sample angles must be finite and match acquisition points")
    return config, data, angles + data.metadata["a3_offset_deg"], path


class _MacsSourcePreparation:
    """One compatibility reference shared by angular and radial projections."""

    def __init__(self, energy_tolerance_meV=DEFAULT_ENERGY_TOLERANCE_MEV):
        self.energy_tolerance_meV = float(energy_tolerance_meV)
        if not np.isfinite(self.energy_tolerance_meV) or self.energy_tolerance_meV < 0:
            raise ValueError("MACS energy_tolerance_meV must be finite and nonnegative")
        self.settings = None
        self.energies = None

    def __call__(self, group, run, shared, mapping):
        masks = (mapping or {}).get(run.id, shared if shared is not None else group.masks)
        config, data, angles, path = _source(group, run, masks)
        settings = tuple(config.get(k, MACS_DEFAULTS.get(k)) for k in _MATCH_SETTINGS)
        ei = float(data.metadata["incident_energy"])
        ef = float(data.metadata.get("fixed_final_energy_meV", ei))
        energies = np.array([ei, ef])
        if self.settings is None:
            self.settings, self.energies = settings, energies
        if settings != self.settings:
            raise ValueError("MACS sample/background detector stream and normalization/energy settings must match")
        if not np.allclose(energies, self.energies, rtol=0, atol=self.energy_tolerance_meV):
            raise ValueError(f"MACS replay requires matching incident and final energies within {self.energy_tolerance_meV:g} meV")
        return config, data, angles, path


def _macs_background_metadata(target, background, sources, records, projection):
    metadata = {k: v for k, v in target.metadata.items() if k not in {
        "nfit_mask", "file_mask", "background_subtractions", "event_statistics", "measurement_contract",
        "poisson_count_model", "cached_background_replay", "normalization_denominator",
        SOURCE_LINEAGE_KEY, SOURCE_SELECTION_LINEAGE_KEY, "source_file", "source_files",
        "source_group", "source_datasets", "scan_point_shape", "import_options",
        "incident_filter_states", "analyzer_mask_reasons", "masked_analyzer_channels"}}
    lineages = [{SOURCE_SELECTION_LINEAGE_KEY: {
        "version": 1, "source_identity": source_identity(
            record["file"], importer="macs_nexus", stream=record["stream"])}} for record in records]
    metadata.update(merge_source_lineage_metadata(*sources, *lineages))
    source_files = list(dict.fromkeys(record["file"] for record in records))
    if len(source_files) == 1:
        metadata["source_file"] = source_files[0]
    metadata.update(signal_semantics="monitor_normalized_counts", zero_event_bins_are_measured=True,
        source_files=source_files, source_group=background.name,
        source_datasets=[run.name for run in sources],
        background_projection={"instrument": "MACS", "source_group": background.id,
            "energy_tolerance_meV": DEFAULT_ENERGY_TOLERANCE_MEV, "sources": records, **projection})
    return metadata


def project_measured_background_macs(
    sample, background, target, *, datasets=None, background_datasets=None,
    inherited_masks=None, background_inherited_masks=None,
    inherited_masks_by_id=None, background_inherited_masks_by_id=None,
    max_batch_bytes=64 * 1024**2, progress_callback=None,
    energy_tolerance_meV=DEFAULT_ENERGY_TOLERANCE_MEV,
):
    """Pool MACS background counts and monitor exposure at every sample A3.

    Groups must have the same detector stream, normalization and energy policies,
    and one matching Ei/Ef condition (within ``energy_tolerance_meV``, default 0.02 meV). Background sample angles
    are undone before applying the sample's A3 offset and UB. Sample acquisition
    exposures weight its exact recorded angles; background acquisitions pool by
    effective monitor exposure. A3 and kidney grids need not coincide. Coverage
    is confined to measured background rays; unsupported cells remain masked.

    Both raw detector counts and calibrated monitor denominators come from the
    normal MACS importer. Repeated copies share primitive identities, including
    across bins, and therefore never create additional counting statistics.
    Source and sample masks use their usual point coordinates. This operation
    requires source files and rejects already transformed/subtracted sources.
    """
    if any(macs_group_config(group) is None for group in (sample, background)):
        raise ValueError("MACS replay requires homogeneous MACS source groups")
    if background.backgrounds:
        raise ValueError("MACS replay requires an unsubtracted background group")
    if len(target.axes) != 4 or any(len(a.values) != n + 1 for a, n in zip(target.axes, target.shape, strict=True)):
        raise ValueError("MACS replay requires a four-dimensional HKLE grid with physical edges")
    basis = np.asarray(target.metadata.get("rebin", {}).get("vectors", np.eye(4)), float)
    if basis.shape != (4, 4) or not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("MACS replay requires a finite invertible HKLE basis")
    if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1:
        raise ValueError("MACS replay cannot mix momentum and energy axes")
    inverse_basis = np.linalg.inv(basis)
    runs = list(sample.datasets if datasets is None else datasets)
    sources = list(background.datasets if background_datasets is None else background_datasets)
    if any(not np.isfinite(r.fit_weight) or r.fit_weight < 0 or not np.isfinite(r.scale_factor)
           for r in [*runs, *sources]):
        raise ValueError("MACS replay requires finite calibration and nonnegative exposure weights")
    runs = [r for r in runs if r.enabled and r.fit_weight > 0]
    sources = [r for r in sources if r.enabled and r.fit_weight > 0]
    if not runs or not sources:
        raise ValueError("MACS replay requires enabled sample and background runs")
    shape, cells = target.shape, target.signal.size
    budget = int(max_batch_bytes)
    if budget < cells * 64:
        raise SourceReplayRequired("MACS replay output exceeds its memory budget")
    prepare = _MacsSourcePreparation(energy_tolerance_meV)

    transforms, angle_weights, channel_acceptance = [], [], []
    sample_paths = set()
    symmetry = _target_symmetry(target)
    for run in runs:
        _, data, angles, path = prepare(sample, run, inherited_masks, inherited_masks_by_id)
        sample_paths.add(path.resolve())
        mask = data.mask.reshape(-1, 20)
        exposure = data.normalization_denominator.reshape(-1, 20)
        weights = np.mean(exposure, axis=1) * run.fit_weight
        inverse_rlu = np.linalg.inv(np.asarray(data.metadata["rlu_to_inv_angstrom_matrix"]))
        for rotation, weight, accepted in zip(_rotation(angles), weights, mask, strict=True):
            if not np.isfinite(weight) or weight <= 0 or not np.any(accepted):
                continue
            for operation in symmetry:
                transforms.append(rotation @ inverse_rlu.T @ operation.T @ inverse_basis[:3, :3])
                angle_weights.append(weight)
                channel_acceptance.append(accepted)
    if not transforms:
        raise ValueError("MACS sample has no unmasked acquisition angles")
    transforms = np.asarray(transforms)
    weights = np.asarray(angle_weights)
    weights /= weights.sum()
    acceptance = np.asarray(channel_acceptance)
    edges = [np.asarray(axis.values, float) for axis in target.axes]
    numerator, denominator, events = (np.zeros(cells) for _ in range(3))
    observations, primitives, coefficients = [], [], []
    source_ids, source_variances = [], []
    source_records = []
    seen = set()
    stored = 0
    for source_number, run in enumerate(sources):
        config, data, angles, path = prepare(background, run, background_inherited_masks, background_inherited_masks_by_id)
        identity = (str(path.resolve()), config.get("stream", "spec"))
        if path.resolve() in sample_paths:
            raise ValueError("MACS sample and background must be independent acquisitions")
        if identity in seen:
            raise ValueError("MACS replay background contains a duplicate acquisition")
        seen.add(identity)
        source_records.append({"file": str(path.resolve()), "dataset_id": run.id,
                               "name": run.name, "stream": identity[1],
                               "reduction_settings": config})
        counts = np.asarray(data.measurement_payload["event_signal_numerator"])
        exposure = data.normalization_denominator
        hkl = np.column_stack((data.H, data.K, data.L))
        q_sample = hkl @ np.asarray(data.metadata["rlu_to_inv_angstrom_matrix"]).T
        rotation = np.repeat(_rotation(angles), 20, axis=0)
        lab = np.einsum("ni,nji->nj", q_sample, rotation)
        namespace = hashlib.sha256(repr(identity).encode()).hexdigest()
        offset = len(source_ids)
        source_ids.extend(f"macs:{namespace}:{i}" for i in range(data.size))
        source_variances.extend(np.where(data.mask, counts, 0.))
        # Bound the transform workspace; concatenate only coalesced bin/primitive links.
        batch = max(1, min(data.size, budget // max(1, len(transforms) * 160)))
        for start in range(0, data.size, batch):
            stop = min(data.size, start + batch)
            ids = np.arange(start, stop)
            q = np.einsum("ni,aij->naj", lab[start:stop], transforms)
            energy = np.broadcast_to(data.E[start:stop, None, None], (*q.shape[:2], 1))
            coords = np.concatenate((q, energy), axis=2)
            bins = flat_bin_indices(coords.reshape(-1, 4), edges, shape).reshape(len(ids), -1)
            valid = data.mask[start:stop, None] & acceptance[:, ids % 20].T & (bins >= 0)
            valid &= ~target.mask.ravel()[np.maximum(bins, 0)]
            rows, columns = np.nonzero(valid)
            locations = bins[rows, columns]
            keys, inverse = np.unique(rows * cells + locations, return_inverse=True)
            combined = np.bincount(inverse, weights=weights[columns] * run.fit_weight)
            rows, locations = keys // cells, keys % cells
            raw_ids = ids[rows]
            np.add.at(numerator, locations, combined * counts[raw_ids] * run.scale_factor)
            np.add.at(denominator, locations, combined * exposure[raw_ids])
            np.add.at(events, locations, combined * counts[raw_ids])
            observations.append(locations)
            primitives.append(raw_ids + offset)
            coefficients.append(combined * run.scale_factor)
            stored += len(combined)
            if stored * 80 + len(source_ids) * 180 + cells * 64 > budget:
                raise SourceReplayRequired("MACS background sensitivities exceed the memory budget; use a coarser target or increase the budget")
            if progress_callback is not None:
                progress_callback({"stage": "macs_background_replay", "iteration": stop,
                                   "total": data.size, "message": f"Replaying MACS background {source_number + 1}/{len(sources)}"})
    obs = np.concatenate(observations)
    deps = SourceDependencies(shape=shape, source_ids=tuple(source_ids),
        source_variances=np.asarray(source_variances), observation_indices=obs,
        source_indices=np.concatenate(primitives),
        coefficients=np.concatenate(coefficients) / denominator[obs], max_bytes=budget)
    signal = np.divide(numerator, denominator, out=np.zeros(cells), where=denominator > 0)
    metadata = _macs_background_metadata(target, background, sources, source_records,
        {"mode": "measured_events", "energy_tolerance_meV": prepare.energy_tolerance_meV,
         "uncertainty": "shared_counting_primitives_across_angles_and_bins",
         "sample_angles": len(transforms)})
    return MDHistoData(axes=target.axes, signal=signal.reshape(shape), errors=np.sqrt(deps.variances()),
        mask=(denominator <= 0).reshape(shape), num_events=events.reshape(shape), metadata=metadata,
        source_dependencies=deps, auxiliary_channels={"normalization_denominator": MDHistoChannel(
            denominator.reshape(shape), label="Replayed effective monitor exposure")})
