"""Isotropic MACS backgrounds pooled in physical momentum magnitude."""

from __future__ import annotations

import hashlib

import numpy as np

from .macs_background import (
    DEFAULT_ENERGY_TOLERANCE_MEV,
    _macs_background_metadata,
    _MacsSourcePreparation,
)
from .macs_reduction import macs_group_config
from .mdhisto import MDHistoChannel, MDHistoData
from .measurement_dependencies import (
    SourceDependencies,
    SourceReplayRequired,
    project_source_dependencies,
)

DEFAULT_Q_BIN_WIDTH = 0.05


def _radial_assignments(q, centers, measured, width, interpolation):
    """Return covered output/radial-bin assignments without bridging gaps."""
    q = np.asarray(q)
    if interpolation == "nearest":
        bins = np.floor(q / width).astype(np.int64)
        valid = (bins >= 0) & (bins < len(centers))
        valid &= measured[np.clip(bins, 0, len(centers) - 1)]
        output = np.flatnonzero(valid)
        return output, bins[valid], np.ones(len(output))
    location = q / width - .5
    lower = np.floor(location).astype(np.int64)
    fraction = location - lower
    exact = np.isclose(fraction, 0., rtol=0, atol=1e-12)
    upper = lower + (~exact).astype(np.int64)
    valid = (lower >= 0) & (upper < len(centers))
    valid &= measured[np.clip(lower, 0, len(centers) - 1)]
    valid &= measured[np.clip(upper, 0, len(centers) - 1)]
    output = np.flatnonzero(valid)
    twice = output[~exact[output]]
    return (np.r_[output, twice], np.r_[lower[output], upper[twice]],
            np.r_[np.where(exact[output], 1., 1. - fraction[output]), fraction[twice]])


def project_isotropic_background_macs(
    sample, background, target, *, q_bin_width=DEFAULT_Q_BIN_WIDTH,
    interpolation="nearest", datasets=None, background_datasets=None,
    inherited_masks=None, background_inherited_masks=None,
    inherited_masks_by_id=None, background_inherited_masks_by_id=None,
    max_batch_bytes=64 * 1024**2, progress_callback=None,
    energy_tolerance_meV=DEFAULT_ENERGY_TOLERANCE_MEV,
    allow_background_reduction_overrides=False,
):
    """Project an isotropic, monitor-normalized MACS background at voxel centers.

    ``q_bin_width`` is the width in inverse angstroms of radial bins anchored at
    zero. Raw background counts and effective monitor denominators pool within
    each bin, using the normal MACS importer and per-run calibration/fit weights.
    The resulting field depends only on physical |Q|, not A3 or detector direction.
    ``nearest`` uses the containing radial bin; ``linear`` interpolates between
    adjacent measured bin centers. Neither fills unmeasured radial bins or goes
    beyond the acquired |Q| range. Missing energy and sample coverage stay masked.

    One matching Ei/Ef condition and the same detector/normalization policies
    are required, as for measured-angle replay. Reused radial values retain the
    original background counting primitives and their cross-bin covariance.
    ``allow_background_reduction_overrides`` permits independently declared
    source corrections as for measured-angle replay, expressing backgrounds in
    the sample's monitor-target and reference-wavevector units. Stream and
    resolved Ei/Ef must match.
    """
    width = float(q_bin_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("MACS isotropic q_bin_width must be finite and positive in inverse angstroms")
    if interpolation not in {"nearest", "linear"}:
        raise ValueError("MACS isotropic interpolation must be nearest or linear")
    if any(macs_group_config(group) is None for group in (sample, background)):
        raise ValueError("MACS isotropic projection requires homogeneous MACS source groups")
    if background.backgrounds:
        raise ValueError("MACS isotropic projection requires an unsubtracted background group")
    if len(target.axes) != 4 or any(len(axis.values) != size + 1 for axis, size in zip(target.axes, target.shape, strict=True)):
        raise ValueError("MACS isotropic projection requires four-dimensional HKLE physical edges")
    basis = np.asarray(target.metadata.get("rebin", {}).get("vectors", np.eye(4)), float)
    if basis.shape != (4, 4) or not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("MACS isotropic projection requires a finite invertible HKLE basis")
    if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1:
        raise ValueError("MACS isotropic projection cannot mix momentum and energy axes")
    runs = list(sample.datasets if datasets is None else datasets)
    sources = list(background.datasets if background_datasets is None else background_datasets)
    if any(not np.isfinite(run.fit_weight) or run.fit_weight < 0 or not np.isfinite(run.scale_factor)
           for run in [*runs, *sources]):
        raise ValueError("MACS isotropic projection requires finite calibration and nonnegative exposure weights")
    runs = [run for run in runs if run.enabled and run.fit_weight > 0]
    sources = [run for run in sources if run.enabled and run.fit_weight > 0]
    if not runs or not sources:
        raise ValueError("MACS isotropic projection requires enabled sample and background runs")
    budget = int(max_batch_bytes)
    if target.signal.size * 128 > budget:
        raise SourceReplayRequired("MACS isotropic output exceeds its memory budget")
    prepare = _MacsSourcePreparation(energy_tolerance_meV)
    rlu = None
    sample_paths = set()
    for run in runs:
        _, data, _, path = prepare(sample, run, inherited_masks, inherited_masks_by_id)
        current = np.asarray(data.metadata["rlu_to_inv_angstrom_matrix"])
        if rlu is not None and not np.allclose(current, rlu, rtol=1e-10, atol=1e-12):
            raise ValueError("MACS isotropic sample runs must share the same reciprocal basis")
        rlu = current
        sample_paths.add(path.resolve())
    source_ids, source_variances = [], []
    q_parts, count_parts, exposure_parts, coefficient_parts = [], [], [], []
    records, energy_ranges, seen = [], [], set()
    for index, run in enumerate(sources):
        config, data, _, path = prepare(background, run, background_inherited_masks,
            background_inherited_masks_by_id, allow_reduction_overrides=allow_background_reduction_overrides)
        identity = (str(path.resolve()), config.get("stream", "spec"))
        if path.resolve() in sample_paths:
            raise ValueError("MACS sample and background must be independent acquisitions")
        if identity in seen:
            raise ValueError("MACS isotropic background contains a duplicate acquisition")
        seen.add(identity)
        records.append({"file": str(path.resolve()), "dataset_id": run.id, "name": run.name,
                        "stream": identity[1], "reduction_settings": config})
        hkl = np.column_stack((data.H, data.K, data.L))
        physical = hkl @ np.asarray(data.metadata["rlu_to_inv_angstrom_matrix"]).T
        q = np.linalg.norm(physical, axis=1)
        valid = data.mask & np.isfinite(q) & np.isfinite(data.E)
        valid &= np.isfinite(data.normalization_denominator) & (data.normalization_denominator > 0)
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        counts = np.asarray(data.measurement_payload["event_signal_numerator"])[valid]
        namespace = hashlib.sha256(repr(identity).encode()).hexdigest()
        source_ids.extend(f"macs:{namespace}:{i}" for i in ids)
        source_variances.extend(counts)
        q_parts.append(q[valid])
        count_parts.append(counts * run.fit_weight * run.scale_factor)
        exposure_parts.append(data.normalization_denominator[valid] * run.fit_weight)
        coefficient_parts.append(np.full(len(ids), run.fit_weight * run.scale_factor))
        energy_ranges.append((float(np.min(data.E[valid])), float(np.max(data.E[valid]))))
        if len(source_ids) * 256 > budget:
            raise SourceReplayRequired("MACS isotropic source primitives exceed the memory budget")
        if progress_callback is not None:
            progress_callback({"stage": "macs_isotropic_background", "iteration": index + 1,
                               "total": len(sources), "message": "Pooling MACS radial counts and monitor exposure"})
    if not q_parts:
        raise ValueError("MACS isotropic background has no valid measured points")
    q = np.concatenate(q_parts)
    if np.max(q) / width + 1 > budget // 128:
        raise SourceReplayRequired("MACS radial grid exceeds its memory budget; increase q_bin_width")
    bins = np.floor(q / width).astype(np.int64)
    size = int(np.max(bins)) + 1
    if size * 128 > budget:
        raise SourceReplayRequired("MACS radial grid exceeds its memory budget; increase q_bin_width")
    numerator = np.bincount(bins, weights=np.concatenate(count_parts), minlength=size)
    exposure = np.bincount(bins, weights=np.concatenate(exposure_parts), minlength=size)
    measured = exposure > 0
    radial_signal = np.divide(numerator, exposure, out=np.zeros(size), where=measured)
    centers = (np.arange(size) + .5) * width
    radial_dependencies = SourceDependencies(shape=(size,), source_ids=tuple(source_ids),
        source_variances=np.asarray(source_variances), observation_indices=bins,
        source_indices=np.arange(len(source_ids)),
        coefficients=np.concatenate(coefficient_parts) / exposure[bins], max_bytes=budget)
    shape = target.shape
    coords = np.column_stack([grid.ravel() for grid in np.meshgrid(
        *(axis.centers for axis in target.axes), indexing="ij")])
    physical = (coords @ basis)[:, :3] @ rlu.T
    target_q = np.linalg.norm(physical, axis=1)
    output, radial, weights = _radial_assignments(target_q, centers, measured, width, interpolation)
    covered = ~target.mask.ravel()
    covered &= (target_q >= np.min(q)) & (target_q <= np.max(q))
    energy_covered = np.zeros(shape[-1], bool)
    energy_edges = np.asarray(target.axes[-1].values)
    for low, high in energy_ranges:
        first = int(np.searchsorted(energy_edges, low, side="right")) - 1
        last = int(np.searchsorted(energy_edges, high, side="right")) - 1
        if low == energy_edges[-1]:
            first = len(energy_covered) - 1
        if high == energy_edges[-1]:
            last = len(energy_covered) - 1
        energy_covered[max(0, first):min(len(energy_covered), last + 1)] = True
    covered &= np.broadcast_to(energy_covered, shape).ravel()
    selected = covered[output]
    output, radial, weights = output[selected], radial[selected], weights[selected]
    dependencies = project_source_dependencies(radial_dependencies, radial, output, weights, shape, max_bytes=budget)
    cells = target.signal.size
    signal = np.bincount(output, weights=weights * radial_signal[radial], minlength=cells)
    denominator = np.bincount(output, weights=weights * exposure[radial], minlength=cells)
    raw_counts = np.bincount(bins, weights=np.asarray(source_variances), minlength=size)
    events = np.bincount(output, weights=weights * raw_counts[radial], minlength=cells)
    metadata = _macs_background_metadata(target, background, sources, records,
        {"mode": "macs_isotropic", "energy_tolerance_meV": prepare.energy_tolerance_meV,
         "q_bin_width": width, "q_unit": "angstrom^-1",
         "background_reduction_overrides": bool(allow_background_reduction_overrides),
         "monitor_target": prepare.monitor_target,
         "monitor_reference_k_inv_angstrom": prepare.monitor_reference_k_inv_angstrom,
         "interpolation": interpolation, "q_range": [float(np.min(q)), float(np.max(q))],
         "uncertainty": "shared_counting_primitives_across_radial_field_bins",
         "angular_assumption": "isotropic in physical momentum magnitude",
         "radial_profile": {"centers": centers.tolist(), "signal": radial_signal.tolist(),
                            "exposure": exposure.tolist(), "measured": measured.tolist()}})
    metadata["num_events_semantics"] = "radial_field_count_proxy_not_independent_counts"
    return MDHistoData(axes=target.axes, signal=signal.reshape(shape), errors=np.sqrt(dependencies.variances()),
        mask=(denominator <= 0).reshape(shape), num_events=events.reshape(shape), metadata=metadata,
        source_dependencies=dependencies, auxiliary_channels={"normalization_denominator": MDHistoChannel(
            denominator.reshape(shape), label="Interpolated radial effective monitor exposure")})
