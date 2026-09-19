"""Replay measured lab-frame backgrounds at the sample's goniometer angles.

Unlike a powder projection, this retains detector-direction dependence. Event
copies are correlated observations, not additional counting statistics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from . import _parallel, mdevent
from .event_masks import reduce_masked_event_runs
from .mdhisto import MDHistoChannel, MDHistoData
from .pipeline import DatasetEntry, DatasetGroup, MaskSpec
from .project_masks import _mdhisto_with_nfit_masks

try:
    from . import _mdevent_background_numba as _REPLAY_NUMBA
    if _REPLAY_NUMBA.config.DISABLE_JIT:
        _REPLAY_NUMBA = None
except ImportError:
    _REPLAY_NUMBA = None

# Bound work as well as memory, so cancellation is checked between short calls.
MAX_REPLAY_TRANSFORM_TASKS = 1_000_000


def project_measured_background_mdevent(
    sample: DatasetGroup,
    background: DatasetGroup,
    target: MDHistoData,
    *,
    datasets: Iterable[DatasetEntry] | None = None,
    background_datasets: Iterable[DatasetEntry] | None = None,
    inherited_masks: Iterable[MaskSpec] | None = None,
    background_inherited_masks: Iterable[MaskSpec] | None = None,
    max_batch_bytes: int = 64 * 1024 * 1024,
    progress_callback=None,
) -> MDHistoData:
    """Bin a measured background as if acquired at every selected sample angle.

    Both groups must be QSample or QLab MDEvent data with matching detector geometry
    and incident energy (meV). Sample proton charges and fit weights determine
    the relative exposure at each angle. Background calibration scales multiply
    counts and uncertainties; its fit weights also weight the normalization.
    No spherical averaging or interpolation is performed. The target's axes,
    basis, symmetry and detector-trajectory integration are reused.

    Masks apply at reconstructed output-bin centers, like native MDEvent
    reduction. Repeated copies of each background event are combined before
    squaring their weights for the variance. Cross-bin covariance is not
    stored. The source's private powder binning does not enter this mode.
    """
    if any("mdevent" not in group.metadata for group in (sample, background)):
        raise ValueError("measured-event replay requires sample and background MDEvent groups")
    if len(target.axes) != 4 or target.signal.ndim != 4:
        raise ValueError("measured-event replay requires a four-dimensional HKLE target")
    if background.backgrounds:
        raise ValueError("measured-event replay requires an unsubtracted background source group")
    for group in (sample, background):
        dimensions = group.metadata["mdevent"].get("dimensions", [])
        frames = {axis.get("frame") for axis in dimensions[:3]}
        if len(dimensions) != 4 or frames not in ({"QSample"}, {"QLab"}):
            raise ValueError("measured-event replay requires QSample or QLab source coordinates")
    runs = list(sample.datasets if datasets is None else datasets)
    sources = list(background.datasets if background_datasets is None else background_datasets)
    for run in [*runs, *sources]:
        if (
            not np.isfinite(run.scale_factor)
            or not np.isfinite(run.fit_weight)
            or run.fit_weight < 0
        ):
            raise ValueError(
                "replay calibration scales must be finite and fit weights finite and nonnegative"
            )
    runs = [run for run in runs if run.enabled and run.fit_weight > 0]
    sources = [run for run in sources if run.enabled and run.fit_weight > 0]
    if not runs or not sources:
        raise ValueError(
            "measured-event replay requires enabled sample and background runs with positive weights"
        )
    if any(run.backgrounds for run in sources):
        raise ValueError("measured-event replay requires unsubtracted background runs")
    basis = np.asarray(target.metadata.get("rebin", {}).get("vectors", np.eye(4)), dtype=float)
    if basis.shape != (4, 4) or not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("measured-event replay requires a finite invertible HKLE basis")
    if np.any(basis[:3, 3]) or np.any(basis[3, :3]) or basis[3, 3] != 1:
        raise ValueError("measured-event replay cannot mix momentum and energy axes")
    inverse_basis = np.linalg.inv(basis)
    mdevent._validate_mdevent_memory(target.shape, max_batch_bytes=max_batch_bytes)
    symmetry = target.metadata.get("symmetry_operations_hkl")
    sample_masks = list(sample.masks if inherited_masks is None else inherited_masks)
    source_masks = list(
        background.masks if background_inherited_masks is None else background_inherited_masks
    )
    exposure = np.asarray([float(run.metadata["proton_charge"]) * run.fit_weight for run in runs])
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("sample exposures must be finite and positive for measured-event replay")
    exposure /= exposure.sum()
    prepared = []
    for run, fraction in zip(runs, exposure, strict=True):
        detectors, payloads = mdevent._trajectory_payloads(
            sample,
            [run],
            inverse_basis,
            symmetry,
            progress_callback=progress_callback,
        )
        masks = [mask for mask in [*sample_masks, *run.masks] if mask.enabled]
        excluded = None
        if masks:
            masked = _mdhisto_with_nfit_masks(
                DatasetEntry("Replay acceptance", None, masks=masks), data=target
            )
            excluded = np.asarray(masked.metadata["nfit_mask"]).ravel()
        prepared.append((detectors, payloads, float(fraction), excluded))
    return reduce_masked_event_runs(
        sources,
        source_masks,
        lambda subset: _replay_runs(
            background,
            subset,
            target,
            prepared,
            max_batch_bytes=max_batch_bytes,
            progress_callback=progress_callback,
        ),
        zero_count_upper=mdevent.FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER,
    )


def _matching_directions(theta, phi, sample_theta, sample_phi):
    """Accept small calibration shifts, not a different detector-bank setting."""

    def directions(polar, azimuth):
        return np.column_stack(
            (
                np.sin(polar) * np.cos(azimuth),
                np.sin(polar) * np.sin(azimuth),
                np.cos(polar),
            )
        )

    # Comparing directions avoids singular azimuth differences near the beam.
    distance = np.linalg.norm(directions(theta, phi) - directions(sample_theta, sample_phi), axis=1)
    return bool(np.all(distance <= 2 * np.sin(np.deg2rad(0.1) / 2)))


def _replay_runs(background, sources, target, prepared, *, max_batch_bytes, progress_callback):
    import h5py

    shape = target.shape
    edges = [np.asarray(axis.values, dtype=float) for axis in target.axes]
    size = int(np.prod(shape))
    numerator, variance, events, denominator = (np.zeros(size) for _ in range(4))
    config = background.metadata["mdevent"]
    for run_index, source in enumerate(sources):
        source_detectors, source_payloads = mdevent._trajectory_payloads(
            background,
            [source],
            np.eye(4),
            progress_callback=progress_callback,
        )
        _, ei, bounds, charge, geometry = source_payloads[0]
        if not np.isfinite(charge) or charge <= 0:
            raise ValueError("background proton charge must be finite and positive")
        ids, theta, phi, solid = source_detectors[geometry]
        transforms, fractions, accepted_ids, excluded_bins = [], [], [], []
        for sample_detectors, payloads, fraction, excluded in prepared:
            replay_payloads = []
            replay_detectors = []
            for inverse, sample_ei, sample_bounds, _, index in payloads:
                sample_ids, sample_theta, sample_phi, sample_solid = sample_detectors[index]
                if (
                    not np.isclose(ei, sample_ei, rtol=1e-3, atol=1e-8)
                    or not np.array_equal(ids, sample_ids)
                    or not _matching_directions(theta, phi, sample_theta, sample_phi)
                ):
                    raise ValueError(
                        "measured-event replay requires matching incident energy and detector geometry; select the matching background bank setting"
                    )
                # Restrict exposure to the energy domain common to source and
                # sample. Events outside it must be excluded as well.
                common = np.array(
                    [max(bounds[0], sample_bounds[0]), min(bounds[-1], sample_bounds[-1])]
                )
                if common[1] <= common[0]:
                    raise ValueError("sample and background have no common energy coverage")
                valid_solid = np.where(sample_solid > 0, solid, 0.0)
                replay_detectors.append((ids, theta, phi, valid_solid))
                replay_payloads.append(
                    (inverse, ei, common, charge * fraction, len(replay_detectors) - 1)
                )
                transforms.append((inverse, common))
                fractions.append(fraction)
                accepted_ids.append(ids[valid_solid > 0])
                excluded_bins.append(excluded)
            norm = mdevent._trajectory_normalization_from_payloads(
                replay_detectors,
                replay_payloads,
                edges,
                shape,
                progress_callback=progress_callback,
            ).ravel()
            if excluded is not None:
                norm[excluded] = 0
            denominator += norm
        transform_count = len(transforms)
        accelerated = _REPLAY_NUMBA is not None
        backend = "numba" if accelerated else "numpy"
        workers = _REPLAY_NUMBA.effective_workers(_parallel.num_threads()) if accelerated else 1
        weights = np.asarray(fractions) * source.fit_weight * source.scale_factor
        if accelerated:
            inverses = np.ascontiguousarray([inverse for inverse, _ in transforms])
            energy_bounds = np.ascontiguousarray([common for _, common in transforms])
            sorted_ids = np.sort(ids)
            acceptance_cache = {}
            acceptance = []
            for valid_ids in accepted_ids:
                key = valid_ids.tobytes()
                if key not in acceptance_cache:
                    acceptance_cache[key] = np.isin(sorted_ids, valid_ids)
                acceptance.append(acceptance_cache[key])
            acceptance, exclusions = _REPLAY_NUMBA.prepare_replay_flags(
                acceptance,
                [np.empty(0, dtype=bool) if mask is None else mask for mask in excluded_bins],
                output_size=size,
            )
            bytes_per_row = 128 + _REPLAY_NUMBA.REPLAY_SCRATCH_BYTES_PER_TASK * transform_count
        else:
            bytes_per_row = 128 * transform_count + 128
        # Never materialize the synthetic event collection, or allocate a full
        # four-dimensional output grid per CPU worker.
        rows = max(1, min(
            100_000,
            int(max_batch_bytes) // max(bytes_per_row, 1),
            MAX_REPLAY_TRANSFORM_TASKS // max(transform_count, 1),
        ))
        workers = min(workers, rows)
        with h5py.File(source.metadata["source_file"], "r") as handle:
            workspace = handle[config["workspace_path"]]
            index = int(source.metadata["mdevent_experiment_index"])
            gonio = mdevent._read_goniometer_matrix(workspace[f"experiment{index}"])
            values = workspace["event_data/event_data"]
            workers = min(workers, max(1, values.shape[0]))

            def report(
                completed, *, workers=workers, accelerated=accelerated,
                total=values.shape[0], backend=backend, run_index=run_index,
            ):
                if progress_callback is not None:
                    mode = f"compiled parallel replay, {workers} CPUs" if accelerated else "NumPy fallback, 1 CPU"
                    progress_callback({
                        "stage": "mdevent_background_replay",
                        "iteration": completed,
                        "total": total,
                        "backend": backend,
                        "workers": workers,
                        "message": (
                            f"replaying background run {run_index + 1}/{len(sources)} "
                            f"at {len(prepared)} sample angles ({mode}): "
                            f"{completed:,}/{total:,} events"
                        ),
                    })

            report(0)
            for start in range(0, values.shape[0], rows):
                batch_workers = workers
                block = np.asarray(values[start : start + rows], dtype=float)
                block = block[block[:, 2].astype(np.int64) == index]
                if block.size:
                    if np.any(block[:, 3] != 0):
                        raise ValueError(
                            "measured-event replay currently requires one goniometer matrix per experiment"
                        )
                    lab = block[:, 5:8]
                    if config["dimensions"][0]["frame"] == "QSample":
                        lab = lab @ gonio.T
                    if accelerated:
                        detector_ids = block[:, 4].astype(np.int64)
                        detector_indices = np.searchsorted(sorted_ids, detector_ids)
                        known = detector_indices < len(sorted_ids)
                        if len(sorted_ids):
                            known &= sorted_ids[np.minimum(detector_indices, len(sorted_ids) - 1)] == detector_ids
                        detector_indices[~known] = -1
                        batch_workers = _REPLAY_NUMBA.accumulate_replayed_events(
                            np.ascontiguousarray(lab), np.ascontiguousarray(block[:, 8]),
                            detector_indices, np.ascontiguousarray(block[:, 0]),
                            np.ascontiguousarray(block[:, 1]), inverses, energy_bounds,
                            weights, acceptance, exclusions, tuple(edges), np.asarray(shape, dtype=np.int64),
                            numerator, variance, events, workers=workers,
                        )
                    else:
                        _replay_numpy_block(
                            block, lab, transforms, accepted_ids, excluded_bins,
                            edges, shape, weights, numerator, variance, events,
                        )
                report(min(start + rows, values.shape[0]), workers=batch_workers)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = numerator / denominator
        errors = np.sqrt(variance) / denominator
    rms = np.sqrt(variance.sum() / events.sum()) if events.sum() else 1.0
    zero = (events == 0) & (denominator > 0)
    errors[zero] = mdevent.FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER * rms / denominator[zero]
    valid = (denominator > 0) & np.isfinite(signal) & np.isfinite(errors)
    # Do not inherit sample masks as background-source mask provenance.
    metadata = {
        key: value
        for key, value in target.metadata.items()
        if key not in {"nfit_mask", "file_mask", "background_subtractions"}
    }
    metadata.update(
        normalization_denominator=denominator.reshape(shape),
        zero_event_bins_are_measured=True,
        event_weight_rms=float(rms),
        background_projection={
            "mode": "measured_events",
            "source_group": background.id,
            "uncertainty": "correlated_event_copies_per_output_bin",
        },
    )
    return MDHistoData(
        axes=target.axes,
        signal=signal.reshape(shape),
        errors=errors.reshape(shape),
        mask=(~valid).reshape(shape),
        num_events=events.reshape(shape),
        metadata=metadata,
        auxiliary_channels={
            "normalization_denominator": MDHistoChannel(
                denominator.reshape(shape), label="Replayed detector-trajectory normalization"
            )
        },
    )


def _replay_numpy_block(
    block, lab, transforms, accepted_ids, excluded_bins,
    edges, shape, weights, numerator, variance, events,
):
    """Reference implementation used when the optional compiled backend is absent."""
    flat = np.empty((block.shape[0], len(transforms)), dtype=np.int64)
    for column, ((inverse, common), valid_ids, excluded) in enumerate(
        zip(transforms, accepted_ids, excluded_bins, strict=True)
    ):
        coords = np.column_stack((lab @ inverse.T, block[:, 8]))
        locations = mdevent._flat_bin_indices(coords, edges, shape)
        valid = (
            np.isin(block[:, 4].astype(np.int64), valid_ids)
            & (block[:, 8] >= common[0])
            & (block[:, 8] <= common[1])
        )
        if excluded is not None:
            valid &= ~excluded[np.maximum(locations, 0)]
        flat[:, column] = np.where(valid, locations, -1)
    _accumulate_correlated_events(
        flat, weights, block[:, 0], block[:, 1], numerator, variance, events,
    )


def _accumulate_correlated_events(
    flat, weights, signal, variance, numerator, output_variance, events
):
    """Merge copies landing in the same bin before computing their variance."""
    order = np.argsort(flat, axis=1, kind="stable")
    locations = np.take_along_axis(flat, order, axis=1)
    sorted_weights = weights[order]
    first = np.ones(locations.shape, dtype=bool)
    first[:, 1:] = locations[:, 1:] != locations[:, :-1]
    starts = np.flatnonzero(first.ravel())
    combined = np.add.reduceat(sorted_weights.ravel(), starts)
    bins = locations.ravel()[starts]
    source_rows = starts // flat.shape[1]
    valid = bins >= 0
    bins, combined, source_rows = bins[valid], combined[valid], source_rows[valid]
    np.add.at(numerator, bins, signal[source_rows] * combined)
    np.add.at(output_variance, bins, variance[source_rows] * combined**2)
    np.add.at(events, bins, 1.0)
