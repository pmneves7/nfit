from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .analysis.coordinates import q_modulus_for_spectral
from .background_channels import accumulate_background_channel
from .mdhisto import MDHistoChannel, MDHistoData, mdhisto_coverage_fraction, mdhisto_measured_bins
from .measurement_dependencies import (
    SourceReplayRequired,
    combine_source_dependencies,
    project_source_dependencies,
)
from .source_lineage import merge_source_lineage_metadata, validate_independent_source_lineage


def background_with_user_mask_zeros(background: MDHistoData) -> MDHistoData:
    """Make explicitly excluded background bins contribute zero, not missing data.

    Native detector gaps and unmeasured bins remain unavailable unless the user
    explicitly excluded them. The source histogram itself stays immutable.
    """
    excluded = background.metadata.get("nfit_mask")
    if excluded is None:
        return background
    excluded = np.asarray(excluded, dtype=bool)
    if excluded.shape != background.shape:
        raise ValueError("background user mask must match its histogram shape")
    if not np.any(excluded):
        return background
    metadata = dict(background.metadata)
    metadata.pop("nfit_mask", None)
    metadata["background_excluded_bins"] = int(np.count_nonzero(excluded))
    metadata["zero_event_bins_are_measured"] = True
    denominator = metadata.get("normalization_denominator")
    if isinstance(denominator, np.ndarray) and denominator.shape == background.shape:
        metadata["normalization_denominator"] = np.where(excluded, 1.0, denominator)
    dependencies = background.source_dependencies
    if dependencies is not None:
        included = np.flatnonzero(~excluded.ravel())
        dependencies = project_source_dependencies(dependencies, included, included, np.ones(len(included)), background.shape)
    from .cached_background_replay import exclude_background_replay_cells

    metadata = exclude_background_replay_cells(background, excluded, metadata)
    return background.with_updates(
        signal=np.where(excluded, 0.0, background.signal),
        errors=np.where(excluded, 0.0, background.errors),
        mask=(background.mask | ~mdhisto_measured_bins(background)) & ~excluded,
        num_events=np.where(excluded, 0.0, background.num_events),
        metadata=metadata,
        source_dependencies=dependencies,
        counting_dependencies=None,
    )


def subtract_aligned_background(
    data: MDHistoData,
    background: MDHistoData,
    *,
    scale: float = 1.0,
) -> MDHistoData:
    """Subtract aligned histograms, preserving declared shared sensitivities.

    Untracked operands retain the documented independent variance convention.
    A mixture of tracked and untracked operands requires explicit source replay.
    """

    if float(scale) == 0.0:
        return data
    background = background_with_user_mask_zeros(background)
    if data.shape != background.shape or len(data.axes) != len(background.axes):
        raise ValueError("aligned background subtraction requires identical shapes")
    for data_axis, background_axis in zip(data.axes, background.axes, strict=True):
        if (
            data_axis.name != background_axis.name
            or data_axis.units != background_axis.units
            or data_axis.values.shape != background_axis.values.shape
            or not np.allclose(
                data_axis.values,
                background_axis.values,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "aligned background subtraction requires identical axis names, units, and bins"
            )
    measured = (
        mdhisto_measured_bins(data)
        & mdhisto_measured_bins(background)
        & np.isfinite(data.signal)
        & np.isfinite(data.errors)
        & np.isfinite(background.signal)
        & np.isfinite(background.errors)
    )
    factor = float(scale)
    signal = np.asarray(data.signal, dtype=float) - factor * np.asarray(
        background.signal, dtype=float
    )
    variance = np.square(np.asarray(data.errors, dtype=float)) + factor**2 * np.square(
        np.asarray(background.errors, dtype=float)
    )
    dependencies = None
    if data.source_dependencies is not None or background.source_dependencies is not None:
        dependencies = combine_source_dependencies((data.source_dependencies, background.source_dependencies), (1.0, -factor))
        variance = dependencies.variance()
    if dependencies is None:
        from .cached_background_replay import replay_represents_background_overlap

        if not replay_represents_background_overlap(data, background):
            validate_independent_source_lineage(data, background)
    metadata = dict(data.metadata)
    metadata.update(merge_source_lineage_metadata(data, background))
    history = list(metadata.get("background_subtractions", []))
    projection = background.metadata.get("background_projection")
    history.append(
        {
            "scale": factor,
            "interpolation": "aligned",
            "source": background.metadata.get("source_file"),
            **(
                {"projection": dict(projection)}
                if isinstance(projection, dict)
                else {}
            ),
        }
    )
    metadata["background_subtractions"] = history
    metadata["background_uncertainty_dependence"] = "tracked_sources" if dependencies is not None else "assumed_independent"
    metadata["num_events_semantics"] = "sample_event_contributions"
    metadata, channels = _background_derived_payload(data, background, metadata, measured)
    result = replace(
        data,
        signal=np.where(measured, signal, np.nan),
        errors=np.where(measured, np.sqrt(np.maximum(variance, 0.0)), np.nan),
        mask=np.asarray(data.mask, dtype=bool) | ~measured,
        num_events=data.num_events,
        metadata=metadata,
        auxiliary_channels=channels,
        source_dependencies=dependencies,
        counting_dependencies=None,
    )
    contribution = np.asarray(background.signal, dtype=float)
    contribution_errors = np.asarray(background.errors, dtype=float)
    if factor != 1.0:
        contribution = _freeze_owned(factor * contribution)
        contribution_errors = _freeze_owned(abs(factor) * contribution_errors)
    result = accumulate_background_channel(
        data,
        result,
        contribution=contribution,
        contribution_errors=contribution_errors,
        valid=measured,
    )
    from .cached_background_replay import background_replay_subtraction

    return background_replay_subtraction(data, background, result, factor)


def subtract_background(
    data: MDHistoData,
    background: MDHistoData,
    *,
    scale: float = 1.0,
    interpolation: str = "linear",
) -> MDHistoData:
    """Subtract either a powder background or an exactly aligned histogram."""

    try:
        _powder_dimensions(background)
    except ValueError:
        return subtract_aligned_background(data, background, scale=scale)
    return subtract_powder_background(
        data,
        background,
        scale=scale,
        interpolation=interpolation,
    )


def subtract_powder_background(
    data: MDHistoData,
    background: MDHistoData,
    *,
    scale: float = 1.0,
    interpolation: str = "linear",
) -> MDHistoData:
    """Interpolate a powder ``|Q|, E`` background and subtract it from data.

    Input datasets are assumed statistically independent, so the scaled
    background variance is added to the data variance. Target bins outside the
    background domain are masked rather than extrapolated.
    """

    if float(scale) == 0.0:
        return data
    if (data.source_dependencies is None) != (background.source_dependencies is None):
        raise SourceReplayRequired("Background interpolation requires tracked dependencies for both operands or source replay")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("background interpolation must be 'linear' or 'nearest'")
    background = background_with_user_mask_zeros(background)
    q_dim, energy_dim = _powder_dimensions(background)
    q_centers = np.asarray(background.axes[q_dim].centers, dtype=float)
    energy_centers = np.asarray(background.axes[energy_dim].centers, dtype=float)
    if q_centers.size < 1 or energy_centers.size < 1:
        raise ValueError("powder background axes must not be empty")
    values = np.moveaxis(background.signal, (q_dim, energy_dim), (0, 1))
    errors = np.moveaxis(background.errors, (q_dim, energy_dim), (0, 1))
    measured = np.moveaxis(mdhisto_measured_bins(background), (q_dim, energy_dim), (0, 1))
    if values.ndim != 2:
        raise ValueError("powder background must have exactly |Q| and energy dimensions")
    safe_values = np.where(measured, values, np.nan)
    safe_variance = np.where(measured, np.square(errors), np.nan)
    value_interpolator = RegularGridInterpolator(
        (q_centers, energy_centers),
        safe_values,
        method=interpolation,
        bounds_error=False,
        fill_value=np.nan,
    )
    variance_interpolator = None
    if interpolation == "nearest":
        variance_interpolator = RegularGridInterpolator(
            (q_centers, energy_centers),
            safe_variance,
            method=interpolation,
            bounds_error=False,
            fill_value=np.nan,
        )
    q = np.broadcast_to(q_modulus_for_spectral(data), data.shape)
    target_energy_dim = _energy_dimension(data)
    energy_shape = [1] * data.signal.ndim
    energy_shape[target_energy_dim] = data.shape[target_energy_dim]
    energy = np.broadcast_to(
        data.axes[target_energy_dim].centers.reshape(energy_shape), data.shape
    )
    points = np.column_stack((q.ravel(), energy.ravel()))
    interpolated = value_interpolator(points).reshape(data.shape)
    if variance_interpolator is None:
        interpolated_variance = _linear_interpolation_variance(
            q_centers,
            energy_centers,
            safe_variance,
            points,
        ).reshape(data.shape)
    else:
        interpolated_variance = variance_interpolator(points).reshape(data.shape)
    valid_background = np.isfinite(interpolated) & np.isfinite(interpolated_variance)
    factor = float(scale)
    dependencies = None
    if data.source_dependencies is not None:
        projected = _powder_interpolation_dependencies(background, points, data.shape,
            q_dim, energy_dim, q_centers, energy_centers, valid_background.ravel(), interpolation)
        interpolated_variance = projected.variance()
        dependencies = combine_source_dependencies((data.source_dependencies, projected), (1, -factor))
    output_signal = np.asarray(data.signal, dtype=float) - factor * interpolated
    output_errors = np.sqrt(
        np.square(np.asarray(data.errors, dtype=float))
        + factor**2 * interpolated_variance
    )
    if dependencies is not None:
        output_errors = np.sqrt(dependencies.variance())
    if dependencies is None:
        validate_independent_source_lineage(data, background)
    metadata = dict(data.metadata)
    metadata.update(merge_source_lineage_metadata(data, background))
    history = list(metadata.get("background_subtractions", []))
    history.append(
        {
            "scale": factor,
            "interpolation": interpolation,
            "projection": {"mode": "center"},
            "source": background.metadata.get("source_file"),
        }
    )
    metadata["background_subtractions"] = history
    metadata["background_uncertainty_dependence"] = "tracked_sources" if dependencies is not None else "assumed_independent"
    metadata["num_events_semantics"] = "sample_event_contributions"
    metadata, channels = _background_derived_payload(data, background, metadata, valid_background & ~data.mask)
    result = replace(
        data,
        signal=np.where(valid_background, output_signal, np.nan),
        errors=np.where(valid_background, output_errors, np.nan),
        mask=np.asarray(data.mask, dtype=bool) | ~valid_background,
        metadata=metadata,
        auxiliary_channels=channels,
        source_dependencies=dependencies,
        counting_dependencies=None,
    )
    contribution = (
        _freeze_owned(interpolated)
        if factor == 1.0
        else _freeze_owned(factor * interpolated)
    )
    interpolated_errors = np.sqrt(np.maximum(interpolated_variance, 0.0))
    contribution_errors = (
        _freeze_owned(interpolated_errors)
        if factor == 1.0
        else _freeze_owned(abs(factor) * interpolated_errors)
    )
    return accumulate_background_channel(
        data,
        result,
        contribution=contribution,
        contribution_errors=contribution_errors,
        valid=valid_background,
    )


def _freeze_owned(array: np.ndarray) -> np.ndarray:
    """Mark a newly allocated numerical result immutable without copying it."""

    array.setflags(write=False)
    return array


def _background_derived_payload(data, background, metadata, measured):
    """Retain provenance and geometric support without stale primary statistics."""
    from .histogram_statistics import (
        EVENT_STATISTICS_CHANNELS,
        EVENT_STATISTICS_KEY,
        NORMALIZATION_DENOMINATOR,
    )
    from .measurement_aggregation import MEASUREMENT_STATISTICS_CHANNELS, MEASUREMENT_STATISTICS_KEY

    explicit = any(source.metadata.get("measurement_target_required", False) or source.source_dependencies is not None or any(key in source.metadata for key in
        ("measurement_contract", MEASUREMENT_STATISTICS_KEY)) for source in (data, background))
    metadata = dict(metadata)
    metadata["measurement_derivation"] = {"version": 1, "operation": "background_subtraction",
        "sample_contract": data.metadata.get("measurement_contract"),
        "background_contract": background.metadata.get("measurement_contract"),
        "uncertainty": metadata["background_uncertainty_dependence"],
        "target": "requires_explicit_selection" if explicit else "legacy_unmarked"}
    if explicit:
        metadata["measurement_target_required"] = True
    for key in ("measurement_contract", EVENT_STATISTICS_KEY, MEASUREMENT_STATISTICS_KEY,
                "counting_uncertainty"):
        metadata.pop(key, None)
    if explicit:
        metadata.pop("normalization_denominator", None)
    metadata["zero_event_bins_are_measured"] = True
    excluded = (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR, *MEASUREMENT_STATISTICS_CHANNELS)
    if not explicit:
        excluded = tuple(name for name in excluded if name != NORMALIZATION_DENOMINATOR)
    channels = {name: channel for name, channel in data.auxiliary_channels.items() if name not in excluded}
    if explicit:
        channels["coverage_fraction"] = MDHistoChannel(np.where(measured, mdhisto_coverage_fraction(data), 0),
            label="Coverage", unit="fraction")
    return metadata, channels


def _powder_interpolation_dependencies(background, points, shape, q_dim, e_dim, q, energy, valid, method):
    """Use the same center interpolation to project background sensitivities."""
    copies = 1 if method == "nearest" else 4
    if len(points) * copies * 80 > background.source_dependencies.max_bytes:
        raise SourceReplayRequired("Background interpolation sensitivities exceed their work budget; replay the final bins")
    coordinates = (q, energy)
    alternatives = []
    for dimension, centers in enumerate(coordinates):
        position = np.clip(np.searchsorted(centers, points[:, dimension]), 0, len(centers)-1)
        previous = np.maximum(position-1, 0)
        if method == "nearest":
            choose = np.where(abs(points[:, dimension]-centers[previous]) <= abs(points[:, dimension]-centers[position]), previous, position)
            alternatives.append(((choose, np.ones(len(points))),))
        else:
            if len(centers) < 2:
                raise SourceReplayRequired("Source-aware linear interpolation needs two centers on each axis or explicit source replay")
            upper = np.clip(np.searchsorted(centers, points[:, dimension], side="right"), 1, len(centers)-1)
            lower = upper-1
            fraction = (points[:, dimension]-centers[lower])/(centers[upper]-centers[lower])
            alternatives.append(((lower, 1-fraction), (upper, fraction)))
    inputs, outputs, coefficients = [], [], []
    for qi, qw in alternatives[0]:
        for ei, ew in alternatives[1]:
            weight = qw*ew
            included = valid & (weight > np.finfo(float).eps)
            source = [None, None]
            source[q_dim], source[e_dim] = qi[included], ei[included]
            inputs.append(np.ravel_multi_index(tuple(source), background.shape))
            outputs.append(np.flatnonzero(included))
            coefficients.append(weight[included])
    return project_source_dependencies(background.source_dependencies, np.concatenate(inputs),
        np.concatenate(outputs), np.concatenate(coefficients), shape)


def _linear_interpolation_variance(
    q_centers: np.ndarray,
    energy_centers: np.ndarray,
    variances: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Propagate independent source variances through bilinear interpolation."""

    q_upper = np.searchsorted(q_centers, points[:, 0], side="right")
    e_upper = np.searchsorted(energy_centers, points[:, 1], side="right")
    q_upper = np.clip(q_upper, 1, q_centers.size - 1)
    e_upper = np.clip(e_upper, 1, energy_centers.size - 1)
    q_lower = q_upper - 1
    e_lower = e_upper - 1
    q_fraction = (points[:, 0] - q_centers[q_lower]) / (
        q_centers[q_upper] - q_centers[q_lower]
    )
    e_fraction = (points[:, 1] - energy_centers[e_lower]) / (
        energy_centers[e_upper] - energy_centers[e_lower]
    )
    inside = (
        (points[:, 0] >= q_centers[0])
        & (points[:, 0] <= q_centers[-1])
        & (points[:, 1] >= energy_centers[0])
        & (points[:, 1] <= energy_centers[-1])
    )
    propagated = np.zeros(points.shape[0], dtype=float)
    valid = inside.copy()
    for q_index, q_weight in (
        (q_lower, 1.0 - q_fraction),
        (q_upper, q_fraction),
    ):
        for e_index, e_weight in (
            (e_lower, 1.0 - e_fraction),
            (e_upper, e_fraction),
        ):
            weight = q_weight * e_weight
            corner_variance = variances[q_index, e_index]
            used = weight > np.finfo(float).eps
            valid &= ~used | np.isfinite(corner_variance)
            propagated += np.where(
                used & np.isfinite(corner_variance),
                np.square(weight) * corner_variance,
                0.0,
            )
    return np.where(valid, propagated, np.nan)


def _powder_dimensions(data: MDHistoData) -> tuple[int, int]:
    q_dimensions = [
        index for index, axis in enumerate(data.axes) if axis.role == "q_modulus"
    ]
    energy_dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(q_dimensions) != 1 or len(energy_dimensions) != 1:
        raise ValueError("background source must contain one |Q| axis and one energy axis")
    return q_dimensions[0], energy_dimensions[0]


def _energy_dimension(data: MDHistoData) -> int:
    dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(dimensions) != 1:
        raise ValueError("background subtraction requires one energy-transfer axis")
    return dimensions[0]
