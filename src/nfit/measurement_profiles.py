"""Bounded histogram-profile estimation shared by GUI and scripting consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .histogram_statistics import (
    EVENT_STATISTICS_CHANNELS,
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_VERSION,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
    normalized_event_statistics,
)
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from .measurement_aggregation import (
    MEASUREMENT_STATISTICS_CHANNELS,
    MEASUREMENT_STATISTICS_KEY,
    initialize_measurement_statistics,
    measurement_statistics_channels,
    normalized_measurement_statistics,
)
from .measurement_contracts import MeasurementContract


@dataclass(frozen=True)
class MeasurementProfile:
    """Immutable profile with its scientific declaration and retained payload.

    ``data`` is one dimensional. Its event-count semantics are recorded; counts
    are never inferred from signal, variance, exposure or the number of bins.
    Bounded primitive source sensitivities propagate when supplied.
    """

    data: MDHistoData
    contract: MeasurementContract

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.data.axes[0].centers,
                np.where(self.data.mask, np.nan, self.data.signal),
                np.where(self.data.mask, np.nan, self.data.errors))


def _view_statistics(view, reference, errors, channel):
    marker = view.get(EVENT_STATISTICS_KEY)
    if channel != "signal" or not isinstance(marker, dict) or marker.get("version") != EVENT_STATISTICS_VERSION:
        return None
    names = (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR)
    if any(name not in view or np.shape(view[name]) != reference.shape for name in names):
        return None
    arrays = tuple(np.asarray(view[name], dtype=float) for name in names)
    signal, variance = normalized_event_statistics(*arrays)
    bundle = view.get("counting_dependencies")
    if bundle is not None:
        from .measurement_dependencies import ratio_source_dependencies
        bundle.numerator_dependencies.validate_variances(arrays[1])
        variance = ratio_source_dependencies(bundle, arrays[0], arrays[2]).variance()
    # Display masks may hide selected cells without changing their statistics.
    visible = np.isfinite(reference) & np.isfinite(errors)
    if not (np.allclose(signal[visible], reference[visible], rtol=1e-12, atol=0)
            and np.allclose(variance[visible], np.square(errors[visible]), rtol=1e-12, atol=0)):
        return None
    return arrays


def prepare_measurement_profile(
    view: Mapping[str, Any], values, errors, *, selected, indices, edges,
    centers=None, coverage_weights=None, coverage_threshold=0.0,
    channel="signal", reference_values=None,
    require_exact_background_uncertainty=False,
) -> MeasurementProfile:
    """Reduce center-selected pixels into declared profile bins.

    A validated primary count signal pools C,V,N. Other channels and legacy
    continuous data retain an independent Gaussian inverse-variance mean.
    ``reference_values`` lets model predictions use the observation's exposure
    and inclusion rule. It does not assign a counting likelihood to predictions.
    Covered count zeros contribute exposure; unexposed/invalid observations do
    not. Coverage is geometric support, separate from normalization exposure.
    Cached directional-background previews retain their existing estimator and
    label their diagonal uncertainty approximation. Exact background uncertainty
    requires the explicit original-grid source replay API.
    """
    values, errors = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
    selected, indices = np.asarray(selected, dtype=bool), np.asarray(indices, dtype=int)
    if values.shape != errors.shape or selected.shape != values.shape or indices.shape != values.shape:
        raise ValueError("Profile values, errors, selection and indices must have matching shapes")
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Profile edges must be finite and increasing")
    bins = len(edges) - 1
    if np.any(selected & ((indices < 0) | (indices >= bins))):
        raise ValueError("Selected profile indices must lie within the output grid")
    reference = values if reference_values is None else np.asarray(reference_values, dtype=float)
    if reference.shape != values.shape:
        raise ValueError("Reference values must match observations")
    if not np.isfinite(coverage_threshold) or not 0 <= coverage_threshold <= 1:
        raise ValueError("Coverage threshold must be between zero and one")
    approximation = view.get("background_profile_uncertainty") if channel == "signal" else None
    if require_exact_background_uncertainty and approximation and approximation != "source_covariance":
        from .measurement_dependencies import SourceReplayRequired

        raise SourceReplayRequired("This cached background preview has diagonal uncertainty; use replay_cached_background_profile on the original histogram for exact background covariance")
    statistics = _view_statistics(view, reference, errors, channel)
    declared = view.get("measurement_contract") if channel == "signal" else None
    if declared is None and channel == "signal" and view.get("measurement_target_required"):
        raise ValueError("Choose an explicit statistical target for derived measurements before profiling")
    if declared is not None:
        declared = MeasurementContract.from_dict(declared)
        expected = ("exposure_pool",) if statistics is not None else ("inverse_variance_mean", "uniform_mean")
        if declared.estimator not in expected or (declared.normalizer != "known" and view.get("counting_dependencies") is None) or (declared.dependence != "independent" and view.get("source_dependencies") is None):
            raise ValueError("This profile path requires known exposure or independent precision means; source replay is needed for this contract")
    good = selected & np.isfinite(values) & np.isfinite(reference) & np.isfinite(errors)
    units = declared.value_units if declared is not None else str(view.get("signal_unit") or "unspecified intensity units")
    channels = {}
    mask = np.asarray(view.get("mask", view.get("combined_mask", np.zeros(values.shape, dtype=bool))), dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("Profile masks must match observations")
    if view.get("masks_applied", True):
        good &= ~mask

    def sum_bins(array, valid):
        return np.bincount(indices[valid], weights=np.asarray(array)[valid], minlength=bins)

    if statistics is not None:
        c, v, n = statistics
        good &= (errors >= 0) & np.isfinite(c) & np.isfinite(v) & (v >= 0) & np.isfinite(n) & (n > 0)
        contract = MeasurementContract(kind="counting", estimator="exposure_pool",
            quantity="normalized intensity", value_units=units,
            exposure_units=str(view.get("exposure_unit") or "arbitrary normalization units"))
        numerator = c if reference_values is None else values * n
        pooled = tuple(sum_bins(array, good) for array in (numerator, v, n))
        signal, variance = normalized_event_statistics(*pooled)
        channels.update(event_statistics_channels(*pooled))
    else:
        contract = declared or MeasurementContract(kind="continuous", estimator="inverse_variance_mean",
            quantity="common value", value_units=units)
        generic_marker = view.get(MEASUREMENT_STATISTICS_KEY)
        generic_stats = None
        if isinstance(generic_marker, dict) and generic_marker.get("version") == 1:
            if generic_marker.get("estimator") != contract.estimator:
                raise ValueError("Stored weights do not match the declared estimator; replay sources")
            if any(name not in view or np.shape(view[name]) != values.shape for name in MEASUREMENT_STATISTICS_CHANNELS):
                raise ValueError("Missing additive measurement payload; replay sources")
            generic_stats = tuple(np.asarray(view[name], float) for name in MEASUREMENT_STATISTICS_CHANNELS)
            original, original_variance = normalized_measurement_statistics(*generic_stats)
            visible = np.isfinite(reference) & np.isfinite(errors)
            if not (np.allclose(original[visible], reference[visible], rtol=1e-12, atol=0)
                    and np.allclose(original_variance[visible], errors[visible]**2, rtol=1e-12, atol=0)):
                raise ValueError("Stale additive measurement payload; replay sources")
        if generic_stats is None:
            from dataclasses import replace
            generic_stats, generic_marker = initialize_measurement_statistics(
                reference, errors, replace(contract, missing="omit", dependence="independent"), mask=~good)
        source_weights = generic_stats[2]
        good &= np.isfinite(source_weights) & (source_weights > 0) & (errors >= 0)
        numerator = generic_stats[0] if reference_values is None else values*source_weights
        pooled = tuple(sum_bins(array, good) for array in (numerator, *generic_stats[1:]))
        signal, variance = normalized_measurement_statistics(*pooled)
        channels.update(measurement_statistics_channels(*pooled))

    if declared is not None:
        contract = declared
        if contract.missing == "reject" and np.any(selected & ~good):
            raise ValueError("The contract rejects missing, masked or unexposed profile observations")

    coverage = np.asarray(view.get("coverage_fraction", np.ones(values.shape)), dtype=float)
    widths = np.broadcast_to(1.0 if coverage_weights is None else coverage_weights, values.shape)
    if coverage.shape != values.shape or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Coverage and positive geometric weights must match observations")
    covered = np.where(good & np.isfinite(coverage), np.clip(coverage, 0, 1), 0)
    support = sum_bins(widths, selected)
    fraction = np.zeros(bins)
    np.divide(sum_bins(covered * widths, selected), support, out=fraction, where=support > 0)
    output_dependencies = None
    output_counting_dependencies = None
    dependencies = view.get("source_dependencies")
    bundle = view.get("counting_dependencies") if statistics is not None else None
    if bundle is not None:
        from .measurement_dependencies import (
            CountingDependencies,
            project_source_dependencies,
            ratio_source_dependencies,
        )
        inputs, outputs = np.flatnonzero(good), indices[good]
        count_bundle = CountingDependencies(
            numerator_dependencies=project_source_dependencies(bundle.numerator_dependencies, inputs, outputs, np.ones(len(inputs)), (bins,)),
            exposure_dependencies=project_source_dependencies(bundle.exposure_dependencies, inputs, outputs, np.ones(len(inputs)), (bins,)))
        observed_numerator = sum_bins(statistics[0], good)
        observed_dependencies = ratio_source_dependencies(count_bundle, observed_numerator, pooled[2])
        variance = np.where(pooled[2] > 0, observed_dependencies.variance(), np.nan)
        pooled = (pooled[0], count_bundle.numerator_dependencies.variance(), pooled[2])
        channels.update(event_statistics_channels(*pooled))
        if reference_values is None:
            output_dependencies, output_counting_dependencies = observed_dependencies, count_bundle
    elif dependencies is not None:
        from .measurement_dependencies import project_source_dependencies
        dependencies.validate_variances(errors**2, mask=~good)
        weights = statistics[2] if statistics is not None else source_weights
        coefficients = weights[good]/pooled[2][indices[good]]
        observed_dependencies = project_source_dependencies(dependencies, np.flatnonzero(good),
            indices[good], coefficients, (bins,))
        variance = np.where(pooled[2] > 0, observed_dependencies.variance(), np.nan)
        if reference_values is None:
            output_dependencies = observed_dependencies
        pooled = (pooled[0], np.where(pooled[2] > 0, variance*pooled[2]**2, 0), *pooled[2:])
        channels.update(event_statistics_channels(*pooled) if statistics is not None
                        else measurement_statistics_channels(*pooled))
    mask = ~np.isfinite(signal) | ~np.isfinite(variance) | (fraction < coverage_threshold)
    error = np.sqrt(variance)
    metadata = {"measurement_contract": contract.to_dict(), "zero_event_bins_are_measured": True,
                "signal_semantics": "density", "signal_unit": contract.value_units,
                "profile_covariance": "source_sensitivities" if output_dependencies is not None else "diagonal_only",
                "measurement_contract_origin": "explicit" if declared is not None else "validated_count_statistics" if statistics is not None else "legacy_precision_mean"}
    if statistics is not None:
        metadata[EVENT_STATISTICS_KEY] = dict(view[EVENT_STATISTICS_KEY])
    else:
        metadata[MEASUREMENT_STATISTICS_KEY] = dict(generic_marker)
    if approximation:
        metadata["background_profile_uncertainty"] = approximation
        metadata["background_profile_target"] = "exposure_weighted_preview_field" if statistics is not None else contract.estimator + "_of_preview_pixels"
        metadata["exact_background_profile_target"] = view.get("exact_background_profile_target")
    if reference_values is not None:
        metadata.pop(EVENT_STATISTICS_KEY, None)
        metadata.pop(MEASUREMENT_STATISTICS_KEY, None)
        metadata["profile_role"] = "model_prediction"
        metadata["profile_uncertainty"] = "observation_error_for_overlay"
    events = np.asarray(view.get("num_events", np.zeros(values.shape)), dtype=float)
    if events.shape != values.shape:
        raise ValueError("Event contributions must match observations")
    events = sum_bins(events, good & np.isfinite(events) & (events >= 0))
    metadata["num_events_semantics"] = view.get("num_events_semantics", "not_available")
    if reference_values is not None:
        events = np.zeros(bins)
        metadata["num_events_semantics"] = "not_available"
    channels["coverage_fraction"] = MDHistoChannel(fraction, label="Coverage", unit="fraction")
    axis_metadata = {} if centers is None else {"discrete_centers": np.asarray(centers, dtype=float).tolist()}
    data = MDHistoData(axes=(MDHistoAxis("Box profile", edges, "", "unknown", metadata=axis_metadata),),
        signal=signal, errors=error, mask=mask, num_events=events,
        metadata=metadata, auxiliary_channels=channels, source_dependencies=output_dependencies, counting_dependencies=output_counting_dependencies)
    return MeasurementProfile(data, contract)
