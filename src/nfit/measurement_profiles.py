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
from .measurement_contracts import MeasurementContract


@dataclass(frozen=True)
class MeasurementProfile:
    """Immutable profile with its scientific declaration and retained payload.

    ``data`` is one dimensional. Its event-count semantics are recorded; counts
    are never inferred from signal, variance, exposure or the number of bins.
    Cross-bin source dependencies are not yet represented by this service.
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
) -> MeasurementProfile:
    """Reduce center-selected pixels into declared profile bins.

    A validated primary count signal pools C,V,N. Other channels and legacy
    continuous data retain an independent Gaussian inverse-variance mean.
    ``reference_values`` lets model predictions use the observation's exposure
    and inclusion rule. It does not assign a counting likelihood to predictions.
    Covered count zeros contribute exposure; unexposed/invalid observations do
    not. Coverage is geometric support, separate from normalization exposure.
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
    statistics = _view_statistics(view, reference, errors, channel)
    declared = view.get("measurement_contract") if channel == "signal" else None
    if declared is not None:
        declared = MeasurementContract.from_dict(declared)
        expected = "exposure_pool" if statistics is not None else "inverse_variance_mean"
        if declared.estimator != expected or declared.dependence != "independent" or declared.normalizer != "known":
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
        good &= errors > 0
        contract = MeasurementContract(kind="continuous", estimator="inverse_variance_mean",
            quantity="common value", value_units=units)
        minimum_error = np.full(bins, np.inf)
        np.minimum.at(minimum_error, indices[good], errors[good])
        weights = np.zeros(values.shape)
        weights[good] = np.square(minimum_error[indices[good]] / errors[good])
        weight = sum_bins(weights, good)
        signal, variance = np.full(bins, np.nan), np.full(bins, np.nan)
        normalized_weights = np.zeros(values.shape)
        normalized_weights[good] = weights[good] / weight[indices[good]]
        signal[weight > 0] = sum_bins(values * normalized_weights, good)[weight > 0]
        variance[weight > 0] = np.square(minimum_error[weight > 0] / np.sqrt(weight[weight > 0]))

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
    mask = ~np.isfinite(signal) | ~np.isfinite(variance) | (fraction < coverage_threshold)
    error = np.sqrt(variance)
    metadata = {"measurement_contract": contract.to_dict(), "zero_event_bins_are_measured": True,
                "signal_semantics": "density", "signal_unit": contract.value_units,
                "profile_covariance": "diagonal_only"}
    if statistics is not None:
        metadata[EVENT_STATISTICS_KEY] = dict(view[EVENT_STATISTICS_KEY])
    if reference_values is not None:
        metadata.pop(EVENT_STATISTICS_KEY, None)
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
        metadata=metadata, auxiliary_channels=channels)
    return MeasurementProfile(data, contract)
