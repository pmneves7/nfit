"""Additive event statistics and separate Poisson rate confidence intervals.

The stored variance is accumulated event variance, including any represented
same-bin copy cross terms, not a confidence interval for an unknown rate.
Normalization is treated as known. Only diagonal variances are represented;
copies landing in different bins require additional covariance.
"""

from __future__ import annotations

import numpy as np

EVENT_SIGNAL_NUMERATOR = "event_signal_numerator"
EVENT_VARIANCE_NUMERATOR = "event_variance_numerator"
NORMALIZATION_DENOMINATOR = "normalization_denominator"
EVENT_STATISTICS_KEY = "event_statistics"
EVENT_STATISTICS_VERSION = 1
EVENT_STATISTICS_METADATA = {
    "version": EVENT_STATISTICS_VERSION,
    "variance": "accumulated_event_variance",
    "exposure": "known",
    "covariance": "diagonal_only",
}
EVENT_STATISTICS_CHANNELS = (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR)


def event_statistics_channels(numerator, variance, normalization):
    """Build immutable channels for an explicitly reduced count histogram."""
    from .mdhisto import MDHistoChannel

    return {
        EVENT_SIGNAL_NUMERATOR: MDHistoChannel(numerator, label="Event signal numerator"),
        EVENT_VARIANCE_NUMERATOR: MDHistoChannel(variance, label="Event numerator variance"),
        NORMALIZATION_DENOMINATOR: MDHistoChannel(
            normalization,
            label="Normalization exposure",
            unit="arbitrary normalization units",
        ),
    }


def scaled_event_statistics_channels(data, factor):
    """Scale numerator by a known factor and its variance by the squared factor.

    Exposure remains unchanged. Other auxiliary channels are preserved; their
    physical transformations are owned by the caller.
    """
    channels = dict(data.auxiliary_channels)
    if not has_event_statistics(data):
        return channels
    from .mdhisto import MDHistoChannel

    for name, power in ((EVENT_SIGNAL_NUMERATOR, 1), (EVENT_VARIANCE_NUMERATOR, 2)):
        source = channels[name]
        values = source.values * np.asarray(factor, dtype=float) ** power
        channels[name] = MDHistoChannel(
            values,
            label=source.label,
            unit=source.unit,
            quantity_type=source.quantity_type,
        )
    return channels


def normalized_event_statistics(numerator, variance, normalization):
    """Return normalized signal and accumulated event variance, including zeros."""
    numerator, variance, normalization = np.broadcast_arrays(numerator, variance, normalization)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(variance)
        & (variance >= 0)
        & np.isfinite(normalization)
        & (normalization > 0)
    )
    signal = np.full(numerator.shape, np.nan, dtype=float)
    rate_variance = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, normalization, out=signal, where=valid)
    np.divide(variance, normalization**2, out=rate_variance, where=valid)
    return signal, rate_variance


def pool_event_statistics(numerator, variance, normalization, axes, *, mask=None):
    """Sum valid independent numerator statistics and exposure before division."""
    numerator, variance, normalization = np.broadcast_arrays(numerator, variance, normalization)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(variance)
        & (variance >= 0)
        & np.isfinite(normalization)
        & (normalization > 0)
    )
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    return tuple(
        np.sum(np.where(valid, array, 0.0), axis=tuple(axes))
        for array in (numerator, variance, normalization)
    )


def has_event_statistics(data):
    """Check the contract and array shapes without accessing array payloads."""
    marker = data.metadata.get(EVENT_STATISTICS_KEY)
    if not isinstance(marker, dict) or marker.get("version") != EVENT_STATISTICS_VERSION:
        return False
    if any(
        data.metadata.get(key)
        for key in (
            "background_subtractions",
            "background_projection",
            "background_channels_version",
        )
    ):
        return False
    channels = data.auxiliary_channels
    return all(
        name in channels and channels[name].values.shape == data.shape
        for name in (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR)
    )


def selected_event_statistics(data, selection=(...,), *, validate=True):
    """Read selected explicit statistics; never infer them from event counts.

    Primary replacements and unsupported transforms can inherit auxiliary
    channels. Selected-array validation prevents those stale statistics from
    overriding current signal/error values, without reading an entire volume.
    Background differences have their own measurement model and are excluded.
    """
    if not has_event_statistics(data):
        return None
    channels = data.auxiliary_channels
    names = (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR)
    arrays = tuple(np.asarray(channels[name].values[selection], dtype=float) for name in names)
    if validate:
        signal, variance = normalized_event_statistics(*arrays)
        stored_signal = np.asarray(data.signal[selection], dtype=float)
        stored_variance = np.square(np.asarray(data.errors[selection], dtype=float))
        if not (
            np.allclose(signal, stored_signal, rtol=1e-12, atol=0.0, equal_nan=True)
            and np.allclose(
                variance, stored_variance, rtol=1e-12, atol=0.0, equal_nan=True
            )
        ):
            return None
    return arrays


def poisson_rate_interval(counts, normalization, *, confidence=0.682689492137, constant_weight=1.0):
    """Exact equal-tailed Garwood interval for independent constant-weight counts.

    ``counts`` are nonnegative integer observations. ``normalization`` is known
    exposure; the rate is ``constant_weight * counts / normalization``. A known
    positive constant event weight is allowed. Heterogeneous correction weights,
    reused events, background differences, and uncertain exposure do not satisfy
    this model. Compute the interval after pooling the final desired bin.
    """
    from scipy.stats import chi2

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not np.isfinite(constant_weight) or constant_weight <= 0:
        raise ValueError("constant_weight must be finite and positive")
    counts, normalization = np.broadcast_arrays(
        np.asarray(counts, dtype=float),
        np.asarray(normalization, dtype=float),
    )
    if np.any(~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts))):
        raise ValueError("counts must be finite nonnegative integers")
    if np.any(~np.isfinite(normalization) | (normalization < 0)):
        raise ValueError("normalization must be finite and nonnegative")
    tail = (1.0 - confidence) / 2.0
    lower_count = np.zeros(counts.shape)
    positive = counts > 0
    lower_count[positive] = 0.5 * chi2.ppf(tail, 2.0 * counts[positive])
    upper_count = 0.5 * chi2.ppf(1.0 - tail, 2.0 * (counts + 1.0))
    lower = np.full(counts.shape, np.nan)
    upper = np.full(counts.shape, np.nan)
    np.divide(constant_weight * lower_count, normalization, out=lower, where=normalization > 0)
    np.divide(constant_weight * upper_count, normalization, out=upper, where=normalization > 0)
    return lower, upper
