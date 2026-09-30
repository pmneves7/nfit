"""Exposure pooling for already normalized histogram intensities."""

from __future__ import annotations

import numpy as np


def normalization_denominator(data, selection=(...,)):
    """Select the physical exposure channel without loading the full volume."""
    channel = data.auxiliary_channels.get("normalization_denominator")
    values = channel.values if channel is not None else data.metadata.get("normalization_denominator")
    if values is None or getattr(values, "shape", None) != data.shape:
        return None
    return np.asarray(values[selection], dtype=float)


def pool_normalized_histogram(values, variance, denominator, axes, *, mask=None):
    """Pool count numerators and independent variances, then divide once.

    A positive exposure with zero counts remains a measurement. Exposure is
    treated as known; this propagates the stored diagonal statistical variance.
    """
    valid = np.isfinite(values) & np.isfinite(denominator) & (denominator > 0.0)
    if variance is not None:
        valid &= np.isfinite(variance) & (variance >= 0.0)
    if mask is not None:
        valid &= ~mask
    weights = np.where(valid, denominator, 0.0)
    axes = tuple(axes)
    exposure = np.sum(weights, axis=axes)
    numerator = np.sum(np.where(valid, values, 0.0) * weights, axis=axes)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, exposure, out=result, where=exposure > 0.0)
    result_variance = None
    if variance is not None:
        numerator_variance = np.sum(np.where(valid, variance, 0.0) * weights**2, axis=axes)
        result_variance = np.full(numerator.shape, np.nan)
        np.divide(numerator_variance, exposure**2, out=result_variance, where=exposure > 0.0)
    return result, result_variance, exposure
