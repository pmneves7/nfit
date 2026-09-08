"""Shared accessors for stored optimizer and posterior fit results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fitting import SamplingResult
from .pipeline import FitTimelineEntry
from .project_io import _decode_float_array

POSTERIOR_DISPLAY_KEY = "posterior_display"


def sampling_result_from_dict(payload: Any) -> SamplingResult | None:
    """Decode one serialized posterior sampling result."""

    if not isinstance(payload, dict):
        return None
    try:
        samples = _decode_float_array(payload["samples"])
    except Exception:
        return None
    log_probability = None
    if isinstance(payload.get("log_probability"), dict):
        try:
            log_probability = _decode_float_array(payload["log_probability"])
        except Exception:
            log_probability = None
    chain = None
    if isinstance(payload.get("chain"), dict):
        try:
            chain = _decode_float_array(payload["chain"])
        except Exception:
            chain = None
    log_probability_chain = None
    if isinstance(payload.get("log_probability_chain"), dict):
        try:
            log_probability_chain = _decode_float_array(payload["log_probability_chain"])
        except Exception:
            log_probability_chain = None
    return SamplingResult(
        samples=samples,
        variable_names=[str(name) for name in payload.get("variable_names", [])],
        log_probability=log_probability,
        metadata=dict(payload.get("metadata", {})),
        chain=chain,
        log_probability_chain=log_probability_chain,
    )


def posterior_display_options(fit_entry: FitTimelineEntry) -> dict[str, Any]:
    """Return the persisted choice of posterior-derived result values."""

    metadata = fit_entry.metadata if isinstance(fit_entry.metadata, dict) else {}
    stored = metadata.get(POSTERIOR_DISPLAY_KEY)
    return dict(stored) if isinstance(stored, dict) else {}


def display_fit_parameters(fit_entry: FitTimelineEntry) -> dict[str, float]:
    """Return the parameter values selected for the active fit result."""

    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = goodness.get("parameters") if isinstance(goodness.get("parameters"), dict) else {}
    displayed = {str(name): float(value) for name, value in params.items()}
    options = posterior_display_options(fit_entry)
    best = options.get("best_sample") if options.get("use_best_sample") else None
    best_params = best.get("parameters") if isinstance(best, dict) else None
    if isinstance(best_params, dict):
        for name, value in best_params.items():
            try:
                displayed[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
    return displayed


def posterior_correlation_matrix(
    fit_entry: FitTimelineEntry,
) -> tuple[np.ndarray, list[str]] | None:
    """Return the finite-sample emcee correlation matrix for display."""

    result = sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if result is None:
        return None
    samples = np.asarray(result.samples, dtype=float)
    names = list(result.variable_names)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != len(names):
        return None
    finite_rows = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite_rows]
    if samples.shape[0] < 2:
        return None
    if samples.shape[1] == 1:
        return np.ones((1, 1), dtype=float), names
    correlation = np.asarray(np.corrcoef(samples, rowvar=False), dtype=float)
    if correlation.shape != (len(names), len(names)):
        return None
    return correlation, names


# Private aliases retained for compatibility with the existing project GUI API.
_sampling_result_from_dict = sampling_result_from_dict
_posterior_display_options = posterior_display_options
_display_fit_parameters = display_fit_parameters
_posterior_correlation_matrix = posterior_correlation_matrix
