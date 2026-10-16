"""Shared accessors for stored optimizer and posterior fit results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fit_sampling import sampling_result_from_dict, sampling_result_is_posterior
from .pipeline import FitTimelineEntry

POSTERIOR_DISPLAY_KEY = "posterior_display"


def sampling_payload_info(payload: Any) -> dict[str, Any]:
    """Inspect encoded sampling shapes without decompressing scientific arrays."""

    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and (
        metadata.get("purpose", "posterior") != "posterior"
        or metadata.get("temperature", 1.0) != 1.0
    ):
        return {}
    samples = payload.get("samples")
    shape = samples.get("shape", []) if isinstance(samples, dict) else []
    chain = payload.get("chain")
    chain_shape = chain.get("shape", []) if isinstance(chain, dict) else []
    if len(shape) != 2:
        return {}
    return {
        "samples": int(shape[0]),
        "variable_names": [str(name) for name in payload.get("variable_names", [])],
        "chain_steps": int(chain_shape[0]) if len(chain_shape) == 3 else 0,
        "chain_walkers": int(chain_shape[1]) if len(chain_shape) == 3 else 0,
        "has_chain": len(chain_shape) == 3 and chain_shape[0] > 0,
    }


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

    info = sampling_payload_info(fit_entry.metadata.get("posterior_samples"))
    if not info:
        return None
    names = info["variable_names"]
    posterior = fit_entry.goodness.get("posterior", {})
    correlation = posterior.get("correlation") if isinstance(posterior, dict) else None
    if isinstance(correlation, dict) and names:
        try:
            matrix = np.asarray([[correlation[name][other] for other in names] for name in names], dtype=float)
            if np.all(np.isfinite(matrix)):
                return matrix, names
        except (KeyError, TypeError, ValueError):
            pass
    result = sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if result is None or not sampling_result_is_posterior(result):
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
