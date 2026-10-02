"""Explicit Gaussian and integer-count fit objectives with admissibility checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurement_contracts import MeasurementContract
from .measurement_dependencies import SourceReplayRequired
from .source_lineage import validate_fit_source_lineage

FIT_LIKELIHOODS = ("gaussian", "gaussian_gls", "poisson_deviance")


def validate_fit_dataset_independence(datasets):
    """Reject reused primitives across separately evaluated dataset blocks."""
    datasets = tuple(datasets)
    validate_fit_source_lineage(datasets)
    seen = set()
    for name, points in datasets:
        payload = points.source_dependencies
        if payload is None:
            continue
        used = set(payload.source_ids[index] for index in np.unique(payload.source_indices))
        if seen & used:
            raise SourceReplayRequired(f"Dataset {name!r} reuses primitives from another fit dataset; combine the selections into one joint GLS dataset or replay independent sources")
        seen.update(used)


@dataclass(frozen=True, kw_only=True)
class PoissonCountModel:
    """Audited independent integer primitives with a known constant event weight.

    This declaration is never inferred from integer-looking corrected arrays.
    ``provenance`` identifies the source-level justification for independent
    counts, known exposure and the constant weight (dimensionless).
    """
    constant_weight: float
    provenance: str

    def __post_init__(self):
        if not np.isfinite(self.constant_weight) or self.constant_weight <= 0:
            raise ValueError("Poisson constant_weight must be finite and positive")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Poisson count model requires source-level provenance")

    def to_dict(self):
        return {"version": 1, "constant_weight": self.constant_weight, "provenance": self.provenance,
                "counts": "independent_integer_primitives", "exposure": "known"}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("counts") != "independent_integer_primitives" or payload.get("exposure") != "known":
            raise ValueError("Poisson likelihood requires an audited independent integer count declaration")
        return cls(constant_weight=float(payload["constant_weight"]), provenance=payload["provenance"])


def poisson_count_statistics(data):
    """Validate an explicit primitive model and its retained counting arrays."""
    from .histogram_statistics import (
        EVENT_SIGNAL_NUMERATOR,
        EVENT_VARIANCE_NUMERATOR,
        NORMALIZATION_DENOMINATOR,
    )
    model = PoissonCountModel.from_dict(data.metadata.get("poisson_count_model"))
    declaration = data.metadata.get("measurement_contract")
    if declaration is not None:
        contract = MeasurementContract.from_dict(declaration)
        if contract.kind != "counting" or contract.dependence != "independent" or contract.normalizer != "known":
            raise ValueError("Poisson likelihood requires independent counts and known exposure")
    if any(data.metadata.get(key) for key in ("background_subtractions", "background_projection", "background_channels_version", "histogram_arithmetic", "bose_separation")):
        raise ValueError("Signed/reconstructed or background-subtracted data require a different likelihood")
    if data.source_dependencies is not None:
        raise ValueError("Poisson primitive likelihood does not accept reused or reconstructed source payloads")
    marker = data.metadata.get("event_statistics", {})
    if marker.get("covariance") not in {None, "diagonal_only"} or marker.get("symmetry_variance") not in {None, "independent_copies"}:
        raise ValueError("Symmetry/reconstructed counts require a source-level likelihood")
    symmetry = data.metadata.get("symmetry_operations_hkl")
    if symmetry is not None and (len(symmetry) != 1 or not np.allclose(symmetry[0], np.eye(3), rtol=0, atol=0)):
        raise ValueError("Symmetry-expanded counts are not independent integer primitives")
    payload = data.measurement_payload
    names = (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)
    if payload is None or any(name not in payload for name in names):
        raise ValueError("Poisson likelihood requires retained numerator, variance and exposure")
    numerator, variance, exposure = (np.asarray(payload[name], float) for name in names)
    counts = numerator / model.constant_weight
    valid = np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts)) & np.isfinite(exposure) & (exposure > 0)
    if not np.all(valid) or not np.allclose(variance, counts * model.constant_weight**2, rtol=1e-12, atol=0):
        raise ValueError("Retained values do not match the declared constant-weight integer primitives")
    if not (np.allclose(data.intensity, numerator / exposure, rtol=1e-12, atol=0)
            and np.allclose(data.sigma**2, variance / exposure**2, rtol=1e-12, atol=0)):
        raise ValueError("Poisson sufficient statistics do not reproduce the fitted observations")
    return counts, exposure, model.constant_weight


def poisson_deviance_residuals(counts, expected_counts):
    """Signed square-root Poisson deviance, including measured zero counts."""
    counts, expected = np.broadcast_arrays(np.asarray(counts, float), np.asarray(expected_counts, float))
    if np.any(~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts))):
        raise ValueError("Poisson observations must be finite nonnegative integers")
    if np.any(~np.isfinite(expected) | (expected < 0)):
        raise ValueError("Poisson model predictions must be finite and nonnegative")
    deviance = expected - counts
    positive = counts > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(positive, np.log(counts) - np.log(expected), 0.)
        deviance = deviance + np.where(positive, counts * log_ratio, 0.)
        relative = np.zeros(counts.shape)
        np.divide(counts - expected, expected, out=relative, where=expected > 0)
        close = positive & (expected > 0) & (np.abs(relative) < 1e-4)
        x = relative
        series = expected * x**2 * (0.5 - x/6 + x**2/12 - x**3/20)
        deviance = np.where(close, series, deviance)
    return np.sign(counts - expected) * np.sqrt(2 * np.maximum(deviance, 0))



def gaussian_whitener(data, *, max_bytes=128 * 1024**2):
    """Bounded exact GLS factor for a small prepared source-aware fit.

    Dense covariance is restricted to fit selections, never entire project
    histograms. Singular covariance requires source-level fitting/replay rather
    than silently dropping deterministic constraints.
    """
    payload = data.source_dependencies
    if payload is None:
        raise SourceReplayRequired("GLS requires tracked source dependencies")
    payload.validate_variances(data.sigma**2)
    size = data.size
    if size * size * 8 * 4 > max_bytes:
        raise SourceReplayRequired("GLS fit covariance exceeds its work budget; fit/replay primitive sources or reduce the selection")
    covariance = np.zeros((size, size))
    # Group by source; each sparse outer product retains signed covariance.
    order = np.argsort(payload.source_indices, kind="stable")
    sources = payload.source_indices[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sources)) + 1, len(sources)]
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        positions = order[start:stop]
        if not positions.size:
            continue
        rows = payload.observation_indices[positions]
        coefficients = payload.coefficients[positions]
        covariance[np.ix_(rows, rows)] += np.outer(coefficients, coefficients) * payload.source_variances[sources[start]]
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise SourceReplayRequired("Singular shared-source covariance requires fitting the primitive sources") from exc
    return np.linalg.solve(factor, np.eye(size))


def fit_measurement_residuals(data, predictions, likelihood, *, data_scale=1., whitener=None):
    """Evaluate one declared objective; keep statistical assumptions explicit."""
    if likelihood not in FIT_LIKELIHOODS:
        raise ValueError(f"Unknown fit likelihood {likelihood!r}")
    if str(data.metadata.get("cross_reflection_covariance", "")).startswith("not_retained"):
        raise SourceReplayRequired("Integrated reflections have unretained cross-reflection covariance; replay or fit their primitive sources")
    predictions = np.asarray(predictions, float)
    if likelihood == "poisson_deviance":
        if not np.isfinite(data_scale) or data_scale <= 0:
            raise ValueError("Poisson data scaling must be finite and positive")
        counts, exposure, weight = poisson_count_statistics(data)
        return poisson_deviance_residuals(counts, exposure * predictions / (data_scale * weight))
    if likelihood == "gaussian_gls":
        if whitener is None:
            whitener = gaussian_whitener(data)
        return (whitener @ (data_scale * data.intensity - predictions)) / max(abs(data_scale), np.finfo(float).tiny)
    declaration = data.metadata.get("measurement_contract")
    if declaration is not None and MeasurementContract.from_dict(declaration).dependence != "independent":
        raise SourceReplayRequired("Independent Gaussian fitting does not represent shared sources; select GLS or fit primitive sources")
    payload = data.source_dependencies
    if payload is not None and len(np.unique(payload.source_indices)) != len(payload.source_indices):
        raise SourceReplayRequired("Independent Gaussian fitting would count shared sources repeatedly; select GLS or replay sources")
    sigma = np.asarray(data.sigma) * max(abs(data_scale), np.finfo(float).tiny)
    if np.any(sigma <= 0):
        raise ValueError("Gaussian fitting requires positive sigma; measured count zeros need a supported count likelihood")
    return (data_scale * data.intensity - predictions) / sigma


def histogram_view_residuals(view):
    """Recompute pointwise Gaussian/count diagnostics at the displayed grid.

    A pooled count diagnostic uses the predicted pooled count mean. Whitened
    coordinates cannot be spatially interpreted as fresh independent residuals;
    their selected/summed components remain a display diagnostic only.
    """
    likelihood = view.get("fit_likelihood", "gaussian")
    if likelihood == "gaussian_gls":
        return view.get("residual")
    result = np.full(np.asarray(view["signal"]).shape, np.nan)
    scale = float(view.get("fit_data_scale", 1.))
    if likelihood == "poisson_deviance":
        model = PoissonCountModel.from_dict(view.get("poisson_count_model"))
        c = np.asarray(view["event_signal_numerator"])
        n = np.asarray(view["normalization_denominator"])
        if scale <= 0:
            raise ValueError("Poisson diagnostic scaling must be positive")
        expected = n * np.asarray(view["fit"]) / (scale * model.constant_weight)
        counts = c / model.constant_weight
        valid = np.isfinite(counts) & np.isfinite(expected) & (n > 0) & (expected >= 0)
        result[valid] = poisson_deviance_residuals(counts[valid], expected[valid])
    else:
        np.divide(scale * np.asarray(view["signal"]) - view["fit"], abs(scale) * np.asarray(view["errors"]), out=result, where=(np.asarray(view["errors"]) > 0) & (scale != 0))
    return result
