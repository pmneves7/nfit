"""Scalar reference estimators for explicit measurement contracts.

These services establish one-bin semantics before project workflow integration.
They do not dispatch existing importers, cuts or fits. Shared-source sensitivities
represent sparse linear covariance without a dense histogram covariance matrix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .histogram_statistics import pool_event_statistics
from .measurement_contracts import MeasurementContract


@dataclass(frozen=True)
class SourceTerm:
    """Sensitivity to one independent primitive source with known variance.

    Reuse ``source_id`` wherever the same primitive source contributes. Its
    variance must have the same units/convention everywhere; ``coefficient``
    converts a source fluctuation into the observation's units.
    """

    source_id: str
    coefficient: float
    variance: float

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a nonempty string")
        if not math.isfinite(self.coefficient):
            raise ValueError("Source coefficient must be finite")
        if not math.isfinite(self.variance) or self.variance < 0:
            raise ValueError("Source variance must be finite and nonnegative")


@dataclass(frozen=True)
class MeasurementEstimate:
    """Estimate and propagated variance conditional on the declared model.

    Support is exposure for counts, coordinate length for sampled functions,
    and included observation count otherwise. It is not geometric coverage
    for counts. ``coverage_fraction`` is defined only for coordinate intervals.
    Missing results have NaN value/variance, rather than an invented zero.
    """

    value: float
    variance: float
    units: str
    support: float
    included: int
    excluded: int
    coverage_fraction: float | None = None
    numerator: float | None = None
    numerator_variance: float | None = None
    exposure: float | None = None
    exposure_variance: float | None = None
    uncertainty: str = "linear_propagation"
    chi2: float | None = None
    degrees_of_freedom: int | None = None

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance)

    @property
    def measured(self) -> bool:
        return math.isfinite(self.value) and math.isfinite(self.variance)


def _arrays(values, variances, mask):
    values = np.asarray(values, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if values.ndim != 1 or variances.shape != values.shape:
        raise ValueError("values and variances must be matching one-dimensional arrays")
    masked = np.zeros(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if masked.shape != values.shape:
        raise ValueError("mask must match observations")
    if np.any((variances < 0) & ~masked):
        raise ValueError("Observation variances must be nonnegative")
    valid = ~masked & np.isfinite(values) & np.isfinite(variances)
    return values, variances, valid


def _terms(variances, valid, source_terms):
    if source_terms is None or len(source_terms) != len(variances):
        raise ValueError("shared_sources requires source terms for every observation")
    result = []
    for variance, okay, terms in zip(variances, valid, source_terms, strict=True):
        terms = tuple(terms)
        if any(not isinstance(term, SourceTerm) for term in terms):
            raise ValueError("Source terms must be SourceTerm objects")
        # Duplicate terms within one observation are also correlated.
        if okay:
            diagonal = _source_variance([(1.0, terms)])
            if not math.isclose(diagonal, float(variance), rel_tol=1e-10, abs_tol=0.0):
                raise ValueError("Source terms do not reproduce the supplied observation variance")
        result.append(terms)
    return tuple(result)


def _source_variance(weighted_terms):
    coefficients = {}
    variances = {}
    for weight, terms in weighted_terms:
        if weight == 0:
            continue
        for term in terms:
            key = term.source_id
            if key in variances and variances[key] != term.variance:
                raise ValueError(f"Inconsistent variance for shared source {key!r}")
            variances[key] = term.variance
            coefficients.setdefault(key, []).append(float(weight) * term.coefficient)
    return math.fsum(_scaled_variance(math.fsum(values), variances[key]) for key, values in coefficients.items())


def _scaled_variance(coefficient, variance):
    coefficient = float(coefficient)
    return coefficient * (coefficient * float(variance))


def _variance(weights, variances, terms):
    if terms is None:
        return math.fsum(_scaled_variance(weight, variance)
                         for weight, variance in zip(weights, variances, strict=True) if weight != 0)
    return _source_variance(zip(weights, terms, strict=True))


def _interval_weights(coordinates, valid, interval):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != valid.shape or np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0):
        raise ValueError("Coordinates must be finite, strictly increasing and match observations")
    if interval is None or len(interval) != 2:
        raise ValueError("Sampled functions require an explicit (lower, upper) interval")
    lower, upper = map(float, interval)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Interval bounds must be finite and increasing")
    weights = np.zeros(coordinates.shape)
    covered = []
    for index in range(len(coordinates) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        left, right = coordinates[index:index + 2]
        a, b = max(left, lower), min(right, upper)
        if b <= a:
            continue
        width = b - a
        # Integrate the two linear interpolation basis functions exactly.
        fraction = ((a - left) + (b - left)) / (2 * (right - left))
        weights[index] += width * (1 - fraction)
        weights[index + 1] += width * fraction
        covered.append(width)
    support = math.fsum(covered)
    coverage = min(1.0, support / (upper - lower))
    return weights, support, coverage


def estimate_measurement_bin(
    contract: MeasurementContract,
    values,
    variances,
    *,
    mask=None,
    exposure=None,
    exposure_variance=None,
    coordinates=None,
    interval=None,
    coefficients=None,
    source_terms: Sequence[Sequence[SourceTerm]] | None = None,
    exposure_source_terms: Sequence[Sequence[SourceTerm]] | None = None,
) -> MeasurementEstimate:
    """Estimate one bin under an explicit contract, without instrument inference.

    For counts, values/variances are corrected numerators and their variance;
    covered zeros contribute exposure. For other kinds they are measurements
    and their variances. Uncertain exposures use first-order ratio propagation;
    this is not a likelihood or confidence interval. Source terms declare all
    represented independent primitives, including shared calibration sources.
    Missing values or masks remove both the value and its corresponding support.
    Missing or masked sampled nodes break adjacent segments. Widely spaced
    valid nodes interpolate according to the explicit linear model.
    """
    if not isinstance(contract, MeasurementContract):
        raise TypeError("contract must be a MeasurementContract")
    values, variances, valid = _arrays(values, variances, mask)
    if contract.kind != "counting" and (exposure is not None or exposure_variance is not None or exposure_source_terms is not None):
        raise ValueError("Exposure inputs apply only to counting")
    if contract.kind != "sampled_function" and (coordinates is not None or interval is not None):
        raise ValueError("Coordinate inputs apply only to sampled functions")
    if contract.kind != "linear_reconstruction" and coefficients is not None:
        raise ValueError("Explicit coefficients apply only to linear reconstructions")
    if contract.dependence == "independent" and (source_terms is not None or exposure_source_terms is not None):
        raise ValueError("Declare shared_sources when supplying source terms")

    if contract.kind == "counting":
        exposure = np.asarray(exposure, dtype=float)
        if exposure.shape != values.shape or np.any((exposure < 0) & valid):
            raise ValueError("Exposure must match observations and be nonnegative")
        valid &= np.isfinite(exposure) & (exposure > 0)
        if np.any((values < 0) & valid):
            raise ValueError("Counting numerators must be nonnegative; signed values need a reconstruction contract")
        if contract.normalizer == "uncertain":
            exposure_variance = np.asarray(exposure_variance, dtype=float)
            if exposure_variance.shape != values.shape or np.any((exposure_variance < 0) & valid):
                raise ValueError("Uncertain exposure requires matching nonnegative exposure_variance")
            valid &= np.isfinite(exposure_variance)
        elif exposure_variance is not None or exposure_source_terms is not None:
            raise ValueError("Known exposure must not carry ignored uncertainty inputs")

    if contract.missing == "reject" and not np.all(valid):
        raise ValueError("The contract rejects missing, masked or unexposed observations")
    terms = _terms(variances, valid, source_terms) if contract.dependence == "shared_sources" else None
    included = int(np.count_nonzero(valid))
    common = dict(units=contract.output_units, included=included, excluded=len(values) - included)
    if contract.kind == "counting":
        c, _v, n = map(float, pool_event_statistics(values, variances, exposure, (0,), mask=~valid))
        weights = valid.astype(float)
        v = _variance(weights, variances, terms)
        nv = 0.0
        if contract.normalizer == "uncertain":
            nterms = _terms(exposure_variance, valid, exposure_source_terms) if terms is not None else None
            nv = _variance(weights, exposure_variance, nterms)
        variance = (v / n) / n if n > 0 else math.nan
        if n > 0 and contract.normalizer == "uncertain":
            if terms is None:
                variance += _scaled_variance(c / n, (nv / n) / n)
            else:
                variance = _source_variance([
                    *((float(w) / n, t) for w, t in zip(weights, terms, strict=True)),
                    *((-(c / n) * float(w) / n, t) for w, t in zip(weights, nterms, strict=True)),
                ])
        return MeasurementEstimate(c / n if n > 0 else math.nan, variance, support=n,
            numerator=c, numerator_variance=v, exposure=n, exposure_variance=nv,
            uncertainty="delta_method" if contract.normalizer == "uncertain" else "conditional_on_exposure", **common)

    chi2 = degrees = coverage = None
    if contract.kind == "sampled_function":
        weights, support, coverage = _interval_weights(coordinates, valid, interval)
        common["included"] = int(np.count_nonzero(weights))
        common["excluded"] = len(values) - common["included"]
        if contract.partial_support == "reject" and not math.isclose(coverage, 1.0, rel_tol=0, abs_tol=1e-14):
            return MeasurementEstimate(math.nan, math.nan, support=support, coverage_fraction=coverage, **common)
        if contract.estimator == "coordinate_mean" and support > 0:
            weights /= support
    elif contract.kind == "continuous":
        support = float(included)
        weights = valid.astype(float)
        if contract.estimator == "inverse_variance_mean":
            if np.any(valid & (variances == 0)):
                raise ValueError("Inverse-variance mean requires positive variances; zero-count errors are not Gaussian precision")
            # Scaling by the smallest variance avoids overflowing reciprocals.
            weights[valid] = np.min(variances[valid]) / variances[valid] if included else 0
        if included:
            weights /= math.fsum(map(float, weights))
    else:
        weights = np.asarray(coefficients, dtype=float)
        if weights.shape != values.shape or np.any(~np.isfinite(weights)):
            raise ValueError("Linear reconstructions require matching finite coefficients")
        weights = np.where(valid, weights, 0.0)
        support = float(included)

    value = math.fsum(float(w) * float(y) for w, y in zip(weights, values, strict=True) if w != 0) if support > 0 else math.nan
    variance = _variance(weights, variances, terms) if support > 0 else math.nan
    if contract.estimator == "inverse_variance_mean" and included:
        chi2 = math.fsum((float(y) - value)**2 / float(v)
                        for y, v, okay in zip(values, variances, valid, strict=True) if okay)
        degrees = included - 1
    return MeasurementEstimate(value, variance, support=support, coverage_fraction=coverage,
        chi2=chi2, degrees_of_freedom=degrees, **common)
