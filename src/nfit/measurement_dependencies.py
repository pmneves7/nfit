"""Bounded sparse sensitivities to independent measurement primitives.

Coefficients describe fluctuations of the primary signal, not a count numerator.
Reusing a source ID preserves dependence across copies and signed arithmetic.
This is linear uncertainty propagation, not a confidence interval or likelihood.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

SOURCE_DEPENDENCIES_VERSION = 1
DEFAULT_SOURCE_DEPENDENCY_BYTES = 256 * 1024**2


class SourceReplayRequired(ValueError):
    """A source-aware operation requires unavailable or over-budget payloads."""


def _immutable(array, dtype):
    result = np.asarray(array, dtype=dtype)
    if result.flags.writeable:
        result = result.copy()
    result.setflags(write=False)
    return result


def _budget(entries, sources, source_ids, max_bytes, observations=0):
    # Reserve sorting, grouping and output working arrays as well as payload.
    estimated = int(entries) * 80 + int(sources) * 16 + int(observations)*16 + sum(len(s.encode("utf-8")) + 64 for s in source_ids)
    if estimated > int(max_bytes):
        raise SourceReplayRequired("Source sensitivities exceed the storage/work budget; replay the sources at the final requested bins")


@dataclass(frozen=True, kw_only=True)
class SourceDependencies:
    """Immutable COO sensitivities for flattened observations.

    ``variance[i] = sum_s coefficient[i,s]**2 * source_variance[s]``.
    Duplicate observation/source entries are added before squaring. Source IDs
    name independent primitives; the same ID must retain the same variance.
    Arrays loaded read-only from mapped archives retain their backing storage.
    """

    shape: tuple[int, ...]
    source_ids: tuple[str, ...]
    source_variances: np.ndarray
    observation_indices: np.ndarray
    source_indices: np.ndarray
    coefficients: np.ndarray
    max_bytes: int = DEFAULT_SOURCE_DEPENDENCY_BYTES

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Source dependency shape must contain nonnegative integer dimensions")
        ids = tuple(self.source_ids)
        if any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("Source IDs must be unique nonempty strings")
        variance = np.asarray(self.source_variances, dtype=float)
        observations = np.asarray(self.observation_indices)
        sources = np.asarray(self.source_indices)
        coefficient = np.asarray(self.coefficients, dtype=float)
        if variance.shape != (len(ids),) or np.any(~np.isfinite(variance) | (variance < 0)):
            raise ValueError("Source variances must be finite, nonnegative and match source IDs")
        if observations.ndim != 1 or sources.shape != observations.shape or coefficient.shape != observations.shape:
            raise ValueError("Source sensitivity arrays must have matching one-dimensional shapes")
        if not np.issubdtype(observations.dtype, np.integer) or not np.issubdtype(sources.dtype, np.integer):
            raise ValueError("Source sensitivity indices must be integers")
        if np.any((observations < 0) | (observations >= math.prod(shape))) or np.any((sources < 0) | (sources >= len(ids))):
            raise ValueError("Source sensitivity index is outside its declared shape")
        if np.any(~np.isfinite(coefficient)):
            raise ValueError("Source coefficients must be finite")
        _budget(len(coefficient), len(ids), ids, self.max_bytes, math.prod(shape))
        # Canonical archives avoid sorting/copying their mapped arrays.
        ordered = len(observations) < 2 or np.all(
            (observations[1:] > observations[:-1])
            | ((observations[1:] == observations[:-1]) & (sources[1:] > sources[:-1])))
        if not ordered or np.any(coefficient == 0):
            order = np.lexsort((sources, observations))
            observations, sources, coefficient = observations[order], sources[order], coefficient[order]
            first = np.r_[True, (observations[1:] != observations[:-1]) | (sources[1:] != sources[:-1])]
            starts = np.flatnonzero(first)
            coefficient = np.add.reduceat(coefficient, starts) if len(starts) else coefficient
            observations, sources = observations[starts], sources[starts]
            keep = coefficient != 0
            observations, sources, coefficient = observations[keep], sources[keep], coefficient[keep]
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "source_ids", ids)
        for name, array, dtype in (("source_variances", variance, float), ("observation_indices", observations, np.int64),
                                   ("source_indices", sources, np.int64), ("coefficients", coefficient, float)):
            object.__setattr__(self, name, _immutable(array, dtype))

    def variances(self):
        """Return represented primary-signal diagonal variances."""
        terms = self.coefficients * (self.coefficients * self.source_variances[self.source_indices])
        if np.any(~np.isfinite(terms)):
            raise ValueError("Source variance propagation overflowed")
        return np.bincount(self.observation_indices, weights=terms, minlength=math.prod(self.shape)).reshape(self.shape)

    def __deepcopy__(self, memo):
        """Frozen factors can be shared without copying mapped array storage."""
        memo[id(self)] = self
        return self

    def variance(self):
        """Return represented variances; singular alias for reduction services."""
        return self.variances()

    def validate_variances(self, variances, *, mask=None):
        """Require agreement with finite supplied variances on selected cells."""
        variance = np.asarray(variances, dtype=float)
        if variance.shape != self.shape:
            raise ValueError("Source dependencies must match the observation shape")
        valid = np.isfinite(variance)
        if mask is not None:
            valid &= ~np.broadcast_to(np.asarray(mask, dtype=bool), self.shape)
        if not np.allclose(self.variances()[valid], variance[valid], rtol=1e-10, atol=0):
            raise ValueError("Source dependencies do not reproduce the observation variances")


@dataclass(frozen=True, kw_only=True)
class CountingDependencies:
    """Primitive sensitivities of count numerator C and uncertain exposure N.

    Shared IDs preserve numerator/exposure covariance. Ratio propagation is a
    first-order delta approximation; it is not a low-count confidence interval.
    """

    numerator_dependencies: SourceDependencies
    exposure_dependencies: SourceDependencies

    def __post_init__(self):
        if not isinstance(self.numerator_dependencies, SourceDependencies) or not isinstance(self.exposure_dependencies, SourceDependencies):
            raise TypeError("Counting dependencies require numerator and exposure SourceDependencies")
        if self.numerator_dependencies.shape != self.exposure_dependencies.shape:
            raise ValueError("Numerator and exposure dependency shapes must agree")
        payloads = (self.numerator_dependencies, self.exposure_dependencies)
        _budget(sum(len(p.coefficients) for p in payloads), sum(len(p.source_ids) for p in payloads),
            tuple(s for p in payloads for s in p.source_ids), min(p.max_bytes for p in payloads), math.prod(self.shape))

    @property
    def shape(self):
        return self.numerator_dependencies.shape


def ratio_source_dependencies(dependencies, numerator, exposure):
    """Propagate C/N using dI=dC/N-C*dN/N² with shared primitive identities."""
    if not isinstance(dependencies, CountingDependencies):
        raise SourceReplayRequired("Uncertain exposure requires numerator/exposure source sensitivities")
    c, n = np.asarray(numerator, float), np.asarray(exposure, float)
    if c.shape != dependencies.shape or n.shape != dependencies.shape:
        raise ValueError("Count numerator and exposure must match their dependency shape")
    valid = np.isfinite(c) & np.isfinite(n) & (n > 0)
    indices = np.flatnonzero(valid)
    c, n = c.ravel()[indices], n.ravel()[indices]
    numerator_payload = project_source_dependencies(dependencies.numerator_dependencies, indices, indices,
        1/n, dependencies.shape)
    exposure_payload = project_source_dependencies(dependencies.exposure_dependencies, indices, indices,
        -(c/n)/n, dependencies.shape)
    return combine_source_dependencies((numerator_payload, exposure_payload), (1, 1),
        max_bytes=min(dependencies.numerator_dependencies.max_bytes, dependencies.exposure_dependencies.max_bytes))


def select_source_dependencies(dependencies, selection):
    """Select array observations while preserving their primitive identities."""
    indices = np.arange(math.prod(dependencies.shape), dtype=np.int64).reshape(dependencies.shape)[selection]
    return project_source_dependencies(dependencies, indices.ravel(), np.arange(indices.size),
        np.ones(indices.size), indices.shape)


def select_counting_dependencies(dependencies, selection):
    """Select corresponding numerator and exposure sensitivities."""
    return CountingDependencies(numerator_dependencies=select_source_dependencies(dependencies.numerator_dependencies, selection),
        exposure_dependencies=select_source_dependencies(dependencies.exposure_dependencies, selection))


def independent_source_dependencies(variances, source_id, *, max_bytes=DEFAULT_SOURCE_DEPENDENCY_BYTES):
    """Declare independent observations in an explicit, stable namespace.

    Reuse the namespace only when the observations are the same measurements.
    Missing cells must be removed or represented by zero variance and a mask.
    """
    variance = np.asarray(variances, dtype=float)
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Independent sources require a nonempty namespace")
    # Check before allocating a potentially large tuple of source labels.
    if variance.size * (len(source_id.encode("utf-8")) + 128) > max_bytes:
        raise SourceReplayRequired("Independent source payload exceeds its budget; replay the source")
    indices = np.arange(variance.size, dtype=np.int64)
    return SourceDependencies(shape=variance.shape, source_ids=tuple(f"{source_id}:{i}" for i in indices),
        source_variances=variance.ravel(), observation_indices=indices, source_indices=indices,
        coefficients=np.ones(variance.size), max_bytes=max_bytes)


def project_source_dependencies(dependencies, input_indices, output_indices, coefficients, output_shape, *, max_bytes=None):
    """Apply sparse linear assignments to observations without dense covariance.

    One input may occur repeatedly, for fractional assignments or copies. Omit
    masked inputs from the assignments. Coefficients must include the complete
    scientific weighting/normalization of the requested output signal.
    """
    if not isinstance(dependencies, SourceDependencies):
        raise SourceReplayRequired("Source-aware projection requires tracked sensitivities or source replay")
    inputs, outputs, weights = np.asarray(input_indices), np.asarray(output_indices), np.asarray(coefficients, dtype=float)
    shape = tuple(output_shape)
    if inputs.ndim != 1 or outputs.shape != inputs.shape or weights.shape != inputs.shape:
        raise ValueError("Projection assignments must be matching one-dimensional arrays")
    if not np.issubdtype(inputs.dtype, np.integer) or not np.issubdtype(outputs.dtype, np.integer):
        raise ValueError("Projection indices must be integers")
    if np.any((inputs < 0) | (inputs >= math.prod(dependencies.shape))) or np.any((outputs < 0) | (outputs >= math.prod(shape))) or np.any(~np.isfinite(weights)):
        raise ValueError("Invalid source projection assignment")
    limit = dependencies.max_bytes if max_bytes is None else max_bytes
    starts = np.searchsorted(dependencies.observation_indices, inputs, side="left")
    stops = np.searchsorted(dependencies.observation_indices, inputs, side="right")
    counts = stops - starts
    total = int(np.sum(counts))
    _budget(total, len(dependencies.source_ids), dependencies.source_ids, limit, math.prod(shape))
    if total:
        owners = np.repeat(np.arange(len(inputs)), counts)
        positions = starts[owners] + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        out, src = outputs[owners], dependencies.source_indices[positions]
        coef = weights[owners] * dependencies.coefficients[positions]
    else:
        out, src, coef = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    return SourceDependencies(shape=shape, source_ids=dependencies.source_ids,
        source_variances=dependencies.source_variances, observation_indices=out, source_indices=src,
        coefficients=coef, max_bytes=limit)


def combine_source_dependencies(dependencies, coefficients, *, max_bytes=DEFAULT_SOURCE_DEPENDENCY_BYTES):
    """Combine aligned signals, preserving signed reuse of shared primitives."""
    payloads, scales = tuple(dependencies), tuple(coefficients)
    if not payloads or len(payloads) != len(scales) or any(not isinstance(p, SourceDependencies) for p in payloads):
        raise SourceReplayRequired("Arithmetic requires tracked dependencies for every operand")
    shape = payloads[0].shape
    if any(p.shape != shape for p in payloads) or any(not np.isfinite(s) for s in scales):
        raise ValueError("Dependency arithmetic requires aligned shapes and finite scales")
    variances, ids, remaps = {}, [], []
    for payload in payloads:
        remap = []
        for name, variance in zip(payload.source_ids, payload.source_variances, strict=True):
            if name in variances:
                index, previous = variances[name]
                if previous != variance:
                    raise ValueError(f"Inconsistent variance for shared source {name!r}")
            else:
                index = len(ids)
                ids.append(name)
                variances[name] = index, float(variance)
            remap.append(index)
        remaps.append(np.asarray(remap, dtype=np.int64))
    _budget(sum(len(p.coefficients) for p in payloads), len(ids), ids, max_bytes, math.prod(shape))
    return SourceDependencies(shape=shape, source_ids=tuple(ids),
        source_variances=np.array([variances[name][1] for name in ids]),
        observation_indices=np.concatenate([p.observation_indices for p in payloads]),
        source_indices=np.concatenate([remap[p.source_indices] for p, remap in zip(payloads, remaps, strict=True)]),
        coefficients=np.concatenate([p.coefficients * s for p, s in zip(payloads, scales, strict=True)]), max_bytes=max_bytes)


def source_dependency_archive_payload(dependencies, *, prefix="source_dependencies_"):
    """Return versioned array members shared by dataset and project codecs."""
    if dependencies is None:
        return {}
    return {f"{prefix}version": np.asarray(SOURCE_DEPENDENCIES_VERSION),
            f"{prefix}shape": np.asarray(dependencies.shape),
            f"{prefix}max_bytes": np.asarray(dependencies.max_bytes),
            f"{prefix}ids": np.asarray(json.dumps(dependencies.source_ids)),
            **{f"{prefix}{name}": getattr(dependencies, name) for name in
               ("source_variances", "observation_indices", "source_indices", "coefficients")}}


def restore_source_dependencies(archive, *, prefix="source_dependencies_"):
    """Restore optional canonical sensitivities, retaining read-only mappings."""
    if f"{prefix}version" not in archive:
        return None
    if int(np.asarray(archive[f"{prefix}version"]).item()) != SOURCE_DEPENDENCIES_VERSION:
        raise ValueError("Unsupported source dependency archive version")
    return SourceDependencies(shape=tuple(int(x) for x in archive[f"{prefix}shape"]),
        source_ids=tuple(json.loads(str(np.asarray(archive[f"{prefix}ids"]).item()))),
        max_bytes=int(np.asarray(archive[f"{prefix}max_bytes"]).item()) if f"{prefix}max_bytes" in archive else DEFAULT_SOURCE_DEPENDENCY_BYTES,
        **{name: archive[f"{prefix}{name}"] for name in
           ("source_variances", "observation_indices", "source_indices", "coefficients")})


def counting_dependency_archive_payload(dependencies):
    """Encode optional numerator/exposure sensitivities as separate arrays."""
    if dependencies is None:
        return {}
    return {"counting_dependencies_version": np.asarray(SOURCE_DEPENDENCIES_VERSION),
        **source_dependency_archive_payload(dependencies.numerator_dependencies, prefix="counting_numerator_dependencies_"),
        **source_dependency_archive_payload(dependencies.exposure_dependencies, prefix="counting_exposure_dependencies_")}


def restore_counting_dependencies(archive):
    """Decode a complete optional uncertain-exposure dependency bundle."""
    if "counting_dependencies_version" not in archive:
        return None
    if int(np.asarray(archive["counting_dependencies_version"]).item()) != SOURCE_DEPENDENCIES_VERSION:
        raise ValueError("Unsupported counting dependency archive version")
    return CountingDependencies(numerator_dependencies=restore_source_dependencies(archive, prefix="counting_numerator_dependencies_"),
        exposure_dependencies=restore_source_dependencies(archive, prefix="counting_exposure_dependencies_"))
