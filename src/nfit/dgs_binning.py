"""Shared requested-grid planning for neutron event histogram reducers.

Grid validation and symmetry matrix handling are independent of source format.
The memory planner receives the reducer's estimator to preserve its resource
policy without importing a source adapter.
"""

from __future__ import annotations

import math

import numpy as np


def dgs_grid_memory(arguments, estimate_peak_memory):
    """Resolve requested counts without allocating output-sized edge arrays."""

    values = arguments.arguments
    for name in ("lower", "upper", "num_bins", "step_size"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    if values.get("bin_edges") is not None:
        values["bin_edges"] = tuple(
            None if edge is None else edge if isinstance(edge, np.ndarray) else tuple(edge)
            for edge in values["bin_edges"]
        )
    counts = list(values["num_bins"])
    steps = values.get("step_size")
    explicit = values.get("bin_edges")
    if steps is not None and len(steps) == len(counts):
        for index, (lower, upper, step) in enumerate(zip(values["lower"], values["upper"], steps, strict=True)):
            if np.isfinite([lower, upper, step]).all() and step > 0 and upper >= lower:
                counts[index] = int(math.ceil((upper - lower) / step)) + 1
    if explicit is not None and len(explicit) == len(counts):
        counts = [len(edge) - 1 if edge is not None else count for edge, count in zip(explicit, counts, strict=True)]
    if any(int(count) < 1 for count in counts):
        return 0  # The numerical entry point supplies its ordinary validation.
    return estimate_peak_memory(counts, max_batch_bytes=values["max_batch_bytes"])


def _validated_minimum_samples(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("minimum_samples must be finite and nonnegative")
    return result


def _requested_edges(lower, upper, num_bins, step_size=None, *, bin_edges=None):
    from .rebin import _uniform_center_edges

    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper < lower):
        raise ValueError("binning bounds must be finite and increasing")
    if np.any(num_bins < 1):
        raise ValueError("bin counts must be positive")
    if step_size is None:
        uniform_edges = [
            _uniform_center_edges(lo, hi, count=count)
            for lo, hi, count in zip(lower, upper, num_bins, strict=True)
        ]
    else:
        steps = np.asarray(tuple(step_size), dtype=float)
        if steps.shape != lower.shape or np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
            raise ValueError("step sizes must be positive and match the requested dimensions")
        uniform_edges = []
        for lo, hi, step in zip(lower, upper, steps, strict=True):
            uniform_edges.append(_uniform_center_edges(lo, hi, step=step))
    if bin_edges is None:
        return uniform_edges
    explicit = list(bin_edges)
    if len(explicit) != lower.size:
        raise ValueError("bin_edges must contain one entry per requested dimension")
    result = []
    for fallback, values in zip(uniform_edges, explicit, strict=True):
        if values is None:
            result.append(fallback)
            continue
        axis_edges = np.asarray(tuple(values), dtype=float)
        if (
            axis_edges.ndim != 1
            or axis_edges.size < 2
            or np.any(~np.isfinite(axis_edges))
            or np.any(np.diff(axis_edges) <= 0.0)
        ):
            raise ValueError("explicit bin edges must be finite and strictly increasing")
        result.append(axis_edges)
    return result


def _symmetry_matrices(operations) -> tuple[np.ndarray, ...]:
    if operations is None:
        return (np.eye(3),)
    matrices = tuple(np.asarray(operation, dtype=float) for operation in operations)
    if not matrices:
        return (np.eye(3),)
    if any(matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("symmetry operations must be finite 3x3 HKL matrices")
    return matrices
