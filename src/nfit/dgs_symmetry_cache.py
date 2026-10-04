"""Reuse additive raw-DGS statistics when a requested symmetry contains a cache.

Only independent-copy HKLE statistics are additive here. Cache ownership and
complete source/configuration signatures belong to the caller; this service
receives narrow callbacks and never imports project coordinators or reducers.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
    has_event_statistics,
)
from .mdhisto import MDHistoData
from .performance import scientific_memory_limit_bytes


def symmetry_reuse_worthwhile(grid_cells: int, source_event_count: int | None, cached_operations: int) -> bool:
    """Avoid dense-cache scans when too little event work would be saved.

    Small grids have negligible validation/join costs. Larger grids require
    known source counts and at least eight saved event copies per output cell.
    This conservative workload estimate changes only the execution path.
    """
    def integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

    if not integer(grid_cells) or grid_cells <= 0 or not integer(cached_operations) or cached_operations <= 0:
        return False
    if grid_cells < 250000:
        return True
    return bool(integer(source_event_count) and source_event_count > 0
                and int(source_event_count) * int(cached_operations) >= 8 * int(grid_cells))


def symmetry_remainder(existing, requested):
    """Return requested matrices absent from an exact existing multiset.

    Preserve requested order and multiplicity. ``None`` means the cache is not
    a proper nonempty subset; an equal multiset uses normal whole-cache reuse.
    Approximate matrix equality is deliberately insufficient for event edges.
    """
    def keys(operations):
        result = []
        for operation in operations:
            matrix = np.asarray(operation, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                return None
            result.append(matrix.tobytes())
        return result

    old, target = keys(existing), keys(requested)
    if old is None or target is None or not old or len(old) >= len(target):
        return None
    remaining = Counter(old)
    if remaining - Counter(target):
        return None
    missing = []
    for operation, key in zip(requested, target, strict=True):
        if remaining[key]:
            remaining[key] -= 1
        else:
            missing.append(operation)
    return tuple(missing)


def _statistics(data):
    """Validate retained native statistics in bounded blocks, including N=0 C."""
    if not isinstance(data, MDHistoData) or not has_event_statistics(data):
        return None
    if (data.source_dependencies is not None or data.counting_dependencies is not None
            or data.metadata.get("raw_dgs", {}).get("format") != "raw-direct-geometry-nexus"
            or data.metadata.get("dgs_reduction_policies", {}).get("symmetry_variance_policy") != "independent_copies"):
        return None
    arrays = tuple(data.auxiliary_channels[name].values for name in
                   (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR))
    arrays = (*arrays, data.num_events)
    for start in range(0, data.signal.size, 250000):
        index = slice(start, start + 250000)
        numerator, variance, exposure, counts = (array.flat[index] for array in arrays)
        if (any(np.any(~np.isfinite(array)) for array in (numerator, variance, exposure, counts))
                or np.any(variance < 0) or np.any(exposure < 0) or np.any(counts < 0)
                or np.any(counts != np.floor(counts))):
            return None
        covered = exposure > 0
        if not (np.allclose(data.signal.flat[index][covered], numerator[covered] / exposure[covered], rtol=1e-12, atol=0)
                and np.allclose(data.errors.flat[index][covered], np.sqrt(variance[covered]) / exposure[covered],
                                rtol=1e-12, atol=0)):
            return None
    return arrays


def _join(base, extra, before, requested, missing, *, minimum_samples):
    if (base.shape != extra.shape or any(not np.array_equal(first.values, second.values)
                                        for first, second in zip(base.axes, extra.axes, strict=True))):
        return None
    additional = _statistics(extra)
    if additional is None:
        return None
    # Three retained numerators plus counts and the final signal/error/mask.
    # Reserve old+delta+joined volumes within half the operation RAM allowance;
    # existing reducers retain their own trajectory-worker allocation checks.
    if base.signal.size * (3 * (7 * 8 + 1)) > scientific_memory_limit_bytes() // 2:
        return None
    joined = [np.empty(base.shape) for _ in range(4)]
    signal, errors, mask = np.empty(base.shape), np.empty(base.shape), np.empty(base.shape, dtype=bool)
    for start in range(0, base.signal.size, 250000):
        index = slice(start, start + 250000)
        for output, first, second in zip(joined, before, additional, strict=True):
            output.flat[index] = first.flat[index] + second.flat[index]
        numerator, variance, exposure, counts = (array.flat[index] for array in joined)
        with np.errstate(divide="ignore", invalid="ignore"):
            signal.flat[index] = numerator / exposure
            errors.flat[index] = np.sqrt(variance) / exposure
        mask.flat[index] = (exposure <= 0) | (counts < minimum_samples)
    numerator, variance, exposure, counts = joined
    metadata = dict(extra.metadata)
    metadata["normalization_denominator"] = exposure
    metadata["symmetry_operations_hkl"] = [np.asarray(operation).tolist() for operation in requested]
    total = float(counts.sum())
    metadata["event_weight_rms"] = float(np.sqrt(variance.sum() / total)) if total else 1.0
    metadata["incremental_symmetry"] = {
        "cached_operations": len(requested) - len(missing),
        "computed_operations": len(missing),
        "variance_policy": "independent_copies",
    }
    channels = {name: channel for name, channel in extra.auxiliary_channels.items() if name != "coverage_fraction"}
    metadata.pop("coverage_fraction", None)
    for array in (*joined, signal, errors, mask):
        array.setflags(write=False)
    channels.update(event_statistics_channels(numerator, variance, exposure))
    return extra.with_updates(signal=signal, errors=errors, mask=mask, num_events=counts,
                              metadata=metadata, auxiliary_channels=channels)


def try_expand_raw_dgs_symmetry(
    config: Mapping[str, Any],
    *,
    candidates: Iterable[tuple[Mapping[str, Any], Callable[[], MDHistoData | None]]],
    signature: Callable[[Mapping[str, Any]], str],
    operations: Callable[[Mapping[str, Any]], Sequence[np.ndarray]],
    reduce_missing: Callable[[Sequence[np.ndarray]], MDHistoData],
    finalize: Callable[[MDHistoData], MDHistoData],
    minimum_samples: float,
    worthwhile: Callable[[int], bool] | None = None,
) -> MDHistoData | None:
    """Expand a current unsubtracted raw-HKLE base or request normal full replay.

    The caller yields only caches matching their *complete* current signature,
    before loading arrays. Comparing signatures with only symmetry neutralized
    additionally protects grid, geometry, UB, policies, sources, weights and
    masks. Never filter partial C/V/counts by partial exposure or display masks.
    Floating-point addition association changes; integral counts do not.
    An optional workload predicate receives the cached operation count before
    any candidate arrays are loaded; false requests ordinary full replay.
    """
    def without_symmetry(recipe):
        result = copy.deepcopy(dict(recipe))
        result.pop("symmetry", None)
        return result

    requested = tuple(operations(config))
    if len(requested) < 2:
        return None
    identity = signature(without_symmetry(config))
    subsets = []
    for candidate_config, load in candidates:
        try:
            existing = tuple(operations(candidate_config))
        except (ValueError, TypeError, ImportError):
            continue  # An unrelated unfinished recipe is not a valid cache.
        missing = symmetry_remainder(existing, requested)
        if missing is not None:
            subsets.append((candidate_config, load, existing, missing))
    # Prefer the most saved work; resolving recipes does not load any arrays.
    for candidate_config, load, existing, missing in sorted(subsets, key=lambda item: len(item[2]), reverse=True):
        if worthwhile is not None and not worthwhile(len(existing)):
            continue
        if signature(without_symmetry(candidate_config)) != identity:
            continue
        base = load()
        before = None if base is None else _statistics(base)
        if before is None:
            continue
        cached_operations = base.metadata.get("symmetry_operations_hkl", ())
        if len(cached_operations) != len(existing) or any(
            not np.array_equal(first, second) for first, second in zip(cached_operations, existing, strict=True)
        ):
            continue
        # Avoid additional reduction if the retained cache cannot fit safely.
        if base.signal.size * (3 * (7 * 8 + 1)) > scientific_memory_limit_bytes() // 2:
            continue
        extra = reduce_missing(missing)
        result = _join(base, extra, before, requested, missing, minimum_samples=minimum_samples)
        if result is not None:
            return finalize(result)
    return None
