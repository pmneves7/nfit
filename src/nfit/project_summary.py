"""Bounded, identity-based summaries for immutable project data containers."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from threading import RLock

import numpy as np

from .mdhisto import MDHistoData, mdhisto_measured_bins_from_arrays

_COUNT_CHUNK_SIZE = 1_000_000


@dataclass
class _MDHistoCounts:
    unmasked: int | None = None
    fit_eligible: tuple[bool, weakref.ReferenceType[np.ndarray] | None, int] | None = None


_COUNTS: dict[int, tuple[weakref.ReferenceType[MDHistoData], _MDHistoCounts]] = {}
_LOCK = RLock()


def _counts_for(data: MDHistoData) -> _MDHistoCounts:
    if data._arrays_mutable:
        return _MDHistoCounts()
    key = id(data)
    with _LOCK:
        cached = _COUNTS.get(key)
        if cached is not None and cached[0]() is data:
            return cached[1]

        counts = _MDHistoCounts()

        def discard(reference: weakref.ReferenceType[MDHistoData]) -> None:
            with _LOCK:
                current = _COUNTS.get(key)
                if current is not None and current[0] is reference:
                    _COUNTS.pop(key, None)

        reference = weakref.ref(data, discard)
        _COUNTS[key] = (reference, counts)
        return counts


def mdhisto_mask_counts(data: MDHistoData) -> tuple[int, int]:
    """Return exact unmasked/masked counts without retaining ``data``."""

    counts = _counts_for(data)
    if counts.unmasked is None:
        unmasked = 0
        for start in range(0, data.mask.size, _COUNT_CHUNK_SIZE):
            stop = min(start + _COUNT_CHUNK_SIZE, data.mask.size)
            unmasked += int(np.count_nonzero(~data.mask.flat[start:stop]))
        counts.unmasked = unmasked
    return counts.unmasked, int(data.mask.size) - counts.unmasked


def mdhisto_fit_bin_count(data: MDHistoData) -> int:
    """Return the exact fit-eligible count using bounded temporary arrays."""

    zero_events_measured = bool(data.metadata.get("zero_event_bins_are_measured", False))
    coverage = data.metadata.get("normalization_denominator")
    has_coverage = isinstance(coverage, np.ndarray) and coverage.shape == data.shape
    coverage_array = coverage if has_coverage else None
    cacheable = not data._arrays_mutable and not (
        coverage_array is not None and coverage_array.flags.writeable
    )
    counts = _counts_for(data)
    cached = counts.fit_eligible
    if cacheable and cached is not None and cached[0] == zero_events_measured:
        same_coverage = (
            cached[1] is None if coverage_array is None
            else cached[1] is not None and cached[1]() is coverage_array
        )
        if same_coverage:
            return cached[2]

    fit_eligible = 0
    for start in range(0, data.mask.size, _COUNT_CHUNK_SIZE):
        stop = min(start + _COUNT_CHUNK_SIZE, data.mask.size)
        keep = mdhisto_measured_bins_from_arrays(
            data.mask.flat[start:stop],
            data.num_events.flat[start:stop],
            zero_event_bins_are_measured=zero_events_measured,
            normalization_denominator=(
                coverage_array.flat[start:stop] if coverage_array is not None else None
            ),
        )
        keep &= np.isfinite(data.signal.flat[start:stop])
        selected_errors = data.errors.flat[start:stop]
        keep &= np.isfinite(selected_errors) & (selected_errors > 0.0)
        fit_eligible += int(np.count_nonzero(keep))
    if cacheable:
        counts.fit_eligible = (
            zero_events_measured,
            weakref.ref(coverage_array) if coverage_array is not None else None,
            fit_eligible,
        )
    return fit_eligible


def _summary_cache_size() -> int:
    """Return live cache entries for focused lifecycle tests."""

    with _LOCK:
        return len(_COUNTS)
