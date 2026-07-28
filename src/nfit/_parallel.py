"""Shared CPU-allocation policy for local accelerated backends."""

from __future__ import annotations

import os

_THREAD_OVERRIDE: int | None = None


def detect_cpu_budget() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def set_num_threads(n: int | None) -> None:
    global _THREAD_OVERRIDE
    _THREAD_OVERRIDE = None if n is None else max(1, int(n))


def num_threads() -> int:
    if _THREAD_OVERRIDE is not None:
        return _THREAD_OVERRIDE
    value = os.environ.get("NFIT_NUM_THREADS")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return detect_cpu_budget()
