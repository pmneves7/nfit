"""Shared CPU-allocation policy for local accelerated backends."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

_THREAD_OVERRIDE: int | None = None
_SCOPED_THREADS: ContextVar[int | None] = ContextVar("nfit_threads", default=None)


@contextmanager
def thread_budget(workers: int | None):
    """Temporarily select a worker ceiling for this execution context only."""
    token = _SCOPED_THREADS.set(None if not workers else max(1, int(workers)))
    try:
        yield
    finally:
        _SCOPED_THREADS.reset(token)


def detect_cpu_budget() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def set_num_threads(n: int | None) -> None:
    global _THREAD_OVERRIDE
    _THREAD_OVERRIDE = None if n is None else max(1, int(n))


def num_threads() -> int:
    scoped = _SCOPED_THREADS.get()
    if scoped is not None:
        return scoped
    if _THREAD_OVERRIDE is not None:
        return _THREAD_OVERRIDE
    value = os.environ.get("NFIT_NUM_THREADS")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return detect_cpu_budget()
