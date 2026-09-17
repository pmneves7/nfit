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
        requested = scoped
    elif _THREAD_OVERRIDE is not None:
        requested = _THREAD_OVERRIDE
    else:
        value = os.environ.get("NFIT_NUM_THREADS")
        if value:
            try:
                requested = max(1, int(value))
            except ValueError:
                requested = detect_cpu_budget()
        else:
            requested = detect_cpu_budget()

    # Keep the preference storage independent from the allocation policy: the
    # local import avoids a module import cycle, while scoped task budgets still
    # obey the user's application-wide CPU ceiling.
    from .performance import load_resource_limits

    # An affinity/cgroup allocation is a hard upper bound even when an
    # environment variable, saved setting, or task-specific context asks for
    # more.  The preference can only make that allocation smaller.
    requested = min(requested, detect_cpu_budget())
    limit = load_resource_limits()["cpu_limit"]
    return max(1, min(requested, limit)) if limit else requested


def bounded_worker_count(workers: int | None) -> int:
    """Resolve a saved worker request without exceeding the shared ceiling."""

    available = num_threads()
    if workers is None or int(workers) == 0:
        return available
    return min(available, max(1, int(workers)))
