"""Process-wide admission and reservations for scientific allocations.

Reservations prevent two concurrent jobs from independently spending the same
remaining RAM. They are estimates, rather than an operating-system memory cap:
Python and third-party libraries can allocate outside these entry points.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import BoundArguments, signature
from threading import RLock, get_ident

from . import performance


class ResourceLimitError(MemoryError):
    """An operation would exceed the configured RAM or available system RAM."""

    def __init__(self, *, requested_bytes: int, used_bytes: int, limit_bytes: int,
                 operation: str, available_bytes: int | None = None) -> None:
        self.requested_bytes = int(requested_bytes)
        self.used_bytes = int(used_bytes)
        self.limit_bytes = int(limit_bytes)
        self.operation = str(operation)
        self.available_bytes = available_bytes
        self.missing_bytes = max(0, self.used_bytes + self.requested_bytes - self.limit_bytes)
        super().__init__(
            f"{self.operation} requires {self.requested_bytes / 1e9:.3f} GB of additional RAM. "
            f"{self.used_bytes / 1e9:.3f} GB is in use or reserved and the limit is "
            f"{self.limit_bytes / 1e9:.3f} GB. Increase the RAM budget or remove "
            "objects from active memory in Resource Manager."
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """Current process RAM, outstanding reservations, and admission limits."""

    process_bytes: int
    reserved_bytes: int
    limit_bytes: int
    available_bytes: int | None
    managed_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        return max(self.process_bytes, self.managed_bytes) + self.reserved_bytes

    @property
    def remaining_bytes(self) -> int:
        remaining = max(0, self.limit_bytes - self.used_bytes)
        if self.available_bytes is not None:
            remaining = min(remaining, max(0, self.available_bytes - self.reserved_bytes))
        return remaining


def _process_rss() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        # The peak is conservative when a current RSS counter is unavailable.
        return int(performance.peak_process_memory_mib() * 1024**2)


def _configured_limit() -> int:
    fixed = performance.load_resource_limits()["ram_limit_mb"] * 1024**2
    return fixed or performance.scientific_memory_limit_bytes()


def _managed_usage() -> int:
    # Count cache storage even when zero-filled arrays have not yet faulted
    # their full pages into RSS. Avoid importing a cache during its startup.
    module = sys.modules.get("nfit.rebin_cache")
    budget = getattr(module, "SHARED_REBIN_CACHE_BUDGET", None)
    return 0 if budget is None else int(budget.total_bytes())


_lock = RLock()
_pending: dict[int, int] = {}
_rss_provider: Callable[[], int] = _process_rss
_limit_provider: Callable[[], int] = _configured_limit
_available_provider: Callable[[], int | None] = performance.available_memory_bytes
_managed_provider: Callable[[], int] = _managed_usage


def configure_memory_providers(*, rss: Callable[[], int] | None = None,
                               limit: Callable[[], int] | None = None,
                               available: Callable[[], int | None] | None = None,
                               managed: Callable[[], int] | None = None) -> None:
    """Configure narrow measurement callbacks; omitted callbacks use defaults."""

    global _rss_provider, _limit_provider, _available_provider, _managed_provider
    with _lock:
        _rss_provider = rss or _process_rss
        _limit_provider = limit or _configured_limit
        _available_provider = available or performance.available_memory_bytes
        _managed_provider = managed or _managed_usage


def snapshot_memory() -> MemorySnapshot:
    """Read process-wide usage without loading any project data."""

    with _lock:
        available = _available_provider()
        return MemorySnapshot(
            max(0, int(_rss_provider())), sum(_pending.values()),
            max(1, int(_limit_provider())),
            None if available is None else max(0, int(available)),
            max(0, int(_managed_provider())),
        )


@dataclass
class _Reservation:
    thread_id: int
    leases: dict[object, int] = field(default_factory=dict)

    @property
    def amount(self) -> int:
        return max(self.leases.values(), default=0)


_active: ContextVar[_Reservation | None] = ContextVar("nfit_memory_reservation", default=None)


def _current_reservation() -> _Reservation | None:
    active = _active.get()
    with _lock:
        if active is None or active.thread_id != get_ident() or id(active) not in _pending:
            return None
        return active


def _check_locked(byte_count: int, operation: str) -> MemorySnapshot:
    snapshot = snapshot_memory()
    if byte_count > snapshot.remaining_bytes:
        effective_limit = snapshot.limit_bytes
        if snapshot.available_bytes is not None:
            effective_limit = min(effective_limit, max(snapshot.process_bytes, snapshot.managed_bytes) + snapshot.available_bytes)
        raise ResourceLimitError(
            requested_bytes=byte_count, used_bytes=snapshot.used_bytes,
            limit_bytes=effective_limit, available_bytes=snapshot.available_bytes,
            operation=operation,
        )
    return snapshot


def check_memory(byte_count: int, *, operation: str = "Loading data",
                 retained_bytes: int = 0) -> MemorySnapshot:
    """Check an additional allocation, allowing coverage by an outer estimate.

    ``retained_bytes`` adds allocations not yet resident, such as untouched
    output buffers. It must not include already-resident process RAM.
    """

    amount = max(0, int(byte_count)) + max(0, int(retained_bytes))
    active = _current_reservation()
    with _lock:
        return _check_locked(max(0, amount - (active.amount if active else 0)), operation)


@contextmanager
def reserve_memory(byte_count: int, *, operation: str = "Loading data",
                   retained_bytes: int = 0) -> Iterator[None]:
    """Reserve an operation's additional peak RAM before allocating arrays.

    Nested calls share the outer reservation. A larger nested peak reserves
    only its excess, and concurrent threads retain independent reservations.
    Reservations are always released when an operation fails or is cancelled.
    """

    amount = max(0, int(byte_count)) + max(0, int(retained_bytes))
    active = _current_reservation()
    own = active is None
    reservation = _Reservation(get_ident()) if own else active
    assert reservation is not None
    lease = object()
    token = _active.set(reservation) if own else None
    try:
        with _lock:
            old_amount = reservation.amount
            increment = max(0, amount - old_amount)
            if increment:
                _check_locked(increment, operation)
            reservation.leases[lease] = amount
            _pending[id(reservation)] = reservation.amount
        yield
    finally:
        with _lock:
            reservation.leases.pop(lease, None)
            # Generators can suspend a nested scope beyond its parent and
            # close scopes out of order. Only live leases contribute; restoring
            # a saved parent amount would revive a released reservation.
            if reservation.leases:
                _pending[id(reservation)] = reservation.amount
            else:
                _pending.pop(id(reservation), None)
        if token is not None:
            _active.reset(token)


def memory_guard(estimate: Callable[[BoundArguments], int], *, operation: str):
    """Guard a public numerical entry point with a domain-owned peak estimate.

    The estimator receives bound, defaulted arguments and may materialize
    iterator parameters by replacing them in ``arguments.arguments``. The
    wrapped function retains its public signature and scripting entry point.
    """

    def decorate(function):
        public_signature = signature(function)

        @wraps(function)
        def guarded(*args, **kwargs):
            arguments = public_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            required = max(0, int(estimate(arguments)))
            with reserve_memory(required, operation=operation):
                return function(*arguments.args, **arguments.kwargs)

        return guarded

    return decorate
