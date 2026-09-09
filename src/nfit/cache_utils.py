from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import FunctionType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def scientific_cache_budget_bytes(
    *,
    minimum: int = 768 * 1024**2,
    maximum: int = 4 * 1024**3,
    memory_fraction: float = 1.0 / 16.0,
) -> int:
    """Return a memory-aware budget for cached scientific array payloads.

    The floor is large enough to retain one typical four-dimensional reduction,
    while the cap prevents high-memory workstations from growing an unbounded
    process cache.  This is a capacity limit, not an allocation: small projects
    retain only the arrays they actually produce.
    """

    total_memory = _total_physical_memory_bytes()
    if total_memory is None:
        return int(minimum)
    proportional = int(total_memory * float(memory_fraction))
    return min(max(proportional, int(minimum)), int(maximum))


def _total_physical_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError):
        try:
            return int(
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            )
        except (AttributeError, OSError, ValueError):
            return None


def readonly_array(value: ArrayLike, dtype: Any) -> np.ndarray:
    """Return an immutable copy of ``value`` with the requested dtype.

    Shared by the frozen scientific dataclasses so that a stored array cannot
    be mutated behind a cache key that was computed from its contents.
    """

    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def array_digest(value: ArrayLike, dtype: Any) -> str:
    """Return a shape-and-content SHA-256 digest of a numerical array.

    Used to key caches on array *contents*: the shape is hashed alongside the
    bytes so that two arrays with the same buffer but different shapes cannot
    collide.
    """

    contiguous = np.ascontiguousarray(value, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def array_payload_nbytes(value: Any) -> int:
    """Estimate the distinct numerical-array bytes retained by ``value``.

    The estimate follows containers, nfit dataclasses, and function closures.
    Repeated references to the same object are counted once. Python object
    overhead is intentionally excluded: large numerical arrays dominate these
    scientific-data caches, and their byte counts are stable across platforms.
    """

    return _array_payload_nbytes(value, seen=set())


def _array_payload_nbytes(value: Any, *, seen: set[int]) -> int:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return 0
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if hasattr(value, "__cuda_array_interface__") and hasattr(value, "nbytes"):
        # Count CuPy and other CUDA array-interface allocations without
        # importing an optional accelerator package.
        return int(value.nbytes)
    if isinstance(value, memoryview):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(
            _array_payload_nbytes(item, seen=seen) for item in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_array_payload_nbytes(item, seen=seen) for item in value)
    if isinstance(value, FunctionType):
        return sum(
            _array_payload_nbytes(cell.cell_contents, seen=seen)
            for cell in (value.__closure__ or ())
        )
    if is_dataclass(value) and not isinstance(value, type):
        # A FitDataBundle's DatasetEntry is owned by the project, not by the
        # cache. Counting it would charge the source arrays twice.
        return sum(
            _array_payload_nbytes(getattr(value, field.name), seen=seen)
            for field in fields(value)
            if not (
                type(value).__name__ == "FitDataBundle"
                and field.name == "dataset"
            )
        )
    return 0


def lru_store(
    cache: OrderedDict,
    key: Any,
    value: Any,
    limit: int,
    max_array_bytes: int | None = None,
) -> None:
    """Store one LRU entry and evict until count and array-byte limits hold."""

    cache[key] = value
    cache.move_to_end(key)
    entry_limit = max(int(limit), 0)
    byte_limit = (
        None if max_array_bytes is None else max(int(max_array_bytes), 0)
    )
    while cache and (
        len(cache) > entry_limit
        or (
            byte_limit is not None
            and array_payload_nbytes(tuple(cache.values())) > byte_limit
        )
    ):
        cache.popitem(last=False)
