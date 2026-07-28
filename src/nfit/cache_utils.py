from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import FunctionType
from typing import Any

import numpy as np


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
