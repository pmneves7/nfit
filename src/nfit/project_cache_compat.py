"""Fail-closed compatibility checks for serialized numerical cache identities."""

from __future__ import annotations

import json
import math
from typing import Any

COMPOSITE_CACHE_SIGNATURE_TAG = "event-scales-and-measured-background-replay-v2"
LEGACY_COMPOSITE_CACHE_SIGNATURE_TAG = "event-user-masks-and-background-windows-v1"


def composite_cache_signatures_match(saved: str | None, current: str) -> bool:
    """Return whether a narrowly supported v1 event cache is valid as v2.

    Exact matches are accepted.  Legacy acceptance is limited to the known
    eleven-field native-MDEvent signature and recursively applies the same
    rules to child composites and group-backed backgrounds.  Unknown or
    malformed payloads fail closed.
    """

    if saved == current:
        return True
    if not isinstance(saved, str) or not isinstance(current, str):
        return False
    try:
        previous = json.loads(saved)
        present = json.loads(current)
    except (TypeError, ValueError):
        return False
    return _legacy_composite_payload_matches(previous, present)


def _legacy_composite_payload_matches(previous: Any, present: Any) -> bool:
    if not (
        isinstance(previous, list)
        and isinstance(present, list)
        and len(previous) == len(present) == 11
        and previous[0] == LEGACY_COMPOSITE_CACHE_SIGNATURE_TAG
        and present[0] == COMPOSITE_CACHE_SIGNATURE_TAG
    ):
        return False
    if not _safe_legacy_event_payload(present):
        return False

    # Recursively compare child and group-background signatures; every other
    # scientific field must remain byte-for-byte equivalent after JSON decode.
    for index in (1, 2, 3, 4, 5, 6, 8, 9):
        if previous[index] != present[index]:
            return False
    if not _signature_pair_lists_match(previous[7], present[7]):
        return False
    return _background_lists_match(previous[10], present[10])


def _safe_legacy_event_payload(payload: list[Any]) -> bool:
    event_config = payload[1]
    if not isinstance(event_config, dict):
        return False
    try:
        numerical = json.loads(payload[4])
    except (TypeError, ValueError):
        return False
    if not isinstance(numerical, dict):
        return False
    powder = numerical.get("coordinate_mode") == "powder"
    dimensions = event_config.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) < 3:
        return False
    for dimension in dimensions[:3]:
        if not isinstance(dimension, dict):
            return False
        frame = dimension.get("frame")
        if (
            not powder
            and frame not in (None, "")
            and str(frame).casefold() != "qsample"
        ):
            return False
    axes = numerical.get("axes")
    if not isinstance(axes, list) or not axes:
        return False
    for axis in axes:
        if not isinstance(axis, dict) or axis.get("auto_lower") or axis.get("auto_upper"):
            return False
        try:
            bounds = (float(axis["lower"]), float(axis["upper"]))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in bounds):
            return False
    datasets = payload[8]
    if not isinstance(datasets, list) or not datasets:
        return False
    for dataset in datasets:
        if not isinstance(dataset, list) or len(dataset) != 10:
            return False
        # Enabled candidates were the only entries serialized by the producer;
        # nevertheless require that invariant rather than relying on it.
        if dataset[4] is not True or dataset[5] != 1.0 or dataset[6] != 1.0:
            return False
    return True


def _signature_pair_lists_match(previous: Any, present: Any) -> bool:
    if not isinstance(previous, list) or not isinstance(present, list):
        return False
    if len(previous) != len(present):
        return False
    for old_pair, new_pair in zip(previous, present, strict=True):
        if not (
            isinstance(old_pair, list) and isinstance(new_pair, list)
            and len(old_pair) == len(new_pair) == 2
            and old_pair[0] == new_pair[0]
            and isinstance(old_pair[1], str) and isinstance(new_pair[1], str)
            and composite_cache_signatures_match(old_pair[1], new_pair[1])
        ):
            return False
    return True


def _background_lists_match(previous: Any, present: Any) -> bool:
    if not isinstance(previous, list) or not isinstance(present, list):
        return False
    if len(previous) != len(present):
        return False
    for old_background, new_background in zip(previous, present, strict=True):
        if not (
            isinstance(old_background, list) and isinstance(new_background, list)
            and len(old_background) == len(new_background) == 8
        ):
            return False
        # Direct dataset backgrounds changed scale semantics in v2.  Reject
        # them conservatively; group backgrounds carry a recursively checked
        # composite signature in their final field.
        if old_background[0] is not None or new_background[0] is not None:
            return False
        if old_background[1] is None or new_background[1] is None:
            return False
        if old_background[5] == "measured_events" or new_background[5] == "measured_events":
            return False
        if old_background[:7] != new_background[:7]:
            return False
        if not (
            isinstance(old_background[7], str)
            and isinstance(new_background[7], str)
            and composite_cache_signatures_match(old_background[7], new_background[7])
        ):
            return False
    return True
