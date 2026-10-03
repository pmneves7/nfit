"""Detector mask metadata retained in Mantid MDEvent experiment records.

Mantid serializes masked detector IDs in its instrument parameter map, rather
than the physical-detector angle arrays. This reader interprets only boolean
``detID`` mask records; it executes no expressions and reads no event arrays.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_MASK_CACHE_MAX_TEXT_BYTES = 1 << 20
_MASK_CACHE_MAX_ENTRIES = 16


def saved_detector_mask_ids(experiment):
    """Read effective masked detector IDs as a sorted immutable int64 array.

    Absent parameter maps mean no saved mask. Repeated records use the final
    serialized boolean value; explicit false/zero records leave IDs unmasked.
    Malformed mask records fail rather than silently granting exposure.
    """
    data = experiment.get("instrument/instrument_parameter_map/data")
    if data is None:
        result = np.empty(0, dtype=np.int64)
        result.setflags(write=False)
        return result
    values = np.asarray(data[()])
    if values.dtype.kind == "u" and values.dtype.itemsize == 1:
        text = values.tobytes().decode("utf-8")
    elif values.dtype.kind in "SUO":
        text = "".join(value.decode("utf-8") if isinstance(value, bytes) else str(value)
                       for value in values.reshape(-1))
    else:
        raise ValueError("Saved instrument parameter map must contain text")
    # Re-read source metadata on every request. Only the pure parse result is
    # reused by exact text, so source edits and geometry/mask changes miss.
    parser = _cached_mask_ids if len(text.encode("utf-8")) <= _MASK_CACHE_MAX_TEXT_BYTES else _parse_mask_ids
    # Cache immutable tuples, not arrays whose write flags a caller can reset.
    result = np.asarray(parser(text), dtype=np.int64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=_MASK_CACHE_MAX_ENTRIES)
def _cached_mask_ids(text):
    return _parse_mask_ids(text)


def _parse_mask_ids(text):
    masks = {}
    for record in text.split("|"):
        fields = [field.strip() for field in record.split(";")]
        if len(fields) < 3 or fields[2] != "masked":
            continue
        if len(fields) < 4 or fields[2] != "masked" or fields[1] != "bool":
            raise ValueError("Malformed saved detector mask record")
        match = re.fullmatch(r"detID:([+-]?[0-9]+)", fields[0])
        if match is None:
            raise ValueError("Saved detector masks must identify an integer detID")
        detector_id = int(match[1])
        if not np.iinfo(np.int64).min <= detector_id <= np.iinfo(np.int64).max:
            raise ValueError("Saved detector mask ID is outside int64 bounds")
        flag = fields[3].lower()
        if flag not in ("0", "1", "false", "true"):
            raise ValueError("Saved detector masked flag must be boolean")
        masks[detector_id] = flag in ("1", "true")
    return tuple(sorted(detector_id for detector_id, masked in masks.items() if masked))


def apply_saved_detector_mask(experiment, detector_ids, weights):
    """Return trajectory weights with saved masked detector exposures set to zero.

    Detector identities select masks independently of array ordering. Input
    calibration weights remain unchanged, including immutable cached arrays.
    """
    detector_ids = np.asarray(detector_ids)
    weights = np.asarray(weights, dtype=float)
    if detector_ids.shape != weights.shape:
        raise ValueError("Detector IDs and normalization weights must have equal shapes")
    masked = saved_detector_mask_ids(experiment)
    if not masked.size:
        return weights
    result = weights.copy()
    result[np.isin(detector_ids, masked)] = 0.
    return result
