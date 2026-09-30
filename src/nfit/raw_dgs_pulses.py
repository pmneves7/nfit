"""SNS pause and bad-pulse time intervals shared by events and beam charge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np


def _timestamps(dataset):
    """Read NeXus timestamps as integer nanoseconds since the Unix epoch."""
    attrs = dataset.attrs
    if "offset_seconds" in attrs:
        origin = (int(attrs["offset_seconds"]) + 631152000) * 10**9
        origin += int(attrs.get("offset_nanoseconds", 0))
    else:
        offset = attrs.get("offset", attrs.get("start"))
        if offset is None:
            return None
        offset = offset.decode() if isinstance(offset, bytes) else str(offset)
        parsed = datetime.fromisoformat(offset.replace("Z", "+00:00"))
        origin = int(parsed.replace(microsecond=0).timestamp()) * 10**9
        fraction = re.search(r"\.(\d+)", offset)
        if fraction:
            origin += int((fraction[1] + "000000000")[:9])
    unit = attrs.get("units", "second")
    unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    scale = {"second": 1e9, "seconds": 1e9, "s": 1e9,
             "microsecond": 1e3, "microseconds": 1e3, "us": 1e3,
             "nanosecond": 1, "nanoseconds": 1, "ns": 1}.get(unit)
    if scale is None:
        raise ValueError(f"unsupported pulse timestamp unit: {unit}")
    return origin + np.rint(np.asarray(dataset, dtype=float) * scale).astype(np.int64)


def _inside(times, intervals):
    if not len(intervals):
        return np.zeros(len(times), dtype=bool)
    i = np.searchsorted(intervals[:, 0], times, side="right") - 1
    return (i >= 0) & (times < intervals[np.maximum(i, 0), 1])


def _value_intervals(times, good, start, stop, *, centre):
    """Mantid FilterByLogValue with zero tolerance and half-open intervals.

    Centre boundaries end at the last good timestamp before a bad value;
    left boundaries end at the first bad timestamp. A final good value is
    extended to the run stop only when its log ends before that stop.
    """
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    first, last = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    intervals = []
    for a, b in zip(first, last, strict=True):
        lo = min(start, times[a]) if a == 0 else times[a]
        hi = times[b - 1] if centre or b == len(times) else times[b]
        if b == len(times):
            hi = max(hi, stop)
        if lo < hi:
            intervals.append((lo, hi))
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def _intersect(a, b):
    intervals = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if lo < hi:
            intervals.append((lo, hi))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class PulseSelection:
    charge_keep: np.ndarray | None
    intervals: np.ndarray | None

    def bank_keep(self, bank):
        if self.intervals is not None and "event_time_zero" in bank:
            times = _timestamps(bank["event_time_zero"])
            if times is not None:
                return _inside(times, self.intervals)
        return self.charge_keep


def select_pulses(entry, threshold):
    """Apply pause [-1,0.5] first, then Mantid's charge-interval filter.

    Older files without timestamp logs retain the value-based fallback. The
    normal SNS path uses identical accepted time intervals for events and charge.
    """
    logs = entry.get("DASlogs")
    if threshold <= 0 and (logs is None or "pause" not in logs):
        return PulseSelection(None, None)
    charge = logs.get("proton_charge") if logs is not None else None
    if charge is None or "value" not in charge:
        return PulseSelection(None, None)
    values = np.asarray(charge["value"], dtype=float).reshape(-1)
    if not len(values):
        return PulseSelection(None, None)
    times = _timestamps(charge["time"]) if "time" in charge else None
    if times is None or len(times) != len(values):
        keep = values >= float(threshold) * 0.01 * np.mean(values) if threshold > 0 else None
        return PulseSelection(keep, None)
    if np.any(np.diff(times) < 0):
        raise ValueError("proton-charge timestamps must be ordered")
    start, stop = times[0], times[-1]
    intervals = np.asarray([(start, stop)], dtype=np.int64) if start < stop else np.empty((0, 2), dtype=np.int64)
    pause = logs.get("pause")
    if pause is not None and "time" in pause and "value" in pause:
        pause_times = _timestamps(pause["time"])
        pause_values = np.asarray(pause["value"], dtype=float).reshape(-1)
        if pause_times is not None and len(pause_times) == len(pause_values) and len(pause_times):
            if np.any(np.diff(pause_times) < 0):
                raise ValueError("pause timestamps must be ordered")
            intervals = _intersect(intervals, _value_intervals(
                pause_times, (pause_values >= -1) & (pause_values <= 0.5), start, stop, centre=False,
            ))
    if threshold > 0 and len(intervals):
        # RemoveDataOutsideTimeROI retains the records bracketing each interval.
        # Those records participate in the subsequent arithmetic charge mean.
        retained_records = np.zeros(len(times), dtype=bool)
        for lo, hi in intervals:
            a = max(0, int(np.searchsorted(times, lo, side="left")) - 1)
            b = min(len(times), int(np.searchsorted(times, hi, side="right")) + 1)
            retained_records[a:b] = True
        filtered_times, filtered_values = times[retained_records], values[retained_records]
        cutoff = float(threshold) * 0.01 * np.mean(filtered_values)
        good = (filtered_values >= cutoff) & (filtered_values <= 1.1 * np.max(filtered_values))
        intervals = _intersect(intervals, _value_intervals(
            filtered_times, good, start, stop, centre=True,
        ))
    return PulseSelection(_inside(times, intervals), intervals)
