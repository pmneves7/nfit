"""One sample orientation per DGS run, from accepted-time log averages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .raw_dgs_pulses import iso_timestamp_ns, nexus_timestamps, select_pulses

GONIOMETER_AVERAGING_CONVENTION = "mantid_accepted_time_mean"


@dataclass(frozen=True)
class SampleRotation:
    """Universal goniometer angles in degrees, without retaining log arrays."""

    omega: float
    phi: float
    chi: float
    varying_logs: bool


def step_log_mean(times, values, intervals):
    """Duration-weight a stepwise log over disjoint half-open time intervals.

    Times and interval endpoints have the same integer nanosecond origin.
    The last record at a duplicate timestamp owns the following duration.
    Endpoint values extend outside the recorded series, as in Mantid.
    """
    times = np.asarray(times, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(times) or len(times) != len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Sample rotation needs matching finite log values and timestamps")
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    mean = total = 0.0
    for lo, hi in np.asarray(intervals, dtype=np.int64).reshape(-1, 2):
        if lo >= hi:
            continue
        index = int(np.searchsorted(times, lo, side="right")) - 1
        cursor = int(lo)
        while cursor < hi:
            end = min(int(hi), int(times[index + 1])) if index + 1 < len(times) else int(hi)
            duration = (end - cursor) * 1e-9
            total += duration
            mean += (duration / total) * (float(values[max(index, 0)]) - mean)
            cursor = end
            if cursor < hi:
                index = int(np.searchsorted(times, cursor, side="right")) - 1
    if total <= 0.0:
        raise ValueError("Sample rotation has no accepted time interval")
    return mean


def _run_end_time(entry):
    for name in ("end_time", "run_end"):
        if name in entry:
            value = np.asarray(entry[name]).reshape(-1)
            if len(value):
                return iso_timestamp_ns(value[0])
    charge = entry.get("DASlogs/proton_charge/time")
    times = nexus_timestamps(charge) if charge is not None else None
    return int(np.max(times)) if times is not None and len(times) else None


def sample_rotation(entry, *, bad_pulse_threshold=0.0):
    """Resolve omega/phi/chi using the same accepted times as DGS reduction.

    Constant and legacy scalar logs need no time or pulse arrays. Varying logs
    require temporal support. An unrestricted log follows Mantid's appended
    run-end record and final-interval extension, rather than a nominal duration.
    """
    angles = {}
    varying = False
    intervals = run_end = None
    have_intervals = have_run_end = False
    for name in ("omega", "phi", "chi"):
        log = entry.get("DASlogs/" + name)
        if log is None:
            angles[name] = 0.0
            continue
        payload = log.get("value", log.get("average_value"))
        if payload is None:
            raise ValueError(f"Sample rotation log {name!r} has no numeric value")
        values = np.asarray(payload, dtype=float).reshape(-1)
        if not len(values) or np.any(~np.isfinite(values)):
            raise ValueError(f"Sample rotation log {name!r} needs finite values")
        if np.all(values == values[0]):
            angles[name] = float(values[0])
            continue
        varying = True
        timestamps = log.get("time")
        times = nexus_timestamps(timestamps) if timestamps is not None else None
        if times is None or len(times) != len(values):
            raise ValueError(f"Sample rotation log {name!r} needs matching timestamps for time averaging")
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        if not have_intervals:
            intervals = select_pulses(entry, bad_pulse_threshold).intervals
            have_intervals = True
        if not have_run_end:
            run_end = _run_end_time(entry)
            have_run_end = True
        if run_end is not None and run_end > times[-1] and np.issubdtype(payload.dtype, np.floating):
            times = np.r_[times, run_end]
            values = np.r_[values, values[-1]]
        active = intervals
        if active is None:
            distinct = np.unique(times)
            if len(distinct) < 2:
                raise ValueError(f"Sample rotation log {name!r} has no positive timestamp interval")
            stop = distinct[-1] + (distinct[-1] - distinct[-2])
            active = np.asarray([(distinct[0], stop)], dtype=np.int64)
        angles[name] = step_log_mean(times, values, active)
    return SampleRotation(**angles, varying_logs=varying)
