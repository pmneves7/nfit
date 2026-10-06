"""Monitor response and optional historical higher-order MACS calibration."""

from __future__ import annotations

import numpy as np

# Public DAVE dm_macs_lambda2correction calibration (Jose's intensity simulation).
# This is an optional empirical model, not an experiment-specific measurement.
_LAMBDA2_RATIOS = np.array([
    2.378025665, 1.793602670, 1.490934555, 1.252089079, 1.127126095,
    0.959301163, 0.792585170, 0.649712929, 0.582762512, 0.521160485,
    0.483133752, 0.444736852, 0.416914118, 0.390288542, 0.362685682,
    0.347918053, 0.339471838, 0.327112776, 0.325793541, 0.305038794,
    0.292795409, 0.284034166, 0.264029518, 0.268863169, 0.259363234,
    0.247959631, 0.246067755, 0.237768686, 0.225825977, 0.228803554,
    0.229217028, 0.235338426, 0.236590056, 0.233790439, 0.242718897,
    0.242768485, 0.242982347,
])
_LAMBDA2_POLYNOMIAL = np.polynomial.Polynomial.fit(np.arange(2.0, 20.25, 0.5), _LAMBDA2_RATIOS, 7)


def incident_filter_states(entry, count, policy):
    """Resolve CFX filters per scan point; missing states stay unknown."""
    if policy in {"in", "out"}:
        return np.full(count, policy == "in", dtype=float)
    states = []
    for name in ("CFXBE", "CFXHOPG"):
        source = entry.get(f"DAS_logs/{name}/primaryNode")
        if source is None:
            states.append(np.full(count, np.nan))
            continue
        values = np.asarray(source[()]).reshape(-1)
        if values.size == 1:
            values = np.repeat(values, count)
        if values.size != count:
            raise ValueError(f"MACS {name} filter log does not match scan points")
        decoded = [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        states.append(np.array([1. if value.strip().lower() == "in" else
                               0. if value.strip().lower() == "out" else np.nan for value in decoded]))
    inserted = (states[0] == 1) | (states[1] == 1)
    removed = (states[0] == 0) & (states[1] == 0)
    return np.where(inserted, 1., np.where(removed, 0., np.nan))


def corrected_monitor(monitor, ei, filters, policy):
    """Apply the opt-in DAVE empirical model only to known unfiltered points."""
    monitor = np.asarray(monitor, float)
    if policy == "off":
        return monitor
    if np.any(~np.isfinite(filters)):
        raise ValueError("MACS incident filter state is unknown; choose Inserted or Removed before higher-order correction")
    selected = filters == 0
    if np.any(selected & ((ei < 2.) | (ei > 20.))):
        raise ValueError("DAVE's MACS higher-order monitor model is calibrated only for 2–20 meV")
    factor = 1. + 0.5 * _LAMBDA2_POLYNOMIAL(ei)
    if np.any(selected & (~np.isfinite(factor) | (factor <= 0))):
        raise ValueError("MACS higher-order monitor correction is not finite and positive")
    return np.where(selected, monitor / factor, monitor)


def monitor_kinematic_factor(ki, kf, settings):
    """Intensity multiplier after recorded-monitor normalization, dimensionless."""
    factor = np.ones(np.shape(ki), dtype=float)
    if settings["monitor_response"] == "inverse_velocity":
        factor *= settings["monitor_reference_k_inv_angstrom"] / ki
    if settings["ki_kf_normalization"]:
        factor *= ki / kf
    return factor
