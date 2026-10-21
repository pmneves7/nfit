"""Monitor-normalized MACS SPEC energy-scan detector calibration.

The fitted coordinate is the scanned Bragg angle in degrees, as in DAVE.
Calibration uncertainty is diagnostic; reduction treats these factors as known.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import curve_fit

from .macs_geometry import MACS_DETECTOR_OFFSETS_DEG, MACS_KIDNEY_ZERO_DEG


def _read(entry: h5py.Group, paths: list[str], n: int) -> np.ndarray | None:
    for path in paths:
        if path in entry:
            a = np.asarray(entry[path][()], dtype=float).reshape(-1)
            if a.size == 1:
                return np.full(n, a[0])
            if a.size != n:
                raise ValueError(f"MACS {path} must have one or {n} values")
            return a
    return None


def _moving(a: np.ndarray | None) -> bool:
    # Fixed MACS motors jitter by hundredths of a degree in real scans.
    return a is not None and np.count_nonzero(np.isfinite(a)) > 1 and np.ptp(a[np.isfinite(a)]) > 0.05


def _angle(entry: h5py.Group, names: list[str], n: int) -> np.ndarray | None:
    return _read(entry, [f"DAS_logs/{name}/{field}" for name in names
                         for field in ("softPosition", "primaryNode")], n)


def _scan(path: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        entries = [g for g in handle.values() if isinstance(g, h5py.Group) and "DAS_logs" in g]
        if not entries:
            raise ValueError(f"{path}: missing MACS DAS_logs")
        e = entries[0]
        instrument = e.get("DAS_logs/experiment/instrument", e.get("instrument/name"))
        if instrument is None or np.asarray(instrument[()]).reshape(-1)[0] not in (b"MACS", "MACS"):
            raise ValueError(f"{path}: expected MACS instrument")
        counts = np.asarray(e["DAS_logs/specDetector/counts"][()], dtype=float)
        if counts.ndim != 2 or 20 not in counts.shape:
            raise ValueError(f"{path}: SPEC counts must have twenty channels")
        if counts.shape[1] != 20:
            counts = counts.T
        n = len(counts)
        monitor = _read(e, ["DAS_logs/counter/liveMonitor", "data/monitor"], n)
        kidney = _angle(e, ["kidneyMotor", "kidney"], n)
        if monitor is None or kidney is None:
            raise ValueError(f"{path}: monitor and kidney angle are required")
        ei = _read(e, ["DAS_logs/ei/energy", "DAS_logs/initialEnergy/energy"], n)
        ef = _read(e, ["DAS_logs/ef/energy", "DAS_logs/finalEnergy/energy"], n)
        mono = _angle(e, ["monoTwoTheta"], n)
        if mono is None:
            theta = _angle(e, ["monoTheta"], n)
            mono = None if theta is None else 2 * theta
        analyzers = []
        for j in range(1, 21):
            a = _angle(e, [f"anaTwoTheta{j:02d}"], n)
            if a is None:
                a = _angle(e, [f"anaTheta{j:02d}"], n)
                a = None if a is None else 2 * a
            if a is None:
                a = _angle(e, ["anaTwoTheta"], n)
            if a is None:
                a = _angle(e, ["anaTheta"], n)
                a = None if a is None else 2 * a
            analyzers.append(a)
        if _moving(mono) and any(_moving(a) for a in analyzers):
            raise ValueError(f"{path}: simultaneous incident and final energy scans are unsupported")
        if _moving(mono):
            axes, sweep, fixed = [mono] * 20, "incident", ef
        elif any(_moving(a) for a in analyzers):
            axes, sweep, fixed = analyzers, "final", ei
        else:
            raise ValueError(f"{path}: no scanned monochromator or analyzer Bragg angle")
        if fixed is None or not np.all(np.isfinite(fixed)) or np.any(fixed <= 0):
            raise ValueError(f"{path}: finite positive fixed energy metadata are required")
        if np.ptp(fixed) > 0.01:
            raise ValueError(f"{path}: fixed energy changes during the scan")
        return dict(counts=counts, monitor=monitor, kidney=kidney, axes=axes,
                    sweep=sweep, fixed_energy_mev=float(np.mean(fixed)))


def _gaussian(x, amplitude, center, sigma, background):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background


def _fit(x: np.ndarray, counts: np.ndarray, monitor: np.ndarray,
         monitor_reference: float) -> dict[str, Any]:
    good = np.isfinite(x) & np.isfinite(counts) & (counts >= 0) & np.isfinite(monitor) & (monitor > 0)
    x, c, m = x[good], counts[good], monitor[good]
    if len(x) < 6 or len(np.unique(x)) < 6:
        return {"status": "insufficient_points", "points": len(x)}
    y = c / m
    # DAVE replaces zero count uncertainties by one at its monitor reference.
    error = np.where(c > 0, np.sqrt(c) / m, 1 / monitor_reference)
    span = float(np.ptp(x))
    amplitude = float(np.ptp(y))
    if amplitude <= 0:
        return {"status": "flat_scan", "points": len(x)}
    try:
        p, covariance = curve_fit(_gaussian, x, y, sigma=error, absolute_sigma=True,
            p0=[amplitude, x[np.argmax(y)], span / 6, np.min(y)],
            bounds=([0, np.min(x), span * 1e-6, -np.inf],
                    [np.inf, np.max(x), span * 2, np.inf]), maxfev=20000,
            x_scale=[amplitude, max(abs(float(np.mean(x))), span), span, amplitude])
        area = float(np.sqrt(2 * np.pi) * p[0] * p[2])
        if not np.all(np.isfinite(covariance)) or not np.isfinite(area) or area <= 0:
            raise ValueError("unidentifiable fit")
        if p[1] - 2 * p[2] < np.min(x) or p[1] + 2 * p[2] > np.max(x):
            return {"status": "incomplete_peak", "points": len(x)}
        return dict(status="ok", points=len(x), area=area, amplitude=float(p[0]),
                    center_deg=float(p[1]), sigma_deg=float(p[2]), background=float(p[3]),
                    reduced_chi_squared=float(np.sum(((y - _gaussian(x, *p)) / error) ** 2) / (len(x) - 4)))
    except (RuntimeError, ValueError, FloatingPointError):
        return {"status": "fit_failed", "points": len(x)}


@lru_cache(maxsize=8)
def _calculate(identities: tuple, mask: float) -> dict[str, Any]:
    scans = [_scan(identity[0]) for identity in identities]
    reference = scans[0]
    valid_reference = reference["monitor"][np.isfinite(reference["monitor"]) & (reference["monitor"] > 0)]
    if not len(valid_reference):
        raise ValueError("First vanadium scan contains no positive finite monitor reference")
    monitor_reference = float(valid_reference[0])
    if any(s["sweep"] != reference["sweep"] or
           abs(s["fixed_energy_mev"] - reference["fixed_energy_mev"]) > 0.01 for s in scans):
        raise ValueError("Vanadium scans must sweep the same energy axis at the same fixed energy (within 0.01 meV)")
    diagnostics = []
    areas: list[list[float]] = [[] for _ in range(20)]
    for identity, scan in zip(identities, scans, strict=True):
        fits = []
        for j, axis in enumerate(scan["axes"]):
            a4 = float(np.mean(scan["kidney"])) + MACS_KIDNEY_ZERO_DEG + MACS_DETECTOR_OFFSETS_DEG[j]
            good = np.full(len(scan["kidney"]), np.isfinite(a4) and abs(a4) > mask)
            if not np.isfinite(a4):
                result = {"status": "invalid_kidney_angle", "points": 0}
            elif abs(a4) <= mask:
                result = {"status": "direct_beam_excluded", "points": 0}
            elif axis is None:
                result = {"status": "missing_angle", "points": 0}
            else:
                result = _fit(axis[good], scan["counts"][good, j], scan["monitor"][good], monitor_reference)
            result["mean_a4_deg"] = float(a4) if np.isfinite(a4) else None
            fits.append(result)
            if result["status"] == "ok":
                areas[j].append(result["area"])
        diagnostics.append({"path": identity[0], "channels": fits})
    inverse = np.array([1 / np.mean(a) if a else np.nan for a in areas])
    valid = np.isfinite(inverse)
    if not np.any(valid):
        raise ValueError("No usable Gaussian peaks in MACS vanadium scans")
    inverse /= np.mean(inverse[valid])
    return dict(format="nfit.macs_vanadium", version=1,
                factors=[float(v) if ok else None for v, ok in zip(inverse, valid, strict=True)],
                usable_channels=valid.tolist(), missing_channels=(np.flatnonzero(~valid) + 1).tolist(),
                a4_mask_deg=mask, monitor_reference=monitor_reference, sweep=reference["sweep"], fixed_energy_mev=reference["fixed_energy_mev"],
                area_coordinate="Bragg two-theta angle (deg)",
                source_files=[dict(path=i[0], available=True, size_bytes=i[1], mtime_ns=i[2], ctime_ns=i[3]) for i in identities],
                diagnostics=diagnostics)


def calculate_macs_vanadium(files, a4_mask_deg: float = 15.0) -> dict[str, Any]:
    """Fit SPEC peaks and return mean-one inverse-area sensitivity factors.

    ``files`` supplies one or more MACS NeXus energy scans. Points with
    ``abs(mean(A4)) <= a4_mask_deg`` are excluded for the entire file. Valid per-file Gaussian areas
    are averaged per detector before inversion. Missing factors are ``None``;
    consumers must mask those channels, never replace them with unity.
    Results are JSON serializable. A bounded file-identity cache avoids repeated
    fits, and each call returns an independent copy of the calibration record.
    """
    mask = float(a4_mask_deg)
    if not np.isfinite(mask) or mask < 0 or mask > 180:
        raise ValueError("MACS vanadium A4 mask must be between 0 and 180 degrees")
    if isinstance(files, (str, Path)):
        files = [files]
    paths = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in files))
    if not paths:
        raise ValueError("At least one MACS vanadium scan is required")
    identities = []
    for path in paths:
        stat = Path(path).stat()
        identities.append((path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return deepcopy(_calculate(tuple(identities), mask))


def vanadium_source_provenance(files) -> list[dict[str, Any]]:
    """Return cheap source identities, retaining unavailable paths explicitly."""
    if isinstance(files, (str, Path)):
        files = [files]
    records = []
    for value in dict.fromkeys(str(Path(p).expanduser().resolve()) for p in files):
        try:
            stat = Path(value).stat()
        except OSError:
            records.append(dict(path=value, available=False))
        else:
            records.append(dict(path=value, available=True, size_bytes=stat.st_size,
                                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
    return records
