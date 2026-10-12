"""Python detector-trajectory accumulation shared by raw and reduced DGS.

These fallback kernels preserve the existing physical-edge and optional Mantid
precision conventions. Source preparation and compiled backend selection remain
with their reducers.
"""

from __future__ import annotations

import numpy as np

from .dgs_reduction_policy import ENERGY_TO_K, dgs_trajectory_bin_indices
from .event_bin_indices import flat_bin_indices as _flat_bin_indices


def _accumulate_detector_trajectory(output, edges, inverse, direction, ei, energy_bounds, weight, mantid_precision=False):
    ki = np.sqrt(max(ei, 0.0) * ENERGY_TO_K)
    kf_values = np.sqrt(np.maximum(ei - np.asarray(energy_bounds, dtype=float), 0.0) * ENERGY_TO_K)
    low_kf, high_kf = float(np.min(kf_values)), float(np.max(kf_values))
    qin = inverse @ np.array([0.0, 0.0, ki])
    qout = inverse @ direction
    intersections = [low_kf, high_kf]
    for dim in range(3):
        if abs(qout[dim]) > 1e-14:
            intersections.extend((qin[dim] - boundary) / qout[dim] for boundary in edges[dim])
    intersections.extend(np.sqrt(np.maximum(ei - edges[3], 0.0) * ENERGY_TO_K))
    points = np.unique(np.clip(np.asarray(intersections), low_kf, high_kf))
    for first, second in zip(points[:-1], points[1:], strict=True):
        delta = (second * second - first * first) / ENERGY_TO_K
        if (mantid_precision and delta < 1e-10) or (not mantid_precision and second - first <= 1e-12):
            continue
        middle = 0.5 * (first + second)
        hkl = qin - qout * middle
        energy = ei - middle * middle / ENERGY_TO_K
        if mantid_precision:
            midpoint32 = np.float32(middle)
            energy = np.float32(ei - float(midpoint32 * midpoint32) / ENERGY_TO_K)
            flat = dgs_trajectory_bin_indices([[*hkl, energy]], edges, output.shape)[0]
        else:
            flat = _flat_bin_indices(np.asarray([[*hkl, energy]]), edges, output.shape)[0]
        if flat >= 0:
            output.ravel()[flat] += weight * delta


def _accumulate_powder_detector_trajectory(
    output, edges, scattering_angle, incident_energy, energy_bounds, weight, mantid_precision=False
):
    """Integrate one detector trajectory through radial-Q/energy bins."""

    ki = np.sqrt(max(incident_energy, 0.0) * ENERGY_TO_K)
    kf_values = np.sqrt(
        np.maximum(incident_energy - np.asarray(energy_bounds, dtype=float), 0.0)
        * ENERGY_TO_K
    )
    low_kf, high_kf = float(np.min(kf_values)), float(np.max(kf_values))
    cosine = float(np.cos(scattering_angle))
    sine_squared = max(0.0, 1.0 - cosine * cosine)
    intersections = [low_kf, high_kf]
    # q^2 = ki^2 + kf^2 - 2 ki kf cos(theta).  Each radial
    # boundary can cross a detector trajectory twice.
    for q_boundary in edges[0]:
        discriminant = q_boundary * q_boundary - ki * ki * sine_squared
        if discriminant < 0.0:
            continue
        root = np.sqrt(discriminant)
        intersections.extend((ki * cosine - root, ki * cosine + root))
    intersections.extend(
        np.sqrt(np.maximum(incident_energy - edges[1], 0.0) * ENERGY_TO_K)
    )
    points = np.unique(np.clip(np.asarray(intersections), low_kf, high_kf))
    for first, second in zip(points[:-1], points[1:], strict=True):
        delta = (second * second - first * first) / ENERGY_TO_K
        if (mantid_precision and delta < 1e-10) or (not mantid_precision and second - first <= 1.0e-12):
            continue
        middle = 0.5 * (first + second)
        q_modulus = np.sqrt(
            max(
                ki * ki + middle * middle - 2.0 * ki * middle * cosine,
                0.0,
            )
        )
        energy = incident_energy - middle * middle / ENERGY_TO_K
        if mantid_precision:
            midpoint32 = np.float32(middle)
            energy = np.float32(incident_energy - float(midpoint32 * midpoint32) / ENERGY_TO_K)
            flat = dgs_trajectory_bin_indices([[q_modulus, energy]], edges, output.shape)[0]
        else:
            flat = _flat_bin_indices(
                np.asarray([[q_modulus, energy]]), edges, output.shape
            )[0]
        if flat >= 0:
            output.ravel()[flat] += (
                weight * delta
            )
