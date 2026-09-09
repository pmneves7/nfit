"""Fused native MDNorm-style detector trajectory accumulation."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

ENERGY_TO_K2 = 2.072124855


@njit(fastmath=False, nogil=True, parallel=True)
def trajectory_normalization(
    theta, phi, solid, inverse_matrices, incident_energies, energy_limits,
    proton_charges, edge0, edge1, edge2, edge3, shape,
):
    workers = get_num_threads()
    output_size = shape[0] * shape[1] * shape[2] * shape[3]
    partial = np.zeros((workers, output_size), dtype=np.float64)
    detectors = theta.size
    total = inverse_matrices.shape[0] * detectors
    max_intersections = edge0.size + edge1.size + edge2.size + edge3.size + 2
    scratch = np.empty((workers, max_intersections), dtype=np.float64)
    for task in prange(total):
        run = task // detectors
        detector = task - run * detectors
        detector_weight = solid[detector]
        if not np.isfinite(detector_weight) or detector_weight <= 0.0:
            continue
        ei = incident_energies[run]
        if ei <= 0.0:
            continue
        ki = np.sqrt(ei / ENERGY_TO_K2)
        kfa = np.sqrt(max(ei - energy_limits[run, 0], 0.0) / ENERGY_TO_K2)
        kfb = np.sqrt(max(ei - energy_limits[run, 1], 0.0) / ENERGY_TO_K2)
        low_kf = min(kfa, kfb)
        high_kf = max(kfa, kfb)
        st = np.sin(theta[detector])
        direction0 = st * np.cos(phi[detector])
        direction1 = st * np.sin(phi[detector])
        direction2 = np.cos(theta[detector])
        inverse = inverse_matrices[run]
        qin0 = inverse[0, 2] * ki
        qin1 = inverse[1, 2] * ki
        qin2 = inverse[2, 2] * ki
        qout0 = inverse[0, 0] * direction0 + inverse[0, 1] * direction1 + inverse[0, 2] * direction2
        qout1 = inverse[1, 0] * direction0 + inverse[1, 1] * direction1 + inverse[1, 2] * direction2
        qout2 = inverse[2, 0] * direction0 + inverse[2, 1] * direction1 + inverse[2, 2] * direction2
        # Clip the detector trajectory to the requested four-dimensional box
        # before finding internal bin crossings.  Thin transverse slices reject
        # most run/detector/symmetry combinations here and avoid scanning every
        # grid boundary for trajectories that cannot contribute.
        clipped_low = low_kf
        clipped_high = high_kf
        outside = False
        for dim in range(3):
            edges = edge0 if dim == 0 else (edge1 if dim == 1 else edge2)
            qin = qin0 if dim == 0 else (qin1 if dim == 1 else qin2)
            qout = qout0 if dim == 0 else (qout1 if dim == 1 else qout2)
            if abs(qout) <= 1e-14:
                if qin < edges[0] or qin > edges[-1]:
                    outside = True
                    break
                continue
            crossing0 = (qin - edges[0]) / qout
            crossing1 = (qin - edges[-1]) / qout
            axis_low = min(crossing0, crossing1)
            axis_high = max(crossing0, crossing1)
            clipped_low = max(clipped_low, axis_low)
            clipped_high = min(clipped_high, axis_high)
            if clipped_high - clipped_low <= 1e-12:
                outside = True
                break
        if outside:
            continue
        energy_kf0 = np.sqrt(max(ei - edge3[-1], 0.0) / ENERGY_TO_K2)
        energy_kf1 = np.sqrt(max(ei - edge3[0], 0.0) / ENERGY_TO_K2)
        clipped_low = max(clipped_low, min(energy_kf0, energy_kf1))
        clipped_high = min(clipped_high, max(energy_kf0, energy_kf1))
        if clipped_high - clipped_low <= 1e-12:
            continue
        thread = get_thread_id()
        intersections = scratch[thread]
        count = 2
        intersections[0] = clipped_low
        intersections[1] = clipped_high
        for dim in range(3):
            edges = edge0 if dim == 0 else (edge1 if dim == 1 else edge2)
            qin = qin0 if dim == 0 else (qin1 if dim == 1 else qin2)
            qout = qout0 if dim == 0 else (qout1 if dim == 1 else qout2)
            if abs(qout) > 1e-14:
                for boundary in edges:
                    value = (qin - boundary) / qout
                    if clipped_low < value < clipped_high:
                        intersections[count] = value
                        count += 1
        for boundary in edge3:
            value = np.sqrt(max(ei - boundary, 0.0) / ENERGY_TO_K2)
            if clipped_low < value < clipped_high:
                intersections[count] = value
                count += 1
        # The intersection count is modest; insertion sort avoids another array.
        for i in range(1, count):
            value = intersections[i]
            j = i - 1
            while j >= 0 and intersections[j] > value:
                intersections[j + 1] = intersections[j]
                j -= 1
            intersections[j + 1] = value
        weight = proton_charges[run] * detector_weight
        for i in range(count - 1):
            first = intersections[i]
            second = intersections[i + 1]
            if second - first <= 1e-12:
                continue
            middle = 0.5 * (first + second)
            coordinate0 = qin0 - qout0 * middle
            coordinate1 = qin1 - qout1 * middle
            coordinate2 = qin2 - qout2 * middle
            coordinate3 = ei - ENERGY_TO_K2 * middle * middle
            index0 = np.searchsorted(edge0, coordinate0, side="right") - 1
            index1 = np.searchsorted(edge1, coordinate1, side="right") - 1
            index2 = np.searchsorted(edge2, coordinate2, side="right") - 1
            index3 = np.searchsorted(edge3, coordinate3, side="right") - 1
            if coordinate0 == edge0[-1]:
                index0 = edge0.size - 2
            if coordinate1 == edge1[-1]:
                index1 = edge1.size - 2
            if coordinate2 == edge2[-1]:
                index2 = edge2.size - 2
            if coordinate3 == edge3[-1]:
                index3 = edge3.size - 2
            if 0 <= index0 < edge0.size - 1 and 0 <= index1 < edge1.size - 1 and 0 <= index2 < edge2.size - 1 and 0 <= index3 < edge3.size - 1:
                flat = ((index0 * shape[1] + index1) * shape[2] + index2) * shape[3] + index3
                partial[thread, flat] += weight * ENERGY_TO_K2 * (second * second - first * first)
    return np.sum(partial, axis=0)


def run_trajectory_normalization(*args, workers: int):
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        return trajectory_normalization(*args)
    finally:
        set_num_threads(previous)
