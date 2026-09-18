"""Fused native MDNorm-style detector trajectory accumulation."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

ENERGY_TO_K2 = 2.072124855


@njit(fastmath=False, nogil=True, parallel=True)
def accumulate_trajectory_normalization(
    partial, scratch, theta, phi, solid, inverse_matrices, incident_energies,
    energy_limits, proton_charges, edge0, edge1, edge2, edge3, shape,
):
    detectors = theta.size
    total = inverse_matrices.shape[0] * detectors
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


class TrajectoryNormalizationAccumulator:
    """Persistent worker-private storage for batched HKLE trajectories."""

    def __init__(self, edge0, edge1, edge2, edge3, shape, *, workers: int):
        available = get_num_threads()
        self.workers = max(1, min(int(workers), available))
        output_size = int(np.prod(np.asarray(shape, dtype=np.int64)))
        intersections = (
            np.asarray(edge0).size
            + np.asarray(edge1).size
            + np.asarray(edge2).size
            + np.asarray(edge3).size
            + 2
        )
        self.partial = np.zeros((self.workers, output_size), dtype=np.float64)
        self.scratch = np.empty((self.workers, intersections), dtype=np.float64)

    def accumulate(self, *args):
        """Add one run batch without clearing or reducing prior contributions."""

        if not self.partial.flags.writeable:
            raise RuntimeError("trajectory normalization accumulator is finalized")
        previous = get_num_threads()
        set_num_threads(self.workers)
        try:
            accumulate_trajectory_normalization(self.partial, self.scratch, *args)
        finally:
            set_num_threads(previous)

    def result(self):
        """Finalize and reduce worker rows; no more batches may be added."""

        if self.workers == 1:
            result = self.partial[0]
            self.partial.setflags(write=False)
            result.setflags(write=False)
            return result
        result = np.sum(self.partial, axis=0)
        self.partial.setflags(write=False)
        return result


def trajectory_normalization(
    theta, phi, solid, inverse_matrices, incident_energies, energy_limits,
    proton_charges, edge0, edge1, edge2, edge3, shape,
):
    """Compatibility entry point for a single trajectory batch."""

    accumulator = TrajectoryNormalizationAccumulator(
        edge0, edge1, edge2, edge3, shape, workers=get_num_threads()
    )
    accumulator.accumulate(
        theta, phi, solid, inverse_matrices, incident_energies, energy_limits,
        proton_charges, edge0, edge1, edge2, edge3, shape,
    )
    return accumulator.result()


def run_trajectory_normalization(*args, workers: int):
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        return trajectory_normalization(*args)
    finally:
        set_num_threads(previous)


def trajectory_normalization_accumulator(
    edge0, edge1, edge2, edge3, shape, *, workers: int
):
    """Create persistent storage for a sequence of compatible run batches."""

    return TrajectoryNormalizationAccumulator(
        edge0, edge1, edge2, edge3, shape, workers=workers
    )


@njit(fastmath=False, nogil=True)
def _powder_background_at(
    q_value,
    energy,
    q_centers,
    energy_centers,
    signal,
    variance,
    interpolation_code,
):
    """Return an interpolated powder value and conservative point uncertainty."""

    if (
        q_value < q_centers[0]
        or q_value > q_centers[-1]
        or energy < energy_centers[0]
        or energy > energy_centers[-1]
    ):
        return False, 0.0, 0.0
    if interpolation_code == 1:
        q_upper = np.searchsorted(q_centers, q_value, side="left")
        e_upper = np.searchsorted(energy_centers, energy, side="left")
        if q_upper == 0:
            q_index = 0
        elif q_upper == q_centers.size:
            q_index = q_centers.size - 1
        else:
            q_index = (
                q_upper - 1
                if q_value - q_centers[q_upper - 1]
                <= q_centers[q_upper] - q_value
                else q_upper
            )
        if e_upper == 0:
            e_index = 0
        elif e_upper == energy_centers.size:
            e_index = energy_centers.size - 1
        else:
            e_index = (
                e_upper - 1
                if energy - energy_centers[e_upper - 1]
                <= energy_centers[e_upper] - energy
                else e_upper
            )
        value = signal[q_index, e_index]
        source_variance = variance[q_index, e_index]
        if not np.isfinite(value) or not np.isfinite(source_variance):
            return False, 0.0, 0.0
        return True, value, np.sqrt(max(source_variance, 0.0))

    q_upper = np.searchsorted(q_centers, q_value, side="right")
    e_upper = np.searchsorted(energy_centers, energy, side="right")
    if q_upper == 0:
        q_upper = 1
    elif q_upper >= q_centers.size:
        q_upper = q_centers.size - 1
    if e_upper == 0:
        e_upper = 1
    elif e_upper >= energy_centers.size:
        e_upper = energy_centers.size - 1
    q_lower = q_upper - 1
    e_lower = e_upper - 1
    q_fraction = (q_value - q_centers[q_lower]) / (
        q_centers[q_upper] - q_centers[q_lower]
    )
    e_fraction = (energy - energy_centers[e_lower]) / (
        energy_centers[e_upper] - energy_centers[e_lower]
    )
    value = 0.0
    point_variance = 0.0
    for q_choice in range(2):
        q_index = q_lower if q_choice == 0 else q_upper
        q_weight = 1.0 - q_fraction if q_choice == 0 else q_fraction
        for e_choice in range(2):
            e_index = e_lower if e_choice == 0 else e_upper
            e_weight = 1.0 - e_fraction if e_choice == 0 else e_fraction
            weight = q_weight * e_weight
            corner_value = signal[q_index, e_index]
            corner_variance = variance[q_index, e_index]
            if weight > np.finfo(np.float64).eps and (
                not np.isfinite(corner_value) or not np.isfinite(corner_variance)
            ):
                return False, 0.0, 0.0
            if np.isfinite(corner_value) and np.isfinite(corner_variance):
                value += weight * corner_value
                point_variance += weight * weight * corner_variance
    return True, value, np.sqrt(max(point_variance, 0.0))


@njit(fastmath=False, nogil=True, parallel=True)
def trajectory_powder_background(
    theta,
    phi,
    solid,
    inverse_matrices,
    incident_energies,
    energy_limits,
    proton_charges,
    edge0,
    edge1,
    edge2,
    edge3,
    shape,
    q_centers,
    energy_centers,
    background_signal,
    background_variance,
    interpolation_code,
):
    """Project a radial powder field through sample detector trajectories."""

    workers = get_num_threads()
    output_size = shape[0] * shape[1] * shape[2] * shape[3]
    partial_signal = np.zeros((workers, output_size), dtype=np.float64)
    partial_sigma = np.zeros((workers, output_size), dtype=np.float64)
    partial_valid = np.zeros((workers, output_size), dtype=np.float64)
    partial_total = np.zeros((workers, output_size), dtype=np.float64)
    detectors = theta.size
    total = inverse_matrices.shape[0] * detectors
    max_intersections = (
        edge0.size
        + edge1.size
        + edge2.size
        + edge3.size
        + 2 * q_centers.size
        + energy_centers.size
        + 2
    )
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
            clipped_low = max(clipped_low, min(crossing0, crossing1))
            clipped_high = min(clipped_high, max(crossing0, crossing1))
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
        cosine = direction2
        sine_squared = max(0.0, 1.0 - cosine * cosine)
        for q_boundary in q_centers:
            discriminant = q_boundary * q_boundary - ki * ki * sine_squared
            if discriminant < 0.0:
                continue
            root = np.sqrt(discriminant)
            first = ki * cosine - root
            second = ki * cosine + root
            if clipped_low < first < clipped_high:
                intersections[count] = first
                count += 1
            if clipped_low < second < clipped_high and abs(second - first) > 1e-14:
                intersections[count] = second
                count += 1
        for boundary in energy_centers:
            value = np.sqrt(max(ei - boundary, 0.0) / ENERGY_TO_K2)
            if clipped_low < value < clipped_high:
                intersections[count] = value
                count += 1
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
            energy = ei - ENERGY_TO_K2 * middle * middle
            index0 = np.searchsorted(edge0, coordinate0, side="right") - 1
            index1 = np.searchsorted(edge1, coordinate1, side="right") - 1
            index2 = np.searchsorted(edge2, coordinate2, side="right") - 1
            index3 = np.searchsorted(edge3, energy, side="right") - 1
            if coordinate0 == edge0[-1]:
                index0 = edge0.size - 2
            if coordinate1 == edge1[-1]:
                index1 = edge1.size - 2
            if coordinate2 == edge2[-1]:
                index2 = edge2.size - 2
            if energy == edge3[-1]:
                index3 = edge3.size - 2
            if not (
                0 <= index0 < edge0.size - 1
                and 0 <= index1 < edge1.size - 1
                and 0 <= index2 < edge2.size - 1
                and 0 <= index3 < edge3.size - 1
            ):
                continue
            flat = ((index0 * shape[1] + index1) * shape[2] + index2) * shape[3] + index3
            segment_weight = weight * ENERGY_TO_K2 * (second * second - first * first)
            partial_total[thread, flat] += segment_weight
            q_value = np.sqrt(
                max(
                    0.0,
                    ki * ki + middle * middle - 2.0 * ki * middle * cosine,
                )
            )
            valid, value, sigma = _powder_background_at(
                q_value,
                energy,
                q_centers,
                energy_centers,
                background_signal,
                background_variance,
                interpolation_code,
            )
            if valid:
                partial_valid[thread, flat] += segment_weight
                partial_signal[thread, flat] += segment_weight * value
                partial_sigma[thread, flat] += segment_weight * sigma
    return (
        np.sum(partial_signal, axis=0),
        np.sum(partial_sigma, axis=0),
        np.sum(partial_valid, axis=0),
        np.sum(partial_total, axis=0),
    )


def run_trajectory_powder_background(*args, workers: int):
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        return trajectory_powder_background(*args)
    finally:
        set_num_threads(previous)


@njit(fastmath=False, nogil=True, parallel=True)
def powder_trajectory_normalization(
    theta,
    solid,
    incident_energies,
    energy_limits,
    proton_charges,
    q_edges,
    energy_edges,
    shape,
):
    """Accumulate radial detector trajectories for raw or stored DGS runs."""

    workers = get_num_threads()
    output_size = shape[0] * shape[1]
    partial = np.zeros((workers, output_size), dtype=np.float64)
    detectors = theta.size
    total = incident_energies.size * detectors
    max_intersections = 2 * q_edges.size + energy_edges.size + 2
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
        cosine = np.cos(theta[detector])
        sine_squared = max(0.0, 1.0 - cosine * cosine)
        thread = get_thread_id()
        intersections = scratch[thread]
        count = 2
        intersections[0] = low_kf
        intersections[1] = high_kf
        for boundary in q_edges:
            discriminant = boundary * boundary - ki * ki * sine_squared
            if discriminant < 0.0:
                continue
            root = np.sqrt(discriminant)
            first = ki * cosine - root
            second = ki * cosine + root
            if low_kf < first < high_kf:
                intersections[count] = first
                count += 1
            if low_kf < second < high_kf and abs(second - first) > 1e-14:
                intersections[count] = second
                count += 1
        for boundary in energy_edges:
            value = np.sqrt(max(ei - boundary, 0.0) / ENERGY_TO_K2)
            if low_kf < value < high_kf:
                intersections[count] = value
                count += 1
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
            q_value = np.sqrt(
                max(
                    0.0,
                    ki * ki
                    + middle * middle
                    - 2.0 * ki * middle * cosine,
                )
            )
            energy = ei - ENERGY_TO_K2 * middle * middle
            q_index = np.searchsorted(q_edges, q_value, side="right") - 1
            energy_index = np.searchsorted(energy_edges, energy, side="right") - 1
            if q_value == q_edges[-1]:
                q_index = q_edges.size - 2
            if energy == energy_edges[-1]:
                energy_index = energy_edges.size - 2
            if (
                0 <= q_index < q_edges.size - 1
                and 0 <= energy_index < energy_edges.size - 1
            ):
                flat = q_index * shape[1] + energy_index
                partial[thread, flat] += (
                    weight * ENERGY_TO_K2 * (second * second - first * first)
                )
    return np.sum(partial, axis=0)


def run_powder_trajectory_normalization(*args, workers: int):
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        return powder_trajectory_normalization(*args)
    finally:
        set_num_threads(previous)
