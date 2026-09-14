"""Fused finite-energy CORELLI cross-correlation kernels."""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

ENERGY_TO_K2 = 2.072124855
TOF_FACTOR = 2286.271549


@njit(inline="always")
def _t0(energy):
    return 101.9 * energy**-0.41 * math.exp(-energy / 282.0)


@njit(inline="always")
def _incident_energy(tof, l2, delta_e, l1, energy_min, energy_max):
    lower = max(energy_min, delta_e + 1.0e-6)
    if lower >= energy_max or tof <= 0.0:
        return -1.0
    energy = (TOF_FACTOR * (l1 + l2) / tof) ** 2 + delta_e * l2 / (l1 + l2)
    energy = min(max(energy, lower), energy_max)
    for _ in range(7):
        final = max(energy - delta_e, 1.0e-8)
        calculated = TOF_FACTOR * (l1 / math.sqrt(energy) + l2 / math.sqrt(final))
        derivative = -0.5 * TOF_FACTOR * (l1 / energy**1.5 + l2 / final**1.5)
        if abs(derivative) <= 1.0e-12:
            return -1.0
        energy = min(max(energy - (calculated - tof) / derivative, lower), energy_max)
    final = energy - delta_e
    if final <= 0.0:
        return -1.0
    calculated = TOF_FACTOR * (l1 / math.sqrt(energy) + l2 / math.sqrt(final))
    if not math.isfinite(calculated) or abs(calculated - tof) > 0.05:
        return -1.0
    return energy


@njit(inline="always")
def _correlation_weight(candidate_ns, tdc_ns, period_ns, sequence, duty, offset_ns):
    tdc_index = np.searchsorted(tdc_ns, candidate_ns - offset_ns, side="right") - 1
    if tdc_index < 0 or tdc_index >= tdc_ns.size:
        return 0.0, False
    angle = 360.0 * (candidate_ns - offset_ns - tdc_ns[tdc_index]) / period_ns
    if angle < 0.0 or angle > sequence[-1]:
        return 0.0, False
    interval = np.searchsorted(sequence, angle, side="left")
    if interval % 2 == 0:
        return -duty / (1.0 - duty), True
    return 1.0, True


@njit(inline="always")
def _flux_density(row, momentum, flux_axis, cumulative):
    if row < 0 or cumulative.shape[0] == 0:
        return 1.0
    if momentum < flux_axis[0] or momentum > flux_axis[-1]:
        return 0.0
    index = np.searchsorted(flux_axis, momentum, side="right")
    index = min(max(index, 1), flux_axis.size - 1)
    left = max(0, index - 1)
    right = min(flux_axis.size - 1, index + 1)
    delta = flux_axis[right] - flux_axis[left]
    if delta <= 0.0:
        return 0.0
    return max(0.0, (cumulative[row, right] - cumulative[row, left]) / delta)


@njit(inline="always")
def _event_weight(correlation, ki, kf, exponent, solid, flux, use_he3, use_ki_kf):
    normalization = solid * flux
    if not math.isfinite(normalization) or normalization <= 0.0:
        return 0.0, False
    weight = correlation / normalization
    if use_he3 and exponent > 0.0 and kf > 0.0:
        wavelength = 2.0 * math.pi / kf
        efficiency = -math.expm1(-exponent * wavelength)
        if efficiency <= 0.0:
            return 0.0, False
        weight /= efficiency
    if use_ki_kf:
        if kf <= 0.0:
            return 0.0, False
        weight *= ki / kf
    return weight, math.isfinite(weight)


@njit(inline="always")
def _bin_index(value, edges):
    index = np.searchsorted(edges, value, side="right") - 1
    if value == edges[-1]:
        index = edges.size - 2
    return index


@njit(inline="always")
def _axis_assignment(value, edges, fractional):
    """Return neighboring-bin indices and the upper-bin fraction."""

    count = edges.size - 1
    if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
        return -1, -1, 0.0, False
    if not fractional or count == 1:
        index = _bin_index(value, edges)
        return index, index, 0.0, 0 <= index < count
    first = 0.5 * (edges[0] + edges[1])
    last = 0.5 * (edges[-2] + edges[-1])
    if value <= first:
        return 0, 0, 0.0, True
    if value >= last:
        return count - 1, count - 1, 0.0, True
    left = 0
    right = count - 1
    while left + 1 < right:
        middle = (left + right) // 2
        centre = 0.5 * (edges[middle] + edges[middle + 1])
        if value < centre:
            right = middle
        else:
            left = middle
    lower_centre = 0.5 * (edges[left] + edges[left + 1])
    upper_centre = 0.5 * (edges[left + 1] + edges[left + 2])
    upper_fraction = (value - lower_centre) / (upper_centre - lower_centre)
    return left, left + 1, upper_fraction, True


@njit(inline="always")
def _deposit_powder(values, variances, counts, q, energy_index, q_edges, energy_count, fractional, weight):
    lower, upper, upper_fraction, valid = _axis_assignment(
        q, q_edges, fractional
    )
    if not valid:
        return
    choices = 2 if upper != lower else 1
    for choice in range(choices):
        if choice:
            index = upper
            spatial = upper_fraction
        else:
            index = lower
            spatial = 1.0 - upper_fraction
        contribution = spatial * weight
        flat = index * energy_count + energy_index
        values[flat] += contribution
        variances[flat] += contribution * contribution
        counts[flat] += spatial


@njit(inline="always")
def _deposit_hkle(
    values,
    variances,
    counts,
    c0,
    c1,
    c2,
    energy_index,
    edge0,
    edge1,
    edge2,
    shape,
    fractional_axes,
    weight,
):
    low0, high0, fraction0, valid0 = _axis_assignment(
        c0, edge0, fractional_axes[0]
    )
    low1, high1, fraction1, valid1 = _axis_assignment(
        c1, edge1, fractional_axes[1]
    )
    low2, high2, fraction2, valid2 = _axis_assignment(
        c2, edge2, fractional_axes[2]
    )
    if not (valid0 and valid1 and valid2):
        return
    choices0 = 2 if high0 != low0 else 1
    choices1 = 2 if high1 != low1 else 1
    choices2 = 2 if high2 != low2 else 1
    for choice0 in range(choices0):
        index0 = high0 if choice0 else low0
        spatial0 = fraction0 if choice0 else 1.0 - fraction0
        for choice1 in range(choices1):
            index1 = high1 if choice1 else low1
            spatial1 = fraction1 if choice1 else 1.0 - fraction1
            for choice2 in range(choices2):
                index2 = high2 if choice2 else low2
                spatial2 = fraction2 if choice2 else 1.0 - fraction2
                spatial = spatial0 * spatial1 * spatial2
                contribution = spatial * weight
                flat = (
                    ((index0 * shape[1] + index1) * shape[2] + index2)
                    * shape[3]
                    + energy_index
                )
                values[flat] += contribution
                variances[flat] += contribution * contribution
                counts[flat] += spatial


@njit(fastmath=False, nogil=True, parallel=True)
def powder_kernel(
    tofs,
    pulse_ns,
    directions,
    l2,
    he3,
    solid,
    flux_rows,
    delta_e_centres,
    l1,
    source_to_chopper,
    energy_min,
    energy_max,
    tdc_ns,
    period_ns,
    sequence,
    duty,
    timing_offset_ns,
    flux_axis,
    cumulative_flux,
    q_edges,
    energy_edges,
    fractional_axes,
    use_he3,
    use_ki_kf,
):
    workers = get_num_threads()
    output_size = (q_edges.size - 1) * (energy_edges.size - 1)
    values = np.zeros((workers, output_size))
    variances = np.zeros((workers, output_size))
    counts = np.zeros((workers, output_size))
    for event in prange(tofs.size):
        thread = get_thread_id()
        for energy_index in range(delta_e_centres.size):
            delta_e = delta_e_centres[energy_index]
            ei = _incident_energy(tofs[event], l2[event], delta_e, l1, energy_min, energy_max)
            if ei <= 0.0:
                continue
            ef = ei - delta_e
            ki = math.sqrt(ei / ENERGY_TO_K2)
            kf = math.sqrt(ef / ENERGY_TO_K2)
            crossing_us = _t0(ei) * (1.0 - source_to_chopper / (l1 + l2[event])) + TOF_FACTOR * source_to_chopper / math.sqrt(ei)
            correlation, valid = _correlation_weight(
                pulse_ns[event] + crossing_us * 1000.0,
                tdc_ns,
                period_ns,
                sequence,
                duty,
                timing_offset_ns,
            )
            if not valid:
                continue
            flux = _flux_density(flux_rows[event], ki, flux_axis, cumulative_flux)
            weight, valid = _event_weight(correlation, ki, kf, he3[event], solid[event], flux, use_he3, use_ki_kf)
            if not valid:
                continue
            cosine = directions[event, 2]
            q = math.sqrt(max(0.0, ki * ki + kf * kf - 2.0 * ki * kf * cosine))
            _deposit_powder(
                values[thread],
                variances[thread],
                counts[thread],
                q,
                energy_index,
                q_edges,
                energy_edges.size - 1,
                fractional_axes[0],
                weight,
            )
    return np.sum(values, axis=0), np.sum(variances, axis=0), np.sum(counts, axis=0)


@njit(fastmath=False, nogil=True, parallel=True)
def hkle_kernel(
    tofs,
    pulse_ns,
    directions,
    l2,
    he3,
    solid,
    flux_rows,
    delta_e_centres,
    l1,
    source_to_chopper,
    energy_min,
    energy_max,
    tdc_ns,
    period_ns,
    sequence,
    duty,
    timing_offset_ns,
    flux_axis,
    cumulative_flux,
    q_to_hkl,
    basis_inverse,
    symmetry,
    edge0,
    edge1,
    edge2,
    energy_edges,
    shape,
    fractional_axes,
    use_he3,
    use_ki_kf,
):
    workers = get_num_threads()
    output_size = shape[0] * shape[1] * shape[2] * shape[3]
    values = np.zeros((workers, output_size))
    variances = np.zeros((workers, output_size))
    counts = np.zeros((workers, output_size))
    for event in prange(tofs.size):
        thread = get_thread_id()
        for energy_index in range(delta_e_centres.size):
            delta_e = delta_e_centres[energy_index]
            ei = _incident_energy(tofs[event], l2[event], delta_e, l1, energy_min, energy_max)
            if ei <= 0.0:
                continue
            ef = ei - delta_e
            ki = math.sqrt(ei / ENERGY_TO_K2)
            kf = math.sqrt(ef / ENERGY_TO_K2)
            crossing_us = _t0(ei) * (1.0 - source_to_chopper / (l1 + l2[event])) + TOF_FACTOR * source_to_chopper / math.sqrt(ei)
            correlation, valid = _correlation_weight(
                pulse_ns[event] + crossing_us * 1000.0,
                tdc_ns,
                period_ns,
                sequence,
                duty,
                timing_offset_ns,
            )
            if not valid:
                continue
            flux = _flux_density(flux_rows[event], ki, flux_axis, cumulative_flux)
            weight, valid = _event_weight(correlation, ki, kf, he3[event], solid[event], flux, use_he3, use_ki_kf)
            if not valid:
                continue
            q0 = -kf * directions[event, 0]
            q1 = -kf * directions[event, 1]
            q2 = ki - kf * directions[event, 2]
            h = q0 * q_to_hkl[0, 0] + q1 * q_to_hkl[1, 0] + q2 * q_to_hkl[2, 0]
            k = q0 * q_to_hkl[0, 1] + q1 * q_to_hkl[1, 1] + q2 * q_to_hkl[2, 1]
            l = q0 * q_to_hkl[0, 2] + q1 * q_to_hkl[1, 2] + q2 * q_to_hkl[2, 2]
            for operation in range(symmetry.shape[0]):
                sh = h * symmetry[operation, 0, 0] + k * symmetry[operation, 0, 1] + l * symmetry[operation, 0, 2]
                sk = h * symmetry[operation, 1, 0] + k * symmetry[operation, 1, 1] + l * symmetry[operation, 1, 2]
                sl = h * symmetry[operation, 2, 0] + k * symmetry[operation, 2, 1] + l * symmetry[operation, 2, 2]
                c0 = sh * basis_inverse[0, 0] + sk * basis_inverse[1, 0] + sl * basis_inverse[2, 0]
                c1 = sh * basis_inverse[0, 1] + sk * basis_inverse[1, 1] + sl * basis_inverse[2, 1]
                c2 = sh * basis_inverse[0, 2] + sk * basis_inverse[1, 2] + sl * basis_inverse[2, 2]
                _deposit_hkle(
                    values[thread],
                    variances[thread],
                    counts[thread],
                    c0,
                    c1,
                    c2,
                    energy_index,
                    edge0,
                    edge1,
                    edge2,
                    shape,
                    fractional_axes,
                    weight,
                )
    return np.sum(values, axis=0), np.sum(variances, axis=0), np.sum(counts, axis=0)


@njit(fastmath=False, nogil=True, parallel=True)
def hkle_energy_kernel(
    tofs,
    pulse_ns,
    directions,
    l2,
    he3,
    solid,
    flux_rows,
    delta_e_centres,
    l1,
    source_to_chopper,
    energy_min,
    energy_max,
    tdc_ns,
    period_ns,
    sequence,
    duty,
    timing_offset_ns,
    flux_axis,
    cumulative_flux,
    q_to_hkl,
    basis_inverse,
    symmetry,
    edge0,
    edge1,
    edge2,
    energy_edges,
    shape,
    fractional_axes,
    use_he3,
    use_ki_kf,
):
    """Accumulate independent energy channels into disjoint output slices."""

    energy_count = delta_e_centres.size
    output_size = shape[0] * shape[1] * shape[2] * shape[3]
    values = np.zeros(output_size)
    variances = np.zeros(output_size)
    counts = np.zeros(output_size)
    for energy_index in prange(energy_count):
        delta_e = delta_e_centres[energy_index]
        for event in range(tofs.size):
            ei = _incident_energy(
                tofs[event], l2[event], delta_e, l1, energy_min, energy_max
            )
            if ei <= 0.0:
                continue
            ef = ei - delta_e
            ki = math.sqrt(ei / ENERGY_TO_K2)
            kf = math.sqrt(ef / ENERGY_TO_K2)
            crossing_us = (
                _t0(ei) * (1.0 - source_to_chopper / (l1 + l2[event]))
                + TOF_FACTOR * source_to_chopper / math.sqrt(ei)
            )
            correlation, valid = _correlation_weight(
                pulse_ns[event] + crossing_us * 1000.0,
                tdc_ns,
                period_ns,
                sequence,
                duty,
                timing_offset_ns,
            )
            if not valid:
                continue
            flux = _flux_density(
                flux_rows[event], ki, flux_axis, cumulative_flux
            )
            weight, valid = _event_weight(
                correlation,
                ki,
                kf,
                he3[event],
                solid[event],
                flux,
                use_he3,
                use_ki_kf,
            )
            if not valid:
                continue
            q0 = -kf * directions[event, 0]
            q1 = -kf * directions[event, 1]
            q2 = ki - kf * directions[event, 2]
            h = (
                q0 * q_to_hkl[0, 0]
                + q1 * q_to_hkl[1, 0]
                + q2 * q_to_hkl[2, 0]
            )
            k = (
                q0 * q_to_hkl[0, 1]
                + q1 * q_to_hkl[1, 1]
                + q2 * q_to_hkl[2, 1]
            )
            l = (
                q0 * q_to_hkl[0, 2]
                + q1 * q_to_hkl[1, 2]
                + q2 * q_to_hkl[2, 2]
            )
            for operation in range(symmetry.shape[0]):
                sh = (
                    h * symmetry[operation, 0, 0]
                    + k * symmetry[operation, 0, 1]
                    + l * symmetry[operation, 0, 2]
                )
                sk = (
                    h * symmetry[operation, 1, 0]
                    + k * symmetry[operation, 1, 1]
                    + l * symmetry[operation, 1, 2]
                )
                sl = (
                    h * symmetry[operation, 2, 0]
                    + k * symmetry[operation, 2, 1]
                    + l * symmetry[operation, 2, 2]
                )
                c0 = (
                    sh * basis_inverse[0, 0]
                    + sk * basis_inverse[1, 0]
                    + sl * basis_inverse[2, 0]
                )
                c1 = (
                    sh * basis_inverse[0, 1]
                    + sk * basis_inverse[1, 1]
                    + sl * basis_inverse[2, 1]
                )
                c2 = (
                    sh * basis_inverse[0, 2]
                    + sk * basis_inverse[1, 2]
                    + sl * basis_inverse[2, 2]
                )
                _deposit_hkle(
                    values,
                    variances,
                    counts,
                    c0,
                    c1,
                    c2,
                    energy_index,
                    edge0,
                    edge1,
                    edge2,
                    shape,
                    fractional_axes,
                    weight,
                )
    return values, variances, counts


def _run(kernel, *args, workers):
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        return kernel(*args)
    finally:
        set_num_threads(previous)


def run_powder(*args, workers):
    return _run(powder_kernel, *args, workers=workers)


def run_hkle(*args, workers):
    kernel = hkle_energy_kernel if np.asarray(args[7]).size > 1 else hkle_kernel
    return _run(kernel, *args, workers=workers)
