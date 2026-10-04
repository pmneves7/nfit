"""Ordered fused projection and event accumulation for both DGS precision modes."""

import numpy as np
from numba import njit

from ._event_bin_numba import physical_bin_index, uniform_edge_reciprocals


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_uniform_projected_events(
    q_sample, energy, weights, variances, enabled, matrix, offset, shape,
    data_sum, variance_sum, event_count, bin_indices=None,
):
    """Fuse projection with assignment, preserving event and float32 stage order.

    The supplied affine already expresses positions in unit-width bin
    coordinates. The upper boundary is excluded, matching the prepared Mantid
    projector. No matrix products are combined and no fast-math is used.
    """
    for row in range(energy.size):
        if enabled is not None and not enabled[row]:
            continue
        q0 = np.float32(q_sample[row, 0])
        q1 = np.float32(q_sample[row, 1])
        q2 = np.float32(q_sample[row, 2])
        e = np.float32(energy[row])
        flat = 0
        valid = True
        for dim in range(4):
            value = np.float32(0.)
            value = np.float32(value + np.float32(q0 * matrix[dim, 0]))
            value = np.float32(value + np.float32(q1 * matrix[dim, 1]))
            value = np.float32(value + np.float32(q2 * matrix[dim, 2]))
            value = np.float32(value + np.float32(e * matrix[dim, 3]))
            value = np.float32(value + offset[dim])
            if not np.isfinite(value) or value < 0. or value >= shape[dim]:
                valid = False
                break
            flat = flat * shape[dim] + int(value)
        if valid:
            if bin_indices is not None:
                bin_indices[row] = flat
            weight = weights[row]
            variance = weight * weight if variances is None else variances[row]
            data_sum[flat] += weight
            variance_sum[flat] += variance
            event_count[flat] += 1.


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_high_precision_projected_events(
    q_sample, energy, weights, variances, enabled, hkl_transform, operation, inverse_basis, edges, shape,
    data_sum, variance_sum, event_count, bin_indices=None,
):
    """Retain the three float64 transform stages without event-sized temporaries.

    Matrix stages are not combined, fast-math is disabled, and events accumulate
    in source order. BLAS and scalar float64 dot products can differ by roundoff.
    Membership is checked against the requested physical edges, including the
    outermost upper edge, using the shared discrete accumulator's lookup.
    """
    reciprocal_steps = uniform_edge_reciprocals(edges)
    for row in range(energy.size):
        if enabled is not None and not enabled[row]:
            continue
        q0, q1, q2 = q_sample[row, 0], q_sample[row, 1], q_sample[row, 2]
        h0 = (q0*hkl_transform[0, 0] + q1*hkl_transform[1, 0]) + q2*hkl_transform[2, 0]
        h1 = (q0*hkl_transform[0, 1] + q1*hkl_transform[1, 1]) + q2*hkl_transform[2, 1]
        h2 = (q0*hkl_transform[0, 2] + q1*hkl_transform[1, 2]) + q2*hkl_transform[2, 2]
        s0 = (h0*operation[0, 0] + h1*operation[1, 0]) + h2*operation[2, 0]
        s1 = (h0*operation[0, 1] + h1*operation[1, 1]) + h2*operation[2, 1]
        s2 = (h0*operation[0, 2] + h1*operation[1, 2]) + h2*operation[2, 2]
        flat = 0
        valid = True
        for dim in range(4):
            value = ((s0*inverse_basis[0, dim] + s1*inverse_basis[1, dim])
                     + s2*inverse_basis[2, dim]) + energy[row]*inverse_basis[3, dim]
            index = physical_bin_index(value, edges[dim], reciprocal_steps[dim])
            if index < 0 or index >= shape[dim]:
                valid = False
                break
            flat = flat * shape[dim] + index
        if valid:
            if bin_indices is not None:
                bin_indices[row] = flat
            weight = weights[row]
            variance = weight * weight if variances is None else variances[row]
            data_sum[flat] += weight
            variance_sum[flat] += variance
            event_count[flat] += 1.
