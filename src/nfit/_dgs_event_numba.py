"""Ordered Mantid affine projection and uniform-grid event accumulation."""

import numpy as np
from numba import njit


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
