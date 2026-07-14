"""Optional fused, streaming CPU kernels for N-dimensional rebinning."""

from __future__ import annotations

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict


def sparse_accumulators():
    """Create independent typed maps for one sparse worker partial."""

    return tuple(
        Dict.empty(key_type=types.int64, value_type=types.float64)
        for _ in range(4)
    )


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_batch(
    coords,
    data,
    errors,
    statistical_weights,
    lower,
    upper,
    step_size,
    num_bins,
    fractional,
    inverse_variance,
    bd_sum,
    err_sum,
    norm_sum,
    ns_sum,
):
    """Accumulate one source batch without contribution or histogram temporaries."""

    n_points, n_dims = coords.shape
    positions = np.empty(n_dims, dtype=np.float64)
    base_indices = np.empty(n_dims, dtype=np.int64)
    partial_weights = np.empty(n_dims, dtype=np.float64)
    edge_valid = np.empty(n_dims, dtype=np.bool_)
    contribution_indices = np.empty(n_dims, dtype=np.int64)
    offset_count = 1 << n_dims if fractional else 1

    for point in range(n_points):
        point_valid = True
        for dim in range(n_dims):
            coordinate = coords[point, dim]
            if not np.isfinite(coordinate) or coordinate < lower[dim] or coordinate > upper[dim]:
                point_valid = False
                break
            if np.isinf(step_size[dim]):
                position = 0.0
            elif coordinate == upper[dim]:
                position = float(num_bins[dim] - 1)
            else:
                position = (coordinate - lower[dim]) / step_size[dim]
            positions[dim] = position - 0.5 if fractional else position
        if not point_valid:
            continue

        statistical_weight = statistical_weights[point]
        if not np.isfinite(statistical_weight) or statistical_weight <= 0.0:
            continue
        if inverse_variance:
            error = errors[point]
            if not np.isfinite(error) or error <= 0.0:
                continue
            point_mean_weight = statistical_weight / (error * error)
        else:
            error = errors[point]
            point_mean_weight = statistical_weight

        if fractional:
            for dim in range(n_dims):
                position = positions[dim]
                floor_position = np.floor(position)
                base_indices[dim] = int(floor_position)
                partial_weights[dim] = 1.0 - (position - floor_position)
                edge_valid[dim] = position >= 0.0 and position <= num_bins[dim] - 1
        else:
            for dim in range(n_dims):
                base_indices[dim] = int(positions[dim])

        for offset_bits in range(offset_count):
            spatial_weight = 1.0
            contribution_valid = True
            flat_index = 0
            for dim in range(n_dims):
                if fractional and ((offset_bits >> dim) & 1):
                    if not edge_valid[dim]:
                        contribution_valid = False
                        break
                    spatial_weight *= 1.0 - partial_weights[dim]
                    index = base_indices[dim] + 1
                elif fractional:
                    if edge_valid[dim]:
                        spatial_weight *= partial_weights[dim]
                    index = base_indices[dim]
                else:
                    index = base_indices[dim]
                if index < 0:
                    index = 0
                elif index >= num_bins[dim]:
                    index = num_bins[dim] - 1
                contribution_indices[dim] = index
            if not contribution_valid or spatial_weight == 0.0 or not np.isfinite(spatial_weight):
                continue
            for dim in range(n_dims):
                flat_index = flat_index * num_bins[dim] + contribution_indices[dim]

            if inverse_variance:
                mean_weight = spatial_weight * point_mean_weight
                bd_sum[flat_index] += mean_weight * data[point]
                err_sum[flat_index] += mean_weight
                norm_sum[flat_index] += mean_weight
            else:
                normalization_weight = spatial_weight * point_mean_weight
                bd_sum[flat_index] += normalization_weight * data[point]
                err_sum[flat_index] += normalization_weight * normalization_weight * error * error
                norm_sum[flat_index] += normalization_weight
            ns_sum[flat_index] += spatial_weight


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_batch_sparse(
    coords, data, errors, statistical_weights, lower, upper, step_size,
    num_bins, fractional, inverse_variance, bd_sum, err_sum, norm_sum, ns_sum,
):
    """Sparse counterpart of :func:`accumulate_batch`, keyed by touched bins."""

    n_points, n_dims = coords.shape
    positions = np.empty(n_dims, dtype=np.float64)
    base_indices = np.empty(n_dims, dtype=np.int64)
    partial_weights = np.empty(n_dims, dtype=np.float64)
    edge_valid = np.empty(n_dims, dtype=np.bool_)
    contribution_indices = np.empty(n_dims, dtype=np.int64)
    offset_count = 1 << n_dims if fractional else 1
    for point in range(n_points):
        point_valid = True
        for dim in range(n_dims):
            coordinate = coords[point, dim]
            if not np.isfinite(coordinate) or coordinate < lower[dim] or coordinate > upper[dim]:
                point_valid = False
                break
            if np.isinf(step_size[dim]):
                position = 0.0
            elif coordinate == upper[dim]:
                position = float(num_bins[dim] - 1)
            else:
                position = (coordinate - lower[dim]) / step_size[dim]
            positions[dim] = position - 0.5 if fractional else position
        if not point_valid:
            continue
        statistical_weight = statistical_weights[point]
        if not np.isfinite(statistical_weight) or statistical_weight <= 0.0:
            continue
        if inverse_variance:
            error = errors[point]
            if not np.isfinite(error) or error <= 0.0:
                continue
            point_mean_weight = statistical_weight / (error * error)
        else:
            error = errors[point]
            point_mean_weight = statistical_weight
        for dim in range(n_dims):
            if fractional:
                position = positions[dim]
                floor_position = np.floor(position)
                base_indices[dim] = int(floor_position)
                partial_weights[dim] = 1.0 - (position - floor_position)
                edge_valid[dim] = position >= 0.0 and position <= num_bins[dim] - 1
            else:
                base_indices[dim] = int(positions[dim])
        for offset_bits in range(offset_count):
            spatial_weight = 1.0
            contribution_valid = True
            flat_index = 0
            for dim in range(n_dims):
                if fractional and ((offset_bits >> dim) & 1):
                    if not edge_valid[dim]:
                        contribution_valid = False
                        break
                    spatial_weight *= 1.0 - partial_weights[dim]
                    index = base_indices[dim] + 1
                elif fractional:
                    if edge_valid[dim]:
                        spatial_weight *= partial_weights[dim]
                    index = base_indices[dim]
                else:
                    index = base_indices[dim]
                if index < 0:
                    index = 0
                elif index >= num_bins[dim]:
                    index = num_bins[dim] - 1
                contribution_indices[dim] = index
            if not contribution_valid or spatial_weight == 0.0 or not np.isfinite(spatial_weight):
                continue
            for dim in range(n_dims):
                flat_index = flat_index * num_bins[dim] + contribution_indices[dim]
            if inverse_variance:
                mean_weight = spatial_weight * point_mean_weight
                bd_value = mean_weight * data[point]
                err_value = mean_weight
                norm_value = mean_weight
            else:
                norm_value = spatial_weight * point_mean_weight
                bd_value = norm_value * data[point]
                err_value = norm_value * norm_value * error * error
            bd_sum[flat_index] = bd_sum.get(flat_index, 0.0) + bd_value
            err_sum[flat_index] = err_sum.get(flat_index, 0.0) + err_value
            norm_sum[flat_index] = norm_sum.get(flat_index, 0.0) + norm_value
            ns_sum[flat_index] = ns_sum.get(flat_index, 0.0) + spatial_weight
