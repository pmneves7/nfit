"""Optional fused, streaming CPU kernels for N-dimensional rebinning."""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict

from ._event_bin_numba import physical_bin_index


def sparse_accumulators():
    """Create independent typed maps for one sparse worker partial."""

    return tuple(
        Dict.empty(key_type=types.int64, value_type=types.float64)
        for _ in range(4)
    )


@njit(cache=True, fastmath=False, nogil=True)
def merge_sparse_accumulator(source, target):
    """Add touched bins without boxing each typed-map item in Python."""

    for index, value in source.items():
        target[index] += value


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _offset_layout(num_bins, fractional_axes):
    """Resolve strides and neighbor offsets once per batch, not per point."""
    strides = np.empty(num_bins.size, dtype=np.int64)
    fractional_dims = np.empty(num_bins.size, dtype=np.int64)
    stride = 1
    for dim in range(num_bins.size - 1, -1, -1):
        strides[dim] = stride
        stride *= num_bins[dim]
    fractional_count = 0
    for dim in range(num_bins.size):
        if fractional_axes[dim] and num_bins[dim] > 1:
            fractional_dims[fractional_count] = dim
            fractional_count += 1
    offsets = np.zeros(1 << fractional_count, dtype=np.int64)
    for bits in range(offsets.size):
        for index in range(fractional_count):
            if (bits >> index) & 1:
                offsets[bits] += strides[fractional_dims[index]]
    return strides, fractional_dims[:fractional_count], offsets


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _point_location(coords, point, lower, upper, step_size, num_bins,
                    fractional_axes, strides, left_weights, right_weights, edges):
    """Resolve each axis once, saturating support at the accepted outer edges."""
    flat_index = 0
    for dim in range(coords.shape[1]):
        coordinate = coords[point, dim]
        if not np.isfinite(coordinate) or coordinate < lower[dim] or coordinate > upper[dim]:
            return -1
        if np.isinf(step_size[dim]):
            position = 0.0
        elif coordinate == upper[dim]:
            position = float(num_bins[dim]) - (0.5 if fractional_axes[dim] else 1.0)
        else:
            position = (coordinate - lower[dim]) / step_size[dim]
        if fractional_axes[dim] and num_bins[dim] > 1:
            position -= 0.5
            if edges is not None:
                # Exact reported centers must not acquire epsilon-sized
                # neighbors which become full-valued bins after normalization.
                candidate = min(max(int(np.floor(position + 0.5)), 0), num_bins[dim] - 1)
                edge = edges[0]
                center_index = edges[1][dim] + candidate
                if coordinate == (edge[center_index] + edge[center_index + 1]) * 0.5:
                    position = float(candidate)
            if position <= 0.0:
                index = 0
                right_weight = 0.0
            elif position >= num_bins[dim] - 1:
                index = num_bins[dim] - 1
                right_weight = 0.0
            else:
                index = int(np.floor(position))
                right_weight = position - index
            left_weights[dim] = 1.0 - right_weight
            right_weights[dim] = right_weight
        elif fractional_axes[dim]:
            index = 0
        elif edges is None:
            index = min(max(int(position), 0), num_bins[dim] - 1)
        else:
            reciprocal = 0.0 if np.isinf(step_size[dim]) else 1.0 / step_size[dim]
            index = physical_bin_index(coordinate, edges[0][edges[1][dim]:edges[1][dim + 1]], reciprocal)
        flat_index += index * strides[dim]
    return flat_index


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _spatial_weights(fractional_dims, left_weights, right_weights, weights):
    weights[0] = 1.0
    count = 1
    for dim in fractional_dims:
        for neighbor in range(count):
            previous = weights[neighbor]
            weights[neighbor + count] = previous * right_weights[dim]
            weights[neighbor] = previous * left_weights[dim]
        count *= 2


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_batch(
    coords, data, errors, statistical_weights, lower, upper, step_size,
    num_bins, fractional_axes, inverse_variance, bd_sum, err_sum, norm_sum, ns_sum, edges=None,
):
    """Deposit multilinear weights with ordered, diagonal-variance updates."""
    strides, fractional_dims, offsets = _offset_layout(num_bins, fractional_axes)
    left_weights = np.empty(num_bins.size, dtype=np.float64)
    right_weights = np.empty(num_bins.size, dtype=np.float64)
    spatial_weights = np.empty(offsets.size, dtype=np.float64)
    for point in range(coords.shape[0]):
        statistical_weight = statistical_weights[point]
        if not np.isfinite(statistical_weight) or statistical_weight <= 0.0:
            continue
        error = errors[point]
        if inverse_variance:
            if not np.isfinite(error) or error <= 0.0:
                continue
            point_mean_weight = statistical_weight / (error * error)
        else:
            point_mean_weight = statistical_weight
        base_index = _point_location(coords, point, lower, upper, step_size, num_bins,
                                     fractional_axes, strides, left_weights, right_weights, edges)
        if base_index < 0:
            continue
        _spatial_weights(fractional_dims, left_weights, right_weights, spatial_weights)
        for bits in range(offsets.size):
            spatial_weight = spatial_weights[bits]
            if spatial_weight == 0.0:
                continue
            flat_index = base_index + offsets[bits]
            mean_weight = spatial_weight * point_mean_weight
            bd_value = mean_weight * data[point]
            # Deterministic sharing squares the full coefficient. Only the
            # marginal variance is retained; neighboring covariance is omitted.
            err_value = mean_weight * mean_weight * error * error
            bd_sum[flat_index] += bd_value
            err_sum[flat_index] += err_value
            norm_sum[flat_index] += mean_weight
            ns_sum[flat_index] += spatial_weight


@njit(cache=True, fastmath=False, nogil=True)
def accumulate_batch_sparse(
    coords, data, errors, statistical_weights, lower, upper, step_size,
    num_bins, fractional_axes, inverse_variance, bd_sum, err_sum, norm_sum, ns_sum, edges=None,
):
    """Deposit multilinear weights with ordered, diagonal-variance updates."""
    strides, fractional_dims, offsets = _offset_layout(num_bins, fractional_axes)
    left_weights = np.empty(num_bins.size, dtype=np.float64)
    right_weights = np.empty(num_bins.size, dtype=np.float64)
    spatial_weights = np.empty(offsets.size, dtype=np.float64)
    for point in range(coords.shape[0]):
        statistical_weight = statistical_weights[point]
        if not np.isfinite(statistical_weight) or statistical_weight <= 0.0:
            continue
        error = errors[point]
        if inverse_variance:
            if not np.isfinite(error) or error <= 0.0:
                continue
            point_mean_weight = statistical_weight / (error * error)
        else:
            point_mean_weight = statistical_weight
        base_index = _point_location(coords, point, lower, upper, step_size, num_bins,
                                     fractional_axes, strides, left_weights, right_weights, edges)
        if base_index < 0:
            continue
        _spatial_weights(fractional_dims, left_weights, right_weights, spatial_weights)
        for bits in range(offsets.size):
            spatial_weight = spatial_weights[bits]
            if spatial_weight == 0.0:
                continue
            flat_index = base_index + offsets[bits]
            mean_weight = spatial_weight * point_mean_weight
            bd_value = mean_weight * data[point]
            # Deterministic sharing squares the full coefficient. Only the
            # marginal variance is retained; neighboring covariance is omitted.
            err_value = mean_weight * mean_weight * error * error
            bd_sum[flat_index] = bd_sum.get(flat_index, 0.0) + bd_value
            err_sum[flat_index] = err_sum.get(flat_index, 0.0) + err_value
            norm_sum[flat_index] = norm_sum.get(flat_index, 0.0) + mean_weight
            ns_sum[flat_index] = ns_sum.get(flat_index, 0.0) + spatial_weight
