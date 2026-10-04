"""Shared compiled physical-edge event membership with uniform-grid acceleration."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def uniform_edge_reciprocals(edges):
    """Find uniform edges without changing any saved physical boundary."""
    reciprocal_steps = np.zeros(len(edges))
    for dim in range(len(edges)):
        edge = edges[dim]
        step = (edge[-1] - edge[0]) / (edge.size - 1)
        tolerance = 16.0 * np.finfo(np.float64).eps * max(abs(edge[0]), abs(edge[-1]), abs(step))
        uniform = step > 0.0 and np.isfinite(1.0 / step)
        for index in range(edge.size):
            if abs(edge[index] - (edge[0] + index * step)) > tolerance:
                uniform = False
                break
        if uniform:
            reciprocal_steps[dim] = 1.0 / step
    return reciprocal_steps


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def physical_bin_index(value, edge, reciprocal_step):
    """Include the outermost upper edge and correct uniform lookup to saved edges."""
    if not np.isfinite(value) or value < edge[0] or value > edge[-1]:
        return -1
    if reciprocal_step > 0.0:
        index = min(int((value - edge[0]) * reciprocal_step), edge.size - 2)
        while index > 0 and value < edge[index]:
            index -= 1
        while index < edge.size - 2 and value >= edge[index + 1]:
            index += 1
        return index
    index = np.searchsorted(edge, value, side="right") - 1
    if value == edge[-1]:
        index = edge.size - 2
    return index
