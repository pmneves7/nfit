"""Shared discrete event membership for arbitrary physical histogram edges."""

import numpy as np


def flat_bin_indices(coords, edges, shape):
    """Return flattened memberships; include the outermost upper edge."""
    indices = []
    valid = np.ones(coords.shape[0], dtype=bool)
    for dim, edge in enumerate(edges):
        index = np.searchsorted(edge, coords[:, dim], side="right") - 1
        index[coords[:, dim] == edge[-1]] = len(edge) - 2
        valid &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    flat = np.full(coords.shape[0], -1, dtype=np.int64)
    flat[valid] = np.ravel_multi_index(tuple(index[valid] for index in indices), shape)
    return flat
