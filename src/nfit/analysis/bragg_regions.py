"""GUI-independent joint voxel membership for Bragg integration regions."""
from __future__ import annotations

import numpy as np


def _box_overlap(edges, components, lower, upper):
    factors = []
    for axis_edges, component in zip(edges, components, strict=True):
        overlap = np.maximum(
            0.0, np.minimum(axis_edges[1:], upper[component])
            - np.maximum(axis_edges[:-1], lower[component]),
        )
        factors.append(overlap / np.diff(axis_edges))
    return factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]


def box_region_fraction(edges, components, center, widths):
    """Exact voxel fractions for a direct-axis HKL box."""
    return _box_overlap(edges, components, center - widths, center + widths)


def _subtract_box(box, excluded):
    """Partition a box minus another box into disjoint rectangular pieces."""
    lower, upper = box
    cut_lower = np.maximum(lower, excluded[0])
    cut_upper = np.minimum(upper, excluded[1])
    if np.any(cut_lower >= cut_upper):
        return [box]
    pieces = []
    middle_lower, middle_upper = lower.copy(), upper.copy()
    for dimension in range(3):
        if middle_lower[dimension] < cut_lower[dimension]:
            slab_upper = middle_upper.copy()
            slab_upper[dimension] = cut_lower[dimension]
            pieces.append((middle_lower.copy(), slab_upper))
            middle_lower[dimension] = cut_lower[dimension]
        if cut_upper[dimension] < middle_upper[dimension]:
            slab_lower = middle_lower.copy()
            slab_lower[dimension] = cut_upper[dimension]
            pieces.append((slab_lower, middle_upper.copy()))
            middle_upper[dimension] = cut_upper[dimension]
    return pieces


def box_shell_fraction(edges, components, center, widths, inner_scale, outer_scale, neighbors):
    """Exact shell-minus-neighbor union, including its full HKL volume."""
    outer = (center - widths * outer_scale, center + widths * outer_scale)
    pieces = _subtract_box(outer, (center - widths * inner_scale, center + widths * inner_scale))
    for neighbor in neighbors:
        excluded = (neighbor - widths, neighbor + widths)
        pieces = [part for box in pieces for part in _subtract_box(box, excluded)]
    fraction = np.zeros(tuple(len(edge) - 1 for edge in edges))
    measure = 0.0
    for lower, upper in pieces:
        fraction += _box_overlap(edges, components, lower, upper)
        measure += float(np.prod(upper - lower))
    return np.clip(fraction, 0.0, 1.0), measure


def sampled_region_fraction(
    edges, vectors, center, widths, rotation, samples, qmatrix, method,
    *, inner_scale=None, outer_scale=1.0, neighbors=(),
):
    """Use common subvoxel points for a region, shell, and all exclusions."""
    if not isinstance(samples, (int, np.integer)) or samples < 1 or samples % 2 != 1:
        raise ValueError("subvoxel_samples must be a positive odd integer")
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("ellipsoid_rotation must be an orthonormal 3x3 matrix")
    offsets = (np.arange(samples) + 0.5) / samples
    coordinates = [edge[:-1, None] + np.diff(edge)[:, None] * offsets for edge in edges]
    fraction = np.zeros(tuple(len(edge) - 1 for edge in edges))

    def inside(hkl, origin, scale):
        delta = hkl - origin
        if qmatrix is not None:
            delta = delta @ qmatrix.T
        if method == "box_sum":
            return np.all(np.abs(delta) <= widths * scale, axis=-1)
        return np.sum(((delta @ rotation) / (widths * scale)) ** 2, axis=-1) <= 1.0

    for i in range(samples):
        for j in range(samples):
            for k in range(samples):
                hkl = (
                    coordinates[0][:, i, None, None, None] * vectors[0]
                    + coordinates[1][None, :, j, None, None] * vectors[1]
                    + coordinates[2][None, None, :, k, None] * vectors[2]
                )
                selected = inside(hkl, center, outer_scale)
                if inner_scale is not None:
                    selected &= ~inside(hkl, center, inner_scale)
                for neighbor in neighbors:
                    selected &= ~inside(hkl, neighbor, 1.0)
                fraction += selected
    return fraction / samples**3
