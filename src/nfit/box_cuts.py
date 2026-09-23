"""Profiles along the principal axes of a rectangular histogram selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxProfiles:
    x: tuple[np.ndarray, np.ndarray, np.ndarray]
    y: tuple[np.ndarray, np.ndarray, np.ndarray]
    selected: np.ndarray


def box_corners(extents: tuple[float, float, float, float], angle: float) -> np.ndarray:
    """Return the four box corners in displayed data coordinates."""

    x0, x1, y0, y1 = extents
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    theta = np.deg2rad(float(angle))
    c, s = np.cos(theta), np.sin(theta)
    local = np.array(
        ((x0 - cx, y0 - cy), (x1 - cx, y0 - cy), (x1 - cx, y1 - cy), (x0 - cx, y1 - cy))
    )
    return local @ np.array(((c, s), (-s, c))) + (cx, cy)


def box_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    extents: tuple[float, float, float, float],
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates parallel to the rotated box sides.

    The origins are the box center projected onto the original x and y axes,
    so an unrotated box retains the original axis coordinates.
    """

    x0, x1, y0, y1 = extents
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    theta = np.deg2rad(float(angle))
    dx, dy = np.asarray(x, dtype=float) - cx, np.asarray(y, dtype=float) - cy
    return cx + dx * np.cos(theta) + dy * np.sin(theta), cy - dx * np.sin(theta) + dy * np.cos(
        theta
    )


def box_membership(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    extents: tuple[float, float, float, float],
    angle: float = 0.0,
) -> np.ndarray:
    """Select pixel centers inside the box in its own coordinate system."""

    x0, x1, y0, y1 = extents
    u, v = box_coordinates(
        np.asarray(x_centers)[None, :], np.asarray(y_centers)[:, None], extents, angle
    )
    return (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)


def _profile(
    coordinates: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    coverage: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2
    index = np.searchsorted(edges, coordinates[selected], side="right") - 1
    index = np.clip(index, 0, bins - 1)
    good = selected & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    good_index = np.searchsorted(edges, coordinates[good], side="right") - 1
    good_index = np.clip(good_index, 0, bins - 1)
    weights = 1 / np.square(errors[good])
    denominator = np.bincount(good_index, weights=weights, minlength=bins)
    numerator = np.bincount(good_index, weights=weights * values[good], minlength=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = numerator / denominator
        uncertainty = np.sqrt(1 / denominator)
    means[denominator == 0] = np.nan
    uncertainty[denominator == 0] = np.nan
    if threshold > 0:
        cover = np.bincount(index, weights=coverage[selected], minlength=bins)
        counts = np.bincount(index, minlength=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            insufficient = cover / counts < threshold
        means[insufficient] = np.nan
        uncertainty[insufficient] = np.nan
    return centers, means, uncertainty


def rotated_box_profiles(
    view: dict[str, np.ndarray],
    values: np.ndarray,
    errors: np.ndarray,
    extents: tuple[float, float, float, float],
    angle: float = 0.0,
    *,
    coverage_threshold: float = 0.0,
) -> BoxProfiles:
    """Inverse-variance weighted cuts along a box's two principal axes.

    Bin centers determine membership. For a rotated box, each selected pixel
    contributes to one projected bin on each axis. The projected pitch follows
    the source grid resolution, keeping the result bounded by 4096 bins.
    """

    x = np.asarray(view["x_centers"], dtype=float)
    y = np.asarray(view["y_centers"], dtype=float)
    z = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    coverage = np.asarray(view["coverage_fraction"], dtype=float)
    if z.shape != (len(y), len(x)) or e.shape != z.shape:
        raise ValueError("box profiles require matching 2D values and errors")
    selected = box_membership(x, y, extents, angle)
    u, v = box_coordinates(x[None, :], y[:, None], extents, angle)
    u = np.broadcast_to(u, z.shape)
    v = np.broadcast_to(v, z.shape)
    theta = np.deg2rad(float(angle))
    xstep = float(np.median(np.diff(x))) if len(x) > 1 else abs(extents[1] - extents[0])
    ystep = float(np.median(np.diff(y))) if len(y) > 1 else abs(extents[3] - extents[2])
    upitch = np.hypot(xstep * np.cos(theta), ystep * np.sin(theta))
    vpitch = np.hypot(xstep * np.sin(theta), ystep * np.cos(theta))

    def edges(low: float, high: float, pitch: float) -> np.ndarray:
        count = int(np.clip(np.ceil((high - low) / max(abs(pitch), 1e-12)), 1, 4096))
        return np.linspace(low, high, count + 1)

    x_edges = edges(extents[0], extents[1], upitch)
    y_edges = edges(extents[2], extents[3], vpitch)
    return BoxProfiles(
        x=_profile(u, z, e, coverage, selected, x_edges, coverage_threshold),
        y=_profile(v, z, e, coverage, selected, y_edges, coverage_threshold),
        selected=selected,
    )


def rotated_box_sum_profile(
    view: dict[str, np.ndarray],
    values: np.ndarray,
    extents: tuple[float, float, float, float],
    angle: float,
    centers: np.ndarray,
    *,
    axis: str,
) -> np.ndarray:
    """Sum values in the same projected bins as a rotated weighted cut."""

    x = np.asarray(view["x_centers"], dtype=float)
    y = np.asarray(view["y_centers"], dtype=float)
    selected = box_membership(x, y, extents, angle)
    u, v = box_coordinates(x[None, :], y[:, None], extents, angle)
    coords = np.broadcast_to(u if axis == "x" else v, selected.shape)
    values = np.asarray(values, dtype=float)
    if len(centers) > 1:
        mids = (centers[:-1] + centers[1:]) / 2
        edges = np.r_[
            centers[0] - (mids[0] - centers[0]), mids, centers[-1] + (centers[-1] - mids[-1])
        ]
    else:
        low, high = (extents[0], extents[1]) if axis == "x" else (extents[2], extents[3])
        edges = np.array([low, high])
    good = selected & np.isfinite(values)
    indices = np.clip(np.searchsorted(edges, coords[good], side="right") - 1, 0, len(centers) - 1)
    return np.bincount(indices, weights=values[good], minlength=len(centers))
