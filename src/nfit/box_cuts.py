"""Profiles along the principal axes of a rectangular histogram selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurement_profiles import MeasurementProfile, prepare_measurement_profile


@dataclass(frozen=True)
class BoxProfiles:
    x: tuple[np.ndarray, np.ndarray, np.ndarray]
    y: tuple[np.ndarray, np.ndarray, np.ndarray]
    selected: np.ndarray
    x_measurement: MeasurementProfile | None = None
    y_measurement: MeasurementProfile | None = None

    @property
    def value_label(self) -> str:
        if self.x_measurement is not None and self.x_measurement.data.metadata.get(
            "background_profile_uncertainty", ""
        ).startswith("diagonal_approximation"):
            return "Weighted mean (diagonal uncertainty)"
        if self.x_measurement is not None and self.x_measurement.contract.kind == "counting":
            return "Pooled intensity"
        return "Weighted mean"


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


@dataclass(frozen=True)
class BoxProfileSelection:
    """Center membership and output bins shared by previews and explicit replay."""

    selected: np.ndarray
    x_indices: np.ndarray
    y_indices: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray | None = None
    y_centers: np.ndarray | None = None
    x_coverage_weights: np.ndarray | None = None
    y_coverage_weights: np.ndarray | None = None


def box_profile_selection(view, extents, angle=0.0, *, force_rotated=False):
    """Build the regular/rotated center-selection rule without estimating data.

    Regular bins retain the displayed edges and centers. Rotated pitch follows
    the existing grid resolution and remains bounded to 4096 output bins.
    """
    x, y = np.asarray(view["x_centers"]), np.asarray(view["y_centers"])
    if len(extents) != 4 or not np.all(np.isfinite(extents)) or not np.isfinite(angle):
        raise ValueError("Box extents and angle must be finite")
    x0, x1, y0, y1 = extents
    if x1 < x0 or y1 < y0:
        raise ValueError("Box extents must be ordered")
    rotated = force_rotated or not np.isclose(float(angle) % 360, 0.0, atol=1e-10)
    if rotated and (x1 == x0 or y1 == y0):
        raise ValueError("Rotated box extents must have positive width and height")
    if not rotated:
        xm, ym = (x >= x0) & (x <= x1), (y >= y0) & (y <= y1)
        selected = ym[:, None] & xm[None, :]
        if not np.any(xm) or not np.any(ym):
            return BoxProfileSelection(selected, np.zeros(selected.shape, int),
                np.zeros(selected.shape, int), np.array([]), np.array([]))
        xe, ye = np.asarray(view["x_edges"]), np.asarray(view["y_edges"])
        xi, yi = np.flatnonzero(xm), np.flatnonzero(ym)
        x_edges, y_edges = xe[xi[0]:xi[-1]+2], ye[yi[0]:yi[-1]+2]
        u, v = np.broadcast_to(x[None, :], selected.shape), np.broadcast_to(y[:, None], selected.shape)
        x_centers, y_centers = x[xm], y[ym]
        x_weights, y_weights = np.diff(ye)[:, None], np.diff(xe)[None, :]
    else:
        selected = box_membership(x, y, extents, angle)
        u, v = box_coordinates(x[None, :], y[:, None], extents, angle)
        u, v = np.broadcast_to(u, selected.shape), np.broadcast_to(v, selected.shape)
        theta = np.deg2rad(float(angle))
        xstep = float(np.median(np.diff(x))) if len(x) > 1 else abs(x1-x0)
        ystep = float(np.median(np.diff(y))) if len(y) > 1 else abs(y1-y0)
        upitch = np.hypot(xstep * np.cos(theta), ystep * np.sin(theta))
        vpitch = np.hypot(xstep * np.sin(theta), ystep * np.cos(theta))
        def edges(low, high, pitch):
            count = int(np.clip(np.ceil((high-low) / max(abs(pitch), 1e-12)), 1, 4096))
            return np.linspace(low, high, count+1)
        x_edges, y_edges = edges(x0, x1, upitch), edges(y0, y1, vpitch)
        x_centers = y_centers = x_weights = y_weights = None
    x_indices = np.clip(np.searchsorted(x_edges, u, side="right")-1, 0, len(x_edges)-2)
    y_indices = np.clip(np.searchsorted(y_edges, v, side="right")-1, 0, len(y_edges)-2)
    return BoxProfileSelection(selected, x_indices, y_indices, x_edges, y_edges,
        x_centers, y_centers, x_weights, y_weights)


def _selected_profiles(view, values, errors, selection, coverage_threshold, channel, reference_values):
    z, e = np.asarray(values), np.asarray(errors)
    if z.shape != selection.selected.shape or e.shape != z.shape:
        raise ValueError("box profiles require matching 2D values and errors")
    if not len(selection.x_edges):
        empty = (np.array([]), np.array([]), np.array([]))
        return BoxProfiles(empty, empty, selection.selected)
    xp, yp = [prepare_measurement_profile(
        view, z, e, selected=selection.selected, indices=indices, edges=edges,
        centers=centers, coverage_weights=weights, coverage_threshold=coverage_threshold,
        channel=channel, reference_values=reference_values,
    ) for indices, edges, centers, weights in (
        (selection.x_indices, selection.x_edges, selection.x_centers, selection.x_coverage_weights),
        (selection.y_indices, selection.y_edges, selection.y_centers, selection.y_coverage_weights),
    )]
    return BoxProfiles(xp.arrays, yp.arrays, selection.selected, xp, yp)


def histogram_box_profiles(
    view, values, errors, extents, angle=0.0, *, coverage_threshold=0.0,
    channel="signal", reference_values=None,
) -> BoxProfiles:
    """Prepare regular or rotated cuts using the same measurement estimator.

    Primary count signals pool their validated numerator, variance and exposure.
    Other channels and legacy continuous signals retain precision means. Regular
    cuts retain source centers/edges; rotated cuts select centers and project them
    to the box axes. No subpixel or missing source covariance is invented.
    """
    selection = box_profile_selection(view, extents, angle)
    return _selected_profiles(view, values, errors, selection, coverage_threshold, channel, reference_values)


def rotated_box_profiles(
    view, values, errors, extents, angle=0.0, *, coverage_threshold=0.0,
    channel="signal", reference_values=None,
) -> BoxProfiles:
    """Prepare cuts using rotated center membership and projected output bins."""
    selection = box_profile_selection(view, extents, angle, force_rotated=True)
    return _selected_profiles(view, values, errors, selection, coverage_threshold, channel, reference_values)


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
