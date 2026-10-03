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


def _profile(view, coordinates, values, errors, selected, edges, threshold,
             *, centers=None, coverage_weights=None, channel="signal", reference_values=None):
    indices = np.clip(np.searchsorted(edges, coordinates, side="right") - 1, 0, len(edges) - 2)
    return prepare_measurement_profile(
        view, values, errors, selected=selected, indices=indices, edges=edges,
        centers=centers, coverage_weights=coverage_weights,
        coverage_threshold=threshold, channel=channel, reference_values=reference_values,
    )


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
    if not np.isclose(float(angle) % 360, 0.0, atol=1e-10):
        return rotated_box_profiles(view, values, errors, extents, angle,
            coverage_threshold=coverage_threshold, channel=channel, reference_values=reference_values)
    x, y = np.asarray(view["x_centers"]), np.asarray(view["y_centers"])
    z, e = np.asarray(values), np.asarray(errors)
    if z.shape != (len(y), len(x)) or e.shape != z.shape:
        raise ValueError("box profiles require matching 2D values and errors")
    x0, x1, y0, y1 = extents
    xm, ym = (x >= x0) & (x <= x1), (y >= y0) & (y <= y1)
    selected = ym[:, None] & xm[None, :]
    if not np.any(xm) or not np.any(ym):
        empty = (np.array([]), np.array([]), np.array([]))
        return BoxProfiles(empty, empty, selected)
    xe, ye = np.asarray(view["x_edges"]), np.asarray(view["y_edges"])
    xi, yi = np.flatnonzero(xm), np.flatnonzero(ym)
    xp = _profile(view, np.broadcast_to(x[None, :], z.shape), z, e, selected,
        xe[xi[0]:xi[-1]+2], coverage_threshold, centers=x[xm],
        coverage_weights=np.diff(ye)[:, None], channel=channel, reference_values=reference_values)
    yp = _profile(view, np.broadcast_to(y[:, None], z.shape), z, e, selected,
        ye[yi[0]:yi[-1]+2], coverage_threshold, centers=y[ym],
        coverage_weights=np.diff(xe)[None, :], channel=channel, reference_values=reference_values)
    return BoxProfiles(xp.arrays, yp.arrays, selected, xp, yp)


def rotated_box_profiles(
    view: dict[str, np.ndarray],
    values: np.ndarray,
    errors: np.ndarray,
    extents: tuple[float, float, float, float],
    angle: float = 0.0,
    *,
    coverage_threshold: float = 0.0,
    channel: str = "signal",
    reference_values=None,
) -> BoxProfiles:
    """Measurement-aware cuts along a box's two principal axes.

    Bin centers determine membership. For a rotated box, each selected pixel
    contributes to one projected bin on each axis. The projected pitch follows
    the source grid resolution, keeping the result bounded by 4096 bins.
    """

    x = np.asarray(view["x_centers"], dtype=float)
    y = np.asarray(view["y_centers"], dtype=float)
    z = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
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
    xp = _profile(view, u, z, e, selected, x_edges, coverage_threshold,
        channel=channel, reference_values=reference_values)
    yp = _profile(view, v, z, e, selected, y_edges, coverage_threshold,
        channel=channel, reference_values=reference_values)
    return BoxProfiles(xp.arrays, yp.arrays, selected, xp, yp)


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
