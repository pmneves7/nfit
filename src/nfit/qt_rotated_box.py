"""Matplotlib interaction for a data-coordinate rotated histogram box."""

from __future__ import annotations

from copy import copy

import numpy as np
from matplotlib.patches import Polygon

from .box_cuts import box_corners


class RotatedBoxSelector:
    """Create, move, resize, and rotate a rectangle on an image axes."""

    def __init__(self, ax, onselect) -> None:
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.onselect = onselect
        self._extents = (0.0, 0.0, 0.0, 0.0)
        self.angle = 0.0
        self.active = False
        self.visible = False
        self._drag = None
        self._press_event = None
        self._press_geometry = None
        self.patch = Polygon(
            np.zeros((4, 2)), closed=True, fill=False, edgecolor="#8a8a8a", lw=1.5, zorder=5
        )
        ax.add_patch(self.patch)
        self.handles = {}
        for name in ("SW", "S", "SE", "W", "C", "E", "NW", "N", "NE", "R"):
            marker = "o" if name == "R" else "s"
            (line,) = ax.plot(
                [],
                [],
                marker=marker,
                markersize=6,
                ls="None",
                zorder=7,
                clip_on=False,
            )
            self.handles[name] = line
        (self.stem,) = ax.plot([], [], color="#8a8a8a", lw=1, zorder=6, clip_on=False)
        self._ids = [
            self.canvas.mpl_connect("button_press_event", self._press),
            self.canvas.mpl_connect("motion_notify_event", self._move),
            self.canvas.mpl_connect("button_release_event", self._release),
        ]
        self._redraw()

    @property
    def extents(self):
        return self._extents

    @property
    def artists(self):
        """Expose the selection patch first, as Matplotlib selectors do."""

        return (self.patch, *self.handles.values(), self.stem)

    @extents.setter
    def extents(self, value):
        x0, x1, y0, y1 = map(float, value)
        self._extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._redraw()

    def set_active(self, active: bool) -> None:
        self.active = bool(active)
        self._redraw()

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self._redraw()

    def set_props(self, **props) -> None:
        self.patch.set(**props)
        self._redraw()

    def set_handle_props(self, **props) -> None:
        for handle in self.handles.values():
            handle.set_markeredgecolor(props.get("markeredgecolor", "#8a8a8a"))
            handle.set_markerfacecolor(props.get("markerfacecolor", "#8a8a8a"))
            handle.set_alpha(props.get("alpha", 1.0))
        self._redraw()

    def disconnect_events(self) -> None:
        for connection in self._ids:
            self.canvas.mpl_disconnect(connection)
        if self.patch in self.ax.patches:
            self.patch.remove()
        if self.stem in self.ax.lines:
            self.stem.remove()
        for handle in self.handles.values():
            if handle in self.ax.lines:
                handle.remove()

    def _points(self):
        sw, se, ne, nw = box_corners(self._extents, self.angle)
        points = {
            "SW": sw,
            "S": (sw + se) / 2,
            "SE": se,
            "W": (sw + nw) / 2,
            "C": (sw + ne) / 2,
            "E": (se + ne) / 2,
            "NW": nw,
            "N": (nw + ne) / 2,
            "NE": ne,
        }
        center_px = self.ax.transData.transform(points["C"])
        north_px = self.ax.transData.transform(points["N"])
        outward = north_px - center_px
        length = np.hypot(*outward)
        outward = outward / length if length > 0 else np.array((0.0, 1.0))
        points["R"] = tuple(self.ax.transData.inverted().transform(north_px + 24 * outward))
        return points

    def _redraw(self) -> None:
        if self.visible and self.patch not in self.ax.patches:
            self.ax.add_patch(self.patch)
        elif not self.visible and self.patch in self.ax.patches:
            self.patch.remove()
        show_handles = self.visible and self.active
        for artist in (*self.handles.values(), self.stem):
            if show_handles and artist not in self.ax.lines:
                self.ax.add_line(artist)
            elif not show_handles and artist in self.ax.lines:
                artist.remove()
        points = self._points()
        self.patch.set_xy([points[name] for name in ("SW", "SE", "NE", "NW")])
        self.patch.set_visible(self.visible)
        for name, handle in self.handles.items():
            handle.set_data([points[name][0]], [points[name][1]])
            handle.set_visible(show_handles)
        self.stem.set_data([points["N"][0], points["R"][0]], [points["N"][1], points["R"][1]])
        self.stem.set_visible(show_handles)
        self.canvas.draw_idle()

    def _press(self, event) -> None:
        if not self.active or not self.visible or event.button != 1:
            return
        points = self._points()
        pixel_points = {name: self.ax.transData.transform(point) for name, point in points.items()}
        nearest = min(
            pixel_points, key=lambda name: np.hypot(*(pixel_points[name] - (event.x, event.y)))
        )
        distance = np.hypot(*(pixel_points[nearest] - (event.x, event.y)))
        if event.inaxes is not self.ax and (nearest != "R" or distance > 10):
            return
        if event.xdata is None or event.ydata is None or event.inaxes is not self.ax:
            event = copy(event)
            event.xdata, event.ydata = self.ax.transData.inverted().transform((event.x, event.y))
        if distance <= 10:
            drag = nearest
        else:
            from matplotlib.path import Path

            polygon = np.array([points[name] for name in ("SW", "SE", "NE", "NW")])
            drag = "C" if Path(polygon).contains_point((event.xdata, event.ydata)) else "CREATE"
        self._drag = drag
        self._press_event = event
        self._press_geometry = (self._extents, self.angle)

    def _move(self, event) -> None:
        if self._drag is None:
            return
        if event.xdata is None or event.ydata is None or event.inaxes is not self.ax:
            event = copy(event)
            event.xdata, event.ydata = self.ax.transData.inverted().transform((event.x, event.y))
        extents, angle = self._press_geometry
        x0, x1, y0, y1 = extents
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if self._drag == "R":
            start = np.arctan2(self._press_event.ydata - cy, self._press_event.xdata - cx)
            current = np.arctan2(event.ydata - cy, event.xdata - cx)
            delta = np.arctan2(np.sin(current - start), np.cos(current - start))
            candidate = (angle + np.rad2deg(delta) + 180) % 360 - 180
            self.angle = (
                round(candidate / 15) * 15 if event.key and "shift" in event.key else candidate
            )
        elif self._drag == "CREATE":
            self.angle = 0.0
            self._extents = (
                min(self._press_event.xdata, event.xdata),
                max(self._press_event.xdata, event.xdata),
                min(self._press_event.ydata, event.ydata),
                max(self._press_event.ydata, event.ydata),
            )
        elif self._drag == "C":
            dx, dy = event.xdata - self._press_event.xdata, event.ydata - self._press_event.ydata
            self._extents = (x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        else:
            theta = np.deg2rad(angle)
            c, s = np.cos(theta), np.sin(theta)
            dx, dy = event.xdata - cx, event.ydata - cy
            u, v = dx * c + dy * s, -dx * s + dy * c
            left, right, bottom, top = -(x1 - x0) / 2, (x1 - x0) / 2, -(y1 - y0) / 2, (y1 - y0) / 2
            if "W" in self._drag:
                left = min(u, right - 1e-12)
            if "E" in self._drag:
                right = max(u, left + 1e-12)
            if "S" in self._drag:
                bottom = min(v, top - 1e-12)
            if "N" in self._drag:
                top = max(v, bottom + 1e-12)
            du, dv = (left + right) / 2, (bottom + top) / 2
            new_cx, new_cy = cx + du * c - dv * s, cy + du * s + dv * c
            self._extents = (
                new_cx - (right - left) / 2,
                new_cx + (right - left) / 2,
                new_cy - (top - bottom) / 2,
                new_cy + (top - bottom) / 2,
            )
        self._redraw()

    def _release(self, event) -> None:
        if self._drag is None:
            return
        self._move(event)
        if self._extents[1] > self._extents[0] and self._extents[3] > self._extents[2]:
            self.onselect(self._press_event, event)
        self._drag = None
        self._press_event = None
        self._press_geometry = None
