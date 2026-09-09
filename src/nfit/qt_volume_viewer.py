from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .colormaps import VOLUME_COLORMAPS
from .dataset import PointListData
from .mdhisto import MDHistoAxis, MDHistoData, mdhisto_measured_bins
from .plotting_core import gaussian_smooth_nan
from .qt_slice_controls import _IntegratedAxisSlider
from .qt_volume_controls import build_volume_panel_ui

TRANSFER_SAMPLES = 256
COLORMAPS = VOLUME_COLORMAPS
RENDER_MODES = ("Volume", "Isosurface")


@dataclass
class VolumeArrays:
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    color: np.ndarray
    opacity: np.ndarray


def supports_volume_view(data: Any) -> bool:
    """Return whether a dataset can provide three grid axes to PyVista."""

    return isinstance(data, MDHistoData) and not isinstance(data, PointListData) and data.signal.ndim >= 3


def volume_channel_names(data: MDHistoData) -> tuple[str, ...]:
    names = ["signal", "errors", "num_events", "combined_mask", "file_mask", "nfit_mask"]
    for name in ("fit", "residual"):
        values = data.metadata.get(name)
        if isinstance(values, np.ndarray) and values.shape == data.shape:
            names.append(name)
    return tuple(names)


def default_volume_axes(data: MDHistoData) -> tuple[int, int, int]:
    non_singleton = [dim for dim, size in enumerate(data.shape) if size > 1]
    if len(non_singleton) >= 3:
        return tuple(non_singleton[-3:])
    singleton = [dim for dim in range(data.signal.ndim) if dim not in non_singleton]
    return tuple((non_singleton + singleton)[:3])


def default_hidden_axis_index(
    data: MDHistoData,
    dim: int,
    *,
    apply_masks: bool = True,
) -> int:
    """Choose the nearest central hidden-axis bin containing measured data."""

    dim = int(dim)
    if not 0 <= dim < data.signal.ndim:
        raise ValueError(f"hidden axis {dim} is outside the dataset dimensions")
    valid = np.isfinite(np.asarray(data.signal, dtype=float))
    if apply_masks:
        valid &= mdhisto_measured_bins(data)
    reduce_axes = tuple(axis for axis in range(valid.ndim) if axis != dim)
    counts = np.sum(valid, axis=reduce_axes) if reduce_axes else valid.astype(int)
    if not np.any(counts > 0):
        return int(data.shape[dim] // 2)
    candidates = np.flatnonzero(counts == np.max(counts))
    midpoint = 0.5 * (data.shape[dim] - 1)
    return int(candidates[np.argmin(np.abs(candidates - midpoint))])


def sample_transfer_curve(
    points: Sequence[tuple[float, float]], values: np.ndarray | int = TRANSFER_SAMPLES
) -> np.ndarray:
    """Linearly sample a normalized piecewise transfer curve."""

    ordered = sorted((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))) for x, y in points)
    if not ordered:
        ordered = [(0.0, 0.0), (1.0, 1.0)]
    x = np.asarray([point[0] for point in ordered], dtype=float)
    y = np.asarray([point[1] for point in ordered], dtype=float)
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    target = np.linspace(0.0, 1.0, int(values)) if np.isscalar(values) else np.asarray(values, dtype=float)
    return np.interp(np.clip(target, 0.0, 1.0), unique_x, unique_y)


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        padding = max(abs(float(low)) * 0.01, 0.5)
        return float(low - padding), float(high + padding)
    return float(low), float(high)


def map_volume_rgba(
    color_values: np.ndarray,
    opacity_values: np.ndarray,
    *,
    cmap: str,
    color_range: tuple[float, float],
    opacity_range: tuple[float, float],
    color_curve: Sequence[tuple[float, float]],
    opacity_curve: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Map independent color and opacity channels to uint8 RGBA values."""

    from matplotlib import colormaps

    color_normalized = _normalize(color_values, color_range)
    opacity_normalized = _normalize(opacity_values, opacity_range)
    color_position = sample_transfer_curve(color_curve, color_normalized)
    alpha = sample_transfer_curve(opacity_curve, opacity_normalized)
    rgba = colormaps.get_cmap("gray" if cmap == "grey" else cmap)(color_position)
    rgba[..., 3] = alpha
    invalid = ~np.isfinite(color_values) | ~np.isfinite(opacity_values)
    rgba[invalid, 3] = 0.0
    return np.asarray(np.round(np.clip(rgba, 0.0, 1.0) * 255.0), dtype=np.uint8)


def build_rectilinear_volume_grid(arrays: VolumeArrays, rgba: np.ndarray):
    """Build the PyVista cell grid shared by rendering and model export."""

    import pyvista as pv

    grid = pv.RectilinearGrid(arrays.x_edges, arrays.y_edges, arrays.z_edges)
    grid.cell_data["rgba"] = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4, order="F")
    grid.cell_data["color_channel"] = arrays.color.ravel(order="F")
    grid.cell_data["opacity_channel"] = arrays.opacity.ravel(order="F")
    return grid


def crop_volume_arrays(
    arrays: VolumeArrays,
    limits: Sequence[tuple[float, float]],
) -> VolumeArrays:
    """Crop X/Y/Z cells by center while retaining their bounding edges."""

    edges = [arrays.x_edges, arrays.y_edges, arrays.z_edges]
    cell_slices = []
    cropped_edges = []
    for axis_edges, axis_limits in zip(edges, limits, strict=True):
        centers = 0.5 * (np.asarray(axis_edges[:-1]) + np.asarray(axis_edges[1:]))
        low, high = sorted((float(axis_limits[0]), float(axis_limits[1])))
        selected = np.flatnonzero((centers >= low) & (centers <= high))
        if selected.size == 0:
            nearest = int(np.argmin(np.abs(centers - 0.5 * (low + high))))
            selected = np.asarray([nearest])
        start, stop = int(selected[0]), int(selected[-1]) + 1
        cell_slices.append(slice(start, stop))
        cropped_edges.append(np.asarray(axis_edges[start : stop + 1], dtype=float))
    index = tuple(cell_slices)
    return VolumeArrays(
        x_edges=cropped_edges[0],
        y_edges=cropped_edges[1],
        z_edges=cropped_edges[2],
        color=np.asarray(arrays.color[index], dtype=float),
        opacity=np.asarray(arrays.opacity[index], dtype=float),
    )


def smooth_volume_arrays(
    arrays: VolumeArrays,
    sigma: Sequence[float],
) -> VolumeArrays:
    """Return a render-only Gaussian-smoothed copy of volume channels."""

    sigma_xyz = tuple(max(float(value), 0.0) for value in sigma)
    return VolumeArrays(
        x_edges=arrays.x_edges.copy(),
        y_edges=arrays.y_edges.copy(),
        z_edges=arrays.z_edges.copy(),
        color=gaussian_smooth_nan(arrays.color, sigma_xyz),
        opacity=gaussian_smooth_nan(arrays.opacity, sigma_xyz),
    )


def scale_rectilinear_grid(grid, scale: Sequence[float]):
    """Return a visual-only coordinate-scaled copy of a rectilinear grid."""

    scaled = grid.copy(deep=True)
    for name, factor in zip(("x", "y", "z"), scale, strict=True):
        coordinates = np.asarray(getattr(scaled, name), dtype=float)
        center = 0.5 * (float(coordinates[0]) + float(coordinates[-1]))
        setattr(scaled, name, center + (coordinates - center) * float(factor))
    return scaled


def rotation_frame_angles(fps: int, duration: float) -> np.ndarray:
    """Return equal per-frame azimuth steps for one complete orbit."""

    frame_count = max(2, int(round(float(duration) * int(fps))))
    return np.full(frame_count, 360.0 / frame_count, dtype=float)


def rotate_camera(camera: Any, angle_degrees: float, axis_name: str) -> None:
    """Orbit a PyVista camera about a displayed axis or its current view-up."""

    normalized = str(axis_name).strip().lower()
    if normalized == "vertical axis":
        camera.Azimuth(float(angle_degrees))
        return
    axes = {
        "x axis": np.asarray([1.0, 0.0, 0.0]),
        "y axis": np.asarray([0.0, 1.0, 0.0]),
        "z axis": np.asarray([0.0, 0.0, 1.0]),
    }
    if normalized not in axes:
        raise ValueError(f"unknown camera rotation axis {axis_name!r}")
    axis = axes[normalized]
    angle = np.deg2rad(float(angle_degrees))
    cross_matrix = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross_matrix
    )
    focal_point = np.asarray(camera.focal_point, dtype=float)
    position = np.asarray(camera.position, dtype=float)
    view_up = np.asarray(camera.up, dtype=float)
    camera.position = tuple(focal_point + rotation @ (position - focal_point))
    camera.up = tuple(rotation @ view_up)
    orthogonalize = getattr(camera, "OrthogonalizeViewUp", None)
    if callable(orthogonalize):
        orthogonalize()


def extract_volume_arrays(
    data: MDHistoData,
    *,
    axes: tuple[int, int, int],
    color_channel: str,
    opacity_channel: str,
    hidden_selections: dict[int, tuple[int, int] | int] | None = None,
    apply_masks: bool = True,
) -> VolumeArrays:
    """Slice/integrate an N-D histogram into X/Y/Z cell arrays."""

    if not supports_volume_view(data):
        raise ValueError("3D viewing requires a gridded dataset with at least three dimensions")
    if len(set(axes)) != 3 or any(dim < 0 or dim >= data.signal.ndim for dim in axes):
        raise ValueError("x, y, and z must be three distinct dataset axes")
    hidden_selections = dict(hidden_selections or {})
    color = _reduce_channel(data, color_channel, axes, hidden_selections, apply_masks)
    opacity = _reduce_channel(data, opacity_channel, axes, hidden_selections, apply_masks)
    return VolumeArrays(
        x_edges=_axis_edges(data.axes[axes[0]]),
        y_edges=_axis_edges(data.axes[axes[1]]),
        z_edges=_axis_edges(data.axes[axes[2]]),
        color=color,
        opacity=opacity,
    )


def _channel_array(data: MDHistoData, channel: str) -> np.ndarray:
    if channel == "signal":
        return np.asarray(data.signal, dtype=float)
    if channel == "errors":
        return np.asarray(data.errors, dtype=float)
    if channel == "num_events":
        return np.asarray(data.num_events, dtype=float)
    if channel == "combined_mask":
        return np.asarray(data.mask, dtype=float)
    if channel in {"file_mask", "nfit_mask"}:
        return np.asarray(data.metadata.get(channel, np.zeros(data.shape, dtype=bool)), dtype=float)
    values = data.metadata.get(channel)
    if isinstance(values, np.ndarray) and values.shape == data.shape:
        return np.asarray(values, dtype=float)
    raise ValueError(f"unknown 3D channel {channel!r}")


def _reduce_channel(
    data: MDHistoData,
    channel: str,
    axes: tuple[int, int, int],
    selections: dict[int, tuple[int, int] | int],
    apply_masks: bool,
) -> np.ndarray:
    values = _channel_array(data, channel).copy()
    is_mask = channel in {"combined_mask", "file_mask", "nfit_mask"}
    variance_channel = channel == "errors"
    valid = None
    if apply_masks and not is_mask:
        invalid = ~mdhisto_measured_bins(data)
        values[invalid] = np.nan
        valid = ~invalid
    if variance_channel:
        values = values**2
    index: list[Any] = []
    integrated_original_dims: list[int] = []
    for dim, size in enumerate(data.shape):
        if dim in axes:
            index.append(slice(None))
            continue
        selection = selections.get(dim, int(size // 2))
        if isinstance(selection, tuple):
            low, high = sorted((int(selection[0]), int(selection[1])))
            index.append(slice(max(0, low), min(size - 1, high) + 1))
            integrated_original_dims.append(dim)
        else:
            index.append(int(np.clip(selection, 0, size - 1)))
    reduced = np.asarray(values[tuple(index)], dtype=float)
    reduced_valid = None if valid is None else np.asarray(valid[tuple(index)], dtype=bool)
    remaining_dims = [dim for dim in range(data.signal.ndim) if dim in axes or dim in integrated_original_dims]
    for original_dim in sorted(integrated_original_dims, reverse=True):
        position = remaining_dims.index(original_dim)
        reduced = np.nanmax(reduced, axis=position) if is_mask else np.nansum(reduced, axis=position)
        if reduced_valid is not None:
            reduced_valid = np.any(reduced_valid, axis=position)
        remaining_dims.pop(position)
    if variance_channel:
        reduced = np.sqrt(reduced)
    if reduced_valid is not None:
        reduced = np.where(reduced_valid, reduced, np.nan)
    transpose = tuple(remaining_dims.index(dim) for dim in axes)
    return np.transpose(reduced, transpose)


def _axis_edges(axis: MDHistoAxis) -> np.ndarray:
    values = np.asarray(axis.values, dtype=float)
    centers = np.asarray(axis.centers, dtype=float)
    if values.size == centers.size + 1:
        return values
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - (midpoints[0] - centers[0])], midpoints, [centers[-1] + (centers[-1] - midpoints[-1])]))


def _normalize(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = (float(limits[0]), float(limits[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=float)
    return np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)


class TransferCurveEditor:
    """Small interactive piecewise-linear transfer-function editor."""

    def __new__(cls, *args, **kwargs):
        return _make_transfer_curve_editor(*args, **kwargs)


def _make_transfer_curve_editor(*, points=None, tooltip: str = ""):
    from PySide6 import QtCore, QtGui, QtWidgets

    class Editor(QtWidgets.QWidget):
        curveChanged = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.points = list(points or [(0.0, 0.0), (1.0, 1.0)])
            self._drag_index = None
            self.setMinimumWidth(0)
            self.setMinimumHeight(120)
            self.setMaximumHeight(180)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            self.setToolTip(tooltip)

        def sampled(self, count: int = TRANSFER_SAMPLES) -> np.ndarray:
            return sample_transfer_curve(self.points, count)

        def set_points(self, new_points: Sequence[tuple[float, float]]) -> None:
            self.points = sorted((float(x), float(y)) for x, y in new_points)
            self.update()
            self.curveChanged.emit()

        def _pixel(self, point):
            margin = 10.0
            return QtCore.QPointF(
                margin + point[0] * max(self.width() - 2.0 * margin, 1.0),
                self.height() - margin - point[1] * max(self.height() - 2.0 * margin, 1.0),
            )

        def _normalized(self, position):
            margin = 10.0
            return (
                float(np.clip((position.x() - margin) / max(self.width() - 2.0 * margin, 1.0), 0.0, 1.0)),
                float(np.clip((self.height() - margin - position.y()) / max(self.height() - 2.0 * margin, 1.0), 0.0, 1.0)),
            )

        def paintEvent(self, _event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), self.palette().base())
            painter.setPen(QtGui.QPen(self.palette().mid().color(), 1.0))
            painter.drawRect(self.rect().adjusted(9, 9, -9, -9))
            path = QtGui.QPainterPath()
            ordered = sorted(self.points)
            if ordered:
                path.moveTo(self._pixel(ordered[0]))
                for point in ordered[1:]:
                    path.lineTo(self._pixel(point))
            painter.setPen(QtGui.QPen(self.palette().highlight().color(), 2.0))
            painter.drawPath(path)
            painter.setBrush(self.palette().highlight())
            for point in ordered:
                painter.drawEllipse(self._pixel(point), 4.5, 4.5)

        def mousePressEvent(self, event):
            position = event.position()
            distances = [QtCore.QLineF(position, self._pixel(point)).length() for point in self.points]
            nearest = int(np.argmin(distances)) if distances else -1
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if nearest not in (0, len(self.points) - 1) and distances[nearest] <= 10.0:
                    self.points.pop(nearest)
                    self.update()
                    self.curveChanged.emit()
                return
            if nearest >= 0 and distances[nearest] <= 10.0:
                self._drag_index = nearest
            else:
                added = self._normalized(position)
                self.points.append(added)
                self.points.sort()
                self._drag_index = self.points.index(added)
                self.update()
                self.curveChanged.emit()

        def mouseMoveEvent(self, event):
            if self._drag_index is None:
                return
            x, y = self._normalized(event.position())
            if self._drag_index == 0:
                x = 0.0
            elif self._drag_index == len(self.points) - 1:
                x = 1.0
            else:
                x = np.clip(x, self.points[self._drag_index - 1][0] + 1e-4, self.points[self._drag_index + 1][0] - 1e-4)
            self.points[self._drag_index] = (float(x), y)
            self.update()
            self.curveChanged.emit()

        def mouseReleaseEvent(self, _event):
            self._drag_index = None

    return Editor()


class QtVolumeViewerPanel:
    """Embedded PyVista controls and render surface for N-D histogram data."""

    def __new__(cls, *args, **kwargs):
        return _make_volume_panel(*args, **kwargs)


def _make_volume_panel(
    datasets: Sequence[MDHistoData],
    *,
    dataset_names: Sequence[str],
    selected_index: int = 0,
):
    from PySide6 import QtCore, QtWidgets

    class Panel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            from .qt_controls import configure_numeric_spin_boxes

            self.datasets = list(datasets)
            self.dataset_names = list(dataset_names)
            self.dataset_index = int(selected_index)
            self.data = self.datasets[self.dataset_index]
            self.hidden_controls = {}
            self.current_grid = None
            self.current_surface = None
            self.current_render_grid = None
            self._shutdown_complete = False
            self.rotation_timer = QtCore.QTimer(self)
            self.rotation_timer.timeout.connect(self._rotate_frame)
            self._build()
            configure_numeric_spin_boxes(QtWidgets.QApplication.instance())
            self._sync_dataset()

        def _build(self):
            build_volume_panel_ui(
                self,
                render_modes=RENDER_MODES,
                transfer_curve_editor=TransferCurveEditor,
            )
        def _set_dataset(self, index):
            if not (0 <= int(index) < len(self.datasets)):
                return
            self.dataset_index = int(index)
            self.data = self.datasets[self.dataset_index]
            if not supports_volume_view(self.data):
                return
            self._sync_dataset()

        def _sync_dataset(self):
            if not supports_volume_view(self.data):
                return
            names = [axis.name for axis in self.data.axes]
            defaults = default_volume_axes(self.data)
            for combo, selected in zip(self.axis_combos, defaults, strict=True):
                blocked = combo.blockSignals(True)
                combo.clear()
                combo.addItems(names)
                combo.setCurrentIndex(selected)
                combo.blockSignals(blocked)
            self._sync_axis_limits()
            channels = volume_channel_names(self.data)
            for combo in (self.color_channel_combo, self.opacity_channel_combo):
                blocked = combo.blockSignals(True)
                combo.clear()
                combo.addItems(channels)
                combo.setCurrentText("signal")
                combo.blockSignals(blocked)
            self._rebuild_hidden_controls()
            self._sync_scale_controls()
            self._link_changed(True)
            self._autoscale_ranges()
            self.plotter.reset_camera()

        def _axes(self):
            return tuple(combo.currentIndex() for combo in self.axis_combos)

        def _axes_changed(self):
            axes = self._axes()
            sender = self.sender()
            if len(set(axes)) != 3:
                changed = self.axis_combos.index(sender) if sender in self.axis_combos else 0
                used = {axes[changed]}
                for index, combo in enumerate(self.axis_combos):
                    if index == changed:
                        continue
                    if combo.currentIndex() in used:
                        replacement = next(dim for dim in range(self.data.signal.ndim) if dim not in used)
                        blocked = combo.blockSignals(True)
                        combo.setCurrentIndex(replacement)
                        combo.blockSignals(blocked)
                    used.add(combo.currentIndex())
            self._sync_axis_limits()
            self._rebuild_hidden_controls()
            self._channel_changed()

        def _sync_axis_limits(self):
            for dim, (low_spin, high_spin) in zip(self._axes(), self.axis_limit_spins, strict=True):
                centers = np.asarray(self.data.axes[dim].centers, dtype=float)
                low, high = float(np.nanmin(centers)), float(np.nanmax(centers))
                for spin, value in ((low_spin, low), (high_spin, high)):
                    blocked = spin.blockSignals(True)
                    spin.setRange(low, high)
                    spin.setValue(value)
                    spin.blockSignals(blocked)

        def _axis_limits(self):
            return tuple((low.value(), high.value()) for low, high in self.axis_limit_spins)

        def _axis_limits_changed(self):
            self._channel_changed()

        def _reset_axis_limits(self):
            self._sync_axis_limits()
            self._channel_changed()

        def _sync_scale_controls(self):
            custom = self.scale_mode_combo.currentText() == "Custom"
            for spin in self.axis_scale_spins:
                spin.setEnabled(custom)

        def _axis_scale(self):
            if self.scale_mode_combo.currentText() != "Custom":
                return 1.0, 1.0, 1.0
            return tuple(float(spin.value()) for spin in self.axis_scale_spins)

        def _scale_changed(self, *_args):
            self._sync_scale_controls()
            if self.current_grid is not None:
                self._render()

        def _smoothing_sigma(self):
            return tuple(float(spin.value()) for spin in self.smoothing_spins)

        def _smoothing_changed(self, *_args):
            if self.current_grid is not None:
                self._render()

        def _rebuild_hidden_controls(self):
            # Reuse the slice viewer's range widget so hidden-axis selection has
            # identical handles and drag behavior in both visualization modes.
            while self.hidden_layout.count():
                item = self.hidden_layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self.hidden_controls = {}
            for dim in range(self.data.signal.ndim):
                if dim in self._axes():
                    continue
                axis = self.data.axes[dim]
                group = QtWidgets.QGroupBox(axis.name)
                row = QtWidgets.QGridLayout(group)
                row.setContentsMargins(8, 4, 8, 4)
                row.setHorizontalSpacing(6)
                row.setVerticalSpacing(3)
                centers = np.asarray(axis.centers, dtype=float)
                low_bound, high_bound = float(np.nanmin(centers)), float(np.nanmax(centers))
                default_index = default_hidden_axis_index(
                    self.data,
                    dim,
                    apply_masks=self.apply_masks_check.isChecked(),
                )
                value_spin = _float_spin(low_bound, high_bound)
                width_spin = _float_spin(0.0, max(high_bound - low_bound, 0.0))
                low_spin = _float_spin(low_bound, high_bound)
                high_spin = _float_spin(low_bound, high_bound)
                range_slider = _IntegratedAxisSlider(centers.size)
                value_spin.setObjectName(f"volume_hidden_{dim}_value_spin")
                width_spin.setObjectName(f"volume_hidden_{dim}_width_spin")
                low_spin.setObjectName(f"volume_hidden_{dim}_low_spin")
                high_spin.setObjectName(f"volume_hidden_{dim}_high_spin")
                range_slider.setObjectName(f"volume_hidden_{dim}_range_slider")
                value_spin.setToolTip(f"Center value selected along hidden axis {axis.name}.")
                width_spin.setToolTip(f"Width integrated along hidden axis {axis.name}.")
                low_spin.setToolTip(f"Lower bound of the integrated range along hidden axis {axis.name}.")
                high_spin.setToolTip(f"Upper bound of the integrated range along hidden axis {axis.name}.")
                range_slider.setToolTip(
                    f"Drag to choose the selected or integrated range along hidden axis {axis.name}."
                )
                integrate = QtWidgets.QCheckBox("Integrate range")
                integrate.setObjectName(f"volume_hidden_{dim}_integrate_check")
                integrate.setToolTip(
                    "Integrate over the selected range on this hidden axis instead of taking "
                    "one slice position."
                )
                low_index = high_index = default_index
                if centers.size > 1:
                    high_index = min(default_index + 1, centers.size - 1)
                    if high_index == default_index:
                        low_index = max(default_index - 1, 0)
                controls = {
                    "centers": centers,
                    "value": value_spin,
                    "width": width_spin,
                    "low": low_spin,
                    "high": high_spin,
                    "integrate": integrate,
                    "slider": range_slider,
                    "syncing": True,
                }
                self.hidden_controls[dim] = controls
                self._set_hidden_control_indices(dim, default_index, low_index, high_index)
                controls["syncing"] = False
                value_spin.valueChanged.connect(
                    lambda _value, d=dim: self._hidden_changed(d, "value_spin")
                )
                width_spin.valueChanged.connect(
                    lambda _value, d=dim: self._hidden_changed(d, "width_spin")
                )
                low_spin.valueChanged.connect(
                    lambda _value, d=dim: self._hidden_changed(d, "low_spin")
                )
                high_spin.valueChanged.connect(
                    lambda _value, d=dim: self._hidden_changed(d, "high_spin")
                )
                range_slider.changed.connect(lambda source, d=dim: self._hidden_changed(d, source))
                integrate.toggled.connect(lambda _value, d=dim: self._hidden_changed(d, "integrate"))
                row.addWidget(QtWidgets.QLabel("Value"), 0, 0)
                row.addWidget(value_spin, 0, 1)
                row.addWidget(QtWidgets.QLabel("Width"), 0, 2)
                row.addWidget(width_spin, 0, 3)
                row.addWidget(QtWidgets.QLabel("Range low"), 1, 0)
                row.addWidget(low_spin, 1, 1)
                row.addWidget(QtWidgets.QLabel("Range high"), 1, 2)
                row.addWidget(high_spin, 1, 3)
                row.addWidget(range_slider, 2, 0, 1, 4)
                row.addWidget(integrate, 3, 0, 1, 4)
                row.setColumnMinimumWidth(1, 82)
                row.setColumnMinimumWidth(3, 82)
                for column in range(4):
                    row.setColumnStretch(column, 0)
                self.hidden_layout.addWidget(group)
            if not self.hidden_controls:
                self.hidden_layout.addWidget(QtWidgets.QLabel("All dimensions are displayed."))

        def _hidden_changed(self, dim, source):
            controls = self.hidden_controls[dim]
            if controls["syncing"]:
                return
            centers = controls["centers"]
            slider = controls["slider"]
            value_index = slider.value_index
            low_index = slider.low_index
            high_index = slider.high_index
            if source == "value_spin":
                new_value = int(np.nanargmin(np.abs(centers - float(controls["value"].value()))))
                value_index, low_index, high_index = self._shift_hidden_range(
                    centers, new_value, low_index, high_index
                )
            elif source == "value":
                value_index, low_index, high_index = self._shift_hidden_range(
                    centers, value_index, low_index, high_index
                )
            elif source == "width_spin":
                self._set_hidden_integrate(controls, True)
                low_index, high_index = self._hidden_range_from_width(
                    centers, value_index, float(controls["width"].value())
                )
            elif source == "low_spin":
                self._set_hidden_integrate(controls, True)
                low_index = int(np.nanargmin(np.abs(centers - float(controls["low"].value()))))
                low_index, high_index = sorted((low_index, high_index))
                value_index = int(round(0.5 * (low_index + high_index)))
            elif source == "high_spin":
                self._set_hidden_integrate(controls, True)
                high_index = int(np.nanargmin(np.abs(centers - float(controls["high"].value()))))
                low_index, high_index = sorted((low_index, high_index))
                value_index = int(round(0.5 * (low_index + high_index)))
            elif source in {"low", "high"}:
                self._set_hidden_integrate(controls, True)
                low_index, high_index = sorted((low_index, high_index))
                value_index = int(round(0.5 * (low_index + high_index)))
            self._set_hidden_control_indices(dim, value_index, low_index, high_index)
            self._channel_changed()

        def _set_hidden_control_indices(self, dim, value_index, low_index, high_index):
            controls = self.hidden_controls[dim]
            centers = controls["centers"]
            value_index = int(np.clip(value_index, 0, centers.size - 1))
            low_index = int(np.clip(low_index, 0, centers.size - 1))
            high_index = int(np.clip(high_index, 0, centers.size - 1))
            low_index, high_index = sorted((low_index, high_index))
            controls["syncing"] = True
            try:
                controls["slider"].set_indices(value_index, low_index, high_index)
                controls["slider"].set_integrate_range(controls["integrate"].isChecked())
                controls["value"].setValue(float(centers[value_index]))
                controls["low"].setValue(float(centers[low_index]))
                controls["high"].setValue(float(centers[high_index]))
                controls["width"].setValue(float(centers[high_index] - centers[low_index]))
            finally:
                controls["syncing"] = False

        @staticmethod
        def _shift_hidden_range(centers, value_index, low_index, high_index):
            width = max(high_index - low_index, 0)
            half_low = width // 2
            half_high = width - half_low
            low_index = value_index - half_low
            high_index = value_index + half_high
            if low_index < 0:
                high_index -= low_index
                low_index = 0
            if high_index >= centers.size:
                low_index -= high_index - (centers.size - 1)
                high_index = centers.size - 1
            return value_index, max(low_index, 0), high_index

        @staticmethod
        def _hidden_range_from_width(centers, value_index, width):
            if centers.size <= 1:
                return 0, 0
            value = float(centers[value_index])
            low = int(np.nanargmin(np.abs(centers - (value - 0.5 * max(width, 0.0)))))
            high = int(np.nanargmin(np.abs(centers - (value + 0.5 * max(width, 0.0)))))
            if low == high and width > 0:
                if high < centers.size - 1:
                    high += 1
                elif low > 0:
                    low -= 1
            return tuple(sorted((low, high)))

        @staticmethod
        def _set_hidden_integrate(controls, checked):
            was_syncing = controls["syncing"]
            controls["syncing"] = True
            try:
                controls["integrate"].setChecked(bool(checked))
            finally:
                controls["syncing"] = was_syncing

        def _selections(self):
            selections = {}
            for dim, controls in self.hidden_controls.items():
                if controls["integrate"].isChecked():
                    selections[dim] = (
                        controls["slider"].low_index,
                        controls["slider"].high_index,
                    )
                else:
                    selections[dim] = controls["slider"].value_index
            return selections

        def _link_changed(self, checked):
            self.opacity_channel_combo.setEnabled(not bool(checked))
            if checked:
                blocked = self.opacity_channel_combo.blockSignals(True)
                self.opacity_channel_combo.setCurrentText(self.color_channel_combo.currentText())
                self.opacity_channel_combo.blockSignals(blocked)
            self._channel_changed()

        def _channel_changed(self, *_args):
            if self.link_channels_check.isChecked():
                blocked = self.opacity_channel_combo.blockSignals(True)
                self.opacity_channel_combo.setCurrentText(self.color_channel_combo.currentText())
                self.opacity_channel_combo.blockSignals(blocked)
            self._autoscale_ranges(render=False)
            self._render()

        def _arrays(self):
            arrays = extract_volume_arrays(
                self.data,
                axes=self._axes(),
                color_channel=self.color_channel_combo.currentText(),
                opacity_channel=self.opacity_channel_combo.currentText(),
                hidden_selections=self._selections(),
                apply_masks=self.apply_masks_check.isChecked(),
            )
            return crop_volume_arrays(arrays, self._axis_limits())

        def _autoscale_ranges(self, _checked=False, *, render=True):
            if not supports_volume_view(self.data):
                return
            arrays = self._arrays()
            for widgets, values in (((self.color_min, self.color_max), arrays.color), ((self.opacity_min, self.opacity_max), arrays.opacity)):
                low, high = finite_range(values)
                for widget, value in zip(widgets, (low, high), strict=True):
                    blocked = widget.blockSignals(True)
                    widget.setValue(value)
                    widget.blockSignals(blocked)
            if render:
                self._render()

        def _rgba(self, arrays):
            return map_volume_rgba(
                arrays.color,
                arrays.opacity,
                cmap=self.cmap_combo.currentText(),
                color_range=(self.color_min.value(), self.color_max.value()),
                opacity_range=(self.opacity_min.value(), self.opacity_max.value()),
                color_curve=self.color_curve.points,
                opacity_curve=self.opacity_curve.points,
            )

        def _grid(self, arrays):
            return build_rectilinear_volume_grid(arrays, self._rgba(arrays))

        def _surface_for_grid(self, grid, level):
            point_grid = grid.cell_data_to_point_data()
            surface = point_grid.contour([level], scalars="opacity_channel")
            if not surface.n_points:
                return surface
            color = np.asarray(surface.point_data["color_channel"], dtype=float)
            opacity = np.asarray(surface.point_data["opacity_channel"], dtype=float)
            surface.point_data["rgba"] = map_volume_rgba(
                color,
                opacity,
                cmap=self.cmap_combo.currentText(),
                color_range=(self.color_min.value(), self.color_max.value()),
                opacity_range=(self.opacity_min.value(), self.opacity_max.value()),
                color_curve=self.color_curve.points,
                opacity_curve=self.opacity_curve.points,
            )
            return surface

        def _render(self, *_args):
            if not supports_volume_view(self.data) or len(set(self._axes())) != 3:
                return
            camera = self.plotter.camera_position if self.current_grid is not None else None
            arrays = self._arrays()
            self.current_grid = self._grid(arrays)
            render_arrays = smooth_volume_arrays(arrays, self._smoothing_sigma())
            render_grid = self._grid(render_arrays)
            self.current_render_grid = scale_rectilinear_grid(render_grid, self._axis_scale())
            self.current_surface = None
            self.plotter.clear()
            has_finite_voxels = bool(
                np.any(
                    np.isfinite(render_arrays.color)
                    & np.isfinite(render_arrays.opacity)
                )
            )
            self.render_status.setVisible(not has_finite_voxels)
            if not has_finite_voxels:
                self.render_status.setText(
                    "No finite voxels are available in this selection. "
                    "Change the remaining-axis range or turn off Apply masks."
                )
            elif self.render_combo.currentText() == "Isosurface":
                low, high = self.opacity_min.value(), self.opacity_max.value()
                fraction = self.surface_slider.value() / 1000.0
                level = low + fraction * (high - low)
                self.surface_label.setText(f"Level {level:g}")
                self.current_surface = self._surface_for_grid(self.current_grid, level)
                render_surface = self._surface_for_grid(self.current_render_grid, level)
                if render_surface.n_points:
                    self.plotter.add_mesh(render_surface, scalars="rgba", rgba=True, smooth_shading=True)
            else:
                self.plotter.add_volume(
                    self.current_render_grid,
                    scalars="rgba",
                    preference="cell",
                    shade=False,
                )
            axes = self._axes()
            self.plotter.show_bounds(
                xtitle=self.data.axes[axes[0]].name,
                ytitle=self.data.axes[axes[1]].name,
                ztitle=self.data.axes[axes[2]].name,
                axes_ranges=tuple(float(value) for value in self.current_grid.bounds),
                color="black",
            )
            if camera is None:
                self.plotter.reset_camera()
            else:
                self.plotter.camera_position = camera
            self.plotter.render()
            self.surface_group.setVisible(self.render_combo.currentText() == "Isosurface")

        def export_model(self):
            if self.current_grid is None:
                return
            if self.render_combo.currentText() == "Isosurface":
                path, selected = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export 3D isosurface", "nfit_isosurface.vtp",
                    "VTK PolyData (*.vtp);;Polygon mesh (*.ply);;STL mesh (*.stl);;glTF scene (*.gltf)",
                )
                if not path:
                    return
                if str(path).lower().endswith(".gltf") or "glTF" in selected:
                    import pyvista as pv

                    export_plotter = pv.Plotter(off_screen=True)
                    try:
                        export_plotter.add_mesh(
                            self.current_surface,
                            scalars="rgba",
                            rgba=True,
                            smooth_shading=True,
                            render=False,
                        )
                        export_plotter.export_gltf(str(path))
                    finally:
                        export_plotter.close()
                elif self.current_surface is not None:
                    self.current_surface.save(str(path))
            else:
                path, _selected = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export 3D volume", "nfit_volume.vtr", "VTK rectilinear grid (*.vtr)"
                )
                if path:
                    self.current_grid.save(str(path))

        def _toggle_rotation(self, checked):
            if checked:
                fps = 30
                self.rotation_timer.start(round(1000 / fps))
            else:
                self.rotation_timer.stop()

        def _rotate_frame(self):
            degrees = float(self.rotation_speed_spin.value()) / 30.0
            rotate_camera(self.plotter.camera, degrees, self.rotation_axis_combo.currentText())
            self.plotter.render()

        def export_still(self):
            path, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export current 3D view", "nfit_3d_view.png", "PNG image (*.png)"
            )
            if path:
                self.plotter.screenshot(str(path))

        def export_rotation_movie(self):
            path, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export 3D rotation movie", "nfit_3d_rotation.mp4", "MP4 movie (*.mp4)"
            )
            if not path:
                return
            was_rotating = self.rotation_timer.isActive()
            self.rotation_timer.stop()
            camera = self.plotter.camera_position
            fps = int(self.movie_fps_spin.value())
            angles = rotation_frame_angles(fps, self.movie_duration_spin.value())
            try:
                self.plotter.open_movie(str(path), framerate=fps, quality=8)
                for angle in angles:
                    self.plotter.write_frame()
                    rotate_camera(
                        self.plotter.camera,
                        float(angle),
                        self.rotation_axis_combo.currentText(),
                    )
                    self.plotter.render()
                self.plotter.mwriter.close()
                self.plotter.mwriter = None
            finally:
                if self.plotter.mwriter is not None:
                    self.plotter.mwriter.close()
                    self.plotter.mwriter = None
                self.plotter.camera_position = camera
                self.plotter.render()
                if was_rotating:
                    self.rotation_timer.start(round(1000 / 30))

        def replace_datasets(self, new_datasets, new_names, selected_index=0):
            self.datasets = list(new_datasets)
            self.dataset_names = list(new_names)
            blocked = self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(self.dataset_names)
            self.dataset_combo.setCurrentIndex(int(selected_index))
            self.dataset_combo.blockSignals(blocked)
            self._set_dataset(selected_index)

        def shutdown(self):
            if self._shutdown_complete:
                return
            self._shutdown_complete = True
            self.rotation_timer.stop()
            interactor = self.plotter.interactor
            interactor.setParent(None)
            self.plotter.close()
            interactor.deleteLater()

        def closeEvent(self, event):
            self.shutdown()
            super().closeEvent(event)

    return Panel()


def _float_spin(minimum: float = -1.0e300, maximum: float = 1.0e300):
    from PySide6 import QtWidgets

    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(float(minimum), float(maximum))
    spin.setDecimals(8)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(0)
    spin.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Ignored,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    return spin
