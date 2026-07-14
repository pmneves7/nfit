from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .dataset import PointListData
from .mdhisto import MDHistoAxis, MDHistoData, mdhisto_measured_bins
from .plotting import gaussian_smooth_nan


TRANSFER_SAMPLES = 256
COLORMAPS = ("viridis", "magma", "plasma", "cividis", "turbo", "coolwarm", "grey")
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
    from matplotlib import colormaps
    from PySide6 import QtCore, QtWidgets
    from pyvistaqt import QtInteractor

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
            root = QtWidgets.QHBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            splitter = QtWidgets.QSplitter()
            render_frame = QtWidgets.QFrame()
            render_layout = QtWidgets.QVBoxLayout(render_frame)
            render_layout.setContentsMargins(8, 8, 8, 8)
            self.plotter = QtInteractor(render_frame)
            render_layout.addWidget(self.plotter.interactor)
            self.plotter.set_background("white")
            self.plotter.add_axes(color="black")
            controls = QtWidgets.QScrollArea()
            controls.setObjectName("volume_controls_scroll")
            controls.setWidgetResizable(True)
            controls.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            controls.setMinimumWidth(390)
            controls.setMaximumWidth(480)
            body = QtWidgets.QWidget()
            body.setObjectName("volume_controls_body")
            body.setMinimumWidth(0)
            body.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            controls.setWidget(body)
            layout = QtWidgets.QVBoxLayout(body)

            dataset_group = QtWidgets.QGroupBox("3D dataset")
            dataset_layout = QtWidgets.QGridLayout(dataset_group)
            self.dataset_combo = QtWidgets.QComboBox()
            self.dataset_combo.setObjectName("volume_dataset_combo")
            self.dataset_combo.addItems(self.dataset_names)
            self.dataset_combo.setCurrentIndex(self.dataset_index)
            self.dataset_combo.setToolTip("Choose a gridded dataset with at least three dimensions for 3D rendering.")
            self.dataset_combo.currentIndexChanged.connect(self._set_dataset)
            self.render_combo = QtWidgets.QComboBox()
            self.render_combo.setObjectName("volume_render_mode_combo")
            self.render_combo.addItems(RENDER_MODES)
            self.render_combo.setToolTip("Choose direct volumetric rendering or an extracted constant-value isosurface.")
            self.render_combo.currentTextChanged.connect(self._render)
            dataset_layout.addWidget(QtWidgets.QLabel("Dataset"), 0, 0)
            dataset_layout.addWidget(self.dataset_combo, 0, 1)
            dataset_layout.addWidget(QtWidgets.QLabel("Mode"), 1, 0)
            dataset_layout.addWidget(self.render_combo, 1, 1)
            layout.addWidget(dataset_group)

            axes_group = QtWidgets.QGroupBox("3D axes")
            axes_layout = QtWidgets.QGridLayout(axes_group)
            self.axis_combos = []
            self.axis_limit_spins = []
            for row, label in enumerate(("X", "Y", "Z")):
                combo = QtWidgets.QComboBox()
                combo.setObjectName(f"volume_{label.lower()}_axis_combo")
                combo.setToolTip(f"Choose the dataset axis mapped to the 3D {label.lower()} coordinate.")
                combo.currentIndexChanged.connect(self._axes_changed)
                self.axis_combos.append(combo)
                axes_layout.addWidget(QtWidgets.QLabel(label), row, 0)
                axes_layout.addWidget(combo, row, 1)
                low_spin = _float_spin()
                high_spin = _float_spin()
                low_spin.setObjectName(f"volume_{label.lower()}_min_spin")
                high_spin.setObjectName(f"volume_{label.lower()}_max_spin")
                low_spin.setToolTip(f"Minimum displayed {label} coordinate; complete bins are selected by center.")
                high_spin.setToolTip(f"Maximum displayed {label} coordinate; complete bins are selected by center.")
                low_spin.editingFinished.connect(self._axis_limits_changed)
                high_spin.editingFinished.connect(self._axis_limits_changed)
                self.axis_limit_spins.append((low_spin, high_spin))
                axes_layout.addWidget(QtWidgets.QLabel(f"{label} min"), row + 3, 0)
                axes_layout.addWidget(low_spin, row + 3, 1)
                axes_layout.addWidget(QtWidgets.QLabel("max"), row + 3, 2)
                axes_layout.addWidget(high_spin, row + 3, 3)
            reset_axis_limits = QtWidgets.QPushButton("Reset axis limits")
            reset_axis_limits.setObjectName("volume_reset_axis_limits_button")
            reset_axis_limits.setToolTip("Restore X, Y, and Z to the complete range of their selected dataset axes.")
            reset_axis_limits.clicked.connect(self._reset_axis_limits)
            axes_layout.addWidget(reset_axis_limits, 6, 0, 1, 4)
            layout.addWidget(axes_group)

            scale_group = QtWidgets.QGroupBox("Axis scale")
            scale_layout = QtWidgets.QGridLayout(scale_group)
            self.scale_mode_combo = QtWidgets.QComboBox()
            self.scale_mode_combo.setObjectName("volume_axis_scale_mode_combo")
            self.scale_mode_combo.addItems(["Equal data units", "Custom"])
            self.scale_mode_combo.setToolTip(
                "Use equal data-unit lengths on every axis, or apply independent visual scale factors without changing data values."
            )
            self.scale_mode_combo.currentTextChanged.connect(self._scale_changed)
            self.axis_scale_spins = []
            scale_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
            scale_layout.addWidget(self.scale_mode_combo, 0, 1, 1, 2)
            for column, label in enumerate(("X", "Y", "Z")):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setObjectName(f"volume_{label.lower()}_scale_spin")
                spin.setRange(0.01, 100.0)
                spin.setDecimals(3)
                spin.setValue(1.0)
                spin.setSingleStep(0.1)
                spin.setMinimumWidth(0)
                spin.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                spin.setToolTip(f"Visual length multiplier for {label}; values below one squish and values above one stretch the axis.")
                spin.valueChanged.connect(self._scale_changed)
                self.axis_scale_spins.append(spin)
                scale_layout.addWidget(QtWidgets.QLabel(label), 1, column)
                scale_layout.addWidget(spin, 2, column)
            layout.addWidget(scale_group)

            smoothing_group = QtWidgets.QGroupBox("Plot smoothing")
            smoothing_layout = QtWidgets.QGridLayout(smoothing_group)
            self.smoothing_spins = []
            for column, label in enumerate(("X", "Y", "Z")):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setObjectName(f"volume_{label.lower()}_smoothing_spin")
                spin.setRange(0.0, 100.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.25)
                spin.setSuffix(" bins")
                spin.setMinimumWidth(0)
                spin.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                spin.setToolTip(
                    f"Gaussian blur sigma along displayed {label}, in bin widths. "
                    "This affects rendering, still images, and movies only; fitting, data values, and model/data exports remain unchanged."
                )
                spin.valueChanged.connect(self._smoothing_changed)
                self.smoothing_spins.append(spin)
                smoothing_layout.addWidget(QtWidgets.QLabel(label), 0, column)
                smoothing_layout.addWidget(spin, 1, column)
            layout.addWidget(smoothing_group)
            self.hidden_group = QtWidgets.QGroupBox("Remaining axes")
            self.hidden_layout = QtWidgets.QVBoxLayout(self.hidden_group)
            self.hidden_group.setToolTip("Choose one position or integrate a range for dimensions not mapped to X, Y, or Z.")
            layout.addWidget(self.hidden_group)

            channels_group = QtWidgets.QGroupBox("Color and opacity")
            channels_layout = QtWidgets.QGridLayout(channels_group)
            self.color_channel_combo = QtWidgets.QComboBox()
            self.color_channel_combo.setObjectName("volume_color_channel_combo")
            self.color_channel_combo.setToolTip("Dataset channel mapped through the colormap.")
            self.opacity_channel_combo = QtWidgets.QComboBox()
            self.opacity_channel_combo.setObjectName("volume_opacity_channel_combo")
            self.opacity_channel_combo.setToolTip("Dataset channel independently mapped to opacity when channels are unlinked.")
            self.link_channels_check = QtWidgets.QCheckBox("Link opacity to color")
            self.link_channels_check.setObjectName("volume_link_channels_check")
            self.link_channels_check.setChecked(True)
            self.link_channels_check.setToolTip("Use the color channel for opacity. Uncheck to select a different opacity channel and range.")
            self.cmap_combo = QtWidgets.QComboBox()
            self.cmap_combo.setObjectName("volume_colormap_combo")
            self.cmap_combo.addItems([name for name in COLORMAPS if name in colormaps])
            self.cmap_combo.setCurrentText("viridis")
            self.cmap_combo.setToolTip("Colormap used after applying the editable color mapping curve.")
            for widget in (self.color_channel_combo, self.opacity_channel_combo):
                widget.currentTextChanged.connect(self._channel_changed)
            self.cmap_combo.currentTextChanged.connect(self._render)
            self.link_channels_check.toggled.connect(self._link_changed)
            channels_layout.addWidget(QtWidgets.QLabel("Color channel"), 0, 0)
            channels_layout.addWidget(self.color_channel_combo, 0, 1)
            channels_layout.addWidget(QtWidgets.QLabel("Opacity channel"), 1, 0)
            channels_layout.addWidget(self.opacity_channel_combo, 1, 1)
            channels_layout.addWidget(self.link_channels_check, 2, 0, 1, 2)
            channels_layout.addWidget(QtWidgets.QLabel("Colormap"), 3, 0)
            channels_layout.addWidget(self.cmap_combo, 3, 1)
            layout.addWidget(channels_group)

            range_group = QtWidgets.QGroupBox("Channel ranges")
            range_layout = QtWidgets.QGridLayout(range_group)
            self.color_min = _float_spin()
            self.color_max = _float_spin()
            self.opacity_min = _float_spin()
            self.opacity_max = _float_spin()
            for widget in (self.color_min, self.color_max, self.opacity_min, self.opacity_max):
                widget.setToolTip("Values outside this range are clipped before the transfer curve is applied.")
                widget.editingFinished.connect(self._render)
            range_layout.addWidget(QtWidgets.QLabel("Color min"), 0, 0)
            range_layout.addWidget(self.color_min, 0, 1)
            range_layout.addWidget(QtWidgets.QLabel("Color max"), 1, 0)
            range_layout.addWidget(self.color_max, 1, 1)
            range_layout.addWidget(QtWidgets.QLabel("Opacity min"), 2, 0)
            range_layout.addWidget(self.opacity_min, 2, 1)
            range_layout.addWidget(QtWidgets.QLabel("Opacity max"), 3, 0)
            range_layout.addWidget(self.opacity_max, 3, 1)
            range_layout.setColumnStretch(1, 1)
            reset_ranges = QtWidgets.QPushButton("Autoscale ranges")
            reset_ranges.setToolTip("Set both channel ranges from the finite 1st and 99th percentiles of the current 3D data.")
            reset_ranges.clicked.connect(self._autoscale_ranges)
            range_layout.addWidget(reset_ranges, 4, 0, 1, 2)
            layout.addWidget(range_group)

            color_curve_group = QtWidgets.QGroupBox("Color mapping curve")
            color_curve_layout = QtWidgets.QVBoxLayout(color_curve_group)
            self.color_curve = TransferCurveEditor(
                tooltip="Map normalized color-channel values (horizontal) to positions in the selected colormap (vertical). Drag points; click to add; right-click an interior point to remove it."
            )
            self.color_curve.setObjectName("volume_color_curve")
            self.color_curve.curveChanged.connect(self._render)
            color_curve_layout.addWidget(self.color_curve)
            layout.addWidget(color_curve_group)
            opacity_curve_group = QtWidgets.QGroupBox("Opacity mapping curve")
            opacity_curve_layout = QtWidgets.QVBoxLayout(opacity_curve_group)
            self.opacity_curve = TransferCurveEditor(
                points=[(0.0, 0.0), (0.35, 0.0), (1.0, 1.0)],
                tooltip="Map normalized opacity-channel values (horizontal) to transparency/opacity (vertical). Drag points; click to add; right-click an interior point to remove it."
            )
            self.opacity_curve.setObjectName("volume_opacity_curve")
            self.opacity_curve.curveChanged.connect(self._render)
            opacity_curve_layout.addWidget(self.opacity_curve)
            layout.addWidget(opacity_curve_group)

            surface_group = QtWidgets.QGroupBox("Isosurface")
            surface_layout = QtWidgets.QGridLayout(surface_group)
            self.surface_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.surface_slider.setObjectName("volume_surface_level_slider")
            self.surface_slider.setRange(0, 1000)
            self.surface_slider.setValue(700)
            self.surface_slider.setToolTip("Select the constant opacity-channel value used to extract the isosurface.")
            self.surface_slider.valueChanged.connect(self._render)
            self.surface_label = QtWidgets.QLabel()
            surface_layout.addWidget(self.surface_label, 0, 0)
            surface_layout.addWidget(self.surface_slider, 0, 1)
            self.surface_group = surface_group
            layout.addWidget(surface_group)

            self.apply_masks_check = QtWidgets.QCheckBox("Apply masks")
            self.apply_masks_check.setChecked(True)
            self.apply_masks_check.setToolTip("Make masked and empty bins transparent before 3D rendering.")
            self.apply_masks_check.toggled.connect(self._channel_changed)
            layout.addWidget(self.apply_masks_check)
            self.export_button = QtWidgets.QPushButton("Export 3D model")
            self.export_button.setObjectName("volume_export_button")
            self.export_button.setToolTip("Export the isosurface as a mesh/scene, or export volumetric data as a VTK rectilinear grid.")
            self.export_button.clicked.connect(self.export_model)
            layout.addWidget(self.export_button)

            animation_group = QtWidgets.QGroupBox("Camera and export")
            animation_layout = QtWidgets.QGridLayout(animation_group)
            self.rotate_check = QtWidgets.QCheckBox("Rotate around vertical axis")
            self.rotate_check.setObjectName("volume_rotate_check")
            self.rotate_check.setText("Animate rotation")
            self.rotate_check.setToolTip("Continuously orbit the camera around the selected rotation axis.")
            self.rotate_check.toggled.connect(self._toggle_rotation)
            self.rotation_axis_combo = QtWidgets.QComboBox()
            self.rotation_axis_combo.setObjectName("volume_rotation_axis_combo")
            self.rotation_axis_combo.addItems(["X axis", "Y axis", "Z axis", "Vertical axis"])
            self.rotation_axis_combo.setCurrentText("Z axis")
            self.rotation_axis_combo.setToolTip(
                "Rotate around displayed X, Y, or Z, or choose Vertical axis to rotate around the direction currently vertical on screen."
            )
            self.rotation_speed_spin = QtWidgets.QDoubleSpinBox()
            self.rotation_speed_spin.setRange(1.0, 360.0)
            self.rotation_speed_spin.setValue(30.0)
            self.rotation_speed_spin.setSuffix(" deg/s")
            self.rotation_speed_spin.setToolTip("Angular speed used by the live rotation preview.")
            self.movie_fps_spin = QtWidgets.QSpinBox()
            self.movie_fps_spin.setRange(1, 120)
            self.movie_fps_spin.setValue(30)
            self.movie_fps_spin.setToolTip("Frames per second written to the MP4 rotation movie.")
            self.movie_duration_spin = QtWidgets.QDoubleSpinBox()
            self.movie_duration_spin.setRange(0.5, 120.0)
            self.movie_duration_spin.setValue(12.0)
            self.movie_duration_spin.setSuffix(" s")
            self.movie_duration_spin.setToolTip("Movie duration for one complete 360-degree orbit.")
            still_button = QtWidgets.QPushButton("Export still image")
            still_button.setObjectName("volume_export_still_button")
            still_button.setToolTip("Save a PNG image of the current 3D camera view.")
            still_button.clicked.connect(self.export_still)
            movie_button = QtWidgets.QPushButton("Export rotation MP4")
            movie_button.setObjectName("volume_export_movie_button")
            movie_button.setToolTip("Render one complete rotation around the selected axis, then restore the current camera view.")
            movie_button.clicked.connect(self.export_rotation_movie)
            animation_layout.addWidget(self.rotate_check, 0, 0, 1, 2)
            animation_layout.addWidget(QtWidgets.QLabel("Rotation axis"), 1, 0)
            animation_layout.addWidget(self.rotation_axis_combo, 1, 1)
            animation_layout.addWidget(QtWidgets.QLabel("Preview speed"), 2, 0)
            animation_layout.addWidget(self.rotation_speed_spin, 2, 1)
            animation_layout.addWidget(QtWidgets.QLabel("Movie FPS"), 3, 0)
            animation_layout.addWidget(self.movie_fps_spin, 3, 1)
            animation_layout.addWidget(QtWidgets.QLabel("Movie duration"), 4, 0)
            animation_layout.addWidget(self.movie_duration_spin, 4, 1)
            animation_layout.addWidget(still_button, 5, 0, 1, 2)
            animation_layout.addWidget(movie_button, 6, 0, 1, 2)
            layout.addWidget(animation_group)
            layout.addStretch(1)
            splitter.addWidget(render_frame)
            splitter.addWidget(controls)
            splitter.setStretchFactor(0, 1)
            splitter.setSizes([980, 420])
            root.addWidget(splitter)

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
                centers = np.asarray(axis.centers, dtype=float)
                low_bound, high_bound = float(np.nanmin(centers)), float(np.nanmax(centers))
                midpoint = float(centers[centers.size // 2])
                value_spin = _float_spin(low_bound, high_bound)
                width_spin = _float_spin(0.0, max(high_bound - low_bound, 0.0))
                low_spin = _float_spin(low_bound, high_bound)
                high_spin = _float_spin(low_bound, high_bound)
                value_spin.setValue(midpoint)
                low_spin.setValue(midpoint)
                high_spin.setValue(midpoint)
                value_spin.setObjectName(f"volume_hidden_{dim}_value_spin")
                width_spin.setObjectName(f"volume_hidden_{dim}_width_spin")
                low_spin.setObjectName(f"volume_hidden_{dim}_low_spin")
                high_spin.setObjectName(f"volume_hidden_{dim}_high_spin")
                value_spin.setToolTip(f"Center value selected along hidden axis {axis.name}.")
                width_spin.setToolTip(f"Integration width along hidden axis {axis.name}.")
                low_spin.setToolTip(f"Lower integration bound along hidden axis {axis.name}.")
                high_spin.setToolTip(f"Upper integration bound along hidden axis {axis.name}.")
                integrate = QtWidgets.QCheckBox("Integrate range")
                integrate.setObjectName(f"volume_hidden_{dim}_integrate_check")
                integrate.setToolTip("Sum bins in the specified low/high range instead of taking one slice position.")
                controls = {
                    "centers": centers,
                    "value": value_spin,
                    "width": width_spin,
                    "low": low_spin,
                    "high": high_spin,
                    "integrate": integrate,
                }
                self.hidden_controls[dim] = controls
                value_spin.valueChanged.connect(lambda _value, d=dim: self._hidden_changed(d, "value"))
                width_spin.valueChanged.connect(lambda _value, d=dim: self._hidden_changed(d, "width"))
                low_spin.valueChanged.connect(lambda _value, d=dim: self._hidden_changed(d, "low"))
                high_spin.valueChanged.connect(lambda _value, d=dim: self._hidden_changed(d, "high"))
                integrate.toggled.connect(lambda _value, d=dim: self._hidden_changed(d, "integrate"))
                row.addWidget(QtWidgets.QLabel("Value"), 0, 0)
                row.addWidget(value_spin, 0, 1)
                row.addWidget(QtWidgets.QLabel("Width"), 1, 0)
                row.addWidget(width_spin, 1, 1)
                row.addWidget(QtWidgets.QLabel("Range low"), 2, 0)
                row.addWidget(low_spin, 2, 1)
                row.addWidget(QtWidgets.QLabel("Range high"), 3, 0)
                row.addWidget(high_spin, 3, 1)
                row.addWidget(integrate, 4, 0, 1, 2)
                row.setColumnStretch(1, 1)
                self.hidden_layout.addWidget(group)
            if not self.hidden_controls:
                self.hidden_layout.addWidget(QtWidgets.QLabel("All dimensions are displayed."))

        def _hidden_changed(self, dim, source):
            controls = self.hidden_controls[dim]
            if source in {"value", "width"}:
                center = float(controls["value"].value())
                half_width = 0.5 * float(controls["width"].value())
                low_bound = float(np.nanmin(controls["centers"]))
                high_bound = float(np.nanmax(controls["centers"]))
                low = max(low_bound, center - half_width)
                high = min(high_bound, center + half_width)
                self._set_spin_value(controls["low"], low)
                self._set_spin_value(controls["high"], high)
            elif source in {"low", "high"}:
                low, high = sorted((float(controls["low"].value()), float(controls["high"].value())))
                self._set_spin_value(controls["low"], low)
                self._set_spin_value(controls["high"], high)
                self._set_spin_value(controls["value"], 0.5 * (low + high))
                self._set_spin_value(controls["width"], high - low)
            self._channel_changed()

        def _set_spin_value(self, spin, value):
            blocked = spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(blocked)

        def _selections(self):
            selections = {}
            for dim, controls in self.hidden_controls.items():
                centers = controls["centers"]
                if controls["integrate"].isChecked():
                    low = int(np.argmin(np.abs(centers - controls["low"].value())))
                    high = int(np.argmin(np.abs(centers - controls["high"].value())))
                    selections[dim] = tuple(sorted((low, high)))
                else:
                    selections[dim] = int(np.argmin(np.abs(centers - controls["value"].value())))
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
            if self.render_combo.currentText() == "Isosurface":
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
