from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mdhisto import MDHistoData
from .plotting import MDHistoSliceViewer


@dataclass
class _HiddenAxisControls:
    integrate: Any
    value: Any
    width: Any
    low: Any
    high: Any
    slider: Any
    centers: np.ndarray
    syncing: bool = False


class QtMDHistoSliceViewer:
    """PySide6 slice viewer with an embedded Matplotlib canvas."""

    def __init__(
        self,
        data: MDHistoData,
        *,
        x_dim: int | str = -1,
        y_dim: int | str = 0,
        channel: str = "signal",
        cmap: str = "viridis",
        color_scale: str = "linear",
        auto_limits: str = "min/max",
        integrate: bool = False,
        masked: bool = True,
    ) -> None:
        self.app = _qt_app()
        self.model = MDHistoSliceViewer(
            data,
            x_dim=x_dim,
            y_dim=y_dim,
            channel=channel,
            cmap=cmap,
            color_scale=color_scale,
            auto_limits=auto_limits,
            integrate=integrate,
            masked=masked,
        )
        self.data = data
        self.window = None
        self.canvas = None
        self.figure = None
        self.grid = None
        self.controls_scroll = None
        self.ax_image = None
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.image = None
        self.colorbar = None
        self.rectangle_selector = None
        self.x_combo = None
        self.y_combo = None
        self.x_min_spin = None
        self.x_max_spin = None
        self.y_min_spin = None
        self.y_max_spin = None
        self.x_reset_button = None
        self.y_reset_button = None
        self.cmap_combo = None
        self.channel_combo = None
        self.scale_combo = None
        self.limits_combo = None
        self.autoscale_check = None
        self.vmin_spin = None
        self.vmax_spin = None
        self.gamma_label = None
        self.gamma_spin = None
        self.limit_n_label = None
        self.limit_n_spin = None
        self.hover_label = None
        self.roi_button = None
        self.show_box_check = None
        self.hist_axes_check = None
        self.xcut_percent_slider = None
        self.ycut_percent_slider = None
        self.xcut_percent_label = None
        self.ycut_percent_label = None
        self.roi_x_center_spin = None
        self.roi_x_width_spin = None
        self.roi_y_center_spin = None
        self.roi_y_width_spin = None
        self.font_size_spin = None
        self.copy_figure_button = None
        self.copy_script_button = None
        self.save_script_button = None
        self.xcut_percent = 20
        self.ycut_percent = 16
        self.font_size = 10.0
        self.hidden_layout = None
        self.hidden_controls: dict[int, _HiddenAxisControls] = {}
        self._current_slice: dict[str, np.ndarray] | None = None
        self._syncing_axes = False
        self._syncing_limits = False
        self._syncing_view_limits = False
        self._syncing_roi_controls = False
        self._roi_extents: tuple[float, float, float, float] | None = None
        self._view_limit_callback_ids: list[int] = []
        self._build()
        self.update_plot()

    def show(self) -> "QtMDHistoSliceViewer":
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self

    def run(self) -> int:
        self.show()
        return int(self.app.exec())

    def slice_arrays(self) -> dict[str, np.ndarray]:
        return self.model.slice_arrays()

    @property
    def image_norm(self):
        return None if self.image is None else self.image.norm

    def copy_figure_to_clipboard(self) -> None:
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())

    def copy_script_to_clipboard(self) -> None:
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setText(self.figure_script())

    def save_script(self) -> None:
        from pathlib import Path

        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save figure script",
            "mdhisto_slice_figure.py",
            "Python scripts (*.py);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        Path(path).write_text(self.figure_script(), encoding="utf-8")

    def figure_script(self) -> str:
        source_file = self.data.metadata.get("source_file") if isinstance(self.data.metadata, dict) else None
        data_line = (
            f"data = load_mantid_mdhisto_nxs({source_file!r}, copy_metadata=False)"
            if source_file
            else "data = ...  # Replace with your MDHistoData object"
        )
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                "from metallix import load_mantid_mdhisto_nxs, plot_mdhisto_slice",
                "",
                data_line,
                "fig = plot_mdhisto_slice(",
                "    data,",
                f"    x_dim={self.data.axes[self.model.x_dim].name!r},",
                f"    y_dim={self.data.axes[self.model.y_dim].name!r},",
                f"    channel={self.model.channel!r},",
                f"    selections={self._export_selections()!r},",
                f"    integrate_checks={self._export_integrate_checks()!r},",
                f"    cmap={self.model.cmap!r},",
                f"    color_scale={self.model.color_scale!r},",
                f"    auto_limits={self.model.auto_limits!r},",
                f"    autoscale={self.model.autoscale!r},",
                f"    manual_vmin={self.model.manual_vmin!r},",
                f"    manual_vmax={self.model.manual_vmax!r},",
                f"    sigma_n={self.model.sigma_n!r},",
                f"    iqr_n={self.model.iqr_n!r},",
                f"    percentile_n={self.model.percentile_n!r},",
                f"    power_gamma={self.model.power_gamma!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    show_histogram_axes={self.hist_axes_check.isChecked()!r},",
                f"    roi_extents={self._roi_extents!r},",
                f"    xcut_percent={self.xcut_percent!r},",
                f"    ycut_percent={self.ycut_percent!r},",
                ")",
                "plt.show()",
                "",
            ]
        )

    def _export_selections(self) -> dict[int, tuple[float, float]]:
        return {int(dim): (float(values[0]), float(values[1])) for dim, values in self.model.selections.items()}

    def _export_integrate_checks(self) -> dict[int, bool]:
        return {int(dim): bool(value) for dim, value in self.model.integrate_checks.items()}

    def _export_limits(self, axis_name: str) -> tuple[float, float]:
        values = self.ax_image.get_xlim() if axis_name == "x" else self.ax_image.get_ylim()
        return float(values[0]), float(values[1])

    def _build(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PySide6 import QtCore, QtGui, QtWidgets

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("metallix MDHisto Slice Viewer")
        self.window.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.grid = self.figure.add_gridspec(
            2,
            3,
            width_ratios=[1.0, self._panel_ratio(self.ycut_percent), 0.045],
            height_ratios=[1.0, self._panel_ratio(self.xcut_percent)],
        )
        self.ax_image = self.figure.add_subplot(self.grid[0, 0])
        self.ax_ycut = self.figure.add_subplot(self.grid[0, 1], sharey=self.ax_image)
        self.ax_colorbar = self.figure.add_subplot(self.grid[0, 2])
        self.ax_xcut = self.figure.add_subplot(self.grid[1, 0], sharex=self.ax_image)
        self.canvas = FigureCanvasQTAgg(self.figure)
        toolbar = NavigationToolbar2QT(self.canvas, self.window)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas, 1)

        controls = QtWidgets.QScrollArea()
        self.controls_scroll = controls
        controls.setWidgetResizable(True)
        controls.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls.setMinimumWidth(340)
        controls.setMaximumWidth(380)
        controls_widget = QtWidgets.QWidget()
        controls.setWidget(controls_widget)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        axes_group = QtWidgets.QGroupBox("Displayed axes")
        axes_form = QtWidgets.QFormLayout(axes_group)
        axes_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        labels = [axis.name for axis in self.data.axes]
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self.x_combo.addItems(labels)
        self.y_combo.addItems(labels)
        _compact_combobox(self.x_combo)
        _compact_combobox(self.y_combo)
        self.x_combo.setCurrentIndex(self.model.x_dim)
        self.y_combo.setCurrentIndex(self.model.y_dim)
        self.x_combo.currentIndexChanged.connect(lambda index: self._set_display_dim("x", index))
        self.y_combo.currentIndexChanged.connect(lambda index: self._set_display_dim("y", index))
        axes_form.addRow("x", self.x_combo)
        axes_form.addRow("y", self.y_combo)
        self.x_min_spin = _make_float_spinbox()
        self.x_max_spin = _make_float_spinbox()
        self.y_min_spin = _make_float_spinbox()
        self.y_max_spin = _make_float_spinbox()
        self.x_reset_button = QtWidgets.QPushButton("Reset")
        self.y_reset_button = QtWidgets.QPushButton("Reset")
        self.x_reset_button.setMaximumWidth(64)
        self.y_reset_button.setMaximumWidth(64)
        x_limits_widget = QtWidgets.QWidget()
        x_limits_layout = QtWidgets.QGridLayout(x_limits_widget)
        x_limits_layout.setContentsMargins(0, 0, 0, 0)
        x_limits_layout.setHorizontalSpacing(6)
        x_limits_layout.addWidget(QtWidgets.QLabel("min"), 0, 0)
        x_limits_layout.addWidget(self.x_min_spin, 0, 1)
        x_limits_layout.addWidget(QtWidgets.QLabel("max"), 1, 0)
        x_limits_layout.addWidget(self.x_max_spin, 1, 1)
        x_limits_layout.addWidget(self.x_reset_button, 0, 2, 2, 1)
        y_limits_widget = QtWidgets.QWidget()
        y_limits_layout = QtWidgets.QGridLayout(y_limits_widget)
        y_limits_layout.setContentsMargins(0, 0, 0, 0)
        y_limits_layout.setHorizontalSpacing(6)
        y_limits_layout.addWidget(QtWidgets.QLabel("min"), 0, 0)
        y_limits_layout.addWidget(self.y_min_spin, 0, 1)
        y_limits_layout.addWidget(QtWidgets.QLabel("max"), 1, 0)
        y_limits_layout.addWidget(self.y_max_spin, 1, 1)
        y_limits_layout.addWidget(self.y_reset_button, 0, 2, 2, 1)
        self.x_min_spin.valueChanged.connect(lambda _value: self._set_view_limits("x"))
        self.x_max_spin.valueChanged.connect(lambda _value: self._set_view_limits("x"))
        self.y_min_spin.valueChanged.connect(lambda _value: self._set_view_limits("y"))
        self.y_max_spin.valueChanged.connect(lambda _value: self._set_view_limits("y"))
        self.x_reset_button.clicked.connect(lambda: self._reset_view_limits("x"))
        self.y_reset_button.clicked.connect(lambda: self._reset_view_limits("y"))
        axes_form.addRow("x limits", x_limits_widget)
        axes_form.addRow("y limits", y_limits_widget)
        controls_layout.addWidget(axes_group)

        self.hidden_group = QtWidgets.QGroupBox("Integrated Axes")
        self.hidden_layout = QtWidgets.QVBoxLayout(self.hidden_group)
        controls_layout.addWidget(self.hidden_group)
        self._rebuild_hidden_axis_controls()

        color_group = QtWidgets.QGroupBox("Color")
        color_form = QtWidgets.QFormLayout(color_group)
        color_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(self.model.COLORMAPS)
        _compact_combobox(self.cmap_combo)
        self.cmap_combo.setCurrentText(self.model.cmap)
        self.cmap_combo.currentTextChanged.connect(self._set_cmap)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(self.model.CHANNELS)
        _compact_combobox(self.channel_combo)
        self.channel_combo.setCurrentText(self.model.channel)
        self.channel_combo.currentTextChanged.connect(self._set_channel)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(self.model.COLOR_SCALES)
        _compact_combobox(self.scale_combo)
        self.scale_combo.setCurrentText(self.model.color_scale)
        self.scale_combo.currentTextChanged.connect(self._set_color_scale)
        self.limits_combo = QtWidgets.QComboBox()
        self.limits_combo.addItems(self.model.AUTO_LIMITS)
        _compact_combobox(self.limits_combo)
        self.limits_combo.setCurrentText(self.model.auto_limits)
        self.limits_combo.currentTextChanged.connect(self._set_auto_limits)
        self.autoscale_check = QtWidgets.QCheckBox("Autoscale")
        self.autoscale_check.setChecked(self.model.autoscale)
        self.autoscale_check.toggled.connect(self._set_autoscale)
        self.vmin_spin = _make_float_spinbox()
        self.vmax_spin = _make_float_spinbox()
        self.gamma_spin = _make_float_spinbox(1.0e-6, 20.0)
        self.gamma_spin.setValue(self.model.power_gamma)
        self.limit_n_spin = _make_float_spinbox(0.0, 50.0)
        self.limit_n_spin.setValue(self._current_limit_n())
        self.vmin_spin.valueChanged.connect(lambda value: self._set_manual_limit("vmin", value))
        self.vmax_spin.valueChanged.connect(lambda value: self._set_manual_limit("vmax", value))
        self.gamma_spin.valueChanged.connect(self._set_power_gamma)
        self.limit_n_spin.valueChanged.connect(self._set_limit_n)
        color_form.addRow("Channel", self.channel_combo)
        color_form.addRow("Colormap", self.cmap_combo)
        color_form.addRow("Scale", self.scale_combo)
        color_form.addRow("Auto limits", self.limits_combo)
        color_form.addRow("", self.autoscale_check)
        color_form.addRow("vmin", self.vmin_spin)
        color_form.addRow("vmax", self.vmax_spin)
        self.gamma_label = QtWidgets.QLabel("gamma")
        color_form.addRow(self.gamma_label, self.gamma_spin)
        self.gamma_label.setVisible(self.model.color_scale == "power")
        self.gamma_spin.setVisible(self.model.color_scale == "power")
        self.limit_n_label = QtWidgets.QLabel("N")
        color_form.addRow(self.limit_n_label, self.limit_n_spin)
        self._sync_limit_n_visibility()
        controls_layout.addWidget(color_group)

        tools_group = QtWidgets.QGroupBox("Histogram box cuts")
        tools_layout = QtWidgets.QGridLayout(tools_group)
        self.roi_button = QtWidgets.QPushButton("Box tool")
        self.roi_button.setCheckable(True)
        self.roi_button.setToolTip("Toggle the rectangle tool used to populate the x/y cut axes.")
        self.roi_button.toggled.connect(self._set_roi_enabled)
        self.show_box_check = QtWidgets.QCheckBox("Show box tool")
        self.show_box_check.setChecked(False)
        self.show_box_check.setToolTip("Show or hide the rectangle selection tool.")
        self.show_box_check.toggled.connect(self._set_box_tool_visible)
        self.hist_axes_check = QtWidgets.QCheckBox("Show histogram axes")
        self.hist_axes_check.setChecked(False)
        self.hist_axes_check.setToolTip("Show or hide the x/y histogram cut panels beside the image.")
        self.hist_axes_check.toggled.connect(self._set_histogram_axes_visible)
        self.xcut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ycut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        for slider, value in (
            (self.xcut_percent_slider, self.xcut_percent),
            (self.ycut_percent_slider, self.ycut_percent),
        ):
            slider.setRange(5, 45)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.setValue(value)
        self.xcut_percent_label = QtWidgets.QLabel()
        self.ycut_percent_label = QtWidgets.QLabel()
        self.roi_x_center_spin = _make_float_spinbox()
        self.roi_x_width_spin = _make_float_spinbox(0.0, 1.0e12)
        self.roi_y_center_spin = _make_float_spinbox()
        self.roi_y_width_spin = _make_float_spinbox(0.0, 1.0e12)
        self.xcut_percent_slider.valueChanged.connect(self._set_xcut_percent)
        self.ycut_percent_slider.valueChanged.connect(self._set_ycut_percent)
        self.roi_x_center_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_x_width_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_y_center_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_y_width_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        tools_layout.addWidget(self.roi_button, 0, 0, 1, 2)
        tools_layout.addWidget(self.show_box_check, 1, 0, 1, 2)
        tools_layout.addWidget(self.hist_axes_check, 2, 0, 1, 2)
        tools_layout.addWidget(self.xcut_percent_label, 3, 0)
        tools_layout.addWidget(self.xcut_percent_slider, 3, 1)
        tools_layout.addWidget(self.ycut_percent_label, 4, 0)
        tools_layout.addWidget(self.ycut_percent_slider, 4, 1)
        tools_layout.addWidget(QtWidgets.QLabel("x center"), 5, 0)
        tools_layout.addWidget(self.roi_x_center_spin, 5, 1)
        tools_layout.addWidget(QtWidgets.QLabel("x width"), 6, 0)
        tools_layout.addWidget(self.roi_x_width_spin, 6, 1)
        tools_layout.addWidget(QtWidgets.QLabel("y center"), 7, 0)
        tools_layout.addWidget(self.roi_y_center_spin, 7, 1)
        tools_layout.addWidget(QtWidgets.QLabel("y width"), 8, 0)
        tools_layout.addWidget(self.roi_y_width_spin, 8, 1)
        tools_layout.setColumnMinimumWidth(1, 120)
        tools_layout.setColumnStretch(1, 1)
        self._sync_histogram_panel_controls()
        controls_layout.addWidget(tools_group)

        figure_group = QtWidgets.QGroupBox("Figure")
        figure_form = QtWidgets.QFormLayout(figure_group)
        figure_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.font_size_spin = _make_float_spinbox(4.0, 48.0)
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setValue(self.font_size)
        self.font_size_spin.valueChanged.connect(self._set_font_size)
        self.copy_figure_button = QtWidgets.QPushButton("Copy figure")
        self.copy_script_button = QtWidgets.QPushButton("Copy script")
        self.save_script_button = QtWidgets.QPushButton("Save script")
        self.copy_figure_button.clicked.connect(self.copy_figure_to_clipboard)
        self.copy_script_button.clicked.connect(self.copy_script_to_clipboard)
        self.save_script_button.clicked.connect(self.save_script)
        figure_form.addRow("Font size", self.font_size_spin)
        figure_form.addRow("", self.copy_figure_button)
        figure_form.addRow("", self.copy_script_button)
        figure_form.addRow("", self.save_script_button)
        controls_layout.addWidget(figure_group)

        hover_group = QtWidgets.QGroupBox("Cursor")
        hover_layout = QtWidgets.QVBoxLayout(hover_group)
        self.hover_label = QtWidgets.QLabel("x: -\ny: -\nI: -\nerr: -")
        self.hover_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        hover_layout.addWidget(self.hover_label)
        controls_layout.addWidget(hover_group)
        controls_layout.addStretch(1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(plot_panel)
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        from matplotlib.widgets import RectangleSelector

        self.rectangle_selector = RectangleSelector(
            self.ax_image,
            self._on_rectangle,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=True,
        )
        self.rectangle_selector.set_active(False)
        self._set_box_tool_visible(self.show_box_check.isChecked())
        self._connect_view_limit_callbacks()
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self.window)
        copy_shortcut.activated.connect(self.copy_figure_to_clipboard)

    def _rebuild_hidden_axis_controls(self) -> None:
        from PySide6 import QtWidgets

        _clear_layout(self.hidden_layout)
        self.hidden_controls = {}
        hidden_dims = [dim for dim in range(self.data.signal.ndim) if dim not in (self.model.x_dim, self.model.y_dim)]
        if not hidden_dims:
            self.hidden_layout.addWidget(QtWidgets.QLabel("No hidden axes."))
            return
        for dim in hidden_dims:
            axis = self.data.axes[dim]
            centers = axis.centers
            low_value = float(np.nanmin(centers))
            high_value = float(np.nanmax(centers))
            current = self.model.selections.get(dim, (float(centers[centers.size // 2]),) * 2)

            group = QtWidgets.QGroupBox(axis.name)
            grid = QtWidgets.QGridLayout(group)
            value_spin = _make_float_spinbox(low_value, high_value)
            width_spin = _make_float_spinbox(0.0, max(high_value - low_value, 0.0))
            low_spin = _make_float_spinbox(low_value, high_value)
            high_spin = _make_float_spinbox(low_value, high_value)
            range_slider = _IntegratedAxisSlider(centers.size)
            integrate_check = QtWidgets.QCheckBox("Integrate range")
            integrate_check.setChecked(self.model.integrate_checks.get(dim, self.model.integrate))
            midpoint = 0.5 * (float(current[0]) + float(current[1]))
            value_index = _nearest_index(centers, midpoint)
            low_index = _nearest_index(centers, float(min(current)))
            high_index = _nearest_index(centers, float(max(current)))
            if low_index == high_index and centers.size > 1:
                high_index = min(low_index + 1, centers.size - 1)
                if high_index == low_index:
                    low_index = max(low_index - 1, 0)
            controls = _HiddenAxisControls(
                integrate_check,
                value_spin,
                width_spin,
                low_spin,
                high_spin,
                range_slider,
                centers,
                True,
            )
            self.hidden_controls[dim] = controls
            self._set_hidden_controls_indices(dim, value_index, low_index, high_index)
            controls.syncing = False
            value_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "value_spin")
            )
            width_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "width_spin")
            )
            low_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "low_spin")
            )
            high_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "high_spin")
            )
            range_slider.changed.connect(
                lambda source, axis_dim=dim: self._sync_hidden_axis(axis_dim, source)
            )
            integrate_check.toggled.connect(
                lambda _checked, axis_dim=dim: self._sync_hidden_axis(axis_dim, "integrate")
            )
            grid.addWidget(QtWidgets.QLabel("Value"), 0, 0)
            grid.addWidget(value_spin, 0, 1)
            grid.addWidget(QtWidgets.QLabel("Width"), 0, 2)
            grid.addWidget(width_spin, 0, 3)
            grid.addWidget(QtWidgets.QLabel("Range low"), 1, 0)
            grid.addWidget(low_spin, 1, 1)
            grid.addWidget(QtWidgets.QLabel("Range high"), 2, 0)
            grid.addWidget(high_spin, 2, 1)
            grid.addWidget(integrate_check, 2, 2, 1, 2)
            grid.addWidget(range_slider, 3, 0, 1, 4)
            grid.setColumnMinimumWidth(1, 104)
            grid.setColumnMinimumWidth(3, 104)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 0)
            grid.setColumnStretch(3, 0)
            self.hidden_layout.addWidget(group)

    def _sync_hidden_axis(self, dim: int, source: str) -> None:
        controls = self.hidden_controls[dim]
        if controls.syncing:
            return
        centers = controls.centers
        value_index = controls.slider.value_index
        low_index = controls.slider.low_index
        high_index = controls.slider.high_index
        if source == "value_spin":
            new_value_index = _nearest_index(centers, float(controls.value.value()))
            value_index, low_index, high_index = self._shift_hidden_range(
                centers, new_value_index, low_index, high_index
            )
        elif source == "value":
            value_index, low_index, high_index = self._shift_hidden_range(
                centers, value_index, low_index, high_index
            )
        elif source == "width_spin":
            self._set_integrate_checked(controls, True)
            low_index, high_index = self._range_from_width(
                centers, value_index, float(controls.width.value())
            )
        elif source == "low_spin":
            self._set_integrate_checked(controls, True)
            low_index = _nearest_index(centers, float(controls.low.value()))
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "high_spin":
            self._set_integrate_checked(controls, True)
            high_index = _nearest_index(centers, float(controls.high.value()))
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "low":
            self._set_integrate_checked(controls, True)
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "high":
            self._set_integrate_checked(controls, True)
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "integrate":
            pass
        self._set_hidden_controls_indices(dim, value_index, low_index, high_index)
        self.model.integrate_checks[dim] = bool(controls.integrate.isChecked())
        if controls.integrate.isChecked():
            self.model.selections[dim] = (float(controls.low.value()), float(controls.high.value()))
        else:
            value = float(controls.value.value())
            self.model.selections[dim] = (value, value)
        self.update_plot()

    def _set_hidden_controls_indices(self, dim: int, value_index: int, low_index: int, high_index: int) -> None:
        controls = self.hidden_controls[dim]
        centers = controls.centers
        value_index = int(np.clip(value_index, 0, centers.size - 1))
        low_index = int(np.clip(low_index, 0, centers.size - 1))
        high_index = int(np.clip(high_index, 0, centers.size - 1))
        low_index, high_index = sorted((low_index, high_index))
        controls.syncing = True
        try:
            controls.slider.set_indices(value_index, low_index, high_index)
            controls.slider.set_integrate_range(controls.integrate.isChecked())
            controls.value.setValue(float(centers[value_index]))
            controls.low.setValue(float(centers[low_index]))
            controls.high.setValue(float(centers[high_index]))
            controls.width.setValue(float(centers[high_index] - centers[low_index]))
        finally:
            controls.syncing = False

    def _shift_hidden_range(
        self, centers: np.ndarray, value_index: int, low_index: int, high_index: int
    ) -> tuple[int, int, int]:
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
        low_index = max(low_index, 0)
        return value_index, low_index, high_index

    def _range_from_width(self, centers: np.ndarray, value_index: int, width: float) -> tuple[int, int]:
        if centers.size <= 1:
            return 0, 0
        value = float(centers[value_index])
        low = _nearest_index(centers, value - 0.5 * max(width, 0.0))
        high = _nearest_index(centers, value + 0.5 * max(width, 0.0))
        if low == high and width > 0:
            if high < centers.size - 1:
                high += 1
            elif low > 0:
                low -= 1
        return tuple(sorted((low, high)))

    def _set_integrate_checked(self, controls: _HiddenAxisControls, checked: bool) -> None:
        was_syncing = controls.syncing
        controls.syncing = True
        try:
            controls.integrate.setChecked(checked)
        finally:
            controls.syncing = was_syncing

    def _set_display_dim(self, axis_name: str, dim: int) -> None:
        if self._syncing_axes:
            return
        if axis_name == "x" and dim == self.model.y_dim:
            self.model.x_dim, self.model.y_dim = self.model.y_dim, self.model.x_dim
            self._sync_axis_combos()
        elif axis_name == "y" and dim == self.model.x_dim:
            self.model.x_dim, self.model.y_dim = self.model.y_dim, self.model.x_dim
            self._sync_axis_combos()
        elif axis_name == "x":
            self.model.x_dim = int(dim)
        else:
            self.model.y_dim = int(dim)
        self._rebuild_hidden_axis_controls()
        self.update_plot(preserve_view=False)

    def _sync_axis_combos(self) -> None:
        self._syncing_axes = True
        try:
            self.x_combo.setCurrentIndex(self.model.x_dim)
            self.y_combo.setCurrentIndex(self.model.y_dim)
        finally:
            self._syncing_axes = False

    def _set_cmap(self, cmap: str) -> None:
        self.model.cmap = str(cmap)
        self.update_plot()

    def _set_channel(self, channel: str) -> None:
        self.model.channel = self.model._resolve_channel(channel)
        self.update_plot()

    def _set_color_scale(self, color_scale: str) -> None:
        self.model.color_scale = str(color_scale)
        self.gamma_label.setVisible(self.model.color_scale == "power")
        self.gamma_spin.setVisible(self.model.color_scale == "power")
        self.update_plot()

    def _set_auto_limits(self, auto_limits: str) -> None:
        self.model.auto_limits = str(auto_limits)
        self.limit_n_spin.setValue(self._current_limit_n())
        self._sync_limit_n_visibility()
        self._set_autoscale_checkbox(True)
        self.update_plot()

    def _set_autoscale(self, autoscale: bool) -> None:
        if self._syncing_limits:
            return
        self.model.autoscale = bool(autoscale)
        self.update_plot()

    def _set_manual_limit(self, which: str, value: float) -> None:
        if self._syncing_limits:
            return
        if which == "vmin":
            self.model.manual_vmin = float(value)
        else:
            self.model.manual_vmax = float(value)
        self._set_autoscale_checkbox(False)
        self.update_plot()

    def _set_power_gamma(self, gamma: float) -> None:
        self.model.power_gamma = max(float(gamma), 1.0e-12)
        if self.model.color_scale == "power":
            self.update_plot()

    def _set_limit_n(self, value: float) -> None:
        if self.model.auto_limits == "N-sigma":
            self.model.sigma_n = max(float(value), 0.0)
        elif self.model.auto_limits == "N IQR":
            self.model.iqr_n = max(float(value), 0.0)
        elif self.model.auto_limits == "Nth percentile":
            self.model.percentile_n = float(np.clip(value, 0.0, 50.0))
        if self.model.auto_limits != "min/max":
            self._set_autoscale_checkbox(True)
            self.update_plot()

    def _set_font_size(self, value: float) -> None:
        self.font_size = float(value)
        self._apply_figure_font_size()
        self.canvas.draw_idle()

    def _apply_figure_font_size(self) -> None:
        if self.figure is None:
            return
        size = float(self.font_size)
        for axis in (self.ax_image, self.ax_xcut, self.ax_ycut, self.ax_colorbar):
            if axis is None:
                continue
            axis.title.set_fontsize(size + 1.0)
            axis.xaxis.label.set_fontsize(size)
            axis.yaxis.label.set_fontsize(size)
            axis.tick_params(axis="both", labelsize=size)
        if self.colorbar is not None:
            self.colorbar.ax.yaxis.label.set_fontsize(size)
            self.colorbar.ax.tick_params(labelsize=size)

    def _current_limit_n(self) -> float:
        if self.model.auto_limits == "N-sigma":
            return float(self.model.sigma_n)
        if self.model.auto_limits == "N IQR":
            return float(self.model.iqr_n)
        if self.model.auto_limits == "Nth percentile":
            return float(self.model.percentile_n)
        return 1.0

    def _sync_limit_n_visibility(self) -> None:
        visible = self.model.auto_limits != "min/max"
        self.limit_n_label.setVisible(visible)
        self.limit_n_spin.setVisible(visible)

    def _set_autoscale_checkbox(self, checked: bool) -> None:
        self._syncing_limits = True
        try:
            self.autoscale_check.setChecked(checked)
        finally:
            self._syncing_limits = False
        self.model.autoscale = bool(checked)

    def _connect_view_limit_callbacks(self) -> None:
        if self.ax_image is None:
            return
        for callback_id in self._view_limit_callback_ids:
            try:
                self.ax_image.callbacks.disconnect(callback_id)
            except Exception:
                pass
        self._view_limit_callback_ids = [
            self.ax_image.callbacks.connect("xlim_changed", lambda _axis: self._sync_view_limit_controls("x")),
            self.ax_image.callbacks.connect("ylim_changed", lambda _axis: self._sync_view_limit_controls("y")),
        ]

    def _set_view_limits(self, axis_name: str) -> None:
        if self._syncing_view_limits or self.ax_image is None:
            return
        if axis_name == "x":
            low = float(self.x_min_spin.value())
            high = float(self.x_max_spin.value())
            if low == high:
                return
            self.ax_image.set_xlim(low, high)
        else:
            low = float(self.y_min_spin.value())
            high = float(self.y_max_spin.value())
            if low == high:
                return
            self.ax_image.set_ylim(low, high)
        self.canvas.draw_idle()

    def _reset_view_limits(self, axis_name: str) -> None:
        if self.ax_image is None:
            return
        if axis_name == "x":
            self.ax_image.set_xlim(*self._default_view_limits(self.model.x_dim))
        else:
            self.ax_image.set_ylim(*self._default_view_limits(self.model.y_dim))
        self.canvas.draw_idle()

    def _sync_view_limit_controls(self, axis_name: str | None = None) -> None:
        if self.ax_image is None or self.x_min_spin is None:
            return
        self._syncing_view_limits = True
        try:
            if axis_name in (None, "x"):
                xmin, xmax = self.ax_image.get_xlim()
                self.x_min_spin.setValue(float(xmin))
                self.x_max_spin.setValue(float(xmax))
                self._set_axis_spin_steps((self.x_min_spin, self.x_max_spin), self.model.x_dim)
            if axis_name in (None, "y"):
                ymin, ymax = self.ax_image.get_ylim()
                self.y_min_spin.setValue(float(ymin))
                self.y_max_spin.setValue(float(ymax))
                self._set_axis_spin_steps((self.y_min_spin, self.y_max_spin), self.model.y_dim)
        finally:
            self._syncing_view_limits = False

    def _default_view_limits(self, dim: int) -> tuple[float, float]:
        edges = self.model._axis_edges(dim)
        return float(edges[0]), float(edges[-1])

    def _set_axis_spin_steps(self, spinboxes, dim: int) -> None:
        low, high = self._default_view_limits(dim)
        step = max(abs(high - low) / 500.0, 1.0e-6)
        for spinbox in spinboxes:
            spinbox.setSingleStep(step)

    def update_plot(self, *, preserve_view: bool = True) -> None:
        previous_xlim = self.ax_image.get_xlim() if preserve_view and self._current_slice is not None else None
        previous_ylim = self.ax_image.get_ylim() if preserve_view and self._current_slice is not None else None
        previous_dims = getattr(self, "_last_plot_dims", None)
        current_dims = (self.model.x_dim, self.model.y_dim)
        self._current_slice = self.model.slice_arrays()
        view = self._current_slice
        for axis in (self.ax_image, self.ax_xcut, self.ax_ycut):
            axis.clear()
        values = self.model._display_values(view)
        vmin, vmax = self.model._color_limits(values)
        norm = self.model._color_norm(values)
        self.image = self.ax_image.pcolormesh(
            view["x_edges"],
            view["y_edges"],
            values,
            shading="auto",
            cmap=self.model.cmap,
            norm=norm,
        )
        self.ax_image.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_ylabel(self.model._axis_label(self.model.y_dim))
        self.ax_image.set_title("MDHisto slice")
        if previous_xlim is not None and previous_ylim is not None and previous_dims == current_dims:
            self.ax_image.set_xlim(previous_xlim)
            self.ax_image.set_ylim(previous_ylim)
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(self.image, cax=self.ax_colorbar)
        else:
            self.colorbar.update_normal(self.image)
        self.colorbar.set_label(self.model._channel_label())
        self.ax_xcut.set_ylabel("Int.")
        self.ax_ycut.set_xlabel("Int.")
        self._sync_limit_spinboxes(vmin, vmax)
        self._last_plot_dims = current_dims
        self._sync_view_limit_controls()
        if self._roi_extents is None or previous_dims != current_dims:
            self._set_roi_extents(self._default_roi_extents(), update_cuts=False, draw=False)
        else:
            self._set_roi_extents(self._roi_extents, update_cuts=False, draw=False)
        self._apply_histogram_axes_layout(draw=False)
        self._apply_figure_font_size()
        self._connect_view_limit_callbacks()
        self.canvas.draw_idle()

    def _sync_limit_spinboxes(self, vmin: float, vmax: float) -> None:
        if not self.model.autoscale:
            return
        self._syncing_limits = True
        try:
            self.vmin_spin.setValue(float(vmin))
            self.vmax_spin.setValue(float(vmax))
        finally:
            self._syncing_limits = False

    def _set_roi_enabled(self, enabled: bool) -> None:
        if self.rectangle_selector is not None:
            visible = self.show_box_check is None or self.show_box_check.isChecked()
            self.rectangle_selector.set_active(bool(enabled) and visible)
            self._set_rectangle_selector_style(active=bool(enabled) and visible)
            self.canvas.draw_idle()

    def _set_box_tool_visible(self, visible: bool) -> None:
        self.roi_button.setEnabled(bool(visible))
        if not visible:
            self.roi_button.setChecked(False)
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_visible(bool(visible))
            self.rectangle_selector.set_active(bool(visible) and self.roi_button.isChecked())
            self._set_rectangle_selector_style(active=bool(visible) and self.roi_button.isChecked())
        self.canvas.draw_idle()

    def _set_rectangle_selector_style(self, *, active: bool) -> None:
        if self.rectangle_selector is None:
            return
        color = "#4f8bd6" if active else "#8a8a8a"
        self.rectangle_selector.set_props(
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
            alpha=1.0 if active else 0.75,
        )
        self.rectangle_selector.set_handle_props(
            markeredgecolor=color,
            markerfacecolor=color,
            alpha=1.0 if active else 0.75,
        )

    def _set_histogram_axes_visible(self, visible: bool) -> None:
        self._sync_histogram_panel_controls()
        self._apply_histogram_axes_layout(draw=True)

    def _set_xcut_percent(self, value: int) -> None:
        self.xcut_percent = int(value)
        self._sync_histogram_panel_controls()
        self._apply_histogram_axes_layout(draw=True)

    def _set_ycut_percent(self, value: int) -> None:
        self.ycut_percent = int(value)
        self._sync_histogram_panel_controls()
        self._apply_histogram_axes_layout(draw=True)

    def _sync_histogram_panel_controls(self) -> None:
        if self.hist_axes_check is None:
            return
        visible = bool(self.hist_axes_check.isChecked())
        self.xcut_percent_slider.setEnabled(visible)
        self.ycut_percent_slider.setEnabled(visible)
        self.xcut_percent_label.setText(f"X cut height: {self.xcut_percent}%")
        self.ycut_percent_label.setText(f"Y cut width: {self.ycut_percent}%")

    def _apply_histogram_axes_layout(self, *, draw: bool) -> None:
        if self.grid is None or self.hist_axes_check is None:
            return
        visible = bool(self.hist_axes_check.isChecked())
        x_ratio = self._panel_ratio(self.xcut_percent) if visible else 0.001
        y_ratio = self._panel_ratio(self.ycut_percent) if visible else 0.001
        self.grid.set_height_ratios([1.0, x_ratio])
        self.grid.set_width_ratios([1.0, y_ratio, 0.045])
        self.ax_xcut.set_visible(visible)
        self.ax_ycut.set_visible(visible)
        if draw:
            self.canvas.draw_idle()

    def _panel_ratio(self, percent: float) -> float:
        fraction = float(np.clip(percent, 1.0, 80.0)) / 100.0
        return fraction / (1.0 - fraction)

    def _set_roi_from_controls(self) -> None:
        if self._syncing_roi_controls:
            return
        x_center = float(self.roi_x_center_spin.value())
        y_center = float(self.roi_y_center_spin.value())
        x_width = max(float(self.roi_x_width_spin.value()), 0.0)
        y_width = max(float(self.roi_y_width_spin.value()), 0.0)
        extents = (
            x_center - 0.5 * x_width,
            x_center + 0.5 * x_width,
            y_center - 0.5 * y_width,
            y_center + 0.5 * y_width,
        )
        self._set_roi_extents(extents, update_cuts=True, draw=True)

    def _set_roi_extents(
        self,
        extents: tuple[float, float, float, float],
        *,
        update_cuts: bool,
        draw: bool,
    ) -> None:
        x0, x1, y0, y1 = self._normalize_roi_extents(extents)
        if self.rectangle_selector is not None:
            self.rectangle_selector.extents = (x0, x1, y0, y1)
            x0, x1, y0, y1 = self._normalize_roi_extents(self.rectangle_selector.extents)
        self._roi_extents = (x0, x1, y0, y1)
        self._sync_roi_controls_from_extents()
        if update_cuts:
            self._update_histogram_cuts_from_extents(self._roi_extents)
        if draw:
            self.canvas.draw_idle()

    def _sync_roi_controls_from_extents(self) -> None:
        if self.roi_x_center_spin is None or self._roi_extents is None:
            return
        x0, x1, y0, y1 = self._roi_extents
        self._syncing_roi_controls = True
        try:
            self._set_axis_spin_steps((self.roi_x_center_spin, self.roi_x_width_spin), self.model.x_dim)
            self._set_axis_spin_steps((self.roi_y_center_spin, self.roi_y_width_spin), self.model.y_dim)
            self.roi_x_center_spin.setValue(0.5 * (x0 + x1))
            self.roi_x_width_spin.setValue(abs(x1 - x0))
            self.roi_y_center_spin.setValue(0.5 * (y0 + y1))
            self.roi_y_width_spin.setValue(abs(y1 - y0))
        finally:
            self._syncing_roi_controls = False

    def _default_roi_extents(self) -> tuple[float, float, float, float]:
        xmin, xmax = self._default_view_limits(self.model.x_dim)
        ymin, ymax = self._default_view_limits(self.model.y_dim)
        x_width = 0.5 * abs(xmax - xmin)
        y_width = 0.5 * abs(ymax - ymin)
        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)
        return (
            x_center - 0.5 * x_width,
            x_center + 0.5 * x_width,
            y_center - 0.5 * y_width,
            y_center + 0.5 * y_width,
        )

    def _normalize_roi_extents(self, extents) -> tuple[float, float, float, float]:
        x0, x1, y0, y1 = [float(value) for value in extents]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return x0, x1, y0, y1

    def _current_roi_extents(self, click=None, release=None) -> tuple[float, float, float, float] | None:
        if self.rectangle_selector is not None:
            extents = self._normalize_roi_extents(self.rectangle_selector.extents)
            if extents[0] != extents[1] and extents[2] != extents[3]:
                return extents
        if click is None or release is None:
            return self._roi_extents
        if click.xdata is None or release.xdata is None or click.ydata is None or release.ydata is None:
            return None
        return self._normalize_roi_extents((click.xdata, release.xdata, click.ydata, release.ydata))

    def _update_histogram_cuts_from_extents(self, extents: tuple[float, float, float, float] | None) -> None:
        if self._current_slice is None or extents is None:
            return
        x0, x1, y0, y1 = extents
        view = self._current_slice
        x_mask = (view["x_centers"] >= x0) & (view["x_centers"] <= x1)
        y_mask = (view["y_centers"] >= y0) & (view["y_centers"] <= y1)
        self.ax_xcut.clear()
        self.ax_ycut.clear()
        if np.any(x_mask) and np.any(y_mask):
            z = self.model._display_values(view)
            x_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=0)
            y_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=1)
            self.ax_xcut.plot(view["x_centers"][x_mask], x_cut, "-", lw=1.2)
            self.ax_ycut.plot(y_cut, view["y_centers"][y_mask], "-", lw=1.2)
        self.ax_xcut.set_ylabel("Int.")
        self.ax_xcut.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_ycut.set_xlabel("Int.")
        self.ax_ycut.set_ylabel(self.model._axis_label(self.model.y_dim))
        self._apply_histogram_axes_layout(draw=False)

    def _on_motion(self, event) -> None:
        if self._current_slice is None or event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        view = self._current_slice
        x_idx = int(np.searchsorted(view["x_edges"], event.xdata, side="right") - 1)
        y_idx = int(np.searchsorted(view["y_edges"], event.ydata, side="right") - 1)
        if not (0 <= x_idx < view["signal"].shape[1] and 0 <= y_idx < view["signal"].shape[0]):
            return
        values = self.model._display_values(view)
        self.hover_label.setText(
            "\n".join(
                [
                    f"x: {view['x_centers'][x_idx]:.6g}",
                    f"y: {view['y_centers'][y_idx]:.6g}",
                    *self._cursor_coordinate_lines(x_idx, y_idx),
                    f"{self.model._channel_label()}: {values[y_idx, x_idx]:.6g}",
                    f"err: {view['errors'][y_idx, x_idx]:.6g}",
                ]
            )
        )

    def _cursor_coordinate_lines(self, x_idx: int, y_idx: int) -> list[str]:
        coords = self._cursor_hkle(x_idx, y_idx)
        return [f"{name}: {coords[name]:.6g}" for name in ("H", "K", "L", "E") if np.isfinite(coords[name])]

    def _cursor_hkle(self, x_idx: int, y_idx: int) -> dict[str, float]:
        coords = {"H": 0.0, "K": 0.0, "L": 0.0, "E": np.nan}
        hidden = self.model._normalized_selections()
        for dim, axis in enumerate(self.data.axes):
            if dim == self.model.x_dim:
                value = self.data.axes[dim].centers[x_idx]
            elif dim == self.model.y_dim:
                value = self.data.axes[dim].centers[y_idx]
            else:
                selection = hidden.get(dim)
                if isinstance(selection, tuple):
                    value = float(np.mean(self.data.axes[dim].centers[selection[0] : selection[1] + 1]))
                elif selection is None:
                    continue
                else:
                    value = self.data.axes[dim].centers[int(selection)]
            for component, coefficient in _axis_components(axis.name).items():
                if component == "E":
                    coords["E"] = float(value)
                else:
                    coords[component] += float(coefficient) * float(value)
        return coords

    def _on_rectangle(self, click, release) -> None:
        extents = self._current_roi_extents(click, release)
        if extents is None:
            return
        self._set_roi_extents(extents, update_cuts=True, draw=True)


def _qt_app():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_float_spinbox(low: float = -1.0e12, high: float = 1.0e12):
    from PySide6 import QtWidgets

    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(6)
    spin.setSingleStep(max((high - low) / 200.0, 0.001) if np.isfinite(high - low) else 0.001)
    spin.setKeyboardTracking(False)
    spin.setMaximumWidth(118)
    spin.setMinimumWidth(92)
    spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
    return spin


def _compact_combobox(combo) -> None:
    from PySide6 import QtWidgets

    combo.setMaximumWidth(150)
    combo.setMinimumWidth(96)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)


def _make_index_slider(size: int):
    from PySide6 import QtCore, QtWidgets

    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, max(int(size) - 1, 0))
    slider.setSingleStep(1)
    slider.setPageStep(max(int(size) // 20, 1))
    return slider


def _qt_classes():
    from PySide6 import QtCore, QtGui, QtWidgets

    return QtCore, QtGui, QtWidgets


QtCore, QtGui, QtWidgets = _qt_classes()


class _IntegratedAxisSlider(QtWidgets.QWidget):
    changed = QtCore.Signal(str)

    def __init__(self, size: int, parent=None) -> None:
        super().__init__(parent)
        self.size = max(int(size), 1)
        self.value_index = 0
        self.low_index = 0
        self.high_index = 0
        self.integrate_range = False
        self._dragging: str | None = None
        self.setMinimumHeight(42)
        self.setMouseTracking(True)

    def set_indices(self, value_index: int, low_index: int, high_index: int) -> None:
        self.value_index = int(np.clip(value_index, 0, self.size - 1))
        self.low_index = int(np.clip(low_index, 0, self.size - 1))
        self.high_index = int(np.clip(high_index, 0, self.size - 1))
        self.low_index, self.high_index = sorted((self.low_index, self.high_index))
        self.update()

    def set_integrate_range(self, integrate_range: bool) -> None:
        self.integrate_range = bool(integrate_range)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self._track_rect()
        y = rect.center().y()
        painter.setPen(QtGui.QPen(QtGui.QColor("#7f7f7f"), 3))
        painter.drawLine(rect.left(), y, rect.right(), y)
        if self.integrate_range:
            low_x = self._index_to_x(self.low_index)
            high_x = self._index_to_x(self.high_index)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4f8bd6"), 5))
            painter.drawLine(low_x, y, high_x, y)
            self._draw_handle(painter, low_x, y, QtGui.QColor("#80b7ff"))
            self._draw_handle(painter, high_x, y, QtGui.QColor("#80b7ff"))
        self._draw_handle(painter, self._index_to_x(self.value_index), y, QtGui.QColor("#f4d35e"))

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._dragging = self._nearest_handle(event.position().x())
        self._set_dragged_index(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            self._set_dragged_index(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        del event
        self._dragging = None

    def _set_dragged_index(self, x: float) -> None:
        index = self._x_to_index(float(x))
        if self._dragging == "value":
            width = self.high_index - self.low_index
            half_low = width // 2
            half_high = width - half_low
            low = index - half_low
            high = index + half_high
            if low < 0:
                high -= low
                low = 0
            if high >= self.size:
                low -= high - (self.size - 1)
                high = self.size - 1
            self.set_indices(index, max(low, 0), high)
            self.changed.emit("value")
        elif self._dragging == "low":
            self.set_indices(int(round(0.5 * (index + self.high_index))), index, self.high_index)
            self.changed.emit("low")
        elif self._dragging == "high":
            self.set_indices(int(round(0.5 * (self.low_index + index))), self.low_index, index)
            self.changed.emit("high")

    def _nearest_handle(self, x: float) -> str:
        candidates = {"value": abs(x - self._index_to_x(self.value_index))}
        if self.integrate_range:
            candidates["low"] = abs(x - self._index_to_x(self.low_index))
            candidates["high"] = abs(x - self._index_to_x(self.high_index))
        return min(candidates, key=candidates.get)

    def _track_rect(self):
        margins = 16
        return self.rect().adjusted(margins, 0, -margins, 0)

    def _index_to_x(self, index: int) -> float:
        rect = self._track_rect()
        if self.size <= 1:
            return float(rect.center().x())
        return float(rect.left() + (rect.width() * index / (self.size - 1)))

    def _x_to_index(self, x: float) -> int:
        rect = self._track_rect()
        if self.size <= 1 or rect.width() <= 0:
            return 0
        fraction = (x - rect.left()) / rect.width()
        return int(np.clip(round(fraction * (self.size - 1)), 0, self.size - 1))

    def _draw_handle(self, painter, x: float, y: float, color) -> None:
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor("#303030"), 1))
        painter.drawEllipse(QtCore.QPointF(x, y), 6.0, 6.0)


def _nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - float(value))))


def _axis_components(name: str) -> dict[str, float]:
    text = name.strip()
    lower = text.lower()
    if "deltae" in lower or lower in {"e", "energy"}:
        return {"E": 1.0}
    if text.startswith("[") and text.endswith("]"):
        parts = [part.strip() for part in text[1:-1].split(",")]
        components: dict[str, float] = {}
        for component, part in zip(("H", "K", "L"), parts, strict=False):
            coefficient = _component_coefficient(part, component)
            if coefficient != 0.0:
                components[component] = coefficient
        return components
    if text in {"H", "K", "L"}:
        return {text: 1.0}
    return {}


def _component_coefficient(part: str, component: str) -> float:
    cleaned = part.replace(" ", "")
    if cleaned in {"0", "0.0", ""}:
        return 0.0
    if cleaned in {"H", "K", "L"}:
        return 1.0
    if cleaned in {"-H", "-K", "-L"}:
        return -1.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)
