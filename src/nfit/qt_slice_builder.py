"""Construction helpers for :mod:`nfit.qt_slice_viewer`.

The viewer retains ownership of widget state and callbacks. This module only
assembles that object graph into focused control groups so construction is easy
to inspect without mixing it with plotting and interaction behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .colormaps import (
    IMAGE_COLORMAP_GROUPS,
    WATERFALL_COLORMAP_GROUPS,
    populate_qt_colormap_combo,
)
from .qt_slice_controls import (
    _compact_combobox,
    _DualRangeSlider,
    _expanding_combobox,
    _make_data_viewer_window_class,
    _make_float_spinbox,
)
from .slice_viewer_state import _option_name


def build_slice_viewer(
    viewer: Any,
    *,
    marker_options: Mapping[str, str],
    line_style_options: Mapping[str, str],
    color_options: Mapping[str, str],
) -> None:
    """Build every widget and connection owned by a slice viewer."""

    _build_window_shell(viewer)
    plot_panel = _build_plot_panel(viewer)
    controls, controls_layout = _build_controls_panel(viewer)
    _build_dataset_controls(viewer, controls_layout)
    _build_axis_controls(viewer, controls_layout)
    _build_tiled_controls(viewer, controls_layout)
    _build_hidden_axis_controls(viewer, controls_layout)
    _build_color_controls(viewer, controls_layout)
    _build_smoothing_controls(viewer, controls_layout)
    _build_histogram_tool_controls(viewer, controls_layout)
    _build_line_controls(
        viewer,
        controls_layout,
        marker_options=marker_options,
        line_style_options=line_style_options,
        color_options=color_options,
    )
    _build_waterfall_controls(
        viewer,
        controls_layout,
        line_style_options=line_style_options,
        color_options=color_options,
    )
    _build_figure_controls(viewer, controls_layout)
    _finish_build(viewer, plot_panel, controls)


def _build_window_shell(viewer: Any) -> None:
    viewer.window = _make_data_viewer_window_class()(viewer)
    viewer.window.setWindowTitle("nfit Data Viewer")
    viewer.window.resize(1400, 900)

    central = QtWidgets.QWidget()
    viewer.window.setCentralWidget(central)
    main_layout = QtWidgets.QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    mode_bar = QtWidgets.QWidget()
    mode_layout = QtWidgets.QHBoxLayout(mode_bar)
    mode_layout.setContentsMargins(8, 6, 8, 6)
    mode_layout.addWidget(QtWidgets.QLabel("Visualization"))
    viewer.view_mode_combo = QtWidgets.QComboBox()
    viewer.view_mode_combo.setObjectName("data_viewer_mode_combo")
    viewer.view_mode_combo.addItems(["Slice viewer", "Waterfall", "Tiled slices", "Volumetric"])
    viewer.view_mode_combo.setToolTip(
        "Switch between standard slices, offset waterfall traces, tiled 2D slices, "
        "and volumetric rendering. Tiled-slice and volumetric modes require at least "
        "three dimensions with more than one bin."
    )
    viewer.view_mode_combo.currentIndexChanged.connect(viewer._set_view_mode)
    mode_layout.addWidget(viewer.view_mode_combo)
    viewer.open_new_viewer_button = QtWidgets.QPushButton("Open new viewer")
    viewer.open_new_viewer_button.setObjectName("data_viewer_open_new_button")
    viewer.open_new_viewer_button.setToolTip(
        "Open an independent data viewer initialized with the current dataset, "
        "visualization mode, axes, ranges, and styling."
    )
    viewer.open_new_viewer_button.clicked.connect(viewer.open_new_viewer)
    mode_layout.addWidget(viewer.open_new_viewer_button)
    viewer.save_plot_button = QtWidgets.QPushButton("Store plot")
    viewer.save_plot_button.setObjectName("data_viewer_store_plot_button")
    viewer.save_plot_button.setToolTip(
        "Store this view as an editable recipe in the workspace Plots section."
    )
    viewer.save_plot_button.setEnabled(False)
    viewer.save_plot_button.clicked.connect(viewer.store_plot)
    mode_layout.addWidget(viewer.save_plot_button)
    viewer.save_new_plot_button = QtWidgets.QPushButton("Save new plot")
    viewer.save_new_plot_button.setObjectName("data_viewer_save_new_plot_button")
    viewer.save_new_plot_button.setToolTip(
        "Create a separate stored plot without changing the original, then continue editing the new plot."
    )
    viewer.save_new_plot_button.setEnabled(False)
    viewer.save_new_plot_button.hide()
    viewer.save_new_plot_button.clicked.connect(viewer.save_new_plot)
    mode_layout.addWidget(viewer.save_new_plot_button)
    viewer.store_plot_status_label = QtWidgets.QLabel("")
    viewer.store_plot_status_label.setObjectName("data_viewer_store_plot_status")
    viewer.store_plot_status_label.setToolTip(
        "Name of the plot recipe most recently stored from this viewer."
    )
    mode_layout.addWidget(viewer.store_plot_status_label)
    mode_layout.addStretch(1)
    main_layout.addWidget(mode_bar)
    viewer.content_stack = QtWidgets.QStackedWidget()
    main_layout.addWidget(viewer.content_stack, 1)


def _build_plot_panel(viewer: Any) -> Any:
    plot_panel = QtWidgets.QWidget()
    plot_layout = QtWidgets.QVBoxLayout(plot_panel)
    plot_layout.setContentsMargins(8, 8, 8, 8)
    viewer.figure = Figure(figsize=(10, 8), constrained_layout=True)
    viewer.grid = viewer.figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, viewer._panel_ratio(viewer.ycut_percent), 0.045],
        height_ratios=[1.0, viewer._panel_ratio(viewer.xcut_percent)],
    )
    viewer.ax_image = viewer.figure.add_subplot(viewer.grid[0, 0])
    viewer.ax_ycut = viewer.figure.add_subplot(viewer.grid[0, 1], sharey=viewer.ax_image)
    viewer.ax_colorbar = viewer.figure.add_subplot(viewer.grid[0, 2])
    viewer.ax_xcut = viewer.figure.add_subplot(viewer.grid[1, 0], sharex=viewer.ax_image)
    viewer._suppress_matplotlib_coordinate_status()
    viewer.canvas = FigureCanvasQTAgg(viewer.figure)
    viewer.canvas.setToolTip(
        "Interactive plot canvas. Move the cursor for coordinate readouts; use the toolbar or box tool to inspect slices."
    )
    viewer.toolbar = NavigationToolbar2QT(viewer.canvas, viewer.window)
    viewer.toolbar.setToolTip(
        "Matplotlib navigation toolbar for pan, zoom, home, configure, and save actions."
    )
    cursor_bar = QtWidgets.QWidget()
    cursor_layout = QtWidgets.QHBoxLayout(cursor_bar)
    cursor_layout.setContentsMargins(4, 0, 4, 0)
    cursor_layout.setSpacing(12)
    viewer.cursor_xy_label = QtWidgets.QLabel("(x, y) = (-, -)")
    viewer.cursor_hkle_label = QtWidgets.QLabel("(H, K, L, E) = (-, -, -, -)")
    viewer.cursor_q_label = QtWidgets.QLabel("|Q| = ? Å⁻¹")
    viewer.cursor_intensity_label = QtWidgets.QLabel("Signal = -")
    for label, width in (
        (viewer.cursor_xy_label, 230),
        (viewer.cursor_hkle_label, 350),
        (viewer.cursor_q_label, 150),
        (viewer.cursor_intensity_label, 220),
    ):
        label.setMinimumWidth(width)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    cursor_layout.addWidget(viewer.cursor_xy_label, 0)
    cursor_layout.addWidget(viewer.cursor_hkle_label, 0)
    cursor_layout.addWidget(viewer.cursor_q_label, 0)
    cursor_layout.addWidget(viewer.cursor_intensity_label, 1)
    plot_layout.addWidget(viewer.toolbar)
    plot_layout.addWidget(cursor_bar)
    plot_layout.addWidget(viewer.canvas, 1)
    viewer._sync_cursor_visibility()

    return plot_panel


def _build_controls_panel(viewer: Any) -> tuple[Any, Any]:
    controls = QtWidgets.QScrollArea()
    viewer.controls_scroll = controls
    controls.setWidgetResizable(True)
    controls.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    controls.setMinimumWidth(430)
    controls.setMaximumWidth(500)
    controls_widget = QtWidgets.QWidget()
    controls_widget.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Ignored,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    controls.setWidget(controls_widget)
    controls_layout = QtWidgets.QVBoxLayout(controls_widget)
    controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

    return controls, controls_layout


def _build_dataset_controls(viewer: Any, controls_layout: Any) -> None:
    dataset_group = QtWidgets.QGroupBox("Dataset")
    dataset_layout = QtWidgets.QGridLayout(dataset_group)
    dataset_layout.setHorizontalSpacing(6)
    dataset_layout.setVerticalSpacing(6)
    viewer.dataset_combo = QtWidgets.QComboBox()
    viewer.dataset_combo.addItems(viewer._source_dataset_options())
    viewer.dataset_combo.setToolTip("Choose which loaded dataset is displayed in the viewer.")
    _expanding_combobox(viewer.dataset_combo)
    viewer.dataset_combo.currentIndexChanged.connect(viewer._set_dataset_selection)
    dataset_layout.addWidget(QtWidgets.QLabel("Dataset"), 0, 0)
    dataset_layout.addWidget(viewer.dataset_combo, 0, 1)
    viewer.binning_combo = QtWidgets.QComboBox()
    viewer.binning_combo.setObjectName("viewer_binning_combo")
    viewer.binning_combo.setToolTip(
        "Choose a named binning of the selected dataset. Only the binning marked for fitting participates in optimization."
    )
    viewer.binning_combo.currentIndexChanged.connect(viewer._set_binning_selection)
    dataset_layout.addWidget(QtWidgets.QLabel("Binning"), 1, 0)
    dataset_layout.addWidget(viewer.binning_combo, 1, 1)
    viewer._sync_dataset_binning_combos()
    viewer.channel_combo = QtWidgets.QComboBox()
    viewer.channel_combo.addItems(viewer.model.CHANNELS)
    viewer.channel_combo.setToolTip(
        "Choose the data channel to display, such as signal, combined_mask, file_mask, nfit_mask, fit, or residual."
    )
    _compact_combobox(viewer.channel_combo)
    viewer.channel_combo.setCurrentText(viewer.model.channel)
    viewer.channel_combo.currentTextChanged.connect(viewer._set_channel)
    viewer._sync_channel_combo()
    viewer.apply_masks_check = QtWidgets.QCheckBox("Apply Masks")
    viewer.apply_masks_check.setChecked(viewer.model.masked)
    viewer.apply_masks_check.setToolTip(
        "Show data after applying file masks and nfit masks. Uncheck to inspect masked-out data."
    )
    viewer.apply_masks_check.toggled.connect(viewer._set_apply_masks)
    dataset_layout.addWidget(QtWidgets.QLabel("Channel"), 2, 0)
    dataset_layout.addWidget(viewer.channel_combo, 2, 1)
    dataset_layout.addWidget(viewer.apply_masks_check, 2, 2)
    viewer.show_fit_check = QtWidgets.QCheckBox("Show model")
    viewer.show_fit_check.setToolTip(
        "Show the current model beside the data (2D) or as a line under the data (1D). "
        "Enabled once current model channels or stored fit channels are available for this dataset."
    )
    viewer.show_fit_check.toggled.connect(viewer._set_show_fit)
    viewer.unmask_model_check = QtWidgets.QCheckBox("Unmask model")
    viewer.unmask_model_check.setObjectName("viewer_unmask_model_check")
    viewer.unmask_model_check.setToolTip(
        "Evaluate and draw the model outside data masks across the full plotted region. "
        "Data remain masked; residuals extend only where underlying data and uncertainties are finite."
    )
    viewer.unmask_model_check.toggled.connect(viewer._set_unmask_model)
    viewer.show_residual_check = QtWidgets.QCheckBox("Show residual")
    viewer.show_residual_check.setToolTip(
        "Also show the normalized residual: a third panel (2D) or axes below the data (1D)."
    )
    viewer.show_residual_check.toggled.connect(viewer._set_show_residual)
    dataset_layout.addWidget(viewer.show_fit_check, 3, 1)
    dataset_layout.addWidget(viewer.show_residual_check, 3, 2)
    dataset_layout.addWidget(viewer.unmask_model_check, 4, 1)
    coverage_label = QtWidgets.QLabel("Coverage")
    viewer.coverage_threshold_label = coverage_label
    viewer.coverage_threshold_spin = _make_float_spinbox(0.0, 1.0)
    viewer.coverage_threshold_spin.setObjectName("viewer_coverage_threshold")
    viewer.coverage_threshold_spin.setDecimals(3)
    viewer.coverage_threshold_spin.setSingleStep(0.05)
    viewer.coverage_threshold_spin.setMaximumWidth(82)
    viewer.coverage_threshold_spin.setValue(viewer.coverage_threshold)
    coverage_tooltip = (
        "Mask slice pixels and histogram-tool reductions whose measured support "
        "is below this fraction of the requested integration volume."
    )
    coverage_label.setToolTip(coverage_tooltip)
    viewer.coverage_threshold_spin.setToolTip(coverage_tooltip)
    viewer.coverage_threshold_spin.valueChanged.connect(viewer._set_coverage_threshold)
    dataset_layout.addWidget(coverage_label, 5, 0)
    dataset_layout.addWidget(viewer.coverage_threshold_spin, 5, 1)
    viewer.residual_split_label = QtWidgets.QLabel()
    viewer.residual_split_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.residual_split_slider.setMinimumWidth(80)
    viewer.residual_split_slider.setRange(10, 70)
    viewer.residual_split_slider.setSingleStep(1)
    viewer.residual_split_slider.setPageStep(5)
    viewer.residual_split_slider.setValue(viewer.residual_percent)
    viewer.residual_split_slider.setToolTip(
        "Vertical position of the separator between the data and residual axes."
    )
    viewer.residual_split_slider.valueChanged.connect(viewer._set_residual_percent)
    dataset_layout.addWidget(viewer.residual_split_label, 6, 0)
    dataset_layout.addWidget(viewer.residual_split_slider, 6, 1, 1, 2)
    dataset_layout.setColumnStretch(1, 1)
    controls_layout.addWidget(dataset_group)


def _build_axis_controls(viewer: Any, controls_layout: Any) -> None:
    axes_group = QtWidgets.QGroupBox("Displayed axes")
    viewer.axes_group = axes_group
    axes_layout = QtWidgets.QGridLayout(axes_group)
    axes_layout.setHorizontalSpacing(6)
    axes_layout.setVerticalSpacing(5)
    viewer.x_combo = QtWidgets.QComboBox()
    viewer.y_combo = QtWidgets.QComboBox()
    viewer._sync_axis_combos(rebuild=True)
    viewer.x_combo.setToolTip("Choose the dataset axis shown horizontally.")
    viewer.y_combo.setToolTip(
        "Choose the dataset axis shown vertically. For 1D data this control is hidden."
    )
    _compact_combobox(viewer.x_combo)
    _compact_combobox(viewer.y_combo)
    viewer.x_combo.currentIndexChanged.connect(lambda index: viewer._set_display_dim("x", index))
    viewer.y_combo.currentIndexChanged.connect(lambda index: viewer._set_display_dim("y", index))
    axis_selector_widget = QtWidgets.QWidget()
    viewer.axis_selector_widget = axis_selector_widget
    axis_selector_layout = QtWidgets.QHBoxLayout(axis_selector_widget)
    axis_selector_layout.setContentsMargins(0, 0, 0, 0)
    axis_selector_layout.setSpacing(6)
    axis_selector_layout.addWidget(QtWidgets.QLabel("x"))
    axis_selector_layout.addWidget(viewer.x_combo)
    viewer._axis_y_label = QtWidgets.QLabel("y")
    axis_selector_layout.addWidget(viewer._axis_y_label)
    axis_selector_layout.addWidget(viewer.y_combo)
    axis_selector_layout.addStretch(1)
    axes_layout.addWidget(axis_selector_widget, 0, 0, 1, 4)
    viewer.x_min_spin = _make_float_spinbox()
    viewer.x_max_spin = _make_float_spinbox()
    viewer.y_min_spin = _make_float_spinbox()
    viewer.y_max_spin = _make_float_spinbox()
    viewer.x_reset_button = QtWidgets.QPushButton("Reset")
    viewer.y_reset_button = QtWidgets.QPushButton("Reset")
    viewer.x_min_spin.setToolTip("Lower displayed limit for the horizontal axis.")
    viewer.x_max_spin.setToolTip("Upper displayed limit for the horizontal axis.")
    viewer.y_min_spin.setToolTip("Lower displayed limit for the vertical axis.")
    viewer.y_max_spin.setToolTip("Upper displayed limit for the vertical axis.")
    viewer.x_reset_button.setToolTip(
        "Reset the horizontal axis limits to the full displayed data range."
    )
    viewer.y_reset_button.setToolTip(
        "Reset the vertical axis limits to the full displayed data range."
    )
    viewer.x_reset_button.setMaximumWidth(64)
    viewer.y_reset_button.setMaximumWidth(64)
    viewer.x_min_spin.valueChanged.connect(lambda _value: viewer._set_view_limits("x"))
    viewer.x_max_spin.valueChanged.connect(lambda _value: viewer._set_view_limits("x"))
    viewer.y_min_spin.valueChanged.connect(lambda _value: viewer._set_view_limits("y"))
    viewer.y_max_spin.valueChanged.connect(lambda _value: viewer._set_view_limits("y"))
    viewer.x_reset_button.clicked.connect(lambda: viewer._reset_view_limits("x"))
    viewer.y_reset_button.clicked.connect(lambda: viewer._reset_view_limits("y"))
    axes_layout.addWidget(QtWidgets.QLabel("min"), 1, 1)
    axes_layout.addWidget(QtWidgets.QLabel("max"), 1, 2)
    axes_layout.addWidget(QtWidgets.QLabel("x limits"), 2, 0)
    axes_layout.addWidget(viewer.x_min_spin, 2, 1)
    axes_layout.addWidget(viewer.x_max_spin, 2, 2)
    axes_layout.addWidget(viewer.x_reset_button, 2, 3)
    axes_layout.addWidget(QtWidgets.QLabel("y limits"), 3, 0)
    axes_layout.addWidget(viewer.y_min_spin, 3, 1)
    axes_layout.addWidget(viewer.y_max_spin, 3, 2)
    axes_layout.addWidget(viewer.y_reset_button, 3, 3)
    axes_layout.setColumnStretch(1, 1)
    axes_layout.setColumnStretch(2, 1)
    controls_layout.addWidget(axes_group)


def _build_tiled_controls(viewer: Any, controls_layout: Any) -> None:
    tiled_group = QtWidgets.QGroupBox("Tiled slices")
    viewer.tiled_group = tiled_group
    tiled_layout = QtWidgets.QGridLayout(tiled_group)
    tiled_layout.setHorizontalSpacing(6)
    tiled_layout.setVerticalSpacing(6)
    viewer.tile_dim_combo = QtWidgets.QComboBox()
    viewer.tile_dim_combo.setToolTip(
        "Choose the third dimension whose coarse slices are arranged as panels."
    )
    _expanding_combobox(viewer.tile_dim_combo)
    viewer.tile_dim_combo.currentIndexChanged.connect(viewer._set_tile_dimension)
    tiled_layout.addWidget(QtWidgets.QLabel("Third dimension"), 0, 0)
    tiled_layout.addWidget(viewer.tile_dim_combo, 0, 1, 1, 3)

    viewer.tile_range_low_spin = _make_float_spinbox()
    viewer.tile_range_high_spin = _make_float_spinbox()
    viewer.tile_range_low_spin.setToolTip(
        "Center of the first tiled panel. Each panel integrates a step-wide window around its center."
    )
    viewer.tile_range_high_spin.setToolTip(
        "Maximum tiled-panel center. Centers advance from Range low by Step size."
    )
    viewer.tile_range_low_spin.valueChanged.connect(viewer._set_tile_range)
    viewer.tile_range_high_spin.valueChanged.connect(viewer._set_tile_range)
    tiled_layout.addWidget(QtWidgets.QLabel("Range low"), 1, 0)
    tiled_layout.addWidget(viewer.tile_range_low_spin, 1, 1)
    tiled_layout.addWidget(QtWidgets.QLabel("Range high"), 1, 2)
    tiled_layout.addWidget(viewer.tile_range_high_spin, 1, 3)

    viewer.tile_step_spin = _make_float_spinbox(1.0e-9, 1.0e12)
    viewer.tile_step_spin.setDecimals(8)
    viewer.tile_step_spin.setToolTip(
        "Width of each coarse bin along the third dimension. Each bin becomes one 2D panel."
    )
    viewer.tile_step_spin.valueChanged.connect(viewer._set_tile_step)
    viewer.tile_step_auto_check = QtWidgets.QCheckBox("Auto")
    viewer.tile_step_auto_check.setChecked(viewer.tile_step_auto)
    viewer.tile_step_auto_check.setToolTip(
        "For metadata dimensions, show one panel per coordinate with its exact value. For other dimensions, choose a width giving up to nine panels."
    )
    viewer.tile_step_auto_check.toggled.connect(viewer._set_tile_step_auto)
    tiled_layout.addWidget(QtWidgets.QLabel("Step size"), 2, 0)
    tiled_layout.addWidget(viewer.tile_step_spin, 2, 1)
    tiled_layout.addWidget(viewer.tile_step_auto_check, 2, 2, 1, 2)
    viewer.tile_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.tile_step_slider.setRange(0, 1000)
    viewer.tile_step_slider.setToolTip(
        "Adjust the third-axis step from one native bin to the selected range span."
    )
    viewer.tile_step_slider.valueChanged.connect(viewer._set_tile_step_from_slider)
    tiled_layout.addWidget(viewer.tile_step_slider, 3, 0, 1, 4)
    tiled_layout.setColumnStretch(1, 1)
    tiled_layout.setColumnStretch(3, 1)
    controls_layout.addWidget(tiled_group)
    viewer._sync_tile_controls(reset_range=True)


def _build_hidden_axis_controls(viewer: Any, controls_layout: Any) -> None:
    viewer.hidden_group = QtWidgets.QGroupBox("Integrated Axes")
    viewer.hidden_layout = QtWidgets.QVBoxLayout(viewer.hidden_group)
    controls_layout.addWidget(viewer.hidden_group)
    viewer._rebuild_hidden_axis_controls()


def _build_color_controls(viewer: Any, controls_layout: Any) -> None:
    color_group = QtWidgets.QGroupBox("Color")
    viewer.color_group = color_group
    color_layout = QtWidgets.QGridLayout(color_group)
    color_layout.setHorizontalSpacing(6)
    color_layout.setVerticalSpacing(6)
    viewer.cmap_combo = QtWidgets.QComboBox()
    populate_qt_colormap_combo(viewer.cmap_combo, IMAGE_COLORMAP_GROUPS)
    viewer.cmap_combo.setToolTip("Choose the colormap used for 2D image data.")
    _compact_combobox(viewer.cmap_combo)
    viewer.cmap_combo.setCurrentText(viewer.model.cmap)
    viewer.cmap_combo.currentTextChanged.connect(viewer._set_cmap)
    viewer.cmap_reverse_button = QtWidgets.QPushButton("Reverse")
    viewer.cmap_reverse_button.setToolTip("Reverse the selected colormap.")
    viewer.cmap_reverse_button.clicked.connect(viewer._toggle_cmap_reverse)
    viewer.scale_combo = QtWidgets.QComboBox()
    viewer.scale_combo.addItems(viewer.model.COLOR_SCALES)
    viewer.scale_combo.setToolTip(
        "Choose linear, logarithmic, symmetric logarithmic, or power color scaling."
    )
    _compact_combobox(viewer.scale_combo)
    viewer.scale_combo.setCurrentText(viewer.model.color_scale)
    viewer.scale_combo.currentTextChanged.connect(viewer._set_color_scale)
    viewer.limits_combo = QtWidgets.QComboBox()
    viewer.limits_combo.addItems(viewer.model.AUTO_LIMITS)
    viewer.limits_combo.setToolTip(
        "Choose how automatic color limits are estimated from the displayed data."
    )
    _compact_combobox(viewer.limits_combo)
    viewer.limits_combo.setCurrentText(viewer.model.auto_limits)
    viewer.limits_combo.currentTextChanged.connect(viewer._set_auto_limits)
    viewer.autoscale_check = QtWidgets.QCheckBox("Autoscale")
    viewer.autoscale_check.setChecked(viewer.model.autoscale)
    viewer.autoscale_check.setToolTip(
        "Automatically recompute color limits when the displayed data or selection changes."
    )
    viewer.autoscale_check.toggled.connect(viewer._set_autoscale)
    viewer.tile_local_color_scales_check = QtWidgets.QCheckBox("Local scale per tiled plot")
    viewer.tile_local_color_scales_check.setObjectName("tile_local_color_scales")
    viewer.tile_local_color_scales_check.setChecked(viewer.tile_local_color_scales)
    viewer.tile_local_color_scales_check.setToolTip(
        "Give each tiled slice its own automatic color limits and adjacent colorbar. This option is available only while Autoscale is enabled."
    )
    viewer.tile_local_color_scales_check.toggled.connect(viewer._set_tile_local_color_scales)
    viewer.vmin_spin = _make_float_spinbox()
    viewer.vmax_spin = _make_float_spinbox()
    viewer.gamma_spin = _make_float_spinbox(1.0e-6, 20.0)
    viewer.gamma_spin.setValue(viewer.model.power_gamma)
    viewer.limit_n_spin = _make_float_spinbox(0.0, 50.0)
    viewer.limit_n_spin.setValue(viewer._current_limit_n())
    viewer.vmin_spin.setToolTip("Manual lower color limit when autoscale is off.")
    viewer.vmax_spin.setToolTip("Manual upper color limit when autoscale is off.")
    viewer.gamma_spin.setToolTip("Exponent used by power color scaling.")
    viewer.limit_n_spin.setToolTip(
        "Width parameter used by the selected automatic color-limit rule."
    )
    viewer.vmin_spin.valueChanged.connect(lambda value: viewer._set_manual_limit("vmin", value))
    viewer.vmax_spin.valueChanged.connect(lambda value: viewer._set_manual_limit("vmax", value))
    viewer.gamma_spin.valueChanged.connect(viewer._set_power_gamma)
    viewer.limit_n_spin.valueChanged.connect(viewer._set_limit_n)
    viewer.gamma_label = QtWidgets.QLabel("gamma")
    viewer.limit_n_label = QtWidgets.QLabel("N")
    color_layout.addWidget(QtWidgets.QLabel("Colormap"), 0, 0)
    color_layout.addWidget(viewer.cmap_combo, 0, 1)
    color_layout.addWidget(viewer.cmap_reverse_button, 0, 2, 1, 2)
    color_layout.addWidget(QtWidgets.QLabel("Scale"), 1, 0)
    color_layout.addWidget(viewer.scale_combo, 1, 1)
    color_layout.addWidget(viewer.gamma_label, 1, 2)
    color_layout.addWidget(viewer.gamma_spin, 1, 3)
    color_layout.addWidget(QtWidgets.QLabel("Auto limits"), 2, 0)
    color_layout.addWidget(viewer.limits_combo, 2, 1)
    color_layout.addWidget(viewer.limit_n_label, 2, 2)
    color_layout.addWidget(viewer.limit_n_spin, 2, 3)
    color_layout.addWidget(viewer.autoscale_check, 3, 1, 1, 3)
    color_layout.addWidget(
        viewer.tile_local_color_scales_check,
        4,
        1,
        1,
        3,
    )
    color_layout.addWidget(QtWidgets.QLabel("vmin"), 5, 0)
    color_layout.addWidget(viewer.vmin_spin, 5, 1)
    color_layout.addWidget(QtWidgets.QLabel("vmax"), 5, 2)
    color_layout.addWidget(viewer.vmax_spin, 5, 3)
    viewer.gamma_label.setVisible(viewer.model.color_scale == "power")
    viewer.gamma_spin.setVisible(viewer.model.color_scale == "power")
    viewer._sync_limit_n_visibility()
    controls_layout.addWidget(color_group)


def _build_smoothing_controls(viewer: Any, controls_layout: Any) -> None:
    smoothing_group = QtWidgets.QGroupBox("Plot smoothing")
    viewer.smoothing_group = smoothing_group
    smoothing_layout = QtWidgets.QGridLayout(smoothing_group)
    viewer.smoothing_x_spin = _make_float_spinbox(0.0, 100.0)
    viewer.smoothing_y_spin = _make_float_spinbox(0.0, 100.0)
    viewer.smoothing_x_spin.setObjectName("slice_smoothing_x_spin")
    viewer.smoothing_y_spin.setObjectName("slice_smoothing_y_spin")
    for axis_name, spin in (("X", viewer.smoothing_x_spin), ("Y", viewer.smoothing_y_spin)):
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setSuffix(" bins")
        spin.setToolTip(
            f"Gaussian blur sigma along displayed {axis_name}, in bin widths. "
            "This changes plotting only; fitting, dataset values, and numerical exports remain unchanged."
        )
    viewer.smoothing_x_spin.valueChanged.connect(
        lambda value: viewer._set_plot_smoothing("x", value)
    )
    viewer.smoothing_y_spin.valueChanged.connect(
        lambda value: viewer._set_plot_smoothing("y", value)
    )
    smoothing_layout.addWidget(QtWidgets.QLabel("X sigma"), 0, 0)
    smoothing_layout.addWidget(viewer.smoothing_x_spin, 0, 1)
    viewer.smoothing_y_label = QtWidgets.QLabel("Y sigma")
    smoothing_layout.addWidget(viewer.smoothing_y_label, 1, 0)
    smoothing_layout.addWidget(viewer.smoothing_y_spin, 1, 1)
    viewer.smoothing_fill_nans_check = QtWidgets.QCheckBox("Fill adjacent NaN bins")
    viewer.smoothing_fill_nans_check.setObjectName("slice_smoothing_fill_nans_check")
    viewer.smoothing_fill_nans_check.setChecked(viewer.smoothing_fill_nans)
    viewer.smoothing_fill_nans_check.setToolTip(
        "Allow plot-only Gaussian smoothing to extend into adjacent bins with no "
        "finite value. Uncheck to keep those pixels empty."
    )
    viewer.smoothing_fill_nans_check.toggled.connect(
        viewer._set_smoothing_fill_nans
    )
    smoothing_layout.addWidget(viewer.smoothing_fill_nans_check, 2, 0, 1, 2)
    smoothing_layout.setColumnStretch(1, 1)
    controls_layout.addWidget(smoothing_group)


def _build_histogram_tool_controls(viewer: Any, controls_layout: Any) -> None:
    tools_group = QtWidgets.QGroupBox("Histogram box cuts")
    viewer.tools_group = tools_group
    tools_layout = QtWidgets.QGridLayout(tools_group)
    viewer.roi_button = QtWidgets.QPushButton("Box tool")
    viewer.roi_button.setCheckable(True)
    viewer.roi_button.setToolTip(
        "Toggle the rectangle tool used to populate inverse-variance weighted x/y profile cuts "
        "with propagated error bars."
    )
    viewer.roi_button.toggled.connect(viewer._set_roi_enabled)
    viewer.show_box_check = QtWidgets.QCheckBox("Show box tool")
    viewer.show_box_check.setChecked(False)
    viewer.show_box_check.setToolTip("Show or hide the rectangle selection tool.")
    viewer.show_box_check.toggled.connect(viewer._set_box_tool_visible)
    viewer.hist_axes_check = QtWidgets.QCheckBox("Show histogram axes")
    viewer.hist_axes_check.setChecked(False)
    viewer.hist_axes_check.setToolTip(
        "Show or hide x/y profile cuts beside the image. Each point is an inverse-variance "
        "weighted mean over the selected box with its propagated standard error."
    )
    viewer.hist_axes_check.toggled.connect(viewer._set_histogram_axes_visible)
    viewer.xcut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.ycut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.xcut_percent_slider.setToolTip(
        "Height of the horizontal cut panel as a percentage of the figure."
    )
    viewer.ycut_percent_slider.setToolTip(
        "Width of the vertical cut panel as a percentage of the figure."
    )
    for slider, value in (
        (viewer.xcut_percent_slider, viewer.xcut_percent),
        (viewer.ycut_percent_slider, viewer.ycut_percent),
    ):
        slider.setRange(5, 45)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setValue(value)
    viewer.xcut_percent_label = QtWidgets.QLabel()
    viewer.ycut_percent_label = QtWidgets.QLabel()
    viewer.roi_x_center_spin = _make_float_spinbox()
    viewer.roi_x_width_spin = _make_float_spinbox(0.0, 1.0e12)
    viewer.roi_y_center_spin = _make_float_spinbox()
    viewer.roi_y_width_spin = _make_float_spinbox(0.0, 1.0e12)
    viewer.roi_x_center_spin.setToolTip(
        "Center of the rectangle selection along the horizontal axis."
    )
    viewer.roi_x_width_spin.setToolTip(
        "Width of the rectangle selection along the horizontal axis."
    )
    viewer.roi_y_center_spin.setToolTip(
        "Center of the rectangle selection along the vertical axis."
    )
    viewer.roi_y_width_spin.setToolTip("Width of the rectangle selection along the vertical axis.")
    viewer.xcut_percent_slider.valueChanged.connect(viewer._set_xcut_percent)
    viewer.ycut_percent_slider.valueChanged.connect(viewer._set_ycut_percent)
    viewer.roi_x_center_spin.valueChanged.connect(lambda _value: viewer._set_roi_from_controls())
    viewer.roi_x_width_spin.valueChanged.connect(lambda _value: viewer._set_roi_from_controls())
    viewer.roi_y_center_spin.valueChanged.connect(lambda _value: viewer._set_roi_from_controls())
    viewer.roi_y_width_spin.valueChanged.connect(lambda _value: viewer._set_roi_from_controls())
    tools_layout.addWidget(viewer.roi_button, 0, 0, 1, 4)
    tools_layout.addWidget(viewer.show_box_check, 1, 0, 1, 2)
    tools_layout.addWidget(viewer.hist_axes_check, 1, 2, 1, 2)
    tools_layout.addWidget(QtWidgets.QLabel("x center"), 2, 0)
    tools_layout.addWidget(viewer.roi_x_center_spin, 2, 1)
    tools_layout.addWidget(QtWidgets.QLabel("width"), 2, 2)
    tools_layout.addWidget(viewer.roi_x_width_spin, 2, 3)
    tools_layout.addWidget(QtWidgets.QLabel("y center"), 3, 0)
    tools_layout.addWidget(viewer.roi_y_center_spin, 3, 1)
    tools_layout.addWidget(QtWidgets.QLabel("width"), 3, 2)
    tools_layout.addWidget(viewer.roi_y_width_spin, 3, 3)
    tools_layout.addWidget(viewer.xcut_percent_label, 4, 0)
    tools_layout.addWidget(viewer.xcut_percent_slider, 4, 1, 1, 3)
    tools_layout.addWidget(viewer.ycut_percent_label, 5, 0)
    tools_layout.addWidget(viewer.ycut_percent_slider, 5, 1, 1, 3)
    tools_layout.setColumnStretch(1, 1)
    tools_layout.setColumnStretch(3, 1)
    viewer._sync_histogram_panel_controls()
    controls_layout.addWidget(tools_group)


def _build_line_controls(
    viewer: Any,
    controls_layout: Any,
    *,
    marker_options: Mapping[str, str],
    line_style_options: Mapping[str, str],
    color_options: Mapping[str, str],
) -> None:
    line_group = QtWidgets.QGroupBox("Line plot")
    viewer.line_group = line_group
    line_layout = QtWidgets.QGridLayout(line_group)
    line_layout.setHorizontalSpacing(6)
    line_layout.setVerticalSpacing(6)
    viewer.marker_combo = QtWidgets.QComboBox()
    viewer.marker_combo.addItems(marker_options.keys())
    viewer.marker_combo.setCurrentText("circle")
    viewer.marker_combo.setToolTip("Marker shape used for 1D line plots.")
    _compact_combobox(viewer.marker_combo)
    viewer.marker_combo.currentTextChanged.connect(viewer._set_marker)
    viewer.line_style_combo = QtWidgets.QComboBox()
    viewer.line_style_combo.addItems(line_style_options.keys())
    viewer.line_style_combo.setCurrentText("none")
    viewer.line_style_combo.setToolTip("Line style connecting points in 1D line plots.")
    _compact_combobox(viewer.line_style_combo)
    viewer.line_style_combo.currentTextChanged.connect(viewer._set_line_style)
    viewer.marker_size_spin = _make_float_spinbox(0.0, 50.0)
    viewer.marker_size_spin.setValue(viewer.marker_size)
    viewer.marker_size_spin.setToolTip("Size of markers in 1D line plots.")
    viewer.marker_size_spin.valueChanged.connect(viewer._set_marker_size)
    viewer.line_plot_width_spin = _make_float_spinbox(0.0, 20.0)
    viewer.line_plot_width_spin.setValue(viewer.line_plot_width)
    viewer.line_plot_width_spin.setToolTip("Width of the plotted line in 1D line plots.")
    viewer.line_plot_width_spin.valueChanged.connect(viewer._set_line_plot_width)
    viewer.marker_edge_width_spin = _make_float_spinbox(0.0, 20.0)
    viewer.marker_edge_width_spin.setValue(viewer.marker_edge_width)
    viewer.marker_edge_width_spin.setToolTip("Width of marker outlines in 1D line plots.")
    viewer.marker_edge_width_spin.valueChanged.connect(viewer._set_marker_edge_width)
    viewer.marker_face_color_combo = QtWidgets.QComboBox()
    viewer.marker_face_color_combo.addItems(
        ["none", "outline", *[name for name in color_options if name != "none"]]
    )
    viewer.marker_face_color_combo.setCurrentText("none")
    viewer.marker_face_color_combo.setToolTip(
        "Fill markers with no color, their outline color, or one common color."
    )
    _compact_combobox(viewer.marker_face_color_combo)
    viewer.marker_face_color_combo.currentTextChanged.connect(viewer._set_marker_face_color)
    viewer.line_color_combo = QtWidgets.QComboBox()
    viewer.line_color_combo.addItems(color_options.keys())
    viewer.line_color_combo.setCurrentText("blue")
    viewer.line_color_combo.setToolTip("Line and marker-edge color for 1D line plots.")
    _compact_combobox(viewer.line_color_combo)
    viewer.line_color_combo.currentTextChanged.connect(viewer._set_line_color)
    viewer.show_errorbars_check = QtWidgets.QCheckBox("Show errorbars")
    viewer.show_errorbars_check.setChecked(viewer.show_errorbars)
    viewer.show_errorbars_check.setToolTip(
        "Show uncertainty bars when the selected channel has errors."
    )
    viewer.show_errorbars_check.toggled.connect(viewer._set_show_errorbars)
    viewer.show_errorbar_caps_check = QtWidgets.QCheckBox("Endcaps")
    viewer.show_errorbar_caps_check.setChecked(viewer.show_errorbar_caps)
    viewer.show_errorbar_caps_check.setToolTip("Draw endcaps on error bars.")
    viewer.show_errorbar_caps_check.toggled.connect(viewer._set_show_errorbar_caps)
    viewer.errorbar_cap_size_spin = _make_float_spinbox(0.0, 30.0)
    viewer.errorbar_cap_size_spin.setValue(viewer.errorbar_cap_size)
    viewer.errorbar_cap_size_spin.setToolTip("Size of error-bar endcaps.")
    viewer.errorbar_cap_size_spin.valueChanged.connect(viewer._set_errorbar_cap_size)
    line_layout.addWidget(QtWidgets.QLabel("Marker"), 0, 0)
    line_layout.addWidget(viewer.marker_combo, 0, 1)
    line_layout.addWidget(QtWidgets.QLabel("Line"), 0, 2)
    line_layout.addWidget(viewer.line_style_combo, 0, 3)
    line_layout.addWidget(QtWidgets.QLabel("Marker size"), 1, 0)
    line_layout.addWidget(viewer.marker_size_spin, 1, 1)
    line_layout.addWidget(QtWidgets.QLabel("Linewidth"), 1, 2)
    line_layout.addWidget(viewer.line_plot_width_spin, 1, 3)
    line_layout.addWidget(QtWidgets.QLabel("Edge width"), 2, 0)
    line_layout.addWidget(viewer.marker_edge_width_spin, 2, 1)
    line_layout.addWidget(viewer.show_errorbars_check, 2, 2, 1, 2)
    line_layout.addWidget(QtWidgets.QLabel("Marker face"), 3, 0)
    line_layout.addWidget(viewer.marker_face_color_combo, 3, 1)
    line_layout.addWidget(QtWidgets.QLabel("Line/color"), 3, 2)
    line_layout.addWidget(viewer.line_color_combo, 3, 3)
    line_layout.addWidget(viewer.show_errorbar_caps_check, 4, 0, 1, 2)
    line_layout.addWidget(QtWidgets.QLabel("Cap size"), 4, 2)
    line_layout.addWidget(viewer.errorbar_cap_size_spin, 4, 3)
    viewer.fit_line_color_combo = QtWidgets.QComboBox()
    viewer.fit_line_color_combo.addItems([name for name in color_options if name != "none"])
    viewer.fit_line_color_combo.setCurrentText(_option_name(color_options, viewer.fit_line_color))
    viewer.fit_line_color_combo.setToolTip("Color used for model overlays in 1D plots.")
    _compact_combobox(viewer.fit_line_color_combo)
    viewer.fit_line_color_combo.currentTextChanged.connect(viewer._set_fit_line_color)
    viewer.fit_line_width_spin = _make_float_spinbox(0.1, 20.0)
    viewer.fit_line_width_spin.setValue(viewer.fit_line_width)
    viewer.fit_line_width_spin.setToolTip("Line width used for model overlays in 1D plots.")
    viewer.fit_line_width_spin.valueChanged.connect(viewer._set_fit_line_width)
    viewer.fit_line_color_label = QtWidgets.QLabel("Model color")
    viewer.fit_line_width_label = QtWidgets.QLabel("Model width")
    line_layout.addWidget(viewer.fit_line_color_label, 5, 0)
    line_layout.addWidget(viewer.fit_line_color_combo, 5, 1)
    line_layout.addWidget(viewer.fit_line_width_label, 5, 2)
    line_layout.addWidget(viewer.fit_line_width_spin, 5, 3)
    controls_layout.addWidget(line_group)


def _build_waterfall_controls(
    viewer: Any,
    controls_layout: Any,
    *,
    line_style_options: Mapping[str, str],
    color_options: Mapping[str, str],
) -> None:
    waterfall_group = QtWidgets.QGroupBox("Waterfall traces")
    viewer.waterfall_group = waterfall_group
    waterfall_layout = QtWidgets.QGridLayout(waterfall_group)
    waterfall_layout.setHorizontalSpacing(6)
    waterfall_layout.setVerticalSpacing(6)
    viewer.waterfall_source_label = QtWidgets.QLabel()
    viewer.waterfall_source_label.setWordWrap(True)
    viewer.waterfall_source_label.setToolTip(
        "For multidimensional data, traces are coarse bins along the selected waterfall axis. "
        "For compatible 1D data, every dataset contributes one trace."
    )
    waterfall_layout.addWidget(viewer.waterfall_source_label, 0, 0, 1, 4)

    viewer.waterfall_step_label = QtWidgets.QLabel("Bin width")
    viewer.waterfall_step_spin = _make_float_spinbox(1.0e-9, 1.0e12)
    viewer.waterfall_step_spin.setDecimals(8)
    viewer.waterfall_step_spin.setValue(viewer.waterfall_step)
    viewer.waterfall_step_spin.setToolTip(
        "Width of each coarse bin along the waterfall axis. Bins are reduced to "
        "inverse-variance weighted mean traces with propagated one-sigma errors."
    )
    viewer.waterfall_step_spin.valueChanged.connect(viewer._set_waterfall_step)
    viewer.waterfall_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.waterfall_step_slider.setRange(0, 1000)
    viewer.waterfall_step_slider.setToolTip(
        "Adjust the bin width from one native waterfall-axis bin to the full waterfall-axis span."
    )
    viewer.waterfall_step_slider.valueChanged.connect(viewer._set_waterfall_step_from_slider)
    viewer.waterfall_step_auto_check = QtWidgets.QCheckBox("Auto (~10)")
    viewer.waterfall_step_auto_check.setChecked(viewer.waterfall_step_auto)
    viewer.waterfall_step_auto_check.setToolTip(
        "Choose a waterfall-axis bin width that produces approximately ten traces."
    )
    viewer.waterfall_step_auto_check.toggled.connect(viewer._set_waterfall_step_auto)
    waterfall_layout.addWidget(viewer.waterfall_step_label, 1, 0)
    waterfall_layout.addWidget(viewer.waterfall_step_spin, 1, 1)
    waterfall_layout.addWidget(viewer.waterfall_step_auto_check, 1, 2, 1, 2)
    waterfall_layout.addWidget(viewer.waterfall_step_slider, 2, 0, 1, 4)

    waterfall_coverage_label = QtWidgets.QLabel("Coverage")
    viewer.waterfall_coverage_threshold_spin = _make_float_spinbox(0.0, 1.0)
    viewer.waterfall_coverage_threshold_spin.setObjectName("waterfall_coverage_threshold")
    viewer.waterfall_coverage_threshold_spin.setDecimals(3)
    viewer.waterfall_coverage_threshold_spin.setSingleStep(0.05)
    viewer.waterfall_coverage_threshold_spin.setMaximumWidth(82)
    viewer.waterfall_coverage_threshold_spin.setValue(viewer.waterfall_coverage_threshold)
    waterfall_coverage_tooltip = (
        "Mask coarse waterfall samples whose measured support is below this "
        "fraction of the requested trace-bin volume."
    )
    waterfall_coverage_label.setToolTip(waterfall_coverage_tooltip)
    viewer.waterfall_coverage_threshold_spin.setToolTip(waterfall_coverage_tooltip)
    viewer.waterfall_coverage_threshold_spin.valueChanged.connect(
        viewer._set_waterfall_coverage_threshold
    )
    waterfall_layout.addWidget(waterfall_coverage_label, 3, 0)
    waterfall_layout.addWidget(viewer.waterfall_coverage_threshold_spin, 3, 1)

    viewer.waterfall_offset_spin = _make_float_spinbox(0.0, 1.0e12)
    viewer.waterfall_offset_spin.setDecimals(8)
    viewer.waterfall_offset_spin.setValue(viewer.waterfall_offset)
    viewer.waterfall_offset_spin.setToolTip(
        "Vertical offset between adjacent traces, in the displayed channel units."
    )
    viewer.waterfall_offset_spin.valueChanged.connect(viewer._set_waterfall_offset)
    viewer.waterfall_offset_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    viewer.waterfall_offset_slider.setRange(0, 1000)
    viewer.waterfall_offset_slider.setToolTip(
        "Adjust the trace offset from zero to the largest absolute data "
        "value in the prepared waterfall traces."
    )
    viewer.waterfall_offset_slider.valueChanged.connect(viewer._set_waterfall_offset_from_slider)
    viewer.waterfall_offset_auto_check = QtWidgets.QCheckBox("Auto (half max)")
    viewer.waterfall_offset_auto_check.setChecked(viewer.waterfall_offset_auto)
    viewer.waterfall_offset_auto_check.setToolTip(
        "Set the trace offset to half the largest absolute intensity among the prepared traces."
    )
    viewer.waterfall_offset_auto_check.toggled.connect(viewer._set_waterfall_offset_auto)
    waterfall_layout.addWidget(QtWidgets.QLabel("Trace offset"), 4, 0)
    waterfall_layout.addWidget(viewer.waterfall_offset_spin, 4, 1)
    waterfall_layout.addWidget(viewer.waterfall_offset_auto_check, 4, 2, 1, 2)
    waterfall_layout.addWidget(viewer.waterfall_offset_slider, 5, 0, 1, 4)

    viewer.waterfall_cmap_combo = QtWidgets.QComboBox()
    populate_qt_colormap_combo(
        viewer.waterfall_cmap_combo,
        WATERFALL_COLORMAP_GROUPS,
    )
    viewer.waterfall_cmap_combo.setCurrentText(viewer.waterfall_cmap)
    viewer.waterfall_cmap_combo.setToolTip(
        "Color sequence sampled uniformly across the displayed waterfall traces."
    )
    _compact_combobox(viewer.waterfall_cmap_combo)
    viewer.waterfall_cmap_combo.currentTextChanged.connect(viewer._set_waterfall_cmap)
    viewer.waterfall_reverse_check = QtWidgets.QCheckBox("Reverse")
    viewer.waterfall_reverse_check.setChecked(viewer.waterfall_reverse_colors)
    viewer.waterfall_reverse_check.setToolTip(
        "Reverse the order of colors sampled from the sequence."
    )
    viewer.waterfall_reverse_check.toggled.connect(viewer._set_waterfall_reverse_colors)
    waterfall_layout.addWidget(QtWidgets.QLabel("Colors"), 6, 0)
    waterfall_layout.addWidget(viewer.waterfall_cmap_combo, 6, 1)
    waterfall_layout.addWidget(viewer.waterfall_reverse_check, 6, 2, 1, 2)

    viewer.waterfall_color_range_slider = _DualRangeSlider()
    viewer.waterfall_color_range_slider.set_values(0, 1000)
    viewer.waterfall_color_range_slider.setToolTip(
        "Drag the two handles to exclude colors near either end of a continuous colormap."
    )
    viewer.waterfall_color_range_slider.changed.connect(viewer._set_waterfall_color_range)
    viewer.waterfall_color_range_label = QtWidgets.QLabel("Color range")
    waterfall_layout.addWidget(viewer.waterfall_color_range_label, 7, 0)
    waterfall_layout.addWidget(
        viewer.waterfall_color_range_slider,
        7,
        1,
        1,
        3,
    )

    viewer.waterfall_zero_check = QtWidgets.QCheckBox("Zero references")
    viewer.waterfall_zero_check.setChecked(viewer.waterfall_show_zero_lines)
    viewer.waterfall_zero_check.setToolTip(
        "Draw a horizontal zero-intensity reference at the offset baseline of every trace."
    )
    viewer.waterfall_zero_check.toggled.connect(viewer._set_waterfall_zero_lines)
    viewer.waterfall_zero_color_combo = QtWidgets.QComboBox()
    viewer.waterfall_zero_color_combo.addItems([name for name in color_options if name != "none"])
    viewer.waterfall_zero_color_combo.setCurrentText(
        _option_name(color_options, viewer.waterfall_zero_color)
    )
    viewer.waterfall_zero_color_combo.setToolTip("Color of the per-trace zero reference lines.")
    _compact_combobox(viewer.waterfall_zero_color_combo)
    viewer.waterfall_zero_color_combo.currentTextChanged.connect(viewer._set_waterfall_zero_color)
    waterfall_layout.addWidget(viewer.waterfall_zero_check, 8, 0, 1, 2)
    waterfall_layout.addWidget(viewer.waterfall_zero_color_combo, 8, 2, 1, 2)

    viewer.waterfall_zero_style_combo = QtWidgets.QComboBox()
    viewer.waterfall_zero_style_combo.addItems(
        [name for name in line_style_options if name != "none"]
    )
    viewer.waterfall_zero_style_combo.setCurrentText(
        _option_name(line_style_options, viewer.waterfall_zero_style)
    )
    viewer.waterfall_zero_style_combo.setToolTip("Line style of the zero references.")
    _compact_combobox(viewer.waterfall_zero_style_combo)
    viewer.waterfall_zero_style_combo.currentTextChanged.connect(viewer._set_waterfall_zero_style)
    viewer.waterfall_zero_width_spin = _make_float_spinbox(0.1, 20.0)
    viewer.waterfall_zero_width_spin.setValue(viewer.waterfall_zero_width)
    viewer.waterfall_zero_width_spin.setToolTip("Line width of the zero references.")
    viewer.waterfall_zero_width_spin.valueChanged.connect(viewer._set_waterfall_zero_width)
    waterfall_layout.addWidget(QtWidgets.QLabel("Reference style"), 9, 0)
    waterfall_layout.addWidget(viewer.waterfall_zero_style_combo, 9, 1)
    waterfall_layout.addWidget(QtWidgets.QLabel("Width"), 9, 2)
    waterfall_layout.addWidget(viewer.waterfall_zero_width_spin, 9, 3)

    viewer.waterfall_model_color_combo = QtWidgets.QComboBox()
    viewer.waterfall_model_color_combo.addItems(
        ["match traces", *[name for name in color_options if name != "none"]]
    )
    viewer.waterfall_model_color_combo.setCurrentText("match traces")
    viewer.waterfall_model_color_combo.setToolTip(
        "Use each trace's color for its model line, or draw every model trace in one selected color."
    )
    _compact_combobox(viewer.waterfall_model_color_combo)
    viewer.waterfall_model_color_combo.currentTextChanged.connect(viewer._set_waterfall_model_color)
    viewer.waterfall_trace_labels_check = QtWidgets.QCheckBox("Trace labels")
    viewer.waterfall_trace_labels_check.setChecked(viewer.waterfall_show_trace_labels)
    viewer.waterfall_trace_labels_check.setToolTip(
        "Label each multidimensional trace by its waterfall-axis center, "
        "or each 1D trace by its dataset name."
    )
    viewer.waterfall_trace_labels_check.toggled.connect(viewer._set_waterfall_trace_labels)
    waterfall_layout.addWidget(QtWidgets.QLabel("Model colors"), 10, 0)
    waterfall_layout.addWidget(viewer.waterfall_model_color_combo, 10, 1)
    waterfall_layout.addWidget(viewer.waterfall_trace_labels_check, 10, 2, 1, 2)

    viewer.waterfall_trace_label_suffix_edit = QtWidgets.QLineEdit()
    viewer.waterfall_trace_label_suffix_edit.setText(viewer.waterfall_trace_label_suffix)
    viewer.waterfall_trace_label_suffix_edit.setPlaceholderText(
        "Optional replacement for generated axis units"
    )
    viewer.waterfall_trace_label_suffix_edit.setToolTip(
        "For coordinate-derived traces, replace the default axis units "
        "(for example meV or r.l.u.) with this text. For grouped 1D "
        "datasets, append it verbatim to the dataset label."
    )
    viewer.waterfall_trace_label_suffix_edit.textChanged.connect(
        viewer._set_waterfall_trace_label_suffix
    )
    waterfall_layout.addWidget(QtWidgets.QLabel("Label suffix"), 11, 0)
    waterfall_layout.addWidget(
        viewer.waterfall_trace_label_suffix_edit,
        11,
        1,
        1,
        3,
    )

    viewer.waterfall_trace_label_font_size_spin = _make_float_spinbox(4.0, 48.0)
    viewer.waterfall_trace_label_font_size_spin.setDecimals(1)
    viewer.waterfall_trace_label_font_size_spin.setValue(viewer.waterfall_trace_label_font_size)
    viewer.waterfall_trace_label_font_size_spin.setToolTip(
        "Font size used only for waterfall trace labels."
    )
    viewer.waterfall_trace_label_font_size_spin.valueChanged.connect(
        viewer._set_waterfall_trace_label_font_size
    )
    viewer.waterfall_trace_label_color_combo = QtWidgets.QComboBox()
    viewer.waterfall_trace_label_color_combo.addItems(
        ["match traces", *[name for name in color_options if name != "none"]]
    )
    viewer.waterfall_trace_label_color_combo.setCurrentText("match traces")
    viewer.waterfall_trace_label_color_combo.setToolTip(
        "Match each label to its trace, or use one common font color for all labels."
    )
    _compact_combobox(viewer.waterfall_trace_label_color_combo)
    viewer.waterfall_trace_label_color_combo.currentTextChanged.connect(
        viewer._set_waterfall_trace_label_color
    )
    waterfall_layout.addWidget(QtWidgets.QLabel("Label size"), 12, 0)
    waterfall_layout.addWidget(
        viewer.waterfall_trace_label_font_size_spin,
        12,
        1,
    )
    waterfall_layout.addWidget(QtWidgets.QLabel("Label color"), 12, 2)
    waterfall_layout.addWidget(
        viewer.waterfall_trace_label_color_combo,
        12,
        3,
    )

    waterfall_layout.setColumnStretch(1, 1)
    waterfall_layout.setColumnStretch(3, 1)
    controls_layout.addWidget(waterfall_group)


def _build_figure_controls(viewer: Any, controls_layout: Any) -> None:
    figure_group = QtWidgets.QGroupBox("Figure")
    figure_layout = QtWidgets.QGridLayout(figure_group)
    figure_layout.setHorizontalSpacing(6)
    figure_layout.setVerticalSpacing(6)
    viewer.font_size_spin = _make_float_spinbox(4.0, 48.0)
    viewer.font_size_spin.setDecimals(1)
    viewer.font_size_spin.setValue(viewer.font_size)
    viewer.font_size_spin.setToolTip(
        "Base font size used for axes labels, ticks, titles, and readouts."
    )
    viewer.font_size_spin.valueChanged.connect(viewer._set_font_size)
    viewer.line_width_spin = _make_float_spinbox(0.1, 10.0)
    viewer.line_width_spin.setDecimals(2)
    viewer.line_width_spin.setSingleStep(0.25)
    viewer.line_width_spin.setValue(viewer.axis_linewidth)
    viewer.line_width_spin.setToolTip("Width of figure axes and frame lines.")
    viewer.line_width_spin.valueChanged.connect(viewer._set_axis_linewidth)
    viewer.show_binning_title_check = QtWidgets.QCheckBox("Show other-axis binning above plot")
    viewer.show_binning_title_check.setObjectName("show_binning_title")
    viewer.show_binning_title_check.setChecked(viewer.show_binning_title)
    viewer.show_binning_title_check.setToolTip(
        "Display the selected or integrated ranges of every non-plotted axis above the figure, including coordinate labels and units."
    )
    viewer.show_binning_title_check.toggled.connect(viewer._set_show_binning_title)
    viewer.show_tile_labels_check = QtWidgets.QCheckBox("Show tiled-slice value labels")
    viewer.show_tile_labels_check.setObjectName("show_tile_labels")
    viewer.show_tile_labels_check.setChecked(viewer.show_tile_labels)
    viewer.show_tile_labels_check.setToolTip(
        "Show the third-axis value or interval in a translucent box at the lower-right of each tiled slice."
    )
    viewer.show_tile_labels_check.toggled.connect(viewer._set_show_tile_labels)
    viewer.tile_label_decimals_spin = QtWidgets.QSpinBox()
    viewer.tile_label_decimals_spin.setRange(0, 10)
    viewer.tile_label_decimals_spin.setPrefix("Decimals: ")
    viewer.tile_label_decimals_spin.setValue(viewer.tile_label_decimals)
    viewer.tile_label_decimals_spin.setToolTip(
        "Decimal places in tiled-slice value labels: 0 gives 5 K, 1 gives 5.2 K. Only label formatting changes."
    )
    viewer.tile_label_decimals_spin.valueChanged.connect(viewer._set_tile_label_decimals)
    viewer.tile_label_options = QtWidgets.QWidget()
    label_layout = QtWidgets.QGridLayout(viewer.tile_label_options)
    label_layout.setContentsMargins(0, 0, 0, 0)
    viewer.tile_label_prefix_edit = QtWidgets.QLineEdit(viewer.tile_label_prefix)
    viewer.tile_label_prefix_edit.setToolTip(
        "Text before the value. Use {axis} for the axis name, T = for a short label, or leave empty for just the value."
    )
    viewer.tile_label_unit_edit = QtWidgets.QLineEdit(viewer.tile_label_unit)
    viewer.tile_label_unit_edit.setToolTip(
        "Unit text after the value. {unit} uses the axis unit; leave empty to hide units. Editing this text does not convert values."
    )
    viewer.tile_label_si_prefix_combo = QtWidgets.QComboBox()
    viewer.tile_label_si_prefix_combo.addItems(["", "p", "n", "µ", "m", "k", "M", "G", "T"])
    viewer.tile_label_si_prefix_combo.setToolTip(
        "Rescale label values and prepend this SI prefix to the unit. For K, m displays millikelvin. Applied relative to the existing axis unit; does not change data or binning."
    )
    viewer.tile_label_prefix_edit.editingFinished.connect(viewer._set_tile_label_text)
    viewer.tile_label_unit_edit.editingFinished.connect(viewer._set_tile_label_text)
    viewer.tile_label_si_prefix_combo.currentTextChanged.connect(viewer._set_tile_label_text)
    label_layout.addWidget(QtWidgets.QLabel("Label text"), 0, 0)
    label_layout.addWidget(viewer.tile_label_prefix_edit, 0, 1, 1, 3)
    label_layout.addWidget(QtWidgets.QLabel("Unit"), 1, 0)
    label_layout.addWidget(viewer.tile_label_unit_edit, 1, 1)
    label_layout.addWidget(QtWidgets.QLabel("SI prefix"), 1, 2)
    label_layout.addWidget(viewer.tile_label_si_prefix_combo, 1, 3)
    viewer.copy_figure_button = QtWidgets.QPushButton("Copy figure")
    viewer.copy_script_button = QtWidgets.QPushButton("Copy script")
    viewer.save_script_button = QtWidgets.QPushButton("Save script")
    for button in (
        viewer.copy_figure_button,
        viewer.copy_script_button,
        viewer.save_script_button,
    ):
        button.setMinimumWidth(0)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
        )
    viewer.copy_figure_button.setToolTip("Copy the current figure image to the clipboard.")
    viewer.copy_script_button.setToolTip(
        "Copy a Python script that recreates the current viewer plot."
    )
    viewer.save_script_button.setToolTip(
        "Save a Python script that recreates the current viewer plot."
    )
    viewer.copy_figure_button.clicked.connect(viewer.copy_figure_to_clipboard)
    viewer.copy_script_button.clicked.connect(viewer.copy_script_to_clipboard)
    viewer.save_script_button.clicked.connect(viewer.save_script)
    figure_layout.addWidget(QtWidgets.QLabel("Font size"), 0, 0)
    figure_layout.addWidget(viewer.font_size_spin, 0, 1)
    figure_layout.addWidget(QtWidgets.QLabel("Linewidth"), 0, 2)
    figure_layout.addWidget(viewer.line_width_spin, 0, 3)
    figure_layout.addWidget(viewer.show_binning_title_check, 1, 0, 1, 4)
    figure_layout.addWidget(viewer.show_tile_labels_check, 2, 0, 1, 2)
    figure_layout.addWidget(viewer.tile_label_decimals_spin, 2, 2, 1, 2)
    figure_layout.addWidget(viewer.tile_label_options, 3, 0, 1, 4)
    figure_layout.addWidget(viewer.copy_figure_button, 4, 0, 1, 4)
    figure_layout.addWidget(viewer.copy_script_button, 5, 0, 1, 2)
    figure_layout.addWidget(viewer.save_script_button, 5, 2, 1, 2)
    controls_layout.addWidget(figure_group)
    controls_layout.addStretch(1)


def _finish_build(viewer: Any, plot_panel: Any, controls: Any) -> None:
    splitter = QtWidgets.QSplitter()
    splitter.addWidget(plot_panel)
    splitter.addWidget(controls)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setChildrenCollapsible(False)
    splitter.setSizes([970, 430])
    viewer.content_stack.addWidget(splitter)

    viewer._plot_layout_mode = ("standard", 1)
    viewer._create_rectangle_selector()
    viewer._connect_view_limit_callbacks()
    viewer.canvas.mpl_connect("motion_notify_event", viewer._on_motion)
    copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, viewer.window)
    copy_shortcut.activated.connect(viewer.copy_figure_to_clipboard)
    viewer.save_project_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Save,
        viewer.window,
    )
    viewer.save_project_shortcut.setEnabled(False)
    viewer.save_project_shortcut.activated.connect(
        lambda: (
            viewer._save_project_callback() if viewer._save_project_callback is not None else None
        )
    )
    viewer.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, viewer.window)
    viewer.close_shortcut.activated.connect(viewer.window.close)
    viewer._sync_fit_channel_controls()
    viewer._sync_view_mode_availability()
