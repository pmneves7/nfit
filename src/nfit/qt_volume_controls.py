from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .colormaps import VOLUME_COLORMAP_GROUPS, populate_qt_colormap_combo
from .qt_pyvista import configure_pyvista_interactor


def build_volume_panel_ui(
    panel: Any,
    *,
    render_modes: Sequence[str],
    transfer_curve_editor: Callable[..., Any],
) -> None:
    """Build and connect the volume panel's Qt controls.

    Rendering and dataset behavior remain on the panel.  Keeping construction in
    this focused module makes the available controls and their signal wiring
    readable without mixing them into the rendering implementation.
    """

    from matplotlib import colormaps
    from PySide6 import QtCore, QtWidgets
    from pyvistaqt import QtInteractor

    root = QtWidgets.QHBoxLayout(panel)
    root.setContentsMargins(0, 0, 0, 0)
    splitter = QtWidgets.QSplitter()
    render_frame = QtWidgets.QFrame()
    render_layout = QtWidgets.QVBoxLayout(render_frame)
    render_layout.setContentsMargins(8, 8, 8, 8)
    panel.plotter = QtInteractor(render_frame)
    configure_pyvista_interactor(panel.plotter)
    panel.render_status = QtWidgets.QLabel()
    panel.render_status.setObjectName("volume_render_status")
    panel.render_status.setWordWrap(True)
    panel.render_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    panel.render_status.setStyleSheet(
        "QLabel { color: #7a2e00; background: #fff4e5; "
        "border: 1px solid #d9a441; padding: 6px; }"
    )
    panel.render_status.hide()
    render_layout.addWidget(panel.render_status)
    render_layout.addWidget(panel.plotter.interactor)
    panel.plotter.set_background("white")
    panel.plotter.add_axes(color="black")

    controls = QtWidgets.QScrollArea()
    controls.setObjectName("volume_controls_scroll")
    controls.setWidgetResizable(True)
    controls.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
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

    _build_dataset_controls(panel, layout, render_modes, QtWidgets)
    _build_axis_controls(panel, layout, QtWidgets)
    _build_scale_controls(panel, layout, QtWidgets)
    _build_smoothing_controls(panel, layout, QtWidgets)

    panel.hidden_group = QtWidgets.QGroupBox("Remaining axes")
    panel.hidden_layout = QtWidgets.QVBoxLayout(panel.hidden_group)
    panel.hidden_group.setToolTip(
        "Choose one position or integrate a range for dimensions not mapped "
        "to X, Y, or Z."
    )
    layout.addWidget(panel.hidden_group)

    _build_channel_controls(panel, layout, colormaps, QtWidgets)
    _build_range_controls(panel, layout, QtWidgets)
    _build_curve_controls(panel, layout, transfer_curve_editor, QtWidgets)
    _build_surface_controls(panel, layout, QtCore, QtWidgets)
    _build_export_controls(panel, layout, QtWidgets)
    _build_animation_controls(panel, layout, QtWidgets)

    layout.addStretch(1)
    splitter.addWidget(render_frame)
    splitter.addWidget(controls)
    splitter.setStretchFactor(0, 1)
    splitter.setSizes([980, 420])
    root.addWidget(splitter)


def _build_dataset_controls(panel, layout, render_modes, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("3D dataset")
    grid = QtWidgets.QGridLayout(group)
    panel.dataset_combo = QtWidgets.QComboBox()
    panel.dataset_combo.setObjectName("volume_dataset_combo")
    panel.dataset_combo.addItems(panel.dataset_names)
    panel.dataset_combo.setCurrentIndex(panel.dataset_index)
    panel.dataset_combo.setToolTip(
        "Choose a gridded dataset with at least three dimensions for 3D rendering."
    )
    panel.dataset_combo.currentIndexChanged.connect(panel._set_dataset)
    panel.render_combo = QtWidgets.QComboBox()
    panel.render_combo.setObjectName("volume_render_mode_combo")
    panel.render_combo.addItems(render_modes)
    panel.render_combo.setToolTip(
        "Choose direct volumetric rendering or an extracted constant-value "
        "isosurface."
    )
    panel.render_combo.currentTextChanged.connect(panel._render)
    grid.addWidget(QtWidgets.QLabel("Dataset"), 0, 0)
    grid.addWidget(panel.dataset_combo, 0, 1)
    grid.addWidget(QtWidgets.QLabel("Mode"), 1, 0)
    grid.addWidget(panel.render_combo, 1, 1)
    layout.addWidget(group)


def _build_axis_controls(panel, layout, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("3D axes")
    grid = QtWidgets.QGridLayout(group)
    panel.axis_combos = []
    panel.axis_limit_spins = []
    for row, label in enumerate(("X", "Y", "Z")):
        combo = QtWidgets.QComboBox()
        combo.setObjectName(f"volume_{label.lower()}_axis_combo")
        combo.setToolTip(
            f"Choose the dataset axis mapped to the 3D {label.lower()} coordinate."
        )
        combo.currentIndexChanged.connect(panel._axes_changed)
        panel.axis_combos.append(combo)
        grid.addWidget(QtWidgets.QLabel(label), row, 0)
        grid.addWidget(combo, row, 1)
        low_spin = _float_spin(QtWidgets)
        high_spin = _float_spin(QtWidgets)
        low_spin.setObjectName(f"volume_{label.lower()}_min_spin")
        high_spin.setObjectName(f"volume_{label.lower()}_max_spin")
        low_spin.setToolTip(
            f"Minimum displayed {label} coordinate; complete bins are selected "
            "by center."
        )
        high_spin.setToolTip(
            f"Maximum displayed {label} coordinate; complete bins are selected "
            "by center."
        )
        low_spin.editingFinished.connect(panel._axis_limits_changed)
        high_spin.editingFinished.connect(panel._axis_limits_changed)
        panel.axis_limit_spins.append((low_spin, high_spin))
        grid.addWidget(QtWidgets.QLabel(f"{label} min"), row + 3, 0)
        grid.addWidget(low_spin, row + 3, 1)
        grid.addWidget(QtWidgets.QLabel("max"), row + 3, 2)
        grid.addWidget(high_spin, row + 3, 3)
    reset = QtWidgets.QPushButton("Reset axis limits")
    reset.setObjectName("volume_reset_axis_limits_button")
    reset.setToolTip(
        "Restore X, Y, and Z to the complete range of their selected dataset axes."
    )
    reset.clicked.connect(panel._reset_axis_limits)
    grid.addWidget(reset, 6, 0, 1, 4)
    layout.addWidget(group)


def _build_scale_controls(panel, layout, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Axis scale")
    grid = QtWidgets.QGridLayout(group)
    panel.scale_mode_combo = QtWidgets.QComboBox()
    panel.scale_mode_combo.setObjectName("volume_axis_scale_mode_combo")
    panel.scale_mode_combo.addItems(["Equal data units", "Custom"])
    panel.scale_mode_combo.setToolTip(
        "Use equal data-unit lengths on every axis, or apply independent visual "
        "scale factors without changing data values."
    )
    panel.scale_mode_combo.currentTextChanged.connect(panel._scale_changed)
    panel.axis_scale_spins = []
    grid.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
    grid.addWidget(panel.scale_mode_combo, 0, 1, 1, 2)
    for column, label in enumerate(("X", "Y", "Z")):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setObjectName(f"volume_{label.lower()}_scale_spin")
        spin.setRange(0.01, 100.0)
        spin.setDecimals(3)
        spin.setValue(1.0)
        spin.setSingleStep(0.1)
        _make_compact(spin, QtWidgets)
        spin.setToolTip(
            f"Visual length multiplier for {label}; values below one squish and "
            "values above one stretch the axis."
        )
        spin.valueChanged.connect(panel._scale_changed)
        panel.axis_scale_spins.append(spin)
        grid.addWidget(QtWidgets.QLabel(label), 1, column)
        grid.addWidget(spin, 2, column)
    layout.addWidget(group)


def _build_smoothing_controls(panel, layout, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Plot smoothing")
    grid = QtWidgets.QGridLayout(group)
    panel.smoothing_spins = []
    for column, label in enumerate(("X", "Y", "Z")):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setObjectName(f"volume_{label.lower()}_smoothing_spin")
        spin.setRange(0.0, 100.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setSuffix(" bins")
        _make_compact(spin, QtWidgets)
        spin.setToolTip(
            f"Gaussian blur sigma along displayed {label}, in bin widths. This "
            "affects rendering, still images, and movies only; fitting, data "
            "values, and model/data exports remain unchanged."
        )
        spin.valueChanged.connect(panel._smoothing_changed)
        panel.smoothing_spins.append(spin)
        grid.addWidget(QtWidgets.QLabel(label), 0, column)
        grid.addWidget(spin, 1, column)
    panel.smoothing_fill_nans_check = QtWidgets.QCheckBox("Fill adjacent NaN bins")
    panel.smoothing_fill_nans_check.setObjectName("volume_smoothing_fill_nans_check")
    panel.smoothing_fill_nans_check.setChecked(True)
    panel.smoothing_fill_nans_check.setToolTip(
        "Allow plot-only Gaussian smoothing to extend into adjacent bins with no "
        "finite value. Uncheck to preserve those gaps in the rendered volume."
    )
    panel.smoothing_fill_nans_check.toggled.connect(panel._smoothing_changed)
    grid.addWidget(panel.smoothing_fill_nans_check, 2, 0, 1, 3)
    layout.addWidget(group)


def _build_channel_controls(panel, layout, colormaps, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Color and opacity")
    grid = QtWidgets.QGridLayout(group)
    panel.color_channel_combo = QtWidgets.QComboBox()
    panel.color_channel_combo.setObjectName("volume_color_channel_combo")
    panel.color_channel_combo.setToolTip("Dataset channel mapped through the colormap.")
    panel.opacity_channel_combo = QtWidgets.QComboBox()
    panel.opacity_channel_combo.setObjectName("volume_opacity_channel_combo")
    panel.opacity_channel_combo.setToolTip(
        "Dataset channel independently mapped to opacity when channels are unlinked."
    )
    panel.link_channels_check = QtWidgets.QCheckBox("Link opacity to color")
    panel.link_channels_check.setObjectName("volume_link_channels_check")
    panel.link_channels_check.setChecked(True)
    panel.link_channels_check.setToolTip(
        "Use the color channel for opacity. Uncheck to select a different opacity "
        "channel and range."
    )
    panel.cmap_combo = QtWidgets.QComboBox()
    panel.cmap_combo.setObjectName("volume_colormap_combo")
    available_groups = tuple(
        tuple(name for name in group_names if name in colormaps)
        for group_names in VOLUME_COLORMAP_GROUPS
    )
    populate_qt_colormap_combo(panel.cmap_combo, available_groups)
    panel.cmap_combo.setCurrentText("viridis")
    panel.cmap_combo.setToolTip(
        "Colormap used after applying the editable color mapping curve."
    )
    for widget in (panel.color_channel_combo, panel.opacity_channel_combo):
        widget.currentTextChanged.connect(panel._channel_changed)
    panel.cmap_combo.currentTextChanged.connect(panel._render)
    panel.link_channels_check.toggled.connect(panel._link_changed)
    grid.addWidget(QtWidgets.QLabel("Color channel"), 0, 0)
    grid.addWidget(panel.color_channel_combo, 0, 1)
    grid.addWidget(QtWidgets.QLabel("Opacity channel"), 1, 0)
    grid.addWidget(panel.opacity_channel_combo, 1, 1)
    grid.addWidget(panel.link_channels_check, 2, 0, 1, 2)
    grid.addWidget(QtWidgets.QLabel("Colormap"), 3, 0)
    grid.addWidget(panel.cmap_combo, 3, 1)
    layout.addWidget(group)


def _build_range_controls(panel, layout, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Channel ranges")
    grid = QtWidgets.QGridLayout(group)
    panel.color_min = _float_spin(QtWidgets)
    panel.color_max = _float_spin(QtWidgets)
    panel.opacity_min = _float_spin(QtWidgets)
    panel.opacity_max = _float_spin(QtWidgets)
    for widget in (
        panel.color_min,
        panel.color_max,
        panel.opacity_min,
        panel.opacity_max,
    ):
        widget.setToolTip(
            "Values outside this range are clipped before the transfer curve is applied."
        )
        widget.editingFinished.connect(panel._render)
    grid.addWidget(QtWidgets.QLabel("Color min"), 0, 0)
    grid.addWidget(panel.color_min, 0, 1)
    grid.addWidget(QtWidgets.QLabel("Color max"), 1, 0)
    grid.addWidget(panel.color_max, 1, 1)
    grid.addWidget(QtWidgets.QLabel("Opacity min"), 2, 0)
    grid.addWidget(panel.opacity_min, 2, 1)
    grid.addWidget(QtWidgets.QLabel("Opacity max"), 3, 0)
    grid.addWidget(panel.opacity_max, 3, 1)
    grid.setColumnStretch(1, 1)
    reset = QtWidgets.QPushButton("Autoscale ranges")
    reset.setToolTip(
        "Set both channel ranges from the finite 1st and 99th percentiles of "
        "the current 3D data."
    )
    reset.clicked.connect(panel._autoscale_ranges)
    grid.addWidget(reset, 4, 0, 1, 2)
    layout.addWidget(group)


def _build_curve_controls(panel, layout, editor_factory, QtWidgets) -> None:
    color_group = QtWidgets.QGroupBox("Color mapping curve")
    color_layout = QtWidgets.QVBoxLayout(color_group)
    panel.color_curve = editor_factory(
        tooltip=(
            "Map normalized color-channel values (horizontal) to positions in "
            "the selected colormap (vertical). Drag points; click to add; "
            "right-click an interior point to remove it."
        )
    )
    panel.color_curve.setObjectName("volume_color_curve")
    panel.color_curve.curveChanged.connect(panel._render)
    color_layout.addWidget(panel.color_curve)
    layout.addWidget(color_group)

    opacity_group = QtWidgets.QGroupBox("Opacity mapping curve")
    opacity_layout = QtWidgets.QVBoxLayout(opacity_group)
    panel.opacity_curve = editor_factory(
        points=[(0.0, 0.0), (0.35, 0.0), (1.0, 1.0)],
        tooltip=(
            "Map normalized opacity-channel values (horizontal) to "
            "transparency/opacity (vertical). Drag points; click to add; "
            "right-click an interior point to remove it."
        ),
    )
    panel.opacity_curve.setObjectName("volume_opacity_curve")
    panel.opacity_curve.curveChanged.connect(panel._render)
    opacity_layout.addWidget(panel.opacity_curve)
    layout.addWidget(opacity_group)


def _build_surface_controls(panel, layout, QtCore, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Isosurface")
    grid = QtWidgets.QGridLayout(group)
    panel.surface_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    panel.surface_slider.setObjectName("volume_surface_level_slider")
    panel.surface_slider.setRange(0, 1000)
    panel.surface_slider.setValue(700)
    panel.surface_slider.setToolTip(
        "Select the constant opacity-channel value used to extract the isosurface."
    )
    panel.surface_slider.valueChanged.connect(panel._render)
    panel.surface_label = QtWidgets.QLabel()
    grid.addWidget(panel.surface_label, 0, 0)
    grid.addWidget(panel.surface_slider, 0, 1)
    panel.surface_group = group
    layout.addWidget(group)


def _build_export_controls(panel, layout, QtWidgets) -> None:
    panel.apply_masks_check = QtWidgets.QCheckBox("Apply masks")
    panel.apply_masks_check.setChecked(True)
    panel.apply_masks_check.setToolTip(
        "Make masked and empty bins transparent before 3D rendering."
    )
    panel.apply_masks_check.toggled.connect(panel._channel_changed)
    layout.addWidget(panel.apply_masks_check)
    panel.export_button = QtWidgets.QPushButton("Export 3D model")
    panel.export_button.setObjectName("volume_export_button")
    panel.export_button.setToolTip(
        "Export the isosurface as a mesh/scene, or export volumetric data as a "
        "VTK rectilinear grid."
    )
    panel.export_button.clicked.connect(panel.export_model)
    layout.addWidget(panel.export_button)


def _build_animation_controls(panel, layout, QtWidgets) -> None:
    group = QtWidgets.QGroupBox("Camera and export")
    grid = QtWidgets.QGridLayout(group)
    panel.rotate_check = QtWidgets.QCheckBox("Rotate around vertical axis")
    panel.rotate_check.setObjectName("volume_rotate_check")
    panel.rotate_check.setText("Animate rotation")
    panel.rotate_check.setToolTip(
        "Continuously orbit the camera around the selected rotation axis."
    )
    panel.rotate_check.toggled.connect(panel._toggle_rotation)
    panel.rotation_axis_combo = QtWidgets.QComboBox()
    panel.rotation_axis_combo.setObjectName("volume_rotation_axis_combo")
    panel.rotation_axis_combo.addItems(
        ["X axis", "Y axis", "Z axis", "Vertical axis"]
    )
    panel.rotation_axis_combo.setCurrentText("Z axis")
    panel.rotation_axis_combo.setToolTip(
        "Rotate around displayed X, Y, or Z, or choose Vertical axis to rotate "
        "around the direction currently vertical on screen."
    )
    panel.rotation_speed_spin = QtWidgets.QDoubleSpinBox()
    panel.rotation_speed_spin.setRange(1.0, 360.0)
    panel.rotation_speed_spin.setValue(30.0)
    panel.rotation_speed_spin.setSuffix(" deg/s")
    panel.rotation_speed_spin.setToolTip(
        "Angular speed used by the live rotation preview."
    )
    panel.movie_fps_spin = QtWidgets.QSpinBox()
    panel.movie_fps_spin.setRange(1, 120)
    panel.movie_fps_spin.setValue(30)
    panel.movie_fps_spin.setToolTip(
        "Frames per second written to the MP4 rotation movie."
    )
    panel.movie_duration_spin = QtWidgets.QDoubleSpinBox()
    panel.movie_duration_spin.setRange(0.5, 120.0)
    panel.movie_duration_spin.setValue(12.0)
    panel.movie_duration_spin.setSuffix(" s")
    panel.movie_duration_spin.setToolTip(
        "Movie duration for one complete 360-degree orbit."
    )
    still_button = QtWidgets.QPushButton("Export still image")
    still_button.setObjectName("volume_export_still_button")
    still_button.setToolTip("Save a PNG image of the current 3D camera view.")
    still_button.clicked.connect(panel.export_still)
    movie_button = QtWidgets.QPushButton("Export rotation MP4")
    movie_button.setObjectName("volume_export_movie_button")
    movie_button.setToolTip(
        "Render one complete rotation around the selected axis, then restore "
        "the current camera view."
    )
    movie_button.clicked.connect(panel.export_rotation_movie)
    grid.addWidget(panel.rotate_check, 0, 0, 1, 2)
    grid.addWidget(QtWidgets.QLabel("Rotation axis"), 1, 0)
    grid.addWidget(panel.rotation_axis_combo, 1, 1)
    grid.addWidget(QtWidgets.QLabel("Preview speed"), 2, 0)
    grid.addWidget(panel.rotation_speed_spin, 2, 1)
    grid.addWidget(QtWidgets.QLabel("Movie FPS"), 3, 0)
    grid.addWidget(panel.movie_fps_spin, 3, 1)
    grid.addWidget(QtWidgets.QLabel("Movie duration"), 4, 0)
    grid.addWidget(panel.movie_duration_spin, 4, 1)
    grid.addWidget(still_button, 5, 0, 1, 2)
    grid.addWidget(movie_button, 6, 0, 1, 2)
    layout.addWidget(group)


def _float_spin(QtWidgets):
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(-1.0e300, 1.0e300)
    spin.setDecimals(8)
    spin.setKeyboardTracking(False)
    _make_compact(spin, QtWidgets)
    return spin


def _make_compact(spin, QtWidgets) -> None:
    spin.setMinimumWidth(0)
    spin.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Ignored,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
