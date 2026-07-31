"""Qt/PyVista rendering for labelled first-Brillouin-zone scenes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .brillouin_zone import BrillouinZoneScene, BrillouinZoneViewOptions
from .qt_pyvista import configure_pyvista_interactor, show_then_render
from .qt_viewer_shell import create_viewer_shell


@lru_cache(maxsize=2)
def _label_font_file(*, bold: bool = False) -> str:
    """Return a bundled font containing Greek and Unicode subscript glyphs."""

    from matplotlib import get_data_path

    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(get_data_path()) / "fonts" / "ttf" / filename
    if not path.is_file():
        raise FileNotFoundError("Matplotlib's DejaVu Sans font is unavailable")
    return str(path)


def _unique_path_labels(
    scene: BrillouinZoneScene,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return one label per coincident path position."""

    positions: list[np.ndarray] = []
    labels: list[list[str]] = []
    raw_positions = np.asarray(
        [node.cartesian_inv_angstrom for node in scene.path_nodes],
        dtype=float,
    )
    scale = max(
        1.0,
        float(np.max(np.linalg.norm(raw_positions, axis=1)))
        if len(raw_positions)
        else 1.0,
    )
    tolerance = 1.0e-10 * scale
    for node, position in zip(scene.path_nodes, raw_positions, strict=True):
        match = next(
            (
                index
                for index, existing in enumerate(positions)
                if np.linalg.norm(position - existing) <= tolerance
            ),
            None,
        )
        if match is None:
            positions.append(position)
            labels.append([node.label])
        elif node.label not in labels[match]:
            labels[match].append(node.label)
    points = np.asarray(positions, dtype=float).reshape((-1, 3))
    return points, tuple("|".join(group) for group in labels)


def _zone_mesh(scene: BrillouinZoneScene) -> Any:
    import pyvista as pv

    encoded = np.concatenate(
        [
            np.asarray((len(face), *face), dtype=np.int64)
            for face in scene.faces
        ]
    )
    return pv.PolyData(
        np.asarray(scene.vertices_inv_angstrom, dtype=float),
        faces=encoded,
    )


def _basis_vector_surface_fraction(
    scene: BrillouinZoneScene,
    vector: Any,
) -> float:
    """Return where a ray from Gamma first exits the convex zone."""

    from scipy.spatial import ConvexHull

    direction = np.asarray(vector, dtype=float)
    hull = ConvexHull(np.asarray(scene.vertices_inv_angstrom, dtype=float))
    denominators = hull.equations[:, :3] @ direction
    active = denominators > 1.0e-12
    if not np.any(active):
        raise ValueError("reciprocal vector does not exit the Brillouin zone")
    fractions = -hull.equations[active, 3] / denominators[active]
    positive = fractions[fractions >= 0.0]
    if not len(positive):
        raise ValueError("could not locate the Brillouin-zone surface")
    return float(np.min(positive))


def _dashed_segments(
    start: Any,
    stop: Any,
    *,
    dash_count: int = 7,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    beginning = np.asarray(start, dtype=float)
    ending = np.asarray(stop, dtype=float)
    result = []
    for index in range(int(dash_count)):
        left = (2.0 * index) / (2.0 * dash_count - 1.0)
        right = (2.0 * index + 1.0) / (2.0 * dash_count - 1.0)
        result.append(
            (
                beginning + left * (ending - beginning),
                beginning + min(right, 1.0) * (ending - beginning),
            )
        )
    return tuple(result)


def _render_brillouin_zone(
    plotter: Any,
    scene: BrillouinZoneScene,
    options: BrillouinZoneViewOptions | None = None,
    *,
    reset_camera: bool = True,
) -> None:
    """Render ``scene`` with reusable, scriptable presentation options."""

    import pyvista as pv

    settings = BrillouinZoneViewOptions() if options is None else options
    plotter.clear()
    plotter.set_background("white")
    if hasattr(plotter, "enable_lightkit"):
        plotter.enable_lightkit()
    zone = _zone_mesh(scene)
    plotter.add_mesh(
        zone,
        color=settings.cell_surface_color,
        opacity=settings.cell_surface_opacity,
        show_edges=False,
        lighting=False,
        smooth_shading=False,
    )
    plotter.add_mesh(
        zone,
        style="wireframe",
        color=settings.cell_outline_color,
        opacity=1.0,
        line_width=settings.cell_outline_thickness,
        lighting=False,
    )
    nodes = np.asarray(
        [node.cartesian_inv_angstrom for node in scene.path_nodes],
        dtype=float,
    )
    if settings.show_path and len(nodes) >= 2:
        for index, (start, stop) in enumerate(
            zip(nodes[:-1], nodes[1:], strict=True),
            start=1,
        ):
            if scene.path_nodes[index].break_before:
                continue
            plotter.add_mesh(
                pv.Line(start, stop),
                color=settings.path_color,
                line_width=settings.path_thickness,
            )
    if settings.show_path and len(nodes):
        plotter.add_points(
            nodes,
            color=settings.path_color,
            point_size=settings.path_point_size,
            render_points_as_spheres=True,
        )
    if settings.show_path_labels and len(nodes):
        label_points, label_text = _unique_path_labels(scene)
        plotter.add_point_labels(
            label_points,
            label_text,
            text_color="black",
            font_size=settings.label_font_size,
            bold=settings.label_bold,
            font_file=_label_font_file(bold=settings.label_bold),
            shape=None,
            show_points=False,
            always_visible=True,
        )
    if settings.show_basis_vectors:
        reciprocal = np.asarray(scene.reciprocal_vectors, dtype=float)
        colors = (
            ("red", "green", "blue")
            if settings.basis_vector_color_mode == "rgb"
            else (settings.basis_vector_color,) * 3
        )
        labels = ("b₁", "b₂", "b₃")
        endpoints = []
        for vector, color in zip(reciprocal, colors, strict=True):
            endpoints.append(vector)
            surface_fraction = _basis_vector_surface_fraction(scene, vector)
            surface = surface_fraction * vector
            arrow_start = (
                np.zeros(3)
                if settings.basis_vector_inside_style == "solid"
                else surface
            )
            arrow_vector = vector - arrow_start
            if settings.basis_vector_inside_style == "dashed":
                shaft_radius = (
                    settings.basis_vector_thickness
                    * float(np.linalg.norm(arrow_vector))
                )
                for dash_start, dash_stop in _dashed_segments(
                    np.zeros(3),
                    surface,
                ):
                    difference = dash_stop - dash_start
                    plotter.add_mesh(
                        pv.Cylinder(
                            center=(dash_start + dash_stop) / 2.0,
                            direction=difference,
                            radius=shaft_radius,
                            height=float(np.linalg.norm(difference)),
                            resolution=16,
                            capping=True,
                        ),
                        color=color,
                        lighting=False,
                    )
            plotter.add_mesh(
                pv.Arrow(
                    start=arrow_start,
                    direction=arrow_vector,
                    scale=float(np.linalg.norm(arrow_vector)),
                    tip_length=0.09,
                    tip_radius=3.0 * settings.basis_vector_thickness,
                    shaft_radius=settings.basis_vector_thickness,
                ),
                color=color,
                lighting=False,
            )
        plotter.add_point_labels(
            np.asarray(endpoints),
            labels,
            text_color="black",
            font_size=settings.label_font_size,
            bold=settings.label_bold,
            font_file=_label_font_file(bold=settings.label_bold),
            shape=None,
            show_points=False,
            always_visible=True,
        )
    if settings.show_compass:
        plotter.add_axes(
            x_color="#FF0000",
            y_color="#00A000",
            z_color="#0000FF",
            color="black",
        )
        if hasattr(plotter, "show_axes"):
            plotter.show_axes()
    elif hasattr(plotter, "hide_axes"):
        plotter.hide_axes()
    if settings.projection == "orthographic":
        if hasattr(plotter, "enable_parallel_projection"):
            plotter.enable_parallel_projection()
    elif hasattr(plotter, "disable_parallel_projection"):
        plotter.disable_parallel_projection()
    if reset_camera:
        plotter.reset_camera()
    elif hasattr(plotter, "render"):
        plotter.render()


def _clear_layout(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _populate_settings_panel(
    settings_panel: Any,
    *,
    initial: BrillouinZoneViewOptions,
    on_change: Callable[[BrillouinZoneViewOptions], None],
    copy_figure: Callable[[], None],
    save_figure: Callable[[], None],
) -> None:
    from PySide6 import QtGui, QtWidgets

    layout = settings_panel.layout()
    _clear_layout(layout)
    state = {"options": initial}

    def update(**changes: Any) -> None:
        state["options"] = replace(state["options"], **changes)
        on_change(state["options"])

    def check_box(
        text: str,
        object_name: str,
        checked: bool,
        tooltip: str,
        field: str,
    ) -> Any:
        control = QtWidgets.QCheckBox(text)
        control.setObjectName(object_name)
        control.setChecked(checked)
        control.setToolTip(tooltip)
        control.toggled.connect(
            lambda value, field=field: update(**{field: bool(value)})
        )
        return control

    def color_button(
        object_name: str,
        color: str,
        tooltip: str,
        field: str,
    ) -> Any:
        def style(value: str) -> str:
            foreground = (
                "white" if QtGui.QColor(value).lightness() < 128 else "black"
            )
            return (
                f"QPushButton {{ background-color: {value}; "
                f"color: {foreground}; }}"
            )

        control = QtWidgets.QPushButton(color.upper())
        control.setObjectName(object_name)
        control.setToolTip(tooltip)
        control.setStyleSheet(style(color))

        def choose_color() -> None:
            selected = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(getattr(state["options"], field)),
                settings_panel,
                "Choose color",
            )
            if not selected.isValid():
                return
            name = selected.name().upper()
            control.setText(name)
            control.setStyleSheet(style(name))
            update(**{field: name})

        control.clicked.connect(choose_color)
        return control

    visibility = QtWidgets.QGroupBox("Visibility")
    visibility.setObjectName("brillouin_zone_visibility_group")
    visibility_layout = QtWidgets.QVBoxLayout(visibility)
    for control in (
        check_box(
            "Reciprocal basis vectors",
            "brillouin_zone_show_basis_vectors",
            initial.show_basis_vectors,
            "Show or hide the reciprocal basis vectors b1, b2, and b3.",
            "show_basis_vectors",
        ),
        check_box(
            "High-symmetry path",
            "brillouin_zone_show_path",
            initial.show_path,
            "Show or hide the line segments joining configured path nodes.",
            "show_path",
        ),
        check_box(
            "High-symmetry labels",
            "brillouin_zone_show_path_labels",
            initial.show_path_labels,
            "Show or hide labels at configured high-symmetry path nodes.",
            "show_path_labels",
        ),
        check_box(
            "XYZ compass",
            "brillouin_zone_show_compass",
            initial.show_compass,
            "Show or hide the Cartesian orientation compass in the viewport.",
            "show_compass",
        ),
    ):
        visibility_layout.addWidget(control)
    layout.addWidget(visibility)

    vectors = QtWidgets.QGroupBox("Reciprocal vectors")
    vectors.setObjectName("brillouin_zone_vectors_group")
    vectors_layout = QtWidgets.QFormLayout(vectors)
    color_mode = QtWidgets.QComboBox()
    color_mode.setObjectName("brillouin_zone_basis_color_mode")
    color_mode.setToolTip(
        "Draw b1, b2, and b3 in red, green, and blue, or use one neutral color."
    )
    color_mode.addItem("One color", "single")
    color_mode.addItem("RGB", "rgb")
    color_mode.setCurrentIndex(
        max(0, color_mode.findData(initial.basis_vector_color_mode))
    )
    vector_color = color_button(
        "brillouin_zone_basis_color",
        initial.basis_vector_color,
        "Choose the shared reciprocal-vector color used in one-color mode.",
        "basis_vector_color",
    )
    vector_color.setEnabled(initial.basis_vector_color_mode == "single")

    def set_color_mode(_index: int) -> None:
        mode = str(color_mode.currentData())
        vector_color.setEnabled(mode == "single")
        update(basis_vector_color_mode=mode)

    color_mode.currentIndexChanged.connect(set_color_mode)
    vector_thickness = QtWidgets.QLineEdit(
        f"{initial.basis_vector_thickness:.6g}"
    )
    vector_thickness.setObjectName("brillouin_zone_basis_thickness")
    vector_thickness.setToolTip(
        "Set the positive reciprocal-vector shaft radius as a fraction of "
        "arrow length. Values near 0.001–0.05 are usually useful."
    )

    def commit_vector_thickness() -> None:
        try:
            value = float(vector_thickness.text())
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError
        except ValueError:
            vector_thickness.setText(
                f"{state['options'].basis_vector_thickness:.6g}"
            )
            return
        vector_thickness.setText(f"{value:.6g}")
        update(basis_vector_thickness=value)

    vector_thickness.editingFinished.connect(commit_vector_thickness)
    vectors_layout.addRow("Colors", color_mode)
    vectors_layout.addRow("Single color", vector_color)
    vectors_layout.addRow("Thickness", vector_thickness)
    inside_style = QtWidgets.QComboBox()
    inside_style.setObjectName("brillouin_zone_basis_inside_style")
    inside_style.setToolTip(
        "Choose whether reciprocal vectors are solid from Gamma, dashed "
        "inside the Wigner–Seitz cell, or begin at its surface."
    )
    inside_style.addItem("Solid throughout", "solid")
    inside_style.addItem("Dashed inside cell", "dashed")
    inside_style.addItem("Start at cell surface", "hidden")
    inside_style.setCurrentIndex(
        max(0, inside_style.findData(initial.basis_vector_inside_style))
    )
    inside_style.currentIndexChanged.connect(
        lambda _index: update(
            basis_vector_inside_style=str(inside_style.currentData())
        )
    )
    vectors_layout.addRow("Inside cell", inside_style)
    layout.addWidget(vectors)

    path = QtWidgets.QGroupBox("Path and labels")
    path.setObjectName("brillouin_zone_path_group")
    path_layout = QtWidgets.QFormLayout(path)
    path_color = color_button(
        "brillouin_zone_path_color",
        initial.path_color,
        "Choose the high-symmetry path color.",
        "path_color",
    )
    path_thickness = QtWidgets.QDoubleSpinBox()
    path_thickness.setObjectName("brillouin_zone_path_thickness")
    path_thickness.setRange(0.5, 20.0)
    path_thickness.setSingleStep(0.5)
    path_thickness.setValue(initial.path_thickness)
    path_thickness.setSuffix(" px")
    path_thickness.setToolTip("Set the rendered high-symmetry path width.")
    path_thickness.valueChanged.connect(
        lambda value: update(path_thickness=float(value))
    )
    path_point_size = QtWidgets.QDoubleSpinBox()
    path_point_size.setObjectName("brillouin_zone_path_point_size")
    path_point_size.setRange(1.0, 80.0)
    path_point_size.setSingleStep(1.0)
    path_point_size.setValue(initial.path_point_size)
    path_point_size.setSuffix(" px")
    path_point_size.setToolTip(
        "Set the diameter of high-symmetry path points. Increase this on "
        "high-resolution displays."
    )
    path_point_size.valueChanged.connect(
        lambda value: update(path_point_size=float(value))
    )
    font_size = QtWidgets.QSpinBox()
    font_size.setObjectName("brillouin_zone_label_font_size")
    font_size.setRange(6, 48)
    font_size.setValue(initial.label_font_size)
    font_size.setSuffix(" pt")
    font_size.setToolTip(
        "Set the font size for path-node and reciprocal-vector labels."
    )
    font_size.valueChanged.connect(
        lambda value: update(label_font_size=int(value))
    )
    label_bold = QtWidgets.QCheckBox("Bold labels")
    label_bold.setObjectName("brillouin_zone_label_bold")
    label_bold.setChecked(initial.label_bold)
    label_bold.setToolTip(
        "Use bold text for high-symmetry and reciprocal-vector labels."
    )
    label_bold.toggled.connect(
        lambda checked: update(label_bold=bool(checked))
    )
    path_layout.addRow("Path color", path_color)
    path_layout.addRow("Path width", path_thickness)
    path_layout.addRow("Point size", path_point_size)
    path_layout.addRow("Label size", font_size)
    path_layout.addRow("", label_bold)
    layout.addWidget(path)

    cell = QtWidgets.QGroupBox("Wigner–Seitz cell")
    cell.setObjectName("brillouin_zone_cell_group")
    cell_layout = QtWidgets.QFormLayout(cell)
    surface_color = color_button(
        "brillouin_zone_surface_color",
        initial.cell_surface_color,
        "Choose the color of the transparent Wigner–Seitz faces.",
        "cell_surface_color",
    )
    surface_opacity = QtWidgets.QDoubleSpinBox()
    surface_opacity.setObjectName("brillouin_zone_surface_opacity")
    surface_opacity.setRange(0.0, 1.0)
    surface_opacity.setDecimals(2)
    surface_opacity.setSingleStep(0.05)
    surface_opacity.setValue(initial.cell_surface_opacity)
    surface_opacity.setToolTip(
        "Set face opacity from zero (invisible) to one (opaque)."
    )
    surface_opacity.valueChanged.connect(
        lambda value: update(cell_surface_opacity=float(value))
    )
    outline_color = color_button(
        "brillouin_zone_outline_color",
        initial.cell_outline_color,
        "Choose the Wigner–Seitz cell outline color.",
        "cell_outline_color",
    )
    outline_thickness = QtWidgets.QDoubleSpinBox()
    outline_thickness.setObjectName("brillouin_zone_outline_thickness")
    outline_thickness.setRange(0.5, 20.0)
    outline_thickness.setSingleStep(0.5)
    outline_thickness.setValue(initial.cell_outline_thickness)
    outline_thickness.setSuffix(" px")
    outline_thickness.setToolTip("Set the Wigner–Seitz outline width.")
    outline_thickness.valueChanged.connect(
        lambda value: update(cell_outline_thickness=float(value))
    )
    cell_layout.addRow("Face color", surface_color)
    cell_layout.addRow("Face opacity", surface_opacity)
    cell_layout.addRow("Outline color", outline_color)
    cell_layout.addRow("Outline width", outline_thickness)
    layout.addWidget(cell)

    camera = QtWidgets.QGroupBox("Camera")
    camera.setObjectName("brillouin_zone_camera_group")
    camera_layout = QtWidgets.QFormLayout(camera)
    projection = QtWidgets.QComboBox()
    projection.setObjectName("brillouin_zone_projection")
    projection.setToolTip(
        "Orthographic projection preserves parallel lines; perspective adds "
        "depth foreshortening."
    )
    projection.addItem("Orthographic", "orthographic")
    projection.addItem("Perspective", "perspective")
    projection.setCurrentIndex(max(0, projection.findData(initial.projection)))
    projection.currentIndexChanged.connect(
        lambda _index: update(projection=str(projection.currentData()))
    )
    camera_layout.addRow("Projection", projection)
    layout.addWidget(camera)

    output = QtWidgets.QGroupBox("Output")
    output.setObjectName("brillouin_zone_output_group")
    output_layout = QtWidgets.QHBoxLayout(output)
    copy_button = QtWidgets.QPushButton("Copy figure")
    copy_button.setObjectName("brillouin_zone_copy_figure")
    copy_button.setToolTip(
        "Copy the current Brillouin-zone viewport image to the clipboard."
    )
    copy_button.clicked.connect(copy_figure)
    save_button = QtWidgets.QPushButton("Save figure")
    save_button.setObjectName("brillouin_zone_save_figure")
    save_button.setToolTip("Save the current Brillouin-zone viewport as an image.")
    save_button.clicked.connect(save_figure)
    output_layout.addWidget(copy_button)
    output_layout.addWidget(save_button)
    layout.addWidget(output)
    layout.addStretch(1)


def show_brillouin_zone_scene(
    scene: BrillouinZoneScene,
    *,
    options: BrillouinZoneViewOptions | None = None,
    parent: Any | None = None,
) -> Any:
    """Open an interactively styled Qt window for a precomputed zone scene."""

    from PySide6 import QtCore, QtGui, QtWidgets
    from pyvistaqt import MainWindow, QtInteractor

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    window = MainWindow(parent=parent, title="First Brillouin zone")
    window.setObjectName("brillouin_zone_viewer")
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key="brillouin_zone",
    )
    settings.setFixedWidth(320)
    plotter = QtInteractor(central, auto_update=False)
    configure_pyvista_interactor(plotter)
    plotter.setObjectName("brillouin_zone_plotter")
    viewport_layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(1180, 820)
    window.signal_close.connect(plotter.close)
    current = BrillouinZoneViewOptions() if options is None else options

    def redraw(updated: BrillouinZoneViewOptions) -> None:
        camera_position = getattr(plotter, "camera_position", None)
        _render_brillouin_zone(
            plotter,
            scene,
            updated,
            reset_camera=camera_position is None,
        )
        if camera_position is not None:
            plotter.camera_position = camera_position
            if hasattr(plotter, "render"):
                plotter.render()
        window._nfit_view_options = updated
        if window.isVisible():
            window.raise_()
            window.activateWindow()
            QtCore.QTimer.singleShot(0, window.raise_)

    def copy_figure() -> None:
        image = np.ascontiguousarray(plotter.screenshot(return_img=True))
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("PyVista returned an unsupported screenshot format")
        height, width, channels = image.shape
        image_format = (
            QtGui.QImage.Format.Format_RGB888
            if channels == 3
            else QtGui.QImage.Format.Format_RGBA8888
        )
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            image_format,
        ).copy()
        QtWidgets.QApplication.clipboard().setImage(qimage)

    def save_figure() -> None:
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            window,
            "Save Brillouin-zone figure",
            "nfit_brillouin_zone.png",
            "Images (*.png *.jpg *.jpeg)",
        )
        if path:
            plotter.screenshot(str(path))

    window._nfit_plotter = plotter
    window._nfit_application = application
    window._nfit_settings_panel = settings
    window._nfit_view_options = current
    window._nfit_copy_figure = copy_figure
    window._nfit_save_figure = save_figure
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_close_shortcut = close_shortcut
    _populate_settings_panel(
        settings,
        initial=current,
        on_change=redraw,
        copy_figure=copy_figure,
        save_figure=save_figure,
    )
    show_then_render(
        window,
        lambda: _render_brillouin_zone(plotter, scene, current),
    )
    return window
