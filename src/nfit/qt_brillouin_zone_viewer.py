"""Qt/PyVista rendering for labelled first-Brillouin-zone scenes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from .brillouin_zone import BrillouinZoneScene, BrillouinZoneViewOptions
from .qt_viewer_shell import create_viewer_shell


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
        for start, stop in zip(nodes[:-1], nodes[1:], strict=True):
            plotter.add_mesh(
                pv.Line(start, stop),
                color=settings.path_color,
                line_width=settings.path_thickness,
            )
    if settings.show_path and len(nodes):
        plotter.add_points(
            nodes,
            color=settings.path_color,
            point_size=max(6.0, 2.5 * settings.path_thickness),
            render_points_as_spheres=True,
        )
    if settings.show_path_labels and len(nodes):
        plotter.add_point_labels(
            nodes,
            [node.label for node in scene.path_nodes],
            text_color="black",
            font_size=settings.label_font_size,
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
        labels = ("b1", "b2", "b3")
        endpoints = []
        for vector, color in zip(reciprocal, colors, strict=True):
            endpoints.append(vector)
            plotter.add_mesh(
                pv.Arrow(
                    start=(0.0, 0.0, 0.0),
                    direction=vector,
                    scale=float(np.linalg.norm(vector)),
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
            shape=None,
            show_points=False,
            always_visible=True,
        )
    if hasattr(plotter, "hide_axes"):
        plotter.hide_axes()
    if settings.show_compass:
        plotter.add_axes(color="black")
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
    vector_thickness = QtWidgets.QDoubleSpinBox()
    vector_thickness.setObjectName("brillouin_zone_basis_thickness")
    vector_thickness.setRange(0.001, 0.05)
    vector_thickness.setDecimals(3)
    vector_thickness.setSingleStep(0.001)
    vector_thickness.setValue(initial.basis_vector_thickness)
    vector_thickness.setToolTip(
        "Set the reciprocal-vector shaft radius as a fraction of arrow length."
    )
    vector_thickness.valueChanged.connect(
        lambda value: update(basis_vector_thickness=float(value))
    )
    vectors_layout.addRow("Colors", color_mode)
    vectors_layout.addRow("Single color", vector_color)
    vectors_layout.addRow("Thickness", vector_thickness)
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
    path_layout.addRow("Path color", path_color)
    path_layout.addRow("Path width", path_thickness)
    path_layout.addRow("Label size", font_size)
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

    from PySide6 import QtWidgets
    from pyvistaqt import QtInteractor

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow(parent)
    window.setObjectName("brillouin_zone_viewer")
    window.setWindowTitle("First Brillouin zone")
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key="brillouin_zone",
    )
    settings.setFixedWidth(320)
    plotter = QtInteractor(central)
    plotter.setObjectName("brillouin_zone_plotter")
    viewport_layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(1180, 820)
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

    def copy_figure() -> None:
        QtWidgets.QApplication.clipboard().setPixmap(plotter.interactor.grab())

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
    _populate_settings_panel(
        settings,
        initial=current,
        on_change=redraw,
        copy_figure=copy_figure,
        save_figure=save_figure,
    )
    _render_brillouin_zone(plotter, scene, current)
    window.show()
    return window
