"""Qt/PyVista rendering for labelled first-Brillouin-zone scenes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .brillouin_zone import BrillouinZoneScene
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


def _render_brillouin_zone(plotter: Any, scene: BrillouinZoneScene) -> None:
    import pyvista as pv

    plotter.clear()
    plotter.set_background("white")
    if hasattr(plotter, "enable_lightkit"):
        plotter.enable_lightkit()
    zone = _zone_mesh(scene)
    plotter.add_mesh(
        zone,
        color="lightsteelblue",
        opacity=0.10,
        show_edges=False,
        lighting=False,
        smooth_shading=False,
    )
    plotter.add_mesh(
        zone,
        style="wireframe",
        color="black",
        opacity=1.0,
        line_width=4,
        lighting=False,
    )
    nodes = np.asarray(
        [node.cartesian_inv_angstrom for node in scene.path_nodes],
        dtype=float,
    )
    if len(nodes) >= 2:
        for start, stop in zip(nodes[:-1], nodes[1:], strict=True):
            plotter.add_mesh(
                pv.Line(start, stop),
                color="royalblue",
                line_width=5,
            )
    if len(nodes):
        plotter.add_points(
            nodes,
            color="royalblue",
            point_size=12,
            render_points_as_spheres=True,
        )
        plotter.add_point_labels(
            nodes,
            [node.label for node in scene.path_nodes],
            text_color="black",
            font_size=14,
            shape=None,
            always_visible=True,
        )
    reciprocal = np.asarray(scene.reciprocal_vectors, dtype=float)
    colors = ("red", "green", "blue")
    labels = ("b1", "b2", "b3")
    endpoints = []
    for vector, color in zip(reciprocal, colors, strict=True):
        endpoint = vector
        endpoints.append(endpoint)
        plotter.add_mesh(
            pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=vector,
                scale=float(np.linalg.norm(endpoint)),
                tip_length=0.08,
                tip_radius=0.025,
                shaft_radius=0.008,
            ),
            color=color,
            lighting=False,
        )
    plotter.add_point_labels(
        np.asarray(endpoints),
        labels,
        text_color="black",
        font_size=12,
        shape=None,
        always_visible=True,
    )
    plotter.add_axes(color="black")
    plotter.reset_camera()


def show_brillouin_zone_scene(
    scene: BrillouinZoneScene,
    *,
    parent: Any | None = None,
) -> Any:
    """Open a Qt window showing a precomputed Brillouin-zone scene."""

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
    plotter = QtInteractor(central)
    plotter.setObjectName("brillouin_zone_plotter")
    viewport_layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(1100, 760)
    window._nfit_plotter = plotter
    window._nfit_application = application
    window._nfit_settings_panel = settings
    _render_brillouin_zone(plotter, scene)
    window.show()
    return window
