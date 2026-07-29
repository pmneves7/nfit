"""Qt/PyVista rendering for labelled first-Brillouin-zone scenes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .brillouin_zone import BrillouinZoneScene


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
    plotter.add_mesh(
        _zone_mesh(scene),
        color="aliceblue",
        opacity=0.22,
        show_edges=True,
        edge_color="black",
        line_width=2,
        lighting=True,
        smooth_shading=True,
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
    vector_scale = 0.52
    colors = ("red", "green", "blue")
    labels = ("b₁", "b₂", "b₃")
    endpoints = []
    for vector, color in zip(reciprocal, colors, strict=True):
        endpoint = vector_scale * vector
        endpoints.append(endpoint)
        plotter.add_mesh(
            pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=vector,
                scale=float(np.linalg.norm(endpoint)),
            ),
            color=color,
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
    window.setWindowTitle("First Brillouin zone")
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)
    plotter = QtInteractor(central)
    plotter.setObjectName("brillouin_zone_plotter")
    layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(900, 760)
    window._nfit_plotter = plotter
    window._nfit_application = application
    _render_brillouin_zone(plotter, scene)
    window.show()
    return window
