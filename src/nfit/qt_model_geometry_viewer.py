"""Qt/PyVista rendering for renderer-independent model geometry scenes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_geometry import ModelGeometryScene, model_geometry_scene


def _render_scene(plotter: Any, scene: ModelGeometryScene) -> None:
    import pyvista as pv

    plotter.clear()
    plotter.set_background("white")
    vertices = np.asarray(scene.cell_vertices, dtype=float)
    for start, stop in scene.cell_edges:
        plotter.add_mesh(
            pv.Line(vertices[start], vertices[stop]),
            color="black",
            line_width=1,
        )
    active = [site for site in scene.sites if site.active]
    ghosts = [site for site in scene.sites if not site.active]
    if ghosts:
        plotter.add_points(
            np.asarray([site.cartesian for site in ghosts]),
            color="lightgray",
            opacity=0.28,
            point_size=15,
            render_points_as_spheres=True,
        )
    if active:
        plotter.add_points(
            np.asarray([site.cartesian for site in active]),
            color="firebrick",
            point_size=20,
            render_points_as_spheres=True,
        )
        plotter.add_point_labels(
            np.asarray([site.cartesian for site in active]),
            [site.representative_label for site in active],
            text_color="black",
            font_size=10,
            shape=None,
            always_visible=True,
        )
    for orbital in scene.orbitals:
        plotter.add_points(
            np.asarray([orbital.display_center_cartesian]),
            color=orbital.color,
            point_size=12,
            render_points_as_spheres=True,
        )
    if scene.orbitals:
        plotter.add_point_labels(
            np.asarray(
                [orbital.display_center_cartesian for orbital in scene.orbitals]
            ),
            [orbital.orbital_label for orbital in scene.orbitals],
            text_color="black",
            font_size=8,
            shape=None,
            always_visible=True,
        )
    frame_scale = 0.12 * min(
        np.linalg.norm(np.asarray(vector, dtype=float))
        for vector in scene.lattice_vectors
    )
    for frame in scene.frames:
        for axis, color in zip(frame.axes_cartesian, ("red", "green", "blue"), strict=True):
            plotter.add_mesh(
                pv.Arrow(
                    start=frame.origin_cartesian,
                    direction=axis,
                    scale=frame_scale,
                ),
                color=color,
            )
    for pathway in scene.pathways:
        plotter.add_mesh(
            pv.Line(pathway.start_cartesian, pathway.end_cartesian),
            color="royalblue" if pathway.kind == "hopping" else "darkorange",
            line_width=5 if pathway.representative else 3,
        )
    plotter.add_axes(color="black")
    plotter.reset_camera()


def show_model_geometry_scene(
    scene: ModelGeometryScene,
    *,
    parent: Any | None = None,
) -> Any:
    """Open a Qt window for a precomputed model-geometry scene."""

    from PySide6 import QtWidgets
    from pyvistaqt import QtInteractor

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow(parent)
    window.setWindowTitle(f"Model geometry — {scene.model_name}")
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)
    plotter = QtInteractor(central)
    plotter.setObjectName("model_geometry_plotter")
    layout.addWidget(plotter.interactor)
    window.setCentralWidget(central)
    window.resize(1000, 760)
    window._nfit_plotter = plotter
    window._nfit_application = application
    _render_scene(plotter, scene)
    window.show()
    return window


def open_model_geometry_viewer(component: Any, *, parent: Any | None = None) -> Any:
    """Open the interactive shared viewer for one model component."""

    from PySide6 import QtWidgets
    from pyvistaqt import QtInteractor

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    class Viewer(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(parent)
            self.setWindowTitle(f"Model geometry — {component.name}")
            central = QtWidgets.QWidget()
            root = QtWidgets.QHBoxLayout(central)
            self.plotter = QtInteractor(central)
            self.plotter.setObjectName("model_geometry_plotter")
            root.addWidget(self.plotter.interactor, 1)
            controls = QtWidgets.QWidget()
            controls.setMaximumWidth(300)
            form = QtWidgets.QVBoxLayout(controls)
            self.ghosts = QtWidgets.QCheckBox("Show inactive atoms as ghosts")
            self.ghosts.setObjectName("model_geometry_show_ghosts")
            self.ghosts.setToolTip(
                "Show every crystallographic atom; sites not used by this model "
                "are translucent."
            )
            self.ghosts.setChecked(True)
            self.orbitals = QtWidgets.QCheckBox("Show orbital tokens")
            self.orbitals.setObjectName("model_geometry_show_orbitals")
            self.orbitals.setToolTip(
                "Show all configured orbitals as separated colored tokens. "
                "Tokens identify basis states; they are not wavefunction isosurfaces."
            )
            self.orbitals.setChecked(component.type == "tight_binding")
            self.frames = QtWidgets.QCheckBox("Show local frames")
            self.frames.setObjectName("model_geometry_show_frames")
            self.frames.setToolTip(
                "Show each manifold's local x, y, and z axes in red, green, and blue."
            )
            self.frames.setChecked(component.type == "tight_binding")
            self.pathway = QtWidgets.QComboBox()
            self.pathway.setObjectName("model_geometry_pathway")
            self.pathway.setToolTip(
                "Show every pathway, or restrict the scene to one hopping or "
                "exchange orbit."
            )
            self.pathway.addItem("All pathways", "")
            initial = model_geometry_scene(component)
            for label in initial.pathway_labels:
                self.pathway.addItem(label, label)
            self.all_equivalent = QtWidgets.QCheckBox(
                "Show all symmetry-equivalent pathways"
            )
            self.all_equivalent.setObjectName("model_geometry_all_equivalent")
            self.all_equivalent.setToolTip(
                "Unchecked shows one representative pathway. Checked shows "
                "every member of the selected symmetry orbit."
            )
            for widget in (
                self.ghosts,
                self.orbitals,
                self.frames,
                QtWidgets.QLabel("Pathway"),
                self.pathway,
                self.all_equivalent,
            ):
                form.addWidget(widget)
            form.addStretch(1)
            root.addWidget(controls)
            self.setCentralWidget(central)
            self.resize(1100, 760)
            self.ghosts.toggled.connect(self.refresh)
            self.orbitals.toggled.connect(self.refresh)
            self.frames.toggled.connect(self.refresh)
            self.pathway.currentIndexChanged.connect(self.refresh)
            self.all_equivalent.toggled.connect(self.refresh)
            self.refresh()

        def refresh(self, *_args: Any) -> None:
            scene = model_geometry_scene(
                component,
                include_ghost_sites=self.ghosts.isChecked(),
                show_orbitals=self.orbitals.isChecked(),
                show_local_frames=self.frames.isChecked(),
                selected_pathway=str(self.pathway.currentData() or "") or None,
                pathway_mode=(
                    "all" if self.all_equivalent.isChecked() else "representative"
                ),
            )
            _render_scene(self.plotter, scene)

        def closeEvent(self, event: Any) -> None:
            self.plotter.close()
            super().closeEvent(event)

    window = Viewer()
    window._nfit_application = application
    window.show()
    return window
