"""Qt/PyVista rendering for renderer-independent model geometry scenes."""

from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

import numpy as np

from .model_geometry import ModelGeometryScene, model_geometry_scene


@lru_cache(maxsize=1)
def _sphere_source() -> Any:
    import pyvista as pv

    return pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=12)


@lru_cache(maxsize=1)
def _arrow_source() -> Any:
    import pyvista as pv

    return pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        tip_resolution=12,
        shaft_resolution=8,
    )


atexit.register(_sphere_source.cache_clear)
atexit.register(_arrow_source.cache_clear)


def _sphere_glyphs(
    points: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
) -> Any:
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(points, dtype=float))
    cloud["display_radius"] = np.asarray(radii, dtype=float)
    cloud["rgb"] = np.asarray(np.rint(255.0 * colors), dtype=np.uint8)
    return cloud.glyph(
        scale="display_radius",
        geom=_sphere_source(),
        orient=False,
    )


def _line_segments(
    starts: np.ndarray,
    stops: np.ndarray,
) -> Any:
    import pyvista as pv

    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    points = np.empty((2 * len(starts), 3), dtype=float)
    points[0::2] = starts
    points[1::2] = stops
    lines = np.column_stack(
        (
            np.full(len(starts), 2, dtype=np.int64),
            2 * np.arange(len(starts), dtype=np.int64),
            2 * np.arange(len(starts), dtype=np.int64) + 1,
        )
    )
    return pv.PolyData(points, lines=lines)


def _arrow_glyphs(origins: np.ndarray, directions: np.ndarray, scale: float) -> Any:
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(origins, dtype=float))
    cloud["direction"] = np.asarray(directions, dtype=float)
    cloud["display_scale"] = np.full(len(origins), float(scale))
    return cloud.glyph(
        orient="direction",
        scale="display_scale",
        geom=_arrow_source(),
    )


def _render_scene(plotter: Any, scene: ModelGeometryScene) -> None:
    plotter.clear()
    plotter.set_background("white")
    if hasattr(plotter, "enable_lightkit"):
        plotter.enable_lightkit()
    vertices = np.asarray(scene.cell_vertices, dtype=float)
    edge_starts = np.asarray([vertices[start] for start, _stop in scene.cell_edges])
    edge_stops = np.asarray([vertices[stop] for _start, stop in scene.cell_edges])
    if len(edge_starts):
        plotter.add_mesh(
            _line_segments(edge_starts, edge_stops),
            color="black",
            line_width=1,
        )
    active = [site for site in scene.sites if site.active]
    ghosts = [site for site in scene.sites if not site.active]
    if ghosts:
        plotter.add_mesh(
            _sphere_glyphs(
                np.asarray([site.cartesian for site in ghosts]),
                np.asarray([site.display_radius for site in ghosts]),
                np.asarray([site.color for site in ghosts]),
            ),
            scalars="rgb",
            rgb=True,
            opacity=0.28,
            smooth_shading=True,
            interpolation="phong",
            lighting=True,
            ambient=0.18,
            diffuse=0.82,
            specular=0.30,
            specular_power=24.0,
        )
    if active:
        plotter.add_mesh(
            _sphere_glyphs(
                np.asarray([site.cartesian for site in active]),
                np.asarray([site.display_radius for site in active]),
                np.asarray([site.color for site in active]),
            ),
            scalars="rgb",
            rgb=True,
            smooth_shading=True,
            interpolation="phong",
            lighting=True,
            ambient=0.18,
            diffuse=0.82,
            specular=0.35,
            specular_power=28.0,
        )
        plotter.add_point_labels(
            np.asarray([site.cartesian for site in active]),
            [site.representative_label for site in active],
            text_color="black",
            font_size=10,
            shape=None,
            always_visible=True,
        )
    if scene.orbitals:
        orbital_radius = 0.024 * min(
            np.linalg.norm(np.asarray(vector, dtype=float))
            for vector in scene.lattice_vectors
        )
        plotter.add_mesh(
            _sphere_glyphs(
                np.asarray(
                    [orbital.display_center_cartesian for orbital in scene.orbitals]
                ),
                np.full(len(scene.orbitals), orbital_radius),
                np.asarray([orbital.color for orbital in scene.orbitals]),
            ),
            scalars="rgb",
            rgb=True,
            smooth_shading=True,
            interpolation="phong",
            lighting=True,
            ambient=0.20,
            diffuse=0.80,
            specular=0.30,
            specular_power=24.0,
        )
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
    for axis_index, color in enumerate(("red", "green", "blue")):
        if scene.frames:
            plotter.add_mesh(
                _arrow_glyphs(
                    np.asarray([frame.origin_cartesian for frame in scene.frames]),
                    np.asarray(
                        [frame.axes_cartesian[axis_index] for frame in scene.frames]
                    ),
                    frame_scale,
                ),
                color=color,
            )
    for kind, representative in (
        ("hopping", True),
        ("hopping", False),
        ("exchange", True),
        ("exchange", False),
    ):
        pathways = [
            pathway
            for pathway in scene.pathways
            if pathway.kind == kind and pathway.representative == representative
        ]
        if pathways:
            plotter.add_mesh(
                _line_segments(
                    np.asarray([pathway.start_cartesian for pathway in pathways]),
                    np.asarray([pathway.end_cartesian for pathway in pathways]),
                ),
                color="royalblue" if kind == "hopping" else "darkorange",
                line_width=5 if representative else 3,
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
            self.hopping_term = QtWidgets.QComboBox()
            self.hopping_term.setObjectName("model_geometry_hopping_term")
            self.hopping_term.setToolTip(
                "Restrict the displayed pathway to the bond orbit generated "
                "by one representative hopping-matrix coefficient."
            )
            self.hopping_term.addItem("All hopping matrix terms", "")
            for term in component.config.get("hopping_terms", ()):
                self.hopping_term.addItem(
                    str(term.get("label", "")),
                    str(term.get("identifier", "")),
                )
            self.hopping_term.setVisible(component.type == "tight_binding")
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
                self.hopping_term,
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
            self.hopping_term.currentIndexChanged.connect(self.refresh)
            self.all_equivalent.toggled.connect(self.refresh)
            self.refresh()

        def refresh(self, *_args: Any) -> None:
            scene = model_geometry_scene(
                component,
                include_ghost_sites=self.ghosts.isChecked(),
                show_orbitals=self.orbitals.isChecked(),
                show_local_frames=self.frames.isChecked(),
                selected_pathway=str(self.pathway.currentData() or "") or None,
                selected_hopping_term=(
                    str(self.hopping_term.currentData() or "") or None
                ),
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
