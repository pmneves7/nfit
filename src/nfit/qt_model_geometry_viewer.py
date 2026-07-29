"""Qt/PyVista rendering for renderer-independent model geometry scenes."""

from __future__ import annotations

import atexit
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .model_geometry import (
    GeometryOrbital,
    GeometryPathway,
    GeometrySite,
    ModelGeometryScene,
    model_geometry_scene,
)
from .qt_pyvista import configure_pyvista_interactor, show_then_render


@dataclass(frozen=True)
class _PickBatch:
    kind: str
    items: tuple[Any, ...]
    radius: float = 0.0


def _actor_key(actor: Any) -> str:
    if actor is None:
        return ""
    if hasattr(actor, "GetAddressAsString"):
        return str(actor.GetAddressAsString(""))
    return f"python:{id(actor)}"


def _distance_to_segment(
    point: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> float:
    direction = stop - start
    length_squared = float(direction @ direction)
    if length_squared <= 1e-20:
        return float(np.linalg.norm(point - start))
    fraction = float((point - start) @ direction) / length_squared
    closest = start + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(point - closest))


def _pick_batch_item(batch: _PickBatch, point: np.ndarray) -> Any | None:
    if not batch.items:
        return None
    location = np.asarray(point, dtype=float)
    if batch.kind == "site":
        return min(
            batch.items,
            key=lambda item: abs(
                np.linalg.norm(location - np.asarray(item.cartesian))
                - item.display_radius
            ),
        )
    if batch.kind == "orbital":
        return min(
            batch.items,
            key=lambda item: abs(
                np.linalg.norm(
                    location - np.asarray(item.display_center_cartesian)
                )
                - batch.radius
            ),
        )
    if batch.kind == "pathway":
        return min(
            batch.items,
            key=lambda item: _distance_to_segment(
                location,
                np.asarray(item.start_cartesian),
                np.asarray(item.end_cartesian),
            ),
        )
    return None


def _selection_text(item: Any) -> str:
    if isinstance(item, GeometrySite):
        activity = "active model site" if item.active else "inactive atom"
        return (
            f"Atom: {item.site_label}\n"
            f"Representative site: {item.representative_label}\n"
            f"Element: {item.element or '(unspecified)'}\n"
            f"Role: {activity}"
        )
    if isinstance(item, GeometryOrbital):
        return (
            f"Orbital: {item.orbital_label}\n"
            f"Manifold: {item.manifold_label}\n"
            f"Site: {item.site_identifier}"
        )
    if isinstance(item, GeometryPathway):
        kind = "Hopping bond" if item.kind == "hopping" else "Exchange bond"
        member = (
            "representative"
            if item.representative
            else "symmetry-equivalent member"
        )
        return (
            f"{kind}: {item.orbit_label}\n"
            f"Member: {member}\n"
            f"Identifier: {item.identifier}"
        )
    return "Nothing selected."


def _selection_from_pick(
    registry: dict[str, _PickBatch],
    actor: Any,
    point: np.ndarray,
) -> str | None:
    batch = registry.get(_actor_key(actor))
    if batch is None:
        return None
    item = _pick_batch_item(batch, point)
    return None if item is None else _selection_text(item)


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


def _render_scene(
    plotter: Any,
    scene: ModelGeometryScene,
) -> dict[str, _PickBatch]:
    registry: dict[str, _PickBatch] = {}

    def register(actor: Any, batch: _PickBatch) -> None:
        key = _actor_key(actor)
        if key:
            registry[key] = batch

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
            pickable=False,
        )
    active = [site for site in scene.sites if site.active]
    ghosts = [site for site in scene.sites if not site.active]
    if ghosts:
        actor = plotter.add_mesh(
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
        register(actor, _PickBatch("site", tuple(ghosts)))
    if active:
        actor = plotter.add_mesh(
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
        register(actor, _PickBatch("site", tuple(active)))
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
        actor = plotter.add_mesh(
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
        register(
            actor,
            _PickBatch(
                "orbital",
                tuple(scene.orbitals),
                orbital_radius,
            ),
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
                pickable=False,
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
            actor = plotter.add_mesh(
                _line_segments(
                    np.asarray([pathway.start_cartesian for pathway in pathways]),
                    np.asarray([pathway.end_cartesian for pathway in pathways]),
                ),
                color="royalblue" if kind == "hopping" else "darkorange",
                line_width=5 if representative else 3,
            )
            register(actor, _PickBatch("pathway", tuple(pathways)))
    plotter.add_axes(color="black")
    plotter.reset_camera()
    return registry


def _selection_panel(QtWidgets: Any) -> tuple[Any, Any]:
    panel = QtWidgets.QGroupBox("Selected object")
    panel.setObjectName("model_geometry_selection_group")
    layout = QtWidgets.QVBoxLayout(panel)
    label = QtWidgets.QLabel(
        "Nothing selected.\nClick an atom, orbital, or bond in the 3D view."
    )
    label.setObjectName("model_geometry_selection")
    label.setToolTip(
        "Reports the crystallographic atom, orbital basis state, or "
        "hopping/exchange pathway under the last left click."
    )
    label.setWordWrap(True)
    layout.addWidget(label)
    return panel, label


def _enable_geometry_picking(
    plotter: Any,
    registry: Callable[[], dict[str, _PickBatch]],
    update_text: Callable[[str], None],
) -> None:
    def picked(point: np.ndarray, picker: Any) -> None:
        actor = picker.GetActor() if hasattr(picker, "GetActor") else None
        text = _selection_from_pick(registry(), actor, point)
        if text is not None:
            update_text(text)

    plotter.enable_point_picking(
        callback=picked,
        left_clicking=True,
        picker="cell",
        show_message=False,
        show_point=False,
        use_picker=True,
        pickable_window=False,
    )


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
    layout = QtWidgets.QHBoxLayout(central)
    plotter = QtInteractor(central, auto_update=False)
    configure_pyvista_interactor(plotter)
    plotter.setObjectName("model_geometry_plotter")
    layout.addWidget(plotter.interactor, 1)
    controls = QtWidgets.QWidget()
    controls.setMaximumWidth(300)
    controls_layout = QtWidgets.QVBoxLayout(controls)
    selection_panel, selection_label = _selection_panel(QtWidgets)
    controls_layout.addWidget(selection_panel)
    controls_layout.addStretch(1)
    layout.addWidget(controls)
    window.setCentralWidget(central)
    window.resize(1100, 760)
    window._nfit_plotter = plotter
    window._nfit_application = application
    window._nfit_pick_registry = {}
    _enable_geometry_picking(
        plotter,
        lambda: window._nfit_pick_registry,
        selection_label.setText,
    )

    def initial_render() -> None:
        window._nfit_pick_registry = _render_scene(plotter, scene)

    show_then_render(window, initial_render)
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
            self.plotter = QtInteractor(central, auto_update=False)
            configure_pyvista_interactor(self.plotter)
            self.plotter.setObjectName("model_geometry_plotter")
            root.addWidget(self.plotter.interactor, 1)
            controls = QtWidgets.QWidget()
            controls.setMaximumWidth(300)
            form = QtWidgets.QVBoxLayout(controls)
            selection_panel, self.selection = _selection_panel(QtWidgets)
            form.addWidget(selection_panel)
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
            self._pick_registry: dict[str, _PickBatch] = {}
            _enable_geometry_picking(
                self.plotter,
                lambda: self._pick_registry,
                self.selection.setText,
            )

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
            self._pick_registry = _render_scene(self.plotter, scene)
            self.selection.setText(
                "Nothing selected.\n"
                "Click an atom, orbital, or bond in the 3D view."
            )

        def closeEvent(self, event: Any) -> None:
            self.plotter.close()
            super().closeEvent(event)

    window = Viewer()
    window._nfit_application = application
    show_then_render(window, window.refresh)
    return window
