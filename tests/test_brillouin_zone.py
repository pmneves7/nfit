import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nfit import (
    BrillouinZoneViewOptions,
    DataGroup,
    ElectronicModel,
    brillouin_zone_scene,
    brillouin_zone_script,
    build_brillouin_zone_scene,
    create_model_component,
    primitive_lattice_vectors,
)
from nfit.electronic_builder import (
    add_tight_binding_orbital_manifold,
    orbital_manifold_preset,
)
from nfit.model_plots import tight_binding_band_structure


def test_cubic_first_brillouin_zone_and_labelled_path():
    scene = build_brillouin_zone_scene(
        np.diag([4.0, 4.0, 4.0]),
        [
            {"label": "Γ", "k": [0.0, 0.0, 0.0]},
            {"label": "X", "k": [0.5, 0.0, 0.0]},
            {"label": "M", "k": [0.5, 0.5, 0.0]},
        ],
    )

    assert len(scene.vertices_inv_angstrom) == 8
    assert len(scene.faces) == 6
    assert [node.label for node in scene.path_nodes] == ["Γ", "X", "M"]
    np.testing.assert_allclose(
        np.asarray(scene.reciprocal_vectors),
        np.diag([np.pi / 2.0] * 3),
    )


def test_f_centered_cubic_zone_uses_primitive_translation_lattice():
    lattice = {
        "a": 8.0,
        "b": 8.0,
        "c": 8.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    conventional = np.diag([8.0, 8.0, 8.0])
    primitive = primitive_lattice_vectors(lattice, "F d -3 m:2")
    scene = build_brillouin_zone_scene(
        conventional,
        [
            {"label": "Γ", "k": [0.0, 0.0, 0.0]},
            {"label": "X", "k": [0.5, 0.0, 0.0]},
        ],
        primitive_lattice=primitive,
    )

    assert len(scene.vertices_inv_angstrom) == 24
    assert len(scene.faces) == 14
    assert sorted(len(face) for face in scene.faces) == [4] * 6 + [6] * 8
    assert np.isclose(
        abs(np.linalg.det(primitive)),
        np.linalg.det(conventional) / 4.0,
    )
    np.testing.assert_allclose(
        scene.path_nodes[1].cartesian_inv_angstrom,
        np.asarray(scene.reciprocal_vectors[0]) / 2.0,
        atol=1e-12,
    )

    group = DataGroup("Centered")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"] = {
        "lattice": lattice,
        "spacegroup": "F d -3 m:2",
        "sites": [
            {
                "label": "M1",
                "element": "Fe",
                "position": [0.0, 0.0, 0.0],
                "ion": "",
            }
        ],
    }
    add_tight_binding_orbital_manifold(
        component,
        orbital_manifold_preset("M1", "effective"),
    )
    component_scene = brillouin_zone_scene(component)
    assert len(component_scene.vertices_inv_angstrom) == 24
    assert len(component_scene.faces) == 14
    bands = tight_binding_band_structure(component)
    expected_first_segment = np.linalg.norm(
        np.asarray(component_scene.reciprocal_vectors[0]) / 2.0
    )
    first_node_index = bands.sampling.labels[1][0]
    assert bands.sampling.path_distance_inv_angstrom[
        first_node_index
    ] == pytest.approx(expected_first_segment)


def test_component_brillouin_zone_and_script_use_configured_band_path():
    group = DataGroup("Electronic")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"]["sites"] = [
        {
            "label": "M1",
            "element": "Fe",
            "position": [0.0, 0.0, 0.0],
            "ion": "",
        }
    ]
    add_tight_binding_orbital_manifold(
        component,
        orbital_manifold_preset("M1", "effective"),
    )
    component.config["band_path"] = [
        {"label": "Γ", "k": [0.0, 0.0, 0.0]},
        {"label": "R", "k": [0.5, 0.5, 0.5]},
    ]

    scene = brillouin_zone_scene(component)
    assert [node.label for node in scene.path_nodes] == ["Γ", "R"]
    canonical = ElectronicModel.from_dict(component.config["model_data"])
    expected_reciprocal = 2.0 * np.pi * np.linalg.inv(
        canonical.direct_lattice
    ).T
    np.testing.assert_allclose(
        np.asarray(scene.reciprocal_vectors).T,
        expected_reciprocal,
    )
    component.config["crystal"]["lattice"]["a"] = 99.0
    np.testing.assert_allclose(
        np.asarray(brillouin_zone_scene(component).reciprocal_vectors).T,
        expected_reciprocal,
    )
    script = brillouin_zone_script(
        component,
        view_options=BrillouinZoneViewOptions(
            path_color="#551111",
            projection="perspective",
        ),
    )
    compile(script, "<brillouin-zone-script>", "exec")
    assert "primitive_lattice" in script
    assert "BrillouinZoneViewOptions" in script
    assert "path_color='#551111'" in script
    assert "projection='perspective'" in script
    assert "show_brillouin_zone_scene" in script


def test_f_centered_default_path_uses_primitive_reciprocal_coordinates():
    lattice = {
        "a": 8.0,
        "b": 8.0,
        "c": 8.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    scene = build_brillouin_zone_scene(
        np.diag([8.0, 8.0, 8.0]),
        [
            {"label": "Γ", "k": [0.0, 0.0, 0.0]},
            {"label": "X", "k": [0.5, 0.0, 0.0]},
            {"label": "M", "k": [0.5, 0.5, 0.0]},
        ],
        primitive_lattice=primitive_lattice_vectors(lattice, "F d -3 m:2"),
    )
    reciprocal = np.asarray(scene.reciprocal_vectors)
    np.testing.assert_allclose(
        scene.path_nodes[1].cartesian_inv_angstrom,
        reciprocal[0] / 2.0,
    )
    np.testing.assert_allclose(
        scene.path_nodes[2].cartesian_inv_angstrom,
        (reciprocal[0] + reciprocal[1]) / 2.0,
    )


def test_view_options_validate_scriptable_appearance():
    options = BrillouinZoneViewOptions(
        basis_vector_color_mode="rgb",
        path_color="#551111",
        show_compass=False,
        projection="perspective",
    )
    assert options.basis_vector_color_mode == "rgb"
    assert options.show_compass is False
    assert BrillouinZoneViewOptions().show_compass is False
    with pytest.raises(ValueError, match="projection"):
        BrillouinZoneViewOptions(projection="fish-eye")
    with pytest.raises(ValueError, match="inside_style"):
        BrillouinZoneViewOptions(basis_vector_inside_style="fade")


def test_basis_vector_surface_and_dashed_modes(monkeypatch):
    from nfit.qt_brillouin_zone_viewer import _render_brillouin_zone

    arrows = []
    lines = []
    fake_pyvista = SimpleNamespace(
        PolyData=lambda points, faces: ("zone", points, faces),
        Line=lambda start, stop: lines.append(
            (np.asarray(start), np.asarray(stop))
        )
        or ("line", start, stop),
        Arrow=lambda **kwargs: arrows.append(kwargs) or ("arrow", kwargs),
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    class Plotter:
        def clear(self):
            pass

        def set_background(self, _color):
            pass

        def add_mesh(self, *_args, **_kwargs):
            pass

        def add_points(self, *_args, **_kwargs):
            pass

        def add_point_labels(self, points, labels, **_kwargs):
            self.labels = (points, labels)

        def add_axes(self, **_kwargs):
            pass

        def reset_camera(self):
            pass

    scene = build_brillouin_zone_scene(
        np.diag([4.0, 4.0, 4.0]),
        [{"label": "Γ", "k": [0.0, 0.0, 0.0]}],
    )
    plotter = Plotter()
    _render_brillouin_zone(
        plotter,
        scene,
        BrillouinZoneViewOptions(basis_vector_inside_style="hidden"),
    )
    reciprocal = np.asarray(scene.reciprocal_vectors)
    for arrow, vector in zip(arrows, reciprocal, strict=True):
        np.testing.assert_allclose(arrow["start"], vector / 2.0)
        assert arrow["scale"] == pytest.approx(np.linalg.norm(vector) / 2.0)
    assert plotter.labels[1] == ("b₁", "b₂", "b₃")

    arrows.clear()
    lines.clear()
    _render_brillouin_zone(
        plotter,
        scene,
        BrillouinZoneViewOptions(basis_vector_inside_style="dashed"),
    )
    assert len(lines) == 21
    assert len(arrows) == 3
    for arrow, vector in zip(arrows, reciprocal, strict=True):
        np.testing.assert_allclose(arrow["start"], vector / 2.0)


def test_zone_renderer_uses_flat_faces_heavy_outline_and_thin_full_vectors(
    monkeypatch,
):
    from nfit.qt_brillouin_zone_viewer import _render_brillouin_zone

    arrow_arguments = []
    fake_pyvista = SimpleNamespace(
        PolyData=lambda points, faces: ("zone", points, faces),
        Line=lambda start, stop: ("line", start, stop),
        Arrow=lambda **kwargs: arrow_arguments.append(kwargs) or ("arrow", kwargs),
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    class Plotter:
        def __init__(self):
            self.meshes = []
            self.point_labels = []

        def clear(self):
            return None

        def set_background(self, _color):
            return None

        def enable_lightkit(self):
            return None

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def add_points(self, *_args, **_kwargs):
            return None

        def add_point_labels(self, *_args, **kwargs):
            self.point_labels.append(kwargs)

        def add_axes(self, **_kwargs):
            return None

        def reset_camera(self):
            return None

    scene = build_brillouin_zone_scene(
        np.diag([4.0, 4.0, 4.0]),
        [{"label": "Γ", "k": [0.0, 0.0, 0.0]}],
    )
    plotter = Plotter()
    options = BrillouinZoneViewOptions(
        basis_vector_color_mode="rgb",
        path_color="#551111",
        cell_surface_opacity=0.25,
        cell_outline_thickness=5.0,
    )
    _render_brillouin_zone(plotter, scene, options)

    face_style = plotter.meshes[0][1]
    edge_style = plotter.meshes[1][1]
    assert face_style["opacity"] == 0.25
    assert face_style["smooth_shading"] is False
    assert face_style["lighting"] is False
    assert edge_style["style"] == "wireframe"
    assert edge_style["line_width"] == 5.0
    assert len(arrow_arguments) == 3
    reciprocal = np.asarray(scene.reciprocal_vectors)
    for arguments, vector in zip(arrow_arguments, reciprocal, strict=True):
        assert arguments["shaft_radius"] == options.basis_vector_thickness
        assert arguments["scale"] == np.linalg.norm(vector)
    assert plotter.point_labels[-1]["show_points"] is False


def test_zone_viewer_uses_standard_right_settings_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    class FakeInteractor:
        def __init__(self, parent):
            self.interactor = QtWidgets.QWidget(parent)
            self.saved_paths = []

        def setObjectName(self, name):
            self.interactor.setObjectName(name)

        def screenshot(self, path=None, *, return_img=False):
            if return_img:
                return np.zeros((12, 16, 3), dtype=np.uint8)
            self.saved_paths.append(path)

    monkeypatch.setitem(
        sys.modules,
        "pyvistaqt",
        SimpleNamespace(QtInteractor=FakeInteractor),
    )
    monkeypatch.setattr(
        "nfit.qt_brillouin_zone_viewer._render_brillouin_zone",
        lambda *_args, **_kwargs: None,
    )
    from nfit.qt_brillouin_zone_viewer import show_brillouin_zone_scene

    scene = build_brillouin_zone_scene(
        np.diag([4.0, 4.0, 4.0]),
        [{"label": "Γ", "k": [0.0, 0.0, 0.0]}],
    )
    window = show_brillouin_zone_scene(scene)
    panel = window.findChild(
        QtWidgets.QGroupBox,
        "brillouin_zone_settings_panel",
    )
    assert panel is not None
    assert panel.title() == "Settings"
    assert window.centralWidget().layout().itemAt(1).widget() is panel
    assert (
        window.findChild(
            QtWidgets.QComboBox,
            "brillouin_zone_basis_color_mode",
        ).currentData()
        == "single"
    )
    assert window.findChild(
        QtWidgets.QPushButton,
        "brillouin_zone_copy_figure",
    ).toolTip()
    assert window.findChild(
        QtWidgets.QPushButton,
        "brillouin_zone_save_figure",
    ).toolTip()
    for object_name in (
        "brillouin_zone_show_basis_vectors",
        "brillouin_zone_show_path",
        "brillouin_zone_show_path_labels",
        "brillouin_zone_show_compass",
        "brillouin_zone_basis_color_mode",
        "brillouin_zone_basis_color",
        "brillouin_zone_basis_thickness",
        "brillouin_zone_basis_inside_style",
        "brillouin_zone_path_color",
        "brillouin_zone_path_thickness",
        "brillouin_zone_label_font_size",
        "brillouin_zone_surface_color",
        "brillouin_zone_surface_opacity",
        "brillouin_zone_outline_color",
        "brillouin_zone_outline_thickness",
        "brillouin_zone_projection",
    ):
        control = window.findChild(QtWidgets.QWidget, object_name)
        assert control is not None
        assert control.toolTip()
    projection = window.findChild(
        QtWidgets.QComboBox,
        "brillouin_zone_projection",
    )
    assert projection.currentData() == "orthographic"
    assert not window.findChild(
        QtWidgets.QCheckBox,
        "brillouin_zone_show_compass",
    ).isChecked()
    projection.setCurrentIndex(projection.findData("perspective"))
    assert window._nfit_view_options.projection == "perspective"
    window._nfit_copy_figure()
    assert not QtWidgets.QApplication.clipboard().pixmap().isNull()
    output_path = tmp_path / "zone.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output_path), "Images (*.png)"),
    )
    window._nfit_save_figure()
    assert window._nfit_plotter.saved_paths == [str(output_path)]
    window.close()
