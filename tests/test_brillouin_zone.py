import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nfit import (
    BrillouinZoneViewOptions,
    DataGroup,
    brillouin_zone_scene,
    brillouin_zone_script,
    build_brillouin_zone_scene,
    create_model_component,
    primitive_lattice_vectors,
    set_tight_binding_standard_path,
    standard_band_path,
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
    canonical_direct = primitive_lattice_vectors(
        component.config["crystal"]["lattice"],
        component.config["crystal"]["spacegroup"],
    )
    expected_reciprocal = 2.0 * np.pi * np.linalg.inv(
        canonical_direct
    ).T
    np.testing.assert_allclose(
        np.asarray(scene.reciprocal_vectors).T,
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


def test_brillouin_zone_label_defaults_and_bold_option():
    options = BrillouinZoneViewOptions()
    assert options.label_font_size == 18
    assert options.label_bold is False
    assert BrillouinZoneViewOptions(label_bold=True).label_bold is True


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
    cylinders = []
    fake_pyvista = SimpleNamespace(
        PolyData=lambda points, faces: ("zone", points, faces),
        Line=lambda start, stop: ("line", start, stop),
        Cylinder=lambda **kwargs: cylinders.append(kwargs)
        or ("cylinder", kwargs),
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
    cylinders.clear()
    _render_brillouin_zone(
        plotter,
        scene,
        BrillouinZoneViewOptions(basis_vector_inside_style="dashed"),
    )
    assert len(cylinders) == 21
    assert len(arrows) == 3
    for arrow, vector in zip(arrows, reciprocal, strict=True):
        np.testing.assert_allclose(arrow["start"], vector / 2.0)
    expected_radius = (
        BrillouinZoneViewOptions().basis_vector_thickness
        * np.linalg.norm(reciprocal[0] / 2.0)
    )
    assert all(
        cylinder["radius"] == pytest.approx(expected_radius)
        for cylinder in cylinders[:7]
    )


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
            self.axes = []
            self.axes_shown = False

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

        def add_axes(self, **kwargs):
            self.axes.append(kwargs)

        def show_axes(self):
            self.axes_shown = True

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
        label_bold=True,
        show_compass=True,
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
    assert plotter.point_labels[-1]["font_size"] == 18
    assert plotter.point_labels[-1]["bold"] is True
    assert plotter.point_labels[-1]["font_file"].endswith("DejaVuSans.ttf")
    assert plotter.axes[-1]["x_color"] == "#FF0000"
    assert plotter.axes[-1]["y_color"] == "#00A000"
    assert plotter.axes[-1]["z_color"] == "#0000FF"
    assert plotter.axes[-1]["color"] == "black"
    assert plotter.axes_shown is True


def test_zone_viewer_uses_standard_right_settings_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    class FakeInteractor:
        def __init__(self, parent, **_kwargs):
            self.interactor = QtWidgets.QWidget(parent)
            self.saved_paths = []
            self.close_count = 0

        def setObjectName(self, name):
            self.interactor.setObjectName(name)

        def screenshot(self, path=None, *, return_img=False):
            if return_img:
                return np.zeros((12, 16, 3), dtype=np.uint8)
            self.saved_paths.append(path)

        def close(self):
            self.close_count += 1

    class FakeMainWindow(QtWidgets.QMainWindow):
        signal_close = QtCore.Signal()

        def __init__(self, parent=None, title=None):
            super().__init__(parent)
            self.setWindowTitle(title or "")

        def closeEvent(self, event):
            self.signal_close.emit()
            super().closeEvent(event)

    monkeypatch.setitem(
        sys.modules,
        "pyvistaqt",
        SimpleNamespace(
            MainWindow=FakeMainWindow,
            QtInteractor=FakeInteractor,
        ),
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
        "brillouin_zone_label_bold",
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
    bold = window.findChild(
        QtWidgets.QCheckBox,
        "brillouin_zone_label_bold",
    )
    assert not bold.isChecked()
    bold.setChecked(True)
    assert window._nfit_view_options.label_bold is True
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
    thickness = window.findChild(
        QtWidgets.QLineEdit,
        "brillouin_zone_basis_thickness",
    )
    thickness.setText("0.009")
    thickness.editingFinished.emit()
    assert window._nfit_view_options.basis_vector_thickness == pytest.approx(
        0.009
    )
    window.close()
    assert window._nfit_plotter.close_count == 1


def test_hinuma_standard_path_uses_primitive_reciprocal_coordinates():
    crystal = {
        "lattice": {
            "a": 8.0,
            "b": 8.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "F d -3 m:2",
        "sites": [
            {
                "label": "M1",
                "element": "Fe",
                "position": [0.0, 0.0, 0.0],
            }
        ],
    }

    path = standard_band_path(crystal)
    labels = {node["label"] for node in path}
    assert {"Γ", "X", "L", "W"}.issubset(labels)
    x_node = next(node for node in path if node["label"] == "X")
    assert x_node["k"] != [0.5, 0.0, 0.0]
    assert any(bool(node.get("break_before", False)) for node in path)


def test_setyawan_curtarolo_path_uses_ase_and_nfit_primitive_coordinates():
    crystal = {
        "lattice": {
            "a": 8.0,
            "b": 8.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "F d -3 m:2",
        "sites": [
            {
                "label": "M1",
                "element": "Fe",
                "position": [0.0, 0.0, 0.0],
            }
        ],
    }

    path = standard_band_path(
        crystal,
        convention="setyawan_curtarolo",
    )
    assert [node["label"] for node in path[:6]] == [
        "Γ",
        "X",
        "W",
        "K",
        "Γ",
        "L",
    ]
    x_node = next(node for node in path if node["label"] == "X")
    np.testing.assert_allclose(x_node["k"], [0.5, 0.0, 0.5], atol=1.0e-12)
    assert any(bool(node.get("break_before", False)) for node in path)

    component = create_model_component(
        DataGroup("electronic"),
        "bands",
        type="tight_binding",
    )
    component.config["crystal"] = crystal
    set_tight_binding_standard_path(
        component,
        "setyawan_curtarolo",
    )
    assert component.config["band_path_convention"] == "setyawan_curtarolo"
    assert component.config["band_path_metadata"]["provider"] == "ASE"
    assert (
        component.config["band_path_metadata"]["convention"]
        == "Setyawan-Curtarolo"
    )
