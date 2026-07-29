import sys
from types import SimpleNamespace

import numpy as np

from nfit import (
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
        [np.pi / 8.0, 0.0, 0.0],
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
    script = brillouin_zone_script(component)
    compile(script, "<brillouin-zone-script>", "exec")
    assert "primitive_lattice" in script
    assert "show_brillouin_zone_scene" in script


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

        def add_point_labels(self, *_args, **_kwargs):
            return None

        def add_axes(self, **_kwargs):
            return None

        def reset_camera(self):
            return None

    scene = build_brillouin_zone_scene(
        np.diag([4.0, 4.0, 4.0]),
        [{"label": "Γ", "k": [0.0, 0.0, 0.0]}],
    )
    plotter = Plotter()
    _render_brillouin_zone(plotter, scene)

    face_style = plotter.meshes[0][1]
    edge_style = plotter.meshes[1][1]
    assert face_style["opacity"] == 0.10
    assert face_style["smooth_shading"] is False
    assert face_style["lighting"] is False
    assert edge_style["style"] == "wireframe"
    assert edge_style["line_width"] == 4
    assert len(arrow_arguments) == 3
    reciprocal = np.asarray(scene.reciprocal_vectors)
    for arguments, vector in zip(arrow_arguments, reciprocal, strict=True):
        assert arguments["shaft_radius"] == 0.008
        assert arguments["scale"] == np.linalg.norm(vector)
