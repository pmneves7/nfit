import numpy as np

from nfit import (
    DataGroup,
    ElectronicModel,
    brillouin_zone_scene,
    brillouin_zone_script,
    build_brillouin_zone_scene,
    create_model_component,
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
    assert "b₁" not in script
    assert "show_brillouin_zone_scene" in script
