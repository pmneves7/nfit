import copy

import numpy as np
import pytest

from nfit import (
    DataGroup,
    ModelComponentSpec,
    NfitProject,
    OrbitalManifold,
    OrbitalSymmetryError,
    add_tight_binding_orbital_manifold,
    create_model_component,
    generate_onsite_terms,
    load_project,
    manifold_symmetry_representation,
    model_geometry_scene,
    model_geometry_script,
    onsite_invariants,
    orbital_manifold_from_site_symmetry,
    orbital_manifold_preset,
    save_project,
    set_tight_binding_onsite_term,
    site_point_group_symbol,
    site_symmetry_harmonic_submanifolds,
    site_symmetry_operations,
    spherical_harmonic_representation,
    tight_binding_structure_script,
)


def _crystal(spacegroup="P 1"):
    return {
        "lattice": {
            "a": 4.0,
            "b": 4.0,
            "c": 4.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": spacegroup,
        "sites": [
            {
                "label": "M1",
                "element": "Fe",
                "position": [0.0, 0.0, 0.0],
                "ion": "",
            },
            {
                "label": "X1",
                "element": "O",
                "position": [0.5, 0.5, 0.5],
                "ion": "",
            },
        ],
    }


def _nonclosed_d_manifold():
    transform = np.zeros((5, 1), dtype=np.complex128)
    transform[0, 0] = 1.0
    return OrbitalManifold(
        site_label="M1",
        label="M1_partial_d",
        basis_kind="real_harmonic",
        orbitals=("d_partial",),
        l=2,
        harmonic_transform=transform,
        preset="custom_subspace",
    )


def test_spherical_harmonic_representations_are_unitary_and_use_local_frames():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    representation = spherical_harmonic_representation(2, rotation)
    np.testing.assert_allclose(
        representation.conj().T @ representation, np.eye(5), atol=1e-10
    )
    inversion = spherical_harmonic_representation(2, -np.eye(3))
    np.testing.assert_allclose(inversion, np.eye(5), atol=1e-10)

    frame = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    manifold = orbital_manifold_preset("M1", "p", local_frame=frame)
    local = manifold_symmetry_representation(manifold, rotation)
    expected = spherical_harmonic_representation(1, frame.T @ rotation @ frame)
    np.testing.assert_allclose(local, expected, atol=1e-10)


def test_cubic_d_shell_splits_into_two_onsite_invariants():
    crystal = _crystal("P m -3 m")
    manifold = orbital_manifold_preset("M1", "d")
    terms = onsite_invariants(crystal, [manifold], "M1")

    assert len(terms) == 2
    assert all(term.kind == "onsite_energy" for term in terms)
    for operation in site_symmetry_operations(crystal, "M1"):
        representation = manifold_symmetry_representation(
            manifold, operation["rotation_cartesian"]
        )
        for term in terms:
            np.testing.assert_allclose(
                representation @ term.matrix @ representation.conj().T,
                term.matrix,
                atol=1e-8,
            )


def test_site_point_group_generates_crystal_specific_harmonic_submanifolds():
    crystal = _crystal("P m -3 m")
    options = site_symmetry_harmonic_submanifolds(crystal, "M1", "d")

    assert site_point_group_symbol(crystal, "M1") == "m-3m"
    assert sorted(option.dimension for option in options) == [2, 3]
    assert {tuple(option.orbitals) for option in options} == {
        ("d_xy", "d_yz", "d_zx"),
        ("d_x2_y2", "d_z2"),
    }
    selected = orbital_manifold_from_site_symmetry(
        crystal,
        "M1",
        "d",
        options[0].identifier,
    )
    assert selected.site_point_group == "m-3m"
    assert selected.submanifold_id == options[0].identifier
    assert selected.preset == "site_symmetry"
    assert len(generate_onsite_terms(crystal, [selected])) == 1


def test_scalar_site_symmetry_does_not_invent_crystal_field_subspaces():
    options = site_symmetry_harmonic_submanifolds(_crystal("P 1"), "M1", "f")

    assert len(options) == 1
    assert options[0].dimension == 7
    assert options[0].orbitals == orbital_manifold_preset("M1", "f").orbitals


def test_nonclosed_submanifold_is_diagnosed_instead_of_projected():
    crystal = _crystal("P m -3 m")
    manifold = _nonclosed_d_manifold()
    with pytest.raises(OrbitalSymmetryError, match="not closed"):
        generate_onsite_terms(crystal, [manifold])


def test_failed_component_symmetry_update_is_atomic():
    group = DataGroup("Electronic")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"] = _crystal("P m -3 m")
    before = copy.deepcopy(component.config)

    with pytest.raises(OrbitalSymmetryError):
        add_tight_binding_orbital_manifold(
            component, _nonclosed_d_manifold()
        )
    assert component.config == before


def test_declared_degeneracy_must_be_closed_under_site_symmetry():
    payload = orbital_manifold_preset("M1", "d").to_dict()
    payload["degeneracy_groups"] = [["d_xy", "d_z2"]]
    manifold = OrbitalManifold.from_dict(payload)

    with pytest.raises(OrbitalSymmetryError, match="degeneracy group.*not closed"):
        generate_onsite_terms(_crystal("P m -3 m"), [manifold])


def test_custom_basis_uses_declared_degeneracy_without_inferred_hybridization():
    manifold = OrbitalManifold(
        site_label="M1",
        label="M1_custom",
        basis_kind="custom",
        orbitals=("a", "b", "c"),
        degeneracy_groups=(("a", "b"),),
        symmetry_mode="none",
    )
    terms = generate_onsite_terms(_crystal(), [manifold])

    assert len(terms) == 2
    assert all(term.kind == "onsite_energy" for term in terms)
    assert all(
        np.allclose(term.matrix, np.diag(np.diag(term.matrix))) for term in terms
    )


def test_component_builder_resolves_values_project_and_script_round_trip(tmp_path):
    group = DataGroup("Electronic")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"] = _crystal("P m -3 m")
    manifold = orbital_manifold_preset(
        "M1", "d", correlated_shell="M1_3d"
    )
    add_tight_binding_orbital_manifold(component, manifold)
    assert len(component.config["onsite_terms"]) == 2
    first = component.config["onsite_terms"][0]["identifier"]
    set_tight_binding_onsite_term(
        component,
        first,
        value=0.025,
        lower=-0.2,
        upper=0.2,
        energy_unit="eV",
        fit=True,
    )

    resolved = component.config["model_data"]
    assert component.config["model_digest"] == resolved["content_digest"]
    assert sorted(component.config["projection_groups"]) == ["M1_d"]
    assert component.config["onsite_terms"][0]["value_meV"] == pytest.approx(25.0)
    assert component.config["onsite_terms"][0]["bounds_meV"] == [-200.0, 200.0]
    assert component.config["onsite_terms"][0]["fit"] is True

    project_path = tmp_path / "builder.nfit"
    save_project(NfitProject([group]), project_path)
    restored = load_project(project_path).data_groups[0].models["bands"]
    assert restored.config["model_digest"] == component.config["model_digest"]
    assert restored.config["orbital_manifolds"] == component.config["orbital_manifolds"]
    assert restored.config["onsite_terms"] == component.config["onsite_terms"]

    script = tight_binding_structure_script(
        component.config["crystal"],
        component.config["periodic_axes"] or [0, 1, 2],
        orbital_manifolds=component.config["orbital_manifolds"],
        onsite_terms=component.config["onsite_terms"],
        expected_model_digest=component.config["model_digest"],
    )
    namespace = {}
    exec(compile(script, "<builder-script>", "exec"), namespace)
    assert (
        namespace["electronic_model"].content_digest
        == component.config["model_digest"]
    )
    assert namespace["model"].config["onsite_terms"][0]["fit"] is True


def test_model_geometry_scene_shows_active_ghost_orbitals_and_frames():
    component = ModelComponentSpec(
        name="electrons",
        type="tight_binding",
        config={
            "crystal": _crystal(),
            "orbital_manifolds": [
                orbital_manifold_preset("M1", "p").to_dict()
            ],
        },
    )
    scene = model_geometry_scene(component)
    active = [site for site in scene.sites if site.active]
    ghosts = [site for site in scene.sites if not site.active]

    assert len(active) == 1
    assert len(ghosts) == 1
    assert len(scene.orbitals) == 3
    assert len(scene.frames) == 1
    assert len(scene.cell_edges) == 12
    assert active[0].color != ghosts[0].color
    assert active[0].display_radius > ghosts[0].display_radius
    active_only = model_geometry_scene(component, include_ghost_sites=False)
    assert all(site.active for site in active_only.sites)
    script = model_geometry_script(component)
    compile(script, "<model-geometry-script>", "exec")
    assert "owns_app" in script


def test_model_geometry_renderer_batches_true_sphere_glyphs():
    from nfit.qt_model_geometry_viewer import _render_scene

    component = ModelComponentSpec(
        name="electrons",
        type="tight_binding",
        config={
            "crystal": _crystal(),
            "orbital_manifolds": [
                orbital_manifold_preset("M1", "p").to_dict()
            ],
        },
    )
    scene = model_geometry_scene(component)

    class Plotter:
        def __init__(self):
            self.meshes = []

        def clear(self):
            pass

        def set_background(self, _color):
            pass

        def add_mesh(self, mesh, **_kwargs):
            self.meshes.append(mesh)

        def add_point_labels(self, *_args, **_kwargs):
            pass

        def add_axes(self, **_kwargs):
            pass

        def reset_camera(self):
            pass

    plotter = Plotter()
    _render_scene(plotter, scene)

    assert len(plotter.meshes) == 7
    assert any(mesh.n_cells > 100 for mesh in plotter.meshes)


def test_model_geometry_scene_reuses_heisenberg_exchange_pathways():
    component = ModelComponentSpec(
        name="spins",
        type="heisenberg_rpa",
        config={
            "crystal": _crystal(),
            "magnetic_sites": ["M1"],
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "distance_angstrom": 4.0,
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
                        {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
                    ],
                }
            ],
        },
    )
    representative = model_geometry_scene(component, selected_pathway="J1")
    equivalent = model_geometry_scene(
        component, selected_pathway="J1", pathway_mode="all"
    )

    assert len(representative.pathways) == 1
    assert representative.pathways[0].kind == "exchange"
    assert len(equivalent.pathways) == 2
