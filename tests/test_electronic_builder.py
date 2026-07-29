import copy

import numpy as np
import pytest

from nfit import (
    DataGroup,
    ElectronicModel,
    ModelComponentSpec,
    NfitProject,
    OrbitalManifold,
    OrbitalSymmetryError,
    add_tight_binding_hopping_term,
    add_tight_binding_orbital_manifold,
    create_model_component,
    electronic_model_from_component,
    generate_onsite_terms,
    hopping_endpoint_orbitals,
    load_project,
    manifold_symmetry_representation,
    model_geometry_scene,
    model_geometry_script,
    onsite_invariants,
    orbital_manifold_from_site_symmetry,
    orbital_manifold_preset,
    regenerate_tight_binding_hopping_terms,
    remove_tight_binding_hopping_term,
    save_project,
    set_tight_binding_hopping_term,
    set_tight_binding_onsite_term,
    set_tight_binding_parameter_state,
    site_point_group_symbol,
    site_symmetry_harmonic_submanifolds,
    site_symmetry_operations,
    spherical_harmonic_representation,
    tight_binding_structure_script,
)
from nfit.fit_config import component_parameter_names


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


def test_trigonal_repeated_irreps_are_named_as_distinct_copies():
    crystal = _crystal("F d -3 m:2")
    crystal["sites"][0]["position"] = [0.5, 0.5, 0.5]
    options = site_symmetry_harmonic_submanifolds(crystal, "M1", "d")

    assert [option.dimension for option in options] == [2, 1, 2]
    assert [option.irrep_label for option in options] == ["Eg", "A1g", "Eg"]
    assert "Eg copy 1/2" in options[0].label
    assert "Eg copy 2/2" in options[2].label
    assert options[0].label != options[2].label


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
    assert component.parameters[first] == pytest.approx(25.0)
    assert component.limits[first] == [-200.0, 200.0]
    assert component.fit_parameters[first] is True
    assert component.sharing[first]["mode"] == "global"
    assert first in component_parameter_names(component)

    set_tight_binding_parameter_state(
        component,
        values_meV={first: 30.0},
        fit_parameters={first: True},
        limits_meV={first: (-250.0, 250.0)},
        sharing={
            first: {
                "mode": "grouped",
                "groups": {"scan1": "low", "scan2": "low"},
            }
        },
    )
    assert component.config["onsite_terms"][0]["value_meV"] == pytest.approx(30.0)
    assert component.config["onsite_terms"][0]["bounds_meV"] == [-250.0, 250.0]
    assert electronic_model_from_component(component).parameter_values[
        first
    ] == pytest.approx(30.0)
    state_before_invalid_update = copy.deepcopy(
        (
            component.parameters,
            component.fit_parameters,
            component.limits,
            component.sharing,
            component.config,
        )
    )
    with pytest.raises(ValueError, match="sharing mode"):
        set_tight_binding_parameter_state(
            component,
            values_meV={first: 99.0},
            sharing={first: {"mode": "invalid"}},
        )
    assert (
        component.parameters,
        component.fit_parameters,
        component.limits,
        component.sharing,
        component.config,
    ) == state_before_invalid_update

    project_path = tmp_path / "builder.nfit"
    save_project(NfitProject([group]), project_path)
    restored = load_project(project_path).data_groups[0].models["bands"]
    assert restored.config["model_digest"] == component.config["model_digest"]
    assert restored.config["orbital_manifolds"] == component.config["orbital_manifolds"]
    assert restored.config["onsite_terms"] == component.config["onsite_terms"]
    assert restored.parameters == component.parameters
    assert restored.limits == component.limits
    assert restored.fit_parameters == component.fit_parameters
    assert restored.sharing == component.sharing

    script = tight_binding_structure_script(
        component.config["crystal"],
        component.config["periodic_axes"] or [0, 1, 2],
        orbital_manifolds=component.config["orbital_manifolds"],
        onsite_terms=component.config["onsite_terms"],
        parameter_values_meV=component.parameters,
        fit_parameters=component.fit_parameters,
        parameter_limits_meV=component.limits,
        parameter_sharing=component.sharing,
        expected_model_digest=component.config["model_digest"],
    )
    namespace = {}
    exec(compile(script, "<builder-script>", "exec"), namespace)
    assert (
        namespace["electronic_model"].content_digest
        == component.config["model_digest"]
    )
    assert namespace["model"].config["onsite_terms"][0]["fit"] is True
    assert namespace["model"].parameters == component.parameters
    assert namespace["model"].limits == component.limits
    assert namespace["model"].sharing == component.sharing


def test_symmetry_generated_hopping_resolves_dispersion_and_script_round_trip():
    crystal = _crystal("P m -3 m")
    crystal["sites"] = crystal["sites"][:1]
    group = DataGroup("Electronic")
    component = create_model_component(group, "bands", type="tight_binding")
    component.config["crystal"] = crystal
    add_tight_binding_orbital_manifold(
        component,
        orbital_manifold_preset("M1", "effective"),
    )

    generation = regenerate_tight_binding_hopping_terms(component, 4.01)
    assert len(generation.orbits) == 1
    assert generation.orbits[0].label == "B1"
    assert generation.orbits[0].multiplicity == 3
    assert len(generation.terms) == 1
    term = generation.terms[0]
    np.testing.assert_allclose(term.matrix, [[1.0]])
    assert "M1_effective[effective]" in term.label
    assert hopping_endpoint_orbitals(term) == (
        ("M1_effective:effective",),
        ("M1_effective:effective",),
    )
    assert len(component.config["hopping_candidates"]) == 1
    assert component.config["hopping_terms"] == []
    flat = ElectronicModel.from_dict(component.config["model_data"])
    np.testing.assert_allclose(
        np.linalg.eigvalsh(flat.hamiltonian([0, 0, 0])),
        [0.0],
    )
    add_tight_binding_hopping_term(component, term.identifier)
    set_tight_binding_hopping_term(
        component,
        term.identifier,
        value=-0.1,
        lower=-0.2,
        upper=0.0,
        energy_unit="eV",
        fit=True,
    )
    assert component.parameters[term.identifier] == pytest.approx(-100.0)
    assert component.fit_parameters[term.identifier] is True
    assert component.limits[term.identifier] == [-200.0, 0.0]
    hopping_scene = model_geometry_scene(
        component,
        selected_hopping_term=term.identifier,
    )
    assert len(hopping_scene.pathways) == 1
    assert hopping_scene.pathways[0].orbit_label == "B1"

    model = ElectronicModel.from_dict(component.config["model_data"])
    assert model.parameter_values[term.identifier] == pytest.approx(-100.0)
    assert np.linalg.eigvalsh(model.hamiltonian([0, 0, 0]))[0] == pytest.approx(
        -600.0
    )
    assert np.linalg.eigvalsh(model.hamiltonian([0.5, 0, 0]))[0] == pytest.approx(
        -200.0
    )
    for vector, block in zip(
        model.translations,
        model.parameter_blocks[term.identifier],
        strict=True,
    ):
        partner_index = np.flatnonzero(
            np.all(model.translations == -vector, axis=1)
        )[0]
        np.testing.assert_allclose(
            block,
            model.parameter_blocks[term.identifier][partner_index].conj().T,
        )

    script = tight_binding_structure_script(
        component.config["crystal"],
        component.config["periodic_axes"] or [0, 1, 2],
        orbital_manifolds=component.config["orbital_manifolds"],
        onsite_terms=component.config["onsite_terms"],
        hopping_cutoff_angstrom=component.config[
            "hopping_cutoff_angstrom"
        ],
        hopping_terms=component.config["hopping_terms"],
        parameter_values_meV=component.parameters,
        fit_parameters=component.fit_parameters,
        parameter_limits_meV=component.limits,
        parameter_sharing=component.sharing,
        expected_model_digest=component.config["model_digest"],
    )
    namespace = {}
    exec(compile(script, "<hopping-builder-script>", "exec"), namespace)
    assert (
        namespace["electronic_model"].content_digest
        == component.config["model_digest"]
    )
    assert len(namespace["model"].config["hopping_candidates"]) == 1
    assert len(namespace["model"].config["hopping_terms"]) == 1

    remove_tight_binding_hopping_term(component, term.identifier)
    assert component.config["hopping_terms"] == []
    assert len(component.config["hopping_candidates"]) == 1
    assert term.identifier not in component.parameters
    assert term.identifier not in component.fit_parameters
    assert term.identifier not in component.limits
    assert term.identifier not in component.sharing
    add_tight_binding_hopping_term(component, term.identifier)
    restored = component.config["hopping_terms"][0]
    assert restored["value_meV"] == pytest.approx(-100.0)
    assert restored["fit"] is True


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


def test_model_geometry_picking_reports_atoms_orbitals_and_bonds(monkeypatch):
    from nfit.qt_model_geometry_viewer import (
        _actor_key,
        _enable_geometry_picking,
        _PickBatch,
        _selection_from_pick,
        _selection_panel,
    )

    component = ModelComponentSpec(
        name="electrons",
        type="tight_binding",
        config={
            "crystal": _crystal(),
            "orbital_manifolds": [
                orbital_manifold_preset("M1", "p").to_dict()
            ],
            "site_positions": [[0.0, 0.0, 0.0]],
            "spatial_orbits": [
                {
                    "label": "B1",
                    "distance_angstrom": 4.0,
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}
                    ],
                }
            ],
        },
    )
    scene = model_geometry_scene(component)

    class Actor:
        def __init__(self, name):
            self.name = name

        def GetAddressAsString(self, _prefix):
            return self.name

    site_actor = Actor("site")
    orbital_actor = Actor("orbital")
    pathway_actor = Actor("pathway")
    registry = {
        _actor_key(site_actor): _PickBatch("site", (scene.sites[0],)),
        _actor_key(orbital_actor): _PickBatch(
            "orbital",
            (scene.orbitals[0],),
            radius=0.1,
        ),
        _actor_key(pathway_actor): _PickBatch(
            "pathway",
            (scene.pathways[0],),
        ),
    }
    site_text = _selection_from_pick(
        registry,
        site_actor,
        np.asarray(scene.sites[0].cartesian),
    )
    orbital_text = _selection_from_pick(
        registry,
        orbital_actor,
        np.asarray(scene.orbitals[0].display_center_cartesian),
    )
    pathway = scene.pathways[0]
    midpoint = 0.5 * (
        np.asarray(pathway.start_cartesian)
        + np.asarray(pathway.end_cartesian)
    )
    pathway_text = _selection_from_pick(
        registry,
        pathway_actor,
        midpoint,
    )
    assert "Atom: M1_1" in site_text
    assert "Representative site: M1" in site_text
    assert "Orbital: p_x" in orbital_text
    assert "Manifold: M1_p" in orbital_text
    assert "Hopping bond: B1" in pathway_text
    assert "representative" in pathway_text

    class Picker:
        def GetActor(self):
            return site_actor

    class Plotter:
        def enable_point_picking(self, **kwargs):
            self.options = kwargs

    updates = []
    plotter = Plotter()
    _enable_geometry_picking(plotter, lambda: registry, updates.append)
    assert plotter.options["left_clicking"] is True
    assert plotter.options["picker"] == "cell"
    assert plotter.options["use_picker"] is True
    plotter.options["callback"](
        np.asarray(scene.sites[0].cartesian),
        Picker(),
    )
    assert "Atom: M1_1" in updates[-1]

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        []
    )
    panel, label = _selection_panel(QtWidgets)
    assert panel.objectName() == "model_geometry_selection_group"
    assert label.objectName() == "model_geometry_selection"
    assert "Click an atom" in label.text()
    assert label.toolTip()
    panel.deleteLater()
    assert application is not None


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
