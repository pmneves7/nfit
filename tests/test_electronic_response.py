from __future__ import annotations

import warnings
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nfit import (
    BasisState,
    DataGroup,
    ElectronicOperatorBasis,
    ElectronicResponseCache,
    FitDatasetInput,
    ModelComponentSpec,
    ResponseConvergenceResult,
    SusceptibilityResult,
    add_tight_binding_orbital_manifold,
    bare_lindhard_susceptibility,
    bare_spin_susceptibility,
    build_electronic_model,
    chemical_potential_for_filling,
    compile_fit_problem,
    correlated_basis_indices,
    create_model_component,
    electron_filling,
    hubbard_hund_spin_vertex,
    isotropic_spin_component,
    k_mesh,
    lift_electronic_model_spin,
    model_definition,
    neutron_spin_contraction,
    orbital_manifold_preset,
    orbital_pair_operator_basis,
    project_implicit_spin_response,
    response_convergence_scan,
    response_k_mesh,
    rpa_dress_susceptibility,
    scalar_stoner_vertex,
)
from nfit.cross_section import KB_MEV_PER_K
from nfit.dataset import PointData4D
from nfit.fitting import evaluate_problem_model


def _chain_model():
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-10.0]]},
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )


def test_operator_basis_and_response_round_trip_preserve_complex_values():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["a", "b"],
        hoppings={(0, 0, 0): np.diag([-1.0, 1.0])},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    basis = orbital_pair_operator_basis(model)

    assert basis.labels == ("a←a", "a←b", "b←a", "b←b")
    assert basis.conjugate_indices == (0, 2, 1, 3)
    restored_basis = ElectronicOperatorBasis.from_dict(basis.to_dict())
    np.testing.assert_array_equal(restored_basis.matrices, basis.matrices)

    response = SusceptibilityResult(
        q_reduced=[[0.25, 0.0, 0.0]],
        Q_reduced=[[1.25, 0.0, 0.0]],
        energy_meV=[2.0],
        values_per_meV_cell=np.asarray([[[1.0 + 2.0j]]]),
        operator_labels=("density",),
        conjugate_indices=(0,),
        model_digest=model.content_digest,
        temperature_K=10.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.5,
        provenance={"test": True},
    )
    restored = SusceptibilityResult.from_dict(response.to_dict())

    np.testing.assert_array_equal(restored.q_reduced, response.q_reduced)
    np.testing.assert_array_equal(restored.Q_reduced, response.Q_reduced)
    np.testing.assert_array_equal(
        restored.values_per_meV_cell,
        response.values_per_meV_cell,
    )
    assert restored.provenance == response.provenance


def test_scalar_rpa_matches_closed_form_and_records_pole_diagnostic():
    chi0 = 0.002 + 0.0004j
    bare = SusceptibilityResult(
        q_reduced=[[0.0, 0.0, 0.0]],
        energy_meV=[1.0],
        values_per_meV_cell=[[[chi0]]],
        operator_labels=("spin",),
        conjugate_indices=(0,),
        model_digest="test",
        temperature_K=10.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.5,
    )
    vertex = scalar_stoner_vertex(("spin",), 0.2, energy_unit="eV")
    dressed = rpa_dress_susceptibility(bare, vertex)

    np.testing.assert_allclose(
        dressed.values_per_meV_cell[0, 0, 0],
        chi0 / (1.0 - 200.0 * chi0),
    )
    assert dressed.response_kind == "rpa_scalar_stoner"
    assert dressed.provenance["dressing"]["multiplication_order"] == "I-chi0@Gamma"


def test_hubbard_hund_vertex_respects_shell_locality_and_rotational_constraints():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=[
            BasisState("A_xz", site="A", orbital="d_xz", correlated_shell="A_d"),
            BasisState("A_yz", site="A", orbital="d_yz", correlated_shell="A_d"),
            BasisState("B_xz", site="B", orbital="d_xz", correlated_shell="B_d"),
        ],
        hoppings={(0, 0, 0): np.diag([-2.0, 0.0, 2.0])},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    indices = correlated_basis_indices(model)
    vertex = hubbard_hund_spin_vertex(
        model,
        indices,
        U=2.0,
        J_H=0.25,
        rotationally_invariant=True,
    )
    pair = {
        tuple(label.split("←")): index
        for index, label in enumerate(vertex.operator_labels)
    }

    assert vertex.values_meV[pair[("A_xz", "A_xz")], pair[("A_xz", "A_xz")]] == 2000.0
    assert vertex.values_meV[pair[("A_xz", "A_yz")], pair[("A_xz", "A_yz")]] == 1500.0
    assert vertex.values_meV[pair[("A_xz", "A_xz")], pair[("A_yz", "A_yz")]] == 250.0
    assert vertex.values_meV[pair[("A_xz", "A_yz")], pair[("A_yz", "A_xz")]] == 250.0
    assert vertex.values_meV[pair[("A_xz", "A_xz")], pair[("B_xz", "B_xz")]] == 0.0


def test_orbital_pair_subspace_projects_to_implicit_spin_response():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=[
            BasisState("a", site="A", orbital="a", correlated_shell="A_d"),
            BasisState("spectator", site="B", orbital="s"),
        ],
        hoppings={(0, 0, 0): np.diag([0.0, 100.0])},
        orbital_centers=[[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )
    indices = correlated_basis_indices(model)
    operators = orbital_pair_operator_basis(model, indices)
    pair_response = bare_lindhard_susceptibility(
        model,
        [1.0, 0.0, 0.0],
        0.0,
        k_mesh(model, (4,)),
        operators,
        temperature_K=20.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.2,
    )
    projected = project_implicit_spin_response(pair_response, model, indices)

    expected = 0.5 * 0.25 / (KB_MEV_PER_K * 20.0)
    np.testing.assert_allclose(isotropic_spin_component(projected), expected)
    assert projected.operator_labels == ("Sx", "Sy", "Sz")


def test_lindhard_response_is_causal_and_retains_extended_zone_transfer():
    model = _chain_model()
    mesh = k_mesh(model, (96,))
    response = bare_spin_susceptibility(
        model,
        [[1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        [1.0, -1.0],
        mesh,
        temperature_K=20.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.4,
    )
    component = isotropic_spin_component(response)

    np.testing.assert_allclose(response.q_reduced[:, 0], 0.5)
    np.testing.assert_allclose(response.Q_reduced[:, 0], 1.5)
    assert component[0].imag > 0.0
    np.testing.assert_allclose(component[1], component[0].conjugate())


def test_response_cache_reuses_eigensystems_and_transition_chunks_are_equivalent():
    model = _chain_model()
    mesh = k_mesh(model, (48,))
    Q = np.asarray([[0.35, 0.0, 0.0], [0.35, 0.0, 0.0]])
    energy = np.asarray([1.0, 2.0])
    cache = ElectronicResponseCache(max_bytes=32 * 1024**2, max_entries=8)
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
        "cache": cache,
    }
    first = bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        transition_max_batch_bytes=256 * 1024**2,
        **settings,
    )
    second = bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        transition_max_batch_bytes=128,
        **settings,
    )

    assert first.provenance["cache"]["misses"] >= 2
    assert second.provenance["cache"]["hits"] >= 2
    assert second.provenance["transition_batch_size"] == 1
    np.testing.assert_allclose(
        second.values_per_meV_cell,
        first.values_per_meV_cell,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_response_cache_is_bounded_and_model_digest_invalidates_entries():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[0.0]]},
        parameter_hoppings={"epsilon": {(0, 0, 0): [[1.0]]}},
        parameter_values={"epsilon": 0.0},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    mesh = k_mesh(model, (4,))
    cache = ElectronicResponseCache(max_bytes=1, max_entries=8)
    kwargs = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.2,
        "cache": cache,
    }
    first = bare_spin_susceptibility(model, [0, 0, 0], 0.0, mesh, **kwargs)
    changed = model.with_parameters(epsilon=1.0, energy_unit="meV")
    second = bare_spin_susceptibility(changed, [0, 0, 0], 0.0, mesh, **kwargs)

    assert cache.entries == 0
    assert first.model_digest != second.model_digest
    assert cache.misses >= 4


def test_response_convergence_separates_mesh_and_broadening_axes():
    model = _chain_model()
    result = response_convergence_scan(
        model,
        [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]],
        [0.5, 1.0],
        mesh_shapes=[(12,), (24,)],
        broadenings_meV=[0.8, 0.4],
        temperature_K=20.0,
        chemical_potential_meV=0.0,
        transition_max_batch_bytes=1024,
    )

    assert result.values_per_meV_cell.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        result.mesh_max_absolute_error[result.reference_mesh_index],
        0.0,
    )
    np.testing.assert_array_equal(
        result.broadening_max_absolute_change[
            :, result.reference_broadening_index
        ],
        0.0,
    )
    restored = ResponseConvergenceResult.from_dict(result.to_dict())
    np.testing.assert_array_equal(
        restored.values_per_meV_cell,
        result.values_per_meV_cell,
    )
    assert "each mesh versus reference mesh" in str(
        result.provenance["comparison"]["mesh"]
    )


def test_response_mesh_reduction_is_little_group_certified_and_fail_closed():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={
            (1, 0, 0): [[-1.0]],
            (0, 1, 0): [[-1.0]],
            (0, 0, 1): [[-1.0]],
        },
        energy_unit="meV",
    )
    rotations = [
        np.diag(signs).astype(int).tolist()
        for signs in product((-1, 1), repeat=3)
    ]
    model = replace(
        model,
        provenance={
            **dict(model.provenance),
            "implicit_spin_degeneracy": 2,
            "reciprocal_symmetry": {
                "certified_by": "nfit_orbital_builder",
                "rotations": rotations,
                "includes_time_reversal": True,
            },
        },
    )
    full = response_k_mesh(model, (8, 8, 8), [0, 0, 0], symmetry="full")
    reduced = response_k_mesh(
        model,
        (8, 8, 8),
        [0, 0, 0],
        symmetry="auto",
    )

    assert reduced.reduced_coordinates.shape[0] < full.reduced_coordinates.shape[0]
    assert reduced.provenance["response_symmetry_reduction"]["applied"]
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.2,
    }
    reference = bare_spin_susceptibility(model, [0, 0, 0], 0.0, full, **settings)
    accelerated = bare_spin_susceptibility(
        model,
        [0, 0, 0],
        0.0,
        reduced,
        **settings,
    )
    assert not reference.provenance["symmetry"]["applied"]
    assert accelerated.provenance["symmetry"]["applied"]
    np.testing.assert_allclose(
        accelerated.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    fallback = response_k_mesh(
        model,
        (8, 8, 8),
        [0.137, 0.271, 0.389],
        symmetry="auto",
    )
    assert not fallback.provenance["response_symmetry_reduction"]["applied"]
    with pytest.raises(ValueError, match="little group"):
        response_k_mesh(
            model,
            (8, 8, 8),
            [0.137, 0.271, 0.389],
            symmetry="reduced",
        )


def test_uniform_static_limit_matches_fermi_derivative():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["flat"],
        hoppings={(0, 0, 0): [[0.0]]},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    temperature = 25.0
    response = bare_spin_susceptibility(
        model,
        [0.0, 0.0, 0.0],
        0.0,
        k_mesh(model, (4,)),
        temperature_K=temperature,
        chemical_potential_meV=0.0,
        broadening_meV=0.2,
    )

    expected = 0.5 * 0.25 / (KB_MEV_PER_K * temperature)
    np.testing.assert_allclose(
        isotropic_spin_component(response),
        expected,
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(response.chi_double_prime, 0.0, atol=1.0e-15)


def test_implicit_and_explicit_collinear_spin_responses_agree():
    orbital_model = _chain_model()
    implicit = lift_electronic_model_spin(orbital_model, treatment="implicit")
    collinear = lift_electronic_model_spin(orbital_model, treatment="collinear")
    Q = np.asarray([[0.35, 0.0, 0.0], [0.35, 0.0, 0.0]])
    energy = np.asarray([1.0, 3.0])
    settings = {
        "temperature_K": 15.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.3,
    }

    implicit_response = bare_spin_susceptibility(
        implicit,
        Q,
        energy,
        k_mesh(implicit, (72,)),
        **settings,
    )
    collinear_response = bare_spin_susceptibility(
        collinear,
        Q,
        energy,
        k_mesh(collinear, (72,)),
        **settings,
    )

    np.testing.assert_allclose(
        implicit_response.values_per_meV_cell,
        collinear_response.values_per_meV_cell,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_filling_solver_and_neutron_projection_use_public_normalization():
    model = _chain_model()
    mesh = k_mesh(model, (128,))
    target = 0.37
    mu = chemical_potential_for_filling(
        model,
        mesh,
        target,
        temperature_K=30.0,
    )
    assert electron_filling(
        model,
        mesh,
        chemical_potential_meV=mu,
        temperature_K=30.0,
    ) == pytest.approx(target, rel=1.0e-11)

    value = 3.0 + 0.5j
    response = SusceptibilityResult(
        q_reduced=[[0.25, 0.0, 0.0]],
        energy_meV=[2.0],
        values_per_meV_cell=np.asarray([np.eye(3) * value]),
        operator_labels=("Sx", "Sy", "Sz"),
        conjugate_indices=(0, 1, 2),
        model_digest=model.content_digest,
        temperature_K=30.0,
        chemical_potential_meV=mu,
        broadening_meV=0.5,
    )
    contracted = neutron_spin_contraction(response, model.reciprocal_lattice)
    np.testing.assert_allclose(contracted, 2.0 * value)


def test_lindhard_registry_plot_is_linked_and_scriptable():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [32],
            "response_mesh_shift": [0.0],
            "plot_q_reduced": [0.5, 0.0, 0.0],
            "plot_energy_min_meV": -2.0,
            "plot_energy_max_meV": 2.0,
            "plot_energy_points": 9,
        },
    )
    plot = model_definition("lindhard").plots[0]
    components = {"bands": tight_binding, "response": lindhard}

    with pytest.raises(ValueError, match="referenced tight-binding"):
        plot.calculate(lindhard)
    result = plot.context_calculate(lindhard, components)
    assert result.values_per_meV_cell.shape == (9, 3, 3)
    figure, axes = plot.render(result)
    assert "Bare spin response" in axes[0].get_title()
    plt.close(figure)

    script = plot.context_script(lindhard, components)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        exec(compile(script, "<lindhard-plot>", "exec"), namespace)
    scripted = namespace["result"]
    np.testing.assert_allclose(
        scripted.values_per_meV_cell,
        result.values_per_meV_cell,
    )
    plt.close(namespace["figure"])


def test_lindhard_convergence_plot_is_scriptable():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [12],
            "response_mesh_shift": [0.0],
            "plot_q_reduced": [0.3, 0.0, 0.0],
            "plot_energy_min_meV": -1.0,
            "plot_energy_max_meV": 1.0,
            "convergence_energy_points": 3,
            "convergence_mesh_scales": [0.5, 1.0],
            "convergence_broadening_scales": [2.0, 1.0],
        },
    )
    components = {"bands": tight_binding, "response": lindhard}
    plot = model_definition("lindhard").plots[1]
    result = plot.context_calculate(lindhard, components)
    assert result.values_per_meV_cell.shape == (2, 2, 3)
    figure, _axes = plot.render(result)
    plt.close(figure)

    script = plot.context_script(lindhard, components)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        exec(compile(script, "<lindhard-convergence>", "exec"), namespace)
    np.testing.assert_allclose(
        namespace["result"].values_per_meV_cell,
        result.values_per_meV_cell,
    )
    plt.close(namespace["figure"])


def test_fit_compiler_uses_electronic_component_as_dependency_not_observable():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        fit_parameters={"broadening": True},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [24],
            "response_mesh_shift": [0.0],
        },
    )
    points = PointData4D(
        H=np.full(3, 0.5),
        K=np.zeros(3),
        L=np.zeros(3),
        E=np.asarray([1.0, 2.0, 3.0]),
        intensity=np.zeros(3),
        sigma=np.ones(3),
        temperature=20.0,
    )
    compiled = compile_fit_problem(
        [tight_binding, lindhard],
        [
            FitDatasetInput(
                "scan",
                points,
                data_type="single_crystal_inelastic",
            )
        ],
    )

    assert compiled.components_by_dataset == {"scan": ["response"]}
    assert [spec.name for spec in compiled.problem.parameter_specs] == [
        "response.broadening"
    ]
    predicted = evaluate_problem_model(
        compiled.problem,
        "scan",
        {"response.broadening": 0.5},
    )
    assert predicted.shape == (3,)
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted >= 0.0)

    lindhard.config["electronic_component"] = "missing"
    with pytest.raises(ValueError, match="missing or disabled"):
        compile_fit_problem(
            [tight_binding, lindhard],
            [
                FitDatasetInput(
                    "scan",
                    points,
                    data_type="single_crystal_inelastic",
                )
            ],
        )


def test_stoner_component_replaces_bare_observable_and_reuses_its_parameters():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="bare",
        type="lindhard",
        parameters={"broadening": 0.5},
        fit_parameters={"broadening": True},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [24],
            "response_mesh_shift": [0.0],
        },
    )
    stoner = ModelComponentSpec(
        name="dressed",
        type="stoner_rpa",
        parameters={"I": 0.1},
        fit_parameters={"I": True},
        config={
            **{
                field.name: field.default
                for field in model_definition("stoner_rpa").config_fields
            },
            "response_component": "bare",
        },
    )
    points = PointData4D(
        H=np.full(2, 0.5),
        K=np.zeros(2),
        L=np.zeros(2),
        E=np.asarray([1.0, 2.0]),
        intensity=np.zeros(2),
        sigma=np.ones(2),
        temperature=20.0,
    )
    dataset = FitDatasetInput(
        "scan",
        points,
        data_type="single_crystal_inelastic",
    )
    compiled = compile_fit_problem([tight_binding, lindhard, stoner], [dataset])

    assert compiled.components_by_dataset == {"scan": ["dressed"]}
    assert {spec.name for spec in compiled.problem.parameter_specs} == {
        "bare.broadening",
        "dressed.I",
    }
    prediction = evaluate_problem_model(
        compiled.problem,
        "scan",
        {"bare.broadening": 0.5, "dressed.I": 0.1},
    )
    assert prediction.shape == (2,)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0.0)


def test_dressed_response_shares_parameters_across_multiple_datasets():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="bare",
        type="lindhard",
        parameters={"broadening": 0.5},
        fit_parameters={"broadening": True},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [16],
            "response_mesh_shift": [0.0],
        },
    )
    stoner = ModelComponentSpec(
        name="dressed",
        type="stoner_rpa",
        parameters={"I": 0.1},
        fit_parameters={"I": True},
        config={
            **{
                field.name: field.default
                for field in model_definition("stoner_rpa").config_fields
            },
            "response_component": "bare",
        },
    )

    def points(q):
        return PointData4D(
            H=np.full(2, q),
            K=np.zeros(2),
            L=np.zeros(2),
            E=np.asarray([1.0, 2.0]),
            intensity=np.zeros(2),
            sigma=np.ones(2),
            temperature=20.0,
        )

    compiled = compile_fit_problem(
        [tight_binding, lindhard, stoner],
        [
            FitDatasetInput(
                "scan_a",
                points(0.25),
                data_type="single_crystal_inelastic",
            ),
            FitDatasetInput(
                "scan_b",
                points(0.5),
                data_type="single_crystal_inelastic",
            ),
        ],
    )

    assert compiled.components_by_dataset == {
        "scan_a": ["dressed"],
        "scan_b": ["dressed"],
    }
    assert compiled.instances_for("dressed", "I")[0].datasets == (
        "scan_a",
        "scan_b",
    )
    params = {"bare.broadening": 0.5, "dressed.I": 0.1}
    first = evaluate_problem_model(compiled.problem, "scan_a", params)
    second = evaluate_problem_model(compiled.problem, "scan_b", params)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    assert not np.allclose(first, second)


def test_stoner_model_plot_and_exported_script_are_equivalent():
    model = _chain_model()
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="bare",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [24],
            "response_mesh_shift": [0.0],
            "plot_q_reduced": [0.5, 0.0, 0.0],
            "plot_energy_min_meV": -2.0,
            "plot_energy_max_meV": 2.0,
            "plot_energy_points": 5,
        },
    )
    stoner = ModelComponentSpec(
        name="dressed",
        type="stoner_rpa",
        parameters={"I": 0.1},
        config={
            **{
                field.name: field.default
                for field in model_definition("stoner_rpa").config_fields
            },
            "response_component": "bare",
        },
    )
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    plot = model_definition("stoner_rpa").plots[0]
    result = plot.context_calculate(stoner, components)
    assert result.response_kind == "rpa_scalar_stoner"

    script = plot.context_script(stoner, components)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        exec(compile(script, "<stoner-rpa-plot>", "exec"), namespace)
    np.testing.assert_allclose(
        namespace["result"].values_per_meV_cell,
        result.values_per_meV_cell,
    )
    plt.close(namespace["figure"])


def test_linked_tight_binding_parameters_inherit_response_dataset_scope():
    group = DataGroup("Electronic")
    tight_binding = create_model_component(
        group,
        "bands",
        type="tight_binding",
    )
    tight_binding.config["crystal"]["sites"].append(
        {
            "label": "M1",
            "element": "M",
            "position": [0.0, 0.0, 0.0],
            "ion": "",
        }
    )
    add_tight_binding_orbital_manifold(
        tight_binding,
        orbital_manifold_preset("M1", "s"),
    )
    parameter_name = next(iter(tight_binding.parameters))
    tight_binding.fit_parameters[parameter_name] = True
    tight_binding.config["onsite_terms"][0]["fit"] = True

    lindhard = create_model_component(group, "response", type="lindhard")
    lindhard.config["electronic_component"] = tight_binding.name
    lindhard.config["response_mesh"] = [2, 2, 2]
    points = PointData4D(
        H=np.asarray([0.0]),
        K=np.asarray([0.0]),
        L=np.asarray([0.0]),
        E=np.asarray([1.0]),
        intensity=np.asarray([0.0]),
        sigma=np.asarray([1.0]),
        temperature=20.0,
    )

    compiled = compile_fit_problem(
        [tight_binding, lindhard],
        [
            FitDatasetInput(
                "scan",
                points,
                data_type="single_crystal_inelastic",
            )
        ],
    )
    spec_names = {spec.name for spec in compiled.problem.parameter_specs}

    assert f"bands.{parameter_name}" in spec_names
    assert compiled.instances_for("bands", parameter_name)[0].datasets == (
        "scan",
    )
