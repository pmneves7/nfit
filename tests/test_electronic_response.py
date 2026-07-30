from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nfit import (
    BasisState,
    DataGroup,
    ElectronicOperatorBasis,
    FitDatasetInput,
    ModelComponentSpec,
    SusceptibilityResult,
    add_tight_binding_orbital_manifold,
    bare_spin_susceptibility,
    build_electronic_model,
    chemical_potential_for_filling,
    compile_fit_problem,
    create_model_component,
    electron_filling,
    isotropic_spin_component,
    k_mesh,
    lift_electronic_model_spin,
    model_definition,
    neutron_spin_contraction,
    orbital_manifold_preset,
    orbital_pair_operator_basis,
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
