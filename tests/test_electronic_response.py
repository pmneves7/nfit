from __future__ import annotations

import warnings
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from matplotlib import pyplot as plt

import nfit.electronic_response as electronic_response_module
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
    configure_lindhard_plot,
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
from nfit.electronic_interactions import rpa_dress_implicit_spin_probe_stoner
from nfit.electronic_response import implicit_spin_probe_susceptibility
from nfit.fitting import evaluate_problem_model
from nfit.form_factors import magnetic_form_factor


def _chain_model():
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-10.0]]},
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )


def _profiled_chain_model():
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[
            BasisState(
                "s",
                site="A",
                orbital="s",
                metadata={
                    "magnetic_form_factor": {
                        "form_factor_mode": "single_ion",
                        "ion": "V3",
                        "form_factor_g_J": 2.0,
                    }
                },
            )
        ],
        hoppings={(1, 0, 0): [[-10.0]]},
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )


def test_orbital_form_factor_weights_probe_but_not_stoner_denominator():
    unit = _chain_model()
    profiled = _profiled_chain_model()
    mesh = k_mesh(unit, (64,))
    Q = np.asarray([[2.375, 0.0, 0.0]])
    energy = np.asarray([2.0])
    settings = {
        "temperature_K": 30.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 2.0,
        "q_evaluation": "direct",
    }
    intrinsic = bare_spin_susceptibility(unit, Q, energy, mesh, **settings)
    mixed = implicit_spin_probe_susceptibility(
        profiled, Q, energy, mesh, **settings
    )
    q_modulus = np.linalg.norm(Q @ profiled.reciprocal_lattice.T, axis=1)
    amplitude = magnetic_form_factor(q_modulus, ion="V3")

    np.testing.assert_allclose(
        mixed.values_per_meV_cell[:, 0, 0],
        amplitude**2 * intrinsic.values_per_meV_cell[:, 0, 0],
    )
    np.testing.assert_allclose(
        mixed.values_per_meV_cell[:, 1, 1],
        intrinsic.values_per_meV_cell[:, 0, 0],
    )
    interaction_eV = 0.1
    from nfit import matrix_interaction_vertex

    vertex = matrix_interaction_vertex(
        mixed.operator_labels,
        ((0.0, 0.0), (0.0, 2.0 * interaction_eV)),
        energy_unit="eV",
        channel="spin",
    )
    dressed = rpa_dress_susceptibility(mixed, vertex)
    optimized = rpa_dress_implicit_spin_probe_stoner(
        mixed, interaction_eV, energy_unit="eV"
    )
    intrinsic_value = intrinsic.values_per_meV_cell[:, 0, 0]
    expected = amplitude**2 * intrinsic_value / (
        1.0 - 200.0 * intrinsic_value
    )
    np.testing.assert_allclose(dressed.values_per_meV_cell[:, 0, 0], expected)
    np.testing.assert_allclose(
        optimized.values_per_meV_cell,
        dressed.values_per_meV_cell,
    )
    assert optimized.provenance["dressing"]["solver"] == (
        "rank_one_probe_stoner"
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


def test_implicit_cartesian_stoner_uses_compact_scalar_solver():
    chi0 = np.asarray([0.001 + 0.0002j, 0.002 + 0.0003j])
    values = np.zeros((2, 3, 3), dtype=np.complex128)
    diagonal = np.arange(3)
    values[:, diagonal, diagonal] = chi0[:, None]
    bare = SusceptibilityResult(
        q_reduced=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        energy_meV=[0.0, 1.0],
        values_per_meV_cell=values,
        operator_labels=("Sx", "Sy", "Sz"),
        conjugate_indices=(0, 1, 2),
        model_digest="test",
        temperature_K=10.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.5,
        provenance={
            "operator_basis": {
                "kind": "cartesian_spin",
                "implicit_spin_trace_factor": 0.5,
            }
        },
    )
    vertex = scalar_stoner_vertex(
        ("Sx", "Sy", "Sz"),
        0.1,
        energy_unit="eV",
    )

    dressed = rpa_dress_susceptibility(bare, vertex)

    expected = chi0 / (1.0 - 200.0 * chi0)
    np.testing.assert_allclose(
        dressed.values_per_meV_cell[:, diagonal, diagonal],
        np.repeat(expected[:, None], 3, axis=1),
    )
    diagnostics = dressed.provenance["dressing"]
    assert diagnostics["solver"] == "scalar_isotropic"
    assert isinstance(diagnostics["minimum_singular_value"], float)
    assert "condition_number" not in diagnostics
    assert diagnostics["static_stability"][
        "sampled_zero_energy_point_count"
    ] == 1


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


def test_response_cache_sizes_each_insert_once(monkeypatch):
    calls = []

    def payload_size(value):
        calls.append(value)
        return int(np.asarray(value).nbytes)

    monkeypatch.setattr(
        electronic_response_module,
        "array_payload_nbytes",
        payload_size,
    )
    cache = ElectronicResponseCache(max_bytes=24, max_entries=2)
    values = [np.arange(2, dtype=float) + index for index in range(4)]
    for index, value in enumerate(values):
        cache.put(("entry", index), value)

    assert len(calls) == len(values)
    assert cache.entries == 1
    assert cache.host_bytes == values[-1].nbytes
    assert cache.get(("entry", 3)) is values[-1]
    assert len(calls) == len(values)


def test_numba_transition_contraction_matches_numpy():
    pytest.importorskip("numba")
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=[
            BasisState("a", site="A", orbital="a"),
            BasisState("b", site="A", orbital="b"),
        ],
        hoppings={
            (0, 0, 0): [[-2.0, 0.4j], [-0.4j, 3.0]],
            (1, 0, 0): [[-6.0, 1.0], [0.5j, 4.0]],
        },
        periodic_axes=(0,),
        energy_unit="meV",
    )
    mesh = k_mesh(model, (48,))
    operators = orbital_pair_operator_basis(model)
    q = np.tile([0.173, 0.0, 0.0], (3, 1))
    energy = np.asarray([-2.0, 0.0, 3.0])
    settings = {
        "temperature_K": 25.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.5,
        "transition_max_batch_bytes": 32 * 1024**2,
    }
    numpy_result = bare_lindhard_susceptibility(
        model,
        q,
        energy,
        mesh,
        operators,
        transition_backend="numpy",
        **settings,
    )
    numba_result = bare_lindhard_susceptibility(
        model,
        q,
        energy,
        mesh,
        operators,
        transition_backend="numba",
        workers=2,
        **settings,
    )
    factorized_result = bare_lindhard_susceptibility(
        model,
        q,
        energy,
        mesh,
        operators,
        transition_backend="auto",
        workers=2,
        **settings,
    )
    factorized_chunked = bare_lindhard_susceptibility(
        model,
        q,
        energy,
        mesh,
        operators,
        transition_backend="auto",
        transition_max_batch_bytes=128,
        workers=2,
        **{
            key: value
            for key, value in settings.items()
            if key != "transition_max_batch_bytes"
        },
    )

    assert numba_result.provenance["transition_backend"] == "numba"
    assert (
        factorized_result.provenance["transition_backend"]
        == "numpy_orbital_pair_factorized"
    )
    np.testing.assert_allclose(
        numba_result.values_per_meV_cell,
        numpy_result.values_per_meV_cell,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        factorized_result.values_per_meV_cell,
        numpy_result.values_per_meV_cell,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        factorized_chunked.values_per_meV_cell,
        numpy_result.values_per_meV_cell,
        rtol=2.0e-13,
        atol=2.0e-14,
    )

    point_operators = np.broadcast_to(
        operators.matrices[None, ...],
        (q.shape[0], *operators.matrices.shape),
    )
    explicit_operators = bare_lindhard_susceptibility(
        model,
        q,
        energy,
        mesh,
        operators,
        operator_matrices_by_point=point_operators,
        transition_backend="auto",
        workers=2,
        **settings,
    )
    assert (
        explicit_operators.provenance["transition_backend"]
        != "numpy_orbital_pair_factorized"
    )
    np.testing.assert_allclose(
        explicit_operators.values_per_meV_cell,
        numpy_result.values_per_meV_cell,
        rtol=2.0e-13,
        atol=2.0e-14,
    )


def test_commensurate_q_reuses_periodic_mesh_eigensystem_exactly():
    model = _chain_model()
    mesh = k_mesh(model, (32,), shift=(0.5,))
    reference_mesh = replace(mesh, provenance={"provider": "reference"})
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
    }
    accelerated = bare_spin_susceptibility(
        model,
        [5.0 / 32.0, 0.0, 0.0],
        1.5,
        mesh,
        **settings,
    )
    reference = bare_spin_susceptibility(
        model,
        [5.0 / 32.0, 0.0, 0.0],
        1.5,
        reference_mesh,
        **settings,
    )

    np.testing.assert_allclose(
        accelerated.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert (
        accelerated.provenance["q_evaluation"][
            "commensurate_permutation_count"
        ]
        == 1
    )


def test_independent_q_groups_parallelize_without_changing_response(
    monkeypatch,
):
    monkeypatch.setattr(
        electronic_response_module,
        "_Q_PARALLEL_MIN_WORK",
        0,
    )
    model = _chain_model()
    mesh = k_mesh(model, (64,))
    q = np.asarray(
        [
            [0.13, 0.0, 0.0],
            [0.27, 0.0, 0.0],
            [0.41, 0.0, 0.0],
            [0.49, 0.0, 0.0],
        ]
    )
    energy = np.asarray([-2.0, -0.5, 1.0, 2.5])
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
        "backend": "numpy",
    }
    serial = bare_spin_susceptibility(
        model,
        q,
        energy,
        mesh,
        workers=1,
        **settings,
    )
    parallel = bare_spin_susceptibility(
        model,
        q,
        energy,
        mesh,
        workers=4,
        **settings,
    )

    assert serial.provenance["q_parallel_execution"]["applied"] is False
    assert parallel.provenance["q_parallel_execution"] == {
        "applied": True,
        "q_workers": 4,
        "inner_workers": 1,
        "unique_q": 4,
        "q_batch_size": 1,
        "q_batches": 4,
        "work_estimate": 256,
    }
    np.testing.assert_array_equal(
        parallel.values_per_meV_cell,
        serial.values_per_meV_cell,
    )


def test_inverse_indices_are_grouped_without_repeated_full_array_scans():
    inverse = np.asarray([2, 0, 1, 2, 0, 1, 1], dtype=np.intp)

    groups = electronic_response_module._group_inverse_indices(inverse)

    assert [group.tolist() for group in groups] == [
        [1, 4],
        [2, 5, 6],
        [0, 3],
    ]


def test_auto_q_interpolation_validates_or_falls_back_to_exact_response():
    model = _chain_model()
    mesh = k_mesh(model, (32,))
    settings = {
        "temperature_K": 30.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 2.0,
    }
    Q = np.asarray([[0.137, 0.0, 0.0], [0.283, 0.0, 0.0]])
    energy = np.asarray([1.0, 2.0])
    reference = bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        q_evaluation="direct",
        **settings,
    )
    automatic = bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        q_evaluation="auto",
        q_interpolation_rtol=0.2,
        q_interpolation_atol=1.0e-8,
        q_interpolation_mesh=(8,),
        q_validation_points=2,
        **settings,
    )
    policy = automatic.provenance["q_evaluation"]
    certificate = automatic.provenance["q_interpolation_certificate"]

    assert policy["resolved_policy"] == "validated_interpolation"
    assert certificate["status"] == "certified"
    assert certificate["certificate_digest"]
    assert certificate["maximum_relative_error"] <= 0.2
    np.testing.assert_allclose(
        automatic.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=0.2,
        atol=1.0e-8,
    )
    strict = bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        q_evaluation="auto",
        q_interpolation_rtol=1.0e-14,
        q_interpolation_atol=0.0,
        q_interpolation_mesh=(8,),
        q_validation_points=2,
        **settings,
    )
    assert strict.provenance["q_evaluation"]["resolved_policy"] == "exact_fallback"
    assert strict.provenance["q_interpolation_certificate"]["status"] == (
        "exact_fallback"
    )
    np.testing.assert_allclose(
        strict.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_interpolation_geometry_cache_is_reused_across_model_parameters():
    model = _chain_model()
    varied = build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-9.9]]},
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )
    mesh = k_mesh(model, (32,))
    cache = ElectronicResponseCache(max_bytes=1024**2, max_entries=16)
    q = np.linspace(0.101, 0.899, 100)
    Q = np.column_stack((q, np.zeros(100), np.zeros(100)))
    energy = np.resize(np.linspace(-2.0, 2.0, 5), 100)
    settings = {
        "temperature_K": 30.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 2.0,
        "q_evaluation": "auto",
        "q_interpolation_rtol": 1.0,
        "q_interpolation_atol": 1.0e-8,
        "q_interpolation_mesh": (8,),
        "q_validation_points": 8,
        "transition_max_batch_bytes": 12_000,
        "cache": cache,
    }

    first = bare_spin_susceptibility(model, Q, energy, mesh, **settings)
    second = bare_spin_susceptibility(varied, Q, energy, mesh, **settings)

    assert first.provenance["q_interpolation"]["geometry_cache_hit"] is False
    assert second.provenance["q_interpolation"]["geometry_cache_hit"] is True


def test_response_cache_reuses_completed_bare_susceptibility():
    model = _chain_model()
    mesh = k_mesh(model, (24,))
    cache = ElectronicResponseCache(max_bytes=32 * 1024**2, max_entries=8)
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
        "cache": cache,
    }
    first = bare_spin_susceptibility(
        model,
        [0.25, 0.0, 0.0],
        1.0,
        mesh,
        **settings,
    )
    hits = cache.hits
    second = bare_spin_susceptibility(
        model,
        [0.25, 0.0, 0.0],
        1.0,
        mesh,
        **settings,
    )

    assert second is not first
    assert cache.hits > hits
    np.testing.assert_array_equal(
        second.values_per_meV_cell,
        first.values_per_meV_cell,
    )


def test_cupy_response_dispatch_keeps_intermediates_in_accelerator_backend(
    monkeypatch,
):
    import nfit.electronic_response as electronic_response

    calls = []

    class FakeCupyResponse:
        @staticmethod
        def evaluate_lindhard(
            model,
            k,
            weights,
            q,
            energy,
            point_operators,
            **kwargs,
        ):
            calls.append((model, k, weights, q, energy, point_operators, kwargs))
            return (
                np.full((q.shape[0], point_operators.shape[1], point_operators.shape[1]), 7.0j),
                {
                    "base_execution": {
                        "requested_backend": "cupy",
                        "resolved_backend": "cupy",
                        "precision": "float64/complex128",
                        "device_resident": True,
                    },
                    "shifted_execution": [],
                    "transition_batch_size": 4,
                    "energy_batch_size": 2,
                    "commensurate_permutation_count": 1,
                    "direct_shift_count": 0,
                },
            )

    monkeypatch.setattr(
        electronic_response,
        "_CUPY_RESPONSE_BACKEND",
        FakeCupyResponse,
    )
    model = _chain_model()
    response = bare_spin_susceptibility(
        model,
        [0.25, 0.0, 0.0],
        1.0,
        k_mesh(model, (4,)),
        temperature_K=20.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.4,
        backend="cupy",
    )

    assert len(calls) == 1
    assert response.provenance["response_execution"] == "cupy_end_to_end"
    assert response.provenance["host_transfer"] == "completed susceptibility only"
    np.testing.assert_array_equal(
        np.diagonal(response.values_per_meV_cell, axis1=1, axis2=2),
        np.full((1, 3), 3.5j),
    )


def test_cupy_end_to_end_lindhard_matches_numpy_when_available():
    import nfit.electronic_response as electronic_response

    if electronic_response._CUPY_RESPONSE_BACKEND is None:
        pytest.skip("CuPy electronic-response backend is unavailable")
    model = _chain_model()
    mesh = k_mesh(model, (16,))
    settings = {
        "temperature_K": 20.0,
        "chemical_potential_meV": 0.0,
        "broadening_meV": 0.4,
    }
    q = np.asarray([[0.25, 0.0, 0.0], [0.137, 0.0, 0.0]])
    energy = np.asarray([0.0, 1.5])
    reference = bare_spin_susceptibility(
        model,
        q,
        energy,
        mesh,
        backend="numpy",
        **settings,
    )
    accelerated = bare_spin_susceptibility(
        model,
        q,
        energy,
        mesh,
        backend="cupy",
        cache=ElectronicResponseCache(
            max_bytes=32 * 1024**2,
            max_entries=8,
        ),
        **settings,
    )

    assert accelerated.provenance["response_execution"] == "cupy_end_to_end"
    np.testing.assert_allclose(
        accelerated.values_per_meV_cell,
        reference.values_per_meV_cell,
        rtol=1.0e-10,
        atol=1.0e-11,
    )


def test_rpa_reports_and_can_reject_sampled_static_instability():
    bare = SusceptibilityResult(
        q_reduced=[[0.0, 0.0, 0.0]],
        energy_meV=[0.0],
        values_per_meV_cell=[[[1.1 + 0.0j]]],
        operator_labels=("spin",),
        conjugate_indices=(0,),
        model_digest="test",
        temperature_K=10.0,
        chemical_potential_meV=0.0,
        broadening_meV=0.5,
    )
    vertex = scalar_stoner_vertex(("spin",), 1.0, energy_unit="meV")
    reported = rpa_dress_susceptibility(
        bare,
        vertex,
        reject_sampled_static_instability=False,
    )
    stability = reported.provenance["dressing"]["static_stability"]

    assert stability["unstable"]
    assert stability["margin"] < 0.0
    with pytest.raises(np.linalg.LinAlgError, match="sampled static RPA"):
        rpa_dress_susceptibility(
            bare,
            vertex,
            reject_sampled_static_instability=True,
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


def test_lindhard_plot_configuration_is_public_and_atomic():
    lindhard = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            field.name: field.default
            for field in model_definition("lindhard").config_fields
        },
    )

    configure_lindhard_plot(
        lindhard,
        "complex_energy_scan",
        plot_q_reduced=[0.5, 0.25, 0.0],
        plot_energy_points=201,
    )
    assert lindhard.config["plot_q_reduced"] == [0.5, 0.25, 0.0]
    assert lindhard.config["plot_energy_points"] == 201

    previous = dict(lindhard.config)
    with pytest.raises(ValueError, match="plot_energy_points"):
        configure_lindhard_plot(
            lindhard,
            "complex_energy_scan",
            plot_energy_points=1,
        )
    assert lindhard.config == previous


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


def test_lindhard_bulk_normalization_uses_inferred_model_cell_formula_units():
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
            "crystal": {
                "lattice": {
                    "a": 2.0,
                    "b": 8.0,
                    "c": 9.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                },
                "spacegroup": "P 1",
                "sites": [
                    {
                        "label": "A1",
                        "element": "Li",
                        "position": [0.0, 0.0, 0.0],
                    },
                    {
                        "label": "A2",
                        "element": "Li",
                        "position": [0.5, 0.0, 0.0],
                    },
                ],
            },
        },
    )
    defaults = {
        field.name: field.default
        for field in model_definition("lindhard").config_fields
    }
    automatic = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            **defaults,
            "electronic_component": "bands",
            "response_mesh": [16],
            "response_mesh_shift": [0.0],
        },
    )
    manual = replace(
        automatic,
        config={
            **automatic.config,
            "formula_units_mode": "manual",
            "formula_units_per_cell": 1.0,
        },
    )
    points = PointData4D(
        H=np.zeros(1),
        K=np.zeros(1),
        L=np.zeros(1),
        E=np.zeros(1),
        intensity=np.zeros(1),
        sigma=np.ones(1),
        temperature=20.0,
        metadata={
            "absolute_units": True,
            "quantity_type": "bulk_susceptibility",
            "unit": "cm^3/mol",
        },
    )
    dataset = FitDatasetInput("bulk", points, data_type="magnetization")

    def prediction(response):
        compiled = compile_fit_problem([tight_binding, response], [dataset])
        return evaluate_problem_model(
            compiled.problem,
            "bulk",
            {
                spec.name: spec.value
                for spec in compiled.problem.parameter_specs
            },
        )

    np.testing.assert_allclose(
        prediction(automatic),
        prediction(manual) / 2.0,
        rtol=1.0e-12,
    )


def test_lindhard_spectral_normalization_uses_dataset_formula_unit_basis():
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
            "crystal": {
                "lattice": {
                    "a": 2.0,
                    "b": 8.0,
                    "c": 9.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                },
                "spacegroup": "P 1",
                "sites": [
                    {"label": "A1", "element": "Li", "position": [0.0, 0.0, 0.0]},
                    {"label": "A2", "element": "Li", "position": [0.5, 0.0, 0.0]},
                ],
            },
        },
    )
    defaults = {
        field.name: field.default
        for field in model_definition("lindhard").config_fields
    }
    automatic = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 0.5},
        config={
            **defaults,
            "electronic_component": "bands",
            "response_mesh": [24],
            "response_mesh_shift": [0.0],
        },
    )
    manual = replace(
        automatic,
        config={
            **automatic.config,
            "formula_units_mode": "manual",
            "formula_units_per_cell": 1.0,
        },
    )
    points = PointData4D(
        H=np.asarray([0.5]),
        K=np.zeros(1),
        L=np.zeros(1),
        E=np.asarray([1.0]),
        intensity=np.zeros(1),
        sigma=np.ones(1),
        temperature=20.0,
        metadata={
            "spectral_observable": {
                "fit_representation": "chi_double_prime",
                "normalization_basis": "per_formula_unit",
                "normalization_label": "",
                "moment_unit": "spin_squared",
                "g_factor": 2.0,
                "kf_ki_state": "removed",
                "unit": "spin^2/meV/f.u.",
            }
        },
    )
    dataset = FitDatasetInput(
        "scan",
        points,
        data_type="single_crystal_inelastic",
    )

    def prediction(response):
        compiled = compile_fit_problem([tight_binding, response], [dataset])
        return evaluate_problem_model(
            compiled.problem,
            "scan",
            {spec.name: spec.value for spec in compiled.problem.parameter_specs},
        )

    np.testing.assert_allclose(
        prediction(automatic),
        prediction(manual) / 2.0,
        rtol=1.0e-12,
    )


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


def test_fit_compiler_shares_one_lindhard_context_across_datasets(monkeypatch):
    import nfit.electronic_response as electronic_response
    import nfit.fit_config as fit_config

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
            "response_mesh": [8],
            "response_mesh_shift": [0.0],
        },
    )
    points = PointData4D(
        H=np.asarray([0.25]),
        K=np.zeros(1),
        L=np.zeros(1),
        E=np.asarray([1.0]),
        intensity=np.zeros(1),
        sigma=np.ones(1),
        temperature=20.0,
    )
    calls = 0
    eigensystem_calls = 0
    original = fit_config._lindhard_factory
    original_eigensystem = electronic_response.evaluate_eigensystem

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def counted_eigensystem(*args, **kwargs):
        nonlocal eigensystem_calls
        eigensystem_calls += 1
        return original_eigensystem(*args, **kwargs)

    monkeypatch.setattr(fit_config, "_lindhard_factory", counted)
    monkeypatch.setattr(
        electronic_response,
        "evaluate_eigensystem",
        counted_eigensystem,
    )
    compiled = compile_fit_problem(
        [tight_binding, lindhard],
        [
            FitDatasetInput(
                "scan_a",
                points,
                data_type="single_crystal_inelastic",
            ),
            FitDatasetInput(
                "scan_b",
                points,
                data_type="single_crystal_inelastic",
            ),
        ],
    )

    assert calls == 1
    first_evaluator = compiled.problem.datasets[0].model.__closure__[0].cell_contents[0]
    second_evaluator = compiled.problem.datasets[1].model.__closure__[0].cell_contents[0]
    assert first_evaluator is second_evaluator
    parameters = {"response.broadening": 0.5}
    first = evaluate_problem_model(compiled.problem, "scan_a", parameters)
    second = evaluate_problem_model(compiled.problem, "scan_b", parameters)
    assert eigensystem_calls == 1
    np.testing.assert_array_equal(second, first)


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


def test_lindhard_fit_evaluation_persists_interpolation_certificate():
    model = _profiled_chain_model()
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
        parameters={"broadening": 2.0},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [32],
            "response_mesh_shift": [0.0],
            "response_q_evaluation": "auto",
            "response_q_interpolation_rtol": 0.2,
            "response_q_interpolation_atol": 1.0e-8,
            "response_q_interpolation_mesh": [8],
            "response_q_validation_points": 2,
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
    points = PointData4D(
        H=np.asarray([0.137, 0.283]),
        K=np.zeros(2),
        L=np.zeros(2),
        E=np.asarray([1.0, 2.0]),
        intensity=np.zeros(2),
        sigma=np.ones(2),
        temperature=30.0,
    )
    compiled = compile_fit_problem(
        [tight_binding, lindhard, stoner],
        [
            FitDatasetInput(
                "scan",
                points,
                data_type="single_crystal_inelastic",
            )
        ],
    )

    evaluate_problem_model(
        compiled.problem,
        "scan",
        {"bare.broadening": 2.0, "dressed.I": 0.1},
    )

    stored = lindhard.config["response_q_interpolation_certificates"]["scan"]
    assert stored["status"] == "certified"
    assert stored["certificates"][0]["certificate_digest"]
    assert stored["certificates"][0]["certified_quantity"] == (
        "rpa_scalar_stoner_probe"
    )


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
    diagnostics = model_definition("stoner_rpa").context_diagnostics(
        stoner,
        None,
        {"bare.broadening": 0.5, "dressed.I": 0.1},
        components,
    )
    assert np.isfinite(diagnostics["stability_margin"])
    assert np.isfinite(diagnostics["minimum_relative_singular_value"])
    assert diagnostics["stability_sample_scope"] == "configured plot Q at E=0"
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


def _cubic_band_model():
    """One s band on a simple cubic lattice, implicit spin."""

    return build_electronic_model(
        direct_lattice=np.diag([3.0, 3.0, 3.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={
            (1, 0, 0): [[-100.0]],
            (0, 1, 0): [[-100.0]],
            (0, 0, 1): [[-100.0]],
        },
        orbital_centers=[[0.0, 0.0, 0.0]],
        energy_unit="meV",
    )


def _thermal_per_spin_dos(model, mesh, mu, temperature):
    """Sum_k w_k sum_n (-df/de), the quantity the static Lindhard limit uses."""

    from nfit.electronic_backends import evaluate_eigensystem
    from nfit.electronic_response import _fermi_function

    eigensystem = evaluate_eigensystem(
        model, mesh.reduced_coordinates, eigenvectors=False
    )
    occupation = _fermi_function(eigensystem.eigenvalues, mu, temperature)
    return float(
        np.einsum("k,kn->", mesh.weights, occupation * (1.0 - occupation))
        / (KB_MEV_PER_K * temperature)
    )


def test_bare_cartesian_spin_response_carries_the_half_spin_trace():
    """chi_s(0, 0) is exactly half the per-spin density of states at mu."""

    model = _cubic_band_model()
    mesh = k_mesh(model, (40, 40, 40), shift=[0.5, 0.5, 0.5], symmetry="full")
    mu, temperature = -200.0, 60.0
    response = bare_spin_susceptibility(
        model,
        np.zeros((1, 3)),
        np.zeros(1),
        mesh,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        broadening_meV=1.0,
    )
    chi = float(isotropic_spin_component(response).real[0])
    assert chi == pytest.approx(
        0.5 * _thermal_per_spin_dos(model, mesh, mu, temperature), rel=1.0e-10
    )


def test_scalar_stoner_uses_the_textbook_stoner_criterion():
    """Enhancement is 1 / (1 - I * D_up(mu)), matching the Hubbard U scale.

    The Cartesian spin response carries the analytic 1/2 spin trace, so the
    vertex conjugate to S_alpha is 2I. Without that factor the instability
    would sit at I * D_up = 2 and I would silently be twice the textbook
    Stoner parameter (and twice the equivalent hubbard_hund_rpa U).
    """

    model = _cubic_band_model()
    mesh = k_mesh(model, (40, 40, 40), shift=[0.5, 0.5, 0.5], symmetry="full")
    mu, temperature = -200.0, 60.0
    bare = bare_spin_susceptibility(
        model,
        np.zeros((1, 3)),
        np.zeros(1),
        mesh,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        broadening_meV=1.0,
    )
    critical = 1.0 / _thermal_per_spin_dos(model, mesh, mu, temperature)
    bare_value = float(isotropic_spin_component(bare).real[0])
    for fraction in (0.25, 0.5, 0.9):
        vertex = scalar_stoner_vertex(
            bare.operator_labels, fraction * critical, energy_unit="meV"
        )
        assert vertex.provenance["spin_channel_vertex_factor"] == 2.0
        dressed = rpa_dress_susceptibility(bare, vertex)
        enhancement = (
            float(isotropic_spin_component(dressed).real[0]) / bare_value
        )
        assert enhancement == pytest.approx(1.0 / (1.0 - fraction), rel=1.0e-9)


def test_stoner_vertex_leaves_a_non_spin_operator_basis_unscaled():
    """The 1/2 spin trace belongs to the spin operator, not the interaction."""

    vertex = scalar_stoner_vertex(("spin",), 0.2, energy_unit="eV")
    assert vertex.provenance["spin_channel_vertex_factor"] == 1.0
    np.testing.assert_allclose(vertex.values_meV, [[200.0]])


def test_density_of_states_matches_the_electron_filling_capacity():
    """DOS and filling must use the same spin convention."""

    from nfit.electronic_structure import density_of_states

    model = _cubic_band_model()
    mesh = k_mesh(model, (36, 36, 36), shift=[0.5, 0.5, 0.5], symmetry="full")
    dos = density_of_states(
        model, mesh, np.linspace(-700.0, 700.0, 2801), broadening_meV=6.0
    )
    capacity = electron_filling(
        model, mesh, chemical_potential_meV=700.0, temperature_K=10.0
    )
    assert dos.provenance["spin_degeneracy"] == 2.0
    assert np.trapezoid(dos.total_per_meV_cell, dos.energy_meV) == pytest.approx(
        capacity, rel=1.0e-6
    )


def test_commensurability_is_resolved_once_per_distinct_wavevector():
    """The per-point commensurability test must agree with the per-q one.

    Response points repeat a handful of wavevectors (one per energy of a
    constant-Q cut), so the test is deduplicated. This locks the deduplicated
    result against the direct permutation test it replaced.
    """

    from nfit.electronic_response import (
        _commensurate_mesh_permutation,
        _commensurate_point_flags,
    )

    model = _cubic_band_model()
    mesh = k_mesh(model, (12, 12, 12), shift=[0.5, 0.5, 0.5], symmetry="full")
    Q = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],      # commensurate with a 12-point axis
            [1.0 / 3.0, 0.0, 0.0], # commensurate
            [0.137, 0.2, 0.0],     # off mesh
            [0.25, 0.0, 0.0],      # repeat
            [2.0, -1.0, 3.0],      # integral, trivially commensurate
        ]
    )
    expected = np.asarray(
        [
            _commensurate_mesh_permutation(model, mesh, np.mod(value, 1.0))
            is not None
            for value in Q
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(
        _commensurate_point_flags(model, mesh, Q), expected
    )
    assert expected.tolist() == [True, True, True, False, True, True]


def test_repeated_wavevectors_do_not_change_the_response():
    """Deduplication must not alter the evaluated susceptibility."""

    model = _cubic_band_model()
    mesh = k_mesh(model, (10, 10, 10), shift=[0.5, 0.5, 0.5], symmetry="full")
    energy = np.linspace(1.0, 40.0, 12)
    q = np.tile([0.2, 0.1, 0.0], (energy.size, 1))
    common = dict(
        temperature_K=60.0, chemical_potential_meV=-150.0, broadening_meV=4.0
    )
    together = isotropic_spin_component(
        bare_spin_susceptibility(model, q, energy, mesh, **common)
    )
    separate = np.concatenate(
        [
            isotropic_spin_component(
                bare_spin_susceptibility(
                    model, q[i : i + 1], energy[i : i + 1], mesh, **common
                )
            )
            for i in range(energy.size)
        ]
    )
    np.testing.assert_allclose(together, separate, rtol=1.0e-12, atol=1.0e-18)


def test_complex_transition_amplitudes_match_two_level_spectral_response():
    """Check probe conjugation against analytic transitions in a rotated basis."""
    gap, broadening = 4.0, 0.2
    pauli = np.array([
        [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
    ], dtype=complex)
    rotation = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    model = build_electronic_model(
        direct_lattice=np.eye(3), basis=["a", "b"], periodic_axes=(0,),
        hoppings={(0, 0, 0): rotation @ np.diag([-gap / 2, gap / 2]) @ rotation.conj().T},
        energy_unit="meV",
    )
    operators = ElectronicOperatorBasis(
        ("Sx", "Sy", "Sz"), rotation @ (pauli / 2) @ rotation.conj().T,
    )
    energy = np.array([-6.0, -1.0, 1.0, 4.0, 6.0])
    response = bare_lindhard_susceptibility(
        model, np.zeros((len(energy), 3)), energy, k_mesh(model, [3]), operators,
        temperature_K=0.0, chemical_potential_meV=0.0, broadening_meV=broadening,
    )
    forward = 1.0 / (gap - energy - 1j * broadening)
    reverse = 1.0 / (gap + energy + 1j * broadening)
    expected = np.zeros((len(energy), 3, 3), dtype=complex)
    expected[:, 0, 0] = expected[:, 1, 1] = (forward + reverse) / 4
    expected[:, 0, 1] = 1j * (forward - reverse) / 4
    expected[:, 1, 0] = -expected[:, 0, 1]
    np.testing.assert_allclose(response.values_per_meV_cell, expected, atol=1e-14)

    absorptive = (expected - expected.conj().swapaxes(-1, -2)) / (2j)
    assert np.max(np.abs(absorptive.imag)) > 0.1
    # The documented accessor is elementwise imaginary; physical absorption
    # is the anti-Hermitian part and can have complex off-diagonal entries.
    np.testing.assert_allclose(response.chi_double_prime, expected.imag, atol=1e-14)
    q = np.tile([0.3, 0.4, np.sqrt(0.75)], (len(energy), 1))
    projector = np.eye(3) - q[:, :, None] * q[:, None, :]
    np.testing.assert_allclose(
        np.einsum("pab,pab->p", projector, absorptive).real,
        np.einsum("pab,pab->p", projector, response.chi_double_prime),
        atol=1e-14,
    )
