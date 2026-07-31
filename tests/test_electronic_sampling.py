from __future__ import annotations

import numpy as np
import pytest

from nfit import (
    ModelComponentSpec,
    SamplingCertificate,
    automatic_mesh_ladder,
    build_electronic_model,
    certify_dos_sampling,
    certify_lindhard_component_sampling,
    certify_lindhard_sampling,
    certify_tight_binding_dos_sampling,
    model_definition,
    reciprocal_mesh_shape_for_spacing,
    sampling_policy,
)


def _one_dimensional_model():
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=["s"],
        hoppings={(1, 0, 0): [[-10.0]]},
        periodic_axes=(0,),
        energy_unit="meV",
    )


def test_sampling_profiles_define_accuracy_not_mesh_size():
    assert sampling_policy("preview").relative_tolerance == pytest.approx(0.05)
    assert sampling_policy("standard").relative_tolerance == pytest.approx(0.01)
    assert sampling_policy("high").relative_tolerance == pytest.approx(0.002)
    custom = sampling_policy("custom", relative_tolerance=0.007)
    assert custom.relative_tolerance == pytest.approx(0.007)
    assert custom.consecutive_passes == 2


def test_automatic_mesh_ladder_uses_physical_reciprocal_anisotropy():
    model = build_electronic_model(
        direct_lattice=np.diag([2.0, 4.0, 8.0]),
        basis=["s"],
        hoppings={(0, 0, 0): [[0.0]]},
        periodic_axes=(0, 1, 2),
        energy_unit="meV",
    )
    meshes = automatic_mesh_ladder(
        model,
        [40, 40, 40],
        policy=sampling_policy("standard"),
        max_refinements=5,
        max_mesh_points=1_000_000,
    )

    assert len(meshes) >= 3
    assert meshes[0][0] > meshes[0][1] > meshes[0][2]
    assert all(
        all(fine[index] > coarse[index] for index in range(3))
        for coarse, fine in zip(meshes, meshes[1:], strict=False)
    )


def test_physical_spacing_resolves_anisotropic_reciprocal_mesh():
    model = build_electronic_model(
        direct_lattice=np.diag([2.0, 4.0, 8.0]),
        basis=["s"],
        hoppings={(0, 0, 0): [[0.0]]},
        periodic_axes=(0, 1, 2),
        energy_unit="meV",
    )

    assert reciprocal_mesh_shape_for_spacing(model, 0.1) == (31, 16, 8)


def test_sampling_progress_reports_each_completed_candidate():
    model = _one_dimensional_model()
    events = []
    certificate = certify_dos_sampling(
        model,
        np.linspace(-25.0, 25.0, 101),
        seed_mesh=[12],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        broadening_meV=8.0,
        progress_callback=events.append,
    )

    started = [event for event in events if event.phase == "started"]
    completed = [event for event in events if event.phase == "completed"]
    assert len(started) == len(completed) == len(
        certificate.attempted_meshes
    )
    assert [event.mesh for event in completed] == list(
        certificate.attempted_meshes
    )
    assert all(event.elapsed_seconds >= 0.0 for event in completed)
    assert completed[0].comparison is None
    assert all(event.comparison is not None for event in completed[1:])


def test_dos_sampling_certificate_round_trip_and_budget_failure():
    model = _one_dimensional_model()
    certificate = certify_dos_sampling(
        model,
        np.linspace(-25.0, 25.0, 101),
        seed_mesh=[12],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        broadening_meV=8.0,
    )
    restored = SamplingCertificate.from_dict(certificate.to_dict())

    assert restored == certificate
    assert restored.observable == "density_of_states"
    assert len(restored.attempted_meshes) >= 2
    assert restored.status in {"certified", "budget_exhausted"}

    exhausted = certify_dos_sampling(
        model,
        np.linspace(-25.0, 25.0, 21),
        seed_mesh=[12],
        policy=sampling_policy("standard"),
        max_refinements=3,
        max_mesh_points=1,
        broadening_meV=8.0,
    )
    assert exhausted.status == "budget_exhausted"
    assert exhausted.chosen_mesh is None
    assert exhausted.attempted_meshes == ()


def test_dos_sampling_selects_tested_middle_mesh_after_two_passing_refinements():
    model = build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=["s"],
        hoppings={(0, 0, 0): [[0.0]]},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    certificate = certify_dos_sampling(
        model,
        np.linspace(-20.0, 20.0, 81),
        seed_mesh=[12],
        policy=sampling_policy("high"),
        max_refinements=4,
        max_mesh_points=64,
        broadening_meV=5.0,
    )

    assert certificate.certified
    assert certificate.chosen_mesh == certificate.attempted_meshes[1]
    assert len(certificate.comparisons) == 2
    assert all(comparison.passed for comparison in certificate.comparisons)


def test_lindhard_certificate_compares_full_complex_tensor():
    model = _one_dimensional_model()
    energy = np.asarray([-2.0, 0.0, 2.0])
    certificate = certify_lindhard_sampling(
        model,
        [0.25, 0.0, 0.0],
        energy,
        seed_mesh=[12],
        temperature_K=20.0,
        broadening_meV=5.0,
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
    )

    assert certificate.observable == "lindhard"
    assert certificate.domain["tensor"] == "complex Cartesian spin susceptibility"
    assert certificate.provenance["q_evaluation_during_certificate"] == "direct"
    assert certificate.comparisons


def test_component_certificates_apply_only_a_certified_concrete_mesh():
    model = _one_dimensional_model()
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
            "dos_mesh": [12],
            "dos_energy_min_meV": -25.0,
            "dos_energy_max_meV": 25.0,
            "dos_energy_points": 101,
            "dos_broadening_meV": 8.0,
            "dos_sampling_accuracy": "preview",
            "dos_sampling_max_refinements": 4,
            "dos_sampling_max_mesh_points": 64,
        },
    )
    dos_certificate = certify_tight_binding_dos_sampling(tight_binding)
    assert tight_binding.config["dos_sampling_certificate"]["status"] == (
        dos_certificate.status
    )
    assert dos_certificate.attempted_meshes[0] == (16,)
    if len(dos_certificate.attempted_meshes) > 1:
        assert dos_certificate.attempted_meshes[1] == (22,)
    if dos_certificate.certified:
        assert tight_binding.config["dos_mesh"] == list(
            dos_certificate.chosen_mesh
        )

    lindhard = ModelComponentSpec(
        name="response",
        type="lindhard",
        parameters={"broadening": 5.0},
        config={
            **{
                field.name: field.default
                for field in model_definition("lindhard").config_fields
            },
            "electronic_component": "bands",
            "response_mesh": [12],
            "response_mesh_shift": [0.0],
            "response_sampling_accuracy": "preview",
            "response_sampling_max_refinements": 4,
            "response_sampling_max_mesh_points": 64,
            "plot_energy_min_meV": -2.0,
            "plot_energy_max_meV": 2.0,
            "convergence_energy_points": 3,
            "plot_temperature_K": 20.0,
        },
    )
    response_certificate = certify_lindhard_component_sampling(
        lindhard,
        {"bands": tight_binding, "response": lindhard},
    )
    assert lindhard.config["response_sampling_certificate"]["status"] == (
        response_certificate.status
    )
    if response_certificate.certified:
        assert lindhard.config["response_mesh"] == list(
            response_certificate.chosen_mesh
        )
