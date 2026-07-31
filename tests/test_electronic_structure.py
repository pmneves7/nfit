from __future__ import annotations

import threading
import time
import warnings
from dataclasses import replace
from itertools import permutations, product

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nfit import (
    BasisState,
    DataGroup,
    ElectronicModel,
    ModelComponentSpec,
    NfitProject,
    WavevectorSampling,
    automatic_dos_energy_limits,
    band_path,
    build_electronic_model,
    calculate_bands,
    create_model_component,
    density_of_states,
    electronic_energy_from_meV,
    electronic_energy_to_meV,
    evaluate_eigensystem,
    fermi_surface,
    import_wannier90,
    k_mesh,
    load_electronic_model,
    load_project,
    model_definition,
    save_electronic_model,
    save_project,
    validate_electronic_backend,
)


def _chain_model(*, hopping=-100.0, onsite=0.0):
    return build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={
            (0, 0, 0): np.array([[onsite]]),
            (1, 0, 0): np.array([[hopping]]),
        },
        orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,),
        energy_unit="meV",
    )


def _square_two_orbital_model():
    onsite = np.diag([-25.0, 25.0])
    hopping = np.diag([-80.0, -40.0])
    return build_electronic_model(
        direct_lattice=np.diag([4.0, 4.0, 12.0]),
        basis=[
            BasisState("d", site="M", orbital="d"),
            BasisState("p", site="X", orbital="p"),
        ],
        hoppings={
            (0, 0, 0): onsite,
            (1, 0, 0): hopping,
            (0, 1, 0): hopping,
        },
        orbital_centers=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        periodic_axes=(0, 1),
        energy_unit="meV",
    )


def _cubic_two_orbital_model():
    onsite = np.diag([-30.0, 30.0])
    hopping = np.diag([-35.0, -20.0])
    return build_electronic_model(
        direct_lattice=np.diag([4.0, 4.0, 4.0]),
        basis=[
            BasisState("d", site="M", orbital="d"),
            BasisState("p", site="X", orbital="p"),
        ],
        hoppings={
            (0, 0, 0): onsite,
            (1, 0, 0): hopping,
            (0, 1, 0): hopping,
            (0, 0, 1): hopping,
        },
        periodic_axes=(0, 1, 2),
        energy_unit="meV",
    )


def test_arbitrary_chain_uses_wannier_phase_and_physical_path_distance():
    model = _chain_model(hopping=-100.0, onsite=12.0)
    wavevectors = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )
    expected = 12.0 - 200.0 * np.cos(2.0 * np.pi * wavevectors[:, 0])

    np.testing.assert_allclose(
        np.linalg.eigvalsh(model.hamiltonian(wavevectors))[:, 0], expected
    )
    np.testing.assert_allclose(
        model.hamiltonian(wavevectors + [1.0, 0.0, 0.0]),
        model.hamiltonian(wavevectors),
    )

    path = band_path(
        model,
        [[0.0], [0.5]],
        labels=[r"$\Gamma$", "X"],
        points_per_inv_angstrom=2.5,
    )
    result = calculate_bands(model, path)
    assert path.labels == ((0, r"$\Gamma$"), (4, "X"))
    assert path.path_distance_inv_angstrom[-1] == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(
        result.energies_meV[:, 0],
        12.0 - 200.0 * np.cos(2.0 * np.pi * path.reduced_coordinates[:, 0]),
    )
    restored_path = WavevectorSampling.from_dict(path.to_dict())
    np.testing.assert_allclose(
        restored_path.path_distance_inv_angstrom,
        path.path_distance_inv_angstrom,
    )

    alternate_reciprocal = model.reciprocal_lattice.copy()
    alternate_reciprocal[:, 0] *= 2.0
    alternate_path = band_path(
        model,
        [[0.0], [0.5]],
        labels=["G", "B"],
        points_per_inv_angstrom=0.25,
        coordinate_reciprocal_lattice=alternate_reciprocal,
    )
    assert alternate_path.path_distance_inv_angstrom[-1] == pytest.approx(np.pi)
    np.testing.assert_allclose(
        alternate_path.reduced_coordinates[-1],
        [1.0, 0.0, 0.0],
    )

    density_path = band_path(
        model,
        [[0.0], [0.25], [1.0]],
        labels=["G", "A", "B"],
        points_per_inv_angstrom=4.0 / np.pi,
    )
    assert density_path.provenance["segment_intervals"] == (1, 3)
    assert density_path.labels == ((0, "G"), (1, "A"), (4, "B"))


def test_projected_bands_and_dos_preserve_state_count():
    model = _square_two_orbital_model()
    mesh = k_mesh(model, (50, 50))
    bands = calculate_bands(
        model,
        mesh,
        projections={"d": [0], "p": [1], "all": [0, 1]},
    )
    np.testing.assert_allclose(
        bands.projected_weights["d"] + bands.projected_weights["p"], 1.0
    )
    np.testing.assert_allclose(bands.projected_weights["all"], 1.0)

    energy = np.linspace(-500.0, 500.0, 2001)
    dos = density_of_states(
        model,
        mesh,
        energy,
        broadening_meV=5.0,
        projections={"d": [0], "p": [1]},
        max_chunk_bytes=1_000_000,
    )
    # Two basis states times the two-fold implicit spin degeneracy.
    assert np.trapezoid(dos.total_per_meV_cell, energy) == pytest.approx(
        4.0, rel=2.0e-3
    )
    assert dos.provenance["spin_degeneracy"] == 2.0
    np.testing.assert_allclose(
        dos.projected_per_meV_cell["d"] + dos.projected_per_meV_cell["p"],
        dos.total_per_meV_cell,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_automatic_dos_range_encloses_bands_and_gaussian_tails():
    model = _chain_model()
    mesh = k_mesh(model, (80,))
    bands = calculate_bands(model, mesh)

    lower, upper = automatic_dos_energy_limits(
        bands.energies_meV,
        broadening_meV=5.0,
    )
    assert lower == pytest.approx(-220.0)
    assert upper == pytest.approx(220.0)

    dos = density_of_states(
        model,
        mesh,
        None,
        broadening_meV=5.0,
        energy_points=301,
    )
    assert dos.energy_meV.shape == (301,)
    assert dos.energy_meV[0] == pytest.approx(lower)
    assert dos.energy_meV[-1] == pytest.approx(upper)
    assert dos.provenance["automatic_energy_range"] is True


def test_tetrahedron_dos_preserves_total_and_projected_state_counts():
    model = _cubic_two_orbital_model()
    mesh = k_mesh(model, (18, 18, 18), symmetry="full")
    energy = np.linspace(-300.0, 300.0, 2401)

    dos = density_of_states(
        model,
        mesh,
        energy,
        broadening_meV=5.0,
        method="tetrahedron",
        projections={"d": [0], "p": [1]},
    )

    assert dos.broadening_meV == 0.0
    assert dos.provenance["method"] == "tetrahedron"
    assert dos.provenance["provider"] == "ASE"
    assert dos.provenance["provider_version"]
    # Two basis states times the two-fold implicit spin degeneracy.
    assert np.trapezoid(dos.total_per_meV_cell, energy) == pytest.approx(
        4.0,
        rel=2.0e-3,
    )
    assert dos.provenance["spin_degeneracy"] == 2.0
    np.testing.assert_allclose(
        dos.projected_per_meV_cell["d"]
        + dos.projected_per_meV_cell["p"],
        dos.total_per_meV_cell,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_tetrahedron_dos_represents_an_exact_flat_band_as_a_delta():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["dispersive", "flat"],
        hoppings={
            (0, 0, 0): np.diag([0.0, 100.0]),
            (1, 0, 0): np.diag([-10.0, 0.0]),
            (0, 1, 0): np.diag([-10.0, 0.0]),
            (0, 0, 1): np.diag([-10.0, 0.0]),
        },
        energy_unit="meV",
    )
    energy = np.linspace(-80.0, 120.0, 2001)
    dos = density_of_states(
        model,
        k_mesh(model, (8, 8, 8), symmetry="full"),
        energy,
        broadening_meV=1.0,
        method="tetrahedron",
        symmetry="full",
        projections={"dispersive": [0], "flat": [1]},
    )

    # One state per band times the two-fold implicit spin degeneracy.
    assert np.trapezoid(dos.total_per_meV_cell, energy) == pytest.approx(4.0)
    assert np.trapezoid(
        dos.projected_per_meV_cell["dispersive"], energy
    ) == pytest.approx(2.0)
    assert np.trapezoid(
        dos.projected_per_meV_cell["flat"], energy
    ) == pytest.approx(2.0)
    flat_peak = int(np.argmax(dos.projected_per_meV_cell["flat"]))
    assert energy[flat_peak] == pytest.approx(100.0)
    assert dos.provenance["flat_band_indices"] == (1,)
    assert dos.provenance["flat_band_energies_meV"] == pytest.approx((100.0,))


def test_tetrahedron_dos_requires_full_three_dimensional_uniform_mesh():
    model = _square_two_orbital_model()
    mesh = k_mesh(model, (8, 8), symmetry="full")

    with pytest.raises(ValueError, match="three-dimensional"):
        density_of_states(
            model,
            mesh,
            np.linspace(-300.0, 300.0, 101),
            broadening_meV=5.0,
            method="tetrahedron",
        )


def test_basis_permutation_and_orbital_origin_do_not_change_bands():
    model = _square_two_orbital_model()
    permutation = np.array([1, 0])
    blocks = {
        tuple(vector): block[np.ix_(permutation, permutation)]
        for vector, block in zip(
            model.translations, model.hamiltonian_blocks, strict=True
        )
        if tuple(vector) >= (0, 0, 0)
    }
    permuted = build_electronic_model(
        direct_lattice=model.direct_lattice,
        basis=[model.basis[index] for index in permutation],
        hoppings=blocks,
        interpolation_weights={
            tuple(vector): weight
            for vector, weight in zip(
                model.translations, model.interpolation_weights, strict=True
            )
        },
        orbital_centers=model.orbital_centers[permutation] + [0.2, -0.1, 0.0],
        periodic_axes=model.periodic_axes,
        energy_unit="meV",
    )
    wavevectors = np.array([[0.13, 0.27, 0.0], [0.4, 0.1, 0.0]])

    np.testing.assert_allclose(
        np.linalg.eigvalsh(permuted.hamiltonian(wavevectors)),
        np.linalg.eigvalsh(model.hamiltonian(wavevectors)),
    )


def test_named_hopping_parameters_return_new_models():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[5.0]]},
        periodic_axes=(0,),
        parameter_values={"t": -10.0},
        parameter_hoppings={"t": {(1, 0, 0): [[1.0]]}},
        energy_unit="meV",
    )
    changed = model.with_parameters(t=-20.0, energy_unit="meV")

    assert model.parameter_values["t"] == -10.0
    assert changed.parameter_values["t"] == -20.0
    assert np.linalg.eigvalsh(model.hamiltonian([0.0, 0.0, 0.0]))[0] == -15.0
    assert np.linalg.eigvalsh(changed.hamiltonian([0.0, 0.0, 0.0]))[0] == -35.0
    assert changed.content_digest != model.content_digest
    assert changed.translations is model.translations
    assert changed.hamiltonian_blocks is model.hamiltonian_blocks
    assert changed.parameter_blocks["t"] is model.parameter_blocks["t"]


def test_immutable_model_caches_do_not_cross_parameter_updates():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[5.0]]},
        periodic_axes=(0,),
        parameter_values={"t": -10.0},
        parameter_hoppings={"t": {(1, 0, 0): [[1.0]]}},
        energy_unit="meV",
    )
    first_blocks = model.resolved_hamiltonian_blocks
    first_digest = model.content_digest
    first_reciprocal = model.reciprocal_lattice

    assert model.resolved_hamiltonian_blocks is first_blocks
    assert model.content_digest is first_digest
    assert model.reciprocal_lattice is first_reciprocal

    changed = model.with_parameters(t=-20.0, energy_unit="meV")
    assert changed.resolved_hamiltonian_blocks is not first_blocks
    assert changed.content_digest != first_digest
    np.testing.assert_allclose(
        changed.resolved_hamiltonian_blocks
        - model.resolved_hamiltonian_blocks,
        changed.parameter_blocks["t"] * -10.0,
    )


def test_existing_full_payload_model_digest_remains_loadable():
    model = _chain_model(hopping=-13.0, onsite=2.0)
    payload = model.to_dict()
    payload["content_digest"] = model._legacy_content_digest()

    restored = ElectronicModel.from_dict(payload)

    np.testing.assert_array_equal(
        restored.hamiltonian([[0.17, 0.0, 0.0]]),
        model.hamiltonian([[0.17, 0.0, 0.0]]),
    )


def test_parameter_trials_reuse_momentum_hamiltonian_components(monkeypatch):
    import nfit.electronic_structure as electronic_structure

    model = build_electronic_model(
        direct_lattice=np.diag([1.2345, 7.0, 9.0]),
        basis=["unique_parameter_cache_state"],
        hoppings={(0, 0, 0): [[3.0]]},
        periodic_axes=(0,),
        parameter_values={"unique_t": -7.0},
        parameter_hoppings={"unique_t": {(1, 0, 0): [[1.0]]}},
        energy_unit="meV",
    )
    coordinates = np.asarray(
        [[0.1234567, 0.0, 0.0], [0.3456789, 0.0, 0.0]]
    )
    changed = model.with_parameters(unique_t=-11.0, energy_unit="meV")
    calls = 0
    original = electronic_structure.np.einsum

    def counted(*args, **kwargs):
        nonlocal calls
        if args and args[0] == "kr,crij->ckij":
            calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(electronic_structure.np, "einsum", counted)
    first = model.hamiltonian(coordinates)
    second = changed.hamiltonian(coordinates)

    assert calls == 1
    assert changed.structure_digest == model.structure_digest
    phase = np.exp(
        2.0j * np.pi * coordinates @ model.translations.astype(float).T
    )
    phase *= model.interpolation_weights[None, :]
    np.testing.assert_allclose(
        first,
        original(
            "kr,rij->kij",
            phase,
            model.resolved_hamiltonian_blocks,
            optimize=True,
        ),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        second,
        original(
            "kr,rij->kij",
            phase,
            changed.resolved_hamiltonian_blocks,
            optimize=True,
        ),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_eigensystem_backends_preserve_reference_values_and_order(monkeypatch):
    model = _square_two_orbital_model()
    coordinates = k_mesh(model, (17, 13)).reduced_coordinates
    reference = evaluate_eigensystem(
        model,
        coordinates,
        eigenvectors=False,
        backend="numpy",
        max_batch_bytes=1024,
    )
    threaded = evaluate_eigensystem(
        model,
        coordinates,
        eigenvectors=False,
        backend="threaded",
        workers=2,
        max_batch_bytes=1024,
    )

    np.testing.assert_array_equal(threaded.eigenvalues, reference.eigenvalues)
    assert reference.eigenvectors is None
    assert threaded.provenance["resolved_backend"] == "threaded"
    assert threaded.provenance["approximation"] == "none"
    assert threaded.provenance["batch_size"] < coordinates.shape[0]
    assert (
        threaded.provenance["thread_schedule"]
        == "serial_hamiltonian_parallel_eigensolver_waves"
    )

    with_vectors = evaluate_eigensystem(
        model,
        coordinates[:4],
        eigenvectors=True,
        backend="numpy",
    )
    assert with_vectors.eigenvectors is not None
    reconstructed = (
        with_vectors.eigenvectors
        * with_vectors.eigenvalues[:, None, :]
    ) @ with_vectors.eigenvectors.conj().transpose(0, 2, 1)
    np.testing.assert_allclose(
        reconstructed,
        model.hamiltonian(coordinates[:4]),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    monkeypatch.setattr("nfit.electronic_backends._CUPY_BACKEND", None)
    fallback = evaluate_eigensystem(
        model,
        coordinates[:4],
        eigenvectors=False,
        backend="cupy",
    )
    assert fallback.provenance["requested_backend"] == "cupy"
    assert fallback.provenance["resolved_backend"] == "numpy"

    with pytest.raises(ValueError, match="at least one"):
        evaluate_eigensystem(
            model,
            np.empty((0, 3)),
            eigenvectors=False,
        )
    with pytest.raises(ValueError, match="workers"):
        evaluate_eigensystem(
            model,
            coordinates[:1],
            eigenvectors=False,
            workers=-1,
        )


def test_threaded_backend_never_overlaps_model_evaluation_with_eigensolver(
    monkeypatch,
):
    import nfit.electronic_backends as backends

    base_model = _square_two_orbital_model()
    lock = threading.Lock()
    active_eigensolvers = 0
    original = backends._numpy_matrices_chunk

    def guarded_eigensolver(matrices, *, eigenvectors):
        nonlocal active_eigensolvers
        with lock:
            active_eigensolvers += 1
        try:
            time.sleep(0.005)
            return original(matrices, eigenvectors=eigenvectors)
        finally:
            with lock:
                active_eigensolvers -= 1

    class GuardedModel:
        def __getattr__(self, name):
            return getattr(base_model, name)

        def hamiltonian(self, coordinates):
            with lock:
                assert active_eigensolvers == 0
            return base_model.hamiltonian(coordinates)

    monkeypatch.setattr(
        backends,
        "_numpy_matrices_chunk",
        guarded_eigensolver,
    )
    result = evaluate_eigensystem(
        GuardedModel(),
        k_mesh(base_model, (31, 17)).reduced_coordinates,
        eigenvectors=True,
        backend="threaded",
        workers=3,
        max_batch_bytes=2048,
    )

    assert result.eigenvectors is not None
    assert result.provenance["resolved_backend"] == "threaded"


def test_backend_validation_certifies_threaded_and_reports_fallback(monkeypatch):
    model = _square_two_orbital_model()
    coordinates = k_mesh(model, (7, 5)).reduced_coordinates
    threaded = validate_electronic_backend(
        model,
        coordinates,
        "threaded",
        workers=2,
    )

    assert threaded.passed
    assert threaded.resolved_backend == "threaded"
    assert threaded.max_absolute_error_meV == 0.0

    monkeypatch.setattr("nfit.electronic_backends._CUPY_BACKEND", None)
    fallback = validate_electronic_backend(
        model,
        coordinates[:2],
        "cupy",
    )
    assert not fallback.passed
    assert fallback.resolved_backend == "numpy"
    assert "resolved to" in fallback.reason


def test_symmetry_reduced_mesh_is_opt_in_and_preserves_total_dos():
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.zeros((3, 3), dtype=int)
            for row, column in enumerate(permutation):
                rotation[row, column] = signs[row]
            rotations.append(rotation.tolist())
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
    model = replace(
        model,
        provenance={
            **dict(model.provenance),
            "reciprocal_symmetry": {
                "certified_by": "nfit_orbital_builder",
                "rotations": rotations,
                "includes_time_reversal": True,
            },
        },
    )
    full = k_mesh(model, (8, 8, 8))
    reduced = k_mesh(model, (8, 8, 8), symmetry="reduced")

    assert full.reduced_coordinates.shape[0] == 512
    assert reduced.reduced_coordinates.shape[0] == 35
    assert reduced.provenance["symmetry_reduction"]["applied"] is True
    assert reduced.weights.sum() == pytest.approx(1.0)

    energy = np.linspace(-7.0, 7.0, 301)
    full_dos = density_of_states(
        model,
        full,
        energy,
        broadening_meV=0.2,
    )
    reduced_dos = density_of_states(
        model,
        reduced,
        energy,
        broadening_meV=0.2,
    )
    np.testing.assert_allclose(
        reduced_dos.total_per_meV_cell,
        full_dos.total_per_meV_cell,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    full_tetrahedron = density_of_states(
        model,
        full,
        energy,
        broadening_meV=0.2,
        method="tetrahedron",
        symmetry="full",
        projections={"all": [0]},
    )
    reduced_tetrahedron = density_of_states(
        model,
        full,
        energy,
        broadening_meV=0.2,
        method="tetrahedron",
        symmetry="reduced",
        projections={"all": [0]},
    )
    np.testing.assert_allclose(
        reduced_tetrahedron.total_per_meV_cell,
        full_tetrahedron.total_per_meV_cell,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        reduced_tetrahedron.projected_per_meV_cell["all"],
        full_tetrahedron.projected_per_meV_cell["all"],
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    reduction = reduced_tetrahedron.provenance["symmetry_reduction"]
    assert reduction["applied"] is True
    assert reduction["full_size"] == 512
    assert reduction["irreducible_size"] == 35

    uncertified = _square_two_orbital_model()
    unchanged = k_mesh(
        uncertified,
        (7, 9),
        symmetry="auto",
    )
    assert unchanged.reduced_coordinates.shape[0] == 63
    assert unchanged.provenance["symmetry_reduction"]["applied"] is False
    with pytest.raises(ValueError, match="not certified"):
        k_mesh(uncertified, (7, 9), symmetry="reduced")

    chain_mesh = k_mesh(
        _chain_model(),
        (8, 1, 1),
        shift=(0.5, 0.0, 0.0),
    )
    assert chain_mesh.shift == (0.5,)
    assert chain_mesh.reduced_coordinates[0, 0] == pytest.approx(0.5 / 8.0)


def test_manual_builder_converts_declared_electronic_units_once():
    ev_model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[0.012]], (1, 0, 0): [[-0.1]]},
        periodic_axes=(0,),
        parameter_values={"shift": 0.003},
        parameter_hoppings={"shift": {(0, 0, 0): [[1.0]]}},
        energy_unit="eV",
        energy_zero=0.004,
    )
    mev_model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={(0, 0, 0): [[12.0]], (1, 0, 0): [[-100.0]]},
        periodic_axes=(0,),
        parameter_values={"shift": 3.0},
        parameter_hoppings={"shift": {(0, 0, 0): [[1.0]]}},
        energy_unit="meV",
        energy_zero=4.0,
    )

    assert ev_model.content_digest == mev_model.content_digest
    assert ev_model.parameter_values["shift"] == 3.0
    assert ev_model.energy_zero_meV == 4.0
    assert ev_model.provenance["source_energy_unit"] == "eV"
    assert ev_model.provenance["canonical_energy_unit"] == "meV"
    assert ev_model.to_dict()["canonical_energy_unit"] == "meV"
    changed = ev_model.with_parameters(shift=0.005, energy_unit="eV")
    assert changed.parameter_values["shift"] == 5.0
    assert electronic_energy_to_meV(0.25, "eV") == 250.0
    assert electronic_energy_from_meV(250.0, "eV") == 0.25
    with pytest.raises(ValueError, match="eV.*meV"):
        electronic_energy_to_meV(1.0, "joule")


def test_fermi_surfaces_cover_one_and_two_dimensions():
    chain = fermi_surface(_chain_model(), (200,), target_energy_meV=0.0)
    points = np.sort(chain.sheets[0].vertices_reduced[:, 0])
    np.testing.assert_allclose(points, [0.25, 0.75], atol=1.0e-12)
    assert chain.sheets[0].vertices_inv_angstrom.shape == (2, 3)

    square = fermi_surface(
        _square_two_orbital_model(),
        (50, 50),
        target_energy_meV=0.0,
        projections={"d": [0], "p": [1]},
    )
    assert square.dimension == 2
    assert square.sheets
    assert all(sheet.connectivity.shape[1] == 2 for sheet in square.sheets)
    assert all(
        sheet.vertices_reduced.shape == sheet.vertices_inv_angstrom.shape
        for sheet in square.sheets
    )


def test_three_dimensional_fermi_surface_is_triangulated():
    model = build_electronic_model(
        direct_lattice=np.eye(3),
        basis=["s"],
        hoppings={
            (0, 0, 0): [[0.0]],
            (1, 0, 0): [[-1.0]],
            (0, 1, 0): [[-1.0]],
            (0, 0, 1): [[-1.0]],
        },
        energy_unit="meV",
    )
    result = fermi_surface(model, (10, 10, 10), target_energy_meV=0.0)

    assert len(result.sheets) == 1
    assert result.sheets[0].vertices_reduced.shape[1] == 3
    assert result.sheets[0].connectivity.shape[1] == 3


def test_canonical_model_round_trip_is_digest_protected_and_immutable(tmp_path):
    model = _square_two_orbital_model()
    path = tmp_path / "model.json"
    save_electronic_model(model, path)
    restored = load_electronic_model(path)

    assert restored.content_digest == model.content_digest
    np.testing.assert_allclose(
        restored.hamiltonian([[0.13, 0.27, 0.0]]),
        model.hamiltonian([[0.13, 0.27, 0.0]]),
    )
    with pytest.raises(ValueError):
        restored.hamiltonian_blocks[0, 0, 0] = 1.0

    payload = restored.to_dict()
    payload["energy_zero_meV"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        ElectronicModel.from_dict(payload)


def _write_hr_fixture(tmp_path):
    hr = tmp_path / "chain_hr.dat"
    hr.write_text(
        "\n".join(
            [
                "created for nfit test",
                "1",
                "3",
                "1 1 1",
                "-1 0 0 1 1 -0.100000 0.0",
                "0 0 0 1 1 0.012000 0.0",
                "1 0 0 1 1 -0.100000 0.0",
                "",
            ]
        )
    )
    (tmp_path / "chain.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "2.0 0.0 0.0",
                "0.0 8.0 0.0",
                "0.0 0.0 9.0",
                "end unit_cell_cart",
                "",
            ]
        )
    )
    (tmp_path / "chain_centres.xyz").write_text(
        "1\nWannier centres\nX 0.5 0.0 0.0\n"
    )
    return hr


def test_wannier90_hr_import_converts_units_centres_and_provenance(tmp_path):
    hr = _write_hr_fixture(tmp_path)
    model = import_wannier90(hr, periodic_axes=(0,))

    assert model.basis[0].label == "w1"
    np.testing.assert_allclose(model.orbital_centers[0], [0.25, 0.0, 0.0])
    energies = np.linalg.eigvalsh(
        model.hamiltonian([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )[:, 0]
    np.testing.assert_allclose(energies, [-188.0, 212.0])
    assert model.provenance["energy_conversion"] == "eV to meV (x1000)"
    assert model.provenance["canonical_energy_unit"] == "meV"
    assert {item["role"] for item in model.provenance["files"]} == {
        "wannier90_hr",
        "wannier90_input",
        "wannier_centres",
    }


def test_wannier90_tb_and_wsvec_import(tmp_path):
    tb = tmp_path / "one_tb.dat"
    tb.write_text(
        "\n".join(
            [
                "created for nfit test",
                "2.0 0.0 0.0",
                "0.0 8.0 0.0",
                "0.0 0.0 9.0",
                "1",
                "3",
                "1 1 1",
                "-1 0 0 1 1 -0.100000 0.0",
                "0 0 0 1 1 0.012000 0.0",
                "1 0 0 1 1 -0.100000 0.0",
                "-1 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "0 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "1 0 0",
                "1 1 0.5 0.0 0.0 0.0 0.0 0.0",
                "",
            ]
        )
    )
    direct = import_wannier90(tb, periodic_axes=(0,))
    np.testing.assert_allclose(direct.orbital_centers[0], [0.25, 0.0, 0.0])

    hr = _write_hr_fixture(tmp_path)
    wsvec = tmp_path / "chain_wsvec.dat"
    wsvec.write_text(
        "\n".join(
            [
                "written with use_ws_distance=.true.",
                "-1 0 0 1 1",
                "1",
                "1 0 0",
                "1 0 0 1 1",
                "1",
                "-1 0 0",
                "",
            ]
        )
    )
    corrected = import_wannier90(hr, periodic_axes=(0,))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(
            corrected.hamiltonian([[0.0, 0.0, 0.0], [0.37, 0.0, 0.0]])
        )[:, 0],
        [-188.0, -188.0],
    )
    assert corrected.provenance["wsvec_applied"] is True


def test_tight_binding_registry_plots_and_scripts_are_component_driven():
    model = _chain_model()
    component = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0],
            "band_path": [
                {"label": "G", "k": [0.0, 0.0, 0.0]},
                {"label": "X", "k": [0.25, 0.0, 0.0]},
                {
                    "label": "Y",
                    "k": [0.5, 0.0, 0.0],
                    "break_before": True,
                },
                {"label": "Z", "k": [0.75, 0.0, 0.0]},
            ],
            "dos_mesh": [60],
            "dos_auto_energy_range": True,
            "fermi_mesh": [100],
        },
    )
    definition = model_definition("tight_binding")
    assert definition.data_types == ("electronic_structure",)
    assert [plot.key for plot in definition.plots] == [
        "bands",
        "dos",
        "fermi_surface",
    ]
    for plot in definition.plots:
        result = plot.calculate(component)
        assert result.model_digest == model.content_digest
        assert result.provenance["display_energy_unit"] == "eV"
        figure, axes = plot.render(result)
        assert figure.axes
        if plot.key == "bands":
            assert "(eV)" in axes.get_ylabel()
            assert "E" in axes.get_ylabel()
            assert r"\epsilon" not in axes.get_ylabel()
            assert [tick.get_text() for tick in axes.get_xticklabels()] == [
                "G",
                "X|Y",
                "Z",
            ]
        elif plot.key == "dos":
            assert result.provenance["automatic_energy_range"] is True
            assert "(eV)" in axes.get_xlabel()
            assert "E" in axes.get_xlabel()
            assert r"\epsilon" not in axes.get_xlabel()
            assert "states / eV" in axes.get_ylabel()
            np.testing.assert_allclose(
                axes.lines[0].get_ydata(), result.total_per_meV_cell * 1000.0
            )
        else:
            assert axes.get_title() == ""
        plt.close(figure)
        script = plot.script(component)
        if plot.key in {"bands", "dos"}:
            assert "ElectronicPlotStyle" in script
        if plot.key == "dos":
            assert "automatic_energy_range" in script
        namespace = {}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive",
                category=UserWarning,
            )
            exec(compile(script, f"<tight-binding-{plot.key}>", "exec"), namespace)
        assert namespace["result"].model_digest == result.model_digest
        plt.close(namespace["figure"])
        assert "model = ElectronicModel.from_dict" in script
        assert "energy_unit = 'eV'" in script
        assert "electronic_energy_to_meV" in script
        assert "electronic_backend = 'auto'" in script
        assert "max_batch_bytes = 268435456" in script
        assert "show_electronic_figure" in script
        if plot.key == "fermi_surface":
            assert "show_fermi_surface_result" in script

    component.config["electronic_energy_unit"] = "meV"
    result = definition.plots[0].calculate(component)
    figure, axis = definition.plots[0].render(result)
    assert result.provenance["display_energy_unit"] == "meV"
    assert "(meV)" in axis.get_ylabel()
    plt.close(figure)
    assert "energy_unit = 'meV'" in definition.plots[0].script(component)


def test_tetrahedron_dos_component_and_exported_script_agree():
    model = _cubic_two_orbital_model()
    component = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **{
                field.name: field.default
                for field in model_definition("tight_binding").config_fields
            },
            "model_data": model.to_dict(),
            "periodic_axes": [0, 1, 2],
            "dos_method": "tetrahedron",
            "dos_symmetry": "full",
            "dos_mesh": [12, 12, 12],
            "dos_energy_min_meV": -300.0,
            "dos_energy_max_meV": 300.0,
            "dos_energy_points": 601,
        },
    )
    plot = next(
        item
        for item in model_definition("tight_binding").plots
        if item.key == "dos"
    )

    result = plot.calculate(component)
    script = plot.script(component)
    namespace = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        exec(compile(script, "<tight-binding-tetrahedron-dos>", "exec"), namespace)

    assert result.provenance["method"] == "tetrahedron"
    assert "method='tetrahedron'" in script
    np.testing.assert_allclose(
        namespace["result"].total_per_meV_cell,
        result.total_per_meV_cell,
    )
    plt.close(namespace["figure"])


def test_manual_electronic_model_reopens_from_a_project(tmp_path):
    model = _chain_model()
    group = DataGroup("Electronic")
    component = create_model_component(group, "chain", type="tight_binding")
    component.config["model_data"] = model.to_dict()
    component.config["periodic_axes"] = [0]
    path = tmp_path / "electronic.nfit"

    save_project(NfitProject([group]), path)
    restored = load_project(path)
    restored_component = restored.data_groups[0].models["chain"]
    result = model_definition("tight_binding").plots[0].calculate(
        restored_component
    )

    assert result.model_digest == model.content_digest
