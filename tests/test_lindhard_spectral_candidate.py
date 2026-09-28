"""Physics checks for the isolated spectral-compression experiment."""
import numpy as np
import pytest

from benchmarks.lindhard_spectral_candidate import (
    direct_sum,
    prepare_transitions,
    spectral_sum,
)
from nfit import (
    ElectronicOperatorBasis,
    bare_lindhard_susceptibility,
    build_electronic_model,
    k_mesh,
)


def _two_level(operators, temperature=0.0):
    model = build_electronic_model(
        direct_lattice=np.eye(3), basis=["a", "b"],
        hoppings={(0, 0, 0): np.diag([-1.13, 1.13])},
        periodic_axes=(0,), energy_unit="meV",
    )
    mesh = k_mesh(model, [3])
    table = prepare_transitions(
        model, mesh, [0, 0, 0], operators, temperature_K=temperature,
        chemical_potential_meV=0.0,
    )
    return model, mesh, table


def test_spectral_complex_probe_preserves_signed_poles_and_converges():
    operators = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]]]) / 2
    model, mesh, table = _two_level(operators)
    energy = np.array([-3.17, -2.26, -0.31, 0, 0.31, 2.26, 3.17])
    eta = 0.3
    response = bare_lindhard_susceptibility(
        model, np.zeros((len(energy), 3)), energy, mesh,
        ElectronicOperatorBasis(("x", "y"), operators), temperature_K=0,
        chemical_potential_meV=0, broadening_meV=eta, backend="numpy", workers=1,
    ).values_per_meV_cell
    np.testing.assert_allclose(direct_sum(table, energy, eta), response, atol=1e-14)
    errors = []
    for density in (4, 8, 16, 32):
        approximate, _ = spectral_sum(table, energy, eta, bins_per_width=density)
        errors.append(np.max(np.abs(approximate - response)))
    assert errors[-1] < errors[0] / 20
    np.testing.assert_allclose(approximate, response, rtol=5e-4, atol=1e-5)
    # Circular probes retain negative tails; no clipping to positive intensity.
    circular = np.array([[[0, 0], [1, 0]]])
    _, _, circular_table = _two_level(circular)
    result, _ = spectral_sum(circular_table, [0.4], eta)
    assert result[0, 0, 0].imag < 0


@pytest.mark.parametrize("temperature", [0.0, 12.0])
def test_spectral_uniform_static_patch_is_kept_separate(temperature):
    operators = np.eye(2)[None]
    _, _, table = _two_level(operators, temperature)
    energies = np.array([-0.5, 0, 0.5])
    for eta in (0.1, 0.7):
        result, _ = spectral_sum(table, energies, eta)
        np.testing.assert_allclose(result, direct_sum(table, energies, eta), atol=1e-14)
        assert result[1, 0, 0].real > 0
        np.testing.assert_allclose(result[[0, 2]], 0, atol=1e-14)


def test_spectral_far_transitions_contribute_real_response_without_fft_wraparound():
    _, _, table = _two_level(np.array([[[0, 1], [1, 0]]]))
    energy = np.array([0.1, 40.013, 80.027])
    result, _ = spectral_sum(table, energy, 0.3, bins_per_width=32)
    np.testing.assert_allclose(result, direct_sum(table, energy, 0.3), rtol=2e-5, atol=1e-7)
    assert abs(result[0, 0, 0].real) > 0.8
    with pytest.raises(ValueError, match="exceeding"):
        spectral_sum(table, energy, 1e-5, max_bins=100)


def test_spectral_multiband_phase_vertices_agree_with_public_api():
    from nfit.electronic_response import spin_operator_matrices

    model = build_electronic_model(
        direct_lattice=np.eye(3), basis=["a", "b"],
        orbital_centers=[[0, 0, 0], [0.37, 0, 0]],
        hoppings={(0, 0, 0): [[-0.3, 0.1j], [-0.1j, 0.8]],
                  (1, 0, 0): [[-1, 0.23], [0.11, 0.7]]},
        periodic_axes=(0,), energy_unit="meV",
    )
    mesh = k_mesh(model, [43])
    Q = [1.173, 0, 0]
    basis, vertices, _ = spin_operator_matrices(model, Q)
    table = prepare_transitions(model, mesh, Q, vertices[0], temperature_K=6,
                                chemical_potential_meV=0.1)
    energy = np.linspace(-6, 6, 97)
    exact = bare_lindhard_susceptibility(
        model, np.tile(Q, (len(energy), 1)), energy, mesh, basis,
        operator_matrices_by_point=np.broadcast_to(vertices, (len(energy), *vertices.shape[1:])),
        temperature_K=6, chemical_potential_meV=0.1, broadening_meV=0.15,
        backend="numpy", workers=1,
    ).values_per_meV_cell
    np.testing.assert_allclose(direct_sum(table, energy, 0.15), exact, atol=1e-14)
    approximate, _ = spectral_sum(table, energy, 0.15, bins_per_width=32)
    assert np.max(np.abs(approximate - exact)) / np.max(np.abs(exact)) < 2e-4
