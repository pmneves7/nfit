"""Exact dissipative limits used for finite neutron intensity at E=0."""

import numpy as np
import pytest

from nfit import spin_fluctuations as sf
from nfit.tensor_rpa import (
    TensorStructure,
    build_tensor_structure,
    cartesian_qhat_per_point,
    tensor_rpa_unpolarized_chipp,
    tensor_rpa_zeeman_unpolarized_chipp,
    tensor_rpa_zeeman_zero_energy_slope,
    tensor_rpa_zero_energy_slope,
    tensor_susceptibility,
    zeeman_cartesian_affine_inverse,
    zeeman_cartesian_dissipative_factor,
    zeeman_cartesian_propagator,
    zeeman_cartesian_propagator_zero_energy_derivative,
    zeeman_cartesian_zero_energy_dissipative_factor,
)


@pytest.fixture(params=sf.available_rpa_backends())
def backend(request):
    original = sf.rpa_backend()
    sf.set_rpa_backend(request.param)
    try:
        yield request.param
    finally:
        sf.set_rpa_backend(original)


def _network():
    positions = [[0.0, 0.0, 0.0], [0.31, 0.2, 0.4]]
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 1, "offset": [0, 0, 0]},
                {"site_i": 0, "site_j": 1, "offset": [-1, 0, 0]},
            ],
        },
        {
            "label": "J2",
            "bonds": [
                {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
            ],
        },
    ]
    hkl = np.array([[0, 0, 0], [0.17, 0.11, 0.07], [0.17, 0.11, 0.07], [0.5, -0.12, 0.1]])
    geometry = sf.build_rpa_geometry(*hkl.T, positions, orbits)
    return positions, orbits, geometry


def _anisotropic_network():
    positions, orbits, geometry = _network()
    anisotropy = {
        "J1": {
            "enabled": True,
            "basis": [
                {"name": "S", "matrix": [[1, 0.2, 0], [0.2, -0.3, 0.1], [0, 0.1, -0.4]]},
                {"name": "D", "matrix": [[0, 0.3, -0.2], [-0.3, 0, 0.1], [0.2, -0.1, 0]]},
            ],
        }
    }
    lattice = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    structure = build_tensor_structure(
        geometry, positions, orbits, anisotropy=anisotropy, lattice=lattice
    )
    values = {"J1": 0.15, "J2": -0.05, "J1_S": 0.08, "J1_D": -0.03}
    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    return structure, geometry, values, q_hat


def test_empty_exchange_gradients_reproduce_local_limit_across_backends(backend):
    geometry = sf.build_rpa_geometry([0, 0.2], [0, 0], [0, 0], [[0, 0, 0]], [])
    energy = np.array([1.0, -1.0])
    value, gradients = sf.heisenberg_rpa_chipp_and_gradients(
        geometry, energy, chi0=0.5, gamma0=2.0, j_values={}
    )
    np.testing.assert_allclose(value, [0.2, -0.2], atol=1e-15)
    np.testing.assert_allclose(gradients["chi0"], [0.4, -0.4], atol=1e-15)
    np.testing.assert_allclose(gradients["gamma0"], [-0.06, 0.06], atol=1e-15)
    np.testing.assert_allclose(gradients["inverse_mode_energy_sq"], [0.32, -0.32], atol=1e-15)


@pytest.mark.parametrize("name", ["chi0", "gamma0"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_local_parameter_validation_is_shared_by_all_rpa_paths(backend, name, invalid):
    _, _, geometry = _network()
    structure = build_tensor_structure(geometry, [[0, 0, 0], [0.31, 0.2, 0.4]], [])
    kwargs = {"chi0": 0.4, "gamma0": 2.0, name: invalid}
    energy = np.ones(geometry.point_index.size)
    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    evaluations = [
        lambda: sf.heisenberg_rpa_susceptibility(geometry, energy, j_values={}, **kwargs),
        lambda: sf.heisenberg_rpa_chipp(geometry, energy, j_values={}, **kwargs),
        lambda: sf.heisenberg_rpa_chipp_and_gradients(geometry, energy, j_values={}, **kwargs),
        lambda: sf.heisenberg_rpa_zero_energy_slope(geometry, j_values={}, **kwargs),
        lambda: tensor_susceptibility(structure, geometry, energy, param_values={}, **kwargs),
        lambda: tensor_rpa_unpolarized_chipp(
            structure, geometry, energy, q_hat, param_values={}, **kwargs
        ),
        lambda: tensor_rpa_zero_energy_slope(structure, geometry, q_hat, param_values={}, **kwargs),
        lambda: zeeman_cartesian_propagator(
            energy, np.array([0, 0, 1]), omega_larmor=1.0, **kwargs
        ),
    ]
    for evaluate in evaluations:
        with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
            evaluate()


@pytest.mark.parametrize("inertia", [0.0, 0.03])
@pytest.mark.parametrize("shift", [0.0, 0.15])
def test_scalar_slope_and_gradients_match_analytic_chain(backend, inertia, shift):
    h = np.array([0.0, 0.125, 0.5, 0.125])
    geometry = sf.build_rpa_geometry(
        h,
        np.zeros(4),
        np.zeros(4),
        [[0, 0, 0]],
        [{"label": "J", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
    )
    chi, gamma, exchange = 0.4, 1.7, 0.7
    phase = 2.0 * np.cos(2.0 * np.pi * h)
    lam = exchange * phase - shift
    delta = 1.0 - chi * lam
    expected = chi / (gamma * delta**2)
    kwargs = dict(
        chi0=chi,
        gamma0=gamma,
        j_values={"J": exchange},
        lambda_shift=shift,
        inverse_mode_energy_sq=inertia,
    )
    slope, gradients = sf.heisenberg_rpa_zero_energy_slope_and_gradients(geometry, **kwargs)
    np.testing.assert_allclose(slope, expected, rtol=1e-13)
    np.testing.assert_allclose(sf.heisenberg_rpa_zero_energy_slope(geometry, **kwargs), expected)
    np.testing.assert_allclose(gradients["chi0"], (1.0 + chi * lam) / (gamma * delta**3))
    np.testing.assert_allclose(gradients["gamma0"], -expected / gamma)
    np.testing.assert_allclose(gradients["J"], 2.0 * chi**2 * phase / (gamma * delta**3))
    np.testing.assert_array_equal(gradients["inverse_mode_energy_sq"], np.zeros(4))


@pytest.mark.parametrize("inertia", [0.0, 0.03])
def test_scalar_slope_gradients_match_dynamic_small_energy_limit(backend, inertia):
    _, _, geometry = _network()
    kwargs = dict(
        chi0=0.4, gamma0=2.0, j_values={"J1": 0.15, "J2": -0.05}, inverse_mode_energy_sq=inertia
    )
    slope, gradients = sf.heisenberg_rpa_zero_energy_slope_and_gradients(geometry, **kwargs)
    energy = np.full(geometry.point_index.size, 1e-5)
    dynamic, dynamic_gradients = sf.heisenberg_rpa_chipp_and_gradients(geometry, energy, **kwargs)
    np.testing.assert_allclose(dynamic / energy, slope, rtol=1e-9)
    for name in ("chi0", "gamma0", "J1", "J2"):
        np.testing.assert_allclose(dynamic_gradients[name] / energy, gradients[name], rtol=1e-9)
    h = 1e-6
    for name in ("chi0", "gamma0", "J1", "J2"):
        plus, minus = kwargs.copy(), kwargs.copy()
        if name in kwargs["j_values"]:
            plus["j_values"] = {**kwargs["j_values"], name: kwargs["j_values"][name] + h}
            minus["j_values"] = {**kwargs["j_values"], name: kwargs["j_values"][name] - h}
        else:
            plus[name] += h
            minus[name] -= h
        difference = (
            sf.heisenberg_rpa_chipp(geometry, energy, **plus)
            - sf.heisenberg_rpa_chipp(geometry, energy, **minus)
        ) / (2 * h * energy)
        np.testing.assert_allclose(gradients[name], difference, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("inertia", [0.0, 0.03])
def test_tensor_slope_matches_isotropic_scalar_limit(backend, inertia):
    positions, orbits, geometry = _network()
    structure = build_tensor_structure(geometry, positions, orbits)
    q_hat = cartesian_qhat_per_point(geometry, np.eye(3))
    values = {"J1": 0.15, "J2": -0.05}
    kwargs = dict(chi0=0.4, gamma0=2.0, lambda_shift=0.12, inverse_mode_energy_sq=inertia)
    scalar = sf.heisenberg_rpa_zero_energy_slope(geometry, j_values=values, **kwargs)
    tensor = tensor_rpa_zero_energy_slope(structure, geometry, q_hat, param_values=values, **kwargs)
    np.testing.assert_allclose(tensor, 2.0 * scalar, rtol=1e-12)


@pytest.mark.parametrize("inertia", [0.0, 0.03])
def test_anisotropic_tensor_slope_matches_dynamic_limit(backend, inertia):
    structure, geometry, values, q_hat = _anisotropic_network()
    kwargs = dict(
        chi0=0.4, gamma0=2.0, param_values=values, lambda_shift=0.12, inverse_mode_energy_sq=inertia
    )
    slope = tensor_rpa_zero_energy_slope(structure, geometry, q_hat, **kwargs)
    energy = np.full(geometry.point_index.size, 1e-5)
    limit = tensor_rpa_unpolarized_chipp(structure, geometry, energy, q_hat, **kwargs) / energy
    np.testing.assert_allclose(slope, limit, rtol=2e-9, atol=1e-10)
    assert np.all(slope > 0.0)


@pytest.mark.parametrize("omega", [0.0, 1.3, -1.3])
def test_zeeman_local_derivative_matches_exact_circular_limit(omega):
    kwargs = dict(
        chi0=0.4, gamma0=2.0, omega_larmor=omega, chi_perp_ratio=1.7, gamma_perp_ratio=0.6
    )
    axis = np.array([0.0, 0.0, 3.0])
    derivative = zeeman_cartesian_propagator_zero_energy_derivative(axis, **kwargs)
    chi_perp, gamma_perp = 0.4 * 1.7, 2.0 * 0.6
    expected = np.diag([1j * chi_perp * gamma_perp / (gamma_perp**2 + omega**2)] * 2 + [0.4j / 2.0])
    expected[0, 1] = -1j * chi_perp * omega / (gamma_perp**2 + omega**2)
    expected[1, 0] = -expected[0, 1]
    np.testing.assert_allclose(derivative, expected, atol=1e-15)
    energy = np.array([-1e-5, 1e-5])
    response = zeeman_cartesian_propagator(energy, axis, **kwargs)
    np.testing.assert_allclose(
        (response[1] - response[0]) / (2e-5), derivative, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("omega", [0.0, 1.3, -1.3])
def test_zeeman_affine_coefficients_match_propagator_inverse_and_circular_sign(omega):
    kwargs = dict(
        chi0=0.4, gamma0=2.0, omega_larmor=omega, chi_perp_ratio=1.7, gamma_perp_ratio=0.6
    )
    axis = np.array([0.0, 0.0, 1.0])
    dynamic, mobility = zeeman_cartesian_affine_inverse(axis, **kwargs)
    np.testing.assert_allclose(dynamic @ mobility, np.eye(3), atol=1e-15)
    circular = np.array([1.0, 1j, 0.0])
    np.testing.assert_allclose(
        mobility @ circular, (0.4 * 1.7) * (2.0 * 0.6 + 1j * omega) * circular, atol=1e-15
    )
    energy = np.array([-0.7, 0.4, 1.1])
    inverse = np.linalg.inv(zeeman_cartesian_propagator(energy, axis, **kwargs))
    static_inverse = np.linalg.inv(zeeman_cartesian_propagator(np.zeros(1), axis, **kwargs)[0])
    np.testing.assert_allclose(
        inverse, static_inverse[None] - 1j * energy[:, None, None] * dynamic, rtol=1e-13, atol=1e-15
    )


@pytest.mark.parametrize("gamma", [2.0, 1e-6, 1e-9])
def test_zeeman_affine_coefficients_transform_covariantly_for_narrow_lines(gamma):
    kwargs = dict(
        chi0=0.7, gamma0=gamma, omega_larmor=3.0, chi_perp_ratio=1.3, gamma_perp_ratio=0.6
    )
    angle = 0.73
    rotation_y = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    angle = -0.37
    rotation_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    rotation = rotation_z @ rotation_y
    field_axis = rotation @ np.array([0.0, 0.0, 1.0])
    dynamic, mobility = zeeman_cartesian_affine_inverse(field_axis, **kwargs)
    original_d, original_m = zeeman_cartesian_affine_inverse(np.array([0, 0, 1]), **kwargs)
    np.testing.assert_allclose(dynamic, rotation @ original_d @ rotation.T, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(mobility, rotation @ original_m @ rotation.T, rtol=1e-14, atol=1e-14)
    # Mobility keeps the tiny dissipative coefficient separately from the
    # antisymmetric precession term, even after rotating the field axis.
    dissipative = 0.5 * (mobility + mobility.T)
    expected = gamma * 0.7 * np.outer(field_axis, field_axis) + gamma * 0.6 * 0.7 * 1.3 * (
        np.eye(3) - np.outer(field_axis, field_axis)
    )
    np.testing.assert_allclose(dissipative, expected, rtol=1e-6, atol=1e-16)


@pytest.mark.parametrize("omega", [0.0, 1.3])
@pytest.mark.parametrize("use_factor", [False, True])
def test_field_on_tensor_slope_matches_dynamic_noncommuting_limit(omega, use_factor):
    structure, geometry, values, q_hat = _anisotropic_network()
    axis = np.array([1.0, 2.0, 3.0])
    local_kwargs = dict(
        chi0=0.4, gamma0=2.0, omega_larmor=omega, chi_perp_ratio=1.7, gamma_perp_ratio=0.6
    )
    static = zeeman_cartesian_propagator(np.zeros(1), axis, **local_kwargs)[0]
    derivative = zeeman_cartesian_propagator_zero_energy_derivative(axis, **local_kwargs)
    factor = zeeman_cartesian_zero_energy_dissipative_factor(axis, **local_kwargs)
    slope = tensor_rpa_zeeman_zero_energy_slope(
        structure,
        geometry,
        q_hat,
        static,
        derivative,
        param_values=values,
        lambda_shift=0.12,
        local_dissipative_factor=factor if use_factor else None,
    )
    energy = np.full(geometry.point_index.size, 1e-5)
    response = zeeman_cartesian_propagator(energy, axis, **local_kwargs)
    limit = (
        tensor_rpa_zeeman_unpolarized_chipp(
            structure,
            geometry,
            energy,
            q_hat,
            response,
            param_values=values,
            lambda_shift=0.12,
            local_dissipative_factor=(
                zeeman_cartesian_dissipative_factor(energy, axis, **local_kwargs)
                if use_factor
                else None
            ),
            static_local_propagator=static,
        )
        / energy
    )
    np.testing.assert_allclose(slope, limit, rtol=2e-6, atol=1e-9)
    assert np.all(slope > 0.0)


@pytest.mark.parametrize("gamma", [1e-6, 1e-9])
def test_factorized_field_response_keeps_signed_narrow_weight_and_bose_continuity(gamma):
    from nfit.cross_section import KB_MEV_PER_K

    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    energy = np.array([-gamma, -gamma * 0.001, 0.0, gamma * 0.001, gamma])
    hkl = np.tile(axis, (energy.size, 1))
    geometry = sf.build_rpa_geometry(*hkl.T, [[0, 0, 0]], [])
    structure = TensorStructure(1, 1, ())
    kwargs = dict(chi0=0.7, gamma0=gamma, omega_larmor=3.0)
    local = zeeman_cartesian_propagator(energy, axis, **kwargs)
    factor = zeeman_cartesian_dissipative_factor(energy, axis, **kwargs)
    response = tensor_rpa_zeeman_unpolarized_chipp(
        structure,
        geometry,
        energy,
        hkl,
        local,
        param_values={},
        static_local_propagator=local[2],
        local_dissipative_factor=factor,
    )
    expected = (
        0.7
        * gamma
        * energy
        * (1.0 / (gamma**2 + (energy - 3.0) ** 2) + 1.0 / (gamma**2 + (energy + 3.0) ** 2))
    )
    np.testing.assert_allclose(response, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(response, -response[::-1], rtol=1e-12, atol=0.0)
    kt = 10.0 * KB_MEV_PER_K
    weighted = np.empty(energy.shape)
    nonzero = energy != 0.0
    weighted[nonzero] = response[nonzero] / -np.expm1(-energy[nonzero] / kt)
    weighted[~nonzero] = kt * 2 * 0.7 * gamma / (gamma**2 + 9.0)
    np.testing.assert_allclose(weighted, np.full(energy.shape, weighted[2]), rtol=1e-6)
    np.testing.assert_allclose(weighted[:2], np.exp(energy[:2] / kt) * weighted[:2:-1], rtol=1e-12)
    assert np.all(weighted > 0.0)


def test_factorized_field_response_matches_complex_susceptibility_for_anisotropic_exchange():
    structure, geometry, values, q_hat = _anisotropic_network()
    axis = np.array([1.0, 2.0, 3.0])
    energy = np.array([-0.7, 0.2, 0.5, 1.1])
    kwargs = dict(chi0=0.4, gamma0=2.0, omega_larmor=1.3, chi_perp_ratio=1.7, gamma_perp_ratio=0.6)
    local = zeeman_cartesian_propagator(energy, axis, **kwargs)
    static = zeeman_cartesian_propagator(np.zeros(1), axis, **kwargs)[0]
    factor = zeeman_cartesian_dissipative_factor(energy, axis, **kwargs)
    legacy = tensor_rpa_zeeman_unpolarized_chipp(
        structure,
        geometry,
        energy,
        q_hat,
        local,
        param_values=values,
        lambda_shift=0.12,
        static_local_propagator=static,
    )
    factored = tensor_rpa_zeeman_unpolarized_chipp(
        structure,
        geometry,
        energy,
        q_hat,
        local,
        param_values=values,
        lambda_shift=0.12,
        static_local_propagator=static,
        local_dissipative_factor=factor,
    )
    np.testing.assert_allclose(factored, legacy, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("gamma", [1e-3, 1e-6, 1e-9])
def test_factorized_zeeman_slope_retains_narrow_transverse_weight_parallel_to_field(gamma):
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    geometry = sf.build_rpa_geometry([axis[0]], [axis[1]], [axis[2]], [[0, 0, 0]], [])
    structure = TensorStructure(1, 1, ())
    kwargs = dict(chi0=0.7, gamma0=gamma, omega_larmor=3.0)
    static = zeeman_cartesian_propagator(np.zeros(1), axis, **kwargs)[0]
    derivative = zeeman_cartesian_propagator_zero_energy_derivative(axis, **kwargs)
    factor = zeeman_cartesian_zero_energy_dissipative_factor(axis, **kwargs)
    slope = tensor_rpa_zeeman_zero_energy_slope(
        structure,
        geometry,
        axis[None],
        static,
        derivative,
        param_values={},
        local_dissipative_factor=factor,
    )
    np.testing.assert_allclose(slope, [2 * 0.7 * gamma / (gamma**2 + 9.0)], rtol=1e-12)
    assert slope[0] > 0.0


def test_tensor_slope_projects_out_a_soft_mode_before_positive_summation(backend):
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    geometry = sf.build_rpa_geometry([0.0], [0.0], [0.0], [[0, 0, 0]], [])
    structure = TensorStructure(1, 1, ("J",), dense_terms={"J": np.outer(axis, axis)[None]})
    slope = tensor_rpa_zero_energy_slope(
        structure, geometry, axis[None], chi0=1.0, gamma0=1.0, param_values={"J": 1.0 - 1e-8}
    )
    # The soft longitudinal mode is removed by neutron polarization; the two
    # orthogonal modes each have the unchanged local slope1/meV^2.
    np.testing.assert_allclose(slope, [2.0], rtol=1e-12)


def test_zero_energy_slopes_enforce_static_stability_and_validate_derivative():
    geometry = sf.build_rpa_geometry([0.0], [0.0], [0.0], [[0, 0, 0]], [])
    q_hat = np.zeros((1, 3))
    structure = TensorStructure(1, 1, ("J",), dense_terms={"J": np.diag([0.6, 0, 0])[None]})
    static = np.diag([2.0, 2.0, 1.0]).astype(complex)
    derivative = 1j * static
    with pytest.raises(ValueError, match="RPA instability"):
        tensor_rpa_zeeman_zero_energy_slope(
            structure, geometry, q_hat, static, derivative, param_values={"J": 1.0}
        )
    with pytest.raises(ValueError, match="finite 3x3"):
        tensor_rpa_zeeman_zero_energy_slope(
            structure, geometry, q_hat, static, derivative * np.nan, param_values={"J": 0.0}
        )
    with pytest.raises(ValueError, match="RPA instability"):
        tensor_rpa_zero_energy_slope(
            structure, geometry, q_hat, chi0=2.0, gamma0=1.0, param_values={"J": 1.0}
        )
