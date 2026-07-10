import numpy as np
import pytest

from metallix.spin_fluctuations import (
    RpaGeometry,
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    local_relaxational_chipp,
    mmp_chipp,
    rpa_exchange_matrix,
)


def _chain_geometry(H, offsets=((1, 0, 0),)):
    orbits = [
        {
            "label": "J1",
            "bonds": [
                {"site_i": 0, "site_j": 0, "offset": list(offset)} for offset in offsets
            ],
        }
    ]
    zeros = np.zeros_like(np.asarray(H, dtype=float))
    return build_rpa_geometry(H, zeros, zeros, [[0.0, 0.0, 0.0]], orbits)


def _two_site_geometry(rng, n_points=7):
    hkl = rng.uniform(-2.0, 2.0, size=(n_points, 3))
    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]]
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
            "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}],
        },
    ]
    geometry = build_rpa_geometry(hkl[:, 0], hkl[:, 1], hkl[:, 2], positions, orbits)
    return geometry


def test_local_relaxational_is_odd_and_peaks_at_gamma():
    E = np.linspace(-10.0, 10.0, 201)
    chipp = local_relaxational_chipp(E, chi_loc=2.0, gamma=3.0)
    np.testing.assert_allclose(chipp, -chipp[::-1], atol=1e-14)
    peak = E[np.argmax(chipp)]
    assert peak == pytest.approx(3.0, abs=0.11)
    np.testing.assert_allclose(
        local_relaxational_chipp(3.0, chi_loc=2.0, gamma=3.0), 2.0 * 0.5 * 3.0 / 3.0
    )


def test_mmp_chipp_is_odd_in_energy_and_peaks_at_q0():
    E = np.linspace(-5.0, 5.0, 101)
    chipp = mmp_chipp(0.25, E, chi_pk=1.0, xi=2.0, omega_sf=1.5)
    np.testing.assert_allclose(chipp, -chipp[::-1], atol=1e-14)

    q_sq = np.array([0.0, 0.5, 2.0])
    at_q0 = mmp_chipp(q_sq, 1.0, chi_pk=1.0, xi=2.0, omega_sf=1.0)
    assert np.all(np.diff(at_q0) < 0.0)
    # at Q0 the lineshape is relaxational with Gamma = omega_sf
    E = np.linspace(-8.0, 8.0, 161)
    np.testing.assert_allclose(
        mmp_chipp(0.0, E, chi_pk=2.0, xi=3.0, omega_sf=1.7),
        local_relaxational_chipp(E, chi_loc=2.0, gamma=1.7),
        rtol=1e-12,
    )


def test_mmp_requires_positive_omega_sf():
    with pytest.raises(ValueError):
        mmp_chipp(0.0, 1.0, chi_pk=1.0, xi=1.0, omega_sf=0.0)


def test_rpa_weights_sum_rule_two_site_random_crystal():
    rng = np.random.default_rng(7)
    geometry = _two_site_geometry(rng)
    exchange = rpa_exchange_matrix(geometry, {"J1": 1.3, "J2": -0.4})
    # Hermiticity of J(Q)
    np.testing.assert_allclose(exchange, np.conj(np.transpose(exchange, (0, 2, 1))), atol=1e-12)
    _, modes = np.linalg.eigh(exchange)
    amplitudes = np.einsum("qa,qan->qn", geometry.site_phases, modes)
    weights = np.abs(amplitudes) ** 2 / geometry.n_sites
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)


def test_rpa_with_zero_exchange_reduces_to_local_relaxational():
    rng = np.random.default_rng(3)
    geometry = _two_site_geometry(rng)
    E = np.linspace(-4.0, 6.0, geometry.point_index.size)
    coupled = heisenberg_rpa_chipp(
        geometry, E, chi0=1.5, gamma0=2.5, j_values={"J1": 0.0, "J2": 0.0}
    )
    local = local_relaxational_chipp(E, chi_loc=1.5, gamma=2.5)
    np.testing.assert_allclose(coupled, local, rtol=1e-12)


def test_rpa_chain_matches_closed_form_dispersion():
    H = np.linspace(0.0, 1.0, 21)
    E = np.full(H.shape, 1.3)
    chi0, gamma0, J = 0.8, 2.0, 0.3
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(geometry, E, chi0=chi0, gamma0=gamma0, j_values={"J1": J})

    denom = 1.0 - 2.0 * J * chi0 * np.cos(2.0 * np.pi * H)
    chi_q = chi0 / denom
    gamma_q = gamma0 * denom
    expected = chi_q * gamma_q * E / (E**2 + gamma_q**2)
    np.testing.assert_allclose(chipp, expected, rtol=1e-12)


def test_rpa_is_odd_in_energy():
    H = np.array([0.1, 0.1, 0.4, 0.4])
    E = np.array([1.0, -1.0, 2.5, -2.5])
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(geometry, E, chi0=0.5, gamma0=1.0, j_values={"J1": 0.2})
    np.testing.assert_allclose(chipp[0], -chipp[1], rtol=1e-12)
    np.testing.assert_allclose(chipp[2], -chipp[3], rtol=1e-12)


def test_rpa_instability_raises():
    geometry = _chain_geometry(np.array([0.0]))
    with pytest.raises(ValueError, match="instability"):
        heisenberg_rpa_chipp(
            geometry, np.array([1.0]), chi0=1.0, gamma0=1.0, j_values={"J1": 0.5}
        )


def test_rpa_relaxation_softens_at_ordering_vector():
    # antiferromagnetic chain: J < 0 favors q = 0.5; the mode weight sum
    # rule means chi'' at small E is largest where Gamma(q) is smallest
    H = np.array([0.0, 0.25, 0.5])
    E = np.full(H.shape, 0.05)
    geometry = _chain_geometry(H)
    chipp = heisenberg_rpa_chipp(
        geometry, E, chi0=0.9, gamma0=2.0, j_values={"J1": -0.5}
    )
    assert chipp[2] > chipp[1] > chipp[0]


def test_geometry_deduplicates_repeated_q_points():
    H = np.array([0.2, 0.2, 0.2, 0.7])
    geometry = _chain_geometry(H)
    assert geometry.n_q == 2
    assert geometry.point_index.size == 4


def test_geometry_validates_inputs():
    with pytest.raises(ValueError, match="site index"):
        build_rpa_geometry(
            [0.0],
            [0.0],
            [0.0],
            [[0.0, 0.0, 0.0]],
            [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 5, "offset": [0, 0, 0]}]}],
        )
    with pytest.raises(ValueError, match="no bonds"):
        build_rpa_geometry([0.0], [0.0], [0.0], [[0.0, 0.0, 0.0]], [{"label": "J1", "bonds": []}])
    geometry = _chain_geometry(np.array([0.1]))
    with pytest.raises(KeyError, match="J9"):
        rpa_exchange_matrix(geometry, {"J9": 1.0})
    with pytest.raises(ValueError, match="gamma0"):
        heisenberg_rpa_chipp(geometry, [1.0], chi0=1.0, gamma0=0.0, j_values={"J1": 0.0})
    with pytest.raises(ValueError, match="E has"):
        heisenberg_rpa_chipp(geometry, [1.0, 2.0], chi0=1.0, gamma0=1.0, j_values={"J1": 0.0})
