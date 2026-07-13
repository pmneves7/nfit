import numpy as np
import pytest

from nfit.cross_section import KB_MEV_PER_K
from nfit.spin_fluctuations import build_rpa_geometry, rpa_exchange_matrix
from nfit.sum_rules import (
    bz_sample_hkl,
    coth_weight,
    kk_static_chi,
    lorentzian_moment,
    mode_amplitude_per_site,
    static_chi_modes,
    tier_b_omega_grid,
    trace_moment_quadrature,
)


def _quad_reference(chi, gamma, temperature_K, cutoff):
    """Adaptive quadrature referee, split at the narrow coth feature ~ k_B T."""
    from scipy.integrate import quad

    kt = KB_MEV_PER_K * temperature_K

    def integrand(w):
        chipp = chi * gamma * w / (w * w + gamma * gamma)
        if temperature_K <= 0:
            return chipp / np.pi
        return chipp / np.tanh(w / (2.0 * kt)) / np.pi

    if temperature_K <= 0 or 50.0 * kt >= cutoff:
        value, _err = quad(integrand, 0.0, cutoff, limit=400)
        return value
    v1, _ = quad(integrand, 0.0, 50.0 * kt, limit=400)
    v2, _ = quad(integrand, 50.0 * kt, cutoff, limit=400)
    return v1 + v2


@pytest.mark.parametrize("temperature_K", [0.0, 0.5, 5.0, 77.0, 300.0, 1500.0])
@pytest.mark.parametrize("gamma", [0.05, 1.0, 12.0])
def test_lorentzian_moment_matches_adaptive_quadrature(temperature_K, gamma):
    """The digamma + tail closed form is locked to adaptive quadrature."""
    chi, cutoff = 0.7, 100.0
    closed = float(lorentzian_moment(chi, gamma, temperature_K, cutoff))
    reference = _quad_reference(chi, gamma, temperature_K, cutoff)
    assert closed == pytest.approx(reference, rel=1e-8)


def test_lorentzian_moment_finite_cutoff_tail_matters():
    """At Lambda ~ few kT the tail correction is significant and correct."""
    chi, gamma, cutoff = 0.4, 2.0, 20.0
    temperature_K = 700.0  # kT ~ 60 meV, comparable to the cutoff
    closed = float(lorentzian_moment(chi, gamma, temperature_K, cutoff))
    reference = _quad_reference(chi, gamma, temperature_K, cutoff)
    assert closed == pytest.approx(reference, rel=1e-7)


def test_classical_limit_is_equipartition():
    """kT >> Lambda >> Gamma: <m^2> -> chi * kT * (2/pi) arctan(Lambda/Gamma)."""
    chi, gamma, cutoff = 0.3, 0.5, 50.0
    temperature_K = 2.0e5  # kT ~ 17 eV >> cutoff
    kt = KB_MEV_PER_K * temperature_K
    expected = chi * kt * (2.0 / np.pi) * np.arctan(cutoff / gamma)
    value = float(lorentzian_moment(chi, gamma, temperature_K, cutoff))
    assert value == pytest.approx(expected, rel=1e-3)


def test_moment_parts_sum_and_zero_point_is_t_independent():
    chi, gamma, cutoff = 0.7, 3.0, 80.0
    zp_cold, th_cold = lorentzian_moment(chi, gamma, 1.0, cutoff, parts=True)
    zp_hot, th_hot = lorentzian_moment(chi, gamma, 200.0, cutoff, parts=True)
    assert float(zp_cold) == pytest.approx(float(zp_hot), rel=1e-14)
    assert float(th_hot) > float(th_cold) > 0.0
    total = float(lorentzian_moment(chi, gamma, 200.0, cutoff))
    assert total == pytest.approx(float(zp_hot) + float(th_hot), rel=1e-14)


def test_coth_weight_limits():
    omega = np.array([0.001, 1.0, 50.0])
    assert np.allclose(coth_weight(omega, 0.0), 1.0)
    kt = KB_MEV_PER_K * 30.0
    weight = coth_weight(omega, 30.0)
    assert weight[0] == pytest.approx(2.0 * kt / omega[0], rel=1e-6)
    assert weight[2] == pytest.approx(1.0, rel=1e-6)


def test_trace_moment_quadrature_matches_lorentzian():
    """The Tier-B grid integral reproduces the closed form on a Lorentzian."""
    chi, gamma, cutoff, temperature_K = 0.6, 2.5, 100.0, 30.0
    omega = tier_b_omega_grid(cutoff, n_points=4000)
    chipp = chi * gamma * omega / (omega**2 + gamma**2)
    value = float(trace_moment_quadrature(chipp, omega, temperature_K))
    closed = float(lorentzian_moment(chi, gamma, temperature_K, cutoff))
    # The grid integral misses the [0, omega_min] sliver; it is tiny.
    assert value == pytest.approx(closed, rel=5e-4)


def test_kk_static_chi_recovers_lorentzian_amplitude():
    chi, gamma = 0.8, 1.2
    omega = tier_b_omega_grid(2000.0, n_points=8000)
    chipp = chi * gamma * omega / (omega**2 + gamma**2)
    static = float(kk_static_chi(chipp, omega))
    assert static == pytest.approx(chi, rel=5e-3)


def test_bz_sample_grid_is_uniform_and_gamma_free():
    grid = bz_sample_hkl(4)
    assert grid.shape == (64, 3)
    assert np.all(grid > 0.0) and np.all(grid < 1.0)
    # No point sits on Gamma or the zone boundary.
    assert not np.any(np.all(np.abs(grid - np.round(grid)) < 1e-12, axis=1))
    with pytest.raises(ValueError):
        bz_sample_hkl(0)


PYROCHLORE = {
    "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "spacegroup": "F d -3 m:2",
    "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "Yb3"}],
}


def _pyrochlore_networks():
    pytest.importorskip("gemmi")
    from nfit.crystal import generate_bond_orbits, orbits_to_config, sites_to_config
    from nfit.spin_fluctuations import reduce_site_network

    a = PYROCHLORE["lattice"]["a"]
    nn = a * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(PYROCHLORE, ["M1"], cutoff_angstrom=nn + 0.01)
    positions = sites_to_config(sites)
    orbit_config = orbits_to_config(orbits)
    reduced_positions, reduced_orbits = reduce_site_network(positions, orbit_config)
    assert np.asarray(reduced_positions).shape == (4, 3)
    return (positions, orbit_config), (reduced_positions, reduced_orbits)


def _bz_eigenvalues(positions, orbits, j_values, n):
    grid = bz_sample_hkl(n)
    geometry = build_rpa_geometry(grid[:, 0], grid[:, 1], grid[:, 2], positions, orbits)
    exchange = rpa_exchange_matrix(geometry, j_values)
    return np.linalg.eigvalsh(exchange)


def test_bz_moment_integral_invariant_under_pyrochlore_reduction():
    """16-site conventional and 4-site primitive networks give the same <m^2>.

    Uniform sampling of the conventional reciprocal torus is uniform on the
    primitive torus, and the site-traced amplitude is cell-independent, so
    the BZ moment integral must agree between the descriptions.
    """
    full, reduced = _pyrochlore_networks()
    j_values = {"J1": 0.08}
    kwargs = dict(
        chi0=0.9, gamma0=3.0, temperature_K=10.0, cutoff_mev=60.0, lambda_shift=0.05
    )

    lam_full = _bz_eigenvalues(full[0], full[1], j_values, n=6)
    lam_reduced = _bz_eigenvalues(reduced[0], reduced[1], j_values, n=6)
    m2_full = mode_amplitude_per_site(lam_full, n_sites=16, **kwargs)
    m2_reduced = mode_amplitude_per_site(lam_reduced, n_sites=4, **kwargs)
    assert m2_reduced == pytest.approx(m2_full, rel=1e-10)


def test_mode_amplitude_instability_raises():
    lam = np.array([[0.5, -0.2], [0.9, 0.1]])
    with pytest.raises(ValueError, match="instability"):
        mode_amplitude_per_site(
            lam, chi0=2.0, gamma0=1.0, temperature_K=5.0, cutoff_mev=50.0, n_sites=2
        )
    # The Onsager shift restores stability for the same parameters.
    value = mode_amplitude_per_site(
        lam,
        chi0=2.0,
        gamma0=1.0,
        temperature_K=5.0,
        cutoff_mev=50.0,
        n_sites=2,
        lambda_shift=0.6,
    )
    assert np.isfinite(value) and value > 0.0


def test_mode_amplitude_decreases_with_onsager_shift():
    """Larger lambda_shift moves modes away from instability => smaller <m^2>."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(-0.5, 0.4, size=(200, 4))
    kwargs = dict(chi0=1.5, gamma0=2.0, temperature_K=20.0, cutoff_mev=80.0, n_sites=4)
    values = [
        mode_amplitude_per_site(lam, lambda_shift=shift, **kwargs)
        for shift in (0.0, 0.2, 0.5)
    ]
    assert values[0] > values[1] > values[2] > 0.0


def test_static_chi_modes_matches_uniform_mode():
    """Single-site FM chain: static chi at Q=0 equals chi0/(1 - J(0) chi0)."""
    positions = [[0.0, 0.0, 0.0]]
    orbits = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}
    ]
    geometry = build_rpa_geometry([0.0], [0.0], [0.0], positions, orbits)
    exchange = rpa_exchange_matrix(geometry, {"J1": 0.1})
    lam, modes = np.linalg.eigh(exchange)
    weights = np.abs(modes.sum(axis=1)) ** 2 / 1
    static = static_chi_modes(lam, weights, chi0=0.5)
    expected = 0.5 / (1.0 - 0.2 * 0.5)  # J(0) = 2 J1 (bond + h.c.)
    assert static[0] == pytest.approx(expected, rel=1e-12)
