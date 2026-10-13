"""Field-on moments must resolve soft modes and sharp resonances independently of a grid."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

import nfit.closures as closures
from nfit.closures import TierAMoments, TierBMoments, solve_onsager, solve_scr, solve_tac
from nfit.cross_section import KB_MEV_PER_K
from nfit.sum_rules import (
    MomentIntegrationError,
    lorentzian_moment,
    tier_b_omega_grid,
    trace_moment_quadrature,
)
from nfit.tensor_rpa import zeeman_cartesian_affine_inverse, zeeman_cartesian_propagator


def _model(field=0.0, exchange=None, *, omega_points=200, direction=(0, 0, 1)):
    def builder(energy, chi0, gamma0):
        return zeeman_cartesian_propagator(
            energy, direction, chi0=chi0, gamma0=gamma0, omega_larmor=field,
        )

    def coefficients(chi0, gamma0):
        return zeeman_cartesian_affine_inverse(direction, chi0=chi0, gamma0=gamma0, omega_larmor=field)

    if exchange is None:
        exchange = np.zeros((1, 3, 3), complex)
    return TierBMoments(
        exchange, exchange.shape[-1] // 3, builder, omega_points=omega_points,
        affine_inverse_builder=coefficients,
    )


@pytest.mark.parametrize("gamma", [1.0, 1e-4, 1e-6, 1e-9])
@pytest.mark.parametrize("temperature", [0.0, 0.1, 10.0, 1e5])
@pytest.mark.parametrize("cutoff", [1e-5, 0.1, 100.0])
def test_zero_field_matches_analytic_moment_without_an_energy_floor(gamma, temperature, cutoff):
    model = _model(omega_points=16)
    actual = model.moment(chi0=0.7, gamma0=gamma, temperature_K=temperature, cutoff_mev=cutoff)
    expected = lorentzian_moment(0.7, gamma, temperature, cutoff, parts=True)
    np.testing.assert_allclose(actual, np.asarray(expected) * 3, rtol=3e-11, atol=3e-13)


def test_zero_field_tensor_soft_modes_match_tier_a():
    exchange = np.diag([2.5 * (1 - 1e-8), -0.3, 0.2])[None].astype(complex)
    tier_a = TierAMoments(np.linalg.eigvalsh(exchange), 1, isotropic_components=1)
    tier_b = _model(exchange=exchange, omega_points=16)
    kwargs = dict(chi0=0.4, gamma0=1e-4, temperature_K=10.0, cutoff_mev=0.1)
    np.testing.assert_allclose(tier_b.moment(**kwargs), tier_a.moment(**kwargs), rtol=3e-8)


@pytest.mark.parametrize("cutoff", [1e-7, 1e-5, 0.1, 3.0, 10.0])
def test_narrow_field_zero_point_has_exact_finite_cutoff(cutoff):
    chi, gamma, field = 0.7, 1e-4, 3.0
    model = _model(field)

    def primitive(center):
        return chi * gamma * (
            0.5 * np.log(((cutoff - center)**2 + gamma**2) / (center**2 + gamma**2))
            + center / gamma * (np.arctan((cutoff - center) / gamma) - np.arctan(-center / gamma))
        ) / np.pi

    expected = sum(primitive(center) for center in (0.0, field, -field))
    zp, thermal = model.moment(chi0=chi, gamma0=gamma, temperature_K=0.0, cutoff_mev=cutoff)
    assert zp == pytest.approx(expected, rel=2e-8, abs=1e-14)
    assert thermal == 0.0


def test_ultranarrow_peak_at_exact_cutoff_reports_unresolved_precision():
    with pytest.raises(MomentIntegrationError, match="did not converge"):
        _model(3).moment(chi0=0.7, gamma0=1e-8, temperature_K=0, cutoff_mev=3)


@pytest.mark.parametrize("gamma", [1e-6, 1e-9])
def test_narrow_rotated_field_has_the_same_integrated_moment(gamma):
    kwargs = dict(chi0=0.7, gamma0=gamma, temperature_K=10, cutoff_mev=10)
    reference = _model(3).moment(**kwargs)
    rotated = _model(3, direction=(1, 2, 3)).moment(**kwargs)
    np.testing.assert_allclose(rotated, reference, rtol=5e-12, atol=2e-13)


@pytest.mark.parametrize("temperature", [1.0, 10.0, 100.0])
def test_narrow_field_moment_approaches_delta_peak_and_equipartition(temperature):
    chi, gamma, field = 0.7, 1e-10, 3.0
    zp, thermal = _model(field, omega_points=16).moment(
        chi0=chi, gamma0=gamma, temperature_K=temperature, cutoff_mev=10.0,
    )
    kt = KB_MEV_PER_K * temperature
    assert zp == pytest.approx(chi * field, rel=1e-8)
    expected_thermal = chi * kt + 2 * chi * field / np.expm1(field / kt)
    assert thermal == pytest.approx(expected_thermal, rel=1e-8)


def test_nonreciprocal_trace_is_integrated_without_symmetrizing_momentum():
    exchange = np.array([[[0.1, 0.2j, 0], [-0.2j, -0.15, 0.08], [0, 0.08, 0.05]]])
    model = _model(1.1, exchange)
    chi, gamma, temperature, cutoff = 0.7, 0.3, 8.0, 2.5

    def direct(energy, thermal):
        x0 = zeeman_cartesian_propagator(
            np.array([energy]), [0, 0, 1], chi0=chi, gamma0=gamma, omega_larmor=1.1,
        )[0]
        response = np.linalg.solve(np.eye(3) - x0 @ exchange[0], x0)
        spectral = np.trace(response).imag / np.pi
        return spectral * (2 / np.expm1(energy / (KB_MEV_PER_K * temperature)) if thermal else 1)

    expected = [quad(lambda e, thermal=thermal: direct(e, thermal), 0, cutoff, epsabs=1e-11, epsrel=1e-11)[0] for thermal in (False, True)]
    np.testing.assert_allclose(
        model.moment(chi0=chi, gamma0=gamma, temperature_K=temperature, cutoff_mev=cutoff),
        expected, rtol=5e-12, atol=3e-13,
    )


@pytest.mark.parametrize("epsilon", [0.0, 1e-10])
def test_near_defective_stable_poles_use_checked_direct_solves(epsilon, monkeypatch):
    exchange = np.diag([0.0, -3.0, -1.0])[None].astype(complex)
    model = _model(0.75 * (1 + epsilon), exchange)
    calls = []
    fallback = model._adaptive_moment

    def observed(*args, **kwargs):
        calls.append(True)
        return fallback(*args, **kwargs)

    monkeypatch.setattr(model, "_adaptive_moment", observed)
    actual = model.moment(chi0=1, gamma0=1, temperature_K=100, cutoff_mev=10)
    expected = []
    for thermal in (False, True):
        def integrand(energy, thermal=thermal):
            x0 = zeeman_cartesian_propagator(np.array([energy]), [0, 0, 1], chi0=1, gamma0=1, omega_larmor=0.75 * (1 + epsilon))[0]
            value = np.trace(np.linalg.solve(np.eye(3) - x0 @ exchange[0], x0)).imag / np.pi
            return value * (2 / np.expm1(energy / (100 * KB_MEV_PER_K)) if thermal else 1)
        expected.append(quad(integrand, 0, 10, epsabs=1e-11, epsrel=1e-11)[0])
    assert calls
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-13)


def test_nonaffine_callable_propagator_retains_analytic_mixture_moment():
    def builder(energy, chi0, gamma0):
        energy = np.asarray(energy)
        scalar = chi0 * (0.7 / (1 - 1j * energy / gamma0) + 0.3 / (1 - 1j * energy / (4 * gamma0)))
        return scalar[:, None, None] * np.eye(3)[None]

    model = TierBMoments(np.zeros((1, 3, 3), complex), 1, builder)
    actual = model.moment(chi0=0.7, gamma0=0.1, temperature_K=10, cutoff_mev=10)
    expected = np.asarray(lorentzian_moment(0.49, 0.1, 10, 10, parts=True)) * 3
    expected += np.asarray(lorentzian_moment(0.21, 0.4, 10, 10, parts=True)) * 3
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=1e-13)


def test_small_nonaffine_correction_is_not_discarded():
    inertia = 5e-12

    def builder(energy, chi0, gamma0):
        scalar = chi0 / (1 - inertia * np.asarray(energy)**2 - 1j * np.asarray(energy) / gamma0)
        return scalar[:, None, None] * np.eye(3)[None]

    model = TierBMoments(np.zeros((1, 3, 3), complex), 1, builder)
    assert model._affine_inverse(1, 1, 10, np.eye(3)) is None
    actual = model.moment(chi0=1, gamma0=1, temperature_K=0, cutoff_mev=10)[0]
    expected = quad(lambda e: (3 / (1 - inertia * e**2 - 1j * e)).imag / np.pi, 0, 10, epsabs=1e-12, epsrel=1e-12)[0]
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)


def test_adaptive_failure_propagates_through_closure_bracketing(monkeypatch):
    model = _model()
    monkeypatch.setattr(model, "_pole_moment", lambda *args: None)
    monkeypatch.setattr(closures, "quad_vec", lambda *args, **kwargs: (np.ones(2), 1.0, SimpleNamespace(success=False)))
    with pytest.raises(MomentIntegrationError, match="did not converge"):
        solve_onsager(model, chi0=0.7, gamma0=1, temperature_K=10, cutoff_mev=10, target=1)


@pytest.mark.parametrize("solver", ["onsager", "scr", "tac"])
def test_narrow_mode_closure_roots_match_the_analytic_limit(solver):
    tier_b = _model(omega_points=16)
    tier_a = TierAMoments(np.zeros((1, 1)), 1)
    kwargs = dict(gamma0=1e-6, temperature_K=10.0, cutoff_mev=0.1)
    def call(model):
        if solver == "onsager":
            return solve_onsager(model, chi0=0.7, target=1, **kwargs)
        if solver == "scr":
            return solve_scr(model, chi0_bare=0.7, mode_coupling_u=0.2, **kwargs)
        return solve_tac(model, total_amplitude=1, guess=0.7, **kwargs)
    actual, expected = call(tier_b), call(tier_a)
    assert actual.lambda_shift == pytest.approx(expected.lambda_shift, rel=2e-10, abs=1e-11)
    assert actual.chi0_eff == pytest.approx(expected.chi0_eff, rel=2e-10)
    assert actual.m2_total == pytest.approx(expected.m2_total, rel=2e-10)


@pytest.mark.parametrize("cutoff", [1e-3, 1e-4, 1e-5, 1e-12, 100])
def test_reference_grid_covers_zero_and_small_cutoffs_in_order(cutoff):
    grid = tier_b_omega_grid(cutoff, 200)
    assert grid[0] == 0 and grid[-1] == cutoff
    assert grid.size == 200 and np.all(np.diff(grid) > 0)


def test_sampled_quadrature_accepts_exact_zero_energy_slope():
    grid = np.linspace(0, 0.01, 2000)
    chi, gamma, temperature = 0.7, 0.4, 10
    spectral = chi * gamma * grid / (grid**2 + gamma**2)
    actual = trace_moment_quadrature(spectral, grid, temperature, zero_energy_slope=chi / gamma)
    assert actual == pytest.approx(float(lorentzian_moment(chi, gamma, temperature, 0.01)), rel=1e-9)
