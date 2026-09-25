import numpy as np
import pytest

from nfit.closures import (
    ClosureSpec,
    TierAMoments,
    TierBMoments,
    solve_closure,
    solve_onsager,
    solve_onsager_temperatures,
    solve_scr,
    solve_scr_temperatures,
    solve_tac,
    solve_tac_temperatures,
)
from nfit.spin_fluctuations import build_rpa_geometry, rpa_exchange_matrix
from nfit.sum_rules import bz_sample_hkl


def _scalar_tier_a(j1=0.1, n=6):
    """Two-sublattice toy network eigen-factored on a BZ grid."""
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    orbits = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}
    ]
    grid = bz_sample_hkl(n)
    geometry = build_rpa_geometry(grid[:, 0], grid[:, 1], grid[:, 2], positions, orbits)
    exchange = rpa_exchange_matrix(geometry, {"J1": j1})
    eigenvalues = np.linalg.eigvalsh(exchange)
    return TierAMoments(eigenvalues, n_sites=2)


KWARGS = dict(chi0=0.8, gamma0=2.0, temperature_K=20.0, cutoff_mev=80.0)


def test_onsager_hits_target_and_is_monotone():
    model = _scalar_tier_a()
    result = solve_onsager(model, target=1.0, **KWARGS)
    assert result.m2_total == pytest.approx(1.0, abs=1e-9)
    assert result.chi0_eff == KWARGS["chi0"]
    # Verify against a direct recomputation at the solved shift.
    zp, th = model.moment(lambda_shift=result.lambda_shift, **KWARGS)
    assert zp + th == pytest.approx(1.0, abs=1e-9)
    assert result.m2_zero_point == pytest.approx(zp)
    # Smaller targets require larger stabilizing shifts.
    smaller = solve_onsager(model, target=0.25, **KWARGS)
    assert smaller.lambda_shift > result.lambda_shift


def test_onsager_temperature_raises_lambda():
    """More thermal amplitude at fixed budget forces a larger reaction field."""
    model = _scalar_tier_a()
    cold = solve_onsager(model, target=2.0, **{**KWARGS, "temperature_K": 5.0})
    hot = solve_onsager(model, target=2.0, **{**KWARGS, "temperature_K": 150.0})
    assert hot.lambda_shift > cold.lambda_shift
    assert hot.m2_thermal > cold.m2_thermal
    assert hot.m2_total == pytest.approx(cold.m2_total, abs=1e-9)


def test_onsager_temperature_continuation_matches_scalar_solver():
    model = _scalar_tier_a(n=8)
    temperatures = np.array([300.0, 5.0, 80.0, 20.0, 150.0])
    batch = solve_onsager_temperatures(
        model,
        chi0=0.8,
        gamma0=2.0,
        temperature_K=temperatures,
        cutoff_mev=80.0,
        target=1.2,
    )
    scalar = [
        solve_onsager(
            model,
            chi0=0.8,
            gamma0=2.0,
            temperature_K=float(temperature),
            cutoff_mev=80.0,
            target=1.2,
        )
        for temperature in temperatures
    ]
    np.testing.assert_allclose(
        [result.lambda_shift for result in batch],
        [result.lambda_shift for result in scalar],
        rtol=2e-10,
        atol=2e-12,
    )


def test_onsager_unreachable_target_raises():
    model = _scalar_tier_a()
    with pytest.raises(ValueError, match="unreachable"):
        solve_onsager(model, target=1e18, **KWARGS)
    with pytest.raises(ValueError, match="positive"):
        solve_onsager(model, target=-1.0, **KWARGS)


def test_scr_reduces_to_bare_rpa_at_zero_coupling():
    model = _scalar_tier_a()
    result = solve_scr(model, chi0_bare=0.8, gamma0=2.0, temperature_K=30.0,
                       cutoff_mev=80.0, mode_coupling_u=0.0)
    assert result.chi0_eff == pytest.approx(0.8, rel=1e-10)
    assert result.lambda_shift == 0.0


def test_scr_suppresses_chi_and_obeys_equation():
    model = _scalar_tier_a()
    u = 0.3
    result = solve_scr(model, chi0_bare=0.8, gamma0=2.0, temperature_K=30.0,
                       cutoff_mev=80.0, mode_coupling_u=u)
    assert 0 < result.chi0_eff < 0.8
    residual = 1.0 / result.chi0_eff - 1.0 / 0.8 - u * result.m2_total
    assert residual == pytest.approx(0.0, abs=1e-9)
    # Curie-Weiss mechanism: hotter => more amplitude => smaller chi0_eff.
    hot = solve_scr(model, chi0_bare=0.8, gamma0=2.0, temperature_K=300.0,
                    cutoff_mev=80.0, mode_coupling_u=u)
    assert hot.chi0_eff < result.chi0_eff


def test_scr_temperature_continuation_matches_scalar_solver():
    model = _scalar_tier_a(n=8)
    temperatures = np.array([300.0, 5.0, 80.0, 20.0, 150.0])
    batch = solve_scr_temperatures(
        model,
        chi0_bare=0.8,
        gamma0=2.0,
        temperature_K=temperatures,
        cutoff_mev=80.0,
        mode_coupling_u=0.3,
    )
    scalar = [
        solve_scr(
            model,
            chi0_bare=0.8,
            gamma0=2.0,
            temperature_K=float(temperature),
            cutoff_mev=80.0,
            mode_coupling_u=0.3,
        )
        for temperature in temperatures
    ]
    np.testing.assert_allclose(
        [result.chi0_eff for result in batch],
        [result.chi0_eff for result in scalar],
        rtol=2e-10,
        atol=2e-12,
    )


def test_tac_conserves_total_amplitude():
    model = _scalar_tier_a()
    budget = 1.5
    cold = solve_tac(model, gamma0=2.0, temperature_K=5.0, cutoff_mev=80.0,
                     total_amplitude=budget, guess=0.5)
    hot = solve_tac(model, gamma0=2.0, temperature_K=200.0, cutoff_mev=80.0,
                    total_amplitude=budget, guess=0.5)
    assert cold.m2_total == pytest.approx(budget, abs=1e-9)
    assert hot.m2_total == pytest.approx(budget, abs=1e-9)
    # Thermal weight grows with T, so the zero-point share (and chi0_eff) drop.
    assert hot.m2_thermal > cold.m2_thermal
    assert hot.chi0_eff < cold.chi0_eff


def test_tac_temperature_continuation_matches_scalar_solver():
    model = _scalar_tier_a(n=8)
    temperatures = np.array([300.0, 5.0, 80.0, 20.0, 150.0])
    batch = solve_tac_temperatures(
        model,
        gamma0=2.0,
        temperature_K=temperatures,
        cutoff_mev=80.0,
        total_amplitude=1.2,
        guess=0.5,
    )
    scalar = [
        solve_tac(
            model,
            gamma0=2.0,
            temperature_K=float(temperature),
            cutoff_mev=80.0,
            total_amplitude=1.2,
            guess=0.5,
        )
        for temperature in temperatures
    ]
    np.testing.assert_allclose(
        [result.chi0_eff for result in batch],
        [result.chi0_eff for result in scalar],
        rtol=2e-10,
        atol=2e-12,
    )


def test_closure_spec_parsing_and_parameters():
    assert ClosureSpec.from_config(None) is None
    assert ClosureSpec.from_config({}) is None
    assert ClosureSpec.from_config({"closure": {"mode": "none"}}) is None
    spec = ClosureSpec.from_config(
        {"closure": {"mode": "onsager", "moment_mode": "fitted", "bz_grid": 8}}
    )
    assert spec.mode == "onsager" and spec.bz_grid == 8
    assert spec.parameter_names() == ("m2_total",)
    fixed = ClosureSpec.from_config({"closure": {"mode": "onsager"}})
    assert fixed.parameter_names() == ()
    assert ClosureSpec.from_config({"closure": {"mode": "scr"}}).parameter_names() == (
        "mode_coupling_u",
    )
    assert ClosureSpec.from_config(
        {"closure": {"mode": "tac", "moment_mode": "fitted"}}
    ).parameter_names() == ("total_amplitude",)
    with pytest.raises(ValueError, match="unknown closure mode"):
        ClosureSpec.from_config({"closure": {"mode": "bogus"}})
    with pytest.raises(ValueError, match="moment_mode"):
        ClosureSpec.from_config({"closure": {"mode": "tac", "moment_mode": "maybe"}})
    with pytest.raises(ValueError, match="cutoff"):
        ClosureSpec.from_config(
            {"closure": {"mode": "scr", "energy_cutoff_mev": -5.0}}
        )


def test_solve_closure_dispatch_reads_fitted_targets():
    model = _scalar_tier_a()
    spec = ClosureSpec.from_config(
        {"closure": {"mode": "onsager", "moment_mode": "fitted", "moment_target": 1.0}}
    )
    result = solve_closure(
        spec, model, chi0=0.8, gamma0=2.0, temperature_K=20.0,
        params={"m2_total": 0.7},
    )
    assert result.m2_total == pytest.approx(0.7, abs=1e-9)


def _tier_b_pair(omega_larmor, omega_points=800):
    """Matched Tier-A (tensor eigenvalues) and Tier-B models on one network."""
    from nfit.tensor_rpa import (
        assemble_tensor_exchange,
        build_tensor_structure,
        zeeman_cartesian_propagator,
    )

    positions = [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]]
    orbits = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]},
    ]
    grid = bz_sample_hkl(4)
    geometry = build_rpa_geometry(grid[:, 0], grid[:, 1], grid[:, 2], positions, orbits)
    structure = build_tensor_structure(geometry, positions, orbits)
    params = {"J1": 0.08, "J2": -0.03}
    exchange = assemble_tensor_exchange(structure, params)
    b_hat = np.array([0.0, 0.0, 1.0])

    def builder(omega, chi0, gamma0):
        return zeeman_cartesian_propagator(
            omega, b_hat, chi0=chi0, gamma0=gamma0, omega_larmor=omega_larmor
        )

    tier_b = TierBMoments(exchange, 2, builder, omega_points=omega_points)
    tier_a = TierAMoments(
        np.linalg.eigvalsh(exchange), n_sites=2, isotropic_components=1
    )
    return tier_a, tier_b


def test_tier_b_moment_matches_tier_a_at_zero_field():
    tier_a, tier_b = _tier_b_pair(omega_larmor=0.0, omega_points=3000)
    kwargs = dict(chi0=0.6, gamma0=2.0, temperature_K=25.0, cutoff_mev=60.0)
    zp_a, th_a = tier_a.moment(**kwargs)
    zp_b, th_b = tier_b.moment(**kwargs)
    assert zp_b == pytest.approx(zp_a, rel=2e-3)
    assert th_b == pytest.approx(th_a, rel=2e-3)


def test_tier_b_onsager_solves_with_field_on():
    _, tier_b = _tier_b_pair(omega_larmor=1.5, omega_points=400)
    result = solve_onsager(
        tier_b, chi0=0.6, gamma0=2.0, temperature_K=25.0, cutoff_mev=60.0, target=0.8
    )
    assert result.m2_total == pytest.approx(0.8, abs=1e-9)
    zp, th = tier_b.moment(
        chi0=0.6, gamma0=2.0, temperature_K=25.0, cutoff_mev=60.0,
        lambda_shift=result.lambda_shift,
    )
    assert zp + th == pytest.approx(0.8, abs=1e-9)


def test_tier_b_rejects_transverse_instability_from_static_local_tensor():
    from nfit.tensor_rpa import zeeman_cartesian_propagator

    exchange = np.diag([0.6, 0.0, 0.0]).astype(complex)[None]

    def builder(omega, chi0, gamma0):
        return zeeman_cartesian_propagator(
            omega,
            np.array([0.0, 0.0, 1.0]),
            chi0=chi0,
            gamma0=gamma0,
            omega_larmor=0.1,
            chi_perp_ratio=2.0,
        )

    model = TierBMoments(exchange, 1, builder, omega_points=20)
    with pytest.raises(ValueError, match="RPA instability"):
        model.moment(
            chi0=1.0,
            gamma0=1.0,
            temperature_K=10.0,
            cutoff_mev=10.0,
        )


def test_tier_b_scr_solves_with_field_on():
    _, tier_b = _tier_b_pair(omega_larmor=1.0, omega_points=300)
    result = solve_scr(
        tier_b, chi0_bare=0.6, gamma0=2.0, temperature_K=25.0, cutoff_mev=60.0,
        mode_coupling_u=0.2,
    )
    assert 0 < result.chi0_eff < 0.6
    residual = 1.0 / result.chi0_eff - 1.0 / 0.6 - 0.2 * result.m2_total
    assert residual == pytest.approx(0.0, abs=1e-8)
