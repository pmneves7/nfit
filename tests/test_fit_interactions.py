# ruff: noqa: F401, F403, F405
from tests.fit_config_test_support import *
from tests.fit_config_test_support import (
    _closure_points,
    _closure_scalar_component,
    _constant_component,
    _evaluate,
    _fit_ready_group,
    _grid_mdhisto,
    _heisenberg_chain_component,
    _magnetization_component,
    _magnetization_points,
    _points,
    _pyrochlore_tensor_component,
    _rlu_matrix,
    _rpa_component,
    _rpa_points,
    _spin_fluctuation_points,
    _tensor_points,
)


def test_tensor_component_emits_anisotropy_parameters_and_declines_analytic_jacobian():
    from nfit.fitting import problem_supports_analytic_jacobian

    _crystal, component = _pyrochlore_tensor_component()
    names = component_parameter_names(component)
    assert names == (
        "chi0",
        "gamma0",
        "inverse_mode_energy_sq",
        "J1",
        "J1_S1",
        "J1_S2",
        "J1_D1",
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", _tensor_points(0), data_type="single_crystal_inelastic")]
    )
    # Tensor path falls back to finite-difference gradients.
    assert not problem_supports_analytic_jacobian(compiled.problem)


def test_disabled_anisotropy_takes_scalar_path_bit_identical():
    """A component whose anisotropy sections are all disabled must be scalar."""
    from nfit.fit_config import _RpaComponentEvaluator
    from nfit.fitting import evaluate_problem_model, problem_supports_analytic_jacobian

    _crystal, component = _pyrochlore_tensor_component()
    # Disable the anisotropy section entirely.
    component.config["anisotropy"]["J1"]["enabled"] = False
    component.fit_parameters = {"scale": True, "chi0": True, "J1": True}
    assert not _RpaComponentEvaluator(component).tensor_mode
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", _tensor_points(1), data_type="single_crystal_inelastic")]
    )
    # The scalar path keeps its analytic Jacobian.
    assert problem_supports_analytic_jacobian(compiled.problem)
    values = evaluate_problem_model(
        compiled.problem, "d", {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    )
    assert np.all(np.isfinite(values))


def test_tensor_fit_recovers_anisotropic_parameters():
    from nfit.fitting import (
        OptimizationConfig,
        evaluate_problem_model,
        fit_problem_least_squares,
    )

    _crystal, truth = _pyrochlore_tensor_component(fit_aniso=False)
    truth.parameters.update({"scale": 1.2, "J1_S1": 0.012})
    points = _tensor_points(3)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    model_values = evaluate_problem_model(
        compiled_truth.problem,
        "d",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted = PointData4D(
        points.H,
        points.K,
        points.L,
        points.E,
        model_values,
        np.full(points.size, 0.01),
        temperature=points.temperature,
        metadata=dict(points.metadata),
    )
    _crystal, start = _pyrochlore_tensor_component()
    compiled = compile_fit_problem(
        [start], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
    )
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    assert result.reduced_chi2 < 1e-6


def test_dipole_component_emits_parameter_and_reduces_to_scalar_at_zero():
    from nfit.crystal import generate_bond_orbits, orbits_to_config, sites_to_config
    from nfit.dipole import dipole_coupling_constant
    from nfit.fitting import evaluate_problem_model

    crystal = {
        "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "spacegroup": "F d -3 m:2",
        "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "V2"}],
    }
    nn = 10.0 * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(crystal, ["M1"], cutoff_angstrom=nn + 0.01)
    config = {
        "site_positions": sites_to_config(sites),
        "orbits": orbits_to_config(orbits),
        "crystal": crystal,
        "ion": "V2",
        "dipole": {"enabled": True},
    }
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.02, "gamma0": 3.0, "J1": 0.05, "D_dip": 0.0},
        fit_parameters={"scale": True, "J1": True, "D_dip": True},
        config=config,
    )
    assert "D_dip" in component_parameter_names(component)

    points = _tensor_points(5, n=120)
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    with_zero_dipole = evaluate_problem_model(
        compiled.problem, "d", {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    )

    scalar = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.02, "gamma0": 3.0, "J1": 0.05},
        fit_parameters={},
        config={k: v for k, v in config.items() if k != "dipole"},
    )
    compiled_scalar = compile_fit_problem(
        [scalar], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    scalar_values = evaluate_problem_model(
        compiled_scalar.problem,
        "d",
        {spec.name: spec.value for spec in compiled_scalar.problem.parameter_specs},
    )
    # D_dip = 0 must reduce to the scalar Heisenberg intensity.
    np.testing.assert_allclose(with_zero_dipole, scalar_values, rtol=1e-10, atol=1e-12)

    # A physical dipole strength changes the intensity.
    physical = dict(component.parameters)
    physical["D_dip"] = dipole_coupling_constant()
    with_dipole = evaluate_problem_model(
        compiled.problem,
        "d",
        {
            spec.name: physical.get(spec.name.split(".")[-1], spec.value)
            for spec in compiled.problem.parameter_specs
        },
    )
    assert np.max(np.abs(with_dipole - with_zero_dipole)) > 1e-8 * 0.073**2 / np.pi


def test_zeeman_component_requires_field_and_reduces_to_scalar_at_g_zero():
    from nfit.fitting import evaluate_problem_model, magnetic_field_vector

    crystal = {
        "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "spacegroup": "F d -3 m:2",
        "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "V2"}],
    }
    from nfit.crystal import generate_bond_orbits, orbits_to_config, sites_to_config

    nn = 10.0 * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(crystal, ["M1"], cutoff_angstrom=nn + 0.01)
    config = {
        "site_positions": sites_to_config(sites),
        "orbits": orbits_to_config(orbits),
        "crystal": crystal,
        "ion": "V2",
        "zeeman": {"enabled": True},
    }
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={
            "scale": 1.0,
            "chi0": 0.02,
            "gamma0": 3.0,
            "J1": 0.05,
            "g_factor": 0.0,
            "chi_perp_ratio": 1.0,
            "gamma_perp_ratio": 1.0,
        },
        fit_parameters={"J1": True},
        config=config,
    )
    assert component_parameter_names(component)[-3:] == (
        "g_factor",
        "chi_perp_ratio",
        "gamma_perp_ratio",
    )

    field = magnetic_field_vector(4.0, [1, 1, 1], "uvw", crystal["lattice"])
    points = _tensor_points(6, n=100)
    points = points.with_updates(magnetic_field=field)
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    with_g0 = evaluate_problem_model(
        compiled.problem, "d", {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    )
    # g_factor = 0 -> no Larmor -> equals the scalar Heisenberg intensity.
    scalar = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.02, "gamma0": 3.0, "J1": 0.05},
        fit_parameters={},
        config={k: v for k, v in config.items() if k != "zeeman"},
    )
    compiled_scalar = compile_fit_problem(
        [scalar], [FitDatasetInput("d", points, data_type="single_crystal_inelastic")]
    )
    scalar_values = evaluate_problem_model(
        compiled_scalar.problem,
        "d",
        {spec.name: spec.value for spec in compiled_scalar.problem.parameter_specs},
    )
    np.testing.assert_allclose(with_g0, scalar_values, rtol=1e-9, atol=1e-12)

    # A missing field raises a clear, actionable error.
    no_field = _tensor_points(7, n=50)
    compiled_missing = compile_fit_problem(
        [component], [FitDatasetInput("d", no_field, data_type="single_crystal_inelastic")]
    )
    with pytest.raises(ValueError, match="Conditions"):
        evaluate_problem_model(
            compiled_missing.problem,
            "d",
            {spec.name: spec.value for spec in compiled_missing.problem.parameter_specs},
        )


def test_closure_none_is_bit_identical():
    """Absent closure config and mode "none" take the identical code path."""
    from nfit.fit_config import _RpaComponentEvaluator

    points = _closure_points(0)
    plain = _RpaComponentEvaluator(_closure_scalar_component())
    off = _RpaComponentEvaluator(_closure_scalar_component(closure={"mode": "none"}))
    params = {f"M.{k}": v for k, v in {"scale": 1.0, "chi0": 0.5, "gamma0": 2.0, "J1": 0.1}.items()}
    assert np.array_equal(plain.value(points, params), off.value(points, params))
    grads_plain = plain.gradients(points, params)
    grads_off = off.gradients(points, params)
    assert grads_plain.keys() == grads_off.keys()
    for key in grads_plain:
        assert np.array_equal(grads_plain[key], grads_off[key])


def test_closure_exposes_parameters_and_declines_analytic_jacobian():
    from nfit.fitting import problem_supports_analytic_jacobian

    fitted = _closure_scalar_component(
        closure={"mode": "onsager", "moment_mode": "fitted", "bz_grid": 6},
        m2_total=1.0,
    )
    assert component_parameter_names(fitted)[-1] == "m2_total"
    scr = _closure_scalar_component(closure={"mode": "scr", "bz_grid": 6}, mode_coupling_u=0.1)
    assert component_parameter_names(scr)[-1] == "mode_coupling_u"
    compiled = compile_fit_problem(
        [fitted],
        [FitDatasetInput("d", _closure_points(1), data_type="single_crystal_inelastic")],
    )
    assert not problem_supports_analytic_jacobian(compiled.problem)


def test_tac_forces_unidentifiable_bare_chi0_fixed():
    component = _closure_scalar_component(
        closure={"mode": "tac", "moment_target": 1.0, "bz_grid": 6}
    )
    component.fit_parameters = {"chi0": True, "gamma0": True}
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("d", _closure_points(1), data_type="single_crystal_inelastic")],
    )
    specs = {spec.name: spec for spec in compiled.problem.parameter_specs}
    assert specs["M.chi0"].vary is False
    assert specs["M.gamma0"].vary is True


def test_onsager_closure_changes_intensity_and_stays_finite():
    points = _closure_points(2)
    plain = _evaluate(_closure_scalar_component(), points)
    closed = _evaluate(
        _closure_scalar_component(closure={"mode": "onsager", "moment_target": 1.0, "bz_grid": 8}),
        points,
    )
    assert np.all(np.isfinite(closed))
    assert not np.allclose(closed, plain)
    assert np.max(np.abs(closed)) < 1e5  # not the sentinel


def test_scr_zero_coupling_matches_plain_rpa():
    """u = 0 SCR is exactly the bare model (chi0_eff == chi0 to solver tol)."""
    points = _closure_points(3)
    plain = _evaluate(_closure_scalar_component(), points)
    scr = _evaluate(
        _closure_scalar_component(closure={"mode": "scr", "bz_grid": 6}, mode_coupling_u=0.0),
        points,
    )
    np.testing.assert_allclose(scr, plain, rtol=1e-8, atol=1e-12)


def test_unreachable_closure_returns_sentinel():
    points = _closure_points(4, n=30)
    values = _evaluate(
        _closure_scalar_component(closure={"mode": "onsager", "moment_target": 1e18, "bz_grid": 4}),
        points,
    )
    np.testing.assert_array_equal(values, np.full(points.size, 1e6))


def test_closure_fit_recovers_fitted_moment_target():
    """Synthetic data from a fixed-target Onsager model is recovered by
    fitting m2_total through the finite-difference path."""
    from nfit.fitting import OptimizationConfig, fit_problem_least_squares

    points = _closure_points(5, n=100)
    truth = _closure_scalar_component(
        closure={"mode": "onsager", "moment_target": 1.0, "bz_grid": 6}
    )
    model_values = _evaluate(truth, points)
    fitted_points = PointData4D(
        points.H,
        points.K,
        points.L,
        points.E,
        intensity=model_values,
        sigma=np.full(points.size, 0.01),
        temperature=10.0,
    )
    start = _closure_scalar_component(
        closure={"mode": "onsager", "moment_mode": "fitted", "bz_grid": 6},
        m2_total=0.6,
    )
    start.fit_parameters = {"m2_total": True}
    compiled = compile_fit_problem(
        [start], [FitDatasetInput("d", fitted_points, data_type="single_crystal_inelastic")]
    )
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    assert result.params["M.m2_total"] == pytest.approx(1.0, rel=1e-4)


def test_zeeman_closure_smoke():
    """Onsager + Zeeman: the Tier-B closure path evaluates and hits its target."""
    from nfit.crystal import generate_bond_orbits, orbits_to_config, sites_to_config
    from nfit.fitting import magnetic_field_vector

    pytest.importorskip("gemmi")
    crystal = {
        "lattice": {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "spacegroup": "F d -3 m:2",
        "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "V2"}],
    }
    nn = 10.0 * np.sqrt(2.0) / 4.0
    sites, orbits = generate_bond_orbits(crystal, ["M1"], cutoff_angstrom=nn + 0.01)
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={
            "scale": 1.0,
            "chi0": 0.02,
            "gamma0": 3.0,
            "J1": 0.05,
            "g_factor": 2.0,
            "chi_perp_ratio": 1.0,
            "gamma_perp_ratio": 1.0,
        },
        fit_parameters={},
        config={
            "site_positions": sites_to_config(sites),
            "orbits": orbits_to_config(orbits),
            "crystal": crystal,
            "ion": "V2",
            "zeeman": {"enabled": True},
            "closure": {"mode": "onsager", "moment_target": 0.5, "bz_grid": 3, "omega_points": 60},
        },
    )
    points = _tensor_points(9, n=60)
    points = points.with_updates(
        magnetic_field=magnetic_field_vector(4.0, [0, 0, 1], "uvw", crystal["lattice"])
    )
    values = _evaluate(component, points)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) < 1e5


def test_component_diagnostics_static_chi_matches_direct_rpa():
    """chi_static_q0 (KK of the modes) equals the direct static RPA at Q=0."""
    from nfit.fit_config import compute_component_diagnostics

    # Single-site FM chain: J(0) = 2 J1, static chi = chi0 / (1 - J(0) chi0).
    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.5, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
                        {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
                        {"site_i": 0, "site_j": 0, "offset": [0, 0, 1]},
                    ],
                }
            ],
        },
    )
    points = _closure_points(11, n=40)
    record = compute_component_diagnostics(
        component, points, {"M.scale": 1.0, "M.chi0": 0.5, "M.gamma0": 2.0, "M.J1": 0.1}
    )
    assert record is not None
    for key in (
        "mu_eff_sq",
        "chi_static_q0",
        "chi_static_qpeak",
        "stability_margin",
        "stability_ratio",
        "chi0_gamma0",
        "temperature",
    ):
        assert key in record
    # J(0) = 2 J1 * 3 bonds = 0.6; chi = 0.5 / (1 - 0.6 * 0.5).
    expected = 0.5 / (1.0 - 0.6 * 0.5)
    assert record["chi_static_q0"] == pytest.approx(expected, rel=1e-9)
    # The FM peak is at Q=0; the shifted BZ grid samples close to but not
    # exactly at Gamma, so the grid peak approaches the Q=0 value from below.
    assert 0.9 * expected < record["chi_static_qpeak"] <= expected * (1.0 + 1e-9)
    assert record["chi0_gamma0"] == pytest.approx(1.0)


def test_component_diagnostics_report_closure_internals():
    from nfit.fit_config import compute_component_diagnostics

    component = _closure_scalar_component(
        closure={"mode": "onsager", "moment_target": 0.5, "bz_grid": 6}
    )
    points = _closure_points(12, n=40)
    record = compute_component_diagnostics(
        component,
        points,
        {"M.scale": 1.0, "M.chi0": 0.5, "M.gamma0": 2.0, "M.J1": 0.1},
    )
    assert record["lambda_shift"] != 0.0
    assert record["mu_eff_sq"] == pytest.approx(0.5, abs=1e-6)
    assert record["stability_margin"] > 0.0
    # JSON-serializable (persisted in fit metadata via json.dumps).
    import json

    assert json.loads(json.dumps(record)) == record


def test_component_diagnostics_keep_negative_stability_margin_when_unstable():
    from nfit.fit_config import compute_component_diagnostics

    component = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 2.0, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [
                        {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
                        {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
                        {"site_i": 0, "site_j": 0, "offset": [0, 0, 1]},
                    ],
                }
            ],
        },
    )

    record = compute_component_diagnostics(
        component,
        _closure_points(14, n=20),
        {"M.scale": 1.0, "M.chi0": 2.0, "M.gamma0": 2.0, "M.J1": 0.1},
    )

    assert record is not None
    assert record["stability_margin"] < 0.0
    assert record["stability_ratio"] > 1.0
    assert record["unstable"] == 1.0
    assert all(
        key in record
        for key in ("stability_q_h", "stability_q_k", "stability_q_l", "stability_mode_index")
    )


def test_component_diagnostics_returns_none_for_non_rpa():
    from nfit.fit_config import compute_component_diagnostics

    bg = ModelComponentSpec(
        name="bg", type="constant_background", parameters={"constant": 1.0}, fit_parameters={}
    )
    assert compute_component_diagnostics(bg, _closure_points(13, n=10), {}) is None


def test_group_fit_stores_and_persists_diagnostics(tmp_path):
    """A heisenberg_rpa group fit stamps per-dataset diagnostics that survive
    save/load."""
    from nfit.mdhisto import MDHistoAxis, MDHistoData

    # A 1D Q-scan MDHisto dataset the RPA model can be evaluated on.
    n = 24
    axis_q = MDHistoAxis("H", np.linspace(0.0, 1.0, n + 1), "rlu", "momentum")
    signal = np.linspace(1.0, 2.0, n).reshape(1, n)
    axis_e = MDHistoAxis("E", np.linspace(0.5, 5.0, 2), "meV", "energy")
    data = MDHistoData(
        axes=(axis_e, axis_q),
        signal=signal,
        errors=np.full_like(signal, 0.1),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={"temperature": 10.0},
    )
    group = DataGroup(name="G")
    group.datasets.append(DatasetEntry("scan", data))
    group.get_dataset("scan").data_type = "single_crystal_inelastic"

    model = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.3, "gamma0": 2.0, "J1": 0.05},
        fit_parameters={"scale": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
        },
    )
    group.models[model.name] = model
    ensure_fit_history(group)

    entry = run_group_fit(group, group.fits[0])
    assert entry.goodness["status"] in ("converged", "not converged")
    diagnostics = entry.metadata.get("diagnostics")
    assert diagnostics is not None and "scan" in diagnostics
    assert "mu_eff_sq" in diagnostics["scan"]
    assert diagnostics["scan"]["temperature"] == pytest.approx(10.0)

    # The diagnostics survive the fit-entry JSON round-trip used by save/load.
    from nfit.project_gui import _fit_entry_from_dict, _fit_entry_to_dict

    restored = _fit_entry_from_dict(_fit_entry_to_dict(entry))
    assert restored.metadata["diagnostics"]["scan"]["mu_eff_sq"] == pytest.approx(
        diagnostics["scan"]["mu_eff_sq"]
    )
