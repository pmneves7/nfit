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


def test_heisenberg_rpa_emits_dynamic_orbit_parameters():
    component = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"chi0": 0.5, "gamma0": 2.0, "J1": 0.1, "J3a": 0.0},
        fit_parameters={"J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]},
                {"label": "J3a", "bonds": [{"site_i": 0, "site_j": 0, "offset": [3, 0, 0]}]},
            ],
        },
    )
    H = np.linspace(0.0, 1.0, 16)
    points = _spin_fluctuation_points(np.ones(16), H, np.ones(16), temperature=10.0)
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    names = [spec.name for spec in compiled.problem.parameter_specs]
    assert names == ["rpa.chi0", "rpa.gamma0", "rpa.J1", "rpa.J3a"]


def test_spin_fluctuation_models_require_temperature():
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 2.0},
        fit_parameters={"chi_loc": True},
    )
    H = np.zeros(8)
    points = _spin_fluctuation_points(np.ones(8), H, np.linspace(0.5, 4.0, 8), temperature=None)
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    with pytest.raises(ValueError, match="temperature"):
        fit_problem_least_squares(compiled.problem)


def test_local_relaxational_fit_recovers_synthetic_parameters():
    from nfit.cross_section import MAGNETIC_GAMMA0_PER_MU_B, intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import local_relaxational_chipp

    rng = np.random.default_rng(11)
    E = np.linspace(0.5, 12.0, 80)
    H = np.zeros_like(E)
    temperature = 25.0
    truth = {"chi_loc": 2.4, "gamma": 3.1}
    clean = intensity_from_chipp(
        local_relaxational_chipp(E, **truth),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.005 * MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi, E.size),
        H,
        E,
        temperature=temperature,
    )
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"scale": 1.0, "chi_loc": 1.0, "gamma": 2.0},
        fit_parameters={"chi_loc": True, "gamma": True},
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["loc.chi_loc"] == pytest.approx(truth["chi_loc"], rel=0.02)
    assert result.params["loc.gamma"] == pytest.approx(truth["gamma"], rel=0.02)


def test_local_relaxational_model_can_emit_selected_chipp_or_mbarn_channel():
    from nfit.fitting import evaluate_problem_model
    from nfit.spin_fluctuations import local_relaxational_chipp

    energy = np.array([1.0, 3.0])
    points = _spin_fluctuation_points(np.ones(2), np.zeros(2), energy, temperature=20.0)
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"chi_loc": 1.5, "gamma": 2.5},
        fit_parameters={},
    )
    convention = {
        "fit_representation": "chi_double_prime",
        "unit": "mu_B^2/meV/f.u.",
        "moment_unit": "mu_B_squared",
        "g_factor": 2.0,
        "kf_ki_state": "removed",
    }
    points.metadata["spectral_observable"] = convention
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    chipp = local_relaxational_chipp(energy, chi_loc=1.5, gamma=2.5)
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "a", params), 2.0**2 * chipp
    )

    convention.update(
        {
            "fit_representation": "cross_section",
            "unit": "mbarn/sr/meV/f.u.",
        }
    )
    from nfit.cross_section import cross_section_from_chipp

    expected = (
        cross_section_from_chipp(
            chipp,
            energy,
            20.0,
            polarization=2.0,
            moment_unit="spin_squared",
            g_factor=2.0,
        )
        * 1000.0
    )
    np.testing.assert_allclose(evaluate_problem_model(compiled.problem, "a", params), expected)


@pytest.mark.parametrize("data_type", ["single_crystal_elastic", "powder_elastic"])
def test_local_relaxational_elastic_uses_quasistatic_response(data_type):
    from nfit.cross_section import quasistatic_cross_section_from_chi
    from nfit.fitting import evaluate_problem_model

    points = _spin_fluctuation_points(
        np.ones(4),
        np.linspace(0.1, 0.7, 4),
        np.zeros(4),
        temperature=18.0,
    )
    if data_type == "powder_elastic":
        points.metadata.update(
            {"coordinate_units": "1/angstrom", "powder_q_modulus_axis": True}
        )
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"chi_loc": 1.5, "gamma": 2.5},
        fit_parameters={},
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("elastic", points, data_type=data_type)]
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    prediction = evaluate_problem_model(compiled.problem, "elastic", params)
    expected = quasistatic_cross_section_from_chi(
        np.full(4, 1.5),
        18.0,
        polarization=2.0,
        moment_unit="spin_squared",
        g_factor=2.0,
    )
    np.testing.assert_allclose(prediction, expected)
    params["loc.gamma"] = 50.0
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "elastic", params),
        expected,
    )


def test_mmp_single_crystal_elastic_uses_static_lorentzian():
    from nfit.cross_section import quasistatic_cross_section_from_chi
    from nfit.fitting import evaluate_problem_model

    h = np.array([0.3, 0.5, 0.7])
    points = _spin_fluctuation_points(
        np.ones(3), h, np.zeros(3), temperature=25.0, lattice_a=4.0
    )
    component = ModelComponentSpec(
        name="mmp",
        type="mmp_relaxational",
        parameters={
            "chi_pk": 3.0,
            "xi": 2.2,
            "omega_sf": 1.8,
            "q0_h": 0.5,
            "q0_k": 0.0,
            "q0_l": 0.0,
        },
        fit_parameters={},
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("elastic", points, data_type="single_crystal_elastic")],
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    q_sq = (2.0 * np.pi / 4.0) ** 2 * (h - 0.5) ** 2
    expected = quasistatic_cross_section_from_chi(
        3.0 / (1.0 + 2.2**2 * q_sq),
        25.0,
        polarization=2.0,
        moment_unit="spin_squared",
        g_factor=2.0,
    )
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "elastic", params), expected
    )


def test_heisenberg_rpa_powder_elastic_is_finite_and_gamma_independent():
    from nfit.fitting import evaluate_problem_model

    component = _heisenberg_chain_component()
    component.config["crystal"] = {
        "lattice": {
            "a": 4.0,
            "b": 4.0,
            "c": 4.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }
    }
    q = np.linspace(0.2, 2.0, 7)
    points = _spin_fluctuation_points(
        np.ones(q.size), q, np.zeros(q.size), temperature=12.0
    )
    points.metadata.update(
        {
            "coordinate_units": "1/angstrom",
            "powder_q_modulus_axis": True,
        }
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("powder", points, data_type="powder_elastic")],
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    prediction = evaluate_problem_model(compiled.problem, "powder", params)
    assert np.all(np.isfinite(prediction))
    assert np.ptp(prediction) > 0.0
    params["rpa.gamma0"] = 100.0
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "powder", params),
        prediction,
    )


def test_mmp_relaxational_fit_recovers_synthetic_parameters():
    from nfit.cross_section import MAGNETIC_GAMMA0_PER_MU_B, intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import mmp_chipp

    rng = np.random.default_rng(5)
    lattice_a = 4.0
    H_axis = np.linspace(0.2, 0.8, 13)
    E_axis = np.linspace(0.5, 8.0, 11)
    H, E = (arr.ravel() for arr in np.meshgrid(H_axis, E_axis))
    temperature = 40.0
    truth = {"chi_pk": 3.0, "xi": 2.2, "omega_sf": 1.8}
    q_sq = (2.0 * np.pi / lattice_a) ** 2 * (H - 0.5) ** 2
    clean = intensity_from_chipp(
        mmp_chipp(q_sq, E, **truth),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.005 * MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi, E.size),
        H,
        E,
        temperature=temperature,
        lattice_a=lattice_a,
    )
    component = ModelComponentSpec(
        name="mmp",
        type="mmp_relaxational",
        parameters={
            "scale": 1.0,
            "chi_pk": 1.5,
            "xi": 1.0,
            "omega_sf": 1.0,
            "q0_h": 0.5,
            "q0_k": 0.0,
            "q0_l": 0.0,
        },
        fit_parameters={"chi_pk": True, "xi": True, "omega_sf": True},
    )
    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["mmp.chi_pk"] == pytest.approx(truth["chi_pk"], rel=0.05)
    assert result.params["mmp.xi"] == pytest.approx(truth["xi"], rel=0.05)
    assert result.params["mmp.omega_sf"] == pytest.approx(truth["omega_sf"], rel=0.05)


def _generalized_paramagnon_component(**overrides):
    component = ModelComponentSpec(
        name="pm",
        type="generalized_paramagnon",
        parameters={
            "chi_peak": 2.0,
            "gamma0": 2.5,
            "relaxation_power": 1.0,
            "inverse_mode_energy_sq": 0.04,
            "xi_x": 2.0,
            "xi_y": 1.0,
            "xi_z": 1.0,
            "xi_yx": 0.0,
            "xi_zx": 0.0,
            "xi_zy": 0.0,
            "q0_h": 0.5,
            "q0_k": 0.0,
            "q0_l": 0.0,
        },
        fit_parameters={},
        config={
            "spatial_power": 2.0,
            "center_offsets": [[0.0, 0.0, 0.0]],
            "center_combination": "sum",
            "periodic": True,
            "powder_orientations": 24,
            "lattice": {},
            "ion": "",
            "form_factor_coefficients": "",
        },
    )
    for key, value in overrides.items():
        setattr(component, key, value)
    return component


def test_generalized_paramagnon_single_crystal_matches_complex_kernel():
    from nfit.cross_section import intensity_from_chipp
    from nfit.fitting import evaluate_problem_model
    from nfit.spin_fluctuations import generalized_paramagnon_chipp

    h = np.array([0.4, 0.5, 0.6])
    energy = np.array([1.0, 1.5, 2.0])
    points = _spin_fluctuation_points(
        np.ones(3), h, energy, temperature=30.0, lattice_a=4.0
    )
    component = _generalized_paramagnon_component()
    compiled = compile_fit_problem([component], [FitDatasetInput("scan", points)])
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    qx = 2.0 * np.pi / 4.0 * (h - 0.5)
    kernel = 1.0 + (2.0 * qx) ** 2
    expected = intensity_from_chipp(
        generalized_paramagnon_chipp(
            kernel,
            energy,
            chi_peak=2.0,
            gamma0=2.5,
            relaxation_power=1.0,
            inverse_mode_energy_sq=0.04,
        ),
        energy,
        30.0,
        polarization=2.0,
    )
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "scan", params), expected
    )


def test_generalized_paramagnon_fit_recovers_damped_mode_parameters():
    from nfit.spin_fluctuations import generalized_paramagnon_chipp

    energy = np.linspace(0.2, 12.0, 100)
    h = np.full(energy.shape, 0.5)
    truth = {
        "chi_peak": 2.4,
        "gamma0": 2.8,
        "inverse_mode_energy_sq": 0.05,
    }
    intensity = generalized_paramagnon_chipp(
        np.ones(energy.shape),
        energy,
        relaxation_power=1.0,
        **truth,
    )
    points = _spin_fluctuation_points(
        intensity, h, energy, temperature=20.0, lattice_a=4.0
    )
    points.metadata["spectral_observable"] = {
        "fit_representation": "chi_double_prime",
        "unit": "spin^2/meV",
        "moment_unit": "spin_squared",
        "g_factor": 2.0,
        "kf_ki_state": "removed",
    }
    component = _generalized_paramagnon_component()
    component.parameters.update(
        chi_peak=1.5,
        gamma0=1.5,
        inverse_mode_energy_sq=0.01,
    )
    component.fit_parameters = {
        "chi_peak": True,
        "gamma0": True,
        "inverse_mode_energy_sq": True,
    }
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("mode", points)]
    )
    result = fit_problem_least_squares(compiled.problem)
    for name, value in truth.items():
        assert result.params[f"pm.{name}"] == pytest.approx(value, rel=1.0e-5)


def test_generalized_paramagnon_elastic_ignores_dynamic_parameters():
    from nfit.cross_section import quasistatic_cross_section_from_chi
    from nfit.fitting import evaluate_problem_model

    h = np.array([0.4, 0.5, 0.6])
    points = _spin_fluctuation_points(
        np.ones(3), h, np.zeros(3), temperature=20.0, lattice_a=4.0
    )
    component = _generalized_paramagnon_component()
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("elastic", points, data_type="single_crystal_elastic")],
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    qx = 2.0 * np.pi / 4.0 * (h - 0.5)
    expected = quasistatic_cross_section_from_chi(
        2.0 / (1.0 + (2.0 * qx) ** 2),
        20.0,
        polarization=2.0,
        moment_unit="spin_squared",
        g_factor=2.0,
    )
    prediction = evaluate_problem_model(compiled.problem, "elastic", params)
    np.testing.assert_allclose(prediction, expected)
    params["pm.gamma0"] = 20.0
    params["pm.inverse_mode_energy_sq"] = 0.5
    np.testing.assert_allclose(
        evaluate_problem_model(compiled.problem, "elastic", params), prediction
    )


@pytest.mark.parametrize("data_type", ["powder_inelastic", "powder_elastic"])
def test_generalized_paramagnon_powder_is_finite(data_type):
    from nfit.fitting import evaluate_problem_model

    q = np.linspace(0.2, 2.0, 5)
    energy = (
        np.zeros(5)
        if data_type == "powder_elastic"
        else np.linspace(0.5, 3.0, 5)
    )
    points = _spin_fluctuation_points(
        np.ones(5), q, energy, temperature=15.0, lattice_a=4.0
    )
    points.metadata.update(
        {"coordinate_units": "1/angstrom", "powder_q_modulus_axis": True}
    )
    component = _generalized_paramagnon_component()
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("powder", points, data_type=data_type)]
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    prediction = evaluate_problem_model(compiled.problem, "powder", params)
    assert prediction.shape == q.shape
    assert np.all(np.isfinite(prediction))


def test_heisenberg_rpa_fit_recovers_synthetic_parameters():
    from nfit.cross_section import MAGNETIC_GAMMA0_PER_MU_B, intensity_from_chipp
    from nfit.fit_config import ISOTROPIC_POLARIZATION
    from nfit.spin_fluctuations import build_rpa_geometry, heisenberg_rpa_chipp

    rng = np.random.default_rng(2)
    H_axis = np.linspace(0.0, 1.0, 15)
    E_axis = np.linspace(0.5, 10.0, 12)
    H, E = (arr.ravel() for arr in np.meshgrid(H_axis, E_axis))
    temperature = 15.0
    truth = {"chi0": 0.7, "gamma0": 3.0, "J1": 0.4}
    orbits = [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}]
    geometry = build_rpa_geometry(H, np.zeros_like(H), np.zeros_like(H), [[0.0, 0.0, 0.0]], orbits)
    clean = intensity_from_chipp(
        heisenberg_rpa_chipp(
            geometry,
            E,
            chi0=truth["chi0"],
            gamma0=truth["gamma0"],
            j_values={"J1": truth["J1"]},
        ),
        E,
        temperature,
        polarization=ISOTROPIC_POLARIZATION,
    )
    points = _spin_fluctuation_points(
        clean + rng.normal(0.0, 0.002 * MAGNETIC_GAMMA0_PER_MU_B**2 / np.pi, E.size),
        H,
        E,
        temperature=temperature,
    )
    compiled = compile_fit_problem([_heisenberg_chain_component()], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["rpa.chi0"] == pytest.approx(truth["chi0"], rel=0.05)
    assert result.params["rpa.gamma0"] == pytest.approx(truth["gamma0"], rel=0.05)
    assert result.params["rpa.J1"] == pytest.approx(truth["J1"], rel=0.05)


def test_heisenberg_rpa_blank_custom_form_factor_uses_selected_ion():
    H = np.linspace(0.0, 1.0, 8)
    points = _spin_fluctuation_points(
        np.ones(H.size),
        H,
        np.ones(H.size),
        temperature=5.0,
        lattice_a=8.24062,
    )
    component = _heisenberg_chain_component()
    component.config["ion"] = "V2"
    component.config["form_factor_coefficients"] = ""
    component.fit_parameters = {}

    compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
    result = fit_problem_least_squares(compiled.problem)

    assert result.success
    assert np.all(np.isfinite(result.model_values))


def test_form_factor_g_J_selects_the_dipole_approximation():
    """form_factor_g_J must reach the evaluated model, and default to spin-only."""

    from nfit.fitting import evaluate_problem_model

    H = np.linspace(0.4, 1.4, 8)
    points = _spin_fluctuation_points(
        np.ones(H.size),
        H,
        np.ones(H.size),
        temperature=5.0,
        lattice_a=8.24062,
    )

    def evaluate(**config):
        component = _heisenberg_chain_component()
        component.config["ion"] = "Yb3"
        component.config["form_factor_coefficients"] = ""
        component.config.update(config)
        component.fit_parameters = {}
        compiled = compile_fit_problem([component], [FitDatasetInput("a", points)])
        params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
        return evaluate_problem_model(compiled.problem, "a", params)

    spin_only = evaluate()
    explicit_two = evaluate(form_factor_g_J=2.0)
    dipole = evaluate(form_factor_g_J=8.0 / 7.0)

    # The default and an explicit g_J = 2 are the spin-only <j0> form factor.
    np.testing.assert_allclose(spin_only, explicit_two, rtol=1.0e-12)
    # Yb(3+) picks up a substantial orbital contribution, growing with |Q|.
    ratio = dipole / spin_only
    assert np.all(ratio > 1.0)
    assert ratio[-1] > ratio[0]


def test_heisenberg_rpa_evaluates_powder_chipp_with_spherical_average():
    from nfit.fitting import evaluate_problem_model
    from nfit.spin_fluctuations import (
        build_rpa_geometry,
        heisenberg_rpa_chipp,
    )

    q = np.array([0.35, 0.6, 0.9])
    energy = np.array([0.5, 1.5, 3.0])
    points = _spin_fluctuation_points(
        np.ones(q.size),
        q,
        energy,
        temperature=10.0,
    )
    points.metadata.update(
        {
            "data_type": "powder_inelastic",
            "coordinate_units": "1/angstrom",
            "powder_q_modulus_axis": True,
            "spectral_observable": {
                "fit_representation": "chi_double_prime",
                "unit": "mu_B^2/meV/f.u.",
                "moment_unit": "mu_B_squared",
                "g_factor": 2.0,
                "kf_ki_state": "removed",
            },
        }
    )
    component = _heisenberg_chain_component()
    component.parameters.update({"chi0": 0.4, "gamma0": 2.0, "J1": 0.0})
    component.fit_parameters = {}
    component.config["crystal"] = {
        "lattice": {
            "a": 8.0,
            "b": 8.0,
            "c": 8.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }
    }
    component.config["powder_orientations"] = 18

    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("powder", points, data_type="powder_inelastic")],
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    result = evaluate_problem_model(compiled.problem, "powder", params)

    reference_geometry = build_rpa_geometry(
        np.zeros(q.size),
        np.zeros(q.size),
        np.zeros(q.size),
        [[0.0, 0.0, 0.0]],
        component.config["orbits"],
    )
    expected_spin_chipp = heisenberg_rpa_chipp(
        reference_geometry,
        energy,
        chi0=0.4,
        gamma0=2.0,
        j_values={"J1": 0.0},
    )
    np.testing.assert_allclose(result, 4.0 * expected_spin_chipp)

    points.metadata["spectral_observable"].update(
        {
            "fit_representation": "cross_section",
            "unit": "mbarn/sr/meV/f.u.",
        }
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("powder", points, data_type="powder_inelastic")],
    )
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    result = evaluate_problem_model(compiled.problem, "powder", params)
    from nfit.cross_section import cross_section_from_chipp

    expected_cross_section = 1000.0 * cross_section_from_chipp(
        expected_spin_chipp,
        energy,
        10.0,
        polarization=2.0,
        moment_unit="spin_squared",
        g_factor=2.0,
    )
    np.testing.assert_allclose(result, expected_cross_section)


def test_heisenberg_rpa_dynamic_parameter_supports_per_dataset_sharing():
    component = _heisenberg_chain_component(
        sharing={"chi0": {"mode": "per_dataset"}},
    )
    H = np.linspace(0.0, 1.0, 10)

    def make():
        return _spin_fluctuation_points(
            np.ones(10),
            H,
            np.ones(10),
            temperature=5.0,
        )

    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("cold", make()), FitDatasetInput("hot", make())],
    )
    names = sorted(spec.name for spec in compiled.problem.parameter_specs)
    assert "rpa.chi0[cold]" in names and "rpa.chi0[hot]" in names
    assert "rpa.J1" in names


def test_heisenberg_rpa_problem_reports_analytic_jacobian():
    from nfit.fitting import problem_supports_analytic_jacobian

    compiled = compile_fit_problem(
        [_rpa_component()],
        [FitDatasetInput("T5", _rpa_points(5.0, 1), data_type="single_crystal_inelastic")],
    )
    assert problem_supports_analytic_jacobian(compiled.problem)
    assert compiled.problem.datasets[0].model_jacobian is not None


def test_analytic_jacobian_matches_finite_differences_with_grouped_sharing():
    from nfit.fitting import (
        _evaluate_problem,
        _evaluate_problem_jacobian,
        _finite_difference_jacobian,
        pack_parameters,
        unpack_parameters,
    )

    # chi0 tied across two temperatures (grouped), scale free per dataset:
    # exercises the binding chain in the Jacobian assembly.
    component = _rpa_component(
        sharing={"chi0": "grouped", "scale": "per_dataset"},
        groups={"chi0": {"T5": "cold", "T50": "cold"}},
    )
    compiled = compile_fit_problem(
        component and [component],
        [
            FitDatasetInput("T5", _rpa_points(5.0, 1), data_type="single_crystal_inelastic"),
            FitDatasetInput("T50", _rpa_points(50.0, 2), data_type="single_crystal_inelastic"),
        ],
    )
    problem = compiled.problem
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    params = unpack_parameters(x0, names, fixed)
    analytic = _evaluate_problem_jacobian(problem, params, names, require_positive_sigma=True)

    def residual_fn(x):
        return _evaluate_problem(
            problem, unpack_parameters(x, names, fixed), require_positive_sigma=True
        ).residuals

    fd = _finite_difference_jacobian(residual_fn, x0, residual_fn(x0), bounds)
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("scale_value", [1.3, 0.7])
def test_analytic_jacobian_includes_fitted_dataset_scale(scale_value):
    from nfit.fitting import (
        _evaluate_problem,
        _evaluate_problem_jacobian,
        _finite_difference_jacobian,
        pack_parameters,
        problem_supports_analytic_jacobian,
        unpack_parameters,
    )

    compiled = compile_fit_problem(
        [_rpa_component()],
        [
            FitDatasetInput(
                "T5",
                _rpa_points(5.0, 17),
                data_type="single_crystal_inelastic",
                scale_value=scale_value,
                scale_vary=True,
            )
        ],
    )
    problem = compiled.problem
    assert problem_supports_analytic_jacobian(problem)
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    params = unpack_parameters(x0, names, fixed)
    analytic = _evaluate_problem_jacobian(problem, params, names, require_positive_sigma=True)

    def residual_fn(x):
        return _evaluate_problem(
            problem, unpack_parameters(x, names, fixed), require_positive_sigma=True
        ).residuals

    fd = _finite_difference_jacobian(residual_fn, x0, residual_fn(x0), bounds)
    scale_index = names.index(dataset_scale_parameter_name("T5"))
    assert np.any(np.abs(analytic[:, scale_index]) > 0.0)
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-6)


def test_analytic_model_fits_dataset_scale():
    from nfit.fitting import evaluate_problem_model, problem_supports_analytic_jacobian

    fixed = {name: False for name in ("scale", "chi0", "gamma0", "J1", "J2")}
    component = _rpa_component(fit_parameters=fixed)
    data = _rpa_points(5.0, 23)
    truth = compile_fit_problem(
        [component],
        [FitDatasetInput("T5", data, data_type="single_crystal_inelastic")],
    )
    model_values = evaluate_problem_model(
        truth.problem,
        "T5",
        {spec.name: spec.value for spec in truth.problem.parameter_specs},
    )
    target_scale = 0.4
    observed = PointData4D(
        data.H,
        data.K,
        data.L,
        data.E,
        model_values / target_scale,
        np.full(data.size, 0.02),
        temperature=data.temperature,
        metadata=dict(data.metadata),
    )
    compiled = compile_fit_problem(
        [component],
        [
            FitDatasetInput(
                "T5",
                observed,
                data_type="single_crystal_inelastic",
                scale_value=1.0,
                scale_vary=True,
            )
        ],
    )
    assert problem_supports_analytic_jacobian(compiled.problem)
    result = fit_problem_least_squares(compiled.problem)
    assert result.success
    assert result.params[dataset_scale_parameter_name("T5")] == pytest.approx(
        target_scale, abs=1e-6
    )


def test_analytic_jacobian_matches_finite_differences_with_constraint():
    from nfit.fitting import (
        _evaluate_problem,
        _evaluate_problem_jacobian,
        _finite_difference_jacobian,
        pack_parameters,
        unpack_parameters,
    )

    # gamma0 >= chi0 reparameterizes gamma0 as a derived (chi0 + offset):
    # exercises the derived-parameter chain rule in the assembly.
    component = _rpa_component(
        constraints=[{"parameter": "gamma0", "op": ">=", "reference": "M.chi0"}]
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("T5", _rpa_points(5.0, 3), data_type="single_crystal_inelastic")],
    )
    problem = compiled.problem
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    params = unpack_parameters(x0, names, fixed)
    analytic = _evaluate_problem_jacobian(problem, params, names, require_positive_sigma=True)

    def residual_fn(x):
        return _evaluate_problem(
            problem, unpack_parameters(x, names, fixed), require_positive_sigma=True
        ).residuals

    fd = _finite_difference_jacobian(residual_fn, x0, residual_fn(x0), bounds)
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=1e-6)


def test_analytic_and_numeric_jacobians_recover_same_fit():
    from nfit.fitting import OptimizationConfig

    # Synthesize data from the model, then fit from a perturbed start with the
    # analytic Jacobian and confirm it recovers the generating parameters.
    truth = _rpa_component()
    data = _rpa_points(5.0, 7, n=120)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("T5", data, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    model_values = evaluate_problem_model(
        compiled_truth.problem,
        "T5",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted_points = PointData4D(
        data.H,
        data.K,
        data.L,
        data.E,
        model_values,
        np.full(data.size, 0.02),
        temperature=data.temperature,
        metadata=dict(data.metadata),
    )
    start = _rpa_component(
        parameters={"chi0": 0.2, "gamma0": 2.5, "J1": 0.05, "J2": 0.0}
    )
    compiled = compile_fit_problem(
        [start], [FitDatasetInput("T5", fitted_points, data_type="single_crystal_inelastic")]
    )
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    # Noise-free data drawn from the model must be fit essentially perfectly.
    assert result.reduced_chi2 < 1e-6
    p = result.params
    assert p["M.gamma0"] == pytest.approx(2.0, rel=1e-3)
    assert p["M.chi0"] == pytest.approx(0.3, rel=1e-3)
    assert p["M.chi0"] * p["M.J1"] == pytest.approx(0.3 * 0.1, rel=1e-3)
    assert p["M.chi0"] * p["M.J2"] == pytest.approx(0.3 * -0.05, rel=1e-3)


def test_heisenberg_rpa_fit_is_backend_invariant():
    """A full fit must converge to the same result on numpy and numba backends.

    The numba path uses the Jacobi eigensolver (different from LAPACK), so this
    locks that the accelerated backend does not change fit results beyond
    floating-point noise.
    """
    pytest.importorskip("numba")
    from nfit import spin_fluctuations as sf
    from nfit.fitting import OptimizationConfig

    # A large-enough single-crystal problem to exercise the numba eigensolver.
    truth = _rpa_component()
    data = _rpa_points(5.0, 11, n=3000)
    compiled_truth = compile_fit_problem(
        [truth], [FitDatasetInput("d", data, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    model_values = evaluate_problem_model(
        compiled_truth.problem,
        "d",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    fitted = PointData4D(
        data.H,
        data.K,
        data.L,
        data.E,
        model_values,
        np.full(data.size, 0.02),
        temperature=data.temperature,
        metadata=dict(data.metadata),
    )

    def fit_with(backend):
        sf.set_rpa_backend(backend)
        try:
            start = _rpa_component(
                parameters={"scale": 1.0, "chi0": 0.2, "gamma0": 2.5, "J1": 0.05, "J2": 0.0}
            )
            compiled = compile_fit_problem(
                [start], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
            )
            return fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
        finally:
            sf.set_rpa_backend("auto")

    result_numba = fit_with("numba")
    result_numpy = fit_with("numpy")
    assert result_numba.success and result_numpy.success
    # Both backends reach the same-quality minimum. Individual parameters can
    # differ because the RPA response is invariant under chi0 -> a*chi0,
    # J -> J/a, scale -> scale/a (a flat valley of equally good fits), so the
    # invariant is the fit quality and the physical model prediction.
    assert result_numba.reduced_chi2 == pytest.approx(result_numpy.reduced_chi2, rel=1e-6)
    compiled = compile_fit_problem(
        [_rpa_component()], [FitDatasetInput("d", fitted, data_type="single_crystal_inelastic")]
    )
    from nfit.fitting import evaluate_problem_model

    pred_numba = evaluate_problem_model(compiled.problem, "d", result_numba.params)
    pred_numpy = evaluate_problem_model(compiled.problem, "d", result_numpy.params)
    np.testing.assert_allclose(pred_numba, pred_numpy, rtol=1e-4, atol=1e-6)


def test_dataset_magnetic_field_validator_gives_actionable_error():
    from nfit.fit_config import _dataset_magnetic_field

    points = _rpa_points(5.0, 3)
    with pytest.raises(ValueError, match="Conditions"):
        _dataset_magnetic_field(points)
    points = points.with_updates(magnetic_field=np.array([0.0, 0.0, 1.5]))
    np.testing.assert_array_equal(_dataset_magnetic_field(points), np.array([0.0, 0.0, 1.5]))
