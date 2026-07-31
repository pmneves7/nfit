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


def test_magnetization_supported_and_linear_response():
    assert model_supports_data_type("heisenberg_rpa", "magnetization")

    # Decoupled spins (J1=0): chi_uniform = chi0, so M = scale * g^2 * chi0 * B.
    component = _magnetization_component(chi0=0.4, J1=0.0)
    fields = np.array([1.0, 2.0, 3.0])
    points = _magnetization_points([10.0, 10.0, 10.0], fields, np.zeros(3))
    from nfit.fitting import evaluate_problem_model

    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )
    expected = 1.0 * (2.0**2) * 0.4 * fields  # g=2 default
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_scalar_bulk_susceptibility_does_not_recompute_for_each_temperature(monkeypatch):
    from nfit.fit_config import _RpaComponentEvaluator

    evaluator = _RpaComponentEvaluator(_magnetization_component(chi0=0.4, J1=0.0))
    calls = 0
    original = evaluator._bulk_static_chi

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator, "_bulk_static_chi", counted)
    temperatures = np.linspace(2.0, 300.0, 500)
    fields = np.linspace(0.0099, 0.0101, temperatures.size)
    points = _magnetization_points(
        temperatures,
        fields,
        np.zeros(temperatures.size),
    )
    params = {
        evaluator.chi0_key: 0.4,
        evaluator.gamma0_key: 2.0,
        evaluator.j_keys["J1"]: 0.0,
    }

    evaluator.value(points, params)

    assert calls == 1


def test_bulk_q0_eigensystem_is_reused_across_closure_states(monkeypatch):
    import nfit.fit_config as fit_config
    from nfit.fit_config import _RpaComponentEvaluator

    evaluator = _RpaComponentEvaluator(_magnetization_component(chi0=0.4, J1=0.05))
    params = {
        evaluator.chi0_key: 0.4,
        evaluator.gamma0_key: 2.0,
        evaluator.j_keys["J1"]: 0.05,
    }
    calls = 0
    original = fit_config.np.linalg.eigh

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_config.np.linalg, "eigh", counted)
    evaluator._bulk_static_chi(params, 0.4, 0.0, None)
    evaluator._bulk_static_chi(params, 0.35, 0.02, None)

    assert calls == 1


def test_magnetization_absolute_normalization():
    """Absolute mode includes all magnetic sites in each formula unit."""
    from nfit.fitting import evaluate_problem_model
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI

    component = _magnetization_component(chi0=0.3, J1=0.0)
    component.config["bulk"] = {"enabled": True, "sites_per_fu": 2}
    fields = np.array([0.5, 1.0])
    points = _magnetization_points([5.0, 5.0], fields, np.zeros(2))
    points.metadata.update(
        {"absolute_units": True, "sample_mass_mg": 10.0, "molar_mass_g_mol": 200.0}
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )
    moles = (10.0 / 1000.0) / 200.0
    factor = EMU_PER_MOL_PER_MODEL_CHI * moles * 1.0e4 * 2.0
    expected = 1.0 * factor * (2.0**2) * 0.3 * fields
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_magnetization_absolute_normalization_can_return_mu_b_per_formula_unit():
    """Formula-unit output includes every magnetic site and divides by one mu_B."""
    from nfit.fitting import evaluate_problem_model
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI, EMU_PER_MOL_PER_MU_B

    component = _magnetization_component(chi0=0.3, J1=0.0)
    component.config["bulk"] = {"enabled": True, "sites_per_fu": 2}
    fields = np.array([0.5, 1.0])
    points = _magnetization_points([5.0, 5.0], fields, np.zeros(2))
    points.metadata.update(
        {
            "absolute_units": True,
            "sample_mass_mg": 10.0,
            "molar_mass_g_mol": 200.0,
            "quantity_type": "magnetic_moment",
            "unit": "mu_B/f.u.",
        }
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )
    factor = EMU_PER_MOL_PER_MODEL_CHI * 1.0e4 * 2.0 / EMU_PER_MOL_PER_MU_B
    expected = factor * (2.0**2) * 0.3 * fields
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_absolute_bulk_susceptibility_prediction_does_not_multiply_by_field():
    from nfit.fitting import evaluate_problem_model
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI

    component = _magnetization_component(chi0=0.3, J1=0.0)
    component.config["bulk"] = {"enabled": True, "sites_per_fu": 2}
    points = _magnetization_points([5.0, 5.0], np.array([0.5, 2.0]), np.zeros(2))
    points.metadata.update(
        {
            "absolute_units": True,
            "sample_mass_mg": 10.0,
            "molar_mass_g_mol": 200.0,
            "quantity_type": "bulk_susceptibility",
            "unit": "cm^3/mol",
        }
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )
    expected = EMU_PER_MOL_PER_MODEL_CHI * 2.0 * (2.0**2) * 0.3
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_heisenberg_bulk_normalization_infers_magnetic_sites_per_formula_unit():
    from nfit.fitting import evaluate_problem_model
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI

    component = _magnetization_component(chi0=0.3, J1=0.0)
    component.config["crystal"] = {
        "lattice": {
            "a": 5.0,
            "b": 5.0,
            "c": 5.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "spacegroup": "P 1",
        "sites": [
            {"label": "Fe1", "element": "Fe", "position": [0.0, 0.0, 0.0]},
            {"label": "Fe2", "element": "Fe", "position": [0.5, 0.5, 0.5]},
            {"label": "O1", "element": "O", "position": [0.25, 0.0, 0.0]},
            {"label": "O2", "element": "O", "position": [0.75, 0.0, 0.0]},
            {"label": "O3", "element": "O", "position": [0.0, 0.25, 0.0]},
            {"label": "O4", "element": "O", "position": [0.0, 0.75, 0.0]},
        ],
    }
    points = _magnetization_points([5.0], np.array([1.0]), np.zeros(1))
    points.metadata.update(
        {
            "absolute_units": True,
            "quantity_type": "bulk_susceptibility",
            "unit": "cm^3/mol",
        }
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("m", points, data_type="magnetization")],
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )

    # Fe2O4 reduces to FeO2, hence one magnetic site per formula unit.
    expected = EMU_PER_MOL_PER_MODEL_CHI * 1.0 * 2.0**2 * 0.3
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_absolute_bulk_normalization_matches_direct_si_formula():
    from nfit.fitting import evaluate_problem_model
    from nfit.sum_rules import SI_M3_PER_MOL_PER_MODEL_CHI

    component = _magnetization_component(chi0=0.3, J1=0.0)
    component.config["bulk"] = {"enabled": True, "sites_per_fu": 2}
    points = _magnetization_points([5.0], np.array([1.0]), np.zeros(1))
    points.metadata.update(
        {
            "absolute_units": True,
            "quantity_type": "bulk_susceptibility",
            "unit": "m^3/mol",
        }
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )

    expected = SI_M3_PER_MOL_PER_MODEL_CHI * 2.0 * 2.0**2 * 0.3
    np.testing.assert_allclose(values, expected, rtol=1.0e-14)


def test_linear_response_moment_preserves_signed_field():
    from nfit.fitting import evaluate_problem_model

    component = _magnetization_component(chi0=0.3, J1=0.0)
    fields = np.array([-2.0, -0.5, 0.5, 2.0])
    points = _magnetization_points(
        np.full(fields.shape, 5.0), fields, np.zeros(fields.shape)
    )
    points.metadata["field_direction_cartesian"] = [0.0, 0.0, 1.0]
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("m", points, data_type="magnetization")]
    )
    values = evaluate_problem_model(
        compiled.problem,
        "m",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )

    np.testing.assert_allclose(values, 2.0**2 * 0.3 * fields)


@pytest.mark.parametrize(
    ("model_type", "parameters", "expected_static"),
    [
        (
            "local_relaxational",
            {"chi_loc": 0.35, "gamma": 2.0},
            0.35,
        ),
        (
            "mmp_relaxational",
            {
                "chi_pk": 0.8,
                "xi": 1.5,
                "omega_sf": 2.0,
                "q0_h": 0.5,
                "q0_k": 0.0,
                "q0_l": 0.0,
            },
            0.8 / (1.0 + 1.5**2 * (2.0 * np.pi / 4.0 * 0.5) ** 2),
        ),
        (
            "generalized_paramagnon",
            {
                "chi_peak": 0.8,
                "gamma0": 2.0,
                "relaxation_power": 1.0,
                "inverse_mode_energy_sq": 0.04,
                "xi_x": 1.5,
                "xi_y": 1.5,
                "xi_z": 1.5,
                "xi_yx": 0.0,
                "xi_zx": 0.0,
                "xi_zy": 0.0,
                "q0_h": 0.5,
                "q0_k": 0.0,
                "q0_l": 0.0,
            },
            0.8 / (1.0 + 1.5**2 * (2.0 * np.pi / 4.0 * 0.5) ** 2),
        ),
    ],
)
def test_phenomenological_models_calculate_absolute_bulk_susceptibility(
    model_type, parameters, expected_static
):
    from nfit.fitting import evaluate_problem_model
    from nfit.model_registry import default_model_config
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI

    config = default_model_config(model_type)
    config.update(
        {
            "bulk_g_factor": 2.2,
            "magnetic_ions_per_formula_unit": 2.0,
        }
    )
    component = ModelComponentSpec(
        name="response",
        type=model_type,
        parameters=parameters,
        fit_parameters={},
        config=config,
    )
    points = _magnetization_points(
        [5.0, 100.0],
        np.array([0.1, 4.0]),
        np.zeros(2),
    )
    points.metadata.update(
        {
            "absolute_units": True,
            "quantity_type": "bulk_susceptibility",
            "unit": "cm^3/mol",
            "rlu_to_inv_angstrom_matrix": _rlu_matrix(4.0),
        }
    )
    compiled = compile_fit_problem(
        [component],
        [FitDatasetInput("bulk", points, data_type="magnetization")],
    )
    values = evaluate_problem_model(
        compiled.problem,
        "bulk",
        {spec.name: spec.value for spec in compiled.problem.parameter_specs},
    )
    expected = (
        EMU_PER_MOL_PER_MODEL_CHI * 2.0 * 2.2**2 * expected_static
    )
    np.testing.assert_allclose(values, expected, rtol=1.0e-12)


def test_local_relaxational_joint_fit_uses_one_static_susceptibility():
    from nfit.fitting import fit_problem_least_squares
    from nfit.model_registry import default_model_config
    from nfit.spin_fluctuations import local_relaxational_chipp
    from nfit.sum_rules import EMU_PER_MOL_PER_MODEL_CHI

    truth_chi = 0.42
    truth_gamma = 2.6
    energy = np.linspace(0.2, 8.0, 60)
    neutron = PointData4D(
        H=np.zeros(energy.size),
        K=np.zeros(energy.size),
        L=np.zeros(energy.size),
        E=energy,
        intensity=local_relaxational_chipp(
            energy, chi_loc=truth_chi, gamma=truth_gamma
        ),
        sigma=np.full(energy.size, 1.0e-4),
        temperature=20.0,
        metadata={
            "data_type": "single_crystal_inelastic",
            "spectral_observable": {
                "fit_representation": "chi_double_prime",
                "unit": "spin^2/meV",
                "moment_unit": "spin_squared",
                "g_factor": 2.0,
                "kf_ki_state": "removed",
            },
        },
    )
    bulk_value = EMU_PER_MOL_PER_MODEL_CHI * 2.0**2 * truth_chi
    bulk = _magnetization_points(
        [20.0],
        np.array([1.0]),
        np.array([bulk_value]),
        sigma=1.0e-6,
    )
    bulk.metadata.update(
        {
            "absolute_units": True,
            "sample_mass_mg": 10.0,
            "molar_mass_g_mol": 200.0,
            "quantity_type": "bulk_susceptibility",
            "unit": "cm^3/mol",
        }
    )
    component = ModelComponentSpec(
        name="loc",
        type="local_relaxational",
        parameters={"chi_loc": 0.25, "gamma": 1.5},
        fit_parameters={"chi_loc": True, "gamma": True},
        config=default_model_config("local_relaxational"),
    )
    result = fit_problem_least_squares(
        compile_fit_problem(
            [component],
            [
                FitDatasetInput(
                    "neutron", neutron, data_type="single_crystal_inelastic"
                ),
                FitDatasetInput("bulk", bulk, data_type="magnetization"),
            ],
        ).problem
    )
    assert result.params["loc.chi_loc"] == pytest.approx(truth_chi, rel=1.0e-5)
    assert result.params["loc.gamma"] == pytest.approx(truth_gamma, rel=1.0e-5)


def test_magnetization_point_data_maps_temperature_and_field():
    from nfit.dataset import PointListData
    from nfit.pipeline import DataGroup, DatasetEntry
    from nfit.project_gui import _magnetization_point_data

    columns = {
        "Temperature": np.array([2.0, 300.0]),
        "Magnetic Field": np.array([10000.0, 50000.0]),  # oersted
        "Moment": np.array([0.1, 0.2]),
    }
    view = PointListData(
        columns=columns,
        units={"Temperature": "K", "Magnetic Field": "Oe", "Moment": "emu"},
        coordinate_names=["Temperature", "Magnetic Field"],
        channels=[{"label": "Moment", "value": "Moment", "error": None}],
        metadata={},
    )
    group = DataGroup(name="G")
    dataset = DatasetEntry("m", None)
    dataset.data_type = "magnetization"
    points = _magnetization_point_data(view, group, dataset)
    assert np.all(points.H == 0) and np.all(points.E == 0)
    np.testing.assert_allclose(points.temperature, [2.0, 300.0])
    # Oersted -> tesla (1 T = 1e4 Oe), along default z.
    np.testing.assert_allclose(points.magnetic_field[:, 2], [1.0, 5.0])
    assert points.metadata["data_type"] == "magnetization"
    assert points.metadata["field_direction_cartesian"] == [0.0, 0.0, 1.0]


def test_cofit_ins_and_magnetization_share_parameters():
    """One parameter set generates both INS and chi(T) data; a joint fit with
    shared chi0/J recovers it."""
    from nfit.fitting import (
        OptimizationConfig,
        evaluate_problem_model,
        fit_problem_least_squares,
    )

    ins = _closure_points(21, n=80)
    temps = np.linspace(5.0, 200.0, 40)
    mag = _magnetization_points(temps, np.full(temps.size, 1.0), np.zeros(temps.size))

    truth = _closure_scalar_component(chi0=0.4, J1=0.08)
    truth.parameters["scale"] = 1.0
    truth_mag = _magnetization_component(chi0=0.4, J1=0.08, scale=1.0)

    compiled_truth = compile_fit_problem(
        [truth],
        [FitDatasetInput("ins", ins, data_type="single_crystal_inelastic")],
    )
    ins_values = evaluate_problem_model(
        compiled_truth.problem,
        "ins",
        {spec.name: spec.value for spec in compiled_truth.problem.parameter_specs},
    )
    compiled_truth_mag = compile_fit_problem(
        [truth_mag], [FitDatasetInput("mag", mag, data_type="magnetization")]
    )
    mag_values = evaluate_problem_model(
        compiled_truth_mag.problem,
        "mag",
        {spec.name: spec.value for spec in compiled_truth_mag.problem.parameter_specs},
    )

    ins_fit = PointData4D(
        ins.H, ins.K, ins.L, ins.E, ins_values, np.full(ins.size, 0.01), temperature=ins.temperature
    )
    mag_fit = _magnetization_points(temps, np.full(temps.size, 1.0), mag_values, sigma=0.001)

    # One shared component fits both datasets. chi0 and J are global; scale is
    # fixed (=1) so absolute intensity pins chi0 rather than absorbing it.
    start = _closure_scalar_component(chi0=0.2, J1=0.02)
    start.parameters["scale"] = 1.0
    start.fit_parameters = {"chi0": True, "J1": True, "gamma0": True}
    compiled = compile_fit_problem(
        [start],
        [
            FitDatasetInput("ins", ins_fit, data_type="single_crystal_inelastic"),
            FitDatasetInput("mag", mag_fit, data_type="magnetization"),
        ],
    )
    assert {d.name for d in compiled.problem.datasets} == {"ins", "mag"}
    result = fit_problem_least_squares(compiled.problem, config=OptimizationConfig())
    assert result.success
    assert result.params["M.chi0"] == pytest.approx(0.4, rel=1e-3)
    assert result.params["M.J1"] == pytest.approx(0.08, rel=1e-2)
