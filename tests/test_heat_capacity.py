import numpy as np
import pytest

from nfit.analysis import AnalysisContext, AnalysisInput, run_analysis_operation
from nfit.dataset import PointData4D, PointListData
from nfit.fit_config import FitDatasetInput, compile_fit_problem, model_supports_data_type
from nfit.fitting import fit_problem_least_squares
from nfit.heat_capacity import (
    debye_heat_capacity,
    debye_temperature_from_beta,
    low_temperature_heat_capacity,
    ppms_heat_capacity_to_molar_mJ,
)
from nfit.pipeline import ModelComponentSpec


def _heat_points(temperature, values, sigma, quantity="heat_capacity", unit="mJ/(mol K)"):
    temperature = np.asarray(temperature, dtype=float)
    zeros = np.zeros(temperature.shape)
    return PointData4D(
        zeros, zeros, zeros, temperature, values, sigma,
        temperature=temperature,
        metadata={"data_type": "heat_capacity", "quantity_type": quantity, "unit": unit},
    )


def test_debye_limits_and_ppms_unit_conversions():
    theta = 300.0
    count = 7.0
    low = debye_heat_capacity(np.array([3.0]), theta, count)[0]
    expected_low = 12.0 * np.pi**4 / 5.0 * count * 8.31446261815324e3 * (3.0 / theta) ** 3
    assert low == pytest.approx(expected_low, rel=1e-8)
    high = debye_heat_capacity(np.array([1.0e6]), theta, count)[0]
    assert high == pytest.approx(3.0 * count * 8.31446261815324e3, rel=1e-7)

    kwargs = {"sample_mass_mg": 5.0, "molar_mass_g_mol": 100.0, "atoms_per_formula_unit": 5.0}
    expected = {
        "uJ/K": 20.0,
        "uJ/(mol K)": 0.001,
        "mJ/(g K)": 100.0,
        "J/(g K)": 100000.0,
        "cal/(g K)": 418400.0,
        "mJ/(mol K)": 1.0,
        "J/(mol K)": 1000.0,
        "cal/(mol K)": 4184.0,
        "J/(gat K)": 5000.0,
        "cal/(gat K)": 20920.0,
    }
    for unit, molar_value in expected.items():
        assert ppms_heat_capacity_to_molar_mJ([1.0], unit, **kwargs)[0] == pytest.approx(molar_value)


def test_debye_and_low_temperature_models_are_heat_capacity_only():
    assert model_supports_data_type("debye_heat_capacity", "heat_capacity")
    assert not model_supports_data_type("debye_heat_capacity", "magnetization")
    assert model_supports_data_type("low_temperature_heat_capacity", "heat_capacity")

    temperature = np.linspace(2.0, 12.0, 80)
    expected = low_temperature_heat_capacity(temperature, 420.0, 0.18)
    points = _heat_points(temperature, expected, np.full(temperature.shape, 0.1))
    component = ModelComponentSpec(
        name="hc",
        type="low_temperature_heat_capacity",
        parameters={"sommerfeld_gamma": 300.0, "debye_beta": 0.1},
        fit_parameters={"sommerfeld_gamma": True, "debye_beta": True},
        limits={"sommerfeld_gamma": [0.0, None], "debye_beta": [0.0, None]},
    )
    compiled = compile_fit_problem(
        [component], [FitDatasetInput("hc-data", points, data_type="heat_capacity")]
    )
    result = fit_problem_least_squares(compiled.problem)
    assert result.params["hc.sommerfeld_gamma"] == pytest.approx(420.0, rel=1e-6)
    assert result.params["hc.debye_beta"] == pytest.approx(0.18, rel=1e-6)


def test_heat_capacity_model_can_fit_c_over_t_channel():
    temperature = np.linspace(2.0, 10.0, 30)
    c_over_t = 380.0 + 0.12 * temperature**2
    points = _heat_points(
        temperature, c_over_t, np.full(temperature.shape, 0.02),
        quantity="heat_capacity_over_temperature", unit="mJ/(mol K^2)",
    )
    component = ModelComponentSpec(
        name="hc", type="low_temperature_heat_capacity",
        parameters={"sommerfeld_gamma": 300.0, "debye_beta": 0.1},
        fit_parameters={"sommerfeld_gamma": True, "debye_beta": True},
    )
    result = fit_problem_least_squares(
        compile_fit_problem([component], [FitDatasetInput("d", points, data_type="heat_capacity")]).problem
    )
    assert result.params["hc.sommerfeld_gamma"] == pytest.approx(380.0, rel=1e-6)
    assert result.params["hc.debye_beta"] == pytest.approx(0.12, rel=1e-6)


def test_low_temperature_heat_capacity_analysis_extracts_gamma_beta_and_theta():
    temperature = np.linspace(2.0, 15.0, 100)
    gamma, beta = 405.0, 0.16
    c = low_temperature_heat_capacity(temperature, gamma, beta)
    data = PointListData(
        {"Puck Temp": np.full(c.shape, np.nan), "Temperature": temperature,
         "C": c, "dC": np.full(c.shape, 0.2)},
        units={"Puck Temp": "K", "Temperature": "K", "C": "mJ/(mol K)", "dC": "mJ/(mol K)"},
        coordinate_names=["Temperature"],
        channels=[{"label": "Heat capacity", "value": "C", "error": "dC",
                   "quantity_type": "heat_capacity", "unit": "mJ/(mol K)"}],
        quantity_types={"Puck Temp": "temperature", "Temperature": "temperature",
                        "C": "heat_capacity", "dC": "heat_capacity"},
    )
    analysis_input = AnalysisInput(
        "a" * 32, "HC", data, AnalysisContext("group", None, None, None, None), "fingerprint"
    )
    result = run_analysis_operation(
        "low_temperature_heat_capacity_fit", [analysis_input],
        {"temperature_min_K": 2.0, "temperature_max_K": 8.0, "atoms_per_formula_unit": 7.0},
    )
    assert result.outputs["sommerfeld_gamma"].value == pytest.approx(gamma, rel=1e-9)
    assert result.outputs["debye_beta"].value == pytest.approx(beta, rel=1e-9)
    assert result.outputs["debye_temperature"].value == pytest.approx(
        debye_temperature_from_beta(beta, 7.0), rel=1e-9
    )
    diagnostic = result.outputs["diagnostic"].data
    assert diagnostic.coordinate_names[0] == "Temperature squared"
    assert diagnostic.metadata["viewer_fit_channel_map"] == {"C/T": "Low-T fit C/T"}


def test_curie_weiss_registered_model_uses_molar_susceptibility_units():
    temperature = np.linspace(50.0, 300.0, 60)
    chi = 0.42 / (temperature + 18.0)
    points = _heat_points(
        temperature, chi, np.full(chi.shape, 1.0e-6),
        quantity="bulk_susceptibility", unit="cm^3/mol",
    )
    component = ModelComponentSpec(
        name="cw", type="curie_weiss",
        parameters={"curie_constant": 0.3, "theta_CW": -10.0},
        fit_parameters={"curie_constant": True, "theta_CW": True},
    )
    result = fit_problem_least_squares(
        compile_fit_problem([component], [FitDatasetInput("m", points, data_type="magnetization")]).problem
    )
    assert result.params["cw.curie_constant"] == pytest.approx(0.42, rel=1e-6)
    assert result.params["cw.theta_CW"] == pytest.approx(-18.0, rel=1e-6)
