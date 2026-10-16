"""Physical coordinate and zero-transfer limits of compiled neutron models."""

from __future__ import annotations

import numpy as np
import pytest

from nfit import FitDatasetInput, ModelComponentSpec, PointData4D, compile_fit_problem
from nfit.fitting import (
    evaluate_problem_model,
    fit_problem_least_squares,
    reciprocal_basis_from_lattice_parameters,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.model_registry import default_model_config, default_model_parameters


def _points(hkl, energy, matrix, *, cartesian=False, response_only=True):
    hkl = np.asarray(hkl, dtype=float)
    coordinates = hkl @ matrix.T if cartesian else hkl
    metadata = {
        "coordinate_units": "1/angstrom" if cartesian else "rlu",
        "rlu_to_inv_angstrom_matrix": matrix.tolist(),
    }
    if response_only:
        metadata["spectral_observable"] = {
            "fit_representation": "chi_double_prime",
            "moment_unit": "spin_squared",
            "polarization_mode": "isotropic_single_component",
        }
    return PointData4D(
        *coordinates.T, E=energy, intensity=np.ones(len(hkl)),
        sigma=np.ones(len(hkl)), temperature=20.0, metadata=metadata,
    )


def _component(kind):
    return ModelComponentSpec(
        name="response", type=kind,
        parameters=default_model_parameters(kind), config=default_model_config(kind),
    )


def _compiled(component, points, data_type="single_crystal_inelastic"):
    compiled = compile_fit_problem([component], [FitDatasetInput("scan", points, data_type=data_type)])
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    return compiled.problem, params


def test_mmp_peak_center_is_rlu_even_for_cartesian_fit_points():
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
    hkl = np.asarray([[0.5, 0.25, -0.1], [0.6, 0.1, 0.0], [0.3, 0.4, 0.2]])
    component = _component("mmp_relaxational")
    component.parameters.update(chi_pk=2.0, xi=3.0, omega_sf=2.0, q0_h=0.5, q0_k=0.25, q0_l=-0.1)
    predictions = []
    for cartesian in (False, True):
        problem, params = _compiled(component, _points(hkl, [1.0]*3, matrix, cartesian=cartesian))
        predictions.append(evaluate_problem_model(problem, "scan", params))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-13)
    assert predictions[0][0] == pytest.approx(0.8)


def test_heisenberg_exchange_phases_and_gradients_use_physical_fit_momentum():
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
    hkl = np.asarray([[0.5, 0.25, -0.1], [0.6, 0.1, 0.0], [0.3, 0.4, 0.2]])
    component = ModelComponentSpec(
        name="rpa", type="heisenberg_rpa",
        parameters={"chi0": 0.5, "gamma0": 2.0, "J1": 0.3},
        config={"site_positions": [[0.0, 0.0, 0.0]], "orbits": [
            {"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]},
        ]},
    )
    predictions, gradients = [], []
    for cartesian in (False, True):
        problem, params = _compiled(component, _points(hkl, [1.0]*3, matrix, cartesian=cartesian))
        predictions.append(evaluate_problem_model(problem, "scan", params))
        dataset = problem.datasets[0]
        gradients.append(dataset.model_jacobian(dataset.data, params))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-13)
    for name in gradients[0]:
        np.testing.assert_allclose(gradients[1][name], gradients[0][name], atol=1e-14)


def test_histogram_fit_preparation_retains_cartesian_coordinate_contract():
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    histogram = MDHistoData(
        axes=(
            MDHistoAxis("H", [0.7, 0.9], "1/angstrom", "momentum"),
            MDHistoAxis("K", [-0.1, 0.1], "1/angstrom", "momentum"),
            MDHistoAxis("L", [-0.1, 0.1], "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", [0.5, 1.5], "meV", "energy"),
        ),
        signal=np.ones((1, 1, 1, 1)), errors=np.ones((1, 1, 1, 1)),
        num_events=np.ones((1, 1, 1, 1)), mask=np.zeros((1, 1, 1, 1), dtype=bool),
        metadata={"coordinate_units": "1/angstrom", "rlu_to_inv_angstrom_matrix": matrix.tolist(), "temperature": 20.0},
    )
    points = prepare_histogram_fit_points(histogram)
    assert points.metadata["coordinate_units"] == "1/angstrom"
    np.testing.assert_allclose(points.metadata["rlu_to_inv_angstrom_matrix"], matrix)
    np.testing.assert_allclose(points.H, [0.8])
    np.testing.assert_allclose(points.E, [1.0])


@pytest.mark.parametrize(("kind", "data_type"), [
    ("local_relaxational", "single_crystal_inelastic"),
    ("mmp_relaxational", "single_crystal_inelastic"),
    ("generalized_paramagnon", "single_crystal_inelastic"),
    ("local_relaxational", "powder_inelastic"),
    ("generalized_paramagnon", "powder_inelastic"),
])
def test_regular_models_use_continuous_bose_limit_at_zero_energy(kind, data_type):
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    points = _points([[0.25, 0.0, 0.0]]*3, [-1e-6, 0.0, 1e-6], matrix, response_only=False)
    component = _component(kind)
    problem, params = _compiled(component, points, data_type)
    prediction = evaluate_problem_model(problem, "scan", params)
    assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(prediction[1], 0.5*(prediction[0]+prediction[2]), rtol=1e-10)
    # An actual optimizer must retain the measured zero-transfer point.
    parameter = {"local_relaxational": "chi_loc", "mmp_relaxational": "chi_pk", "generalized_paramagnon": "chi_peak"}[kind]
    points = points.with_updates(intensity=prediction)
    component.fit_parameters[parameter] = True
    component.parameters[parameter] *= 0.7
    problem, _ = _compiled(component, points, data_type)
    result = fit_problem_least_squares(problem)
    assert result.success
    assert result.dataset_sizes["scan"] == 3
    assert result.chi2 < 1e-12


def test_unsupported_zero_energy_bose_limit_raises_actionable_error():
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    points = _points([[0.0, 0.0, 0.0]], [0.0], matrix, response_only=False)
    problem, params = _compiled(_component("conserved_ferromagnetic"), points)
    with pytest.raises(ValueError, match="regular zero-energy dissipative slope"):
        evaluate_problem_model(problem, "scan", params)


def test_neutron_closure_rejects_temperature_sweep_before_instability_penalty():
    matrix = reciprocal_basis_from_lattice_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    points = _points([[0.25, 0.0, 0.0]]*3, [1.0, 2.0, 3.0], matrix)
    points = points.with_updates(temperature=[10.0, 15.0, 20.0])
    component = ModelComponentSpec(
        name="rpa", type="heisenberg_rpa",
        parameters={"chi0": 0.5, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={"gamma0": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [{"label": "J1", "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}]}],
            "closure": {"mode": "tac", "moment_mode": "fixed", "S": 0.5, "bz_grid": 2},
        },
    )
    problem, params = _compiled(component, points)
    with pytest.raises(ValueError, match="single temperature per dataset"):
        evaluate_problem_model(problem, "scan", params)
    with pytest.raises(ValueError, match="single temperature per dataset"):
        fit_problem_least_squares(problem)
