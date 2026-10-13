"""Independent full-cell references for RPA fit orchestration and limits."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from nfit.closures import TierAMoments, TierBMoments, solve_onsager
from nfit.cross_section import KB_MEV_PER_K, intensity_from_chipp
from nfit.dataset import PointData4D
from nfit.fit_config import (
    FitDatasetInput,
    _RpaComponentEvaluator,
    compile_fit_problem,
    compute_component_diagnostics,
)
from nfit.fitting import evaluate_problem_model, fit_problem_least_squares
from nfit.pipeline import ModelComponentSpec
from nfit.spin_fluctuations import (
    build_rpa_geometry,
    heisenberg_rpa_chipp,
    rpa_exchange_matrix,
)
from nfit.sum_rules import MomentIntegrationError, bz_sample_hkl
from nfit.tensor_rpa import (
    MU_B_MEV_PER_T,
    assemble_tensor_exchange,
    build_tensor_structure,
    tensor_rpa_unpolarized_chipp,
)

LATTICE = dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0)


def _chain_component(count=3, *, centered=False, tensor=False, closure=True):
    direction = np.ones(3) if centered else np.array([1.0, 0.0, 0.0])
    positions = [((index / count) * direction).tolist() for index in range(count)]
    bonds = [
        dict(site_i=index, site_j=(index + 1) % count,
             offset=(direction if index == count - 1 else np.zeros(3)).tolist())
        for index in range(count)
    ]
    config = dict(site_positions=positions, orbits=[dict(label="J1", bonds=bonds)],
                  crystal=dict(lattice=LATTICE))
    parameters = dict(chi0=0.8, gamma0=2.0, J1=0.4)
    if closure:
        config["closure"] = dict(mode="onsager", moment_target=8.0,
                                 energy_cutoff_mev=50.0, bz_grid=6)
    if tensor:
        identity = np.eye(3).tolist()
        rotation = np.diag([-1.0, -1.0, 1.0]).tolist()
        for index, bond in enumerate(bonds):
            bond.update(rotation=identity if index % 2 == 0 else rotation,
                        reversed=bool(index % 2))
        config["site_rotations"] = [identity if i % 2 == 0 else rotation
                                    for i in range(count)]
        config["anisotropy"] = {
            "J1": dict(enabled=True, basis=[dict(name="S1", matrix=np.diag([2., -1., -1.]).tolist())])
        }
        config["sia"] = {
            "site": dict(enabled=True, sites=list(range(count)),
                         basis=[dict(name="K1", matrix=np.diag([1., -1., 0.]).tolist())])
        }
        parameters.update(J1_S1=0.025, K1_site=0.03)
    return ModelComponentSpec(name="R", type="heisenberg_rpa",
                              parameters=parameters, config=config)


def _points(temperature=30.0, *, energies=(0.2, 1.0, 3.0), bulk=False):
    n = len(energies)
    return PointData4D(
        H=np.linspace(0.11, 0.61, n) if not bulk else np.zeros(n),
        K=np.zeros(n), L=np.zeros(n), E=np.asarray(energies),
        intensity=np.zeros(n), sigma=np.ones(n), temperature=temperature,
        magnetic_field=[0.0, 0.0, 1.0],
        metadata={"rlu_to_inv_angstrom_matrix": (2 * np.pi * np.eye(3)).tolist()},
    )


def _compiled(component, points, data_type="single_crystal_inelastic"):
    compiled = compile_fit_problem([component], [FitDatasetInput("d", points, data_type=data_type)])
    params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
    return compiled, params


@pytest.mark.parametrize("count,centered", [(3, False), (2, True)])
@pytest.mark.parametrize("tensor", [False, True])
def test_compiled_closure_retains_all_full_cell_trace_sectors(count, centered, tensor):
    component = _chain_component(count, centered=centered, tensor=tensor)
    points = _points()
    compiled, params = _compiled(component, points)
    actual = evaluate_problem_model(compiled.problem, "d", params)
    record = compute_component_diagnostics(component, points, params)
    config = component.config
    original_config = copy.deepcopy(config)
    grid = bz_sample_hkl(config["closure"]["bz_grid"])
    geometry = build_rpa_geometry(*grid.T, config["site_positions"], config["orbits"])
    neutron_geometry = build_rpa_geometry(points.H, points.K, points.L,
                                           config["site_positions"], config["orbits"])
    if tensor:
        def structure(geom):
            return build_tensor_structure(
                geom, config["site_positions"], orbits=config["orbits"], lattice=LATTICE,
                anisotropy=config["anisotropy"], sia=config["sia"],
                site_rotations=config["site_rotations"],
            )
        exchange = assemble_tensor_exchange(structure(geometry), component.parameters)
        model = TierAMoments(np.linalg.eigvalsh(exchange), count, isotropic_components=1)
    else:
        exchange = rpa_exchange_matrix(geometry, {"J1": 0.4})
        model = TierAMoments(np.linalg.eigvalsh(exchange), count)
    expected_closure = solve_onsager(model, chi0=0.8, gamma0=2.0, temperature_K=30.0,
                                     cutoff_mev=50.0, target=8.0)
    assert record["lambda_shift"] == pytest.approx(expected_closure.lambda_shift, abs=2e-11)
    assert sum(model.moment(chi0=0.8, gamma0=2.0, temperature_K=30.0,
                            cutoff_mev=50.0, lambda_shift=record["lambda_shift"])) == pytest.approx(8.0)
    if tensor:
        response = tensor_rpa_unpolarized_chipp(
            structure(neutron_geometry), neutron_geometry, points.E,
            np.tile([1.0, 0.0, 0.0], (points.size, 1)),
            chi0=0.8, gamma0=2.0, param_values=component.parameters,
            lambda_shift=expected_closure.lambda_shift,
        )
    else:
        response = heisenberg_rpa_chipp(neutron_geometry, points.E, chi0=0.8,
                                       gamma0=2.0, j_values={"J1": 0.4},
                                       lambda_shift=expected_closure.lambda_shift)
    expected = intensity_from_chipp(response, points.E, 30.0, polarization=1 if tensor else 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-11)
    evaluator = _RpaComponentEvaluator(component)
    assert len(evaluator.site_positions) == 1  # Neutron reduction remains active.
    assert evaluator._closure_context()[0].n_sites == count
    assert config == original_config


@pytest.mark.parametrize("mode", ["onsager", "scr", "tac"])
def test_compiled_magnetization_closures_preserve_exact_temperature_keys(mode):
    component = _chain_component(1)
    component.config["closure"] = dict(mode=mode, moment_target=1.0, bz_grid=2)
    component.parameters.update(J1=0.0, mode_coupling_u=0.1)
    temperatures = np.array([10.123456789123, 10.123456789223, 10.123456789123])
    points = _points(temperatures, energies=(0.0, 0.0, 0.0), bulk=True)
    compiled, params = _compiled(component, points, "magnetization")
    actual = evaluate_problem_model(compiled.problem, "d", params)
    assert np.all(np.isfinite(actual)) and np.all(actual < 1e6)
    for index, temperature in enumerate(temperatures):
        single = _points(float(temperature), energies=(0.0,), bulk=True)
        reference, reference_params = _compiled(component, single, "magnetization")
        expected = evaluate_problem_model(reference.problem, "d", reference_params)
        assert actual[index] == pytest.approx(expected[0], rel=2e-10)


def test_magnetization_batch_failure_isolated_per_temperature(monkeypatch):
    component = _chain_component(1)
    component.config["closure"] = dict(mode="tac", moment_target=1.0, bz_grid=2)
    component.parameters["J1"] = 0.0
    original_moment = TierAMoments.moment

    def moment(self, **kwargs):
        if kwargs["temperature_K"] == 5.0:
            raise ValueError("no stable closure solution at this temperature")
        return original_moment(self, **kwargs)

    monkeypatch.setattr(TierAMoments, "moment", moment)
    points = _points(np.array([10.123456789123, 5.0, 10.123456789123]),
                     energies=(0.0, 0.0, 0.0), bulk=True)
    compiled, params = _compiled(component, points, "magnetization")
    actual = evaluate_problem_model(compiled.problem, "d", params)
    assert actual[1] == 1e6
    assert actual[0] == actual[2] and 0 < actual[0] < 1e6
    single = _points(10.123456789123, energies=(0.0,), bulk=True)
    reference, reference_params = _compiled(component, single, "magnetization")
    expected = evaluate_problem_model(reference.problem, "d", reference_params)[0]
    np.testing.assert_allclose(actual[[0, 2]], np.full(2, expected))


def test_unstable_scr_magnetization_trial_returns_penalty():
    component = _chain_component(1)
    component.config["closure"] = dict(mode="scr", bz_grid=4)
    component.parameters.update(J1=5.0, mode_coupling_u=0.0)
    points = _points(10.0, energies=(0.0,), bulk=True)
    compiled, params = _compiled(component, points, "magnetization")
    np.testing.assert_array_equal(evaluate_problem_model(compiled.problem, "d", params), [1e6])


@pytest.mark.parametrize("mode", ["tensor", "zeeman"])
def test_static_diagnostics_keep_one_component_isotropic_normalization(mode):
    scalar = _chain_component(1, closure=False)
    scalar.parameters["J1"] = 0.0
    points = _points(10.0, energies=(1.0,)).with_updates(magnetic_field=[0.0, 0.0, 0.0])
    baseline, params = _compiled(scalar, points)
    reference = compute_component_diagnostics(scalar, points, params)
    tensor = copy.deepcopy(scalar)
    if mode == "tensor":
        tensor.config["anisotropy"] = {"J1": dict(enabled=True, basis=[])}
    else:
        tensor.config["zeeman"] = dict(enabled=True)
        tensor.parameters.update(g_factor=2.0, chi_perp_ratio=1.0, gamma_perp_ratio=1.0)
    compiled, tensor_params = _compiled(tensor, points)
    actual = compute_component_diagnostics(tensor, points, tensor_params)
    assert actual is not None and actual["unstable"] == 0
    for key in ("chi_static_q0", "chi_static_qpeak"):
        assert actual[key] == pytest.approx(reference[key])
    np.testing.assert_allclose(evaluate_problem_model(compiled.problem, "d", tensor_params),
                               evaluate_problem_model(baseline.problem, "d", params))


def test_compiled_scalar_zero_energy_jacobian_matches_value_derivatives():
    component = _chain_component(1, closure=False)
    component.parameters.update(chi0=0.5, J1=0.1, inverse_mode_energy_sq=0.03)
    points = _points(10.0, energies=(0.0, 0.5, 1.4))
    compiled, params = _compiled(component, points)
    dataset = compiled.problem.datasets[0]
    columns = dataset.model_jacobian(dataset.data, params)
    assert np.all(np.isfinite(evaluate_problem_model(compiled.problem, "d", params)))
    for parameter in ("chi0", "gamma0", "J1", "inverse_mode_energy_sq"):
        key = f"R.{parameter}"
        step = 1e-6
        plus, minus = dict(params), dict(params)
        plus[key] += step
        minus[key] -= step
        expected = (evaluate_problem_model(compiled.problem, "d", plus)
                    - evaluate_problem_model(compiled.problem, "d", minus)) / (2 * step)
        assert np.all(np.isfinite(columns[key]))
        np.testing.assert_allclose(columns[key], expected, rtol=2e-7, atol=2e-10)
    assert columns["R.inverse_mode_energy_sq"][0] == 0.0


def test_compiled_scalar_fit_recovers_chi_with_exact_zero_energy_points():
    truth = _chain_component(1, closure=False)
    truth.parameters.update(chi0=0.5, J1=0.1)
    truth.fit_parameters = {name: False for name in truth.parameters}
    energies = (-0.2, 0.0, 0.2, -0.5, 0.0, 0.5, -1.0, 0.0, 1.0)
    points = _points(10.0, energies=energies).with_updates(sigma=np.full(len(energies), 1e-3))
    reference, truth_params = _compiled(truth, points)
    observed = evaluate_problem_model(reference.problem, "d", truth_params)
    start = copy.deepcopy(truth)
    start.parameters["chi0"] = 0.3
    start.fit_parameters["chi0"] = True
    compiled, _ = _compiled(start, points.with_updates(intensity=observed))
    assert compiled.problem.datasets[0].model_jacobian is not None
    result = fit_problem_least_squares(compiled.problem)
    assert result.success
    assert result.params["R.chi0"] == pytest.approx(0.5, rel=1e-8)


@pytest.mark.parametrize("mode", ["scalar", "tensor", "zeeman"])
@pytest.mark.parametrize("temperature", [0.0, 10.0])
def test_compiled_zero_energy_prediction_matches_two_sided_limit(mode, temperature):
    component = _chain_component(1, tensor=mode == "tensor", closure=False)
    component.parameters.update(chi0=0.5, J1=0.1)
    if mode == "zeeman":
        component.config["zeeman"] = dict(enabled=True)
        component.parameters.update(g_factor=2.1, chi_perp_ratio=1.2, gamma_perp_ratio=0.8)
    zero = _points(temperature, energies=(0.0,))
    compiled, params = _compiled(component, zero)
    actual = evaluate_problem_model(compiled.problem, "d", params)
    assert np.all(np.isfinite(actual))
    neighbors = []
    for energy in (-1e-7, 1e-7):
        points = zero.with_updates(E=[energy])
        reference, reference_params = _compiled(component, points)
        neighbors.append(evaluate_problem_model(reference.problem, "d", reference_params))
    if temperature == 0.0:
        np.testing.assert_array_equal(actual, [0.0])
        assert abs(neighbors[1][0]) < 1e-7
    else:
        np.testing.assert_allclose(actual, np.mean(neighbors, axis=0), rtol=2e-10)


@pytest.mark.parametrize("mode", ["onsager", "scr", "tac"])
def test_compiled_closure_zero_energy_uses_solved_local_response(mode):
    component = _chain_component(1)
    component.config["closure"] = dict(mode=mode, moment_target=1.0, bz_grid=4)
    component.parameters.update(chi0=0.5, J1=0.1, mode_coupling_u=0.3)
    points = _points(10.0, energies=(0.0,))
    compiled, params = _compiled(component, points)
    assert compiled.problem.datasets[0].model_jacobian is None
    actual = evaluate_problem_model(compiled.problem, "d", params)
    assert 0 < actual[0] < 1e6
    neighbors = []
    for energy in (-1e-7, 1e-7):
        reference, reference_params = _compiled(component, points.with_updates(E=[energy]))
        neighbors.append(evaluate_problem_model(reference.problem, "d", reference_params))
    np.testing.assert_allclose(actual, np.mean(neighbors, axis=0), rtol=2e-10)


@pytest.mark.parametrize("closure,exchange", [(False, 0.0), (False, 2.0), (True, 0.0)])
def test_diagnostics_distinguish_numerical_integration_failure_from_instability(monkeypatch, closure, exchange):
    component = _chain_component(1, closure=closure)
    component.parameters.update(J1=exchange, g_factor=2.0, chi_perp_ratio=1.0, gamma_perp_ratio=1.0)
    component.config["zeeman"] = dict(enabled=True)
    points = _points(10.0, energies=(1.0,))
    compiled, params = _compiled(component, points)

    def unresolved(self, **kwargs):
        raise MomentIntegrationError("integration budget exhausted")

    monkeypatch.setattr(TierBMoments, "moment", unresolved)
    record = compute_component_diagnostics(component, points, params)
    assert record["numerical_integration_failed"] == 1.0
    assert record["moment_integral_available"] == 0.0
    assert record["diagnostic_error"] == "integration budget exhausted"
    assert "mu_eff_sq" not in record
    if closure:
        assert np.isnan(record["unstable"])
        assert np.isnan(record["chi0_eff"]) and np.isnan(record["lambda_shift"])
        assert record["closure_solution"] == 0.0
        assert record["stability_uses_effective_closure"] == 0.0
        assert record["stability_margin"] == record["bare_stability_margin"]
        np.testing.assert_array_equal(evaluate_problem_model(compiled.problem, "d", params), [1e6])
    else:
        assert record["chi0_eff"] == component.parameters["chi0"]
        assert record["lambda_shift"] == 0.0
        assert record["unstable"] == float(record["stability_margin"] <= 0.0)
        assert bool(record["unstable"]) == (exchange > 0.0)


@pytest.mark.parametrize("closure", [False, True])
def test_compiled_narrow_zeeman_moment_is_invariant_to_field_rotation(closure):
    component = _chain_component(1, closure=closure)
    component.parameters.update(
        chi0=0.5, gamma0=1e-8, J1=0.0,
        g_factor=2.1, chi_perp_ratio=1.3, gamma_perp_ratio=0.7,
    )
    component.config["zeeman"] = dict(enabled=True)
    if closure:
        component.config["closure"].update(moment_target=1.0, bz_grid=2)
    field = np.array([0.35, -0.22, 0.49])
    magnitude = np.linalg.norm(field)
    records = []
    for direction in (field, np.array([0.0, 0.0, magnitude])):
        points = _points(10.0, energies=(1.0,)).with_updates(magnetic_field=direction)
        compiled, params = _compiled(component, points)
        record = compute_component_diagnostics(component, points, params)
        assert record["moment_integral_available"] == 1.0
        assert record["unstable"] == 0.0
        records.append(record)
    for key in ("mu_eff_sq", "m2_zero_point", "m2_thermal", "lambda_shift"):
        assert records[0][key] == pytest.approx(records[1][key], rel=2e-8, abs=1e-9)
    if closure:
        assert records[0]["mu_eff_sq"] == pytest.approx(1.0, abs=1e-9)
    else:
        # Independent narrow-width FDT limit: longitudinal equipartition plus
        # the two transverse delta peaks at the Larmor energy.
        chi0 = component.parameters["chi0"]
        larmor = component.parameters["g_factor"] * MU_B_MEV_PER_T * magnitude
        kt = KB_MEV_PER_K * 10.0
        expected = chi0 * kt + chi0 * 1.3 * larmor / np.tanh(larmor / (2 * kt))
        assert records[0]["mu_eff_sq"] == pytest.approx(expected, rel=2e-6)


def test_compiled_narrow_zeeman_zero_energy_preserves_transverse_response():
    component = _chain_component(1, closure=False)
    component.parameters.update(
        chi0=0.5, gamma0=1e-10, J1=0.0,
        g_factor=2.1, chi_perp_ratio=1.3, gamma_perp_ratio=0.7,
    )
    component.config["zeeman"] = dict(enabled=True)
    field = np.array([0.35, -0.22, 0.49])
    points = _points(10.0, energies=(0.0,)).with_updates(
        H=[field[0]], K=[field[1]], L=[field[2]], magnetic_field=field,
    )
    compiled, params = _compiled(component, points)
    actual = evaluate_problem_model(compiled.problem, "d", params)
    larmor = component.parameters["g_factor"] * MU_B_MEV_PER_T * np.linalg.norm(field)
    chi_perp = component.parameters["chi0"] * component.parameters["chi_perp_ratio"]
    gamma_perp = component.parameters["gamma0"] * component.parameters["gamma_perp_ratio"]
    slope = 2 * chi_perp * gamma_perp / (gamma_perp**2 + larmor**2)
    expected = intensity_from_chipp(0.0, 0.0, 10.0, zero_energy_slope=slope)
    assert actual[0] > 0.0
    assert actual[0] == pytest.approx(float(expected), rel=2e-9)
    for energy in (-1e-13, 1e-13):
        neighbor, neighbor_params = _compiled(component, points.with_updates(E=[energy]))
        value = evaluate_problem_model(neighbor.problem, "d", neighbor_params)
        assert value[0] > 0.0
        assert value[0] == pytest.approx(actual[0], rel=2e-9)
