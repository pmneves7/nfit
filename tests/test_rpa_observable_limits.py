"""Equivalent scalar and tensor models must share dataset normalization."""

import copy

import numpy as np
import pytest

from nfit import BasisState, build_electronic_model, model_definition
from nfit.cross_section import chipp_from_cross_section
from nfit.dataset import PointData4D
from nfit.fit_config import FitDatasetInput, compile_fit_problem
from nfit.fitting import evaluate_problem_model
from nfit.pipeline import ModelComponentSpec
from nfit.spectral_channels import spectral_polarization_factor
from tests.fit_config_test_support import _pyrochlore_tensor_component, _tensor_points


@pytest.mark.parametrize("moment_unit", ["spin_squared", "mu_B_squared"])
@pytest.mark.parametrize("representation", ["chi_double_prime", "cross_section"])
@pytest.mark.parametrize("mode,factor", [
    ("isotropic_single_component", 2.0), ("isotropic_trace", 2.0 / 3.0),
    ("already_corrected", 1.0), ("custom_scalar", 1.7),
])
def test_isotropic_tensor_matches_scalar_spectral_conventions(moment_unit, representation, mode, factor):
    _, tensor = _pyrochlore_tensor_component(fit_aniso=False)
    for name in ("J1_S1", "J1_S2", "J1_D1"):
        tensor.parameters[name] = 0.0
    tensor.fit_parameters = {}
    scalar = copy.deepcopy(tensor)
    scalar.config["anisotropy"]["J1"]["enabled"] = False
    points = _tensor_points(17, n=4)
    convention = dict(
        fit_representation=representation, moment_unit=moment_unit, g_factor=2.3,
        kf_ki_state="removed", polarization_mode=mode, polarization_scalar=factor,
    )
    assert spectral_polarization_factor(convention) == factor

    def evaluate(component, metadata):
        data = points.with_updates(metadata={**points.metadata, "spectral_observable": metadata})
        compiled = compile_fit_problem([
            component
        ], [FitDatasetInput("d", data, data_type="single_crystal_inelastic")])
        params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
        return evaluate_problem_model(compiled.problem, "d", params)

    actual = evaluate(tensor, convention)
    np.testing.assert_allclose(actual, evaluate(scalar, convention), rtol=2e-12, atol=1e-13)
    if representation == "chi_double_prime":
        single_component = {**convention, "polarization_mode": "isotropic_single_component"}
        np.testing.assert_allclose(actual, evaluate(scalar, single_component) * 2.0 / factor)


@pytest.mark.parametrize("moment_unit", ["spin_squared", "mu_B_squared"])
@pytest.mark.parametrize("mode,factor", [("isotropic_single_component", 2.0), ("custom_scalar", 1.7)])
def test_compiled_electronic_chi_channel_matches_projected_cross_section_inversion(moment_unit, mode, factor):
    electronic_model = build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=[BasisState("s", site="A", orbital="s")],
        hoppings={(1, 0, 0): [[-10.0]]}, orbital_centers=[[0.25, 0.0, 0.0]],
        periodic_axes=(0,), energy_unit="meV",
    )
    bands = ModelComponentSpec(name="bands", type="tight_binding", config={
        **{field.name: field.default for field in model_definition("tight_binding").config_fields},
        "model_data": electronic_model.to_dict(), "periodic_axes": [0],
    })
    response = ModelComponentSpec(name="response", type="lindhard", parameters={"broadening": 0.5}, config={
        **{field.name: field.default for field in model_definition("lindhard").config_fields},
        "electronic_component": "bands", "response_mesh": [16], "response_mesh_shift": [0.0],
        "formula_units_mode": "manual", "formula_units_per_cell": 1.0,
    })
    energy = np.array([1.0, 3.0])
    convention = dict(moment_unit=moment_unit, g_factor=2.3, kf_ki_state="removed",
                      normalization_basis="per_formula_unit", normalization_label="",
                      polarization_mode=mode, polarization_scalar=factor)

    def evaluate(representation):
        response_unit = "spin^2/meV/f.u." if moment_unit == "spin_squared" else "mu_B^2/meV/f.u."
        data = PointData4D(
            H=[0.2, 0.37], K=[0, 0], L=[0, 0], E=energy,
            intensity=[0, 0], sigma=[1, 1], temperature=20.0,
            metadata={"spectral_observable": {
                **convention, "fit_representation": representation,
                "unit": "barn/sr/meV/f.u." if representation == "cross_section" else response_unit,
            }},
        )
        compiled = compile_fit_problem([bands, response], [FitDatasetInput("d", data)])
        params = {spec.name: spec.value for spec in compiled.problem.parameter_specs}
        return evaluate_problem_model(compiled.problem, "d", params)

    projected_cross_section = evaluate("cross_section")
    expected = chipp_from_cross_section(
        projected_cross_section, energy, 20.0, polarization=factor,
        moment_unit=moment_unit, g_factor=2.3,
    )
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(evaluate("chi_double_prime"), expected, rtol=2e-12)
