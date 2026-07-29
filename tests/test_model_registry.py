from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nfit
from nfit import (
    MODEL_TYPE_REGISTRY,
    DataGroup,
    FitDatasetInput,
    ModelComponentSpec,
    ModelDefinition,
    ModelParameterDefinition,
    ModelPlotDefinition,
    compile_fit_problem,
    model_definition,
    model_plot_definitions,
    register_model_definition,
    serialize_model_component,
)
from nfit.dataset import from_arrays
from nfit.fit_config import compute_component_diagnostics
from nfit.model_registry import (
    default_model_config,
    default_model_fit_parameters,
    default_model_parameters,
)
from nfit.project_gui import create_model_component
from nfit.project_io import _model_to_dict
from nfit.workflow import _model_spec


def _temporary_definition(key: str, **overrides) -> ModelDefinition:
    values = {
        "key": key,
        "label": "Test model",
        "description": "A model used to exercise the shared registry.",
        "data_types": ("single_crystal_inelastic",),
        "factory": lambda _component: lambda data, _params: np.zeros(data.size),
        "parameter_fields": (
            ModelParameterDefinition(
                name="amplitude",
                default=2.0,
                description="Test amplitude.",
                allowed="Any finite number.",
                type="float",
                example="3.0",
                unit="arb.",
            ),
        ),
    }
    values.update(overrides)
    return ModelDefinition(**values)


def test_builtin_registry_is_the_gui_and_fit_source_of_truth():
    definition = model_definition("local_relaxational")

    assert all(
        isinstance(registered, ModelDefinition)
        for registered in MODEL_TYPE_REGISTRY.values()
    )
    assert definition.parameters == ("chi_loc", "gamma")
    assert tuple(field.name for field in definition.parameter_fields) == (
        "chi_loc",
        "gamma",
    )
    assert default_model_parameters("local_relaxational") == {
        "chi_loc": 1.0,
        "gamma": 2.0,
    }
    assert default_model_config("local_relaxational") == {
        "ion": "",
        "form_factor_coefficients": "",
        "bulk_g_factor": 2.0,
        "magnetic_ions_per_formula_unit": 1.0,
    }
    assert default_model_fit_parameters("local_relaxational") == {
        "chi_loc": False,
        "gamma": False,
    }
    assert not hasattr(nfit, "MODEL_TYPE_DEFINITIONS")
    assert not hasattr(nfit, "ModelTypeInfo")


def test_generalized_paramagnon_registry_exposes_complete_extension_contract():
    definition = model_definition("generalized_paramagnon")
    assert definition.diagnostics is not None
    assert definition.report_sections is not None
    assert definition.validate_component is not None
    assert definition.documentation == "generalized_paramagnon.md"
    assert set(definition.data_types) == {
        "single_crystal_inelastic",
        "powder_inelastic",
        "single_crystal_elastic",
        "powder_elastic",
        "magnetization",
    }
    plot = model_plot_definitions("generalized_paramagnon")[0]
    result = plot.calculate(
        [-2.0, 0.0, 2.0],
        spatial_kernels=[1.0, 2.0],
        chi_peak=1.5,
        gamma0=2.0,
        inverse_mode_energy_sq=0.04,
    )
    assert result["susceptibility"].shape == (2, 3)
    script = plot.script(
        energy=[-2.0, 0.0, 2.0],
        spatial_kernels=[1.0, 2.0],
        chi_peak=1.5,
        gamma0=2.0,
        inverse_mode_energy_sq=0.04,
    )
    assert "generalized_paramagnon_energy_scan" in script
    compile(script, "<model-plot>", "exec")


def test_every_physical_model_has_a_standard_documentation_page():
    docs = Path(__file__).resolve().parents[1] / "docs"
    physical = [
        definition
        for definition in MODEL_TYPE_REGISTRY.values()
        if definition.category != "background"
    ]

    for definition in physical:
        page = docs / definition.documentation
        assert page.is_file(), definition.key
        text = page.read_text()
        assert "## Calculable data" in text, definition.key
        assert "## Parameters" in text, definition.key


@pytest.mark.parametrize(
    "model_type",
    ["local_relaxational", "mmp_relaxational", "generalized_paramagnon"],
)
def test_scalar_bulk_models_reject_invalid_normalization(model_type):
    component = ModelComponentSpec(
        name="response",
        type=model_type,
        config={
            **default_model_config(model_type),
            "bulk_g_factor": 0.0,
        },
    )

    with pytest.raises(ValueError, match="bulk_g_factor"):
        nfit.validate_model_component(component)


def test_registered_model_drives_creation_diagnostics_plots_and_serialization():
    key = "_registry_contract_test"

    def diagnostics(_component, _data, params):
        return {"resolved_amplitude": params["test.amplitude"]}

    def project_serializer(component):
        payload = {
            "name": component.name,
            "type": component.type,
            "parameters": dict(component.parameters),
            "config": dict(component.config),
            "fit_parameters": dict(component.fit_parameters),
            "sharing": dict(component.sharing),
            "limits": dict(component.limits),
            "constraints": list(component.constraints),
            "applies_to": component.applies_to,
            "enabled": component.enabled,
            "metadata": {**component.metadata, "serialized_by": "project"},
        }
        return payload

    def workflow_serializer(component):
        payload = project_serializer(component)
        payload["metadata"]["serialized_by"] = "workflow"
        return payload

    plot = ModelPlotDefinition(
        key="summary",
        label="Summary",
        description="A scriptable model summary.",
        calculate=lambda component: component.parameters["amplitude"],
    )
    definition = _temporary_definition(
        key,
        diagnostics=diagnostics,
        plots=(plot,),
        project_serializer=project_serializer,
        workflow_serializer=workflow_serializer,
    )
    register_model_definition(definition)
    try:
        component = create_model_component(DataGroup("group"), "test", type=key)
        assert component.parameters == {"amplitude": 2.0}
        assert component.fit_parameters == {"amplitude": False}
        assert model_plot_definitions(key) == (plot,)

        points = from_arrays(
            H=[0.0],
            K=[0.0],
            L=[0.0],
            E=[1.0],
            intensity=[0.0],
            sigma=[1.0],
        )
        record = compute_component_diagnostics(
            component,
            points,
            {"test.amplitude": 4.5},
        )
        assert record == {"resolved_amplitude": 4.5}
        assert _model_to_dict(component)["metadata"]["serialized_by"] == "project"
        assert _model_spec(component)["metadata"]["serialized_by"] == "workflow"
        assert (
            serialize_model_component(component, purpose="project")["type"] == key
        )
    finally:
        MODEL_TYPE_REGISTRY.pop(key, None)


def test_registration_rejects_duplicate_keys_and_invalid_bounds():
    key = "_registry_validation_test"
    definition = _temporary_definition(key)
    register_model_definition(definition)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_model_definition(definition)
    finally:
        MODEL_TYPE_REGISTRY.pop(key, None)

    invalid = _temporary_definition(
        key,
        default_lower_bounds=(("missing", 0.0),),
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        register_model_definition(invalid)


def test_fit_compilation_runs_the_registered_component_validator():
    key = "_registry_validator_test"

    def validate(component):
        if component.config.get("invalid"):
            raise ValueError("invalid test configuration")

    register_model_definition(
        _temporary_definition(key, validate_component=validate)
    )
    try:
        component = create_model_component(DataGroup("group"), "test", type=key)
        component.config["invalid"] = True
        points = from_arrays(
            H=[0.0],
            K=[0.0],
            L=[0.0],
            E=[1.0],
            intensity=[0.0],
            sigma=[1.0],
        )
        with pytest.raises(ValueError, match="invalid test configuration"):
            compile_fit_problem(
                [component],
                [
                    FitDatasetInput(
                        name="data",
                        data=points,
                        data_type="single_crystal_inelastic",
                    )
                ],
            )
    finally:
        MODEL_TYPE_REGISTRY.pop(key, None)


def test_serialization_rejects_a_hook_that_changes_model_type():
    key = "_registry_serializer_test"
    definition = _temporary_definition(
        key,
        project_serializer=lambda _component: {"type": "different"},
    )
    register_model_definition(definition)
    try:
        component = ModelComponentSpec(name="test", type=key)
        with pytest.raises(ValueError, match="changed the model type"):
            serialize_model_component(component)
    finally:
        MODEL_TYPE_REGISTRY.pop(key, None)
