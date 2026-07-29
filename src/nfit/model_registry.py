"""Shared definitions and extension hooks for project model components.

The registry is the single public source for fitting compatibility, GUI field
metadata, diagnostics, reports, model plots, citations, and serialization.
Numerical factories remain lazy so this module does not depend on the GUI and
can be imported by scripts and extension packages.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MODEL_TYPE_REGISTRY",
    "ModelConfigDefinition",
    "ModelDefinition",
    "ModelFieldDefinition",
    "ModelParameterDefinition",
    "ModelPlotDefinition",
    "available_model_types",
    "default_model_config",
    "default_model_fit_parameters",
    "default_model_parameters",
    "model_config_tooltip",
    "model_definition",
    "model_parameter_tooltip",
    "model_plot_definitions",
    "register_model_definition",
    "serialize_model_component",
    "validate_model_component",
]

ModelFactory = Callable[[Any], Callable[..., Any]]
ModelJacobianFactory = Callable[[Any], Callable[..., Any] | None]
DynamicParameters = Callable[[Any], tuple[str, ...]]
ModelDiagnostics = Callable[[Any, Any, Mapping[str, float]], Mapping[str, Any] | None]
ModelValidator = Callable[[Any], None]
ModelReportSections = Callable[
    [Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Sequence[tuple[str, str]],
]
ModelSerializer = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class ModelFieldDefinition:
    """One serializable parameter or fixed-configuration field."""

    name: str
    default: Any
    description: str
    allowed: str
    type: str
    example: str
    unit: str = ""
    choices: Any = None

@dataclass(frozen=True)
class ModelParameterDefinition(ModelFieldDefinition):
    """One optimizer-visible model parameter."""


@dataclass(frozen=True)
class ModelConfigDefinition(ModelFieldDefinition):
    """One fixed, serializable model configuration field."""


@dataclass(frozen=True)
class ModelPlotDefinition:
    """One model-owned calculation and plot exposed to scripts and the GUI."""

    key: str
    label: str
    description: str
    calculate: Callable[..., Any]
    render: Callable[..., Any] | None = None
    script: Callable[..., str] | None = None


def _default_component_serializer(component: Any) -> dict[str, Any]:
    """Serialize the common :class:`ModelComponentSpec` scientific state."""

    payload = {
        "name": str(component.name),
        "type": str(component.type),
        "parameters": copy.deepcopy(component.parameters),
        "config": copy.deepcopy(component.config),
        "fit_parameters": {
            str(name): bool(value) for name, value in component.fit_parameters.items()
        },
        "sharing": copy.deepcopy(component.sharing),
        "limits": copy.deepcopy(component.limits),
        "constraints": copy.deepcopy(component.constraints),
        "applies_to": (
            None if component.applies_to is None else [str(name) for name in component.applies_to]
        ),
        "enabled": bool(component.enabled),
        "metadata": copy.deepcopy(component.metadata),
    }
    json.dumps(payload)
    return payload


@dataclass(frozen=True)
class ModelDefinition:
    """Complete public contract for one registered model-component type."""

    key: str
    label: str
    description: str
    data_types: tuple[str, ...]
    factory: ModelFactory
    parameter_fields: tuple[ModelParameterDefinition, ...] = ()
    config_fields: tuple[ModelConfigDefinition, ...] = ()
    version: int = 1
    category: str = "physical"
    dynamic_parameters: DynamicParameters | None = None
    dynamic_parameter_description: str = "Configuration-derived fit parameter {name}."
    dynamic_parameter_unit: str = ""
    jacobian_factory: ModelJacobianFactory | None = None
    validate_component: ModelValidator | None = None
    diagnostics: ModelDiagnostics | None = None
    report_sections: ModelReportSections | None = None
    plots: tuple[ModelPlotDefinition, ...] = ()
    project_serializer: ModelSerializer = _default_component_serializer
    workflow_serializer: ModelSerializer = _default_component_serializer
    default_lower_bounds: tuple[tuple[str, float], ...] = ()
    structured_config: bool = False
    documentation: str = ""
    citations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> tuple[str, ...]:
        """Static parameter names retained for fit-registry compatibility."""

        return tuple(item.name for item in self.parameter_fields)

    def parameter(self, name: str) -> ModelParameterDefinition:
        for item in self.parameter_fields:
            if item.name == name:
                return item
        raise KeyError(f"unknown parameter {name!r} for model {self.key!r}")

    def config(self, name: str) -> ModelConfigDefinition:
        for item in self.config_fields:
            if item.name == name:
                return item
        raise KeyError(f"unknown configuration field {name!r} for model {self.key!r}")

MODEL_TYPE_REGISTRY: dict[str, ModelDefinition] = {}


def register_model_definition(
    definition: ModelDefinition, *, replace: bool = False
) -> None:
    """Register a model definition after validating its shared contract."""

    if not definition.key:
        raise ValueError("model type key cannot be empty")
    if definition.key in MODEL_TYPE_REGISTRY and not replace:
        raise ValueError(f"model type {definition.key!r} is already registered")
    if definition.version < 1:
        raise ValueError(f"model type {definition.key!r} has an invalid version")
    if not definition.data_types:
        raise ValueError(f"model type {definition.key!r} must declare compatible data types")
    if not callable(definition.factory):
        raise TypeError(f"model type {definition.key!r} factory is not callable")
    if definition.validate_component is not None and not callable(
        definition.validate_component
    ):
        raise TypeError(f"model type {definition.key!r} validator is not callable")
    optional_hooks = {
        "dynamic parameter provider": definition.dynamic_parameters,
        "Jacobian factory": definition.jacobian_factory,
        "diagnostics provider": definition.diagnostics,
        "report provider": definition.report_sections,
    }
    for label, hook in optional_hooks.items():
        if hook is not None and not callable(hook):
            raise TypeError(f"model type {definition.key!r} {label} is not callable")
    for label, serializer in (
        ("project serializer", definition.project_serializer),
        ("workflow serializer", definition.workflow_serializer),
    ):
        if not callable(serializer):
            raise TypeError(f"model type {definition.key!r} {label} is not callable")
    parameter_names = [item.name for item in definition.parameter_fields]
    config_names = [item.name for item in definition.config_fields]
    if any(not name for name in (*parameter_names, *config_names)):
        raise ValueError(f"model type {definition.key!r} has an empty field name")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError(f"model type {definition.key!r} has duplicate parameters")
    if len(config_names) != len(set(config_names)):
        raise ValueError(f"model type {definition.key!r} has duplicate configuration fields")
    overlap = set(parameter_names) & set(config_names)
    if overlap:
        raise ValueError(
            f"model type {definition.key!r} uses {sorted(overlap)[0]!r} as both "
            "a parameter and configuration field"
        )
    unknown_bounds = set(dict(definition.default_lower_bounds)) - set(parameter_names)
    if unknown_bounds:
        raise ValueError(
            f"model type {definition.key!r} has a bound for unknown parameter "
            f"{sorted(unknown_bounds)[0]!r}"
        )
    plot_keys = [item.key for item in definition.plots]
    if len(plot_keys) != len(set(plot_keys)):
        raise ValueError(f"model type {definition.key!r} has duplicate plot providers")
    for plot in definition.plots:
        if not plot.key:
            raise ValueError(f"model type {definition.key!r} has an empty plot key")
        if not callable(plot.calculate):
            raise TypeError(
                f"model type {definition.key!r} plot {plot.key!r} calculator "
                "is not callable"
            )
        if plot.render is not None and not callable(plot.render):
            raise TypeError(
                f"model type {definition.key!r} plot {plot.key!r} renderer is not callable"
            )
        if plot.script is not None and not callable(plot.script):
            raise TypeError(
                f"model type {definition.key!r} plot {plot.key!r} script provider "
                "is not callable"
            )
    json.dumps(dict(definition.metadata))
    MODEL_TYPE_REGISTRY[definition.key] = definition


def model_definition(type_name: str) -> ModelDefinition:
    """Return one registered model definition."""

    try:
        definition = MODEL_TYPE_REGISTRY[type_name]
    except KeyError as exc:
        raise KeyError(f"unknown model type {type_name!r}") from exc
    return definition


def available_model_types() -> tuple[str, ...]:
    """Return model type keys in registration order."""

    return tuple(MODEL_TYPE_REGISTRY)


def default_model_parameters(type_name: str) -> dict[str, Any]:
    """Return independent copies of a model's static parameter defaults."""

    return {
        item.name: copy.deepcopy(item.default)
        for item in model_definition(type_name).parameter_fields
    }


def default_model_config(type_name: str) -> dict[str, Any]:
    """Return independent copies of a model's fixed configuration defaults."""

    return {
        item.name: copy.deepcopy(item.default)
        for item in model_definition(type_name).config_fields
    }


def default_model_fit_parameters(type_name: str) -> dict[str, bool]:
    """Return fixed optimizer flags for all static model parameters."""

    return {item.name: False for item in model_definition(type_name).parameter_fields}


def model_parameter_tooltip(type_name: str, parameter_name: str) -> str:
    """Return standard hover text for a static or configuration-derived parameter."""

    definition = model_definition(type_name)
    try:
        item = definition.parameter(parameter_name)
    except KeyError:
        description = definition.dynamic_parameter_description.format(name=parameter_name)
        return "\n".join(
            [
                f"Parameter: {parameter_name}",
                f"Description: {description}",
                "Data type: float",
                f"Units: {definition.dynamic_parameter_unit or 'dimensionless'}",
                "Default: 0",
                "Fit: checked means the optimizer may vary this parameter; unchecked means it is fixed at the displayed value.",
                "Sharing: choose one global value, independent per-dataset values, or named dataset groups.",
            ]
        )
    return "\n".join(
        [
            f"Parameter: {parameter_name}",
            f"Description: {item.description}",
            f"Allowed values: {item.allowed}",
            f"Data type: {item.type}",
            f"Units: {item.unit or 'dimensionless'}",
            f"Default: {_value_text(item.default)}",
            f"Example: {item.example}",
            "Fit: checked means the optimizer may vary this parameter; unchecked means it is fixed at the displayed value.",
            "Sharing: choose one global value, independent per-dataset values, or named dataset groups.",
        ]
    )


def model_config_tooltip(type_name: str, setting_name: str) -> str:
    """Return standard hover text for a fixed model configuration field."""

    item = model_definition(type_name).config(setting_name)
    return "\n".join(
        [
            f"Configuration setting: {setting_name}",
            f"Description: {item.description}",
            f"Allowed values: {item.allowed}",
            f"Data type: {item.type}",
            f"Units: {item.unit or 'dimensionless'}",
            f"Default: {_value_text(item.default)}",
            f"Example: {item.example}",
            "Configuration settings are fixed model options and are not optimized by the fitter.",
        ]
    )


def model_plot_definitions(type_name: str) -> tuple[ModelPlotDefinition, ...]:
    """Return model-owned plot providers in display order."""

    return model_definition(type_name).plots


def validate_model_component(component: Any) -> None:
    """Run a registered model's optional component-level validation hook."""

    definition = MODEL_TYPE_REGISTRY[str(component.type)]
    validator = definition.validate_component
    if validator is not None:
        validator(component)


def serialize_model_component(
    component: Any, *, purpose: str = "project"
) -> dict[str, Any]:
    """Serialize a component through its registered project or workflow hook."""

    definition = model_definition(str(component.type))
    if purpose == "project":
        serializer = definition.project_serializer
    elif purpose == "workflow":
        serializer = definition.workflow_serializer
    else:
        raise ValueError("model serialization purpose must be 'project' or 'workflow'")
    payload = serializer(component)
    if not isinstance(payload, dict):
        raise TypeError(
            f"{definition.key!r} {purpose} serializer must return a dictionary"
        )
    if str(payload.get("type", "")) != definition.key:
        raise ValueError(
            f"{definition.key!r} {purpose} serializer changed the model type"
        )
    required = {
        "name",
        "type",
        "parameters",
        "config",
        "fit_parameters",
        "sharing",
        "limits",
        "constraints",
        "applies_to",
        "enabled",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{definition.key!r} {purpose} serializer omitted required field "
            f"{sorted(missing)[0]!r}"
        )
    extra = set(payload) - required
    if extra:
        raise ValueError(
            f"{definition.key!r} {purpose} serializer added unsupported field "
            f"{sorted(extra)[0]!r}; model-specific state belongs in config or metadata"
        )
    json.dumps(payload)
    return payload


def _value_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return ""
    return str(value)


def _fit_factory(name: str) -> ModelFactory:
    def factory(component: Any) -> Callable[..., Any]:
        from . import fit_config

        return getattr(fit_config, name)(component)

    return factory


def _fit_jacobian_factory(name: str) -> ModelJacobianFactory:
    def factory(component: Any) -> Callable[..., Any] | None:
        from . import fit_config

        return getattr(fit_config, name)(component)

    return factory


def _fit_dynamic_parameters(name: str) -> DynamicParameters:
    def parameters(component: Any) -> tuple[str, ...]:
        from . import fit_config

        return tuple(getattr(fit_config, name)(component))

    return parameters


def _fit_diagnostics(name: str) -> ModelDiagnostics:
    def diagnostics(
        component: Any, data: Any, params: Mapping[str, float]
    ) -> Mapping[str, Any] | None:
        from . import fit_config

        return getattr(fit_config, name)(component, data, params)

    return diagnostics


def _fit_validator(name: str) -> ModelValidator:
    def validate(component: Any) -> None:
        from . import fit_config

        getattr(fit_config, name)(component)

    return validate


def _report_sections(name: str) -> ModelReportSections:
    def sections(
        fit_entry: Any,
        model: Mapping[str, Any],
        goodness: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Sequence[tuple[str, str]]:
        from . import report

        return getattr(report, name)(fit_entry, model, goodness, context)

    return sections


def _model_plot_calculator(name: str) -> Callable[..., Any]:
    def calculate(*args: Any, **kwargs: Any) -> Any:
        from . import model_plots

        return getattr(model_plots, name)(*args, **kwargs)

    return calculate


def _model_plot_renderer(name: str) -> Callable[..., Any]:
    def render(*args: Any, **kwargs: Any) -> Any:
        from . import model_plots

        return getattr(model_plots, name)(*args, **kwargs)

    return render


def _model_plot_script(name: str) -> Callable[..., str]:
    def script(*args: Any, **kwargs: Any) -> str:
        from . import model_plots

        return str(getattr(model_plots, name)(*args, **kwargs))

    return script


def _tight_binding_plot_script(plot_key: str) -> Callable[..., str]:
    def script(component: Any) -> str:
        from .model_plots import tight_binding_plot_script

        return tight_binding_plot_script(component, plot_key)

    return script


def _parameter(
    name: str,
    default: Any,
    description: str,
    allowed: str,
    unit: str,
    example: str,
) -> ModelParameterDefinition:
    return ModelParameterDefinition(
        name=name,
        default=default,
        description=description,
        allowed=allowed,
        type="float",
        unit=unit,
        example=example,
    )


def _config_field(
    name: str,
    default: Any,
    description: str,
    allowed: str,
    type_name: str,
    example: str,
    unit: str = "",
) -> ModelConfigDefinition:
    return ModelConfigDefinition(
        name=name,
        default=default,
        description=description,
        allowed=allowed,
        type=type_name,
        example=example,
        unit=unit,
    )


def _form_factor_fields(
    example_ion: str, *, mention_lattice_requirement: bool = False
) -> tuple[ModelConfigDefinition, ...]:
    lattice_note = (
        " Requires lattice metadata for |Q|." if mention_lattice_requirement else ""
    )
    return (
        ModelConfigDefinition(
            name="ion",
            default="",
            description=(
                "Magnetic form factor multiplying the intensity: choose a tabulated ion, "
                "Custom for explicit <j0> coefficients, or none for no form factor."
                f"{lattice_note}"
            ),
            allowed="An ion label from the ILL <j0> tables, Custom, or empty.",
            type="str",
            example=example_ion,
            choices="form_factor_ions",
        ),
        ModelConfigDefinition(
            name="form_factor_coefficients",
            default="",
            description=(
                "Custom <j0> coefficients A, a, B, b, C, c, D overriding the ion "
                "table (see https://www.ill.eu/sites/ccsl/ffacts/)."
            ),
            allowed="Seven comma-separated numbers, or empty to use the ion table.",
            type="str",
            example="0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119",
        ),
    )


def _bulk_response_fields() -> tuple[ModelConfigDefinition, ...]:
    """Shared normalization fields for scalar uniform-response models."""

    return (
        ModelConfigDefinition(
            name="bulk_g_factor",
            default=2.0,
            description=(
                "Lande g factor used to convert the uniform spin susceptibility "
                "to bulk susceptibility or linear-response moment."
            ),
            allowed="Positive finite number.",
            type="float",
            example="2.0",
            unit="dimensionless",
        ),
        ModelConfigDefinition(
            name="magnetic_ions_per_formula_unit",
            default=1.0,
            description=(
                "Number of equivalent magnetic ions represented by the response "
                "per formula unit for molar bulk normalization."
            ),
            allowed="Positive finite number.",
            type="float",
            example="2.0",
            unit="ions/f.u.",
        ),
    )


def _register_builtin_models() -> None:
    register_model_definition(
        ModelDefinition(
            key="constant_background",
            label="Constant background",
            description="Measured-intensity background that is constant across Q and energy.",
            category="background",
            data_types=("*",),
            factory=_fit_factory("_constant_background_factory"),
            jacobian_factory=_fit_jacobian_factory(
                "_constant_background_jacobian_factory"
            ),
            parameter_fields=(
                _parameter(
                    "constant",
                    0.0,
                    "Flat measured-intensity offset added to every point.",
                    "Any finite number in the dataset intensity units.",
                    "dataset intensity",
                    "0.1",
                ),
            ),
            documentation="modeling_pipeline.md#compound-models",
        )
    )
    register_model_definition(
        ModelDefinition(
            key="linear_background",
            label="Linear background",
            description="Measured-intensity background linear in energy transfer.",
            category="background",
            data_types=("single_crystal_inelastic", "powder_inelastic"),
            factory=_fit_factory("_linear_background_factory"),
            jacobian_factory=_fit_jacobian_factory(
                "_linear_background_jacobian_factory"
            ),
            parameter_fields=(
                _parameter(
                    "c0",
                    0.0,
                    "Energy-independent background offset.",
                    "Any finite number in the dataset intensity units.",
                    "dataset intensity",
                    "0.1",
                ),
                _parameter(
                    "c1",
                    0.0,
                    "Slope of the background versus energy transfer.",
                    "Any finite number in intensity units per meV.",
                    "dataset intensity/meV",
                    "0.02",
                ),
            ),
            documentation="modeling_pipeline.md#compound-models",
        )
    )
    register_model_definition(
        ModelDefinition(
            key="local_relaxational",
            label="Local relaxational spin",
            description=(
                "Fully local spin relaxing at rate Gamma: chi'' = chi_loc * Gamma * "
                "E / (E^2 + Gamma^2), converted to intensity with the Bose factor, "
                "magnetic form factor, and one-component isotropic polarization "
                "factor P = 2. Requires a dataset temperature."
            ),
            category="spin_fluctuation",
            data_types=(
                "single_crystal_inelastic",
                "powder_inelastic",
                "single_crystal_elastic",
                "powder_elastic",
                "magnetization",
            ),
            factory=_fit_factory("_local_relaxational_factory"),
            parameter_fields=(
                _parameter(
                    "chi_loc",
                    1.0,
                    "Static local susceptibility (1/meV up to the intensity normalization).",
                    "Positive finite number.",
                    "meV^-1",
                    "2.0",
                ),
                _parameter(
                    "gamma",
                    2.0,
                    "Relaxation rate Gamma: chi'' peaks at E = Gamma.",
                    "Positive finite number in meV.",
                    "meV",
                    "3.0",
                ),
            ),
            config_fields=_form_factor_fields(
                "Fe2",
                mention_lattice_requirement=True,
            )
            + _bulk_response_fields(),
            validate_component=_fit_validator("_validate_scalar_bulk_config"),
            default_lower_bounds=(("chi_loc", 0.0), ("gamma", 0.0)),
            documentation="local_relaxational.md",
        )
    )
    register_model_definition(
        ModelDefinition(
            key="mmp_relaxational",
            label="MMP relaxational (nearly AFM)",
            description=(
                "Millis-Monien-Pines susceptibility of a nearly antiferromagnetic "
                "metal: chi(q,w) = chi_pk / (1 + xi^2 |q-Q0|^2 - i "
                "w/omega_sf), converted to intensity with Bose, form-factor, and "
                "one-component isotropic polarization P = 2. Requires dataset "
                "temperature and lattice metadata."
            ),
            category="spin_fluctuation",
            data_types=(
                "single_crystal_inelastic",
                "single_crystal_elastic",
                "magnetization",
            ),
            factory=_fit_factory("_mmp_relaxational_factory"),
            parameter_fields=(
                _parameter(
                    "chi_pk",
                    1.0,
                    "Static susceptibility at the ordering vector Q0 "
                    "(1/meV up to normalization).",
                    "Positive finite number.",
                    "meV^-1",
                    "3.0",
                ),
                _parameter(
                    "xi",
                    1.0,
                    "Magnetic correlation length.",
                    "Positive finite number in Angstrom.",
                    "angstrom",
                    "2.5",
                ),
                _parameter(
                    "omega_sf",
                    1.0,
                    "Spin-fluctuation energy: relaxation rate of the mode at Q0.",
                    "Positive finite number in meV.",
                    "meV",
                    "1.8",
                ),
                _parameter(
                    "q0_h",
                    0.5,
                    "H coordinate of the ordering vector Q0.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.5",
                ),
                _parameter(
                    "q0_k",
                    0.0,
                    "K coordinate of the ordering vector Q0.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.5",
                ),
                _parameter(
                    "q0_l",
                    0.0,
                    "L coordinate of the ordering vector Q0.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.0",
                ),
            ),
            config_fields=_form_factor_fields("Fe2") + _bulk_response_fields(),
            validate_component=_fit_validator("_validate_scalar_bulk_config"),
            default_lower_bounds=(
                ("chi_pk", 0.0),
                ("xi", 0.0),
                ("omega_sf", 0.0),
            ),
            documentation="mmp_relaxational.md",
            citations=(
                "https://doi.org/10.1103/PhysRevB.42.167",
                "https://doi.org/10.1103/PhysRevB.47.6069",
            ),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="generalized_paramagnon",
            label="Generalized paramagnon / damped mode",
            description=(
                "Causal anisotropic paramagnon response with a positive correlation "
                "metric, configurable momentum-space centers, critical slowing down, "
                "and an optional inertial term. It continuously covers local and MMP "
                "relaxation and damped propagating modes."
            ),
            category="spin_fluctuation",
            data_types=(
                "single_crystal_inelastic",
                "powder_inelastic",
                "single_crystal_elastic",
                "powder_elastic",
                "magnetization",
            ),
            factory=_fit_factory("_generalized_paramagnon_factory"),
            parameter_fields=(
                _parameter(
                    "chi_peak",
                    1.0,
                    "Static susceptibility of one peak at its center.",
                    "Nonnegative finite number.",
                    "meV^-1",
                    "3.0",
                ),
                _parameter(
                    "gamma0",
                    2.0,
                    "Relaxation energy at a peak center.",
                    "Positive finite number in meV.",
                    "meV",
                    "2.0",
                ),
                _parameter(
                    "relaxation_power",
                    1.0,
                    "Exponent z in Gamma(q) = gamma0 A(q)^z.",
                    "Nonnegative finite number.",
                    "dimensionless",
                    "1.0",
                ),
                _parameter(
                    "inverse_mode_energy_sq",
                    0.0,
                    "Inertial coefficient 1/E0^2; zero is exactly relaxational.",
                    "Nonnegative finite number in meV^-2.",
                    "meV^-2",
                    "0.04",
                ),
                _parameter(
                    "xi_x",
                    1.0,
                    "x diagonal of the correlation-metric Cholesky factor.",
                    "Nonnegative finite number in Angstrom.",
                    "angstrom",
                    "4.0",
                ),
                _parameter(
                    "xi_y",
                    1.0,
                    "y diagonal of the correlation-metric Cholesky factor.",
                    "Nonnegative finite number in Angstrom.",
                    "angstrom",
                    "2.0",
                ),
                _parameter(
                    "xi_z",
                    1.0,
                    "z diagonal of the correlation-metric Cholesky factor.",
                    "Nonnegative finite number in Angstrom.",
                    "angstrom",
                    "1.0",
                ),
                _parameter(
                    "xi_yx",
                    0.0,
                    "yx off-diagonal of the correlation-metric Cholesky factor.",
                    "Any finite number in Angstrom.",
                    "angstrom",
                    "0.0",
                ),
                _parameter(
                    "xi_zx",
                    0.0,
                    "zx off-diagonal of the correlation-metric Cholesky factor.",
                    "Any finite number in Angstrom.",
                    "angstrom",
                    "0.0",
                ),
                _parameter(
                    "xi_zy",
                    0.0,
                    "zy off-diagonal of the correlation-metric Cholesky factor.",
                    "Any finite number in Angstrom.",
                    "angstrom",
                    "0.0",
                ),
                _parameter(
                    "q0_h",
                    0.5,
                    "H coordinate of the primary peak center.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.5",
                ),
                _parameter(
                    "q0_k",
                    0.0,
                    "K coordinate of the primary peak center.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.0",
                ),
                _parameter(
                    "q0_l",
                    0.0,
                    "L coordinate of the primary peak center.",
                    "Finite number in reciprocal lattice units.",
                    "rlu",
                    "0.0",
                ),
            ),
            config_fields=(
                ModelConfigDefinition(
                    name="spatial_power",
                    default=2.0,
                    description=(
                        "Power p in A(q) = 1 + [q^T C q]^(p/2); p = 2 is "
                        "Ornstein-Zernike."
                    ),
                    allowed="Positive finite number.",
                    type="float",
                    unit="dimensionless",
                    example="2.0",
                ),
                ModelConfigDefinition(
                    name="center_offsets",
                    default=[[0.0, 0.0, 0.0]],
                    description=(
                        "Peak-center offsets added to fitted Q0. Each row is "
                        "[dH, dK, dL] in reciprocal lattice units."
                    ),
                    allowed="Nonempty JSON list of unique three-number rows.",
                    type="list",
                    unit="rlu",
                    example="[[0, 0, 0], [0.5, 0.5, 0]]",
                ),
                ModelConfigDefinition(
                    name="center_combination",
                    default="sum",
                    description=(
                        "Sum responses from all listed centers or retain the nearest "
                        "center at each momentum."
                    ),
                    allowed="'sum' or 'nearest'.",
                    type="str",
                    unit="",
                    example="sum",
                ),
                ModelConfigDefinition(
                    name="periodic",
                    default=True,
                    description=(
                        "When true, use the closest reciprocal-lattice image of each "
                        "center."
                    ),
                    allowed="true or false.",
                    type="bool",
                    unit="",
                    example="true",
                ),
                ModelConfigDefinition(
                    name="powder_orientations",
                    default=50,
                    description="Number of deterministic sphere directions in a powder average.",
                    allowed="Integer of at least 6.",
                    type="int",
                    unit="directions",
                    example="96",
                ),
                ModelConfigDefinition(
                    name="lattice",
                    default={},
                    description=(
                        "Optional lattice dictionary with a, b, c in Angstrom and "
                        "alpha, beta, gamma in degrees. Dataset reciprocal metadata "
                        "takes precedence."
                    ),
                    allowed="JSON dictionary or empty dictionary.",
                    type="dict",
                    unit="",
                    example=(
                        '{"a": 4, "b": 4, "c": 6, "alpha": 90, '
                        '"beta": 90, "gamma": 90}'
                    ),
                ),
                *_form_factor_fields("Fe2"),
                *_bulk_response_fields(),
            ),
            validate_component=_fit_validator(
                "_validate_generalized_paramagnon_component"
            ),
            diagnostics=_fit_diagnostics(
                "_generalized_paramagnon_component_diagnostics"
            ),
            report_sections=_report_sections(
                "generalized_paramagnon_report_sections"
            ),
            plots=(
                ModelPlotDefinition(
                    key="complex_energy_scan",
                    label="Complex susceptibility versus energy",
                    description=(
                        "Plot chi' and chi'' at one or more dimensionless spatial "
                        "kernel values A(q)."
                    ),
                    calculate=_model_plot_calculator(
                        "generalized_paramagnon_energy_scan"
                    ),
                    render=_model_plot_renderer(
                        "render_generalized_paramagnon_energy_scan"
                    ),
                    script=_model_plot_script(
                        "generalized_paramagnon_energy_scan_script"
                    ),
                ),
            ),
            default_lower_bounds=(
                ("chi_peak", 0.0),
                ("gamma0", 0.0),
                ("relaxation_power", 0.0),
                ("inverse_mode_energy_sq", 0.0),
                ("xi_x", 0.0),
                ("xi_y", 0.0),
                ("xi_z", 0.0),
            ),
            documentation="generalized_paramagnon.md",
            citations=(
                "https://doi.org/10.1007/978-3-642-82499-9",
                "https://doi.org/10.1103/PhysRevB.42.167",
            ),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="heisenberg_rpa",
            label="Heisenberg RPA spin fluctuations",
            description=(
                "Local relaxational spins coupled by Heisenberg exchange in the "
                "RPA: chi(Q,w) = [1 - chi0(w) J(Q)]^-1 chi0(w) with chi0(w) = "
                "chi0/(1 - i w/Gamma0). J(Q) is built from symmetry-distinct bond "
                "orbits (J1, J2, J3a, ...), each an exchange fit parameter in meV; "
                "J > 0 favors ordering where J(Q) is maximal and the fit is "
                "restricted to the paramagnetic side max J(Q) chi0 < 1. Configure "
                "the crystal and generate bond orbits, then fit. Requires dataset "
                "temperature."
            ),
            category="spin_fluctuation",
            data_types=(
                "single_crystal_inelastic",
                "powder_inelastic",
                "single_crystal_elastic",
                "powder_elastic",
                "magnetization",
            ),
            factory=_fit_factory("_heisenberg_rpa_factory"),
            parameter_fields=(
                _parameter(
                    "chi0",
                    0.1,
                    "Single-site static susceptibility entering the RPA denominator "
                    "(1/meV: the product J * chi0 is dimensionless). The magnetic "
                    "instability is at max J(Q) chi0 = 1.",
                    "Positive finite number in 1/meV.",
                    "meV^-1",
                    "0.5",
                ),
                _parameter(
                    "gamma0",
                    5.0,
                    "Bare single-site relaxation rate; the coupled mode at Q relaxes "
                    "at Gamma0 (1 - J(Q) chi0), softening toward the ordering vector.",
                    "Positive finite number in meV.",
                    "meV",
                    "5.0",
                ),
            ),
            config_fields=_form_factor_fields("Yb3"),
            dynamic_parameters=_fit_dynamic_parameters(
                "heisenberg_rpa_parameter_labels"
            ),
            dynamic_parameter_description=(
                "Heisenberg exchange constant of symmetry orbit {name} in meV; one "
                "shared value for every bond in the orbit. Positive J favors "
                "ordering at the wavevector maximizing J(Q)."
            ),
            dynamic_parameter_unit="meV",
            jacobian_factory=_fit_jacobian_factory(
                "_heisenberg_rpa_jacobian_factory"
            ),
            diagnostics=_fit_diagnostics(
                "_heisenberg_rpa_component_diagnostics"
            ),
            report_sections=_report_sections("heisenberg_rpa_report_sections"),
            default_lower_bounds=(("chi0", 0.0), ("gamma0", 0.0)),
            structured_config=True,
            documentation="heisenberg_rpa.md",
            citations=("https://doi.org/10.1007/978-3-642-82499-9",),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="debye_heat_capacity",
            label="Debye phonon heat capacity",
            description=(
                "Debye lattice heat capacity C_D(T). The oscillator count is the "
                "number of atoms represented per formula unit, so the "
                "high-temperature limit is 3 n R."
            ),
            category="heat_capacity",
            data_types=("heat_capacity",),
            factory=_fit_factory("_debye_heat_capacity_factory"),
            parameter_fields=(
                _parameter(
                    "debye_temperature",
                    300.0,
                    "Debye temperature theta_D.",
                    "Positive finite temperature in kelvin.",
                    "K",
                    "420.0",
                ),
                _parameter(
                    "oscillator_count",
                    1.0,
                    "Number of atoms represented per formula unit.",
                    "Positive finite dimensionless number.",
                    "dimensionless",
                    "7.0",
                ),
            ),
            default_lower_bounds=(
                ("debye_temperature", 0.0),
                ("oscillator_count", 0.0),
            ),
            documentation="debye_heat_capacity.md",
            citations=("https://doi.org/10.1002/andp.19123441404",),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="low_temperature_heat_capacity",
            label="Low-temperature heat capacity",
            description="Electronic plus leading Debye term: C = gamma T + beta T^3.",
            category="heat_capacity",
            data_types=("heat_capacity",),
            factory=_fit_factory("_low_temperature_heat_capacity_factory"),
            parameter_fields=(
                _parameter(
                    "sommerfeld_gamma",
                    100.0,
                    "Sommerfeld coefficient gamma.",
                    "Non-negative finite number in mJ/(mol K^2).",
                    "mJ/(mol K^2)",
                    "400.0",
                ),
                _parameter(
                    "debye_beta",
                    0.1,
                    "Leading Debye coefficient beta multiplying T^3.",
                    "Non-negative finite number in mJ/(mol K^4).",
                    "mJ/(mol K^4)",
                    "0.08",
                ),
            ),
            default_lower_bounds=(
                ("sommerfeld_gamma", 0.0),
                ("debye_beta", 0.0),
            ),
            documentation="low_temperature_heat_capacity.md",
            citations=(
                "https://doi.org/10.1007/BF01391052",
                "https://doi.org/10.1002/andp.19123441404",
            ),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="curie_weiss",
            label="Curie-Weiss susceptibility",
            description="Absolute molar susceptibility chi = C / (T - theta_CW).",
            category="magnetization",
            data_types=("magnetization",),
            factory=_fit_factory("_curie_weiss_factory"),
            parameter_fields=(
                _parameter(
                    "curie_constant",
                    1.0,
                    "Molar Curie constant C in cm^3 K/mol.",
                    "Positive finite number.",
                    "cm^3 K/mol",
                    "0.42",
                ),
                _parameter(
                    "theta_CW",
                    0.0,
                    "Curie-Weiss temperature theta_CW in kelvin.",
                    "Finite number below the fitted temperature interval.",
                    "K",
                    "-18.0",
                ),
            ),
            default_lower_bounds=(("curie_constant", 0.0),),
            documentation="curie_weiss.md",
            citations=(
                "https://doi.org/10.1051/jphystap:019070060066100",
            ),
        )
    )
    register_model_definition(
        ModelDefinition(
            key="tight_binding",
            label="Tight-binding electronic structure",
            description=(
                "Material-independent orthonormal electronic Hamiltonian for "
                "manual or Wannier90 models, with bands, orbital projections, "
                "density of states, and Fermi surfaces."
            ),
            category="electronic_structure",
            data_types=("electronic_structure",),
            factory=_fit_factory("_electronic_structure_factory"),
            validate_component=_fit_validator("_validate_tight_binding_config"),
            dynamic_parameters=_fit_dynamic_parameters(
                "tight_binding_parameter_names"
            ),
            dynamic_parameter_description=(
                "Named static onsite or hopping coefficient {name}. Values are "
                "canonical meV and multiply the symmetry-generated Hamiltonian basis."
            ),
            dynamic_parameter_unit="meV",
            config_fields=(
                _config_field(
                    "source_path",
                    "",
                    "Wannier90 *_hr.dat or *_tb.dat source; empty for a manual model.",
                    "Existing Wannier90 Hamiltonian path, or empty.",
                    "str",
                    "/path/to/material_tb.dat",
                ),
                _config_field(
                    "model_digest",
                    "",
                    "SHA-256 digest of the canonical electronic model.",
                    "A 64-character hexadecimal digest, or empty.",
                    "str",
                    "",
                ),
                _config_field(
                    "model_data",
                    {},
                    "Portable ElectronicModel.to_dict() data for a manual model.",
                    "Canonical model dictionary, or empty.",
                    "dict",
                    "{}",
                ),
                _config_field(
                    "model_stale",
                    False,
                    (
                        "Whether compact builder edits have invalidated the "
                        "cached canonical Hamiltonian."
                    ),
                    "Boolean derived-cache state; calculations resolve it lazily.",
                    "bool",
                    "false",
                ),
                _config_field(
                    "use_primitive_cell",
                    True,
                    (
                        "Fold GUI-built conventional-cell Hamiltonians onto "
                        "the primitive translation lattice before evaluation."
                    ),
                    "Boolean; disable only for diagnostic comparison.",
                    "bool",
                    "true",
                ),
                _config_field(
                    "hopping_parameterization",
                    "slater_koster",
                    (
                        "Basis used for GUI-generated hopping coefficients: "
                        "compact two-centre Slater-Koster channels or the full "
                        "space-group-allowed matrix basis."
                    ),
                    "Either slater_koster or general.",
                    "str",
                    "slater_koster",
                ),
                _config_field(
                    "crystal",
                    {
                        "lattice": {
                            "a": 5.0,
                            "b": 5.0,
                            "c": 5.0,
                            "alpha": 90.0,
                            "beta": 90.0,
                            "gamma": 90.0,
                        },
                        "spacegroup": "P 1",
                        "sites": [],
                    },
                    "Editable crystal geometry for CIF-based or manual construction.",
                    "Crystal dictionary with lattice, space group, and site records.",
                    "dict",
                    '{"lattice": {"a": 4, "b": 4, "c": 6, '
                    '"alpha": 90, "beta": 90, "gamma": 90}, '
                    '"spacegroup": "P 1", "sites": []}',
                ),
                _config_field(
                    "orbital_manifolds",
                    [],
                    (
                        "Editable site-attached orbital manifolds, local frames, "
                        "and symmetry conventions used by the structure-first builder."
                    ),
                    "A list of OrbitalManifold dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "spin_treatment",
                    "auto",
                    (
                        "Requested spin representation. Auto keeps spin "
                        "implicit unless active SOC requires a full spinor basis."
                    ),
                    "One of auto, implicit, collinear, or spinor.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "soc_terms",
                    [],
                    (
                        "Optional manifold-resolved onsite lambda L.S terms "
                        "with canonical meV coefficients."
                    ),
                    "A list of SpinOrbitTerm dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "onsite_terms",
                    [],
                    (
                        "Generated symmetry-allowed onsite energies and "
                        "hybridizations with canonical meV coefficients."
                    ),
                    "A list of OnsiteInvariant dictionaries generated from the crystal and orbitals.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_cutoff_angstrom",
                    0.0,
                    (
                        "Maximum real-space bond distance used by the "
                        "symmetry-aware hopping generator."
                    ),
                    "Zero before generation, otherwise a positive length in Angstrom.",
                    "float",
                    "4.5",
                    "angstrom",
                ),
                _config_field(
                    "spatial_orbits",
                    [],
                    (
                        "Generated symmetry-distinct hopping pathways and "
                        "their space-group mapping operations."
                    ),
                    "A generated list of BondOrbit dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_candidates",
                    [],
                    (
                        "Generated symmetry-allowed hopping suggestions; only "
                        "selected candidates enter the Hamiltonian."
                    ),
                    "A generated list of HoppingInvariant dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_terms",
                    [],
                    (
                        "Generated symmetry-covariant hopping matrices with "
                        "canonical meV coefficients."
                    ),
                    "A generated list of HoppingInvariant dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "periodic_axes",
                    [],
                    "Direct-lattice periodic axes; empty infers them from hoppings.",
                    "Empty, or one to three unique indices chosen from 0, 1, and 2.",
                    "list",
                    "[0, 1]",
                ),
                _config_field(
                    "electronic_energy_unit",
                    "eV",
                    (
                        "Input and plot unit for electronic-structure energies; "
                        "canonical model data remain in meV."
                    ),
                    "Either eV or meV.",
                    "str",
                    "eV",
                ),
                _config_field(
                    "chemical_potential_meV",
                    0.0,
                    (
                        "Canonical chemical potential used as the plotted energy "
                        "zero; the GUI converts it to the selected electronic unit."
                    ),
                    "Any finite energy stored in meV.",
                    "float",
                    "12.5",
                    "meV",
                ),
                _config_field(
                    "projection_groups",
                    {},
                    "Named orbital projections as zero-based basis-index lists.",
                    "JSON dictionary mapping labels to index lists.",
                    "dict",
                    '{"d": [0, 1, 2], "p": [3, 4]}',
                ),
                _config_field(
                    "band_path",
                    [
                        {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
                        {"label": "X", "k": [0.5, 0.0, 0.0]},
                        {"label": "M", "k": [0.5, 0.5, 0.0]},
                        {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
                    ],
                    (
                        "Ordered labeled nodes of the band path in the primitive "
                        "reciprocal basis."
                    ),
                    (
                        "JSON list of labeled three-coordinate nodes. A coordinate "
                        "of 0.5 reaches halfway to the neighboring reciprocal point "
                        "along that primitive basis vector."
                    ),
                    "list",
                    '[{"label": "G", "k": [0, 0, 0]}, '
                    '{"label": "X", "k": [0.5, 0, 0]}]',
                ),
                _config_field(
                    "band_path_convention",
                    "hinuma",
                    (
                        "Source convention for the configured high-symmetry "
                        "path."
                    ),
                    "Either manual or hinuma.",
                    "str",
                    "hinuma",
                ),
                _config_field(
                    "band_path_metadata",
                    {},
                    (
                        "Provider, version, convention, and tolerance used to "
                        "generate an automatic high-symmetry path."
                    ),
                    "JSON dictionary; empty for a manual path.",
                    "dict",
                    (
                        '{"provider": "seekpath", "provider_version": "2.2.1", '
                        '"convention": "HPKOT", "symprec": 1e-5}'
                    ),
                ),
                _config_field(
                    "band_points_per_segment",
                    60,
                    "Interpolation intervals in each band-path segment.",
                    "Positive integer.",
                    "int",
                    "80",
                ),
                _config_field(
                    "dos_mesh",
                    [40, 40, 40],
                    "Uniform Brillouin-zone mesh for density of states.",
                    "One size per periodic dimension, or three lattice-axis sizes.",
                    "list",
                    "[80, 80, 1]",
                ),
                _config_field(
                    "dos_energy_min_meV",
                    -500.0,
                    "Canonical lower density-of-states energy.",
                    "Finite energy below dos_energy_max_meV.",
                    "float",
                    "-250",
                    "meV",
                ),
                _config_field(
                    "dos_energy_max_meV",
                    500.0,
                    "Canonical upper density-of-states energy.",
                    "Finite energy above dos_energy_min_meV.",
                    "float",
                    "250",
                    "meV",
                ),
                _config_field(
                    "dos_energy_points",
                    600,
                    "Number of points in the density-of-states energy grid.",
                    "Integer of at least 2.",
                    "int",
                    "1000",
                ),
                _config_field(
                    "dos_broadening_meV",
                    5.0,
                    "Canonical Gaussian standard deviation for density of states.",
                    "Positive finite energy.",
                    "float",
                    "2.0",
                    "meV",
                ),
                _config_field(
                    "fermi_mesh",
                    [100, 100, 40],
                    "Periodic grid used to extract the Fermi surface.",
                    "One size per periodic dimension, or three lattice-axis sizes.",
                    "list",
                    "[200, 200, 1]",
                ),
                _config_field(
                    "fermi_energy_meV",
                    0.0,
                    "Canonical target energy for Fermi-surface extraction.",
                    "Any finite energy.",
                    "float",
                    "0.0",
                    "meV",
                ),
            ),
            plots=(
                ModelPlotDefinition(
                    key="bands",
                    label="Band structure",
                    description=(
                        "Plot bands and configured orbital projections in the "
                        "selected electronic energy unit (eV by default)."
                    ),
                    calculate=_model_plot_calculator("tight_binding_band_structure"),
                    render=_model_plot_renderer("render_band_structure"),
                    script=_tight_binding_plot_script("bands"),
                ),
                ModelPlotDefinition(
                    key="dos",
                    label="Density of states",
                    description=(
                        "Plot total and orbital-projected DOS with both axes "
                        "converted to the selected electronic energy unit."
                    ),
                    calculate=_model_plot_calculator(
                        "tight_binding_density_of_states"
                    ),
                    render=_model_plot_renderer("render_density_of_states"),
                    script=_tight_binding_plot_script("dos"),
                ),
                ModelPlotDefinition(
                    key="fermi_surface",
                    label="Fermi surface",
                    description=(
                        "Plot 1D Fermi points, 2D contours, or a 3D surface and "
                        "label its target in the selected electronic energy unit."
                    ),
                    calculate=_model_plot_calculator("tight_binding_fermi_surface"),
                    render=_model_plot_renderer("render_fermi_surface"),
                    script=_tight_binding_plot_script("fermi_surface"),
                ),
            ),
            report_sections=_report_sections("tight_binding_report_sections"),
            documentation="tight_binding.md",
            citations=(
                "https://doi.org/10.1016/j.cpc.2007.11.016",
                "https://doi.org/10.1088/1361-648X/ab51ff",
            ),
            metadata={"component_plot_actions": True},
        )
    )


_register_builtin_models()
