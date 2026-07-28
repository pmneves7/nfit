from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core import AnalysisExecution, AnalysisInput


@dataclass(frozen=True)
class AnalysisParameterDefinition:
    name: str
    label: str
    type: str
    default: Any
    description: str
    allowed: str
    example: str
    choices: tuple[tuple[str, str], ...] = ()
    required: bool = False


@dataclass(frozen=True)
class AnalysisOperationDefinition:
    key: str
    label: str
    version: int
    description: str
    min_inputs: int
    max_inputs: int | None
    accepted_containers: tuple[str, ...]
    parameters: tuple[AnalysisParameterDefinition, ...]
    validate: Callable[[Sequence[AnalysisInput], Mapping[str, Any]], None]
    execute: Callable[..., AnalysisExecution]


_OPERATIONS: dict[str, AnalysisOperationDefinition] = {}
def register_analysis_operation(
    definition: AnalysisOperationDefinition, *, replace: bool = False
) -> None:
    """Register one operation definition."""

    if definition.key in _OPERATIONS and not replace:
        raise ValueError(f"analysis type {definition.key!r} is already registered")
    parameter_names = [parameter.name for parameter in definition.parameters]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError(f"analysis type {definition.key!r} has duplicate parameters")
    _OPERATIONS[definition.key] = definition


def available_analysis_types() -> tuple[str, ...]:
    return tuple(_OPERATIONS)


def analysis_definition(type_name: str) -> AnalysisOperationDefinition:
    try:
        return _OPERATIONS[type_name]
    except KeyError as exc:
        raise KeyError(f"unknown analysis type {type_name!r}") from exc


def default_analysis_parameters(type_name: str) -> dict[str, Any]:
    return {
        parameter.name: copy.deepcopy(parameter.default)
        for parameter in analysis_definition(type_name).parameters
    }


def analysis_parameter_tooltip(type_name: str, parameter_name: str) -> str:
    for parameter in analysis_definition(type_name).parameters:
        if parameter.name == parameter_name:
            parts = [parameter.description, f"Allowed: {parameter.allowed}"]
            if parameter.example:
                parts.append(f"Example: {parameter.example}")
            return "\n".join(parts)
    raise KeyError(f"unknown parameter {parameter_name!r} for analysis {type_name!r}")


def validate_analysis(
    type_name: str,
    inputs: Sequence[AnalysisInput],
    parameters: Mapping[str, Any],
) -> None:
    definition = analysis_definition(type_name)
    count = len(inputs)
    if count < definition.min_inputs:
        raise ValueError(f"{type_name} requires at least {definition.min_inputs} input(s)")
    if definition.max_inputs is not None and count > definition.max_inputs:
        raise ValueError(f"{type_name} accepts at most {definition.max_inputs} input(s)")
    known = {parameter.name for parameter in definition.parameters}
    unknown = set(parameters) - known
    if unknown:
        raise ValueError(f"unknown {type_name} parameter(s): {', '.join(sorted(unknown))}")
    missing = {
        parameter.name
        for parameter in definition.parameters
        if parameter.required and parameter.name not in parameters
    }
    if missing:
        raise ValueError(f"missing {type_name} parameter(s): {', '.join(sorted(missing))}")
    if definition.accepted_containers:
        for item in inputs:
            if type(item.data).__name__ not in definition.accepted_containers:
                raise TypeError(
                    f"{type_name} does not accept {type(item.data).__name__} inputs"
                )
    definition.validate(inputs, parameters)


def run_analysis_operation(
    type_name: str,
    inputs: Sequence[AnalysisInput],
    parameters: Mapping[str, Any],
    progress_callback: Callable[..., Any] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> AnalysisExecution:
    validate_analysis(type_name, inputs, parameters)
    return analysis_definition(type_name).execute(
        inputs,
        parameters,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
