"""Low-temperature ``C/T`` versus ``T^2`` heat-capacity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import PointListData
from ..heat_capacity import debye_temperature_from_beta
from .core import AnalysisExecution, AnalysisInput, DatasetOutput, ScalarOutput


def validate_low_temperature_heat_capacity(inputs, parameters) -> None:
    if not isinstance(inputs[0].data, PointListData):
        raise TypeError("low-temperature heat-capacity fitting requires a point-list dataset")
    minimum = float(parameters["temperature_min_K"])
    maximum = float(parameters["temperature_max_K"])
    atoms = float(parameters["atoms_per_formula_unit"])
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("heat-capacity Tmax must be greater than Tmin")
    if not np.isfinite(atoms) or atoms <= 0.0:
        raise ValueError("atoms per formula unit must be positive")
    _temperature_column(inputs[0].data)
    _heat_capacity_channel(inputs[0].data)


def execute_low_temperature_heat_capacity(
    inputs: list[AnalysisInput], parameters: dict[str, Any], **_callbacks: Any
) -> AnalysisExecution:
    """Fit ``C/T = gamma + beta*T^2`` over a selected temperature range."""

    data = inputs[0].data
    if not isinstance(data, PointListData):
        raise TypeError("low-temperature heat-capacity fitting requires a point-list dataset")
    temperature_name = _temperature_column(data)
    channel = _heat_capacity_channel(data)
    temperature = np.asarray(data.column(temperature_name), dtype=float)
    values = np.asarray(data.channel_values(channel["label"]), dtype=float)
    errors = data.channel_errors(channel["label"])
    errors = None if errors is None else np.asarray(errors, dtype=float)
    quantity = data.channel_quantity_type(channel["label"])
    unit = data.unit(channel["value"])
    if quantity == "heat_capacity":
        if unit != "mJ/(mol K)":
            raise ValueError("low-temperature analysis requires C in mJ/(mol K)")
        with np.errstate(divide="ignore", invalid="ignore"):
            c_over_t = values / temperature
            c_over_t_error = None if errors is None else errors / np.abs(temperature)
    else:
        if unit != "mJ/(mol K^2)":
            raise ValueError("low-temperature analysis requires C/T in mJ/(mol K^2)")
        c_over_t = values
        c_over_t_error = errors

    minimum = float(parameters["temperature_min_K"])
    maximum = float(parameters["temperature_max_K"])
    x = temperature**2
    mask = (
        np.isfinite(temperature)
        & np.isfinite(x)
        & np.isfinite(c_over_t)
        & (temperature >= minimum)
        & (temperature <= maximum)
    )
    if c_over_t_error is not None:
        mask &= np.isfinite(c_over_t_error) & (c_over_t_error > 0.0)
    if np.count_nonzero(mask) < 3:
        raise ValueError("low-temperature heat-capacity fitting requires at least three valid points")

    design = np.column_stack([np.ones(np.count_nonzero(mask)), x[mask]])
    if c_over_t_error is None:
        weighted_design = design
        weighted_values = c_over_t[mask]
    else:
        weights = 1.0 / c_over_t_error[mask]
        weighted_design = design * weights[:, None]
        weighted_values = c_over_t[mask] * weights
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        weighted_design, weighted_values, rcond=None
    )
    if rank < 2:
        raise ValueError("selected heat-capacity range is not sufficient to determine gamma and beta")
    gamma, beta = (float(value) for value in coefficients)
    residual = c_over_t[mask] - design @ coefficients
    dof = int(np.count_nonzero(mask) - 2)
    if c_over_t_error is None:
        variance_scale = float(np.sum(residual**2) / dof) if dof > 0 else float("nan")
        covariance = variance_scale * np.linalg.inv(design.T @ design)
        chi_square = float(np.sum(residual**2))
    else:
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        chi_square = float(np.sum((residual / c_over_t_error[mask]) ** 2))
    uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    fitted = gamma + beta * x
    atoms = float(parameters["atoms_per_formula_unit"])
    theta = debye_temperature_from_beta(beta, atoms)
    theta_uncertainty = (
        float(theta * uncertainties[1] / (3.0 * beta))
        if np.isfinite(theta) and beta > 0.0 else float("nan")
    )
    diagnostic = PointListData(
        columns={
            temperature_name: temperature,
            "Temperature squared": x,
            "C/T": c_over_t,
            "C/T uncertainty": (
                np.full(c_over_t.shape, np.nan) if c_over_t_error is None else c_over_t_error
            ),
            "Low-T fit C/T": fitted,
        },
        units={
            temperature_name: "K",
            "Temperature squared": "K^2",
            "C/T": "mJ/(mol K^2)",
            "C/T uncertainty": "mJ/(mol K^2)",
            "Low-T fit C/T": "mJ/(mol K^2)",
        },
        coordinate_names=["Temperature squared", temperature_name],
        channels=[
            {"label": "C/T", "value": "C/T",
             "error": "C/T uncertainty" if c_over_t_error is not None else None,
             "quantity_type": "heat_capacity_over_temperature", "unit": "mJ/(mol K^2)"},
            {"label": "Low-T fit C/T", "value": "Low-T fit C/T", "error": None,
             "quantity_type": "heat_capacity_over_temperature", "unit": "mJ/(mol K^2)"},
        ],
        metadata={
            "analysis_type": "low_temperature_heat_capacity_fit",
            "viewer_hidden_channels": ["Low-T fit C/T"],
            "viewer_fit_channel_map": {"C/T": "Low-T fit C/T"},
            "viewer_show_fit": True,
            "vertical_reference_lines": [
                {"value": minimum**2, "label": "Tmin²"},
                {"value": maximum**2, "label": "Tmax²"},
            ],
        },
        quantity_types={
            temperature_name: "temperature",
            "C/T": "heat_capacity_over_temperature",
            "C/T uncertainty": "heat_capacity_over_temperature",
            "Low-T fit C/T": "heat_capacity_over_temperature",
        },
    )
    outputs = {
        "sommerfeld_gamma": ScalarOutput(gamma, float(uncertainties[0]), "mJ/(mol K^2)", "Sommerfeld coefficient gamma"),
        "debye_beta": ScalarOutput(beta, float(uncertainties[1]), "mJ/(mol K^4)", "Debye T^3 coefficient beta"),
        "debye_temperature": ScalarOutput(theta, theta_uncertainty, "K", "Debye temperature inferred from beta"),
        "diagnostic": DatasetOutput(diagnostic, "Low-temperature heat-capacity diagnostic", "derived_analysis"),
    }
    return AnalysisExecution(
        outputs,
        diagnostics={
            "fit_points": int(np.count_nonzero(mask)),
            "total_points": int(temperature.size),
            "degrees_of_freedom": dof,
            "chi_square": chi_square,
            "reduced_chi_square": chi_square / dof if dof > 0 else None,
            "parameter_covariance": covariance.tolist(),
        },
    )


def _temperature_column(data: PointListData) -> str:
    for name in data.coordinate_names:
        if data.quantity_type(name) == "temperature" or "temp" in name.lower():
            return name
    for name in data.columns:
        if data.quantity_type(name) == "temperature" or "temp" in name.lower():
            return name
    raise ValueError("heat-capacity analysis requires a temperature column")


def _heat_capacity_channel(data: PointListData) -> dict[str, Any]:
    for label in data.channel_labels:
        if data.channel_quantity_type(label) in {"heat_capacity", "heat_capacity_over_temperature"}:
            return data.channel(label)
    raise ValueError("heat-capacity analysis requires a molar C or C/T channel")
