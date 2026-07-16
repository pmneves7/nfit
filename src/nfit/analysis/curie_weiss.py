"""Curie-Weiss fitting for absolute molar susceptibility data."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from ..dataset import PointListData
from ..quantities import convert_quantity
from .core import AnalysisExecution, AnalysisInput, DatasetOutput, ScalarOutput

_AVOGADRO = 6.02214076e23
_BOLTZMANN_CGS = 1.380649e-16  # erg/K
_MU_B_CGS = 9.2740100783e-21  # erg/G
_MU_EFF_PER_SQRT_C = np.sqrt(3.0 * _BOLTZMANN_CGS / (_AVOGADRO * _MU_B_CGS**2))


def curie_weiss_susceptibility(
    temperature_K: np.ndarray, curie_constant: float, theta_K: float
) -> np.ndarray:
    """Return ``C / (T - theta_CW)`` in CGS molar susceptibility units."""

    temperature = np.asarray(temperature_K, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return curie_constant / (temperature - theta_K)


def validate_curie_weiss(inputs, parameters) -> None:
    if not isinstance(inputs[0].data, PointListData):
        raise TypeError("Curie-Weiss fitting requires a point-list dataset")
    minimum = float(parameters["temperature_min_K"])
    maximum = float(parameters["temperature_max_K"])
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Curie-Weiss Tmax must be greater than Tmin")
    _susceptibility_channel(inputs[0].data)
    _temperature_column(inputs[0].data)


def execute_curie_weiss(
    inputs: list[AnalysisInput], parameters: dict[str, Any], **_callbacks: Any
) -> AnalysisExecution:
    """Fit a two-parameter Curie-Weiss law and return a viewer-ready diagnostic."""

    data = inputs[0].data
    if not isinstance(data, PointListData):
        raise TypeError("Curie-Weiss fitting requires a point-list dataset")
    channel = _susceptibility_channel(data)
    temperature_name = _temperature_column(data)
    temperature = np.asarray(data.column(temperature_name), dtype=float)
    susceptibility = np.asarray(data.channel_values(channel["label"]), dtype=float)
    uncertainty = data.channel_errors(channel["label"])
    uncertainty = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
    source_unit = data.unit(channel["value"])
    if source_unit not in {"cm^3/mol", "m^3/mol"}:
        raise ValueError(
            "Curie-Weiss effective moment requires absolute molar susceptibility "
            "in cm^3/mol or m^3/mol"
        )
    susceptibility_cgs = convert_quantity(
        susceptibility, "bulk_susceptibility", source_unit, "cm^3/mol"
    )
    uncertainty_cgs = (
        None
        if uncertainty is None
        else np.abs(convert_quantity(
            uncertainty, "bulk_susceptibility", source_unit, "cm^3/mol"
        ))
    )

    minimum = float(parameters["temperature_min_K"])
    maximum = float(parameters["temperature_max_K"])
    fit_mask = (
        np.isfinite(temperature)
        & np.isfinite(susceptibility_cgs)
        & (susceptibility_cgs != 0.0)
        & (temperature >= minimum)
        & (temperature <= maximum)
    )
    if uncertainty_cgs is not None:
        fit_mask &= np.isfinite(uncertainty_cgs) & (uncertainty_cgs > 0.0)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("Curie-Weiss fitting requires at least three valid points in the selected temperature range")

    fit_temperature = temperature[fit_mask]
    fit_susceptibility = susceptibility_cgs[fit_mask]
    fit_uncertainty = None if uncertainty_cgs is None else uncertainty_cgs[fit_mask]
    curie_guess, theta_guess = _initial_guess(
        fit_temperature, fit_susceptibility, fit_uncertainty
    )
    gap = max(np.ptp(fit_temperature), 1.0) * 1.0e-8
    upper_theta = float(np.min(fit_temperature) - gap)
    theta_guess = min(theta_guess, upper_theta - gap)
    parameters_fit, covariance = curve_fit(
        curie_weiss_susceptibility,
        fit_temperature,
        fit_susceptibility,
        p0=(curie_guess, theta_guess),
        sigma=fit_uncertainty,
        absolute_sigma=fit_uncertainty is not None,
        bounds=((np.finfo(float).tiny, -np.inf), (np.inf, upper_theta)),
        maxfev=20_000,
    )
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("Curie-Weiss fit covariance is not finite; choose a wider or better-conditioned temperature range")
    curie_constant, theta = (float(value) for value in parameters_fit)
    curie_uncertainty, theta_uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    effective_moment = float(_MU_EFF_PER_SQRT_C * np.sqrt(curie_constant))
    effective_moment_uncertainty = float(
        effective_moment * curie_uncertainty / (2.0 * curie_constant)
    )

    fitted_cgs = curie_weiss_susceptibility(temperature, curie_constant, theta)
    fitted_source = convert_quantity(
        fitted_cgs, "bulk_susceptibility", "cm^3/mol", source_unit
    )
    inverse, inverse_uncertainty = _inverse_with_uncertainty(susceptibility, uncertainty)
    fitted_inverse, _unused = _inverse_with_uncertainty(fitted_source, None)
    inverse_unit = "mol/cm^3" if source_unit == "cm^3/mol" else "mol/m^3"
    diagnostic = PointListData(
        columns={
            temperature_name: temperature,
            "Susceptibility": susceptibility,
            "Susceptibility uncertainty": (
                np.full(temperature.shape, np.nan) if uncertainty is None else uncertainty
            ),
            "Inverse susceptibility": inverse,
            "Inverse susceptibility uncertainty": inverse_uncertainty,
            "CW fit susceptibility": fitted_source,
            "CW fit inverse susceptibility": fitted_inverse,
        },
        units={
            temperature_name: "K",
            "Susceptibility": source_unit,
            "Susceptibility uncertainty": source_unit,
            "Inverse susceptibility": inverse_unit,
            "Inverse susceptibility uncertainty": inverse_unit,
            "CW fit susceptibility": source_unit,
            "CW fit inverse susceptibility": inverse_unit,
        },
        coordinate_names=[temperature_name],
        channels=[
            {
                "label": "Susceptibility",
                "value": "Susceptibility",
                "error": "Susceptibility uncertainty" if uncertainty is not None else None,
                "quantity_type": "bulk_susceptibility",
                "unit": source_unit,
            },
            {
                "label": "Inverse susceptibility",
                "value": "Inverse susceptibility",
                "error": "Inverse susceptibility uncertainty" if uncertainty is not None else None,
                "quantity_type": "inverse_bulk_susceptibility",
                "unit": inverse_unit,
            },
            {
                "label": "CW fit susceptibility",
                "value": "CW fit susceptibility",
                "error": None,
                "quantity_type": "bulk_susceptibility",
                "unit": source_unit,
            },
            {
                "label": "CW fit inverse susceptibility",
                "value": "CW fit inverse susceptibility",
                "error": None,
                "quantity_type": "inverse_bulk_susceptibility",
                "unit": inverse_unit,
            },
        ],
        metadata={
            "analysis_type": "curie_weiss_fit",
            "viewer_hidden_channels": [
                "CW fit susceptibility",
                "CW fit inverse susceptibility",
            ],
            "viewer_fit_channel_map": {
                "Susceptibility": "CW fit susceptibility",
                "Inverse susceptibility": "CW fit inverse susceptibility",
            },
            "viewer_show_fit": True,
            "vertical_reference_lines": [
                {"value": minimum, "label": "Tmin"},
                {"value": maximum, "label": "Tmax"},
            ],
            "fit_temperature_min_K": minimum,
            "fit_temperature_max_K": maximum,
        },
        quantity_types={
            temperature_name: "temperature",
            "Susceptibility": "bulk_susceptibility",
            "Susceptibility uncertainty": "bulk_susceptibility",
            "Inverse susceptibility": "inverse_bulk_susceptibility",
            "Inverse susceptibility uncertainty": "inverse_bulk_susceptibility",
            "CW fit susceptibility": "bulk_susceptibility",
            "CW fit inverse susceptibility": "inverse_bulk_susceptibility",
        },
    )

    residual = fit_susceptibility - curie_weiss_susceptibility(
        fit_temperature, curie_constant, theta
    )
    chi_squared = (
        float(np.sum((residual / fit_uncertainty) ** 2))
        if fit_uncertainty is not None
        else float(np.sum(residual**2))
    )
    dof = int(fit_temperature.size - 2)
    diagnostics = {
        "fit_points": int(fit_temperature.size),
        "total_points": int(temperature.size),
        "degrees_of_freedom": dof,
        "chi_squared": chi_squared,
        "reduced_chi_squared": chi_squared / dof if dof > 0 else None,
        "weighted_fit": fit_uncertainty is not None,
        "parameter_covariance": covariance.tolist(),
        "C_theta_correlation": float(
            covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        ),
    }
    outputs = {
        "curie_constant": ScalarOutput(
            curie_constant,
            float(curie_uncertainty),
            "cm^3 K/mol",
            "Curie constant C",
        ),
        "theta_CW": ScalarOutput(
            theta, float(theta_uncertainty), "K", "Curie-Weiss temperature theta_CW"
        ),
        "effective_moment": ScalarOutput(
            effective_moment,
            effective_moment_uncertainty,
            "mu_B/f.u.",
            "Effective moment mu_eff",
        ),
        "diagnostic": DatasetOutput(
            diagnostic,
            "Curie-Weiss diagnostic",
            "derived_analysis",
            {"diagnostic_kind": "curie_weiss"},
        ),
    }
    return AnalysisExecution(outputs, diagnostics=diagnostics)


def _temperature_column(data: PointListData) -> str:
    for name in data.column_names:
        if data.quantity_type(name) == "temperature" or "temperature" in name.lower():
            return name
    raise ValueError("Curie-Weiss fitting requires a temperature column")


def _susceptibility_channel(data: PointListData) -> dict[str, Any]:
    candidates = [
        channel
        for channel in data.channels
        if channel.get("quantity_type") == "bulk_susceptibility"
        or data.quantity_type(str(channel.get("value", ""))) == "bulk_susceptibility"
    ]
    if not candidates:
        raise ValueError(
            "Curie-Weiss fitting requires a bulk-susceptibility channel; enable "
            "absolute susceptibility for the magnetization dataset first"
        )
    return next(
        (channel for channel in candidates if channel.get("label") == "Susceptibility"),
        candidates[0],
    )


def _initial_guess(
    temperature: np.ndarray,
    susceptibility: np.ndarray,
    uncertainty: np.ndarray | None,
) -> tuple[float, float]:
    inverse = 1.0 / susceptibility
    weights = None
    if uncertainty is not None:
        inverse_uncertainty = uncertainty / susceptibility**2
        weights = 1.0 / inverse_uncertainty
    slope, intercept = np.polyfit(temperature, inverse, 1, w=weights)
    if not np.isfinite(slope) or slope <= 0.0:
        span = max(float(np.ptp(temperature)), 1.0)
        curie_constant = max(float(np.nanmedian(np.abs(susceptibility))) * span, 1.0e-12)
        return curie_constant, float(np.min(temperature) - span)
    return float(1.0 / slope), float(-intercept / slope)


def _inverse_with_uncertainty(
    values: np.ndarray, uncertainty: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values != 0.0)
    inverse = np.full(values.shape, np.nan, dtype=float)
    inverse[valid] = 1.0 / values[valid]
    inverse_uncertainty = np.full(values.shape, np.nan, dtype=float)
    if uncertainty is not None:
        errors = np.asarray(uncertainty, dtype=float)
        inverse_uncertainty[valid] = np.abs(errors[valid]) / values[valid] ** 2
    return inverse, inverse_uncertainty
