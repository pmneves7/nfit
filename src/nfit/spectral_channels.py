"""Paired INS cross-section and dynamical-susceptibility channels."""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis.coordinates import q_modulus_for_spectral
from .cross_section import (
    MILLIBARN_PER_BARN,
    kf_over_ki,
)
from .form_factors import form_factor_sq
from .mdhisto import MDHistoChannel, MDHistoData
from .spectral_metadata import (
    normalization_basis_suffix,
    spectral_unit_with_basis,
    target_spectral_record,
)

SPECTRAL_CHANNEL_CONFIG_KEY = "spectral_channels"
CROSS_SECTION_CHANNEL = "scattering_cross_section"
CHIPP_CHANNEL = "dynamic_susceptibility"
IMPORTED_SIGNAL_CHANNEL = "imported_signal"

def _unit_matches_representation(unit: str, representation: str) -> bool:
    if unit == "arbitrary":
        return True
    if representation == "cross_section":
        return unit.startswith(("mbarn/sr/meV", "barn/sr/meV", "1/meV"))
    return unit.startswith(("mu_B^2/meV", "spin^2/meV"))


def default_spectral_channel_config() -> dict[str, Any]:
    """Return a new, JSON-ready dataset spectral convention."""

    return {
        "enabled": True,
        "source_representation": "cross_section",
        "source_unit": "arbitrary",
        "fit_representation": "cross_section",
        "normalization_basis": "per_formula_unit",
        "normalization_label": "",
        "signal_per_mbarn": 0.0,
        "form_factor_ion": "",
        "custom_form_factor": None,
        "polarization_mode": "isotropic_single_component",
        "polarization_scalar": 2.0,
        "moment_unit": "mu_B_squared",
        "g_factor": 2.0,
        "kf_ki_state": "removed",
        "incident_energy_meV": None,
        "final_energy_meV": None,
    }


def normalized_spectral_channel_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Fill defaults and validate a saved spectral-channel configuration."""

    result = default_spectral_channel_config()
    if isinstance(config, dict):
        result.update(config)
    if result["source_representation"] not in {"cross_section", "chi_double_prime"}:
        raise ValueError("source representation must be cross section or dynamic susceptibility")
    if result["fit_representation"] not in {"cross_section", "chi_double_prime"}:
        raise ValueError("fit representation must be cross section or dynamic susceptibility")
    if not _unit_matches_representation(
        str(result["source_unit"]), str(result["source_representation"])
    ):
        result["source_unit"] = "arbitrary"
    if str(result["source_unit"]).startswith("spin^2/"):
        result["moment_unit"] = "spin_squared"
    elif str(result["source_unit"]).startswith("mu_B^2/"):
        result["moment_unit"] = "mu_B_squared"
    if result["normalization_basis"] not in {
        "per_formula_unit",
        "per_magnetic_ion",
        "per_unit_cell",
        "unknown",
    }:
        raise ValueError("unknown INS normalization basis")
    if result["moment_unit"] not in {"mu_B_squared", "spin_squared"}:
        raise ValueError("unknown dynamical-susceptibility moment convention")
    if result["kf_ki_state"] not in {"included", "removed"}:
        raise ValueError("k_f/k_i state must be included or removed")
    return result


def _energy_grid(data: MDHistoData) -> np.ndarray:
    dimensions = [
        index
        for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(dimensions) != 1:
        raise ValueError("INS conversion requires exactly one energy-transfer axis")
    dimension = dimensions[0]
    shape = [1] * data.signal.ndim
    shape[dimension] = data.shape[dimension]
    return data.axes[dimension].centers.reshape(shape)


def spectral_polarization_factor(config: dict[str, Any]) -> float:
    """Return the scalar polarization correction declared by a spectral channel."""

    mode = str(config.get("polarization_mode", "isotropic_single_component"))
    if mode == "custom_array":
        raise ValueError("Pointwise array polarization is not supported for model fitting")
    factors = {
        "already_corrected": 1.0,
        "isotropic_single_component": 2.0,
        "isotropic_trace": 2.0 / 3.0,
        "custom_scalar": float(config.get("polarization_scalar", 2.0)),
    }
    try:
        value = float(factors[mode])
    except KeyError as exc:
        raise ValueError(f"unknown polarization convention {mode!r}") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("polarization factor must be finite and positive")
    return value


_polarization_factor = spectral_polarization_factor


def _form_factor(data: MDHistoData, config: dict[str, Any]) -> float | np.ndarray:
    ion = str(config.get("form_factor_ion", "") or "").strip()
    coefficients = config.get("custom_form_factor")
    if not ion and not coefficients:
        return 1.0
    q = q_modulus_for_spectral(data)
    return form_factor_sq(q, ion=ion or None, coefficients=coefficients)


def _kinematic_factor(
    energy: np.ndarray, config: dict[str, Any]
) -> float | np.ndarray:
    if config["kf_ki_state"] == "removed":
        return 1.0
    incident = config.get("incident_energy_meV")
    final = config.get("final_energy_meV")
    if incident in (None, "") and final in (None, ""):
        raise ValueError(
            "k_f/k_i is marked included; provide fixed incident or final energy"
        )
    return kf_over_ki(
        energy,
        incident_energy_meV=None if incident in (None, "") else float(incident),
        final_energy_meV=None if final in (None, "") else float(final),
    )


def _basis_suffix(config: dict[str, Any]) -> str:
    return normalization_basis_suffix(
        str(config["normalization_basis"]), config.get("normalization_label", ""),
    )


def _output_units(
    config: dict[str, Any],
    *,
    absolute_cross_section: bool,
    absolute_chipp: bool,
) -> tuple[str, str]:
    suffix = _basis_suffix(config)
    moment = "mu_B^2" if config["moment_unit"] == "mu_B_squared" else "spin^2"
    cross = f"mbarn/sr/meV{suffix}" if absolute_cross_section else "arb. units"
    chipp = f"{moment}/meV{suffix}" if absolute_chipp else "arb. units"
    return cross, chipp


def _arbitrary_cross_from_chipp(
    chipp: np.ndarray,
    energy: np.ndarray,
    temperature_K: float,
    *,
    form_factor_sq_values: float | np.ndarray,
    polarization: float,
    kf_ki: float | np.ndarray,
    moment_unit: str,
    g_factor: float | None,
) -> np.ndarray:
    from .spectral_conversion import spectral_conversion_factor

    return np.asarray(chipp, dtype=float) * spectral_conversion_factor(
        energy, temperature_K, source_representation="chi_double_prime",
        target_representation="cross_section", form_factor_sq=form_factor_sq_values,
        polarization=polarization, kf_ki=kf_ki, moment_unit=moment_unit,
        g_factor=g_factor, form_factor_state="removed", polarization_state="removed",
        bose_state="removed", absolute=False,
    )


def _arbitrary_chipp_from_cross(
    cross_section: np.ndarray,
    energy: np.ndarray,
    temperature_K: float,
    *,
    form_factor_sq_values: float | np.ndarray,
    polarization: float,
    kf_ki: float | np.ndarray,
    moment_unit: str,
    g_factor: float | None,
) -> np.ndarray:
    from .spectral_conversion import spectral_conversion_factor

    return np.asarray(cross_section, dtype=float) * spectral_conversion_factor(
        energy, temperature_K, source_representation="cross_section",
        target_representation="chi_double_prime", form_factor_sq=form_factor_sq_values,
        polarization=polarization, kf_ki=kf_ki, moment_unit=moment_unit,
        g_factor=g_factor, absolute=False,
    )


def with_paired_spectral_channels(
    data: MDHistoData,
    config: dict[str, Any],
    *,
    temperature_K: float | np.ndarray | None,
) -> MDHistoData:
    """Return a view with cross-section and, when possible, ``chi''`` channels.

    The input arrays are never modified.  Absolute inputs are converted through
    the magnetic cross-section constant; arbitrary inputs remove or apply only
    the Q-, polarization-, kinematic-, and detailed-balance shapes, leaving the
    unknown global calibration in arbitrary units.
    Temperatures may be scalar or broadcastable to the histogram, including a
    discrete temperature coordinate in kelvin.
    """

    convention = normalized_spectral_channel_config(config)
    if not bool(convention["enabled"]):
        return data
    values = np.asarray(data.signal, dtype=float)
    errors = np.asarray(data.errors, dtype=float)
    energy = _energy_grid(data)
    form_factor = _form_factor(data, convention)
    polarization = _polarization_factor(convention)
    channel_error: str | None = None
    try:
        kinematic: float | np.ndarray | None = _kinematic_factor(
            energy, convention
        )
    except ValueError as exc:
        kinematic = None
        channel_error = str(exc)
    source_representation = str(convention["source_representation"])
    source_unit = str(convention["source_unit"])
    calibration = float(convention.get("signal_per_mbarn", 0.0) or 0.0)
    absolute_cross_section = (
        source_unit.startswith(("mbarn/sr/meV", "barn/sr/meV"))
        or calibration > 0.0
    )
    absolute_chipp = source_unit.startswith(("mu_B^2/meV", "spin^2/meV"))
    if source_representation == "cross_section":
        absolute_chipp = absolute_cross_section
    else:
        absolute_cross_section = absolute_chipp
    moment_unit = str(convention["moment_unit"])
    g_factor = float(convention["g_factor"]) if convention.get("g_factor") is not None else None

    from .spectral_conversion import spectral_conversion_factor

    cross_scale = (
        MILLIBARN_PER_BARN if source_unit.startswith("barn/sr/meV")
        else 1.0 if source_unit.startswith("mbarn/sr/meV")
        else 1.0 / calibration if calibration > 0.0 else 1.0
    )
    cross_factor = cross_scale if source_representation == "cross_section" else None
    chipp_factor = 1.0 if source_representation == "chi_double_prime" else None
    if temperature_K is not None and kinematic is not None:
        factor = spectral_conversion_factor(
            energy, temperature_K, source_representation=source_representation,
            target_representation=("chi_double_prime" if source_representation == "cross_section" else "cross_section"),
            source_scale=(cross_scale / MILLIBARN_PER_BARN if absolute_cross_section else cross_scale),
            form_factor_sq=form_factor, polarization=polarization, kf_ki=kinematic,
            form_factor_state="included" if source_representation == "cross_section" else "removed",
            polarization_state="included" if source_representation == "cross_section" else "removed",
            bose_state="included" if source_representation == "cross_section" else "removed",
            moment_unit=moment_unit, g_factor=g_factor, absolute=absolute_chipp,
        )
        if source_representation == "cross_section":
            chipp_factor = factor
        else:
            cross_factor = factor * (MILLIBARN_PER_BARN if absolute_cross_section else 1.0)
    elif temperature_K is None:
        channel_error = (
            "Set a dataset temperature to calculate dynamic susceptibility."
            if source_representation == "cross_section"
            else "Set a dataset temperature to calculate scattering cross section."
        )
    cross = None if cross_factor is None else values * cross_factor
    cross_error = None if cross_factor is None else errors * np.abs(cross_factor)
    chipp = None if chipp_factor is None else values * chipp_factor
    chipp_error = None if chipp_factor is None else errors * np.abs(chipp_factor)

    cross_unit, chipp_unit = _output_units(
        convention,
        absolute_cross_section=absolute_cross_section,
        absolute_chipp=absolute_chipp,
    )
    if (
        source_representation == "cross_section"
        and source_unit not in {"arbitrary"}
        and not absolute_cross_section
    ):
        cross_unit = source_unit
    channels = dict(data.auxiliary_channels)
    channels[IMPORTED_SIGNAL_CHANNEL] = MDHistoChannel(
        values,
        errors,
        label="Imported signal",
        unit="arb. units" if source_unit == "arbitrary" else source_unit,
        quantity_type=(
            "scattering_intensity"
            if source_representation == "cross_section" and source_unit == "arbitrary"
            else "differential_cross_section"
            if source_representation == "cross_section"
            else "dynamic_susceptibility"
        ),
    )
    if cross is not None:
        channels[CROSS_SECTION_CHANNEL] = MDHistoChannel(
            cross,
            cross_error,
            label="Scattering cross section",
            unit=cross_unit,
            quantity_type="differential_cross_section",
        )
    if chipp is not None:
        channels[CHIPP_CHANNEL] = MDHistoChannel(
            chipp,
            chipp_error,
            label="Dynamical susceptibility χ″",
            unit=chipp_unit,
            quantity_type="dynamic_susceptibility",
        )

    selected = str(convention["fit_representation"])
    if selected == "chi_double_prime" and chipp is not None:
        quantity_type, unit, label = "dynamic_susceptibility", chipp_unit, "Dynamical susceptibility χ″"
    elif cross is not None:
        quantity_type, unit, label = "differential_cross_section", cross_unit, "Scattering cross section"
    else:
        quantity_type, unit, label = "dynamic_susceptibility", chipp_unit, "Dynamical susceptibility χ″"

    selected_chipp = (selected == "chi_double_prime" and chipp is not None) or cross is None
    factor = chipp_factor if selected_chipp else cross_factor

    metadata = dict(data.metadata)
    metadata.pop("combined_scientific_identity", None)
    previous = metadata.get("spectral_convention", {})
    ions = convention.get("magnetic_ions_per_basis")
    if ions is None and isinstance(previous, dict) and previous.get("normalization_basis") == convention["normalization_basis"]:
        ions = previous.get("magnetic_ions_per_basis")
    target = target_spectral_record(
        {
            **convention,
            "magnetic_ions_per_basis": ions,
            "moment_unit": moment_unit,
            "g_factor": g_factor,
        },
        representation="chi_double_prime" if selected_chipp else "cross_section",
        unit=unit,
        absolute_scale=absolute_chipp if selected_chipp else absolute_cross_section,
    )
    metadata["spectral_convention"] = target
    if temperature_K is not None:
        temperature = np.asarray(temperature_K, dtype=float)
        metadata["temperature"] = float(temperature) if temperature.ndim == 0 else temperature.tolist()
    metadata.update(
        {
            "signal_label": label,
            "signal_quantity_type": quantity_type,
            "signal_unit": unit,
            "spectral_channel_error": channel_error,
            "spectral_observable": {
                **convention,
                **target,
                "fit_representation": "chi_double_prime" if selected_chipp else "cross_section",
                "absolute_scale": absolute_cross_section or absolute_chipp,
                "quantity_type": quantity_type,
                "unit": unit,
                "temperature_K": temperature_K,
            },
        }
    )
    from .spectral_conversion import convert_histogram_spectral_values

    converted = convert_histogram_spectral_values(
        data, factor, unit=unit, quantity_type=quantity_type, metadata=metadata,
    )
    # The transform owns measurement and background payloads; the three INS
    # channels are presentations of the original input and its paired response.
    converted_channels = dict(converted.auxiliary_channels)
    for name in (IMPORTED_SIGNAL_CHANNEL, CROSS_SECTION_CHANNEL, CHIPP_CHANNEL):
        if name in channels:
            converted_channels[name] = channels[name]
    return converted.with_updates(auxiliary_channels=converted_channels)


def updated_spectral_channel_config(config, key, value):
    """Update one representation setting and reconcile dependent units."""
    config = normalized_spectral_channel_config(config)
    if key == "source_representation" and value != config.get(key):
        config["source_unit"] = "arbitrary"
    if key == "source_unit":
        if str(value).startswith("spin^2/"):
            config["moment_unit"] = "spin_squared"
        elif str(value).startswith("mu_B^2/"):
            config["moment_unit"] = "mu_B_squared"
    if key == "normalization_basis" and config["source_unit"] != "arbitrary":
        config["source_unit"] = spectral_unit_with_basis(
            str(config["source_unit"]), str(value), config.get("normalization_label", ""),
            representation=str(config["source_representation"]),
        )
    if key == "normalization_label" and config["source_unit"] != "arbitrary":
        config["source_unit"] = spectral_unit_with_basis(
            str(config["source_unit"]), "per_magnetic_ion", str(value).strip(),
        )
    config[key] = value
    return normalized_spectral_channel_config(config)
