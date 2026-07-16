"""Physical quantity labels and conservative unit conversions.

The fitting layer uses these names as a contract between imported data and
model outputs.  This is intentionally smaller than a general unit package:
only conversions whose physical meaning is unambiguous are accepted.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

QUANTITY_TYPES: Final[tuple[str, ...]] = (
    "unknown",
    "temperature",
    "magnetic_field",
    "momentum",
    "reciprocal_lattice_coordinate",
    "energy_transfer",
    "scattering_intensity",
    "differential_cross_section",
    "dynamic_susceptibility",
    "magnetic_moment",
    "magnetization",
    "bulk_susceptibility",
)

_UNIT_ALIASES = {
    "": "",
    "1": "1",
    "k": "K",
    "kelvin": "K",
    "oe": "Oe",
    "oersted": "Oe",
    "t": "T",
    "tesla": "T",
    "a/m": "A/m",
    "emu": "emu",
    "a m^2": "A m^2",
    "a m2": "A m^2",
    "am^2": "A m^2",
    "emu/oe": "emu/Oe",
    "emu/(mol oe)": "cm^3/mol",
    "emu/mol/oe": "cm^3/mol",
    "cm^3/mol": "cm^3/mol",
    "cm3/mol": "cm^3/mol",
    "m^3/mol": "m^3/mol",
    "m3/mol": "m^3/mol",
    "emu/g/oe": "cm^3/g",
    "cm^3/g": "cm^3/g",
    "m^3/kg": "m^3/kg",
    "mev": "meV",
    "rlu": "rlu",
    "a^-1": "Å⁻¹",
    "angstrom^-1": "Å⁻¹",
    "1/angstrom": "Å⁻¹",
    "å⁻¹": "Å⁻¹",
    "counts": "counts",
    "count": "counts",
    "barn/sr/mev": "barn/sr/meV",
    "barn/(sr mev)": "barn/sr/meV",
    "mu_b^2/mev": "mu_B^2/meV",
    "mub^2/mev": "mu_B^2/meV",
}


def normalize_unit(unit: str | None) -> str:
    """Return a stable spelling for a supported unit, preserving unknown text."""

    text = "" if unit is None else str(unit).strip()
    return _UNIT_ALIASES.get(text.lower(), text)


def infer_quantity_type(name: str, unit: str = "") -> str:
    """Infer a conservative quantity type from a column name and unit."""

    lowered = str(name).strip().lower()
    normalized = normalize_unit(unit)
    if "temperature" in lowered or normalized == "K":
        return "temperature"
    if "field" in lowered or normalized in {"Oe", "T", "A/m"}:
        return "magnetic_field"
    if "suscept" in lowered or normalized in {
        "emu/Oe", "cm^3/mol", "m^3/mol", "cm^3/g", "m^3/kg"
    }:
        return "bulk_susceptibility"
    if "moment" in lowered or normalized in {"emu", "A m^2"}:
        return "magnetic_moment"
    if lowered in {"h", "k", "l"} or normalized == "rlu":
        return "reciprocal_lattice_coordinate"
    if lowered in {"e", "deltae", "energy", "energy transfer"} or normalized == "meV":
        return "energy_transfer"
    if lowered in {"q", "|q|"} or normalized == "Å⁻¹":
        return "momentum"
    if normalized == "barn/sr/meV":
        return "differential_cross_section"
    if normalized == "mu_B^2/meV":
        return "dynamic_susceptibility"
    if any(token in lowered for token in ("intensity", "signal", "counts")):
        return "scattering_intensity"
    return "unknown"


def _conversion_factor(quantity_type: str, source: str, target: str) -> float:
    source = normalize_unit(source)
    target = normalize_unit(target)
    if source == target:
        return 1.0
    factors: dict[str, dict[tuple[str, str], float]] = {
        "magnetic_field": {
            ("Oe", "T"): 1.0e-4,
            ("T", "Oe"): 1.0e4,
            ("Oe", "A/m"): 1000.0 / (4.0 * np.pi),
            ("A/m", "Oe"): 4.0 * np.pi / 1000.0,
            ("T", "A/m"): 1.0e7 / (4.0 * np.pi),
            ("A/m", "T"): 4.0 * np.pi / 1.0e7,
        },
        "magnetic_moment": {
            ("emu", "A m^2"): 1.0e-3,
            ("A m^2", "emu"): 1.0e3,
        },
        "bulk_susceptibility": {
            # Molar CGS susceptibility (cm^3/mol) to rationalized SI.
            ("cm^3/mol", "m^3/mol"): 4.0 * np.pi * 1.0e-6,
            ("m^3/mol", "cm^3/mol"): 1.0e6 / (4.0 * np.pi),
            ("cm^3/g", "m^3/kg"): 4.0 * np.pi * 1.0e-3,
            ("m^3/kg", "cm^3/g"): 1.0e3 / (4.0 * np.pi),
        },
    }
    try:
        return factors[quantity_type][(source, target)]
    except KeyError as exc:
        raise ValueError(
            f"cannot convert {quantity_type} from {source or '(unknown)'} "
            f"to {target or '(unknown)'}"
        ) from exc


def convert_quantity(
    values: ArrayLike, quantity_type: str, source_unit: str, target_unit: str
) -> FloatArray:
    """Convert values between compatible units for one physical quantity."""

    if quantity_type not in QUANTITY_TYPES:
        raise ValueError(f"unknown quantity type {quantity_type!r}")
    factor = _conversion_factor(quantity_type, source_unit, target_unit)
    return np.asarray(values, dtype=float) * factor


def require_quantity_unit(quantity_type: str, unit: str) -> str:
    """Validate an explicit quantity/unit pairing and return normalized unit."""

    if quantity_type not in QUANTITY_TYPES or quantity_type == "unknown":
        raise ValueError("quantity type must be explicit")
    normalized = normalize_unit(unit)
    if not normalized:
        raise ValueError(f"{quantity_type} requires an explicit unit")
    return normalized
