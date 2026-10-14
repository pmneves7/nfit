"""Spectral declarations and normalization-unit formatting, without conversion.

Callers retain their own validation and unit-acceptance rules. In particular,
the analysis API records barns while paired dataset channels record millibarns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .quantities import normalize_unit

_BASIS_SUFFIXES = {
    "per_formula_unit": "/f.u.",
    "per_unit_cell": "/unit cell",
    "unknown": "",
}
_SPECTRAL_UNIT_PREFIXES = (
    "mbarn/sr/meV", "barn/sr/meV", "mu_B^2/meV", "spin^2/meV", "1/meV",
)


def normalization_basis_suffix(
    basis: str, label: str = "", *, spaced: bool = False,
) -> str:
    """Format a declared basis as a stored-unit or display-unit suffix."""

    if basis == "per_magnetic_ion":
        suffix = f"/{str(label or '').strip() or 'magnetic ion'}"
    else:
        suffix = _BASIS_SUFFIXES[basis]
    return f" / {suffix[1:]}" if spaced and suffix else suffix


def spectral_unit_with_basis(
    unit: str, basis: str, label: str = "", *, representation: str | None = None,
) -> str:
    """Rewrite a channel unit's basis using its existing prefix conventions.

    A representation requests the historical basis-edit fallback for units that
    do not match a standard prefix. Without it, unfamiliar units stay unchanged.
    This formats units; it does not decide which units an API accepts.
    """

    if representation == "cross_section":
        prefix = (
            "1/meV" if unit.startswith("1/meV")
            else "mbarn/sr/meV" if unit.startswith("mbarn/")
            else "barn/sr/meV"
        )
    elif representation is not None:
        prefix = "spin^2/meV" if unit.startswith("spin^2/") else "mu_B^2/meV"
    else:
        prefix = next((item for item in _SPECTRAL_UNIT_PREFIXES if unit.startswith(item)), None)
        if prefix is None:
            return unit
    return prefix + normalization_basis_suffix(basis, label)


def unit_without_normalization_basis(unit: Any, basis: str | None, label: str | None = None) -> str:
    """Normalize a unit and remove only a suffix declared by its known basis."""

    unit = normalize_unit(str(unit))
    if basis == "per_magnetic_ion":
        suffixes = ("/magnetic ion", f"/{label}") if label else ("/magnetic ion",)
    elif basis in {"per_formula_unit", "per_unit_cell"}:
        suffixes = (_BASIS_SUFFIXES[basis],)
    else:
        suffixes = ()
    for suffix in suffixes:
        if unit.endswith(suffix):
            return unit[:-len(suffix)]
    return unit


def target_spectral_record(
    source: Mapping[str, Any], *, representation: str, unit: str, absolute_scale: bool,
) -> dict[str, Any]:
    """Build the saved convention for a converted primary spectral signal.

    Susceptibility removes form-factor, polarization and Bose multipliers.
    The kinematic state and fixed energy remain the reference cross-section
    geometry, including on susceptibility records. Callers validate the source
    and choose their output unit and absolute-scale policy.
    """

    state = "included" if representation == "cross_section" else "removed"
    return {
        "representation": representation,
        "unit": unit,
        "normalization_basis": source["normalization_basis"],
        "magnetic_ions_per_basis": source.get("magnetic_ions_per_basis"),
        "moment_unit": source["moment_unit"],
        "g_factor": source.get("g_factor"),
        "form_factor_state": state,
        "polarization_state": state,
        "bose_state": state,
        "kf_ki_state": source["kf_ki_state"],
        "absolute_scale": absolute_scale,
        "incident_energy_meV": source.get("incident_energy_meV"),
        "final_energy_meV": source.get("final_energy_meV"),
    }
