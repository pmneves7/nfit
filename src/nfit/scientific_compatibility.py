"""Physical declarations required before pooling or adding measured signals.

Instrument names and acquisition energies are provenance once kinematics have
been removed. Unknown legacy declarations may agree with one another, but may
not stand in for a declared physical quantity or normalization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData
from .quantities import normalize_unit
from .spectral_metadata import unit_without_normalization_basis

SCIENTIFIC_IDENTITY_KEY = "combined_scientific_identity"


def _known(value: Any) -> Any:
    return None if value is None or isinstance(value, str) and value in ("", "unknown", "arbitrary", "arb. units") else value


def _known_unit(value: Any) -> Any:
    value = _known(value)
    if isinstance(value, str) and value.casefold().startswith(("arb. units/", "arbitrary/", "a.u./")):
        return None
    return value


def _comparable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _comparable(value.tolist())
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coordinate_unit(value: str) -> str:
    return "rlu" if value.casefold().replace(".", "").replace(" ", "") == "rlu" else normalize_unit(value)


def scientific_data_identity(data: Any, dataset: Any = None) -> dict[str, Any]:
    """Describe the actual primary signal, including its correction convention.

Dataset spectral settings describe the imported signal only when the data do
not already carry a converted primary-signal convention. No conversion or
calibration is inferred from a numeric dataset scale.
"""

    parameters = getattr(dataset, "parameters", {}) if dataset is not None else {}
    entry_metadata = getattr(dataset, "metadata", {}) if dataset is not None else {}
    metadata = {**entry_metadata, **dict(getattr(data, "metadata", {}) or {})}
    stored = metadata.get(SCIENTIFIC_IDENTITY_KEY)
    if isinstance(stored, Mapping) and not any(isinstance(metadata.get(key), Mapping) for key in ("spectral_convention", "spectral_observable")):
        return _comparable(dict(stored))
    convention = metadata.get("spectral_convention")
    observable = metadata.get("spectral_observable")
    config = parameters.get("spectral_channels")
    declaration = (
        convention if isinstance(convention, Mapping)
        else observable if isinstance(observable, Mapping)
        else config if isinstance(config, Mapping) and config.get("enabled", True)
        else {}
    )
    contract = metadata.get("measurement_contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    actual_convention = isinstance(convention, Mapping) or isinstance(observable, Mapping)
    representation = declaration.get("representation")
    if representation is None:
        representation = declaration.get("fit_representation" if actual_convention else "source_representation")
    unit = _known_unit(metadata.get("signal_unit")) or _known_unit(contract.get("value_units"))
    if unit is None:
        unit = _known_unit(declaration.get("unit" if actual_convention else "source_unit"))
    quantity = _known(metadata.get("signal_quantity_type")) or _known(contract.get("quantity"))
    if quantity is None:
        quantity = {
            "cross_section": "scattering_intensity" if declaration.get("source_unit") == "arbitrary" else "differential_cross_section" if unit else "scattering_intensity",
            "chi_double_prime": "dynamic_susceptibility",
            "s_qw": "dynamic_structure_factor",
            "measured_intensity": "scattering_intensity",
        }.get(representation)
    if quantity is None and declaration.get("source_unit") is not None:
        quantity = "scattering_intensity" if declaration["source_unit"] == "arbitrary" else None
    quantity = {
        "scattering intensity": "scattering_intensity",
        "differential cross section": "differential_cross_section",
        "dynamic susceptibility": "dynamic_susceptibility",
    }.get(quantity, quantity)
    coordinates = None
    momentum_units = energy_units = None
    if isinstance(data, MDHistoData):
        momentum_units = sorted({_coordinate_unit(axis.units) for axis in data.axes if axis.kind == "momentum"})
        energy_units = sorted({_coordinate_unit(axis.units) for axis in data.axes if axis.kind == "energy" or axis.role == "energy_transfer"})
    elif isinstance(data, PointData4D):
        momentum_units = [_coordinate_unit(str(metadata.get("coordinate_units", "rlu")))]
        energy_units = [_coordinate_unit(str(metadata.get("energy_unit", "meV")))]
    if isinstance(data, PointListData):
        coordinates = tuple((name, normalize_unit(data.unit(name))) for name in data.coordinate_names)
        if data.channels:
            name = data.channels[0]["value"]
            unit = unit or _known_unit(data.unit(name))
            quantity = quantity or _known(data.quantity_type(name))
    identity = {
        "quantity": quantity,
        "unit": normalize_unit(str(unit)) if unit is not None else None,
        "signal_semantics": _known(metadata.get("signal_semantics", "density" if isinstance(data, MDHistoData) else None)),
        "normalization_basis": _known(declaration.get("normalization_basis", metadata.get("normalization_basis"))),
        "normalization_label": _known(declaration.get("normalization_label", metadata.get("normalization_label"))),
        "magnetic_ions_per_basis": declaration.get("magnetic_ions_per_basis"),
        "absolute_scale": declaration.get("absolute_scale"),
        "coordinate_units": coordinates,
        "momentum_units": momentum_units,
        "energy_units": energy_units,
    }
    for name in ("form_factor_state", "polarization_state", "bose_state", "kf_ki_state", "moment_unit", "g_factor"):
        identity[name] = declaration.get(name, metadata.get(name))
    if actual_convention and representation == "chi_double_prime":
        # Ei/Ef retained on a susceptibility convention describe the reference
        # cross section, not a kinematic multiplier in the susceptibility.
        identity["kf_ki_state"] = "removed"
    if identity["kf_ki_state"] is None and metadata.get("ki_kf_corrected") is not None:
        identity["kf_ki_state"] = "removed" if metadata["ki_kf_corrected"] else "included"
    if not actual_convention and declaration:
        # Different raw calibrations/correction assumptions cannot be applied
        # after pooling by selecting one representative source configuration.
        for name in ("signal_per_mbarn", "form_factor_ion", "custom_form_factor", "polarization_mode", "polarization_scalar"):
            identity[name] = declaration.get(name)
    if identity["absolute_scale"] is not True:
        for name in ("normalization", "normalization_denominator_semantics", "exposure_units"):
            identity[name] = metadata.get(name, contract.get(name))
    if identity["kf_ki_state"] == "included":
        for name, aliases in (
            ("incident_energy_meV", ("incident_energy_meV", "incident_energy", "Ei", "ei")),
            ("final_energy_meV", ("final_energy_meV", "final_energy", "Ef", "ef")),
        ):
            value = None
            for source in (declaration, parameters, metadata, entry_metadata):
                value = next((source[key] for key in aliases if source.get(key) is not None), None)
                if value is not None:
                    break
            identity[name] = value
    return _comparable(identity)


def validate_scientific_compatibility(sources: Sequence[tuple[Any, Any]], *, compare_coordinates: bool = True) -> dict[str, Any]:
    """Reject physically unlike primary signals before any numerical pooling.

Each pair is ``(dataset entry or None, prepared data)``. The returned identity
is retained on a composite so nested groups obey the same rule.
"""

    identities = [(getattr(entry, "name", f"input {index + 1}"), scientific_data_identity(data, entry))
                  for index, (entry, data) in enumerate(sources)]
    if not identities:
        return {}
    name, first = identities[0]
    for other_name, identity in identities[1:]:
        keys = set(first) | set(identity)
        if not compare_coordinates:
            # Powder-background projection explicitly transforms coordinates;
            # the physical ordinate still has to agree before subtraction.
            keys -= {"coordinate_units", "momentum_units"}
        differing = sorted(key for key in keys if first.get(key) != identity.get(key))
        if differing:
            details = "; ".join(f"{key}: {first.get(key)!r} versus {identity.get(key)!r}" for key in differing)
            raise ValueError(
                f"Cannot combine {name!r} and {other_name!r}: incompatible scientific units or normalization ({details}). "
                "Prepare each source in the same physical convention before combining. "
                "Different incident/final energies require source-specific k_f/k_i removal before pooling."
            )
    return first


def combined_scientific_metadata(sources: Sequence[tuple[Any, Any]], *, signal_semantics: str | None = None) -> dict[str, Any]:
    """Retain common primary-signal declarations without acquisition provenance."""

    identity = dict(validate_scientific_compatibility(sources))
    if signal_semantics is not None:
        identity["signal_semantics"] = signal_semantics
    metadata = {SCIENTIFIC_IDENTITY_KEY: identity}
    for name in ("signal_unit", "signal_quantity_type", "spectral_convention", "spectral_observable", "normalization_basis"):
        values = [getattr(data, "metadata", {}).get(name) for _, data in sources]
        if values and values[0] is not None and all(_comparable(value) == _comparable(values[0]) for value in values[1:]):
            metadata[name] = values[0]
    return metadata


def validate_spectral_source_settings(data: Any, settings: Mapping[str, Any]) -> None:
    """Prevent a known pooled observable from being relabeled as raw input."""

    actual = scientific_data_identity(data)
    quantity = actual.get("quantity")
    if quantity not in {"differential_cross_section", "dynamic_susceptibility"} or actual.get("unit") is None:
        return
    representation = "chi_double_prime" if quantity == "dynamic_susceptibility" else "cross_section"
    mismatches = []
    if settings.get("source_representation") != representation:
        mismatches.append("source representation")
    basis = actual.get("normalization_basis")
    if basis != _known(settings.get("normalization_basis")):
        mismatches.append("normalization basis")

    label = actual.get("normalization_label")
    actual_base_unit = unit_without_normalization_basis(actual["unit"], basis, label)
    source_base_unit = unit_without_normalization_basis(settings.get("source_unit"), basis, label)
    if actual_base_unit != source_base_unit:
        mismatches.append("source unit")
    if representation == "chi_double_prime":
        if actual.get("moment_unit") is not None and actual["moment_unit"] != settings.get("moment_unit"):
            mismatches.append("moment convention")
        if actual.get("moment_unit") == "spin_squared" and actual.get("g_factor") != settings.get("g_factor"):
            mismatches.append("g factor")
        if any(actual.get(field) == "included" for field in ("form_factor_state", "polarization_state", "bose_state")):
            mismatches.append("susceptibility corrections must be removed before this conversion")
    else:
        if actual.get("bose_state") == "removed":
            mismatches.append("Bose correction was already removed")
        state = actual.get("kf_ki_state")
        if state is not None and state != settings.get("kf_ki_state"):
            mismatches.append("k_f/k_i state")
        if state == "included" and any(actual.get(field) != settings.get(field) for field in ("incident_energy_meV", "final_energy_meV")):
            mismatches.append("fixed incident/final energy")
        pol = actual.get("polarization_state")
        if pol is not None and (pol == "removed") != (settings.get("polarization_mode") == "already_corrected"):
            mismatches.append("polarization correction state")
        if actual.get("form_factor_state") == "removed" and (settings.get("form_factor_ion") or settings.get("custom_form_factor")):
            mismatches.append("form-factor correction was already removed")
    if mismatches:
        raise ValueError(
            "Composite INS source settings do not match its prepared physical signal "
            f"({', '.join(mismatches)}). Select matching source quantity, units, normalization "
            "and correction settings; calibrate or convert each source before combining."
        )
