"""Point-list configuration and prepared-data transformations.

This GUI-independent module owns point-list role overrides, derived scientific
channels, and the bounded prepared-data cache.  ``project_data`` re-exports its
established API for compatibility.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np

from .cache_utils import lru_store as _lru_store
from .dataset import PointListData
from .pipeline import DatasetEntry
from .project_imports import DATA_TYPE_DEFINITIONS

DATASET_POINT_LIST_KEY = "point_list"
SUSCEPTIBILITY_CHANNEL_LABEL = "Susceptibility"
INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL = "Inverse susceptibility"
HEAT_CAPACITY_CHANNEL_LABEL = "Heat capacity"
HEAT_CAPACITY_OVER_T_CHANNEL_LABEL = "C/T"
TEMPERATURE_SQUARED_COLUMN = "Temperature squared"
Q_COORDINATE_NAME = "q"
D_SPACING_COORDINATE_NAME = "d"

_PREPARED_POINT_LIST_CACHE: OrderedDict[str, tuple[str, PointListData]] = OrderedDict()
_PREPARED_POINT_LIST_CACHE_LIMIT = 16
_PREPARED_POINT_LIST_CACHE_MAX_BYTES = 128 * 1024**2


def point_list_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return the point-list transform configuration, creating defaults if needed."""

    config = dataset.parameters.get(DATASET_POINT_LIST_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.parameters[DATASET_POINT_LIST_KEY] = config
    data = dataset.data if isinstance(dataset.data, PointListData) else None
    definition = DATA_TYPE_DEFINITIONS.get(dataset.data_type, {})

    if data is not None:
        config.setdefault("coordinate_names", list(data.coordinate_names))
        config.setdefault("channels", copy.deepcopy(data.channels))
    else:
        config.setdefault("coordinate_names", [])
        config.setdefault("channels", [])

    if definition.get("scale"):
        scale = config.get("scale")
        if not isinstance(scale, dict):
            scale = {}
            config["scale"] = scale
        scale.setdefault("factor", 1.0)
        scale.setdefault("units", "")
        default_channel = config["channels"][0]["label"] if config["channels"] else ""
        scale.setdefault("channel", default_channel)
    if definition.get("susceptibility"):
        susc = config.get("susceptibility")
        if not isinstance(susc, dict):
            susc = {}
            config["susceptibility"] = susc
        susc.setdefault("enabled", False)
        field_default = next(
            (name for name in config["coordinate_names"] if "field" in name.lower()),
            (config["coordinate_names"][0] if config["coordinate_names"] else ""),
        )
        susc.setdefault("field", field_default)
        moment_default = config["channels"][0]["label"] if config["channels"] else ""
        susc.setdefault("moment", moment_default)
        susc.setdefault("output_unit", "cm^3/mol")
    if definition.get("susceptibility"):
        # This is deliberately separate from the input moment unit above: the
        # MPMS column is ordinarily a sample moment in emu, while a user may
        # choose to view and fit the same measurement per formula unit.
        dataset.parameters.setdefault("magnetization_output_unit", "emu")
    if definition.get("heat_capacity"):
        hc = config.get("heat_capacity")
        if not isinstance(hc, dict):
            hc = {}
            config["heat_capacity"] = hc
        temperature_default = next(
            (name for name in config["coordinate_names"] if "temp" in name.lower()),
            (config["coordinate_names"][0] if config["coordinate_names"] else ""),
        )
        hc.setdefault("temperature", temperature_default)
        hc.setdefault(
            "source_channel", config["channels"][0]["label"] if config["channels"] else ""
        )
        source_channel = next(
            (item for item in config["channels"] if item.get("label") == hc["source_channel"]),
            None,
        )
        hc.setdefault(
            "source_unit",
            str(source_channel.get("unit", "uJ/K")) if source_channel is not None else "uJ/K",
        )
        hc.setdefault("fit_channel", HEAT_CAPACITY_CHANNEL_LABEL)
    if definition.get("wavelength"):
        wavelength = config.get("wavelength")
        if not isinstance(wavelength, dict):
            wavelength = {}
            config["wavelength"] = wavelength
        wavelength.setdefault("value", 0.0)
        two_theta_default = next(
            (name for name in config["coordinate_names"] if "theta" in name.lower()),
            (config["coordinate_names"][0] if config["coordinate_names"] else ""),
        )
        wavelength.setdefault("two_theta", two_theta_default)
    return config


def _prepared_point_list_signature(dataset: DatasetEntry) -> str:
    """Cheap identity/config signature for immutable prepared point-list views."""

    data = dataset.data
    columns = getattr(data, "columns", {})
    payload = (
        dataset.data_cache_token,
        tuple(
            (name, id(values), np.asarray(values).shape, str(np.asarray(values).dtype))
            for name, values in columns.items()
        ),
        repr(getattr(data, "units", {})),
        repr(getattr(data, "coordinate_names", [])),
        repr(getattr(data, "channels", [])),
        repr(getattr(data, "quantity_types", {})),
        repr(dataset.parameters),
        repr(dataset.transforms),
        dataset.data_type,
    )
    return repr(payload)


def prepared_point_list_data(dataset: DatasetEntry) -> PointListData:
    """Return the point-list data after applying role overrides and transforms."""

    if not isinstance(dataset.data, PointListData):
        raise TypeError("dataset does not contain PointListData")
    cache_key = dataset.id
    signature = _prepared_point_list_signature(dataset)
    cached = _PREPARED_POINT_LIST_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        _PREPARED_POINT_LIST_CACHE.move_to_end(cache_key)
        return cached[1]
    config = point_list_config(dataset)
    base = dataset.data
    columns = {name: np.array(values, dtype=float) for name, values in base.columns.items()}
    units = dict(base.units)
    quantity_types = dict(base.quantity_types)

    coordinate_names = [name for name in config.get("coordinate_names", []) if name in columns]
    if not coordinate_names:
        coordinate_names = list(base.coordinate_names)
    channels = [
        dict(channel) for channel in config.get("channels", []) if channel.get("value") in columns
    ]
    if not channels:
        channels = [dict(channel) for channel in base.channels]
    for channel in channels:
        value_name = channel.get("value")
        if value_name not in columns:
            continue
        declared_unit = str(channel.get("unit", "") or "")
        declared_type = str(channel.get("quantity_type", "") or "")
        if declared_unit:
            units[value_name] = declared_unit
            error_name = channel.get("error")
            if error_name in columns:
                units[error_name] = declared_unit
        if declared_type:
            quantity_types[value_name] = declared_type
            error_name = channel.get("error")
            if error_name in columns:
                quantity_types[error_name] = declared_type

    definition = DATA_TYPE_DEFINITIONS.get(dataset.data_type, {})

    if definition.get("heat_capacity"):
        hc = config.get("heat_capacity", {})
        temperature_name = str(hc.get("temperature", ""))
        source_label = str(hc.get("source_channel", ""))
        source_channel = next((c for c in channels if c.get("label") == source_label), None)
        if temperature_name in columns and source_channel is not None:
            source_value = str(source_channel.get("value", ""))
            source_error = source_channel.get("error")
            temperature = np.asarray(columns[temperature_name], dtype=float)
            columns[TEMPERATURE_SQUARED_COLUMN] = temperature**2
            units[TEMPERATURE_SQUARED_COLUMN] = "K^2"
            quantity_types[TEMPERATURE_SQUARED_COLUMN] = "unknown"
            if TEMPERATURE_SQUARED_COLUMN not in coordinate_names:
                coordinate_names.append(TEMPERATURE_SQUARED_COLUMN)
            if bool(dataset.parameters.get("absolute_units", False)):
                from .heat_capacity import ppms_heat_capacity_to_molar_mJ

                source_unit = str(hc.get("source_unit") or units.get(source_value, ""))
                conversion = {
                    "source_unit": source_unit,
                    "sample_mass_mg": dataset.parameters.get("sample_mass_mg"),
                    "molar_mass_g_mol": dataset.parameters.get("molar_mass_g_mol"),
                    "atoms_per_formula_unit": dataset.parameters.get("atoms_per_formula_unit"),
                }
                heat_capacity = ppms_heat_capacity_to_molar_mJ(
                    np.asarray(columns[source_value], dtype=float), **conversion
                )
                heat_capacity_error = (
                    None
                    if source_error not in columns
                    else np.abs(
                        ppms_heat_capacity_to_molar_mJ(
                            np.asarray(columns[source_error], dtype=float), **conversion
                        )
                    )
                )
                c_name = f"{HEAT_CAPACITY_CHANNEL_LABEL} value"
                dc_name = f"{HEAT_CAPACITY_CHANNEL_LABEL} error"
                ct_name = f"{HEAT_CAPACITY_OVER_T_CHANNEL_LABEL} value"
                dct_name = f"{HEAT_CAPACITY_OVER_T_CHANNEL_LABEL} error"
                columns[c_name] = heat_capacity
                columns[ct_name] = heat_capacity / temperature
                units[c_name] = "mJ/(mol K)"
                units[ct_name] = "mJ/(mol K^2)"
                quantity_types[c_name] = "heat_capacity"
                quantity_types[ct_name] = "heat_capacity_over_temperature"
                if heat_capacity_error is not None:
                    columns[dc_name] = heat_capacity_error
                    columns[dct_name] = heat_capacity_error / np.abs(temperature)
                    units[dc_name] = "mJ/(mol K)"
                    units[dct_name] = "mJ/(mol K^2)"
                    quantity_types[dc_name] = "heat_capacity"
                    quantity_types[dct_name] = "heat_capacity_over_temperature"
                derived = [
                    {
                        "label": HEAT_CAPACITY_CHANNEL_LABEL,
                        "value": c_name,
                        "error": dc_name if heat_capacity_error is not None else None,
                        "quantity_type": "heat_capacity",
                        "unit": "mJ/(mol K)",
                    },
                    {
                        "label": HEAT_CAPACITY_OVER_T_CHANNEL_LABEL,
                        "value": ct_name,
                        "error": dct_name if heat_capacity_error is not None else None,
                        "quantity_type": "heat_capacity_over_temperature",
                        "unit": "mJ/(mol K^2)",
                    },
                ]
                labels = {item["label"] for item in derived}
                channels = derived + [item for item in channels if item.get("label") not in labels]
                selected = str(hc.get("fit_channel", HEAT_CAPACITY_CHANNEL_LABEL))
                channels.sort(key=lambda item: item.get("label") != selected)

    # Powder wavelength -> q and d-spacing coordinates (created if absent).
    if definition.get("wavelength"):
        wavelength = config.get("wavelength", {})
        value = float(wavelength.get("value", 0.0) or 0.0)
        two_theta = wavelength.get("two_theta")
        if value > 0.0 and two_theta in columns:
            theta = np.deg2rad(columns[two_theta]) / 2.0
            sin_theta = np.sin(theta)
            if Q_COORDINATE_NAME not in columns:
                # q = 4*pi*sin(theta)/lambda, in inverse angstroms.
                columns[Q_COORDINATE_NAME] = 4.0 * np.pi * sin_theta / value
                units[Q_COORDINATE_NAME] = "Å⁻¹"
            if D_SPACING_COORDINATE_NAME not in columns:
                # d = lambda / (2*sin(theta)) = 2*pi/q, in angstroms.
                with np.errstate(divide="ignore", invalid="ignore"):
                    columns[D_SPACING_COORDINATE_NAME] = value / (2.0 * sin_theta)
                units[D_SPACING_COORDINATE_NAME] = "Å"
            # Powder data is one-dimensional: q, d and 2theta are collinear, so
            # make q the single independent coordinate (used for rebin/fits) and
            # leave d and 2theta as viewable columns.
            collinear = {Q_COORDINATE_NAME, D_SPACING_COORDINATE_NAME, str(two_theta)}
            coordinate_names = [Q_COORDINATE_NAME] + [
                name for name in coordinate_names if name not in collinear
            ]

    # Magnetization scale factor + unit relabel applied to value and error.
    if definition.get("scale"):
        scale = config.get("scale", {})
        factor = float(scale.get("factor", 1.0) or 1.0)
        target = scale.get("channel")
        new_units = str(scale.get("units", "") or "")
        if factor != 1.0 or new_units:
            for channel in channels:
                if channel.get("label") != target:
                    continue
                value_name = channel.get("value")
                error_name = channel.get("error")
                if value_name in columns:
                    columns[value_name] = columns[value_name] * factor
                    if new_units:
                        units[value_name] = new_units
                if error_name in columns:
                    columns[error_name] = columns[error_name] * abs(factor)
                    if new_units:
                        units[error_name] = new_units

    # Magnetization susceptibility: moment / field (value and error divided by field).
    if definition.get("susceptibility"):
        susc = config.get("susceptibility", {})
        field_name = str(susc.get("field", ""))
        if field_name in columns and susc.get("field_unit"):
            units[field_name] = str(susc["field_unit"])
            quantity_types[field_name] = "magnetic_field"
        moment_label = susc.get("moment")
        moment_channel = next((c for c in channels if c.get("label") == moment_label), None)
        if moment_channel is not None:
            source_value = moment_channel.get("value")
            if source_value in columns and susc.get("moment_unit"):
                units[source_value] = str(susc["moment_unit"])
                quantity_types[source_value] = "magnetic_moment"
                source_error = moment_channel.get("error")
                if source_error in columns:
                    units[source_error] = units[source_value]
                    quantity_types[source_error] = "magnetic_moment"
        if susc.get("enabled"):
            if moment_channel is not None and field_name in columns:
                moment = columns[moment_channel["value"]]
                field = columns[field_name]
                moment_error_name = moment_channel.get("error")
                moment_error = columns.get(moment_error_name) if moment_error_name else None
                moment_unit = units.get(moment_channel["value"], "")
                field_unit = units.get(field_name, "")
                absolute = bool(dataset.parameters.get("absolute_units", False))
                if absolute:
                    from .quantities import convert_quantity

                    mass_g = float(dataset.parameters.get("sample_mass_mg", 0.0) or 0.0) / 1000.0
                    molar_mass = float(dataset.parameters.get("molar_mass_g_mol", 0.0) or 0.0)
                    if mass_g <= 0.0 or molar_mass <= 0.0:
                        raise ValueError(
                            "absolute susceptibility requires positive sample mass and molar mass"
                        )
                    moles = mass_g / molar_mass
                    moment = convert_quantity(moment, "magnetic_moment", moment_unit, "emu")
                    if moment_error is not None:
                        moment_error = np.abs(
                            convert_quantity(moment_error, "magnetic_moment", moment_unit, "emu")
                        )
                    field = convert_quantity(field, "magnetic_field", field_unit, "Oe")
                    output_unit = str(susc.get("output_unit", "cm^3/mol"))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        susc_value = moment / field / moles
                    if output_unit == "m^3/mol":
                        susc_value = convert_quantity(
                            susc_value, "bulk_susceptibility", "cm^3/mol", output_unit
                        )
                    elif output_unit != "cm^3/mol":
                        raise ValueError(
                            "absolute susceptibility output must be cm^3/mol or m^3/mol"
                        )
                    susc_unit = output_unit
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        susc_value = moment / field
                    susc_unit = f"{moment_unit}/{field_unit}" if moment_unit and field_unit else ""
                with np.errstate(divide="ignore", invalid="ignore"):
                    error_values = None if moment_error is None else moment_error / np.abs(field)
                if absolute and error_values is not None:
                    error_values = error_values / moles
                    if susc_unit == "m^3/mol":
                        error_values = convert_quantity(
                            error_values, "bulk_susceptibility", "cm^3/mol", susc_unit
                        )
                value_col = f"{SUSCEPTIBILITY_CHANNEL_LABEL} value"
                columns[value_col] = susc_value
                units[value_col] = susc_unit
                quantity_types[value_col] = "bulk_susceptibility"
                error_col = None
                if error_values is not None:
                    error_col = f"{SUSCEPTIBILITY_CHANNEL_LABEL} error"
                    columns[error_col] = error_values
                    units[error_col] = susc_unit
                    quantity_types[error_col] = "bulk_susceptibility"
                channels.append(
                    {
                        "label": SUSCEPTIBILITY_CHANNEL_LABEL,
                        "value": value_col,
                        "error": error_col,
                        "quantity_type": "bulk_susceptibility",
                        "unit": susc_unit,
                    }
                )
                inverse_unit = {
                    "cm^3/mol": "mol/cm^3",
                    "m^3/mol": "mol/m^3",
                    "emu/Oe": "Oe/emu",
                }.get(susc_unit, f"1/({susc_unit})" if susc_unit else "")
                inverse_value_col = f"{INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL} value"
                inverse_values = np.full(susc_value.shape, np.nan, dtype=float)
                valid_inverse = np.isfinite(susc_value) & (susc_value != 0.0)
                inverse_values[valid_inverse] = 1.0 / susc_value[valid_inverse]
                columns[inverse_value_col] = inverse_values
                units[inverse_value_col] = inverse_unit
                quantity_types[inverse_value_col] = "inverse_bulk_susceptibility"
                inverse_error_col = None
                if error_values is not None:
                    inverse_error_col = f"{INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL} error"
                    inverse_errors = np.full(error_values.shape, np.nan, dtype=float)
                    inverse_errors[valid_inverse] = (
                        np.abs(error_values[valid_inverse]) / susc_value[valid_inverse] ** 2
                    )
                    columns[inverse_error_col] = inverse_errors
                    units[inverse_error_col] = inverse_unit
                    quantity_types[inverse_error_col] = "inverse_bulk_susceptibility"
                channels.append(
                    {
                        "label": INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL,
                        "value": inverse_value_col,
                        "error": inverse_error_col,
                        "quantity_type": "inverse_bulk_susceptibility",
                        "unit": inverse_unit,
                    }
                )

        # Convert the displayed/fitted moment after deriving susceptibility so
        # susceptibility always uses the declared *input* moment unit.
        output_unit = str(dataset.parameters.get("magnetization_output_unit", "emu"))
        absolute = bool(dataset.parameters.get("absolute_units", False))
        if moment_channel is not None and output_unit != "emu":
            value_name = str(moment_channel.get("value", ""))
            error_name = moment_channel.get("error")
            if value_name in columns:
                from .quantities import convert_quantity
                from .sum_rules import EMU_PER_MOL_PER_MU_B

                input_unit = units.get(value_name, "emu")
                moment_emu = convert_quantity(
                    columns[value_name], "magnetic_moment", input_unit, "emu"
                )
                error_emu = (
                    None
                    if error_name not in columns
                    else np.abs(
                        convert_quantity(columns[error_name], "magnetic_moment", input_unit, "emu")
                    )
                )
                if output_unit == "A m^2":
                    columns[value_name] = convert_quantity(
                        moment_emu, "magnetic_moment", "emu", output_unit
                    )
                    if error_emu is not None:
                        columns[error_name] = convert_quantity(
                            error_emu, "magnetic_moment", "emu", output_unit
                        )
                elif output_unit in {"emu/mol", "mu_B/f.u."}:
                    mass_g = float(dataset.parameters.get("sample_mass_mg", 0.0) or 0.0) / 1000.0
                    molar_mass = float(dataset.parameters.get("molar_mass_g_mol", 0.0) or 0.0)
                    if not absolute or mass_g <= 0.0 or molar_mass <= 0.0:
                        raise ValueError(
                            "formula-unit moment normalization requires absolute units and positive sample mass and molar mass"
                        )
                    moles = mass_g / molar_mass
                    divisor = moles
                    if output_unit == "mu_B/f.u.":
                        divisor *= EMU_PER_MOL_PER_MU_B
                    columns[value_name] = moment_emu / divisor
                    if error_emu is not None:
                        columns[error_name] = error_emu / divisor
                else:
                    raise ValueError(f"unsupported magnetization output unit {output_unit!r}")
                units[value_name] = output_unit
                quantity_types[value_name] = "magnetic_moment"
                if error_name in columns:
                    units[error_name] = output_unit
                    quantity_types[error_name] = "magnetic_moment"
                moment_channel["unit"] = output_unit

    result = PointListData(
        columns=columns,
        units=units,
        coordinate_names=coordinate_names,
        channels=channels,
        metadata=dict(base.metadata),
        quantity_types=quantity_types,
    )
    _lru_store(
        _PREPARED_POINT_LIST_CACHE,
        cache_key,
        (_prepared_point_list_signature(dataset), result),
        _PREPARED_POINT_LIST_CACHE_LIMIT,
        _PREPARED_POINT_LIST_CACHE_MAX_BYTES,
    )
    return result
