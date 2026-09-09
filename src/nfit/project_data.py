"""GUI-independent project dataset preparation and composite services.

This module owns project-data transformations shared by the project explorer,
workflow export, and analysis execution.  It intentionally has no Qt imports;
``project_gui`` re-exports the established API for compatibility.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.artifacts import dataset_artifact_bytes
from .analysis.coordinates import signal_semantics
from .analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from .analysis.fingerprint import recipe_hash
from .analysis.registry import analysis_definition, default_analysis_parameters
from .backgrounds import subtract_background
from .cache_utils import lru_store as _lru_store
from .dataset import PointData4D, PointListData
from .importers import IMPORTERS
from .mdevent import bin_mdevent_group, bin_mdevent_powder_group
from .mdhisto import (
    MDHistoAxis,
    MDHistoChannel,
    MDHistoData,
    load_mantid_mdhisto_nxs,
    mdhisto_coverage_fraction,
    mdhisto_measured_bins,
)
from .performance import initialize_rebin_performance
from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from .project_archive import replace_dataset_artifact
from .project_history import _dataset_group_paths
from .project_imports import (
    DATA_TYPE_DEFINITIONS,
    GROUP_COMPOSITE_KEY,
    _adopt_imported_crystal,
    _loaded_data_point_count,
    _unique_dataset_name,
    data_type_container,
)
from .project_imports import (
    _ensure_dataset_data_loaded as _ensure_dataset_data_loaded_impl,
)
from .project_imports import (
    _reload_dataset_copy as _reload_dataset_copy_impl,
)
from .project_imports import (
    reload_data_group as _reload_data_group_impl,
)
from .project_imports import (
    reload_dataset_data as _reload_dataset_data_impl,
)
from .raw_dgs import bin_raw_dgs_group
from .rebin import rebin_nd, rebin_nd_symmetry
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY, with_paired_spectral_channels
from .symmetry import SymmetrySpec, resolve_symmetry, symmetry_config, symmetry_spec_from_config

DATASET_REBIN_KEY = "rebin"
DATASET_MASK_APPLICATION_KEY = "mask_application"
GROUP_COMPOSITE_NAME = "Composite"
DERIVED_RECIPE_KEY = "derived_recipe"
VIRTUAL_DERIVED_ANALYSIS_TYPES = {"dataset_clone", "histogram_arithmetic"}
DEFAULT_REBIN_MAX_BATCH_MB = 192
DEFAULT_MINIMUM_COVERAGE = 0.9
DEFAULT_MINIMUM_SAMPLES = 0.0
REBIN_COORDINATE_BASIS_VERSION = 2
REBIN_RESOLUTION_MODE_KEY = "resolution_mode"
REBIN_AXIS_MODES = frozenset({"discrete", "step", "bins", "edges", "tolerance"})
REBIN_SETTINGS_CLIPBOARD_SCHEMA = "nfit.rebin-settings"
REBIN_SETTINGS_CLIPBOARD_VERSION = 1
REBIN_SETTINGS_KEYS = (
    "enabled", "axes", "auto_rebin", "mean_weighting",
    "minimum_coverage", "minimum_samples", "max_batch_mb", "workers",
    "normalize", "symmetry",
    "coordinate_basis_version", "coordinate_mode", "metadata_dimensions",
)
REBIN_AUTO_MAX_CONTRIBUTIONS = 5_000_000
REBIN_AUTO_MAX_OUTPUT_BINS = 2_000_000
MASK_AUTO_MAX_POINTS = 5_000_000
DATASET_POINT_LIST_KEY = "point_list"
SUSCEPTIBILITY_CHANNEL_LABEL = "Susceptibility"
INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL = "Inverse susceptibility"
HEAT_CAPACITY_CHANNEL_LABEL = "Heat capacity"
HEAT_CAPACITY_OVER_T_CHANNEL_LABEL = "C/T"
TEMPERATURE_SQUARED_COLUMN = "Temperature squared"
Q_COORDINATE_NAME = "q"
D_SPACING_COORDINATE_NAME = "d"
COORDINATE_RANGE_AXIS_PREFIX = "axis_"
COORDINATE_RANGE_PARAMETER_NAMES = ("H", "K", "L", "E")
KINEMATIC_KF_KI_INCLUDED_KEY = "kf_ki_included"

_PREPARED_POINT_LIST_CACHE: OrderedDict[str, tuple[str, PointListData]] = OrderedDict()
_PREPARED_POINT_LIST_CACHE_LIMIT = 16
_PREPARED_POINT_LIST_CACHE_MAX_BYTES = 128 * 1024**2

# Loading, rebinning, and masking full datasets is reused across passive GUI
# refreshes and script/API calls. Signatures are content based; scale factors
# are deliberately applied after the cached preparation step.
_VIEWER_VIEW_CACHE: OrderedDict[str, tuple[str, Any]] = OrderedDict()
_VIEWER_VIEW_CACHE_LIMIT = 8
_VIEWER_VIEW_CACHE_MAX_BYTES = 256 * 1024**2
_COMPOSITE_DATA_CACHE: OrderedDict[int, tuple[str, Any]] = OrderedDict()
_COMPOSITE_DATA_CACHE_LIMIT = 4
_COMPOSITE_DATA_CACHE_MAX_BYTES = 256 * 1024**2


def _parameter_to_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _parse_parameter_text(text: str) -> Any:
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped

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
        hc.setdefault("source_channel", config["channels"][0]["label"] if config["channels"] else "")
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
        tuple((name, id(values), np.asarray(values).shape, str(np.asarray(values).dtype)) for name, values in columns.items()),
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
        dict(channel)
        for channel in config.get("channels", [])
        if channel.get("value") in columns
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
                    None if source_error not in columns
                    else np.abs(ppms_heat_capacity_to_molar_mJ(
                        np.asarray(columns[source_error], dtype=float), **conversion
                    ))
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
                    {"label": HEAT_CAPACITY_CHANNEL_LABEL, "value": c_name,
                     "error": dc_name if heat_capacity_error is not None else None,
                     "quantity_type": "heat_capacity", "unit": "mJ/(mol K)"},
                    {"label": HEAT_CAPACITY_OVER_T_CHANNEL_LABEL, "value": ct_name,
                     "error": dct_name if heat_capacity_error is not None else None,
                     "quantity_type": "heat_capacity_over_temperature", "unit": "mJ/(mol K^2)"},
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
                    error_values = (
                        None if moment_error is None else moment_error / np.abs(field)
                    )
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
                        np.abs(error_values[valid_inverse])
                        / susc_value[valid_inverse] ** 2
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
                    else np.abs(convert_quantity(
                        columns[error_name], "magnetic_moment", input_unit, "emu"
                    ))
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


def effective_dataset_masks(group: DataGroup, dataset: DatasetEntry) -> list[MaskSpec]:
    """Return masks inherited from the ancestor group chain of ``dataset``.

    Walks from the data group down through nested dataset groups to the
    dataset's parent, collecting each level's shared masks (outer to inner).
    Returns an empty list when the dataset is not found.
    """

    def search(node: Any) -> list[MaskSpec] | None:
        if any(item.id == dataset.id for item in getattr(node, "datasets", [])):
            return list(node.masks)
        for subgroup in getattr(node, "subgroups", []):
            below = search(subgroup)
            if below is not None:
                return list(node.masks) + below
        return None

    return search(group) or []


@dataclass(frozen=True)
class _CompositeScope:
    root: DataGroup
    node: DataGroup | DatasetGroup

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def metadata(self) -> dict[str, Any]:
        return self.node.metadata

    @property
    def masks(self) -> list[MaskSpec]:
        masks = list(self.root.masks)
        if self.node is not self.root:
            masks.extend(self.node.masks)
        return masks

    @property
    def backgrounds(self) -> list[BackgroundSpec]:
        backgrounds = list(self.root.backgrounds)
        if self.node is not self.root:
            backgrounds.extend(self.node.backgrounds)
        return backgrounds

    def iter_datasets(self):
        return self.node.iter_datasets()


def _composite_scope(
    root: DataGroup,
    node: DataGroup | DatasetGroup | None = None,
) -> DataGroup | _CompositeScope:
    return root if node is None or node is root else _CompositeScope(root, node)


def _composite_root(group: DataGroup | _CompositeScope) -> DataGroup:
    return group.root if isinstance(group, _CompositeScope) else group


def _composite_cache_key(group: DataGroup | _CompositeScope) -> int:
    return id(group.node) if isinstance(group, _CompositeScope) else id(group)


def data_group_composite_config(group: DataGroup | _CompositeScope) -> dict[str, Any]:
    """Return the group-level composite dataset configuration."""

    config = group.metadata.get(GROUP_COMPOSITE_KEY)
    if not isinstance(config, dict):
        config = {}
        group.metadata[GROUP_COMPOSITE_KEY] = config
    initialize_rebin_performance(config)
    config.setdefault("enabled", False)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    if config.get(REBIN_RESOLUTION_MODE_KEY) not in {"step", "bins"}:
        config[REBIN_RESOLUTION_MODE_KEY] = "step"
    if config.get("mean_weighting") == "normalization":
        config["mean_weighting"] = "uniform"
    elif config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "uniform"
    config["minimum_coverage"] = _rebin_minimum_coverage(config)
    config["minimum_samples"] = _rebin_minimum_samples(config)
    try:
        config["max_batch_mb"] = max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        config["max_batch_mb"] = DEFAULT_REBIN_MAX_BATCH_MB
    config["normalize"] = True
    mdevent = group.metadata.get("mdevent") if isinstance(group.metadata, dict) else None
    raw_dgs = group.metadata.get("raw_dgs") if isinstance(group.metadata, dict) else None
    event_config = mdevent if isinstance(mdevent, dict) else raw_dgs if isinstance(raw_dgs, dict) else None
    coordinate_mode = str(config.get("coordinate_mode", "hkle"))
    if coordinate_mode not in {"hkle", "powder"} or (
        coordinate_mode == "powder" and not isinstance(mdevent, dict)
    ):
        coordinate_mode = "hkle"
    config["coordinate_mode"] = coordinate_mode
    axes = config.get("axes")
    if isinstance(event_config, dict):
        dimensions = list(event_config.get("dimensions", []))
        hkl_bounds = list(event_config.get("hkl_bounds", []))
        if coordinate_mode == "powder":
            q_limits = []
            for index in range(3):
                source_dim = dimensions[index] if index < len(dimensions) else {}
                q_limits.append(
                    max(
                        abs(float(source_dim.get("lower", 0.0))),
                        abs(float(source_dim.get("upper", 0.0))),
                    )
                )
            q_upper = float(np.linalg.norm(q_limits)) or 5.0
            energy = dimensions[3] if len(dimensions) > 3 else {}
            energy_lower = float(energy.get("lower", -50.0))
            energy_upper = float(energy.get("upper", 50.0))
            default_axes = [
                {
                    "name": "|Q|",
                    "variable": "Q",
                    "lower": 0.0,
                    "upper": q_upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": 0.0,
                    "auto_upper_value": q_upper,
                    "num_bins": 100,
                    "step_size": q_upper / 100.0,
                },
                {
                    "name": "DeltaE",
                    "variable": "E",
                    "lower": energy_lower,
                    "upper": energy_upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": energy_lower,
                    "auto_upper_value": energy_upper,
                    "num_bins": 100,
                    "step_size": (energy_upper - energy_lower) / 100.0,
                },
            ]
        else:
            names = ("H", "K", "L", "DeltaE")
            default_axes = []
            for index, name in enumerate(names):
                source_dim = dimensions[index] if index < len(dimensions) else {}
                bounds = hkl_bounds[index] if index < len(hkl_bounds) else None
                lower = float(bounds[0] if bounds is not None else source_dim.get("lower", -5.0 if index < 3 else -50.0))
                upper = float(bounds[1] if bounds is not None else source_dim.get("upper", 5.0 if index < 3 else 50.0))
                default_axes.append({
                    "name": name,
                    "variable": ("H", "K", "L", "E")[index],
                    "vector": _identity_vector(index, 4),
                    "lower": lower,
                    "upper": upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": lower,
                    "auto_upper_value": upper,
                    "num_bins": 50 if index == 3 else 20,
                    "step_size": (upper - lower) / (50.0 if index == 3 else 20.0),
                })
    else:
        reference = _composite_reference_data(group)
        default_axes = _default_rebin_axes(reference) if reference is not None else []
    if not isinstance(axes, list) or not axes:
        # A composite can be restored before its reference data is loaded.
        # Do not create an empty placeholder that would prevent defaults from
        # being generated once a reference becomes available.
        if default_axes:
            config["axes"] = default_axes
    else:
        if len(axes) == len(default_axes):
            sanitized_axes = []
            for axis_config, default_axis in zip(axes, default_axes, strict=True):
                if not isinstance(axis_config, dict):
                    axis_config = {}
                for key in ("lower", "upper"):
                    auto_key = f"auto_{key}"
                    if auto_key not in axis_config:
                        axis_config[auto_key] = bool(
                            key in axis_config
                            and np.isclose(
                                float(axis_config[key]), float(default_axis[key])
                            )
                        )
                    if bool(axis_config.get(auto_key, False)):
                        axis_config.setdefault(f"{auto_key}_value", axis_config.get(key))
                if bool(default_axis.get("auto_step_size", False)) and (
                    "auto_step_size" not in axis_config
                ):
                    axis_config["auto_step_size"] = bool(
                        "step_size" in axis_config
                        and np.isclose(
                            float(axis_config["step_size"]),
                            float(default_axis["step_size"]),
                        )
                        and bool(default_axis.get("auto_step_size", False))
                    )
                if bool(default_axis.get("auto_step_size", False)) and bool(
                    axis_config.get("auto_step_size", False)
                ):
                    axis_config.setdefault(
                        "auto_step_size_value", axis_config.get("step_size")
                    )
                for key, value in default_axis.items():
                    axis_config.setdefault(key, value)
                sanitized_axes.append(_sanitize_rebin_axis_config(axis_config))
            config["axes"] = sanitized_axes
        # Preserve a saved basis if reference data are temporarily unavailable
        # or differ during a refresh. A user edit is the only operation that
        # should replace its coordinate-axis vectors.
    if "auto_rebin" not in config:
        # File-backed MDEvent composites require a complete source scan even
        # when only a few selected events contribute to the current view.
        config["auto_rebin"] = False if isinstance(event_config, dict) else not _composite_rebin_is_large(group, config)
    _migrate_rebin_axis_modes(config)
    config.setdefault("stale", False)
    return config


def _composite_source_points(group: DataGroup) -> int:
    mdevent = group.metadata.get("mdevent") if isinstance(group.metadata, dict) else None
    if isinstance(mdevent, dict):
        return int(mdevent.get("event_count", 0) or 0)
    raw_dgs = group.metadata.get("raw_dgs") if isinstance(group.metadata, dict) else None
    if isinstance(raw_dgs, dict):
        return int(raw_dgs.get("event_count", 0) or 0)
    return sum(_dataset_rebin_source_points(dataset) for dataset in _composite_candidates(group))


def _dataset_collection_point_count(node: DataGroup | DatasetGroup) -> int:
    mdevent = node.metadata.get("mdevent") if isinstance(node.metadata, dict) else None
    if isinstance(mdevent, dict):
        return int(mdevent.get("event_count", 0) or 0)
    raw_dgs = node.metadata.get("raw_dgs") if isinstance(node.metadata, dict) else None
    if isinstance(raw_dgs, dict):
        return int(raw_dgs.get("event_count", 0) or 0)
    return sum(_dataset_data_point_count(dataset) for dataset in node.iter_datasets())


def _composite_output_bins(config: dict[str, Any]) -> int:
    total = 1
    axes = config.get("axes", [])
    if not isinstance(axes, list):
        return 0
    for axis in axes:
        if isinstance(axis, dict):
            total *= max(int(axis.get("num_bins", 1) or 1), 1)
    return int(total)


def _composite_estimated_contributions(group: DataGroup, config: dict[str, Any]) -> int:
    axes = config.get("axes", []) or []
    multiplier = 2 ** sum(_rebin_fractional_axes(config, axes))
    return int(_composite_source_points(group) * multiplier)


def _composite_rebin_is_large(group: DataGroup, config: dict[str, Any]) -> bool:
    return (
        _composite_estimated_contributions(group, config) > REBIN_AUTO_MAX_CONTRIBUTIONS
        or _composite_output_bins(config) > REBIN_AUTO_MAX_OUTPUT_BINS
    )


def _composite_rebin_status_text(group: DataGroup, config: dict[str, Any]) -> str:
    if not bool(config.get("enabled", False)):
        return "Composite rebinning is off."
    size_note = "large" if _composite_rebin_is_large(group, config) else "small"
    if bool(config.get("auto_rebin", True)):
        if bool(config.get("stale", False)):
            return f"Automatic composite rebinning is on ({size_note}); pending edits will recompute on the next refresh."
        return f"Automatic composite rebinning is on ({size_note}); edits recompute the cached composite."
    if bool(config.get("stale", False)):
        return f"Manual composite rebinning is on ({size_note}); Rebin now, fit, or open the data viewer to apply pending edits."
    return f"Manual composite rebinning is on ({size_note}); the cached composite is current."


def _composite_auto_enabled(group: DataGroup) -> bool:
    return bool(data_group_composite_config(group).get("auto_rebin", True))


def _composite_rebin_is_stale(group: DataGroup) -> bool:
    return bool(data_group_composite_config(group).get("stale", False))


def _should_defer_composite_rebin(group: DataGroup, *, force_rebin: bool) -> bool:
    return (
        data_group_composite_enabled(group)
        and not force_rebin
        and not _composite_auto_enabled(group)
        and _composite_rebin_is_stale(group)
    )


def data_group_composite_enabled(group: DataGroup) -> bool:
    config = group.metadata.get(GROUP_COMPOSITE_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _composite_dataset_name(group: DataGroup) -> str:
    if isinstance(group, _CompositeScope):
        paths = _dataset_group_paths(group.root)
        if sum(node.name == group.name for _path, node in paths) > 1:
            path = next((path for path, node in paths if node is group.node), group.name)
            return f"{path} {GROUP_COMPOSITE_NAME}"
    return f"{group.name} {GROUP_COMPOSITE_NAME}"


def _composite_candidates(group: DataGroup, *, include_backgrounds: bool = False) -> list[DatasetEntry]:
    node = group.node if isinstance(group, _CompositeScope) else group
    background_ids = {
        background.source_dataset_id
        for background in getattr(group, "backgrounds", [])
        if background.enabled
    }

    def considered_datasets(
        current: DataGroup | DatasetGroup,
        *,
        selected_root: bool = False,
    ) -> Iterator[DatasetEntry]:
        if isinstance(current, DatasetGroup) and not current.enabled and not selected_root:
            return
        yield from current.datasets
        for subgroup in current.subgroups:
            yield from considered_datasets(subgroup)

    return [
        dataset
        for dataset in considered_datasets(node, selected_root=True)
        if dataset.enabled and (include_backgrounds or dataset.id not in background_ids)
    ]


def _hierarchical_composite_scopes(
    group: DataGroup | _CompositeScope,
) -> list[_CompositeScope]:
    """Return direct enabled child composites used by a parent recipe."""

    node = group.node if isinstance(group, _CompositeScope) else group
    if any(dataset.enabled for dataset in node.datasets):
        return []
    root = _composite_root(group)
    return [
        _CompositeScope(root, subgroup)
        for subgroup in node.subgroups
        if subgroup.enabled and data_group_composite_enabled(_CompositeScope(root, subgroup))
    ]


def _dataset_composite_kind(dataset: DatasetEntry) -> str:
    if dataset.kind == "mdevent":
        return "mdevent"
    if dataset.kind == "raw_dgs_nexus":
        return "raw_dgs_nexus"
    if dataset.metadata.get("importer") == "macs_nexus":
        return "point_data_4d"
    if data_type_container(dataset.data_type) == "point_list" or isinstance(dataset.data, PointListData):
        return "point_list"
    if isinstance(dataset.data, PointData4D):
        return "point_data_4d"
    if isinstance(dataset.data, MDHistoData) or data_type_container(dataset.data_type) == "mdhisto":
        return "mdhisto"
    return type(dataset.data).__name__ if dataset.data is not None else "unknown"


def data_group_composite_status(group: DataGroup) -> tuple[bool, str]:
    child_scopes = _hierarchical_composite_scopes(group)
    if child_scopes:
        for child in child_scopes:
            ok, message = data_group_composite_status(child)
            if not ok:
                return False, f"Child composite {child.name!r} is not ready: {message}"
        return True, "Ready to combine the enabled child composites."
    datasets = _composite_candidates(group, include_backgrounds=bool(group.metadata.get("metadata_dimensions")))
    if not datasets:
        return False, "No enabled datasets are available to combine."
    kinds = {_dataset_composite_kind(dataset) for dataset in datasets}
    if len(kinds) != 1:
        return False, "Composite datasets require all enabled datasets to hold the same kind of data."
    if next(iter(kinds)) == "raw_dgs_nexus":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            return False, "Select the raw TOF dataset group to configure its HKLE composite."
    if next(iter(kinds)) not in {"mdhisto", "point_list", "point_data_4d", "mdevent", "raw_dgs_nexus"}:
        return False, "This dataset kind cannot be composited yet."
    return True, "Ready to combine enabled datasets into one rebinned composite."


def _composite_reference_data(group: DataGroup) -> Any | None:
    ok, _message = data_group_composite_status(group)
    if not ok:
        return None
    child_scopes = _hierarchical_composite_scopes(group)
    if child_scopes:
        try:
            return _cached_composite_dataset_data(child_scopes[0], force_rebin=True)
        except Exception:
            return None
    dataset = _composite_candidates(group, include_backgrounds=bool(group.metadata.get("metadata_dimensions")))[0]
    try:
        return _source_data_for_group_composite(group, dataset)
    except Exception:
        return dataset.data


def _ensure_dataset_data_loaded(dataset: DatasetEntry) -> Any:
    """Load data through the GUI-compatible project import service."""

    return _ensure_dataset_data_loaded_impl(
        dataset,
        mdhisto_loader=load_mantid_mdhisto_nxs,
        dataset_file_loader=_load_nfit_dataset_file,
    )


def _reload_dataset_copy(dataset: DatasetEntry) -> DatasetEntry:
    return _reload_dataset_copy_impl(dataset, data_loader=_ensure_dataset_data_loaded)


def reload_dataset_data(dataset: DatasetEntry) -> Any:
    """Reload one dataset without mutating it when source reading fails."""

    return _reload_dataset_data_impl(
        dataset, data_loader=_ensure_dataset_data_loaded
    )


def reload_data_group(group: DataGroup | DatasetGroup) -> list[DatasetEntry]:
    """Reload every descendant dataset that has a configured data source."""

    return _reload_data_group_impl(
        group, data_loader=_ensure_dataset_data_loaded
    )


def _source_data_for_group_composite(
    group: DataGroup | _CompositeScope,
    dataset: DatasetEntry,
    *,
    include_source_masks: bool = True,
) -> Any:
    data = _ensure_dataset_data_loaded(dataset)
    node = group.node if isinstance(group, _CompositeScope) else group
    _adopt_imported_crystal(
        _composite_root(group),
        dataset,
        node if isinstance(node, DatasetGroup) else None,
    )
    extra_masks = effective_dataset_masks(_composite_root(group), dataset)
    if isinstance(data, PointListData):
        return prepared_point_list_data(dataset) if include_source_masks else data
    if isinstance(data, MDHistoData):
        if include_source_masks:
            return _mdhisto_with_nfit_masks(dataset, data=data, extra_masks=extra_masks)
        return _mdhisto_without_nfit_masks(data)
    if isinstance(data, PointData4D):
        if include_source_masks:
            return _point_data_with_nfit_masks(dataset, data, extra_masks=extra_masks)
        return data
    return data


def _composite_cache_signature(
    group: DataGroup,
    _trail: frozenset[int] = frozenset(),
) -> str:
    cache_key = _composite_cache_key(group)
    if cache_key in _trail:
        raise ValueError("composite dependency cycle through a live group background")
    trail = _trail | {cache_key}
    config = data_group_composite_config(group)
    child_scopes = _hierarchical_composite_scopes(group)
    payload = [
        json.dumps(config, sort_keys=True, default=str),
        group.metadata.get("metadata_dimensions", []),
        metadata_dimension_preview(group) if group.metadata.get("metadata_dimensions") else [],
        [
            [child.name, _composite_cache_signature(child, trail)]
            for child in child_scopes
        ],
        [
            [
                dataset.name,
                dataset.data_cache_token,
                dataset.data_type,
                dataset.kind,
                bool(dataset.enabled),
                float(dataset.fit_weight),
                float(dataset.scale_factor),
                _mask_signature(getattr(dataset, "masks", None)),
                _mask_signature(effective_dataset_masks(_composite_root(group), dataset)),
            ]
            for dataset in _composite_candidates(
                group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))
            )
        ],
        _mask_signature(getattr(group, "masks", None)),
        [
            [
                background.source_dataset_id,
                background.source_group_id,
                bool(background.enabled),
                float(background.scale),
                background.interpolation,
                (
                    _viewer_view_signature(
                        background.source_entry,
                        effective_dataset_masks(_composite_root(group), background.source_entry),
                    )
                    if background.source_entry is not None
                    else None
                ),
                (
                    _composite_cache_signature(
                        _CompositeScope(_composite_root(group), background.source_group),
                        trail,
                    )
                    if background.source_group is not None
                    else None
                ),
            ]
            for background in getattr(group, "backgrounds", [])
        ],
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def composite_dataset_data(
    group: DataGroup | _CompositeScope,
    *,
    node: DatasetGroup | None = None,
    metadata_dimensions_override: Sequence[dict[str, Any]] | None = None,
    progress_callback: Any | None = None,
    config_override: Mapping[str, Any] | None = None,
    include_source_masks: bool = True,
) -> MDHistoData | PointListData | PointData4D:
    """Build a composite using its saved rebin configuration and worker ceiling.

    ``node`` selects a nested collection. Saved plots can pass a coordinate
    recipe snapshot through ``metadata_dimensions_override``; an empty list
    explicitly requests no metadata dimensions.
    """
    from ._parallel import thread_budget

    if node is not None:
        group = _composite_scope(group, node)
    config = config_override if config_override is not None else data_group_composite_config(group)
    with thread_budget(config.get("workers")):
        return _composite_dataset_data(
            group, progress_callback=progress_callback, config_override=config_override,
            include_source_masks=include_source_masks,
            metadata_dimensions_override=metadata_dimensions_override,
        )


def _composite_progress_callback(
    callback: Any | None,
    datasets_total: int,
) -> Any | None:
    """Annotate composite progress with its source-dataset count."""

    if callback is None:
        return None

    def report(event: dict[str, Any]) -> None:
        enriched = dict(event)
        if str(enriched.get("stage", "")).startswith("rebin"):
            enriched["datasets_total"] = int(datasets_total)
            iteration = enriched.get("iteration")
            total = enriched.get("total")
            if enriched.get("stage") == "rebin" and iteration is not None and total:
                enriched["message"] = (
                    f"rebinning {datasets_total} datasets: "
                    f"{iteration}/{total} point contributions"
                )
        callback(enriched)

    return report


def metadata_dimension_preview(group, dimensions=None) -> list[dict[str, Any]]:
    """Resolve metadata coordinates for each enabled source without rebinning."""
    import hashlib

    from .metadata_dimensions import MetadataDimension, metadata_dimension_coordinates

    specs = [
        MetadataDimension(**item)
        for item in (
            group.metadata.get("metadata_dimensions", []) if dimensions is None else dimensions
        )
    ]
    rows = []
    for dataset in _composite_candidates(group, include_backgrounds=True):
        _ensure_dataset_data_loaded(dataset)
        for spec in specs:
            values = metadata_dimension_coordinates(dataset, spec)

            rows.append(
                {
                    "dataset": dataset.name,
                    "dimension": spec.name,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "count": int(values.size),
                    "signature": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )
    return rows


def set_metadata_dimensions(group, dimensions) -> None:
    """Save validated discrete-coordinate recipes on a dataset collection.

    Pass MetadataDimension objects or their dictionary representations; an
    empty list removes the dimensions. Recompute with composite_dataset_data.
    """
    from .metadata_dimensions import MetadataDimension

    specs = [
        item if isinstance(item, MetadataDimension) else MetadataDimension(**item)
        for item in dimensions
    ]
    if len({spec.name.casefold() for spec in specs}) != len(specs):
        raise ValueError("metadata dimension names must be unique")
    config = group.metadata.setdefault(GROUP_COMPOSITE_KEY, {})
    if {spec.name.casefold() for spec in specs} & {
        str(axis.get("name", "")).casefold() for axis in config.get("axes", [])
    }:
        raise ValueError("metadata dimension names must differ from existing axes")
    if specs:
        kinds = {_dataset_composite_kind(item) for item in _composite_candidates(group, include_backgrounds=True)}
        if not kinds or not kinds <= {"point_data_4d", "mdhisto"}:
            raise ValueError(
                "metadata dimensions currently require loaded neutron points or histograms; raw event logs need an alignment adapter"
            )
    group.metadata["metadata_dimensions"] = [spec.to_dict() for spec in specs]
    config["stale"] = True


def _metadata_composite_data(group, config, dimensions, *, include_source_masks, progress_callback):
    from .metadata_dimensions import (
        MetadataDimension,
        metadata_dimension_coordinates,
        metadata_dimension_grid,
        stack_metadata_histograms,
    )

    specs = [MetadataDimension(**item) for item in dimensions]
    if _hierarchical_composite_scopes(group):
        raise ValueError(
            "configure metadata dimensions on the collection containing the source datasets"
        )
    entries, coordinates = [], []
    origins = {}
    for dataset in _composite_candidates(group, include_backgrounds=True):
        data = _source_data_for_group_composite(
            group, dataset, include_source_masks=include_source_masks
        )
        if not isinstance(data, (PointData4D, MDHistoData)):
            raise ValueError(
                "metadata dimensions require neutron points or histograms; event logs need an alignment adapter"
            )
        if isinstance(data, MDHistoData) and any(
            "metadata_dimension" in axis.metadata for axis in data.axes
        ):
            raise ValueError("add all metadata dimensions on the original source collection")
        coordinates.append([metadata_dimension_coordinates(dataset, spec) for spec in specs])
        prepared_entry = dataset.copy(data=data)
        entries.append(prepared_entry)
        origins[prepared_entry.id] = dataset
    grids = [
        metadata_dimension_grid(spec, [row[i] for row in coordinates])
        for i, spec in enumerate(specs)
    ]
    centers = [grid[0] for grid in grids]
    assignments = list(zip(*(grid[1] for grid in grids), strict=True))
    reducer = (
        _composite_point_data
        if isinstance(entries[0].data, PointData4D)
        else _composite_mdhisto_data
    )
    # Reference slices share the sample grid without expanding its auto limits.
    sample_ids = {entry.id for entry in _composite_candidates(group)}
    grid_entries = [entry for entry in entries if origins[entry.id].id in sample_ids] or entries
    template = reducer(group, config, datasets=grid_entries, progress_callback=progress_callback)
    from .mdevent import _available_memory_bytes

    output_bins = math.prod(template.shape) * math.prod(len(center) for center in centers)
    bytes_per_bin = 33 + sum(
        8 if channel.errors is None else 16 for channel in template.auxiliary_channels.values()
    )
    available = _available_memory_bytes()
    if available is not None and output_bins * bytes_per_bin * 3 > available * 0.7:
        raise MemoryError(
            "The metadata grid exceeds available memory. Use fewer explicit metadata coordinates or a coarser spatial grid."
        )
    fixed = copy.deepcopy(config)
    for axis_config, axis in zip(fixed["axes"], template.axes, strict=True):
        axis_config.update(
            bin_edges=axis.values.tolist(), auto_lower=False, auto_upper=False, auto_step_size=False
        )
    partitions = {}
    for entry, indices in zip(entries, assignments, strict=True):
        size = entry.data.size if isinstance(entry.data, PointData4D) else entry.data.signal.size
        keys = np.column_stack([np.broadcast_to(index, (size,)) for index in indices])
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        positions = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
        for key, selected in zip(unique, positions, strict=True):
            if np.any(key < 0):
                continue
            partitions.setdefault(tuple(int(i) for i in key), []).append((entry, selected))
    slices = {}
    for key, sources in partitions.items():
        selected_entries = []
        for entry, selected in sources:
            if isinstance(entry.data, PointData4D):
                changes = {
                    name: getattr(entry.data, name)[selected]
                    for name in ("H", "K", "L", "E", "intensity", "sigma", "mask")
                }
                if isinstance(entry.data.temperature, np.ndarray):
                    changes["temperature"] = entry.data.temperature[selected]
                if (
                    isinstance(entry.data.magnetic_field, np.ndarray)
                    and entry.data.magnetic_field.ndim == 2
                ):
                    changes["magnetic_field"] = entry.data.magnetic_field[selected]
                data = entry.data.with_updates(**changes)
                usable = np.any(data.valid_mask(require_positive_sigma=False))
            else:
                reject = np.ones(entry.data.signal.size, dtype=bool)
                reject[selected] = False
                data = entry.data.with_updates(
                    mask=entry.data.mask | reject.reshape(entry.data.shape)
                )
                usable = np.any(
                    ~data.mask
                    & np.isfinite(data.signal)
                    & np.isfinite(data.errors)
                    & mdhisto_measured_bins(data)
                )
            if usable:
                selected_entries.append(entry.copy(data=data))
        if not selected_entries:
            continue
        data = reducer(group, fixed, datasets=selected_entries, progress_callback=progress_callback)
        data = _apply_mdhisto_coverage_threshold(data, config)
        slices[key] = _apply_composite_backgrounds(
            group, data, config=fixed, progress_callback=progress_callback,
            reference_entry=(
                origins[sources[0][0].id] if len(sources) == len(selected_entries) == 1
                and len(sources[0][1]) == (
                    sources[0][0].data.size if isinstance(sources[0][0].data, PointData4D)
                    else sources[0][0].data.signal.size
                ) else None
            ),
        )
    # Background subtraction can add channels. Use its result as the schema.
    if slices:
        template = next(iter(slices.values()))
    return stack_metadata_histograms(template, slices, specs, centers)


def _composite_dataset_data(
    group: DataGroup | _CompositeScope,
    *,
    progress_callback: Any | None = None,
    config_override: Mapping[str, Any] | None = None,
    include_source_masks: bool = True,
    metadata_dimensions_override: Sequence[dict[str, Any]] | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Build a composite from underlying sources on the requested output grid.

    ``config_override`` is used by source-linked derived datasets. It changes
    only the target reduction grid; source membership, calibration scales, and
    attached background recipes remain owned by the source collection.
    """

    ok, message = data_group_composite_status(group)
    if not ok:
        raise ValueError(message)
    progress_callback = _composite_progress_callback(
        progress_callback,
        len(_composite_candidates(group)),
    )
    config = (
        copy.deepcopy(dict(config_override))
        if config_override is not None
        else data_group_composite_config(group)
    )
    dimensions = (
        group.metadata.get("metadata_dimensions", [])
        if metadata_dimensions_override is None
        else metadata_dimensions_override
    )
    if dimensions:
        return _metadata_composite_data(
            group, config, dimensions, include_source_masks=include_source_masks,
            progress_callback=progress_callback,
        )
    child_scopes = _hierarchical_composite_scopes(group)
    child_entries: list[DatasetEntry] | None = None
    if child_scopes:
        child_entries = []
        for child in child_scopes:
            child_data = composite_dataset_data(
                child,
                progress_callback=progress_callback,
                config_override=config,
                include_source_masks=include_source_masks,
            )
            if not isinstance(child_data, MDHistoData):
                raise TypeError("hierarchical composites currently require gridded child composites")
            child_entries.append(
                DatasetEntry(
                    child.name,
                    child_data,
                    kind="mdhisto",
                    data_type=(
                        "powder_inelastic"
                        if len(child_data.axes) == 2
                        else "single_crystal_inelastic"
                    ),
                    metadata={"composite": True, "source_group_id": child.node.id},
                )
            )
        kind = "mdhisto"
    else:
        kind = _dataset_composite_kind(_composite_candidates(group)[0])
    if kind in {"mdevent", "raw_dgs_nexus"}:
        config = copy.deepcopy(config)
        event_axes = [
            _sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])
        ]
        event_bounds = [
            (float(axis["lower"]), float(axis["upper"])) for axis in event_axes
        ]
        event_symmetry = _rebin_symmetry_matrices(
            config, _composite_root(group).lattice_parameters
        )
        if event_symmetry is not None and len(event_axes) == 4:
            event_basis = _validate_mdhisto_rebin_basis(event_axes, 4)
            corner_grids = np.meshgrid(
                *([lower, upper] for lower, upper in event_bounds), indexing="ij"
            )
            output_corners = np.stack(corner_grids, axis=-1).reshape(-1, 4)
            event_bounds = _symmetry_projected_coordinate_bounds(
                output_corners @ event_basis,
                event_symmetry,
                event_basis,
            )
        config["axes"] = _resolve_auto_rebin_axes(
            event_axes,
            event_bounds,
        )
    result: MDHistoData | PointListData | PointData4D
    if kind == "mdevent":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            raise ValueError("MDEvent composites must be imported inside a dataset group")
        lower, upper, num_bins = _composite_rebin_bounds(config)
        allow_overcommit = bool(config.pop("_allow_memory_overcommit_once", False))
        if config.get("coordinate_mode") == "powder":
            result = bin_mdevent_powder_group(
                node,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=_composite_rebin_step_sizes(config),
                bin_edges=_composite_rebin_bin_edges(config),
                minimum_samples=_rebin_minimum_samples(config),
                datasets=_composite_candidates(group),
                max_batch_bytes=_rebin_max_batch_bytes(config),
                progress_callback=progress_callback,
            )
        else:
            result = bin_mdevent_group(
                node,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=_composite_rebin_step_sizes(config),
                bin_edges=_composite_rebin_bin_edges(config),
                minimum_samples=_rebin_minimum_samples(config),
                datasets=_composite_candidates(group),
                vectors=[
                    axis.get("vector", _identity_vector(index, 4))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                axis_names=[
                    str(axis.get("name", ("H", "K", "L", "DeltaE")[index]))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                max_batch_bytes=_rebin_max_batch_bytes(config),
                enforce_memory_limit=not allow_overcommit,
                progress_callback=progress_callback,
                symmetry_operations=_rebin_symmetry_matrices(config, _composite_root(group).lattice_parameters),
            )
    elif kind == "raw_dgs_nexus":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            raise ValueError("raw direct-geometry composites must be imported inside a dataset group")
        lower, upper, num_bins = _composite_rebin_bounds(config)
        result = bin_raw_dgs_group(
            node, lower=lower, upper=upper, num_bins=num_bins,
            step_size=_composite_rebin_step_sizes(config),
            bin_edges=_composite_rebin_bin_edges(config),
            minimum_samples=_rebin_minimum_samples(config),
            datasets=_composite_candidates(group),
            vectors=[axis.get("vector", _identity_vector(index, 4)) for index, axis in enumerate(config.get("axes", []))],
            axis_names=[str(axis.get("name", ("H", "K", "L", "DeltaE")[index])) for index, axis in enumerate(config.get("axes", []))],
            max_batch_bytes=_rebin_max_batch_bytes(config), progress_callback=progress_callback,
            symmetry_operations=_rebin_symmetry_matrices(config, _composite_root(group).lattice_parameters),
        )
    elif kind == "mdhisto":
        result = _composite_mdhisto_data(
            group,
            config,
            datasets=child_entries,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    elif kind == "point_list":
        result = _composite_point_list_data(
            group,
            config,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    elif kind == "point_data_4d":
        result = _composite_point_data(
            group,
            config,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    else:
        raise ValueError(f"unsupported composite dataset kind {kind!r}")
    if isinstance(result, MDHistoData):
        result = _apply_mdhisto_coverage_threshold(result, config)
    return _apply_composite_backgrounds(
        group,
        result,
        config=config,
        progress_callback=progress_callback,
    )


def _apply_mdhisto_coverage_threshold(
    data: MDHistoData,
    config: dict[str, Any],
) -> MDHistoData:
    coverage = mdhisto_coverage_fraction(data)
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    channels = dict(data.auxiliary_channels)
    channels["coverage_fraction"] = MDHistoChannel(
        coverage,
        label="Coverage",
        unit="fraction",
    )
    metadata = dict(data.metadata)
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    rebin_metadata = dict(metadata.get("rebin", {}))
    rebin_metadata["minimum_coverage"] = _rebin_minimum_coverage(config)
    metadata["rebin"] = rebin_metadata
    return data.with_updates(
        mask=np.asarray(data.mask, dtype=bool) | coverage_mask,
        metadata=metadata,
        auxiliary_channels=channels,
    )


def _apply_composite_backgrounds(
    group: DataGroup | _CompositeScope,
    data: MDHistoData | PointListData | PointData4D,
    *,
    config: Mapping[str, Any] | None = None,
    reference_entry: DatasetEntry | None = None,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Apply backgrounds owned by a composite scope after it is combined."""

    backgrounds = list(getattr(group, "backgrounds", []))
    if not backgrounds:
        return data
    if not isinstance(data, MDHistoData):
        raise TypeError("group backgrounds require a gridded composite")
    root = _composite_root(group)
    metadata = dict(data.metadata)
    if root.lattice_parameters and not isinstance(
        metadata.get("lattice_parameters"), dict
    ):
        metadata["lattice_parameters"] = dict(root.lattice_parameters)
    result = replace(data, metadata=metadata)
    for background in backgrounds:
        if not background.enabled:
            continue
        source = background.source_entry
        source_group = background.source_group
        if source is None and source_group is None:
            raise ValueError(
                f"background {background.name!r} refers to a missing dataset or group"
            )
        if source_group is not None:
            source_data = _cached_composite_dataset_data(
                _CompositeScope(root, source_group),
                force_rebin=True,
                progress_callback=progress_callback,
            )
        else:
            source_data = _viewer_data_before_scale(
                source,
                extra_masks=effective_dataset_masks(root, source),
                force_rebin=True,
                force_masks=True,
                progress_callback=progress_callback,
            )
            if isinstance(source_data, PointData4D):
                # An unrebinned neutron reference follows the resolved sample
                # grid. Work on copies so its own viewing recipe stays intact.
                aligned_config = copy.deepcopy(dict(
                    data_group_composite_config(group) if config is None else config
                ))
                if len(data.axes) != 4 or len(aligned_config.get("axes", [])) != 4:
                    raise ValueError("point-data group backgrounds require a four-axis HKLE grid")
                for settings, axis in zip(aligned_config["axes"], data.axes, strict=True):
                    settings.update(
                        name=axis.name, units=axis.units, bin_edges=axis.values.tolist(),
                        auto_lower=False, auto_upper=False, auto_step_size=False,
                    )
                source_metadata = dict(source_data.metadata)
                if root.lattice_parameters:
                    source_metadata.setdefault("lattice_parameters", dict(root.lattice_parameters))
                source_data = _rebin_point_data(
                    source_data.with_updates(metadata=source_metadata), aligned_config,
                    progress_callback=progress_callback,
                )
                if source.backgrounds:
                    source_data = _apply_dataset_backgrounds(source, source_data)
        if not isinstance(source_data, MDHistoData):
            raise TypeError(
                f"background {background.name!r} must refer to gridded histogram data"
            )
        cancels_self = (
            reference_entry is not None and source is not None
            and reference_entry.id == source.id
            and reference_entry.scale_factor == background.scale
            and result.shape == source_data.shape
            and np.allclose(result.signal, background.scale * source_data.signal,
                            rtol=1e-12, atol=1e-12, equal_nan=True)
            and np.allclose(result.num_events, source_data.num_events, rtol=1e-12, atol=1e-12)
        )
        result = subtract_background(
            result,
            source_data,
            scale=background.scale,
            interpolation=background.interpolation,
        )
        if cancels_self:
            # Identical observations are fully correlated: Var(X - X) = 0.
            result = result.with_updates(
                signal=np.where(result.mask, np.nan, 0.0),
                errors=np.where(result.mask, np.nan, 0.0),
            )
    return result


def _cached_composite_dataset_data(
    group: DataGroup,
    *,
    force_rebin: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D | None:
    signature = _composite_cache_signature(group)
    cache_key = _composite_cache_key(group)
    cached = _COMPOSITE_DATA_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        _COMPOSITE_DATA_CACHE.move_to_end(cache_key)
        return cached[1]
    if cached is not None and _should_defer_composite_rebin(group, force_rebin=force_rebin):
        return cached[1]
    if _should_defer_composite_rebin(group, force_rebin=force_rebin):
        return None
    result = composite_dataset_data(group, progress_callback=progress_callback)
    config = data_group_composite_config(group)
    config["stale"] = False
    signature = _composite_cache_signature(group)
    _lru_store(
        _COMPOSITE_DATA_CACHE,
        cache_key,
        (signature, result),
        _COMPOSITE_DATA_CACHE_LIMIT,
        _COMPOSITE_DATA_CACHE_MAX_BYTES,
    )
    return result


def _derived_analysis_for_dataset(
    dataset: DatasetEntry,
) -> tuple[DataGroup, AnalysisEntry] | None:
    recipe = dataset.metadata.get(DERIVED_RECIPE_KEY)
    group = getattr(dataset, "_derived_owner_group", None)
    if not isinstance(recipe, dict) or not isinstance(group, DataGroup):
        return None
    analysis_id = str(recipe.get("analysis_id", ""))
    analysis = next((item for item in group.analyses if item.id == analysis_id), None)
    return (group, analysis) if analysis is not None else None


def _derived_source_scope(
    group: DataGroup,
    source_id: str,
) -> DataGroup | _CompositeScope | None:
    prefix = "group-composite:"
    if not str(source_id).startswith(prefix):
        return None
    suffix = str(source_id)[len(prefix) :]
    if suffix == "root":
        return group
    node = next(
        (candidate for candidate in group.iter_subgroups() if candidate.id == suffix),
        None,
    )
    if node is None:
        raise KeyError(f"derived dataset refers to missing group composite {source_id}")
    return _CompositeScope(group, node)


def _derived_output_config_for_source(
    group: DataGroup,
    source_id: str,
) -> dict[str, Any]:
    scope = _derived_source_scope(group, source_id)
    if scope is not None:
        config = copy.deepcopy(data_group_composite_config(scope))
    else:
        source = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        if source is None:
            raise KeyError(f"derived dataset refers to missing dataset ID {source_id}")
        loaded = _ensure_dataset_data_loaded(source)
        config = copy.deepcopy(source.parameters.get(DATASET_REBIN_KEY, {}))
        if not isinstance(config.get("axes"), list) or not config.get("axes"):
            config["axes"] = _default_rebin_axes(loaded)
    config["enabled"] = True
    config["stale"] = True
    config.setdefault("auto_rebin", False)
    config.setdefault("normalize", True)
    config.setdefault("mean_weighting", "uniform")
    config.setdefault("minimum_coverage", 0.0)
    config.setdefault("minimum_samples", DEFAULT_MINIMUM_SAMPLES)
    config.setdefault("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    return config


def create_derived_analysis_dataset(
    group: DataGroup,
    analysis: AnalysisEntry,
    *,
    name: str | None = None,
    rebin_config: Mapping[str, Any] | None = None,
) -> DatasetEntry:
    """Create or update a live derived dataset evaluated from source data.

    Clone and histogram-arithmetic recipes own their output grid. The grid is
    pushed into every source reduction, so raw MDEvent or point data are binned
    directly rather than rebinning an already-binned input histogram.
    """

    if analysis.type not in VIRTUAL_DERIVED_ANALYSIS_TYPES:
        raise ValueError(f"analysis type {analysis.type!r} is not a live derived dataset")
    expected = 1 if analysis.type == "dataset_clone" else 2
    if len(analysis.input_dataset_ids) != expected:
        raise ValueError(f"{analysis.type} requires exactly {expected} source(s)")
    config = (
        copy.deepcopy(dict(rebin_config))
        if rebin_config is not None
        else _derived_output_config_for_source(group, analysis.input_dataset_ids[0])
    )
    config["enabled"] = True
    analysis.metadata["output_rebin_config"] = copy.deepcopy(config)
    linked = next(
        (
            dataset
            for dataset in group.iter_datasets()
            if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id")
            == analysis.id
        ),
        None,
    )
    if linked is None:
        derived = next(
            (node for node in group.subgroups if node.name == "Derived data"),
            None,
        )
        if derived is None:
            derived = DatasetGroup("Derived data")
            group.subgroups.append(derived)
        if analysis.type == "dataset_clone":
            default_name = f"{analysis.name} dataset"
        else:
            default_name = analysis.name
        linked = DatasetEntry(
            _unique_dataset_name(name or default_name, group.dataset_names),
            None,
            kind="derived_recipe",
            data_type=(
                "powder_inelastic"
                if len(config.get("axes", [])) == 2
                else "single_crystal_inelastic"
            ),
            metadata={},
            parameters={DATASET_REBIN_KEY: config},
            enabled=False,
            fit_weight=0.0,
        )
        derived.datasets.append(linked)
    else:
        linked.kind = "derived_recipe"
        linked.parameters[DATASET_REBIN_KEY] = config
        linked.replace_data(None, source_backed=False)
    linked.metadata.pop("source_file", None)
    linked.metadata.pop("analysis_artifact_path", None)
    linked.metadata["derived_from_analysis"] = {
        "analysis_id": analysis.id,
        "output_key": "clone" if analysis.type == "dataset_clone" else "histogram",
    }
    linked.metadata[DERIVED_RECIPE_KEY] = {
        "analysis_id": analysis.id,
        "operation": analysis.type,
        "input_source_ids": list(analysis.input_dataset_ids),
        "source_stage": "underlying_data",
    }
    linked._derived_owner_group = group
    if analysis not in group.analyses:
        group.analyses.append(analysis)
    output_key = "clone" if analysis.type == "dataset_clone" else "histogram"
    analysis.result = AnalysisResultRecord(
        recipe_hash=recipe_hash(
            analysis.type,
            analysis_definition(analysis.type).version,
            {**default_analysis_parameters(analysis.type), **analysis.parameters},
            analysis.input_dataset_ids,
        ),
        input_fingerprints={},
        outputs=[
            AnalysisOutputRef(
                output_key,
                linked.name,
                "dataset",
                dataset_id=linked.id,
                metadata={
                    "data_type": linked.data_type,
                    "fit_enabled": False,
                    "fit_weight": 0.0,
                    "virtual": True,
                },
            )
        ],
        status="live",
        created_at=datetime.now().isoformat(timespec="seconds"),
        diagnostics={"source_stage": "underlying_data"},
    )
    return linked


def _derived_source_data(
    group: DataGroup,
    source_id: str,
    config: Mapping[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    scope = _derived_source_scope(group, source_id)
    if scope is not None:
        return composite_dataset_data(
            scope,
            progress_callback=progress_callback,
            config_override=config,
            include_source_masks=False,
        )
    source = next(
        (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
        None,
    )
    if source is None:
        raise KeyError(f"derived dataset refers to missing dataset ID {source_id}")
    loaded = _ensure_dataset_data_loaded(source)
    reduction_source = (
        _mdhisto_without_nfit_masks(loaded)
        if isinstance(loaded, MDHistoData)
        else loaded
    )
    temporary = source.copy(
        data=reduction_source,
        parameters={**copy.deepcopy(source.parameters), DATASET_REBIN_KEY: copy.deepcopy(dict(config))},
        masks=[],
    )
    temporary.parameters[DATASET_REBIN_KEY]["enabled"] = True
    result = rebinned_dataset_data(
        temporary,
        extra_masks=[],
        progress_callback=progress_callback,
    )
    if isinstance(result, MDHistoData) and temporary.backgrounds:
        result = _apply_dataset_backgrounds(temporary, result)
    return _apply_dataset_scale(source, result)


def derived_analysis_dataset_data(
    dataset: DatasetEntry,
    *,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Evaluate one live derived recipe on its underlying source data."""

    resolved = _derived_analysis_for_dataset(dataset)
    if resolved is None:
        raise ValueError(f"derived dataset {dataset.name!r} has no live analysis owner")
    group, analysis = resolved
    config = copy.deepcopy(dataset_rebin_config(dataset))
    config["enabled"] = True
    sources = [
        _derived_source_data(
            group,
            source_id,
            config,
            progress_callback=progress_callback,
        )
        for source_id in analysis.input_dataset_ids
    ]
    if analysis.type == "dataset_clone":
        result = sources[0]
    elif analysis.type == "histogram_arithmetic":
        if not all(isinstance(item, MDHistoData) for item in sources):
            raise TypeError("histogram arithmetic requires gridded source reductions")
        from .analysis.data_reduction import combine_aligned_histograms

        result = combine_aligned_histograms(
            sources[0],
            sources[1],
            operation=str(analysis.parameters.get("operation", "subtract")),
            right_scale=float(analysis.parameters.get("right_scale", 1.0)),
        )
    else:
        raise ValueError(f"unsupported live derived operation {analysis.type!r}")
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata[DERIVED_RECIPE_KEY] = {
        "analysis_id": analysis.id,
        "operation": analysis.type,
        "input_source_ids": list(analysis.input_dataset_ids),
        "source_stage": "underlying_data",
    }
    if isinstance(result, MDHistoData):
        return result.with_updates(metadata=metadata)
    if isinstance(result, PointListData):
        return result.with_updates(metadata=metadata)
    return result.with_updates(metadata=metadata)


def composite_dataset_entry(
    group: DataGroup,
    *,
    force_rebin: bool = True,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    child_scopes = _hierarchical_composite_scopes(group)
    datasets = _composite_candidates(group, include_backgrounds=bool(group.metadata.get("metadata_dimensions")))
    first = datasets[0] if datasets else None
    config = data_group_composite_config(group)
    return DatasetEntry(
        name=_composite_dataset_name(group),
        data=_cached_composite_dataset_data(
            group,
            force_rebin=force_rebin,
            progress_callback=progress_callback,
        ),
        kind=("mdhisto" if child_scopes else first.kind if first is not None else ""),
        data_type=(
            (
                "powder_inelastic"
                if len(config.get("axes", [])) == 2
                else "single_crystal_inelastic"
            )
            if child_scopes
            else first.data_type if first is not None else ""
        ),
        metadata={"source_group": group.name, "composite": True},
        parameters=(
            {
                SPECTRAL_CHANNEL_CONFIG_KEY: copy.deepcopy(
                    first.parameters[SPECTRAL_CHANNEL_CONFIG_KEY]
                )
            }
            if first is not None
            and SPECTRAL_CHANNEL_CONFIG_KEY in first.parameters
            else {}
        ),
        enabled=True,
        fit_weight=1.0,
        scale_factor=1.0,
    )


def materialize_composite_dataset(
    project_path: str | Path,
    group: DataGroup,
    node: DataGroup | DatasetGroup | None = None,
    *,
    name: str | None = None,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    """Store a current composite as a project-owned, reusable dataset."""

    scope = _composite_scope(group, node)
    data = composite_dataset_data(scope, progress_callback=progress_callback)
    if not isinstance(data, MDHistoData):
        raise TypeError("materialized composites currently require gridded histogram data")
    source_name = scope.name
    entry = DatasetEntry(
        name=_unique_dataset_name(
            name or f"{source_name} composite", group.dataset_names
        ),
        data=data,
        kind="project_artifact",
        data_type=(
            "powder_inelastic" if len(data.axes) == 2 else "single_crystal_inelastic"
        ),
        metadata={
            "materialized_from_composite": {
                "source_group": source_name,
                "config": copy.deepcopy(data_group_composite_config(scope)),
            },
            "import_status": "loaded",
        },
    )
    artifact_path = replace_dataset_artifact(
        project_path, entry.id, dataset_artifact_bytes(data)
    )
    entry.metadata.update(
        {
            "source_file": artifact_path,
            "project_artifact_path": artifact_path,
            "_project_path": str(Path(project_path)),
        }
    )
    entry.replace_data(data, source_backed=True)
    destination = next(
        (subgroup for subgroup in group.subgroups if subgroup.name == "Materialized data"),
        None,
    )
    if destination is None:
        # Keep a materialized output from becoming an input to the same
        # composite on its next refresh. It remains available as a background,
        # analysis input, or for moving into a downstream collection.
        destination = DatasetGroup("Materialized data", enabled=False)
        group.subgroups.append(destination)
    destination.datasets.append(entry)
    return entry


def _scaled_error_for_weight(dataset: DatasetEntry, errors: np.ndarray) -> np.ndarray:
    return np.asarray(errors, dtype=float) * abs(float(dataset.scale_factor))


def _dataset_statistical_weight(dataset: DatasetEntry, size: int) -> np.ndarray:
    weight = float(dataset.fit_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        return np.zeros(size, dtype=float)
    return np.full(size, weight, dtype=float)


def _composite_rebin_bounds(config: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    return (
        [axis["lower"] for axis in axes_config],
        [axis["upper"] for axis in axes_config],
        [axis["num_bins"] for axis in axes_config],
    )


def _composite_mdhisto_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    datasets: list[DatasetEntry] | None = None,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> MDHistoData:
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "loading, masking, and normalizing source datasets",
            }
        )
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    coverage_inputs: list[tuple[MDHistoData, np.ndarray]] = []
    first_data: MDHistoData | None = None
    weighting_mode = _rebin_mean_weighting(config)
    normalization_weighted = False
    for dataset in datasets if datasets is not None else _composite_candidates(group):
        data = (
            dataset.data
            if datasets is not None
            else _source_data_for_group_composite(
                group,
                dataset,
                include_source_masks=include_source_masks,
            )
        )
        if not isinstance(data, MDHistoData):
            continue
        if any("metadata_dimension" in axis.metadata for axis in data.axes):
            raise ValueError("Rebin the original collection to preserve its discrete metadata dimensions.")
        if first_data is None:
            first_data = data
        source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
        coords = np.stack(source_grids, axis=-1)
        coverage_inputs.append((data, coords))
        valid = np.isfinite(data.signal) & np.isfinite(data.errors) & ~np.asarray(data.mask, dtype=bool)
        normalization_channel = data.auxiliary_channels.get(
            "normalization_denominator"
        )
        if normalization_channel is not None:
            normalization_weighted = True
            normalization_values = np.asarray(
                normalization_channel.values, dtype=float
            )
            valid &= np.isfinite(normalization_values) & (normalization_values > 0.0)
        if data.num_events is not None:
            valid &= mdhisto_measured_bins(data)
        if not np.any(valid):
            continue
        scale = float(dataset.scale_factor)
        signal = np.asarray(data.signal[valid], dtype=float) * scale
        errors = _scaled_error_for_weight(dataset, np.asarray(data.errors[valid], dtype=float))
        if normalization_channel is not None:
            weights = normalization_values[valid] * float(dataset.fit_weight)
        else:
            weights = _dataset_statistical_weight(dataset, signal.size)
        coords_parts.append(coords[valid])
        signal_parts.append(signal)
        error_parts.append(errors)
        weight_parts.append(weights)
    if first_data is None or not signal_parts:
        raise ValueError("no valid data points remain before compositing")
    coords_all = np.concatenate(coords_parts, axis=0)
    signal_all = np.concatenate(signal_parts)
    errors_all = np.concatenate(error_parts)
    weights_all = np.concatenate(weight_parts)
    output_basis = (
        _validate_mdhisto_rebin_basis(axes_config, 4)
        if len(axes_config) == 4
        else None
    )
    limit_coordinates = (
        coords_all @ np.linalg.inv(output_basis)
        if output_basis is not None
        else coords_all
    )
    axes_config = _resolve_auto_rebin_axes(
        axes_config, _finite_coordinate_bounds(limit_coordinates)
    )
    axes_config = _resolve_data_driven_rebin_axes(
        config, axes_config, limit_coordinates
    )
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    result = rebin_nd(
        signal_all,
        coords_all,
        data_errs=errors_all,
        data_weights=weights_all,
        axes=output_basis,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", True)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=weighting_mode,
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None or result.bins_list is None:
        raise RuntimeError("composite rebinning did not produce binned data")
    axes = tuple(
        MDHistoAxis(
            name=str(axis_config.get("name") or source_axis.name),
            values=np.asarray(bins, dtype=float),
            units=str(axis_config.get("units") if axis_config.get("units") is not None else source_axis.units),
            kind=source_axis.kind,
            frame=source_axis.frame,
            path=source_axis.path,
            metadata={
                **dict(source_axis.metadata),
                **(
                    {"discrete_centers": axis_config["resolved_centers"]}
                    if axis_config.get("resolved_centers") is not None
                    else {}
                ),
            },
        )
        for source_axis, axis_config, bins in zip(first_data.axes, axes_config, result.bins_list, strict=True)
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_coverage",
                "message": "calculating geometric coverage for output bins",
            }
        )
    coverage = np.maximum.reduce(
        [
            _rebin_mdhisto_coverage(
                source,
                source_coords,
                config,
                axes_config,
                result.bins_list,
            )
            for source, source_coords in coverage_inputs
        ]
    )
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata = {
        "composite": True,
        "source_group": group.name,
        "source_datasets": [dataset.name for dataset in _composite_candidates(group)],
        "signal_semantics": "density",
        "signal_semantics_source": "nfit_normalized_rebin",
        "coverage_mask_count": int(np.count_nonzero(coverage_mask)),
        "rebin": {
            "lower": lower,
            "upper": upper,
            "step_size": np.asarray(result.step_size, dtype=float).tolist(),
            "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
            "bin_edges": [np.asarray(edges, dtype=float).tolist() for edges in result.bins_list],
            "vectors": [
                _rebin_axis_vector(axis_config, index, len(axes_config)).tolist()
                for index, axis_config in enumerate(axes_config)
            ],
            "fractional_axes": _rebin_fractional_axes(config, axes_config),
            "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
            "normalize": True,
            "mean_weighting": weighting_mode,
            "minimum_coverage": _rebin_minimum_coverage(config),
            "minimum_samples": _rebin_minimum_samples(config),
            "max_batch_mb": _rebin_max_batch_mb(config),
            "max_batch_bytes": _rebin_max_batch_bytes(config),
            "weighted_by_fit_weight": True,
            "weighted_by_normalization_denominator": normalization_weighted,
        },
    }
    for key in (
        "lattice_parameters",
        "oriented_lattice",
        "rlu_to_inv_angstrom_matrix",
        "ub_matrix",
        "orientation_matrix",
    ):
        if key in first_data.metadata:
            metadata[key] = copy.deepcopy(first_data.metadata[key])
    root = _composite_root(group)
    if root.lattice_parameters and "lattice_parameters" not in metadata:
        metadata["lattice_parameters"] = dict(root.lattice_parameters)
    auxiliary_channels = {
        "coverage_fraction": MDHistoChannel(
            coverage,
            label="Coverage",
            unit="fraction",
        )
    }
    if (
        normalization_weighted
        and weighting_mode == "uniform"
        and result._normalization is not None
    ):
        auxiliary_channels["normalization_denominator"] = MDHistoChannel(
            np.asarray(result._normalization, dtype=float),
            label="Combined detector-trajectory normalization",
            unit="arbitrary normalization units",
        )
    return MDHistoData(
        axes=axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=first_data.coordinate_system,
        visual_normalization=first_data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _composite_point_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    datasets: list[DatasetEntry] | None = None,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> MDHistoData:
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    first_data: PointData4D | None = None
    normalization_weighted = False
    for dataset in datasets if datasets is not None else _composite_candidates(group):
        data = dataset.data if datasets is not None else _source_data_for_group_composite(
            group,
            dataset,
            include_source_masks=include_source_masks,
        )
        if not isinstance(data, PointData4D):
            continue
        if first_data is None:
            first_data = data
        source = data.valid(require_positive_sigma=False)
        if source.size == 0:
            continue
        coords_parts.append(np.column_stack(source.coordinates()))
        scale = float(dataset.scale_factor)
        signal = np.asarray(source.intensity, dtype=float) * scale
        signal_parts.append(signal)
        error_parts.append(_scaled_error_for_weight(dataset, source.sigma))
        weights = _dataset_statistical_weight(dataset, source.size)
        if source.normalization_denominator is not None:
            normalization_weighted = True
            weights *= np.asarray(source.normalization_denominator, dtype=float)
        weight_parts.append(weights)
    if first_data is None or not signal_parts:
        raise ValueError("no valid data points remain before compositing")
    return _point_data_histogram(
        np.concatenate(coords_parts, axis=0),
        np.concatenate(signal_parts),
        np.concatenate(error_parts),
        config,
        data_weights=np.concatenate(weight_parts),
        source_metadata=first_data.metadata,
        coordinate_system=first_data.metadata.get("coordinate_system"),
        visual_normalization=first_data.metadata.get("visual_normalization"),
        metadata_updates={
            "composite": True,
            "source_group": group.name,
            "source_datasets": [
                dataset.name for dataset in _composite_candidates(group)
            ],
            "weighted_by_fit_weight": True,
            "weighted_by_normalization_denominator": normalization_weighted,
        },
        progress_callback=progress_callback,
    )


def _composite_point_list_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> PointListData:
    datasets = _composite_candidates(group)
    prepared = [
        _source_data_for_group_composite(
            group,
            dataset,
            include_source_masks=include_source_masks,
        )
        for dataset in datasets
    ]
    point_lists = [data for data in prepared if isinstance(data, PointListData)]
    if not point_lists:
        raise ValueError("no point-list datasets are available to composite")
    coordinate_names = list(point_lists[0].coordinate_names)
    channel_label = point_lists[0].channel_labels[0] if point_lists[0].channel_labels else None
    if not coordinate_names or channel_label is None:
        raise ValueError("point-list composites require coordinates and at least one channel")
    for data in point_lists[1:]:
        if list(data.coordinate_names) != coordinate_names or channel_label not in data.channel_labels:
            raise ValueError("point-list composites require matching coordinates and channel labels")
    lower, upper, num_bins = _composite_rebin_bounds(config)
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for dataset, data in zip(datasets, point_lists, strict=False):
        coords = np.column_stack([data.column(name) for name in coordinate_names])
        finite = np.all(np.isfinite(coords), axis=1)
        values = np.asarray(data.channel_values(channel_label), dtype=float)
        errors = data.channel_errors(channel_label)
        if errors is None:
            errors = np.ones(values.shape, dtype=float)
        valid = finite & np.isfinite(values) & np.isfinite(errors)
        if not np.any(valid):
            continue
        scale = float(dataset.scale_factor)
        coords_parts.append(coords[valid])
        signal_parts.append(values[valid] * scale)
        error_parts.append(_scaled_error_for_weight(dataset, np.asarray(errors[valid], dtype=float)))
        weight_parts.append(_dataset_statistical_weight(dataset, int(np.count_nonzero(valid))))
    if not signal_parts:
        raise ValueError("no valid point-list rows remain before compositing")
    coordinates_all = np.concatenate(coords_parts, axis=0)
    axes_config = _resolve_auto_rebin_axes(
        axes_config, _finite_coordinate_bounds(coordinates_all)
    )
    axes_config = _resolve_data_driven_rebin_axes(
        config, axes_config, coordinates_all
    )
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    result = rebin_nd(
        np.concatenate(signal_parts),
        coordinates_all,
        data_errs=np.concatenate(error_parts),
        data_weights=np.concatenate(weight_parts),
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", True)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None or result.bin_centers_list is None:
        raise RuntimeError("composite rebinning did not produce binned data")
    occupied = np.isfinite(result.binned_data) & (result.n_samples > 0.0)
    center_grids = np.meshgrid(
        *(
            np.asarray(axis.get("resolved_centers", centers), dtype=float)
            for axis, centers in zip(
                axes_config, result.bin_centers_list, strict=True
            )
        ),
        indexing="ij",
    )
    value_name = point_lists[0].channel(channel_label)["value"]
    error_name = point_lists[0].channel(channel_label).get("error") or f"{value_name}_error"
    columns = {name: grid[occupied] for name, grid in zip(coordinate_names, center_grids, strict=True)}
    columns[value_name] = np.asarray(result.binned_data, dtype=float)[occupied]
    columns[error_name] = np.asarray(result.binned_data_errs, dtype=float)[occupied]
    columns["n_samples"] = np.asarray(result.n_samples, dtype=float)[occupied]
    return PointListData(
        columns=columns,
        units={**{name: point_lists[0].unit(name) for name in coordinate_names}, value_name: point_lists[0].unit(value_name), error_name: point_lists[0].unit(error_name)},
        coordinate_names=coordinate_names,
        channels=[{"label": channel_label, "value": value_name, "error": error_name}],
        metadata={
            "composite": True,
            "source_group": group.name,
            "source_datasets": [dataset.name for dataset in datasets],
            "rebin": {
                "bin_edges": [
                    np.asarray(edges, dtype=float).tolist()
                    for edges in (result.bins_list or [])
                ],
                "fractional_axes": _rebin_fractional_axes(config, axes_config),
                "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
                "minimum_samples": _rebin_minimum_samples(config),
            },
        },
        quantity_types={
            name: point_lists[0].quantity_type(name)
            for name in columns
            if name in point_lists[0].columns
        },
    )


def dataset_for_slice_viewer(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D | None:
    """Return a viewer-ready dataset, loading from source metadata if needed.

    ``extra_masks`` are masks inherited from ancestor dataset groups; they are
    applied ahead of the dataset's own masks.
    """

    result = _viewer_data_before_scale(
        dataset,
        extra_masks=extra_masks,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
    )
    if result is None:
        return None
    scaled = _apply_dataset_scale(dataset, result)
    prepared = _apply_spectral_channel_view(dataset, scaled)
    return _with_viewer_dataset_metadata(dataset, prepared)


def _apply_spectral_channel_view(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Apply paired INS channels, falling back to the legacy kinematic path."""

    config = dataset.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY)
    if (
        isinstance(data, MDHistoData)
        and dataset.data_type in {"single_crystal_inelastic", "powder_inelastic"}
        and isinstance(config, dict)
    ):
        temperature = dataset.parameters.get("temperature", data.metadata.get("temperature"))
        from .metadata_dimensions import metadata_temperature_grid

        temperatures = metadata_temperature_grid(data)
        return with_paired_spectral_channels(
            data,
            config,
            temperature_K=(
                temperatures if temperatures is not None
                else None if temperature in (None, "") else float(temperature)
            ),
        )
    return _apply_kinematic_normalization_to_view(dataset, data)


def _with_viewer_dataset_metadata(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    metadata = dict(getattr(data, "metadata", {}) or {})
    metadata["nfit_data_type"] = dataset.data_type
    metadata["nfit_dataset_kind"] = dataset.kind
    if isinstance(data, MDHistoData):
        return replace(data, metadata=metadata)
    if isinstance(data, PointListData):
        return PointListData(
            columns={name: np.array(values, dtype=float) for name, values in data.columns.items()},
            units=dict(data.units),
            coordinate_names=list(data.coordinate_names),
            channels=[dict(channel) for channel in data.channels],
            metadata=metadata,
            quantity_types=dict(data.quantity_types),
        )
    if isinstance(data, PointData4D):
        return data.with_updates(metadata=metadata)
    return data


def _kinematic_energy_metadata(
    dataset: DatasetEntry,
    data_metadata: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Find scalar incident/final energies recorded on a dataset or its data."""

    sources = (dataset.parameters, dataset.metadata, data_metadata or {})

    def value_for(names: tuple[str, ...]) -> float | None:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for name in names:
                try:
                    value = float(source[name])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return value
        return None

    return (
        value_for(("incident_energy", "incident_energy_meV", "Ei", "ei")),
        value_for(("final_energy", "final_energy_meV", "Ef", "ef")),
    )


def _kinematic_kf_ki_factor(
    energy_transfer_meV: Any,
    *,
    incident_energy_meV: float | None,
    final_energy_meV: float | None,
) -> np.ndarray | None:
    """Return ``k_f/k_i`` for ``E = E_i - E_f``, if one energy is known."""

    if incident_energy_meV is None and final_energy_meV is None:
        return None
    energy = np.asarray(energy_transfer_meV, dtype=float)
    if incident_energy_meV is not None:
        ratio_sq = (float(incident_energy_meV) - energy) / float(incident_energy_meV)
    else:
        ratio_sq = float(final_energy_meV) / (float(final_energy_meV) + energy)
    factor = np.full(energy.shape, np.nan, dtype=float)
    np.sqrt(ratio_sq, out=factor, where=np.isfinite(ratio_sq) & (ratio_sq >= 0.0))
    return factor


def _apply_kinematic_normalization_to_view(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Normalize binned data to the cross-section ``k_f/k_i`` convention."""

    if bool(dataset.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)):
        return data
    if isinstance(data, PointData4D):
        return _apply_kinematic_normalization_to_points(dataset, data)
    if not isinstance(data, MDHistoData):
        return data
    energy_dim = next(
        (index for index, axis in enumerate(data.axes) if axis.kind == "energy"),
        None,
    )
    if energy_dim is None:
        return data
    incident, final = _kinematic_energy_metadata(dataset, data.metadata)
    factor_1d = _kinematic_kf_ki_factor(
        data.axes[energy_dim].centers,
        incident_energy_meV=incident,
        final_energy_meV=final,
    )
    if factor_1d is None:
        return data
    shape = [1] * data.signal.ndim
    shape[energy_dim] = factor_1d.size
    factor = factor_1d.reshape(shape)
    metadata = dict(data.metadata)
    metadata["nfit_kinematic_kf_ki_normalized"] = True
    metadata["nfit_kinematic_kf_ki_source"] = "Ei" if incident is not None else "Ef"
    return replace(
        data,
        signal=np.asarray(data.signal, dtype=float) * factor,
        errors=np.asarray(data.errors, dtype=float) * np.abs(factor),
        metadata=metadata,
    )


def _apply_kinematic_normalization_to_points(
    dataset: DatasetEntry,
    points: PointData4D,
) -> PointData4D:
    """Return fit points in the selected kinematic convention."""

    if bool(dataset.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)):
        return points
    if points.metadata.get("nfit_kinematic_kf_ki_normalized"):
        return points
    incident, final = _kinematic_energy_metadata(dataset, points.metadata)
    factor = _kinematic_kf_ki_factor(
        points.E,
        incident_energy_meV=incident,
        final_energy_meV=final,
    )
    if factor is None:
        return points
    metadata = dict(points.metadata)
    metadata["nfit_kinematic_kf_ki_normalized"] = True
    metadata["nfit_kinematic_kf_ki_source"] = "Ei" if incident is not None else "Ef"
    return points.with_updates(
        intensity=np.asarray(points.intensity, dtype=float) * factor,
        sigma=np.asarray(points.sigma, dtype=float) * np.abs(factor),
        metadata=metadata,
    )


def _mask_signature(masks: list[MaskSpec] | None) -> list[Any]:
    return [
        [m.type, bool(m.enabled), bool(m.invert), bool(m.additive),
         json.dumps(m.parameters, sort_keys=True, default=str)]
        for m in (masks or [])
    ]


def _viewer_view_signature(
    dataset: DatasetEntry, extra_masks: list[MaskSpec] | None
) -> str:
    rebin = (
        json.dumps(dataset_rebin_config(dataset), sort_keys=True, default=str)
        if dataset_rebin_enabled(dataset)
        else None
    )
    payload = [
        dataset.data_cache_token,
        dataset.data_type,
        dataset.kind,
        rebin,
        _mask_signature(getattr(dataset, "masks", None)),
        _mask_signature(extra_masks),
        [
            [
                background.source_dataset_id,
                bool(background.enabled),
                float(background.scale),
                background.interpolation,
                (
                    background.source_entry.data_cache_token
                    if background.source_entry is not None
                    else None
                ),
            ]
            for background in dataset.backgrounds
        ],
        _derived_recipe_dependency_signature(dataset),
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def _derived_recipe_dependency_signature(dataset: DatasetEntry) -> Any:
    resolved = _derived_analysis_for_dataset(dataset)
    if resolved is None:
        return None
    group, analysis = resolved
    dependencies = []
    for source_id in analysis.input_dataset_ids:
        scope = _derived_source_scope(group, source_id)
        if scope is not None:
            dependencies.append([source_id, _composite_cache_signature(scope)])
            continue
        source = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        dependencies.append(
            [
                source_id,
                None if source is None else source.data_cache_token,
                None if source is None else float(source.scale_factor),
                None if source is None else bool(source.enabled),
            ]
        )
    return [
        analysis.type,
        json.dumps(analysis.parameters, sort_keys=True, default=str),
        dependencies,
    ]


def _viewer_data_before_scale(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | None:
    key = dataset.id
    signature = _viewer_view_signature(dataset, extra_masks)
    deferred_masks = _should_defer_dataset_masks(dataset, force_masks=force_masks)
    cached = _VIEWER_VIEW_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        _VIEWER_VIEW_CACHE.move_to_end(key)
        if force_masks:
            dataset_mask_application_config(dataset)["stale"] = False
        return cached[1]
    if cached is not None and deferred_masks:
        return cached[1]
    if cached is not None and _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
        return cached[1]
    result = _viewer_data_before_scale_uncached(
        dataset,
        extra_masks=extra_masks,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
    )
    if isinstance(result, MDHistoData) and dataset.backgrounds:
        result = _apply_dataset_backgrounds(dataset, result)
    if result is not None and not deferred_masks:
        # Recompute the signature: the uncached path may have lazily loaded the
        # data and incremented the dataset revision.
        signature = _viewer_view_signature(dataset, extra_masks)
        _lru_store(
            _VIEWER_VIEW_CACHE,
            key,
            (signature, result),
            _VIEWER_VIEW_CACHE_LIMIT,
            _VIEWER_VIEW_CACHE_MAX_BYTES,
        )
        dataset_mask_application_config(dataset)["stale"] = False
    return result


def _apply_dataset_backgrounds(
    dataset: DatasetEntry,
    data: MDHistoData,
) -> MDHistoData:
    result = data
    for background in dataset.backgrounds:
        if not background.enabled:
            continue
        source = background.source_entry
        if source is None:
            raise ValueError(
                f"background {background.name!r} refers to a missing dataset"
            )
        source_data = _viewer_data_before_scale_uncached(
            source,
            force_rebin=True,
            force_masks=True,
        )
        if not isinstance(source_data, MDHistoData):
            raise TypeError(
                f"background {background.name!r} must refer to gridded histogram data"
            )
        result = subtract_background(
            result,
            source_data,
            scale=background.scale,
            interpolation=background.interpolation,
        )
    return result


def _viewer_data_before_scale_uncached(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | None:
    if isinstance(dataset.metadata.get(DERIVED_RECIPE_KEY), dict):
        derived = derived_analysis_dataset_data(
            dataset,
            progress_callback=progress_callback,
        )
        dataset_rebin_config(dataset)["stale"] = False
        if isinstance(derived, MDHistoData):
            return _mdhisto_with_nfit_masks(
                dataset,
                data=derived,
                extra_masks=extra_masks,
            )
        return derived
    loaded = _ensure_dataset_data_loaded(dataset)
    if data_type_container(dataset.data_type) == "point_list" or isinstance(
        loaded, PointListData
    ):
        if not isinstance(loaded, PointListData):
            return None
        if dataset_rebin_enabled(dataset):
            if _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
                return prepared_point_list_data(dataset)
            return rebinned_dataset_data(dataset, progress_callback=progress_callback)
        return prepared_point_list_data(dataset)
    if isinstance(loaded, MDHistoData):
        if _should_defer_dataset_masks(dataset, force_masks=force_masks):
            return _mdhisto_without_nfit_masks(loaded)
        if dataset_rebin_enabled(dataset):
            if _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
                return _mdhisto_with_nfit_masks(dataset, data=loaded, extra_masks=extra_masks)
            return rebinned_dataset_data(dataset, extra_masks=extra_masks, progress_callback=progress_callback)
        return _mdhisto_with_nfit_masks(dataset, data=loaded, extra_masks=extra_masks)
    if isinstance(loaded, PointData4D):
        if dataset_rebin_enabled(dataset):
            return rebinned_dataset_data(
                dataset, extra_masks=extra_masks, progress_callback=progress_callback
            )
        return _point_data_with_nfit_masks(dataset, loaded, extra_masks=extra_masks)
    return None


def _mdhisto_without_nfit_masks(data: MDHistoData) -> MDHistoData:
    """Return a cheap file-mask-only view while manual nfit masks are pending."""

    file_mask = np.asarray(data.mask, dtype=bool)
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask
    metadata["nfit_mask"] = np.zeros(0, dtype=bool)
    metadata["file_mask_count"] = int(np.count_nonzero(file_mask))
    metadata["nfit_mask_count"] = 0
    metadata["combined_mask_count"] = metadata["file_mask_count"]
    metadata["mask_application_pending"] = True
    return replace(data, mask=file_mask, metadata=metadata)


def _apply_dataset_scale(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData | PointData4D,
) -> MDHistoData | PointListData | PointData4D:
    """Multiply a dataset's signal and errors by its scale factor (both channels)."""

    scale = float(getattr(dataset, "scale_factor", 1.0) or 1.0)
    if scale == 1.0:
        return data
    if isinstance(data, MDHistoData):
        from dataclasses import replace

        return replace(
            data,
            signal=np.asarray(data.signal, dtype=float) * scale,
            errors=np.asarray(data.errors, dtype=float) * abs(scale),
        )
    if isinstance(data, PointListData):
        columns = {name: np.array(values, dtype=float) for name, values in data.columns.items()}
        for channel in data.channels:
            value_name = channel.get("value")
            error_name = channel.get("error")
            if value_name in columns:
                columns[value_name] = columns[value_name] * scale
            if error_name in columns:
                columns[error_name] = columns[error_name] * abs(scale)
        return PointListData(
            columns=columns,
            units=dict(data.units),
            coordinate_names=list(data.coordinate_names),
            channels=[dict(channel) for channel in data.channels],
            metadata=dict(data.metadata),
            quantity_types=dict(data.quantity_types),
        )
    if isinstance(data, PointData4D):
        return data.with_updates(
            intensity=np.asarray(data.intensity, dtype=float) * scale,
            sigma=np.asarray(data.sigma, dtype=float) * abs(scale),
        )
    return data


def dataset_rebin_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return a dataset rebin configuration, creating default axis settings if needed."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.parameters[DATASET_REBIN_KEY] = config
    initialize_rebin_performance(config)
    config.setdefault("enabled", False)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    if config.get(REBIN_RESOLUTION_MODE_KEY) not in {"step", "bins"}:
        config[REBIN_RESOLUTION_MODE_KEY] = "step"
    try:
        config["max_batch_mb"] = max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        config["max_batch_mb"] = DEFAULT_REBIN_MAX_BATCH_MB
    if config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "uniform"
    config["minimum_coverage"] = _rebin_minimum_coverage(config)
    config["minimum_samples"] = _rebin_minimum_samples(config)
    config["normalize"] = True
    axes = config.get("axes")
    # Only materialize transformed coordinates when defaults are actually
    # needed. Existing axes already carry their complete saved basis.
    if isinstance(dataset.data, PointListData) and (not isinstance(axes, list) or not axes):
        default_axes = _default_rebin_axes(prepared_point_list_data(dataset))
    elif isinstance(dataset.data, PointListData):
        default_axes = []
    else:
        default_axes = _default_rebin_axes(dataset.data)
    if not isinstance(axes, list) or not axes:
        # A saved project may call this while its file-backed dataset is still
        # unloaded. Leave axes absent in that state so first data load can
        # create defaults, but never replace a saved non-empty basis.
        if default_axes:
            config["axes"] = default_axes
            if isinstance(dataset.data, MDHistoData) and len(dataset.data.axes) == 4:
                config["coordinate_basis_version"] = REBIN_COORDINATE_BASIS_VERSION
    else:
        sanitized_axes = list(axes)
        if len(axes) == len(default_axes):
            sanitized_axes = []
            for axis_config, default_axis in zip(axes, default_axes, strict=True):
                if not isinstance(axis_config, dict):
                    axis_config = {}
                for key in ("lower", "upper"):
                    auto_key = f"auto_{key}"
                    if auto_key not in axis_config:
                        axis_config[auto_key] = bool(
                            key in axis_config
                            and np.isclose(
                                float(axis_config[key]), float(default_axis[key])
                            )
                        )
                    if bool(axis_config.get(auto_key, False)):
                        axis_config.setdefault(f"{auto_key}_value", axis_config.get(key))
                if bool(default_axis.get("auto_step_size", False)) and (
                    "auto_step_size" not in axis_config
                ):
                    axis_config["auto_step_size"] = bool(
                        "step_size" in axis_config
                        and np.isclose(
                            float(axis_config["step_size"]),
                            float(default_axis["step_size"]),
                        )
                        and bool(default_axis.get("auto_step_size", False))
                    )
                if bool(default_axis.get("auto_step_size", False)) and bool(
                    axis_config.get("auto_step_size", False)
                ):
                    axis_config.setdefault(
                        "auto_step_size_value", axis_config.get("step_size")
                    )
                for key, value in default_axis.items():
                    axis_config.setdefault(key, value)
                if isinstance(dataset.data, MDHistoData) and "vector" in axis_config:
                    axis_config["variable"] = str(
                        axis_config.get("variable") or default_axis.get("variable", "")
                    )
                    axis_config["name"] = _rebin_axis_name(
                        axis_config["variable"], axis_config.get("vector", [])
                    )
                sanitized_axes.append(_sanitize_rebin_axis_config(axis_config))
            config["axes"] = sanitized_axes
        if (
            len(axes) == len(default_axes)
            and isinstance(dataset.data, MDHistoData)
            and len(dataset.data.axes) == 4
        ):
            try:
                basis_version = int(config["coordinate_basis_version"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "saved 4D rebin settings are missing a supported "
                    "coordinate_basis_version; recreate the rebin settings"
                ) from exc
            if basis_version != REBIN_COORDINATE_BASIS_VERSION:
                raise ValueError(
                    f"unsupported rebin coordinate basis version {basis_version}; "
                    "recreate the rebin settings"
                )
    if "auto_rebin" not in config:
        config["auto_rebin"] = not _dataset_rebin_is_large(dataset, config)
    _migrate_rebin_axis_modes(config)
    config.setdefault("stale", False)
    return config


def _rebin_settings_clipboard_text(config: dict[str, Any]) -> str:
    """Serialize user-editable rebin settings for the system clipboard."""

    settings = {
        key: copy.deepcopy(config[key])
        for key in REBIN_SETTINGS_KEYS
        if key in config
    }
    return json.dumps(
        {
            "schema": REBIN_SETTINGS_CLIPBOARD_SCHEMA,
            "version": REBIN_SETTINGS_CLIPBOARD_VERSION,
            "settings": settings,
        },
        indent=2,
        sort_keys=True,
    )


def _rebin_config_from_clipboard_text(
    text: str,
    target_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a pasted rebin config after validating target compatibility."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The clipboard does not contain valid nfit rebin settings.") from exc
    if not isinstance(payload, dict) or payload.get("schema") != REBIN_SETTINGS_CLIPBOARD_SCHEMA:
        raise ValueError("The clipboard does not contain nfit rebin settings.")
    if payload.get("version") != REBIN_SETTINGS_CLIPBOARD_VERSION:
        raise ValueError("The clipboard rebin-settings version is not supported.")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The clipboard rebin settings are incomplete.")
    if settings.get("metadata_dimensions") and "metadata_dimensions" not in target_config:
        raise ValueError("Metadata rebin settings must be pasted into a source collection's composite panel.")
    if "metadata_dimensions" in settings:
        from .metadata_dimensions import MetadataDimension

        try:
            for recipe in settings["metadata_dimensions"]:
                MetadataDimension(**recipe)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid metadata rebin settings: {exc}") from exc
    source_axes = settings.get("axes")
    target_axes = target_config.get("axes")
    if not isinstance(source_axes, list) or not source_axes:
        raise ValueError("The copied rebin settings do not contain any axes.")
    if isinstance(target_axes, list) and target_axes and len(source_axes) != len(target_axes):
        raise ValueError(
            f"The copied settings have {len(source_axes)} axes, but this rebin panel has "
            f"{len(target_axes)} axes."
        )
    for index, source_axis in enumerate(source_axes):
        if not isinstance(source_axis, dict):
            raise ValueError(f"Copied rebin axis {index + 1} is invalid.")
        source_vector = source_axis.get("vector")
        target_vector = (
            target_axes[index].get("vector")
            if isinstance(target_axes, list)
            and index < len(target_axes)
            and isinstance(target_axes[index], dict)
            else None
        )
        if isinstance(source_vector, list) and isinstance(target_vector, list):
            if len(source_vector) != len(target_vector):
                raise ValueError(
                    f"Copied rebin axis {index + 1} uses a {len(source_vector)}-component "
                    f"coordinate vector, but this panel expects {len(target_vector)} components."
                )
    pasted = copy.deepcopy(target_config)
    for key in REBIN_SETTINGS_KEYS:
        if key in settings:
            pasted[key] = copy.deepcopy(settings[key])
    pasted["normalize"] = True
    pasted["stale"] = True
    return pasted


def dataset_mask_application_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return persistent automatic/manual mask materialization settings."""

    config = dataset.metadata.get(DATASET_MASK_APPLICATION_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.metadata[DATASET_MASK_APPLICATION_KEY] = config
    if "auto_apply" not in config:
        config["auto_apply"] = not _dataset_mask_is_large(dataset)
    config.setdefault("stale", False)
    return config


def _dataset_mask_point_count(dataset: DatasetEntry) -> int:
    data = dataset.data
    if isinstance(data, MDHistoData):
        return int(np.asarray(data.signal).size)
    if isinstance(data, PointData4D):
        return int(data.size)
    if isinstance(data, PointListData):
        return int(data.size)
    return 0


def _dataset_mask_is_large(dataset: DatasetEntry) -> bool:
    return _dataset_mask_point_count(dataset) > MASK_AUTO_MAX_POINTS


def _dataset_mask_auto_enabled(dataset: DatasetEntry) -> bool:
    return bool(dataset_mask_application_config(dataset).get("auto_apply", True))


def _dataset_mask_is_stale(dataset: DatasetEntry) -> bool:
    config = dataset.metadata.get(DATASET_MASK_APPLICATION_KEY)
    return bool(isinstance(config, dict) and config.get("stale", False))


def _should_defer_dataset_masks(dataset: DatasetEntry, *, force_masks: bool) -> bool:
    # Manual mode only materializes masks at explicit synchronization points.
    # An exact cached view is still returned before this policy is consulted,
    # so a current manual result remains cheap to reuse. This also protects a
    # freshly loaded project, where no process-local masked cache exists yet.
    return bool(not force_masks and not _dataset_mask_auto_enabled(dataset))


def _dataset_mask_status_text(dataset: DatasetEntry) -> str:
    config = dataset_mask_application_config(dataset)
    size_note = "large dataset" if _dataset_mask_is_large(dataset) else "small dataset"
    if bool(config.get("auto_apply", True)):
        if bool(config.get("stale", False)):
            return f"Automatic mask application is on ({size_note}); pending masks apply on the next refresh."
        return f"Automatic mask application is on ({size_note}); applied masks are current."
    if bool(config.get("stale", False)) or dataset.id not in _VIEWER_VIEW_CACHE:
        return "Manual mask application pending; press Apply masks now or start a fit/open the data viewer."
    return f"Manual mask application is on ({size_note}); applied masks are current."


def dataset_rebin_enabled(dataset: DatasetEntry) -> bool:
    """Return whether this dataset should use its rebinned representation."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _dataset_rebin_source_points(dataset: DatasetEntry) -> int:
    return _dataset_data_point_count(dataset)


def _dataset_rebin_output_bins(config: dict[str, Any]) -> int:
    total = 1
    axes = config.get("axes")
    if not isinstance(axes, list) or not axes:
        return 0
    for axis in axes:
        if not isinstance(axis, dict):
            return 0
        total *= max(int(axis.get("num_bins", 1) or 1), 1)
    return int(total)


def _dataset_rebin_estimated_contributions(dataset: DatasetEntry, config: dict[str, Any]) -> int:
    source_points = _dataset_rebin_source_points(dataset)
    axes = config.get("axes", []) or [{}]
    multiplier = 2 ** sum(_rebin_fractional_axes(config, axes))
    return int(source_points * multiplier * _rebin_symmetry_count(config, _dataset_lattice_parameters(dataset)))


def _dataset_lattice_parameters(dataset: DatasetEntry) -> dict[str, Any] | None:
    data = dataset.data
    metadata = getattr(data, "metadata", None)
    return metadata.get("lattice_parameters") if isinstance(metadata, dict) and isinstance(metadata.get("lattice_parameters"), dict) else None


def _rebin_symmetry_operations(
    config: dict[str, Any],
    lattice_parameters: dict[str, Any] | None = None,
) -> tuple:
    payload = config.get("symmetry")
    spec = symmetry_spec_from_config(payload)
    stored_lattice = payload.get("lattice_parameters") if isinstance(payload, dict) else None
    lattice = lattice_parameters if lattice_parameters is not None else stored_lattice
    return resolve_symmetry(spec, lattice_parameters=lattice)


def _rebin_symmetry_count(config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None) -> int:
    try:
        return len(_rebin_symmetry_operations(config, lattice_parameters))
    except (ImportError, ValueError):
        return 1


def _rebin_symmetry_matrices(
    config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None
) -> tuple[np.ndarray, ...] | None:
    operations = _rebin_symmetry_operations(config, lattice_parameters)
    if len(operations) == 1 and np.allclose(operations[0].matrix_hkl, np.eye(3)):
        return None
    return tuple(np.asarray(operation.matrix_hkl, dtype=float) for operation in operations)


def _rebin_symmetry_metadata(
    config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    spec = symmetry_spec_from_config(config.get("symmetry"))
    if not spec.enabled:
        return None
    operations = _rebin_symmetry_operations(config, lattice_parameters)
    return {
        "mode": spec.mode,
        "expression": spec.expression,
        "operation_count": len(operations),
        "operations_hkl": [np.asarray(operation.matrix_hkl, dtype=float).tolist() for operation in operations],
        "labels": [operation.label for operation in operations],
        "energy_unchanged": True,
    }


def _dataset_rebin_is_large(dataset: DatasetEntry, config: dict[str, Any]) -> bool:
    return (
        _dataset_rebin_estimated_contributions(dataset, config) > REBIN_AUTO_MAX_CONTRIBUTIONS
        or _dataset_rebin_output_bins(config) > REBIN_AUTO_MAX_OUTPUT_BINS
    )


def _dataset_rebin_status_text(dataset: DatasetEntry, config: dict[str, Any]) -> str:
    if not bool(config.get("enabled", False)):
        return "Rebinning is disabled."
    size_note = "large dataset" if _dataset_rebin_is_large(dataset, config) else "small dataset"
    if bool(config.get("auto_rebin", True)):
        if bool(config.get("stale", False)):
            return f"Automatic rebinning is on ({size_note}); pending edits will recompute on the next refresh."
        return f"Automatic rebinning is on ({size_note}); edits recompute the cached rebin."
    if bool(config.get("stale", False)):
        return "Manual rebin pending; press Rebin now or start a fit/view/export operation to compute it."
    return f"Manual rebinning is on ({size_note}); cached rebin is current."


def _dataset_rebin_auto_enabled(dataset: DatasetEntry) -> bool:
    config = dataset_rebin_config(dataset)
    return bool(config.get("auto_rebin", True))


def _dataset_rebin_is_stale(dataset: DatasetEntry) -> bool:
    config = dataset.parameters.get(DATASET_REBIN_KEY)
    return bool(isinstance(config, dict) and config.get("stale", False))


def _should_defer_dataset_rebin(dataset: DatasetEntry, *, force_rebin: bool) -> bool:
    return bool(
        dataset_rebin_enabled(dataset)
        and not force_rebin
        and _dataset_rebin_is_stale(dataset)
        and not _dataset_rebin_auto_enabled(dataset)
    )


def rebinned_dataset_data(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Return a rebinned copy using the saved configuration and worker ceiling."""
    from ._parallel import thread_budget

    config = dataset_rebin_config(dataset)
    with thread_budget(config.get("workers")):
        return _rebinned_dataset_data(dataset, extra_masks=extra_masks, progress_callback=progress_callback)


def _rebinned_dataset_data(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Return a rebinned copy of a supported dataset according to its configuration."""

    config = dataset_rebin_config(dataset)
    if isinstance(dataset.data, PointListData):
        result = _rebin_point_list_data(dataset, config)
        config["stale"] = False
        return result
    if isinstance(dataset.data, PointData4D):
        result = _rebin_point_data(
            _point_data_with_nfit_masks(dataset, dataset.data, extra_masks=extra_masks),
            config,
            progress_callback=progress_callback,
        )
        config["stale"] = False
        return result
    if not isinstance(dataset.data, MDHistoData):
        return dataset.data
    masked = _mdhisto_with_nfit_masks(dataset, data=dataset.data, extra_masks=extra_masks)
    result = _with_rebinned_mask_metadata(
        _rebin_mdhisto_data(masked, config, progress_callback=progress_callback),
        masked,
    )
    config["stale"] = False
    return result


def _with_rebinned_mask_metadata(rebinned: MDHistoData, source: MDHistoData) -> MDHistoData:
    """Annotate a rebinned MDHisto result whose source masks were applied up front."""

    metadata = dict(rebinned.metadata)
    rebin_metadata = dict(metadata.get("rebin", {}))
    for key in ("file_mask_count", "nfit_mask_count", "combined_mask_count"):
        if key in source.metadata:
            rebin_metadata[f"source_{key}"] = int(source.metadata[key])
    metadata["rebin"] = rebin_metadata
    output_mask = np.asarray(rebinned.mask, dtype=bool)
    metadata["file_mask"] = np.zeros(rebinned.shape, dtype=bool)
    metadata["nfit_mask"] = np.zeros(rebinned.shape, dtype=bool)
    metadata["file_mask_count"] = 0
    metadata["nfit_mask_count"] = 0
    metadata["combined_mask_count"] = int(np.count_nonzero(output_mask))
    return replace(rebinned, metadata=metadata)


def _rebin_point_list_data(dataset: DatasetEntry, config: dict[str, Any]) -> PointListData:
    """Rebin transformed point-list data over its coordinates into a histogram."""

    prepared = prepared_point_list_data(dataset)
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coordinate_names = [axis.get("name") for axis in axes_config if axis.get("name") in prepared.columns]
    if not coordinate_names:
        coordinate_names = list(prepared.coordinate_names)
    selected_axes = axes_config[: len(coordinate_names)]
    symmetry = _rebin_symmetry_matrices(
        config, prepared.metadata.get("lattice_parameters")
    )
    coordinates = np.column_stack(
        [prepared.columns[name] for name in coordinate_names]
    )
    data_bounds = (
        _symmetry_projected_coordinate_bounds(
            coordinates, symmetry, np.eye(len(coordinate_names))
        )
        if symmetry is not None
        else _finite_coordinate_bounds(coordinates)
    )
    selected_axes = _resolve_auto_rebin_axes(selected_axes, data_bounds)
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        coordinates,
        physical_coordinates=coordinates,
        symmetry=symmetry,
        output_basis=np.eye(len(coordinate_names)),
    )
    selected_axes = _resolve_data_driven_rebin_axes(
        config, selected_axes, mode_coordinates
    )
    lower = [axis["lower"] for axis in selected_axes] or None
    upper = [axis["upper"] for axis in selected_axes] or None
    result = prepared.rebin_to_histogram(
        coordinate_names,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, selected_axes),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, selected_axes),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        symmetry_operations=symmetry,
    )
    metadata = dict(result.metadata)
    symmetry_metadata = _rebin_symmetry_metadata(config, prepared.metadata.get("lattice_parameters"))
    if symmetry_metadata is not None:
        metadata.setdefault("rebin", {})["symmetry"] = symmetry_metadata
    return result.with_updates(metadata=metadata)


def _rebin_mean_weighting(config: dict[str, Any]) -> str:
    value = config.get("mean_weighting")
    if value == "normalization":
        return "uniform"
    return str(value) if value in {"inverse_variance", "uniform"} else "uniform"


def _rebin_axis_mode(
    config: Mapping[str, Any], axis: Mapping[str, Any]
) -> str:
    """Return one axis's grid/assignment mode, including legacy migration."""

    mode = str(axis.get("mode", "")).casefold()
    if mode in REBIN_AXIS_MODES:
        return mode
    if axis.get("bin_edges") is not None:
        return "edges"
    if axis.get("fractional") is False:
        return "discrete"
    return "bins" if config.get(REBIN_RESOLUTION_MODE_KEY) == "bins" else "step"


def _migrate_rebin_axis_modes(config: dict[str, Any]) -> None:
    """Persist the per-axis mode representation used by current projects."""

    axes = config.get("axes")
    if not isinstance(axes, list):
        return
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        axis["mode"] = _rebin_axis_mode(config, axis)
        axis.pop("fractional", None)
    config.pop("fractional", None)


def _rebin_fractional_axes(
    config: Mapping[str, Any], axes_config: Sequence[Mapping[str, Any]]
) -> list[bool]:
    """Return assignment behavior implied by each axis's mode."""

    return [
        _rebin_axis_mode(config, axis) in {"step", "bins", "edges"}
        for axis in axes_config
    ]


def _rebin_minimum_coverage(config: dict[str, Any]) -> float:
    try:
        value = float(config.get("minimum_coverage", DEFAULT_MINIMUM_COVERAGE))
    except (TypeError, ValueError):
        return DEFAULT_MINIMUM_COVERAGE
    return float(np.clip(value, 0.0, 1.0))


def _rebin_minimum_samples(config: dict[str, Any]) -> float:
    """Return the minimum effective sample contribution for an output bin."""

    try:
        value = float(config.get("minimum_samples", DEFAULT_MINIMUM_SAMPLES))
    except (TypeError, ValueError):
        return DEFAULT_MINIMUM_SAMPLES
    if not np.isfinite(value) or value < 0.0:
        return DEFAULT_MINIMUM_SAMPLES
    return value


def _rebin_max_batch_mb(config: dict[str, Any]) -> int:
    try:
        return max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        return DEFAULT_REBIN_MAX_BATCH_MB


def _rebin_max_batch_bytes(config: dict[str, Any]) -> int:
    return _rebin_max_batch_mb(config) * 1024 * 1024


def _rebin_resolution_mode(config: dict[str, Any]) -> str:
    """Return the active resolution control mode for a rebin panel."""

    return "bins" if config.get(REBIN_RESOLUTION_MODE_KEY) == "bins" else "step"


def _rebin_axis_bound_is_auto(axis: Mapping[str, Any], key: str) -> bool:
    """Return whether a saved bound still matches its last automatic value."""

    if key not in {"lower", "upper"} or not bool(axis.get(f"auto_{key}", False)):
        return False
    try:
        value = float(axis[key])
        automatic = float(axis.get(f"auto_{key}_value", value))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isclose(value, automatic))


def _resolve_auto_rebin_axes(
    axes_config: Sequence[dict[str, Any]],
    data_bounds: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Resolve blank limits and align uniform bins so zero is a bin center."""

    if len(axes_config) != len(data_bounds):
        raise ValueError("automatic rebin bounds must match the coordinate dimensions")
    resolved: list[dict[str, Any]] = []
    for axis_config, (data_lower, data_upper) in zip(
        axes_config, data_bounds, strict=True
    ):
        axis = _sanitize_rebin_axis_config(dict(axis_config))
        if axis.get("bin_edges") is not None:
            resolved.append(axis)
            continue
        auto_lower = _rebin_axis_bound_is_auto(axis, "lower")
        auto_upper = _rebin_axis_bound_is_auto(axis, "upper")
        if not (auto_lower or auto_upper):
            resolved.append(axis)
            continue
        step = float(axis.get("step_size", 0.0) or 0.0)
        auto_step = bool(axis.get("auto_step_size", False)) and np.isclose(
            step,
            float(axis.get("auto_step_size_value", step)),
        )
        if auto_step:
            target_lower = float(data_lower) if auto_lower else float(axis["lower"])
            target_upper = float(data_upper) if auto_upper else float(axis["upper"])
            width = target_upper - target_lower
            if np.isfinite(width) and width > 0.0:
                step = width / max(int(axis.get("num_bins", 1)) - 1, 1)
        if not np.isfinite(step) or step <= 0.0:
            width = float(data_upper) - float(data_lower)
            step = width / max(int(axis.get("num_bins", 1)) - 1, 1)
        if not np.isfinite(step) or step <= 0.0:
            step = 1.0
        # Uniform edges at (n + 1/2)*step put an integer multiple of the
        # step, including zero, at every bin center.
        # Choose half-step edges strictly outside the finite data interval.
        # The resulting centers are integer multiples of ``step``. Strict
        # containment also keeps samples exactly on half-step boundaries from
        # collapsing into the same final bin.
        aligned_lower = (
            math.ceil(float(data_lower) / step - 0.5) - 0.5
        ) * step
        aligned_upper = (
            math.floor(float(data_upper) / step - 0.5) + 1.5
        ) * step
        if aligned_upper <= aligned_lower:
            aligned_upper = aligned_lower + step
        if auto_lower:
            axis["lower"] = aligned_lower + step / 2
        if auto_upper:
            axis["upper"] = aligned_upper - step / 2
        axis["step_size"] = step
        axis["num_bins"] = max(
            int(round((float(axis["upper"]) - float(axis["lower"])) / step)) + 1, 1
        )
        resolved.append(axis)
    return resolved


def _cluster_coordinate_centers(values: Any, tolerance: float) -> np.ndarray:
    """Cluster finite coordinates while keeping every member within tolerance."""

    finite = np.sort(np.asarray(values, dtype=float).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot determine discrete bins without finite coordinates")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("axis tolerance must be finite and nonnegative")
    if tolerance == 0.0:
        return np.unique(finite)
    centers: list[float] = []
    start = 0
    running_sum = float(finite[0])
    count = 1
    for stop in range(1, finite.size):
        candidate_sum = running_sum + float(finite[stop])
        candidate_count = count + 1
        candidate_mean = candidate_sum / candidate_count
        if max(
            candidate_mean - float(finite[start]),
            float(finite[stop]) - candidate_mean,
        ) <= tolerance + 1e-12:
            running_sum = candidate_sum
            count = candidate_count
            continue
        centers.append(running_sum / count)
        start = stop
        running_sum = float(finite[stop])
        count = 1
    centers.append(running_sum / count)
    return np.asarray(centers, dtype=float)


def _coordinate_center_edges(
    centers: Any, *, singleton_half_width: float = 0.5
) -> np.ndarray:
    """Build nearest-center boundaries for an ordered set of centers."""

    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0.0):
        raise ValueError("discrete bin centers must be finite and strictly increasing")
    if values.size == 1:
        half_width = max(float(singleton_half_width), np.finfo(float).eps)
        return np.asarray([values[0] - half_width, values[0] + half_width])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.r_[
        values[0] - (values[1] - values[0]) / 2.0,
        midpoints,
        values[-1] + (values[-1] - values[-2]) / 2.0,
    ]


def _resolve_data_driven_rebin_axes(
    config: Mapping[str, Any],
    axes_config: Sequence[dict[str, Any]],
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    """Resolve Discrete/Tolerance axes from the projected source coordinates."""

    values = np.asarray(coordinates, dtype=float).reshape(-1, len(axes_config))
    resolved: list[dict[str, Any]] = []
    for index, axis_config in enumerate(axes_config):
        axis = _sanitize_rebin_axis_config(dict(axis_config))
        mode = _rebin_axis_mode(config, axis)
        axis["mode"] = mode
        if mode not in {"discrete", "tolerance"}:
            resolved.append(axis)
            continue
        tolerance = 0.0 if mode == "discrete" else float(axis["tolerance"])
        centers = _cluster_coordinate_centers(values[:, index], tolerance)
        edges = _coordinate_center_edges(
            centers,
            singleton_half_width=tolerance if tolerance > 0.0 else 0.5,
        )
        axis.update(
            {
                "bin_edges": edges.tolist(),
                "resolved_centers": centers.tolist(),
                "lower": float(centers[0]),
                "upper": float(centers[-1]),
                "num_bins": int(centers.size),
                "step_size": (
                    float(np.median(np.diff(centers)))
                    if centers.size > 1
                    else max(2.0 * tolerance, 1.0)
                ),
            }
        )
        resolved.append(axis)
    return resolved


def _axis_mode_coordinates_with_symmetry(
    config: Mapping[str, Any],
    projected_coordinates: np.ndarray,
    *,
    physical_coordinates: np.ndarray | None = None,
    symmetry: Sequence[np.ndarray] | None = None,
    output_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Return all projected coordinates needed to derive dynamic axis grids."""

    axes = config.get("axes", [])
    if not any(
        isinstance(axis, Mapping)
        and _rebin_axis_mode(config, axis) in {"discrete", "tolerance"}
        for axis in axes
    ):
        return np.asarray(projected_coordinates, dtype=float)
    if symmetry is None:
        return np.asarray(projected_coordinates, dtype=float)
    if physical_coordinates is None or output_basis is None:
        raise ValueError("dynamic axis modes require projected symmetry coordinates")
    physical = np.asarray(physical_coordinates, dtype=float)
    inverse_basis = np.linalg.inv(np.asarray(output_basis, dtype=float))
    ndim = physical.shape[-1]
    projected = []
    for operation in symmetry:
        transform = np.eye(ndim, dtype=float)
        transform[:3, :3] = np.asarray(operation, dtype=float).T
        projected.append(physical @ transform @ inverse_basis)
    return np.concatenate(projected, axis=0)


def _finite_coordinate_bounds(coordinates: np.ndarray) -> list[tuple[float, float]]:
    values = np.asarray(coordinates, dtype=float).reshape(-1, coordinates.shape[-1])
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.size == 0:
        raise ValueError("cannot determine automatic limits without finite coordinates")
    return [
        (float(np.min(values[:, index])), float(np.max(values[:, index])))
        for index in range(values.shape[1])
    ]


def _symmetry_projected_coordinate_bounds(
    physical_coordinates: np.ndarray,
    symmetry: Sequence[np.ndarray],
    output_basis: np.ndarray,
) -> list[tuple[float, float]]:
    """Return output-coordinate bounds over every reciprocal-symmetry image."""

    physical = np.asarray(physical_coordinates, dtype=float)
    basis = np.asarray(output_basis, dtype=float)
    ndim = physical.shape[-1]
    if ndim < 3 or basis.shape != (ndim, ndim):
        raise ValueError(
            "symmetry auto limits require HKL as the first three coordinates"
        )
    inverse_basis = np.linalg.inv(basis)
    lower = np.full(ndim, np.inf, dtype=float)
    upper = np.full(ndim, -np.inf, dtype=float)
    for operation in symmetry:
        transform = np.eye(ndim, dtype=float)
        transform[:3, :3] = np.asarray(operation, dtype=float).T
        projected = physical @ transform @ inverse_basis
        operation_bounds = _finite_coordinate_bounds(projected)
        lower = np.minimum(lower, [bound[0] for bound in operation_bounds])
        upper = np.maximum(upper, [bound[1] for bound in operation_bounds])
    return list(zip(lower.tolist(), upper.tolist(), strict=True))


def _rebin_grid_kwargs(
    config: dict[str, Any], axes_config: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a mixed uniform/explicit-edge grid for the public rebinner."""

    from .rebin import _uniform_center_edges

    # Step axes retain constant-width indexing. Bins axes are materialized as
    # explicit edges so the one-bin lower/upper interval convention and exact
    # requested count remain valid when modes are mixed across dimensions.
    result: dict[str, Any] = {
        "step_size": [
            float(axis["step_size"])
            for axis in axes_config
        ]
    }
    bin_edges = [
        (
            _uniform_center_edges(
                axis["lower"], axis["upper"], count=int(axis["num_bins"])
            ).tolist()
            if _rebin_axis_mode(config, axis) == "bins"
            else axis.get("bin_edges")
            if _rebin_axis_mode(config, axis) in {"edges", "discrete", "tolerance"}
            else None
        )
        for axis in axes_config
    ]
    if any(edges is not None for edges in bin_edges):
        result["bin_edges"] = bin_edges
    return result


def _composite_rebin_step_sizes(config: dict[str, Any]) -> list[float] | None:
    """Return composite step sizes only when the composite is in Step mode."""

    if _rebin_resolution_mode(config) != "step":
        return None
    return [
        float(axis["step_size"])
        for axis in (_sanitize_rebin_axis_config(axis) for axis in config.get("axes", []))
    ]


def _composite_rebin_bin_edges(
    config: dict[str, Any],
) -> list[list[float] | None] | None:
    """Return per-axis explicit edges, retaining ``None`` for uniform axes."""

    axes = [
        _sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])
    ]
    edges = [axis.get("bin_edges") for axis in axes]
    return edges if any(values is not None for values in edges) else None


def create_rebinned_dataset(
    group: DataGroup,
    dataset: DatasetEntry,
    *,
    name: str | None = None,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    """Materialize a dataset's rebinned view as an independent dataset."""

    data = rebinned_dataset_data(
        dataset,
        extra_masks=effective_dataset_masks(group, dataset),
        progress_callback=progress_callback,
    )
    parameters = {}
    for key in (
        "temperature",
        "magnetic_field",
        KINEMATIC_KF_KI_INCLUDED_KEY,
        SPECTRAL_CHANNEL_CONFIG_KEY,
    ):
        if key in dataset.parameters:
            parameters[key] = copy.deepcopy(dataset.parameters[key])
    new_entry = DatasetEntry(
        name=_unique_dataset_name(name or f"{dataset.name} rebinned", group.dataset_names),
        data=data,
        kind=dataset.kind,
        metadata={
            **copy.deepcopy(dataset.metadata),
            "source_dataset": dataset.name,
            "rebin_materialized": True,
        },
        parameters=parameters,
        masks=copy.deepcopy(dataset.masks),
    )
    new_entry.replace_data(data, source_backed=False)
    group.add_dataset(new_entry)
    return new_entry


def save_dataset_file(dataset: DatasetEntry, path: str | Path, *, use_view: bool = True) -> None:
    """Save a supported dataset to a portable, re-importable ``.npz`` file."""

    data = dataset_for_slice_viewer(dataset) if use_view else dataset.data
    if isinstance(data, PointListData):
        _save_point_list_file(data, path)
        return
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset saving currently supports MDHistoData or PointListData datasets")
    saved_metadata = dict(data.metadata)
    saved_metadata.setdefault("signal_semantics", signal_semantics(data))
    payload: dict[str, Any] = {
        "nfit_dataset_format": np.asarray("nfit-dataset"),
        "nfit_dataset_version": np.asarray(3, dtype=int),
        "nfit_data_container": np.asarray("mdhisto"),
        "signal": data.signal,
        "errors": data.errors,
        "mask": data.mask,
        "num_events": data.num_events,
        "metadata_json": json.dumps(_json_safe_value(saved_metadata), sort_keys=True),
        "axis_count": np.asarray(len(data.axes), dtype=int),
        "coordinate_system": np.asarray(-1 if data.coordinate_system is None else data.coordinate_system),
        "visual_normalization": np.asarray(
            -1 if data.visual_normalization is None else data.visual_normalization
        ),
    }
    context = {
        key: copy.deepcopy(dataset.parameters[key])
        for key in (
            "temperature",
            "magnetic_field",
            KINEMATIC_KF_KI_INCLUDED_KEY,
            SPECTRAL_CHANNEL_CONFIG_KEY,
        )
        if key in dataset.parameters
    }
    observable = data.metadata.get("spectral_observable")
    if isinstance(observable, dict) and SPECTRAL_CHANNEL_CONFIG_KEY in context:
        saved_config = dict(context[SPECTRAL_CHANNEL_CONFIG_KEY])
        saved_config["source_representation"] = str(
            observable.get("fit_representation", saved_config["source_representation"])
        )
        saved_config["source_unit"] = (
            str(data.metadata.get("signal_unit", "arbitrary")) or "arbitrary"
        )
        saved_config["signal_per_mbarn"] = 0.0
        context[SPECTRAL_CHANNEL_CONFIG_KEY] = saved_config
    payload["dataset_context_json"] = np.asarray(json.dumps(_json_safe_value(context), sort_keys=True))
    for index, axis in enumerate(data.axes):
        payload[f"axis_{index}_values"] = axis.values
        payload[f"axis_{index}_name"] = np.asarray(axis.name)
        payload[f"axis_{index}_units"] = np.asarray(axis.units)
        payload[f"axis_{index}_kind"] = np.asarray(axis.kind)
        payload[f"axis_{index}_frame"] = np.asarray(axis.frame or "")
        payload[f"axis_{index}_path"] = np.asarray(axis.path or "")
        payload[f"axis_{index}_metadata_json"] = np.asarray(json.dumps(_json_safe_value(axis.metadata), sort_keys=True))
    payload["auxiliary_channel_names_json"] = np.asarray(json.dumps(list(data.auxiliary_channels)))
    for index, channel in enumerate(data.auxiliary_channels.values()):
        payload[f"auxiliary_{index}_values"] = channel.values
        if channel.errors is not None:
            payload[f"auxiliary_{index}_errors"] = channel.errors
        payload[f"auxiliary_{index}_label"] = np.asarray(channel.label)
        payload[f"auxiliary_{index}_unit"] = np.asarray(channel.unit)
        payload[f"auxiliary_{index}_quantity_type"] = np.asarray(
            channel.quantity_type
        )
    np.savez_compressed(path, **payload)


def _save_point_list_file(data: PointListData, path: str | Path) -> None:
    """Save point-list columns and roles to a script-readable ``.npz`` file."""

    payload: dict[str, Any] = {
        "nfit_dataset_format": np.asarray("nfit-dataset"),
        "nfit_dataset_version": np.asarray(1, dtype=int),
        "nfit_data_container": np.asarray("point_list"),
        "column_names_json": json.dumps(list(data.column_names)),
        "coordinate_names_json": json.dumps(list(data.coordinate_names)),
        "channels_json": json.dumps(_json_safe_value(data.channels)),
        "units_json": json.dumps(_json_safe_value(data.units)),
        "quantity_types_json": json.dumps(_json_safe_value(data.quantity_types)),
        "metadata_json": json.dumps(_json_safe_value(data.metadata), sort_keys=True),
    }
    for index, name in enumerate(data.column_names):
        payload[f"column_{index}"] = np.asarray(data.column(name), dtype=float)
    np.savez_compressed(path, **payload)


def _load_nfit_dataset_file(path: str | Path) -> tuple[MDHistoData | PointListData, dict[str, Any]]:
    """Load an nfit dataset archive written by :func:`save_dataset_file`."""

    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read nfit dataset archive {source}") from exc
    with archive:
        files = set(archive.files)
        if {"signal", "errors", "mask", "num_events", "axis_count"} <= files:
            data = _load_nfit_mdhisto_archive(archive, source)
        elif {"column_names_json", "coordinate_names_json", "channels_json", "units_json"} <= files:
            data = _load_nfit_point_list_archive(archive, source)
        else:
            raise ValueError(f"{source} is not an nfit dataset archive")
        context = _nfit_archive_json_mapping(archive, "dataset_context_json")
    return data, context


def _load_nfit_mdhisto_archive(archive: Any, source: Path) -> MDHistoData:
    axis_count = int(np.asarray(archive["axis_count"]).item())
    axes = []
    for index in range(axis_count):
        prefix = f"axis_{index}_"
        values_key = f"{prefix}values"
        if values_key not in archive:
            raise ValueError(f"{source} is missing {values_key}")
        axes.append(
            MDHistoAxis(
                name=_nfit_archive_text(archive, f"{prefix}name", f"Axis {index}"),
                values=np.asarray(archive[values_key], dtype=float),
                units=_nfit_archive_text(archive, f"{prefix}units"),
                kind=_nfit_archive_text(archive, f"{prefix}kind", "unknown"),
                frame=_nfit_archive_text(archive, f"{prefix}frame") or None,
                path=_nfit_archive_text(archive, f"{prefix}path") or None,
                metadata=_nfit_archive_json_mapping(archive, f"{prefix}metadata_json"),
            )
        )
    metadata = _nfit_archive_json_mapping(archive, "metadata_json")
    metadata["export_file"] = str(source)
    if "signal_semantics" not in metadata:
        raise ValueError(
            f"{source} is missing signal_semantics metadata; "
            "re-export it with a current nfit version"
        )
    channel_names = json.loads(_nfit_archive_text(archive, "auxiliary_channel_names_json", "[]"))
    auxiliary_channels = {
        str(name): MDHistoChannel(
            values=np.asarray(archive[f"auxiliary_{index}_values"], dtype=float),
            errors=(np.asarray(archive[f"auxiliary_{index}_errors"], dtype=float) if f"auxiliary_{index}_errors" in archive else None),
            label=_nfit_archive_text(archive, f"auxiliary_{index}_label", str(name)),
            unit=_nfit_archive_text(archive, f"auxiliary_{index}_unit"),
            quantity_type=_nfit_archive_text(
                archive, f"auxiliary_{index}_quantity_type", "unknown"
            ),
        )
        for index, name in enumerate(channel_names)
    }
    return MDHistoData(
        axes=tuple(axes),
        signal=np.asarray(archive["signal"], dtype=float),
        errors=np.asarray(archive["errors"], dtype=float),
        mask=np.asarray(archive["mask"], dtype=bool),
        num_events=np.asarray(archive["num_events"], dtype=float),
        coordinate_system=_nfit_archive_optional_int(archive, "coordinate_system"),
        visual_normalization=_nfit_archive_optional_int(archive, "visual_normalization"),
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _load_nfit_point_list_archive(archive: Any, source: Path) -> PointListData:
    names = json.loads(_nfit_archive_text(archive, "column_names_json"))
    columns = {str(name): np.asarray(archive[f"column_{index}"], dtype=float) for index, name in enumerate(names)}
    metadata = _nfit_archive_json_mapping(archive, "metadata_json")
    metadata["export_file"] = str(source)
    return PointListData(
        columns=columns,
        units=_nfit_archive_json_mapping(archive, "units_json"),
        coordinate_names=json.loads(_nfit_archive_text(archive, "coordinate_names_json")),
        channels=json.loads(_nfit_archive_text(archive, "channels_json")),
        metadata=metadata,
        quantity_types=_nfit_archive_json_mapping(archive, "quantity_types_json"),
    )


def _nfit_archive_text(archive: Any, key: str, default: str = "") -> str:
    return str(np.asarray(archive[key]).item()) if key in archive else default


def _nfit_archive_optional_int(archive: Any, key: str) -> int | None:
    """Read an optional archive integer, where ``-1`` represents ``None``."""

    if key not in archive:
        return None
    value = int(np.asarray(archive[key]).item())
    return None if value < 0 else value


def _nfit_archive_json_mapping(archive: Any, key: str) -> dict[str, Any]:
    if key not in archive:
        return {}
    try:
        value = json.loads(_nfit_archive_text(archive, key))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON field {key!r} in nfit dataset archive") from exc
    return dict(value) if isinstance(value, dict) else {}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _default_rebin_axes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, PointListData):
        axes = []
        coordinate_names = list(data.coordinate_names) or list(data.column_names)[:1]
        for name in coordinate_names:
            values = np.asarray(data.column(name), dtype=float)
            finite = values[np.isfinite(values)]
            lower = float(np.min(finite)) if finite.size else 0.0
            upper = float(np.max(finite)) if finite.size else 1.0
            num_bins = max(int(np.unique(finite).size), 1) if finite.size else 1
            num_bins = min(num_bins, 200)
            axes.append(
                {
                    "name": name,
                    "units": data.unit(name),
                    "lower": lower,
                    "upper": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
                }
            )
        return axes
    if isinstance(data, MDHistoData):
        axes: list[dict[str, Any]] = []
        ndim = len(data.axes)
        for index, (axis, size) in enumerate(zip(data.axes, data.shape, strict=True)):
            lower, upper = float(axis.centers[0]), float(axis.centers[-1])
            num_bins = max(int(size), 1)
            native_lower, native_upper = _axis_bounds(axis, size)
            vector = _mdhisto_rebin_axis_vector(axis, index, ndim)
            variable = _mdhisto_rebin_axis_variable(axis, index, ndim)
            axes.append(
                {
                    "name": _rebin_axis_name(variable, vector),
                    "variable": variable,
                    "units": axis.units,
                    "vector": vector,
                    "lower": lower,
                    "upper": upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": lower,
                    "auto_upper_value": upper,
                    "num_bins": num_bins,
                    "step_size": (native_upper - native_lower) if size == 1 else _step_size_from_bounds(lower, upper, num_bins),
                }
            )
        return axes
    if isinstance(data, PointData4D):
        axes = []
        for index, (name, units, values) in enumerate(
            (
                ("H", "rlu", data.H),
                ("K", "rlu", data.K),
                ("L", "rlu", data.L),
                ("E", "meV", data.E),
            )
        ):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            lower = float(np.min(finite)) if finite.size else 0.0
            upper = float(np.max(finite)) if finite.size else 1.0
            # Point clouds commonly contain a distinct floating-point value at
            # nearly every observation. Treating every unique value as a grid
            # coordinate would create an unusably large Cartesian product
            # (MACS HKLE scans are a representative case). Keep the automatic
            # preview below the normal two-million-bin safety threshold; users
            # can then choose instrument-appropriate step sizes explicitly.
            num_bins = min(max(int(np.unique(finite).size), 1), 32) if finite.size else 1
            axes.append(
                {
                    "name": name,
                    "variable": name,
                    "units": units,
                    "vector": _identity_vector(index, 4),
                    "lower": lower,
                    "upper": upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": lower,
                    "auto_upper_value": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
                    "auto_step_size": True,
                    "auto_step_size_value": _step_size_from_bounds(
                        lower, upper, num_bins
                    ),
                }
            )
        return axes
    return []


def _axis_bounds(axis: MDHistoAxis, size: int) -> tuple[float, float]:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return float(values[0]), float(values[-1])
    centers = axis.centers
    if centers.size == 0:
        return 0.0, 1.0
    if centers.size == 1:
        return float(centers[0] - 0.5), float(centers[0] + 0.5)
    step = float(np.nanmedian(np.diff(centers)))
    return float(centers[0] - 0.5 * step), float(centers[-1] + 0.5 * step)


def _step_size_from_bounds(lower: float, upper: float, num_bins: int) -> float:
    if num_bins <= 0:
        return 0.0
    if upper == lower:
        return 1.0
    return float((float(upper) - float(lower)) / float(max(num_bins - 1, 1)))


def _num_bins_from_step_size(lower: Any, upper: Any, step_size: float) -> int:
    width = abs(float(upper) - float(lower))
    if width == 0.0:
        return 1
    return max(int(np.floor(width / float(step_size) + 1.e-10)) + 1, 1)


def _normalize_rebin_bin_edges(value: Any) -> list[float]:
    """Validate one axis's optional explicit, potentially nonuniform edges."""

    edges = np.asarray(value, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin edges must be a one-dimensional list with at least two values")
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin edges must be finite and strictly increasing")
    return edges.tolist()


def _sanitize_rebin_axis_config(axis_config: dict[str, Any]) -> dict[str, Any]:
    lower = float(axis_config.get("lower", 0.0))
    upper = float(axis_config.get("upper", lower))
    num_bins = max(int(axis_config.get("num_bins", 1)), 1)
    try:
        step_size = float(axis_config.get("step_size"))
    except (TypeError, ValueError):
        step_size = 0.0
    if not np.isfinite(step_size) or step_size <= 0.0:
        step_size = _step_size_from_bounds(lower, upper, num_bins)
    sanitized = {
        **axis_config,
        "lower": lower,
        "upper": upper,
        "num_bins": num_bins,
        "step_size": step_size,
    }
    mode = str(axis_config.get("mode", "")).casefold()
    if mode:
        sanitized["mode"] = mode if mode in REBIN_AXIS_MODES else "step"
    if "tolerance" in axis_config or mode == "tolerance":
        try:
            tolerance = float(axis_config.get("tolerance", step_size))
        except (TypeError, ValueError):
            tolerance = step_size
        sanitized["tolerance"] = (
            tolerance if np.isfinite(tolerance) and tolerance > 0.0 else step_size
        )
    explicit_edges = axis_config.get("bin_edges")
    if explicit_edges is not None and not (
        isinstance(explicit_edges, str) and not explicit_edges.strip()
    ):
        edges = _normalize_rebin_bin_edges(explicit_edges)
        sanitized["bin_edges"] = edges
        sanitized["lower"] = (edges[0] + edges[1]) / 2 if len(edges) > 2 else edges[0]
        sanitized["upper"] = (edges[-2] + edges[-1]) / 2 if len(edges) > 2 else edges[-1]
        sanitized["num_bins"] = len(edges) - 1
        sanitized["step_size"] = (edges[-1] - edges[0]) / (len(edges) - 1)
    else:
        sanitized.pop("bin_edges", None)
    vector = axis_config.get("vector")
    if isinstance(vector, (list, tuple, np.ndarray)):
        sanitized["vector"] = [_clean_axis_weight(component) for component in vector]
    return sanitized


def _rebin_axis_vector(axis_config: dict[str, Any], index: int, ndim: int) -> np.ndarray:
    """Return the projection vector for a rebin output axis, falling back to identity."""

    vector = axis_config.get("vector")
    if isinstance(vector, (list, tuple, np.ndarray)) and len(vector) == ndim:
        candidate = np.asarray(vector, dtype=float)
        if np.all(np.isfinite(candidate)):
            return candidate
    return np.asarray(_identity_vector(index if 0 <= index < ndim else 0, ndim), dtype=float)


def _mdhisto_rebin_axis_variable(axis: MDHistoAxis, index: int, ndim: int) -> str:
    """Return the scalar variable used to parameterize one rebin axis row."""

    role_variables = {
        "h": "H",
        "k": "K",
        "l": "L",
        "energy": "E",
        "energy_transfer": "E",
    }
    if axis.role in role_variables:
        return role_variables[axis.role]
    if str(axis.name).casefold().replace("_", "") in {
        "deltae",
        "energy",
        "energytransfer",
    }:
        return "E"
    symbols = re.findall(r"[HKL]", str(axis.name).upper())
    if symbols:
        return symbols[0]
    if ndim == 4 and 0 <= index < 4:
        return ("H", "K", "L", "E")[index]
    return str(axis.name or f"Axis {index + 1}")


def _rebin_axis_name(variable: str, vector: Any) -> str:
    """Generate the plotted axis name from its variable and HKLE direction."""

    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size == 4 and str(variable).upper() == "E":
        coefficient = _clean_axis_weight(values[3])
        if coefficient == 1.0:
            return "DeltaE"
        return f"{coefficient:g}DeltaE"
    if values.size == 4:
        symbol = str(variable).upper()
        terms = []
        for value in values[:3]:
            coefficient = _clean_axis_weight(value)
            if coefficient == 0.0:
                terms.append("0")
            elif coefficient == 1.0:
                terms.append(symbol)
            elif coefficient == -1.0:
                terms.append(f"-{symbol}")
            else:
                terms.append(f"{coefficient:g}{symbol}")
        return f"[{','.join(terms)}]"
    return str(variable)


def _validate_mdhisto_rebin_basis(axes_config: list[dict[str, Any]], ndim: int) -> np.ndarray:
    """Validate and return the output basis represented by rebin axis rows."""

    basis = np.vstack(
        [_rebin_axis_vector(axis_config, index, ndim) for index, axis_config in enumerate(axes_config)]
    )
    if basis.shape != (ndim, ndim) or not np.all(np.isfinite(basis)):
        raise ValueError("coordinate axes must form a finite square basis")
    if ndim == 4:
        for index, (axis_config, vector) in enumerate(zip(axes_config, basis, strict=True)):
            variable = str(
                axis_config.get("variable", ("H", "K", "L", "E")[index])
            ).upper()
            if variable == "E":
                if np.any(vector[:3] != 0.0) or vector[3] == 0.0:
                    raise ValueError(
                        f"axis row {index + 1} is the energy variable and must contain only a nonzero E component"
                    )
            elif vector[3] != 0.0:
                raise ValueError(
                    f"axis row {index + 1} is a momentum variable; momentum and energy components cannot be mixed"
                )
    if np.linalg.matrix_rank(basis) != ndim:
        rank = int(np.linalg.matrix_rank(basis))
        raise ValueError(
            "coordinate axis vectors must form an invertible basis; "
            f"this matrix has rank {rank} rather than {ndim}"
        )
    return basis


def _momentum_rebin_vector_text(axis_config: dict[str, Any], index: int) -> str:
    """Format one GUI momentum-basis row without exposing the energy column."""

    return _parameter_to_text(_rebin_axis_vector(axis_config, index, 4)[:3].tolist())


def _momentum_rebin_axis_indices(axes_config: Sequence[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, axis in enumerate(axes_config)
        if str(axis.get("variable", ("H", "K", "L", "E")[index])).upper()
        != "E"
    ]


def _momentum_rebin_matrix(axes_config: Sequence[dict[str, Any]]) -> list[list[float]]:
    indices = _momentum_rebin_axis_indices(axes_config)
    if len(axes_config) != 4 or len(indices) != 3:
        raise ValueError("HKLE rebinning requires three momentum coordinates and one energy coordinate")
    return [
        _rebin_axis_vector(axes_config[index], index, 4)[:3].tolist()
        for index in indices
    ]


def _momentum_coordinate_variables(matrix: np.ndarray) -> list[str]:
    """Choose concise H/K/L scalar labels for momentum-matrix rows."""

    symbols = ["H", "K", "L"]
    variables: list[str | None] = [None, None, None]
    used: set[str] = set()
    for row, values in enumerate(matrix):
        nonzero = np.flatnonzero(~np.isclose(values, 0.0))
        if nonzero.size != 1:
            continue
        symbol = symbols[int(nonzero[0])]
        if symbol not in used:
            variables[row] = symbol
            used.add(symbol)
    remaining = iter(symbol for symbol in symbols if symbol not in used)
    return [value if value is not None else next(remaining) for value in variables]


def _rebin_config_basis_bounds(
    axes_config: list[dict[str, Any]],
    candidate_basis: np.ndarray,
) -> list[tuple[float, float]]:
    """Transform the configured output box into a replacement coordinate basis."""

    current_basis = _validate_mdhisto_rebin_basis(axes_config, 4)
    endpoints = [
        (float(axis["lower"]), float(axis["upper"])) for axis in axes_config
    ]
    projected_corners = np.asarray(
        [
            [endpoints[index][bit] for index, bit in enumerate(bits)]
            for bits in np.ndindex(*(2 for _ in endpoints))
        ],
        dtype=float,
    )
    physical_corners = projected_corners @ current_basis
    new_coordinates = physical_corners @ np.linalg.inv(candidate_basis)
    return [
        (float(np.min(new_coordinates[:, index])), float(np.max(new_coordinates[:, index])))
        for index in range(4)
    ]


def _point_data_rebin_basis_bounds(
    data: PointData4D,
    axes_config: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return exact finite point-cloud bounds in the requested output basis."""

    basis = _validate_mdhisto_rebin_basis(axes_config, 4)
    coordinates = np.column_stack(data.coordinates())
    coordinates = coordinates[np.all(np.isfinite(coordinates), axis=1)]
    if coordinates.size == 0:
        raise ValueError("point dataset has no finite HKLE coordinates")
    projected = coordinates @ np.linalg.inv(basis)
    bounds: list[tuple[float, float]] = []
    for index in range(4):
        lower = float(np.min(projected[:, index]))
        upper = float(np.max(projected[:, index]))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        bounds.append((lower, upper))
    return bounds


def _update_rebin_momentum_basis(
    axes_config: list[dict[str, Any]],
    index: int,
    momentum_vector: Sequence[float],
    *,
    data: MDHistoData | PointData4D | None = None,
) -> None:
    """Apply one row of the user-facing 3x3 momentum coordinate block."""

    if len(axes_config) != 4 or not (0 <= index < 4):
        raise ValueError("momentum-coordinate editing requires four HKLE axes")
    variable = str(
        axes_config[index].get("variable", ("H", "K", "L", "E")[index])
    ).upper()
    if variable == "E":
        raise ValueError("the energy coordinate is fixed and is not part of the 3x3 momentum block")
    vector = np.asarray(momentum_vector, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("a momentum coordinate row must contain three finite H, K, and L values")
    matrix = _momentum_rebin_matrix(axes_config)
    matrix[_momentum_rebin_axis_indices(axes_config).index(index)] = vector.tolist()
    _update_rebin_momentum_matrix(axes_config, matrix, data=data)


def _update_rebin_momentum_matrix(
    axes_config: list[dict[str, Any]],
    momentum_matrix: Sequence[Sequence[float]],
    *,
    data: MDHistoData | PointData4D | None = None,
) -> None:
    """Replace the complete 3x3 momentum block atomically."""

    matrix = np.asarray(momentum_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("the momentum-coordinate matrix must be a finite 3x3 array")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 3:
        raise ValueError(
            "the momentum-coordinate matrix must be invertible; "
            f"this matrix has rank {rank} rather than 3"
        )
    candidate = [dict(axis) for axis in axes_config]
    indices = _momentum_rebin_axis_indices(candidate)
    if len(candidate) != 4 or len(indices) != 3:
        raise ValueError("HKLE rebinning requires three momentum coordinates and one energy coordinate")
    for axis_index, axis in enumerate(candidate):
        axis.setdefault("variable", ("H", "K", "L", "E")[axis_index])
    variables = _momentum_coordinate_variables(matrix)
    for row, axis_index in enumerate(indices):
        candidate[axis_index]["variable"] = variables[row]
        candidate[axis_index]["vector"] = [
            *[_clean_axis_weight(component) for component in matrix[row]],
            0.0,
        ]
    basis = _validate_mdhisto_rebin_basis(candidate, 4)
    if isinstance(data, MDHistoData):
        bounds = _mdhisto_rebin_basis_bounds(data, candidate)
    elif isinstance(data, PointData4D):
        bounds = _point_data_rebin_basis_bounds(data, candidate)
    else:
        bounds = _rebin_config_basis_bounds(axes_config, basis)
    for axis_config, (lower, upper) in zip(candidate, bounds, strict=True):
        previous_step = float(axis_config.get("step_size", 0.0) or 0.0)
        axis_config.pop("bin_edges", None)
        axis_config["lower"] = lower
        axis_config["upper"] = upper
        if bool(axis_config.get("auto_lower", False)):
            axis_config["auto_lower_value"] = lower
        if bool(axis_config.get("auto_upper", False)):
            axis_config["auto_upper_value"] = upper
        if previous_step > 0.0 and np.isfinite(previous_step):
            axis_config["num_bins"] = _num_bins_from_step_size(
                lower, upper, previous_step
            )
        axis_config["name"] = _rebin_axis_name(
            str(axis_config.get("variable", "")), axis_config.get("vector", [])
        )
        axis_config.update(_sanitize_rebin_axis_config(axis_config))
    axes_config[:] = candidate


def _mdhisto_rebin_basis_transform(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
) -> np.ndarray:
    """Map source bin coordinates into coordinates of the requested basis."""

    ndim = len(data.axes)
    output_basis = _validate_mdhisto_rebin_basis(axes_config, ndim)
    source_vectors = _mdhisto_rebin_source_axis_vectors(data)
    if any(vector is None for vector in source_vectors):
        raise ValueError("source axes do not define a complete coordinate basis")
    source_basis = np.vstack(source_vectors)
    if source_basis.shape != (ndim, ndim) or np.linalg.matrix_rank(source_basis) != ndim:
        raise ValueError("source coordinate axes must be linearly independent")
    return source_basis @ np.linalg.inv(output_basis)


def _mdhisto_rebin_basis_bounds(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return a bounding box containing the full source grid in a new basis."""

    transform = _mdhisto_rebin_basis_transform(data, axes_config)
    source_bounds = [_axis_bounds(axis, size) for axis, size in zip(data.axes, data.shape, strict=True)]
    bounds = []
    for output_index in range(transform.shape[1]):
        lower = 0.0
        upper = 0.0
        for (source_lower, source_upper), coefficient in zip(
            source_bounds, transform[:, output_index], strict=True
        ):
            endpoints = (coefficient * source_lower, coefficient * source_upper)
            lower += min(endpoints)
            upper += max(endpoints)
        bounds.append((float(lower), float(upper)))
    return bounds


def _update_mdhisto_rebin_basis(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
    index: int,
    vector: list[float],
) -> None:
    """Apply one basis edit, regenerating names and full-data output bounds."""

    candidate = [dict(axis) for axis in axes_config]
    candidate[index]["vector"] = list(vector)
    _validate_mdhisto_rebin_basis(candidate, len(data.axes))
    bounds = _mdhisto_rebin_basis_bounds(data, candidate)
    for axis_config, (lower, upper) in zip(candidate, bounds, strict=True):
        previous_step = float(axis_config.get("step_size", 0.0) or 0.0)
        axis_config["lower"] = lower
        axis_config["upper"] = upper
        if previous_step > 0.0 and np.isfinite(previous_step):
            axis_config["num_bins"] = _num_bins_from_step_size(lower, upper, previous_step)
        axis_config["name"] = _rebin_axis_name(
            str(axis_config.get("variable", "")), axis_config.get("vector", [])
        )
        axis_config.update(_sanitize_rebin_axis_config(axis_config))
    axes_config[:] = candidate


def _mdhisto_rebin_axis_vector(axis: MDHistoAxis, index: int, ndim: int) -> list[float]:
    """Return a default rebin vector that follows the displayed MDHisto axis."""

    if ndim == 4:
        role = axis.role
        role_vectors = {
            "h": [1.0, 0.0, 0.0, 0.0],
            "k": [0.0, 1.0, 0.0, 0.0],
            "l": [0.0, 0.0, 1.0, 0.0],
            "energy": [0.0, 0.0, 0.0, 1.0],
            "energy_transfer": [0.0, 0.0, 0.0, 1.0],
        }
        if role in role_vectors:
            return role_vectors[role]
        projection = _axis_projection_vector(axis.name)
        if projection is not None:
            return [_clean_axis_weight(value) for value in [*projection.tolist(), 0.0]]
    return _identity_vector(index, ndim)


def _mdhisto_rebin_source_axis_vectors(data: MDHistoData) -> list[np.ndarray | None]:
    vectors: list[np.ndarray | None] = []
    if len(data.axes) != 4:
        return [None for _axis in data.axes]
    for axis in data.axes:
        vectors.append(np.asarray(_mdhisto_rebin_axis_vector(axis, len(vectors), len(data.axes)), dtype=float))
    return vectors


def _mdhisto_rebin_component(
    data: MDHistoData,
    source_grids: list[np.ndarray],
    axis_config: dict[str, Any],
    index: int,
) -> np.ndarray:
    ndim = len(data.axes)
    vector = _rebin_axis_vector(axis_config, index, ndim)
    component = np.zeros(data.shape, dtype=float)
    if vector.size == 4:
        source_vectors = _mdhisto_rebin_source_axis_vectors(data)
        for source_vector, grid in zip(source_vectors, source_grids, strict=True):
            if source_vector is not None and np.allclose(vector, source_vector):
                return np.asarray(grid, dtype=float)
        coords = _mdhisto_coordinate_grids(data)
        if all(name in coords for name in ("H", "K", "L", "E")):
            for weight, name in zip(vector, ("H", "K", "L", "E"), strict=True):
                if weight:
                    component = component + float(weight) * coords[name]
            return component
    for weight, grid in zip(vector, source_grids, strict=True):
        if weight:
            component = component + float(weight) * grid
    return component


def _mdhisto_axis_edges(axis: MDHistoAxis, size: int) -> np.ndarray:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return values
    if values.size == size:
        if size == 1:
            return np.asarray([values[0] - 0.5, values[0] + 0.5])
        edges = np.empty(size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return edges
    raise ValueError(f"axis {axis.name!r} does not define {size} bins")


def _mdhisto_cell_volumes(data: MDHistoData) -> np.ndarray:
    """Return native hypervolumes represented by MDHisto bin centers."""

    volume = np.ones(data.shape, dtype=float)
    for dim, (axis, size) in enumerate(zip(data.axes, data.shape, strict=True)):
        edges = _mdhisto_axis_edges(axis, size)
        widths = np.diff(edges)
        shape = [1] * data.signal.ndim
        shape[dim] = size
        volume *= widths.reshape(shape)
    return np.abs(volume)


def _output_bin_volumes(bins_list: Sequence[np.ndarray]) -> np.ndarray:
    shape = tuple(len(values) - 1 for values in bins_list)
    volume = np.ones(shape, dtype=float)
    for dim, values in enumerate(bins_list):
        widths = np.abs(np.diff(np.asarray(values, dtype=float)))
        reshape = [1] * len(shape)
        reshape[dim] = widths.size
        volume *= widths.reshape(reshape)
    return volume


def _rebin_mdhisto_coverage(
    data: MDHistoData,
    coords: np.ndarray,
    config: dict[str, Any],
    axes_config: Sequence[dict[str, Any]],
    bins_list: Sequence[np.ndarray],
    *,
    symmetry: Sequence[np.ndarray] | None = None,
    output_axes: np.ndarray | None = None,
) -> np.ndarray:
    """Map measured native-bin hypervolume into requested output bins."""

    source_coverage = mdhisto_coverage_fraction(data)
    usable = (
        np.isfinite(data.signal)
        & np.isfinite(data.errors)
        & ~np.asarray(data.mask, dtype=bool)
        & mdhisto_measured_bins(data)
    )
    source_coverage = np.where(usable, source_coverage, 0.0)
    source_volume = _mdhisto_cell_volumes(data)
    vectors = np.asarray(
        [
            _rebin_axis_vector(axis_config, index, data.signal.ndim)
            for index, axis_config in enumerate(axes_config)
        ],
        dtype=float,
    )
    mapping = (
        _mdhisto_rebin_basis_transform(data, list(axes_config))
        if data.signal.ndim == 4
        else vectors.T
    )
    if mapping.shape == (data.signal.ndim, data.signal.ndim):
        source_volume = source_volume * abs(float(np.linalg.det(mapping)))
    if (
        symmetry is None
        and mapping.shape == (data.signal.ndim, data.signal.ndim)
        and np.allclose(mapping, np.eye(data.signal.ndim))
    ):
        covered_volume = source_coverage
        for dim, (axis, size, output_edges) in enumerate(
            zip(data.axes, data.shape, bins_list, strict=True)
        ):
            source_edges = _mdhisto_axis_edges(axis, size)
            output_edges = np.asarray(output_edges, dtype=float)
            overlap = np.maximum(
                0.0,
                np.minimum(output_edges[1:, None], source_edges[None, 1:])
                - np.maximum(output_edges[:-1, None], source_edges[None, :-1]),
            )
            covered_volume = np.tensordot(
                overlap,
                covered_volume,
                axes=(1, dim),
            )
            covered_volume = np.moveaxis(covered_volume, 0, dim)
        output_volume = _output_bin_volumes(bins_list)
        coverage = np.zeros(output_volume.shape, dtype=float)
        np.divide(
            covered_volume,
            output_volume,
            out=coverage,
            where=output_volume > 0.0,
        )
        return np.clip(coverage, 0.0, 1.0)
    weights = (source_volume * source_coverage).ravel()
    kwargs = dict(
        data_weights=weights,
        lower=[float(np.asarray(values)[0]) for values in bins_list],
        upper=[float(np.asarray(values)[-1]) for values in bins_list],
        **_rebin_grid_kwargs(config, list(axes_config)),
        # Coverage is geometric even when the signal reducer uses nearest-bin
        # assignment. Cloud-in-cell deposition avoids assigning an entire
        # rotated native voxel to whichever output bin contains its center.
        fractional=True,
        normalize=False,
        mean_weighting="uniform",
        max_batch_bytes=_rebin_max_batch_bytes(config),
    )
    # Reuse the resolved edges, rather than interpreting them as centers again.
    kwargs["bin_edges"] = bins_list
    coverage_rebin = (
        rebin_nd_symmetry(
            np.ones(data.signal.size, dtype=float),
            np.asarray(coords, dtype=float).reshape(-1, data.signal.ndim),
            symmetry,
            axes=output_axes,
            **kwargs,
        )
        if symmetry is not None
        else rebin_nd(
            np.ones(data.signal.size, dtype=float),
            np.asarray(coords, dtype=float).reshape(-1, data.signal.ndim),
            **kwargs,
        )
    )
    if coverage_rebin.binned_data is None:
        raise RuntimeError("coverage rebinning did not produce binned data")
    output_volume = _output_bin_volumes(bins_list)
    coverage = np.zeros(output_volume.shape, dtype=float)
    np.divide(
        np.asarray(coverage_rebin.binned_data, dtype=float),
        output_volume,
        out=coverage,
        where=output_volume > 0.0,
    )
    return np.clip(coverage, 0.0, 1.0)


def _rebin_mdhisto_data(
    data: MDHistoData,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData:
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "projecting source coordinates and applying masks",
            }
        )
    if any("metadata_dimension" in axis.metadata for axis in data.axes):
        raise ValueError("Rebin the original collection to preserve its discrete metadata dimensions.")
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    if len(axes_config) != len(data.axes):
        axes_config = _default_rebin_axes(data)

    ndim = len(data.axes)
    source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    if ndim == 4:
        transform = _mdhisto_rebin_basis_transform(data, axes_config)
        projected = []
        for output_index in range(ndim):
            component = np.zeros(data.shape, dtype=float)
            for source_grid, coefficient in zip(
                source_grids, transform[:, output_index], strict=True
            ):
                if coefficient:
                    component = component + float(coefficient) * source_grid
            projected.append(component)
    else:
        projected = [
            _mdhisto_rebin_component(data, list(source_grids), axis_config, index)
            for index, axis_config in enumerate(axes_config)
        ]
    coords = np.stack(projected, axis=-1)
    symmetry = _rebin_symmetry_matrices(config, data.metadata.get("lattice_parameters"))
    output_axes = None
    if symmetry is not None:
        if ndim != 4:
            raise ValueError("rebin symmetry requires a four-dimensional HKLE dataset")
        physical = _mdhisto_coordinate_grids(data)
        if not all(name in physical for name in ("H", "K", "L", "E")):
            raise ValueError("rebin symmetry requires reconstructable H, K, L, and energy coordinates")
        coords = np.stack([physical[name] for name in ("H", "K", "L", "E")], axis=-1)
        output_axes = _validate_mdhisto_rebin_basis(axes_config, ndim)
        data_bounds = _symmetry_projected_coordinate_bounds(
            coords, symmetry, output_axes
        )
    else:
        data_bounds = _finite_coordinate_bounds(coords)
    axes_config = _resolve_auto_rebin_axes(axes_config, data_bounds)
    valid = np.isfinite(data.signal) & np.isfinite(data.errors) & ~data.mask
    normalization_channel = data.auxiliary_channels.get(
        "normalization_denominator"
    )
    normalization_values = None
    if normalization_channel is not None:
        normalization_values = np.asarray(normalization_channel.values, dtype=float)
        valid &= np.isfinite(normalization_values) & (normalization_values > 0.0)
    if data.num_events is not None:
        valid &= mdhisto_measured_bins(data)
    signal = data.signal[valid]
    errors = data.errors[valid]
    coords_valid = coords[valid]
    if signal.size == 0:
        raise ValueError("no valid data points remain before rebinning")
    projected_valid = (
        coords_valid @ np.linalg.inv(output_axes)
        if output_axes is not None
        else coords_valid
    )
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        projected_valid,
        physical_coordinates=coords_valid if symmetry is not None else None,
        symmetry=symmetry,
        output_basis=output_axes,
    )
    axes_config = _resolve_data_driven_rebin_axes(
        config, axes_config, mode_coordinates
    )
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    kwargs = dict(
        data_errs=errors,
        data_weights=(
            None
            if normalization_values is None
            else normalization_values[valid]
        ),
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    result = (
        rebin_nd_symmetry(signal, coords_valid, symmetry, axes=output_axes, **kwargs)
        if symmetry is not None
        else rebin_nd(signal, coords_valid, **kwargs)
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None:
        raise RuntimeError("rebinning did not produce binned data")
    if result.bins_list is None:
        raise RuntimeError("rebinning did not produce bins")
    vectors = [_rebin_axis_vector(axis_config, index, ndim).tolist() for index, axis_config in enumerate(axes_config)]
    rebinned_axes = tuple(
        MDHistoAxis(
            name=str(axis_config.get("name") or source_axis.name),
            values=np.asarray(bins, dtype=float),
            units=str(axis_config.get("units") if axis_config.get("units") is not None else source_axis.units),
            kind=source_axis.kind,
            frame=source_axis.frame,
            path=source_axis.path,
            metadata={
                **dict(source_axis.metadata),
                "variable": str(axis_config.get("variable", "")),
                **(
                    {"discrete_centers": axis_config["resolved_centers"]}
                    if axis_config.get("resolved_centers") is not None
                    else {}
                ),
            },
        )
        for source_axis, axis_config, bins in zip(data.axes, axes_config, result.bins_list, strict=True)
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_coverage",
                "message": "calculating geometric coverage for output bins",
            }
        )
    coverage = _rebin_mdhisto_coverage(
        data,
        coords,
        config,
        axes_config,
        result.bins_list,
        symmetry=symmetry,
        output_axes=output_axes,
    )
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata = dict(data.metadata)
    metadata["signal_semantics"] = "density"
    metadata["signal_semantics_source"] = "nfit_normalized_rebin"
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    metadata["rebin"] = {
        "lower": lower,
        "upper": upper,
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "bin_edges": [np.asarray(edges, dtype=float).tolist() for edges in result.bins_list],
        "vectors": vectors,
        "fractional_axes": _rebin_fractional_axes(config, axes_config),
        "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
        "normalize": True,
        "mean_weighting": _rebin_mean_weighting(config),
        "minimum_coverage": _rebin_minimum_coverage(config),
        "minimum_samples": _rebin_minimum_samples(config),
        "max_batch_mb": _rebin_max_batch_mb(config),
        "max_batch_bytes": _rebin_max_batch_bytes(config),
        "weighted_by_normalization_denominator": normalization_values is not None,
    }
    symmetry_metadata = _rebin_symmetry_metadata(config, data.metadata.get("lattice_parameters"))
    if symmetry_metadata is not None:
        metadata["rebin"]["symmetry"] = symmetry_metadata
    auxiliary_channels = {
        "coverage_fraction": MDHistoChannel(
            coverage,
            label="Coverage",
            unit="fraction",
        )
    }
    if (
        normalization_values is not None
        and _rebin_mean_weighting(config) == "uniform"
        and result._normalization is not None
    ):
        auxiliary_channels["normalization_denominator"] = MDHistoChannel(
            np.asarray(result._normalization, dtype=float),
            label="Combined detector-trajectory normalization",
            unit="arbitrary normalization units",
        )
    return MDHistoData(
        axes=rebinned_axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=data.coordinate_system,
        visual_normalization=data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _rebin_point_data(
    data: PointData4D,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData:
    source = data.valid(require_positive_sigma=False)
    if source.size == 0:
        raise ValueError("no valid data points remain before rebinning")
    normalization_weighted = source.normalization_denominator is not None
    return _point_data_histogram(
        np.column_stack(source.coordinates()),
        np.asarray(source.intensity, dtype=float),
        np.asarray(source.sigma, dtype=float),
        config,
        data_weights=(
            None
            if source.normalization_denominator is None
            else np.asarray(source.normalization_denominator, dtype=float)
        ),
        source_metadata=data.metadata,
        coordinate_system=data.metadata.get("coordinate_system"),
        visual_normalization=data.metadata.get("visual_normalization"),
        metadata_updates={
            "weighted_by_normalization_denominator": normalization_weighted,
        },
        progress_callback=progress_callback,
    )


def _point_data_histogram(
    coordinates: np.ndarray,
    signal: np.ndarray,
    errors: np.ndarray,
    config: dict[str, Any],
    *,
    data_weights: np.ndarray | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    coordinate_system: int | None = None,
    visual_normalization: int | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> MDHistoData:
    """Bin physical HKLE point coordinates into a viewer-ready histogram."""

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "projecting point coordinates and resolving per-axis grids",
            }
        )

    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    if len(axes_config) != 4:
        raise ValueError("HKLE point-data rebinning requires four output axes")
    for index, axis_config in enumerate(axes_config):
        axis_config.setdefault("variable", ("H", "K", "L", "E")[index])
    basis = _validate_mdhisto_rebin_basis(axes_config, 4)
    physical_coordinates = np.asarray(coordinates, dtype=float)
    projected_coordinates = physical_coordinates @ np.linalg.inv(basis)
    metadata = copy.deepcopy(dict(source_metadata or {}))
    symmetry = _rebin_symmetry_matrices(config, metadata.get("lattice_parameters"))
    data_bounds = (
        _symmetry_projected_coordinate_bounds(
            physical_coordinates, symmetry, basis
        )
        if symmetry is not None
        else _finite_coordinate_bounds(projected_coordinates)
    )
    axes_config = _resolve_auto_rebin_axes(
        axes_config, data_bounds
    )
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        projected_coordinates,
        physical_coordinates=physical_coordinates,
        symmetry=symmetry,
        output_basis=basis,
    )
    axes_config = _resolve_data_driven_rebin_axes(
        config, axes_config, mode_coordinates
    )
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    kwargs = dict(
        data_errs=np.asarray(errors, dtype=float),
        data_weights=(
            None if data_weights is None else np.asarray(data_weights, dtype=float)
        ),
        axes=basis,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    result = (
        rebin_nd_symmetry(
            np.asarray(signal, dtype=float),
            physical_coordinates,
            symmetry,
            **kwargs,
        )
        if symmetry is not None
        else rebin_nd(
            np.asarray(signal, dtype=float),
            physical_coordinates,
            **kwargs,
        )
    )
    if (
        result.binned_data is None
        or result.binned_data_errs is None
        or result.n_samples is None
        or result.bins_list is None
    ):
        raise RuntimeError("point-data rebinning did not produce binned data")
    mask = (
        ~np.isfinite(result.binned_data)
        | ~np.isfinite(result.binned_data_errs)
        | (result.n_samples <= 0.0)
    )
    coverage = np.asarray(result.n_samples > 0.0, dtype=float)
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata.update(copy.deepcopy(dict(metadata_updates or {})))
    metadata["signal_semantics"] = "density"
    metadata["signal_semantics_source"] = "nfit_normalized_rebin"
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    metadata["rebin"] = {
        "lower": lower,
        "upper": upper,
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "bin_edges": [
            np.asarray(edges, dtype=float).tolist() for edges in result.bins_list
        ],
        "vectors": basis.tolist(),
        "fractional_axes": _rebin_fractional_axes(config, axes_config),
        "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
        "normalize": True,
        "mean_weighting": _rebin_mean_weighting(config),
        "minimum_coverage": _rebin_minimum_coverage(config),
        "minimum_samples": _rebin_minimum_samples(config),
        "max_batch_mb": _rebin_max_batch_mb(config),
        "max_batch_bytes": _rebin_max_batch_bytes(config),
    }
    for key in (
        "weighted_by_fit_weight",
        "weighted_by_normalization_denominator",
    ):
        if key in metadata:
            metadata["rebin"][key] = bool(metadata[key])
    symmetry_metadata = _rebin_symmetry_metadata(config, metadata.get("lattice_parameters"))
    if symmetry_metadata is not None:
        metadata["rebin"]["symmetry"] = symmetry_metadata
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_output",
                "message": "building the rebinned dataset and output channels",
            }
        )
    return MDHistoData(
        axes=tuple(
            MDHistoAxis(
                name=str(axis_config.get("name") or ("H", "K", "L", "DeltaE")[index]),
                values=np.asarray(edges, dtype=float),
                units=str(axis_config.get("units") or ("rlu" if index < 3 else "meV")),
                kind="momentum" if index < 3 else "energy_transfer",
                frame="HKL" if index < 3 else "General Frame",
                metadata={
                    "variable": str(
                        axis_config.get("variable", ("H", "K", "L", "E")[index])
                    ),
                    **(
                        {"discrete_centers": axis_config["resolved_centers"]}
                        if axis_config.get("resolved_centers") is not None
                        else {}
                    ),
                },
            )
            for index, (axis_config, edges) in enumerate(
                zip(axes_config, result.bins_list, strict=True)
            )
        ),
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=coordinate_system,
        visual_normalization=visual_normalization,
        metadata=metadata,
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(
                coverage,
                label="Coverage",
                unit="fraction",
            )
        },
    )


def _point_data_with_nfit_masks(
    dataset: DatasetEntry,
    data: PointData4D,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> PointData4D:
    reject = ~np.asarray(data.mask, dtype=bool)
    nfit_mask = _nfit_mask_for_point_data(dataset, data, extra_masks=extra_masks)
    combined_reject = reject | nfit_mask
    if not np.any(combined_reject) and np.all(np.asarray(data.mask, dtype=bool)):
        return data
    metadata = dict(data.metadata)
    metadata["nfit_mask_count"] = int(np.count_nonzero(nfit_mask))
    metadata["combined_mask_count"] = int(np.count_nonzero(combined_reject))
    temperature = data.temperature.copy() if isinstance(data.temperature, np.ndarray) else data.temperature
    return PointData4D(
        H=data.H.copy(),
        K=data.K.copy(),
        L=data.L.copy(),
        E=data.E.copy(),
        intensity=data.intensity.copy(),
        sigma=data.sigma.copy(),
        mask=~combined_reject,
        temperature=temperature,
        magnetic_field=None if data.magnetic_field is None else np.array(data.magnetic_field),
        metadata=metadata,
        normalization_denominator=(
            None
            if data.normalization_denominator is None
            else np.array(data.normalization_denominator)
        ),
    )


def _nfit_mask_for_point_data(
    dataset: DatasetEntry,
    data: PointData4D,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> np.ndarray:
    combined = np.zeros(data.size, dtype=bool)
    for mask in [*(extra_masks or []), *dataset.masks]:
        if not mask.enabled:
            continue
        mask_values = _evaluate_point_data_mask(data, mask)
        if mask.invert:
            mask_values = ~mask_values
        if mask.additive:
            combined &= ~mask_values
        else:
            combined |= mask_values
    return combined


def _evaluate_point_data_mask(data: PointData4D, mask: MaskSpec) -> np.ndarray:
    if mask.type == "coordinate_range":
        return _point_data_coordinate_range_mask(data, mask.parameters)
    if mask.type == "energy_q_range":
        return _point_data_energy_q_range_mask(data, mask.parameters)
    if mask.type == "phonon_cone":
        return _point_data_phonon_cone_mask(data, mask.parameters)
    if mask.type == "box":
        return _point_data_box_mask(data, mask.parameters)
    if mask.type == "ellipsoid":
        return _point_data_ellipsoid_mask(data, mask.parameters)
    return np.zeros(data.size, dtype=bool)


def _point_data_coordinate_range_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.size, dtype=bool)
    active = False
    coords = _point_data_coordinate_values(data)
    for name in COORDINATE_RANGE_PARAMETER_NAMES:
        bounds = _exclusion_parameter_range(parameters.get(name))
        if bounds is None or name not in coords:
            continue
        active = True
        reject &= _values_in_range(coords[name], bounds)
    return reject if active else np.zeros(data.size, dtype=bool)


def _point_data_energy_q_range_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.size, dtype=bool)
    energy = _exclusion_parameter_range(parameters.get("energy"))
    q_modulus = _exclusion_parameter_range(parameters.get("q_modulus"))
    if energy is None and q_modulus is None:
        return np.zeros(data.size, dtype=bool)
    if energy is not None:
        reject &= _values_in_range(np.asarray(data.E, dtype=float), energy)
    if q_modulus is not None:
        reject &= _values_in_range(_point_data_q_modulus(data), q_modulus)
    return reject


def _point_data_box_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    resolved = _point_data_projected_region_inputs(data, parameters, extent_key="width")
    if resolved is None:
        return np.zeros(data.size, dtype=bool)
    values, center, width = resolved
    half_widths = [0.5 * value for value in width]
    if any(half_width <= 0.0 for half_width in half_widths):
        return np.zeros(data.size, dtype=bool)
    reject = np.ones(data.size, dtype=bool)
    for value, coordinate, half_width in zip(values, center, half_widths, strict=True):
        reject &= np.abs(value - coordinate) <= half_width
    return reject


def _point_data_ellipsoid_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    resolved = _point_data_projected_region_inputs(data, parameters, extent_key="radii")
    if resolved is None:
        return np.zeros(data.size, dtype=bool)
    values, center, radii = resolved
    if any(radius <= 0.0 for radius in radii):
        return np.zeros(data.size, dtype=bool)
    scaled_square = np.zeros(data.size, dtype=float)
    for value, coordinate, radius in zip(values, center, radii, strict=True):
        scaled = (value - coordinate) / radius
        scaled_square = scaled_square + scaled * scaled
    return scaled_square <= 1.0


def _point_data_projected_region_inputs(
    data: PointData4D,
    parameters: dict[str, Any],
    *,
    extent_key: str,
) -> tuple[list[np.ndarray], list[float], list[float]] | None:
    axes = _mask_axis_names(parameters)
    if not axes:
        return None
    center = _parameter_float_sequence(parameters.get("center"))
    extent = _parameter_float_sequence(parameters.get(extent_key))
    if center is None or extent is None:
        return None
    if not len(axes) == len(center) == len(extent):
        return None
    coords = _point_data_coordinate_values(data)
    values: list[np.ndarray] = []
    for name in axes:
        value = _resolve_projected_axis_grid(name, coords)
        if value is None:
            return None
        values.append(value)
    return values, center, extent


def _point_data_phonon_cone_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    slope = _parameter_float(parameters.get("slope"))
    if slope is None or slope <= 0.0:
        return np.zeros(data.size, dtype=bool)
    centers = _coordinate_centers(parameters.get("center"), 3)
    if centers is None:
        return np.zeros(data.size, dtype=bool)
    radius = max(_parameter_float(parameters.get("radius")) or 0.0, 0.0)
    q_vectors = _point_data_q_vectors(data)
    cone_radius = np.abs(np.asarray(data.E, dtype=float)) / slope + radius
    masked = np.zeros(data.size, dtype=bool)
    matrix = _mdhisto_q_matrix(data.metadata)
    centers_q = centers if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata) else centers @ matrix.T
    for center_q in centers_q:
        masked |= np.linalg.norm(q_vectors - center_q, axis=-1) <= cone_radius
    return masked


def _point_data_coordinate_values(data: PointData4D) -> dict[str, np.ndarray]:
    return {
        "H": np.asarray(data.H, dtype=float),
        "K": np.asarray(data.K, dtype=float),
        "L": np.asarray(data.L, dtype=float),
        "E": np.asarray(data.E, dtype=float),
    }


def _point_data_q_vectors(data: PointData4D) -> np.ndarray:
    hkl = np.column_stack([data.H, data.K, data.L])
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        return hkl
    return hkl @ _mdhisto_q_matrix(data.metadata).T


def _point_data_q_modulus(data: PointData4D) -> np.ndarray:
    return np.linalg.norm(_point_data_q_vectors(data), axis=1)


def _mdhisto_with_nfit_masks(
    dataset: DatasetEntry,
    *,
    data: MDHistoData | None = None,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData:
    data = dataset.data if data is None else data
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset does not contain MDHistoData")
    file_mask = np.asarray(data.mask, dtype=bool)
    nfit_mask = _nfit_mask_for_mdhisto(dataset, data, extra_masks=extra_masks)
    combined_mask = file_mask | nfit_mask
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask.copy()
    metadata["nfit_mask"] = nfit_mask.copy()
    metadata["nfit_mask_count"] = int(np.count_nonzero(nfit_mask))
    metadata["file_mask_count"] = int(np.count_nonzero(file_mask))
    metadata["combined_mask_count"] = int(np.count_nonzero(combined_mask))
    return MDHistoData(
        axes=data.axes,
        signal=data.signal,
        errors=data.errors,
        mask=combined_mask,
        num_events=data.num_events,
        coordinate_system=data.coordinate_system,
        visual_normalization=data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=data.auxiliary_channels,
    )


def _nfit_mask_for_mdhisto(
    dataset: DatasetEntry,
    data: MDHistoData,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> np.ndarray:
    combined = np.zeros(data.shape, dtype=bool)
    # Ancestor group masks apply first, then the dataset's own masks.
    for mask in [*(extra_masks or []), *dataset.masks]:
        if not mask.enabled:
            continue
        mask_values = _evaluate_mdhisto_mask(data, mask)
        if mask.invert:
            mask_values = ~mask_values
        if mask.additive:
            combined &= ~mask_values
        else:
            combined |= mask_values
    return combined


def _evaluate_mdhisto_mask(data: MDHistoData, mask: MaskSpec) -> np.ndarray:
    if mask.type == "coordinate_range":
        return _mdhisto_coordinate_range_mask(data, mask.parameters)
    if mask.type == "energy_q_range":
        return _mdhisto_energy_q_range_mask(data, mask.parameters)
    if mask.type == "phonon_cone":
        return _mdhisto_phonon_cone_mask(data, mask.parameters)
    if mask.type == "box":
        return _mdhisto_box_mask(data, mask.parameters)
    if mask.type == "ellipsoid":
        return _mdhisto_ellipsoid_mask(data, mask.parameters)
    return np.zeros(data.shape, dtype=bool)


def _mdhisto_box_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    """Mask a projected box, mirroring :func:`nfit.fitting.mask_out_box`.

    ``axes`` names the projected coordinates, ``center`` locates the box, and
    ``width`` gives the full extent along each axis. A point is masked when it
    falls inside the box along every listed axis; the zero-width default (see
    MASK_TYPE_DEFINITIONS) therefore masks nothing.
    """

    resolved = _mdhisto_projected_region_inputs(data, parameters, extent_key="width")
    if resolved is None:
        return np.zeros(data.shape, dtype=bool)
    grids, center, width = resolved
    half_widths = [0.5 * value for value in width]
    if any(half_width <= 0.0 for half_width in half_widths):
        return np.zeros(data.shape, dtype=bool)
    reject = np.ones(data.shape, dtype=bool)
    for grid, coordinate, half_width in zip(grids, center, half_widths, strict=True):
        reject &= np.abs(grid - coordinate) <= half_width
    return reject


def _mdhisto_ellipsoid_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    """Mask a projected ellipsoid, mirroring :func:`nfit.fitting.mask_out_ellipsoid`.

    ``axes`` names the projected coordinates, ``center`` locates the ellipsoid,
    and ``radii`` gives the radius along each axis. The zero-radius default
    masks nothing.
    """

    resolved = _mdhisto_projected_region_inputs(data, parameters, extent_key="radii")
    if resolved is None:
        return np.zeros(data.shape, dtype=bool)
    grids, center, radii = resolved
    if any(radius <= 0.0 for radius in radii):
        return np.zeros(data.shape, dtype=bool)
    scaled_square = np.zeros(data.shape, dtype=float)
    for grid, coordinate, radius in zip(grids, center, radii, strict=True):
        scaled = (grid - coordinate) / radius
        scaled_square = scaled_square + scaled * scaled
    return scaled_square <= 1.0


def _mdhisto_projected_region_inputs(
    data: MDHistoData,
    parameters: dict[str, Any],
    *,
    extent_key: str,
) -> tuple[list[np.ndarray], list[float], list[float]] | None:
    """Resolve the projected axis grids, center, and extent for box/ellipsoid masks.

    Returns ``None`` when the parameters are incomplete, mismatched in length,
    or name coordinates that cannot be projected onto the dataset axes, so the
    caller can fall back to masking nothing.
    """

    axes = _mask_axis_names(parameters)
    if not axes:
        return None
    center = _parameter_float_sequence(parameters.get("center"))
    extent = _parameter_float_sequence(parameters.get(extent_key))
    if center is None or extent is None:
        return None
    if not len(axes) == len(center) == len(extent):
        return None
    # Range-axis grids supply projected/leftover axis names, but the true
    # reciprocal coordinates (H/K/L/E) must win when a name refers to them.
    coords = _mdhisto_coordinate_range_axis_grids(data, parameters)
    coords.update(_mdhisto_coordinate_grids(data))
    grids: list[np.ndarray] = []
    for name in axes:
        grid = _resolve_projected_axis_grid(name, coords)
        if grid is None:
            return None
        grids.append(grid)
    return grids, center, extent


def _mask_axis_names(parameters: dict[str, Any]) -> list[Any]:
    axes = parameters.get("axes")
    if isinstance(axes, str):
        axes = _parse_parameter_text(axes)
    if not isinstance(axes, (list, tuple)):
        return []
    return list(axes)


def _parameter_float_sequence(value: Any) -> list[float] | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    result: list[float] = []
    for item in value:
        number = _parameter_float(item)
        if number is None:
            return None
        result.append(number)
    return result


def _resolve_projected_axis_grid(name: Any, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    """Resolve a projected axis name (or vector) to a coordinate grid."""

    if isinstance(name, str):
        key = name.strip()
        if key in coords:
            return coords[key]
        if key.upper() in coords:
            return coords[key.upper()]
        projection = _axis_projection_vector(key)
        if projection is not None:
            return _project_hkl_vector(projection, coords)
        parsed = _parse_parameter_text(key)
        if isinstance(parsed, (list, tuple)):
            return _project_hkle_vector(parsed, coords)
        return None
    if isinstance(name, (list, tuple, np.ndarray)):
        return _project_hkle_vector(name, coords)
    return None


def _project_hkl_vector(vector: np.ndarray, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    if not {"H", "K", "L"}.issubset(coords):
        return None
    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    return hkl @ np.asarray(vector, dtype=float)


def _project_hkle_vector(vector: Any, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    try:
        weights = np.asarray(vector, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(weights)):
        return None
    if weights.shape == (3,):
        return _project_hkl_vector(weights, coords)
    if weights.shape == (4,) and {"H", "K", "L", "E"}.issubset(coords):
        hkle = np.stack([coords["H"], coords["K"], coords["L"], coords["E"]], axis=-1)
        return hkle @ weights
    return None


def _mdhisto_phonon_cone_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    slope = _parameter_float(parameters.get("slope"))
    if slope is None or slope <= 0.0:
        # A non-positive slope leaves the starter mask inert (see MASK_TYPE_DEFINITIONS).
        return np.zeros(data.shape, dtype=bool)
    centers = _coordinate_centers(parameters.get("center"), 3)
    if centers is None:
        return np.zeros(data.shape, dtype=bool)
    coords = _mdhisto_coordinate_grids(data)
    if not {"H", "K", "L", "E"}.issubset(coords):
        return np.zeros(data.shape, dtype=bool)
    radius = _parameter_float(parameters.get("radius")) or 0.0
    radius = max(radius, 0.0)

    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        q_vectors = hkl
        centers_q = centers
    else:
        matrix = _mdhisto_q_matrix(data.metadata)
        q_vectors = np.einsum("ij,...j->...i", matrix, hkl)
        centers_q = centers @ matrix.T

    cone_radius = np.abs(coords["E"]) / slope + radius
    masked = np.zeros(data.shape, dtype=bool)
    for center_q in centers_q:
        masked |= np.linalg.norm(q_vectors - center_q, axis=-1) <= cone_radius
    return masked


def _coordinate_centers(value: Any, ndim: int) -> np.ndarray | None:
    """Parse one coordinate vector or a nonempty list of coordinate vectors."""

    if isinstance(value, str):
        value = _parse_parameter_text(value)
    try:
        centers = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if centers.shape == (ndim,):
        centers = centers.reshape(1, ndim)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != ndim:
        return None
    return centers if np.all(np.isfinite(centers)) else None


def _parameter_float(value: Any) -> float | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mdhisto_coordinate_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.shape, dtype=bool)
    active = False
    coords = _mdhisto_coordinate_grids(data)
    coords.update(_mdhisto_coordinate_range_axis_grids(data, parameters))
    for name in COORDINATE_RANGE_PARAMETER_NAMES:
        bounds = _exclusion_parameter_range(parameters.get(name))
        if bounds is None or name not in coords:
            continue
        active = True
        reject &= _values_in_range(coords[name], bounds)
    return reject if active else np.zeros(data.shape, dtype=bool)


def _mdhisto_energy_q_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.shape, dtype=bool)
    energy = _exclusion_parameter_range(parameters.get("energy"))
    q_modulus = _exclusion_parameter_range(parameters.get("q_modulus"))
    coords = _mdhisto_coordinate_grids(data)
    if energy is None and q_modulus is None:
        return np.zeros(data.shape, dtype=bool)
    if energy is not None:
        if "E" not in coords:
            if not _range_is_unrestricted(energy):
                return np.zeros(data.shape, dtype=bool)
        else:
            reject &= _values_in_range(coords["E"], energy)
    if q_modulus is not None:
        q_values = coords.get("q_modulus")
        if q_values is None:
            q_values = _mdhisto_q_modulus_grid(data, coords)
        reject &= _values_in_range(q_values, q_modulus)
    return reject


def _parameter_range(value: Any) -> tuple[float | None, float | None] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = _parse_parameter_text(value)
        if value in (None, ""):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lower = None if value[0] in (None, "") else float(value[0])
    upper = None if value[1] in (None, "") else float(value[1])
    return lower, upper


def _exclusion_parameter_range(value: Any) -> tuple[float | None, float | None] | None:
    """Return an active exclusion interval; [0, 0] is the neutral GUI default."""

    bounds = _parameter_range(value)
    if bounds is None:
        return None
    if bounds[0] == 0.0 and bounds[1] == 0.0:
        return None
    return bounds


def _values_in_range(values: np.ndarray, bounds: tuple[float | None, float | None]) -> np.ndarray:
    lower, upper = bounds
    selected = np.ones(values.shape, dtype=bool)
    if lower is not None:
        selected &= values >= lower - _range_tolerance(lower)
    if upper is not None:
        selected &= values <= upper + _range_tolerance(upper)
    return selected


def _range_is_unrestricted(bounds: tuple[float | None, float | None]) -> bool:
    lower, upper = bounds
    return lower is None and upper is None


def _range_tolerance(value: float) -> float:
    return max(1.0, abs(float(value))) * 1.0e-12


def _mdhisto_coordinate_range_axis_grids(
    data: MDHistoData,
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    specs = _coordinate_range_axis_specs(data)
    if not specs:
        return {}
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    source_vectors = _mdhisto_rebin_source_axis_vectors(data)
    coords: dict[str, np.ndarray] = {}
    for index, spec in enumerate(specs):
        key = f"{COORDINATE_RANGE_AXIS_PREFIX}{index}"
        vector = _coordinate_axis_vector(parameters.get(key), len(data.axes))
        if vector is None:
            vector = np.asarray(spec["vector"], dtype=float)
        matched_grid = next(
            (
                axis_grid
                for source_vector, axis_grid in zip(source_vectors, axis_values, strict=True)
                if source_vector is not None and np.allclose(vector, source_vector)
            ),
            None,
        )
        if matched_grid is not None:
            coords[str(spec["name"])] = np.asarray(matched_grid, dtype=float)
            continue
        values = np.zeros(data.shape, dtype=float)
        for axis_weight, axis_grid in zip(vector, axis_values, strict=True):
            if axis_weight:
                values = values + float(axis_weight) * axis_grid
        coords[str(spec["name"])] = values
    return coords


def _coordinate_range_axis_specs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, MDHistoData):
        return _mdhisto_coordinate_range_axis_specs(data)
    if isinstance(data, PointData4D):
        names = ["H", "K", "L", "E"]
        return [
            {"name": name, "vector": _identity_vector(index, len(names))}
            for index, name in enumerate(names)
        ]
    return []


def _mdhisto_coordinate_range_axis_specs(data: MDHistoData) -> list[dict[str, Any]]:
    ndim = len(data.axes)
    role_names = {"h": "H", "k": "K", "l": "L", "energy_transfer": "E"}
    assigned = {
        role_names[axis.role]
        for axis in data.axes
        if axis.role in role_names
    }
    remaining_names = iter(name for name in COORDINATE_RANGE_PARAMETER_NAMES if name not in assigned)
    specs = []
    for index, axis in enumerate(data.axes):
        name = role_names.get(axis.role)
        if name is None:
            name = next(remaining_names, str(axis.name or f"Axis {index}"))
        specs.append(
            {
                "name": name,
                "vector": _mdhisto_rebin_axis_vector(axis, index, ndim),
            }
        )
    return specs


def _identity_vector(index: int, ndim: int) -> list[float]:
    return [1.0 if axis_index == index else 0.0 for axis_index in range(ndim)]


def _clean_axis_weight(value: Any) -> float:
    """Round a coordinate-axis weight to drop floating-point noise (e.g. 0.5000000000000001 -> 0.5)."""

    rounded = round(float(value), 10)
    return rounded + 0.0


def _coordinate_axis_vector(value: Any, ndim: int) -> np.ndarray | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if not _is_vector_length(value, ndim):
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _is_vector_length(value: Any, length: int) -> bool:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == length


def _mdhisto_coordinate_grids(data: MDHistoData) -> dict[str, np.ndarray]:
    """Return physical HKLE grids, reconstructing them from any rebinned basis."""

    shape = data.shape
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    coords: dict[str, np.ndarray] = {}
    hkle = np.zeros((*shape, 4), dtype=float)
    hkle_contributions = 0
    for index, (axis, values) in enumerate(zip(data.axes, axis_values, strict=True)):
        if "metadata_dimension" in axis.metadata:
            continue
        role = axis.role
        if role == "q_modulus":
            coords["q_modulus"] = values
            continue
        vector = _mdhisto_axis_coordinate_vector(data, index)
        if vector is None:
            vector = {
                "h": np.array([1.0, 0.0, 0.0, 0.0]),
                "k": np.array([0.0, 1.0, 0.0, 0.0]),
                "l": np.array([0.0, 0.0, 1.0, 0.0]),
                "energy_transfer": np.array([0.0, 0.0, 0.0, 1.0]),
            }.get(role)
        if vector is not None:
            hkle += values[..., np.newaxis] * vector
            hkle_contributions += 1
    if hkle_contributions:
        coords.update({"H": hkle[..., 0], "K": hkle[..., 1], "L": hkle[..., 2]})
        if len(data.axes) >= 4 or np.any(hkle[..., 3] != 0.0):
            coords["E"] = hkle[..., 3]
    return coords


def _mdhisto_axis_coordinate_vector(data: MDHistoData, index: int) -> np.ndarray | None:
    """Return an output-axis vector in physical HKLE coordinates when known."""

    rebin = data.metadata.get("rebin") if isinstance(data.metadata, dict) else None
    vectors = rebin.get("vectors") if isinstance(rebin, dict) else None
    if isinstance(vectors, (list, tuple)) and 0 <= index < len(vectors):
        try:
            vector = np.asarray(vectors[index], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            vector = np.asarray([], dtype=float)
        if vector.size in {3, 4} and np.all(np.isfinite(vector)):
            return np.pad(vector, (0, 4 - vector.size))
    projection = _axis_projection_vector(data.axes[index].name)
    if projection is not None:
        return np.pad(projection, (0, 1))
    return None


def _axis_projection_vector(name: str) -> np.ndarray | None:
    match = re.search(r"\[([^\]]+)\]", str(name))
    if match is None:
        return None
    pieces = [piece.strip() for piece in match.group(1).split(",")]
    if len(pieces) != 3:
        return None
    return np.asarray([_projection_piece_value(piece) for piece in pieces], dtype=float)


def _projection_piece_value(piece: str) -> float:
    cleaned = piece.replace(" ", "")
    if cleaned in {"", "0"}:
        return 0.0
    if cleaned in {"H", "K", "L"}:
        return 1.0
    if cleaned in {"-H", "-K", "-L"}:
        return -1.0
    cleaned = re.sub(r"[HKL]", "", cleaned)
    if cleaned in {"", "+"}:
        return 1.0
    if cleaned == "-":
        return -1.0
    return float(cleaned)


def _mdhisto_q_modulus_grid(data: MDHistoData, coords: dict[str, np.ndarray]) -> np.ndarray:
    if not {"H", "K", "L"}.issubset(coords):
        raise ValueError("|Q| masks require H, K, and L coordinates")
    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        q_vectors = hkl
    else:
        matrix = _mdhisto_q_matrix(data.metadata)
        q_vectors = np.einsum("ij,...j->...i", matrix, hkl)
    return np.linalg.norm(q_vectors, axis=-1)


def _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(metadata: dict[str, Any]) -> bool:
    candidates = [metadata.get("coordinate_units"), metadata.get("momentum_units"), metadata.get("q_units")]
    return any("angstrom" in str(value).lower() and "rlu" not in str(value).lower() for value in candidates if value)


def _mdhisto_q_matrix(metadata: dict[str, Any]) -> np.ndarray:
    for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
        if key in metadata:
            matrix = np.asarray(metadata[key], dtype=float)
            return matrix if key == "rlu_to_inv_angstrom_matrix" or _matrix_includes_2pi(metadata, key) else 2.0 * np.pi * matrix
    oriented_lattice = metadata.get("oriented_lattice")
    if isinstance(oriented_lattice, dict):
        for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
            if key in oriented_lattice:
                matrix = np.asarray(oriented_lattice[key], dtype=float)
                return matrix if key == "rlu_to_inv_angstrom_matrix" or _matrix_includes_2pi(oriented_lattice, key) else 2.0 * np.pi * matrix
    lattice = metadata.get("lattice_parameters")
    if isinstance(lattice, dict):
        from .fitting import reciprocal_basis_from_lattice_parameters

        return reciprocal_basis_from_lattice_parameters(
            a=float(lattice["a"]),
            b=float(lattice["b"]),
            c=float(lattice["c"]),
            alpha=float(lattice.get("alpha", 90.0)),
            beta=float(lattice.get("beta", 90.0)),
            gamma=float(lattice.get("gamma", 90.0)),
            include_2pi=bool(lattice.get("include_2pi", True)),
        )
    raise ValueError("|Q| masks require inverse-angstrom coordinates or an attached lattice/UB matrix")


def _matrix_includes_2pi(metadata: dict[str, Any], key: str) -> bool:
    for flag_key in (f"{key}_includes_2pi", "q_matrix_includes_2pi", "include_2pi", "includes_2pi"):
        if flag_key in metadata:
            return bool(metadata[flag_key])
    lattice = metadata.get("lattice_parameters")
    return bool(isinstance(lattice, dict) and lattice.get("include_2pi"))


def _dataset_data_point_count(dataset: DatasetEntry) -> int:
    data = dataset.data
    if data is not None:
        count = _loaded_data_point_count(data)
        if dataset.data_matches_source:
            dataset.metadata["source_point_count"] = count
        return count
    stored_count = dataset.metadata.get("source_point_count")
    try:
        count = int(stored_count)
    except (TypeError, ValueError):
        count = -1
    if count >= 0:
        return count
    importer = IMPORTERS.get(str(dataset.metadata.get("importer", "")))
    source = dataset.metadata.get("source_file")
    if importer is not None and importer.point_counter is not None and source:
        try:
            options = dataset.metadata.get("import_options")
            count = int(
                importer.point_counter(
                    source, options if isinstance(options, dict) else None
                )
            )
        except (OSError, TypeError, ValueError):
            count = -1
        if count >= 0:
            dataset.metadata["source_point_count"] = count
            return count
    if isinstance(dataset.metadata.get(DERIVED_RECIPE_KEY), dict):
        return _dataset_rebin_output_bins(dataset_rebin_config(dataset))
    return 0
