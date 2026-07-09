from __future__ import annotations

import json
import copy
import ast
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .fitting import rebin_point_data
from .importers import IMPORTERS, import_with, importers_for_data_type
from .mdhisto import MDHistoAxis, MDHistoData, load_mantid_mdhisto_nxs
from .pipeline import DataGroup, DatasetEntry, DatasetGroup, FitTimelineEntry, MaskSpec, ModelComponentSpec
from .rebin import rebin_nd

QtMDHistoSliceViewer = None
RECENT_PROJECT_LIMIT = 10
RECENT_PROJECTS_KEY = "recent_projects"
DATASET_REBIN_KEY = "rebin"
DATASET_POINT_LIST_KEY = "point_list"
SUSCEPTIBILITY_CHANNEL_LABEL = "Susceptibility"
Q_COORDINATE_NAME = "q"
D_SPACING_COORDINATE_NAME = "d"
COORDINATE_RANGE_AXIS_PREFIX = "axis_"
COORDINATE_RANGE_PARAMETER_NAMES = ("H", "K", "L", "E")


# Data types the GUI can attach to a dataset. ``container`` is "mdhisto" for
# gridded neutron data loaded from Mantid ``.nxs`` (the existing path) or
# "point_list" for tabular point data loaded by a registered importer. Types
# without importers are selectable but fall back to the ``.nxs``/MDHisto loader.
DATA_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "single_crystal_inelastic": {
        "label": "Single crystal inelastic",
        "container": "mdhisto",
    },
    "powder_inelastic": {
        "label": "Powder inelastic",
        "container": "mdhisto",
    },
    "single_crystal_elastic": {
        "label": "Single crystal elastic",
        "container": "mdhisto",
    },
    "powder_elastic": {
        "label": "Powder elastic",
        "container": "point_list",
        "wavelength": True,
    },
    "magnetization": {
        "label": "Magnetization",
        "container": "point_list",
        "scale": True,
        "susceptibility": True,
    },
}

DEFAULT_DATA_TYPE = "single_crystal_inelastic"


MASK_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "coordinate_range": {
        "label": "Coordinate range",
        "parameters": {
            "H": {
                "default": [-1.0e99, 1.0e99],
                "description": "Inclusive reciprocal-lattice H range to keep. Values outside the range are masked.",
                "allowed": "Two numbers [min, max], or an empty string to leave H unrestricted.",
                "type": "list[float] | string",
                "example": "[-0.5, 0.5]",
            },
            "K": {
                "default": [-1.0e99, 1.0e99],
                "description": "Inclusive reciprocal-lattice K range to keep. Values outside the range are masked.",
                "allowed": "Two numbers [min, max], or an empty string to leave K unrestricted.",
                "type": "list[float] | string",
                "example": "[-0.5, 0.5]",
            },
            "L": {
                "default": [-1.0e99, 1.0e99],
                "description": "Inclusive reciprocal-lattice L range to keep. Values outside the range are masked.",
                "allowed": "Two numbers [min, max], or an empty string to leave L unrestricted.",
                "type": "list[float] | string",
                "example": "[0, 4]",
            },
            "E": {
                "default": [-1.0e99, 1.0e99],
                "description": "Inclusive energy-transfer range to keep. Values outside the range are masked.",
                "allowed": "Two numbers [min, max] in meV, or an empty string to leave energy unrestricted.",
                "type": "list[float] | string",
                "example": "[-2, 30]",
            },
        },
    },
    "energy_q_range": {
        "label": "Energy / |Q| range",
        "parameters": {
            "energy": {
                "default": [-1.0e99, 1.0e99],
                "description": "Energy-transfer interval for an energy-|Q| exclusion mask.",
                "allowed": "Two numbers [min, max] in meV. Use a very broad range for the no-op default.",
                "type": "list[float]",
                "example": "[-2, 1]",
            },
            "q_modulus": {
                "default": [0.0, 1.0e99],
                "description": "Magnitude of Q interval for an energy-|Q| exclusion mask.",
                "allowed": "Two numbers [min, max] in inverse angstrom; |Q| is non-negative so the lower bound defaults to 0.",
                "type": "list[float]",
                "example": "[0.2, 1.5]",
            },
        },
    },
    "box": {
        "label": "Projected box",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Center of the projected box in the coordinates named by axes.",
                "allowed": "List of numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0, 0, 0, 10]",
            },
            "width": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Full box width along each projected axis. A zero-width default masks no data.",
                "allowed": "Non-negative numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0.2, 0.2, 0.2, 1.0]",
            },
            "axes": {
                "default": ["H", "K", "L", "E"],
                "description": "Coordinate names used by the projected box.",
                "allowed": "List containing coordinate names such as H, K, L, E, or supported projected axes.",
                "type": "list[str]",
                "example": '["H", "K", "L", "E"]',
            },
        },
    },
    "ellipsoid": {
        "label": "Projected ellipsoid",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Center of the projected ellipsoid in the coordinates named by axes.",
                "allowed": "List of numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0, 0, 0, 10]",
            },
            "radii": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Ellipsoid radius along each projected axis. A zero-radius default masks no data.",
                "allowed": "Non-negative numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0.2, 0.2, 0.2, 1.0]",
            },
            "axes": {
                "default": ["H", "K", "L", "E"],
                "description": "Coordinate names used by the projected ellipsoid.",
                "allowed": "List containing coordinate names such as H, K, L, E, or supported projected axes.",
                "type": "list[str]",
                "example": '["H", "K", "L", "E"]',
            },
        },
    },
    "phonon_cone": {
        "label": "Phonon cone",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0],
                "description": "Reciprocal-space center of the acoustic phonon cone.",
                "allowed": "Three numbers [H, K, L] in reciprocal lattice units unless center_units later says otherwise.",
                "type": "list[float]",
                "example": "[2, 2, 2]",
            },
            "slope": {
                "default": 0.0,
                "description": "Cone slope dE/d|Q|. The zero default makes the starter mask effectively inert.",
                "allowed": "Non-negative number in meV per inverse angstrom.",
                "type": "float",
                "example": "35.0",
            },
            "radius": {
                "default": 0.0,
                "description": "Additional reciprocal-space radius around the cone. The zero default masks no volume by itself.",
                "allowed": "Non-negative number in inverse angstrom.",
                "type": "float",
                "example": "0.15",
            },
        },
    },
}


MODEL_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "constant_background": {
        "label": "Constant background",
        "description": "Measured-intensity background that is constant across Q and energy.",
        "parameters": {
            "constant": {
                "default": 0.0,
                "description": "Flat measured-intensity offset added to every point.",
                "allowed": "Any finite number in the dataset intensity units.",
                "type": "float",
                "example": "0.1",
                "global_fit": True,
            },
        },
        "config": {},
    },
    "linear_background": {
        "label": "Linear background",
        "description": "Measured-intensity background linear in energy transfer.",
        "parameters": {
            "c0": {
                "default": 0.0,
                "description": "Energy-independent background offset.",
                "allowed": "Any finite number in the dataset intensity units.",
                "type": "float",
                "example": "0.1",
                "global_fit": True,
            },
            "c1": {
                "default": 0.0,
                "description": "Slope of the background versus energy transfer.",
                "allowed": "Any finite number in intensity units per meV.",
                "type": "float",
                "example": "0.02",
                "global_fit": True,
            },
        },
        "config": {},
    },
    "single_q_paramagnon": {
        "label": "Single-Q paramagnon",
        "description": "Overdamped paramagnon chi'' centered at one reciprocal-space position.",
        "parameters": {
            "amplitude": {
                "default": 1.0,
                "description": "Overall spectral amplitude before cross-section and scale factors.",
                "allowed": "Non-negative finite number.",
                "type": "float",
                "example": "4.0",
                "global_fit": True,
            },
            "q0_h": {
                "default": 0.0,
                "description": "H coordinate of the paramagnon center.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.5",
                "global_fit": True,
            },
            "q0_k": {
                "default": 0.0,
                "description": "K coordinate of the paramagnon center.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.5",
                "global_fit": True,
            },
            "q0_l": {
                "default": 0.0,
                "description": "L coordinate of the paramagnon center.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.0",
                "global_fit": True,
            },
            "kappa": {
                "default": 1.0,
                "description": "Reciprocal-space width controlling the Q falloff.",
                "allowed": "Positive finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.22",
                "global_fit": True,
            },
            "omega_sf": {
                "default": 1.0,
                "description": "Spin-fluctuation energy scale.",
                "allowed": "Positive finite number in meV.",
                "type": "float",
                "example": "3.0",
                "global_fit": True,
            },
        },
        "config": {
            "cross_section": {
                "default": "magnetic",
                "description": "Calculation convention used when turning model susceptibility into measured intensity.",
                "allowed": "String naming a supported calculation mode. The initial supported value is magnetic.",
                "type": "str",
                "example": "magnetic",
            },
        },
    },
}


@dataclass
class MetallixProject:
    """Serializable workspace state for the project explorer GUI."""

    data_groups: list[DataGroup] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def create_data_group(project: MetallixProject, name: str | None = None) -> DataGroup:
    """Add a data group to a project and return it."""

    group = DataGroup(name=next_data_group_name(project.data_groups) if name is None else name)
    if group.name in {existing.name for existing in project.data_groups}:
        raise ValueError(f"duplicate data group name {group.name!r}")
    project.data_groups.append(group)
    return group


def available_data_types() -> list[tuple[str, str]]:
    """Return ``(type, label)`` pairs for every registered data type."""

    return [(name, definition["label"]) for name, definition in DATA_TYPE_DEFINITIONS.items()]


def data_type_label(data_type: str) -> str:
    """Return the human-readable label for a data type."""

    definition = DATA_TYPE_DEFINITIONS.get(data_type)
    return definition["label"] if definition else (data_type or "-")


def data_type_container(data_type: str) -> str:
    """Return the container kind ("mdhisto" or "point_list") for a data type."""

    definition = DATA_TYPE_DEFINITIONS.get(data_type, {})
    return str(definition.get("container", "mdhisto"))


def default_importer_for_data_type(data_type: str) -> str | None:
    """Return the default importer name for a data type, or ``None``."""

    specs = importers_for_data_type(data_type)
    return specs[0].name if specs else None


def import_dataset_paths(
    group: DataGroup,
    paths: list[str | Path],
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
    into: DatasetGroup | None = None,
) -> list[DatasetEntry]:
    """Add dataset entries for one or more source files.

    Point-list types are loaded eagerly with their importer; MDHisto/``.nxs``
    types stay as lazy placeholders loaded on first view. ``into`` optionally
    places the datasets inside a nested dataset group instead of the group root.
    """

    entries: list[DatasetEntry] = []
    for path in paths:
        source = Path(path)
        entry = dataset_entry_from_path(source, data_type=data_type, importer_name=importer_name)
        entry.name = _unique_dataset_name(entry.name, group.dataset_names)
        group.add_dataset(entry, into=into)
        entries.append(entry)
    return entries


def delete_data_group(project: MetallixProject, group: DataGroup) -> None:
    """Remove a data group from a project."""

    project.data_groups.remove(group)


def delete_dataset(group: DataGroup, dataset: DatasetEntry) -> None:
    """Remove a dataset entry from anywhere in the group tree."""

    parent = _dataset_parent_node(group, dataset)
    if parent is None:
        raise ValueError(f"dataset {dataset.name!r} is not in group {group.name!r}")
    parent.datasets.remove(dataset)


def set_dataset_source(dataset: DatasetEntry, path: str | Path) -> None:
    """Point a dataset entry at a new source file and mark loaded data stale."""

    source = Path(path)
    dataset.metadata["source_file"] = str(source)
    dataset.metadata["import_status"] = "pending"
    dataset.kind = source.suffix.lstrip(".").lower()
    dataset.data = None
    if data_type_container(dataset.data_type) == "point_list":
        _load_point_list_dataset(dataset)


def set_dataset_data_type(
    dataset: DatasetEntry,
    data_type: str,
    *,
    importer_name: str | None = None,
) -> None:
    """Change a dataset's data type and reload/reset its data accordingly."""

    if data_type not in DATA_TYPE_DEFINITIONS:
        raise ValueError(f"unknown data type {data_type!r}")
    dataset.data_type = data_type
    dataset.metadata.pop("import_error", None)
    if data_type_container(data_type) == "point_list":
        chosen = importer_name or default_importer_for_data_type(data_type)
        if chosen is not None:
            dataset.metadata["importer"] = chosen
        dataset.data = None
        if dataset.metadata.get("source_file"):
            # Reload with the new type's importer, but a mismatched importer must
            # not crash the type switch; record the error for the details panel.
            try:
                _load_point_list_dataset(dataset)
            except Exception as exc:
                dataset.data = None
                dataset.metadata["import_status"] = "error"
                dataset.metadata["import_error"] = str(exc)
    else:
        dataset.metadata.pop("importer", None)
        # Fall back to the lazy MDHisto/.nxs loader on next view.
        if not isinstance(dataset.data, MDHistoData):
            dataset.data = None
            dataset.metadata["import_status"] = "pending"


def _load_point_list_dataset(dataset: DatasetEntry) -> PointListData | None:
    """Load a point-list dataset from its source file using a registered importer."""

    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if not source:
        return None
    importer_name = dataset.metadata.get("importer") or default_importer_for_data_type(dataset.data_type)
    if importer_name is None:
        return None
    data = import_with(importer_name, source)
    dataset.data = data
    dataset.metadata["importer"] = importer_name
    dataset.metadata["import_status"] = "loaded"
    if not dataset.kind:
        dataset.kind = Path(source).suffix.lstrip(".").lower()
    return data


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


def prepared_point_list_data(dataset: DatasetEntry) -> PointListData:
    """Return the point-list data after applying role overrides and transforms."""

    if not isinstance(dataset.data, PointListData):
        raise TypeError("dataset does not contain PointListData")
    config = point_list_config(dataset)
    base = dataset.data
    columns = {name: np.array(values, dtype=float) for name, values in base.columns.items()}
    units = dict(base.units)

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

    definition = DATA_TYPE_DEFINITIONS.get(dataset.data_type, {})

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
        if susc.get("enabled"):
            field_name = susc.get("field")
            moment_label = susc.get("moment")
            moment_channel = next((c for c in channels if c.get("label") == moment_label), None)
            if moment_channel is not None and field_name in columns:
                field = columns[field_name]
                with np.errstate(divide="ignore", invalid="ignore"):
                    susc_value = columns[moment_channel["value"]] / field
                value_col = f"{SUSCEPTIBILITY_CHANNEL_LABEL} value"
                columns[value_col] = susc_value
                moment_unit = units.get(moment_channel["value"], "")
                field_unit = units.get(field_name, "")
                susc_unit = f"{moment_unit}/{field_unit}" if moment_unit and field_unit else ""
                units[value_col] = susc_unit
                error_name = moment_channel.get("error")
                error_col = None
                if error_name in columns:
                    error_col = f"{SUSCEPTIBILITY_CHANNEL_LABEL} error"
                    with np.errstate(divide="ignore", invalid="ignore"):
                        columns[error_col] = columns[error_name] / np.abs(field)
                    units[error_col] = susc_unit
                channels.append(
                    {"label": SUSCEPTIBILITY_CHANNEL_LABEL, "value": value_col, "error": error_col}
                )

    return PointListData(
        columns=columns,
        units=units,
        coordinate_names=coordinate_names,
        channels=channels,
        metadata=dict(base.metadata),
    )


def create_mask(dataset: DatasetEntry, name: str | None = None, *, type: str = "coordinate_range") -> MaskSpec:
    """Add a mask spec to a dataset and return it."""

    if type not in MASK_TYPE_DEFINITIONS:
        raise ValueError(f"unknown mask type {type!r}")
    mask = MaskSpec(
        name=next_mask_name(dataset.masks) if name is None else name,
        type=type,
        parameters=default_mask_parameters(type),
    )
    ensure_coordinate_range_mask_axes(mask, dataset)
    if mask.name in {existing.name for existing in dataset.masks}:
        raise ValueError(f"duplicate mask name {mask.name!r}")
    dataset.masks.append(mask)
    return mask


def create_group_mask(
    subgroup: DatasetGroup,
    reference_dataset: DatasetEntry | None = None,
    name: str | None = None,
    *,
    type: str = "coordinate_range",
) -> MaskSpec:
    """Add a shared mask to a nested dataset group, applied to all descendants."""

    if type not in MASK_TYPE_DEFINITIONS:
        raise ValueError(f"unknown mask type {type!r}")
    mask = MaskSpec(
        name=next_mask_name(subgroup.masks) if name is None else name,
        type=type,
        parameters=default_mask_parameters(type),
    )
    if reference_dataset is not None:
        ensure_coordinate_range_mask_axes(mask, reference_dataset)
    if mask.name in {existing.name for existing in subgroup.masks}:
        raise ValueError(f"duplicate mask name {mask.name!r}")
    subgroup.masks.append(mask)
    return mask


def _group_reference_dataset(subgroup: DatasetGroup) -> DatasetEntry | None:
    """Return the first descendant dataset with MDHisto data, for mask axis inference."""

    for dataset in subgroup.iter_datasets():
        if isinstance(dataset.data, MDHistoData):
            return dataset
    return None


def delete_mask(dataset: DatasetEntry, mask: MaskSpec) -> None:
    """Remove a mask from a dataset."""

    dataset.masks.remove(mask)


def next_dataset_group_name(existing_names: Any) -> str:
    """Return the next available default subgroup name."""

    taken = set(existing_names)
    index = 1
    while f"Group{index}" in taken:
        index += 1
    return f"Group{index}"


def create_dataset_group(
    data_group: DataGroup,
    parent_node: Any,
    name: str | None = None,
) -> DatasetGroup:
    """Create a nested dataset group under ``parent_node`` (a DataGroup or DatasetGroup)."""

    existing = {sub.name for sub in data_group.iter_subgroups()}
    subgroup = DatasetGroup(name=name or next_dataset_group_name(existing))
    if subgroup.name in existing:
        raise ValueError(f"duplicate dataset group name {subgroup.name!r}")
    parent_node.subgroups.append(subgroup)
    return subgroup


def delete_dataset_group(data_group: DataGroup, subgroup: DatasetGroup) -> bool:
    """Remove a nested dataset group (and its contents) from the group tree."""

    def remove_from(node: Any) -> bool:
        if subgroup in node.subgroups:
            node.subgroups.remove(subgroup)
            return True
        return any(remove_from(child) for child in node.subgroups)

    return remove_from(data_group)


def _dataset_group_parent(data_group: DataGroup, subgroup: DatasetGroup) -> Any:
    """Return the node whose ``subgroups`` contains ``subgroup`` (DataGroup or DatasetGroup)."""

    def search(node: Any) -> Any:
        if subgroup in node.subgroups:
            return node
        for child in node.subgroups:
            found = search(child)
            if found is not None:
                return found
        return None

    return search(data_group)


def _dataset_parent_node(root: Any, dataset: DatasetEntry) -> Any:
    """Return the node whose ``datasets`` contains ``dataset`` (DataGroup or DatasetGroup)."""

    if dataset in root.datasets:
        return root
    for subgroup in root.subgroups:
        found = _dataset_parent_node(subgroup, dataset)
        if found is not None:
            return found
    return None


def _group_contains_node(ancestor: DatasetGroup, node: Any) -> bool:
    """Return True if ``node`` is ``ancestor`` or nested anywhere inside it."""

    return node is ancestor or node in ancestor.iter_subgroups()


def create_model_component(
    group: DataGroup,
    name: str | None = None,
    *,
    type: str = "constant_background",
) -> ModelComponentSpec:
    """Add a model component spec to a data group and return it."""

    if type not in MODEL_TYPE_DEFINITIONS:
        raise ValueError(f"unknown model type {type!r}")
    model = ModelComponentSpec(
        name=next_model_name(group.models),
        type=type,
        parameters=default_model_parameters(type),
        config=default_model_config(type),
        fit_parameters=default_model_fit_parameters(type),
        global_fit=default_model_global_fit(type),
    )
    if name is not None:
        model.name = name
    model.name = _unique_name(model.name, list(group.models))
    group.models[model.name] = model
    return model


def ensure_fit_history(group: DataGroup) -> list[FitTimelineEntry]:
    """Ensure a data group has an Initial fit-history state."""

    if not group.fits:
        group.fits.append(
            FitTimelineEntry(
                name="Initial",
                kind="initial",
                snapshot=snapshot_data_group_state(group),
                created_at=_timestamp_now(),
            )
        )
    return group.fits


def create_placeholder_fit_result(
    group: DataGroup,
    parent: FitTimelineEntry,
    *,
    branch_timeline: bool = False,
) -> FitTimelineEntry:
    """Create a placeholder fit result from current group state."""

    start = time.perf_counter()
    snapshot = snapshot_data_group_state(group)
    duration = time.perf_counter() - start
    result = FitTimelineEntry(
        name=next_fit_result_name(group.fits),
        kind="result",
        snapshot=copy.deepcopy(snapshot),
        created_at=_timestamp_now(),
        duration_seconds=duration,
        optimizer=str(parent.optimizer or "least_squares"),
        optimizer_config=copy.deepcopy(parent.optimizer_config),
        goodness={"status": "not run", "message": "Fit execution is not wired yet."},
    )
    if branch_timeline:
        timeline = FitTimelineEntry(
            name=next_fit_timeline_name(group.fits),
            kind="timeline",
            created_at=_timestamp_now(),
            children=[result, current_state_fit_entry(group)],
        )
        parent.children.append(timeline)
    else:
        siblings = _fit_siblings(group.fits, parent)
        if siblings is None:
            siblings = group.fits
        insert_at = siblings.index(parent) + 1 if parent in siblings else len(siblings)
        siblings.insert(insert_at, result)
        _replace_current_state(siblings, group)
    return result


def current_state_fit_entry(group: DataGroup) -> FitTimelineEntry:
    """Return a snapshot entry representing the current mutable GUI state."""

    return FitTimelineEntry(
        name="Current state",
        kind="current",
        snapshot=snapshot_data_group_state(group),
        created_at=_timestamp_now(),
    )


def refresh_current_state_fit_entries(group: DataGroup) -> None:
    """Refresh the top-level Current state node from live GUI state."""

    snapshot = snapshot_data_group_state(group)
    for entry in group.fits:
        if entry.kind == "current":
            entry.snapshot = copy.deepcopy(snapshot)
            entry.created_at = _timestamp_now()


def _named_group_nodes(group: DataGroup) -> list[tuple[str, Any]]:
    """Return ``(key, node)`` pairs for the data group ("") and each subgroup by name."""

    nodes: list[tuple[str, Any]] = [("", group)]
    nodes.extend((subgroup.name, subgroup) for subgroup in group.iter_subgroups())
    return nodes


def snapshot_data_group_state(group: DataGroup) -> dict[str, Any]:
    """Capture serializable dataset mask and model configuration state."""

    return {
        "datasets": [
            {
                "name": dataset.name,
                "parameters": copy.deepcopy(dataset.parameters),
                "enabled": bool(dataset.enabled),
                "fit_weight": float(dataset.fit_weight),
                "scale_factor": float(dataset.scale_factor),
                "masks": [_mask_to_dict(mask) for mask in dataset.masks],
            }
            for dataset in group.iter_datasets()
        ],
        "group_masks": {
            node_name: [_mask_to_dict(mask) for mask in node.masks]
            for node_name, node in _named_group_nodes(group)
        },
        "models": [
            _model_to_dict(model)
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
        ],
    }


def restore_data_group_state(group: DataGroup, snapshot: dict[str, Any]) -> None:
    """Restore dataset masks and model component settings from a snapshot."""

    datasets_by_name = {dataset.name: dataset for dataset in group.iter_datasets()}
    for dataset_payload in snapshot.get("datasets", []):
        dataset = datasets_by_name.get(str(dataset_payload.get("name", "")))
        if dataset is None:
            continue
        dataset.parameters = dict(dataset_payload.get("parameters", {}))
        dataset.enabled = bool(dataset_payload.get("enabled", dataset.enabled))
        dataset.fit_weight = float(dataset_payload.get("fit_weight", dataset.fit_weight))
        dataset.scale_factor = float(dataset_payload.get("scale_factor", dataset.scale_factor))
        dataset.masks = [_mask_from_dict(mask_payload) for mask_payload in dataset_payload.get("masks", [])]
    group_masks = snapshot.get("group_masks", {})
    if isinstance(group_masks, dict):
        for node_name, node in _named_group_nodes(group):
            if node_name in group_masks:
                node.masks = [_mask_from_dict(mask_payload) for mask_payload in group_masks[node_name]]
    for model_payload in snapshot.get("models", []):
        name = str(model_payload.get("name", ""))
        existing = group.models.get(name)
        if not isinstance(existing, ModelComponentSpec):
            continue
        existing.type = str(model_payload.get("type", existing.type))
        existing.parameters = dict(model_payload.get("parameters", {}))
        existing.config = dict(model_payload.get("config", {}))
        existing.fit_parameters = {
            str(key): bool(value)
            for key, value in dict(model_payload.get("fit_parameters", {})).items()
        }
        existing.global_fit = {
            str(key): bool(value)
            for key, value in dict(model_payload.get("global_fit", {})).items()
        }
        existing.enabled = bool(model_payload.get("enabled", existing.enabled))
        existing.metadata = dict(model_payload.get("metadata", {}))


def delete_fit_entry(group: DataGroup, fit_entry: FitTimelineEntry) -> bool:
    """Delete a fit-history entry from a group unless it is the only Initial state."""

    if fit_entry.kind == "initial" and len(group.fits) == 1:
        return False
    siblings = _fit_siblings(group.fits, fit_entry)
    if siblings is None:
        return False
    siblings.remove(fit_entry)
    if not group.fits:
        ensure_fit_history(group)
    return True


def delete_model_component(group: DataGroup, model: ModelComponentSpec) -> None:
    """Remove a model component spec from a data group."""

    for name, existing in list(group.models.items()):
        if existing is model:
            del group.models[name]
            return
    raise ValueError(f"model {model.name!r} is not in data group {group.name!r}")


def copy_mask_to_dataset(mask: MaskSpec, dataset: DatasetEntry) -> MaskSpec:
    """Copy a mask spec to another dataset, choosing a non-conflicting name."""

    copied = copy.deepcopy(mask)
    copied.name = _unique_name(copied.name, [existing.name for existing in dataset.masks])
    dataset.masks.append(copied)
    return copied


def _move_mask_within_dataset(
    dataset: DatasetEntry,
    mask: MaskSpec,
    target_mask: MaskSpec | None,
) -> bool:
    if mask not in dataset.masks:
        return False
    old_index = dataset.masks.index(mask)
    if target_mask is mask:
        return False
    dataset.masks.pop(old_index)
    if target_mask is None or target_mask not in dataset.masks:
        new_index = len(dataset.masks)
    else:
        new_index = dataset.masks.index(target_mask)
    dataset.masks.insert(new_index, mask)
    return old_index != new_index


def _move_items_within_list(items: list[Any], moving: list[Any], insert_index: int) -> bool:
    """Move selected objects inside one ordered list, preserving selection order."""

    moving_ids = {id(item) for item in moving}
    original = list(items)
    insert_index = max(0, min(insert_index, len(items)))
    insert_index -= sum(1 for index, item in enumerate(items) if id(item) in moving_ids and index < insert_index)
    items[:] = [item for item in items if id(item) not in moving_ids]
    for offset, item in enumerate(moving):
        items.insert(insert_index + offset, item)
    return items != original


def copy_dataset_to_group(dataset: DatasetEntry, group: DataGroup) -> DatasetEntry:
    """Copy a dataset entry to another data group, choosing a non-conflicting name."""

    copied = copy.deepcopy(dataset)
    copied.name = _unique_name(copied.name, group.dataset_names)
    group.add_dataset(copied)
    return copied


def available_mask_types() -> list[str]:
    """Return registered mask type names."""

    return list(MASK_TYPE_DEFINITIONS)


def available_model_types() -> list[str]:
    """Return registered model type names."""

    return list(MODEL_TYPE_DEFINITIONS)


def default_mask_parameters(type: str) -> dict[str, Any]:
    """Return default parameter values for a registered mask type."""

    return {
        name: copy.deepcopy(metadata["default"])
        for name, metadata in MASK_TYPE_DEFINITIONS[type]["parameters"].items()
    }


def ensure_coordinate_range_mask_axes(mask: MaskSpec, dataset: DatasetEntry | None) -> None:
    """Ensure a coordinate-range mask has one coordinate-axis vector per dataset dimension."""

    if mask.type != "coordinate_range" or dataset is None:
        return
    specs = _coordinate_range_axis_specs(dataset.data)
    if not specs:
        return
    stale_keys = [
        key
        for key in mask.parameters
        if key.startswith(COORDINATE_RANGE_AXIS_PREFIX)
        and key[len(COORDINATE_RANGE_AXIS_PREFIX) :].isdigit()
        and int(key[len(COORDINATE_RANGE_AXIS_PREFIX) :]) >= len(specs)
    ]
    for key in stale_keys:
        mask.parameters.pop(key, None)
    for index, spec in enumerate(specs):
        key = f"{COORDINATE_RANGE_AXIS_PREFIX}{index}"
        value = mask.parameters.get(key)
        if not _is_vector_length(value, len(spec["vector"])):
            mask.parameters[key] = copy.deepcopy(spec["vector"])


def mask_parameter_tooltip(type: str, parameter_name: str) -> str:
    """Return standard hover text for a mask parameter editor."""

    if type == "coordinate_range" and parameter_name.startswith(COORDINATE_RANGE_AXIS_PREFIX):
        axis_number = parameter_name.removeprefix(COORDINATE_RANGE_AXIS_PREFIX)
        return "\n".join(
            [
                f"Parameter: {parameter_name}",
                f"Description: Coordinate-axis vector {axis_number}. It defines one mask coordinate as a linear combination of the dataset axes.",
                "Allowed values: A list of N finite numbers, where N is the number of dataset dimensions.",
                "Data type: list[float]",
                "Default: identity basis vector for the corresponding dataset coordinate.",
                "Example: [1, 0, 0, 0]",
            ]
        )
    metadata = MASK_TYPE_DEFINITIONS[type]["parameters"][parameter_name]
    return "\n".join(
        [
            f"Parameter: {parameter_name}",
            f"Description: {metadata['description']}",
            f"Allowed values: {metadata['allowed']}",
            f"Data type: {metadata['type']}",
            f"Default: {_parameter_to_text(metadata['default'])}",
            f"Example: {metadata['example']}",
        ]
    )


def _mask_parameter_names(mask: MaskSpec, dataset: DatasetEntry | None = None) -> list[str]:
    names = list(MASK_TYPE_DEFINITIONS[mask.type]["parameters"])
    if mask.type != "coordinate_range":
        return names
    ensure_coordinate_range_mask_axes(mask, dataset)
    axis_names = sorted(
        (
            key
            for key in mask.parameters
            if key.startswith(COORDINATE_RANGE_AXIS_PREFIX)
            and key[len(COORDINATE_RANGE_AXIS_PREFIX) :].isdigit()
        ),
        key=lambda key: int(key[len(COORDINATE_RANGE_AXIS_PREFIX) :]),
    )
    return [*names, *axis_names]


def default_model_parameters(type: str) -> dict[str, Any]:
    """Return default parameter values for a registered model type."""

    return {
        name: copy.deepcopy(metadata["default"])
        for name, metadata in MODEL_TYPE_DEFINITIONS[type]["parameters"].items()
    }


def default_model_config(type: str) -> dict[str, Any]:
    """Return default non-optimizable configuration settings for a model type."""

    return {
        name: copy.deepcopy(metadata["default"])
        for name, metadata in MODEL_TYPE_DEFINITIONS[type].get("config", {}).items()
    }


def default_model_global_fit(type: str) -> dict[str, bool]:
    """Return default global-fit flags for a registered model type."""

    return {
        name: bool(metadata.get("global_fit", True))
        for name, metadata in MODEL_TYPE_DEFINITIONS[type]["parameters"].items()
    }


def default_model_fit_parameters(type: str) -> dict[str, bool]:
    """Return default optimizer-inclusion flags for a registered model type."""

    return {name: False for name in MODEL_TYPE_DEFINITIONS[type]["parameters"]}


def model_parameter_tooltip(type: str, parameter_name: str) -> str:
    """Return standard hover text for a model parameter editor."""

    metadata = MODEL_TYPE_DEFINITIONS[type]["parameters"][parameter_name]
    return "\n".join(
        [
            f"Parameter: {parameter_name}",
            f"Description: {metadata['description']}",
            f"Allowed values: {metadata['allowed']}",
            f"Data type: {metadata['type']}",
            f"Default: {_parameter_to_text(metadata['default'])}",
            f"Example: {metadata['example']}",
            "Fit: checked means the optimizer may vary this parameter; unchecked means it is fixed at the displayed value.",
            "Global fit: checked means one shared value is fitted across datasets; unchecked means each dataset may fit its own value.",
        ]
    )


def model_config_tooltip(type: str, setting_name: str) -> str:
    """Return standard hover text for a model configuration setting editor."""

    metadata = MODEL_TYPE_DEFINITIONS[type].get("config", {})[setting_name]
    return "\n".join(
        [
            f"Configuration setting: {setting_name}",
            f"Description: {metadata['description']}",
            f"Allowed values: {metadata['allowed']}",
            f"Data type: {metadata['type']}",
            f"Default: {_parameter_to_text(metadata['default'])}",
            f"Example: {metadata['example']}",
            "Configuration settings are fixed model options and are not optimized by the fitter.",
        ]
    )


def dataset_details_text(dataset: DatasetEntry, *, group: DataGroup | None = None) -> str:
    """Return a human-readable summary of one imported dataset."""

    lines: list[str] = []
    for title, section_lines in dataset_detail_sections(dataset, group=group):
        if lines:
            lines.append("")
        lines.append(title)
        lines.extend(section_lines)
    return "\n".join(lines)


def fit_details_text(fit_entry: FitTimelineEntry) -> str:
    """Return a human-readable summary of one fit-history entry."""

    lines = [
        f"Type: {fit_entry.kind}",
        f"Created: {fit_entry.created_at or '-'}",
        f"Optimizer: {fit_entry.optimizer or '-'}",
        f"Duration: {_format_number(fit_entry.duration_seconds) + ' s' if fit_entry.duration_seconds is not None else '-'}",
    ]
    if fit_entry.optimizer_config:
        lines.extend(["", "Optimizer config"])
        lines.extend(_mapping_lines(fit_entry.optimizer_config))
    if fit_entry.goodness:
        lines.extend(["", "Goodness of fit"])
        lines.extend(_mapping_lines(fit_entry.goodness))
    snapshot = fit_entry.snapshot or {}
    lines.extend(
        [
            "",
            "Snapshot",
            f"Datasets: {len(snapshot.get('datasets', []))}",
            f"Models: {len(snapshot.get('models', []))}",
        ]
    )
    if fit_entry.children:
        lines.extend(["", f"Timeline entries: {len(fit_entry.children)}"])
    return "\n".join(lines)


def dataset_detail_sections(
    dataset: DatasetEntry,
    *,
    group: DataGroup | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ordered dataset detail sections for text and GUI rendering."""

    axes_lines, data_lines = _dataset_axes_and_data_lines(dataset.data)
    data_lines.append(f"Masks: {len(dataset.masks)}")
    source_lines = _dataset_source_lines(dataset)
    metadata_lines = _dataset_metadata_lines(dataset)
    if dataset.parameters:
        metadata_lines = [*metadata_lines, "", "Parameters", *_mapping_lines(dataset.parameters)]
    return [
        (
            "Dataset",
            [
                f"Name: {dataset.name}",
                "Dataset",
                f"Data type: {data_type_label(dataset.data_type)}",
                f"Kind: {dataset.kind or '-'}",
                f"Enabled for fitting: {dataset.enabled}",
                f"Fit weight: {_format_number(dataset.fit_weight)}",
                f"Scale factor: {_format_number(dataset.scale_factor)}",
            ],
        ),
        ("Axes", axes_lines),
        ("Crystal", _dataset_crystal_lines(dataset, group)),
        ("Data", data_lines),
        ("Source", source_lines or ["No source file recorded."]),
        ("Metadata", metadata_lines or ["No additional metadata."]),
    ]


def effective_dataset_masks(group: DataGroup, dataset: DatasetEntry) -> list[MaskSpec]:
    """Return masks inherited from the ancestor group chain of ``dataset``.

    Walks from the data group down through nested dataset groups to the
    dataset's parent, collecting each level's shared masks (outer to inner).
    Returns an empty list when the dataset is not found.
    """

    def search(node: Any) -> list[MaskSpec] | None:
        if dataset in getattr(node, "datasets", []):
            return list(node.masks)
        for subgroup in getattr(node, "subgroups", []):
            below = search(subgroup)
            if below is not None:
                return list(node.masks) + below
        return None

    return search(group) or []


def slice_viewer_datasets(
    group: DataGroup,
) -> tuple[list[MDHistoData], list[str]]:
    """Return data and labels for every dataset in the group tree, with shared masks."""

    data: list[MDHistoData] = []
    names: list[str] = []
    for dataset in group.iter_datasets():
        extra_masks = effective_dataset_masks(group, dataset)
        view_data = dataset_for_slice_viewer(dataset, extra_masks=extra_masks)
        if view_data is not None:
            data.append(view_data)
            names.append(dataset.name)
    return data, names


def dataset_for_slice_viewer(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData | PointListData | None:
    """Return a viewer-ready dataset, loading from source metadata if needed.

    ``extra_masks`` are masks inherited from ancestor dataset groups; they are
    applied ahead of the dataset's own masks (MDHisto datasets only).
    """

    result = _viewer_data_before_scale(dataset, extra_masks=extra_masks)
    if result is None:
        return None
    return _with_viewer_dataset_metadata(dataset, _apply_dataset_scale(dataset, result))


def _with_viewer_dataset_metadata(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData,
) -> MDHistoData | PointListData:
    metadata = dict(getattr(data, "metadata", {}) or {})
    metadata["metallix_data_type"] = dataset.data_type
    metadata["metallix_dataset_kind"] = dataset.kind
    if isinstance(data, MDHistoData):
        return replace(data, metadata=metadata)
    if isinstance(data, PointListData):
        return PointListData(
            columns={name: np.array(values, dtype=float) for name, values in data.columns.items()},
            units=dict(data.units),
            coordinate_names=list(data.coordinate_names),
            channels=[dict(channel) for channel in data.channels],
            metadata=metadata,
        )
    return data


def _viewer_data_before_scale(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData | PointListData | None:
    if data_type_container(dataset.data_type) == "point_list" or isinstance(dataset.data, PointListData):
        if dataset.data is None:
            _load_point_list_dataset(dataset)
        if not isinstance(dataset.data, PointListData):
            return None
        if dataset_rebin_enabled(dataset):
            return rebinned_dataset_data(dataset)
        return prepared_point_list_data(dataset)
    if isinstance(dataset.data, MDHistoData):
        data = rebinned_dataset_data(dataset) if dataset_rebin_enabled(dataset) else dataset.data
        return _mdhisto_with_metallix_masks(dataset, data=data, extra_masks=extra_masks)
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if not source:
        return None
    source_path = Path(source)
    if source_path.suffix.lower() not in {".nxs", ".h5", ".hdf5"}:
        return None
    loaded = load_mantid_mdhisto_nxs(source_path, copy_metadata=False)
    dataset.data = loaded
    dataset.kind = dataset.kind or source_path.suffix.lstrip(".").lower()
    dataset.metadata["import_status"] = "loaded"
    data = rebinned_dataset_data(dataset) if dataset_rebin_enabled(dataset) else dataset.data
    return _mdhisto_with_metallix_masks(dataset, data=data, extra_masks=extra_masks)


def _apply_dataset_scale(
    dataset: DatasetEntry,
    data: MDHistoData | PointListData,
) -> MDHistoData | PointListData:
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
        )
    return data


def dataset_rebin_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return a dataset rebin configuration, creating default axis settings if needed."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.parameters[DATASET_REBIN_KEY] = config
    config.setdefault("enabled", False)
    config.setdefault("fractional", True)
    config["normalize"] = True
    axes = config.get("axes")
    # Point-list rebin binds over the transformed coordinates (e.g. a derived q).
    if isinstance(dataset.data, PointListData):
        default_axes = _default_rebin_axes(prepared_point_list_data(dataset))
    else:
        default_axes = _default_rebin_axes(dataset.data)
    if not isinstance(axes, list) or len(axes) != len(default_axes):
        config["axes"] = default_axes
    else:
        sanitized_axes = []
        for axis_config, default_axis in zip(axes, default_axes, strict=True):
            if not isinstance(axis_config, dict):
                axis_config = {}
            for key, value in default_axis.items():
                axis_config.setdefault(key, value)
            sanitized_axes.append(_sanitize_rebin_axis_config(axis_config))
        config["axes"] = sanitized_axes
    return config


def dataset_rebin_enabled(dataset: DatasetEntry) -> bool:
    """Return whether this dataset should use its rebinned representation."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def rebinned_dataset_data(dataset: DatasetEntry) -> Any:
    """Return a rebinned copy of a supported dataset according to its configuration."""

    config = dataset_rebin_config(dataset)
    if isinstance(dataset.data, PointListData):
        return _rebin_point_list_data(dataset, config)
    if isinstance(dataset.data, PointData4D):
        return _rebin_point_data(dataset.data, config)
    if not isinstance(dataset.data, MDHistoData):
        return dataset.data
    return _rebin_mdhisto_data(dataset.data, config)


def _rebin_point_list_data(dataset: DatasetEntry, config: dict[str, Any]) -> PointListData:
    """Rebin transformed point-list data over its coordinates into a histogram."""

    prepared = prepared_point_list_data(dataset)
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coordinate_names = [axis.get("name") for axis in axes_config if axis.get("name") in prepared.columns]
    if not coordinate_names:
        coordinate_names = list(prepared.coordinate_names)
    lower = [axis["lower"] for axis in axes_config[: len(coordinate_names)]] or None
    upper = [axis["upper"] for axis in axes_config[: len(coordinate_names)]] or None
    num_bins = [axis["num_bins"] for axis in axes_config[: len(coordinate_names)]] or None
    return prepared.rebin_to_histogram(
        coordinate_names,
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", False)),
        normalize=True,
    )


def create_rebinned_dataset(
    group: DataGroup,
    dataset: DatasetEntry,
    *,
    name: str | None = None,
) -> DatasetEntry:
    """Materialize a dataset's rebinned view as an independent dataset."""

    data = rebinned_dataset_data(dataset)
    if data is dataset.data:
        data = copy.deepcopy(data)
    new_entry = DatasetEntry(
        name=_unique_dataset_name(name or f"{dataset.name} rebinned", group.dataset_names),
        data=data,
        kind=dataset.kind,
        metadata={
            **copy.deepcopy(dataset.metadata),
            "source_dataset": dataset.name,
            "rebin_materialized": True,
        },
        parameters={},
        masks=copy.deepcopy(dataset.masks),
    )
    group.add_dataset(new_entry)
    return new_entry


def save_dataset_file(dataset: DatasetEntry, path: str | Path, *, use_view: bool = True) -> None:
    """Save a supported dataset to a script-readable ``.npz`` file."""

    data = dataset_for_slice_viewer(dataset) if use_view else dataset.data
    if isinstance(data, PointListData):
        _save_point_list_file(data, path)
        return
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset saving currently supports MDHistoData or PointListData datasets")
    payload: dict[str, Any] = {
        "signal": data.signal,
        "errors": data.errors,
        "mask": data.mask,
        "num_events": data.num_events,
        "metadata_json": json.dumps(_json_safe_value(data.metadata), sort_keys=True),
        "axis_count": np.asarray(len(data.axes), dtype=int),
    }
    for index, axis in enumerate(data.axes):
        payload[f"axis_{index}_values"] = axis.values
        payload[f"axis_{index}_name"] = np.asarray(axis.name)
        payload[f"axis_{index}_units"] = np.asarray(axis.units)
        payload[f"axis_{index}_kind"] = np.asarray(axis.kind)
        payload[f"axis_{index}_frame"] = np.asarray(axis.frame or "")
        payload[f"axis_{index}_path"] = np.asarray(axis.path or "")
        payload[f"axis_{index}_metadata_json"] = np.asarray(json.dumps(_json_safe_value(axis.metadata), sort_keys=True))
    np.savez_compressed(path, **payload)


def _save_point_list_file(data: PointListData, path: str | Path) -> None:
    """Save point-list columns and roles to a script-readable ``.npz`` file."""

    payload: dict[str, Any] = {
        "column_names_json": json.dumps(list(data.column_names)),
        "coordinate_names_json": json.dumps(list(data.coordinate_names)),
        "channels_json": json.dumps(_json_safe_value(data.channels)),
        "units_json": json.dumps(_json_safe_value(data.units)),
        "metadata_json": json.dumps(_json_safe_value(data.metadata), sort_keys=True),
    }
    for index, name in enumerate(data.column_names):
        payload[f"column_{index}"] = np.asarray(data.column(name), dtype=float)
    np.savez_compressed(path, **payload)


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
            lower, upper = _axis_bounds(axis, size)
            num_bins = max(int(size), 1)
            axes.append(
                {
                    "name": axis.name,
                    "units": axis.units,
                    "vector": _identity_vector(index, ndim),
                    "lower": lower,
                    "upper": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
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
            num_bins = max(int(np.unique(finite).size), 1) if finite.size else 1
            axes.append(
                {
                    "name": name,
                    "units": units,
                    "vector": _identity_vector(index, 4),
                    "lower": lower,
                    "upper": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
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
    return float((float(upper) - float(lower)) / float(num_bins))


def _num_bins_from_step_size(lower: Any, upper: Any, step_size: float) -> int:
    width = abs(float(upper) - float(lower))
    if width == 0.0:
        return 1
    return max(int(np.ceil(width / float(step_size))), 1)


def _sanitize_rebin_axis_config(axis_config: dict[str, Any]) -> dict[str, Any]:
    lower = float(axis_config.get("lower", 0.0))
    upper = float(axis_config.get("upper", lower))
    num_bins = max(int(axis_config.get("num_bins", 1)), 1)
    step_size = _step_size_from_bounds(lower, upper, num_bins)
    sanitized = {
        **axis_config,
        "lower": lower,
        "upper": upper,
        "num_bins": num_bins,
        "step_size": step_size,
    }
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


def _rebin_mdhisto_data(data: MDHistoData, config: dict[str, Any]) -> MDHistoData:
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    if len(axes_config) != len(data.axes):
        axes_config = _default_rebin_axes(data)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    num_bins = [axis["num_bins"] for axis in axes_config]

    ndim = len(data.axes)
    source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    projected = []
    for index, axis_config in enumerate(axes_config):
        vector = _rebin_axis_vector(axis_config, index, ndim)
        component = np.zeros(data.shape, dtype=float)
        for weight, grid in zip(vector, source_grids, strict=True):
            if weight:
                component = component + float(weight) * grid
        projected.append(component)
    coords = np.stack(projected, axis=-1)
    valid = np.isfinite(data.signal) & np.isfinite(data.errors) & ~data.mask
    if data.num_events is not None:
        valid &= np.asarray(data.num_events) > 0.0
    signal = data.signal[valid]
    errors = data.errors[valid]
    coords_valid = coords[valid]
    if signal.size == 0:
        raise ValueError("no valid data points remain before rebinning")
    result = rebin_nd(
        signal,
        coords_valid,
        data_errs=errors,
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", False)),
        normalize=True,
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
            metadata=dict(source_axis.metadata),
        )
        for source_axis, axis_config, bins in zip(data.axes, axes_config, result.bins_list, strict=True)
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    metadata = dict(data.metadata)
    metadata["rebin"] = {
        "lower": lower,
        "upper": upper,
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "vectors": vectors,
        "fractional": bool(config.get("fractional", False)),
        "normalize": True,
    }
    return MDHistoData(
        axes=rebinned_axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=data.coordinate_system,
        visual_normalization=data.visual_normalization,
        metadata=metadata,
    )


def _rebin_point_data(data: PointData4D, config: dict[str, Any]) -> PointData4D:
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    if len(axes_config) != 4:
        axes_config = _default_rebin_axes(data)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    num_bins = [axis["num_bins"] for axis in axes_config]
    return rebin_point_data(
        data,
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", False)),
        normalize=True,
    )


def _mdhisto_with_metallix_masks(
    dataset: DatasetEntry,
    *,
    data: MDHistoData | None = None,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData:
    data = dataset.data if data is None else data
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset does not contain MDHistoData")
    file_mask = np.asarray(data.mask, dtype=bool)
    metallix_mask = _metallix_mask_for_mdhisto(dataset, data, extra_masks=extra_masks)
    combined_mask = file_mask | metallix_mask
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask.copy()
    metadata["metallix_mask"] = metallix_mask.copy()
    metadata["metallix_mask_count"] = int(np.count_nonzero(metallix_mask))
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
    )


def _metallix_mask_for_mdhisto(
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
    return np.zeros(data.shape, dtype=bool)


def _mdhisto_coordinate_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    keep = np.ones(data.shape, dtype=bool)
    coords = _mdhisto_coordinate_grids(data)
    coords.update(_mdhisto_coordinate_range_axis_grids(data, parameters))
    for name in COORDINATE_RANGE_PARAMETER_NAMES:
        bounds = _parameter_range(parameters.get(name))
        if bounds is None or name not in coords:
            continue
        lower, upper = bounds
        values = coords[name]
        if lower is not None:
            keep &= values >= lower
        if upper is not None:
            keep &= values <= upper
    return ~keep


def _mdhisto_energy_q_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.shape, dtype=bool)
    energy = _parameter_range(parameters.get("energy"))
    q_modulus = _parameter_range(parameters.get("q_modulus"))
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
    return (lower is None or lower <= -1.0e90) and (upper is None or upper >= 1.0e90)


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
    coords: dict[str, np.ndarray] = {}
    for index, spec in enumerate(specs):
        key = f"{COORDINATE_RANGE_AXIS_PREFIX}{index}"
        vector = _coordinate_axis_vector(parameters.get(key), len(data.axes))
        if vector is None:
            vector = np.asarray(spec["vector"], dtype=float)
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
    specs_by_name: dict[str, list[float]] = {}
    projection_rows: list[np.ndarray] = []
    projection_indices: list[int] = []
    used_indices: set[int] = set()
    for index, axis in enumerate(data.axes):
        role = axis.role
        if role in {"h", "k", "l", "energy_transfer"}:
            name = {"h": "H", "k": "K", "l": "L", "energy_transfer": "E"}[role]
            specs_by_name.setdefault(name, _identity_vector(index, ndim))
            used_indices.add(index)
            continue
        projection = _axis_projection_vector(axis.name)
        if projection is not None:
            projection_rows.append(projection)
            projection_indices.append(index)
    if projection_rows:
        matrix = np.vstack(projection_rows)
        pseudo_inverse = np.linalg.pinv(matrix)
        for coord_index, name in enumerate(("H", "K", "L")):
            if name in specs_by_name:
                continue
            vector = [0.0] * ndim
            for projection_index, weight in zip(projection_indices, pseudo_inverse[coord_index], strict=True):
                vector[projection_index] = _clean_axis_weight(weight)
                used_indices.add(projection_index)
            specs_by_name[name] = vector
    specs = [
        {"name": name, "vector": specs_by_name[name]}
        for name in COORDINATE_RANGE_PARAMETER_NAMES
        if name in specs_by_name
    ]
    for index, axis in enumerate(data.axes):
        if index in used_indices:
            continue
        specs.append({"name": str(axis.name or f"Axis {index}"), "vector": _identity_vector(index, ndim)})
    return specs[:ndim]


def _identity_vector(index: int, ndim: int) -> list[float]:
    return [1.0 if axis_index == index else 0.0 for axis_index in range(ndim)]


def _clean_axis_weight(value: Any) -> float:
    """Round a coordinate-axis weight to drop floating-point noise (e.g. 0.5000000000000001 -> 0.5)."""

    rounded = round(float(value), 10)
    return rounded + 0.0  # normalize -0.0 to 0.0


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
    shape = data.shape
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    coords: dict[str, np.ndarray] = {}
    projection_contributions: list[tuple[np.ndarray, np.ndarray]] = []
    for axis, values in zip(data.axes, axis_values, strict=True):
        role = axis.role
        if role in {"h", "k", "l", "energy_transfer", "q_modulus"}:
            key = {"h": "H", "k": "K", "l": "L", "energy_transfer": "E", "q_modulus": "q_modulus"}[role]
            coords[key] = values
            continue
        projection = _axis_projection_vector(axis.name)
        if projection is not None:
            projection_contributions.append((projection, values))
    if not {"H", "K", "L"}.issubset(coords) and projection_contributions:
        hkl = np.zeros((*shape, 3), dtype=float)
        for projection, values in projection_contributions:
            hkl = hkl + values[..., np.newaxis] * projection
        coords.setdefault("H", hkl[..., 0])
        coords.setdefault("K", hkl[..., 1])
        coords.setdefault("L", hkl[..., 2])
    return coords


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


def next_data_group_name(groups: list[DataGroup]) -> str:
    """Return the first available WorkspaceN name for a top-level workspace."""

    taken = {group.name for group in groups}
    index = 1
    while f"Workspace{index}" in taken:
        index += 1
    return f"Workspace{index}"


def next_mask_name(masks: list[MaskSpec]) -> str:
    """Return the first available MaskN name."""

    taken = {mask.name for mask in masks}
    index = 1
    while f"Mask{index}" in taken:
        index += 1
    return f"Mask{index}"


def next_model_name(models: dict[str, Any]) -> str:
    """Return the first available ModelN name."""

    taken = set(models)
    index = 1
    while f"Model{index}" in taken:
        index += 1
    return f"Model{index}"


def next_fit_result_name(fits: list[FitTimelineEntry]) -> str:
    """Return the first available Fit ResultN name across a fit tree."""

    return _next_fit_tree_name(fits, "Fit Result")


def next_fit_timeline_name(fits: list[FitTimelineEntry]) -> str:
    """Return the first available TimelineN name across a fit tree."""

    return _next_fit_tree_name(fits, "Timeline")


def _next_fit_tree_name(fits: list[FitTimelineEntry], prefix: str) -> str:
    taken = {entry.name for entry in _walk_fit_entries(fits)}
    index = 1
    while f"{prefix}{index}" in taken:
        index += 1
    return f"{prefix}{index}"


def _walk_fit_entries(fits: list[FitTimelineEntry]):
    for entry in fits:
        yield entry
        yield from _walk_fit_entries(entry.children)


def _fit_entry_in_tree(fits: list[FitTimelineEntry], target: FitTimelineEntry | None) -> bool:
    if target is None:
        return False
    return any(entry is target for entry in _walk_fit_entries(fits))


def _fit_siblings(
    entries: list[FitTimelineEntry],
    target: FitTimelineEntry,
) -> list[FitTimelineEntry] | None:
    if target in entries:
        return entries
    for entry in entries:
        found = _fit_siblings(entry.children, target)
        if found is not None:
            return found
    return None


def _fit_parent(
    entries: list[FitTimelineEntry],
    target: FitTimelineEntry,
) -> FitTimelineEntry | None:
    for entry in entries:
        if target in entry.children:
            return entry
        found = _fit_parent(entry.children, target)
        if found is not None:
            return found
    return None


def _last_result_at_level(entries: list[FitTimelineEntry]) -> FitTimelineEntry | None:
    for entry in reversed(entries):
        if entry.kind == "result":
            return entry
    return None


def _current_state_at_level(entries: list[FitTimelineEntry]) -> FitTimelineEntry | None:
    for entry in reversed(entries):
        if entry.kind == "current":
            return entry
    return None


def _fit_entry_matches_current_state(entries: list[FitTimelineEntry], fit_entry: FitTimelineEntry) -> bool:
    current = _current_state_at_level(entries)
    return bool(current is not None and fit_entry.snapshot == current.snapshot)


def _current_state_after_result(group: DataGroup, result: FitTimelineEntry) -> FitTimelineEntry | None:
    siblings = _fit_siblings(group.fits, result)
    if siblings is None or result not in siblings:
        return None
    result_index = siblings.index(result)
    for entry in siblings[result_index + 1 :]:
        if entry.kind == "current":
            return entry
    return None


def _should_branch_fit_now(group: DataGroup, parent: FitTimelineEntry) -> bool:
    if parent.kind in {"initial", "current"}:
        return False
    siblings = _fit_siblings(group.fits, parent)
    if siblings is None:
        return True
    return not (
        parent is _last_result_at_level(siblings)
        and _fit_entry_matches_current_state(siblings, parent)
    )


def _replace_current_state(entries: list[FitTimelineEntry], group: DataGroup) -> None:
    entries[:] = [entry for entry in entries if entry.kind != "current"]
    if any(entry.kind == "result" for entry in entries):
        entries.append(current_state_fit_entry(group))


def _top_level_current_state_entry(group: DataGroup) -> FitTimelineEntry | None:
    for entry in reversed(group.fits):
        if entry.kind == "current":
            return entry
    return None


def _set_fit_current_snapshot(entry: FitTimelineEntry, group: DataGroup) -> None:
    entry.snapshot = snapshot_data_group_state(group)
    entry.created_at = _timestamp_now()


def _style_enabled_tree_item(item: Any, enabled: bool) -> None:
    from PySide6 import QtGui

    # Enabled items use the theme's default text color (bright in dark mode,
    # dark in light mode); disabled items use a medium gray readable on both.
    if enabled:
        item.setForeground(0, QtGui.QBrush())
    else:
        item.setForeground(0, QtGui.QBrush(QtGui.QColor("#8a8a8a")))


def _style_tree_hierarchy_item(item: Any, *, bold: bool = False, underline: bool = False) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont(item.font(0))
    font.setBold(bold)
    font.setUnderline(underline)
    item.setFont(0, font)


_TREE_ICON_CACHE: dict[str, Any] = {}


def _tree_item_icon(kind: str) -> Any:
    from PySide6 import QtCore, QtGui

    if kind in _TREE_ICON_CACHE:
        return _TREE_ICON_CACHE[kind]

    colors = {
        "folder": "#5f8fa8",
        "mask_folder": "#a88fc5",
        "model_folder": "#c7a45b",
        "fit_folder": "#7cab80",
        "dataset": "#6d9fc7",
        "mask": "#a88fc5",
        "model": "#c7a45b",
        "fit_result": "#7cab80",
        "fit_initial": "#7cab80",
        "fit_current": "#7cab80",
    }
    accent = QtGui.QColor(colors.get(kind, "#9ca3ad"))
    line = QtGui.QColor("#c7ccd1")

    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(accent, 1.4)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    if kind in {"folder", "mask_folder", "model_folder", "fit_folder"}:
        fill = QtGui.QColor(accent)
        fill.setAlpha(34)
        painter.setPen(QtGui.QPen(accent, 1.15))
        painter.setBrush(fill)
        path = QtGui.QPainterPath(QtCore.QPointF(2.5, 4.6))
        path.lineTo(QtCore.QPointF(5.7, 4.6))
        path.lineTo(QtCore.QPointF(7.0, 6.1))
        path.lineTo(QtCore.QPointF(13.5, 6.1))
        path.lineTo(QtCore.QPointF(13.5, 12.9))
        path.lineTo(QtCore.QPointF(2.5, 12.9))
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    elif kind == "dataset":
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawRoundedRect(QtCore.QRectF(2.3, 4.0, 11.4, 8.7), 1.0, 1.0)
        painter.setPen(QtGui.QPen(accent, 1.0))
        for x in (6.1, 9.9):
            painter.drawLine(QtCore.QPointF(x, 4.3), QtCore.QPointF(x, 12.4))
        for y in (6.9, 9.8):
            painter.drawLine(QtCore.QPointF(2.6, y), QtCore.QPointF(13.4, y))
    elif kind == "mask":
        removed = QtGui.QColor(accent)
        removed.setAlpha(88)
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawRoundedRect(QtCore.QRectF(3.0, 3.0, 10.0, 10.0), 1.2, 1.2)
        painter.setPen(QtGui.QPen(accent, 1.0))
        painter.setBrush(removed)
        painter.drawRect(QtCore.QRectF(3.4, 3.4, 4.0, 4.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawLine(QtCore.QPointF(4.1, 4.1), QtCore.QPointF(6.7, 6.7))
        painter.drawLine(QtCore.QPointF(6.7, 4.1), QtCore.QPointF(4.1, 6.7))
    elif kind == "model":
        path = QtGui.QPainterPath(QtCore.QPointF(2.8, 10.8))
        path.cubicTo(QtCore.QPointF(5.0, 3.0), QtCore.QPointF(7.2, 13.0), QtCore.QPointF(10.0, 5.2))
        path.cubicTo(QtCore.QPointF(11.2, 2.0), QtCore.QPointF(12.8, 5.5), QtCore.QPointF(13.2, 8.0))
        painter.drawPath(path)
        painter.setBrush(accent)
        painter.drawEllipse(QtCore.QPointF(5.2, 7.0), 1.1, 1.1)
        painter.drawEllipse(QtCore.QPointF(10.0, 5.2), 1.1, 1.1)
    elif kind == "fit_result":
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawLine(QtCore.QPointF(3.0, 12.0), QtCore.QPointF(13.0, 12.0))
        painter.drawLine(QtCore.QPointF(3.0, 12.0), QtCore.QPointF(3.0, 4.0))
        painter.setPen(QtGui.QPen(accent, 1.4))
        painter.drawPolyline(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(4.0, 10.5),
                    QtCore.QPointF(6.3, 8.4),
                    QtCore.QPointF(8.4, 9.3),
                    QtCore.QPointF(11.8, 5.0),
                ]
            )
        )
    elif kind == "fit_initial":
        painter.setPen(QtGui.QPen(accent, 1.4))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 4.2, 4.2)
        painter.drawLine(QtCore.QPointF(8.0, 3.8), QtCore.QPointF(8.0, 6.0))
    elif kind == "fit_current":
        painter.setPen(QtGui.QPen(line, 1.1))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 4.5, 4.5)
        painter.setPen(QtGui.QPen(accent, 1.3))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 2.1, 2.1)
        painter.setBrush(accent)
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 0.9, 0.9)

    painter.end()
    icon = QtGui.QIcon(pixmap)
    _TREE_ICON_CACHE[kind] = icon
    return icon


def _set_tree_item_icon(item: Any, kind: str) -> None:
    item.setIcon(0, _tree_item_icon(kind))


def _enabled_state_for_role(
    role: str,
    entry: DatasetEntry | None,
    mask: MaskSpec | None,
    model: ModelComponentSpec | None,
) -> bool | None:
    if role == "dataset" and entry is not None:
        return bool(entry.enabled)
    if role == "mask" and mask is not None:
        return bool(mask.enabled)
    if role == "model" and model is not None:
        return bool(model.enabled)
    return None


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dataset_entry_from_path(
    path: str | Path,
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
) -> DatasetEntry:
    """Create a dataset entry for a file selected in the GUI.

    Point-list types are loaded immediately; MDHisto/``.nxs`` types are created
    as lazy placeholders and loaded on first view.
    """

    source = Path(path)
    resolved_type = data_type or DEFAULT_DATA_TYPE
    entry = DatasetEntry(
        name=_unique_dataset_name(source.stem or source.name, []),
        data=None,
        kind=source.suffix.lstrip(".").lower(),
        data_type=resolved_type,
        metadata={"source_file": str(source), "import_status": "pending"},
    )
    if data_type_container(resolved_type) == "point_list":
        chosen = importer_name or default_importer_for_data_type(resolved_type)
        if chosen is not None:
            entry.metadata["importer"] = chosen
        _load_point_list_dataset(entry)
    return entry


def save_project(project: MetallixProject, path: str | Path) -> None:
    """Persist the GUI project as editable JSON."""

    Path(path).write_text(
        json.dumps(_project_to_dict(project), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_project(path: str | Path) -> MetallixProject:
    """Load a project saved by :func:`save_project`."""

    return _project_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def recent_project_paths(settings: Any | None = None) -> list[Path]:
    """Return recently opened project paths from app settings."""

    settings = _settings() if settings is None else settings
    value = settings.value(RECENT_PROJECTS_KEY, [])
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    return [Path(str(path)) for path in values]


def remember_recent_project(path: str | Path, settings: Any | None = None) -> list[Path]:
    """Record a project path as most-recent and return the updated list."""

    settings = _settings() if settings is None else settings
    resolved = Path(path).expanduser()
    recent = [existing for existing in recent_project_paths(settings) if existing != resolved]
    recent.insert(0, resolved)
    recent = recent[:RECENT_PROJECT_LIMIT]
    settings.setValue(RECENT_PROJECTS_KEY, [str(path) for path in recent])
    return recent


def forget_missing_recent_projects(settings: Any | None = None) -> list[Path]:
    """Drop recent projects whose files are no longer present."""

    settings = _settings() if settings is None else settings
    recent = [path for path in recent_project_paths(settings) if path.exists()]
    settings.setValue(RECENT_PROJECTS_KEY, [str(path) for path in recent])
    return recent


class MetallixProjectExplorer:
    """PySide6 project explorer for building metallix analysis pipelines."""

    def __init__(self, project: MetallixProject | None = None) -> None:
        self.app = _qt_app()
        self.project = MetallixProject() if project is None else project
        self.project_path: Path | None = None
        self.has_unsaved_changes = False
        self._allow_window_close = False
        self.window = None
        self.file_menu = None
        self.recent_projects_menu = None
        self.tree = None
        self.title_label = None
        self.enabled_check = None
        self.fit_weight_widget = None
        self.fit_weight_spin = None
        self.scale_factor_spin = None
        self.group_bulk_widget = None
        self.group_fit_weight_edit = None
        self.group_scale_edit = None
        self.details_label = None
        self.details_scroll = None
        self.details_widget = None
        self.details_layout = None
        self.import_dataset_button = None
        self.add_model_button = None
        self.view_slice_button = None
        self.load_dataset_button = None
        self.add_mask_button = None
        self.add_dataset_group_button = None
        self.save_dataset_button = None
        self.mask_type_combo = None
        self.mask_parameter_widget = None
        self.mask_parameter_layout = None
        self.model_type_combo = None
        self.model_parameter_widget = None
        self.model_parameter_layout = None
        self.fit_editor_widget = None
        self.fit_optimizer_combo = None
        self.fit_optimizer_config_editor = None
        self.fit_branch_check = None
        self.fit_now_button = None
        self.expand_all_button = None
        self.collapse_all_button = None
        self.create_group_button = None
        self.delete_button = None
        self._clipboard: tuple[str, DatasetEntry | MaskSpec] | None = None
        self._slice_viewers: dict[int, Any] = {}
        self._restoring_fit_selection = False
        self._active_fit_group: DataGroup | None = None
        self._active_fit_anchor: FitTimelineEntry | None = None
        self._active_fit_current: FitTimelineEntry | None = None
        self._active_branch_current: FitTimelineEntry | None = None
        self._item_roles: dict[
            int,
            tuple[
                str,
                DataGroup | None,
                DatasetEntry | None,
                MaskSpec | None,
                ModelComponentSpec | None,
            ],
        ] = {}
        self._fit_item_roles: dict[int, FitTimelineEntry] = {}
        self._dataset_group_roles: dict[int, DatasetGroup] = {}
        self._expanded_state: dict[tuple[Any, ...], bool] = {}
        self._build()
        self._refresh_tree()
        self._sync_details()

    def show(self) -> "MetallixProjectExplorer":
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self

    def run(self) -> int:
        self.show()
        return int(self.app.exec())

    def create_data_group(self) -> DataGroup:
        group = create_data_group(self.project)
        self._mark_dirty()
        self._refresh_tree(select_group=group, edit_group=True)
        return group

    def import_dataset_paths(
        self,
        group: DataGroup,
        paths: list[str | Path],
        *,
        data_type: str | None = None,
        importer_name: str | None = None,
        into: DatasetGroup | None = None,
    ) -> list[DatasetEntry]:
        from PySide6 import QtWidgets

        try:
            entries = import_dataset_paths(
                group, paths, data_type=data_type, importer_name=importer_name, into=into
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Import datasets",
                f"Could not import datasets:\n{exc}",
            )
            return []
        if entries:
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        return entries

    def selected_data_group(self) -> DataGroup | None:
        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"group", "datasets", "models", "dataset", "fits", "fit", "fit_timeline", "dataset_group"}:
            return group
        return None

    def _selected_import_target(self) -> tuple[DataGroup | None, DatasetGroup | None]:
        """Return the (data group, nested group) that an import should populate."""

        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if role == "dataset_group":
            return group, self._dataset_group_for_item(item)
        if role in {"group", "datasets", "models", "dataset", "fits", "fit", "fit_timeline"}:
            return group, None
        return None, None

    def delete_selected(self) -> None:
        item = self._current_item()
        group, entry, mask, model, role = self._objects_for_item(item)
        if role == "group" and group is not None:
            delete_data_group(self.project, group)
            self._close_slice_viewer(group)
            if self._active_fit_group is group:
                self._clear_active_fit_state()
            self._mark_dirty()
            self._refresh_tree()
        elif role == "dataset" and group is not None and entry is not None:
            delete_dataset(group, entry)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        elif role == "mask" and group is not None and entry is not None and mask is not None:
            delete_mask(entry, mask)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        elif role == "dataset_group" and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None and delete_dataset_group(group, subgroup):
                self._record_data_group_state_change(group)
                self._mark_dirty()
                self._refresh_tree(select_group=group)
        elif role == "group_mask" and group is not None and mask is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None and mask in subgroup.masks:
                subgroup.masks.remove(mask)
                self._record_data_group_state_change(group)
                self._mark_dirty()
                self._refresh_tree(select_group=group)
        elif role == "model" and group is not None and model is not None:
            delete_model_component(group, model)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        elif role in {"fit", "fit_timeline"} and group is not None:
            fit_entry = self._fit_entry_for_item(item)
            if fit_entry is not None and delete_fit_entry(group, fit_entry):
                self._mark_dirty()
                self._refresh_tree(select_group=group)

    def new_project(self) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        self._close_all_slice_viewers()
        self.project = MetallixProject()
        self.project_path = None
        self.has_unsaved_changes = False
        self._clear_active_fit_state()
        self._refresh_tree()
        self._sync_window_title()
        return True

    def close_project(self) -> bool:
        if self.project_path is None:
            return self.quit_application()
        return self.new_project()

    def quit_application(self) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        self._close_all_slice_viewers()
        self._allow_window_close = True
        self.window.close()
        self._allow_window_close = False
        self.app.quit()
        return True

    def load_dataset_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        try:
            loaded = dataset_for_slice_viewer(entry)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Load dataset",
                f"Could not load dataset:\n{exc}",
            )
            return False
        if loaded is None:
            return False
        self._sync_details()
        if group is not None:
            self.refresh_slice_viewer(group)
        return True

    def save(self) -> bool:
        if self.project_path is None:
            return self.save_as()
        save_project(self.project, self.project_path)
        self.has_unsaved_changes = False
        self._sync_window_title()
        return True

    def save_as(self) -> bool:
        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save metallix project",
            "metallix_project.mtlx",
            "metallix projects (*.mtlx);;All files (*)",
        )
        if not path:
            return False
        self.project_path = Path(path)
        save_project(self.project, self.project_path)
        self._remember_recent_project(self.project_path)
        self.has_unsaved_changes = False
        self._sync_window_title()
        return True

    def open_project(self) -> bool:
        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Open metallix project",
            "",
            "metallix projects (*.mtlx);;All files (*)",
        )
        if not path:
            return False
        return self.open_project_path(path)

    def open_project_path(self, path: str | Path, *, remember: bool = True) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        path = Path(path)
        if not path.exists():
            forget_missing_recent_projects()
            self._refresh_recent_projects_menu()
            return False
        self._close_all_slice_viewers()
        self.project = load_project(path)
        self.project_path = path
        self.has_unsaved_changes = False
        self._clear_active_fit_state()
        if remember:
            self._remember_recent_project(path)
        self._refresh_tree()
        self._sync_window_title()
        return True

    def import_dataset_dialog(self) -> None:
        from PySide6 import QtWidgets

        group, into = self._selected_import_target()
        if group is None:
            return
        paths, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self.window,
            "Import datasets",
            "",
            "Data files (*);;All files (*)",
        )
        if not paths:
            return
        choice = self._prompt_import_data_type()
        if choice is None:
            return
        data_type, importer_name = choice
        self.import_dataset_paths(group, paths, data_type=data_type, importer_name=importer_name, into=into)

    def _prompt_import_data_type(self) -> tuple[str, str | None] | None:
        from PySide6 import QtWidgets

        types = available_data_types()
        labels = [label for _name, label in types]
        default_index = next((i for i, (name, _label) in enumerate(types) if name == DEFAULT_DATA_TYPE), 0)
        label, accepted = QtWidgets.QInputDialog.getItem(
            self.window,
            "Data type",
            "What type of data is this?",
            labels,
            default_index,
            editable=False,
        )
        if not accepted:
            return None
        data_type = types[labels.index(label)][0]
        importer_name = default_importer_for_data_type(data_type)
        importer_specs = importers_for_data_type(data_type)
        if len(importer_specs) > 1:
            importer_labels = [spec.label for spec in importer_specs]
            picked, ok = QtWidgets.QInputDialog.getItem(
                self.window,
                "Importer",
                "Which importer should read this file?",
                importer_labels,
                0,
                editable=False,
            )
            if not ok:
                return None
            importer_name = importer_specs[importer_labels.index(picked)].name
        return data_type, importer_name

    def add_mask_to_selection(self) -> MaskSpec | None:
        item = self._current_item()
        group, entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"group_masks", "dataset_group"} and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is None:
                return None
            mask = create_group_mask(subgroup, _group_reference_dataset(subgroup))
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_mask=mask, edit_mask=True)
            return mask
        if role not in {"dataset", "masks"} or group is None or entry is None:
            return None
        mask = create_mask(entry)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_mask=mask, edit_mask=True)
        return mask

    def add_dataset_group_to_selection(self) -> DatasetGroup | None:
        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if group is None:
            return None
        if role in {"group", "datasets"}:
            parent_node: Any = group
        elif role == "dataset_group":
            parent_node = self._dataset_group_for_item(item)
        else:
            return None
        if parent_node is None:
            return None
        subgroup = create_dataset_group(group, parent_node)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_dataset_group=subgroup, edit_group=True)
        return subgroup

    def add_model_to_selection(self) -> ModelComponentSpec | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"group", "models"} or group is None:
            return None
        model = create_model_component(group)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_model=model, edit_model=True)
        return model

    def materialize_rebin_for_selection(self) -> DatasetEntry | None:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return None
        try:
            rebinned = create_rebinned_dataset(group, entry)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Create dataset from rebin",
                f"Could not create rebinned dataset:\n{exc}",
            )
            return None
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_dataset=rebinned)
        return rebinned

    def save_dataset_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        _group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save dataset",
            f"{entry.name}.npz",
            "NumPy archives (*.npz);;All files (*)",
        )
        if not path:
            return False
        try:
            save_dataset_file(entry, path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Save dataset",
                f"Could not save dataset:\n{exc}",
            )
            return False
        return True

    def fit_now_for_selection(self) -> FitTimelineEntry | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        self._set_selected_fit_optimizer_config()
        should_branch = bool(self.fit_branch_check.isChecked()) or _should_branch_fit_now(group, fit_entry)
        result = create_placeholder_fit_result(
            group,
            fit_entry,
            branch_timeline=should_branch,
        )
        self.fit_branch_check.setChecked(False)
        item_to_select = _current_state_after_result(group, result) if should_branch else result
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_fit=item_to_select or result)
        return result

    def open_slice_viewer_for_selection(self) -> Any | None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        allowed = {"group", "datasets", "dataset", "masks", "mask", "dataset_group", "group_masks", "group_mask"}
        if role not in allowed or group is None:
            return None
        # For a mask or the Masks node, entry is the owning dataset.
        selected_name = entry.name if role in {"dataset", "masks", "mask"} and entry is not None else None
        return self.open_slice_viewer(group, selected_dataset_name=selected_name)

    def open_slice_viewer(
        self,
        group: DataGroup,
        *,
        selected_dataset_name: str | None = None,
    ) -> Any | None:
        from PySide6 import QtWidgets

        try:
            datasets, names = slice_viewer_datasets(group)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Data viewer",
                f"Could not load datasets for the data viewer:\n{exc}",
            )
            return None
        self._sync_details()
        if not datasets:
            return None
        viewer = self._replace_slice_viewer(group, datasets, names)
        if selected_dataset_name in names:
            viewer.dataset_combo.setCurrentIndex(names.index(selected_dataset_name))
        viewer.show()
        return viewer

    def refresh_slice_viewer(self, group: DataGroup) -> Any | None:
        if id(group) not in self._slice_viewers:
            return None
        selected_name = None
        current_viewer = self._slice_viewers[id(group)]
        if current_viewer.dataset_combo is not None:
            selected_name = current_viewer.dataset_combo.currentText()
        try:
            datasets, names = slice_viewer_datasets(group)
        except Exception:
            return None
        if not datasets:
            self._close_slice_viewer(group)
            return None
        if hasattr(current_viewer, "replace_datasets"):
            current_viewer.replace_datasets(
                datasets,
                dataset_names=names,
                selected_dataset_name=selected_name,
            )
        else:
            self._replace_slice_viewer(group, datasets, names)
        return self._slice_viewers.get(id(group))

    def copy_selected(self) -> None:
        _group, entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role == "dataset" and entry is not None:
            self._clipboard = ("dataset", copy.deepcopy(entry))
        elif role == "mask" and mask is not None:
            self._clipboard = ("mask", copy.deepcopy(mask))

    def paste_into_selection(self) -> None:
        if self._clipboard is None:
            return
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        clip_role, payload = self._clipboard
        if clip_role == "dataset" and isinstance(payload, DatasetEntry) and role in {"group", "datasets"} and group is not None:
            copied = copy_dataset_to_group(payload, group)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_dataset=copied)
        elif clip_role == "mask" and isinstance(payload, MaskSpec) and role in {"dataset", "masks"} and group is not None and entry is not None:
            copied = copy_mask_to_dataset(payload, entry)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_mask=copied)

    def rename_selected(self) -> None:
        item = self._current_item()
        if item is not None and self._is_renameable_item(item):
            self.tree.editItem(item, 0)

    def expand_all(self) -> None:
        self.tree.expandAll()
        self._expanded_state = self._current_expanded_state()

    def collapse_all(self) -> None:
        self.tree.collapseAll()
        self._expanded_state = self._current_expanded_state()

    def _is_renameable_item(self, item: Any) -> bool:
        return _is_renameable_role(self._objects_for_item(item)[4])

    def show_file_location_for_selection(self) -> bool:
        _group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        source = _dataset_source_path(entry)
        if source is None:
            return False
        return _show_file_location(source)

    def change_file_source_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        current = _dataset_source_path(entry)
        start_dir = str(current.parent) if current is not None else ""
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Change dataset source",
            start_dir,
            "Data files (*);;All files (*)",
        )
        if not path:
            return False
        set_dataset_source(entry, path)
        self._mark_dirty()
        if group is not None:
            self._refresh_tree(select_group=group, select_dataset=entry)
        return True

    def _resolve_dataset_drop_target(self, target_item: Any) -> tuple[DataGroup | None, Any]:
        """Return the (data group, container node) a dataset drop should land in."""

        group, entry, _mask, _model, role = self._objects_for_item(target_item)
        if group is None:
            return None, None
        if role in {"group", "datasets"}:
            return group, group
        if role == "dataset_group":
            node = self._dataset_group_for_item(target_item)
            return (group, node) if node is not None else (None, None)
        if role in {"dataset", "masks"} and entry is not None:
            node = _dataset_parent_node(group, entry)
            return (group, node) if node is not None else (None, None)
        return None, None

    def _tree_item_sort_key(self, item: Any) -> tuple[int, ...]:
        path: list[int] = []
        while item is not None:
            parent = item.parent()
            if parent is None:
                path.append(self.tree.indexOfTopLevelItem(item))
            else:
                path.append(parent.indexOfChild(item))
            item = parent
        return tuple(reversed(path))

    def _selected_items_for_drag_role(self, role: str) -> list[Any]:
        items = [item for item in self.tree.selectedItems() if self._objects_for_item(item)[4] == role]
        current = self._current_item()
        if current is not None and self._objects_for_item(current)[4] == role and current not in items:
            items.append(current)
        return sorted(items, key=self._tree_item_sort_key)

    def _is_below_drop(self, drop_position: Any) -> bool:
        from PySide6 import QtWidgets

        return drop_position == QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem

    def _dataset_drop_target_with_index(
        self,
        target_item: Any,
        drop_position: Any = None,
    ) -> tuple[DataGroup | None, Any, int | None]:
        target_group, target_node = self._resolve_dataset_drop_target(target_item)
        if target_group is None or target_node is None:
            return None, None, None
        _group, target_entry, _mask, _model, target_role = self._objects_for_item(target_item)
        if target_role == "dataset" and target_entry is not None:
            parent_node = _dataset_parent_node(target_group, target_entry)
            if parent_node is not None:
                index = parent_node.datasets.index(target_entry)
                if self._is_below_drop(drop_position):
                    index += 1
                return target_group, parent_node, index
        return target_group, target_node, len(target_node.datasets)

    def _mask_drop_target_with_index(
        self,
        target_item: Any,
        drop_position: Any = None,
    ) -> tuple[DataGroup | None, DatasetEntry | None, int | None]:
        group, entry, target_mask, _model, role = self._objects_for_item(target_item)
        if group is None or entry is None or role not in {"dataset", "masks", "mask"}:
            return None, None, None
        if role == "mask" and target_mask is not None:
            index = entry.masks.index(target_mask)
            if self._is_below_drop(drop_position):
                index += 1
            return group, entry, index
        return group, entry, len(entry.masks)

    def _move_or_copy_datasets(
        self,
        dataset_items: list[Any],
        target_item: Any,
        *,
        copy_item: bool,
        drop_position: Any = None,
    ) -> bool:
        target_group, target_node, insert_index = self._dataset_drop_target_with_index(target_item, drop_position)
        if target_group is None or target_node is None:
            return False
        entries: list[tuple[DataGroup, DatasetEntry]] = []
        for item in dataset_items:
            source_group, source_entry, _mask, _model, role = self._objects_for_item(item)
            if role == "dataset" and source_group is not None and source_entry is not None:
                entries.append((source_group, source_entry))
        if not entries:
            return False
        touched_groups: set[int] = set()
        groups_by_id: dict[int, DataGroup] = {}
        last_entry: DatasetEntry | None = None

        def touch(group: DataGroup) -> None:
            touched_groups.add(id(group))
            groups_by_id[id(group)] = group

        if not copy_item and all(
            source_group is target_group and _dataset_parent_node(source_group, source_entry) is target_node
            for source_group, source_entry in entries
        ):
            if _move_items_within_list(target_node.datasets, [entry for _group, entry in entries], insert_index or 0):
                touch(target_group)
                last_entry = entries[-1][1]
            else:
                last_entry = entries[-1][1]
            for group in groups_by_id.values():
                self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_dataset=last_entry)
            return True

        index = len(target_node.datasets) if insert_index is None else max(0, min(insert_index, len(target_node.datasets)))
        for source_group, source_entry in entries:
            if copy_item:
                new_entry = copy_dataset_to_group(source_entry, target_group)
                target_group.datasets.remove(new_entry)
                target_node.datasets.insert(index, new_entry)
                index += 1
                touch(target_group)
                last_entry = new_entry
            elif source_group is target_group:
                parent_node = _dataset_parent_node(source_group, source_entry)
                if parent_node is not None:
                    parent_node.datasets.remove(source_entry)
                    target_node.datasets.insert(index, source_entry)
                    index += 1
                touch(source_group)
                last_entry = source_entry
            else:
                new_entry = copy_dataset_to_group(source_entry, target_group)
                target_group.datasets.remove(new_entry)
                target_node.datasets.insert(index, new_entry)
                index += 1
                delete_dataset(source_group, source_entry)
                touch(source_group)
                touch(target_group)
                last_entry = new_entry
        for group in groups_by_id.values():
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=target_group, select_dataset=last_entry)
        return True

    def _move_selected_groups(self, target_item: Any, drop_position: Any = None) -> bool:
        _source_group, _entry, _mask, _model, target_role = self._objects_for_item(target_item)
        target_group = self._objects_for_item(target_item)[0]
        if target_role != "group" or target_group is None:
            return False
        group_items = self._selected_items_for_drag_role("group")
        groups = [self._objects_for_item(item)[0] for item in group_items]
        groups = [group for group in groups if group is not None]
        if not groups:
            return False
        index = self.project.data_groups.index(target_group)
        if self._is_below_drop(drop_position):
            index += 1
        moved = _move_items_within_list(self.project.data_groups, groups, index)
        self._mark_dirty()
        self._refresh_tree(select_group=groups[-1] if groups else target_group)
        return moved or True

    def _move_or_copy_masks(
        self,
        mask_items: list[Any],
        target_item: Any,
        *,
        copy_item: bool,
        drop_position: Any = None,
    ) -> bool:
        target_group, target_entry, insert_index = self._mask_drop_target_with_index(target_item, drop_position)
        if target_group is None or target_entry is None:
            return False
        masks: list[tuple[DataGroup, DatasetEntry, MaskSpec]] = []
        for item in mask_items:
            source_group, source_entry, source_mask, _model, role = self._objects_for_item(item)
            if role == "mask" and source_group is not None and source_entry is not None and source_mask is not None:
                masks.append((source_group, source_entry, source_mask))
        if not masks:
            return False
        if not copy_item and all(source_entry is target_entry for _group, source_entry, _mask in masks):
            _move_items_within_list(target_entry.masks, [mask for _group, _entry, mask in masks], insert_index or 0)
            self._record_data_group_state_change(target_group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_mask=masks[-1][2])
            return True
        index = len(target_entry.masks) if insert_index is None else max(0, min(insert_index, len(target_entry.masks)))
        touched_groups: dict[int, DataGroup] = {}
        last_mask: MaskSpec | None = None
        for source_group, source_entry, source_mask in masks:
            moved = copy_mask_to_dataset(source_mask, target_entry)
            target_entry.masks.remove(moved)
            target_entry.masks.insert(index, moved)
            index += 1
            last_mask = moved
            touched_groups[id(target_group)] = target_group
            if not copy_item:
                delete_mask(source_entry, source_mask)
                touched_groups[id(source_group)] = source_group
        for group in touched_groups.values():
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=target_group, select_mask=last_mask)
        return True

    def move_or_copy_selected_to_item(self, target_item: Any, *, copy_item: bool, drop_position: Any = None) -> bool:
        source_item = self._current_item()
        source_group, source_entry, source_mask, _source_model, source_role = self._objects_for_item(source_item)
        target_group, target_entry, target_mask, _target_model, target_role = self._objects_for_item(target_item)

        if source_role == "group" and not copy_item:
            return self._move_selected_groups(target_item, drop_position)

        # Move/copy one or more selected datasets into the drop target.
        if source_role == "dataset":
            return self._move_or_copy_datasets(
                self._selected_items_for_drag_role("dataset"),
                target_item,
                copy_item=copy_item,
                drop_position=drop_position,
            )

        # Re-parent a subgroup within the same group.
        if (
            source_role == "dataset_group"
            and not copy_item
            and source_group is not None
            and target_group is source_group
            and target_role in {"group", "datasets", "dataset_group"}
        ):
            subgroup = self._dataset_group_for_item(source_item)
            target_node = source_group if target_role in {"group", "datasets"} else self._dataset_group_for_item(target_item)
            if (
                subgroup is not None
                and target_node is not None
                and not _group_contains_node(subgroup, target_node)
            ):
                parent_node = _dataset_group_parent(source_group, subgroup)
                if parent_node is not None and parent_node is not target_node:
                    parent_node.subgroups.remove(subgroup)
                    target_node.subgroups.append(subgroup)
                    self._record_data_group_state_change(source_group)
                    self._mark_dirty()
                    self._refresh_tree(select_group=source_group, select_dataset_group=subgroup)
            return True

        if source_role == "mask":
            return self._move_or_copy_masks(
                self._selected_items_for_drag_role("mask"),
                target_item,
                copy_item=copy_item,
                drop_position=drop_position,
            )
        return False

    def _build(self) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        window_class = _make_project_window_class()
        self.window = window_class(self)
        self.window.setWindowTitle("metallix Project Explorer")
        self.window.resize(1120, 760)

        toolbar = QtWidgets.QToolBar("Project")
        toolbar.setMovable(False)
        self.window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        file_button = QtWidgets.QToolButton()
        file_button.setObjectName("file_menu_button")
        file_button.setText("File")
        file_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(file_button)
        self.file_menu = menu
        new_action = menu.addAction("New", self.new_project)
        new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        open_action = menu.addAction("Open", self.open_project)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.recent_projects_menu = menu.addMenu("Recent projects")
        self.recent_projects_menu.aboutToShow.connect(self._refresh_recent_projects_menu)
        menu.addSeparator()
        save_action = menu.addAction("Save", self.save)
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_as_action = menu.addAction("Save As", self.save_as)
        save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        menu.addSeparator()
        close_action = menu.addAction("Close", self.close_project)
        close_action.setShortcut(QtGui.QKeySequence.StandardKey.Close)
        quit_action = menu.addAction("Quit", self.quit_application)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        file_button.setMenu(menu)
        toolbar.addWidget(file_button)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.window.setCentralWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        project_tree_class = _make_project_tree_class()
        self.tree = project_tree_class(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(18)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        # DragDrop (not DropOnly) lets the view initiate drags via our custom
        # startDrag; DropOnly disables drag initiation entirely.
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.tree.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.tree.setDropIndicatorShown(True)
        # Allow shift/ctrl(cmd) range and multi selection for dragging several
        # datasets into a group at once.
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.tree.currentItemChanged.connect(lambda _current, _previous: self._sync_details())
        self.tree.itemChanged.connect(self._tree_item_changed)
        toolbar_font = QtGui.QFont(self.tree.font())
        if toolbar_font.pointSize() > 0:
            toolbar_font.setPointSize(toolbar_font.pointSize() + 1)
        else:
            toolbar_font.setPointSizeF(toolbar_font.pointSizeF() + 1.0)
        toolbar.setFont(toolbar_font)
        file_button.setFont(toolbar_font)
        menu.setFont(toolbar_font)
        left_layout.addWidget(self.tree, 1)

        tree_expand_row = QtWidgets.QHBoxLayout()
        tree_expand_row.setContentsMargins(8, 0, 8, 0)
        self.expand_all_button = QtWidgets.QPushButton("Expand all")
        self.collapse_all_button = QtWidgets.QPushButton("Collapse all")
        self.expand_all_button.clicked.connect(self.expand_all)
        self.collapse_all_button.clicked.connect(self.collapse_all)
        tree_expand_row.addWidget(self.expand_all_button)
        tree_expand_row.addWidget(self.collapse_all_button)
        left_layout.addLayout(tree_expand_row)

        tree_button_row = QtWidgets.QHBoxLayout()
        tree_button_row.setContentsMargins(8, 0, 8, 8)
        self.create_group_button = QtWidgets.QPushButton("Create workspace")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.create_group_button.clicked.connect(self.create_data_group)
        self.delete_button.clicked.connect(self.delete_selected)
        tree_button_row.addWidget(self.create_group_button)
        tree_button_row.addWidget(self.delete_button)
        left_layout.addLayout(tree_button_row)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        self.title_label = QtWidgets.QLabel("Project")
        title_font = QtGui.QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(self.title_label, 1)
        self.enabled_check = QtWidgets.QCheckBox("Enabled")
        self.enabled_check.toggled.connect(self._set_selected_enabled)
        title_row.addWidget(self.enabled_check)
        self.fit_weight_widget = QtWidgets.QWidget()
        fit_weight_layout = QtWidgets.QHBoxLayout(self.fit_weight_widget)
        fit_weight_layout.setContentsMargins(0, 0, 0, 0)
        fit_weight_layout.setSpacing(6)
        fit_weight_layout.addWidget(QtWidgets.QLabel("Fit weight"))
        self.fit_weight_spin = QtWidgets.QDoubleSpinBox()
        self.fit_weight_spin.setRange(0.0, 1.0e12)
        self.fit_weight_spin.setDecimals(6)
        self.fit_weight_spin.setSingleStep(0.1)
        self.fit_weight_spin.setValue(1.0)
        self.fit_weight_spin.valueChanged.connect(self._set_selected_dataset_fit_weight)
        fit_weight_layout.addWidget(self.fit_weight_spin)
        fit_weight_layout.addWidget(QtWidgets.QLabel("Scale"))
        self.scale_factor_spin = QtWidgets.QDoubleSpinBox()
        self.scale_factor_spin.setObjectName("dataset_scale_factor")
        self.scale_factor_spin.setRange(-1.0e12, 1.0e12)
        self.scale_factor_spin.setDecimals(6)
        self.scale_factor_spin.setSingleStep(0.1)
        self.scale_factor_spin.setValue(1.0)
        self.scale_factor_spin.valueChanged.connect(self._set_selected_dataset_scale_factor)
        fit_weight_layout.addWidget(self.scale_factor_spin)
        title_row.addWidget(self.fit_weight_widget)

        # Bulk editors for nested dataset groups: blank when descendants differ,
        # editing overwrites the fit weight / scale of every descendant dataset.
        self.group_bulk_widget = QtWidgets.QWidget()
        group_bulk_layout = QtWidgets.QHBoxLayout(self.group_bulk_widget)
        group_bulk_layout.setContentsMargins(0, 0, 0, 0)
        group_bulk_layout.setSpacing(6)
        group_bulk_layout.addWidget(QtWidgets.QLabel("Fit weight"))
        self.group_fit_weight_edit = QtWidgets.QLineEdit()
        self.group_fit_weight_edit.setObjectName("group_fit_weight_edit")
        self.group_fit_weight_edit.setPlaceholderText("(mixed)")
        self.group_fit_weight_edit.setMaximumWidth(90)
        self.group_fit_weight_edit.editingFinished.connect(
            lambda: self._set_group_bulk_value("fit_weight", self.group_fit_weight_edit.text())
        )
        group_bulk_layout.addWidget(self.group_fit_weight_edit)
        group_bulk_layout.addWidget(QtWidgets.QLabel("Scale"))
        self.group_scale_edit = QtWidgets.QLineEdit()
        self.group_scale_edit.setObjectName("group_scale_edit")
        self.group_scale_edit.setPlaceholderText("(mixed)")
        self.group_scale_edit.setMaximumWidth(90)
        self.group_scale_edit.editingFinished.connect(
            lambda: self._set_group_bulk_value("scale_factor", self.group_scale_edit.text())
        )
        group_bulk_layout.addWidget(self.group_scale_edit)
        title_row.addWidget(self.group_bulk_widget)
        self.details_label = QtWidgets.QLabel()
        self.details_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.details_scroll = QtWidgets.QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.details_widget = QtWidgets.QWidget()
        self.details_layout = QtWidgets.QVBoxLayout(self.details_widget)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(8)
        self.details_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.details_scroll.setWidget(self.details_widget)

        actions_row = QtWidgets.QHBoxLayout()
        self.import_dataset_button = QtWidgets.QPushButton("Import dataset")
        self.add_model_button = QtWidgets.QPushButton("Add model")
        self.view_slice_button = QtWidgets.QPushButton("View in data viewer")
        self.load_dataset_button = QtWidgets.QPushButton("Load now")
        self.add_mask_button = QtWidgets.QPushButton("Add mask")
        self.add_dataset_group_button = QtWidgets.QPushButton("New dataset group")
        self.save_dataset_button = QtWidgets.QPushButton("Save dataset")
        self.import_dataset_button.clicked.connect(self.import_dataset_dialog)
        self.add_model_button.clicked.connect(self.add_model_to_selection)
        self.view_slice_button.clicked.connect(self.open_slice_viewer_for_selection)
        self.load_dataset_button.clicked.connect(self.load_dataset_for_selection)
        self.add_mask_button.clicked.connect(self.add_mask_to_selection)
        self.add_dataset_group_button.clicked.connect(self.add_dataset_group_to_selection)
        self.save_dataset_button.clicked.connect(self.save_dataset_for_selection)
        actions_row.addWidget(self.import_dataset_button)
        actions_row.addWidget(self.add_model_button)
        actions_row.addWidget(self.view_slice_button)
        actions_row.addWidget(self.load_dataset_button)
        actions_row.addWidget(self.add_mask_button)
        actions_row.addWidget(self.add_dataset_group_button)
        actions_row.addWidget(self.save_dataset_button)
        actions_row.addStretch(1)

        mask_combo_class = _make_refreshing_combo_class()
        self.mask_type_combo = mask_combo_class(self._refresh_mask_type_combo)
        self.mask_type_combo.setObjectName("mask_type_combo")
        self.mask_type_combo.currentTextChanged.connect(self._set_selected_mask_type)
        self.mask_parameter_widget = QtWidgets.QWidget()
        self.mask_parameter_layout = QtWidgets.QGridLayout(self.mask_parameter_widget)
        self.mask_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.mask_parameter_layout.setColumnStretch(1, 1)
        model_combo_class = _make_refreshing_combo_class()
        self.model_type_combo = model_combo_class(self._refresh_model_type_combo)
        self.model_type_combo.setObjectName("model_type_combo")
        self.model_type_combo.currentTextChanged.connect(self._set_selected_model_type)
        self.model_parameter_widget = QtWidgets.QWidget()
        self.model_parameter_layout = QtWidgets.QGridLayout(self.model_parameter_widget)
        self.model_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.model_parameter_layout.setColumnStretch(1, 1)
        fit_editor = QtWidgets.QWidget()
        fit_editor_layout = QtWidgets.QGridLayout(fit_editor)
        fit_editor_layout.setContentsMargins(0, 0, 0, 0)
        fit_editor_layout.setColumnStretch(1, 1)
        self.fit_optimizer_combo = QtWidgets.QComboBox()
        self.fit_optimizer_combo.addItems(["least_squares"])
        self.fit_optimizer_combo.currentTextChanged.connect(self._set_selected_fit_optimizer)
        self.fit_optimizer_config_editor = QtWidgets.QLineEdit("{}")
        self.fit_optimizer_config_editor.setToolTip("JSON optimizer configuration for the selected fit state.")
        self.fit_optimizer_config_editor.editingFinished.connect(self._set_selected_fit_optimizer_config)
        self.fit_branch_check = QtWidgets.QCheckBox("Branch timeline")
        self.fit_now_button = QtWidgets.QPushButton("Fit now")
        self.fit_now_button.clicked.connect(self.fit_now_for_selection)
        fit_editor_layout.addWidget(QtWidgets.QLabel("Optimizer"), 0, 0)
        fit_editor_layout.addWidget(self.fit_optimizer_combo, 0, 1)
        fit_editor_layout.addWidget(QtWidgets.QLabel("Config"), 1, 0)
        fit_editor_layout.addWidget(self.fit_optimizer_config_editor, 1, 1)
        fit_editor_layout.addWidget(self.fit_branch_check, 2, 1)
        self.fit_editor_widget = fit_editor

        right_layout.addLayout(title_row)
        right_layout.addWidget(self.mask_type_combo)
        right_layout.addWidget(self.mask_parameter_widget)
        right_layout.addWidget(self.model_type_combo)
        right_layout.addWidget(self.model_parameter_widget)
        right_layout.addWidget(self.fit_editor_widget)
        right_layout.addWidget(self.details_scroll, 1)
        right_layout.addLayout(actions_row)
        right_layout.addWidget(self.fit_now_button)

        splitter.addWidget(right_panel)
        splitter.setSizes([360, 760])
        self._sync_window_title()

    def _refresh_tree(
        self,
        *,
        select_group: DataGroup | None = None,
        select_dataset: DatasetEntry | None = None,
        select_mask: MaskSpec | None = None,
        select_model: ModelComponentSpec | None = None,
        select_fit: FitTimelineEntry | None = None,
        select_dataset_group: DatasetGroup | None = None,
        edit_group: bool = False,
        edit_mask: bool = False,
        edit_model: bool = False,
    ) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        self._expanded_state = self._current_expanded_state()
        self._item_roles.clear()
        self._fit_item_roles.clear()
        self._dataset_group_roles.clear()
        self.tree.blockSignals(True)
        self.tree.clear()
        item_to_select = None
        for group in self.project.data_groups:
            ensure_fit_history(group)
            refresh_current_state_fit_entries(group)
            group_item = QtWidgets.QTreeWidgetItem([group.name])
            group_item.setFlags(group_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            _style_tree_hierarchy_item(group_item, bold=True, underline=True)
            _set_tree_item_icon(group_item, "folder")
            self._remember_item(group_item, "group", group)
            self.tree.addTopLevelItem(group_item)

            datasets_item = QtWidgets.QTreeWidgetItem(["Datasets"])
            _style_tree_hierarchy_item(datasets_item, bold=True)
            _set_tree_item_icon(datasets_item, "folder")
            self._remember_item(datasets_item, "datasets", group)
            group_item.addChild(datasets_item)
            found = self._render_dataset_node(
                datasets_item,
                group,
                group,
                select_dataset=select_dataset,
                select_mask=select_mask,
                select_dataset_group=select_dataset_group,
            )
            if found is not None and item_to_select is None:
                item_to_select = found

            models_item = QtWidgets.QTreeWidgetItem(["Models"])
            _style_tree_hierarchy_item(models_item, bold=True)
            _set_tree_item_icon(models_item, "model_folder")
            self._remember_item(models_item, "models", group)
            group_item.addChild(models_item)
            for name, model in group.models.items():
                model_item = QtWidgets.QTreeWidgetItem([name])
                _set_tree_item_icon(model_item, "model")
                if isinstance(model, ModelComponentSpec):
                    model_item.setFlags(model_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    self._remember_item(model_item, "model", group, model=model)
                    _style_enabled_tree_item(model_item, model.enabled)
                    if select_model is model:
                        item_to_select = model_item
                else:
                    self._remember_item(model_item, "fit_model_session", group)
                models_item.addChild(model_item)

            group_item.setExpanded(self._expanded_state.get(("group", id(group)), True))
            datasets_item.setExpanded(self._expanded_state.get(("datasets", id(group)), False))
            models_item.setExpanded(self._expanded_state.get(("models", id(group)), False))
            fits_item = QtWidgets.QTreeWidgetItem(["Fits"])
            _style_tree_hierarchy_item(fits_item, bold=True)
            _set_tree_item_icon(fits_item, "fit_folder")
            self._remember_item(fits_item, "fits", group)
            group_item.addChild(fits_item)
            for fit_entry in group.fits:
                fit_item = self._add_fit_tree_item(fits_item, group, fit_entry, select_fit=select_fit)
                if item_to_select is None and select_fit is not None:
                    item_to_select = self._selected_fit_tree_item(fit_item, select_fit)
            fits_item.setExpanded(self._expanded_state.get(("fits", id(group)), False))
            if select_group is group and item_to_select is None:
                item_to_select = group_item
        self.tree.blockSignals(False)
        if item_to_select is not None:
            self.tree.setCurrentItem(item_to_select)
            if edit_group:
                self.tree.editItem(item_to_select, 0)
            if edit_mask:
                self.tree.editItem(item_to_select, 0)
            if edit_model:
                self.tree.editItem(item_to_select, 0)
        self._sync_details()
        self.refresh_open_slice_viewers()

    def _render_dataset_node(
        self,
        parent_item: Any,
        group: DataGroup,
        node: Any,
        *,
        select_dataset: DatasetEntry | None,
        select_mask: MaskSpec | None,
        select_dataset_group: DatasetGroup | None,
    ) -> Any:
        """Recursively render a group's datasets and nested subgroups. Returns any matched item."""

        from PySide6 import QtCore, QtWidgets

        found = None
        for dataset in node.datasets:
            dataset_item = QtWidgets.QTreeWidgetItem([dataset.name])
            dataset_item.setFlags(dataset_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            _set_tree_item_icon(dataset_item, "dataset")
            self._remember_item(dataset_item, "dataset", group, dataset)
            _style_enabled_tree_item(dataset_item, dataset.enabled)
            parent_item.addChild(dataset_item)
            masks_item = QtWidgets.QTreeWidgetItem(["Masks"])
            _set_tree_item_icon(masks_item, "mask_folder")
            _style_enabled_tree_item(masks_item, dataset.enabled)
            self._remember_item(masks_item, "masks", group, dataset)
            dataset_item.addChild(masks_item)
            for mask in dataset.masks:
                mask_item = QtWidgets.QTreeWidgetItem([mask.name])
                mask_item.setFlags(mask_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                _set_tree_item_icon(mask_item, "mask")
                self._remember_item(mask_item, "mask", group, dataset, mask)
                _style_enabled_tree_item(mask_item, mask.enabled)
                masks_item.addChild(mask_item)
                if select_mask is mask and found is None:
                    found = mask_item
            if select_dataset is dataset and found is None:
                found = dataset_item
            dataset_item.setExpanded(self._expanded_state.get(("dataset", id(dataset)), False))
            masks_item.setExpanded(self._expanded_state.get(("masks", id(dataset)), False))

        for subgroup in node.subgroups:
            subgroup_item = QtWidgets.QTreeWidgetItem([subgroup.name])
            subgroup_item.setFlags(subgroup_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            _set_tree_item_icon(subgroup_item, "folder")
            self._remember_item(subgroup_item, "dataset_group", group, node=subgroup)
            parent_item.addChild(subgroup_item)
            gmasks_item = QtWidgets.QTreeWidgetItem(["Masks"])
            _set_tree_item_icon(gmasks_item, "mask_folder")
            self._remember_item(gmasks_item, "group_masks", group, node=subgroup)
            subgroup_item.addChild(gmasks_item)
            for mask in subgroup.masks:
                gmask_item = QtWidgets.QTreeWidgetItem([mask.name])
                gmask_item.setFlags(gmask_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                _set_tree_item_icon(gmask_item, "mask")
                self._remember_item(gmask_item, "group_mask", group, mask=mask, node=subgroup)
                _style_enabled_tree_item(gmask_item, mask.enabled)
                gmasks_item.addChild(gmask_item)
                if select_mask is mask and found is None:
                    found = gmask_item
            child_found = self._render_dataset_node(
                subgroup_item,
                group,
                subgroup,
                select_dataset=select_dataset,
                select_mask=select_mask,
                select_dataset_group=select_dataset_group,
            )
            if child_found is not None and found is None:
                found = child_found
            if select_dataset_group is subgroup and found is None:
                found = subgroup_item
            subgroup_item.setExpanded(self._expanded_state.get(("dataset_group", id(subgroup)), True))
            gmasks_item.setExpanded(self._expanded_state.get(("group_masks", id(subgroup)), False))
        return found

    def _current_expanded_state(self) -> dict[tuple[Any, ...], bool]:
        state: dict[tuple[Any, ...], bool] = {}
        if self.tree is None:
            return state
        for index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(index)
            group, _entry, _mask, _model, role = self._objects_for_item(group_item)
            if role != "group" or group is None:
                continue
            state[("group", id(group))] = group_item.isExpanded()
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                _child_group, _child_entry, _child_mask, _child_model, child_role = self._objects_for_item(child)
                if child_role in {"datasets", "models"}:
                    state[(child_role, id(group))] = child.isExpanded()
                if child_role == "fits":
                    state[("fits", id(group))] = child.isExpanded()
                    self._collect_fit_expanded_state(child, state)
                if child_role == "datasets":
                    for dataset_index in range(child.childCount()):
                        dataset_item = child.child(dataset_index)
                        _group, dataset, _mask, _model, dataset_role = self._objects_for_item(dataset_item)
                        if dataset_role != "dataset" or dataset is None:
                            continue
                        state[("dataset", id(dataset))] = dataset_item.isExpanded()
                        for mask_parent_index in range(dataset_item.childCount()):
                            masks_item = dataset_item.child(mask_parent_index)
                            _group, masks_dataset, _mask, _model, masks_role = self._objects_for_item(masks_item)
                            if masks_role == "masks" and masks_dataset is not None:
                                state[("masks", id(masks_dataset))] = masks_item.isExpanded()
        return state

    def _collect_fit_expanded_state(self, item: Any, state: dict[tuple[Any, ...], bool]) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            fit_entry = self._fit_entry_for_item(child)
            if fit_entry is not None:
                state[("fit", id(fit_entry))] = child.isExpanded()
                self._collect_fit_expanded_state(child, state)

    def _remember_item(
        self,
        item: Any,
        role: str,
        group: DataGroup | None = None,
        entry: DatasetEntry | None = None,
        mask: MaskSpec | None = None,
        model: ModelComponentSpec | None = None,
        node: DatasetGroup | None = None,
    ) -> None:
        self._item_roles[id(item)] = (role, group, entry, mask, model)
        if node is not None:
            self._dataset_group_roles[id(item)] = node

    def _dataset_group_for_item(self, item: Any) -> DatasetGroup | None:
        if item is None:
            return None
        return self._dataset_group_roles.get(id(item))

    def _add_fit_tree_item(
        self,
        parent_item: Any,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        select_fit: FitTimelineEntry | None = None,
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        item = QtWidgets.QTreeWidgetItem([fit_entry.name])
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        role = "fit_timeline" if fit_entry.kind == "timeline" else "fit"
        icon_kind = {
            "timeline": "fit_folder",
            "initial": "fit_initial",
            "current": "fit_current",
            "result": "fit_result",
        }.get(fit_entry.kind, "fit_result")
        _set_tree_item_icon(item, icon_kind)
        self._remember_item(item, role, group)
        self._fit_item_roles[id(item)] = fit_entry
        parent_item.addChild(item)
        for child in fit_entry.children:
            self._add_fit_tree_item(item, group, child, select_fit=select_fit)
        item.setExpanded(
            self._expanded_state.get(
                ("fit", id(fit_entry)),
                fit_entry.kind == "timeline" or _fit_entry_in_tree(fit_entry.children, select_fit),
            )
        )
        return item

    def _selected_fit_tree_item(self, item: Any, select_fit: FitTimelineEntry) -> Any | None:
        if self._fit_entry_for_item(item) is select_fit:
            return item
        for index in range(item.childCount()):
            found = self._selected_fit_tree_item(item.child(index), select_fit)
            if found is not None:
                return found
        return None

    def _tree_item_changed(self, item: Any, column: int) -> None:
        if column != 0:
            return
        changed = False
        group, entry, mask, model, role = self._objects_for_item(item)
        if role == "group" and group is not None:
            new_name = item.text(0).strip() or next_data_group_name(self.project.data_groups)
            changed = group.name != new_name
            group.name = new_name
            item.setText(0, group.name)
        elif role == "dataset" and group is not None and entry is not None:
            new_name = _unique_name(
                item.text(0).strip() or entry.name,
                [dataset.name for dataset in group.datasets if dataset is not entry],
            )
            changed = entry.name != new_name
            entry.name = new_name
            item.setText(0, entry.name)
        elif role in {"mask", "group_mask"} and mask is not None:
            new_name = item.text(0).strip() or "mask"
            changed = mask.name != new_name
            mask.name = new_name
            item.setText(0, mask.name)
        elif role == "dataset_group" and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None:
                existing = [sub.name for sub in group.iter_subgroups() if sub is not subgroup]
                new_name = _unique_name(item.text(0).strip() or subgroup.name, existing)
                changed = subgroup.name != new_name
                subgroup.name = new_name
                item.setText(0, subgroup.name)
        elif role == "model" and group is not None and model is not None:
            old_name = _model_key(group, model)
            new_name = _unique_name(item.text(0).strip() or model.name, [name for name in group.models if name != old_name])
            if old_name is not None and old_name != new_name:
                del group.models[old_name]
                group.models[new_name] = model
                changed = True
            changed = changed or model.name != new_name
            model.name = new_name
            item.setText(0, model.name)
        elif role in {"fit", "fit_timeline"}:
            fit_entry = self._fit_entry_for_item(item)
            if fit_entry is not None:
                new_name = item.text(0).strip() or fit_entry.name
                changed = fit_entry.name != new_name
                fit_entry.name = new_name
                item.setText(0, fit_entry.name)
        if changed:
            if role in {"dataset", "mask", "model", "dataset_group", "group_mask"} and group is not None:
                self._record_data_group_state_change(group)
            self._mark_dirty()
        self._sync_details()

    def _current_item(self) -> Any:
        return self.tree.currentItem()

    def _objects_for_item(
        self,
        item: Any,
    ) -> tuple[DataGroup | None, DatasetEntry | None, MaskSpec | None, ModelComponentSpec | None, str]:
        if item is None:
            return None, None, None, None, "project"
        role, group, entry, mask, model = self._item_roles.get(id(item), ("project", None, None, None, None))
        return group, entry, mask, model, role

    def _fit_entry_for_item(self, item: Any) -> FitTimelineEntry | None:
        if item is None:
            return None
        return self._fit_item_roles.get(id(item))

    def _sync_details(self) -> None:
        group, entry, mask, model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        can_import = role in {"group", "datasets", "dataset_group"}
        can_add_model = role in {"group", "models"}
        self._sync_selected_state_controls(role, entry, mask, model)
        self.import_dataset_button.setVisible(can_import)
        self.add_model_button.setVisible(can_add_model)
        self.view_slice_button.setVisible(
            role in {"group", "datasets", "dataset", "masks", "mask", "dataset_group", "group_masks", "group_mask"}
        )
        self.view_slice_button.setEnabled(bool(group is not None and _has_slice_viewer_candidates(group)))
        self.load_dataset_button.setVisible(
            role == "dataset" and entry is not None and _dataset_can_load(entry)
        )
        self.add_mask_button.setVisible(role in {"dataset", "masks", "group_masks", "dataset_group"})
        self.add_dataset_group_button.setVisible(role in {"group", "datasets", "dataset_group"})
        self.save_dataset_button.setVisible(role == "dataset")
        self.save_dataset_button.setEnabled(
            bool(role == "dataset" and entry is not None and _dataset_can_save(entry))
        )
        self.delete_button.setEnabled(
            role in {"group", "dataset", "mask", "model", "fit", "fit_timeline", "dataset_group", "group_mask"}
        )
        mask_editing = role in {"mask", "group_mask"}
        self.mask_type_combo.setVisible(mask_editing)
        self.mask_parameter_widget.setVisible(mask_editing)
        self.model_type_combo.setVisible(role == "model")
        self.model_parameter_widget.setVisible(role == "model")
        self.fit_editor_widget.setVisible(role == "fit" and fit_entry is not None)
        self.fit_now_button.setVisible(role == "fit" and fit_entry is not None)

        if role == "group" and group is not None:
            self.title_label.setText(group.name)
            ensure_fit_history(group)
            result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
            self._set_details_text(
                f"Workspace\n\nDatasets: {len(group.datasets)}\nModels: {len(group.models)}\nFit results: {result_count}"
            )
        elif role == "datasets" and group is not None:
            self.title_label.setText(f"{group.name} / Datasets")
            self._set_details_text(f"{len(group.datasets)} dataset(s)")
        elif role == "dataset" and entry is not None:
            self.title_label.setText(entry.name)
            self._set_dataset_details(entry, group)
        elif role == "masks" and entry is not None:
            self.title_label.setText(f"{entry.name} / Masks")
            self._set_details_text(f"{len(entry.masks)} mask(s)")
        elif role == "mask" and entry is not None and mask is not None:
            self.title_label.setText(mask.name)
            label = MASK_TYPE_DEFINITIONS.get(mask.type, {}).get("label", mask.type)
            self._set_details_text(
                f"Mask\n\nType: {label}\nEnabled: {mask.enabled}\nInvert: {mask.invert}\nAdditive: {mask.additive}"
            )
            self._sync_mask_editor(mask, entry)
        elif role == "dataset_group":
            subgroup = self._dataset_group_for_item(self._current_item())
            name = subgroup.name if subgroup is not None else "Dataset group"
            self.title_label.setText(name)
            count = len(list(subgroup.iter_datasets())) if subgroup is not None else 0
            mask_count = len(subgroup.masks) if subgroup is not None else 0
            self._set_details_text(
                f"Dataset group\n\nDatasets (incl. nested): {count}\nShared masks: {mask_count}"
            )
        elif role == "group_masks":
            subgroup = self._dataset_group_for_item(self._current_item())
            name = subgroup.name if subgroup is not None else "-"
            mask_count = len(subgroup.masks) if subgroup is not None else 0
            self.title_label.setText(f"{name} / Masks")
            self._set_details_text(f"{mask_count} shared mask(s)")
        elif role == "group_mask" and mask is not None:
            subgroup = self._dataset_group_for_item(self._current_item())
            self.title_label.setText(mask.name)
            label = MASK_TYPE_DEFINITIONS.get(mask.type, {}).get("label", mask.type)
            self._set_details_text(
                f"Shared mask\n\nType: {label}\nEnabled: {mask.enabled}\nInvert: {mask.invert}\nAdditive: {mask.additive}"
            )
            reference = _group_reference_dataset(subgroup) if subgroup is not None else None
            self._sync_mask_editor(mask, reference)
        elif role == "models" and group is not None:
            self.title_label.setText(f"{group.name} / Models")
            self._set_details_text(f"{len(group.models)} model(s)")
        elif role == "fits" and group is not None:
            ensure_fit_history(group)
            self.title_label.setText(f"{group.name} / Fits")
            result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
            self._set_details_text(f"Fit history\n\nResults: {result_count}")
        elif role in {"fit", "fit_timeline"} and group is not None and fit_entry is not None:
            self.title_label.setText(fit_entry.name)
            if role == "fit" and not self._restoring_fit_selection:
                self._restore_selected_fit_state(group, fit_entry)
            self._sync_fit_editor(fit_entry)
            self._set_details_text(fit_details_text(fit_entry))
        elif role == "model" and model is not None:
            self.title_label.setText(model.name)
            label = MODEL_TYPE_DEFINITIONS.get(model.type, {}).get("label", model.type)
            self._set_details_text(f"Model\n\nType: {label}\nEnabled: {model.enabled}")
            self._sync_model_editor(model)
        elif role == "fit_model_session" and group is not None:
            self.title_label.setText("Fit model session")
            self._set_details_text("Existing fit session")
        else:
            self.title_label.setText("Project")
            self._set_details_text(f"{len(self.project.data_groups)} workspace(s)")
        if role not in {"mask", "group_mask"}:
            self._clear_mask_parameter_editor()
        if role != "model":
            self._clear_model_parameter_editor()
        if role != "fit":
            self.fit_branch_check.setChecked(False)

    def _sync_selected_state_controls(
        self,
        role: str,
        entry: DatasetEntry | None,
        mask: MaskSpec | None,
        model: ModelComponentSpec | None,
    ) -> None:
        has_enabled = role in {"dataset", "mask", "model"}
        self.enabled_check.setVisible(has_enabled)
        self.fit_weight_widget.setVisible(role == "dataset")
        self.enabled_check.blockSignals(True)
        try:
            if role == "dataset" and entry is not None:
                self.enabled_check.setChecked(bool(entry.enabled))
            elif role == "mask" and mask is not None:
                self.enabled_check.setChecked(bool(mask.enabled))
            elif role == "model" and model is not None:
                self.enabled_check.setChecked(bool(model.enabled))
            else:
                self.enabled_check.setChecked(False)
        finally:
            self.enabled_check.blockSignals(False)
        self.fit_weight_spin.blockSignals(True)
        try:
            self.fit_weight_spin.setValue(float(entry.fit_weight) if role == "dataset" and entry is not None else 1.0)
        finally:
            self.fit_weight_spin.blockSignals(False)
        self.scale_factor_spin.blockSignals(True)
        try:
            self.scale_factor_spin.setValue(
                float(entry.scale_factor) if role == "dataset" and entry is not None else 1.0
            )
        finally:
            self.scale_factor_spin.blockSignals(False)
        self._sync_group_bulk_controls(role)

    def _group_bulk_datasets(self, role: str) -> list[DatasetEntry]:
        item = self._current_item()
        _group, _entry, _mask, _model, _item_role = self._objects_for_item(item)
        if role in {"dataset_group", "group_masks"}:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None:
                return list(subgroup.iter_datasets())
        return []

    def _sync_group_bulk_controls(self, role: str) -> None:
        show = role in {"dataset_group", "group_masks"}
        self.group_bulk_widget.setVisible(show)
        if not show:
            return
        datasets = self._group_bulk_datasets(role)
        for edit, attribute in (
            (self.group_fit_weight_edit, "fit_weight"),
            (self.group_scale_edit, "scale_factor"),
        ):
            values = {float(getattr(dataset, attribute)) for dataset in datasets}
            edit.blockSignals(True)
            try:
                edit.setText(_format_number(next(iter(values))) if len(values) == 1 else "")
            finally:
                edit.blockSignals(False)

    def _set_group_bulk_value(self, attribute: str, text: str) -> None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        text = text.strip()
        if not text or role not in {"dataset_group", "group_masks"}:
            return
        try:
            value = float(text)
        except ValueError:
            return
        datasets = self._group_bulk_datasets(role)
        changed = False
        for dataset in datasets:
            if float(getattr(dataset, attribute)) != value:
                setattr(dataset, attribute, value)
                changed = True
        if not changed:
            return
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._sync_details()

    def _set_selected_enabled(self, checked: bool) -> None:
        group, entry, mask, model, role = self._objects_for_item(self._current_item())
        changed = False
        if role == "dataset" and entry is not None:
            changed = entry.enabled != bool(checked)
            entry.enabled = bool(checked)
        elif role == "mask" and mask is not None:
            changed = mask.enabled != bool(checked)
            mask.enabled = bool(checked)
        elif role == "model" and model is not None:
            changed = model.enabled != bool(checked)
            model.enabled = bool(checked)
        if not changed:
            return
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_tree(select_group=group, select_dataset=entry, select_mask=mask, select_model=model)

    def _set_selected_dataset_fit_weight(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        weight = float(value)
        if entry.fit_weight == weight:
            return
        entry.fit_weight = weight
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._sync_details()

    def _set_selected_dataset_scale_factor(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        scale = float(value)
        if entry.scale_factor == scale:
            return
        entry.scale_factor = scale
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._sync_details()

    def _set_details_text(self, text: str) -> None:
        from PySide6 import QtCore, QtWidgets

        self.details_label.setText(text)
        self._clear_details_panel()
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_layout.addWidget(label)
        self.details_layout.addStretch(1)

    def _set_dataset_details(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        # Point-list data is cheap to load; populate it lazily (e.g. after a
        # project reload) so the details and rebin panels have real columns.
        if dataset.data is None and data_type_container(dataset.data_type) == "point_list":
            try:
                _load_point_list_dataset(dataset)
            except Exception:
                pass
        self.details_label.setText(dataset_details_text(dataset, group=group))
        self._clear_details_panel()
        is_point_list = isinstance(dataset.data, PointListData)
        for title, lines in dataset_detail_sections(dataset, group=group):
            if title == "Axes":
                self.details_layout.addWidget(self._dataset_axes_group_box(dataset, group, lines))
                if is_point_list:
                    self.details_layout.addWidget(self._dataset_point_list_group_box(dataset, group))
            elif title == "Dataset":
                self.details_layout.addWidget(self._dataset_type_group_box(dataset, group, lines))
            elif title == "Metadata":
                self.details_layout.addWidget(self._dataset_metadata_group_box(dataset))
            else:
                self.details_layout.addWidget(self._details_group_box(title, lines))
        self.details_layout.addStretch(1)

    def _dataset_point_list_group_box(self, dataset: DatasetEntry, group: DataGroup | None) -> Any:
        from PySide6 import QtWidgets

        config = point_list_config(dataset)
        data = dataset.data
        columns = list(data.column_names)
        definition = DATA_TYPE_DEFINITIONS.get(dataset.data_type, {})

        group_box = QtWidgets.QGroupBox("Variables and Channels")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)

        # Independent coordinates (comma-separated column names).
        coord_row = QtWidgets.QHBoxLayout()
        coord_row.addWidget(QtWidgets.QLabel("Coordinates"))
        coord_edit = QtWidgets.QLineEdit(", ".join(config.get("coordinate_names", [])))
        coord_edit.setObjectName("point_list_coordinates")
        coord_edit.setToolTip("Comma-separated column names to use as independent coordinates.")
        coord_edit.editingFinished.connect(
            lambda editor=coord_edit: self._set_point_list_coordinates(dataset, group, editor.text())
        )
        coord_row.addWidget(coord_edit, 1)
        layout.addLayout(coord_row)

        # Channels: value + error column per channel.
        channels_box = QtWidgets.QGroupBox("Channels")
        channels_layout = QtWidgets.QGridLayout(channels_box)
        channels_layout.setContentsMargins(8, 6, 8, 6)
        channels_layout.addWidget(_bold_label(QtWidgets, "Value"), 0, 0)
        channels_layout.addWidget(_bold_label(QtWidgets, "Error"), 0, 1)
        for row, channel in enumerate(config.get("channels", []), start=1):
            value_combo = QtWidgets.QComboBox()
            value_combo.addItems(columns)
            value_combo.setCurrentText(str(channel.get("value", "")))
            value_combo.currentTextChanged.connect(
                lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "value", text)
            )
            error_combo = QtWidgets.QComboBox()
            error_combo.addItems(["(none)", *columns])
            error_combo.setCurrentText(str(channel.get("error") or "(none)"))
            error_combo.currentTextChanged.connect(
                lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "error", text)
            )
            channels_layout.addWidget(value_combo, row, 0)
            channels_layout.addWidget(error_combo, row, 1)
            remove_button = QtWidgets.QPushButton("Remove")
            remove_button.clicked.connect(
                lambda _checked=False, index=row - 1: self._remove_point_list_channel(dataset, group, index)
            )
            channels_layout.addWidget(remove_button, row, 2)
        add_button = QtWidgets.QPushButton("Add channel")
        add_button.clicked.connect(lambda: self._add_point_list_channel(dataset, group))
        channels_layout.addWidget(add_button, len(config.get("channels", [])) + 1, 0)
        layout.addWidget(channels_box)

        if definition.get("scale"):
            layout.addWidget(self._point_list_scale_box(dataset, group, config, columns))
        if definition.get("susceptibility"):
            layout.addWidget(self._point_list_susceptibility_box(dataset, group, config))
        if definition.get("wavelength"):
            layout.addWidget(self._point_list_wavelength_box(dataset, group, config))
        return group_box

    def _point_list_scale_box(self, dataset, group, config, columns) -> Any:
        from PySide6 import QtWidgets

        scale = config["scale"]
        box = QtWidgets.QGroupBox("Scale && units")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.addWidget(QtWidgets.QLabel("Channel"), 0, 0)
        channel_combo = QtWidgets.QComboBox()
        channel_combo.addItems([str(c["label"]) for c in config.get("channels", [])])
        channel_combo.setCurrentText(str(scale.get("channel", "")))
        channel_combo.currentTextChanged.connect(
            lambda text: self._set_point_list_scale(dataset, group, "channel", text)
        )
        grid.addWidget(channel_combo, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Factor"), 1, 0)
        factor_spin = QtWidgets.QDoubleSpinBox()
        factor_spin.setDecimals(8)
        factor_spin.setRange(-1.0e12, 1.0e12)
        factor_spin.setValue(float(scale.get("factor", 1.0)))
        factor_spin.setObjectName("point_list_scale_factor")
        factor_spin.valueChanged.connect(
            lambda value: self._set_point_list_scale(dataset, group, "factor", value)
        )
        grid.addWidget(factor_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Units"), 2, 0)
        units_edit = QtWidgets.QLineEdit(str(scale.get("units", "")))
        units_edit.editingFinished.connect(
            lambda editor=units_edit: self._set_point_list_scale(dataset, group, "units", editor.text())
        )
        grid.addWidget(units_edit, 2, 1)
        return box

    def _point_list_susceptibility_box(self, dataset, group, config) -> Any:
        from PySide6 import QtWidgets

        susc = config["susceptibility"]
        box = QtWidgets.QGroupBox("Susceptibility (moment / field)")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        enable = QtWidgets.QCheckBox("Divide moment by field")
        enable.setObjectName("point_list_susceptibility_enabled")
        enable.setChecked(bool(susc.get("enabled", False)))
        enable.toggled.connect(lambda checked: self._set_point_list_susceptibility(dataset, group, "enabled", checked))
        grid.addWidget(enable, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Moment"), 1, 0)
        moment_combo = QtWidgets.QComboBox()
        moment_combo.addItems([str(c["label"]) for c in config.get("channels", [])])
        moment_combo.setCurrentText(str(susc.get("moment", "")))
        moment_combo.currentTextChanged.connect(
            lambda text: self._set_point_list_susceptibility(dataset, group, "moment", text)
        )
        grid.addWidget(moment_combo, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Field"), 2, 0)
        field_combo = QtWidgets.QComboBox()
        field_combo.addItems(list(dataset.data.column_names))
        field_combo.setCurrentText(str(susc.get("field", "")))
        field_combo.currentTextChanged.connect(
            lambda text: self._set_point_list_susceptibility(dataset, group, "field", text)
        )
        grid.addWidget(field_combo, 2, 1)
        return box

    def _point_list_wavelength_box(self, dataset, group, config) -> Any:
        from PySide6 import QtWidgets

        wavelength = config["wavelength"]
        box = QtWidgets.QGroupBox("Neutron wavelength -> q")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.addWidget(QtWidgets.QLabel("2theta column"), 0, 0)
        two_theta_combo = QtWidgets.QComboBox()
        two_theta_combo.addItems(list(dataset.data.column_names))
        two_theta_combo.setCurrentText(str(wavelength.get("two_theta", "")))
        two_theta_combo.currentTextChanged.connect(
            lambda text: self._set_point_list_wavelength(dataset, group, "two_theta", text)
        )
        grid.addWidget(two_theta_combo, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Wavelength (A)"), 1, 0)
        wavelength_spin = QtWidgets.QDoubleSpinBox()
        wavelength_spin.setObjectName("point_list_wavelength")
        wavelength_spin.setDecimals(5)
        wavelength_spin.setRange(0.0, 100.0)
        wavelength_spin.setValue(float(wavelength.get("value", 0.0)))
        wavelength_spin.valueChanged.connect(
            lambda value: self._set_point_list_wavelength(dataset, group, "value", value)
        )
        grid.addWidget(wavelength_spin, 1, 1)
        return box

    def _set_point_list_coordinates(self, dataset, group, text: str) -> None:
        names = [part.strip() for part in text.split(",") if part.strip()]
        columns = dataset.data.column_names if isinstance(dataset.data, PointListData) else []
        names = [name for name in names if name in columns]
        config = point_list_config(dataset)
        if config.get("coordinate_names") == names:
            return
        config["coordinate_names"] = names
        self._after_point_list_changed(dataset, group)

    def _set_point_list_channel(self, dataset, group, index: int, key: str, text: str) -> None:
        config = point_list_config(dataset)
        channels = config.get("channels", [])
        if not (0 <= index < len(channels)):
            return
        value = None if (key == "error" and text == "(none)") else text
        if channels[index].get(key) == value:
            return
        channels[index][key] = value
        if key == "value":
            channels[index]["label"] = value
        self._after_point_list_changed(dataset, group)

    def _add_point_list_channel(self, dataset, group) -> None:
        config = point_list_config(dataset)
        columns = list(dataset.data.column_names)
        if not columns:
            return
        config.setdefault("channels", []).append(
            {"label": columns[0], "value": columns[0], "error": None}
        )
        self._after_point_list_changed(dataset, group)

    def _remove_point_list_channel(self, dataset, group, index: int) -> None:
        config = point_list_config(dataset)
        channels = config.get("channels", [])
        if 0 <= index < len(channels):
            channels.pop(index)
            self._after_point_list_changed(dataset, group)

    def _set_point_list_scale(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        scale = config.setdefault("scale", {})
        new_value = float(value) if key == "factor" else value
        if scale.get(key) == new_value:
            return
        scale[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _set_point_list_susceptibility(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        susc = config.setdefault("susceptibility", {})
        new_value = bool(value) if key == "enabled" else value
        if susc.get(key) == new_value:
            return
        susc[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _set_point_list_wavelength(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        wavelength = config.setdefault("wavelength", {})
        new_value = float(value) if key == "value" else value
        if wavelength.get(key) == new_value:
            return
        wavelength[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _after_point_list_changed(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._set_dataset_details(dataset, group)

    def _dataset_type_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        lines: list[str],
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        group_box = QtWidgets.QGroupBox("Dataset")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        label = QtWidgets.QLabel("\n".join(lines) if lines else "-")
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)

        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("Data type"))
        type_combo = QtWidgets.QComboBox()
        type_combo.setObjectName("dataset_data_type")
        for name, type_label in available_data_types():
            type_combo.addItem(type_label, name)
        current = dataset.data_type or DEFAULT_DATA_TYPE
        index = type_combo.findData(current)
        type_combo.setCurrentIndex(max(index, 0))
        type_combo.currentIndexChanged.connect(
            lambda _index, combo=type_combo: self._set_selected_data_type(dataset, group, combo.currentData())
        )
        type_row.addWidget(type_combo, 1)
        layout.addLayout(type_row)
        return group_box

    def _set_selected_data_type(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        data_type: str,
    ) -> None:
        if not data_type or data_type == dataset.data_type:
            return
        set_dataset_data_type(dataset, data_type)
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._set_dataset_details(dataset, group)

    def _dataset_axes_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        lines: list[str],
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        group_box = QtWidgets.QGroupBox("Axes")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        label = QtWidgets.QLabel("\n".join(lines) if lines else "-")
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)

        if not _dataset_can_rebin(dataset):
            return group_box

        rebin_box = QtWidgets.QGroupBox("Rebin")
        rebin_layout = QtWidgets.QVBoxLayout(rebin_box)
        rebin_layout.setContentsMargins(10, 8, 10, 8)

        config = dataset_rebin_config(dataset)
        enable_check = QtWidgets.QCheckBox("Use rebinned data")
        enable_check.setObjectName("dataset_rebin_enabled")
        enable_check.setChecked(bool(config.get("enabled", False)))
        enable_check.toggled.connect(lambda checked: self._set_dataset_rebin_enabled(dataset, group, checked))
        rebin_layout.addWidget(enable_check)

        controls = QtWidgets.QWidget()
        controls.setObjectName("dataset_rebin_controls")
        controls.setVisible(bool(config.get("enabled", False)))
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        show_vectors = isinstance(dataset.data, MDHistoData)
        headers = ["Axis"]
        if show_vectors:
            headers.append("Coord axis")
        headers.extend(["Lower", "Upper", "Bins", "Step"])
        last_column = len(headers) - 1
        controls_layout.setColumnStretch(last_column, 1)
        for column, text in enumerate(headers):
            header = QtWidgets.QLabel(text)
            header.setStyleSheet("font-weight: 600")
            controls_layout.addWidget(header, 0, column)
        for row, axis_config in enumerate(config.get("axes", []), start=1):
            axis_config = _sanitize_rebin_axis_config(axis_config)
            axis_label = QtWidgets.QLabel(f"{axis_config.get('name', f'Axis {row}')} ({axis_config.get('units', '-') or '-'})")
            lower_edit = QtWidgets.QLineEdit(_format_number(axis_config["lower"]))
            upper_edit = QtWidgets.QLineEdit(_format_number(axis_config["upper"]))
            bins_edit = QtWidgets.QLineEdit(str(int(axis_config["num_bins"])))
            step_edit = QtWidgets.QLineEdit(_format_number(axis_config["step_size"]))
            for editor in (lower_edit, upper_edit, bins_edit, step_edit):
                editor.setMinimumWidth(72)
            lower_edit.editingFinished.connect(
                lambda row=row - 1, editor=lower_edit: self._set_dataset_rebin_axis_value(dataset, group, row, "lower", editor.text())
            )
            upper_edit.editingFinished.connect(
                lambda row=row - 1, editor=upper_edit: self._set_dataset_rebin_axis_value(dataset, group, row, "upper", editor.text())
            )
            bins_edit.editingFinished.connect(
                lambda row=row - 1, editor=bins_edit: self._set_dataset_rebin_axis_value(dataset, group, row, "num_bins", editor.text())
            )
            step_edit.editingFinished.connect(
                lambda row=row - 1, editor=step_edit: self._set_dataset_rebin_axis_value(dataset, group, row, "step_size", editor.text())
            )
            controls_layout.addWidget(axis_label, row, 0)
            column = 1
            if show_vectors:
                vector_edit = QtWidgets.QLineEdit(_parameter_to_text(axis_config.get("vector", [])))
                vector_edit.setMinimumWidth(110)
                vector_edit.setToolTip(
                    "Projection vector defining this rebin coordinate as a linear combination of the dataset axes. "
                    "Defaults to the dataset coordinate axis. Example for a 4D dataset: [1, 0, 0, 0]."
                )
                vector_edit.editingFinished.connect(
                    lambda row=row - 1, editor=vector_edit: self._set_dataset_rebin_axis_vector(dataset, group, row, editor.text())
                )
                controls_layout.addWidget(vector_edit, row, column)
                column += 1
            controls_layout.addWidget(lower_edit, row, column)
            controls_layout.addWidget(upper_edit, row, column + 1)
            controls_layout.addWidget(bins_edit, row, column + 2)
            controls_layout.addWidget(step_edit, row, column + 3)

        option_row = QtWidgets.QHBoxLayout()
        fractional_check = QtWidgets.QCheckBox("Fractional binning")
        fractional_check.setObjectName("dataset_rebin_fractional")
        fractional_check.setChecked(bool(config.get("fractional", False)))
        fractional_check.toggled.connect(lambda checked: self._set_dataset_rebin_option(dataset, group, "fractional", checked))
        create_button = QtWidgets.QPushButton("Create dataset from rebin")
        create_button.setEnabled(_dataset_can_rebin(dataset))
        create_button.clicked.connect(self.materialize_rebin_for_selection)
        option_row.addWidget(fractional_check)
        option_row.addWidget(create_button)
        option_row.addStretch(1)
        controls_layout.addLayout(option_row, len(config.get("axes", [])) + 1, 0, 1, last_column + 1)
        rebin_layout.addWidget(controls)
        layout.addWidget(rebin_box)
        return group_box

    def _details_group_box(self, title: str, lines: list[str]) -> Any:
        from PySide6 import QtCore, QtWidgets

        group_box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        text = "\n".join(lines) if lines else "-"
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        return group_box

    def _dataset_metadata_group_box(self, dataset: DatasetEntry) -> Any:
        from PySide6 import QtCore, QtWidgets

        group_box = QtWidgets.QGroupBox("Metadata")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        metadata = _dataset_metadata_mapping(dataset)
        if not metadata and not dataset.parameters:
            label = QtWidgets.QLabel("No additional metadata.")
            label.setWordWrap(True)
            layout.addWidget(label)
            return group_box

        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("dataset_metadata_tree")
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Field", "Value"])
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree.setMinimumHeight(120)
        tree.setMaximumHeight(260)
        tree.header().setStretchLastSection(True)
        tree.header().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        for key in sorted(metadata):
            _add_metadata_tree_item(tree, str(key), metadata[key])
        if dataset.parameters:
            parameters_item = _add_metadata_tree_item(tree, "Parameters", dict(dataset.parameters))
            parameters_item.setExpanded(True)
        for index in range(min(4, tree.topLevelItemCount())):
            tree.topLevelItem(index).setExpanded(True)
        tree.resizeColumnToContents(0)
        layout.addWidget(tree)
        return group_box

    def _set_dataset_rebin_enabled(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        checked: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        if bool(config.get("enabled", False)) == bool(checked):
            return
        config["enabled"] = bool(checked)
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_option(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        key: str,
        checked: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        if bool(config.get(key, False)) == bool(checked):
            return
        config[key] = bool(checked)
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_axis_value(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        key: str,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        axis = axes[index]
        try:
            if key == "num_bins":
                value: Any = max(int(float(text)), 1)
                if axis.get("num_bins") == value:
                    return
                axis["num_bins"] = value
            else:
                value = float(text)
                if key == "step_size":
                    if value <= 0.0 or not np.isfinite(value):
                        return
                    num_bins = _num_bins_from_step_size(axis.get("lower", 0.0), axis.get("upper", 0.0), value)
                    if axis.get("num_bins") == num_bins:
                        return
                    axis["num_bins"] = num_bins
                else:
                    if axis.get(key) == value:
                        return
                    axis[key] = value
        except ValueError:
            return
        axis.update(_sanitize_rebin_axis_config(axis))
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_axis_vector(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        ndim = len(dataset.data.axes) if isinstance(dataset.data, MDHistoData) else len(axes)
        parsed = _parse_parameter_text(text)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != ndim:
            self._set_dataset_details(dataset, group)  # revert editor to the stored vector
            return
        try:
            vector = [_clean_axis_weight(component) for component in parsed]
        except (TypeError, ValueError):
            self._set_dataset_details(dataset, group)
            return
        axis = axes[index]
        if axis.get("vector") == vector:
            return
        axis["vector"] = vector
        self._after_dataset_rebin_changed(dataset, group)

    def _after_dataset_rebin_changed(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._set_dataset_details(dataset, group)

    def _clear_details_panel(self) -> None:
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _restore_selected_fit_state(self, group: DataGroup, fit_entry: FitTimelineEntry) -> None:
        if not fit_entry.snapshot:
            fit_entry.snapshot = snapshot_data_group_state(group)
        self._restoring_fit_selection = True
        try:
            restore_data_group_state(group, fit_entry.snapshot)
            self._refresh_tree(select_group=group, select_fit=fit_entry)
        finally:
            self._restoring_fit_selection = False
        self._active_fit_group = group
        self._active_fit_anchor = fit_entry if fit_entry.kind in {"initial", "result"} else None
        self._active_fit_current = fit_entry if fit_entry.kind == "current" else None
        self._active_branch_current = None
        self._mark_dirty()
        self.refresh_slice_viewer(group)

    def _record_data_group_state_change(self, group: DataGroup) -> bool:
        """Update fit-history current state, creating an edit branch when needed."""

        if self._restoring_fit_selection:
            return False
        ensure_fit_history(group)
        branch_created = False
        current_entry: FitTimelineEntry | None = None
        if self._active_fit_group is group:
            if _fit_entry_in_tree(group.fits, self._active_fit_current):
                current_entry = self._active_fit_current
            elif _fit_entry_in_tree(group.fits, self._active_fit_anchor):
                if not _fit_entry_in_tree(group.fits, self._active_branch_current):
                    timeline = FitTimelineEntry(
                        name=next_fit_timeline_name(group.fits),
                        kind="timeline",
                        created_at=_timestamp_now(),
                        metadata={
                            "branch_reason": "dataset/model state edited after restoring a fit",
                            "branched_from": self._active_fit_anchor.name,
                        },
                    )
                    current_entry = current_state_fit_entry(group)
                    timeline.children.append(current_entry)
                    self._active_fit_anchor.children.append(timeline)
                    self._active_branch_current = current_entry
                    branch_created = True
                else:
                    current_entry = self._active_branch_current
            else:
                self._active_fit_group = None
                self._active_fit_anchor = None
                self._active_fit_current = None
                self._active_branch_current = None
        if current_entry is None:
            current_entry = _top_level_current_state_entry(group)
        if current_entry is not None:
            _set_fit_current_snapshot(current_entry, group)
        return branch_created

    def _clear_active_fit_state(self) -> None:
        self._active_fit_group = None
        self._active_fit_anchor = None
        self._active_fit_current = None
        self._active_branch_current = None

    def _sync_fit_editor(self, fit_entry: FitTimelineEntry) -> None:
        self.fit_optimizer_combo.blockSignals(True)
        self.fit_optimizer_combo.setCurrentText(fit_entry.optimizer or "least_squares")
        self.fit_optimizer_combo.blockSignals(False)
        self.fit_optimizer_config_editor.blockSignals(True)
        self.fit_optimizer_config_editor.setText(json.dumps(fit_entry.optimizer_config, sort_keys=True))
        self.fit_optimizer_config_editor.blockSignals(False)

    def _set_selected_fit_optimizer(self, optimizer: str) -> None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        if fit_entry.optimizer != str(optimizer):
            fit_entry.optimizer = str(optimizer)
            self._mark_dirty()

    def _set_selected_fit_optimizer_config(self) -> None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        try:
            config = json.loads(self.fit_optimizer_config_editor.text() or "{}")
        except json.JSONDecodeError:
            config = {}
            self.fit_optimizer_config_editor.setText("{}")
        if not isinstance(config, dict):
            config = {}
            self.fit_optimizer_config_editor.setText("{}")
        if fit_entry.optimizer_config != config:
            fit_entry.optimizer_config = config
            self._mark_dirty()

    def context_menu_action_names(self, item: Any | None) -> list[str]:
        """Return context-menu action names for a tree item."""

        return [name for name, _enabled in self._context_menu_action_specs(item)]

    def _context_menu_action_specs(self, item: Any | None) -> list[tuple[str, bool]]:
        _group, entry, mask, model, role = self._objects_for_item(item)
        can_paste = self._can_paste_into_role(role, entry)
        has_source = bool(entry is not None and _dataset_source_path(entry) is not None)
        enabled_state = _enabled_state_for_role(role, entry, mask, model)
        specs: list[tuple[str, bool]] = []
        if role in {"dataset", "mask"}:
            specs.append(("Copy", True))
        if role in {"group", "datasets", "dataset", "masks"}:
            specs.append(("Paste", can_paste))
        if enabled_state is not None:
            specs.append(("Disable" if enabled_state else "Enable", True))
        if role in {"group", "dataset", "mask", "model", "fit", "fit_timeline", "dataset_group", "group_mask"}:
            specs.append(("Rename", True))
            specs.append(("Delete", True))
        if role in {"group", "datasets", "dataset", "masks", "mask", "dataset_group", "group_masks", "group_mask"}:
            specs.append(("View in data viewer", True))
        if role == "dataset":
            specs.append(("Show file location", has_source))
            specs.append(("Change file source", True))
        if role in {"dataset", "masks", "group_masks", "dataset_group"}:
            specs.append(("Add mask", True))
        if role in {"group", "datasets", "dataset_group"}:
            specs.append(("New dataset group", True))
        if role in {"group", "models"}:
            specs.append(("Add model", True))
        if role == "fit":
            specs.append(("Fit now", True))
        return specs

    def _show_context_menu(self, item: Any | None, global_pos: Any) -> None:
        from PySide6 import QtWidgets

        if item is None:
            return
        self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self.tree)
        actions = {
            "Copy": self.copy_selected,
            "Paste": self.paste_into_selection,
            "Rename": self.rename_selected,
            "Delete": self.delete_selected,
            "View in data viewer": self.open_slice_viewer_for_selection,
            "Show file location": self.show_file_location_for_selection,
            "Change file source": self.change_file_source_for_selection,
            "Add mask": self.add_mask_to_selection,
            "New dataset group": self.add_dataset_group_to_selection,
            "Add model": self.add_model_to_selection,
            "Fit now": self.fit_now_for_selection,
            "Enable": lambda: self._set_selected_enabled(True),
            "Disable": lambda: self._set_selected_enabled(False),
        }
        for name, enabled in self._context_menu_action_specs(item):
            action = menu.addAction(name)
            action.setEnabled(enabled)
            action.triggered.connect(actions[name])
        if not menu.isEmpty():
            menu.exec(global_pos)

    def _can_paste_into_role(self, role: str, entry: DatasetEntry | None) -> bool:
        if self._clipboard is None:
            return False
        clip_role, payload = self._clipboard
        if clip_role == "dataset" and isinstance(payload, DatasetEntry):
            return role in {"group", "datasets"}
        if clip_role == "mask" and isinstance(payload, MaskSpec):
            return role in {"dataset", "masks"} and entry is not None
        return False

    def _sync_window_title(self) -> None:
        suffix = "Untitled" if self.project_path is None else str(self.project_path)
        marker = " *" if self.has_unsaved_changes else ""
        self.window.setWindowTitle(f"metallix Project Explorer - {suffix}{marker}")

    def _mark_dirty(self) -> None:
        self.has_unsaved_changes = True
        self._sync_window_title()

    def _confirm_save_before_closing_project(self) -> bool:
        from PySide6 import QtWidgets

        if not self.has_unsaved_changes:
            return True
        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Question)
        message.setWindowTitle("Unsaved metallix project")
        message.setText("Save changes before closing this project?")
        save_button = message.addButton(
            QtWidgets.QMessageBox.StandardButton.Save
        )
        cancel_button = message.addButton(
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        close_without_saving_button = message.addButton(
            "Close without saving",
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
        )
        message.setDefaultButton(save_button)
        message.exec()
        clicked_button = message.clickedButton()
        if clicked_button is cancel_button:
            return False
        if clicked_button is save_button:
            return self.save()
        return clicked_button is close_without_saving_button

    def _handle_window_close(self, event: Any) -> None:
        if self._allow_window_close or self._confirm_save_before_closing_project():
            self._close_all_slice_viewers()
            event.accept()
            return
        event.ignore()

    def _remember_recent_project(self, path: str | Path) -> None:
        remember_recent_project(path)
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        if self.recent_projects_menu is None:
            return
        self.recent_projects_menu.clear()
        recent = recent_project_paths()
        if not recent:
            empty_action = self.recent_projects_menu.addAction("No recent projects")
            empty_action.setEnabled(False)
            return
        for path in recent:
            action = self.recent_projects_menu.addAction(str(path))
            action.setEnabled(path.exists())
            action.triggered.connect(lambda _checked=False, path=path: self.open_project_path(path))
        if any(not path.exists() for path in recent):
            self.recent_projects_menu.addSeparator()
            self.recent_projects_menu.addAction("Remove missing projects", self._remove_missing_recent_projects)

    def _remove_missing_recent_projects(self) -> None:
        forget_missing_recent_projects()
        self._refresh_recent_projects_menu()

    def _refresh_mask_type_combo(self) -> None:
        current = self.mask_type_combo.currentData()
        self.mask_type_combo.blockSignals(True)
        self.mask_type_combo.clear()
        for type_name in available_mask_types():
            self.mask_type_combo.addItem(MASK_TYPE_DEFINITIONS[type_name]["label"], type_name)
        if current is not None:
            index = self.mask_type_combo.findData(current)
            if index >= 0:
                self.mask_type_combo.setCurrentIndex(index)
        self.mask_type_combo.blockSignals(False)

    def _sync_mask_editor(self, mask: MaskSpec, dataset: DatasetEntry | None = None) -> None:
        ensure_coordinate_range_mask_axes(mask, dataset)
        self._refresh_mask_type_combo()
        index = self.mask_type_combo.findData(mask.type)
        self.mask_type_combo.blockSignals(True)
        self.mask_type_combo.setCurrentIndex(max(index, 0))
        self.mask_type_combo.blockSignals(False)
        self._rebuild_mask_parameter_editor(mask, dataset)

    def _set_selected_mask_type(self, _label: str) -> None:
        group, entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        type_name = self.mask_type_combo.currentData()
        if not type_name or type_name == mask.type:
            return
        mask.type = str(type_name)
        defaults = default_mask_parameters(mask.type)
        mask.parameters = {name: mask.parameters.get(name, value) for name, value in defaults.items()}
        ensure_coordinate_range_mask_axes(mask, entry)
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_mask_parameter_editor(mask, entry)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_mask=mask)
            else:
                self.refresh_slice_viewer(group)

    def _rebuild_mask_parameter_editor(self, mask: MaskSpec, dataset: DatasetEntry | None = None) -> None:
        from PySide6 import QtWidgets

        ensure_coordinate_range_mask_axes(mask, dataset)
        self._clear_mask_parameter_editor()
        invert_check = QtWidgets.QCheckBox("Invert")
        invert_check.setChecked(bool(mask.invert))
        invert_check.setToolTip("Flip this mask contribution before applying it: masked bins become unmasked and unmasked bins become masked.")
        invert_check.toggled.connect(lambda checked: self._set_mask_invert(checked))
        additive_check = QtWidgets.QCheckBox("Additive")
        additive_check.setChecked(bool(mask.additive))
        additive_check.setToolTip("Add bins back into the accumulated metallix mask. This never overrides the file mask.")
        additive_check.toggled.connect(lambda checked: self._set_mask_additive(checked))
        self.mask_parameter_layout.addWidget(invert_check, 0, 0)
        self.mask_parameter_layout.addWidget(additive_check, 0, 1)
        parameter_names = _mask_parameter_names(mask, dataset)
        layout_row = 1
        added_axis_section = False
        for parameter_name in parameter_names:
            if (
                mask.type == "coordinate_range"
                and parameter_name.startswith(COORDINATE_RANGE_AXIS_PREFIX)
                and not added_axis_section
            ):
                section_label = QtWidgets.QLabel("Coordinate axes")
                section_label.setStyleSheet("font-weight: 600")
                self.mask_parameter_layout.addWidget(section_label, layout_row, 0, 1, 2)
                layout_row += 1
                added_axis_section = True
            label = QtWidgets.QLabel(parameter_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(mask.parameters.get(parameter_name, "")))
            tooltip = mask_parameter_tooltip(mask.type, parameter_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=editor: self._set_mask_parameter(parameter_name, editor.text())
            )
            self.mask_parameter_layout.addWidget(label, layout_row, 0)
            self.mask_parameter_layout.addWidget(editor, layout_row, 1)
            layout_row += 1

    def _clear_mask_parameter_editor(self) -> None:
        while self.mask_parameter_layout.count():
            item = self.mask_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_mask_parameter(self, name: str, text: str) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        value = _parse_parameter_text(text)
        if mask.parameters.get(name) != value:
            mask.parameters[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_mask=mask)
                return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _set_mask_invert(self, checked: bool) -> None:
        self._set_mask_option("invert", bool(checked))

    def _set_mask_additive(self, checked: bool) -> None:
        self._set_mask_option("additive", bool(checked))

    def _set_mask_option(self, name: str, value: bool) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        if bool(getattr(mask, name)) == bool(value):
            return
        setattr(mask, name, bool(value))
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_mask=mask)
            return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _refresh_model_type_combo(self) -> None:
        current = self.model_type_combo.currentData()
        self.model_type_combo.blockSignals(True)
        self.model_type_combo.clear()
        for type_name in available_model_types():
            self.model_type_combo.addItem(MODEL_TYPE_DEFINITIONS[type_name]["label"], type_name)
        if current is not None:
            index = self.model_type_combo.findData(current)
            if index >= 0:
                self.model_type_combo.setCurrentIndex(index)
        self.model_type_combo.blockSignals(False)

    def _sync_model_editor(self, model: ModelComponentSpec) -> None:
        self._refresh_model_type_combo()
        index = self.model_type_combo.findData(model.type)
        self.model_type_combo.blockSignals(True)
        self.model_type_combo.setCurrentIndex(max(index, 0))
        self.model_type_combo.blockSignals(False)
        self._rebuild_model_parameter_editor(model)

    def _set_selected_model_type(self, _label: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        type_name = self.model_type_combo.currentData()
        if not type_name or type_name == model.type:
            return
        model.type = str(type_name)
        defaults = default_model_parameters(model.type)
        default_config = default_model_config(model.type)
        default_fit = default_model_fit_parameters(model.type)
        default_global = default_model_global_fit(model.type)
        model.parameters = {name: model.parameters.get(name, value) for name, value in defaults.items()}
        model.config = {name: model.config.get(name, value) for name, value in default_config.items()}
        model.fit_parameters = {name: model.fit_parameters.get(name, value) for name, value in default_fit.items()}
        model.global_fit = {name: model.global_fit.get(name, value) for name, value in default_global.items()}
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_model_parameter_editor(model)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            else:
                self.refresh_slice_viewer(group)

    def _rebuild_model_parameter_editor(self, model: ModelComponentSpec) -> None:
        from PySide6 import QtWidgets

        self._clear_model_parameter_editor()
        fit_group = QtWidgets.QGroupBox("Fit Parameters")
        fit_group.setObjectName("model_fit_parameters_group")
        fit_layout = QtWidgets.QGridLayout(fit_group)
        fit_layout.setColumnStretch(1, 1)
        for row, parameter_name in enumerate(MODEL_TYPE_DEFINITIONS[model.type]["parameters"]):
            label = QtWidgets.QLabel(parameter_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(model.parameters.get(parameter_name, "")))
            fit_check = QtWidgets.QCheckBox("Fit")
            fit_check.setChecked(bool(model.fit_parameters.get(parameter_name, False)))
            global_check = QtWidgets.QCheckBox("Global fit")
            global_check.setChecked(bool(model.global_fit.get(parameter_name, True)))
            tooltip = model_parameter_tooltip(model.type, parameter_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            fit_check.setToolTip(tooltip)
            global_check.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=editor: self._set_model_parameter(parameter_name, editor.text())
            )
            fit_check.toggled.connect(
                lambda checked, parameter_name=parameter_name: self._set_model_fit_parameter(parameter_name, checked)
            )
            global_check.toggled.connect(
                lambda checked, parameter_name=parameter_name: self._set_model_global_fit(parameter_name, checked)
            )
            fit_layout.addWidget(label, row, 0)
            fit_layout.addWidget(editor, row, 1)
            fit_layout.addWidget(fit_check, row, 2)
            fit_layout.addWidget(global_check, row, 3)
        self.model_parameter_layout.addWidget(fit_group, 0, 0, 1, 4)

        config_group = QtWidgets.QGroupBox("Configuration Settings")
        config_group.setObjectName("model_config_group")
        config_layout = QtWidgets.QGridLayout(config_group)
        config_layout.setColumnStretch(1, 1)
        config_definitions = MODEL_TYPE_DEFINITIONS[model.type].get("config", {})
        if not config_definitions:
            config_layout.addWidget(QtWidgets.QLabel("No configuration settings."), 0, 0, 1, 2)
        for row, setting_name in enumerate(config_definitions):
            label = QtWidgets.QLabel(setting_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get(setting_name, "")))
            tooltip = model_config_tooltip(model.type, setting_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda setting_name=setting_name, editor=editor: self._set_model_config_setting(setting_name, editor.text())
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(editor, row, 1)
        self.model_parameter_layout.addWidget(config_group, 1, 0, 1, 4)

    def _clear_model_parameter_editor(self) -> None:
        while self.model_parameter_layout.count():
            item = self.model_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _set_model_parameter(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = _parse_parameter_text(text)
        if model.parameters.get(name) != value:
            model.parameters[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
                return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _set_model_config_setting(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = _parse_parameter_text(text)
        if model.config.get(name) != value:
            model.config[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
                return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _set_model_global_fit(self, name: str, checked: bool) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = bool(checked)
        if model.global_fit.get(name) != value:
            model.global_fit[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
                return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _set_model_fit_parameter(self, name: str, checked: bool) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = bool(checked)
        if model.fit_parameters.get(name) != value:
            model.fit_parameters[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
                return
        if group is not None:
            self.refresh_slice_viewer(group)

    def refresh_open_slice_viewers(self) -> None:
        for group in list(self.project.data_groups):
            self.refresh_slice_viewer(group)

    def _replace_slice_viewer(
        self,
        group: DataGroup,
        datasets: list[MDHistoData],
        names: list[str],
    ) -> Any:
        global QtMDHistoSliceViewer

        if QtMDHistoSliceViewer is None:
            from .qt_slice_viewer import QtMDHistoSliceViewer as viewer_class

            QtMDHistoSliceViewer = viewer_class
        existing = self._slice_viewers.get(id(group))
        if existing is not None and hasattr(existing, "replace_datasets"):
            existing.replace_datasets(datasets, dataset_names=names)
            viewer = existing
        else:
            if existing is not None and existing.window is not None:
                existing.window.close()
            viewer = QtMDHistoSliceViewer(datasets, dataset_names=names)
            self._slice_viewers[id(group)] = viewer
        if viewer.window is not None:
            viewer.window.setWindowTitle(f"metallix Data Viewer - {group.name}")
        return viewer

    def _close_slice_viewer(self, group: DataGroup) -> None:
        viewer = self._slice_viewers.pop(id(group), None)
        if viewer is not None and viewer.window is not None:
            viewer.window.close()

    def _close_all_slice_viewers(self) -> None:
        for group_id in list(self._slice_viewers):
            viewer = self._slice_viewers.pop(group_id)
            if viewer is not None and viewer.window is not None:
                viewer.window.close()


def _make_project_window_class():
    from PySide6 import QtWidgets

    class ProjectWindow(QtWidgets.QMainWindow):
        def __init__(self, explorer: MetallixProjectExplorer) -> None:
            super().__init__()
            self.explorer = explorer

        def closeEvent(self, event):
            self.explorer._handle_window_close(event)

    return ProjectWindow


def _make_project_tree_class():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ProjectTree(QtWidgets.QTreeWidget):
        def __init__(self, explorer: MetallixProjectExplorer) -> None:
            super().__init__()
            self.explorer = explorer

        def contextMenuEvent(self, event):
            self.explorer._show_context_menu(self.itemAt(event.pos()), event.globalPos())

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_Delete:
                self.explorer.delete_selected()
                return
            if event.matches(QtGui.QKeySequence.StandardKey.Copy):
                self.explorer.copy_selected()
                return
            if event.matches(QtGui.QKeySequence.StandardKey.Paste):
                self.explorer.paste_into_selection()
                return
            super().keyPressEvent(event)

        def startDrag(self, supported_actions):
            item = self.currentItem()
            if item is None:
                return
            role = self.explorer._objects_for_item(item)[4]
            if role not in {"group", "dataset", "mask", "dataset_group"}:
                return
            selected_roles = {self.explorer._objects_for_item(it)[4] for it in self.selectedItems()}
            if item not in self.selectedItems():
                selected_roles = {role}
            if role in {"group", "dataset", "mask"} and not selected_roles.issubset({role}):
                return
            if role == "dataset_group" and selected_roles - {"dataset_group"}:
                return
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setData("application/x-metallix-tree-item", role.encode("utf-8"))
            drag.setMimeData(mime_data)
            drag.exec(QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction)

        def dragEnterEvent(self, event):
            if event.mimeData().hasFormat("application/x-metallix-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragEnterEvent(event)

        def dragMoveEvent(self, event):
            if event.mimeData().hasFormat("application/x-metallix-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragMoveEvent(event)

        def dropEvent(self, event):
            # Internal item drags: always handled here, never by the default
            # QTreeWidget item-move machinery.
            if event.mimeData().hasFormat("application/x-metallix-tree-item"):
                target = self.itemAt(event.position().toPoint())
                copied = event.dropAction() == QtCore.Qt.DropAction.CopyAction
                if (
                    target is not None
                    and self.explorer.move_or_copy_selected_to_item(
                        target,
                        copy_item=copied,
                        drop_position=self.dropIndicatorPosition(),
                    )
                ):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return
            if event.mimeData().hasUrls():
                target_item = self.itemAt(event.position().toPoint())
                group, node = self.explorer._resolve_dataset_drop_target(target_item)
                paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
                if group is not None and paths:
                    into = node if node is not group else None
                    self.explorer.import_dataset_paths(group, paths, into=into)
                    event.acceptProposedAction()
                    return
            super().dropEvent(event)

    return ProjectTree


def _bold_label(QtWidgets: Any, text: str) -> Any:
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("font-weight: 600")
    return label


def _is_renameable_role(role: str) -> bool:
    return role in {"group", "dataset", "mask", "model", "fit", "fit_timeline", "dataset_group", "group_mask"}


def _dataset_source_path(dataset: DatasetEntry) -> Path | None:
    if not isinstance(dataset.metadata, dict):
        return None
    source = dataset.metadata.get("source_file")
    if not source:
        return None
    return Path(source)


def _show_file_location(path: Path) -> bool:
    target = path if path.exists() else path.parent
    if not target.exists():
        return False
    if platform.system() == "Darwin" and path.exists():
        subprocess.Popen(["open", "-R", str(path)])
        return True
    from PySide6 import QtCore, QtGui

    location = target if target.is_dir() else target.parent
    return bool(QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(location))))


def _make_refreshing_combo_class():
    from PySide6 import QtWidgets

    class RefreshingComboBox(QtWidgets.QComboBox):
        def __init__(self, refresh_callback) -> None:
            super().__init__()
            self._refresh_callback = refresh_callback

        def showPopup(self):
            self._refresh_callback()
            super().showPopup()

    return RefreshingComboBox


def _has_slice_viewer_candidates(group: DataGroup) -> bool:
    for dataset in group.datasets:
        if isinstance(dataset.data, (MDHistoData, PointListData)):
            return True
        if data_type_container(dataset.data_type) == "point_list":
            return True
        source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
        if source and Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5"}:
            return True
    return False


def _dataset_can_load(dataset: DatasetEntry) -> bool:
    if dataset.data is not None:
        return False
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if not source:
        return False
    if data_type_container(dataset.data_type) == "point_list":
        return True
    return bool(Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5"})


def _dataset_can_rebin(dataset: DatasetEntry) -> bool:
    return isinstance(dataset.data, (MDHistoData, PointData4D, PointListData))


def _dataset_can_save(dataset: DatasetEntry) -> bool:
    return isinstance(dataset.data, (MDHistoData, PointListData)) or dataset_rebin_enabled(dataset)


def _dataset_axes_and_data_lines(data: Any) -> tuple[list[str], list[str]]:
    if data is None:
        return ["Not loaded."], ["Imported data: not loaded"]
    split = _split_dataset_summary_lines(_dataset_data_summary_lines(data))
    axes = split.get("Axes", ["No axis information available."])
    data_lines = split.get("Data")
    if data_lines is None:
        data_lines = split.get("", [])
        if not data_lines:
            data_lines = ["No data summary available."]
    return axes, data_lines


def _split_dataset_summary_lines(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in lines:
        if not line:
            continue
        if line in {"Axes", "Data"}:
            current = line
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _dataset_data_summary_lines(data: Any) -> list[str]:
    if isinstance(data, MDHistoData):
        return _mdhisto_summary_lines(data)
    if isinstance(data, PointData4D):
        return _point_data_summary_lines(data)
    if isinstance(data, PointListData):
        return _point_list_summary_lines(data)
    if data is None:
        return ["", "Imported data: not loaded"]
    shape = getattr(data, "shape", None)
    lines = ["", f"Imported data type: {type(data).__name__}"]
    if shape is not None:
        lines.append(f"Shape: {_format_shape(shape)}")
    size = _object_nbytes(data)
    if size is not None:
        lines.append(f"Imported data size: {_format_bytes(size)}")
    return lines


def _mdhisto_summary_lines(data: MDHistoData) -> list[str]:
    lines = [
        "",
        "Axes",
        f"Dimensions: {len(data.axes)}",
        f"Shape: {_format_shape(data.shape)}",
    ]
    for index, axis in enumerate(data.axes):
        axis_size = data.shape[index]
        values = axis.centers
        value_range = ""
        if values.size:
            value_range = f", range {_format_number(values[0])} to {_format_number(values[-1])}"
        lines.append(
            f"{index}: {axis.name} ({axis.units or '-'}) - {axis_size} bins{value_range}"
        )
        if axis.kind:
            lines.append(f"   kind: {axis.kind}")
        if axis.frame:
            lines.append(f"   frame: {axis.frame}")
    valid = int(np.count_nonzero(~data.mask))
    masked = int(np.count_nonzero(data.mask))
    lines.extend(
        [
            "",
            "Data",
            f"Total bins: {int(np.prod(data.shape))}",
            f"Unmasked bins: {valid}",
            f"Masked bins: {masked}",
            f"Imported data size: {_format_bytes(_mdhisto_nbytes(data))}",
        ]
    )
    if data.coordinate_system is not None:
        lines.append(f"Coordinate system: {data.coordinate_system}")
    if data.visual_normalization is not None:
        lines.append(f"Visual normalization: {data.visual_normalization}")
    return lines


def _point_data_summary_lines(data: PointData4D) -> list[str]:
    lines = [
        "",
        "Axes",
        "Dimensions: 4",
    ]
    for name, units, values in (
        ("H", "rlu", data.H),
        ("K", "rlu", data.K),
        ("L", "rlu", data.L),
        ("E", "meV", data.E),
    ):
        lines.append(
            f"{name} ({units}) - {data.size} points, range {_format_number(np.nanmin(values))} to {_format_number(np.nanmax(values))}"
        )
    lines.extend(
        [
            "",
            "Data",
            f"Points: {data.size}",
            f"Valid points: {int(np.count_nonzero(data.valid_mask()))}",
            f"Imported data size: {_format_bytes(_point_data_nbytes(data))}",
        ]
    )
    if data.temperature is not None:
        lines.append(f"Temperature: {_condition_value_text(data.temperature)} K")
    return lines


def _point_list_summary_lines(data: PointListData) -> list[str]:
    lines = [
        "",
        "Axes",
        f"Points: {data.size}",
        f"Columns: {len(data.columns)}",
    ]
    for name in data.coordinate_names:
        values = data.column(name)
        unit = data.unit(name)
        value_range = ""
        if values.size:
            value_range = f", range {_format_number(np.nanmin(values))} to {_format_number(np.nanmax(values))}"
        lines.append(f"coord {name} ({unit or '-'}){value_range}")
    lines.extend(["", "Data", f"Points: {data.size}"])
    for channel in data.channels:
        label = str(channel["label"])
        unit = data.unit(str(channel["value"]))
        error = "with error" if channel.get("error") else "no error"
        lines.append(f"channel {label} ({unit or '-'}) - {error}")
    return lines


def _dataset_source_summary_lines(dataset: DatasetEntry) -> list[str]:
    lines = _dataset_source_lines(dataset)
    if not lines:
        return []
    return ["", "Source", *lines]


def _dataset_source_lines(dataset: DatasetEntry) -> list[str]:
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if not source:
        return []
    lines = [f"File: {source}"]
    path = Path(source)
    if path.exists():
        lines.append(f"File size on disk: {_format_bytes(path.stat().st_size)}")
    else:
        lines.append("File size on disk: unavailable")
    return lines


def _dataset_crystal_lines(dataset: DatasetEntry, group: DataGroup | None = None) -> list[str]:
    metadata = _merged_dataset_metadata(dataset)
    lines: list[str] = []
    lattice = None
    if group is not None and group.lattice_parameters:
        lattice = group.lattice_parameters
    if isinstance(metadata.get("lattice_parameters"), dict):
        lattice = metadata["lattice_parameters"]
    if isinstance(lattice, dict) and lattice:
        lines.append("Lattice parameters")
        lines.extend(_mapping_lines(lattice))
    matrix_name, matrix = _dataset_orientation_matrix(metadata)
    if matrix is not None:
        if lines:
            lines.append("")
        lines.append(matrix_name)
        lines.extend(_matrix_lines(matrix))
    if not lines:
        lines.append("No crystal metadata available.")
    return lines


def _merged_dataset_metadata(dataset: DatasetEntry) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(getattr(dataset.data, "metadata", None), dict):
        merged.update(dataset.data.metadata)
    if isinstance(dataset.metadata, dict):
        merged.update(dataset.metadata)
    return merged


def _dataset_orientation_matrix(metadata: dict[str, Any]) -> tuple[str, Any | None]:
    for key, label in (
        ("rlu_to_inv_angstrom_matrix", "RLU to inverse angstrom matrix"),
        ("ub_matrix", "UB matrix"),
        ("orientation_matrix", "Orientation matrix"),
    ):
        if key in metadata:
            return label, metadata[key]
    oriented_lattice = metadata.get("oriented_lattice")
    if isinstance(oriented_lattice, dict):
        for key, label in (
            ("rlu_to_inv_angstrom_matrix", "Oriented lattice RLU to inverse angstrom matrix"),
            ("ub_matrix", "Oriented lattice UB matrix"),
            ("orientation_matrix", "Oriented lattice orientation matrix"),
        ):
            if key in oriented_lattice:
                return label, oriented_lattice[key]
    return "", None


def _matrix_lines(matrix: Any) -> list[str]:
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return [_metadata_value_text(matrix)]
    if array.ndim != 2:
        return [_metadata_value_text(matrix)]
    return ["[" + ", ".join(_format_number(value) for value in row) + "]" for row in array]


def _dataset_metadata_lines(dataset: DatasetEntry) -> list[str]:
    return _mapping_lines(_dataset_metadata_mapping(dataset))


def _dataset_metadata_mapping(dataset: DatasetEntry) -> dict[str, Any]:
    merged = _merged_dataset_metadata(dataset)
    merged.pop("source_file", None)
    merged.pop("import_status", None)
    merged.pop("lattice_parameters", None)
    merged.pop("rlu_to_inv_angstrom_matrix", None)
    merged.pop("ub_matrix", None)
    merged.pop("orientation_matrix", None)
    merged.pop("oriented_lattice", None)
    return merged


def _mapping_lines(mapping: dict[str, Any]) -> list[str]:
    return [f"{key}: {_metadata_value_text(value)}" for key, value in sorted(mapping.items())]


def _metadata_value_text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _add_metadata_tree_item(parent: Any, key: str, value: Any, *, depth: int = 0) -> Any:
    from PySide6 import QtWidgets

    item = QtWidgets.QTreeWidgetItem([str(key), _metadata_tree_value_summary(value)])
    item.setToolTip(0, str(key))
    item.setToolTip(1, _metadata_value_text(value))
    parent.addChild(item) if hasattr(parent, "addChild") else parent.addTopLevelItem(item)
    if depth >= 6:
        return item
    if isinstance(value, dict):
        for child_key in sorted(value):
            _add_metadata_tree_item(item, str(child_key), value[child_key], depth=depth + 1)
    elif _metadata_is_expandable_sequence(value):
        for index, child_value in enumerate(list(value)[:64]):
            _add_metadata_tree_item(item, f"[{index}]", child_value, depth=depth + 1)
        if len(value) > 64:
            QtWidgets.QTreeWidgetItem(item, ["...", f"{len(value) - 64} more item(s)"])
    return item


def _metadata_tree_value_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"{len(value)} field(s)"
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if _metadata_is_expandable_sequence(value):
        return f"{len(value)} item(s)"
    return _metadata_value_text(value)


def _metadata_is_expandable_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, np.ndarray)):
        return False
    if not isinstance(value, (list, tuple)):
        return False
    return len(value) > 8 or any(isinstance(item, (dict, list, tuple, np.ndarray)) for item in value)


def _array_summary(value: np.ndarray) -> str:
    array = np.asarray(value)
    shape = "x".join(str(size) for size in array.shape) or "scalar"
    summary = f"array {shape}, {array.dtype}"
    if array.size == 0:
        return f"{summary}, empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            return f"{summary}, min {_format_number(np.nanmin(finite))}, max {_format_number(np.nanmax(finite))}"
    if array.size <= 8:
        return f"{summary}, {_metadata_value_text(array.tolist())}"
    return summary


def _condition_value_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "empty"
        finite = value[np.isfinite(value)]
        if finite.size == 0:
            return "nonfinite"
        if np.allclose(finite, finite[0]):
            return _format_number(finite[0])
        return f"{_format_number(np.nanmin(finite))} to {_format_number(np.nanmax(finite))}"
    return _format_number(value) if isinstance(value, (int, float, np.number)) else str(value)


def _format_shape(shape: Any) -> str:
    return " x ".join(str(int(value)) for value in tuple(shape))


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _object_nbytes(data: Any) -> int | None:
    nbytes = getattr(data, "nbytes", None)
    if nbytes is None:
        return None
    return int(nbytes)


def _mdhisto_nbytes(data: MDHistoData) -> int:
    arrays = [data.signal, data.errors, data.mask, data.num_events]
    arrays.extend(axis.values for axis in data.axes)
    return int(sum(array.nbytes for array in arrays))


def _point_data_nbytes(data: PointData4D) -> int:
    arrays = [data.H, data.K, data.L, data.E, data.intensity, data.sigma, data.mask]
    if isinstance(data.temperature, np.ndarray):
        arrays.append(data.temperature)
    return int(sum(array.nbytes for array in arrays))


def _project_to_dict(project: MetallixProject) -> dict[str, Any]:
    return {
        "format": "metallix-project",
        "version": 1,
        "settings": _json_mapping(project.settings),
        "data_groups": [_data_group_to_dict(group) for group in project.data_groups],
    }


def _project_from_dict(payload: dict[str, Any]) -> MetallixProject:
    if payload.get("format") != "metallix-project":
        raise ValueError("not a metallix project file")
    project = MetallixProject(settings=dict(payload.get("settings", {})))
    for group_payload in payload.get("data_groups", []):
        group = DataGroup(
            name=str(group_payload["name"]),
            datasets=[_dataset_from_dict(d) for d in group_payload.get("datasets", [])],
            subgroups=[_dataset_group_from_dict(s) for s in group_payload.get("subgroups", [])],
            masks=[_mask_from_dict(m) for m in group_payload.get("masks", [])],
            lattice_parameters=group_payload.get("lattice_parameters"),
            spacegroup=group_payload.get("spacegroup"),
            metadata=dict(group_payload.get("metadata", {})),
        )
        for model_payload in group_payload.get("models", []):
            model_type = str(model_payload.get("type", "constant_background"))
            fit_payload = dict(model_payload.get("fit_parameters", {}))
            model = ModelComponentSpec(
                name=str(model_payload["name"]),
                type=model_type,
                parameters=dict(model_payload.get("parameters", {})),
                config=dict(model_payload.get("config", default_model_config(model_type))),
                fit_parameters={
                    name: bool(fit_payload.get(name, value))
                    for name, value in default_model_fit_parameters(model_type).items()
                },
                global_fit={
                    str(name): bool(value)
                    for name, value in dict(model_payload.get("global_fit", {})).items()
                },
                enabled=bool(model_payload.get("enabled", True)),
                metadata=dict(model_payload.get("metadata", {})),
            )
            group.models[model.name] = model
        group.fits = [
            _fit_entry_from_dict(fit_payload)
            for fit_payload in group_payload.get("fits", [])
        ]
        ensure_fit_history(group)
        project.data_groups.append(group)
    return project


def _mask_from_dict(mask_payload: dict[str, Any]) -> MaskSpec:
    return MaskSpec(
        name=str(mask_payload["name"]),
        type=str(mask_payload.get("type", "coordinate_range")),
        parameters=dict(mask_payload.get("parameters", {})),
        enabled=bool(mask_payload.get("enabled", True)),
        invert=bool(mask_payload.get("invert", False)),
        additive=bool(mask_payload.get("additive", False)),
        metadata=dict(mask_payload.get("metadata", {})),
    )


def _dataset_from_dict(dataset_payload: dict[str, Any]) -> DatasetEntry:
    return DatasetEntry(
        name=str(dataset_payload["name"]),
        data=None,
        kind=str(dataset_payload.get("kind", "")),
        data_type=str(dataset_payload.get("data_type", "")),
        metadata=dict(dataset_payload.get("metadata", {})),
        parameters=dict(dataset_payload.get("parameters", {})),
        enabled=bool(dataset_payload.get("enabled", True)),
        fit_weight=float(dataset_payload.get("fit_weight", 1.0)),
        scale_factor=float(dataset_payload.get("scale_factor", 1.0)),
        masks=[_mask_from_dict(mask_payload) for mask_payload in dataset_payload.get("masks", [])],
    )


def _dataset_group_from_dict(payload: dict[str, Any]) -> DatasetGroup:
    return DatasetGroup(
        name=str(payload["name"]),
        datasets=[_dataset_from_dict(d) for d in payload.get("datasets", [])],
        subgroups=[_dataset_group_from_dict(s) for s in payload.get("subgroups", [])],
        masks=[_mask_from_dict(m) for m in payload.get("masks", [])],
        resolution=dict(payload.get("resolution", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _dataset_group_to_dict(group: DatasetGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "datasets": [_dataset_to_dict(dataset) for dataset in group.datasets],
        "subgroups": [_dataset_group_to_dict(sub) for sub in group.subgroups],
        "masks": [_mask_to_dict(mask) for mask in group.masks],
        "resolution": _json_mapping(group.resolution),
        "metadata": _json_mapping(group.metadata),
    }


def _data_group_to_dict(group: DataGroup) -> dict[str, Any]:
    ensure_fit_history(group)
    refresh_current_state_fit_entries(group)
    model_payloads = []
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            raise TypeError("project JSON save does not yet support fit model sessions")
        model_payloads.append(_model_to_dict(model))
    return {
        "name": group.name,
        "lattice_parameters": _json_mapping(group.lattice_parameters),
        "spacegroup": group.spacegroup,
        "metadata": _json_mapping(group.metadata),
        "datasets": [_dataset_to_dict(dataset) for dataset in group.datasets],
        "subgroups": [_dataset_group_to_dict(sub) for sub in group.subgroups],
        "masks": [_mask_to_dict(mask) for mask in group.masks],
        "models": model_payloads,
        "fits": [_fit_entry_to_dict(fit_entry) for fit_entry in group.fits],
    }


def _dataset_to_dict(dataset: DatasetEntry) -> dict[str, Any]:
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if dataset.data is not None and not source:
        raise TypeError("project JSON save does not yet support embedded dataset objects")
    if dataset.transforms:
        raise TypeError("project JSON save does not yet support dataset transforms")
    return {
        "name": dataset.name,
        "kind": dataset.kind,
        "data_type": dataset.data_type,
        "metadata": _json_mapping(dataset.metadata),
        "parameters": _json_mapping(dataset.parameters),
        "enabled": bool(dataset.enabled),
        "fit_weight": float(dataset.fit_weight),
        "scale_factor": float(dataset.scale_factor),
        "masks": [_mask_to_dict(mask) for mask in dataset.masks],
    }


def _mask_to_dict(mask: MaskSpec) -> dict[str, Any]:
    return {
        "name": mask.name,
        "type": mask.type,
        "parameters": _json_mapping(mask.parameters),
        "enabled": bool(mask.enabled),
        "invert": bool(mask.invert),
        "additive": bool(mask.additive),
        "metadata": _json_mapping(mask.metadata),
    }


def _model_to_dict(model: ModelComponentSpec) -> dict[str, Any]:
    return {
        "name": model.name,
        "type": model.type,
        "parameters": _json_mapping(model.parameters),
        "config": _json_mapping(model.config),
        "fit_parameters": {name: bool(value) for name, value in model.fit_parameters.items()},
        "global_fit": {name: bool(value) for name, value in model.global_fit.items()},
        "enabled": bool(model.enabled),
        "metadata": _json_mapping(model.metadata),
    }


def _fit_entry_to_dict(fit_entry: FitTimelineEntry) -> dict[str, Any]:
    return {
        "name": fit_entry.name,
        "kind": fit_entry.kind,
        "snapshot": copy.deepcopy(fit_entry.snapshot),
        "created_at": fit_entry.created_at,
        "duration_seconds": fit_entry.duration_seconds,
        "optimizer": fit_entry.optimizer,
        "optimizer_config": _json_mapping(fit_entry.optimizer_config),
        "goodness": _json_mapping(fit_entry.goodness),
        "children": [_fit_entry_to_dict(child) for child in fit_entry.children],
        "metadata": _json_mapping(fit_entry.metadata),
    }


def _fit_entry_from_dict(payload: dict[str, Any]) -> FitTimelineEntry:
    return FitTimelineEntry(
        name=str(payload.get("name", "Fit")),
        kind=str(payload.get("kind", "result")),
        snapshot=dict(payload.get("snapshot", {})),
        created_at=payload.get("created_at"),
        duration_seconds=payload.get("duration_seconds"),
        optimizer=str(payload.get("optimizer", "least_squares")),
        optimizer_config=dict(payload.get("optimizer_config", {})),
        goodness=dict(payload.get("goodness", {})),
        children=[
            _fit_entry_from_dict(child)
            for child in payload.get("children", [])
        ],
        metadata=dict(payload.get("metadata", {})),
    )


def _model_key(group: DataGroup, model: ModelComponentSpec) -> str | None:
    for name, existing in group.models.items():
        if existing is model:
            return name
    return None


def _json_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    json.dumps(value)
    return dict(value)


def _qt_app():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _settings():
    from PySide6 import QtCore

    return QtCore.QSettings("metallix", "metallix")


def _unique_dataset_name(base: str, existing: list[str]) -> str:
    return _unique_name(base or "Dataset", existing)


def _unique_name(base: str, existing: list[str]) -> str:
    candidate = base or "Dataset"
    if candidate not in existing:
        return candidate
    index = 1
    while f"{candidate}{index}" in existing:
        index += 1
    return f"{candidate}{index}"


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


def main() -> int:
    """Launch the metallix project explorer."""

    return MetallixProjectExplorer().run()
