from __future__ import annotations

import base64
import json
import copy
import ast
import platform
import re
import signal
import subprocess
import time
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .fit_config import (
    MODEL_TYPE_REGISTRY,
    CompiledFitProblem,
    FitDatasetInput,
    compile_fit_problem,
    component_parameter_names,
    model_supports_data_type,
    qualified_parameter_name,
)
from .fitting import (
    OptimizationConfig,
    SamplerConfig,
    SamplingResult,
    _evaluate_problem,
    evaluate_problem_model,
    fit_problem_least_squares,
    rebin_point_data,
    reciprocal_basis_from_lattice_parameters,
    sample_problem_parameters,
)
from .form_factors import available_ions
from .importers import IMPORTERS, import_with, importers_for_data_type
from .mdhisto import MDHistoAxis, MDHistoData, load_mantid_mdhisto_nxs
from .pipeline import DataGroup, DatasetEntry, DatasetGroup, FitTimelineEntry, MaskSpec, ModelComponentSpec
from .rebin import rebin_nd

QtMDHistoSliceViewer = None
RECENT_PROJECT_LIMIT = 10
RECENT_PROJECTS_KEY = "recent_projects"
DATASET_REBIN_KEY = "rebin"
GROUP_COMPOSITE_KEY = "composite"
GROUP_COMPOSITE_NAME = "Composite"
DEFAULT_REBIN_MAX_BATCH_MB = 192
DATASET_POINT_LIST_KEY = "point_list"
SUSCEPTIBILITY_CHANNEL_LABEL = "Susceptibility"
Q_COORDINATE_NAME = "q"
D_SPACING_COORDINATE_NAME = "d"
COORDINATE_RANGE_AXIS_PREFIX = "axis_"
COORDINATE_RANGE_PARAMETER_NAMES = ("H", "K", "L", "E")
CUSTOM_FORM_FACTOR_CHOICE = "__custom__"


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
    "local_relaxational": {
        "label": "Local relaxational spin",
        "description": (
            "Fully local spin relaxing at rate Gamma: chi'' = chi_loc * Gamma * E / (E^2 + Gamma^2), "
            "converted to intensity with the Bose factor, magnetic form factor, and isotropic "
            "polarization factor 2/3. Requires a dataset temperature."
        ),
        "parameters": {
            "scale": {
                "default": 1.0,
                "description": (
                    "Overall intensity scale for unnormalized data. Degenerate with chi_loc; "
                    "fix one of the two."
                ),
                "allowed": "Positive finite number.",
                "type": "float",
                "example": "1.0",
                "global_fit": True,
            },
            "chi_loc": {
                "default": 1.0,
                "description": "Static local susceptibility (1/meV up to the intensity normalization).",
                "allowed": "Positive finite number.",
                "type": "float",
                "example": "2.0",
                "global_fit": True,
            },
            "gamma": {
                "default": 2.0,
                "description": "Relaxation rate Gamma: chi'' peaks at E = Gamma.",
                "allowed": "Positive finite number in meV.",
                "type": "float",
                "example": "3.0",
                "global_fit": True,
            },
        },
        "config": {
            "ion": {
                "default": "",
                "description": (
                    "Magnetic form factor multiplying the intensity: choose a tabulated ion, "
                    "Custom for explicit <j0> coefficients, or none for no form factor. "
                    "Requires lattice metadata for |Q|."
                ),
                "allowed": "An ion label from the ILL <j0> tables, Custom, or empty.",
                "type": "str",
                "example": "Fe2",
                "choices": "form_factor_ions",
            },
            "form_factor_coefficients": {
                "default": "",
                "description": (
                    "Custom <j0> coefficients A, a, B, b, C, c, D overriding the ion table "
                    "(see https://www.ill.eu/sites/ccsl/ffacts/)."
                ),
                "allowed": "Seven comma-separated numbers, or empty to use the ion table.",
                "type": "str",
                "example": "0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119",
            },
        },
    },
    "mmp_relaxational": {
        "label": "MMP relaxational (nearly AFM)",
        "description": (
            "Millis-Monien-Pines susceptibility of a nearly antiferromagnetic metal: "
            "chi(q,w) = chi_pk / (1 + xi^2 |q-Q0|^2 - i w/omega_sf), converted to intensity "
            "with Bose, form-factor, and isotropic polarization factors. "
            "Requires dataset temperature and lattice metadata."
        ),
        "parameters": {
            "scale": {
                "default": 1.0,
                "description": (
                    "Overall intensity scale for unnormalized data. Degenerate with chi_pk; "
                    "fix one of the two."
                ),
                "allowed": "Positive finite number.",
                "type": "float",
                "example": "1.0",
                "global_fit": True,
            },
            "chi_pk": {
                "default": 1.0,
                "description": "Static susceptibility at the ordering vector Q0 (1/meV up to normalization).",
                "allowed": "Positive finite number.",
                "type": "float",
                "example": "3.0",
                "global_fit": True,
            },
            "xi": {
                "default": 1.0,
                "description": "Magnetic correlation length.",
                "allowed": "Positive finite number in Angstrom.",
                "type": "float",
                "example": "2.5",
                "global_fit": True,
            },
            "omega_sf": {
                "default": 1.0,
                "description": "Spin-fluctuation energy: relaxation rate of the mode at Q0.",
                "allowed": "Positive finite number in meV.",
                "type": "float",
                "example": "1.8",
                "global_fit": True,
            },
            "q0_h": {
                "default": 0.5,
                "description": "H coordinate of the ordering vector Q0.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.5",
                "global_fit": True,
            },
            "q0_k": {
                "default": 0.0,
                "description": "K coordinate of the ordering vector Q0.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.5",
                "global_fit": True,
            },
            "q0_l": {
                "default": 0.0,
                "description": "L coordinate of the ordering vector Q0.",
                "allowed": "Finite number in reciprocal lattice units.",
                "type": "float",
                "example": "0.0",
                "global_fit": True,
            },
        },
        "config": {
            "ion": {
                "default": "",
                "description": (
                    "Magnetic form factor multiplying the intensity: choose a tabulated ion, "
                    "Custom for explicit <j0> coefficients, or none for no form factor."
                ),
                "allowed": "An ion label from the ILL <j0> tables, Custom, or empty.",
                "type": "str",
                "example": "Fe2",
                "choices": "form_factor_ions",
            },
            "form_factor_coefficients": {
                "default": "",
                "description": (
                    "Custom <j0> coefficients A, a, B, b, C, c, D overriding the ion table "
                    "(see https://www.ill.eu/sites/ccsl/ffacts/)."
                ),
                "allowed": "Seven comma-separated numbers, or empty to use the ion table.",
                "type": "str",
                "example": "0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119",
            },
        },
    },
    "heisenberg_rpa": {
        "label": "Heisenberg RPA spin fluctuations",
        "description": (
            "Local relaxational spins coupled by Heisenberg exchange in the RPA: "
            "chi(Q,w) = [1 - chi0(w) J(Q)]^-1 chi0(w) with chi0(w) = chi0/(1 - i w/Gamma0). "
            "J(Q) is built from symmetry-distinct bond orbits (J1, J2, J3a, ...), each an "
            "exchange fit parameter in meV; J > 0 favors ordering where J(Q) is maximal and "
            "the fit is restricted to the paramagnetic side max J(Q) chi0 < 1. "
            "Configure the crystal and generate bond orbits, then fit. Requires dataset temperature."
        ),
        "structured_config": True,
        "dynamic_parameter_description": (
            "Heisenberg exchange constant of symmetry orbit {name} in meV; one shared value "
            "for every bond in the orbit. Positive J favors ordering at the wavevector "
            "maximizing J(Q)."
        ),
        "parameters": {
            "scale": {
                "default": 1.0,
                "description": (
                    "Overall intensity scale for unnormalized data. Degenerate with chi0's "
                    "magnitude only in part (chi0 also sets the RPA denominator), but fitting "
                    "both is usually ill-conditioned; consider fixing one."
                ),
                "allowed": "Positive finite number.",
                "type": "float",
                "example": "1.0",
                "global_fit": True,
            },
            "chi0": {
                "default": 0.1,
                "description": (
                    "Single-site static susceptibility entering the RPA denominator "
                    "(1/meV: the product J * chi0 is dimensionless). The magnetic instability "
                    "is at max J(Q) chi0 = 1."
                ),
                "allowed": "Positive finite number in 1/meV.",
                "type": "float",
                "example": "0.5",
                "global_fit": True,
            },
            "gamma0": {
                "default": 5.0,
                "description": (
                    "Bare single-site relaxation rate; the coupled mode at Q relaxes at "
                    "Gamma0 (1 - J(Q) chi0), softening toward the ordering vector."
                ),
                "allowed": "Positive finite number in meV.",
                "type": "float",
                "example": "5.0",
                "global_fit": True,
            },
        },
        "config": {
            "ion": {
                "default": "",
                "description": (
                    "Magnetic form factor multiplying the intensity: choose a tabulated ion, "
                    "Custom for explicit <j0> coefficients, or none for no form factor."
                ),
                "allowed": "An ion label from the ILL <j0> tables, Custom, or empty.",
                "type": "str",
                "example": "Yb3",
                "choices": "form_factor_ions",
            },
            "form_factor_coefficients": {
                "default": "",
                "description": (
                    "Custom <j0> coefficients A, a, B, b, C, c, D overriding the ion table "
                    "(see https://www.ill.eu/sites/ccsl/ffacts/)."
                ),
                "allowed": "Seven comma-separated numbers, or empty to use the ion table.",
                "type": "str",
                "example": "0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119",
            },
        },
    },
}


@dataclass
class NfitProject:
    """Serializable workspace state for the project explorer GUI."""

    data_groups: list[DataGroup] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def create_data_group(project: NfitProject, name: str | None = None) -> DataGroup:
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


def delete_data_group(project: NfitProject, group: DataGroup) -> None:
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


def create_fit_result_entry(
    group: DataGroup,
    parent: FitTimelineEntry,
    *,
    branch_timeline: bool = False,
    replace_current_state: bool = True,
    goodness: dict[str, Any] | None = None,
    channels: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
) -> FitTimelineEntry:
    """Insert a fit-result entry snapshotting the current group state."""

    snapshot = snapshot_data_group_state(group)
    result = FitTimelineEntry(
        name=next_fit_result_name(group.fits),
        kind="result",
        snapshot=copy.deepcopy(snapshot),
        created_at=_timestamp_now(),
        duration_seconds=duration_seconds,
        optimizer=str(parent.optimizer or "least_squares"),
        optimizer_config=copy.deepcopy(parent.optimizer_config),
        goodness=(
            {"status": "not run", "message": "The fit has not been executed."}
            if goodness is None
            else dict(goodness)
        ),
        channels={} if channels is None else dict(channels),
        metadata={} if metadata is None else dict(metadata),
    )
    if branch_timeline:
        timeline = FitTimelineEntry(
            name=next_fit_timeline_name(group.fits),
            kind="timeline",
            created_at=_timestamp_now(),
            children=[result],
        )
        parent.children.append(timeline)
    else:
        siblings = _fit_siblings(group.fits, parent)
        if siblings is None:
            siblings = group.fits
        insert_at = siblings.index(parent) + 1 if parent in siblings else len(siblings)
        siblings.insert(insert_at, result)
        if replace_current_state:
            _replace_current_state(siblings, group)
    return result


def run_group_fit(
    group: DataGroup,
    parent: FitTimelineEntry,
    *,
    branch_timeline: bool = False,
    progress_callback: Any | None = None,
) -> FitTimelineEntry:
    """Execute the group's fit and record the outcome in the fit history.

    The optimizer honors masks, disabled datasets/components, fit weights,
    and scale factors. Optimized parameters are written back to the model
    components before the result snapshot is taken, so the stored snapshot
    reproduces the fitted state. Fit and residual channels are evaluated once
    here and saved on the result entry; they are never recomputed on the fly.
    A failed fit still records an entry with the failure in ``goodness``.
    """

    start = time.perf_counter()
    channels: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    failed = False
    try:
        outcome = perform_group_fit(
            group,
            optimizer_config=parent.optimizer_config,
            progress_callback=progress_callback,
        )
        goodness = outcome["goodness"]
        channels = outcome["channels"]
        metadata = outcome.get("metadata", {})
    except Exception as exc:
        failed = True
        goodness = {"status": "failed", "message": str(exc)}
    duration = time.perf_counter() - start
    result = create_fit_result_entry(
        group,
        parent,
        branch_timeline=branch_timeline,
        replace_current_state=not failed,
        goodness=goodness,
        channels=channels,
        metadata=metadata,
        duration_seconds=duration,
    )
    if failed:
        _current_state_for_failed_fit(group, parent, result)
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
        existing.sharing = _sharing_from_payload(model_payload.get("sharing"))
        existing.limits = dict(model_payload.get("limits", {}) or {})
        existing.constraints = [
            dict(constraint) for constraint in model_payload.get("constraints", []) or []
        ]
        existing.applies_to = _applies_to_from_payload(model_payload.get("applies_to"))
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


def model_parameter_names(model: ModelComponentSpec) -> list[str]:
    """Return all parameter names of a model, static plus config-derived.

    Models such as ``heisenberg_rpa`` emit one exchange parameter per bond
    orbit in their configuration; this mirrors
    :func:`nfit.fit_config.component_parameter_names` for the GUI.
    """

    from .fit_config import component_parameter_names

    return list(component_parameter_names(model))


DEFAULT_BOND_CUTOFF_ANGSTROM = 6.0


class _NoChange(Exception):
    """Signals that a model-config mutation left the state untouched."""


def model_crystal_config(model: ModelComponentSpec) -> dict[str, Any]:
    """Return (creating if needed) the nested crystal config of a model.

    Shape: ``{"lattice": {"a", "b", "c", "alpha", "beta", "gamma"},
    "spacegroup": str, "sites": [{"label", "position", "ion"}]}``. The dict
    lives inside ``model.config`` and is plain JSON data, so it serializes
    with the project file.
    """

    crystal = model.config.get("crystal")
    if not isinstance(crystal, dict):
        crystal = {}
        model.config["crystal"] = crystal
    lattice = crystal.setdefault(
        "lattice",
        {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    for name, fallback in (
        ("a", 5.0), ("b", 5.0), ("c", 5.0),
        ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0),
    ):
        lattice.setdefault(name, fallback)
    crystal.setdefault("spacegroup", "P 1")
    crystal.setdefault("sites", [])
    return crystal


def reconcile_model_orbit_parameters(model: ModelComponentSpec) -> None:
    """Align a model's exchange parameters with its configured bond orbits.

    Values of orbits whose labels persist are kept; new orbits start at 0 meV
    and fixed; parameters of removed orbits are dropped along with their fit
    flags, limits, and sharing entries.
    """

    static = set(MODEL_TYPE_DEFINITIONS[model.type]["parameters"])
    labels = [str(orbit.get("label", "")) for orbit in model.config.get("orbits", [])]
    keep = static | set(labels)
    for mapping in (
        model.parameters,
        model.fit_parameters,
        model.global_fit,
        model.limits,
        model.sharing,
    ):
        if isinstance(mapping, dict):
            for key in [name for name in mapping if name not in keep]:
                mapping.pop(key)
    for label in labels:
        model.parameters.setdefault(label, 0.0)
        model.fit_parameters.setdefault(label, False)
        model.global_fit.setdefault(label, True)


def import_cif_into_model(
    model: ModelComponentSpec, path: str, *, group: DataGroup | None = None
) -> dict[str, Any]:
    """Load a CIF file into a model's crystal config (and optionally its group).

    Existing bond orbits are cleared because their site indices refer to the
    previous crystal. Returns the imported crystal dict.
    """

    from .crystal import crystal_from_cif

    imported = crystal_from_cif(path)
    model.config["crystal"] = imported
    model.config["magnetic_sites"] = []
    model.config.pop("orbits", None)
    model.config.pop("site_positions", None)
    reconcile_model_orbit_parameters(model)
    if group is not None:
        group.lattice_parameters = dict(imported["lattice"])
        group.spacegroup = imported["spacegroup"]
        group.metadata["crystal"] = copy.deepcopy(imported)
    return imported


def generate_model_bond_orbits(model: ModelComponentSpec) -> list[str]:
    """Generate symmetry-distinct bond orbits from a model's crystal config.

    Writes ``config["orbits"]`` and ``config["site_positions"]`` (the expanded
    magnetic sites the bond indices refer to), reconciles the exchange
    parameters, and returns the orbit labels.
    """

    from .crystal import generate_bond_orbits, orbits_to_config, sites_to_config

    crystal = model_crystal_config(model)
    magnetic = [str(label) for label in model.config.get("magnetic_sites", [])]
    if not magnetic:
        raise ValueError(
            "select at least one magnetic site (check 'Magnetic') before "
            "generating bond orbits"
        )
    cutoff = float(model.config.get("bond_cutoff_angstrom", DEFAULT_BOND_CUTOFF_ANGSTROM))
    sites, orbits = generate_bond_orbits(crystal, magnetic, cutoff)
    model.config["site_positions"] = sites_to_config(sites)
    model.config["orbits"] = orbits_to_config(orbits)
    reconcile_model_orbit_parameters(model)
    return [orbit.label for orbit in orbits]


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
    """Return standard hover text for a model parameter editor.

    Config-derived parameters (e.g. exchange constants of generated bond
    orbits) are not in the static definitions; they use the model type's
    ``dynamic_parameter_description`` template.
    """

    definition = MODEL_TYPE_DEFINITIONS[type]
    metadata = definition["parameters"].get(parameter_name)
    if metadata is None:
        template = definition.get(
            "dynamic_parameter_description",
            "Configuration-derived fit parameter {name}.",
        )
        return "\n".join(
            [
                f"Parameter: {parameter_name}",
                f"Description: {template.format(name=parameter_name)}",
                "Data type: float",
                "Default: 0",
                "Fit: checked means the optimizer may vary this parameter; unchecked means it is fixed at the displayed value.",
                "Global fit: checked means one shared value is fitted across datasets; unchecked means each dataset may fit its own value.",
            ]
        )
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
    if fit_entry.channels:
        lines.extend(["", f"Stored fit channels: {', '.join(sorted(fit_entry.channels))}"])
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
    data_lines.extend(_dataset_fit_summary_lines(dataset, group=group))
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


def data_group_composite_config(group: DataGroup) -> dict[str, Any]:
    """Return the group-level composite dataset configuration."""

    config = group.metadata.get(GROUP_COMPOSITE_KEY)
    if not isinstance(config, dict):
        config = {}
        group.metadata[GROUP_COMPOSITE_KEY] = config
    config.setdefault("enabled", False)
    config.setdefault("fractional", True)
    if config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "inverse_variance"
    try:
        config["max_batch_mb"] = max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        config["max_batch_mb"] = DEFAULT_REBIN_MAX_BATCH_MB
    config["normalize"] = True
    reference = _composite_reference_data(group)
    default_axes = _default_rebin_axes(reference) if reference is not None else []
    axes = config.get("axes")
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


def data_group_composite_enabled(group: DataGroup) -> bool:
    config = group.metadata.get(GROUP_COMPOSITE_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _composite_dataset_name(group: DataGroup) -> str:
    return f"{group.name} {GROUP_COMPOSITE_NAME}"


def _composite_candidates(group: DataGroup) -> list[DatasetEntry]:
    return [dataset for dataset in group.iter_datasets() if dataset.enabled]


def _dataset_composite_kind(dataset: DatasetEntry) -> str:
    if data_type_container(dataset.data_type) == "point_list" or isinstance(dataset.data, PointListData):
        return "point_list"
    if isinstance(dataset.data, MDHistoData) or data_type_container(dataset.data_type) == "mdhisto":
        return "mdhisto"
    if isinstance(dataset.data, PointData4D):
        return "point_data_4d"
    return type(dataset.data).__name__ if dataset.data is not None else "unknown"


def data_group_composite_status(group: DataGroup) -> tuple[bool, str]:
    datasets = _composite_candidates(group)
    if not datasets:
        return False, "No enabled datasets are available to combine."
    kinds = {_dataset_composite_kind(dataset) for dataset in datasets}
    if len(kinds) != 1:
        return False, "Composite datasets require all enabled datasets to hold the same kind of data."
    if next(iter(kinds)) not in {"mdhisto", "point_list", "point_data_4d"}:
        return False, "This dataset kind cannot be composited yet."
    return True, "Ready to combine enabled datasets into one rebinned composite."


def _composite_reference_data(group: DataGroup) -> Any | None:
    ok, _message = data_group_composite_status(group)
    if not ok:
        return None
    dataset = _composite_candidates(group)[0]
    try:
        return _source_data_for_group_composite(group, dataset)
    except Exception:
        return dataset.data


def _ensure_dataset_data_loaded(dataset: DatasetEntry) -> Any:
    if dataset.data is not None:
        return dataset.data
    if data_type_container(dataset.data_type) == "point_list":
        loaded = _load_point_list_dataset(dataset)
        if loaded is not None:
            return loaded
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if source and Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5"}:
        loaded = load_mantid_mdhisto_nxs(Path(source), copy_metadata=False)
        dataset.data = loaded
        dataset.kind = dataset.kind or Path(source).suffix.lstrip(".").lower()
        dataset.metadata["import_status"] = "loaded"
        return loaded
    return dataset.data


def _source_data_for_group_composite(group: DataGroup, dataset: DatasetEntry) -> Any:
    data = _ensure_dataset_data_loaded(dataset)
    extra_masks = effective_dataset_masks(group, dataset)
    if isinstance(data, PointListData):
        return prepared_point_list_data(dataset)
    if isinstance(data, MDHistoData):
        return _mdhisto_with_nfit_masks(dataset, data=data, extra_masks=extra_masks)
    if isinstance(data, PointData4D):
        return _point_data_with_nfit_masks(dataset, data, extra_masks=extra_masks)
    return data


def composite_dataset_data(group: DataGroup) -> MDHistoData | PointListData | PointData4D:
    """Build the group's rebinned composite dataset from enabled members."""

    ok, message = data_group_composite_status(group)
    if not ok:
        raise ValueError(message)
    config = data_group_composite_config(group)
    kind = _dataset_composite_kind(_composite_candidates(group)[0])
    if kind == "mdhisto":
        return _composite_mdhisto_data(group, config)
    if kind == "point_list":
        return _composite_point_list_data(group, config)
    if kind == "point_data_4d":
        return _composite_point_data(group, config)
    raise ValueError(f"unsupported composite dataset kind {kind!r}")


def composite_dataset_entry(group: DataGroup) -> DatasetEntry:
    datasets = _composite_candidates(group)
    first = datasets[0] if datasets else None
    return DatasetEntry(
        name=_composite_dataset_name(group),
        data=composite_dataset_data(group),
        kind=(first.kind if first is not None else ""),
        data_type=(first.data_type if first is not None else ""),
        metadata={"source_group": group.name, "composite": True},
        parameters={},
        enabled=True,
        fit_weight=1.0,
        scale_factor=1.0,
    )


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


def _composite_mdhisto_data(group: DataGroup, config: dict[str, Any]) -> MDHistoData:
    lower, upper, num_bins = _composite_rebin_bounds(config)
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    first_data: MDHistoData | None = None
    for dataset in _composite_candidates(group):
        data = _source_data_for_group_composite(group, dataset)
        if not isinstance(data, MDHistoData):
            continue
        if first_data is None:
            first_data = data
        source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
        coords = np.stack(source_grids, axis=-1)
        valid = np.isfinite(data.signal) & np.isfinite(data.errors) & ~np.asarray(data.mask, dtype=bool)
        if data.num_events is not None:
            valid &= np.asarray(data.num_events) > 0.0
        if not np.any(valid):
            continue
        scale = float(dataset.scale_factor)
        signal = np.asarray(data.signal[valid], dtype=float) * scale
        errors = _scaled_error_for_weight(dataset, np.asarray(data.errors[valid], dtype=float))
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
    result = rebin_nd(
        signal_all,
        coords_all,
        data_errs=errors_all,
        data_weights=weights_all,
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", True)),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None or result.bins_list is None:
        raise RuntimeError("composite rebinning did not produce binned data")
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    axes = tuple(
        MDHistoAxis(
            name=str(axis_config.get("name") or source_axis.name),
            values=np.asarray(bins, dtype=float),
            units=str(axis_config.get("units") if axis_config.get("units") is not None else source_axis.units),
            kind=source_axis.kind,
            frame=source_axis.frame,
            path=source_axis.path,
            metadata=dict(source_axis.metadata),
        )
        for source_axis, axis_config, bins in zip(first_data.axes, axes_config, result.bins_list, strict=True)
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    metadata = {
        "composite": True,
        "source_group": group.name,
        "source_datasets": [dataset.name for dataset in _composite_candidates(group)],
        "rebin": {
            "lower": lower,
            "upper": upper,
            "step_size": np.asarray(result.step_size, dtype=float).tolist(),
            "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
            "fractional": bool(config.get("fractional", True)),
            "normalize": True,
            "mean_weighting": _rebin_mean_weighting(config),
            "max_batch_mb": _rebin_max_batch_mb(config),
            "max_batch_bytes": _rebin_max_batch_bytes(config),
            "weighted_by_fit_weight": True,
        },
    }
    return MDHistoData(
        axes=axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=first_data.coordinate_system,
        visual_normalization=first_data.visual_normalization,
        metadata=metadata,
    )


def _composite_point_data(group: DataGroup, config: dict[str, Any]) -> PointData4D:
    lower, upper, num_bins = _composite_rebin_bounds(config)
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for dataset in _composite_candidates(group):
        data = _source_data_for_group_composite(group, dataset)
        if not isinstance(data, PointData4D):
            continue
        source = data.valid(require_positive_sigma=False)
        if source.size == 0:
            continue
        coords_parts.append(np.column_stack(source.coordinates()))
        scale = float(dataset.scale_factor)
        signal = np.asarray(source.intensity, dtype=float) * scale
        signal_parts.append(signal)
        error_parts.append(_scaled_error_for_weight(dataset, source.sigma))
        weight_parts.append(_dataset_statistical_weight(dataset, source.size))
    if not signal_parts:
        raise ValueError("no valid data points remain before compositing")
    result = rebin_nd(
        np.concatenate(signal_parts),
        np.concatenate(coords_parts, axis=0),
        data_errs=np.concatenate(error_parts),
        data_weights=np.concatenate(weight_parts),
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", True)),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
    )
    if result.bin_centers_list is None or result.binned_data is None or result.binned_data_errs is None or result.n_samples is None:
        raise RuntimeError("composite rebinning did not produce binned data")
    H_grid, K_grid, L_grid, E_grid = np.meshgrid(*result.bin_centers_list, indexing="ij")
    mask = np.isfinite(result.binned_data) & np.isfinite(result.binned_data_errs) & (result.n_samples > 0.0)
    return PointData4D(
        H_grid.ravel(),
        K_grid.ravel(),
        L_grid.ravel(),
        E_grid.ravel(),
        np.asarray(result.binned_data, dtype=float).ravel(),
        np.asarray(result.binned_data_errs, dtype=float).ravel(),
        mask=mask.ravel(),
        metadata={"composite": True, "source_group": group.name},
    )


def _composite_point_list_data(group: DataGroup, config: dict[str, Any]) -> PointListData:
    datasets = _composite_candidates(group)
    prepared = [_source_data_for_group_composite(group, dataset) for dataset in datasets]
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
    result = rebin_nd(
        np.concatenate(signal_parts),
        np.concatenate(coords_parts, axis=0),
        data_errs=np.concatenate(error_parts),
        data_weights=np.concatenate(weight_parts),
        lower=lower,
        upper=upper,
        num_bins=num_bins,
        fractional=bool(config.get("fractional", True)),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None or result.bin_centers_list is None:
        raise RuntimeError("composite rebinning did not produce binned data")
    occupied = result.n_samples > 0.0
    center_grids = np.meshgrid(*result.bin_centers_list, indexing="ij")
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
        metadata={"composite": True, "source_group": group.name, "source_datasets": [dataset.name for dataset in datasets]},
    )


def slice_viewer_datasets(
    group: DataGroup,
    *,
    use_composite: bool = True,
) -> tuple[list[MDHistoData], list[str]]:
    """Return data and labels for every dataset in the group tree, with shared masks."""

    data: list[MDHistoData] = []
    names: list[str] = []
    model_channels = current_model_channels(group)
    if use_composite and data_group_composite_enabled(group):
        composite = composite_dataset_entry(group)
        view_data = dataset_for_slice_viewer(composite)
        if view_data is not None:
            composite_name = composite.name
            attach_fit_channels_to_view(
                group,
                composite_name,
                view_data,
                fallback_payload=model_channels.get(composite_name),
            )
            data.append(view_data)
            names.append(composite_name)
        return data, names
    for dataset in group.iter_datasets():
        extra_masks = effective_dataset_masks(group, dataset)
        view_data = dataset_for_slice_viewer(dataset, extra_masks=extra_masks)
        if view_data is not None:
            attach_fit_channels_to_view(
                group,
                dataset.name,
                view_data,
                fallback_payload=model_channels.get(dataset.name),
            )
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
        )
    return data


FIT_CHANNEL_NAMES = ("fit", "residual")


@dataclass
class FitDataBundle:
    """Fit-ready representations of one dataset.

    ``view`` is the masked, scaled data exactly as the data viewer shows it.
    ``points`` flattens every view point into :class:`PointData4D`; the point
    mask is ``False`` where the view masks a point, so the optimizer only sees
    unmasked data while the fitted model can still be evaluated everywhere.
    ``grid_shape`` reshapes point-ordered channel arrays back onto the MDHisto
    grid (``None`` for point lists).
    """

    dataset: DatasetEntry
    view: Any
    points: PointData4D
    grid_shape: tuple[int, ...] | None


def fit_data_bundle(group: DataGroup, dataset: DatasetEntry) -> FitDataBundle | None:
    """Build the fit-ready views of one dataset, or ``None`` if unsupported."""

    extra_masks = effective_dataset_masks(group, dataset)
    if isinstance(dataset.data, PointData4D):
        points = dataset.data
        _apply_sample_context_to_points(group, dataset, points)
        return FitDataBundle(dataset=dataset, view=dataset.data, points=points, grid_shape=None)
    view = dataset_for_slice_viewer(dataset, extra_masks=extra_masks)
    if isinstance(view, MDHistoData):
        points = _point_data_from_mdhisto_view(view)
    elif isinstance(view, PointListData):
        points = _point_data_from_point_list_view(view)
    else:
        return None
    _apply_sample_context_to_points(group, dataset, points)
    return FitDataBundle(
        dataset=dataset,
        view=view,
        points=points,
        grid_shape=view.shape if isinstance(view, MDHistoData) else None,
    )


def effective_dataset_temperature(
    group: DataGroup, dataset: DatasetEntry
) -> float | None:
    """Return the dataset temperature override in K, or ``None`` if unset.

    ``dataset.parameters["temperature"]`` takes precedence over any
    temperature carried by the imported data itself.
    """

    value = dataset.parameters.get("temperature")
    if value in (None, ""):
        return None
    return float(value)


def _apply_sample_context_to_points(
    group: DataGroup, dataset: DatasetEntry, points: PointData4D
) -> None:
    """Stamp per-dataset temperature and group lattice metadata onto fit points.

    Physics models read the sample temperature from ``PointData4D.temperature``
    and convert HKL to ``|Q|`` through ``rlu_to_inv_angstrom_matrix`` metadata;
    both are supplied here so models never depend on GUI state.
    """

    override = effective_dataset_temperature(group, dataset)
    if override is not None:
        points.temperature = override
    if (
        "rlu_to_inv_angstrom_matrix" not in points.metadata
        and isinstance(group.lattice_parameters, dict)
        and all(key in group.lattice_parameters for key in ("a", "b", "c"))
    ):
        lattice = group.lattice_parameters
        matrix = reciprocal_basis_from_lattice_parameters(
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice.get("alpha", 90.0)),
            float(lattice.get("beta", 90.0)),
            float(lattice.get("gamma", 90.0)),
        )
        points.metadata["lattice_parameters"] = {
            "a": float(lattice["a"]),
            "b": float(lattice["b"]),
            "c": float(lattice["c"]),
            "alpha": float(lattice.get("alpha", 90.0)),
            "beta": float(lattice.get("beta", 90.0)),
            "gamma": float(lattice.get("gamma", 90.0)),
            "angle_units": "degree",
            "include_2pi": True,
        }
        points.metadata["rlu_to_inv_angstrom_matrix"] = matrix.tolist()


def _point_data_from_mdhisto_view(data: MDHistoData) -> PointData4D:
    """Flatten a masked MDHisto view into fit points using axis roles."""

    coords = _mdhisto_coordinate_grids(data)
    zeros = np.zeros(data.shape, dtype=float)
    keep = ~np.asarray(data.mask, dtype=bool)
    keep &= np.asarray(data.num_events, dtype=float) > 0.0
    metadata: dict[str, Any] = {
        "fit_coordinates": sorted(name for name in ("H", "K", "L", "E") if name in coords),
    }
    for key in ("oriented_lattice", "coordinate_units", "rlu_to_inv_angstrom_matrix"):
        if key in data.metadata:
            metadata[key] = data.metadata[key]
    temperature = data.metadata.get("temperature")
    return PointData4D(
        H=coords.get("H", zeros).ravel(),
        K=coords.get("K", zeros).ravel(),
        L=coords.get("L", zeros).ravel(),
        E=coords.get("E", zeros).ravel(),
        intensity=np.asarray(data.signal, dtype=float).ravel(),
        sigma=np.asarray(data.errors, dtype=float).ravel(),
        mask=keep.ravel(),
        temperature=float(temperature) if temperature is not None else None,
        metadata=metadata,
    )


def _point_data_from_point_list_view(data: PointListData) -> PointData4D:
    """Map a point-list view onto fit points.

    Columns named ``H``, ``K``, ``L``, or ``E`` (case-insensitive) become the
    matching fit coordinates. When no energy-like column exists, the first
    coordinate column is stored in ``E`` so one-dimensional models have an
    axis to work with; the mapping is recorded in the point metadata.
    """

    if not data.channel_labels:
        raise ValueError("point-list dataset defines no data channels")
    label = data.channel_labels[0]
    intensity = np.asarray(data.channel_values(label), dtype=float)
    errors = data.channel_errors(label)
    sigma_known = errors is not None
    sigma = (
        np.asarray(errors, dtype=float)
        if sigma_known
        else np.ones(intensity.shape, dtype=float)
    )

    columns_by_role: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    for name in data.coordinate_names:
        role = name.strip().upper()
        if role in ("H", "K", "L", "E") and role not in columns_by_role:
            columns_by_role[role] = np.asarray(data.column(name), dtype=float)
            mapping[role] = name
    if "E" not in columns_by_role and data.coordinate_names:
        first = data.coordinate_names[0]
        if first not in mapping.values():
            columns_by_role["E"] = np.asarray(data.column(first), dtype=float)
            mapping["E"] = first

    n = intensity.size
    zeros = np.zeros(n, dtype=float)
    mask = np.isfinite(intensity) & np.isfinite(sigma)
    if sigma_known:
        mask &= sigma > 0.0
    temperature: float | np.ndarray | None = None
    for name in data.columns:
        if name.strip().lower() == "temperature":
            temperature = np.asarray(data.column(name), dtype=float)
            break
    if temperature is None and data.metadata.get("temperature") is not None:
        temperature = float(data.metadata["temperature"])
    return PointData4D(
        H=columns_by_role.get("H", zeros),
        K=columns_by_role.get("K", zeros),
        L=columns_by_role.get("L", zeros),
        E=columns_by_role.get("E", zeros),
        intensity=intensity,
        sigma=sigma,
        mask=mask,
        temperature=temperature,
        metadata={
            "fit_coordinate_mapping": mapping,
            "fit_channel": label,
            "sigma_known": sigma_known,
        },
    )


def fit_dataset_inputs(
    group: DataGroup,
) -> tuple[list[FitDatasetInput], dict[str, FitDataBundle]]:
    """Prepare every enabled dataset in a group for the fit compiler."""

    inputs: list[FitDatasetInput] = []
    bundles: dict[str, FitDataBundle] = {}
    if data_group_composite_enabled(group):
        composite = composite_dataset_entry(group)
        bundle = fit_data_bundle(group, composite)
        if bundle is None:
            return inputs, bundles
        inputs.append(
            FitDatasetInput(
                name=composite.name,
                data=bundle.points,
                weight=1.0,
                data_type=composite.data_type or DEFAULT_DATA_TYPE,
            )
        )
        bundles[composite.name] = bundle
        return inputs, bundles
    for dataset in group.iter_datasets():
        if not dataset.enabled:
            continue
        bundle = fit_data_bundle(group, dataset)
        if bundle is None:
            continue
        inputs.append(
            FitDatasetInput(
                name=dataset.name,
                data=bundle.points,
                weight=float(dataset.fit_weight),
                data_type=dataset.data_type or DEFAULT_DATA_TYPE,
            )
        )
        bundles[dataset.name] = bundle
    return inputs, bundles


_OPTIMIZER_KWARG_NAMES = ("max_nfev", "xtol", "ftol", "gtol", "loss", "f_scale")


def _optimizer_kwargs(optimizer_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(optimizer_config, dict):
        return {}
    kwargs: dict[str, Any] = {}
    for name in _OPTIMIZER_KWARG_NAMES:
        value = optimizer_config.get(name)
        if value in (None, ""):
            continue
        if name == "max_nfev":
            kwargs[name] = int(value)
        elif name == "loss":
            kwargs[name] = str(value)
        else:
            kwargs[name] = float(value)
    init_config = optimizer_config.get("initialization")
    if isinstance(init_config, dict) and init_config.get("enabled", False):
        kwargs["initialization"] = {
            key: value for key, value in init_config.items() if key != "enabled"
        }
        kwargs["initialization"].setdefault("method", "differential_evolution")
    return kwargs


def _sampler_config(optimizer_config: dict[str, Any] | None) -> SamplerConfig | None:
    if not isinstance(optimizer_config, dict):
        return None
    sampler = optimizer_config.get("sampler")
    if not isinstance(sampler, dict) or not sampler.get("enabled", False):
        return None
    return SamplerConfig(
        method=str(sampler.get("method", "emcee")),
        n_walkers=_optional_int(sampler.get("n_walkers")),
        n_steps=_optional_int(sampler.get("n_steps")) or 1000,
        burn_in=int(sampler.get("burn_in", 0) or 0),
        thin=max(1, int(sampler.get("thin", 1) or 1)),
        random_seed=_optional_int(sampler.get("random_seed")),
        kwargs={
            **(dict(sampler.get("kwargs", {})) if isinstance(sampler.get("kwargs"), dict) else {}),
            **(
                {"workers": int(sampler.get("workers"))}
                if sampler.get("workers") not in (None, "")
                else {}
            ),
        },
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def perform_group_fit(
    group: DataGroup,
    *,
    optimizer_config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Fit the group's enabled model components to its enabled datasets.

    Masks, active/inactive datasets, fit weights, and scale factors are all
    honored because the fit points come from the same prepared views the data
    viewer shows. Optimized parameter values are written back to the model
    components, and per-dataset fit/residual channels are evaluated once over
    each dataset's full view so they can be stored with the fit result.
    """

    components = [
        model for model in group.models.values() if isinstance(model, ModelComponentSpec)
    ]
    if not any(component.enabled for component in components):
        raise ValueError("the data group has no enabled model components")
    inputs, bundles = fit_dataset_inputs(group)
    if not inputs:
        raise ValueError("no enabled dataset could be prepared for fitting")

    compiled = compile_fit_problem(components, inputs, description=group.name)
    config = OptimizationConfig(kwargs=_optimizer_kwargs(optimizer_config))
    progress_events: list[dict[str, Any]] = []

    def record_progress(event: dict[str, Any]) -> None:
        if len(progress_events) < 200:
            progress_events.append(_progress_event_summary(event))
        if progress_callback is not None:
            progress_callback(event)

    result = fit_problem_least_squares(
        compiled.problem,
        config=config,
        progress_callback=record_progress,
    )
    sampler_result: SamplingResult | None = None
    sampler = _sampler_config(optimizer_config)
    if sampler is not None:
        sampler_result = sample_problem_parameters(
            compiled.problem,
            sampler,
            initial_params=result.params,
            require_positive_sigma=config.require_positive_sigma,
            progress_callback=record_progress,
        )
    _write_back_fitted_parameters(group, components, compiled, result)
    channels = _fit_channels_from_result(compiled, result, bundles)
    goodness: dict[str, Any] = {
        "status": "converged" if result.success else "not converged",
        "message": result.message,
        "chi2": float(result.chi2),
        "reduced_chi2": float(result.reduced_chi2),
        "n_points": int(sum(result.dataset_sizes.values())),
        "n_variables": len(result.variable_names),
        "parameters": {name: float(value) for name, value in result.params.items()},
        "stderr": (
            {name: float(value) for name, value in result.stderr.items()}
            if result.stderr
            else {}
        ),
        "covariance": _matrix_summary(result.covariance, result.variable_names),
        "dataset_chi2": {name: float(value) for name, value in result.dataset_chi2.items()},
        "dataset_reduced_chi2": {
            name: float(value) for name, value in result.dataset_reduced_chi2.items()
        },
        "skipped_datasets": list(compiled.skipped_datasets),
    }
    metadata: dict[str, Any] = {}
    parameter_labels = _fit_parameter_labels_from_components(components, compiled, result.variable_names)
    if parameter_labels:
        metadata["parameter_labels"] = parameter_labels
    if progress_events:
        metadata["progress_log"] = progress_events
    if sampler_result is not None:
        posterior = _posterior_summary(sampler_result)
        goodness["posterior"] = posterior
        metadata["posterior_samples"] = _sampling_result_to_dict(sampler_result)
    return {
        "result": result,
        "compiled": compiled,
        "goodness": goodness,
        "channels": channels,
        "metadata": metadata,
    }


# Overlay recomputation is dominated by rebuilding the fit bundles (rebinning
# the full volume) and the RPA phase geometry. Neither depends on model
# *parameter values*, so both are cached per group and reused when only a
# parameter changes -- turning an O(10 s) rebuild on every edit into an O(0.1 s)
# re-evaluation. The cache is keyed on a structural signature that excludes
# parameter values (see _overlay_cache_signature).
_MODEL_OVERLAY_CACHE: dict[int, dict[str, Any]] = {}
_MODEL_OVERLAY_CACHE_LIMIT = 6


def _overlay_cache_signature(group: DataGroup) -> str:
    """Structural fingerprint of a group's overlay inputs, excluding values.

    Everything that changes the bundles (data identity, masks, rebin,
    temperature, scale) or the model structure (types, config, sharing,
    applies-to, enablement) is included; per-parameter *values* are excluded so
    editing e.g. ``J2`` reuses the cached bundles and geometry.
    """

    datasets: list[Any] = []
    for dataset in group.iter_datasets():
        datasets.append(
            [
                dataset.name,
                bool(dataset.enabled),
                dataset.data_type,
                dataset.kind,
                id(dataset),
                float(dataset.fit_weight),
                json.dumps(dataset.parameters, sort_keys=True, default=str),
                effective_dataset_temperature(group, dataset),
                [
                    [mask.type, bool(mask.enabled), bool(mask.invert), bool(mask.additive),
                     json.dumps(mask.parameters, sort_keys=True, default=str)]
                    for mask in effective_dataset_masks(group, dataset)
                ],
            ]
        )
    models: list[Any] = []
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            continue
        models.append(
            [
                model.name,
                model.type,
                bool(model.enabled),
                list(model.applies_to) if model.applies_to is not None else None,
                json.dumps(model.config, sort_keys=True, default=str),
                json.dumps(model.sharing, sort_keys=True, default=str),
                json.dumps(model.constraints, sort_keys=True, default=str),
            ]
        )
    payload = [
        datasets,
        models,
        json.dumps(group.lattice_parameters, sort_keys=True, default=str),
        group.spacegroup,
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def _overlay_current_params(
    group: DataGroup, compiled: CompiledFitProblem
) -> dict[str, float]:
    """Map current component parameter values onto a compiled problem's specs.

    The compiled problem may be cached (its spec values are stale), so read the
    live values from the components through the instance bookkeeping. Derived
    (constraint) parameters keep their compiled default.
    """

    params = {spec.name: float(spec.value) for spec in compiled.problem.parameter_specs}
    for instance in compiled.parameter_instances.values():
        component = group.models.get(instance.component)
        if not isinstance(component, ModelComponentSpec):
            continue
        value = component.parameters.get(instance.parameter)
        if value is None:
            continue
        try:
            params[instance.name] = float(value)
        except (TypeError, ValueError):
            continue
    return params


def current_model_channels(group: DataGroup) -> dict[str, dict[str, Any]]:
    """Evaluate enabled model components at their current parameter values."""

    components = [
        model for model in group.models.values() if isinstance(model, ModelComponentSpec)
    ]
    if not any(component.enabled for component in components):
        _MODEL_OVERLAY_CACHE.pop(id(group), None)
        return {}
    signature = _overlay_cache_signature(group)
    cached = _MODEL_OVERLAY_CACHE.get(id(group))
    if cached is not None and cached["signature"] == signature:
        compiled = cached["compiled"]
        bundles = cached["bundles"]
        subsets = cached["subsets"]
    else:
        inputs, bundles = fit_dataset_inputs(group)
        if not inputs:
            _MODEL_OVERLAY_CACHE.pop(id(group), None)
            return {}
        try:
            compiled = compile_fit_problem(components, inputs, description=group.name)
        except Exception:
            return {}
        subsets = {}
        if len(_MODEL_OVERLAY_CACHE) >= _MODEL_OVERLAY_CACHE_LIMIT:
            _MODEL_OVERLAY_CACHE.clear()
        _MODEL_OVERLAY_CACHE[id(group)] = {
            "signature": signature,
            "compiled": compiled,
            "bundles": bundles,
            "subsets": subsets,
        }
    try:
        params = _overlay_current_params(group, compiled)
        return _fit_channels_from_params(compiled, params, bundles, subset_cache=subsets)
    except Exception:
        return {}


def _compiled_problem_for_fit_entry(
    group: DataGroup,
    fit_entry: FitTimelineEntry,
) -> CompiledFitProblem:
    """Compile the fitting problem from a stored fit snapshot without leaving state changed."""

    current_snapshot = snapshot_data_group_state(group)
    try:
        if fit_entry.snapshot:
            restore_data_group_state(group, fit_entry.snapshot)
        components = [
            model for model in group.models.values() if isinstance(model, ModelComponentSpec)
        ]
        inputs, _bundles = fit_dataset_inputs(group)
        if not inputs:
            raise ValueError("no enabled dataset could be prepared for posterior sampling")
        return compile_fit_problem(components, inputs, description=group.name)
    finally:
        restore_data_group_state(group, current_snapshot)


def _matrix_summary(matrix: Any, names: list[str]) -> dict[str, Any]:
    if matrix is None:
        return {}
    arr = np.asarray(matrix, dtype=float)
    summary: dict[str, Any] = {"variables": list(names), "shape": list(arr.shape)}
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] == len(names):
        summary["matrix"] = arr.tolist()
        summary["diagonal"] = {name: float(arr[index, index]) for index, name in enumerate(names)}
        denom = np.sqrt(np.outer(np.diag(arr), np.diag(arr)))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
        summary["correlation"] = {
            name: {
                other: float(corr[i, j])
                for j, other in enumerate(names)
            }
            for i, name in enumerate(names)
        }
    return summary


def _progress_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "stage": str(event.get("stage", "fit")),
        "message": str(event.get("message", "")),
    }
    for key in ("iteration", "total", "cost", "convergence", "elapsed_seconds", "seconds_per_step"):
        if event.get(key) is not None:
            value = event[key]
            summary[key] = (
                float(value)
                if key in {"cost", "convergence", "elapsed_seconds", "seconds_per_step"}
                else int(value)
            )
    params = event.get("parameters")
    if isinstance(params, dict):
        summary["parameters"] = {name: float(value) for name, value in params.items()}
    return summary


def _posterior_summary(result: SamplingResult) -> dict[str, Any]:
    samples = np.asarray(result.samples, dtype=float)
    summary: dict[str, Any] = dict(result.metadata)
    if samples.size == 0:
        summary["samples"] = 0
        return summary
    summary["samples"] = int(samples.shape[0])
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    summary["parameters"] = {
        name: {
            "p16": float(percentiles[0, index]),
            "median": float(percentiles[1, index]),
            "p84": float(percentiles[2, index]),
        }
        for index, name in enumerate(result.variable_names)
    }
    if samples.shape[1] > 1:
        corr = np.corrcoef(samples, rowvar=False)
        summary["correlation"] = {
            name: {
                other: float(corr[i, j])
                for j, other in enumerate(result.variable_names)
            }
            for i, name in enumerate(result.variable_names)
        }
    return summary


def _sampling_result_to_dict(result: SamplingResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "variable_names": list(result.variable_names),
        "samples": _encode_float_array(result.samples),
        "metadata": dict(result.metadata),
    }
    if result.log_probability is not None:
        payload["log_probability"] = _encode_float_array(result.log_probability)
    if result.chain is not None:
        payload["chain"] = _encode_float_array(result.chain)
    if result.log_probability_chain is not None:
        payload["log_probability_chain"] = _encode_float_array(result.log_probability_chain)
    return payload


def _sampling_result_from_dict(payload: Any) -> SamplingResult | None:
    if not isinstance(payload, dict):
        return None
    try:
        samples = _decode_float_array(payload["samples"])
    except Exception:
        return None
    log_probability = None
    if isinstance(payload.get("log_probability"), dict):
        try:
            log_probability = _decode_float_array(payload["log_probability"])
        except Exception:
            log_probability = None
    chain = None
    if isinstance(payload.get("chain"), dict):
        try:
            chain = _decode_float_array(payload["chain"])
        except Exception:
            chain = None
    log_probability_chain = None
    if isinstance(payload.get("log_probability_chain"), dict):
        try:
            log_probability_chain = _decode_float_array(payload["log_probability_chain"])
        except Exception:
            log_probability_chain = None
    return SamplingResult(
        samples=samples,
        variable_names=[str(name) for name in payload.get("variable_names", [])],
        log_probability=log_probability,
        metadata=dict(payload.get("metadata", {})),
        chain=chain,
        log_probability_chain=log_probability_chain,
    )


def _sampling_result_with_window(result: SamplingResult, burn_in: int, thin: int) -> SamplingResult:
    """Return a copy flattened with a different burn-in/thinning window."""

    if result.chain is None:
        raise ValueError("stored posterior does not include a raw emcee chain")
    chain = np.asarray(result.chain, dtype=float)
    if chain.ndim != 3 or chain.shape[0] == 0:
        raise ValueError("stored posterior chain is empty or malformed")
    burn = min(max(0, int(burn_in)), max(0, chain.shape[0] - 1))
    step = max(1, int(thin))
    sliced = chain[burn::step]
    samples = sliced.reshape((-1, chain.shape[2]))
    log_probability = None
    if result.log_probability_chain is not None:
        log_chain = np.asarray(result.log_probability_chain, dtype=float)
        if log_chain.shape[:2] == chain.shape[:2]:
            log_probability = log_chain[burn::step].reshape(-1)
    metadata = dict(result.metadata)
    metadata["burn_in"] = burn
    metadata["thin"] = step
    metadata["samples"] = int(samples.shape[0])
    metadata["n_steps"] = int(chain.shape[0])
    metadata["n_walkers"] = int(chain.shape[1])
    return SamplingResult(
        samples=samples,
        variable_names=list(result.variable_names),
        log_probability=log_probability,
        metadata=metadata,
        chain=chain,
        log_probability_chain=result.log_probability_chain,
    )


def _combined_sampling_result(
    original: SamplingResult,
    appended: SamplingResult,
    *,
    burn_in: int,
    thin: int,
) -> SamplingResult:
    """Concatenate two emcee chains and re-flatten them with the requested window."""

    if original.chain is None or appended.chain is None:
        raise ValueError("appending posterior samples requires raw emcee chains")
    if list(original.variable_names) != list(appended.variable_names):
        raise ValueError("appended posterior variables do not match the stored chain")
    chain = np.concatenate(
        [np.asarray(original.chain, dtype=float), np.asarray(appended.chain, dtype=float)],
        axis=0,
    )
    log_probability_chain = None
    if original.log_probability_chain is not None and appended.log_probability_chain is not None:
        log_probability_chain = np.concatenate(
            [
                np.asarray(original.log_probability_chain, dtype=float),
                np.asarray(appended.log_probability_chain, dtype=float),
            ],
            axis=0,
        )
    metadata = dict(original.metadata)
    metadata.update(dict(appended.metadata))
    metadata["n_steps"] = int(chain.shape[0])
    metadata["n_walkers"] = int(chain.shape[1])
    metadata["appended_steps"] = int(appended.chain.shape[0])
    result = SamplingResult(
        samples=np.empty((0, chain.shape[2]), dtype=float),
        variable_names=list(original.variable_names),
        log_probability=None,
        metadata=metadata,
        chain=chain,
        log_probability_chain=log_probability_chain,
    )
    return _sampling_result_with_window(result, burn_in=burn_in, thin=thin)


def _store_sampling_result_on_fit_entry(fit_entry: FitTimelineEntry, result: SamplingResult) -> None:
    fit_entry.metadata["posterior_samples"] = _sampling_result_to_dict(result)
    fit_entry.goodness["posterior"] = _posterior_summary(result)


def _best_posterior_sample(
    result: SamplingResult,
) -> tuple[dict[str, float], float, dict[str, int]] | None:
    """Return the stored sample with the highest finite emcee log probability."""

    names = list(result.variable_names)
    if not names:
        return None
    if result.chain is not None and result.log_probability_chain is not None:
        chain = np.asarray(result.chain, dtype=float)
        log_probability = np.asarray(result.log_probability_chain, dtype=float)
        if chain.ndim == 3 and log_probability.shape == chain.shape[:2]:
            finite = np.isfinite(log_probability)
            if np.any(finite):
                masked = np.where(finite, log_probability, -np.inf)
                flat_index = int(np.argmax(masked))
                step, walker = np.unravel_index(flat_index, log_probability.shape)
                sample = chain[int(step), int(walker)]
                if sample.shape[0] == len(names):
                    return (
                        {name: float(sample[index]) for index, name in enumerate(names)},
                        float(log_probability[int(step), int(walker)]),
                        {"step": int(step), "walker": int(walker)},
                    )
    if result.log_probability is None:
        return None
    samples = np.asarray(result.samples, dtype=float)
    log_probability = np.asarray(result.log_probability, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(names) or log_probability.shape != (samples.shape[0],):
        return None
    finite = np.isfinite(log_probability)
    if not np.any(finite):
        return None
    sample_index = int(np.argmax(np.where(finite, log_probability, -np.inf)))
    sample = samples[sample_index]
    return (
        {name: float(sample[index]) for index, name in enumerate(names)},
        float(log_probability[sample_index]),
        {"sample": int(sample_index)},
    )


def _log_probability_for_params(
    compiled: CompiledFitProblem,
    params: dict[str, float],
) -> float:
    """Evaluate the Gaussian log likelihood used by emcee for a parameter set."""

    trial = {spec.name: float(spec.value) for spec in compiled.problem.parameter_specs}
    trial.update({str(name): float(value) for name, value in params.items()})
    evaluation = _evaluate_problem(
        compiled.problem,
        trial,
        require_positive_sigma=True,
    )
    chi2 = float(np.dot(evaluation.residuals, evaluation.residuals))
    return -0.5 * chi2


def _best_posterior_promotion_candidate(
    fit_entry: FitTimelineEntry,
    compiled: CompiledFitProblem | None = None,
) -> tuple[dict[str, float], float, float, dict[str, int]] | None:
    """Return a best-sample candidate only when it improves on the fit result."""

    stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if stored is None:
        return None
    best = _best_posterior_sample(stored)
    if best is None:
        return None
    sample_params, sample_log_probability, location = best
    try:
        baseline_log_probability = -0.5 * float(fit_entry.goodness["chi2"])
    except (KeyError, TypeError, ValueError):
        goodness_params = fit_entry.goodness.get("parameters")
        if compiled is None or not isinstance(goodness_params, dict):
            return None
        baseline_log_probability = _log_probability_for_params(compiled, goodness_params)
    if not np.isfinite(sample_log_probability) or not np.isfinite(baseline_log_probability):
        return None
    if sample_log_probability <= baseline_log_probability:
        return None
    return sample_params, sample_log_probability, baseline_log_probability, location


def _fit_parameter_labels_from_components(
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    variable_names: list[str],
) -> dict[str, str]:
    by_component = {component.name: component for component in components}
    labels: dict[str, str] = {}
    for name in variable_names:
        instance = compiled.parameter_instances.get(name)
        if instance is None:
            continue
        component = by_component.get(instance.component)
        if component is None:
            continue
        parameter_labels = component.metadata.get("parameter_labels")
        if not isinstance(parameter_labels, dict):
            continue
        label = str(parameter_labels.get(instance.parameter, "")).strip()
        if label:
            labels[name] = label
    return labels


def _write_back_parameter_values(
    group: DataGroup,
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    params: dict[str, float],
) -> None:
    """Store fit-problem parameter values on their model components.

    Globally shared (and constrained) parameters update the component's
    ``parameters`` directly. Per-dataset and grouped instances are stored per
    tie key under ``metadata["fitted_values"]`` because a single parameter
    box cannot display several values.
    """

    for component in components:
        for parameter in component_parameter_names(component):
            qualified = qualified_parameter_name(component.name, parameter)
            if qualified in params:
                component.parameters[parameter] = float(params[qualified])
                continue
            values = {
                instance.scope: float(params[instance.name])
                for instance in compiled.instances_for(component.name, parameter)
                if instance.name in params
            }
            if values:
                component.metadata.setdefault("fitted_values", {})[parameter] = values


def _write_back_fitted_parameters(
    group: DataGroup,
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    result: Any,
) -> None:
    """Store optimized values on their model components."""

    _write_back_parameter_values(group, components, compiled, result.params)


def _fit_channels_from_result(
    compiled: CompiledFitProblem,
    result: Any,
    bundles: dict[str, FitDataBundle],
) -> dict[str, dict[str, Any]]:
    """Evaluate fit and residual channels over each fitted dataset's view."""

    return _fit_channels_from_params(compiled, result.params, bundles)


def _fit_channels_from_params(
    compiled: CompiledFitProblem,
    params: dict[str, float],
    bundles: dict[str, FitDataBundle],
    subset_cache: dict[str, tuple[np.ndarray, PointData4D]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate fit and residual channels for a compiled problem.

    ``subset_cache`` optionally memoizes each dataset's valid-point mask and
    subset across calls; reusing the same subset object also lets the model's
    per-dataset geometry cache hit, so repeated overlay refreshes at new
    parameter values skip both the mask scan and the RPA geometry rebuild.
    """

    channels: dict[str, dict[str, Any]] = {}
    fitted_names = {dataset.name for dataset in compiled.problem.datasets}
    for name, bundle in bundles.items():
        if name not in fitted_names:
            continue
        points = bundle.points
        # Evaluate the model only where the data is valid (unmasked, finite,
        # positive sigma) and scatter back onto the full grid with NaN
        # elsewhere. Masked cells display as NaN regardless, so this avoids
        # running the model over the (often 10x larger) masked remainder and
        # over the extra unique Q those masked shells introduce.
        cached_subset = subset_cache.get(name) if subset_cache is not None else None
        if cached_subset is not None:
            keep, subset = cached_subset
        else:
            keep = points.valid_mask()
            subset = _subset_points(points, keep)
            if subset_cache is not None:
                subset_cache[name] = (keep, subset)
        fit_values = np.full(points.size, np.nan, dtype=float)
        if subset.size:
            fit_values[keep] = np.asarray(
                evaluate_problem_model(compiled.problem, name, params, data=subset),
                dtype=float,
            )
        intensity = np.asarray(points.intensity, dtype=float)
        sigma = np.asarray(points.sigma, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual_values = (intensity - fit_values) / sigma
        residual_values = np.where(
            np.isfinite(intensity) & np.isfinite(sigma) & (sigma > 0.0),
            residual_values,
            np.nan,
        )
        if bundle.grid_shape is not None:
            fit_values = fit_values.reshape(bundle.grid_shape)
            residual_values = residual_values.reshape(bundle.grid_shape)
        channels[name] = {
            "kind": "grid" if bundle.grid_shape is not None else "points",
            "fit": fit_values,
            "residual": residual_values,
        }
    return channels


def _subset_points(points: PointData4D, keep: np.ndarray) -> PointData4D:
    """Return the ``keep``-selected subset of ``points`` (order preserved)."""

    if isinstance(points.temperature, np.ndarray):
        temperature: Any = points.temperature[keep]
    else:
        temperature = points.temperature
    return PointData4D(
        H=points.H[keep],
        K=points.K[keep],
        L=points.L[keep],
        E=points.E[keep],
        intensity=points.intensity[keep],
        sigma=points.sigma[keep],
        mask=np.ones(int(np.count_nonzero(keep)), dtype=bool),
        temperature=temperature,
        metadata=dict(points.metadata),
    )


def latest_fit_channels(group: DataGroup, dataset_name: str) -> dict[str, Any] | None:
    """Return the stored fit channels for a dataset from the newest fit result."""

    latest: dict[str, Any] | None = None
    for entry in _walk_fit_entries(group.fits):
        if entry.kind != "result":
            continue
        payload = entry.channels.get(dataset_name)
        if payload:
            latest = payload
    return latest


def _group_has_fit_channels(group: DataGroup) -> bool:
    """Return whether any dataset in the group has stored fit channels."""

    return any(entry.channels for entry in _walk_fit_entries(group.fits) if entry.kind == "result")


def _group_has_enabled_model_components(group: DataGroup) -> bool:
    """Return whether the group has a model that can be evaluated."""

    return any(
        isinstance(model, ModelComponentSpec) and model.enabled
        for model in group.models.values()
    )


def attach_fit_channels_to_view(
    group: DataGroup,
    dataset_name: str,
    view: MDHistoData | PointListData,
    *,
    fallback_payload: dict[str, Any] | None = None,
) -> None:
    """Attach live-model or saved fit/residual channels to a viewer-ready dataset.

    Live current-model channels take precedence when available. Channels are
    only attached when their shape still matches the current view, so stale fits
    after mask or rebin changes are silently skipped rather than misaligned.
    """

    payload = fallback_payload or latest_fit_channels(group, dataset_name)
    if payload is None:
        return
    arrays: dict[str, np.ndarray] = {}
    for channel_name in FIT_CHANNEL_NAMES:
        decoded = _fit_channel_array(payload.get(channel_name))
        if decoded is None:
            return
        arrays[channel_name] = decoded
    if isinstance(view, MDHistoData):
        if any(array.shape != view.shape for array in arrays.values()):
            return
        for channel_name, array in arrays.items():
            view.metadata[channel_name] = array
        return
    if isinstance(view, PointListData):
        if any(array.shape != (view.size,) for array in arrays.values()):
            return
        existing = set(view.channel_labels)
        for channel_name, array in arrays.items():
            view.columns[channel_name] = array
            if channel_name not in existing:
                view.channels.append(
                    {"label": channel_name, "value": channel_name, "error": None}
                )


def _fit_channel_array(value: Any) -> np.ndarray | None:
    """Return a stored fit channel as a float array, decoding if needed."""

    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return _decode_float_array(value)
        except (KeyError, ValueError, TypeError, zlib.error):
            return None
    return np.asarray(value, dtype=float)


def _encode_float_array(value: Any) -> dict[str, Any]:
    """Encode a float array as compressed base64 for project JSON.

    Values are stored as float32: fit and residual channels are visualization
    aids, and halving the payload matters more than the last digits.
    """

    if isinstance(value, dict) and value.get("__ndarray__"):
        return value
    array = np.asarray(value, dtype=np.float32)
    return {
        "__ndarray__": True,
        "dtype": "float32",
        "shape": [int(size) for size in array.shape],
        "data": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii"),
    }


def _decode_float_array(payload: dict[str, Any]) -> np.ndarray:
    """Decode an array stored by :func:`_encode_float_array`."""

    raw = zlib.decompress(base64.b64decode(payload["data"]))
    array = np.frombuffer(raw, dtype=np.dtype(str(payload.get("dtype", "float32"))))
    return array.reshape([int(size) for size in payload["shape"]]).astype(float)


# Building a viewer view (loading, rebinning, and masking the full volume) is
# expensive on large datasets and is redone on every slice-viewer refresh --
# including when a fit result or "Current state" node is merely selected. The
# result depends only on the dataset's data, rebin config, and masks (not on the
# selection or model parameters), so it is cached and reused. The signature
# excludes the dataset scale factor, which is applied cheaply afterward.
_VIEWER_VIEW_CACHE: dict[int, tuple[str, Any]] = {}
_VIEWER_VIEW_CACHE_LIMIT = 8


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
        id(dataset.data),
        dataset.data_type,
        dataset.kind,
        rebin,
        _mask_signature(getattr(dataset, "masks", None)),
        _mask_signature(extra_masks),
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def _viewer_data_before_scale(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData | PointListData | None:
    key = id(dataset)
    signature = _viewer_view_signature(dataset, extra_masks)
    cached = _VIEWER_VIEW_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    result = _viewer_data_before_scale_uncached(dataset, extra_masks=extra_masks)
    if result is not None:
        # Recompute the signature: the uncached path may have lazily loaded the
        # data (changing id(dataset.data)), so key the entry on the loaded id.
        signature = _viewer_view_signature(dataset, extra_masks)
        if len(_VIEWER_VIEW_CACHE) >= _VIEWER_VIEW_CACHE_LIMIT:
            _VIEWER_VIEW_CACHE.clear()
        _VIEWER_VIEW_CACHE[key] = (signature, result)
    return result


def _viewer_data_before_scale_uncached(
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
        if dataset_rebin_enabled(dataset):
            return rebinned_dataset_data(dataset, extra_masks=extra_masks)
        return _mdhisto_with_nfit_masks(dataset, data=dataset.data, extra_masks=extra_masks)
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
    if dataset_rebin_enabled(dataset):
        return rebinned_dataset_data(dataset, extra_masks=extra_masks)
    return _mdhisto_with_nfit_masks(dataset, data=dataset.data, extra_masks=extra_masks)


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
    try:
        config["max_batch_mb"] = max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        config["max_batch_mb"] = DEFAULT_REBIN_MAX_BATCH_MB
    if config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "inverse_variance"
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


def rebinned_dataset_data(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> Any:
    """Return a rebinned copy of a supported dataset according to its configuration."""

    config = dataset_rebin_config(dataset)
    if isinstance(dataset.data, PointListData):
        return _rebin_point_list_data(dataset, config)
    if isinstance(dataset.data, PointData4D):
        return _rebin_point_data(_point_data_with_nfit_masks(dataset, dataset.data, extra_masks=extra_masks), config)
    if not isinstance(dataset.data, MDHistoData):
        return dataset.data
    masked = _mdhisto_with_nfit_masks(dataset, data=dataset.data, extra_masks=extra_masks)
    return _with_rebinned_mask_metadata(_rebin_mdhisto_data(masked, config), masked)


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
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
    )


def _rebin_mean_weighting(config: dict[str, Any]) -> str:
    value = config.get("mean_weighting")
    return str(value) if value in {"inverse_variance", "uniform"} else "inverse_variance"


def _rebin_max_batch_mb(config: dict[str, Any]) -> int:
    try:
        return max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        return DEFAULT_REBIN_MAX_BATCH_MB


def _rebin_max_batch_bytes(config: dict[str, Any]) -> int:
    return _rebin_max_batch_mb(config) * 1024 * 1024


def create_rebinned_dataset(
    group: DataGroup,
    dataset: DatasetEntry,
    *,
    name: str | None = None,
) -> DatasetEntry:
    """Materialize a dataset's rebinned view as an independent dataset."""

    data = rebinned_dataset_data(dataset, extra_masks=effective_dataset_masks(group, dataset))
    if data is dataset.data:
        data = copy.deepcopy(data)
    parameters = {}
    if "temperature" in dataset.parameters:
        parameters["temperature"] = copy.deepcopy(dataset.parameters["temperature"])
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
                    "vector": _mdhisto_rebin_axis_vector(axis, index, ndim),
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


def _mdhisto_rebin_axis_vector(axis: MDHistoAxis, index: int, ndim: int) -> list[float]:
    """Return a default rebin vector that follows the displayed MDHisto axis."""

    if ndim == 4:
        role = axis.role
        role_vectors = {
            "h": [1.0, 0.0, 0.0, 0.0],
            "k": [0.0, 1.0, 0.0, 0.0],
            "l": [0.0, 0.0, 1.0, 0.0],
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
        projected.append(_mdhisto_rebin_component(data, list(source_grids), axis_config, index))
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
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
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
        "mean_weighting": _rebin_mean_weighting(config),
        "max_batch_mb": _rebin_max_batch_mb(config),
        "max_batch_bytes": _rebin_max_batch_bytes(config),
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
        mean_weighting=_rebin_mean_weighting(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
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
        metadata=metadata,
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
    keep = np.ones(data.size, dtype=bool)
    coords = _point_data_coordinate_values(data)
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


def _point_data_energy_q_range_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.size, dtype=bool)
    energy = _parameter_range(parameters.get("energy"))
    q_modulus = _parameter_range(parameters.get("q_modulus"))
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
    center = _coordinate_axis_vector(parameters.get("center"), 3)
    if center is None:
        return np.zeros(data.size, dtype=bool)
    radius = max(_parameter_float(parameters.get("radius")) or 0.0, 0.0)
    q_vectors = _point_data_q_vectors(data)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        center_q = center
    else:
        center_q = _mdhisto_q_matrix(data.metadata) @ center
    cone_radius = np.abs(np.asarray(data.E, dtype=float)) / slope + radius
    distance = np.linalg.norm(q_vectors - center_q, axis=-1)
    return distance <= cone_radius


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
    center = _coordinate_axis_vector(parameters.get("center"), 3)
    if center is None:
        return np.zeros(data.shape, dtype=bool)
    coords = _mdhisto_coordinate_grids(data)
    if not {"H", "K", "L", "E"}.issubset(coords):
        return np.zeros(data.shape, dtype=bool)
    radius = _parameter_float(parameters.get("radius")) or 0.0
    radius = max(radius, 0.0)

    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        q_vectors = hkl
        center_q = center
    else:
        matrix = _mdhisto_q_matrix(data.metadata)
        q_vectors = np.einsum("ij,...j->...i", matrix, hkl)
        center_q = matrix @ center

    cone_radius = np.abs(coords["E"]) / slope + radius
    distance = np.linalg.norm(q_vectors - center_q, axis=-1)
    return distance <= cone_radius


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


def _fit_entry_path(
    fits: list[FitTimelineEntry],
    target: FitTimelineEntry | None,
) -> list[int] | None:
    """Return the list of child indices from the fit root to ``target``.

    Names are not unique across branches (every branch has a "Current state"),
    so a positional path is the stable identifier for persisting a selection.
    """

    if target is None:
        return None
    for index, entry in enumerate(fits):
        if entry is target:
            return [index]
        below = _fit_entry_path(entry.children, target)
        if below is not None:
            return [index, *below]
    return None


def _fit_entry_at_path(
    fits: list[FitTimelineEntry],
    path: list[int] | None,
) -> FitTimelineEntry | None:
    """Return the fit entry at a positional path, or ``None`` if it is gone."""

    if not path:
        return None
    entries = fits
    entry: FitTimelineEntry | None = None
    for index in path:
        if not (0 <= index < len(entries)):
            return None
        entry = entries[index]
        entries = entry.children
    return entry


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


def _ensure_current_state_after_result(
    group: DataGroup,
    result: FitTimelineEntry,
) -> tuple[FitTimelineEntry | None, bool]:
    """Return/create the mutable current state immediately after ``result``."""

    siblings = _fit_siblings(group.fits, result)
    if siblings is None or result not in siblings:
        return None, False
    existing = _current_state_after_result(group, result)
    if existing is not None:
        return existing, False
    current = current_state_fit_entry(group)
    siblings.insert(siblings.index(result) + 1, current)
    return current, True


def _current_state_for_failed_fit(
    group: DataGroup,
    parent: FitTimelineEntry,
    result: FitTimelineEntry,
) -> FitTimelineEntry | None:
    """Return/create the current state that should stay active after failure."""

    siblings = _fit_siblings(group.fits, result)
    if siblings is None:
        return parent if parent.kind == "current" and _fit_entry_in_tree(group.fits, parent) else None
    if parent.kind == "current" and parent in siblings:
        _set_fit_current_snapshot(parent, group)
        siblings.remove(parent)
        siblings.insert(siblings.index(result) + 1, parent)
        return parent
    if parent.kind in {"initial", "result"}:
        current, _created = _ensure_current_state_after_result(group, result)
        if current is not None:
            _set_fit_current_snapshot(current, group)
        return current
    return None


def _fit_entry_to_select_after_run(
    group: DataGroup,
    parent: FitTimelineEntry,
    result: FitTimelineEntry,
) -> FitTimelineEntry:
    if str(result.goodness.get("status", "")) == "failed":
        current = _current_state_after_result(group, result)
        if current is not None:
            return current
        if parent.kind == "current" and _fit_entry_in_tree(group.fits, parent):
            return parent
    return result


def _is_editable_initial_baseline(
    group: DataGroup,
    fit_entry: FitTimelineEntry | None,
) -> bool:
    """Return whether edits should update Initial instead of creating a branch."""

    if fit_entry is None or fit_entry.kind != "initial":
        return False
    if group.fits != [fit_entry]:
        return False
    return not fit_entry.children


def _should_branch_fit_now(group: DataGroup, parent: FitTimelineEntry) -> bool:
    if parent.kind in {"initial", "current"}:
        return False
    siblings = _fit_siblings(group.fits, parent)
    if siblings is None:
        return True
    return parent is not _last_result_at_level(siblings)


def _replace_current_state(entries: list[FitTimelineEntry], group: DataGroup) -> None:
    del group
    entries[:] = [entry for entry in entries if entry.kind != "current"]


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


def _style_active_fit_tree_item(item: Any) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont(item.font(0))
    font.setBold(True)
    font.setItalic(True)
    item.setFont(0, font)
    item.setForeground(0, QtGui.QBrush(QtGui.QColor("#62b884")))
    item.setToolTip(0, "Active fit state currently applied to the workspace.")


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


def save_project(project: NfitProject, path: str | Path) -> None:
    """Persist the GUI project as editable JSON."""

    Path(path).write_text(
        json.dumps(_project_to_dict(project), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_project(path: str | Path) -> NfitProject:
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


class _FitProgressDialog:
    """Small live progress window for optimizer and sampler runs."""

    def __init__(self, parent: Any) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        self.dialog = QtWidgets.QDialog(parent.window if hasattr(parent, "window") else parent)
        self.dialog.setWindowTitle("Fit progress")
        self.dialog.setModal(False)
        self.dialog.resize(720, 520)
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self.dialog)
        self.close_shortcut.activated.connect(self.dialog.close)
        layout = QtWidgets.QVBoxLayout(self.dialog)
        self.stage_label = QtWidgets.QLabel("Ready")
        stage_font = self.stage_label.font()
        stage_font.setBold(True)
        self.stage_label.setFont(stage_font)
        self.stage_label.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("No fit is running.")
        self.status_label.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.parameter_table = QtWidgets.QTableWidget(0, 2)
        self.parameter_table.setObjectName("fit_progress_parameter_table")
        self.parameter_table.setToolTip("Current parameter values reported by the active optimizer or sampler stage.")
        self.parameter_table.setHorizontalHeaderLabels(["Parameter", "Current value"])
        self.parameter_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameter_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.parameter_table.setAlternatingRowColors(True)
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.parameter_table.setMinimumHeight(120)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("fit_progress_log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setToolTip("Short live progress log. Current parameter values are shown in the table above.")
        self.panel_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.panel_splitter.setObjectName("fit_progress_panel_splitter")
        self.panel_splitter.setChildrenCollapsible(False)
        self.panel_splitter.addWidget(self.parameter_table)
        self.panel_splitter.addWidget(self.log)
        self.panel_splitter.setStretchFactor(0, 1)
        self.panel_splitter.setStretchFactor(1, 1)
        self.panel_splitter.setSizes([240, 220])
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("Request cancellation of the active fit or posterior sampler.")
        self.cancel_button.clicked.connect(self._cancel_requested)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.setToolTip("Close this progress window after the fit pipeline finishes.")
        self.close_button.clicked.connect(self.dialog.close)
        self._cancel_callback: Any | None = None
        layout.addWidget(self.stage_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.panel_splitter, 1)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def reset(self, title: str = "Starting fit pipeline...") -> None:
        from PySide6 import QtWidgets

        self.stage_label.setText(title)
        self.status_label.setStyleSheet("")
        self.status_label.setText("Preparing data and fit problem.")
        self.progress.setRange(0, 0)
        self.parameter_table.setRowCount(0)
        self.log.clear()
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self._cancel_callback = None
        QtWidgets.QApplication.processEvents()

    def set_cancel_callback(self, callback: Any | None) -> None:
        self._cancel_callback = callback
        self.cancel_button.setEnabled(callback is not None)

    def _cancel_requested(self) -> None:
        if self._cancel_callback is None:
            return
        self._cancel_callback()
        self.status_label.setText("Cancellation requested. Waiting for the active step to stop...")
        self.cancel_button.setEnabled(False)

    def show(self) -> None:
        from PySide6 import QtWidgets

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        QtWidgets.QApplication.processEvents()

    def update_progress(self, event: dict[str, Any]) -> None:
        from PySide6 import QtWidgets

        stage = str(event.get("stage", "fit"))
        iteration = event.get("iteration")
        total = event.get("total")
        message = str(event.get("message", stage))
        stage_title = {
            "initialization": "Initialization: differential evolution",
            "least_squares": "Least-squares fit",
            "emcee": "Posterior sampling: emcee",
        }.get(stage, stage.replace("_", " ").title())
        self.stage_label.setText(stage_title)
        status_parts: list[str] = []
        if iteration is not None:
            status_parts.append(f"Step {iteration}" + (f" of {total}" if total else ""))
        if event.get("cost") is not None:
            status_parts.append(f"cost {_format_number(float(event['cost']))}")
        if event.get("convergence") is not None:
            status_parts.append(f"convergence {_format_number(float(event['convergence']))}")
        if event.get("seconds_per_step") is not None:
            status_parts.append(f"{_format_seconds_per_step(float(event['seconds_per_step']))}/step")
        self.status_label.setText(" | ".join(status_parts) if status_parts else message)
        params = event.get("parameters")
        if isinstance(params, dict) and params:
            self._set_parameters(params)
        log_parts = [stage_title]
        if iteration is not None:
            log_parts.append(f"step {iteration}" + (f"/{total}" if total else ""))
        if event.get("cost") is not None:
            log_parts.append(f"cost {_format_number(float(event['cost']))}")
        if event.get("seconds_per_step") is not None:
            log_parts.append(f"{_format_seconds_per_step(float(event['seconds_per_step']))}/step")
        self.log.appendPlainText(" | ".join(log_parts))
        if total and iteration is not None:
            self.progress.setRange(0, int(total))
            self.progress.setValue(min(int(iteration), int(total)))
        else:
            self.progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

    def _set_parameters(self, params: dict[str, Any]) -> None:
        from PySide6 import QtCore, QtWidgets

        items = list(params.items())
        self.parameter_table.setRowCount(len(items))
        for row, (name, value) in enumerate(items):
            name_item = QtWidgets.QTableWidgetItem(str(name))
            value_text = _format_number(value)
            value_item = QtWidgets.QTableWidgetItem(value_text)
            name_item.setToolTip(str(name))
            value_item.setToolTip(value_text)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.parameter_table.setItem(row, 0, name_item)
            self.parameter_table.setItem(row, 1, value_item)
        self.parameter_table.resizeColumnsToContents()

    def finish(self, message: str) -> None:
        from PySide6 import QtWidgets

        self.stage_label.setText(message)
        self.status_label.setText("Done.")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.cancel_button.setEnabled(False)
        self._cancel_callback = None
        self.close_button.setEnabled(True)
        QtWidgets.QApplication.processEvents()

    def fail(self, message: str) -> None:
        """Show a fit failure and keep the window open for the user to read."""

        from PySide6 import QtWidgets

        self.stage_label.setText("Fit failed")
        self.status_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        self.status_label.setText(message)
        self.log.appendPlainText(f"ERROR: {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_button.setEnabled(False)
        self._cancel_callback = None
        self.close_button.setEnabled(True)
        self.show()
        QtWidgets.QApplication.processEvents()

    def close(self) -> None:
        self.dialog.close()


class _FitDiagnosticsPlotWindow:
    """Dedicated Matplotlib window for fit covariance and posterior diagnostics."""

    def __init__(self, fit_entry: FitTimelineEntry, parent: Any) -> None:
        from PySide6 import QtGui, QtWidgets

        self.fit_entry = fit_entry
        self.window = QtWidgets.QMainWindow(parent.window if hasattr(parent, "window") else parent)
        self.window.setWindowTitle("Fit diagnostics")
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self.window)
        self.close_shortcut.activated.connect(self.window.close)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setToolTip("Fit diagnostics from covariance estimates and stored emcee samples.")
        layout.addWidget(self.tabs, 1)
        self.label_table = self._make_label_table()
        layout.addWidget(self.label_table)
        self.window.setCentralWidget(central)
        self._redraw_plots()

    def _make_label_table(self) -> Any:
        from PySide6 import QtCore, QtWidgets

        names = _fit_entry_diagnostic_parameter_names(self.fit_entry)
        labels = _fit_parameter_plot_labels(self.fit_entry, names)
        table = QtWidgets.QTableWidget(len(names), 2)
        table.setObjectName("fit_diagnostics_label_table")
        table.setToolTip(
            "Edit plot labels for this fit result. Plain text or Matplotlib mathtext/LaTeX-style labels are accepted."
        )
        table.setHorizontalHeaderLabels(["Full parameter name", "Plot label"])
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(names):
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            label_item = QtWidgets.QTableWidgetItem(labels.get(name, f"p{row + 1}"))
            name_item.setToolTip(name)
            label_item.setToolTip("Editable plot label. Use p1-style names, plain text, or mathtext such as $\\Gamma$.")
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, label_item)
        table.itemChanged.connect(self._label_table_changed)
        table.setMaximumHeight(150)
        table.resizeColumnsToContents()
        _tooltip_table_corner_buttons(table, "Select all parameter-label rows.")
        return table

    def _label_table_changed(self, item: Any) -> None:
        if item.column() != 1:
            return
        labels = dict(self.fit_entry.metadata.get("parameter_labels", {}))
        full_name_item = self.label_table.item(item.row(), 0)
        if full_name_item is None:
            return
        full_name = full_name_item.text()
        labels[full_name] = item.text().strip() or _compact_diagnostic_labels([full_name])[0]
        self.fit_entry.metadata["parameter_labels"] = labels
        self._redraw_plots()

    def _redraw_plots(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        active_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        self.tabs.clear()

        covariance = _covariance_matrix_from_fit_entry(self.fit_entry)
        if covariance is not None:
            matrix, names, title = covariance
            labels = _fit_parameter_plot_labels(self.fit_entry, names)
            cov_fig = Figure(figsize=(max(7, 1.0 * len(names) + 4), max(5, 0.75 * len(names) + 3)))
            _draw_centered_matrix_heatmap(cov_fig, matrix, names, labels=labels, title=title)
            self.tabs.addTab(FigureCanvas(cov_fig), title)

        result = _sampling_result_from_dict(self.fit_entry.metadata.get("posterior_samples"))
        if result is None:
            self._restore_active_tab(active_tab)
            return

        samples = np.asarray(result.samples, dtype=float)
        names = list(result.variable_names)
        labels = _fit_parameter_plot_labels(self.fit_entry, names)
        summaries = _fit_parameter_summaries(self.fit_entry)

        trace_fig = Figure(figsize=(8, max(3, 1.4 * max(1, len(names)))))
        trace_axes = trace_fig.subplots(max(1, len(names)), 1, squeeze=False)
        for index, name in enumerate(names):
            ax = trace_axes[index, 0]
            _draw_trace_panel(
                ax,
                result,
                parameter_index=index,
                summary=summaries.get(name, {}),
            )
            ax.set_ylabel(labels.get(name, f"p{index + 1}"))
        trace_axes[-1, 0].set_xlabel("MCMC step" if result.chain is not None else "sample")
        trace_fig.subplots_adjust(left=0.18, right=0.98, bottom=0.1, top=0.95, hspace=0.28)
        self.tabs.addTab(FigureCanvas(trace_fig), "Trace")

        n = len(names)
        corner_fig = Figure(figsize=(max(6, 2.45 * max(1, n)), max(6, 2.45 * max(1, n))))
        axes = corner_fig.subplots(max(1, n), max(1, n), squeeze=False)
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if row == col:
                    _draw_corner_histogram_panel(
                        ax,
                        samples[:, col],
                        labels.get(names[col], f"p{col + 1}"),
                        summaries.get(names[col], {}),
                    )
                elif row > col:
                    _draw_corner_density_panel(ax, samples[:, col], samples[:, row])
                    _draw_corner_reference_lines(
                        ax,
                        summaries.get(names[col], {}),
                        summaries.get(names[row], {}),
                    )
                else:
                    ax.axis("off")
                _disable_axis_offset_text(ax)
                if row == n - 1:
                    ax.set_xlabel(labels.get(names[col], f"p{col + 1}"))
                if col == 0 and row > 0:
                    ax.set_ylabel(labels.get(names[row], f"p{row + 1}"))
                elif row == col and col == 0:
                    ax.set_ylabel("Count")
        corner_fig.subplots_adjust(
            left=0.18,
            right=0.98,
            bottom=0.18,
            top=0.9,
            hspace=0.48,
            wspace=0.48,
        )
        self.tabs.addTab(FigureCanvas(corner_fig), "Corner")
        self._restore_active_tab(active_tab)

    def _restore_active_tab(self, tab_text: str) -> None:
        if not tab_text:
            return
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == tab_text:
                self.tabs.setCurrentIndex(index)
                return

    def show(self) -> None:
        self.window.resize(900, 700)
        self.window.show()


def _draw_matrix_heatmap(
    ax: Any,
    matrix: Any,
    names: list[str],
    *,
    labels: dict[str, str] | None = None,
    title: str,
    colorbar_ax: Any | None = None,
) -> None:
    """Draw a labeled covariance or correlation heatmap."""

    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return
    plot_labels = labels or {name: label for name, label in zip(names, _compact_diagnostic_labels(names))}
    finite = arr[np.isfinite(arr)]
    if title.lower().startswith("correlation"):
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
    elif finite.size:
        limit = float(np.nanmax(np.abs(finite)))
        vmin, vmax = (-limit, limit) if limit > 0 else (-1.0, 1.0)
        cmap = "coolwarm"
    else:
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
    image = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels([plot_labels.get(name, f"p{index + 1}") for index, name in enumerate(names)], rotation=0)
    ax.set_yticklabels([plot_labels.get(name, f"p{index + 1}") for index, name in enumerate(names)])
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            value = arr[row, col]
            if np.isfinite(value):
                ax.text(col, row, _format_number(value), ha="center", va="center", fontsize=7)
    if colorbar_ax is None:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.figure.colorbar(image, cax=colorbar_ax)


def _draw_centered_matrix_heatmap(
    fig: Any,
    matrix: Any,
    names: list[str],
    *,
    labels: dict[str, str] | None = None,
    title: str,
) -> Any:
    """Draw the covariance/correlation matrix centered in the diagnostics tab."""

    arr = np.asarray(matrix, dtype=float)
    label_count = max(1, len(names))
    square_size = min(0.66, max(0.42, 0.10 * label_count + 0.34))
    colorbar_width = 0.028
    gap = 0.025
    total_width = square_size + gap + colorbar_width
    left = max(0.08, (1.0 - total_width) / 2.0)
    bottom = max(0.14, (1.0 - square_size) / 2.0)
    ax = fig.add_axes([left, bottom, square_size, square_size])
    colorbar_ax = fig.add_axes([left + square_size + gap, bottom, colorbar_width, square_size])
    _draw_matrix_heatmap(ax, arr, names, labels=labels, title=title, colorbar_ax=colorbar_ax)
    return ax


def _compact_diagnostic_labels(names: list[str]) -> list[str]:
    return [f"p{index + 1}" for index in range(len(names))]


def _draw_trace_panel(
    ax: Any,
    result: SamplingResult,
    *,
    parameter_index: int,
    summary: dict[str, float],
) -> None:
    """Draw one posterior trace panel, preferring raw walker chains."""

    if result.chain is not None:
        chain = np.asarray(result.chain, dtype=float)
        if chain.ndim == 3 and parameter_index < chain.shape[2]:
            steps = np.arange(chain.shape[0])
            ax.plot(steps, chain[:, :, parameter_index], linewidth=0.55, alpha=0.75)
            burn_in = int(result.metadata.get("burn_in", 0) or 0)
            if 0 < burn_in < chain.shape[0]:
                ax.axvline(
                    burn_in,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                )
                top = ax.get_ylim()[1]
                ax.annotate(
                    "burn-in",
                    xy=(burn_in, top),
                    xytext=(4, -4),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="black",
                )
            if summary.get("best") is not None:
                ax.axhline(float(summary["best"]), color="red", linewidth=0.9, alpha=0.8)
            return

    samples = np.asarray(result.samples, dtype=float)
    if samples.ndim == 2 and parameter_index < samples.shape[1]:
        ax.plot(samples[:, parameter_index], linewidth=0.7)
    if summary.get("best") is not None:
        ax.axhline(float(summary["best"]), color="red", linewidth=0.9, alpha=0.8)


def _fit_entry_diagnostic_parameter_names(fit_entry: FitTimelineEntry) -> list[str]:
    names: list[str] = []
    covariance = _covariance_matrix_from_fit_entry(fit_entry)
    if covariance is not None:
        _matrix, cov_names, _title = covariance
        names.extend(cov_names)
    samples = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if samples is not None:
        names.extend(samples.variable_names)
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = goodness.get("parameters") if isinstance(goodness.get("parameters"), dict) else {}
    names.extend(str(name) for name in params)
    return list(dict.fromkeys(names))


def _fit_parameter_plot_labels(
    fit_entry: FitTimelineEntry,
    names: list[str],
) -> dict[str, str]:
    stored = fit_entry.metadata.get("parameter_labels")
    labels = dict(stored) if isinstance(stored, dict) else {}
    out: dict[str, str] = {}
    defaults_changed = False
    for index, name in enumerate(names):
        label = str(labels.get(name, "")).strip()
        if not label:
            label = f"p{index + 1}"
            labels[name] = label
            defaults_changed = True
        out[name] = label
    if defaults_changed:
        fit_entry.metadata["parameter_labels"] = labels
    return out


def _fit_parameter_summaries(fit_entry: FitTimelineEntry) -> dict[str, dict[str, float]]:
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = goodness.get("parameters") if isinstance(goodness.get("parameters"), dict) else {}
    stderr = goodness.get("stderr") if isinstance(goodness.get("stderr"), dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    posterior_params = (
        posterior.get("parameters")
        if isinstance(posterior.get("parameters"), dict)
        else {}
    )
    names = set(params) | set(stderr) | set(posterior_params)
    summaries: dict[str, dict[str, float]] = {}
    for name in names:
        summary: dict[str, float] = {}
        if name in params:
            summary["best"] = float(params[name])
        if name in stderr:
            err = float(stderr[name])
            summary["stderr"] = err
            if "best" in summary:
                summary.setdefault("low", summary["best"] - err)
                summary.setdefault("high", summary["best"] + err)
        posterior_row = posterior_params.get(name)
        if isinstance(posterior_row, dict):
            for source, target in (("median", "median"), ("p16", "low"), ("p84", "high")):
                if source in posterior_row:
                    summary[target] = float(posterior_row[source])
        summaries[str(name)] = summary
    return summaries


def _draw_corner_histogram_panel(
    ax: Any,
    values: Any,
    name: str,
    summary: dict[str, float],
) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    ax.hist(arr, bins=40, histtype="step", color="0.1", linewidth=1.1)
    center = summary.get("best", summary.get("median"))
    low = summary.get("low")
    high = summary.get("high")
    if center is not None:
        ax.axvline(center, color="red", linewidth=1.0)
    if low is not None:
        ax.axvline(low, color="0.25", linestyle="--", linewidth=0.8)
    if high is not None:
        ax.axvline(high, color="0.25", linestyle="--", linewidth=0.8)
    ax.set_title(_corner_histogram_title(name, summary), fontsize=9, pad=8)


def _draw_corner_reference_lines(
    ax: Any,
    x_summary: dict[str, float],
    y_summary: dict[str, float],
) -> None:
    x_center = x_summary.get("best", x_summary.get("median"))
    y_center = y_summary.get("best", y_summary.get("median"))
    if x_center is not None:
        ax.axvline(x_center, color="red", linewidth=0.8, alpha=0.85)
    if y_center is not None:
        ax.axhline(y_center, color="red", linewidth=0.8, alpha=0.85)
    if x_center is not None and y_center is not None:
        ax.plot([x_center], [y_center], marker="o", color="red", markersize=3)


def _corner_histogram_title(name: str, summary: dict[str, float]) -> str:
    center = summary.get("best", summary.get("median"))
    if center is None:
        return name
    low = summary.get("low")
    high = summary.get("high")
    if low is None or high is None:
        return rf"${_mathtext_label(name)} = {_format_number(center)}$"
    plus = high - center
    minus = center - low
    return (
        rf"${_mathtext_label(name)} = {_format_number(center)}"
        rf"\,\pm^{{+{_format_number(plus)}}}_{{-{_format_number(minus)}}}$"
    )


def _mathtext_label(label: str) -> str:
    stripped = str(label).strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1]
    if re.search(r"[\\{}_^]", stripped):
        return stripped
    return stripped.replace(" ", r"\ ")


def _disable_axis_offset_text(ax: Any) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)
        if hasattr(formatter, "set_scientific"):
            formatter.set_scientific(False)


def _covariance_matrix_from_fit_entry(
    fit_entry: FitTimelineEntry,
) -> tuple[np.ndarray, list[str], str] | None:
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    covariance = goodness.get("covariance") if isinstance(goodness.get("covariance"), dict) else {}
    names = [str(name) for name in covariance.get("variables", [])]
    matrix = covariance.get("matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if len(names) != arr.shape[0]:
                names = [f"p{index}" for index in range(arr.shape[0])]
            return arr, names, "Covariance"
    correlation = covariance.get("correlation")
    if isinstance(correlation, dict) and correlation:
        names = names or [str(name) for name in correlation]
        arr = np.asarray(
            [
                [float(dict(correlation.get(row_name, {})).get(col_name, np.nan)) for col_name in names]
                for row_name in names
            ],
            dtype=float,
        )
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr, names, "Correlation"
    return None


def _fit_entry_has_diagnostic_plots(fit_entry: FitTimelineEntry | None) -> bool:
    if fit_entry is None:
        return False
    if _covariance_matrix_from_fit_entry(fit_entry) is not None:
        return True
    return _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples")) is not None


def _draw_corner_density_panel(ax: Any, x: Any, y: Any) -> None:
    """Draw a 2D posterior panel with density shading, contours, and samples."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size == 0 or np.ptp(x_arr) == 0.0 or np.ptp(y_arr) == 0.0:
        ax.scatter(x_arr, y_arr, s=2, alpha=0.25, color="tab:blue", linewidths=0)
        return

    counts, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=48)
    counts = counts.T
    if np.any(counts > 0):
        positive = counts[counts > 0]
        image = ax.imshow(
            counts,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            cmap="Blues",
            alpha=0.6,
            interpolation="nearest",
        )
        del image
        if positive.size >= 4:
            levels = np.percentile(positive, [50.0, 75.0, 90.0])
            levels = np.unique(levels[levels > 0])
            if levels.size:
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                ax.contour(
                    x_centers,
                    y_centers,
                    counts,
                    levels=levels,
                    colors="0.15",
                    linewidths=0.8,
                    alpha=0.85,
                )
    ax.scatter(x_arr, y_arr, s=1.4, alpha=0.12, color="tab:blue", linewidths=0)


class NfitProjectExplorer:
    """PySide6 project explorer for building nfit analysis pipelines."""

    def __init__(self, project: NfitProject | None = None) -> None:
        self.app = _qt_app()
        self.project = NfitProject() if project is None else project
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
        self.dataset_temperature_spin = None
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
        self.model_parameter_scroll = None
        self.model_parameter_widget = None
        self.model_parameter_layout = None
        self.fit_editor_widget = None
        self.fit_settings_panel = None
        self.fit_optimizer_combo = None
        self.fit_optimizer_config_editor = None
        self.fit_loss_combo = None
        self.fit_f_scale_spin = None
        self.fit_de_check = None
        self.fit_de_maxiter_spin = None
        self.fit_de_popsize_spin = None
        self.fit_de_workers_spin = None
        self.fit_emcee_check = None
        self.fit_emcee_walkers_spin = None
        self.fit_emcee_steps_spin = None
        self.fit_emcee_burn_spin = None
        self.fit_emcee_thin_spin = None
        self.fit_emcee_workers_spin = None
        self.fit_branch_check = None
        self.fit_now_button = None
        self.fit_corner_button = None
        self.show_data_fit_button = None
        self._fit_progress_dialog: _FitProgressDialog | None = None
        self._fit_worker_thread = None
        self._fit_worker = None
        self._fit_worker_handler = None
        self._fit_disabled_widget_states: list[tuple[Any, bool]] = []
        self.expand_all_button = None
        self.collapse_all_button = None
        self.create_group_button = None
        self.delete_button = None
        self._clipboard: tuple[str, DatasetEntry | MaskSpec] | None = None
        self._slice_viewers: dict[int, Any] = {}
        self._overlay_refresh_timer = None
        self._pending_overlay_groups: dict[int, DataGroup] = {}
        # True only while the Qt event loop is running (set in run()); in
        # headless/test use it stays False so overlay refreshes are synchronous.
        self._interactive = False
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

    def show(self) -> "NfitProjectExplorer":
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self

    def run(self) -> int:
        self.show()
        self._interactive = True
        interrupt_timer, previous_interrupt_handler = _install_cli_interrupt_handler(self.app)
        try:
            return int(self.app.exec())
        except KeyboardInterrupt:
            self.app.exit(130)
            return 130
        finally:
            if interrupt_timer is not None:
                interrupt_timer.stop()
            _restore_cli_interrupt_handler(previous_interrupt_handler)

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
        self.project = NfitProject()
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

    def _stamp_active_fit_path(self) -> None:
        """Refresh the active group's stored fit path before persisting."""

        group = self._active_fit_group
        if group is None or group not in self.project.data_groups:
            return
        group.active_fit_path = _fit_entry_path(group.fits, self._active_fit_entry(group))

    def save(self) -> bool:
        if self.project_path is None:
            return self.save_as()
        self._stamp_active_fit_path()
        save_project(self.project, self.project_path)
        self.has_unsaved_changes = False
        self._sync_window_title()
        return True

    def save_as(self) -> bool:
        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save nfit project",
            "nfit_project.nfit",
            "nfit projects (*.nfit);;All files (*)",
        )
        if not path:
            return False
        self.project_path = Path(path)
        self._stamp_active_fit_path()
        save_project(self.project, self.project_path)
        self._remember_recent_project(self.project_path)
        self.has_unsaved_changes = False
        self._sync_window_title()
        return True

    def open_project(self) -> bool:
        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Open nfit project",
            "",
            "nfit projects (*.nfit);;All files (*)",
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
        self._restore_active_fit_selection()
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

    def save_rebin_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return False
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save rebinned dataset",
            f"{entry.name} rebinned.npz",
            "NumPy archives (*.npz);;All files (*)",
        )
        if not path:
            return False
        try:
            data = rebinned_dataset_data(entry, extra_masks=effective_dataset_masks(group, entry))
            rebinned_entry = DatasetEntry(
                name=f"{entry.name} rebinned",
                data=data,
                kind=entry.kind,
                metadata={**copy.deepcopy(entry.metadata), "source_dataset": entry.name, "rebin_saved": True},
                parameters=copy.deepcopy(entry.parameters),
                masks=copy.deepcopy(entry.masks),
            )
            save_dataset_file(rebinned_entry, path, use_view=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Save rebinned dataset",
                f"Could not save rebinned dataset:\n{exc}",
            )
            return False
        return True

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
        progress = self._fit_progress_dialog
        if progress is None:
            progress = _FitProgressDialog(self)
            self._fit_progress_dialog = progress
        progress.reset("Starting fit pipeline...")
        progress.show()
        result = run_group_fit(
            group,
            fit_entry,
            branch_timeline=should_branch,
            progress_callback=progress.update_progress,
        )
        failed = str(result.goodness.get("status", "")) == "failed"
        if failed:
            progress.fail(str(result.goodness.get("message", "The fit did not run.")))
        else:
            progress.finish("Fit pipeline finished.")
        self.fit_branch_check.setChecked(False)
        self.refresh_slice_viewer(group)
        item_to_select = _fit_entry_to_select_after_run(group, fit_entry, result)
        self._set_active_fit_state(group, item_to_select)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_fit=item_to_select)
        return result

    def _start_background_task(
        self,
        *,
        title: str,
        failure_title: str,
        task: Any,
        on_success: Any,
        success_message: str,
        close_on_success: bool = True,
    ) -> bool:
        from PySide6 import QtCore, QtWidgets

        thread = self._fit_worker_thread
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self.window,
                failure_title,
                "Another fit or posterior sampler is already running.",
            )
            return False

        class Worker(QtCore.QObject):
            progress = QtCore.Signal(dict)
            finished = QtCore.Signal(object)
            failed = QtCore.Signal(str)

            def __init__(self) -> None:
                super().__init__()
                self.cancel_requested = False

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    def progress_callback(event: dict[str, Any]) -> None:
                        if self.cancel_requested:
                            raise RuntimeError("Operation cancelled by user.")
                        self.progress.emit(event)

                    self.finished.emit(task(progress_callback))
                except Exception as exc:
                    self.failed.emit(str(exc))

            def cancel(self) -> None:
                self.cancel_requested = True

        class Handler(QtCore.QObject):
            @QtCore.Slot(dict)
            def handle_progress(self, event: dict[str, Any]) -> None:
                progress.update_progress(event)

            @QtCore.Slot(object)
            def handle_success(self, result: Any) -> None:
                should_finish = on_success(result)
                if should_finish is not False:
                    progress.finish(success_message)
                    if close_on_success:
                        progress.close()
                worker_thread.quit()

            @QtCore.Slot(str)
            def handle_failure(self, message: str) -> None:
                progress.fail(message)
                QtWidgets.QMessageBox.warning(self.window, failure_title, message)
                worker_thread.quit()

        progress = self._fit_progress_dialog
        if progress is None:
            progress = _FitProgressDialog(self)
            self._fit_progress_dialog = progress
        progress.reset(title)
        progress.show()

        worker_thread = QtCore.QThread(self.window)
        worker = Worker()
        handler = Handler(self.window)
        worker.moveToThread(worker_thread)
        self._fit_worker_thread = worker_thread
        self._fit_worker = worker
        self._fit_worker_handler = handler
        self._fit_disabled_widget_states = []
        for widget in (
            self.tree,
            self.fit_editor_widget,
            self.details_scroll,
            self.fit_now_button,
            self.fit_corner_button,
            self.show_data_fit_button,
            self.delete_button,
        ):
            if widget is not None:
                self._fit_disabled_widget_states.append((widget, widget.isEnabled()))
                widget.setEnabled(False)
        progress.set_cancel_callback(worker.cancel)
        worker_thread.started.connect(worker.run)
        worker.progress.connect(handler.handle_progress)

        def cleanup() -> None:
            progress.set_cancel_callback(None)
            for widget, enabled in self._fit_disabled_widget_states:
                widget.setEnabled(enabled)
            self._fit_disabled_widget_states = []
            self._fit_worker_thread = None
            self._fit_worker = None
            self._fit_worker_handler = None

        worker.finished.connect(handler.handle_success)
        worker.failed.connect(handler.handle_failure)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker_thread.finished.connect(cleanup)
        worker_thread.finished.connect(worker_thread.deleteLater)
        worker_thread.start()
        return True

    def start_fit_for_selection(self) -> bool:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return False
        self._set_selected_fit_optimizer_config()
        should_branch = bool(self.fit_branch_check.isChecked()) or _should_branch_fit_now(group, fit_entry)

        def task(progress_callback: Any) -> FitTimelineEntry:
            return run_group_fit(
                group,
                fit_entry,
                branch_timeline=should_branch,
                progress_callback=progress_callback,
            )

        def on_success(result: FitTimelineEntry) -> None:
            failed = str(result.goodness.get("status", "")) == "failed"
            if failed:
                progress = self._fit_progress_dialog
                if progress is not None:
                    progress.fail(str(result.goodness.get("message", "The fit did not run.")))
            self.fit_branch_check.setChecked(False)
            self.refresh_slice_viewer(group)
            item_to_select = _fit_entry_to_select_after_run(group, fit_entry, result)
            self._set_active_fit_state(group, item_to_select)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_fit=item_to_select)
            return not failed

        return self._start_background_task(
            title="Starting fit pipeline...",
            failure_title="Fit now",
            task=task,
            on_success=on_success,
            success_message="Fit pipeline finished.",
            close_on_success=False,
        )

    def open_fit_diagnostics_plots_for_selection(self) -> Any | None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return None
        if not _fit_entry_has_diagnostic_plots(fit_entry):
            return None
        window = _FitDiagnosticsPlotWindow(fit_entry, self)
        window.show()
        self._slice_viewers[id(window)] = window
        return window

    def apply_posterior_sampling_window(
        self,
        fit_entry: FitTimelineEntry,
        burn_in: int,
        thin: int,
    ) -> bool:
        from PySide6 import QtWidgets

        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        if stored is None or stored.chain is None:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain a raw emcee chain to re-window.",
            )
            return False
        try:
            updated = _sampling_result_with_window(stored, burn_in=burn_in, thin=thin)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not update the posterior sampling window:\n{exc}",
            )
            return False
        _store_sampling_result_on_fit_entry(fit_entry, updated)
        self._mark_dirty()
        self._set_fit_details(fit_entry)
        return True

    def promote_best_posterior_sample_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
    ) -> bool:
        from PySide6 import QtWidgets

        try:
            candidate = _best_posterior_promotion_candidate(fit_entry)
            if candidate is None and "chi2" not in fit_entry.goodness:
                compiled = _compiled_problem_for_fit_entry(group, fit_entry)
                candidate = _best_posterior_promotion_candidate(fit_entry, compiled)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not inspect the posterior samples:\n{exc}",
            )
            return False
        if candidate is None:
            QtWidgets.QMessageBox.information(
                self.window,
                "Posterior sampler",
                "The stored emcee chain does not contain a finite sample with a better likelihood than this fit result.",
            )
            return False
        sample_params, sample_log_probability, baseline_log_probability, location = candidate
        current_snapshot = snapshot_data_group_state(group)
        self._set_active_fit_state(group, fit_entry)
        try:
            if fit_entry.snapshot:
                restore_data_group_state(group, fit_entry.snapshot)
            components = [
                model for model in group.models.values() if isinstance(model, ModelComponentSpec)
            ]
            inputs, _bundles = fit_dataset_inputs(group)
            compiled_for_state = compile_fit_problem(components, inputs, description=group.name)
            promoted_params = {
                spec.name: float(spec.value)
                for spec in compiled_for_state.problem.parameter_specs
            }
            promoted_params.update(sample_params)
            promoted_params = compiled_for_state.problem.resolve_parameters(promoted_params)
            _write_back_parameter_values(
                group,
                components,
                compiled_for_state,
                promoted_params,
            )
        except Exception as exc:
            restore_data_group_state(group, current_snapshot)
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not promote the posterior sample:\n{exc}",
            )
            return False
        branch_created = self._record_data_group_state_change(group)
        promoted = self._active_fit_entry(group)
        if promoted is not None and promoted.kind == "current":
            promoted.metadata.setdefault("promoted_posterior_sample", {})
            promoted.metadata["promoted_posterior_sample"] = {
                "source_fit": fit_entry.name,
                "log_probability": float(sample_log_probability),
                "previous_log_probability": float(baseline_log_probability),
                **location,
            }
        self._mark_dirty()
        self.refresh_slice_viewer(group)
        if branch_created:
            self._refresh_tree(select_group=group, select_fit=promoted)
        else:
            self._refresh_tree(select_group=group, select_fit=promoted or fit_entry)
        return True

    def run_posterior_sampler_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int = 1,
        append: bool = False,
    ) -> bool:
        from PySide6 import QtWidgets

        if not isinstance(fit_entry.goodness.get("parameters"), dict):
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain best-fit parameters to start emcee.",
            )
            return False
        progress = self._fit_progress_dialog
        if progress is None:
            progress = _FitProgressDialog(self)
            self._fit_progress_dialog = progress
        progress.reset("Starting emcee posterior sampler...")
        progress.show()
        try:
            result = self._posterior_sampler_result_for_fit(
                group,
                fit_entry,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed,
                workers=workers,
                append=append,
                progress_callback=progress.update_progress,
            )
        except Exception as exc:
            progress.fail(str(exc))
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not run emcee posterior sampling:\n{exc}",
            )
            return False
        _store_sampling_result_on_fit_entry(fit_entry, result)
        self._mark_dirty()
        progress.finish("emcee posterior sampling finished.")
        progress.close()
        self._set_fit_details(fit_entry)
        return True

    def start_posterior_sampler_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int = 1,
        append: bool = False,
    ) -> bool:
        from PySide6 import QtWidgets

        if not isinstance(fit_entry.goodness.get("parameters"), dict):
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain best-fit parameters to start emcee.",
            )
            return False

        def task(progress_callback: Any) -> SamplingResult:
            return self._posterior_sampler_result_for_fit(
                group,
                fit_entry,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed,
                workers=workers,
                append=append,
                progress_callback=progress_callback,
            )

        def on_success(result: SamplingResult) -> bool:
            _store_sampling_result_on_fit_entry(fit_entry, result)
            self._mark_dirty()
            self._set_fit_details(fit_entry)
            return True

        return self._start_background_task(
            title="Starting emcee posterior sampler...",
            failure_title="Posterior sampler",
            task=task,
            on_success=on_success,
            success_message="emcee posterior sampling finished.",
        )

    def _posterior_sampler_result_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int,
        append: bool,
        progress_callback: Any | None,
    ) -> SamplingResult:
        compiled = _compiled_problem_for_fit_entry(group, fit_entry)
        initial_params = {
            str(name): float(value)
            for name, value in dict(fit_entry.goodness.get("parameters", {})).items()
        }
        initial_walkers = None
        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        walkers = None if n_walkers <= 0 else int(n_walkers)
        if append:
            if stored is None or stored.chain is None:
                raise ValueError("append requires an existing raw emcee chain")
            compiled_names = [
                spec.name for spec in compiled.problem.parameter_specs if spec.vary
            ]
            if list(stored.variable_names) != compiled_names:
                raise ValueError(
                    "stored posterior parameter order no longer matches the fit problem"
                )
            chain = np.asarray(stored.chain, dtype=float)
            if chain.ndim != 3 or chain.shape[0] == 0:
                raise ValueError("stored raw emcee chain is empty or malformed")
            initial_walkers = np.asarray(chain[-1], dtype=float)
            walkers = int(initial_walkers.shape[0])
        config = SamplerConfig(
            n_walkers=walkers,
            n_steps=int(n_steps),
            burn_in=int(burn_in),
            thin=max(1, int(thin)),
            random_seed=random_seed,
            kwargs={"workers": int(workers)},
        )
        sampled = sample_problem_parameters(
            compiled.problem,
            config,
            initial_params=initial_params if initial_walkers is None else None,
            initial_walkers=initial_walkers,
            progress_callback=progress_callback,
        )
        return (
            _combined_sampling_result(stored, sampled, burn_in=burn_in, thin=thin)
            if append and stored is not None
            else sampled
        )

    def show_data_and_fit_for_selection(self) -> Any | None:
        """Open the data viewer with the model overlay enabled."""

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        if not (_group_has_fit_channels(group) or _group_has_enabled_model_components(group)):
            return None
        viewer = self.open_slice_viewer(group)
        if viewer is None:
            return None
        check = getattr(viewer, "show_fit_check", None)
        if check is not None and check.isEnabled():
            check.setChecked(True)
        residual_check = getattr(viewer, "show_residual_check", None)
        if residual_check is not None:
            residual_check.setChecked(False)
        return viewer

    def open_slice_viewer_for_selection(self) -> Any | None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        allowed = {"group", "datasets", "dataset", "masks", "mask", "dataset_group", "group_masks", "group_mask"}
        if role not in allowed or group is None:
            return None
        # For a mask or the Masks node, entry is the owning dataset.
        selected_name = entry.name if role in {"dataset", "masks", "mask"} and entry is not None else None
        use_composite = selected_name is None
        return self.open_slice_viewer(group, selected_dataset_name=selected_name, use_composite=use_composite)

    def open_slice_viewer(
        self,
        group: DataGroup,
        *,
        selected_dataset_name: str | None = None,
        use_composite: bool = True,
    ) -> Any | None:
        from PySide6 import QtWidgets

        try:
            datasets, names = slice_viewer_datasets(group, use_composite=use_composite)
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
        setattr(viewer, "_nfit_use_composite", bool(use_composite))
        if selected_dataset_name in names:
            viewer.dataset_combo.setCurrentIndex(names.index(selected_dataset_name))
        viewer.show()
        return viewer

    def _request_overlay_refresh(self, group: DataGroup) -> None:
        """Debounce a slice-viewer refresh, coalescing rapid triggers.

        Bursts of edits (dragging a value, fast typing) or repeated tree
        refreshes collapse into a single recompute after a short idle, so the
        (expensive) overlay is not re-evaluated on every event. Outside the Qt
        event loop -- headless/test use, where ``_interactive`` is False -- it
        refreshes synchronously so callers see the update immediately.
        """

        if id(group) not in self._slice_viewers:
            return
        if not self._interactive:
            self.refresh_slice_viewer(group)
            return
        self._pending_overlay_groups[id(group)] = group
        timer = self._overlay_refresh_timer
        if timer is None:
            try:
                from PySide6 import QtCore
            except Exception:
                self.refresh_slice_viewer(group)
                return
            timer = QtCore.QTimer(self.window)
            timer.setSingleShot(True)
            timer.setInterval(200)
            timer.timeout.connect(self._run_pending_overlay_refresh)
            self._overlay_refresh_timer = timer
        timer.start()

    def _run_pending_overlay_refresh(self) -> None:
        pending = list(self._pending_overlay_groups.values())
        self._pending_overlay_groups.clear()
        for group in pending:
            self.refresh_slice_viewer(group)

    def refresh_slice_viewer(self, group: DataGroup) -> Any | None:
        if id(group) not in self._slice_viewers:
            return None
        selected_name = None
        current_viewer = self._slice_viewers[id(group)]
        use_composite = bool(getattr(current_viewer, "_nfit_use_composite", True))
        if current_viewer.dataset_combo is not None:
            selected_name = current_viewer.dataset_combo.currentText()
        try:
            datasets, names = slice_viewer_datasets(group, use_composite=use_composite)
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
        self.window.setWindowTitle("nfit Project Explorer")
        self.window.resize(1120, 760)

        toolbar = QtWidgets.QToolBar("Project")
        toolbar.setMovable(False)
        self.window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        file_button = QtWidgets.QToolButton()
        file_button.setObjectName("file_menu_button")
        file_button.setText("File")
        file_button.setToolTip("Open project file operations such as New, Open, Save, Close, and Quit.")
        file_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(file_button)
        menu.setToolTipsVisible(True)
        self.file_menu = menu
        new_action = menu.addAction("New", self.new_project)
        new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        new_action.setToolTip("Start a new empty nfit project.")
        new_action.setStatusTip("Start a new empty nfit project.")
        open_action = menu.addAction("Open", self.open_project)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.setToolTip("Open a saved nfit project file.")
        open_action.setStatusTip("Open a saved nfit project file.")
        self.recent_projects_menu = menu.addMenu("Recent projects")
        self.recent_projects_menu.setToolTipsVisible(True)
        self.recent_projects_menu.setToolTip("Open one of the most recently used nfit project files.")
        self.recent_projects_menu.aboutToShow.connect(self._refresh_recent_projects_menu)
        menu.addSeparator()
        save_action = menu.addAction("Save", self.save)
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.setToolTip("Save the current project to its existing project file.")
        save_action.setStatusTip("Save the current project to its existing project file.")
        save_as_action = menu.addAction("Save As", self.save_as)
        save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        save_as_action.setToolTip("Choose a new file path and save the current project there.")
        save_as_action.setStatusTip("Choose a new file path and save the current project there.")
        menu.addSeparator()
        close_action = menu.addAction("Close", self.close_project)
        close_action.setShortcut(QtGui.QKeySequence.StandardKey.Close)
        close_action.setToolTip("Close the current project after prompting to save unsaved changes.")
        close_action.setStatusTip("Close the current project after prompting to save unsaved changes.")
        quit_action = menu.addAction("Quit", self.quit_application)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.setToolTip("Quit nfit after prompting to save unsaved project changes.")
        quit_action.setStatusTip("Quit nfit after prompting to save unsaved project changes.")
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
        self.tree.setToolTip(
            "Project explorer tree. Select items to edit them, expand folders to navigate, "
            "drag supported items to reorder or move them, and right-click for actions."
        )
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
        self.expand_all_button.setToolTip("Expand every workspace, folder, dataset, and fit timeline in the tree.")
        self.collapse_all_button.setToolTip("Collapse the tree to the top-level workspaces.")
        self.expand_all_button.clicked.connect(self.expand_all)
        self.collapse_all_button.clicked.connect(self.collapse_all)
        tree_expand_row.addWidget(self.expand_all_button)
        tree_expand_row.addWidget(self.collapse_all_button)
        left_layout.addLayout(tree_expand_row)

        tree_button_row = QtWidgets.QHBoxLayout()
        tree_button_row.setContentsMargins(8, 0, 8, 8)
        self.create_group_button = QtWidgets.QPushButton("Create workspace")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.create_group_button.setToolTip("Create a new top-level workspace and immediately rename it.")
        self.delete_button.setToolTip("Delete the selected workspace, dataset, mask, model, or fit item when allowed.")
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
        self.enabled_check.setToolTip("Include or exclude the selected dataset, mask, or model from viewing and fitting.")
        self.enabled_check.toggled.connect(self._set_selected_enabled)
        title_row.addWidget(self.enabled_check)
        self.fit_weight_widget = QtWidgets.QWidget()
        fit_weight_layout = QtWidgets.QHBoxLayout(self.fit_weight_widget)
        fit_weight_layout.setContentsMargins(0, 0, 0, 0)
        fit_weight_layout.setSpacing(6)
        fit_weight_layout.addWidget(QtWidgets.QLabel("Fit weight"))
        self.fit_weight_spin = QtWidgets.QDoubleSpinBox()
        self.fit_weight_spin.setToolTip("Relative fitting weight for the selected dataset. Larger values make this dataset count more in the fit.")
        self.fit_weight_spin.setRange(0.0, 1.0e12)
        self.fit_weight_spin.setDecimals(6)
        self.fit_weight_spin.setSingleStep(0.1)
        self.fit_weight_spin.setValue(1.0)
        self.fit_weight_spin.valueChanged.connect(self._set_selected_dataset_fit_weight)
        fit_weight_layout.addWidget(self.fit_weight_spin)
        fit_weight_layout.addWidget(QtWidgets.QLabel("Scale"))
        self.scale_factor_spin = QtWidgets.QDoubleSpinBox()
        self.scale_factor_spin.setObjectName("dataset_scale_factor")
        self.scale_factor_spin.setToolTip("Scale factor applied to the selected dataset before viewing and fitting.")
        self.scale_factor_spin.setRange(-1.0e12, 1.0e12)
        self.scale_factor_spin.setDecimals(6)
        self.scale_factor_spin.setSingleStep(0.1)
        self.scale_factor_spin.setValue(1.0)
        self.scale_factor_spin.valueChanged.connect(self._set_selected_dataset_scale_factor)
        fit_weight_layout.addWidget(self.scale_factor_spin)
        fit_weight_layout.addWidget(QtWidgets.QLabel("T (K)"))
        self.dataset_temperature_spin = QtWidgets.QDoubleSpinBox()
        self.dataset_temperature_spin.setObjectName("dataset_temperature")
        self.dataset_temperature_spin.setToolTip(
            "Sample temperature override in kelvin for the selected dataset. "
            "Physics models use it for the Bose factor; set to '(from data)' "
            "(spin to the minimum) to use the temperature imported with the data."
        )
        self.dataset_temperature_spin.setRange(-1.0, 1.0e4)
        self.dataset_temperature_spin.setDecimals(3)
        self.dataset_temperature_spin.setSingleStep(1.0)
        self.dataset_temperature_spin.setSpecialValueText("(from data)")
        self.dataset_temperature_spin.setValue(-1.0)
        self.dataset_temperature_spin.valueChanged.connect(self._set_selected_dataset_temperature)
        fit_weight_layout.addWidget(self.dataset_temperature_spin)
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
        self.group_fit_weight_edit.setToolTip(
            "Bulk edit the fit weight for every dataset in this dataset group. Blank means descendant values differ."
        )
        self.group_fit_weight_edit.setPlaceholderText("(mixed)")
        self.group_fit_weight_edit.setMaximumWidth(90)
        self.group_fit_weight_edit.editingFinished.connect(
            lambda: self._set_group_bulk_value("fit_weight", self.group_fit_weight_edit.text())
        )
        group_bulk_layout.addWidget(self.group_fit_weight_edit)
        group_bulk_layout.addWidget(QtWidgets.QLabel("Scale"))
        self.group_scale_edit = QtWidgets.QLineEdit()
        self.group_scale_edit.setObjectName("group_scale_edit")
        self.group_scale_edit.setToolTip(
            "Bulk edit the scale factor for every dataset in this dataset group. Blank means descendant values differ."
        )
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
        self.import_dataset_button.setToolTip("Import one or more data files into the selected workspace or dataset group.")
        self.add_model_button.setToolTip("Add a new model component to the selected workspace.")
        self.view_slice_button.setToolTip("Open or refresh the data viewer for the selected workspace or dataset.")
        self.load_dataset_button.setToolTip("Load this dataset from disk now so its axes, data, and metadata are available.")
        self.add_mask_button.setToolTip("Create a new mask under the selected dataset or shared mask folder.")
        self.add_dataset_group_button.setToolTip("Create a nested dataset group for organizing related datasets and shared masks.")
        self.save_dataset_button.setToolTip("Export the selected dataset, including current nfit processing, to a data file.")
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
        self.mask_type_combo.setToolTip("Choose the mask type for the selected mask. The parameter editor updates to match this type.")
        self.mask_type_combo.currentTextChanged.connect(self._set_selected_mask_type)
        self.mask_parameter_widget = QtWidgets.QWidget()
        self.mask_parameter_layout = QtWidgets.QGridLayout(self.mask_parameter_widget)
        self.mask_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.mask_parameter_layout.setColumnStretch(1, 1)
        model_combo_class = _make_refreshing_combo_class()
        self.model_type_combo = model_combo_class(self._refresh_model_type_combo)
        self.model_type_combo.setObjectName("model_type_combo")
        self.model_type_combo.setToolTip("Choose the model function used by the selected model component.")
        self.model_type_combo.currentTextChanged.connect(self._set_selected_model_type)
        self.model_parameter_scroll = QtWidgets.QScrollArea()
        self.model_parameter_scroll.setObjectName("model_parameter_scroll")
        self.model_parameter_scroll.setWidgetResizable(True)
        self.model_parameter_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.model_parameter_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.model_parameter_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.model_parameter_scroll.setMinimumHeight(240)
        self.model_parameter_widget = QtWidgets.QWidget()
        self.model_parameter_layout = QtWidgets.QGridLayout(self.model_parameter_widget)
        self.model_parameter_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.model_parameter_layout.setColumnStretch(1, 1)
        self.model_parameter_scroll.setWidget(self.model_parameter_widget)
        fit_editor = QtWidgets.QWidget()
        fit_editor_layout = QtWidgets.QHBoxLayout(fit_editor)
        fit_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_optimizer_combo = QtWidgets.QComboBox()
        self.fit_optimizer_combo.addItems(["least_squares"])
        self.fit_optimizer_combo.setToolTip("Choose the optimizer used when fitting from this fit state.")
        self.fit_optimizer_combo.currentTextChanged.connect(self._set_selected_fit_optimizer)
        self.fit_loss_combo = QtWidgets.QComboBox()
        self.fit_loss_combo.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        self.fit_loss_combo.setToolTip(
            "Least-squares loss. Use linear for ordinary chi-squared; robust losses reduce the influence of outliers."
        )
        self.fit_loss_combo.currentTextChanged.connect(self._set_selected_fit_controls_config)
        self.fit_f_scale_spin = QtWidgets.QDoubleSpinBox()
        self.fit_f_scale_spin.setRange(1.0e-9, 1.0e9)
        self.fit_f_scale_spin.setDecimals(6)
        self.fit_f_scale_spin.setValue(1.0)
        self.fit_f_scale_spin.setToolTip(
            "Residual scale where robust losses begin down-weighting points. With normalized residuals, 1.0 means about one sigma."
        )
        self.fit_f_scale_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_de_check = QtWidgets.QCheckBox("Differential evolution initialization")
        self.fit_de_check.setToolTip(
            "Search the bounded parameter space before least squares. Requires finite bounds on every fitted parameter."
        )
        self.fit_de_check.stateChanged.connect(self._set_selected_fit_controls_config)
        self.fit_de_maxiter_spin = QtWidgets.QSpinBox()
        self.fit_de_maxiter_spin.setRange(1, 10000)
        self.fit_de_maxiter_spin.setValue(60)
        self.fit_de_maxiter_spin.setToolTip("Maximum differential-evolution generations before least-squares polishing.")
        self.fit_de_maxiter_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_de_popsize_spin = QtWidgets.QSpinBox()
        self.fit_de_popsize_spin.setRange(2, 200)
        self.fit_de_popsize_spin.setValue(10)
        self.fit_de_popsize_spin.setToolTip("Population multiplier for differential evolution. Larger values explore more but run longer.")
        self.fit_de_popsize_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_de_workers_spin = QtWidgets.QSpinBox()
        self.fit_de_workers_spin.setRange(-1, 256)
        self.fit_de_workers_spin.setValue(1)
        self.fit_de_workers_spin.setToolTip(
            "Parallel worker threads for differential-evolution objective evaluations. Use 1 for serial execution or -1 for an automatic CPU-based choice."
        )
        self.fit_de_workers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_check = QtWidgets.QCheckBox("Sample posterior with emcee")
        self.fit_emcee_check.setToolTip(
            "After least squares, run emcee walkers near the best fit to estimate posterior intervals and correlations."
        )
        self.fit_emcee_check.stateChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_walkers_spin = QtWidgets.QSpinBox()
        self.fit_emcee_walkers_spin.setRange(0, 10000)
        self.fit_emcee_walkers_spin.setValue(0)
        self.fit_emcee_walkers_spin.setToolTip("Number of emcee walkers. Use 0 to choose an automatic value from the parameter count.")
        self.fit_emcee_walkers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_steps_spin = QtWidgets.QSpinBox()
        self.fit_emcee_steps_spin.setRange(1, 1000000)
        self.fit_emcee_steps_spin.setValue(1000)
        self.fit_emcee_steps_spin.setToolTip("Number of emcee steps per walker.")
        self.fit_emcee_steps_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_burn_spin = QtWidgets.QSpinBox()
        self.fit_emcee_burn_spin.setRange(0, 1000000)
        self.fit_emcee_burn_spin.setValue(200)
        self.fit_emcee_burn_spin.setToolTip("Initial emcee steps to discard before summarizing posterior samples.")
        self.fit_emcee_burn_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_thin_spin = QtWidgets.QSpinBox()
        self.fit_emcee_thin_spin.setRange(1, 10000)
        self.fit_emcee_thin_spin.setValue(1)
        self.fit_emcee_thin_spin.setToolTip("Keep every Nth emcee sample after burn-in.")
        self.fit_emcee_thin_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_emcee_workers_spin = QtWidgets.QSpinBox()
        self.fit_emcee_workers_spin.setRange(-1, 256)
        self.fit_emcee_workers_spin.setValue(1)
        self.fit_emcee_workers_spin.setToolTip(
            "Parallel worker threads for emcee log-probability evaluations. Use 1 for serial execution or -1 for an automatic CPU-based choice."
        )
        self.fit_emcee_workers_spin.valueChanged.connect(self._set_selected_fit_controls_config)
        self.fit_optimizer_config_editor = QtWidgets.QLineEdit("{}")
        self.fit_optimizer_config_editor.setToolTip(
            "Advanced JSON optimizer configuration for the selected fit state. GUI controls update this; edit directly for extra SciPy/emcee options."
        )
        self.fit_optimizer_config_editor.editingFinished.connect(self._set_selected_fit_optimizer_config)
        self.fit_branch_check = QtWidgets.QCheckBox("Branch timeline")
        self.fit_now_button = QtWidgets.QPushButton("Fit now")
        self.fit_corner_button = QtWidgets.QPushButton("Fit diagnostics")
        self.show_data_fit_button = QtWidgets.QPushButton("Show data and model")
        self.fit_branch_check.setToolTip("Start a new nested fit timeline instead of appending to the current timeline.")
        self.fit_now_button.setToolTip("Run the optimizer from the selected fit state and store the result in the fit history.")
        self.fit_corner_button.setToolTip(
            "Open covariance/correlation heatmaps and posterior trace or corner-style plots when diagnostics are stored."
        )
        self.show_data_fit_button.setToolTip(
            "Open the data viewer with the model overlay enabled, using current model parameters or stored fit channels."
        )
        self.fit_now_button.clicked.connect(self.start_fit_for_selection)
        self.fit_corner_button.clicked.connect(self.open_fit_diagnostics_plots_for_selection)
        self.show_data_fit_button.clicked.connect(self.show_data_and_fit_for_selection)
        fit_editor_layout.addWidget(self.fit_branch_check)
        fit_editor_layout.addStretch(1)
        self.fit_editor_widget = fit_editor

        self.fit_settings_panel = QtWidgets.QWidget()
        self.fit_settings_panel.setObjectName("fit_settings_panel")
        fit_settings_layout = QtWidgets.QVBoxLayout(self.fit_settings_panel)
        fit_settings_layout.setContentsMargins(0, 0, 0, 0)
        fit_settings_layout.setSpacing(8)

        optimizer_group = QtWidgets.QGroupBox("Optimizer")
        optimizer_group.setObjectName("fit_optimizer_settings_group")
        optimizer_layout = QtWidgets.QGridLayout(optimizer_group)
        optimizer_layout.setColumnStretch(1, 1)
        optimizer_layout.addWidget(QtWidgets.QLabel("Optimizer"), 0, 0)
        optimizer_layout.addWidget(self.fit_optimizer_combo, 0, 1)
        optimizer_layout.addWidget(QtWidgets.QLabel("Loss"), 1, 0)
        optimizer_layout.addWidget(self.fit_loss_combo, 1, 1)
        optimizer_layout.addWidget(QtWidgets.QLabel("Loss scale"), 2, 0)
        optimizer_layout.addWidget(self.fit_f_scale_spin, 2, 1)
        optimizer_layout.addWidget(QtWidgets.QLabel("Advanced config"), 3, 0)
        optimizer_layout.addWidget(self.fit_optimizer_config_editor, 3, 1)
        fit_settings_layout.addWidget(optimizer_group)

        de_group = QtWidgets.QGroupBox("Differential Evolution")
        de_group.setObjectName("fit_de_settings_group")
        de_layout = QtWidgets.QGridLayout(de_group)
        de_layout.setColumnStretch(1, 1)
        de_layout.addWidget(self.fit_de_check, 0, 1)
        de_layout.addWidget(QtWidgets.QLabel("DE generations"), 1, 0)
        de_layout.addWidget(self.fit_de_maxiter_spin, 1, 1)
        de_layout.addWidget(QtWidgets.QLabel("DE population"), 2, 0)
        de_layout.addWidget(self.fit_de_popsize_spin, 2, 1)
        de_layout.addWidget(QtWidgets.QLabel("DE workers"), 3, 0)
        de_layout.addWidget(self.fit_de_workers_spin, 3, 1)
        fit_settings_layout.addWidget(de_group)

        posterior_group = QtWidgets.QGroupBox("Posterior")
        posterior_group.setObjectName("fit_posterior_settings_group")
        posterior_layout = QtWidgets.QGridLayout(posterior_group)
        posterior_layout.setColumnStretch(1, 1)
        posterior_layout.addWidget(self.fit_emcee_check, 0, 1)
        posterior_layout.addWidget(QtWidgets.QLabel("Walkers"), 1, 0)
        posterior_layout.addWidget(self.fit_emcee_walkers_spin, 1, 1)
        posterior_layout.addWidget(QtWidgets.QLabel("Steps"), 2, 0)
        posterior_layout.addWidget(self.fit_emcee_steps_spin, 2, 1)
        posterior_layout.addWidget(QtWidgets.QLabel("Burn-in"), 3, 0)
        posterior_layout.addWidget(self.fit_emcee_burn_spin, 3, 1)
        posterior_layout.addWidget(QtWidgets.QLabel("Thin"), 4, 0)
        posterior_layout.addWidget(self.fit_emcee_thin_spin, 4, 1)
        posterior_layout.addWidget(QtWidgets.QLabel("emcee workers"), 5, 0)
        posterior_layout.addWidget(self.fit_emcee_workers_spin, 5, 1)
        fit_settings_layout.addWidget(posterior_group)

        right_layout.addLayout(title_row)
        right_layout.addWidget(self.mask_type_combo)
        right_layout.addWidget(self.mask_parameter_widget)
        right_layout.addWidget(self.model_type_combo)
        right_layout.addWidget(self.model_parameter_scroll, 5)
        right_layout.addWidget(self.fit_editor_widget)
        right_layout.addWidget(self.details_scroll, 1)
        right_layout.addLayout(actions_row)
        right_layout.addWidget(self.fit_now_button)
        right_layout.addWidget(self.fit_corner_button)
        right_layout.addWidget(self.show_data_fit_button)

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
        if self._active_fit_entry(group) is fit_entry:
            _style_active_fit_tree_item(item)
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
        self.model_parameter_scroll.setVisible(role == "model")
        self.fit_editor_widget.setVisible(role == "fit" and fit_entry is not None)
        self.fit_now_button.setVisible(role == "fit" and fit_entry is not None)
        self.fit_corner_button.setVisible(
            role == "fit"
            and fit_entry is not None
            and _fit_entry_has_diagnostic_plots(fit_entry)
        )
        self.show_data_fit_button.setVisible(
            role == "fit"
            and fit_entry is not None
            and group is not None
            and (_group_has_fit_channels(group) or _group_has_enabled_model_components(group))
        )

        if role == "group" and group is not None:
            self.title_label.setText(group.name)
            ensure_fit_history(group)
            self._set_group_details(group)
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
            self._set_fit_details(fit_entry)
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
        self.dataset_temperature_spin.blockSignals(True)
        try:
            override = (
                entry.parameters.get("temperature")
                if role == "dataset" and entry is not None
                else None
            )
            self.dataset_temperature_spin.setValue(
                float(override) if override not in (None, "") else -1.0
            )
        finally:
            self.dataset_temperature_spin.blockSignals(False)
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
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        self._sync_details()

    def _set_selected_dataset_scale_factor(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        scale = float(value)
        if entry.scale_factor == scale:
            return
        entry.scale_factor = scale
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        if group is not None:
            self.refresh_slice_viewer(group)
        self._sync_details()

    def _set_selected_dataset_temperature(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        override = None if float(value) < 0.0 else float(value)
        if entry.parameters.get("temperature") == override:
            return
        if override is None:
            entry.parameters.pop("temperature", None)
        else:
            entry.parameters["temperature"] = override
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
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

    def _set_group_details(self, group: DataGroup) -> None:
        result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
        datasets = list(group.iter_datasets())
        text = (
            f"Workspace\n\nDatasets: {len(datasets)}\nModels: {len(group.models)}\n"
            f"Fit results: {result_count}"
        )
        self.details_label.setText(text)
        self._clear_details_panel()
        self.details_layout.addWidget(
            self._details_group_box(
                "Workspace",
                [
                    f"Datasets: {len(datasets)}",
                    f"Data points: {_format_number(_group_data_point_count(group))}",
                    f"Dataset types: {_group_dataset_type_summary(group)}",
                    f"Models: {len(group.models)}",
                    f"Fit results: {result_count}",
                ],
            )
        )
        self.details_layout.addWidget(self._group_dataset_weights_group_box(group))
        self.details_layout.addWidget(self._group_composite_group_box(group))
        self.details_layout.addStretch(1)

    def _group_dataset_weights_group_box(self, group: DataGroup) -> Any:
        from PySide6 import QtWidgets

        box = QtWidgets.QGroupBox("Datasets")
        layout = QtWidgets.QGridLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        headers = ["Name", "Type", "Points", "Fit weight", "Scale"]
        for column, label in enumerate(headers):
            layout.addWidget(QtWidgets.QLabel(label), 0, column)
        for row, dataset in enumerate(group.iter_datasets(), start=1):
            layout.addWidget(QtWidgets.QLabel(dataset.name), row, 0)
            layout.addWidget(QtWidgets.QLabel(data_type_label(dataset.data_type)), row, 1)
            layout.addWidget(QtWidgets.QLabel(_format_number(_dataset_data_point_count(dataset))), row, 2)
            layout.addWidget(QtWidgets.QLabel(_format_number(dataset.fit_weight)), row, 3)
            layout.addWidget(QtWidgets.QLabel(_format_number(dataset.scale_factor)), row, 4)
        box.setToolTip(
            "Datasets in this workspace. In composite mode each dataset's signal is multiplied by its scale factor, "
            "its uncertainty by the absolute scale factor, and its statistical contribution by the fit weight. "
            "Use a negative scale factor to subtract a dataset from the composite."
        )
        return box

    def _group_composite_group_box(self, group: DataGroup) -> Any:
        from PySide6 import QtWidgets

        box = QtWidgets.QGroupBox("Composite dataset")
        box.setToolTip(
            "Combine compatible enabled datasets into one rebinned effective dataset for this workspace. "
            "Enable the checkbox to show the composite rebin controls."
        )
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        config = data_group_composite_config(group)
        can_combine, message = data_group_composite_status(group)
        enable_check = QtWidgets.QCheckBox("Combine enabled datasets into one effective dataset")
        enable_check.setObjectName("group_composite_enabled")
        enable_check.setChecked(bool(config.get("enabled", False)))
        enable_check.setEnabled(can_combine)
        enable_check.setToolTip(
            "When checked, this workspace plots and fits one rebinned composite dataset. "
            "The fitter receives only the composite and does not see the constituent datasets. "
            "All enabled datasets must have the same data kind. Negative dataset scale factors subtract data."
        )
        enable_check.toggled.connect(lambda checked: self._set_group_composite_enabled(group, checked))
        layout.addWidget(enable_check)
        message_label = QtWidgets.QLabel(message)
        message_label.setWordWrap(True)
        message_label.setToolTip("Composite status. All enabled datasets must have the same data kind before they can be combined.")
        layout.addWidget(message_label)

        controls = QtWidgets.QWidget()
        controls.setEnabled(can_combine)
        controls.setVisible(bool(config.get("enabled", False)))
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        axes = config.get("axes", [])
        headers = ["Axis", "Lower", "Upper", "Bins", "Step"]
        for column, label in enumerate(headers):
            controls_layout.addWidget(QtWidgets.QLabel(label), 0, column)
        for row, axis_config in enumerate(axes, start=1):
            axis = _sanitize_rebin_axis_config(axis_config)
            controls_layout.addWidget(QtWidgets.QLabel(str(axis.get("name", f"Axis {row}"))), row, 0)
            for column, key in enumerate(("lower", "upper", "num_bins", "step_size"), start=1):
                edit = QtWidgets.QLineEdit(_parameter_to_text(axis.get(key)))
                edit.setMinimumWidth(72)
                edit.setToolTip(
                    f"Composite rebin {key.replace('_', ' ')} for this axis. "
                    "These bounds and bins are applied after all enabled datasets are scaled, weighted, and collected."
                )
                edit.editingFinished.connect(
                    lambda row=row - 1, key=key, editor=edit: self._set_group_composite_axis_value(group, row, key, editor.text())
                )
                controls_layout.addWidget(edit, row, column)
        option_row = QtWidgets.QHBoxLayout()
        fractional_check = QtWidgets.QCheckBox("Fractional binning")
        fractional_check.setObjectName("group_composite_fractional")
        fractional_check.setChecked(bool(config.get("fractional", True)))
        fractional_check.setToolTip(
            "Distribute source points fractionally into neighboring composite bins after dataset scale and fit-weight factors are applied."
        )
        fractional_check.toggled.connect(lambda checked: self._set_group_composite_option(group, "fractional", checked))
        mean_label = QtWidgets.QLabel("Mean")
        mean_combo = QtWidgets.QComboBox()
        mean_combo.setObjectName("group_composite_mean_weighting")
        mean_combo.setToolTip(
            "Choose the composite averaging mode. Inverse variance uses fit_weight/sigma^2 after dataset scale factors are applied; "
            "uniform keeps a simple weighted geometric mean."
        )
        mean_combo.addItem("Inverse variance", "inverse_variance")
        mean_combo.addItem("Uniform", "uniform")
        mean_combo.setCurrentIndex(max(mean_combo.findData(_rebin_mean_weighting(config)), 0))
        mean_combo.currentIndexChanged.connect(
            lambda _index, combo=mean_combo: self._set_group_composite_mean_weighting(group, str(combo.currentData() or "inverse_variance"))
        )
        batch_label = QtWidgets.QLabel("Batch target")
        batch_spin = QtWidgets.QSpinBox()
        batch_spin.setObjectName("group_composite_max_batch_mb")
        batch_spin.setRange(1, 1_048_576)
        batch_spin.setSuffix(" MB")
        batch_spin.setValue(_rebin_max_batch_mb(config))
        batch_tooltip = (
            "Approximate per-batch working-memory target in MB. Smaller batches usually use less temporary memory "
            "but require more computational time. This is not a cap on total rebinner memory use. The optimum depends "
            "on dataset size, output grid size, dimensionality, and available memory."
        )
        batch_label.setToolTip(batch_tooltip)
        batch_spin.setToolTip(batch_tooltip)
        batch_spin.valueChanged.connect(lambda value: self._set_group_composite_max_batch_mb(group, int(value)))
        option_row.addWidget(fractional_check)
        option_row.addWidget(mean_label)
        option_row.addWidget(mean_combo)
        option_row.addWidget(batch_label)
        option_row.addWidget(batch_spin)
        option_row.addStretch(1)
        controls_layout.addLayout(option_row, len(axes) + 1, 0, 1, len(headers))
        layout.addWidget(controls)
        return box

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

    def _set_fit_details(self, fit_entry: FitTimelineEntry) -> None:
        from PySide6 import QtCore, QtWidgets

        group, _entry, _mask, _model, _role = self._objects_for_item(self._current_item())
        self.details_label.setText(fit_details_text(fit_entry))
        self._clear_details_panel()
        duration = (
            f"{_format_number(fit_entry.duration_seconds)} s"
            if fit_entry.duration_seconds is not None
            else "-"
        )
        if self.fit_settings_panel is not None:
            self.details_layout.addWidget(self.fit_settings_panel)
        self.details_layout.addWidget(
            self._details_group_box(
                "Fit",
                [
                    f"Type: {fit_entry.kind}",
                    f"Created: {fit_entry.created_at or '-'}",
                    f"Optimizer: {fit_entry.optimizer or '-'}",
                    f"Duration: {duration}",
                    f"Timeline entries: {len(fit_entry.children)}",
                ],
            )
        )
        parameter_group = None
        lower_widgets = []
        if _fit_results_rows(fit_entry):
            parameter_group = self._fit_results_group_box(fit_entry)
            if group is not None and fit_entry.kind == "result":
                lower_widgets.append(self._posterior_sampler_group_box(group, fit_entry))
        elif _snapshot_parameter_rows(fit_entry):
            parameter_group = self._parameter_values_group_box(fit_entry)
        if fit_entry.optimizer_config:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Optimizer config",
                    dict(fit_entry.optimizer_config),
                    object_name="fit_optimizer_config_tree",
                    empty_text="No optimizer configuration.",
                )
            )
        if fit_entry.goodness:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Goodness of fit",
                    dict(fit_entry.goodness),
                    object_name="fit_goodness_tree",
                    empty_text="No goodness-of-fit values.",
                )
            )
        if fit_entry.channels:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Stored fit channels",
                    dict(fit_entry.channels),
                    object_name="fit_channels_tree",
                    empty_text="No stored fit channels.",
                )
            )
        if fit_entry.metadata:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Metadata",
                    dict(fit_entry.metadata),
                    object_name="fit_metadata_tree",
                    empty_text="No fit metadata.",
                )
            )
        if fit_entry.snapshot:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Snapshot",
                    dict(fit_entry.snapshot),
                    object_name="fit_snapshot_tree",
                    empty_text="No fit snapshot.",
                )
            )
        if parameter_group is not None and lower_widgets:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            splitter.setObjectName("fit_details_parameter_splitter")
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(parameter_group)
            lower_panel = QtWidgets.QWidget()
            lower_layout = QtWidgets.QVBoxLayout(lower_panel)
            lower_layout.setContentsMargins(0, 0, 0, 0)
            lower_layout.setSpacing(8)
            for widget in lower_widgets:
                lower_layout.addWidget(widget)
            lower_layout.addStretch(1)
            splitter.addWidget(lower_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([260, 420])
            self.details_layout.addWidget(splitter)
        else:
            if parameter_group is not None:
                self.details_layout.addWidget(parameter_group)
            for widget in lower_widgets:
                self.details_layout.addWidget(widget)
        self.details_layout.addStretch(1)

    def _fit_results_group_box(self, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtCore, QtWidgets

        rows = _fit_results_rows(fit_entry)
        group_box = QtWidgets.QGroupBox("Fit results")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        table = QtWidgets.QTableWidget(len(rows), 6)
        table.setObjectName("fit_results_table")
        table.setToolTip(
            "Human-readable best-fit parameters. Standard errors come from the least-squares covariance; "
            "posterior columns come from emcee samples when available."
        )
        table.setHorizontalHeaderLabels(
            ["Parameter", "Best fit", "Std err", "Posterior median", "16%", "84%"]
        )
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(120)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(("name", "value", "stderr", "median", "p16", "p84")):
                item = QtWidgets.QTableWidgetItem(row.get(key, "-"))
                item.setToolTip(row.get(key, "-"))
                if column_index > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row_index, column_index, item)
        _tooltip_table_corner_buttons(table, "Select all fit-result rows.")
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return group_box

    def _posterior_sampler_group_box(self, group: DataGroup, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtWidgets

        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        metadata = dict(stored.metadata) if stored is not None else {}
        sampler = fit_entry.optimizer_config.get("sampler", {}) if isinstance(fit_entry.optimizer_config, dict) else {}
        chain_steps = 0
        chain_walkers = 0
        if stored is not None and stored.chain is not None:
            chain = np.asarray(stored.chain, dtype=float)
            if chain.ndim == 3:
                chain_steps = int(chain.shape[0])
                chain_walkers = int(chain.shape[1])
        promotion_candidate = _best_posterior_promotion_candidate(fit_entry)
        promotion_message = "Stored posterior samples do not include a better finite likelihood than this fit result."
        if promotion_candidate is not None:
            _params, best_log_probability, baseline_log_probability, location = promotion_candidate
            location_text = ", ".join(f"{key} {value}" for key, value in location.items())
            promotion_message = (
                "Create a Current state from the best stored emcee sample "
                f"({location_text}; log probability {_format_number(best_log_probability)} "
                f"vs {_format_number(baseline_log_probability)} for this fit result)."
            )
        walkers_default = int(metadata.get("n_walkers") or sampler.get("n_walkers") or 0)
        steps_default = int(metadata.get("n_steps") or sampler.get("n_steps") or 1000)
        burn_default = int(metadata.get("burn_in") or sampler.get("burn_in") or 0)
        thin_default = max(1, int(metadata.get("thin") or sampler.get("thin") or 1))
        workers_default = int(sampler.get("workers", metadata.get("workers", 1)) or 1)
        seed_value = metadata.get("random_seed", sampler.get("random_seed", None))

        group_box = QtWidgets.QGroupBox("Posterior sampler")
        group_box.setToolTip(
            "Inspect or update emcee posterior samples for this fit result without running least squares or creating a timeline entry."
        )
        layout = QtWidgets.QGridLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)

        status = QtWidgets.QLabel(
            f"Stored samples: {0 if stored is None else len(stored.samples):d}    "
            f"Raw chain: {chain_steps:d} steps x {chain_walkers:d} walkers"
        )
        status.setToolTip("Current posterior storage for this fit result.")
        layout.addWidget(status, 0, 0, 1, 4)

        walkers_spin = QtWidgets.QSpinBox()
        walkers_spin.setObjectName("fit_posterior_walkers_spin")
        walkers_spin.setRange(0, 10000)
        walkers_spin.setValue(max(0, walkers_default))
        walkers_spin.setToolTip(
            "Number of walkers for a full emcee rerun. Use 0 to choose an automatic value; append continues from the stored walkers."
        )

        steps_spin = QtWidgets.QSpinBox()
        steps_spin.setObjectName("fit_posterior_steps_spin")
        steps_spin.setRange(1, 1000000)
        steps_spin.setValue(max(1, steps_default))
        steps_spin.setToolTip("Number of emcee steps to run. For append, this many new steps are added to the stored chain.")

        burn_spin = QtWidgets.QSpinBox()
        burn_spin.setObjectName("fit_posterior_burn_spin")
        burn_spin.setRange(0, 1000000)
        burn_spin.setValue(max(0, burn_default))
        burn_spin.setToolTip(
            "Initial chain steps to discard before computing posterior summaries and diagnostic plots. "
            "Changing this only changes the view of stored samples."
        )

        thin_spin = QtWidgets.QSpinBox()
        thin_spin.setObjectName("fit_posterior_thin_spin")
        thin_spin.setRange(1, 10000)
        thin_spin.setValue(thin_default)
        thin_spin.setToolTip(
            "Keep every Nth stored sample after burn-in when computing summaries and diagnostic plots."
        )

        seed_spin = QtWidgets.QSpinBox()
        seed_spin.setObjectName("fit_posterior_seed_spin")
        seed_spin.setRange(-1, 2147483647)
        seed_spin.setValue(-1 if seed_value in (None, "") else int(seed_value))
        seed_spin.setToolTip("Random seed for a full rerun. Use -1 for a fresh random initialization.")

        workers_spin = QtWidgets.QSpinBox()
        workers_spin.setObjectName("fit_posterior_workers_spin")
        workers_spin.setRange(-1, 256)
        workers_spin.setValue(workers_default)
        workers_spin.setToolTip(
            "Parallel worker threads for emcee log-probability evaluations during rerun or append. Use -1 for an automatic CPU-based choice."
        )

        controls = [
            ("Walkers", walkers_spin),
            ("Steps", steps_spin),
            ("Burn-in", burn_spin),
            ("Thin", thin_spin),
            ("Seed", seed_spin),
            ("Workers", workers_spin),
        ]
        for index, (label, widget) in enumerate(controls, start=1):
            layout.addWidget(QtWidgets.QLabel(label), index, 0)
            layout.addWidget(widget, index, 1, 1, 3)

        apply_button = QtWidgets.QPushButton("Apply burn-in/thin")
        apply_button.setObjectName("fit_posterior_apply_button")
        apply_button.setToolTip("Recompute posterior summaries and plots from the stored raw chain without rerunning emcee.")
        apply_button.setEnabled(stored is not None and stored.chain is not None)
        apply_button.clicked.connect(
            lambda _checked=False: self.apply_posterior_sampling_window(
                fit_entry, burn_spin.value(), thin_spin.value()
            )
        )

        rerun_button = QtWidgets.QPushButton("Rerun emcee")
        rerun_button.setObjectName("fit_posterior_rerun_button")
        rerun_button.setToolTip(
            "Run a new emcee posterior sample from the best-fit parameters and replace the stored posterior for this fit result."
        )
        rerun_button.clicked.connect(
            lambda _checked=False: self.start_posterior_sampler_for_fit(
                group,
                fit_entry,
                n_walkers=walkers_spin.value(),
                n_steps=steps_spin.value(),
                burn_in=burn_spin.value(),
                thin=thin_spin.value(),
                random_seed=None if seed_spin.value() < 0 else seed_spin.value(),
                workers=workers_spin.value(),
                append=False,
            )
        )

        append_button = QtWidgets.QPushButton("Append steps")
        append_button.setObjectName("fit_posterior_append_button")
        append_button.setToolTip(
            "Continue the stored emcee chain from its last walker positions and then recompute summaries with the selected burn-in/thin."
        )
        append_button.setEnabled(stored is not None and stored.chain is not None)
        append_button.clicked.connect(
            lambda _checked=False: self.start_posterior_sampler_for_fit(
                group,
                fit_entry,
                n_walkers=walkers_spin.value(),
                n_steps=steps_spin.value(),
                burn_in=burn_spin.value(),
                thin=thin_spin.value(),
                random_seed=None if seed_spin.value() < 0 else seed_spin.value(),
                workers=workers_spin.value(),
                append=True,
            )
        )

        promote_button = QtWidgets.QPushButton("Use best sample")
        promote_button.setObjectName("fit_posterior_promote_button")
        promote_button.setToolTip(promotion_message)
        promote_button.setEnabled(promotion_candidate is not None)
        promote_button.clicked.connect(
            lambda _checked=False: self.promote_best_posterior_sample_for_fit(
                group,
                fit_entry,
            )
        )

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(apply_button)
        button_row.addWidget(rerun_button)
        button_row.addWidget(append_button)
        button_row.addWidget(promote_button)
        layout.addLayout(button_row, len(controls) + 1, 0, 1, 4)
        return group_box

    def _parameter_values_group_box(self, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtCore, QtWidgets

        rows = _snapshot_parameter_rows(fit_entry)
        group_box = QtWidgets.QGroupBox("Parameter values")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setObjectName("parameter_values_table")
        table.setToolTip(
            "Current model parameter values for this state. Run a fit to produce best-fit values with uncertainties."
        )
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(100)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(("name", "value")):
                item = QtWidgets.QTableWidgetItem(row.get(key, "-"))
                item.setToolTip(row.get(key, "-"))
                if column_index > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row_index, column_index, item)
        _tooltip_table_corner_buttons(table, "Select all parameter rows.")
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return group_box

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
            value_combo.setToolTip("Column used as the signal values for this plotted or fitted channel.")
            value_combo.currentTextChanged.connect(
                lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "value", text)
            )
            error_combo = QtWidgets.QComboBox()
            error_combo.addItems(["(none)", *columns])
            error_combo.setCurrentText(str(channel.get("error") or "(none)"))
            error_combo.setToolTip("Optional column containing one-sigma uncertainties for this channel.")
            error_combo.currentTextChanged.connect(
                lambda text, index=row - 1: self._set_point_list_channel(dataset, group, index, "error", text)
            )
            channels_layout.addWidget(value_combo, row, 0)
            channels_layout.addWidget(error_combo, row, 1)
            remove_button = QtWidgets.QPushButton("Remove")
            remove_button.setToolTip("Remove this channel definition from the dataset configuration.")
            remove_button.clicked.connect(
                lambda _checked=False, index=row - 1: self._remove_point_list_channel(dataset, group, index)
            )
            channels_layout.addWidget(remove_button, row, 2)
        add_button = QtWidgets.QPushButton("Add channel")
        add_button.setToolTip("Add another signal channel using columns from this point-list dataset.")
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
        channel_combo.setToolTip("Signal channel to scale and assign units for viewing and fitting.")
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
        factor_spin.setToolTip("Multiplicative factor applied to the selected signal channel.")
        factor_spin.valueChanged.connect(
            lambda value: self._set_point_list_scale(dataset, group, "factor", value)
        )
        grid.addWidget(factor_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Units"), 2, 0)
        units_edit = QtWidgets.QLineEdit(str(scale.get("units", "")))
        units_edit.setToolTip("Display and export units for the scaled signal channel.")
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
        enable.setToolTip("Convert a moment channel to susceptibility by dividing by the selected field column.")
        enable.toggled.connect(lambda checked: self._set_point_list_susceptibility(dataset, group, "enabled", checked))
        grid.addWidget(enable, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Moment"), 1, 0)
        moment_combo = QtWidgets.QComboBox()
        moment_combo.addItems([str(c["label"]) for c in config.get("channels", [])])
        moment_combo.setCurrentText(str(susc.get("moment", "")))
        moment_combo.setToolTip("Signal channel containing the measured moment.")
        moment_combo.currentTextChanged.connect(
            lambda text: self._set_point_list_susceptibility(dataset, group, "moment", text)
        )
        grid.addWidget(moment_combo, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Field"), 2, 0)
        field_combo = QtWidgets.QComboBox()
        field_combo.addItems(list(dataset.data.column_names))
        field_combo.setCurrentText(str(susc.get("field", "")))
        field_combo.setToolTip("Column containing the applied field used to compute susceptibility.")
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
        two_theta_combo.setToolTip("Column containing scattering angle 2theta, used with wavelength to compute |Q|.")
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
        wavelength_spin.setToolTip("Neutron wavelength in Angstroms used with 2theta to compute |Q|.")
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
        type_combo.setToolTip(
            "Interpret this dataset as a specific experimental data type. "
            "This controls metadata handling, viewer readouts, and fitting compatibility."
        )
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
        enable_check.setToolTip("Use the rebinned version of this dataset for viewing and fitting.")
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
            lower_edit.setToolTip("Lower bound of the rebinned axis.")
            upper_edit.setToolTip("Upper bound of the rebinned axis.")
            bins_edit.setToolTip("Number of bins on this rebinned axis. Editing this updates the step size.")
            step_edit.setToolTip("Approximate bin step size on this rebinned axis. Editing this updates the number of bins.")
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
                    "Projection vector defining this rebin coordinate. For 4D MDHisto data, vectors follow "
                    "the displayed H,K,L,E coordinate axis when possible, e.g. DeltaE -> [0, 0, 0, 1] "
                    "and [H,-H,0] -> [1, -1, 0, 0]."
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
        fractional_check.setToolTip("Allow partial source bins to contribute fractionally when rebinning.")
        fractional_check.toggled.connect(lambda checked: self._set_dataset_rebin_option(dataset, group, "fractional", checked))
        mean_label = QtWidgets.QLabel("Mean")
        mean_combo = QtWidgets.QComboBox()
        mean_combo.setObjectName("dataset_rebin_mean_weighting")
        mean_combo.setToolTip(
            "Choose how multiple source points in a rebinned bin are averaged. "
            "Inverse variance uses 1/sigma^2 weights; uniform keeps the simple mean."
        )
        mean_combo.addItem("Inverse variance", "inverse_variance")
        mean_combo.addItem("Uniform", "uniform")
        mean_index = mean_combo.findData(_rebin_mean_weighting(config))
        mean_combo.setCurrentIndex(max(mean_index, 0))
        mean_combo.currentIndexChanged.connect(
            lambda _index, combo=mean_combo: self._set_dataset_rebin_mean_weighting(
                dataset,
                group,
                str(combo.currentData() or "inverse_variance"),
            )
        )
        batch_tooltip = (
            "Approximate per-batch working-memory target in MB. Smaller batches usually use less temporary memory "
            "but require more computational time. This is not a cap on total rebinner memory use; total memory also "
            "depends on dataset size, output grid size, dimensionality, and other arrays. The optimum depends on the "
            "dataset size and available memory."
        )
        batch_label = QtWidgets.QLabel("Batch target")
        batch_label.setToolTip(batch_tooltip)
        batch_spin = QtWidgets.QSpinBox()
        batch_spin.setObjectName("dataset_rebin_max_batch_mb")
        batch_spin.setRange(1, 1_048_576)
        batch_spin.setSuffix(" MB")
        batch_spin.setValue(_rebin_max_batch_mb(config))
        batch_spin.setToolTip(batch_tooltip)
        batch_spin.setAccelerated(True)
        batch_spin.valueChanged.connect(
            lambda value: self._set_dataset_rebin_max_batch_mb(dataset, group, int(value))
        )
        create_button = QtWidgets.QPushButton("Create dataset from rebin")
        create_button.setObjectName("dataset_rebin_create")
        create_button.setEnabled(_dataset_can_rebin(dataset))
        create_button.setToolTip("Materialize the current rebinned data as a new independent dataset.")
        create_button.clicked.connect(self.materialize_rebin_for_selection)
        save_rebin_button = QtWidgets.QPushButton("Save rebin to disk")
        save_rebin_button.setObjectName("dataset_rebin_save")
        save_rebin_button.setEnabled(_dataset_can_rebin(dataset))
        save_rebin_button.setToolTip("Export the current rebinned dataset directly to a NumPy archive.")
        save_rebin_button.clicked.connect(self.save_rebin_for_selection)
        option_row.addWidget(fractional_check)
        option_row.addWidget(mean_label)
        option_row.addWidget(mean_combo)
        option_row.addWidget(batch_label)
        option_row.addWidget(batch_spin)
        option_row.addStretch(1)
        controls_layout.addLayout(option_row, len(config.get("axes", [])) + 1, 0, 1, last_column + 1)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(create_button)
        action_row.addWidget(save_rebin_button)
        action_row.addStretch(1)
        controls_layout.addLayout(action_row, len(config.get("axes", [])) + 2, 0, 1, last_column + 1)
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
        metadata = dict(_dataset_metadata_mapping(dataset))
        if dataset.parameters:
            metadata["Parameters"] = dict(dataset.parameters)
        return self._metadata_tree_group_box(
            "Metadata",
            metadata,
            object_name="dataset_metadata_tree",
            empty_text="No additional metadata.",
        )

    def _metadata_tree_group_box(
        self,
        title: str,
        mapping: dict[str, Any],
        *,
        object_name: str,
        empty_text: str,
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        group_box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        if not mapping:
            label = QtWidgets.QLabel(empty_text)
            label.setWordWrap(True)
            layout.addWidget(label)
            return group_box

        tree = QtWidgets.QTreeWidget()
        tree.setObjectName(object_name)
        tree.setToolTip("Expandable metadata table. Expand rows to inspect nested fields and hover values for full text.")
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

        for key in sorted(mapping):
            item = _add_metadata_tree_item(tree, str(key), mapping[key])
            if key == "Parameters":
                item.setExpanded(True)
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

    def _set_dataset_rebin_mean_weighting(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        value: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        value = value if value in {"inverse_variance", "uniform"} else "inverse_variance"
        if _rebin_mean_weighting(config) == value:
            return
        config["mean_weighting"] = value
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_max_batch_mb(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        value: int,
    ) -> None:
        config = dataset_rebin_config(dataset)
        value = max(int(value), 1)
        if _rebin_max_batch_mb(config) == value:
            return
        config["max_batch_mb"] = value
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _set_group_composite_enabled(self, group: DataGroup, checked: bool) -> None:
        config = data_group_composite_config(group)
        if bool(config.get("enabled", False)) == bool(checked):
            return
        config["enabled"] = bool(checked)
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_option(self, group: DataGroup, key: str, checked: bool) -> None:
        config = data_group_composite_config(group)
        if bool(config.get(key, False)) == bool(checked):
            return
        config[key] = bool(checked)
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_mean_weighting(self, group: DataGroup, value: str) -> None:
        config = data_group_composite_config(group)
        value = value if value in {"inverse_variance", "uniform"} else "inverse_variance"
        if _rebin_mean_weighting(config) == value:
            return
        config["mean_weighting"] = value
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_max_batch_mb(self, group: DataGroup, value: int) -> None:
        config = data_group_composite_config(group)
        value = max(int(value), 1)
        if _rebin_max_batch_mb(config) == value:
            return
        config["max_batch_mb"] = value
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_axis_value(self, group: DataGroup, index: int, key: str, text: str) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes")
        if not isinstance(axes, list) or not (0 <= index < len(axes)):
            return
        axis = dict(axes[index])
        try:
            if key == "num_bins":
                axis[key] = max(int(float(text)), 1)
            else:
                axis[key] = float(text)
        except ValueError:
            self._set_group_details(group)
            return
        if key == "step_size":
            step = float(axis.get("step_size", 0.0))
            if step > 0.0:
                axis["num_bins"] = _num_bins_from_step_size(axis.get("lower", 0.0), axis.get("upper", 0.0), step)
        axis.update(_sanitize_rebin_axis_config(axis))
        axes[index] = axis
        config["normalize"] = True
        self._after_group_composite_changed(group)

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

    def _after_group_composite_changed(self, group: DataGroup) -> None:
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self.refresh_slice_viewer(group)
        self._request_overlay_refresh(group)
        self._set_group_details(group)

    def _clear_details_panel(self) -> None:
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget is self.fit_settings_panel:
                    widget.setParent(None)
                else:
                    widget.setParent(None)
                    widget.deleteLater()

    def _restore_selected_fit_state(self, group: DataGroup, fit_entry: FitTimelineEntry) -> None:
        if not fit_entry.snapshot:
            fit_entry.snapshot = snapshot_data_group_state(group)
        self._set_active_fit_state(group, fit_entry)
        self._restoring_fit_selection = True
        try:
            restore_data_group_state(group, fit_entry.snapshot)
            # _refresh_tree already refreshes every open slice viewer for the
            # restored state; an explicit refresh here would recompute the
            # (expensive) overlay a second time.
            self._refresh_tree(select_group=group, select_fit=fit_entry)
        finally:
            self._restoring_fit_selection = False
        self._mark_dirty()

    def _active_fit_entry(self, group: DataGroup) -> FitTimelineEntry | None:
        if self._active_fit_group is not group:
            return None
        for entry in (
            self._active_branch_current,
            self._active_fit_current,
            self._active_fit_anchor,
        ):
            if _fit_entry_in_tree(group.fits, entry):
                return entry
        return None

    def _set_active_fit_state(self, group: DataGroup, fit_entry: FitTimelineEntry | None) -> None:
        self._active_fit_group = group
        self._active_fit_anchor = (
            fit_entry if fit_entry is not None and fit_entry.kind in {"initial", "result"} else None
        )
        self._active_fit_current = (
            fit_entry if fit_entry is not None and fit_entry.kind == "current" else None
        )
        self._active_branch_current = None
        # Remember which fit entry is active so it can be persisted and restored.
        group.active_fit_path = _fit_entry_path(group.fits, fit_entry)

    def _record_data_group_state_change(self, group: DataGroup) -> bool:
        """Update fit-history current state, creating an edit branch when needed.

        Returns True when the fit tree or active fit entry changed and callers
        should refresh the tree.
        """

        if self._restoring_fit_selection:
            return False
        ensure_fit_history(group)
        tree_changed = False
        current_entry: FitTimelineEntry | None = None
        if self._active_fit_group is group:
            if _fit_entry_in_tree(group.fits, self._active_fit_current):
                current_entry = self._active_fit_current
            elif _fit_entry_in_tree(group.fits, self._active_fit_anchor):
                if _is_editable_initial_baseline(group, self._active_fit_anchor):
                    _set_fit_current_snapshot(self._active_fit_anchor, group)
                    return False
                siblings = _fit_siblings(group.fits, self._active_fit_anchor)
                if (
                    self._active_fit_anchor.kind == "result"
                    and siblings is not None
                    and self._active_fit_anchor is _last_result_at_level(siblings)
                ):
                    current_entry, _created = _ensure_current_state_after_result(
                        group, self._active_fit_anchor
                    )
                    self._active_fit_current = current_entry
                    self._active_branch_current = None
                    tree_changed = True
                    if current_entry is not None:
                        group.active_fit_path = _fit_entry_path(group.fits, current_entry)
                elif self._active_fit_anchor.kind == "result":
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
                        self._active_fit_current = None
                        tree_changed = True
                        group.active_fit_path = _fit_entry_path(group.fits, current_entry)
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
        return tree_changed

    def _clear_active_fit_state(self) -> None:
        self._active_fit_group = None
        self._active_fit_anchor = None
        self._active_fit_current = None
        self._active_branch_current = None

    def _restore_active_fit_selection(self) -> bool:
        """Select and re-activate the fit entry persisted with the project."""

        for group in self.project.data_groups:
            entry = _fit_entry_at_path(group.fits, group.active_fit_path)
            if entry is None:
                continue
            self._set_active_fit_state(group, entry)
            self._refresh_tree(select_group=group, select_fit=entry)
            return True
        return False

    def _sync_fit_editor(self, fit_entry: FitTimelineEntry) -> None:
        self.fit_optimizer_combo.blockSignals(True)
        self.fit_optimizer_combo.setCurrentText(fit_entry.optimizer or "least_squares")
        self.fit_optimizer_combo.blockSignals(False)
        config = fit_entry.optimizer_config if isinstance(fit_entry.optimizer_config, dict) else {}
        self._sync_fit_control_values(config)
        self.fit_optimizer_config_editor.blockSignals(True)
        self.fit_optimizer_config_editor.setText(json.dumps(config, sort_keys=True))
        self.fit_optimizer_config_editor.blockSignals(False)

    def _sync_fit_control_values(self, config: dict[str, Any]) -> None:
        widgets = [
            self.fit_loss_combo,
            self.fit_f_scale_spin,
            self.fit_de_check,
            self.fit_de_maxiter_spin,
            self.fit_de_popsize_spin,
            self.fit_de_workers_spin,
            self.fit_emcee_check,
            self.fit_emcee_walkers_spin,
            self.fit_emcee_steps_spin,
            self.fit_emcee_burn_spin,
            self.fit_emcee_thin_spin,
            self.fit_emcee_workers_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.fit_loss_combo.setCurrentText(str(config.get("loss", "linear")))
        self.fit_f_scale_spin.setValue(float(config.get("f_scale", 1.0) or 1.0))
        initialization = config.get("initialization") if isinstance(config.get("initialization"), dict) else {}
        self.fit_de_check.setChecked(bool(initialization.get("enabled", False)))
        self.fit_de_maxiter_spin.setValue(int(initialization.get("maxiter", 60) or 60))
        self.fit_de_popsize_spin.setValue(int(initialization.get("popsize", 10) or 10))
        self.fit_de_workers_spin.setValue(int(initialization.get("workers", 1) or 1))
        sampler = config.get("sampler") if isinstance(config.get("sampler"), dict) else {}
        self.fit_emcee_check.setChecked(bool(sampler.get("enabled", False)))
        self.fit_emcee_walkers_spin.setValue(int(sampler.get("n_walkers", 0) or 0))
        self.fit_emcee_steps_spin.setValue(int(sampler.get("n_steps", 1000) or 1000))
        self.fit_emcee_burn_spin.setValue(int(sampler.get("burn_in", 200) or 0))
        self.fit_emcee_thin_spin.setValue(int(sampler.get("thin", 1) or 1))
        self.fit_emcee_workers_spin.setValue(int(sampler.get("workers", 1) or 1))
        for widget in widgets:
            widget.blockSignals(False)

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
            self._sync_fit_control_values(config)
            self._mark_dirty()

    def _set_selected_fit_controls_config(self, *args: Any) -> None:
        del args
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        config = self._fit_config_from_controls(fit_entry.optimizer_config)
        if fit_entry.optimizer_config != config:
            fit_entry.optimizer_config = config
            self.fit_optimizer_config_editor.blockSignals(True)
            self.fit_optimizer_config_editor.setText(json.dumps(config, sort_keys=True))
            self.fit_optimizer_config_editor.blockSignals(False)
            self._mark_dirty()

    def _fit_config_from_controls(self, existing: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(existing or {})
        loss = str(self.fit_loss_combo.currentText() or "linear")
        if loss == "linear":
            config.pop("loss", None)
            config.pop("f_scale", None)
        else:
            config["loss"] = loss
            config["f_scale"] = float(self.fit_f_scale_spin.value())
        if self.fit_de_check.isChecked():
            config["initialization"] = {
                "enabled": True,
                "method": "differential_evolution",
                "maxiter": int(self.fit_de_maxiter_spin.value()),
                "popsize": int(self.fit_de_popsize_spin.value()),
                "workers": int(self.fit_de_workers_spin.value()),
            }
        else:
            config.pop("initialization", None)
        if self.fit_emcee_check.isChecked():
            sampler = {
                "enabled": True,
                "method": "emcee",
                "n_steps": int(self.fit_emcee_steps_spin.value()),
                "burn_in": int(self.fit_emcee_burn_spin.value()),
                "thin": int(self.fit_emcee_thin_spin.value()),
                "workers": int(self.fit_emcee_workers_spin.value()),
            }
            walkers = int(self.fit_emcee_walkers_spin.value())
            if walkers > 0:
                sampler["n_walkers"] = walkers
            config["sampler"] = sampler
        else:
            config.pop("sampler", None)
        return config

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
        menu.setToolTipsVisible(True)
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
            "Fit now": self.start_fit_for_selection,
            "Enable": lambda: self._set_selected_enabled(True),
            "Disable": lambda: self._set_selected_enabled(False),
        }
        tooltips = {
            "Copy": "Copy the selected dataset or mask so it can be pasted elsewhere in the project.",
            "Paste": "Paste the copied dataset or mask into the selected compatible destination.",
            "Rename": "Rename the selected tree item.",
            "Delete": "Delete the selected item when that operation is allowed.",
            "View in data viewer": "Open or refresh the data viewer for this selection.",
            "Show file location": "Reveal the selected dataset's source file in the operating system file browser.",
            "Change file source": "Point this dataset at a different source file on disk.",
            "Add mask": "Create a new mask for the selected dataset or shared mask folder.",
            "New dataset group": "Create a nested dataset group under the selected workspace or group.",
            "Add model": "Create a new model component in the selected workspace.",
            "Fit now": "Run the optimizer from the selected fit state and store a new fit result.",
            "Enable": "Enable this item for viewing and fitting.",
            "Disable": "Disable this item for viewing and fitting.",
        }
        for name, enabled in self._context_menu_action_specs(item):
            action = menu.addAction(name)
            action.setEnabled(enabled)
            action.setToolTip(tooltips.get(name, name))
            action.setStatusTip(tooltips.get(name, name))
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
        self.window.setWindowTitle(f"nfit Project Explorer - {suffix}{marker}")

    def _mark_dirty(self) -> None:
        self.has_unsaved_changes = True
        self._sync_window_title()

    def _confirm_save_before_closing_project(self) -> bool:
        from PySide6 import QtWidgets

        if not self.has_unsaved_changes:
            return True
        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Question)
        message.setWindowTitle("Unsaved nfit project")
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
            empty_action.setToolTip("No recent nfit project files are available.")
            empty_action.setStatusTip("No recent nfit project files are available.")
            return
        for path in recent:
            action = self.recent_projects_menu.addAction(str(path))
            action.setEnabled(path.exists())
            tip = f"Open recent project: {path}" if path.exists() else f"Recent project file is missing: {path}"
            action.setToolTip(tip)
            action.setStatusTip(tip)
            action.triggered.connect(lambda _checked=False, path=path: self.open_project_path(path))
        if any(not path.exists() for path in recent):
            self.recent_projects_menu.addSeparator()
            action = self.recent_projects_menu.addAction("Remove missing projects", self._remove_missing_recent_projects)
            action.setToolTip("Remove recent-project entries whose files no longer exist.")
            action.setStatusTip("Remove recent-project entries whose files no longer exist.")

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
        additive_check.setToolTip("Add bins back into the accumulated nfit mask. This never overrides the file mask.")
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
            self._request_overlay_refresh(group)

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
            self._request_overlay_refresh(group)

    def _rebuild_model_parameter_editor(self, model: ModelComponentSpec) -> None:
        from PySide6 import QtWidgets

        self._clear_model_parameter_editor()
        fit_group = QtWidgets.QGroupBox("Fit Parameters")
        fit_group.setObjectName("model_fit_parameters_group")
        fit_layout = QtWidgets.QGridLayout(fit_group)
        fit_layout.setVerticalSpacing(4)
        fit_layout.setColumnStretch(1, 1)
        header_label = QtWidgets.QLabel("Plot label")
        header_min = QtWidgets.QLabel("Min")
        header_max = QtWidgets.QLabel("Max")
        plot_label_tooltip = (
            "Optional short label used in fit diagnostic plots. "
            "Accepts plain text or Matplotlib mathtext such as $\\Gamma$."
        )
        header_label.setToolTip(plot_label_tooltip)
        fit_layout.addWidget(header_label, 0, 2)
        fit_layout.addWidget(header_min, 0, 3)
        fit_layout.addWidget(header_max, 0, 4)
        for index, parameter_name in enumerate(model_parameter_names(model)):
            row = index + 1
            label = QtWidgets.QLabel(parameter_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(model.parameters.get(parameter_name, "")))
            parameter_labels = model.metadata.get("parameter_labels")
            plot_label_editor = QtWidgets.QLineEdit(
                str(parameter_labels.get(parameter_name, ""))
                if isinstance(parameter_labels, dict)
                else ""
            )
            plot_label_editor.setObjectName(f"model_parameter_plot_label_{parameter_name}")
            plot_label_editor.setPlaceholderText(f"p{index + 1}")
            lower_text, upper_text = _model_limit_texts(model, parameter_name)
            min_editor = QtWidgets.QLineEdit(lower_text)
            min_editor.setObjectName(f"model_parameter_min_{parameter_name}")
            min_editor.setPlaceholderText("-inf")
            min_editor.setMaximumWidth(70)
            max_editor = QtWidgets.QLineEdit(upper_text)
            max_editor.setObjectName(f"model_parameter_max_{parameter_name}")
            max_editor.setPlaceholderText("inf")
            max_editor.setMaximumWidth(70)
            fit_check = QtWidgets.QCheckBox("Fit")
            fit_check.setChecked(bool(model.fit_parameters.get(parameter_name, False)))
            global_check = QtWidgets.QCheckBox("Global fit")
            global_check.setChecked(bool(model.global_fit.get(parameter_name, True)))
            tooltip = model_parameter_tooltip(model.type, parameter_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            plot_label_editor.setToolTip(plot_label_tooltip)
            limits_tooltip = (
                "Optional bound applied when the optimizer varies this parameter. "
                "Leave blank for an unbounded side."
            )
            min_editor.setToolTip(limits_tooltip)
            max_editor.setToolTip(limits_tooltip)
            fit_check.setToolTip(
                f"Optimize parameter {parameter_name!r} during fitting. "
                "Unchecked parameters stay fixed at their current value."
            )
            global_check.setToolTip(
                f"Share parameter {parameter_name!r} across all fitted datasets. "
                "Uncheck to allow independent per-dataset values."
            )
            editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=editor: self._set_model_parameter(parameter_name, editor.text())
            )
            plot_label_editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=plot_label_editor: self._set_model_parameter_plot_label(parameter_name, editor.text())
            )
            min_editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=min_editor: self._set_model_limit(parameter_name, 0, editor.text())
            )
            max_editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=max_editor: self._set_model_limit(parameter_name, 1, editor.text())
            )
            fit_check.toggled.connect(
                lambda checked, parameter_name=parameter_name: self._set_model_fit_parameter(parameter_name, checked)
            )
            global_check.toggled.connect(
                lambda checked, parameter_name=parameter_name: self._set_model_global_fit(parameter_name, checked)
            )
            fit_layout.addWidget(label, row, 0)
            fit_layout.addWidget(editor, row, 1)
            fit_layout.addWidget(plot_label_editor, row, 2)
            fit_layout.addWidget(min_editor, row, 3)
            fit_layout.addWidget(max_editor, row, 4)
            fit_layout.addWidget(fit_check, row, 5)
            fit_layout.addWidget(global_check, row, 6)
        self.model_parameter_layout.addWidget(fit_group, 0, 0, 1, 4)

        scope_group = QtWidgets.QGroupBox("Dataset Scope")
        scope_group.setObjectName("model_dataset_scope_group")
        scope_layout = QtWidgets.QGridLayout(scope_group)
        scope_layout.setColumnStretch(1, 1)
        applies_label = QtWidgets.QLabel("Applies to")
        applies_editor = QtWidgets.QLineEdit(
            "" if model.applies_to is None else ", ".join(model.applies_to)
        )
        applies_editor.setObjectName("model_applies_to_editor")
        applies_tooltip = (
            "Comma-separated dataset names this model component fits. "
            "Leave blank to apply the model to every compatible dataset."
        )
        applies_label.setToolTip(applies_tooltip)
        applies_editor.setToolTip(applies_tooltip)
        applies_editor.setPlaceholderText("all compatible datasets")
        applies_editor.editingFinished.connect(
            lambda editor=applies_editor: self._set_model_applies_to(editor.text())
        )
        scope_layout.addWidget(applies_label, 0, 0)
        scope_layout.addWidget(applies_editor, 0, 1)
        self.model_parameter_layout.addWidget(scope_group, 1, 0, 1, 4)

        config_group = QtWidgets.QGroupBox("Configuration Settings")
        config_group.setObjectName("model_config_group")
        config_layout = QtWidgets.QGridLayout(config_group)
        config_layout.setColumnStretch(1, 1)
        config_definitions = MODEL_TYPE_DEFINITIONS[model.type].get("config", {})
        if not config_definitions:
            config_layout.addWidget(QtWidgets.QLabel("No configuration settings."), 0, 0, 1, 2)
        row = 0
        for setting_name in config_definitions:
            if setting_name == "form_factor_coefficients":
                continue
            label = QtWidgets.QLabel(setting_name)
            tooltip = model_config_tooltip(model.type, setting_name)
            label.setToolTip(tooltip)
            if config_definitions[setting_name].get("choices") == "form_factor_ions":
                label.setText("form_factor")
                combo = QtWidgets.QComboBox()
                combo.setObjectName(f"model_config_choice_{setting_name}")
                combo.addItem("(none)", "")
                for ion in available_ions():
                    combo.addItem(ion, ion)
                combo.addItem("Custom...", CUSTOM_FORM_FACTOR_CHOICE)
                current = str(model.config.get(setting_name, "") or "")
                if str(model.config.get("form_factor_coefficients", "") or "").strip():
                    current = CUSTOM_FORM_FACTOR_CHOICE
                combo.setCurrentIndex(max(combo.findData(current), 0))
                combo.setToolTip(tooltip)
                combo.currentIndexChanged.connect(
                    lambda _index, combo=combo: self._set_model_form_factor_choice(
                        str(combo.currentData() or "")
                    )
                )
                config_layout.addWidget(label, row, 0)
                config_layout.addWidget(combo, row, 1)
                row += 1
                if current == CUSTOM_FORM_FACTOR_CHOICE:
                    coeff_tooltip = model_config_tooltip(model.type, "form_factor_coefficients")
                    coeff_label = QtWidgets.QLabel("custom coefficients")
                    coeff_label.setToolTip(coeff_tooltip)
                    coeff_editor = QtWidgets.QLineEdit(
                        _parameter_to_text(model.config.get("form_factor_coefficients", ""))
                    )
                    coeff_editor.setObjectName("model_config_form_factor_coefficients")
                    coeff_editor.setToolTip(coeff_tooltip)
                    coeff_editor.editingFinished.connect(
                        lambda editor=coeff_editor: self._set_model_config_setting(
                            "form_factor_coefficients", editor.text()
                        )
                    )
                    config_layout.addWidget(coeff_label, row, 0)
                    config_layout.addWidget(coeff_editor, row, 1)
                    row += 1
                continue
            editor = QtWidgets.QLineEdit(_parameter_to_text(model.config.get(setting_name, "")))
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda setting_name=setting_name, editor=editor: self._set_model_config_setting(setting_name, editor.text())
            )
            config_layout.addWidget(label, row, 0)
            config_layout.addWidget(editor, row, 1)
            row += 1
        self.model_parameter_layout.addWidget(config_group, 2, 0, 1, 4)
        if MODEL_TYPE_DEFINITIONS[model.type].get("structured_config"):
            self._build_model_crystal_editor(model)

    def _clear_model_parameter_editor(self) -> None:
        while self.model_parameter_layout.count():
            item = self.model_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _build_model_crystal_editor(self, model: ModelComponentSpec) -> None:
        """Structured crystal / magnetic-site / bond-orbit editor.

        Shown for model types flagged ``structured_config`` (heisenberg_rpa).
        Everything edits plain JSON data in ``model.config`` so the state
        serializes with the project and drives the fit factory directly.
        """

        from PySide6 import QtWidgets

        crystal = model_crystal_config(model)
        lattice = crystal["lattice"]

        crystal_group = QtWidgets.QGroupBox("Crystal")
        crystal_group.setObjectName("model_crystal_group")
        crystal_layout = QtWidgets.QGridLayout(crystal_group)
        lattice_tooltip = (
            "Unit-cell parameter used to build exchange bonds and convert HKL "
            "to |Q|. Lengths in Angstrom, angles in degrees."
        )
        for column, name in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            label = QtWidgets.QLabel(name)
            label.setToolTip(lattice_tooltip)
            editor = QtWidgets.QLineEdit(_parameter_to_text(lattice.get(name, "")))
            editor.setObjectName(f"model_crystal_{name}")
            editor.setToolTip(lattice_tooltip)
            editor.setMaximumWidth(70)
            editor.editingFinished.connect(
                lambda name=name, editor=editor: self._set_model_crystal_lattice(name, editor.text())
            )
            crystal_layout.addWidget(label, 0, 2 * column)
            crystal_layout.addWidget(editor, 0, 2 * column + 1)
        spacegroup_label = QtWidgets.QLabel("Space group")
        spacegroup_tooltip = (
            "Hermann-Mauguin space group symbol (e.g. 'F d -3 m:2'). Used to "
            "expand the magnetic sites and to group bonds into symmetry-"
            "distinct orbits sharing one exchange constant."
        )
        spacegroup_label.setToolTip(spacegroup_tooltip)
        spacegroup_editor = QtWidgets.QLineEdit(str(crystal.get("spacegroup", "P 1")))
        spacegroup_editor.setObjectName("model_crystal_spacegroup")
        spacegroup_editor.setToolTip(spacegroup_tooltip)
        spacegroup_editor.editingFinished.connect(
            lambda editor=spacegroup_editor: self._set_model_crystal_spacegroup(editor.text())
        )
        crystal_layout.addWidget(spacegroup_label, 1, 0, 1, 2)
        crystal_layout.addWidget(spacegroup_editor, 1, 2, 1, 4)
        import_button = QtWidgets.QPushButton("Import CIF...")
        import_button.setObjectName("model_crystal_import_cif")
        import_button.setToolTip(
            "Load lattice, space group, and atomic sites from a CIF file into "
            "this model (and offer them to the data group). Clears previously "
            "generated bond orbits."
        )
        import_button.clicked.connect(self._import_cif_into_selected_model)
        crystal_layout.addWidget(import_button, 1, 6, 1, 2)
        from_group_button = QtWidgets.QPushButton("Use group crystal")
        from_group_button.setObjectName("model_crystal_from_group")
        from_group_button.setToolTip(
            "Copy the crystal (lattice, space group, sites) stored on this "
            "dataset group into the model configuration."
        )
        from_group_button.clicked.connect(self._use_group_crystal_for_selected_model)
        crystal_layout.addWidget(from_group_button, 1, 8, 1, 2)
        self.model_parameter_layout.addWidget(crystal_group, 3, 0, 1, 4)

        sites_group = QtWidgets.QGroupBox("Atomic Sites")
        sites_group.setObjectName("model_crystal_sites_group")
        sites_layout = QtWidgets.QGridLayout(sites_group)
        magnetic_labels = {str(name) for name in model.config.get("magnetic_sites", [])}
        for header_column, header in enumerate(("Label", "x", "y", "z", "Ion", "", "")):
            if header:
                sites_layout.addWidget(QtWidgets.QLabel(header), 0, header_column)
        site_tooltip = (
            "Wyckoff site of the crystal: label, fractional coordinates, and "
            "the magnetic ion for the <j0> form factor. Check 'Magnetic' to "
            "include the site in exchange-bond generation."
        )
        for index, site in enumerate(crystal["sites"]):
            row = index + 1
            label_editor = QtWidgets.QLineEdit(str(site.get("label", "")))
            label_editor.setObjectName(f"model_crystal_site_label_{index}")
            label_editor.setToolTip(site_tooltip)
            label_editor.editingFinished.connect(
                lambda index=index, editor=label_editor: self._set_model_crystal_site(index, "label", editor.text())
            )
            sites_layout.addWidget(label_editor, row, 0)
            position = site.get("position", [0.0, 0.0, 0.0])
            for axis in range(3):
                editor = QtWidgets.QLineEdit(_parameter_to_text(position[axis]))
                editor.setObjectName(f"model_crystal_site_{index}_{'xyz'[axis]}")
                editor.setToolTip(site_tooltip)
                editor.setMaximumWidth(70)
                editor.editingFinished.connect(
                    lambda index=index, axis=axis, editor=editor: self._set_model_crystal_site(index, axis, editor.text())
                )
                sites_layout.addWidget(editor, row, 1 + axis)
            ion_combo = QtWidgets.QComboBox()
            ion_combo.setObjectName(f"model_crystal_site_ion_{index}")
            ion_combo.setToolTip(
                "Magnetic ion of this site; sets the tabulated <j0> form "
                "factor key. '(none)' leaves the site without a form factor."
            )
            ion_combo.addItem("(none)", "")
            for ion in available_ions():
                ion_combo.addItem(ion, ion)
            ion_combo.setCurrentIndex(max(ion_combo.findData(str(site.get("ion", "") or "")), 0))
            ion_combo.currentIndexChanged.connect(
                lambda _index, index=index, combo=ion_combo: self._set_model_crystal_site(index, "ion", str(combo.currentData() or ""))
            )
            sites_layout.addWidget(ion_combo, row, 4)
            magnetic_check = QtWidgets.QCheckBox("Magnetic")
            magnetic_check.setObjectName(f"model_crystal_site_magnetic_{index}")
            magnetic_check.setToolTip(
                "Include this site in magnetic-site expansion and exchange-"
                "bond generation."
            )
            magnetic_check.setChecked(str(site.get("label", "")) in magnetic_labels)
            magnetic_check.toggled.connect(
                lambda checked, index=index: self._set_model_site_magnetic(index, checked)
            )
            sites_layout.addWidget(magnetic_check, row, 5)
            remove_button = QtWidgets.QPushButton("Remove")
            remove_button.setObjectName(f"model_crystal_site_remove_{index}")
            remove_button.setToolTip("Remove this atomic site from the crystal.")
            remove_button.clicked.connect(
                lambda _checked=False, index=index: self._remove_model_crystal_site(index)
            )
            sites_layout.addWidget(remove_button, row, 6)
        add_site_button = QtWidgets.QPushButton("Add site")
        add_site_button.setObjectName("model_crystal_site_add")
        add_site_button.setToolTip("Append a new atomic site to the crystal.")
        add_site_button.clicked.connect(self._add_model_crystal_site)
        sites_layout.addWidget(add_site_button, len(crystal["sites"]) + 1, 0)
        self.model_parameter_layout.addWidget(sites_group, 4, 0, 1, 4)

        bonds_group = QtWidgets.QGroupBox("Exchange Bonds")
        bonds_group.setObjectName("model_bonds_group")
        bonds_layout = QtWidgets.QGridLayout(bonds_group)
        cutoff_label = QtWidgets.QLabel("Bond cutoff (A)")
        cutoff_tooltip = (
            "Maximum bond length in Angstrom when enumerating exchange "
            "paths. Each symmetry-distinct orbit within the cutoff becomes "
            "one exchange fit parameter (J1, J2, J3a, ...)."
        )
        cutoff_label.setToolTip(cutoff_tooltip)
        cutoff_editor = QtWidgets.QLineEdit(
            _parameter_to_text(model.config.get("bond_cutoff_angstrom", DEFAULT_BOND_CUTOFF_ANGSTROM))
        )
        cutoff_editor.setObjectName("model_bonds_cutoff")
        cutoff_editor.setToolTip(cutoff_tooltip)
        cutoff_editor.setMaximumWidth(70)
        cutoff_editor.editingFinished.connect(
            lambda editor=cutoff_editor: self._set_model_config_setting("bond_cutoff_angstrom", editor.text())
        )
        bonds_layout.addWidget(cutoff_label, 0, 0)
        bonds_layout.addWidget(cutoff_editor, 0, 1)
        generate_button = QtWidgets.QPushButton("Generate symmetry orbits")
        generate_button.setObjectName("model_bonds_generate")
        generate_button.setToolTip(
            "Expand the magnetic sites through the space group, enumerate "
            "bonds up to the cutoff, and group them into symmetry-distinct "
            "orbits. Each orbit becomes one exchange parameter; values of "
            "orbits whose labels persist are kept."
        )
        generate_button.clicked.connect(self._generate_selected_model_bond_orbits)
        bonds_layout.addWidget(generate_button, 0, 2)
        orbits = model.config.get("orbits") or []
        table = QtWidgets.QTableWidget(len(orbits), 4)
        table.setObjectName("model_bonds_table")
        table.setToolTip(
            "Symmetry-distinct bond orbits of the current crystal. Each row "
            "is one exchange fit parameter; multiplicity counts the bonds "
            "sharing that constant."
        )
        table.setHorizontalHeaderLabels(["Orbit", "Distance (A)", "Multiplicity", "Example bond"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, orbit in enumerate(orbits):
            bonds = orbit.get("bonds", [])
            example = ""
            if bonds:
                bond = bonds[0]
                example = (
                    f"site {bond.get('site_i')} -> site {bond.get('site_j')} "
                    f"+ {tuple(bond.get('offset', (0, 0, 0)))}"
                )
            for column, text in enumerate(
                (
                    str(orbit.get("label", "")),
                    _format_number(float(orbit.get("distance_angstrom", 0.0))),
                    str(len(bonds)),
                    example,
                )
            ):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        visible_rows = min(max(len(orbits), 4), 10)
        row_height = max(table.verticalHeader().defaultSectionSize(), 24)
        table_height = (
            table.horizontalHeader().height()
            + row_height * visible_rows
            + 2 * table.frameWidth()
            + 8
        )
        table.setMinimumHeight(table_height)
        table.setMaximumHeight(table_height)
        bonds_layout.addWidget(table, 1, 0, 1, 3)
        self.model_parameter_layout.addWidget(bonds_group, 5, 0, 1, 4)

    def _selected_model_and_group(self) -> tuple[DataGroup | None, ModelComponentSpec | None]:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return None, None
        return group, model

    def _mutate_selected_model(self, mutate) -> None:
        """Apply a config mutation to the selected model and refresh the editor."""

        group, model = self._selected_model_and_group()
        if model is None:
            return
        try:
            mutate(model, group)
        except (ValueError, ImportError, KeyError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(self.window, "Model configuration", str(exc))
            return
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_model_parameter_editor(model)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            self._request_overlay_refresh(group)

    def _set_model_crystal_lattice(self, name: str, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            lattice = model_crystal_config(model)["lattice"]
            value = float(_parse_parameter_text(text))
            if lattice.get(name) == value:
                raise _NoChange()
            lattice[name] = value

        self._mutate_selected_model_quietly(mutate)

    def _set_model_crystal_spacegroup(self, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            value = text.strip() or "P 1"
            if crystal.get("spacegroup") == value:
                raise _NoChange()
            crystal["spacegroup"] = value

        self._mutate_selected_model_quietly(mutate)

    def _set_model_crystal_site(self, index: int, field: Any, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            sites = model_crystal_config(model)["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            site = sites[index]
            if field == "label":
                value = text.strip()
                if site.get("label") == value:
                    raise _NoChange()
                magnetic = [str(name) for name in model.config.get("magnetic_sites", [])]
                model.config["magnetic_sites"] = [
                    value if name == str(site.get("label", "")) else name for name in magnetic
                ]
                site["label"] = value
            elif field == "ion":
                if site.get("ion") == text:
                    raise _NoChange()
                site["ion"] = text
            else:
                position = list(site.get("position", [0.0, 0.0, 0.0]))
                value = float(_parse_parameter_text(text))
                if position[int(field)] == value:
                    raise _NoChange()
                position[int(field)] = value
                site["position"] = position

        self._mutate_selected_model_quietly(mutate)

    def _set_model_site_magnetic(self, index: int, checked: bool) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            sites = model_crystal_config(model)["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            label = str(sites[index].get("label", ""))
            magnetic = [str(name) for name in model.config.get("magnetic_sites", [])]
            if bool(checked) == (label in magnetic):
                raise _NoChange()
            if checked:
                magnetic.append(label)
            else:
                magnetic = [name for name in magnetic if name != label]
            model.config["magnetic_sites"] = magnetic

        self._mutate_selected_model_quietly(mutate)

    def _add_model_crystal_site(self) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            sites = model_crystal_config(model)["sites"]
            sites.append(
                {
                    "label": f"Site{len(sites) + 1}",
                    "position": [0.0, 0.0, 0.0],
                    "ion": "",
                }
            )

        self._mutate_selected_model(mutate)

    def _remove_model_crystal_site(self, index: int) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            sites = model_crystal_config(model)["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            removed = sites.pop(index)
            label = str(removed.get("label", ""))
            model.config["magnetic_sites"] = [
                str(name) for name in model.config.get("magnetic_sites", []) if str(name) != label
            ]

        self._mutate_selected_model(mutate)

    def _import_cif_into_selected_model(self) -> None:
        from PySide6 import QtWidgets

        group, model = self._selected_model_and_group()
        if model is None:
            return
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Import CIF", "", "CIF files (*.cif);;All files (*)"
        )
        if not path:
            return

        def mutate(model: ModelComponentSpec, group: DataGroup | None) -> None:
            import_cif_into_model(model, path, group=group)

        self._mutate_selected_model(mutate)

    def _use_group_crystal_for_selected_model(self) -> None:
        def mutate(model: ModelComponentSpec, group: DataGroup | None) -> None:
            if group is None:
                raise ValueError("the model is not attached to a data group")
            stored = group.metadata.get("crystal")
            if isinstance(stored, dict) and stored.get("sites"):
                model.config["crystal"] = copy.deepcopy(stored)
            elif isinstance(group.lattice_parameters, dict):
                crystal = model_crystal_config(model)
                crystal["lattice"] = {
                    name: float(group.lattice_parameters.get(name, fallback))
                    for name, fallback in (
                        ("a", 5.0), ("b", 5.0), ("c", 5.0),
                        ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0),
                    )
                }
                if group.spacegroup:
                    crystal["spacegroup"] = str(group.spacegroup)
            else:
                raise ValueError(
                    "the data group stores no crystal information; import a "
                    "CIF or set lattice parameters on the group first"
                )
            model.config.pop("orbits", None)
            model.config.pop("site_positions", None)
            reconcile_model_orbit_parameters(model)

        self._mutate_selected_model(mutate)

    def _generate_selected_model_bond_orbits(self) -> None:
        from PySide6 import QtWidgets

        cutoff_editor = self.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_bonds_cutoff"
        )

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if cutoff_editor is not None:
                model.config["bond_cutoff_angstrom"] = _parse_parameter_text(
                    cutoff_editor.text()
                )
            generate_model_bond_orbits(model)

        self._mutate_selected_model(mutate)

    def _mutate_selected_model_quietly(self, mutate) -> None:
        """Like ``_mutate_selected_model`` but a ``_NoChange`` is not an error."""

        def wrapped(model: ModelComponentSpec, group: DataGroup | None) -> None:
            mutate(model, group)

        try:
            self._mutate_selected_model(wrapped)
        except _NoChange:
            return

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
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_parameter_plot_label(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        labels = dict(model.metadata.get("parameter_labels", {}))
        value = text.strip()
        if value:
            labels[name] = value
        else:
            labels.pop(name, None)
        if model.metadata.get("parameter_labels", {}) != labels:
            if labels:
                model.metadata["parameter_labels"] = labels
            else:
                model.metadata.pop("parameter_labels", None)
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)

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
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_form_factor_choice(self, choice: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        choice = str(choice or "")
        changed = False
        if choice == CUSTOM_FORM_FACTOR_CHOICE:
            if model.config.get("ion") != CUSTOM_FORM_FACTOR_CHOICE:
                model.config["ion"] = CUSTOM_FORM_FACTOR_CHOICE
                changed = True
            model.config.setdefault("form_factor_coefficients", "")
        else:
            if model.config.get("ion") != choice:
                model.config["ion"] = choice
                changed = True
            if model.config.get("form_factor_coefficients"):
                model.config["form_factor_coefficients"] = ""
                changed = True
        if changed:
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            self._rebuild_model_parameter_editor(model)
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

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
        if group is not None:
            self._request_overlay_refresh(group)

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
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_limit(self, name: str, side: int, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        raw = _parse_parameter_text(text.strip()) if text.strip() else None
        value = None if raw in (None, "") else raw
        current = model.limits.get(name)
        limits = list(current) if isinstance(current, (list, tuple)) and len(current) == 2 else [None, None]
        if limits[side] == value:
            return
        limits[side] = value
        if limits == [None, None]:
            model.limits.pop(name, None)
        else:
            model.limits[name] = limits
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_applies_to(self, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        names = [part.strip() for part in text.split(",") if part.strip()]
        value = names or None
        if model.applies_to == value:
            return
        model.applies_to = value
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def refresh_open_slice_viewers(self) -> None:
        # Debounced: a tree refresh (selection change, timeline switch, fit
        # completion, ...) schedules the overlay recompute instead of blocking
        # on it. In headless/test use this runs synchronously.
        for group in list(self.project.data_groups):
            self._request_overlay_refresh(group)

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
            viewer.window.setWindowTitle(f"nfit Data Viewer - {group.name}")
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
        def __init__(self, explorer: NfitProjectExplorer) -> None:
            super().__init__()
            self.explorer = explorer

        def closeEvent(self, event):
            self.explorer._handle_window_close(event)

    return ProjectWindow


def _make_project_tree_class():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ProjectTree(QtWidgets.QTreeWidget):
        def __init__(self, explorer: NfitProjectExplorer) -> None:
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
            mime_data.setData("application/x-nfit-tree-item", role.encode("utf-8"))
            drag.setMimeData(mime_data)
            drag.exec(QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction)

        def dragEnterEvent(self, event):
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragEnterEvent(event)

        def dragMoveEvent(self, event):
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragMoveEvent(event)

        def dropEvent(self, event):
            # Internal item drags: always handled here, never by the default
            # QTreeWidget item-move machinery.
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
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
    for dataset in group.iter_datasets():
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


def _dataset_data_point_count(dataset: DatasetEntry) -> int:
    data = dataset.data
    if isinstance(data, MDHistoData):
        return int(np.prod(data.shape))
    if isinstance(data, PointListData):
        return int(data.size)
    if isinstance(data, PointData4D):
        return int(data.size)
    return 0


def _group_data_point_count(group: DataGroup) -> int:
    return sum(_dataset_data_point_count(dataset) for dataset in group.iter_datasets())


def _group_dataset_type_summary(group: DataGroup) -> str:
    counts: dict[str, int] = {}
    for dataset in group.iter_datasets():
        label = data_type_label(dataset.data_type)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "-"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


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


def _mdhisto_fit_bin_count(view: MDHistoData) -> int:
    """Count fit-eligible bins without materializing coordinate grids.

    Matches ``_point_data_from_mdhisto_view(view).valid_mask()``: an MDHisto
    grid's H/K/L/E bin centers are always finite, so only the mask, event count,
    and finite/positive intensity and error need checking. This avoids building
    the (potentially tens of millions of points) coordinate meshgrid just to
    report a count.
    """

    keep = ~np.asarray(view.mask, dtype=bool)
    keep &= np.asarray(view.num_events, dtype=float) > 0.0
    keep &= np.isfinite(np.asarray(view.signal, dtype=float))
    errors = np.asarray(view.errors, dtype=float)
    keep &= np.isfinite(errors)
    keep &= errors > 0.0
    return int(np.count_nonzero(keep))


def _dataset_fit_summary_lines(
    dataset: DatasetEntry,
    *,
    group: DataGroup | None = None,
) -> list[str]:
    """Return fit-eligible bin/point counts using the same masks as fitting."""

    if dataset.data is None:
        return []
    extra_masks = effective_dataset_masks(group, dataset) if group is not None else []
    try:
        view = dataset_for_slice_viewer(dataset, extra_masks=extra_masks)
        if isinstance(view, MDHistoData):
            fit_bins = _mdhisto_fit_bin_count(view)
            total_bins = int(np.prod(view.shape))
            return [f"Fit bins: {fit_bins} of {total_bins}"]
        if isinstance(view, PointListData):
            points = _point_data_from_point_list_view(view)
            fit_points = int(np.count_nonzero(points.valid_mask()))
            return [f"Fit points: {fit_points} of {points.size}"]
        if isinstance(view, PointData4D):
            fit_points = int(np.count_nonzero(view.valid_mask()))
            return [f"Fit points: {fit_points} of {view.size}"]
    except Exception as exc:
        label = "Fit bins" if isinstance(dataset.data, MDHistoData) else "Fit points"
        return [f"{label}: unavailable ({exc})"]
    return []


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


def _tooltip_table_corner_buttons(table: Any, tooltip: str) -> None:
    """Give a table's internal Qt corner/select buttons a tooltip.

    QTableWidget's corner "select all" button is an untooltipped
    ``QAbstractButton``; label it so it does not read as a bare control.
    """

    from PySide6 import QtWidgets

    for button in table.findChildren(QtWidgets.QAbstractButton):
        if not button.toolTip():
            button.setToolTip(tooltip)


def _snapshot_parameter_rows(fit_entry: FitTimelineEntry) -> list[dict[str, str]]:
    """Return current parameter values from a fit entry's snapshot models.

    Used for the Current state (and Initial) entries, which carry live model
    parameter values but no optimizer results.
    """

    snapshot = fit_entry.snapshot if isinstance(fit_entry.snapshot, dict) else {}
    models = snapshot.get("models", [])
    if not isinstance(models, list):
        return []
    rows: list[dict[str, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        component = str(model.get("name", ""))
        params = model.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for name, value in params.items():
            label = f"{component}.{name}" if component else str(name)
            rows.append({"name": label, "value": _format_number(value)})
    return rows


def _fit_results_rows(fit_entry: FitTimelineEntry) -> list[dict[str, str]]:
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = goodness.get("parameters")
    if not isinstance(params, dict) or not params:
        return []
    stderr = goodness.get("stderr") if isinstance(goodness.get("stderr"), dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    posterior_params = (
        posterior.get("parameters")
        if isinstance(posterior.get("parameters"), dict)
        else {}
    )
    rows: list[dict[str, str]] = []
    for name, value in params.items():
        posterior_row = (
            posterior_params.get(name)
            if isinstance(posterior_params.get(name), dict)
            else {}
        )
        rows.append(
            {
                "name": str(name),
                "value": _format_number(value),
                "stderr": _format_number(stderr[name]) if name in stderr else "-",
                "median": (
                    _format_number(posterior_row["median"])
                    if "median" in posterior_row
                    else "-"
                ),
                "p16": _format_number(posterior_row["p16"]) if "p16" in posterior_row else "-",
                "p84": _format_number(posterior_row["p84"]) if "p84" in posterior_row else "-",
            }
        )
    return rows


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


def _format_seconds_per_step(value: float) -> str:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds < 0.0:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.3g} ms"
    return f"{seconds:.3g} s"


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


def _project_to_dict(project: NfitProject) -> dict[str, Any]:
    return {
        "format": "nfit-project",
        "version": 1,
        "settings": _json_mapping(project.settings),
        "data_groups": [_data_group_to_dict(group) for group in project.data_groups],
    }


def _project_from_dict(payload: dict[str, Any]) -> NfitProject:
    if payload.get("format") != "nfit-project":
        raise ValueError("not a nfit project file")
    project = NfitProject(settings=dict(payload.get("settings", {})))
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
                    # defaults first, then every saved flag: config-derived
                    # parameter names (e.g. J1 of generated bond orbits) are
                    # not in the static defaults and must survive a load
                    **default_model_fit_parameters(model_type),
                    **{str(name): bool(value) for name, value in fit_payload.items()},
                },
                global_fit={
                    str(name): bool(value)
                    for name, value in dict(model_payload.get("global_fit", {})).items()
                },
                sharing=_sharing_from_payload(model_payload.get("sharing")),
                limits=dict(model_payload.get("limits", {}) or {}),
                constraints=[
                    dict(constraint)
                    for constraint in model_payload.get("constraints", []) or []
                ],
                applies_to=_applies_to_from_payload(model_payload.get("applies_to")),
                enabled=bool(model_payload.get("enabled", True)),
                metadata=dict(model_payload.get("metadata", {})),
            )
            group.models[model.name] = model
        group.fits = [
            _fit_entry_from_dict(fit_payload)
            for fit_payload in group_payload.get("fits", [])
        ]
        active_path = group_payload.get("active_fit_path")
        if isinstance(active_path, list) and all(isinstance(index, int) for index in active_path):
            group.active_fit_path = list(active_path)
        ensure_fit_history(group)
        project.data_groups.append(group)
    return project


def _model_limit_texts(model: ModelComponentSpec, parameter_name: str) -> tuple[str, str]:
    """Return display texts for a parameter's (min, max) bounds."""

    raw = model.limits.get(parameter_name)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return "", ""
    lower, upper = raw
    return (
        "" if lower in (None, "") else _parameter_to_text(lower),
        "" if upper in (None, "") else _parameter_to_text(upper),
    )


def _sharing_from_payload(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    sharing: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        if isinstance(entry, dict):
            sharing[str(name)] = {
                "mode": str(entry.get("mode", "global")),
                "groups": {
                    str(dataset): str(key)
                    for dataset, key in dict(entry.get("groups", {})).items()
                },
            }
    return sharing


def _applies_to_from_payload(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(name) for name in value]
    return None


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
        "active_fit_path": (
            list(group.active_fit_path) if group.active_fit_path is not None else None
        ),
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
        "sharing": _json_mapping(model.sharing),
        "limits": _json_mapping(model.limits),
        "constraints": [dict(constraint) for constraint in model.constraints],
        "applies_to": None if model.applies_to is None else list(model.applies_to),
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
        "channels": _fit_channels_to_dict(fit_entry.channels),
        "children": [_fit_entry_to_dict(child) for child in fit_entry.children],
        "metadata": _json_mapping(fit_entry.metadata),
    }


def _fit_channels_to_dict(channels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset_name, entry in (channels or {}).items():
        encoded: dict[str, Any] = {"kind": str(entry.get("kind", "points"))}
        for channel_name in FIT_CHANNEL_NAMES:
            if entry.get(channel_name) is not None:
                encoded[channel_name] = _encode_float_array(entry[channel_name])
        payload[str(dataset_name)] = encoded
    return payload


def _fit_channels_from_dict(payload: Any) -> dict[str, dict[str, Any]]:
    channels: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return channels
    for dataset_name, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        decoded: dict[str, Any] = {"kind": str(entry.get("kind", "points"))}
        for channel_name in FIT_CHANNEL_NAMES:
            array = _fit_channel_array(entry.get(channel_name))
            if array is not None:
                decoded[channel_name] = array
        channels[str(dataset_name)] = decoded
    return channels


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
        channels=_fit_channels_from_dict(payload.get("channels")),
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


def _install_cli_interrupt_handler(app: Any) -> tuple[Any | None, Any | None]:
    """Let terminal Ctrl+C interrupt the Qt event loop."""

    from PySide6 import QtCore

    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda signum, _frame: app.exit(128 + signum))
    except ValueError:
        return None, None

    interrupt_timer = QtCore.QTimer()
    interrupt_timer.setInterval(200)
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start()
    return interrupt_timer, previous_handler


def _restore_cli_interrupt_handler(previous_handler: Any | None) -> None:
    if previous_handler is None:
        return
    try:
        signal.signal(signal.SIGINT, previous_handler)
    except ValueError:
        return


def _settings():
    from PySide6 import QtCore

    return QtCore.QSettings("nfit", "nfit")


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
    """Launch the nfit project explorer."""

    return NfitProjectExplorer().run()
