from __future__ import annotations

import json
import copy
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoData, load_mantid_mdhisto_nxs
from .pipeline import DataGroup, DatasetEntry, FitTimelineEntry, MaskSpec, ModelComponentSpec

QtMDHistoSliceViewer = None
RECENT_PROJECT_LIMIT = 10
RECENT_PROJECTS_KEY = "recent_projects"


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
                "default": [-1.0e99, 1.0e99],
                "description": "Magnitude of Q interval for an energy-|Q| exclusion mask.",
                "allowed": "Two numbers [min, max] in inverse angstrom.",
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


def import_dataset_paths(group: DataGroup, paths: list[str | Path]) -> list[DatasetEntry]:
    """Add placeholder dataset entries for one or more source files."""

    entries: list[DatasetEntry] = []
    for path in paths:
        source = Path(path)
        entry = dataset_entry_from_path(source)
        entry.name = _unique_dataset_name(entry.name, group.dataset_names)
        group.add_dataset(entry)
        entries.append(entry)
    return entries


def delete_data_group(project: MetallixProject, group: DataGroup) -> None:
    """Remove a data group from a project."""

    project.data_groups.remove(group)


def delete_dataset(group: DataGroup, dataset: DatasetEntry) -> None:
    """Remove a dataset entry from a data group."""

    group.datasets.remove(dataset)


def set_dataset_source(dataset: DatasetEntry, path: str | Path) -> None:
    """Point a dataset entry at a new source file and mark loaded data stale."""

    source = Path(path)
    dataset.metadata["source_file"] = str(source)
    dataset.metadata["import_status"] = "pending"
    dataset.kind = source.suffix.lstrip(".").lower()
    dataset.data = None


def create_mask(dataset: DatasetEntry, name: str | None = None, *, type: str = "coordinate_range") -> MaskSpec:
    """Add a mask spec to a dataset and return it."""

    if type not in MASK_TYPE_DEFINITIONS:
        raise ValueError(f"unknown mask type {type!r}")
    mask = MaskSpec(
        name=next_mask_name(dataset.masks) if name is None else name,
        type=type,
        parameters=default_mask_parameters(type),
    )
    if mask.name in {existing.name for existing in dataset.masks}:
        raise ValueError(f"duplicate mask name {mask.name!r}")
    dataset.masks.append(mask)
    return mask


def delete_mask(dataset: DatasetEntry, mask: MaskSpec) -> None:
    """Remove a mask from a dataset."""

    dataset.masks.remove(mask)


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


def snapshot_data_group_state(group: DataGroup) -> dict[str, Any]:
    """Capture serializable dataset mask and model configuration state."""

    return {
        "datasets": [
            {
                "name": dataset.name,
                "parameters": copy.deepcopy(dataset.parameters),
                "masks": [_mask_to_dict(mask) for mask in dataset.masks],
            }
            for dataset in group.datasets
        ],
        "models": [
            _model_to_dict(model)
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
        ],
    }


def restore_data_group_state(group: DataGroup, snapshot: dict[str, Any]) -> None:
    """Restore dataset masks and model component settings from a snapshot."""

    datasets_by_name = {dataset.name: dataset for dataset in group.datasets}
    for dataset_payload in snapshot.get("datasets", []):
        dataset = datasets_by_name.get(str(dataset_payload.get("name", "")))
        if dataset is None:
            continue
        dataset.parameters = dict(dataset_payload.get("parameters", {}))
        dataset.masks = [
            MaskSpec(
                name=str(mask_payload["name"]),
                type=str(mask_payload.get("type", "coordinate_range")),
                parameters=dict(mask_payload.get("parameters", {})),
                enabled=bool(mask_payload.get("enabled", True)),
                metadata=dict(mask_payload.get("metadata", {})),
            )
            for mask_payload in dataset_payload.get("masks", [])
        ]
    for model_payload in snapshot.get("models", []):
        name = str(model_payload.get("name", ""))
        existing = group.models.get(name)
        if not isinstance(existing, ModelComponentSpec):
            continue
        existing.type = str(model_payload.get("type", existing.type))
        existing.parameters = dict(model_payload.get("parameters", {}))
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


def mask_parameter_tooltip(type: str, parameter_name: str) -> str:
    """Return standard hover text for a mask parameter editor."""

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


def default_model_parameters(type: str) -> dict[str, Any]:
    """Return default parameter values for a registered model type."""

    return {
        name: copy.deepcopy(metadata["default"])
        for name, metadata in MODEL_TYPE_DEFINITIONS[type]["parameters"].items()
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
        ("Dataset", [f"Name: {dataset.name}", "Dataset", f"Kind: {dataset.kind or '-'}"]),
        ("Axes", axes_lines),
        ("Crystal", _dataset_crystal_lines(dataset, group)),
        ("Data", data_lines),
        ("Source", source_lines or ["No source file recorded."]),
        ("Metadata", metadata_lines or ["No additional metadata."]),
    ]


def slice_viewer_datasets(
    group: DataGroup,
) -> tuple[list[MDHistoData], list[str]]:
    """Return data and labels from a group that can be shown in the slice viewer."""

    data: list[MDHistoData] = []
    names: list[str] = []
    for dataset in group.datasets:
        view_data = dataset_for_slice_viewer(dataset)
        if view_data is not None:
            data.append(view_data)
            names.append(dataset.name)
    return data, names


def dataset_for_slice_viewer(dataset: DatasetEntry) -> MDHistoData | None:
    """Return a viewer-ready MDHisto dataset, loading from source metadata if needed."""

    if isinstance(dataset.data, MDHistoData):
        return _mdhisto_with_metallix_masks(dataset)
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
    return _mdhisto_with_metallix_masks(dataset)


def _mdhisto_with_metallix_masks(dataset: DatasetEntry) -> MDHistoData:
    data = dataset.data
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset does not contain MDHistoData")
    file_mask = np.asarray(data.mask, dtype=bool)
    metallix_mask = _metallix_mask_for_mdhisto(dataset, data)
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


def _metallix_mask_for_mdhisto(dataset: DatasetEntry, data: MDHistoData) -> np.ndarray:
    combined = np.zeros(data.shape, dtype=bool)
    for mask in dataset.masks:
        if not mask.enabled:
            continue
        combined |= _evaluate_mdhisto_mask(data, mask)
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
    for name in ("H", "K", "L", "E"):
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
            return np.zeros(data.shape, dtype=bool)
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
        selected &= values >= lower
    if upper is not None:
        selected &= values <= upper
    return selected


def _mdhisto_coordinate_grids(data: MDHistoData) -> dict[str, np.ndarray]:
    shape = data.shape
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    coords: dict[str, np.ndarray] = {}
    projection_rows: list[np.ndarray] = []
    projection_values: list[np.ndarray] = []
    for axis, values in zip(data.axes, axis_values, strict=True):
        role = axis.role
        if role in {"h", "k", "l", "energy_transfer", "q_modulus"}:
            key = {"h": "H", "k": "K", "l": "L", "energy_transfer": "E", "q_modulus": "q_modulus"}[role]
            coords[key] = values
            continue
        projection = _axis_projection_vector(axis.name)
        if projection is not None:
            projection_rows.append(projection)
            projection_values.append(values)
    if not {"H", "K", "L"}.issubset(coords) and projection_rows:
        matrix = np.vstack(projection_rows)
        pseudo_inverse = np.linalg.pinv(matrix)
        flat_values = np.vstack([values.reshape(1, -1) for values in projection_values])
        hkl = pseudo_inverse @ flat_values
        coords.setdefault("H", hkl[0].reshape(shape))
        coords.setdefault("K", hkl[1].reshape(shape))
        coords.setdefault("L", hkl[2].reshape(shape))
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
    raise ValueError("|Q| masks require inverse-angstrom coordinates or an attached lattice/UB matrix")


def _matrix_includes_2pi(metadata: dict[str, Any], key: str) -> bool:
    for flag_key in (f"{key}_includes_2pi", "q_matrix_includes_2pi", "include_2pi", "includes_2pi"):
        if flag_key in metadata:
            return bool(metadata[flag_key])
    lattice = metadata.get("lattice_parameters")
    return bool(isinstance(lattice, dict) and lattice.get("include_2pi"))


def next_data_group_name(groups: list[DataGroup]) -> str:
    """Return the first available DatagroupN name."""

    taken = {group.name for group in groups}
    index = 1
    while f"Datagroup{index}" in taken:
        index += 1
    return f"Datagroup{index}"


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


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dataset_entry_from_path(path: str | Path) -> DatasetEntry:
    """Create a placeholder dataset entry for a file selected in the GUI."""

    source = Path(path)
    return DatasetEntry(
        name=_unique_dataset_name(source.stem or source.name, []),
        data=None,
        kind=source.suffix.lstrip(".").lower(),
        metadata={"source_file": str(source), "import_status": "pending"},
    )


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
        self.details_label = None
        self.details_scroll = None
        self.details_widget = None
        self.details_layout = None
        self.import_dataset_button = None
        self.add_model_button = None
        self.view_slice_button = None
        self.load_dataset_button = None
        self.add_mask_button = None
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

    def import_dataset_paths(self, group: DataGroup, paths: list[str | Path]) -> list[DatasetEntry]:
        entries = import_dataset_paths(group, paths)
        if entries:
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        return entries

    def selected_data_group(self) -> DataGroup | None:
        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"group", "datasets", "models", "dataset", "fits", "fit", "fit_timeline"}:
            return group
        return None

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

        group = self.selected_data_group()
        if group is None:
            return
        paths, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self.window,
            "Import datasets",
            "",
            "Data files (*);;All files (*)",
        )
        self.import_dataset_paths(group, paths)

    def add_mask_to_selection(self) -> MaskSpec | None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return None
        mask = create_mask(entry)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_mask=mask, edit_mask=True)
        return mask

    def add_model_to_selection(self) -> ModelComponentSpec | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"group", "models"} or group is None:
            return None
        model = create_model_component(group)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_model=model, edit_model=True)
        return model

    def fit_now_for_selection(self) -> FitTimelineEntry | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        self._set_selected_fit_optimizer_config()
        result = create_placeholder_fit_result(
            group,
            fit_entry,
            branch_timeline=bool(self.fit_branch_check.isChecked()),
        )
        self.fit_branch_check.setChecked(False)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_fit=result)
        return result

    def open_slice_viewer_for_selection(self) -> Any | None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"group", "datasets", "dataset"} or group is None:
            return None
        selected_name = entry.name if role == "dataset" and entry is not None else None
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
                "Slice viewer",
                f"Could not load datasets for the slice viewer:\n{exc}",
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

    def move_or_copy_selected_to_item(self, target_item: Any, *, copy_item: bool) -> bool:
        source_group, source_entry, source_mask, _source_model, source_role = self._objects_for_item(self._current_item())
        target_group, target_entry, _target_mask, _target_model, target_role = self._objects_for_item(target_item)
        if (
            source_role == "dataset"
            and source_group is not None
            and source_entry is not None
            and target_group is not None
            and target_group is not source_group
            and target_role in {"group", "datasets"}
        ):
            moved = copy_dataset_to_group(source_entry, target_group)
            if not copy_item:
                delete_dataset(source_group, source_entry)
                self._record_data_group_state_change(source_group)
            self._record_data_group_state_change(target_group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_dataset=moved)
            return True
        if (
            source_role == "mask"
            and source_group is not None
            and source_entry is not None
            and source_mask is not None
            and target_group is not None
            and target_entry is not None
            and target_entry is not source_entry
            and target_role in {"dataset", "masks"}
        ):
            moved = copy_mask_to_dataset(source_mask, target_entry)
            if not copy_item:
                delete_mask(source_entry, source_mask)
                self._record_data_group_state_change(source_group)
            self._record_data_group_state_change(target_group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_mask=moved)
            return True
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
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)
        self.tree.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.tree.currentItemChanged.connect(lambda _current, _previous: self._sync_details())
        self.tree.itemChanged.connect(self._tree_item_changed)
        toolbar.setFont(self.tree.font())
        file_button.setFont(self.tree.font())
        menu.setFont(self.tree.font())
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
        self.create_group_button = QtWidgets.QPushButton("Create data group")
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
        self.view_slice_button = QtWidgets.QPushButton("View in slice viewer")
        self.load_dataset_button = QtWidgets.QPushButton("Load now")
        self.add_mask_button = QtWidgets.QPushButton("Add mask")
        self.import_dataset_button.clicked.connect(self.import_dataset_dialog)
        self.add_model_button.clicked.connect(self.add_model_to_selection)
        self.view_slice_button.clicked.connect(self.open_slice_viewer_for_selection)
        self.load_dataset_button.clicked.connect(self.load_dataset_for_selection)
        self.add_mask_button.clicked.connect(self.add_mask_to_selection)
        actions_row.addWidget(self.import_dataset_button)
        actions_row.addWidget(self.add_model_button)
        actions_row.addWidget(self.view_slice_button)
        actions_row.addWidget(self.load_dataset_button)
        actions_row.addWidget(self.add_mask_button)
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

        right_layout.addWidget(self.title_label)
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
        edit_group: bool = False,
        edit_mask: bool = False,
        edit_model: bool = False,
    ) -> None:
        from PySide6 import QtCore, QtWidgets

        self._expanded_state = self._current_expanded_state()
        self._item_roles.clear()
        self._fit_item_roles.clear()
        self.tree.blockSignals(True)
        self.tree.clear()
        item_to_select = None
        for group in self.project.data_groups:
            ensure_fit_history(group)
            refresh_current_state_fit_entries(group)
            group_item = QtWidgets.QTreeWidgetItem([group.name])
            group_item.setFlags(group_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            self._remember_item(group_item, "group", group)
            self.tree.addTopLevelItem(group_item)

            datasets_item = QtWidgets.QTreeWidgetItem(["Datasets"])
            self._remember_item(datasets_item, "datasets", group)
            group_item.addChild(datasets_item)
            for dataset in group.datasets:
                dataset_item = QtWidgets.QTreeWidgetItem([dataset.name])
                dataset_item.setFlags(dataset_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                self._remember_item(dataset_item, "dataset", group, dataset)
                datasets_item.addChild(dataset_item)
                masks_item = QtWidgets.QTreeWidgetItem(["Masks"])
                self._remember_item(masks_item, "masks", group, dataset)
                dataset_item.addChild(masks_item)
                for mask in dataset.masks:
                    mask_item = QtWidgets.QTreeWidgetItem([mask.name])
                    mask_item.setFlags(mask_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    self._remember_item(mask_item, "mask", group, dataset, mask)
                    masks_item.addChild(mask_item)
                    if select_mask is mask:
                        item_to_select = mask_item
                if select_dataset is dataset and item_to_select is None:
                    item_to_select = dataset_item
                dataset_item.setExpanded(self._expanded_state.get(("dataset", id(dataset)), False))
                masks_item.setExpanded(self._expanded_state.get(("masks", id(dataset)), False))

            models_item = QtWidgets.QTreeWidgetItem(["Models"])
            self._remember_item(models_item, "models", group)
            group_item.addChild(models_item)
            for name, model in group.models.items():
                model_item = QtWidgets.QTreeWidgetItem([name])
                if isinstance(model, ModelComponentSpec):
                    model_item.setFlags(model_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    self._remember_item(model_item, "model", group, model=model)
                    if select_model is model:
                        item_to_select = model_item
                else:
                    self._remember_item(model_item, "fit_model_session", group)
                models_item.addChild(model_item)

            group_item.setExpanded(self._expanded_state.get(("group", id(group)), True))
            datasets_item.setExpanded(self._expanded_state.get(("datasets", id(group)), False))
            models_item.setExpanded(self._expanded_state.get(("models", id(group)), False))
            fits_item = QtWidgets.QTreeWidgetItem(["Fits"])
            self._remember_item(fits_item, "fits", group)
            group_item.addChild(fits_item)
            for fit_entry in group.fits:
                fit_item = self._add_fit_tree_item(fits_item, group, fit_entry)
                if item_to_select is None and select_fit is fit_entry:
                    item_to_select = fit_item
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
    ) -> None:
        self._item_roles[id(item)] = (role, group, entry, mask, model)

    def _add_fit_tree_item(self, parent_item: Any, group: DataGroup, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtCore, QtWidgets

        item = QtWidgets.QTreeWidgetItem([fit_entry.name])
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        role = "fit_timeline" if fit_entry.kind == "timeline" else "fit"
        self._remember_item(item, role, group)
        self._fit_item_roles[id(item)] = fit_entry
        parent_item.addChild(item)
        for child in fit_entry.children:
            self._add_fit_tree_item(item, group, child)
        item.setExpanded(self._expanded_state.get(("fit", id(fit_entry)), fit_entry.kind == "timeline"))
        return item

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
        elif role == "mask" and mask is not None:
            new_name = item.text(0).strip() or "mask"
            changed = mask.name != new_name
            mask.name = new_name
            item.setText(0, mask.name)
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
            if role in {"dataset", "mask", "model"} and group is not None:
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
        can_import = role in {"group", "datasets"}
        can_add_model = role in {"group", "models"}
        self.import_dataset_button.setVisible(can_import)
        self.add_model_button.setVisible(can_add_model)
        self.view_slice_button.setVisible(role in {"group", "datasets", "dataset"})
        self.view_slice_button.setEnabled(bool(group is not None and _has_slice_viewer_candidates(group)))
        self.load_dataset_button.setVisible(
            role == "dataset" and entry is not None and _dataset_can_load(entry)
        )
        self.add_mask_button.setVisible(role == "dataset")
        self.delete_button.setEnabled(role in {"group", "dataset", "mask", "model", "fit", "fit_timeline"})
        self.mask_type_combo.setVisible(role == "mask")
        self.mask_parameter_widget.setVisible(role == "mask")
        self.model_type_combo.setVisible(role == "model")
        self.model_parameter_widget.setVisible(role == "model")
        self.fit_editor_widget.setVisible(role == "fit" and fit_entry is not None)
        self.fit_now_button.setVisible(role == "fit" and fit_entry is not None)

        if role == "group" and group is not None:
            self.title_label.setText(group.name)
            ensure_fit_history(group)
            result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
            self._set_details_text(
                f"Data group\n\nDatasets: {len(group.datasets)}\nModels: {len(group.models)}\nFit results: {result_count}"
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
            self._set_details_text("Mask")
            self._sync_mask_editor(mask)
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
            self._set_details_text(f"{len(self.project.data_groups)} data group(s)")
        if role != "mask":
            self._clear_mask_parameter_editor()
        if role != "model":
            self._clear_model_parameter_editor()
        if role != "fit":
            self.fit_branch_check.setChecked(False)

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
        self.details_label.setText(dataset_details_text(dataset, group=group))
        self._clear_details_panel()
        for title, lines in dataset_detail_sections(dataset, group=group):
            self.details_layout.addWidget(self._details_group_box(title, lines))
        self.details_layout.addStretch(1)

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
        _group, entry, _mask, _model, role = self._objects_for_item(item)
        can_paste = self._can_paste_into_role(role, entry)
        has_source = bool(entry is not None and _dataset_source_path(entry) is not None)
        specs: list[tuple[str, bool]] = []
        if role in {"dataset", "mask"}:
            specs.append(("Copy", True))
        if role in {"group", "datasets", "dataset", "masks"}:
            specs.append(("Paste", can_paste))
        if role in {"group", "dataset", "mask", "model", "fit", "fit_timeline"}:
            specs.append(("Rename", True))
            specs.append(("Delete", True))
        if role in {"group", "datasets", "dataset"}:
            specs.append(("View in slice viewer", True))
        if role == "dataset":
            specs.append(("Show file location", has_source))
            specs.append(("Change file source", True))
            specs.append(("Add mask", True))
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
            "View in slice viewer": self.open_slice_viewer_for_selection,
            "Show file location": self.show_file_location_for_selection,
            "Change file source": self.change_file_source_for_selection,
            "Add mask": self.add_mask_to_selection,
            "Add model": self.add_model_to_selection,
            "Fit now": self.fit_now_for_selection,
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

    def _sync_mask_editor(self, mask: MaskSpec) -> None:
        self._refresh_mask_type_combo()
        index = self.mask_type_combo.findData(mask.type)
        self.mask_type_combo.blockSignals(True)
        self.mask_type_combo.setCurrentIndex(max(index, 0))
        self.mask_type_combo.blockSignals(False)
        self._rebuild_mask_parameter_editor(mask)

    def _set_selected_mask_type(self, _label: str) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role != "mask" or mask is None:
            return
        type_name = self.mask_type_combo.currentData()
        if not type_name or type_name == mask.type:
            return
        mask.type = str(type_name)
        defaults = default_mask_parameters(mask.type)
        mask.parameters = {name: mask.parameters.get(name, value) for name, value in defaults.items()}
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_mask_parameter_editor(mask)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_mask=mask)
            else:
                self.refresh_slice_viewer(group)

    def _rebuild_mask_parameter_editor(self, mask: MaskSpec) -> None:
        from PySide6 import QtWidgets

        self._clear_mask_parameter_editor()
        for row, parameter_name in enumerate(MASK_TYPE_DEFINITIONS[mask.type]["parameters"]):
            label = QtWidgets.QLabel(parameter_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(mask.parameters.get(parameter_name, "")))
            tooltip = mask_parameter_tooltip(mask.type, parameter_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=editor: self._set_mask_parameter(parameter_name, editor.text())
            )
            self.mask_parameter_layout.addWidget(label, row, 0)
            self.mask_parameter_layout.addWidget(editor, row, 1)

    def _clear_mask_parameter_editor(self) -> None:
        while self.mask_parameter_layout.count():
            item = self.mask_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_mask_parameter(self, name: str, text: str) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role != "mask" or mask is None:
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
        default_fit = default_model_fit_parameters(model.type)
        default_global = default_model_global_fit(model.type)
        model.parameters = {name: model.parameters.get(name, value) for name, value in defaults.items()}
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
            self.model_parameter_layout.addWidget(label, row, 0)
            self.model_parameter_layout.addWidget(editor, row, 1)
            self.model_parameter_layout.addWidget(fit_check, row, 2)
            self.model_parameter_layout.addWidget(global_check, row, 3)

    def _clear_model_parameter_editor(self) -> None:
        while self.model_parameter_layout.count():
            item = self.model_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
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
            viewer.window.setWindowTitle(f"metallix Slice Viewer - {group.name}")
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
            if role not in {"dataset", "mask"}:
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
            if event.mimeData().hasFormat("application/x-metallix-tree-item"):
                target = self.itemAt(event.position().toPoint())
                if target is not None:
                    copied = event.dropAction() == QtCore.Qt.DropAction.CopyAction
                    if self.explorer.move_or_copy_selected_to_item(target, copy_item=copied):
                        event.acceptProposedAction()
                        return
            group = self._drop_group(event.position().toPoint())
            if group is None:
                super().dropEvent(event)
                return
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self.explorer.import_dataset_paths(group, paths)
                event.acceptProposedAction()
                return
            super().dropEvent(event)

        def _drop_group(self, point):
            item = self.itemAt(point)
            if item is None:
                return None
            group, _entry, _mask, _model, role = self.explorer._objects_for_item(item)
            if role in {"group", "datasets", "dataset"}:
                return group
            return None

    return ProjectTree


def _is_renameable_role(role: str) -> bool:
    return role in {"group", "dataset", "mask", "model", "fit", "fit_timeline"}


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
        if isinstance(dataset.data, MDHistoData):
            return True
        source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
        if source and Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5"}:
            return True
    return False


def _dataset_can_load(dataset: DatasetEntry) -> bool:
    if dataset.data is not None:
        return False
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    return bool(source and Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5"})


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
    merged = _merged_dataset_metadata(dataset)
    merged.pop("source_file", None)
    merged.pop("import_status", None)
    merged.pop("lattice_parameters", None)
    merged.pop("rlu_to_inv_angstrom_matrix", None)
    merged.pop("ub_matrix", None)
    merged.pop("orientation_matrix", None)
    merged.pop("oriented_lattice", None)
    return _mapping_lines(merged)


def _mapping_lines(mapping: dict[str, Any]) -> list[str]:
    return [f"{key}: {_metadata_value_text(value)}" for key, value in sorted(mapping.items())]


def _metadata_value_text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


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
            lattice_parameters=group_payload.get("lattice_parameters"),
            spacegroup=group_payload.get("spacegroup"),
            metadata=dict(group_payload.get("metadata", {})),
        )
        for dataset_payload in group_payload.get("datasets", []):
            group.add_dataset(
                DatasetEntry(
                    name=str(dataset_payload["name"]),
                    data=None,
                    kind=str(dataset_payload.get("kind", "")),
                    metadata=dict(dataset_payload.get("metadata", {})),
                    parameters=dict(dataset_payload.get("parameters", {})),
                    masks=[
                        MaskSpec(
                            name=str(mask_payload["name"]),
                            type=str(mask_payload.get("type", "coordinate_range")),
                            parameters=dict(mask_payload.get("parameters", {})),
                            enabled=bool(mask_payload.get("enabled", True)),
                            metadata=dict(mask_payload.get("metadata", {})),
                        )
                        for mask_payload in dataset_payload.get("masks", [])
                    ],
                )
            )
        for model_payload in group_payload.get("models", []):
            model_type = str(model_payload.get("type", "constant_background"))
            fit_payload = dict(model_payload.get("fit_parameters", {}))
            model = ModelComponentSpec(
                name=str(model_payload["name"]),
                type=model_type,
                parameters=dict(model_payload.get("parameters", {})),
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
        "metadata": _json_mapping(dataset.metadata),
        "parameters": _json_mapping(dataset.parameters),
        "masks": [_mask_to_dict(mask) for mask in dataset.masks],
    }


def _mask_to_dict(mask: MaskSpec) -> dict[str, Any]:
    return {
        "name": mask.name,
        "type": mask.type,
        "parameters": _json_mapping(mask.parameters),
        "enabled": bool(mask.enabled),
        "metadata": _json_mapping(mask.metadata),
    }


def _model_to_dict(model: ModelComponentSpec) -> dict[str, Any]:
    return {
        "name": model.name,
        "type": model.type,
        "parameters": _json_mapping(model.parameters),
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
        return stripped


def main() -> int:
    """Launch the metallix project explorer."""

    return MetallixProjectExplorer().run()
