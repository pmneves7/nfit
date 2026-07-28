from __future__ import annotations

import base64
import copy
import json
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from .model_registry import serialize_model_component
from .pipeline import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
)
from .plot_recipes import plot_entry_from_dict, plot_entry_to_dict

FIT_CHANNEL_NAMES = ("fit", "residual")


@dataclass
class NfitProject:
    """Serializable workspace state for the project explorer GUI."""

    data_groups: list[DataGroup] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


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


def _project_to_dict(project: NfitProject) -> dict[str, Any]:
    return {
        "format": "nfit-project",
        "version": 3,
        "settings": _json_mapping(project.settings),
        "data_groups": [_data_group_to_dict(group) for group in project.data_groups],
    }


def _project_from_dict(payload: dict[str, Any]) -> NfitProject:
    from .project_gui import (
        default_model_config,
        default_model_fit_parameters,
        ensure_fit_history,
        reconcile_model_orbit_parameters,
    )

    if payload.get("format") != "nfit-project":
        raise ValueError("not a nfit project file")
    version = int(payload.get("version", 0))
    if version != 3:
        raise ValueError(f"unsupported nfit project version {version}")
    project = NfitProject(settings=dict(payload.get("settings", {})))
    for group_payload in payload.get("data_groups", []):
        group = DataGroup(
            name=str(group_payload["name"]),
            datasets=[_dataset_from_dict(d) for d in group_payload.get("datasets", [])],
            subgroups=[_dataset_group_from_dict(s) for s in group_payload.get("subgroups", [])],
            masks=[_mask_from_dict(m) for m in group_payload.get("masks", [])],
            backgrounds=[
                _background_from_dict(background)
                for background in group_payload.get("backgrounds", [])
            ],
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
                sharing=_sharing_from_payload(model_payload.get("sharing")),
                limits=dict(model_payload.get("limits", {}) or {}),
                constraints=[
                    dict(constraint) for constraint in model_payload.get("constraints", []) or []
                ],
                applies_to=_applies_to_from_payload(model_payload.get("applies_to")),
                enabled=bool(model_payload.get("enabled", True)),
                metadata=dict(model_payload.get("metadata", {})),
            )
            reconcile_model_orbit_parameters(model)
            group.models[model.name] = model
        group.fits = [
            _fit_entry_from_dict(fit_payload) for fit_payload in group_payload.get("fits", [])
        ]
        group.analyses = [_analysis_from_dict(item) for item in group_payload.get("analyses", [])]
        group.plots = [plot_entry_from_dict(item) for item in group_payload.get("plots", [])]
        active_path = group_payload.get("active_fit_path")
        if isinstance(active_path, list) and all(isinstance(index, int) for index in active_path):
            group.active_fit_path = list(active_path)
        ensure_fit_history(group)
        project.data_groups.append(group)
    _validate_unique_dataset_ids(project)
    _link_project_backgrounds(project)
    return project


def _link_project_backgrounds(project: NfitProject) -> None:
    for group in project.data_groups:
        _link_group_backgrounds(group)


def _link_group_backgrounds(group: DataGroup) -> None:
    by_id = {dataset.id: dataset for dataset in group.iter_datasets()}
    for dataset in by_id.values():
        for background in dataset.backgrounds:
            background.source_entry = by_id.get(background.source_dataset_id)
    for node in (group, *group.iter_subgroups()):
        for background in node.backgrounds:
            background.source_entry = by_id.get(background.source_dataset_id)


def _validate_unique_dataset_ids(project: NfitProject) -> None:
    seen: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for group in project.data_groups:
        for index, dataset in enumerate(group.iter_datasets()):
            location = f"{group.name}/{dataset.name} (dataset {index + 1})"
            previous = seen.get(dataset.id)
            if previous is None:
                seen[dataset.id] = location
                continue
            duplicates.setdefault(dataset.id, [previous]).append(location)
    if duplicates:
        details = "; ".join(
            f"{dataset_id}: {', '.join(locations)}" for dataset_id, locations in duplicates.items()
        )
        raise ValueError(
            "duplicate dataset IDs make project references ambiguous: "
            f"{details}. Re-import the affected datasets or repair the project file "
            "so every dataset has a unique ID."
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
                    str(dataset): str(key) for dataset, key in dict(entry.get("groups", {})).items()
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


def _background_from_dict(payload: dict[str, Any]) -> BackgroundSpec:
    return BackgroundSpec(
        name=str(payload.get("name", "Background")),
        source_dataset_id=str(payload.get("source_dataset_id", "")),
        scale=float(payload.get("scale", 1.0)),
        enabled=bool(payload.get("enabled", True)),
        interpolation=str(payload.get("interpolation", "linear")),
        metadata=dict(payload.get("metadata", {})),
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
        scale_factor_vary=bool(dataset_payload.get("scale_factor_vary", False)),
        scale_factor_group=(
            None
            if dataset_payload.get("scale_factor_group") in (None, "")
            else str(dataset_payload["scale_factor_group"])
        ),
        masks=[_mask_from_dict(mask_payload) for mask_payload in dataset_payload.get("masks", [])],
        backgrounds=[
            _background_from_dict(background_payload)
            for background_payload in dataset_payload.get("backgrounds", [])
        ],
        id=str(dataset_payload.get("id") or DatasetEntry("", None).id),
    )


def _dataset_group_from_dict(payload: dict[str, Any]) -> DatasetGroup:
    return DatasetGroup(
        name=str(payload["name"]),
        datasets=[_dataset_from_dict(d) for d in payload.get("datasets", [])],
        subgroups=[_dataset_group_from_dict(s) for s in payload.get("subgroups", [])],
        enabled=bool(payload.get("enabled", True)),
        masks=[_mask_from_dict(m) for m in payload.get("masks", [])],
        backgrounds=[
            _background_from_dict(background) for background in payload.get("backgrounds", [])
        ],
        resolution=dict(payload.get("resolution", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _dataset_group_to_dict(group: DatasetGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "datasets": [_dataset_to_dict(dataset) for dataset in group.datasets],
        "subgroups": [_dataset_group_to_dict(sub) for sub in group.subgroups],
        "enabled": bool(group.enabled),
        "masks": [_mask_to_dict(mask) for mask in group.masks],
        "backgrounds": [_background_to_dict(background) for background in group.backgrounds],
        "resolution": _json_mapping(group.resolution),
        "metadata": _json_mapping(group.metadata),
    }


def _data_group_to_dict(group: DataGroup) -> dict[str, Any]:
    from .project_gui import ensure_fit_history, refresh_current_state_fit_entries

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
        "backgrounds": [_background_to_dict(background) for background in group.backgrounds],
        "models": model_payloads,
        "fits": [_fit_entry_to_dict(fit_entry) for fit_entry in group.fits],
        "active_fit_path": (
            list(group.active_fit_path) if group.active_fit_path is not None else None
        ),
        "analyses": [_analysis_to_dict(analysis) for analysis in group.analyses],
        "plots": [plot_entry_to_dict(plot) for plot in group.plots],
    }


def _dataset_to_dict(dataset: DatasetEntry) -> dict[str, Any]:
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if dataset.data is not None and (not source or not dataset.data_matches_source):
        raise TypeError(
            f"dataset {dataset.name!r} contains replacement data that are not "
            "stored in its source file; save the dataset to a portable .npz "
            "file and re-import it before saving the project"
        )
    if dataset.transforms:
        raise TypeError("project JSON save does not yet support dataset transforms")
    serialized_metadata = copy.deepcopy(dataset.metadata)
    if serialized_metadata.get("derived_from_analysis") and serialized_metadata.get(
        "analysis_artifact_path"
    ):
        serialized_metadata["source_file"] = serialized_metadata["analysis_artifact_path"]
    return {
        "id": dataset.id,
        "name": dataset.name,
        "kind": dataset.kind,
        "data_type": dataset.data_type,
        "metadata": _json_mapping(serialized_metadata),
        "parameters": _json_mapping(dataset.parameters),
        "enabled": bool(dataset.enabled),
        "fit_weight": float(dataset.fit_weight),
        "scale_factor": float(dataset.scale_factor),
        "scale_factor_vary": bool(dataset.scale_factor_vary),
        "scale_factor_group": dataset.scale_factor_group,
        "masks": [_mask_to_dict(mask) for mask in dataset.masks],
        "backgrounds": [_background_to_dict(background) for background in dataset.backgrounds],
    }


def _analysis_to_dict(analysis: AnalysisEntry) -> dict[str, Any]:
    payload = {
        "id": analysis.id,
        "name": analysis.name,
        "type": analysis.type,
        "input_dataset_ids": list(analysis.input_dataset_ids),
        "parameters": _json_mapping(analysis.parameters),
        "operation_version": analysis.operation_version,
        "enabled": analysis.enabled,
        "metadata": _json_mapping(analysis.metadata),
        "result": None,
    }
    if analysis.result is not None:
        result = analysis.result
        payload["result"] = {
            "recipe_hash": result.recipe_hash,
            "input_fingerprints": dict(result.input_fingerprints),
            "outputs": [vars(output) for output in result.outputs],
            "status": result.status,
            "created_at": result.created_at,
            "duration_seconds": result.duration_seconds,
            "warnings": list(result.warnings),
            "diagnostics": _json_mapping(result.diagnostics),
            "error": result.error,
        }
    return payload


def _analysis_from_dict(payload: dict[str, Any]) -> AnalysisEntry:
    analysis_type = str(payload["type"])
    parameters = dict(payload.get("parameters", {}))
    if analysis_type == "bragg_integration" and "edge_policy" in parameters:
        raise ValueError(
            "Bragg analysis parameter 'edge_policy' is no longer supported; "
            "use 'minimum_peak_coverage'"
        )
    metadata = dict(payload.get("metadata", {}))
    result_payload = payload.get("result")
    result = None
    if isinstance(result_payload, dict):
        result = AnalysisResultRecord(
            recipe_hash=str(result_payload["recipe_hash"]),
            input_fingerprints={
                str(k): str(v) for k, v in result_payload.get("input_fingerprints", {}).items()
            },
            outputs=[AnalysisOutputRef(**item) for item in result_payload.get("outputs", [])],
            status=str(result_payload.get("status", "success")),
            created_at=str(result_payload.get("created_at", "")),
            duration_seconds=result_payload.get("duration_seconds"),
            warnings=list(result_payload.get("warnings", [])),
            diagnostics=dict(result_payload.get("diagnostics", {})),
            error=result_payload.get("error"),
        )
    return AnalysisEntry(
        name=str(payload["name"]),
        type=analysis_type,
        input_dataset_ids=[str(value) for value in payload.get("input_dataset_ids", [])],
        parameters=parameters,
        id=str(payload.get("id") or AnalysisEntry("", "", [], {}).id),
        operation_version=int(payload.get("operation_version", 1)),
        enabled=bool(payload.get("enabled", True)),
        result=result,
        metadata=metadata,
    )


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


def _background_to_dict(background: BackgroundSpec) -> dict[str, Any]:
    return {
        "name": background.name,
        "source_dataset_id": background.source_dataset_id,
        "scale": float(background.scale),
        "enabled": bool(background.enabled),
        "interpolation": background.interpolation,
        "metadata": _json_mapping(background.metadata),
    }


def _model_to_dict(model: ModelComponentSpec) -> dict[str, Any]:
    return serialize_model_component(model, purpose="project")


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
        "id": fit_entry.id,
    }


def _fit_channels_to_dict(channels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset_name, entry in (channels or {}).items():
        encoded: dict[str, Any] = {"kind": str(entry.get("kind", "points"))}
        if entry.get("visualization_only"):
            encoded["visualization_only"] = True
        if entry.get("fit_channel"):
            encoded["fit_channel"] = str(entry["fit_channel"])
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
        if entry.get("visualization_only"):
            decoded["visualization_only"] = True
        if entry.get("fit_channel"):
            decoded["fit_channel"] = str(entry["fit_channel"])
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
        children=[_fit_entry_from_dict(child) for child in payload.get("children", [])],
        metadata=dict(payload.get("metadata", {})),
        id=str(payload.get("id") or FitTimelineEntry("").id),
    )


def _json_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    json.dumps(value)
    return dict(value)
