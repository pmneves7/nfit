"""Project snapshots, fit-history state, and reference relinking.

These operations are part of the serializable project model rather than the Qt
explorer.  They are kept here so persistence and scripting code can use them
without importing :mod:`nfit.project_gui`.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any

from .model_registry import serialize_model_component
from .pipeline import (
    BackgroundSpec,
    DataGroup,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
)
from .project_models import reconcile_model_orbit_parameters


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    json.dumps(value)
    return dict(value)


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


def _mask_from_dict(payload: dict[str, Any]) -> MaskSpec:
    return MaskSpec(
        name=str(payload["name"]),
        type=str(payload.get("type", "coordinate_range")),
        parameters=dict(payload.get("parameters", {})),
        enabled=bool(payload.get("enabled", True)),
        invert=bool(payload.get("invert", False)),
        additive=bool(payload.get("additive", False)),
        metadata=dict(payload.get("metadata", {})),
    )


def _background_to_dict(background: BackgroundSpec) -> dict[str, Any]:
    return {
        "name": background.name,
        "source_dataset_id": background.source_dataset_id,
        "source_group_id": background.source_group_id,
        "scale": float(background.scale),
        "enabled": bool(background.enabled),
        "interpolation": background.interpolation,
        "metadata": _json_mapping(background.metadata),
    }


def _background_from_dict(payload: dict[str, Any]) -> BackgroundSpec:
    return BackgroundSpec(
        name=str(payload.get("name", "Background")),
        source_dataset_id=str(payload.get("source_dataset_id", "")),
        source_group_id=(
            None
            if payload.get("source_group_id") in (None, "")
            else str(payload["source_group_id"])
        ),
        scale=float(payload.get("scale", 1.0)),
        enabled=bool(payload.get("enabled", True)),
        interpolation=str(payload.get("interpolation", "linear")),
        metadata=dict(payload.get("metadata", {})),
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


def _model_to_dict(model: ModelComponentSpec) -> dict[str, Any]:
    return serialize_model_component(model, purpose="project")


def _named_group_nodes(group: DataGroup) -> list[tuple[str, Any]]:
    nodes: list[tuple[str, Any]] = [("", group)]
    nodes.extend((subgroup.name, subgroup) for subgroup in group.iter_subgroups())
    return nodes


def _dataset_group_paths(group: DataGroup) -> list[tuple[str, DatasetGroup]]:
    nodes: list[tuple[str, DatasetGroup]] = []

    def visit(node: DatasetGroup, path: tuple[str, ...]) -> None:
        nodes.append(("/".join(path), node))
        for subgroup in node.subgroups:
            visit(subgroup, (*path, subgroup.name))

    for subgroup in group.subgroups:
        visit(subgroup, (subgroup.name,))
    return nodes


def _link_group_backgrounds(group: DataGroup) -> None:
    """Resolve serialized background IDs to their in-memory objects."""

    by_id = {dataset.id: dataset for dataset in group.iter_datasets()}
    groups_by_id = {node.id: node for node in group.iter_subgroups()}
    for dataset in by_id.values():
        if isinstance(dataset.metadata.get("derived_recipe"), dict):
            dataset._derived_owner_group = group
        for background in dataset.backgrounds:
            background.source_entry = by_id.get(background.source_dataset_id)
            background.source_group = groups_by_id.get(background.source_group_id or "")
    for node in (group, *group.iter_subgroups()):
        for background in node.backgrounds:
            background.source_entry = by_id.get(background.source_dataset_id)
            background.source_group = groups_by_id.get(background.source_group_id or "")


def _link_project_backgrounds(project: Any) -> None:
    for group in project.data_groups:
        _link_group_backgrounds(group)


def snapshot_data_group_state(group: DataGroup) -> dict[str, Any]:
    """Capture serializable dataset, mask, background, and model state."""

    return {
        "datasets": [
            {
                "name": dataset.name,
                "data_type": dataset.data_type,
                "parameters": copy.deepcopy(dataset.parameters),
                "enabled": bool(dataset.enabled),
                "fit_weight": float(dataset.fit_weight),
                "scale_factor": float(dataset.scale_factor),
                "scale_factor_vary": bool(dataset.scale_factor_vary),
                "scale_factor_group": dataset.scale_factor_group,
                "masks": [_mask_to_dict(mask) for mask in dataset.masks],
                "backgrounds": [
                    _background_to_dict(background) for background in dataset.backgrounds
                ],
            }
            for dataset in group.iter_datasets()
        ],
        "group_masks": {
            node_name: [_mask_to_dict(mask) for mask in node.masks]
            for node_name, node in _named_group_nodes(group)
        },
        "dataset_group_enabled": {
            path: bool(node.enabled) for path, node in _dataset_group_paths(group)
        },
        "group_backgrounds": {
            node_name: [
                _background_to_dict(background) for background in node.backgrounds
            ]
            for node_name, node in _named_group_nodes(group)
        },
        "models": [
            _model_to_dict(model)
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
        ],
    }


def restore_data_group_state(group: DataGroup, snapshot: dict[str, Any]) -> None:
    """Restore dataset, mask, background, and model state from a snapshot."""

    datasets_by_name = {dataset.name: dataset for dataset in group.iter_datasets()}
    for dataset_payload in snapshot.get("datasets", []):
        dataset = datasets_by_name.get(str(dataset_payload.get("name", "")))
        if dataset is None:
            continue
        dataset.data_type = str(dataset_payload.get("data_type", dataset.data_type))
        dataset.parameters = dict(dataset_payload.get("parameters", {}))
        dataset.enabled = bool(dataset_payload.get("enabled", dataset.enabled))
        dataset.fit_weight = float(dataset_payload.get("fit_weight", dataset.fit_weight))
        dataset.scale_factor = float(
            dataset_payload.get("scale_factor", dataset.scale_factor)
        )
        dataset.scale_factor_vary = bool(
            dataset_payload.get("scale_factor_vary", dataset.scale_factor_vary)
        )
        dataset.scale_factor_group = (
            None
            if dataset_payload.get("scale_factor_group") in (None, "")
            else str(dataset_payload["scale_factor_group"])
        )
        dataset.masks = [
            _mask_from_dict(payload) for payload in dataset_payload.get("masks", [])
        ]
        dataset.backgrounds = [
            _background_from_dict(payload)
            for payload in dataset_payload.get("backgrounds", [])
        ]

    enabled = snapshot.get("dataset_group_enabled", {})
    if isinstance(enabled, dict):
        for path, node in _dataset_group_paths(group):
            if path in enabled:
                node.enabled = bool(enabled[path])
    group_masks = snapshot.get("group_masks", {})
    if isinstance(group_masks, dict):
        for node_name, node in _named_group_nodes(group):
            if node_name in group_masks:
                node.masks = [_mask_from_dict(item) for item in group_masks[node_name]]
    backgrounds = snapshot.get("group_backgrounds", {})
    if isinstance(backgrounds, dict):
        for node_name, node in _named_group_nodes(group):
            if node_name in backgrounds:
                node.backgrounds = [
                    _background_from_dict(item) for item in backgrounds[node_name]
                ]
    _link_group_backgrounds(group)

    for model_payload in snapshot.get("models", []):
        existing = group.models.get(str(model_payload.get("name", "")))
        if not isinstance(existing, ModelComponentSpec):
            continue
        existing.type = str(model_payload.get("type", existing.type))
        existing.parameters = dict(model_payload.get("parameters", {}))
        existing.config = dict(model_payload.get("config", {}))
        existing.fit_parameters = {
            str(key): bool(value)
            for key, value in dict(model_payload.get("fit_parameters", {})).items()
        }
        existing.sharing = _sharing_from_payload(model_payload.get("sharing"))
        existing.limits = dict(model_payload.get("limits", {}) or {})
        existing.constraints = [
            dict(constraint) for constraint in model_payload.get("constraints", []) or []
        ]
        existing.applies_to = _applies_to_from_payload(model_payload.get("applies_to"))
        existing.enabled = bool(model_payload.get("enabled", existing.enabled))
        existing.metadata = dict(model_payload.get("metadata", {}))
        reconcile_model_orbit_parameters(existing)


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


def refresh_current_state_fit_entries(group: DataGroup) -> None:
    """Refresh the top-level Current state node from live project state."""

    snapshot = snapshot_data_group_state(group)
    for entry in group.fits:
        if entry.kind == "current":
            entry.snapshot = copy.deepcopy(snapshot)
            entry.created_at = _timestamp_now()
