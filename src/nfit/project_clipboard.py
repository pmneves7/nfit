"""GUI-independent copy and paste rules for project explorer objects."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .pipeline import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
)


@dataclass(frozen=True)
class ClipboardCapability:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ProjectClipboardPayload:
    """An immutable description of explorer objects selected for copying."""

    kind: str
    items: tuple[Any, ...]


@dataclass(frozen=True)
class PasteResult:
    items: tuple[Any, ...]
    changed_datasets: bool = False


COPYABLE_ROLES = frozenset(
    {
        "group",
        "datasets",
        "dataset",
        "dataset_page",
        "dataset_group",
        "masks",
        "mask",
        "group_masks",
        "group_mask",
        "backgrounds",
        "background",
        "group_backgrounds",
        "group_background",
        "models",
        "model",
        "fits",
        "fit",
        "fit_timeline",
        "analyses",
        "analysis",
        "plots",
        "plot",
    }
)


def copy_capability(role: str, objects: Iterable[Any]) -> ClipboardCapability:
    items = tuple(objects)
    if role == "analysis_output":
        return ClipboardCapability(
            False, "Analysis outputs are references owned by their analysis recipe."
        )
    if role == "fit_model_session":
        return ClipboardCapability(False, "This row is a live fitting-session reference.")
    if role not in COPYABLE_ROLES:
        return ClipboardCapability(False, "This explorer row does not own copyable project data.")
    if not items:
        return ClipboardCapability(False, "The collection is empty.")
    if role in {"models", "model"} and not all(
        isinstance(item, ModelComponentSpec) for item in items
    ):
        return ClipboardCapability(
            False, "Live fitting-session rows cannot be copied as model components."
        )
    return ClipboardCapability(True)


def make_payload(role: str, objects: Iterable[Any]) -> ProjectClipboardPayload:
    items = tuple(objects)
    capability = copy_capability(role, items)
    if not capability.allowed:
        raise ValueError(capability.reason)
    kind = {
        "dataset_page": "dataset",
        "datasets": "dataset_tree",
        "masks": "mask", "group_mask": "mask",
        "group_masks": "mask",
        "backgrounds": "background", "group_background": "background",
        "group_backgrounds": "background",
        "models": "model",
        "fits": "fit",
        "fit_timeline": "fit",
        "analyses": "analysis",
        "plots": "plot",
    }.get(role, role)
    return ProjectClipboardPayload(kind, tuple(_snapshot_item(item) for item in items))


def _snapshot_item(item: Any) -> Any:
    """Freeze clipboard configuration without duplicating numerical arrays."""

    if isinstance(item, DatasetEntry):
        result = _clone_dataset(item)
        result.id = item.id
        return result
    if isinstance(item, DatasetGroup):
        return _snapshot_dataset_group(item)
    if isinstance(item, DataGroup):
        return clone_workspace(item)
    if isinstance(item, BackgroundSpec):
        return _clone_background(item)
    return copy.deepcopy(item)


def _snapshot_dataset_group(source: DatasetGroup) -> DatasetGroup:
    result = DatasetGroup(
        source.name,
        enabled=source.enabled,
        masks=copy.deepcopy(source.masks),
        backgrounds=[_clone_background(item) for item in source.backgrounds],
        resolution=copy.deepcopy(source.resolution),
        metadata=copy.deepcopy(source.metadata),
        id=source.id,
    )
    result.datasets = [_snapshot_item(item) for item in source.datasets]
    result.subgroups = [_snapshot_dataset_group(item) for item in source.subgroups]
    return result


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    index = 1
    while f"{name}{index}" in used:
        index += 1
    return f"{name}{index}"


def _clone_dataset(source: DatasetEntry) -> DatasetEntry:
    """Clone configuration with a fresh ID while sharing immutable numerical data."""

    result = source.copy()
    result.metadata = copy.deepcopy(source.metadata)
    result.parameters = copy.deepcopy(source.parameters)
    result.masks = copy.deepcopy(source.masks)
    result.backgrounds = [_clone_background(item) for item in source.backgrounds]
    result.transforms = copy.deepcopy(source.transforms)
    return result


def _clone_background(source: BackgroundSpec) -> BackgroundSpec:
    return BackgroundSpec(
        name=source.name,
        source_dataset_id=source.source_dataset_id,
        scale=source.scale,
        enabled=source.enabled,
        interpolation=source.interpolation,
        metadata=copy.deepcopy(source.metadata),
        source_group_id=source.source_group_id,
        projection=source.projection,
    )


def _fresh_fit_ids(entries: list[FitTimelineEntry]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def visit(entry: FitTimelineEntry) -> None:
        old_id = entry.id
        entry.id = uuid4().hex
        mapping[old_id] = entry.id
        for child in entry.children:
            visit(child)

    for entry in entries:
        visit(entry)
    return mapping


def _remap_backgrounds(
    owners: Iterable[Any], dataset_ids: dict[str, str], group_ids: dict[str, str]
) -> None:
    for owner in owners:
        for background in owner.backgrounds:
            background.source_dataset_id = dataset_ids.get(
                background.source_dataset_id, background.source_dataset_id
            )
            if background.source_group_id:
                background.source_group_id = group_ids.get(
                    background.source_group_id, background.source_group_id
                )
            background.source_entry = None
            background.source_group = None


def _background_references_are_valid(owners: Iterable[Any], group: DataGroup) -> bool:
    owners = tuple(owners)
    dataset_ids = {item.id for item in group.iter_datasets()}
    group_ids = {item.id for item in group.iter_subgroups()}
    for owner in owners:
        if isinstance(owner, DatasetEntry):
            dataset_ids.add(owner.id)
        elif isinstance(owner, DatasetGroup):
            group_ids.add(owner.id)
    for owner in owners:
        for background in owner.backgrounds:
            if background.source_dataset_id and background.source_dataset_id not in dataset_ids:
                return False
            if background.source_group_id and background.source_group_id not in group_ids:
                return False
    return True


def _link_backgrounds(owners: Iterable[Any], group: DataGroup) -> None:
    owners = tuple(owners)
    datasets = {item.id: item for item in group.iter_datasets()}
    groups = {item.id: item for item in group.iter_subgroups()}
    for owner in owners:
        if isinstance(owner, DatasetEntry):
            datasets[owner.id] = owner
        elif isinstance(owner, DatasetGroup):
            groups[owner.id] = owner
    for owner in owners:
        for background in owner.backgrounds:
            background.source_entry = datasets.get(background.source_dataset_id)
            background.source_group = groups.get(background.source_group_id or "")


def clone_dataset_tree(
    direct_datasets: Iterable[DatasetEntry], subgroups: Iterable[DatasetGroup]
) -> tuple[list[DatasetEntry], list[DatasetGroup], dict[str, str], dict[str, str]]:
    dataset_ids: dict[str, str] = {}
    group_ids: dict[str, str] = {}

    def clone_dataset(dataset: DatasetEntry) -> DatasetEntry:
        result = _clone_dataset(dataset)
        dataset_ids[dataset.id] = result.id
        return result

    def clone_group(group: DatasetGroup) -> DatasetGroup:
        result = DatasetGroup(
            name=group.name,
            enabled=group.enabled,
            masks=copy.deepcopy(group.masks),
            backgrounds=[_clone_background(item) for item in group.backgrounds],
            resolution=copy.deepcopy(group.resolution),
            metadata=copy.deepcopy(group.metadata),
        )
        group_ids[group.id] = result.id
        result.datasets = [clone_dataset(item) for item in group.datasets]
        result.subgroups = [clone_group(item) for item in group.subgroups]
        return result

    datasets = [clone_dataset(item) for item in direct_datasets]
    groups = [clone_group(item) for item in subgroups]
    owners: list[Any] = list(datasets)
    for group in groups:
        owners.extend((group, *group.iter_subgroups(), *group.iter_datasets()))
    _remap_backgrounds(owners, dataset_ids, group_ids)
    return datasets, groups, dataset_ids, group_ids


def _uniquify_tree_names(
    data_group: DataGroup,
    datasets: Iterable[DatasetEntry],
    groups: Iterable[DatasetGroup],
) -> None:
    used_dataset_names = set(data_group.dataset_names)
    used_group_names = {item.name for item in data_group.iter_subgroups()}

    def visit_dataset(dataset: DatasetEntry) -> None:
        dataset.name = _unique_name(dataset.name, used_dataset_names)
        used_dataset_names.add(dataset.name)

    def visit_group(group: DatasetGroup) -> None:
        group.name = _unique_name(group.name, used_group_names)
        used_group_names.add(group.name)
        for dataset in group.datasets:
            visit_dataset(dataset)
        for child in group.subgroups:
            visit_group(child)

    for dataset in datasets:
        visit_dataset(dataset)
    for group in groups:
        visit_group(group)


def clone_workspace(source: DataGroup, used_names: Iterable[str] = ()) -> DataGroup:
    """Clone a complete workspace and remap every internal stable reference."""

    datasets, subgroups, dataset_ids, group_ids = clone_dataset_tree(
        source.datasets, source.subgroups
    )
    result = DataGroup(
        name=_unique_name(source.name, set(used_names)),
        datasets=datasets,
        subgroups=subgroups,
        masks=copy.deepcopy(source.masks),
        backgrounds=[_clone_background(item) for item in source.backgrounds],
        lattice_parameters=copy.deepcopy(source.lattice_parameters),
        spacegroup=source.spacegroup,
        metadata=copy.deepcopy(source.metadata),
        models=copy.deepcopy(source.models),
    )
    _remap_backgrounds((result,), dataset_ids, group_ids)
    result.fits = copy.deepcopy(source.fits)
    fit_ids = _fresh_fit_ids(result.fits)
    for fit in result.fits:
        for entry in _iter_fits(fit):
            _remap_fit_snapshot_backgrounds(entry.snapshot, dataset_ids, group_ids)
    result.active_fit_path = copy.deepcopy(source.active_fit_path)
    result.analyses = copy.deepcopy(source.analyses)
    analysis_ids: dict[str, str] = {}
    for source_analysis, analysis in zip(
        source.analyses, result.analyses, strict=True
    ):
        analysis.id = uuid4().hex
        analysis_ids[source_analysis.id] = analysis.id
        analysis.input_dataset_ids = [
            _remap_analysis_source_id(item, dataset_ids, group_ids)
            for item in analysis.input_dataset_ids
        ]
        if analysis.result is not None:
            analysis.result.input_fingerprints = {
                dataset_ids.get(key, key): value
                for key, value in analysis.result.input_fingerprints.items()
            }
            for output in analysis.result.outputs:
                if output.dataset_id:
                    output.dataset_id = dataset_ids.get(output.dataset_id, output.dataset_id)
    for dataset in result.iter_datasets():
        provenance = dataset.metadata.get("derived_from_analysis")
        if isinstance(provenance, dict) and provenance.get("analysis_id"):
            provenance["analysis_id"] = analysis_ids.get(
                provenance["analysis_id"], provenance["analysis_id"]
            )
        recipe = dataset.metadata.get("derived_recipe")
        if isinstance(recipe, dict):
            if recipe.get("analysis_id"):
                recipe["analysis_id"] = analysis_ids.get(
                    recipe["analysis_id"], recipe["analysis_id"]
                )
            recipe["input_source_ids"] = [
                _remap_analysis_source_id(item, dataset_ids, group_ids)
                for item in recipe.get("input_source_ids", ())
            ]
            dataset._derived_owner_group = result
    result.plots = copy.deepcopy(source.plots)
    for plot in result.plots:
        plot.id = uuid4().hex
        for ref in plot.sources:
            if ref.dataset_id:
                ref.dataset_id = dataset_ids.get(ref.dataset_id, ref.dataset_id)
            if ref.fit_id:
                ref.fit_id = fit_ids.get(ref.fit_id, ref.fit_id)
        composite = plot.settings.get("source_composite")
        if isinstance(composite, dict) and composite.get("dataset_group_id"):
            composite["dataset_group_id"] = group_ids.get(
                composite["dataset_group_id"], composite["dataset_group_id"]
            )
        rebin_configs = plot.settings.get("source_rebin_configs")
        if isinstance(rebin_configs, dict):
            plot.settings["source_rebin_configs"] = {
                dataset_ids.get(key, key): value for key, value in rebin_configs.items()
            }
    return result


def _remap_analysis_source_id(
    source_id: str, dataset_ids: dict[str, str], group_ids: dict[str, str]
) -> str:
    if source_id in dataset_ids:
        return dataset_ids[source_id]
    for prefix in ("group-composite:", "group:"):
        if source_id.startswith(prefix):
            suffix = source_id.removeprefix(prefix)
            return f"{prefix}{group_ids.get(suffix, suffix)}"
    return source_id


def _remap_fit_snapshot_backgrounds(
    snapshot: dict[str, Any], dataset_ids: dict[str, str], group_ids: dict[str, str]
) -> None:
    """Remap only stable background references in a fit-history snapshot."""

    payloads: list[dict[str, Any]] = []
    for dataset in snapshot.get("datasets", ()):
        payloads.extend(dataset.get("backgrounds", ()))
    for backgrounds in snapshot.get("group_backgrounds", {}).values():
        payloads.extend(backgrounds)
    for background in payloads:
        source_dataset_id = background.get("source_dataset_id")
        if source_dataset_id:
            background["source_dataset_id"] = dataset_ids.get(
                source_dataset_id, source_dataset_id
            )
        source_group_id = background.get("source_group_id")
        if source_group_id:
            background["source_group_id"] = group_ids.get(
                source_group_id, source_group_id
            )


def paste_capability(
    payload: ProjectClipboardPayload | None,
    target_role: str,
    *,
    data_group: DataGroup | None = None,
) -> ClipboardCapability:
    if payload is None:
        return ClipboardCapability(False, "The explorer clipboard is empty.")
    allowed = {
        "group": {"project", "group"},
        "dataset_tree": {"group", "datasets", "dataset_group"},
        "dataset": {"group", "datasets", "dataset_group", "dataset", "dataset_page"},
        "dataset_group": {"group", "datasets", "dataset_group"},
        "mask": {"dataset", "masks", "mask", "dataset_group", "group_masks", "group_mask"},
        "background": {
            "datasets", "dataset", "backgrounds", "background", "dataset_group",
            "group_backgrounds", "group_background",
        },
        "model": {"group", "models", "model"},
        "fit": {"group", "fits", "fit", "fit_timeline"},
        "analysis": {"group", "analyses", "analysis"},
        "plot": {"group", "plots", "plot"},
    }.get(payload.kind, set())
    if target_role not in allowed:
        return ClipboardCapability(
            False, f"{payload.kind.replace('_', ' ').title()} items cannot be pasted here."
        )
    if data_group is None or payload.kind == "group":
        return ClipboardCapability(True)
    reason = _dependency_error(payload, data_group)
    return ClipboardCapability(reason is None, reason or "")


def _dependency_error(
    payload: ProjectClipboardPayload, data_group: DataGroup
) -> str | None:
    dataset_ids = {item.id for item in data_group.iter_datasets()}
    group_ids = {item.id for item in data_group.iter_subgroups()}
    valid_analysis_sources = dataset_ids | {
        f"group-composite:{item}" for item in group_ids
    } | {f"group:{item}" for item in group_ids} | {
        "group-composite:root", "group:root"
    }
    if payload.kind in {"dataset", "dataset_tree", "dataset_group"}:
        datasets = [item for item in payload.items if isinstance(item, DatasetEntry)]
        for subgroup in (item for item in payload.items if isinstance(item, DatasetGroup)):
            datasets.extend(subgroup.iter_datasets())
        analysis_ids = {item.id for item in data_group.analyses}
        for item in datasets:
            provenance = item.metadata.get("derived_from_analysis")
            analysis_id = provenance.get("analysis_id") if isinstance(provenance, dict) else None
            if analysis_id and analysis_id not in analysis_ids:
                return "The destination does not contain the analysis referenced by this derived dataset."
    if payload.kind == "background":
        for item in payload.items:
            if item.source_dataset_id and item.source_dataset_id not in dataset_ids:
                return "The destination does not contain the background source dataset."
            if item.source_group_id and item.source_group_id not in group_ids:
                return "The destination does not contain the background source group."
    if payload.kind == "analysis" and any(
        not set(item.input_dataset_ids) <= valid_analysis_sources for item in payload.items
    ):
        return "The destination does not contain every dataset or group referenced by this analysis."
    if payload.kind == "plot":
        fit_ids = {candidate.id for entry in data_group.fits for candidate in _iter_fits(entry)}
        if any(
            (ref.dataset_id and ref.dataset_id not in dataset_ids)
            or (ref.fit_id and ref.fit_id not in fit_ids)
            for item in payload.items
            for ref in item.sources
        ):
            return "The destination does not contain every dataset or fit referenced by this plot."
        for item in payload.items:
            composite = item.settings.get("source_composite")
            if (
                isinstance(composite, dict)
                and composite.get("dataset_group_id")
                and composite["dataset_group_id"] not in group_ids
            ):
                return "The destination does not contain the group referenced by this plot."
            rebin_configs = item.settings.get("source_rebin_configs")
            if isinstance(rebin_configs, dict) and not set(rebin_configs) <= dataset_ids:
                return "The destination does not contain every dataset setting referenced by this plot."
    if payload.kind == "fit":
        dataset_names = set(data_group.dataset_names)
        model_names = set(data_group.models)
        if any(
            item.get("name") not in dataset_names
            for entry in payload.items
            for candidate in _iter_fits(entry)
            for item in candidate.snapshot.get("datasets", ())
        ) or any(
            item.get("name") not in model_names
            for entry in payload.items
            for candidate in _iter_fits(entry)
            for item in candidate.snapshot.get("models", ())
        ):
            return "The destination does not contain every dataset or model referenced by this fit."
        for entry in payload.items:
            for candidate in _iter_fits(entry):
                background_payloads = [
                    background
                    for dataset in candidate.snapshot.get("datasets", ())
                    for background in dataset.get("backgrounds", ())
                ]
                background_payloads.extend(
                    background
                    for backgrounds in candidate.snapshot.get(
                        "group_backgrounds", {}
                    ).values()
                    for background in backgrounds
                )
                if any(
                    (background.get("source_dataset_id") and background["source_dataset_id"] not in dataset_ids)
                    or (background.get("source_group_id") and background["source_group_id"] not in group_ids)
                    for background in background_payloads
                ):
                    return "The destination does not contain every background source referenced by this fit."
    return None


def paste_payload(
    payload: ProjectClipboardPayload,
    *,
    target_role: str,
    data_group: DataGroup | None = None,
    dataset_node: DataGroup | DatasetGroup | None = None,
    dataset: DatasetEntry | None = None,
    project_groups: list[DataGroup] | None = None,
) -> PasteResult:
    """Paste a payload into a resolved domain target, or raise ``ValueError``."""

    capability = paste_capability(payload, target_role, data_group=data_group)
    if not capability.allowed:
        raise ValueError(capability.reason)
    if payload.kind == "group":
        if project_groups is None:
            raise ValueError("A project workspace list is required.")
        pasted: list[DataGroup] = []
        used_names = {item.name for item in project_groups}
        for item in payload.items:
            cloned = clone_workspace(item, used_names)
            used_names.add(cloned.name)
            pasted.append(cloned)
        project_groups.extend(pasted)
        return PasteResult(tuple(pasted), True)
    if data_group is None:
        raise ValueError("A destination workspace is required.")
    node = dataset_node or data_group
    if payload.kind in {"dataset", "dataset_tree"}:
        source_datasets: list[DatasetEntry] = []
        source_groups: list[DatasetGroup] = []
        source_masks: list[MaskSpec] = []
        source_backgrounds: list[BackgroundSpec] = []
        for item in payload.items:
            if isinstance(item, DatasetEntry):
                source_datasets.append(item)
            elif isinstance(item, DatasetGroup):
                source_groups.append(item)
            elif isinstance(item, MaskSpec):
                source_masks.append(item)
            elif isinstance(item, BackgroundSpec):
                source_backgrounds.append(item)
        destination_analysis_ids = {item.id for item in data_group.analyses}
        source_tree_datasets = [
            *source_datasets,
            *(
                child
                for subgroup in source_groups
                for child in subgroup.iter_datasets()
            ),
        ]
        for source in source_tree_datasets:
            provenance = source.metadata.get("derived_from_analysis")
            analysis_id = (
                provenance.get("analysis_id")
                if isinstance(provenance, dict)
                else None
            )
            if analysis_id and analysis_id not in destination_analysis_ids:
                raise ValueError(
                    "The destination does not contain the analysis referenced by this derived dataset."
                )
        datasets, groups, dataset_ids, group_ids = clone_dataset_tree(
            source_datasets, source_groups
        )
        masks = list(copy.deepcopy(source_masks))
        backgrounds = [_clone_background(item) for item in source_backgrounds]
        _remap_backgrounds(
            (type("BackgroundOwner", (), {"backgrounds": backgrounds})(),),
            dataset_ids,
            group_ids,
        )
        for item in [
            *datasets,
            *(child for subgroup in groups for child in subgroup.iter_datasets()),
        ]:
            if isinstance(item.metadata.get("derived_recipe"), dict):
                item._derived_owner_group = data_group
        copied_owners: list[Any] = list(datasets)
        for subgroup in groups:
            copied_owners.extend(
                (subgroup, *subgroup.iter_subgroups(), *subgroup.iter_datasets())
            )
        root_owner = type("BackgroundOwner", (), {"backgrounds": backgrounds})()
        copied_owners.append(root_owner)
        if not _background_references_are_valid(copied_owners, data_group):
            raise ValueError(
                "The destination does not contain every dataset or group referenced by a background."
            )
        _link_backgrounds(copied_owners, data_group)
        _uniquify_tree_names(data_group, datasets, groups)
        node.datasets.extend(datasets)
        node.subgroups.extend(groups)
        used_mask_names = {item.name for item in node.masks}
        for item in masks:
            item.name = _unique_name(item.name, used_mask_names)
            used_mask_names.add(item.name)
        used_background_names = {item.name for item in node.backgrounds}
        for item in backgrounds:
            item.name = _unique_name(item.name, used_background_names)
            used_background_names.add(item.name)
        node.masks.extend(masks)
        node.backgrounds.extend(backgrounds)
        return PasteResult(tuple([*datasets, *groups, *masks, *backgrounds]), True)
    if payload.kind == "dataset_group":
        _, groups, _, _ = clone_dataset_tree((), payload.items)
        _uniquify_tree_names(data_group, (), groups)
        node.subgroups.extend(groups)
        return PasteResult(tuple(groups), True)
    if payload.kind == "mask":
        owner = dataset if target_role in {"dataset", "masks", "mask"} else node
        copied = copy.deepcopy(payload.items)
        used = {item.name for item in owner.masks}
        for item in copied:
            item.name = _unique_name(item.name, used)
            used.add(item.name)
        owner.masks.extend(copied)
        return PasteResult(tuple(copied), True)
    if payload.kind == "background":
        owner = dataset if target_role in {"dataset", "backgrounds", "background"} else node
        copied = tuple(_clone_background(item) for item in payload.items)
        probe = type("BackgroundOwner", (), {"backgrounds": copied})()
        if not _background_references_are_valid((probe,), data_group):
            raise ValueError(
                "The destination does not contain the background source dataset or group."
            )
        used = {item.name for item in owner.backgrounds}
        for item in copied:
            item.name = _unique_name(item.name, used)
            used.add(item.name)
        _link_backgrounds((probe,), data_group)
        owner.backgrounds.extend(copied)
        return PasteResult(tuple(copied), True)
    if payload.kind == "model":
        copied = copy.deepcopy(payload.items)
        used = set(data_group.models)
        name_map: dict[str, str] = {}
        for item in copied:
            old_name = item.name
            item.name = _unique_name(old_name, used)
            name_map[old_name] = item.name
            used.add(item.name)
        for item in copied:
            _remap_model_constraints(item, name_map)
            data_group.models[item.name] = item
        return PasteResult(tuple(copied))
    if payload.kind == "fit":
        destination_dataset_names = set(data_group.dataset_names)
        destination_model_names = set(data_group.models)
        for entry in payload.items:
            for candidate in _iter_fits(entry):
                snapshot = candidate.snapshot
                if any(
                    item.get("name") not in destination_dataset_names
                    for item in snapshot.get("datasets", ())
                ) or any(
                    item.get("name") not in destination_model_names
                    for item in snapshot.get("models", ())
                ):
                    raise ValueError(
                        "The destination does not contain every dataset or model referenced by this fit."
                    )
        copied = list(copy.deepcopy(payload.items))
        _fresh_fit_ids(copied)
        for entry in copied:
            for candidate in _iter_fits(entry):
                if candidate.kind in {"initial", "current"}:
                    candidate.kind = "result"
        data_group.fits.extend(copied)
        return PasteResult(tuple(copied))
    if payload.kind == "analysis":
        copied = copy.deepcopy(payload.items)
        for item in copied:
            item.id = uuid4().hex
            # Results and their output IDs belong to the original recipe. The
            # pasted item is an independent, ready-to-run recipe.
            item.result = None
        data_group.analyses.extend(copied)
        return PasteResult(tuple(copied))
    if payload.kind == "plot":
        dataset_ids = {item.id for item in data_group.iter_datasets()}
        fit_ids: set[str] = set()

        def collect(entry: FitTimelineEntry) -> None:
            fit_ids.add(entry.id)
            for child in entry.children:
                collect(child)

        for entry in data_group.fits:
            collect(entry)
        for plot in payload.items:
            if any(
                (ref.dataset_id and ref.dataset_id not in dataset_ids)
                or (ref.fit_id and ref.fit_id not in fit_ids)
                for ref in plot.sources
            ):
                raise ValueError(
                    "The destination does not contain every dataset or fit referenced by this plot."
                )
        copied = copy.deepcopy(payload.items)
        used = {item.name for item in data_group.plots}
        for item in copied:
            item.id = uuid4().hex
            item.name = _unique_name(item.name, used)
            used.add(item.name)
        data_group.plots.extend(copied)
        return PasteResult(tuple(copied))
    raise ValueError(f"Unsupported clipboard payload {payload.kind!r}.")


def _iter_fits(entry: FitTimelineEntry) -> Iterable[FitTimelineEntry]:
    yield entry
    for child in entry.children:
        yield from _iter_fits(child)


def _remap_model_constraints(
    model: ModelComponentSpec, name_map: dict[str, str]
) -> None:
    """Remap qualified component names only in documented constraint fields."""

    changed = {old: new for old, new in name_map.items() if old != new}
    if not changed:
        return
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(
            re.escape(name) for name in sorted(changed, key=len, reverse=True)
        ) + r")\."
    )

    def replace(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return pattern.sub(lambda match: f"{changed[match.group(1)]}.", value)

    for constraint in model.constraints:
        for field in ("parameter", "reference", "expression"):
            if field in constraint:
                constraint[field] = replace(constraint[field])
