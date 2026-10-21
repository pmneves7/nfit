"""Metadata-only viewer planning and preparation of individual named binnings.

Preparation callbacks keep this service independent of GUI coordinators and
the project-data facade. Catalog construction never accesses numerical caches.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .binning_lifecycle import binning_is_enabled
from .pipeline import DataGroup, DatasetEntry
from .project_composites import (
    GROUP_COMPOSITE_BINNINGS_KEY,
    _composite_dataset_name,
    _composite_scope,
    _CompositeScope,
    composite_dataset_entry,
    data_group_composite_binnings,
    data_group_composite_enabled,
)
from .viewer_data import DeferredViewerDatasets, ViewerDatasetDescriptor


def _composite_scopes(
    group: DataGroup,
    *,
    scope_factory: Callable[[DataGroup, Any], Any] = _composite_scope,
):
    yield group
    for subgroup in group.iter_subgroups():
        yield scope_factory(group, subgroup)


def _binning_view_name(source_name: str, binning_name: str) -> str:
    return f"{source_name} · {binning_name}"


def project_binning_targets(
    project: Any,
    *,
    rebin_key: str,
    dataset_binnings: Callable[[DatasetEntry], list[dict[str, Any]]],
    composite_binnings: Callable[[Any], list[dict[str, Any]]] = data_group_composite_binnings,
    composite_enabled: Callable[[Any], bool] = data_group_composite_enabled,
    composite_scopes: Callable[[DataGroup], Any] = _composite_scopes,
    binning_view_name: Callable[[str, str], str] = _binning_view_name,
) -> list[tuple[str, str, DataGroup, Any, str, dict[str, Any]]]:
    """Return enabled independent binnings and their live configuration references.

    Registry callbacks retain the normal configuration sanitization without
    importing the project-data facade or loading numerical payloads.
    """

    from .project_binning_policy import controlling_composite

    targets = []
    for group in project.data_groups:
        for dataset in group.iter_datasets():
            if controlling_composite(group, dataset) is not None:
                continue
            if not isinstance(dataset.parameters.get(rebin_key), dict):
                continue
            for binning in dataset_binnings(dataset):
                if binning["config"].get("enabled") and binning_is_enabled(binning["config"]):
                    targets.append(
                        (
                            "dataset",
                            binning_view_name(dataset.name, binning["name"]),
                            group,
                            dataset,
                            binning["id"],
                            binning["config"],
                        )
                    )
        for scope in composite_scopes(group):
            node = scope.node if isinstance(scope, _CompositeScope) else scope
            if controlling_composite(group, node) is not None:
                continue
            if (
                not composite_enabled(scope)
                and not isinstance(
                    scope.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY), dict
                )
            ):
                continue
            for binning in composite_binnings(scope):
                if binning["config"].get("enabled") and binning_is_enabled(binning["config"]):
                    targets.append(
                        (
                            "dataset group",
                            binning_view_name(scope.name, binning["name"]),
                            group,
                            scope,
                            binning["id"],
                            binning["config"],
                        )
                    )
    return targets


@dataclass(frozen=True)
class ProjectViewerItem:
    """One selectable binning, containing configuration references only."""

    source_name: str
    dataset: DatasetEntry | None
    scope: Any
    binning: dict[str, Any] | None

    @property
    def name(self) -> str:
        if self.binning is None or self.binning.get("fit", False):
            return self.source_name
        return f"{self.source_name} · {self.binning['name']}"


def viewer_dataset_is_selectable(dataset: DatasetEntry) -> bool:
    """Select independent results and declared lazy results, without file I/O.

    Acquisition-only event entries require their collection histogram. Empty
    placeholders have no view to select. Ordinary data and declared source,
    project-artifact or derived outputs remain selectable even when unloaded,
    excluded from fitting, or temporarily unavailable on disk.
    """

    metadata = dataset.metadata
    if dataset.data is not None or metadata.get("derived_recipe"):
        return True
    if dataset.kind in {"raw_dgs_nexus", "mdevent"}:
        return False
    return any(
        bool(metadata.get(key)) for key in (
            "source_file", "project_artifact_path", "analysis_artifact_path",
        )
    )


def has_viewer_candidates(group: DataGroup) -> bool:
    """Test declared viewer availability without validating scientific inputs.

    Navigation uses the catalog's collection-ownership and source policy.
    Source freshness, conditions and numerical readiness are checked when a
    result is prepared, rather than while enabling a navigation control.
    """
    def visit(node):
        if data_group_composite_enabled(_composite_scope(group, node)):
            return True
        return any(viewer_dataset_is_selectable(dataset) for dataset in node.datasets) or any(
            visit(child) for child in node.subgroups
        )

    return visit(group)


def plan_viewer_items(
    group: DataGroup,
    *,
    use_composite: bool,
    dataset_binnings: Callable[[DatasetEntry], list[dict[str, Any]]],
    disabled_binning_available: Callable[[Any, Any, dict[str, Any]], bool] | None = None,
) -> list[ProjectViewerItem]:
    """List the same effective sources as the eager viewer without loading data."""

    result = []

    def add(source_name, dataset, scope, binnings):
        if not binnings:
            result.append(ProjectViewerItem(source_name, dataset, scope, None))
        for item in binnings:
            if (
                not binning_is_enabled(item["config"])
                and disabled_binning_available is not None
                and not disabled_binning_available(dataset, scope, item)
            ):
                continue
            if item.get("fit", False) or item["config"].get("enabled", False):
                result.append(ProjectViewerItem(source_name, dataset, scope, item))

    def visit(node):
        scope = _composite_scope(group, node)
        if use_composite and data_group_composite_enabled(scope):
            add(_composite_dataset_name(scope), None, scope, data_group_composite_binnings(scope))
            return
        for dataset in node.datasets:
            if not viewer_dataset_is_selectable(dataset):
                continue
            add(dataset.name, dataset, None, dataset_binnings(dataset))
        for child in node.subgroups:
            visit(child)

    visit(group)
    return result


def viewer_binning_entry(
    dataset: DatasetEntry,
    binning: dict[str, Any],
    *,
    enabled: bool,
    rebin_key: str,
    derived_key: str,
    composite_data: Any = None,
) -> DatasetEntry:
    """Create the shared eager/deferred viewer alias for a named binning."""

    fit = bool(binning.get("fit", False))
    composite = bool(dataset.metadata.get("composite"))
    if fit:
        alias = dataset.copy(name=dataset.name)
        alias.id = dataset.id
        alias._viewer_source_entry = dataset
        alias.enabled = enabled
    elif composite:
        alias = dataset.copy(data=composite_data, name=f"{dataset.name} · {binning['name']}")
    else:
        parameters = copy.deepcopy(dataset.parameters)
        parameters[rebin_key] = binning["config"]
        alias = dataset.copy(name=f"{dataset.name} · {binning['name']}", parameters=parameters)
        if not isinstance(dataset.metadata.get(derived_key), dict):
            alias._viewer_source_entry = dataset
            alias._viewer_rebin_config = binning["config"]
            alias._viewer_cache_id = binning["id"]
    owner = getattr(dataset, "_derived_owner_group", None)
    if owner is not None:
        alias._derived_owner_group = owner
    composite_owner = getattr(dataset, "_viewer_composite_cache_owner", None)
    if composite_owner is not None:
        alias._viewer_composite_cache_owner = composite_owner
        alias._viewer_composite_binning_id = str(binning["id"])
        alias._viewer_binning_config = binning["config"]
    alias._viewer_source_dataset_id = dataset.id
    if not fit:
        alias.id = f"{dataset.id}:{binning['id']}"
        alias.fit_weight = 0.0
        alias.scale_factor_vary = False
    alias.metadata = {
        **copy.deepcopy(dataset.metadata),
        "binning_id": binning["id"],
        "binning_name": binning["name"],
        "source_dataset_name": dataset.name,
        **({"visualization_binning": True} if not fit else {}),
    }
    return alias


def deferred_viewer_datasets(
    group: DataGroup,
    items: list[ProjectViewerItem],
    *,
    prepare: Callable[[DatasetEntry], Any],
    enabled: Callable[[DatasetEntry], bool],
    rebin_key: str,
    derived_key: str,
    group_keys: list[str],
    selected_dataset_name: str | None = None,
    selected_binning_name: str | None = None,
    force_rebin: bool = True,
    progress_callback: Any = None,
):
    """Return a lazy sequence; each request computes only its selected binning."""

    if not items:
        return [], []
    descriptors = [
        ViewerDatasetDescriptor(
            name=item.name,
            source_dataset_name=item.source_name,
            binning_name=item.binning["name"] if item.binning else "Default",
            binning_id=str(item.binning["id"]) if item.binning else "fit",
            group_key=key,
            crystal_context={
                "spacegroup": group.spacegroup,
                "lattice_parameters": copy.deepcopy(group.lattice_parameters),
            },
            source_dataset_id=item.dataset.id if item.dataset is not None else "",
            source_scope_id=(
                id(getattr(item.scope, "node", item.scope))
                if item.scope is not None else None
            ),
        )
        for item, key in zip(items, group_keys, strict=True)
    ]
    candidates = [
        index for index, item in enumerate(items)
        if selected_dataset_name in (item.name, item.source_name)
    ]
    initial = next(
        (index for index in candidates if descriptors[index].binning_name == selected_binning_name),
        candidates[0] if candidates else 0,
    )

    def load(index):
        item = items[index]
        if progress_callback is not None:
            progress_callback({
                "stage": "viewer_prepare", "iteration": 0, "total": 0,
                "message": f"Loading viewer binning: {item.name}",
            })
        dataset = item.dataset
        if dataset is None:
            named = item.binning is not None and not item.binning.get("fit", False)
            dataset = composite_dataset_entry(
                item.scope, force_rebin=force_rebin, progress_callback=progress_callback,
                config_override=item.binning["config"] if named else None,
                binning_id=item.binning["id"] if named else None,
            )
        if item.binning is not None:
            dataset = viewer_binning_entry(
                dataset, item.binning, enabled=enabled(dataset),
                rebin_key=rebin_key, derived_key=derived_key, composite_data=dataset.data,
            )
        result = prepare(dataset)
        if result is None:
            raise ValueError(f"No viewer data available for {item.name}")
        return result

    return DeferredViewerDatasets(descriptors, load, initial_index=initial), [item.name for item in items]
