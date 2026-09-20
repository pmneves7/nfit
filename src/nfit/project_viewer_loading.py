"""Metadata-only viewer planning and preparation of individual named binnings.

Preparation callbacks keep this service independent of GUI coordinators and
the project-data facade. Catalog construction never accesses numerical caches.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .pipeline import DataGroup, DatasetEntry, DatasetGroup
from .project_composites import (
    _composite_dataset_name,
    _composite_scope,
    composite_dataset_entry,
    data_group_composite_binnings,
    data_group_composite_enabled,
)
from .viewer_data import DeferredViewerDatasets, ViewerDatasetDescriptor


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


def plan_viewer_items(
    group: DataGroup,
    *,
    use_composite: bool,
    dataset_binnings: Callable[[DatasetEntry], list[dict[str, Any]]],
) -> list[ProjectViewerItem]:
    """List the same effective sources as the eager viewer without loading data."""

    result = []

    def add(source_name, dataset, scope, binnings):
        if not binnings:
            result.append(ProjectViewerItem(source_name, dataset, scope, None))
        for item in binnings:
            if item.get("fit", False) or item["config"].get("enabled", False):
                result.append(ProjectViewerItem(source_name, dataset, scope, item))

    def visit(node):
        scope = _composite_scope(group, node)
        if use_composite and data_group_composite_enabled(scope):
            add(_composite_dataset_name(scope), None, scope, data_group_composite_binnings(scope))
            return
        if isinstance(node, DatasetGroup) and isinstance(node.metadata.get("mdevent"), dict):
            return
        for dataset in node.datasets:
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
