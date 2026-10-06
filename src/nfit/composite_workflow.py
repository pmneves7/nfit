"""Portable source closure and scientific topology for composite workflows.

The recipe contains source descriptors, native reduction recipes and ordinary
workspace/collection state. Numerical caches, models and historical fit state
are not workflow inputs. Source files and calibration files remain required.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .pipeline import DataGroup, DatasetGroup
from .project_history import (
    _background_from_dict,
    _background_to_dict,
    _link_project_backgrounds,
    _mask_from_dict,
    _mask_to_dict,
)
from .project_io import NfitProject, _dataset_to_dict
from .reduction_recipes import export_reduction_recipe, reduction_family, replay_reduction_recipe

COMPOSITE_RECIPE_VERSION = 1


def _metadata(value):
    # These keys reference numerical cache assets rather than scientific input.
    return copy.deepcopy({key: item for key, item in value.items()
                          if not key.startswith("_") and key not in {
                              "raw_dgs_reduction_cache", "reduction_recipe",
                          }})


def export_composite_recipe(project: NfitProject, group_name: str, *, node_id: str | None = None) -> dict[str, Any]:
    """Export the selected composite's source and ancestor dependency closure.

    Native DGS/CORELLI/MDEvent collections and ordinary file-backed datasets
    are supported. Source-less/replaced data, callable transforms and live
    derived-analysis datasets must be exported as source datasets first.
    Ancestors retain masks, lattice, binning, scaling and background links;
    unrelated siblings are omitted. Linked background sources may belong to
    other workspaces. Disabled inputs within the selected tree are retained.
    """
    roots = [root for root in project.data_groups if root.name == group_name]
    if len(roots) != 1:
        raise ValueError(f"composite workspace name {group_name!r} is absent or ambiguous")
    selected_root = roots[0]
    target = selected_root if node_id is None else next(
        (node for node in selected_root.iter_subgroups() if node.id == node_id), None)
    if target is None:
        raise ValueError(f"composite collection {node_id!r} is not in {group_name!r}")
    nodes, datasets, parent, owner, root_keys = {}, {}, {}, {}, {}

    def index(node, root, ancestors):
        if isinstance(node, DatasetGroup):
            if node.id in nodes:
                raise ValueError(f"duplicate collection ID {node.id!r}")
            nodes[node.id] = node
        parent[id(node)] = ancestors
        owner[id(node)] = root
        for dataset in node.datasets:
            if dataset.id in datasets:
                raise ValueError(f"duplicate dataset ID {dataset.id!r}")
            datasets[dataset.id] = dataset
            parent[id(dataset)] = (*ancestors, node)
            owner[id(dataset)] = root
        for child in node.subgroups:
            index(child, root, (*ancestors, node))

    for i, root in enumerate(project.data_groups):
        root_keys[id(root)] = f"workspace:{i}"
        index(root, root, ())
    retained_nodes, retained_data = set(), set()
    pending = []

    def retain(item, subtree=False):
        is_dataset = not isinstance(item, (DataGroup, DatasetGroup))
        container = retained_data if is_dataset else retained_nodes
        if id(item) not in container:
            container.add(id(item))
            pending.append(item)
        for ancestor in parent[id(item)]:
            retain(ancestor)
        if subtree and not is_dataset:
            for dataset in item.datasets:
                retain(dataset)
            for child in item.subgroups:
                retain(child, True)

    retain(target, True)
    while pending:
        item = pending.pop()
        for background in item.backgrounds:
            source = (nodes.get(background.source_group_id)
                      if background.source_group_id else datasets.get(background.source_dataset_id))
            if source is None:
                if background.enabled:
                    raise ValueError(f"background {background.name!r} refers to an absent source")
                continue
            retain(source, isinstance(source, DatasetGroup))

    recipes = {}

    def state(node):
        payload = {
            "name": node.name, "metadata": _metadata(node.metadata),
            "masks": [_mask_to_dict(mask) for mask in node.masks],
            "backgrounds": [_background_to_dict(bg) for bg in node.backgrounds],
            "subgroups": [state(child) for child in node.subgroups if id(child) in retained_nodes],
        }
        if isinstance(node, DatasetGroup):
            payload.update(id=node.id, enabled=node.enabled, resolution=copy.deepcopy(node.resolution))
        else:
            payload.update(key=root_keys[id(node)], lattice_parameters=copy.deepcopy(node.lattice_parameters),
                           spacegroup=node.spacegroup)
        family = reduction_family(node) if isinstance(node, DatasetGroup) else None
        if family:
            recipes[node.id] = export_reduction_recipe(node)
            payload["reduction_recipe_id"] = node.id
            # The native recipe owns source membership and scientific defaults.
            payload["metadata"].pop("raw_dgs", None)
            payload["metadata"].pop("mdevent", None)
            payload["metadata"].pop("source_selection", None)
            payload.pop("masks")
            payload.pop("backgrounds")
            payload["datasets"] = []
        else:
            payload["datasets"] = []
            for dataset in node.datasets:
                if id(dataset) not in retained_data:
                    continue
                if dataset.metadata.get("derived_recipe") or dataset.metadata.get("derived_from_analysis"):
                    raise ValueError(f"dataset {dataset.name!r} is a live derived source; export its source data first")
                if not dataset.metadata.get("source_file"):
                    raise ValueError(f"dataset {dataset.name!r} has no reproducible source_file")
                if dataset.kind in {"raw_dgs_nexus", "mdevent"}:
                    raise ValueError(f"dataset {dataset.name!r} requires its native source collection")
                descriptor = _dataset_to_dict(dataset)
                descriptor["metadata"] = _metadata(descriptor["metadata"])
                payload["datasets"].append(descriptor)
        return payload

    result = {
        "version": COMPOSITE_RECIPE_VERSION,
        "target_workspace": root_keys[id(selected_root)], "target_node_id": node_id,
        "workspaces": [state(root) for root in project.data_groups if id(root) in retained_nodes],
        "reduction_recipes": recipes,
    }
    try:
        # Validate portability without coercing unknown objects into strings.
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("composite scientific state must contain finite JSON values") from exc


def replay_composite_recipe(recipe: dict[str, Any], *, progress_callback=None) -> tuple[DataGroup, DatasetGroup | None]:
    """Rebuild a portable composite closure without a saved project or Qt.

    Returns its owning workspace and optional selected collection. Scientific
    arrays are imported through the same public import service; native events
    remain lazy until reduction/binning is requested. Runtime background owner
    context is resolved after all source collections have been reconstructed.
    """
    from .project_dataset_io import _load_nfit_dataset_file
    from .project_imports import dataset_entry_from_path

    if recipe.get("version") != COMPOSITE_RECIPE_VERSION:
        raise ValueError("unsupported composite workflow recipe version")
    payload = copy.deepcopy(recipe)
    native = payload.get("reduction_recipes", {})
    nodes, roots = {}, {}

    def build(spec, root=False):
        if root:
            node = DataGroup(spec["name"], lattice_parameters=spec.get("lattice_parameters"),
                             spacegroup=spec.get("spacegroup"))
            if spec["key"] in roots:
                raise ValueError("duplicate composite workspace key")
            roots[spec["key"]] = node
        elif "reduction_recipe_id" in spec:
            node = replay_reduction_recipe(native[spec["reduction_recipe_id"]], progress_callback=progress_callback)
            if node.id != spec["id"]:
                raise ValueError("native reduction recipe collection ID differs from topology")
        else:
            node = DatasetGroup(spec["name"], id=spec["id"])
        if not root:
            if node.id in nodes:
                raise ValueError("duplicate composite collection ID")
            nodes[node.id] = node
            node.name = spec["name"]
            node.enabled = bool(spec.get("enabled", True))
            node.resolution = spec.get("resolution", {})
        node.metadata.update(spec.get("metadata", {}))
        # Native masks/backgrounds/run state have one authoritative recipe.
        if root or "reduction_recipe_id" not in spec:
            node.masks = [_mask_from_dict(item) for item in spec.get("masks", [])]
            node.backgrounds = [_background_from_dict(item) for item in spec.get("backgrounds", [])]
            for descriptor in spec.get("datasets", []):
                metadata = descriptor.get("metadata", {})
                entry = dataset_entry_from_path(metadata["source_file"],
                    data_type=descriptor["data_type"], importer_name=metadata.get("importer"),
                    importer_options=metadata.get("import_options"), dataset_file_loader=_load_nfit_dataset_file)
                for key in ("id", "name", "kind", "data_type", "parameters", "enabled", "fit_weight",
                            "scale_factor", "scale_factor_vary", "scale_factor_group"):
                    if key in descriptor:
                        setattr(entry, key, descriptor[key])
                entry.metadata.update(metadata)
                entry.masks = [_mask_from_dict(item) for item in descriptor.get("masks", [])]
                entry.backgrounds = [_background_from_dict(item) for item in descriptor.get("backgrounds", [])]
                node.datasets.append(entry)
        node.subgroups = [build(child) for child in spec.get("subgroups", [])]
        return node

    project = NfitProject([build(spec, True) for spec in payload.get("workspaces", [])])
    _link_project_backgrounds(project)
    for workspace in project.data_groups:
        for owner in (workspace, *workspace.iter_subgroups(), *workspace.iter_datasets()):
            for bg in owner.backgrounds:
                if bg.enabled and bg.source_entry is None and bg.source_group is None:
                    raise ValueError(f"background {bg.name!r} refers to an absent replay source")
    root = roots[payload["target_workspace"]]
    from .dataset_criteria import prepare_dataset_criteria

    prepare_dataset_criteria(root, progress_callback=progress_callback)
    node_id = payload.get("target_node_id")
    node = nodes[node_id] if node_id is not None else None
    if node is not None and not any(item is node for item in root.iter_subgroups()):
        raise ValueError("selected composite collection is not in its workspace")
    return root, node
