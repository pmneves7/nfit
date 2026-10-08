"""Signature-checked migration of lazy binnings after project-object copies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .cache_utils import dataset_content_signature
from .pipeline import DataGroup, DatasetEntry, DatasetGroup
from .project_transfer import _ORIGIN


@dataclass(frozen=True)
class TransferBinning:
    cache: Any
    key: Any
    signature: str
    config: dict


def _normal(value, *, strings=None, identities=None):
    if isinstance(value, (list, tuple)):
        key = json.dumps(value, sort_keys=True)
        if identities and key in identities:
            return _normal(identities[key])
        return [_normal(item, strings=strings, identities=identities) for item in value]
    if isinstance(value, dict):
        return {key: _normal(item, strings=strings, identities=identities) for key, item in value.items()}
    if isinstance(value, str):
        if strings and value in strings:
            return strings[value]
        if value.startswith(("[", "{")):
            try:
                return _normal(json.loads(value), strings=strings, identities=identities)
            except ValueError:
                pass
    return value


def install_selection_binnings(selection, destination, result, target_group, *, resolve):
    """Retain only caches with unchanged numerical settings and source contents.

    IDs, unique display names and byte-identical adopted data archives may
    change. Added masks, different lattice settings, changed files or any
    other numerical difference reject reuse. No arrays are decoded here.
    ``resolve`` is the owner's narrow configuration/key/signature callback.
    """
    if selection.path is None:
        return 0
    new_roots = list(result.items) if selection.payload.kind == "group" else [target_group]
    copied_datasets, copied_groups = [], []
    for item in result.items:
        if isinstance(item, DataGroup):
            copied_datasets.extend(item.iter_datasets())
            copied_groups.extend(item.iter_subgroups())
        elif isinstance(item, DatasetGroup):
            copied_datasets.extend(item.iter_datasets())
            copied_groups.extend([item, *item.iter_subgroups()])
        elif isinstance(item, DatasetEntry):
            copied_datasets.append(item)
    origin = _ORIGIN
    data_map = {identity: item for item in copied_datasets for identity in item.metadata.get(origin, [])}
    group_map = {identity: item for item in copied_groups for identity in item.metadata.get(origin, [])}
    strings, identities = {}, {}
    unchanged = True

    def remap_name(source, target):
        nonlocal unchanged
        if source in strings and strings[source] != target:
            # Duplicate display names cannot prove a signature correspondence.
            unchanged = False
        strings[source] = target

    for root in selection.project.data_groups:
        for source in root.iter_datasets():
            target = data_map.get(source.id)
            if target is None:
                continue
            strings[source.id] = target.id
            remap_name(source.name, target.name)
            previous = selection.description["identities"].get(source.id)
            if previous is not None:
                current = list(dataset_content_signature(source))
                if not hasattr(source, "_project_artifact_source") and previous != current:
                    unchanged = False
                identities[json.dumps(previous, sort_keys=True)] = list(dataset_content_signature(target))
        for source in root.iter_subgroups():
            target = group_map.get(source.id)
            if target is not None:
                strings[source.id] = target.id
                remap_name(source.name, target.name)
    if not unchanged:
        return 0
    restored = 0
    for record in selection.project.settings.get("binning_cache_entries", []):
        try:
            entry = dict(record)
            index = int(entry["group_index"])
            if index < 0 or index >= len(new_roots) or new_roots[index] is None:
                continue
            root = new_roots[index]
            entry["group_index"] = destination.data_groups.index(root)
            if entry.get("dataset_id"):
                target = data_map.get(entry["dataset_id"])
                if target is None:
                    continue
                entry["dataset_id"] = target.id
            if entry.get("node_id"):
                target = group_map.get(entry["node_id"])
                if target is None:
                    continue
                entry["node_id"] = target.id
            binding = resolve(destination, entry)
            if binding is None:
                continue
            saved = _normal(json.loads(record["signature"]), strings=strings, identities=identities)
            current = _normal(json.loads(binding.signature))
            if saved != current:
                continue
            binding.cache.set_project_backing(binding.key, signature=binding.signature,
                                             project_path=selection.path, member=record["member"], lazy=True)
            binding.config["stale"] = False
            restored += 1
        except (IndexError, KeyError, StopIteration, TypeError, ValueError):
            continue
    return restored
