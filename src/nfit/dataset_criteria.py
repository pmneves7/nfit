"""Serializable dataset-group exclusion rules and scalar previews.

Selection does not mutate Enabled flags. GUI/read-only selection requires a
current scalar preview; explicit preparation and native binning may calculate
missing values. Threshold edits never change diagnostic cache signatures.
"""

from __future__ import annotations

import copy
import operator

import numpy as np

from .dataset_criterion_values import criterion_value, value_signature
from .pipeline import MaskSpec

DATASET_CRITERION_TYPE = "dataset_criterion"
_COMPARISONS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_DEFAULTS = dict(
    channel="elastic",
    energy_min=-0.5,
    energy_max=0.5,
    source="metadata/incident_energy",
    statistic="time_average",
    left_value=None,
    left_operator="<",
    right_value=None,
    right_operator=">",
)
DATASET_CRITERION_DEFINITION = {
    "label": "Dataset condition",
    "parameters": {
        key: dict(
            default=value,
            description=f"Dataset condition: {key}.",
            allowed="Configured in the dataset-condition editor.",
            type="scalar",
            example=str(value),
        )
        for key, value in _DEFAULTS.items()
    },
}
_CACHE_KEY = "dataset_criterion_values"


def dataset_criterion_mask(group, name="Dataset condition", **parameters):
    """Attach a whole-dataset exclusion rule; bounds are optional/inactive."""
    if any(mask.name == name for mask in group.masks):
        raise ValueError(f"Duplicate mask name: {name}")
    unknown = set(parameters) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown dataset condition parameters: {sorted(unknown)}")
    mask = MaskSpec(name, DATASET_CRITERION_TYPE, {**copy.deepcopy(_DEFAULTS), **parameters})
    criterion_matches(0.0, mask.parameters)
    group.masks.append(mask)
    return mask


def criterion_matches(value, parameters):
    """Evaluate ``left op value op right``; active comparisons are ANDed."""
    matches, active = True, False
    for side in ("left", "right"):
        bound = parameters.get(f"{side}_value")
        op = parameters.get(f"{side}_operator", "<" if side == "left" else ">")
        if op not in _COMPARISONS:
            raise ValueError("Comparison must be <, <=, >, or >=")
        if bound is None:
            continue
        if not np.isfinite(float(bound)):
            raise ValueError("Dataset condition limits must be finite or inactive")
        active = True
        first, second = (float(bound), value) if side == "left" else (value, float(bound))
        matches &= bool(_COMPARISONS[op](first, second))
    return bool(active and np.isfinite(value) and matches)


def _contexts(root):
    def visit(node, inherited, reduction_owner):
        owner = (
            node if "raw_dgs" in node.metadata or "mdevent" in node.metadata else reduction_owner
        )
        masks = [*inherited, *node.masks]
        for dataset in node.datasets:
            if dataset.metadata.get("composite") or dataset.metadata.get("derived_recipe"):
                continue
            yield dataset, owner or node, [*masks, *dataset.masks]
        for child in node.subgroups:
            yield from visit(child, masks, owner)

    return {dataset.id: (dataset, owner, masks) for dataset, owner, masks in visit(root, [], None)}


def _value(mask, owner, dataset, *, compute=False, progress_callback=None):
    signature = value_signature(owner, dataset, mask.parameters)
    cache = mask.metadata.get(_CACHE_KEY, {})
    current = cache.get(dataset.id)
    if current is None or current.get("signature") != signature:
        if not compute:
            raise ValueError(
                f"{mask.name}: calculate dataset values again; {dataset.name} is new or changed"
            )
        result = criterion_value(
            owner, dataset, mask.parameters, progress_callback=progress_callback
        )
        current = {"signature": signature, "result": result}
        cache = {**cache, dataset.id: current}
        mask.metadata[_CACHE_KEY] = cache
    return current["result"]


def dataset_criterion_preview(group, mask, *, progress_callback=None, refresh=False):
    """Calculate/reuse one scalar per descendant, including disabled runs.

    An incomplete preview is not published as a complete preview. Reduced-event
    archives stay lazy; only bounded chunks or numeric metadata are inspected.
    """
    if mask.type != DATASET_CRITERION_TYPE:
        raise ValueError("Expected a dataset condition mask")
    working = copy.deepcopy(mask)
    if refresh:
        working.metadata.pop(_CACHE_KEY, None)
    rows = []
    contexts = _contexts(group)
    for index, (dataset, owner, _masks) in enumerate(contexts.values(), 1):
        if progress_callback is not None:
            progress_callback(
                dict(
                    stage="dataset_criterion",
                    iteration=index - 1,
                    total=len(contexts),
                    message=f"Calculating dataset {index}/{len(contexts)}: {dataset.name}",
                )
            )
        result = _value(working, owner, dataset, compute=True, progress_callback=progress_callback)
        number = dataset.metadata.get("run_number", index)
        try:
            number = float(number)
        except (ValueError, TypeError):
            number = float(index)
        rows.append(
            dict(
                dataset_id=dataset.id,
                name=dataset.name,
                number=number,
                enabled=dataset.enabled,
                **result,
            )
        )
    # Keep only current input identities, bounding scalar cache storage when
    # sources or diagnostic settings are edited repeatedly.
    working.metadata[_CACHE_KEY] = {
        dataset_id: working.metadata[_CACHE_KEY][dataset_id] for dataset_id in contexts
    }
    if len({row.get("units", "") for row in rows}) > 1:
        raise ValueError(
            "Dataset diagnostic units differ; convert sources to consistent units first"
        )
    if progress_callback is not None:
        progress_callback(
            dict(
                stage="dataset_criterion",
                iteration=len(rows),
                total=len(rows),
                message=f"Calculated {len(rows)} dataset values",
            )
        )
    mask.metadata[_CACHE_KEY] = working.metadata.get(_CACHE_KEY, {})
    mask.metadata["dataset_criterion_preview"] = rows
    return rows


def filter_dataset_criteria(
    root, datasets, *, compute=False, progress_callback=None, require_current=True
):
    """Return original entries surviving ordered inherited dataset conditions."""
    contexts = None
    result = []
    units = {}
    # Keep the common case free of source stats and diagnostic calculations.
    if not any(
        mask.type == DATASET_CRITERION_TYPE and mask.enabled
        for node in [root, *root.iter_subgroups()]
        for mask in [*node.masks, *(m for d in node.datasets for m in d.masks)]
    ):
        return list(datasets)
    contexts = _contexts(root)
    for dataset in datasets:
        context = contexts.get(dataset.id)
        if context is None:  # Generated composites are already selected.
            result.append(dataset)
            continue
        _entry, owner, masks = context
        excluded = False
        for mask in masks:
            if not mask.enabled or mask.type != DATASET_CRITERION_TYPE:
                continue
            if all(mask.parameters.get(f"{s}_value") is None for s in ("left", "right")):
                continue
            try:
                diagnostic = _value(
                    mask, owner, dataset, compute=compute, progress_callback=progress_callback
                )
                value = diagnostic["value"]
                unit = diagnostic.get("units", "")
                if units.setdefault(id(mask), unit) != unit:
                    raise ValueError("Dataset diagnostic units differ; convert sources first")
            except (ValueError, OSError):
                if require_current:
                    raise
                continue  # Presentation only: pending conditions never hide runs.
            matches = criterion_matches(value, mask.parameters)
            if mask.invert:
                matches = not matches
            excluded = excluded and not matches if mask.additive else excluded or matches
        if not excluded:
            result.append(dataset)
    return result


def prepare_dataset_criteria(root, *, datasets=None, progress_callback=None):
    """Refresh active selection diagnostics before replaying a group recipe."""
    filter_dataset_criteria(
        root,
        list(root.iter_datasets()) if datasets is None else datasets,
        compute=True,
        progress_callback=progress_callback,
    )
