"""Editable MACS background source records without project dataset ownership.

The shared ``background_files`` list owns membership. Sparse source records
store individual choices and use declared ``source_file`` paths so project
relocation and script export follow the same references as sample files.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path

from .macs_reduction import MACS_REDUCTION_FORMAT, adopt_macs_reduction, macs_group_config
from .reduction_recipes import (
    ReductionEdit,
    _mark_stale,
    effective_reduction_config,
    ensure_reduction_recipe,
    validated_reduction_settings,
)

MACS_BACKGROUND_SOURCE_OVERRIDES = "background_source_overrides"


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("MACS background source must be a nonempty file path")
    return str(Path(value).expanduser().resolve())


def normalize_macs_background_source_overrides(config):
    """Validate sparse source records and drop references outside the file list.

    This pure helper accepts a reduction configuration, never opens files, and
    returns fresh JSON containers. Only supported per-run MACS reduction fields
    can be overridden; stream and shared subtraction settings stay group owned.
    """
    files = config.get("background_files", ())
    if not isinstance(files, (tuple, list)):
        raise ValueError("MACS background files must be a list of file paths")
    paths = {_path(path) for path in files}
    records = config.get(MACS_BACKGROUND_SOURCE_OVERRIDES, [])
    if not isinstance(records, (tuple, list)):
        raise ValueError("MACS background source overrides must be a list of records")
    result, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("MACS background source overrides must contain mappings")
        unknown = set(record) - {"source_file", "enabled", "scale_factor", "reduction_overrides"}
        if unknown:
            raise ValueError(f"unsupported MACS background source settings: {', '.join(sorted(unknown))}")
        path = _path(record.get("source_file"))
        if path not in paths:
            continue
        if path in seen:
            raise ValueError("duplicate MACS background source override records")
        seen.add(path)
        enabled = record.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("MACS background enabled must be a boolean")
        scale = record.get("scale_factor", 1.0)
        if isinstance(scale, bool):
            raise ValueError("MACS background scale factor must be a finite number")
        try:
            scale = float(scale)
        except (ValueError, TypeError) as error:
            raise ValueError("MACS background scale factor must be a finite number") from error
        if not math.isfinite(scale):
            raise ValueError("MACS background scale factor must be a finite number")
        overrides = validated_reduction_settings(
            MACS_REDUCTION_FORMAT, record.get("reduction_overrides", {}), base=config, per_run=True)
        if enabled and scale == 1.0 and not overrides:
            continue
        result.append({"source_file": path, "enabled": enabled, "scale_factor": scale,
                       "reduction_overrides": overrides})
    return result


def _sources(config):
    overrides = {record["source_file"]: record
                 for record in normalize_macs_background_source_overrides(config)}
    return [copy.deepcopy(overrides.get(path, {"source_file": path, "enabled": True,
                                              "scale_factor": 1.0, "reduction_overrides": {}}))
            for path in dict.fromkeys(_path(path) for path in config.get("background_files", ()))]


def list_macs_background_sources(group):
    """List background file choices, including disabled files, without loading.

    Each returned dictionary has ``source_file``, ``enabled``, dimensionless
    ``scale_factor``, and explicit ``reduction_overrides``. Unedited files use
    true, one, and an empty mapping respectively. Returned records are copies.
    """
    config = macs_group_config(group)
    return [] if config is None else _sources(config)


def effective_macs_background_config(group, path, *, sample_dataset=None):
    """Overlay one background's choices on its matched sample's reduction.

    Without ``sample_dataset``, group defaults supply the inherited choices.
    The stream, coordinate transform, vanadium calibration, and other unedited
    settings follow the sample. This function does not import the source.
    """
    path = _path(path)
    record = next((record for record in list_macs_background_sources(group)
                   if record["source_file"] == path), None)
    if record is None:
        raise ValueError("background source must belong to this MACS group's background_files")
    result = effective_reduction_config(group, sample_dataset)
    result.update(validated_reduction_settings(
        group, record["reduction_overrides"], base=result, per_run=True))
    return result


def set_macs_background_source_settings(
    group, path, *, enabled=None, scale_factor=None, reduction_overrides=None, inherit=False,
):
    """Edit one listed background and invalidate its dependent composite grids.

    Reduction overrides merge by key. ``inherit=True`` with a mapping removes
    only those reduction keys; without a mapping it resets the complete record
    to enabled, unit scale, and inherited reduction. Validation is atomic and
    covers every matched sample's effective configuration. Sample point data
    and their source reduction caches are retained.
    """
    if macs_group_config(group) is None:
        raise ValueError("background source settings require a MACS reduction group")
    path = _path(path)
    records = list_macs_background_sources(group)
    record = next((record for record in records if record["source_file"] == path), None)
    if record is None:
        raise ValueError("background source must belong to this MACS group's background_files")
    previous = copy.deepcopy(records)
    if inherit and reduction_overrides is None:
        if enabled is not None or scale_factor is not None:
            raise ValueError("resetting inherited background settings cannot include explicit choices")
        record.update(enabled=True, scale_factor=1.0, reduction_overrides={})
    else:
        if enabled is not None:
            record["enabled"] = enabled
        if scale_factor is not None:
            record["scale_factor"] = scale_factor
        if reduction_overrides is not None:
            if not isinstance(reduction_overrides, Mapping):
                raise TypeError("background reduction overrides must be a mapping")
            if inherit:
                # Validate ownership and key names even when values are unused.
                from .reduction_recipes import reduction_settings_schema

                allowed = {field.key for field in reduction_settings_schema(group) if field.per_run}
                if set(reduction_overrides) - allowed:
                    raise ValueError("inherit requires supported per-run reduction settings")
                for key in reduction_overrides:
                    record["reduction_overrides"].pop(key, None)
            else:
                record["reduction_overrides"].update(reduction_overrides)
    config = effective_reduction_config(group)
    config[MACS_BACKGROUND_SOURCE_OVERRIDES] = records
    normalized = normalize_macs_background_source_overrides(config)
    config[MACS_BACKGROUND_SOURCE_OVERRIDES] = normalized
    for dataset in group.datasets:
        effective = effective_reduction_config(group, dataset)
        for source in normalized:
            validated_reduction_settings(group, source["reduction_overrides"], base=effective, per_run=True)
    changed = previous != _sources(config)
    if changed:
        adopt_macs_reduction(group)
        if normalized:
            group.metadata["macs"][MACS_BACKGROUND_SOURCE_OVERRIDES] = normalized
        else:
            group.metadata["macs"].pop(MACS_BACKGROUND_SOURCE_OVERRIDES, None)
        _mark_stale(group, [])
        ensure_reduction_recipe(group)
    return ReductionEdit(tuple(dataset.id for dataset in group.datasets) if changed else (),
                         (MACS_BACKGROUND_SOURCE_OVERRIDES,) if changed else (),
                         ("background",) if changed else (), changed)
