"""Composite dataset services for project groups.

This module owns composite scopes, configuration, status, caching, metadata
stacking, background alignment, and materialization.  It is GUI-independent;
``project_data`` retains compatibility aliases for the established API.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .analysis.artifacts import dataset_artifact_bytes
from .backgrounds import subtract_background
from .cache_utils import (
    dataset_content_signature,
    scientific_cache_budget_bytes,
)
from .cache_utils import (
    lru_store as _lru_store,
)
from .dataset import PointData4D, PointListData
from .mdevent import (
    bin_mdevent_group,
    bin_mdevent_powder_group,
    project_powder_background_mdevent,
)
from .mdhisto import (
    MDHistoAxis,
    MDHistoChannel,
    MDHistoData,
    mdhisto_coverage_fraction,
    mdhisto_measured_bins,
)
from .performance import initialize_rebin_performance
from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from .project_archive import replace_dataset_artifact
from .project_coordinates import _identity_vector, _mdhisto_rebin_source_axis_vectors
from .project_history import _dataset_group_paths
from .project_imports import (
    GROUP_COMPOSITE_KEY,
    _adopt_imported_crystal,
    _unique_dataset_name,
    data_type_container,
)
from .project_masks import _mdhisto_with_nfit_masks, _point_data_with_nfit_masks
from .project_point_lists import prepared_point_list_data
from .raw_dgs import bin_raw_dgs_group, bin_raw_dgs_powder_group
from .rebin import rebin_nd
from .rebin_cache import SHARED_REBIN_CACHE_BUDGET, RebinCache
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY
from .symmetry import SymmetrySpec, symmetry_config

# Standalone defaults.  A configured project's live facade values remain
# authoritative so existing runtime configuration and monkeypatch seams work.
GROUP_COMPOSITE_NAME = "Composite"
DEFAULT_REBIN_MAX_BATCH_MB = 192
REBIN_AUTO_MAX_CONTRIBUTIONS = 5_000_000
REBIN_AUTO_MAX_OUTPUT_BINS = 2_000_000
REBIN_RESOLUTION_MODE_KEY = "resolution_mode"
GROUP_COMPOSITE_BINNINGS_KEY = "composite_binnings"
FIT_BINNING_ID = "fit"
CORELLI_ASSIGNMENT_VERSION = 1

_BACKEND_NAMESPACE: Mapping[str, Any] | None = None


def configure_composite_backend(namespace: Mapping[str, Any]) -> None:
    """Bind live lower-level dataset/rebin helpers without an import cycle."""

    global _BACKEND_NAMESPACE
    missing = sorted(name for name in _BACKEND_NAMES if name not in namespace)
    if missing:
        raise ValueError(f"composite backend is missing helpers: {', '.join(missing)}")
    _BACKEND_NAMESPACE = namespace


def _backend_function(name: str) -> Callable[..., Any]:
    def delegated(*args: Any, **kwargs: Any) -> Any:
        if _BACKEND_NAMESPACE is None:
            raise RuntimeError("composite backend has not been configured")
        return _BACKEND_NAMESPACE[name](*args, **kwargs)

    return delegated


def _backend_value(name: str, default: Any) -> Any:
    if _BACKEND_NAMESPACE is None:
        return default
    return _BACKEND_NAMESPACE.get(name, default)


_BACKEND_NAMES = (
    "DEFAULT_REBIN_MAX_BATCH_MB",
    "GROUP_COMPOSITE_NAME",
    "REBIN_AUTO_MAX_CONTRIBUTIONS",
    "REBIN_AUTO_MAX_OUTPUT_BINS",
    "REBIN_RESOLUTION_MODE_KEY",
    "_apply_dataset_backgrounds",
    "_composite_rebin_bin_edges",
    "_composite_rebin_step_sizes",
    "_dataset_data_point_count",
    "_dataset_rebin_source_points",
    "_default_rebin_axes",
    "_ensure_dataset_data_loaded",
    "_finite_coordinate_bounds",
    "_mask_signature",
    "_mdhisto_without_nfit_masks",
    "_migrate_rebin_axis_modes",
    "_point_data_histogram",
    "_rebin_axis_mode",
    "_rebin_axis_fractional",
    "_rebin_axis_vector",
    "_rebin_fractional_axes",
    "_rebin_grid_kwargs",
    "_rebin_max_batch_bytes",
    "_rebin_max_batch_mb",
    "_rebin_mdhisto_coverage",
    "_rebin_mean_weighting",
    "_rebin_minimum_coverage",
    "_rebin_minimum_samples",
    "_rebin_point_data",
    "_rebin_symmetry_matrices",
    "_resolve_auto_rebin_axes",
    "_resolve_data_driven_rebin_axes",
    "_sanitize_rebin_axis_config",
    "_symmetry_projected_coordinate_bounds",
    "_validate_mdhisto_rebin_basis",
    "_viewer_view_signature",
    "effective_dataset_masks",
)

_apply_dataset_backgrounds = _backend_function("_apply_dataset_backgrounds")
_composite_rebin_bin_edges = _backend_function("_composite_rebin_bin_edges")
_composite_rebin_step_sizes = _backend_function("_composite_rebin_step_sizes")
_dataset_data_point_count = _backend_function("_dataset_data_point_count")
_dataset_rebin_source_points = _backend_function("_dataset_rebin_source_points")
_default_rebin_axes = _backend_function("_default_rebin_axes")
_ensure_dataset_data_loaded = _backend_function("_ensure_dataset_data_loaded")
_finite_coordinate_bounds = _backend_function("_finite_coordinate_bounds")
_mask_signature = _backend_function("_mask_signature")
_mdhisto_without_nfit_masks = _backend_function("_mdhisto_without_nfit_masks")
_migrate_rebin_axis_modes = _backend_function("_migrate_rebin_axis_modes")
_point_data_histogram = _backend_function("_point_data_histogram")
_rebin_axis_mode = _backend_function("_rebin_axis_mode")
_rebin_axis_fractional = _backend_function("_rebin_axis_fractional")
_rebin_axis_vector = _backend_function("_rebin_axis_vector")
_rebin_fractional_axes = _backend_function("_rebin_fractional_axes")
_rebin_grid_kwargs = _backend_function("_rebin_grid_kwargs")
_rebin_max_batch_bytes = _backend_function("_rebin_max_batch_bytes")
_rebin_max_batch_mb = _backend_function("_rebin_max_batch_mb")
_rebin_mdhisto_coverage = _backend_function("_rebin_mdhisto_coverage")
_rebin_mean_weighting = _backend_function("_rebin_mean_weighting")
_rebin_minimum_coverage = _backend_function("_rebin_minimum_coverage")
_rebin_minimum_samples = _backend_function("_rebin_minimum_samples")
_rebin_point_data = _backend_function("_rebin_point_data")
_rebin_symmetry_matrices = _backend_function("_rebin_symmetry_matrices")
_resolve_auto_rebin_axes = _backend_function("_resolve_auto_rebin_axes")
_resolve_data_driven_rebin_axes = _backend_function("_resolve_data_driven_rebin_axes")
_sanitize_rebin_axis_config = _backend_function("_sanitize_rebin_axis_config")
_symmetry_projected_coordinate_bounds = _backend_function(
    "_symmetry_projected_coordinate_bounds"
)
_validate_mdhisto_rebin_basis = _backend_function("_validate_mdhisto_rebin_basis")
_viewer_view_signature = _backend_function("_viewer_view_signature")
effective_dataset_masks = _backend_function("effective_dataset_masks")

_COMPOSITE_DATA_CACHE: OrderedDict[Any, tuple[str, Any]] = RebinCache(
    SHARED_REBIN_CACHE_BUDGET
)
_COMPOSITE_DATA_CACHE_LIMIT = None
# Viewer-ready and composite bin results use the same machine-level allowance;
# this snapshots that preference when the application starts.
_COMPOSITE_DATA_CACHE_MAX_BYTES = scientific_cache_budget_bytes()


@dataclass(frozen=True)
class _CompositeScope:
    root: DataGroup
    node: DataGroup | DatasetGroup

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def metadata(self) -> dict[str, Any]:
        return self.node.metadata

    @property
    def masks(self) -> list[MaskSpec]:
        masks = list(self.root.masks)
        if self.node is not self.root:
            masks.extend(self.node.masks)
        return masks

    @property
    def backgrounds(self) -> list[BackgroundSpec]:
        backgrounds = list(self.root.backgrounds)
        if self.node is not self.root:
            backgrounds.extend(self.node.backgrounds)
        return backgrounds

    def iter_datasets(self):
        return self.node.iter_datasets()


def _composite_scope(
    root: DataGroup,
    node: DataGroup | DatasetGroup | None = None,
) -> DataGroup | _CompositeScope:
    return root if node is None or node is root else _CompositeScope(root, node)


def _composite_root(group: DataGroup | _CompositeScope) -> DataGroup:
    return group.root if isinstance(group, _CompositeScope) else group


def _composite_cache_key(
    group: DataGroup | _CompositeScope, binning_id: str | None = None
) -> Any:
    owner_id = id(group.node) if isinstance(group, _CompositeScope) else id(group)
    return owner_id if binning_id is None else (owner_id, str(binning_id))


def _composite_rebin_registry(group: DataGroup | _CompositeScope) -> dict[str, Any]:
    registry = group.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY)
    if not isinstance(registry, dict):
        registry = {"fit_name": "Default", "items": []}
        group.metadata[GROUP_COMPOSITE_BINNINGS_KEY] = registry
    registry.setdefault("fit_name", "Default")
    if not isinstance(registry.get("items"), list):
        registry["items"] = []
    return registry


def data_group_composite_binnings(
    group: DataGroup | _CompositeScope,
) -> list[dict[str, Any]]:
    """Return all named composite binnings, with the fit binning first."""

    fit = data_group_composite_config(group)
    registry = _composite_rebin_registry(group)
    fit_id = str(fit.setdefault("_binning_id", FIT_BINNING_ID))
    result = [{
        "id": fit_id,
        "name": str(registry.get("fit_name") or "Default"),
        "fit": True,
        "config": fit,
    }]
    seen = {fit_id}
    sanitized = []
    for item in registry["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("config"), dict):
            continue
        binning_id = str(item.get("id") or uuid4().hex)
        if binning_id in seen:
            binning_id = uuid4().hex
        seen.add(binning_id)
        name = str(item.get("name") or f"Binning {len(result) + 1}")
        config = data_group_composite_config(group, config_override=item["config"])
        config["_binning_id"] = binning_id
        sanitized_item = {"id": binning_id, "name": name, "config": config}
        sanitized.append(sanitized_item)
        result.append({**sanitized_item, "fit": False})
    registry["items"] = sanitized
    return result


def data_group_composite_config_by_id(
    group: DataGroup | _CompositeScope, binning_id: str
) -> dict[str, Any]:
    for item in data_group_composite_binnings(group):
        if item["id"] == str(binning_id):
            return item["config"]
    raise KeyError(f"unknown composite binning ID {binning_id!r}")


def add_data_group_composite_binning(
    group: DataGroup | _CompositeScope,
    *,
    name: str | None = None,
    duplicate_from: str | None = None,
) -> str:
    binnings = data_group_composite_binnings(group)
    source = (
        data_group_composite_config_by_id(group, duplicate_from)
        if duplicate_from is not None
        else binnings[0]["config"]
    )
    existing = {str(item["name"]).casefold() for item in binnings}
    base = str(name or "New binning").strip() or "New binning"
    candidate = base
    suffix = 2
    while candidate.casefold() in existing:
        candidate = f"{base} {suffix}"
        suffix += 1
    binning_id = uuid4().hex
    config = copy.deepcopy(source)
    config["_binning_id"] = binning_id
    config["stale"] = True
    _composite_rebin_registry(group)["items"].append(
        {"id": binning_id, "name": candidate, "config": config}
    )
    return binning_id


def rename_data_group_composite_binning(
    group: DataGroup | _CompositeScope, binning_id: str, name: str
) -> None:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("binning name cannot be empty")
    binnings = data_group_composite_binnings(group)
    if any(
        item["id"] != str(binning_id)
        and str(item["name"]).casefold() == cleaned.casefold()
        for item in binnings
    ):
        raise ValueError(f"binning name {cleaned!r} is already in use")
    registry = _composite_rebin_registry(group)
    if binnings[0]["id"] == str(binning_id):
        registry["fit_name"] = cleaned
        return
    for item in registry["items"]:
        if item["id"] == str(binning_id):
            item["name"] = cleaned
            return
    raise KeyError(f"unknown composite binning ID {binning_id!r}")


def remove_data_group_composite_binning(
    group: DataGroup | _CompositeScope, binning_id: str
) -> None:
    binnings = data_group_composite_binnings(group)
    if binnings[0]["id"] == str(binning_id):
        raise ValueError("the fit binning cannot be removed")
    registry = _composite_rebin_registry(group)
    before = len(registry["items"])
    registry["items"] = [item for item in registry["items"] if item["id"] != str(binning_id)]
    if len(registry["items"]) == before:
        raise KeyError(f"unknown composite binning ID {binning_id!r}")


def make_data_group_fit_binning(
    group: DataGroup | _CompositeScope, binning_id: str
) -> None:
    binnings = data_group_composite_binnings(group)
    current = binnings[0]
    chosen = next((item for item in binnings if item["id"] == str(binning_id)), None)
    if chosen is None:
        raise KeyError(f"unknown composite binning ID {binning_id!r}")
    if chosen["fit"]:
        return
    registry = _composite_rebin_registry(group)
    selected_item = next(item for item in registry["items"] if item["id"] == chosen["id"])
    old_config = copy.deepcopy(current["config"])
    old_name = str(current["name"])
    group.metadata[GROUP_COMPOSITE_KEY] = copy.deepcopy(selected_item["config"])
    registry["fit_name"] = str(selected_item["name"])
    selected_item.update(id=current["id"], name=old_name, config=old_config)


def data_group_composite_config(
    group: DataGroup | _CompositeScope,
    *,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return and normalize one group-level composite configuration."""

    config = config_override
    if config is None:
        config = group.metadata.get(GROUP_COMPOSITE_KEY)
        if not isinstance(config, dict):
            config = {}
            group.metadata[GROUP_COMPOSITE_KEY] = config
    config.setdefault("_binning_id", FIT_BINNING_ID)
    initialize_rebin_performance(config)
    config.setdefault("enabled", False)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    resolution_mode_key = _backend_value(
        "REBIN_RESOLUTION_MODE_KEY", REBIN_RESOLUTION_MODE_KEY
    )
    if config.get(resolution_mode_key) not in {"step", "bins"}:
        config[resolution_mode_key] = "step"
    if config.get("mean_weighting") == "normalization":
        config["mean_weighting"] = "uniform"
    elif config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "uniform"
    config["minimum_coverage"] = _rebin_minimum_coverage(config)
    config["minimum_samples"] = _rebin_minimum_samples(config)
    try:
        default_max_batch_mb = _backend_value(
            "DEFAULT_REBIN_MAX_BATCH_MB", DEFAULT_REBIN_MAX_BATCH_MB
        )
        config["max_batch_mb"] = max(
            int(config.get("max_batch_mb", default_max_batch_mb)), 1
        )
    except (TypeError, ValueError):
        config["max_batch_mb"] = _backend_value(
            "DEFAULT_REBIN_MAX_BATCH_MB", DEFAULT_REBIN_MAX_BATCH_MB
        )
    config["normalize"] = True
    mdevent = group.metadata.get("mdevent") if isinstance(group.metadata, dict) else None
    raw_dgs = group.metadata.get("raw_dgs") if isinstance(group.metadata, dict) else None
    is_corelli = (
        isinstance(raw_dgs, dict)
        and raw_dgs.get("format") == "corelli-correlation-nexus"
    )
    event_config = (
        mdevent if isinstance(mdevent, dict) else raw_dgs if isinstance(raw_dgs, dict) else None
    )
    coordinate_mode = str(config.get("coordinate_mode", "hkle"))
    if coordinate_mode not in {"hkle", "powder"}:
        coordinate_mode = "hkle"
    config["coordinate_mode"] = coordinate_mode
    axes = config.get("axes")
    if isinstance(event_config, dict):
        dimensions = list(event_config.get("dimensions", []))
        hkl_bounds = list(event_config.get("hkl_bounds", []))
        if coordinate_mode == "powder":
            q_bounds = event_config.get("q_modulus_bounds")
            if isinstance(q_bounds, (list, tuple)) and len(q_bounds) == 2:
                q_upper = max(float(q_bounds[1]), 0.0) or 5.0
            else:
                q_limits = []
                for index in range(3):
                    source_dim = dimensions[index] if index < len(dimensions) else {}
                    q_limits.append(
                        max(
                            abs(float(source_dim.get("lower", 0.0))),
                            abs(float(source_dim.get("upper", 0.0))),
                        )
                    )
                q_upper = float(np.linalg.norm(q_limits)) or 5.0
            energy = dimensions[3] if len(dimensions) > 3 else {}
            energy_lower = float(energy.get("lower", -50.0))
            energy_upper = float(energy.get("upper", 50.0))
            default_axes = [
                {
                    "name": "|Q|",
                    "variable": "Q",
                    "lower": 0.0,
                    "upper": q_upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": 0.0,
                    "auto_upper_value": q_upper,
                    "num_bins": 100,
                    "step_size": q_upper / 100.0,
                    "fractional": bool(is_corelli),
                },
                {
                    "name": "DeltaE",
                    "variable": "E",
                    "lower": energy_lower,
                    "upper": energy_upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": energy_lower,
                    "auto_upper_value": energy_upper,
                    "num_bins": 100,
                    "step_size": (energy_upper - energy_lower) / 100.0,
                    "fractional": False if is_corelli else True,
                },
            ]
        else:
            names = ("H", "K", "L", "DeltaE")
            default_axes = []
            for index, name in enumerate(names):
                source_dim = dimensions[index] if index < len(dimensions) else {}
                bounds = hkl_bounds[index] if index < len(hkl_bounds) else None
                lower = float(
                    bounds[0]
                    if bounds is not None
                    else source_dim.get("lower", -5.0 if index < 3 else -50.0)
                )
                upper = float(
                    bounds[1]
                    if bounds is not None
                    else source_dim.get("upper", 5.0 if index < 3 else 50.0)
                )
                default_axes.append(
                    {
                        "name": name,
                        "variable": ("H", "K", "L", "E")[index],
                        "vector": _identity_vector(index, 4),
                        "lower": lower,
                        "upper": upper,
                        "auto_lower": True,
                        "auto_upper": True,
                        "auto_lower_value": lower,
                        "auto_upper_value": upper,
                        "num_bins": 50 if index == 3 else 20,
                        "step_size": (upper - lower) / (50.0 if index == 3 else 20.0),
                        "fractional": index < 3 if is_corelli else True,
                    }
                )
    else:
        # A saved grid is already sufficient to construct and display the
        # composite settings.  Resolving a hierarchical reference here can
        # force a complete child rebin merely because the collection was
        # selected in the project tree.  Load a reference only when a new or
        # legacy configuration genuinely needs default axes.
        reference = (
            _composite_reference_data(group)
            if not isinstance(axes, list) or not axes
            else None
        )
        default_axes = (
            _default_rebin_axes(reference) if reference is not None else []
        )
    if not isinstance(axes, list) or not axes:
        # A composite can be restored before its reference data is loaded.
        # Do not create an empty placeholder that would prevent defaults from
        # being generated once a reference becomes available.
        if default_axes:
            config["axes"] = default_axes
    else:
        if len(axes) == len(default_axes):
            sanitized_axes = []
            for axis_config, default_axis in zip(axes, default_axes, strict=True):
                if not isinstance(axis_config, dict):
                    axis_config = {}
                for key in ("lower", "upper"):
                    auto_key = f"auto_{key}"
                    if auto_key not in axis_config:
                        axis_config[auto_key] = bool(
                            key in axis_config
                            and np.isclose(float(axis_config[key]), float(default_axis[key]))
                        )
                    if bool(axis_config.get(auto_key, False)):
                        axis_config.setdefault(f"{auto_key}_value", axis_config.get(key))
                if bool(default_axis.get("auto_step_size", False)) and (
                    "auto_step_size" not in axis_config
                ):
                    axis_config["auto_step_size"] = bool(
                        "step_size" in axis_config
                        and np.isclose(
                            float(axis_config["step_size"]),
                            float(default_axis["step_size"]),
                        )
                        and bool(default_axis.get("auto_step_size", False))
                    )
                if bool(default_axis.get("auto_step_size", False)) and bool(
                    axis_config.get("auto_step_size", False)
                ):
                    axis_config.setdefault("auto_step_size_value", axis_config.get("step_size"))
                for key, value in default_axis.items():
                    axis_config.setdefault(key, value)
                sanitized_axes.append(_sanitize_rebin_axis_config(axis_config))
            config["axes"] = sanitized_axes
        # Preserve a saved basis if reference data are temporarily unavailable
        # or differ during a refresh. A user edit is the only operation that
        # should replace its coordinate-axis vectors.
    if is_corelli and config.get("corelli_assignment_version") != CORELLI_ASSIGNMENT_VERSION:
        configured_axes = config.get("axes")
        if isinstance(configured_axes, list) and configured_axes:
            for index, axis in enumerate(configured_axes):
                if isinstance(axis, dict):
                    axis["fractional"] = index < len(configured_axes) - 1
            config["stale"] = True
        config["corelli_assignment_version"] = CORELLI_ASSIGNMENT_VERSION
    if "auto_rebin" not in config:
        # File-backed MDEvent composites require a complete source scan even
        # when only a few selected events contribute to the current view.
        config["auto_rebin"] = (
            False
            if isinstance(event_config, dict)
            else not _composite_rebin_is_large(group, config)
        )
    _migrate_rebin_axis_modes(config)
    config.setdefault("stale", False)
    return config


def _composite_source_points(group: DataGroup) -> int:
    mdevent = group.metadata.get("mdevent") if isinstance(group.metadata, dict) else None
    if isinstance(mdevent, dict):
        return int(mdevent.get("event_count", 0) or 0)
    raw_dgs = group.metadata.get("raw_dgs") if isinstance(group.metadata, dict) else None
    if isinstance(raw_dgs, dict):
        return int(raw_dgs.get("event_count", 0) or 0)
    return sum(_dataset_rebin_source_points(dataset) for dataset in _composite_candidates(group))


def _dataset_collection_point_count(node: DataGroup | DatasetGroup) -> int:
    mdevent = node.metadata.get("mdevent") if isinstance(node.metadata, dict) else None
    if isinstance(mdevent, dict):
        return int(mdevent.get("event_count", 0) or 0)
    raw_dgs = node.metadata.get("raw_dgs") if isinstance(node.metadata, dict) else None
    if isinstance(raw_dgs, dict):
        return int(raw_dgs.get("event_count", 0) or 0)
    return sum(_dataset_data_point_count(dataset) for dataset in node.iter_datasets())


def _composite_output_bins(config: dict[str, Any]) -> int:
    total = 1
    axes = config.get("axes", [])
    if not isinstance(axes, list):
        return 0
    for axis in axes:
        if isinstance(axis, dict):
            total *= max(int(axis.get("num_bins", 1) or 1), 1)
    return int(total)


def _composite_estimated_contributions(group: DataGroup, config: dict[str, Any]) -> int:
    axes = _composite_rebin_axes(group, config)
    multiplier = 2 ** sum(_rebin_fractional_axes(config, axes))
    return int(_composite_source_points(group) * multiplier)


def _composite_rebin_axes(group: DataGroup, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return physical and metadata axes participating in one composite rebin."""

    from .metadata_dimensions import MetadataDimension, metadata_rebin_axis_config

    axes = [dict(axis) for axis in config.get("axes", []) if isinstance(axis, dict)]
    axes.extend(
        metadata_rebin_axis_config(MetadataDimension(**recipe))
        for recipe in group.metadata.get("metadata_dimensions", [])
    )
    return axes


def _composite_rebin_is_large(group: DataGroup, config: dict[str, Any]) -> bool:
    effective = {**config, "axes": _composite_rebin_axes(group, config)}
    return (
        _composite_estimated_contributions(group, config)
        > _backend_value(
            "REBIN_AUTO_MAX_CONTRIBUTIONS", REBIN_AUTO_MAX_CONTRIBUTIONS
        )
        or _composite_output_bins(effective)
        > _backend_value("REBIN_AUTO_MAX_OUTPUT_BINS", REBIN_AUTO_MAX_OUTPUT_BINS)
    )


def _composite_rebin_status_text(group: DataGroup, config: dict[str, Any]) -> str:
    if not bool(config.get("enabled", False)):
        return "Composite rebinning is off."
    size_note = "large" if _composite_rebin_is_large(group, config) else "small"
    if bool(config.get("auto_rebin", True)):
        if bool(config.get("stale", False)):
            return f"Automatic composite rebinning is on ({size_note}); pending edits will recompute on the next refresh."
        return f"Automatic composite rebinning is on ({size_note}); edits recompute the cached composite."
    if bool(config.get("stale", False)):
        return f"Manual composite rebinning is on ({size_note}); Rebin now, fit, or open the data viewer to apply pending edits."
    return f"Manual composite rebinning is on ({size_note}); the cached composite is current."


def _composite_auto_enabled(group: DataGroup) -> bool:
    return bool(data_group_composite_config(group).get("auto_rebin", True))


def _composite_rebin_is_stale(group: DataGroup) -> bool:
    return bool(data_group_composite_config(group).get("stale", False))


def _should_defer_composite_rebin(group: DataGroup, *, force_rebin: bool) -> bool:
    return (
        data_group_composite_enabled(group)
        and not force_rebin
        and not _composite_auto_enabled(group)
        and _composite_rebin_is_stale(group)
    )


def data_group_composite_enabled(group: DataGroup) -> bool:
    config = group.metadata.get(GROUP_COMPOSITE_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _composite_dataset_name(group: DataGroup) -> str:
    if isinstance(group, _CompositeScope):
        paths = _dataset_group_paths(group.root)
        if sum(node.name == group.name for _path, node in paths) > 1:
            path = next((path for path, node in paths if node is group.node), group.name)
            return f"{path} {_backend_value('GROUP_COMPOSITE_NAME', GROUP_COMPOSITE_NAME)}"
    return f"{group.name} {_backend_value('GROUP_COMPOSITE_NAME', GROUP_COMPOSITE_NAME)}"


def _composite_candidates(
    group: DataGroup, *, include_backgrounds: bool = False
) -> list[DatasetEntry]:
    node = group.node if isinstance(group, _CompositeScope) else group
    background_ids = {
        background.source_dataset_id
        for background in getattr(group, "backgrounds", [])
        if background.enabled
    }

    def considered_datasets(
        current: DataGroup | DatasetGroup,
        *,
        selected_root: bool = False,
    ) -> Iterator[DatasetEntry]:
        if isinstance(current, DatasetGroup) and not current.enabled and not selected_root:
            return
        yield from current.datasets
        for subgroup in current.subgroups:
            yield from considered_datasets(subgroup)

    return [
        dataset
        for dataset in considered_datasets(node, selected_root=True)
        if dataset.enabled and (include_backgrounds or dataset.id not in background_ids)
    ]


def _hierarchical_composite_scopes(
    group: DataGroup | _CompositeScope,
) -> list[_CompositeScope]:
    """Return direct enabled child composites used by a parent recipe."""

    node = group.node if isinstance(group, _CompositeScope) else group
    if any(dataset.enabled for dataset in node.datasets):
        return []
    root = _composite_root(group)
    return [
        _CompositeScope(root, subgroup)
        for subgroup in node.subgroups
        if subgroup.enabled and data_group_composite_enabled(_CompositeScope(root, subgroup))
    ]


def _dataset_composite_kind(dataset: DatasetEntry) -> str:
    if dataset.kind == "mdevent":
        return "mdevent"
    if dataset.kind == "raw_dgs_nexus":
        return "raw_dgs_nexus"
    if dataset.metadata.get("importer") == "macs_nexus":
        return "point_data_4d"
    if data_type_container(dataset.data_type) == "point_list" or isinstance(
        dataset.data, PointListData
    ):
        return "point_list"
    if isinstance(dataset.data, PointData4D):
        return "point_data_4d"
    if isinstance(dataset.data, MDHistoData) or data_type_container(dataset.data_type) == "mdhisto":
        return "mdhisto"
    return type(dataset.data).__name__ if dataset.data is not None else "unknown"


def data_group_composite_status(group: DataGroup) -> tuple[bool, str]:
    child_scopes = _hierarchical_composite_scopes(group)
    if child_scopes:
        for child in child_scopes:
            ok, message = data_group_composite_status(child)
            if not ok:
                return False, f"Child composite {child.name!r} is not ready: {message}"
        return True, "Ready to combine the enabled child composites."
    datasets = _composite_candidates(
        group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))
    )
    if not datasets:
        return False, "No enabled datasets are available to combine."
    kinds = {_dataset_composite_kind(dataset) for dataset in datasets}
    if len(kinds) != 1:
        return (
            False,
            "Composite datasets require all enabled datasets to hold the same kind of data.",
        )
    if next(iter(kinds)) == "raw_dgs_nexus":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            return False, "Select the raw TOF dataset group to configure its HKLE composite."
    if next(iter(kinds)) not in {
        "mdhisto",
        "point_list",
        "point_data_4d",
        "mdevent",
        "raw_dgs_nexus",
    }:
        return False, "This dataset kind cannot be composited yet."
    return True, "Ready to combine enabled datasets into one rebinned composite."


def _composite_reference_data(group: DataGroup) -> Any | None:
    ok, _message = data_group_composite_status(group)
    if not ok:
        return None
    child_scopes = _hierarchical_composite_scopes(group)
    if child_scopes:
        try:
            return _cached_composite_dataset_data(child_scopes[0], force_rebin=True)
        except Exception:
            return None
    dataset = _composite_candidates(
        group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))
    )[0]
    try:
        return _source_data_for_group_composite(group, dataset)
    except Exception:
        return dataset.data


def _source_data_for_group_composite(
    group: DataGroup | _CompositeScope,
    dataset: DatasetEntry,
    *,
    include_source_masks: bool = True,
) -> Any:
    data = _ensure_dataset_data_loaded(dataset)
    node = group.node if isinstance(group, _CompositeScope) else group
    _adopt_imported_crystal(
        _composite_root(group),
        dataset,
        node if isinstance(node, DatasetGroup) else None,
    )
    extra_masks = effective_dataset_masks(_composite_root(group), dataset)
    if isinstance(data, PointListData):
        return prepared_point_list_data(dataset) if include_source_masks else data
    if isinstance(data, MDHistoData):
        if include_source_masks:
            return _mdhisto_with_nfit_masks(dataset, data=data, extra_masks=extra_masks)
        return _mdhisto_without_nfit_masks(data)
    if isinstance(data, PointData4D):
        if include_source_masks:
            return _point_data_with_nfit_masks(dataset, data, extra_masks=extra_masks)
        return data
    return data


def _composite_numerical_config(group, config):
    """Exclude UI/runtime controls and options unused by event reducers."""
    result = copy.deepcopy(dict(config))
    for key in ("stale", "auto_rebin", "workers", "max_batch_mb", "_binning_id"):
        result.pop(key, None)
    node = group.node if isinstance(group, _CompositeScope) else group
    is_event = "mdevent" in node.metadata
    if is_event:
        for key in ("fractional", "mean_weighting", "normalize"):
            result.pop(key, None)
    for axis in result.get("axes", []):
        for key in ("auto_lower_value", "auto_upper_value"):
            axis.pop(key, None)
        for key in ("auto_lower", "auto_upper"):
            if not axis.get(key):
                axis.pop(key, None)
        if is_event:
            axis.pop("fractional", None)
    return result


def _composite_cache_signature(
    group: DataGroup,
    _trail: frozenset[Any] = frozenset(),
    *,
    config_override: Mapping[str, Any] | None = None,
    binning_id: str | None = None,
) -> str:
    cache_key = _composite_cache_key(group, binning_id)
    if cache_key in _trail:
        raise ValueError("composite dependency cycle through a live group background")
    trail = _trail | {cache_key}
    config = (
        data_group_composite_config(group)
        if config_override is None
        else data_group_composite_config(group, config_override=dict(config_override))
    )
    child_scopes = _hierarchical_composite_scopes(group)
    dimensions = group.metadata.get("metadata_dimensions", [])
    node = group.node if isinstance(group, _CompositeScope) else group
    raw_config = node.metadata.get("raw_dgs", {}) if isinstance(node, DatasetGroup) else {}
    event_pulse_metadata = bool(dimensions) and (
        raw_config.get("format") == "corelli-correlation-nexus"
        and all(
            isinstance(recipe, Mapping)
            and recipe.get("sampling") == "event_pulse_time"
            for recipe in dimensions
        )
    )
    payload = [
        json.dumps(_composite_numerical_config(group, config), sort_keys=True, default=str),
        dimensions,
        # Raw CORELLI event metadata are evaluated directly from the NeXus
        # timestamp log during rebinning. They are not scalar DatasetEntry
        # channels, so previewing them here prevents the GUI tree from
        # rendering before a rebin has been requested. The raw source-file
        # signature below and the serialized recipe still invalidate caches.
        []
        if event_pulse_metadata
        else metadata_dimension_preview(group)
        if dimensions
        else [],
        [[child.name, _composite_cache_signature(child, trail)] for child in child_scopes],
        [
            [
                dataset.name,
                dataset_content_signature(dataset),
                dataset.data_type,
                dataset.kind,
                bool(dataset.enabled),
                float(dataset.fit_weight),
                float(dataset.scale_factor),
                _mask_signature(getattr(dataset, "masks", None)),
                _mask_signature(effective_dataset_masks(_composite_root(group), dataset)),
            ]
            for dataset in _composite_candidates(
                group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))
            )
        ],
        _mask_signature(getattr(group, "masks", None)),
        [
            [
                background.source_dataset_id,
                background.source_group_id,
                bool(background.enabled),
                float(background.scale),
                background.interpolation,
                background.projection,
                (
                    _viewer_view_signature(
                        background.source_entry,
                        effective_dataset_masks(_composite_root(group), background.source_entry),
                    )
                    if background.source_entry is not None
                    else None
                ),
                (
                    _composite_cache_signature(
                        _CompositeScope(_composite_root(group), background.source_group),
                        trail,
                    )
                    if background.source_group is not None
                    else None
                ),
            ]
            for background in getattr(group, "backgrounds", [])
        ],
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def composite_dataset_data(
    group: DataGroup | _CompositeScope,
    *,
    node: DatasetGroup | None = None,
    metadata_dimensions_override: Sequence[dict[str, Any]] | None = None,
    progress_callback: Any | None = None,
    config_override: Mapping[str, Any] | None = None,
    include_source_masks: bool = True,
    apply_spectral_channels: bool = True,
) -> MDHistoData | PointListData | PointData4D:
    """Build a composite using its saved rebin configuration and worker ceiling.

    ``node`` selects a nested collection. Saved plots can pass a coordinate
    recipe snapshot through ``metadata_dimensions_override``; an empty list
    explicitly requests no metadata dimensions. ``apply_spectral_channels=False``
    returns the underlying composite before its saved INS conversion.
    """
    from ._parallel import thread_budget

    if node is not None:
        group = _composite_scope(group, node)
    config = config_override if config_override is not None else data_group_composite_config(group)
    with thread_budget(config.get("workers")):
        composite_builder = _backend_value(
            "_composite_dataset_data", _composite_dataset_data
        )
        result = composite_builder(
            group,
            progress_callback=progress_callback,
            config_override=config_override,
            include_source_masks=include_source_masks,
            metadata_dimensions_override=metadata_dimensions_override,
        )
        if not apply_spectral_channels:
            return result
        from .composite_spectral import apply_composite_spectral_channels

        return apply_composite_spectral_channels(
            result, config, _composite_candidates(group, include_backgrounds=bool(
                group.metadata.get("metadata_dimensions", [])
                if metadata_dimensions_override is None else metadata_dimensions_override
            )),
        )


def _composite_progress_callback(
    callback: Any | None,
    datasets_total: int,
) -> Any | None:
    """Annotate composite progress with its source-dataset count."""

    if callback is None:
        return None

    datasets_completed = 0

    def report(event: dict[str, Any]) -> None:
        nonlocal datasets_completed
        enriched = dict(event)
        if str(enriched.get("stage", "")).startswith("rebin"):
            enriched["datasets_total"] = int(datasets_total)
            if enriched.get("datasets_completed") is not None:
                datasets_completed = max(
                    datasets_completed,
                    min(int(enriched["datasets_completed"]), int(datasets_total)),
                )
            enriched["datasets_completed"] = datasets_completed
            iteration = enriched.get("iteration")
            total = enriched.get("total")
            if enriched.get("stage") == "rebin" and iteration is not None and total:
                enriched["message"] = (
                    f"rebinning {datasets_total:,} datasets: "
                    f"{int(iteration):,}/{int(total):,} point contributions"
                )
        callback(enriched)

    return report


def _report_source_dataset_progress(
    progress_callback: Any | None,
    *,
    completed: int,
    total: int,
    dataset_name: str,
) -> None:
    """Report completion of source preparation before global point binning."""

    if progress_callback is None:
        return
    progress_callback(
        {
            "stage": "rebin_sources",
            "datasets_completed": int(completed),
            "datasets_total": int(total),
            "message": (
                f"prepared source dataset {completed:,}/{total:,}: {dataset_name}"
                if completed
                else f"preparing {total:,} source datasets"
            ),
        }
    )


def metadata_dimension_preview(group, dimensions=None) -> list[dict[str, Any]]:
    """Resolve metadata coordinates for each enabled source without rebinning."""
    import hashlib

    from .metadata_dimensions import MetadataDimension, metadata_dimension_coordinates

    specs = [
        MetadataDimension(**item)
        for item in (
            group.metadata.get("metadata_dimensions", []) if dimensions is None else dimensions
        )
    ]
    rows = []
    for dataset in _composite_candidates(group, include_backgrounds=True):
        _ensure_dataset_data_loaded(dataset)
        for spec in specs:
            values = metadata_dimension_coordinates(dataset, spec)

            rows.append(
                {
                    "dataset": dataset.name,
                    "dimension": spec.name,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "count": int(values.size),
                    "signature": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )
    return rows


def set_metadata_dimensions(group, dimensions) -> None:
    """Save validated discrete-coordinate recipes on a dataset collection.

    Pass MetadataDimension objects or their dictionary representations; an
    empty list removes the dimensions. Recompute with composite_dataset_data.
    """
    from .metadata_dimensions import MetadataDimension

    specs = [
        item if isinstance(item, MetadataDimension) else MetadataDimension(**item)
        for item in dimensions
    ]
    if len({spec.name.casefold() for spec in specs}) != len(specs):
        raise ValueError("metadata dimension names must be unique")
    config = group.metadata.setdefault(GROUP_COMPOSITE_KEY, {})
    if {spec.name.casefold() for spec in specs} & {
        str(axis.get("name", "")).casefold() for axis in config.get("axes", [])
    }:
        raise ValueError("metadata dimension names must differ from existing axes")
    if specs:
        kinds = {
            _dataset_composite_kind(item)
            for item in _composite_candidates(group, include_backgrounds=True)
        }
        raw_event_metadata = (
            kinds == {"raw_dgs_nexus"}
            and all(spec.sampling == "event_pulse_time" for spec in specs)
        )
        if not raw_event_metadata and (not kinds or not kinds <= {"point_data_4d", "mdhisto"}):
            raise ValueError(
                "metadata dimensions require loaded neutron points or histograms; raw event data require event-pulse-time sampling"
            )
    group.metadata["metadata_dimensions"] = [spec.to_dict() for spec in specs]
    config["stale"] = True


def _metadata_composite_data(group, config, dimensions, *, include_source_masks, progress_callback):
    from .metadata_dimensions import (
        MetadataDimension,
        metadata_rebin_axis_config,
    )

    specs = [MetadataDimension(**item) for item in dimensions]
    if _hierarchical_composite_scopes(group):
        raise ValueError(
            "configure metadata dimensions on the collection containing the source datasets"
        )
    node = group.node if isinstance(group, _CompositeScope) else group
    raw_config = node.metadata.get("raw_dgs", {}) if isinstance(node, DatasetGroup) else {}
    if raw_config.get("format") == "corelli-correlation-nexus":
        if any(spec.sampling != "event_pulse_time" for spec in specs):
            raise ValueError(
                "raw CORELLI metadata dimensions must use event-pulse-time sampling"
            )
        unified_config = copy.deepcopy(config)
        unified_config["axes"] = [
            *unified_config.get("axes", []),
            *(metadata_rebin_axis_config(spec) for spec in specs),
        ]
        lower, upper, num_bins = _composite_rebin_bounds(unified_config)
        return bin_raw_dgs_group(
            node,
            lower=lower,
            upper=upper,
            num_bins=num_bins,
            step_size=_composite_rebin_step_sizes(unified_config),
            bin_edges=_composite_rebin_bin_edges(unified_config),
            minimum_samples=_rebin_minimum_samples(unified_config),
            datasets=_composite_candidates(group),
            vectors=[
                axis.get("vector", _identity_vector(index, 4))
                for index, axis in enumerate(config.get("axes", [])[:4])
            ],
            axis_names=[
                str(axis.get("name", ("H", "K", "L", "DeltaE")[index]))
                for index, axis in enumerate(config.get("axes", [])[:4])
            ],
            max_batch_bytes=_rebin_max_batch_bytes(unified_config),
            progress_callback=progress_callback,
            symmetry_operations=_rebin_symmetry_matrices(
                config, _composite_root(group).lattice_parameters
            ),
            fractional_axes=_rebin_fractional_axes(config, config.get("axes", [])),
            metadata_dimensions=specs,
        )
    entries = []
    source_datasets = _composite_candidates(group, include_backgrounds=True)
    _report_source_dataset_progress(
        progress_callback,
        completed=0,
        total=len(source_datasets),
        dataset_name="",
    )
    for dataset_index, dataset in enumerate(source_datasets):
        data = _source_data_for_group_composite(
            group, dataset, include_source_masks=include_source_masks
        )
        _report_source_dataset_progress(
            progress_callback,
            completed=dataset_index + 1,
            total=len(source_datasets),
            dataset_name=dataset.name,
        )
        if not isinstance(data, (PointData4D, MDHistoData)):
            raise ValueError(
                "metadata dimensions require neutron points or histograms; event logs need an alignment adapter"
            )
        if isinstance(data, MDHistoData) and any(
            "metadata_dimension" in axis.metadata for axis in data.axes
        ):
            raise ValueError("add all metadata dimensions on the original source collection")
        prepared_entry = dataset.copy(data=data)
        entries.append(prepared_entry)
    reducer = (
        _composite_point_data
        if isinstance(entries[0].data, PointData4D)
        else _composite_mdhisto_data
    )
    unified_config = copy.deepcopy(config)
    unified_config["axes"] = [
        *unified_config.get("axes", []),
        *(metadata_rebin_axis_config(spec) for spec in specs),
    ]
    data = reducer(
        group,
        unified_config,
        datasets=entries,
        metadata_dimensions=specs,
        progress_callback=progress_callback,
    )
    return _apply_composite_backgrounds(
        group,
        data,
        config=unified_config,
        progress_callback=progress_callback,
    )


def _composite_dataset_data(
    group: DataGroup | _CompositeScope,
    *,
    progress_callback: Any | None = None,
    config_override: Mapping[str, Any] | None = None,
    include_source_masks: bool = True,
    metadata_dimensions_override: Sequence[dict[str, Any]] | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Build a composite from underlying sources on the requested output grid.

    ``config_override`` is used by source-linked derived datasets. It changes
    only the target reduction grid; source membership, calibration scales, and
    attached background recipes remain owned by the source collection.
    """

    ok, message = data_group_composite_status(group)
    if not ok:
        raise ValueError(message)
    config = (
        copy.deepcopy(dict(config_override))
        if config_override is not None
        else data_group_composite_config(group)
    )
    dimensions = (
        group.metadata.get("metadata_dimensions", [])
        if metadata_dimensions_override is None
        else metadata_dimensions_override
    )
    progress_callback = _composite_progress_callback(
        progress_callback,
        len(_composite_candidates(group, include_backgrounds=bool(dimensions))),
    )
    if dimensions:
        return _metadata_composite_data(
            group,
            config,
            dimensions,
            include_source_masks=include_source_masks,
            progress_callback=progress_callback,
        )
    child_scopes = _hierarchical_composite_scopes(group)
    child_entries: list[DatasetEntry] | None = None
    if child_scopes:
        child_entries = []
        for child in child_scopes:
            if include_source_masks:
                requested = _composite_numerical_config(child, config)
                native = _composite_numerical_config(child, data_group_composite_config(child))
                same_config = requested == native
                variant = hashlib.sha256(
                    json.dumps(requested, sort_keys=True, default=str).encode()
                ).hexdigest()
                child_data = _cached_composite_dataset_data(
                    child,
                    apply_spectral_channels=False,
                    progress_callback=progress_callback,
                    config_override=None if same_config else config,
                    binning_id=None if same_config else f"dependency-{variant}",
                )
            else:
                child_data = composite_dataset_data(
                    child,
                    apply_spectral_channels=False,
                    progress_callback=progress_callback,
                    config_override=config,
                    include_source_masks=False,
                )
            if not isinstance(child_data, MDHistoData):
                raise TypeError(
                    "hierarchical composites currently require gridded child composites"
                )
            child_entries.append(
                DatasetEntry(
                    child.name,
                    child_data,
                    kind="mdhisto",
                    data_type=(
                        "powder_inelastic"
                        if len(child_data.axes) == 2
                        else "single_crystal_inelastic"
                    ),
                    metadata={"composite": True, "source_group_id": child.node.id},
                )
            )
        kind = "mdhisto"
    else:
        kind = _dataset_composite_kind(_composite_candidates(group)[0])
    if kind in {"mdevent", "raw_dgs_nexus"}:
        config = copy.deepcopy(config)
        event_axes = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
        event_bounds = [(float(axis["lower"]), float(axis["upper"])) for axis in event_axes]
        event_symmetry = _rebin_symmetry_matrices(config, _composite_root(group).lattice_parameters)
        if event_symmetry is not None and len(event_axes) == 4:
            event_basis = _validate_mdhisto_rebin_basis(event_axes, 4)
            corner_grids = np.meshgrid(
                *([lower, upper] for lower, upper in event_bounds), indexing="ij"
            )
            output_corners = np.stack(corner_grids, axis=-1).reshape(-1, 4)
            event_bounds = _symmetry_projected_coordinate_bounds(
                output_corners @ event_basis,
                event_symmetry,
                event_basis,
            )
        config["axes"] = _resolve_auto_rebin_axes(
            event_axes,
            event_bounds,
        )
    result: MDHistoData | PointListData | PointData4D
    if kind == "mdevent":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            raise ValueError("MDEvent composites must be imported inside a dataset group")
        lower, upper, num_bins = _composite_rebin_bounds(config)
        allow_overcommit = bool(config.pop("_allow_memory_overcommit_once", False))
        if config.get("coordinate_mode") == "powder":
            result = bin_mdevent_powder_group(
                node,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=_composite_rebin_step_sizes(config),
                bin_edges=_composite_rebin_bin_edges(config),
                minimum_samples=_rebin_minimum_samples(config),
                datasets=_composite_candidates(group),
                max_batch_bytes=_rebin_max_batch_bytes(config),
                progress_callback=progress_callback,
            )
        else:
            result = bin_mdevent_group(
                node,
                lower=lower,
                upper=upper,
                num_bins=num_bins,
                step_size=_composite_rebin_step_sizes(config),
                bin_edges=_composite_rebin_bin_edges(config),
                minimum_samples=_rebin_minimum_samples(config),
                datasets=_composite_candidates(group),
                vectors=[
                    axis.get("vector", _identity_vector(index, 4))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                axis_names=[
                    str(axis.get("name", ("H", "K", "L", "DeltaE")[index]))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                max_batch_bytes=_rebin_max_batch_bytes(config),
                enforce_memory_limit=not allow_overcommit,
                progress_callback=progress_callback,
                symmetry_operations=_rebin_symmetry_matrices(
                    config, _composite_root(group).lattice_parameters
                ),
            )
    elif kind == "raw_dgs_nexus":
        node = group.node if isinstance(group, _CompositeScope) else group
        if not isinstance(node, DatasetGroup):
            raise ValueError(
                "raw direct-geometry composites must be imported inside a dataset group"
            )
        lower, upper, num_bins = _composite_rebin_bounds(config)
        common = {
            "lower": lower,
            "upper": upper,
            "num_bins": num_bins,
            "step_size": _composite_rebin_step_sizes(config),
            "bin_edges": _composite_rebin_bin_edges(config),
            "minimum_samples": _rebin_minimum_samples(config),
            "datasets": _composite_candidates(group),
            "max_batch_bytes": _rebin_max_batch_bytes(config),
            "progress_callback": progress_callback,
        }
        raw_config = node.metadata.get("raw_dgs", {})
        if raw_config.get("format") == "corelli-correlation-nexus":
            common["fractional_axes"] = _rebin_fractional_axes(
                config,
                config.get("axes", []),
            )
        if config.get("coordinate_mode") == "powder":
            result = bin_raw_dgs_powder_group(node, **common)
        else:
            result = bin_raw_dgs_group(
                node,
                vectors=[
                    axis.get("vector", _identity_vector(index, 4))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                axis_names=[
                    str(axis.get("name", ("H", "K", "L", "DeltaE")[index]))
                    for index, axis in enumerate(config.get("axes", []))
                ],
                symmetry_operations=_rebin_symmetry_matrices(
                    config, _composite_root(group).lattice_parameters
                ),
                **common,
            )
    elif kind == "mdhisto":
        result = _composite_mdhisto_data(
            group,
            config,
            datasets=child_entries,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    elif kind == "point_list":
        result = _composite_point_list_data(
            group,
            config,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    elif kind == "point_data_4d":
        result = _composite_point_data(
            group,
            config,
            progress_callback=progress_callback,
            include_source_masks=include_source_masks,
        )
    else:
        raise ValueError(f"unsupported composite dataset kind {kind!r}")
    if isinstance(result, MDHistoData):
        result = _apply_mdhisto_coverage_threshold(result, config)
    return _apply_composite_backgrounds(
        group,
        result,
        config=config,
        progress_callback=progress_callback,
    )


def _apply_mdhisto_coverage_threshold(
    data: MDHistoData,
    config: dict[str, Any],
) -> MDHistoData:
    coverage = mdhisto_coverage_fraction(data)
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    channels = dict(data.auxiliary_channels)
    channels["coverage_fraction"] = MDHistoChannel(
        coverage,
        label="Coverage",
        unit="fraction",
    )
    metadata = dict(data.metadata)
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    rebin_metadata = dict(metadata.get("rebin", {}))
    rebin_metadata["minimum_coverage"] = _rebin_minimum_coverage(config)
    metadata["rebin"] = rebin_metadata
    return data.with_updates(
        mask=np.asarray(data.mask, dtype=bool) | coverage_mask,
        metadata=metadata,
        auxiliary_channels=channels,
    )


def _broadcast_background_over_metadata(
    background: MDHistoData,
    target: MDHistoData,
) -> MDHistoData:
    """Broadcast a physical background across target metadata coordinates."""

    extra = len(target.axes) - len(background.axes)
    if extra <= 0:
        return background
    if not all(
        "metadata_dimension" in axis.metadata for axis in target.axes[-extra:]
    ):
        return background
    shape = background.shape + (1,) * extra
    target_shape = target.shape
    return background.with_updates(
        axes=target.axes,
        signal=np.broadcast_to(background.signal.reshape(shape), target_shape),
        errors=np.broadcast_to(background.errors.reshape(shape), target_shape),
        mask=np.broadcast_to(background.mask.reshape(shape), target_shape),
        num_events=np.broadcast_to(background.num_events.reshape(shape), target_shape),
        auxiliary_channels={},
    )


def _metadata_reference_slice(
    data: MDHistoData,
    source: DatasetEntry,
) -> tuple[Any, ...] | None:
    """Locate a scalar metadata source in an augmented output grid."""

    from .metadata_dimensions import MetadataDimension, assigned_metadata_coordinates

    indices: list[int] = []
    first_metadata = len(data.axes)
    for index, axis in enumerate(data.axes):
        recipe = axis.metadata.get("metadata_dimension")
        if recipe is None:
            continue
        first_metadata = min(first_metadata, index)
        values = assigned_metadata_coordinates(source, MetadataDimension(**recipe))
        if values.size == 0 or not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            return None
        value = float(values[0])
        edges = np.asarray(axis.values, dtype=float)
        bin_index = int(np.searchsorted(edges, value, side="right") - 1)
        if value == edges[-1]:
            bin_index = len(edges) - 2
        if bin_index < 0 or bin_index >= len(edges) - 1:
            return None
        indices.append(bin_index)
    if not indices:
        return None
    return (slice(None),) * first_metadata + tuple(indices)


def _apply_composite_backgrounds(
    group: DataGroup | _CompositeScope,
    data: MDHistoData | PointListData | PointData4D,
    *,
    config: Mapping[str, Any] | None = None,
    reference_entry: DatasetEntry | None = None,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Apply backgrounds owned by a composite scope after it is combined."""

    backgrounds = list(getattr(group, "backgrounds", []))
    if not backgrounds:
        return data
    if not isinstance(data, MDHistoData):
        raise TypeError("group backgrounds require a gridded composite")
    root = _composite_root(group)
    metadata = dict(data.metadata)
    if root.lattice_parameters and not isinstance(metadata.get("lattice_parameters"), dict):
        metadata["lattice_parameters"] = dict(root.lattice_parameters)
    result = replace(data, metadata=metadata)
    for background in backgrounds:
        if not background.enabled:
            continue
        source = background.source_entry
        source_group = background.source_group
        if source is None and source_group is None:
            raise ValueError(f"background {background.name!r} refers to a missing dataset or group")
        if source_group is not None:
            source_data = _cached_composite_dataset_data(
                _CompositeScope(root, source_group),
                apply_spectral_channels=False,
                force_rebin=True,
                progress_callback=progress_callback,
            )
        else:
            # A raw point-data background belongs on the composite's resolved
            # output grid.  Do not honor the background dataset's private
            # viewer-rebin recipe here: that recipe may describe an older or
            # otherwise unrelated grid and aligned subtraction would then fail.
            source_data = _source_data_for_group_composite(
                group,
                source,
            )
            if isinstance(source_data, PointData4D):
                # The neutron reference follows the resolved sample grid. Work
                # on copies so its own viewing recipe stays intact.
                aligned_config = copy.deepcopy(
                    dict(data_group_composite_config(group) if config is None else config)
                )
                if len(data.axes) < 4 or len(aligned_config.get("axes", [])) < 4:
                    raise ValueError("point-data group backgrounds require four HKLE axes")
                aligned_config["axes"] = aligned_config["axes"][:4]
                for settings, axis in zip(
                    aligned_config["axes"], data.axes[:4], strict=True
                ):
                    settings.update(
                        name=axis.name,
                        units=axis.units,
                        bin_edges=axis.values.tolist(),
                        mode="edges",
                        auto_lower=False,
                        auto_upper=False,
                        auto_step_size=False,
                    )
                source_metadata = dict(source_data.metadata)
                if root.lattice_parameters:
                    source_metadata.setdefault("lattice_parameters", dict(root.lattice_parameters))
                source_data = _rebin_point_data(
                    source_data.with_updates(metadata=source_metadata),
                    aligned_config,
                    progress_callback=progress_callback,
                )
                if source.backgrounds:
                    source_data = _apply_dataset_backgrounds(source, source_data)
        if not isinstance(source_data, MDHistoData):
            raise TypeError(f"background {background.name!r} must refer to gridded histogram data")
        if background.projection not in {"center", "sample_trajectories"}:
            raise ValueError(
                f"background {background.name!r} has unknown projection mode "
                f"{background.projection!r}"
            )
        if background.projection == "sample_trajectories":
            node = group.node if isinstance(group, _CompositeScope) else group
            if not isinstance(node, DatasetGroup) or "mdevent" not in node.metadata:
                raise ValueError(
                    "sample-trajectory powder projection requires a background "
                    "owned by an MDEvent dataset group"
                )
            source_data = project_powder_background_mdevent(
                node,
                source_data,
                result,
                datasets=_composite_candidates(group),
                interpolation=background.interpolation,
                progress_callback=progress_callback,
            )
        source_data = _broadcast_background_over_metadata(source_data, result)
        before_subtraction = result
        cancels_self = (
            reference_entry is not None
            and source is not None
            and reference_entry.id == source.id
            and reference_entry.scale_factor == background.scale
            and result.shape == source_data.shape
            and np.allclose(
                result.signal,
                background.scale * source_data.signal,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            and np.allclose(result.num_events, source_data.num_events, rtol=1e-12, atol=1e-12)
        )
        result = subtract_background(
            result,
            source_data,
            scale=background.scale,
            interpolation=background.interpolation,
        )
        if cancels_self:
            # Identical observations are fully correlated: Var(X - X) = 0.
            result = result.with_updates(
                signal=np.where(result.mask, np.nan, 0.0),
                errors=np.where(result.mask, np.nan, 0.0),
            )
        elif (
            source is not None
            and source.scale_factor == background.scale
            and (selection := _metadata_reference_slice(result, source)) is not None
        ):
            measured = ~np.asarray(result.mask[selection], dtype=bool)
            correlated = measured & np.isclose(
                before_subtraction.signal[selection],
                background.scale * source_data.signal[selection],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
            signal = np.array(result.signal, copy=True)
            errors = np.array(result.errors, copy=True)
            signal[selection] = np.where(correlated, 0.0, signal[selection])
            errors[selection] = np.where(correlated, 0.0, errors[selection])
            result = result.with_updates(signal=signal, errors=errors)
    return result


def _cached_composite_dataset_data(
    group: DataGroup,
    *,
    force_rebin: bool = True,
    progress_callback: Any | None = None,
    config_override: dict[str, Any] | None = None,
    binning_id: str | None = None,
    apply_spectral_channels: bool = True,
) -> MDHistoData | PointListData | PointData4D | None:
    from .composite_spectral import apply_composite_spectral_channels

    def finish(data):
        if not apply_spectral_channels:
            return data
        return apply_composite_spectral_channels(
            data, config_override if config_override is not None else data_group_composite_config(group),
            _composite_candidates(group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))),
        )

    signature = _composite_cache_signature(
        group, config_override=config_override, binning_id=binning_id
    )
    cache_key = _composite_cache_key(group, binning_id)
    cached = _COMPOSITE_DATA_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        _COMPOSITE_DATA_CACHE.move_to_end(cache_key)
        return finish(cached[1])
    config = (
        data_group_composite_config(group)
        if config_override is None
        else data_group_composite_config(group, config_override=config_override)
    )
    deferred = bool(
        config.get("stale", False)
        and not force_rebin
        and not config.get("auto_rebin", True)
    )
    if cached is not None and deferred:
        return finish(cached[1])
    if deferred:
        return None
    result = composite_dataset_data(
        group,
        progress_callback=progress_callback,
        config_override=config,
        apply_spectral_channels=False,
    )
    config["stale"] = False
    signature = _composite_cache_signature(
        group, config_override=config, binning_id=binning_id
    )
    binning_names = data_group_composite_binnings(group)
    name = next(
        (item["name"] for item in binning_names if item["id"] == binning_id),
        binning_names[0]["name"],
    )
    _COMPOSITE_DATA_CACHE.set_label(cache_key, f"{group.name} · {name}")
    _lru_store(
        _COMPOSITE_DATA_CACHE,
        cache_key,
        (signature, result),
        _COMPOSITE_DATA_CACHE_LIMIT,
        _COMPOSITE_DATA_CACHE_MAX_BYTES,
    )
    return finish(result)


def _peek_cached_composite_dataset_data(
    group: DataGroup | _CompositeScope,
    *,
    config_override: dict[str, Any] | None = None,
    binning_id: str | None = None,
) -> MDHistoData | PointListData | PointData4D | None:
    """Return a current cached composite without starting any computation."""

    cached = _COMPOSITE_DATA_CACHE.get(_composite_cache_key(group, binning_id))
    if cached is None or cached[0] != _composite_cache_signature(
        group, config_override=config_override, binning_id=binning_id
    ):
        return None
    return cached[1]


def composite_dataset_entry(
    group: DataGroup,
    *,
    force_rebin: bool = True,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    child_scopes = _hierarchical_composite_scopes(group)
    datasets = _composite_candidates(
        group, include_backgrounds=bool(group.metadata.get("metadata_dimensions"))
    )
    first = datasets[0] if datasets else None
    config = data_group_composite_config(group)
    return DatasetEntry(
        name=_composite_dataset_name(group),
        data=_cached_composite_dataset_data(
            group,
            force_rebin=force_rebin,
            progress_callback=progress_callback,
        ),
        kind=("mdhisto" if child_scopes else first.kind if first is not None else ""),
        data_type=(
            ("powder_inelastic" if len(config.get("axes", [])) == 2 else "single_crystal_inelastic")
            if child_scopes
            else first.data_type
            if first is not None
            else ""
        ),
        metadata={
            "source_group": group.name,
            "composite": True,
            "composite_scope_id": (
                group.node.id if isinstance(group, _CompositeScope) else None
            ),
        },
        parameters=(
            {
                SPECTRAL_CHANNEL_CONFIG_KEY: copy.deepcopy(
                    first.parameters[SPECTRAL_CHANNEL_CONFIG_KEY]
                )
            }
            if (SPECTRAL_CHANNEL_CONFIG_KEY not in config
                and first is not None and SPECTRAL_CHANNEL_CONFIG_KEY in first.parameters)
            else {}
        ),
        enabled=True,
        fit_weight=1.0,
        scale_factor=1.0,
    )


def materialize_composite_dataset(
    project_path: str | Path,
    group: DataGroup,
    node: DataGroup | DatasetGroup | None = None,
    *,
    name: str | None = None,
    progress_callback: Any | None = None,
    config_override: dict[str, Any] | None = None,
) -> DatasetEntry:
    """Store a current composite as a project-owned, reusable dataset."""

    scope = _composite_scope(group, node)
    data = composite_dataset_data(
        scope,
        progress_callback=progress_callback,
        config_override=config_override,
        apply_spectral_channels=False,
    )
    if not isinstance(data, MDHistoData):
        raise TypeError("materialized composites currently require gridded histogram data")
    from .composite_spectral import apply_composite_spectral_channels, composite_spectral_config

    config = config_override if config_override is not None else data_group_composite_config(scope)
    sources = _composite_candidates(scope, include_backgrounds=bool(scope.metadata.get("metadata_dimensions")))
    prepared = apply_composite_spectral_channels(data, config, sources)
    parameters = {}
    if SPECTRAL_CHANNEL_CONFIG_KEY in config:
        parameters[SPECTRAL_CHANNEL_CONFIG_KEY] = composite_spectral_config(config, sources)
        temperature = prepared.metadata.get("spectral_observable", {}).get("temperature_K")
        if temperature is not None and np.ndim(temperature) == 0:
            parameters["temperature"] = float(temperature)
    source_name = scope.name
    entry = DatasetEntry(
        name=_unique_dataset_name(name or f"{source_name} composite", group.dataset_names),
        data=data,
        parameters=parameters,
        kind="project_artifact",
        data_type=("powder_inelastic" if len(data.axes) == 2 else "single_crystal_inelastic"),
        metadata={
            "materialized_from_composite": {
                "source_group": source_name,
                "config": copy.deepcopy(
                    config_override
                    if config_override is not None
                    else data_group_composite_config(scope)
                ),
            },
            "import_status": "loaded",
        },
    )
    artifact_path = replace_dataset_artifact(project_path, entry.id, dataset_artifact_bytes(data))
    entry.metadata.update(
        {
            "source_file": artifact_path,
            "project_artifact_path": artifact_path,
            "_project_path": str(Path(project_path)),
        }
    )
    entry.replace_data(data, source_backed=True)
    destination = next(
        (subgroup for subgroup in group.subgroups if subgroup.name == "Materialized data"),
        None,
    )
    if destination is None:
        # Keep a materialized output from becoming an input to the same
        # composite on its next refresh. It remains available as a background,
        # analysis input, or for moving into a downstream collection.
        destination = DatasetGroup("Materialized data", enabled=False)
        group.subgroups.append(destination)
    destination.datasets.append(entry)
    return entry


def _scaled_error_for_weight(dataset: DatasetEntry, errors: np.ndarray) -> np.ndarray:
    return np.asarray(errors, dtype=float) * abs(float(dataset.scale_factor))


def _dataset_statistical_weight(dataset: DatasetEntry, size: int) -> np.ndarray:
    weight = float(dataset.fit_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        return np.zeros(size, dtype=float)
    return np.full(size, weight, dtype=float)


def _composite_rebin_bounds(config: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    return (
        [axis["lower"] for axis in axes_config],
        [axis["upper"] for axis in axes_config],
        [axis["num_bins"] for axis in axes_config],
    )


def _composite_mdhisto_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    datasets: list[DatasetEntry] | None = None,
    metadata_dimensions: Sequence[Any] | None = None,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> MDHistoData:
    from .metadata_dimensions import assigned_metadata_coordinates

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "loading, masking, and normalizing source datasets",
            }
        )
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    coverage_inputs: list[tuple[MDHistoData, np.ndarray]] = []
    first_data: MDHistoData | None = None
    weighting_mode = _rebin_mean_weighting(config)
    normalization_weighted = False
    source_datasets = datasets if datasets is not None else _composite_candidates(group)
    report_sources = datasets is None
    for dataset_index, dataset in enumerate(source_datasets):
        data = (
            dataset.data
            if datasets is not None
            else _source_data_for_group_composite(
                group,
                dataset,
                include_source_masks=include_source_masks,
            )
        )
        if report_sources:
            _report_source_dataset_progress(
                progress_callback,
                completed=dataset_index + 1,
                total=len(source_datasets),
                dataset_name=dataset.name,
            )
        if not isinstance(data, MDHistoData):
            continue
        if any("metadata_dimension" in axis.metadata for axis in data.axes):
            raise ValueError(
                "Rebin the original collection to preserve its discrete metadata dimensions."
            )
        if first_data is None:
            first_data = data
        source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
        coords = np.stack(source_grids, axis=-1)
        if len(data.axes) == 4:
            # Histogram axes are coordinates in their own basis, not HKLE.
            # Reconstruct physical coordinates before projecting into the
            # common output basis (including for already rebinned children).
            coords = coords @ np.vstack(_mdhisto_rebin_source_axis_vectors(data))
        if metadata_dimensions:
            metadata_grids = []
            for dimension in metadata_dimensions:
                values = assigned_metadata_coordinates(dataset, dimension)
                values = np.broadcast_to(values, (data.signal.size,)).reshape(data.shape)
                metadata_grids.append(np.asarray(values, dtype=float))
            coords = np.concatenate(
                (coords, np.stack(metadata_grids, axis=-1)),
                axis=-1,
            )
        coverage_inputs.append((data, coords))
        valid = (
            np.isfinite(data.signal) & np.isfinite(data.errors) & ~np.asarray(data.mask, dtype=bool)
        )
        normalization_channel = data.auxiliary_channels.get("normalization_denominator")
        if normalization_channel is not None:
            normalization_weighted = True
            normalization_values = np.asarray(normalization_channel.values, dtype=float)
            valid &= np.isfinite(normalization_values) & (normalization_values > 0.0)
        if data.num_events is not None:
            valid &= mdhisto_measured_bins(data)
        if not np.any(valid):
            continue
        scale = float(dataset.scale_factor)
        signal = np.asarray(data.signal[valid], dtype=float) * scale
        errors = _scaled_error_for_weight(dataset, np.asarray(data.errors[valid], dtype=float))
        if normalization_channel is not None:
            weights = normalization_values[valid] * float(dataset.fit_weight)
        else:
            weights = _dataset_statistical_weight(dataset, signal.size)
        coords_parts.append(coords[valid])
        signal_parts.append(signal)
        error_parts.append(errors)
        weight_parts.append(weights)
    if first_data is None or not signal_parts:
        raise ValueError("no valid data points remain before compositing")
    coords_all = np.concatenate(coords_parts, axis=0)
    signal_all = np.concatenate(signal_parts)
    errors_all = np.concatenate(error_parts)
    weights_all = np.concatenate(weight_parts)
    output_basis = None
    if len(axes_config) >= 4 and coords_all.shape[1] == len(axes_config):
        output_basis = np.eye(len(axes_config), dtype=float)
        output_basis[:4, :4] = _validate_mdhisto_rebin_basis(axes_config[:4], 4)
    limit_coordinates = (
        coords_all @ np.linalg.inv(output_basis) if output_basis is not None else coords_all
    )
    axes_config = _resolve_auto_rebin_axes(
        axes_config, _finite_coordinate_bounds(limit_coordinates)
    )
    axes_config = _resolve_data_driven_rebin_axes(config, axes_config, limit_coordinates)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    result = rebin_nd(
        signal_all,
        coords_all,
        data_errs=errors_all,
        data_weights=weights_all,
        axes=output_basis,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", True)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=weighting_mode,
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    if (
        result.binned_data is None
        or result.binned_data_errs is None
        or result.n_samples is None
        or result.bins_list is None
    ):
        raise RuntimeError("composite rebinning did not produce binned data")
    source_axes = list(first_data.axes)
    if metadata_dimensions:
        source_axes.extend(
            MDHistoAxis(
                dimension.name,
                [-0.5, 0.5],
                dimension.units,
                "unknown",
                metadata={
                    "metadata_dimension": dimension.to_dict(),
                    "interpolation": "none",
                },
            )
            for dimension in metadata_dimensions
        )
    axes = tuple(
        MDHistoAxis(
            name=str(axis_config.get("name") or source_axis.name),
            values=np.asarray(bins, dtype=float),
            units=str(
                axis_config.get("units")
                if axis_config.get("units") is not None
                else source_axis.units
            ),
            kind=source_axis.kind,
            frame=source_axis.frame,
            path=source_axis.path,
            metadata={
                **dict(source_axis.metadata),
                **(
                    {"discrete_centers": axis_config["resolved_centers"]}
                    if axis_config.get("resolved_centers") is not None
                    else {}
                ),
            },
        )
        for source_axis, axis_config, bins in zip(
            source_axes, axes_config, result.bins_list, strict=True
        )
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_coverage",
                "message": "calculating geometric coverage for output bins",
            }
        )
    coverage = np.maximum.reduce(
        [
            _rebin_mdhisto_coverage(
                source,
                source_coords,
                config,
                axes_config,
                result.bins_list,
                output_axes=output_basis,
            )
            for source, source_coords in coverage_inputs
        ]
    )
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata = {
        "composite": True,
        "source_group": group.name,
        "source_datasets": [dataset.name for dataset in _composite_candidates(group)],
        **(
            {
                "metadata_dimensions": [
                    dimension.to_dict() for dimension in metadata_dimensions
                ]
            }
            if metadata_dimensions
            else {}
        ),
        "signal_semantics": "density",
        "signal_semantics_source": "nfit_normalized_rebin",
        "coverage_mask_count": int(np.count_nonzero(coverage_mask)),
        "rebin": {
            "lower": lower,
            "upper": upper,
            "step_size": np.asarray(result.step_size, dtype=float).tolist(),
            "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
            "bin_edges": [np.asarray(edges, dtype=float).tolist() for edges in result.bins_list],
            "vectors": (
                output_basis.tolist()
                if output_basis is not None
                else [
                    _rebin_axis_vector(axis_config, index, len(axes_config)).tolist()
                    for index, axis_config in enumerate(axes_config)
                ]
            ),
            "fractional_axes": _rebin_fractional_axes(config, axes_config),
            "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
            "normalize": True,
            "mean_weighting": weighting_mode,
            "minimum_coverage": _rebin_minimum_coverage(config),
            "minimum_samples": _rebin_minimum_samples(config),
            "max_batch_mb": _rebin_max_batch_mb(config),
            "max_batch_bytes": _rebin_max_batch_bytes(config),
            "weighted_by_fit_weight": True,
            "weighted_by_normalization_denominator": normalization_weighted,
        },
    }
    for key in (
        "lattice_parameters",
        "oriented_lattice",
        "rlu_to_inv_angstrom_matrix",
        "ub_matrix",
        "orientation_matrix",
    ):
        if key in first_data.metadata:
            metadata[key] = copy.deepcopy(first_data.metadata[key])
    root = _composite_root(group)
    if root.lattice_parameters and "lattice_parameters" not in metadata:
        metadata["lattice_parameters"] = dict(root.lattice_parameters)
    auxiliary_channels = {
        "coverage_fraction": MDHistoChannel(
            coverage,
            label="Coverage",
            unit="fraction",
        )
    }
    if normalization_weighted and weighting_mode == "uniform" and result._normalization is not None:
        auxiliary_channels["normalization_denominator"] = MDHistoChannel(
            np.asarray(result._normalization, dtype=float),
            label="Combined detector-trajectory normalization",
            unit="arbitrary normalization units",
        )
    return MDHistoData(
        axes=axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=first_data.coordinate_system,
        visual_normalization=first_data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _composite_point_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    datasets: list[DatasetEntry] | None = None,
    metadata_dimensions: Sequence[Any] | None = None,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> MDHistoData:
    from .metadata_dimensions import assigned_metadata_coordinates

    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    first_data: PointData4D | None = None
    normalization_weighted = False
    source_datasets = datasets if datasets is not None else _composite_candidates(group)
    report_sources = datasets is None
    if report_sources:
        _report_source_dataset_progress(
            progress_callback,
            completed=0,
            total=len(source_datasets),
            dataset_name="",
        )
    for dataset_index, dataset in enumerate(source_datasets):
        data = (
            dataset.data
            if datasets is not None
            else _source_data_for_group_composite(
                group,
                dataset,
                include_source_masks=include_source_masks,
            )
        )
        if report_sources:
            _report_source_dataset_progress(
                progress_callback,
                completed=dataset_index + 1,
                total=len(source_datasets),
                dataset_name=dataset.name,
            )
        if not isinstance(data, PointData4D):
            continue
        if first_data is None:
            first_data = data
        source = data.valid(require_positive_sigma=False)
        if source.size == 0:
            continue
        coordinates = np.column_stack(source.coordinates())
        if metadata_dimensions:
            valid = data.valid_mask(require_positive_sigma=False)
            metadata_columns = []
            for dimension in metadata_dimensions:
                values = assigned_metadata_coordinates(dataset, dimension)
                values = np.broadcast_to(values, (data.size,))
                metadata_columns.append(np.asarray(values, dtype=float)[valid])
            coordinates = np.column_stack((coordinates, *metadata_columns))
        coords_parts.append(coordinates)
        scale = float(dataset.scale_factor)
        signal = np.asarray(source.intensity, dtype=float) * scale
        signal_parts.append(signal)
        error_parts.append(_scaled_error_for_weight(dataset, source.sigma))
        weights = _dataset_statistical_weight(dataset, source.size)
        if source.normalization_denominator is not None:
            normalization_weighted = True
            weights *= np.asarray(source.normalization_denominator, dtype=float)
        weight_parts.append(weights)
    if first_data is None or not signal_parts:
        raise ValueError("no valid data points remain before compositing")
    return _point_data_histogram(
        np.concatenate(coords_parts, axis=0),
        np.concatenate(signal_parts),
        np.concatenate(error_parts),
        config,
        data_weights=np.concatenate(weight_parts),
        source_metadata=first_data.metadata,
        coordinate_system=first_data.metadata.get("coordinate_system"),
        visual_normalization=first_data.metadata.get("visual_normalization"),
        metadata_updates={
            "composite": True,
            "source_group": group.name,
            "source_datasets": [dataset.name for dataset in _composite_candidates(group)],
            "weighted_by_fit_weight": True,
            "weighted_by_normalization_denominator": normalization_weighted,
            **(
                {
                    "metadata_dimensions": [
                        dimension.to_dict() for dimension in metadata_dimensions
                    ]
                }
                if metadata_dimensions
                else {}
            ),
        },
        progress_callback=progress_callback,
    )


def _composite_point_list_data(
    group: DataGroup,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
    include_source_masks: bool = True,
) -> PointListData:
    datasets = _composite_candidates(group)
    _report_source_dataset_progress(
        progress_callback,
        completed=0,
        total=len(datasets),
        dataset_name="",
    )
    prepared = []
    for dataset_index, dataset in enumerate(datasets):
        prepared.append(
            _source_data_for_group_composite(
                group,
                dataset,
                include_source_masks=include_source_masks,
            )
        )
        _report_source_dataset_progress(
            progress_callback,
            completed=dataset_index + 1,
            total=len(datasets),
            dataset_name=dataset.name,
        )
    point_lists = [data for data in prepared if isinstance(data, PointListData)]
    if not point_lists:
        raise ValueError("no point-list datasets are available to composite")
    coordinate_names = list(point_lists[0].coordinate_names)
    channel_label = point_lists[0].channel_labels[0] if point_lists[0].channel_labels else None
    if not coordinate_names or channel_label is None:
        raise ValueError("point-list composites require coordinates and at least one channel")
    for data in point_lists[1:]:
        if (
            list(data.coordinate_names) != coordinate_names
            or channel_label not in data.channel_labels
        ):
            raise ValueError(
                "point-list composites require matching coordinates and channel labels"
            )
    lower, upper, num_bins = _composite_rebin_bounds(config)
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coords_parts: list[np.ndarray] = []
    signal_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for dataset, data in zip(datasets, point_lists, strict=False):
        coords = np.column_stack([data.column(name) for name in coordinate_names])
        finite = np.all(np.isfinite(coords), axis=1)
        values = np.asarray(data.channel_values(channel_label), dtype=float)
        errors = data.channel_errors(channel_label)
        if errors is None:
            errors = np.ones(values.shape, dtype=float)
        valid = finite & np.isfinite(values) & np.isfinite(errors)
        if not np.any(valid):
            continue
        scale = float(dataset.scale_factor)
        coords_parts.append(coords[valid])
        signal_parts.append(values[valid] * scale)
        error_parts.append(
            _scaled_error_for_weight(dataset, np.asarray(errors[valid], dtype=float))
        )
        weight_parts.append(_dataset_statistical_weight(dataset, int(np.count_nonzero(valid))))
    if not signal_parts:
        raise ValueError("no valid point-list rows remain before compositing")
    coordinates_all = np.concatenate(coords_parts, axis=0)
    axes_config = _resolve_auto_rebin_axes(axes_config, _finite_coordinate_bounds(coordinates_all))
    axes_config = _resolve_data_driven_rebin_axes(config, axes_config, coordinates_all)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    result = rebin_nd(
        np.concatenate(signal_parts),
        coordinates_all,
        data_errs=np.concatenate(error_parts),
        data_weights=np.concatenate(weight_parts),
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", True)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        progress_callback=progress_callback,
    )
    if (
        result.binned_data is None
        or result.binned_data_errs is None
        or result.n_samples is None
        or result.bin_centers_list is None
    ):
        raise RuntimeError("composite rebinning did not produce binned data")
    occupied = np.isfinite(result.binned_data) & (result.n_samples > 0.0)
    center_grids = np.meshgrid(
        *(
            np.asarray(axis.get("resolved_centers", centers), dtype=float)
            for axis, centers in zip(axes_config, result.bin_centers_list, strict=True)
        ),
        indexing="ij",
    )
    value_name = point_lists[0].channel(channel_label)["value"]
    error_name = point_lists[0].channel(channel_label).get("error") or f"{value_name}_error"
    columns = {
        name: grid[occupied] for name, grid in zip(coordinate_names, center_grids, strict=True)
    }
    columns[value_name] = np.asarray(result.binned_data, dtype=float)[occupied]
    columns[error_name] = np.asarray(result.binned_data_errs, dtype=float)[occupied]
    columns["n_samples"] = np.asarray(result.n_samples, dtype=float)[occupied]
    return PointListData(
        columns=columns,
        units={
            **{name: point_lists[0].unit(name) for name in coordinate_names},
            value_name: point_lists[0].unit(value_name),
            error_name: point_lists[0].unit(error_name),
        },
        coordinate_names=coordinate_names,
        channels=[{"label": channel_label, "value": value_name, "error": error_name}],
        metadata={
            "composite": True,
            "source_group": group.name,
            "source_datasets": [dataset.name for dataset in datasets],
            "rebin": {
                "bin_edges": [
                    np.asarray(edges, dtype=float).tolist() for edges in (result.bins_list or [])
                ],
                "fractional_axes": _rebin_fractional_axes(config, axes_config),
                "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
                "minimum_samples": _rebin_minimum_samples(config),
            },
        },
        quantity_types={
            name: point_lists[0].quantity_type(name)
            for name in columns
            if name in point_lists[0].columns
        },
    )
