"""Shared output-grid planning for live derived dataset reductions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData
from .project_rebinning import (
    _axis_bounds,
    _finite_coordinate_bounds,
    _mdhisto_rebin_basis_bounds,
    _point_data_rebin_basis_bounds,
    _rebin_axis_bound_is_auto,
    _rebin_axis_mode,
    _resolve_auto_rebin_axes,
    _sanitize_rebin_axis_config,
)


def union_coordinate_bounds(
    source_bounds: Sequence[Sequence[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Return finite per-axis bounds covering every derived source."""

    if not source_bounds:
        raise ValueError("derived output bounds require at least one source")
    dimensions = len(source_bounds[0])
    if dimensions == 0 or any(len(bounds) != dimensions for bounds in source_bounds):
        raise ValueError("derived sources must use the same coordinate dimensions")
    result: list[tuple[float, float]] = []
    for index in range(dimensions):
        values = np.asarray(
            [bound for bounds in source_bounds for bound in bounds[index]], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("derived source bounds must be finite")
        result.append((float(np.min(values)), float(np.max(values))))
    return result


def shared_derived_rebin_config(
    config: Mapping[str, Any],
    source_bounds: Sequence[Sequence[tuple[float, float]]],
) -> dict[str, Any]:
    """Resolve automatic output limits once across all derived sources.

    The returned execution-only configuration freezes the resolved automatic
    limits.  The saved recipe remains automatic and can therefore follow later
    source changes on its next evaluation.
    """

    resolved = copy.deepcopy(dict(config))
    axes = resolved.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("derived output rebinning requires configured axes")
    automatic = any(
        _rebin_axis_bound_is_auto(axis, key)
        for axis in axes
        if isinstance(axis, Mapping)
        for key in ("lower", "upper")
    )
    if not automatic:
        return resolved
    resolved_axes = _resolve_auto_rebin_axes(axes, union_coordinate_bounds(source_bounds))
    for original, axis in zip(axes, resolved_axes, strict=True):
        for key in ("lower", "upper"):
            if _rebin_axis_bound_is_auto(original, key):
                axis[f"auto_{key}"] = False
    resolved["axes"] = resolved_axes
    return resolved


def _loaded_data_bounds(data: Any, axes: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if isinstance(data, MDHistoData):
        try:
            return _mdhisto_rebin_basis_bounds(data, axes)
        except ValueError as exc:
            if "source axes do not define a complete coordinate basis" not in str(exc):
                raise
            if len(data.axes) != len(axes):
                raise
            return [
                _axis_bounds(axis, size)
                for axis, size in zip(data.axes, data.shape, strict=True)
            ]
    if isinstance(data, PointData4D):
        return _point_data_rebin_basis_bounds(data, axes)
    if isinstance(data, PointListData):
        available = list(data.coordinate_names)
        names = [str(axis.get("variable") or axis.get("name")) for axis in axes]
        if any(name not in available for name in names):
            names = available
        if len(names) != len(axes):
            raise ValueError("point-list coordinates must match derived output dimensions")
        return _finite_coordinate_bounds(np.column_stack([data.column(name) for name in names]))
    raise TypeError("derived output bounds require histogram or point source data")


def plan_shared_derived_grid(
    config: Mapping[str, Any],
    source_ids: Sequence[str],
    *,
    resolve_source: Callable[[str], Mapping[str, Any]],
    load_dataset: Callable[[Any], Any],
    event_bounds: Callable[[Any, Sequence[Any], list[dict[str, Any]]], Sequence[tuple[float, float]]],
    symmetry_bounds: Callable[[Sequence[tuple[float, float]], list[dict[str, Any]]], Sequence[tuple[float, float]]],
) -> dict[str, Any]:
    """Inspect all sources and return their common transient output recipe.

    ``resolve_source`` supplies either ``{"dataset": entry}`` or a composite
    descriptor containing ``node`` and its selected leaf ``datasets``.  This
    keeps project traversal and loading policy outside this numerical service.
    """

    axes = [_sanitize_rebin_axis_config(dict(axis)) for axis in config.get("axes", [])]
    if not axes:
        raise ValueError("derived output rebinning requires configured axes")
    automatic = any(
        _rebin_axis_bound_is_auto(axis, key)
        for axis in axes
        for key in ("lower", "upper")
    )
    if not automatic:
        return copy.deepcopy(dict(config))
    if any(
        _rebin_axis_mode(config, axis) in {"discrete", "tolerance"}
        and axis.get("candidate_centers") is None
        for axis in axes
    ):
        raise ValueError(
            "automatic live derived arithmetic with Discrete or Tolerance axes "
            "requires shared candidate centers; use Step, Bins, or Edges instead"
        )
    all_bounds: list[Sequence[tuple[float, float]]] = []
    for source_id in source_ids:
        descriptor = resolve_source(source_id)
        dataset = descriptor.get("dataset")
        if dataset is not None:
            all_bounds.append(symmetry_bounds(_loaded_data_bounds(load_dataset(dataset), axes), axes))
            continue
        components = list(descriptor.get("components", ())) or [descriptor]
        component_bounds = []
        for component in components:
            datasets = list(component.get("datasets", ()))
            node = component.get("node")
            if node is None or not datasets:
                raise ValueError("derived composite source has no selected datasets")
            if isinstance(getattr(node, "metadata", None), Mapping) and isinstance(
                node.metadata.get("mdevent"), Mapping
            ):
                component_bounds.append(list(event_bounds(node, datasets, axes)))
            else:
                component_bounds.extend(
                    symmetry_bounds(_loaded_data_bounds(load_dataset(item), axes), axes)
                    for item in datasets
                )
        all_bounds.append(union_coordinate_bounds(component_bounds))
    return shared_derived_rebin_config(config, all_bounds)
