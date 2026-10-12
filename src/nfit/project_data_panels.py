# ruff: noqa: F401
"""Compatibility dispatch for extracted project detail-panel builders."""

from __future__ import annotations

from collections.abc import Callable
from types import FunctionType
from typing import Any

from .project_dataset_panels import (
    _add_rebin_performance_controls,
    _analysis_table_group_box,
    _dataset_axes_group_box,
    _dataset_metadata_group_box,
    _dataset_point_list_group_box,
    _dataset_signal_semantics_group_box,
    _dataset_spectral_channels_group_box,
    _dataset_type_group_box,
    _details_group_box,
    _heat_capacity_box,
    _magnetization_absolute_box,
    _metadata_tree_group_box,
    _point_list_scale_box,
    _point_list_susceptibility_box,
    _point_list_wavelength_box,
)
from .project_group_panels import (
    GroupInventoryPanelContext,
    _group_composite_group_box,
    _group_dataset_weights_group_box,
)


def invoke_panel_builder(
    builder: Callable[..., Any],
    explorer: Any,
    *args: Any,
    helper_namespace: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Invoke an extracted builder with the explorer module's live helpers.

    Migrated collection inventory uses its explicit explorer context. Legacy
    builders receive a fresh function object whose nested callbacks keep the
    original method's live global lookup without mutating shared module state.
    """

    if builder is _group_dataset_weights_group_box:
        if "context" not in kwargs:
            group = args[0] if args else kwargs["group"]
            kwargs["context"] = explorer._group_inventory_panel_context(group)
        return builder(explorer, *args, **kwargs)

    rebound = FunctionType(
        builder.__code__,
        helper_namespace,
        builder.__name__,
        builder.__defaults__,
        builder.__closure__,
    )
    rebound.__kwdefaults__ = builder.__kwdefaults__
    return rebound(explorer, *args, **kwargs)
