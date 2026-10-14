# ruff: noqa: F401
"""Compatibility dispatch for extracted project detail-panel builders."""

from __future__ import annotations

from collections.abc import Callable
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
from .project_panel_contexts import (
    DatasetAxesPanelContext,
    DatasetPointListPanelContext,
    DatasetSignalSemanticsPanelContext,
    DatasetSpectralChannelsPanelContext,
    DatasetTypePanelContext,
    GroupCompositePanelContext,
    HeatCapacityPanelContext,
    MagnetizationAbsolutePanelContext,
    PointListScalePanelContext,
    PointListSusceptibilityPanelContext,
    PointListWavelengthPanelContext,
    RebinPerformancePanelContext,
)
from .project_readonly_panels import (
    AnalysisTablePanelContext,
    DatasetMetadataPanelContext,
    MetadataTreePanelContext,
)


def invoke_panel_builder(
    builder: Callable[..., Any],
    explorer: Any,
    *args: Any,
    helper_namespace: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Invoke a compatibility builder through its explicit explorer context.

    ``helper_namespace`` is retained for callers of the former dispatch API.
    Named live helpers now come from explorer context factories; the namespace
    is not inspected or installed into the builder or its callbacks.
    """

    if builder is _group_dataset_weights_group_box:
        if "context" not in kwargs:
            group = args[0] if args else kwargs["group"]
            kwargs["context"] = explorer._group_inventory_panel_context(group)
        return builder(explorer, *args, **kwargs)

    context_factories = {
        _dataset_metadata_group_box: "_dataset_metadata_panel_context",
        _analysis_table_group_box: "_analysis_table_panel_context",
        _metadata_tree_group_box: "_metadata_tree_panel_context",
        _dataset_point_list_group_box: "_dataset_point_list_panel_context",
        _heat_capacity_box: "_heat_capacity_panel_context",
        _magnetization_absolute_box: "_magnetization_absolute_panel_context",
        _point_list_scale_box: "_point_list_scale_panel_context",
        _point_list_susceptibility_box: "_point_list_susceptibility_panel_context",
        _point_list_wavelength_box: "_point_list_wavelength_panel_context",
        _dataset_type_group_box: "_dataset_type_panel_context",
        _dataset_axes_group_box: "_dataset_axes_panel_context",
        _add_rebin_performance_controls: "_rebin_performance_panel_context",
        _dataset_signal_semantics_group_box: "_dataset_signal_semantics_panel_context",
        _dataset_spectral_channels_group_box: "_dataset_spectral_channels_panel_context",
        _group_composite_group_box: "_group_composite_panel_context",
    }
    if builder in context_factories and "context" not in kwargs:
        kwargs["context"] = getattr(explorer, context_factories[builder])()
    return builder(explorer, *args, **kwargs)
