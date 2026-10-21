"""Explicit capabilities for editable dataset and composite Qt panels.

Contexts hold only named presentation hooks and actions needed by each builder.
Pure service defaults come from their owning modules; the explorer can supply
live compatibility wrappers without altering a builder's global namespace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .analysis.coordinates import signal_semantics
from .pipeline import DataGroup, DatasetEntry, DatasetGroup
from .project_composites import (
    _composite_rebin_status_text,
    _composite_root,
    _CompositeScope,
    _peek_cached_composite_dataset_data,
    data_group_composite_binnings,
    data_group_composite_status,
)
from .project_data import (
    _dataset_lattice_parameters,
    _dataset_rebin_status_text,
    _peek_cached_dataset_view,
    dataset_rebin_binnings,
)
from .project_imports import available_data_types
from .project_masks import effective_dataset_masks
from .project_point_lists import point_list_config
from .project_rebinning import (
    _momentum_rebin_matrix,
    _momentum_rebin_vector_text,
    _parameter_to_text,
    _rebin_axis_bound_is_auto,
    _rebin_axis_fractional,
    _rebin_axis_mode,
    _rebin_mean_weighting,
    _rebin_minimum_coverage,
    _rebin_minimum_samples,
    _sanitize_rebin_axis_config,
)
from .symmetry import symmetry_spec_from_config


@dataclass(frozen=True, kw_only=True)
class DatasetPointListPanelContext:
    """Named actions and presentation services for dataset point list group box."""

    add_point_list_channel: Callable[[DatasetEntry, DataGroup | None], None]
    heat_capacity_box: Callable[[DatasetEntry, DataGroup | None, dict[str, Any]], Any]
    magnetization_absolute_box: Callable[[DatasetEntry, DataGroup | None], Any]
    point_list_scale_box: Callable[[DatasetEntry, DataGroup | None, dict[str, Any], list[str]], Any]
    point_list_susceptibility_box: Callable[[DatasetEntry, DataGroup | None, dict[str, Any]], Any]
    point_list_wavelength_box: Callable[[DatasetEntry, DataGroup | None, dict[str, Any]], Any]
    remove_point_list_channel: Callable[[DatasetEntry, DataGroup | None, int], None]
    set_point_list_channel: Callable[[DatasetEntry, DataGroup | None, int, str, str], None]
    set_point_list_coordinates: Callable[[DatasetEntry, DataGroup | None, str], None]
    bold_label: Callable[[Any, str], Any]
    point_list_config: Callable[..., Any] = point_list_config


@dataclass(frozen=True, kw_only=True)
class HeatCapacityPanelContext:
    """Named actions and presentation services for heat capacity box."""

    set_heat_capacity_fit_channel: Callable[[DatasetEntry, DataGroup | None, str], None]
    set_heat_capacity_setting: Callable[[DatasetEntry, DataGroup | None, str, Any], None]
    set_heat_capacity_source_unit: Callable[[DatasetEntry, DataGroup | None, str], None]
    compact_point_list_form: Callable[[Any, Any], None]
    parameter_to_text: Callable[[Any], str] = _parameter_to_text
    parse_parameter_text: Callable[[str], Any]


@dataclass(frozen=True, kw_only=True)
class MagnetizationAbsolutePanelContext:
    """Named actions and presentation services for magnetization absolute box."""

    set_magnetization_absolute: Callable[[DatasetEntry, DataGroup | None, str, Any], None]
    compact_point_list_form: Callable[[Any, Any], None]
    parameter_to_text: Callable[[Any], str] = _parameter_to_text
    parse_parameter_text: Callable[[str], Any]


@dataclass(frozen=True, kw_only=True)
class PointListScalePanelContext:
    """Named actions and presentation services for point list scale box."""

    set_point_list_scale: Callable[[DatasetEntry, DataGroup | None, str, Any], None]


@dataclass(frozen=True, kw_only=True)
class PointListSusceptibilityPanelContext:
    """Named actions and presentation services for point list susceptibility box."""

    set_point_list_susceptibility: Callable[[DatasetEntry, DataGroup | None, str, Any], None]
    compact_point_list_form: Callable[[Any, Any], None]


@dataclass(frozen=True, kw_only=True)
class PointListWavelengthPanelContext:
    """Named actions and presentation services for point list wavelength box."""

    set_point_list_wavelength: Callable[[DatasetEntry, DataGroup | None, str, Any], None]


@dataclass(frozen=True, kw_only=True)
class DatasetTypePanelContext:
    """Named actions and presentation services for dataset type group box."""

    set_selected_data_type: Callable[[DatasetEntry, DataGroup | None, str], None]
    available_data_types: Callable[..., Any] = available_data_types


@dataclass(frozen=True, kw_only=True)
class DatasetAxesPanelContext:
    """Named actions and presentation services for dataset axes group box."""

    add_dataset_binning: Callable[..., Any]
    add_rebin_performance_controls: Callable[..., Any]
    copy_rebin_settings: Callable[[dict[str, Any]], None]
    dataset_rebin_symmetry_preview: Callable[[DatasetEntry, DataGroup | None], str]
    make_dataset_fit_binning: Callable[[DatasetEntry], None]
    navigate_to_binning_owner: Callable[[DataGroup, DataGroup | DatasetGroup], bool]
    paste_dataset_rebin_settings: Callable[[DatasetEntry, DataGroup | None], bool]
    remove_dataset_binning: Callable[[DatasetEntry], None]
    rename_dataset_binning: Callable[[DatasetEntry], None]
    select_dataset_binning: Callable[[DatasetEntry, str], None]
    selected_dataset_binning: Callable[[DatasetEntry], dict[str, Any]]
    set_dataset_details_preserving_scroll: Callable[..., Any]
    set_dataset_binning_enabled: Callable[[DatasetEntry, DataGroup | None, bool], None]
    set_dataset_rebin_auto: Callable[[DatasetEntry, DataGroup | None, bool], None]
    set_dataset_rebin_axis_edges: Callable[[DatasetEntry, DataGroup | None, int, str], None]
    set_dataset_rebin_axis_fractional: Callable[[DatasetEntry, DataGroup | None, int, bool], None]
    set_dataset_rebin_axis_mode: Callable[[DatasetEntry, DataGroup | None, int, str], None]
    set_dataset_rebin_axis_value: Callable[[DatasetEntry, DataGroup | None, int, str, str], None]
    set_dataset_rebin_axis_vector: Callable[[DatasetEntry, DataGroup | None, int, str], None]
    set_dataset_rebin_enabled: Callable[[DatasetEntry, DataGroup | None, bool], None]
    set_dataset_rebin_mean_weighting: Callable[[DatasetEntry, DataGroup | None, str], None]
    set_dataset_rebin_minimum_coverage: Callable[[DatasetEntry, DataGroup | None, Any], None]
    set_dataset_rebin_minimum_samples: Callable[[DatasetEntry, DataGroup | None, Any], None]
    set_dataset_rebin_momentum_matrix: Callable[[DatasetEntry, DataGroup | None, str], None]
    set_dataset_rebin_symmetry_enabled: Callable[[DatasetEntry, DataGroup | None, bool], None]
    set_dataset_rebin_symmetry_expression: Callable[[DatasetEntry, DataGroup | None, str], None]
    set_dataset_rebin_symmetry_mode: Callable[[DatasetEntry, DataGroup | None, str], None]
    materialize_rebin_for_selection: Callable[[], DatasetEntry | None]
    rebin_all_dataset_binnings_now: Callable[[DatasetEntry, DataGroup | None], bool]
    rebin_dataset_now: Callable[[DatasetEntry, DataGroup | None], bool]
    save_rebin_for_selection: Callable[[], bool]
    expanded_independent_rebin_ids: Callable[[], set[str]]
    set_expanded_independent_rebin_ids: Callable[[set[str]], None]
    project: Callable[[], Any]
    dataset_can_rebin: Callable[[DatasetEntry], bool]
    dataset_lattice_parameters: Callable[..., Any] = _dataset_lattice_parameters
    dataset_rebin_status_text: Callable[..., Any] = _dataset_rebin_status_text
    format_number: Callable[[Any], str]
    momentum_rebin_matrix: Callable[..., Any] = _momentum_rebin_matrix
    momentum_rebin_vector_text: Callable[..., Any] = _momentum_rebin_vector_text
    parameter_to_text: Callable[[Any], str] = _parameter_to_text
    peek_cached_dataset_view: Callable[..., Any] = _peek_cached_dataset_view
    rebin_axis_bound_is_auto: Callable[..., Any] = _rebin_axis_bound_is_auto
    rebin_axis_fractional: Callable[..., Any] = _rebin_axis_fractional
    rebin_axis_mode: Callable[..., Any] = _rebin_axis_mode
    rebin_mean_weighting: Callable[..., Any] = _rebin_mean_weighting
    rebin_minimum_coverage: Callable[..., Any] = _rebin_minimum_coverage
    rebin_minimum_samples: Callable[..., Any] = _rebin_minimum_samples
    sanitize_rebin_axis_config: Callable[..., Any] = _sanitize_rebin_axis_config
    saved_binning_compressed_size: Callable[..., Any]
    dataset_rebin_binnings: Callable[..., Any] = dataset_rebin_binnings
    effective_dataset_masks: Callable[..., Any] = effective_dataset_masks
    symmetry_spec_from_config: Callable[..., Any] = symmetry_spec_from_config


@dataclass(frozen=True, kw_only=True)
class RebinPerformancePanelContext:
    """Named actions and presentation services for add rebin performance controls."""

    project: Callable[[], Any]
    window: Callable[[], Any]
    composite_root: Callable[..., Any] = _composite_root


@dataclass(frozen=True, kw_only=True)
class DatasetSignalSemanticsPanelContext:
    """Named actions and presentation services for dataset signal semantics group box."""

    set_dataset_signal_semantics: Callable[[DatasetEntry, DataGroup | None, str], None]
    signal_semantics: Callable[..., Any] = signal_semantics


@dataclass(frozen=True, kw_only=True)
class DatasetSpectralChannelsPanelContext:
    """Named actions and presentation services for dataset spectral channels group box."""

    dataset_spectral_channel_config: Callable[[DatasetEntry], dict[str, Any]]
    set_dataset_spectral_channel_setting: Callable[[DatasetEntry, DataGroup | None, str, Any], None]


@dataclass(frozen=True, kw_only=True)
class GroupCompositePanelContext:
    """Named actions and presentation services for group composite group box."""

    metadata_rebin_rows: Callable[[DataGroup | _CompositeScope, Any, int], int]
    metadata_dimensions_panel: Callable[..., Any]
    composite_physics_panel: Callable[[DataGroup | _CompositeScope, dict[str, Any]], Any]
    add_composite_binning: Callable[..., Any]
    add_rebin_performance_controls: Callable[..., Any]
    copy_rebin_settings: Callable[[dict[str, Any]], None]
    make_composite_fit_binning: Callable[[DataGroup | _CompositeScope], None]
    navigate_to_binning_owner: Callable[[DataGroup, DataGroup | DatasetGroup], bool]
    paste_group_composite_settings: Callable[[DataGroup | _CompositeScope], bool]
    remove_composite_binning: Callable[[DataGroup | _CompositeScope], None]
    rename_composite_binning: Callable[[DataGroup | _CompositeScope], None]
    select_composite_binning: Callable[[DataGroup | _CompositeScope, str], None]
    selected_composite_binning: Callable[[DataGroup | _CompositeScope], dict[str, Any]]
    set_dataset_collection_details: Callable[[DataGroup, DataGroup | DatasetGroup], None]
    set_group_composite_binning_enabled: Callable[[DataGroup | _CompositeScope, bool], None]
    set_group_composite_auto: Callable[[DataGroup | _CompositeScope, bool], None]
    set_group_composite_axis_edges: Callable[[DataGroup | _CompositeScope, int, str], None]
    set_group_composite_axis_fractional: Callable[[DataGroup | _CompositeScope, int, bool], None]
    set_group_composite_axis_mode: Callable[[DataGroup | _CompositeScope, int, str], None]
    set_group_composite_axis_value: Callable[[DataGroup | _CompositeScope, int, str, str], None]
    set_group_composite_axis_vector: Callable[[DataGroup | _CompositeScope, int, str], None]
    set_group_composite_coordinate_mode: Callable[[DataGroup | _CompositeScope, str], None]
    set_group_composite_enabled: Callable[[DataGroup | _CompositeScope, bool], None]
    set_group_composite_mean_weighting: Callable[[DataGroup | _CompositeScope, str], None]
    set_group_composite_minimum_coverage: Callable[[DataGroup | _CompositeScope, Any], None]
    set_group_composite_minimum_samples: Callable[[DataGroup | _CompositeScope, Any], None]
    set_group_composite_momentum_matrix: Callable[[DataGroup | _CompositeScope, str], None]
    set_group_composite_symmetry_enabled: Callable[[DataGroup | _CompositeScope, bool], None]
    set_group_composite_symmetry_expression: Callable[[DataGroup | _CompositeScope, str], None]
    set_group_composite_symmetry_mode: Callable[[DataGroup | _CompositeScope, str], None]
    materialize_composite_for_group: Callable[[DataGroup | _CompositeScope], bool]
    rebin_all_composite_binnings_now: Callable[[DataGroup | _CompositeScope], bool]
    rebin_composite_now: Callable[[DataGroup | _CompositeScope], bool]
    expanded_inactive_composite_ids: Callable[[], set[str]]
    set_expanded_inactive_composite_ids: Callable[[set[str]], None]
    selected_composite_binning_ids: Callable[[], dict[str, str]]
    project: Callable[[], Any]
    composite_rebin_status_text: Callable[..., Any] = _composite_rebin_status_text
    composite_root: Callable[..., Any] = _composite_root
    format_number: Callable[[Any], str]
    momentum_rebin_matrix: Callable[..., Any] = _momentum_rebin_matrix
    momentum_rebin_vector_text: Callable[..., Any] = _momentum_rebin_vector_text
    parameter_to_text: Callable[[Any], str] = _parameter_to_text
    peek_cached_composite_dataset_data: Callable[..., Any] = _peek_cached_composite_dataset_data
    rebin_axis_bound_is_auto: Callable[..., Any] = _rebin_axis_bound_is_auto
    rebin_axis_fractional: Callable[..., Any] = _rebin_axis_fractional
    rebin_axis_mode: Callable[..., Any] = _rebin_axis_mode
    rebin_mean_weighting: Callable[..., Any] = _rebin_mean_weighting
    rebin_minimum_coverage: Callable[..., Any] = _rebin_minimum_coverage
    rebin_minimum_samples: Callable[..., Any] = _rebin_minimum_samples
    sanitize_rebin_axis_config: Callable[..., Any] = _sanitize_rebin_axis_config
    saved_binning_compressed_size: Callable[..., Any]
    data_group_composite_binnings: Callable[..., Any] = data_group_composite_binnings
    data_group_composite_status: Callable[..., Any] = data_group_composite_status
    symmetry_spec_from_config: Callable[..., Any] = symmetry_spec_from_config
