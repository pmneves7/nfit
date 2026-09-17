from __future__ import annotations

import ast
import copy
import json
import math
import platform
import re
import signal
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import project_data as _project_data
from . import project_data_panels as _project_data_panels
from . import project_lindhard_editor as _project_lindhard_editor
from . import project_model_editor as _project_model_editor
from . import project_tight_binding_editor as _project_tight_binding_editor
from .analysis.artifacts import (
    read_project_dataset_artifact,
    write_dataset_artifact,
)
from .analysis.coordinates import signal_semantics  # noqa: F401 - panel helper injection
from .analysis.core import (
    AnalysisEntry,
    AnalysisOutputRef,
)
from .analysis.fingerprint import dataset_entry_fingerprint, recipe_hash
from .analysis.registry import analysis_definition, default_analysis_parameters
from .application_preferences import application_settings
from .cache_utils import lru_store as _lru_store
from .dataset import PointData4D, PointListData
from .file_dialogs import (
    get_open_file_name,
    get_open_file_names,
    get_save_file_name,
    set_active_project_path,
)
from .fit_config import (
    SHARING_MODES,
    CompiledFitProblem,
    FitDatasetInput,
    compile_fit_problem,
    component_parameter_names,
    compute_component_diagnostics,
    parameter_is_derived_by_closure,  # noqa: F401 - compatibility re-export
    qualified_parameter_name,
    sharing_mode,
)
from .fit_diagnostics_gui import (
    _compact_diagnostic_labels,  # noqa: F401 - compatibility re-export
    _corner_histogram_title,  # noqa: F401 - compatibility re-export
    _covariance_matrix_from_fit_entry,
    _disable_axis_offset_text,  # noqa: F401 - compatibility re-export
    _draw_centered_matrix_heatmap,  # noqa: F401 - compatibility re-export
    _draw_corner_density_panel,  # noqa: F401 - compatibility re-export
    _draw_corner_histogram_panel,  # noqa: F401 - compatibility re-export
    _draw_corner_reference_lines,  # noqa: F401 - compatibility re-export
    _draw_matrix_heatmap,  # noqa: F401 - compatibility re-export
    _draw_trace_panel,  # noqa: F401 - compatibility re-export
    _fit_entry_diagnostic_parameter_names,  # noqa: F401 - compatibility re-export
    _fit_entry_has_diagnostic_plots,
    _fit_parameter_plot_labels,  # noqa: F401 - compatibility re-export
    _fit_parameter_summaries,  # noqa: F401 - compatibility re-export
    _FitDiagnosticsPlotWindow,
    _mathtext_label,  # noqa: F401 - compatibility re-export
)
from .fit_results import (
    _display_fit_parameters,
    _posterior_correlation_matrix,  # noqa: F401 - compatibility re-export
    _posterior_display_options,
    _sampling_result_from_dict,
)
from .fitting import (
    FitCancellationRequested,
    OptimizationConfig,
    SamplerConfig,
    SamplingCancelled,
    SamplingResult,
    _evaluate_problem,
    evaluate_parameter_expression,
    evaluate_problem_model,
    fit_problem_least_squares,
    magnetic_field_vector,
    parameter_expression_names,
    reciprocal_basis_from_lattice_parameters,
    sample_problem_parameters,
)
from .form_factors import available_ions
from .importers import (
    IMPORTERS,
    importers_for_data_type,
)
from .mdevent import (
    assess_mdevent_memory,
    inspect_mdevent_workspace,
)
from .mdhisto import (
    MDHistoData,
    load_mantid_mdhisto_nxs,
    mdhisto_measured_bins,
)
from .model_registry import (
    MODEL_CATEGORY_LABELS,
    MODEL_TYPE_REGISTRY,
    available_model_categories,
    default_model_config,
    default_model_fit_parameters,
    default_model_parameters,
    model_config_tooltip,  # noqa: F401 - compatibility re-export
    model_definition,
    model_parameter_tooltip,  # noqa: F401 - compatibility re-export
    model_plot_definitions,
    model_types_in_category,
)
from .performance import (
    assess_output_rebin_memory,
    assess_rebin_cache_memory,
    estimate_rebin_result_bytes,
)
from .pipeline import (
    BackgroundSpec,
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
    PlotEntry,
    PlotSourceRef,
)
from .plot_recipes import new_plot_entry, plot_script, render_plot
from .project_archive import (
    ArchiveContent,
    ArchiveMember,
    binning_artifact_member,
    project_artifact_compressed_size,
    project_artifact_exists,
    project_artifact_size,
    read_project_manifest,
    write_project_manifest,
)
from .project_history import (
    _dataset_group_paths,
    _timestamp_now,
    ensure_fit_history,
    refresh_current_state_fit_entries,
    restore_data_group_state,
    snapshot_data_group_state,
)
from .project_import_dialogs import (
    prompt_import_choice,
    prompt_macs_nexus_options,
    prompt_powder_ins_csv_options,
)
from .project_imports import (
    DATA_TYPE_DEFINITIONS,  # noqa: F401 - panel helper injection
    DEFAULT_DATA_TYPE,
    GROUP_COMPOSITE_KEY,
    _adopt_imported_crystal,
    _dataset_can_load,
    _dataset_can_reload,
    _dataset_group_import_stream,
    _dataset_ub_for_editor,
    available_data_types,
    data_type_container,
    data_type_label,
    default_importer_for_data_type,
    import_mdevent_dataset_group,  # noqa: F401 - compatibility re-export
    parse_dataset_numors,
    set_dataset_data_type,
    set_dataset_source,
)
from .project_imports import (
    _ensure_dataset_data_loaded as _ensure_dataset_data_loaded_impl,
)
from .project_imports import (
    _reload_dataset_copy as _reload_dataset_copy_impl,
)
from .project_imports import (
    dataset_entry_from_path as _dataset_entry_from_path_impl,
)
from .project_imports import (
    import_dataset_paths as _import_dataset_paths_impl,
)
from .project_imports import (
    reload_data_group as _reload_data_group_impl,
)
from .project_imports import (
    reload_dataset_data as _reload_dataset_data_impl,
)
from .project_io import (
    FIT_CHANNEL_NAMES,
    NfitProject,
    _analysis_from_dict,  # noqa: F401 - compatibility re-export
    _analysis_to_dict,  # noqa: F401 - compatibility re-export
    _binning_signatures_match,
    _encode_float_array,
    _fit_channel_array,
    _fit_channels_from_dict,  # noqa: F401 - compatibility re-export
    _fit_channels_to_dict,  # noqa: F401 - compatibility re-export
    _fit_entry_from_dict,  # noqa: F401 - compatibility re-export
    _fit_entry_to_dict,  # noqa: F401 - compatibility re-export
    _json_mapping,  # noqa: F401 - compatibility re-export
    _project_from_dict,
    _project_to_dict,
)
from .project_model_editor import (
    CUSTOM_FORM_FACTOR_CHOICE,
    _model_limit_texts,
    _model_parameter_limit_side,  # noqa: F401 - compatibility re-export
    _parameter_is_at_bound,
    _parameter_to_text,
    _sampling_float_matches,  # noqa: F401 - compatibility re-export
    _sharing_groups_text,  # noqa: F401 - compatibility re-export
    model_crystal_config,
    model_parameter_names,
)
from .project_models import reconcile_model_orbit_parameters
from .project_rebinning import estimated_rebin_shape
from .qt_branding import configure_application_icon
from .qt_controls import configure_numeric_spin_boxes
from .rebin_cache import SHARED_REBIN_CACHE_BUDGET
from .spectral_channels import (
    SPECTRAL_CHANNEL_CONFIG_KEY,
    default_spectral_channel_config,  # noqa: F401 - compatibility re-export
    normalized_spectral_channel_config,
)
from .symmetry import symmetry_spec_from_config
from .workflow import (
    WorkflowValidationError,
    analysis_workflow_script,
    dataset_workflow_script,
    fit_workflow_script,
)

# Preserve the established project_gui import surface while keeping the data
# preparation implementation available to non-Qt callers.
point_list_config = _project_data.point_list_config
_prepared_point_list_signature = _project_data._prepared_point_list_signature
prepared_point_list_data = _project_data.prepared_point_list_data
effective_dataset_masks = _project_data.effective_dataset_masks
_CompositeScope = _project_data._CompositeScope
_composite_scope = _project_data._composite_scope
_composite_root = _project_data._composite_root
_composite_cache_key = _project_data._composite_cache_key
_fit_data_group_composite_config = _project_data.data_group_composite_config
data_group_composite_binnings = _project_data.data_group_composite_binnings
data_group_composite_config_by_id = _project_data.data_group_composite_config_by_id
add_data_group_composite_binning = _project_data.add_data_group_composite_binning
rename_data_group_composite_binning = _project_data.rename_data_group_composite_binning
remove_data_group_composite_binning = _project_data.remove_data_group_composite_binning
make_data_group_fit_binning = _project_data.make_data_group_fit_binning
_composite_source_points = _project_data._composite_source_points
_dataset_collection_point_count = _project_data._dataset_collection_point_count
_composite_output_bins = _project_data._composite_output_bins
_composite_estimated_contributions = _project_data._composite_estimated_contributions
_composite_rebin_is_large = _project_data._composite_rebin_is_large
_peek_cached_composite_dataset_data = _project_data._peek_cached_composite_dataset_data
_composite_rebin_status_text = _project_data._composite_rebin_status_text
_composite_auto_enabled = _project_data._composite_auto_enabled
_composite_rebin_is_stale = _project_data._composite_rebin_is_stale
_should_defer_composite_rebin = _project_data._should_defer_composite_rebin
data_group_composite_enabled = _project_data.data_group_composite_enabled
_composite_dataset_name = _project_data._composite_dataset_name
_composite_candidates = _project_data._composite_candidates
_hierarchical_composite_scopes = _project_data._hierarchical_composite_scopes
_dataset_composite_kind = _project_data._dataset_composite_kind
data_group_composite_status = _project_data.data_group_composite_status
_composite_reference_data = _project_data._composite_reference_data
_source_data_for_group_composite = _project_data._source_data_for_group_composite
_composite_cache_signature = _project_data._composite_cache_signature
composite_dataset_data = _project_data.composite_dataset_data
_composite_progress_callback = _project_data._composite_progress_callback
metadata_dimension_preview = _project_data.metadata_dimension_preview
set_metadata_dimensions = _project_data.set_metadata_dimensions
_metadata_composite_data = _project_data._metadata_composite_data
_composite_dataset_data = _project_data._composite_dataset_data
_apply_mdhisto_coverage_threshold = _project_data._apply_mdhisto_coverage_threshold
_apply_composite_backgrounds = _project_data._apply_composite_backgrounds
_cached_composite_dataset_data = _project_data._cached_composite_dataset_data
_derived_analysis_for_dataset = _project_data._derived_analysis_for_dataset
_derived_source_scope = _project_data._derived_source_scope
_derived_output_config_for_source = _project_data._derived_output_config_for_source
create_derived_analysis_dataset = _project_data.create_derived_analysis_dataset
_derived_source_data = _project_data._derived_source_data
derived_analysis_dataset_data = _project_data.derived_analysis_dataset_data
composite_dataset_entry = _project_data.composite_dataset_entry
materialize_composite_dataset = _project_data.materialize_composite_dataset
_scaled_error_for_weight = _project_data._scaled_error_for_weight
_dataset_statistical_weight = _project_data._dataset_statistical_weight
_composite_rebin_bounds = _project_data._composite_rebin_bounds
_composite_mdhisto_data = _project_data._composite_mdhisto_data
_composite_point_data = _project_data._composite_point_data
_composite_point_list_data = _project_data._composite_point_list_data
_apply_spectral_channel_view = _project_data._apply_spectral_channel_view
_with_viewer_dataset_metadata = _project_data._with_viewer_dataset_metadata
_kinematic_energy_metadata = _project_data._kinematic_energy_metadata
_kinematic_kf_ki_factor = _project_data._kinematic_kf_ki_factor
_apply_kinematic_normalization_to_view = _project_data._apply_kinematic_normalization_to_view
_apply_kinematic_normalization_to_points = _project_data._apply_kinematic_normalization_to_points
_mask_signature = _project_data._mask_signature
_viewer_view_signature = _project_data._viewer_view_signature
_derived_recipe_dependency_signature = _project_data._derived_recipe_dependency_signature
_viewer_data_before_scale = _project_data._viewer_data_before_scale
_apply_dataset_backgrounds = _project_data._apply_dataset_backgrounds
_viewer_data_before_scale_uncached = _project_data._viewer_data_before_scale_uncached
_mdhisto_without_nfit_masks = _project_data._mdhisto_without_nfit_masks
_apply_dataset_scale = _project_data._apply_dataset_scale
_fit_dataset_rebin_config = _project_data.dataset_rebin_config
dataset_rebin_binnings = _project_data.dataset_rebin_binnings
dataset_rebin_config_by_id = _project_data.dataset_rebin_config_by_id
add_dataset_rebin_binning = _project_data.add_dataset_rebin_binning
rename_dataset_rebin_binning = _project_data.rename_dataset_rebin_binning
remove_dataset_rebin_binning = _project_data.remove_dataset_rebin_binning
make_dataset_fit_binning = _project_data.make_dataset_fit_binning


def dataset_rebin_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return the binning currently selected for editing in this GUI process."""

    selected = getattr(dataset, "_nfit_selected_binning_id", None)
    if selected is not None:
        try:
            return dataset_rebin_config_by_id(dataset, str(selected))
        except KeyError:
            pass
    return _fit_dataset_rebin_config(dataset)


def data_group_composite_config(
    group: DataGroup | _CompositeScope,
) -> dict[str, Any]:
    """Return the composite binning currently selected for GUI editing."""

    owner = group.node if isinstance(group, _CompositeScope) else group
    selected = getattr(owner, "_nfit_selected_binning_id", None)
    if selected is not None:
        try:
            return data_group_composite_config_by_id(group, str(selected))
        except KeyError:
            pass
    return _fit_data_group_composite_config(group)
_rebin_settings_clipboard_text = _project_data._rebin_settings_clipboard_text
_rebin_config_from_clipboard_text = _project_data._rebin_config_from_clipboard_text
dataset_mask_application_config = _project_data.dataset_mask_application_config
_dataset_mask_point_count = _project_data._dataset_mask_point_count
_dataset_mask_is_large = _project_data._dataset_mask_is_large
_dataset_mask_auto_enabled = _project_data._dataset_mask_auto_enabled
_dataset_mask_is_stale = _project_data._dataset_mask_is_stale
_should_defer_dataset_masks = _project_data._should_defer_dataset_masks
_peek_cached_dataset_view = _project_data._peek_cached_dataset_view
_dataset_mask_status_text = _project_data._dataset_mask_status_text
dataset_rebin_enabled = _project_data.dataset_rebin_enabled
_dataset_rebin_source_points = _project_data._dataset_rebin_source_points
_dataset_rebin_output_bins = _project_data._dataset_rebin_output_bins
_dataset_rebin_estimated_contributions = _project_data._dataset_rebin_estimated_contributions
_dataset_lattice_parameters = _project_data._dataset_lattice_parameters
_rebin_symmetry_operations = _project_data._rebin_symmetry_operations
_rebin_symmetry_count = _project_data._rebin_symmetry_count
_rebin_symmetry_matrices = _project_data._rebin_symmetry_matrices
_rebin_symmetry_metadata = _project_data._rebin_symmetry_metadata
_dataset_rebin_is_large = _project_data._dataset_rebin_is_large
_dataset_rebin_status_text = _project_data._dataset_rebin_status_text
_dataset_rebin_auto_enabled = _project_data._dataset_rebin_auto_enabled
_dataset_rebin_is_stale = _project_data._dataset_rebin_is_stale
_should_defer_dataset_rebin = _project_data._should_defer_dataset_rebin
rebinned_dataset_data = _project_data.rebinned_dataset_data
_rebinned_dataset_data = _project_data._rebinned_dataset_data
_with_rebinned_mask_metadata = _project_data._with_rebinned_mask_metadata
_rebin_point_list_data = _project_data._rebin_point_list_data
_rebin_mean_weighting = _project_data._rebin_mean_weighting
_rebin_axis_mode = _project_data._rebin_axis_mode
_rebin_axis_fractional = _project_data._rebin_axis_fractional
_rebin_minimum_coverage = _project_data._rebin_minimum_coverage
_rebin_minimum_samples = _project_data._rebin_minimum_samples
_rebin_max_batch_mb = _project_data._rebin_max_batch_mb
_rebin_max_batch_bytes = _project_data._rebin_max_batch_bytes
_rebin_resolution_mode = _project_data._rebin_resolution_mode
_rebin_axis_bound_is_auto = _project_data._rebin_axis_bound_is_auto
_resolve_auto_rebin_axes = _project_data._resolve_auto_rebin_axes
_finite_coordinate_bounds = _project_data._finite_coordinate_bounds
_symmetry_projected_coordinate_bounds = _project_data._symmetry_projected_coordinate_bounds
_rebin_grid_kwargs = _project_data._rebin_grid_kwargs
_composite_rebin_step_sizes = _project_data._composite_rebin_step_sizes
_composite_rebin_bin_edges = _project_data._composite_rebin_bin_edges
create_rebinned_dataset = _project_data.create_rebinned_dataset
save_dataset_file = _project_data.save_dataset_file
_save_point_list_file = _project_data._save_point_list_file
_load_nfit_dataset_file = _project_data._load_nfit_dataset_file
_load_nfit_mdhisto_archive = _project_data._load_nfit_mdhisto_archive
_load_nfit_point_list_archive = _project_data._load_nfit_point_list_archive
_nfit_archive_text = _project_data._nfit_archive_text
_nfit_archive_optional_int = _project_data._nfit_archive_optional_int
_nfit_archive_json_mapping = _project_data._nfit_archive_json_mapping
_json_safe_value = _project_data._json_safe_value
_default_rebin_axes = _project_data._default_rebin_axes
_axis_bounds = _project_data._axis_bounds
_step_size_from_bounds = _project_data._step_size_from_bounds
_num_bins_from_step_size = _project_data._num_bins_from_step_size
_normalize_rebin_bin_edges = _project_data._normalize_rebin_bin_edges
_sanitize_rebin_axis_config = _project_data._sanitize_rebin_axis_config
_rebin_axis_vector = _project_data._rebin_axis_vector
_mdhisto_rebin_axis_variable = _project_data._mdhisto_rebin_axis_variable
_rebin_axis_name = _project_data._rebin_axis_name
_validate_mdhisto_rebin_basis = _project_data._validate_mdhisto_rebin_basis
_momentum_rebin_vector_text = _project_data._momentum_rebin_vector_text
_momentum_rebin_axis_indices = _project_data._momentum_rebin_axis_indices
_momentum_rebin_matrix = _project_data._momentum_rebin_matrix
_momentum_coordinate_variables = _project_data._momentum_coordinate_variables
_rebin_config_basis_bounds = _project_data._rebin_config_basis_bounds
_point_data_rebin_basis_bounds = _project_data._point_data_rebin_basis_bounds
_update_rebin_momentum_basis = _project_data._update_rebin_momentum_basis
_update_rebin_momentum_matrix = _project_data._update_rebin_momentum_matrix
_mdhisto_rebin_basis_transform = _project_data._mdhisto_rebin_basis_transform
_mdhisto_rebin_basis_bounds = _project_data._mdhisto_rebin_basis_bounds
_update_mdhisto_rebin_basis = _project_data._update_mdhisto_rebin_basis
_mdhisto_rebin_axis_vector = _project_data._mdhisto_rebin_axis_vector
_mdhisto_rebin_source_axis_vectors = _project_data._mdhisto_rebin_source_axis_vectors
_mdhisto_rebin_component = _project_data._mdhisto_rebin_component
_mdhisto_axis_edges = _project_data._mdhisto_axis_edges
_mdhisto_cell_volumes = _project_data._mdhisto_cell_volumes
_output_bin_volumes = _project_data._output_bin_volumes
_rebin_mdhisto_coverage = _project_data._rebin_mdhisto_coverage
_rebin_mdhisto_data = _project_data._rebin_mdhisto_data
_rebin_point_data = _project_data._rebin_point_data
_point_data_histogram = _project_data._point_data_histogram
_point_data_with_nfit_masks = _project_data._point_data_with_nfit_masks
_nfit_mask_for_point_data = _project_data._nfit_mask_for_point_data
_evaluate_point_data_mask = _project_data._evaluate_point_data_mask
_point_data_coordinate_range_mask = _project_data._point_data_coordinate_range_mask
_point_data_energy_q_range_mask = _project_data._point_data_energy_q_range_mask
_point_data_box_mask = _project_data._point_data_box_mask
_point_data_ellipsoid_mask = _project_data._point_data_ellipsoid_mask
_point_data_projected_region_inputs = _project_data._point_data_projected_region_inputs
_point_data_phonon_cone_mask = _project_data._point_data_phonon_cone_mask
_point_data_coordinate_values = _project_data._point_data_coordinate_values
_point_data_q_vectors = _project_data._point_data_q_vectors
_point_data_q_modulus = _project_data._point_data_q_modulus
_mdhisto_with_nfit_masks = _project_data._mdhisto_with_nfit_masks
_nfit_mask_for_mdhisto = _project_data._nfit_mask_for_mdhisto
_evaluate_mdhisto_mask = _project_data._evaluate_mdhisto_mask
_mdhisto_box_mask = _project_data._mdhisto_box_mask
_mdhisto_ellipsoid_mask = _project_data._mdhisto_ellipsoid_mask
_mdhisto_projected_region_inputs = _project_data._mdhisto_projected_region_inputs
_mask_axis_names = _project_data._mask_axis_names
_parameter_float_sequence = _project_data._parameter_float_sequence
_resolve_projected_axis_grid = _project_data._resolve_projected_axis_grid
_project_hkl_vector = _project_data._project_hkl_vector
_project_hkle_vector = _project_data._project_hkle_vector
_mdhisto_phonon_cone_mask = _project_data._mdhisto_phonon_cone_mask
_coordinate_centers = _project_data._coordinate_centers
_parameter_float = _project_data._parameter_float
_mdhisto_coordinate_range_mask = _project_data._mdhisto_coordinate_range_mask
_mdhisto_energy_q_range_mask = _project_data._mdhisto_energy_q_range_mask
_parameter_range = _project_data._parameter_range
_exclusion_parameter_range = _project_data._exclusion_parameter_range
_values_in_range = _project_data._values_in_range
_range_is_unrestricted = _project_data._range_is_unrestricted
_range_tolerance = _project_data._range_tolerance
_mdhisto_coordinate_range_axis_grids = _project_data._mdhisto_coordinate_range_axis_grids
_coordinate_range_axis_specs = _project_data._coordinate_range_axis_specs
_mdhisto_coordinate_range_axis_specs = _project_data._mdhisto_coordinate_range_axis_specs
_identity_vector = _project_data._identity_vector
_clean_axis_weight = _project_data._clean_axis_weight
_coordinate_axis_vector = _project_data._coordinate_axis_vector
_is_vector_length = _project_data._is_vector_length
_mdhisto_coordinate_grids = _project_data._mdhisto_coordinate_grids
_mdhisto_axis_coordinate_vector = _project_data._mdhisto_axis_coordinate_vector
_axis_projection_vector = _project_data._axis_projection_vector
_projection_piece_value = _project_data._projection_piece_value
_mdhisto_q_modulus_grid = _project_data._mdhisto_q_modulus_grid
_metadata_coordinate_units_are_inv_angstrom_for_mdhisto = _project_data._metadata_coordinate_units_are_inv_angstrom_for_mdhisto
_mdhisto_q_matrix = _project_data._mdhisto_q_matrix
_matrix_includes_2pi = _project_data._matrix_includes_2pi
_dataset_data_point_count = _project_data._dataset_data_point_count
_PREPARED_POINT_LIST_CACHE = _project_data._PREPARED_POINT_LIST_CACHE
_PREPARED_POINT_LIST_CACHE_LIMIT = _project_data._PREPARED_POINT_LIST_CACHE_LIMIT
_PREPARED_POINT_LIST_CACHE_MAX_BYTES = _project_data._PREPARED_POINT_LIST_CACHE_MAX_BYTES
_VIEWER_VIEW_CACHE = _project_data._VIEWER_VIEW_CACHE
_VIEWER_VIEW_CACHE_LIMIT = _project_data._VIEWER_VIEW_CACHE_LIMIT
_VIEWER_VIEW_CACHE_MAX_BYTES = _project_data._VIEWER_VIEW_CACHE_MAX_BYTES
_COMPOSITE_DATA_CACHE = _project_data._COMPOSITE_DATA_CACHE
_COMPOSITE_DATA_CACHE_LIMIT = _project_data._COMPOSITE_DATA_CACHE_LIMIT
_COMPOSITE_DATA_CACHE_MAX_BYTES = _project_data._COMPOSITE_DATA_CACHE_MAX_BYTES

PROJECT_CACHE_BINNINGS_KEY = "cache_binnings"
PROJECT_BINNING_CACHE_ENTRIES_KEY = "binning_cache_entries"
PROJECT_BINNING_CACHE_FORMAT_VERSION = 6
PROJECT_BINNING_CACHE_COMPATIBLE_FORMATS = frozenset({4, 5, 6})


def _ensure_dataset_data_loaded(
    dataset: DatasetEntry,
    *,
    progress_callback: Any | None = None,
) -> Any:
    """Load data while preserving the project GUI's injectable loader seam."""

    return _ensure_dataset_data_loaded_impl(
        dataset,
        mdhisto_loader=load_mantid_mdhisto_nxs,
        dataset_file_loader=_load_nfit_dataset_file,
        progress_callback=progress_callback,
    )


def _reload_dataset_copy(dataset: DatasetEntry) -> DatasetEntry:
    return _reload_dataset_copy_impl(
        dataset,
        data_loader=_ensure_dataset_data_loaded,
    )


def reload_dataset_data(dataset: DatasetEntry) -> Any:
    """Reload a dataset through the GUI-visible source loader hooks."""

    return _reload_dataset_data_impl(
        dataset,
        data_loader=_ensure_dataset_data_loaded,
    )


def reload_data_group(group: DataGroup | DatasetGroup) -> list[DatasetEntry]:
    """Reload descendant datasets through the GUI-visible source loader hooks."""

    return _reload_data_group_impl(
        group,
        data_loader=_ensure_dataset_data_loaded,
    )


def dataset_for_slice_viewer(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
    rebin_config: dict[str, Any] | None = None,
    cache_id: str | None = None,
) -> MDHistoData | PointListData | PointData4D | None:
    """Prepare viewer data while preserving legacy loader injection."""

    if _peek_cached_dataset_view(
        dataset,
        extra_masks=extra_masks,
        rebin_config=rebin_config,
        cache_id=cache_id,
    ) is None:
        if progress_callback is None:
            # Preserve the long-standing one-argument injection seam used by
            # extensions and tests that replace the GUI loader.
            _ensure_dataset_data_loaded(dataset)
        else:
            _ensure_dataset_data_loaded(
                dataset,
                progress_callback=progress_callback,
            )
    return _project_data.dataset_for_slice_viewer(
        dataset,
        extra_masks=extra_masks,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
        rebin_config=rebin_config,
        cache_id=cache_id,
    )


QtMDHistoSliceViewer = None
RECENT_PROJECT_LIMIT = 10
RECENT_PROJECTS_KEY = "recent_projects"
PROJECT_WINDOW_TARGET_SIZE = (1560, 1000)
PROJECT_WINDOW_SCREEN_MARGIN = 48
TREE_DATASET_COMPACT_THRESHOLD = 12
TREE_DATASET_PAGE_SIZE = 50
DETAIL_DATASET_PAGE_SIZE = 20
DATASET_REBIN_KEY = "rebin"
FIT_BINNING_ID = _project_data.FIT_BINNING_ID
GROUP_COMPOSITE_BINNINGS_KEY = _project_data.GROUP_COMPOSITE_BINNINGS_KEY
PLOT_SOURCE_REBIN_CONFIGS_KEY = "source_rebin_configs"
PLOT_SOURCE_COMPOSITE_KEY = "source_composite"
DATASET_MASK_APPLICATION_KEY = "mask_application"
GROUP_COMPOSITE_NAME = "Composite"
DERIVED_RECIPE_KEY = "derived_recipe"
VIRTUAL_DERIVED_ANALYSIS_TYPES = {"dataset_clone", "histogram_arithmetic"}
DEFAULT_REBIN_MAX_BATCH_MB = 192
DEFAULT_MINIMUM_COVERAGE = 0.0
DEFAULT_MINIMUM_SAMPLES = 0.0
REBIN_COORDINATE_BASIS_VERSION = 2
REBIN_RESOLUTION_MODE_KEY = "resolution_mode"
REBIN_AXIS_MODES = _project_data.REBIN_AXIS_MODES
REBIN_SETTINGS_CLIPBOARD_SCHEMA = "nfit.rebin-settings"
REBIN_SETTINGS_CLIPBOARD_VERSION = 1
REBIN_SETTINGS_KEYS = (
    "enabled",
    "axes",
    "auto_rebin",
    "mean_weighting",
    "minimum_coverage",
    "minimum_samples",
    "max_batch_mb",
    "workers",
    "normalize",
    "symmetry",
    "coordinate_basis_version",
    "coordinate_mode",
    "metadata_dimensions",
)
REBIN_AUTO_MAX_CONTRIBUTIONS = 5_000_000
REBIN_AUTO_MAX_OUTPUT_BINS = 2_000_000
MASK_AUTO_MAX_POINTS = 5_000_000
DATASET_POINT_LIST_KEY = "point_list"
SUSCEPTIBILITY_CHANNEL_LABEL = "Susceptibility"
INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL = "Inverse susceptibility"
HEAT_CAPACITY_CHANNEL_LABEL = "Heat capacity"
HEAT_CAPACITY_OVER_T_CHANNEL_LABEL = "C/T"
TEMPERATURE_SQUARED_COLUMN = "Temperature squared"
Q_COORDINATE_NAME = "q"
D_SPACING_COORDINATE_NAME = "d"
COORDINATE_RANGE_AXIS_PREFIX = "axis_"
COORDINATE_RANGE_PARAMETER_NAMES = ("H", "K", "L", "E")
EFFECTIVE_FORM_FACTOR_CHOICE = "__mixture__"
POSTERIOR_DISPLAY_KEY = "posterior_display"


def _screen_aware_project_window_size(
    available_width: int,
    available_height: int,
) -> tuple[int, int]:
    """Return the preferred project-window size bounded by one screen."""

    width = min(
        PROJECT_WINDOW_TARGET_SIZE[0],
        max(1, int(available_width) - PROJECT_WINDOW_SCREEN_MARGIN),
    )
    height = min(
        PROJECT_WINDOW_TARGET_SIZE[1],
        max(1, int(available_height) - PROJECT_WINDOW_SCREEN_MARGIN),
    )
    return width, height


MASK_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "coordinate_range": {
        "label": "Coordinate range",
        "parameters": {
            "H": {
                "default": [0.0, 0.0],
                "description": "Inclusive reciprocal-lattice H interval to mask out. [0, 0] leaves H inactive.",
                "allowed": "Two numbers [min, max], with [0, 0] or an empty string meaning no H restriction.",
                "type": "list[float] | string",
                "example": "[-0.5, 0.5]",
            },
            "K": {
                "default": [0.0, 0.0],
                "description": "Inclusive reciprocal-lattice K interval to mask out. [0, 0] leaves K inactive.",
                "allowed": "Two numbers [min, max], with [0, 0] or an empty string meaning no K restriction.",
                "type": "list[float] | string",
                "example": "[-0.5, 0.5]",
            },
            "L": {
                "default": [0.0, 0.0],
                "description": "Inclusive reciprocal-lattice L interval to mask out. [0, 0] leaves L inactive.",
                "allowed": "Two numbers [min, max], with [0, 0] or an empty string meaning no L restriction.",
                "type": "list[float] | string",
                "example": "[0, 4]",
            },
            "E": {
                "default": [0.0, 0.0],
                "description": "Inclusive energy-transfer interval to mask out. [0, 0] leaves energy inactive.",
                "allowed": "Two numbers [min, max] in meV, with [0, 0] or an empty string meaning no energy restriction.",
                "type": "list[float] | string",
                "example": "[-2, 30]",
            },
        },
    },
    "energy_q_range": {
        "label": "Energy / |Q| range",
        "parameters": {
            "energy": {
                "default": [0.0, 0.0],
                "description": "Inclusive energy-transfer interval to mask out; [0, 0] leaves this dimension inactive.",
                "allowed": "Two numbers [min, max] in meV, with [0, 0] meaning no energy restriction.",
                "type": "list[float]",
                "example": "[-2, 1]",
            },
            "q_modulus": {
                "default": [0.0, 0.0],
                "description": "Inclusive |Q| interval to mask out; [0, 0] leaves this dimension inactive.",
                "allowed": "Two numbers [min, max] in inverse angstrom, with [0, 0] meaning no |Q| restriction.",
                "type": "list[float]",
                "example": "[0.2, 1.5]",
            },
        },
    },
    "box": {
        "label": "Projected box",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Center of the projected box in the coordinates named by axes.",
                "allowed": "List of numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0, 0, 0, 10]",
            },
            "width": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Full box width along each projected axis. A zero-width default masks no data.",
                "allowed": "Non-negative numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0.2, 0.2, 0.2, 1.0]",
            },
            "axes": {
                "default": ["H", "K", "L", "E"],
                "description": "Coordinate names used by the projected box.",
                "allowed": "List containing coordinate names such as H, K, L, E, or supported projected axes.",
                "type": "list[str]",
                "example": '["H", "K", "L", "E"]',
            },
        },
    },
    "ellipsoid": {
        "label": "Projected ellipsoid",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Center of the projected ellipsoid in the coordinates named by axes.",
                "allowed": "List of numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0, 0, 0, 10]",
            },
            "radii": {
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Ellipsoid radius along each projected axis. A zero-radius default masks no data.",
                "allowed": "Non-negative numbers with the same length as axes.",
                "type": "list[float]",
                "example": "[0.2, 0.2, 0.2, 1.0]",
            },
            "axes": {
                "default": ["H", "K", "L", "E"],
                "description": "Coordinate names used by the projected ellipsoid.",
                "allowed": "List containing coordinate names such as H, K, L, E, or supported projected axes.",
                "type": "list[str]",
                "example": '["H", "K", "L", "E"]',
            },
        },
    },
    "phonon_cone": {
        "label": "Phonon cone",
        "parameters": {
            "center": {
                "default": [0.0, 0.0, 0.0],
                "description": "One physical HKL reciprocal-space center or a list of Bragg centers whose acoustic phonon cones are masked out. Centers remain in physical HKL coordinates even when the displayed axes are rebinned projections.",
                "allowed": "One [H, K, L] vector or a nonempty list of [H, K, L] vectors in reciprocal lattice units.",
                "type": "list[float] | list[list[float]]",
                "example": "[[1, 1, 0], [2, 2, 0], [3, 3, 0]]",
            },
            "slope": {
                "default": 0.0,
                "description": "Cone slope dE/d|Q| defining the area to mask out. Zero disables the starter mask.",
                "allowed": "Non-negative number in meV per inverse angstrom.",
                "type": "float",
                "example": "35.0",
            },
            "radius": {
                "default": 0.0,
                "description": "Additional reciprocal-space radius included in the area to mask out. Zero adds no extra radius.",
                "allowed": "Non-negative number in inverse angstrom.",
                "type": "float",
                "example": "0.15",
            },
        },
    },
}



def _new_gui_project() -> NfitProject:
    """Return the clean initial project shown by the GUI."""

    return NfitProject(data_groups=[DataGroup(name="Workspace1")])


def _project_file_signature(path: str | Path | None) -> tuple[int, int, int, int] | None:
    """Return a cheap identity/content-change signature for a project file."""

    if path is None:
        return None
    try:
        info = Path(path).stat()
    except OSError:
        return None
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
    )


def _project_file_size(path: str | Path | None) -> int | None:
    """Return the current archive size, or ``None`` for an unsaved project."""

    if path is None:
        return None
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return None


def _format_project_file_size(num_bytes: int) -> str:
    """Format a saved project archive size for the project-window title."""

    return f"{float(num_bytes) / 1024**3:.2f} GB"


def create_data_group(project: NfitProject, name: str | None = None) -> DataGroup:
    """Add a data group to a project and return it."""

    group = DataGroup(name=next_data_group_name(project.data_groups) if name is None else name)
    if group.name in {existing.name for existing in project.data_groups}:
        raise ValueError(f"duplicate data group name {group.name!r}")
    project.data_groups.append(group)
    return group


def import_dataset_paths(
    group: DataGroup,
    paths: list[str | Path],
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
    importer_options: dict[str, dict[str, Any]] | None = None,
    into: DatasetGroup | None = None,
    stream_group_mode: str = "reuse",
    progress_callback: Any | None = None,
) -> list[DatasetEntry]:
    """Import sources through the GUI-independent project import service."""

    return _import_dataset_paths_impl(
        group,
        paths,
        data_type=data_type,
        importer_name=importer_name,
        importer_options=importer_options,
        into=into,
        stream_group_mode=stream_group_mode,
        progress_callback=progress_callback,
        dataset_file_loader=_load_nfit_dataset_file,
    )


def delete_data_group(project: NfitProject, group: DataGroup) -> None:
    """Remove a data group from a project."""

    project.data_groups.remove(group)


def delete_dataset(group: DataGroup, dataset: DatasetEntry) -> None:
    """Remove a dataset entry from anywhere in the group tree."""

    parent = _dataset_parent_node(group, dataset)
    if parent is None:
        raise ValueError(f"dataset {dataset.name!r} is not in group {group.name!r}")
    for candidate in group.iter_datasets():
        if candidate is dataset:
            continue
        candidate.backgrounds[:] = [
            background
            for background in candidate.backgrounds
            if background.source_dataset_id != dataset.id
        ]
    for node in (group, *group.iter_subgroups()):
        node.backgrounds[:] = [
            background
            for background in node.backgrounds
            if background.source_dataset_id != dataset.id
        ]
    parent.datasets.remove(dataset)


def set_mask_collection_enabled(
    owner: DatasetEntry | DataGroup | DatasetGroup,
    enabled: bool,
) -> None:
    """Enable or disable every mask directly owned by a dataset or collection."""

    for mask in owner.masks:
        mask.enabled = bool(enabled)


def set_background_collection(
    owner: DatasetEntry | DataGroup | DatasetGroup,
    *,
    enabled: bool | None = None,
    scale: float | None = None,
) -> None:
    """Bulk configure backgrounds directly owned by a dataset or collection."""

    if scale is not None and not np.isfinite(float(scale)):
        raise ValueError("background scale must be finite")
    for background in owner.backgrounds:
        if enabled is not None:
            background.enabled = bool(enabled)
        if scale is not None:
            background.scale = float(scale)




def create_mask(dataset: DatasetEntry, name: str | None = None, *, type: str = "coordinate_range") -> MaskSpec:
    """Add a mask spec to a dataset and return it."""

    if type not in MASK_TYPE_DEFINITIONS:
        raise ValueError(f"unknown mask type {type!r}")
    mask = MaskSpec(
        name=next_mask_name(dataset.masks) if name is None else name,
        type=type,
        parameters=default_mask_parameters(type),
    )
    ensure_coordinate_range_mask_axes(mask, dataset)
    if mask.name in {existing.name for existing in dataset.masks}:
        raise ValueError(f"duplicate mask name {mask.name!r}")
    dataset.masks.append(mask)
    return mask


def create_group_mask(
    subgroup: DatasetGroup,
    reference_dataset: DatasetEntry | None = None,
    name: str | None = None,
    *,
    type: str = "coordinate_range",
) -> MaskSpec:
    """Add a shared mask to a nested dataset group, applied to all descendants."""

    if type not in MASK_TYPE_DEFINITIONS:
        raise ValueError(f"unknown mask type {type!r}")
    mask = MaskSpec(
        name=next_mask_name(subgroup.masks) if name is None else name,
        type=type,
        parameters=default_mask_parameters(type),
    )
    if reference_dataset is not None:
        ensure_coordinate_range_mask_axes(mask, reference_dataset)
    if mask.name in {existing.name for existing in subgroup.masks}:
        raise ValueError(f"duplicate mask name {mask.name!r}")
    subgroup.masks.append(mask)
    return mask


def _group_reference_dataset(subgroup: DatasetGroup) -> DatasetEntry | None:
    """Return the first descendant dataset with MDHisto data, for mask axis inference."""

    for dataset in subgroup.iter_datasets():
        if isinstance(dataset.data, MDHistoData):
            return dataset
    return None


def delete_mask(dataset: DatasetEntry, mask: MaskSpec) -> None:
    """Remove a mask from a dataset."""

    dataset.masks.remove(mask)


def next_dataset_group_name(existing_names: Any) -> str:
    """Return the next available default subgroup name."""

    taken = set(existing_names)
    index = 1
    while f"Group{index}" in taken:
        index += 1
    return f"Group{index}"


def create_dataset_group(
    data_group: DataGroup,
    parent_node: Any,
    name: str | None = None,
) -> DatasetGroup:
    """Create a nested dataset group under ``parent_node`` (a DataGroup or DatasetGroup)."""

    existing = {sub.name for sub in data_group.iter_subgroups()}
    subgroup = DatasetGroup(name=name or next_dataset_group_name(existing))
    if subgroup.name in existing:
        raise ValueError(f"duplicate dataset group name {subgroup.name!r}")
    parent_node.subgroups.append(subgroup)
    return subgroup


def copy_dataset_group_to_parent(
    data_group: DataGroup,
    source: DatasetGroup,
    parent_node: DataGroup | DatasetGroup,
) -> DatasetGroup:
    """Deep-copy a dataset-group subtree with fresh group and dataset IDs."""

    used_group_names = {item.name for item in data_group.iter_subgroups()}
    used_dataset_names = set(data_group.dataset_names)
    dataset_id_map: dict[str, DatasetEntry] = {}
    group_id_map: dict[str, DatasetGroup] = {}

    def clone(node: DatasetGroup) -> DatasetGroup:
        name = _unique_name(node.name, used_group_names)
        used_group_names.add(name)
        copied = DatasetGroup(
            name=name,
            enabled=bool(node.enabled),
            masks=copy.deepcopy(node.masks),
            backgrounds=copy.deepcopy(node.backgrounds),
            resolution=copy.deepcopy(node.resolution),
            metadata=copy.deepcopy(node.metadata),
        )
        group_id_map[node.id] = copied
        for dataset in node.datasets:
            item = copy.deepcopy(dataset).copy()
            item.name = _unique_name(item.name, used_dataset_names)
            used_dataset_names.add(item.name)
            dataset_id_map[dataset.id] = item
            copied.datasets.append(item)
        copied.subgroups = [clone(child) for child in node.subgroups]
        return copied

    result = clone(source)
    existing_datasets = {item.id: item for item in data_group.iter_datasets()}
    existing_groups = {item.id: item for item in data_group.iter_subgroups()}
    for node in (result, *result.iter_subgroups()):
        owners = [*node.datasets, node]
        for owner in owners:
            for background in owner.backgrounds:
                replacement = dataset_id_map.get(
                    background.source_dataset_id
                ) or existing_datasets.get(background.source_dataset_id)
                if replacement is not None:
                    background.source_dataset_id = replacement.id
                background.source_entry = replacement
                replacement_group = group_id_map.get(
                    background.source_group_id or ""
                ) or existing_groups.get(background.source_group_id or "")
                if replacement_group is not None:
                    background.source_group_id = replacement_group.id
                background.source_group = replacement_group
    parent_node.subgroups.append(result)
    return result


def delete_dataset_group(data_group: DataGroup, subgroup: DatasetGroup) -> bool:
    """Remove a nested dataset group (and its contents) from the group tree."""

    removed_group_ids = {subgroup.id, *(node.id for node in subgroup.iter_subgroups())}
    removed_dataset_ids = {dataset.id for dataset in subgroup.iter_datasets()}

    def remove_from(node: Any) -> bool:
        if subgroup in node.subgroups:
            node.subgroups.remove(subgroup)
            return True
        return any(remove_from(child) for child in node.subgroups)

    removed = remove_from(data_group)
    if removed:
        for dataset in data_group.iter_datasets():
            dataset.backgrounds[:] = [
                background
                for background in dataset.backgrounds
                if background.source_group_id not in removed_group_ids
                and background.source_dataset_id not in removed_dataset_ids
            ]
        for node in (data_group, *data_group.iter_subgroups()):
            node.backgrounds[:] = [
                background
                for background in node.backgrounds
                if background.source_group_id not in removed_group_ids
                and background.source_dataset_id not in removed_dataset_ids
            ]
    return removed


def _dataset_group_parent(data_group: DataGroup, subgroup: DatasetGroup) -> Any:
    """Return the node whose ``subgroups`` contains ``subgroup`` (DataGroup or DatasetGroup)."""

    def search(node: Any) -> Any:
        if subgroup in node.subgroups:
            return node
        for child in node.subgroups:
            found = search(child)
            if found is not None:
                return found
        return None

    return search(data_group)


def _dataset_parent_node(root: Any, dataset: DatasetEntry) -> Any:
    """Return the node whose ``datasets`` contains ``dataset`` (DataGroup or DatasetGroup)."""

    if dataset in root.datasets:
        return root
    for subgroup in root.subgroups:
        found = _dataset_parent_node(subgroup, dataset)
        if found is not None:
            return found
    return None


def _dataset_is_effectively_enabled(
    data_group: DataGroup,
    dataset: DatasetEntry,
) -> bool:
    """Return whether a dataset and every containing dataset group are enabled."""

    if not dataset.enabled:
        return False
    node = _dataset_parent_node(data_group, dataset)
    while isinstance(node, DatasetGroup):
        if not node.enabled:
            return False
        node = _dataset_group_parent(data_group, node)
    return True


def _group_contains_node(ancestor: DatasetGroup, node: Any) -> bool:
    """Return True if ``node`` is ``ancestor`` or nested anywhere inside it."""

    return node is ancestor or node in ancestor.iter_subgroups()


def create_model_component(
    group: DataGroup,
    name: str | None = None,
    *,
    type: str = "constant_background",
) -> ModelComponentSpec:
    """Add a model component spec to a data group and return it."""

    if type not in MODEL_TYPE_REGISTRY:
        raise ValueError(f"unknown model type {type!r}")
    config = default_model_config(type)
    if type == "lindhard":
        electronic_sources = [
            candidate
            for candidate in group.models.values()
            if candidate.enabled and candidate.type == "tight_binding"
        ]
        if len(electronic_sources) == 1:
            axes = tuple(
                int(axis)
                for axis in electronic_sources[0].config.get(
                    "periodic_axes",
                    [0, 1, 2],
                )
            )
            representative_q = [0.0, 0.0, 0.0]
            mesh_shift = [0.0, 0.0, 0.0]
            for axis in axes:
                if 0 <= axis < 3:
                    representative_q[axis] = 0.5
                    mesh_shift[axis] = 0.5
            config["plot_q_reduced"] = representative_q
            config["response_mesh_shift"] = mesh_shift
    model = ModelComponentSpec(
        name=next_model_name(group.models),
        type=type,
        parameters=default_model_parameters(type),
        config=config,
        fit_parameters=default_model_fit_parameters(type),
        sharing={
            name: {"mode": "global", "groups": {}}
            for name in default_model_parameters(type)
        },
    )
    if name is not None:
        model.name = name
    model.name = _unique_name(model.name, list(group.models))
    group.models[model.name] = model
    return model


def create_fit_result_entry(
    group: DataGroup,
    parent: FitTimelineEntry,
    *,
    branch_timeline: bool = False,
    replace_current_state: bool = True,
    goodness: dict[str, Any] | None = None,
    channels: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
) -> FitTimelineEntry:
    """Insert a fit-result entry snapshotting the current group state."""

    snapshot = snapshot_data_group_state(group)
    result = FitTimelineEntry(
        name=next_fit_result_name(group.fits),
        kind="result",
        snapshot=copy.deepcopy(snapshot),
        created_at=_timestamp_now(),
        duration_seconds=duration_seconds,
        optimizer=str(parent.optimizer or "least_squares"),
        optimizer_config=copy.deepcopy(parent.optimizer_config),
        goodness=(
            {"status": "not run", "message": "The fit has not been executed."}
            if goodness is None
            else dict(goodness)
        ),
        channels={} if channels is None else dict(channels),
        metadata={} if metadata is None else dict(metadata),
    )
    if branch_timeline:
        timeline = FitTimelineEntry(
            name=next_fit_timeline_name(group.fits),
            kind="timeline",
            created_at=_timestamp_now(),
            children=[result],
        )
        parent.children.append(timeline)
    else:
        siblings = _fit_siblings(group.fits, parent)
        if siblings is None:
            siblings = group.fits
        insert_at = siblings.index(parent) + 1 if parent in siblings else len(siblings)
        siblings.insert(insert_at, result)
        if replace_current_state:
            _replace_current_state(siblings, group)
    return result


def run_group_fit(
    group: DataGroup,
    parent: FitTimelineEntry,
    *,
    branch_timeline: bool = False,
    progress_callback: Any | None = None,
) -> FitTimelineEntry:
    """Execute the group's fit and record the outcome in the fit history.

    The optimizer honors masks, disabled datasets/components, fit weights,
    and scale factors. Optimized parameters are written back to the model
    components before the result snapshot is taken, so the stored snapshot
    reproduces the fitted state. Fit and residual channels are evaluated once
    here and saved on the result entry; they are never recomputed on the fly.
    A failed fit still records an entry with the failure in ``goodness``.
    """

    start = time.perf_counter()
    channels: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    failed = False
    try:
        outcome = perform_group_fit(
            group,
            optimizer_config=parent.optimizer_config,
            progress_callback=progress_callback,
        )
        goodness = outcome["goodness"]
        channels = outcome["channels"]
        metadata = outcome.get("metadata", {})
    except Exception as exc:
        failed = True
        goodness = {"status": "failed", "message": str(exc)}
    duration = time.perf_counter() - start
    result = create_fit_result_entry(
        group,
        parent,
        branch_timeline=branch_timeline,
        replace_current_state=not failed,
        goodness=goodness,
        channels=channels,
        metadata=metadata,
        duration_seconds=duration,
    )
    if failed:
        _current_state_for_failed_fit(group, parent, result)
    return result


def current_state_fit_entry(
    group: DataGroup,
    source: FitTimelineEntry | None = None,
) -> FitTimelineEntry:
    """Return a mutable state snapshot, inheriting its source fit configuration."""

    return FitTimelineEntry(
        name="Current state",
        kind="current",
        snapshot=snapshot_data_group_state(group),
        created_at=_timestamp_now(),
        optimizer=str(source.optimizer or "least_squares") if source is not None else "least_squares",
        optimizer_config=copy.deepcopy(source.optimizer_config) if source is not None else {},
    )


def delete_fit_entry(group: DataGroup, fit_entry: FitTimelineEntry) -> bool:
    """Delete a fit-history entry from a group unless it is the only Initial state."""

    if fit_entry.kind == "initial" and len(group.fits) == 1:
        return False
    siblings = _fit_siblings(group.fits, fit_entry)
    if siblings is None:
        return False
    siblings.remove(fit_entry)
    if not group.fits:
        ensure_fit_history(group)
    return True


def delete_model_component(group: DataGroup, model: ModelComponentSpec) -> None:
    """Remove a model component spec from a data group."""

    for name, existing in list(group.models.items()):
        if existing is model:
            del group.models[name]
            return
    raise ValueError(f"model {model.name!r} is not in data group {group.name!r}")


# Tree-item roles that support deletion. A batch delete is restricted to a
# single logical role so that a range selection of, for example, fit results
# never sweeps in the enclosing workspace or folder headers.
_DELETABLE_TREE_ROLES = frozenset(
    {
        "group", "dataset", "mask", "background", "group_background",
        "dataset_group", "group_mask", "model", "fit", "fit_timeline",
        "analysis", "plot",
    }
)

# Roles that should be treated as interchangeable when deciding which items a
# multi-selection delete may touch together.
_DELETABLE_ROLE_GROUP: dict[str, frozenset[str]] = {
    "fit": frozenset({"fit", "fit_timeline"}),
    "fit_timeline": frozenset({"fit", "fit_timeline"}),
}


def copy_mask_to_dataset(mask: MaskSpec, dataset: DatasetEntry) -> MaskSpec:
    """Copy a mask spec to another dataset, choosing a non-conflicting name."""

    copied = copy.deepcopy(mask)
    copied.name = _unique_name(copied.name, [existing.name for existing in dataset.masks])
    dataset.masks.append(copied)
    return copied


def _move_mask_within_dataset(
    dataset: DatasetEntry,
    mask: MaskSpec,
    target_mask: MaskSpec | None,
) -> bool:
    if mask not in dataset.masks:
        return False
    old_index = dataset.masks.index(mask)
    if target_mask is mask:
        return False
    dataset.masks.pop(old_index)
    if target_mask is None or target_mask not in dataset.masks:
        new_index = len(dataset.masks)
    else:
        new_index = dataset.masks.index(target_mask)
    dataset.masks.insert(new_index, mask)
    return old_index != new_index


def _move_items_within_list(items: list[Any], moving: list[Any], insert_index: int) -> bool:
    """Move selected objects inside one ordered list, preserving selection order."""

    moving_ids = {id(item) for item in moving}
    original = list(items)
    insert_index = max(0, min(insert_index, len(items)))
    insert_index -= sum(1 for index, item in enumerate(items) if id(item) in moving_ids and index < insert_index)
    items[:] = [item for item in items if id(item) not in moving_ids]
    for offset, item in enumerate(moving):
        items.insert(insert_index + offset, item)
    return items != original


def copy_dataset_to_group(dataset: DatasetEntry, group: DataGroup) -> DatasetEntry:
    """Copy a dataset entry to another data group, choosing a non-conflicting name."""

    copied = copy.deepcopy(dataset).copy()
    copied.name = _unique_name(copied.name, group.dataset_names)
    group.add_dataset(copied)
    return copied


def available_mask_types() -> list[str]:
    """Return registered mask type names."""

    return list(MASK_TYPE_DEFINITIONS)


def default_mask_parameters(type: str) -> dict[str, Any]:
    """Return default parameter values for a registered mask type."""

    return {
        name: copy.deepcopy(metadata["default"])
        for name, metadata in MASK_TYPE_DEFINITIONS[type]["parameters"].items()
    }


def ensure_coordinate_range_mask_axes(mask: MaskSpec, dataset: DatasetEntry | None) -> None:
    """Ensure a coordinate-range mask has one coordinate-axis vector per dataset dimension."""

    if mask.type != "coordinate_range" or dataset is None:
        return
    specs = _coordinate_range_axis_specs(dataset.data)
    if not specs:
        return
    stale_keys = [
        key
        for key in mask.parameters
        if key.startswith(COORDINATE_RANGE_AXIS_PREFIX)
        and key[len(COORDINATE_RANGE_AXIS_PREFIX) :].isdigit()
        and int(key[len(COORDINATE_RANGE_AXIS_PREFIX) :]) >= len(specs)
    ]
    for key in stale_keys:
        mask.parameters.pop(key, None)
    for index, spec in enumerate(specs):
        key = f"{COORDINATE_RANGE_AXIS_PREFIX}{index}"
        value = mask.parameters.get(key)
        if not _is_vector_length(value, len(spec["vector"])):
            mask.parameters[key] = copy.deepcopy(spec["vector"])


def mask_parameter_tooltip(type: str, parameter_name: str) -> str:
    """Return standard hover text for a mask parameter editor."""

    if type == "coordinate_range" and parameter_name.startswith(COORDINATE_RANGE_AXIS_PREFIX):
        axis_number = parameter_name.removeprefix(COORDINATE_RANGE_AXIS_PREFIX)
        return "\n".join(
            [
                f"Parameter: {parameter_name}",
                f"Description: Physical coordinate vector for source bin axis {axis_number}. It uses the same axis direction shown by the rebinner.",
                "Allowed values: A list of N finite numbers, where N is the number of dataset dimensions.",
                "Data type: list[float]",
                "Default: the corresponding source bin-axis vector, including projected directions such as [1, -1, 0, 0].",
                "Example: [1, -1, 0, 0]",
            ]
        )
    metadata = MASK_TYPE_DEFINITIONS[type]["parameters"][parameter_name]
    return "\n".join(
        [
            f"Parameter: {parameter_name}",
            f"Description: {metadata['description']}",
            f"Allowed values: {metadata['allowed']}",
            f"Data type: {metadata['type']}",
            f"Default: {_parameter_to_text(metadata['default'])}",
            f"Example: {metadata['example']}",
        ]
    )


def _mask_parameter_names(mask: MaskSpec, dataset: DatasetEntry | None = None) -> list[str]:
    names = list(MASK_TYPE_DEFINITIONS[mask.type]["parameters"])
    if mask.type != "coordinate_range":
        return names
    ensure_coordinate_range_mask_axes(mask, dataset)
    axis_names = sorted(
        (
            key
            for key in mask.parameters
            if key.startswith(COORDINATE_RANGE_AXIS_PREFIX)
            and key[len(COORDINATE_RANGE_AXIS_PREFIX) :].isdigit()
        ),
        key=lambda key: int(key[len(COORDINATE_RANGE_AXIS_PREFIX) :]),
    )
    return [*names, *axis_names]




def _model_parameter_from_qualified_name(
    group: DataGroup, qualified: str
) -> tuple[ModelComponentSpec, str]:
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            continue
        for parameter in model_parameter_names(model):
            if qualified_parameter_name(model.name, parameter) == qualified:
                return model, parameter
    raise ValueError(f"unknown model parameter {qualified!r}")


def _rename_model_constraint_references(
    group: DataGroup,
    renamed: ModelComponentSpec,
    old_name: str,
    new_name: str,
) -> None:
    replacements = {
        qualified_parameter_name(old_name, parameter): qualified_parameter_name(new_name, parameter)
        for parameter in model_parameter_names(renamed)
    }
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            continue
        for constraint in model.constraints:
            reference = constraint.get("reference")
            if isinstance(reference, str) and reference in replacements:
                constraint["reference"] = replacements[reference]
            expression = constraint.get("expression")
            if not isinstance(expression, str):
                continue
            for old, new in replacements.items():
                expression = expression.replace(f"`{old}`", f"`{new}`")
                expression = re.sub(
                    rf"(?<![A-Za-z0-9_.]){re.escape(old)}(?![A-Za-z0-9_.])",
                    new,
                    expression,
                )
            constraint["expression"] = expression


def _validate_model_constraints(group: DataGroup) -> None:
    """Validate workspace constraints without requiring loaded fit datasets."""

    available = {
        qualified_parameter_name(model.name, parameter)
        for model in group.models.values()
        if isinstance(model, ModelComponentSpec)
        for parameter in model_parameter_names(model)
    }
    exact_dependencies: dict[str, tuple[str, ...]] = {}
    targets: set[str] = set()
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            continue
        for constraint in model.constraints:
            parameter = str(constraint.get("parameter", ""))
            target = qualified_parameter_name(model.name, parameter)
            if target not in available:
                raise ValueError(f"constraint target {target!r} is not a model parameter")
            if target in targets:
                raise ValueError(f"parameter {target!r} has more than one constraint")
            targets.add(target)
            if not bool(model.fit_parameters.get(parameter, False)):
                raise ValueError(f"dependent parameter {target!r} must be enabled for fitting")
            if sharing_mode(model, parameter) != "global":
                raise ValueError(f"dependent parameter {target!r} must use global sharing")
            lower, upper = _model_limit_texts(model, parameter)
            if lower or upper:
                raise ValueError(f"constrained parameter {target!r} cannot also have min/max bounds")
            op = str(constraint.get("op", "="))
            if op == "=":
                expression = str(constraint.get("expression", "")).strip()
                dependencies = parameter_expression_names(expression)
                unknown = [name for name in dependencies if name not in available]
                if unknown:
                    raise ValueError(f"constraint on {target!r} references unknown parameter {unknown[0]!r}")
                for dependency in dependencies:
                    dependency_model, dependency_parameter = _model_parameter_from_qualified_name(
                        group, dependency
                    )
                    if sharing_mode(dependency_model, dependency_parameter) != "global":
                        raise ValueError(
                            f"constraint reference {dependency!r} must use global sharing"
                        )
                exact_dependencies[target] = dependencies
            elif op in (">=", "<="):
                reference = constraint.get("reference")
                if isinstance(reference, str) and reference not in available:
                    raise ValueError(f"constraint on {target!r} references unknown parameter {reference!r}")
                if isinstance(reference, str):
                    reference_model, reference_parameter = _model_parameter_from_qualified_name(
                        group, reference
                    )
                    if sharing_mode(reference_model, reference_parameter) != "global":
                        raise ValueError(
                            f"constraint reference {reference!r} must use global sharing"
                        )
                if not isinstance(reference, (str, int, float)):
                    raise ValueError(f"constraint on {target!r} requires one parameter or numeric constant")
            else:
                raise ValueError(f"unsupported constraint relation {op!r}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"cyclic exact constraint involving {name!r}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in exact_dependencies.get(name, ()):
            if dependency in exact_dependencies:
                visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for target in exact_dependencies:
        visit(target)

    values = {
        qualified_parameter_name(model.name, parameter): float(model.parameters.get(parameter, 0.0))
        for model in group.models.values()
        if isinstance(model, ModelComponentSpec)
        for parameter in model_parameter_names(model)
    }
    pending = dict(exact_dependencies)
    while pending:
        for target, dependencies in list(pending.items()):
            if any(dependency in pending for dependency in dependencies):
                continue
            model, parameter = _model_parameter_from_qualified_name(group, target)
            constraint = next(item for item in model.constraints if item.get("parameter") == parameter)
            values[target] = evaluate_parameter_expression(str(constraint["expression"]), values)
            del pending[target]
            break


DEFAULT_BOND_CUTOFF_ANGSTROM = 6.0


class _NoChange(Exception):
    """Signals that a model-config mutation left the state untouched."""




def _mark_tight_binding_crystal_manual(
    model: ModelComponentSpec, crystal: dict[str, Any]
) -> None:
    """Prevent a manually edited structure script from reloading stale CIF data."""

    if model.type == "tight_binding":
        crystal["provenance"] = {"source": "manual"}


def _refresh_tight_binding_builder(model: ModelComponentSpec) -> None:
    """Regenerate builder-derived onsite data after crystal or orbital edits."""

    if model.type != "tight_binding" or not model.config.get("orbital_manifolds"):
        return
    from .electronic_builder import regenerate_tight_binding_onsite_terms

    regenerate_tight_binding_onsite_terms(model)


def set_model_crystal(
    model: ModelComponentSpec,
    crystal: dict[str, Any],
    *,
    group: DataGroup | None = None,
    periodic_axes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Install validated shared crystal geometry on a model component.

    Electronic models default a newly installed crystal to three-dimensional
    periodicity. Heisenberg models clear selections and derived bond data whose
    indices belonged to the previous crystal. The optional data group receives
    the same lattice, space group, and crystal metadata.
    """

    from .crystal import validate_crystal

    payload = copy.deepcopy(crystal)
    validate_crystal(payload)
    model.config["crystal"] = payload
    if model.type == "tight_binding":
        axes = (0, 1, 2) if periodic_axes is None else tuple(
            int(axis) for axis in periodic_axes
        )
        if not axes or len(set(axes)) != len(axes) or any(
            axis not in (0, 1, 2) for axis in axes
        ):
            raise ValueError(
                "periodic_axes must contain one to three unique indices from 0, 1, 2"
            )
        model.config["periodic_axes"] = list(axes)
        if (
            axes == (0, 1, 2)
            and payload.get("sites")
            and str(
                model.config.get("band_path_convention", "hinuma")
            ) in {"hinuma", "setyawan_curtarolo"}
        ):
            from .brillouin_zone import set_tight_binding_standard_path

            set_tight_binding_standard_path(
                model,
                str(model.config.get("band_path_convention", "hinuma")),
            )
        for name in ("spatial_orbits", "expanded_crystal_sites"):
            model.config.pop(name, None)
        if model.config.get("orbital_manifolds"):
            available = {
                str(site.get("label", "")) for site in payload.get("sites", ())
            }
            used = {
                str(item.get("site_label", ""))
                for item in model.config.get("orbital_manifolds", ())
            }
            if used.issubset(available):
                _refresh_tight_binding_builder(model)
            else:
                model.config["orbital_manifolds"] = []
                model.config["onsite_terms"] = []
                model.config["hopping_cutoff_angstrom"] = 0.0
                model.config["hopping_candidates"] = []
                model.config["hopping_terms"] = []
                model.config["spatial_orbits"] = []
                model.config["site_positions"] = []
                model.config["expanded_crystal_sites"] = []
                model.config["model_data"] = {}
                model.config["model_digest"] = ""
                from .electronic_builder import reconcile_tight_binding_parameters

                reconcile_tight_binding_parameters(model)
    elif model.type == "heisenberg_rpa":
        model.config["magnetic_sites"] = []
        for name in ("orbits", "site_positions", "site_rotations"):
            model.config.pop(name, None)
        model.config.pop("anisotropy", None)
        model.config.pop("sia", None)
        reconcile_model_orbit_parameters(model)
    if group is not None:
        group.lattice_parameters = dict(payload["lattice"])
        group.spacegroup = str(payload["spacegroup"])
        group.metadata["crystal"] = copy.deepcopy(payload)
    return payload


def import_cif_into_model(
    model: ModelComponentSpec, path: str, *, group: DataGroup | None = None
) -> dict[str, Any]:
    """Load a CIF file into a model's crystal config (and optionally its group).

    Model-specific derived geometry whose indices referred to the previous
    crystal is cleared. Returns the imported crystal dict.
    """

    from .crystal import crystal_from_cif

    imported = crystal_from_cif(path)
    return set_model_crystal(model, imported, group=group)


def generate_model_bond_orbits(model: ModelComponentSpec) -> list[str]:
    """Generate symmetry-distinct bond orbits from a model's crystal config.

    Writes ``config["orbits"]`` and ``config["site_positions"]`` (the expanded
    magnetic sites the bond indices refer to), reconciles the exchange
    parameters, and returns the orbit labels.
    """

    from .crystal import (
        generate_bond_orbits,
        orbits_to_config,
        site_rotations_to_config,
        sites_to_config,
    )

    crystal = model_crystal_config(model)
    magnetic = [str(label) for label in model.config.get("magnetic_sites", [])]
    if not magnetic:
        raise ValueError(
            "select at least one magnetic site (check 'Magnetic') before "
            "generating bond orbits"
        )
    cutoff = float(model.config.get("bond_cutoff_angstrom", DEFAULT_BOND_CUTOFF_ANGSTROM))
    sites, orbits = generate_bond_orbits(crystal, magnetic, cutoff)
    model.config["site_positions"] = sites_to_config(sites)
    model.config["site_rotations"] = site_rotations_to_config(sites)
    model.config["orbits"] = orbits_to_config(orbits)
    # Re-generating the network invalidates the snapshotted tensor bases (they
    # depend on the specific bonds/sites); drop them so they are regenerated.
    model.config.pop("anisotropy", None)
    model.config.pop("sia", None)
    reconcile_model_orbit_parameters(model)
    return [orbit.label for orbit in orbits]


def set_model_anisotropic_exchange(model: ModelComponentSpec, enabled: bool) -> None:
    """Enable/disable symmetry-allowed anisotropic exchange on every bond orbit.

    When enabling, projects the allowed rank-2 tensor basis of each orbit and
    snapshots it into ``config["anisotropy"]``; orbits with no allowed
    anisotropy contribute nothing. Reconciles the ``<orbit>_S1``/``_D1`` fit
    parameters either way.
    """

    if not enabled:
        model.config.pop("anisotropy", None)
        reconcile_model_orbit_parameters(model)
        return
    from .crystal import (
        expand_magnetic_sites,
        orbits_from_config,
        symmetry_allowed_exchange_basis,
    )

    crystal = model_crystal_config(model)
    magnetic = [str(label) for label in model.config.get("magnetic_sites", [])]
    sites = expand_magnetic_sites(crystal, magnetic)
    orbits = orbits_from_config(model.config.get("orbits", []))
    anisotropy: dict[str, Any] = {}
    for orbit in orbits:
        basis = symmetry_allowed_exchange_basis(crystal, sites, orbit)
        if basis:
            anisotropy[orbit.label] = {"enabled": True, "basis": basis}
    model.config["anisotropy"] = anisotropy
    reconcile_model_orbit_parameters(model)


def set_model_single_ion_anisotropy(model: ModelComponentSpec, enabled: bool) -> None:
    """Enable/disable symmetry-allowed single-ion anisotropy on every site class."""

    if not enabled:
        model.config.pop("sia", None)
        reconcile_model_orbit_parameters(model)
        return
    from .crystal import expand_magnetic_sites, symmetry_allowed_sia_basis

    crystal = model_crystal_config(model)
    magnetic = [str(label) for label in model.config.get("magnetic_sites", [])]
    sites = expand_magnetic_sites(crystal, magnetic)
    sia: dict[str, Any] = {}
    for label in magnetic:
        basis = symmetry_allowed_sia_basis(crystal, label)
        if basis:
            indices = [i for i, site in enumerate(sites) if site.label.startswith(f"{label}_")]
            sia[label] = {"enabled": True, "sites": indices, "basis": basis}
    model.config["sia"] = sia
    reconcile_model_orbit_parameters(model)


def set_model_dipole(model: ModelComponentSpec, enabled: bool) -> None:
    """Enable/disable Ewald dipole-dipole coupling (one fitted ``D_dip``)."""

    model.config["dipole"] = {"enabled": bool(enabled)}
    if enabled:
        from .dipole import dipole_coupling_constant

        model.parameters.setdefault("D_dip", dipole_coupling_constant())
    reconcile_model_orbit_parameters(model)


def set_model_zeeman(model: ModelComponentSpec, enabled: bool) -> None:
    """Enable/disable the Zeeman (applied-field) term (g_factor + ratios)."""

    model.config["zeeman"] = {"enabled": bool(enabled)}
    reconcile_model_orbit_parameters(model)


# Default closure config seeded when a mode is first selected.
_CLOSURE_CONFIG_DEFAULTS = {
    "mode": "none",
    "energy_cutoff_mev": 100.0,
    "bz_grid": 16,
    "omega_points": 200,
    "moment_mode": "fixed",
    "moment_target": 1.0,
}


def set_model_closure(model: ModelComponentSpec, updates: dict[str, Any]) -> None:
    """Merge closure settings into ``config["closure"]`` and reconcile params.

    ``updates`` carries any of the closure keys (``mode``, ``energy_cutoff_mev``,
    ``bz_grid``, ``omega_points``, ``moment_mode``, ``moment_target``). Selecting
    mode ``"none"`` drops the section entirely so the component takes the exact
    legacy code path; any other mode seeds the defaults and adds the closure's
    dynamic fit parameters (``m2_total``/``mode_coupling_u``/``total_amplitude``).
    """

    closure = dict(_CLOSURE_CONFIG_DEFAULTS)
    closure.update(model.config.get("closure") or {})
    closure.update(updates)
    if str(closure.get("mode", "none")).lower() == "none":
        model.config.pop("closure", None)
    else:
        model.config["closure"] = closure
    reconcile_model_orbit_parameters(model)


def dataset_details_text(dataset: DatasetEntry, *, group: DataGroup | None = None) -> str:
    """Return a human-readable summary of one imported dataset."""

    lines: list[str] = []
    for title, section_lines in dataset_detail_sections(dataset, group=group):
        if lines:
            lines.append("")
        lines.append(title)
        lines.extend(section_lines)
    return "\n".join(lines)


def fit_details_text(fit_entry: FitTimelineEntry) -> str:
    """Return a human-readable summary of one fit-history entry."""

    lines = [
        f"Type: {fit_entry.kind}",
        f"Created: {fit_entry.created_at or '-'}",
        f"Optimizer: {fit_entry.optimizer or '-'}",
        f"Duration: {_format_number(fit_entry.duration_seconds) + ' s' if fit_entry.duration_seconds is not None else '-'}",
    ]
    lines.extend(_fit_chi_squared_summary_lines(fit_entry))
    if fit_entry.optimizer_config:
        lines.extend(["", "Optimizer config"])
        lines.extend(_mapping_lines(fit_entry.optimizer_config))
    if fit_entry.goodness:
        lines.extend(["", "Goodness of fit"])
        lines.extend(_mapping_lines(fit_entry.goodness))
    if fit_entry.channels:
        lines.extend(["", f"Stored fit channels: {', '.join(sorted(fit_entry.channels))}"])
    snapshot = fit_entry.snapshot or {}
    lines.extend(
        [
            "",
            "Snapshot",
            f"Datasets: {len(snapshot.get('datasets', []))}",
            f"Models: {len(snapshot.get('models', []))}",
        ]
    )
    if fit_entry.children:
        lines.extend(["", f"Timeline entries: {len(fit_entry.children)}"])
    return "\n".join(lines)


def _fit_chi_squared_summary_lines(fit_entry: FitTimelineEntry) -> list[str]:
    """Return compact goodness-of-fit rows for a completed fit result."""

    if fit_entry.kind != "result":
        return []
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}

    def formatted(key: str) -> str:
        try:
            return _format_number(float(goodness[key]))
        except (KeyError, TypeError, ValueError):
            return "-"

    return [f"Chi^2: {formatted('chi2')}", f"Reduced Chi^2: {formatted('reduced_chi2')}"]


def dataset_detail_sections(
    dataset: DatasetEntry,
    *,
    group: DataGroup | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ordered dataset detail sections for text and GUI rendering."""

    axes_lines, data_lines = _dataset_axes_and_data_lines(dataset.data)
    data_lines.extend(_dataset_fit_summary_lines(dataset, group=group))
    data_lines.append(f"Masks: {len(dataset.masks)}")
    data_lines.append(f"Backgrounds: {len(dataset.backgrounds)}")
    source_lines = _dataset_source_lines(dataset)
    metadata_lines = _dataset_metadata_lines(dataset)
    if dataset.parameters:
        metadata_lines = [*metadata_lines, "", "Parameters", *_mapping_lines(dataset.parameters)]
    return [
        (
            "Dataset",
            [
                f"Name: {dataset.name}",
                "Dataset",
                f"Data type: {data_type_label(dataset.data_type)}",
                f"Kind: {dataset.kind or '-'}",
                f"Enabled for fitting: {dataset.enabled}",
                f"Fit weight: {_format_number(dataset.fit_weight)}",
                f"Visualization only: {bool(dataset.enabled and dataset.fit_weight == 0.0)}",
                f"Scale factor: {_format_number(dataset.scale_factor)}",
                f"Scale fitted: {bool(dataset.scale_factor_vary)}",
                f"Shared scale: {dataset.scale_factor_group or '-'}",
            ],
        ),
        ("Axes", axes_lines),
        ("Crystal", _dataset_crystal_lines(dataset, group)),
        ("Data", data_lines),
        ("Source", source_lines or ["No source file recorded."]),
        ("Metadata", metadata_lines or ["No additional metadata."]),
    ]

def _effective_dataset_entries(
    group: DataGroup,
    node: DataGroup | DatasetGroup,
    *,
    use_composite: bool,
    force_rebin: bool,
    force_masks: bool,
    progress_callback: Any | None,
    include_disabled_groups: bool = True,
    batch_progress: dict[str, Any] | None = None,
):
    if (
        isinstance(node, DatasetGroup)
        and not node.enabled
        and not include_disabled_groups
    ):
        return
    scope = _composite_scope(group, node)
    if use_composite and data_group_composite_enabled(scope):
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=node.name,
            kind="dataset group",
            completed=False,
        )
        yield composite_dataset_entry(
            scope,
            force_rebin=force_rebin,
            progress_callback=progress_callback,
        )
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=node.name,
            kind="dataset group",
            completed=True,
        )
        return
    if isinstance(node, DatasetGroup) and isinstance(
        node.metadata.get("mdevent"), dict
    ):
        return
    for dataset in node.datasets:
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=dataset.name,
            kind="dataset",
            completed=False,
        )
        yield dataset
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=dataset.name,
            kind="dataset",
            completed=True,
        )
    for subgroup in node.subgroups:
        yield from _effective_dataset_entries(
            group,
            subgroup,
            use_composite=use_composite,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
            include_disabled_groups=include_disabled_groups,
            batch_progress=batch_progress,
        )


def _effective_dataset_entry_count(
    group: DataGroup,
    node: DataGroup | DatasetGroup,
    *,
    use_composite: bool,
    include_disabled_groups: bool = True,
) -> int:
    """Count effective viewer entries without preparing their data."""

    if (
        isinstance(node, DatasetGroup)
        and not node.enabled
        and not include_disabled_groups
    ):
        return 0
    scope = _composite_scope(group, node)
    if use_composite and data_group_composite_enabled(scope):
        return 1
    if isinstance(node, DatasetGroup) and isinstance(node.metadata.get("mdevent"), dict):
        return 0
    return len(node.datasets) + sum(
        _effective_dataset_entry_count(
            group,
            subgroup,
            use_composite=use_composite,
            include_disabled_groups=include_disabled_groups,
        )
        for subgroup in node.subgroups
    )


def _viewer_progress_work_counts(
    group: DataGroup,
    node: DataGroup | DatasetGroup,
    *,
    use_composite: bool,
    include_disabled_groups: bool = True,
) -> tuple[int, int, int]:
    """Count materializations, named composites, and viewer-ready entries."""

    if (
        isinstance(node, DatasetGroup)
        and not node.enabled
        and not include_disabled_groups
    ):
        return 0, 0, 0
    scope = _composite_scope(group, node)
    if use_composite and data_group_composite_enabled(scope):
        binnings = data_group_composite_binnings(scope)
        named = sum(
            bool(item["config"].get("enabled", False))
            for item in binnings[1:]
        )
        return 1, named, 1 + named
    if isinstance(node, DatasetGroup) and isinstance(node.metadata.get("mdevent"), dict):
        return 0, 0, 0

    materializations = len(node.datasets)
    viewer_entries = 0
    for dataset in node.datasets:
        binnings = (
            dataset_rebin_binnings(dataset)
            if isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict)
            else []
        )
        viewer_entries += 1 + sum(
            bool(item["config"].get("enabled", False))
            for item in binnings[1:]
        )
    named_composites = 0
    for subgroup in node.subgroups:
        child_materializations, child_named, child_viewer_entries = (
            _viewer_progress_work_counts(
                group,
                subgroup,
                use_composite=use_composite,
                include_disabled_groups=include_disabled_groups,
            )
        )
        materializations += child_materializations
        named_composites += child_named
        viewer_entries += child_viewer_entries
    return materializations, named_composites, viewer_entries


def _viewer_batch_progress_state(
    group: DataGroup,
    *,
    use_composite: bool,
    include_window: bool,
    completed: int = 0,
) -> dict[str, Any]:
    """Return a truthful outer-work denominator for opening a data viewer."""

    materializations, named_composites, viewer_entries = _viewer_progress_work_counts(
        group,
        group,
        use_composite=use_composite,
    )
    return {
        "total": (
            materializations
            + named_composites
            + viewer_entries
            + int(include_window)
        ),
        "completed": completed,
        "batch_operation": "Preparing data viewer",
        "batch_unit": "work item",
    }


def _report_effective_dataset_batch(
    progress_callback: Any | None,
    batch_progress: dict[str, Any] | None,
    *,
    name: str,
    kind: str,
    completed: bool,
) -> None:
    """Report the outer viewer item around its detailed rebin events."""

    if progress_callback is None or batch_progress is None:
        return
    if completed:
        batch_progress["completed"] += 1
    event = {
        "stage": "rebin_batch",
        "batch_total": batch_progress["total"],
        "batch_completed": batch_progress["completed"],
        "batch_name": name,
        "batch_kind": kind,
        "batch_item_complete": completed,
        "message": (
            f"finished {kind} {name}"
            if completed
            else f"preparing {kind} {name}"
        ),
    }
    if batch_progress.get("batch_operation"):
        event["batch_operation"] = batch_progress["batch_operation"]
        event["batch_unit"] = batch_progress.get("batch_unit", "work item")
    progress_callback(event)


def _named_binning_progress_callback(
    progress_callback: Any | None,
    *,
    name: str,
    index: int,
    total: int,
    kind: str,
) -> Any | None:
    """Wrap detailed rebin progress with its named-binning batch position."""

    if progress_callback is None:
        return None

    def report(event: dict[str, Any]) -> None:
        progress_callback(
            {
                **event,
                "batch_total": total,
                "batch_completed": index,
                "batch_name": name,
                "batch_kind": kind,
                "batch_item_complete": False,
                "rebin_total": total,
                "rebin_completed": index,
                "rebin_name": name,
            }
        )

    progress_callback(
        {
            "stage": "rebin_batch",
            "batch_total": total,
            "batch_completed": index,
            "batch_name": name,
            "batch_kind": kind,
            "batch_item_complete": False,
            "rebin_total": total,
            "rebin_completed": index,
            "rebin_name": name,
            "message": f"preparing {kind} {name}",
        }
    )
    return report


def _finish_named_binning_progress(
    progress_callback: Any | None,
    *,
    name: str,
    completed: int,
    total: int,
    kind: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "stage": "rebin_batch",
            "batch_total": total,
            "batch_completed": completed,
            "batch_name": name,
            "batch_kind": kind,
            "batch_item_complete": True,
            "rebin_total": total,
            "rebin_completed": completed,
            "rebin_name": name,
            "message": f"finished {kind} {name}",
        }
    )


def _composite_scopes(group: DataGroup):
    yield group
    for subgroup in group.iter_subgroups():
        yield _composite_scope(group, subgroup)


def slice_viewer_datasets(
    group: DataGroup,
    *,
    use_composite: bool = True,
    unmask_model: bool = False,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
    defer_progress_completion: bool = False,
) -> tuple[list[MDHistoData], list[str]]:
    """Return data and labels for every dataset in the group tree, with shared masks.

    GUI callers may set ``defer_progress_completion`` while they construct and
    render the viewer window after these numerical preparations return.
    """

    data: list[MDHistoData] = []
    names: list[str] = []
    model_channels = current_model_channels(
        group,
        force_masks=force_masks,
        unmask_model=unmask_model,
    )
    batch_progress = (
        _viewer_batch_progress_state(
            group,
            use_composite=use_composite,
            include_window=defer_progress_completion,
        )
        if progress_callback is not None
        else None
    )
    entries = list(
        _effective_dataset_entries(
            group,
            group,
            use_composite=use_composite,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
            batch_progress=batch_progress,
        )
    )
    entries = _entries_with_visualization_binnings(
        group,
        entries,
        force_rebin=force_rebin,
        progress_callback=progress_callback,
        batch_progress=batch_progress,
    )
    for dataset in entries:
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=dataset.name,
            kind="viewer dataset",
            completed=False,
        )
        if progress_callback is not None:
            progress_callback({
                "stage": "viewer_prepare",
                "iteration": 0,
                "total": 0,
                "message": f"preparing viewer data for {dataset.name}",
            })
        is_composite = bool(dataset.metadata.get("composite"))
        extra_masks = [] if is_composite else effective_dataset_masks(group, dataset)
        view_data = dataset_for_slice_viewer(
            dataset,
            extra_masks=extra_masks,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
        )
        if view_data is not None:
            view_data = attach_fit_channels_to_view(
                group,
                dataset.name,
                view_data,
                fallback_payload=model_channels.get(dataset.name),
            )
            data.append(view_data)
            names.append(dataset.name)
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name=dataset.name,
            kind="viewer dataset",
            completed=True,
        )
    if defer_progress_completion:
        _report_effective_dataset_batch(
            progress_callback,
            batch_progress,
            name="viewer window",
            kind="stage",
            completed=False,
        )
    return data, names


def _finish_viewer_batch_progress(
    group: DataGroup,
    *,
    use_composite: bool,
    progress_callback: Any | None,
) -> None:
    """Complete the viewer-window work item reserved by the data preparation."""

    if progress_callback is None:
        return
    state = _viewer_batch_progress_state(
        group,
        use_composite=use_composite,
        include_window=True,
    )
    state["completed"] = max(state["total"] - 1, 0)
    _report_effective_dataset_batch(
        progress_callback,
        state,
        name="viewer window",
        kind="stage",
        completed=True,
    )


def _binning_view_name(source_name: str, binning_name: str) -> str:
    return f"{source_name} · {binning_name}"


def _entries_with_visualization_binnings(
    group: DataGroup,
    entries: list[DatasetEntry],
    *,
    force_rebin: bool,
    progress_callback: Any | None = None,
    batch_progress: dict[str, Any] | None = None,
) -> list[DatasetEntry]:
    """Add zero-weight named visualization binnings beside canonical fit entries."""

    expanded: list[DatasetEntry] = []
    scopes_by_id = {
        (scope.node.id if isinstance(scope, _CompositeScope) else None): scope
        for scope in _composite_scopes(group)
    }
    for dataset in entries:
        source_name = dataset.name
        if dataset.metadata.get("composite"):
            scope = scopes_by_id.get(dataset.metadata.get("composite_scope_id"))
            binnings = data_group_composite_binnings(scope) if scope is not None else []
        else:
            scope = None
            binnings = (
                dataset_rebin_binnings(dataset)
                if isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict)
                else []
            )
        if not binnings:
            expanded.append(dataset)
            continue
        fit_item = binnings[0]
        fit_entry = dataset.copy(name=source_name)
        owner = getattr(dataset, "_derived_owner_group", None)
        if owner is not None:
            # DatasetEntry.copy intentionally copies only serializable fields.
            # Preserve this process-local link on temporary viewer aliases so
            # a live derived recipe can still resolve its owning analysis.
            fit_entry._derived_owner_group = owner
        fit_entry.id = dataset.id
        fit_entry._viewer_source_dataset_id = dataset.id
        fit_entry.enabled = _dataset_is_effectively_enabled(group, dataset)
        fit_entry.metadata = {
            **copy.deepcopy(dataset.metadata),
            "binning_id": fit_item["id"],
            "binning_name": fit_item["name"],
            "source_dataset_name": source_name,
        }
        expanded.append(fit_entry)
        for item in binnings[1:]:
            config = item["config"]
            if not bool(config.get("enabled", False)):
                continue
            view_name = _binning_view_name(source_name, str(item["name"]))
            if scope is not None:
                _report_effective_dataset_batch(
                    progress_callback,
                    batch_progress,
                    name=view_name,
                    kind="named composite binning",
                    completed=False,
                )
                aux_data = _cached_composite_dataset_data(
                    scope,
                    force_rebin=force_rebin,
                    progress_callback=progress_callback,
                    config_override=config,
                    binning_id=item["id"],
                )
                auxiliary = dataset.copy(data=aux_data, name=view_name)
                _report_effective_dataset_batch(
                    progress_callback,
                    batch_progress,
                    name=view_name,
                    kind="named composite binning",
                    completed=True,
                )
            else:
                parameters = copy.deepcopy(dataset.parameters)
                parameters[DATASET_REBIN_KEY] = config
                auxiliary = dataset.copy(name=view_name, parameters=parameters)
                if owner is not None:
                    auxiliary._derived_owner_group = owner
            auxiliary.id = f"{dataset.id}:{item['id']}"
            auxiliary._viewer_source_dataset_id = dataset.id
            auxiliary.fit_weight = 0.0
            auxiliary.scale_factor_vary = False
            auxiliary.metadata = {
                **copy.deepcopy(dataset.metadata),
                "binning_id": item["id"],
                "binning_name": item["name"],
                "source_dataset_name": source_name,
                "visualization_binning": True,
            }
            expanded.append(auxiliary)
    return expanded


def _waterfall_group_keys(group: DataGroup, names: list[str]) -> list[str]:
    """Return the immediate project data-group key for each viewer dataset."""

    by_name = {dataset.name: "root" for dataset in group.datasets}

    def visit(node: DatasetGroup, path: tuple[str, ...]) -> None:
        key = "/".join(path)
        for dataset in node.datasets:
            by_name[dataset.name] = key
        for subgroup in node.subgroups:
            visit(subgroup, (*path, subgroup.name))

    for subgroup in group.subgroups:
        visit(subgroup, (subgroup.name,))
    return [
        by_name.get(name.split(" · ", 1)[0], name.split(" · ", 1)[0])
        for name in names
    ]


KINEMATIC_KF_KI_INCLUDED_KEY = "kf_ki_included"


@dataclass
class FitDataBundle:
    """Fit-ready representations of one dataset.

    ``view`` is the masked, scaled data exactly as the data viewer shows it.
    ``points`` flattens every view point into :class:`PointData4D`; the point
    mask is ``False`` where the view masks a point, so the optimizer only sees
    unmasked data while the fitted model can still be evaluated everywhere.
    ``grid_shape`` reshapes point-ordered channel arrays back onto the MDHisto
    grid (``None`` for point lists).
    """

    dataset: DatasetEntry
    view: Any
    points: PointData4D
    grid_shape: tuple[int, ...] | None


def fit_data_bundle(
    group: DataGroup,
    dataset: DatasetEntry,
    *,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> FitDataBundle | None:
    """Build the fit-ready views of one dataset, or ``None`` if unsupported."""

    extra_masks = effective_dataset_masks(group, dataset)
    if dataset.scale_factor_vary:
        raw_view = _viewer_data_before_scale(
            dataset,
            extra_masks=extra_masks,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
        )
        view = (
            _with_viewer_dataset_metadata(
                dataset,
                _apply_spectral_channel_view(dataset, raw_view),
            )
            if raw_view is not None
            else None
        )
    else:
        view = dataset_for_slice_viewer(
            dataset,
            extra_masks=extra_masks,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
        )
    if isinstance(view, MDHistoData):
        points = _point_data_from_mdhisto_view(view)
    elif isinstance(view, PointData4D):
        points = view
    elif isinstance(view, PointListData) and dataset.data_type == "magnetization":
        points = _magnetization_point_data(view, group, dataset)
    elif isinstance(view, PointListData) and dataset.data_type == "heat_capacity":
        points = _heat_capacity_point_data(view, dataset)
    elif isinstance(view, PointListData):
        points = _point_data_from_point_list_view(view)
    else:
        return None
    points = _apply_sample_context_to_points(group, dataset, points)
    # Record the data type so evaluators can branch (e.g. bulk magnetization
    # vs inelastic intensity) without threading it through every call.
    metadata = dict(points.metadata)
    metadata.setdefault("data_type", dataset.data_type)
    points = points.with_updates(metadata=metadata)
    return FitDataBundle(
        dataset=dataset,
        view=view,
        points=points,
        grid_shape=view.shape if isinstance(view, MDHistoData) else None,
    )


def _field_direction_cartesian(group: DataGroup, dataset: DatasetEntry) -> np.ndarray:
    """Unit Cartesian direction of the applied field, defaulting to z-hat.

    Uses the direction stored in ``dataset.parameters["magnetic_field"]`` (uvw
    or hkl frame, oriented with the group lattice); falls back to the c*/z axis
    when no direction or lattice is available (an MPMS scan along the field).
    """

    payload = dataset.parameters.get("magnetic_field")
    lattice = group.lattice_parameters
    if (
        isinstance(payload, dict)
        and payload.get("direction") is not None
        and isinstance(lattice, dict)
        and all(key in lattice for key in ("a", "b", "c"))
    ):
        try:
            vector = magnetic_field_vector(
                1.0, payload["direction"], str(payload.get("frame", "uvw")), lattice
            )
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                return np.asarray(vector, dtype=float) / norm
        except (TypeError, ValueError):
            pass
    return np.array([0.0, 0.0, 1.0])


# Column name fragments that identify the MPMS field sweep axis (in oersted).
_MPMS_FIELD_COLUMN_HINTS = ("magnetic field", "field")
_OERSTED_PER_TESLA = 1.0e4


def _magnetization_point_data(
    view: PointListData, group: DataGroup, dataset: DatasetEntry
) -> PointData4D:
    """Map an MPMS magnetization point list onto fit points.

    Magnetization carries no momentum transfer, so ``H=K=L=E=0``; the
    Temperature column becomes the per-point temperature and the Magnetic
    Field column (oersted) becomes the per-point Cartesian field in tesla,
    oriented along the dataset's field direction. The moment channel is the
    fitted intensity. Absolute-unit sample metadata is forwarded so the
    evaluator can pin the emu/mol normalization.
    """

    if not view.channel_labels:
        raise ValueError("magnetization dataset defines no moment channel")
    config = point_list_config(dataset)
    susc = config.get("susceptibility", {})
    preferred = (
        SUSCEPTIBILITY_CHANNEL_LABEL
        if susc.get("enabled") and SUSCEPTIBILITY_CHANNEL_LABEL in view.channel_labels
        else str(susc.get("moment") or view.channel_labels[0])
    )
    label = preferred if preferred in view.channel_labels else view.channel_labels[0]
    intensity = np.asarray(view.channel_values(label), dtype=float)
    errors = view.channel_errors(label)
    sigma_known = errors is not None
    sigma = (
        np.asarray(errors, dtype=float)
        if sigma_known
        else np.ones(intensity.shape, dtype=float)
    )
    n = intensity.size
    zeros = np.zeros(n, dtype=float)

    def _find_column(hints: tuple[str, ...]) -> tuple[str, np.ndarray] | None:
        for name in view.columns:
            lowered = name.strip().lower()
            if any(hint in lowered for hint in hints):
                return name, np.asarray(view.column(name), dtype=float)
        return None

    temperature_match = _find_column(("temperature",))
    field_match = _find_column(_MPMS_FIELD_COLUMN_HINTS)
    temperature = None if temperature_match is None else temperature_match[1]
    direction = _field_direction_cartesian(group, dataset)
    if field_match is not None:
        from .quantities import convert_quantity

        field_name, field_values = field_match
        field_tesla = convert_quantity(
            field_values, "magnetic_field", view.unit(field_name), "T"
        )
        magnetic_field = field_tesla[:, None] * direction[None, :]
    else:
        magnetic_field = None

    mask = np.isfinite(intensity) & np.isfinite(sigma)
    if sigma_known:
        mask &= sigma > 0.0

    metadata: dict[str, Any] = {
        "fit_channel": label,
        "sigma_known": sigma_known,
        "data_type": "magnetization",
        "quantity_type": view.channel_quantity_type(label),
        "unit": view.unit(view.channel(label)["value"]),
        "field_direction_cartesian": direction.tolist(),
    }
    for key in ("absolute_units", "sample_mass_mg", "molar_mass_g_mol"):
        if key in dataset.parameters:
            metadata[key] = dataset.parameters[key]
    return PointData4D(
        H=zeros, K=zeros, L=zeros, E=zeros,
        intensity=intensity, sigma=sigma, mask=mask,
        temperature=temperature if temperature is not None else dataset.parameters.get("temperature"),
        magnetic_field=magnetic_field,
        metadata=metadata,
    )


def _heat_capacity_point_data(view: PointListData, dataset: DatasetEntry) -> PointData4D:
    """Map a molar heat-capacity or C/T channel onto fit points."""

    config = point_list_config(dataset).get("heat_capacity", {})
    preferred = str(config.get("fit_channel", HEAT_CAPACITY_CHANNEL_LABEL))
    label = preferred if preferred in view.channel_labels else view.channel_labels[0]
    channel = view.channel(label)
    quantity_type = view.channel_quantity_type(label)
    if quantity_type not in {"heat_capacity", "heat_capacity_over_temperature"}:
        raise ValueError("heat-capacity fitting requires a molar C or C/T channel")
    temperature_name = str(config.get("temperature", ""))
    if temperature_name not in view.columns:
        temperature_name = next(
            (name for name in view.columns if "temp" in name.lower() and view.unit(name) == "K"),
            "",
        )
    if not temperature_name:
        raise ValueError("heat-capacity dataset requires a temperature column")
    temperature = np.asarray(view.column(temperature_name), dtype=float)
    intensity = np.asarray(view.channel_values(label), dtype=float)
    errors = view.channel_errors(label)
    sigma_known = errors is not None
    sigma = np.asarray(errors, dtype=float) if sigma_known else np.ones(intensity.shape)
    mask = np.isfinite(temperature) & np.isfinite(intensity) & np.isfinite(sigma)
    if sigma_known:
        mask &= sigma > 0.0
    zeros = np.zeros(intensity.shape, dtype=float)
    return PointData4D(
        H=zeros,
        K=zeros,
        L=zeros,
        E=temperature,
        intensity=intensity,
        sigma=sigma,
        mask=mask,
        temperature=temperature,
        metadata={
            "fit_coordinate_mapping": {"E": temperature_name},
            "fit_channel": label,
            "sigma_known": sigma_known,
            "data_type": "heat_capacity",
            "quantity_type": quantity_type,
            "unit": view.unit(channel["value"]),
        },
    )


def effective_dataset_temperature(
    group: DataGroup, dataset: DatasetEntry
) -> float | None:
    """Return the dataset temperature override in K, or ``None`` if unset.

    ``dataset.parameters["temperature"]`` takes precedence over any
    temperature carried by the imported data itself.
    """

    value = dataset.parameters.get("temperature")
    if value in (None, ""):
        return None
    return float(value)


def effective_dataset_field(
    group: DataGroup, dataset: DatasetEntry
) -> np.ndarray | None:
    """Return the dataset's applied field as a Cartesian Tesla 3-vector.

    Reads ``dataset.parameters["magnetic_field"]`` — a dict with
    ``magnitude_T`` (Tesla), ``direction`` (3-vector), and ``frame`` (``"uvw"``
    for direct-lattice directions, ``"hkl"`` for reciprocal). Returns ``None``
    when no field is set, the magnitude is zero, or the group has no lattice
    parameters to orient the direction with.
    """

    payload = dataset.parameters.get("magnetic_field")
    if not isinstance(payload, dict):
        return None
    try:
        magnitude = float(payload.get("magnitude_T", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if magnitude == 0.0:
        return None
    direction = payload.get("direction")
    lattice = group.lattice_parameters
    if (
        direction is None
        or not isinstance(lattice, dict)
        or not all(key in lattice for key in ("a", "b", "c"))
    ):
        return None
    try:
        return magnetic_field_vector(
            magnitude, direction, str(payload.get("frame", "uvw")), lattice
        )
    except (TypeError, ValueError):
        return None


def _apply_sample_context_to_points(
    group: DataGroup, dataset: DatasetEntry, points: PointData4D
) -> PointData4D:
    """Return fit points with per-dataset temperature, field, and lattice context.

    Physics models read the sample temperature from ``PointData4D.temperature``
    and the applied field from ``PointData4D.magnetic_field`` (Cartesian
    Tesla), and convert HKL to ``|Q|`` through ``rlu_to_inv_angstrom_matrix``
    metadata; all are supplied here so models never depend on GUI state.
    """

    # Per-point temperature/field (e.g. an MPMS sweep) is the physics axis and
    # must not be overwritten by a scalar sample-environment override.
    temperature = points.temperature
    magnetic_field = points.magnetic_field
    metadata = dict(points.metadata)
    if not isinstance(temperature, np.ndarray):
        override = effective_dataset_temperature(group, dataset)
        if override is not None:
            temperature = override
    if not (isinstance(magnetic_field, np.ndarray) and magnetic_field.ndim == 2):
        field_vector = effective_dataset_field(group, dataset)
        if field_vector is not None:
            magnetic_field = field_vector
    if (
        "rlu_to_inv_angstrom_matrix" not in metadata
        and isinstance(group.lattice_parameters, dict)
        and all(key in group.lattice_parameters for key in ("a", "b", "c"))
    ):
        lattice = group.lattice_parameters
        matrix = reciprocal_basis_from_lattice_parameters(
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice.get("alpha", 90.0)),
            float(lattice.get("beta", 90.0)),
            float(lattice.get("gamma", 90.0)),
        )
        metadata["lattice_parameters"] = {
            "a": float(lattice["a"]),
            "b": float(lattice["b"]),
            "c": float(lattice["c"]),
            "alpha": float(lattice.get("alpha", 90.0)),
            "beta": float(lattice.get("beta", 90.0)),
            "gamma": float(lattice.get("gamma", 90.0)),
            "angle_units": "degree",
            "include_2pi": True,
        }
        metadata["rlu_to_inv_angstrom_matrix"] = matrix.tolist()
    contextual = points.with_updates(
        temperature=temperature,
        magnetic_field=magnetic_field,
        metadata=metadata,
    )
    return _apply_kinematic_normalization_to_points(dataset, contextual)


def _point_data_from_mdhisto_view(data: MDHistoData) -> PointData4D:
    """Flatten a masked MDHisto view into fit points using axis roles."""

    coords = _mdhisto_coordinate_grids(data)
    zeros = np.zeros(data.shape, dtype=float)
    keep = ~np.asarray(data.mask, dtype=bool)
    keep &= mdhisto_measured_bins(data)
    metadata: dict[str, Any] = {
        "fit_coordinates": sorted(
            name for name in ("H", "K", "L", "E", "q_modulus") if name in coords
        ),
    }
    for key in (
        "oriented_lattice",
        "coordinate_units",
        "rlu_to_inv_angstrom_matrix",
        "nfit_kinematic_kf_ki_normalized",
        "nfit_kinematic_kf_ki_source",
        "signal_quantity_type",
        "signal_unit",
        "spectral_observable",
    ):
        if key in data.metadata:
            metadata[key] = data.metadata[key]
    if "signal_quantity_type" in metadata:
        metadata["quantity_type"] = metadata["signal_quantity_type"]
    if "signal_unit" in metadata:
        metadata["unit"] = metadata["signal_unit"]
    temperature = data.metadata.get("temperature")
    from .metadata_dimensions import metadata_temperature_grid

    temperatures = metadata_temperature_grid(data)
    powder_q = coords.get("q_modulus")
    powder_only = powder_q is not None and not any(
        axis.role in {"h", "k", "l"} for axis in data.axes
    )
    if powder_only:
        metadata["coordinate_units"] = "1/angstrom"
        metadata["powder_q_modulus_axis"] = True
    return PointData4D(
        # PointData4D has a Cartesian-vector momentum slot. For powder data,
        # place |Q| on x and tag the coordinates as inverse angstroms so
        # q_modulus_inv_angstrom recovers the measured scalar without a lattice.
        H=(
            powder_q
            if powder_only
            else coords.get("H", zeros)
        ).ravel(),
        K=coords.get("K", zeros).ravel(),
        L=coords.get("L", zeros).ravel(),
        E=coords.get("E", zeros).ravel(),
        intensity=np.asarray(data.signal, dtype=float).ravel(),
        sigma=np.asarray(data.errors, dtype=float).ravel(),
        mask=keep.ravel(),
        temperature=(
            np.broadcast_to(temperatures, data.shape).ravel() if temperatures is not None
            else float(temperature) if temperature is not None else None
        ),
        metadata=metadata,
    )


def _point_data_from_point_list_view(data: PointListData) -> PointData4D:
    """Map a point-list view onto fit points.

    Columns named ``H``, ``K``, ``L``, or ``E`` (case-insensitive) become the
    matching fit coordinates. When no energy-like column exists, the first
    coordinate column is stored in ``E`` so one-dimensional models have an
    axis to work with; the mapping is recorded in the point metadata.
    """

    if not data.channel_labels:
        raise ValueError("point-list dataset defines no data channels")
    label = data.channel_labels[0]
    intensity = np.asarray(data.channel_values(label), dtype=float)
    errors = data.channel_errors(label)
    sigma_known = errors is not None
    sigma = (
        np.asarray(errors, dtype=float)
        if sigma_known
        else np.ones(intensity.shape, dtype=float)
    )

    columns_by_role: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    powder_q_name: str | None = None
    for name in data.coordinate_names:
        role = name.strip().upper()
        if role in ("H", "K", "L", "E") and role not in columns_by_role:
            columns_by_role[role] = np.asarray(data.column(name), dtype=float)
            mapping[role] = name
        elif name.strip().lower() in {"q", "|q|", "q_modulus"}:
            powder_q_name = name
            columns_by_role["H"] = np.asarray(data.column(name), dtype=float)
            mapping["q_modulus"] = name
    if "E" not in columns_by_role and data.coordinate_names:
        first = data.coordinate_names[0]
        if first not in mapping.values():
            columns_by_role["E"] = np.asarray(data.column(first), dtype=float)
            mapping["E"] = first

    n = intensity.size
    zeros = np.zeros(n, dtype=float)
    mask = np.isfinite(intensity) & np.isfinite(sigma)
    if sigma_known:
        mask &= sigma > 0.0
    temperature: float | np.ndarray | None = None
    for name in data.columns:
        if name.strip().lower() == "temperature":
            temperature = np.asarray(data.column(name), dtype=float)
            break
    if temperature is None and data.metadata.get("temperature") is not None:
        temperature = float(data.metadata["temperature"])
    metadata: dict[str, Any] = {
        "fit_coordinate_mapping": mapping,
        "fit_channel": label,
        "sigma_known": sigma_known,
    }
    for key in (
        "spectral_observable",
        "signal_quantity_type",
        "signal_unit",
    ):
        if key in data.metadata:
            metadata[key] = data.metadata[key]
    if powder_q_name is not None:
        metadata["coordinate_units"] = "1/angstrom"
        metadata["powder_q_modulus_axis"] = True
    return PointData4D(
        H=columns_by_role.get("H", zeros),
        K=columns_by_role.get("K", zeros),
        L=columns_by_role.get("L", zeros),
        E=columns_by_role.get("E", zeros),
        intensity=intensity,
        sigma=sigma,
        mask=mask,
        temperature=temperature,
        metadata=metadata,
    )


def fit_dataset_inputs(
    group: DataGroup,
    *,
    purpose: str = "all",
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> tuple[list[FitDatasetInput], dict[str, FitDataBundle]]:
    """Prepare datasets for fitting, visualization, or both.

    Disabled datasets, datasets inside disabled groups, and non-composite
    datasets with zero fit weight are visualization-only. They are omitted
    entirely when ``purpose="fit"`` and selected exclusively when
    ``purpose="visualization"``. ``purpose="overlay"`` includes both fitted
    and visualization-only datasets. The default retains enabled datasets for
    callers that prepare ordinary data selections.
    """

    if purpose not in {"all", "fit", "visualization", "overlay"}:
        raise ValueError(
            "dataset input purpose must be 'all', 'fit', 'visualization', or 'overlay'"
        )

    inputs: list[FitDatasetInput] = []
    bundles: dict[str, FitDataBundle] = {}
    entries = _effective_dataset_entries(
        group,
        group,
        use_composite=True,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
        include_disabled_groups=purpose in {"visualization", "overlay"},
    )
    if purpose in {"visualization", "overlay"}:
        entries = _entries_with_visualization_binnings(
            group,
            entries,
            force_rebin=force_rebin,
            progress_callback=progress_callback,
        )
    for dataset in entries:
        disabled = not _dataset_is_effectively_enabled(group, dataset)
        if disabled and purpose not in {"visualization", "overlay"}:
            continue
        is_composite = bool(dataset.metadata.get("composite"))
        fit_weight = 1.0 if is_composite else float(dataset.fit_weight)
        if not np.isfinite(fit_weight) or fit_weight < 0.0:
            raise ValueError(f"dataset {dataset.name!r} fit weight must be finite and non-negative")
        visualization_only = disabled or (not is_composite and fit_weight == 0.0)
        if purpose == "fit" and visualization_only:
            continue
        if purpose == "visualization" and not visualization_only:
            continue
        bundle = fit_data_bundle(
            group,
            dataset,
            force_rebin=force_rebin,
            force_masks=force_masks,
            progress_callback=progress_callback,
        )
        if bundle is None:
            continue
        inputs.append(
            FitDatasetInput(
                name=dataset.name,
                data=bundle.points,
                weight=0.0 if visualization_only else fit_weight,
                data_type=dataset.data_type or DEFAULT_DATA_TYPE,
                scale_value=(1.0 if is_composite else float(dataset.scale_factor)),
                scale_vary=(False if is_composite else bool(dataset.scale_factor_vary)),
                scale_group=(
                    None if is_composite else dataset.scale_factor_group
                ),
            )
        )
        bundles[dataset.name] = bundle
    return inputs, bundles


def certify_group_lindhard_sampling(
    group: DataGroup,
    component: ModelComponentSpec,
    *,
    progress_callback: Any | None = None,
) -> Any:
    """Certify one Lindhard mesh against its complete fitted observables.

    Dataset import, masks, rebinning, temperatures, fields, and normalization
    follow the same preparation path as an ordinary fit.  The lower-level
    pipeline certificate then evaluates the selected tight-binding--Lindhard
    dependency closure and every enabled RPA consumer on deterministic points
    from each applicable positive-weight dataset.
    """

    if component.type != "lindhard":
        raise TypeError("group response sampling requires a Lindhard component")
    if group.models.get(component.name) is not component:
        raise ValueError("the selected Lindhard component is not in the data group")
    inputs, _bundles = fit_dataset_inputs(group, purpose="fit")
    if not inputs:
        raise ValueError(
            "the data group has no enabled positive-weight dataset to certify"
        )
    from .model_plots import certify_lindhard_pipeline_sampling

    return certify_lindhard_pipeline_sampling(
        component,
        group.models,
        inputs,
        progress_callback=progress_callback,
    )


_OPTIMIZER_KWARG_NAMES = (
    "max_nfev",
    "xtol",
    "ftol",
    "gtol",
    "loss",
    "f_scale",
)


def _optimizer_kwargs(optimizer_config: dict[str, Any] | None) -> dict[str, Any]:
    config = optimizer_config if isinstance(optimizer_config, dict) else {}
    kwargs: dict[str, Any] = {"finite_difference_workers": -1}
    for name in _OPTIMIZER_KWARG_NAMES:
        value = config.get(name)
        if value in (None, ""):
            continue
        if name == "max_nfev":
            kwargs[name] = int(value)
        elif name == "loss":
            kwargs[name] = str(value)
        else:
            kwargs[name] = float(value)
    init_config = config.get("initialization")
    if isinstance(init_config, dict) and init_config.get("enabled", False):
        legacy_workers = init_config.get("workers", 1)
        kwargs["initialization"] = {
            key: value
            for key, value in init_config.items()
            if key not in {"enabled", "workers"}
        }
        kwargs["initialization"].setdefault("method", "differential_evolution")
        # Preserve differential evolution's deferred-update algorithm when an
        # older project explicitly requested parallel evaluation, but resolve
        # its worker count from the central CPU policy at execution time.
        try:
            legacy_parallel = int(legacy_workers or 1) != 1
        except (TypeError, ValueError):
            legacy_parallel = False
        uses_deferred_updates = (
            str(kwargs["initialization"].get("updating", "")).lower() == "deferred"
        )
        if legacy_parallel or uses_deferred_updates:
            kwargs["initialization"]["workers"] = -1
        if legacy_parallel:
            kwargs["initialization"].setdefault("updating", "deferred")
    return kwargs


def _covariance_mode(optimizer_config: dict[str, Any] | None) -> str:
    if not isinstance(optimizer_config, dict):
        return "absolute"
    mode = str(optimizer_config.get("covariance_mode", "absolute"))
    if mode not in {"absolute", "residual"}:
        raise ValueError("covariance_mode must be 'absolute' or 'residual'")
    return mode


def _sampler_config(optimizer_config: dict[str, Any] | None) -> SamplerConfig | None:
    if not isinstance(optimizer_config, dict):
        return None
    sampler = optimizer_config.get("sampler")
    if not isinstance(sampler, dict) or not sampler.get("enabled", False):
        return None
    sampler_kwargs = (
        dict(sampler.get("kwargs", {}))
        if isinstance(sampler.get("kwargs"), dict)
        else {}
    )
    sampler_kwargs.pop("workers", None)
    sampler_kwargs.pop("parallel_workers", None)
    # GUI recipes use the central CPU ceiling.  ``-1`` is the public fitting
    # API's automatic request; omitting it would select its serial default.
    sampler_kwargs["workers"] = -1
    return SamplerConfig(
        method=str(sampler.get("method", "emcee")),
        n_walkers=_optional_int(sampler.get("n_walkers")),
        n_steps=_optional_int(sampler.get("n_steps")) or 1000,
        burn_in=int(sampler.get("burn_in", 0) or 0),
        thin=max(1, int(sampler.get("thin", 1) or 1)),
        random_seed=_optional_int(sampler.get("random_seed")),
        kwargs=sampler_kwargs,
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


_ISAW_TO_MANTID_COORDINATES = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float
)


def read_isaw_ub(path: str | Path) -> tuple[np.ndarray, dict[str, float]]:
    """Read an ISAW ``.mat`` file into Mantid/SNS instrument coordinates."""

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values = line.split()
        if values:
            rows.append([float(value) for value in values])
    if len(rows) < 4 or any(len(row) < 3 for row in rows[:3]) or len(rows[3]) < 6:
        raise ValueError("ISAW UB files require three matrix rows and one six-value lattice row")
    ub_ipns = np.asarray([row[:3] for row in rows[:3]], dtype=float).T
    ub = _ISAW_TO_MANTID_COORDINATES @ ub_ipns
    lattice = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), rows[3][:6], strict=True))
    if not np.all(np.isfinite(ub)) or abs(np.linalg.det(ub)) < 1e-14:
        raise ValueError("ISAW UB matrix must be finite and invertible")
    return ub, lattice


def write_isaw_ub(path: str | Path, ub: Any, lattice: dict[str, Any]) -> None:
    """Write a Mantid/SNS-frame UB in transposed IPNS/ISAW convention."""

    matrix = np.asarray(ub, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("UB must be a finite 3 x 3 matrix")
    names = ("a", "b", "c", "alpha", "beta", "gamma")
    values = [float(lattice[name]) for name in names]
    matrix_ipns = _ISAW_TO_MANTID_COORDINATES.T @ matrix
    lines = [" ".join(f"{value: .8f}" for value in row) for row in matrix_ipns.T]
    alpha, beta, gamma = np.deg2rad(values[3:6])
    volume = values[0] * values[1] * values[2] * np.sqrt(
        max(1.0 + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2, 0.0)
    )
    lines.append(" ".join(f"{value:11.4f}" for value in [*values, volume]))
    lines.append(" ".join(f"{0.0:11.4f}" for _ in range(7)))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ub_from_lattice_orientation(lattice: dict[str, Any], u: Any, v: Any) -> np.ndarray:
    """Build a Mantid/SNS-frame UB with ``u`` along the incident beam."""

    basis = reciprocal_basis_from_lattice_parameters(
        *(float(lattice[name]) for name in ("a", "b", "c", "alpha", "beta", "gamma"))
    ) / (2.0 * np.pi)
    q_u = basis @ np.asarray(u, dtype=float)
    q_v = basis @ np.asarray(v, dtype=float)
    if np.linalg.norm(q_u) <= 1e-14 or np.linalg.norm(np.cross(q_u, q_v)) <= 1e-14:
        raise ValueError("u must be nonzero and u/v must define a plane")
    x_axis = q_u / np.linalg.norm(q_u)
    z_axis = np.cross(q_u, q_v)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    ub_ipns = np.vstack((x_axis, y_axis, z_axis)) @ basis
    return _ISAW_TO_MANTID_COORDINATES @ ub_ipns


class UBSetupDialog:
    """Reusable lattice/orientation editor for single-crystal data scopes."""

    def __init__(self, parent: Any, *, ub: Any, lattice: dict[str, Any] | None, u: Any, v: Any):
        from PySide6 import QtWidgets

        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("UB setup")
        self.dialog.setMinimumWidth(620)
        outer = QtWidgets.QVBoxLayout(self.dialog)
        form = QtWidgets.QGridLayout()
        self.lattice_edits = {}
        defaults = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
        values = {**defaults, **(lattice or {})}
        for index, name in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            row, column = divmod(index, 3)
            form.addWidget(QtWidgets.QLabel(name), row, column * 2)
            edit = QtWidgets.QLineEdit(_format_number(values[name]))
            edit.setObjectName(f"ub_lattice_{name}")
            edit.setToolTip("Unit-cell length in angstrom." if index < 3 else "Unit-cell angle in degrees.")
            form.addWidget(edit, row, column * 2 + 1)
            self.lattice_edits[name] = edit
        self.orientation_edits = {}
        for vector_row, (name, vector) in enumerate((("u", u), ("v", v)), start=2):
            for component, value in enumerate(np.asarray(vector, dtype=float).reshape(3)):
                label = f"{name}{'xyz'[component]}"
                form.addWidget(QtWidgets.QLabel(label), vector_row, component * 2)
                edit = QtWidgets.QLineEdit(_format_number(value))
                edit.setObjectName(f"ub_orientation_{name}_{component}")
                edit.setToolTip("Reciprocal-lattice orientation vector. u points along the incident beam (+z); u and v define the horizontal plane, with +y vertical.")
                form.addWidget(edit, vector_row, component * 2 + 1)
                self.orientation_edits[(name, component)] = edit
        outer.addLayout(form)
        source_row = QtWidgets.QHBoxLayout()
        for text, slot in (("UB from NeXus", self._load_nexus), ("UB from ISAW", self._load_isaw), ("Calculate from lattice and u/v", self._calculate)):
            button = QtWidgets.QPushButton(text)
            button.setToolTip("Load orientation metadata directly from a NeXus file." if "NeXus" in text else "Read or calculate the UB matrix using the displayed convention.")
            button.clicked.connect(slot)
            source_row.addWidget(button)
        outer.addLayout(source_row)
        outer.addWidget(QtWidgets.QLabel("UB matrix (maps column [h,k,l] to Q' in inverse angstrom)"))
        self.matrix_table = QtWidgets.QTableWidget(3, 3)
        self.matrix_table.setObjectName("ub_matrix_table")
        self.matrix_table.horizontalHeader().setVisible(False)
        self.matrix_table.verticalHeader().setVisible(False)
        self.matrix_table.setFixedHeight(132)
        outer.addWidget(self.matrix_table)
        self._set_matrix(np.asarray(ub, dtype=float))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Apply | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        save = buttons.addButton("Save ISAW", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        save.setToolTip("Save a .mat file with UB transposed and converted to the IPNS x-beam/z-vertical convention.")
        save.clicked.connect(self._save_isaw)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._accept)
        buttons.rejected.connect(self.dialog.reject)
        outer.addWidget(buttons)
        self.result = None

    def exec(self):
        return self.dialog.exec()

    def _lattice(self):
        return {name: float(edit.text()) for name, edit in self.lattice_edits.items()}

    def _vector(self, name):
        return np.asarray([float(self.orientation_edits[(name, index)].text()) for index in range(3)])

    def _matrix(self):
        return np.asarray([[float(self.matrix_table.item(row, column).text()) for column in range(3)] for row in range(3)])

    def _set_matrix(self, matrix):
        from PySide6 import QtWidgets
        for row in range(3):
            for column in range(3):
                self.matrix_table.setItem(row, column, QtWidgets.QTableWidgetItem(f"{matrix[row, column]:.8g}"))

    def _calculate(self):
        from PySide6 import QtWidgets
        try:
            self._set_matrix(ub_from_lattice_orientation(self._lattice(), self._vector("u"), self._vector("v")))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"Could not calculate UB:\n{exc}")

    def _load_isaw(self):
        from PySide6 import QtWidgets
        path, _ = get_open_file_name(self.dialog, "Load ISAW UB", "", "ISAW matrix (*.mat);;All files (*)")
        if not path:
            return
        try:
            matrix, lattice = read_isaw_ub(path)
            self._set_matrix(matrix)
            for name, value in lattice.items():
                self.lattice_edits[name].setText(_format_number(value))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"Could not read ISAW file:\n{exc}")

    def _load_nexus(self):
        from PySide6 import QtWidgets
        path, _ = get_open_file_name(self.dialog, "Load UB from NeXus", "", "NeXus/HDF5 (*.nxs *.h5 *.hdf5);;All files (*)")
        if not path:
            return
        try:
            try:
                info = inspect_mdevent_workspace(path)
                matrix = info.ub_matrix
                lattice = info.lattice_parameters
            except Exception as mdevent_error:
                imported = load_mantid_mdhisto_nxs(path, copy_metadata=True)
                matrix = _dataset_ub_for_editor(imported.metadata)
                lattice = imported.metadata.get("lattice_parameters")
                if matrix is None:
                    raise ValueError(
                        "NeXus file does not contain a UB/orientation matrix"
                    ) from mdevent_error
            self._set_matrix(np.asarray(matrix, dtype=float))
            for name, value in (lattice or {}).items():
                if name not in self.lattice_edits:
                    continue
                self.lattice_edits[name].setText(_format_number(value))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"Could not read NeXus orientation:\n{exc}")

    def _save_isaw(self):
        from PySide6 import QtWidgets
        path, _ = get_save_file_name(self.dialog, "Save ISAW UB", "UB.mat", "ISAW matrix (*.mat)")
        if path:
            try:
                write_isaw_ub(path, self._matrix(), self._lattice())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"Could not save ISAW file:\n{exc}")

    def _accept(self):
        from PySide6 import QtWidgets
        try:
            matrix = self._matrix()
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-14:
                raise ValueError("UB matrix must be finite and invertible")
            self.result = {"ub_matrix": matrix.tolist(), "lattice_parameters": self._lattice(), "u": self._vector("u").tolist(), "v": self._vector("v").tolist()}
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"Invalid UB setup:\n{exc}")
            return
        self.dialog.accept()


def perform_group_fit(
    group: DataGroup,
    *,
    optimizer_config: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Fit the group's enabled model components to its enabled datasets.

    Masks, active/inactive datasets, fit weights, and scale factors are all
    honored because the fit points come from the same prepared views the data
    viewer shows. Optimized parameter values are written back to the model
    components, and per-dataset fit/residual channels are evaluated once over
    each dataset's full view so they can be stored with the fit result.
    """

    components = [
        model for model in group.models.values() if isinstance(model, ModelComponentSpec)
    ]
    if not any(component.enabled for component in components):
        raise ValueError("the data group has no enabled model components")
    progress_events: list[dict[str, Any]] = []

    def record_progress(event: dict[str, Any]) -> None:
        if len(progress_events) < 200:
            progress_events.append(_progress_event_summary(event))
        if progress_callback is not None:
            progress_callback(event)

    inputs, bundles = fit_dataset_inputs(
        group,
        purpose="fit",
        progress_callback=record_progress,
    )
    if not inputs:
        raise ValueError("no enabled positive-weight dataset could be prepared for fitting")

    compiled = compile_fit_problem(components, inputs, description=group.name)
    config = OptimizationConfig(
        covariance_mode=_covariance_mode(optimizer_config),
        kwargs=_optimizer_kwargs(optimizer_config),
    )

    result = fit_problem_least_squares(
        compiled.problem,
        config=config,
        progress_callback=record_progress,
    )
    sampler_result: SamplingResult | None = None
    sampler_cancelled = False
    sampler = None if result.cancelled else _sampler_config(optimizer_config)
    if sampler is not None:
        try:
            sampler_result = sample_problem_parameters(
                compiled.problem,
                sampler,
                initial_params=result.params,
                require_positive_sigma=config.require_positive_sigma,
                progress_callback=record_progress,
            )
        except SamplingCancelled as exc:
            sampler_result = exc.result
            sampler_cancelled = True
    _write_back_fitted_parameters(group, components, compiled, result)
    channels = _fit_channels_from_result(compiled, result, bundles)
    visualization_error = None
    visualization_names: list[str] = []
    try:
        visualization_channels = _visualization_only_channels(
            group,
            components,
            progress_callback=None if result.cancelled else record_progress,
        )
        visualization_names = list(visualization_channels)
        channels.update(visualization_channels)
    except Exception as exc:
        # A plotting-only dataset must never turn a completed optimization into
        # a failed fit. Preserve the fit and record the post-fit failure.
        visualization_error = str(exc)
    goodness: dict[str, Any] = {
        "status": (
            "cancelled"
            if result.cancelled or sampler_cancelled
            else ("converged" if result.success else "not converged")
        ),
        "message": (
            "emcee posterior sampling cancelled; partial posterior samples were saved."
            if sampler_cancelled
            else result.message
        ),
        "chi2": float(result.chi2),
        "reduced_chi2": float(result.reduced_chi2),
        "n_points": int(sum(result.dataset_sizes.values())),
        "n_variables": len(result.variable_names),
        "degrees_of_freedom": int(
            sum(result.dataset_sizes.values()) - len(result.variable_names)
        ),
        "parameters": {name: float(value) for name, value in result.params.items()},
        "stderr": (
            {name: float(value) for name, value in result.stderr.items()}
            if result.stderr
            else {}
        ),
        "covariance": _matrix_summary(result.covariance, result.variable_names),
        "covariance_mode": result.covariance_mode,
        "covariance_scale_factor": float(result.covariance_scale_factor),
        "dataset_chi2": {name: float(value) for name, value in result.dataset_chi2.items()},
        "dataset_reduced_chi2": {
            name: float(value) for name, value in result.dataset_reduced_chi2.items()
        },
        "dataset_n_points": {
            name: int(size) for name, size in result.dataset_sizes.items()
        },
        "skipped_datasets": list(compiled.skipped_datasets),
    }
    limit_hits = _fit_parameter_limit_hits(compiled.problem.parameter_specs, result.params)
    if limit_hits:
        goodness["parameters_at_limits"] = limit_hits
    metadata: dict[str, Any] = {}
    if visualization_names:
        metadata["visualization_only_datasets"] = visualization_names
    if visualization_error:
        metadata["visualization_only_error"] = visualization_error
    parameter_labels = _fit_parameter_labels_from_components(components, compiled, result.variable_names)
    if parameter_labels:
        metadata["parameter_labels"] = parameter_labels
    diagnostics = _fit_diagnostics_from_result(compiled, result, bundles, components)
    if diagnostics:
        metadata["diagnostics"] = diagnostics
    if progress_events:
        metadata["progress_log"] = progress_events
    if sampler_result is not None:
        posterior = _posterior_summary(sampler_result)
        goodness["posterior"] = posterior
        metadata["posterior_samples"] = _sampling_result_to_dict(sampler_result)
    return {
        "result": result,
        "compiled": compiled,
        "goodness": goodness,
        "channels": channels,
        "metadata": metadata,
    }


def _fit_parameter_limit_hits(specs: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return optimized parameters that finish at or effectively pin a bound."""

    hits: list[dict[str, Any]] = []
    for spec in specs:
        if not bool(getattr(spec, "vary", False)):
            continue
        name = str(getattr(spec, "name", ""))
        if name not in params:
            continue
        try:
            value = float(params[name])
        except (TypeError, ValueError):
            continue
        lower = getattr(spec, "min", None)
        upper = getattr(spec, "max", None)
        for side, bound, other_bound in (
            ("lower", lower, upper),
            ("upper", upper, lower),
        ):
            if bound is None:
                continue
            try:
                bound_value = float(bound)
            except (TypeError, ValueError):
                continue
            if np.isfinite(bound_value) and _parameter_is_at_bound(
                value, bound_value, other_bound
            ):
                hits.append({"name": name, "side": side, "bound": bound_value})
                break
    return hits




def _fit_limit_hits_from_goodness(goodness: Any) -> dict[str, dict[str, Any]]:
    """Index saved fit-bound warnings by parameter name for GUI rendering."""

    if not isinstance(goodness, dict):
        return {}
    raw = goodness.get("parameters_at_limits")
    if not isinstance(raw, list):
        return {}
    return {
        str(hit["name"]): dict(hit)
        for hit in raw
        if isinstance(hit, dict) and isinstance(hit.get("name"), str)
    }


def _fit_limit_warning_text(limit_hits: Any) -> str:
    hits = _fit_limit_hits_from_goodness({"parameters_at_limits": limit_hits})
    if not hits:
        return ""
    descriptions = [f"{name} ({hit.get('side', 'bound')})" for name, hit in hits.items()]
    return "Warning: fit parameter limit reached: " + ", ".join(descriptions) + "."


# Overlay recomputation is dominated by rebuilding the fit bundles (rebinning
# the full volume) and the RPA phase geometry. Neither depends on model
# *parameter values*, so both are cached per group and reused when only a
# parameter changes -- turning an O(10 s) rebuild on every edit into an O(0.1 s)
# re-evaluation. The cache is keyed on a structural signature that excludes
# parameter values (see _overlay_cache_signature).
_MODEL_OVERLAY_CACHE: OrderedDict[int, dict[str, Any]] = OrderedDict()
_MODEL_OVERLAY_CACHE_LIMIT = 6
_MODEL_OVERLAY_CACHE_MAX_BYTES = 256 * 1024**2
_MODEL_OVERLAY_ERRORS: OrderedDict[int, dict[str, str]] = OrderedDict()


def _overlay_cache_signature(group: DataGroup) -> str:
    """Structural fingerprint of a group's overlay inputs, excluding values.

    Everything that changes the bundles (data identity, masks, rebin,
    temperature, scale) or the model structure (types, config, sharing,
    applies-to, enablement) is included; per-parameter *values* are excluded so
    editing e.g. ``J2`` reuses the cached bundles and geometry.
    """

    datasets: list[Any] = []
    for dataset in group.iter_datasets():
        data_metadata = getattr(dataset.data, "metadata", {})
        data_rebin = data_metadata.get("rebin") if isinstance(data_metadata, dict) else None
        datasets.append(
            [
                dataset.name,
                bool(dataset.enabled),
                dataset.data_type,
                dataset.kind,
                dataset.id,
                dataset.data_cache_token,
                float(dataset.fit_weight),
                float(dataset.scale_factor),
                bool(dataset.scale_factor_vary),
                dataset.scale_factor_group,
                json.dumps(dataset.parameters, sort_keys=True, default=str),
                # Rebin settings and the output basis stored on a materialized
                # histogram both change physical HKLE coordinates. Either one
                # must rebuild cached fit points and model geometry.
                json.dumps(dataset.parameters.get(DATASET_REBIN_KEY), sort_keys=True, default=str),
                json.dumps(data_rebin, sort_keys=True, default=str),
                effective_dataset_temperature(group, dataset),
                [
                    [mask.type, bool(mask.enabled), bool(mask.invert), bool(mask.additive),
                     json.dumps(mask.parameters, sort_keys=True, default=str)]
                    for mask in effective_dataset_masks(group, dataset)
                ],
            ]
        )
    models: list[Any] = []
    for model in group.models.values():
        if not isinstance(model, ModelComponentSpec):
            continue
        models.append(
            [
                model.name,
                model.type,
                bool(model.enabled),
                list(model.applies_to) if model.applies_to is not None else None,
                json.dumps(model.config, sort_keys=True, default=str),
                json.dumps(model.sharing, sort_keys=True, default=str),
                json.dumps(model.constraints, sort_keys=True, default=str),
            ]
        )
    payload = [
        datasets,
        [
            [path, bool(subgroup.enabled)]
            for path, subgroup in _dataset_group_paths(group)
        ],
        models,
        json.dumps(group.lattice_parameters, sort_keys=True, default=str),
        group.spacegroup,
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def _overlay_current_params(
    group: DataGroup, compiled: CompiledFitProblem
) -> dict[str, float]:
    """Map current component parameter values onto a compiled problem's specs.

    The compiled problem may be cached (its spec values are stale), so read the
    live values from the components through the instance bookkeeping. Derived
    (constraint) parameters keep their compiled default.
    """

    params = {spec.name: float(spec.value) for spec in compiled.problem.parameter_specs}
    for instance in compiled.parameter_instances.values():
        if instance.component == "dataset" and instance.parameter == "scale_factor":
            if not instance.datasets:
                continue
            try:
                dataset = group.get_dataset(instance.datasets[0])
            except KeyError:
                continue
            params[instance.name] = float(dataset.scale_factor)
            continue
        component = group.models.get(instance.component)
        if not isinstance(component, ModelComponentSpec):
            continue
        fitted_values = component.metadata.get("fitted_values", {})
        per_scope = (
            fitted_values.get(instance.parameter, {})
            if isinstance(fitted_values, dict)
            else {}
        )
        value = (
            per_scope.get(instance.scope)
            if instance.scope != "global" and isinstance(per_scope, dict)
            else component.parameters.get(instance.parameter)
        )
        if (
            value is None
            and instance.scope != "global"
            and isinstance(instance.scope, str)
            and " · " in instance.scope
            and isinstance(per_scope, dict)
        ):
            value = per_scope.get(instance.scope.split(" · ", 1)[0])
        if value is None:
            value = component.parameters.get(instance.parameter)
        if value is None:
            continue
        try:
            params[instance.name] = float(value)
        except (TypeError, ValueError):
            continue
    return params


def current_model_channels(
    group: DataGroup,
    *,
    force_masks: bool = False,
    unmask_model: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate enabled model components at their current parameter values.

    When ``unmask_model`` is true, model values are extrapolated over every
    finite HKLE coordinate in the prepared dataset rather than only fit-valid
    data points. Residuals are likewise retained wherever the underlying data
    and uncertainty are finite.
    """

    errors: dict[str, str] = {}
    _MODEL_OVERLAY_ERRORS[id(group)] = errors
    _MODEL_OVERLAY_ERRORS.move_to_end(id(group))
    while len(_MODEL_OVERLAY_ERRORS) > _MODEL_OVERLAY_CACHE_LIMIT:
        _MODEL_OVERLAY_ERRORS.popitem(last=False)
    components = [
        model for model in group.models.values() if isinstance(model, ModelComponentSpec)
    ]
    if not any(component.enabled for component in components):
        _MODEL_OVERLAY_CACHE.pop(id(group), None)
        return {}
    signature = _overlay_cache_signature(group)
    cached = _MODEL_OVERLAY_CACHE.get(id(group))
    pending_masks = any(_dataset_mask_is_stale(dataset) for dataset in group.iter_datasets())
    if (
        cached is not None
        and cached["signature"] == signature
        and not (force_masks and pending_masks)
    ):
        _MODEL_OVERLAY_CACHE.move_to_end(id(group))
        compiled = cached["compiled"]
        bundles = cached["bundles"]
        subsets = cached["subsets"]
    else:
        inputs, bundles = fit_dataset_inputs(
            group,
            purpose="overlay",
            force_rebin=False,
            force_masks=force_masks,
        )
        if not inputs:
            _MODEL_OVERLAY_CACHE.pop(id(group), None)
            return {}
        try:
            overlay_components = _components_with_binning_aliases(components, inputs)
            compiled = compile_fit_problem(
                overlay_components, inputs, description=group.name
            )
        except Exception as exc:
            errors["fit problem"] = f"{type(exc).__name__}: {exc}"
            return {}
        subsets = {}
        # Lazy loading during fit-data preparation increments dataset revisions.
        signature = _overlay_cache_signature(group)
        _lru_store(
            _MODEL_OVERLAY_CACHE,
            id(group),
            {
                "signature": signature,
                "compiled": compiled,
                "bundles": bundles,
                "subsets": subsets,
            },
            _MODEL_OVERLAY_CACHE_LIMIT,
            _MODEL_OVERLAY_CACHE_MAX_BYTES,
        )
    try:
        params = _overlay_current_params(group, compiled)
    except Exception as exc:
        errors["parameters"] = f"{type(exc).__name__}: {exc}"
        return {}
    return _fit_channels_from_params(
        compiled,
        params,
        bundles,
        subset_cache=subsets,
        evaluate_masked=unmask_model,
        dataset_errors=errors,
    )


def evaluate_current_state_model(
    group: DataGroup,
    current: FitTimelineEntry | None = None,
) -> FitTimelineEntry:
    """Evaluate live model channels and record them on a Current state entry."""

    ensure_fit_history(group)
    if current is None or current.kind != "current":
        current = _top_level_current_state_entry(group)
    if current is None:
        source = _last_result_at_level(group.fits) or group.fits[0]
        current = current_state_fit_entry(group, source)
        group.fits.append(current)
    _MODEL_OVERLAY_ERRORS.pop(id(group), None)
    error = ""
    try:
        channels = current_model_channels(group)
    except Exception as exc:
        channels = {}
        error = str(exc)
    errors = dict(_MODEL_OVERLAY_ERRORS.get(id(group), {}))
    if error:
        errors.setdefault("fit problem", error)
    current.channels = channels
    current.metadata = dict(current.metadata)
    current.metadata["model_evaluation_status"] = (
        "partially failed"
        if channels and errors
        else "failed"
        if errors
        else "evaluated" if channels else "no compatible model prediction"
    )
    current.metadata["model_evaluation_datasets"] = sorted(channels)
    if errors:
        current.metadata["model_evaluation_errors"] = errors
        current.metadata["model_evaluation_error"] = "; ".join(
            f"{name}: {message}" for name, message in errors.items()
        )
    else:
        current.metadata.pop("model_evaluation_errors", None)
        current.metadata.pop("model_evaluation_error", None)
    _set_fit_current_snapshot(current, group)
    return current


def defer_current_state_model_evaluation(
    group: DataGroup,
    current: FitTimelineEntry | None = None,
) -> FitTimelineEntry:
    """Record a Current state without loading lazy datasets on the GUI thread."""

    ensure_fit_history(group)
    if current is None or current.kind != "current":
        current = _top_level_current_state_entry(group)
    if current is None:
        source = _last_result_at_level(group.fits) or group.fits[0]
        current = current_state_fit_entry(group, source)
        group.fits.append(current)
    current.channels = {}
    current.metadata = dict(current.metadata)
    current.metadata["model_evaluation_status"] = "deferred until data are viewed or fitted"
    current.metadata["model_evaluation_datasets"] = []
    current.metadata.pop("model_evaluation_errors", None)
    current.metadata.pop("model_evaluation_error", None)
    _set_fit_current_snapshot(current, group)
    return current


def _compiled_problem_for_fit_entry(
    group: DataGroup,
    fit_entry: FitTimelineEntry,
) -> CompiledFitProblem:
    """Compile the fitting problem from a stored fit snapshot without leaving state changed."""

    current_snapshot = snapshot_data_group_state(group)
    try:
        if fit_entry.snapshot:
            restore_data_group_state(group, fit_entry.snapshot)
        components = [
            model for model in group.models.values() if isinstance(model, ModelComponentSpec)
        ]
        inputs, _bundles = fit_dataset_inputs(group, purpose="fit")
        if not inputs:
            raise ValueError("no enabled dataset could be prepared for posterior sampling")
        return compile_fit_problem(components, inputs, description=group.name)
    finally:
        restore_data_group_state(group, current_snapshot)


def _matrix_summary(matrix: Any, names: list[str]) -> dict[str, Any]:
    if matrix is None:
        return {}
    arr = np.asarray(matrix, dtype=float)
    summary: dict[str, Any] = {"variables": list(names), "shape": list(arr.shape)}
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] == len(names):
        summary["matrix"] = arr.tolist()
        summary["diagonal"] = {name: float(arr[index, index]) for index, name in enumerate(names)}
        denom = np.sqrt(np.outer(np.diag(arr), np.diag(arr)))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
        summary["correlation"] = {
            name: {
                other: float(corr[i, j])
                for j, other in enumerate(names)
            }
            for i, name in enumerate(names)
        }
    return summary


def _progress_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "stage": str(event.get("stage", "fit")),
        "message": str(event.get("message", "")),
    }
    for key in ("iteration", "total", "cost", "convergence", "elapsed_seconds", "seconds_per_step"):
        if event.get(key) is not None:
            value = event[key]
            summary[key] = (
                float(value)
                if key in {"cost", "convergence", "elapsed_seconds", "seconds_per_step"}
                else int(value)
            )
    params = event.get("parameters")
    if isinstance(params, dict):
        summary["parameters"] = {name: float(value) for name, value in params.items()}
    return summary


def _posterior_summary(result: SamplingResult) -> dict[str, Any]:
    samples = np.asarray(result.samples, dtype=float)
    summary: dict[str, Any] = dict(result.metadata)
    if samples.size == 0:
        summary["samples"] = 0
        return summary
    summary["samples"] = int(samples.shape[0])
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    summary["parameters"] = {
        name: {
            "p16": float(percentiles[0, index]),
            "median": float(percentiles[1, index]),
            "p84": float(percentiles[2, index]),
        }
        for index, name in enumerate(result.variable_names)
    }
    if samples.shape[1] > 1:
        corr = np.corrcoef(samples, rowvar=False)
        summary["correlation"] = {
            name: {
                other: float(corr[i, j])
                for j, other in enumerate(result.variable_names)
            }
            for i, name in enumerate(result.variable_names)
        }
    return summary


def _sampling_result_to_dict(result: SamplingResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "variable_names": list(result.variable_names),
        "samples": _encode_float_array(result.samples),
        "metadata": dict(result.metadata),
    }
    if result.log_probability is not None:
        payload["log_probability"] = _encode_float_array(result.log_probability)
    if result.chain is not None:
        payload["chain"] = _encode_float_array(result.chain)
    if result.log_probability_chain is not None:
        payload["log_probability_chain"] = _encode_float_array(result.log_probability_chain)
    return payload


def _autocorrelation_progress_summary(result: SamplingResult | None) -> list[str]:
    """Return readable emcee convergence estimates for the progress log."""

    if result is None:
        return []
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    raw_tau = metadata.get("autocorrelation_time")
    if not isinstance(raw_tau, (list, tuple)):
        return []
    tau = []
    for value in raw_tau:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric > 0.0:
            tau.append(numeric)
    if not tau:
        return [
            "emcee integrated autocorrelation time is unavailable for this chain; "
            "run more steps before using it to estimate a chain length."
        ]
    recommended = metadata.get("autocorrelation_recommended_steps")
    additional = metadata.get("autocorrelation_recommended_additional_steps")
    tau_text = ", ".join(f"{value:.3g}" for value in tau)
    summary = f"emcee integrated autocorrelation time (steps): [{tau_text}]"
    if recommended is not None:
        summary += f"; recommended total chain length: at least {int(recommended):,} steps"
    if additional is not None:
        summary += f"; estimated additional steps: {int(additional):,}"
    return [summary + "."]


def _sampling_result_with_window(result: SamplingResult, burn_in: int, thin: int) -> SamplingResult:
    """Return a copy flattened with a different burn-in/thinning window."""

    if result.chain is None:
        raise ValueError("stored posterior does not include a raw emcee chain")
    chain = np.asarray(result.chain, dtype=float)
    if chain.ndim != 3 or chain.shape[0] == 0:
        raise ValueError("stored posterior chain is empty or malformed")
    burn = min(max(0, int(burn_in)), max(0, chain.shape[0] - 1))
    step = max(1, int(thin))
    sliced = chain[burn::step]
    samples = sliced.reshape((-1, chain.shape[2]))
    log_probability = None
    if result.log_probability_chain is not None:
        log_chain = np.asarray(result.log_probability_chain, dtype=float)
        if log_chain.shape[:2] == chain.shape[:2]:
            log_probability = log_chain[burn::step].reshape(-1)
    metadata = dict(result.metadata)
    metadata["burn_in"] = burn
    metadata["thin"] = step
    metadata["samples"] = int(samples.shape[0])
    metadata["n_steps"] = int(chain.shape[0])
    metadata["n_walkers"] = int(chain.shape[1])
    return SamplingResult(
        samples=samples,
        variable_names=list(result.variable_names),
        log_probability=log_probability,
        metadata=metadata,
        chain=chain,
        log_probability_chain=result.log_probability_chain,
    )


def _combined_sampling_result(
    original: SamplingResult,
    appended: SamplingResult,
    *,
    burn_in: int,
    thin: int,
) -> SamplingResult:
    """Concatenate two emcee chains and re-flatten them with the requested window."""

    if original.chain is None or appended.chain is None:
        raise ValueError("appending posterior samples requires raw emcee chains")
    if list(original.variable_names) != list(appended.variable_names):
        raise ValueError("appended posterior variables do not match the stored chain")
    chain = np.concatenate(
        [np.asarray(original.chain, dtype=float), np.asarray(appended.chain, dtype=float)],
        axis=0,
    )
    log_probability_chain = None
    if original.log_probability_chain is not None and appended.log_probability_chain is not None:
        log_probability_chain = np.concatenate(
            [
                np.asarray(original.log_probability_chain, dtype=float),
                np.asarray(appended.log_probability_chain, dtype=float),
            ],
            axis=0,
        )
    metadata = dict(original.metadata)
    metadata.update(dict(appended.metadata))
    metadata["n_steps"] = int(chain.shape[0])
    metadata["n_walkers"] = int(chain.shape[1])
    metadata["appended_steps"] = int(appended.chain.shape[0])
    result = SamplingResult(
        samples=np.empty((0, chain.shape[2]), dtype=float),
        variable_names=list(original.variable_names),
        log_probability=None,
        metadata=metadata,
        chain=chain,
        log_probability_chain=log_probability_chain,
    )
    return _sampling_result_with_window(result, burn_in=burn_in, thin=thin)


def _store_sampling_result_on_fit_entry(fit_entry: FitTimelineEntry, result: SamplingResult) -> None:
    fit_entry.metadata["posterior_samples"] = _sampling_result_to_dict(result)
    fit_entry.goodness["posterior"] = _posterior_summary(result)
    display = fit_entry.metadata.get(POSTERIOR_DISPLAY_KEY)
    if isinstance(display, dict) and display.get("use_best_sample"):
        best = _best_posterior_sample(result)
        if best is None:
            display["use_best_sample"] = False
            display.pop("best_sample", None)
        else:
            params, log_probability, location = best
            display["best_sample"] = {
                "parameters": params,
                "log_probability": log_probability,
                **location,
            }


def _posterior_interval_errors(
    fit_entry: FitTimelineEntry,
    name: str,
    center: float,
) -> tuple[float, float] | None:
    """Return asymmetric 68% posterior errors about a displayed parameter value."""

    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    rows = posterior.get("parameters") if isinstance(posterior.get("parameters"), dict) else {}
    row = rows.get(name) if isinstance(rows.get(name), dict) else None
    if row is None:
        return None
    try:
        lower = float(row["p16"])
        upper = float(row["p84"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return max(0.0, center - lower), max(0.0, upper - center)


def _best_posterior_sample(
    result: SamplingResult,
) -> tuple[dict[str, float], float, dict[str, int]] | None:
    """Return the stored sample with the highest finite emcee log probability."""

    names = list(result.variable_names)
    if not names:
        return None
    if result.chain is not None and result.log_probability_chain is not None:
        chain = np.asarray(result.chain, dtype=float)
        log_probability = np.asarray(result.log_probability_chain, dtype=float)
        if chain.ndim == 3 and log_probability.shape == chain.shape[:2]:
            finite = np.isfinite(log_probability)
            if np.any(finite):
                masked = np.where(finite, log_probability, -np.inf)
                flat_index = int(np.argmax(masked))
                step, walker = np.unravel_index(flat_index, log_probability.shape)
                sample = chain[int(step), int(walker)]
                if sample.shape[0] == len(names):
                    return (
                        {name: float(sample[index]) for index, name in enumerate(names)},
                        float(log_probability[int(step), int(walker)]),
                        {"step": int(step), "walker": int(walker)},
                    )
    if result.log_probability is None:
        return None
    samples = np.asarray(result.samples, dtype=float)
    log_probability = np.asarray(result.log_probability, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(names) or log_probability.shape != (samples.shape[0],):
        return None
    finite = np.isfinite(log_probability)
    if not np.any(finite):
        return None
    sample_index = int(np.argmax(np.where(finite, log_probability, -np.inf)))
    sample = samples[sample_index]
    return (
        {name: float(sample[index]) for index, name in enumerate(names)},
        float(log_probability[sample_index]),
        {"sample": int(sample_index)},
    )


def _log_probability_for_params(
    compiled: CompiledFitProblem,
    params: dict[str, float],
) -> float:
    """Evaluate the Gaussian log likelihood used by emcee for a parameter set."""

    trial = {spec.name: float(spec.value) for spec in compiled.problem.parameter_specs}
    trial.update({str(name): float(value) for name, value in params.items()})
    evaluation = _evaluate_problem(
        compiled.problem,
        trial,
        require_positive_sigma=True,
    )
    chi2 = float(np.dot(evaluation.residuals, evaluation.residuals))
    return -0.5 * chi2


def _best_posterior_promotion_candidate(
    fit_entry: FitTimelineEntry,
    compiled: CompiledFitProblem | None = None,
) -> tuple[dict[str, float], float, float, dict[str, int]] | None:
    """Return a best-sample candidate only when it improves on the fit result."""

    stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if stored is None:
        return None
    best = _best_posterior_sample(stored)
    if best is None:
        return None
    sample_params, sample_log_probability, location = best
    try:
        baseline_log_probability = -0.5 * float(fit_entry.goodness["chi2"])
    except (KeyError, TypeError, ValueError):
        goodness_params = fit_entry.goodness.get("parameters")
        if compiled is None or not isinstance(goodness_params, dict):
            return None
        baseline_log_probability = _log_probability_for_params(compiled, goodness_params)
    if not np.isfinite(sample_log_probability) or not np.isfinite(baseline_log_probability):
        return None
    if sample_log_probability <= baseline_log_probability:
        return None
    return sample_params, sample_log_probability, baseline_log_probability, location


def _fit_parameter_labels_from_components(
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    variable_names: list[str],
) -> dict[str, str]:
    by_component = {component.name: component for component in components}
    labels: dict[str, str] = {}
    for name in variable_names:
        instance = compiled.parameter_instances.get(name)
        if instance is None:
            continue
        component = by_component.get(instance.component)
        if component is None:
            continue
        parameter_labels = component.metadata.get("parameter_labels")
        if not isinstance(parameter_labels, dict):
            continue
        label = str(parameter_labels.get(instance.parameter, "")).strip()
        if label:
            labels[name] = label
    return labels


def _write_back_parameter_values(
    group: DataGroup,
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    params: dict[str, float],
) -> None:
    """Store fit-problem parameter values on their model components.

    Globally shared (and constrained) parameters update the component's
    ``parameters`` directly. Per-dataset and grouped instances are stored per
    tie key under ``metadata["fitted_values"]`` because a single parameter
    box cannot display several values.
    """

    for component in components:
        for parameter in component_parameter_names(component):
            qualified = qualified_parameter_name(component.name, parameter)
            if qualified in params:
                component.parameters[parameter] = float(params[qualified])
                continue
            values = {
                instance.scope: float(params[instance.name])
                for instance in compiled.instances_for(component.name, parameter)
                if instance.name in params
            }
            if values:
                component.metadata.setdefault("fitted_values", {})[parameter] = values


def _write_back_fitted_parameters(
    group: DataGroup,
    components: list[ModelComponentSpec],
    compiled: CompiledFitProblem,
    result: Any,
) -> None:
    """Store optimized values on their model components."""

    _write_back_parameter_values(group, components, compiled, result.params)
    _write_back_dataset_scale_factors(group, compiled, result.params)


def _apply_displayed_fit_parameters(
    group: DataGroup,
    fit_entry: FitTimelineEntry,
) -> None:
    """Apply a result's selected parameter values to its restored live state.

    The fit snapshot is the least-squares state.  When the user selects the
    best emcee sample, its values are written on top of that state without
    changing the saved snapshot, so unselecting it can restore the exact LM
    result.
    """

    components = [
        model for model in group.models.values() if isinstance(model, ModelComponentSpec)
    ]
    if not components:
        return
    inputs, _bundles = fit_dataset_inputs(group, purpose="fit")
    if not inputs:
        return
    compiled = compile_fit_problem(components, inputs, description=group.name)
    params = {
        spec.name: float(spec.value)
        for spec in compiled.problem.parameter_specs
    }
    params.update(_display_fit_parameters(fit_entry))
    resolved = compiled.problem.resolve_parameters(params)
    _write_back_parameter_values(group, components, compiled, resolved)
    _write_back_dataset_scale_factors(group, compiled, resolved)


def _write_back_dataset_scale_factors(
    group: DataGroup,
    compiled: CompiledFitProblem,
    params: dict[str, float],
) -> None:
    """Store optimized dataset scale factors on their datasets."""

    for instance in compiled.parameter_instances.values():
        if instance.component != "dataset" or instance.parameter != "scale_factor":
            continue
        if instance.name not in params:
            continue
        for dataset_name in instance.datasets:
            try:
                dataset = group.get_dataset(dataset_name)
            except KeyError:
                continue
            dataset.scale_factor = float(params[instance.name])


def _visualization_only_channels(
    group: DataGroup,
    components: list[ModelComponentSpec],
    *,
    progress_callback: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate disabled and zero-weight datasets using the fitted state."""

    inputs, bundles = fit_dataset_inputs(
        group,
        purpose="visualization",
        progress_callback=progress_callback,
    )
    if not inputs:
        return {}
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "visualization",
                "iteration": 0,
                "total": len(inputs),
                "message": "evaluating disabled and zero-weight datasets for display",
            }
        )
    components = _components_with_binning_aliases(components, inputs)
    compiled = compile_fit_problem(components, inputs, description=group.name)
    params = compiled.problem.resolve_parameters(_overlay_current_params(group, compiled))
    channels = _fit_channels_from_params(compiled, params, bundles)
    for payload in channels.values():
        payload["visualization_only"] = True
    return channels


def _components_with_binning_aliases(
    components: list[ModelComponentSpec], inputs: list[FitDatasetInput]
) -> list[ModelComponentSpec]:
    """Extend explicit applies-to lists to named views of the same dataset."""

    names = [item.name for item in inputs]
    if not any(" · " in name for name in names):
        return components
    result = copy.deepcopy(components)
    for component in result:
        aliases_by_source = {
            name: name.split(" · ", 1)[0]
            for name in names
            if " · " in name
        }
        if component.applies_to is not None:
            aliases = [
                alias
                for alias, source in aliases_by_source.items()
                if source in component.applies_to
            ]
            component.applies_to = list(
                dict.fromkeys([*component.applies_to, *aliases])
            )
        for sharing in component.sharing.values():
            if not isinstance(sharing, dict) or sharing.get("mode") != "grouped":
                continue
            groups = sharing.setdefault("groups", {})
            if not isinstance(groups, dict):
                continue
            for alias, source in aliases_by_source.items():
                groups[alias] = str(groups.get(source, source))
    return result


def _fit_channels_from_result(
    compiled: CompiledFitProblem,
    result: Any,
    bundles: dict[str, FitDataBundle],
) -> dict[str, dict[str, Any]]:
    """Evaluate fit and residual channels over each fitted dataset's view."""

    return _fit_channels_from_params(compiled, result.params, bundles)


def _fit_diagnostics_from_result(
    compiled: CompiledFitProblem,
    result: Any,
    bundles: dict[str, FitDataBundle],
    components: list[ModelComponentSpec],
) -> dict[str, dict[str, Any]]:
    """Collect registered per-dataset physics diagnostics after a fit.

    Returns ``{dataset_name: {metric: value}}`` (metrics prefixed by the
    component name when a dataset carries more than one diagnostic component).
    Stored in the fit result's metadata, which round-trips as JSON. A component
    that cannot be evaluated on a dataset is skipped.
    """

    diagnostic_by_name = {
        component.name: component
        for component in components
        if component.enabled
        and component.type in MODEL_TYPE_REGISTRY
        and (
            MODEL_TYPE_REGISTRY[component.type].diagnostics is not None
            or MODEL_TYPE_REGISTRY[component.type].context_diagnostics is not None
        )
    }
    if not diagnostic_by_name:
        return {}
    resolved = compiled.problem.resolve_parameters(result.params)
    fitted_names = {dataset.name for dataset in compiled.problem.datasets}
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, bundle in bundles.items():
        if name not in fitted_names:
            continue
        applicable = [
            diagnostic_by_name[component_name]
            for component_name in compiled.components_by_dataset.get(name, [])
            if component_name in diagnostic_by_name
        ]
        if not applicable:
            continue
        subset = _subset_points(bundle.points, bundle.points.valid_mask())
        if not subset.size:
            continue
        per_dataset: dict[str, Any] = {}
        for component in applicable:
            record = compute_component_diagnostics(
                component,
                subset,
                resolved,
                components={
                    item.name: item for item in components if item.enabled
                },
            )
            if not record:
                continue
            if len(applicable) == 1:
                per_dataset = copy.deepcopy(record)
            else:
                per_dataset.update(
                    {
                        f"{component.name}.{key}": copy.deepcopy(value)
                        for key, value in record.items()
                    }
                )
        if per_dataset:
            diagnostics[name] = per_dataset
    return diagnostics


def _fit_channels_from_params(
    compiled: CompiledFitProblem,
    params: dict[str, float],
    bundles: dict[str, FitDataBundle],
    subset_cache: dict[Any, tuple[np.ndarray, PointData4D]] | None = None,
    *,
    evaluate_masked: bool = False,
    dataset_errors: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate fit and residual channels for a compiled problem.

    ``subset_cache`` optionally memoizes each dataset's valid-point mask and
    subset across calls; reusing the same subset object also lets the model's
    per-dataset geometry cache hit, so repeated overlay refreshes at new
    parameter values skip both the mask scan and the RPA geometry rebuild.
    """

    channels: dict[str, dict[str, Any]] = {}
    fitted_names = {dataset.name for dataset in compiled.problem.datasets}
    for name, bundle in bundles.items():
        if name not in fitted_names:
            continue
        points = bundle.points
        # Normally evaluate only fit-valid data and scatter back with NaN
        # elsewhere. The viewer's explicit unmask option instead evaluates all
        # finite HKLE coordinates; keeping the paths separate avoids paying for
        # the often much larger masked volume during ordinary interaction.
        cache_key: Any = (name, "unmasked") if evaluate_masked else name
        cached_subset = subset_cache.get(cache_key) if subset_cache is not None else None
        if cached_subset is not None:
            keep, subset = cached_subset
        else:
            if evaluate_masked:
                keep = np.isfinite(points.H) & np.isfinite(points.K)
                keep &= np.isfinite(points.L) & np.isfinite(points.E)
                if isinstance(points.temperature, np.ndarray):
                    keep &= np.isfinite(points.temperature)
            else:
                keep = points.valid_mask()
            subset = _subset_points(points, keep)
            if subset_cache is not None:
                subset_cache[cache_key] = (keep, subset)
        fit_values = np.full(points.size, np.nan, dtype=float)
        if subset.size:
            try:
                fit_values[keep] = np.asarray(
                    evaluate_problem_model(
                        compiled.problem,
                        name,
                        params,
                        data=subset,
                    ),
                    dtype=float,
                )
            except Exception as exc:
                if dataset_errors is None:
                    raise
                dataset_errors[name] = f"{type(exc).__name__}: {exc}"
                continue
        fit_dataset = next(
            dataset for dataset in compiled.problem.datasets if dataset.name == name
        )
        intensity = np.asarray(points.intensity, dtype=float)
        sigma = np.asarray(points.sigma, dtype=float)
        if fit_dataset.data_scale_parameter:
            resolved = compiled.problem.resolve_parameters(params)
            scale = float(resolved[fit_dataset.data_scale_parameter])
            intensity = intensity * scale
            sigma = sigma * max(abs(scale), np.finfo(float).tiny)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual_values = (intensity - fit_values) / sigma
        residual_values = np.where(
            np.isfinite(intensity) & np.isfinite(sigma) & (sigma > 0.0),
            residual_values,
            np.nan,
        )
        if bundle.grid_shape is not None:
            fit_values = fit_values.reshape(bundle.grid_shape)
            residual_values = residual_values.reshape(bundle.grid_shape)
        channels[name] = {
            "kind": "grid" if bundle.grid_shape is not None else "points",
            "fit": fit_values,
            "residual": residual_values,
            "fit_channel": str(points.metadata.get("fit_channel", "")),
        }
    return channels


def _subset_points(points: PointData4D, keep: np.ndarray) -> PointData4D:
    """Return the ``keep``-selected subset of ``points`` (order preserved)."""

    if isinstance(points.temperature, np.ndarray):
        temperature: Any = points.temperature[keep]
    else:
        temperature = points.temperature
    field = points.magnetic_field
    if isinstance(field, np.ndarray) and field.ndim == 2:
        magnetic_field: Any = field[keep]
    elif field is None:
        magnetic_field = None
    else:
        magnetic_field = np.array(field)
    return PointData4D(
        H=points.H[keep],
        K=points.K[keep],
        L=points.L[keep],
        E=points.E[keep],
        intensity=points.intensity[keep],
        sigma=points.sigma[keep],
        mask=np.ones(int(np.count_nonzero(keep)), dtype=bool),
        temperature=temperature,
        magnetic_field=magnetic_field,
        metadata=dict(points.metadata),
    )


def latest_fit_channels(group: DataGroup, dataset_name: str) -> dict[str, Any] | None:
    """Return the stored fit channels for a dataset from the newest fit result."""

    latest: dict[str, Any] | None = None
    for entry in _walk_fit_entries(group.fits):
        if entry.kind != "result":
            continue
        payload = entry.channels.get(dataset_name)
        if payload:
            latest = payload
    return latest


def _group_has_fit_channels(group: DataGroup) -> bool:
    """Return whether any dataset in the group has stored fit channels."""

    return any(entry.channels for entry in _walk_fit_entries(group.fits) if entry.kind == "result")


def _group_has_enabled_model_components(group: DataGroup) -> bool:
    """Return whether the group has a model that can be evaluated."""

    return any(
        isinstance(model, ModelComponentSpec) and model.enabled
        for model in group.models.values()
    )


def attach_fit_channels_to_view(
    group: DataGroup,
    dataset_name: str,
    view: MDHistoData | PointListData,
    *,
    fallback_payload: dict[str, Any] | None = None,
) -> MDHistoData | PointListData:
    """Return a view with live-model or saved fit/residual channels attached.

    Live current-model channels take precedence when available. Channels are
    only attached when their shape still matches the current view, so stale fits
    after mask or rebin changes are silently skipped rather than misaligned.
    """

    payload = fallback_payload or latest_fit_channels(group, dataset_name)
    if payload is None:
        return view
    arrays: dict[str, np.ndarray] = {}
    for channel_name in FIT_CHANNEL_NAMES:
        decoded = _fit_channel_array(payload.get(channel_name))
        if decoded is None:
            return view
        arrays[channel_name] = decoded
    if isinstance(view, MDHistoData):
        if any(array.shape != view.shape for array in arrays.values()):
            return view
        metadata = dict(view.metadata)
        metadata.update(arrays)
        return view.with_updates(metadata=metadata)
    if isinstance(view, PointListData):
        if any(array.shape != (view.size,) for array in arrays.values()):
            return view
        columns = dict(view.columns)
        channels = [dict(channel) for channel in view.channels]
        metadata = dict(view.metadata)
        existing = set(view.channel_labels)
        fit_channel = _point_fit_channel_label(group, dataset_name, view, payload)
        for channel_name, array in arrays.items():
            columns[channel_name] = array
            if channel_name not in existing:
                channels.append(
                    {"label": channel_name, "value": channel_name, "error": None}
                )
        for channel_name in FIT_CHANNEL_NAMES:
            key = f"viewer_{channel_name}_channel_map"
            stored = metadata.get(key, {})
            mapping = dict(stored) if isinstance(stored, dict) else {}
            mapping = {
                label: target
                for label, target in mapping.items()
                if target != channel_name
            }
            if fit_channel:
                mapping[fit_channel] = channel_name
            if mapping:
                metadata[key] = mapping
            else:
                metadata.pop(key, None)
        return view.with_updates(
            columns=columns,
            channels=channels,
            metadata=metadata,
        )
    return view


def _point_fit_channel_label(
    group: DataGroup,
    dataset_name: str,
    view: PointListData,
    payload: dict[str, Any],
) -> str:
    """Return the explicitly recorded data channel for a point-list fit."""

    explicit = str(payload.get("fit_channel", "")).strip()
    return explicit if explicit in view.channel_labels else ""


def next_data_group_name(groups: list[DataGroup]) -> str:
    """Return the first available WorkspaceN name for a top-level workspace."""

    taken = {group.name for group in groups}
    index = 1
    while f"Workspace{index}" in taken:
        index += 1
    return f"Workspace{index}"


def next_mask_name(masks: list[MaskSpec]) -> str:
    """Return the first available MaskN name."""

    taken = {mask.name for mask in masks}
    index = 1
    while f"Mask{index}" in taken:
        index += 1
    return f"Mask{index}"


def next_model_name(models: dict[str, Any]) -> str:
    """Return the first available ModelN name."""

    taken = set(models)
    index = 1
    while f"Model{index}" in taken:
        index += 1
    return f"Model{index}"


def next_fit_result_name(fits: list[FitTimelineEntry]) -> str:
    """Return the first available Fit ResultN name across a fit tree."""

    return _next_fit_tree_name(fits, "Fit Result")


def next_fit_timeline_name(fits: list[FitTimelineEntry]) -> str:
    """Return the first available TimelineN name across a fit tree."""

    return _next_fit_tree_name(fits, "Timeline")


def _next_fit_tree_name(fits: list[FitTimelineEntry], prefix: str) -> str:
    taken = {entry.name for entry in _walk_fit_entries(fits)}
    index = 1
    while f"{prefix}{index}" in taken:
        index += 1
    return f"{prefix}{index}"


def _walk_fit_entries(fits: list[FitTimelineEntry]):
    for entry in fits:
        yield entry
        yield from _walk_fit_entries(entry.children)


def _fit_entry_in_tree(fits: list[FitTimelineEntry], target: FitTimelineEntry | None) -> bool:
    if target is None:
        return False
    return any(entry is target for entry in _walk_fit_entries(fits))


def _fit_entry_path(
    fits: list[FitTimelineEntry],
    target: FitTimelineEntry | None,
) -> list[int] | None:
    """Return the list of child indices from the fit root to ``target``.

    Names are not unique across branches (every branch has a "Current state"),
    so a positional path is the stable identifier for persisting a selection.
    """

    if target is None:
        return None
    for index, entry in enumerate(fits):
        if entry is target:
            return [index]
        below = _fit_entry_path(entry.children, target)
        if below is not None:
            return [index, *below]
    return None


def _fit_entry_at_path(
    fits: list[FitTimelineEntry],
    path: list[int] | None,
) -> FitTimelineEntry | None:
    """Return the fit entry at a positional path, or ``None`` if it is gone."""

    if not path:
        return None
    entries = fits
    entry: FitTimelineEntry | None = None
    for index in path:
        if not (0 <= index < len(entries)):
            return None
        entry = entries[index]
        entries = entry.children
    return entry


def _fit_siblings(
    entries: list[FitTimelineEntry],
    target: FitTimelineEntry,
) -> list[FitTimelineEntry] | None:
    if target in entries:
        return entries
    for entry in entries:
        found = _fit_siblings(entry.children, target)
        if found is not None:
            return found
    return None


def _fit_parent(
    entries: list[FitTimelineEntry],
    target: FitTimelineEntry,
) -> FitTimelineEntry | None:
    for entry in entries:
        if target in entry.children:
            return entry
        found = _fit_parent(entry.children, target)
        if found is not None:
            return found
    return None


def _last_result_at_level(entries: list[FitTimelineEntry]) -> FitTimelineEntry | None:
    for entry in reversed(entries):
        if entry.kind == "result":
            return entry
    return None


def _current_state_at_level(entries: list[FitTimelineEntry]) -> FitTimelineEntry | None:
    for entry in reversed(entries):
        if entry.kind == "current":
            return entry
    return None


def _fit_entry_matches_current_state(entries: list[FitTimelineEntry], fit_entry: FitTimelineEntry) -> bool:
    current = _current_state_at_level(entries)
    return bool(current is not None and fit_entry.snapshot == current.snapshot)


def _current_state_after_result(group: DataGroup, result: FitTimelineEntry) -> FitTimelineEntry | None:
    siblings = _fit_siblings(group.fits, result)
    if siblings is None or result not in siblings:
        return None
    result_index = siblings.index(result)
    for entry in siblings[result_index + 1 :]:
        if entry.kind == "current":
            return entry
    return None


def _ensure_current_state_after_result(
    group: DataGroup,
    result: FitTimelineEntry,
) -> tuple[FitTimelineEntry | None, bool]:
    """Return/create the mutable current state immediately after ``result``."""

    siblings = _fit_siblings(group.fits, result)
    if siblings is None or result not in siblings:
        return None, False
    existing = _current_state_after_result(group, result)
    if existing is not None:
        return existing, False
    current = current_state_fit_entry(group, result)
    siblings.insert(siblings.index(result) + 1, current)
    return current, True


def _current_state_for_failed_fit(
    group: DataGroup,
    parent: FitTimelineEntry,
    result: FitTimelineEntry,
) -> FitTimelineEntry | None:
    """Return/create the current state that should stay active after failure."""

    siblings = _fit_siblings(group.fits, result)
    if siblings is None:
        return parent if parent.kind == "current" and _fit_entry_in_tree(group.fits, parent) else None
    if parent.kind == "current" and parent in siblings:
        _set_fit_current_snapshot(parent, group)
        siblings.remove(parent)
        siblings.insert(siblings.index(result) + 1, parent)
        return parent
    if parent.kind in {"initial", "result"}:
        current, _created = _ensure_current_state_after_result(group, result)
        if current is not None:
            _set_fit_current_snapshot(current, group)
        return current
    return None


def _fit_entry_to_select_after_run(
    group: DataGroup,
    parent: FitTimelineEntry,
    result: FitTimelineEntry,
) -> FitTimelineEntry:
    if str(result.goodness.get("status", "")) == "failed":
        current = _current_state_after_result(group, result)
        if current is not None:
            return current
        if parent.kind == "current" and _fit_entry_in_tree(group.fits, parent):
            return parent
    return result


def _is_editable_initial_baseline(
    group: DataGroup,
    fit_entry: FitTimelineEntry | None,
) -> bool:
    """Return whether edits should update Initial instead of creating a branch."""

    if fit_entry is None or fit_entry.kind != "initial":
        return False
    if group.fits != [fit_entry]:
        return False
    return not fit_entry.children


def _should_branch_fit_now(group: DataGroup, parent: FitTimelineEntry) -> bool:
    if parent.kind in {"initial", "current"}:
        return False
    siblings = _fit_siblings(group.fits, parent)
    if siblings is None:
        return True
    return parent is not _last_result_at_level(siblings)


def _replace_current_state(entries: list[FitTimelineEntry], group: DataGroup) -> None:
    del group
    entries[:] = [entry for entry in entries if entry.kind != "current"]


def _top_level_current_state_entry(group: DataGroup) -> FitTimelineEntry | None:
    for entry in reversed(group.fits):
        if entry.kind == "current":
            return entry
    return None


def _set_fit_current_snapshot(entry: FitTimelineEntry, group: DataGroup) -> None:
    entry.snapshot = snapshot_data_group_state(group)
    entry.created_at = _timestamp_now()


@dataclass(frozen=True)
class ProjectStateIssue:
    """One inconsistency between live project state and persisted fit history."""

    group_name: str
    code: str
    message: str
    active_fit_path: tuple[int, ...] | None = None


class ProjectStateConsistencyError(ValueError):
    """Raised when live project state and its active fit snapshot disagree."""


def project_state_issues(project: NfitProject) -> tuple[ProjectStateIssue, ...]:
    """Report active fit-history records that disagree with live project state.

    The live workspace is the authoritative state saved in the project. An
    active fit entry records where subsequent edits should branch, but its
    snapshot should still reproduce the live workspace at save time. Historical
    entries that are not active are intentionally excluded.
    """

    issues: list[ProjectStateIssue] = []
    for group in project.data_groups:
        path = group.active_fit_path
        if path is None:
            continue
        entry = _fit_entry_at_path(group.fits, path)
        normalized_path = tuple(int(index) for index in path)
        if entry is None:
            issues.append(
                ProjectStateIssue(
                    group_name=group.name,
                    code="invalid_active_fit_path",
                    message=(
                        f"workspace {group.name!r} refers to a missing active "
                        f"fit entry at {list(path)!r}"
                    ),
                    active_fit_path=normalized_path,
                )
            )
            continue
        if entry.kind == "timeline" or not entry.snapshot:
            issues.append(
                ProjectStateIssue(
                    group_name=group.name,
                    code="active_fit_snapshot_missing",
                    message=(
                        f"workspace {group.name!r} active fit {entry.name!r} "
                        "does not contain a restorable state snapshot"
                    ),
                    active_fit_path=normalized_path,
                )
            )
            continue
        live = snapshot_data_group_state(group)
        if entry.snapshot != live:
            model_difference = entry.snapshot.get("models") != live.get("models")
            detail = "model state" if model_difference else "workspace state"
            issues.append(
                ProjectStateIssue(
                    group_name=group.name,
                    code="active_fit_snapshot_differs",
                    message=(
                        f"workspace {group.name!r} live {detail} differs from "
                        f"active fit {entry.name!r}; reconcile external edits "
                        "before saving"
                    ),
                    active_fit_path=normalized_path,
                )
            )
    return tuple(issues)


def validate_project_state(project: NfitProject) -> None:
    """Raise when an active fit snapshot would not reproduce live project state."""

    issues = project_state_issues(project)
    if issues:
        raise ProjectStateConsistencyError("; ".join(issue.message for issue in issues))


def reconcile_external_project_edit(group: DataGroup) -> FitTimelineEntry:
    """Record a script-driven workspace edit without rewriting fit results.

    A lone ``Initial`` state and an existing ``Current state`` are mutable.
    Editing from a historical initial/result state creates a current-state
    branch, while editing from the latest result creates its adjacent current
    state. Completed result snapshots remain immutable.
    """

    ensure_fit_history(group)
    active = _fit_entry_at_path(group.fits, group.active_fit_path)
    if active is None and len(group.fits) == 1 and _is_editable_initial_baseline(
        group, group.fits[0]
    ):
        active = group.fits[0]

    if active is not None and active.kind == "initial" and _is_editable_initial_baseline(
        group, active
    ):
        _set_fit_current_snapshot(active, group)
        group.active_fit_path = _fit_entry_path(group.fits, active)
        return active

    if active is not None and active.kind == "current":
        _set_fit_current_snapshot(active, group)
        group.active_fit_path = _fit_entry_path(group.fits, active)
        return active

    if active is not None and active.kind == "result":
        siblings = _fit_siblings(group.fits, active)
        if siblings is not None and active is _last_result_at_level(siblings):
            current, _created = _ensure_current_state_after_result(group, active)
            if current is None:
                raise RuntimeError("could not create current state after active fit result")
            _set_fit_current_snapshot(current, group)
            group.active_fit_path = _fit_entry_path(group.fits, current)
            return current

    if active is not None and active.kind in {"initial", "result"}:
        timeline = FitTimelineEntry(
            name=next_fit_timeline_name(group.fits),
            kind="timeline",
            created_at=_timestamp_now(),
            metadata={
                "branch_reason": "workspace state edited through the public project API",
                "branched_from": active.name,
            },
        )
        current = current_state_fit_entry(group, active)
        timeline.children.append(current)
        active.children.append(timeline)
        group.active_fit_path = _fit_entry_path(group.fits, current)
        return current

    current = _top_level_current_state_entry(group)
    if current is None:
        current = current_state_fit_entry(group)
        group.fits.append(current)
    else:
        _set_fit_current_snapshot(current, group)
    group.active_fit_path = _fit_entry_path(group.fits, current)
    return current


def reconcile_project_external_edits(project: NfitProject) -> dict[str, FitTimelineEntry]:
    """Reconcile every workspace after editing a project outside the GUI."""

    return {
        group.name: reconcile_external_project_edit(group)
        for group in project.data_groups
    }


def _style_enabled_tree_item(item: Any, enabled: bool) -> None:
    from PySide6 import QtGui

    # Enabled items use the theme's default text color (bright in dark mode,
    # dark in light mode); disabled items use a medium gray readable on both.
    if enabled:
        item.setForeground(0, QtGui.QBrush())
    else:
        item.setForeground(0, QtGui.QBrush(QtGui.QColor("#8a8a8a")))


def _style_tree_hierarchy_item(item: Any, *, bold: bool = False, underline: bool = False) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont(item.font(0))
    font.setBold(bold)
    font.setUnderline(underline)
    item.setFont(0, font)


def _style_active_fit_tree_item(item: Any) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont(item.font(0))
    font.setBold(True)
    font.setItalic(True)
    item.setFont(0, font)
    item.setForeground(0, QtGui.QBrush(QtGui.QColor("#62b884")))
    item.setToolTip(0, "Active fit state currently applied to the workspace.")


_TREE_ICON_COLORS = {
    "folder": "#5f8fa8",
    "background_folder": "#d17a5f",
    "mask_folder": "#a88fc5",
    "model_folder": "#c7a45b",
    "fit_folder": "#7cab80",
    "analysis_folder": "#c56f9d",
    "plot_folder": "#58a6a6",
    "dataset": "#6d9fc7",
    "mask": "#a88fc5",
    "model": "#c7a45b",
    "fit_result": "#7cab80",
    "fit_initial": "#7cab80",
    "fit_current": "#7cab80",
}


_TREE_ICON_CACHE: dict[tuple[str, bool], Any] = {}


def _tree_item_icon(kind: str, *, cached: bool = False) -> Any:
    from PySide6 import QtCore, QtGui

    cache_key = (kind, bool(cached))
    if cache_key in _TREE_ICON_CACHE:
        return _TREE_ICON_CACHE[cache_key]

    accent = QtGui.QColor(_TREE_ICON_COLORS.get(kind, "#9ca3ad"))
    line = QtGui.QColor("#c7ccd1")

    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(accent, 1.4)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    if kind in {
        "folder",
        "background_folder",
        "mask_folder",
        "model_folder",
        "fit_folder",
        "analysis_folder",
        "plot_folder",
    }:
        fill = QtGui.QColor(accent)
        fill.setAlpha(34)
        painter.setPen(QtGui.QPen(accent, 1.15))
        painter.setBrush(fill)
        path = QtGui.QPainterPath(QtCore.QPointF(2.5, 4.6))
        path.lineTo(QtCore.QPointF(5.7, 4.6))
        path.lineTo(QtCore.QPointF(7.0, 6.1))
        path.lineTo(QtCore.QPointF(13.5, 6.1))
        path.lineTo(QtCore.QPointF(13.5, 12.9))
        path.lineTo(QtCore.QPointF(2.5, 12.9))
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    elif kind == "dataset":
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawRoundedRect(QtCore.QRectF(2.3, 4.0, 11.4, 8.7), 1.0, 1.0)
        painter.setPen(QtGui.QPen(accent, 1.0))
        for x in (6.1, 9.9):
            painter.drawLine(QtCore.QPointF(x, 4.3), QtCore.QPointF(x, 12.4))
        for y in (6.9, 9.8):
            painter.drawLine(QtCore.QPointF(2.6, y), QtCore.QPointF(13.4, y))
    elif kind == "mask":
        removed = QtGui.QColor(accent)
        removed.setAlpha(88)
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawRoundedRect(QtCore.QRectF(3.0, 3.0, 10.0, 10.0), 1.2, 1.2)
        painter.setPen(QtGui.QPen(accent, 1.0))
        painter.setBrush(removed)
        painter.drawRect(QtCore.QRectF(3.4, 3.4, 4.0, 4.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawLine(QtCore.QPointF(4.1, 4.1), QtCore.QPointF(6.7, 6.7))
        painter.drawLine(QtCore.QPointF(6.7, 4.1), QtCore.QPointF(4.1, 6.7))
    elif kind == "model":
        path = QtGui.QPainterPath(QtCore.QPointF(2.8, 10.8))
        path.cubicTo(QtCore.QPointF(5.0, 3.0), QtCore.QPointF(7.2, 13.0), QtCore.QPointF(10.0, 5.2))
        path.cubicTo(QtCore.QPointF(11.2, 2.0), QtCore.QPointF(12.8, 5.5), QtCore.QPointF(13.2, 8.0))
        painter.drawPath(path)
        painter.setBrush(accent)
        painter.drawEllipse(QtCore.QPointF(5.2, 7.0), 1.1, 1.1)
        painter.drawEllipse(QtCore.QPointF(10.0, 5.2), 1.1, 1.1)
    elif kind == "fit_result":
        painter.setPen(QtGui.QPen(line, 1.0))
        painter.drawLine(QtCore.QPointF(3.0, 12.0), QtCore.QPointF(13.0, 12.0))
        painter.drawLine(QtCore.QPointF(3.0, 12.0), QtCore.QPointF(3.0, 4.0))
        painter.setPen(QtGui.QPen(accent, 1.4))
        painter.drawPolyline(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(4.0, 10.5),
                    QtCore.QPointF(6.3, 8.4),
                    QtCore.QPointF(8.4, 9.3),
                    QtCore.QPointF(11.8, 5.0),
                ]
            )
        )
    elif kind == "fit_initial":
        painter.setPen(QtGui.QPen(accent, 1.4))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 4.2, 4.2)
        painter.drawLine(QtCore.QPointF(8.0, 3.8), QtCore.QPointF(8.0, 6.0))
    elif kind == "fit_current":
        painter.setPen(QtGui.QPen(line, 1.1))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 4.5, 4.5)
        painter.setPen(QtGui.QPen(accent, 1.3))
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 2.1, 2.1)
        painter.setBrush(accent)
        painter.drawEllipse(QtCore.QPointF(8.0, 8.0), 0.9, 0.9)

    if cached:
        painter.setPen(QtGui.QPen(QtGui.QColor("#f4f7f5"), 1.1))
        painter.setBrush(QtGui.QColor("#22b85a"))
        painter.drawEllipse(QtCore.QRectF(10.0, 10.0, 5.2, 5.2))
    painter.end()
    icon = QtGui.QIcon(pixmap)
    _TREE_ICON_CACHE[cache_key] = icon
    return icon


def _set_tree_item_icon(item: Any, kind: str, *, cached: bool = False) -> None:
    item.setIcon(0, _tree_item_icon(kind, cached=cached))


def _enabled_state_for_role(
    role: str,
    entry: DatasetEntry | None,
    mask: MaskSpec | None,
    model: ModelComponentSpec | None,
) -> bool | None:
    if role == "dataset" and entry is not None:
        return bool(entry.enabled)
    if role == "mask" and mask is not None:
        return bool(mask.enabled)
    if role == "model" and model is not None:
        return bool(model.enabled)
    return None


def dataset_entry_from_path(
    path: str | Path,
    *,
    data_type: str | None = None,
    importer_name: str | None = None,
    importer_options: dict[str, Any] | None = None,
) -> DatasetEntry:
    """Create a dataset entry for a selected source file."""

    return _dataset_entry_from_path_impl(
        path,
        data_type=data_type,
        importer_name=importer_name,
        importer_options=importer_options,
        dataset_file_loader=_load_nfit_dataset_file,
    )


def project_cache_binnings_enabled(project: NfitProject) -> bool:
    """Return whether this project embeds current rebin caches when saved."""

    return bool(project.settings.get(PROJECT_CACHE_BINNINGS_KEY, False))


def _project_binning_targets(
    project: NfitProject,
) -> list[tuple[str, str, DataGroup, Any, str, dict[str, Any]]]:
    """Return configured dataset and composite binnings in project order."""

    targets = []
    for group in project.data_groups:
        for dataset in group.iter_datasets():
            if not isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict):
                continue
            for binning in dataset_rebin_binnings(dataset):
                if binning["config"].get("enabled"):
                    targets.append(
                        (
                            "dataset",
                            _binning_view_name(dataset.name, binning["name"]),
                            group,
                            dataset,
                            binning["id"],
                            binning["config"],
                        )
                    )
        for scope in _composite_scopes(group):
            if (
                not data_group_composite_enabled(scope)
                and not isinstance(
                    scope.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY), dict
                )
            ):
                continue
            for binning in data_group_composite_binnings(scope):
                if binning["config"].get("enabled"):
                    targets.append(
                        (
                            "dataset group",
                            _binning_view_name(scope.name, binning["name"]),
                            group,
                            scope,
                            binning["id"],
                            binning["config"],
                        )
                    )
    return targets


def _project_binning_is_current(
    kind: str,
    group: DataGroup,
    target: Any,
    binning_id: str,
    config: dict[str, Any],
) -> bool:
    if kind == "dataset":
        key = target.id if config is _fit_dataset_rebin_config(target) else f"{target.id}:{binning_id}"
        if hasattr(_VIEWER_VIEW_CACHE, "has_signature"):
            return _VIEWER_VIEW_CACHE.has_signature(
                key, _viewer_view_signature(target, effective_dataset_masks(group, target), config)
            )
        return (
            _peek_cached_dataset_view(
                target,
                extra_masks=effective_dataset_masks(group, target),
                rebin_config=config,
                cache_id=(None if config is _fit_dataset_rebin_config(target) else binning_id),
            )
            is not None
        )
    fit = config is _fit_data_group_composite_config(target)
    if hasattr(_COMPOSITE_DATA_CACHE, "has_signature"):
        return _COMPOSITE_DATA_CACHE.has_signature(
            _composite_cache_key(target, None if fit else binning_id),
            (
                _composite_cache_signature(target)
                if fit
                else _composite_cache_signature(
                    target, config_override=config, binning_id=binning_id
                )
            ),
        )
    return _peek_cached_composite_dataset_data(
        target,
        config_override=(None if fit else config),
        binning_id=(None if fit else binning_id),
    ) is not None


def _saved_binning_compressed_size(
    project: NfitProject,
    *,
    kind: str,
    group: DataGroup,
    target: DatasetEntry | DataGroup | _CompositeScope,
    binning_id: str,
    config: dict[str, Any],
) -> int | None:
    """Return the saved archive size of a current persisted binning."""

    project_path = getattr(project, "_project_path", None)
    entries = project.settings.get(PROJECT_BINNING_CACHE_ENTRIES_KEY, [])
    if project_path is None or not isinstance(entries, list):
        return None
    try:
        group_index = next(
            index
            for index, candidate in enumerate(project.data_groups)
            if candidate is group
        )
    except StopIteration:
        return None
    entry_type = "dataset" if kind == "dataset" else "composite"
    node_id = (
        target.node.id
        if isinstance(target, _CompositeScope)
        else None
    )
    current_signature = (
        _viewer_view_signature(
            target,
            effective_dataset_masks(group, target),
            config,
        )
        if entry_type == "dataset"
        else (
            _composite_cache_signature(target)
            if config is _fit_data_group_composite_config(target)
            else _composite_cache_signature(
                target,
                config_override=config,
                binning_id=binning_id,
            )
        )
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != entry_type:
            continue
        try:
            if int(entry.get("group_index", -1)) != group_index:
                continue
        except (TypeError, ValueError):
            continue
        if str(entry.get("binning_id", "")) != str(binning_id):
            continue
        if entry_type == "dataset":
            if str(entry.get("dataset_id", "")) != target.id:
                continue
        elif entry.get("node_id") != node_id:
            continue
        try:
            format_version = int(entry.get("format_version", 0))
        except (TypeError, ValueError):
            continue
        if format_version not in PROJECT_BINNING_CACHE_COMPATIBLE_FORMATS:
            continue
        saved_signature = entry.get("signature")
        if format_version >= 6:
            if not _binning_signatures_match(saved_signature, current_signature):
                return None
        elif not _project_binning_is_current(
            kind, group, target, binning_id, config
        ):
            return None
        member = entry.get("member")
        if not isinstance(member, str):
            return None
        return project_artifact_compressed_size(project_path, member)
    return None


def _dataset_cached_binnings_current(
    group: DataGroup, dataset: DatasetEntry
) -> bool:
    """Return whether every enabled configured binning has a current cache."""

    if not isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict):
        return False
    enabled = [
        item
        for item in dataset_rebin_binnings(dataset)
        if bool(item["config"].get("enabled", False))
    ]
    return bool(enabled) and all(
        _project_binning_is_current(
            "dataset", group, dataset, item["id"], item["config"]
        )
        for item in enabled
    )


def _composite_cached_binnings_current(
    group: DataGroup, node: DataGroup | DatasetGroup
) -> bool:
    """Return whether every enabled composite binning has a current cache."""

    scope = _composite_scope(group, node)
    if not data_group_composite_enabled(scope):
        return False
    enabled = [
        item
        for item in data_group_composite_binnings(scope)
        if bool(item["config"].get("enabled", False))
    ]
    return bool(enabled) and all(
        _project_binning_is_current(
            "dataset group", group, scope, item["id"], item["config"]
        )
        for item in enabled
    )


def project_binnings_need_refresh(project: NfitProject) -> bool:
    """Return whether any configured binning lacks a current process cache."""

    return any(
        not _project_binning_is_current(kind, group, target, binning_id, config)
        for kind, _name, group, target, binning_id, config in _project_binning_targets(project)
    )


def prepare_project_binning_cache(
    project: NfitProject,
    *,
    progress_callback: Any | None = None,
) -> int:
    """Recompute only stale/missing configured binnings and return their count."""

    targets = _project_binning_targets(project)
    stale = [
        target
        for target in targets
        if not _project_binning_is_current(
            target[0], target[2], target[3], target[4], target[5]
        )
    ]
    state = {"total": len(stale), "completed": 0}
    rebin_totals: dict[tuple[str, int], int] = {}
    for kind, _name, _group, target, _binning_id, _config in targets:
        key = (kind, id(target))
        rebin_totals[key] = rebin_totals.get(key, 0) + 1
    stale_totals: dict[tuple[str, int], int] = {}
    for kind, _name, _group, target, _binning_id, _config in stale:
        key = (kind, id(target))
        stale_totals[key] = stale_totals.get(key, 0) + 1
    rebin_completed = {
        key: total - stale_totals.get(key, 0)
        for key, total in rebin_totals.items()
    }
    for kind, name, group, target, binning_id, config in stale:
        rebin_key = (kind, id(target))
        rebin_total = rebin_totals[rebin_key]
        rebin_index = rebin_completed.get(rebin_key, 0)
        rebin_name = name.rsplit(" · ", 1)[-1]
        rebin_progress = {"completed": rebin_index}

        def item_progress(
            event: dict[str, Any],
            *,
            _total: int = rebin_total,
            _name: str = rebin_name,
            _progress: dict[str, int] = rebin_progress,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        **event,
                        "rebin_total": _total,
                        "rebin_completed": _progress["completed"],
                        "rebin_name": _name,
                    }
                )

        callback = item_progress if progress_callback is not None else None
        _report_effective_dataset_batch(
            callback,
            state,
            name=name,
            kind=kind,
            completed=False,
        )
        if kind == "dataset":
            dataset_for_slice_viewer(
                target,
                extra_masks=effective_dataset_masks(group, target),
                force_rebin=True,
                force_masks=True,
                progress_callback=callback,
                rebin_config=config,
                cache_id=(None if config is _fit_dataset_rebin_config(target) else binning_id),
            )
        else:
            fit = config is _fit_data_group_composite_config(target)
            _cached_composite_dataset_data(
                target,
                force_rebin=True,
                progress_callback=callback,
                config_override=(None if fit else config),
                binning_id=(None if fit else binning_id),
            )
        rebin_completed[rebin_key] = rebin_index + 1
        rebin_progress["completed"] += 1
        _report_effective_dataset_batch(
            callback,
            state,
            name=name,
            kind=kind,
            completed=True,
        )
    return len(stale)


def _project_binning_artifacts(
    project: NfitProject,
    directory: Path,
) -> tuple[dict[str, ArchiveContent], list[dict[str, Any]]]:
    """Write current signature-matching rebin caches to temporary artifacts."""

    artifacts: dict[str, ArchiveContent] = {}
    entries: list[dict[str, Any]] = []
    for group_index, group in enumerate(project.data_groups):
        for dataset in group.iter_datasets():
            if not isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict):
                continue
            fit_config = _fit_dataset_rebin_config(dataset)
            for binning in dataset_rebin_binnings(dataset):
                config = binning["config"]
                if not config.get("enabled"):
                    continue
                is_fit = config is fit_config
                signature = _viewer_view_signature(
                    dataset, effective_dataset_masks(group, dataset), config
                )
                key = dataset.id if is_fit else f"{dataset.id}:{binning['id']}"
                cache_id = f"dataset-{dataset.id}-{binning['id']}"
                member = binning_artifact_member(cache_id)
                backing = (
                    _VIEWER_VIEW_CACHE.project_backing(key, signature)
                    if hasattr(_VIEWER_VIEW_CACHE, "project_backing")
                    else None
                )
                if backing is not None:
                    artifacts[member] = ArchiveMember(*backing)
                else:
                    data = _peek_cached_dataset_view(
                        dataset,
                        extra_masks=effective_dataset_masks(group, dataset),
                        rebin_config=config,
                        cache_id=(None if is_fit else binning["id"]),
                    )
                    if not isinstance(data, (MDHistoData, PointListData)):
                        continue
                    artifact_path = directory / f"{cache_id}.npz"
                    write_dataset_artifact(data, artifact_path)
                    artifacts[member] = artifact_path
                entries.append(
                    {
                        "type": "dataset",
                        "format_version": PROJECT_BINNING_CACHE_FORMAT_VERSION,
                        "signature": signature,
                        "group_index": group_index,
                        "dataset_id": dataset.id,
                        "binning_id": binning["id"],
                        "member": member,
                    }
                )
        for scope in _composite_scopes(group):
            if (
                not data_group_composite_enabled(scope)
                and not isinstance(
                    scope.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY), dict
                )
            ):
                continue
            node_id = scope.node.id if isinstance(scope, _CompositeScope) else None
            fit_config = _fit_data_group_composite_config(scope)
            for binning in data_group_composite_binnings(scope):
                config = binning["config"]
                if not config.get("enabled"):
                    continue
                is_fit = config is fit_config
                signature = (
                    _composite_cache_signature(scope)
                    if is_fit
                    else _composite_cache_signature(
                        scope,
                        config_override=config,
                        binning_id=binning["id"],
                    )
                )
                key = _composite_cache_key(
                    scope, None if is_fit else binning["id"]
                )
                owner = node_id if node_id is not None else f"root-{group_index}"
                cache_id = f"composite-{owner}-{binning['id']}"
                member = binning_artifact_member(cache_id)
                backing = (
                    _COMPOSITE_DATA_CACHE.project_backing(key, signature)
                    if hasattr(_COMPOSITE_DATA_CACHE, "project_backing")
                    else None
                )
                if backing is not None:
                    artifacts[member] = ArchiveMember(*backing)
                else:
                    data = _peek_cached_composite_dataset_data(
                        scope,
                        config_override=(None if is_fit else config),
                        binning_id=(None if is_fit else binning["id"]),
                    )
                    if not isinstance(data, (MDHistoData, PointListData)):
                        continue
                    artifact_path = directory / f"{cache_id}.npz"
                    write_dataset_artifact(data, artifact_path)
                    artifacts[member] = artifact_path
                entries.append(
                    {
                        "type": "composite",
                        "format_version": PROJECT_BINNING_CACHE_FORMAT_VERSION,
                        "signature": signature,
                        "group_index": group_index,
                        "node_id": node_id,
                        "binning_id": binning["id"],
                        "member": member,
                    }
                )
    return artifacts, entries


def _adopt_saved_project_binning_backing(
    project: NfitProject,
    path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Replace session spill files with lazy references to saved project members."""

    for entry in entries:
        try:
            group = project.data_groups[int(entry["group_index"])]
            binning_id = str(entry["binning_id"])
            signature = str(entry["signature"])
            member = str(entry["member"])
            if entry.get("type") == "dataset":
                dataset = next(
                    item
                    for item in group.iter_datasets()
                    if item.id == str(entry["dataset_id"])
                )
                config = dataset_rebin_config_by_id(dataset, binning_id)
                key = (
                    dataset.id
                    if config is _fit_dataset_rebin_config(dataset)
                    else f"{dataset.id}:{binning_id}"
                )
                _VIEWER_VIEW_CACHE.set_project_backing(
                    key,
                    signature=signature,
                    project_path=path,
                    member=member,
                )
            elif entry.get("type") == "composite":
                node_id = entry.get("node_id")
                scope = (
                    group
                    if node_id is None
                    else _composite_scope(
                        group,
                        next(
                            node
                            for node in group.iter_subgroups()
                            if node.id == str(node_id)
                        ),
                    )
                )
                config = data_group_composite_config_by_id(scope, binning_id)
                key = _composite_cache_key(
                    scope,
                    None
                    if config is _fit_data_group_composite_config(scope)
                    else binning_id,
                )
                _COMPOSITE_DATA_CACHE.set_project_backing(
                    key,
                    signature=signature,
                    project_path=path,
                    member=member,
                )
        except (IndexError, KeyError, StopIteration, TypeError, ValueError):
            continue


def _restore_project_binning_cache(project: NfitProject, path: Path) -> None:
    """Restore embedded binnings into the process-local caches."""

    if not project_cache_binnings_enabled(project):
        return
    entries = project.settings.get(PROJECT_BINNING_CACHE_ENTRIES_KEY, [])
    if not isinstance(entries, list):
        return
    resolved: list[
        tuple[
            str,
            Any,
            Any,
            str,
            dict[str, Any],
            bool,
            int,
            str | None,
            str,
        ]
    ] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            format_version = int(entry.get("format_version", 0))
            if format_version not in PROJECT_BINNING_CACHE_COMPATIBLE_FORMATS:
                continue
            saved_signature = entry.get("signature")
            if saved_signature is not None and not isinstance(saved_signature, str):
                continue
            group = project.data_groups[int(entry["group_index"])]
            member = str(entry["member"])
            if not project_artifact_exists(path, member):
                continue
            data = (
                None
                if format_version >= 6
                else read_project_dataset_artifact(path, member)
            )
            if entry.get("type") == "dataset":
                dataset = next(
                    item
                    for item in group.iter_datasets()
                    if item.id == str(entry["dataset_id"])
                )
                binning_id = str(entry["binning_id"])
                config = dataset_rebin_config_by_id(dataset, binning_id)
                resolved.append(
                    (
                        "dataset",
                        (group, dataset),
                        data,
                        binning_id,
                        config,
                        config is _fit_dataset_rebin_config(dataset),
                        format_version,
                        saved_signature,
                        member,
                    )
                )
            elif entry.get("type") == "composite":
                node_id = entry.get("node_id")
                scope = (
                    group
                    if node_id is None
                    else _composite_scope(
                        group,
                        next(
                            node
                            for node in group.iter_subgroups()
                            if node.id == str(node_id)
                        ),
                    )
                )
                binning_id = str(entry["binning_id"])
                config = data_group_composite_config_by_id(scope, binning_id)
                resolved.append(
                    (
                        "composite",
                        scope,
                        data,
                        binning_id,
                        config,
                        config is _fit_data_group_composite_config(scope),
                        format_version,
                        saved_signature,
                        member,
                    )
                )
        except (IndexError, KeyError, OSError, StopIteration, TypeError, ValueError):
            continue

    # Signature construction may lazily load a metadata channel or background
    # source and thereby advance a dataset's content token. Prime every
    # dependency before capturing the signatures installed in the caches; a
    # one-pass restore can otherwise invalidate entries restored earlier in
    # archive order.
    for (
        kind,
        target,
        _data,
        binning_id,
        config,
        is_fit,
        _version,
        _saved,
        _member,
    ) in resolved:
        if kind == "dataset":
            group, dataset = target
            _viewer_view_signature(
                dataset, effective_dataset_masks(group, dataset), config
            )
        else:
            if is_fit:
                _composite_cache_signature(target)
            else:
                _composite_cache_signature(
                    target,
                    config_override=config,
                    binning_id=binning_id,
                )
    for (
        kind,
        target,
        data,
        binning_id,
        config,
        is_fit,
        version,
        saved,
        member,
    ) in resolved:
        if kind == "dataset":
            group, dataset = target
            signature = _viewer_view_signature(
                dataset,
                effective_dataset_masks(group, dataset),
                config,
            )
            if version >= 6 and not _binning_signatures_match(saved, signature):
                continue
            key = dataset.id if is_fit else f"{dataset.id}:{binning_id}"
            if version >= 6:
                _VIEWER_VIEW_CACHE.set_project_backing(
                    key,
                    signature=signature,
                    project_path=path,
                    member=member,
                    lazy=True,
                )
            else:
                _lru_store(
                    _VIEWER_VIEW_CACHE,
                    key,
                    (signature, data),
                    _VIEWER_VIEW_CACHE_LIMIT,
                    _VIEWER_VIEW_CACHE_MAX_BYTES,
                )
        else:
            signature = (
                _composite_cache_signature(target)
                if is_fit
                else _composite_cache_signature(
                    target,
                    config_override=config,
                    binning_id=binning_id,
                )
            )
            if version >= 6 and saved != signature:
                continue
            key = _composite_cache_key(target, None if is_fit else binning_id)
            if version >= 6:
                _COMPOSITE_DATA_CACHE.set_project_backing(
                    key,
                    signature=signature,
                    project_path=path,
                    member=member,
                    lazy=True,
                )
            else:
                _lru_store(
                    _COMPOSITE_DATA_CACHE,
                    key,
                    (signature, data),
                    _COMPOSITE_DATA_CACHE_LIMIT,
                    _COMPOSITE_DATA_CACHE_MAX_BYTES,
                )


def save_project(
    project: NfitProject,
    path: str | Path,
    *,
    asset_source: str | Path | None = None,
    progress_callback: Any | None = None,
) -> None:
    """Persist project state and analysis artifacts in one nfit archive."""

    target = Path(path)
    if asset_source is None:
        asset_source = getattr(project, "_project_path", None)
    missing_entries = object()
    previous_entries = project.settings.get(
        PROJECT_BINNING_CACHE_ENTRIES_KEY,
        missing_entries,
    )

    def restore_previous_entries() -> None:
        if previous_entries is missing_entries:
            project.settings.pop(PROJECT_BINNING_CACHE_ENTRIES_KEY, None)
        else:
            project.settings[PROJECT_BINNING_CACHE_ENTRIES_KEY] = previous_entries

    if project_cache_binnings_enabled(project):
        prepare_project_binning_cache(
            project,
            progress_callback=progress_callback,
        )
        with tempfile.TemporaryDirectory(prefix="nfit-binning-cache-") as temporary:
            artifacts, entries = _project_binning_artifacts(project, Path(temporary))
            project.settings[PROJECT_BINNING_CACHE_ENTRIES_KEY] = entries
            try:
                write_project_manifest(
                    target,
                    _project_to_dict(project),
                    asset_source=asset_source,
                    preserve_existing=asset_source is not None,
                    binning_artifacts=artifacts,
                )
            except Exception:
                restore_previous_entries()
                raise
        _adopt_saved_project_binning_backing(project, target, entries)
    else:
        project.settings.pop(PROJECT_BINNING_CACHE_ENTRIES_KEY, None)
        try:
            write_project_manifest(
                target,
                _project_to_dict(project),
                asset_source=asset_source,
                preserve_existing=asset_source is not None,
                binning_artifacts={},
            )
        except Exception:
            restore_previous_entries()
            raise
        _COMPOSITE_DATA_CACHE.clear_project_backing(target)
        _VIEWER_VIEW_CACHE.clear_project_backing(target)
    project._project_path = target
    _bind_project_analysis_sources(project, target, load_data=False)


def _bind_project_analysis_sources(
    project: NfitProject,
    project_path: Path,
    *,
    load_data: bool = True,
) -> None:
    for group in project.data_groups:
        for dataset in group.iter_datasets():
            if not (
                dataset.metadata.get("derived_from_analysis")
                or dataset.metadata.get("project_artifact_path")
            ):
                continue
            artifact = (
                dataset.metadata.get("project_artifact_path")
                or dataset.metadata.get("analysis_artifact_path")
                or dataset.metadata.get("source_file")
            )
            if not artifact:
                continue
            if dataset.metadata.get("derived_from_analysis"):
                dataset.metadata["analysis_artifact_path"] = str(artifact)
            else:
                dataset.metadata["project_artifact_path"] = str(artifact)
            dataset.metadata["source_file"] = str(artifact)
            dataset.metadata["_project_path"] = str(project_path)
            # Project-owned composite materializations can be multi-gigabyte.
            # Keep them lazy on project open; ordinary analysis outputs retain
            # their historical eager-load behavior for the Analysis Window.
            if (
                not load_data
                or dataset.data is not None
                or dataset.metadata.get("project_artifact_path")
            ):
                continue
            try:
                data = read_project_dataset_artifact(project_path, str(artifact))
            except (KeyError, OSError, TypeError, ValueError):
                dataset.metadata["import_status"] = "error"
                dataset.metadata["import_error"] = "analysis artifact is unavailable"
                continue
            dataset.replace_data(data, source_backed=True)
            dataset.metadata["import_status"] = "loaded"
            dataset.metadata.pop("import_error", None)


def load_project(path: str | Path) -> NfitProject:
    """Load a project saved by :func:`save_project`."""

    project_path = Path(path)
    project = _project_from_dict(read_project_manifest(project_path))
    project._project_path = project_path
    _bind_project_analysis_sources(project, project_path)
    _restore_project_binning_cache(project, project_path)
    return project


@contextmanager
def edit_project_file(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
):
    """Load, reconcile, validate, and atomically save a scripted project edit.

    The yielded :class:`NfitProject` is an ordinary GUI-independent project
    object. On successful context exit, live workspace state is recorded in
    editable fit history (or a new branch), checked for consistency, and saved.
    Exceptions inside the context leave the source archive unchanged.
    """

    source = Path(path)
    target = source if output_path is None else Path(output_path)
    project = load_project(source)
    original_states = {
        id(group): snapshot_data_group_state(group)
        for group in project.data_groups
    }
    yield project
    for group in project.data_groups:
        if (
            id(group) not in original_states
            or snapshot_data_group_state(group) != original_states[id(group)]
        ):
            reconcile_external_project_edit(group)
    validate_project_state(project)
    save_project(project, target, asset_source=source)


def _saved_plot_source_views(
    group: DataGroup,
    plot: PlotEntry,
) -> tuple[list[MDHistoData], list[str]]:
    """Prepare saved-plot sources with their plot-owned rebin snapshots."""

    composite_recipe = plot.settings.get(PLOT_SOURCE_COMPOSITE_KEY)
    if isinstance(composite_recipe, dict):
        subgroup_id = composite_recipe.get("dataset_group_id")
        if subgroup_id is None:
            scope: DataGroup | _CompositeScope = group
        else:
            subgroup = next(
                (item for item in group.iter_subgroups() if item.id == subgroup_id),
                None,
            )
            if subgroup is None:
                raise ValueError("saved plot composite dataset group is missing")
            scope = _composite_scope(group, subgroup)
        config = composite_recipe.get("config")
        if not isinstance(config, dict):
            raise ValueError("saved plot composite configuration is missing")
        view = composite_dataset_data(
            scope, config_override=copy.deepcopy(config),
            metadata_dimensions_override=composite_recipe.get("metadata_dimensions", []),
        )
        if not isinstance(view, MDHistoData):
            raise TypeError("saved plots currently require MDHisto data")
        name = str(composite_recipe.get("name") or _composite_dataset_name(scope))
        return [view], [name]

    entries_by_id = {item.id: item for item in group.iter_datasets()}
    rebin_configs = plot.settings.get(PLOT_SOURCE_REBIN_CONFIGS_KEY, {})
    if not isinstance(rebin_configs, dict):
        rebin_configs = {}
    model_channels = current_model_channels(
        group,
        unmask_model=bool(plot.settings.get("unmask_model", False)),
    )
    views: list[MDHistoData] = []
    names: list[str] = []
    for source in plot.sources:
        if not source.dataset_id:
            continue
        dataset = entries_by_id.get(source.dataset_id)
        if dataset is None:
            raise ValueError("saved plot source dataset is missing")
        prepared_entry = dataset
        saved_rebin = rebin_configs.get(dataset.id)
        if isinstance(saved_rebin, dict):
            parameters = copy.deepcopy(dataset.parameters)
            parameters[DATASET_REBIN_KEY] = copy.deepcopy(saved_rebin)
            prepared_entry = replace(dataset, parameters=parameters)
        view = dataset_for_slice_viewer(
            prepared_entry,
            extra_masks=effective_dataset_masks(group, dataset),
            force_rebin=True,
            force_masks=True,
        )
        if not isinstance(view, MDHistoData):
            raise TypeError("saved plots currently require MDHisto data")
        view.metadata.update(
            {
                "source_dataset_name": dataset.name,
                "binning_name": str(plot.settings.get("binning_name") or "Default"),
            }
        )
        views.append(
            attach_fit_channels_to_view(
                group,
                dataset.name,
                view,
                fallback_payload=model_channels.get(dataset.name),
            )
        )
        names.append(dataset.name)
    return views, names


def render_project_plot(
    project: NfitProject,
    plot_id: str,
    *,
    settings: dict[str, Any] | None = None,
):
    """Render a saved plot through project data preparation without opening Qt."""

    for group in project.data_groups:
        for plot in group.plots:
            if plot.id != plot_id:
                continue
            if settings is not None:
                plot = replace(plot, settings=dict(settings))
            if plot.type == "fit_covariance":
                fit_id = plot.sources[0].fit_id if plot.sources else None
                fit_entry = next((fit for fit in _walk_fit_entries(group.fits) if fit.id == fit_id), None)
                if fit_entry is None:
                    raise ValueError("saved plot source fit is missing")
                return render_plot(plot, fit_entry=fit_entry)
            source_ids = [
                source.dataset_id
                for source in plot.sources
                if source.dataset_id
            ]
            if not source_ids:
                raise ValueError("saved plot has no dataset source")
            prepared, _names = _saved_plot_source_views(group, plot)
            is_composite = isinstance(
                plot.settings.get(PLOT_SOURCE_COMPOSITE_KEY), dict
            )
            if not prepared or (not is_composite and len(prepared) != len(source_ids)):
                raise ValueError("saved plot source dataset is missing")
            return render_plot(
                plot,
                prepared if plot.type == "mdhisto_waterfall" else prepared[0],
            )
    raise ValueError(f"unknown saved plot {plot_id!r}")


def _is_pytest_temporary_project(path: Path) -> bool:
    """Return whether *path* belongs to pytest's per-run temporary hierarchy."""

    parts = path.expanduser().parts
    return any(part.startswith("pytest-of-") for part in parts) and any(
        part.startswith("pytest-") for part in parts
    )


def recent_project_paths(settings: Any | None = None) -> list[Path]:
    """Return recently opened project paths from app settings.

    Pytest opens temporary projects through the ordinary GUI paths. Excluding
    those entries here also removes stale test entries created by older runs.
    """

    settings = _settings() if settings is None else settings
    value = settings.value(RECENT_PROJECTS_KEY, [])
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    recent = [Path(str(path)) for path in values]
    filtered = [path for path in recent if not _is_pytest_temporary_project(path)]
    if filtered != recent:
        settings.setValue(RECENT_PROJECTS_KEY, [str(path) for path in filtered])
    return filtered


def remember_recent_project(path: str | Path, settings: Any | None = None) -> list[Path]:
    """Record a project path as most-recent and return the updated list."""

    settings = _settings() if settings is None else settings
    resolved = Path(path).expanduser()
    recent = [existing for existing in recent_project_paths(settings) if existing != resolved]
    if _is_pytest_temporary_project(resolved):
        return recent
    recent.insert(0, resolved)
    recent = recent[:RECENT_PROJECT_LIMIT]
    settings.setValue(RECENT_PROJECTS_KEY, [str(path) for path in recent])
    return recent


def forget_missing_recent_projects(settings: Any | None = None) -> list[Path]:
    """Drop recent projects whose files are no longer present."""

    settings = _settings() if settings is None else settings
    recent = [path for path in recent_project_paths(settings) if path.exists()]
    settings.setValue(RECENT_PROJECTS_KEY, [str(path) for path in recent])
    return recent


def _format_progress_duration(seconds: float) -> str:
    """Return a compact, stable duration for progress displays."""

    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds_value:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds_value:02d}s"
    return f"{seconds_value:d}s"


def _progress_timer_text(started_at: float) -> str:
    """Return elapsed time for a progress display."""

    elapsed = max(time.monotonic() - started_at, 0.0)
    return f"elapsed {_format_progress_duration(elapsed)}"


class RebinCancellationRequested(RuntimeError):
    """Raised cooperatively when the user cancels a rebin operation."""


class _RebinProgressDialog:
    """Compact one- or two-level progress window for every rebin workflow."""

    def __init__(self, parent: Any, title: str, *, aggregate: bool = False) -> None:
        from PySide6 import QtCore, QtWidgets

        self._title = title
        self._batch_name = ""
        self._batch_kind = ""
        self._batch_operation = ""
        self._batch_unit = "work item"
        self._rebin_name = ""
        self._rebin_completed = 0
        self._rebin_total = 0
        self._batch_started_at = time.monotonic()
        self._detail_started_at = self._batch_started_at
        self._batch_completed = 0
        self._batch_total = 0
        self._detail_completed = 0
        self._detail_total = 0
        self._batch_base_text = title
        self._detail_base_text = title
        self._detail_start_key: tuple[str, int] | None = None
        self._cancel_requested = False
        self._cancel_callback = None

        owner = parent.window if hasattr(parent, "window") else parent
        self.dialog = QtWidgets.QDialog(owner)
        self.dialog.setObjectName("rebin_progress_dialog")
        self.dialog.setWindowTitle("Rebin progress")
        self.dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
            if owner is not None
            else QtCore.Qt.WindowModality.ApplicationModal
        )
        if platform.system() == "Linux":
            self.dialog.setWindowFlag(
                QtCore.Qt.WindowType.WindowStaysOnTopHint,
                True,
            )
        self.dialog.setFixedWidth(680)
        layout = QtWidgets.QVBoxLayout(self.dialog)

        self.batch_label = QtWidgets.QLabel("")
        self.batch_label.setObjectName("rebin_batch_status_label")
        self.batch_label.setWordWrap(True)
        self.batch_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.batch_label)
        self.current_label = QtWidgets.QLabel("")
        self.current_label.setObjectName("rebin_current_item_label")
        self.current_label.setWordWrap(True)
        self.current_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_label)
        self.batch_bar = QtWidgets.QProgressBar()
        self.batch_bar.setObjectName("rebin_batch_progress")
        self.batch_bar.setTextVisible(False)
        self.batch_bar.setToolTip(
            "Completed work items across the complete rebin and preparation task."
        )
        layout.addWidget(self.batch_bar)

        self.detail_label = QtWidgets.QLabel(title)
        self.detail_label.setObjectName("rebin_progress_label")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detail_label)
        self.detail_bar = QtWidgets.QProgressBar()
        self.detail_bar.setObjectName("rebin_detail_progress")
        self.detail_bar.setTextVisible(False)
        self.detail_bar.setToolTip(
            "Progress within the current work item. This bar reaches its end only "
            "when the overall bar advances."
        )
        self.detail_bar.setRange(0, 0)
        layout.addWidget(self.detail_bar)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("rebin_cancel_button")
        self.cancel_button.setToolTip(
            "Stop the rebin at the next processing checkpoint."
        )
        self.cancel_button.clicked.connect(self._request_cancel)
        layout.addWidget(
            self.cancel_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight
        )

        # Compatibility attributes used by GUI tests and downstream extensions.
        self.dialog._nfit_label = self.detail_label
        self.dialog._nfit_batch_label = self.batch_label
        self.dialog._nfit_current_label = self.current_label
        self.dialog._nfit_batch_progress = self.batch_bar
        self.dialog._nfit_detail_progress = self.detail_bar

        self._set_batch_visible(False)
        self._timer = QtCore.QTimer(self.dialog)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_labels)
        self._timer.start()

    def _set_batch_visible(self, visible: bool) -> None:
        self.batch_label.setVisible(visible)
        self.current_label.setVisible(visible)
        self.batch_bar.setVisible(visible)

    def _refresh_labels(self) -> None:
        if self._batch_total > 1:
            timer = _progress_timer_text(self._batch_started_at)
            self.batch_label.setText(f"{self._batch_base_text} · {timer}")
        detail_timer = _progress_timer_text(self._detail_started_at)
        self.detail_label.setText(f"{self._detail_base_text} · {detail_timer}")

    def _refresh_current_label(self) -> None:
        parts = []
        if self._rebin_total > 1:
            if self._batch_name and self._batch_kind != "binning":
                batch_name = self._batch_name
                rebin_suffix = f" · {self._rebin_name}"
                if self._rebin_name and batch_name.endswith(rebin_suffix):
                    batch_name = batch_name[: -len(rebin_suffix)]
                parts.append(f"Current {self._batch_kind}: {batch_name}")
            parts.append(
                f"{self._rebin_completed:,}/{self._rebin_total:,} rebins completed"
            )
            if self._rebin_name:
                parts.append(f"Current rebin: {self._rebin_name}")
        elif self._batch_name:
            parts.append(f"Current {self._batch_kind}: {self._batch_name}")
        self.current_label.setText(" · ".join(parts))

    def reset(self, title: str | None = None) -> None:
        from PySide6 import QtWidgets

        if title is not None:
            self._title = title
        now = time.monotonic()
        self._batch_started_at = now
        self._detail_started_at = now
        self._batch_completed = self._batch_total = 0
        self._detail_completed = self._detail_total = 0
        self._batch_name = self._batch_kind = ""
        self._batch_operation = ""
        self._batch_unit = "work item"
        self._rebin_name = ""
        self._rebin_completed = self._rebin_total = 0
        self._batch_base_text = self._title
        self._detail_base_text = self._title
        self._detail_start_key = None
        self._cancel_requested = False
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self._set_batch_visible(False)
        self.detail_bar.setRange(0, 0)
        self._refresh_labels()
        QtWidgets.QApplication.processEvents()

    def show(self) -> None:
        from PySide6 import QtWidgets

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        QtWidgets.QApplication.processEvents()

    def update_progress(self, event: dict[str, Any]) -> None:
        from PySide6 import QtWidgets

        if event.get("batch_name") is not None:
            self._batch_name = str(event["batch_name"])
            self._batch_kind = str(event.get("batch_kind") or "item")
            if not bool(event.get("batch_item_complete")):
                start_key = (
                    self._batch_name,
                    int(event.get("batch_completed") or 0),
                )
                if start_key != self._detail_start_key:
                    self._detail_start_key = start_key
                    self._detail_started_at = time.monotonic()
        if event.get("batch_operation") is not None:
            self._batch_operation = str(event["batch_operation"])
            self._batch_unit = str(event.get("batch_unit") or "work item")

        if event.get("rebin_name") is not None:
            self._rebin_name = str(event["rebin_name"])
        rebin_total = max(int(event.get("rebin_total") or 0), 0)
        if rebin_total:
            self._rebin_total = rebin_total
            self._rebin_completed = min(
                max(int(event.get("rebin_completed") or 0), 0),
                rebin_total,
            )

        batch_total = max(int(event.get("batch_total") or 0), 0)
        batch_completed = min(
            max(int(event.get("batch_completed") or 0), 0), batch_total
        )
        if batch_total:
            self._batch_total = batch_total
            self._batch_completed = batch_completed
            multi_item = batch_total > 1
            self._set_batch_visible(multi_item)
            if self._rebin_total > 1:
                self.current_label.setVisible(True)
            if multi_item:
                self.batch_bar.setRange(0, batch_total)
                self.batch_bar.setValue(batch_completed)
                percentage = 100.0 * batch_completed / batch_total
                if self._batch_operation:
                    unit = self._batch_unit
                    plural_unit = unit if batch_total == 1 else f"{unit}s"
                    self._batch_base_text = (
                        f"{self._batch_operation}: "
                        f"{batch_completed:,}/{batch_total:,} {plural_unit} completed "
                        f"({percentage:.1f}%)"
                    )
                else:
                    item_kind = self._batch_kind or "dataset group"
                    plural_kind = item_kind if batch_total == 1 else f"{item_kind}s"
                    self._batch_base_text = (
                        f"Rebinning {batch_total:,} {plural_kind}: "
                        f"{batch_completed:,}/{batch_total:,} {plural_kind} binned "
                        f"({percentage:.1f}%)"
                    )
        if self.current_label.isVisible():
            self._refresh_current_label()

        total = max(int(event.get("total") or 0), 0)
        iteration = max(int(event.get("iteration") or 0), 0)
        self._detail_total = total
        self._detail_completed = min(iteration, total) if total else 0
        message = str(event.get("message") or self._title)
        details = []
        output_bins = int(event.get("output_bins") or 0)
        if output_bins:
            details.append(f"{output_bins:,} output bins")
        workers = int(event.get("workers") or 0)
        if workers:
            details.append(f"{workers:,} CPU{'s' if workers != 1 else ''}")
        working_bytes = int(event.get("estimated_working_bytes") or 0)
        if working_bytes:
            details.append(f"~{working_bytes / 1024**2:.1f} MiB working memory")
        if total:
            batch_boundary = bool(event.get("batch_item_complete"))
            if total <= 2_000_000_000:
                self.detail_bar.setRange(0, total)
                detail_value = self._detail_completed
                if self._batch_total > 1 and not batch_boundary:
                    detail_value = min(detail_value, max(total - 1, 0))
                self.detail_bar.setValue(detail_value)
            else:
                # QProgressBar uses signed C++ integers.  Preserve smooth
                # progress for symmetry-expanded event counts beyond that
                # range by displaying a fixed-resolution fraction.
                self.detail_bar.setRange(0, 10_000)
                detail_value = round(10_000 * self._detail_completed / total)
                if self._batch_total > 1 and not batch_boundary:
                    detail_value = min(detail_value, 9_999)
                self.detail_bar.setValue(detail_value)
            percentage = 100.0 * self._detail_completed / total
            suffix = f"\n{' · '.join(details)}" if details else ""
            self._detail_base_text = f"{message} ({percentage:.1f}%){suffix}"
        else:
            if event.get("stage") == "rebin_batch":
                if event.get("batch_item_complete"):
                    self.detail_bar.setRange(0, 1)
                    self.detail_bar.setValue(1)
                else:
                    self.detail_bar.setRange(0, 0)
            self._detail_base_text = message
        self._refresh_labels()
        QtWidgets.QApplication.processEvents()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def _request_cancel(self) -> None:
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")
        self._detail_base_text = "Cancelling rebin at the next processing checkpoint"
        self._refresh_labels()
        if self._cancel_callback is not None:
            self._cancel_callback()

    def set_cancel_callback(self, callback: Any | None) -> None:
        """Forward cancellation to a worker when the rebin runs off-thread."""

        self._cancel_callback = callback

    def finish(self, message: str, **_kwargs: Any) -> None:
        self._detail_base_text = message
        if self._batch_total > 1 and self._batch_completed < self._batch_total:
            self._batch_completed = self._batch_total
            self.batch_bar.setValue(self._batch_total)
            if self._batch_operation:
                plural_unit = (
                    self._batch_unit
                    if self._batch_total == 1
                    else f"{self._batch_unit}s"
                )
                self._batch_base_text = (
                    f"{self._batch_operation}: {self._batch_total:,}/"
                    f"{self._batch_total:,} {plural_unit} completed (100.0%)"
                )
            else:
                item_kind = self._batch_kind or "dataset group"
                plural_kind = (
                    item_kind if self._batch_total == 1 else f"{item_kind}s"
                )
                self._batch_base_text = (
                    f"Rebinning {self._batch_total:,} {plural_kind}: "
                    f"{self._batch_total:,}/{self._batch_total:,} "
                    f"{plural_kind} binned (100.0%)"
                )
        if self._detail_total > 0:
            self._detail_completed = self._detail_total
            self.detail_bar.setValue(
                self._detail_total
                if self._detail_total <= 2_000_000_000
                else 10_000
            )
        self._refresh_labels()
        self.cancel_button.setEnabled(False)

    def fail(self, message: str) -> None:
        self._detail_base_text = f"Rebin failed: {message}"
        self._refresh_labels()
        self.cancel_button.setEnabled(False)

    def close(self) -> None:
        self._timer.stop()
        self.dialog.close()


class _FitProgressDialog:
    """Small live progress window for fits, analyses, and samplers."""

    def __init__(self, parent: Any) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        self.dialog = QtWidgets.QDialog(parent.window if hasattr(parent, "window") else parent)
        self.dialog.setWindowTitle("Fit progress")
        self.dialog.setModal(False)
        self.dialog.resize(720, 520)
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self.dialog)
        self.close_shortcut.activated.connect(self.dialog.close)
        layout = QtWidgets.QVBoxLayout(self.dialog)
        self.stage_label = QtWidgets.QLabel("Ready")
        stage_font = self.stage_label.font()
        stage_font.setBold(True)
        self.stage_label.setFont(stage_font)
        self.stage_label.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("No fit is running.")
        self.status_label.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("fit_stage_progress")
        self.progress.setToolTip(
            "Progress within the current stage, such as processed point contributions."
        )
        self.progress.setRange(0, 0)
        self.parameter_table = QtWidgets.QTableWidget(0, 2)
        self.parameter_table.setObjectName("fit_progress_parameter_table")
        self.parameter_table.setToolTip("Current parameter values reported by the active optimizer or sampler stage.")
        self.parameter_table.setHorizontalHeaderLabels(["Parameter", "Current value"])
        self.parameter_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameter_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.parameter_table.setAlternatingRowColors(True)
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.parameter_table.setMinimumHeight(120)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("fit_progress_log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setToolTip("Short live progress log. Current parameter values are shown in the table above.")
        self.panel_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.panel_splitter.setObjectName("fit_progress_panel_splitter")
        self.panel_splitter.setChildrenCollapsible(False)
        self.panel_splitter.addWidget(self.parameter_table)
        self.panel_splitter.addWidget(self.log)
        self.panel_splitter.setStretchFactor(0, 1)
        self.panel_splitter.setStretchFactor(1, 1)
        self.panel_splitter.setSizes([240, 220])
        self.cancel_button = QtWidgets.QPushButton("Terminate")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip(
            "Terminate the active optimization or posterior sampler at its next progress update. "
            "Least squares keeps the lowest-objective parameter set evaluated so far; if emcee "
            "has recorded samples, its partial chain is also saved for inspection or continuation."
        )
        self.cancel_button.clicked.connect(self._cancel_requested)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.setToolTip("Close this progress window after the fit pipeline finishes.")
        self.close_button.clicked.connect(self.dialog.close)
        self._cancel_callback: Any | None = None
        layout.addWidget(self.stage_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.panel_splitter, 1)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def reset(self, title: str = "Starting fit pipeline...") -> None:
        from PySide6 import QtWidgets

        self.stage_label.setText(title)
        self.status_label.setStyleSheet("")
        self.status_label.setText("Preparing data and fit problem.")
        self.progress.setRange(0, 0)
        self.parameter_table.setRowCount(0)
        self.log.clear()
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self._cancel_callback = None
        QtWidgets.QApplication.processEvents()

    def set_cancel_callback(self, callback: Any | None) -> None:
        self._cancel_callback = callback
        self.cancel_button.setEnabled(callback is not None)

    def _cancel_requested(self) -> None:
        if self._cancel_callback is None:
            return
        self._cancel_callback()
        self.status_label.setText("Termination requested. Waiting for the active step to stop...")
        self.cancel_button.setEnabled(False)

    def show(self) -> None:
        from PySide6 import QtWidgets

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        QtWidgets.QApplication.processEvents()

    def update_progress(self, event: dict[str, Any]) -> None:
        from PySide6 import QtWidgets

        stage = str(event.get("stage", "fit"))
        iteration = event.get("iteration", event.get("completed"))
        total = event.get("total")
        message = str(event.get("message", stage))
        stage_title = {
            "initialization": "Initialization: differential evolution",
            "bragg_integration": "Bragg integration",
            "rebin": "Rebinning data",
            "rebin_prepare": "Preparing rebin",
            "rebin_sources": "Preparing source data",
            "rebin_ready": "Rebin grid ready",
            "rebin_coverage": "Calculating coverage",
            "rebin_output": "Building rebinned dataset",
            "rebin_ui": "Refreshing viewers and controls",
            "rebin_finalize": "Finalizing rebin",
            "rebin_complete": "Rebin complete",
            "mdevent_events": "Binning MDEvents",
            "mdevent_normalization_setup": "Preparing detector normalization",
            "mdevent_normalization": "Integrating detector normalization",
            "mdevent_finalize": "Finalizing MDEvent reduction",
            "derived_source": "Preparing derived-dataset sources",
            "derived_operation": "Evaluating derived dataset",
            "viewer_prepare": "Opening data viewer",
            "least_squares": "Least-squares fit",
            "emcee": "Posterior sampling: emcee",
        }.get(stage, stage.replace("_", " ").title())
        self.stage_label.setText(stage_title)
        status_parts: list[str] = []
        datasets_total = event.get("datasets_total")
        datasets_completed = event.get("datasets_completed")
        if datasets_total is not None:
            datasets_total_value = max(int(datasets_total), 0)
            if datasets_completed is not None:
                status_parts.append(
                    f"Datasets {int(datasets_completed):,} of {datasets_total_value:,}"
                )
            else:
                status_parts.append(f"Datasets {datasets_total_value:,}")
        if event.get("output_bins") is not None:
            status_parts.append(f"Bins {int(event['output_bins']):,}")
        if event.get("workers") is not None:
            status_parts.append(f"CPUs {int(event['workers']):,}")
        if event.get("estimated_working_bytes") is not None:
            status_parts.append(
                f"Memory ~{int(event['estimated_working_bytes']) / 1024**2:.1f} MiB"
            )
        if iteration is not None:
            counter = (
                "Reflection" if stage == "bragg_integration"
                else "Points" if stage == "rebin"
                else "Step"
            )
            iteration_value = int(iteration)
            status_parts.append(
                f"{counter} {iteration_value:,}"
                + (f" of {int(total):,}" if total else "")
            )
        if event.get("accepted_count") is not None:
            status_parts.append(f"{int(event['accepted_count']):,} accepted")
        if event.get("rejected_count") is not None:
            status_parts.append(f"{int(event['rejected_count']):,} rejected")
        if event.get("cost") is not None:
            status_parts.append(f"cost {_format_number(float(event['cost']))}")
        if event.get("convergence") is not None:
            status_parts.append(f"convergence {_format_number(float(event['convergence']))}")
        if event.get("seconds_per_step") is not None:
            status_parts.append(f"{_format_seconds_per_step(float(event['seconds_per_step']))}/step")
        self.status_label.setText(" | ".join(status_parts) if status_parts else message)
        params = event.get("parameters")
        if isinstance(params, dict) and params:
            self._set_parameters(params)
        log_parts = [stage_title]
        if datasets_total is not None:
            log_parts.append(
                f"datasets {int(datasets_completed):,}/{int(datasets_total):,}"
                if datasets_completed is not None
                else f"{int(datasets_total):,} datasets"
            )
        if event.get("output_bins") is not None:
            log_parts.append(f"{int(event['output_bins']):,} bins")
        if event.get("workers") is not None:
            log_parts.append(f"{int(event['workers']):,} CPUs")
        if event.get("estimated_working_bytes") is not None:
            log_parts.append(
                f"~{int(event['estimated_working_bytes']) / 1024**2:.1f} MiB"
            )
        if iteration is not None:
            counter = (
                "reflection" if stage == "bragg_integration"
                else "points" if stage == "rebin"
                else "step"
            )
            iteration_value = int(iteration)
            log_parts.append(
                f"{counter} {iteration_value:,}"
                + (f"/{int(total):,}" if total else "")
            )
        if message and message != stage:
            log_parts.append(message)
        if event.get("accepted_count") is not None:
            log_parts.append(f"{int(event['accepted_count']):,} accepted")
        if event.get("rejected_count") is not None:
            log_parts.append(f"{int(event['rejected_count']):,} rejected")
        if event.get("cost") is not None:
            log_parts.append(f"cost {_format_number(float(event['cost']))}")
        if event.get("seconds_per_step") is not None:
            log_parts.append(f"{_format_seconds_per_step(float(event['seconds_per_step']))}/step")
        self.log.appendPlainText(" | ".join(log_parts))
        if total and iteration is not None:
            self.progress.setRange(0, int(total))
            self.progress.setValue(min(int(iteration), int(total)))
        else:
            self.progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

    def _set_parameters(self, params: dict[str, Any], limit_hits: Any = None) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        items = list(params.items())
        hits_by_name = _fit_limit_hits_from_goodness({"parameters_at_limits": limit_hits})
        self.parameter_table.setRowCount(len(items))
        for row, (name, value) in enumerate(items):
            name_item = QtWidgets.QTableWidgetItem(str(name))
            value_text = _format_number(value)
            value_item = QtWidgets.QTableWidgetItem(value_text)
            name_item.setToolTip(str(name))
            value_item.setToolTip(value_text)
            hit = hits_by_name.get(str(name))
            if hit is not None:
                warning = f"Reached the {hit.get('side', 'configured')} bound ({_format_number(hit.get('bound'))})."
                for item in (name_item, value_item):
                    item.setForeground(QtGui.QColor("#c0392b"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setToolTip(f"{item.toolTip()}\n{warning}")
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.parameter_table.setItem(row, 0, name_item)
            self.parameter_table.setItem(row, 1, value_item)
        self.parameter_table.resizeColumnsToContents()

    def finish(
        self,
        message: str,
        *,
        parameters: dict[str, Any] | None = None,
        limit_hits: Any = None,
        summary_lines: Sequence[str] | None = None,
    ) -> None:
        from PySide6 import QtWidgets

        self.stage_label.setText(message)
        if parameters:
            self._set_parameters(parameters, limit_hits)
        warning = _fit_limit_warning_text(limit_hits)
        self.status_label.setStyleSheet("color: #c0392b; font-weight: bold;" if warning else "")
        self.status_label.setText(f"Done. {warning}" if warning else "Done.")
        if warning:
            self.log.appendPlainText(warning)
        for line in summary_lines or ():
            if str(line).strip():
                self.log.appendPlainText(str(line))
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.cancel_button.setEnabled(False)
        self._cancel_callback = None
        self.close_button.setEnabled(True)
        QtWidgets.QApplication.processEvents()

    def fail(self, message: str) -> None:
        """Show a fit failure and keep the window open for the user to read."""

        from PySide6 import QtWidgets

        self.stage_label.setText("Fit failed")
        self.status_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        self.status_label.setText(message)
        self.log.appendPlainText(f"ERROR: {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_button.setEnabled(False)
        self._cancel_callback = None
        self.close_button.setEnabled(True)
        self.show()
        QtWidgets.QApplication.processEvents()

    def close(self) -> None:
        self.dialog.close()


class NfitProjectExplorer:
    """PySide6 project explorer for building nfit analysis pipelines."""

    def __init__(self, project: NfitProject | None = None) -> None:
        self.app = _qt_app()
        self.project = _new_gui_project() if project is None else project
        stored_path = getattr(self.project, "_project_path", None)
        self.project_path: Path | None = (
            None if stored_path is None else Path(stored_path)
        )
        self._saved_project_size_bytes = _project_file_size(self.project_path)
        set_active_project_path(self.project_path)
        self._project_disk_signature: tuple[int, int, int, int] | None = None
        self._ignored_project_disk_signature: tuple[int, int, int, int] | None = None
        self._external_change_timer = None
        self.has_unsaved_changes = False
        self._selected_dataset_binning_ids: dict[str, str] = {}
        self._selected_composite_binning_ids: dict[str, str] = {}
        self._allow_window_close = False
        self.window = None
        self.file_menu = None
        self.recent_projects_menu = None
        self.reload_project_action = None
        self.cache_binnings_action = None
        self.tree = None
        self.title_label = None
        self.enabled_check = None
        self.fit_weight_widget = None
        self.fit_weight_spin = None
        self.scale_factor_spin = None
        self.dataset_temperature_spin = None
        self.dataset_kf_ki_included_check = None
        self.sample_environment_widget = None
        self.dataset_field_magnitude_spin = None
        self.dataset_field_frame_combo = None
        self.dataset_field_direction_edit = None
        self.group_bulk_widget = None
        self.group_fit_weight_edit = None
        self.group_scale_edit = None
        self.group_scale_fit_check = None
        self.background_bulk_widget = None
        self.background_bulk_scale_edit = None
        self.details_label = None
        self.details_scroll = None
        self.details_widget = None
        self.details_layout = None
        self.import_dataset_button = None
        self.add_model_button = None
        self.new_analysis_button = None
        self.view_slice_button = None
        self.load_dataset_button = None
        self.reload_data_button = None
        self.add_mask_button = None
        self.add_background_button = None
        self.add_dataset_group_button = None
        self.save_dataset_button = None
        self.mask_type_combo = None
        self.mask_parameter_widget = None
        self.mask_parameter_layout = None
        self.mask_application_status_label = None
        self.model_selector_widget = None
        self.model_category_combo = None
        self.model_type_combo = None
        self.model_parameter_scroll = None
        self.model_parameter_widget = None
        self.model_parameter_layout = None
        self.fit_editor_widget = None
        self.fit_settings_panel = None
        self.fit_optimizer_combo = None
        self.fit_optimizer_config_editor = None
        self.fit_loss_combo = None
        self.fit_f_scale_spin = None
        self.fit_finite_difference_workers_spin = None
        self.fit_de_check = None
        self.fit_de_maxiter_spin = None
        self.fit_de_popsize_spin = None
        self.fit_de_workers_spin = None
        self.fit_emcee_check = None
        self.fit_emcee_walkers_spin = None
        self.fit_emcee_steps_spin = None
        self.fit_emcee_burn_spin = None
        self.fit_emcee_thin_spin = None
        self.fit_emcee_seed_spin = None
        self.fit_emcee_workers_spin = None
        self.fit_posterior_layout = None
        self.fit_posterior_result_actions = None
        self.fit_branch_check = None
        self.fit_now_button = None
        self.fit_corner_button = None
        self.show_data_fit_button = None
        self.fit_export_report_button = None
        self.fit_copy_script_button = None
        self.fit_save_script_button = None
        self.plot_open_button = None
        self.plot_edit_button = None
        self.plot_copy_script_button = None
        self.plot_save_script_button = None
        self._fit_progress_dialog: _FitProgressDialog | None = None
        self._fit_worker_thread = None
        self._fit_worker = None
        self._fit_worker_handler = None
        self._fit_disabled_widget_states: list[tuple[Any, bool]] = []
        self.expand_all_button = None
        self.collapse_all_button = None
        self.create_group_button = None
        self.delete_button = None
        self._clipboard: tuple[
            str,
            DatasetEntry | DatasetGroup | MaskSpec | list[DatasetEntry],
        ] | None = None
        self._analysis_window = None
        self._slice_viewers: dict[int, list[Any]] = {}
        self._auxiliary_windows: dict[int, Any] = {}
        self._overlay_refresh_timer = None
        self._pending_overlay_groups: dict[int, DataGroup] = {}
        self._compressed_cache_prompt = None
        # True only while the Qt event loop is running (set in run()); in
        # headless/test use it stays False so overlay refreshes are synchronous.
        self._interactive = False
        self._restoring_fit_selection = False
        self._active_fit_group: DataGroup | None = None
        self._active_fit_anchor: FitTimelineEntry | None = None
        self._active_fit_current: FitTimelineEntry | None = None
        self._active_branch_current: FitTimelineEntry | None = None
        self._item_roles: dict[
            int,
            tuple[
                str,
                DataGroup | None,
                DatasetEntry | None,
                MaskSpec | None,
                ModelComponentSpec | None,
            ],
        ] = {}
        self._fit_item_roles: dict[int, FitTimelineEntry] = {}
        self._background_item_roles: dict[int, BackgroundSpec] = {}
        self._analysis_output_roles: dict[int, AnalysisOutputRef] = {}
        self._analysis_item_roles: dict[int, AnalysisEntry] = {}
        self._plot_item_roles: dict[int, PlotEntry] = {}
        self._plot_windows: dict[str, Any] = {}
        self._dataset_group_roles: dict[int, DatasetGroup] = {}
        self._dataset_page_roles: dict[
            int,
            tuple[DataGroup | DatasetGroup, tuple[DatasetEntry, ...], int],
        ] = {}
        self._expanded_state: dict[tuple[Any, ...], bool] = {}
        self._build()
        self._refresh_tree()
        self._sync_details()
        self._apply_initial_window_size()
        configure_numeric_spin_boxes(self.app)

    def _apply_initial_window_size(self) -> None:
        """Apply a roomy startup size capped to the screen under the cursor."""

        from PySide6 import QtGui

        central = self.window.centralWidget()
        if central is not None:
            for index in range(central.count()):
                layout = central.widget(index).layout()
                if layout is not None:
                    layout.activate()
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if screen is None:
            screen = self.window.screen() or self.app.primaryScreen()
        if screen is None:
            size = PROJECT_WINDOW_TARGET_SIZE
        else:
            available = screen.availableGeometry()
            size = _screen_aware_project_window_size(
                available.width(),
                available.height(),
            )
        self.window.resize(*size)

    def show(self) -> NfitProjectExplorer:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self

    def run(self) -> int:
        self.show()
        splash = getattr(self.app, "_nfit_startup_splash", None)
        if splash is not None:
            splash.close()
            self.app._nfit_startup_splash = None
        self._interactive = True
        from .project_cache_gui import CompressedCachePrompt

        cache_prompt = CompressedCachePrompt(self.window)
        self._compressed_cache_prompt = cache_prompt
        _COMPOSITE_DATA_CACHE.before_discard = cache_prompt.request
        _VIEWER_VIEW_CACHE.before_discard = cache_prompt.request
        self._update_controller.schedule_startup()
        if self._external_change_timer is not None:
            self._external_change_timer.start()
        interrupt_timer, previous_interrupt_handler = _install_cli_interrupt_handler(self.app)
        try:
            return int(self.app.exec())
        except KeyboardInterrupt:
            self.app.exit(130)
            return 130
        finally:
            _COMPOSITE_DATA_CACHE.before_discard = None
            _VIEWER_VIEW_CACHE.before_discard = None
            _COMPOSITE_DATA_CACHE.clear_disk_cache()
            _VIEWER_VIEW_CACHE.clear_disk_cache()
            cache_prompt.cleanup()
            self._compressed_cache_prompt = None
            if self._external_change_timer is not None:
                self._external_change_timer.stop()
            if interrupt_timer is not None:
                interrupt_timer.stop()
            _restore_cli_interrupt_handler(previous_interrupt_handler)

    def create_data_group(self) -> DataGroup:
        group = create_data_group(self.project)
        self._mark_dirty()
        self._refresh_tree(select_group=group, edit_group=True)
        return group

    def import_dataset_paths(
        self,
        group: DataGroup,
        paths: list[str | Path],
        *,
        data_type: str | None = None,
        importer_name: str | None = None,
        importer_options: dict[str, dict[str, Any]] | None = None,
        into: DatasetGroup | None = None,
        stream_group_mode: str = "reuse",
    ) -> list[DatasetEntry]:
        from PySide6 import QtWidgets

        progress = self._make_rebin_progress_callback("Loading datasets...")
        progress({"stage": "import", "iteration": 0, "total": 0, "message": "opening dataset files"})
        try:
            entries = import_dataset_paths(
                group,
                paths,
                data_type=data_type,
                importer_name=importer_name,
                importer_options=importer_options,
                into=into,
                stream_group_mode=stream_group_mode,
                progress_callback=progress,
            )
        except RebinCancellationRequested:
            return []
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Import datasets",
                f"Could not import datasets:\n{exc}",
            )
            return []
        finally:
            self._close_rebin_progress(progress)
        if entries:
            self._record_data_group_state_change(group)
            self._evaluate_model_after_dataset_activation(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group)
        return entries

    def _request_dataset_import(
        self,
        group: DataGroup,
        paths: list[str | Path],
        *,
        data_type: str | None = None,
        importer_name: str | None = None,
        importer_options: dict[str, dict[str, Any]] | None = None,
        into: DatasetGroup | None = None,
        stream_group_mode: str = "reuse",
    ) -> bool:
        """Import from GUI actions without blocking the Qt event loop."""

        if not self._interactive:
            return bool(
                self.import_dataset_paths(
                    group,
                    paths,
                    data_type=data_type,
                    importer_name=importer_name,
                    importer_options=importer_options,
                    into=into,
                    stream_group_mode=stream_group_mode,
                )
            )

        def task(progress_callback: Any) -> DataGroup:
            staging = DataGroup("Import staging")
            import_dataset_paths(
                staging,
                paths,
                data_type=data_type,
                importer_name=importer_name,
                importer_options=importer_options,
                stream_group_mode="new",
                progress_callback=progress_callback,
            )
            return staging

        def on_success(staging: DataGroup) -> None:
            parent = into if into is not None else group
            for dataset in list(staging.datasets):
                dataset.name = _unique_dataset_name(dataset.name, group.dataset_names)
                group.add_dataset(dataset, into=into)
            existing_groups = {subgroup.name for subgroup in parent.subgroups}
            for subgroup in list(staging.subgroups):
                target = None
                identity = _dataset_group_import_stream(subgroup)
                if stream_group_mode == "reuse" and identity is not None:
                    target = next(
                        (
                            candidate
                            for candidate in parent.subgroups
                            if _dataset_group_import_stream(candidate) == identity
                        ),
                        None,
                    )
                if target is not None:
                    target.metadata.setdefault("importer", identity[0])
                    target.metadata.setdefault("source_stream", identity[1])
                    for dataset in list(subgroup.datasets):
                        dataset.name = _unique_dataset_name(
                            dataset.name, group.dataset_names
                        )
                        group.add_dataset(dataset, into=target)
                        _adopt_imported_crystal(group, dataset, target)
                    target.subgroups.extend(subgroup.subgroups)
                    target.masks.extend(subgroup.masks)
                    if GROUP_COMPOSITE_KEY in target.metadata:
                        data_group_composite_config(target)["stale"] = True
                    continue
                for dataset in subgroup.iter_datasets():
                    dataset.name = _unique_dataset_name(
                        dataset.name, group.dataset_names
                    )
                subgroup.name = _unique_name(subgroup.name, existing_groups)
                existing_groups.add(subgroup.name)
                parent.subgroups.append(subgroup)
            if any(True for _dataset in staging.iter_datasets()):
                self._record_data_group_state_change(group)
                self._evaluate_model_after_dataset_activation(group)
                self._mark_dirty()
                self._refresh_tree(select_group=group, select_dataset_group=into)

        return self._start_background_task(
            title="Loading datasets...",
            failure_title="Import datasets",
            task=task,
            on_success=on_success,
            success_message="Dataset import finished.",
        )

    def _dataset_importing_config(self, group: DataGroup) -> dict[str, Any]:
        config = group.metadata.get("dataset_importing")
        if not isinstance(config, dict):
            config = {}
            group.metadata["dataset_importing"] = config
        config.setdefault("enabled", False)
        config.setdefault("path", "")
        config.setdefault("prefix", "")
        config.setdefault("suffix", "")
        config.setdefault("numors", "")
        return config

    def _set_dataset_importing_enabled(self, group: DataGroup, enabled: bool) -> None:
        config = self._dataset_importing_config(group)
        if bool(config["enabled"]) == bool(enabled):
            return
        config["enabled"] = bool(enabled)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._sync_details()

    def _set_dataset_importing_text(self, group: DataGroup, key: str, text: str) -> None:
        config = self._dataset_importing_config(group)
        value = str(text).strip()
        if config.get(key) == value:
            return
        config[key] = value
        self._record_data_group_state_change(group)
        self._mark_dirty()

    def _add_dataset_import_files(self, group: DataGroup) -> None:

        paths, _selected_filter = get_open_file_names(
            self.window, "Add datasets", "", "Data files (*);;All files (*)"
        )
        if paths:
            self._request_dataset_import(group, paths, data_type=DEFAULT_DATA_TYPE)

    def _import_dataset_importing_range(self, group: DataGroup) -> None:
        from PySide6 import QtWidgets

        config = self._dataset_importing_config(group)
        try:
            numors = parse_dataset_numors(config.get("numors", ""))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Import datasets", str(exc))
            return
        directory = Path(str(config.get("path", "")).strip()).expanduser()
        prefix = str(config.get("prefix", ""))
        suffix = str(config.get("suffix", ""))
        if not directory.is_dir():
            QtWidgets.QMessageBox.warning(self.window, "Import datasets", f"Dataset directory does not exist:\n{directory}")
            return
        paths = [directory / f"{prefix}{numor}{suffix}" for numor in numors]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            shown = ", ".join(missing[:10])
            suffix_text = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
            QtWidgets.QMessageBox.warning(
                self.window, "Import datasets",
                f"The requested files do not exist:\n{shown}{suffix_text}",
            )
            return
        self._request_dataset_import(group, paths, data_type=DEFAULT_DATA_TYPE)

    def _clear_imported_datasets(self, group: DataGroup) -> None:
        from PySide6 import QtWidgets

        dataset_count = sum(1 for _dataset in group.iter_datasets())
        if dataset_count == 0:
            return
        answer = QtWidgets.QMessageBox.question(
            self.window,
            "Clear datasets",
            f"Remove all {dataset_count} datasets and nested dataset groups from {group.name!r}?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        group.datasets.clear()
        group.subgroups.clear()
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group)

    def selected_data_group(self) -> DataGroup | None:
        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"group", "datasets", "models", "dataset", "fits", "fit", "fit_timeline", "dataset_group"}:
            return group
        return None

    def _selected_import_target(self) -> tuple[DataGroup | None, DatasetGroup | None]:
        """Return the (data group, nested group) that an import should populate."""

        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if role == "dataset_group":
            return group, self._dataset_group_for_item(item)
        if role in {"group", "datasets", "models", "dataset", "fits", "fit", "fit_timeline"}:
            return group, None
        return None, None

    def delete_selected(self) -> None:
        items = self._selected_deletable_items()
        if not items:
            return
        changed = False
        affected_groups: list[DataGroup] = []
        for item in items:
            did_change, group = self._delete_tree_item(item)
            if did_change:
                changed = True
                if group is not None and group not in affected_groups:
                    affected_groups.append(group)
        if not changed:
            return
        self._mark_dirty()
        select_group = affected_groups[0] if len(affected_groups) == 1 else None
        self._refresh_tree(select_group=select_group)

    def _delete_analysis_entry(
        self,
        group: DataGroup,
        analysis: AnalysisEntry,
    ) -> bool:
        """Delete an analysis recipe and datasets explicitly derived from it."""

        if analysis not in group.analyses:
            return False
        group.analyses.remove(analysis)
        for collection in (group, *group.iter_subgroups()):
            collection.datasets[:] = [
                dataset
                for dataset in collection.datasets
                if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id")
                != analysis.id
            ]
        return True

    def _is_multi_select_gesture(self) -> bool:
        """True while the user extends a selection with shift/ctrl/cmd.

        Selecting a single fit restores its snapshot into the live model, which
        rebuilds the tree. During a range or toggle selection that rebuild would
        destroy the very multi-selection the user is building (for example to
        delete several fit results at once), so the restore is suppressed while
        a modifier is held.
        """

        from PySide6 import QtCore, QtWidgets

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        extend = (
            QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        return bool(modifiers & extend)

    def _prune_tree_selection(self) -> None:
        """Keep multi-selections homogeneous by role.

        The tree uses ``ExtendedSelection``, so a shift-click selects every
        visible row between the anchor and the click, which otherwise sweeps in
        the workspace root and the ``Datasets``/``Models``/``Fits`` folder
        headers. Restricting the selection to items sharing the current (just
        clicked) item's role lets the user grab a contiguous run of fits,
        datasets, masks, or models without dragging in unrelated rows.
        """

        selected = self.tree.selectedItems()
        if len(selected) <= 1:
            return
        current = self._current_item()
        current_role = self._objects_for_item(current)[4]
        keep_roles = _DELETABLE_ROLE_GROUP.get(current_role, {current_role})
        stale = [item for item in selected if self._objects_for_item(item)[4] not in keep_roles]
        if not stale:
            return
        self.tree.blockSignals(True)
        try:
            for item in stale:
                item.setSelected(False)
        finally:
            self.tree.blockSignals(False)

    def _selected_deletable_items(self) -> list[Any]:
        """Selected tree items that can be deleted, restricted to one role.

        Range/multi selection can span several roles; a batch delete only makes
        sense within a single role, so the returned items all share the current
        item's role (falling back to the current item when nothing else is
        selected).
        """

        current = self._current_item()
        current_role = self._objects_for_item(current)[4]
        if current_role not in _DELETABLE_TREE_ROLES:
            return []
        keep_roles = _DELETABLE_ROLE_GROUP.get(current_role, {current_role})
        items = [
            item
            for item in self.tree.selectedItems()
            if self._objects_for_item(item)[4] in keep_roles
        ]
        if current is not None and current not in items:
            items.append(current)
        return items

    def _delete_tree_item(self, item: Any) -> tuple[bool, DataGroup | None]:
        """Delete a single tree item's object without refreshing the tree.

        Returns ``(changed, group_to_reselect)``; ``group_to_reselect`` is the
        data group the caller should reselect after refreshing, or ``None`` when
        no selection should be restored (for example after deleting a whole
        workspace).
        """

        group, entry, mask, model, role = self._objects_for_item(item)
        if role == "group" and group is not None:
            delete_data_group(self.project, group)
            self._close_slice_viewer(group)
            if self._active_fit_group is group:
                self._clear_active_fit_state()
            return True, None
        if role == "dataset" and group is not None and entry is not None:
            delete_dataset(group, entry)
            self._record_data_group_state_change(group)
            return True, group
        if role == "mask" and group is not None and entry is not None and mask is not None:
            delete_mask(entry, mask)
            self._mark_mask_datasets_stale([entry])
            self._record_data_group_state_change(group)
            return True, group
        if role == "background" and group is not None and entry is not None:
            background = self._background_for_item(item)
            if background is not None and background in entry.backgrounds:
                entry.backgrounds.remove(background)
                self._record_data_group_state_change(group)
                return True, group
        if role == "group_background" and group is not None:
            owner = self._background_owner_for_item(item)
            background = self._background_for_item(item)
            if owner is not None and background is not None and background in owner.backgrounds:
                owner.backgrounds.remove(background)
                self._record_data_group_state_change(group)
                return True, group
        if role == "dataset_group" and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None and delete_dataset_group(group, subgroup):
                self._record_data_group_state_change(group)
                return True, group
            return False, None
        if role == "group_mask" and group is not None and mask is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None and mask in subgroup.masks:
                self._mark_mask_datasets_stale(list(subgroup.iter_datasets()))
                subgroup.masks.remove(mask)
                self._record_data_group_state_change(group)
                return True, group
            return False, None
        if role == "model" and group is not None and model is not None:
            delete_model_component(group, model)
            self._record_data_group_state_change(group)
            return True, group
        if role in {"fit", "fit_timeline"} and group is not None:
            fit_entry = self._fit_entry_for_item(item)
            if fit_entry is not None and delete_fit_entry(group, fit_entry):
                return True, group
            return False, None
        if role == "analysis" and group is not None:
            analysis = self._analysis_item_roles.get(id(item))
            if analysis is not None and self._delete_analysis_entry(group, analysis):
                if (
                    self._analysis_window is not None
                    and self._analysis_window.group is group
                ):
                    self._analysis_window._refresh_analysis_list()
                    self._analysis_window._clear_result_views()
                return True, group
            return False, None
        if role == "plot" and group is not None:
            plot = self._plot_for_item(item)
            if plot is not None and plot in group.plots:
                group.plots.remove(plot)
                self._plot_windows.pop(plot.id, None)
                return True, group
        return False, None

    def new_project(self) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        self._close_all_slice_viewers()
        self.project = _new_gui_project()
        self.project_path = None
        self._saved_project_size_bytes = None
        set_active_project_path(None)
        self._project_disk_signature = None
        self._ignored_project_disk_signature = None
        self.has_unsaved_changes = False
        self._clear_active_fit_state()
        self._sync_cache_binnings_action()
        self._refresh_tree()
        self._sync_window_title()
        return True

    def close_project(self) -> bool:
        if self.project_path is None:
            return self.quit_application()
        return self.new_project()

    def quit_application(self) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        self._close_all_slice_viewers()
        self._allow_window_close = True
        self.window.close()
        self._allow_window_close = False
        self.app.quit()
        return True

    def load_dataset_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        if self._interactive:
            def task(progress_callback: Any) -> Any:
                return dataset_for_slice_viewer(
                    entry,
                    progress_callback=progress_callback,
                )

            def on_success(loaded: Any) -> None:
                if loaded is not None:
                    self._sync_details()
                    if group is not None:
                        self.refresh_slice_viewer(group)

            return self._start_background_task(
                title="Loading dataset...",
                failure_title="Load dataset",
                task=task,
                on_success=on_success,
                success_message="Dataset load finished.",
            )
        try:
            loaded = dataset_for_slice_viewer(entry)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Load dataset",
                f"Could not load dataset:\n{exc}",
            )
            return False
        if loaded is None:
            return False
        self._sync_details()
        if group is not None:
            self.refresh_slice_viewer(group)
        return True

    def reload_data_for_selection(self) -> bool:
        """Reload the selected dataset or dataset collection from its sources."""

        from PySide6 import QtWidgets

        item = self._current_item()
        group, entry, _mask, _model, role = self._objects_for_item(item)
        if group is None or role not in {"group", "datasets", "dataset", "dataset_group"}:
            return False
        target = entry if role == "dataset" else (
            self._dataset_group_for_item(item) if role == "dataset_group" else group
        )
        if target is None:
            return False

        def task(_progress_callback: Any) -> list[DatasetEntry]:
            if isinstance(target, DatasetEntry):
                reload_dataset_data(target)
                return [target]
            return reload_data_group(target)

        def on_success(reloaded: list[DatasetEntry]) -> None:
            for scope in _composite_scopes(group):
                data_group_composite_config(scope)["stale"] = True
            self._refresh_cache_badges()
            self._sync_details()
            self.refresh_slice_viewer(group, force_rebin=True)

        if self._interactive:
            return self._start_background_task(
                title="Reloading data...",
                failure_title="Reload data",
                task=task,
                on_success=on_success,
                success_message="Data reload finished.",
                completion_summary=lambda items: [f"Reloaded datasets: {len(items):,}"],
            )
        try:
            reloaded = task(None)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Reload data",
                f"Could not reload data:\n{exc}",
            )
            return False
        on_success(reloaded)
        return True

    def _stamp_active_fit_path(self) -> None:
        """Refresh the active group's stored fit path before persisting."""

        group = self._active_fit_group
        if group is None or group not in self.project.data_groups:
            return
        group.active_fit_path = _fit_entry_path(group.fits, self._active_fit_entry(group))

    def _update_project_disk_signature(self) -> None:
        self._project_disk_signature = _project_file_signature(self.project_path)
        self._ignored_project_disk_signature = None

    def _project_changed_on_disk(self) -> bool:
        if self.project_path is None or self._project_disk_signature is None:
            return False
        current = _project_file_signature(self.project_path)
        return current is not None and current != self._project_disk_signature

    def _prompt_external_change_action(self) -> str:
        """Ask how to handle a project changed by another process."""

        from PySide6 import QtWidgets

        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Project changed on disk")
        message.setText(
            "The current nfit project changed on disk after it was opened."
        )
        discard_note = (
            " Reloading will discard unsaved in-memory changes."
            if self.has_unsaved_changes
            else ""
        )
        message.setInformativeText(
            "Reload the external version, save this in-memory state under a "
            "different name, or keep working without changing either copy."
            + discard_note
        )
        reload_button = message.addButton(
            "Reload from Disk", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        save_as_button = message.addButton(
            "Save As…", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        message.addButton(
            "Keep Current State", QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        message.setDefaultButton(reload_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is reload_button:
            return "reload"
        if clicked is save_as_button:
            return "save_as"
        return "keep"

    def _prompt_save_conflict_action(self) -> str:
        """Ask whether an explicit Save may overwrite an external change."""

        from PySide6 import QtWidgets

        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Project changed on disk")
        message.setText(
            "Saving now would overwrite a project version written by another process."
        )
        reload_button = message.addButton(
            "Reload from Disk", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        save_as_button = message.addButton(
            "Save As…", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        overwrite_button = message.addButton(
            "Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_button = message.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        message.setDefaultButton(cancel_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is reload_button:
            return "reload"
        if clicked is save_as_button:
            return "save_as"
        if clicked is overwrite_button:
            return "overwrite"
        return "cancel"

    def _prompt_reload_unsaved_action(self) -> str:
        """Ask how to preserve unsaved state before an explicit reload."""

        from PySide6 import QtWidgets

        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Reload project from disk")
        message.setText("Reloading will discard unsaved in-memory project changes.")
        reload_button = message.addButton(
            "Reload and Discard", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        save_as_button = message.addButton(
            "Save As…", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        cancel_button = message.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        message.setDefaultButton(cancel_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is reload_button:
            return "reload"
        if clicked is save_as_button:
            return "save_as"
        return "cancel"

    def _load_project_path(self, path: Path, *, remember: bool) -> bool:
        self._close_all_slice_viewers()
        self.project = load_project(path)
        self.project_path = path
        self._saved_project_size_bytes = _project_file_size(path)
        set_active_project_path(path)
        self.has_unsaved_changes = False
        self._clear_active_fit_state()
        self._sync_cache_binnings_action()
        if remember:
            self._remember_recent_project(path)
        self._refresh_tree()
        self._restore_active_fit_selection()
        self._update_project_disk_signature()
        self._sync_window_title()
        return True

    def _set_cache_binnings_enabled(self, enabled: bool) -> None:
        """Set the project-specific persisted-binning preference."""

        value = bool(enabled)
        if project_cache_binnings_enabled(self.project) == value:
            return
        self.project.settings[PROJECT_CACHE_BINNINGS_KEY] = value
        if not value:
            self.project.settings.pop(PROJECT_BINNING_CACHE_ENTRIES_KEY, None)
        self._mark_dirty()

    def _sync_cache_binnings_action(self) -> None:
        action = self.cache_binnings_action
        if action is None:
            return
        action.blockSignals(True)
        try:
            action.setChecked(project_cache_binnings_enabled(self.project))
        finally:
            action.blockSignals(False)

    def reload_project_from_disk(self) -> bool:
        """Reload the current archive, confirming before discarding GUI edits."""

        if self.project_path is None or not self.project_path.exists():
            return False
        path = self.project_path
        if self.has_unsaved_changes:
            action = self._prompt_reload_unsaved_action()
            if action == "cancel":
                return False
            if action == "save_as" and not self.save_as():
                return False
        return self._load_project_path(path, remember=True)

    def check_for_external_project_change(self) -> bool:
        """Prompt once when the open project has been replaced on disk."""

        if not self._project_changed_on_disk():
            return False
        current = _project_file_signature(self.project_path)
        if current == self._ignored_project_disk_signature:
            return False
        action = self._prompt_external_change_action()
        if action == "reload":
            return self._load_project_path(self.project_path, remember=True)
        if action == "save_as":
            return self.save_as()
        self._ignored_project_disk_signature = current
        return True

    def save(self) -> bool:
        if self.project_path is None:
            return self.save_as()
        if self._project_changed_on_disk():
            action = self._prompt_save_conflict_action()
            if action == "reload":
                return self._load_project_path(self.project_path, remember=True)
            if action == "save_as":
                return self.save_as()
            if action != "overwrite":
                return False
        if (
            project_cache_binnings_enabled(self.project)
            and project_binnings_need_refresh(self.project)
            and not self._confirm_rebin_cache_memory(
                self._pending_project_rebin_cache_bytes(),
                operation="Saving the project",
            )
        ):
            return False
        self._stamp_active_fit_path()
        progress = (
            self._make_rebin_progress_callback(
                "Updating cached binnings...",
                aggregate=True,
            )
            if project_cache_binnings_enabled(self.project)
            and project_binnings_need_refresh(self.project)
            else None
        )
        try:
            save_project(
                self.project,
                self.project_path,
                progress_callback=progress,
            )
        except RebinCancellationRequested:
            return False
        finally:
            self._close_rebin_progress(progress)
        self.has_unsaved_changes = False
        self._update_project_disk_signature()
        self._saved_project_size_bytes = _project_file_size(self.project_path)
        self._sync_window_title()
        self._sync_details()
        return True

    def save_as(self) -> bool:

        path, _selected_filter = get_save_file_name(
            self.window,
            "Save nfit project",
            "nfit_project.nfit",
            "nfit projects (*.nfit);;All files (*)",
        )
        if not path:
            return False
        if (
            project_cache_binnings_enabled(self.project)
            and project_binnings_need_refresh(self.project)
            and not self._confirm_rebin_cache_memory(
                self._pending_project_rebin_cache_bytes(),
                operation="Saving the project",
            )
        ):
            return False
        old_path = self.project_path
        new_path = Path(path)
        self._stamp_active_fit_path()
        progress = (
            self._make_rebin_progress_callback(
                "Updating cached binnings...",
                aggregate=True,
            )
            if project_cache_binnings_enabled(self.project)
            and project_binnings_need_refresh(self.project)
            else None
        )
        try:
            save_project(
                self.project,
                new_path,
                asset_source=old_path,
                progress_callback=progress,
            )
        except RebinCancellationRequested:
            return False
        finally:
            self._close_rebin_progress(progress)
        self.project_path = new_path
        set_active_project_path(new_path)
        self._remember_recent_project(self.project_path)
        self.has_unsaved_changes = False
        self._update_project_disk_signature()
        self._saved_project_size_bytes = _project_file_size(self.project_path)
        self._sync_window_title()
        self._sync_details()
        return True

    def open_project(self) -> bool:

        path, _selected_filter = get_open_file_name(
            self.window,
            "Open nfit project",
            "",
            "nfit projects (*.nfit);;All files (*)",
        )
        if not path:
            return False
        return self.open_project_path(path)

    def open_project_path(self, path: str | Path, *, remember: bool = True) -> bool:
        if not self._confirm_save_before_closing_project():
            return False
        path = Path(path)
        if not path.exists():
            forget_missing_recent_projects()
            self._refresh_recent_projects_menu()
            return False
        return self._load_project_path(path, remember=remember)

    def import_dataset_dialog(self) -> None:

        group, into = self._selected_import_target()
        if group is None:
            return
        paths, _selected_filter = get_open_file_names(
            self.window,
            "Import datasets",
            "",
            "Data files (*);;All files (*)",
        )
        if not paths:
            return
        choice = self._prompt_import_data_type()
        if choice is None:
            return
        data_type, importer_name = choice
        if importer_name is not None and not all(
            IMPORTERS[importer_name].can_read(path) for path in paths
        ):
            importer_name = None
        importer_options = self._prompt_importer_options(importer_name, paths)
        if importer_options is False:
            return
        stream_group_mode = "reuse"
        if isinstance(importer_options, tuple):
            importer_options, stream_group_mode = importer_options
        self._request_dataset_import(
            group,
            paths,
            data_type=data_type,
            importer_name=importer_name,
            importer_options=(
                importer_options if isinstance(importer_options, dict) else None
            ),
            into=into,
            stream_group_mode=stream_group_mode,
        )

    def _prompt_import_data_type(self) -> tuple[str, str | None] | None:
        types = available_data_types()
        default_index = next((i for i, (name, _label) in enumerate(types) if name == DEFAULT_DATA_TYPE), 0)
        data_type = prompt_import_choice(
            self.window,
            title="Data type",
            prompt="What type of data is this?",
            choices=[(label, name) for name, label in types],
            default_index=default_index,
            object_name="import_data_type",
        )
        if data_type is None:
            return None
        importer_name = default_importer_for_data_type(data_type)
        importer_specs = importers_for_data_type(data_type)
        if len(importer_specs) > 1:
            importer_name = prompt_import_choice(
                self.window,
                title="Importer",
                prompt="Which importer should read this file?",
                choices=[(spec.label, spec.name) for spec in importer_specs],
                object_name="importer_choice",
            )
            if importer_name is None:
                return None
        return data_type, importer_name

    def _prompt_importer_options(
        self,
        importer_name: str | None,
        paths: list[str | Path],
    ) -> tuple[dict[str, dict[str, Any]], str] | dict[str, dict[str, Any]] | bool | None:
        """Collect importer-specific options through a reusable dispatch hook."""

        if importer_name is None:
            return None
        spec = IMPORTERS[importer_name]
        if spec.options_kind == "powder_ins_csv":
            return self._prompt_powder_ins_csv_options(paths)
        if spec.options_kind == "macs_nexus":
            return self._prompt_macs_nexus_options(paths)
        return None

    def _prompt_macs_nexus_options(
        self,
        paths: list[str | Path],
    ) -> tuple[dict[str, dict[str, Any]], str] | bool:
        """Configure a batch of MACS files before expanding SPEC and DIFF."""

        return prompt_macs_nexus_options(self.window, paths)

    def _prompt_powder_ins_csv_options(
        self,
        paths: list[str | Path],
    ) -> dict[str, dict[str, Any]] | bool:
        """Configure a batch of digitized powder INS cuts and maps."""

        return prompt_powder_ins_csv_options(self.window, paths)

    def add_mask_to_selection(self) -> MaskSpec | None:
        item = self._current_item()
        group, entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"group_masks", "dataset_group"} and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is None:
                return None
            mask = create_group_mask(subgroup, _group_reference_dataset(subgroup))
            self._mark_mask_datasets_stale(list(subgroup.iter_datasets()))
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_mask=mask, edit_mask=True)
            return mask
        if role not in {"dataset", "masks"} or group is None or entry is None:
            return None
        mask = create_mask(entry)
        self._mark_mask_datasets_stale([entry])
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_mask=mask, edit_mask=True)
        return mask

    def add_background_to_selection(self) -> BackgroundSpec | None:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(
            self._current_item()
        )
        if group is None:
            return None
        if role in {"dataset", "backgrounds"} and entry is not None:
            owner: DatasetEntry | DataGroup | DatasetGroup = entry
        elif role in {"dataset_group", "group_backgrounds"}:
            owner = self._background_owner_for_item(self._current_item())
            if owner is None:
                return None
        elif role == "datasets":
            owner = group
        else:
            return None
        candidates = [
            candidate
            for candidate in group.iter_datasets()
            if candidate is not owner
            and candidate.data_type in {"powder_inelastic", "single_crystal_inelastic"}
        ]
        group_candidates = [
            node
            for node in group.iter_subgroups()
            if node is not owner and data_group_composite_enabled(_composite_scope(group, node))
        ]
        if not candidates and not group_candidates:
            QtWidgets.QMessageBox.information(
                self.window,
                "Add background",
                "This workspace has no gridded dataset or enabled group composite available as a background.",
            )
            return None
        choices = [
            (candidate.name, "dataset", candidate) for candidate in candidates
        ] + [
            (f"{candidate.name} [live composite]", "group", candidate)
            for candidate in group_candidates
        ]
        labels = [label for label, _kind, _candidate in choices]
        label, accepted = QtWidgets.QInputDialog.getItem(
            self.window,
            "Add background",
            "Background dataset",
            labels,
            0,
            editable=False,
        )
        if not accepted:
            return None
        _label, source_kind, source = choices[labels.index(label)]
        name = _unique_name(source.name, [item.name for item in owner.backgrounds])
        background = BackgroundSpec(
            name=name,
            source_dataset_id=source.id if source_kind == "dataset" else "",
            source_group_id=source.id if source_kind == "group" else None,
            source_entry=source if source_kind == "dataset" else None,
            source_group=source if source_kind == "group" else None,
        )
        owner.backgrounds.append(background)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_background=background)
        return background

    def add_dataset_group_to_selection(self) -> DatasetGroup | None:
        item = self._current_item()
        group, _entry, _mask, _model, role = self._objects_for_item(item)
        if group is None:
            return None
        if role in {"group", "datasets"}:
            parent_node: Any = group
        elif role == "dataset_group":
            parent_node = self._dataset_group_for_item(item)
        else:
            return None
        if parent_node is None:
            return None
        subgroup = create_dataset_group(group, parent_node)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_dataset_group=subgroup, edit_group=True)
        return subgroup

    def add_model_to_selection(self) -> ModelComponentSpec | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"group", "models"} or group is None:
            return None
        model = create_model_component(group)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_model=model, edit_model=True)
        return model

    def materialize_rebin_for_selection(self) -> DatasetEntry | None:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return None
        selected = self._selected_dataset_binning(entry)
        config = selected["config"]
        if not self._confirm_dataset_rebin_memory(
            [selected], operation="Creating a dataset from the rebin"
        ):
            return None
        progress = self._make_rebin_progress_callback("Creating rebinned dataset...") if _dataset_rebin_is_large(entry, config) else None
        try:
            rebinned = create_rebinned_dataset(
                group,
                entry,
                progress_callback=progress,
                config_override=config,
            )
        except RebinCancellationRequested:
            return None
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Create dataset from rebin",
                f"Could not create rebinned dataset:\n{exc}",
            )
            return None
        finally:
            self._close_rebin_progress(progress)
        self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_dataset=rebinned)
        return rebinned

    def save_rebin_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return False
        path, _selected_filter = get_save_file_name(
            self.window,
            "Save rebinned dataset",
            f"{entry.name} rebinned.npz",
            "NumPy archives (*.npz);;All files (*)",
        )
        if not path:
            return False
        selected = self._selected_dataset_binning(entry)
        config = selected["config"]
        if not self._confirm_dataset_rebin_memory(
            [selected], operation="Saving the rebinned dataset"
        ):
            return False
        progress = self._make_rebin_progress_callback("Saving rebinned dataset...") if _dataset_rebin_is_large(entry, config) else None
        try:
            data = rebinned_dataset_data(
                entry,
                extra_masks=effective_dataset_masks(group, entry),
                progress_callback=progress,
                config_override=config,
            )
            rebinned_entry = DatasetEntry(
                name=f"{entry.name} rebinned",
                data=data,
                kind=entry.kind,
                metadata={**copy.deepcopy(entry.metadata), "source_dataset": entry.name, "rebin_saved": True},
                parameters=copy.deepcopy(entry.parameters),
                masks=copy.deepcopy(entry.masks),
            )
            save_dataset_file(rebinned_entry, path, use_view=False)
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Save rebinned dataset",
                f"Could not save rebinned dataset:\n{exc}",
            )
            return False
        finally:
            self._close_rebin_progress(progress)
        return True

    def rebin_now_for_selection(self) -> bool:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or group is None or entry is None:
            return False
        return self.rebin_dataset_now(entry, group)

    def rebin_dataset_now(self, entry: DatasetEntry, group: DataGroup | None) -> bool:
        """Rebin one dataset immediately, independent of the tree selection."""

        from PySide6 import QtWidgets

        if group is None or not _dataset_can_rebin(entry):
            return False
        selected = self._selected_dataset_binning(entry)
        config = selected["config"]
        cache_id = None if selected["fit"] else selected["id"]
        if not self._confirm_dataset_rebin_memory(
            [selected], operation="Rebinning the selected dataset"
        ):
            return False
        if self._interactive:
            def task(progress_callback: Any) -> Any:
                return dataset_for_slice_viewer(
                    entry,
                    extra_masks=effective_dataset_masks(group, entry),
                    force_rebin=True,
                    progress_callback=progress_callback,
                    rebin_config=config,
                    cache_id=cache_id,
                )

            def on_success(view: Any) -> None:
                if view is not None:
                    self.refresh_slice_viewer(group)
                    self._refresh_cache_badges()
                    self._set_dataset_details(entry, group)

            return self._start_background_task(
                title="Rebinning dataset...",
                progress_window_title="Rebin progress",
                failure_title="Rebin now",
                task=task,
                on_success=on_success,
                on_settled=self._refresh_cache_badges,
                success_message="Dataset rebin finished.",
                finishing_progress_event={
                    "stage": "rebin_ui",
                    "message": "refreshing data viewers and bin information",
                },
            )
        progress = self._make_rebin_progress_callback(
            "Rebinning dataset..."
        ) if _dataset_rebin_is_large(entry, config) else None
        try:
            view = dataset_for_slice_viewer(
                entry,
                extra_masks=effective_dataset_masks(group, entry),
                force_rebin=True,
                progress_callback=progress,
                rebin_config=config,
                cache_id=cache_id,
            )
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin now",
                f"Could not rebin dataset:\n{exc}",
            )
            return False
        finally:
            self._close_rebin_progress(progress)
            self._refresh_cache_badges()
        if view is None:
            return False
        self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._set_dataset_details(entry, group)
        return True

    def rebin_all_dataset_binnings_now(
        self, entry: DatasetEntry, group: DataGroup | None
    ) -> bool:
        """Recompute every enabled named binning owned by one dataset."""

        from PySide6 import QtWidgets

        if group is None or not _dataset_can_rebin(entry):
            return False
        binnings = [
            item
            for item in dataset_rebin_binnings(entry)
            if bool(item["config"].get("enabled", False))
        ]
        if not binnings:
            return False
        if not self._confirm_dataset_rebin_memory(
            binnings, operation="Rebinning all dataset binnings"
        ):
            return False

        def task(progress_callback: Any | None) -> int:
            completed = 0
            for item in binnings:
                item_progress = _named_binning_progress_callback(
                    progress_callback,
                    name=str(item["name"]),
                    index=completed,
                    total=len(binnings),
                    kind="binning",
                )
                view = dataset_for_slice_viewer(
                    entry,
                    extra_masks=effective_dataset_masks(group, entry),
                    force_rebin=True,
                    progress_callback=item_progress,
                    rebin_config=item["config"],
                    cache_id=None if item["fit"] else item["id"],
                )
                if view is None:
                    raise ValueError(f"Could not prepare binning {item['name']!r}.")
                completed += 1
                _finish_named_binning_progress(
                    progress_callback,
                    name=str(item["name"]),
                    completed=completed,
                    total=len(binnings),
                    kind="binning",
                )
            return completed

        def on_success(completed: int) -> None:
            if completed:
                self.refresh_slice_viewer(group)
                self._refresh_cache_badges()
                self._set_dataset_details(entry, group)

        if self._interactive:
            return self._start_background_task(
                title="Rebinning all dataset binnings...",
                progress_window_title="Rebin progress",
                failure_title="Rebin all now",
                task=task,
                on_success=on_success,
                on_settled=self._refresh_cache_badges,
                success_message="All dataset binnings finished.",
                finishing_progress_event={
                    "stage": "rebin_ui",
                    "message": "refreshing data viewers and bin information",
                },
            )
        progress = self._make_rebin_progress_callback(
            "Rebinning all dataset binnings...", aggregate=True
        )
        try:
            completed = task(progress)
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window, "Rebin all now", f"Could not rebin all dataset binnings:\n{exc}"
            )
            return False
        finally:
            self._close_rebin_progress(progress)
            self._refresh_cache_badges()
        on_success(completed)
        return bool(completed)

    def _estimated_rebin_cache_bytes(
        self, config: dict[str, Any], *, composite: bool
    ) -> int:
        """Estimate the persistent numerical payload of one requested result."""

        try:
            if composite:
                _lower, _upper, shape = _composite_rebin_bounds(config)
            else:
                shape = estimated_rebin_shape(config)
            return estimate_rebin_result_bytes(math.prod(shape)) if shape else 0
        except (KeyError, TypeError, ValueError):
            return 0

    def _confirm_rebin_cache_memory(
        self,
        result_bytes: list[int],
        *,
        operation: str,
    ) -> bool:
        """Warn before an operation approaches the shared result-cache ceiling."""

        estimates = [int(value) for value in result_bytes if int(value) > 0]
        current = SHARED_REBIN_CACHE_BUDGET.total_bytes()
        added, projected, limit, warn = assess_rebin_cache_memory(
            estimates,
            current_cache_bytes=current,
            cache_limit_bytes=_project_data.scientific_cache_budget_bytes(),
        )
        if not warn:
            return True
        from .project_cache_gui import confirm_rebin_cache_preflight

        return confirm_rebin_cache_preflight(
            self.window,
            operation=operation,
            result_count=len(estimates),
            added_bytes=added,
            current_bytes=current,
            projected_bytes=projected,
            limit_bytes=limit,
            choose_disk_cache=(
                self._compressed_cache_prompt.choose_cache_directory
                if self._compressed_cache_prompt is not None
                else None
            ),
        )

    def _pending_project_rebin_cache_bytes(
        self,
        *,
        group: DataGroup | None = None,
        include_composites: bool = True,
    ) -> list[int]:
        """Return result estimates for stale binnings an operation may compute."""

        estimates = []
        for kind, _name, owner, target, binning_id, config in _project_binning_targets(
            self.project
        ):
            if group is not None and owner is not group:
                continue
            if kind != "dataset" and not include_composites:
                continue
            if _project_binning_is_current(
                kind, owner, target, binning_id, config
            ):
                continue
            estimate = self._estimated_rebin_cache_bytes(
                config, composite=kind != "dataset"
            )
            if estimate:
                estimates.append(estimate)
        return estimates

    def _confirm_dataset_rebin_memory(
        self,
        binnings: list[dict[str, Any]],
        *,
        operation: str = "Rebinning dataset data",
    ) -> bool:
        """Confirm an ordinary rebin when its output workspace is very large."""

        from PySide6 import QtWidgets

        if not self._confirm_rebin_cache_memory(
            [
                self._estimated_rebin_cache_bytes(item["config"], composite=False)
                for item in binnings
            ],
            operation=operation,
        ):
            return False

        risky = []
        for item in binnings:
            config = item["config"]
            output_bins = _dataset_rebin_output_bins(config)
            if not output_bins:
                continue
            estimate, available, warn = assess_output_rebin_memory(
                output_bins, max_batch_bytes=_rebin_max_batch_bytes(config)
            )
            if warn:
                risky.append((item, output_bins, estimate, available))
        if not risky:
            return True
        item, output_bins, estimate, available = max(risky, key=lambda row: row[2])
        answer = QtWidgets.QMessageBox.warning(
            self.window,
            "Rebin memory estimate",
            f"The requested {item['name']!r} rebin has {output_bins:,} output bins. "
            f"Its estimated array workspace is {estimate / 1024**3:.1f} GB, "
            f"over half of the currently available {available / 1024**3:.1f} GB "
            "of RAM. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_composite_rebin_memory(
        self,
        group: DataGroup | _CompositeScope,
        binnings: list[dict[str, Any]],
        *,
        operation: str = "Rebinning composite data",
    ) -> bool:
        """Confirm large native or ordinary composite grids before rebinning."""

        from PySide6 import QtWidgets

        if not self._confirm_rebin_cache_memory(
            [
                self._estimated_rebin_cache_bytes(item["config"], composite=True)
                for item in binnings
            ],
            operation=operation,
        ):
            return False

        candidates = _composite_candidates(group)
        if not candidates:
            return True
        native = _dataset_composite_kind(candidates[0]) == "mdevent"
        risky = []
        for item in binnings:
            config = item["config"]
            _lower, _upper, num_bins = _composite_rebin_bounds(config)
            if native:
                estimate, available, warn = assess_mdevent_memory(
                    num_bins, max_batch_bytes=_rebin_max_batch_bytes(config)
                )
            else:
                estimate, available, warn = assess_output_rebin_memory(
                    math.prod(num_bins), max_batch_bytes=_rebin_max_batch_bytes(config)
                )
            if warn:
                risky.append((item, num_bins, estimate, available))
        if not risky:
            return True
        item, num_bins, estimate, available = max(risky, key=lambda values: values[2])
        grid = " x ".join(str(value) for value in num_bins)
        prefix = (
            f"{len(risky)} requested binnings may use over half the currently available RAM. "
            f"The largest is {item['name']!r}: "
            if len(risky) > 1
            else "The requested "
        )
        answer = QtWidgets.QMessageBox.warning(
            self.window,
            "Rebin memory estimate",
            f"{prefix}{grid} grid ({math.prod(num_bins):,} bins) is estimated to use "
            f"{estimate / 1024**3:.1f} GB. Currently available RAM is "
            f"{available / 1024**3:.1f} GB; this is over half of that allowance.\n\n"
            "Continuing may cause heavy swapping or terminate nfit. "
            "Do you want to continue anyway?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        for risky_item, _bins, _estimate, _available in risky:
            risky_item["config"]["_allow_memory_overcommit_once"] = True
        return True

    def _confirm_viewer_rebin_memory(
        self, group: DataGroup, *, use_composite: bool
    ) -> bool:
        """Warn once for the largest pending output in a viewer load."""

        from PySide6 import QtWidgets

        if not self._confirm_rebin_cache_memory(
            self._pending_project_rebin_cache_bytes(
                group=group, include_composites=use_composite
            ),
            operation="Opening the data viewer",
        ):
            return False

        risky = []
        for dataset in group.iter_datasets():
            if not isinstance(dataset.parameters.get(DATASET_REBIN_KEY), dict):
                continue
            for item in dataset_rebin_binnings(dataset):
                config = item["config"]
                if not config.get("enabled", False) or _project_binning_is_current(
                    "dataset", group, dataset, item["id"], config
                ):
                    continue
                bins = estimated_rebin_shape(config)
                if not bins:
                    continue
                estimate, available, warn = assess_output_rebin_memory(
                    math.prod(bins), max_batch_bytes=_rebin_max_batch_bytes(config)
                )
                if warn:
                    risky.append((item, bins, estimate, available))
        if use_composite:
            for scope in _composite_scopes(group):
                if not data_group_composite_enabled(scope):
                    continue
                candidates = _composite_candidates(scope)
                if not candidates:
                    continue
                native = _dataset_composite_kind(candidates[0]) == "mdevent"
                for item in data_group_composite_binnings(scope):
                    config = item["config"]
                    if not config.get("enabled", False) or _project_binning_is_current(
                        "dataset group", group, scope, item["id"], config
                    ):
                        continue
                    _lower, _upper, bins = _composite_rebin_bounds(config)
                    if native:
                        estimate, available, warn = assess_mdevent_memory(
                            bins, max_batch_bytes=_rebin_max_batch_bytes(config)
                        )
                    else:
                        estimate, available, warn = assess_output_rebin_memory(
                            math.prod(bins),
                            max_batch_bytes=_rebin_max_batch_bytes(config),
                        )
                    if warn:
                        risky.append((item, bins, estimate, available))
        if not risky:
            return True
        item, bins, estimate, available = max(risky, key=lambda row: row[2])
        answer = QtWidgets.QMessageBox.warning(
            self.window,
            "Data viewer memory estimate",
            f"Opening the viewer may rebin {len(risky)} large result(s). The largest, "
            f"{item['name']!r}, has a {' x '.join(map(str, bins))} grid "
            f"({math.prod(bins):,} bins). Its estimated peak is "
            f"{estimate / 1024**3:.1f} GB, over half of the currently available "
            f"{available / 1024**3:.1f} GB of RAM. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        for risky_item, _bins, _estimate, _available in risky:
            risky_item["config"]["_allow_memory_overcommit_once"] = True
        return True

    def rebin_composite_now(self, group: DataGroup | _CompositeScope) -> bool:
        from PySide6 import QtWidgets

        if group is None:
            return False
        selected = self._selected_composite_binning(group)
        config = selected["config"]
        if not bool(config.get("enabled", False)):
            return False
        cache_id = None if selected["fit"] else selected["id"]
        if not self._confirm_composite_rebin_memory(
            group, [selected], operation="Rebinning the selected composite"
        ):
            return False
        if self._interactive:
            def task(progress_callback: Any) -> Any:
                return _cached_composite_dataset_data(
                    group,
                    force_rebin=True,
                    progress_callback=progress_callback,
                    config_override=(None if selected["fit"] else config),
                    binning_id=cache_id,
                )

            def on_success(data: Any) -> None:
                if data is not None:
                    root = _composite_root(group)
                    self.refresh_slice_viewer(root)
                    self._sync_details()

            return self._start_background_task(
                title="Rebinning composite dataset...",
                progress_window_title="Rebin progress",
                failure_title="Rebin composite",
                task=task,
                on_success=on_success,
                on_settled=self._refresh_cache_badges,
                success_message="Composite rebin finished.",
                finishing_progress_event={
                    "stage": "rebin_ui",
                    "message": "refreshing data viewers and bin information",
                },
            )
        progress = self._make_rebin_progress_callback("Rebinning composite dataset...")
        progress({"stage": "prepare", "iteration": 0, "total": 0, "message": "preparing composite rebin"})
        try:
            data = _cached_composite_dataset_data(
                group,
                force_rebin=True,
                progress_callback=progress,
                config_override=(None if selected["fit"] else config),
                binning_id=cache_id,
            )
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin now",
                f"Could not rebin composite dataset:\n{exc}",
            )
            return False
        finally:
            self._close_rebin_progress(progress)
            self._refresh_cache_badges()
        if data is None:
            return False
        root = _composite_root(group)
        self.refresh_slice_viewer(root)
        self._request_overlay_refresh(root)
        self._sync_details()
        return True

    def rebin_all_composite_binnings_now(
        self, group: DataGroup | _CompositeScope
    ) -> bool:
        """Recompute every enabled named composite binning for one collection."""

        from PySide6 import QtWidgets

        if group is None:
            return False
        binnings = [
            item
            for item in data_group_composite_binnings(group)
            if bool(item["config"].get("enabled", False))
        ]
        if not binnings:
            return False
        if not self._confirm_composite_rebin_memory(
            group, binnings, operation="Rebinning all composite binnings"
        ):
            return False

        def task(progress_callback: Any | None) -> int:
            completed = 0
            for item in binnings:
                item_progress = _named_binning_progress_callback(
                    progress_callback,
                    name=str(item["name"]),
                    index=completed,
                    total=len(binnings),
                    kind="binning",
                )
                data = _cached_composite_dataset_data(
                    group,
                    force_rebin=True,
                    progress_callback=item_progress,
                    config_override=None if item["fit"] else item["config"],
                    binning_id=None if item["fit"] else item["id"],
                )
                if data is None:
                    raise ValueError(f"Could not prepare binning {item['name']!r}.")
                completed += 1
                _finish_named_binning_progress(
                    progress_callback,
                    name=str(item["name"]),
                    completed=completed,
                    total=len(binnings),
                    kind="binning",
                )
            return completed

        root = _composite_root(group)

        def on_success(completed: int) -> None:
            if completed:
                self.refresh_slice_viewer(root)
                self._sync_details()

        if self._interactive:
            return self._start_background_task(
                title="Rebinning all composite binnings...",
                progress_window_title="Rebin progress",
                failure_title="Rebin all now",
                task=task,
                on_success=on_success,
                on_settled=self._refresh_cache_badges,
                success_message="All composite binnings finished.",
                finishing_progress_event={
                    "stage": "rebin_ui",
                    "message": "refreshing data viewers and bin information",
                },
            )
        progress = self._make_rebin_progress_callback(
            "Rebinning all composite binnings...", aggregate=True
        )
        try:
            completed = task(progress)
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin all now",
                f"Could not rebin all composite binnings:\n{exc}",
            )
            return False
        finally:
            self._close_rebin_progress(progress)
            self._refresh_cache_badges()
        on_success(completed)
        return bool(completed)

    def materialize_composite_for_group(
        self, group: DataGroup | _CompositeScope
    ) -> bool:
        """Compute and persist a selected group composite inside the project."""

        from PySide6 import QtWidgets

        if self.project_path is None:
            return False
        selected = self._selected_composite_binning(group)
        config = selected["config"]
        if not bool(config.get("enabled", False)):
            return False
        if not self._confirm_composite_rebin_memory(
            group, [selected], operation="Creating a dataset from the composite"
        ):
            return False
        root = _composite_root(group)
        node = group.node if isinstance(group, _CompositeScope) else root
        # Commit current configuration before adding an archive asset. Later
        # project saves preserve that asset atomically.
        save_project(self.project, self.project_path)

        def task(progress_callback: Any) -> DatasetEntry:
            return materialize_composite_dataset(
                self.project_path,
                root,
                node,
                progress_callback=progress_callback,
                config_override=config,
            )

        def on_success(entry: DatasetEntry) -> None:
            self._record_data_group_state_change(root)
            self._mark_dirty()
            self._refresh_tree(select_group=root, select_dataset=entry)

        if self._interactive:
            return self._start_background_task(
                title="Creating dataset from composite...",
                progress_window_title="Rebin progress",
                failure_title="Create dataset from composite",
                task=task,
                on_success=on_success,
                success_message="Composite dataset was stored in the project.",
            )
        progress = self._make_rebin_progress_callback(
            "Creating dataset from composite..."
        )
        try:
            entry = task(progress)
        except RebinCancellationRequested:
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Create dataset from composite",
                f"Could not materialize composite dataset:\n{exc}",
            )
            return False
        finally:
            self._close_rebin_progress(progress)
        on_success(entry)
        return True

    def export_fit_report_for_selection(self) -> bool:
        """Export the selected fit result as a LaTeX or PDF report."""

        from PySide6 import QtCore, QtGui, QtWidgets

        from .report import LatexCompileError, compile_latex_pdf, render_fit_report_latex

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or fit_entry is None or fit_entry.kind != "result":
            return False
        group_name = group.name if group is not None else "project"
        default_name = f"{group_name}_{fit_entry.name}".replace(" ", "_")
        path, _selected_filter = get_save_file_name(
            self.window,
            "Export fit report",
            f"{default_name}.pdf",
            "PDF report (*.pdf);;LaTeX source (*.tex);;All files (*)",
        )
        if not path:
            return False
        try:
            tex_source = render_fit_report_latex(fit_entry, group_name=group_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Export fit report",
                f"Could not render the report:\n{exc}",
            )
            return False
        output = Path(path)
        if output.suffix.lower() != ".pdf":
            if output.suffix.lower() != ".tex":
                output = output.with_suffix(".tex")
            try:
                output.write_text(tex_source, encoding="utf-8")
            except OSError as exc:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Export fit report",
                    f"Could not write the LaTeX file:\n{exc}",
                )
                return False
            return True
        try:
            compile_latex_pdf(tex_source, output)
        except LatexCompileError as exc:
            # The LaTeX source is still valuable: write it next to the
            # requested PDF and tell the user where it went.
            tex_path = output.with_suffix(".tex")
            try:
                tex_path.write_text(tex_source, encoding="utf-8")
                fallback = f"\n\nThe LaTeX source was saved to:\n{tex_path}"
            except OSError:
                fallback = ""
            detail = f"\n\nBuild log tail:\n{exc.log_tail}" if exc.log_tail else ""
            QtWidgets.QMessageBox.warning(
                self.window,
                "Export fit report",
                f"PDF compilation failed:\n{exc}{detail}{fallback}",
            )
            return False
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(output)))
        return True

    def fit_script_for_selection(self) -> str | None:
        """Return a backend-only script for the live workspace fit state."""

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        return fit_workflow_script(self.project, group.name)

    def copy_fit_script_for_selection(self) -> bool:
        return self.copy_workflow_script_for_selection()

    def save_fit_script_for_selection(self) -> bool:
        return self.save_workflow_script_for_selection()

    def save_dataset_for_selection(self) -> bool:
        from PySide6 import QtWidgets

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        if group is not None and dataset_rebin_enabled(entry):
            selected = next(
                item for item in dataset_rebin_binnings(entry) if item["fit"]
            )
            if not _project_binning_is_current(
                "dataset", group, entry, selected["id"], selected["config"]
            ) and not self._confirm_dataset_rebin_memory(
                [selected], operation="Saving the dataset"
            ):
                return False
        path, _selected_filter = get_save_file_name(
            self.window,
            "Save dataset",
            f"{entry.name}.npz",
            "NumPy archives (*.npz);;All files (*)",
        )
        if not path:
            return False
        try:
            save_dataset_file(entry, path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Save dataset",
                f"Could not save dataset:\n{exc}",
            )
            return False
        return True

    def dataset_workflow_script_for_selection(self) -> str | None:
        """Return a script that reconstructs the selected dataset from source."""

        _group, entry, _mask, _model, role = self._objects_for_item(
            self._current_item()
        )
        if role != "dataset" or entry is None:
            return None
        return dataset_workflow_script(self.project, entry.id)

    def workflow_script_for_selection(self) -> tuple[str, str] | None:
        """Return ``(label, script)`` for a supported selected workflow target."""

        item = self._current_item()
        _group, entry, _mask, _model, role = self._objects_for_item(item)
        if role == "dataset" and entry is not None:
            return entry.name, dataset_workflow_script(self.project, entry.id)
        if role == "analysis":
            analysis = self._analysis_item_roles.get(id(item))
            if analysis is not None:
                return (
                    analysis.name,
                    analysis_workflow_script(self.project, analysis.id),
                )
        if role == "fit" and _group is not None:
            return _group.name, fit_workflow_script(self.project, _group.name)
        return None

    def copy_workflow_script_for_selection(self) -> bool:
        """Copy the selected target's reproducible workflow to the clipboard."""

        from PySide6 import QtWidgets

        try:
            payload = self.workflow_script_for_selection()
        except (KeyError, NotImplementedError, OSError, WorkflowValidationError) as exc:
            QtWidgets.QMessageBox.information(
                self.window,
                "Copy workflow script",
                f"This workflow cannot yet be exported:\n{exc}",
            )
            return False
        if payload is None:
            return False
        _label, script = payload
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def save_workflow_script_for_selection(self) -> bool:
        """Save the selected target's reproducible workflow as Python."""

        from PySide6 import QtWidgets

        try:
            payload = self.workflow_script_for_selection()
        except (KeyError, NotImplementedError, OSError, WorkflowValidationError) as exc:
            QtWidgets.QMessageBox.information(
                self.window,
                "Save workflow script",
                f"This workflow cannot yet be exported:\n{exc}",
            )
            return False
        if payload is None:
            return False
        label, script = payload
        stem = re.sub(r"\W+", "_", label).strip("_") or "workflow"
        path, _selected_filter = get_save_file_name(
            self.window,
            "Save workflow script",
            f"{stem}_workflow.py",
            "Python scripts (*.py);;All files (*)",
        )
        if not path:
            return False
        try:
            Path(path).write_text(script, encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Save workflow script",
                f"Could not write the script:\n{exc}",
            )
            return False
        return True

    def copy_dataset_workflow_script_for_selection(self) -> bool:
        """Copy the selected dataset's reproducible workflow to the clipboard."""

        return self.copy_workflow_script_for_selection()

    def save_dataset_workflow_script_for_selection(self) -> bool:
        """Save the selected dataset's reproducible workflow as Python."""

        return self.save_workflow_script_for_selection()

    def fit_now_for_selection(self) -> FitTimelineEntry | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        if not self._confirm_rebin_cache_memory(
            self._pending_project_rebin_cache_bytes(group=group),
            operation="Running the fit",
        ):
            return None
        self._set_selected_fit_optimizer_config()
        should_branch = bool(self.fit_branch_check.isChecked()) or _should_branch_fit_now(group, fit_entry)
        progress = self._fit_progress_dialog
        if progress is None:
            progress = _FitProgressDialog(self)
            self._fit_progress_dialog = progress
        progress.reset("Starting fit pipeline...")
        progress.show()
        result = run_group_fit(
            group,
            fit_entry,
            branch_timeline=should_branch,
            progress_callback=progress.update_progress,
        )
        failed = str(result.goodness.get("status", "")) == "failed"
        if failed:
            progress.fail(str(result.goodness.get("message", "The fit did not run.")))
        else:
            progress.finish(
                "Fit pipeline finished.",
                parameters=result.goodness.get("parameters"),
                limit_hits=result.goodness.get("parameters_at_limits"),
                summary_lines=_autocorrelation_progress_summary(
                    _sampling_result_from_dict(result.metadata.get("posterior_samples"))
                ),
            )
        self.fit_branch_check.setChecked(False)
        self.refresh_slice_viewer(group)
        item_to_select = _fit_entry_to_select_after_run(group, fit_entry, result)
        self._set_active_fit_state(group, item_to_select)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_fit=item_to_select)
        return result

    def _start_background_task(
        self,
        *,
        title: str,
        failure_title: str,
        task: Any,
        on_success: Any,
        success_message: str,
        on_settled: Any | None = None,
        close_on_success: bool = True,
        completion_summary: Any | None = None,
        progress_window_title: str | None = None,
        finishing_progress_event: dict[str, Any] | None = None,
    ) -> bool:
        from PySide6 import QtCore, QtWidgets

        thread = self._fit_worker_thread
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self.window,
                failure_title,
                "Another background operation is already running.",
            )
            return False

        class Worker(QtCore.QObject):
            progress = QtCore.Signal(dict)
            finished = QtCore.Signal(object)
            cancelled = QtCore.Signal(object)
            aborted = QtCore.Signal()
            failed = QtCore.Signal(str)

            def __init__(self) -> None:
                super().__init__()
                self.cancel_requested = False

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    def progress_callback(event: dict[str, Any]) -> None:
                        if self.cancel_requested:
                            raise FitCancellationRequested(
                                "Operation cancelled by user."
                            )
                        self.progress.emit(event)

                    self.finished.emit(task(progress_callback))
                except SamplingCancelled as exc:
                    self.cancelled.emit(exc.result)
                except FitCancellationRequested:
                    self.aborted.emit()
                except Exception as exc:
                    self.failed.emit(str(exc))

            def cancel(self) -> None:
                self.cancel_requested = True

        class Handler(QtCore.QObject):
            def __init__(self, parent_window: Any) -> None:
                super().__init__(parent_window)
                self._parent_window = parent_window

            @QtCore.Slot(dict)
            def handle_progress(self, event: dict[str, Any]) -> None:
                progress.update_progress(event)

            def settle(self) -> None:
                if on_settled is not None:
                    on_settled()

            @QtCore.Slot(object)
            def handle_success(self, result: Any) -> None:
                try:
                    if finishing_progress_event is not None:
                        progress.update_progress(finishing_progress_event)
                    should_finish = on_success(result)
                    if should_finish is not False:
                        lines = completion_summary(result) if completion_summary is not None else None
                        progress.finish(success_message, summary_lines=lines)
                        if close_on_success:
                            progress.close()
                finally:
                    try:
                        self.settle()
                    finally:
                        worker_thread.quit()

            @QtCore.Slot(object)
            def handle_cancelled(self, result: Any) -> None:
                try:
                    on_success(result)
                    lines = completion_summary(result) if completion_summary is not None else None
                    progress.finish(
                        "emcee posterior sampling cancelled; partial samples saved.",
                        summary_lines=lines,
                    )
                finally:
                    try:
                        self.settle()
                    finally:
                        worker_thread.quit()

            @QtCore.Slot(str)
            def handle_failure(self, message: str) -> None:
                progress.fail(message)
                try:
                    QtWidgets.QMessageBox.warning(self._parent_window, failure_title, message)
                finally:
                    try:
                        self.settle()
                    finally:
                        worker_thread.quit()

            @QtCore.Slot()
            def handle_aborted(self) -> None:
                try:
                    progress.finish("Operation cancelled.")
                    progress.close()
                finally:
                    try:
                        self.settle()
                    finally:
                        worker_thread.quit()

        is_rebin_progress = progress_window_title == "Rebin progress"
        if is_rebin_progress:
            progress = _RebinProgressDialog(self, title)
        else:
            progress = self._fit_progress_dialog
            if progress is None:
                progress = _FitProgressDialog(self)
                self._fit_progress_dialog = progress
            progress.dialog.setWindowTitle(progress_window_title or "Fit progress")
        progress.reset(title)
        progress.show()

        worker_thread = QtCore.QThread(self.window)
        worker = Worker()
        handler = Handler(self.window)
        worker.moveToThread(worker_thread)
        self._fit_worker_thread = worker_thread
        self._fit_worker = worker
        self._fit_worker_handler = handler
        self._fit_disabled_widget_states = []
        for widget in (
            self.tree,
            self.fit_editor_widget,
            self.details_scroll,
            self.fit_now_button,
            self.fit_corner_button,
            self.show_data_fit_button,
            self.fit_export_report_button,
            self.delete_button,
        ):
            if widget is not None:
                self._fit_disabled_widget_states.append((widget, widget.isEnabled()))
                widget.setEnabled(False)
        progress.set_cancel_callback(worker.cancel)
        worker_thread.started.connect(worker.run)
        worker.progress.connect(handler.handle_progress)

        def cleanup() -> None:
            progress.set_cancel_callback(None)
            for widget, enabled in self._fit_disabled_widget_states:
                widget.setEnabled(enabled)
            self._fit_disabled_widget_states = []
            self._fit_worker_thread = None
            self._fit_worker = None
            self._fit_worker_handler = None

        worker.finished.connect(handler.handle_success)
        worker.cancelled.connect(handler.handle_cancelled)
        worker.aborted.connect(handler.handle_aborted)
        worker.failed.connect(handler.handle_failure)
        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.aborted.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker_thread.finished.connect(cleanup)
        worker_thread.finished.connect(worker_thread.deleteLater)
        worker_thread.start()
        return True

    def start_fit_for_selection(self) -> bool:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return False
        if not self._confirm_rebin_cache_memory(
            self._pending_project_rebin_cache_bytes(group=group),
            operation="Running the fit",
        ):
            return False
        self._set_selected_fit_optimizer_config()
        should_branch = bool(self.fit_branch_check.isChecked()) or _should_branch_fit_now(group, fit_entry)

        def task(progress_callback: Any) -> FitTimelineEntry:
            return run_group_fit(
                group,
                fit_entry,
                branch_timeline=should_branch,
                progress_callback=progress_callback,
            )

        def on_success(result: FitTimelineEntry) -> None:
            failed = str(result.goodness.get("status", "")) == "failed"
            cancelled = str(result.goodness.get("status", "")) == "cancelled"
            if failed:
                progress = self._fit_progress_dialog
                if progress is not None:
                    progress.fail(str(result.goodness.get("message", "The fit did not run.")))
            elif cancelled:
                progress = self._fit_progress_dialog
                if progress is not None:
                    progress.finish(
                        str(result.goodness.get("message", "Partial posterior samples were saved.")),
                        parameters=result.goodness.get("parameters"),
                        limit_hits=result.goodness.get("parameters_at_limits"),
                        summary_lines=_autocorrelation_progress_summary(
                            _sampling_result_from_dict(result.metadata.get("posterior_samples"))
                        ),
                    )
            else:
                progress = self._fit_progress_dialog
                if progress is not None:
                    progress.finish(
                        "Fit pipeline finished.",
                        parameters=result.goodness.get("parameters"),
                        limit_hits=result.goodness.get("parameters_at_limits"),
                        summary_lines=_autocorrelation_progress_summary(
                            _sampling_result_from_dict(result.metadata.get("posterior_samples"))
                        ),
                    )
            self.fit_branch_check.setChecked(False)
            self.refresh_slice_viewer(group)
            item_to_select = _fit_entry_to_select_after_run(group, fit_entry, result)
            self._set_active_fit_state(group, item_to_select)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_fit=item_to_select)
            return False

        return self._start_background_task(
            title="Starting fit pipeline...",
            failure_title="Fit now",
            task=task,
            on_success=on_success,
            success_message="Fit pipeline finished.",
            close_on_success=False,
        )

    def open_fit_diagnostics_plots_for_selection(self) -> Any | None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return None
        if not _fit_entry_has_diagnostic_plots(fit_entry):
            return None
        window = _FitDiagnosticsPlotWindow(fit_entry, self)
        window.show()
        self._auxiliary_windows[id(window)] = window
        return window

    def apply_posterior_sampling_window(
        self,
        fit_entry: FitTimelineEntry,
        burn_in: int,
        thin: int,
    ) -> bool:
        from PySide6 import QtWidgets

        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        if stored is None or stored.chain is None:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain a raw emcee chain to re-window.",
            )
            return False
        try:
            updated = _sampling_result_with_window(stored, burn_in=burn_in, thin=thin)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not update the posterior sampling window:\n{exc}",
            )
            return False
        _store_sampling_result_on_fit_entry(fit_entry, updated)
        self._mark_dirty()
        self._set_fit_details(fit_entry)
        return True

    def promote_best_posterior_sample_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
    ) -> bool:
        from PySide6 import QtWidgets

        try:
            candidate = _best_posterior_promotion_candidate(fit_entry)
            if candidate is None and "chi2" not in fit_entry.goodness:
                compiled = _compiled_problem_for_fit_entry(group, fit_entry)
                candidate = _best_posterior_promotion_candidate(fit_entry, compiled)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not inspect the posterior samples:\n{exc}",
            )
            return False
        if candidate is None:
            QtWidgets.QMessageBox.information(
                self.window,
                "Posterior sampler",
                "The stored emcee chain does not contain a finite sample with a better likelihood than this fit result.",
            )
            return False
        sample_params, sample_log_probability, baseline_log_probability, location = candidate
        current_snapshot = snapshot_data_group_state(group)
        self._set_active_fit_state(group, fit_entry)
        try:
            if fit_entry.snapshot:
                restore_data_group_state(group, fit_entry.snapshot)
            components = [
                model for model in group.models.values() if isinstance(model, ModelComponentSpec)
            ]
            inputs, _bundles = fit_dataset_inputs(group, purpose="fit")
            compiled_for_state = compile_fit_problem(components, inputs, description=group.name)
            promoted_params = {
                spec.name: float(spec.value)
                for spec in compiled_for_state.problem.parameter_specs
            }
            promoted_params.update(sample_params)
            promoted_params = compiled_for_state.problem.resolve_parameters(promoted_params)
            _write_back_parameter_values(
                group,
                components,
                compiled_for_state,
                promoted_params,
            )
        except Exception as exc:
            restore_data_group_state(group, current_snapshot)
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not promote the posterior sample:\n{exc}",
            )
            return False
        branch_created = self._record_data_group_state_change(group)
        promoted = self._active_fit_entry(group)
        if promoted is not None and promoted.kind == "current":
            promoted.metadata.setdefault("promoted_posterior_sample", {})
            promoted.metadata["promoted_posterior_sample"] = {
                "source_fit": fit_entry.name,
                "log_probability": float(sample_log_probability),
                "previous_log_probability": float(baseline_log_probability),
                **location,
            }
        self._mark_dirty()
        self.refresh_slice_viewer(group)
        if branch_created:
            self._refresh_tree(select_group=group, select_fit=promoted)
        else:
            self._refresh_tree(select_group=group, select_fit=promoted or fit_entry)
        return True

    def run_posterior_sampler_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int = -1,
        append: bool = False,
    ) -> bool:
        from PySide6 import QtWidgets

        if not isinstance(fit_entry.goodness.get("parameters"), dict):
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain best-fit parameters to start emcee.",
            )
            return False
        if not self._confirm_rebin_cache_memory(
            self._pending_project_rebin_cache_bytes(group=group),
            operation="Running the posterior sampler",
        ):
            return False
        progress = self._fit_progress_dialog
        if progress is None:
            progress = _FitProgressDialog(self)
            self._fit_progress_dialog = progress
        progress.reset("Starting emcee posterior sampler...")
        progress.show()
        try:
            result = self._posterior_sampler_result_for_fit(
                group,
                fit_entry,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed,
                workers=workers,
                append=append,
                progress_callback=progress.update_progress,
            )
        except SamplingCancelled as exc:
            result = exc.result
            _store_sampling_result_on_fit_entry(fit_entry, result)
            self._mark_dirty()
            progress.finish(
                "emcee posterior sampling cancelled; partial samples saved.",
                summary_lines=_autocorrelation_progress_summary(result),
            )
            self._set_fit_details(fit_entry)
            return True
        except Exception as exc:
            progress.fail(str(exc))
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                f"Could not run emcee posterior sampling:\n{exc}",
            )
            return False
        _store_sampling_result_on_fit_entry(fit_entry, result)
        self._mark_dirty()
        progress.finish(
            "emcee posterior sampling finished.",
            summary_lines=_autocorrelation_progress_summary(result),
        )
        self._set_fit_details(fit_entry)
        return True

    def start_posterior_sampler_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int = 1,
        append: bool = False,
    ) -> bool:
        from PySide6 import QtWidgets

        if not isinstance(fit_entry.goodness.get("parameters"), dict):
            QtWidgets.QMessageBox.warning(
                self.window,
                "Posterior sampler",
                "This fit result does not contain best-fit parameters to start emcee.",
            )
            return False
        if not self._confirm_rebin_cache_memory(
            self._pending_project_rebin_cache_bytes(group=group),
            operation="Running the posterior sampler",
        ):
            return False

        def task(progress_callback: Any) -> SamplingResult:
            return self._posterior_sampler_result_for_fit(
                group,
                fit_entry,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed,
                workers=workers,
                append=append,
                progress_callback=progress_callback,
            )

        def on_success(result: SamplingResult) -> bool:
            _store_sampling_result_on_fit_entry(fit_entry, result)
            self._mark_dirty()
            self._set_fit_details(fit_entry)
            return True

        return self._start_background_task(
            title="Starting emcee posterior sampler...",
            failure_title="Posterior sampler",
            task=task,
            on_success=on_success,
            success_message="emcee posterior sampling finished.",
            close_on_success=False,
            completion_summary=_autocorrelation_progress_summary,
        )

    def confirm_and_start_posterior_rerun(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        **sampler_kwargs: Any,
    ) -> bool:
        from PySide6 import QtWidgets

        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        if stored is not None:
            answer = QtWidgets.QMessageBox.question(
                self.window,
                "Replace emcee samples?",
                "This fit result already has emcee samples. Rerunning will replace "
                "the stored samples and raw chain. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        return self.start_posterior_sampler_for_fit(
            group,
            fit_entry,
            append=False,
            **sampler_kwargs,
        )

    def _posterior_sampler_result_for_fit(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        n_walkers: int,
        n_steps: int,
        burn_in: int,
        thin: int,
        random_seed: int | None,
        workers: int,
        append: bool,
        progress_callback: Any | None,
    ) -> SamplingResult:
        compiled = _compiled_problem_for_fit_entry(group, fit_entry)
        initial_params = {
            str(name): float(value)
            for name, value in dict(fit_entry.goodness.get("parameters", {})).items()
        }
        initial_walkers = None
        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        walkers = None if n_walkers <= 0 else int(n_walkers)
        if append:
            if stored is None or stored.chain is None:
                raise ValueError("append requires an existing raw emcee chain")
            compiled_names = [
                spec.name for spec in compiled.problem.parameter_specs if spec.vary
            ]
            if list(stored.variable_names) != compiled_names:
                raise ValueError(
                    "stored posterior parameter order no longer matches the fit problem"
                )
            chain = np.asarray(stored.chain, dtype=float)
            if chain.ndim != 3 or chain.shape[0] == 0:
                raise ValueError("stored raw emcee chain is empty or malformed")
            initial_walkers = np.asarray(chain[-1], dtype=float)
            walkers = int(initial_walkers.shape[0])
        config = SamplerConfig(
            n_walkers=walkers,
            n_steps=int(n_steps),
            burn_in=int(burn_in),
            thin=max(1, int(thin)),
            random_seed=random_seed,
            kwargs={"workers": int(workers)},
        )
        try:
            sampled = sample_problem_parameters(
                compiled.problem,
                config,
                initial_params=initial_params if initial_walkers is None else None,
                initial_walkers=initial_walkers,
                progress_callback=progress_callback,
            )
        except SamplingCancelled as exc:
            partial = (
                _combined_sampling_result(stored, exc.result, burn_in=burn_in, thin=thin)
                if append and stored is not None
                else exc.result
            )
            raise SamplingCancelled(str(exc), partial) from exc
        return (
            _combined_sampling_result(stored, sampled, burn_in=burn_in, thin=thin)
            if append and stored is not None
            else sampled
        )

    def show_data_and_fit_for_selection(self) -> Any | None:
        """Open the data viewer with the model overlay enabled."""

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None:
            return None
        if not (_group_has_fit_channels(group) or _group_has_enabled_model_components(group)):
            return None
        viewer = self.open_slice_viewer(group)
        if viewer is None:
            return None
        check = getattr(viewer, "show_fit_check", None)
        if check is not None and check.isEnabled():
            check.setChecked(True)
        residual_check = getattr(viewer, "show_residual_check", None)
        if residual_check is not None:
            residual_check.setChecked(False)
        return viewer

    def open_slice_viewer_for_selection(self) -> Any | None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role == "analysis_output" and group is not None:
            output = self._analysis_output_roles.get(id(self._current_item()))
            if output is None or output.dataset_id is None:
                return None
            dataset = next(
                (item for item in group.iter_datasets() if item.id == output.dataset_id),
                None,
            )
            if dataset is None:
                return None
            return self.open_slice_viewer(
                group, selected_dataset_name=dataset.name, use_composite=False
            )
        allowed = {
            "group", "datasets", "dataset", "masks", "mask", "backgrounds",
            "background", "group_backgrounds", "group_background",
            "dataset_group", "group_masks", "group_mask",
        }
        if role not in allowed or group is None:
            return None
        # For a mask or the Masks node, entry is the owning dataset.
        selected_name = entry.name if role in {"dataset", "masks", "mask", "backgrounds", "background"} and entry is not None else None
        use_composite = selected_name is None
        node = self._dataset_group_for_item(self._current_item())
        if use_composite and isinstance(node, DatasetGroup):
            ancestor_item = self._current_item().parent()
            while ancestor_item is not None:
                ancestor = self._dataset_group_for_item(ancestor_item)
                if isinstance(ancestor, (DataGroup, DatasetGroup)) and data_group_composite_enabled(_composite_scope(group, ancestor)):
                    node = ancestor
                ancestor_item = ancestor_item.parent()
            # Match the selected collection to the same effective entry traversal
            # used by the viewer, including a collection's nested composites.
            def first_name(node):
                scope = _composite_scope(group, node)
                if data_group_composite_enabled(scope):
                    return _composite_dataset_name(scope)
                if node.datasets:
                    return node.datasets[0].name
                return next((name for child in node.subgroups if (name := first_name(child))), None)

            selected_name = first_name(node)
        return self.open_slice_viewer(group, selected_dataset_name=selected_name, use_composite=use_composite)

    def open_slice_viewer(
        self,
        group: DataGroup,
        *,
        selected_dataset_name: str | None = None,
        use_composite: bool = True,
    ) -> Any | None:
        from PySide6 import QtWidgets

        if not self._confirm_viewer_rebin_memory(
            group, use_composite=use_composite
        ):
            return None
        progress = self._rebin_progress_callback_for_group(group, use_composite=use_composite)
        try:
            datasets, names = slice_viewer_datasets(
                group,
                use_composite=use_composite,
                force_rebin=True,
                progress_callback=progress,
                defer_progress_completion=progress is not None,
            )
        except RebinCancellationRequested:
            self._close_rebin_progress(progress)
            return None
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Data viewer",
                f"Could not load datasets for the data viewer:\n{exc}",
            )
            self._close_rebin_progress(progress)
            return None
        if not datasets:
            self._close_rebin_progress(progress)
            return None
        try:
            if progress is not None:
                progress(
                    {
                        "stage": "viewer_prepare",
                        "iteration": 0,
                        "total": 3,
                        "message": "updating project controls before opening the data viewer",
                    }
                )
            # Preparing a composite viewer can populate caches for several
            # sibling dataset groups. Refresh the complete tree so every
            # newly current binning receives its cache badge, rather than
            # updating only the selected row in _sync_details().
            self._refresh_cache_badges()
            self._sync_details()
            if progress is not None:
                progress(
                    {
                        "stage": "viewer_prepare",
                        "iteration": 1,
                        "total": 3,
                        "message": "constructing data viewer controls and initial slice",
                    }
                )
            viewer = self._create_slice_viewer(group, datasets, names)
            viewer._nfit_use_composite = bool(use_composite)
            viewer._nfit_group = group
            viewer._nfit_dataset_ids = {
                dataset.name: dataset.id for dataset in group.iter_datasets()
            }
            if hasattr(viewer, "set_save_plot_callback"):
                viewer.set_save_plot_callback(
                    lambda viewer=viewer, group=group: self.save_plot_from_viewer(group, viewer),
                    new_plot_callback=lambda viewer=viewer, group=group: self.save_plot_from_viewer(group, viewer, as_new=True),
                )
            if hasattr(viewer, "set_save_project_callback"):
                viewer.set_save_project_callback(self.save)
            if hasattr(viewer, "set_open_new_viewer_callback"):
                viewer.set_open_new_viewer_callback(
                    lambda selected_name, group=group, use_composite=use_composite: self.open_slice_viewer(
                        group,
                        selected_dataset_name=selected_name,
                        use_composite=use_composite,
                    )
                )
            if hasattr(viewer, "set_unmask_model_callback"):
                viewer.set_unmask_model_callback(
                    lambda _enabled, group=group: self._request_overlay_refresh(group)
                )
            if selected_dataset_name in getattr(viewer, "source_dataset_names", names):
                if hasattr(viewer, "_set_dataset_selection"):
                    viewer._set_dataset_selection(
                        viewer._source_dataset_options().index(selected_dataset_name)
                    )
                else:
                    viewer.dataset_combo.setCurrentIndex(names.index(selected_dataset_name))
            if progress is not None:
                progress(
                    {
                        "stage": "viewer_prepare",
                        "iteration": 2,
                        "total": 3,
                        "message": "rendering the initial data-viewer plot",
                    }
                )
            viewer.show()
            if progress is not None:
                progress(
                    {
                        "stage": "viewer_prepare",
                        "iteration": 3,
                        "total": 3,
                        "message": "data viewer ready",
                    }
                )
                _finish_viewer_batch_progress(
                    group,
                    use_composite=use_composite,
                    progress_callback=progress,
                )
            return viewer
        finally:
            self._close_rebin_progress(progress)

    def save_plot_from_viewer(self, group: DataGroup, viewer: Any, *, as_new: bool = False) -> PlotEntry | None:
        """Create or update a workspace plot using the interactive viewer state."""

        name = viewer.dataset_combo.currentText() if viewer.dataset_combo is not None else "Plot"
        source_names = (
            viewer.waterfall_source_dataset_names()
            if getattr(viewer, "_waterfall_mode_active", lambda: False)()
            else [name]
        )
        id_by_name = getattr(viewer, "_nfit_dataset_ids", {})
        dataset_ids = [
            id_by_name[source_name]
            for source_name in source_names
            if source_name in id_by_name
        ]
        composite_scope: DataGroup | _CompositeScope | None = None
        if not dataset_ids and len(source_names) == 1:
            composite_scope = next(
                (
                    scope
                    for scope in _composite_scopes(group)
                    if _composite_dataset_name(scope) == source_names[0]
                ),
                None,
            )
            if composite_scope is not None:
                dataset_ids = [
                    dataset.id for dataset in _composite_candidates(composite_scope)
                ]
        if not dataset_ids:
            return None
        dataset_id = dataset_ids[0]
        settings = viewer.current_plot_settings()
        displayed_data = viewer.datasets[viewer.dataset_index]
        displayed_binning_id = str(
            displayed_data.metadata.get("binning_id", FIT_BINNING_ID)
        )
        displayed_binning_name = str(
            displayed_data.metadata.get("binning_name", "Default")
        )
        binning_ids_by_source = {
            str(data.metadata.get("source_dataset_name", loaded_name)): str(
                data.metadata.get("binning_id", FIT_BINNING_ID)
            )
            for data, loaded_name in zip(
                viewer.datasets, viewer.dataset_names, strict=True
            )
            if str(data.metadata.get("binning_name", "Default"))
            == displayed_binning_name
        }
        if composite_scope is not None:
            saved_composite = getattr(viewer, "_nfit_plot_composite_recipe", None)
            if isinstance(saved_composite, dict):
                settings[PLOT_SOURCE_COMPOSITE_KEY] = copy.deepcopy(saved_composite)
            else:
                node = (
                    composite_scope.node
                    if isinstance(composite_scope, _CompositeScope)
                    else None
                )
                settings[PLOT_SOURCE_COMPOSITE_KEY] = {
                    "dataset_group_id": node.id if node is not None else None,
                    "name": source_names[0],
                    "metadata_dimensions": copy.deepcopy(composite_scope.metadata.get("metadata_dimensions", [])),
                    "config": copy.deepcopy(
                        data_group_composite_config_by_id(
                            composite_scope, displayed_binning_id
                        )
                    ),
                }
        entries_by_id = {item.id: item for item in group.iter_datasets()}
        viewer_rebin_configs = getattr(viewer, "_nfit_plot_rebin_configs", None)
        if isinstance(viewer_rebin_configs, dict):
            settings[PLOT_SOURCE_REBIN_CONFIGS_KEY] = copy.deepcopy(
                viewer_rebin_configs
            )
        else:
            settings[PLOT_SOURCE_REBIN_CONFIGS_KEY] = {
                source_id: copy.deepcopy(
                    dataset_rebin_config_by_id(
                        entries_by_id[source_id],
                        binning_ids_by_source.get(
                            entries_by_id[source_id].name,
                            dataset_rebin_binnings(entries_by_id[source_id])[0]["id"],
                        ),
                    )
                )
                for source_id in dataset_ids
                if source_id in entries_by_id
            }
        plot_type = (
            "mdhisto_waterfall" if settings.get("view_mode") == "waterfall" else
            "mdhisto_tiled_slices" if settings.get("view_mode") == "tiled_slices" else
            "fit_comparison" if bool(settings.get("show_fit")) else
            "mdhisto_line" if viewer._is_effective_1d() else "mdhisto_slice"
        )
        editing_id = None if as_new else getattr(viewer, "_nfit_editing_plot_id", None)
        existing = next((plot for plot in group.plots if plot.id == editing_id), None)
        if existing is None:
            plot = new_plot_entry(
                _unique_name(f"{name} plot", [item.name for item in group.plots]),
                dataset_id,
                settings,
                plot_type=plot_type,
                dataset_ids=dataset_ids,
            )
            group.plots.append(plot)
            viewer._nfit_editing_plot_id = plot.id
        else:
            existing.type = plot_type
            existing.settings = settings
            existing.sources = [
                PlotSourceRef(dataset_id=source_id)
                for source_id in dataset_ids
            ]
            plot = existing
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_plot=plot)
        return plot

    def open_saved_plot_for_selection(self) -> Any | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        plot = self._plot_for_item(self._current_item())
        if role != "plot" or group is None or plot is None or not plot.sources:
            return None
        if plot.type == "fit_covariance":
            fit_entry = next((fit for fit in _walk_fit_entries(group.fits) if fit.id == plot.sources[0].fit_id), None)
            if fit_entry is None:
                return None
            from .plot_gui import PlotWindow

            window = PlotWindow(plot, None, fit_entry=fit_entry, project_path=self.project_path, on_update=lambda _plot: self._mark_dirty())
            self._plot_windows[plot.id] = window
            return window.show()
        source_ids = [
            source.dataset_id for source in plot.sources if source.dataset_id
        ]
        if not source_ids:
            return None
        try:
            prepared, _names = _saved_plot_source_views(group, plot)
        except (TypeError, ValueError):
            return None
        from .plot_gui import PlotWindow

        plot_data = prepared if plot.type == "mdhisto_waterfall" else prepared[0]
        window = PlotWindow(plot, plot_data, project_path=self.project_path, on_update=lambda _plot: self._mark_dirty())
        self._plot_windows[plot.id] = window
        return window.show()

    def edit_saved_plot_in_viewer(self) -> Any | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        plot = self._plot_for_item(self._current_item())
        if role != "plot" or group is None or plot is None or not plot.sources:
            return None
        dataset = next((item for item in group.iter_datasets() if item.id == plot.sources[0].dataset_id), None)
        if dataset is None:
            return None
        composite_recipe = plot.settings.get(PLOT_SOURCE_COMPOSITE_KEY)
        viewer = self.open_slice_viewer(
            group,
            selected_dataset_name=(
                str(composite_recipe.get("name"))
                if isinstance(composite_recipe, dict)
                else dataset.name
            ),
            use_composite=isinstance(composite_recipe, dict),
        )
        if viewer is not None:
            try:
                prepared, names = _saved_plot_source_views(group, plot)
            except (TypeError, ValueError):
                return None
            replacement = {
                "dataset_names": names,
                "dataset_group_keys": _waterfall_group_keys(group, names),
                "selected_dataset_name": names[0],
            }
            if hasattr(viewer, "crystal_contexts"):
                replacement["crystal_contexts"] = [
                    {
                        "spacegroup": group.spacegroup,
                        "lattice_parameters": copy.deepcopy(group.lattice_parameters),
                    }
                    for _dataset in prepared
                ]
            viewer.replace_datasets(prepared, **replacement)
            viewer._nfit_editing_plot_id = plot.id
            viewer._nfit_plot_rebin_configs = copy.deepcopy(
                plot.settings.get(PLOT_SOURCE_REBIN_CONFIGS_KEY, {})
            )
            if isinstance(composite_recipe, dict):
                viewer._nfit_plot_composite_recipe = copy.deepcopy(composite_recipe)
            viewer.apply_plot_settings(plot.settings)
            viewer.set_saved_plot_editing()
        return viewer

    def plot_script_for_selection(self) -> str | None:
        """Return a backend-only script for the selected saved plot."""

        _group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        plot = self._plot_for_item(self._current_item())
        if role != "plot" or plot is None or self.project_path is None:
            return None
        return plot_script(plot, project_path=self.project_path)

    def copy_plot_script_for_selection(self) -> bool:
        """Copy the selected plot's backend-only generating script."""

        from PySide6 import QtWidgets

        script = self.plot_script_for_selection()
        if script is None:
            QtWidgets.QMessageBox.information(
                self.window,
                "Copy plot script",
                "Save the project first so the generated script can load its data and plot recipe.",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def save_plot_script_for_selection(self) -> bool:
        """Save the selected plot's backend-only generating script."""

        from PySide6 import QtWidgets

        script = self.plot_script_for_selection()
        if script is None:
            QtWidgets.QMessageBox.information(
                self.window,
                "Save plot script",
                "Save the project first so the generated script can load its data and plot recipe.",
            )
            return False
        plot = self._plot_for_item(self._current_item())
        stem = plot.name.replace(" ", "_") if plot is not None else "plot"
        path, _selected_filter = get_save_file_name(
            self.window,
            "Save plot script",
            f"{stem}.py",
            "Python scripts (*.py);;All files (*)",
        )
        if not path:
            return False
        Path(path).write_text(script, encoding="utf-8")
        return True

    def create_fit_covariance_plot_for_selection(self) -> PlotEntry | None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        fit_entry = self._fit_entry_for_item(self._current_item())
        if role != "fit" or group is None or fit_entry is None or _covariance_matrix_from_fit_entry(fit_entry) is None:
            return None
        plot = PlotEntry(
            name=_unique_name(f"{fit_entry.name} covariance", [item.name for item in group.plots]),
            type="fit_covariance",
            sources=[PlotSourceRef(fit_id=fit_entry.id)],
        )
        group.plots.append(plot)
        self._mark_dirty()
        self._refresh_tree(select_group=group)
        return plot

    def _request_overlay_refresh(self, group: DataGroup) -> None:
        """Debounce a slice-viewer refresh, coalescing rapid triggers.

        Bursts of edits (dragging a value, fast typing) or repeated tree
        refreshes collapse into a single recompute after a short idle, so the
        (expensive) overlay is not re-evaluated on every event. Outside the Qt
        event loop -- headless/test use, where ``_interactive`` is False -- it
        refreshes synchronously so callers see the update immediately.
        """

        if id(group) not in self._slice_viewers:
            return
        if not self._interactive:
            self.refresh_slice_viewer(group)
            return
        self._pending_overlay_groups[id(group)] = group
        timer = self._overlay_refresh_timer
        if timer is None:
            try:
                from PySide6 import QtCore
            except Exception:
                self.refresh_slice_viewer(group)
                return
            timer = QtCore.QTimer(self.window)
            timer.setSingleShot(True)
            timer.setInterval(200)
            timer.timeout.connect(self._run_pending_overlay_refresh)
            self._overlay_refresh_timer = timer
        timer.start()

    def _run_pending_overlay_refresh(self) -> None:
        pending = list(self._pending_overlay_groups.values())
        self._pending_overlay_groups.clear()
        for group in pending:
            self.refresh_slice_viewer(group)

    def refresh_slice_viewer(self, group: DataGroup, *, force_rebin: bool = False) -> Any | None:
        if id(group) not in self._slice_viewers:
            return None
        viewers = list(self._slice_viewers[id(group)])
        prepared: dict[tuple[bool, bool], tuple[list[MDHistoData], list[str]]] = {}
        for current_viewer in viewers:
            selected_name = None
            selected_binning_name = None
            use_composite = bool(getattr(current_viewer, "_nfit_use_composite", True))
            unmask_model = bool(getattr(current_viewer, "unmask_model", False))
            if current_viewer.dataset_combo is not None:
                selected_name = current_viewer.dataset_combo.currentText()
            if getattr(current_viewer, "binning_combo", None) is not None:
                selected_binning_name = current_viewer.binning_combo.currentText()
            cache_key = (use_composite, unmask_model)
            try:
                if cache_key not in prepared:
                    prepared[cache_key] = slice_viewer_datasets(
                        group,
                        use_composite=use_composite,
                        unmask_model=unmask_model,
                        force_rebin=force_rebin,
                        force_masks=False,
                    )
                datasets, names = prepared[cache_key]
            except Exception:
                continue
            if not datasets:
                self._forget_slice_viewer(group, current_viewer, close=True)
                continue
            if hasattr(current_viewer, "replace_datasets"):
                replacement = {
                    "dataset_names": names,
                    "dataset_group_keys": _waterfall_group_keys(group, names),
                    "selected_dataset_name": selected_name,
                }
                if hasattr(current_viewer, "crystal_contexts"):
                    replacement["crystal_contexts"] = [
                        {
                            "spacegroup": group.spacegroup,
                            "lattice_parameters": copy.deepcopy(group.lattice_parameters),
                        }
                        for _dataset in datasets
                    ]
                if getattr(current_viewer, "binning_combo", None) is not None:
                    replacement["selected_binning_name"] = selected_binning_name
                current_viewer.replace_datasets(datasets, **replacement)
        if prepared:
            # One viewer refresh may prepare multiple dataset or composite
            # binnings, including entries other than the current tree row.
            self._refresh_cache_badges()
        remaining = self._slice_viewers.get(id(group), [])
        return remaining[0] if remaining else None

    def _rebin_progress_callback_for_group(self, group: DataGroup, *, use_composite: bool = True) -> Any | None:
        if use_composite and any(
            data_group_composite_enabled(scope)
            and _composite_rebin_is_large(scope, data_group_composite_config(scope))
            for scope in _composite_scopes(group)
        ):
            return self._make_rebin_progress_callback(
                "Rebinning composite datasets...",
                aggregate=True,
            )
        if not use_composite and any(
            dataset.kind == "mdevent" and dataset.data is None
            for dataset in group.iter_datasets()
        ):
            return self._make_rebin_progress_callback(
                "Loading MDEvent data...",
                aggregate=True,
            )
        if not any(
            dataset_rebin_enabled(dataset)
            and _dataset_rebin_is_large(dataset, dataset_rebin_config(dataset))
            for dataset in group.iter_datasets()
        ):
            return None
        return self._make_rebin_progress_callback(
            "Rebinning datasets...",
            aggregate=True,
        )

    def _make_rebin_progress_callback(
        self,
        title: str,
        *,
        aggregate: bool = False,
    ) -> Any | None:
        try:
            progress = _RebinProgressDialog(self, title, aggregate=aggregate)
        except ImportError:
            return None
        progress.show()

        def callback(event: dict[str, Any]) -> None:
            progress.update_progress(event)
            if progress.cancel_requested:
                raise RebinCancellationRequested("Rebin cancelled by user.")

        dialog = progress.dialog
        callback._nfit_progress_dialog = dialog
        callback._nfit_progress_controller = progress
        return callback

    def _close_rebin_progress(self, callback: Any | None) -> None:
        controller = getattr(callback, "_nfit_progress_controller", None)
        if controller is not None:
            controller.close()
            return
        dialog = getattr(callback, "_nfit_progress_dialog", None)
        if dialog is not None:
            dialog.close()

    def copy_selected(self) -> None:
        _group, entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role == "dataset" and entry is not None:
            datasets = [
                self._objects_for_item(item)[1]
                for item in self._selected_items_for_drag_role("dataset")
            ]
            datasets = [item for item in datasets if item is not None]
            if len(datasets) > 1:
                self._clipboard = ("datasets", copy.deepcopy(datasets))
            else:
                self._clipboard = ("dataset", copy.deepcopy(entry))
        elif role == "mask" and mask is not None:
            self._clipboard = ("mask", copy.deepcopy(mask))
        elif role == "dataset_group":
            subgroup = self._dataset_group_for_item(self._current_item())
            if subgroup is not None:
                self._clipboard = ("dataset_group", copy.deepcopy(subgroup))

    def paste_into_selection(self) -> None:
        if self._clipboard is None:
            return
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        clip_role, payload = self._clipboard
        if (
            clip_role == "dataset"
            and isinstance(payload, DatasetEntry)
            and role in {"group", "datasets", "dataset_group"}
            and group is not None
        ):
            target = (
                self._dataset_group_for_item(self._current_item())
                if role == "dataset_group"
                else group
            )
            if target is None:
                return
            copied = copy.deepcopy(payload).copy()
            copied.name = _unique_name(copied.name, group.dataset_names)
            target.datasets.append(copied)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_dataset=copied)
        elif (
            clip_role == "datasets"
            and isinstance(payload, list)
            and role in {"group", "datasets", "dataset_group"}
            and group is not None
        ):
            target = (
                self._dataset_group_for_item(self._current_item())
                if role == "dataset_group"
                else group
            )
            if target is None:
                return
            copied = [
                copy.deepcopy(item).copy()
                for item in payload
                if isinstance(item, DatasetEntry)
            ]
            if not copied:
                return
            used_names = set(group.dataset_names)
            for item in copied:
                item.name = _unique_name(item.name, used_names)
                used_names.add(item.name)
                target.datasets.append(item)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_dataset=copied[-1])
        elif (
            clip_role == "dataset_group"
            and isinstance(payload, DatasetGroup)
            and role in {"group", "datasets", "dataset_group"}
            and group is not None
        ):
            parent = (
                self._dataset_group_for_item(self._current_item())
                if role == "dataset_group"
                else group
            )
            if parent is None:
                return
            copied = copy_dataset_group_to_parent(group, payload, parent)
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_dataset_group=copied)
        elif clip_role == "mask" and isinstance(payload, MaskSpec) and role in {"dataset", "masks"} and group is not None and entry is not None:
            copied = copy_mask_to_dataset(payload, entry)
            self._mark_mask_datasets_stale([entry])
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=group, select_mask=copied)

    def rename_selected(self) -> None:
        item = self._current_item()
        if item is not None and self._is_renameable_item(item):
            self.tree.editItem(item, 0)

    def expand_all(self) -> None:
        def expand_structure(item: Any) -> None:
            role = self._objects_for_item(item)[4]
            if role == "dataset_page":
                item.setExpanded(False)
                return
            item.setExpanded(True)
            for index in range(item.childCount()):
                expand_structure(item.child(index))

        for index in range(self.tree.topLevelItemCount()):
            expand_structure(self.tree.topLevelItem(index))
        self._expanded_state = self._current_expanded_state()

    def collapse_all(self) -> None:
        self.tree.collapseAll()
        self._expanded_state = self._current_expanded_state()

    def _is_renameable_item(self, item: Any) -> bool:
        return _is_renameable_role(self._objects_for_item(item)[4])

    def show_file_location_for_selection(self) -> bool:
        _group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        source = _dataset_source_path(entry)
        if source is None:
            return False
        return _show_file_location(source)

    def change_file_source_for_selection(self) -> bool:

        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return False
        current = _dataset_source_path(entry)
        start_dir = str(current.parent) if current is not None else ""
        path, _selected_filter = get_open_file_name(
            self.window,
            "Change dataset source",
            start_dir,
            "Data files (*);;All files (*)",
        )
        if not path:
            return False
        set_dataset_source(entry, path)
        self._mark_dirty()
        if group is not None:
            self._refresh_tree(select_group=group, select_dataset=entry)
        return True

    def _resolve_dataset_drop_target(self, target_item: Any) -> tuple[DataGroup | None, Any]:
        """Return the (data group, container node) a dataset drop should land in."""

        group, entry, _mask, _model, role = self._objects_for_item(target_item)
        if group is None:
            return None, None
        if role in {"group", "datasets"}:
            return group, group
        if role == "dataset_group":
            node = self._dataset_group_for_item(target_item)
            return (group, node) if node is not None else (None, None)
        if role in {"dataset", "masks"} and entry is not None:
            node = _dataset_parent_node(group, entry)
            return (group, node) if node is not None else (None, None)
        return None, None

    def load_dropped_paths(self, paths: list[str | Path], target_item: Any = None) -> bool:
        """Open one dropped project or import dropped data files into the tree."""

        from PySide6 import QtWidgets

        dropped = [Path(path).expanduser() for path in paths if str(path)]
        if not dropped:
            return False
        project_paths = [path for path in dropped if path.suffix.lower() == ".nfit"]
        data_paths = [path for path in dropped if path.suffix.lower() != ".nfit"]
        if project_paths:
            if len(project_paths) != 1 or data_paths:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Open nfit project",
                    "Drop one nfit project file at a time. Drop datasets separately.",
                )
                return False
            return self.open_project_path(project_paths[0])

        group, node = self._resolve_dataset_drop_target(target_item)
        if group is None:
            group, node = self._selected_import_target()
        created_group = False
        if group is None and not self.project.data_groups:
            group = create_data_group(self.project)
            node = None
            created_group = True
        if group is None:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Import datasets",
                "Drop datasets onto a workspace or dataset group, or select one first.",
            )
            return False

        started = self._request_dataset_import(
            group,
            data_paths,
            into=node if isinstance(node, DatasetGroup) else None,
        )
        if started:
            return True
        if created_group:
            self.project.data_groups.remove(group)
            self._refresh_tree()
        return False

    def _tree_item_sort_key(self, item: Any) -> tuple[int, ...]:
        path: list[int] = []
        while item is not None:
            parent = item.parent()
            if parent is None:
                path.append(self.tree.indexOfTopLevelItem(item))
            else:
                path.append(parent.indexOfChild(item))
            item = parent
        return tuple(reversed(path))

    def _selected_items_for_drag_role(self, role: str) -> list[Any]:
        items = [item for item in self.tree.selectedItems() if self._objects_for_item(item)[4] == role]
        current = self._current_item()
        if current is not None and self._objects_for_item(current)[4] == role and current not in items:
            items.append(current)
        return sorted(items, key=self._tree_item_sort_key)

    def _is_below_drop(self, drop_position: Any) -> bool:
        from PySide6 import QtWidgets

        return drop_position == QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem

    def _dataset_drop_target_with_index(
        self,
        target_item: Any,
        drop_position: Any = None,
    ) -> tuple[DataGroup | None, Any, int | None]:
        target_group, target_node = self._resolve_dataset_drop_target(target_item)
        if target_group is None or target_node is None:
            return None, None, None
        _group, target_entry, _mask, _model, target_role = self._objects_for_item(target_item)
        if target_role == "dataset" and target_entry is not None:
            parent_node = _dataset_parent_node(target_group, target_entry)
            if parent_node is not None:
                index = parent_node.datasets.index(target_entry)
                if self._is_below_drop(drop_position):
                    index += 1
                return target_group, parent_node, index
        return target_group, target_node, len(target_node.datasets)

    def _mask_drop_target_with_index(
        self,
        target_item: Any,
        drop_position: Any = None,
    ) -> tuple[DataGroup | None, DatasetEntry | None, int | None]:
        group, entry, target_mask, _model, role = self._objects_for_item(target_item)
        if group is None or entry is None or role not in {"dataset", "masks", "mask"}:
            return None, None, None
        if role == "mask" and target_mask is not None:
            index = entry.masks.index(target_mask)
            if self._is_below_drop(drop_position):
                index += 1
            return group, entry, index
        return group, entry, len(entry.masks)

    def _move_or_copy_datasets(
        self,
        dataset_items: list[Any],
        target_item: Any,
        *,
        copy_item: bool,
        drop_position: Any = None,
    ) -> bool:
        target_group, target_node, insert_index = self._dataset_drop_target_with_index(target_item, drop_position)
        if target_group is None or target_node is None:
            return False
        entries: list[tuple[DataGroup, DatasetEntry]] = []
        for item in dataset_items:
            source_group, source_entry, _mask, _model, role = self._objects_for_item(item)
            if role == "dataset" and source_group is not None and source_entry is not None:
                entries.append((source_group, source_entry))
        if not entries:
            return False
        touched_groups: set[int] = set()
        groups_by_id: dict[int, DataGroup] = {}
        last_entry: DatasetEntry | None = None

        def touch(group: DataGroup) -> None:
            touched_groups.add(id(group))
            groups_by_id[id(group)] = group

        if not copy_item and all(
            source_group is target_group and _dataset_parent_node(source_group, source_entry) is target_node
            for source_group, source_entry in entries
        ):
            if _move_items_within_list(target_node.datasets, [entry for _group, entry in entries], insert_index or 0):
                touch(target_group)
                last_entry = entries[-1][1]
            else:
                last_entry = entries[-1][1]
            for group in groups_by_id.values():
                self._record_data_group_state_change(group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_dataset=last_entry)
            return True

        index = len(target_node.datasets) if insert_index is None else max(0, min(insert_index, len(target_node.datasets)))
        for source_group, source_entry in entries:
            if copy_item:
                new_entry = copy_dataset_to_group(source_entry, target_group)
                target_group.datasets.remove(new_entry)
                target_node.datasets.insert(index, new_entry)
                index += 1
                touch(target_group)
                last_entry = new_entry
            elif source_group is target_group:
                parent_node = _dataset_parent_node(source_group, source_entry)
                if parent_node is not None:
                    parent_node.datasets.remove(source_entry)
                    target_node.datasets.insert(index, source_entry)
                    index += 1
                touch(source_group)
                last_entry = source_entry
            else:
                new_entry = copy_dataset_to_group(source_entry, target_group)
                target_group.datasets.remove(new_entry)
                target_node.datasets.insert(index, new_entry)
                index += 1
                delete_dataset(source_group, source_entry)
                touch(source_group)
                touch(target_group)
                last_entry = new_entry
        for group in groups_by_id.values():
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=target_group, select_dataset=last_entry)
        return True

    def _move_selected_groups(self, target_item: Any, drop_position: Any = None) -> bool:
        _source_group, _entry, _mask, _model, target_role = self._objects_for_item(target_item)
        target_group = self._objects_for_item(target_item)[0]
        if target_role != "group" or target_group is None:
            return False
        group_items = self._selected_items_for_drag_role("group")
        groups = [self._objects_for_item(item)[0] for item in group_items]
        groups = [group for group in groups if group is not None]
        if not groups:
            return False
        index = self.project.data_groups.index(target_group)
        if self._is_below_drop(drop_position):
            index += 1
        moved = _move_items_within_list(self.project.data_groups, groups, index)
        self._mark_dirty()
        self._refresh_tree(select_group=groups[-1] if groups else target_group)
        return moved or True

    def _move_or_copy_masks(
        self,
        mask_items: list[Any],
        target_item: Any,
        *,
        copy_item: bool,
        drop_position: Any = None,
    ) -> bool:
        target_group, target_entry, insert_index = self._mask_drop_target_with_index(target_item, drop_position)
        if target_group is None or target_entry is None:
            return False
        masks: list[tuple[DataGroup, DatasetEntry, MaskSpec]] = []
        for item in mask_items:
            source_group, source_entry, source_mask, _model, role = self._objects_for_item(item)
            if role == "mask" and source_group is not None and source_entry is not None and source_mask is not None:
                masks.append((source_group, source_entry, source_mask))
        if not masks:
            return False
        if not copy_item and all(source_entry is target_entry for _group, source_entry, _mask in masks):
            _move_items_within_list(target_entry.masks, [mask for _group, _entry, mask in masks], insert_index or 0)
            self._record_data_group_state_change(target_group)
            self._mark_dirty()
            self._refresh_tree(select_group=target_group, select_mask=masks[-1][2])
            return True
        index = len(target_entry.masks) if insert_index is None else max(0, min(insert_index, len(target_entry.masks)))
        touched_groups: dict[int, DataGroup] = {}
        last_mask: MaskSpec | None = None
        for source_group, source_entry, source_mask in masks:
            moved = copy_mask_to_dataset(source_mask, target_entry)
            target_entry.masks.remove(moved)
            target_entry.masks.insert(index, moved)
            index += 1
            last_mask = moved
            touched_groups[id(target_group)] = target_group
            if not copy_item:
                delete_mask(source_entry, source_mask)
                touched_groups[id(source_group)] = source_group
        for group in touched_groups.values():
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._refresh_tree(select_group=target_group, select_mask=last_mask)
        return True

    def move_or_copy_selected_to_item(self, target_item: Any, *, copy_item: bool, drop_position: Any = None) -> bool:
        source_item = self._current_item()
        source_group, source_entry, source_mask, _source_model, source_role = self._objects_for_item(source_item)
        target_group, target_entry, target_mask, _target_model, target_role = self._objects_for_item(target_item)

        if source_role == "group" and not copy_item:
            return self._move_selected_groups(target_item, drop_position)

        # Move/copy one or more selected datasets into the drop target.
        if source_role == "dataset":
            return self._move_or_copy_datasets(
                self._selected_items_for_drag_role("dataset"),
                target_item,
                copy_item=copy_item,
                drop_position=drop_position,
            )

        # Re-parent a subgroup within the same group.
        if (
            source_role == "dataset_group"
            and not copy_item
            and source_group is not None
            and target_group is source_group
            and target_role in {"group", "datasets", "dataset_group"}
        ):
            subgroup = self._dataset_group_for_item(source_item)
            target_node = source_group if target_role in {"group", "datasets"} else self._dataset_group_for_item(target_item)
            if (
                subgroup is not None
                and target_node is not None
                and not _group_contains_node(subgroup, target_node)
            ):
                parent_node = _dataset_group_parent(source_group, subgroup)
                if parent_node is not None and parent_node is not target_node:
                    parent_node.subgroups.remove(subgroup)
                    target_node.subgroups.append(subgroup)
                    self._record_data_group_state_change(source_group)
                    self._mark_dirty()
                    self._refresh_tree(select_group=source_group, select_dataset_group=subgroup)
            return True

        if source_role == "mask":
            return self._move_or_copy_masks(
                self._selected_items_for_drag_role("mask"),
                target_item,
                copy_item=copy_item,
                drop_position=drop_position,
            )
        return False

    def show_help(self) -> None:
        """Open bundled offline documentation or the local source build."""
        from PySide6 import QtCore, QtGui, QtWidgets

        from .app_distribution import local_help_index

        index = local_help_index(source_file=__file__)
        if index is None:
            QtWidgets.QMessageBox.information(
                self.window,
                "Documentation not built",
                "The local documentation is missing.\n\n"
                "Build it from the nfit source directory with:\n"
                "python -m sphinx -b html docs docs/_build/html",
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(index)))

    def show_preferences(self) -> None:
        from .preferences_gui import PreferencesDialog

        PreferencesDialog(self.window).exec()

    def _build(self) -> None:
        from .project_window_builder import build_project_window

        build_project_window(
            self,
            project_window_class_factory=_make_project_window_class,
            project_tree_class_factory=_make_project_tree_class,
            refreshing_combo_class_factory=_make_refreshing_combo_class,
        )

    def _refresh_tree(
        self,
        *,
        select_group: DataGroup | None = None,
        select_dataset: DatasetEntry | None = None,
        select_mask: MaskSpec | None = None,
        select_background: BackgroundSpec | None = None,
        select_model: ModelComponentSpec | None = None,
        select_fit: FitTimelineEntry | None = None,
        select_plot: PlotEntry | None = None,
        select_dataset_group: DatasetGroup | None = None,
        edit_group: bool = False,
        edit_mask: bool = False,
        edit_model: bool = False,
        refresh_viewers: bool = True,
    ) -> None:
        from PySide6 import QtCore, QtWidgets

        self._expanded_state = self._current_expanded_state()
        self._item_roles.clear()
        self._fit_item_roles.clear()
        self._background_item_roles.clear()
        self._analysis_output_roles.clear()
        self._analysis_item_roles.clear()
        self._plot_item_roles.clear()
        self._dataset_group_roles.clear()
        self._dataset_page_roles.clear()
        self.tree.blockSignals(True)
        # Clear the stale accessible/current index before removing every row.
        # Otherwise Qt's accessibility bridge can query the old timeline row
        # while QTreeWidget has already resized its model to zero rows.
        self.tree.clearSelection()
        self.tree.setCurrentItem(None)
        self.tree.clear()
        item_to_select = None
        for group in self.project.data_groups:
            ensure_fit_history(group)
            # Selecting a historical result restores its snapshot into the
            # live group so its fit settings can be inspected. Do not let
            # that restoration overwrite the separately saved editable
            # Current state snapshot.
            active_fit = self._active_fit_entry(group)
            if active_fit is None or active_fit.kind == "current":
                refresh_current_state_fit_entries(group)
            group_item = QtWidgets.QTreeWidgetItem([group.name])
            group_item.setFlags(group_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            _style_tree_hierarchy_item(group_item, bold=True, underline=True)
            _set_tree_item_icon(group_item, "folder")
            self._remember_item(group_item, "group", group)
            self.tree.addTopLevelItem(group_item)

            datasets_item = QtWidgets.QTreeWidgetItem(["Datasets"])
            _style_tree_hierarchy_item(datasets_item, bold=True)
            root_cached = _composite_cached_binnings_current(group, group)
            _set_tree_item_icon(datasets_item, "folder", cached=root_cached)
            if root_cached:
                datasets_item.setToolTip(
                    0, "All enabled composite binnings are cached and up to date."
                )
            self._remember_item(datasets_item, "datasets", group)
            group_item.addChild(datasets_item)
            found = self._render_dataset_node(
                datasets_item,
                group,
                group,
                select_dataset=select_dataset,
                select_mask=select_mask,
                select_background=select_background,
                select_dataset_group=select_dataset_group,
            )
            if found is not None and item_to_select is None:
                item_to_select = found

            models_item = QtWidgets.QTreeWidgetItem(["Models"])
            _style_tree_hierarchy_item(models_item, bold=True)
            _set_tree_item_icon(models_item, "model_folder")
            self._remember_item(models_item, "models", group)
            group_item.addChild(models_item)
            for name, model in group.models.items():
                model_item = QtWidgets.QTreeWidgetItem([name])
                _set_tree_item_icon(model_item, "model")
                if isinstance(model, ModelComponentSpec):
                    model_item.setFlags(model_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    self._remember_item(model_item, "model", group, model=model)
                    _style_enabled_tree_item(model_item, model.enabled)
                    if select_model is model:
                        item_to_select = model_item
                else:
                    self._remember_item(model_item, "fit_model_session", group)
                models_item.addChild(model_item)

            group_item.setExpanded(self._expanded_state.get(("group", id(group)), True))
            datasets_item.setExpanded(self._expanded_state.get(("datasets", id(group)), True))
            models_item.setExpanded(self._expanded_state.get(("models", id(group)), True))
            fits_item = QtWidgets.QTreeWidgetItem(["Fits"])
            _style_tree_hierarchy_item(fits_item, bold=True)
            _set_tree_item_icon(fits_item, "fit_folder")
            self._remember_item(fits_item, "fits", group)
            group_item.addChild(fits_item)
            for fit_entry in group.fits:
                fit_item = self._add_fit_tree_item(fits_item, group, fit_entry, select_fit=select_fit)
                if item_to_select is None and select_fit is not None:
                    item_to_select = self._selected_fit_tree_item(fit_item, select_fit)
            fits_item.setExpanded(self._expanded_state.get(("fits", id(group)), True))
            analyses_item = QtWidgets.QTreeWidgetItem(["Analyses"])
            _style_tree_hierarchy_item(analyses_item, bold=True)
            _set_tree_item_icon(analyses_item, "analysis_folder")
            self._remember_item(analyses_item, "analyses", group)
            group_item.addChild(analyses_item)
            for analysis in group.analyses:
                status = self._analysis_display_status(group, analysis)
                analysis_item = QtWidgets.QTreeWidgetItem([f"{analysis.name} [{status}]"])
                analysis_item.setToolTip(0, f"{analysis.type}; status: {status}")
                _set_tree_item_icon(analysis_item, "model")
                self._remember_item(analysis_item, "analysis", group)
                self._analysis_item_roles[id(analysis_item)] = analysis
                analyses_item.addChild(analysis_item)
                if analysis.result is not None:
                    for output in analysis.result.outputs:
                        output_item = QtWidgets.QTreeWidgetItem([output.label])
                        output_item.setToolTip(0, f"{output.kind} output: {output.key}")
                        _set_tree_item_icon(output_item, "dataset")
                        self._remember_item(output_item, "analysis_output", group)
                        self._analysis_output_roles[id(output_item)] = output
                        analysis_item.addChild(output_item)
            analyses_item.setExpanded(self._expanded_state.get(("analyses", id(group)), True))
            plots_item = QtWidgets.QTreeWidgetItem(["Plots"])
            _style_tree_hierarchy_item(plots_item, bold=True)
            _set_tree_item_icon(plots_item, "plot_folder")
            self._remember_item(plots_item, "plots", group)
            group_item.addChild(plots_item)
            for plot in group.plots:
                plot_item = QtWidgets.QTreeWidgetItem([plot.name])
                plot_item.setFlags(plot_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                plot_item.setToolTip(0, f"{plot.type}; reopen, edit, export, or render this saved plot")
                _set_tree_item_icon(plot_item, "dataset")
                self._remember_item(plot_item, "plot", group)
                self._plot_item_roles[id(plot_item)] = plot
                plots_item.addChild(plot_item)
                if select_plot is plot:
                    item_to_select = plot_item
            plots_item.setExpanded(self._expanded_state.get(("plots", id(group)), True))
            if select_group is group and item_to_select is None:
                item_to_select = group_item
        self.tree.blockSignals(False)
        if item_to_select is not None:
            self.tree.setCurrentItem(item_to_select)
            if edit_group:
                self.tree.editItem(item_to_select, 0)
            if edit_mask:
                self.tree.editItem(item_to_select, 0)
            if edit_model:
                self.tree.editItem(item_to_select, 0)
        self._sync_details()
        if refresh_viewers:
            self.refresh_open_slice_viewers()

    def _render_dataset_node(
        self,
        parent_item: Any,
        group: DataGroup,
        node: Any,
        *,
        select_dataset: DatasetEntry | None,
        select_mask: MaskSpec | None,
        select_background: BackgroundSpec | None,
        select_dataset_group: DatasetGroup | None,
    ) -> Any:
        """Recursively render a group's datasets and nested subgroups. Returns any matched item."""

        from PySide6 import QtCore, QtWidgets

        found = None
        direct_datasets = list(node.datasets)
        if len(direct_datasets) > TREE_DATASET_COMPACT_THRESHOLD:
            for start in range(0, len(direct_datasets), TREE_DATASET_PAGE_SIZE):
                page = tuple(direct_datasets[start : start + TREE_DATASET_PAGE_SIZE])
                end = start + len(page)
                page_item = QtWidgets.QTreeWidgetItem(
                    [_dataset_tree_page_label(page, start=start, total=len(direct_datasets))]
                )
                page_item.setToolTip(
                    0,
                    f"Runs {start + 1}-{end} of {len(direct_datasets)}. Expand to load only this page into the tree.",
                )
                _set_tree_item_icon(page_item, "folder")
                self._remember_item(page_item, "dataset_page", group)
                self._dataset_page_roles[id(page_item)] = (node, page, start)
                parent_item.addChild(page_item)
                target_on_page = (
                    select_dataset in page
                    or any(select_mask in dataset.masks for dataset in page)
                    or any(select_background in dataset.backgrounds for dataset in page)
                )
                page_key = ("dataset_page", id(node), start)
                if target_on_page or self._expanded_state.get(page_key, False):
                    page_found = self._populate_dataset_page(
                        page_item,
                        select_dataset=select_dataset,
                        select_mask=select_mask,
                        select_background=select_background,
                    )
                    page_item.setExpanded(True)
                    if page_found is not None and found is None:
                        found = page_found
                else:
                    placeholder = QtWidgets.QTreeWidgetItem(["Load runs..."])
                    placeholder.setDisabled(True)
                    page_item.addChild(placeholder)
        else:
            for dataset in direct_datasets:
                dataset_found = self._add_dataset_tree_item(
                    parent_item,
                    group,
                    dataset,
                    select_dataset=select_dataset,
                    select_mask=select_mask,
                    select_background=select_background,
                )
                if dataset_found is not None and found is None:
                    found = dataset_found

        if node.backgrounds:
            group_backgrounds_item = QtWidgets.QTreeWidgetItem(["Backgrounds"])
            group_backgrounds_item.setToolTip(
                0,
                "Background histograms subtracted after the enabled datasets in this group are combined.",
            )
            _set_tree_item_icon(group_backgrounds_item, "background_folder")
            self._remember_item(
                group_backgrounds_item, "group_backgrounds", group, node=node
            )
            parent_item.addChild(group_backgrounds_item)
            for background in node.backgrounds:
                background_item = QtWidgets.QTreeWidgetItem([background.name])
                background_item.setFlags(
                    background_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable
                )
                background_item.setToolTip(
                    0, "Background histogram applied once to this group's composite."
                )
                _set_tree_item_icon(background_item, "dataset")
                self._remember_item(
                    background_item, "group_background", group, node=node
                )
                self._background_item_roles[id(background_item)] = background
                _style_enabled_tree_item(background_item, background.enabled)
                group_backgrounds_item.addChild(background_item)
                if select_background is background and found is None:
                    found = background_item
            group_backgrounds_item.setExpanded(
                self._expanded_state.get(("group_backgrounds", id(node)), False)
            )

        for subgroup in node.subgroups:
            subgroup_item = QtWidgets.QTreeWidgetItem([subgroup.name])
            subgroup_item.setFlags(subgroup_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            subgroup_cached = _composite_cached_binnings_current(group, subgroup)
            _set_tree_item_icon(
                subgroup_item, "folder", cached=subgroup_cached
            )
            if subgroup_cached:
                subgroup_item.setToolTip(
                    0, "All enabled composite binnings are cached and up to date."
                )
            self._remember_item(subgroup_item, "dataset_group", group, node=subgroup)
            _style_enabled_tree_item(subgroup_item, subgroup.enabled)
            parent_item.addChild(subgroup_item)
            gmasks_item = None
            if subgroup.masks:
                gmasks_item = QtWidgets.QTreeWidgetItem(["Masks"])
                _set_tree_item_icon(gmasks_item, "mask_folder")
                self._remember_item(gmasks_item, "group_masks", group, node=subgroup)
                subgroup_item.addChild(gmasks_item)
                for mask in subgroup.masks:
                    gmask_item = QtWidgets.QTreeWidgetItem([mask.name])
                    gmask_item.setFlags(gmask_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    _set_tree_item_icon(gmask_item, "mask")
                    self._remember_item(gmask_item, "group_mask", group, mask=mask, node=subgroup)
                    _style_enabled_tree_item(gmask_item, mask.enabled)
                    gmasks_item.addChild(gmask_item)
                    if select_mask is mask and found is None:
                        found = gmask_item
            child_found = self._render_dataset_node(
                subgroup_item,
                group,
                subgroup,
                select_dataset=select_dataset,
                select_mask=select_mask,
                select_background=select_background,
                select_dataset_group=select_dataset_group,
            )
            if child_found is not None and found is None:
                found = child_found
            if select_dataset_group is subgroup and found is None:
                found = subgroup_item
            default_expanded = len(subgroup.datasets) <= TREE_DATASET_COMPACT_THRESHOLD
            subgroup_item.setExpanded(
                self._expanded_state.get(
                    ("dataset_group", id(subgroup)), default_expanded
                )
            )
            if gmasks_item is not None:
                gmasks_item.setExpanded(
                    self._expanded_state.get(("group_masks", id(subgroup)), False)
                )
        return found

    def _refresh_cache_badges(self, item: Any | None = None) -> None:
        """Refresh green cache indicators without rebuilding the project tree."""

        from PySide6 import QtCore

        def tree_items(parent: Any | None = None):
            count = (
                self.tree.topLevelItemCount()
                if parent is None
                else parent.childCount()
            )
            for index in range(count):
                item = (
                    self.tree.topLevelItem(index)
                    if parent is None
                    else parent.child(index)
                )
                yield item
                yield from tree_items(item)

        blocker = QtCore.QSignalBlocker(self.tree)
        try:
            items = tree_items() if item is None else (item,)
            for candidate in items:
                group, entry, _mask, _model, role = self._objects_for_item(candidate)
                if role == "dataset" and group is not None and entry is not None:
                    current = _dataset_cached_binnings_current(group, entry)
                    _set_tree_item_icon(candidate, "dataset", cached=current)
                    candidate.setToolTip(
                        0,
                        "All enabled binnings are cached and up to date."
                        if current
                        else "",
                    )
                elif role == "dataset_group" and group is not None:
                    node = self._dataset_group_for_item(candidate)
                    if node is not None:
                        current = _composite_cached_binnings_current(group, node)
                        _set_tree_item_icon(candidate, "folder", cached=current)
                        candidate.setToolTip(
                            0,
                            "All enabled composite binnings are cached and up to date."
                            if current
                            else "",
                        )
                elif role == "datasets" and group is not None:
                    current = _composite_cached_binnings_current(group, group)
                    _set_tree_item_icon(candidate, "folder", cached=current)
                    candidate.setToolTip(
                        0,
                        "All enabled composite binnings are cached and up to date."
                        if current
                        else "",
                    )
        finally:
            del blocker

    def _add_dataset_tree_item(
        self,
        parent_item: Any,
        group: DataGroup,
        dataset: DatasetEntry,
        *,
        select_dataset: DatasetEntry | None = None,
        select_mask: MaskSpec | None = None,
        select_background: BackgroundSpec | None = None,
    ) -> Any | None:
        """Add one fully interactive dataset item and return a matched child."""

        from PySide6 import QtCore, QtWidgets

        found = None
        dataset_item = QtWidgets.QTreeWidgetItem([dataset.name])
        dataset_item.setFlags(dataset_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        dataset_cached = _dataset_cached_binnings_current(group, dataset)
        _set_tree_item_icon(dataset_item, "dataset", cached=dataset_cached)
        if dataset_cached:
            dataset_item.setToolTip(
                0, "All enabled binnings are cached and up to date."
            )
        self._remember_item(dataset_item, "dataset", group, dataset)
        _style_enabled_tree_item(dataset_item, dataset.enabled)
        parent_item.addChild(dataset_item)
        masks_item = None
        if dataset.masks:
            masks_item = QtWidgets.QTreeWidgetItem(["Masks"])
            _set_tree_item_icon(masks_item, "mask_folder")
            _style_enabled_tree_item(masks_item, dataset.enabled)
            self._remember_item(masks_item, "masks", group, dataset)
            dataset_item.addChild(masks_item)
            for mask in dataset.masks:
                mask_item = QtWidgets.QTreeWidgetItem([mask.name])
                mask_item.setFlags(mask_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                _set_tree_item_icon(mask_item, "mask")
                self._remember_item(mask_item, "mask", group, dataset, mask)
                _style_enabled_tree_item(mask_item, mask.enabled)
                masks_item.addChild(mask_item)
                if select_mask is mask and found is None:
                    found = mask_item
        backgrounds_item = None
        if dataset.backgrounds:
            backgrounds_item = QtWidgets.QTreeWidgetItem(["Backgrounds"])
            _set_tree_item_icon(backgrounds_item, "background_folder")
            _style_enabled_tree_item(backgrounds_item, dataset.enabled)
            self._remember_item(backgrounds_item, "backgrounds", group, dataset)
            dataset_item.addChild(backgrounds_item)
            for background in dataset.backgrounds:
                background_item = QtWidgets.QTreeWidgetItem([background.name])
                background_item.setFlags(
                    background_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable
                )
                _set_tree_item_icon(background_item, "dataset")
                self._remember_item(background_item, "background", group, dataset)
                self._background_item_roles[id(background_item)] = background
                _style_enabled_tree_item(background_item, background.enabled)
                backgrounds_item.addChild(background_item)
                if select_background is background and found is None:
                    found = background_item
        if select_dataset is dataset and found is None:
            found = dataset_item
        dataset_item.setExpanded(self._expanded_state.get(("dataset", id(dataset)), False))
        if masks_item is not None:
            masks_item.setExpanded(
                self._expanded_state.get(("masks", id(dataset)), False)
            )
        if backgrounds_item is not None:
            backgrounds_item.setExpanded(
                self._expanded_state.get(("backgrounds", id(dataset)), False)
            )
        return found

    def _populate_dataset_page(
        self,
        page_item: Any,
        *,
        select_dataset: DatasetEntry | None = None,
        select_mask: MaskSpec | None = None,
        select_background: BackgroundSpec | None = None,
    ) -> Any | None:
        """Populate one lazy run page, leaving all other pages unloaded."""

        payload = self._dataset_page_roles.get(id(page_item))
        if payload is None:
            return None
        _node, datasets, _start = payload
        if page_item.childCount() and self._objects_for_item(page_item.child(0))[4] == "dataset":
            return None
        page_item.takeChildren()
        group = self._objects_for_item(page_item)[0]
        found = None
        if group is None:
            return None
        for dataset in datasets:
            dataset_found = self._add_dataset_tree_item(
                page_item,
                group,
                dataset,
                select_dataset=select_dataset,
                select_mask=select_mask,
                select_background=select_background,
            )
            if dataset_found is not None and found is None:
                found = dataset_found
        return found

    def _dataset_page_expanded(self, item: Any) -> None:
        payload = self._dataset_page_roles.get(id(item))
        if payload is None:
            return
        node, _datasets, start = payload
        self._expanded_state[("dataset_page", id(node), start)] = True
        self._populate_dataset_page(item)

    def _dataset_page_collapsed(self, item: Any) -> None:
        payload = self._dataset_page_roles.get(id(item))
        if payload is None:
            return
        node, _datasets, start = payload
        self._expanded_state[("dataset_page", id(node), start)] = False

    def _current_expanded_state(self) -> dict[tuple[Any, ...], bool]:
        state: dict[tuple[Any, ...], bool] = {}
        if self.tree is None:
            return state

        def visit(item: Any) -> None:
            group, dataset, _mask, _model, role = self._objects_for_item(item)
            node = self._dataset_group_for_item(item)
            if role == "group" and group is not None:
                state[("group", id(group))] = item.isExpanded()
            elif role in {"datasets", "models", "fits", "analyses", "plots"} and group is not None:
                state[(role, id(group))] = item.isExpanded()
            elif role == "dataset_group" and node is not None:
                state[("dataset_group", id(node))] = item.isExpanded()
            elif role in {"group_masks", "group_backgrounds"} and node is not None:
                state[(role, id(node))] = item.isExpanded()
            elif role == "dataset" and dataset is not None:
                state[("dataset", id(dataset))] = item.isExpanded()
            elif role in {"masks", "backgrounds"} and dataset is not None:
                state[(role, id(dataset))] = item.isExpanded()
            elif role == "dataset_page":
                payload = self._dataset_page_roles.get(id(item))
                if payload is not None:
                    page_node, _datasets, start = payload
                    state[("dataset_page", id(page_node), start)] = item.isExpanded()
            fit_entry = self._fit_entry_for_item(item)
            if fit_entry is not None:
                state[("fit", id(fit_entry))] = item.isExpanded()
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for index in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(index))
        return state

    def _collect_fit_expanded_state(self, item: Any, state: dict[tuple[Any, ...], bool]) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            fit_entry = self._fit_entry_for_item(child)
            if fit_entry is not None:
                state[("fit", id(fit_entry))] = child.isExpanded()
                self._collect_fit_expanded_state(child, state)

    def _remember_item(
        self,
        item: Any,
        role: str,
        group: DataGroup | None = None,
        entry: DatasetEntry | None = None,
        mask: MaskSpec | None = None,
        model: ModelComponentSpec | None = None,
        node: DataGroup | DatasetGroup | None = None,
    ) -> None:
        self._item_roles[id(item)] = (role, group, entry, mask, model)
        if node is not None:
            self._dataset_group_roles[id(item)] = node

    def _dataset_group_for_item(self, item: Any) -> DatasetGroup | None:
        if item is None:
            return None
        return self._dataset_group_roles.get(id(item))

    def _background_owner_for_item(
        self, item: Any
    ) -> DatasetEntry | DataGroup | DatasetGroup | None:
        group, entry, _mask, _model, role = self._objects_for_item(item)
        if role in {"dataset", "backgrounds", "background"}:
            return entry
        if role == "datasets":
            return group
        if role in {"dataset_group", "group_backgrounds", "group_background"}:
            return self._dataset_group_roles.get(id(item)) or group
        return None

    def _add_fit_tree_item(
        self,
        parent_item: Any,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        *,
        select_fit: FitTimelineEntry | None = None,
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        item = QtWidgets.QTreeWidgetItem([fit_entry.name])
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        role = "fit_timeline" if fit_entry.kind == "timeline" else "fit"
        icon_kind = {
            "timeline": "fit_folder",
            "initial": "fit_initial",
            "current": "fit_current",
            "result": "fit_result",
        }.get(fit_entry.kind, "fit_result")
        _set_tree_item_icon(item, icon_kind)
        if self._active_fit_entry(group) is fit_entry:
            _style_active_fit_tree_item(item)
        self._remember_item(item, role, group)
        self._fit_item_roles[id(item)] = fit_entry
        parent_item.addChild(item)
        for child in fit_entry.children:
            self._add_fit_tree_item(item, group, child, select_fit=select_fit)
        item.setExpanded(
            self._expanded_state.get(
                ("fit", id(fit_entry)),
                fit_entry.kind == "timeline" or _fit_entry_in_tree(fit_entry.children, select_fit),
            )
        )
        return item

    def _selected_fit_tree_item(self, item: Any, select_fit: FitTimelineEntry) -> Any | None:
        if self._fit_entry_for_item(item) is select_fit:
            return item
        for index in range(item.childCount()):
            found = self._selected_fit_tree_item(item.child(index), select_fit)
            if found is not None:
                return found
        return None

    def _tree_item_changed(self, item: Any, column: int) -> None:
        if column != 0:
            return
        changed = False
        group, entry, mask, model, role = self._objects_for_item(item)
        if role == "group" and group is not None:
            new_name = item.text(0).strip() or next_data_group_name(self.project.data_groups)
            changed = group.name != new_name
            group.name = new_name
            item.setText(0, group.name)
        elif role == "dataset" and group is not None and entry is not None:
            new_name = _unique_name(
                item.text(0).strip() or entry.name,
                [dataset.name for dataset in group.datasets if dataset is not entry],
            )
            changed = entry.name != new_name
            entry.name = new_name
            item.setText(0, entry.name)
        elif role in {"mask", "group_mask"} and mask is not None:
            new_name = item.text(0).strip() or "mask"
            changed = mask.name != new_name
            mask.name = new_name
            item.setText(0, mask.name)
        elif role in {"background", "group_background"}:
            background = self._background_for_item(item)
            if background is not None:
                new_name = item.text(0).strip() or "Background"
                changed = background.name != new_name
                background.name = new_name
                item.setText(0, background.name)
        elif role == "dataset_group" and group is not None:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None:
                existing = [sub.name for sub in group.iter_subgroups() if sub is not subgroup]
                new_name = _unique_name(item.text(0).strip() or subgroup.name, existing)
                changed = subgroup.name != new_name
                subgroup.name = new_name
                item.setText(0, subgroup.name)
        elif role == "model" and group is not None and model is not None:
            old_name = _model_key(group, model)
            new_name = _unique_name(item.text(0).strip() or model.name, [name for name in group.models if name != old_name])
            if old_name is not None and old_name != new_name:
                _rename_model_constraint_references(group, model, old_name, new_name)
                del group.models[old_name]
                group.models[new_name] = model
                changed = True
            changed = changed or model.name != new_name
            model.name = new_name
            item.setText(0, model.name)
        elif role in {"fit", "fit_timeline"}:
            fit_entry = self._fit_entry_for_item(item)
            if fit_entry is not None:
                new_name = item.text(0).strip() or fit_entry.name
                changed = fit_entry.name != new_name
                fit_entry.name = new_name
                item.setText(0, fit_entry.name)
        elif role == "plot" and group is not None:
            plot = self._plot_for_item(item)
            if plot is not None:
                new_name = _unique_name(item.text(0).strip() or plot.name, [value.name for value in group.plots if value is not plot])
                changed = plot.name != new_name
                plot.name = new_name
                item.setText(0, plot.name)
        if changed:
            if role in {"dataset", "mask", "background", "group_background", "model", "dataset_group", "group_mask"} and group is not None:
                self._record_data_group_state_change(group)
            self._mark_dirty()
        self._sync_details()

    def _current_item(self) -> Any:
        return self.tree.currentItem()

    def _objects_for_item(
        self,
        item: Any,
    ) -> tuple[DataGroup | None, DatasetEntry | None, MaskSpec | None, ModelComponentSpec | None, str]:
        if item is None:
            return None, None, None, None, "project"
        role, group, entry, mask, model = self._item_roles.get(id(item), ("project", None, None, None, None))
        return group, entry, mask, model, role

    def _fit_entry_for_item(self, item: Any) -> FitTimelineEntry | None:
        if item is None:
            return None
        return self._fit_item_roles.get(id(item))

    def _background_for_item(self, item: Any) -> BackgroundSpec | None:
        if item is None:
            return None
        return self._background_item_roles.get(id(item))

    def _plot_for_item(self, item: Any) -> PlotEntry | None:
        return None if item is None else self._plot_item_roles.get(id(item))

    def _sync_details(self) -> None:
        current_item = self._current_item()
        self._refresh_cache_badges(current_item)
        group, entry, mask, model, role = self._objects_for_item(current_item)
        fit_entry = self._fit_entry_for_item(current_item)
        can_import = role in {"group", "datasets", "dataset_group"}
        can_add_model = role in {"group", "models"}
        analysis_selection = role in {"analyses", "analysis", "analysis_output"}
        self._sync_selected_state_controls(role, entry, mask, model)
        self.import_dataset_button.setVisible(can_import)
        self.add_model_button.setVisible(can_add_model)
        self.create_group_button.setVisible(not analysis_selection)
        self.open_analysis_button.setVisible(analysis_selection and group is not None)
        self.new_analysis_button.setVisible(role == "analyses" and group is not None)
        self.view_slice_button.setVisible(
            role in {"group", "datasets", "dataset", "masks", "mask", "backgrounds", "background", "group_backgrounds", "group_background", "dataset_group", "group_masks", "group_mask"}
        )
        self.view_slice_button.setEnabled(bool(group is not None and _has_slice_viewer_candidates(group)))
        self.load_dataset_button.setVisible(
            role == "dataset" and entry is not None and _dataset_can_load(entry)
        )
        can_reload_selection = False
        if role == "dataset" and entry is not None:
            can_reload_selection = _dataset_can_reload(entry)
        elif role in {"group", "datasets"} and group is not None:
            can_reload_selection = any(_dataset_can_reload(item) for item in group.iter_datasets())
        elif role == "dataset_group":
            node = self._dataset_group_for_item(self._current_item())
            can_reload_selection = bool(
                node is not None
                and any(_dataset_can_reload(item) for item in node.iter_datasets())
            )
        self.reload_data_button.setVisible(
            role in {"group", "datasets", "dataset", "dataset_group"}
        )
        self.reload_data_button.setEnabled(can_reload_selection)
        self.add_mask_button.setVisible(role in {"dataset", "masks", "group_masks", "dataset_group"})
        self.add_background_button.setVisible(
            role in {"datasets", "dataset", "backgrounds", "dataset_group", "group_backgrounds"}
        )
        self.add_dataset_group_button.setVisible(role in {"group", "datasets", "dataset_group"})
        self.save_dataset_button.setVisible(role == "dataset")
        self.save_dataset_button.setEnabled(
            bool(role == "dataset" and entry is not None and _dataset_can_save(entry))
        )
        self.delete_button.setEnabled(
            role in _DELETABLE_TREE_ROLES
        )
        mask_editing = role in {"mask", "group_mask"}
        self.mask_type_combo.setVisible(mask_editing)
        self.mask_parameter_widget.setVisible(mask_editing)
        self.model_selector_widget.setVisible(role == "model")
        self.model_parameter_scroll.setVisible(role == "model")
        # The dedicated model editor already contains the selected model's
        # settings. Let it use the generic-details area rather than leaving a
        # redundant model summary below the scrollable controls.
        self.details_scroll.setVisible(role != "model")
        self.fit_editor_widget.setVisible(role == "fit" and fit_entry is not None)
        self.fit_now_button.setVisible(role == "fit" and fit_entry is not None)
        self.fit_corner_button.setVisible(
            role == "fit"
            and fit_entry is not None
            and _fit_entry_has_diagnostic_plots(fit_entry)
        )
        self.show_data_fit_button.setVisible(
            role == "fit"
            and fit_entry is not None
            and group is not None
            and (_group_has_fit_channels(group) or _group_has_enabled_model_components(group))
        )
        self.fit_export_report_button.setVisible(
            role == "fit" and fit_entry is not None and fit_entry.kind == "result"
        )
        fit_script_available = role == "fit" and fit_entry is not None
        self.fit_copy_script_button.setVisible(fit_script_available)
        self.fit_save_script_button.setVisible(fit_script_available)
        self.fit_copy_script_button.setEnabled(fit_script_available)
        self.fit_save_script_button.setEnabled(fit_script_available)

        selected_plot = self._plot_for_item(self._current_item()) if role == "plot" else None
        plot_available = selected_plot is not None
        plot_has_dataset = bool(
            selected_plot is not None
            and selected_plot.sources
            and selected_plot.sources[0].dataset_id
        )
        for button in (
            self.plot_open_button,
            self.plot_edit_button,
            self.plot_copy_script_button,
            self.plot_save_script_button,
        ):
            button.setVisible(plot_available)
        self.plot_open_button.setEnabled(plot_available)
        self.plot_edit_button.setEnabled(plot_has_dataset)
        self.plot_copy_script_button.setEnabled(plot_available and self.project_path is not None)
        self.plot_save_script_button.setEnabled(plot_available and self.project_path is not None)

        if role == "group" and group is not None:
            self.title_label.setText(group.name)
            ensure_fit_history(group)
            self._set_group_details(group)
        elif role == "datasets" and group is not None:
            self.title_label.setText(f"{group.name} / Datasets")
            self._set_dataset_collection_details(group, group)
        elif role == "dataset_page":
            payload = self._dataset_page_roles.get(id(self._current_item()))
            if payload is not None:
                _node, datasets, start = payload
                end = start + len(datasets)
                self.title_label.setText(f"Runs {start + 1}-{end}")
                self._set_details_text(
                    f"Lazy run page\n\nRuns {start + 1}-{end}\n"
                    f"First: {datasets[0].name}\nLast: {datasets[-1].name}\n\n"
                    "Expand this page to inspect or edit individual runs."
                )
        elif role == "dataset" and entry is not None:
            self.title_label.setText(entry.name)
            self._set_dataset_details(entry, group)
        elif role == "masks" and entry is not None:
            self.title_label.setText(f"{entry.name} / Masks")
            self._set_details_text(f"{len(entry.masks)} mask(s)")
        elif role == "mask" and entry is not None and mask is not None:
            self.title_label.setText(mask.name)
            label = MASK_TYPE_DEFINITIONS.get(mask.type, {}).get("label", mask.type)
            self._set_details_text(
                f"Mask\n\nType: {label}\nEnabled: {mask.enabled}\nInvert: {mask.invert}\nAdditive: {mask.additive}"
            )
            self._sync_mask_editor(mask, entry)
        elif role == "backgrounds" and entry is not None:
            self.title_label.setText(f"{entry.name} / Backgrounds")
            self._set_details_text(
                f"{len(entry.backgrounds)} background(s)\n\n"
                "Enabled backgrounds are interpolated onto this dataset and subtracted in list order."
            )
        elif role == "background" and entry is not None:
            background = self._background_for_item(self._current_item())
            if background is not None:
                self.title_label.setText(background.name)
                self._set_background_details(group, entry, background)
        elif role == "group_backgrounds" and group is not None:
            owner = self._background_owner_for_item(self._current_item())
            if owner is not None:
                self.title_label.setText(f"{owner.name} / Backgrounds")
                self._set_details_text(
                    f"{len(owner.backgrounds)} background(s)\n\n"
                    "Enabled backgrounds are subtracted once, after this "
                    "group's enabled datasets are combined."
                )
        elif role == "group_background" and group is not None:
            owner = self._background_owner_for_item(self._current_item())
            background = self._background_for_item(self._current_item())
            if owner is not None and background is not None:
                self.title_label.setText(background.name)
                self._set_background_details(group, owner, background)
        elif role == "dataset_group":
            subgroup = self._dataset_group_for_item(self._current_item())
            name = subgroup.name if subgroup is not None else "Dataset group"
            self.title_label.setText(name)
            if group is not None and subgroup is not None:
                self._set_dataset_collection_details(group, subgroup)
            else:
                self._set_details_text("Dataset group")
        elif role == "group_masks":
            subgroup = self._dataset_group_for_item(self._current_item())
            name = subgroup.name if subgroup is not None else "-"
            mask_count = len(subgroup.masks) if subgroup is not None else 0
            self.title_label.setText(f"{name} / Masks")
            self._set_details_text(f"{mask_count} shared mask(s)")
        elif role == "group_mask" and mask is not None:
            subgroup = self._dataset_group_for_item(self._current_item())
            self.title_label.setText(mask.name)
            label = MASK_TYPE_DEFINITIONS.get(mask.type, {}).get("label", mask.type)
            self._set_details_text(
                f"Shared mask\n\nType: {label}\nEnabled: {mask.enabled}\nInvert: {mask.invert}\nAdditive: {mask.additive}"
            )
            reference = _group_reference_dataset(subgroup) if subgroup is not None else None
            self._sync_mask_editor(mask, reference)
        elif role == "models" and group is not None:
            self.title_label.setText(f"{group.name} / Models")
            self._set_model_collection_details(group)
        elif role == "fits" and group is not None:
            ensure_fit_history(group)
            self.title_label.setText(f"{group.name} / Fits")
            self._set_fit_history_details(group)
        elif role == "plots" and group is not None:
            self.title_label.setText(f"{group.name} / Plots")
            self._set_details_text(f"Saved plots: {len(group.plots)}")
        elif role == "plot" and group is not None:
            plot = self._plot_for_item(self._current_item())
            if plot is not None:
                source = plot.sources[0].dataset_id if plot.sources else "missing"
                self.title_label.setText(plot.name)
                self._set_details_text(
                    f"Saved plot\n\nType: {plot.type}\nSource dataset ID: {source}\n"
                    "Open it for a clean figure or edit it in the data viewer."
                )
        elif role == "analysis_output" and group is not None:
            output = self._analysis_output_roles.get(id(self._current_item()))
            if output is not None:
                self._set_analysis_output_details(output)
        elif role == "analysis" and group is not None:
            analysis = self._analysis_item_roles.get(id(self._current_item()))
            if analysis is not None:
                self.title_label.setText(analysis.name)
                result = analysis.result
                lines = [
                    f"Operation: {analysis_definition(analysis.type).label}",
                    f"Inputs: {len(analysis.input_dataset_ids)}",
                    f"Status: {result.status if result is not None else 'never run'}",
                ]
                if result is not None:
                    lines.extend(
                        [
                            f"Created: {result.created_at}",
                            f"Outputs: {len(result.outputs)}",
                            *[f"{key}: {_metadata_value_text(value)}" for key, value in sorted(result.diagnostics.items()) if key != "status_bits"],
                        ]
                    )
                self._set_details_text("\n".join(lines))
        elif role in {"fit", "fit_timeline"} and group is not None and fit_entry is not None:
            self.title_label.setText(fit_entry.name)
            if (
                role == "fit"
                and not self._restoring_fit_selection
                and not self._is_multi_select_gesture()
            ):
                self._restore_selected_fit_state(group, fit_entry)
            self._sync_fit_editor(fit_entry)
            self._set_fit_details(fit_entry)
        elif role == "model" and model is not None:
            self.title_label.setText(model.name)
            definition = MODEL_TYPE_REGISTRY.get(model.type)
            label = definition.label if definition is not None else model.type
            self._set_details_text(f"Model\n\nType: {label}\nEnabled: {model.enabled}")
            self._sync_model_editor(model)
        elif role == "fit_model_session" and group is not None:
            self.title_label.setText("Fit model session")
            self._set_details_text("Existing fit session")
        else:
            self.title_label.setText("Project")
            self._set_details_text(f"{len(self.project.data_groups)} workspace(s)")
        if role not in {"mask", "group_mask"}:
            self._clear_mask_parameter_editor()
        if role != "model":
            self._clear_model_parameter_editor()
        if role != "fit":
            self.fit_branch_check.setChecked(False)

    def _sync_selected_state_controls(
        self,
        role: str,
        entry: DatasetEntry | None,
        mask: MaskSpec | None,
        model: ModelComponentSpec | None,
    ) -> None:
        from PySide6 import QtCore

        has_enabled = role in {
            "dataset",
            "dataset_group",
            "masks",
            "mask",
            "backgrounds",
            "group_masks",
            "group_mask",
            "group_backgrounds",
            "model",
        }
        self.enabled_check.setVisible(has_enabled)
        self.fit_weight_widget.setVisible(role == "dataset")
        self.enabled_check.blockSignals(True)
        try:
            collection_states = None
            if role == "masks" and entry is not None:
                collection_states = [bool(item.enabled) for item in entry.masks]
            elif role == "backgrounds" and entry is not None:
                collection_states = [bool(item.enabled) for item in entry.backgrounds]
            elif role == "group_masks":
                node = self._dataset_group_for_item(self._current_item())
                collection_states = [] if node is None else [
                    bool(item.enabled) for item in node.masks
                ]
            elif role == "group_backgrounds":
                owner = self._background_owner_for_item(self._current_item())
                collection_states = [] if owner is None else [
                    bool(item.enabled) for item in owner.backgrounds
                ]
            self.enabled_check.setTristate(collection_states is not None)
            if collection_states is not None:
                if collection_states and all(collection_states):
                    self.enabled_check.setCheckState(QtCore.Qt.CheckState.Checked)
                elif any(collection_states):
                    self.enabled_check.setCheckState(
                        QtCore.Qt.CheckState.PartiallyChecked
                    )
                else:
                    self.enabled_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
            if role == "dataset" and entry is not None:
                self.enabled_check.setChecked(bool(entry.enabled))
            elif role == "dataset_group":
                subgroup = self._dataset_group_for_item(self._current_item())
                self.enabled_check.setChecked(
                    bool(subgroup.enabled) if subgroup is not None else False
                )
            elif role in {"mask", "group_mask"} and mask is not None:
                self.enabled_check.setChecked(bool(mask.enabled))
            elif role == "model" and model is not None:
                self.enabled_check.setChecked(bool(model.enabled))
            elif collection_states is None:
                self.enabled_check.setChecked(False)
        finally:
            self.enabled_check.blockSignals(False)
        self.fit_weight_spin.blockSignals(True)
        try:
            self.fit_weight_spin.setValue(float(entry.fit_weight) if role == "dataset" and entry is not None else 1.0)
        finally:
            self.fit_weight_spin.blockSignals(False)
        self.scale_factor_spin.blockSignals(True)
        try:
            self.scale_factor_spin.setValue(
                float(entry.scale_factor) if role == "dataset" and entry is not None else 1.0
            )
        finally:
            self.scale_factor_spin.blockSignals(False)
        self.scale_factor_fit_check.blockSignals(True)
        try:
            self.scale_factor_fit_check.setChecked(
                bool(entry.scale_factor_vary) if role == "dataset" and entry is not None else False
            )
        finally:
            self.scale_factor_fit_check.blockSignals(False)
        self._sync_sample_environment_controls(entry, role)
        self._sync_group_bulk_controls(role)
        self._sync_background_bulk_controls(role)

    def _background_collection_owner(
        self, role: str
    ) -> DatasetEntry | DataGroup | DatasetGroup | None:
        _group, entry, _mask, _model, _item_role = self._objects_for_item(
            self._current_item()
        )
        if role == "backgrounds":
            return entry
        if role == "group_backgrounds":
            return self._background_owner_for_item(self._current_item())
        return None

    def _sync_background_bulk_controls(self, role: str) -> None:
        show = role in {"backgrounds", "group_backgrounds"}
        self.background_bulk_widget.setVisible(show)
        if not show:
            return
        owner = self._background_collection_owner(role)
        values = set() if owner is None else {
            float(background.scale) for background in owner.backgrounds
        }
        self.background_bulk_scale_edit.blockSignals(True)
        try:
            self.background_bulk_scale_edit.setText(
                _format_number(next(iter(values))) if len(values) == 1 else ""
            )
        finally:
            self.background_bulk_scale_edit.blockSignals(False)

    def _set_background_bulk_scale(self, text: str) -> None:
        group, _entry, _mask, _model, role = self._objects_for_item(
            self._current_item()
        )
        owner = self._background_collection_owner(role)
        text = text.strip()
        if group is None or owner is None or not text:
            return
        try:
            value = float(text)
        except ValueError:
            self._sync_background_bulk_controls(role)
            return
        if not np.isfinite(value):
            self._sync_background_bulk_controls(role)
            return
        if all(background.scale == value for background in owner.backgrounds):
            return
        set_background_collection(owner, scale=value)
        self._background_changed(group, owner)
        self._sync_background_bulk_controls(role)

    def _group_bulk_datasets(self, role: str) -> list[DatasetEntry]:
        item = self._current_item()
        _group, _entry, _mask, _model, _item_role = self._objects_for_item(item)
        if role in {"dataset_group", "group_masks"}:
            subgroup = self._dataset_group_for_item(item)
            if subgroup is not None:
                return list(subgroup.iter_datasets())
        return []

    def _sync_group_bulk_controls(self, role: str) -> None:
        show = role in {"dataset_group", "group_masks"}
        self.group_bulk_widget.setVisible(show)
        if not show:
            return
        datasets = self._group_bulk_datasets(role)
        for edit, attribute in (
            (self.group_fit_weight_edit, "fit_weight"),
            (self.group_scale_edit, "scale_factor"),
        ):
            values = {float(getattr(dataset, attribute)) for dataset in datasets}
            edit.blockSignals(True)
            try:
                edit.setText(_format_number(next(iter(values))) if len(values) == 1 else "")
            finally:
                edit.blockSignals(False)
        subgroup = self._dataset_group_for_item(self._current_item())
        tied = bool(
            subgroup is not None
            and datasets
            and all(
                dataset.scale_factor_vary
                and dataset.scale_factor_group == subgroup.name
                for dataset in datasets
            )
        )
        self.group_scale_fit_check.blockSignals(True)
        try:
            self.group_scale_fit_check.setChecked(tied)
        finally:
            self.group_scale_fit_check.blockSignals(False)

    def _set_group_bulk_value(self, attribute: str, text: str) -> None:
        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        text = text.strip()
        if not text or role not in {"dataset_group", "group_masks"}:
            return
        try:
            value = float(text)
        except ValueError:
            return
        datasets = self._group_bulk_datasets(role)
        changed = False
        for dataset in datasets:
            if float(getattr(dataset, attribute)) != value:
                setattr(dataset, attribute, value)
                changed = True
        if not changed:
            return
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._sync_details()

    def _set_group_shared_scale(self, checked: bool) -> None:
        group, _entry, _mask, _model, role = self._objects_for_item(
            self._current_item()
        )
        if role not in {"dataset_group", "group_masks"}:
            return
        subgroup = self._dataset_group_for_item(self._current_item())
        if subgroup is None:
            return
        datasets = list(subgroup.iter_datasets())
        if not datasets:
            return
        changed = False
        if checked:
            common_scale = float(datasets[0].scale_factor)
            for dataset in datasets:
                if (
                    dataset.scale_factor != common_scale
                    or not dataset.scale_factor_vary
                    or dataset.scale_factor_group != subgroup.name
                ):
                    dataset.scale_factor = common_scale
                    dataset.scale_factor_vary = True
                    dataset.scale_factor_group = subgroup.name
                    changed = True
        else:
            for dataset in datasets:
                if dataset.scale_factor_group == subgroup.name:
                    dataset.scale_factor_group = None
                    dataset.scale_factor_vary = True
                    changed = True
        if not changed:
            return
        branch_created = bool(
            group is not None and self._record_data_group_state_change(group)
        )
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, refresh_viewers=False)
            return
        self._refresh_cache_badges()
        self._sync_details()

    def _set_selected_enabled(self, checked: bool) -> None:
        current = self._current_item()
        group, entry, mask, model, role = self._objects_for_item(current)
        selected = [
            item
            for item in self.tree.selectedItems()
            if self._objects_for_item(item)[4] == role
        ]
        if current is not None and current not in selected:
            selected.append(current)
        if not selected:
            return
        changed = False
        changed_mask_datasets: list[DatasetEntry] = []
        affected_groups: list[DataGroup] = []
        for item in selected:
            item_group, item_entry, item_mask, item_model, item_role = (
                self._objects_for_item(item)
            )
            subgroup = (
                self._dataset_group_for_item(item)
                if item_role in {"dataset_group", "group_masks"}
                else None
            )
            target = None
            targets = []
            if item_role == "dataset":
                target = item_entry
            elif item_role == "dataset_group":
                target = subgroup
            elif item_role in {"mask", "group_mask"}:
                target = item_mask
            elif item_role == "model":
                target = item_model
            if target is not None:
                targets = [target]
            elif item_role == "masks" and item_entry is not None:
                targets = list(item_entry.masks)
            elif item_role == "backgrounds" and item_entry is not None:
                targets = list(item_entry.backgrounds)
            elif item_role == "group_masks" and subgroup is not None:
                targets = list(subgroup.masks)
            elif item_role == "group_backgrounds":
                owner = self._background_owner_for_item(item)
                targets = [] if owner is None else list(owner.backgrounds)
            changed_here = any(
                target.enabled != bool(checked) for target in targets
            )
            if not changed_here:
                continue
            if item_role == "masks" and item_entry is not None:
                set_mask_collection_enabled(item_entry, bool(checked))
            elif item_role == "group_masks" and subgroup is not None:
                set_mask_collection_enabled(subgroup, bool(checked))
            elif item_role == "backgrounds" and item_entry is not None:
                set_background_collection(item_entry, enabled=bool(checked))
            elif item_role == "group_backgrounds":
                owner = self._background_owner_for_item(item)
                if owner is not None:
                    set_background_collection(owner, enabled=bool(checked))
            else:
                for target in targets:
                    target.enabled = bool(checked)
            changed = True
            if item_group is not None and item_group not in affected_groups:
                affected_groups.append(item_group)
            if item_role in {"mask", "group_mask", "masks", "group_masks"}:
                mask_datasets = (
                    [item_entry]
                    if item_role in {"mask", "masks"} and item_entry is not None
                    else list(subgroup.iter_datasets()) if subgroup is not None else []
                )
                known_ids = {id(dataset) for dataset in changed_mask_datasets}
                changed_mask_datasets.extend(
                    dataset for dataset in mask_datasets if id(dataset) not in known_ids
                )
            if item_role == "group_backgrounds":
                owner = self._background_owner_for_item(item)
                if owner is not None and not isinstance(owner, DatasetEntry):
                    data_group_composite_config(
                        _composite_scope(item_group, owner)
                    )["stale"] = True
        if not changed:
            return
        for affected_group in affected_groups:
            self._record_data_group_state_change(affected_group)
            if role in {"dataset", "dataset_group"} and bool(checked):
                self._evaluate_model_after_dataset_activation(affected_group)
        self._mark_mask_datasets_stale(changed_mask_datasets)
        self._mark_dirty()
        # _refresh_tree refreshes open viewers once after the tree state is
        # rebuilt. Avoid doing the same full-volume refresh twice here.
        single_selection = len(selected) == 1
        self._refresh_tree(
            select_group=affected_groups[0] if len(affected_groups) == 1 else group,
            select_dataset=entry if single_selection else None,
            select_mask=mask if single_selection else None,
            select_model=model if single_selection else None,
            select_dataset_group=(
                self._dataset_group_for_item(current)
                if single_selection and role == "dataset_group"
                else None
            ),
        )

    def _set_selected_dataset_fit_weight(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        weight = float(value)
        if entry.fit_weight == weight:
            return
        entry.fit_weight = weight
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        self._refresh_cache_badges()
        self._sync_details()

    def _set_selected_dataset_scale_factor(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        scale = float(value)
        tied = (
            [
                dataset
                for dataset in group.iter_datasets()
                if dataset.scale_factor_group == entry.scale_factor_group
            ]
            if group is not None and entry.scale_factor_group is not None
            else []
        )
        targets = tied or [entry]
        if all(dataset.scale_factor == scale for dataset in targets):
            return
        for dataset in targets:
            dataset.scale_factor = scale
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._sync_details()

    def _set_selected_dataset_scale_factor_vary(self, checked: bool) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        vary = bool(checked)
        if entry.scale_factor_vary == vary and entry.scale_factor_group is None:
            return
        entry.scale_factor_vary = vary
        entry.scale_factor_group = None
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(
                select_group=group,
                select_dataset=entry,
                refresh_viewers=False,
            )
            return
        self._sync_details()

    def _set_selected_dataset_temperature(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        override = None if float(value) < 0.0 else float(value)
        if entry.parameters.get("temperature") == override:
            return
        if override is None:
            entry.parameters.pop("temperature", None)
        else:
            entry.parameters["temperature"] = override
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        self._sync_details()

    def _set_selected_dataset_kf_ki_included(self, included: bool) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        value = bool(included)
        if bool(entry.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)) == value:
            return
        entry.parameters[KINEMATIC_KF_KI_INCLUDED_KEY] = value
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._sync_details()

    def _build_sample_environment_panel(self) -> None:
        """Build the persistent dataset conditions details panel.

        Holds per-dataset temperature, fixed-cut, and applied-field controls.
        It is a persistent widget (like the fit-settings panel) inserted into
        the dataset-details column between the Dataset and Axes panels; it is
        removed but not destroyed on each details refresh.
        """

        from PySide6 import QtWidgets

        self.sample_environment_widget = QtWidgets.QGroupBox("Conditions")
        self.sample_environment_widget.setObjectName("dataset_sample_environment_group")
        layout = QtWidgets.QGridLayout(self.sample_environment_widget)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        layout.addWidget(QtWidgets.QLabel("T (K)"), 0, 0)
        self.dataset_temperature_spin = QtWidgets.QDoubleSpinBox()
        self.dataset_temperature_spin.setObjectName("dataset_temperature")
        self.dataset_temperature_spin.setToolTip(
            "Sample temperature override in kelvin for the selected dataset. "
            "Physics models use it for the Bose factor; set to '(from data)' "
            "(spin to the minimum) to use the temperature imported with the data."
        )
        self.dataset_temperature_spin.setRange(-1.0, 1.0e4)
        self.dataset_temperature_spin.setDecimals(3)
        self.dataset_temperature_spin.setSingleStep(1.0)
        self.dataset_temperature_spin.setSpecialValueText("(from data)")
        self.dataset_temperature_spin.setValue(-1.0)
        self.dataset_temperature_spin.valueChanged.connect(self._set_selected_dataset_temperature)
        layout.addWidget(self.dataset_temperature_spin, 0, 1, 1, 3)

        self.dataset_fixed_q_label = QtWidgets.QLabel("Fixed Q (Å⁻¹)")
        self.dataset_fixed_q_edit = QtWidgets.QLineEdit()
        self.dataset_fixed_q_edit.setObjectName("dataset_fixed_q")
        self.dataset_fixed_q_edit.setToolTip(
            "Fixed momentum transfer for an imported constant-Q powder cut. "
            "Editing this value updates the singleton Q axis used for plotting and fitting."
        )
        self.dataset_fixed_q_edit.editingFinished.connect(
            lambda: self._set_selected_dataset_fixed_condition(
                "constant_q_inv_angstrom", "q_modulus", self.dataset_fixed_q_edit
            )
        )
        layout.addWidget(self.dataset_fixed_q_label, 1, 0)
        layout.addWidget(self.dataset_fixed_q_edit, 1, 1, 1, 3)

        self.dataset_fixed_energy_label = QtWidgets.QLabel("Fixed E (meV)")
        self.dataset_fixed_energy_edit = QtWidgets.QLineEdit()
        self.dataset_fixed_energy_edit.setObjectName("dataset_fixed_energy")
        self.dataset_fixed_energy_edit.setToolTip(
            "Fixed energy transfer for an imported constant-E powder cut. "
            "Editing this value updates the singleton energy axis used for plotting and fitting."
        )
        self.dataset_fixed_energy_edit.editingFinished.connect(
            lambda: self._set_selected_dataset_fixed_condition(
                "constant_energy_meV", "energy_transfer", self.dataset_fixed_energy_edit
            )
        )
        layout.addWidget(self.dataset_fixed_energy_label, 2, 0)
        layout.addWidget(self.dataset_fixed_energy_edit, 2, 1, 1, 3)

        self.dataset_kf_ki_included_check = QtWidgets.QCheckBox("k_f/k_i included")
        self.dataset_kf_ki_included_check.setObjectName("dataset_kf_ki_included")
        self.dataset_kf_ki_included_check.setToolTip(
            "Whether the reduced dataset already includes the neutron kinematic k_f/k_i factor. "
            "Checked is the default. When unchecked, nfit multiplies signal and uncertainty by k_f/k_i "
            "using the dataset's Ei or Ef and energy transfer before plotting and fitting."
        )
        self.dataset_kf_ki_included_check.toggled.connect(
            self._set_selected_dataset_kf_ki_included
        )
        layout.addWidget(self.dataset_kf_ki_included_check, 3, 0, 1, 4)

        layout.addWidget(QtWidgets.QLabel("Field (T)"), 4, 0)
        self.dataset_field_magnitude_spin = QtWidgets.QDoubleSpinBox()
        self.dataset_field_magnitude_spin.setObjectName("dataset_field_magnitude")
        self.dataset_field_magnitude_spin.setToolTip(
            "Applied magnetic field magnitude in tesla for the selected "
            "dataset. Used by the Zeeman term of spin-fluctuation models. Set "
            "to '(none)' (spin to the minimum) for zero field. The direction "
            "below and the data group's lattice orient the field vector."
        )
        self.dataset_field_magnitude_spin.setRange(-1.0, 1.0e3)
        self.dataset_field_magnitude_spin.setDecimals(4)
        self.dataset_field_magnitude_spin.setSingleStep(0.1)
        self.dataset_field_magnitude_spin.setSpecialValueText("(none)")
        self.dataset_field_magnitude_spin.setValue(-1.0)
        self.dataset_field_magnitude_spin.valueChanged.connect(
            self._set_selected_dataset_field_magnitude
        )
        layout.addWidget(self.dataset_field_magnitude_spin, 4, 1)

        self.dataset_field_frame_combo = QtWidgets.QComboBox()
        self.dataset_field_frame_combo.setObjectName("dataset_field_frame")
        self.dataset_field_frame_combo.setToolTip(
            "Frame the field direction is expressed in: direct-lattice "
            "[u v w] (the usual experimental statement, e.g. B ∥ [111]) or "
            "reciprocal (H K L). For cubic crystals the two agree; for lower "
            "symmetry they differ."
        )
        self.dataset_field_frame_combo.addItem("[u v w] direct", "uvw")
        self.dataset_field_frame_combo.addItem("(H K L) reciprocal", "hkl")
        self.dataset_field_frame_combo.currentIndexChanged.connect(
            self._set_selected_dataset_field_frame
        )
        layout.addWidget(self.dataset_field_frame_combo, 4, 2)

        self.dataset_field_direction_edit = QtWidgets.QLineEdit()
        self.dataset_field_direction_edit.setObjectName("dataset_field_direction")
        self.dataset_field_direction_edit.setToolTip(
            "Field direction as three components, e.g. '1 1 1'. Interpreted in "
            "the frame selected at left; only the orientation matters (the "
            "magnitude above sets the strength)."
        )
        self.dataset_field_direction_edit.setPlaceholderText("1 1 1")
        self.dataset_field_direction_edit.editingFinished.connect(
            self._set_selected_dataset_field_direction
        )
        layout.addWidget(self.dataset_field_direction_edit, 4, 3)

        self.sample_environment_widget.setParent(None)

    def _selected_field_payload(self, entry: DatasetEntry) -> dict[str, Any]:
        payload = entry.parameters.get("magnetic_field")
        if isinstance(payload, dict):
            return dict(payload)
        return {"magnitude_T": 0.0, "direction": [1.0, 1.0, 1.0], "frame": "uvw"}

    def _set_selected_dataset_fixed_condition(
        self,
        parameter_name: str,
        axis_role: str,
        editor: Any,
    ) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        try:
            value = float(editor.text())
        except ValueError:
            self._sync_details()
            return
        if not np.isfinite(value) or entry.parameters.get(parameter_name) == value:
            return
        entry.parameters[parameter_name] = value
        if isinstance(entry.data, MDHistoData):
            axes = list(entry.data.axes)
            for index, axis in enumerate(axes):
                if axis.role == axis_role and axis.centers.size == 1:
                    shift = value - float(axis.centers[0])
                    axes[index] = replace(
                        axis,
                        values=np.asarray(axis.values, dtype=float) + shift,
                    )
                    entry.replace_data(
                        entry.data.with_updates(axes=tuple(axes)),
                        source_backed=False,
                    )
                    break
        import_options = entry.metadata.get("import_options")
        if isinstance(import_options, dict):
            import_options["fixed_value"] = value
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._sync_details()

    def _commit_field_payload(
        self, group: DataGroup | None, entry: DatasetEntry, payload: dict[str, Any]
    ) -> None:
        magnitude = float(payload.get("magnitude_T", 0.0) or 0.0)
        if magnitude <= 0.0:
            if "magnetic_field" not in entry.parameters:
                return
            entry.parameters.pop("magnetic_field", None)
        else:
            if entry.parameters.get("magnetic_field") == payload:
                return
            entry.parameters["magnetic_field"] = payload
        branch_created = False
        if group is not None:
            branch_created = self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_dataset=entry)
            return
        self._sync_details()

    def _set_selected_dataset_field_magnitude(self, value: float) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        payload = self._selected_field_payload(entry)
        payload["magnitude_T"] = 0.0 if float(value) < 0.0 else float(value)
        self._commit_field_payload(group, entry, payload)

    def _set_selected_dataset_field_frame(self, _index: int) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        payload = self._selected_field_payload(entry)
        payload["frame"] = str(self.dataset_field_frame_combo.currentData() or "uvw")
        self._commit_field_payload(group, entry, payload)

    def _set_selected_dataset_field_direction(self) -> None:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "dataset" or entry is None:
            return
        text = self.dataset_field_direction_edit.text().replace(",", " ")
        try:
            components = [float(part) for part in text.split()]
        except ValueError:
            components = []
        if len(components) != 3 or all(c == 0.0 for c in components):
            self._sync_details()  # revert the edit to the stored value
            return
        payload = self._selected_field_payload(entry)
        payload["direction"] = components
        self._commit_field_payload(group, entry, payload)

    def _sync_sample_environment_controls(
        self, entry: DatasetEntry | None, role: str
    ) -> None:
        is_dataset = role == "dataset" and entry is not None
        override = entry.parameters.get("temperature") if is_dataset else None
        self.dataset_temperature_spin.blockSignals(True)
        try:
            self.dataset_temperature_spin.setValue(
                float(override) if override not in (None, "") else -1.0
            )
        finally:
            self.dataset_temperature_spin.blockSignals(False)

        fixed_q = entry.parameters.get("constant_q_inv_angstrom") if is_dataset else None
        fixed_energy = entry.parameters.get("constant_energy_meV") if is_dataset else None
        for label, editor, value in (
            (self.dataset_fixed_q_label, self.dataset_fixed_q_edit, fixed_q),
            (
                self.dataset_fixed_energy_label,
                self.dataset_fixed_energy_edit,
                fixed_energy,
            ),
        ):
            visible = value not in (None, "")
            label.setVisible(visible)
            editor.setVisible(visible)
            editor.blockSignals(True)
            try:
                editor.setText("" if not visible else _format_number(float(value)))
            finally:
                editor.blockSignals(False)

        self.dataset_kf_ki_included_check.blockSignals(True)
        try:
            self.dataset_kf_ki_included_check.setChecked(
                bool(entry.parameters.get(KINEMATIC_KF_KI_INCLUDED_KEY, True)) if is_dataset else True
            )
            self.dataset_kf_ki_included_check.setVisible(
                not (
                    is_dataset
                    and entry.data_type
                    in {"single_crystal_inelastic", "powder_inelastic"}
                    and isinstance(
                        entry.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY), dict
                    )
                )
            )
        finally:
            self.dataset_kf_ki_included_check.blockSignals(False)

        payload = entry.parameters.get("magnetic_field") if is_dataset else None
        if not isinstance(payload, dict):
            payload = {}
        magnitude = float(payload.get("magnitude_T", 0.0) or 0.0)
        direction = payload.get("direction", [1.0, 1.0, 1.0])
        frame = str(payload.get("frame", "uvw"))
        self.dataset_field_magnitude_spin.blockSignals(True)
        try:
            self.dataset_field_magnitude_spin.setValue(magnitude if magnitude > 0.0 else -1.0)
        finally:
            self.dataset_field_magnitude_spin.blockSignals(False)
        self.dataset_field_frame_combo.blockSignals(True)
        try:
            index = self.dataset_field_frame_combo.findData(frame)
            self.dataset_field_frame_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.dataset_field_frame_combo.blockSignals(False)
        self.dataset_field_direction_edit.blockSignals(True)
        try:
            self.dataset_field_direction_edit.setText(
                " ".join(_format_number(float(x)) for x in direction)
            )
        finally:
            self.dataset_field_direction_edit.blockSignals(False)

    def _set_details_text(self, text: str) -> None:
        from PySide6 import QtCore, QtWidgets

        self.details_label.setText(text)
        self._clear_details_panel()
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_layout.addWidget(label)
        self.details_layout.addStretch(1)

    def _set_model_collection_details(self, group: DataGroup) -> None:
        """Build the workspace-level editor for hard parameter constraints."""

        from PySide6 import QtCore, QtWidgets

        self.details_label.setText(f"{len(group.models)} model(s)")
        self._clear_details_panel()
        parameters = [
            qualified_parameter_name(model.name, parameter)
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
            for parameter in model_parameter_names(model)
        ]
        constraint_group = QtWidgets.QGroupBox("Fit constraints")
        constraint_group.setObjectName("model_constraints_group")
        layout = QtWidgets.QVBoxLayout(constraint_group)
        table = QtWidgets.QTableWidget()
        table.setObjectName("model_constraints_table")
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Dependent parameter", "Relation", "Expression", ""])
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.setToolTip(
            "Hard relationships between globally shared fit parameters. Exact relationships "
            "remove the dependent parameter from the optimizer; inequalities constrain it "
            "through a nonnegative offset."
        )
        rows = [
            (model, dict(constraint))
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
            for constraint in model.constraints
        ]
        table.setRowCount(len(rows))
        for row, (owner, constraint) in enumerate(rows):
            self._populate_model_constraint_row(
                group, table, row, parameters, owner, constraint
            )
        layout.addWidget(table)

        examples = QtWidgets.QLabel(
            "Examples: A.x = `B.y`; A.x = 10 - `B.y` (fixed sum); "
            "A.x = 10 / `B.y` (fixed product); A.x >= `B.y`."
        )
        examples.setWordWrap(True)
        examples.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        examples.setToolTip(
            "Use backticks around a parameter reference. Exact expressions support +, -, *, /, **, "
            "abs, sqrt, exp, log, log10, sin, cos, and tan."
        )
        layout.addWidget(examples)

        controls = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add constraint")
        add_button.setObjectName("add_model_constraint_button")
        add_button.setToolTip(
            "Add a hard relationship. The dependent parameter must be globally shared and enabled for fitting."
        )
        check_button = QtWidgets.QPushButton("Check constraints")
        check_button.setObjectName("check_model_constraints_button")
        check_button.setToolTip("Validate parameter references, expressions, and dependency cycles without running a fit.")
        status = QtWidgets.QLabel()
        status.setObjectName("model_constraints_status")
        status.setWordWrap(True)
        controls.addWidget(add_button)
        controls.addWidget(check_button)
        controls.addWidget(status, 1)
        layout.addLayout(controls)
        add_button.setEnabled(bool(parameters))
        add_button.clicked.connect(
            lambda: self._add_model_constraint_row(group, table, parameters)
        )
        check_button.clicked.connect(lambda: self._check_model_constraints(group, status))
        self.details_layout.addWidget(
            self._details_group_box("Models", [f"Components: {len(group.models)}"])
        )
        self.details_layout.addWidget(constraint_group)
        self.details_layout.addStretch(1)

    def _populate_model_constraint_row(
        self,
        group: DataGroup,
        table: Any,
        row: int,
        parameters: list[str],
        owner: ModelComponentSpec,
        constraint: dict[str, Any],
    ) -> None:
        from PySide6 import QtCore, QtWidgets

        target = qualified_parameter_name(owner.name, str(constraint.get("parameter", "")))
        target_combo = QtWidgets.QComboBox()
        target_combo.addItems(parameters)
        target_combo.setCurrentIndex(max(target_combo.findText(target), 0))
        target_combo.setToolTip("Parameter determined or bounded by this relationship.")
        relation_combo = QtWidgets.QComboBox()
        for label, value in (("=", "="), (">=", ">="), ("<=", "<=")):
            relation_combo.addItem(label, value)
        relation_combo.setCurrentIndex(max(relation_combo.findData(str(constraint.get("op", "="))), 0))
        relation_combo.setToolTip(
            "Exact equality removes one independent fit parameter. An inequality preserves the independent-parameter count."
        )
        if relation_combo.currentData() == "=":
            expression_text = str(constraint.get("expression", constraint.get("reference", "0")))
        else:
            reference = constraint.get("reference", 0.0)
            expression_text = f"`{reference}`" if isinstance(reference, str) else _format_number(float(reference))
        expression = QtWidgets.QLineEdit(expression_text)
        expression.setPlaceholderText("constant or expression")
        expression.setToolTip(
            "Right-hand side. Use backticks around qualified parameters, for example `Model1.scale`. "
            "Start typing a backtick to choose a parameter from completion suggestions. "
            "Inequalities currently accept one parameter or one numeric constant."
        )
        completer = QtWidgets.QCompleter([f"`{name}`" for name in parameters], expression)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        expression.setCompleter(completer)
        remove = QtWidgets.QToolButton()
        remove.setText("Remove")
        remove.setToolTip("Delete this fit constraint.")
        table.setCellWidget(row, 0, target_combo)
        table.setCellWidget(row, 1, relation_combo)
        table.setCellWidget(row, 2, expression)
        table.setCellWidget(row, 3, remove)
        target_combo.currentIndexChanged.connect(lambda: self._store_model_constraint_table(group, table))
        relation_combo.currentIndexChanged.connect(lambda: self._store_model_constraint_table(group, table))
        expression.editingFinished.connect(lambda: self._store_model_constraint_table(group, table))
        remove.clicked.connect(lambda: self._remove_model_constraint_row(group, table, remove))

    def _add_model_constraint_row(self, group: DataGroup, table: Any, parameters: list[str]) -> None:
        if not parameters:
            return
        row = table.rowCount()
        table.insertRow(row)
        owner, parameter = _model_parameter_from_qualified_name(group, parameters[0])
        self._populate_model_constraint_row(
            group,
            table,
            row,
            parameters,
            owner,
            {"parameter": parameter, "op": "=", "expression": "0"},
        )
        self._store_model_constraint_table(group, table)

    def _remove_model_constraint_row(self, group: DataGroup, table: Any, button: Any) -> None:
        for row in range(table.rowCount()):
            if table.cellWidget(row, 3) is button:
                table.removeRow(row)
                self._store_model_constraint_table(group, table)
                return

    def _store_model_constraint_table(self, group: DataGroup, table: Any) -> None:
        constraints: dict[str, list[dict[str, Any]]] = {name: [] for name in group.models}
        for row in range(table.rowCount()):
            target = table.cellWidget(row, 0).currentText()
            relation = str(table.cellWidget(row, 1).currentData())
            expression = table.cellWidget(row, 2).text().strip()
            owner, parameter = _model_parameter_from_qualified_name(group, target)
            entry: dict[str, Any] = {"parameter": parameter, "op": relation}
            if relation == "=":
                entry["expression"] = expression
            else:
                reference = expression
                if len(reference) >= 2 and reference.startswith("`") and reference.endswith("`"):
                    reference = reference[1:-1].strip()
                try:
                    entry["reference"] = float(reference)
                except ValueError:
                    entry["reference"] = reference
            constraints[owner.name].append(entry)
        changed = False
        for name, model in group.models.items():
            if not isinstance(model, ModelComponentSpec):
                continue
            updated = constraints.get(name, [])
            if model.constraints != updated:
                model.constraints = updated
                changed = True
        if changed:
            self._record_data_group_state_change(group)
            self._mark_dirty()
            self._request_overlay_refresh(group)

    def _check_model_constraints(self, group: DataGroup, status: Any) -> None:
        try:
            _validate_model_constraints(group)
        except (TypeError, ValueError) as exc:
            status.setStyleSheet("color: #c0392b; font-weight: 600;")
            status.setText(str(exc))
            return
        count = sum(
            len(model.constraints)
            for model in group.models.values()
            if isinstance(model, ModelComponentSpec)
        )
        status.setStyleSheet("color: #1f9d68; font-weight: 600;")
        status.setText(f"{count} constraint(s) valid")

    def _set_fit_history_details(self, group: DataGroup) -> None:
        """Show fit-history summary and its destructive compaction action."""

        from PySide6 import QtWidgets

        result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
        self.details_label.setText(f"Fit history\n\nResults: {result_count}")
        self._clear_details_panel()
        self.details_layout.addWidget(
            self._details_group_box("Fit history", [f"Results: {result_count}"])
        )
        self.clear_fit_history_button = QtWidgets.QPushButton("Clear history")
        self.clear_fit_history_button.setObjectName("clear_fit_history_button")
        self.clear_fit_history_button.setToolTip(
            "Delete every saved fit result and make the currently selected fit result or "
            "Current state the new Initial state."
        )
        self.clear_fit_history_button.setEnabled(bool(group.fits))
        self.clear_fit_history_button.clicked.connect(lambda: self.clear_fit_history(group))
        self.details_layout.addWidget(self.clear_fit_history_button)
        self.details_layout.addStretch(1)

    def clear_fit_history(self, group: DataGroup | None = None) -> bool:
        """Replace a group's fit tree with the selected state as its Initial state."""

        from PySide6 import QtWidgets

        if group is None:
            selected_group, _entry, _mask, _model, role = self._objects_for_item(
                self._current_item()
            )
            if role != "fits":
                return False
            group = selected_group
        if group is None:
            return False
        ensure_fit_history(group)
        selected = self._active_fit_entry(group)
        if selected is None or selected.kind not in {"initial", "result", "current"}:
            results = [entry for entry in _walk_fit_entries(group.fits) if entry.kind == "result"]
            selected = results[-1] if results else _top_level_current_state_entry(group)
            if selected is None:
                selected = group.fits[0]
        results = [entry for entry in _walk_fit_entries(group.fits) if entry.kind == "result"]
        latest_result = results[-1] if results else None
        if latest_result is not None and selected is not latest_result:
            answer = QtWidgets.QMessageBox.question(
                self.window,
                "Clear fit history",
                f"{selected.name} is not the most recent fit result ({latest_result.name}). "
                "Clear all saved fit results and use the selected state as the new Initial state?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        snapshot = (
            copy.deepcopy(selected.snapshot)
            if selected.snapshot
            else snapshot_data_group_state(group)
        )
        self._restoring_fit_selection = True
        try:
            restore_data_group_state(group, snapshot)
        finally:
            self._restoring_fit_selection = False
        initial = FitTimelineEntry(
            name="Initial",
            kind="initial",
            snapshot=copy.deepcopy(snapshot),
            created_at=_timestamp_now(),
            optimizer=str(selected.optimizer or "least_squares"),
            optimizer_config=copy.deepcopy(selected.optimizer_config),
        )
        group.fits[:] = [initial]
        self._set_active_fit_state(group, initial)
        self._mark_dirty()
        self._refresh_tree(select_group=group, select_fit=initial)
        return True

    def _set_background_details(
        self,
        group: DataGroup | None,
        owner: DatasetEntry | DataGroup | DatasetGroup,
        background: BackgroundSpec,
    ) -> None:
        from PySide6 import QtWidgets

        self._clear_details_panel()
        box = QtWidgets.QGroupBox("Background subtraction")
        box.setToolTip(
            "A |Q|-energy source is interpolated onto the target grid. A single-crystal "
            "histogram must have identical axes and bins. Unrebinned neutron point references "
            "on a group background follow the group's grid automatically. The scaled signal is subtracted "
            "and the scaled variance is added."
        )
        layout = QtWidgets.QGridLayout(box)
        enabled = QtWidgets.QCheckBox("Enabled")
        enabled.setChecked(background.enabled)
        enabled.setToolTip("Enable or temporarily bypass this background subtraction.")
        enabled.toggled.connect(
            lambda checked: self._update_background(
                group, owner, background, enabled=bool(checked)
            )
        )
        layout.addWidget(enabled, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Source"), 1, 0)
        source_combo = QtWidgets.QComboBox()
        source_combo.setObjectName("background_source_dataset")
        source_combo.setToolTip(
            "Powder |Q|-energy data, an identically binned histogram, or a live group composite. "
            "For group backgrounds, unrebinned neutron point data automatically use the sample grid."
        )
        candidates = [] if group is None else [
            candidate
            for candidate in group.iter_datasets()
            if candidate is not owner
            and candidate.data_type in {"powder_inelastic", "single_crystal_inelastic"}
        ]
        for candidate in candidates:
            source_combo.addItem(candidate.name, candidate.id)
        if group is not None:
            for candidate in group.iter_subgroups():
                if candidate is owner:
                    continue
                if data_group_composite_enabled(_composite_scope(group, candidate)):
                    source_combo.addItem(
                        f"{candidate.name} [live composite]",
                        f"group:{candidate.id}",
                    )
        selected_source = (
            f"group:{background.source_group_id}"
            if background.source_group_id
            else background.source_dataset_id
        )
        source_index = source_combo.findData(selected_source)
        if source_index >= 0:
            source_combo.setCurrentIndex(source_index)
        source_combo.currentIndexChanged.connect(
            lambda _index: self._set_background_source(
                group, owner, background, source_combo.currentData()
            )
        )
        layout.addWidget(source_combo, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Scale"), 2, 0)
        scale = QtWidgets.QDoubleSpinBox()
        scale.setObjectName("background_scale")
        scale.setRange(-1.0e12, 1.0e12)
        scale.setDecimals(8)
        scale.setValue(float(background.scale))
        scale.setToolTip(
            "Multiplier applied to this background before subtraction. Its absolute value also scales the uncertainty."
        )
        scale.valueChanged.connect(
            lambda value: self._update_background(
                group, owner, background, scale=float(value)
            )
        )
        layout.addWidget(scale, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Projection"), 3, 0)
        projection = QtWidgets.QComboBox()
        projection.setObjectName("background_projection")
        projection.addItem("Voxel center (legacy)", "center")
        supports_trajectory_projection = (
            isinstance(owner, DatasetGroup) and "mdevent" in owner.metadata
        )
        if supports_trajectory_projection or background.projection == "sample_trajectories":
            projection.addItem("Sample detector trajectories", "sample_trajectories")
        projection.setToolTip(
            "Voxel center evaluates B(|Q|, E) once at each target-bin center. "
            "Sample detector trajectories forward-projects a powder background through "
            "every MDEvent sample angle and the same detector-trajectory normalization; "
            "it is available for backgrounds owned by MDEvent dataset groups."
        )
        projection.setCurrentIndex(max(projection.findData(background.projection), 0))
        projection.currentIndexChanged.connect(
            lambda _index: self._update_background(
                group,
                owner,
                background,
                projection=str(projection.currentData()),
            )
        )
        layout.addWidget(projection, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Interpolation"), 4, 0)
        interpolation = QtWidgets.QComboBox()
        interpolation.addItem("Linear", "linear")
        interpolation.addItem("Nearest", "nearest")
        interpolation.setToolTip(
            "Interpolation used for powder data. Identically binned single-crystal subtraction does not interpolate."
        )
        interpolation.setCurrentIndex(
            max(interpolation.findData(background.interpolation), 0)
        )
        interpolation.currentIndexChanged.connect(
            lambda _index: self._update_background(
                group,
                owner,
                background,
                interpolation=str(interpolation.currentData()),
            )
        )
        layout.addWidget(interpolation, 4, 1)
        self.details_layout.addWidget(box)
        self.details_layout.addStretch(1)

    def _set_background_source(
        self,
        group: DataGroup | None,
        owner: DatasetEntry | DataGroup | DatasetGroup,
        background: BackgroundSpec,
        source_id: Any,
    ) -> None:
        if group is None or source_id is None:
            return
        if str(source_id).startswith("group:"):
            group_id = str(source_id).split(":", 1)[1]
            source_group = next(
                (candidate for candidate in group.iter_subgroups() if candidate.id == group_id),
                None,
            )
            if source_group is None or source_group is owner:
                return
            background.source_dataset_id = ""
            background.source_group_id = source_group.id
            background.source_entry = None
            background.source_group = source_group
            self._background_changed(group, owner)
            return
        source = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        if source is None or source is owner:
            return
        background.source_dataset_id = source.id
        background.source_group_id = None
        background.source_entry = source
        background.source_group = None
        self._background_changed(group, owner)

    def _update_background(
        self,
        group: DataGroup | None,
        owner: DatasetEntry | DataGroup | DatasetGroup,
        background: BackgroundSpec,
        **changes: Any,
    ) -> None:
        changed = False
        for name, value in changes.items():
            if getattr(background, name) != value:
                setattr(background, name, value)
                changed = True
        if changed:
            self._background_changed(group, owner)

    def _background_changed(
        self,
        group: DataGroup | None,
        owner: DatasetEntry | DataGroup | DatasetGroup,
    ) -> None:
        if group is not None:
            if not isinstance(owner, DatasetEntry):
                data_group_composite_config(_composite_scope(group, owner))["stale"] = True
            self._record_data_group_state_change(group)
            self.refresh_slice_viewer(group, force_rebin=True)
        self._refresh_cache_badges()
        self._mark_dirty()

    def _set_group_details(self, group: DataGroup) -> None:
        result_count = sum(1 for fit in _walk_fit_entries(group.fits) if fit.kind == "result")
        datasets = list(group.iter_datasets())
        text = (
            f"Workspace\n\nDatasets: {len(datasets)}\nModels: {len(group.models)}\n"
            f"Fit results: {result_count}"
        )
        self.details_label.setText(text)
        self._clear_details_panel()
        self.details_layout.addWidget(
            self._details_group_box(
                "Workspace",
                [
                    f"Datasets: {len(datasets)}",
                    f"Data points: {_format_number(_group_data_point_count(group))}",
                    f"Dataset types: {_group_dataset_type_summary(group)}",
                    f"Models: {len(group.models)}",
                    f"Fit results: {result_count}",
                ],
            )
        )
        self.details_layout.addWidget(self._dataset_importing_group_box(group))
        self.details_layout.addWidget(self._group_crystal_symmetry_group_box(group))
        if any(dataset.data_type.startswith("single_crystal") for dataset in group.iter_datasets()):
            self.details_layout.addWidget(self._ub_setup_group_box(group, group))
        self.details_layout.addStretch(1)

    def _group_crystal_symmetry_group_box(self, group: DataGroup) -> Any:
        """Return the workspace-wide lattice-symmetry controls."""

        from PySide6 import QtWidgets

        box = QtWidgets.QGroupBox("Crystal symmetry")
        layout = QtWidgets.QGridLayout(box)
        layout.setColumnStretch(1, 1)
        label = QtWidgets.QLabel("Space group")
        tooltip = (
            "Workspace Hermann-Mauguin space group used by Bragg peak generation and "
            "Brillouin-zone analyses. Enter an IT number (for example 227) or a Gemmi "
            "symbol such as 'F d -3 m:2'. CIF imports normalize legacy suffixes such as 'Z'."
        )
        label.setToolTip(tooltip)
        editor = QtWidgets.QLineEdit(str(group.spacegroup or "P 1"))
        editor.setObjectName("group_spacegroup_editor")
        editor.setToolTip(tooltip)
        editor.editingFinished.connect(
            lambda editor=editor: self._set_group_spacegroup(group, editor.text())
        )
        layout.addWidget(label, 0, 0)
        layout.addWidget(editor, 0, 1)
        return box

    def _set_group_spacegroup(self, group: DataGroup, text: str) -> None:
        """Store a user-supplied workspace symmetry expression."""

        value = text.strip() or "P 1"
        if group.spacegroup == value:
            return
        group.spacegroup = value
        crystal = group.metadata.get("crystal")
        if isinstance(crystal, dict):
            crystal["spacegroup"] = value
        self._mark_dirty()
        self._sync_details()

    def _dataset_importing_group_box(self, group: DataGroup) -> Any:
        from PySide6 import QtWidgets

        config = self._dataset_importing_config(group)
        box = QtWidgets.QGroupBox("Dataset importing")
        box.setToolTip(
            "Add single-crystal dataset files directly to this data group. Use the range fields for numbered files, "
            "or Add files to choose an arbitrary set in the file browser."
        )
        layout = QtWidgets.QVBoxLayout(box)
        enabled = QtWidgets.QCheckBox("Import datasets from files")
        enabled.setObjectName("dataset_importing_enabled")
        enabled.setChecked(bool(config.get("enabled", False)))
        enabled.setToolTip(
            "Enable file-based dataset importing for this data group. The setting is saved with the project."
        )
        enabled.toggled.connect(lambda checked: self._set_dataset_importing_enabled(group, checked))
        layout.addWidget(enabled)

        controls = QtWidgets.QWidget()
        controls.setObjectName("dataset_importing_controls")
        controls.setEnabled(bool(config.get("enabled", False)))
        grid = QtWidgets.QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        add_files = QtWidgets.QPushButton("Add files")
        add_files.setObjectName("dataset_importing_add_files")
        add_files.setToolTip("Choose one or more dataset files to add as single-crystal dataset entries.")
        add_files.clicked.connect(lambda: self._add_dataset_import_files(group))
        grid.addWidget(add_files, 0, 0, 1, 2)

        fields = (
            ("Path", "path", "Directory containing numbered dataset files."),
            ("Prefix", "prefix", "Text before each run number, for example SEQ_."),
            ("Suffix", "suffix", "Text after each run number, for example .nxs.h5."),
            ("Numors", "numors", "Run numbers and inclusive ranges: 409981:409995, 409981:3:409995, or comma-separated ranges."),
        )
        for row, (label, key, tooltip) in enumerate(fields, start=1):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit(str(config.get(key, "")))
            edit.setObjectName(f"dataset_importing_{key}")
            edit.setToolTip(tooltip)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: self._set_dataset_importing_text(group, key, edit.text())
            )
            grid.addWidget(edit, row, 1)

        import_button = QtWidgets.QPushButton("Import datasets")
        import_button.setObjectName("dataset_importing_import_range")
        import_button.setToolTip("Build paths from Path, Prefix, Suffix, and Numors, then add each existing file as a dataset.")
        import_button.clicked.connect(lambda: self._import_dataset_importing_range(group))
        grid.addWidget(import_button, len(fields) + 1, 0, 1, 2)

        clear_button = QtWidgets.QPushButton("Clear datasets")
        clear_button.setObjectName("dataset_importing_clear")
        clear_button.setToolTip("Remove every direct and nested dataset from this data group after confirmation. Models, masks, and fit history are kept.")
        clear_button.clicked.connect(lambda: self._clear_imported_datasets(group))
        grid.addWidget(clear_button, len(fields) + 2, 0, 1, 2)
        layout.addWidget(controls)
        return box

    def _set_dataset_collection_details(
        self,
        root: DataGroup,
        node: DataGroup | DatasetGroup,
    ) -> None:
        datasets = list(node.iter_datasets())
        if isinstance(node, DatasetGroup):
            reference = next(
                (dataset for dataset in datasets if dataset.data is not None),
                None,
            )
            if reference is not None:
                _adopt_imported_crystal(root, reference, node)
        text = (
            f"Dataset collection\n\nDatasets (incl. nested): {len(datasets)}\n"
            f"Data points: {_format_number(_dataset_collection_point_count(node))}\n"
            f"Dataset types: {_group_dataset_type_summary(node)}"
        )
        if isinstance(node, DatasetGroup):
            text += f"\nShared masks: {len(node.masks)}"
        self.details_label.setText(text)
        self._clear_details_panel()
        summary_lines = [
            f"Datasets (incl. nested): {len(datasets)}",
            f"Data points: {_format_number(_dataset_collection_point_count(node))}",
            f"Dataset types: {_group_dataset_type_summary(node)}",
        ]
        if isinstance(node, DatasetGroup):
            summary_lines.append(f"Shared masks: {len(node.masks)}")
        self.details_layout.addWidget(
            self._details_group_box(
                "Dataset collection",
                summary_lines,
            )
        )
        if isinstance(node, DatasetGroup) and isinstance(node.metadata.get("mdevent"), dict):
            self.details_layout.addWidget(self._mdevent_group_box(node))
        if isinstance(node, DatasetGroup) and isinstance(node.metadata.get("raw_dgs"), dict):
            self.details_layout.addWidget(self._raw_dgs_group_box(node))
        if any(dataset.data_type.startswith("single_crystal") for dataset in node.iter_datasets()):
            self.details_layout.addWidget(self._ub_setup_group_box(root, node))
        scope = _composite_scope(root, node)
        self.details_layout.addWidget(self._group_composite_group_box(scope))
        self.details_layout.addWidget(self._group_dataset_weights_group_box(scope))
        self.details_layout.addStretch(1)

    def _mdevent_group_box(self, node: DatasetGroup) -> Any:
        from PySide6 import QtWidgets

        config = node.metadata["mdevent"]
        box = QtWidgets.QGroupBox("MDEvent shared setup")
        box.setToolTip(
            "Shared configuration for every run in this MDEvent dataset group. "
            "Orientation, detector mask, vanadium normalization, and overrides are stored once."
        )
        layout = QtWidgets.QGridLayout(box)
        fields = [
            ("Normalization", "normalization_file", "Vanadium detector workspace read directly by nfit and used for detector efficiency and solid-angle normalization. Mantid is not required."),
            ("Detector mask", "mask_file", "Detector workspace read directly by nfit; zero, negative, or invalid detector values are excluded from both events and normalization coverage."),
        ]
        for row, (label, key, tooltip) in enumerate(fields):
            layout.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit(str(config.get(key) or ""))
            edit.setObjectName(f"mdevent_{key}")
            edit.setToolTip(tooltip)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: self._set_mdevent_group_value(node, key, edit.text().strip() or None)
            )
            layout.addWidget(edit, row, 1, 1, 3)

        layout.addWidget(QtWidgets.QLabel("Ei override"), 2, 0)
        ei = QtWidgets.QDoubleSpinBox()
        ei.setObjectName("mdevent_incident_energy_override")
        ei.setRange(-1.0, 1.0e5)
        ei.setDecimals(6)
        ei.setSpecialValueText("(from each run)")
        ei.setValue(float(config.get("incident_energy_override") or -1.0))
        ei.setToolTip(
            "Shared incident-energy override in meV. The run Ei is used at the minimum. "
            "For an already converted MDEvent file this changes normalization trajectories, not stored event coordinates."
        )
        ei.valueChanged.connect(
            lambda value: self._set_mdevent_group_value(node, "incident_energy_override", None if value < 0.0 else float(value))
        )
        layout.addWidget(ei, 2, 1)

        layout.addWidget(QtWidgets.QLabel("T0 override"), 2, 2)
        t0 = QtWidgets.QDoubleSpinBox()
        t0.setObjectName("mdevent_t0_override")
        t0.setRange(-1.0, 1.0e6)
        t0.setDecimals(6)
        t0.setSpecialValueText("(from each run)")
        t0.setValue(float(config.get("t0_override") if config.get("t0_override") is not None else -1.0))
        t0.setToolTip(
            "Shared time-zero override in microseconds. It is retained for importing raw individual runs. "
            "It cannot move coordinates already converted and stored in an MDEvent workspace."
        )
        t0.valueChanged.connect(
            lambda value: self._set_mdevent_group_value(node, "t0_override", None if value < 0.0 else float(value))
        )
        layout.addWidget(t0, 2, 3)

        layout.addWidget(QtWidgets.QLabel("UB matrix"), 3, 0)
        ub = QtWidgets.QLineEdit(_parameter_to_text(config.get("ub_matrix", np.eye(3).tolist())))
        ub.setObjectName("mdevent_ub_matrix")
        ub.setToolTip(
            "Shared Mantid UB matrix. Event Q_sample is converted with (2*pi*UB)^-1; "
            "normalization trajectories additionally include each run goniometer."
        )
        ub.editingFinished.connect(lambda: self._set_mdevent_group_ub(node, ub))
        layout.addWidget(ub, 3, 1, 1, 3)
        return box

    def _raw_dgs_group_box(self, node: DatasetGroup) -> Any:
        """Shared controls for streamed native raw TOF reduction."""
        from PySide6 import QtWidgets

        config = node.metadata["raw_dgs"]
        if config.get("format") == "corelli-correlation-nexus":
            return self._corelli_group_box(node)
        box = QtWidgets.QGroupBox("Raw TOF shared setup")
        box.setToolTip("Shared configuration for all raw direct-geometry runs. nfit reads the detector geometry from each NeXus file and streams events directly into the HKLE composite.")
        layout = QtWidgets.QGridLayout(box)
        for row, (label, key, tooltip) in enumerate((
            ("Vanadium normalization", "normalization_file", "Optional processed vanadium workspace. Zero or negative spectra exclude those detectors; positive values weight detector trajectories in the MDNorm-style normalization denominator."),
            ("Detector mask", "mask_file", "Optional detector workspace. Zero, negative, or invalid values exclude detector events before TOF-to-HKLE conversion and trajectory normalization."),
        )):
            layout.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit(str(config.get(key) or ""))
            edit.setObjectName(f"raw_dgs_{key}")
            edit.setToolTip(tooltip)
            edit.editingFinished.connect(lambda edit=edit, key=key: self._set_raw_dgs_group_value(node, key, edit.text().strip() or None))
            layout.addWidget(edit, row, 1, 1, 2)
            browse = QtWidgets.QPushButton("Browse…")
            browse.setObjectName(f"raw_dgs_{key}_browse")
            browse.setToolTip(f"Choose the {label.lower()} NeXus file.")
            browse.clicked.connect(
                lambda _checked=False, edit=edit, key=key, label=label: self._browse_raw_dgs_setup_file(
                    node, key, label, edit
                )
            )
            layout.addWidget(browse, row, 3)
        for column, (label, key, tooltip) in enumerate((
            ("Ei override", "incident_energy_override", "Shared incident-energy override in meV. Leave unset to apply Mantid's local GetEi path for each run: monitor fitting where applicable, or the instrument's requested-Ei formula."),
            ("T0 override", "t0_override", "Shared time-zero correction in microseconds, subtracted from raw event TOF before calculating final energy. Leave unset to apply Mantid's local GetEi path per run, including formula-derived T0 on instruments that define one."),
        )):
            layout.addWidget(QtWidgets.QLabel(label), 2, column * 2)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1.0, 1e6)
            spin.setDecimals(6)
            spin.setSpecialValueText("(from each run)")
            spin.setValue(
                float(config.get(key) if config.get(key) is not None else -1.0)
            )
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(lambda value, key=key: self._set_raw_dgs_group_value(node, key, None if value < 0.0 else float(value)))
            layout.addWidget(spin, 2, column * 2 + 1)
        for column, (label, key, default, maximum, tooltip) in enumerate((
            ("Emin / Ei", "energy_min_fraction", -0.95, 0.999999, "Minimum accepted energy transfer divided by the incident energy. The default is -0.95. This same limit is used for event selection and detector-trajectory normalization."),
            ("Emax / Ei", "energy_max_fraction", 0.95, 0.999999, "Maximum accepted energy transfer divided by the incident energy. It must remain below 1 so the final neutron energy is positive. The default is 0.95. This same limit is used for event selection and detector-trajectory normalization."),
        )):
            layout.addWidget(QtWidgets.QLabel(label), 3, column * 2)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName(f"raw_dgs_{key}")
            spin.setRange(-100.0, maximum)
            spin.setDecimals(6)
            spin.setValue(float(config.get(key, default)))
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(
                lambda value, key=key: self._set_raw_dgs_group_value(
                    node, key, float(value)
                )
            )
            layout.addWidget(spin, 3, column * 2 + 1)
        correction = QtWidgets.QCheckBox("Apply ki/kf correction")
        correction.setChecked(bool(config.get("ki_kf_normalization", config.get("kf_ki_normalization", True))))
        correction.setToolTip("Multiply each accepted event by ki/kf, the incident-to-final wavevector ratio used by Mantid direct-geometry reduction. Event variances receive the square of this factor; the choice is recorded in rebinned metadata.")
        correction.toggled.connect(lambda checked: self._set_raw_dgs_group_value(node, "ki_kf_normalization", bool(checked)))
        layout.addWidget(correction, 4, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("UB matrix"), 5, 0)
        ub = QtWidgets.QLineEdit(_parameter_to_text(config.get("ub_matrix", np.eye(3).tolist())))
        ub.setToolTip("Shared Mantid/SNS-frame UB matrix. Raw Q is rotated into the sample frame and converted with (2*pi*UB)^-1 before binning.")
        ub.editingFinished.connect(lambda: self._set_raw_dgs_group_ub(node, ub))
        layout.addWidget(ub, 5, 1, 1, 3)
        return box

    def _corelli_group_box(self, node: DatasetGroup) -> Any:
        """Shared controls for native CORELLI correlation reconstruction."""
        from PySide6 import QtWidgets

        config = node.metadata["raw_dgs"]
        box = QtWidgets.QGroupBox("CORELLI finite-energy reconstruction")
        box.setToolTip(
            "Reconstruct signed finite-energy CORELLI intensity from raw event TOF, "
            "pulse time, and correlation-chopper phase."
        )
        layout = QtWidgets.QGridLayout(box)
        file_rows = (
            (
                "Solid-angle workspace",
                "normalization_file",
                "Optional CORELLI solid-angle/vanadium MatrixWorkspace. Positive detector "
                "values correct event weights; zero or negative values mask detectors.",
            ),
            (
                "Incident-flux workspace",
                "flux_file",
                "Optional CORELLI cumulative flux MatrixWorkspace. nfit differentiates each "
                "bank spectrum and corrects reconstructed events at their incident wavevector.",
            ),
            (
                "Detector mask",
                "mask_file",
                "Optional Mantid detector-mask XML or MatrixWorkspace. Listed XML detector "
                "IDs, or zero, negative, or invalid workspace values, exclude events.",
            ),
        )
        for row, (label, key, tooltip) in enumerate(file_rows):
            field_label = QtWidgets.QLabel(label)
            field_label.setToolTip(tooltip)
            layout.addWidget(field_label, row, 0)
            edit = QtWidgets.QLineEdit(str(config.get(key) or ""))
            edit.setObjectName(f"raw_dgs_{key}")
            edit.setToolTip(tooltip)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: self._set_raw_dgs_group_value(
                    node, key, edit.text().strip() or None
                )
            )
            layout.addWidget(edit, row, 1, 1, 2)
            browse = QtWidgets.QPushButton("Browse…")
            browse.setObjectName(f"raw_dgs_{key}_browse")
            browse.setToolTip(f"Choose the {label.lower()} NeXus file.")
            browse.clicked.connect(
                lambda _checked=False, edit=edit, key=key, label=label: self._browse_raw_dgs_setup_file(
                    node, key, label, edit
                )
            )
            layout.addWidget(browse, row, 3)

        numeric_fields = (
            (
                "Timing offset (ns)",
                "timing_offset_ns",
                14_000.0,
                0.0,
                1.0e7,
                0,
                "Correlation-chopper TDC timing offset in nanoseconds. Use the value calibrated "
                "for the experiment cycle; 14,000 ns is the 2026A autoreduction value.",
            ),
            (
                "Minimum wavelength (Å)",
                "wavelength_min_angstrom",
                0.6,
                0.01,
                100.0,
                5,
                "Shortest reconstructed incident wavelength in angstrom.",
            ),
            (
                "Maximum wavelength (Å)",
                "wavelength_max_angstrom",
                2.5,
                0.01,
                100.0,
                5,
                "Longest reconstructed incident wavelength in angstrom.",
            ),
        )
        for column, (label, key, default, minimum, maximum, decimals, tooltip) in enumerate(
            numeric_fields
        ):
            field_label = QtWidgets.QLabel(label)
            field_label.setToolTip(tooltip)
            layout.addWidget(field_label, 3, column)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName(f"raw_dgs_{key}")
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setValue(float(config.get(key, default)))
            spin.setToolTip(tooltip)
            spin.valueChanged.connect(
                lambda value, key=key: self._set_raw_dgs_group_value(
                    node, key, float(value)
                )
            )
            layout.addWidget(spin, 4, column)

        corrections = QtWidgets.QWidget()
        corrections_layout = QtWidgets.QVBoxLayout(corrections)
        corrections_layout.setContentsMargins(0, 0, 0, 0)
        ki_kf = QtWidgets.QCheckBox("Apply ki/kf correction")
        ki_kf.setObjectName("raw_dgs_ki_kf_normalization")
        ki_kf.setChecked(bool(config.get("ki_kf_normalization", True)))
        ki_kf.setToolTip(
            "Multiply each reconstructed hypothesis by the incident-to-final wavevector "
            "ratio ki/kf and apply the square of that factor to its variance."
        )
        ki_kf.toggled.connect(
            lambda checked: self._set_raw_dgs_group_value(
                node, "ki_kf_normalization", bool(checked)
            )
        )
        corrections_layout.addWidget(ki_kf)
        he3 = QtWidgets.QCheckBox("Apply He-3 detector efficiency")
        he3.setObjectName("raw_dgs_he3_detector_efficiency_correction")
        he3.setChecked(
            bool(config.get("he3_detector_efficiency_correction", True))
        )
        he3.setToolTip(
            "Correct each reconstructed hypothesis for the wavelength-dependent "
            "He-3 tube efficiency from the embedded instrument definition."
        )
        he3.toggled.connect(
            lambda checked: self._set_raw_dgs_group_value(
                node, "he3_detector_efficiency_correction", bool(checked)
            )
        )
        corrections_layout.addWidget(he3)
        layout.addWidget(corrections, 4, 3)
        ub_label = QtWidgets.QLabel("UB matrix")
        ub_label.setToolTip(
            "Shared Mantid/SNS-frame UB matrix used to convert reconstructed sample-frame Q to HKL. ISAW files are converted when loaded."
        )
        layout.addWidget(ub_label, 5, 0)
        ub = QtWidgets.QLineEdit(
            _parameter_to_text(config.get("ub_matrix", np.eye(3).tolist()))
        )
        ub.setObjectName("raw_dgs_ub_matrix")
        ub.setToolTip(ub_label.toolTip())
        ub.editingFinished.connect(lambda: self._set_raw_dgs_group_ub(node, ub))
        layout.addWidget(ub, 5, 1, 1, 3)
        caveat = QtWidgets.QLabel(
            "Energy channels share measured events and therefore have correlated statistical "
            "errors. Stored errors contain the diagonal variance."
        )
        caveat.setWordWrap(True)
        caveat.setToolTip(
            "CORELLI cross correlation reconstructs every requested energy channel from the "
            "same phase-tagged detector events."
        )
        layout.addWidget(caveat, 6, 0, 1, 4)
        return box

    def _ub_setup_group_box(
        self,
        root: DataGroup,
        target: DataGroup | DatasetGroup | DatasetEntry,
    ) -> Any:
        from PySide6 import QtCore, QtWidgets

        box = QtWidgets.QGroupBox("Crystal orientation")
        box.setToolTip(
            "Edit the unit cell, orientation vectors, and UB matrix for this single-crystal scope. "
            "UB maps column [h,k,l] to Q' in inverse angstrom with beam +x and vertical +z."
        )
        if isinstance(target, DatasetEntry):
            orientation_metadata = _merged_dataset_metadata(target)
        else:
            orientation_metadata = dict(target.metadata)
            if isinstance(target, DatasetGroup):
                for key in ("mdevent", "raw_dgs"):
                    shared = target.metadata.get(key)
                    if isinstance(shared, dict):
                        orientation_metadata = {
                            **shared,
                            **orientation_metadata,
                        }
        setup = orientation_metadata.get("ub_setup")
        ub = _dataset_ub_for_editor(orientation_metadata)
        if ub is None and isinstance(setup, dict):
            ub = _dataset_ub_for_editor(setup)

        layout = QtWidgets.QVBoxLayout(box)
        spacegroup_row = QtWidgets.QHBoxLayout()
        spacegroup_row.addWidget(QtWidgets.QLabel("Space group / centering"))
        spacegroup_edit = QtWidgets.QLineEdit(str(root.spacegroup or ""))
        spacegroup_edit.setObjectName("crystal_orientation_spacegroup")
        spacegroup_edit.setPlaceholderText("P, I, F, C, or full symbol")
        spacegroup_edit.setToolTip(
            "Set the Hermann–Mauguin space group or lattice-centering letter. "
            "The data viewer uses its centering to draw optional Brillouin-zone boundaries."
        )
        spacegroup_edit.editingFinished.connect(
            lambda: self._set_group_spacegroup(root, spacegroup_edit.text())
        )
        spacegroup_row.addWidget(spacegroup_edit, 1)
        layout.addLayout(spacegroup_row)
        if ub is None:
            matrix_text = "UB matrix: not set"
        else:
            matrix_text = "UB matrix (Å⁻¹):\n" + "\n".join(_matrix_lines(ub))
        matrix_display = QtWidgets.QLabel(matrix_text)
        matrix_display.setObjectName("ub_matrix_display")
        matrix_display.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        matrix_display.setToolTip(
            "Current UB matrix without the 2π factor. It maps column [h, k, l] "
            "to the sample-frame scattering vector Q′ in inverse angstrom."
        )
        layout.addWidget(matrix_display)
        button_row = QtWidgets.QHBoxLayout()
        button = QtWidgets.QPushButton("UB setup...")
        button.setObjectName("open_ub_setup")
        button.setToolTip(
            "Open the UB editor. It can calculate UB from lattice and u/v vectors, load NeXus or ISAW .mat files, "
            "and convert between the internal Mantid/SNS frame and the transposed IPNS/ISAW convention on disk."
        )
        button.clicked.connect(lambda: self._open_ub_setup(root, target))
        button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return box

    def _open_ub_setup(
        self,
        root: DataGroup,
        target: DataGroup | DatasetGroup | DatasetEntry,
    ) -> bool:
        from PySide6 import QtWidgets

        if isinstance(target, DatasetGroup):
            reference = next(iter(target.iter_datasets()), None)
            if reference is not None:
                _ensure_dataset_data_loaded(reference)
                _adopt_imported_crystal(root, reference, target)
        metadata = target.metadata if isinstance(target.metadata, dict) else {}
        setup = metadata.get("ub_setup") if isinstance(metadata.get("ub_setup"), dict) else {}
        if isinstance(target, DatasetGroup) and isinstance(metadata.get("mdevent"), dict):
            shared = metadata["mdevent"]
            ub = shared.get("ub_matrix", np.eye(3))
            lattice = shared.get("lattice_parameters") or root.lattice_parameters
        elif isinstance(target, DatasetGroup) and isinstance(metadata.get("raw_dgs"), dict):
            shared = metadata["raw_dgs"]
            ub = shared.get("ub_matrix", np.eye(3))
            lattice = shared.get("lattice_parameters") or root.lattice_parameters
        elif isinstance(target, DatasetEntry):
            merged = _merged_dataset_metadata(target)
            imported = _dataset_ub_for_editor(merged)
            ub = setup.get("ub_matrix", imported if imported is not None else np.eye(3))
            lattice = setup.get("lattice_parameters") or merged.get("lattice_parameters") or root.lattice_parameters
        elif isinstance(target, DatasetGroup):
            imported = _dataset_ub_for_editor(metadata)
            ub = setup.get("ub_matrix", imported if imported is not None else np.eye(3))
            lattice = (
                setup.get("lattice_parameters")
                or metadata.get("lattice_parameters")
                or root.lattice_parameters
            )
        else:
            ub = setup.get("ub_matrix", metadata.get("ub_matrix", np.eye(3)))
            lattice = setup.get("lattice_parameters") or root.lattice_parameters
        dialog = UBSetupDialog(
            self.window,
            ub=ub,
            lattice=lattice,
            u=setup.get("u", [1.0, 0.0, 0.0]),
            v=setup.get("v", [0.0, 1.0, 0.0]),
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.result is None:
            return False
        result = dialog.result
        target.metadata["ub_setup"] = copy.deepcopy(result)
        target.metadata["ub_matrix"] = copy.deepcopy(result["ub_matrix"])
        if isinstance(target, DatasetEntry):
            target.metadata["lattice_parameters"] = copy.deepcopy(result["lattice_parameters"])
        else:
            root.lattice_parameters = copy.deepcopy(result["lattice_parameters"])
        if isinstance(target, DatasetGroup) and isinstance(target.metadata.get("mdevent"), dict):
            target.metadata["mdevent"]["ub_matrix"] = copy.deepcopy(result["ub_matrix"])
            target.metadata["mdevent"]["lattice_parameters"] = copy.deepcopy(result["lattice_parameters"])
            data_group_composite_config(_composite_scope(root, target))["stale"] = True
        if isinstance(target, DatasetGroup) and isinstance(target.metadata.get("raw_dgs"), dict):
            target.metadata["raw_dgs"]["ub_matrix"] = copy.deepcopy(result["ub_matrix"])
            target.metadata["raw_dgs"]["lattice_parameters"] = copy.deepcopy(result["lattice_parameters"])
            data_group_composite_config(_composite_scope(root, target))["stale"] = True
        self._record_data_group_state_change(root)
        self._mark_dirty()
        self._refresh_cache_badges()
        self._sync_details()
        return True

    def _set_mdevent_group_ub(self, node: DatasetGroup, edit: Any) -> None:
        value = _parse_parameter_text(edit.text())
        try:
            matrix = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            matrix = np.empty((0, 0))
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-14:
            edit.setText(_parameter_to_text(node.metadata["mdevent"].get("ub_matrix")))
            return
        self._set_mdevent_group_value(node, "ub_matrix", matrix.tolist())

    def _set_mdevent_group_value(self, node: DatasetGroup, key: str, value: Any) -> None:
        config = node.metadata["mdevent"]
        if config.get(key) == value:
            return
        config[key] = value
        composite = data_group_composite_config(_composite_scope(self._objects_for_item(self._current_item())[0], node))
        composite["stale"] = True
        self._mark_dirty()
        self._refresh_cache_badges()

    def _set_raw_dgs_group_ub(self, node: DatasetGroup, edit: Any) -> None:
        try:
            matrix = np.asarray(_parse_parameter_text(edit.text()), dtype=float)
        except (TypeError, ValueError):
            matrix = np.empty((0, 0))
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-14:
            edit.setText(_parameter_to_text(node.metadata["raw_dgs"].get("ub_matrix")))
            return
        self._set_raw_dgs_group_value(node, "ub_matrix", matrix.tolist())

    def _browse_raw_dgs_setup_file(
        self, node: DatasetGroup, key: str, label: str, edit: Any
    ) -> None:
        current = str(node.metadata["raw_dgs"].get(key) or "")
        file_filter = "Mantid NeXus workspace (*.nxs *.nx5 *.h5 *.hdf5);;All files (*)"
        if (
            key == "mask_file"
            and node.metadata["raw_dgs"].get("format")
            == "corelli-correlation-nexus"
        ):
            file_filter = (
                "Mantid detector mask (*.xml *.nxs *.nx5 *.h5 *.hdf5);;"
                "All files (*)"
            )
        path, _selected_filter = get_open_file_name(
            self.window,
            f"Choose {label.lower()}",
            current,
            file_filter,
        )
        if not path:
            return
        edit.setText(path)
        self._set_raw_dgs_group_value(node, key, path)

    def _set_raw_dgs_group_value(self, node: DatasetGroup, key: str, value: Any) -> None:
        config = node.metadata["raw_dgs"]
        if config.get(key) == value:
            return
        config[key] = value
        composite = data_group_composite_config(_composite_scope(self._objects_for_item(self._current_item())[0], node))
        composite["stale"] = True
        self._mark_dirty()
        self._refresh_cache_badges()

    def _group_dataset_weights_group_box(self, group: DataGroup | _CompositeScope) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._group_dataset_weights_group_box,
            self,
            group,
            helper_namespace=globals(),
        )

    def _group_composite_group_box(self, group: DataGroup | _CompositeScope) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._group_composite_group_box,
            self,
            group,
            helper_namespace=globals(),
        )

    def _set_dataset_details(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        # Selection must remain a metadata-only operation. File-backed point
        # lists are loaded by the explicit Load action or a viewer/fit job.
        from PySide6 import QtCore, QtWidgets

        sections = dataset_detail_sections(dataset, group=group)
        summary_lines: list[str] = []
        for title, lines in sections:
            if summary_lines:
                summary_lines.append("")
            summary_lines.extend([title, *lines])
        self.details_label.setText("\n".join(summary_lines))
        existing_tabs = self.details_widget.findChild(
            QtWidgets.QTabWidget, "dataset_details_tabs"
        )
        selected_tab = (
            existing_tabs.tabText(existing_tabs.currentIndex())
            if existing_tabs is not None and existing_tabs.currentIndex() >= 0
            else "Overview"
        )
        self._clear_details_panel()

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("dataset_details_tabs")
        tabs.setDocumentMode(True)
        tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        tabs.setMinimumHeight(480)

        def add_tab(title: str, object_name: str) -> Any:
            scroll = QtWidgets.QScrollArea()
            scroll.setObjectName(object_name)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            content = QtWidgets.QWidget()
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(4, 6, 4, 6)
            content_layout.setSpacing(8)
            content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(content)
            tabs.addTab(scroll, title)
            return content_layout

        overview_layout = add_tab("Overview", "dataset_details_overview")
        physics_layout = add_tab("Physics", "dataset_details_physics")
        processing_layout = add_tab("Binning & channels", "dataset_details_processing")
        metadata_layout = add_tab("Metadata", "dataset_details_metadata")
        for button in tabs.tabBar().findChildren(QtWidgets.QToolButton):
            button.setToolTip("Scroll the dataset detail tabs when space is limited.")

        section_lines = {title: lines for title, lines in sections}
        if group is not None and isinstance(
            dataset.metadata.get(DERIVED_RECIPE_KEY), dict
        ):
            processing_layout.addWidget(
                self._derived_recipe_group_box(dataset, group)
            )

        overview_layout.addWidget(
            self._dataset_type_group_box(
                dataset, group, section_lines.get("Dataset", [])
            )
        )
        if dataset.data_type != "magnetization":
            overview_layout.addWidget(self.sample_environment_widget)
        overview_layout.addWidget(
            self._details_group_box("Data", section_lines.get("Data", []))
        )
        overview_layout.addWidget(
            self._details_group_box("Source", section_lines.get("Source", []))
        )

        crystal_row = QtWidgets.QHBoxLayout()
        if group is not None and dataset.data_type.startswith("single_crystal"):
            crystal_row.addWidget(self._ub_setup_group_box(group, dataset), 1)
        crystal_row.addWidget(
            self._details_group_box("Crystal", section_lines.get("Crystal", [])), 1
        )
        physics_layout.addLayout(crystal_row)
        if dataset.data_type in {"single_crystal_inelastic", "powder_inelastic"}:
            physics_layout.addWidget(
                self._dataset_spectral_channels_group_box(dataset, group)
            )
        physics_layout.addWidget(
            self._dataset_signal_semantics_group_box(dataset, group)
        )

        is_point_list = isinstance(dataset.data, PointListData)
        processing_layout.addWidget(
            self._dataset_axes_group_box(
                dataset, group, section_lines.get("Axes", [])
            )
        )
        if is_point_list:
            processing_layout.addWidget(
                self._dataset_point_list_group_box(dataset, group)
            )
        metadata_box = self._dataset_metadata_group_box(dataset)
        for tree in metadata_box.findChildren(QtWidgets.QTreeWidget):
            tree.setMaximumHeight(16777215)
        metadata_layout.setAlignment(QtCore.Qt.AlignmentFlag(0))
        metadata_layout.addWidget(metadata_box, 1)

        for content_layout in (
            overview_layout,
            physics_layout,
            processing_layout,
        ):
            content_layout.addStretch(1)
        selected_index = next(
            (
                index
                for index in range(tabs.count())
                if tabs.tabText(index) == selected_tab
            ),
            0,
        )
        tabs.setCurrentIndex(selected_index)
        self.details_layout.addWidget(tabs, 1)

    def _derived_recipe_source_choices(
        self,
        dataset: DatasetEntry,
        group: DataGroup,
    ) -> list[tuple[str, str]]:
        """Return source choices that cannot directly contain the output."""

        from .analysis.runner import (
            COMPOSITE_ANALYSIS_SOURCE_PREFIX,
            analysis_source_choices,
        )

        choices: list[tuple[str, str]] = []
        for label, source_id in analysis_source_choices(group):
            source_id = str(source_id)
            if source_id == dataset.id:
                continue
            if not source_id.startswith(COMPOSITE_ANALYSIS_SOURCE_PREFIX):
                source = next(
                    (
                        candidate
                        for candidate in group.iter_datasets()
                        if candidate.id == source_id
                    ),
                    None,
                )
                if source is None or isinstance(
                    source.metadata.get(DERIVED_RECIPE_KEY), dict
                ):
                    continue
            else:
                suffix = source_id[len(COMPOSITE_ANALYSIS_SOURCE_PREFIX) :]
                node = group if suffix == "root" else next(
                    (
                        candidate
                        for candidate in group.iter_subgroups()
                        if candidate.id == suffix
                    ),
                    None,
                )
                if node is None or any(
                    candidate is dataset for candidate in node.iter_datasets()
                ):
                    continue
            choices.append((label, source_id))
        return choices

    def _derived_recipe_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup,
    ) -> Any:
        from PySide6 import QtWidgets

        box = QtWidgets.QGroupBox("Derived recipe")
        box.setObjectName("derived_recipe_controls")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(10, 8, 10, 8)
        resolved = _derived_analysis_for_dataset(dataset)
        if resolved is None:
            label = QtWidgets.QLabel("The linked analysis recipe is missing.")
            label.setWordWrap(True)
            form.addRow(label)
            return box
        _owner, analysis = resolved
        choices = self._derived_recipe_source_choices(dataset, group)
        source_tooltip = (
            "Choose the original dataset or live composite. The derived output "
            "grid is applied to its underlying event or point data; cached source "
            "histogram bins are not reused."
        )
        source_labels = (
            ("Source",)
            if analysis.type == "dataset_clone"
            else ("Left source", "Right source")
        )
        for source_index, label_text in enumerate(source_labels):
            combo = QtWidgets.QComboBox()
            combo.setObjectName(f"derived_recipe_source_{source_index}")
            combo.setToolTip(source_tooltip)
            for label, source_id in choices:
                combo.addItem(label, source_id)
            current_id = (
                analysis.input_dataset_ids[source_index]
                if source_index < len(analysis.input_dataset_ids)
                else ""
            )
            current_index = combo.findData(current_id)
            if current_index < 0 and current_id:
                combo.addItem(f"Missing source [{current_id}]", current_id)
                current_index = combo.count() - 1
            combo.setCurrentIndex(max(current_index, 0))
            combo.currentIndexChanged.connect(
                lambda _index, selector=combo, position=source_index: self._set_derived_recipe_source(
                    dataset,
                    group,
                    position,
                    str(selector.currentData()),
                )
            )
            form.addRow(label_text, combo)

        if analysis.type == "histogram_arithmetic":
            operation = QtWidgets.QComboBox()
            operation.setObjectName("derived_recipe_operation")
            operation.addItem("Subtract", "subtract")
            operation.addItem("Add", "add")
            operation.setCurrentIndex(
                max(operation.findData(analysis.parameters.get("operation", "subtract")), 0)
            )
            operation.setToolTip(
                "Choose whether the independently reduced right source is subtracted "
                "from or added to the left source."
            )
            operation.currentIndexChanged.connect(
                lambda _index, selector=operation: self._set_derived_recipe_parameter(
                    dataset,
                    group,
                    "operation",
                    str(selector.currentData()),
                )
            )
            form.addRow("Operation", operation)

            scale = QtWidgets.QDoubleSpinBox()
            scale.setObjectName("derived_recipe_right_scale")
            scale.setRange(-1.0e12, 1.0e12)
            scale.setDecimals(8)
            scale.setValue(float(analysis.parameters.get("right_scale", 1.0)))
            scale.setToolTip(
                "Multiplier applied to the right source after both sources have "
                "been reduced from underlying data onto this derived dataset's grid."
            )
            scale.editingFinished.connect(
                lambda editor=scale: self._set_derived_recipe_parameter(
                    dataset,
                    group,
                    "right_scale",
                    float(editor.value()),
                )
            )
            form.addRow("Right scale", scale)

        stage = QtWidgets.QLabel(
            "Source stage: underlying data (the output binning and symmetry below "
            "are applied before the derived operation)."
        )
        stage.setObjectName("derived_recipe_source_stage")
        stage.setWordWrap(True)
        stage.setToolTip(
            "This recipe is virtual and source-linked. Viewing, fitting, or explicit "
            "rebinning recomputes it from the original source data."
        )
        form.addRow(stage)
        return box

    def _set_derived_recipe_source(
        self,
        dataset: DatasetEntry,
        group: DataGroup,
        index: int,
        source_id: str,
    ) -> None:
        resolved = _derived_analysis_for_dataset(dataset)
        if resolved is None or not source_id:
            return
        _owner, analysis = resolved
        if not 0 <= index < len(analysis.input_dataset_ids):
            return
        if analysis.input_dataset_ids[index] == source_id:
            return
        analysis.input_dataset_ids[index] = source_id
        self._after_derived_recipe_changed(dataset, group, analysis)

    def _set_derived_recipe_parameter(
        self,
        dataset: DatasetEntry,
        group: DataGroup,
        key: str,
        value: Any,
    ) -> None:
        resolved = _derived_analysis_for_dataset(dataset)
        if resolved is None:
            return
        _owner, analysis = resolved
        if analysis.parameters.get(key) == value:
            return
        analysis.parameters[key] = value
        self._after_derived_recipe_changed(dataset, group, analysis)

    def _after_derived_recipe_changed(
        self,
        dataset: DatasetEntry,
        group: DataGroup,
        analysis: AnalysisEntry,
    ) -> None:
        config = copy.deepcopy(dataset_rebin_config(dataset))
        create_derived_analysis_dataset(
            group,
            analysis,
            rebin_config=config,
        )
        config = dataset_rebin_config(dataset)
        config["stale"] = True
        self._record_data_group_state_change(group)
        self._mark_dirty()
        if bool(config.get("auto_rebin", True)):
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()

    def _set_dataset_details_preserving_scroll(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        *,
        scroll_position: int | None = None,
        focus_object_name: str | None = None,
    ) -> None:
        """Rebuild dataset details without jumping the right panel to the top."""

        from PySide6 import QtCore, QtWidgets

        scrollbar = self.details_scroll.verticalScrollBar()
        position = int(scrollbar.value()) if scroll_position is None else int(scroll_position)
        if focus_object_name is None:
            focused = self.window.focusWidget()
            if focused is not None and self.details_widget.isAncestorOf(focused):
                focus_object_name = str(focused.objectName() or "")
        self._set_dataset_details(dataset, group)

        def restore_position() -> None:
            _group, current, _mask, _model, role = self._objects_for_item(self._current_item())
            if role == "dataset" and current is dataset:
                self.details_layout.activate()
                if focus_object_name:
                    replacement = self.details_widget.findChild(QtWidgets.QWidget, focus_object_name)
                    if replacement is not None and replacement.isEnabled():
                        replacement.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                scrollbar.setValue(min(position, scrollbar.maximum()))

        restore_position()
        QtCore.QTimer.singleShot(0, restore_position)
        QtCore.QTimer.singleShot(0, lambda: QtCore.QTimer.singleShot(0, restore_position))

    def _set_fit_details(self, fit_entry: FitTimelineEntry) -> None:
        from PySide6 import QtCore, QtWidgets

        group, _entry, _mask, _model, _role = self._objects_for_item(self._current_item())
        self._set_fit_posterior_result_actions(
            group if fit_entry.kind == "result" else None,
            fit_entry if fit_entry.kind == "result" else None,
        )
        self.details_label.setText(fit_details_text(fit_entry))
        self._clear_details_panel()
        duration = (
            f"{_format_number(fit_entry.duration_seconds)} s"
            if fit_entry.duration_seconds is not None
            else "-"
        )
        fit_summary_lines = [
            f"Type: {fit_entry.kind}",
            f"Created: {fit_entry.created_at or '-'}",
            f"Optimizer: {fit_entry.optimizer or '-'}",
            f"Duration: {duration}",
            *_fit_chi_squared_summary_lines(fit_entry),
            f"Timeline entries: {len(fit_entry.children)}",
        ]
        if self.fit_settings_panel is not None:
            self.details_layout.addWidget(self.fit_settings_panel)
        self.details_layout.addWidget(
            self._details_group_box(
                "Fit",
                fit_summary_lines,
            )
        )
        parameter_group = None
        lower_widgets = []
        if _fit_results_rows(fit_entry):
            parameter_group = self._fit_results_group_box(fit_entry)
        elif _snapshot_parameter_rows(fit_entry):
            parameter_group = self._parameter_values_group_box(fit_entry)
        if fit_entry.optimizer_config:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Optimizer config",
                    dict(fit_entry.optimizer_config),
                    object_name="fit_optimizer_config_tree",
                    empty_text="No optimizer configuration.",
                )
            )
        if fit_entry.goodness:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Goodness of fit",
                    dict(fit_entry.goodness),
                    object_name="fit_goodness_tree",
                    empty_text="No goodness-of-fit values.",
                )
            )
        diagnostics_group = self._fit_diagnostics_group_box(fit_entry)
        if diagnostics_group is not None:
            lower_widgets.append(diagnostics_group)
        if fit_entry.channels:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Stored fit channels",
                    dict(fit_entry.channels),
                    object_name="fit_channels_tree",
                    empty_text="No stored fit channels.",
                )
            )
        if fit_entry.metadata:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Metadata",
                    dict(fit_entry.metadata),
                    object_name="fit_metadata_tree",
                    empty_text="No fit metadata.",
                )
            )
        if fit_entry.snapshot:
            lower_widgets.append(
                self._metadata_tree_group_box(
                    "Snapshot",
                    dict(fit_entry.snapshot),
                    object_name="fit_snapshot_tree",
                    empty_text="No fit snapshot.",
                )
            )
        if parameter_group is not None and lower_widgets:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            splitter.setObjectName("fit_details_parameter_splitter")
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(parameter_group)
            lower_panel = QtWidgets.QWidget()
            lower_layout = QtWidgets.QVBoxLayout(lower_panel)
            lower_layout.setContentsMargins(0, 0, 0, 0)
            lower_layout.setSpacing(8)
            for widget in lower_widgets:
                lower_layout.addWidget(widget)
            lower_layout.addStretch(1)
            splitter.addWidget(lower_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([260, 420])
            self.details_layout.addWidget(splitter)
        else:
            if parameter_group is not None:
                self.details_layout.addWidget(parameter_group)
            for widget in lower_widgets:
                self.details_layout.addWidget(widget)
        self.details_layout.addStretch(1)

    def _set_fit_posterior_result_actions(
        self,
        group: DataGroup | None,
        fit_entry: FitTimelineEntry | None,
    ) -> None:
        """Show result-only posterior actions beneath the shared settings."""
        if self.fit_posterior_layout is None:
            return
        if self.fit_posterior_result_actions is not None:
            self.fit_posterior_layout.removeWidget(self.fit_posterior_result_actions)
            self.fit_posterior_result_actions.setParent(None)
            self.fit_posterior_result_actions.deleteLater()
            self.fit_posterior_result_actions = None
        if group is None or fit_entry is None or not _fit_results_rows(fit_entry):
            return
        self.fit_posterior_result_actions = self._posterior_sampler_group_box(group, fit_entry)
        self.fit_posterior_layout.addWidget(self.fit_posterior_result_actions, 7, 0, 1, 2)

    def _fit_results_group_box(self, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtCore, QtGui, QtWidgets

        rows = _fit_results_rows(fit_entry)
        use_posterior_errors = bool(
            _posterior_display_options(fit_entry).get("use_posterior_uncertainties")
        )
        group_box = QtWidgets.QGroupBox("Fit results")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        table = QtWidgets.QTableWidget(len(rows), 6)
        table.setObjectName("fit_results_table")
        table.setToolTip(
            "Human-readable best-fit parameters. The posterior display controls can replace least-squares "
            "standard errors with asymmetric emcee 68% intervals and replace displayed best fits with the "
            "highest-probability emcee sample."
        )
        table.setHorizontalHeaderLabels(
            [
                "Parameter", "Best fit", "68% error" if use_posterior_errors else "Std err",
                "Posterior median", "16%", "84%",
            ]
        )
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(120)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(("name", "value", "uncertainty", "median", "p16", "p84")):
                item = QtWidgets.QTableWidgetItem(row.get(key, "-"))
                item.setToolTip(row.get(key, "-"))
                if row.get("at_limit"):
                    warning = f"Reached the {row.get('limit_side', 'configured')} bound ({row.get('limit_bound', '-')})."
                    item.setForeground(QtGui.QColor("#ff9c94"))
                    item.setBackground(QtGui.QColor("#5a2929"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setToolTip(f"{item.toolTip()}\n{warning}")
                if column_index > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row_index, column_index, item)
        _tooltip_table_corner_buttons(table, "Select all fit-result rows.")
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return group_box

    # Diagnostic metrics shown per dataset, in display order: (metadata key,
    # column header, tooltip).
    _DIAGNOSTIC_COLUMNS = (
        ("temperature", "T (K)", "Dataset temperature."),
        ("mu_eff_sq", "mu_eff^2", "Effective fluctuating moment per site: the "
         "Brillouin-zone and energy integral of chi'' (with the closure's "
         "energy cutoff), in model units."),
        ("chi_static_q0", "chi(0)", "Uniform static susceptibility at Q=0 "
         "(Kramers-Kronig of the modes); the zero-field bulk susceptibility."),
        ("chi_static_qpeak", "chi_peak", "Peak static susceptibility over the "
         "BZ grid; locates the incipient ordering vector."),
        ("chi0_gamma0", "chi0*gamma0", "Product of the local susceptibility and "
         "relaxation rate (tracks the local spectral weight)."),
        ("stability_margin", "D_min", "Smallest RPA denominator on the "
         "Brillouin-zone grid: positive is stable, zero is the ordering "
         "boundary, and negative is unstable."),
        ("stability_ratio", "r_max", "Largest dimensionless RPA feedback "
         "(lambda_max - lambda_shift) * chi0_eff; D_min = 1 - r_max."),
        ("minimum_relative_singular_value", "RPA s_min", "Smallest relative "
         "singular value of the evaluated electronic-RPA denominator. Values "
         "near zero indicate a sampled pole."),
        ("lambda_shift", "lambda_shift", "Onsager reaction-field energy "
         "subtracted from every interaction eigenvalue to enforce the moment "
         "sum rule. It is zero without an Onsager closure."),
        ("chi0_eff", "chi0_eff", "Effective local susceptibility after the "
         "closure: the value actually used in the RPA denominator. It equals "
         "the fitted chi0 when no closure is active."),
    )

    def _fit_diagnostics_group_box(self, fit_entry: FitTimelineEntry) -> Any:
        """Per-dataset physics diagnostics table, or ``None`` when absent."""

        from PySide6 import QtCore, QtGui, QtWidgets

        diagnostics = (
            fit_entry.metadata.get("diagnostics")
            if isinstance(fit_entry.metadata, dict)
            else None
        )
        if not isinstance(diagnostics, dict) or not diagnostics:
            return None

        datasets = sorted(diagnostics)
        columns = self._DIAGNOSTIC_COLUMNS
        group_box = QtWidgets.QGroupBox("Physics diagnostics")
        group_box.setObjectName("fit_diagnostics_group")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        table = QtWidgets.QTableWidget(len(datasets), len(columns) + 1)
        table.setObjectName("fit_diagnostics_table")
        table.setToolTip(
            "Model-derived physical quantities per dataset, computed after the "
            "fit (see the theory notes). These are plottable vs temperature "
            "across a temperature series."
        )
        table.setHorizontalHeaderLabels(
            ["Dataset"] + [header for _key, header, _tip in columns]
        )
        for column_index, (_key, _header, tip) in enumerate(columns):
            table.horizontalHeaderItem(column_index + 1).setToolTip(tip)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for row_index, dataset_name in enumerate(datasets):
            record = diagnostics[dataset_name] or {}
            name_item = QtWidgets.QTableWidgetItem(str(dataset_name))
            name_item.setToolTip(str(dataset_name))
            if record.get("unstable"):
                name_item.setToolTip(
                    f"{dataset_name}: parameters were at or beyond the RPA "
                    "instability when diagnostics were computed."
                )
                name_item.setBackground(QtGui.QColor("#5a2929"))
            elif record.get("near_rpa_instability") or record.get("near_rpa_pole"):
                name_item.setToolTip(
                    f"{dataset_name}: the configured electronic-RPA stability "
                    "probe is near an instability or pole."
                )
                name_item.setBackground(QtGui.QColor("#65521f"))
            table.setItem(row_index, 0, name_item)
            for column_index, (key, _header, tip) in enumerate(columns):
                value = record.get(key)
                text = _format_number(float(value)) if value is not None else "-"
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(tip)
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                table.setItem(row_index, column_index + 1, item)
        _tooltip_table_corner_buttons(table, "Select all diagnostics rows.")
        table.resizeColumnsToContents()
        table.setMinimumHeight(90)
        layout.addWidget(table)
        return group_box

    def _posterior_sampler_group_box(self, group: DataGroup, fit_entry: FitTimelineEntry) -> Any:
        """Return result-only actions for the shared Posterior settings panel."""
        from PySide6 import QtWidgets

        stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
        chain_steps = 0
        chain_walkers = 0
        if stored is not None and stored.chain is not None:
            chain = np.asarray(stored.chain, dtype=float)
            if chain.ndim == 3:
                chain_steps = int(chain.shape[0])
                chain_walkers = int(chain.shape[1])
        display_options = _posterior_display_options(fit_entry)
        best_sample = _best_posterior_sample(stored) if stored is not None else None
        widget = QtWidgets.QWidget()
        widget.setObjectName("fit_posterior_result_actions")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        status = QtWidgets.QLabel(
            f"Stored samples: {0 if stored is None else len(stored.samples):d}    "
            f"Raw chain: {chain_steps:d} steps x {chain_walkers:d} walkers"
        )
        status.setToolTip("Current posterior storage for this fit result.")
        layout.addWidget(status)

        use_uncertainties = QtWidgets.QCheckBox("Use emcee uncertainties, correlations, and asymmetry")
        use_uncertainties.setObjectName("fit_posterior_use_uncertainties_check")
        use_uncertainties.setChecked(bool(display_options.get("use_posterior_uncertainties")))
        use_uncertainties.setEnabled(stored is not None and len(stored.samples) > 1)
        use_uncertainties.setToolTip(
            "Display 16%--84% emcee intervals instead of least-squares standard errors and use the emcee "
            "correlation matrix in Fit diagnostics. This affects displayed results and exported reports only."
        )
        use_uncertainties.toggled.connect(
            lambda checked: self._set_posterior_display_option(
                group, fit_entry, "use_posterior_uncertainties", checked
            )
        )
        layout.addWidget(use_uncertainties)

        use_best_sample = QtWidgets.QCheckBox("Use best sample")
        use_best_sample.setObjectName("fit_posterior_use_best_sample_check")
        use_best_sample.setChecked(bool(display_options.get("use_best_sample")))
        use_best_sample.setEnabled(best_sample is not None)
        use_best_sample.setToolTip(
            "Apply the highest-log-probability stored emcee sample to the live model and data-viewer overlay. "
            "Uncheck to restore the saved least-squares fit values."
        )
        use_best_sample.toggled.connect(
            lambda checked: self._set_posterior_display_option(
                group, fit_entry, "use_best_sample", checked
            )
        )
        layout.addWidget(use_best_sample)

        apply_button = QtWidgets.QPushButton("Apply burn-in/thin")
        apply_button.setObjectName("fit_posterior_apply_button")
        apply_button.setToolTip("Recompute posterior summaries and plots from the stored raw chain without rerunning emcee.")
        apply_button.setEnabled(stored is not None and stored.chain is not None)
        apply_button.clicked.connect(
            lambda _checked=False: self.apply_posterior_sampling_window(
                fit_entry, self.fit_emcee_burn_spin.value(), self.fit_emcee_thin_spin.value()
            )
        )

        rerun_button = QtWidgets.QPushButton("Rerun emcee")
        rerun_button.setObjectName("fit_posterior_rerun_button")
        rerun_button.setToolTip(
            "Run a new emcee posterior sample from the best-fit parameters. If samples already exist, "
            "you will be asked to confirm before the stored samples and raw chain are replaced."
        )
        rerun_button.clicked.connect(
            lambda _checked=False: self.confirm_and_start_posterior_rerun(
                group,
                fit_entry,
                n_walkers=self.fit_emcee_walkers_spin.value(),
                n_steps=self.fit_emcee_steps_spin.value(),
                burn_in=self.fit_emcee_burn_spin.value(),
                thin=self.fit_emcee_thin_spin.value(),
                random_seed=None if self.fit_emcee_seed_spin.value() < 0 else self.fit_emcee_seed_spin.value(),
                workers=-1,
            )
        )

        append_button = QtWidgets.QPushButton("Append steps")
        append_button.setObjectName("fit_posterior_append_button")
        append_button.setToolTip(
            "Continue the stored emcee chain from its last walker positions and then recompute summaries with the selected burn-in/thin."
        )
        append_button.setEnabled(stored is not None and stored.chain is not None)
        append_button.clicked.connect(
            lambda _checked=False: self.start_posterior_sampler_for_fit(
                group,
                fit_entry,
                n_walkers=self.fit_emcee_walkers_spin.value(),
                n_steps=self.fit_emcee_steps_spin.value(),
                burn_in=self.fit_emcee_burn_spin.value(),
                thin=self.fit_emcee_thin_spin.value(),
                random_seed=None if self.fit_emcee_seed_spin.value() < 0 else self.fit_emcee_seed_spin.value(),
                workers=-1,
                append=True,
            )
        )

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(apply_button)
        button_row.addWidget(rerun_button)
        button_row.addWidget(append_button)
        layout.addLayout(button_row)
        return widget

    def _set_posterior_display_option(
        self,
        group: DataGroup,
        fit_entry: FitTimelineEntry,
        option: str,
        checked: bool,
    ) -> None:
        """Persist a posterior result choice and update the active live state."""

        display = _posterior_display_options(fit_entry)
        display[option] = bool(checked)
        if option == "use_best_sample":
            if checked:
                stored = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
                best = _best_posterior_sample(stored) if stored is not None else None
                if best is None:
                    display[option] = False
                    display.pop("best_sample", None)
                else:
                    params, log_probability, location = best
                    display["best_sample"] = {
                        "parameters": params,
                        "log_probability": log_probability,
                        **location,
                    }
            else:
                display.pop("best_sample", None)
        fit_entry.metadata[POSTERIOR_DISPLAY_KEY] = display
        if option == "use_best_sample" and self._active_fit_entry(group) is fit_entry:
            if fit_entry.snapshot:
                restore_data_group_state(group, fit_entry.snapshot)
            if display.get("use_best_sample"):
                try:
                    _apply_displayed_fit_parameters(group, fit_entry)
                except Exception:
                    # Keep the known-good least-squares snapshot live if an
                    # old or incomplete saved posterior cannot be compiled.
                    restore_data_group_state(group, fit_entry.snapshot)
                    display[option] = False
                    display.pop("best_sample", None)
                    fit_entry.metadata[POSTERIOR_DISPLAY_KEY] = display
            self._request_overlay_refresh(group)
        self._mark_dirty()
        self._set_fit_details(fit_entry)

    def _parameter_values_group_box(self, fit_entry: FitTimelineEntry) -> Any:
        from PySide6 import QtCore, QtGui, QtWidgets

        rows = _snapshot_parameter_rows(fit_entry)
        group_box = QtWidgets.QGroupBox("Parameter values")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(10, 8, 10, 8)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setObjectName("parameter_values_table")
        table.setToolTip(
            "Current model parameter values for this state. Run a fit to produce best-fit values with uncertainties."
        )
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(100)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(("name", "value")):
                item = QtWidgets.QTableWidgetItem(row.get(key, "-"))
                item.setToolTip(row.get(key, "-"))
                if row.get("at_limit"):
                    item.setForeground(QtGui.QColor("#c0392b"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setToolTip(f"{item.toolTip()}\nReached a configured parameter bound.")
                if column_index > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row_index, column_index, item)
        _tooltip_table_corner_buttons(table, "Select all parameter rows.")
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return group_box

    def _dataset_point_list_group_box(self, dataset: DatasetEntry, group: DataGroup | None) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_point_list_group_box,
            self,
            dataset,
            group,
            helper_namespace=globals(),
        )

    def _heat_capacity_box(self, dataset, group, config) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._heat_capacity_box,
            self,
            dataset,
            group,
            config,
            helper_namespace=globals(),
        )

    def _set_heat_capacity_setting(self, dataset, group, key: str, value) -> None:
        if dataset.parameters.get(key) == value:
            return
        dataset.parameters[key] = value
        self._after_point_list_changed(dataset, group)

    def _set_heat_capacity_fit_channel(self, dataset, group, value: str) -> None:
        config = point_list_config(dataset)
        hc = config.setdefault("heat_capacity", {})
        if hc.get("fit_channel") == value:
            return
        hc["fit_channel"] = value
        self._after_point_list_changed(dataset, group)

    def _set_heat_capacity_source_unit(self, dataset, group, value: str) -> None:
        config = point_list_config(dataset)
        hc = config.setdefault("heat_capacity", {})
        if hc.get("source_unit") == value:
            return
        hc["source_unit"] = value
        self._after_point_list_changed(dataset, group)

    def _magnetization_absolute_box(self, dataset, group) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._magnetization_absolute_box,
            self,
            dataset,
            group,
            helper_namespace=globals(),
        )

    def _set_magnetization_absolute(self, dataset, group, key: str, value) -> None:
        if dataset.parameters.get(key) == value:
            return
        dataset.parameters[key] = value
        self._after_point_list_changed(dataset, group)

    def _point_list_scale_box(self, dataset, group, config, columns) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._point_list_scale_box,
            self,
            dataset,
            group,
            config,
            columns,
            helper_namespace=globals(),
        )

    def _point_list_susceptibility_box(self, dataset, group, config) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._point_list_susceptibility_box,
            self,
            dataset,
            group,
            config,
            helper_namespace=globals(),
        )

    def _point_list_wavelength_box(self, dataset, group, config) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._point_list_wavelength_box,
            self,
            dataset,
            group,
            config,
            helper_namespace=globals(),
        )

    def _set_point_list_coordinates(self, dataset, group, text: str) -> None:
        names = [part.strip() for part in text.split(",") if part.strip()]
        columns = dataset.data.column_names if isinstance(dataset.data, PointListData) else []
        names = [name for name in names if name in columns]
        config = point_list_config(dataset)
        if config.get("coordinate_names") == names:
            return
        config["coordinate_names"] = names
        self._after_point_list_changed(dataset, group)

    def _set_point_list_channel(self, dataset, group, index: int, key: str, text: str) -> None:
        config = point_list_config(dataset)
        channels = config.get("channels", [])
        if not (0 <= index < len(channels)):
            return
        value = None if (key == "error" and text == "(none)") else text
        if channels[index].get(key) == value:
            return
        channels[index][key] = value
        if key == "value":
            channels[index]["label"] = value
            if isinstance(dataset.data, PointListData):
                channels[index]["quantity_type"] = dataset.data.quantity_type(str(value))
                channels[index]["unit"] = dataset.data.unit(str(value))
        self._after_point_list_changed(dataset, group)

    def _add_point_list_channel(self, dataset, group) -> None:
        config = point_list_config(dataset)
        columns = list(dataset.data.column_names)
        if not columns:
            return
        config.setdefault("channels", []).append(
            {"label": columns[0], "value": columns[0], "error": None}
        )
        self._after_point_list_changed(dataset, group)

    def _remove_point_list_channel(self, dataset, group, index: int) -> None:
        config = point_list_config(dataset)
        channels = config.get("channels", [])
        if 0 <= index < len(channels):
            channels.pop(index)
            self._after_point_list_changed(dataset, group)

    def _set_point_list_scale(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        scale = config.setdefault("scale", {})
        new_value = float(value) if key == "factor" else value
        if scale.get(key) == new_value:
            return
        scale[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _set_point_list_susceptibility(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        susc = config.setdefault("susceptibility", {})
        new_value = bool(value) if key == "enabled" else value
        if susc.get(key) == new_value:
            return
        susc[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _set_point_list_wavelength(self, dataset, group, key: str, value) -> None:
        config = point_list_config(dataset)
        wavelength = config.setdefault("wavelength", {})
        new_value = float(value) if key == "value" else value
        if wavelength.get(key) == new_value:
            return
        wavelength[key] = new_value
        self._after_point_list_changed(dataset, group)

    def _after_point_list_changed(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._set_dataset_details(dataset, group)

    def _dataset_type_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        lines: list[str],
    ) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_type_group_box,
            self,
            dataset,
            group,
            lines,
            helper_namespace=globals(),
        )

    def _set_selected_data_type(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        data_type: str,
    ) -> None:
        if not data_type or data_type == dataset.data_type:
            return
        set_dataset_data_type(dataset, data_type)
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._set_dataset_details(dataset, group)

    def _dataset_axes_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        lines: list[str],
    ) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_axes_group_box,
            self,
            dataset,
            group,
            lines,
            helper_namespace=globals(),
        )

    def _add_rebin_performance_controls(self, row, *, dataset=None, group=None, composite=False):
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._add_rebin_performance_controls,
            self,
            row,
            dataset=dataset,
            group=group,
            composite=composite,
            helper_namespace=globals(),
        )

    def _details_group_box(self, title: str, lines: list[str]) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._details_group_box,
            self,
            title,
            lines,
            helper_namespace=globals(),
        )

    def _dataset_metadata_group_box(self, dataset: DatasetEntry) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_metadata_group_box,
            self,
            dataset,
            helper_namespace=globals(),
        )

    def _set_analysis_output_details(self, output: AnalysisOutputRef) -> None:
        """Show persisted analysis table values and metadata in the project panel."""

        self.title_label.setText(output.label)
        self._clear_details_panel()
        summary = [f"Kind: {output.kind}", f"Output key: {output.key}"]
        if output.scalar_value is not None:
            summary.append(f"Value: {_format_number(output.scalar_value)} {output.unit}")
        self.details_layout.addWidget(self._details_group_box("Analysis output", summary))
        data = None
        if self.project_path is not None and output.artifact_path:
            try:
                data = read_project_dataset_artifact(
                    self.project_path,
                    output.artifact_path,
                )
            except (OSError, TypeError, ValueError) as exc:
                self.details_layout.addWidget(
                    self._details_group_box("Artifact", [f"Could not load output: {exc}"])
                )
        if isinstance(data, PointListData):
            self.details_layout.addWidget(self._analysis_table_group_box(data))
            metadata = {**dict(output.metadata), **dict(data.metadata)}
            self.details_layout.addWidget(
                self._metadata_tree_group_box(
                    "Metadata",
                    metadata,
                    object_name="analysis_output_metadata_tree",
                    empty_text="No output metadata.",
                )
            )
        elif isinstance(data, MDHistoData):
            axes, lines = _dataset_axes_and_data_lines(data)
            self.details_layout.addWidget(self._details_group_box("Axes", axes))
            self.details_layout.addWidget(self._details_group_box("Data", lines))
            self.details_layout.addWidget(
                self._metadata_tree_group_box(
                    "Metadata",
                    {**dict(output.metadata), **dict(data.metadata)},
                    object_name="analysis_output_metadata_tree",
                    empty_text="No output metadata.",
                )
            )
        self.details_layout.addStretch(1)

    def _analysis_table_group_box(self, data: PointListData) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._analysis_table_group_box,
            self,
            data,
            helper_namespace=globals(),
        )

    def _dataset_signal_semantics_group_box(
        self, dataset: DatasetEntry, group: DataGroup | None
    ) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_signal_semantics_group_box,
            self,
            dataset,
            group,
            helper_namespace=globals(),
        )

    def _dataset_spectral_channel_config(
        self, dataset: DatasetEntry
    ) -> dict[str, Any]:
        """Return the normalized paired-channel INS configuration."""

        config = normalized_spectral_channel_config(
            dataset.parameters.get(SPECTRAL_CHANNEL_CONFIG_KEY)
        )
        if config["source_unit"] != "arbitrary":
            if config["normalization_basis"] == "per_magnetic_ion":
                label = str(config.get("normalization_label", "") or "").strip()
                suffix = f"/{label}" if label else "/magnetic ion"
            else:
                suffix = {
                    "per_formula_unit": "/f.u.",
                    "per_unit_cell": "/unit cell",
                    "unknown": "",
                }[config["normalization_basis"]]
            unit = str(config["source_unit"])
            prefix = next(
                (
                    candidate
                    for candidate in (
                        "mbarn/sr/meV",
                        "barn/sr/meV",
                        "mu_B^2/meV",
                        "spin^2/meV",
                        "1/meV",
                    )
                    if unit.startswith(candidate)
                ),
                unit,
            )
            config["source_unit"] = prefix + suffix
        for target, names in (
            ("incident_energy_meV", ("incident_energy_meV", "incident_energy", "Ei", "ei")),
            ("final_energy_meV", ("final_energy_meV", "final_energy", "Ef", "ef")),
        ):
            if config.get(target) not in (None, ""):
                continue
            for source in (dataset.metadata, getattr(dataset.data, "metadata", {})):
                if not isinstance(source, dict):
                    continue
                value = next((source[name] for name in names if name in source), None)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    config[target] = value
                    break
        dataset.parameters[SPECTRAL_CHANNEL_CONFIG_KEY] = config
        return config

    def _set_dataset_spectral_channel_setting(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        key: str,
        value: Any,
    ) -> None:
        config = self._dataset_spectral_channel_config(dataset)
        if config.get(key) == value:
            return
        from .spectral_channels import updated_spectral_channel_config

        dataset.parameters[SPECTRAL_CHANNEL_CONFIG_KEY] = updated_spectral_channel_config(
            config, key, value
        )
        if group is not None:
            self._record_data_group_state_change(group)
            self.refresh_slice_viewer(group)
        self._mark_dirty()
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _dataset_spectral_channels_group_box(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
    ) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._dataset_spectral_channels_group_box,
            self,
            dataset,
            group,
            helper_namespace=globals(),
        )

    def _set_dataset_signal_semantics(
        self, dataset: DatasetEntry, group: DataGroup | None, semantics: str
    ) -> None:
        if not isinstance(dataset.data, MDHistoData):
            return
        if semantics not in {"density", "bin_integral", "unknown"}:
            return
        if dataset.data.metadata.get("signal_semantics") == semantics:
            return
        metadata = dict(dataset.data.metadata)
        metadata["signal_semantics"] = semantics
        metadata["signal_semantics_source"] = "user_selected"
        dataset.replace_data(
            dataset.data.with_updates(metadata=metadata),
            source_backed=False,
        )
        if group is not None:
            self._record_data_group_state_change(group)
            self.refresh_slice_viewer(group)
        self._mark_dirty()
        self._refresh_cache_badges()
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _metadata_tree_group_box(
        self,
        title: str,
        mapping: dict[str, Any],
        *,
        object_name: str,
        empty_text: str,
    ) -> Any:
        return _project_data_panels.invoke_panel_builder(
            _project_data_panels._metadata_tree_group_box,
            self,
            title,
            mapping,
            object_name=object_name,
            empty_text=empty_text,
            helper_namespace=globals(),
        )

    def _copy_rebin_settings(self, config: dict[str, Any]) -> None:
        """Copy one dataset or composite rebin recipe to the system clipboard."""

        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setText(
            _rebin_settings_clipboard_text(config)
        )

    def _selected_dataset_binning(self, dataset: DatasetEntry) -> dict[str, Any]:
        binnings = dataset_rebin_binnings(dataset)
        selected_id = self._selected_dataset_binning_ids.get(dataset.id, binnings[0]["id"])
        selected = next((item for item in binnings if item["id"] == selected_id), binnings[0])
        self._selected_dataset_binning_ids[dataset.id] = selected["id"]
        dataset._nfit_selected_binning_id = selected["id"]
        return selected

    def _group_for_dataset(self, dataset: DatasetEntry) -> DataGroup | None:
        return next(
            (
                group
                for group in self.project.data_groups
                if any(candidate is dataset for candidate in group.iter_datasets())
            ),
            None,
        )

    def _selected_composite_binning(
        self, group: DataGroup | _CompositeScope
    ) -> dict[str, Any]:
        owner = group.node if isinstance(group, _CompositeScope) else group
        owner_key = getattr(owner, "id", f"root:{id(owner)}")
        binnings = data_group_composite_binnings(group)
        selected_id = self._selected_composite_binning_ids.get(owner_key, binnings[0]["id"])
        selected = next((item for item in binnings if item["id"] == selected_id), binnings[0])
        self._selected_composite_binning_ids[owner_key] = selected["id"]
        owner._nfit_selected_binning_id = selected["id"]
        return selected

    def _select_dataset_binning(self, dataset: DatasetEntry, binning_id: str) -> None:
        self._selected_dataset_binning_ids[dataset.id] = str(binning_id)
        dataset._nfit_selected_binning_id = str(binning_id)
        group = self._group_for_dataset(dataset)
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _select_composite_binning(
        self, group: DataGroup | _CompositeScope, binning_id: str
    ) -> None:
        owner = group.node if isinstance(group, _CompositeScope) else group
        owner_key = getattr(owner, "id", f"root:{id(owner)}")
        self._selected_composite_binning_ids[owner_key] = str(binning_id)
        owner._nfit_selected_binning_id = str(binning_id)
        self._sync_details()

    def _add_dataset_binning(self, dataset: DatasetEntry, *, duplicate: bool) -> None:
        from PySide6 import QtWidgets

        selected = self._selected_dataset_binning(dataset)
        default_name = f"{selected['name']} copy" if duplicate else "New binning"
        name, accepted = QtWidgets.QInputDialog.getText(
            self.window, "Add binning", "Binning name", text=default_name
        )
        if not accepted:
            return
        binning_id = add_dataset_rebin_binning(
            dataset,
            name=name,
            duplicate_from=(selected["id"] if duplicate else None),
        )
        self._selected_dataset_binning_ids[dataset.id] = binning_id
        dataset._nfit_selected_binning_id = binning_id
        self._mark_dirty()
        self._refresh_cache_badges()
        self._set_dataset_details_preserving_scroll(dataset, self._group_for_dataset(dataset))

    def _rename_dataset_binning(self, dataset: DatasetEntry) -> None:
        from PySide6 import QtWidgets

        selected = self._selected_dataset_binning(dataset)
        name, accepted = QtWidgets.QInputDialog.getText(
            self.window, "Rename binning", "Binning name", text=selected["name"]
        )
        if not accepted:
            return
        try:
            rename_dataset_rebin_binning(dataset, selected["id"], name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Rename binning", str(exc))
            return
        self._mark_dirty()
        self._set_dataset_details_preserving_scroll(dataset, self._group_for_dataset(dataset))

    def _remove_dataset_binning(self, dataset: DatasetEntry) -> None:
        selected = self._selected_dataset_binning(dataset)
        remove_dataset_rebin_binning(dataset, selected["id"])
        self._selected_dataset_binning_ids.pop(dataset.id, None)
        dataset._nfit_selected_binning_id = _fit_dataset_rebin_config(dataset).get(
            "_binning_id", FIT_BINNING_ID
        )
        self._mark_dirty()
        group = self._group_for_dataset(dataset)
        if group is not None:
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _make_dataset_fit_binning(self, dataset: DatasetEntry) -> None:
        selected = self._selected_dataset_binning(dataset)
        make_dataset_fit_binning(dataset, selected["id"])
        self._mark_dirty()
        group = self._group_for_dataset(dataset)
        if group is not None:
            self._record_data_group_state_change(group)
            self.refresh_slice_viewer(group, force_rebin=True)
            self._request_overlay_refresh(group)
        self._refresh_cache_badges()
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _add_composite_binning(
        self, group: DataGroup | _CompositeScope, *, duplicate: bool
    ) -> None:
        from PySide6 import QtWidgets

        selected = self._selected_composite_binning(group)
        default_name = f"{selected['name']} copy" if duplicate else "New binning"
        name, accepted = QtWidgets.QInputDialog.getText(
            self.window, "Add composite binning", "Binning name", text=default_name
        )
        if not accepted:
            return
        binning_id = add_data_group_composite_binning(
            group,
            name=name,
            duplicate_from=(selected["id"] if duplicate else None),
        )
        owner = group.node if isinstance(group, _CompositeScope) else group
        owner_key = getattr(owner, "id", f"root:{id(owner)}")
        self._selected_composite_binning_ids[owner_key] = binning_id
        owner._nfit_selected_binning_id = binning_id
        self._mark_dirty()
        self._refresh_cache_badges()
        self._sync_details()

    def _rename_composite_binning(self, group: DataGroup | _CompositeScope) -> None:
        from PySide6 import QtWidgets

        selected = self._selected_composite_binning(group)
        name, accepted = QtWidgets.QInputDialog.getText(
            self.window, "Rename composite binning", "Binning name", text=selected["name"]
        )
        if not accepted:
            return
        try:
            rename_data_group_composite_binning(group, selected["id"], name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Rename composite binning", str(exc))
            return
        self._mark_dirty()
        self._sync_details()

    def _remove_composite_binning(self, group: DataGroup | _CompositeScope) -> None:
        selected = self._selected_composite_binning(group)
        remove_data_group_composite_binning(group, selected["id"])
        owner = group.node if isinstance(group, _CompositeScope) else group
        owner_key = getattr(owner, "id", f"root:{id(owner)}")
        self._selected_composite_binning_ids.pop(owner_key, None)
        owner._nfit_selected_binning_id = _fit_data_group_composite_config(group).get(
            "_binning_id", FIT_BINNING_ID
        )
        self._mark_dirty()
        self.refresh_slice_viewer(_composite_root(group))
        self._refresh_cache_badges()
        self._sync_details()

    def _make_composite_fit_binning(self, group: DataGroup | _CompositeScope) -> None:
        selected = self._selected_composite_binning(group)
        make_data_group_fit_binning(group, selected["id"])
        root = _composite_root(group)
        self._record_data_group_state_change(root)
        self._mark_dirty()
        self.refresh_slice_viewer(root, force_rebin=True)
        self._request_overlay_refresh(root)
        self._refresh_cache_badges()
        self._sync_details()

    def _pasted_rebin_config(self, target_config: dict[str, Any]) -> dict[str, Any] | None:
        """Read and validate rebin settings from the system clipboard."""

        from PySide6 import QtWidgets

        try:
            return _rebin_config_from_clipboard_text(
                QtWidgets.QApplication.clipboard().text(),
                target_config,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Paste rebin settings",
                str(exc),
            )
            return None

    def _paste_dataset_rebin_settings(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
    ) -> bool:
        selected = self._selected_dataset_binning(dataset)
        pasted = self._pasted_rebin_config(selected["config"])
        if pasted is None:
            return False
        selected["config"].clear()
        selected["config"].update(pasted)
        self._after_dataset_rebin_changed(dataset, group)
        # Pasting can change every control, including enablement and resolution
        # mode, so rebuild the panel instead of relying on the lightweight
        # numeric-field refresh used for ordinary single-value edits.
        self._set_dataset_details_preserving_scroll(dataset, group)
        return True

    def _paste_group_composite_settings(
        self,
        group: DataGroup | _CompositeScope,
    ) -> bool:
        pasted = self._pasted_rebin_config({
            **self._selected_composite_binning(group)["config"],
            "metadata_dimensions": group.metadata.get("metadata_dimensions", []),
        })
        if pasted is None:
            return False
        from PySide6 import QtWidgets

        dimensions = pasted.pop("metadata_dimensions")
        try:
            set_metadata_dimensions(group, dimensions)
        except (ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Paste rebin settings", str(exc))
            return False
        selected = self._selected_composite_binning(group)["config"]
        selected.clear()
        selected.update(pasted)
        self._after_group_composite_changed(group)
        return True

    def _set_dataset_rebin_enabled(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        checked: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        if bool(config.get("enabled", False)) == bool(checked):
            return
        config["enabled"] = bool(checked)
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_option(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        key: str,
        checked: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        if bool(config.get(key, False)) == bool(checked):
            return
        config[key] = bool(checked)
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_auto(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        checked: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        if bool(config.get("auto_rebin", True)) == bool(checked):
            return
        config["auto_rebin"] = bool(checked)
        config["auto_rebin_user_override"] = bool(checked)
        if checked and config.get("stale") and group is not None:
            self._after_dataset_rebin_changed(dataset, group)
            return
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _set_dataset_rebin_mean_weighting(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        value: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        value = value if value in {"inverse_variance", "uniform"} else "uniform"
        if _rebin_mean_weighting(config) == value:
            return
        config["mean_weighting"] = value
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_minimum_coverage(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        editor: Any,
    ) -> None:
        config = dataset_rebin_config(dataset)
        try:
            value = float(editor.text())
        except ValueError:
            editor.setText(_format_number(_rebin_minimum_coverage(config)))
            return
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            editor.setText(_format_number(_rebin_minimum_coverage(config)))
            return
        if np.isclose(_rebin_minimum_coverage(config), value):
            return
        config["minimum_coverage"] = value
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_minimum_samples(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        editor: Any,
    ) -> None:
        config = dataset_rebin_config(dataset)
        try:
            value = float(editor.text())
        except ValueError:
            editor.setText(_format_number(_rebin_minimum_samples(config)))
            return
        if not np.isfinite(value) or value < 0.0:
            editor.setText(_format_number(_rebin_minimum_samples(config)))
            return
        if np.isclose(_rebin_minimum_samples(config), value):
            return
        config["minimum_samples"] = value
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_max_batch_mb(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        value: int,
    ) -> None:
        config = dataset_rebin_config(dataset)
        value = max(int(value), 1)
        if _rebin_max_batch_mb(config) == value:
            return
        config["max_batch_mb"] = value
        config["normalize"] = True
        self._after_dataset_rebin_changed(dataset, group)

    def _dataset_rebin_symmetry_preview(
        self, dataset: DatasetEntry, group: DataGroup | None
    ) -> str:
        config = dataset_rebin_config(dataset)
        lattice = _dataset_lattice_parameters(dataset)
        if lattice is None and group is not None:
            lattice = group.lattice_parameters
        try:
            operations = _rebin_symmetry_operations(config, lattice)
        except (ImportError, ValueError) as exc:
            return f"Invalid: {exc}"
        spec = symmetry_spec_from_config(config.get("symmetry"))
        return "No symmetry" if not spec.enabled else f"{len(operations)} operations"

    def _set_dataset_rebin_symmetry_enabled(
        self, dataset: DatasetEntry, group: DataGroup | None, checked: bool
    ) -> None:
        config = dataset_rebin_config(dataset)
        payload = config["symmetry"]
        modes = {"space_group", "point_group", "operations", "generators"}
        if checked:
            previous_mode = str(payload.get("last_mode", "space_group"))
            payload["mode"] = previous_mode if previous_mode in modes else "space_group"
            if not str(payload.get("expression", "")).strip():
                payload["expression"] = str(
                    group.spacegroup
                    if group is not None and group.spacegroup
                    else "P 1"
                )
        else:
            current_mode = str(payload.get("mode", "space_group"))
            if current_mode in modes:
                payload["last_mode"] = current_mode
            payload["mode"] = "none"
        if group is not None and isinstance(group.lattice_parameters, dict):
            payload["lattice_parameters"] = copy.deepcopy(group.lattice_parameters)
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_symmetry_mode(
        self, dataset: DatasetEntry, group: DataGroup | None, mode: str
    ) -> None:
        config = dataset_rebin_config(dataset)
        payload = config["symmetry"]
        selected_mode = (
            mode
            if mode in {"space_group", "point_group", "operations", "generators"}
            else "space_group"
        )
        if str(payload.get("mode", "none")) == "none":
            payload["last_mode"] = selected_mode
        else:
            payload["mode"] = selected_mode
        if group is not None and isinstance(group.lattice_parameters, dict):
            payload["lattice_parameters"] = copy.deepcopy(group.lattice_parameters)
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_symmetry_expression(
        self, dataset: DatasetEntry, group: DataGroup | None, expression: str
    ) -> None:
        config = dataset_rebin_config(dataset)
        config["symmetry"]["expression"] = str(expression).strip()
        if group is not None and isinstance(group.lattice_parameters, dict):
            config["symmetry"]["lattice_parameters"] = copy.deepcopy(group.lattice_parameters)
        self._after_dataset_rebin_changed(dataset, group)

    def _set_group_composite_enabled(self, group: DataGroup | _CompositeScope, checked: bool) -> None:
        config = data_group_composite_config(group)
        if bool(config.get("enabled", False)) == bool(checked):
            return
        config["enabled"] = bool(checked)
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_option(self, group: DataGroup | _CompositeScope, key: str, checked: bool) -> None:
        config = data_group_composite_config(group)
        if bool(config.get(key, False)) == bool(checked):
            return
        config[key] = bool(checked)
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_auto(self, group: DataGroup | _CompositeScope, checked: bool) -> None:
        config = data_group_composite_config(group)
        if bool(config.get("auto_rebin", True)) == bool(checked):
            return
        config["auto_rebin"] = bool(checked)
        config["auto_rebin_user_override"] = bool(checked)
        if checked and config.get("stale"):
            self._after_group_composite_changed(group)
            return
        self._record_data_group_state_change(_composite_root(group))
        self._mark_dirty()
        self._sync_details()

    def _set_group_composite_mean_weighting(self, group: DataGroup | _CompositeScope, value: str) -> None:
        config = data_group_composite_config(group)
        value = (
            value
            if value in {"inverse_variance", "uniform"}
            else "uniform"
        )
        if _rebin_mean_weighting(config) == value:
            return
        config["mean_weighting"] = value
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_minimum_coverage(
        self,
        group: DataGroup | _CompositeScope,
        editor: Any,
    ) -> None:
        config = data_group_composite_config(group)
        try:
            value = float(editor.text())
        except ValueError:
            editor.setText(_format_number(_rebin_minimum_coverage(config)))
            return
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            editor.setText(_format_number(_rebin_minimum_coverage(config)))
            return
        if np.isclose(_rebin_minimum_coverage(config), value):
            return
        config["minimum_coverage"] = value
        self._after_group_composite_changed(group)

    def _set_group_composite_minimum_samples(
        self,
        group: DataGroup | _CompositeScope,
        editor: Any,
    ) -> None:
        config = data_group_composite_config(group)
        try:
            value = float(editor.text())
        except ValueError:
            editor.setText(_format_number(_rebin_minimum_samples(config)))
            return
        if not np.isfinite(value) or value < 0.0:
            editor.setText(_format_number(_rebin_minimum_samples(config)))
            return
        if np.isclose(_rebin_minimum_samples(config), value):
            return
        config["minimum_samples"] = value
        self._after_group_composite_changed(group)

    def _set_group_composite_max_batch_mb(self, group: DataGroup | _CompositeScope, value: int) -> None:
        config = data_group_composite_config(group)
        value = max(int(value), 1)
        if _rebin_max_batch_mb(config) == value:
            return
        config["max_batch_mb"] = value
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_symmetry_enabled(self, group: DataGroup | _CompositeScope, checked: bool) -> None:
        config = data_group_composite_config(group)
        payload = config["symmetry"]
        root = _composite_root(group)
        modes = {"space_group", "point_group", "operations", "generators"}
        if checked:
            previous_mode = str(payload.get("last_mode", "space_group"))
            payload["mode"] = previous_mode if previous_mode in modes else "space_group"
            if not str(payload.get("expression", "")).strip():
                payload["expression"] = str(root.spacegroup or "P 1")
        else:
            current_mode = str(payload.get("mode", "space_group"))
            if current_mode in modes:
                payload["last_mode"] = current_mode
            payload["mode"] = "none"
        if isinstance(root.lattice_parameters, dict):
            payload["lattice_parameters"] = copy.deepcopy(root.lattice_parameters)
        self._after_group_composite_changed(group)

    def _set_group_composite_symmetry_mode(self, group: DataGroup | _CompositeScope, mode: str) -> None:
        config = data_group_composite_config(group)
        payload = config["symmetry"]
        selected_mode = (
            mode
            if mode in {"space_group", "point_group", "operations", "generators"}
            else "space_group"
        )
        if str(payload.get("mode", "none")) == "none":
            payload["last_mode"] = selected_mode
        else:
            payload["mode"] = selected_mode
        root = _composite_root(group)
        if isinstance(root.lattice_parameters, dict):
            config["symmetry"]["lattice_parameters"] = copy.deepcopy(root.lattice_parameters)
        self._after_group_composite_changed(group)

    def _set_group_composite_symmetry_expression(self, group: DataGroup | _CompositeScope, expression: str) -> None:
        config = data_group_composite_config(group)
        config["symmetry"]["expression"] = str(expression).strip()
        root = _composite_root(group)
        if isinstance(root.lattice_parameters, dict):
            config["symmetry"]["lattice_parameters"] = copy.deepcopy(root.lattice_parameters)
        self._after_group_composite_changed(group)

    def _set_group_composite_resolution_mode(
        self,
        group: DataGroup | _CompositeScope,
        mode: str,
    ) -> None:
        mode = "bins" if mode == "bins" else "step"
        config = data_group_composite_config(group)
        if _rebin_resolution_mode(config) == mode:
            return
        config[REBIN_RESOLUTION_MODE_KEY] = mode
        if mode == "bins":
            for axis in config.get("axes", []):
                axis["step_size"] = _step_size_from_bounds(
                    axis.get("lower", 0.0), axis.get("upper", 0.0), max(int(axis.get("num_bins", 1)), 1)
                )
        self._after_group_composite_changed(group)

    def _set_group_composite_coordinate_mode(
        self, group: DataGroup | _CompositeScope, value: str
    ) -> None:
        """Switch an event-data composite between HKLE and powder coordinates."""

        mode = "powder" if value == "powder" else "hkle"
        config = data_group_composite_config(group)
        if config.get("coordinate_mode") == mode:
            return
        config["coordinate_mode"] = mode
        config.pop("axes", None)
        data_group_composite_config(group)
        config["stale"] = True
        self._record_data_group_state_change(_composite_root(group))
        self._mark_dirty()
        self._refresh_tree(select_group=_composite_root(group))
        self._sync_details()

    def _set_group_composite_axis_value(self, group: DataGroup | _CompositeScope, index: int, key: str, text: str) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes")
        if not isinstance(axes, list) or not (0 <= index < len(axes)):
            return
        axis = dict(axes[index])
        previous_step = float(axis.get("step_size", 0.0) or 0.0)
        if key in {"lower", "upper"} and not text.strip():
            if _rebin_axis_bound_is_auto(axis, key):
                return
            axis[f"auto_{key}"] = True
            automatic = axis.get(f"auto_{key}_value")
            if automatic is not None and np.isfinite(float(automatic)):
                axis[key] = float(automatic)
            axes[index] = axis
            self._after_group_composite_changed(group)
            return
        try:
            if key == "num_bins":
                axis[key] = max(int(float(text)), 1)
                axis["step_size"] = _step_size_from_bounds(
                    axis.get("lower", 0.0), axis.get("upper", 0.0), axis["num_bins"]
                )
            else:
                axis[key] = float(text)
                if key == "step_size":
                    axis["auto_step_size"] = False
                if key in {"lower", "upper"}:
                    axis[f"auto_{key}"] = False
        except ValueError:
            self._sync_details()
            return
        if key == "step_size":
            step = float(axis.get("step_size", 0.0))
            if step > 0.0:
                axis["num_bins"] = _num_bins_from_step_size(axis.get("lower", 0.0), axis.get("upper", 0.0), step)
        elif key in {"lower", "upper"} and _rebin_axis_mode(config, axis) == "step":
            if previous_step > 0.0 and np.isfinite(previous_step):
                axis["num_bins"] = _num_bins_from_step_size(
                    axis.get("lower", 0.0), axis.get("upper", 0.0), previous_step
                )
        elif key in {"lower", "upper"}:
            axis["step_size"] = _step_size_from_bounds(
                axis.get("lower", 0.0), axis.get("upper", 0.0), axis["num_bins"]
            )
        axis.update(_sanitize_rebin_axis_config(axis))
        axes[index] = axis
        config["normalize"] = True
        self._after_group_composite_changed(group)

    def _set_group_composite_axis_edges(
        self,
        group: DataGroup | _CompositeScope,
        index: int,
        text: str,
    ) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        try:
            parsed = _parse_parameter_text(text)
            if parsed == "":
                axes[index].pop("bin_edges", None)
            else:
                axes[index].update(
                    _sanitize_rebin_axis_config(
                        {**axes[index], "bin_edges": _normalize_rebin_bin_edges(parsed)}
                    )
                )
        except (TypeError, ValueError):
            self._sync_details()
            return
        self._after_group_composite_changed(group)

    def _set_group_composite_axis_mode(
        self,
        group: DataGroup | _CompositeScope,
        index: int,
        value: str,
    ) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        mode = value if value in REBIN_AXIS_MODES else "step"
        if _rebin_axis_mode(config, axes[index]) == mode:
            return
        axes[index]["mode"] = mode
        if mode in {"discrete", "tolerance"}:
            axes[index]["fractional"] = False
        if mode == "tolerance":
            axes[index].setdefault(
                "tolerance", max(float(axes[index].get("step_size", 0.1)), 1e-12)
            )
        self._after_group_composite_changed(group)

    def _set_group_composite_axis_fractional(
        self,
        group: DataGroup | _CompositeScope,
        index: int,
        fractional: bool,
    ) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        value = bool(fractional) and _rebin_axis_mode(config, axes[index]) not in {
            "discrete",
            "tolerance",
        }
        if _rebin_axis_fractional(config, axes[index]) == value:
            return
        axes[index]["fractional"] = value
        self._after_group_composite_changed(group)

    def _set_group_composite_axis_vector(self, group: DataGroup | _CompositeScope, index: int, text: str) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        try:
            parsed = _parse_parameter_text(text)
            if not isinstance(parsed, (list, tuple)):
                raise ValueError("the momentum row must be a list of three values")
            _update_rebin_momentum_basis(axes, index, parsed)
        except (TypeError, ValueError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin momentum coordinates",
                f"Could not change the momentum-coordinate matrix:\n{exc}",
            )
            self._sync_details()
            return
        self._after_group_composite_changed(group)

    def _set_group_composite_momentum_matrix(
        self, group: DataGroup | _CompositeScope, text: str
    ) -> None:
        config = data_group_composite_config(group)
        axes = config.get("axes", [])
        try:
            matrix = _parse_parameter_text(text)
            _update_rebin_momentum_matrix(axes, matrix)
        except (TypeError, ValueError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin momentum coordinates",
                f"Could not change the momentum-coordinate matrix:\n{exc}",
            )
            self._sync_details()
            return
        self._after_group_composite_changed(group)

    def _set_dataset_rebin_axis_value(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        key: str,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        axis = axes[index]
        previous_step = float(axis.get("step_size", 0.0) or 0.0)
        if key in {"lower", "upper"} and not text.strip():
            if _rebin_axis_bound_is_auto(axis, key):
                return
            axis[f"auto_{key}"] = True
            automatic = axis.get(f"auto_{key}_value")
            if automatic is not None and np.isfinite(float(automatic)):
                axis[key] = float(automatic)
            self._after_dataset_rebin_changed(dataset, group)
            return
        try:
            if key == "num_bins":
                value: Any = max(int(float(text)), 1)
                if axis.get("num_bins") == value:
                    return
                axis["num_bins"] = value
                axis["step_size"] = _step_size_from_bounds(
                    axis.get("lower", 0.0), axis.get("upper", 0.0), value
                )
            else:
                value = float(text)
                if key == "step_size":
                    if value <= 0.0 or not np.isfinite(value):
                        return
                    axis["step_size"] = value
                    axis["auto_step_size"] = False
                    num_bins = _num_bins_from_step_size(axis.get("lower", 0.0), axis.get("upper", 0.0), value)
                    axis["num_bins"] = num_bins
                else:
                    if axis.get(key) == value and not bool(
                        axis.get(f"auto_{key}", False)
                    ):
                        return
                    axis[key] = value
                    if key in {"lower", "upper"}:
                        axis[f"auto_{key}"] = False
                    if (
                        key in {"lower", "upper"}
                        and _rebin_axis_mode(config, axis) == "step"
                        and previous_step > 0.0
                        and np.isfinite(previous_step)
                    ):
                        axis["num_bins"] = _num_bins_from_step_size(
                            axis.get("lower", 0.0), axis.get("upper", 0.0), previous_step
                        )
                    elif key in {"lower", "upper"}:
                        axis["step_size"] = _step_size_from_bounds(
                            axis.get("lower", 0.0), axis.get("upper", 0.0), axis["num_bins"]
                        )
        except ValueError:
            return
        axis.update(_sanitize_rebin_axis_config(axis))
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_axis_edges(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        try:
            parsed = _parse_parameter_text(text)
            if parsed == "":
                axes[index].pop("bin_edges", None)
            else:
                axes[index].update(
                    _sanitize_rebin_axis_config(
                        {**axes[index], "bin_edges": _normalize_rebin_bin_edges(parsed)}
                    )
                )
        except (TypeError, ValueError):
            self._set_dataset_details_preserving_scroll(dataset, group)
            return
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_axis_mode(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        value: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        mode = value if value in REBIN_AXIS_MODES else "step"
        if _rebin_axis_mode(config, axes[index]) == mode:
            return
        axes[index]["mode"] = mode
        if mode in {"discrete", "tolerance"}:
            axes[index]["fractional"] = False
        if mode == "tolerance":
            axes[index].setdefault(
                "tolerance", max(float(axes[index].get("step_size", 0.1)), 1e-12)
            )
        self._after_dataset_rebin_changed(dataset, group)
        self._set_dataset_details_preserving_scroll(dataset, group)

    def _set_dataset_rebin_axis_fractional(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        fractional: bool,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        value = bool(fractional) and _rebin_axis_mode(config, axes[index]) not in {
            "discrete",
            "tolerance",
        }
        if _rebin_axis_fractional(config, axes[index]) == value:
            return
        axes[index]["fractional"] = value
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_resolution_mode(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        mode: str,
    ) -> None:
        mode = "bins" if mode == "bins" else "step"
        config = dataset_rebin_config(dataset)
        if _rebin_resolution_mode(config) == mode:
            return
        config[REBIN_RESOLUTION_MODE_KEY] = mode
        if mode == "bins":
            for axis in config.get("axes", []):
                axis["step_size"] = _step_size_from_bounds(
                    axis.get("lower", 0.0), axis.get("upper", 0.0), max(int(axis.get("num_bins", 1)), 1)
                )
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_axis_vector(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        index: int,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        axes = config.get("axes", [])
        if not (0 <= index < len(axes)):
            return
        parsed = _parse_parameter_text(text)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            self._set_dataset_details_preserving_scroll(dataset, group)  # revert the editor
            return
        try:
            current = _rebin_axis_vector(axes[index], index, 4)[:3]
            vector = [_clean_axis_weight(component) for component in parsed]
            if np.array_equal(current, np.asarray(vector, dtype=float)):
                return
            _update_rebin_momentum_basis(
                axes,
                index,
                vector,
                data=(
                    dataset.data
                    if isinstance(dataset.data, (MDHistoData, PointData4D))
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin momentum coordinates",
                f"Could not change the momentum-coordinate matrix:\n{exc}",
            )
            self._set_dataset_details_preserving_scroll(dataset, group)
            return
        self._after_dataset_rebin_changed(dataset, group)

    def _set_dataset_rebin_momentum_matrix(
        self,
        dataset: DatasetEntry,
        group: DataGroup | None,
        text: str,
    ) -> None:
        config = dataset_rebin_config(dataset)
        try:
            matrix = _parse_parameter_text(text)
            _update_rebin_momentum_matrix(
                config.get("axes", []),
                matrix,
                data=(
                    dataset.data
                    if isinstance(dataset.data, (MDHistoData, PointData4D))
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                self.window,
                "Rebin momentum coordinates",
                f"Could not change the momentum-coordinate matrix:\n{exc}",
            )
            self._set_dataset_details_preserving_scroll(dataset, group)
            return
        self._after_dataset_rebin_changed(dataset, group)

    def _after_dataset_rebin_changed(self, dataset: DatasetEntry, group: DataGroup | None) -> None:
        scrollbar = self.details_scroll.verticalScrollBar()
        scroll_position = int(scrollbar.value())
        focused = self.window.focusWidget()
        focus_object_name = (
            str(focused.objectName() or "")
            if focused is not None and self.details_widget.isAncestorOf(focused)
            else None
        )
        config = dataset_rebin_config(dataset)
        config["stale"] = True
        if (
            bool(config.get("auto_rebin", True))
            and not bool(config.get("auto_rebin_user_override", False))
            and _dataset_rebin_is_large(dataset, config)
        ):
            config["auto_rebin"] = False
        selected_id = self._selected_dataset_binning(dataset)["id"]
        resolved = _derived_analysis_for_dataset(dataset)
        if resolved is not None and selected_id == dataset_rebin_binnings(dataset)[0]["id"]:
            _owner, analysis = resolved
            analysis.metadata["output_rebin_config"] = copy.deepcopy(config)
        if group is not None:
            self._record_data_group_state_change(group)
        self._mark_dirty()
        if group is not None and bool(config.get("auto_rebin", True)):
            self.refresh_slice_viewer(group)
        self._refresh_cache_badges()
        if not self._refresh_dataset_rebin_controls(dataset, group):
            self._set_dataset_details_preserving_scroll(
                dataset,
                group,
                scroll_position=scroll_position,
                focus_object_name=focus_object_name,
            )

    def _refresh_dataset_rebin_controls(
        self, dataset: DatasetEntry, group: DataGroup | None
    ) -> bool:
        """Update rebin controls in place so ordinary edits keep their scroll/focus."""

        from PySide6 import QtWidgets

        controls = self.details_widget.findChild(QtWidgets.QWidget, "dataset_rebin_controls")
        if controls is None:
            return False
        config = dataset_rebin_config(dataset)
        controls.setVisible(bool(config.get("enabled", False)))
        auto_check = self.details_widget.findChild(
            QtWidgets.QCheckBox, "dataset_rebin_auto"
        )
        if auto_check is not None:
            auto_check.blockSignals(True)
            try:
                auto_check.setChecked(bool(config.get("auto_rebin", True)))
            finally:
                auto_check.blockSignals(False)
        for index, axis_config in enumerate(config.get("axes", [])):
            axis = _sanitize_rebin_axis_config(axis_config)
            axis_mode = _rebin_axis_mode(config, axis)
            mode_combo = self.details_widget.findChild(
                QtWidgets.QComboBox, f"dataset_rebin_axis_mode_{index}"
            )
            if mode_combo is not None:
                mode_combo.blockSignals(True)
                try:
                    mode_combo.setCurrentIndex(max(mode_combo.findData(axis_mode), 0))
                finally:
                    mode_combo.blockSignals(False)
            assignment_combo = self.details_widget.findChild(
                QtWidgets.QComboBox, f"dataset_rebin_axis_assignment_{index}"
            )
            if assignment_combo is not None:
                assignment_combo.blockSignals(True)
                try:
                    assignment_combo.setCurrentIndex(
                        max(
                            assignment_combo.findData(
                                _rebin_axis_fractional(config, axis)
                            ),
                            0,
                        )
                    )
                    assignment_combo.setEnabled(
                        axis_mode not in {"discrete", "tolerance"}
                    )
                finally:
                    assignment_combo.blockSignals(False)
            resolution_key = (
                "num_bins"
                if axis_mode == "bins"
                else "tolerance"
                if axis_mode == "tolerance"
                else "step_size"
            )
            values = {
                f"dataset_rebin_axis_lower_{index}": (
                    "" if _rebin_axis_bound_is_auto(axis, "lower") else _format_number(axis["lower"])
                ),
                f"dataset_rebin_axis_upper_{index}": (
                    "" if _rebin_axis_bound_is_auto(axis, "upper") else _format_number(axis["upper"])
                ),
                f"dataset_rebin_resolution_value_{index}": (
                    ""
                    if axis_mode in {"discrete", "edges"}
                    else
                    str(int(axis["num_bins"]))
                    if resolution_key == "num_bins"
                    else _format_number(axis[resolution_key])
                ),
                f"dataset_rebin_axis_vector_{index}": _momentum_rebin_vector_text(
                    axis, index
                ),
                f"dataset_rebin_axis_edges_{index}": _parameter_to_text(
                    axis.get("bin_edges", "")
                ),
            }
            for object_name, text in values.items():
                editor = self.details_widget.findChild(QtWidgets.QLineEdit, object_name)
                if editor is None:
                    continue
                editor.blockSignals(True)
                try:
                    editor.setText(text)
                finally:
                    editor.blockSignals(False)
        matrix_editor = self.details_widget.findChild(
            QtWidgets.QLineEdit, "dataset_rebin_momentum_matrix"
        )
        if matrix_editor is not None:
            matrix_editor.setText(
                _parameter_to_text(_momentum_rebin_matrix(config.get("axes", [])))
            )
        status = self.details_widget.findChild(QtWidgets.QLabel, "dataset_rebin_status")
        if status is not None:
            status.setText(_dataset_rebin_status_text(dataset, config))
        selected_binning = self._selected_dataset_binning(dataset)
        compressed_disk_bytes = (
            _saved_binning_compressed_size(
                self.project,
                kind="dataset",
                group=group,
                target=dataset,
                binning_id=selected_binning["id"],
                config=config,
            )
            if group is not None
            else None
        )
        memory = self.details_widget.findChild(
            QtWidgets.QLabel, "dataset_rebin_memory_estimate"
        )
        if memory is not None:
            from .project_rebin_panels import rebin_memory_estimate_text

            memory.setText(
                rebin_memory_estimate_text(
                    config,
                    data=_peek_cached_dataset_view(
                        dataset,
                        extra_masks=(
                            effective_dataset_masks(group, dataset)
                            if group is not None
                            else None
                        ),
                        rebin_config=config,
                        cache_id=(
                            None
                            if selected_binning["fit"]
                            else selected_binning["id"]
                        ),
                        resident_only=True,
                    ),
                    compressed_disk_bytes=compressed_disk_bytes,
                )
            )
        symmetry = symmetry_spec_from_config(config.get("symmetry"))
        symmetry_mode = self.details_widget.findChild(
            QtWidgets.QComboBox, "dataset_rebin_symmetry_mode"
        )
        if symmetry_mode is not None:
            displayed_mode = (
                symmetry.mode
                if symmetry.mode != "none"
                else str(config.get("symmetry", {}).get("last_mode", "space_group"))
            )
            symmetry_mode.blockSignals(True)
            try:
                symmetry_mode.setCurrentIndex(
                    max(symmetry_mode.findData(displayed_mode), 0)
                )
            finally:
                symmetry_mode.blockSignals(False)
        symmetry_expression = self.details_widget.findChild(
            QtWidgets.QLineEdit, "dataset_rebin_symmetry_expression"
        )
        if symmetry_expression is not None:
            symmetry_expression.setText(symmetry.expression)
        symmetry_preview = self.details_widget.findChild(
            QtWidgets.QLabel, "dataset_rebin_symmetry_preview"
        )
        if symmetry_preview is not None:
            symmetry_preview.setText(
                self._dataset_rebin_symmetry_preview(dataset, group)
            )
        if isinstance(controls, QtWidgets.QTabWidget):
            from .project_rebin_panels import rebin_bin_information_widget

            info_index = next(
                (
                    index
                    for index in range(controls.count())
                    if controls.tabText(index) == "Bin information"
                ),
                -1,
            )
            if info_index >= 0:
                selected = controls.currentIndex()
                old_widget = controls.widget(info_index)
                controls.removeTab(info_index)
                old_widget.deleteLater()
                controls.insertTab(
                    info_index,
                    rebin_bin_information_widget(
                        config,
                        data=_peek_cached_dataset_view(
                            dataset,
                            extra_masks=(
                                effective_dataset_masks(group, dataset)
                                if group is not None
                                else None
                            ),
                            rebin_config=config,
                            cache_id=(
                                None
                                if selected_binning["fit"]
                                else selected_binning["id"]
                            ),
                            resident_only=True,
                        ),
                        object_prefix="dataset_rebin",
                        compressed_disk_bytes=compressed_disk_bytes,
                    ),
                    "Bin information",
                )
                controls.setCurrentIndex(selected)
        return True

    def _after_group_composite_changed(self, group: DataGroup | _CompositeScope) -> None:
        config = data_group_composite_config(group)
        config["stale"] = True
        if (
            bool(config.get("auto_rebin", True))
            and not bool(config.get("auto_rebin_user_override", False))
            and _composite_rebin_is_large(group, config)
        ):
            config["auto_rebin"] = False
        root = _composite_root(group)
        self._record_data_group_state_change(root)
        self._mark_dirty()
        if bool(config.get("auto_rebin", True)):
            self.refresh_slice_viewer(root)
            self._request_overlay_refresh(root)
        self._refresh_cache_badges()
        self._sync_details()

    def _clear_details_panel(self) -> None:
        # The mask editor lives inside this panel. Clear its status-label
        # reference before Qt deletes the old details widgets.
        self.mask_application_status_label = None
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if (
                    self.sample_environment_widget is not None
                    and widget.isAncestorOf(self.sample_environment_widget)
                ):
                    self.sample_environment_widget.setParent(None)
                if widget is self.fit_settings_panel or widget is self.sample_environment_widget:
                    widget.setParent(None)
                else:
                    widget.setParent(None)
                    widget.deleteLater()

    def _restore_selected_fit_state(self, group: DataGroup, fit_entry: FitTimelineEntry) -> None:
        if not fit_entry.snapshot:
            fit_entry.snapshot = snapshot_data_group_state(group)
        self._set_active_fit_state(group, fit_entry)
        self._restoring_fit_selection = True
        try:
            restore_data_group_state(group, fit_entry.snapshot)
            if _posterior_display_options(fit_entry).get("use_best_sample"):
                _apply_displayed_fit_parameters(group, fit_entry)
            # _refresh_tree already refreshes every open slice viewer for the
            # restored state; an explicit refresh here would recompute the
            # (expensive) overlay a second time.
            self._refresh_tree(select_group=group, select_fit=fit_entry)
        finally:
            self._restoring_fit_selection = False
        self._mark_dirty()

    def _active_fit_entry(self, group: DataGroup) -> FitTimelineEntry | None:
        if self._active_fit_group is not group:
            return None
        for entry in (
            self._active_branch_current,
            self._active_fit_current,
            self._active_fit_anchor,
        ):
            if _fit_entry_in_tree(group.fits, entry):
                return entry
        return None

    def _set_active_fit_state(self, group: DataGroup, fit_entry: FitTimelineEntry | None) -> None:
        self._active_fit_group = group
        self._active_fit_anchor = (
            fit_entry if fit_entry is not None and fit_entry.kind in {"initial", "result"} else None
        )
        self._active_fit_current = (
            fit_entry if fit_entry is not None and fit_entry.kind == "current" else None
        )
        self._active_branch_current = None
        # Remember which fit entry is active so it can be persisted and restored.
        group.active_fit_path = _fit_entry_path(group.fits, fit_entry)

    def _record_data_group_state_change(self, group: DataGroup) -> bool:
        """Update fit-history current state, creating an edit branch when needed.

        Returns True when the fit tree or active fit entry changed and callers
        should refresh the tree.
        """

        if self._restoring_fit_selection:
            return False
        ensure_fit_history(group)
        tree_changed = False
        current_entry: FitTimelineEntry | None = None
        if self._active_fit_group is group:
            if _fit_entry_in_tree(group.fits, self._active_fit_current):
                current_entry = self._active_fit_current
            elif _fit_entry_in_tree(group.fits, self._active_fit_anchor):
                if _is_editable_initial_baseline(group, self._active_fit_anchor):
                    _set_fit_current_snapshot(self._active_fit_anchor, group)
                    return False
                siblings = _fit_siblings(group.fits, self._active_fit_anchor)
                if (
                    self._active_fit_anchor.kind == "result"
                    and siblings is not None
                    and self._active_fit_anchor is _last_result_at_level(siblings)
                ):
                    current_entry, _created = _ensure_current_state_after_result(
                        group, self._active_fit_anchor
                    )
                    self._active_fit_current = current_entry
                    self._active_branch_current = None
                    tree_changed = True
                    if current_entry is not None:
                        group.active_fit_path = _fit_entry_path(group.fits, current_entry)
                elif self._active_fit_anchor.kind == "result":
                    if not _fit_entry_in_tree(group.fits, self._active_branch_current):
                        timeline = FitTimelineEntry(
                            name=next_fit_timeline_name(group.fits),
                            kind="timeline",
                            created_at=_timestamp_now(),
                            metadata={
                                "branch_reason": "dataset/model state edited after restoring a fit",
                                "branched_from": self._active_fit_anchor.name,
                            },
                        )
                        current_entry = current_state_fit_entry(group, self._active_fit_anchor)
                        timeline.children.append(current_entry)
                        self._active_fit_anchor.children.append(timeline)
                        self._active_branch_current = current_entry
                        self._active_fit_current = None
                        tree_changed = True
                        group.active_fit_path = _fit_entry_path(group.fits, current_entry)
                    else:
                        current_entry = self._active_branch_current
            else:
                self._active_fit_group = None
                self._active_fit_anchor = None
                self._active_fit_current = None
                self._active_branch_current = None
        if current_entry is None:
            current_entry = _top_level_current_state_entry(group)
        if current_entry is not None:
            _set_fit_current_snapshot(current_entry, group)
        return tree_changed

    def _evaluate_model_after_dataset_activation(
        self,
        group: DataGroup,
    ) -> FitTimelineEntry:
        """Record a live model evaluation after importing or enabling data."""

        current = self._active_fit_entry(group)
        if any(dataset.enabled and dataset.data is None for dataset in group.iter_datasets()):
            current = defer_current_state_model_evaluation(group, current)
        else:
            current = evaluate_current_state_model(group, current)
        self._set_active_fit_state(group, current)
        self._request_overlay_refresh(group)
        return current

    def _clear_active_fit_state(self) -> None:
        self._active_fit_group = None
        self._active_fit_anchor = None
        self._active_fit_current = None
        self._active_branch_current = None

    def _restore_active_fit_selection(self) -> bool:
        """Select the persisted fit entry without replacing live project state.

        The project manifest already contains the authoritative live workspace.
        Fit snapshots are applied only after an explicit user selection, never
        as a side effect of opening or reloading a project.
        """

        for group in self.project.data_groups:
            entry = _fit_entry_at_path(group.fits, group.active_fit_path)
            if entry is None:
                continue
            self._set_active_fit_state(group, entry)
            self._restoring_fit_selection = True
            try:
                self._refresh_tree(select_group=group, select_fit=entry)
            finally:
                self._restoring_fit_selection = False
            return True
        return False

    def _sync_fit_editor(self, fit_entry: FitTimelineEntry) -> None:
        self.fit_optimizer_combo.blockSignals(True)
        self.fit_optimizer_combo.setCurrentText(fit_entry.optimizer or "least_squares")
        self.fit_optimizer_combo.blockSignals(False)
        config = dict(fit_entry.optimizer_config) if isinstance(fit_entry.optimizer_config, dict) else {}
        self._sync_fit_control_values(config)
        self.fit_optimizer_config_editor.blockSignals(True)
        self.fit_optimizer_config_editor.setText(json.dumps(config, sort_keys=True))
        self.fit_optimizer_config_editor.blockSignals(False)

    def _sync_fit_control_values(self, config: dict[str, Any]) -> None:
        widgets = [
            self.fit_loss_combo,
            self.fit_covariance_mode_combo,
            self.fit_f_scale_spin,
            self.fit_finite_difference_workers_spin,
            self.fit_de_check,
            self.fit_de_maxiter_spin,
            self.fit_de_popsize_spin,
            self.fit_de_workers_spin,
            self.fit_emcee_check,
            self.fit_emcee_walkers_spin,
            self.fit_emcee_steps_spin,
            self.fit_emcee_burn_spin,
            self.fit_emcee_thin_spin,
            self.fit_emcee_seed_spin,
            self.fit_emcee_workers_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.fit_loss_combo.setCurrentText(str(config.get("loss", "linear")))
        covariance_index = self.fit_covariance_mode_combo.findData(
            str(config.get("covariance_mode", "absolute"))
        )
        self.fit_covariance_mode_combo.setCurrentIndex(max(covariance_index, 0))
        self.fit_f_scale_spin.setValue(float(config.get("f_scale", 1.0) or 1.0))
        self.fit_finite_difference_workers_spin.setValue(
            int(config.get("finite_difference_workers", -1))
        )
        initialization = config.get("initialization") if isinstance(config.get("initialization"), dict) else {}
        self.fit_de_check.setChecked(bool(initialization.get("enabled", False)))
        self.fit_de_maxiter_spin.setValue(int(initialization.get("maxiter", 60) or 60))
        self.fit_de_popsize_spin.setValue(int(initialization.get("popsize", 10) or 10))
        self.fit_de_workers_spin.setValue(int(initialization.get("workers", 1) or 1))
        sampler = config.get("sampler") if isinstance(config.get("sampler"), dict) else {}
        self.fit_emcee_check.setChecked(bool(sampler.get("enabled", False)))
        self.fit_emcee_walkers_spin.setValue(int(sampler.get("n_walkers", 0) or 0))
        self.fit_emcee_steps_spin.setValue(int(sampler.get("n_steps", 1000) or 1000))
        self.fit_emcee_burn_spin.setValue(int(sampler.get("burn_in", 200) or 0))
        self.fit_emcee_thin_spin.setValue(int(sampler.get("thin", 1) or 1))
        seed_value = sampler.get("random_seed", -1)
        self.fit_emcee_seed_spin.setValue(-1 if seed_value in (None, "") else int(seed_value))
        workers_value = sampler.get("workers", -1)
        self.fit_emcee_workers_spin.setValue(-1 if workers_value is None else int(workers_value))
        for widget in widgets:
            widget.blockSignals(False)

    def _set_selected_fit_optimizer(self, optimizer: str) -> None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        if fit_entry.optimizer != str(optimizer):
            fit_entry.optimizer = str(optimizer)
            self._mark_dirty()

    def _set_selected_fit_optimizer_config(self) -> None:
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        try:
            config = json.loads(self.fit_optimizer_config_editor.text() or "{}")
        except json.JSONDecodeError:
            config = {}
            self.fit_optimizer_config_editor.setText("{}")
        if not isinstance(config, dict):
            config = {}
            self.fit_optimizer_config_editor.setText("{}")
        if fit_entry.optimizer_config != config:
            fit_entry.optimizer_config = config
            self._sync_fit_control_values(config)
            self._mark_dirty()

    def _set_selected_fit_controls_config(self, *args: Any) -> None:
        del args
        fit_entry = self._fit_entry_for_item(self._current_item())
        if fit_entry is None:
            return
        config = self._fit_config_from_controls(fit_entry.optimizer_config)
        if fit_entry.optimizer_config != config:
            fit_entry.optimizer_config = config
            self.fit_optimizer_config_editor.blockSignals(True)
            self.fit_optimizer_config_editor.setText(json.dumps(config, sort_keys=True))
            self.fit_optimizer_config_editor.blockSignals(False)
            self._mark_dirty()

    def _fit_config_from_controls(self, existing: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(existing or {})
        existing_initialization = config.get("initialization")
        keep_deferred_updates = False
        if isinstance(existing_initialization, dict):
            keep_deferred_updates = (
                str(existing_initialization.get("updating", "")).lower() == "deferred"
            )
            if not keep_deferred_updates:
                try:
                    keep_deferred_updates = int(existing_initialization.get("workers", 1) or 1) != 1
                except (TypeError, ValueError):
                    pass
        loss = str(self.fit_loss_combo.currentText() or "linear")
        if loss == "linear":
            config.pop("loss", None)
            config.pop("f_scale", None)
        else:
            config["loss"] = loss
            config["f_scale"] = float(self.fit_f_scale_spin.value())
        covariance_mode = str(
            self.fit_covariance_mode_combo.currentData() or "absolute"
        )
        if covariance_mode == "absolute":
            config.pop("covariance_mode", None)
        else:
            config["covariance_mode"] = covariance_mode
        # Per-fit worker counts are retained when loading old recipes but are
        # no longer authored by the GUI. The central Preferences CPU limit is
        # authoritative for GUI execution.
        config.pop("finite_difference_workers", None)
        if self.fit_de_check.isChecked():
            config["initialization"] = {
                "enabled": True,
                "method": "differential_evolution",
                "maxiter": int(self.fit_de_maxiter_spin.value()),
                "popsize": int(self.fit_de_popsize_spin.value()),
            }
            if keep_deferred_updates:
                config["initialization"]["updating"] = "deferred"
        else:
            config.pop("initialization", None)
        if self.fit_emcee_check.isChecked():
            sampler = {
                "enabled": True,
                "method": "emcee",
                "n_steps": int(self.fit_emcee_steps_spin.value()),
                "burn_in": int(self.fit_emcee_burn_spin.value()),
                "thin": int(self.fit_emcee_thin_spin.value()),
            }
            if self.fit_emcee_seed_spin.value() >= 0:
                sampler["random_seed"] = int(self.fit_emcee_seed_spin.value())
            walkers = int(self.fit_emcee_walkers_spin.value())
            if walkers > 0:
                sampler["n_walkers"] = walkers
            config["sampler"] = sampler
        else:
            config.pop("sampler", None)
        return config

    def context_menu_action_names(self, item: Any | None) -> list[str]:
        """Return context-menu action names for a tree item."""

        return [name for name, _enabled in self._context_menu_action_specs(item)]

    def _context_menu_action_specs(self, item: Any | None) -> list[tuple[str, bool]]:
        _group, entry, mask, model, role = self._objects_for_item(item)
        can_paste = self._can_paste_into_role(role, entry)
        has_source = bool(entry is not None and _dataset_source_path(entry) is not None)
        enabled_state = _enabled_state_for_role(role, entry, mask, model)
        specs: list[tuple[str, bool]] = []
        if role in {"dataset", "dataset_group", "mask"}:
            specs.append(("Copy", True))
        if role in {"group", "datasets", "dataset", "dataset_group", "masks"}:
            specs.append(("Paste", can_paste))
        if enabled_state is not None:
            specs.append(("Disable" if enabled_state else "Enable", True))
        if role in {"group", "dataset", "mask", "background", "group_background", "model", "fit", "fit_timeline", "dataset_group", "group_mask", "plot"}:
            specs.append(("Rename", True))
        if role in _DELETABLE_TREE_ROLES:
            specs.append(("Delete", True))
        if role in {"group", "datasets", "dataset", "analyses", "analysis", "analysis_output"}:
            specs.append(("Open Analysis Window", True))
        if role == "analyses":
            specs.append(("New analysis", True))
        if role == "analysis":
            specs.append(("Copy workflow script", True))
            specs.append(("Save workflow script...", True))
        if role in {"group", "datasets", "dataset", "masks", "mask", "backgrounds", "background", "group_backgrounds", "group_background", "dataset_group", "group_masks", "group_mask", "analysis_output"}:
            output = self._analysis_output_roles.get(id(item)) if role == "analysis_output" else None
            specs.append(("View in data viewer", role != "analysis_output" or bool(output and output.dataset_id)))
        if role == "dataset":
            specs.append(("Reload data", _dataset_can_reload(entry)))
            specs.append(("Show file location", has_source))
            specs.append(("Change file source", True))
            specs.append(("Copy workflow script", True))
            specs.append(("Save workflow script...", True))
        if role in {"dataset", "masks", "group_masks", "dataset_group"}:
            specs.append(("Add mask", True))
        if role in {"datasets", "dataset", "backgrounds", "dataset_group", "group_backgrounds"}:
            specs.append(("Add background", True))
        if role in {"group", "datasets", "dataset_group"}:
            node = self._dataset_group_for_item(item) if role == "dataset_group" else _group
            specs.append((
                "Reload data",
                bool(node is not None and any(_dataset_can_reload(dataset) for dataset in node.iter_datasets())),
            ))
            specs.append(("Add dataset", True))
            specs.append(("New dataset group", True))
        if role in {"group", "models"}:
            specs.append(("Add model", True))
        if role == "fit":
            specs.append(("Fit now", True))
            specs.append(("Copy workflow script", True))
            specs.append(("Save workflow script...", True))
            fit_entry = self._fit_entry_for_item(item)
            specs.append(
                ("Export report...", fit_entry is not None and fit_entry.kind == "result")
            )
            specs.append(("Create covariance plot", fit_entry is not None and _covariance_matrix_from_fit_entry(fit_entry) is not None))
        if role == "plot":
            specs.extend([("Open plot", True), ("Edit in data viewer", True)])
        return specs

    def _show_context_menu(self, item: Any | None, global_pos: Any) -> None:
        from PySide6 import QtCore, QtWidgets

        if item is None:
            return
        if item in self.tree.selectedItems():
            self.tree.selectionModel().setCurrentIndex(
                self.tree.indexFromItem(item),
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        else:
            self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self.tree)
        menu.setToolTipsVisible(True)
        actions = {
            "Open Analysis Window": self.open_data_playground_for_selection,
            "New analysis": self.new_analysis_for_selection,
            "Copy": self.copy_selected,
            "Paste": self.paste_into_selection,
            "Rename": self.rename_selected,
            "Delete": self.delete_selected,
            "View in data viewer": self.open_slice_viewer_for_selection,
            "Reload data": self.reload_data_for_selection,
            "Show file location": self.show_file_location_for_selection,
            "Change file source": self.change_file_source_for_selection,
            "Copy workflow script": self.copy_workflow_script_for_selection,
            "Save workflow script...": self.save_workflow_script_for_selection,
            "Add dataset": self.add_dataset_to_selection,
            "Add mask": self.add_mask_to_selection,
            "Add background": self.add_background_to_selection,
            "New dataset group": self.add_dataset_group_to_selection,
            "Add model": self.add_model_to_selection,
            "Fit now": self.start_fit_for_selection,
            "Export report...": self.export_fit_report_for_selection,
            "Create covariance plot": self.create_fit_covariance_plot_for_selection,
            "Open plot": self.open_saved_plot_for_selection,
            "Edit in data viewer": self.edit_saved_plot_in_viewer,
            "Enable": lambda: self._set_selected_enabled(True),
            "Disable": lambda: self._set_selected_enabled(False),
        }
        tooltips = {
            "Open Analysis Window": "Create, configure, run, and inspect non-fitting dataset analyses.",
            "New analysis": "Open the Analysis Window with a fresh analysis recipe for this workspace.",
            "Copy": "Copy the selected dataset batch, dataset group, or mask so it can be pasted elsewhere in the project.",
            "Paste": "Paste the copied datasets, dataset group, or mask into the selected compatible destination.",
            "Rename": "Rename the selected tree item.",
            "Delete": "Delete all selected compatible items when that operation is allowed.",
            "View in data viewer": "Open or refresh the data viewer for this selection.",
            "Reload data": (
                "Reread the selected dataset, or every dataset in the selected "
                "collection, from its configured source."
            ),
            "Show file location": "Reveal the selected dataset's source file in the operating system file browser.",
            "Change file source": "Point this dataset at a different source file on disk.",
            "Copy workflow script": (
                "Copy readable Python that reloads and prepares this dataset "
                "from its source files."
            ),
            "Save workflow script...": (
                "Save readable Python that reloads and prepares this dataset "
                "from its source files."
            ),
            "Add dataset": "Choose data files to import into this dataset collection.",
            "Add mask": "Create a new mask for the selected dataset or shared mask folder.",
            "Add background": (
                "Attach a scaled powder |Q|-energy background to the selected "
                "dataset or composite dataset group."
            ),
            "New dataset group": "Create a nested dataset group under the selected workspace or group.",
            "Add model": "Create a new model component in the selected workspace.",
            "Fit now": "Run the optimizer from the selected fit state and store a new fit result.",
            "Export report...": (
                "Export a publication-grade LaTeX or PDF report of this fit "
                "result: per-dataset statistics, the model Hamiltonian term by "
                "term, fitted parameters, and physics diagnostics."
            ),
            "Create covariance plot": "Save this fit's covariance or correlation matrix as an editable plot recipe.",
            "Open plot": "Open this saved plot in the clean presentation window.",
            "Edit in data viewer": "Reopen this saved plot in the data viewer so every viewer control can be adjusted.",
            "Enable": "Enable all selected compatible items for viewing and fitting.",
            "Disable": "Disable all selected compatible items for viewing and fitting.",
        }
        for name, enabled in self._context_menu_action_specs(item):
            action = menu.addAction(name)
            action.setEnabled(enabled)
            action.setToolTip(tooltips.get(name, name))
            action.setStatusTip(tooltips.get(name, name))
            action.triggered.connect(actions[name])
        if not menu.isEmpty():
            menu.exec(global_pos)

    def add_dataset_to_selection(self) -> None:
        """Open the file importer for the selected workspace or dataset folder."""

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if group is not None and role in {"group", "datasets", "dataset_group"}:
            self._add_dataset_import_files(group)

    def open_data_playground_for_selection(self) -> Any | None:
        item = self._current_item()
        group, entry, _mask, _model, role = self._objects_for_item(item)
        if group is None:
            return None
        from .analysis_gui import DataPlaygroundWindow

        if self._analysis_window is None:
            self._analysis_window = DataPlaygroundWindow(self)
        self._analysis_window.select_group(group, dataset=entry)
        selected_analysis = self._analysis_item_roles.get(id(item)) if role == "analysis" else None
        if selected_analysis is None and role == "analysis_output" and item.parent() is not None:
            selected_analysis = self._analysis_item_roles.get(id(item.parent()))
        if selected_analysis is not None:
            index = self._analysis_window.analysis_combo.findData(selected_analysis.id)
            if index >= 0:
                self._analysis_window.analysis_combo.setCurrentIndex(index)
                self._analysis_window._select_analysis()
        self._analysis_window.show()
        self._analysis_window.raise_()
        return self._analysis_window

    def new_analysis_for_selection(self) -> Any | None:
        """Open a fresh Analysis Window recipe for the selected Analyses branch."""

        group, _entry, _mask, _model, role = self._objects_for_item(self._current_item())
        if role != "analyses" or group is None:
            return None
        playground = self.open_data_playground_for_selection()
        if playground is not None:
            playground.new_analysis()
        return playground

    def _analysis_display_status(self, group: DataGroup, analysis: AnalysisEntry) -> str:
        if analysis.result is None:
            return "never run"
        if analysis.result.status != "success":
            return analysis.result.status
        try:
            definition = analysis_definition(analysis.type)
            datasets = {dataset.id: dataset for dataset in group.iter_datasets()}
            selected = [datasets[dataset_id] for dataset_id in analysis.input_dataset_ids]
            parameters = {**default_analysis_parameters(analysis.type), **analysis.parameters}
            current_recipe = recipe_hash(analysis.type, definition.version, parameters, analysis.input_dataset_ids)
            fingerprints = {dataset.id: dataset_entry_fingerprint(dataset, group) for dataset in selected}
        except (KeyError, TypeError, ValueError, OSError):
            return "unavailable"
        if self.project_path is not None:
            for output in analysis.result.outputs:
                if output.artifact_path and not project_artifact_exists(
                    self.project_path,
                    output.artifact_path,
                ):
                    return "unavailable artifact"
        if current_recipe != analysis.result.recipe_hash or fingerprints != analysis.result.input_fingerprints:
            return "stale"
        return "fresh"

    def _can_paste_into_role(self, role: str, entry: DatasetEntry | None) -> bool:
        if self._clipboard is None:
            return False
        clip_role, payload = self._clipboard
        if clip_role == "dataset" and isinstance(payload, DatasetEntry):
            return role in {"group", "datasets", "dataset_group"}
        if clip_role == "datasets" and isinstance(payload, list):
            return role in {"group", "datasets", "dataset_group"} and all(
                isinstance(item, DatasetEntry) for item in payload
            )
        if clip_role == "dataset_group" and isinstance(payload, DatasetGroup):
            return role in {"group", "datasets", "dataset_group"}
        if clip_role == "mask" and isinstance(payload, MaskSpec):
            return role in {"dataset", "masks"} and entry is not None
        return False

    def _sync_window_title(self) -> None:
        suffix = "Untitled" if self.project_path is None else str(self.project_path)
        size = (
            ""
            if self._saved_project_size_bytes is None
            else f" ({_format_project_file_size(self._saved_project_size_bytes)})"
        )
        marker = " *" if self.has_unsaved_changes else ""
        self.window.setWindowTitle(f"nfit Project Explorer - {suffix}{size}{marker}")
        if self.reload_project_action is not None:
            self.reload_project_action.setEnabled(self.project_path is not None)

    def _mark_dirty(self) -> None:
        self.has_unsaved_changes = True
        self._sync_window_title()

    def _confirm_save_before_closing_project(self) -> bool:
        from PySide6 import QtWidgets

        if not self.has_unsaved_changes:
            return True
        message = QtWidgets.QMessageBox(self.window)
        message.setIcon(QtWidgets.QMessageBox.Icon.Question)
        message.setWindowTitle("Unsaved nfit project")
        message.setText("Save changes before closing this project?")
        save_button = message.addButton(
            QtWidgets.QMessageBox.StandardButton.Save
        )
        cancel_button = message.addButton(
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        close_without_saving_button = message.addButton(
            "Close without saving",
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
        )
        message.setDefaultButton(save_button)
        message.exec()
        clicked_button = message.clickedButton()
        if clicked_button is cancel_button:
            return False
        if clicked_button is save_button:
            return self.save()
        return clicked_button is close_without_saving_button

    def _handle_window_close(self, event: Any) -> None:
        if self._allow_window_close or self._confirm_save_before_closing_project():
            self._close_all_slice_viewers()
            event.accept()
            return
        event.ignore()

    def _remember_recent_project(self, path: str | Path) -> None:
        remember_recent_project(path)
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        if self.recent_projects_menu is None:
            return
        self.recent_projects_menu.clear()
        recent = recent_project_paths()
        if not recent:
            empty_action = self.recent_projects_menu.addAction("No recent projects")
            empty_action.setEnabled(False)
            empty_action.setToolTip("No recent nfit project files are available.")
            empty_action.setStatusTip("No recent nfit project files are available.")
            return
        for path in recent:
            action = self.recent_projects_menu.addAction(str(path))
            action.setEnabled(path.exists())
            tip = f"Open recent project: {path}" if path.exists() else f"Recent project file is missing: {path}"
            action.setToolTip(tip)
            action.setStatusTip(tip)
            action.triggered.connect(lambda _checked=False, path=path: self.open_project_path(path))
        if any(not path.exists() for path in recent):
            self.recent_projects_menu.addSeparator()
            action = self.recent_projects_menu.addAction("Remove missing projects", self._remove_missing_recent_projects)
            action.setToolTip("Remove recent-project entries whose files no longer exist.")
            action.setStatusTip("Remove recent-project entries whose files no longer exist.")

    def _remove_missing_recent_projects(self) -> None:
        forget_missing_recent_projects()
        self._refresh_recent_projects_menu()

    def _refresh_mask_type_combo(self) -> None:
        current = self.mask_type_combo.currentData()
        self.mask_type_combo.blockSignals(True)
        self.mask_type_combo.clear()
        for type_name in available_mask_types():
            self.mask_type_combo.addItem(MASK_TYPE_DEFINITIONS[type_name]["label"], type_name)
        if current is not None:
            index = self.mask_type_combo.findData(current)
            if index >= 0:
                self.mask_type_combo.setCurrentIndex(index)
        self.mask_type_combo.blockSignals(False)

    def _sync_mask_editor(self, mask: MaskSpec, dataset: DatasetEntry | None = None) -> None:
        ensure_coordinate_range_mask_axes(mask, dataset)
        self._refresh_mask_type_combo()
        index = self.mask_type_combo.findData(mask.type)
        self.mask_type_combo.blockSignals(True)
        self.mask_type_combo.setCurrentIndex(max(index, 0))
        self.mask_type_combo.blockSignals(False)
        self._rebuild_mask_parameter_editor(mask, dataset)

    def _set_selected_mask_type(self, _label: str) -> None:
        group, entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        type_name = self.mask_type_combo.currentData()
        if not type_name or type_name == mask.type:
            return
        mask.type = str(type_name)
        defaults = default_mask_parameters(mask.type)
        mask.parameters = {name: mask.parameters.get(name, value) for name, value in defaults.items()}
        ensure_coordinate_range_mask_axes(mask, entry)
        affected = self._selected_mask_datasets(group, entry, role)
        self._mark_mask_datasets_stale(affected)
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_mask_parameter_editor(mask, entry)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_mask=mask)
            else:
                self.refresh_slice_viewer(group)

    def _rebuild_mask_parameter_editor(self, mask: MaskSpec, dataset: DatasetEntry | None = None) -> None:
        from PySide6 import QtWidgets

        ensure_coordinate_range_mask_axes(mask, dataset)
        self._clear_mask_parameter_editor()
        invert_check = QtWidgets.QCheckBox("Invert")
        invert_check.setChecked(bool(mask.invert))
        invert_check.setToolTip(
            "Flip this mask contribution. Normally the specified region is masked out; inverted mode masks everything outside it instead."
        )
        invert_check.toggled.connect(lambda checked: self._set_mask_invert(checked))
        additive_check = QtWidgets.QCheckBox("Additive")
        additive_check.setChecked(bool(mask.additive))
        additive_check.setToolTip(
            "Remove this specified region from masks accumulated earlier, adding those bins back to the analysis. This never overrides the file mask."
        )
        additive_check.toggled.connect(lambda checked: self._set_mask_additive(checked))
        self.mask_parameter_layout.addWidget(invert_check, 0, 0)
        self.mask_parameter_layout.addWidget(additive_check, 0, 1)
        affected = self._selected_mask_datasets_for_current_item()
        auto_check = QtWidgets.QCheckBox("Automatic mask application")
        auto_check.setObjectName("mask_auto_apply")
        auto_check.setChecked(bool(affected) and all(_dataset_mask_auto_enabled(item) for item in affected))
        auto_check.setToolTip(
            "Automatically apply mask edits to cached viewer/fit data. Large datasets default off. "
            "When off, edits remain pending until Apply masks now is pressed, a fit starts, or the data viewer is opened."
        )
        auto_check.toggled.connect(self._set_selected_mask_auto_apply)
        apply_button = QtWidgets.QPushButton("Apply masks now")
        apply_button.setObjectName("mask_apply_now")
        apply_button.setEnabled(bool(affected))
        apply_button.setToolTip(
            "Apply all enabled dataset and inherited masks now and update the cached viewer/fit representation."
        )
        apply_button.clicked.connect(self.apply_masks_now_for_selection)
        status_label = QtWidgets.QLabel(self._mask_status_for_datasets(affected))
        status_label.setObjectName("mask_application_status")
        status_label.setWordWrap(True)
        status_label.setToolTip(
            "Shows whether mask edits are already materialized. Pending masks are forced current before fitting or opening the data viewer."
        )
        self.mask_application_status_label = status_label
        self.mask_parameter_layout.addWidget(auto_check, 1, 0, 1, 2)
        self.mask_parameter_layout.addWidget(apply_button, 2, 0, 1, 2)
        self.mask_parameter_layout.addWidget(status_label, 3, 0, 1, 2)
        parameter_names = _mask_parameter_names(mask, dataset)
        layout_row = 4
        added_axis_section = False
        for parameter_name in parameter_names:
            if (
                mask.type == "coordinate_range"
                and parameter_name.startswith(COORDINATE_RANGE_AXIS_PREFIX)
                and not added_axis_section
            ):
                section_label = QtWidgets.QLabel("Coordinate axes")
                section_label.setStyleSheet("font-weight: 600")
                self.mask_parameter_layout.addWidget(section_label, layout_row, 0, 1, 2)
                layout_row += 1
                added_axis_section = True
            label = QtWidgets.QLabel(parameter_name)
            editor = QtWidgets.QLineEdit(_parameter_to_text(mask.parameters.get(parameter_name, "")))
            tooltip = mask_parameter_tooltip(mask.type, parameter_name)
            label.setToolTip(tooltip)
            editor.setToolTip(tooltip)
            editor.editingFinished.connect(
                lambda parameter_name=parameter_name, editor=editor: self._set_mask_parameter(parameter_name, editor.text())
            )
            self.mask_parameter_layout.addWidget(label, layout_row, 0)
            self.mask_parameter_layout.addWidget(editor, layout_row, 1)
            layout_row += 1

    def _clear_mask_parameter_editor(self) -> None:
        self.mask_application_status_label = None
        while self.mask_parameter_layout.count():
            item = self.mask_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _set_mask_parameter(self, name: str, text: str) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        value = _parse_parameter_text(text)
        changed = False
        if mask.parameters.get(name) != value:
            mask.parameters[name] = value
            changed = True
            self._mark_mask_datasets_stale(self._selected_mask_datasets(group, _entry, role))
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_mask=mask)
                return
        if changed:
            self._rebuild_mask_parameter_editor(mask, _entry)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_mask_invert(self, checked: bool) -> None:
        self._set_mask_option("invert", bool(checked))

    def _set_mask_additive(self, checked: bool) -> None:
        self._set_mask_option("additive", bool(checked))

    def _set_mask_option(self, name: str, value: bool) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        if bool(getattr(mask, name)) == bool(value):
            return
        setattr(mask, name, bool(value))
        self._mark_mask_datasets_stale(self._selected_mask_datasets(group, _entry, role))
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_mask=mask)
            return
        if group is not None:
            self.refresh_slice_viewer(group)

    def _selected_mask_datasets(
        self,
        group: DataGroup | None,
        entry: DatasetEntry | None,
        role: str,
    ) -> list[DatasetEntry]:
        if role == "mask" and entry is not None:
            return [entry]
        if role == "group_mask" and group is not None:
            subgroup = self._dataset_group_for_item(self._current_item())
            return list(subgroup.iter_datasets()) if subgroup is not None else []
        return []

    def _selected_mask_datasets_for_current_item(self) -> list[DatasetEntry]:
        group, entry, _mask, _model, role = self._objects_for_item(self._current_item())
        return self._selected_mask_datasets(group, entry, role)

    def _mark_mask_datasets_stale(self, datasets: list[DatasetEntry]) -> None:
        for dataset in datasets:
            dataset_mask_application_config(dataset)["stale"] = True
        if datasets:
            self._refresh_cache_badges()
        label = self.mask_application_status_label
        if label is not None:
            try:
                label.setText(self._mask_status_for_datasets(datasets))
            except RuntimeError:
                # A queued details-panel deletion can invalidate this widget
                # before the owning reference is rebuilt.
                self.mask_application_status_label = None

    def _mask_status_for_datasets(self, datasets: list[DatasetEntry]) -> str:
        statuses = {_dataset_mask_status_text(item) for item in datasets}
        if len(statuses) == 1:
            return next(iter(statuses))
        if not statuses:
            return "No datasets are affected by this mask."
        return "Mask application settings differ across affected datasets."

    def _set_selected_mask_auto_apply(self, checked: bool) -> None:
        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or mask is None:
            return
        affected = self._selected_mask_datasets_for_current_item()
        changed = False
        for dataset in affected:
            config = dataset_mask_application_config(dataset)
            if bool(config.get("auto_apply", True)) != bool(checked):
                config["auto_apply"] = bool(checked)
                changed = True
        if not changed:
            return
        self._mark_dirty()
        if checked and group is not None:
            self._request_overlay_refresh(group)
        self._rebuild_mask_parameter_editor(mask, _entry)

    def apply_masks_now_for_selection(self) -> bool:
        from PySide6 import QtCore, QtWidgets

        group, _entry, mask, _model, role = self._objects_for_item(self._current_item())
        if role not in {"mask", "group_mask"} or group is None or mask is None:
            return False
        affected = self._selected_mask_datasets_for_current_item()
        progress = QtWidgets.QProgressDialog(
            "Applying masks...",
            "",
            0,
            len(affected),
            self.window,
        )
        progress.setWindowTitle("Apply masks")
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        try:
            for index, dataset in enumerate(affected, start=1):
                progress.setLabelText(f"Applying masks to {dataset.name}...")
                QtWidgets.QApplication.processEvents()
                rebin_progress = None
                if dataset_rebin_enabled(dataset):
                    # Applying masks can immediately enter a long rebin. Hide
                    # the coarse per-dataset mask progress while the rebinner
                    # reports its own batch-level progress.
                    progress.hide()
                    rebin_progress = self._make_rebin_progress_callback(
                        f"Rebinning {dataset.name}..."
                    )
                try:
                    dataset_for_slice_viewer(
                        dataset,
                        extra_masks=effective_dataset_masks(group, dataset),
                        force_rebin=True,
                        force_masks=True,
                        progress_callback=rebin_progress,
                    )
                except RebinCancellationRequested:
                    return False
                finally:
                    self._close_rebin_progress(rebin_progress)
                    if dataset_rebin_enabled(dataset) and index < len(affected):
                        progress.show()
                progress.setValue(index)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Apply masks",
                f"Could not apply masks:\n{exc}",
            )
            return False
        finally:
            progress.close()
        self._mark_dirty()
        self.refresh_slice_viewer(group)
        self._rebuild_mask_parameter_editor(mask, _entry)
        return True

    def _refresh_model_category_combo(
        self,
        selected_category: str | None = None,
    ) -> None:
        current = (
            selected_category
            if selected_category is not None
            else self.model_category_combo.currentData()
        )
        self.model_category_combo.blockSignals(True)
        self.model_category_combo.clear()
        for category in available_model_categories():
            self.model_category_combo.addItem(
                MODEL_CATEGORY_LABELS[category],
                category,
            )
        index = self.model_category_combo.findData(current)
        self.model_category_combo.setCurrentIndex(max(index, 0))
        self.model_category_combo.blockSignals(False)

    def _refresh_model_type_combo(
        self,
        selected_type: str | None = None,
        *,
        include_placeholder: bool = False,
    ) -> None:
        current = (
            selected_type
            if selected_type is not None
            else self.model_type_combo.currentData()
        )
        category = self.model_category_combo.currentData()
        self.model_type_combo.blockSignals(True)
        self.model_type_combo.clear()
        if include_placeholder or current is None:
            self.model_type_combo.addItem("Choose model…", None)
        for type_name in (
            model_types_in_category(str(category))
            if category is not None
            else ()
        ):
            self.model_type_combo.addItem(model_definition(type_name).label, type_name)
        if current is not None:
            index = self.model_type_combo.findData(current)
            if index >= 0:
                self.model_type_combo.setCurrentIndex(index)
        self.model_type_combo.blockSignals(False)

    def _sync_model_editor(self, model: ModelComponentSpec) -> None:
        category = model_definition(model.type).category
        self._refresh_model_category_combo(category)
        self._refresh_model_type_combo(model.type)
        self._rebuild_model_parameter_editor(model)

    def _set_selected_model_category(self, _index: int) -> None:
        category = self.model_category_combo.currentData()
        if category is None:
            return
        _group, _entry, _mask, model, role = self._objects_for_item(
            self._current_item()
        )
        selected_type = (
            model.type
            if role == "model"
            and model is not None
            and model_definition(model.type).category == category
            else None
        )
        self._refresh_model_type_combo(
            selected_type,
            include_placeholder=selected_type is None,
        )

    def _set_selected_model_type(self, _label: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        type_name = self.model_type_combo.currentData()
        if not type_name or type_name == model.type:
            return
        model.type = str(type_name)
        defaults = default_model_parameters(model.type)
        default_config = default_model_config(model.type)
        default_fit = default_model_fit_parameters(model.type)
        model.parameters = {name: model.parameters.get(name, value) for name, value in defaults.items()}
        model.config = {name: model.config.get(name, value) for name, value in default_config.items()}
        model.fit_parameters = {name: model.fit_parameters.get(name, value) for name, value in default_fit.items()}
        model.sharing = {
            name: copy.deepcopy(
                model.sharing.get(name, {"mode": "global", "groups": {}})
            )
            for name in defaults
        }
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        self._rebuild_model_parameter_editor(model)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            self._request_overlay_refresh(group)

    def _rebuild_model_parameter_editor(self, model: ModelComponentSpec) -> None:
        return _project_model_editor._rebuild_model_parameter_editor(self, model)

    def _rebuild_model_parameter_editor_preserving_scroll(
        self,
        model: ModelComponentSpec,
    ) -> None:
        return _project_model_editor._rebuild_model_parameter_editor_preserving_scroll(
            self, model
        )

    def _clear_model_parameter_editor(self) -> None:
        while self.model_parameter_layout.count():
            item = self.model_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _build_tight_binding_state_editor(
        self,
        model: ModelComponentSpec,
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_state_editor(
            self, model
        )

    def _tight_binding_dos_sampling_status(
        self,
        model: ModelComponentSpec,
    ) -> str:
        return _project_tight_binding_editor._tight_binding_dos_sampling_status(
            self, model
        )

    def _build_tight_binding_dos_sampling_editor(
        self,
        model: ModelComponentSpec,
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_dos_sampling_editor(
            self, model
        )

    def _organize_tight_binding_editor(
        self,
        model: ModelComponentSpec,
    ) -> None:
        return _project_tight_binding_editor._organize_tight_binding_editor(
            self, model
        )

    def _organize_heisenberg_editor(
        self,
        model: ModelComponentSpec,
        *,
        fit_group: Any,
        scope_group: Any,
        advanced_group: Any,
    ) -> None:
        """Collect the Heisenberg RPA workflow into focused tabs."""

        from PySide6 import QtWidgets

        def take(widget: Any | None) -> Any | None:
            if widget is not None:
                self.model_parameter_layout.removeWidget(widget)
            return widget

        def named(name: str) -> Any | None:
            return self.model_parameter_widget.findChild(QtWidgets.QWidget, name)

        groups = {
            "crystal": take(named("model_crystal_group")),
            "sites": take(named("model_crystal_sites_group")),
            "bonds": take(named("model_bonds_group")),
            "fit": take(fit_group),
            "interactions": take(named("model_interactions_group")),
            "closure": take(named("model_closure_group")),
            "scope": take(scope_group),
            "plots": take(named("model_plot_actions_group")),
            "advanced": take(advanced_group),
            "numerical": take(named("heisenberg_numerical_group")),
        }

        interaction_names = []
        if model.config.get("anisotropy"):
            interaction_names.append("anisotropic exchange")
        if model.config.get("sia"):
            interaction_names.append("single-ion anisotropy")
        if (model.config.get("dipole") or {}).get("enabled"):
            interaction_names.append("dipole")
        if (model.config.get("zeeman") or {}).get("enabled"):
            interaction_names.append("field")
        closure = str((model.config.get("closure") or {}).get("mode", "none"))
        summary = QtWidgets.QGroupBox("Model summary")
        summary.setObjectName("heisenberg_model_summary_group")
        summary_layout = QtWidgets.QVBoxLayout(summary)
        summary_label = QtWidgets.QLabel(
            f"{len(model.config.get('site_positions') or ())} magnetic site(s) · "
            f"{len(model.config.get('orbits') or ())} exchange orbit(s) · "
            f"{', '.join(interaction_names) if interaction_names else 'isotropic'} · "
            f"{'bare RPA' if closure == 'none' else closure.upper() + ' closure'}"
        )
        summary_label.setObjectName("heisenberg_model_summary")
        summary_label.setWordWrap(True)
        summary_label.setToolTip(
            "Resolved status of the generated magnetic network, enabled "
            "interactions, and self-consistency treatment."
        )
        summary_layout.addWidget(summary_label)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("heisenberg_builder_tabs")
        tabs.setToolTip(
            "Define the magnetic structure and exchange network, configure the "
            "response and fit, and tune numerical convergence."
        )

        def add_page(label: str, names: tuple[str, ...]) -> None:
            page = QtWidgets.QWidget()
            page.setObjectName(
                f"heisenberg_{label.lower().replace(' ', '_')}_tab"
            )
            layout = QtWidgets.QVBoxLayout(page)
            for name in names:
                widget = groups.get(name)
                if widget is not None:
                    layout.addWidget(widget)
            layout.addStretch(1)
            tabs.addTab(page, label)

        add_page("Structure and exchange", ("crystal", "sites", "bonds"))
        add_page(
            "Response and fit",
            ("fit", "interactions", "closure", "scope"),
        )
        if groups["plots"] is not None:
            add_page("Calculate and inspect", ("plots",))
        add_page("Advanced", ("advanced", "numerical"))
        self.model_parameter_layout.addWidget(summary, 0, 0, 1, 4)
        self.model_parameter_layout.addWidget(tabs, 1, 0, 1, 4)

    def _lindhard_source_component(
        self,
        model: ModelComponentSpec,
    ) -> ModelComponentSpec | None:
        return _project_lindhard_editor._lindhard_source_component(self, model)

    def _build_lindhard_editor(self, model: ModelComponentSpec) -> None:
        return _project_lindhard_editor._build_lindhard_editor(self, model)

    def _organize_lindhard_editor(
        self,
        model: ModelComponentSpec,
        *,
        fit_group: Any,
        scope_group: Any,
        advanced_group: Any,
    ) -> None:
        return _project_lindhard_editor._organize_lindhard_editor(
            self,
            model,
            fit_group=fit_group,
            scope_group=scope_group,
            advanced_group=advanced_group,
        )

    def _build_tight_binding_editor(self, model: ModelComponentSpec) -> None:
        return _project_tight_binding_editor._build_tight_binding_editor(self, model)

    def _build_model_plot_actions(self, model: ModelComponentSpec) -> None:
        """Add registered plot and script actions for a non-electronic model."""

        from PySide6 import QtWidgets

        plots = model_plot_definitions(model.type)
        if not plots:
            return
        group = QtWidgets.QGroupBox("Model plots")
        group.setObjectName("model_plot_actions_group")
        layout = QtWidgets.QGridLayout(group)
        for row, plot in enumerate(plots):
            calculate = QtWidgets.QPushButton(plot.label)
            calculate.setObjectName(f"model_plot_{plot.key}")
            calculate.setToolTip(plot.description)
            calculate.clicked.connect(
                lambda _checked=False, model=model, key=plot.key: self._open_model_plot(
                    model, key
                )
            )
            copy_script = QtWidgets.QPushButton("Copy script")
            copy_script.setObjectName(f"model_plot_script_{plot.key}")
            copy_script.setToolTip(
                f"Copy editable Python that reproduces the "
                f"{plot.label.lower()} calculation."
            )
            copy_script.clicked.connect(
                lambda _checked=False, model=model, key=plot.key: self._copy_model_plot_script(
                    model, key
                )
            )
            layout.addWidget(calculate, row, 0)
            layout.addWidget(copy_script, row, 1)
        self.model_parameter_layout.addWidget(group, 9, 0, 1, 4)

    def _open_electronic_matrix_inspector(
        self,
        model: ModelComponentSpec,
    ) -> bool:
        from PySide6 import QtWidgets

        from .electronic_matrix import electronic_matrix_catalog
        from .qt_electronic_matrix_viewer import (
            show_electronic_matrix_catalog,
        )

        try:
            window = show_electronic_matrix_catalog(
                electronic_matrix_catalog(model),
                parent=self.window,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Electronic matrix inspector",
                f"Could not open the matrix inspector:\n{exc}",
            )
            return False
        self._plot_windows[f"model:{id(model)}:matrix"] = window
        return True

    def _copy_electronic_matrix_script(
        self,
        model: ModelComponentSpec,
    ) -> bool:
        from PySide6 import QtWidgets

        from .electronic_matrix import electronic_matrix_script

        try:
            script = electronic_matrix_script(model)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Electronic matrix script",
                f"Could not create the matrix-viewer script:\n{exc}",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def _open_brillouin_zone(self, model: ModelComponentSpec) -> bool:
        from PySide6 import QtWidgets

        from .brillouin_zone import brillouin_zone_scene
        from .qt_brillouin_zone_viewer import show_brillouin_zone_scene

        try:
            window = show_brillouin_zone_scene(
                brillouin_zone_scene(model),
                parent=self.window,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Brillouin zone",
                f"Could not open the Brillouin-zone viewer:\n{exc}",
            )
            return False
        self._plot_windows[f"model:{id(model)}:brillouin_zone"] = window
        return True

    def _copy_brillouin_zone_script(
        self, model: ModelComponentSpec
    ) -> bool:
        from PySide6 import QtWidgets

        from .brillouin_zone import brillouin_zone_script

        try:
            script = brillouin_zone_script(model)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Brillouin-zone script",
                f"Could not create the Brillouin-zone script:\n{exc}",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def _build_tight_binding_orbital_editor(
        self, model: ModelComponentSpec
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_orbital_editor(
            self, model
        )

    def _build_tight_binding_spin_editor(
        self,
        model: ModelComponentSpec,
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_spin_editor(
            self, model
        )

    def _tight_binding_sharing_controls(
        self,
        model: ModelComponentSpec,
        identifier: str,
        *,
        object_prefix: str,
    ) -> tuple[Any, Any]:
        return _project_tight_binding_editor._tight_binding_sharing_controls(
            self,
            model,
            identifier,
            object_prefix=object_prefix,
        )

    def _build_tight_binding_onsite_editor(
        self, model: ModelComponentSpec
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_onsite_editor(
            self, model
        )

    def _build_tight_binding_hopping_editor(
        self, model: ModelComponentSpec
    ) -> None:
        return _project_tight_binding_editor._build_tight_binding_hopping_editor(
            self, model
        )

    def _import_wannier90_model(self, model: ModelComponentSpec) -> bool:
        from PySide6 import QtWidgets

        from .electronic_structure import import_wannier90

        path, _selected_filter = get_open_file_name(
            self.window,
            "Import Wannier90 Hamiltonian",
            "",
            "Wannier90 Hamiltonians (*_tb.dat *_hr.dat);;All files (*)",
        )
        if not path:
            return False
        try:
            raw_axes = model.config.get("periodic_axes", [])
            imported = import_wannier90(
                path,
                periodic_axes=(
                    None
                    if not raw_axes
                    else tuple(int(value) for value in raw_axes)
                ),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Import Wannier90 Hamiltonian",
                f"Could not import the electronic model:\n{exc}",
            )
            return False
        model.config["source_path"] = str(Path(path).resolve())
        model.config["model_digest"] = imported.content_digest
        model.config["model_data"] = {}
        model.config["periodic_axes"] = list(imported.periodic_axes)
        model.config["orbital_manifolds"] = []
        model.config["spin_treatment"] = "auto"
        model.config["soc_terms"] = []
        model.config["onsite_terms"] = []
        model.config["hopping_cutoff_angstrom"] = 0.0
        model.config["hopping_candidates"] = []
        model.config["hopping_terms"] = []
        model.config["spatial_orbits"] = []
        model.config["site_positions"] = []
        model.config["expanded_crystal_sites"] = []
        from .electronic_builder import reconcile_tight_binding_parameters

        reconcile_tight_binding_parameters(model)
        owner = self._group_for_model(model)
        if owner is not None:
            self._record_data_group_state_change(owner)
        self._mark_dirty()
        self._rebuild_model_parameter_editor(model)
        return True

    def _group_for_model(self, model: ModelComponentSpec) -> DataGroup | None:
        return next(
            (
                group
                for group in self.project.data_groups
                if any(candidate is model for candidate in group.models.values())
            ),
            None,
        )

    def _tight_binding_plot_settings(
        self,
        model: ModelComponentSpec,
        plot_key: str,
    ) -> dict[str, Any]:
        """Return viewer-owned settings in their displayed electronic unit."""

        from .electronic_structure import (
            electronic_energy_from_meV,
            normalize_electronic_energy_unit,
        )

        config = model.config
        unit = normalize_electronic_energy_unit(
            config.get("electronic_energy_unit", "eV")
        )
        common = {"electronic_energy_unit": unit}
        if plot_key == "bands":
            return {
                **common,
                "band_path_convention": config.get(
                    "band_path_convention",
                    "hinuma",
                ),
                "band_path": _parameter_to_text(
                    config.get("band_path", [])
                ),
                "band_points_per_inv_angstrom": _parameter_to_text(
                    config.get("band_points_per_inv_angstrom", 80.0)
                ),
            }
        if plot_key == "dos":
            return {
                **common,
                "dos_method": config.get("dos_method", "gaussian"),
                "dos_sampling_status": (
                    self._tight_binding_dos_sampling_status(model)
                ),
                "dos_symmetry": config.get("dos_symmetry", "auto"),
                "dos_auto_energy_range": bool(
                    config.get("dos_auto_energy_range", False)
                ),
                "dos_energy_min_meV": _parameter_to_text(
                    electronic_energy_from_meV(
                        config.get("dos_energy_min_meV", -500.0),
                        unit,
                    )
                ),
                "dos_energy_max_meV": _parameter_to_text(
                    electronic_energy_from_meV(
                        config.get("dos_energy_max_meV", 500.0),
                        unit,
                    )
                ),
                "dos_energy_points": _parameter_to_text(
                    config.get("dos_energy_points", 600)
                ),
                "dos_broadening_meV": _parameter_to_text(
                    electronic_energy_from_meV(
                        config.get("dos_broadening_meV", 5.0),
                        unit,
                    )
                ),
            }
        if plot_key == "fermi_surface":
            resolved_mesh = config.get("fermi_mesh", [64, 64, 64])
            if str(config.get("fermi_mesh_mode", "spacing")) == "spacing":
                try:
                    from .electronic_structure import (
                        electronic_model_from_component,
                        reciprocal_mesh_shape_for_spacing,
                    )

                    resolved_mesh = reciprocal_mesh_shape_for_spacing(
                        electronic_model_from_component(model),
                        float(
                            config.get(
                                "fermi_spacing_inv_angstrom",
                                0.025,
                            )
                        ),
                    )
                except (ImportError, TypeError, ValueError):
                    pass
            return {
                **common,
                "fermi_mesh_mode": config.get(
                    "fermi_mesh_mode", "spacing"
                ),
                "fermi_spacing_inv_angstrom": _parameter_to_text(
                    config.get("fermi_spacing_inv_angstrom", 0.025)
                ),
                "fermi_mesh": _parameter_to_text(
                    resolved_mesh
                ),
                "fermi_energy_meV": _parameter_to_text(
                    electronic_energy_from_meV(
                        config.get(
                            "fermi_energy_meV",
                            config.get("chemical_potential_meV", 0.0),
                        ),
                        unit,
                    )
                ),
            }
        return common

    def _apply_tight_binding_plot_settings(
        self,
        model: ModelComponentSpec,
        plot_key: str,
        values: Mapping[str, str],
    ) -> bool:
        """Validate and store one viewer's calculation settings."""

        from PySide6 import QtWidgets

        from .electronic_structure import (
            electronic_energy_to_meV,
            normalize_electronic_energy_unit,
        )
        from .model_plots import configure_tight_binding_plot

        previous = copy.deepcopy(model.config)
        unit = normalize_electronic_energy_unit(
            model.config.get("electronic_energy_unit", "eV")
        )
        try:
            if plot_key == "bands":
                convention = str(
                    values.get("band_path_convention", "manual")
                )
                updates = {
                    "band_path_convention": convention,
                    "band_points_per_inv_angstrom": float(
                        _parse_parameter_text(
                            values["band_points_per_inv_angstrom"]
                        )
                    ),
                }
                if convention == "manual":
                    path = _parse_parameter_text(
                        values["band_path"]
                    )
                    if not isinstance(path, list) or len(path) < 2:
                        raise ValueError(
                            "manual band_path must contain at least two nodes"
                        )
                    updates["band_path"] = path
                configure_tight_binding_plot(
                    model,
                    plot_key,
                    **updates,
                )
            elif plot_key == "dos":
                method = str(values["dos_method"])
                automatic_energy_range = (
                    str(values.get("dos_auto_energy_range", "false"))
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "on"}
                )
                symmetry = str(values["dos_symmetry"])
                energies: dict[str, float] = {}
                for name in (
                    "dos_energy_min_meV",
                    "dos_energy_max_meV",
                    "dos_broadening_meV",
                ):
                    energies[name] = float(
                        electronic_energy_to_meV(
                            float(_parse_parameter_text(values[name])),
                            unit,
                        )
                    )
                energy_points = int(
                    _parse_parameter_text(values["dos_energy_points"])
                )
                if energy_points < 2:
                    raise ValueError(
                        "dos_energy_points must be at least two"
                    )
                configure_tight_binding_plot(
                    model,
                    plot_key,
                    dos_method=method,
                    dos_symmetry=symmetry,
                    dos_auto_energy_range=automatic_energy_range,
                    dos_energy_points=energy_points,
                    **energies,
                )
            elif plot_key == "fermi_surface":
                mesh_mode = str(
                    values.get("fermi_mesh_mode", "spacing")
                )
                spacing = float(
                    _parse_parameter_text(
                        values.get(
                            "fermi_spacing_inv_angstrom",
                            "0.025",
                        )
                    )
                )
                mesh = _parse_parameter_text(
                    values["fermi_mesh"]
                )
                if (
                    not isinstance(mesh, (list, tuple))
                    or not mesh
                    or any(int(size) < 2 for size in mesh)
                ):
                    raise ValueError(
                        "fermi_mesh must contain integer sizes of at least two"
                    )
                if mesh_mode not in {"spacing", "size"}:
                    raise ValueError(
                        "fermi_mesh_mode must be spacing or size"
                    )
                if not np.isfinite(spacing) or spacing <= 0.0:
                    raise ValueError(
                        "Fermi-surface spacing must be positive and finite"
                    )
                energy = float(
                    electronic_energy_to_meV(
                        float(
                            _parse_parameter_text(
                                values["fermi_energy_meV"]
                            )
                        ),
                        unit,
                    )
                )
                configure_tight_binding_plot(
                    model,
                    plot_key,
                    fermi_mesh_mode=mesh_mode,
                    fermi_spacing_inv_angstrom=spacing,
                    fermi_mesh=mesh,
                    fermi_energy_meV=energy,
                )
            else:
                raise ValueError(
                    f"unsupported tight-binding plot settings {plot_key!r}"
                )
        except Exception as exc:
            model.config = previous
            QtWidgets.QMessageBox.warning(
                self.window,
                "Electronic plot settings",
                f"Could not apply the settings:\n{exc}",
            )
            return False

        if model.config == previous:
            return True
        group = self._group_for_model(model)
        branch_created = (
            self._record_data_group_state_change(group)
            if group is not None
            else False
        )
        self._mark_dirty()
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            self._request_overlay_refresh(group)
        return True

    def _lindhard_plot_settings(
        self,
        model: ModelComponentSpec,
        plot_key: str,
    ) -> dict[str, Any]:
        """Return the scientific settings owned by one Lindhard viewer."""

        names = {
            "complex_energy_scan": (
                "plot_q_reduced",
                "plot_energy_min_meV",
                "plot_energy_max_meV",
                "plot_energy_points",
                "plot_temperature_K",
            ),
            "convergence": (
                "plot_q_reduced",
                "plot_energy_min_meV",
                "plot_energy_max_meV",
                "plot_temperature_K",
                "convergence_mesh_scales",
                "convergence_broadening_scales",
                "convergence_energy_points",
            ),
        }.get(plot_key, ())
        return {
            name: _parameter_to_text(model.config.get(name, ""))
            for name in names
        }

    def _apply_lindhard_plot_settings(
        self,
        model: ModelComponentSpec,
        plot_key: str,
        values: Mapping[str, str],
    ) -> bool:
        """Validate and store one Lindhard viewer's calculation settings."""

        from PySide6 import QtWidgets

        from .model_plots import configure_lindhard_plot

        previous = copy.deepcopy(model.config)
        try:
            updates = {
                name: _parse_parameter_text(text)
                for name, text in values.items()
            }
            q = updates.get("plot_q_reduced")
            if q is not None and (
                not isinstance(q, (list, tuple)) or len(q) != 3
            ):
                raise ValueError(
                    "plot_q_reduced must contain three reduced coordinates"
                )
            configure_lindhard_plot(model, plot_key, **updates)
        except Exception as exc:
            model.config = previous
            QtWidgets.QMessageBox.warning(
                self.window,
                "Lindhard plot settings",
                f"Could not apply the settings:\n{exc}",
            )
            return False

        if model.config == previous:
            return True
        group = self._group_for_model(model)
        branch_created = (
            self._record_data_group_state_change(group)
            if group is not None
            else False
        )
        self._mark_dirty()
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            self._request_overlay_refresh(group)
        return True

    def _open_model_plot(self, model: ModelComponentSpec, plot_key: str) -> bool:
        from PySide6 import QtWidgets

        from .qt_electronic_viewer import show_electronic_figure

        plot = next(
            (
                candidate
                for candidate in model_plot_definitions(model.type)
                if candidate.key == plot_key
            ),
            None,
        )
        if plot is None or plot.render is None:
            return False
        window_holder: dict[str, Any] = {}

        def calculate_result() -> Any:
            owner = self._group_for_model(model)
            components = {} if owner is None else owner.models
            return (
                plot.context_calculate(model, components)
                if plot.context_calculate is not None
                else plot.calculate(model)
            )

        def apply_settings(values: dict[str, str]) -> None:
            applied = (
                self._apply_tight_binding_plot_settings(
                    model,
                    plot_key,
                    values,
                )
                if model.type == "tight_binding"
                else self._apply_lindhard_plot_settings(
                    model,
                    plot_key,
                    values,
                )
                if model.type == "lindhard"
                else False
            )
            if not applied:
                return
            current = window_holder.get("window")
            if current is None:
                return
            try:
                updated_result = calculate_result()
                if plot_key == "fermi_surface" and updated_result.dimension == 3:
                    current._nfit_replace_result(updated_result)
                else:
                    updated_figure, _axes = plot.render(updated_result)
                    current._nfit_replace_figure(updated_figure)
                if hasattr(current, "_nfit_update_calculation_settings"):
                    current._nfit_update_calculation_settings(
                        self._tight_binding_plot_settings(model, plot_key)
                        if model.type == "tight_binding"
                        else self._lindhard_plot_settings(model, plot_key)
                    )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Model plot",
                    f"Could not recalculate the plot:\n{exc}",
                )

        try:
            result = calculate_result()
            if plot_key == "fermi_surface" and result.dimension == 3:
                from .qt_fermi_surface_viewer import show_fermi_surface_result

                window = show_fermi_surface_result(
                    result,
                    parent=self.window,
                    settings_config=self._tight_binding_plot_settings(
                        model,
                        plot_key,
                    ),
                    on_apply_settings=apply_settings,
                )
            else:
                figure, _axes = plot.render(result)
                viewer_keys = {
                    "bands": "band_structure",
                    "dos": "density_of_states",
                    "fermi_surface": "fermi_surface",
                    "complex_energy_scan": "susceptibility",
                    "convergence": "response_convergence",
                }
                settings_config = (
                    self._tight_binding_plot_settings(model, plot_key)
                    if model.type == "tight_binding"
                    else self._lindhard_plot_settings(model, plot_key)
                    if model.type == "lindhard"
                    else None
                )
                window = show_electronic_figure(
                    figure,
                    viewer_key=viewer_keys.get(plot_key, "model_plot"),
                    parent=self.window,
                    settings_config=settings_config,
                    on_apply_settings=(
                        apply_settings
                        if model.type in {"tight_binding", "lindhard"}
                        else None
                    ),
                )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Model plot",
                f"Could not calculate the plot:\n{exc}",
            )
            return False
        window_holder["window"] = window
        self._plot_windows[f"model:{id(model)}:{plot_key}"] = window
        return True

    def _copy_model_plot_script(
        self, model: ModelComponentSpec, plot_key: str
    ) -> bool:
        from PySide6 import QtWidgets

        plot = next(
            (
                candidate
                for candidate in model_plot_definitions(model.type)
                if candidate.key == plot_key
            ),
            None,
        )
        if plot is None or plot.script is None:
            return False
        try:
            owner = self._group_for_model(model)
            components = {} if owner is None else owner.models
            script = (
                plot.context_script(model, components)
                if plot.context_script is not None
                else plot.script(model)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Model plot script",
                f"Could not create the script:\n{exc}",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def _copy_tight_binding_structure_script(
        self, model: ModelComponentSpec
    ) -> bool:
        from PySide6 import QtWidgets

        from .electronic_structure import (
            electronic_model_from_component,
            tight_binding_structure_script,
        )

        owner = self._group_for_model(model)
        try:
            electronic_model_from_component(model)
            script = tight_binding_structure_script(
                model_crystal_config(model),
                model.config.get("periodic_axes") or [0, 1, 2],
                group_name=owner.name if owner is not None else "Electronic",
                model_name=model.name,
                electronic_energy_unit=model.config.get(
                    "electronic_energy_unit", "eV"
                ),
                use_primitive_cell=bool(
                    model.config.get("use_primitive_cell", True)
                ),
                hopping_parameterization=str(
                    model.config.get(
                        "hopping_parameterization",
                        "slater_koster",
                    )
                ),
                band_path_nodes=model.config.get("band_path", ()),
                band_path_convention=str(
                    model.config.get("band_path_convention", "hinuma")
                ),
                band_path_metadata=model.config.get(
                    "band_path_metadata",
                    {},
                ),
                orbital_manifolds=model.config.get("orbital_manifolds", ()),
                onsite_terms=model.config.get("onsite_terms", ()),
                hopping_cutoff_angstrom=(
                    float(model.config.get("hopping_cutoff_angstrom", 0.0))
                    or None
                ),
                hopping_terms=model.config.get("hopping_terms", ()),
                spin_treatment=str(
                    model.config.get("spin_treatment", "auto")
                ),
                soc_terms=model.config.get("soc_terms", ()),
                parameter_values_meV=model.parameters,
                fit_parameters=model.fit_parameters,
                parameter_limits_meV=model.limits,
                parameter_sharing=model.sharing,
                expected_model_digest=str(model.config.get("model_digest", "")),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Tight-binding structure script",
                f"Could not create the script:\n{exc}",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def _build_model_crystal_editor(
        self, model: ModelComponentSpec, *, structure_only: bool = False
    ) -> None:
        """Structured crystal editor with optional magnetic interactions.

        Tight binding uses the shared lattice and site sections without
        magnetic selection or exchange bonds. Heisenberg RPA continues into
        the bond, interaction, and closure sections.
        """

        from PySide6 import QtWidgets

        crystal = model_crystal_config(model)
        lattice = crystal["lattice"]

        crystal_group = QtWidgets.QGroupBox("Crystal")
        crystal_group.setObjectName("model_crystal_group")
        crystal_layout = QtWidgets.QGridLayout(crystal_group)
        lattice_tooltip = (
            "Unit-cell parameter used for orbital locations and electronic "
            "wavevectors. Lengths in Angstrom, angles in degrees."
            if structure_only
            else "Unit-cell parameter used to build exchange bonds and convert "
            "HKL to |Q|. Lengths in Angstrom, angles in degrees."
        )
        for column, name in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            label = QtWidgets.QLabel(name)
            label.setToolTip(lattice_tooltip)
            editor = QtWidgets.QLineEdit(_parameter_to_text(lattice.get(name, "")))
            editor.setObjectName(f"model_crystal_{name}")
            editor.setToolTip(lattice_tooltip)
            editor.setMaximumWidth(70)
            editor.editingFinished.connect(
                lambda name=name, editor=editor: self._set_model_crystal_lattice(name, editor.text())
            )
            crystal_layout.addWidget(label, 0, 2 * column)
            crystal_layout.addWidget(editor, 0, 2 * column + 1)
        spacegroup_label = QtWidgets.QLabel("Space group")
        spacegroup_tooltip = (
            "Hermann-Mauguin space group symbol (e.g. 'F d -3 m:2'). It will "
            "expand orbital-bearing sites and constrain onsite and hopping "
            "terms in later builder stages."
            if structure_only
            else "Hermann-Mauguin space group symbol (e.g. 'F d -3 m:2'). "
            "Used to expand the magnetic sites and to group bonds into "
            "symmetry-distinct orbits sharing one exchange constant."
        )
        spacegroup_label.setToolTip(spacegroup_tooltip)
        spacegroup_editor = QtWidgets.QLineEdit(str(crystal.get("spacegroup", "P 1")))
        spacegroup_editor.setObjectName("model_crystal_spacegroup")
        spacegroup_editor.setToolTip(spacegroup_tooltip)
        spacegroup_editor.editingFinished.connect(
            lambda editor=spacegroup_editor: self._set_model_crystal_spacegroup(editor.text())
        )
        crystal_layout.addWidget(spacegroup_label, 1, 0, 1, 2)
        crystal_layout.addWidget(spacegroup_editor, 1, 2, 1, 4)
        import_button = QtWidgets.QPushButton("Import CIF...")
        import_button.setObjectName("model_crystal_import_cif")
        import_button.setToolTip(
            "Load lattice, space group, and atomic sites from a CIF file into "
            "this model and its data group."
            + (
                " Sets periodic axes to all three lattice directions."
                if structure_only
                else " Clears previously generated bond orbits."
            )
        )
        import_button.clicked.connect(self._import_cif_into_selected_model)
        crystal_layout.addWidget(import_button, 1, 6, 1, 2)
        from_group_button = QtWidgets.QPushButton("Use group crystal")
        from_group_button.setObjectName("model_crystal_from_group")
        from_group_button.setToolTip(
            "Copy the crystal (lattice, space group, sites) stored on this "
            "dataset group into the model configuration."
        )
        from_group_button.clicked.connect(self._use_group_crystal_for_selected_model)
        crystal_layout.addWidget(from_group_button, 1, 8, 1, 2)
        geometry_button = QtWidgets.QPushButton("View model in 3D")
        geometry_button.setObjectName("model_geometry_viewer")
        geometry_button.setToolTip(
            "Inspect the unit cell, active and ghost sites, local orbital "
            "frames, orbital tokens, and representative or symmetry-equivalent "
            "hopping/exchange pathways in the shared model-geometry viewer."
        )
        geometry_button.clicked.connect(self._open_selected_model_geometry)
        crystal_layout.addWidget(geometry_button, 2, 0, 1, 3)
        geometry_script = QtWidgets.QPushButton("Copy 3D viewer script")
        geometry_script.setObjectName("model_geometry_script")
        geometry_script.setToolTip(
            "Copy editable Python that reconstructs the renderer-independent "
            "model-geometry scene and opens the same viewer without project widgets."
        )
        geometry_script.clicked.connect(self._copy_selected_model_geometry_script)
        crystal_layout.addWidget(geometry_script, 2, 3, 1, 3)
        self.model_parameter_layout.addWidget(crystal_group, 3, 0, 1, 4)

        sites_group = QtWidgets.QGroupBox("Atomic Sites")
        sites_group.setObjectName("model_crystal_sites_group")
        sites_layout = QtWidgets.QGridLayout(sites_group)
        magnetic_labels = {str(name) for name in model.config.get("magnetic_sites", [])}
        headers = (
            ("Label", "x", "y", "z", "Element", "")
            if structure_only
            else ("Label", "x", "y", "z", "Ion", "", "")
        )
        for header_column, header in enumerate(headers):
            if header:
                sites_layout.addWidget(QtWidgets.QLabel(header), 0, header_column)
        site_tooltip = (
            "Crystallographic site label, chemical element, and fractional "
            "coordinates. These sites become candidate orbital locations."
            if structure_only
            else "Wyckoff site of the crystal: label, fractional coordinates, "
            "and the magnetic ion for the <j0> form factor. Check 'Magnetic' "
            "to include the site in exchange-bond generation."
        )
        for index, site in enumerate(crystal["sites"]):
            row = index + 1
            label_editor = QtWidgets.QLineEdit(str(site.get("label", "")))
            label_editor.setObjectName(f"model_crystal_site_label_{index}")
            label_editor.setToolTip(site_tooltip)
            label_editor.editingFinished.connect(
                lambda index=index, editor=label_editor: self._set_model_crystal_site(index, "label", editor.text())
            )
            sites_layout.addWidget(label_editor, row, 0)
            position = site.get("position", [0.0, 0.0, 0.0])
            for axis in range(3):
                editor = QtWidgets.QLineEdit(_parameter_to_text(position[axis]))
                editor.setObjectName(f"model_crystal_site_{index}_{'xyz'[axis]}")
                editor.setToolTip(site_tooltip)
                editor.setMaximumWidth(70)
                editor.editingFinished.connect(
                    lambda index=index, axis=axis, editor=editor: self._set_model_crystal_site(index, axis, editor.text())
                )
                sites_layout.addWidget(editor, row, 1 + axis)
            if structure_only:
                element_editor = QtWidgets.QLineEdit(str(site.get("element", "")))
                element_editor.setObjectName(
                    f"model_crystal_site_element_{index}"
                )
                element_editor.setToolTip(site_tooltip)
                element_editor.setMaximumWidth(70)
                element_editor.editingFinished.connect(
                    lambda index=index, editor=element_editor: self._set_model_crystal_site(
                        index, "element", editor.text()
                    )
                )
                sites_layout.addWidget(element_editor, row, 4)
            else:
                ion_combo = QtWidgets.QComboBox()
                ion_combo.setObjectName(f"model_crystal_site_ion_{index}")
                ion_combo.setToolTip(
                    "Magnetic ion of this site; sets the tabulated <j0> form "
                    "factor key. '(none)' leaves the site without a form factor."
                )
                ion_combo.addItem("(none)", "")
                for ion in available_ions():
                    ion_combo.addItem(ion, ion)
                ion_combo.setCurrentIndex(
                    max(ion_combo.findData(str(site.get("ion", "") or "")), 0)
                )
                ion_combo.currentIndexChanged.connect(
                    lambda _index, index=index, combo=ion_combo: self._set_model_crystal_site(
                        index, "ion", str(combo.currentData() or "")
                    )
                )
                sites_layout.addWidget(ion_combo, row, 4)
                magnetic_check = QtWidgets.QCheckBox("Magnetic")
                magnetic_check.setObjectName(
                    f"model_crystal_site_magnetic_{index}"
                )
                magnetic_check.setToolTip(
                    "Include this site in magnetic-site expansion and exchange-"
                    "bond generation."
                )
                magnetic_check.setChecked(
                    str(site.get("label", "")) in magnetic_labels
                )
                magnetic_check.toggled.connect(
                    lambda checked, index=index: self._set_model_site_magnetic(
                        index, checked
                    )
                )
                sites_layout.addWidget(magnetic_check, row, 5)
            remove_button = QtWidgets.QPushButton("Remove")
            remove_button.setObjectName(f"model_crystal_site_remove_{index}")
            remove_button.setToolTip("Remove this atomic site from the crystal.")
            remove_button.clicked.connect(
                lambda _checked=False, index=index: self._remove_model_crystal_site(index)
            )
            sites_layout.addWidget(remove_button, row, 5 if structure_only else 6)
        add_site_button = QtWidgets.QPushButton("Add site")
        add_site_button.setObjectName("model_crystal_site_add")
        add_site_button.setToolTip("Append a new atomic site to the crystal.")
        add_site_button.clicked.connect(self._add_model_crystal_site)
        sites_layout.addWidget(add_site_button, len(crystal["sites"]) + 1, 0)
        if structure_only:
            status = QtWidgets.QLabel(
                "The crystal supplies candidate orbital sites. Add manifolds "
                "below to construct the Stage 3.2 onsite Hamiltonian."
            )
            status.setObjectName("tight_binding_structure_status")
            status.setToolTip(
                "Crystal geometry, orbital manifolds, onsite invariants, and "
                "the resolved canonical model are shared by the GUI and scripts."
            )
            status.setWordWrap(True)
            sites_layout.addWidget(
                status, len(crystal["sites"]) + 2, 0, 1, len(headers)
            )
        self.model_parameter_layout.addWidget(sites_group, 4, 0, 1, 4)

        if structure_only:
            return

        bonds_group = QtWidgets.QGroupBox("Exchange Bonds")
        bonds_group.setObjectName("model_bonds_group")
        bonds_layout = QtWidgets.QGridLayout(bonds_group)
        cutoff_label = QtWidgets.QLabel("Bond cutoff (A)")
        cutoff_tooltip = (
            "Maximum bond length in Angstrom when enumerating exchange "
            "paths. Each symmetry-distinct orbit within the cutoff becomes "
            "one exchange fit parameter (J1, J2, J3a, ...)."
        )
        cutoff_label.setToolTip(cutoff_tooltip)
        cutoff_editor = QtWidgets.QLineEdit(
            _parameter_to_text(model.config.get("bond_cutoff_angstrom", DEFAULT_BOND_CUTOFF_ANGSTROM))
        )
        cutoff_editor.setObjectName("model_bonds_cutoff")
        cutoff_editor.setToolTip(cutoff_tooltip)
        cutoff_editor.setMaximumWidth(70)
        cutoff_editor.editingFinished.connect(
            lambda editor=cutoff_editor: self._set_model_config_setting("bond_cutoff_angstrom", editor.text())
        )
        bonds_layout.addWidget(cutoff_label, 0, 0)
        bonds_layout.addWidget(cutoff_editor, 0, 1)
        generate_button = QtWidgets.QPushButton("Generate symmetry orbits")
        generate_button.setObjectName("model_bonds_generate")
        generate_button.setToolTip(
            "Expand the magnetic sites through the space group, enumerate "
            "bonds up to the cutoff, and group them into symmetry-distinct "
            "orbits. Each orbit becomes one exchange parameter; values of "
            "orbits whose labels persist are kept."
        )
        generate_button.clicked.connect(self._generate_selected_model_bond_orbits)
        bonds_layout.addWidget(generate_button, 0, 2)
        orbits = model.config.get("orbits") or []
        table = QtWidgets.QTableWidget(len(orbits), 4)
        table.setObjectName("model_bonds_table")
        table.setToolTip(
            "Symmetry-distinct bond orbits of the current crystal. Each row "
            "is one exchange fit parameter; multiplicity counts the bonds "
            "sharing that constant."
        )
        table.setHorizontalHeaderLabels(["Orbit", "Distance (A)", "Multiplicity", "Example bond"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, orbit in enumerate(orbits):
            bonds = orbit.get("bonds", [])
            example = ""
            if bonds:
                bond = bonds[0]
                example = (
                    f"site {bond.get('site_i')} -> site {bond.get('site_j')} "
                    f"+ {tuple(bond.get('offset', (0, 0, 0)))}"
                )
            for column, text in enumerate(
                (
                    str(orbit.get("label", "")),
                    _format_number(float(orbit.get("distance_angstrom", 0.0))),
                    str(len(bonds)),
                    example,
                )
            ):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        visible_rows = min(max(len(orbits), 4), 10)
        row_height = max(table.verticalHeader().defaultSectionSize(), 24)
        table_height = (
            table.horizontalHeader().height()
            + row_height * visible_rows
            + 2 * table.frameWidth()
            + 8
        )
        table.setMinimumHeight(table_height)
        table.setMaximumHeight(table_height)
        bonds_layout.addWidget(table, 1, 0, 1, 3)
        self.model_parameter_layout.addWidget(bonds_group, 5, 0, 1, 4)

        self._build_model_interactions_editor(model)

    def _build_model_interactions_editor(self, model: ModelComponentSpec) -> None:
        """Anisotropy / SIA / dipole / Zeeman toggles for the heisenberg_rpa model."""

        from PySide6 import QtWidgets

        group = QtWidgets.QGroupBox("Interactions")
        group.setObjectName("model_interactions_group")
        layout = QtWidgets.QGridLayout(group)

        has_orbits = bool(model.config.get("orbits"))
        has_sites = bool(model.config.get("magnetic_sites"))

        anisotropy = model.config.get("anisotropy") or {}
        n_aniso = sum(len(entry.get("basis", ())) for entry in anisotropy.values())
        sia = model.config.get("sia") or {}
        n_sia = sum(len(entry.get("basis", ())) for entry in sia.values())

        def add_toggle(row, kind, text, tooltip, checked, enabled, summary):
            box = QtWidgets.QCheckBox(text)
            box.setObjectName(f"model_interaction_{kind}")
            box.setToolTip(tooltip)
            box.setChecked(bool(checked))
            box.setEnabled(bool(enabled))
            box.toggled.connect(
                lambda state, k=kind: self._set_model_interaction(k, state)
            )
            layout.addWidget(box, row, 0)
            label = QtWidgets.QLabel(summary)
            label.setObjectName(f"model_interaction_{kind}_summary")
            label.setToolTip(tooltip)
            label.setWordWrap(True)
            layout.addWidget(label, row, 1)

        add_toggle(
            0,
            "anisotropy",
            "Anisotropic exchange",
            "Project the symmetry-allowed rank-2 exchange tensors (symmetric "
            "off-diagonal + Dzyaloshinskii-Moriya) of every bond orbit and fit "
            "one coefficient per allowed component (J1_S1, J1_D1, ...). The "
            "isotropic part stays the existing Heisenberg parameter. Requires "
            "generated bond orbits.",
            bool(anisotropy),
            has_orbits,
            (
                f"{n_aniso} tensor parameter(s) across {len(anisotropy)} orbit(s)."
                if anisotropy
                else (
                    "Enable to snapshot allowed exchange tensors."
                    if has_orbits
                    else "Generate bond orbits first."
                )
            ),
        )
        add_toggle(
            1,
            "sia",
            "Single-ion anisotropy",
            "Add the symmetry-allowed single-ion anisotropy (rank-2, "
            "traceless) for each magnetic site class, rotated per site by its "
            "generating operation. Fits one K coefficient per allowed "
            "component. Requires magnetic sites.",
            bool(sia),
            has_sites,
            (
                f"{n_sia} anisotropy parameter(s) across {len(sia)} site class(es)."
                if sia
                else (
                    "Enable to snapshot allowed on-site tensors."
                    if has_sites
                    else "Define magnetic sites first."
                )
            ),
        )
        add_toggle(
            2,
            "dipole",
            "Dipole-dipole (Ewald)",
            "Include long-range magnetic dipole-dipole coupling via Ewald "
            "summation. Fits one strength D_dip that multiplies the cached "
            "dipole tensor; its default is the physical (mu0/4pi)(g mu_B)^2 "
            "value. Pin (vary off) to keep the physical strength.",
            bool((model.config.get("dipole") or {}).get("enabled")),
            True,
            "Physical strength seeds D_dip; long-range, cached per geometry.",
        )
        add_toggle(
            3,
            "zeeman",
            "Zeeman (applied field)",
            "Enable the applied-field (Larmor) term. Uses the per-dataset "
            "magnetic field from Conditions and fits g_factor plus "
            "transverse chi/gamma ratios. Datasets fitted with this on must "
            "have a field set.",
            bool((model.config.get("zeeman") or {}).get("enabled")),
            True,
            "Needs a per-dataset field (Dataset details -> Conditions).",
        )

        self.model_parameter_layout.addWidget(group, 6, 0, 1, 4)

        self._build_model_closure_editor(model)

    def _build_model_closure_editor(self, model: ModelComponentSpec) -> None:
        """Self-consistency closure controls for the heisenberg_rpa model."""

        from PySide6 import QtWidgets

        closure = model.config.get("closure") or {}
        mode = str(closure.get("mode", "none")).lower()

        group = QtWidgets.QGroupBox("Self-consistency closure")
        group.setObjectName("model_closure_group")
        layout = QtWidgets.QGridLayout(group)

        mode_tooltip = (
            "Sum-rule closure that makes the local response self-consistent "
            "instead of freely fitted (see the theory notes). 'None' keeps the "
            "bare RPA. 'Onsager' solves a reaction field so the fluctuation "
            "moment hits a target; 'SCR' (Moriya) renormalizes chi0 through the "
            "mode-coupling u; 'TAC' (Takahashi) conserves the total zero-point "
            "plus thermal amplitude. Closures add their own fit parameters and "
            "use finite-difference gradients."
        )
        mode_label = QtWidgets.QLabel("Closure")
        mode_label.setToolTip(mode_tooltip)
        layout.addWidget(mode_label, 0, 0)
        mode_combo = QtWidgets.QComboBox()
        mode_combo.setObjectName("model_closure_mode")
        mode_combo.setToolTip(mode_tooltip)
        for label, value in (
            ("None (bare RPA)", "none"),
            ("Onsager reaction field", "onsager"),
            ("Moriya SCR", "scr"),
            ("Takahashi TAC", "tac"),
        ):
            mode_combo.addItem(label, value)
        mode_combo.setCurrentIndex(max(mode_combo.findData(mode), 0))
        mode_combo.currentIndexChanged.connect(
            lambda _idx, combo=mode_combo: self._set_model_closure(
                {"mode": combo.currentData()}
            )
        )
        layout.addWidget(mode_combo, 0, 1)

        if mode == "none":
            self.model_parameter_layout.addWidget(group, 7, 0, 1, 4)
            return

        def add_numeric(row, key, label_text, tooltip, value):
            label = QtWidgets.QLabel(label_text)
            label.setToolTip(tooltip)
            layout.addWidget(label, row, 0)
            editor = QtWidgets.QLineEdit(_parameter_to_text(value))
            editor.setObjectName(f"model_closure_{key}")
            editor.setToolTip(tooltip)
            editor.setMaximumWidth(90)
            editor.editingFinished.connect(
                lambda ed=editor, k=key: self._set_model_closure(
                    {k: _parse_parameter_text(ed.text())}
                )
            )
            layout.addWidget(editor, row, 1)

        add_numeric(
            1,
            "energy_cutoff_mev",
            "Energy cutoff (meV)",
            "Upper energy limit Lambda of the moment integral. Physically the "
            "bandwidth or crystal-field scale; the relaxational amplitude grows "
            "logarithmically with it, so it must be set deliberately.",
            closure.get("energy_cutoff_mev", 100.0),
        )
        if mode in ("onsager", "tac"):
            moment_tooltip = (
                "Expose the conserved moment budget as a model parameter "
                "(m2_total for Onsager or total_amplitude for TAC). When "
                "enabled, use its Fit checkbox in Fit Parameters above to let "
                "the optimizer vary it, or leave Fit unchecked to hold that "
                "parameter fixed. When disabled, the closure uses Target value."
            )
            moment_label = QtWidgets.QLabel("Fit moment target")
            moment_label.setToolTip(moment_tooltip)
            layout.addWidget(moment_label, 3, 0)
            moment_check = QtWidgets.QCheckBox("Expose fit parameter")
            moment_check.setObjectName("model_closure_moment_mode")
            moment_check.setToolTip(moment_tooltip)
            moment_check.setChecked(
                str(closure.get("moment_mode", "fixed")) == "fitted"
            )
            moment_check.toggled.connect(
                lambda checked: self._set_model_closure(
                    {"moment_mode": "fitted" if checked else "fixed"}
                )
            )
            layout.addWidget(moment_check, 3, 1)
            if str(closure.get("moment_mode", "fixed")) == "fixed":
                add_numeric(
                    4,
                    "moment_target",
                    "Target value",
                    "The conserved per-site amplitude <m^2> (Onsager) or total "
                    "zero-point + thermal amplitude (TAC), in model units. For a "
                    "rigid local moment this is ~ S(S+1).",
                    closure.get("moment_target", 1.0),
                )
        self.model_parameter_layout.addWidget(group, 7, 0, 1, 4)

    def _build_heisenberg_advanced_editor(
        self,
        model: ModelComponentSpec,
    ) -> None:
        """Build convergence controls used only by self-consistent RPA."""

        from PySide6 import QtWidgets

        closure = model.config.get("closure") or {}
        mode = str(closure.get("mode", "none")).lower()
        group = QtWidgets.QGroupBox("Numerical convergence")
        group.setObjectName("heisenberg_numerical_group")
        layout = QtWidgets.QFormLayout(group)

        if mode == "none":
            status = QtWidgets.QLabel(
                "Bare RPA evaluates the requested data points directly and "
                "does not use a Brillouin-zone integration grid."
            )
            status.setObjectName("heisenberg_numerical_status")
            status.setWordWrap(True)
            status.setToolTip(
                "A full-zone grid is introduced only by Onsager, SCR, or TAC "
                "self-consistency."
            )
            layout.addRow(status)
        else:
            grid_tooltip = (
                "Brillouin-zone sampling for the self-consistency integral, "
                "using an N x N x N grid over the reduced cell. Increase N "
                "until fitted values are converged; cost grows as N cubed."
            )
            grid_label = QtWidgets.QLabel("Closure BZ grid (N per axis)")
            grid_label.setToolTip(grid_tooltip)
            grid = QtWidgets.QLineEdit(
                _parameter_to_text(closure.get("bz_grid", 16))
            )
            grid.setObjectName("model_closure_bz_grid")
            grid.setToolTip(grid_tooltip)
            grid.setMaximumWidth(90)
            grid.editingFinished.connect(
                lambda editor=grid: self._set_model_closure(
                    {"bz_grid": _parse_parameter_text(editor.text())}
                )
            )
            layout.addRow(grid_label, grid)

            if (model.config.get("zeeman") or {}).get("enabled"):
                omega_tooltip = (
                    "Energy-quadrature points for the field-on closure moment "
                    "integral. Increase this together with the BZ grid until "
                    "the self-consistent result is stable."
                )
                omega_label = QtWidgets.QLabel("Field energy points")
                omega_label.setToolTip(omega_tooltip)
                omega = QtWidgets.QLineEdit(
                    _parameter_to_text(closure.get("omega_points", 200))
                )
                omega.setObjectName("model_closure_omega_points")
                omega.setToolTip(omega_tooltip)
                omega.setMaximumWidth(90)
                omega.editingFinished.connect(
                    lambda editor=omega: self._set_model_closure(
                        {"omega_points": _parse_parameter_text(editor.text())}
                    )
                )
                layout.addRow(omega_label, omega)

        self.model_parameter_layout.addWidget(group, 8, 0, 1, 4)

    def _add_tight_binding_manifold(
        self,
        site_label: str,
        preset: str,
        submanifold_id: str | None = None,
    ) -> None:
        from .electronic_builder import (
            add_tight_binding_orbital_manifold,
            orbital_manifold_from_site_symmetry,
            orbital_manifold_preset,
        )

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if model.type != "tight_binding":
                raise ValueError("orbital manifolds require a tight-binding model")
            if not site_label:
                raise ValueError("select a crystallographic site first")
            base = (
                f"{site_label}_{preset}_subspace"
                if submanifold_id
                else f"{site_label}_{preset}"
            )
            existing = {
                str(item.get("label", ""))
                for item in model.config.get("orbital_manifolds", ())
            }
            label = base
            suffix = 2
            while label in existing:
                label = f"{base}_{suffix}"
                suffix += 1
            add_tight_binding_orbital_manifold(
                model,
                (
                    orbital_manifold_from_site_symmetry(
                        model_crystal_config(model),
                        site_label,
                        preset,
                        submanifold_id,
                        label=label,
                    )
                    if submanifold_id
                    else orbital_manifold_preset(
                        site_label,
                        preset,
                        label=label,
                    )
                ),
            )

        self._mutate_selected_model(mutate)

    def _set_tight_binding_manifold_field(
        self, index: int, field: str, text: str
    ) -> None:
        from .electronic_builder import (
            OrbitalManifold,
            set_tight_binding_orbital_manifolds,
        )

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            payloads = copy.deepcopy(model.config.get("orbital_manifolds", ()))
            if not (0 <= int(index) < len(payloads)):
                raise _NoChange()
            payload = payloads[int(index)]
            if field in {"label", "correlated_shell"}:
                value: Any = text.strip()
            else:
                value = _parse_parameter_text(text)
                if field == "orbitals" and not isinstance(value, list):
                    raise ValueError("custom orbitals must be entered as a JSON list")
                if field == "local_frame" and (
                    not isinstance(value, list)
                    or len(value) != 3
                    or any(
                        not isinstance(row, list) or len(row) != 3
                        for row in value
                    )
                ):
                    raise ValueError("local_frame must be a 3x3 JSON matrix")
                if field == "degeneracy_groups" and not isinstance(value, list):
                    raise ValueError(
                        "degeneracy_groups must be a JSON list of orbital-label lists"
                    )
                if field == "magnetic_form_factors" and not isinstance(value, dict):
                    raise ValueError(
                        "magnetic_form_factors must be a JSON mapping by orbital label"
                    )
            if payload.get(field) == value:
                raise _NoChange()
            payload[field] = value
            set_tight_binding_orbital_manifolds(
                model, [OrbitalManifold.from_dict(item) for item in payloads]
            )

        self._mutate_selected_model_quietly(mutate)

    def _remove_tight_binding_manifold(self, label: str) -> None:
        from .electronic_builder import remove_tight_binding_orbital_manifold

        self._mutate_selected_model(
            lambda model, _group: remove_tight_binding_orbital_manifold(
                model, label
            )
        )

    def _set_tight_binding_spin_treatment(self, treatment: str) -> None:
        from .electronic_builder import set_tight_binding_spin_treatment

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_spin_treatment(
                model,
                treatment,
            )
        )

    def _set_tight_binding_use_primitive_cell(self, enabled: bool) -> None:
        from .electronic_builder import invalidate_tight_binding_model

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            selected = bool(enabled)
            if bool(model.config.get("use_primitive_cell", True)) == selected:
                raise _NoChange()
            model.config["use_primitive_cell"] = selected
            invalidate_tight_binding_model(model)

        self._mutate_selected_model_quietly(
            mutate,
            rebuild_editor=False,
        )

    def _generate_tight_binding_standard_path(
        self,
        convention: str,
    ) -> None:
        from .brillouin_zone import set_tight_binding_standard_path

        if convention == "manual":
            def select_manual(
                model: ModelComponentSpec,
                _group: DataGroup | None,
            ) -> None:
                model.config["band_path_convention"] = "manual"
                model.config["band_path_metadata"] = {}

            self._mutate_selected_model(
                select_manual,
                rebuild_editor=False,
            )
            return
        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_standard_path(
                model,
                convention,
            ),
            rebuild_editor=False,
        )

    def _toggle_tight_binding_soc(
        self,
        manifold_label: str,
        enabled: bool,
    ) -> None:
        from .electronic_builder import set_tight_binding_soc_term

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_soc_term(
                model,
                manifold_label,
                enabled=enabled,
            )
        )

    def _set_tight_binding_soc_field(
        self,
        manifold_label: str,
        field: str,
        text: str,
    ) -> None:
        from .electronic_builder import set_tight_binding_soc_term

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if field == "prescription":
                set_tight_binding_soc_term(
                    model,
                    manifold_label,
                    prescription=text,
                )
                return
            if field == "value":
                set_tight_binding_soc_term(
                    model,
                    manifold_label,
                    value=float(_parse_parameter_text(text)),
                )
                return
            bound = (
                None if not text.strip() else float(_parse_parameter_text(text))
            )
            set_tight_binding_soc_term(
                model,
                manifold_label,
                **{field: bound},
            )

        self._mutate_selected_model(mutate, rebuild_editor=False)

    def _set_tight_binding_soc_fit(
        self,
        manifold_label: str,
        checked: bool,
    ) -> None:
        from .electronic_builder import set_tight_binding_soc_term

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_soc_term(
                model,
                manifold_label,
                fit=checked,
            ),
            rebuild_editor=False,
        )

    def _set_tight_binding_onsite_field(
        self, identifier: str, field: str, text: str
    ) -> None:
        from .electronic_builder import set_tight_binding_onsite_term

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if field == "value":
                set_tight_binding_onsite_term(
                    model,
                    identifier,
                    value=float(_parse_parameter_text(text)),
                )
                return
            value = (
                None if not text.strip() else float(_parse_parameter_text(text))
            )
            set_tight_binding_onsite_term(
                model,
                identifier,
                **{field: value},
            )

        self._mutate_selected_model(mutate, rebuild_editor=False)

    def _set_tight_binding_onsite_fit(
        self, identifier: str, checked: bool
    ) -> None:
        from .electronic_builder import set_tight_binding_onsite_term

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_onsite_term(
                model, identifier, fit=bool(checked)
            ),
            rebuild_editor=False,
        )

    def _regenerate_tight_binding_hoppings(self, text: str) -> None:
        from .electronic_builder import (
            regenerate_tight_binding_hopping_terms,
        )

        self._mutate_selected_model(
            lambda model, _group: regenerate_tight_binding_hopping_terms(
                model,
                float(_parse_parameter_text(text)),
            )
        )

    def _set_tight_binding_hopping_parameterization(
        self,
        parameterization: str,
    ) -> None:
        from .electronic_builder import (
            set_tight_binding_hopping_parameterization,
        )

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_hopping_parameterization(
                model,
                parameterization,
            )
        )

    def _add_selected_tight_binding_hoppings(self, table: Any) -> None:
        from PySide6 import QtCore

        from .electronic_builder import add_tight_binding_hopping_term

        identifiers = {
            str(
                table.item(index.row(), 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
            )
            for index in table.selectionModel().selectedRows()
            if table.item(index.row(), 0) is not None
        }
        if not identifiers:
            return

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            for identifier in sorted(identifiers):
                add_tight_binding_hopping_term(model, identifier)

        self._mutate_selected_model(mutate)

    def _remove_tight_binding_hopping(self, identifier: str) -> None:
        from .electronic_builder import remove_tight_binding_hopping_term

        self._mutate_selected_model(
            lambda model, _group: remove_tight_binding_hopping_term(
                model,
                identifier,
            )
        )

    def _set_tight_binding_hopping_field(
        self, identifier: str, field: str, text: str
    ) -> None:
        from .electronic_builder import set_tight_binding_hopping_term

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if field == "value":
                set_tight_binding_hopping_term(
                    model,
                    identifier,
                    value=float(_parse_parameter_text(text)),
                )
                return
            value = (
                None if not text.strip() else float(_parse_parameter_text(text))
            )
            set_tight_binding_hopping_term(
                model,
                identifier,
                **{field: value},
            )

        self._mutate_selected_model(mutate, rebuild_editor=False)

    def _set_tight_binding_hopping_fit(
        self, identifier: str, checked: bool
    ) -> None:
        from .electronic_builder import set_tight_binding_hopping_term

        self._mutate_selected_model(
            lambda model, _group: set_tight_binding_hopping_term(
                model,
                identifier,
                fit=bool(checked),
            ),
            rebuild_editor=False,
        )

    def _open_selected_model_geometry(self) -> bool:
        from PySide6 import QtWidgets

        from .qt_model_geometry_viewer import open_model_geometry_viewer

        _group, model = self._selected_model_and_group()
        if model is None:
            return False
        try:
            window = open_model_geometry_viewer(model, parent=self.window)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Model geometry",
                f"Could not open the 3D model viewer:\n{exc}",
            )
            return False
        self._plot_windows[f"model:{id(model)}:geometry"] = window
        return True

    def _copy_selected_model_geometry_script(self) -> bool:
        from PySide6 import QtWidgets

        from .model_geometry import model_geometry_script

        _group, model = self._selected_model_and_group()
        if model is None:
            return False
        try:
            script = model_geometry_script(model)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Model geometry script",
                f"Could not create the viewer script:\n{exc}",
            )
            return False
        QtWidgets.QApplication.clipboard().setText(script)
        return True

    def _set_model_interaction(self, kind: str, enabled: bool) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if kind == "anisotropy":
                set_model_anisotropic_exchange(model, enabled)
            elif kind == "sia":
                set_model_single_ion_anisotropy(model, enabled)
            elif kind == "dipole":
                set_model_dipole(model, enabled)
            elif kind == "zeeman":
                set_model_zeeman(model, enabled)

        self._mutate_selected_model(mutate)

    def _set_model_closure(self, updates: dict[str, Any]) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            set_model_closure(model, updates)

        self._mutate_selected_model(mutate)

    def _selected_model_and_group(self) -> tuple[DataGroup | None, ModelComponentSpec | None]:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return None, None
        return group, model

    def _mutate_selected_model(
        self,
        mutate,
        *,
        rebuild_editor: bool = True,
    ) -> None:
        """Apply a config mutation to the selected model and refresh the editor."""

        group, model = self._selected_model_and_group()
        if model is None:
            return
        try:
            mutate(model, group)
        except (ValueError, ImportError, KeyError) as exc:
            from PySide6 import QtWidgets

            QtWidgets.QMessageBox.warning(self.window, "Model configuration", str(exc))
            return
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if rebuild_editor:
            self._rebuild_model_parameter_editor_preserving_scroll(model)
        if group is not None:
            if branch_created:
                self._refresh_tree(select_group=group, select_model=model)
            self._request_overlay_refresh(group)

    def _set_model_crystal_lattice(self, name: str, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            lattice = crystal["lattice"]
            value = float(_parse_parameter_text(text))
            if lattice.get(name) == value:
                raise _NoChange()
            lattice[name] = value
            _mark_tight_binding_crystal_manual(model, crystal)
            _refresh_tight_binding_builder(model)

        self._mutate_selected_model_quietly(mutate)

    def _set_model_crystal_spacegroup(self, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            value = text.strip() or "P 1"
            if crystal.get("spacegroup") == value:
                raise _NoChange()
            crystal["spacegroup"] = value
            _mark_tight_binding_crystal_manual(model, crystal)
            _refresh_tight_binding_builder(model)

        self._mutate_selected_model_quietly(mutate)

    def _set_model_crystal_site(self, index: int, field: Any, text: str) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            sites = crystal["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            site = sites[index]
            if field == "label":
                value = text.strip()
                if site.get("label") == value:
                    raise _NoChange()
                previous_label = str(site.get("label", ""))
                if model.type == "heisenberg_rpa":
                    magnetic = [
                        str(name)
                        for name in model.config.get("magnetic_sites", [])
                    ]
                    model.config["magnetic_sites"] = [
                        value if name == str(site.get("label", "")) else name
                        for name in magnetic
                    ]
                elif model.type == "tight_binding":
                    for manifold in model.config.get("orbital_manifolds", ()):
                        if str(manifold.get("site_label", "")) == previous_label:
                            manifold["site_label"] = value
                site["label"] = value
            elif field in {"ion", "element"}:
                value = text.strip()
                if site.get(str(field), "") == value:
                    raise _NoChange()
                site[str(field)] = value
            else:
                position = list(site.get("position", [0.0, 0.0, 0.0]))
                value = float(_parse_parameter_text(text))
                if position[int(field)] == value:
                    raise _NoChange()
                position[int(field)] = value
                site["position"] = position
            _mark_tight_binding_crystal_manual(model, crystal)
            _refresh_tight_binding_builder(model)

        self._mutate_selected_model_quietly(mutate)

    def _set_model_site_magnetic(self, index: int, checked: bool) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            sites = model_crystal_config(model)["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            label = str(sites[index].get("label", ""))
            magnetic = [str(name) for name in model.config.get("magnetic_sites", [])]
            if bool(checked) == (label in magnetic):
                raise _NoChange()
            if checked:
                magnetic.append(label)
            else:
                magnetic = [name for name in magnetic if name != label]
            model.config["magnetic_sites"] = magnetic

        self._mutate_selected_model_quietly(mutate)

    def _add_model_crystal_site(self) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            sites = crystal["sites"]
            sites.append(
                {
                    "label": f"Site{len(sites) + 1}",
                    "element": "",
                    "position": [0.0, 0.0, 0.0],
                    "ion": "",
                }
            )
            _mark_tight_binding_crystal_manual(model, crystal)

        self._mutate_selected_model(mutate)

    def _remove_model_crystal_site(self, index: int) -> None:
        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            crystal = model_crystal_config(model)
            sites = crystal["sites"]
            if not (0 <= index < len(sites)):
                raise _NoChange()
            removed = sites.pop(index)
            label = str(removed.get("label", ""))
            if model.type == "heisenberg_rpa":
                model.config["magnetic_sites"] = [
                    str(name)
                    for name in model.config.get("magnetic_sites", [])
                    if str(name) != label
                ]
            elif model.type == "tight_binding":
                from .electronic_builder import set_tight_binding_orbital_manifolds

                retained = [
                    item
                    for item in model.config.get("orbital_manifolds", ())
                    if str(item.get("site_label", "")) != label
                ]
                set_tight_binding_orbital_manifolds(model, retained)
            _mark_tight_binding_crystal_manual(model, crystal)

        self._mutate_selected_model(mutate)

    def _import_cif_into_selected_model(self) -> None:

        group, model = self._selected_model_and_group()
        if model is None:
            return
        path, _selected = get_open_file_name(
            self.window, "Import CIF", "", "CIF files (*.cif);;All files (*)"
        )
        if not path:
            return

        def mutate(model: ModelComponentSpec, group: DataGroup | None) -> None:
            import_cif_into_model(model, path, group=group)

        self._mutate_selected_model(mutate)

    def _use_group_crystal_for_selected_model(self) -> None:
        def mutate(model: ModelComponentSpec, group: DataGroup | None) -> None:
            if group is None:
                raise ValueError("the model is not attached to a data group")
            stored = group.metadata.get("crystal")
            if isinstance(stored, dict) and stored.get("sites"):
                set_model_crystal(model, stored)
            elif isinstance(group.lattice_parameters, dict):
                crystal = copy.deepcopy(model_crystal_config(model))
                crystal["lattice"] = {
                    name: float(group.lattice_parameters.get(name, fallback))
                    for name, fallback in (
                        ("a", 5.0), ("b", 5.0), ("c", 5.0),
                        ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0),
                    )
                }
                if group.spacegroup:
                    crystal["spacegroup"] = str(group.spacegroup)
                set_model_crystal(model, crystal)
            else:
                raise ValueError(
                    "the data group stores no crystal information; import a "
                    "CIF or set lattice parameters on the group first"
                )

        self._mutate_selected_model(mutate)

    def _generate_selected_model_bond_orbits(self) -> None:
        from PySide6 import QtWidgets

        cutoff_editor = self.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_bonds_cutoff"
        )

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if cutoff_editor is not None:
                model.config["bond_cutoff_angstrom"] = _parse_parameter_text(
                    cutoff_editor.text()
                )
            generate_model_bond_orbits(model)

        self._mutate_selected_model(mutate)

    def _mutate_selected_model_quietly(
        self,
        mutate,
        *,
        rebuild_editor: bool = True,
    ) -> None:
        """Like ``_mutate_selected_model`` but a ``_NoChange`` is not an error."""

        def wrapped(model: ModelComponentSpec, group: DataGroup | None) -> None:
            mutate(model, group)

        try:
            self._mutate_selected_model(
                wrapped,
                rebuild_editor=rebuild_editor,
            )
        except _NoChange:
            return

    def _set_model_parameter(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = _parse_parameter_text(text)
        if model.parameters.get(name) == value:
            return
        model.parameters[name] = value
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(
                select_group=group,
                select_model=model,
                refresh_viewers=False,
            )
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_parameter_plot_label(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        labels = dict(model.metadata.get("parameter_labels", {}))
        value = text.strip()
        if value:
            labels[name] = value
        else:
            labels.pop(name, None)
        if model.metadata.get("parameter_labels", {}) != labels:
            if labels:
                model.metadata["parameter_labels"] = labels
            else:
                model.metadata.pop("parameter_labels", None)
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)

    def _set_model_config_setting(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = _parse_parameter_text(text)
        if model.config.get(name) != value:
            model.config[name] = value
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_lindhard_config_values(self, **updates: Any) -> None:
        """Apply ordinary Lindhard choices and refresh conditional controls."""

        def mutate(
            model: ModelComponentSpec,
            _group: DataGroup | None,
        ) -> None:
            if model.type != "lindhard":
                raise ValueError("expected a Lindhard response component")
            changed_updates = {
                name: value
                for name, value in updates.items()
                if model.config.get(name) != value
            }
            if not changed_updates:
                raise _NoChange()
            from .electronic_normalization import (
                LINDHARD_COUPLING_FIELDS,
                configure_lindhard_experimental_coupling,
            )

            coupling = {
                name: value
                for name, value in changed_updates.items()
                if name in LINDHARD_COUPLING_FIELDS
            }
            ordinary = {
                name: value
                for name, value in changed_updates.items()
                if name not in LINDHARD_COUPLING_FIELDS
            }
            if coupling:
                configure_lindhard_experimental_coupling(model, **coupling)
            model.config.update(ordinary)

        self._mutate_selected_model_quietly(mutate)

    def _set_lindhard_coupling_text(self, name: str, text: str) -> None:
        """Parse one text control through the public coupling configurator."""

        self._set_lindhard_config_values(**{name: _parse_parameter_text(text)})

    def _set_lindhard_q_accuracy(self, accuracy: str) -> None:
        """Select exact or validated automatic Q evaluation as one policy."""

        if accuracy == "advanced":
            return
        if accuracy == "exact":
            updates = {
                "response_q_evaluation": "auto",
                "response_q_interpolation_rtol": 0.0,
                "response_q_interpolation_atol": 0.0,
            }
        elif accuracy == "validated":
            updates = {
                "response_q_evaluation": "auto",
                "response_q_interpolation_rtol": 0.01,
            }
        else:
            raise ValueError(f"unknown Lindhard Q accuracy {accuracy!r}")
        self._set_lindhard_config_values(**updates)

    def _certify_lindhard_sampling(
        self,
        expected_model: ModelComponentSpec,
    ) -> None:
        """Run the explicit pre-fit Lindhard mesh search."""

        from PySide6 import QtWidgets

        from .qt_sampling_progress import SamplingProgressDialog

        result: dict[str, Any] = {}
        progress = SamplingProgressDialog(
            "Lindhard mesh convergence",
            parent=self.window,
        )
        try:
            def mutate(
                model: ModelComponentSpec,
                group: DataGroup | None,
            ) -> None:
                if model is not expected_model or group is None:
                    raise ValueError(
                        "select the Lindhard component before certification"
                    )
                result["certificate"] = certify_group_lindhard_sampling(
                    group,
                    model,
                    progress_callback=progress.update,
                )

            self._mutate_selected_model(mutate)
        except Exception as exc:
            progress.fail(str(exc))
            QtWidgets.QMessageBox.warning(
                self.window,
                "Response-mesh convergence",
                f"Could not certify the response mesh:\n{exc}",
            )
            return
        certificate = result.get("certificate")
        if certificate is not None:
            progress.finish(certificate)
        if certificate is not None and not certificate.certified:
            QtWidgets.QMessageBox.information(
                self.window,
                "Response-mesh convergence",
                "The requested accuracy was not certified within the "
                "configured mesh budget. The attempted meshes and errors "
                "were saved; increase the Advanced budget or loosen the "
                "accuracy profile.",
            )

    def _certify_tight_binding_dos_sampling(
        self,
        expected_model: ModelComponentSpec,
    ) -> None:
        """Run the explicit model-level DOS mesh search."""

        from PySide6 import QtWidgets

        from .model_plots import certify_tight_binding_dos_sampling
        from .qt_sampling_progress import SamplingProgressDialog

        result: dict[str, Any] = {}
        progress = SamplingProgressDialog(
            "Density-of-states mesh convergence",
            parent=self.window,
        )
        try:
            def mutate(
                model: ModelComponentSpec,
                _group: DataGroup | None,
            ) -> None:
                if model is not expected_model or model.type != "tight_binding":
                    raise ValueError(
                        "select the tight-binding component before certification"
                    )
                result["certificate"] = certify_tight_binding_dos_sampling(
                    model,
                    progress_callback=progress.update,
                )

            self._mutate_selected_model(mutate)
        except Exception as exc:
            progress.fail(str(exc))
            QtWidgets.QMessageBox.warning(
                self.window,
                "DOS mesh convergence",
                f"Could not certify the DOS mesh:\n{exc}",
            )
            return
        certificate = result.get("certificate")
        if certificate is not None:
            progress.finish(certificate)
        if certificate is not None and not certificate.certified:
            QtWidgets.QMessageBox.information(
                self.window,
                "DOS mesh convergence",
                "The requested accuracy was not certified within the "
                "configured mesh budget. The attempted meshes and errors "
                "were saved; increase the Advanced budget or loosen the "
                "accuracy profile.",
            )

    def _set_tight_binding_energy_unit(self, unit: str) -> None:
        from .electronic_structure import (
            normalize_electronic_energy_unit,
            set_electronic_energy_unit,
        )

        canonical = normalize_electronic_energy_unit(unit)

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if model.type != "tight_binding":
                raise ValueError("electronic energy units apply only to tight binding")
            if model.config.get("electronic_energy_unit", "eV") == canonical:
                raise _NoChange()
            set_electronic_energy_unit(model, canonical)

        self._mutate_selected_model_quietly(mutate)

    def _set_tight_binding_energy_config(self, name: str, text: str) -> None:
        from .electronic_structure import (
            electronic_energy_to_meV,
            normalize_electronic_energy_unit,
        )

        def mutate(model: ModelComponentSpec, _group: DataGroup | None) -> None:
            if model.type != "tight_binding" or not name.endswith("_meV"):
                raise ValueError("expected a canonical tight-binding energy setting")
            unit = normalize_electronic_energy_unit(
                model.config.get("electronic_energy_unit", "eV")
            )
            displayed = float(_parse_parameter_text(text))
            canonical = float(electronic_energy_to_meV(displayed, unit))
            if model.config.get(name) == canonical:
                raise _NoChange()
            model.config[name] = canonical

        self._mutate_selected_model_quietly(mutate)

    def _set_model_form_factor_choice(self, choice: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        choice = str(choice or "")
        changed = False
        if choice == EFFECTIVE_FORM_FACTOR_CHOICE:
            if not model.config.get("form_factor_mixture"):
                prior_ion = str(model.config.get("ion", "") or "").strip()
                if prior_ion in {"", CUSTOM_FORM_FACTOR_CHOICE}:
                    prior_ion = "Fe2"
                model.config["form_factor_mixture"] = [
                    {"ion": prior_ion, "weight": 1.0}
                ]
                changed = True
        elif choice == CUSTOM_FORM_FACTOR_CHOICE:
            if model.config.get("ion") != CUSTOM_FORM_FACTOR_CHOICE:
                model.config["ion"] = CUSTOM_FORM_FACTOR_CHOICE
                changed = True
            model.config.setdefault("form_factor_coefficients", "")
        else:
            if model.config.get("ion") != choice:
                model.config["ion"] = choice
                changed = True
            if model.config.get("form_factor_coefficients"):
                model.config["form_factor_coefficients"] = ""
                changed = True
        if changed:
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            self._rebuild_model_parameter_editor_preserving_scroll(model)
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_sharing_mode(self, name: str, mode: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        if mode not in SHARING_MODES:
            return
        current = model.sharing.get(name)
        entry = dict(current) if isinstance(current, dict) else {}
        if entry.get("mode") != mode:
            entry["mode"] = mode
            entry.setdefault("groups", {})
            model.sharing[name] = entry
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            self._rebuild_model_parameter_editor(model)
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_sharing_groups(self, name: str, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        groups: dict[str, str] = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                return
            dataset, tie_group = (piece.strip() for piece in item.split("=", 1))
            if not dataset or not tie_group:
                return
            groups[dataset] = tie_group
        current = model.sharing.get(name)
        entry = dict(current) if isinstance(current, dict) else {}
        if entry.get("groups", {}) != groups:
            entry["mode"] = "grouped"
            entry["groups"] = groups
            model.sharing[name] = entry
            branch_created = self._record_data_group_state_change(group) if group is not None else False
            self._mark_dirty()
            if group is not None and branch_created:
                self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_fit_parameter(self, name: str, checked: bool) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        value = bool(checked)
        if model.fit_parameters.get(name) == value:
            return
        model.fit_parameters[name] = value
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            # Fit selection changes optimizer configuration only. They do not
            # change the model evaluated at its current parameter values, so
            # keep an already-rendered data-viewer overlay intact.
            self._refresh_tree(
                select_group=group,
                select_model=model,
                refresh_viewers=False,
            )

    def _set_model_limit(self, name: str, side: int, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        raw = _parse_parameter_text(text.strip()) if text.strip() else None
        value = None if raw in (None, "") else raw
        current = model.limits.get(name)
        limits = list(current) if isinstance(current, (list, tuple)) and len(current) == 2 else [None, None]
        if limits[side] == value:
            return
        limits[side] = value
        if limits == [None, None]:
            model.limits.pop(name, None)
        else:
            model.limits[name] = limits
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def _set_model_applies_to(self, text: str) -> None:
        group, _entry, _mask, model, role = self._objects_for_item(self._current_item())
        if role != "model" or model is None:
            return
        names = [part.strip() for part in text.split(",") if part.strip()]
        value = names or None
        if model.applies_to == value:
            return
        model.applies_to = value
        branch_created = self._record_data_group_state_change(group) if group is not None else False
        self._mark_dirty()
        if group is not None and branch_created:
            self._refresh_tree(select_group=group, select_model=model)
        if group is not None:
            self._request_overlay_refresh(group)

    def refresh_open_slice_viewers(self) -> None:
        # Debounced: a tree refresh (selection change, timeline switch, fit
        # completion, ...) schedules the overlay recompute instead of blocking
        # on it. In headless/test use this runs synchronously.
        for group in list(self.project.data_groups):
            self._request_overlay_refresh(group)

    def _create_slice_viewer(
        self,
        group: DataGroup,
        datasets: list[MDHistoData],
        names: list[str],
    ) -> Any:
        global QtMDHistoSliceViewer

        if QtMDHistoSliceViewer is None:
            from .qt_slice_viewer import QtMDHistoSliceViewer as viewer_class

            QtMDHistoSliceViewer = viewer_class
        group_keys = _waterfall_group_keys(group, names)
        viewer = QtMDHistoSliceViewer(
            datasets,
            dataset_names=names,
            dataset_group_keys=group_keys,
        )
        if hasattr(viewer, "crystal_contexts"):
            viewer.crystal_contexts = [
                {
                    "spacegroup": group.spacegroup,
                    "lattice_parameters": copy.deepcopy(group.lattice_parameters),
                }
                for _dataset in datasets
            ]
        if hasattr(viewer, "set_brillouin_zone_context_callback"):
            def persist_crystal_context(_dataset_name, context, *, group=group):
                group.spacegroup = str(context.get("spacegroup", "")).strip() or None
                crystal = group.metadata.get("crystal")
                if isinstance(crystal, dict) and group.spacegroup:
                    crystal["spacegroup"] = group.spacegroup
                lattice = context.get("lattice_parameters")
                if isinstance(lattice, dict):
                    group.lattice_parameters = {
                        key: float(value) for key, value in lattice.items()
                    }
                self._record_data_group_state_change(group)
                self._mark_dirty()
                self._sync_details()

            viewer.set_brillouin_zone_context_callback(persist_crystal_context)
        self._slice_viewers.setdefault(id(group), []).append(viewer)
        if hasattr(viewer, "set_close_callback"):
            viewer.set_close_callback(
                lambda group=group, viewer=viewer: self._forget_slice_viewer(group, viewer)
            )
        if viewer.window is not None:
            viewer.window.setWindowTitle(f"nfit Data Viewer - {group.name}")
        return viewer

    def _forget_slice_viewer(
        self,
        group: DataGroup,
        viewer: Any,
        *,
        close: bool = False,
    ) -> None:
        viewers = self._slice_viewers.get(id(group))
        if viewers is not None and viewer in viewers:
            viewers.remove(viewer)
            if not viewers:
                self._slice_viewers.pop(id(group), None)
        if close and viewer is not None and viewer.window is not None:
            viewer.window.close()

    def _close_slice_viewer(self, group: DataGroup) -> None:
        viewers = self._slice_viewers.pop(id(group), [])
        for viewer in viewers:
            if viewer is not None and viewer.window is not None:
                viewer.window.close()

    def _close_all_slice_viewers(self) -> None:
        for group_id in list(self._slice_viewers):
            viewers = self._slice_viewers.pop(group_id)
            for viewer in viewers:
                if viewer is not None and viewer.window is not None:
                    viewer.window.close()
        for window_id in list(self._auxiliary_windows):
            window = self._auxiliary_windows.pop(window_id)
            if window is not None:
                window.close()


def _make_project_window_class():
    from PySide6 import QtWidgets

    class ProjectWindow(QtWidgets.QMainWindow):
        def __init__(self, explorer: NfitProjectExplorer) -> None:
            super().__init__()
            self.explorer = explorer

        def closeEvent(self, event):
            self.explorer._handle_window_close(event)

    return ProjectWindow


def _make_project_tree_class():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ProjectTree(QtWidgets.QTreeWidget):
        def __init__(self, explorer: NfitProjectExplorer) -> None:
            super().__init__()
            self.explorer = explorer

        def contextMenuEvent(self, event):
            self.explorer._show_context_menu(self.itemAt(event.pos()), event.globalPos())

        def mousePressEvent(self, event):
            item = self.itemAt(event.position().toPoint())
            if (
                event.button() == QtCore.Qt.MouseButton.RightButton
                and item is not None
                and item in self.selectedItems()
            ):
                self.selectionModel().setCurrentIndex(
                    self.indexFromItem(item),
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )
                event.accept()
                return
            super().mousePressEvent(event)

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_Delete:
                self.explorer.delete_selected()
                return
            if event.matches(QtGui.QKeySequence.StandardKey.Copy):
                self.explorer.copy_selected()
                return
            if event.matches(QtGui.QKeySequence.StandardKey.Paste):
                self.explorer.paste_into_selection()
                return
            super().keyPressEvent(event)

        def startDrag(self, supported_actions):
            item = self.currentItem()
            if item is None:
                return
            role = self.explorer._objects_for_item(item)[4]
            if role not in {"group", "dataset", "mask", "dataset_group"}:
                return
            selected_roles = {self.explorer._objects_for_item(it)[4] for it in self.selectedItems()}
            if item not in self.selectedItems():
                selected_roles = {role}
            if role in {"group", "dataset", "mask"} and not selected_roles.issubset({role}):
                return
            if role == "dataset_group" and selected_roles - {"dataset_group"}:
                return
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setData("application/x-nfit-tree-item", role.encode("utf-8"))
            drag.setMimeData(mime_data)
            drag.exec(QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction)

        def dragEnterEvent(self, event):
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragEnterEvent(event)

        def dragMoveEvent(self, event):
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
                event.acceptProposedAction()
                return
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            super().dragMoveEvent(event)

        def dropEvent(self, event):
            # Internal item drags: always handled here, never by the default
            # QTreeWidget item-move machinery.
            if event.mimeData().hasFormat("application/x-nfit-tree-item"):
                target = self.itemAt(event.position().toPoint())
                copied = event.dropAction() == QtCore.Qt.DropAction.CopyAction
                if (
                    target is not None
                    and self.explorer.move_or_copy_selected_to_item(
                        target,
                        copy_item=copied,
                        drop_position=self.dropIndicatorPosition(),
                    )
                ):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return
            if event.mimeData().hasUrls():
                target_item = self.itemAt(event.position().toPoint())
                paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
                if paths and self.explorer.load_dropped_paths(paths, target_item):
                    event.acceptProposedAction()
                    return
                event.ignore()
                return
            super().dropEvent(event)

    return ProjectTree


def _bold_label(QtWidgets: Any, text: str) -> Any:
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("font-weight: 600")
    return label


def _compact_point_list_form(box: Any, QtWidgets: Any) -> None:
    """Let point-list controls shrink inside the details scroll area."""

    ignored = QtWidgets.QSizePolicy.Policy.Ignored
    preferred = QtWidgets.QSizePolicy.Policy.Preferred
    for combo in box.findChildren(QtWidgets.QComboBox):
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(10)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(ignored, preferred)
    for edit in box.findChildren(QtWidgets.QLineEdit):
        edit.setMinimumWidth(0)
        edit.setSizePolicy(ignored, preferred)


def _is_renameable_role(role: str) -> bool:
    return role in {"group", "dataset", "mask", "model", "fit", "fit_timeline", "dataset_group", "group_mask", "plot"}


def _dataset_source_path(dataset: DatasetEntry) -> Path | None:
    if not isinstance(dataset.metadata, dict):
        return None
    source = dataset.metadata.get("source_file")
    if not source:
        return None
    if dataset.metadata.get("derived_from_analysis") and dataset.metadata.get(
        "_project_path"
    ):
        return Path(dataset.metadata["_project_path"])
    return Path(source)


def _show_file_location(path: Path) -> bool:
    target = path if path.exists() else path.parent
    if not target.exists():
        return False
    if platform.system() == "Darwin" and path.exists():
        subprocess.Popen(["open", "-R", str(path)])
        return True
    from PySide6 import QtCore, QtGui

    location = target if target.is_dir() else target.parent
    return bool(QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(location))))


def _make_refreshing_combo_class():
    from PySide6 import QtWidgets

    class RefreshingComboBox(QtWidgets.QComboBox):
        def __init__(self, refresh_callback) -> None:
            super().__init__()
            self._refresh_callback = refresh_callback

        def showPopup(self):
            self._refresh_callback()
            super().showPopup()

    return RefreshingComboBox


def _has_slice_viewer_candidates(group: DataGroup) -> bool:
    for scope in _composite_scopes(group):
        if not data_group_composite_enabled(scope):
            continue
        ready, _message = data_group_composite_status(scope)
        if ready:
            return True
    for dataset in group.iter_datasets():
        if dataset.kind == "raw_dgs_nexus":
            continue
        if isinstance(dataset.metadata.get(DERIVED_RECIPE_KEY), dict):
            return True
        if isinstance(dataset.data, (MDHistoData, PointListData)):
            return True
        if isinstance(dataset.data, PointData4D) and dataset_rebin_enabled(dataset):
            return True
        if data_type_container(dataset.data_type) == "point_list":
            return True
        if dataset.metadata.get("derived_from_analysis") and dataset.metadata.get(
            "analysis_artifact_path"
        ):
            return True
        source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
        if source and Path(source).suffix.lower() in {".nxs", ".h5", ".hdf5", ".npz"}:
            return True
    return False


def _dataset_can_rebin(dataset: DatasetEntry) -> bool:
    return isinstance(dataset.data, (MDHistoData, PointData4D, PointListData)) or isinstance(
        dataset.metadata.get(DERIVED_RECIPE_KEY), dict
    )


def _dataset_can_save(dataset: DatasetEntry) -> bool:
    return isinstance(dataset.data, (MDHistoData, PointListData)) or dataset_rebin_enabled(dataset)


def _group_data_point_count(group: DataGroup) -> int:
    return sum(_dataset_data_point_count(dataset) for dataset in group.iter_datasets())


def _group_dataset_type_summary(group: DataGroup) -> str:
    counts: dict[str, int] = {}
    for dataset in group.iter_datasets():
        label = data_type_label(dataset.data_type)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "-"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def _dataset_axes_and_data_lines(data: Any) -> tuple[list[str], list[str]]:
    if data is None:
        return ["Not loaded."], ["Imported data: not loaded"]
    split = _split_dataset_summary_lines(_dataset_data_summary_lines(data))
    axes = split.get("Axes", ["No axis information available."])
    data_lines = split.get("Data")
    if data_lines is None:
        data_lines = split.get("", [])
        if not data_lines:
            data_lines = ["No data summary available."]
    return axes, data_lines


def _mdhisto_fit_bin_count(view: MDHistoData) -> int:
    """Count fit-eligible bins without materializing coordinate grids.

    Matches ``_point_data_from_mdhisto_view(view).valid_mask()``: an MDHisto
    grid's H/K/L/E bin centers are always finite, so only the mask, event count,
    and finite/positive intensity and error need checking. This avoids building
    the (potentially tens of millions of points) coordinate meshgrid just to
    report a count.
    """

    keep = ~np.asarray(view.mask, dtype=bool)
    keep &= mdhisto_measured_bins(view)
    keep &= np.isfinite(np.asarray(view.signal, dtype=float))
    errors = np.asarray(view.errors, dtype=float)
    keep &= np.isfinite(errors)
    keep &= errors > 0.0
    return int(np.count_nonzero(keep))


def _dataset_fit_summary_lines(
    dataset: DatasetEntry,
    *,
    group: DataGroup | None = None,
) -> list[str]:
    """Return fit-eligible bin/point counts using the same masks as fitting."""

    if dataset.data is None:
        return []
    extra_masks = effective_dataset_masks(group, dataset) if group is not None else []
    try:
        if dataset.scale_factor_vary:
            view = _viewer_data_before_scale(
                dataset,
                extra_masks=extra_masks,
                force_rebin=False,
                force_masks=False,
            )
        else:
            view = dataset_for_slice_viewer(
                dataset,
                extra_masks=extra_masks,
                force_rebin=False,
                force_masks=False,
            )
        if isinstance(view, MDHistoData):
            fit_bins = _mdhisto_fit_bin_count(view)
            total_bins = int(np.prod(view.shape))
            return [f"Fit bins: {fit_bins} of {total_bins}"]
        if isinstance(view, PointListData):
            points = _point_data_from_point_list_view(view)
            fit_points = int(np.count_nonzero(points.valid_mask()))
            return [f"Fit points: {fit_points} of {points.size}"]
        if isinstance(view, PointData4D):
            fit_points = int(np.count_nonzero(view.valid_mask()))
            return [f"Fit points: {fit_points} of {view.size}"]
    except Exception as exc:
        label = "Fit bins" if isinstance(dataset.data, MDHistoData) else "Fit points"
        return [f"{label}: unavailable ({exc})"]
    return []


def _split_dataset_summary_lines(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in lines:
        if not line:
            continue
        if line in {"Axes", "Data"}:
            current = line
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _dataset_data_summary_lines(data: Any) -> list[str]:
    if isinstance(data, MDHistoData):
        return _mdhisto_summary_lines(data)
    if isinstance(data, PointData4D):
        return _point_data_summary_lines(data)
    if isinstance(data, PointListData):
        return _point_list_summary_lines(data)
    if data is None:
        return ["", "Imported data: not loaded"]
    shape = getattr(data, "shape", None)
    lines = ["", f"Imported data type: {type(data).__name__}"]
    if shape is not None:
        lines.append(f"Shape: {_format_shape(shape)}")
    size = _object_nbytes(data)
    if size is not None:
        lines.append(f"Imported data size: {_format_bytes(size)}")
    return lines


def _mdhisto_summary_lines(data: MDHistoData) -> list[str]:
    lines = [
        "",
        "Axes",
        f"Dimensions: {len(data.axes)}",
        f"Shape: {_format_shape(data.shape)}",
    ]
    for index, axis in enumerate(data.axes):
        axis_size = data.shape[index]
        values = axis.centers
        value_range = ""
        if values.size:
            value_range = f", range {_format_number(values[0])} to {_format_number(values[-1])}"
        lines.append(
            f"{index}: {axis.name} ({axis.units or '-'}) - {axis_size} bins{value_range}"
        )
        if axis.kind:
            lines.append(f"   kind: {axis.kind}")
        if axis.frame:
            lines.append(f"   frame: {axis.frame}")
    valid = int(np.count_nonzero(~data.mask))
    masked = int(np.count_nonzero(data.mask))
    lines.extend(
        [
            "",
            "Data",
            f"Total bins: {int(np.prod(data.shape))}",
            f"Unmasked bins: {valid}",
            f"Masked bins: {masked}",
            f"Imported data size: {_format_bytes(_mdhisto_nbytes(data))}",
        ]
    )
    if data.coordinate_system is not None:
        lines.append(f"Coordinate system: {data.coordinate_system}")
    if data.visual_normalization is not None:
        lines.append(f"Visual normalization: {data.visual_normalization}")
    return lines


def _point_data_summary_lines(data: PointData4D) -> list[str]:
    lines = [
        "",
        "Axes",
        "Dimensions: 4",
    ]
    for name, units, values in (
        ("H", "rlu", data.H),
        ("K", "rlu", data.K),
        ("L", "rlu", data.L),
        ("E", "meV", data.E),
    ):
        lines.append(
            f"{name} ({units}) - {data.size} points, range {_format_number(np.nanmin(values))} to {_format_number(np.nanmax(values))}"
        )
    lines.extend(
        [
            "",
            "Data",
            f"Points: {data.size}",
            f"Valid points: {int(np.count_nonzero(data.valid_mask()))}",
            f"Imported data size: {_format_bytes(_point_data_nbytes(data))}",
        ]
    )
    if data.temperature is not None:
        lines.append(f"Temperature: {_condition_value_text(data.temperature)} K")
    return lines


def _point_list_summary_lines(data: PointListData) -> list[str]:
    lines = [
        "",
        "Axes",
        f"Points: {data.size}",
        f"Columns: {len(data.columns)}",
    ]
    for name in data.coordinate_names:
        values = data.column(name)
        unit = data.unit(name)
        value_range = ""
        if values.size:
            value_range = f", range {_format_number(np.nanmin(values))} to {_format_number(np.nanmax(values))}"
        lines.append(f"coord {name} ({unit or '-'}){value_range}")
    lines.extend(["", "Data", f"Points: {data.size}"])
    for channel in data.channels:
        label = str(channel["label"])
        unit = data.unit(str(channel["value"]))
        error = "with error" if channel.get("error") else "no error"
        quantity = data.channel_quantity_type(label)
        lines.append(f"channel {label} [{quantity}] ({unit or '-'}) - {error}")
    return lines


def _dataset_source_summary_lines(dataset: DatasetEntry) -> list[str]:
    lines = _dataset_source_lines(dataset)
    if not lines:
        return []
    return ["", "Source", *lines]


def _dataset_source_lines(dataset: DatasetEntry) -> list[str]:
    source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
    if not source:
        return []
    if dataset.metadata.get("derived_from_analysis"):
        project_path = dataset.metadata.get("_project_path")
        artifact_path = dataset.metadata.get("analysis_artifact_path") or source
        lines = [f"Project artifact: {artifact_path}"]
        size = (
            project_artifact_size(project_path, artifact_path)
            if project_path
            else None
        )
        lines.append(
            f"Artifact size: {_format_bytes(size)}"
            if size is not None
            else "Artifact size: unavailable"
        )
        return lines
    lines = [f"File: {source}"]
    path = Path(source)
    if path.exists():
        lines.append(f"File size on disk: {_format_bytes(path.stat().st_size)}")
    else:
        lines.append("File size on disk: unavailable")
    return lines


def _dataset_crystal_lines(dataset: DatasetEntry, group: DataGroup | None = None) -> list[str]:
    metadata = _merged_dataset_metadata(dataset)
    lines: list[str] = []
    lattice = None
    if group is not None and group.lattice_parameters:
        lattice = group.lattice_parameters
    if isinstance(metadata.get("lattice_parameters"), dict):
        lattice = metadata["lattice_parameters"]
    if isinstance(lattice, dict) and lattice:
        lines.append("Lattice parameters")
        lines.extend(_mapping_lines(lattice))
    matrix_name, matrix = _dataset_orientation_matrix(metadata)
    if matrix is not None:
        if lines:
            lines.append("")
        lines.append(matrix_name)
        lines.extend(_matrix_lines(matrix))
    if not lines:
        lines.append("No crystal metadata available.")
    return lines


def _merged_dataset_metadata(dataset: DatasetEntry) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(getattr(dataset.data, "metadata", None), dict):
        merged.update(dataset.data.metadata)
    if isinstance(dataset.metadata, dict):
        merged.update(dataset.metadata)
    return merged


def _dataset_orientation_matrix(metadata: dict[str, Any]) -> tuple[str, Any | None]:
    for key, label in (
        ("rlu_to_inv_angstrom_matrix", "RLU to inverse angstrom matrix"),
        ("ub_matrix", "UB matrix"),
        ("orientation_matrix", "Orientation matrix"),
    ):
        if key in metadata:
            return label, metadata[key]
    oriented_lattice = metadata.get("oriented_lattice")
    if isinstance(oriented_lattice, dict):
        for key, label in (
            ("rlu_to_inv_angstrom_matrix", "Oriented lattice RLU to inverse angstrom matrix"),
            ("ub_matrix", "Oriented lattice UB matrix"),
            ("orientation_matrix", "Oriented lattice orientation matrix"),
        ):
            if key in oriented_lattice:
                return label, oriented_lattice[key]
    return "", None


def _matrix_lines(matrix: Any) -> list[str]:
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return [_metadata_value_text(matrix)]
    if array.ndim != 2:
        return [_metadata_value_text(matrix)]
    return ["[" + ", ".join(_format_number(value) for value in row) + "]" for row in array]


def _dataset_metadata_lines(dataset: DatasetEntry) -> list[str]:
    return _mapping_lines(_dataset_metadata_mapping(dataset))


def _dataset_metadata_mapping(dataset: DatasetEntry) -> dict[str, Any]:
    merged = _merged_dataset_metadata(dataset)
    merged.pop("source_file", None)
    merged.pop("import_status", None)
    merged.pop("lattice_parameters", None)
    merged.pop("rlu_to_inv_angstrom_matrix", None)
    merged.pop("ub_matrix", None)
    merged.pop("orientation_matrix", None)
    merged.pop("oriented_lattice", None)
    return merged


def _mapping_lines(mapping: dict[str, Any]) -> list[str]:
    return [f"{key}: {_metadata_value_text(value)}" for key, value in sorted(mapping.items())]


def _metadata_value_text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _add_metadata_tree_item(parent: Any, key: str, value: Any, *, depth: int = 0) -> Any:
    from PySide6 import QtWidgets

    item = QtWidgets.QTreeWidgetItem([str(key), _metadata_tree_value_summary(value)])
    item.setToolTip(0, str(key))
    item.setToolTip(1, _metadata_value_text(value))
    parent.addChild(item) if hasattr(parent, "addChild") else parent.addTopLevelItem(item)
    if depth >= 6:
        return item
    if isinstance(value, dict):
        for child_key in sorted(value):
            _add_metadata_tree_item(item, str(child_key), value[child_key], depth=depth + 1)
    elif _metadata_is_expandable_sequence(value):
        for index, child_value in enumerate(list(value)[:64]):
            _add_metadata_tree_item(item, f"[{index}]", child_value, depth=depth + 1)
        if len(value) > 64:
            QtWidgets.QTreeWidgetItem(item, ["...", f"{len(value) - 64} more item(s)"])
    return item


def _tooltip_table_corner_buttons(table: Any, tooltip: str) -> None:
    """Give a table's internal Qt corner/select buttons a tooltip.

    QTableWidget's corner "select all" button is an untooltipped
    ``QAbstractButton``; label it so it does not read as a bare control.
    """

    from PySide6 import QtWidgets

    for button in table.findChildren(QtWidgets.QAbstractButton):
        if not button.toolTip():
            button.setToolTip(tooltip)


def _snapshot_parameter_rows(fit_entry: FitTimelineEntry) -> list[dict[str, str]]:
    """Return current parameter values from a fit entry's snapshot models.

    Used for the Current state (and Initial) entries, which carry live model
    parameter values but no optimizer results.
    """

    snapshot = fit_entry.snapshot if isinstance(fit_entry.snapshot, dict) else {}
    models = snapshot.get("models", [])
    if not isinstance(models, list):
        return []
    rows: list[dict[str, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        component = str(model.get("name", ""))
        params = model.get("parameters", {})
        if not isinstance(params, dict):
            continue
        limits = model.get("limits", {}) if isinstance(model.get("limits"), dict) else {}
        fitted = model.get("fit_parameters", {}) if isinstance(model.get("fit_parameters"), dict) else {}
        for name, value in params.items():
            label = f"{component}.{name}" if component else str(name)
            raw_limits = limits.get(name)
            side = None
            if bool(fitted.get(name, False)) and isinstance(raw_limits, (list, tuple)) and len(raw_limits) == 2:
                for candidate_side, bound, other_bound in (
                    ("lower", raw_limits[0], raw_limits[1]),
                    ("upper", raw_limits[1], raw_limits[0]),
                ):
                    if bound not in (None, "") and _parameter_is_at_bound(
                        value, bound, other_bound
                    ):
                        side = candidate_side
                        break
            rows.append({"name": label, "value": _format_number(value), "at_limit": bool(side), "limit_side": side or ""})
    return rows


def _fit_results_rows(fit_entry: FitTimelineEntry) -> list[dict[str, str]]:
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = _display_fit_parameters(fit_entry)
    if not isinstance(params, dict) or not params:
        return []
    stderr = goodness.get("stderr") if isinstance(goodness.get("stderr"), dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    posterior_params = (
        posterior.get("parameters")
        if isinstance(posterior.get("parameters"), dict)
        else {}
    )
    use_posterior_errors = bool(_posterior_display_options(fit_entry).get("use_posterior_uncertainties"))
    limit_hits = _fit_result_limit_hits(fit_entry, params)
    rows: list[dict[str, str]] = []
    for name, value in params.items():
        limit_hit = limit_hits.get(str(name))
        posterior_row = (
            posterior_params.get(name)
            if isinstance(posterior_params.get(name), dict)
            else {}
        )
        interval_errors = _posterior_interval_errors(fit_entry, str(name), float(value))
        uncertainty = _format_number(stderr[name]) if name in stderr else "-"
        if use_posterior_errors:
            uncertainty = (
                f"-{_format_number(interval_errors[0])} / +{_format_number(interval_errors[1])}"
                if interval_errors is not None
                else "-"
            )
        rows.append(
            {
                "name": str(name),
                "value": _format_number(value),
                "uncertainty": uncertainty,
                "median": (
                    _format_number(posterior_row["median"])
                    if "median" in posterior_row
                    else "-"
                ),
                "p16": _format_number(posterior_row["p16"]) if "p16" in posterior_row else "-",
                "p84": _format_number(posterior_row["p84"]) if "p84" in posterior_row else "-",
                "at_limit": limit_hit is not None,
                "limit_side": str(limit_hit.get("side", "")) if limit_hit else "",
                "limit_bound": _format_number(limit_hit.get("bound")) if limit_hit else "",
            }
        )
    return rows


def _fit_result_limit_hits(
    fit_entry: FitTimelineEntry,
    parameters: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return recorded LM hits or evaluate the displayed posterior sample."""

    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    # Stored LM hits describe the least-squares result.  When the best emcee
    # sample is active, evaluate its displayed values against the same saved
    # limits instead.
    use_best_sample = bool(_posterior_display_options(fit_entry).get("use_best_sample"))
    hits = {} if use_best_sample else _fit_limit_hits_from_goodness(goodness)
    if not use_best_sample:
        return hits
    by_casefold = {name.casefold(): name for name in parameters}
    snapshot = fit_entry.snapshot if isinstance(fit_entry.snapshot, dict) else {}
    models = snapshot.get("models", [])
    if not isinstance(models, list):
        return hits

    for model in models:
        if not isinstance(model, dict):
            continue
        component = str(model.get("name", "")).strip()
        values = model.get("parameters")
        limits = model.get("limits")
        fitted = model.get("fit_parameters")
        if not isinstance(values, dict) or not isinstance(limits, dict) or not isinstance(fitted, dict):
            continue
        for parameter_name in values:
            if not bool(fitted.get(parameter_name, False)):
                continue
            full_name = f"{component}.{parameter_name}" if component else str(parameter_name)
            result_name = by_casefold.get(full_name.casefold())
            if result_name is None or result_name in hits:
                continue
            bounds = limits.get(parameter_name)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            for side, bound, other_bound in (
                ("lower", bounds[0], bounds[1]),
                ("upper", bounds[1], bounds[0]),
            ):
                if bound not in (None, "") and _parameter_is_at_bound(
                    parameters[result_name], bound, other_bound
                ):
                    hits[result_name] = {"name": result_name, "side": side, "bound": bound}
                    break
    return hits




def _metadata_tree_value_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"{len(value)} field(s)"
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if _metadata_is_expandable_sequence(value):
        return f"{len(value)} item(s)"
    return _metadata_value_text(value)


def _metadata_is_expandable_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, np.ndarray)):
        return False
    if not isinstance(value, (list, tuple)):
        return False
    return len(value) > 8 or any(isinstance(item, (dict, list, tuple, np.ndarray)) for item in value)


def _array_summary(value: np.ndarray) -> str:
    array = np.asarray(value)
    shape = "x".join(str(size) for size in array.shape) or "scalar"
    summary = f"array {shape}, {array.dtype}"
    if array.size == 0:
        return f"{summary}, empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            return f"{summary}, min {_format_number(np.nanmin(finite))}, max {_format_number(np.nanmax(finite))}"
    if array.size <= 8:
        return f"{summary}, {_metadata_value_text(array.tolist())}"
    return summary


def _condition_value_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "empty"
        finite = value[np.isfinite(value)]
        if finite.size == 0:
            return "nonfinite"
        if np.allclose(finite, finite[0]):
            return _format_number(finite[0])
        return f"{_format_number(np.nanmin(finite))} to {_format_number(np.nanmax(finite))}"
    return _format_number(value) if isinstance(value, (int, float, np.number)) else str(value)


def _format_shape(shape: Any) -> str:
    return " x ".join(str(int(value)) for value in tuple(shape))


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _dataset_tree_page_label(
    datasets: Sequence[DatasetEntry],
    *,
    start: int,
    total: int,
) -> str:
    """Return a compact, stable label for one lazy dataset page."""

    end = start + len(datasets)
    if not datasets:
        return f"Runs {start + 1}-{end} of {total}"
    first = datasets[0].name
    last = datasets[-1].name
    range_text = first if first == last else f"{first} – {last}"
    return f"Runs {start + 1}-{end} of {total} ({range_text})"


def _format_seconds_per_step(value: float) -> str:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds < 0.0:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.3g} ms"
    return f"{seconds:.3g} s"


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _object_nbytes(data: Any) -> int | None:
    nbytes = getattr(data, "nbytes", None)
    if nbytes is None:
        return None
    return int(nbytes)


def _mdhisto_nbytes(data: MDHistoData) -> int:
    arrays = [data.signal, data.errors, data.mask, data.num_events]
    arrays.extend(axis.values for axis in data.axes)
    return int(sum(array.nbytes for array in arrays))


def _point_data_nbytes(data: PointData4D) -> int:
    arrays = [data.H, data.K, data.L, data.E, data.intensity, data.sigma, data.mask]
    if isinstance(data.temperature, np.ndarray):
        arrays.append(data.temperature)
    return int(sum(array.nbytes for array in arrays))










































def _model_key(group: DataGroup, model: ModelComponentSpec) -> str | None:
    for name, existing in group.models.items():
        if existing is model:
            return name
    return None




def _qt_app():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    configure_application_icon(app)
    configure_numeric_spin_boxes(app)
    return app


def _install_cli_interrupt_handler(app: Any) -> tuple[Any | None, Any | None]:
    """Let terminal Ctrl+C interrupt the Qt event loop."""

    from PySide6 import QtCore

    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda signum, _frame: app.exit(128 + signum))
    except ValueError:
        return None, None

    interrupt_timer = QtCore.QTimer()
    interrupt_timer.setInterval(200)
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start()
    return interrupt_timer, previous_handler


def _restore_cli_interrupt_handler(previous_handler: Any | None) -> None:
    if previous_handler is None:
        return
    try:
        signal.signal(signal.SIGINT, previous_handler)
    except ValueError:
        return


def _settings():
    return application_settings()


def _unique_dataset_name(base: str, existing: list[str]) -> str:
    return _unique_name(base or "Dataset", existing)


def _unique_name(base: str, existing: list[str]) -> str:
    candidate = base or "Dataset"
    if candidate not in existing:
        return candidate
    index = 1
    while f"{candidate}{index}" in existing:
        index += 1
    return f"{candidate}{index}"






def _parse_parameter_text(text: str) -> Any:
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped


def main() -> int:
    """Launch the nfit project explorer."""

    return NfitProjectExplorer().run()
