"""GUI-independent project dataset preparation orchestration.

This module owns project-data transformations shared by the project explorer,
workflow export, and analysis execution.  It intentionally has no Qt imports;
``project_gui`` re-exports the established API for compatibility.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np

from .analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from .analysis.fingerprint import recipe_hash
from .analysis.registry import analysis_definition, default_analysis_parameters
from .backgrounds import subtract_background
from .cache_utils import lru_store as _lru_store
from .dataset import PointData4D, PointListData
from .importers import IMPORTERS
from .mdhisto import MDHistoData, load_mantid_mdhisto_nxs
from .performance import initialize_rebin_performance
from .pipeline import DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from .project_coordinates import (
    COORDINATE_RANGE_AXIS_PREFIX as COORDINATE_RANGE_AXIS_PREFIX,
)
from .project_coordinates import (
    COORDINATE_RANGE_PARAMETER_NAMES as COORDINATE_RANGE_PARAMETER_NAMES,
)
from .project_coordinates import (
    _axis_projection_vector as _axis_projection_vector,
)
from .project_coordinates import (
    _clean_axis_weight as _clean_axis_weight,
)
from .project_coordinates import (
    _coordinate_axis_vector as _coordinate_axis_vector,
)
from .project_coordinates import (
    _coordinate_range_axis_specs as _coordinate_range_axis_specs,
)
from .project_coordinates import (
    _identity_vector as _identity_vector,
)
from .project_coordinates import (
    _is_vector_length as _is_vector_length,
)
from .project_coordinates import (
    _mdhisto_axis_coordinate_vector as _mdhisto_axis_coordinate_vector,
)
from .project_coordinates import (
    _mdhisto_coordinate_grids as _mdhisto_coordinate_grids,
)
from .project_coordinates import (
    _mdhisto_coordinate_range_axis_grids as _mdhisto_coordinate_range_axis_grids,
)
from .project_coordinates import (
    _mdhisto_coordinate_range_axis_specs as _mdhisto_coordinate_range_axis_specs,
)
from .project_coordinates import (
    _mdhisto_rebin_axis_vector as _mdhisto_rebin_axis_vector,
)
from .project_coordinates import (
    _mdhisto_rebin_source_axis_vectors as _mdhisto_rebin_source_axis_vectors,
)
from .project_coordinates import (
    _parse_parameter_text as _parse_parameter_text,
)
from .project_coordinates import (
    _project_hkl_vector as _project_hkl_vector,
)
from .project_coordinates import (
    _project_hkle_vector as _project_hkle_vector,
)
from .project_coordinates import (
    _projection_piece_value as _projection_piece_value,
)
from .project_coordinates import (
    _resolve_projected_axis_grid as _resolve_projected_axis_grid,
)
from .project_dataset_io import (
    _json_safe_value as _json_safe_value,
)
from .project_dataset_io import (
    _load_nfit_dataset_file as _load_nfit_dataset_file,
)
from .project_dataset_io import (
    _load_nfit_mdhisto_archive as _load_nfit_mdhisto_archive,
)
from .project_dataset_io import (
    _load_nfit_point_list_archive as _load_nfit_point_list_archive,
)
from .project_dataset_io import (
    _nfit_archive_json_mapping as _nfit_archive_json_mapping,
)
from .project_dataset_io import (
    _nfit_archive_optional_int as _nfit_archive_optional_int,
)
from .project_dataset_io import (
    _nfit_archive_text as _nfit_archive_text,
)
from .project_dataset_io import (
    _save_point_list_file as _save_point_list_file,
)
from .project_dataset_io import save_dataset_file as save_dataset_file
from .project_imports import (
    _ensure_dataset_data_loaded as _ensure_dataset_data_loaded_impl,
)
from .project_imports import (
    _loaded_data_point_count,
    _unique_dataset_name,
    data_type_container,
)
from .project_imports import (
    _reload_dataset_copy as _reload_dataset_copy_impl,
)
from .project_imports import (
    reload_data_group as _reload_data_group_impl,
)
from .project_imports import (
    reload_dataset_data as _reload_dataset_data_impl,
)
from .project_masks import (
    _coordinate_centers as _coordinate_centers,
)
from .project_masks import (
    _evaluate_mdhisto_mask as _evaluate_mdhisto_mask,
)
from .project_masks import (
    _evaluate_point_data_mask as _evaluate_point_data_mask,
)
from .project_masks import (
    _exclusion_parameter_range as _exclusion_parameter_range,
)
from .project_masks import (
    _mask_axis_names as _mask_axis_names,
)
from .project_masks import (
    _matrix_includes_2pi as _matrix_includes_2pi,
)
from .project_masks import (
    _mdhisto_box_mask as _mdhisto_box_mask,
)
from .project_masks import (
    _mdhisto_coordinate_range_mask as _mdhisto_coordinate_range_mask,
)
from .project_masks import (
    _mdhisto_ellipsoid_mask as _mdhisto_ellipsoid_mask,
)
from .project_masks import (
    _mdhisto_energy_q_range_mask as _mdhisto_energy_q_range_mask,
)
from .project_masks import (
    _mdhisto_phonon_cone_mask as _mdhisto_phonon_cone_mask,
)
from .project_masks import (
    _mdhisto_projected_region_inputs as _mdhisto_projected_region_inputs,
)
from .project_masks import (
    _mdhisto_q_matrix as _mdhisto_q_matrix,
)
from .project_masks import (
    _mdhisto_q_modulus_grid as _mdhisto_q_modulus_grid,
)
from .project_masks import (
    _mdhisto_with_nfit_masks as _mdhisto_with_nfit_masks,
)
from .project_masks import (
    _metadata_coordinate_units_are_inv_angstrom_for_mdhisto as _metadata_coordinate_units_are_inv_angstrom_for_mdhisto,
)
from .project_masks import (
    _nfit_mask_for_mdhisto as _nfit_mask_for_mdhisto,
)
from .project_masks import (
    _nfit_mask_for_point_data as _nfit_mask_for_point_data,
)
from .project_masks import (
    _parameter_float as _parameter_float,
)
from .project_masks import (
    _parameter_float_sequence as _parameter_float_sequence,
)
from .project_masks import (
    _parameter_range as _parameter_range,
)
from .project_masks import (
    _point_data_box_mask as _point_data_box_mask,
)
from .project_masks import (
    _point_data_coordinate_range_mask as _point_data_coordinate_range_mask,
)
from .project_masks import (
    _point_data_coordinate_values as _point_data_coordinate_values,
)
from .project_masks import (
    _point_data_ellipsoid_mask as _point_data_ellipsoid_mask,
)
from .project_masks import (
    _point_data_energy_q_range_mask as _point_data_energy_q_range_mask,
)
from .project_masks import (
    _point_data_phonon_cone_mask as _point_data_phonon_cone_mask,
)
from .project_masks import (
    _point_data_projected_region_inputs as _point_data_projected_region_inputs,
)
from .project_masks import (
    _point_data_q_modulus as _point_data_q_modulus,
)
from .project_masks import (
    _point_data_q_vectors as _point_data_q_vectors,
)
from .project_masks import (
    _point_data_with_nfit_masks as _point_data_with_nfit_masks,
)
from .project_masks import (
    _range_is_unrestricted as _range_is_unrestricted,
)
from .project_masks import (
    _range_tolerance as _range_tolerance,
)
from .project_masks import (
    _values_in_range as _values_in_range,
)
from .project_point_lists import (
    _PREPARED_POINT_LIST_CACHE as _PREPARED_POINT_LIST_CACHE,
)
from .project_point_lists import (
    _PREPARED_POINT_LIST_CACHE_LIMIT as _PREPARED_POINT_LIST_CACHE_LIMIT,
)
from .project_point_lists import (
    _PREPARED_POINT_LIST_CACHE_MAX_BYTES as _PREPARED_POINT_LIST_CACHE_MAX_BYTES,
)
from .project_point_lists import (
    D_SPACING_COORDINATE_NAME as D_SPACING_COORDINATE_NAME,
)
from .project_point_lists import (
    DATASET_POINT_LIST_KEY as DATASET_POINT_LIST_KEY,
)
from .project_point_lists import (
    HEAT_CAPACITY_CHANNEL_LABEL as HEAT_CAPACITY_CHANNEL_LABEL,
)
from .project_point_lists import (
    HEAT_CAPACITY_OVER_T_CHANNEL_LABEL as HEAT_CAPACITY_OVER_T_CHANNEL_LABEL,
)
from .project_point_lists import (
    INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL as INVERSE_SUSCEPTIBILITY_CHANNEL_LABEL,
)
from .project_point_lists import (
    Q_COORDINATE_NAME as Q_COORDINATE_NAME,
)
from .project_point_lists import (
    SUSCEPTIBILITY_CHANNEL_LABEL as SUSCEPTIBILITY_CHANNEL_LABEL,
)
from .project_point_lists import (
    TEMPERATURE_SQUARED_COLUMN as TEMPERATURE_SQUARED_COLUMN,
)
from .project_point_lists import (
    _prepared_point_list_signature as _prepared_point_list_signature,
)
from .project_point_lists import (
    point_list_config as point_list_config,
)
from .project_point_lists import (
    prepared_point_list_data as prepared_point_list_data,
)
from .project_rebinning import (
    _axis_bounds as _axis_bounds,
)
from .project_rebinning import (
    _axis_mode_coordinates_with_symmetry as _axis_mode_coordinates_with_symmetry,
)
from .project_rebinning import (
    _cluster_coordinate_centers as _cluster_coordinate_centers,
)
from .project_rebinning import (
    _composite_rebin_bin_edges as _composite_rebin_bin_edges,
)
from .project_rebinning import (
    _composite_rebin_step_sizes as _composite_rebin_step_sizes,
)
from .project_rebinning import (
    _coordinate_center_edges as _coordinate_center_edges,
)
from .project_rebinning import (
    _default_rebin_axes as _default_rebin_axes,
)
from .project_rebinning import (
    _finite_coordinate_bounds as _finite_coordinate_bounds,
)
from .project_rebinning import (
    _mdhisto_axis_edges as _mdhisto_axis_edges,
)
from .project_rebinning import (
    _mdhisto_cell_volumes as _mdhisto_cell_volumes,
)
from .project_rebinning import (
    _mdhisto_rebin_axis_variable as _mdhisto_rebin_axis_variable,
)
from .project_rebinning import (
    _mdhisto_rebin_basis_bounds as _mdhisto_rebin_basis_bounds,
)
from .project_rebinning import (
    _mdhisto_rebin_basis_transform as _mdhisto_rebin_basis_transform,
)
from .project_rebinning import (
    _mdhisto_rebin_component as _mdhisto_rebin_component,
)
from .project_rebinning import (
    _migrate_rebin_axis_modes as _migrate_rebin_axis_modes,
)
from .project_rebinning import (
    _momentum_coordinate_variables as _momentum_coordinate_variables,
)
from .project_rebinning import (
    _momentum_rebin_axis_indices as _momentum_rebin_axis_indices,
)
from .project_rebinning import (
    _momentum_rebin_matrix as _momentum_rebin_matrix,
)
from .project_rebinning import (
    _momentum_rebin_vector_text as _momentum_rebin_vector_text,
)
from .project_rebinning import (
    _normalize_rebin_bin_edges as _normalize_rebin_bin_edges,
)
from .project_rebinning import (
    _num_bins_from_step_size as _num_bins_from_step_size,
)
from .project_rebinning import (
    _output_bin_volumes as _output_bin_volumes,
)
from .project_rebinning import (
    _parameter_to_text as _parameter_to_text,
)
from .project_rebinning import (
    _point_data_histogram as _point_data_histogram,
)
from .project_rebinning import (
    _point_data_rebin_basis_bounds as _point_data_rebin_basis_bounds,
)
from .project_rebinning import (
    _rebin_axis_bound_is_auto as _rebin_axis_bound_is_auto,
)
from .project_rebinning import (
    _rebin_axis_mode as _rebin_axis_mode,
)
from .project_rebinning import (
    _rebin_axis_name as _rebin_axis_name,
)
from .project_rebinning import (
    _rebin_axis_vector as _rebin_axis_vector,
)
from .project_rebinning import (
    _rebin_config_basis_bounds as _rebin_config_basis_bounds,
)
from .project_rebinning import (
    _rebin_fractional_axes as _rebin_fractional_axes,
)
from .project_rebinning import (
    _rebin_grid_kwargs as _rebin_grid_kwargs,
)
from .project_rebinning import (
    _rebin_max_batch_bytes as _rebin_max_batch_bytes,
)
from .project_rebinning import (
    _rebin_max_batch_mb as _rebin_max_batch_mb,
)
from .project_rebinning import (
    _rebin_mdhisto_coverage as _rebin_mdhisto_coverage,
)
from .project_rebinning import (
    _rebin_mdhisto_data as _rebin_mdhisto_data,
)
from .project_rebinning import (
    _rebin_mean_weighting as _rebin_mean_weighting,
)
from .project_rebinning import (
    _rebin_minimum_coverage as _rebin_minimum_coverage,
)
from .project_rebinning import (
    _rebin_minimum_samples as _rebin_minimum_samples,
)
from .project_rebinning import (
    _rebin_point_data as _rebin_point_data,
)
from .project_rebinning import (
    _rebin_resolution_mode as _rebin_resolution_mode,
)
from .project_rebinning import (
    _rebin_symmetry_count as _rebin_symmetry_count,
)
from .project_rebinning import (
    _rebin_symmetry_matrices as _rebin_symmetry_matrices,
)
from .project_rebinning import (
    _rebin_symmetry_metadata as _rebin_symmetry_metadata,
)
from .project_rebinning import (
    _rebin_symmetry_operations as _rebin_symmetry_operations,
)
from .project_rebinning import (
    _resolve_auto_rebin_axes as _resolve_auto_rebin_axes,
)
from .project_rebinning import (
    _resolve_data_driven_rebin_axes as _resolve_data_driven_rebin_axes,
)
from .project_rebinning import (
    _sanitize_rebin_axis_config as _sanitize_rebin_axis_config,
)
from .project_rebinning import (
    _step_size_from_bounds as _step_size_from_bounds,
)
from .project_rebinning import (
    _symmetry_projected_coordinate_bounds as _symmetry_projected_coordinate_bounds,
)
from .project_rebinning import (
    _update_mdhisto_rebin_basis as _update_mdhisto_rebin_basis,
)
from .project_rebinning import (
    _update_rebin_momentum_basis as _update_rebin_momentum_basis,
)
from .project_rebinning import (
    _update_rebin_momentum_matrix as _update_rebin_momentum_matrix,
)
from .project_rebinning import (
    _validate_mdhisto_rebin_basis as _validate_mdhisto_rebin_basis,
)
from .project_view_data import (
    KINEMATIC_KF_KI_INCLUDED_KEY as KINEMATIC_KF_KI_INCLUDED_KEY,
)
from .project_view_data import (
    _apply_dataset_scale as _apply_dataset_scale,
)
from .project_view_data import (
    _apply_kinematic_normalization_to_points as _apply_kinematic_normalization_to_points,
)
from .project_view_data import (
    _apply_kinematic_normalization_to_view as _apply_kinematic_normalization_to_view,
)
from .project_view_data import (
    _apply_spectral_channel_view as _apply_spectral_channel_view,
)
from .project_view_data import (
    _kinematic_energy_metadata as _kinematic_energy_metadata,
)
from .project_view_data import (
    _kinematic_kf_ki_factor as _kinematic_kf_ki_factor,
)
from .project_view_data import (
    _mdhisto_without_nfit_masks as _mdhisto_without_nfit_masks,
)
from .project_view_data import (
    _with_viewer_dataset_metadata as _with_viewer_dataset_metadata,
)
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY
from .symmetry import SymmetrySpec, symmetry_config

DATASET_REBIN_KEY = "rebin"
DATASET_MASK_APPLICATION_KEY = "mask_application"
GROUP_COMPOSITE_NAME = "Composite"
DERIVED_RECIPE_KEY = "derived_recipe"
VIRTUAL_DERIVED_ANALYSIS_TYPES = {"dataset_clone", "histogram_arithmetic"}
DEFAULT_REBIN_MAX_BATCH_MB = 192
DEFAULT_MINIMUM_COVERAGE = 0.9
DEFAULT_MINIMUM_SAMPLES = 0.0
REBIN_COORDINATE_BASIS_VERSION = 2
REBIN_RESOLUTION_MODE_KEY = "resolution_mode"
REBIN_AXIS_MODES = frozenset({"discrete", "step", "bins", "edges", "tolerance"})
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

# Loading, rebinning, and masking full datasets is reused across passive GUI
# refreshes and script/API calls. Signatures are content based; scale factors
# are deliberately applied after the cached preparation step.
_VIEWER_VIEW_CACHE: OrderedDict[str, tuple[str, Any]] = OrderedDict()
_VIEWER_VIEW_CACHE_LIMIT = 8
_VIEWER_VIEW_CACHE_MAX_BYTES = 256 * 1024**2


def effective_dataset_masks(group: DataGroup, dataset: DatasetEntry) -> list[MaskSpec]:
    """Return masks inherited from the ancestor group chain of ``dataset``.

    Walks from the data group down through nested dataset groups to the
    dataset's parent, collecting each level's shared masks (outer to inner).
    Returns an empty list when the dataset is not found.
    """

    def search(node: Any) -> list[MaskSpec] | None:
        if any(item.id == dataset.id for item in getattr(node, "datasets", [])):
            return list(node.masks)
        for subgroup in getattr(node, "subgroups", []):
            below = search(subgroup)
            if below is not None:
                return list(node.masks) + below
        return None

    return search(group) or []


def _ensure_dataset_data_loaded(dataset: DatasetEntry) -> Any:
    """Load data through the GUI-compatible project import service."""

    return _ensure_dataset_data_loaded_impl(
        dataset,
        mdhisto_loader=load_mantid_mdhisto_nxs,
        dataset_file_loader=_load_nfit_dataset_file,
    )


def _reload_dataset_copy(dataset: DatasetEntry) -> DatasetEntry:
    return _reload_dataset_copy_impl(dataset, data_loader=_ensure_dataset_data_loaded)


def reload_dataset_data(dataset: DatasetEntry) -> Any:
    """Reload one dataset without mutating it when source reading fails."""

    return _reload_dataset_data_impl(dataset, data_loader=_ensure_dataset_data_loaded)


def reload_data_group(group: DataGroup | DatasetGroup) -> list[DatasetEntry]:
    """Reload every descendant dataset that has a configured data source."""

    return _reload_data_group_impl(group, data_loader=_ensure_dataset_data_loaded)


def _derived_analysis_for_dataset(
    dataset: DatasetEntry,
) -> tuple[DataGroup, AnalysisEntry] | None:
    recipe = dataset.metadata.get(DERIVED_RECIPE_KEY)
    group = getattr(dataset, "_derived_owner_group", None)
    if not isinstance(recipe, dict) or not isinstance(group, DataGroup):
        return None
    analysis_id = str(recipe.get("analysis_id", ""))
    analysis = next((item for item in group.analyses if item.id == analysis_id), None)
    return (group, analysis) if analysis is not None else None


def _derived_source_scope(
    group: DataGroup,
    source_id: str,
) -> DataGroup | _CompositeScope | None:
    prefix = "group-composite:"
    if not str(source_id).startswith(prefix):
        return None
    suffix = str(source_id)[len(prefix) :]
    if suffix == "root":
        return group
    node = next(
        (candidate for candidate in group.iter_subgroups() if candidate.id == suffix),
        None,
    )
    if node is None:
        raise KeyError(f"derived dataset refers to missing group composite {source_id}")
    return _CompositeScope(group, node)


def _derived_output_config_for_source(
    group: DataGroup,
    source_id: str,
) -> dict[str, Any]:
    scope = _derived_source_scope(group, source_id)
    if scope is not None:
        config = copy.deepcopy(data_group_composite_config(scope))
    else:
        source = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        if source is None:
            raise KeyError(f"derived dataset refers to missing dataset ID {source_id}")
        loaded = _ensure_dataset_data_loaded(source)
        config = copy.deepcopy(source.parameters.get(DATASET_REBIN_KEY, {}))
        if not isinstance(config.get("axes"), list) or not config.get("axes"):
            config["axes"] = _default_rebin_axes(loaded)
    config["enabled"] = True
    config["stale"] = True
    config.setdefault("auto_rebin", False)
    config.setdefault("normalize", True)
    config.setdefault("mean_weighting", "uniform")
    config.setdefault("minimum_coverage", 0.0)
    config.setdefault("minimum_samples", DEFAULT_MINIMUM_SAMPLES)
    config.setdefault("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    return config


def create_derived_analysis_dataset(
    group: DataGroup,
    analysis: AnalysisEntry,
    *,
    name: str | None = None,
    rebin_config: Mapping[str, Any] | None = None,
) -> DatasetEntry:
    """Create or update a live derived dataset evaluated from source data.

    Clone and histogram-arithmetic recipes own their output grid. The grid is
    pushed into every source reduction, so raw MDEvent or point data are binned
    directly rather than rebinning an already-binned input histogram.
    """

    if analysis.type not in VIRTUAL_DERIVED_ANALYSIS_TYPES:
        raise ValueError(f"analysis type {analysis.type!r} is not a live derived dataset")
    expected = 1 if analysis.type == "dataset_clone" else 2
    if len(analysis.input_dataset_ids) != expected:
        raise ValueError(f"{analysis.type} requires exactly {expected} source(s)")
    config = (
        copy.deepcopy(dict(rebin_config))
        if rebin_config is not None
        else _derived_output_config_for_source(group, analysis.input_dataset_ids[0])
    )
    config["enabled"] = True
    analysis.metadata["output_rebin_config"] = copy.deepcopy(config)
    linked = next(
        (
            dataset
            for dataset in group.iter_datasets()
            if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id") == analysis.id
        ),
        None,
    )
    if linked is None:
        derived = next(
            (node for node in group.subgroups if node.name == "Derived data"),
            None,
        )
        if derived is None:
            derived = DatasetGroup("Derived data")
            group.subgroups.append(derived)
        if analysis.type == "dataset_clone":
            default_name = f"{analysis.name} dataset"
        else:
            default_name = analysis.name
        linked = DatasetEntry(
            _unique_dataset_name(name or default_name, group.dataset_names),
            None,
            kind="derived_recipe",
            data_type=(
                "powder_inelastic"
                if len(config.get("axes", [])) == 2
                else "single_crystal_inelastic"
            ),
            metadata={},
            parameters={DATASET_REBIN_KEY: config},
            enabled=False,
            fit_weight=0.0,
        )
        derived.datasets.append(linked)
    else:
        linked.kind = "derived_recipe"
        linked.parameters[DATASET_REBIN_KEY] = config
        linked.replace_data(None, source_backed=False)
    linked.metadata.pop("source_file", None)
    linked.metadata.pop("analysis_artifact_path", None)
    linked.metadata["derived_from_analysis"] = {
        "analysis_id": analysis.id,
        "output_key": "clone" if analysis.type == "dataset_clone" else "histogram",
    }
    linked.metadata[DERIVED_RECIPE_KEY] = {
        "analysis_id": analysis.id,
        "operation": analysis.type,
        "input_source_ids": list(analysis.input_dataset_ids),
        "source_stage": "underlying_data",
    }
    linked._derived_owner_group = group
    if analysis not in group.analyses:
        group.analyses.append(analysis)
    output_key = "clone" if analysis.type == "dataset_clone" else "histogram"
    analysis.result = AnalysisResultRecord(
        recipe_hash=recipe_hash(
            analysis.type,
            analysis_definition(analysis.type).version,
            {**default_analysis_parameters(analysis.type), **analysis.parameters},
            analysis.input_dataset_ids,
        ),
        input_fingerprints={},
        outputs=[
            AnalysisOutputRef(
                output_key,
                linked.name,
                "dataset",
                dataset_id=linked.id,
                metadata={
                    "data_type": linked.data_type,
                    "fit_enabled": False,
                    "fit_weight": 0.0,
                    "virtual": True,
                },
            )
        ],
        status="live",
        created_at=datetime.now().isoformat(timespec="seconds"),
        diagnostics={"source_stage": "underlying_data"},
    )
    return linked


def _derived_source_data(
    group: DataGroup,
    source_id: str,
    config: Mapping[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    scope = _derived_source_scope(group, source_id)
    if scope is not None:
        return composite_dataset_data(
            scope,
            progress_callback=progress_callback,
            config_override=config,
            include_source_masks=False,
        )
    source = next(
        (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
        None,
    )
    if source is None:
        raise KeyError(f"derived dataset refers to missing dataset ID {source_id}")
    loaded = _ensure_dataset_data_loaded(source)
    reduction_source = (
        _mdhisto_without_nfit_masks(loaded) if isinstance(loaded, MDHistoData) else loaded
    )
    temporary = source.copy(
        data=reduction_source,
        parameters={
            **copy.deepcopy(source.parameters),
            DATASET_REBIN_KEY: copy.deepcopy(dict(config)),
        },
        masks=[],
    )
    temporary.parameters[DATASET_REBIN_KEY]["enabled"] = True
    result = rebinned_dataset_data(
        temporary,
        extra_masks=[],
        progress_callback=progress_callback,
    )
    if isinstance(result, MDHistoData) and temporary.backgrounds:
        result = _apply_dataset_backgrounds(temporary, result)
    return _apply_dataset_scale(source, result)


def derived_analysis_dataset_data(
    dataset: DatasetEntry,
    *,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D:
    """Evaluate one live derived recipe on its underlying source data."""

    resolved = _derived_analysis_for_dataset(dataset)
    if resolved is None:
        raise ValueError(f"derived dataset {dataset.name!r} has no live analysis owner")
    group, analysis = resolved
    config = copy.deepcopy(dataset_rebin_config(dataset))
    config["enabled"] = True
    sources = [
        _derived_source_data(
            group,
            source_id,
            config,
            progress_callback=progress_callback,
        )
        for source_id in analysis.input_dataset_ids
    ]
    if analysis.type == "dataset_clone":
        result = sources[0]
    elif analysis.type == "histogram_arithmetic":
        if not all(isinstance(item, MDHistoData) for item in sources):
            raise TypeError("histogram arithmetic requires gridded source reductions")
        from .analysis.data_reduction import combine_aligned_histograms

        result = combine_aligned_histograms(
            sources[0],
            sources[1],
            operation=str(analysis.parameters.get("operation", "subtract")),
            right_scale=float(analysis.parameters.get("right_scale", 1.0)),
        )
    else:
        raise ValueError(f"unsupported live derived operation {analysis.type!r}")
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata[DERIVED_RECIPE_KEY] = {
        "analysis_id": analysis.id,
        "operation": analysis.type,
        "input_source_ids": list(analysis.input_dataset_ids),
        "source_stage": "underlying_data",
    }
    if isinstance(result, MDHistoData):
        return result.with_updates(metadata=metadata)
    if isinstance(result, PointListData):
        return result.with_updates(metadata=metadata)
    return result.with_updates(metadata=metadata)


def dataset_for_slice_viewer(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | PointData4D | None:
    """Return a viewer-ready dataset, loading from source metadata if needed.

    ``extra_masks`` are masks inherited from ancestor dataset groups; they are
    applied ahead of the dataset's own masks.
    """

    result = _viewer_data_before_scale(
        dataset,
        extra_masks=extra_masks,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
    )
    if result is None:
        return None
    scaled = _apply_dataset_scale(dataset, result)
    prepared = _apply_spectral_channel_view(dataset, scaled)
    return _with_viewer_dataset_metadata(dataset, prepared)


def _mask_signature(masks: list[MaskSpec] | None) -> list[Any]:
    return [
        [
            m.type,
            bool(m.enabled),
            bool(m.invert),
            bool(m.additive),
            json.dumps(m.parameters, sort_keys=True, default=str),
        ]
        for m in (masks or [])
    ]


def _viewer_view_signature(dataset: DatasetEntry, extra_masks: list[MaskSpec] | None) -> str:
    rebin = (
        json.dumps(dataset_rebin_config(dataset), sort_keys=True, default=str)
        if dataset_rebin_enabled(dataset)
        else None
    )
    payload = [
        dataset.data_cache_token,
        dataset.data_type,
        dataset.kind,
        rebin,
        _mask_signature(getattr(dataset, "masks", None)),
        _mask_signature(extra_masks),
        [
            [
                background.source_dataset_id,
                bool(background.enabled),
                float(background.scale),
                background.interpolation,
                (
                    background.source_entry.data_cache_token
                    if background.source_entry is not None
                    else None
                ),
            ]
            for background in dataset.backgrounds
        ],
        _derived_recipe_dependency_signature(dataset),
    ]
    return json.dumps(payload, sort_keys=True, default=str)


def _derived_recipe_dependency_signature(dataset: DatasetEntry) -> Any:
    resolved = _derived_analysis_for_dataset(dataset)
    if resolved is None:
        return None
    group, analysis = resolved
    dependencies = []
    for source_id in analysis.input_dataset_ids:
        scope = _derived_source_scope(group, source_id)
        if scope is not None:
            dependencies.append([source_id, _composite_cache_signature(scope)])
            continue
        source = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        dependencies.append(
            [
                source_id,
                None if source is None else source.data_cache_token,
                None if source is None else float(source.scale_factor),
                None if source is None else bool(source.enabled),
            ]
        )
    return [
        analysis.type,
        json.dumps(analysis.parameters, sort_keys=True, default=str),
        dependencies,
    ]


def _viewer_data_before_scale(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | None:
    key = dataset.id
    signature = _viewer_view_signature(dataset, extra_masks)
    deferred_masks = _should_defer_dataset_masks(dataset, force_masks=force_masks)
    cached = _VIEWER_VIEW_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        _VIEWER_VIEW_CACHE.move_to_end(key)
        if force_masks:
            dataset_mask_application_config(dataset)["stale"] = False
        return cached[1]
    if cached is not None and deferred_masks:
        return cached[1]
    if cached is not None and _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
        return cached[1]
    result = _viewer_data_before_scale_uncached(
        dataset,
        extra_masks=extra_masks,
        force_rebin=force_rebin,
        force_masks=force_masks,
        progress_callback=progress_callback,
    )
    if isinstance(result, MDHistoData) and dataset.backgrounds:
        result = _apply_dataset_backgrounds(dataset, result)
    if result is not None and not deferred_masks:
        # Recompute the signature: the uncached path may have lazily loaded the
        # data and incremented the dataset revision.
        signature = _viewer_view_signature(dataset, extra_masks)
        _lru_store(
            _VIEWER_VIEW_CACHE,
            key,
            (signature, result),
            _VIEWER_VIEW_CACHE_LIMIT,
            _VIEWER_VIEW_CACHE_MAX_BYTES,
        )
        dataset_mask_application_config(dataset)["stale"] = False
    return result


def _apply_dataset_backgrounds(
    dataset: DatasetEntry,
    data: MDHistoData,
) -> MDHistoData:
    result = data
    for background in dataset.backgrounds:
        if not background.enabled:
            continue
        source = background.source_entry
        if source is None:
            raise ValueError(f"background {background.name!r} refers to a missing dataset")
        source_data = _viewer_data_before_scale_uncached(
            source,
            force_rebin=True,
            force_masks=True,
        )
        if not isinstance(source_data, MDHistoData):
            raise TypeError(f"background {background.name!r} must refer to gridded histogram data")
        result = subtract_background(
            result,
            source_data,
            scale=background.scale,
            interpolation=background.interpolation,
        )
    return result


def _viewer_data_before_scale_uncached(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    force_rebin: bool = True,
    force_masks: bool = True,
    progress_callback: Any | None = None,
) -> MDHistoData | PointListData | None:
    if isinstance(dataset.metadata.get(DERIVED_RECIPE_KEY), dict):
        derived = derived_analysis_dataset_data(
            dataset,
            progress_callback=progress_callback,
        )
        dataset_rebin_config(dataset)["stale"] = False
        if isinstance(derived, MDHistoData):
            return _mdhisto_with_nfit_masks(
                dataset,
                data=derived,
                extra_masks=extra_masks,
            )
        return derived
    loaded = _ensure_dataset_data_loaded(dataset)
    if data_type_container(dataset.data_type) == "point_list" or isinstance(loaded, PointListData):
        if not isinstance(loaded, PointListData):
            return None
        if dataset_rebin_enabled(dataset):
            if _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
                return prepared_point_list_data(dataset)
            return rebinned_dataset_data(dataset, progress_callback=progress_callback)
        return prepared_point_list_data(dataset)
    if isinstance(loaded, MDHistoData):
        if _should_defer_dataset_masks(dataset, force_masks=force_masks):
            return _mdhisto_without_nfit_masks(loaded)
        if dataset_rebin_enabled(dataset):
            if _should_defer_dataset_rebin(dataset, force_rebin=force_rebin):
                return _mdhisto_with_nfit_masks(dataset, data=loaded, extra_masks=extra_masks)
            return rebinned_dataset_data(
                dataset, extra_masks=extra_masks, progress_callback=progress_callback
            )
        return _mdhisto_with_nfit_masks(dataset, data=loaded, extra_masks=extra_masks)
    if isinstance(loaded, PointData4D):
        if dataset_rebin_enabled(dataset):
            return rebinned_dataset_data(
                dataset, extra_masks=extra_masks, progress_callback=progress_callback
            )
        return _point_data_with_nfit_masks(dataset, loaded, extra_masks=extra_masks)
    return None


def dataset_rebin_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return a dataset rebin configuration, creating default axis settings if needed."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.parameters[DATASET_REBIN_KEY] = config
    initialize_rebin_performance(config)
    config.setdefault("enabled", False)
    if not isinstance(config.get("symmetry"), dict):
        config["symmetry"] = symmetry_config(SymmetrySpec())
    if config.get(REBIN_RESOLUTION_MODE_KEY) not in {"step", "bins"}:
        config[REBIN_RESOLUTION_MODE_KEY] = "step"
    try:
        config["max_batch_mb"] = max(int(config.get("max_batch_mb", DEFAULT_REBIN_MAX_BATCH_MB)), 1)
    except (TypeError, ValueError):
        config["max_batch_mb"] = DEFAULT_REBIN_MAX_BATCH_MB
    if config.get("mean_weighting") not in {"inverse_variance", "uniform"}:
        config["mean_weighting"] = "uniform"
    config["minimum_coverage"] = _rebin_minimum_coverage(config)
    config["minimum_samples"] = _rebin_minimum_samples(config)
    config["normalize"] = True
    axes = config.get("axes")
    # Only materialize transformed coordinates when defaults are actually
    # needed. Existing axes already carry their complete saved basis.
    if isinstance(dataset.data, PointListData) and (not isinstance(axes, list) or not axes):
        default_axes = _default_rebin_axes(prepared_point_list_data(dataset))
    elif isinstance(dataset.data, PointListData):
        default_axes = []
    else:
        default_axes = _default_rebin_axes(dataset.data)
    if not isinstance(axes, list) or not axes:
        # A saved project may call this while its file-backed dataset is still
        # unloaded. Leave axes absent in that state so first data load can
        # create defaults, but never replace a saved non-empty basis.
        if default_axes:
            config["axes"] = default_axes
            if isinstance(dataset.data, MDHistoData) and len(dataset.data.axes) == 4:
                config["coordinate_basis_version"] = REBIN_COORDINATE_BASIS_VERSION
    else:
        sanitized_axes = list(axes)
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
                if isinstance(dataset.data, MDHistoData) and "vector" in axis_config:
                    axis_config["variable"] = str(
                        axis_config.get("variable") or default_axis.get("variable", "")
                    )
                    axis_config["name"] = _rebin_axis_name(
                        axis_config["variable"], axis_config.get("vector", [])
                    )
                sanitized_axes.append(_sanitize_rebin_axis_config(axis_config))
            config["axes"] = sanitized_axes
        if (
            len(axes) == len(default_axes)
            and isinstance(dataset.data, MDHistoData)
            and len(dataset.data.axes) == 4
        ):
            try:
                basis_version = int(config["coordinate_basis_version"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "saved 4D rebin settings are missing a supported "
                    "coordinate_basis_version; recreate the rebin settings"
                ) from exc
            if basis_version != REBIN_COORDINATE_BASIS_VERSION:
                raise ValueError(
                    f"unsupported rebin coordinate basis version {basis_version}; "
                    "recreate the rebin settings"
                )
    if "auto_rebin" not in config:
        config["auto_rebin"] = not _dataset_rebin_is_large(dataset, config)
    _migrate_rebin_axis_modes(config)
    config.setdefault("stale", False)
    return config


def _rebin_settings_clipboard_text(config: dict[str, Any]) -> str:
    """Serialize user-editable rebin settings for the system clipboard."""

    settings = {key: copy.deepcopy(config[key]) for key in REBIN_SETTINGS_KEYS if key in config}
    return json.dumps(
        {
            "schema": REBIN_SETTINGS_CLIPBOARD_SCHEMA,
            "version": REBIN_SETTINGS_CLIPBOARD_VERSION,
            "settings": settings,
        },
        indent=2,
        sort_keys=True,
    )


def _rebin_config_from_clipboard_text(
    text: str,
    target_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a pasted rebin config after validating target compatibility."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The clipboard does not contain valid nfit rebin settings.") from exc
    if not isinstance(payload, dict) or payload.get("schema") != REBIN_SETTINGS_CLIPBOARD_SCHEMA:
        raise ValueError("The clipboard does not contain nfit rebin settings.")
    if payload.get("version") != REBIN_SETTINGS_CLIPBOARD_VERSION:
        raise ValueError("The clipboard rebin-settings version is not supported.")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The clipboard rebin settings are incomplete.")
    if settings.get("metadata_dimensions") and "metadata_dimensions" not in target_config:
        raise ValueError(
            "Metadata rebin settings must be pasted into a source collection's composite panel."
        )
    if "metadata_dimensions" in settings:
        from .metadata_dimensions import MetadataDimension

        try:
            for recipe in settings["metadata_dimensions"]:
                MetadataDimension(**recipe)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid metadata rebin settings: {exc}") from exc
    source_axes = settings.get("axes")
    target_axes = target_config.get("axes")
    if not isinstance(source_axes, list) or not source_axes:
        raise ValueError("The copied rebin settings do not contain any axes.")
    if isinstance(target_axes, list) and target_axes and len(source_axes) != len(target_axes):
        raise ValueError(
            f"The copied settings have {len(source_axes)} axes, but this rebin panel has "
            f"{len(target_axes)} axes."
        )
    for index, source_axis in enumerate(source_axes):
        if not isinstance(source_axis, dict):
            raise ValueError(f"Copied rebin axis {index + 1} is invalid.")
        source_vector = source_axis.get("vector")
        target_vector = (
            target_axes[index].get("vector")
            if isinstance(target_axes, list)
            and index < len(target_axes)
            and isinstance(target_axes[index], dict)
            else None
        )
        if isinstance(source_vector, list) and isinstance(target_vector, list):
            if len(source_vector) != len(target_vector):
                raise ValueError(
                    f"Copied rebin axis {index + 1} uses a {len(source_vector)}-component "
                    f"coordinate vector, but this panel expects {len(target_vector)} components."
                )
    pasted = copy.deepcopy(target_config)
    for key in REBIN_SETTINGS_KEYS:
        if key in settings:
            pasted[key] = copy.deepcopy(settings[key])
    pasted["normalize"] = True
    pasted["stale"] = True
    return pasted


def dataset_mask_application_config(dataset: DatasetEntry) -> dict[str, Any]:
    """Return persistent automatic/manual mask materialization settings."""

    config = dataset.metadata.get(DATASET_MASK_APPLICATION_KEY)
    if not isinstance(config, dict):
        config = {}
        dataset.metadata[DATASET_MASK_APPLICATION_KEY] = config
    if "auto_apply" not in config:
        config["auto_apply"] = not _dataset_mask_is_large(dataset)
    config.setdefault("stale", False)
    return config


def _dataset_mask_point_count(dataset: DatasetEntry) -> int:
    data = dataset.data
    if isinstance(data, MDHistoData):
        return int(np.asarray(data.signal).size)
    if isinstance(data, PointData4D):
        return int(data.size)
    if isinstance(data, PointListData):
        return int(data.size)
    return 0


def _dataset_mask_is_large(dataset: DatasetEntry) -> bool:
    return _dataset_mask_point_count(dataset) > MASK_AUTO_MAX_POINTS


def _dataset_mask_auto_enabled(dataset: DatasetEntry) -> bool:
    return bool(dataset_mask_application_config(dataset).get("auto_apply", True))


def _dataset_mask_is_stale(dataset: DatasetEntry) -> bool:
    config = dataset.metadata.get(DATASET_MASK_APPLICATION_KEY)
    return bool(isinstance(config, dict) and config.get("stale", False))


def _should_defer_dataset_masks(dataset: DatasetEntry, *, force_masks: bool) -> bool:
    # Manual mode only materializes masks at explicit synchronization points.
    # An exact cached view is still returned before this policy is consulted,
    # so a current manual result remains cheap to reuse. This also protects a
    # freshly loaded project, where no process-local masked cache exists yet.
    return bool(not force_masks and not _dataset_mask_auto_enabled(dataset))


def _dataset_mask_status_text(dataset: DatasetEntry) -> str:
    config = dataset_mask_application_config(dataset)
    size_note = "large dataset" if _dataset_mask_is_large(dataset) else "small dataset"
    if bool(config.get("auto_apply", True)):
        if bool(config.get("stale", False)):
            return f"Automatic mask application is on ({size_note}); pending masks apply on the next refresh."
        return f"Automatic mask application is on ({size_note}); applied masks are current."
    if bool(config.get("stale", False)) or dataset.id not in _VIEWER_VIEW_CACHE:
        return "Manual mask application pending; press Apply masks now or start a fit/open the data viewer."
    return f"Manual mask application is on ({size_note}); applied masks are current."


def dataset_rebin_enabled(dataset: DatasetEntry) -> bool:
    """Return whether this dataset should use its rebinned representation."""

    config = dataset.parameters.get(DATASET_REBIN_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _dataset_rebin_source_points(dataset: DatasetEntry) -> int:
    return _dataset_data_point_count(dataset)


def _dataset_rebin_output_bins(config: dict[str, Any]) -> int:
    total = 1
    axes = config.get("axes")
    if not isinstance(axes, list) or not axes:
        return 0
    for axis in axes:
        if not isinstance(axis, dict):
            return 0
        total *= max(int(axis.get("num_bins", 1) or 1), 1)
    return int(total)


def _dataset_rebin_estimated_contributions(dataset: DatasetEntry, config: dict[str, Any]) -> int:
    source_points = _dataset_rebin_source_points(dataset)
    axes = config.get("axes", []) or [{}]
    multiplier = 2 ** sum(_rebin_fractional_axes(config, axes))
    return int(
        source_points
        * multiplier
        * _rebin_symmetry_count(config, _dataset_lattice_parameters(dataset))
    )


def _dataset_lattice_parameters(dataset: DatasetEntry) -> dict[str, Any] | None:
    data = dataset.data
    metadata = getattr(data, "metadata", None)
    return (
        metadata.get("lattice_parameters")
        if isinstance(metadata, dict) and isinstance(metadata.get("lattice_parameters"), dict)
        else None
    )


def _dataset_rebin_is_large(dataset: DatasetEntry, config: dict[str, Any]) -> bool:
    return (
        _dataset_rebin_estimated_contributions(dataset, config) > REBIN_AUTO_MAX_CONTRIBUTIONS
        or _dataset_rebin_output_bins(config) > REBIN_AUTO_MAX_OUTPUT_BINS
    )


def _dataset_rebin_status_text(dataset: DatasetEntry, config: dict[str, Any]) -> str:
    if not bool(config.get("enabled", False)):
        return "Rebinning is disabled."
    size_note = "large dataset" if _dataset_rebin_is_large(dataset, config) else "small dataset"
    if bool(config.get("auto_rebin", True)):
        if bool(config.get("stale", False)):
            return f"Automatic rebinning is on ({size_note}); pending edits will recompute on the next refresh."
        return f"Automatic rebinning is on ({size_note}); edits recompute the cached rebin."
    if bool(config.get("stale", False)):
        return "Manual rebin pending; press Rebin now or start a fit/view/export operation to compute it."
    return f"Manual rebinning is on ({size_note}); cached rebin is current."


def _dataset_rebin_auto_enabled(dataset: DatasetEntry) -> bool:
    config = dataset_rebin_config(dataset)
    return bool(config.get("auto_rebin", True))


def _dataset_rebin_is_stale(dataset: DatasetEntry) -> bool:
    config = dataset.parameters.get(DATASET_REBIN_KEY)
    return bool(isinstance(config, dict) and config.get("stale", False))


def _should_defer_dataset_rebin(dataset: DatasetEntry, *, force_rebin: bool) -> bool:
    return bool(
        dataset_rebin_enabled(dataset)
        and not force_rebin
        and _dataset_rebin_is_stale(dataset)
        and not _dataset_rebin_auto_enabled(dataset)
    )


def rebinned_dataset_data(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Return a rebinned copy using the saved configuration and worker ceiling."""
    from ._parallel import thread_budget

    config = dataset_rebin_config(dataset)
    with thread_budget(config.get("workers")):
        return _rebinned_dataset_data(
            dataset, extra_masks=extra_masks, progress_callback=progress_callback
        )


def _rebinned_dataset_data(
    dataset: DatasetEntry,
    *,
    extra_masks: list[MaskSpec] | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Return a rebinned copy of a supported dataset according to its configuration."""

    config = dataset_rebin_config(dataset)
    if isinstance(dataset.data, PointListData):
        result = _rebin_point_list_data(dataset, config)
        config["stale"] = False
        return result
    if isinstance(dataset.data, PointData4D):
        result = _rebin_point_data(
            _point_data_with_nfit_masks(dataset, dataset.data, extra_masks=extra_masks),
            config,
            progress_callback=progress_callback,
        )
        config["stale"] = False
        return result
    if not isinstance(dataset.data, MDHistoData):
        return dataset.data
    masked = _mdhisto_with_nfit_masks(dataset, data=dataset.data, extra_masks=extra_masks)
    result = _with_rebinned_mask_metadata(
        _rebin_mdhisto_data(masked, config, progress_callback=progress_callback),
        masked,
    )
    config["stale"] = False
    return result


def _with_rebinned_mask_metadata(rebinned: MDHistoData, source: MDHistoData) -> MDHistoData:
    """Annotate a rebinned MDHisto result whose source masks were applied up front."""

    metadata = dict(rebinned.metadata)
    rebin_metadata = dict(metadata.get("rebin", {}))
    for key in ("file_mask_count", "nfit_mask_count", "combined_mask_count"):
        if key in source.metadata:
            rebin_metadata[f"source_{key}"] = int(source.metadata[key])
    metadata["rebin"] = rebin_metadata
    output_mask = np.asarray(rebinned.mask, dtype=bool)
    metadata["file_mask"] = np.zeros(rebinned.shape, dtype=bool)
    metadata["nfit_mask"] = np.zeros(rebinned.shape, dtype=bool)
    metadata["file_mask_count"] = 0
    metadata["nfit_mask_count"] = 0
    metadata["combined_mask_count"] = int(np.count_nonzero(output_mask))
    return replace(rebinned, metadata=metadata)


def _rebin_point_list_data(dataset: DatasetEntry, config: dict[str, Any]) -> PointListData:
    """Rebin transformed point-list data over its coordinates into a histogram."""

    prepared = prepared_point_list_data(dataset)
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    coordinate_names = [
        axis.get("name") for axis in axes_config if axis.get("name") in prepared.columns
    ]
    if not coordinate_names:
        coordinate_names = list(prepared.coordinate_names)
    selected_axes = axes_config[: len(coordinate_names)]
    symmetry = _rebin_symmetry_matrices(config, prepared.metadata.get("lattice_parameters"))
    coordinates = np.column_stack([prepared.columns[name] for name in coordinate_names])
    data_bounds = (
        _symmetry_projected_coordinate_bounds(coordinates, symmetry, np.eye(len(coordinate_names)))
        if symmetry is not None
        else _finite_coordinate_bounds(coordinates)
    )
    selected_axes = _resolve_auto_rebin_axes(selected_axes, data_bounds)
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        coordinates,
        physical_coordinates=coordinates,
        symmetry=symmetry,
        output_basis=np.eye(len(coordinate_names)),
    )
    selected_axes = _resolve_data_driven_rebin_axes(config, selected_axes, mode_coordinates)
    lower = [axis["lower"] for axis in selected_axes] or None
    upper = [axis["upper"] for axis in selected_axes] or None
    result = prepared.rebin_to_histogram(
        coordinate_names,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, selected_axes),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, selected_axes),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        symmetry_operations=symmetry,
    )
    metadata = dict(result.metadata)
    symmetry_metadata = _rebin_symmetry_metadata(
        config, prepared.metadata.get("lattice_parameters")
    )
    if symmetry_metadata is not None:
        metadata.setdefault("rebin", {})["symmetry"] = symmetry_metadata
    return result.with_updates(metadata=metadata)


def create_rebinned_dataset(
    group: DataGroup,
    dataset: DatasetEntry,
    *,
    name: str | None = None,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    """Materialize a dataset's rebinned view as an independent dataset."""

    data = rebinned_dataset_data(
        dataset,
        extra_masks=effective_dataset_masks(group, dataset),
        progress_callback=progress_callback,
    )
    parameters = {}
    for key in (
        "temperature",
        "magnetic_field",
        KINEMATIC_KF_KI_INCLUDED_KEY,
        SPECTRAL_CHANNEL_CONFIG_KEY,
    ):
        if key in dataset.parameters:
            parameters[key] = copy.deepcopy(dataset.parameters[key])
    new_entry = DatasetEntry(
        name=_unique_dataset_name(name or f"{dataset.name} rebinned", group.dataset_names),
        data=data,
        kind=dataset.kind,
        metadata={
            **copy.deepcopy(dataset.metadata),
            "source_dataset": dataset.name,
            "rebin_materialized": True,
        },
        parameters=parameters,
        masks=copy.deepcopy(dataset.masks),
    )
    new_entry.replace_data(data, source_backed=False)
    group.add_dataset(new_entry)
    return new_entry


def _dataset_data_point_count(dataset: DatasetEntry) -> int:
    data = dataset.data
    if data is not None:
        count = _loaded_data_point_count(data)
        if dataset.data_matches_source:
            dataset.metadata["source_point_count"] = count
        return count
    stored_count = dataset.metadata.get("source_point_count")
    try:
        count = int(stored_count)
    except (TypeError, ValueError):
        count = -1
    if count >= 0:
        return count
    importer = IMPORTERS.get(str(dataset.metadata.get("importer", "")))
    source = dataset.metadata.get("source_file")
    if importer is not None and importer.point_counter is not None and source:
        try:
            options = dataset.metadata.get("import_options")
            count = int(
                importer.point_counter(source, options if isinstance(options, dict) else None)
            )
        except (OSError, TypeError, ValueError):
            count = -1
        if count >= 0:
            dataset.metadata["source_point_count"] = count
            return count
    if isinstance(dataset.metadata.get(DERIVED_RECIPE_KEY), dict):
        return _dataset_rebin_output_bins(dataset_rebin_config(dataset))
    return 0

# Imported after the dataset/view backend is defined: project_composites uses
# those lower-level operations while this module preserves the public API.
from . import project_composites as _project_composites  # noqa: E402

_project_composites.configure_composite_backend(globals())

_CompositeScope = _project_composites._CompositeScope
_composite_scope = _project_composites._composite_scope
_composite_root = _project_composites._composite_root
_composite_cache_key = _project_composites._composite_cache_key
data_group_composite_config = _project_composites.data_group_composite_config
_composite_source_points = _project_composites._composite_source_points
_dataset_collection_point_count = _project_composites._dataset_collection_point_count
_composite_output_bins = _project_composites._composite_output_bins
_composite_estimated_contributions = _project_composites._composite_estimated_contributions
_composite_rebin_is_large = _project_composites._composite_rebin_is_large
_composite_rebin_status_text = _project_composites._composite_rebin_status_text
_composite_auto_enabled = _project_composites._composite_auto_enabled
_composite_rebin_is_stale = _project_composites._composite_rebin_is_stale
_should_defer_composite_rebin = _project_composites._should_defer_composite_rebin
data_group_composite_enabled = _project_composites.data_group_composite_enabled
_composite_dataset_name = _project_composites._composite_dataset_name
_composite_candidates = _project_composites._composite_candidates
_hierarchical_composite_scopes = _project_composites._hierarchical_composite_scopes
_dataset_composite_kind = _project_composites._dataset_composite_kind
data_group_composite_status = _project_composites.data_group_composite_status
_composite_reference_data = _project_composites._composite_reference_data
_source_data_for_group_composite = _project_composites._source_data_for_group_composite
_composite_cache_signature = _project_composites._composite_cache_signature
composite_dataset_data = _project_composites.composite_dataset_data
_composite_progress_callback = _project_composites._composite_progress_callback
metadata_dimension_preview = _project_composites.metadata_dimension_preview
set_metadata_dimensions = _project_composites.set_metadata_dimensions
_metadata_composite_data = _project_composites._metadata_composite_data
_composite_dataset_data = _project_composites._composite_dataset_data
_apply_mdhisto_coverage_threshold = _project_composites._apply_mdhisto_coverage_threshold
_apply_composite_backgrounds = _project_composites._apply_composite_backgrounds
_cached_composite_dataset_data = _project_composites._cached_composite_dataset_data
composite_dataset_entry = _project_composites.composite_dataset_entry
materialize_composite_dataset = _project_composites.materialize_composite_dataset
_scaled_error_for_weight = _project_composites._scaled_error_for_weight
_dataset_statistical_weight = _project_composites._dataset_statistical_weight
_composite_rebin_bounds = _project_composites._composite_rebin_bounds
_composite_mdhisto_data = _project_composites._composite_mdhisto_data
_composite_point_data = _project_composites._composite_point_data
_composite_point_list_data = _project_composites._composite_point_list_data
_COMPOSITE_DATA_CACHE = _project_composites._COMPOSITE_DATA_CACHE
_COMPOSITE_DATA_CACHE_LIMIT = _project_composites._COMPOSITE_DATA_CACHE_LIMIT
_COMPOSITE_DATA_CACHE_MAX_BYTES = _project_composites._COMPOSITE_DATA_CACHE_MAX_BYTES
