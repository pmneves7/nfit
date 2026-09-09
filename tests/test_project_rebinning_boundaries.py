from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nfit.project_data as project_data
import nfit.project_rebinning as project_rebinning

PACKAGE_ROOT = Path(project_data.__file__).parent

REBIN_SERVICE_EXPORTS = (
    "_axis_bounds",
    "_axis_mode_coordinates_with_symmetry",
    "_cluster_coordinate_centers",
    "_composite_rebin_bin_edges",
    "_composite_rebin_step_sizes",
    "_coordinate_center_edges",
    "_default_rebin_axes",
    "_finite_coordinate_bounds",
    "_mdhisto_axis_edges",
    "_mdhisto_cell_volumes",
    "_mdhisto_rebin_axis_variable",
    "_mdhisto_rebin_basis_bounds",
    "_mdhisto_rebin_basis_transform",
    "_mdhisto_rebin_component",
    "_migrate_rebin_axis_modes",
    "_momentum_coordinate_variables",
    "_momentum_rebin_axis_indices",
    "_momentum_rebin_matrix",
    "_momentum_rebin_vector_text",
    "_normalize_rebin_bin_edges",
    "_num_bins_from_step_size",
    "_output_bin_volumes",
    "_parameter_to_text",
    "_point_data_histogram",
    "_point_data_rebin_basis_bounds",
    "_rebin_axis_bound_is_auto",
    "_rebin_axis_mode",
    "_rebin_axis_name",
    "_rebin_axis_vector",
    "_rebin_config_basis_bounds",
    "_rebin_fractional_axes",
    "_rebin_grid_kwargs",
    "_rebin_max_batch_bytes",
    "_rebin_max_batch_mb",
    "_rebin_mdhisto_coverage",
    "_rebin_mdhisto_data",
    "_rebin_mean_weighting",
    "_rebin_minimum_coverage",
    "_rebin_minimum_samples",
    "_rebin_point_data",
    "_rebin_resolution_mode",
    "_rebin_symmetry_count",
    "_rebin_symmetry_matrices",
    "_rebin_symmetry_metadata",
    "_rebin_symmetry_operations",
    "_resolve_auto_rebin_axes",
    "_resolve_data_driven_rebin_axes",
    "_sanitize_rebin_axis_config",
    "_step_size_from_bounds",
    "_symmetry_projected_coordinate_bounds",
    "_update_mdhisto_rebin_basis",
    "_update_rebin_momentum_basis",
    "_update_rebin_momentum_matrix",
    "_validate_mdhisto_rebin_basis",
)


@pytest.mark.parametrize("name", REBIN_SERVICE_EXPORTS)
def test_project_data_reexports_rebin_service_objects(name: str) -> None:
    assert getattr(project_data, name) is getattr(project_rebinning, name)


def test_rebin_service_does_not_import_project_facade_or_qt() -> None:
    tree = ast.parse((PACKAGE_ROOT / "project_rebinning.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert "project_data" not in imported_modules
    assert not any("PySide" in module or module.startswith("qt") for module in imported_modules)


def test_mixed_axis_grid_configuration_matches_facade_contract() -> None:
    config = {"resolution_mode": "step"}
    axes = [
        {"mode": "step", "lower": -1.0, "upper": 1.0, "step_size": 1.0, "num_bins": 3},
        {
            "mode": "edges",
            "lower": 0.0,
            "upper": 2.0,
            "step_size": 1.0,
            "num_bins": 2,
            "bin_edges": [0.0, 0.5, 2.0],
        },
        {
            "mode": "discrete",
            "lower": 10.0,
            "upper": 20.0,
            "step_size": 10.0,
            "num_bins": 2,
            "bin_edges": [5.0, 15.0, 25.0],
        },
    ]

    assert project_data._rebin_grid_kwargs(config, axes) == {
        "step_size": [1.0, 1.0, 10.0],
        "bin_edges": [None, [0.0, 0.5, 2.0], [5.0, 15.0, 25.0]],
    }
    assert project_data._rebin_fractional_axes(config, axes) == [True, True, False]


@pytest.mark.parametrize("service", (project_data, project_rebinning))
def test_rebin_service_uses_live_facade_defaults(monkeypatch, service) -> None:
    monkeypatch.setattr(project_data, "DEFAULT_REBIN_MAX_BATCH_MB", 37)
    monkeypatch.setattr(project_data, "DEFAULT_MINIMUM_COVERAGE", 0.25)
    monkeypatch.setattr(project_data, "DEFAULT_MINIMUM_SAMPLES", 2.5)
    monkeypatch.setattr(project_data, "REBIN_RESOLUTION_MODE_KEY", "legacy_mode")
    monkeypatch.setattr(project_data, "REBIN_AXIS_MODES", frozenset({"legacy"}))

    assert service._rebin_max_batch_mb({}) == 37
    assert service._rebin_minimum_coverage({}) == 0.25
    assert service._rebin_minimum_samples({}) == 2.5
    assert service._rebin_resolution_mode({"legacy_mode": "bins"}) == "bins"
    assert service._rebin_axis_mode({}, {"mode": "legacy"}) == "legacy"
