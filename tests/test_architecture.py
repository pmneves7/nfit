from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def test_package_and_unit_tests_do_not_import_external_reduction_engines():
    # Actual engine comparisons live in manual diagnostics outside pytest.
    violations = []
    for root in (PACKAGE_ROOT, Path(__file__).parent):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules = (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(module.split(".")[0] in {"mantid", "shiver"} for module in modules):
                    violations.append(f"{path}:{node.lineno}")
    assert not violations, "External reduction-engine imports: " + ", ".join(violations)


def test_project_cache_facade_shares_authoritative_stores():
    from nfit import project_caches, project_gui

    assert project_gui._MODEL_OVERLAY_CACHE is project_caches.MODEL_OVERLAY_CACHE
    assert project_gui._MODEL_OVERLAY_ERRORS is project_caches.MODEL_OVERLAY_ERRORS
    assert project_gui.clear_project_caches is project_caches.clear_project_caches


def test_raw_geometry_service_is_authoritative_and_has_no_reducer_dependencies():
    from nfit import raw_dgs, raw_dgs_geometry

    for name in ("evaluate_log_expression", "location_transform", "resolved_idf_xml",
                 "resolved_geometry_signature", "source_distance"):
        assert getattr(raw_dgs, name) is getattr(raw_dgs_geometry, name)
    tree = ast.parse((PACKAGE_ROOT / "raw_dgs_geometry.py").read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(module.startswith(("raw_dgs", "project_", "PySide", "PyQt", "qt_")) for module in modules)


GUI_INDEPENDENT_MODULES = (
    PACKAGE_ROOT / "measurement_contracts.py",
    PACKAGE_ROOT / "measurement_statistics.py",
    PACKAGE_ROOT / "measurement_profiles.py",
    PACKAGE_ROOT / "measurement_aggregation.py",
    PACKAGE_ROOT / "measurement_waterfalls.py",
    PACKAGE_ROOT / "measurement_fit_data.py",
    PACKAGE_ROOT / "measurement_likelihoods.py",
    PACKAGE_ROOT / "measurement_dependencies.py",
    PACKAGE_ROOT / "measurement_rebinning.py",
    PACKAGE_ROOT / "measurement_regions.py",
    PACKAGE_ROOT / "measurement_replay.py",
    PACKAGE_ROOT / "measurement_point_bins.py",
    PACKAGE_ROOT / "measurement_scaling.py",
    PACKAGE_ROOT / "measurement_diagnostics.py",
    PACKAGE_ROOT / "measurement_smoothing.py",
    PACKAGE_ROOT / "composite_workflow.py",
    PACKAGE_ROOT / "point_data_archive.py",
    PACKAGE_ROOT / "project_caches.py",
    PACKAGE_ROOT / "figure_export.py",
    PACKAGE_ROOT / "resource_usage.py",
    PACKAGE_ROOT / "axes_ratio.py",
    PACKAGE_ROOT / "box_cuts.py",
    PACKAGE_ROOT / "cached_background_replay.py",
    PACKAGE_ROOT / "background_profile_queries.py",
    PACKAGE_ROOT / "array_archive.py",
    PACKAGE_ROOT / "analysis" / "artifacts.py",
    PACKAGE_ROOT / "corelli.py",
    PACKAGE_ROOT / "corelli_constants.py",
    PACKAGE_ROOT / "raw_dgs_cache.py",
    PACKAGE_ROOT / "raw_dgs_monitors.py",
    PACKAGE_ROOT / "raw_dgs_geometry_precision.py",
    PACKAGE_ROOT / "raw_dgs_geometry.py",
    PACKAGE_ROOT / "raw_dgs_hyspec.py",
    PACKAGE_ROOT / "raw_dgs_goniometer.py",
    PACKAGE_ROOT / "raw_dgs_pulses.py",
    PACKAGE_ROOT / "mdevent_detector_masks.py",
    PACKAGE_ROOT / "histogram_reduction.py",
    PACKAGE_ROOT / "dgs_normalization.py",
    PACKAGE_ROOT / "dgs_reduction_policy.py",
    PACKAGE_ROOT / "dgs_reduction_settings.py",
    PACKAGE_ROOT / "reduction_recipes.py",
    PACKAGE_ROOT / "reduction_runtime.py",
    PACKAGE_ROOT / "source_selection.py",
    PACKAGE_ROOT / "source_lineage.py",
    PACKAGE_ROOT / "source_selection_imports.py",
    PACKAGE_ROOT / "composite_spectral.py",
    PACKAGE_ROOT / "rebin_cache.py",
    PACKAGE_ROOT / "slice_viewer_cache.py",
    PACKAGE_ROOT / "project_composites.py",
    PACKAGE_ROOT / "project_background_cache.py",
    PACKAGE_ROOT / "composite_scaling.py",
    PACKAGE_ROOT / "project_cache_compat.py",
    PACKAGE_ROOT / "project_clipboard.py",
    PACKAGE_ROOT / "project_data.py",
    PACKAGE_ROOT / "project_derived_grid.py",
    PACKAGE_ROOT / "project_history.py",
    PACKAGE_ROOT / "project_imports.py",
    PACKAGE_ROOT / "project_io.py",
    PACKAGE_ROOT / "project_models.py",
    PACKAGE_ROOT / "project_summary.py",
    PACKAGE_ROOT / "workflow.py",
    PACKAGE_ROOT / "kpath.py",
    PACKAGE_ROOT / "analysis" / "runner.py",
)


@pytest.mark.parametrize("name", ["measurement_contracts", "measurement_statistics", "measurement_profiles", "measurement_aggregation", "measurement_waterfalls", "measurement_likelihoods"])
def test_measurement_contract_services_have_no_gui_or_project_dependencies(name):
    tree = ast.parse((PACKAGE_ROOT / f"{name}.py").read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(module.startswith(("PySide", "PyQt", "project_", "qt_")) for module in modules)


def test_measurement_contract_public_exports_are_authoritative():
    import nfit
    from nfit import measurement_contracts, measurement_statistics

    assert nfit.MeasurementContract is measurement_contracts.MeasurementContract
    for name in ("MeasurementEstimate", "SourceTerm", "estimate_measurement_bin"):
        assert getattr(nfit, name) is getattr(measurement_statistics, name)


def test_measurement_diagnostic_and_replay_exports_are_authoritative():
    import nfit
    from nfit import (
        composite_workflow,
        measurement_smoothing,
        source_selection_imports,
    )

    # Import modules explicitly: the same-named public function is intentional.
    measurement_diagnostics = importlib.import_module("nfit.measurement_diagnostics")
    for module, names in (
        (measurement_diagnostics, ("measurement_diagnostics", "measurement_diagnostics_text", "poisson_interval_channels")),
        (measurement_smoothing, ("smooth_count_histogram",)),
        (composite_workflow, ("export_composite_recipe", "replay_composite_recipe")),
        (source_selection_imports, ("update_source_selection", "SourceSelectionEdit")),
    ):
        for name in names:
            assert getattr(nfit, name) is getattr(module, name)


def test_measurement_workflow_public_exports_are_authoritative():
    import nfit
    from nfit import (
        measurement_dependencies,
        measurement_fit_data,
        measurement_likelihoods,
        measurement_point_bins,
        measurement_rebinning,
        measurement_regions,
        measurement_replay,
        measurement_scaling,
    )
    for module, names in (
        (measurement_dependencies, ("SourceDependencies", "CountingDependencies", "SourceReplayRequired")),
        (measurement_rebinning, ("coarsen_measurement_histogram", "combine_measurement_histograms")),
        (measurement_point_bins, ("bin_measurement_points",)),
        (measurement_regions, ("estimate_measurement_region",)),
        (measurement_replay, ("replay_measurement_histogram",)),
        (measurement_scaling, ("scale_measurement_data",)),
        (measurement_fit_data, ("prepare_histogram_fit_points",)),
        (measurement_likelihoods, ("PoissonCountModel", "poisson_deviance_residuals")),
    ):
        for name in names:
            assert getattr(nfit, name) is getattr(module, name)


PROJECT_GUI_CLIENT_MODULES = (
    PACKAGE_ROOT / "qt_figure_export.py",
    PACKAGE_ROOT / "qt_resource_monitor.py",
    PACKAGE_ROOT / "qt_box_cut_viewers.py",
    PACKAGE_ROOT / "qt_widget_state.py",
    PACKAGE_ROOT / "qt_operation_guard.py",
    PACKAGE_ROOT / "project_composite_physics.py",
    PACKAGE_ROOT / "analysis_gui.py",
    PACKAGE_ROOT / "metadata_dimensions_gui.py",
    PACKAGE_ROOT / "performance_benchmark.py",
    PACKAGE_ROOT / "performance_gui.py",
    PACKAGE_ROOT / "preferences_gui.py",
    PACKAGE_ROOT / "dgs_reduction_policy_gui.py",
    PACKAGE_ROOT / "reduction_recipe_gui.py",
    PACKAGE_ROOT / "collection_workflow_gui.py",
    PACKAGE_ROOT / "source_selection_gui.py",
)


def _imports_project_gui(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "project_gui" or alias.name.endswith(".project_gui")
                for alias in node.names
            ):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports_gui_module = module in {"", "nfit"} and any(
                alias.name == "project_gui" for alias in node.names
            )
            if module in {"project_gui", "nfit.project_gui"} or imports_gui_module:
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", GUI_INDEPENDENT_MODULES, ids=lambda path: path.stem)
def test_project_services_do_not_import_project_gui(path: Path) -> None:
    assert _imports_project_gui(path) == []


def test_project_clipboard_service_does_not_import_qt() -> None:
    tree = ast.parse((PACKAGE_ROOT / "project_clipboard.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(module.startswith(("PySide", "PyQt")) for module in modules)


def test_box_cut_profiles_have_no_gui_imports() -> None:
    tree = ast.parse((PACKAGE_ROOT / "box_cuts.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith(("PySide", "PyQt", "qt_", "project_gui")) for name in imported)


def test_composite_cache_compat_is_stdlib_only_and_owns_signature_tag() -> None:
    compat = ast.parse((PACKAGE_ROOT / "project_cache_compat.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(compat)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(compat)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imports <= {"json", "math", "typing"}

    composites = (PACKAGE_ROOT / "project_composites.py").read_text(encoding="utf-8")
    assert "from .project_cache_compat import COMPOSITE_CACHE_SIGNATURE_TAG" in composites
    assert '"event-scales-and-measured-background-replay-v2"' not in composites


@pytest.mark.parametrize("module", ["project_composites", "project_derived_grid"])
def test_composite_service_does_not_import_project_data_facade(module) -> None:
    tree = ast.parse(
        (PACKAGE_ROOT / f"{module}.py").read_text(encoding="utf-8")
    )
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "project_data" not in imported_modules
    assert "nfit.project_data" not in imported_modules
    assert not any(module.startswith(("PySide", "PyQt")) for module in imported_modules)


@pytest.mark.parametrize("path", PROJECT_GUI_CLIENT_MODULES, ids=lambda path: path.stem)
def test_project_gui_clients_use_focused_services(path: Path) -> None:
    assert _imports_project_gui(path) == []


def test_package_initializer_does_not_eagerly_import_project_gui() -> None:
    tree = ast.parse(
        (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"),
        filename=str(PACKAGE_ROOT / "__init__.py"),
    )
    top_level_imports = {
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert not top_level_imports.intersection(
        _imports_project_gui(PACKAGE_ROOT / "__init__.py")
    )


def test_package_public_api_keeps_gui_owned_names_lazy() -> None:
    code = """
import sys
import nfit

assert "nfit.project_gui" not in sys.modules
assert nfit.NfitProject.__module__ == "nfit.project_io"
assert nfit.composite_dataset_data.__module__ == "nfit.project_composites"
assert "save_project" in dir(nfit)
save_project = nfit.save_project
assert "nfit.project_gui" in sys.modules
assert save_project is sys.modules["nfit.project_gui"].save_project
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_composite_service_and_facade_import_cleanly_in_a_fresh_process() -> None:
    code = """
from nfit import project_composites, project_data

assert project_data.composite_dataset_data is project_composites.composite_dataset_data
assert project_data._COMPOSITE_DATA_CACHE is project_composites._COMPOSITE_DATA_CACHE
assert project_data.refresh_composite_dataset is project_composites.refresh_composite_dataset
assert project_data._composite_histograms_current is project_composites._composite_histograms_current
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("name", "module_name"),
    (
        ("BackgroundSpec", "nfit.pipeline"),
        ("DATA_TYPE_DEFINITIONS", "nfit.project_imports"),
        ("GROUP_COMPOSITE_KEY", "nfit.project_imports"),
        ("MDHistoAxis", "nfit.mdhisto"),
        ("MDHistoChannel", "nfit.mdhisto"),
        ("dataset_artifact_bytes", "nfit.analysis.artifacts"),
        ("bin_mdevent_group", "nfit.mdevent"),
        ("bin_mdevent_powder_group", "nfit.mdevent"),
        ("bin_raw_dgs_group", "nfit.raw_dgs"),
        ("mdhisto_coverage_fraction", "nfit.mdhisto"),
        ("mdhisto_measured_bins", "nfit.mdhisto"),
        ("rebin_nd", "nfit.rebin"),
        ("rebin_nd_symmetry", "nfit.rebin"),
        ("replace_dataset_artifact", "nfit.project_archive"),
        ("resolve_symmetry", "nfit.symmetry"),
        ("signal_semantics", "nfit.analysis.coordinates"),
        ("symmetry_spec_from_config", "nfit.symmetry"),
        ("with_paired_spectral_channels", "nfit.spectral_channels"),
    ),
)
def test_project_data_preserves_dependency_exports(name: str, module_name: str) -> None:
    project_data = importlib.import_module("nfit.project_data")
    source_module = importlib.import_module(module_name)

    assert getattr(project_data, name) is getattr(source_module, name)


@pytest.mark.parametrize("module_name", ("project_composites", "project_data"))
def test_project_data_services_do_not_import_qt(module_name: str) -> None:
    tree = ast.parse((PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("PySide" in module or module.startswith("qt_") for module in imported_modules)


@pytest.mark.parametrize("module", ["raw_dgs_monitors", "raw_dgs_pulses", "raw_dgs_goniometer",
                                    "histogram_reduction", "dgs_normalization"])
def test_reduction_services_do_not_import_their_coordinators(module):
    tree = ast.parse((PACKAGE_ROOT / f"{module}.py").read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported.intersection({"raw_dgs", "plotting_core", "project_rebinning"})


def test_raw_dgs_preserves_monitor_helper_import_identity():
    from nfit import raw_dgs, raw_dgs_monitors

    for name in ["TOF_US_PER_M_SQRT_MEV", "_mantid_getei_peak_region", "_mantid_getei_v2_peak"]:
        assert getattr(raw_dgs, name) is getattr(raw_dgs_monitors, name)


def test_sample_goniometer_uses_authoritative_pulse_timestamps_and_selection():
    from nfit import raw_dgs_goniometer, raw_dgs_pulses

    assert raw_dgs_goniometer.nexus_timestamps is raw_dgs_pulses.nexus_timestamps
    assert raw_dgs_goniometer.select_pulses is raw_dgs_pulses.select_pulses
    assert raw_dgs_pulses._timestamps is raw_dgs_pulses.nexus_timestamps


def test_background_panel_builder_has_no_reverse_coordinator_import():
    tree = ast.parse((PACKAGE_ROOT / "project_background_panels.py").read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "project_gui" not in imported
    coordinator = ast.parse((PACKAGE_ROOT / "project_gui.py").read_text())
    method = next(node for node in ast.walk(coordinator) if isinstance(node, ast.FunctionDef) and node.name == "_set_background_details")
    assert any(isinstance(node, ast.ImportFrom) and node.module == "project_background_panels" for node in ast.walk(method))


def test_corelli_backends_share_independent_authoritative_kinematics():
    from nfit import corelli, corelli_constants, dgs_reduction_policy

    assert corelli.ENERGY_TO_K2 is corelli_constants.ENERGY_TO_K2
    assert corelli.CORELLI_TOF_US_PER_M_SQRT_MEV is corelli_constants.CORELLI_TOF_US_PER_M_SQRT_MEV
    assert corelli.ENERGY_TO_K2 == 2.072124855
    assert corelli.ENERGY_TO_K2 != dgs_reduction_policy.ENERGY_TO_K2
    if corelli._CORELLI_NUMBA is not None:
        assert corelli._CORELLI_NUMBA.ENERGY_TO_K2 is corelli_constants.ENERGY_TO_K2
        assert corelli._CORELLI_NUMBA.TOF_FACTOR is corelli_constants.CORELLI_TOF_US_PER_M_SQRT_MEV
    tree = ast.parse((PACKAGE_ROOT / "corelli_constants.py").read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports.intersection({"corelli", "mdevent", "raw_dgs", "dgs_reduction_policy"})


def test_histogram_fit_preparation_gui_compatibility_uses_public_service():
    import numpy as np

    from nfit import project_gui
    from nfit.mdhisto import MDHistoAxis, MDHistoData
    from nfit.measurement_fit_data import prepare_histogram_fit_points
    data = MDHistoData((MDHistoAxis("DeltaE", [0.,1.,2.], "meV", "energy"),), np.array([1.,2.]), np.ones(2), np.zeros(2,bool), np.ones(2))
    expected, actual = prepare_histogram_fit_points(data), project_gui._point_data_from_mdhisto_view(data)
    np.testing.assert_array_equal(actual.intensity, expected.intensity)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    assert actual.metadata == expected.metadata


def test_source_selection_public_exports_are_authoritative():
    import nfit
    from nfit import source_selection, source_selection_imports

    for name in ("SourceSelection", "SourceSelectionPlan", "SourceAppearance", "parse_run_expression", "resolve_source_selection"):
        assert getattr(nfit, name) is getattr(source_selection, name)
    for name in ("import_source_selection", "source_selection_script"):
        assert getattr(nfit, name) is getattr(source_selection_imports, name)
