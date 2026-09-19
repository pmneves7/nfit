from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
GUI_INDEPENDENT_MODULES = (
    PACKAGE_ROOT / "array_archive.py",
    PACKAGE_ROOT / "analysis" / "artifacts.py",
    PACKAGE_ROOT / "corelli.py",
    PACKAGE_ROOT / "composite_spectral.py",
    PACKAGE_ROOT / "rebin_cache.py",
    PACKAGE_ROOT / "slice_viewer_cache.py",
    PACKAGE_ROOT / "project_composites.py",
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
PROJECT_GUI_CLIENT_MODULES = (
    PACKAGE_ROOT / "qt_widget_state.py",
    PACKAGE_ROOT / "qt_operation_guard.py",
    PACKAGE_ROOT / "project_composite_physics.py",
    PACKAGE_ROOT / "analysis_gui.py",
    PACKAGE_ROOT / "metadata_dimensions_gui.py",
    PACKAGE_ROOT / "performance_benchmark.py",
    PACKAGE_ROOT / "performance_gui.py",
    PACKAGE_ROOT / "preferences_gui.py",
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
