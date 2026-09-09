from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
GUI_INDEPENDENT_MODULES = (
    PACKAGE_ROOT / "project_data.py",
    PACKAGE_ROOT / "project_history.py",
    PACKAGE_ROOT / "project_imports.py",
    PACKAGE_ROOT / "project_io.py",
    PACKAGE_ROOT / "project_models.py",
    PACKAGE_ROOT / "workflow.py",
    PACKAGE_ROOT / "analysis" / "runner.py",
)
PROJECT_GUI_CLIENT_MODULES = (
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
assert nfit.composite_dataset_data.__module__ == "nfit.project_data"
assert "save_project" in dir(nfit)
save_project = nfit.save_project
assert "nfit.project_gui" in sys.modules
assert save_project is sys.modules["nfit.project_gui"].save_project
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_project_data_service_does_not_import_qt() -> None:
    tree = ast.parse(
        (PACKAGE_ROOT / "project_data.py").read_text(encoding="utf-8")
    )
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
