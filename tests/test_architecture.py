from __future__ import annotations

import ast
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
