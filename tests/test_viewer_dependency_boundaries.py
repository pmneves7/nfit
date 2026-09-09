from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).parents[1] / "src" / "nfit"
VIEWER_MODULES = {
    "plotting",
    "plotting_core",
    "qt_slice_builder",
    "qt_slice_modes",
    "qt_slice_viewer",
    "qt_volume_viewer",
    "slice_viewer_state",
}


def _viewer_dependencies(module_name: str) -> set[str]:
    tree = ast.parse((SOURCE_DIR / f"{module_name}.py").read_text(encoding="utf-8"))
    dependencies = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        dependency = node.module.split(".", maxsplit=1)[0]
        if dependency in VIEWER_MODULES:
            dependencies.add(dependency)
    return dependencies


def test_viewer_module_dependencies_are_acyclic():
    graph = {name: _viewer_dependencies(name) for name in VIEWER_MODULES}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module_name: str, path: tuple[str, ...]) -> None:
        if module_name in visiting:
            cycle_start = path.index(module_name)
            cycle = (*path[cycle_start:], module_name)
            pytest.fail(f"viewer dependency cycle: {' -> '.join(cycle)}")
        if module_name in visited:
            return
        visiting.add(module_name)
        for dependency in graph[module_name]:
            visit(dependency, (*path, module_name))
        visiting.remove(module_name)
        visited.add(module_name)

    for module_name in graph:
        visit(module_name, ())


@pytest.mark.parametrize(
    "modules",
    [
        ("nfit.qt_volume_viewer", "nfit.qt_slice_viewer", "nfit.plotting"),
        ("nfit.plotting", "nfit.slice_viewer_state", "nfit.qt_slice_viewer"),
    ],
)
def test_viewer_modules_import_cleanly_in_fresh_process(modules):
    pytest.importorskip("PySide6")
    script = "\n".join(
        [
            "import importlib",
            *(f"importlib.import_module({module_name!r})" for module_name in modules),
            "from nfit import plotting, plotting_core",
            "assert plotting.MDHistoSliceViewer is plotting_core.MDHistoSliceViewer",
            "assert plotting._resolve_mdhisto_dim is plotting_core._resolve_mdhisto_dim",
        ]
    )
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
