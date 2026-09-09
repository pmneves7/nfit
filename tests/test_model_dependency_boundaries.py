from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
MODEL_PIPELINE_MODULES = frozenset(
    {
        "electronic_pipeline_sampling",
        "fit_config",
        "model_plots",
        "model_registry",
        "report",
    }
)


def _focused_dependencies(module_name: str) -> set[str]:
    path = PACKAGE_ROOT / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                candidates = tuple(alias.name for alias in node.names)
            elif node.module:
                candidates = (
                    node.module.removeprefix("nfit.").split(".")[0],
                )
            else:
                continue
        elif isinstance(node, ast.Import):
            candidates = tuple(
                alias.name.removeprefix("nfit.").split(".")[0]
                for alias in node.names
            )
        else:
            continue
        dependencies.update(MODEL_PIPELINE_MODULES.intersection(candidates))
    return dependencies


def _dependency_cycle(graph: dict[str, set[str]]) -> tuple[str, ...]:
    visited: set[str] = set()
    active: list[str] = []

    def visit(module_name: str) -> tuple[str, ...]:
        if module_name in active:
            start = active.index(module_name)
            return (*active[start:], module_name)
        if module_name in visited:
            return ()
        active.append(module_name)
        for dependency in graph[module_name]:
            cycle = visit(dependency)
            if cycle:
                return cycle
        active.pop()
        visited.add(module_name)
        return ()

    for module_name in graph:
        cycle = visit(module_name)
        if cycle:
            return cycle
    return ()


def test_model_pipeline_dependency_graph_is_acyclic() -> None:
    graph = {
        module_name: _focused_dependencies(module_name)
        for module_name in MODEL_PIPELINE_MODULES
    }

    assert _dependency_cycle(graph) == ()


def test_registry_and_fit_config_keep_dependency_direction_one_way() -> None:
    assert _focused_dependencies("model_registry") == set()
    assert "model_plots" not in _focused_dependencies("fit_config")
