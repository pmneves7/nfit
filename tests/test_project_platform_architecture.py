from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"


def test_binning_compatibility_facades_share_the_authoritative_service():
    from nfit import project_cache_compat, project_gui, project_io

    assert project_io._binning_signatures_match is project_cache_compat.binning_signatures_match
    assert project_gui._binning_signatures_match is project_cache_compat.binning_signatures_match
    assert project_io.composite_cache_signatures_match is project_cache_compat.composite_cache_signatures_match


@pytest.mark.parametrize("name", [
    "project_resources", "data_workspace", "workspace_sessions", "mapped_archive",
    "resource_budget", "storage_budget", "project_paths", "project_cache_compat",
])
def test_storage_and_resource_services_have_no_gui_imports(name):
    tree = ast.parse((PACKAGE_ROOT / f"{name}.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            if node.module in (None, "nfit"):
                modules.extend(alias.name for alias in node.names)
    assert not any(
        module.startswith(("PySide", "PyQt"))
        or any(part == "project_gui" or part.startswith("qt_") for part in module.split("."))
        for module in modules
    )


def test_cache_compatibility_service_does_not_import_serialization_or_preparation():
    tree = ast.parse((PACKAGE_ROOT / "project_cache_compat.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            if node.module in (None, "nfit"):
                modules.extend(alias.name for alias in node.names)
    assert not any(module.split(".")[-1] in {"project_io", "project_gui", "project_data"} for module in modules)


@pytest.mark.parametrize("token, allowed", [
    (["memory", 0, 12345], True), (["memory", 2, 12345], True),
    (["memory", True, 12345], False), (["memory", 0, True], False),
    (["memory", -1, 12345], False), (["memory", 0, 0], False),
])
def test_legacy_derived_migration_requires_a_valid_memory_identity(token, allowed):
    from nfit.project_cache_compat import binning_signatures_match

    current = [["derived_recipe", "result-id"], 1, "derived_recipe", {}, [], [], None, {"operation": "sum"}]
    saved = [token, *current[1:]]
    assert binning_signatures_match(json.dumps(saved), json.dumps(current)) is allowed


def test_legacy_derived_migration_rejects_unresolved_recipes_and_changed_sources():
    from nfit.project_cache_compat import binning_signatures_match

    current = [["derived_recipe", "result-id"], 1, "derived_recipe", {}, [], [], None, None]
    saved = [["memory", 0, 12345], *current[1:]]
    assert not binning_signatures_match(json.dumps(saved), json.dumps(current))
    current[-1] = {"operation": "sum", "source_ids": ["source-id"]}
    saved[-1] = {"operation": "sum", "source_ids": ["other-source-id"]}
    assert not binning_signatures_match(json.dumps(saved), json.dumps(current))
