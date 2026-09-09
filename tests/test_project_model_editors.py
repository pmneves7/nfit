from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nfit.project_lindhard_editor as lindhard_editor
import nfit.project_model_editor as model_editor
import nfit.project_tight_binding_editor as tight_binding_editor
from nfit import fit_config, model_registry, project_gui
from nfit.pipeline import DataGroup
from nfit.project_gui import NfitProject, NfitProjectExplorer, create_model_component

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
DELEGATE_MODULES = {
    "_rebuild_model_parameter_editor": "_project_model_editor",
    "_rebuild_model_parameter_editor_preserving_scroll": "_project_model_editor",
    "_build_tight_binding_state_editor": "_project_tight_binding_editor",
    "_tight_binding_dos_sampling_status": "_project_tight_binding_editor",
    "_build_tight_binding_dos_sampling_editor": "_project_tight_binding_editor",
    "_organize_tight_binding_editor": "_project_tight_binding_editor",
    "_lindhard_source_component": "_project_lindhard_editor",
    "_build_lindhard_editor": "_project_lindhard_editor",
    "_organize_lindhard_editor": "_project_lindhard_editor",
    "_build_tight_binding_editor": "_project_tight_binding_editor",
    "_build_tight_binding_orbital_editor": "_project_tight_binding_editor",
    "_build_tight_binding_spin_editor": "_project_tight_binding_editor",
    "_tight_binding_sharing_controls": "_project_tight_binding_editor",
    "_build_tight_binding_onsite_editor": "_project_tight_binding_editor",
    "_build_tight_binding_hopping_editor": "_project_tight_binding_editor",
}


def test_project_gui_preserves_model_editor_helper_exports() -> None:
    assert (
        project_gui.parameter_is_derived_by_closure
        is fit_config.parameter_is_derived_by_closure
    )
    assert project_gui.model_config_tooltip is model_registry.model_config_tooltip
    assert (
        project_gui.model_parameter_tooltip
        is model_registry.model_parameter_tooltip
    )


def test_explorer_model_editor_compatibility_methods_are_thin_delegates() -> None:
    tree = ast.parse(
        (PACKAGE_ROOT / "project_gui.py").read_text(encoding="utf-8")
    )
    explorer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NfitProjectExplorer"
    )
    methods = {
        node.name: node
        for node in explorer.body
        if isinstance(node, ast.FunctionDef) and node.name in DELEGATE_MODULES
    }

    assert methods.keys() == DELEGATE_MODULES.keys()
    for name, module_name in DELEGATE_MODULES.items():
        method = methods[name]
        assert len(method.body) == 1
        returned = method.body[0]
        assert isinstance(returned, ast.Return)
        assert isinstance(returned.value, ast.Call)
        assert isinstance(returned.value.func, ast.Attribute)
        assert returned.value.func.attr == name
        assert isinstance(returned.value.func.value, ast.Name)
        assert returned.value.func.value.id == module_name


@pytest.mark.parametrize(
    "module_path",
    (
        "project_model_editor.py",
        "project_lindhard_editor.py",
        "project_tight_binding_editor.py",
    ),
)
def test_focused_editor_modules_do_not_import_project_gui(module_path: str) -> None:
    tree = ast.parse((PACKAGE_ROOT / module_path).read_text(encoding="utf-8"))
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
    assert not any(name.endswith("project_gui") for name in imported_modules)


def test_extracted_editor_builders_preserve_tight_binding_and_lindhard_tabs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    calls: list[tuple[str, str]] = []
    original_generic = model_editor._rebuild_model_parameter_editor
    original_tight_binding = tight_binding_editor._build_tight_binding_editor
    original_lindhard = lindhard_editor._build_lindhard_editor

    def record_generic(explorer, model):
        calls.append(("generic", model.type))
        return original_generic(explorer, model)

    def record_tight_binding(explorer, model):
        calls.append(("tight_binding", model.type))
        return original_tight_binding(explorer, model)

    def record_lindhard(explorer, model):
        calls.append(("lindhard", model.type))
        return original_lindhard(explorer, model)

    monkeypatch.setattr(
        model_editor,
        "_rebuild_model_parameter_editor",
        record_generic,
    )
    monkeypatch.setattr(
        tight_binding_editor,
        "_build_tight_binding_editor",
        record_tight_binding,
    )
    monkeypatch.setattr(
        lindhard_editor,
        "_build_lindhard_editor",
        record_lindhard,
    )

    group = DataGroup("Electronic")
    bands = create_model_component(group, "bands", type="tight_binding")
    response = create_model_component(group, "response", type="lindhard")
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer._refresh_tree(select_group=group, select_model=bands)
    tight_binding_tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget,
        "tight_binding_builder_tabs",
    )
    assert tight_binding_tabs is not None
    assert [
        tight_binding_tabs.tabText(index)
        for index in range(tight_binding_tabs.count())
    ] == [
        "Structure and basis",
        "Hamiltonian",
        "Calculate and inspect",
        "Advanced",
    ]

    explorer._refresh_tree(select_group=group, select_model=response)
    lindhard_tabs = explorer.model_parameter_widget.findChild(
        QtWidgets.QTabWidget,
        "lindhard_builder_tabs",
    )
    assert lindhard_tabs is not None
    assert [
        lindhard_tabs.tabText(index) for index in range(lindhard_tabs.count())
    ] == [
        "Response",
        "Sampling",
        "Experimental coupling",
        "Calculate and inspect",
        "Advanced",
    ]
    assert ("tight_binding", "tight_binding") in calls
    assert ("lindhard", "lindhard") in calls
    assert ("generic", "tight_binding") in calls
    assert ("generic", "lindhard") in calls
    explorer.has_unsaved_changes = False
    explorer.window.close()
