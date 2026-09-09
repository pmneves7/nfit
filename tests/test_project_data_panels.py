from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nfit.project_data_panels as data_panels
import nfit.project_gui as project_gui
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.project_gui import NfitProject, NfitProjectExplorer

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "nfit"
PANEL_BUILDERS = {
    "_group_dataset_weights_group_box": "project_group_panels.py",
    "_group_composite_group_box": "project_group_panels.py",
    "_dataset_point_list_group_box": "project_dataset_panels.py",
    "_heat_capacity_box": "project_dataset_panels.py",
    "_magnetization_absolute_box": "project_dataset_panels.py",
    "_point_list_scale_box": "project_dataset_panels.py",
    "_point_list_susceptibility_box": "project_dataset_panels.py",
    "_point_list_wavelength_box": "project_dataset_panels.py",
    "_dataset_type_group_box": "project_dataset_panels.py",
    "_dataset_axes_group_box": "project_dataset_panels.py",
    "_add_rebin_performance_controls": "project_dataset_panels.py",
    "_details_group_box": "project_dataset_panels.py",
    "_dataset_metadata_group_box": "project_dataset_panels.py",
    "_analysis_table_group_box": "project_dataset_panels.py",
    "_dataset_signal_semantics_group_box": "project_dataset_panels.py",
    "_dataset_spectral_channels_group_box": "project_dataset_panels.py",
    "_metadata_tree_group_box": "project_dataset_panels.py",
}


def test_explorer_panel_methods_are_thin_compatibility_delegates() -> None:
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
        if isinstance(node, ast.FunctionDef) and node.name in PANEL_BUILDERS
    }

    assert methods.keys() == PANEL_BUILDERS.keys()
    for name, method in methods.items():
        assert len(method.body) == 1
        returned = method.body[0]
        assert isinstance(returned, ast.Return)
        assert isinstance(returned.value, ast.Call)
        assert isinstance(returned.value.func, ast.Attribute)
        assert returned.value.func.attr == "invoke_panel_builder"
        assert any(
            isinstance(node, ast.Attribute) and node.attr == name
            for node in ast.walk(returned.value)
        )


@pytest.mark.parametrize(
    "module_path",
    ("project_group_panels.py", "project_dataset_panels.py"),
)
def test_panel_modules_are_focused_and_do_not_import_project_gui(
    module_path: str,
) -> None:
    tree = ast.parse((PACKAGE_ROOT / module_path).read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
    }
    assert functions == {
        name for name, owner in PANEL_BUILDERS.items() if owner == module_path
    }
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


def test_panel_dispatch_preserves_live_helpers_and_explorer_callbacks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Workspace")
    explorer = NfitProjectExplorer(NfitProject([group]))
    calls: list[tuple[DataGroup, bool]] = []

    monkeypatch.setattr(
        project_gui,
        "data_group_composite_status",
        lambda candidate: (candidate is group, "Injected compatibility status"),
    )
    monkeypatch.setattr(
        explorer,
        "_set_group_composite_enabled",
        lambda candidate, checked: calls.append((candidate, checked)),
    )

    panel = explorer._group_composite_group_box(group)
    try:
        assert any(
            label.text() == "Injected compatibility status"
            for label in panel.findChildren(QtWidgets.QLabel)
        )
        enabled = panel.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
        assert enabled is not None and enabled.toolTip()
        enabled.setChecked(not enabled.isChecked())
        assert calls == [(group, True)]
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_panel_callbacks_keep_live_project_gui_helper_lookup(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup(
        "Paged",
        datasets=[
            DatasetEntry(f"scan {index}", None, data_type="magnetization")
            for index in range(project_gui.DETAIL_DATASET_PAGE_SIZE + 1)
        ],
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    panel = explorer._group_dataset_weights_group_box(group)
    try:
        page = panel.findChild(QtWidgets.QComboBox, "group_datasets_page")
        table = panel.findChild(QtWidgets.QTableWidget, "group_datasets_table")
        assert page is not None and table is not None
        monkeypatch.setattr(project_gui, "data_type_label", lambda _kind: "Patched")
        page.setCurrentIndex(1)
        assert table.item(0, 1).text() == "Patched"
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_panel_facade_reexports_each_extracted_builder() -> None:
    assert all(getattr(data_panels, name) is not None for name in PANEL_BUILDERS)
