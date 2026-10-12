from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nfit.project_data_panels as data_panels
import nfit.project_group_panels as group_panels
import nfit.project_gui as project_gui
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
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
        if name == "_group_dataset_weights_group_box":
            assert returned.value.func.attr == name
            assert [keyword.arg for keyword in returned.value.keywords] == ["context"]
            assert not any(
                isinstance(node, ast.Name) and node.id == "globals"
                for node in ast.walk(returned.value)
            )
        else:
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
        monkeypatch.setattr(project_gui, "_dataset_data_point_count", lambda _dataset: 42)
        monkeypatch.setattr(project_gui, "_format_number", lambda value: f"live:{value}")
        page.setCurrentIndex(1)
        assert table.item(0, 1).text() == "Patched"
        assert table.item(0, 2).text() == "live:42"
        assert table.item(0, 3).text() == "live:1.0"
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_panel_facade_reexports_each_extracted_builder() -> None:
    assert all(getattr(data_panels, name) is not None for name in PANEL_BUILDERS)


@pytest.fixture
def inventory_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield QtWidgets
    assert application is not None


def test_inventory_constructs_independently_with_bounded_isolated_contexts(inventory_qt):
    QtWidgets = inventory_qt
    group = DataGroup("Runs", datasets=[DatasetEntry(f"run {index}", None) for index in range(5)])
    seen = [[], []]
    corners = []

    def context(index, page_size):
        def dataset_cells(dataset):
            seen[index].append(dataset.id)
            return (dataset.name, f"context {index}", "0", "1", "1")

        return data_panels.GroupInventoryPanelContext(
            node=group,
            page_size=lambda: page_size,
            dataset_cells=dataset_cells,
            child_collection_cells=lambda child: pytest.fail("Dataset panel queried child rows"),
            set_corner_tooltip=lambda table, text: corners.append((table, text)),
        )

    panels = [
        group_panels._group_dataset_weights_group_box(None, None, context=context(0, 2)),
        group_panels._group_dataset_weights_group_box(None, None, context=context(1, 1)),
    ]
    try:
        tables = [panel.findChild(QtWidgets.QTableWidget, "group_datasets_table") for panel in panels]
        assert [table.rowCount() for table in tables] == [2, 1]
        assert [len(calls) for calls in seen] == [2, 1]
        assert all(dataset.data is None for dataset in group.datasets)
        panels[0].findChild(QtWidgets.QComboBox, "group_datasets_page").setCurrentIndex(2)
        assert tables[0].item(0, 0).text() == "run 4"
        assert tables[0].item(0, 1).text() == "context 0"
        assert tables[1].item(0, 0).text() == "run 0"
        assert tables[1].item(0, 1).text() == "context 1"
        assert [len(calls) for calls in seen] == [3, 1]
        assert len(corners) == 2 and all(text for _, text in corners)
    finally:
        for panel in panels:
            panel.close()


@pytest.mark.parametrize("with_children", [False, True])
def test_inventory_empty_and_child_selections_use_only_their_capabilities(inventory_qt, with_children):
    QtWidgets = inventory_qt
    child = DatasetGroup("Child", datasets=[DatasetEntry("run", None)])
    group = DataGroup("Parent", subgroups=[child] if with_children else [])
    seen = []

    def child_cells(candidate):
        seen.append(candidate)
        return (candidate.name, "Scoped composite", "1", "1", "Yes")

    context = data_panels.GroupInventoryPanelContext(
        node=group,
        page_size=lambda: 20,
        dataset_cells=lambda dataset: pytest.fail("Selection queried dataset rows"),
        child_collection_cells=child_cells,
        set_corner_tooltip=lambda table, text: table.setToolTip(text),
    )
    panel = group_panels._group_dataset_weights_group_box(None, None, context=context)
    try:
        if with_children:
            table = panel.findChild(QtWidgets.QTableWidget, "group_child_collections_table")
            assert table.rowCount() == 1
            assert table.item(0, 1).text() == "Scoped composite"
            assert seen == [child]
        else:
            table = panel.findChild(QtWidgets.QTableWidget, "group_datasets_table")
            assert table.rowCount() == 0
            assert panel.findChild(QtWidgets.QLabel, "group_datasets_page_label").text() == "No direct datasets"
            assert seen == []
    finally:
        panel.close()


def test_inventory_explorer_delegate_bypasses_namespace_rebinding(monkeypatch, inventory_qt):
    group = DataGroup("Workspace")
    explorer = NfitProjectExplorer(NfitProject([group]))
    calls = []
    original = explorer._group_inventory_panel_context
    monkeypatch.setattr(explorer, "_group_inventory_panel_context", lambda candidate: (
        calls.append(candidate), original(candidate)
    )[1])
    monkeypatch.setattr(data_panels, "invoke_panel_builder", lambda *args, **kwargs: (
        pytest.fail("Inventory panel used namespace rebinding")
    ))
    panel = explorer._group_dataset_weights_group_box(group)
    try:
        assert calls == [group]
        assert panel.findChild(inventory_qt.QTableWidget, "group_datasets_table") is not None
        assert data_panels._group_dataset_weights_group_box is group_panels._group_dataset_weights_group_box
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_inventory_builder_has_no_namespace_or_placeholder_dependencies():
    tree = ast.parse((PACKAGE_ROOT / "project_group_panels.py").read_text(encoding="utf-8"))
    placeholders = {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    }
    builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_group_dataset_weights_group_box"
    )
    loaded_names = {
        node.id for node in ast.walk(builder)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    # Compatibility-only postponed annotations may name legacy scope types.
    body_names = {
        node.id for statement in builder.body for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not body_names & placeholders
    assert not loaded_names & {"FunctionType", "globals", "invoke_panel_builder", "helper_namespace"}


@pytest.mark.parametrize("keyword_group", [False, True])
def test_legacy_inventory_dispatch_builds_explicit_context_without_rebinding(
    monkeypatch, inventory_qt, keyword_group,
):
    group = DataGroup("Workspace", datasets=[DatasetEntry("run", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    monkeypatch.setattr(data_panels, "FunctionType", lambda *args, **kwargs: (
        pytest.fail("Inventory compatibility dispatch used namespace rebinding")
    ))
    monkeypatch.setattr(project_gui, "data_type_label", lambda _kind: "Compatibility label")
    panel = data_panels.invoke_panel_builder(
        data_panels._group_dataset_weights_group_box,
        explorer,
        *(() if keyword_group else (group,)),
        helper_namespace=vars(project_gui),
        **({"group": group} if keyword_group else {}),
    )
    try:
        table = panel.findChild(inventory_qt.QTableWidget, "group_datasets_table")
        assert table.item(0, 1).text() == "Compatibility label"
    finally:
        panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()
