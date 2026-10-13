from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import nfit.project_data_panels as data_panels
import nfit.project_dataset_panels as dataset_panels
import nfit.project_group_panels as group_panels
import nfit.project_gui as project_gui
import nfit.project_readonly_panels as readonly_panels
from nfit.dataset import PointListData
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
    "_details_group_box": "project_readonly_panels.py",
    "_dataset_metadata_group_box": "project_readonly_panels.py",
    "_analysis_table_group_box": "project_readonly_panels.py",
    "_dataset_signal_semantics_group_box": "project_dataset_panels.py",
    "_dataset_spectral_channels_group_box": "project_dataset_panels.py",
    "_metadata_tree_group_box": "project_readonly_panels.py",
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
        if name == "_group_dataset_weights_group_box" or PANEL_BUILDERS[name] == "project_readonly_panels.py":
            assert returned.value.func.attr == name
            keywords = [keyword.arg for keyword in returned.value.keywords]
            if name != "_details_group_box":
                assert "context" in keywords
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
    ("project_group_panels.py", "project_dataset_panels.py", "project_readonly_panels.py"),
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


def test_readonly_builders_have_explicit_dependencies_and_compatibility_identity():
    tree = ast.parse((PACKAGE_ROOT / "project_readonly_panels.py").read_text(encoding="utf-8"))
    for name, owner in PANEL_BUILDERS.items():
        if owner != "project_readonly_panels.py":
            continue
        builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        loaded_names = {
            node.id for statement in builder.body for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert not loaded_names & {
            "FunctionType", "globals", "invoke_panel_builder", "helper_namespace",
            "_format_number", "_dataset_metadata_mapping", "_add_metadata_tree_item",
        }
        assert getattr(data_panels, name) is getattr(readonly_panels, name)
        assert getattr(dataset_panels, name) is getattr(readonly_panels, name)
    assert not any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(tree)
    )


def test_readonly_panels_construct_independently_with_isolated_contexts(inventory_qt):
    QtWidgets = inventory_qt
    data = PointListData({"Value": [1.25, np.nan], "Accepted": [1, 0]})
    panels = [readonly_panels._analysis_table_group_box(
        None, data,
        context=readonly_panels.AnalysisTablePanelContext(
            format_number=lambda value, prefix=prefix: f"{prefix}:{value}",
        ),
    ) for prefix in ("first", "second")]
    details = readonly_panels._details_group_box(None, "Empty", [])
    try:
        tables = [panel.findChild(QtWidgets.QTableWidget, "analysis_output_table") for panel in panels]
        assert [table.item(0, 0).text() for table in tables] == ["first:1.25", "second:1.25"]
        assert all(table.item(1, 0).text() == "-" for table in tables)
        assert all(table.item(1, 0).foreground().color().name() == "#d94b45" for table in tables)
        assert all(table.toolTip() and table.minimumHeight() == 180 and table.maximumHeight() == 420 for table in tables)
        assert details.findChild(QtWidgets.QLabel).text() == "-"
        assert np.isnan(data.column("Value")[1])
    finally:
        for panel in [*panels, details]:
            panel.close()


def test_dataset_metadata_context_preserves_lazy_payload_and_tree_override(inventory_qt):
    dataset = DatasetEntry("Unloaded", None, parameters={"temperature": 12})
    source_mapping = {"Experiment": "scan"}
    calls = []
    sentinel = object()
    context = readonly_panels.DatasetMetadataPanelContext(
        metadata_mapping=lambda candidate: (
            calls.append(candidate), source_mapping,
        )[1],
        metadata_tree=lambda *args, **kwargs: (
            calls.append((args, kwargs)), sentinel,
        )[1],
    )
    assert readonly_panels._dataset_metadata_group_box(None, dataset, context=context) is sentinel
    assert calls == [dataset, (("Metadata", {
        "Experiment": "scan", "Parameters": {"temperature": 12},
    }), {"object_name": "dataset_metadata_tree", "empty_text": "No additional metadata."})]
    assert source_mapping == {"Experiment": "scan"}
    assert dataset.data is None


def test_readonly_explorer_contexts_preserve_live_helpers_and_overrides(monkeypatch, inventory_qt):
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace")]))
    dataset = DatasetEntry("Unloaded", None)
    contexts = (
        explorer._dataset_metadata_panel_context(),
        explorer._analysis_table_panel_context(),
        explorer._metadata_tree_panel_context(),
    )
    calls = []
    sentinel = object()
    monkeypatch.setattr(project_gui, "_dataset_metadata_mapping", lambda candidate: {"Live": candidate.name})
    monkeypatch.setattr(project_gui, "_format_number", lambda value: f"live:{value}")
    monkeypatch.setattr(project_gui, "_add_metadata_tree_item", lambda *args: calls.append(args))
    monkeypatch.setattr(explorer, "_metadata_tree_group_box", lambda *args, **kwargs: (
        calls.append((args, kwargs)), sentinel,
    )[1])
    try:
        assert readonly_panels._dataset_metadata_group_box(None, dataset, context=contexts[0]) is sentinel
        assert calls[0][0] == ("Metadata", {"Live": "Unloaded"})
        assert contexts[1].format_number(3) == "live:3"
        contexts[2].add_item("parent", "key", "value")
        assert calls[1] == ("parent", "key", "value")
        assert dataset.data is None
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_metadata_tree_preserves_names_expansion_tooltips_and_render_bounds(inventory_qt):
    QtWidgets = inventory_qt
    nested = {"leaf": 1}
    for _ in range(8):
        nested = {"child": nested}
    panel = readonly_panels._metadata_tree_group_box(
        None, "Metadata", {"Parameters": {"temperature": 12}, "Sequence": list(range(70)), "Deep": nested},
        object_name="test_metadata_tree", empty_text="Empty",
        context=readonly_panels.MetadataTreePanelContext(add_item=project_gui._add_metadata_tree_item),
    )
    empty = readonly_panels._metadata_tree_group_box(
        None, "Metadata", {}, object_name="empty_tree", empty_text="No metadata",
        context=readonly_panels.MetadataTreePanelContext(add_item=lambda *args: pytest.fail("Empty metadata rendered an item")),
    )
    try:
        tree = panel.findChild(QtWidgets.QTreeWidget, "test_metadata_tree")
        assert tree.toolTip() and tree.minimumHeight() == 120 and tree.maximumHeight() == 260
        items = {tree.topLevelItem(index).text(0): tree.topLevelItem(index) for index in range(tree.topLevelItemCount())}
        assert items["Parameters"].isExpanded()
        assert items["Parameters"].child(0).toolTip(1) == "12"
        assert items["Sequence"].childCount() == 65
        assert items["Sequence"].child(64).text(1) == "6 more item(s)"
        item = items["Deep"]
        depth = 0
        while item.childCount():
            item = item.child(0)
            depth += 1
        assert depth == 6
        assert empty.findChild(QtWidgets.QTreeWidget) is None
        assert empty.findChild(QtWidgets.QLabel).text() == "No metadata"
    finally:
        panel.close()
        empty.close()


@pytest.mark.parametrize("builder_name", [name for name, owner in PANEL_BUILDERS.items() if owner == "project_readonly_panels.py"])
@pytest.mark.parametrize("keyword_arguments", [False, True])
def test_readonly_explorer_and_legacy_dispatch_bypass_rebinding(
    monkeypatch, inventory_qt, builder_name, keyword_arguments,
):
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace")]))
    arguments = {
        "_details_group_box": {"title": "Details", "lines": ["one", "two"]},
        "_dataset_metadata_group_box": {"dataset": DatasetEntry("Unloaded", None)},
        "_analysis_table_group_box": {"data": PointListData({"Value": [1]})},
        "_metadata_tree_group_box": {"title": "Metadata", "mapping": {"key": 1}},
    }[builder_name]
    kwargs = {"object_name": "legacy_tree", "empty_text": "Empty"} if builder_name == "_metadata_tree_group_box" else {}
    monkeypatch.setattr(data_panels, "FunctionType", lambda *args, **kwargs: pytest.fail("Read-only panel used namespace rebinding"))
    panels = []
    try:
        panels.append(data_panels.invoke_panel_builder(
            getattr(data_panels, builder_name), explorer,
            *(tuple(arguments.values()) if not keyword_arguments else ()),
            helper_namespace=vars(project_gui),
            **(arguments if keyword_arguments else {}), **kwargs,
        ))
        monkeypatch.setattr(data_panels, "invoke_panel_builder", lambda *args, **kwargs: pytest.fail("Explorer used namespace dispatch"))
        panels.append(getattr(explorer, builder_name)(**arguments, **kwargs))
        assert panels[0].title() == panels[1].title()
    finally:
        for panel in panels:
            panel.close()
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_legacy_readonly_dispatch_accepts_explicit_context_without_explorer(monkeypatch, inventory_qt):
    monkeypatch.setattr(data_panels, "FunctionType", lambda *args, **kwargs: pytest.fail("Explicit context used namespace rebinding"))
    sentinel = object()
    dataset = DatasetEntry("Unloaded", None)
    assert data_panels.invoke_panel_builder(
        data_panels._dataset_metadata_group_box, None, dataset=dataset,
        helper_namespace={},
        context=readonly_panels.DatasetMetadataPanelContext(
            metadata_mapping=lambda candidate: {},
            metadata_tree=lambda *args, **kwargs: sentinel,
        ),
    ) is sentinel
    table_panel = data_panels.invoke_panel_builder(
        data_panels._analysis_table_group_box, None, PointListData({"Value": [2]}),
        helper_namespace={},
        context=readonly_panels.AnalysisTablePanelContext(format_number=str),
    )
    tree_panel = data_panels.invoke_panel_builder(
        data_panels._metadata_tree_group_box, None, "Metadata", {},
        object_name="explicit_context_tree", empty_text="Empty", helper_namespace={},
        context=readonly_panels.MetadataTreePanelContext(add_item=lambda *args: pytest.fail("Empty metadata rendered an item")),
    )
    try:
        table = table_panel.findChild(inventory_qt.QTableWidget, "analysis_output_table")
        assert table.item(0, 0).text() == "2.0"
        assert table.item(0, 0).foreground().color().name() != "#d94b45"
        assert tree_panel.findChild(inventory_qt.QLabel).text() == "Empty"
    finally:
        table_panel.close()
        tree_panel.close()
