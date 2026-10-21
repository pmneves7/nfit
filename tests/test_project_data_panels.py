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
        assert returned.value.func.attr == name
        keywords = [keyword.arg for keyword in returned.value.keywords]
        if name != "_details_group_box":
            assert "context" in keywords
        assert not any(
            isinstance(node, ast.Name) and node.id == "globals"
            for node in ast.walk(returned.value)
        )
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
    assert not hasattr(data_panels, "FunctionType")
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
    assert not hasattr(data_panels, "FunctionType")
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
    assert not hasattr(data_panels, "FunctionType")
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


def test_all_panel_dependencies_are_explicit_and_facades_keep_identity():
    for filename in (
        "project_dataset_panels.py", "project_group_panels.py",
        "project_panel_contexts.py", "project_data_panels.py",
    ):
        tree = ast.parse((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Name) and node.id in {"FunctionType", "globals"}
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Constant) and node.value.value is None
            for node in tree.body
        )
        assert not any(
            isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("project_gui")
            for node in ast.walk(tree)
        )
        if filename in {"project_dataset_panels.py", "project_group_panels.py"}:
            # The compatibility self argument is never read or passed onward.
            assert not any(
                isinstance(node, ast.Name) and node.id == "self"
                and isinstance(node.ctx, ast.Load)
                for node in ast.walk(tree)
            )
    for name, owner in PANEL_BUILDERS.items():
        module = {
            "project_dataset_panels.py": dataset_panels,
            "project_group_panels.py": group_panels,
            "project_readonly_panels.py": readonly_panels,
        }[owner]
        assert getattr(data_panels, name) is getattr(module, name)


def _isolated_context(context_class, **capabilities):
    """Fail if a standalone panel unexpectedly reaches another capability."""
    from dataclasses import MISSING, fields

    def unexpected(*args, **kwargs):
        pytest.fail("Panel used an unprovided capability")

    required = {
        field.name: unexpected for field in fields(context_class)
        if field.default is MISSING and field.default_factory is MISSING
    }
    return context_class(**{**required, **capabilities})


@pytest.mark.parametrize("entry_kind", ["dataset", "named", "composite"])
def test_gui_viewer_wrapper_does_not_load_disabled_unavailable_source(monkeypatch, entry_kind):
    from nfit.project_composites import FIT_BINNING_ID
    from nfit.project_data import DATASET_REBIN_KEY, dataset_rebin_config

    source = DatasetEntry("Unloaded", None, kind="mdhisto", metadata={"source_file": "absent.npz"})
    config = dataset_rebin_config(source)
    config.update(enabled=True, binning_enabled=False)
    if entry_kind == "named":
        alias = source.copy()
        alias._viewer_source_entry = source
        alias._viewer_rebin_config = {**config, "_binning_id": "named"}
        alias._viewer_cache_id = "named"
    elif entry_kind == "composite":
        alias = source
        alias.parameters.pop(DATASET_REBIN_KEY)
        alias._viewer_composite_cache_owner = (id(source), "composite")
        alias._viewer_composite_binning_id = FIT_BINNING_ID
        alias._viewer_binning_config = config
    else:
        alias = source

    def forbidden(*_args, **_kwargs):
        pytest.fail("Disabled viewer output tried to load or calculate its source")

    monkeypatch.setattr(project_gui, "_ensure_dataset_data_loaded", forbidden)
    monkeypatch.setattr(project_gui._project_data, "_ensure_dataset_data_loaded", forbidden)
    with pytest.raises(ValueError, match="disabled.*enable"):
        project_gui.dataset_for_slice_viewer(alias)
    assert source.data is None


def test_editable_point_list_controls_use_isolated_contexts(inventory_qt):
    QtWidgets = inventory_qt
    dataset = DatasetEntry("Values", PointListData({"X": [1], "Y": [2]}))
    calls = []
    panels = []
    try:
        for prefix in ("first", "second"):
            context = _isolated_context(
                data_panels.PointListScalePanelContext,
                set_point_list_scale=lambda *args, prefix=prefix: calls.append((prefix, args)),
            )
            panels.append(dataset_panels._point_list_scale_box(
                None, dataset, None, {"scale": {"factor": 1}}, ["X", "Y"], context=context,
            ))
        for value, panel in enumerate(panels, start=2):
            spin = panel.findChild(QtWidgets.QDoubleSpinBox)
            assert spin.toolTip()
            spin.setValue(value)
            spin.editingFinished.emit()
        assert calls == [
            ("first", (dataset, None, "factor", 2.0)),
            ("second", (dataset, None, "factor", 3.0)),
        ]
        assert dataset.data.column("Y").tolist() == [2]
    finally:
        for panel in panels:
            panel.close()


def test_point_list_parent_and_type_panel_preserve_callbacks(inventory_qt):
    QtWidgets = inventory_qt
    dataset = DatasetEntry("Values", PointListData({"X": [1], "Y": [2]}), data_type="custom")
    calls = []
    context = _isolated_context(
        data_panels.DatasetPointListPanelContext,
        point_list_config=lambda candidate: {"coordinate_names": ["X"], "channels": []},
        bold_label=lambda widgets, text: widgets.QLabel(text),
        set_point_list_coordinates=lambda *args: calls.append(args),
    )
    panel = dataset_panels._dataset_point_list_group_box(None, dataset, None, context=context)
    type_panel = dataset_panels._dataset_type_group_box(
        None, dataset, None, ["Description"],
        context=data_panels.DatasetTypePanelContext(
            available_data_types=lambda: [("custom", "Custom"), ("magnetization", "Moment")],
            set_selected_data_type=lambda *args: calls.append(args),
        ),
    )
    try:
        coordinates = panel.findChild(QtWidgets.QLineEdit, "point_list_coordinates")
        coordinates.setText("Y")
        coordinates.editingFinished.emit()
        type_panel.findChild(QtWidgets.QComboBox, "dataset_data_type").setCurrentIndex(1)
        assert calls == [(dataset, None, "Y"), (dataset, None, "magnetization")]
        assert dataset.data_type == "custom"
    finally:
        panel.close()
        type_panel.close()


def test_dataset_rebin_panel_constructs_without_an_explorer(inventory_qt):
    from nfit.dataset import PointData4D
    from nfit.project_data import dataset_rebin_binnings

    QtWidgets = inventory_qt
    data = PointData4D([0], [0], [0], [1], [2], [1])
    dataset = DatasetEntry("Points", data)
    binning = dataset_rebin_binnings(dataset)[0]
    binning["config"]["enabled"] = True
    calls = []
    context = _isolated_context(
        data_panels.DatasetAxesPanelContext,
        dataset_can_rebin=lambda candidate: True,
        selected_dataset_binning=lambda candidate: binning,
        dataset_rebin_symmetry_preview=lambda *args: "Preview",
        format_number=lambda value: str(value),
        add_rebin_performance_controls=lambda *args, **kwargs: None,
        set_dataset_rebin_enabled=lambda *args: calls.append(args),
        rebin_dataset_now=lambda *args: calls.append(args),
    )
    panel = dataset_panels._dataset_axes_group_box(None, dataset, None, [], context=context)
    try:
        assert panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_binning_enabled").isChecked()
        enabled = panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
        assert enabled.isChecked() and enabled.toolTip()
        enabled.setChecked(False)
        panel.findChild(QtWidgets.QPushButton, "dataset_rebin_now").click()
        assert calls == [(dataset, None, False), (dataset, None)]
        assert dataset.data is data
    finally:
        panel.close()


def test_composite_panel_constructs_with_explicit_nested_builders_and_state(inventory_qt):
    from nfit.project_composites import data_group_composite_binnings, data_group_composite_config

    QtWidgets = inventory_qt
    group = DataGroup("Composite")
    data_group_composite_config(group)["enabled"] = True
    binning = data_group_composite_binnings(group)[0]
    calls = []
    context = _isolated_context(
        data_panels.GroupCompositePanelContext,
        selected_composite_binning=lambda candidate: binning,
        data_group_composite_status=lambda candidate: (True, "Standalone status"),
        expanded_inactive_composite_ids=lambda: set(),
        project=lambda: NfitProject([group]),
        format_number=lambda value: str(value),
        saved_binning_compressed_size=lambda *args, **kwargs: None,
        metadata_rebin_rows=lambda candidate, layout, row: (calls.append("rows"), 0)[1],
        metadata_dimensions_panel=lambda candidate, **kwargs: QtWidgets.QLabel("Metadata override"),
        composite_physics_panel=lambda candidate, config: QtWidgets.QLabel("Physics override"),
        add_rebin_performance_controls=lambda *args, **kwargs: None,
        rebin_composite_now=lambda candidate: calls.append(candidate),
    )
    panel = group_panels._group_composite_group_box(None, group, context=context)
    try:
        assert panel.findChild(QtWidgets.QCheckBox, "group_composite_binning_enabled").isChecked()
        tabs = panel.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Rebin settings", "Metadata dimensions", "Physics", "Bin information",
        ]
        panel.findChild(QtWidgets.QPushButton, "group_composite_rebin_now").click()
        assert calls == ["rows", group]
        assert not group.datasets
    finally:
        panel.close()


@pytest.mark.parametrize("selected_fit", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_dataset_binning_enabled_controls_calculation_and_forwards_callback(
    inventory_qt, selected_fit, enabled,
):
    import copy

    from nfit.dataset import PointData4D
    from nfit.project_data import dataset_rebin_binnings

    QtWidgets = inventory_qt
    dataset = DatasetEntry("Points", PointData4D([0], [0], [0], [1], [2], [1]))
    fit = dataset_rebin_binnings(dataset)[0]
    fit["config"]["enabled"] = True
    visual = {"id": "visual", "name": "Visualization", "fit": False,
              "config": copy.deepcopy(fit["config"])}
    binnings = [fit, visual]
    selected = fit if selected_fit else visual
    selected["config"]["binning_enabled"] = enabled
    calls = []
    context = _isolated_context(
        data_panels.DatasetAxesPanelContext,
        dataset_can_rebin=lambda candidate: True,
        dataset_rebin_binnings=lambda candidate: binnings,
        selected_dataset_binning=lambda candidate: selected,
        dataset_rebin_symmetry_preview=lambda *args: "Preview",
        format_number=str,
        add_rebin_performance_controls=lambda *args, **kwargs: None,
        peek_cached_dataset_view=lambda *args, **kwargs: None,
        set_dataset_binning_enabled=lambda *args: calls.append(args),
        set_dataset_rebin_axis_value=lambda *args: calls.append(args),
    )
    panel = dataset_panels._dataset_axes_group_box(None, dataset, None, [], context=context)
    try:
        check = panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_binning_enabled")
        assert check.text() == "Enabled" and check.isChecked() is enabled
        assert "project storage" in check.toolTip() and "stale" in check.toolTip()
        assert panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled").isChecked()
        for name in ("dataset_rebin_now", "dataset_rebin_create", "dataset_rebin_save"):
            button = panel.findChild(QtWidgets.QPushButton, name)
            assert button.isEnabled() is enabled and button.toolTip()
        assert panel.findChild(QtWidgets.QPushButton, "dataset_rebin_all_now").isEnabled()
        editor = panel.findChild(QtWidgets.QLineEdit, "dataset_rebin_axis_lower_0")
        assert editor.isEnabled()
        editor.setText("-2")
        editor.editingFinished.emit()
        check.setChecked(not enabled)
        assert calls == [(dataset, None, 0, "lower", "-2"), (dataset, None, not enabled)]
        assert selected["config"]["enabled"] is True
    finally:
        panel.close()


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("selected_fit", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_composite_binning_enabled_keeps_recipes_editable_and_forwards_scope(
    inventory_qt, nested, selected_fit, enabled,
):
    import copy

    from nfit.dataset import PointData4D
    from nfit.project_composites import _composite_scope, data_group_composite_binnings

    QtWidgets = inventory_qt
    dataset = DatasetEntry("Points", PointData4D([0], [0], [0], [1], [2], [1]))
    node = DatasetGroup("Collection", datasets=[dataset])
    root = DataGroup(
        "Workspace", subgroups=[node] if nested else [], datasets=[] if nested else [dataset],
    )
    group = _composite_scope(root, node) if nested else root
    fit = data_group_composite_binnings(group)[0]
    fit["config"]["enabled"] = True
    visual = {"id": "visual", "name": "Visualization", "fit": False,
              "config": copy.deepcopy(fit["config"])}
    binnings = [fit, visual]
    selected = fit if selected_fit else visual
    selected["config"]["binning_enabled"] = enabled
    calls = []
    context = _isolated_context(
        data_panels.GroupCompositePanelContext,
        data_group_composite_binnings=lambda candidate: binnings,
        selected_composite_binning=lambda candidate: selected,
        data_group_composite_status=lambda candidate: (True, "Compatible"),
        expanded_inactive_composite_ids=lambda: set(),
        project=lambda: NfitProject([root]),
        format_number=str,
        saved_binning_compressed_size=lambda *args, **kwargs: None,
        metadata_rebin_rows=lambda candidate, layout, row: 0,
        metadata_dimensions_panel=lambda *args, **kwargs: QtWidgets.QWidget(),
        composite_physics_panel=lambda *args: None,
        add_rebin_performance_controls=lambda *args, **kwargs: None,
        peek_cached_composite_dataset_data=lambda *args, **kwargs: None,
        set_group_composite_binning_enabled=lambda *args: calls.append(args),
        set_group_composite_axis_value=lambda *args: calls.append(args),
    )
    panel = group_panels._group_composite_group_box(None, group, context=context)
    try:
        check = panel.findChild(QtWidgets.QCheckBox, "group_composite_binning_enabled")
        assert check.text() == "Enabled" and check.isChecked() is enabled
        assert "project storage" in check.toolTip() and "stale" in check.toolTip()
        assert panel.findChild(QtWidgets.QCheckBox, "group_composite_enabled").isChecked()
        for name in ("group_composite_rebin_now", "group_composite_create"):
            button = panel.findChild(QtWidgets.QPushButton, name)
            assert button.isEnabled() is enabled and button.toolTip()
        assert panel.findChild(QtWidgets.QPushButton, "group_composite_rebin_all_now").isEnabled()
        editor = panel.findChild(QtWidgets.QLineEdit, "group_composite_axis_lower_0")
        assert editor.isEnabled()
        editor.setText("-2")
        editor.editingFinished.emit()
        check.setChecked(not enabled)
        assert calls == [(group, 0, "lower", "-2"), (group, not enabled)]
        assert selected["config"]["enabled"] is True
    finally:
        panel.close()


def test_editable_context_helpers_and_callbacks_remain_live_after_construction(monkeypatch, inventory_qt):
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace")]))
    context = explorer._point_list_scale_panel_context()
    rebin_context = explorer._dataset_axes_panel_context()
    composite_context = explorer._group_composite_panel_context()
    calls = []
    monkeypatch.setattr(explorer, "_set_point_list_scale", lambda *args: calls.append(args))
    monkeypatch.setattr(project_gui, "_format_number", lambda value: f"live:{value}")
    monkeypatch.setattr(explorer, "materialize_rebin_for_selection", lambda: calls.append("materialize"))
    monkeypatch.setattr(explorer, "_set_dataset_binning_enabled", lambda *args: calls.append(args))
    monkeypatch.setattr(explorer, "_set_group_composite_binning_enabled", lambda *args: calls.append(args))
    try:
        context.set_point_list_scale("dataset", "group", "factor", 2)
        rebin_context.materialize_rebin_for_selection()
        rebin_context.set_dataset_binning_enabled("dataset", "group", False)
        composite_context.set_group_composite_binning_enabled("group", False)
        assert calls == [
            ("dataset", "group", "factor", 2), "materialize",
            ("dataset", "group", False), ("group", False),
        ]
        assert rebin_context.format_number(7) == "live:7"
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_compatibility_dispatch_ignores_namespace_and_accepts_standalone_editable_context(inventory_qt):
    class UnreadableNamespace(dict):
        def __getitem__(self, key):
            pytest.fail("Compatibility dispatcher read helper_namespace")

    dataset = DatasetEntry("Unloaded", None)
    context = data_panels.DatasetTypePanelContext(
        available_data_types=lambda: [("custom", "Custom")],
        set_selected_data_type=lambda *args: None,
    )
    panel = data_panels.invoke_panel_builder(
        data_panels._dataset_type_group_box, None, dataset=dataset, group=None, lines=[],
        context=context, helper_namespace=UnreadableNamespace(),
    )
    try:
        assert panel.findChild(inventory_qt.QComboBox, "dataset_data_type").currentText() == "Custom"
        assert dataset.data is None
    finally:
        panel.close()
