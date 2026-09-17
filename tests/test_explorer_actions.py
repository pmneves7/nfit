"""Explorer deletion semantics for leaves, folders, and lazy run pages."""

from __future__ import annotations

import pytest

from nfit import project_clipboard
from nfit.analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from nfit.pipeline import BackgroundSpec, PlotEntry
from nfit.project_gui import ensure_fit_history
from tests.project_gui_test_support import (
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    NfitProject,
    NfitProjectExplorer,
    create_mask,
)


def _item_with_role(explorer, role: str):
    def visit(item):
        if explorer._objects_for_item(item)[4] == role:
            return item
        for index in range(item.childCount()):
            found = visit(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(explorer.tree.topLevelItemCount()):
        found = visit(explorer.tree.topLevelItem(index))
        if found is not None:
            return found
    raise AssertionError(f"missing tree role {role!r}")


def test_deleting_unexpanded_lazy_page_removes_its_payload_only(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    datasets = [DatasetEntry(f"run {index}", None) for index in range(151)]
    group = DataGroup("Workspace", datasets=datasets)
    explorer = NfitProjectExplorer(NfitProject([group]))
    group_item = explorer.tree.topLevelItem(0)
    group_item.setExpanded(True)
    page = _item_with_role(explorer, "dataset_page")
    assert page.child(0).text(0) == "Load runs..."

    explorer.tree.setCurrentItem(page)
    explorer.delete_selected()

    assert [dataset.name for dataset in group.datasets] == [
        f"run {index}" for index in range(50, 151)
    ]
    assert explorer.tree.topLevelItem(0).isExpanded()
    # Rebuilding after page deletion preserves unrelated top-level sections.
    assert [explorer.tree.topLevelItem(0).child(index).text(0) for index in range(5)] == [
        "Datasets", "Models", "Fits", "Analyses", "Plots"
    ]


def test_page_and_loaded_child_are_deduplicated_for_copy_and_delete(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    datasets = [DatasetEntry(f"run {index}", None) for index in range(151)]
    group = DataGroup("Workspace", datasets=datasets)
    explorer = NfitProjectExplorer(NfitProject([group]))
    page = _item_with_role(explorer, "dataset_page")
    page.setExpanded(True)
    child = page.child(0)
    explorer.tree.setCurrentItem(page)
    page.setSelected(True)
    child.setSelected(True)

    explorer.copy_selected()
    assert len(explorer._clipboard.items) == 50
    assert len({item.id for item in explorer._clipboard.items}) == 50

    explorer.delete_selected()
    assert [dataset.name for dataset in group.datasets] == [
        f"run {index}" for index in range(50, 151)
    ]


def test_copy_combines_a_page_with_a_dataset_from_another_page(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    datasets = [DatasetEntry(f"run {index}", None) for index in range(151)]
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace", datasets=datasets)]))
    datasets_item = _item_with_role(explorer, "datasets")
    pages = [
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if explorer._objects_for_item(datasets_item.child(index))[4] == "dataset_page"
    ]
    pages[1].setExpanded(True)
    child_on_second_page = pages[1].child(0)
    explorer.tree.setCurrentItem(pages[0])
    pages[0].setSelected(True)
    child_on_second_page.setSelected(True)

    explorer.copy_selected()
    copied = explorer._clipboard.items
    assert len(copied) == 51
    assert {item.name for item in copied} == {
        *(f"run {index}" for index in range(50)),
        "run 50",
    }


def test_multiple_unexpanded_pages_and_nested_groups_delete_once(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    datasets = [DatasetEntry(f"run {index}", None) for index in range(151)]
    child = DatasetGroup("child", datasets=[DatasetEntry("nested run", None)])
    parent = DatasetGroup("parent", subgroups=[child])
    group = DataGroup("Workspace", datasets=datasets, subgroups=[parent])
    explorer = NfitProjectExplorer(NfitProject([group]))
    pages = []
    datasets_item = _item_with_role(explorer, "datasets")
    for index in range(datasets_item.childCount()):
        item = datasets_item.child(index)
        if explorer._objects_for_item(item)[4] == "dataset_page":
            pages.append(item)
    assert len(pages) == 4
    explorer.tree.setCurrentItem(pages[1])
    pages[0].setSelected(True)
    pages[1].setSelected(True)
    assert all(page.child(0).text(0) == "Load runs..." for page in pages[:2])
    explorer.delete_selected()
    assert [item.name for item in group.datasets] == [
        f"run {index}" for index in range(100, 151)
    ]

    parent_item = _item_with_role(explorer, "dataset_group")
    child_item = next(
        parent_item.child(index)
        for index in range(parent_item.childCount())
        if explorer._objects_for_item(parent_item.child(index))[4] == "dataset_group"
    )
    explorer.tree.setCurrentItem(parent_item)
    parent_item.setSelected(True)
    child_item.setSelected(True)
    explorer.delete_selected()
    assert group.subgroups == []


def test_folder_delete_keeps_workspace_structure_and_initial_fit(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", None)
    create_mask(dataset, "mask")
    group = DataGroup("Workspace", datasets=[dataset])
    group.models["component"] = object()
    ensure_fit_history(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    models = _item_with_role(explorer, "models")
    assert "Delete" in explorer.context_menu_action_names(models)
    explorer.tree.setCurrentItem(models)
    explorer.delete_selected()
    assert group.models == {}
    assert _item_with_role(explorer, "models").text(0) == "Models"

    initial = next(entry for entry in group.fits if entry.kind == "initial")
    initial_item = _item_with_role(explorer, "fit")
    explorer.tree.setCurrentItem(initial_item)
    explorer.delete_selected()
    assert initial in group.fits

    fits = _item_with_role(explorer, "fits")
    explorer.tree.setCurrentItem(fits)
    explorer.delete_selected()
    assert group.fits == [initial]
    assert _item_with_role(explorer, "fits").text(0) == "Fits"


def test_protected_initial_fit_and_analysis_output_have_no_delete_or_copy(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", None)
    analysis = AnalysisEntry("analysis", "identity", [dataset.id], {})
    analysis.result = AnalysisResultRecord(
        recipe_hash="recipe",
        input_fingerprints={},
        outputs=[AnalysisOutputRef("summary", "Summary", "scalar", scalar_value=1.0)],
        status="success",
        created_at="now",
    )
    group = DataGroup("Workspace", datasets=[dataset], analyses=[analysis])
    ensure_fit_history(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    initial = _item_with_role(explorer, "fit")
    delete_specs = dict(explorer._context_menu_action_specs(initial))
    assert delete_specs["Delete"] is False
    explorer.tree.setCurrentItem(initial)
    assert not explorer.delete_button.isEnabled()
    explorer.delete_selected()
    assert next(entry for entry in group.fits if entry.kind == "initial") is not None

    output = _item_with_role(explorer, "analysis_output")
    actions = explorer.context_menu_action_names(output)
    assert "Copy" not in actions
    assert "Delete" not in actions


def test_blank_project_context_can_paste_a_copied_workspace(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Workspace", datasets=[DatasetEntry("scan", None)])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._clipboard = project_clipboard.make_payload("group", [group])
    explorer.project.data_groups.clear()
    explorer._refresh_tree()

    assert dict(explorer._context_menu_action_specs(None))["Paste"] is True
    explorer.paste_into_selection()
    assert [item.name for item in explorer.project.data_groups] == ["Workspace"]


@pytest.mark.parametrize(
    "role",
    [
        "datasets",
        "masks",
        "backgrounds",
        "group_masks",
        "group_backgrounds",
        "analyses",
        "plots",
    ],
)
def test_structural_folder_delete_clears_only_its_owned_contents(monkeypatch, role):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    direct = DatasetEntry("direct", None)
    child = DatasetEntry("child", None)
    nested = DatasetGroup("nested", datasets=[child])
    group = DataGroup("Workspace", datasets=[direct], subgroups=[nested])
    group.plots.append(PlotEntry("plot"))
    analysis = AnalysisEntry("analysis", "identity", [direct.id], {})
    group.analyses.append(analysis)
    derived = DatasetEntry(
        "derived", None, metadata={"derived_from_analysis": {"analysis_id": analysis.id}}
    )
    group.datasets.append(derived)
    if role == "masks":
        create_mask(direct, "dataset mask")
    elif role == "backgrounds":
        direct.backgrounds.append(BackgroundSpec("dataset background", ""))
    elif role == "group_masks":
        create_mask(child, "source")
        nested.masks.append(create_mask(DatasetEntry("template", None), "group mask"))
    elif role == "group_backgrounds":
        nested.backgrounds.append(BackgroundSpec("group background", ""))

    explorer = NfitProjectExplorer(NfitProject([group]))
    item = _item_with_role(explorer, role)
    explorer.tree.setCurrentItem(item)
    explorer.delete_selected()

    if role == "datasets":
        assert group.datasets == [] and group.subgroups == []
        assert group.analyses == [analysis] and group.plots[0].name == "plot"
    elif role == "masks":
        assert direct.masks == []
        assert group.datasets[0] is direct and nested.datasets[0] is child
    elif role == "backgrounds":
        assert direct.backgrounds == []
        assert group.datasets[0] is direct and nested.datasets[0] is child
    elif role == "group_masks":
        assert nested.masks == []
        assert direct in group.datasets and child in nested.datasets
    elif role == "group_backgrounds":
        assert nested.backgrounds == []
        assert direct in group.datasets and child in nested.datasets
    elif role == "analyses":
        assert group.analyses == []
        assert derived not in group.datasets
        assert group.plots[0].name == "plot"
    elif role == "plots":
        assert group.plots == []
        assert group.analyses == [analysis] and direct in group.datasets
