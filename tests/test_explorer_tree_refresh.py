"""Regression coverage for rebuilding the project-explorer tree."""

import pytest

import nfit.project_gui as project_gui
from nfit.analysis.core import AnalysisEntry, AnalysisOutputRef, AnalysisResultRecord
from nfit.pipeline import (
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    ModelComponentSpec,
)
from nfit.project_data import create_derived_analysis_dataset
from nfit.project_gui import NfitProjectExplorer, delete_dataset_group
from nfit.project_io import NfitProject
from tests.project_gui_test_support import _tiny_mdhisto_data


def _tree_item_for_role(explorer, role, *, node=None):
    """Return the rendered item for a role, optionally for one collection."""

    def visit(item):
        actual_role = explorer._objects_for_item(item)[4]
        if actual_role == role and (
            node is None or explorer._dataset_group_for_item(item) is node
        ):
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
    raise AssertionError(f"No tree item with role {role!r}")


def _tree_item_for_dataset(explorer, dataset):
    """Return the rendered row for one dataset object."""

    def visit(item):
        _group, candidate, _mask, _model, role = explorer._objects_for_item(item)
        if role == "dataset" and candidate is dataset:
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
    raise AssertionError(f"No rendered item for {dataset.name!r}")


def test_tree_refresh_keeps_sections_and_expanded_lazy_page_after_collection_delete(
    monkeypatch,
):
    """Deleting a sibling collection must not hide the workspace's other sections."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    runs = DatasetGroup(
        "Runs",
        datasets=[DatasetEntry(f"run {index}", None) for index in range(51)],
    )

    removed = DatasetGroup("Remove me", datasets=[DatasetEntry("discard", None)])
    group = DataGroup(
        "Workspace1",
        subgroups=[runs, removed],
        models={"constant": ModelComponentSpec("constant")},
        fits=[
            FitTimelineEntry(
                "Fit timeline",
                kind="timeline",
                children=[FitTimelineEntry("Result")],
            )
        ],
    )
    analysis = AnalysisEntry("Summary", "bragg_integration", [], {})
    analysis.result = AnalysisResultRecord(
        "recipe",
        {},
        [AnalysisOutputRef("count", "Count", "scalar", scalar_value=1.0)],
        "success",
        "now",
    )
    group.analyses.append(analysis)
    explorer = NfitProjectExplorer(NfitProject([group]))

    group_item = _tree_item_for_role(explorer, "group")
    datasets_item = _tree_item_for_role(explorer, "datasets")
    models_item = _tree_item_for_role(explorer, "models")
    fits_item = _tree_item_for_role(explorer, "fits")
    analyses_item = _tree_item_for_role(explorer, "analyses")
    analysis_item = _tree_item_for_role(explorer, "analysis")
    runs_item = _tree_item_for_role(explorer, "dataset_group", node=runs)
    page_item = _tree_item_for_role(explorer, "dataset_page")

    for item in (
        group_item,
        datasets_item,
        models_item,
        fits_item,
        analyses_item,
        analysis_item,
        runs_item,
    ):
        item.setExpanded(True)
    page_item.setExpanded(True)
    QtWidgets.QApplication.processEvents()
    assert page_item.childCount() == 50

    assert delete_dataset_group(group, removed)
    explorer._refresh_tree(select_group=group, refresh_viewers=False)

    rebuilt_group = _tree_item_for_role(explorer, "group")
    rebuilt_runs = _tree_item_for_role(explorer, "dataset_group", node=runs)
    rebuilt_page = _tree_item_for_role(explorer, "dataset_page")
    assert rebuilt_group.isExpanded()
    assert _tree_item_for_role(explorer, "datasets").isExpanded()
    assert _tree_item_for_role(explorer, "models").isExpanded()
    assert _tree_item_for_role(explorer, "fits").isExpanded()
    assert _tree_item_for_role(explorer, "analyses").isExpanded()
    assert _tree_item_for_role(explorer, "analysis").isExpanded()
    assert rebuilt_runs.isExpanded()
    assert rebuilt_page.isExpanded()
    assert rebuilt_page.childCount() == 50
    assert not any(
        explorer._dataset_group_for_item(item) is removed
        for item in (rebuilt_runs,)
    )


def test_tree_refresh_without_target_restores_multiselection_current_row_and_viewport(
    monkeypatch,
):
    """An incidental refresh keeps the user's place in a populated lazy page."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    datasets = [DatasetEntry(f"run {index}", None) for index in range(51)]
    group = DataGroup("Workspace1", datasets=datasets)
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.window.show()
    page = _tree_item_for_role(explorer, "dataset_page")
    page.setExpanded(True)
    QtWidgets.QApplication.processEvents()

    first = _tree_item_for_dataset(explorer, datasets[42])
    second = _tree_item_for_dataset(explorer, datasets[43])
    explorer.tree.setCurrentItem(second)
    first.setSelected(True)
    second.setSelected(True)
    explorer.tree.setVerticalScrollMode(
        QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
    )
    explorer.tree.scrollToItem(second, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
    explorer.tree.verticalScrollBar().setValue(
        min(
            explorer.tree.verticalScrollBar().value() + 5,
            explorer.tree.verticalScrollBar().maximum(),
        )
    )
    QtWidgets.QApplication.processEvents()
    prior_anchor = explorer._tree_item_state_key(explorer.tree.itemAt(0, 0))
    prior_anchor_offset = explorer.tree.visualItemRect(explorer.tree.itemAt(0, 0)).top()
    assert explorer.tree.verticalScrollBar().value() > 0

    explorer._refresh_tree(refresh_viewers=False)
    QtWidgets.QApplication.processEvents()

    restored_first = _tree_item_for_dataset(explorer, datasets[42])
    restored_second = _tree_item_for_dataset(explorer, datasets[43])
    assert explorer.tree.currentItem() is restored_second
    assert set(explorer.tree.selectedItems()) == {restored_first, restored_second}
    assert explorer._tree_item_state_key(explorer.tree.itemAt(0, 0)) == prior_anchor
    assert explorer.tree.visualItemRect(explorer.tree.itemAt(0, 0)).top() == prior_anchor_offset
    assert explorer.tree.verticalScrollBar().value() > 0

    explorer._refresh_tree(
        select_group=group,
        select_dataset=datasets[43],
        refresh_viewers=False,
    )
    QtWidgets.QApplication.processEvents()

    assert explorer.tree.currentItem() is _tree_item_for_dataset(explorer, datasets[43])
    assert explorer._tree_item_state_key(explorer.tree.itemAt(0, 0)) == prior_anchor
    assert explorer.tree.verticalScrollBar().value() > 0
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_tree_refresh_survives_a_stale_derived_cache_dependency(monkeypatch):
    """A cache-signature lookup cannot prevent later workspace sections rendering."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    dataset = DatasetEntry("derived", None)
    dataset.parameters[project_gui.DATASET_REBIN_KEY] = {"enabled": True}
    group = DataGroup(
        "Workspace1",
        datasets=[dataset],
        models={"constant": ModelComponentSpec("constant")},
        fits=[FitTimelineEntry("Fit timeline", kind="timeline")],
    )
    monkeypatch.setattr(
        project_gui,
        "_project_binning_is_current",
        lambda *_args: (_ for _ in ()).throw(
            KeyError("derived dataset refers to missing source")
        ),
    )

    explorer = NfitProjectExplorer(NfitProject([group]))

    assert _tree_item_for_role(explorer, "datasets") is not None
    assert _tree_item_for_role(explorer, "models") is not None
    assert _tree_item_for_role(explorer, "fits") is not None
    assert _tree_item_for_role(explorer, "analyses") is not None
    assert _tree_item_for_role(explorer, "plots") is not None


def test_tree_refresh_survives_deleted_source_of_live_derived_composite(monkeypatch):
    """Deleting a composite source leaves the rest of its workspace explorable."""

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    source = DatasetGroup("Low temperature", [DatasetEntry("scan", _tiny_mdhisto_data(1.0))])
    group = DataGroup(
        "Workspace1",
        subgroups=[source],
        models={"constant": ModelComponentSpec("constant")},
        fits=[FitTimelineEntry("Fit timeline", kind="timeline")],
    )
    analysis = AnalysisEntry(
        "Live composite", "dataset_clone", [f"group-composite:{source.id}"], {}
    )
    create_derived_analysis_dataset(group, analysis)
    explorer = NfitProjectExplorer(NfitProject([group]))

    assert delete_dataset_group(group, source)
    explorer._refresh_tree(select_group=group, refresh_viewers=False)

    assert _tree_item_for_role(explorer, "datasets") is not None
    assert _tree_item_for_role(explorer, "models") is not None
    assert _tree_item_for_role(explorer, "fits") is not None
    assert _tree_item_for_role(explorer, "analyses") is not None
    assert _tree_item_for_role(explorer, "plots") is not None
