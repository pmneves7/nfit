"""Run-page drag and drop behavior in the project explorer."""

from __future__ import annotations

import pytest

from tests.project_gui_test_support import DataGroup, DatasetEntry, NfitProject, NfitProjectExplorer


def _pages(explorer, group_index: int = 0):
    datasets_item = explorer.tree.topLevelItem(group_index).child(0)
    return [
        datasets_item.child(index)
        for index in range(datasets_item.childCount())
        if explorer._objects_for_item(datasets_item.child(index))[4] == "dataset_page"
    ]


def _select(explorer, *items):
    explorer.tree.clearSelection()
    explorer.tree.setCurrentItem(items[0])
    for item in items:
        item.setSelected(True)


def _explorer_with_run_pages(*, target_runs: int = 0):
    source = DataGroup(
        "source", datasets=[DatasetEntry(f"source {index}", None) for index in range(151)]
    )
    target = DataGroup(
        "target", datasets=[DatasetEntry(f"target {index}", None) for index in range(target_runs)]
    )
    return NfitProjectExplorer(NfitProject([source, target])), source, target


def test_dragging_a_run_page_creates_the_internal_dataset_drag(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtGui = pytest.importorskip("PySide6.QtGui")

    explorer, _source, _target = _explorer_with_run_pages()
    page = _pages(explorer)[0]
    _select(explorer, page)
    captured = {}

    class FakeDrag:
        def __init__(self, parent):
            captured["parent"] = parent

        def setMimeData(self, mime_data):
            captured["mime_data"] = mime_data

        def exec(self, actions, default_action):
            captured["actions"] = actions
            captured["default_action"] = default_action

    monkeypatch.setattr(QtGui, "QDrag", FakeDrag)
    explorer.tree.startDrag(QtCore.Qt.DropAction.MoveAction)

    assert captured["parent"] is explorer.tree
    assert bytes(captured["mime_data"].data("application/x-nfit-tree-item")) == b"dataset_page"
    assert captured["actions"] == (
        QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction
    )
    assert captured["default_action"] == QtCore.Qt.DropAction.MoveAction


def test_drag_gesture_on_a_run_page_starts_drag_without_extending_selection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtTest = pytest.importorskip("PySide6.QtTest")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    explorer, _source, _target = _explorer_with_run_pages()
    explorer.window.show()
    QtWidgets.QApplication.processEvents()
    page = _pages(explorer)[0]
    _select(explorer, page)
    captured = {}

    class FakeDrag:
        def __init__(self, parent):
            captured["parent"] = parent

        def setMimeData(self, mime_data):
            captured["mime_data"] = mime_data

        def exec(self, actions, default_action):
            captured["actions"] = actions
            captured["default_action"] = default_action

    monkeypatch.setattr(QtGui, "QDrag", FakeDrag)
    viewport = explorer.tree.viewport()
    start = explorer.tree.visualItemRect(page).center()
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    distance = QtWidgets.QApplication.startDragDistance() + 2
    moved = start + QtCore.QPoint(distance, 0)
    move_event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(moved),
        QtCore.QPointF(viewport.mapToGlobal(moved)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(viewport, move_event)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=moved)

    assert captured["parent"] is explorer.tree
    assert bytes(captured["mime_data"].data("application/x-nfit-tree-item")) == b"dataset_page"
    assert explorer.tree.selectedItems() == [page]
    explorer.window.close()


@pytest.mark.parametrize("copy_item", [False, True])
def test_dragging_an_unexpanded_run_page_moves_or_copies_all_its_runs_without_loading_it(
    monkeypatch, copy_item
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    explorer, source, target = _explorer_with_run_pages()
    page = _pages(explorer)[0]
    assert page.child(0).text(0) == "Load runs..."
    _select(explorer, page)
    target_datasets = explorer.tree.topLevelItem(1).child(0)

    assert explorer.move_or_copy_selected_to_item(target_datasets, copy_item=copy_item)
    assert [dataset.name for dataset in target.datasets] == [
        f"source {index}" for index in range(50)
    ]
    expected_source = range(151) if copy_item else range(50, 151)
    assert [dataset.name for dataset in source.datasets] == [
        f"source {index}" for index in expected_source
    ]
    # The rebuilt lazy pages are still placeholders.  Dragging their payload
    # must not expand 50 dataset rows just to carry out the operation.
    assert all(
        not source_page.isExpanded() and source_page.child(0).text(0) == "Load runs..."
        for source_page in _pages(explorer, 0)
    )


def test_dragging_a_page_with_one_of_its_loaded_runs_deduplicates_the_move(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    explorer, source, target = _explorer_with_run_pages()
    page = _pages(explorer)[0]
    page.setExpanded(True)
    loaded_run = page.child(0)
    _select(explorer, page, loaded_run)
    target_datasets = explorer.tree.topLevelItem(1).child(0)

    assert explorer.move_or_copy_selected_to_item(target_datasets, copy_item=False)
    assert [dataset.name for dataset in target.datasets] == [
        f"source {index}" for index in range(50)
    ]
    assert [dataset.name for dataset in source.datasets] == [
        f"source {index}" for index in range(50, 151)
    ]


@pytest.mark.parametrize("drop_position, expected_insert_index", [("above", 50), ("below", 100)])
def test_dropping_on_a_run_page_uses_that_pages_dataset_collection_and_position(
    monkeypatch, drop_position, expected_insert_index
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    explorer, source, target = _explorer_with_run_pages(target_runs=151)
    source_page = _pages(explorer, 0)[0]
    target_page = _pages(explorer, 1)[1]
    _select(explorer, source_page)
    position = (
        QtWidgets.QAbstractItemView.DropIndicatorPosition.AboveItem
        if drop_position == "above"
        else QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem
    )

    assert explorer.move_or_copy_selected_to_item(
        target_page, copy_item=False, drop_position=position
    )
    assert [dataset.name for dataset in source.datasets] == [
        f"source {index}" for index in range(50, 151)
    ]
    assert [dataset.name for dataset in target.datasets] == [
        *(f"target {index}" for index in range(expected_insert_index)),
        *(f"source {index}" for index in range(50)),
        *(f"target {index}" for index in range(expected_insert_index, 151)),
    ]
