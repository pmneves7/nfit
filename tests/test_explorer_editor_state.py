"""Edits retain the visible page and focused field across detail rebuilds."""

import pytest

from nfit import project_gui
from nfit.dataset import PointListData
from nfit.pipeline import DataGroup, DatasetEntry, DatasetGroup
from nfit.project_gui import NfitProjectExplorer
from nfit.project_io import NfitProject
from tests.project_gui_test_support import _tiny_mdhisto_data


def _flush(QtCore, QtWidgets):
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(3):
        QtWidgets.QApplication.processEvents()


@pytest.fixture
def editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    groups = [DatasetGroup(f"collection {i}", datasets=[
        DatasetEntry(f"scan {i}", _tiny_mdhisto_data(1.0))
    ]) for i in range(2)]
    workspace = DataGroup("workspace", subgroups=groups)
    explorer = NfitProjectExplorer(NfitProject([workspace]))
    explorer.window.resize(900, 500)
    explorer.window.show()
    explorer._refresh_tree(select_dataset_group=groups[0], refresh_viewers=False)
    explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "group_composite_enabled"
    ).setChecked(True)
    explorer.details_scroll.setFixedHeight(220)
    explorer.details_widget.setMinimumHeight(1800)
    _flush(QtCore, QtWidgets)
    yield explorer, groups, QtCore, QtWidgets
    explorer.has_unsaved_changes = False
    explorer.window.close()
    _flush(QtCore, QtWidgets)


def test_collection_details_refresh_keeps_field_cursor_and_scroll(editor):
    explorer, _groups, QtCore, QtWidgets = editor
    field_name = "group_composite_minimum_coverage"
    field = explorer.details_widget.findChild(QtWidgets.QLineEdit, field_name)
    field.setText("0.125")
    field.setFocus()
    assert explorer.window.focusWidget() is field
    field.setSelection(0, 1)
    scrollbar = explorer.details_scroll.verticalScrollBar()
    scrollbar.setValue(450)
    position = scrollbar.value()
    assert position > 0

    explorer._sync_details()
    _flush(QtCore, QtWidgets)

    replacement = explorer.details_widget.findChild(QtWidgets.QLineEdit, field_name)
    assert replacement is not field
    assert explorer.window.focusWidget() is replacement
    # Scientific values come from the model, never from the presentation snapshot.
    assert replacement.text() != "0.125"
    assert replacement.selectionStart() == 0
    assert replacement.selectedText() == replacement.text()[:1]
    assert scrollbar.value() == position


def test_collection_refresh_preserves_tab_and_later_focus_change(editor):
    explorer, _groups, QtCore, QtWidgets = editor
    tabs = explorer.details_widget.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
    title = "Metadata dimensions"
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == title))
    tabs.setFocus()
    explorer.details_scroll.verticalScrollBar().setValue(450)
    explorer._sync_details()
    explorer.tree.setFocus()
    explorer.details_scroll.verticalScrollBar().setValue(120)
    _flush(QtCore, QtWidgets)
    tabs = explorer.details_widget.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
    assert tabs.tabText(tabs.currentIndex()) == title
    assert explorer.window.focusWidget() is explorer.tree
    assert explorer.details_scroll.verticalScrollBar().value() == 120


def test_pending_detail_restore_does_not_change_new_selection(editor):
    explorer, groups, QtCore, QtWidgets = editor
    explorer.details_scroll.verticalScrollBar().setValue(450)
    explorer._sync_details()
    explorer._refresh_tree(select_dataset_group=groups[1], refresh_viewers=False)
    explorer.details_scroll.verticalScrollBar().setValue(90)
    explorer.tree.setFocus()
    _flush(QtCore, QtWidgets)
    assert explorer._dataset_group_for_item(explorer.tree.currentItem()) is groups[1]
    assert explorer.details_scroll.verticalScrollBar().value() == 90
    assert explorer.window.focusWidget() is explorer.tree


@pytest.mark.parametrize("key", ["step_size", "num_bins", "lower", "upper"])
def test_unchanged_rebin_field_does_not_rebuild_or_invalidate(editor, monkeypatch, key):
    explorer, groups, _QtCore, _QtWidgets = editor
    workspace = explorer.project.data_groups[0]
    scope = project_gui._composite_scope(workspace, groups[0])
    dataset = groups[0].datasets[0]
    configs = [project_gui.dataset_rebin_config(dataset),
               project_gui.data_group_composite_config(scope)]
    for config in configs:
        config["axes"][0][f"auto_{key}"] = False
    monkeypatch.setattr(explorer, "_after_dataset_rebin_changed",
                        lambda *_: pytest.fail("unchanged dataset field invalidated data"))
    monkeypatch.setattr(explorer, "_after_group_composite_changed",
                        lambda *_: pytest.fail("unchanged collection field invalidated data"))
    explorer._set_dataset_rebin_axis_value(dataset, workspace, 0, key, str(configs[0]["axes"][0][key]))
    explorer._set_group_composite_axis_value(scope, 0, key, str(configs[1]["axes"][0][key]))


@pytest.mark.parametrize("kind", ["scale", "wavelength"])
def test_point_list_numeric_edits_apply_once_on_commit(editor, monkeypatch, kind):
    explorer, _groups, _QtCore, QtWidgets = editor
    dataset = DatasetEntry("points", PointListData({"angle": [1.0], "signal": [2.0]}))
    changes = []
    monkeypatch.setattr(explorer, f"_set_point_list_{kind}", lambda *args: changes.append(args))
    config = {"scale": {}, "wavelength": {}, "channels": []}
    if kind == "scale":
        panel = explorer._point_list_scale_box(dataset, None, config, ["angle", "signal"])
        name = "point_list_scale_factor"
    else:
        panel = explorer._point_list_wavelength_box(dataset, None, config)
        name = "point_list_wavelength"
    spin = panel.findChild(QtWidgets.QDoubleSpinBox, name)
    spin.setValue(2.0)
    spin.setValue(3.0)
    assert changes == []
    spin.editingFinished.emit()
    assert len(changes) == 1
    assert changes[0][-1] == 3.0
    assert "Press Enter" in spin.toolTip()
    panel.deleteLater()
