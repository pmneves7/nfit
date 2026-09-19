"""Binning-owner links select existing rows without running scientific work."""

import pytest

from nfit import DataGroup, DatasetGroup, NfitProject
from nfit.project_gui import NfitProjectExplorer


def test_binning_owner_navigation_selects_nested_and_root_collections(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    child = DatasetGroup("Sample angles")
    folder = DatasetGroup("Organization", subgroups=[child])
    root = DataGroup("Workspace", subgroups=[folder])
    explorer = NfitProjectExplorer(NfitProject([root]))
    shown = []
    monkeypatch.setattr(explorer, "_set_dataset_collection_details", lambda group, node: shown.append(node))
    monkeypatch.setattr(explorer, "_refresh_tree", lambda **kwargs: (_ for _ in ()).throw(AssertionError("tree rebuilt")))
    rows = explorer._tree_items_by_state_key()
    folder_item = rows[("dataset_group", folder.id)]
    folder_item.setExpanded(False)
    assert explorer._navigate_to_binning_owner(root, child)
    assert explorer._dataset_group_for_item(explorer.tree.currentItem()) is child
    assert folder_item.isExpanded()
    assert shown[-1] is child
    assert explorer._navigate_to_binning_owner(root, root)
    assert explorer._objects_for_item(explorer.tree.currentItem())[4] == "datasets"
    assert shown[-1] is root
    assert not explorer._navigate_to_binning_owner(root, DatasetGroup("missing"))
    assert not explorer._navigate_to_binning_owner(DataGroup("other"), child)
    explorer.window.close()


def test_background_binning_note_updates_with_projection_and_enabled_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    from nfit import BackgroundSpec

    source = DatasetGroup("Background")
    sample = DatasetGroup("Sample", metadata={"mdevent": {}})
    background = BackgroundSpec("instrument", source_group_id=source.id, source_group=source)
    sample.backgrounds.append(background)
    root = DataGroup("Workspace", subgroups=[source, sample])
    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(explorer, "_background_changed", lambda *_: None)
    explorer._set_background_details(root, sample, background)
    label = explorer.details_widget.findChild(QtWidgets.QLabel, "background_binning_explanation")
    projection = explorer.details_widget.findChild(QtWidgets.QComboBox, "background_projection")
    assert label is not None and label.toolTip()
    assert "composite rebin recipe" in label.text()
    projection.setCurrentIndex(projection.findData("measured_events"))
    assert "private view recipe is not used" in label.text()
    enabled = next(widget for widget in explorer.details_widget.findChildren(QtWidgets.QCheckBox) if widget.text() == "Enabled")
    enabled.setChecked(False)
    assert "disabled" in label.text()
    enabled.setChecked(True)
    assert "sample grid" in label.text()
    explorer.window.close()
