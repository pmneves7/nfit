from __future__ import annotations

import pytest

import nfit.project_data as project_data
import nfit.project_gui as project_gui
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from nfit.project_gui import NfitProject, NfitProjectExplorer
from tests.project_gui_test_support import _tiny_mdhisto_data


def test_linked_dataset_defaults_to_owner_notice_and_keeps_private_editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("run", _tiny_mdhisto_data(1.0), kind="mdhisto")
    collection = DatasetGroup("sample runs", datasets=[dataset])
    root = DataGroup("workspace", subgroups=[collection])
    project_gui.data_group_composite_config(
        project_gui._composite_scope(root, collection)
    )["enabled"] = True
    explorer = NfitProjectExplorer(NfitProject([root]))
    dataset_item = next(
        item
        for item in explorer._tree_items_by_state_key().values()
        if explorer._objects_for_item(item)[1] is dataset
        and explorer._objects_for_item(item)[4] == "dataset"
    )
    explorer.tree.setCurrentItem(dataset_item)

    notice = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_rebin_ownership"
    )
    message = explorer.details_widget.findChild(
        QtWidgets.QLabel, "dataset_rebin_ownership_message"
    )
    editor = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_rebin_editor"
    )
    edit_separately = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_edit_independent"
    )
    assert notice is not None and notice.toolTip()
    assert "enabled composite" in message.text()
    assert editor is None
    assert edit_separately.toolTip()

    edit_separately.click()
    editor = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_rebin_editor"
    )
    assert editor is not None

    navigate = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_navigate_owner"
    )
    navigate.click()
    assert explorer._dataset_group_for_item(explorer.tree.currentItem()) is collection


def test_native_lazy_event_member_shows_owner_without_private_editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry(
        "lazy event run",
        None,
        kind="mdevent",
        metadata={"source_file": "/not/read/while/selecting.nxs"},
    )
    collection = DatasetGroup(
        "event sample", datasets=[dataset], metadata={"mdevent": {"source": "runs"}}
    )
    root = DataGroup("workspace", subgroups=[collection])
    explorer = NfitProjectExplorer(NfitProject([root]))
    monkeypatch.setattr(
        project_gui,
        "_ensure_dataset_data_loaded",
        lambda *_args: pytest.fail("ownership presentation loaded event data"),
    )

    explorer._set_dataset_details(dataset, root)

    message = explorer.details_widget.findChild(
        QtWidgets.QLabel, "dataset_rebin_ownership_message"
    )
    assert "histogrammed on its collection's grid" in message.text()
    assert explorer.details_widget.findChild(
        QtWidgets.QPushButton, "dataset_rebin_edit_independent"
    ) is None
    assert explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_rebin_editor"
    ) is None


def test_organizational_collection_offers_explicit_combined_output(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    collection = DatasetGroup("folder")
    root = DataGroup("workspace", subgroups=[collection])
    explorer = NfitProjectExplorer(NfitProject([root]))

    explorer._set_dataset_collection_details(root, collection)

    notice = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "group_composite_ownership"
    )
    action = explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "group_composite_enabled"
    )
    assert notice is not None and notice.toolTip()
    assert action.text() == "Combine as rebinned output"
    assert action.toolTip()
    assert not action.isEnabled()
    status = explorer.details_widget.findChild(
        QtWidgets.QLabel, "group_composite_ownership_status"
    )
    assert status is not None and status.toolTip()
    assert explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_binning"
    ) is None

    # Top-level workspaces have no persistent id, but use the same compact path.
    explorer._set_dataset_collection_details(root, root)
    assert explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "group_composite_ownership"
    ) is not None


def test_disabled_fit_composite_retains_enabled_visualization_recipe(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    collection = DatasetGroup(
        "sample", datasets=[DatasetEntry("run", _tiny_mdhisto_data(1.0))]
    )
    root = DataGroup("workspace", subgroups=[collection])
    scope = project_gui._composite_scope(root, collection)
    overview_id = project_data.add_data_group_composite_binning(
        scope, name="Overview"
    )
    project_data.data_group_composite_config_by_id(scope, overview_id)[
        "enabled"
    ] = True
    explorer = NfitProjectExplorer(NfitProject([root]))

    explorer._set_dataset_collection_details(root, collection)

    selector = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_composite_binning"
    )
    assert selector is not None and selector.currentText() == "Overview"
    assert selector.toolTip()
    assert explorer.details_widget.findChild(
        QtWidgets.QTabWidget, "group_composite_tabs"
    ) is None
    edit = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "group_composite_edit_visualization"
    )
    assert edit is not None and edit.toolTip()
    edit.click()
    assert explorer.details_widget.findChild(
        QtWidgets.QTabWidget, "group_composite_tabs"
    ) is not None

    explorer._expanded_inactive_composite_ids.discard(collection.id)
    explorer._set_dataset_collection_details(root, collection)
    combine = explorer.details_widget.findChild(
        QtWidgets.QCheckBox, "group_composite_enabled"
    )
    combine.setChecked(True)
    assert project_data.data_group_composite_config(scope)["enabled"] is True
    assert project_data.data_group_composite_config_by_id(scope, overview_id)[
        "enabled"
    ] is True


def test_dataset_and_collection_panels_explain_background_consumers(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    source_dataset = DatasetEntry("dataset background", _tiny_mdhisto_data(1.0))
    consumer_dataset = DatasetEntry("sample run", _tiny_mdhisto_data(2.0))
    consumer_dataset.backgrounds.append(
        BackgroundSpec("dataset link", source_entry=source_dataset)
    )
    source_group = DatasetGroup(
        "background collection",
        datasets=[DatasetEntry("background run", _tiny_mdhisto_data(3.0))],
    )
    sample_group = DatasetGroup(
        "sample collection", datasets=[consumer_dataset]
    )
    sample_group.backgrounds.append(
        BackgroundSpec("collection link", source_group=source_group)
    )
    root = DataGroup(
        "workspace", datasets=[source_dataset], subgroups=[source_group, sample_group]
    )
    explorer = NfitProjectExplorer(NfitProject([root]))

    explorer._set_dataset_details(source_dataset, root)
    dataset_notes = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "dataset_background_source_explanations"
    )
    assert dataset_notes is not None and dataset_notes.toolTip()
    assert "Used by 'sample run'" in " ".join(
        label.text() for label in dataset_notes.findChildren(QtWidgets.QLabel)
    )

    explorer._set_dataset_collection_details(root, source_group)
    group_notes = explorer.details_widget.findChild(
        QtWidgets.QGroupBox, "group_background_source_explanations"
    )
    assert group_notes is not None and group_notes.toolTip()
    assert "Used by 'sample collection'" in " ".join(
        label.text() for label in group_notes.findChildren(QtWidgets.QLabel)
    )
