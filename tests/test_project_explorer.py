# ruff: noqa: F401, F403, F405
import copy

from nfit.app_distribution import application_version
from nfit.project_archive import read_project_manifest
from tests.project_gui_test_support import *
from tests.project_gui_test_support import (
    _explorer_with_fit_result,
    _grid_mdhisto_data,
    _he_mdhisto_data,
    _points_for_dynamic_writeback,
    _rpa_overlay_group,
    _standard_shortcut_text,
    _tiny_mdhisto_data,
    _tree_items_with_children,
)


def test_standard_shortcut_text_accepts_platforms_without_a_binding():
    class KeySequenceWithoutBindings:
        class SequenceFormat:
            PortableText = object()

        @staticmethod
        def keyBindings(_standard_key):
            return []

    class QtGuiWithoutBindings:
        QKeySequence = KeySequenceWithoutBindings

    assert _standard_shortcut_text(QtGuiWithoutBindings, object()) == ""


def test_project_explorer_opens_at_screen_aware_initial_size(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    explorer = NfitProjectExplorer()
    explorer.show()
    QtWidgets.QApplication.processEvents()

    screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
    if screen is None:
        screen = explorer.window.screen() or explorer.app.primaryScreen()
    available = screen.availableGeometry()
    expected = project_gui._screen_aware_project_window_size(
        available.width(),
        available.height(),
    )
    assert explorer.window.size().toTuple() == expected
    explorer.window.close()


def test_project_window_size_prefers_roomy_layout_but_fits_small_screens():
    assert project_gui._screen_aware_project_window_size(1920, 1080) == (
        1560,
        1000,
    )
    assert project_gui._screen_aware_project_window_size(1280, 720) == (
        1232,
        672,
    )


def test_project_helpers_name_import_and_round_trip(tmp_path):
    assert nfit.create_data_group is create_data_group
    assert nfit.import_dataset_paths is import_dataset_paths
    assert nfit.edit_project_file is project_gui.edit_project_file
    assert nfit.project_state_issues is project_gui.project_state_issues
    assert nfit.validate_project_state is project_gui.validate_project_state

    project = NfitProject(
        data_groups=[
            DataGroup("Workspace1"),
            DataGroup("Workspace3"),
        ]
    )

    assert next_data_group_name(project.data_groups) == "Workspace2"

    group = create_data_group(project, "scan_group")
    first, second = import_dataset_paths(
        group,
        [tmp_path / "scan.nxs", tmp_path / "scan.nxs"],
    )
    assert first.id != second.id
    first.enabled = False
    first.fit_weight = 2.5
    mask = create_mask(first)
    mask.type = "box"
    mask.parameters["center"] = [0.0, 0.0, 0.0, 10.0]
    mask.invert = True
    mask.additive = True
    copy_mask_to_dataset(mask, second)
    copied_group = create_data_group(project, "copied_group")
    copied = copy_dataset_to_group(first, copied_group)
    assert copied.id != first.id
    model = create_model_component(group)
    model.parameters["constant"] = 0.25
    model.config["script_note"] = "fixed"

    assert group.dataset_names == ["scan", "scan1"]
    assert first.kind == "nxs"
    assert first.metadata["source_file"].endswith("scan.nxs")
    assert first.metadata["import_status"] == "pending"

    project_path = tmp_path / "project.nfit"
    save_project(project, project_path)

    payload = read_project_manifest(project_path)
    assert payload["format"] == "nfit-project"
    assert payload["data_groups"][2]["datasets"][0]["name"] == "scan"
    assert payload["data_groups"][2]["datasets"][0]["masks"][0]["type"] == "box"
    assert payload["data_groups"][2]["datasets"][0]["masks"][0]["parameters"]["center"] == [
        0.0,
        0.0,
        0.0,
        10.0,
    ]
    assert payload["data_groups"][2]["models"][0]["type"] == "constant_background"
    assert payload["data_groups"][2]["models"][0]["parameters"]["constant"] == 0.25
    assert payload["data_groups"][2]["models"][0]["config"]["script_note"] == "fixed"
    assert payload["data_groups"][2]["models"][0]["fit_parameters"]["constant"] is False

    loaded = load_project(project_path)

    assert [group.name for group in loaded.data_groups] == [
        "Workspace1",
        "Workspace3",
        "scan_group",
        "copied_group",
    ]
    assert loaded.data_groups[2].dataset_names == ["scan", "scan1"]
    assert loaded.data_groups[2].datasets[0].enabled is False
    assert loaded.data_groups[2].datasets[0].fit_weight == 2.5
    assert loaded.data_groups[2].datasets[0].masks[0].type == "box"
    assert loaded.data_groups[2].datasets[0].masks[0].invert is True
    assert loaded.data_groups[2].datasets[0].masks[0].additive is True
    assert loaded.data_groups[2].datasets[1].masks[0].name == "Mask1"
    assert loaded.data_groups[2].models["Model1"].parameters["constant"] == 0.25
    assert loaded.data_groups[2].models["Model1"].config["script_note"] == "fixed"
    assert loaded.data_groups[2].models["Model1"].fit_parameters["constant"] is False
    assert loaded.data_groups[2].models["Model1"].sharing["constant"]["mode"] == "global"
    assert loaded.data_groups[3].datasets[0].name == "scan"


def test_project_save_rejects_replacement_arrays_not_stored_in_source(tmp_path):
    source = tmp_path / "scan.npz"
    source.touch()
    dataset = DatasetEntry(
        "scan",
        _grid_mdhisto_data(),
        kind="mdhisto",
        metadata={"source_file": str(source)},
    )
    editable = dataset.data.mutable_copy()
    editable.signal.flat[0] += 1.0
    dataset.replace_data(editable)

    with pytest.raises(TypeError, match="portable \\.npz"):
        save_project(
            NfitProject([DataGroup("Workspace1", datasets=[dataset])]),
            tmp_path / "project.nfit",
        )


def test_obsolete_bragg_edge_policy_is_rejected():
    with pytest.raises(ValueError, match="edge_policy.*no longer supported"):
        project_gui._analysis_from_dict(
            {
                "name": "Peaks",
                "type": "bragg_integration",
                "input_dataset_ids": ["data"],
                "parameters": {"edge_policy": "report_partial"},
            }
        )


def test_waterfall_group_keys_follow_immediate_dataset_groups():
    direct = DatasetEntry("direct", _grid_mdhisto_data(), kind="mdhisto")
    nested = DatasetEntry("nested", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup(
        "Workspace1",
        datasets=[direct],
        subgroups=[DatasetGroup("Group1", datasets=[nested])],
    )

    assert project_gui._waterfall_group_keys(
        group,
        ["direct", "nested"],
    ) == ["root", "Group1"]


@pytest.mark.parametrize("built", [True, False])
def test_project_explorer_help_opens_local_documentation(monkeypatch, tmp_path, built):
    import nfit.project_gui as project_gui

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    opened = []
    messages = []
    monkeypatch.setattr(project_gui, "__file__", str(tmp_path / "src" / "nfit" / "project_gui.py"))
    index = tmp_path / "docs" / "_build" / "html" / "index.html"
    if built:
        index.parent.mkdir(parents=True)
        index.write_text("<html>Help</html>")
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    explorer = NfitProjectExplorer(NfitProject([]))
    toolbar = explorer.window.findChild(QtWidgets.QToolBar)
    buttons = [toolbar.widgetForAction(action) for action in toolbar.actions()]
    assert [button.text() for button in buttons[:2]] == ["File", "Help"]
    help_button = toolbar.findChild(QtWidgets.QToolButton, "help_button")
    assert help_button.toolTip()
    help_button.click()
    assert opened == ([index.as_uri()] if built else [])
    assert len(messages) == (0 if built else 1)
    if not built:
        assert "python -m sphinx" in messages[0]
    explorer.window.close()


def test_project_explorer_preserves_tree_expansion_and_toolbar_font(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group_a = DataGroup("Datagroup1")
    group_b = DataGroup("Datagroup2")
    explorer = NfitProjectExplorer(NfitProject([group_a, group_b]))
    tree = explorer.tree
    toolbar = explorer.window.findChild(QtWidgets.QToolBar)
    file_button = toolbar.findChild(QtWidgets.QToolButton, "file_menu_button")

    first_group_item = tree.topLevelItem(0)
    second_group_item = tree.topLevelItem(1)
    second_datasets_item = second_group_item.child(0)
    first_group_item.setExpanded(False)
    second_group_item.setExpanded(True)
    second_datasets_item.setExpanded(True)

    explorer.import_dataset_paths(group_b, [tmp_path / "scan.nxs"])

    assert tree.topLevelItem(0).isExpanded() is False
    assert tree.topLevelItem(1).isExpanded() is True
    assert tree.topLevelItem(1).child(0).isExpanded() is True
    assert toolbar.font().pointSize() == tree.font().pointSize() + 1
    assert file_button.font().pointSize() == tree.font().pointSize() + 1

    explorer.expand_all()
    assert all(item.isExpanded() for item in _tree_items_with_children(tree))

    explorer.collapse_all()
    assert all(not item.isExpanded() for item in _tree_items_with_children(tree))

    sequence_format = QtGui.QKeySequence.SequenceFormat.PortableText
    shortcuts = {
        action.text(): action.shortcut().toString(sequence_format)
        for action in file_button.menu().actions()
        if not action.isSeparator() and action.menu() is None
    }
    assert shortcuts == {
        "New": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.New),
        "Open": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.Open),
        "Reload from Disk": _standard_shortcut_text(
            QtGui, QtGui.QKeySequence.StandardKey.Refresh
        ),
        "Cache binnings": "",
        "Save": _standard_shortcut_text(
            QtGui, QtGui.QKeySequence.StandardKey.Save
        ),
        "Save As": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.SaveAs),
        "Preferences…": "",
        f"nfit version {application_version()}": "",
        "Close": _standard_shortcut_text(
            QtGui, QtGui.QKeySequence.StandardKey.Close
        ),
        "Quit": _standard_shortcut_text(
            QtGui, QtGui.QKeySequence.StandardKey.Quit
        ),
        "Check for updates…": "",
        "Update settings…": "",
    }


def test_import_and_reenable_advance_to_evaluated_current_state(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.import_dataset_paths(
        group,
        [tmp_path / "scan.nxs"],
        data_type="single_crystal_inelastic",
    )

    current = group.fits[-1]
    assert current.kind == "current"
    assert (
        current.metadata["model_evaluation_status"]
        == "deferred until data are viewed or fitted"
    )

    dataset = group.datasets[0]
    dataset.replace_data(_grid_mdhisto_data())
    dataset.enabled = False
    explorer._refresh_tree(select_group=group, select_dataset=dataset)
    explorer.enabled_check.setChecked(True)

    assert dataset.enabled
    assert group.fits[-1].kind == "current"
    assert "model_evaluation_status" in group.fits[-1].metadata


def test_import_does_not_evaluate_models_that_require_lazy_data(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    evaluated = []
    monkeypatch.setattr(
        project_gui,
        "evaluate_current_state_model",
        lambda *_args, **_kwargs: evaluated.append(True),
    )

    explorer.import_dataset_paths(
        group,
        [tmp_path / "scan.nxs"],
        data_type="single_crystal_inelastic",
    )

    assert not evaluated
    assert group.fits[-1].channels == {}
    assert (
        group.fits[-1].metadata["model_evaluation_status"]
        == "deferred until data are viewed or fitted"
    )


def test_project_explorer_tree_hierarchy_fonts(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    create_mask(dataset, "Mask1")
    dataset.backgrounds.append(project_gui.BackgroundSpec("Background", "source-id"))
    model = create_model_component(DataGroup("unused"))
    group = DataGroup(
        "Workspace1",
        datasets=[dataset],
        models={model.name: model},
        fits=[
            FitTimelineEntry("Initial", kind="initial"),
            FitTimelineEntry("Fit Result1", kind="result"),
            FitTimelineEntry("Current state", kind="current"),
        ],
    )
    explorer = NfitProjectExplorer(NfitProject([group]))

    workspace_item = explorer.tree.topLevelItem(0)
    datasets_item = workspace_item.child(0)
    models_item = workspace_item.child(1)
    fits_item = workspace_item.child(2)
    dataset_item = datasets_item.child(0)
    masks_item = dataset_item.child(0)
    backgrounds_item = dataset_item.child(1)
    mask_item = masks_item.child(0)
    model_item = models_item.child(0)
    initial_item = fits_item.child(0)
    result_item = fits_item.child(1)
    current_item = fits_item.child(2)

    # Initial project opening exposes workspace contents without expanding
    # individual datasets, masks, or nested dataset groups.
    assert workspace_item.isExpanded()
    assert datasets_item.isExpanded()
    assert models_item.isExpanded()
    assert fits_item.isExpanded()
    assert not dataset_item.isExpanded()
    assert not masks_item.isExpanded()
    assert not backgrounds_item.isExpanded()

    assert workspace_item.font(0).bold()
    assert workspace_item.font(0).underline()
    assert datasets_item.font(0).bold()
    assert models_item.font(0).bold()
    assert fits_item.font(0).bold()
    for item in (
        dataset_item,
        masks_item,
        backgrounds_item,
        mask_item,
        model_item,
        initial_item,
        result_item,
        current_item,
    ):
        assert not item.font(0).bold()
        assert not item.font(0).underline()
    for item in (
        workspace_item,
        datasets_item,
        models_item,
        fits_item,
        dataset_item,
        masks_item,
        backgrounds_item,
        mask_item,
        model_item,
        initial_item,
        result_item,
        current_item,
    ):
        assert not item.icon(0).isNull()
    assert project_gui._TREE_ICON_COLORS["background_folder"] not in {
        project_gui._TREE_ICON_COLORS[kind]
        for kind in ("folder", "mask_folder", "model_folder", "fit_folder")
    }
    specialized_folders = (
        "background_folder",
        "mask_folder",
        "model_folder",
        "fit_folder",
        "analysis_folder",
        "plot_folder",
    )
    assert len(
        {project_gui._TREE_ICON_COLORS[kind] for kind in specialized_folders}
    ) == len(specialized_folders)


def test_large_dataset_groups_use_lazy_tree_pages_and_paginated_details(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    datasets = [DatasetEntry(f"run {500000 + index}", None) for index in range(125)]
    runs = DatasetGroup("Rotation runs", datasets=datasets)
    parent = DatasetGroup("Temperature", subgroups=[runs])
    group = DataGroup("Workspace1", subgroups=[parent])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_dataset_group=runs, refresh_viewers=False)

    runs_item = explorer.tree.currentItem()
    assert runs_item.text(0) == "Rotation runs"
    assert not runs_item.isExpanded()
    pages = [
        runs_item.child(index)
        for index in range(runs_item.childCount())
        if explorer._objects_for_item(runs_item.child(index))[4] == "dataset_page"
    ]
    assert len(pages) == 3
    assert all(not page.isExpanded() and page.childCount() == 1 for page in pages)
    assert pages[0].child(0).text(0) == "Load runs..."
    assert pages[0].toolTip(0)

    table = explorer.details_widget.findChild(
        QtWidgets.QTableWidget, "group_datasets_table"
    )
    page_combo = explorer.details_widget.findChild(
        QtWidgets.QComboBox, "group_datasets_page"
    )
    assert table is not None and table.toolTip()
    assert table.rowCount() == 20
    assert page_combo is not None and page_combo.toolTip()
    assert page_combo.count() == 7

    pages[0].setExpanded(True)
    QtWidgets.QApplication.processEvents()
    assert pages[0].childCount() == 50
    assert explorer._objects_for_item(pages[0].child(0))[4] == "dataset"
    assert pages[0].child(0).text(0) == "run 500000"
    assert not pages[0].child(0).isExpanded()

    explorer.expand_all()
    assert runs_item.isExpanded()
    assert all(not page.isExpanded() for page in pages)
    assert not pages[0].child(0).isExpanded()


def test_parent_dataset_details_show_immediate_children_not_flattened_runs(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetGroup(
        "34",
        datasets=[DatasetEntry(f"run {index}", None) for index in range(30)],
    )
    second = DatasetGroup(
        "70",
        datasets=[DatasetEntry(f"run {index}", None) for index in range(20)],
    )
    parent = DatasetGroup("Low temperature", subgroups=[first, second])
    group = DataGroup("Workspace1", subgroups=[parent])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer._refresh_tree(select_dataset_group=parent, refresh_viewers=False)

    table = explorer.details_widget.findChild(
        QtWidgets.QTableWidget, "group_child_collections_table"
    )
    assert table is not None and table.toolTip()
    assert table.rowCount() == 2
    assert [table.item(row, 0).text() for row in range(2)] == ["34", "70"]
    assert [table.item(row, 3).text() for row in range(2)] == ["30", "20"]


def test_project_explorer_opens_and_reloads_independent_group_slice_viewers(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    created_viewers = []

    class FakeCombo:
        def __init__(self, names):
            self.names = list(names)
            self.index = 0

        def setCurrentIndex(self, index):
            self.index = int(index)

        def currentText(self):
            return self.names[self.index]

    class FakeWindow:
        def __init__(self, viewer):
            self.viewer = viewer
            self.closed = False
            self.title = ""

        def close(self):
            self.closed = True
            if self.viewer.close_callback is not None:
                callback = self.viewer.close_callback
                self.viewer.close_callback = None
                callback()

        def setWindowTitle(self, title):
            self.title = title

    class FakeViewer:
        def __init__(self, datasets, *, dataset_names, dataset_group_keys=None):
            self.datasets = list(datasets)
            self.dataset_names = list(dataset_names)
            self.dataset_combo = FakeCombo(dataset_names)
            self.close_callback = None
            self.window = FakeWindow(self)
            self.shown = False
            self.replaced = False
            self.open_new_viewer_callback = None
            created_viewers.append(self)

        def set_close_callback(self, callback):
            self.close_callback = callback

        def set_open_new_viewer_callback(self, callback):
            self.open_new_viewer_callback = callback

        def show(self):
            self.shown = True

        def replace_datasets(
            self,
            datasets,
            *,
            dataset_names,
            dataset_group_keys=None,
            selected_dataset_name=None,
        ):
            self.replaced = True
            self.datasets = list(datasets)
            self.dataset_names = list(dataset_names)
            self.dataset_combo = FakeCombo(dataset_names)
            if selected_dataset_name in self.dataset_names:
                self.dataset_combo.setCurrentIndex(self.dataset_names.index(selected_dataset_name))

    monkeypatch.setattr(project_gui, "QtMDHistoSliceViewer", FakeViewer)

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0))
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(1)
    explorer.tree.setCurrentItem(dataset_item)
    assert not explorer.view_slice_button.isHidden()
    assert explorer.view_slice_button.isEnabled()

    viewer = explorer.open_slice_viewer_for_selection()

    assert viewer is created_viewers[0]
    assert viewer.dataset_names == ["first", "second"]
    assert viewer.dataset_combo.currentText() == "second"
    assert viewer.shown
    assert callable(viewer.open_new_viewer_callback)

    second_viewer = explorer.open_slice_viewer(
        group,
        selected_dataset_name="first",
        use_composite=False,
    )
    assert second_viewer is created_viewers[1]
    assert second_viewer is not viewer
    assert second_viewer.dataset_combo.currentText() == "first"
    assert len(explorer._slice_viewers[id(group)]) == 2

    group.add_dataset(DatasetEntry("third", _tiny_mdhisto_data(3.0)))
    viewer.shown = False
    reloaded = explorer.refresh_slice_viewer(group)

    assert reloaded is viewer
    assert len(created_viewers) == 2
    assert not viewer.window.closed
    assert viewer.replaced
    assert second_viewer.replaced
    assert not viewer.shown
    assert reloaded.dataset_names == ["first", "second", "third"]
    assert reloaded.dataset_combo.currentText() == "second"
    assert second_viewer.dataset_combo.currentText() == "first"

    viewer.window.close()
    assert explorer._slice_viewers[id(group)] == [second_viewer]


def test_project_explorer_loads_dataset_and_refreshes_details(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    source = tmp_path / "scan.nxs"
    source.write_bytes(b"placeholder")
    dataset = DatasetEntry(
        "scan",
        data=None,
        kind="nxs",
        metadata={"source_file": str(source), "import_status": "pending"},
    )
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)

    assert not explorer.load_dataset_button.isHidden()
    assert "Imported data: not loaded" in explorer.details_label.text()

    monkeypatch.setattr(
        project_gui,
        "load_mantid_mdhisto_nxs",
        lambda path, copy_metadata=False: _tiny_mdhisto_data(1.0),
    )

    assert explorer.load_dataset_for_selection() is True
    assert dataset.data is not None
    assert explorer.load_dataset_button.isHidden()
    assert "Axes\nDimensions: 2" in explorer.details_label.text()


def test_reload_data_helpers_and_collection_button_refresh_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    paths = [tmp_path / "first.nxs", tmp_path / "second.nxs"]
    for path in paths:
        path.write_bytes(b"source")
    first = DatasetEntry(
        "first", _tiny_mdhisto_data(1.0), kind="nxs",
        metadata={"source_file": str(paths[0]), "import_status": "loaded"},
    )
    second = DatasetEntry(
        "second", _tiny_mdhisto_data(2.0), kind="nxs",
        metadata={"source_file": str(paths[1]), "import_status": "loaded"},
    )
    subgroup = DatasetGroup("series", datasets=[first, second])
    group = DataGroup("workspace", subgroups=[subgroup])
    values = {paths[0]: 10.0, paths[1]: 20.0}
    monkeypatch.setattr(
        project_gui,
        "load_mantid_mdhisto_nxs",
        lambda path, copy_metadata=False: _tiny_mdhisto_data(values[path]),
    )

    original_revision = first.data_revision
    assert nfit.reload_dataset_data(first) is first.data
    assert first.data_revision == original_revision + 1
    np.testing.assert_allclose(first.data.signal, 10.0)
    assert first.data_matches_source

    explorer = NfitProjectExplorer(NfitProject([group]))
    subgroup_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(subgroup_item)
    assert not explorer.reload_data_button.isHidden()
    assert explorer.reload_data_button.isEnabled()
    refreshes = []
    monkeypatch.setattr(
        explorer,
        "refresh_slice_viewer",
        lambda owner, *, force_rebin=False: refreshes.append((owner, force_rebin)),
    )
    values[paths[0]] = 30.0
    values[paths[1]] = 40.0

    assert explorer.reload_data_for_selection()
    np.testing.assert_allclose(first.data.signal, 30.0)
    np.testing.assert_allclose(second.data.signal, 40.0)
    assert refreshes == [(group, True)]


def test_reload_dataset_failure_preserves_current_data(monkeypatch, tmp_path):
    source = tmp_path / "scan.nxs"
    source.write_bytes(b"source")
    original = _tiny_mdhisto_data(3.0)
    dataset = DatasetEntry(
        "scan", original, kind="nxs",
        metadata={"source_file": str(source), "import_status": "loaded"},
    )
    current = dataset.data
    monkeypatch.setattr(
        project_gui,
        "load_mantid_mdhisto_nxs",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unreadable")),
    )

    with pytest.raises(OSError, match="unreadable"):
        nfit.reload_dataset_data(dataset)
    assert dataset.data is current
    assert dataset.metadata["import_status"] == "loaded"


def test_project_explorer_refreshes_details_after_slice_viewer_lazy_load(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    class FakeWindow:
        def close(self):
            pass

        def setWindowTitle(self, title):
            self.title = title

    class FakeCombo:
        def __init__(self):
            self.index = 0

        def setCurrentIndex(self, index):
            self.index = int(index)

        def currentText(self):
            return "scan"

    class FakeViewer:
        def __init__(self, datasets, *, dataset_names, dataset_group_keys=None):
            self.datasets = list(datasets)
            self.dataset_names = list(dataset_names)
            self.dataset_combo = FakeCombo()
            self.window = FakeWindow()

        def show(self):
            pass

    source = tmp_path / "scan.nxs"
    source.write_bytes(b"placeholder")
    dataset = DatasetEntry(
        "scan",
        data=None,
        kind="nxs",
        metadata={"source_file": str(source), "import_status": "pending"},
    )
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    monkeypatch.setattr(project_gui, "QtMDHistoSliceViewer", FakeViewer)
    monkeypatch.setattr(
        project_gui,
        "load_mantid_mdhisto_nxs",
        lambda path, copy_metadata=False: _tiny_mdhisto_data(1.0),
    )

    viewer = explorer.open_slice_viewer_for_selection()

    assert viewer is not None
    assert dataset.data is not None
    assert explorer.load_dataset_button.isHidden()
    assert "Axes\nDimensions: 2" in explorer.details_label.text()


def test_project_explorer_interactive_controls_have_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    def missing_tooltips(root):
        interactive_classes = (
            QtWidgets.QAbstractButton,
            QtWidgets.QComboBox,
            QtWidgets.QLineEdit,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QSlider,
            QtWidgets.QTreeWidget,
        )
        ignored_object_names = {"qt_spinbox_lineedit", "qt_toolbar_ext_button"}
        missing = []
        for widget in root.findChildren(QtWidgets.QWidget):
            if not isinstance(widget, interactive_classes):
                continue
            if widget.objectName() in ignored_object_names:
                continue
            if not widget.toolTip().strip():
                label = widget.text() if hasattr(widget, "text") else widget.objectName()
                missing.append(f"{type(widget).__name__}:{label}")
        return missing

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    group = DataGroup("Workspace1", datasets=[dataset])
    create_mask(dataset)
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))
    tree_items = [
        explorer.tree.topLevelItem(0),
        explorer.tree.topLevelItem(0).child(0).child(0),
        explorer.tree.topLevelItem(0).child(0).child(0).child(0).child(0),
        explorer.tree.topLevelItem(0).child(1).child(0),
        explorer.tree.topLevelItem(0).child(2).child(0),
    ]
    for item in tree_items:
        explorer.tree.setCurrentItem(item)

    assert missing_tooltips(explorer.window) == []
    assert all(
        action.toolTip() for action in explorer.file_menu.actions() if not action.isSeparator()
    )

    point_data = PointListData(
        columns={
            "Temperature": np.array([1.0, 2.0]),
            "Moment": np.array([3.0, 4.0]),
            "Moment error": np.array([0.1, 0.2]),
            "Field": np.array([5.0, 6.0]),
        },
        units={"Temperature": "K", "Field": "T"},
        coordinate_names=["Temperature"],
        channels=[{"label": "Moment", "value": "Moment", "error": "Moment error"}],
    )
    point_dataset = DatasetEntry("magnetization", point_data, data_type="magnetization")
    point_explorer = NfitProjectExplorer(
        NfitProject([DataGroup("Workspace1", datasets=[point_dataset])])
    )
    point_explorer.tree.setCurrentItem(point_explorer.tree.topLevelItem(0).child(0).child(0))

    assert missing_tooltips(point_explorer.window) == []


def test_project_explorer_numeric_controls_ignore_wheel_and_hide_buttons(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace1", datasets=[dataset])]))
    spin_boxes = explorer.window.findChildren(QtWidgets.QAbstractSpinBox)
    assert spin_boxes
    assert all(
        spin_box.buttonSymbols() == QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        for spin_box in spin_boxes
    )

    spin_box = explorer.fit_weight_spin
    spin_box.setValue(2.0)
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(spin_box, wheel)

    assert spin_box.value() == 2.0

    later_spin_box = QtWidgets.QSpinBox()
    later_spin_box.show()
    QtWidgets.QApplication.processEvents()
    assert later_spin_box.buttonSymbols() == QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
    later_spin_box.close()


def test_dataset_rebin_panel_edits_one_nonuniform_axis_and_minimum_samples(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    data = PointListData(
        columns={
            "Temperature": np.array([2.0, 5.0, 12.0, 30.0]),
            "Moment": np.ones(4),
            "Moment error": np.ones(4),
        },
        units={"Temperature": "K"},
        coordinate_names=["Temperature"],
        channels=[{"label": "Moment", "value": "Moment", "error": "Moment error"}],
    )
    dataset = DatasetEntry("field series", data, data_type="magnetization")
    config = dataset_rebin_config(dataset)
    config.update({"enabled": True, "auto_rebin": False})
    explorer = NfitProjectExplorer(
        NfitProject([DataGroup("Workspace1", datasets=[dataset])])
    )
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    edges = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_axis_edges_0"
    )
    samples = explorer.details_widget.findChild(
        QtWidgets.QLineEdit, "dataset_rebin_minimum_samples"
    )
    assert edges is not None and edges.toolTip()
    assert samples is not None and samples.toolTip()
    edges.setText("[0, 3, 10, 40]")
    edges.editingFinished.emit()
    samples.setText("2")
    samples.editingFinished.emit()

    assert config["axes"][0]["bin_edges"] == [0.0, 3.0, 10.0, 40.0]
    assert config["axes"][0]["num_bins"] == 3
    assert config["minimum_samples"] == 2.0


def test_project_explorer_adds_edits_and_copies_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0))
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))

    first_dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(first_dataset_item)
    assert not explorer.add_mask_button.isHidden()
    mask = explorer.add_mask_to_selection()

    assert mask is first.masks[0]
    assert mask.name == "Mask1"
    expected_parameters = default_mask_parameters("coordinate_range")
    assert {name: mask.parameters[name] for name in expected_parameters} == expected_parameters
    assert mask.parameters["axis_0"] == [1.0, 0.0]
    assert mask.parameters["axis_1"] == [0.0, 1.0]
    assert explorer.tree.currentItem().text(0) == "Mask1"

    combo = explorer.mask_type_combo
    combo.setCurrentIndex(combo.findData("box"))
    assert mask.parameters["width"] == [0.0, 0.0, 0.0, 0.0]
    explorer._set_mask_parameter("center", "[1, 2, 3, 4]")

    assert mask.type == "box"
    assert mask.parameters["center"] == [1, 2, 3, 4]
    assert "width" in mask.parameters

    tooltip = mask_parameter_tooltip("box", "center")
    editors = explorer.mask_parameter_widget.findChildren(QtWidgets.QLineEdit)
    assert any(editor.toolTip() == tooltip for editor in editors)
    assert "Parameter: center" in tooltip
    assert "Description:" in tooltip
    assert "Allowed values:" in tooltip
    assert "Data type:" in tooltip
    assert "Default:" in tooltip
    assert "Example:" in tooltip

    explorer.copy_selected()
    second_dataset_item = explorer.tree.topLevelItem(0).child(0).child(1)
    explorer.tree.setCurrentItem(second_dataset_item)
    # Switching away from the source mask deletes its details-panel label on
    # the next event cycle. Pasting must not retain and touch that object.
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    explorer.paste_into_selection()

    assert len(second.masks) == 1
    assert second.masks[0].type == "box"
    assert second.masks[0].parameters["center"] == [1, 2, 3, 4]

    second_mask = create_mask(first)
    explorer._refresh_tree(select_group=group, select_mask=second_mask)
    mask_parent = explorer.tree.topLevelItem(0).child(0).child(0).child(0)
    first_mask_item = mask_parent.child(0)
    second_mask_item = mask_parent.child(1)
    explorer.tree.setCurrentItem(second_mask_item)

    assert explorer.move_or_copy_selected_to_item(first_mask_item, copy_item=False)
    assert first.masks == [second_mask, mask]


def test_dataset_for_slice_viewer_combines_file_and_nfit_coordinate_masks():
    h_axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    e_axis = MDHistoAxis("E", np.array([0.0, 1.0]), "meV", "energy")
    file_mask = np.zeros((2, 1), dtype=bool)
    file_mask[1, 0] = True
    data = MDHistoData(
        axes=(h_axis, e_axis),
        signal=np.arange(2, dtype=float).reshape(2, 1),
        errors=np.ones((2, 1)),
        mask=file_mask,
        num_events=np.ones((2, 1)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset)
    mask.parameters["H"] = [1.25, 1.75]

    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["file_mask"], file_mask)
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[False], [True]])
    np.testing.assert_array_equal(viewed.mask, [[False], [True]])


def test_coordinate_range_mask_axis_vectors_define_coordinates():
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 10.0]
    mask.parameters["axis_0"] = [0.0, 1.0]
    mask.parameters["axis_1"] = [1.0, 0.0]

    viewed = dataset_for_slice_viewer(dataset)

    expected = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)


def test_dataset_for_slice_viewer_applies_mask_order_invert_and_additive():
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5, 2.5]), "rlu", "momentum")
    e_axis = MDHistoAxis("E", np.array([0.0]), "meV", "energy")
    file_mask = np.zeros((3, 1), dtype=bool)
    file_mask[2, 0] = True
    data = MDHistoData(
        axes=(h_axis, e_axis),
        signal=np.arange(3, dtype=float).reshape(3, 1),
        errors=np.ones((3, 1)),
        mask=file_mask,
        num_events=np.ones((3, 1)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    masking = create_mask(dataset)
    masking.parameters["H"] = [1.0, 2.0]
    additive = create_mask(dataset)
    additive.parameters["H"] = [1.0, 2.0]
    additive.additive = True

    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[False], [False], [False]])
    np.testing.assert_array_equal(viewed.mask, [[False], [False], [True]])

    dataset.masks = [additive, masking]
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[False], [True], [True]])

    masking.invert = True
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[True], [False], [False]])

    masking.enabled = False
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[False], [False], [False]])


def test_dataset_for_slice_viewer_applies_energy_q_range_mask():
    e_axis = MDHistoAxis("E", np.array([3.0, 5.0, 7.0]), "meV", "energy")
    q_axis = MDHistoAxis("|Q|", np.array([0.0, 0.5, 1.0, 1.5]), "Angstrom^-1", "momentum")
    data = MDHistoData(
        axes=(e_axis, q_axis),
        signal=np.arange(6, dtype=float).reshape(2, 3),
        errors=np.ones((2, 3)),
        mask=np.zeros((2, 3), dtype=bool),
        num_events=np.ones((2, 3)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="energy_q_range")
    mask.parameters["energy"] = [3.0, 5.0]
    mask.parameters["q_modulus"] = [0.5, 1.0]

    viewed = dataset_for_slice_viewer(dataset)

    expected = np.zeros((2, 3), dtype=bool)
    expected[0, 1] = True
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)
    np.testing.assert_array_equal(viewed.mask, expected)


def test_energy_q_range_mask_accepts_leading_decimal_and_projected_q_axes():
    hh_axis = MDHistoAxis("[H,H,0]", np.array([0.0, 0.0, 1.0]), "rlu", "momentum")
    l_axis = MDHistoAxis("[0,0,L]", np.array([0.0, 0.0, 1.0]), "rlu", "momentum")
    data = MDHistoData(
        axes=(hh_axis, l_axis),
        signal=np.arange(4, dtype=float).reshape(2, 2),
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={
            "lattice_parameters": {
                "a": 2.0 * np.pi,
                "b": 2.0 * np.pi,
                "c": 2.0 * np.pi,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            }
        },
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="energy_q_range")
    mask.parameters["q_modulus"] = "[0, .5]"

    viewed = dataset_for_slice_viewer(dataset)

    expected = np.array([[True, True], [False, False]])
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)


def test_dataset_for_slice_viewer_applies_phonon_cone_mask():
    # Bin edges chosen so centers are H in {0, 1} and E in {0, 10}.
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5]), "rlu", "h")
    k_axis = MDHistoAxis("K", np.array([-0.5, 0.5]), "rlu", "k")
    l_axis = MDHistoAxis("L", np.array([-0.5, 0.5]), "rlu", "l")
    e_axis = MDHistoAxis("E", np.array([-5.0, 5.0, 15.0]), "meV", "energy_transfer")
    shape = (2, 1, 1, 2)
    data = MDHistoData(
        axes=(h_axis, k_axis, l_axis, e_axis),
        signal=np.zeros(shape),
        errors=np.ones(shape),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        # Cubic 2*pi lattice makes 1 rlu equal 1 inverse angstrom.
        metadata={"lattice_parameters": {"a": 2.0 * np.pi, "b": 2.0 * np.pi, "c": 2.0 * np.pi}},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="phonon_cone")
    mask.parameters["center"] = [0.0, 0.0, 0.0]
    mask.parameters["slope"] = 35.0
    mask.parameters["radius"] = 0.0

    viewed = dataset_for_slice_viewer(dataset)

    # Only the H=0 column falls inside the cone; the H=1 column is 1 inv-angstrom away.
    expected = np.zeros(shape, dtype=bool)
    expected[0] = True
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)


def test_phonon_cone_mask_combines_multiple_bragg_centers_for_histogram_and_points():
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5, 2.5]), "rlu", "h")
    k_axis = MDHistoAxis("K", np.array([-0.5, 0.5]), "rlu", "k")
    l_axis = MDHistoAxis("L", np.array([-0.5, 0.5]), "rlu", "l")
    e_axis = MDHistoAxis("E", np.array([-0.5, 0.5]), "meV", "energy_transfer")
    shape = (3, 1, 1, 1)
    metadata = {
        "lattice_parameters": {
            "a": 2.0 * np.pi,
            "b": 2.0 * np.pi,
            "c": 2.0 * np.pi,
        }
    }
    histogram = DatasetEntry(
        "grid",
        MDHistoData(
            axes=(h_axis, k_axis, l_axis, e_axis),
            signal=np.zeros(shape),
            errors=np.ones(shape),
            mask=np.zeros(shape, dtype=bool),
            num_events=np.ones(shape),
            metadata=metadata,
        ),
    )
    histogram_mask = create_mask(histogram, type="phonon_cone")
    histogram_mask.parameters.update(
        {"center": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "slope": 10.0, "radius": 0.1}
    )

    viewed = dataset_for_slice_viewer(histogram)
    np.testing.assert_array_equal(
        viewed.metadata["nfit_mask"].reshape(-1),
        [True, False, True],
    )

    points = PointData4D(
        H=[0.0, 1.0, 2.0],
        K=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0],
        E=[0.0, 0.0, 0.0],
        intensity=[1.0, 1.0, 1.0],
        sigma=[1.0, 1.0, 1.0],
        metadata=metadata,
    )
    point_dataset = DatasetEntry("points", points)
    point_mask = create_mask(point_dataset, type="phonon_cone")
    point_mask.parameters.update(histogram_mask.parameters)

    np.testing.assert_array_equal(
        project_gui._nfit_mask_for_point_data(point_dataset, points),
        [True, False, True],
    )


def test_phonon_cone_mask_reconstructs_physical_hkl_from_rebinned_axes():
    axes = (
        MDHistoAxis("[H,H,H]", np.array([0.5, 1.5]), "rlu", "h"),
        MDHistoAxis("[K,-K,0]", np.array([-0.5, 0.5]), "rlu", "k"),
        MDHistoAxis("[L,L,-2L]", np.array([1.5, 2.5]), "rlu", "l"),
        MDHistoAxis("DeltaE", np.array([-0.5, 0.5]), "meV", "energy_transfer"),
    )
    data = MDHistoData(
        axes=axes,
        signal=np.ones((1, 1, 1, 1)),
        errors=np.ones((1, 1, 1, 1)),
        mask=np.zeros((1, 1, 1, 1), dtype=bool),
        num_events=np.ones((1, 1, 1, 1)),
        metadata={
            "lattice_parameters": {"a": 2.0 * np.pi, "b": 2.0 * np.pi, "c": 2.0 * np.pi},
            "rebin": {
                "vectors": [
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0],
                    [1.0, 1.0, -2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
        },
    )
    dataset = DatasetEntry("projected", data)
    mask = create_mask(dataset, type="phonon_cone")
    mask.parameters.update({"center": [3.0, 3.0, -3.0], "slope": 10.0, "radius": 0.1})

    coords = project_gui._mdhisto_coordinate_grids(data)
    assert coords["H"].item() == pytest.approx(3.0)
    assert coords["K"].item() == pytest.approx(3.0)
    assert coords["L"].item() == pytest.approx(-3.0)
    viewed = dataset_for_slice_viewer(dataset)
    assert viewed.metadata["nfit_mask"].item()


def test_phonon_cone_mask_is_inert_with_nonpositive_slope():
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5]), "rlu", "h")
    k_axis = MDHistoAxis("K", np.array([-0.5, 0.5]), "rlu", "k")
    l_axis = MDHistoAxis("L", np.array([-0.5, 0.5]), "rlu", "l")
    e_axis = MDHistoAxis("E", np.array([-5.0, 5.0, 15.0]), "meV", "energy_transfer")
    shape = (2, 1, 1, 2)
    data = MDHistoData(
        axes=(h_axis, k_axis, l_axis, e_axis),
        signal=np.zeros(shape),
        errors=np.ones(shape),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        metadata={"lattice_parameters": {"a": 2.0 * np.pi, "b": 2.0 * np.pi, "c": 2.0 * np.pi}},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="phonon_cone")
    mask.parameters["center"] = [0.0, 0.0, 0.0]
    # The default zero slope must leave the starter mask inert instead of raising.
    mask.parameters["slope"] = 0.0

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed.metadata["nfit_mask_count"] == 0


def test_dataset_for_slice_viewer_applies_box_mask():
    data = _he_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="box")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [0.0, 0.0]
    # Full widths of 1.0 (H) and 5.0 (E) => half-widths 0.5 and 2.5.
    mask.parameters["width"] = [1.0, 5.0]

    viewed = dataset_for_slice_viewer(dataset)

    # Only the H=0, E=0 bin falls inside the box.
    expected = np.zeros(data.shape, dtype=bool)
    expected[0, 0] = True
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)
    np.testing.assert_array_equal(viewed.mask, expected)


def test_box_mask_is_inert_with_zero_width():
    data = _he_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="box")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [0.0, 0.0]
    # The default zero width must leave the starter mask inert.

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed.metadata["nfit_mask_count"] == 0


def test_dataset_for_slice_viewer_applies_ellipsoid_mask():
    data = _he_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="ellipsoid")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [0.0, 0.0]
    # (H / 1)^2 + (E / 10)^2 <= 1 selects H in {0, 1} at E=0 and H=0 at E=10.
    mask.parameters["radii"] = [1.0, 10.0]

    viewed = dataset_for_slice_viewer(dataset)

    expected = np.zeros(data.shape, dtype=bool)
    expected[0, 0] = True  # H=0, E=0
    expected[1, 0] = True  # H=1, E=0 (on the boundary)
    expected[0, 1] = True  # H=0, E=10 (on the boundary)
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)
    np.testing.assert_array_equal(viewed.mask, expected)


def test_ellipsoid_mask_is_inert_with_zero_radius():
    data = _he_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="ellipsoid")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [0.0, 0.0]
    # The default zero radii must leave the starter mask inert.

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed.metadata["nfit_mask_count"] == 0


def test_box_mask_resolves_projected_axes():
    hh_axis = MDHistoAxis("[H,H,0]", np.array([-0.5, 0.5, 1.5]), "rlu", "momentum")
    l_axis = MDHistoAxis("[0,0,L]", np.array([-0.5, 0.5, 1.5]), "rlu", "momentum")
    shape = (2, 2)
    data = MDHistoData(
        axes=(hh_axis, l_axis),
        signal=np.zeros(shape),
        errors=np.ones(shape),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset, type="box")
    # Project onto the H and L reciprocal-space coordinates.
    mask.parameters["axes"] = ["H", "L"]
    mask.parameters["center"] = [0.0, 0.0]
    mask.parameters["width"] = [1.0, 1.0]

    viewed = dataset_for_slice_viewer(dataset)

    # Only the bin whose [H,H,0]=0 and [0,0,L]=0 lands inside the box.
    expected = np.zeros(shape, dtype=bool)
    expected[0, 0] = True
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], expected)


def test_project_explorer_adds_and_edits_models(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    group_item = explorer.tree.topLevelItem(0)

    explorer.tree.setCurrentItem(group_item)
    assert not explorer.add_model_button.isHidden()

    model = explorer.add_model_to_selection()

    assert model is group.models["Model1"]
    assert model.name == "Model1"
    assert model.parameters == default_model_parameters("constant_background")
    assert model.config == default_model_config("constant_background")
    assert model.fit_parameters == default_model_fit_parameters("constant_background")
    assert model.sharing == {"constant": {"mode": "global", "groups": {}}}
    assert explorer.tree.currentItem().text(0) == "Model1"

    fit_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_fit_parameters_group"
    )
    config_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_config_group"
    )
    model_scroll = explorer.window.findChild(QtWidgets.QScrollArea, "model_parameter_scroll")
    assert fit_group is not None
    assert config_group is not None
    assert model_scroll is not None
    assert model_scroll.widget() is explorer.model_parameter_widget
    assert explorer.details_scroll.isHidden()
    right_layout = model_scroll.parentWidget().layout()
    assert right_layout.stretch(right_layout.indexOf(model_scroll)) > right_layout.stretch(
        right_layout.indexOf(explorer.details_scroll)
    )
    assert fit_group.title() == "Fit Parameters"
    assert config_group.title() == "Configuration Settings"

    current_item = explorer.tree.currentItem()
    current_item.setText(0, "background")
    explorer._tree_item_changed(current_item, 0)
    assert "background" in group.models
    assert model.name == "background"

    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("linear_background"))
    assert model.type == "linear_background"
    assert model.parameters["c0"] == 0.0
    assert model.parameters["c1"] == 0.0
    assert model.fit_parameters["c0"] is False
    assert model.fit_parameters["c1"] is False
    assert model.sharing["c0"]["mode"] == "global"
    assert model.sharing["c1"]["mode"] == "global"
    assert model.config == default_model_config("linear_background")

    explorer._set_model_parameter("c1", "0.02")
    explorer._set_model_fit_parameter("c1", True)
    explorer._set_model_sharing_mode("c1", "per_dataset")

    assert model.parameters["c1"] == 0.02
    assert model.fit_parameters["c1"] is True
    assert model.sharing["c1"]["mode"] == "per_dataset"

    tooltip = model_parameter_tooltip("linear_background", "c1")
    editors = explorer.model_parameter_widget.findChildren(QtWidgets.QLineEdit)
    checks = explorer.model_parameter_widget.findChildren(QtWidgets.QCheckBox)
    sharing = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox,
        "model_parameter_sharing_c1",
    )
    assert any(editor.toolTip() == tooltip for editor in editors)
    assert any(check.text() == "Fit" for check in checks)
    assert sharing is not None
    assert sharing.currentData() == "per_dataset"
    assert sharing.toolTip()
    assert "Fit:" in tooltip
    assert "Sharing:" in tooltip

    plot_label_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_parameter_plot_label_c1",
    )
    assert plot_label_editor is not None
    assert plot_label_editor.toolTip()
    plot_label_editor.setText(r"$c_1$")
    explorer._set_model_parameter_plot_label("c1", plot_label_editor.text())
    assert model.metadata["parameter_labels"]["c1"] == r"$c_1$"

    models_item = explorer.tree.topLevelItem(0).child(1)
    explorer.tree.setCurrentItem(models_item)
    assert not explorer.add_model_button.isHidden()


def test_models_folder_edits_and_validates_fit_constraints(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    group = DataGroup("Datagroup1")
    first = create_model_component(group, "A")
    second = create_model_component(group, "B")
    first.fit_parameters["constant"] = True
    second.fit_parameters["constant"] = True
    second.constraints = [{"parameter": "constant", "op": "=", "expression": "10 - `A.constant`"}]
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(1))
    table = explorer.details_widget.findChild(QtWidgets.QTableWidget, "model_constraints_table")
    add_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "add_model_constraint_button"
    )
    check_button = explorer.details_widget.findChild(
        QtWidgets.QPushButton, "check_model_constraints_button"
    )
    status = explorer.details_widget.findChild(QtWidgets.QLabel, "model_constraints_status")

    assert table is not None and table.rowCount() == 1
    assert table.cellWidget(0, 0).currentText() == "B.constant"
    assert table.cellWidget(0, 1).currentData() == "="
    assert table.cellWidget(0, 2).text() == "10 - `A.constant`"
    assert all(widget.toolTip() for widget in (table, add_button, check_button))

    check_button.click()
    assert status.text() == "1 constraint(s) valid"
    add_button.click()
    assert table.rowCount() == 2
    assert len(first.constraints) + len(second.constraints) == 2


def test_model_constraint_validation_rejects_cycles():
    group = DataGroup("Datagroup1")
    first = create_model_component(group, "A")
    second = create_model_component(group, "B")
    first.fit_parameters["constant"] = True
    second.fit_parameters["constant"] = True
    first.constraints = [{"parameter": "constant", "op": "=", "expression": "`B.constant`"}]
    second.constraints = [{"parameter": "constant", "op": "=", "expression": "`A.constant`"}]

    with pytest.raises(ValueError, match="cyclic exact constraint"):
        project_gui._validate_model_constraints(group)


def test_renaming_model_updates_constraint_parameter_references():
    group = DataGroup("Datagroup1")
    first = create_model_component(group, "A")
    second = create_model_component(group, "B")
    second.constraints = [
        {"parameter": "constant", "op": "=", "expression": "10 - `A.constant`"},
        {"parameter": "constant", "op": ">=", "reference": "A.constant"},
    ]

    project_gui._rename_model_constraint_references(group, first, "A", "Signal")

    assert second.constraints[0]["expression"] == "10 - `Signal.constant`"
    assert second.constraints[1]["reference"] == "Signal.constant"


def test_project_explorer_model_limits_and_applies_to_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtWidgets

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))
    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)

    fit_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_fit_parameters_group"
    )
    assert fit_group.findChild(QtWidgets.QLineEdit, "model_parameter_min_constant") is not None
    assert fit_group.findChild(QtWidgets.QLineEdit, "model_parameter_max_constant") is not None
    explorer._set_model_limit("constant", 0, "0.0")
    explorer._set_model_limit("constant", 1, "5")
    assert model.limits["constant"] == [0.0, 5]
    explorer._set_model_limit("constant", 0, "")
    assert model.limits["constant"] == [None, 5]
    explorer._set_model_limit("constant", 1, "")
    assert "constant" not in model.limits

    scope_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_dataset_scope_group"
    )
    assert scope_group is not None
    assert scope_group.findChild(QtWidgets.QLineEdit, "model_applies_to_editor") is not None
    explorer._set_model_applies_to("first, second")
    assert model.applies_to == ["first", "second"]
    explorer._set_model_applies_to("")
    assert model.applies_to is None


def test_model_selector_filters_models_by_category(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    category = explorer.model_category_combo
    model_type = explorer.model_type_combo
    assert category is not None and category.toolTip()
    assert model_type is not None and model_type.toolTip()
    assert [
        category.itemText(index) for index in range(category.count())
    ] == [
        "Primitive models",
        "Spin fluctuations",
        "Electronic structure",
        "Heat capacity",
        "Magnetization",
    ]
    assert category.currentData() == "primitive"
    assert [
        model_type.itemData(index) for index in range(model_type.count())
    ] == ["constant_background", "linear_background"]

    category.setCurrentIndex(category.findData("electronic_structure"))
    assert model.type == "constant_background"
    assert model_type.currentData() is None
    assert [
        model_type.itemData(index)
        for index in range(1, model_type.count())
    ] == [
        "tight_binding",
        "lindhard",
        "stoner_rpa",
        "matrix_rpa",
        "hubbard_hund_rpa",
    ]

    model_type.setCurrentIndex(model_type.findData("lindhard"))
    assert model.type == "lindhard"
    assert category.currentData() == "electronic_structure"
    assert not explorer.model_selector_widget.isHidden()
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_spin_model_form_factor_custom_choice_controls_coefficients(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    category = explorer.model_category_combo
    category.setCurrentIndex(category.findData("spin_fluctuation"))
    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("local_relaxational"))

    form_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_config_choice_ion"
    )
    assert form_combo is not None
    assert form_combo.findData(project_gui.CUSTOM_FORM_FACTOR_CHOICE) >= 0
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_config_form_factor_coefficients"
        )
        is None
    )

    form_combo.setCurrentIndex(form_combo.findData("Fe2"))
    assert model.config["ion"] == "Fe2"
    assert model.config["form_factor_coefficients"] == ""

    form_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_config_choice_ion"
    )
    form_combo.setCurrentIndex(form_combo.findData(project_gui.CUSTOM_FORM_FACTOR_CHOICE))
    coeff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_form_factor_coefficients"
    )
    assert coeff_editor is not None
    coeff_editor.setText("0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119")
    explorer._set_model_config_setting("form_factor_coefficients", coeff_editor.text())
    assert model.config["ion"] == project_gui.CUSTOM_FORM_FACTOR_CHOICE
    assert len(model.config["form_factor_coefficients"]) == 7

    form_combo = explorer.model_parameter_widget.findChild(
        QtWidgets.QComboBox, "model_config_choice_ion"
    )
    form_combo.setCurrentIndex(form_combo.findData("Mn2"))
    assert model.config["ion"] == "Mn2"
    assert model.config["form_factor_coefficients"] == ""
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_config_form_factor_coefficients"
        )
        is None
    )


def test_project_explorer_context_menu_actions_and_source_change(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    source = tmp_path / "old.nxs"
    replacement = tmp_path / "new.nxs"
    source.write_bytes(b"old")
    replacement.write_bytes(b"new")
    dataset = DatasetEntry(
        "first",
        _tiny_mdhisto_data(1.0),
        kind="nxs",
        metadata={"source_file": str(source), "import_status": "loaded"},
    )
    mask = create_mask(dataset)
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)
    assert not explorer.enabled_check.isHidden()
    assert explorer.enabled_check.isChecked()
    explorer.fit_weight_spin.setValue(2.25)
    assert dataset.fit_weight == 2.25
    explorer.enabled_check.setChecked(False)
    assert dataset.enabled is False
    assert explorer.context_menu_action_names(explorer.tree.currentItem()) == [
        "Copy",
        "Paste",
        "Enable",
        "Rename",
        "Delete",
        "Open Analysis Window",
        "View in data viewer",
        "Reload data",
        "Show file location",
        "Change file source",
        "Copy workflow script",
        "Save workflow script...",
        "Add mask",
        "Add background",
    ]

    dataset_item = explorer.tree.currentItem()
    dataset_item.setText(0, "renamed")
    explorer._tree_item_changed(dataset_item, 0)
    assert dataset.name == "renamed"
    explorer.enabled_check.setChecked(True)
    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(replacement), "All files (*)"),
    )
    assert explorer.change_file_source_for_selection() is True
    assert dataset.metadata["source_file"] == str(replacement)
    assert dataset.metadata["import_status"] == "pending"
    assert dataset.data is None
    assert dataset.kind == "nxs"

    mask_item = explorer.tree.topLevelItem(0).child(0).child(0).child(0).child(0)
    assert explorer.context_menu_action_names(mask_item) == [
        "Copy",
        "Disable",
        "Rename",
        "Delete",
        "View in data viewer",
    ]
    explorer.tree.setCurrentItem(mask_item)
    explorer.enabled_check.setChecked(False)
    assert mask.enabled is False
    assert explorer.context_menu_action_names(explorer.tree.currentItem()) == [
        "Copy",
        "Enable",
        "Rename",
        "Delete",
        "View in data viewer",
    ]

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    assert explorer.context_menu_action_names(model_item) == ["Disable", "Rename", "Delete"]
    explorer.tree.setCurrentItem(model_item)
    explorer.enabled_check.setChecked(False)
    assert model.enabled is False
    assert explorer.context_menu_action_names(explorer.tree.currentItem()) == [
        "Enable",
        "Rename",
        "Delete",
    ]

    models_item = explorer.tree.topLevelItem(0).child(1)
    assert explorer.context_menu_action_names(models_item) == ["Add model"]
    datasets_item = explorer.tree.topLevelItem(0).child(0)
    assert "Add dataset" in explorer.context_menu_action_names(datasets_item)
    assert model.name in group.models


def test_mask_and_background_folders_have_bulk_enabled_and_scale_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("sample", _tiny_mdhisto_data(5.0), kind="mdhisto")
    first_mask = create_mask(dataset, "Mask1")
    second_mask = create_mask(dataset, "Mask2")
    second_mask.enabled = False
    first_background = project_gui.BackgroundSpec("Background1", "first", scale=1.0)
    second_background = project_gui.BackgroundSpec(
        "Background2", "second", scale=2.0, enabled=False
    )
    dataset.backgrounds.extend([first_background, second_background])
    group = DataGroup("workspace", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    monkeypatch.setattr(explorer, "_record_data_group_state_change", lambda _group: False)
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda *args, **kwargs: None)

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    masks_item = dataset_item.child(0)
    explorer.tree.setCurrentItem(masks_item)
    assert not explorer.enabled_check.isHidden()
    assert explorer.enabled_check.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    explorer.enabled_check.setChecked(False)
    assert not first_mask.enabled
    assert not second_mask.enabled
    assert project_gui.dataset_mask_application_config(dataset)["stale"] is True

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    backgrounds_item = dataset_item.child(1)
    explorer.tree.setCurrentItem(backgrounds_item)
    assert not explorer.enabled_check.isHidden()
    assert explorer.enabled_check.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    assert not explorer.background_bulk_widget.isHidden()
    assert explorer.background_bulk_scale_edit.text() == ""
    explorer.background_bulk_scale_edit.setText("3.5")
    explorer._set_background_bulk_scale("3.5")
    assert [item.scale for item in dataset.backgrounds] == [3.5, 3.5]
    explorer.enabled_check.setChecked(True)
    assert all(item.enabled for item in dataset.backgrounds)

    nfit.set_mask_collection_enabled(dataset, True)
    nfit.set_background_collection(dataset, enabled=False, scale=0.25)
    assert all(item.enabled for item in dataset.masks)
    assert not any(item.enabled for item in dataset.backgrounds)
    assert [item.scale for item in dataset.backgrounds] == [0.25, 0.25]


def test_shared_mask_and_background_folders_apply_bulk_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    pytest.importorskip("PySide6.QtWidgets")

    sample = DatasetEntry("sample", _tiny_mdhisto_data(5.0), kind="mdhisto")
    first_mask = project_gui.MaskSpec("Mask1", "coordinate_range", enabled=True)
    second_mask = project_gui.MaskSpec("Mask2", "coordinate_range", enabled=False)
    first_background = project_gui.BackgroundSpec("Background1", "first", scale=1.0)
    second_background = project_gui.BackgroundSpec(
        "Background2", "second", scale=2.0, enabled=False
    )
    subgroup = DatasetGroup(
        "series",
        datasets=[sample],
        masks=[first_mask, second_mask],
        backgrounds=[first_background, second_background],
    )
    group = DataGroup("workspace", subgroups=[subgroup])
    explorer = NfitProjectExplorer(NfitProject([group]))
    monkeypatch.setattr(explorer, "_record_data_group_state_change", lambda _group: False)
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda *args, **kwargs: None)

    subgroup_item = explorer.tree.topLevelItem(0).child(0).child(0)
    masks_item = next(
        subgroup_item.child(index)
        for index in range(subgroup_item.childCount())
        if explorer._objects_for_item(subgroup_item.child(index))[4] == "group_masks"
    )
    explorer.tree.setCurrentItem(masks_item)
    assert explorer.enabled_check.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    explorer.enabled_check.setChecked(True)
    assert all(mask.enabled for mask in subgroup.masks)

    subgroup_item = explorer.tree.topLevelItem(0).child(0).child(0)
    backgrounds_item = next(
        subgroup_item.child(index)
        for index in range(subgroup_item.childCount())
        if explorer._objects_for_item(subgroup_item.child(index))[4]
        == "group_backgrounds"
    )
    explorer.tree.setCurrentItem(backgrounds_item)
    explorer._set_background_bulk_scale("4")
    assert [background.scale for background in subgroup.backgrounds] == [4.0, 4.0]
    explorer.enabled_check.setChecked(False)
    assert not any(background.enabled for background in subgroup.backgrounds)
    assert project_gui.data_group_composite_config(
        project_gui._composite_scope(group, subgroup)
    )["stale"] is True


def test_recent_project_helpers_and_file_menu(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    class FakeSettings:
        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    settings = FakeSettings()
    monkeypatch.setattr(project_gui, "_is_pytest_temporary_project", lambda path: False)
    first = tmp_path / "first.nfit"
    second = tmp_path / "second.nfit"
    missing = tmp_path / "missing.nfit"
    save_project(NfitProject(), first)
    save_project(NfitProject([DataGroup("Datagroup1")]), second)

    remember_recent_project(first, settings)
    remember_recent_project(second, settings)
    remember_recent_project(first, settings)

    assert recent_project_paths(settings) == [first, second]

    remember_recent_project(missing, settings)
    assert recent_project_paths(settings)[0] == missing
    assert forget_missing_recent_projects(settings) == [first, second]

    monkeypatch.setattr(project_gui, "_settings", lambda: settings)
    explorer = NfitProjectExplorer()
    explorer._refresh_recent_projects_menu()
    action_texts = [action.text() for action in explorer.recent_projects_menu.actions()]
    assert str(first) in action_texts
    assert str(second) in action_texts

    assert explorer.open_project_path(second)
    assert explorer.project_path == second
    assert [group.name for group in explorer.project.data_groups] == ["Datagroup1"]
    assert recent_project_paths(settings)[0] == second
    reload_action = next(
        action for action in explorer.file_menu.actions()
        if action.text() == "Reload from Disk"
    )
    assert reload_action.isEnabled()
    assert reload_action.toolTip()


def test_project_title_shows_saved_size_and_refreshes_it_only_after_save(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    unsaved = NfitProjectExplorer()
    assert unsaved.window.windowTitle() == "nfit Project Explorer - Untitled"

    path = tmp_path / "sized.nfit"
    save_project(NfitProject([DataGroup("saved")]), path)
    explorer = NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    opened_title = explorer.window.windowTitle()
    assert opened_title == (
        f"nfit Project Explorer - {path} "
        f"({project_gui._format_project_file_size(path.stat().st_size)})"
    )

    with path.open("ab") as stream:
        stream.write(b"external size change")
    explorer._mark_dirty()
    assert explorer.window.windowTitle() == f"{opened_title} *"

    def fake_save_project(project, target, **_kwargs):
        Path(target).write_bytes(b"saved archive")
        project._project_path = Path(target)

    monkeypatch.setattr(explorer, "_project_changed_on_disk", lambda: False)
    monkeypatch.setattr(project_gui, "save_project", fake_save_project)
    monkeypatch.setattr(project_gui, "_project_file_size", lambda _path: 2 * 1024**3)
    assert explorer.save()
    assert explorer.window.windowTitle() == (
        f"nfit Project Explorer - {path} (2.00 GB)"
    )


def test_project_explorer_detects_reloads_and_protects_external_file_changes(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    path = tmp_path / "shared.nfit"
    save_project(NfitProject([DataGroup("first")]), path)
    explorer = NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)

    save_project(NfitProject([DataGroup("external")]), path)
    assert explorer._project_changed_on_disk()
    monkeypatch.setattr(explorer, "_prompt_external_change_action", lambda: "reload")
    assert explorer.check_for_external_project_change()
    assert [group.name for group in explorer.project.data_groups] == ["external"]
    assert explorer.has_unsaved_changes is False
    assert not explorer._project_changed_on_disk()

    save_project(NfitProject([DataGroup("ignored")]), path)
    monkeypatch.setattr(explorer, "_prompt_external_change_action", lambda: "keep")
    assert explorer.check_for_external_project_change()
    assert [group.name for group in explorer.project.data_groups] == ["external"]
    assert not explorer.check_for_external_project_change()

    explorer.project.data_groups[0].name = "local"
    explorer._mark_dirty()
    monkeypatch.setattr(explorer, "_prompt_save_conflict_action", lambda: "cancel")
    assert not explorer.save()
    assert load_project(path).data_groups[0].name == "ignored"

    monkeypatch.setattr(explorer, "_prompt_save_conflict_action", lambda: "overwrite")
    assert explorer.save()
    assert load_project(path).data_groups[0].name == "local"
    assert explorer.has_unsaved_changes is False


def test_reload_from_disk_confirms_before_discarding_unsaved_state(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    path = tmp_path / "reload.nfit"
    save_project(NfitProject([DataGroup("saved")]), path)
    explorer = NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    explorer.project.data_groups[0].name = "unsaved"
    explorer._mark_dirty()

    monkeypatch.setattr(explorer, "_prompt_reload_unsaved_action", lambda: "cancel")
    assert not explorer.reload_project_from_disk()
    assert explorer.project.data_groups[0].name == "unsaved"

    monkeypatch.setattr(explorer, "_prompt_reload_unsaved_action", lambda: "reload")
    assert explorer.reload_project_from_disk()
    assert explorer.project.data_groups[0].name == "saved"
    assert explorer.has_unsaved_changes is False


def test_recent_projects_ignore_and_purge_pytest_temporary_projects(tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    settings = FakeSettings()
    temporary = tmp_path / "dropped.nfit"
    durable = Path("/research/fit.nfit")
    settings.setValue(project_gui.RECENT_PROJECTS_KEY, [str(temporary), str(durable)])

    assert recent_project_paths(settings) == [durable]
    assert settings.value(project_gui.RECENT_PROJECTS_KEY) == [str(durable)]
    assert remember_recent_project(temporary, settings) == [durable]
    assert settings.value(project_gui.RECENT_PROJECTS_KEY) == [str(durable)]


def test_tree_drop_loads_project_or_creates_a_workspace_for_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    project_path = tmp_path / "dropped.nfit"
    save_project(NfitProject([DataGroup("dropped workspace")]), project_path)
    explorer = NfitProjectExplorer(NfitProject([DataGroup("existing")]))
    monkeypatch.setattr(explorer, "_confirm_save_before_closing_project", lambda: True)

    assert explorer.load_dropped_paths([project_path], explorer.tree.topLevelItem(0))
    assert explorer.project_path == project_path
    assert [group.name for group in explorer.project.data_groups] == ["dropped workspace"]

    dataset_path = tmp_path / "scan.nxs"
    dataset_path.touch()
    empty_explorer = NfitProjectExplorer(NfitProject())

    assert empty_explorer.load_dropped_paths([dataset_path])
    assert [group.name for group in empty_explorer.project.data_groups] == ["Workspace1"]
    assert empty_explorer.project.data_groups[0].datasets[0].metadata["source_file"] == str(
        dataset_path
    )


def test_project_explorer_prompts_for_unsaved_close_and_quit(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    explorer = NfitProjectExplorer()
    assert [item.name for item in explorer.project.data_groups] == ["Workspace1"]
    assert explorer.has_unsaved_changes is False

    group = explorer.create_data_group()

    assert group.name == "Workspace2"
    assert explorer.has_unsaved_changes is True
    assert explorer.window.windowTitle().endswith("*")

    prompts = []
    monkeypatch.setattr(
        explorer,
        "_confirm_save_before_closing_project",
        lambda: prompts.append("prompt") or False,
    )

    assert explorer.close_project() is False
    assert [item.name for item in explorer.project.data_groups] == [
        "Workspace1",
        "Workspace2",
    ]
    assert explorer.quit_application() is False
    assert prompts == ["prompt", "prompt"]

    closed = []
    monkeypatch.setattr(explorer, "quit_application", lambda: closed.append("quit") or True)
    assert explorer.close_project() is True
    assert closed == ["quit"]

    explorer.project_path = Path("/tmp/opened.nfit")
    explorer.project = NfitProject([group])
    explorer.has_unsaved_changes = True
    monkeypatch.setattr(explorer, "_confirm_save_before_closing_project", lambda: True)

    assert explorer.close_project() is True
    assert [item.name for item in explorer.project.data_groups] == ["Workspace1"]
    assert explorer.has_unsaved_changes is False


def test_saved_plot_details_offer_open_edit_and_script_actions(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
    plot = PlotEntry(
        "scan plot",
        type="mdhisto_line",
        sources=[PlotSourceRef(dataset_id=dataset.id)],
        settings={"x_dim": 0, "channel": "signal"},
    )
    group = DataGroup("Workspace1", datasets=[dataset], plots=[plot])
    explorer = NfitProjectExplorer(NfitProject([group]))

    def find_plot_item(item):
        if explorer._plot_for_item(item) is plot:
            return item
        for index in range(item.childCount()):
            found = find_plot_item(item.child(index))
            if found is not None:
                return found
        return None

    plot_item = find_plot_item(explorer.tree.topLevelItem(0))
    assert plot_item is not None
    explorer.tree.setCurrentItem(plot_item)

    buttons = {
        name: explorer.window.findChild(QtWidgets.QPushButton, object_name)
        for name, object_name in {
            "Open plot": "plot_open_button",
            "Open in data viewer": "plot_edit_button",
            "Copy script": "plot_copy_script_button",
            "Save script...": "plot_save_script_button",
        }.items()
    }
    assert all(button is not None for button in buttons.values())
    assert all(button.text() == name for name, button in buttons.items())
    assert all(button.toolTip().strip() for button in buttons.values())
    assert all(not button.isHidden() for button in buttons.values())
    assert buttons["Open plot"].isEnabled()
    assert buttons["Open in data viewer"].isEnabled()
    assert not buttons["Copy script"].isEnabled()
    assert not buttons["Save script..."].isEnabled()

    explorer.project_path = tmp_path / "plots.nfit"
    explorer._sync_details()
    assert buttons["Copy script"].isEnabled()
    assert buttons["Save script..."].isEnabled()

    assert explorer.copy_plot_script_for_selection()
    copied = QtWidgets.QApplication.clipboard().text()
    assert f"PLOT_ID = {plot.id!r}" in copied
    assert str(explorer.project_path) in copied

    target = tmp_path / "scan_plot.py"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "Python scripts (*.py)")),
    )
    assert explorer.save_plot_script_for_selection()
    assert target.read_text(encoding="utf-8") == copied


def test_project_explorer_saves_grouped_waterfall_sources(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    axes = (
        MDHistoAxis("fixed E", np.array([0.0, 1.0]), "meV", "energy"),
        MDHistoAxis("Q", np.linspace(0.0, 1.0, 5), "1/angstrom", "momentum"),
    )

    def cut(value):
        signal = np.full((1, 4), value, dtype=float)
        return MDHistoData(
            axes,
            signal,
            np.ones_like(signal),
            np.zeros_like(signal, dtype=bool),
            np.ones_like(signal),
        )

    first = DatasetEntry("0.5 meV", cut(1.0))
    second = DatasetEntry("1.4 meV", cut(2.0))
    group = DataGroup("Workspace1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))
    viewer = QtMDHistoSliceViewer(
        [first.data, second.data],
        dataset_names=[first.name, second.name],
    )
    viewer._nfit_dataset_ids = {
        first.name: first.id,
        second.name: second.id,
    }
    viewer.view_mode_combo.setCurrentIndex(1)

    plot = explorer.save_plot_from_viewer(group, viewer)

    assert plot is not None
    assert plot.type == "mdhisto_waterfall"
    assert [source.dataset_id for source in plot.sources] == [
        first.id,
        second.id,
    ]
    assert plot.settings["waterfall_dataset_names"] == [
        first.name,
        second.name,
    ]
    rendered = project_gui.render_project_plot(explorer.project, plot.id)
    assert len(rendered.axes[0]._nfit_waterfall_traces) == 2


def test_stored_plots_snapshot_independent_rebin_bases(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.plot_recipes import plot_entry_from_dict, plot_entry_to_dict
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.plotting_test_data import tiny_mdhisto_data

    dataset = DatasetEntry("scan", tiny_mdhisto_data())
    group = DataGroup("Workspace1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    config["fractional"] = False

    first_basis = (
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    )
    second_basis = (
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, -2.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    )
    for axis, vector in zip(config["axes"][1:], first_basis, strict=True):
        axis["vector"] = vector

    viewer = QtMDHistoSliceViewer(dataset.data, dataset_names=[dataset.name])
    viewer._nfit_dataset_ids = {dataset.name: dataset.id}
    first_plot = explorer.save_plot_from_viewer(group, viewer)
    assert viewer.save_plot_button.text() == "Store plot"
    assert viewer.save_new_plot_button.isHidden()

    for axis, vector in zip(config["axes"][1:], second_basis, strict=True):
        axis["vector"] = vector
    viewer._nfit_editing_plot_id = None
    second_plot = explorer.save_plot_from_viewer(group, viewer)

    key = project_gui.PLOT_SOURCE_REBIN_CONFIGS_KEY
    assert [axis["vector"] for axis in first_plot.settings[key][dataset.id]["axes"][1:]] == list(first_basis)
    assert [axis["vector"] for axis in second_plot.settings[key][dataset.id]["axes"][1:]] == list(second_basis)
    first_views, _ = project_gui._saved_plot_source_views(group, first_plot)
    second_views, _ = project_gui._saved_plot_source_views(group, second_plot)
    assert first_views[0].metadata["rebin"]["vectors"][1:] == list(first_basis)
    assert second_views[0].metadata["rebin"]["vectors"][1:] == list(second_basis)
    assert explorer._plot_for_item(explorer.tree.currentItem()) is second_plot
    loaded_plots = [
        plot_entry_from_dict(json.loads(json.dumps(plot_entry_to_dict(plot))))
        for plot in (first_plot, second_plot)
    ]
    assert [
        axis["vector"]
        for axis in loaded_plots[0].settings[key][dataset.id]["axes"][1:]
    ] == list(first_basis)
    assert [
        axis["vector"]
        for axis in loaded_plots[1].settings[key][dataset.id]["axes"][1:]
    ] == list(second_basis)
    plots_item = explorer.tree.topLevelItem(0).child(4)
    explorer.tree.setCurrentItem(plots_item.child(0))
    editor = explorer.edit_saved_plot_in_viewer()
    assert editor.save_plot_button.text() == "Save plot"
    assert not editor.save_new_plot_button.isHidden()
    assert editor.save_new_plot_button.isEnabled()
    assert editor.save_new_plot_button.toolTip()
    assert editor.data.metadata["rebin"]["vectors"][1:] == list(first_basis)
    assert editor._nfit_plot_rebin_configs[dataset.id]["axes"][1]["vector"] == first_basis[0]
    updated = editor.store_plot()
    assert updated is first_plot
    assert [
        axis["vector"]
        for axis in updated.settings[key][dataset.id]["axes"][1:]
    ] == list(first_basis)
    original_settings = copy.deepcopy(first_plot.settings)
    editor.cmap_combo.setCurrentText("plasma")
    editor.save_new_plot_button.click()
    copied = group.plots[-1]
    assert copied.id != first_plot.id
    assert copied.name != first_plot.name
    assert copied.settings["cmap"] == "plasma"
    assert copied.settings[key] == first_plot.settings[key]
    assert copied.settings[key] is not first_plot.settings[key]
    assert first_plot.settings == original_settings
    assert editor._nfit_editing_plot_id == copied.id
    count = len(group.plots)
    editor.cmap_combo.setCurrentText("magma")
    editor.save_plot_button.click()
    assert len(group.plots) == count
    assert copied.settings["cmap"] == "magma"
    assert first_plot.settings == original_settings
    fresh = editor.open_new_viewer()
    assert fresh.save_plot_button.text() == "Store plot"
    assert fresh.save_new_plot_button.isHidden()
    fresh.window.close()
    editor.window.close()
    viewer.window.close()
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_stored_tiled_composite_plot_keeps_composite_rebin_recipe(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from tests.plotting_test_data import tiny_mdhisto_data

    first = DatasetEntry("first", tiny_mdhisto_data(), kind="mdhisto")
    second = DatasetEntry("second", tiny_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Workspace1", datasets=[first, second])
    config = project_gui.data_group_composite_config(group)
    config.update({"enabled": True, "fractional": False})
    explorer = NfitProjectExplorer(NfitProject([group]))
    viewer = explorer.open_slice_viewer(group, use_composite=True)
    tiled_index = viewer.view_mode_combo.findText("Tiled slices")
    assert tiled_index >= 0
    viewer.view_mode_combo.setCurrentIndex(tiled_index)

    plot = viewer.store_plot()

    assert plot is not None
    assert plot.type == "mdhisto_tiled_slices"
    assert [source.dataset_id for source in plot.sources] == [first.id, second.id]
    recipe = plot.settings[project_gui.PLOT_SOURCE_COMPOSITE_KEY]
    assert recipe["dataset_group_id"] is None
    assert recipe["name"] == "Workspace1 Composite"
    saved_axes = copy.deepcopy(recipe["config"]["axes"])
    saved_views, _ = project_gui._saved_plot_source_views(group, plot)
    config["axes"][0]["lower"] -= 10.0
    views, names = project_gui._saved_plot_source_views(group, plot)
    assert names == ["Workspace1 Composite"]
    assert recipe["config"]["axes"] == saved_axes
    for actual, expected in zip(views[0].axes, saved_views[0].axes, strict=True):
        np.testing.assert_allclose(actual.values, expected.values)
    assert project_gui.render_project_plot(explorer.project, plot.id) is not None
    editor = explorer.edit_saved_plot_in_viewer()
    assert editor.store_plot().settings[project_gui.PLOT_SOURCE_COMPOSITE_KEY][
        "config"
    ]["axes"] == saved_axes
    editor.window.close()
    viewer.window.close()


def test_dataset_group_copy_and_nested_dataset_paste(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    child_dataset = DatasetEntry("child scan", _tiny_mdhisto_data(2.0))
    source_dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
    source = DatasetGroup(
        "Temperature",
        datasets=[source_dataset],
        subgroups=[DatasetGroup("Runs", datasets=[child_dataset])],
    )
    destination = DatasetGroup("Destination")
    group = DataGroup("Workspace1", subgroups=[source, destination])
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer._refresh_tree(select_dataset_group=source, refresh_viewers=False)
    assert "Copy" in explorer.context_menu_action_names(explorer.tree.currentItem())
    explorer.copy_selected()
    explorer._refresh_tree(select_dataset_group=destination, refresh_viewers=False)
    destination_item = explorer.tree.currentItem()
    assert "Paste" in explorer.context_menu_action_names(destination_item)
    explorer.paste_into_selection()

    copied_group = destination.subgroups[0]
    assert copied_group.name == "Temperature1"
    assert copied_group.id != source.id
    assert copied_group.datasets[0].id != source_dataset.id
    assert copied_group.subgroups[0].datasets[0].id != child_dataset.id

    explorer._clipboard = ("datasets", [copy.deepcopy(source_dataset)])
    explorer._refresh_tree(select_dataset_group=destination, refresh_viewers=False)
    assert explorer._can_paste_into_role("dataset_group", None)
    explorer.paste_into_selection()
    assert destination.datasets[-1].name.startswith("scan")


def test_dataset_right_click_preserves_batch_selection_and_actions(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    datasets = [
        DatasetEntry(f"scan {index}", _tiny_mdhisto_data(float(index)))
        for index in range(1, 4)
    ]
    remaining = datasets[2]
    source = DataGroup("Source", datasets=datasets)
    target = DataGroup("Target")
    explorer = NfitProjectExplorer(NfitProject([source, target]))
    explorer.window.show()
    source_items = [
        explorer.tree.topLevelItem(0).child(0).child(index)
        for index in range(3)
    ]
    explorer.tree.setCurrentItem(source_items[1])
    source_items[0].setSelected(True)
    source_items[1].setSelected(True)
    QtWidgets.QApplication.processEvents()

    position = explorer.tree.visualItemRect(source_items[0]).center()
    QtTest.QTest.mousePress(
        explorer.tree.viewport(),
        QtCore.Qt.MouseButton.RightButton,
        pos=position,
    )
    assert set(explorer.tree.selectedItems()) == set(source_items[:2])

    explorer._set_selected_enabled(False)
    assert [dataset.enabled for dataset in datasets] == [False, False, True]

    source_items = [
        explorer.tree.topLevelItem(0).child(0).child(index)
        for index in range(3)
    ]
    explorer.tree.setCurrentItem(source_items[1])
    source_items[0].setSelected(True)
    source_items[1].setSelected(True)
    explorer.copy_selected()
    target_folder = explorer.tree.topLevelItem(1).child(0)
    explorer.tree.setCurrentItem(target_folder)
    explorer.paste_into_selection()
    assert [dataset.name for dataset in target.datasets] == ["scan 1", "scan 2"]

    source_items = [
        explorer.tree.topLevelItem(0).child(0).child(index)
        for index in range(3)
    ]
    explorer.tree.setCurrentItem(source_items[1])
    source_items[0].setSelected(True)
    source_items[1].setSelected(True)
    explorer.delete_selected()
    assert source.datasets == [remaining]


def test_auxiliary_project_windows_standard_close_shortcut(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtGui = pytest.importorskip("PySide6.QtGui")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace1")]))

    progress = project_gui._FitProgressDialog(explorer)
    progress.show()
    assert progress.dialog.isVisible()
    assert progress.close_shortcut.key() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Close
    )
    progress.update_progress(
        {
            "stage": "rebin",
            "iteration": 75,
            "total": 100,
            "datasets_total": 6,
            "datasets_completed": 4,
            "message": "rebinning 6 datasets: 75/100 point contributions",
        }
    )
    assert progress.progress.value() == 75
    assert "Datasets 4 of 6" in progress.status_label.text()
    assert "Points 75 of 100" in progress.status_label.text()
    progress.reset()
    progress.close_shortcut.activated.emit()
    QtWidgets.QApplication.processEvents()
    assert not progress.dialog.isVisible()

    rebin = explorer._make_rebin_progress_callback(
        "Rebinning datasets...",
        aggregate=True,
    )
    rebin_dialog = rebin._nfit_progress_dialog
    batch_bar = rebin_dialog._nfit_batch_progress
    detail_bar = rebin_dialog._nfit_detail_progress
    rebin(
        {
            "stage": "rebin_batch",
            "batch_total": 8,
            "batch_completed": 2,
            "batch_name": "MACS SPEC 5meV 2K",
            "batch_kind": "dataset group",
        }
    )
    rebin(
        {
            "stage": "rebin",
            "iteration": 75,
            "total": 100,
            "message": "rebinning 7 datasets",
        }
    )
    assert batch_bar.isVisible()
    assert batch_bar.maximum() == 8
    assert batch_bar.value() == 2
    assert detail_bar.value() == 75
    assert rebin_dialog._nfit_batch_label.text().startswith(
        "Rebinning 8 dataset groups: 2/8 dataset groups binned (25.0%)"
    )
    assert "elapsed " in rebin_dialog._nfit_batch_label.text()
    assert "remaining" not in rebin_dialog._nfit_batch_label.text()
    assert rebin_dialog._nfit_current_label.text() == (
        "Current dataset group: MACS SPEC 5meV 2K"
    )
    rebin(
        {
            "stage": "rebin",
            "iteration": 80,
            "total": 100,
            "batch_total": 8,
            "batch_completed": 2,
            "batch_name": "MACS SPEC 5meV 2K · Fd-3m",
            "batch_kind": "dataset group",
            "rebin_total": 3,
            "rebin_completed": 1,
            "rebin_name": "Fd-3m",
            "message": "rebinning 7 datasets",
        }
    )
    assert rebin_dialog._nfit_current_label.text() == (
        "Current dataset group: MACS SPEC 5meV 2K · "
        "1/3 rebins completed · Current rebin: Fd-3m"
    )
    assert len(rebin_dialog.findChildren(QtWidgets.QProgressBar)) == 2
    assert rebin_dialog._nfit_label.text().startswith(
        "rebinning 7 datasets (80.0%)"
    )
    assert "elapsed " in rebin_dialog._nfit_label.text()
    assert "remaining" not in rebin_dialog._nfit_label.text()
    progress_layout = rebin_dialog.layout()
    assert progress_layout.indexOf(rebin_dialog._nfit_batch_label) < progress_layout.indexOf(
        batch_bar
    )
    assert progress_layout.indexOf(rebin_dialog._nfit_current_label) < progress_layout.indexOf(
        batch_bar
    )
    assert progress_layout.indexOf(rebin_dialog._nfit_label) > progress_layout.indexOf(
        batch_bar
    )
    assert rebin_dialog.minimumWidth() == rebin_dialog.maximumWidth() == 680
    rebin(
        {
            "stage": "rebin",
            "iteration": 100,
            "total": 100,
            "message": "finishing the current internal stage",
        }
    )
    assert batch_bar.value() == 2
    assert detail_bar.maximum() == 100
    assert detail_bar.value() == 99
    rebin(
        {
            "stage": "rebin_batch",
            "batch_total": 8,
            "batch_completed": 3,
            "batch_name": "MACS SPEC 5meV 2K",
            "batch_kind": "dataset group",
            "batch_item_complete": True,
        }
    )
    assert batch_bar.value() == 3
    assert detail_bar.maximum() == detail_bar.value() == 1
    rebin._nfit_progress_controller.finish("Rebinning complete")
    assert batch_bar.value() == batch_bar.maximum() == 8
    explorer._close_rebin_progress(rebin)

    single_rebin = explorer._make_rebin_progress_callback("Rebinning dataset...")
    single_dialog = single_rebin._nfit_progress_dialog
    single_rebin(
        {
            "stage": "rebin_batch",
            "batch_total": 1,
            "batch_completed": 0,
            "batch_name": "large source",
            "batch_kind": "dataset",
        }
    )
    single_rebin(
        {
            "stage": "mdevent_scan",
            "iteration": 23_301_675,
            "total": 123_690_949,
            "message": "reading MDEvents 23,301,675/123,690,949",
        }
    )
    assert not single_dialog._nfit_batch_progress.isVisible()
    assert not single_dialog._nfit_batch_label.isVisible()
    assert not single_dialog._nfit_current_label.isVisible()
    assert "23,301,675/123,690,949" in single_dialog._nfit_label.text()
    assert "elapsed " in single_dialog._nfit_label.text()
    assert "remaining" not in single_dialog._nfit_label.text()
    single_rebin(
        {
            "stage": "mdevent_events",
            "iteration": 2_968_582_776,
            "total": 5_937_165_552,
            "message": "binning symmetry-expanded MDEvent contributions",
        }
    )
    assert single_dialog._nfit_detail_progress.maximum() == 10_000
    assert single_dialog._nfit_detail_progress.value() == 5_000
    explorer._close_rebin_progress(single_rebin)

    diagnostics = project_gui._FitDiagnosticsPlotWindow(FitTimelineEntry("Fit Result1"), explorer)
    diagnostics.show()
    assert diagnostics.window.isVisible()
    assert diagnostics.close_shortcut.key() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Close
    )
    diagnostics.close_shortcut.activated.emit()
    QtWidgets.QApplication.processEvents()
    assert not diagnostics.window.isVisible()


def test_unloaded_mdevent_viewer_uses_visible_progress_dialog(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry(
        "run 100",
        data=None,
        kind="mdevent",
        data_type="single_crystal_inelastic",
        metadata={"source_file": "/data/events.nxs"},
    )
    explorer = NfitProjectExplorer(NfitProject([DataGroup("Data", datasets=[dataset])]))
    monkeypatch.setattr(project_gui.platform, "system", lambda: "Linux")

    progress = explorer._rebin_progress_callback_for_group(
        explorer.project.data_groups[0],
        use_composite=False,
    )

    assert progress is not None
    dialog = progress._nfit_progress_dialog
    assert dialog.isVisible()
    assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal
    assert dialog.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint
    explorer._close_rebin_progress(progress)
    dialog.deleteLater()
    QtWidgets.QApplication.processEvents()


def test_cache_binnings_file_action_is_project_specific(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace1")]))
    try:
        action = explorer.cache_binnings_action
        assert action.isCheckable()
        assert not action.isChecked()
        assert action.toolTip()
        texts = [item.text() for item in explorer.file_menu.actions()]
        assert texts.index("Cache binnings") < texts.index("Save")

        action.trigger()
        assert explorer.project.settings[project_gui.PROJECT_CACHE_BINNINGS_KEY] is True
        assert explorer.has_unsaved_changes
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_cli_interrupt_handler_maps_sigint_to_qt_exit(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    project_gui._qt_app()
    exits = []

    class FakeApp:
        def exit(self, code=0):
            exits.append(code)

    previous_handler = object()
    installed_handlers = {}

    monkeypatch.setattr(project_gui.signal, "getsignal", lambda signum: previous_handler)
    monkeypatch.setattr(
        project_gui.signal,
        "signal",
        lambda signum, handler: installed_handlers.setdefault(signum, handler),
    )

    timer, restored_handler = project_gui._install_cli_interrupt_handler(FakeApp())
    try:
        assert restored_handler is previous_handler
        installed_handlers[signal.SIGINT](signal.SIGINT, None)
        assert exits == [130]
    finally:
        if timer is not None:
            timer.stop()


def test_project_explorer_run_handles_keyboard_interrupt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    class FakeApp:
        def __init__(self):
            self.exit_codes = []

        def exec(self):
            raise KeyboardInterrupt

        def exit(self, code=0):
            self.exit_codes.append(code)

    class FakeTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    fake_app = FakeApp()
    fake_timer = FakeTimer()
    restored_handlers = []
    explorer = NfitProjectExplorer()
    explorer.app = fake_app
    monkeypatch.setattr(explorer, "show", lambda: explorer)
    monkeypatch.setattr(
        project_gui,
        "_install_cli_interrupt_handler",
        lambda app: (fake_timer, "previous-handler"),
    )
    monkeypatch.setattr(
        project_gui,
        "_restore_cli_interrupt_handler",
        lambda handler: restored_handlers.append(handler),
    )

    assert explorer.run() == 130
    assert fake_app.exit_codes == [130]
    assert fake_timer.stopped is True
    assert restored_handlers == ["previous-handler"]


def test_unsaved_prompt_options(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    real_message_box = QtWidgets.QMessageBox

    class FakeMessageBox:
        class Icon:
            Question = "question"

        class StandardButton:
            Save = "save"
            Cancel = "cancel"

        class ButtonRole:
            DestructiveRole = "destructive"

        clicked_label = "cancel"
        instances = []

        def __init__(self, parent=None):
            self.parent = parent
            self.buttons = []
            self.button_by_label = {}
            self.default_button = None
            self._clicked_button = None
            FakeMessageBox.instances.append(self)

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, text):
            self.text = text

        def addButton(self, label, role=None):
            button = object()
            self.buttons.append(label)
            self.button_by_label[label] = button
            return button

        def setDefaultButton(self, button):
            self.default_button = button

        def exec(self):
            self._clicked_button = self.button_by_label[FakeMessageBox.clicked_label]

        def clickedButton(self):
            return self._clicked_button

    monkeypatch.setattr(QtWidgets, "QMessageBox", FakeMessageBox)

    explorer = NfitProjectExplorer()
    explorer._mark_dirty()

    FakeMessageBox.clicked_label = FakeMessageBox.StandardButton.Cancel
    assert explorer._confirm_save_before_closing_project() is False

    instance = FakeMessageBox.instances[-1]
    assert instance.buttons == [
        FakeMessageBox.StandardButton.Save,
        FakeMessageBox.StandardButton.Cancel,
        "Close without saving",
    ]
    assert instance.default_button is instance.button_by_label[FakeMessageBox.StandardButton.Save]

    FakeMessageBox.clicked_label = "Close without saving"
    assert explorer._confirm_save_before_closing_project() is True

    saved = []
    monkeypatch.setattr(explorer, "save", lambda: saved.append(True) or True)
    FakeMessageBox.clicked_label = FakeMessageBox.StandardButton.Save
    assert explorer._confirm_save_before_closing_project() is True
    assert saved == [True]

    monkeypatch.setattr(QtWidgets, "QMessageBox", real_message_box)
