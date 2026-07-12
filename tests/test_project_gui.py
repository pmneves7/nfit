import json
import signal
import time
from pathlib import Path

import numpy as np
import pytest

import nfit
import nfit.project_gui as project_gui
from nfit.dataset import PointData4D, PointListData
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.project_gui import (
    NfitProject,
    NfitProjectExplorer,
    available_data_types,
    create_mask,
    create_data_group,
    create_model_component,
    copy_dataset_to_group,
    copy_mask_to_dataset,
    data_type_label,
    dataset_details_text,
    dataset_for_slice_viewer,
    dataset_rebin_config,
    default_mask_parameters,
    default_model_config,
    default_model_fit_parameters,
    default_model_global_fit,
    default_model_parameters,
    import_dataset_paths,
    load_project,
    model_config_tooltip,
    mask_parameter_tooltip,
    model_parameter_tooltip,
    next_data_group_name,
    recent_project_paths,
    remember_recent_project,
    forget_missing_recent_projects,
    save_dataset_file,
    save_project,
    set_dataset_data_type,
    set_dataset_source,
)
from nfit.pipeline import (
    DataGroup,
    DatasetEntry,
    DatasetGroup,
    FitTimelineEntry,
    MaskSpec,
    ModelComponentSpec,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MPMS_FILE = DATA_DIR / "MPMS" / "test_MPMS.dat"
HB2A_FILE = DATA_DIR / "HB2A" / "test_powder_diffraction.dat"


def test_project_helpers_name_import_and_round_trip(tmp_path):
    assert nfit.create_data_group is create_data_group
    assert nfit.import_dataset_paths is import_dataset_paths

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
    first.enabled = False
    first.fit_weight = 2.5
    mask = create_mask(first)
    mask.type = "box"
    mask.parameters["center"] = [0.0, 0.0, 0.0, 10.0]
    mask.invert = True
    mask.additive = True
    copy_mask_to_dataset(mask, second)
    copied_group = create_data_group(project, "copied_group")
    copy_dataset_to_group(first, copied_group)
    model = create_model_component(group)
    model.parameters["constant"] = 0.25
    model.config["script_note"] = "fixed"

    assert group.dataset_names == ["scan", "scan1"]
    assert first.kind == "nxs"
    assert first.metadata["source_file"].endswith("scan.nxs")
    assert first.metadata["import_status"] == "pending"

    project_path = tmp_path / "project.nfit"
    save_project(project, project_path)

    payload = json.loads(project_path.read_text(encoding="utf-8"))
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
    assert loaded.data_groups[2].models["Model1"].global_fit["constant"] is True
    assert loaded.data_groups[3].datasets[0].name == "scan"


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
        "Save": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.Save),
        "Save As": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.SaveAs),
        "Close": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.Close),
        "Quit": _standard_shortcut_text(QtGui, QtGui.QKeySequence.StandardKey.Quit),
    }


def test_project_explorer_tree_hierarchy_fonts(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    create_mask(dataset, "Mask1")
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
    mask_item = masks_item.child(0)
    model_item = models_item.child(0)
    initial_item = fits_item.child(0)
    result_item = fits_item.child(1)
    current_item = fits_item.child(2)

    assert workspace_item.font(0).bold()
    assert workspace_item.font(0).underline()
    assert datasets_item.font(0).bold()
    assert models_item.font(0).bold()
    assert fits_item.font(0).bold()
    for item in (dataset_item, masks_item, mask_item, model_item, initial_item, result_item, current_item):
        assert not item.font(0).bold()
        assert not item.font(0).underline()
    for item in (
        workspace_item,
        datasets_item,
        models_item,
        fits_item,
        dataset_item,
        masks_item,
        mask_item,
        model_item,
        initial_item,
        result_item,
        current_item,
    ):
        assert not item.icon(0).isNull()


def test_project_explorer_opens_and_reloads_group_slice_viewer(monkeypatch):
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
        def __init__(self):
            self.closed = False
            self.title = ""

        def close(self):
            self.closed = True

        def setWindowTitle(self, title):
            self.title = title

    class FakeViewer:
        def __init__(self, datasets, *, dataset_names):
            self.datasets = list(datasets)
            self.dataset_names = list(dataset_names)
            self.dataset_combo = FakeCombo(dataset_names)
            self.window = FakeWindow()
            self.shown = False
            self.replaced = False
            created_viewers.append(self)

        def show(self):
            self.shown = True

        def replace_datasets(self, datasets, *, dataset_names, selected_dataset_name=None):
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

    group.add_dataset(DatasetEntry("third", _tiny_mdhisto_data(3.0)))
    viewer.shown = False
    reloaded = explorer.refresh_slice_viewer(group)

    assert reloaded is viewer
    assert len(created_viewers) == 1
    assert not viewer.window.closed
    assert viewer.replaced
    assert not viewer.shown
    assert reloaded.dataset_names == ["first", "second", "third"]
    assert reloaded.dataset_combo.currentText() == "second"


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
        def __init__(self, datasets, *, dataset_names):
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
        action.toolTip()
        for action in explorer.file_menu.actions()
        if not action.isSeparator()
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
    point_explorer = NfitProjectExplorer(NfitProject([DataGroup("Workspace1", datasets=[point_dataset])]))
    point_explorer.tree.setCurrentItem(point_explorer.tree.topLevelItem(0).child(0).child(0))

    assert missing_tooltips(point_explorer.window) == []


def test_project_explorer_adds_edits_and_copies_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
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
    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[True], [False]])
    np.testing.assert_array_equal(viewed.mask, [[True], [True]])


def test_coordinate_range_mask_axis_vectors_define_coordinates():
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 10.0]
    mask.parameters["axis_0"] = [0.0, 1.0]
    mask.parameters["axis_1"] = [1.0, 0.0]

    viewed = dataset_for_slice_viewer(dataset)

    expected = np.array([[False, True], [False, True]])
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

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[True], [False], [False]])

    masking.invert = True
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["nfit_mask"], [[False], [True], [True]])

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


def _he_mdhisto_data():
    # Bin edges chosen so H centers are {0, 1, 2} and E centers are {0, 10}.
    h_axis = MDHistoAxis("H", np.array([-0.5, 0.5, 1.5, 2.5]), "rlu", "h")
    e_axis = MDHistoAxis("E", np.array([-5.0, 5.0, 15.0]), "meV", "energy_transfer")
    shape = (3, 2)
    return MDHistoData(
        axes=(h_axis, e_axis),
        signal=np.zeros(shape),
        errors=np.ones(shape),
        mask=np.zeros(shape, dtype=bool),
        num_events=np.ones(shape),
        metadata={},
    )


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
    assert model.global_fit == default_model_global_fit("constant_background")
    assert explorer.tree.currentItem().text(0) == "Model1"

    fit_group = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_fit_parameters_group")
    config_group = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_config_group")
    model_scroll = explorer.window.findChild(QtWidgets.QScrollArea, "model_parameter_scroll")
    assert fit_group is not None
    assert config_group is not None
    assert model_scroll is not None
    assert model_scroll.widget() is explorer.model_parameter_widget
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
    assert model.global_fit["c0"] is True
    assert model.global_fit["c1"] is True
    assert model.config == default_model_config("linear_background")

    explorer._set_model_parameter("c1", "0.02")
    explorer._set_model_fit_parameter("c1", True)
    explorer._set_model_global_fit("c1", False)

    assert model.parameters["c1"] == 0.02
    assert model.fit_parameters["c1"] is True
    assert model.global_fit["c1"] is False

    tooltip = model_parameter_tooltip("linear_background", "c1")
    editors = explorer.model_parameter_widget.findChildren(QtWidgets.QLineEdit)
    checks = explorer.model_parameter_widget.findChildren(QtWidgets.QCheckBox)
    assert any(editor.toolTip() == tooltip for editor in editors)
    assert any(check.text() == "Fit" for check in checks)
    assert any(check.text() == "Global fit" for check in checks)
    assert "Fit:" in tooltip
    assert "Global fit:" in tooltip

    combo.setCurrentIndex(combo.findData("single_q_paramagnon"))
    assert model.config == {"cross_section": "magnetic"}
    explorer._set_model_config_setting("cross_section", "kinematic")

    assert model.config["cross_section"] == "kinematic"
    config_tooltip = model_config_tooltip("single_q_paramagnon", "cross_section")
    config_group = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_config_group")
    config_editors = config_group.findChildren(QtWidgets.QLineEdit)
    config_checks = config_group.findChildren(QtWidgets.QCheckBox)
    assert any(editor.toolTip() == config_tooltip for editor in config_editors)
    assert config_checks == []
    assert "Configuration settings are fixed model options" in config_tooltip

    plot_label_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit,
        "model_parameter_plot_label_amplitude",
    )
    assert plot_label_editor is not None
    assert plot_label_editor.toolTip()
    plot_label_editor.setText(r"$A$")
    explorer._set_model_parameter_plot_label("amplitude", plot_label_editor.text())
    assert model.metadata["parameter_labels"]["amplitude"] == r"$A$"

    models_item = explorer.tree.topLevelItem(0).child(1)
    explorer.tree.setCurrentItem(models_item)
    assert not explorer.add_model_button.isHidden()


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


def test_spin_model_form_factor_custom_choice_controls_coefficients(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("local_relaxational"))

    form_combo = explorer.model_parameter_widget.findChild(QtWidgets.QComboBox, "model_config_choice_ion")
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

    form_combo = explorer.model_parameter_widget.findChild(QtWidgets.QComboBox, "model_config_choice_ion")
    form_combo.setCurrentIndex(form_combo.findData(project_gui.CUSTOM_FORM_FACTOR_CHOICE))
    coeff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_config_form_factor_coefficients"
    )
    assert coeff_editor is not None
    coeff_editor.setText("0.0263, 34.96, 0.3668, 15.94, 0.6188, 5.594, -0.0119")
    explorer._set_model_config_setting("form_factor_coefficients", coeff_editor.text())
    assert model.config["ion"] == project_gui.CUSTOM_FORM_FACTOR_CHOICE
    assert len(model.config["form_factor_coefficients"]) == 7

    form_combo = explorer.model_parameter_widget.findChild(QtWidgets.QComboBox, "model_config_choice_ion")
    form_combo.setCurrentIndex(form_combo.findData("Mn2"))
    assert model.config["ion"] == "Mn2"
    assert model.config["form_factor_coefficients"] == ""
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_config_form_factor_coefficients"
        )
        is None
    )


def test_project_explorer_fit_history_creates_results_branches_and_restores(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    mask = create_mask(dataset)
    mask.parameters["H"] = [-1.0, 1.0]
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 0.5
    explorer = NfitProjectExplorer(NfitProject([group]))

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    initial_item = fits_item.child(0)
    assert fits_item.text(0) == "Fits"
    assert initial_item.text(0) == "Initial"

    explorer.tree.setCurrentItem(initial_item)
    result = explorer.fit_now_for_selection()

    assert result is not None
    assert result.name == "Fit Result1"
    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result1"]
    assert group.fits[1].goodness["status"] in {"converged", "not converged"}
    assert "first" in group.fits[1].channels

    result.goodness["diagnostics"] = {"gradient_norm": 0.01, "history": list(range(12))}
    result.metadata["fit_pipeline"] = {"version": "test", "notes": ["prepared", "solved"]}
    explorer._set_fit_details(result)

    def current_detail_boxes():
        boxes = []
        seen = set()

        def collect(widget):
            if isinstance(widget, QtWidgets.QGroupBox) and id(widget) not in seen:
                seen.add(id(widget))
                boxes.append(widget)
            for child in widget.findChildren(QtWidgets.QGroupBox):
                if id(child) not in seen:
                    seen.add(id(child))
                    boxes.append(child)

        for index in range(explorer.details_layout.count()):
            widget = explorer.details_layout.itemAt(index).widget()
            if widget is not None:
                collect(widget)
        return boxes

    panel_titles = [box.title() for box in current_detail_boxes()]
    parameter_splitter = explorer.details_widget.findChild(
        QtWidgets.QSplitter, "fit_details_parameter_splitter"
    )
    assert parameter_splitter is not None
    assert "Fit results" in panel_titles
    assert "Goodness of fit" in panel_titles
    assert "Stored fit channels" in panel_titles
    assert "Metadata" in panel_titles

    boxes_by_title = {box.title(): box for box in current_detail_boxes()}
    results_table = boxes_by_title["Fit results"].findChild(QtWidgets.QTableWidget, "fit_results_table")
    goodness_tree = boxes_by_title["Goodness of fit"].findChild(QtWidgets.QTreeWidget, "fit_goodness_tree")
    channels_tree = boxes_by_title["Stored fit channels"].findChild(QtWidgets.QTreeWidget, "fit_channels_tree")
    metadata_tree = boxes_by_title["Metadata"].findChild(QtWidgets.QTreeWidget, "fit_metadata_tree")
    assert results_table is not None
    assert parameter_splitter.widget(0).title() == "Fit results"
    assert goodness_tree is not None
    assert channels_tree is not None
    assert metadata_tree is not None
    assert results_table.columnCount() == 6
    assert results_table.rowCount() >= 1
    assert goodness_tree.maximumHeight() == 260
    goodness_items = {
        goodness_tree.topLevelItem(index).text(0): goodness_tree.topLevelItem(index)
        for index in range(goodness_tree.topLevelItemCount())
    }
    assert goodness_items["diagnostics"].text(1) == "2 field(s)"
    metadata_items = {
        metadata_tree.topLevelItem(index).text(0): metadata_tree.topLevelItem(index)
        for index in range(metadata_tree.topLevelItemCount())
    }
    assert metadata_items["fit_pipeline"].text(1) == "2 field(s)"
    assert channels_tree.topLevelItemCount() >= 1

    dataset.masks[0].parameters["H"] = [0.0, 0.0]
    model.parameters["constant"] = 9.0
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    assert dataset.masks[0].parameters["H"] == [-1.0, 1.0]
    assert model.parameters["constant"] == 0.5

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    restored_result_item = explorer.tree.topLevelItem(0).child(2).child(1)
    assert explorer.tree.currentItem() is model_item
    assert restored_result_item.font(0).italic()
    assert restored_result_item.font(0).bold()
    assert "Active fit state" in restored_result_item.toolTip(0)

    explorer._set_model_parameter("constant", "1.25")

    assert group.fits[1].children == []
    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result1", "Current state"]
    assert group.fits[2].kind == "current"
    assert group.fits[2].snapshot["models"][0]["parameters"]["constant"] == 1.25
    edit_current_item = explorer.tree.topLevelItem(0).child(2).child(2)
    assert explorer._fit_entry_for_item(edit_current_item) is group.fits[2]
    assert explorer.tree.currentItem().text(0) == "Model1"
    assert edit_current_item.font(0).italic()
    assert edit_current_item.font(0).bold()

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(2))
    explorer.fit_branch_check.setChecked(True)
    branched = explorer.fit_now_for_selection()

    assert branched is not None
    explicit_branch = group.fits[2].children[-1]
    assert explicit_branch.kind == "timeline"
    assert explicit_branch.children == [branched]
    assert explorer.tree.currentItem().text(0) == branched.name
    assert explorer.tree.currentItem().font(0).italic()
    assert explorer.tree.currentItem().font(0).bold()
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()


def test_project_explorer_deletes_a_range_of_selected_fits(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    group.fits.extend(FitTimelineEntry(name=f"Fit Result{index}") for index in range(1, 5))
    explorer._refresh_tree(select_group=group)

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    assert fits_item.text(0) == "Fits"
    fits_item.setExpanded(True)
    assert [group_item.child(2).child(i).text(0) for i in range(fits_item.childCount())] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
    ]

    # A shift/ctrl selection suppresses the per-fit state restore (and its tree
    # rebuild) so the growing multi-selection survives.
    monkeypatch.setattr(explorer, "_is_multi_select_gesture", lambda: True)

    # Select a contiguous run of result fits and delete them in one action.
    range_items = [fits_item.child(index) for index in (1, 2, 3)]
    explorer.tree.setCurrentItem(range_items[-1])
    for item in range_items:
        item.setSelected(True)

    explorer.delete_selected()

    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result4"]


def test_project_explorer_range_selection_stays_within_role(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    group.fits.append(FitTimelineEntry(name="Fit Result1"))
    explorer._refresh_tree(select_group=group)

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    fits_item.setExpanded(True)
    fit_item = fits_item.child(1)
    assert fit_item.text(0) == "Fit Result1"

    # Emulate a shift-click whose visual range sweeps in the workspace root and
    # folder headers: the current (clicked) item is a fit, so the non-fit rows
    # must be pruned back out of the selection.
    monkeypatch.setattr(explorer, "_is_multi_select_gesture", lambda: True)
    explorer.tree.setCurrentItem(fit_item)
    fit_item.setSelected(True)
    group_item.setSelected(True)
    fits_item.setSelected(True)

    selected_roles = {
        explorer._objects_for_item(item)[4] for item in explorer.tree.selectedItems()
    }
    assert selected_roles == {"fit"}


def test_fit_now_updates_live_model_parameters_and_editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _tiny_mdhisto_data(3.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    result = explorer.fit_now_for_selection()

    assert result is not None
    assert model.parameters["constant"] == pytest.approx(3.0, abs=1e-6)
    assert result.snapshot["models"][0]["parameters"]["constant"] == pytest.approx(3.0, abs=1e-6)

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    tooltip = model_parameter_tooltip("constant_background", "constant")
    parameter_editor = next(
        editor
        for editor in explorer.model_parameter_widget.findChildren(QtWidgets.QLineEdit)
        if editor.toolTip() == tooltip
    )
    assert float(parameter_editor.text()) == pytest.approx(3.0, abs=1e-6)


def test_write_back_fitted_parameters_includes_dynamic_orbit_parameters():
    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.1, "gamma0": 5.0, "J1": 0.0},
        fit_parameters={"scale": True, "chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0]],
            "orbits": [
                {
                    "label": "J1",
                    "bonds": [{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}],
                }
            ],
        },
    )
    compiled = project_gui.compile_fit_problem(
        [model],
        [project_gui.FitDatasetInput("scan", _points_for_dynamic_writeback(), data_type="single_crystal_inelastic")],
    )
    result = type(
        "Result",
        (),
        {"params": {"rpa.scale": 2.0, "rpa.chi0": 0.2, "rpa.gamma0": 4.0, "rpa.J1": 1.25}},
    )()

    project_gui._write_back_fitted_parameters(DataGroup("Datagroup1"), [model], compiled, result)

    assert model.parameters["scale"] == pytest.approx(2.0)
    assert model.parameters["J1"] == pytest.approx(1.25)


def _points_for_dynamic_writeback() -> PointData4D:
    return PointData4D(
        H=[0.0],
        K=[0.0],
        L=[0.0],
        E=[1.0],
        intensity=[1.0],
        sigma=[1.0],
        temperature=5.0,
    )


def test_slice_viewer_datasets_attach_current_model_before_fit():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 7.0

    datasets, names = project_gui.slice_viewer_datasets(group)

    assert names == ["scan"]
    assert "fit" in datasets[0].metadata
    assert "residual" in datasets[0].metadata
    np.testing.assert_allclose(datasets[0].metadata["fit"], np.full(dataset.data.shape, 7.0))
    np.testing.assert_allclose(datasets[0].metadata["residual"], dataset.data.signal - 7.0)


def test_slice_viewer_datasets_prefer_current_model_over_stored_fit_channels():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 7.0
    group.fits = [
        FitTimelineEntry(
            "Fit Result1",
            kind="result",
            channels={
                "scan": {
                    "fit": np.full(dataset.data.shape, 3.0),
                    "residual": np.full(dataset.data.shape, -2.0),
                }
            },
        )
    ]

    datasets, names = project_gui.slice_viewer_datasets(group)

    assert names == ["scan"]
    np.testing.assert_allclose(datasets[0].metadata["fit"], np.full(dataset.data.shape, 7.0))
    np.testing.assert_allclose(datasets[0].metadata["residual"], dataset.data.signal - 7.0)


def test_project_explorer_fit_pipeline_controls_have_tooltips_and_update_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    controls = [
        explorer.fit_loss_combo,
        explorer.fit_f_scale_spin,
        explorer.fit_de_check,
        explorer.fit_de_maxiter_spin,
        explorer.fit_de_popsize_spin,
        explorer.fit_de_workers_spin,
        explorer.fit_emcee_check,
        explorer.fit_emcee_walkers_spin,
        explorer.fit_emcee_steps_spin,
        explorer.fit_emcee_burn_spin,
        explorer.fit_emcee_thin_spin,
        explorer.fit_emcee_workers_spin,
        explorer.fit_optimizer_config_editor,
        explorer.fit_branch_check,
        explorer.fit_now_button,
        explorer.fit_corner_button,
    ]
    assert all(control.toolTip() for control in controls)
    assert explorer.fit_branch_check.parentWidget() is explorer.fit_editor_widget
    assert explorer.fit_settings_panel.parentWidget() is explorer.details_widget
    assert explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_optimizer_settings_group") is not None
    assert explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_de_settings_group") is not None
    assert explorer.details_widget.findChild(QtWidgets.QGroupBox, "fit_posterior_settings_group") is not None

    explorer.fit_loss_combo.setCurrentText("soft_l1")
    explorer.fit_f_scale_spin.setValue(2.0)
    explorer.fit_de_check.setChecked(True)
    explorer.fit_de_maxiter_spin.setValue(11)
    explorer.fit_de_popsize_spin.setValue(4)
    explorer.fit_de_workers_spin.setValue(-1)
    explorer.fit_emcee_check.setChecked(True)
    explorer.fit_emcee_walkers_spin.setValue(16)
    explorer.fit_emcee_steps_spin.setValue(25)
    explorer.fit_emcee_burn_spin.setValue(5)
    explorer.fit_emcee_thin_spin.setValue(2)
    explorer.fit_emcee_workers_spin.setValue(3)

    fit_entry = group.fits[0]
    assert fit_entry.optimizer_config["loss"] == "soft_l1"
    assert fit_entry.optimizer_config["f_scale"] == 2.0
    assert fit_entry.optimizer_config["initialization"] == {
        "enabled": True,
        "method": "differential_evolution",
        "maxiter": 11,
        "popsize": 4,
        "workers": -1,
    }
    assert fit_entry.optimizer_config["sampler"] == {
        "enabled": True,
        "method": "emcee",
        "n_walkers": 16,
        "n_steps": 25,
        "burn_in": 5,
        "thin": 2,
        "workers": 3,
    }


def test_sampling_result_serializes_raw_chain_and_rewindows():
    result = project_gui.SamplingResult(
        samples=np.zeros((1, 1), dtype=float),
        variable_names=["level"],
        log_probability=np.zeros(1, dtype=float),
        metadata={"method": "emcee", "n_steps": 4, "n_walkers": 2, "burn_in": 0, "thin": 1},
        chain=np.arange(8, dtype=float).reshape(4, 2, 1),
        log_probability_chain=np.arange(8, dtype=float).reshape(4, 2),
    )

    restored = project_gui._sampling_result_from_dict(project_gui._sampling_result_to_dict(result))
    assert restored is not None
    assert restored.chain is not None
    np.testing.assert_allclose(restored.chain, result.chain)

    rewindowed = project_gui._sampling_result_with_window(restored, burn_in=1, thin=2)
    np.testing.assert_allclose(rewindowed.samples[:, 0], [2.0, 3.0, 6.0, 7.0])
    assert rewindowed.metadata["burn_in"] == 1
    assert rewindowed.metadata["thin"] == 2
    assert rewindowed.metadata["samples"] == 4


def test_fit_pipeline_saves_partial_emcee_when_sampler_is_cancelled(monkeypatch):
    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 0.0
    model.fit_parameters["constant"] = True
    chain = np.arange(4, dtype=float).reshape(2, 2, 1)
    partial = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        metadata={
            "method": "emcee",
            "n_walkers": 2,
            "n_steps": 2,
            "requested_n_steps": 10,
            "burn_in": 0,
            "thin": 1,
            "cancelled": True,
            "completed": False,
        },
        chain=chain,
        log_probability_chain=np.zeros((2, 2), dtype=float),
    )

    def cancel_sampling(*_args, **_kwargs):
        raise project_gui.SamplingCancelled("cancelled", partial)

    monkeypatch.setattr(project_gui, "sample_problem_parameters", cancel_sampling)

    outcome = project_gui.perform_group_fit(
        group,
        optimizer_config={"sampler": {"enabled": True, "n_walkers": 2, "n_steps": 10}},
    )

    assert outcome["goodness"]["status"] == "cancelled"
    assert "partial posterior" in outcome["goodness"]["message"]
    stored = project_gui._sampling_result_from_dict(outcome["metadata"].get("posterior_samples"))
    assert stored is not None
    assert stored.metadata["cancelled"] is True
    assert stored.chain is not None
    np.testing.assert_allclose(stored.chain, chain)


def test_fit_details_posterior_sampler_controls_update_burn_without_timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    chain = np.arange(8, dtype=float).reshape(4, 2, 1)
    sampling = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        metadata={"method": "emcee", "n_steps": 4, "n_walkers": 2, "burn_in": 0, "thin": 1},
        chain=chain,
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(sampling)},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result]
    explorer = NfitProjectExplorer(NfitProject([group]))

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    controls = [
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_walkers_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_steps_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_burn_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_thin_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_seed_spin"),
        explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_workers_spin"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_apply_button"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_rerun_button"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_append_button"),
        explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_promote_button"),
    ]
    assert all(control is not None and control.toolTip() for control in controls)

    burn_spin = explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_burn_spin")
    thin_spin = explorer.window.findChild(QtWidgets.QSpinBox, "fit_posterior_thin_spin")
    burn_spin.setValue(1)
    thin_spin.setValue(2)
    before_count = len(group.fits)

    assert explorer.apply_posterior_sampling_window(result, burn_spin.value(), thin_spin.value())

    assert len(group.fits) == before_count
    posterior = result.goodness["posterior"]
    assert posterior["burn_in"] == 1
    assert posterior["thin"] == 2
    assert posterior["samples"] == 4


def test_promote_best_posterior_sample_creates_current_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(2.0))
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    model.fit_parameters["constant"] = True
    parameter_name = f"{model.name}.constant"
    chain = np.array([[[1.0]], [[2.0]]], dtype=float)
    log_probability_chain = np.array([[-0.5], [0.0]], dtype=float)
    sampling = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=[parameter_name],
        log_probability=log_probability_chain.reshape(-1),
        metadata={"method": "emcee", "n_steps": 2, "n_walkers": 1, "burn_in": 0, "thin": 1},
        chain=chain,
        log_probability_chain=log_probability_chain,
    )
    result = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"chi2": 1.0, "parameters": {parameter_name: 1.0}},
        metadata={"posterior_samples": project_gui._sampling_result_to_dict(sampling)},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result]
    explorer = NfitProjectExplorer(NfitProject([group]))

    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)
    promote_button = explorer.window.findChild(QtWidgets.QPushButton, "fit_posterior_promote_button")

    assert promote_button is not None
    assert promote_button.isEnabled()
    assert explorer.promote_best_posterior_sample_for_fit(group, result)

    assert model.parameters["constant"] == pytest.approx(2.0)
    assert [fit.kind for fit in group.fits] == ["result", "current"]
    current = group.fits[1]
    assert current.snapshot["models"][0]["parameters"]["constant"] == pytest.approx(2.0)
    assert current.metadata["promoted_posterior_sample"]["source_fit"] == "Fit Result1"
    assert current.metadata["promoted_posterior_sample"]["step"] == 1
    assert explorer._active_fit_entry(group) is current


def test_posterior_corner_density_panel_draws_contours():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_corner_density_panel

    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 1.0, 1500)
    y = 0.7 * x + rng.normal(0.0, 0.35, 1500)
    fig = Figure()
    ax = fig.subplots()

    _draw_corner_density_panel(ax, x, y)

    assert ax.images
    assert ax.collections


def test_fit_diagnostics_detects_covariance_without_posterior_samples():
    from nfit.project_gui import _covariance_matrix_from_fit_entry, _fit_entry_has_diagnostic_plots

    entry = FitTimelineEntry(
        "Fit Result1",
        goodness={
            "covariance": {
                "variables": ["a", "b"],
                "matrix": [[4.0, -1.0], [-1.0, 9.0]],
            }
        },
    )

    matrix, names, title = _covariance_matrix_from_fit_entry(entry)
    np.testing.assert_allclose(matrix, [[4.0, -1.0], [-1.0, 9.0]])
    assert names == ["a", "b"]
    assert title == "Covariance"
    assert _fit_entry_has_diagnostic_plots(entry)


def test_fit_diagnostics_matrix_heatmap_draws_image():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_matrix_heatmap

    fig = Figure()
    ax = fig.subplots()

    _draw_matrix_heatmap(
        ax,
        [[1.0, 0.5], [0.5, 2.0]],
        ["a", "b"],
        labels={"a": "$A$", "b": "p2"},
        title="Covariance",
    )

    assert ax.images
    assert ax.get_title() == "Covariance"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["$A$", "p2"]


def test_fit_diagnostics_matrix_heatmap_is_centered_in_figure():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_centered_matrix_heatmap

    fig = Figure(figsize=(8, 6))
    ax = _draw_centered_matrix_heatmap(
        fig,
        [[1.0, 0.2], [0.2, 3.0]],
        ["a", "b"],
        labels={"a": "p1", "b": "p2"},
        title="Covariance",
    )

    bbox = ax.get_position()
    assert abs((bbox.x0 + bbox.x1) / 2.0 - 0.5) < 0.08
    assert abs((bbox.y0 + bbox.y1) / 2.0 - 0.5) < 0.08


def test_corner_histogram_panel_draws_step_histogram_and_reference_lines():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_corner_histogram_panel

    fig = Figure()
    ax = fig.subplots()

    _draw_corner_histogram_panel(
        ax,
        np.linspace(0.0, 1.0, 200),
        "amplitude",
        {"best": 0.5, "low": 0.4, "high": 0.7},
    )
    ax.set_ylabel("Count")

    assert ax.patches
    assert not ax.patches[0].get_fill()
    assert len(ax.lines) >= 3
    assert r"\pm" in ax.get_title()
    assert ax.get_ylabel() == "Count"


def test_corner_reference_lines_are_solid_only():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_corner_reference_lines

    fig = Figure()
    ax = fig.subplots()

    _draw_corner_reference_lines(
        ax,
        {"best": 1.0, "low": 0.5, "high": 1.5},
        {"best": 2.0, "low": 1.5, "high": 2.5},
    )

    assert ax.lines
    assert all(line.get_linestyle() != ":" for line in ax.lines)


def test_trace_panel_draws_walkers_and_burn_in_marker():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from nfit.project_gui import _draw_trace_panel

    chain = np.stack(
        [
            np.column_stack([np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6)]),
            np.column_stack([np.linspace(0.5, 1.5, 6), np.linspace(1.5, 2.5, 6)]),
        ],
        axis=1,
    )
    result = project_gui.SamplingResult(
        samples=chain.reshape(-1, 2),
        variable_names=["a", "b"],
        metadata={"burn_in": 2},
        chain=chain,
    )
    fig = Figure()
    ax = fig.subplots()

    _draw_trace_panel(ax, result, parameter_index=0, summary={"best": 0.75})

    assert len(ax.lines) == 4
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    assert [text.get_text() for text in ax.texts] == ["burn-in"]


def test_fit_diagnostics_label_table_updates_plot_labels(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import _FitDiagnosticsPlotWindow

    entry = FitTimelineEntry(
        "Fit Result1",
        goodness={
            "parameters": {"a": 1.0, "b": 2.0},
            "covariance": {
                "variables": ["a", "b"],
                "matrix": [[1.0, 0.1], [0.1, 2.0]],
            },
        },
        metadata={
            "posterior_samples": {
                "variable_names": ["a", "b"],
                "samples": project_gui._encode_float_array(
                    np.column_stack(
                        [
                            np.linspace(0.8, 1.2, 64),
                            np.linspace(1.8, 2.2, 64),
                        ]
                    )
                ),
            }
        },
    )
    explorer = NfitProjectExplorer(NfitProject())

    window = _FitDiagnosticsPlotWindow(entry, explorer)
    table = window.window.findChild(QtWidgets.QTableWidget, "fit_diagnostics_label_table")

    assert table is not None
    assert table.item(0, 1).text() == "p1"
    corner_index = [window.tabs.tabText(index) for index in range(window.tabs.count())].index("Corner")
    window.tabs.setCurrentIndex(corner_index)
    table.item(0, 1).setText(r"$\Gamma$")

    assert entry.metadata["parameter_labels"]["a"] == r"$\Gamma$"
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Corner"


def test_fit_diagnostics_trace_uses_chain_steps_when_available(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import _FitDiagnosticsPlotWindow

    chain = np.arange(24, dtype=float).reshape(6, 2, 2)
    entry = FitTimelineEntry(
        "Fit Result1",
        metadata={
            "posterior_samples": project_gui._sampling_result_to_dict(
                project_gui.SamplingResult(
                    samples=chain.reshape(-1, 2),
                    variable_names=["a", "b"],
                    metadata={"burn_in": 2},
                    chain=chain,
                )
            )
        },
    )
    explorer = NfitProjectExplorer(NfitProject())

    window = _FitDiagnosticsPlotWindow(entry, explorer)
    trace_index = [window.tabs.tabText(index) for index in range(window.tabs.count())].index("Trace")
    canvas = window.tabs.widget(trace_index)

    assert canvas.figure.axes[-1].get_xlabel() == "MCMC step"


def test_fit_progress_dialog_uses_parameter_table_and_resets(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    from nfit.project_gui import _FitProgressDialog

    explorer = NfitProjectExplorer(NfitProject())
    dialog = _FitProgressDialog(explorer)

    dialog.update_progress(
        {
            "stage": "least_squares",
            "iteration": 10,
            "cost": 12.5,
            "seconds_per_step": 0.125,
            "parameters": {
                "model.constant[a]": 1.25,
                "model.constant[b]": 2.5,
            },
        }
    )

    table = dialog.dialog.findChild(QtWidgets.QTableWidget, "fit_progress_parameter_table")
    splitter = dialog.dialog.findChild(QtWidgets.QSplitter, "fit_progress_panel_splitter")
    log = dialog.dialog.findChild(QtWidgets.QPlainTextEdit, "fit_progress_log")
    assert table is not None
    assert splitter is not None
    assert log is dialog.log
    assert splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert splitter.indexOf(table) == 0
    assert splitter.indexOf(log) == 1
    assert table.rowCount() == 2
    assert "model.constant" not in dialog.log.toPlainText()
    assert "Least-squares fit" in dialog.stage_label.text()
    assert "125 ms/step" in dialog.status_label.text()
    assert "125 ms/step" in dialog.log.toPlainText()

    dialog.reset()

    assert table.rowCount() == 0
    assert dialog.log.toPlainText() == ""

    summary = project_gui._progress_event_summary(
        {
            "stage": "emcee",
            "iteration": 3,
            "elapsed_seconds": 1.5,
            "seconds_per_step": 0.5,
        }
    )
    assert summary["elapsed_seconds"] == 1.5
    assert summary["seconds_per_step"] == 0.5


def test_project_explorer_reuses_fit_progress_dialog(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fake_run_group_fit(group, parent, *, branch_timeline=False, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "least_squares",
                    "iteration": 1,
                    "parameters": {"model.constant": 1.0},
                }
            )
        return parent

    monkeypatch.setattr(project_gui, "run_group_fit", fake_run_group_fit)
    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    explorer.fit_now_for_selection()
    first_dialog = explorer._fit_progress_dialog
    assert first_dialog is not None
    assert first_dialog.dialog.isVisible()
    assert first_dialog.close_button.isEnabled()
    explorer.fit_now_for_selection()

    assert explorer._fit_progress_dialog is first_dialog
    assert first_dialog.dialog.isVisible()


def test_project_explorer_start_fit_runs_in_background_worker(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))
    calls = []

    def fake_run_group_fit(group, parent, *, branch_timeline=False, progress_callback=None):
        calls.append("run")
        if progress_callback is not None:
            progress_callback({"stage": "least_squares", "iteration": 1, "total": 1})
        return project_gui.create_fit_result_entry(
            group,
            parent,
            branch_timeline=branch_timeline,
            goodness={"status": "converged", "parameters": {}},
        )

    monkeypatch.setattr(project_gui, "run_group_fit", fake_run_group_fit)
    fit_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(fit_item)

    assert explorer.start_fit_for_selection()
    assert explorer._fit_worker_thread is not None

    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    assert calls == ["run"]
    assert explorer._fit_worker_thread is None
    assert explorer._fit_progress_dialog is not None
    assert explorer._fit_progress_dialog.dialog.isVisible()
    assert explorer._fit_progress_dialog.close_button.isEnabled()


def test_background_posterior_cancel_saves_partial_chain_and_reenables_gui(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    model.parameters["constant"] = 1.0
    result_entry = FitTimelineEntry(
        "Fit Result1",
        kind="result",
        goodness={"parameters": {"model1.constant": 1.0}},
        snapshot=project_gui.snapshot_data_group_state(group),
    )
    group.fits = [result_entry]
    explorer = NfitProjectExplorer(NfitProject([group]))
    chain = np.arange(6, dtype=float).reshape(3, 2, 1)
    partial = project_gui.SamplingResult(
        samples=chain.reshape(-1, 1),
        variable_names=["model1.constant"],
        log_probability=np.zeros(6, dtype=float),
        metadata={
            "method": "emcee",
            "n_walkers": 2,
            "n_steps": 3,
            "requested_n_steps": 20,
            "burn_in": 0,
            "thin": 1,
            "cancelled": True,
            "completed": False,
        },
        chain=chain,
        log_probability_chain=np.zeros((3, 2), dtype=float),
    )

    def cancel_sampler(_group, _fit_entry, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback({"stage": "emcee", "iteration": 3, "total": 20})
        raise project_gui.SamplingCancelled("cancelled", partial)

    monkeypatch.setattr(explorer, "_posterior_sampler_result_for_fit", cancel_sampler)

    assert explorer.start_posterior_sampler_for_fit(
        group,
        result_entry,
        n_walkers=2,
        n_steps=20,
        burn_in=0,
        thin=1,
        random_seed=None,
        workers=1,
    )
    assert not explorer.tree.isEnabled()
    assert explorer._fit_worker_thread is not None

    deadline = time.monotonic() + 3.0
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()

    assert explorer._fit_worker_thread is None
    assert explorer.tree.isEnabled()
    assert explorer.details_scroll.isEnabled()
    stored = project_gui._sampling_result_from_dict(result_entry.metadata.get("posterior_samples"))
    assert stored is not None
    assert stored.metadata["cancelled"] is True
    assert stored.metadata["n_steps"] == 3
    assert stored.chain is not None
    np.testing.assert_allclose(stored.chain, chain)
    assert result_entry.goodness["posterior"]["samples"] == 6
    assert explorer._fit_progress_dialog is not None
    assert explorer._fit_progress_dialog.close_button.isEnabled()


def test_project_explorer_edits_initial_state_in_place_without_results(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    model = create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    initial_item = explorer.tree.topLevelItem(0).child(2).child(0)
    explorer.tree.setCurrentItem(initial_item)
    assert [fit.name for fit in group.fits] == ["Initial"]

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    explorer._set_model_fit_parameter("constant", True)
    explorer._set_model_global_fit("constant", False)

    assert model.fit_parameters["constant"] is True
    assert model.global_fit["constant"] is False
    assert [fit.name for fit in group.fits] == ["Initial"]
    assert group.fits[0].children == []
    snapshot_model = group.fits[0].snapshot["models"][0]
    assert snapshot_model["fit_parameters"]["constant"] is True
    assert snapshot_model["global_fit"]["constant"] is False


def test_project_explorer_fit_now_from_earlier_result_creates_nested_timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = NfitProjectExplorer(NfitProject([group]))

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    explorer.fit_now_for_selection()
    for _index in range(4):
        fits_item = explorer.tree.topLevelItem(0).child(2)
        explorer.tree.setCurrentItem(fits_item.child(fits_item.childCount() - 1))
        explorer.fit_now_for_selection()

    assert [fit.name for fit in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
        "Fit Result5",
    ]

    fit_result3_item = explorer.tree.topLevelItem(0).child(2).child(3)
    explorer.tree.setCurrentItem(fit_result3_item)
    branched = explorer.fit_now_for_selection()

    assert branched is not None
    assert [fit.name for fit in group.fits] == [
        "Initial",
        "Fit Result1",
        "Fit Result2",
        "Fit Result3",
        "Fit Result4",
        "Fit Result5",
    ]
    assert group.fits[3].children[0].kind == "timeline"
    assert group.fits[3].children[0].children[0] is branched
    assert group.fits[3].children[0].children == [branched]
    assert explorer.tree.currentItem().text(0) == branched.name
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    explorer._set_model_parameter("constant", "2.5")

    timeline = group.fits[3].children[0]
    assert [entry.kind for entry in timeline.children] == ["result", "current"]
    assert timeline.children[1].snapshot["models"][0]["parameters"]["constant"] == 2.5
    assert timeline.children[0].children == []


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
        "View in data viewer",
        "Show file location",
        "Change file source",
        "Add mask",
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
    assert explorer.context_menu_action_names(explorer.tree.currentItem()) == ["Enable", "Rename", "Delete"]

    models_item = explorer.tree.topLevelItem(0).child(1)
    assert explorer.context_menu_action_names(models_item) == ["Add model"]
    assert model.name in group.models


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


def test_project_explorer_prompts_for_unsaved_close_and_quit(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    explorer = NfitProjectExplorer()
    group = explorer.create_data_group()

    assert group.name == "Workspace1"
    assert explorer.has_unsaved_changes is True
    assert explorer.window.windowTitle().endswith("*")

    prompts = []
    monkeypatch.setattr(
        explorer,
        "_confirm_save_before_closing_project",
        lambda: prompts.append("prompt") or False,
    )

    assert explorer.close_project() is False
    assert explorer.project.data_groups == [group]
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
    assert explorer.project.data_groups == []
    assert explorer.has_unsaved_changes is False


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
    progress.close_shortcut.activated.emit()
    QtWidgets.QApplication.processEvents()
    assert not progress.dialog.isVisible()

    diagnostics = project_gui._FitDiagnosticsPlotWindow(FitTimelineEntry("Fit Result1"), explorer)
    diagnostics.show()
    assert diagnostics.window.isVisible()
    assert diagnostics.close_shortcut.key() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Close
    )
    diagnostics.close_shortcut.activated.emit()
    QtWidgets.QApplication.processEvents()
    assert not diagnostics.window.isVisible()


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


def test_dataset_details_text_summarizes_axes_source_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    source = tmp_path / "scan.nxs"
    source.write_bytes(b"12345")
    data = _tiny_mdhisto_data(1.0)
    data.metadata.update(
        {
            "temperature": 12.5,
            "field": "7 T",
            "sample_environment": {"temperature": 12.5, "field": "7 T", "log": list(range(12))},
            "oriented_lattice": {"orientation_matrix": np.eye(3).tolist()},
        }
    )
    dataset = DatasetEntry(
        "scan",
        data,
        kind="nxs",
        metadata={"source_file": str(source), "timestamp": "2026-01-02T03:04:05"},
        parameters={"sample": "NiO"},
    )
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={"a": 4.17, "b": 4.17, "c": 4.17, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    text = explorer.details_label.text()
    panel_titles = [box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)]

    assert panel_titles == [
        "Dataset",
        "Sample environment",
        "Axes",
        "Rebin",
        "Crystal",
        "Data",
        "Source",
        "Metadata",
    ]
    assert "Axes\nDimensions: 2" in text
    assert "Crystal\nLattice parameters" in text
    assert "a: 4.17" in text
    assert "Oriented lattice orientation matrix" in text
    assert "0: H (rlu) - 1 bins" in text
    assert "1: E (meV) - 1 bins" in text
    assert "File size on disk: 5 B" in text
    assert text.index("Axes") < text.index("Metadata")
    assert "temperature: 12.5" in text
    assert "field: 7 T" in text
    assert "timestamp: 2026-01-02T03:04:05" in text
    assert "sample: NiO" in text

    metadata_tree = explorer.details_widget.findChild(QtWidgets.QTreeWidget, "dataset_metadata_tree")
    assert metadata_tree is not None
    assert metadata_tree.maximumHeight() == 260
    assert metadata_tree.headerItem().text(0) == "Field"
    assert metadata_tree.headerItem().text(1) == "Value"
    top_level = {
        metadata_tree.topLevelItem(index).text(0): metadata_tree.topLevelItem(index)
        for index in range(metadata_tree.topLevelItemCount())
    }
    assert top_level["sample_environment"].text(1) == "3 field(s)"
    child_names = {
        top_level["sample_environment"].child(index).text(0)
        for index in range(top_level["sample_environment"].childCount())
    }
    assert {"field", "temperature", "log"}.issubset(child_names)


def test_dataset_details_fit_bins_include_file_dataset_and_group_masks():
    data = _grid_mdhisto_data()
    data.mask[0, 0] = True
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    mask = create_mask(dataset, type="box")
    mask.parameters["axes"] = ["H", "E"]
    mask.parameters["center"] = [1.5, 5.0]
    mask.parameters["width"] = [0.1, 0.1]
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        masks=[
            MaskSpec(
                "SharedMask",
                type="box",
                parameters={
                    "axes": ["H", "E"],
                    "center": [0.5, 15.0],
                    "width": [0.1, 0.1],
                },
            )
        ],
    )

    text = dataset_details_text(dataset, group=group)

    assert "Total bins: 4" in text
    assert "Unmasked bins: 3" in text
    assert "Fit bins: 1 of 4" in text
    viewed = dataset_for_slice_viewer(dataset, extra_masks=project_gui.effective_dataset_masks(group, dataset))
    assert int(np.count_nonzero(~viewed.mask)) == 1


def test_dataset_rebin_config_updates_slice_viewer_materializes_and_saves(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto", parameters={"temperature": 12.5})
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    config = dataset_rebin_config(dataset)
    assert config["axes"][0]["num_bins"] == 2
    assert config["axes"][1]["num_bins"] == 2
    assert config["fractional"] is True
    assert config["max_batch_mb"] == 192
    assert config["auto_rebin"] is True

    enable_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_check is not None
    rebin_panel = next(
        box for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox) if box.title() == "Rebin"
    )
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fractional").isChecked()
    auto_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_auto")
    assert auto_check is not None
    assert auto_check.isChecked()
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fit_enabled") is None
    mean_combo = rebin_panel.findChild(QtWidgets.QComboBox, "dataset_rebin_mean_weighting")
    assert mean_combo is not None
    assert mean_combo.currentData() == "inverse_variance"
    batch_spin = rebin_panel.findChild(QtWidgets.QSpinBox, "dataset_rebin_max_batch_mb")
    assert batch_spin is not None
    assert batch_spin.value() == 192
    assert "not a cap on total rebinner memory use" in batch_spin.toolTip()
    batch_spin.setValue(64)
    assert dataset_rebin_config(dataset)["max_batch_mb"] == 64
    create_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_create")
    rebin_now_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_now")
    save_rebin_button = rebin_panel.findChild(QtWidgets.QPushButton, "dataset_rebin_save")
    assert create_button is not None
    assert rebin_now_button is not None
    assert rebin_now_button.text() == "Rebin now"
    assert create_button.text() == "Create dataset from rebin"
    assert save_rebin_button is not None
    assert save_rebin_button.text() == "Save rebin to disk"
    enable_check.setChecked(True)
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "num_bins", "1")
    explorer._set_dataset_rebin_axis_value(dataset, group, 1, "num_bins", "1")
    assert dataset_rebin_config(dataset)["axes"][0]["step_size"] == 2.0
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "step_size", "0.75")
    axis_config = dataset_rebin_config(dataset)["axes"][0]
    assert axis_config["num_bins"] == 3
    assert axis_config["step_size"] == pytest.approx(2.0 / 3.0)

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    assert viewed.shape == (3, 1)
    assert viewed.metadata["rebin"]["normalize"] is True
    assert viewed.metadata["combined_mask_count"] == 0

    dialog_save_path = tmp_path / "dialog-rebinned.npz"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(dialog_save_path), "NumPy archives (*.npz)"),
    )
    assert explorer.save_rebin_for_selection()
    dialog_saved = np.load(dialog_save_path)
    assert dialog_saved["signal"].shape == (3, 1)
    assert int(dialog_saved["axis_count"]) == 2

    rebinned = explorer.materialize_rebin_for_selection()

    assert rebinned is not None
    assert group.dataset_names == ["scan", "scan rebinned"]
    assert isinstance(rebinned.data, MDHistoData)
    assert rebinned.data.shape == (3, 1)
    assert rebinned.parameters["temperature"] == pytest.approx(12.5)

    save_path = tmp_path / "rebinned.npz"
    save_dataset_file(dataset, save_path)

    saved = np.load(save_path)
    assert saved["signal"].shape == (3, 1)
    assert int(saved["axis_count"]) == 2


def test_large_dataset_rebin_defaults_manual_and_defers_refresh(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setattr(project_gui, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)
    project_gui._VIEWER_VIEW_CACHE.clear()

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    config = dataset_rebin_config(dataset)
    assert config["auto_rebin"] is False
    rebin_panel = next(
        box for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox) if box.title() == "Rebin"
    )
    auto_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_auto")
    status = rebin_panel.findChild(QtWidgets.QLabel, "dataset_rebin_status")
    assert auto_check is not None and not auto_check.isChecked()
    assert status is not None
    assert "manual" in status.text().lower() or "disabled" in status.text().lower()

    enable_check = rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_check is not None
    enable_check.setChecked(True)
    refreshes = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda group: refreshes.append(group))
    explorer._set_dataset_rebin_axis_value(dataset, group, 0, "num_bins", "1")

    assert refreshes == []
    assert dataset_rebin_config(dataset)["stale"] is True
    deferred = project_gui._viewer_data_before_scale(dataset, force_rebin=False)
    assert deferred is not None
    assert deferred.shape == data.shape
    assert dataset_rebin_config(dataset)["stale"] is True

    assert explorer.rebin_now_for_selection()
    assert dataset_rebin_config(dataset)["stale"] is False
    forced = dataset_for_slice_viewer(dataset)
    assert forced is not None
    assert forced.shape[0] == 1


def test_data_group_composite_uses_scale_fit_weight_and_rebinning():
    first = DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic")
    second = DatasetEntry(
        "second",
        _tiny_mdhisto_data(5.0),
        kind="mdhisto",
        data_type="single_crystal_inelastic",
        scale_factor=-1.0,
        fit_weight=3.0,
    )
    group = DataGroup("Datagroup1", datasets=[first, second])
    config = project_gui.data_group_composite_config(group)
    config["enabled"] = True
    config["fractional"] = False

    composite = project_gui.composite_dataset_data(group)

    assert isinstance(composite, MDHistoData)
    np.testing.assert_allclose(composite.signal, [[-3.5]])
    np.testing.assert_allclose(composite.errors, [[0.5]])
    np.testing.assert_allclose(composite.num_events, [[2.0]])
    assert composite.metadata["rebin"]["weighted_by_fit_weight"] is True

    datasets, names = project_gui.slice_viewer_datasets(group)
    assert names == ["Datagroup1 Composite"]
    np.testing.assert_allclose(datasets[0].signal, [[-3.5]])

    constituent_datasets, constituent_names = project_gui.slice_viewer_datasets(group, use_composite=False)
    assert constituent_names == ["first", "second"]
    np.testing.assert_allclose(constituent_datasets[1].signal, [[-5.0]])

    inputs, bundles = project_gui.fit_dataset_inputs(group)
    assert [item.name for item in inputs] == ["Datagroup1 Composite"]
    assert list(bundles) == ["Datagroup1 Composite"]


def test_data_group_composite_controls_show_summary_and_update_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic")
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto", data_type="single_crystal_inelastic")
    second.scale_factor = -1.0
    second.fit_weight = 2.0
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

    text = explorer.details_label.text()
    assert "Datasets: 2" in text
    checkbox = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert checkbox is not None
    assert checkbox.toolTip()
    batch_spin = explorer.details_widget.findChild(QtWidgets.QSpinBox, "group_composite_max_batch_mb")
    assert batch_spin is not None
    assert batch_spin.value() == 192
    auto_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_auto")
    assert auto_check is not None
    assert auto_check.isChecked()
    rebin_now_button = explorer.details_widget.findChild(QtWidgets.QPushButton, "group_composite_rebin_now")
    assert rebin_now_button is not None
    assert rebin_now_button.toolTip()

    checkbox.setChecked(True)

    config = project_gui.data_group_composite_config(group)
    assert config["enabled"] is True
    assert config["mean_weighting"] == "inverse_variance"


def test_large_data_group_composite_defaults_manual_and_defers_refresh(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    monkeypatch.setattr(project_gui, "REBIN_AUTO_MAX_CONTRIBUTIONS", 1)
    project_gui._COMPOSITE_DATA_CACHE.clear()

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0), kind="mdhisto", data_type="single_crystal_inelastic")
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0), kind="mdhisto", data_type="single_crystal_inelastic")
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

    config = project_gui.data_group_composite_config(group)
    assert config["auto_rebin"] is False
    auto_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_auto")
    assert auto_check is not None
    assert not auto_check.isChecked()
    status_label = explorer.details_widget.findChild(QtWidgets.QLabel, "group_composite_status")
    assert status_label is not None

    refresh_calls = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda group: refresh_calls.append(group))
    checkbox = explorer.details_widget.findChild(QtWidgets.QCheckBox, "group_composite_enabled")
    assert checkbox is not None
    checkbox.setChecked(True)

    assert refresh_calls == []
    assert config["stale"] is True
    datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert datasets == []
    assert names == []

    assert explorer.rebin_composite_now(group) is True
    assert config["stale"] is False
    datasets, names = project_gui.slice_viewer_datasets(group, force_rebin=False)
    assert names == ["Datagroup1 Composite"]
    np.testing.assert_allclose(datasets[0].signal, [[1.5]])


def test_mdhisto_rebin_applies_enabled_masks_before_binning():
    axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    data = MDHistoData(
        axes=(axis,),
        signal=np.array([1.0, 100.0]),
        errors=np.array([1.0, 1.0]),
        mask=np.array([False, False]),
        num_events=np.array([1.0, 1.0]),
    )
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 1.0]
    mask.enabled = False
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    config["fractional"] = False
    config["axes"][0].update({"lower": 0.0, "upper": 2.0, "num_bins": 1})

    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    np.testing.assert_allclose(viewed.signal, [50.5])
    assert viewed.metadata["rebin"]["source_nfit_mask_count"] == 0

    mask.enabled = True
    viewed = dataset_for_slice_viewer(dataset)

    assert viewed is not None
    np.testing.assert_allclose(viewed.signal, [1.0])
    assert viewed.metadata["rebin"]["source_nfit_mask_count"] == 1


def test_point_data_rebin_applies_enabled_masks_before_binning():
    data = PointData4D(
        H=[0.25, 0.75],
        K=[0.0, 0.0],
        L=[0.0, 0.0],
        E=[0.0, 0.0],
        intensity=[1.0, 100.0],
        sigma=[1.0, 1.0],
    )
    dataset = DatasetEntry("points", data, kind="point")
    mask = create_mask(dataset)
    mask.parameters["H"] = [0.0, 0.5]
    mask.enabled = False
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    config["fractional"] = False
    for axis in config["axes"]:
        axis.update({"lower": 0.0, "upper": 1.0, "num_bins": 1})

    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.intensity, [50.5])

    mask.enabled = True
    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.intensity, [1.0])


def test_import_dataset_paths_dispatches_by_data_type_and_round_trips(tmp_path):
    group = DataGroup("Datagroup1")

    mpms = import_dataset_paths(group, [MPMS_FILE], data_type="magnetization")[0]
    assert mpms.data_type == "magnetization"
    assert isinstance(mpms.data, PointListData)
    assert mpms.metadata["importer"] == "mpms_dat"
    assert mpms.data.coordinate_names == ["Temperature", "Magnetic Field"]

    powder = import_dataset_paths(group, [HB2A_FILE], data_type="powder_elastic")[0]
    assert isinstance(powder.data, PointListData)

    nxs = import_dataset_paths(group, ["/fake/scan.nxs"], data_type="single_crystal_inelastic")[0]
    assert nxs.data is None  # lazy MDHisto placeholder
    assert nxs.data_type == "single_crystal_inelastic"

    project = NfitProject([group])
    path = tmp_path / "proj.nfit"
    save_project(project, path)
    reloaded = load_project(path)
    types = [ds.data_type for ds in reloaded.data_groups[0].datasets]
    assert types == ["magnetization", "powder_elastic", "single_crystal_inelastic"]
    assert reloaded.data_groups[0].datasets[0].metadata["importer"] == "mpms_dat"


def test_set_dataset_data_type_reloads_and_resets():
    group = DataGroup("Datagroup1")
    entry = import_dataset_paths(group, [HB2A_FILE], data_type="powder_elastic")[0]
    assert isinstance(entry.data, PointListData)

    # Switch to an MDHisto/nxs type: point data is dropped for lazy reload.
    set_dataset_data_type(entry, "single_crystal_inelastic")
    assert entry.data is None
    assert entry.data_type == "single_crystal_inelastic"

    # Switch back to a point-list type: importer reloads the columns.
    set_dataset_data_type(entry, "powder_elastic")
    assert isinstance(entry.data, PointListData)
    assert entry.data.coordinate_names == ["2theta"]


def test_project_explorer_data_type_dropdown_switches_type(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [HB2A_FILE], data_type="powder_elastic")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    combo = explorer.details_widget.findChild(QtWidgets.QComboBox, "dataset_data_type")
    assert combo is not None
    assert combo.currentData() == "powder_elastic"
    labels = [label for _name, label in available_data_types()]
    assert combo.count() == len(labels)

    # Switch to an MDHisto/nxs type: data resets for lazy reload.
    combo.setCurrentIndex(combo.findData("single_crystal_inelastic"))
    assert group.datasets[0].data_type == "single_crystal_inelastic"
    assert group.datasets[0].data is None


def test_nested_dataset_groups_share_masks_and_round_trip(tmp_path):
    from nfit.project_gui import (
        effective_dataset_masks,
        slice_viewer_datasets,
    )

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1", datasets=[d2])
    group = DataGroup("Datagroup1", datasets=[d1], subgroups=[sub])
    project_gui.create_group_mask(sub, d2)
    sub.masks[0].parameters["H"] = [-10.0, 0.9]

    # Recursive iteration sees every dataset.
    assert group.dataset_names == ["d1", "d2"]
    assert [d.name for d in group.select()] == ["d1", "d2"]

    # Group mask applies only to the subgroup's descendants (d2), not d1.
    assert [m.name for m in effective_dataset_masks(group, d2)] == ["Mask1"]
    assert effective_dataset_masks(group, d1) == []
    data, names = slice_viewer_datasets(group)
    by_name = dict(zip(names, data))
    assert by_name["d2"].metadata["nfit_mask_count"] > 0
    assert by_name["d1"].metadata["nfit_mask_count"] == 0

    # Round-trip nesting + shared masks through JSON (lazy placeholders).
    save_group = DataGroup("Datagroup2")
    p1 = import_dataset_paths(save_group, [tmp_path / "a.nxs"])[0]
    save_sub = DatasetGroup("SubB")
    save_group.subgroups.append(save_sub)
    p2 = DatasetEntry("b", None, kind="nxs", metadata={"source_file": str(tmp_path / "b.nxs"), "import_status": "pending"})
    save_sub.datasets.append(p2)
    project_gui.create_group_mask(save_sub, None)
    path = tmp_path / "proj.nfit"
    save_project(NfitProject([save_group]), path)
    reloaded = load_project(path).data_groups[0]
    assert reloaded.dataset_names == ["a", "b"]
    assert reloaded.subgroups[0].name == "SubB"
    assert reloaded.subgroups[0].masks[0].name == "Mask1"


def test_multi_select_move_and_import_into_subgroup(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    d3 = DatasetEntry("d3", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1")
    group = DataGroup("Datagroup1", datasets=[d1, d2, d3], subgroups=[sub])
    explorer = NfitProjectExplorer(NfitProject([group]))

    def dataset_item(name):
        datasets_item = explorer.tree.topLevelItem(0).child(0)
        return next(
            datasets_item.child(i)
            for i in range(datasets_item.childCount())
            if datasets_item.child(i).text(0) == name
        )

    # Multi-select d1 + d3 and drop them into the subgroup.
    explorer.tree.setCurrentItem(dataset_item("d1"))
    dataset_item("d1").setSelected(True)
    dataset_item("d3").setSelected(True)
    assert explorer.move_or_copy_selected_to_item(dataset_item("Group1"), copy_item=False)
    assert [d.name for d in group.datasets] == ["d2"]
    assert [d.name for d in sub.datasets] == ["d1", "d3"]

    # Import button is available on a subgroup, and imports land inside it.
    subgroup_item = dataset_item("Group1")
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._sync_details()
    assert not explorer.import_dataset_button.isHidden()
    resolved_group, into = explorer._selected_import_target()
    assert resolved_group is group and into is sub
    explorer.import_dataset_paths(group, ["/fake/new.nxs"], data_type="single_crystal_inelastic", into=sub)
    assert [d.name for d in sub.datasets] == ["d1", "d3", "new"]


def test_project_explorer_drag_reorders_groups_datasets_and_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    d3 = DatasetEntry("d3", _grid_mdhisto_data(), kind="mdhisto")
    m1 = create_mask(d1, "Mask1")
    m2 = create_mask(d1, "Mask2")
    m3 = create_mask(d1, "Mask3")
    group1 = DataGroup("Datagroup1", datasets=[d1, d2, d3])
    group2 = DataGroup("Datagroup2")
    group3 = DataGroup("Datagroup3")
    explorer = NfitProjectExplorer(NfitProject([group1, group2, group3]))
    below = QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.topLevelItem(0).setSelected(True)
    assert explorer.move_or_copy_selected_to_item(explorer.tree.topLevelItem(2), copy_item=False, drop_position=below)
    assert [group.name for group in explorer.project.data_groups] == ["Datagroup2", "Datagroup3", "Datagroup1"]

    datasets_item = explorer.tree.topLevelItem(2).child(0)
    dataset_items = {datasets_item.child(i).text(0): datasets_item.child(i) for i in range(datasets_item.childCount())}
    explorer.tree.clearSelection()
    explorer.tree.setCurrentItem(dataset_items["d1"])
    dataset_items["d1"].setSelected(True)
    dataset_items["d2"].setSelected(True)
    assert explorer.move_or_copy_selected_to_item(dataset_items["d3"], copy_item=False, drop_position=below)
    assert [dataset.name for dataset in group1.datasets] == ["d3", "d1", "d2"]

    datasets_item = explorer.tree.topLevelItem(2).child(0)
    masks_item = next(
        datasets_item.child(i).child(0)
        for i in range(datasets_item.childCount())
        if datasets_item.child(i).text(0) == "d1"
    )
    mask_items = {masks_item.child(i).text(0): masks_item.child(i) for i in range(masks_item.childCount())}
    explorer.tree.clearSelection()
    explorer.tree.setCurrentItem(mask_items["Mask1"])
    mask_items["Mask1"].setSelected(True)
    mask_items["Mask2"].setSelected(True)
    assert explorer.move_or_copy_selected_to_item(mask_items["Mask3"], copy_item=False, drop_position=below)
    assert [mask.name for mask in d1.masks] == ["Mask3", "Mask1", "Mask2"]


def test_project_explorer_nested_group_bulk_edit_and_tree(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1", datasets=[d2])
    group = DataGroup("Datagroup1", datasets=[d1], subgroups=[sub])
    explorer = NfitProjectExplorer(NfitProject([group]))

    datasets_item = explorer.tree.topLevelItem(0).child(0)
    assert [datasets_item.child(i).text(0) for i in range(datasets_item.childCount())] == ["d1", "Group1"]
    subgroup_item = datasets_item.child(1)
    assert [subgroup_item.child(i).text(0) for i in range(subgroup_item.childCount())] == ["Masks", "d2"]

    # Bulk-set scale on the subgroup overwrites all descendants.
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._set_group_bulk_value("scale_factor", "5")
    assert [d.scale_factor for d in sub.iter_datasets()] == [5.0]

    # The top-level workspace is organizational: fit weight and scale stay neutral
    # there, so the bulk editor is hidden and helper edits are ignored.
    top_item = explorer.tree.topLevelItem(0)
    explorer.tree.setCurrentItem(top_item)
    explorer._sync_details()
    assert explorer.group_bulk_widget.isHidden()
    explorer._set_group_bulk_value("fit_weight", "3")
    assert [d.fit_weight for d in group.iter_datasets()] == [1.0, 1.0]

    # Nested dataset groups still support bulk dataset fit controls.
    d1.fit_weight = 2.0
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._sync_details()
    assert not explorer.group_bulk_widget.isHidden()
    assert explorer.group_fit_weight_edit.text() == "1"


def test_dataset_scale_factor_scales_viewed_data_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    scale_spin = explorer.window.findChild(QtWidgets.QDoubleSpinBox, "dataset_scale_factor")
    scale_fit_check = explorer.window.findChild(QtWidgets.QCheckBox, "dataset_scale_factor_vary")
    assert scale_spin is not None
    assert scale_fit_check is not None
    assert scale_spin.value() == 1.0
    assert not scale_fit_check.isChecked()
    scale_spin.setValue(3.0)
    scale_fit_check.setChecked(True)
    assert dataset.scale_factor == 3.0
    assert dataset.scale_factor_vary is True

    viewed = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(viewed.signal, np.asarray(data.signal, dtype=float) * 3.0)
    np.testing.assert_allclose(viewed.errors, np.asarray(data.errors, dtype=float) * 3.0)

    # Round-trip through JSON using a lazy placeholder dataset.
    save_group = DataGroup("Datagroup2")
    placeholder = import_dataset_paths(save_group, [tmp_path / "scan.nxs"])[0]
    placeholder.scale_factor = 3.0
    placeholder.scale_factor_vary = True
    path = tmp_path / "proj.nfit"
    save_project(NfitProject([save_group]), path)
    payload = json.loads(path.read_text())["data_groups"][0]["datasets"][0]
    assert payload["scale_factor"] == 3.0
    assert payload["scale_factor_vary"] is True
    loaded = load_project(path).data_groups[0].datasets[0]
    assert loaded.scale_factor == 3.0
    assert loaded.scale_factor_vary is True


def test_point_list_scale_and_susceptibility_transforms():
    from nfit.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [MPMS_FILE], data_type="magnetization")[0]
    raw = dataset.data
    config = point_list_config(dataset)
    config["scale"] = {"factor": 2.0, "channel": "Moment", "units": "emu/mol"}
    config["susceptibility"] = {"enabled": True, "field": "Magnetic Field", "moment": "Moment"}

    prepared = prepared_point_list_data(dataset)

    # Scale multiplies value and error and relabels units.
    np.testing.assert_allclose(
        prepared.channel_values("Moment"), raw.channel_values("Moment") * 2.0
    )
    np.testing.assert_allclose(
        prepared.channel_errors("Moment"), raw.channel_errors("Moment") * 2.0
    )
    assert prepared.unit(prepared.channel("Moment")["value"]) == "emu/mol"

    # Susceptibility = scaled moment / field.
    assert "Susceptibility" in prepared.channel_labels
    expected = raw.channel_values("Moment") * 2.0 / raw.column("Magnetic Field")
    np.testing.assert_allclose(prepared.channel_values("Susceptibility"), expected)
    assert prepared.unit(prepared.channel("Susceptibility")["value"]) == "emu/mol/Oe"


def test_powder_wavelength_to_q_and_point_rebin():
    from nfit.project_gui import (
        dataset_for_slice_viewer,
        dataset_rebin_config,
        point_list_config,
        prepared_point_list_data,
    )

    group = DataGroup("Datagroup1")
    dataset = import_dataset_paths(group, [HB2A_FILE], data_type="powder_elastic")[0]
    config = point_list_config(dataset)
    config["wavelength"] = {"value": 2.41, "two_theta": "2theta"}

    prepared = prepared_point_list_data(dataset)
    # Powder is 1D: q is the single coordinate; d and 2theta remain columns.
    assert prepared.coordinate_names == ["q"]
    assert "d" in prepared.column_names
    two_theta = dataset.data.column("2theta")
    theta = np.deg2rad(two_theta) / 2.0
    expected_q = 4.0 * np.pi * np.sin(theta) / 2.41
    np.testing.assert_allclose(prepared.column("q"), expected_q)
    # d-spacing in angstroms equals 2*pi/q and lambda/(2 sin theta).
    np.testing.assert_allclose(prepared.column("d"), 2.41 / (2.0 * np.sin(theta)))
    np.testing.assert_allclose(prepared.column("d"), 2.0 * np.pi / prepared.column("q"))
    assert prepared.unit("q") == "Å⁻¹"
    assert prepared.unit("d") == "Å"

    rebin = dataset_rebin_config(dataset)
    assert rebin["axes"][0]["name"] == "q"
    rebin["enabled"] = True
    rebin["axes"][0]["num_bins"] = 100
    viewed = dataset_for_slice_viewer(dataset)
    assert isinstance(viewed, PointListData)
    assert viewed.size < dataset.data.size
    assert "q" in viewed.coordinate_names


def test_point_list_variables_panel_edits_config(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [MPMS_FILE], data_type="magnetization")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))
    dataset = group.datasets[0]

    panel = next(
        box
        for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)
        if box.title() == "Variables and Channels"
    )
    assert panel is not None
    factor_spin = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "point_list_scale_factor")
    factor_spin.setValue(4.0)
    susc_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "point_list_susceptibility_enabled")
    susc_check.setChecked(True)

    config = point_list_config(dataset)
    assert config["scale"]["factor"] == 4.0
    assert config["susceptibility"]["enabled"] is True
    assert "Susceptibility" in prepared_point_list_data(dataset).channel_labels


def test_point_list_dataset_opens_in_data_viewer_as_1d(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [MPMS_FILE], data_type="magnetization")
    datasets, names = project_gui.slice_viewer_datasets(group)
    assert isinstance(datasets[0], PointListData)

    viewer = QtMDHistoSliceViewer(datasets, dataset_names=names)
    viewer.show()
    viewer.update_plot(preserve_view=False)

    assert viewer._is_effective_1d()
    assert viewer.model.is_point_list
    # x-axis selector leads with the coordinates, then offers other non-channel
    # columns; the channel combo offers the channels.
    x_items = [viewer.x_combo.itemText(i) for i in range(viewer.x_combo.count())]
    assert x_items[:2] == ["Temperature", "Magnetic Field"]
    assert "Moment" not in x_items  # channels are not x-axis options
    assert "Moment" in [viewer.channel_combo.itemText(i) for i in range(viewer.channel_combo.count())]

    viewer._set_channel("Moment")
    viewer._set_display_dim("x", x_items.index("Magnetic Field"))
    view = viewer.model.slice_arrays()
    assert view["signal"].size == datasets[0].size
    assert viewer.model._channel_label() == "Moment (emu)"
    assert viewer.model._axis_label(0) == "Magnetic Field (Oe)"

    # The channel has an error column, so error bars are drawn for point data.
    assert viewer.show_errorbars
    assert len(viewer.ax_image.containers) == 1

    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][0]),
        ydata=float(view["signal"][0]),
    )
    viewer._on_motion(event)

    assert viewer.cursor_hkle_label.isHidden()
    assert viewer.cursor_q_label.isHidden()
    assert viewer.cursor_intensity_label.text().startswith("Signal = ")


def test_energy_q_range_mask_default_min_q_is_zero():
    parameters = default_mask_parameters("energy_q_range")
    assert parameters["q_modulus"] == [0.0, 1.0e99]


def test_coordinate_range_axis_vectors_are_rounded():
    hh_axis = MDHistoAxis("[H,H,0]", np.array([0.0, 0.5, 1.0]), "rlu", "momentum")
    l_axis = MDHistoAxis("[0,0,L]", np.array([0.0, 0.5, 1.0]), "rlu", "momentum")
    data = MDHistoData(
        axes=(hh_axis, l_axis),
        signal=np.zeros((2, 2)),
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    mask = create_mask(dataset)

    for key in ("axis_0", "axis_1"):
        vector = mask.parameters[key]
        assert len(vector) == 2
        for component in vector:
            assert component == round(component, 10)
    assert 0.5 in mask.parameters["axis_0"]


def test_rebin_axis_vector_projects_new_coordinate():
    h_axis = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    e_axis = MDHistoAxis("E", np.array([0.0, 10.0, 20.0]), "meV", "energy")
    signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = MDHistoData(
        axes=(h_axis, e_axis),
        signal=signal,
        errors=np.ones((2, 2)),
        mask=np.zeros((2, 2), dtype=bool),
        num_events=np.ones((2, 2)),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)
    config = dataset_rebin_config(dataset)
    config["enabled"] = True
    # Swap which physical coordinate maps to each output axis.
    config["axes"][0].update({"vector": [0.0, 1.0], "lower": 0.0, "upper": 20.0, "num_bins": 2})
    config["axes"][1].update({"vector": [1.0, 0.0], "lower": 0.0, "upper": 2.0, "num_bins": 2})

    rebinned = project_gui.rebinned_dataset_data(dataset)

    assert rebinned.shape == (2, 2)
    np.testing.assert_allclose(rebinned.signal, signal.T)
    assert rebinned.metadata["rebin"]["vectors"] == [[0.0, 1.0], [1.0, 0.0]]


def test_rebin_defaults_follow_mdhisto_axis_coordinate_vectors():
    axes = (
        MDHistoAxis("DeltaE", np.array([0.0, 1.0, 2.0]), "meV", "energy"),
        MDHistoAxis("[H,-H,0]", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
        MDHistoAxis("[0,0,L]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
        MDHistoAxis("[H,H,0]", np.array([0.0, 1.0, 2.0]), "rlu", "momentum"),
    )
    signal = np.arange(16, dtype=float).reshape((2, 2, 2, 2))
    data = MDHistoData(
        axes=axes,
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )
    dataset = DatasetEntry("scan", data)

    config = dataset_rebin_config(dataset)

    assert [axis["vector"] for axis in config["axes"]] == [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    config["enabled"] = True
    config["fractional"] = False
    rebinned = project_gui.rebinned_dataset_data(dataset)

    np.testing.assert_allclose(rebinned.signal, signal)
    assert rebinned.metadata["rebin"]["vectors"] == [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_add_mask_and_slice_viewer_from_masks_node(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    opened = []

    class FakeViewer:
        def __init__(self, datasets, *, dataset_names):
            self.dataset_names = list(dataset_names)
            self.selected = None
            self.shown = False
            self.window = None
            opened.append(self)

            class _Combo:
                def __init__(self, names):
                    self.names = names
                    self.index = 0

                def setCurrentIndex(self, index):
                    self.index = int(index)

                def currentText(self):
                    return self.names[self.index]

            self.dataset_combo = _Combo(self.dataset_names)

        def show(self):
            self.shown = True

    monkeypatch.setattr(project_gui, "QtMDHistoSliceViewer", FakeViewer)

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))

    masks_item = explorer.tree.topLevelItem(0).child(0).child(0).child(0)
    explorer.tree.setCurrentItem(masks_item)

    action_names = explorer.context_menu_action_names(masks_item)
    assert "Add mask" in action_names
    assert "View in data viewer" in action_names
    assert not explorer.add_mask_button.isHidden()
    assert not explorer.view_slice_button.isHidden()

    mask = explorer.add_mask_to_selection()
    assert mask is not None
    assert dataset.masks == [mask]

    # Re-select the Masks node and open the viewer for the owning dataset.
    masks_item = explorer.tree.topLevelItem(0).child(0).child(0).child(0)
    explorer.tree.setCurrentItem(masks_item)
    viewer = explorer.open_slice_viewer_for_selection()

    assert viewer is opened[0]
    assert viewer.dataset_combo.currentText() == "scan"


def test_dataset_details_text_summarizes_point_data_conditions():
    data = PointData4D(
        H=[0.0, 1.0],
        K=[0.0, 0.0],
        L=[0.0, 0.5],
        E=[3.0, 4.0],
        intensity=[10.0, 11.0],
        sigma=[1.0, 1.0],
        temperature=[5.0, 10.0],
        metadata={"field": "3 T"},
    )
    dataset = DatasetEntry("points", data, kind="point", parameters={"scan": 17})

    text = dataset_details_text(dataset)

    assert "Dimensions: 4" in text
    assert "H (rlu) - 2 points, range 0 to 1" in text
    assert "E (meV) - 2 points, range 3 to 4" in text
    assert "Valid points: 2" in text
    assert "Temperature: 5 to 10 K" in text
    assert "field: 3 T" in text
    assert "scan: 17" in text


def _standard_shortcut_text(QtGui, standard_key):
    return QtGui.QKeySequence.keyBindings(standard_key)[0].toString(
        QtGui.QKeySequence.SequenceFormat.PortableText
    )


def _tree_items_with_children(tree):
    items = []

    def append_item(item):
        if item.childCount():
            items.append(item)
        for index in range(item.childCount()):
            append_item(item.child(index))

    for index in range(tree.topLevelItemCount()):
        append_item(tree.topLevelItem(index))
    return items


def _tiny_mdhisto_data(value):
    axis_x = MDHistoAxis("H", np.array([0.0, 1.0]), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.array([0.0, 1.0]), "meV", "energy")
    signal = np.array([[value]])
    return MDHistoData(
        axes=(axis_x, axis_e),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def _grid_mdhisto_data():
    axis_x = MDHistoAxis("H", np.array([0.0, 1.0, 2.0]), "rlu", "momentum")
    axis_e = MDHistoAxis("E", np.array([0.0, 10.0, 20.0]), "meV", "energy")
    signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    return MDHistoData(
        axes=(axis_x, axis_e),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
        metadata={},
    )


def test_fit_points_carry_temperature_and_group_lattice():
    data = _grid_mdhisto_data()
    data.metadata["temperature"] = 4.2
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={"a": 4.0, "b": 4.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )

    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle is not None
    assert bundle.points.temperature == 4.2
    matrix = np.asarray(bundle.points.metadata["rlu_to_inv_angstrom_matrix"])
    np.testing.assert_allclose(matrix[0, 0], 2.0 * np.pi / 4.0)
    np.testing.assert_allclose(matrix[2, 2], 2.0 * np.pi / 8.0)

    dataset.parameters["temperature"] = 100.0
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.temperature == 100.0
    assert project_gui.effective_dataset_temperature(group, dataset) == 100.0


def test_fit_points_leave_temperature_unset_without_metadata():
    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.temperature is None
    assert "rlu_to_inv_angstrom_matrix" not in bundle.points.metadata


def test_dataset_temperature_spin_writes_override(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    spin = explorer.dataset_temperature_spin
    assert spin.specialValueText() == "(from data)"
    assert spin.value() == -1.0

    spin.setValue(4.2)
    assert dataset.parameters["temperature"] == 4.2

    spin.setValue(-1.0)
    assert "temperature" not in dataset.parameters


def test_dataset_temperature_edit_from_active_result_refreshes_current_state(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    result = project_gui.create_fit_result_entry(
        group,
        group.fits[0],
        goodness={"status": "converged", "parameters": {}},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))

    result_item = explorer.tree.topLevelItem(0).child(2).child(1)
    explorer.tree.setCurrentItem(result_item)
    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)

    explorer.dataset_temperature_spin.setValue(12.5)

    assert [fit.kind for fit in group.fits] == ["initial", "result", "current"]
    assert group.fits[2].snapshot["datasets"][0]["parameters"]["temperature"] == 12.5
    fits_item = explorer.tree.topLevelItem(0).child(2)
    assert fits_item.childCount() == 3
    assert explorer._fit_entry_for_item(fits_item.child(2)) is group.fits[2]
    assert result.children == []


def test_failed_fit_from_result_creates_current_state_for_temperature_fix(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    first_result = project_gui.create_fit_result_entry(
        group,
        group.fits[0],
        goodness={"status": "converged", "parameters": {}},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fail_fit(group, *, optimizer_config=None, progress_callback=None):
        raise ValueError("this model requires a valid sample temperature")

    monkeypatch.setattr(project_gui, "perform_group_fit", fail_fit)
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    failed = explorer.fit_now_for_selection()

    assert failed is not None
    assert failed.goodness["status"] == "failed"
    assert [fit.kind for fit in group.fits] == ["initial", "result", "result", "current"]
    assert group.fits[1] is first_result
    assert group.fits[2] is failed
    current = group.fits[3]
    assert explorer._active_fit_entry(group) is current
    assert explorer.tree.currentItem().text(0) == "Current state"

    dataset_item = explorer.tree.topLevelItem(0).child(0).child(0)
    explorer.tree.setCurrentItem(dataset_item)
    explorer.dataset_temperature_spin.setValue(8.0)

    assert [fit.kind for fit in group.fits] == ["initial", "result", "result", "current"]
    assert group.fits[3] is current
    assert current.snapshot["datasets"][0]["parameters"]["temperature"] == 8.0
    assert explorer._active_fit_entry(group) is current


def test_failed_fit_from_current_state_keeps_current_state_selected(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("scan", _grid_mdhisto_data(), kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    create_model_component(group)
    project_gui.ensure_fit_history(group)
    current = project_gui.current_state_fit_entry(group)
    group.fits.append(current)
    explorer = NfitProjectExplorer(NfitProject([group]))

    def fail_fit(group, *, optimizer_config=None, progress_callback=None):
        raise ValueError("this model requires a valid sample temperature")

    monkeypatch.setattr(project_gui, "perform_group_fit", fail_fit)
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    failed = explorer.fit_now_for_selection()

    assert failed is not None
    assert failed.goodness["status"] == "failed"
    assert [fit.kind for fit in group.fits] == ["initial", "result", "current"]
    assert group.fits[1] is failed
    assert group.fits[2] is current
    assert explorer._active_fit_entry(group) is current
    assert explorer.tree.currentItem().text(0) == "Current state"


def test_heisenberg_rpa_editor_generates_orbits_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("gemmi")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))
    assert model.type == "heisenberg_rpa"

    crystal_box = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_crystal_group")
    bonds_box = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_bonds_group")
    sites_box = explorer.model_parameter_widget.findChild(QtWidgets.QGroupBox, "model_crystal_sites_group")
    assert crystal_box is not None and bonds_box is not None and sites_box is not None

    # configure an FCC crystal through the handlers
    for name in ("a", "b", "c"):
        explorer._set_model_crystal_lattice(name, "4.0")
    explorer._set_model_crystal_spacegroup("F m -3 m")
    explorer._add_model_crystal_site()
    explorer._set_model_crystal_site(0, "label", "Ni1")
    explorer._set_model_crystal_site(0, "ion", "Ni2")
    explorer._set_model_site_magnetic(0, True)

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_bonds_cutoff"
    )
    assert cutoff_editor is not None
    cutoff_editor.setText("3.0")
    explorer._generate_selected_model_bond_orbits()

    assert [orbit["label"] for orbit in model.config["orbits"]] == ["J1"]
    assert len(model.config["site_positions"]) == 4
    assert model.parameters["J1"] == 0.0
    assert model.fit_parameters["J1"] is False

    j1_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_parameter_min_J1"
    )
    assert j1_editor is not None  # J1 row exists in the fit-parameter grid

    orbit_table = explorer.model_parameter_widget.findChild(QtWidgets.QTableWidget, "model_bonds_table")
    assert orbit_table is not None
    assert orbit_table.rowCount() == 1
    assert orbit_table.item(0, 0).text() == "J1"
    assert orbit_table.item(0, 2).text() == "24"

    cutoff_editor = explorer.model_parameter_widget.findChild(
        QtWidgets.QLineEdit, "model_bonds_cutoff"
    )
    assert cutoff_editor is not None
    cutoff_editor.setText("4.1")
    explorer._generate_selected_model_bond_orbits()

    assert model.config["bond_cutoff_angstrom"] == 4.1
    assert [orbit["label"] for orbit in model.config["orbits"]] == ["J1", "J2"]

    explorer._set_model_parameter("J1", "0.15")
    explorer._set_model_fit_parameter("J1", True)

    path = tmp_path / "project.json"
    save_project(NfitProject([group]), path)
    loaded = load_project(path)
    loaded_model = next(iter(loaded.data_groups[0].models.values()))
    assert [orbit["label"] for orbit in loaded_model.config["orbits"]] == ["J1", "J2"]
    assert loaded_model.parameters["J1"] == 0.15
    assert loaded_model.fit_parameters["J1"] is True
    assert loaded_model.config["site_positions"] == model.config["site_positions"]


def test_heisenberg_rpa_editor_scrolls_while_fit_parameters_grow(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")

    group = DataGroup("Datagroup1")
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    model = explorer.add_model_to_selection()

    combo = explorer.model_type_combo
    combo.setCurrentIndex(combo.findData("heisenberg_rpa"))
    model.config["orbits"] = [{"label": f"J{index}", "bonds": []} for index in range(1, 25)]
    project_gui.reconcile_model_orbit_parameters(model)
    explorer._rebuild_model_parameter_editor(model)

    fit_group = explorer.model_parameter_widget.findChild(
        QtWidgets.QGroupBox, "model_fit_parameters_group"
    )
    model_scroll = explorer.window.findChild(QtWidgets.QScrollArea, "model_parameter_scroll")
    orbit_table = explorer.model_parameter_widget.findChild(QtWidgets.QTableWidget, "model_bonds_table")
    assert fit_group is not None
    assert model_scroll is not None
    assert orbit_table is not None
    assert not model_scroll.isHidden()
    assert model_scroll.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert model_scroll.widget().minimumSizeHint().height() > model_scroll.minimumHeight()
    assert fit_group.findChild(QtWidgets.QScrollArea, "model_fit_parameters_scroll") is None
    assert fit_group.minimumSizeHint().height() > model_scroll.minimumHeight()
    assert orbit_table.maximumHeight() == orbit_table.minimumHeight()
    assert orbit_table.maximumHeight() >= orbit_table.horizontalHeader().height() + 4 * 24
    assert (
        explorer.model_parameter_widget.findChild(
            QtWidgets.QLineEdit, "model_parameter_min_J24"
        )
        is not None
    )


def test_import_cif_into_model_populates_config_and_group(tmp_path):
    pytest.importorskip("gemmi")
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import generate_model_bond_orbits, import_cif_into_model

    cif = tmp_path / "fcc.cif"
    cif.write_text(
        """
data_fcc
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
"""
    )
    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.1, "gamma0": 5.0},
    )
    group = DataGroup("Datagroup1")
    imported = import_cif_into_model(model, str(cif), group=group)
    assert model.config["crystal"]["lattice"]["a"] == pytest.approx(4.0)
    assert group.lattice_parameters["a"] == pytest.approx(4.0)
    assert group.metadata["crystal"] == imported

    model.config["magnetic_sites"] = ["Ni1"]
    model.config["bond_cutoff_angstrom"] = 3.0
    labels = generate_model_bond_orbits(model)
    assert labels == ["J1"]
    assert model.parameters["J1"] == 0.0


def test_generate_model_bond_orbits_requires_magnetic_site():
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import generate_model_bond_orbits, model_crystal_config

    model = ModelComponentSpec(name="rpa", type="heisenberg_rpa", parameters={})
    model_crystal_config(model)["sites"].append(
        {"label": "X1", "position": [0.0, 0.0, 0.0], "ion": ""}
    )
    with pytest.raises(ValueError, match="magnetic site"):
        generate_model_bond_orbits(model)


def test_reconcile_model_orbit_parameters_keeps_and_drops():
    from nfit.pipeline import ModelComponentSpec
    from nfit.project_gui import reconcile_model_orbit_parameters

    model = ModelComponentSpec(
        name="rpa",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.1, "gamma0": 5.0, "J1": 0.2, "J9": 0.7},
        fit_parameters={"J1": True, "J9": True},
        limits={"J9": [0.0, 1.0]},
    )
    model.config["orbits"] = [
        {"label": "J1", "bonds": []},
        {"label": "J2", "bonds": []},
    ]
    reconcile_model_orbit_parameters(model)
    assert model.parameters["J1"] == 0.2
    assert model.parameters["J2"] == 0.0
    assert "J9" not in model.parameters
    assert "J9" not in model.fit_parameters
    assert "J9" not in model.limits
    assert model.fit_parameters["J1"] is True


def _rpa_overlay_group():
    data = _grid_mdhisto_data()
    data.metadata["temperature"] = 5.0
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    model = ModelComponentSpec(
        name="M",
        type="heisenberg_rpa",
        parameters={"scale": 1.0, "chi0": 0.3, "gamma0": 2.0, "J1": 0.1},
        fit_parameters={"scale": True, "chi0": True, "gamma0": True, "J1": True},
        config={
            "site_positions": [[0.0, 0.0, 0.0], [0.31, 0.47, 0.11]],
            "orbits": [
                {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]}
            ],
        },
    )
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={"a": 4.0, "b": 4.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    group.models[model.name] = model
    return group, dataset, model


def test_overlay_cache_reuses_compiled_problem_across_parameter_edits():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, _dataset, model = _rpa_overlay_group()

    first = project_gui.current_model_channels(group)
    assert "scan" in first
    cached = project_gui._MODEL_OVERLAY_CACHE[id(group)]
    compiled_first = cached["compiled"]

    # A parameter-value edit must reuse the cached bundles/compiled problem...
    model.parameters["J1"] = 0.25
    second = project_gui.current_model_channels(group)
    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is compiled_first
    # ...while still reflecting the new value in the overlay.
    a = np.asarray(first["scan"]["fit"], dtype=float)
    b = np.asarray(second["scan"]["fit"], dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.any()
    assert np.nanmax(np.abs(a[finite] - b[finite])) > 0.0


def test_overlay_cache_invalidates_on_structural_change():
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, _dataset, model = _rpa_overlay_group()
    project_gui.current_model_channels(group)
    compiled_first = project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"]

    # Changing the bond network (structure) must rebuild, not reuse.
    model.config["orbits"] = [
        {"label": "J1", "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}]},
        {"label": "J2", "bonds": [{"site_i": 0, "site_j": 0, "offset": [0, 0, 1]}]},
    ]
    model.parameters["J2"] = 0.05
    project_gui.current_model_channels(group)
    assert project_gui._MODEL_OVERLAY_CACHE[id(group)]["compiled"] is not compiled_first


def test_overlay_evaluates_only_valid_points(monkeypatch):
    project_gui._MODEL_OVERLAY_CACHE.clear()
    group, dataset, _model = _rpa_overlay_group()
    # Mask one grid cell; the overlay there must be NaN, and the model must not
    # be evaluated over the masked point.
    dataset.data.mask[0, 0] = True

    seen_sizes = []
    original = project_gui.evaluate_problem_model

    def spy(problem, name, params, data=None):
        seen_sizes.append(data.size if data is not None else None)
        return original(problem, name, params, data=data)

    monkeypatch.setattr(project_gui, "evaluate_problem_model", spy)
    channels = project_gui.current_model_channels(group)
    fit = np.asarray(channels["scan"]["fit"], dtype=float)
    assert not np.isfinite(fit[0, 0])  # masked cell is NaN
    # exactly the unmasked points were evaluated (fewer than the full grid)
    assert seen_sizes and seen_sizes[0] == int(np.isfinite(fit).sum())


def test_request_overlay_refresh_coalesces_without_event_loop(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    group, _dataset, _model = _rpa_overlay_group()
    explorer = NfitProjectExplorer(NfitProject([group]))

    refreshed = []
    monkeypatch.setattr(explorer, "refresh_slice_viewer", lambda g: refreshed.append(g))
    # No viewer open -> request is a no-op.
    explorer._request_overlay_refresh(group)
    assert refreshed == []

    explorer._slice_viewers[id(group)] = object()

    # Headless/non-interactive: refresh happens synchronously so callers and
    # tests observe the update immediately.
    assert explorer._interactive is False
    explorer._request_overlay_refresh(group)
    assert refreshed == [group]

    # Interactive: rapid requests coalesce into the pending set + a debounce
    # timer instead of refreshing on every call.
    refreshed.clear()
    explorer._interactive = True
    explorer._request_overlay_refresh(group)
    explorer._request_overlay_refresh(group)
    assert refreshed == []
    assert id(group) in explorer._pending_overlay_groups
    assert explorer._overlay_refresh_timer is not None
    # Firing the debounced slot runs exactly one refresh and clears the queue.
    explorer._run_pending_overlay_refresh()
    assert refreshed == [group]
    assert not explorer._pending_overlay_groups


def test_viewer_view_cache_reuses_and_invalidates():
    project_gui._VIEWER_VIEW_CACHE.clear()
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    group = DataGroup("Datagroup1", datasets=[dataset])

    first = project_gui._viewer_data_before_scale(dataset)
    second = project_gui._viewer_data_before_scale(dataset)
    assert first is second  # identical masked view reused

    # A mask change must invalidate the cache (new object, new masking).
    dataset.masks = [
        MaskSpec(name="m", type="coordinate_range", parameters={"E": [0.0, 5.0]})
    ]
    third = project_gui._viewer_data_before_scale(dataset)
    assert third is not second
    # ...and is reused again until the next change.
    assert project_gui._viewer_data_before_scale(dataset) is third


def test_mdhisto_fit_bin_count_matches_point_based_count():
    data = _grid_mdhisto_data()
    data.mask[0, 0] = True
    data.signal[1, 1] = np.nan
    dataset = DatasetEntry("scan", data, kind="mdhisto", data_type="single_crystal_inelastic")
    group = DataGroup("Datagroup1", datasets=[dataset])
    view = project_gui.dataset_for_slice_viewer(dataset)
    fast = project_gui._mdhisto_fit_bin_count(view)
    reference = int(np.count_nonzero(project_gui._point_data_from_mdhisto_view(view).valid_mask()))
    assert fast == reference


def test_fit_points_carry_magnetic_field_from_dataset_parameters():
    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={"a": 3.0, "b": 5.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )

    # No field set -> nothing stamped.
    bundle = project_gui.fit_data_bundle(group, dataset)
    assert bundle.points.magnetic_field is None
    assert project_gui.effective_dataset_field(group, dataset) is None

    dataset.parameters["magnetic_field"] = {
        "magnitude_T": 2.0,
        "direction": [1, 1, 0],
        "frame": "uvw",
    }
    bundle = project_gui.fit_data_bundle(group, dataset)
    expected = 2.0 * np.array([3.0, 5.0, 0.0]) / np.linalg.norm([3.0, 5.0, 0.0])
    np.testing.assert_allclose(bundle.points.magnetic_field, expected, atol=1e-12)

    # hkl frame differs for this orthorhombic lattice.
    dataset.parameters["magnetic_field"]["frame"] = "hkl"
    bundle = project_gui.fit_data_bundle(group, dataset)
    expected_hkl = 2.0 * np.array([1 / 3.0, 1 / 5.0, 0.0]) / np.linalg.norm([1 / 3.0, 1 / 5.0, 0.0])
    np.testing.assert_allclose(bundle.points.magnetic_field, expected_hkl, atol=1e-12)

    # Zero magnitude means no field.
    dataset.parameters["magnetic_field"]["magnitude_T"] = 0.0
    assert project_gui.effective_dataset_field(group, dataset) is None

    # Without lattice parameters the direction cannot be oriented.
    dataset.parameters["magnetic_field"]["magnitude_T"] = 2.0
    bare_group = DataGroup("Bare", datasets=[dataset])
    assert project_gui.effective_dataset_field(bare_group, dataset) is None


def test_sample_environment_panel_hosts_temperature_and_field(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup(
        "Datagroup1",
        datasets=[dataset],
        lattice_parameters={"a": 3.0, "b": 5.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    )
    explorer = NfitProjectExplorer(NfitProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    # Panel exists, is titled correctly, and sits between Dataset and Axes.
    panel = explorer.details_widget.findChild(QtWidgets.QGroupBox, "dataset_sample_environment_group")
    assert panel is not None
    titles = [box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)]
    assert titles.index("Sample environment") == titles.index("Dataset") + 1
    assert titles.index("Sample environment") < titles.index("Axes")

    # Temperature control relocated but keeps its objectName + behavior.
    temp = explorer.window.findChild(QtWidgets.QDoubleSpinBox, "dataset_temperature")
    assert temp is explorer.dataset_temperature_spin
    explorer._set_selected_dataset_temperature(12.5)
    assert dataset.parameters["temperature"] == 12.5

    # Field controls write the structured parameter.
    explorer.dataset_field_magnitude_spin.setValue(2.0)
    explorer.dataset_field_direction_edit.setText("1 1 0")
    explorer._set_selected_dataset_field_direction()
    field = dataset.parameters["magnetic_field"]
    assert field["magnitude_T"] == 2.0
    assert field["direction"] == [1.0, 1.0, 0.0]
    assert field["frame"] == "uvw"

    # Switching the frame updates the payload; the panel re-syncs on reselect.
    idx = explorer.dataset_field_frame_combo.findData("hkl")
    explorer.dataset_field_frame_combo.setCurrentIndex(idx)
    assert dataset.parameters["magnetic_field"]["frame"] == "hkl"

    # Zeroing the magnitude clears the field entirely.
    explorer.dataset_field_magnitude_spin.setValue(-1.0)
    assert "magnetic_field" not in dataset.parameters

    # Field survives the data-group state snapshot/restore round-trip (the same
    # mechanism the fit timeline and project files serialize through).
    dataset.parameters["magnetic_field"] = {"magnitude_T": 3.0, "direction": [0, 0, 1], "frame": "hkl"}
    snapshot = project_gui.snapshot_data_group_state(group)
    dataset.parameters.pop("magnetic_field")
    project_gui.restore_data_group_state(group, snapshot)
    restored = list(group.iter_datasets())[0]
    assert restored.parameters["magnetic_field"] == {
        "magnitude_T": 3.0,
        "direction": [0, 0, 1],
        "frame": "hkl",
    }
