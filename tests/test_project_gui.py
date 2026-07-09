import json
from pathlib import Path

import numpy as np
import pytest

import metallix
import metallix.project_gui as project_gui
from metallix.dataset import PointData4D, PointListData
from metallix.mdhisto import MDHistoAxis, MDHistoData
from metallix.project_gui import (
    MetallixProject,
    MetallixProjectExplorer,
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
from metallix.pipeline import DataGroup, DatasetEntry, DatasetGroup


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MPMS_FILE = DATA_DIR / "MPMS" / "test_MPMS.dat"
HB2A_FILE = DATA_DIR / "HB2A" / "test_powder_diffraction.dat"


def test_project_helpers_name_import_and_round_trip(tmp_path):
    assert metallix.create_data_group is create_data_group
    assert metallix.import_dataset_paths is import_dataset_paths

    project = MetallixProject(
        data_groups=[
            DataGroup("Datagroup1"),
            DataGroup("Datagroup3"),
        ]
    )

    assert next_data_group_name(project.data_groups) == "Datagroup2"

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

    project_path = tmp_path / "project.mtlx"
    save_project(project, project_path)

    payload = json.loads(project_path.read_text(encoding="utf-8"))
    assert payload["format"] == "metallix-project"
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
        "Datagroup1",
        "Datagroup3",
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
    explorer = MetallixProjectExplorer(MetallixProject([group_a, group_b]))
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
    assert toolbar.font().pointSize() == tree.font().pointSize()

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
    explorer = MetallixProjectExplorer(MetallixProject([group]))

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
    explorer = MetallixProjectExplorer(MetallixProject([group]))
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
    explorer = MetallixProjectExplorer(MetallixProject([group]))
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


def test_project_explorer_adds_edits_and_copies_masks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    first = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    second = DatasetEntry("second", _tiny_mdhisto_data(2.0))
    group = DataGroup("Datagroup1", datasets=[first, second])
    explorer = MetallixProjectExplorer(MetallixProject([group]))

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


def test_dataset_for_slice_viewer_combines_file_and_metallix_coordinate_masks():
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
    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], [[True], [False]])
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
    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], expected)


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

    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], [[False], [False], [False]])
    np.testing.assert_array_equal(viewed.mask, [[False], [False], [True]])

    dataset.masks = [additive, masking]
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], [[True], [False], [False]])

    masking.invert = True
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], [[False], [True], [True]])

    masking.enabled = False
    viewed = dataset_for_slice_viewer(dataset)

    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], [[False], [False], [False]])


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
    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], expected)
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
    np.testing.assert_array_equal(viewed.metadata["metallix_mask"], expected)


def test_project_explorer_adds_and_edits_models(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1")
    explorer = MetallixProjectExplorer(MetallixProject([group]))
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
    assert fit_group is not None
    assert config_group is not None
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

    models_item = explorer.tree.topLevelItem(0).child(1)
    explorer.tree.setCurrentItem(models_item)
    assert not explorer.add_model_button.isHidden()


def test_project_explorer_fit_history_creates_results_branches_and_restores(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    dataset = DatasetEntry("first", _tiny_mdhisto_data(1.0))
    mask = create_mask(dataset)
    mask.parameters["H"] = [-1.0, 1.0]
    group = DataGroup("Datagroup1", datasets=[dataset])
    model = create_model_component(group)
    model.parameters["constant"] = 0.5
    explorer = MetallixProjectExplorer(MetallixProject([group]))

    group_item = explorer.tree.topLevelItem(0)
    fits_item = group_item.child(2)
    initial_item = fits_item.child(0)
    assert fits_item.text(0) == "Fits"
    assert initial_item.text(0) == "Initial"

    explorer.tree.setCurrentItem(initial_item)
    result = explorer.fit_now_for_selection()

    assert result is not None
    assert result.name == "Fit Result1"
    assert [fit.name for fit in group.fits] == ["Initial", "Fit Result1", "Current state"]
    assert group.fits[1].goodness["status"] == "not run"

    dataset.masks[0].parameters["H"] = [0.0, 0.0]
    model.parameters["constant"] = 9.0
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))

    assert dataset.masks[0].parameters["H"] == [-1.0, 1.0]
    assert model.parameters["constant"] == 0.5

    model_item = explorer.tree.topLevelItem(0).child(1).child(0)
    explorer.tree.setCurrentItem(model_item)
    explorer._set_model_parameter("constant", "1.25")

    edit_branch = group.fits[1].children[0]
    assert edit_branch.kind == "timeline"
    assert edit_branch.children[-1].kind == "current"
    assert edit_branch.children[-1].snapshot["models"][0]["parameters"]["constant"] == 1.25

    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(1))
    explorer.fit_branch_check.setChecked(True)
    branched = explorer.fit_now_for_selection()

    assert branched is not None
    explicit_branch = group.fits[1].children[-1]
    assert explicit_branch.kind == "timeline"
    assert explicit_branch.children[0] is branched
    assert explicit_branch.children[-1].name == "Current state"
    assert explorer.tree.currentItem().text(0) == "Current state"
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()


def test_project_explorer_fit_now_from_earlier_result_creates_nested_timeline(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    group = DataGroup("Datagroup1", datasets=[DatasetEntry("first", _tiny_mdhisto_data(1.0))])
    create_model_component(group)
    explorer = MetallixProjectExplorer(MetallixProject([group]))

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
        "Current state",
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
        "Current state",
    ]
    assert group.fits[3].children[0].kind == "timeline"
    assert group.fits[3].children[0].children[0] is branched
    assert group.fits[3].children[0].children[-1].name == "Current state"
    assert explorer.tree.currentItem().text(0) == "Current state"
    assert not explorer.fit_now_button.isHidden()
    assert explorer.import_dataset_button.isHidden()


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
    explorer = MetallixProjectExplorer(MetallixProject([group]))

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
    first = tmp_path / "first.mtlx"
    second = tmp_path / "second.mtlx"
    missing = tmp_path / "missing.mtlx"
    save_project(MetallixProject(), first)
    save_project(MetallixProject([DataGroup("Datagroup1")]), second)

    remember_recent_project(first, settings)
    remember_recent_project(second, settings)
    remember_recent_project(first, settings)

    assert recent_project_paths(settings) == [first, second]

    remember_recent_project(missing, settings)
    assert recent_project_paths(settings)[0] == missing
    assert forget_missing_recent_projects(settings) == [first, second]

    monkeypatch.setattr(project_gui, "_settings", lambda: settings)
    explorer = MetallixProjectExplorer()
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

    explorer = MetallixProjectExplorer()
    group = explorer.create_data_group()

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

    explorer.project_path = Path("/tmp/opened.mtlx")
    explorer.project = MetallixProject([group])
    explorer.has_unsaved_changes = True
    monkeypatch.setattr(explorer, "_confirm_save_before_closing_project", lambda: True)

    assert explorer.close_project() is True
    assert explorer.project.data_groups == []
    assert explorer.has_unsaved_changes is False


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

    explorer = MetallixProjectExplorer()
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
    explorer = MetallixProjectExplorer(MetallixProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    text = explorer.details_label.text()
    panel_titles = [box.title() for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox)]

    assert panel_titles == ["Dataset", "Axes", "Rebin", "Crystal", "Data", "Source", "Metadata"]
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


def test_dataset_rebin_config_updates_slice_viewer_materializes_and_saves(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = MetallixProjectExplorer(MetallixProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    config = dataset_rebin_config(dataset)
    assert config["axes"][0]["num_bins"] == 2
    assert config["axes"][1]["num_bins"] == 2
    assert config["fractional"] is True

    enable_check = explorer.details_widget.findChild(QtWidgets.QCheckBox, "dataset_rebin_enabled")
    assert enable_check is not None
    rebin_panel = next(
        box for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox) if box.title() == "Rebin"
    )
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fractional").isChecked()
    assert rebin_panel.findChild(QtWidgets.QCheckBox, "dataset_rebin_fit_enabled") is None
    assert not rebin_panel.findChildren(QtWidgets.QComboBox)
    assert any(button.text() == "Create dataset from rebin" for button in rebin_panel.findChildren(QtWidgets.QPushButton))
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

    rebinned = explorer.materialize_rebin_for_selection()

    assert rebinned is not None
    assert group.dataset_names == ["scan", "scan rebinned"]
    assert isinstance(rebinned.data, MDHistoData)
    assert rebinned.data.shape == (3, 1)

    save_path = tmp_path / "rebinned.npz"
    save_dataset_file(dataset, save_path)

    saved = np.load(save_path)
    assert saved["signal"].shape == (3, 1)
    assert int(saved["axis_count"]) == 2


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

    project = MetallixProject([group])
    path = tmp_path / "proj.mtlx"
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
    explorer = MetallixProjectExplorer(MetallixProject([group]))
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
    from metallix.project_gui import (
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
    assert by_name["d2"].metadata["metallix_mask_count"] > 0
    assert by_name["d1"].metadata["metallix_mask_count"] == 0

    # Round-trip nesting + shared masks through JSON (lazy placeholders).
    save_group = DataGroup("Datagroup2")
    p1 = import_dataset_paths(save_group, [tmp_path / "a.nxs"])[0]
    save_sub = DatasetGroup("SubB")
    save_group.subgroups.append(save_sub)
    p2 = DatasetEntry("b", None, kind="nxs", metadata={"source_file": str(tmp_path / "b.nxs"), "import_status": "pending"})
    save_sub.datasets.append(p2)
    project_gui.create_group_mask(save_sub, None)
    path = tmp_path / "proj.mtlx"
    save_project(MetallixProject([save_group]), path)
    reloaded = load_project(path).data_groups[0]
    assert reloaded.dataset_names == ["a", "b"]
    assert reloaded.subgroups[0].name == "SubB"
    assert reloaded.subgroups[0].masks[0].name == "Mask1"


def test_project_explorer_nested_group_bulk_edit_and_tree(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    d1 = DatasetEntry("d1", _grid_mdhisto_data(), kind="mdhisto")
    d2 = DatasetEntry("d2", _grid_mdhisto_data(), kind="mdhisto")
    sub = DatasetGroup("Group1", datasets=[d2])
    group = DataGroup("Datagroup1", datasets=[d1], subgroups=[sub])
    explorer = MetallixProjectExplorer(MetallixProject([group]))

    datasets_item = explorer.tree.topLevelItem(0).child(0)
    assert [datasets_item.child(i).text(0) for i in range(datasets_item.childCount())] == ["d1", "Group1"]
    subgroup_item = datasets_item.child(1)
    assert [subgroup_item.child(i).text(0) for i in range(subgroup_item.childCount())] == ["Masks", "d2"]

    # Bulk-set scale on the subgroup overwrites all descendants.
    explorer.tree.setCurrentItem(subgroup_item)
    explorer._set_group_bulk_value("scale_factor", "5")
    assert [d.scale_factor for d in sub.iter_datasets()] == [5.0]

    # Top group bulk-set covers every dataset; a divergent child blanks the box.
    top_item = explorer.tree.topLevelItem(0)
    explorer.tree.setCurrentItem(top_item)
    explorer._set_group_bulk_value("fit_weight", "3")
    assert [d.fit_weight for d in group.iter_datasets()] == [3.0, 3.0]
    d1.fit_weight = 2.0
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer._sync_details()
    assert explorer.group_fit_weight_edit.text() == ""  # mixed


def test_dataset_scale_factor_scales_viewed_data_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    data = _grid_mdhisto_data()
    dataset = DatasetEntry("scan", data, kind="mdhisto")
    group = DataGroup("Datagroup1", datasets=[dataset])
    explorer = MetallixProjectExplorer(MetallixProject([group]))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(0).child(0))

    scale_spin = explorer.window.findChild(QtWidgets.QDoubleSpinBox, "dataset_scale_factor")
    assert scale_spin is not None
    assert scale_spin.value() == 1.0
    scale_spin.setValue(3.0)
    assert dataset.scale_factor == 3.0

    viewed = dataset_for_slice_viewer(dataset)
    np.testing.assert_allclose(viewed.signal, np.asarray(data.signal, dtype=float) * 3.0)
    np.testing.assert_allclose(viewed.errors, np.asarray(data.errors, dtype=float) * 3.0)

    # Round-trip through JSON using a lazy placeholder dataset.
    save_group = DataGroup("Datagroup2")
    placeholder = import_dataset_paths(save_group, [tmp_path / "scan.nxs"])[0]
    placeholder.scale_factor = 3.0
    path = tmp_path / "proj.mtlx"
    save_project(MetallixProject([save_group]), path)
    assert json.loads(path.read_text())["data_groups"][0]["datasets"][0]["scale_factor"] == 3.0
    assert load_project(path).data_groups[0].datasets[0].scale_factor == 3.0


def test_point_list_scale_and_susceptibility_transforms():
    from metallix.project_gui import point_list_config, prepared_point_list_data

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
    from metallix.project_gui import (
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
    assert prepared.unit("q") == "Angstrom^-1"
    assert prepared.unit("d") == "Angstrom"

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
    from metallix.project_gui import point_list_config, prepared_point_list_data

    group = DataGroup("Datagroup1")
    import_dataset_paths(group, [MPMS_FILE], data_type="magnetization")
    explorer = MetallixProjectExplorer(MetallixProject([group]))
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
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

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

    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][0]),
        ydata=float(view["signal"][0]),
    )
    viewer._on_motion(event)

    assert viewer.cursor_hkle_label.text() == "(H, K, L, E) = (0, 0, 0, nan)"
    assert viewer.cursor_intensity_label.text().startswith("I = ")


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
    explorer = MetallixProjectExplorer(MetallixProject([group]))

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
