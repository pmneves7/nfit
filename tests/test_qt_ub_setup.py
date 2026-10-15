"""Every orientation editor uses the same coherent, transactional UB popup."""

import copy

import numpy as np
import pytest

from nfit import project_gui
from nfit.qt_ub_setup import UBSetupDialog, ub_matrix_widget
from nfit.ub_orientation import lattice_orientation_from_ub, ub_from_lattice_orientation


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


@pytest.fixture
def warnings(monkeypatch):
    from PySide6 import QtWidgets

    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    return messages


def assert_coherent(dialog, matrix):
    cell, u, v = lattice_orientation_from_ub(matrix)
    assert dialog._lattice() == pytest.approx(cell)
    np.testing.assert_allclose(dialog._vector("u"), u, atol=1e-10)
    np.testing.assert_allclose(dialog._vector("v"), v, atol=1e-10)


def test_popup_initial_matrix_overrides_stale_cell_and_vectors(app):
    matrix = np.diag([0.25, 0.2, 1 / 6])
    dialog = UBSetupDialog(ub=matrix, lattice={"a": 99}, u=[1, 0, 0], v=[0, 1, 0])
    assert_coherent(dialog, matrix)
    np.testing.assert_allclose(dialog._vector("u"), [0, 0, 6])
    np.testing.assert_allclose(dialog._vector("v"), [4, 0, 0])
    dialog._accept()
    np.testing.assert_allclose(dialog.result["ub_matrix"], matrix)


def test_cell_edit_apply_calculates_before_accepting(app):
    dialog = UBSetupDialog(ub=np.eye(3))
    dialog.lattice_edits["a"].setText("4")
    assert not dialog.matrix_table.isEnabled()
    expected = ub_from_lattice_orientation(dialog._lattice(), dialog._vector("u"), dialog._vector("v"))
    dialog._accept()
    assert dialog.matrix_table.isEnabled()
    np.testing.assert_allclose(dialog.result["ub_matrix"], expected, atol=1e-11)
    assert_coherent(dialog, expected)


@pytest.mark.parametrize("edit", ["invalid_cell", "collinear_uv", "invalid_matrix"])
def test_invalid_edits_cannot_apply_previous_matrix(app, warnings, edit):
    from PySide6 import QtWidgets

    dialog = UBSetupDialog(ub=np.eye(3))
    if edit == "invalid_cell":
        dialog.lattice_edits["a"].setText("-4")
    elif edit == "collinear_uv":
        for index in range(3):
            dialog.orientation_edits["v", index].setText(dialog.orientation_edits["u", index].text())
    else:
        dialog.matrix_table.item(2, 2).setText("0")
    dialog._accept()
    assert dialog.result is None
    assert dialog.dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    assert warnings


def test_manual_matrix_edit_refreshes_cell_and_vectors(app):
    dialog = UBSetupDialog(ub=np.eye(3))
    dialog.matrix_table.item(0, 0).setText("0.25")
    assert_coherent(dialog, np.diag([0.25, 1, 1]))
    dialog._accept()
    assert dialog.result["lattice_parameters"]["a"] == pytest.approx(4)


@pytest.mark.parametrize("kind", ["nexus", "isaw"])
def test_import_replaces_all_fields_without_loading_data(app, monkeypatch, warnings, kind):
    import nfit.qt_ub_setup as gui

    matrix = np.diag([0.25, 0.2, 1 / 6])
    dialog = UBSetupDialog(ub=np.eye(3))
    dialog.lattice_edits["a"].setText("invalid pending edit")
    monkeypatch.setattr(gui, "get_open_file_name", lambda *_args: ("orientation", ""))
    monkeypatch.setattr(gui, f"read_{kind}_ub", lambda _path: (matrix, {"a": 99}))
    getattr(dialog, f"_load_{kind}")()
    assert_coherent(dialog, matrix)
    assert dialog.matrix_table.isEnabled()
    assert not warnings


def test_import_error_preserves_current_setup(app, monkeypatch, warnings):
    import nfit.qt_ub_setup as gui

    dialog = UBSetupDialog(ub=np.eye(3))
    monkeypatch.setattr(gui, "get_open_file_name", lambda *_args: ("broken.nxs", ""))

    def invalid(_path):
        raise ValueError("different orientation matrices at /one and /two")

    monkeypatch.setattr(gui, "read_nexus_ub", invalid)
    dialog._load_nexus()
    np.testing.assert_allclose(dialog._matrix(), np.eye(3))
    assert "/one and /two" in warnings[0]
    assert dialog.result is None


def test_cancel_does_not_mutate_caller_or_invoke_callback(app, monkeypatch):
    from PySide6 import QtWidgets

    matrix = np.eye(3).tolist()
    before = copy.deepcopy(matrix)
    changes = []
    widget = ub_matrix_widget(matrix, on_changed=changes.append)
    monkeypatch.setattr(UBSetupDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Rejected)
    widget.findChild(QtWidgets.QPushButton, "open_ub_setup").click()
    assert matrix == before
    assert not changes


def test_all_schema_ub_surfaces_use_shared_popup(app, monkeypatch):
    from PySide6 import QtWidgets

    from nfit.reduction_recipe_gui import _field_editor
    from nfit.reduction_recipes import reduction_settings_schema

    changes = []

    def accepted(self):
        self._set_matrix(np.diag([0.5, 0.25, 0.2]))
        self._accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(UBSetupDialog, "exec", accepted)
    for format_name in ("raw-direct-geometry-nexus", "mantid-mdevent", "macs-step-nexus", "corelli-correlation-nexus"):
        field = next(field for field in reduction_settings_schema(format_name) if field.key == "ub_matrix")
        widget = _field_editor(field, field.default, object_name="test_ub", on_changed=changes.append)
        assert not widget.findChildren(QtWidgets.QLineEdit)
        button = widget.findChild(QtWidgets.QPushButton, "test_ub_setup")
        assert button is not None and button.toolTip()
        button.click()
        np.testing.assert_allclose(changes[-1], np.diag([0.5, 0.25, 0.2]))
        display = widget.findChild(QtWidgets.QLabel, "test_ub_display")
        assert "0.5" in display.text()
        if format_name == "macs-step-nexus":
            widget.findChild(QtWidgets.QPushButton, "test_ub_source").click()
            assert changes[-1] is None
            assert "from source" in display.text()
        else:
            assert widget.findChild(QtWidgets.QPushButton, "test_ub_source") is None


def test_project_facade_keeps_shared_implementations():
    from nfit import ub_orientation

    assert project_gui.UBSetupDialog is UBSetupDialog
    assert project_gui.read_isaw_ub is ub_orientation.read_isaw_ub
    assert project_gui.write_isaw_ub is ub_orientation.write_isaw_ub
    assert project_gui.ub_from_lattice_orientation is ub_orientation.ub_from_lattice_orientation


def test_popup_interactable_controls_have_tooltips(app):
    from PySide6 import QtWidgets

    dialog = UBSetupDialog(ub=np.eye(3))
    for kind in (QtWidgets.QLineEdit, QtWidgets.QPushButton, QtWidgets.QTableWidget):
        for control in dialog.dialog.findChildren(kind):
            assert control.toolTip(), control.objectName() or control.text()


def test_untouched_apply_preserves_float64_matrix_exactly(app):
    matrix = ub_from_lattice_orientation(
        {"a": 4.44, "b": 5.67, "c": 6.21, "alpha": 79.1, "beta": 82.3, "gamma": 113.6},
        [1.1, 0.2, 0.4], [-0.6, 1.3, 0.2],
    )
    dialog = UBSetupDialog(ub=matrix)
    dialog._accept()
    np.testing.assert_array_equal(dialog.result["ub_matrix"], matrix)


def test_failed_import_derivation_keeps_entire_editor_unchanged(app, monkeypatch, warnings):
    import nfit.qt_ub_setup as gui

    dialog = UBSetupDialog(ub=np.eye(3))
    monkeypatch.setattr(gui, "get_open_file_name", lambda *_args: ("extreme.mat", ""))
    monkeypatch.setattr(gui, "read_isaw_ub", lambda _path: (np.eye(3) * 1e-320, {}))
    dialog._load_isaw()
    np.testing.assert_array_equal(dialog._matrix(), np.eye(3))
    assert_coherent(dialog, np.eye(3))
    assert warnings


def test_left_handed_matrix_is_preserved_and_calculation_notice_is_visible(app):
    matrix = np.diag([-0.5, 0.25, 0.2])
    dialog = UBSetupDialog(ub=matrix)
    assert "left-handed" in dialog.status.text()
    dialog._accept()
    np.testing.assert_array_equal(dialog.result["ub_matrix"], matrix)


def test_missing_matrix_uses_existing_cell_as_explicit_proposal(app):
    cell = dict(a=4.44, b=5.11, c=6.23, alpha=90, beta=90, gamma=90)
    dialog = UBSetupDialog(ub=None, lattice=cell)
    assert dialog._lattice() == pytest.approx(cell)
    assert "Proposed UB" in dialog.status.text()
    dialog._accept()
    assert dialog.result["lattice_parameters"] == pytest.approx(cell)


def test_native_popup_cancel_and_run_apply_use_shared_owner_without_loading(app, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    from nfit import DataGroup, NfitProject, raw_dgs_dataset_group
    from nfit.project_ub_setup import get_ub_setup
    from tests.test_raw_dgs import _write_raw_dgs

    path = tmp_path / "SEQ_1.nxs"
    _write_raw_dgs(path)
    native = raw_dgs_dataset_group([path])
    root = DataGroup("sample", subgroups=[native])
    explorer = project_gui.NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_dataset_group=native)
    original = copy.deepcopy(native.metadata)
    monkeypatch.setattr(UBSetupDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Rejected)
    assert not explorer._open_ub_setup(root, native)
    assert native.metadata == original
    assert all(dataset.data is None for dataset in native.datasets)
    matrix = np.diag([0.5, 0.25, 0.2])

    def accepted(self):
        self._set_matrix(matrix)
        self._accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(UBSetupDialog, "exec", accepted)
    assert explorer._open_ub_setup(root, native.datasets[0])
    np.testing.assert_array_equal(native.metadata["raw_dgs"]["ub_matrix"], matrix)
    np.testing.assert_array_equal(get_ub_setup(root, native)["ub_matrix"], matrix)
    assert all(dataset.data is None for dataset in native.datasets)
    assert "ub_setup" not in native.datasets[0].metadata
    assert explorer._set_reduction_recipe_value(native, "ub_matrix", np.eye(3).tolist())
    np.testing.assert_array_equal(get_ub_setup(root, native)["ub_matrix"], np.eye(3))
    app.processEvents()
    explorer.has_unsaved_changes = False
    explorer.window.close()


@pytest.mark.parametrize("calculate", [False, True])
def test_macs_schema_popup_uses_mantid_but_preserves_native_refplane_coordinates(app, tmp_path, monkeypatch, calculate):
    from PySide6 import QtWidgets

    from nfit.macs import import_macs_nexus
    from nfit.reduction_recipe_gui import _field_editor
    from nfit.reduction_recipes import reduction_settings_schema
    from nfit.ub_orientation import mantid_ub_from_ipns
    from tests.test_macs import _write_macs_nexus

    path = _write_macs_nexus(tmp_path / "orientation.nxs.ng0")
    original = import_macs_nexus(path, {"stream": "spec"})
    native = np.asarray(original.metadata["ub_matrix"])
    canonical = mantid_ub_from_ipns(native)
    changes = []

    def accepted(self):
        np.testing.assert_array_equal(self._matrix(), canonical)
        np.testing.assert_allclose(self._matrix() @ [1, 1, 0], [0, 0, np.sqrt(2) / 8.24], atol=1e-14)
        np.testing.assert_allclose(self._matrix() @ [0, 0, 1], [1 / 8.24, 0, 0], atol=1e-14)
        if calculate:
            assert self._calculate()
        self._accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(UBSetupDialog, "exec", accepted)
    field = next(field for field in reduction_settings_schema("macs-step-nexus") if field.key == "ub_matrix")
    widget = _field_editor(field, native, object_name="macs_ub_matrix", on_changed=changes.append, ub_frame="ipns")
    widget.findChild(QtWidgets.QPushButton, "macs_ub_matrix_setup").click()
    assert len(changes) == 1
    if not calculate:
        np.testing.assert_array_equal(changes[0], native)
    else:
        np.testing.assert_allclose(changes[0], native, atol=1e-14)
    replayed = import_macs_nexus(path, {"stream": "spec", "ub_matrix": changes[0]})
    for coordinate in ("H", "K", "L"):
        np.testing.assert_allclose(getattr(replayed, coordinate), getattr(original, coordinate), atol=1e-12)


@pytest.mark.parametrize("explicit_override", [False, True])
def test_macs_reduction_panel_callback_preserves_native_frame(app, tmp_path, monkeypatch, explicit_override):
    from PySide6 import QtWidgets

    from nfit import DataGroup, NfitProject
    from nfit.project_imports import import_dataset_paths
    from nfit.project_ub_setup import get_ub_setup
    from nfit.reduction_recipes import effective_reduction_config, set_reduction_settings
    from nfit.ub_orientation import mantid_ub_from_ipns
    from tests.test_macs import _write_macs_nexus

    path = _write_macs_nexus(tmp_path / "orientation.nxs.ng0")
    root = DataGroup("MACS")
    import_dataset_paths(root, [path], data_type="single_crystal_inelastic")
    owner = root.subgroups[0]
    native = np.asarray(owner.metadata["ub_matrix"])
    if explicit_override:
        set_reduction_settings(owner, {"ub_matrix": native.tolist()})
    explorer = project_gui.NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_dataset_group=owner)

    def accepted(self):
        np.testing.assert_array_equal(self._matrix(), mantid_ub_from_ipns(native))
        self._accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(UBSetupDialog, "exec", accepted)
    if not explicit_override:
        source_display = explorer.details_widget.findChild(QtWidgets.QLabel, "macs_ub_matrix_display")
        assert "from source" in source_display.text()
        assert "not set" not in source_display.text()
    explorer.details_widget.findChild(QtWidgets.QPushButton, "macs_ub_matrix_setup").click()
    np.testing.assert_array_equal(effective_reduction_config(owner)["ub_matrix"], native)
    np.testing.assert_array_equal(get_ub_setup(root, owner)["ub_matrix"], mantid_ub_from_ipns(native))
    app.processEvents()
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_macs_import_passes_explicit_ipns_context_to_shared_field(app, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    import nfit.reduction_recipe_gui as recipe_gui
    from nfit.project_import_dialogs import prompt_macs_nexus_options

    original = recipe_gui._field_editor
    frames = []

    def capture(field, value, **kwargs):
        if field.key == "ub_matrix":
            frames.append(kwargs["ub_frame"])
        return original(field, value, **kwargs)

    monkeypatch.setattr(recipe_gui, "_field_editor", capture)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Rejected)
    assert prompt_macs_nexus_options(None, [tmp_path / "source.nxs.ng0"]) is False
    assert frames == ["ipns"]
