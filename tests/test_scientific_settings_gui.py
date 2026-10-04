"""Scientific controls describe the estimator that the public adapter executes."""

import copy
import json

import pytest

from nfit import DataGroup, NfitProject, NfitProjectExplorer


@pytest.fixture(params=["raw", "mde", "corelli"])
def native_explorer(request, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    source = tmp_path / "source.nxs"
    if request.param == "raw":
        from nfit import raw_dgs_dataset_group
        from tests.test_raw_dgs import _write_raw_dgs

        _write_raw_dgs(source, with_he3=True)
        group = raw_dgs_dataset_group([source])
    elif request.param == "mde":
        from nfit import mdevent_dataset_group
        from tests.test_mdevent import _write_mdevent

        _write_mdevent(source)
        group = mdevent_dataset_group(source)
    else:
        from nfit import corelli_dataset_group
        from tests.test_corelli import _write_corelli

        _write_corelli(source)
        group = corelli_dataset_group([source])
    from nfit.project_composites import data_group_composite_config

    data_group_composite_config(group).update(enabled=True, auto_rebin=False)
    explorer = NfitProjectExplorer(NfitProject([DataGroup("sample", subgroups=[group])]))
    explorer._refresh_tree(select_dataset_group=group)
    yield explorer, group, request.param
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_native_estimator_and_assignment_match_executed_adapter(native_explorer):
    from PySide6 import QtWidgets

    from nfit.project_composites import data_group_composite_config

    explorer, group, family = native_explorer
    config = data_group_composite_config(group)
    saved = copy.deepcopy(config)
    mean = explorer.details_widget.findChild(QtWidgets.QComboBox, "group_composite_mean_weighting")
    assert not mean.isEnabled()
    assert mean.count() == 1 and "pooled numerator / exposure" in mean.currentText()
    assert mean.currentData() == config["mean_weighting"]
    assert "Exposure is treated as fixed" in mean.toolTip()
    composite = next(box for box in explorer.details_widget.findChildren(QtWidgets.QGroupBox) if box.title() == "Composite dataset")
    assert "zero observed event variance" in composite.toolTip()
    for index in range(4):
        assignment = explorer.details_widget.findChild(QtWidgets.QComboBox, f"group_composite_axis_assignment_{index}")
        if family != "corelli":
            assert not assignment.isEnabled() and assignment.currentData() is False
            assert "not implemented" in assignment.toolTip()
        elif index < 3:
            assert assignment.isEnabled()
            assert assignment.findData(True) >= 0 and assignment.findData(False) >= 0
        else:
            assert not assignment.isEnabled() and "correlated energy channel" in assignment.toolTip()
    coverage = explorer.details_widget.findChild(QtWidgets.QLineEdit, "group_composite_minimum_coverage")
    assert "statistical precision" in coverage.toolTip()
    assert config == saved
    assert all(dataset.data is None for dataset in group.datasets)


def test_all_native_schema_settings_are_visible_and_scriptable(native_explorer):
    from PySide6 import QtCore, QtWidgets

    from nfit.reduction_recipes import effective_reduction_config, reduction_settings_schema

    explorer, group, family = native_explorer
    prefix = "mdevent" if family == "mde" else "raw_dgs"
    for field in reduction_settings_schema(group):
        editor = explorer.details_widget.findChild(QtWidgets.QWidget, f"{prefix}_{field.key}")
        assert editor is not None and editor.toolTip() == field.tooltip
        override = explorer.details_widget.findChild(QtWidgets.QWidget, f"{prefix}_run_{field.key}")
        assert (override is not None) == field.per_run
        if field.kind == "choice":
            assert isinstance(editor, QtWidgets.QComboBox)
            assert {editor.itemData(index) for index in range(editor.count())} == {value for value, _ in field.choices}
            assert all(editor.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole) for index in range(editor.count()))
    if family != "corelli":
        symmetry = explorer.details_widget.findChild(QtWidgets.QComboBox, f"{prefix}_symmetry_variance_policy")
        assert "Neither choice retains covariance between different bins" in symmetry.toolTip()
        symmetry.setCurrentIndex(symmetry.findData("within_bin_covariance"))
        explorer._refresh_tree(select_dataset_group=group)
    button = explorer.details_widget.findChild(QtWidgets.QPushButton, f"{prefix}_copy_reduction_recipe_script")
    button.click()
    script = QtWidgets.QApplication.clipboard().text()
    namespace = {"group": copy.deepcopy(group)}
    exec(script, namespace)
    assert json.loads(json.dumps(effective_reduction_config(namespace["group"]))) == json.loads(json.dumps(effective_reduction_config(group)))
    assert all(dataset.data is None for dataset in group.datasets)


def test_ordinary_measurement_target_choices_remain_actionable(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.project_rebin_panels import measurement_average_choices

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = QtWidgets.QComboBox()
    measurement_average_choices(combo)
    assert combo.isEnabled()
    assert [combo.itemData(index) for index in range(combo.count())] == ["inverse_variance", "uniform"]
    assert combo.toolTip()
    combo.close()
    assert app is not None
