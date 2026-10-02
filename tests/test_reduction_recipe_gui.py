"""Schema controls reproduce public recipes without loading scientific payloads."""

import copy
import json

import pytest

from nfit import DataGroup, NfitProject, NfitProjectExplorer, raw_dgs_dataset_group
from tests.test_raw_dgs import _write_raw_dgs


@pytest.fixture
def recipe_explorer(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    files = [tmp_path / f"SEQ_{run}.nxs" for run in (1, 2)]
    for source in files:
        _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group(files)
    explorer = NfitProjectExplorer(NfitProject([DataGroup("sample", subgroups=[group])]))
    explorer._refresh_tree(select_dataset_group=group)
    yield explorer, group
    explorer.has_unsaved_changes = False
    explorer.window.close()


def test_every_schema_setting_has_a_tooltip_and_override_scope(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import reduction_settings_schema

    explorer, group = recipe_explorer
    assert all(dataset.data is None for dataset in group.datasets)
    for field in reduction_settings_schema(group):
        editor = explorer.details_widget.findChild(QtWidgets.QWidget, f"raw_dgs_{field.key}")
        assert editor is not None and editor.toolTip()
        run_editor = explorer.details_widget.findChild(QtWidgets.QWidget, f"raw_dgs_run_{field.key}")
        assert (run_editor is not None) == field.per_run
        if field.per_run:
            inherit = explorer.details_widget.findChild(
                QtWidgets.QCheckBox, f"raw_dgs_run_{field.key}_inherit"
            )
            assert inherit is not None and inherit.isChecked() and inherit.toolTip()
    for name in ("raw_dgs_bad_pulse_threshold", "raw_dgs_he3_detector_efficiency_correction"):
        assert explorer.details_widget.findChild(QtWidgets.QWidget, name) is not None
    assert all(dataset.data is None for dataset in group.datasets)


def test_negative_time_zero_uses_explicit_automatic_mode(recipe_explorer):
    from PySide6 import QtWidgets

    explorer, group = recipe_explorer
    automatic = explorer.details_widget.findChild(QtWidgets.QCheckBox, "raw_dgs_t0_override_automatic")
    value = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "raw_dgs_t0_override")
    assert automatic.isChecked() and not value.isEnabled()
    automatic.setChecked(False)
    value.setValue(-240.5)
    assert group.metadata["raw_dgs"]["t0_override"] == -240.5
    automatic.setChecked(True)
    assert group.metadata["raw_dgs"]["t0_override"] is None


def test_run_override_is_explicit_and_reinherits_shared_setting(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import effective_reduction_config

    explorer, group = recipe_explorer
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, "raw_dgs_run_t0_override_inherit")
    inherit.setChecked(False)
    automatic = explorer.details_widget.findChild(QtWidgets.QCheckBox, "raw_dgs_run_t0_override_automatic")
    value = explorer.details_widget.findChild(QtWidgets.QDoubleSpinBox, "raw_dgs_run_t0_override")
    automatic.setChecked(False)
    value.setValue(-120.25)
    assert effective_reduction_config(group, group.datasets[0])["t0_override"] == -120.25
    assert effective_reduction_config(group, group.datasets[1])["t0_override"] is None
    inherit = explorer.details_widget.findChild(QtWidgets.QCheckBox, "raw_dgs_run_t0_override_inherit")
    inherit.setChecked(True)
    assert "t0_override" not in group.datasets[0].metadata.get("reduction_overrides", {})
    assert effective_reduction_config(group, group.datasets[0])["t0_override"] is None


def test_complete_recipe_clipboard_replays_without_qt(recipe_explorer):
    from PySide6 import QtWidgets

    from nfit.reduction_recipes import (
        effective_reduction_config,
        reduction_recipe_script,
        set_reduction_settings,
    )

    explorer, group = recipe_explorer
    set_reduction_settings(group, {"t0_override": -80, "bad_pulse_threshold": 90})
    set_reduction_settings(group, {"t0_override": -90}, dataset_ids=[group.datasets[1].id])
    button = explorer.details_widget.findChild(QtWidgets.QPushButton, "raw_dgs_copy_reduction_recipe_script")
    assert button is not None and button.toolTip()
    button.click()
    script = QtWidgets.QApplication.clipboard().text()
    assert script == reduction_recipe_script(group)
    namespace = {"group": copy.deepcopy(group)}
    exec(script, namespace)
    rebuilt = namespace["group"]
    assert json.loads(json.dumps(effective_reduction_config(rebuilt))) == json.loads(json.dumps(effective_reduction_config(group)))
    for actual, expected in zip(rebuilt.datasets, group.datasets, strict=True):
        assert json.loads(json.dumps(effective_reduction_config(rebuilt, actual))) == json.loads(json.dumps(effective_reduction_config(group, expected)))
    assert all(dataset.data is None for dataset in group.datasets)
