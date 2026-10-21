"""Shared MACS background lists are portable recipes, never raw point edits."""

import copy
import json

import numpy as np
import pytest

from nfit import DataGroup
from nfit.macs import import_macs_nexus
from nfit.macs_reduction import MACS_DEFAULTS, MACS_REDUCTION_FORMAT
from nfit.project_imports import import_dataset_paths
from nfit.reduction_recipes import (
    effective_reduction_config,
    ensure_reduction_recipe,
    reduction_recipe_script,
    reduction_settings_schema,
    set_reduction_settings,
    validated_reduction_settings,
)
from tests.test_macs import _write_macs_nexus

BACKGROUND_KEYS = {key for key in MACS_DEFAULTS if key.startswith("background_")}


def _group(tmp_path):
    root = DataGroup("MACS")
    import_dataset_paths(root, [_write_macs_nexus(tmp_path / "sample.ng0")])
    return root.subgroups[0]


def test_shared_background_settings_preserve_raw_points_and_refresh_recipe(tmp_path):
    group = _group(tmp_path)
    dataset = group.datasets[0]
    points = dataset.data
    options = copy.deepcopy(dataset.metadata["import_options"])
    background = tmp_path / "background.ng0"
    background.write_bytes(b"background fingerprint")
    edit = set_reduction_settings(group, {
        "background_files": [background], "background_delta_e_tolerance_meV": 0.075,
        "background_method": "isotropic", "background_scale": 1.5,
        "background_q_bin_width": 0.1,
    })
    assert set(edit.changed_keys) == BACKGROUND_KEYS
    assert dataset.data is points
    assert dataset.metadata["import_options"] == options
    assert group.metadata["composite"]["stale"]
    config = effective_reduction_config(group, dataset)
    assert config["background_files"] == [str(background)]
    recipe = ensure_reduction_recipe(group)
    assert recipe["shared_defaults"]["background_delta_e_tolerance_meV"] == 0.075
    fingerprint = recipe["provenance"]["calibrations_by_run"][dataset.id]["background_files"][0]
    assert fingerprint["path"] == str(background)
    assert fingerprint["size_bytes"] == len(b"background fingerprint")
    namespace = {}
    exec(reduction_recipe_script(group), namespace)
    rebuilt = effective_reduction_config(namespace["group"])
    assert all(rebuilt[key] == config[key] for key in BACKGROUND_KEYS)


@pytest.mark.parametrize("default_options", [False, True])
def test_import_into_calibrated_group_keeps_backgrounds_shared(tmp_path, default_options):
    root = DataGroup("MACS")
    import_dataset_paths(root, [_write_macs_nexus(tmp_path / "first.ng0")])
    group = root.subgroups[0]
    background = str(tmp_path / "background.ng0")
    set_reduction_settings(group, {"background_files": [background]})
    options = {"macs_nexus": copy.deepcopy(MACS_DEFAULTS)} if default_options else None
    import_dataset_paths(root, [_write_macs_nexus(tmp_path / "second.ng0")], importer_options=options)
    assert len(group.datasets) == 2
    for dataset in group.datasets:
        assert not BACKGROUND_KEYS.intersection(dataset.metadata.get("reduction_overrides", {}))
        assert effective_reduction_config(group, dataset)["background_files"] == [background]


@pytest.mark.parametrize("key,value", [
    ("background_files", "a.nxs"), ("background_files", [""]),
    ("background_delta_e_tolerance_meV", -0.1),
    ("background_delta_e_tolerance_meV", float("nan")),
    ("background_method", "guess"), ("background_scale", float("inf")),
    ("background_q_bin_width", 0), ("background_q_bin_width", -0.1),
])
def test_background_settings_reject_invalid_values(key, value):
    with pytest.raises(ValueError):
        validated_reduction_settings(MACS_REDUCTION_FORMAT, {key: value})


def test_background_settings_are_shared_and_raw_import_remains_unsubtracted(tmp_path):
    schema = {field.key: field for field in reduction_settings_schema(MACS_REDUCTION_FORMAT)}
    for key in BACKGROUND_KEYS:
        assert not schema[key].per_run
        assert not schema[key].affects_reduced_events
        with pytest.raises(ValueError, match="shared group ownership"):
            validated_reduction_settings(MACS_REDUCTION_FORMAT, {key: MACS_DEFAULTS[key]}, per_run=True)
    source = _write_macs_nexus(tmp_path / "sample.ng0")
    plain = import_macs_nexus(source)
    configured = import_macs_nexus(source, {"background_files": [tmp_path / "missing.nxs"]})
    np.testing.assert_array_equal(configured.intensity, plain.intensity)
    np.testing.assert_array_equal(configured.sigma, plain.sigma)


def test_background_file_picker_has_shared_controls_and_uses_public_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.reduction_recipe_gui import build_reduction_recipe_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = _group(tmp_path)
    group.datasets[0].unload_data()
    selected = [str(tmp_path / "background 1.nxs"), str(tmp_path / "background 2.nxs")]
    def choose_files(*args, **kwargs):
        assert "*.bt9" in args[3] and "*.ng0" in args[3]
        return selected, ""

    monkeypatch.setattr("nfit.file_dialogs.get_open_file_names", choose_files)
    panel = build_reduction_recipe_panel(
        group, on_shared_changed=lambda key, value: set_reduction_settings(group, {key: value}),
        on_run_changed=lambda *args: None,
    )
    try:
        for key in BACKGROUND_KEYS:
            editor = panel.findChild(QtWidgets.QWidget, f"macs_{key}")
            assert editor is not None and editor.toolTip()
            assert panel.findChild(QtWidgets.QWidget, f"macs_run_{key}") is None
        files = panel.findChild(QtWidgets.QLineEdit, "macs_background_files")
        assert "Clear the list to disable this file-based correction" in files.toolTip()
        browse = panel.findChild(QtWidgets.QPushButton, "macs_background_files_browse")
        assert browse.toolTip()
        browse.click()
        assert json.loads(files.text()) == selected
        assert effective_reduction_config(group)["background_files"] == selected
        tolerance = panel.findChild(QtWidgets.QDoubleSpinBox, "macs_background_delta_e_tolerance_meV")
        tolerance.setValue(0.075)
        assert effective_reduction_config(group)["background_delta_e_tolerance_meV"] == 0.075
        files.clear()
        files.editingFinished.emit()
        assert effective_reduction_config(group)["background_files"] == []
        assert group.datasets[0].data is None
    finally:
        panel.close()
        app.processEvents()
