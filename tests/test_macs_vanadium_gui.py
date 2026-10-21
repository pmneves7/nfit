"""Multiple calibration scans remain editable and replayable through public recipes."""

import json

from nfit import DataGroup
from nfit.project_imports import import_dataset_paths
from nfit.reduction_recipes import (
    effective_reduction_config,
    reduction_recipe_script,
    reduction_settings_schema,
    set_reduction_settings,
)
from tests.test_macs import _write_macs_nexus
from tests.test_macs_vanadium import _scan


def test_macs_vanadium_file_picker_and_recipe_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.reduction_recipe_gui import build_reduction_recipe_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    root = DataGroup("MACS")
    import_dataset_paths(root, [_write_macs_nexus(tmp_path / "sample.ng0")])
    group = root.subgroups[0]
    group.datasets[0].unload_data()
    chosen = [str(tmp_path / "kidney 1, scan.nxs"), str(tmp_path / "kidney 2.nxs")]
    for kidney, path in zip((0., 40.), chosen, strict=True):
        _scan(path, kidney=kidney, fixed=3.7, final=False)
    selections = iter((chosen, []))
    monkeypatch.setattr("nfit.file_dialogs.get_open_file_names",
                        lambda *args, **kwargs: (next(selections), ""))

    panel = build_reduction_recipe_panel(
        group,
        on_shared_changed=lambda key, value: set_reduction_settings(group, {key: value}),
        on_run_changed=lambda *args: None,
    )
    try:
        files = panel.findChild(QtWidgets.QLineEdit, "macs_vanadium_files")
        browse = panel.findChild(QtWidgets.QPushButton, "macs_vanadium_files_browse")
        angle = panel.findChild(QtWidgets.QDoubleSpinBox, "macs_vanadium_a4_mask_deg")
        assert files.toolTip() and browse.toolTip() and angle.toolTip()
        assert angle.value() == 15.
        browse.click()
        assert json.loads(files.text()) == chosen
        assert effective_reduction_config(group)["vanadium_files"] == chosen
        # Cancel leaves the existing selection intact.
        browse.click()
        assert json.loads(files.text()) == chosen
        angle.setValue(18.)
        namespace = {}
        exec(reduction_recipe_script(group), namespace)
        rebuilt = effective_reduction_config(namespace["group"])
        original = effective_reduction_config(group)
        assert all(rebuilt[field.key] == original[field.key]
                   for field in reduction_settings_schema(group))
        assert effective_reduction_config(group)["vanadium_a4_mask_deg"] == 18.
        files.setText(json.dumps(chosen[:1]))
        files.editingFinished.emit()
        assert effective_reduction_config(group)["vanadium_files"] == chosen[:1]
        files.clear()
        files.editingFinished.emit()
        assert effective_reduction_config(group)["vanadium_files"] == []
        assert group.datasets[0].data is None
    finally:
        panel.close()
        app.processEvents()
