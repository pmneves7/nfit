import pytest


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit import dataset_criterion_mask
    from nfit.dataset_criterion_gui import dataset_criterion_panel
    from tests.test_dataset_criteria import metadata_group

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    group = metadata_group()
    mask = dataset_criterion_mask(group, channel="metadata", source="metadata/temperature")
    jobs, changes = [], []
    widget = dataset_criterion_panel(group, mask, changed=lambda: changes.append(True),
                                     start_task=lambda **job: jobs.append(job))
    yield widget, group, mask, jobs, changes, app
    widget.deleteLater()
    app.processEvents()


def test_controls_have_tooltips_and_use_public_preview(panel, monkeypatch):
    from PySide6 import QtWidgets
    widget, _group, mask, jobs, _changes, _app = panel
    for key in ("channel", "energy_min", "energy_max", "source", "statistic", "left_value",
                "left_operator", "right_value", "right_operator", "yscale", "calculate", "invert", "additive"):
        control = widget.findChild(QtWidgets.QWidget, f"dataset_criterion_{key}")
        assert control is not None and control.toolTip()
    widget.findChild(QtWidgets.QPushButton, "dataset_criterion_calculate").click()
    assert len(jobs) == 1
    rows = jobs[0]["task"](lambda _progress: None)
    assert not mask.metadata  # Async result is published on successful GUI delivery only.
    jobs[0]["on_success"](rows)
    assert len(mask.metadata["dataset_criterion_preview"]) == 3
    import nfit.dataset_criteria as api
    monkeypatch.setattr(api, "criterion_value", lambda *a, **k: pytest.fail("read source"))
    bound = widget.findChild(QtWidgets.QLineEdit, "dataset_criterion_right_value")
    bound.setText("2.5")
    assert widget.criterion_lines[1].get_visible()
    assert list(widget.criterion_lines[1].get_ydata()) == [2.5, 2.5]
    bound.editingFinished.emit()
    assert mask.parameters["right_value"] == 2.5
    widget.findChild(QtWidgets.QComboBox, "dataset_criterion_yscale").setCurrentText("log")
    assert widget.criterion_axes.get_yscale() == "log"
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    assert widget.findChild(NavigationToolbar2QT) is not None


def test_invalid_window_does_not_start_job_and_edits_preserve_enabled(panel):
    from PySide6 import QtWidgets
    widget, _group, mask, jobs, _changes, _app = panel
    widget.findChild(QtWidgets.QLineEdit, "dataset_criterion_right_value").setText("2")
    widget.findChild(QtWidgets.QLineEdit, "dataset_criterion_right_value").editingFinished.emit()
    assert mask.enabled
    widget.findChild(QtWidgets.QLineEdit, "dataset_criterion_energy_min").setText("nan")
    widget.findChild(QtWidgets.QPushButton, "dataset_criterion_calculate").click()
    assert not jobs


def test_explorer_group_mask_installs_panel_and_marks_selection_stale(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit import DataGroup, DatasetGroup, NfitProject, dataset_criterion_mask
    from nfit.project_gui import NfitProjectExplorer
    from tests.test_dataset_criteria import metadata_group
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    child = DatasetGroup("runs", datasets=metadata_group().datasets)
    root = DataGroup("workspace", subgroups=[child])
    mask = dataset_criterion_mask(child, channel="metadata", source="metadata/temperature")
    explorer = NfitProjectExplorer(NfitProject([root]))
    explorer._refresh_tree(select_group=root, select_mask=mask)
    control = explorer.window.findChild(QtWidgets.QLineEdit, "dataset_criterion_right_value")
    assert control is not None
    control.setText("2")
    control.editingFinished.emit()
    assert mask.parameters["right_value"] == 2
    assert explorer.has_unsaved_changes
    explorer.has_unsaved_changes = False
    explorer.window.close()
    app.processEvents()


def test_available_types_respect_group_only_scope():
    from nfit import DatasetEntry, DatasetGroup
    from nfit.project_gui import available_mask_types, create_group_mask, create_mask
    assert "dataset_criterion" not in available_mask_types()
    assert "dataset_criterion" in available_mask_types(scope="group")
    assert create_group_mask(DatasetGroup("runs"), type="dataset_criterion").type == "dataset_criterion"
    with pytest.raises(ValueError, match="dataset groups"):
        create_mask(DatasetEntry("run", None), type="dataset_criterion")
