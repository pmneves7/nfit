"""Saving and refreshing fit details must not replay historical state."""

import copy

import pytest

from nfit import project_gui
from nfit.pipeline import DataGroup, DatasetEntry, FitTimelineEntry
from nfit.project_gui import NfitProjectExplorer, create_mask, create_model_component
from nfit.project_io import NfitProject
from tests.project_gui_test_support import _tiny_mdhisto_data


@pytest.fixture
def fit_explorer(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    explorers = []

    def build(kind="initial"):
        dataset = DatasetEntry("scan", _tiny_mdhisto_data(1.0))
        source = tmp_path / "scan.npz"
        project_gui.save_dataset_file(dataset, source, use_view=False)
        dataset = project_gui.dataset_entry_from_path(source)
        create_mask(dataset)
        group = DataGroup("workspace", datasets=[dataset])
        model = create_model_component(group)
        model.parameters["constant"] = 1.0
        snapshot = project_gui.snapshot_data_group_state(group)
        fit = FitTimelineEntry(kind.title(), kind=kind, snapshot=snapshot)
        group.fits = [fit]
        explorer = NfitProjectExplorer(NfitProject([group]))
        monkeypatch.setattr(explorer, "_confirm_save_before_closing_project", lambda: True)
        explorer._refresh_tree(select_group=group, select_fit=fit, refresh_viewers=False)
        explorers.append(explorer)
        return explorer, group, fit

    yield build
    for explorer in explorers:
        explorer.has_unsaved_changes = False
        explorer.window.close()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("kind", ["initial", "result", "current"])
@pytest.mark.parametrize("action", ["save", "save_as"])
def test_save_with_fit_selected_stays_clean_without_replaying_snapshot(
    fit_explorer, monkeypatch, tmp_path, kind, action
):
    explorer, group, fit = fit_explorer(kind)
    path = tmp_path / f"{kind}-{action}.nfit"
    if action == "save":
        explorer.project_path = path
    else:
        monkeypatch.setattr(project_gui, "get_save_file_name", lambda *_args: (str(path), ""))
    restorations = []
    original_restore = project_gui.restore_data_group_state

    def restore(*args):
        restorations.append(args)
        return original_restore(*args)

    monkeypatch.setattr(project_gui, "restore_data_group_state", restore)
    model = next(iter(group.models.values()))
    mask = group.datasets[0].masks[0]
    fit_snapshot = copy.deepcopy(fit.snapshot)
    viewer_refreshes = []
    monkeypatch.setattr(explorer, "refresh_open_slice_viewers", lambda: viewer_refreshes.append(True))
    explorer._mark_dirty()

    assert getattr(explorer, action)()
    for _ in range(3):
        explorer.app.processEvents()

    assert not explorer.has_unsaved_changes
    assert not explorer.window.windowTitle().endswith("*")
    assert restorations == []
    assert viewer_refreshes == []
    assert next(iter(group.models.values())) is model
    assert group.datasets[0].masks[0] is mask
    assert fit.snapshot == fit_snapshot
    assert explorer._fit_entry_for_item(explorer.tree.currentItem()) is fit
    restored = project_gui.load_project(path)
    assert restored.data_groups[0].fits[0].optimizer_config == fit.optimizer_config


def test_fit_detail_refresh_preserves_live_state_but_explicit_selection_restores(
    fit_explorer, monkeypatch, tmp_path
):
    explorer, group, fit = fit_explorer()
    group.active_fit_path = [0]
    model_name = next(iter(group.models))
    group.models[model_name].parameters["constant"] = 2.0
    path = tmp_path / "authoritative-live-state.nfit"
    project_gui.save_project(explorer.project, path)
    assert explorer.open_project_path(path, remember=False)
    group = explorer.project.data_groups[0]
    fit = group.fits[0]
    assert explorer._fit_entry_for_item(explorer.tree.currentItem()) is fit

    explorer._sync_details()
    explorer._refresh_tree(refresh_viewers=False)
    for _ in range(3):
        explorer.app.processEvents()

    assert group.models[model_name].parameters["constant"] == 2.0
    assert not explorer.has_unsaved_changes
    assert explorer.save()
    assert project_gui.load_project(path).data_groups[0].models[model_name].parameters["constant"] == 2.0
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(2).child(0))
    assert group.models[model_name].parameters["constant"] == 1.0
    assert explorer.has_unsaved_changes


def test_fit_user_edit_remains_dirty_after_detail_refresh(fit_explorer, tmp_path):
    explorer, group, fit = fit_explorer()
    explorer.has_unsaved_changes = False
    explorer.fit_optimizer_config_editor.setText('{"max_nfev": 42}')
    explorer.fit_optimizer_config_editor.editingFinished.emit()
    assert fit.optimizer_config == {"max_nfev": 42}
    assert explorer.has_unsaved_changes

    explorer._sync_details()
    assert fit.optimizer_config == {"max_nfev": 42}
    assert explorer.has_unsaved_changes
    explorer.project_path = tmp_path / "fit-edit.nfit"
    assert explorer.save()
    assert not explorer.has_unsaved_changes
    assert project_gui.load_project(explorer.project_path).data_groups[0].fits[0].optimizer_config == {
        "max_nfev": 42
    }


def test_worker_save_with_fit_selected_stays_clean(fit_explorer, monkeypatch, tmp_path):
    explorer, _group, fit = fit_explorer()
    explorer.project_path = tmp_path / "worker-save.nfit"
    explorer.window.show()
    explorer._interactive = True
    explorer._mark_dirty()
    monkeypatch.setattr(
        project_gui, "restore_data_group_state",
        lambda *_args: pytest.fail("save refresh replayed a historical snapshot"),
    )
    try:
        assert explorer.save()
        for _ in range(3):
            explorer.app.processEvents()
        assert not explorer.has_unsaved_changes
        assert explorer._fit_entry_for_item(explorer.tree.currentItem()) is fit
    finally:
        explorer._interactive = False
