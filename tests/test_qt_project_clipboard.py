from __future__ import annotations

import pytest

from nfit import project_gui
from nfit.pipeline import DataGroup, DatasetEntry
from nfit.project_io import NfitProject
from tests.project_gui_test_support import _tiny_mdhisto_data


def test_external_clipboard_overrides_previous_local_copy_and_adopts_data(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail(str(args[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: pytest.fail(str(args[2])))

    from nfit.project_clipboard import make_payload
    from nfit.qt_project_clipboard import clipboard_descriptor, publish_project_clipboard

    group = DataGroup("Source", datasets=[DatasetEntry("run", _tiny_mdhisto_data(5.))])
    source = project_gui.NfitProjectExplorer(NfitProject([group]))
    destination = project_gui.NfitProjectExplorer(NfitProject())
    source.project_path = tmp_path / "source.nfit"
    try:
        source._refresh_tree(select_group=group)
        publish_project_clipboard(source, "group", [group],
                                  collect_binnings=project_gui._project_binning_artifacts, has_binnings=lambda _project: False)
        descriptor = clipboard_descriptor()
        assert descriptor["archive"]
        # This previous local selection cannot be pasted into the project root.
        destination._clipboard = make_payload("dataset", [DatasetEntry("old", None)])
        assert destination._can_paste_into_role("project", None)
        destination.paste_into_selection()
        copied = destination.project.data_groups[0].datasets[0]
        assert copied.data is None and copied.id != group.datasets[0].id
        source._project_clipboard_owner.cleanup()
        assert copied._project_artifact_source.path.exists()
        result = project_gui.dataset_for_slice_viewer(copied)
        assert (result.signal == group.datasets[0].data.signal).all()
    finally:
        QApplication.clipboard().clear()
        source.has_unsaved_changes = destination.has_unsaved_changes = False
        source.window.close()
        destination.window.close()


def test_interactive_copy_uses_worker_and_reports_success(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from nfit.qt_project_clipboard import publish_project_clipboard

    group = DataGroup("Source", datasets=[DatasetEntry("run", _tiny_mdhisto_data(5.))])
    explorer = project_gui.NfitProjectExplorer(NfitProject([group]))
    explorer.project_path = tmp_path / "source.nfit"
    calls = []
    explorer._interactive = True
    monkeypatch.setattr(explorer, "_start_background_task", lambda **kwargs: calls.append(kwargs))
    try:
        publish_project_clipboard(explorer, "group", [group],
                                  collect_binnings=project_gui._project_binning_artifacts, has_binnings=lambda _project: False)
        assert len(calls) == 1 and calls[0]["success_message"]
        calls[0]["on_success"](calls[0]["task"](lambda _event: None))
        assert explorer._project_clipboard_owner is not None
    finally:
        explorer._interactive = False
        QApplication.clipboard().clear()
        explorer.window.close()
