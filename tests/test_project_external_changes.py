"""External-change polling must not mistake publication by its own GUI for a conflict."""

from threading import Event

import pytest
from PySide6 import QtCore, QtWidgets

from nfit import project_gui as gui
from nfit.pipeline import DataGroup


def test_unchanged_file_poll_keeps_open_editor_popups(tmp_path):
    path = tmp_path / "project.nfit"
    gui.save_project(gui.NfitProject([DataGroup("saved")]), path)
    explorer = gui.NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    combo = QtWidgets.QComboBox(explorer.window)
    combo.addItems(["first", "second"])
    explorer.window.show()
    combo.showPopup()
    try:
        assert combo.view().isVisible()
        assert not explorer.check_for_external_project_change()
        assert combo.view().isVisible()
    finally:
        combo.hidePopup()
        explorer.window.close()


@pytest.mark.parametrize("operation", ["save", "save_as"])
def test_external_change_poll_during_own_save_is_deferred(tmp_path, monkeypatch, operation):
    path = tmp_path / "project.nfit"
    gui.save_project(gui.NfitProject([DataGroup("saved")]), path)
    explorer = gui.NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    explorer.project.data_groups[0].name = "edited"
    explorer._mark_dirty()
    explorer._interactive = True
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_args, **_kwargs: (str(path), ""))
    prompts, checks = [], []
    monkeypatch.setattr(explorer, "_prompt_external_change_action",
                        lambda: prompts.append("external") or "keep")
    published, observed = Event(), Event()
    original = gui.save_project

    def save(*args, **kwargs):
        original(*args, **kwargs)
        published.set()
        assert observed.wait(5), "GUI did not poll during the save publication gap"

    monkeypatch.setattr(gui, "save_project", save)

    def poll():
        if published.is_set() and not observed.is_set():
            try:
                checks.append((explorer._project_changed_on_disk(),
                               explorer.check_for_external_project_change()))
            finally:
                observed.set()

    timer = QtCore.QTimer(explorer.window)
    timer.setInterval(1)
    timer.timeout.connect(poll)
    timer.start()
    try:
        assert getattr(explorer, operation)()
        assert checks == [(True, False)]
        assert prompts == []
        assert not explorer.has_unsaved_changes
        assert not explorer.check_for_external_project_change()
        assert gui.load_project(path).data_groups[0].name == "edited"
        # A subsequent genuine external replacement still needs review.
        gui.save_project(gui.NfitProject([DataGroup("external")]), path)
        assert explorer.check_for_external_project_change()
        assert prompts == ["external"]
        assert explorer.project.data_groups[0].name == "edited"
    finally:
        observed.set()
        timer.stop()
        explorer._interactive = False
        explorer.has_unsaved_changes = False
        explorer.window.close()


def test_external_change_dialog_does_not_reenter_its_own_poll(tmp_path, monkeypatch):
    path = tmp_path / "project.nfit"
    gui.save_project(gui.NfitProject([DataGroup("saved")]), path)
    explorer = gui.NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    gui.save_project(gui.NfitProject([DataGroup("external")]), path)
    checks, prompts = [], []

    def prompt():
        # Modal questions run another event loop where the same timer can fire.
        prompts.append(True)
        if len(prompts) == 1:
            checks.append(explorer.check_for_external_project_change())
        return "keep"

    monkeypatch.setattr(explorer, "_prompt_external_change_action", prompt)
    try:
        assert explorer.check_for_external_project_change()
        assert checks == [False]
        assert len(prompts) == 1
        assert not explorer.check_for_external_project_change()
    finally:
        explorer.window.close()


def test_failed_save_does_not_adopt_an_external_replacement(tmp_path, monkeypatch):
    from nfit import project_resource_gui

    path = tmp_path / "project.nfit"
    gui.save_project(gui.NfitProject([DataGroup("saved")]), path)
    explorer = gui.NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    explorer.project.data_groups[0].name = "unsaved"
    explorer._mark_dirty()
    original = gui.save_project

    def fail(*_args, **_kwargs):
        original(gui.NfitProject([DataGroup("external")]), path)
        assert not explorer.check_for_external_project_change()
        raise OSError("Save could not publish its own state")

    monkeypatch.setattr(gui, "save_project", fail)
    monkeypatch.setattr(project_resource_gui, "handle_resource_io_error", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(explorer, "_prompt_external_change_action", lambda: "keep")
    try:
        assert not explorer.save()
        assert explorer.has_unsaved_changes
        assert explorer._project_changed_on_disk()
        assert explorer.check_for_external_project_change()
        assert explorer.project.data_groups[0].name == "unsaved"
        assert gui.load_project(path).data_groups[0].name == "external"
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()
