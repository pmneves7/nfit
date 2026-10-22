"""External-change polling must not mistake publication by its own GUI for a conflict."""

from threading import Event

import pytest
from PySide6 import QtCore, QtWidgets

from nfit import project_gui as gui
from nfit.pipeline import DataGroup


@pytest.fixture
def changed_project(tmp_path):
    path = tmp_path / "shared.nfit"
    gui.save_project(gui.NfitProject([DataGroup("original")]), path)
    explorer = gui.NfitProjectExplorer()
    assert explorer.open_project_path(path, remember=False)
    gui.save_project(gui.NfitProject([DataGroup("external")]), path)
    try:
        yield explorer, path
    finally:
        explorer.has_unsaved_changes = False
        explorer.window.close()


def _notice_button(explorer, action):
    return explorer._external_change_notice.toolbar.findChild(
        QtWidgets.QPushButton, f"project_external_change_{action}"
    )


def test_background_file_change_has_no_modal_dialog_or_application_input_barrier(changed_project, monkeypatch):
    from PySide6 import QtTest

    explorer, _path = changed_project
    other_window = QtWidgets.QWidget()
    editor = QtWidgets.QLineEdit(other_window)
    other_window.show()
    # The project window is hidden, modelling a notification while elsewhere.
    assert not explorer.window.isVisible()
    monkeypatch.setattr(explorer.window, "activateWindow", lambda: pytest.fail("stole window focus"))
    monkeypatch.setattr(explorer.window, "raise_", lambda: pytest.fail("raised background window"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda *_: pytest.fail("opened a modal file-change dialog"))
    try:
        assert explorer.check_for_external_project_change()
        assert not explorer.window.isVisible()
        assert not gui.gui_operation_active()
        assert QtWidgets.QApplication.activeModalWidget() is None
        QtTest.QTest.keyClicks(editor, "still responsive")
        assert editor.text() == "still responsive"
        assert not explorer._external_change_notice.toolbar.isHidden()
        assert explorer._external_change_notice.toolbar.window() is explorer.window
        assert explorer.project.data_groups[0].name == "original"
        explorer.window.show()
        QtWidgets.QApplication.processEvents()
        assert explorer._external_change_notice.toolbar.isVisible()
        for action in ("reload", "save_as", "keep"):
            button = _notice_button(explorer, action)
            assert button.toolTip()
            assert not button.isDefault() and not button.autoDefault()
    finally:
        other_window.close()


def test_notice_does_not_close_editor_popups(changed_project):
    explorer, _path = changed_project
    explorer.window.show()
    combo = QtWidgets.QComboBox(explorer.window)
    combo.addItems(["first", "second"])
    combo.showPopup()
    try:
        assert combo.view().isVisible()
        assert explorer.check_for_external_project_change()
        assert combo.view().isVisible()
    finally:
        combo.hidePopup()


def test_keep_acknowledges_only_observed_revision_and_preserves_save_conflict(changed_project, monkeypatch):
    explorer, path = changed_project
    assert explorer.check_for_external_project_change()
    assert not explorer.check_for_external_project_change()
    observed = explorer._external_change_notice.identity[1]
    gui.save_project(gui.NfitProject([DataGroup("newer external")]), path)
    _notice_button(explorer, "keep").click()
    assert explorer._ignored_project_disk_signature == observed
    assert explorer._external_change_notice.toolbar.isHidden()
    assert explorer._project_changed_on_disk()
    assert explorer.check_for_external_project_change()
    assert explorer._external_change_notice.identity[1] != observed
    _notice_button(explorer, "keep").click()
    assert not explorer.check_for_external_project_change()
    monkeypatch.setattr(explorer, "_prompt_save_conflict_action", lambda: "cancel")
    assert not explorer.save()
    assert gui.load_project(path).data_groups[0].name == "newer external"
    assert explorer.project.data_groups[0].name == "original"


def test_notice_reload_confirms_unsaved_changes_and_survives_cancellation(changed_project, monkeypatch):
    explorer, path = changed_project
    assert explorer.check_for_external_project_change()
    explorer.project.data_groups[0].name = "local edits"
    explorer._mark_dirty()
    assert not explorer.check_for_external_project_change()
    assert "unsaved changes" in explorer._external_change_notice.label.text()
    monkeypatch.setattr(explorer, "_prompt_reload_unsaved_action", lambda: "cancel")
    _notice_button(explorer, "reload").click()
    assert explorer.project.data_groups[0].name == "local edits"
    assert explorer.has_unsaved_changes
    assert not explorer._external_change_notice.toolbar.isHidden()
    monkeypatch.setattr(explorer, "_prompt_reload_unsaved_action", lambda: "reload")
    _notice_button(explorer, "reload").click()
    assert explorer.project.data_groups[0].name == "external"
    assert not explorer.has_unsaved_changes
    assert explorer._external_change_notice.toolbar.isHidden()
    assert not explorer.check_for_external_project_change()
    assert gui.load_project(path).data_groups[0].name == "external"


@pytest.mark.parametrize("accept", [False, True])
def test_notice_save_as_preserves_external_file(changed_project, tmp_path, monkeypatch, accept):
    explorer, path = changed_project
    copy_path = tmp_path / "local copy.nfit"
    explorer.project.data_groups[0].name = "local edits"
    explorer._mark_dirty()
    assert explorer.check_for_external_project_change()
    monkeypatch.setattr(gui, "get_save_file_name", lambda *_a, **_k: (str(copy_path) if accept else "", ""))
    _notice_button(explorer, "save_as").click()
    assert gui.load_project(path).data_groups[0].name == "external"
    assert explorer.project.data_groups[0].name == "local edits"
    if accept:
        assert explorer.project_path == copy_path
        assert gui.load_project(copy_path).data_groups[0].name == "local edits"
        assert explorer._external_change_notice.toolbar.isHidden()
    else:
        assert explorer.project_path == path
        assert explorer.has_unsaved_changes
        assert not explorer._external_change_notice.toolbar.isHidden()


def test_notice_is_retired_on_project_switch_and_old_actions_are_ignored(changed_project):
    explorer, path = changed_project
    assert explorer.check_for_external_project_change()
    _old_path, observed = explorer._external_change_notice.identity
    assert explorer.new_project()
    assert explorer._external_change_notice.toolbar.isHidden()
    assert explorer._external_change_notice.identity is None
    assert not explorer._handle_external_change_action("reload", path, observed)
    assert explorer.project_path is None


def test_notice_waits_for_guarded_operations_and_does_not_resolve_during_them(changed_project, monkeypatch):
    explorer, _path = changed_project
    with gui.GuiOperationGuard():
        assert not explorer.check_for_external_project_change()
        assert explorer._external_change_notice.identity is None
    assert explorer.check_for_external_project_change()
    monkeypatch.setattr(explorer, "reload_project_from_disk", lambda: pytest.fail("reloaded during an operation"))
    with gui.GuiOperationGuard():
        _notice_button(explorer, "reload").click()
    assert not explorer._external_change_notice.toolbar.isHidden()
    assert not gui.gui_operation_active()


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
