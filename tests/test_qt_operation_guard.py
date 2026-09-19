from __future__ import annotations

import time

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nfit.qt_operation_guard import GuiOperationGuard, close_operation_window, close_widget_popups


@pytest.fixture
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application
    guard_filter = getattr(application, "_nfit_operation_input_filter", None)
    assert guard_filter is None or guard_filter.depth == 0


def _selector():
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    combo = QtWidgets.QComboBox()
    combo.addItems(["min/max", "percentile", "symmetric"])
    layout.addWidget(combo)
    window.show()
    return window, combo


def test_guard_blocks_other_windows_and_queued_input_but_not_cancel(app):
    window, combo = _selector()
    progress = QtWidgets.QDialog(window)
    progress.setProperty("nfit_operation_dialog", True)
    button = QtWidgets.QPushButton("Cancel", progress)
    cancelled = []
    button.clicked.connect(lambda: cancelled.append(True))
    combo.showPopup()
    app.processEvents()
    with GuiOperationGuard():
        assert not combo.view().isVisible()
        progress.show()
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
        assert combo.currentIndex() == 0
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        assert cancelled == [True]
        # Native-window events must also reach a progress dialog on macOS.
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress, QtCore.QPointF(1, 1),
            QtCore.QPointF(1, 1), QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
        )
        assert not app._nfit_operation_input_filter.eventFilter(progress.windowHandle(), event)
        late_window, late_combo = _selector()
        QtTest.QTest.keyClick(late_combo, QtCore.Qt.Key.Key_Down)
        assert late_combo.currentIndex() == 0
        window.close()
        assert window.isVisible()
        QtCore.QCoreApplication.postEvent(combo, QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Down,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ))
    assert combo.currentIndex() == 0  # queued click was discarded, not replayed
    progress.close()
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
    assert combo.currentIndex() == 1
    assert combo.isEnabled()
    window.close()
    late_window.close()


def test_nested_guards_restore_after_exception_without_enabling_disabled_controls(app):
    window, combo = _selector()
    combo.setEnabled(False)
    with pytest.raises(ValueError), GuiOperationGuard():
        with GuiOperationGuard():
            assert app._nfit_operation_input_filter.depth == 2
        assert app._nfit_operation_input_filter.depth == 1
        raise ValueError("operation failed")
    assert not combo.isEnabled()
    window.close()


def test_popup_closed_before_control_rebuild(app):
    window, combo = _selector()
    combo.showPopup()
    app.processEvents()
    close_widget_popups(window)
    assert not combo.view().isVisible()
    window.close()


def test_operation_can_retire_windows_and_hide_menus_while_input_is_guarded(app):
    window, _combo = _selector()
    menu = QtWidgets.QMenu(window)
    menu.addAction("Test")
    with GuiOperationGuard():
        menu.popup(window.pos())
        close_widget_popups(window)
        assert not menu.isVisible()
        close_operation_window(window)
        assert not window.isVisible()


def test_escape_requests_rebin_cancellation_without_unlocking_early(app):
    from nfit import project_gui as gui

    explorer = gui.NfitProjectExplorer(gui.NfitProject())
    callback = explorer._make_rebin_progress_callback("Test rebin")
    controller = callback._nfit_progress_controller
    QtTest.QTest.keyClick(controller.dialog, QtCore.Qt.Key.Key_Escape)
    assert controller.cancel_requested
    assert app._nfit_operation_input_filter.depth == 1
    with pytest.raises(gui.RebinCancellationRequested):
        callback({"iteration": 1, "total": 2, "stage": "rebin"})
    explorer._close_rebin_progress(callback)
    assert app._nfit_operation_input_filter.depth == 0
    explorer.window.close()


@pytest.mark.parametrize("fail", [False, True])
def test_save_guard_covers_archive_write_and_final_refresh(app, tmp_path, monkeypatch, fail):
    from nfit import project_gui as gui

    explorer = gui.NfitProjectExplorer(gui.NfitProject())
    explorer.project_path = tmp_path / "project.nfit"
    window, combo = _selector()
    visited = []

    def check(phase):
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
        assert combo.currentIndex() == 0
        visited.append(phase)

    def save(*_args, **_kwargs):
        check("write")
        if fail:
            raise OSError("write failed")

    monkeypatch.setattr(explorer, "_project_changed_on_disk", lambda: False)
    monkeypatch.setattr(explorer, "_update_project_disk_signature", lambda: None)
    monkeypatch.setattr(gui, "save_project", save)
    monkeypatch.setattr(explorer, "_sync_details", lambda: check("refresh"))
    if fail:
        with pytest.raises(OSError):
            explorer.save()
        assert visited == ["write"]
    else:
        assert explorer.save()
        assert visited == ["write", "refresh"]
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
    assert combo.currentIndex() == 1
    window.close()
    explorer.window.close()


@pytest.mark.parametrize("result", ["success", "failure", "cancel", "refresh_failure"])
def test_background_guard_covers_completion_and_releases_on_all_paths(app, monkeypatch, result):
    from nfit import project_gui as gui

    explorer = gui.NfitProjectExplorer(gui.NfitProject())
    window, combo = _selector()
    settled = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_args: None)

    def task(_progress):
        if result == "failure":
            raise ValueError("test failure")
        if result == "cancel":
            raise gui.FitCancellationRequested("test cancel")
        return 1

    def settle():
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
        settled.append(combo.currentIndex())

    def success(_result):
        settle()
        if result == "refresh_failure":
            raise ValueError("viewer refresh failed")

    assert explorer._start_background_task(
        title="Test", failure_title="Test", task=task,
        on_success=success, on_settled=settle,
        success_message="Done", progress_window_title="Rebin progress",
    )
    deadline = time.monotonic() + 5
    while explorer._fit_worker_thread is not None and time.monotonic() < deadline:
        app.processEvents()
    assert explorer._fit_worker_thread is None
    assert settled and all(value == 0 for value in settled)
    assert app._nfit_operation_input_filter.depth == 0
    # A failed progress dialog may remain visible for acknowledgement.
    for widget in app.topLevelWidgets():
        if widget.objectName() == "rebin_progress_dialog":
            widget.close()
    QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
    assert combo.currentIndex() == 1
    window.close()
    explorer.window.close()
