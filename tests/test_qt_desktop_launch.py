"""Help launcher failures are visible without blocking the Qt event loop."""

import pytest


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from shiboken6 import isValid

    from nfit import qt_desktop_launch
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    document = tmp_path / "index.html"
    document.write_text("help")
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: messages.append(args[2]))
    yield qt_desktop_launch, parent, document, messages, QtCore, app
    if isValid(parent):
        parent.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_missing_opener_reports_offline_path(desktop, monkeypatch):
    module, parent, document, messages, _core, _app = desktop

    def launch(_path):
        raise FileNotFoundError("xdg-open is unavailable")

    monkeypatch.setattr(module, "open_local_document", launch)
    assert not module.open_local_document_in_desktop(parent, document)
    assert len(messages) == 1
    assert "xdg-open is unavailable" in messages[0]
    assert "default web browser" in messages[0]
    assert str(document) in messages[0]


@pytest.mark.parametrize("code", [0, 4])
def test_opener_completion_is_nonblocking_and_reports_failure(desktop, monkeypatch, code):
    module, parent, document, messages, QtCore, app = desktop

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(module, "open_local_document", lambda _path: process)
    assert module.open_local_document_in_desktop(parent, document)
    timer = parent.findChild(QtCore.QTimer)
    assert timer.isActive()
    assert messages == []
    # A still-running desktop opener never requires a synchronous wait.
    timer.timeout.emit()
    assert timer.isActive()
    process.returncode = code
    timer.timeout.emit()
    assert not timer.isActive()
    assert len(messages) == (1 if code else 0)
    if code:
        assert "exit status 4" in messages[0]
        assert str(document) in messages[0]
    app.processEvents()


def test_destroyed_parent_retires_help_failure_monitor(desktop, monkeypatch):
    module, parent, document, messages, QtCore, app = desktop
    from shiboken6 import isValid

    class Process:
        def poll(self):
            pytest.fail("deleted help window still polled the desktop opener")

    monkeypatch.setattr(module, "open_local_document", lambda _path: Process())
    assert module.open_local_document_in_desktop(parent, document)
    timer = parent.findChild(QtCore.QTimer)
    parent.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    assert not isValid(parent)
    assert not isValid(timer)
    assert messages == []
