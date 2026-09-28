from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit import qt_resource_monitor as gui

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    value = gui.ResourceMonitor()
    yield value, gui, app
    import shiboken6

    if shiboken6.isValid(value):
        value.close()


def test_sampling_runs_off_thread_and_stops_on_hide(monitor, monkeypatch):
    value, gui, app = monitor
    calls = []

    class Sampler:
        def __init__(self):
            calls.append(threading.get_ident())

        def sample(self):
            calls.append(threading.get_ident())
            from nfit.resource_usage import ResourceUsageSnapshot

            return ResourceUsageSnapshot(12, 3, 45, 67)

    monkeypatch.setattr(gui, "ResourceUsageSampler", Sampler)
    monkeypatch.setattr(gui, "_SAMPLE_SECONDS", 0.01)
    assert value._thread is None
    value.show()
    app.processEvents()
    thread = value._thread
    deadline = time.monotonic() + 1
    while value._latest[0] is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert value._latest[0] is not None
    value._refresh()
    assert value.text() == "nfit CPU 12.0%  RAM 3.0%   |   System CPU 45.0%  RAM 67.0%"
    assert all(ident != threading.get_ident() for ident in calls)
    assert "logical CPU" in value.toolTip()
    value.hide()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not value._timer.isActive()
    value.show()
    app.processEvents()
    assert value._thread is not thread


def test_failed_sampling_is_nonfatal(monitor, monkeypatch):
    value, gui, _app = monitor
    stop = threading.Event()
    latest = ["stale"]

    class Sampler:
        def sample(self):
            stop.set()
            raise OSError("unavailable")

    monkeypatch.setattr(gui, "ResourceUsageSampler", Sampler)
    monkeypatch.setattr(gui, "_SAMPLE_SECONDS", 0.001)
    gui._sample_resources(stop, latest)
    assert latest == [None]
    value._latest = latest
    value._refresh()
    assert "—" in value.text()


def test_destroyed_widget_stops_worker(monitor, monkeypatch):
    value, gui, app = monitor
    from PySide6 import QtCore

    monkeypatch.setattr(gui, "_SAMPLE_SECONDS", 0.01)
    value.show()
    app.processEvents()
    worker = value._thread
    value.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    worker.join(timeout=1)
    assert not worker.is_alive()
