from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    from nfit import app_updates_gui as gui
    from nfit.app_distribution import UpdateConfiguration

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    settings = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.Format.IniFormat
    )
    monkeypatch.setattr(
        gui,
        "update_configuration",
        lambda: UpdateConfiguration("pmneves7/nfit", "read-only-token"),
    )
    value = gui.UpdateController(
        window, quit_application=lambda: True, settings=settings
    )
    yield value, gui, app
    window.close()


def test_offline_startup_does_not_interrupt_user(controller, monkeypatch):
    value, gui, _app = controller
    messages = []
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "information",
        lambda *args: messages.append(args),
    )
    value._manual = False
    value._check_failed(gui.UpdateError("offline"))
    assert messages == []
    value._manual = True
    value._check_failed(gui.UpdateError("offline"))
    assert len(messages) == 1


def test_cancelled_project_save_never_opens_installer(
    controller, monkeypatch, tmp_path
):
    value, gui, _app = controller
    opened = []
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "question",
        lambda *args: gui.QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        gui.QtGui.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url) or True,
    )
    value.quit_application = lambda: False
    value._downloaded(tmp_path / "nfit-update.pkg")
    assert opened == []


def test_declining_install_never_quits_or_launches(
    controller, monkeypatch, tmp_path
):
    value, gui, _app = controller
    events = []
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "question",
        lambda *args: gui.QtWidgets.QMessageBox.StandardButton.No,
    )
    monkeypatch.setattr(
        gui.QtGui.QDesktopServices,
        "openUrl",
        lambda url: events.append("open"),
    )
    value.quit_application = lambda: events.append("quit")
    value._downloaded(tmp_path / "nfit-update.pkg")
    assert events == []


def test_approved_install_saves_and_quits_before_opening(
    controller, monkeypatch, tmp_path
):
    value, gui, _app = controller
    events = []
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "question",
        lambda *args: gui.QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        gui.QtGui.QDesktopServices,
        "openUrl",
        lambda url: events.append("open") or True,
    )
    value.quit_application = lambda: events.append("quit") or True
    value._downloaded(tmp_path / "nfit-update.pkg")
    assert events == ["quit", "open"]


def test_splash_shows_branding_progress_and_offline_help(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from tools.distribution.startup_splash import StartupSplash

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    splash = StartupSplash()
    try:
        details = splash.findChild(QtWidgets.QLabel, "startup_details")
        documentation = splash.findChild(
            QtWidgets.QLabel, "startup_documentation"
        )
        progress = splash.findChild(QtWidgets.QProgressBar, "startup_progress")
        logo = splash.findChild(QtWidgets.QLabel, "startup_logo")
        assert "Paul M. Neves" in details.text()
        assert "Version 0.87.1" in details.text()
        assert documentation.openExternalLinks()
        assert documentation.toolTip()
        assert progress.minimum() == 0 and progress.maximum() == 0
        assert progress.toolTip()
        assert logo.pixmap() is not None and not logo.pixmap().isNull()
    finally:
        splash.close()
        app.processEvents()


def test_frozen_entry_shows_splash_before_importing_nfit():
    entry = (
        Path(__file__).parents[1] / "tools/distribution/entry.py"
    ).read_text(encoding="utf-8")
    assert entry.index("show_startup_splash()") < entry.index(
        "from nfit.app_bootstrap import main"
    )
    splash_tree = ast.parse(
        (
            Path(__file__).parents[1]
            / "tools/distribution/startup_splash.py"
        ).read_text(encoding="utf-8")
    )
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(splash_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(splash_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "nfit" not in imported_roots
