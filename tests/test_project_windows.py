from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from nfit import project_windows


def test_source_window_command_uses_current_interpreter_and_literal_project_path(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    path = tmp_path / "project with spaces.nfit"
    command = project_windows.project_window_command(path)
    assert command[:2] == [sys.executable, "-c"]
    assert command[-2:] == ["--project", str(path)]
    assert "app_bootstrap" in command[2]
    ast.parse(command[2])


def test_frozen_window_resets_bootloader_but_preserves_bundle_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "bundled-libraries")
    path = tmp_path / "project.nfit"
    path.touch()
    calls = []
    marker = object()
    monkeypatch.setattr(subprocess, "Popen", lambda command, **options: calls.append((command, options)) or marker)
    assert project_windows.launch_project_window(path) is marker
    command, options = calls[0]
    assert command == [sys.executable, "--project", str(path)]
    assert options["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert options["env"]["LD_LIBRARY_PATH"] == "bundled-libraries"
    assert options["start_new_session"] and options["stdin"] is subprocess.DEVNULL
    assert not options.get("shell", False)


def test_launch_new_window_does_not_create_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    project_windows.launch_project_window()
    assert calls
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        project_windows.launch_project_window(tmp_path / "missing.nfit")
    assert len(calls) == 1


def test_bootstrap_project_argument_uses_public_project_startup(monkeypatch, tmp_path):
    from nfit import app_bootstrap, project_gui

    calls = []
    monkeypatch.setattr(project_gui, "main", lambda **kwargs: calls.append(kwargs) or 0)
    path = tmp_path / "project with spaces.nfit"
    assert app_bootstrap.main(["--project", str(path)]) == 0
    assert calls == [{"project_path": path}]


def test_window_menu_actions_and_tooltips(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit import file_dialogs, qt_project_windows
    from nfit.project_gui import NfitProject, NfitProjectExplorer

    calls = []
    monkeypatch.setattr(qt_project_windows, "_launch", lambda parent, path=None: calls.append(path))
    path = tmp_path / "next.nfit"
    monkeypatch.setattr(file_dialogs, "get_open_file_name", lambda *args, **kwargs: (str(path), ""))
    explorer = NfitProjectExplorer(NfitProject())
    try:
        actions = [action for action in explorer.window_menu.actions() if not action.isSeparator()]
        assert [action.text() for action in actions] == ["New project window", "Open project in new window…", "Save copy/paste script…"]
        assert explorer.window_menu_button.toolTip()
        assert all(action.toolTip() for action in actions)
        actions[0].trigger()
        actions[1].trigger()
        assert calls == [None, str(path)]
    finally:
        explorer.window.close()


def test_clipboard_shutdown_does_not_crash_in_an_independent_process(tmp_path):
    script = """
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QMimeData
from nfit.qt_project_clipboard import _install_clipboard_cleanup
from nfit.project_transfer import TRANSFER_MIME
app = QApplication([])
_install_clipboard_cleanup()
mime = QMimeData()
mime.setData(TRANSFER_MIME, b'{}')
app.clipboard().setMimeData(mime)
"""
    import os
    outcome = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                             env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, timeout=60)
    assert outcome.returncode == 0, outcome.stderr.decode()
