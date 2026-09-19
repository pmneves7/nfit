from pathlib import Path
from subprocess import CompletedProcess

import pytest
from PySide6 import QtCore, QtWidgets

from nfit import file_dialogs


def test_file_dialog_uses_project_then_remembers_selected_directory(monkeypatch, tmp_path):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat
    )
    monkeypatch.setattr(file_dialogs, "application_settings", lambda: settings)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_path = project_dir / "analysis.nfit"
    selected_dir = tmp_path / "data"
    selected_dir.mkdir()
    selected_path = selected_dir / "scan.nxs"
    starts = []

    def choose(_parent, _caption, start, _filter, **kwargs):
        starts.append(Path(start))
        assert "options" in kwargs
        return str(selected_path), "Data files (*)"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose)
    path, _ = file_dialogs.get_open_file_name(
        None, "Import", "", "Data files (*)", project_path=project_path
    )
    assert Path(path) == selected_path
    assert starts == [project_dir]
    assert settings.value(file_dialogs.LAST_FILE_DIALOG_DIRECTORY_KEY) == str(selected_dir)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda _parent, _caption, start, _filter, **kwargs: (
            (starts.append(Path(start)) or [str(selected_path)]),
            "Data files (*)" if "options" in kwargs else "",
        ),
    )
    paths, selected_filter = file_dialogs.get_open_file_names(
        None, "Import", "", "Data files (*)"
    )
    assert paths == [str(selected_path)]
    assert selected_filter == "Data files (*)"
    assert starts[-1] == selected_dir

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda _parent, _caption, start, _filter, **_kwargs: (
            starts.append(Path(start)) or "",
            "",
        ),
    )
    file_dialogs.get_save_file_name(None, "Save", "result.npz", "NumPy (*.npz)")
    assert starts[-1] == selected_dir / "result.npz"
    application.processEvents()


def test_linux_file_dialogs_avoid_desktop_portals(monkeypatch):
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")

    assert file_dialogs._dialog_options() & (
        QtWidgets.QFileDialog.Option.DontUseNativeDialog
    )


def test_linux_file_dialog_is_raised_above_its_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    parent = QtWidgets.QWidget()
    dialog = file_dialogs._linux_file_dialog(
        parent,
        "Import datasets",
        str(tmp_path),
        "NeXus files (*.nxs)",
        file_mode=QtWidgets.QFileDialog.FileMode.ExistingFiles,
        accept_mode=QtWidgets.QFileDialog.AcceptMode.AcceptOpen,
    )

    assert dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
    assert dialog.fileMode() == QtWidgets.QFileDialog.FileMode.ExistingFiles
    assert dialog.acceptMode() == QtWidgets.QFileDialog.AcceptMode.AcceptOpen
    assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal
    assert dialog.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint


def test_gtk_open_dialog_translates_filters_and_multiple_selection(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(file_dialogs.shutil, "which", lambda _name: "/usr/bin/zenity")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundled")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, f"{tmp_path / 'one.nxs'}\n{tmp_path / 'two.h5'}\n", "")

    monkeypatch.setattr(file_dialogs.subprocess, "run", run)
    result = file_dialogs._gtk_open_file_dialog(
        "Import datasets",
        str(tmp_path),
        "NeXus files (*.nxs *.h5);;All files (*)",
        multiple=True,
    )

    assert result == ([str(tmp_path / "one.nxs"), str(tmp_path / "two.h5")], "")
    command, kwargs = calls[0]
    assert command[:4] == [
        "/usr/bin/zenity",
        "--file-selection",
        "--title=Import datasets",
        f"--filename={tmp_path}/",
    ]
    assert "--file-filter=NeXus files | *.nxs *.h5" in command
    assert "--file-filter=All files | *" in command
    assert command[-2:] == ["--multiple", "--separator=\n"]
    assert kwargs["env"]["LD_LIBRARY_PATH"] == "/system"


def test_gtk_dialog_uses_yad_when_zenity_is_unavailable(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        file_dialogs.shutil,
        "which",
        lambda name: "/usr/bin/yad" if name == "yad" else None,
    )
    monkeypatch.setattr(
        file_dialogs.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or CompletedProcess(command, 0, "/data/scan.nxs\n", "")
        ),
    )

    assert file_dialogs._gtk_open_file_dialog(
        "Import", str(tmp_path), "Data files (*)", multiple=False
    ) == (["/data/scan.nxs"], "")
    assert commands[0][:2] == ["/usr/bin/yad", "--file"]


def test_gtk_dialog_cancel_does_not_open_qt_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs.shutil, "which", lambda _name: "/usr/bin/zenity")
    monkeypatch.setattr(
        file_dialogs.subprocess,
        "run",
        lambda command, **_kwargs: CompletedProcess(command, 1, "", ""),
    )
    monkeypatch.setattr(
        file_dialogs,
        "_exec_linux_file_dialog",
        lambda _dialog: (_ for _ in ()).throw(AssertionError("Qt fallback opened")),
    )

    assert file_dialogs._linux_open_file_dialog(
        None, "Import", str(tmp_path), "Data files (*)", multiple=False
    ) == ([], "")


def test_linux_gtk_save_uses_save_mode_and_remembers_directory(monkeypatch, tmp_path):
    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat
    )
    settings.setValue(file_dialogs.LAST_FILE_DIALOG_DIRECTORY_KEY, str(tmp_path))
    monkeypatch.setattr(file_dialogs, "application_settings", lambda: settings)
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    monkeypatch.setattr(file_dialogs.shutil, "which", lambda _name: "/usr/bin/zenity")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundled")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system")
    output = tmp_path / "result.npz"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, f"{output}\n", "")

    monkeypatch.setattr(file_dialogs.subprocess, "run", run)
    monkeypatch.setattr(
        file_dialogs,
        "_linux_file_dialog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Qt chooser constructed")
        ),
    )

    result = file_dialogs.get_save_file_name(
        None, "Save result", "result.npz", "NumPy archives (*.npz);;All files (*)"
    )

    assert result == (str(output), "")
    command, kwargs = calls[0]
    assert command[:4] == [
        "/usr/bin/zenity",
        "--file-selection",
        "--title=Save result",
        f"--filename={tmp_path / 'result.npz'}",
    ]
    assert "--save" in command
    assert "--confirm-overwrite" in command
    assert "--file-filter=NumPy archives | *.npz" in command
    assert kwargs["env"]["LD_LIBRARY_PATH"] == "/system"
    assert settings.value(file_dialogs.LAST_FILE_DIALOG_DIRECTORY_KEY) == str(tmp_path)


def test_linux_gtk_directory_mode_remembers_selection(monkeypatch, tmp_path):
    settings = QtCore.QSettings(
        str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat
    )
    monkeypatch.setattr(file_dialogs, "application_settings", lambda: settings)
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    monkeypatch.setattr(
        file_dialogs.shutil,
        "which",
        lambda name: "/usr/bin/yad" if name == "yad" else None,
    )
    selected = tmp_path / "cache"
    selected.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, f"{selected}\n", "")

    monkeypatch.setattr(file_dialogs.subprocess, "run", run)
    monkeypatch.setattr(
        file_dialogs,
        "_linux_file_dialog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Qt chooser constructed")
        ),
    )

    assert file_dialogs.get_existing_directory(None, "Choose cache") == str(selected)
    command, _kwargs = calls[0]
    assert command[:2] == ["/usr/bin/yad", "--file"]
    assert "--directory" in command
    assert "--save" not in command
    assert settings.value(file_dialogs.LAST_FILE_DIALOG_DIRECTORY_KEY) == str(selected)


@pytest.mark.parametrize("mode", ["save", "directory"])
@pytest.mark.parametrize("return_code", [1, 5])
def test_linux_gtk_cancel_does_not_construct_qt_dialog(
    monkeypatch, tmp_path, mode, return_code
):
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    monkeypatch.setattr(file_dialogs.shutil, "which", lambda _name: "/usr/bin/zenity")
    monkeypatch.setattr(
        file_dialogs.subprocess,
        "run",
        lambda command, **_kwargs: CompletedProcess(command, return_code, "", ""),
    )
    monkeypatch.setattr(
        file_dialogs,
        "_linux_file_dialog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Qt chooser constructed")
        ),
    )

    if mode == "save":
        assert file_dialogs.get_save_file_name(
            None, "Save", str(tmp_path / "result.npz"), "NumPy (*.npz)"
        ) == ("", "")
    else:
        assert file_dialogs.get_existing_directory(None, "Cache", str(tmp_path)) == ""


@pytest.mark.parametrize("mode", ["save", "directory"])
@pytest.mark.parametrize("failure", ["unavailable", "error"])
def test_linux_chooser_falls_back_to_raised_qt_dialog(
    monkeypatch, tmp_path, mode, failure
):
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    monkeypatch.setattr(
        file_dialogs.shutil, "which",
        lambda _name: None if failure == "unavailable" else "/usr/bin/zenity",
    )
    monkeypatch.setattr(
        file_dialogs.subprocess, "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 2, "", "helper failed"),
    )
    seen = []
    monkeypatch.setattr(
        file_dialogs,
        "_linux_file_dialog",
        lambda *args, **kwargs: seen.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        file_dialogs,
        "_exec_linux_file_dialog",
        lambda dialog: ([str(tmp_path / "fallback.npz")], "NumPy (*.npz)"),
    )

    if mode == "save":
        assert file_dialogs.get_save_file_name(
            None, "Save", str(tmp_path / "result.npz"), "NumPy (*.npz)"
        ) == (str(tmp_path / "fallback.npz"), "NumPy (*.npz)")
        expected_mode = QtWidgets.QFileDialog.FileMode.AnyFile
        expected_accept = QtWidgets.QFileDialog.AcceptMode.AcceptSave
    else:
        assert file_dialogs.get_existing_directory(
            None, "Cache", str(tmp_path)
        ) == str(tmp_path / "fallback.npz")
        expected_mode = QtWidgets.QFileDialog.FileMode.Directory
        expected_accept = QtWidgets.QFileDialog.AcceptMode.AcceptOpen
    assert len(seen) == 1
    assert seen[0][1]["file_mode"] == expected_mode
    assert seen[0][1]["accept_mode"] == expected_accept


def test_linux_save_preserves_explicit_format_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs.sys, "platform", "linux")
    monkeypatch.setattr(
        file_dialogs, "_gtk_file_dialog",
        lambda *args, **kwargs: pytest.fail("GTK cannot return the selected filter"),
    )
    monkeypatch.setattr(file_dialogs, "_linux_file_dialog", lambda *a, **kw: object())
    chosen = (str(tmp_path / "surface.vtp"), "glTF scene (*.gltf)")
    monkeypatch.setattr(
        file_dialogs, "_exec_linux_file_dialog", lambda dialog: ([chosen[0]], chosen[1])
    )
    monkeypatch.setattr(file_dialogs, "remember_file_dialog_path", lambda path: None)
    assert file_dialogs.get_save_file_name(
        None, "Export surface", str(tmp_path / "surface.vtp"),
        "VTK PolyData (*.vtp);;glTF scene (*.gltf)", require_selected_filter=True,
    ) == chosen


def test_cache_save_chooser_uses_active_modal_parent(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nfit import project_cache_gui

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    progress = QtWidgets.QDialog(window)
    monkeypatch.setattr(QtWidgets.QApplication, "activeModalWidget", lambda: progress)

    def choose_save(message):
        assert message.parentWidget() is progress
        next(button for button in message.buttons() if "Save" in button.text()).click()

    destination = tmp_path / "bin.npz"

    def save_dialog(parent, *_args):
        assert parent is progress
        return str(destination), "NumPy archives (*.npz)"

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", choose_save)
    monkeypatch.setattr(project_cache_gui, "get_save_file_name", save_dialog)
    written = []
    prompt = project_cache_gui.CompressedCachePrompt(window)
    prompt.request("cached bin", SimpleNamespace(write_npz=written.append))
    assert written == [destination]
    application.processEvents()
