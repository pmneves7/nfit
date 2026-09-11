from pathlib import Path

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
