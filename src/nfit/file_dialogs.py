"""Application-wide file dialogs with safe, remembered starting locations."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from .application_preferences import application_settings

LAST_FILE_DIALOG_DIRECTORY_KEY = "files/last_directory"
_active_project_path: Path | None = None


def set_active_project_path(path: str | Path | None) -> None:
    """Set the open project used when no dialog location has been remembered."""

    global _active_project_path
    _active_project_path = None if path is None else Path(path).expanduser()


def preferred_file_dialog_directory(
    *, settings: QtCore.QSettings | None = None, project_path: str | Path | None = None
) -> Path:
    """Return recent, project, Documents, or home directory in that order."""

    store = application_settings() if settings is None else settings
    recent = str(store.value(LAST_FILE_DIALOG_DIRECTORY_KEY, "") or "").strip()
    if recent and Path(recent).expanduser().is_dir():
        return Path(recent).expanduser()
    candidate = Path(project_path).expanduser() if project_path else _active_project_path
    if candidate is not None:
        directory = candidate if candidate.is_dir() else candidate.parent
        if directory.is_dir():
            return directory
    documents = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.DocumentsLocation
    )
    if documents and Path(documents).is_dir():
        return Path(documents)
    return Path.home()


def remember_file_dialog_path(
    path: str | Path, *, settings: QtCore.QSettings | None = None
) -> None:
    """Remember the containing directory for the local nfit installation."""

    selected = Path(path).expanduser()
    directory = selected if selected.is_dir() else selected.parent
    if directory.is_dir():
        (application_settings() if settings is None else settings).setValue(
            LAST_FILE_DIALOG_DIRECTORY_KEY, str(directory)
        )


def _initial_path(directory: str, project_path: str | Path | None) -> str:
    requested = Path(directory).expanduser() if str(directory).strip() else None
    if requested is not None and requested.is_absolute():
        return str(requested)
    base = preferred_file_dialog_directory(project_path=project_path)
    return str(base / requested) if requested is not None else str(base)


def _dialog_options() -> QtWidgets.QFileDialog.Option:
    """Use the bundled chooser where Linux desktop portals can block Qt."""

    if sys.platform.startswith("linux"):
        return QtWidgets.QFileDialog.Option.DontUseNativeDialog
    return QtWidgets.QFileDialog.Option(0)


def _linux_file_dialog(
    parent: Any,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    file_mode: QtWidgets.QFileDialog.FileMode,
    accept_mode: QtWidgets.QFileDialog.AcceptMode,
) -> QtWidgets.QFileDialog:
    """Build a Linux chooser that remote-desktop window managers keep visible."""

    dialog = QtWidgets.QFileDialog(parent, caption, directory, file_filter)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(file_mode)
    dialog.setAcceptMode(accept_mode)
    dialog.setWindowModality(
        QtCore.Qt.WindowModality.WindowModal
        if parent is not None
        else QtCore.Qt.WindowModality.ApplicationModal
    )
    dialog.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
    return dialog


def _exec_linux_file_dialog(dialog: QtWidgets.QFileDialog) -> tuple[list[str], str]:
    """Show and explicitly raise a Linux chooser before entering its modal loop."""

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return [], ""
    return dialog.selectedFiles(), dialog.selectedNameFilter()


def _zenity_filters(file_filter: str) -> list[str]:
    """Translate Qt name filters into Zenity command-line arguments."""

    arguments: list[str] = []
    for specification in file_filter.split(";;"):
        match = re.fullmatch(r"\s*(.*?)\s*\(([^()]*)\)\s*", specification)
        if match is None:
            continue
        label, patterns = (part.strip() for part in match.groups())
        arguments.append(f"--file-filter={label or patterns} | {patterns}")
    return arguments


def _gtk_open_file_dialog(
    caption: str, directory: str, file_filter: str, *, multiple: bool
) -> tuple[list[str], str] | None:
    """Open a GTK chooser on Linux, or return ``None`` when unavailable."""

    executable = shutil.which("zenity")
    chooser_option = "--file-selection"
    if executable is None:
        executable = shutil.which("yad")
        chooser_option = "--file"
    if executable is None:
        return None
    initial = Path(directory).expanduser()
    initial_value = f"{initial}{os.sep}" if initial.is_dir() else str(initial)
    command = [
        executable,
        chooser_option,
        f"--title={caption}",
        f"--filename={initial_value}",
        "--modal",
        *_zenity_filters(file_filter),
    ]
    if multiple:
        command.extend(("--multiple", "--separator=\n"))
    environment = os.environ.copy()
    original_library_path = environment.get("LD_LIBRARY_PATH_ORIG")
    if original_library_path is None:
        environment.pop("LD_LIBRARY_PATH", None)
    else:
        environment["LD_LIBRARY_PATH"] = original_library_path
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError:
        return None
    if completed.returncode in {1, 5}:
        return [], ""
    if completed.returncode != 0:
        return None
    return [path for path in completed.stdout.splitlines() if path], ""


def _linux_open_file_dialog(
    parent: Any,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    multiple: bool,
) -> tuple[list[str], str]:
    """Prefer the GTK chooser and fall back to an explicitly raised Qt chooser."""

    result = _gtk_open_file_dialog(
        caption, directory, file_filter, multiple=multiple
    )
    if result is not None:
        return result
    return _exec_linux_file_dialog(
        _linux_file_dialog(
            parent,
            caption,
            directory,
            file_filter,
            file_mode=(
                QtWidgets.QFileDialog.FileMode.ExistingFiles
                if multiple
                else QtWidgets.QFileDialog.FileMode.ExistingFile
            ),
            accept_mode=QtWidgets.QFileDialog.AcceptMode.AcceptOpen,
        )
    )


def get_open_file_name(
    parent: Any, caption: str, directory: str = "", file_filter: str = "", *, project_path=None
):
    if sys.platform.startswith("linux"):
        selected, selected_filter = _linux_open_file_dialog(
            parent,
            caption,
            _initial_path(directory, project_path),
            file_filter,
            multiple=False,
        )
        result = (selected[0] if selected else "", selected_filter)
    else:
        result = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption,
            _initial_path(directory, project_path),
            file_filter,
            options=_dialog_options(),
        )
    if result[0]:
        remember_file_dialog_path(result[0])
    return result


def get_open_file_names(
    parent: Any, caption: str, directory: str = "", file_filter: str = "", *, project_path=None
):
    if sys.platform.startswith("linux"):
        result = _linux_open_file_dialog(
            parent,
            caption,
            _initial_path(directory, project_path),
            file_filter,
            multiple=True,
        )
    else:
        result = QtWidgets.QFileDialog.getOpenFileNames(
            parent,
            caption,
            _initial_path(directory, project_path),
            file_filter,
            options=_dialog_options(),
        )
    if result[0]:
        remember_file_dialog_path(result[0][0])
    return result


def get_save_file_name(
    parent: Any, caption: str, directory: str = "", file_filter: str = "", *, project_path=None
):
    if sys.platform.startswith("linux"):
        selected, selected_filter = _exec_linux_file_dialog(
            _linux_file_dialog(
                parent,
                caption,
                _initial_path(directory, project_path),
                file_filter,
                file_mode=QtWidgets.QFileDialog.FileMode.AnyFile,
                accept_mode=QtWidgets.QFileDialog.AcceptMode.AcceptSave,
            )
        )
        result = (selected[0] if selected else "", selected_filter)
    else:
        result = QtWidgets.QFileDialog.getSaveFileName(
            parent,
            caption,
            _initial_path(directory, project_path),
            file_filter,
            options=_dialog_options(),
        )
    if result[0]:
        remember_file_dialog_path(result[0])
    return result
