"""Application-wide file dialogs with safe, remembered starting locations."""

from __future__ import annotations

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


def get_open_file_name(
    parent: Any, caption: str, directory: str = "", file_filter: str = "", *, project_path=None
):
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
