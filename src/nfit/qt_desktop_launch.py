"""Nonblocking system document launch and actionable desktop errors."""

from pathlib import Path

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from .desktop_launch import open_local_document, show_in_file_manager


def open_local_document_in_desktop(parent, path: str | Path) -> bool:
    """Launch a local document and monitor opener failure without blocking Qt."""
    return _open_in_desktop(
        parent, path, open_local_document, "Cannot open documentation",
        "Configure a default web browser in your desktop settings, "
        "or open this file directly in a browser:",
    )


def show_in_file_manager_in_desktop(parent, path: str | Path) -> bool:
    """Open a project's directory with nonblocking desktop error reporting."""
    return _open_in_desktop(
        parent, path, show_in_file_manager, "Cannot open project location",
        "Configure a default file manager in your desktop settings, "
        "or open this file's containing folder manually:",
    )


def _open_in_desktop(parent, path, opener, title, failure_hint) -> bool:
    document = Path(path).expanduser().absolute()

    def failed(reason):
        if parent is not None and not isValid(parent):
            return
        QtWidgets.QMessageBox.warning(
            parent, title, f"{reason}\n\n{failure_hint}\n{document}",
        )

    try:
        process = opener(document)
    except OSError as error:
        failed(str(error))
        return False
    if process is None:
        return True

    # Some desktop openers wait until a newly launched browser exits. Polling
    # avoids a synchronous wait or a pipe that the browser could hold open.
    timer = QtCore.QTimer(parent)
    timer.setInterval(250)

    def check_exit():
        if not isValid(timer) or (parent is not None and not isValid(parent)):
            return
        code = process.poll()
        if code is None:
            return
        timer.stop()
        timer.deleteLater()
        if code != 0:
            failed(f"The desktop document opener failed (exit status {code}).")

    timer.timeout.connect(check_exit)
    timer.start()
    return True
