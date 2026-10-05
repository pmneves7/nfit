"""Nonblocking system document launch and actionable desktop errors."""

from pathlib import Path

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from .desktop_launch import open_local_document


def open_local_document_in_desktop(parent, path: str | Path) -> bool:
    """Launch a local document and monitor opener failure without blocking Qt."""
    document = Path(path).expanduser().absolute()

    def failed(reason):
        if parent is not None and not isValid(parent):
            return
        QtWidgets.QMessageBox.warning(
            parent, "Cannot open documentation",
            f"{reason}\n\nConfigure a default web browser in your desktop settings, "
            f"or open this file directly in a browser:\n{document}",
        )

    try:
        process = open_local_document(document)
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
