"""Window-menu actions for independent project processes."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from .project_windows import launch_project_window


def add_window_menu(explorer, toolbar):
    """Place independent-project actions between File and Help."""
    button = QtWidgets.QToolButton()
    button.setObjectName("window_menu_button")
    button.setText("Window")
    button.setToolTip("Open independent project windows with separate jobs and caches.")
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(button)
    menu.setToolTipsVisible(True)
    new = menu.addAction("New project window", lambda: _launch(explorer.window))
    new.setObjectName("new_project_window_action")
    new.setToolTip("Launch a separate nfit process with an empty project. Keep this project open.")

    def choose():
        from .file_dialogs import get_open_file_name

        path, _filter = get_open_file_name(explorer.window, "Open project in new window",
                                         str(explorer.project_path.parent) if explorer.project_path else "",
                                         "nfit projects (*.nfit)")
        if path:
            _launch(explorer.window, path)

    action = menu.addAction("Open project in new window…", choose)
    action.setObjectName("open_project_window_action")
    action.setToolTip("Open a saved .nfit project in an independent application window.")
    menu.addSeparator()
    from .qt_project_clipboard import save_clipboard_script

    action = menu.addAction("Save copy/paste script…", lambda: save_clipboard_script(explorer))
    action.setObjectName("save_project_copy_script_action")
    action.setToolTip("Save the copied selection and an editable Python script to paste it without a GUI. Scientific arrays remain lazy.")
    button.setMenu(menu)
    toolbar.addWidget(button)
    explorer.window_menu = menu
    explorer.window_menu_button = button


def _launch(parent, path=None):
    def failed(message):
        if isValid(parent):
            QtWidgets.QMessageBox.warning(parent, "Cannot open project window", message)

    try:
        process = launch_project_window(path)
    except OSError as error:
        failed(str(error))
        return False
    timer = QtCore.QTimer(parent)
    timer.setInterval(250)

    def finished():
        code = process.poll()
        if code is None:
            return
        timer.stop()
        timer.deleteLater()
        if code:
            failed(f"The new nfit process exited with status {code}. See the launching terminal for details.")

    timer.timeout.connect(finished)
    timer.start()
    return True
