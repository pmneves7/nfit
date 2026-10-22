"""Nonmodal, window-owned notification of an externally changed project."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

DiskSignature = tuple[int, int, int, int]


class ProjectExternalChangeNotice:
    """Keep file-change choices visible without a dialog or an input barrier."""

    def __init__(
        self, window, *, on_action: Callable[[str, Path, DiskSignature], object],
    ):
        from PySide6 import QtCore, QtWidgets

        self._on_action = on_action
        self.identity: tuple[Path, DiskSignature] | None = None
        self.toolbar = QtWidgets.QToolBar("Project changed on disk", window)
        self.toolbar.setObjectName("project_external_change_notice")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setAllowedAreas(QtCore.Qt.ToolBarArea.TopToolBarArea)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 4)
        icon = QtWidgets.QLabel()
        icon.setPixmap(window.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        ).pixmap(20, 20))
        layout.addWidget(icon)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("project_external_change_text")
        self.label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        for text, action, tooltip in (
            ("Reload from Disk", "reload",
             "Load the version on disk. Unsaved changes require confirmation before discarding them."),
            ("Save As…", "save_as",
             "Save the current in-memory project under another name, preserving the external version."),
            ("Keep Current State", "keep",
             "Keep editing without loading or overwriting the external version. Saving still checks for a conflict."),
        ):
            button = QtWidgets.QPushButton(text)
            button.setObjectName(f"project_external_change_{action}")
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, action=action: self._choose(action))
            layout.addWidget(button)
        self.toolbar.addWidget(content)
        window.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.toolbar.hide()

    def present(self, path: Path, signature: DiskSignature, *, unsaved: bool) -> bool:
        """Show/update the notice, returning whether this is a new revision."""

        identity = (path, signature)
        changed = identity != self.identity
        self.identity = identity
        note = " Your current state has unsaved changes." if unsaved else ""
        self.label.setText(f"{path.name} changed on disk. The external version has not been loaded.{note}")
        self.label.setToolTip(str(path))
        # This is a child of the project window, never a separate window that
        # macOS could put on another Space while blocking the application.
        self.toolbar.show()
        return changed

    def clear(self) -> None:
        """Retire a notice after acknowledgement or a project switch/save."""

        self.identity = None
        self.toolbar.hide()

    def _choose(self, action: str) -> None:
        if self.identity is not None:
            path, signature = self.identity
            self._on_action(action, path, signature)
