"""GUI policy for full compressed rebin caches."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from .file_dialogs import get_save_file_name
from .rebin_cache import CompressedBinning


@dataclass
class _DiscardRequest:
    key: Any
    artifact: CompressedBinning
    finished: threading.Event = field(default_factory=threading.Event)


class CompressedCachePrompt(QtCore.QObject):
    """Marshal cache-discard choices to the Qt GUI thread."""

    requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._choice_completed = False
        self.requested.connect(self._handle_request)

    def request(self, key: Any, artifact: CompressedBinning) -> None:
        if self._choice_completed:
            return
        item = _DiscardRequest(key, artifact)
        if QtCore.QThread.currentThread() == self.thread():
            self._handle_request(item)
        else:
            self.requested.emit(item)
            item.finished.wait()

    @QtCore.Slot(object)
    def _handle_request(self, item: _DiscardRequest) -> None:
        try:
            if self._choice_completed:
                return
            self._choose(item.key, item.artifact)
            self._choice_completed = True
        finally:
            item.finished.set()

    def _choose(self, key: Any, artifact: CompressedBinning) -> None:
        label = str(key)
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:80]
        filename = filename or "nfit_binning"
        while True:
            # A rebin progress dialog is window-modal and stays above the main
            # window on Linux. Parent the choice to that active modal so ThinLinc
            # and other remote desktops cannot hide it behind the progress UI.
            parent = QtWidgets.QApplication.activeModalWidget() or self.parent_window
            message = QtWidgets.QMessageBox(parent)
            message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            message.setWindowTitle("Compressed binning cache is full")
            message.setText(
                "nfit has reached its total in-memory result-cache limit. "
                f"The oldest bin ({label}) needs to leave memory. "
                "This is the single combined allowance configured under "
                "Preferences → Performance. Save this bin "
                "as a compressed NPZ where you choose, or discard it. After this "
                "choice, nfit will automatically discard additional old compressed "
                "bins for the rest of this session."
            )
            save_button = message.addButton(
                "Save compressed NPZ…", QtWidgets.QMessageBox.ButtonRole.ActionRole
            )
            discard_button = message.addButton(
                "Discard old bins as needed",
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            save_button.setToolTip(
                "Choose a destination for the compressed binning before it leaves memory."
            )
            discard_button.setToolTip(
                "Remove this cached result from RAM and discard later old bins "
                "without another prompt during this nfit session."
            )
            message.setDefaultButton(save_button)
            message.exec()
            if message.clickedButton() is discard_button:
                return
            if message.clickedButton() is not save_button:
                continue
            path, _selected_filter = get_save_file_name(
                self.parent_window,
                "Save compressed cached binning",
                f"{filename}.npz",
                "NumPy archives (*.npz);;All files (*)",
            )
            if not path:
                continue
            try:
                artifact.write_npz(Path(path))
            except (OSError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self.parent_window,
                    "Could not save compressed binning",
                    str(exc),
                )
                continue
            return
