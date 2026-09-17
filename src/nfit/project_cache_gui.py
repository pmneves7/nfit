"""GUI policy for full compressed rebin caches."""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from .file_dialogs import get_existing_directory, get_save_file_name
from .rebin_cache import CompressedBinning


def _format_memory(value: int) -> str:
    gib = max(int(value), 0) / 1024**3
    return f"{gib:.1f} GB" if gib >= 0.1 else f"{max(int(value), 0) / 1024**2:.0f} MiB"


def confirm_rebin_cache_preflight(
    parent: QtWidgets.QWidget,
    *,
    operation: str,
    result_count: int,
    added_bytes: int,
    current_bytes: int,
    projected_bytes: int,
    limit_bytes: int,
    choose_disk_cache: Callable[[], bool] | None = None,
) -> bool:
    """Confirm a multi-result operation before it approaches the RAM cache limit."""

    message = QtWidgets.QMessageBox(parent)
    message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    message.setWindowTitle("Rebin cache memory estimate")
    noun = "result" if result_count == 1 else "results"
    percent = 100.0 * projected_bytes / max(limit_bytes, 1)
    message.setText(
        f"{operation} may add about {_format_memory(added_bytes)} across "
        f"{result_count} rebin {noun}. Together with {_format_memory(current_bytes)} "
        f"already cached, that would use about {_format_memory(projected_bytes)} "
        f"({percent:.0f}%) of the {_format_memory(limit_bytes)} RAM limit.\n\n"
        "Expand the RAM limit in Preferences → Performance, "
        "designate a disk cache location when nfit offers to save an evicted "
        "binning, or download more RAM.\n\nCancel before rebinning starts, or continue "
        "knowing that nfit may need to evict cached results."
    )
    continue_button = message.addButton(
        "Continue anyway", QtWidgets.QMessageBox.ButtonRole.AcceptRole
    )
    cancel_button = message.addButton(
        "Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole
    )
    cache_button = (
        message.addButton(
            "Choose disk cache & continue…",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        if choose_disk_cache is not None
        else None
    )
    continue_button.setToolTip(
        "Start the operation even though cached rebin results may need to be evicted."
    )
    cancel_button.setToolTip(
        "Stop before any pending rebin begins so memory settings can be changed."
    )
    if cache_button is not None:
        cache_button.setToolTip(
            "Choose a folder for this session's evicted binnings, then start the operation."
        )
    message.setDefaultButton(cancel_button)
    while True:
        message.exec()
        if cache_button is not None and message.clickedButton() is cache_button:
            if choose_disk_cache is not None and choose_disk_cache():
                return True
            continue
        return message.clickedButton() is continue_button


@dataclass
class _DiscardRequest:
    key: Any
    artifact: CompressedBinning
    destination: Path | None = None
    finished: threading.Event = field(default_factory=threading.Event)


class CompressedCachePrompt(QtCore.QObject):
    """Marshal cache-discard choices to the Qt GUI thread."""

    requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._choice_completed = False
        self._discard_rest = False
        self._cache_directory: Path | None = None
        self.requested.connect(self._handle_request)

    def request(self, key: Any, artifact: CompressedBinning) -> Path | None:
        item = _DiscardRequest(key, artifact)
        if QtCore.QThread.currentThread() == self.thread():
            self._handle_request(item)
        else:
            self.requested.emit(item)
            item.finished.wait()
        return item.destination

    @QtCore.Slot(object)
    def _handle_request(self, item: _DiscardRequest) -> None:
        try:
            if self._discard_rest:
                return
            if self._cache_directory is not None:
                try:
                    item.destination = self._write_session_cache(
                        item.key, item.artifact
                    )
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(
                        self.parent_window,
                        "Could not write disk cache",
                        str(exc),
                    )
                    self._cache_directory = None
                    item.destination = self._choose(item.key, item.artifact)
            else:
                item.destination = self._choose(item.key, item.artifact)
        finally:
            item.finished.set()

    def _cache_filename(self, key: Any) -> str:
        label = str(key)
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:80]
        return filename or "nfit_binning"

    def _write_session_cache(
        self, key: Any, artifact: CompressedBinning
    ) -> Path:
        if self._cache_directory is None:
            raise RuntimeError("a session cache directory has not been selected")
        stem = self._cache_filename(key)
        path = self._cache_directory / f"{stem}.npz"
        suffix = 2
        while path.exists():
            path = self._cache_directory / f"{stem}-{suffix}.npz"
            suffix += 1
        artifact.write_npz(path)
        return path

    def choose_cache_directory(self) -> bool:
        """Choose and remember one private disk-cache folder for this session."""

        parent = QtWidgets.QApplication.activeModalWidget() or self.parent_window
        directory = get_existing_directory(
            parent,
            "Choose session rebin cache folder",
        )
        if not directory:
            return False
        cache_directory = (
            Path(directory) / f".nfit-session-cache-{uuid.uuid4().hex[:12]}"
        )
        try:
            cache_directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                parent,
                "Could not create disk cache",
                str(exc),
            )
            return False
        self._cache_directory = cache_directory
        self._choice_completed = True
        return True

    def _choose(self, key: Any, artifact: CompressedBinning) -> Path | None:
        label = str(key)
        filename = self._cache_filename(key)
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
                "Choose a folder to cache this and later evictions, export only "
                "this bin as a compressed NPZ, or discard old bins as needed."
            )
            save_button = message.addButton(
                "Save compressed NPZ…", QtWidgets.QMessageBox.ButtonRole.ActionRole
            )
            cache_button = message.addButton(
                "Choose disk cache folder…",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            discard_button = message.addButton(
                "Discard old bins as needed",
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            save_button.setToolTip(
                "Export this one compressed binning before it leaves the cache."
            )
            cache_button.setToolTip(
                "Keep this and later evicted binnings in one disk cache for the rest of this session."
            )
            discard_button.setToolTip(
                "Remove this cached result from RAM and discard later old bins "
                "without another prompt during this nfit session."
            )
            message.setDefaultButton(save_button)
            message.exec()
            if message.clickedButton() is discard_button:
                self._discard_rest = True
                self._choice_completed = True
                return None
            if message.clickedButton() is cache_button:
                if not self.choose_cache_directory():
                    continue
                try:
                    destination = self._write_session_cache(key, artifact)
                except OSError as exc:
                    self._cache_directory = None
                    QtWidgets.QMessageBox.warning(
                        parent,
                        "Could not create disk cache",
                        str(exc),
                    )
                    continue
                self._choice_completed = True
                return destination
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
            self._discard_rest = True
            self._choice_completed = True
            return None

    def cleanup(self) -> None:
        """Remove the empty private session directory after cache files are released."""

        if self._cache_directory is None:
            return
        try:
            self._cache_directory.rmdir()
        except OSError:
            pass
