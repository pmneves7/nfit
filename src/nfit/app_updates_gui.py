"""Application-update presentation and background work."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .app_distribution import application_version, update_configuration
from .app_updates import (
    DownloadCancelled,
    Release,
    UpdateError,
    check_for_update,
    download_installer,
)


class _Worker(QtCore.QObject):
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)
    progress = QtCore.Signal(int)

    def start(self, task: Callable[[], Any]) -> None:
        def run():
            try:
                result = task()
            except Exception as error:
                try:
                    self.failed.emit(error)
                except RuntimeError:  # The application window has already closed.
                    pass
            else:
                try:
                    self.succeeded.emit(result)
                except RuntimeError:
                    pass

        threading.Thread(target=run, name="nfit-update", daemon=True).start()


class UpdateController(QtCore.QObject):
    """Run one update operation at a time and ask before each user-facing step."""

    def __init__(self, window, *, quit_application: Callable[[], bool], settings=None):
        super().__init__(window)
        self.window = window
        self.quit_application = quit_application
        self.settings = settings or QtCore.QSettings("nfit", "nfit")
        self.configuration = update_configuration()
        self._worker: _Worker | None = None
        self._progress = None
        self._cancelled = threading.Event()
        self._manual = False

    def schedule_startup(self) -> None:
        default = bool(getattr(sys, "frozen", False))
        enabled = self.settings.value("updates/check_on_startup", default, type=bool)
        if enabled and self.configuration.repository:
            QtCore.QTimer.singleShot(1200, self, lambda: self.check(manual=False))

    def configure(self) -> None:
        if self._worker is not None:
            return
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle("nfit Updates")
        layout = QtWidgets.QVBoxLayout(dialog)
        if self.configuration.repository:
            access = "private beta" if self.configuration.token else "public"
            note_text = (
                f"Updates come from {self.configuration.repository} on GitHub "
                f"using {access} release access. Updates remain optional, and "
                "nfit and its help work offline."
            )
        else:
            note_text = (
                "This source installation has no release updater configured. "
                "Packaged nfit installers include the release configuration."
            )
        note = QtWidgets.QLabel(note_text)
        note.setObjectName("updates_repository")
        note.setWordWrap(True)
        note.setToolTip("Shows where this installed copy checks for nfit releases.")
        layout.addWidget(note)
        automatic = QtWidgets.QCheckBox("Check for updates when nfit opens")
        automatic.setObjectName("updates_automatic")
        automatic.setToolTip(
            "Check GitHub in the background at startup and ask before downloading."
        )
        automatic.setEnabled(bool(self.configuration.repository))
        automatic.setChecked(
            self.settings.value(
                "updates/check_on_startup",
                bool(getattr(sys, "frozen", False)),
                type=bool,
            )
        )
        layout.addWidget(automatic)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setToolTip(
            "Save the startup update preference."
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            "Keep the existing update preference."
        )

        def save():
            self.settings.setValue(
                "updates/check_on_startup", automatic.isChecked()
            )
            dialog.accept()

        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def check(self, *, manual: bool = True) -> None:
        if self._worker is not None:
            return
        if not self.configuration.repository:
            if manual:
                QtWidgets.QMessageBox.information(
                    self.window,
                    "nfit Updates",
                    "This copy of nfit has no release updater configured.",
                )
            return
        self._manual = manual
        worker = _Worker(self)
        self._worker = worker
        worker.succeeded.connect(self._checked)
        worker.failed.connect(self._check_failed)
        worker.start(
            lambda: check_for_update(
                self.configuration.repository,
                application_version(),
                token=self.configuration.token,
            )
        )

    def _finish_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def _check_failed(self, error: Exception) -> None:
        self._finish_worker()
        if self._manual:
            message = (
                str(error)
                if isinstance(error, UpdateError)
                else "The update check could not be completed. Try again later."
            )
            QtWidgets.QMessageBox.information(
                self.window, "Update check", message
            )

    def _checked(self, release: Release | None) -> None:
        self._finish_worker()
        if release is None:
            if self._manual:
                QtWidgets.QMessageBox.information(
                    self.window,
                    "nfit Updates",
                    "No newer compatible installer is available.",
                )
            return
        box = QtWidgets.QMessageBox(self.window)
        box.setWindowTitle("nfit update available")
        box.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        box.setText(f"nfit {release.version} is available. Download the update?")
        box.setInformativeText(
            (release.notes + "\n\n" if release.notes else "")
            + "You can continue working during the download. Installation starts "
            "only after you approve it."
        )
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        if box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            self._download(release)

    def _download(self, release: Release) -> None:
        self._cancelled = threading.Event()
        self._progress = QtWidgets.QProgressDialog(
            "Downloading nfit update…", "Cancel", 0, 100, self.window
        )
        self._progress.setWindowTitle("nfit Update")
        self._progress.setMinimumDuration(0)
        self._progress.setToolTip(
            "Shows download progress; canceling removes the partial installer."
        )
        self._progress.canceled.connect(self._cancelled.set)
        cache = (
            Path(
                QtCore.QStandardPaths.writableLocation(
                    QtCore.QStandardPaths.StandardLocation.CacheLocation
                )
            )
            / "nfit-updates"
        )
        worker = _Worker(self)
        self._worker = worker
        worker.progress.connect(self._progress.setValue)
        worker.succeeded.connect(self._downloaded)
        worker.failed.connect(self._download_failed)
        worker.start(
            lambda: download_installer(
                release,
                cache,
                token=self.configuration.token,
                progress=lambda count, total: worker.progress.emit(
                    min(99, count * 100 // total)
                ),
                cancelled=self._cancelled.is_set,
            )
        )

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
        self._finish_worker()

    def _download_failed(self, error: Exception) -> None:
        self._close_progress()
        if not isinstance(error, DownloadCancelled):
            message = (
                str(error)
                if isinstance(error, UpdateError)
                else "The download could not be completed. Nothing was installed."
            )
            QtWidgets.QMessageBox.warning(
                self.window, "Update download", message
            )

    def _downloaded(self, installer: Path) -> None:
        self._close_progress()
        choice = QtWidgets.QMessageBox.question(
            self.window,
            "Install nfit update",
            "The installer has been downloaded and verified. Close nfit and open "
            "the installer now?\n\nYou will be prompted to save any unsaved "
            "project changes.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if choice != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not self.quit_application():
            return
        if not QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(installer))
        ):
            QtWidgets.QMessageBox.warning(
                None,
                "Open the installer",
                "The installer could not be opened automatically. Open this "
                f"downloaded file to install the update:\n{installer}",
            )


def add_update_actions(explorer) -> None:
    """Attach application actions without putting update services in the explorer."""
    controller = UpdateController(
        explorer.window, quit_application=explorer.quit_application
    )
    explorer._update_controller = controller
    menu = explorer.file_menu
    menu.addSeparator()
    action = menu.addAction("Check for updates…", lambda: controller.check())
    action.setObjectName("check_updates_action")
    action.setToolTip("Check GitHub for a newer compatible nfit installer.")
    action.setStatusTip(action.toolTip())
    settings = menu.addAction("Update settings…", controller.configure)
    settings.setObjectName("update_settings_action")
    settings.setToolTip("Choose whether nfit checks for updates when it opens.")
    settings.setStatusTip(settings.toolTip())
