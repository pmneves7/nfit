"""Early startup splash that loads before the nfit package and scientific stack."""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets


def _resource_root() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen is not None:
        return Path(frozen)
    return Path(__file__).resolve().parents[2]


def _first_file(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _logo_path() -> Path | None:
    root = _resource_root()
    return _first_file(
        root / "nfit/resources/nfit-logo.svg",
        root / "docs/_static/nfit-logo.svg",
    )


def _icon_path() -> Path | None:
    root = _resource_root()
    return _first_file(
        root / "nfit/resources/nfit-icon.png",
        root / "src/nfit/resources/nfit-icon.png",
    )


def _help_index() -> Path | None:
    root = _resource_root()
    return _first_file(
        root / "nfit/resources/help/index.html",
        root / "docs/_build/html/index.html",
    )


def _render_svg(path: Path, size: QtCore.QSize) -> QtGui.QPixmap:
    """Render vector artwork at twice the display resolution for crisp HiDPI output."""
    renderer = QtSvg.QSvgRenderer(str(path))
    if not renderer.isValid():
        return QtGui.QPixmap()
    renderer.setAspectRatioMode(QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    rendered_size = size * 2
    pixmap = QtGui.QPixmap(rendered_size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return pixmap


class StartupSplash(QtWidgets.QWidget):
    """Show branding and an indeterminate loading indicator during startup."""

    def __init__(self):
        super().__init__(
            None,
            QtCore.Qt.WindowType.SplashScreen
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("nfit_startup_splash")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet(
            "QWidget#nfit_startup_splash { background: white; "
            "border: 1px solid #b8bcc4; border-radius: 10px; }"
            "QLabel { color: #20242a; }"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(38, 28, 38, 26)
        layout.setSpacing(10)

        logo = QtWidgets.QLabel()
        logo.setObjectName("startup_logo")
        logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo_path = _logo_path()
        logo_size = QtCore.QSize(420, 235)
        pixmap = (
            _render_svg(logo_path, logo_size)
            if logo_path is not None
            else QtGui.QPixmap()
        )
        if pixmap.isNull() and (icon_path := _icon_path()) is not None:
            pixmap = QtGui.QPixmap(str(icon_path))
        if not pixmap.isNull():
            logo.setPixmap(pixmap)
        else:
            logo.setText("nfit")
            logo.setStyleSheet("font: 48pt serif;")
        layout.addWidget(logo)

        details = QtWidgets.QLabel(
            f"Version {version('nfit')}\n"
            "Paul M. Neves · Johns Hopkins University"
        )
        details.setObjectName("startup_details")
        details.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        details.setStyleSheet("font-size: 13px;")
        layout.addWidget(details)

        help_index = _help_index()
        documentation_url = (
            QtCore.QUrl.fromLocalFile(str(help_index)).toString()
            if help_index is not None
            else "https://nfit.readthedocs.io/"
        )
        documentation = QtWidgets.QLabel(
            f'<a href="{documentation_url}">Open nfit documentation</a>'
        )
        documentation.setObjectName("startup_documentation")
        documentation.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        documentation.setOpenExternalLinks(True)
        documentation.setToolTip(
            "Open the bundled documentation, which remains available offline."
        )
        layout.addWidget(documentation)

        self.status = QtWidgets.QLabel("Loading scientific libraries…")
        self.status.setObjectName("startup_status")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        progress = QtWidgets.QProgressBar()
        progress.setObjectName("startup_progress")
        progress.setRange(0, 0)
        progress.setTextVisible(False)
        progress.setFixedHeight(8)
        progress.setToolTip(
            "nfit is loading its scientific and graphical components."
        )
        layout.addWidget(progress)
        self.setFixedSize(520, 405)

    def show_status(self, text: str) -> None:
        self.status.setText(text)
        QtWidgets.QApplication.processEvents()


def show_startup_splash() -> tuple[QtWidgets.QApplication, StartupSplash]:
    """Create the splash before importing the main nfit package."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("nfit")
    app.setOrganizationName("nfit")
    if (icon_path := _icon_path()) is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    splash = StartupSplash()
    splash.show()
    splash.raise_()
    app.processEvents()
    app._nfit_startup_splash = splash
    return app, splash
