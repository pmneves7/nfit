"""Shared application icon for nfit Qt windows and the OS Dock/taskbar."""

from pathlib import Path


def configure_application_icon(app):
    """Install the packaged icon, including when reusing a Qt application."""
    from PySide6.QtGui import QIcon

    app.setWindowIcon(QIcon(str(Path(__file__).parent / "resources" / "nfit-icon.png")))
