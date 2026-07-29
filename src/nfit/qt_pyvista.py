"""Shared safeguards for PyVistaQt widgets embedded in nfit."""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any


def _qt_version_tuple(version: str) -> tuple[int, ...]:
    values = []
    for item in str(version).split("."):
        try:
            values.append(int(item))
        except ValueError:
            break
    return tuple(values)


def configure_pyvista_interactor(plotter: Any) -> None:
    """Apply platform safeguards before an embedded VTK window first renders."""

    from PySide6 import QtCore

    if (
        platform.system() == "Darwin"
        and _qt_version_tuple(QtCore.qVersion()) >= (6, 10)
    ):
        # Qt 6.10+ can produce a paint-event/render loop for VTK's QWidget
        # interactor on macOS. Besides hanging, repeated Cocoa layer attachment
        # can dereference a stale NSView. VTK !12956 removes this attribute.
        widget = getattr(plotter, "interactor", plotter)
        widget.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_PaintOnScreen,
            False,
        )


def show_then_render(
    window: Any,
    render: Callable[[], None],
) -> None:
    """Show a Qt window, then render once its native surface is available."""

    from PySide6 import QtCore

    window.show()
    timer = QtCore.QTimer(window)
    timer.setSingleShot(True)

    def guarded_render() -> None:
        if window.isVisible():
            render()

    timer.timeout.connect(guarded_render)
    timer.start(0)
    window._nfit_initial_render_timer = timer
