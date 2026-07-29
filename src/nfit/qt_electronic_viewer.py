"""Qt windows for Matplotlib electronic-structure figures."""

from __future__ import annotations

from typing import Any

from .qt_viewer_shell import create_viewer_shell

_VIEWER_TITLES = {
    "band_structure": "Band structure",
    "density_of_states": "Density of states",
    "fermi_surface": "Fermi surface",
}


def show_electronic_figure(
    figure: Any,
    *,
    viewer_key: str,
    parent: Any | None = None,
) -> Any:
    """Show an electronic-structure figure beside its settings panel."""

    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
    from PySide6 import QtGui, QtWidgets

    try:
        title = _VIEWER_TITLES[viewer_key]
    except KeyError as exc:
        choices = ", ".join(sorted(_VIEWER_TITLES))
        raise ValueError(
            f"unknown electronic viewer {viewer_key!r}; expected one of {choices}"
        ) from exc

    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow(parent)
    window.setObjectName(f"{viewer_key}_viewer")
    window.setWindowTitle(title)
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key=viewer_key,
    )
    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName(f"{viewer_key}_canvas")
    canvas.setToolTip(f"Interactive {title.lower()} visualization.")
    toolbar = NavigationToolbar2QT(canvas, central)
    toolbar.setObjectName(f"{viewer_key}_toolbar")
    viewport_layout.addWidget(toolbar)
    viewport_layout.addWidget(canvas, 1)
    window.setCentralWidget(central)
    window.resize(1100, 760)
    window._nfit_application = application
    window._nfit_owns_application = owns_application
    window._nfit_figure = figure
    window._nfit_canvas = canvas
    window._nfit_settings_panel = settings
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_close_shortcut = close_shortcut
    window.show()
    return window
