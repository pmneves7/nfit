from __future__ import annotations

import pytest


def test_electronic_viewers_share_right_settings_panel(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import show_electronic_figure

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    for viewer_key in (
        "band_structure",
        "density_of_states",
        "fermi_surface",
    ):
        figure = Figure()
        figure.add_subplot(111)
        window = show_electronic_figure(figure, viewer_key=viewer_key)
        panel = window.findChild(
            QtWidgets.QGroupBox,
            f"{viewer_key}_settings_panel",
        )
        placeholder = window.findChild(
            QtWidgets.QLabel,
            f"{viewer_key}_settings_placeholder",
        )
        canvas = window.findChild(
            QtWidgets.QWidget,
            f"{viewer_key}_canvas",
        )

        assert panel is not None
        assert panel.title() == "Settings"
        assert panel.minimumWidth() == panel.maximumWidth() == 280
        assert panel.toolTip()
        assert placeholder is not None
        assert canvas is not None
        root = window.centralWidget().layout()
        assert root.itemAt(root.count() - 1).widget() is panel
        window.close()

    assert application is QtWidgets.QApplication.instance()
