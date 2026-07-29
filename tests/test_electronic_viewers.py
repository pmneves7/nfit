from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nfit import FermiSurfaceResult, FermiSurfaceSheet


def _fermi_surface_result() -> FermiSurfaceResult:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return FermiSurfaceResult(
        target_energy_meV=0.0,
        dimension=3,
        periodic_axes=(0, 1, 2),
        sheets=(
            FermiSurfaceSheet(
                band_index=2,
                vertices_reduced=vertices,
                vertices_inv_angstrom=vertices,
                connectivity=np.asarray([[0, 1, 2]]),
                projected_weights={},
            ),
        ),
        mesh_shape=(10, 10, 10),
        model_digest="model",
        provenance={"display_energy_unit": "eV"},
    )


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
        assert window._nfit_close_shortcut is not None
        window._nfit_close_shortcut.activated.emit()
        assert not window.isVisible()

    assert application is QtWidgets.QApplication.instance()


def test_fermi_surface_renderer_uses_pyvista_triangle_mesh(monkeypatch):
    from nfit.qt_fermi_surface_viewer import _render_fermi_surface

    created_meshes = []
    monkeypatch.setitem(
        sys.modules,
        "pyvista",
        SimpleNamespace(
            PolyData=lambda points, *, faces: created_meshes.append(
                (points, faces)
            )
            or ("mesh", points, faces)
        ),
    )

    class Plotter:
        def __init__(self):
            self.meshes = []
            self.bounds = None
            self.legend = None
            self.text = None
            self.isometric = False

        def clear(self):
            return None

        def set_background(self, _color):
            return None

        def enable_lightkit(self):
            return None

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def show_bounds(self, **kwargs):
            self.bounds = kwargs

        def add_legend(self, **kwargs):
            self.legend = kwargs

        def add_text(self, text, **kwargs):
            self.text = (text, kwargs)

        def view_isometric(self):
            self.isometric = True

        def reset_camera(self):
            return None

    plotter = Plotter()
    _render_fermi_surface(plotter, _fermi_surface_result())

    np.testing.assert_array_equal(created_meshes[0][1], [3, 0, 1, 2])
    assert plotter.meshes[0][1]["label"] == "band 2"
    assert plotter.meshes[0][1]["show_edges"] is False
    assert plotter.bounds["bounds"] == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert plotter.legend is not None
    assert plotter.text[0].endswith("0 eV")
    assert plotter.isometric is True


def test_gpu_fermi_surface_viewer_uses_standard_shell(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])

    class FakeInteractor:
        def __init__(self, parent, **_kwargs):
            self.interactor = QtWidgets.QWidget(parent)
            self.close_count = 0

        def setObjectName(self, name):
            self.interactor.setObjectName(name)

        def screenshot(self, _path=None, *, return_img=False):
            if return_img:
                return np.zeros((12, 16, 3), dtype=np.uint8)

        def close(self):
            self.close_count += 1

    class FakeMainWindow(QtWidgets.QMainWindow):
        signal_close = QtCore.Signal()

        def __init__(self, parent=None, title=None):
            super().__init__(parent)
            self.setWindowTitle(title or "")

        def closeEvent(self, event):
            self.signal_close.emit()
            super().closeEvent(event)

    monkeypatch.setitem(
        sys.modules,
        "pyvistaqt",
        SimpleNamespace(
            MainWindow=FakeMainWindow,
            QtInteractor=FakeInteractor,
        ),
    )
    monkeypatch.setattr(
        "nfit.qt_fermi_surface_viewer._render_fermi_surface",
        lambda *_args, **_kwargs: None,
    )
    from nfit.qt_fermi_surface_viewer import show_fermi_surface_result

    window = show_fermi_surface_result(_fermi_surface_result())
    panel = window.findChild(
        QtWidgets.QGroupBox,
        "fermi_surface_settings_panel",
    )
    summary = window.findChild(
        QtWidgets.QLabel,
        "fermi_surface_mesh_summary",
    )

    assert panel is not None
    assert summary.text().endswith("1 displayed triangles")
    assert window.findChild(
        QtWidgets.QPushButton,
        "fermi_surface_copy_figure",
    ).toolTip()
    assert window.findChild(
        QtWidgets.QPushButton,
        "fermi_surface_save_figure",
    ).toolTip()
    assert window._nfit_close_shortcut is not None
    window._nfit_close_shortcut.activated.emit()
    assert not window.isVisible()
    assert window._nfit_plotter.close_count == 1
