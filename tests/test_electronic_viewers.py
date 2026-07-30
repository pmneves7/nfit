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
        assert panel.minimumWidth() == panel.maximumWidth() == 360
        assert panel.toolTip()
        assert placeholder is not None
        assert canvas is not None
        root = window.centralWidget().layout()
        assert root.itemAt(root.count() - 1).widget() is panel
        assert window._nfit_close_shortcut is not None
        window._nfit_close_shortcut.activated.emit()
        assert not window.isVisible()

    assert application is QtWidgets.QApplication.instance()


def test_electronic_plot_style_sets_publication_frame_and_reference_lines():
    from matplotlib.figure import Figure

    from nfit import ElectronicPlotStyle, apply_electronic_plot_style

    defaults = ElectronicPlotStyle()
    assert defaults.border_width == pytest.approx(1.5)
    assert defaults.fermi_line_width == pytest.approx(1.0)
    assert defaults.symmetry_line_width == pytest.approx(1.0)

    figure = Figure()
    axis = figure.add_subplot(111)
    data = axis.plot([0.0, 1.0], [1.0, 2.0], label="band")[0]
    data.set_gid("nfit-electronic-data")
    fermi = axis.axhline(0.0)
    fermi.set_gid("nfit-fermi-line")
    symmetry = axis.axvline(0.5)
    symmetry.set_gid("nfit-symmetry-line")
    legend = axis.legend()
    style = ElectronicPlotStyle(
        line_color="#123456",
        line_width=2.5,
        marker="s",
        marker_size=7.0,
        marker_face_color="none",
        font_size=15.0,
        border_width=2.0,
        fermi_line_color="#654321",
        fermi_line_width=1.5,
        fermi_line_style=":",
        symmetry_line_color="#abcdef",
        symmetry_line_width=1.2,
        symmetry_line_style="-.",
    )

    assert apply_electronic_plot_style(figure, style) is style
    assert data.get_color() == "#123456"
    assert data.get_linewidth() == pytest.approx(2.5)
    assert data.get_marker() == "s"
    assert data.get_markerfacecolor() == "none"
    assert fermi.get_color() == "#654321"
    assert fermi.get_linestyle() == ":"
    assert symmetry.get_color() == "#abcdef"
    assert symmetry.get_linestyle() == "-."
    assert axis.xaxis.majorTicks[0]._tickdir == "in"
    assert axis.xaxis.majorTicks[0].tick2line.get_visible()
    assert axis.yaxis.majorTicks[0].tick2line.get_visible()
    assert all(
        spine.get_linewidth() == pytest.approx(2.0)
        for spine in axis.spines.values()
    )
    legend = axis.get_legend()
    assert legend.get_frame().get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert legend.get_frame().get_linewidth() == pytest.approx(2.0)
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(15.0)


def test_band_viewer_plot_controls_update_figure(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import show_electronic_figure

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    figure = Figure()
    axis = figure.add_subplot(111)
    data = axis.plot([0.0, 1.0], [0.0, 1.0])[0]
    data.set_gid("nfit-electronic-data")
    projection = axis.scatter([0.5], [0.5])
    projection.set_gid("nfit-electronic-projection")
    projection_key = axis.plot([], [], marker="o", label="d orbital")[0]
    projection_key.set_gid("nfit-electronic-projection-key")
    fermi = axis.axhline(0.0)
    fermi.set_gid("nfit-fermi-line")
    symmetry = axis.axvline(0.5)
    symmetry.set_gid("nfit-symmetry-line")
    axis.legend()

    window = show_electronic_figure(figure, viewer_key="band_structure")
    assert window.findChild(
        QtWidgets.QScrollArea,
        "band_structure_settings_scroll",
    ) is not None
    for object_name in (
        "band_structure_line_color",
        "band_structure_line_width",
        "band_structure_marker",
        "band_structure_marker_size",
        "band_structure_marker_fill_color",
        "band_structure_font_size",
        "band_structure_border_width",
        "band_structure_show_legend",
        "band_structure_show_orbital_projections",
        "band_structure_fermi_line_color",
        "band_structure_fermi_line_width",
        "band_structure_fermi_line_style",
        "band_structure_symmetry_line_color",
        "band_structure_symmetry_line_width",
        "band_structure_symmetry_line_style",
    ):
        control = window.findChild(QtWidgets.QWidget, object_name)
        assert control is not None
        assert control.toolTip()
    marker = window.findChild(QtWidgets.QComboBox, "band_structure_marker")
    assert marker.currentData() == ""
    marker.setCurrentIndex(marker.findData("^"))
    assert data.get_marker() == "^"
    width = window.findChild(
        QtWidgets.QDoubleSpinBox,
        "band_structure_line_width",
    )
    width.setValue(3.0)
    assert data.get_linewidth() == pytest.approx(3.0)
    show_projections = window.findChild(
        QtWidgets.QCheckBox,
        "band_structure_show_orbital_projections",
    )
    show_projections.setChecked(False)
    assert data.get_visible()
    assert not projection.get_visible()
    assert not projection_key.get_visible()
    assert not axis.get_legend().get_visible()
    show_projections.setChecked(True)
    assert projection.get_visible()
    assert projection_key.get_visible()
    show_legend = window.findChild(
        QtWidgets.QCheckBox,
        "band_structure_show_legend",
    )
    show_legend.setChecked(False)
    assert not axis.get_legend().get_visible()
    assert window._nfit_plot_style.show_legend is False
    window.close()


@pytest.mark.parametrize(
    ("viewer_key", "values", "edited_key", "edited_value"),
    [
        (
            "band_structure",
            {
                "band_path_convention": "manual",
                "band_path": '[{"label":"G","k":[0,0,0]},{"label":"X","k":[1,0,0]}]',
                "band_points_per_inv_angstrom": "80",
            },
            "band_points_per_inv_angstrom",
            "120",
        ),
        (
            "density_of_states",
            {
                "electronic_energy_unit": "eV",
                "dos_method": "gaussian",
                "dos_mesh": "[40, 40, 40]",
                "dos_symmetry": "auto",
                "dos_auto_energy_range": False,
                "dos_energy_min_meV": "-0.5",
                "dos_energy_max_meV": "0.5",
                "dos_energy_points": "600",
                "dos_broadening_meV": "0.005",
            },
            "dos_mesh",
            "[48, 48, 48]",
        ),
        (
            "fermi_surface",
            {
                "electronic_energy_unit": "eV",
                "fermi_mesh": "[64, 64, 64]",
                "fermi_energy_meV": "0",
            },
            "fermi_mesh",
            "[72, 72, 72]",
        ),
    ],
)
def test_electronic_viewer_settings_apply_plot_owned_configuration(
    monkeypatch,
    viewer_key,
    values,
    edited_key,
    edited_value,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import show_electronic_figure

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    applied = []
    figure = Figure()
    figure.add_subplot(111)
    window = show_electronic_figure(
        figure,
        viewer_key=viewer_key,
        settings_config=values,
        on_apply_settings=applied.append,
    )
    editor = window.findChild(
        QtWidgets.QLineEdit,
        f"{viewer_key}_setting_{edited_key}",
    )
    assert editor is not None and editor.toolTip()
    editor.setText(edited_value)
    if viewer_key == "density_of_states":
        auto_range = window.findChild(
            QtWidgets.QCheckBox,
            "density_of_states_setting_dos_auto_energy_range",
        )
        minimum = window.findChild(
            QtWidgets.QLineEdit,
            "density_of_states_setting_dos_energy_min_meV",
        )
        maximum = window.findChild(
            QtWidgets.QLineEdit,
            "density_of_states_setting_dos_energy_max_meV",
        )
        assert auto_range is not None and auto_range.toolTip()
        auto_range.setChecked(True)
        assert not minimum.isEnabled()
        assert not maximum.isEnabled()
    apply = window.findChild(
        QtWidgets.QPushButton,
        f"{viewer_key}_apply_settings",
    )
    assert apply is not None and apply.toolTip()
    apply.click()
    assert applied and applied[0][edited_key] == edited_value
    if viewer_key == "density_of_states":
        assert applied[0]["dos_auto_energy_range"] == "true"
    window.close()


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
    rendered = []
    monkeypatch.setattr(
        "nfit.qt_fermi_surface_viewer._render_fermi_surface",
        lambda *_args, **_kwargs: rendered.append(True),
    )
    from nfit.qt_fermi_surface_viewer import show_fermi_surface_result

    window = show_fermi_surface_result(_fermi_surface_result())
    application.processEvents()
    panel = window.findChild(
        QtWidgets.QGroupBox,
        "fermi_surface_settings_panel",
    )
    summary = window.findChild(
        QtWidgets.QLabel,
        "fermi_surface_mesh_summary",
    )

    assert panel is not None
    assert rendered == [True]
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


def test_macos_qt_pyvista_safeguard_clears_paint_on_screen(monkeypatch):
    QtCore = pytest.importorskip("PySide6.QtCore")
    from nfit.qt_pyvista import configure_pyvista_interactor

    calls = []

    class Widget:
        def setAttribute(self, attribute, enabled):
            calls.append((attribute, enabled))

    monkeypatch.setattr("nfit.qt_pyvista.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "nfit.qt_pyvista._qt_version_tuple",
        lambda _version: (6, 11, 1),
    )
    configure_pyvista_interactor(SimpleNamespace(interactor=Widget()))

    assert calls == [
        (QtCore.Qt.WidgetAttribute.WA_PaintOnScreen, False),
    ]
