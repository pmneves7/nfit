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
        "susceptibility",
        "response_convergence",
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
        if viewer_key in {"band_structure", "density_of_states"}:
            assert window.findChild(
                QtWidgets.QPushButton,
                f"{viewer_key}_copy_figure",
            ).toolTip()
            assert window.findChild(
                QtWidgets.QPushButton,
                f"{viewer_key}_save_figure",
            ).toolTip()
        window._nfit_close_shortcut.activated.emit()
        assert not window.isVisible()

    assert application is QtWidgets.QApplication.instance()


def test_replacing_dos_figure_matches_the_existing_canvas(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import show_electronic_figure

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    original = Figure()
    original.add_subplot(111).plot([0.0, 1.0], [0.0, 1.0])
    window = show_electronic_figure(
        original,
        viewer_key="density_of_states",
    )
    application.processEvents()
    replacement = Figure()
    replacement.add_subplot(111).plot([0.0, 1.0], [1.0, 0.0])
    window._nfit_replace_figure(replacement)
    canvas = window._nfit_canvas
    expected = np.asarray(
        [
            canvas.width() * canvas.device_pixel_ratio,
            canvas.height() * canvas.device_pixel_ratio,
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        replacement.get_size_inches() * replacement.dpi,
        expected,
        rtol=0.0,
        atol=1.0,
    )
    assert canvas.figure is replacement
    window.close()


def test_replacement_figure_inherits_retina_dpi_before_resize():
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import _fit_replacement_figure_to_canvas

    figure = Figure(dpi=100.0)
    canvas = SimpleNamespace(
        device_pixel_ratio=2.0,
        width=lambda: 800,
        height=lambda: 600,
    )
    _fit_replacement_figure_to_canvas(figure, canvas)

    assert figure.dpi == pytest.approx(200.0)
    np.testing.assert_allclose(figure.get_size_inches(), [8.0, 6.0])


@pytest.mark.parametrize(
    "viewer_key",
    ("band_structure", "density_of_states"),
)
def test_band_and_dos_viewers_copy_and_save_figure(
    monkeypatch,
    tmp_path,
    viewer_key,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.figure import Figure

    from nfit.qt_electronic_viewer import show_electronic_figure

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    figure = Figure()
    figure.add_subplot(111).plot([0.0, 1.0], [0.0, 1.0])
    saved = []
    monkeypatch.setattr(
        figure,
        "savefig",
        lambda path: saved.append(str(path)),
    )
    output_path = tmp_path / f"{viewer_key}.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output_path), "Figures (*.png)"),
    )
    window = show_electronic_figure(figure, viewer_key=viewer_key)
    application.processEvents()

    window.findChild(
        QtWidgets.QPushButton,
        f"{viewer_key}_copy_figure",
    ).click()
    assert not QtWidgets.QApplication.clipboard().pixmap().isNull()
    window.findChild(
        QtWidgets.QPushButton,
        f"{viewer_key}_save_figure",
    ).click()
    assert saved == [str(output_path)]
    window.close()


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
    canvas = window._nfit_canvas
    replacement = Figure()
    replacement_axis = replacement.add_subplot(111)
    replacement_data = replacement_axis.plot([0.0, 1.0], [1.0, 0.0])[0]
    replacement_data.set_gid("nfit-electronic-data")
    window._nfit_replace_figure(replacement)
    assert window._nfit_canvas is canvas
    assert canvas.figure is replacement
    assert window._nfit_figure is replacement
    assert replacement_data.get_linewidth() == pytest.approx(3.0)
    width.setValue(4.0)
    assert replacement_data.get_linewidth() == pytest.approx(4.0)
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
                "dos_sampling_status": "Not certified.",
                "dos_symmetry": "auto",
                "dos_auto_energy_range": False,
                "dos_energy_min_meV": "-0.5",
                "dos_energy_max_meV": "0.5",
                "dos_energy_points": "600",
                "dos_broadening_meV": "0.005",
            },
            "dos_energy_points",
            "800",
        ),
        (
            "fermi_surface",
            {
                "electronic_energy_unit": "eV",
                "fermi_mesh_mode": "spacing",
                "fermi_spacing_inv_angstrom": "0.025",
                "fermi_mesh": "[64, 64, 64]",
                "fermi_energy_meV": "0",
            },
            "fermi_spacing_inv_angstrom",
            "0.02",
        ),
        (
            "susceptibility",
            {
                "plot_q_reduced": "[0.5, 0.5, 0]",
                "plot_energy_min_meV": "-50",
                "plot_energy_max_meV": "50",
                "plot_energy_points": "401",
                "plot_temperature_K": "10",
            },
            "plot_energy_points",
            "501",
        ),
        (
            "response_convergence",
            {
                "plot_q_reduced": "[0.5, 0.5, 0]",
                "plot_energy_min_meV": "-50",
                "plot_energy_max_meV": "50",
                "plot_temperature_K": "10",
                "convergence_mesh_scales": "[0.5, 1.0]",
                "convergence_broadening_scales": "[2.0, 1.0]",
                "convergence_energy_points": "9",
            },
            "convergence_energy_points",
            "17",
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
        sampling_status = window.findChild(
            QtWidgets.QLabel,
            "density_of_states_setting_dos_sampling_status",
        )
        assert sampling_status is not None and sampling_status.toolTip()
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
    if viewer_key == "fermi_surface":
        mode = window.findChild(
            QtWidgets.QComboBox,
            "fermi_surface_setting_fermi_mesh_mode",
        )
        mesh = window.findChild(
            QtWidgets.QLineEdit,
            "fermi_surface_setting_fermi_mesh",
        )
        assert mode is not None and mode.toolTip()
        assert mesh is not None and not mesh.isEnabled()
        mode.setCurrentIndex(mode.findData("size"))
        assert mesh.isEnabled() and not editor.isEnabled()
    apply = window.findChild(
        QtWidgets.QPushButton,
        f"{viewer_key}_apply_settings",
    )
    assert apply is not None and apply.toolTip()
    apply.click()
    assert applied and applied[0][edited_key] == edited_value
    if viewer_key == "density_of_states":
        assert applied[0]["dos_auto_energy_range"] == "true"
    refreshed = dict(values)
    refreshed[edited_key] = "96"
    window._nfit_update_calculation_settings(refreshed)
    refreshed_editor = window.findChild(
        QtWidgets.QLineEdit,
        f"{viewer_key}_setting_{edited_key}",
    )
    assert refreshed_editor.text() == "96"
    window.close()


def test_fermi_surface_renderer_uses_pyvista_triangle_mesh(monkeypatch):
    from nfit.qt_fermi_surface_viewer import (
        FermiSurfaceViewOptions,
        _render_fermi_surface,
    )

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
            return SimpleNamespace()

        def add_legend(self, **kwargs):
            self.legend = kwargs

        def view_isometric(self):
            self.isometric = True

        def reset_camera(self):
            return None

    plotter = Plotter()
    _render_fermi_surface(
        plotter,
        _fermi_surface_result(),
        options=FermiSurfaceViewOptions(
            band_opacity=0.4,
            text_size=18,
            grid_line_width=2.0,
            shading="smooth",
        ),
    )

    np.testing.assert_array_equal(created_meshes[0][1], [3, 0, 1, 2])
    assert plotter.meshes[0][1]["label"] == "band 2"
    assert plotter.meshes[0][1]["opacity"] == pytest.approx(0.4)
    assert plotter.meshes[0][1]["show_edges"] is False
    assert plotter.meshes[0][1]["smooth_shading"] is True
    assert plotter.bounds["bounds"] == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert plotter.bounds["font_size"] == 18
    assert plotter.legend is not None
    assert plotter.isometric is True

    hidden_legend = Plotter()
    _render_fermi_surface(
        hidden_legend,
        _fermi_surface_result(),
        options=FermiSurfaceViewOptions(
            show_legend=False,
            shading="flat",
        ),
    )
    assert hidden_legend.legend is None
    assert hidden_legend.meshes[0][1]["smooth_shading"] is False
    with pytest.raises(ValueError, match="shading"):
        FermiSurfaceViewOptions(shading="gouraud")


def test_sampling_progress_dialog_records_metrics(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    from nfit import SamplingComparison, SamplingProgress
    from nfit.qt_sampling_progress import SamplingProgressDialog

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    progress = SamplingProgressDialog("DOS convergence")
    progress.update(
        SamplingProgress(
            observable="density_of_states",
            iteration=2,
            candidate_count=4,
            mesh=(24, 16, 8),
            phase="completed",
            elapsed_seconds=1.25,
            comparison=SamplingComparison(
                coarse_mesh=(18, 12, 6),
                fine_mesh=(24, 16, 8),
                maximum_relative_error=0.008,
                rms_relative_error=0.004,
                integrated_relative_error=0.002,
                passed=True,
            ),
            consecutive_passes=1,
            required_passes=2,
        )
    )
    assert progress.table.rowCount() == 2
    assert progress.table.item(1, 1).text() == "24 × 16 × 8"
    assert progress.table.item(1, 6).text() == "Pass"
    progress.finish(
        SimpleNamespace(certified=True, chosen_mesh=(24, 16, 8))
    )
    assert progress.close_button.isEnabled()
    progress.dialog.close()


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
    opacity = window.findChild(
        QtWidgets.QDoubleSpinBox,
        "fermi_surface_band_opacity",
    )
    text_size = window.findChild(
        QtWidgets.QSpinBox,
        "fermi_surface_text_size",
    )
    grid_width = window.findChild(
        QtWidgets.QDoubleSpinBox,
        "fermi_surface_grid_line_width",
    )
    shading = window.findChild(
        QtWidgets.QComboBox,
        "fermi_surface_shading",
    )
    legend = window.findChild(
        QtWidgets.QCheckBox,
        "fermi_surface_show_legend",
    )
    assert all(
        control is not None and control.toolTip()
        for control in (opacity, text_size, grid_width, shading, legend)
    )
    opacity.setValue(0.35)
    text_size.setValue(20)
    grid_width.setValue(2.5)
    shading.setCurrentIndex(shading.findData("flat"))
    legend.setChecked(False)
    assert window._nfit_view_options.band_opacity == pytest.approx(0.35)
    assert window._nfit_view_options.text_size == 20
    assert window._nfit_view_options.grid_line_width == pytest.approx(2.5)
    assert window._nfit_view_options.shading == "flat"
    assert window._nfit_view_options.show_legend is False
    render_count = len(rendered)
    replacement = _fermi_surface_result()
    window._nfit_replace_result(replacement)
    assert window._nfit_result is replacement
    assert len(rendered) == render_count + 1
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
