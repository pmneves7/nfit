import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.colors as mcolors
import numpy as np
import pytest

from metallix import PointData4D
from metallix.mdhisto import MDHistoAxis, MDHistoData
from metallix.plotting import (
    MDHistoSliceViewer,
    _DropdownSelect,
    plot_2d_map,
    plot_energy_cut,
    plot_mdhisto_slice,
    plot_q_cut,
)


def test_plotting_helpers_return_axes():
    data = PointData4D(
        H=[0.0, 0.1, 0.2, 0.3],
        K=[0.0, 0.0, 0.1, 0.1],
        L=[0.0, 0.0, 0.0, 0.0],
        E=[1.0, 2.0, 3.0, 4.0],
        intensity=[1.0, 2.0, 1.5, 2.5],
        sigma=[0.1, 0.1, 0.1, 0.1],
    )
    assert plot_energy_cut(data) is not None
    assert plot_q_cut(data, "H") is not None
    assert plot_2d_map(data.H, data.K, np.asarray(data.intensity)) is not None


def test_plot_mdhisto_slice_renders_static_figure_with_histogram_cuts():
    data = _tiny_mdhisto_data()

    fig = plot_mdhisto_slice(
        data,
        x_dim=3,
        y_dim=2,
        selections={0: (0.25, 0.75), 1: (1.5, 1.5)},
        integrate_checks={0: True, 1: False},
        xlim=(-1.0, 1.0),
        ylim=(-0.5, 0.5),
        font_size=13.0,
        show_histogram_axes=True,
        roi_extents=(-1.0, 1.0, -0.5, 0.5),
    )

    assert len(fig.axes) == 4
    ax_image, ax_ycut, _ax_colorbar, ax_xcut = fig.axes
    np.testing.assert_allclose(ax_image.get_xlim(), (-1.0, 1.0))
    np.testing.assert_allclose(ax_image.get_ylim(), (-0.5, 0.5))
    assert ax_image.xaxis.label.get_fontsize() == pytest.approx(13.0)
    assert len(ax_xcut.lines) == 1
    assert len(ax_ycut.lines) == 1


def test_plot_mdhisto_slice_can_render_non_signal_channel():
    data = _tiny_mdhisto_data()

    fig = plot_mdhisto_slice(data, x_dim=3, y_dim=2, channel="multiplicity")

    ax_image, ax_colorbar = fig.axes
    rendered = ax_image.collections[0].get_array()
    np.testing.assert_allclose(rendered, data.num_events[1, 1, :, :])
    assert ax_colorbar.yaxis.label.get_text() == "Multiplicity"


def test_mdhisto_slice_viewer_selects_hidden_axis_positions():
    data = _tiny_mdhisto_data()
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.selections[0] = (0.75, 0.75)
    viewer.selections[1] = (1.5, 1.5)

    view = viewer.slice_arrays()

    np.testing.assert_allclose(view["signal"], data.signal[1, 1, :, :])
    np.testing.assert_allclose(view["errors"], data.errors[1, 1, :, :])
    assert view["signal"].shape == (4, 5)


def test_mdhisto_slice_viewer_integrates_hidden_axis_ranges():
    data = _tiny_mdhisto_data()
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.selections[0] = (0.25, 0.75)
    viewer.selections[1] = (1.5, 1.5)
    viewer.integrate_checks[0] = True
    viewer.integrate_checks[1] = False

    view = viewer.slice_arrays()

    np.testing.assert_allclose(view["signal"], np.sum(data.signal[:, 1, :, :], axis=0))
    np.testing.assert_allclose(view["errors"], np.sqrt(np.sum(data.errors[:, 1, :, :] ** 2, axis=0)))
    np.testing.assert_allclose(view["num_events"], np.sum(data.num_events[:, 1, :, :], axis=0))


def test_mdhisto_slice_viewer_blanks_empty_bins_when_integrating_ranges():
    data = _tiny_mdhisto_data()
    data.num_events[:, 1, 2, 3] = 0.0
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.selections[0] = (0.25, 0.75)
    viewer.selections[1] = (1.5, 1.5)
    viewer.integrate_checks[0] = True
    viewer.integrate_checks[1] = False

    view = viewer.slice_arrays()

    assert view["num_events"][2, 3] == 0.0
    assert np.isnan(view["signal"][2, 3])
    assert np.isnan(view["errors"][2, 3])
    assert view["mask"][2, 3]


def test_mdhisto_slice_viewer_auto_color_limits():
    data = _tiny_mdhisto_data()
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    values = np.array([1.0, 2.0, 3.0, 100.0, np.nan])

    viewer.auto_limits = "min/max"
    assert viewer._color_limits(values) == (1.0, 100.0)

    viewer.auto_limits = "N-sigma"
    lo, hi = viewer._color_limits(values)
    assert lo < np.mean(values[np.isfinite(values)]) < hi

    viewer.auto_limits = "N IQR"
    lo, hi = viewer._color_limits(values)
    assert lo < hi

    viewer.auto_limits = "Nth percentile"
    viewer.percentile_n = 10.0
    assert viewer._color_limits(values) == (1.3, 70.90000000000002)


def test_mdhisto_slice_viewer_swaps_display_axes():
    data = _tiny_mdhisto_data()
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show()

    viewer._set_display_dim("x", 2)

    assert viewer.x_dim == 2
    assert viewer.y_dim == 3
    assert viewer.x_radio.value_selected == "[0,0,L]"
    assert viewer.y_radio.value_selected == "[H,H,0]"


def test_mdhisto_slice_viewer_widget_state_matches_color_scale():
    data = _tiny_mdhisto_data()
    viewer = MDHistoSliceViewer(data, x_dim=3, y_dim=2, color_scale="log")
    viewer.show()

    assert isinstance(viewer.image.norm, mcolors.LogNorm)
    assert "Scale: log" in viewer.dropdowns[1].button.label.get_text()


def test_dropdown_select_updates_value_and_hides_options():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    selected = []
    dropdown = _DropdownSelect(fig, [0.1, 0.8, 0.2, 0.04], "Scale", ["linear", "log"], "linear", selected.append)

    dropdown.toggle()
    assert dropdown.option_axes[0].get_visible()
    dropdown.select("log")

    assert selected == ["log"]
    assert dropdown.value == "log"
    assert "Scale: log" in dropdown.button.label.get_text()
    assert not dropdown.option_axes[0].get_visible()


def test_qt_slice_viewer_uses_real_comboboxes_and_swaps_axes():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert isinstance(viewer.cmap_combo, QtWidgets.QComboBox)
    assert isinstance(viewer.scale_combo, QtWidgets.QComboBox)
    assert viewer.scale_combo.currentText() == "linear"

    viewer.x_combo.setCurrentIndex(2)

    assert viewer.model.x_dim == 2
    assert viewer.model.y_dim == 3
    assert viewer.x_combo.currentText() == "[0,0,L]"
    assert viewer.y_combo.currentText() == "[H,H,0]"


def test_slice_viewer_returns_qt_viewer():
    pytest.importorskip("PySide6")
    from metallix.plotting import slice_viewer
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = slice_viewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert isinstance(viewer, QtMDHistoSliceViewer)
    assert viewer.slice_arrays()["signal"].shape == (4, 5)


def test_qt_control_panel_uses_compact_widgets_without_horizontal_scroll():
    pytest.importorskip("PySide6")
    from PySide6 import QtCore

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert viewer.controls_scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert viewer.controls_scroll.maximumWidth() <= 380
    assert viewer.x_min_spin.maximumWidth() <= 118
    assert viewer.vmax_spin.maximumWidth() <= 118
    assert viewer.cmap_combo.maximumWidth() <= 150


def test_qt_channel_dropdown_switches_displayed_channel_and_export_script():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.channel_combo.setCurrentText("errors")

    assert viewer.model.channel == "errors"
    assert viewer.colorbar.ax.yaxis.label.get_text() == "Error"
    np.testing.assert_allclose(viewer.image.get_array(), viewer.slice_arrays()["errors"])
    assert "channel='errors'" in viewer.figure_script()


def test_qt_hidden_axis_sliders_and_spins_stay_linked():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=1)
    controls = viewer.hidden_controls[2]

    controls.slider.set_indices(0, 0, 1)
    controls.slider.changed.emit("low")
    assert controls.slider.value_index == 0

    controls.slider.set_indices(2, controls.slider.low_index, controls.slider.high_index)
    controls.slider.changed.emit("value")

    assert controls.slider.low_index == 2
    assert controls.slider.high_index == 3
    np.testing.assert_allclose(controls.low.value(), controls.centers[2])
    np.testing.assert_allclose(controls.high.value(), controls.centers[3])

    controls.slider.set_indices(controls.slider.value_index, 0, controls.slider.high_index)
    controls.slider.changed.emit("low")

    assert controls.slider.value_index == 2


def test_qt_integrated_axis_width_enables_range_without_changing_value():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=1)
    controls = viewer.hidden_controls[2]
    original_value = controls.value.value()

    controls.width.setValue(1.0)

    assert controls.integrate.isChecked()
    assert controls.value.value() == pytest.approx(original_value)
    assert controls.high.value() - controls.low.value() == pytest.approx(1.0)


def test_qt_autoscale_limits_and_manual_override():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert viewer.autoscale_check.isChecked()
    assert viewer.vmax_spin.value() > viewer.vmin_spin.value()

    viewer.vmin_spin.setValue(viewer.vmin_spin.value() + 1.0)

    assert not viewer.autoscale_check.isChecked()
    assert not viewer.model.autoscale

    viewer.limits_combo.setCurrentText("N-sigma")

    assert viewer.autoscale_check.isChecked()
    assert viewer.model.autoscale

    viewer.limit_n_spin.setValue(2.0)
    assert viewer.model.sigma_n == pytest.approx(2.0)

    viewer.limits_combo.setCurrentText("Nth percentile")
    viewer.limit_n_spin.setValue(5.0)
    assert viewer.model.percentile_n == pytest.approx(5.0)


def test_qt_color_scale_preserves_manual_view_and_power_gamma():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.ax_image.set_xlim(-0.3, 0.7)
    viewer.ax_image.set_ylim(-0.5, 0.5)

    viewer.scale_combo.setCurrentText("power")
    viewer.gamma_spin.setValue(0.8)

    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-0.3, 0.7))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-0.5, 0.5))
    assert not viewer.gamma_spin.isHidden()
    assert viewer.model.power_gamma == pytest.approx(0.8)


def test_qt_repeated_redraws_reuse_colorbar_without_locator_recursion():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    colorbar = viewer.colorbar
    axes_count = len(viewer.figure.axes)

    for value in np.linspace(-1.0, 1.0, 20):
        viewer.model.selections[2] = (float(value), float(value))
        viewer.update_plot()

    assert viewer.colorbar is colorbar
    assert len(viewer.figure.axes) == axes_count
    viewer.canvas.draw()


def test_qt_font_size_control_updates_figure_text():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.font_size_spin.setValue(14.0)

    assert viewer.font_size == pytest.approx(14.0)
    assert viewer.ax_image.xaxis.label.get_fontsize() == pytest.approx(14.0)
    assert viewer.ax_image.yaxis.label.get_fontsize() == pytest.approx(14.0)
    assert viewer.colorbar.ax.yaxis.label.get_fontsize() == pytest.approx(14.0)


def test_qt_copy_and_save_script_exports_current_display_state():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.metadata["source_file"] = "/tmp/example.nxs"
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.font_size_spin.setValue(12.5)
    viewer.hist_axes_check.setChecked(True)
    viewer._set_roi_extents((-1.0, 1.0, -0.5, 0.5), update_cuts=True, draw=False)
    viewer.ax_image.set_xlim(-1.0, 1.0)
    viewer.ax_image.set_ylim(-0.5, 0.5)

    script = viewer.figure_script()
    viewer.copy_script_to_clipboard()

    assert "plot_mdhisto_slice" in script
    assert "load_mantid_mdhisto_nxs('/tmp/example.nxs'" in script
    assert "x_dim='[H,H,0]'" in script
    assert "y_dim='[0,0,L]'" in script
    assert "font_size=12.5" in script
    assert "show_histogram_axes=True" in script
    assert "roi_extents=(-1.0, 1.0, -0.5, 0.5)" in script
    assert QtWidgets.QApplication.clipboard().text() == script


def test_qt_copy_figure_to_clipboard_sets_pixmap():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.copy_figure_to_clipboard()

    assert not QtWidgets.QApplication.clipboard().pixmap().isNull()


def test_qt_view_limit_controls_track_set_and_reset_main_axes():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    np.testing.assert_allclose((viewer.x_min_spin.value(), viewer.x_max_spin.value()), viewer.ax_image.get_xlim())
    np.testing.assert_allclose((viewer.y_min_spin.value(), viewer.y_max_spin.value()), viewer.ax_image.get_ylim())

    viewer.ax_image.set_xlim(-1.0, 1.0)
    viewer.ax_image.set_ylim(-0.75, 0.75)

    np.testing.assert_allclose((viewer.x_min_spin.value(), viewer.x_max_spin.value()), (-1.0, 1.0))
    np.testing.assert_allclose((viewer.y_min_spin.value(), viewer.y_max_spin.value()), (-0.75, 0.75))

    viewer.x_min_spin.setValue(-0.5)
    viewer.x_max_spin.setValue(0.5)
    viewer.y_min_spin.setValue(-0.25)
    viewer.y_max_spin.setValue(0.25)

    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-0.5, 0.5))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-0.25, 0.25))

    viewer.x_reset_button.click()
    viewer.y_reset_button.click()

    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-2.0, 2.0))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-1.0, 1.0))


def test_qt_roi_button_enables_rectangle_selector_and_cursor_hkle():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.rectangle_selector.active
    assert not viewer.show_box_check.isChecked()
    viewer.show_box_check.setChecked(True)
    viewer.roi_button.setChecked(True)
    assert viewer.rectangle_selector.active

    coords = viewer._cursor_hkle(x_idx=2, y_idx=3)

    assert coords["H"] == pytest.approx(1.5)
    assert coords["K"] == pytest.approx(-1.5)
    assert coords["L"] == pytest.approx(0.75)
    assert coords["E"] == pytest.approx(0.75)


def test_qt_box_tool_visibility_checkbox_controls_rectangle_selector():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.show_box_check.isChecked()
    assert not viewer.roi_button.isEnabled()
    assert not viewer.rectangle_selector.active

    viewer.show_box_check.setChecked(True)
    viewer.roi_button.setChecked(True)
    assert viewer.rectangle_selector.active
    assert viewer.rectangle_selector.artists[0].get_edgecolor() == pytest.approx(mcolors.to_rgba("#4f8bd6"))

    viewer.roi_button.setChecked(False)

    assert not viewer.rectangle_selector.active
    assert viewer.rectangle_selector.artists[0].get_edgecolor() == pytest.approx(mcolors.to_rgba("#8a8a8a", 0.75))

    viewer.roi_button.setChecked(True)

    viewer.show_box_check.setChecked(False)

    assert not viewer.roi_button.isEnabled()
    assert not viewer.roi_button.isChecked()
    assert not viewer.rectangle_selector.active

    viewer.show_box_check.setChecked(True)
    viewer.roi_button.setChecked(True)

    assert viewer.roi_button.isEnabled()
    assert viewer.rectangle_selector.active


def test_qt_roi_center_width_controls_sync_with_rectangle_extents():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.show_box_check.setChecked(True)
    viewer.roi_button.setChecked(True)

    viewer._set_roi_extents((-1.0, 1.0, -0.5, 0.5), update_cuts=True, draw=False)

    assert viewer.roi_x_center_spin.value() == pytest.approx(0.0)
    assert viewer.roi_x_width_spin.value() == pytest.approx(2.0)
    assert viewer.roi_y_center_spin.value() == pytest.approx(0.0)
    assert viewer.roi_y_width_spin.value() == pytest.approx(1.0)
    assert len(viewer.ax_xcut.lines) == 1
    assert len(viewer.ax_ycut.lines) == 1

    viewer.roi_x_center_spin.setValue(0.5)
    viewer.roi_x_width_spin.setValue(1.0)

    x0, x1, y0, y1 = viewer._roi_extents
    assert 0.5 * (x0 + x1) == pytest.approx(0.5)
    assert x1 - x0 == pytest.approx(1.0)
    assert y1 - y0 == pytest.approx(1.0)

    viewer.rectangle_selector.extents = (-1.5, -0.5, -0.25, 0.75)
    viewer._on_rectangle(None, None)

    assert viewer.roi_x_center_spin.value() == pytest.approx(-1.0)
    assert viewer.roi_x_width_spin.value() == pytest.approx(1.0)
    assert viewer.roi_y_center_spin.value() == pytest.approx(0.25)
    assert viewer.roi_y_width_spin.value() == pytest.approx(1.0)


def test_qt_histogram_axes_visibility_and_panel_percent_controls():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.hist_axes_check.isChecked()
    assert not viewer.ax_xcut.get_visible()
    assert not viewer.ax_ycut.get_visible()
    assert not viewer.xcut_percent_slider.isEnabled()
    assert not viewer.ycut_percent_slider.isEnabled()

    viewer.hist_axes_check.setChecked(True)

    assert viewer.ax_xcut.get_visible()
    assert viewer.ax_ycut.get_visible()
    assert viewer.xcut_percent_slider.isEnabled()
    assert viewer.ycut_percent_slider.isEnabled()

    viewer.xcut_percent_slider.setValue(30)
    viewer.ycut_percent_slider.setValue(25)

    assert viewer.ax_xcut.get_visible()
    assert viewer.ax_ycut.get_visible()
    assert viewer.xcut_percent == 30
    assert viewer.ycut_percent == 25
    np.testing.assert_allclose(viewer.grid.get_height_ratios(), [1.0, 30.0 / 70.0])
    np.testing.assert_allclose(viewer.grid.get_width_ratios(), [1.0, 25.0 / 75.0, 0.045])


def _tiny_mdhisto_data() -> MDHistoData:
    signal = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("DeltaE", np.array([0.0, 0.5, 1.0]), "meV", "energy"),
            MDHistoAxis("[H,-H,0]", np.array([0.0, 1.0, 2.0, 3.0]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,0,L]", np.linspace(-1.0, 1.0, 5), "r.l.u.", "momentum"),
            MDHistoAxis("[H,H,0]", np.linspace(-2.0, 2.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )
