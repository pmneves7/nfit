import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.colors as mcolors
import numpy as np
import pytest

from metallix import (
    FitComparisonModelView,
    FitComparisonResultView,
    PointData4D,
    attach_fit_comparisons,
)
from metallix.mdhisto import MDHistoAxis, MDHistoData
from metallix.plotting import (
    MDHistoSliceViewer,
    _DropdownSelect,
    mdhisto_with_signal_like,
    plot_2d_map,
    plot_energy_cut,
    plot_mdhisto_auto,
    plot_mdhisto_fit_comparison,
    plot_mdhisto_fit_line_comparison,
    plot_mdhisto_line,
    plot_mdhisto_slice,
    plot_q_cut,
    residual_mdhisto,
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


def test_hyspec_test_datasets_example_defaults_to_single_dataset_viewer(monkeypatch):
    import sys

    import examples.plot_hyspec_test_datasets as example

    loaded_paths = []
    captured = {}

    class Viewer:
        def run(self):
            captured["ran"] = True

    def fake_load(path, *, copy_metadata):
        loaded_paths.append((path.name, copy_metadata))
        return path.name

    def fake_slice_viewer(datasets, **kwargs):
        captured["datasets"] = datasets
        captured["kwargs"] = kwargs
        return Viewer()

    monkeypatch.setattr(example, "load_mantid_mdhisto_nxs", fake_load)
    monkeypatch.setattr(example, "slice_viewer", fake_slice_viewer)
    monkeypatch.setattr(example.Path, "exists", lambda _path: True)
    monkeypatch.setattr(sys, "argv", ["plot_hyspec_test_datasets.py"])

    example.main()

    assert loaded_paths == [
        ("1D_test.nxs", False),
        ("2D_test.nxs", False),
        ("4D_test.nxs", False),
    ]
    assert captured["datasets"] == ["1D_test.nxs", "2D_test.nxs", "4D_test.nxs"]
    assert captured["kwargs"]["dataset_names"] == ["1D", "2D", "4D"]
    assert captured["kwargs"]["channel"] == "signal"
    assert captured["ran"]


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
        axes_linewidth=2.25,
        show_histogram_axes=True,
        roi_extents=(-1.0, 1.0, -0.5, 0.5),
    )

    assert len(fig.axes) == 4
    ax_image, ax_ycut, ax_colorbar, ax_xcut = fig.axes
    np.testing.assert_allclose(ax_image.get_xlim(), (-1.0, 1.0))
    np.testing.assert_allclose(ax_image.get_ylim(), (-0.5, 0.5))
    assert ax_image.xaxis.label.get_fontsize() == pytest.approx(13.0)
    assert ax_image.spines["left"].get_linewidth() == pytest.approx(2.25)
    assert ax_colorbar.spines["left"].get_linewidth() == pytest.approx(2.25)
    assert ax_image.xaxis.majorTicks[0].tick2line.get_visible()
    assert ax_image.yaxis.majorTicks[0].tick2line.get_visible()
    assert len(ax_xcut.lines) == 1
    assert len(ax_ycut.lines) == 1


def test_plot_mdhisto_slice_can_render_non_signal_channel():
    data = _tiny_mdhisto_data()

    fig = plot_mdhisto_slice(data, x_dim=3, y_dim=2, channel="multiplicity")

    ax_image, ax_colorbar = fig.axes
    rendered = ax_image.collections[0].get_array()
    np.testing.assert_allclose(rendered, data.num_events[1, 1, :, :])
    assert ax_colorbar.yaxis.label.get_text() == "Multiplicity"


def test_plot_mdhisto_line_and_auto_dispatch_for_single_non_singleton_axis():
    data = _tiny_1d_mdhisto_data()

    ax = plot_mdhisto_line(data, channel="signal")
    auto_ax = plot_mdhisto_auto(data, channel="errors")

    assert ax.get_xlabel() == "[H,H,H] (r.l.u.)"
    assert ax.get_ylabel() == "Signal"
    assert len(ax.lines) == 1
    assert auto_ax.get_ylabel() == "Error"
    assert len(auto_ax.lines) == 1


def test_plot_mdhisto_fit_comparison_renders_data_fit_residual_panels():
    data = _tiny_mdhisto_data()
    fit = mdhisto_with_signal_like(data, data.signal + 0.5)

    fig = plot_mdhisto_fit_comparison(data, fit, x_dim=3, y_dim=2)
    residual = residual_mdhisto(data, fit)

    panel_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert panel_titles == ["Data", "Fit", "Residual"]
    assert len(fig.axes) == 6
    np.testing.assert_allclose(residual.signal, -0.5)


def test_plot_mdhisto_fit_line_comparison_draws_expected_1d_layers():
    data = _tiny_1d_mdhisto_data()
    fit = mdhisto_with_signal_like(data, data.signal + 0.5)

    ax = plot_mdhisto_fit_line_comparison(data, fit)

    assert ax.get_xlabel() == "[H,H,H] (r.l.u.)"
    assert ax.get_ylabel() == "Signal"
    assert ax.containers
    assert any(
        line.get_marker() == "o" and line.get_linestyle() == "None"
        for line in ax.lines
    )
    assert any(
        line.get_marker() == "None" and line.get_linestyle() == "-"
        for line in ax.lines
    )
    assert any(line.get_label() == "residual" for line in ax.lines)


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
    viewer.window.show()
    viewer.app.processEvents()

    assert viewer.controls_scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert viewer.controls_scroll.minimumWidth() >= 430
    assert viewer.controls_scroll.maximumWidth() >= viewer.controls_scroll.sizeHint().width()
    assert viewer.controls_scroll.widget().width() >= viewer.controls_scroll.widget().minimumSizeHint().width()
    assert viewer.dataset_combo.width() > viewer.channel_combo.width()
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


def test_qt_dataset_dropdown_switches_between_loaded_datasets():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data_a = _tiny_mdhisto_data()
    data_b = _tiny_mdhisto_data()
    data_b.signal = data_b.signal + 1000.0

    viewer = QtMDHistoSliceViewer([data_a, data_b], dataset_names=["first", "second"], x_dim=3, y_dim=2)

    assert viewer.dataset_combo.currentText() == "first"
    np.testing.assert_allclose(viewer.image.get_array(), data_a.signal[1, 1, :, :])

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.dataset_index == 1
    assert viewer.data is data_b
    assert viewer.dataset_combo.currentText() == "second"
    np.testing.assert_allclose(viewer.image.get_array(), data_b.signal[1, 1, :, :])
    assert viewer.x_combo.currentText() == "[H,H,0]"
    assert viewer.y_combo.currentText() == "[0,0,L]"


def test_qt_dataset_dropdown_keeps_plot_configs_independent():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data_a = _tiny_mdhisto_data()
    data_b = _tiny_mdhisto_data()
    data_b.signal = data_b.signal + 1000.0

    viewer = QtMDHistoSliceViewer([data_a, data_b], dataset_names=["first", "second"], x_dim=3, y_dim=2)

    viewer.channel_combo.setCurrentText("errors")
    viewer.cmap_combo.setCurrentText("magma")
    viewer.x_combo.setCurrentIndex(1)
    viewer.ax_image.set_xlim(-0.25, 0.25)
    viewer.ax_image.set_ylim(-0.5, 0.5)
    viewer.vmin_spin.setValue(viewer.vmin_spin.value() + 1.0)
    viewer.font_size_spin.setValue(14.0)

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.model.channel == "signal"
    assert viewer.model.cmap == "viridis"
    assert viewer.model.autoscale
    assert viewer.font_size == pytest.approx(12.0)
    assert viewer.x_combo.currentText() == "[H,H,0]"
    assert viewer.y_combo.currentText() == "[0,0,L]"
    assert not np.allclose(viewer.ax_image.get_xlim(), (-0.25, 0.25))

    viewer.cmap_combo.setCurrentText("cividis")
    viewer.ax_image.set_xlim(-1.5, 1.5)

    viewer.dataset_combo.setCurrentIndex(0)

    assert viewer.model.channel == "errors"
    assert viewer.model.cmap == "magma"
    assert not viewer.model.autoscale
    assert viewer.font_size == pytest.approx(14.0)
    assert viewer.x_combo.currentText() == "[H,-H,0]"
    assert viewer.y_combo.currentText() == "[0,0,L]"
    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-0.25, 0.25))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-0.5, 0.5))
    assert "channel='errors'" in viewer.figure_script()

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.model.channel == "signal"
    assert viewer.model.cmap == "cividis"
    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-1.5, 1.5))
    assert "channel='signal'" in viewer.figure_script()


def test_qt_dataset_dropdown_handles_1d_line_and_2d_slice_modes():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    line_data = _tiny_1d_mdhisto_data()
    slice_data = _tiny_mdhisto_data()

    viewer = QtMDHistoSliceViewer([line_data, slice_data], dataset_names=["line", "slice"])

    assert viewer.dataset_combo.currentText() == "line"
    assert viewer.image is None
    assert len(viewer.ax_image.lines) == 1
    assert not viewer.ax_colorbar.get_visible()
    assert not viewer.axes_group.isHidden()
    assert viewer.axis_selector_widget.isHidden()
    assert viewer.hidden_group.isHidden()
    assert viewer.color_group.isHidden()
    assert viewer.tools_group.isHidden()
    assert not viewer.line_group.isHidden()

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.dataset_combo.currentText() == "slice"
    assert viewer.image is not None
    assert viewer.ax_colorbar.get_visible()
    assert not viewer.axes_group.isHidden()
    assert not viewer.color_group.isHidden()
    assert not viewer.tools_group.isHidden()
    assert viewer.line_group.isHidden()
    assert viewer.model.x_dim == 3
    assert viewer.model.y_dim == 2


def test_qt_fit_compare_panel_draws_data_fit_and_optional_residual():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    fit = mdhisto_with_signal_like(data, data.signal + 0.5)
    residual = residual_mdhisto(data, fit)
    attach_fit_comparisons(
        data,
        [
            FitComparisonModelView(
                "constant_background",
                [
                    FitComparisonResultView(
                        "fit 0",
                        fit=fit,
                        data=data,
                        residual=residual,
                    )
                ],
            )
        ],
    )

    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)

    assert not viewer.fit_compare_group.isHidden()
    assert not viewer.fit_compare_check.isChecked()

    viewer.fit_compare_check.setChecked(True)

    assert viewer.fit_compare_model_combo.currentText() == "constant_background"
    assert viewer.fit_compare_result_combo.currentText() == "fit 0"
    assert [axis.get_title() for axis in viewer._compare_axes] == ["Data", "Fit", "Residual"]
    assert viewer.tools_group.isHidden()

    viewer.ax_image.set_xlim(-0.5, 0.5)

    np.testing.assert_allclose(viewer._compare_axes[1].get_xlim(), (-0.5, 0.5))

    viewer.fit_compare_residual_check.setChecked(False)

    assert [axis.get_title() for axis in viewer._compare_axes] == ["Data", "Fit"]


def test_qt_singleton_axes_are_not_controlled():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_2d_mdhisto_data_with_singletons()

    viewer = QtMDHistoSliceViewer(data)

    assert [viewer.x_combo.itemText(i) for i in range(viewer.x_combo.count())] == ["[0,0,L]", "[H,H,H]"]
    assert [viewer.y_combo.itemText(i) for i in range(viewer.y_combo.count())] == ["[0,0,L]", "[H,H,H]"]
    assert viewer.hidden_controls == {}


def test_qt_1d_line_plot_controls_update_rendered_line():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_1d_mdhisto_data())

    assert viewer.marker_face_color == "none"
    assert viewer.marker_edge_width == pytest.approx(1.5)

    viewer.show_errorbars_check.setChecked(False)
    viewer.marker_combo.setCurrentText("square")
    viewer.line_style_combo.setCurrentText("dashed")
    viewer.marker_size_spin.setValue(8.0)
    viewer.line_plot_width_spin.setValue(2.0)
    viewer.marker_edge_width_spin.setValue(1.5)
    viewer.marker_face_color_combo.setCurrentText("orange")
    viewer.line_color_combo.setCurrentText("red")

    line = viewer.ax_image.lines[0]
    assert line.get_marker() == "s"
    assert line.get_linestyle() == "--"
    assert line.get_markersize() == pytest.approx(8.0)
    assert line.get_linewidth() == pytest.approx(2.0)
    assert line.get_markeredgewidth() == pytest.approx(1.5)
    assert line.get_markerfacecolor() == "#ff7f0e"
    assert line.get_markeredgecolor() == "#d62728"
    assert line.get_color() == "#d62728"
    assert "plot_mdhisto_line" in viewer.figure_script()


def test_qt_1d_line_plot_errorbar_caps_share_linewidth():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_1d_mdhisto_data())

    viewer.show_errorbar_caps_check.setChecked(True)
    viewer.errorbar_cap_size_spin.setValue(6.0)
    viewer.line_plot_width_spin.setValue(2.25)

    data_line, caplines, _barcols = viewer.ax_image.containers[0].lines
    assert data_line.get_markerfacecolor() == "none"
    assert len(caplines) == 2
    assert all(capline.get_markersize() == pytest.approx(12.0) for capline in caplines)
    assert all(capline.get_linewidth() == pytest.approx(2.25) for capline in caplines)


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

    assert viewer.font_size == pytest.approx(12.0)
    assert viewer.axis_linewidth == pytest.approx(1.5)

    viewer.font_size_spin.setValue(14.0)

    assert viewer.font_size == pytest.approx(14.0)
    assert viewer.ax_image.xaxis.label.get_fontsize() == pytest.approx(14.0)
    assert viewer.ax_image.yaxis.label.get_fontsize() == pytest.approx(14.0)
    assert viewer.colorbar.ax.yaxis.label.get_fontsize() == pytest.approx(14.0)

    viewer.line_width_spin.setValue(2.5)

    assert viewer.axis_linewidth == pytest.approx(2.5)
    assert viewer.ax_image.spines["left"].get_linewidth() == pytest.approx(2.5)
    assert viewer.ax_xcut.spines["left"].get_linewidth() == pytest.approx(2.5)
    assert viewer.ax_ycut.spines["left"].get_linewidth() == pytest.approx(2.5)
    assert viewer.colorbar.outline.get_linewidth() == pytest.approx(2.5)
    assert viewer.ax_image.xaxis.majorTicks[0].tick2line.get_visible()
    assert viewer.ax_image.yaxis.majorTicks[0].tick2line.get_visible()
    assert viewer.ax_xcut.xaxis.majorTicks[0].tick2line.get_visible()
    assert viewer.ax_ycut.yaxis.majorTicks[0].tick2line.get_visible()


def test_qt_copy_and_save_script_exports_current_display_state():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.metadata["source_file"] = "/tmp/example.nxs"
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.font_size_spin.setValue(12.5)
    viewer.line_width_spin.setValue(2.0)
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
    assert "axes_linewidth=2.0" in script
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


def test_qt_cursor_readout_uses_fixed_labels_and_uncertainty_precision():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.signal[1, 1, 3, 2] = -0.0067
    data.errors[1, 1, 3, 2] = 0.0013
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][2]),
        ydata=float(view["y_centers"][3]),
    )

    viewer._on_motion(event)

    assert viewer.cursor_xy_label.text() == "(x, y) = (0, 0.75)"
    assert viewer.cursor_hkle_label.text() == "(H, K, L, E) = (1.5, -1.5, 0.75, 0.75)"
    assert viewer.cursor_intensity_label.text() == "I = -0.0067 ± 0.0013"


def test_qt_1d_cursor_readout_tracks_nearest_point_and_hkle():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_1d_mdhisto_data()
    data.signal[0, 0, 0, 2] = 3.25
    data.errors[0, 0, 0, 2] = 0.12
    viewer = QtMDHistoSliceViewer(data)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][2]) + 0.01,
        ydata=3.0,
    )

    viewer._on_motion(event)

    assert viewer.cursor_xy_label.text() == "(x, y) = (0.5, 3.25)"
    assert viewer.cursor_hkle_label.text() == "(H, K, L, E) = (0.5, 0.5, 0.5, 3)"
    assert viewer.cursor_intensity_label.text() == "I = 3.25 ± 0.12"


def test_qt_box_tool_visibility_checkbox_controls_rectangle_selector():
    pytest.importorskip("PySide6")
    from metallix.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.show_box_check.isChecked()
    assert viewer.roi_button.isEnabled()
    assert not viewer.rectangle_selector.active

    viewer.show_box_check.setChecked(True)
    assert viewer.hist_axes_check.isChecked()
    viewer.roi_button.setChecked(True)
    assert viewer.rectangle_selector.active
    assert viewer.rectangle_selector.artists[0].get_edgecolor() == pytest.approx(mcolors.to_rgba("#4f8bd6"))

    viewer.roi_button.setChecked(False)

    assert not viewer.rectangle_selector.active
    assert viewer.rectangle_selector.artists[0].get_edgecolor() == pytest.approx(mcolors.to_rgba("#8a8a8a", 0.75))

    viewer.roi_button.setChecked(True)

    viewer.show_box_check.setChecked(False)

    assert viewer.roi_button.isEnabled()
    assert not viewer.roi_button.isChecked()
    assert not viewer.rectangle_selector.active

    viewer.roi_button.setChecked(True)

    assert viewer.show_box_check.isChecked()
    assert viewer.hist_axes_check.isChecked()
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


def _tiny_1d_mdhisto_data() -> MDHistoData:
    signal = np.arange(5, dtype=float).reshape(1, 1, 1, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("[L,L,-2L]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.array([2.5, 3.5]), "meV", "energy"),
            MDHistoAxis("[H,H,H]", np.linspace(0.0, 1.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )


def _tiny_2d_mdhisto_data_with_singletons() -> MDHistoData:
    signal = np.arange(4 * 5, dtype=float).reshape(1, 1, 4, 5)
    return MDHistoData(
        axes=(
            MDHistoAxis("[L,L,-2L]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[H,-H,0]", np.array([-0.5, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("[0,0,L]", np.linspace(-1.0, 1.0, 5), "r.l.u.", "momentum"),
            MDHistoAxis("[H,H,H]", np.linspace(0.0, 1.0, 6), "r.l.u.", "momentum"),
        ),
        signal=signal,
        errors=np.ones_like(signal),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )
