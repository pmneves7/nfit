import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.colors as mcolors
import numpy as np
import pytest

from nfit import PointData4D
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.plotting import (
    MDHistoSliceViewer,
    _DropdownSelect,
    gaussian_smooth_nan,
    inverse_variance_weighted_profile,
    mdhisto_with_signal_like,
    plot_2d_map,
    plot_energy_cut,
    plot_mdhisto_auto,
    plot_mdhisto_fit_comparison,
    plot_mdhisto_fit_line_comparison,
    plot_mdhisto_line,
    plot_mdhisto_slice,
    plot_mdhisto_waterfall,
    plot_q_cut,
    prepare_mdhisto_waterfall,
    residual_mdhisto,
)


def test_inverse_variance_weighted_profile_returns_propagated_error():
    values = np.asarray([[1.0, 10.0], [3.0, 14.0], [np.nan, 18.0]])
    errors = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, np.nan]])

    mean, uncertainty = inverse_variance_weighted_profile(values, errors, axis=0)

    np.testing.assert_allclose(mean, [2.0, 12.0])
    np.testing.assert_allclose(uncertainty, [1.0 / np.sqrt(2.0), np.sqrt(2.0)])


def test_gaussian_plot_smoothing_preserves_masked_bins():
    values = np.asarray([0.0, np.nan, 10.0, 0.0])
    smoothed = gaussian_smooth_nan(values, 1.0)

    assert np.isnan(smoothed[1])
    assert smoothed[0] > 0.0
    assert smoothed[2] < 10.0


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
    assert len(ax_xcut.containers) == 1
    assert len(ax_ycut.containers) == 1


def test_plot_mdhisto_slice_can_render_non_signal_channel():
    data = _tiny_mdhisto_data()

    fig = plot_mdhisto_slice(data, x_dim=3, y_dim=2, channel="multiplicity")

    ax_image, ax_colorbar = fig.axes
    rendered = ax_image.collections[0].get_array()
    np.testing.assert_allclose(rendered, data.num_events[1, 1, :, :])
    assert ax_colorbar.yaxis.label.get_text() == "Multiplicity"


def test_mdhisto_neutron_channels_use_quantity_symbols_and_units():
    data = _tiny_mdhisto_data()
    data.metadata.update(
        {
            "signal_label": "Scattering cross section",
            "signal_quantity_type": "differential_cross_section",
            "signal_unit": "mbarn/sr/meV/f.u.",
        }
    )
    data.auxiliary_channels["dynamic_susceptibility"] = MDHistoChannel(
        np.ones(data.shape),
        np.full(data.shape, 0.1),
        label="Dynamical susceptibility χ″",
        unit="mu_B^2/meV/f.u.",
        quantity_type="dynamic_susceptibility",
    )

    signal = MDHistoSliceViewer(data, x_dim=3, y_dim=2, channel="signal")
    error = MDHistoSliceViewer(data, x_dim=3, y_dim=2, channel="errors")
    susceptibility = MDHistoSliceViewer(
        data,
        x_dim=3,
        y_dim=2,
        channel="dynamic_susceptibility",
    )

    assert signal._axis_label(0) == "ΔE (meV)"
    assert signal._channel_label() == (
        r"$\mathrm{d}^2\sigma/\mathrm{d}\Omega\,\mathrm{d}E$ "
        "(mbarn/(sr meV f.u.))"
    )
    assert error._channel_label().startswith(
        r"Uncertainty in $\mathrm{d}^2\sigma"
    )
    assert susceptibility._channel_label() == (
        r"$\chi''$ (μ$_{\mathrm{B}}^2$/meV/f.u.)"
    )


def test_plot_mdhisto_line_and_auto_dispatch_for_single_non_singleton_axis():
    data = _tiny_1d_mdhisto_data()

    ax = plot_mdhisto_line(data, channel="signal")
    auto_ax = plot_mdhisto_auto(data, channel="errors")

    assert ax.get_xlabel() == "[H,H,H] (r.l.u.)"
    assert ax.get_ylabel() == r"$I(\mathbf{Q},E)$ (a.u.)"
    assert len(ax.lines) == 1
    assert auto_ax.get_ylabel() == r"Uncertainty in $I(\mathbf{Q},E)$ (a.u.)"
    assert len(auto_ax.lines) == 1


def test_prepare_mdhisto_waterfall_coarsens_axis_with_propagated_errors():
    data = _with_fit_channels(_tiny_2d_mdhisto_data_with_singletons())

    traces = prepare_mdhisto_waterfall(
        data,
        x_dim=3,
        waterfall_dim=2,
        waterfall_step=1.0,
        include_model=True,
    )

    assert len(traces) == 2
    np.testing.assert_allclose(
        traces[0].values,
        np.mean(data.signal[0, 0, :2, :], axis=0),
    )
    np.testing.assert_allclose(traces[0].errors, np.full(5, 1.0 / np.sqrt(2.0)))
    np.testing.assert_allclose(traces[0].model_values, traces[0].values + 0.5)
    assert traces[0].label.endswith("r.l.u.")


def test_prepare_mdhisto_waterfall_uses_value_and_energy_units_for_labels():
    traces = prepare_mdhisto_waterfall(
        _tiny_mdhisto_data(),
        x_dim=3,
        waterfall_dim=0,
        waterfall_step=0.5,
    )

    assert traces[0].label.endswith(" meV")
    assert "ΔE" not in traces[0].label
    assert "DeltaE" not in traces[0].label


def test_waterfall_custom_suffix_replaces_generated_axis_units():
    ax = plot_mdhisto_waterfall(
        _tiny_mdhisto_data(),
        x_dim=3,
        waterfall_dim=0,
        waterfall_step=0.5,
        trace_offset=1.0,
        trace_label_suffix=" K",
    )

    labels = [text.get_text() for text in ax.texts]
    assert labels
    assert all(label.endswith(" K") for label in labels)
    assert all("meV" not in label for label in labels)


def test_plot_mdhisto_waterfall_supports_grouped_1d_data_and_model_lines():
    first = _with_fit_channels(_tiny_1d_mdhisto_data())
    second = _with_fit_channels(_tiny_1d_mdhisto_data())
    second.signal = second.signal + 2.0

    ax = plot_mdhisto_waterfall(
        [first, second],
        dataset_labels=["0.5 meV", "1.4 meV"],
        trace_offset=3.0,
        cmap="plasma",
        marker_face="outline",
        show_model=True,
        show_zero_lines=True,
        trace_label_suffix=" at 6 K",
        trace_label_font_size=14.0,
        trace_label_color="#000000",
    )

    assert len(ax._nfit_waterfall_traces) == 2
    assert ax._nfit_waterfall_offset == pytest.approx(3.0)
    assert {text.get_text() for text in ax.texts} == {
        "0.5 meV at 6 K",
        "1.4 meV at 6 K",
    }
    assert all(text.get_fontsize() == pytest.approx(14.0) for text in ax.texts)
    assert all(text.get_color() == "#000000" for text in ax.texts)
    assert len(ax.containers) == 2
    assert len(ax.lines) >= 4
    data_line = ax.containers[0].lines[0]
    assert data_line.get_markerfacecolor() == data_line.get_markeredgecolor()


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
    assert ax.get_ylabel() == r"$I(\mathbf{Q},E)$ (a.u.)"
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


def test_mdhisto_slice_viewer_reduces_after_fixed_hidden_axis():
    data = _tiny_mdhisto_data()
    data.metadata["fit"] = data.signal * 2.0
    viewer = MDHistoSliceViewer(data, x_dim=1, y_dim=2)
    viewer.selections[0] = (data.axes[0].centers[1], data.axes[0].centers[1])
    viewer.selections[3] = (data.axes[3].centers[0], data.axes[3].centers[-1])
    viewer.integrate_checks[0] = False
    viewer.integrate_checks[3] = True

    view = viewer.slice_arrays()

    expected_signal = np.sum(data.signal[1], axis=2).T
    expected_fit = np.sum(data.metadata["fit"][1], axis=2).T
    np.testing.assert_allclose(view["signal"], expected_signal)
    np.testing.assert_allclose(view["fit"], expected_fit)


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
    assert view["combined_mask"][2, 3]
    assert view["mask"][2, 3]


def test_qt_slice_viewer_apply_masks_toggle_shows_masked_bins():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.mask[1, 1, 2, 3] = True
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.model.selections[0] = (0.75, 0.75)
    viewer.model.selections[1] = (1.5, 1.5)
    viewer.update_plot()

    assert viewer.apply_masks_check.isChecked()
    assert np.isnan(viewer.slice_arrays()["signal"][2, 3])

    viewer.apply_masks_check.setChecked(False)

    assert not np.isnan(viewer.slice_arrays()["signal"][2, 3])
    assert viewer.slice_arrays()["combined_mask"][2, 3]


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

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert isinstance(viewer.cmap_combo, QtWidgets.QComboBox)
    assert isinstance(viewer.scale_combo, QtWidgets.QComboBox)
    assert viewer.scale_combo.currentText() == "linear"

    viewer.x_combo.setCurrentIndex(2)

    assert viewer.model.x_dim == 2
    assert viewer.model.y_dim == 3
    assert viewer.x_combo.currentText() == "[0,0,L]"
    assert viewer.y_combo.currentText() == "[H,H,0]"


def test_qt_slice_viewer_defaults_to_the_first_two_axes():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data())

    assert viewer.view_mode_combo.currentText() == "Slice viewer"
    assert viewer.model.x_dim == 0
    assert viewer.model.y_dim == 1
    assert viewer.x_combo.currentText() == "DeltaE"
    assert viewer.y_combo.currentText() == "[H,-H,0]"
    viewer.window.close()


def test_qt_slice_viewer_effective_1d_defaults_to_its_varying_axis():
    from nfit.qt_slice_viewer import _initial_display_dims

    assert _initial_display_dims(_tiny_1d_mdhisto_data(), -1, 0) == (3, 0)


def test_qt_constant_q_layout_uses_energy_as_its_varying_axis():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    signal = np.linspace(0.0, 1.0, 55)[None, :]
    data = MDHistoData(
        axes=(
            MDHistoAxis("Q", np.asarray([0.55, 0.65]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.linspace(0.0, 5.5, 56), "meV", "energy"),
        ),
        signal=signal,
        errors=np.full_like(signal, 0.1),
        mask=np.zeros_like(signal, dtype=bool),
        num_events=np.ones_like(signal),
    )

    viewer = QtMDHistoSliceViewer(data)

    assert viewer.model.x_dim == 1
    assert viewer.ax_image.lines[0].get_xdata().shape == (55,)
    assert viewer.ax_image.lines[0].get_ydata().shape == (55,)
    viewer.window.close()


def test_qt_slice_viewer_interactive_controls_have_tooltips():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    interactive_classes = (
        QtWidgets.QAbstractButton,
        QtWidgets.QComboBox,
        QtWidgets.QLineEdit,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QSlider,
    )
    ignored_object_names = {"qt_spinbox_lineedit", "qt_toolbar_ext_button"}
    missing = []
    for widget in viewer.window.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, interactive_classes):
            continue
        if widget.objectName() in ignored_object_names:
            continue
        if not widget.toolTip().strip():
            label = widget.text() if hasattr(widget, "text") else widget.objectName()
            missing.append(f"{type(widget).__name__}:{label}")

    assert missing == []


def test_qt_slice_viewer_save_plot_button_uses_project_callback():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    plot_buttons = [
        button
        for button in viewer.window.findChildren(QtWidgets.QPushButton)
        if button.text() == "Save plot"
    ]

    assert plot_buttons == [viewer.save_plot_button]
    assert not viewer.save_plot_button.isEnabled()
    assert not any(
        button.text() == "Plot"
        for button in viewer.window.findChildren(QtWidgets.QToolButton)
    )

    calls = []
    viewer.set_save_plot_callback(lambda: calls.append("saved"))
    viewer.save_plot_button.click()

    assert calls == ["saved"]
    assert viewer.save_plot_button.parentWidget() is viewer.copy_figure_button.parentWidget()


def test_qt_slice_viewer_standard_save_shortcut_uses_project_callback():
    pytest.importorskip("PySide6")
    from PySide6 import QtGui

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.save_project_shortcut.isEnabled()
    assert viewer.save_project_shortcut.key() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Save
    )

    calls = []
    viewer.set_save_project_callback(lambda: calls.append("saved"))
    viewer.save_project_shortcut.activated.emit()

    assert viewer.save_project_shortcut.isEnabled()
    assert calls == ["saved"]


def test_qt_slice_viewer_standard_close_shortcut_closes_window():
    pytest.importorskip("PySide6")
    from PySide6 import QtGui, QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.window.show()
    QtWidgets.QApplication.processEvents()

    assert viewer.window.isVisible()
    assert viewer.close_shortcut.key() in QtGui.QKeySequence.keyBindings(
        QtGui.QKeySequence.StandardKey.Close
    )

    viewer.close_shortcut.activated.emit()
    QtWidgets.QApplication.processEvents()

    assert not viewer.window.isVisible()


def test_qt_slice_viewer_boolean_channels_use_grey_unit_scale_and_reverse():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.mask[1, 1, 2, 3] = True
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2, channel="mask")

    assert "grey" in viewer.model.COLORMAPS
    assert viewer.model.channel == "combined_mask"
    assert isinstance(viewer.cmap_reverse_button, QtWidgets.QPushButton)
    assert viewer.model._color_limits(viewer.model._display_values(viewer.slice_arrays())) == (0.0, 1.0)
    assert viewer.image.cmap.name == "gray"

    viewer.cmap_reverse_button.click()

    assert viewer.model.cmap_reversed is True
    assert viewer.image.cmap.name == "gray_r"


def test_slice_viewer_returns_qt_viewer():
    pytest.importorskip("PySide6")
    from nfit.plotting import slice_viewer
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = slice_viewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert isinstance(viewer, QtMDHistoSliceViewer)
    assert viewer.slice_arrays()["signal"].shape == (4, 5)


def test_qt_control_panel_uses_compact_widgets_without_horizontal_scroll():
    pytest.importorskip("PySide6")
    from PySide6 import QtCore

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.channel_combo.setCurrentText("errors")

    assert viewer.model.channel == "errors"
    assert (
        viewer.colorbar.ax.yaxis.label.get_text()
        == r"Uncertainty in $I(\mathbf{Q},E)$ (a.u.)"
    )
    np.testing.assert_allclose(viewer.image.get_array(), viewer.slice_arrays()["errors"])
    assert "channel='errors'" in viewer.figure_script()


def test_qt_plot_smoothing_is_axis_specific_and_does_not_modify_dataset_values():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    original_signal = data.signal.copy()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    raw_slice = viewer.slice_arrays()["signal"].copy()

    viewer.smoothing_x_spin.setValue(1.0)
    x_smoothed = viewer._current_slice["signal"].copy()
    assert not np.allclose(x_smoothed, raw_slice)
    np.testing.assert_allclose(viewer.slice_arrays()["signal"], raw_slice)
    np.testing.assert_allclose(data.signal, original_signal)

    viewer.smoothing_y_spin.setValue(1.5)
    xy_smoothed = viewer._current_slice["signal"]
    assert not np.allclose(xy_smoothed, x_smoothed)
    script = viewer.figure_script()
    assert "smoothing_sigma_x=1.0" in script
    assert "smoothing_sigma_y=1.5" in script
    assert "plotting only" in viewer.smoothing_x_spin.toolTip()


def test_data_viewer_closes_volume_panel_before_window_children_are_destroyed():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    class Panel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.shutdown_count = 0

        def shutdown(self):
            self.shutdown_count += 1

    panel = Panel()
    viewer.volume_panel = panel
    viewer.content_stack.addWidget(panel)
    viewer.content_stack.setCurrentWidget(panel)
    viewer.window.show()
    viewer.window.close()
    viewer.app.processEvents()

    assert panel.shutdown_count == 1
    import shiboken6

    assert not shiboken6.isValid(panel)
    assert viewer.volume_panel is None


def test_qt_dataset_dropdown_switches_between_loaded_datasets():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    assert viewer.model.x_dim == 0
    assert viewer.model.y_dim == 1


def _with_fit_channels(data: MDHistoData) -> MDHistoData:
    data.metadata["fit"] = np.asarray(data.signal, dtype=float) + 0.5
    data.metadata["residual"] = np.full(data.shape, -0.5, dtype=float)
    return data


def test_qt_show_fit_draws_side_by_side_panels_with_shared_view():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)

    assert viewer.show_fit_check.text() == "Show model"
    assert viewer.unmask_model_check.text() == "Unmask model"
    assert viewer.unmask_model_check.toolTip()
    assert viewer.show_fit_check.isEnabled()
    assert not viewer.unmask_model_check.isEnabled()
    assert not viewer.show_fit_check.isChecked()
    assert not viewer.show_residual_check.isEnabled()

    viewer.show_fit_check.setChecked(True)

    assert viewer.show_residual_check.isEnabled()
    assert viewer.unmask_model_check.isEnabled()
    assert [axis.get_title() for axis in viewer._compare_axes] == ["Data", "Fit"]
    # The box tool remains available in fit compare so cuts can be integrated.
    assert not viewer.tools_group.isHidden()

    viewer.show_residual_check.setChecked(True)

    assert [axis.get_title() for axis in viewer._compare_axes] == ["Data", "Fit", "Residual"]

    viewer.ax_image.set_xlim(-0.5, 0.5)
    viewer._sync_view_limit_controls()

    np.testing.assert_allclose(viewer._compare_axes[1].get_xlim(), (-0.5, 0.5))
    np.testing.assert_allclose(viewer._compare_axes[2].get_xlim(), (-0.5, 0.5))

    viewer.show_fit_check.setChecked(False)

    assert viewer._compare_axes == []
    assert not viewer.show_residual_check.isChecked()
    assert not viewer.show_residual_check.isEnabled()
    assert not viewer.tools_group.isHidden()


def test_qt_unmask_model_extends_2d_panels_box_cuts_and_residuals():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    data.mask[..., 2] = True
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show_fit_check.setChecked(True)
    viewer.show_residual_check.setChecked(True)
    viewer.hist_axes_check.setChecked(True)

    masked_fit = viewer._comparison_panel_model(data, "fit").slice_arrays()["fit"]
    masked_residual = viewer._comparison_panel_model(data, "residual").slice_arrays()[
        "residual"
    ]
    assert np.all(np.isnan(masked_fit[:, 2]))
    assert np.all(np.isnan(masked_residual[:, 2]))

    callback_values = []
    viewer.set_unmask_model_callback(callback_values.append)
    viewer.unmask_model_check.setChecked(True)

    unmasked_fit = viewer._comparison_panel_model(data, "fit").slice_arrays()["fit"]
    unmasked_residual = viewer._comparison_panel_model(data, "residual").slice_arrays()[
        "residual"
    ]
    assert np.all(np.isfinite(unmasked_fit[:, 2]))
    assert np.all(np.isfinite(unmasked_residual[:, 2]))
    fit_cut = next(
        line for line in viewer.ax_fit_cut.get_lines() if line.get_label() == "fit"
    )
    assert np.isfinite(np.asarray(fit_cut.get_ydata(), dtype=float)[2])
    assert callback_values == [True]


def test_qt_unmask_model_extends_1d_model_and_residual():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_1d_mdhisto_data())
    data.mask[0, 0, 0, 2] = True
    viewer = QtMDHistoSliceViewer(data)
    viewer.show_fit_check.setChecked(True)

    masked_fit = next(
        line for line in viewer.ax_image.get_lines() if line.get_label() == "fit"
    )
    assert np.isnan(np.asarray(masked_fit.get_ydata(), dtype=float)[2])

    viewer.unmask_model_check.setChecked(True)
    unmasked_fit = next(
        line for line in viewer.ax_image.get_lines() if line.get_label() == "fit"
    )
    assert np.isfinite(np.asarray(unmasked_fit.get_ydata(), dtype=float)[2])
    assert viewer.current_plot_settings()["unmask_model"] is True

    viewer.show_residual_check.setChecked(True)
    residual = viewer._slice_1d_channel(viewer._current_slice, "residual")
    assert residual is not None
    assert np.isfinite(residual[2])


def test_qt_histogram_layout_noop_outside_standard_grid():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show_fit_check.setChecked(True)

    # Fit compare uses a non-3-column grid. Even if cut-axis references are
    # stale (as can happen after a figure rebuild on viewer reuse), applying
    # the standard histogram layout must not touch or clobber this grid.
    assert viewer._plot_layout_mode[0] == "fit_compare"
    viewer.ax_xcut = viewer._compare_axes[0]
    viewer.ax_ycut = viewer._compare_axes[0]
    ncols = viewer.grid.ncols
    viewer._apply_histogram_axes_layout(draw=False)  # must not raise
    assert viewer.grid.ncols == ncols


def test_qt_reopen_after_fit_compare_does_not_crash():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show_fit_check.setChecked(True)
    viewer.hist_axes_check.setChecked(True)  # fit compare with box-tool cuts

    # Reopening the data viewer reuses the widget via replace_datasets, which
    # restores dataset state and re-applies the histogram layout.
    fresh = _with_fit_channels(_tiny_mdhisto_data())
    viewer.replace_datasets([fresh], dataset_names=["first"])  # must not raise


def test_qt_replace_datasets_enables_fit_controls_when_channels_appear():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), dataset_names=["scan"], x_dim=3, y_dim=2)
    assert not viewer.show_fit_check.isEnabled()

    viewer.replace_datasets(
        [_with_fit_channels(_tiny_mdhisto_data())],
        dataset_names=["scan"],
        selected_dataset_name="scan",
    )

    assert viewer.show_fit_check.isEnabled()
    viewer.show_fit_check.setChecked(True)
    assert viewer.show_residual_check.isEnabled()


def test_qt_show_fit_checkbox_disabled_without_fit_channels():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert not viewer.show_fit_check.isEnabled()
    assert not viewer.show_residual_check.isEnabled()


def test_qt_fit_compare_box_tool_draws_overlaid_data_fit_cut_and_residual_cut():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show_fit_check.setChecked(True)

    # No cut axes until the histogram box tool is enabled.
    assert viewer.ax_fit_cut is None

    viewer.hist_axes_check.setChecked(True)
    assert viewer.ax_fit_cut is not None
    assert viewer.ax_ycut is not None
    assert viewer.ax_residual_cut is None
    labels = [line.get_label() for line in viewer.ax_fit_cut.get_lines()]
    assert "fit" in labels  # integrated fit line overlaid on the data cut
    y_labels = [line.get_label() for line in viewer.ax_ycut.get_lines()]
    assert "fit" in y_labels  # vertical fit cut is restored at the far right
    assert len(viewer.ax_fit_cut.containers) == 1
    assert len(viewer.ax_ycut.containers) == 1

    # Enabling residuals adds a separate residual cut axes.
    viewer.show_residual_check.setChecked(True)
    assert viewer.ax_residual_cut is not None
    assert viewer.ax_residual_ycut is not None
    assert [axis.get_title() for axis in viewer._compare_axes] == ["Data", "Fit", "Residual"]
    viewer.canvas.draw()
    data_box = viewer._compare_axes[0].get_position()
    residual_box = viewer._compare_axes[2].get_position()
    fit_cut_box = viewer.ax_fit_cut.get_position()
    residual_cut_box = viewer.ax_residual_cut.get_position()
    ycut_box = viewer.ax_ycut.get_position()
    residual_ycut_box = viewer.ax_residual_ycut.get_position()
    assert fit_cut_box.x0 == pytest.approx(data_box.x0)
    assert fit_cut_box.x1 == pytest.approx(data_box.x1)
    assert residual_cut_box.x0 == pytest.approx(residual_box.x0)
    assert residual_cut_box.x1 == pytest.approx(residual_box.x1)
    assert ycut_box.x0 > residual_box.x1
    assert residual_ycut_box.x0 > ycut_box.x1
    assert ycut_box.y0 == pytest.approx(data_box.y0)
    assert ycut_box.y1 == pytest.approx(data_box.y1)
    assert residual_ycut_box.y0 == pytest.approx(residual_box.y0)
    assert residual_ycut_box.y1 == pytest.approx(residual_box.y1)

    # Dragging a box updates both cut axes.
    viewer._set_roi_extents((-1.5, 1.5, -0.5, 1.5), update_cuts=True, draw=True)
    assert any(line.get_label() == "fit" for line in viewer.ax_fit_cut.get_lines())
    assert any(line.get_label() == "fit" for line in viewer.ax_ycut.get_lines())
    assert len(viewer.ax_residual_cut.get_lines()) > 0
    assert len(viewer.ax_residual_ycut.get_lines()) > 0

    # The cut-height slider resizes the cut region.
    before = list(viewer.grid.get_height_ratios())
    viewer.xcut_percent_slider.setValue(min(viewer.xcut_percent + 10, 45))
    after = list(viewer.grid.get_height_ratios())
    assert after[1] >= before[1]

    # Turning the box tool off removes the cut axes.
    viewer.hist_axes_check.setChecked(False)
    assert viewer.ax_fit_cut is None
    assert viewer.ax_ycut is None
    assert viewer.ax_residual_cut is None
    assert viewer.ax_residual_ycut is None


def test_qt_fit_compare_box_extents_survive_fit_and_residual_toggles():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.hist_axes_check.setChecked(True)
    viewer._set_roi_extents((-1.2, 0.8, -0.25, 1.75), update_cuts=True, draw=False)

    expected = viewer._roi_extents
    expected_x_center = viewer.roi_x_center_spin.value()
    expected_x_width = viewer.roi_x_width_spin.value()
    expected_y_center = viewer.roi_y_center_spin.value()
    expected_y_width = viewer.roi_y_width_spin.value()

    viewer.show_fit_check.setChecked(True)
    assert viewer._roi_extents == pytest.approx(expected)
    assert viewer.rectangle_selector.extents == pytest.approx(expected)
    assert viewer.roi_x_center_spin.value() == pytest.approx(expected_x_center)
    assert viewer.roi_x_width_spin.value() == pytest.approx(expected_x_width)
    assert viewer.roi_y_center_spin.value() == pytest.approx(expected_y_center)
    assert viewer.roi_y_width_spin.value() == pytest.approx(expected_y_width)

    viewer.show_residual_check.setChecked(True)
    assert viewer._roi_extents == pytest.approx(expected)
    assert viewer.rectangle_selector.extents == pytest.approx(expected)
    assert viewer.roi_x_center_spin.value() == pytest.approx(expected_x_center)
    assert viewer.roi_x_width_spin.value() == pytest.approx(expected_x_width)
    assert viewer.roi_y_center_spin.value() == pytest.approx(expected_y_center)
    assert viewer.roi_y_width_spin.value() == pytest.approx(expected_y_width)

    viewer.show_residual_check.setChecked(False)
    viewer.show_fit_check.setChecked(False)
    assert viewer._roi_extents == pytest.approx(expected)
    assert viewer.rectangle_selector.extents == pytest.approx(expected)
    assert viewer.roi_x_center_spin.value() == pytest.approx(expected_x_center)
    assert viewer.roi_x_width_spin.value() == pytest.approx(expected_x_width)
    assert viewer.roi_y_center_spin.value() == pytest.approx(expected_y_center)
    assert viewer.roi_y_width_spin.value() == pytest.approx(expected_y_width)


def test_qt_1d_show_fit_draws_line_behind_data_and_residual_axes():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_1d_mdhisto_data())
    viewer = QtMDHistoSliceViewer(data)

    assert viewer.show_fit_check.isEnabled()
    assert viewer.residual_split_slider.isHidden() or not viewer.residual_split_slider.isVisible()

    viewer.show_fit_check.setChecked(True)

    fit_lines = [
        line
        for line in viewer.ax_image.get_lines()
        if line.get_label() == "fit"
    ]
    assert len(fit_lines) == 1
    assert fit_lines[0].get_linestyle() == "-"
    assert fit_lines[0].get_zorder() < 2.0
    np.testing.assert_allclose(fit_lines[0].get_ydata(), data.signal.reshape(-1) + 0.5)
    assert viewer.ax_residual is None

    viewer.show_residual_check.setChecked(True)

    assert viewer.ax_residual is not None
    assert viewer._plot_layout_mode == ("residual_1d", 2)
    assert viewer.ax_residual.get_ylabel() == "Res. (σ)"
    viewer.window.resize(900, 620)
    viewer.window.show()
    viewer.app.processEvents()
    assert viewer.controls_scroll.widget().width() <= viewer.controls_scroll.viewport().width()
    assert viewer.save_script_button.geometry().right() <= viewer.save_script_button.parentWidget().width()

    ratios_before = list(viewer.grid.get_height_ratios())
    viewer.residual_split_slider.setValue(60)
    ratios_after = list(viewer.grid.get_height_ratios())
    assert ratios_after[1] > ratios_before[1]

    viewer.fit_line_color_combo.setCurrentText("green")
    viewer.fit_line_width_spin.setValue(3.5)
    fit_lines = [line for line in viewer.ax_image.get_lines() if line.get_label() == "fit"]
    assert fit_lines[0].get_color() == "#2ca02c"
    assert fit_lines[0].get_linewidth() == pytest.approx(3.5)

    viewer.show_residual_check.setChecked(False)

    assert viewer._plot_layout_mode == ("standard", 1)


def test_qt_singleton_axes_are_not_controlled():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_2d_mdhisto_data_with_singletons()

    viewer = QtMDHistoSliceViewer(data)

    assert [viewer.x_combo.itemText(i) for i in range(viewer.x_combo.count())] == ["[0,0,L]", "[H,H,H]"]
    assert [viewer.y_combo.itemText(i) for i in range(viewer.y_combo.count())] == ["[0,0,L]", "[H,H,H]"]
    assert viewer.hidden_controls == {}


def test_qt_1d_line_plot_controls_update_rendered_line():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_1d_mdhisto_data())

    viewer.show_errorbar_caps_check.setChecked(True)
    viewer.errorbar_cap_size_spin.setValue(6.0)
    viewer.line_plot_width_spin.setValue(2.25)

    data_line, caplines, _barcols = viewer.ax_image.containers[0].lines
    assert data_line.get_markerfacecolor() == "none"
    assert len(caplines) == 2
    assert all(capline.get_markersize() == pytest.approx(12.0) for capline in caplines)
    assert all(capline.get_linewidth() == pytest.approx(2.25) for capline in caplines)


def test_qt_1d_auxiliary_channel_draws_propagated_errorbars():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_1d_mdhisto_data()
    propagated = np.asarray(data.errors, dtype=float) * 2.5
    data.auxiliary_channels["derived"] = MDHistoChannel(
        values=np.asarray(data.signal, dtype=float) * 3.0,
        errors=propagated,
        label="Derived",
        unit="arb. units",
    )
    viewer = QtMDHistoSliceViewer(data)

    viewer.channel_combo.setCurrentText("derived")

    np.testing.assert_allclose(
        viewer._current_slice["derived_errors"].reshape(-1),
        propagated.reshape(-1),
    )
    assert len(viewer.ax_image.containers) == 1
    data_line, _caplines, bar_collections = viewer.ax_image.containers[0].lines
    assert data_line.get_ydata().size == propagated.size
    assert len(bar_collections[0].get_segments()) == propagated.size


def test_qt_waterfall_mode_exposes_controls_and_exports_script():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_2d_mdhisto_data_with_singletons())
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)

    viewer.marker_face_color_combo.setCurrentText("orange")
    viewer.view_mode_combo.setCurrentIndex(1)

    assert viewer._plot_layout_mode == ("waterfall", 1)
    assert viewer.marker_face_color == "none"
    assert viewer.marker_face_color_combo.currentText() == "none"
    assert all(
        line.get_markerfacecolor() == "none"
        for line in viewer.ax_image.lines
        if line.get_marker() not in {"None", "none", ""}
    )
    assert not viewer.waterfall_group.isHidden()
    assert not viewer.waterfall_step_spin.isHidden()
    assert viewer.waterfall_offset_auto_check.isChecked()
    assert viewer.waterfall_offset_auto is True
    for control in (
        viewer.waterfall_source_label,
        viewer.waterfall_step_spin,
        viewer.waterfall_step_slider,
        viewer.waterfall_step_auto_check,
        viewer.waterfall_offset_spin,
        viewer.waterfall_offset_slider,
        viewer.waterfall_offset_auto_check,
        viewer.waterfall_cmap_combo,
        viewer.waterfall_color_range_slider,
        viewer.waterfall_reverse_check,
        viewer.waterfall_zero_check,
        viewer.waterfall_zero_color_combo,
        viewer.waterfall_zero_style_combo,
        viewer.waterfall_zero_width_spin,
        viewer.waterfall_model_color_combo,
        viewer.waterfall_trace_labels_check,
        viewer.waterfall_trace_label_suffix_edit,
        viewer.waterfall_trace_label_font_size_spin,
        viewer.waterfall_trace_label_color_combo,
    ):
        assert control.toolTip().strip()
    assert len(viewer._current_waterfall_traces) == 4
    assert viewer.waterfall_offset == pytest.approx(
        0.5
        * max(
            np.nanmax(np.abs(trace.values))
            for trace in viewer._current_waterfall_traces
        )
    )

    viewer.show_fit_check.setChecked(True)
    viewer.waterfall_cmap_combo.setCurrentText("plasma")
    viewer.waterfall_color_range_slider.set_values(100, 850)
    viewer.waterfall_color_range_slider.changed.emit(100, 850)
    viewer.waterfall_zero_style_combo.setCurrentText("dotted")
    viewer.waterfall_model_color_combo.setCurrentText("black")
    viewer.waterfall_trace_label_suffix_edit.setText(" at 6 K")
    viewer.waterfall_trace_label_font_size_spin.setValue(14.0)
    viewer.waterfall_trace_label_color_combo.setCurrentText("red")
    viewer.marker_face_color_combo.setCurrentText("outline")
    viewer.waterfall_offset_slider.setValue(250)
    viewer.waterfall_step_slider.setValue(600)

    settings = viewer.current_plot_settings()
    script = viewer.figure_script()
    assert settings["view_mode"] == "waterfall"
    assert settings["waterfall_cmap"] == "plasma"
    assert settings["waterfall_color_min"] == pytest.approx(0.1)
    assert settings["waterfall_color_max"] == pytest.approx(0.85)
    assert settings["waterfall_zero_style"] == ":"
    assert settings["waterfall_model_color"] == "#000000"
    assert settings["waterfall_trace_label_suffix"] == " at 6 K"
    assert settings["waterfall_trace_label_font_size"] == pytest.approx(14.0)
    assert settings["waterfall_trace_label_color"] == "#d62728"
    assert settings["marker_face_color"] == "outline"
    assert 0.0 <= viewer.waterfall_offset <= max(
        np.nanmax(np.abs(trace.values))
        for trace in viewer._current_waterfall_traces
    )
    assert "plot_mdhisto_waterfall" in script
    compile(script, "waterfall_figure.py", "exec")

    restored = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    restored.apply_plot_settings(settings)
    assert restored.view_mode_combo.currentText() == "Waterfall"
    assert restored.waterfall_cmap == "plasma"
    assert restored.waterfall_color_min == pytest.approx(0.1)
    assert restored.waterfall_color_max == pytest.approx(0.85)
    assert restored.waterfall_zero_style == ":"
    assert restored.waterfall_trace_label_suffix == " at 6 K"
    assert restored.waterfall_trace_label_font_size == pytest.approx(14.0)
    assert restored.waterfall_trace_label_color == "#d62728"
    assert restored.marker_face_color == "outline"

    viewer.view_mode_combo.setCurrentIndex(0)
    assert viewer.marker_face_color == "#ff7f0e"
    viewer.view_mode_combo.setCurrentIndex(1)
    assert viewer.marker_face_color == "outline"


def test_qt_waterfall_half_max_stays_enabled_when_initial_range_shrinks():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_2d_mdhisto_data_with_singletons()
    data.signal *= 1.0e-3
    data.errors *= 1.0e-3
    viewer = QtMDHistoSliceViewer(data)

    viewer.view_mode_combo.setCurrentText("Waterfall")

    assert viewer.waterfall_offset_spin.maximum() < 1.0
    assert viewer.waterfall_offset_auto is True
    assert viewer.waterfall_offset_auto_check.isChecked()
    assert viewer.waterfall_offset == pytest.approx(
        0.5
        * max(
            np.nanmax(np.abs(trace.values))
            for trace in viewer._current_waterfall_traces
        )
    )


def test_qt_waterfall_mode_groups_compatible_1d_datasets():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    first = _with_fit_channels(_tiny_1d_mdhisto_data())
    second = _with_fit_channels(_tiny_1d_mdhisto_data())
    viewer = QtMDHistoSliceViewer(
        [first, second],
        dataset_names=["0.5 meV", "1.4 meV"],
    )

    viewer.view_mode_combo.setCurrentIndex(1)

    assert viewer.waterfall_source_dataset_names() == ["0.5 meV", "1.4 meV"]
    assert [trace.label for trace in viewer._current_waterfall_traces] == [
        "0.5 meV",
        "1.4 meV",
    ]
    assert viewer.waterfall_step_spin.isHidden()
    assert viewer.y_combo.isHidden()
    assert viewer.current_plot_settings()["waterfall_dataset_names"] == [
        "0.5 meV",
        "1.4 meV",
    ]


def test_qt_waterfall_groups_1d_datasets_by_project_group_key():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    datasets = [_tiny_1d_mdhisto_data() for _index in range(4)]
    viewer = QtMDHistoSliceViewer(
        datasets,
        dataset_names=["g1 a", "g1 b", "g2 a", "g2 b"],
        dataset_group_keys=["Group1", "Group1", "Group2", "Group2"],
    )
    viewer.dataset_combo.setCurrentIndex(2)
    viewer.view_mode_combo.setCurrentIndex(1)

    assert viewer.waterfall_source_dataset_names() == ["g2 a", "g2 b"]
    assert [trace.label for trace in viewer._current_waterfall_traces] == [
        "g2 a",
        "g2 b",
    ]


def test_qt_hidden_axis_sliders_and_spins_stay_linked():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=1)
    controls = viewer.hidden_controls[2]
    original_value = controls.value.value()

    controls.width.setValue(1.0)

    assert controls.integrate.isChecked()
    assert controls.value.value() == pytest.approx(original_value)
    assert controls.high.value() - controls.low.value() == pytest.approx(1.0)


def test_qt_autoscale_limits_and_manual_override():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.copy_figure_to_clipboard()

    assert not QtWidgets.QApplication.clipboard().pixmap().isNull()


def test_qt_view_limit_controls_track_set_and_reset_main_axes():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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


def test_qt_cursor_hkle_uses_rebin_basis_for_projected_axes():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = MDHistoData(
        axes=(
            MDHistoAxis("[H,H,H]", np.array([0.0, 1.0, 2.0]), "r.l.u.", "momentum"),
            MDHistoAxis("[L,L,-2L]", np.array([1.0, 2.0, 3.0]), "r.l.u.", "momentum"),
            MDHistoAxis("[H,-H,0]", np.array([0.0, 0.5]), "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", np.array([1.5, 1.8]), "meV", "energy"),
        ),
        signal=np.ones((2, 2, 1, 1)),
        errors=np.ones((2, 2, 1, 1)),
        mask=np.zeros((2, 2, 1, 1), dtype=bool),
        num_events=np.ones((2, 2, 1, 1)),
        metadata={
            "rebin": {
                "vectors": [
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, -2.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        },
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1)

    coords = viewer._cursor_hkle(x_idx=1, y_idx=0)

    assert coords == pytest.approx({"H": 3.25, "K": 2.75, "L": -1.5, "E": 1.65})


def test_qt_cursor_readout_uses_fixed_labels_and_uncertainty_precision():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    assert viewer.cursor_q_label.text() == "|Q| = ? Å⁻¹"
    assert viewer.cursor_intensity_label.text() == "Signal = -0.0067 ± 0.0013"
    assert viewer.ax_image.format_coord(1.0, 2.0) == ""


def test_qt_cursor_readout_formats_q_modulus_when_lattice_matrix_is_available():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.metadata["rlu_to_inv_angstrom_matrix"] = np.eye(3).tolist()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][2]),
        ydata=float(view["y_centers"][3]),
    )

    viewer._on_motion(event)

    assert viewer.cursor_q_label.text() == "|Q| = 2.25 Å⁻¹"


def test_qt_cursor_readout_hides_crystal_coordinates_for_powder_and_magnetization():
    pytest.importorskip("PySide6")

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    powder = _tiny_mdhisto_data()
    powder.metadata["nfit_data_type"] = "powder_inelastic"
    powder_viewer = QtMDHistoSliceViewer(powder, x_dim=3, y_dim=2)

    assert powder_viewer.cursor_hkle_label.isHidden()
    assert not powder_viewer.cursor_q_label.isHidden()

    magnetization = PointListData(
        columns={
            "Temperature": np.array([2.0, 4.0]),
            "Magnetic Field": np.array([1.0, 1.0]),
            "Moment": np.array([0.1, 0.2]),
            "Moment error": np.array([0.01, 0.01]),
        },
        units={"Temperature": "K", "Magnetic Field": "T"},
        coordinate_names=["Temperature", "Magnetic Field"],
        channels=[{"label": "Moment", "value": "Moment", "error": "Moment error"}],
        metadata={"nfit_data_type": "magnetization"},
    )
    magnetization_viewer = QtMDHistoSliceViewer(magnetization)

    assert magnetization_viewer.cursor_hkle_label.isHidden()
    assert magnetization_viewer.cursor_q_label.isHidden()


def test_qt_point_list_show_fit_draws_line_and_residual_axes():
    pytest.importorskip("PySide6")

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    temperature = np.linspace(1.0, 10.0, 8)
    data = PointListData(
        columns={
            "Temperature": temperature,
            "Moment": np.full(8, 2.0),
            "Moment error": np.full(8, 0.1),
            "fit": np.full(8, 2.0),
            "residual": np.zeros(8),
        },
        units={"Temperature": "K"},
        coordinate_names=["Temperature"],
        channels=[
            {"label": "Moment", "value": "Moment", "error": "Moment error"},
            {"label": "fit", "value": "fit", "error": None},
            {"label": "residual", "value": "residual", "error": None},
        ],
        metadata={"nfit_data_type": "magnetization"},
    )
    viewer = QtMDHistoSliceViewer(data)

    assert viewer._is_effective_1d()
    assert viewer.show_fit_check.isEnabled()
    # the fit/residual columns should not appear as selectable data channels
    channel_items = [viewer.channel_combo.itemText(i) for i in range(viewer.channel_combo.count())]
    assert channel_items == ["Moment", "fit", "residual"]

    viewer.show_fit_check.setChecked(True)
    fit_lines = [line for line in viewer.ax_image.get_lines() if line.get_label() == "fit"]
    assert len(fit_lines) == 1
    np.testing.assert_allclose(fit_lines[0].get_ydata(), np.full(8, 2.0))

    viewer.show_residual_check.setChecked(True)
    assert viewer._plot_layout_mode == ("residual_1d", 2)
    assert viewer.ax_residual is not None


def test_qt_point_list_model_control_follows_fitted_channel():
    pytest.importorskip("PySide6")

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    temperature = np.linspace(10.0, 30.0, 3)
    data = PointListData(
        columns={
            "Temperature": temperature,
            "Moment": np.array([1.0, 0.8, 0.6]),
            "Susceptibility": np.array([0.1, 0.08, 0.06]),
            "fit": np.array([0.11, 0.09, 0.07]),
            "residual": np.array([-1.0, -1.0, -1.0]),
        },
        units={"Temperature": "K"},
        coordinate_names=["Temperature"],
        channels=[
            {"label": "Moment", "value": "Moment", "error": None},
            {
                "label": "Susceptibility",
                "value": "Susceptibility",
                "error": None,
            },
            {"label": "fit", "value": "fit", "error": None},
            {"label": "residual", "value": "residual", "error": None},
        ],
        metadata={
            "viewer_fit_channel_map": {"Susceptibility": "fit"},
            "viewer_residual_channel_map": {"Susceptibility": "residual"},
        },
    )
    viewer = QtMDHistoSliceViewer(data)

    assert viewer.channel_combo.currentText() == "Moment"
    assert not viewer.show_fit_check.isEnabled()
    viewer.channel_combo.setCurrentText("Susceptibility")
    assert viewer.show_fit_check.isEnabled()
    viewer.show_fit_check.setChecked(True)
    assert len(
        [line for line in viewer.ax_image.get_lines() if line.get_label() == "fit"]
    ) == 1
    viewer.channel_combo.setCurrentText("Moment")
    assert not viewer.show_fit_check.isEnabled()
    assert not viewer.show_fit_check.isChecked()
    viewer.window.close()


def test_qt_powder_point_cursor_readout_uses_q_column():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = PointListData(
        columns={
            "q": np.array([0.25, 0.6, 0.9]),
            "Signal": np.array([1.0, 2.0, 3.0]),
            "Error": np.array([0.1, 0.1, 0.1]),
        },
        units={"q": "Å⁻¹"},
        coordinate_names=["q"],
        channels=[{"label": "Signal", "value": "Signal", "error": "Error"}],
        metadata={"nfit_data_type": "powder_elastic"},
    )
    viewer = QtMDHistoSliceViewer(data)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][1]),
        ydata=float(view["signal"][1]),
    )

    viewer._on_motion(event)

    assert viewer.cursor_hkle_label.isHidden()
    assert not viewer.cursor_q_label.isHidden()
    assert viewer.cursor_q_label.text() == "|Q| = 0.6 Å⁻¹"


def test_qt_powder_point_cursor_readout_calculates_q_from_two_theta_and_wavelength():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    wavelength = 2.41
    two_theta = np.array([10.0, 30.0, 50.0])
    data = PointListData(
        columns={
            "2theta": two_theta,
            "Signal": np.array([1.0, 2.0, 3.0]),
            "Error": np.array([0.1, 0.1, 0.1]),
        },
        units={"2theta": "°"},
        coordinate_names=["2theta"],
        channels=[{"label": "Signal", "value": "Signal", "error": "Error"}],
        metadata={
            "nfit_data_type": "powder_elastic",
            "wavelength": {"value": wavelength, "two_theta": "2theta"},
        },
    )
    viewer = QtMDHistoSliceViewer(data)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][1]),
        ydata=float(view["signal"][1]),
    )

    viewer._on_motion(event)

    expected_q = 4.0 * np.pi * np.sin(np.deg2rad(two_theta[1]) / 2.0) / wavelength
    assert viewer.cursor_q_label.text() == f"|Q| = {expected_q:.5g} Å⁻¹"


def test_qt_cursor_readout_applies_2pi_for_orientation_matrix_convention():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.metadata["oriented_lattice"] = {"orientation_matrix": (0.1 * np.eye(3)).tolist()}
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)

    assert viewer._q_modulus_inv_angstrom({"H": 1.0, "K": 0.0, "L": 0.0, "E": 0.0}) == pytest.approx(
        0.2 * np.pi
    )

    data.metadata["oriented_lattice"]["orientation_matrix_includes_2pi"] = True
    assert viewer._q_modulus_inv_angstrom({"H": 1.0, "K": 0.0, "L": 0.0, "E": 0.0}) == pytest.approx(
        0.1
    )


def test_qt_slice_viewer_replace_datasets_preserves_plot_settings():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    first = _tiny_mdhisto_data()
    second = _tiny_mdhisto_data()
    viewer = QtMDHistoSliceViewer([first, second], dataset_names=["first", "second"], x_dim=3, y_dim=2)
    viewer.dataset_combo.setCurrentIndex(1)
    viewer._set_cmap("magma")
    viewer._set_color_scale("log")
    viewer._set_manual_limit("vmin", 0.25)
    viewer._set_font_size(17.0)
    viewer._set_xcut_percent(31)

    replacement_second = _tiny_mdhisto_data()
    replacement_third = _tiny_mdhisto_data()
    viewer.replace_datasets(
        [first, replacement_second, replacement_third],
        dataset_names=["first", "second", "third"],
        selected_dataset_name="second",
    )

    assert viewer.dataset_combo.currentText() == "second"
    assert viewer.data is replacement_second
    assert viewer.model.cmap == "magma"
    assert viewer.model.color_scale == "log"
    assert viewer.model.manual_vmin == pytest.approx(0.25)
    assert viewer.font_size == pytest.approx(17.0)
    assert viewer.xcut_percent == 31


def test_qt_1d_cursor_readout_tracks_nearest_point_and_hkle():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    assert viewer.cursor_q_label.text() == "|Q| = ? Å⁻¹"
    assert viewer.cursor_intensity_label.text() == "Signal = 3.25 ± 0.12"


def test_qt_box_tool_visibility_checkbox_controls_rectangle_selector():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

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
