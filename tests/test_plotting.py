import matplotlib

matplotlib.use("Agg")

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.colors as mcolors
import numpy as np
import pytest

from nfit.colormaps import (
    CMCRAMERI_SEQUENTIAL_COLORMAPS,
    COLORCET_CATEGORICAL_COLORMAPS,
    COLORCET_CONTINUOUS_COLORMAPS,
    IMAGE_COLORMAP_GROUPS,
    MATPLOTLIB_DIVERGING_COLORMAPS,
    MATPLOTLIB_IMAGE_COLORMAPS,
    MATPLOTLIB_QUALITATIVE_COLORMAPS,
    MATPLOTLIB_SEQUENTIAL_COLORMAPS,
    MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS,
    WATERFALL_COLORMAP_GROUPS,
    WATERFALL_DISCRETE_COLORMAPS,
)
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.plotting import (
    MDHistoSliceViewer,
    _DropdownSelect,
    exponential_colormap,
    integrated_box_sum,
    plot_mdhisto_slice,
)
from tests.plotting_test_data import (
    tiny_1d_mdhisto_data as _tiny_1d_mdhisto_data,
)
from tests.plotting_test_data import (
    tiny_2d_mdhisto_data_with_singletons as _tiny_2d_mdhisto_data_with_singletons,
)
from tests.plotting_test_data import (
    tiny_mdhisto_data as _tiny_mdhisto_data,
)
from tests.plotting_test_data import (
    with_fit_channels as _with_fit_channels,
)


def test_qt_slice_viewer_apply_masks_toggle_shows_masked_bins():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    editable = data.mutable_copy()
    editable.mask[1, 1, 2, 3] = True
    data = editable.immutable_copy()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.model.selections[0] = (0.75, 0.75)
    viewer.model.selections[1] = (1.5, 1.5)
    viewer.update_plot()

    assert viewer.apply_masks_check.isChecked()
    assert np.isnan(viewer.slice_arrays()["signal"][2, 3])

    viewer.apply_masks_check.setChecked(False)

    assert not np.isnan(viewer.slice_arrays()["signal"][2, 3])
    assert viewer.slice_arrays()["combined_mask"][2, 3]


def test_static_and_qt_slice_views_draw_scriptable_brillouin_zone_boundaries():
    pytest.importorskip("PySide6")
    from matplotlib.collections import LineCollection

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data().with_updates(
        metadata={
            "lattice_parameters": {
                "a": 4.0, "b": 4.0, "c": 4.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            },
            "spacegroup": "P 1",
        }
    )
    figure = plot_mdhisto_slice(
        data,
        x_dim=3,
        y_dim=2,
        show_brillouin_zone_boundaries=True,
    )
    assert any(
        isinstance(artist, LineCollection)
        and artist.get_gid() == "nfit-brillouin-zone-boundaries"
        for artist in figure.axes[0].collections
    )

    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.show_brillouin_zone_check.setChecked(True)
    settings = viewer.current_plot_settings()
    assert settings["show_brillouin_zone_boundaries"] is True
    assert settings["brillouin_zone_spacegroup"] == "P 1"
    assert "show_brillouin_zone_boundaries=True" in viewer.figure_script()

    assert viewer.brillouin_zone_linewidth == pytest.approx(1.5)
    assert viewer.brillouin_zone_alpha == pytest.approx(1.0)
    viewer.show_major_gridlines_check.setChecked(True)
    assert viewer.show_major_gridlines
    assert not viewer.show_brillouin_zone_boundaries
    assert not viewer.show_brillouin_zone_check.isChecked()
    assert any(line.get_visible() for line in viewer.ax_image.get_xgridlines())
    settings = viewer.current_plot_settings()
    assert settings["show_major_gridlines"] is True
    assert "show_major_gridlines=True" in viewer.figure_script()

    viewer.show_brillouin_zone_check.setChecked(True)
    assert viewer.show_brillouin_zone_boundaries
    assert not viewer.show_major_gridlines
    assert not viewer.show_major_gridlines_check.isChecked()


def test_static_major_gridlines_use_shared_style_and_exclude_zone_boundaries():
    data = _tiny_mdhisto_data()
    figure = plot_mdhisto_slice(
        data,
        x_dim=3,
        y_dim=2,
        show_major_gridlines=True,
    )
    axis = figure.axes[0]
    gridlines = [
        line
        for line in (*axis.get_xgridlines(), *axis.get_ygridlines())
        if line.get_visible()
    ]

    assert gridlines
    assert all(line.get_linewidth() == pytest.approx(1.5) for line in gridlines)
    assert all(line.get_alpha() == pytest.approx(1.0) for line in gridlines)
    with pytest.raises(ValueError, match="mutually exclusive"):
        plot_mdhisto_slice(
            data,
            x_dim=3,
            y_dim=2,
            show_brillouin_zone_boundaries=True,
            show_major_gridlines=True,
        )


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

    viewer.symmetric_about_zero = True
    assert viewer._color_limits(values) == (
        -70.90000000000002,
        70.90000000000002,
    )

    viewer.autoscale = False
    viewer.manual_vmin = -2.0
    viewer.manual_vmax = 7.0
    assert viewer._color_limits(values) == (-7.0, 7.0)


def test_exponential_colormap_warps_colors_but_not_normalization():
    from matplotlib import colormaps

    data = _tiny_mdhisto_data()
    base = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    shifted = MDHistoSliceViewer(data, x_dim=3, y_dim=2, color_alpha=4.0)
    values = base._display_values(base.slice_arrays())

    base_norm = base._color_norm(values)
    shifted_norm = shifted._color_norm(values)
    assert (shifted_norm.vmin, shifted_norm.vmax) == (
        base_norm.vmin,
        base_norm.vmax,
    )
    assert shifted._display_cmap()(0.5) != pytest.approx(
        colormaps[base._effective_cmap()](0.5)
    )
    expected_coordinate = np.expm1(4.0 * 0.5) / np.expm1(4.0)
    assert shifted._display_cmap()(0.5) == pytest.approx(
        colormaps[base._effective_cmap()](expected_coordinate),
        abs=0.01,
    )
    assert exponential_colormap("viridis", 0.0).name == "viridis"


def test_integrated_box_sum_propagates_independent_errors():
    total, uncertainty, count = integrated_box_sum(
        np.array([[1.0, 2.0], [3.0, np.nan]]),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
    )

    assert total == pytest.approx(6.0)
    assert uncertainty == pytest.approx(np.sqrt(0.1**2 + 0.2**2 + 0.3**2))
    assert count == 3


def test_static_box_histogram_labels_total_with_propagated_error():
    import matplotlib.pyplot as plt

    data = _tiny_mdhisto_data()
    model = MDHistoSliceViewer(data, x_dim=3, y_dim=2)
    view = model.slice_arrays()
    extents = (
        float(view["x_centers"][0]),
        float(view["x_centers"][-1]),
        float(view["y_centers"][0]),
        float(view["y_centers"][-1]),
    )
    expected, expected_error, _count = integrated_box_sum(
        model._display_values(view),
        view["errors"],
    )

    fig = plot_mdhisto_slice(
        data,
        x_dim=3,
        y_dim=2,
        show_histogram_axes=True,
        roi_extents=extents,
    )
    title = fig.axes[0].get_title(loc="left")

    assert f"{expected:.5g}" in title
    assert f"{expected_error:.2g}" in title
    plt.close(fig)


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


def test_qt_slice_viewer_can_title_plot_with_hidden_axis_binning():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data = replace(
        data,
        axes=(
            MDHistoAxis("[H,H,0]", data.axes[0].values, "rlu", "momentum"),
            MDHistoAxis("[K,K,-2K]", data.axes[1].values, "rlu", "momentum"),
            MDHistoAxis("[0,0,L]", data.axes[2].values, "rlu", "momentum"),
            MDHistoAxis("DeltaE", data.axes[3].values, "meV", "energy_transfer"),
        ),
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=0, y_dim=2)
    viewer.model.selections[1] = (0.9, 1.1)
    viewer.model.selections[3] = (1.0, 2.0)
    viewer.model.integrate_checks[1] = True
    viewer.model.integrate_checks[3] = True
    viewer.show_binning_title_check.setChecked(True)

    expected = "[K,K,-2K]=[0.9,1.1] r.l.u., ΔE=[1,2] meV"
    assert viewer.figure._suptitle.get_text() == expected
    assert repr(expected) in viewer.figure_script()
    assert viewer.current_plot_settings()["show_binning_title"] is True
    viewer.window.close()


def test_qt_slice_viewer_title_shows_single_hidden_bin_edges():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data = replace(
        data,
        axes=(
            data.axes[0],
            MDHistoAxis(
                "[L,-L,0]", np.array([-0.15, -0.05, 0.05, 0.15]),
                "rlu", "momentum",
            ),
            *data.axes[2:],
        ),
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    controls = viewer.hidden_controls[1]
    controls.integrate.setChecked(False)
    assert controls.value.value() == pytest.approx(0.0)
    assert controls.width.value() == pytest.approx(0.1)
    viewer.show_binning_title_check.setChecked(True)

    expected = "[L,-L,0]=[-0.05,0.05] r.l.u."
    assert expected in viewer.figure._suptitle.get_text()
    assert repr(viewer._binning_title_text()) in viewer.figure_script()
    viewer.window.close()


def test_qt_slice_viewer_open_new_viewer_duplicates_current_state():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    first = _tiny_mdhisto_data()
    second = _tiny_mdhisto_data()
    viewer = QtMDHistoSliceViewer(
        [first, second],
        dataset_names=["first", "second"],
        x_dim=3,
        y_dim=2,
    )
    viewer.dataset_combo.setCurrentIndex(1)
    viewer.marker_combo.setCurrentText("square")
    viewer.cmap_combo.setCurrentText("plasma")
    viewer._toggle_cmap_reverse()
    viewer.ax_image.set_xlim(-0.25, 0.75)
    viewer.ax_image.set_ylim(-0.5, 0.5)

    opened = []

    def create_viewer(_selected_name):
        duplicate = QtMDHistoSliceViewer(
            [first, second],
            dataset_names=["first", "second"],
        )
        opened.append(duplicate)
        return duplicate

    viewer.set_open_new_viewer_callback(create_viewer)
    button = viewer.window.findChild(
        QtWidgets.QPushButton,
        "data_viewer_open_new_button",
    )
    mode_layout = viewer.view_mode_combo.parentWidget().layout()

    assert button is viewer.open_new_viewer_button
    assert mode_layout.indexOf(button) == mode_layout.indexOf(viewer.view_mode_combo) + 1
    assert button.toolTip().strip()

    button.click()

    duplicate = opened[0]
    assert duplicate.dataset_combo.currentText() == "second"
    assert duplicate.marker == "s"
    assert duplicate.model._effective_cmap() == "plasma_r"
    assert duplicate.ax_image.get_xlim() == pytest.approx((-0.25, 0.75))
    assert duplicate.ax_image.get_ylim() == pytest.approx((-0.5, 0.5))
    duplicate.window.close()
    viewer.window.close()


def test_qt_slice_viewer_store_plot_button_uses_project_callback_and_reports_name():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    plot_buttons = [
        button
        for button in viewer.window.findChildren(QtWidgets.QPushButton)
        if button.text() == "Store plot"
    ]

    assert plot_buttons == [viewer.save_plot_button]
    assert not viewer.save_plot_button.isEnabled()
    assert not any(
        button.text() == "Plot"
        for button in viewer.window.findChildren(QtWidgets.QToolButton)
    )

    calls = []
    class StoredPlot:
        name = "scan plot"

    def store():
        calls.append("saved")
        return StoredPlot()

    viewer.set_save_plot_callback(store)
    viewer.save_plot_button.click()

    assert calls == ["saved"]
    assert viewer.store_plot_status_label.text() == "Stored as scan plot"
    assert (
        viewer.save_plot_button.parentWidget()
        is viewer.open_new_viewer_button.parentWidget()
    )


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
    editable = data.mutable_copy()
    editable.mask[1, 1, 2, 3] = True
    data = editable.immutable_copy()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2, channel="mask")

    assert "grey" in viewer.model.COLORMAPS
    assert viewer.model.channel == "combined_mask"
    assert isinstance(viewer.cmap_reverse_button, QtWidgets.QPushButton)
    assert viewer.model._color_limits(viewer.model._display_values(viewer.slice_arrays())) == (0.0, 1.0)
    assert viewer.image.cmap.name == "gray"

    viewer.cmap_reverse_button.click()

    assert viewer.model.cmap_reversed is True
    assert viewer.image.cmap.name == "gray_r"


def test_qt_colormap_menus_group_matplotlib_and_named_colorcet_maps():
    pytest.importorskip("PySide6")
    from matplotlib import colormaps

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert tuple(viewer.model.COLORMAPS) == tuple(
        name for group in IMAGE_COLORMAP_GROUPS for name in group
    )
    folder_action = viewer.cmap_combo.actions()[0]
    assert folder_action.text() == "Open custom colormap folder…"
    assert "restart" in folder_action.toolTip()
    assert set(MATPLOTLIB_SEQUENTIAL_COLORMAPS) <= set(viewer.model.COLORMAPS)
    assert set(MATPLOTLIB_DIVERGING_COLORMAPS) <= set(viewer.model.COLORMAPS)
    assert set(MATPLOTLIB_SPECIALIZED_CONTINUOUS_COLORMAPS) <= set(
        viewer.model.COLORMAPS
    )
    assert "cet_fire" in COLORCET_CONTINUOUS_COLORMAPS
    assert "cet_glasbey" in COLORCET_CATEGORICAL_COLORMAPS
    assert colormaps.get_cmap("cet_fire").name == "cet_fire"
    assert "bluewhitered" not in viewer.model.COLORMAPS
    assert "CMRmap" in viewer.model.COLORMAPS
    assert "gnuplot2" in viewer.model.COLORMAPS
    np.testing.assert_allclose(
        colormaps.get_cmap("bluewhitered")(np.linspace(0.0, 1.0, 256)),
        colormaps.get_cmap("bwr")(np.linspace(0.0, 1.0, 256)),
    )
    np.testing.assert_allclose(
        colormaps.get_cmap("young_rdbu")(0.0)[:3],
        np.asarray((5.0, 48.0, 97.0)) / 256.0,
    )
    np.testing.assert_allclose(
        colormaps.get_cmap("young_quadratic")(1.0)[:3],
        np.asarray((255.0, 255.0, 0.0)) / 256.0,
    )

    image_separator = len(MATPLOTLIB_IMAGE_COLORMAPS)
    assert viewer.cmap_combo.itemText(image_separator) == ""
    assert not viewer.cmap_combo.model().item(image_separator).isEnabled()
    assert viewer.cmap_combo.itemText(image_separator + 1) == "Greys"
    assert not viewer.cmap_combo.itemIcon(
        viewer.cmap_combo.findText("RdBu")
    ).isNull()
    fire_index = viewer.cmap_combo.findText("cet_fire")
    assert fire_index > image_separator
    assert not viewer.cmap_combo.itemIcon(fire_index).isNull()

    item_index = 0
    for group_index, group in enumerate(group for group in WATERFALL_COLORMAP_GROUPS if group):
        if group_index:
            assert viewer.waterfall_cmap_combo.itemText(item_index) == ""
            assert not viewer.waterfall_cmap_combo.model().item(item_index).isEnabled()
            item_index += 1
        assert viewer.waterfall_cmap_combo.itemText(item_index) == group[0]
        item_index += len(group)
    categorical_index = viewer.waterfall_cmap_combo.findText(
        COLORCET_CATEGORICAL_COLORMAPS[0]
    )
    assert not viewer.waterfall_cmap_combo.itemIcon(
        categorical_index
    ).isNull()
    assert set(MATPLOTLIB_QUALITATIVE_COLORMAPS) <= {
        viewer.waterfall_cmap_combo.itemText(index)
        for index in range(viewer.waterfall_cmap_combo.count())
    }

    viewer.cmap_combo.setCurrentText("cet_fire")
    assert viewer.image.cmap.name == "cet_fire"


def test_data_viewer_colormap_popups_start_at_the_first_entry():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    try:
        for combo in (viewer.cmap_combo, viewer.waterfall_cmap_combo):
            combo.setCurrentText("cet_fire")
            combo.showPopup()
            scroll_bar = combo.view().verticalScrollBar()

            assert scroll_bar.maximum() > scroll_bar.minimum()
            assert scroll_bar.value() == scroll_bar.minimum()

            combo.hidePopup()
    finally:
        viewer.window.close()


@pytest.mark.parametrize("selected", ["cmc.batlow", "cmo.thermal", "mycarta.Cube1", "carto.SunsetDark"])
def test_cmcrameri_maps_preserve_tables_and_tiled_plot_settings(selected):
    from cmcrameri import cm
    from matplotlib import colormaps

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    # Upstream classification catches omissions and accidental inclusion of
    # terrain, cyclic or categorical maps when the dependency is updated.
    assert set(CMCRAMERI_SEQUENTIAL_COLORMAPS) == {
        f"cmc.{name}" for name in cm._cmap_names_sequential
    }
    samples = np.linspace(0, 1, 256)
    for name in CMCRAMERI_SEQUENTIAL_COLORMAPS:
        assert name not in WATERFALL_DISCRETE_COLORMAPS
        np.testing.assert_array_equal(
            colormaps[name](samples), cm.cmaps[name[4:]](samples)
        )
        np.testing.assert_array_equal(
            colormaps[name + "_r"](samples), colormaps[name](samples)[::-1]
        )
    data = _tiny_mdhisto_data()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.view_mode_combo.setCurrentText("Tiled slices")
    viewer.cmap_combo.setCurrentText(selected)
    viewer.cmap_reverse_button.click()
    assert not viewer.cmap_combo.itemIcon(viewer.cmap_combo.findText(selected)).isNull()
    settings = viewer.current_plot_settings()
    assert selected + "_r" in viewer.figure_script()
    restored = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    restored.apply_plot_settings(settings)
    assert restored.model._effective_cmap() == selected + "_r"


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
    assert viewer.dataset_combo.minimumWidth() >= 260
    assert viewer.coverage_threshold_spin.value() == pytest.approx(0.0)
    assert viewer.waterfall_coverage_threshold_spin.value() == pytest.approx(0.0)
    assert viewer.x_min_spin.maximumWidth() <= 118
    assert viewer.vmax_spin.maximumWidth() <= 118
    assert viewer.cmap_combo.maximumWidth() <= 150


def test_qt_channel_dropdown_switches_displayed_channel_and_export_script():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.channel_combo.setCurrentText("errors")
    viewer.coverage_threshold_spin.setValue(0.83)

    assert viewer.model.channel == "errors"
    assert (
        viewer.colorbar.ax.yaxis.label.get_text()
        == r"Uncertainty in $I(\mathbf{Q},E)$ (a.u.)"
    )
    np.testing.assert_allclose(viewer.image.get_array(), viewer.slice_arrays()["errors"])
    assert "channel='errors'" in viewer.figure_script()
    assert "coverage_threshold=0.83" in viewer.figure_script()
    assert viewer.current_plot_settings()["coverage_threshold"] == pytest.approx(0.83)


def test_qt_slice_viewer_selects_derived_coverage_channels_without_storage():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.channel_combo.setCurrentText("coverage_fraction")
    assert viewer.model.channel == "coverage_fraction"
    assert viewer.colorbar.ax.yaxis.label.get_text() == "Coverage (fraction)"

    viewer.channel_combo.setCurrentText("coverage_mask")
    assert viewer.model.channel == "coverage_mask"
    assert viewer.colorbar.ax.yaxis.label.get_text() == "Coverage mask"


def test_qt_appearance_updates_reuse_slice_but_data_updates_recompute(monkeypatch):
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    calls = 0
    original_slice_arrays = viewer.model.slice_arrays

    def tracked_slice_arrays():
        nonlocal calls
        calls += 1
        return original_slice_arrays()

    monkeypatch.setattr(viewer.model, "slice_arrays", tracked_slice_arrays)

    viewer._set_cmap("magma")
    viewer._toggle_cmap_reverse()
    viewer._set_color_alpha(0.5)
    assert calls == 0

    viewer.model.selections[0] = (0.25, 0.25)
    viewer._set_cmap("viridis")
    assert calls == 1

    nfit_mask = np.zeros(data.shape, dtype=bool)
    nfit_mask[0, 1, 2, 3] = True
    data.metadata["nfit_mask"] = nfit_mask
    viewer.update_plot()
    assert calls == 2
    assert viewer._current_slice["nfit_mask"][2, 3]


def test_qt_reused_slice_refreshes_mutable_metadata_masks(monkeypatch):
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    nfit_mask = np.zeros(data.shape, dtype=bool)
    data.metadata["nfit_mask"] = nfit_mask
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.channel_combo.setCurrentText("nfit_mask")
    calls = 0
    original_slice_arrays = viewer.model.slice_arrays

    def tracked_slice_arrays():
        nonlocal calls
        calls += 1
        return original_slice_arrays()

    monkeypatch.setattr(viewer.model, "slice_arrays", tracked_slice_arrays)
    nfit_mask[1, 1, 2, 3] = True

    viewer._set_cmap("magma")

    assert calls == 0
    assert viewer._current_slice["nfit_mask"][2, 3]


def test_qt_histogram_tool_recomputes_coverage_over_selected_box():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    coverage = np.ones(data.shape, dtype=float)
    coverage[:, :, :2, :] = 0.5
    data = data.with_updates(
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(
                coverage,
                label="Coverage",
                unit="fraction",
            )
        }
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.coverage_threshold_spin.setValue(0.8)
    view = viewer._current_slice
    viewer._update_histogram_cuts_from_extents(
        (
            float(view["x_edges"][0]),
            float(view["x_edges"][-1]),
            float(view["y_edges"][0]),
            float(view["y_edges"][-1]),
        )
    )

    assert viewer.ax_xcut.lines
    assert np.all(np.isnan(viewer.ax_xcut.lines[0].get_ydata()))


def test_qt_histogram_box_shows_sum_and_propagated_uncertainty():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.show_binning_title_check.setChecked(True)
    viewer.show_box_check.setChecked(True)
    view = viewer._current_slice
    extents = (
        float(view["x_edges"][0]),
        float(view["x_edges"][-1]),
        float(view["y_edges"][0]),
        float(view["y_edges"][-1]),
    )
    expected, expected_error, _count = integrated_box_sum(
        viewer.model._display_values(view),
        view["errors"],
    )

    viewer._set_roi_extents(extents, update_cuts=True, draw=False)

    assert viewer.roi_sum_text is not None
    assert f"{expected:.5g}" in viewer.roi_sum_text.get_text()
    assert f"{expected_error:.2g}" in viewer.roi_sum_text.get_text()
    assert viewer.ax_image.get_title(loc="left") == viewer.roi_sum_text.get_text()
    assert viewer.figure._suptitle.get_text()
    viewer.show_box_check.setChecked(False)
    assert viewer.roi_sum_text is None


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


def test_qt_macs_plot_preserves_empty_bins():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_2d_mdhisto_data_with_singletons()
    editable = data.mutable_copy()
    editable.signal[0, 0, 1, 1] = np.nan
    data = editable.immutable_copy().with_updates(
        metadata={"importer": "macs_nexus", "detector_stream": "SPEC"}
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)

    assert np.isnan(viewer._current_slice["signal"][1, 1])
    assert np.ma.getmaskarray(viewer.image.get_array())[1, 1]
    assert not viewer.smoothing_fill_nans_check.isChecked()
    assert viewer.smoothing_fill_nans_check.toolTip()
    viewer.smoothing_x_spin.setValue(1.0)
    viewer.smoothing_y_spin.setValue(1.0)
    assert np.isnan(viewer._current_slice["signal"][1, 1])
    viewer.smoothing_fill_nans_check.setChecked(True)
    assert np.isfinite(viewer._current_slice["signal"][1, 1])
    viewer.smoothing_fill_nans_check.setChecked(False)
    assert np.isnan(viewer._current_slice["signal"][1, 1])
    assert viewer.current_plot_settings()["smoothing_fill_nans"] is False
    assert "smoothing_fill_nans=False" in viewer.figure_script()
    assert "empty_bin_fill_neighbors" not in viewer.current_plot_settings()
    assert "empty_bin_fill_neighbors" not in viewer.figure_script()


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
    data_b = data_b.with_updates(signal=data_b.signal + 1000.0)

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
    data_b = data_b.with_updates(signal=data_b.signal + 1000.0)

    viewer = QtMDHistoSliceViewer([data_a, data_b], dataset_names=["first", "second"], x_dim=3, y_dim=2)
    default_cmap = viewer._initial_cmap

    viewer.channel_combo.setCurrentText("errors")
    viewer.cmap_combo.setCurrentText("magma")
    viewer.x_combo.setCurrentIndex(1)
    viewer.ax_image.set_xlim(-0.25, 0.25)
    viewer.ax_image.set_ylim(-0.5, 0.5)
    viewer.vmin_spin.setValue(viewer.vmin_spin.value() + 1.0)
    viewer.font_size_spin.setValue(14.0)

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.model.channel == "signal"
    assert viewer.model.cmap == default_cmap
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


def test_qt_hold_view_settings_carries_compatible_state_to_another_dataset():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    first = _tiny_mdhisto_data()
    second = _tiny_mdhisto_data().with_updates(signal=first.signal + 1000.0)
    viewer = QtMDHistoSliceViewer(
        [first, second],
        dataset_names=["first", "second"],
        x_dim=3,
        y_dim=2,
    )

    assert not viewer.hold_view_settings_check.isChecked()
    viewer.channel_combo.setCurrentText("errors")
    viewer.cmap_combo.setCurrentText("magma")
    viewer.x_combo.setCurrentIndex(1)
    viewer.ax_image.set_xlim(-0.25, 0.25)
    viewer.ax_image.set_ylim(-0.5, 0.5)
    viewer.font_size_spin.setValue(14.0)
    viewer.symmetric_about_zero_check.setChecked(True)
    viewer.hold_view_settings_check.setChecked(True)

    viewer.dataset_combo.setCurrentIndex(1)

    assert viewer.model.channel == "errors"
    assert viewer.model.cmap == "magma"
    assert viewer.model.symmetric_about_zero
    assert viewer.font_size == pytest.approx(14.0)
    assert viewer.x_combo.currentText() == "[H,-H,0]"
    assert viewer.y_combo.currentText() == "[0,0,L]"
    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-0.25, 0.25))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-0.5, 0.5))


def test_qt_symmetric_color_limits_follow_autoscale_and_manual_edits():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    viewer.symmetric_about_zero_check.setChecked(True)

    assert viewer.model.autoscale
    assert viewer.vmin_spin.value() == pytest.approx(-viewer.vmax_spin.value())
    assert viewer.image.norm.vmin == pytest.approx(-viewer.image.norm.vmax)

    viewer.vmax_spin.setValue(25.0)

    assert not viewer.model.autoscale
    assert viewer.vmin_spin.value() == pytest.approx(-25.0)
    assert viewer.model.manual_vmin == pytest.approx(-25.0)
    assert viewer.model.manual_vmax == pytest.approx(25.0)
    assert viewer.current_plot_settings()["symmetric_about_zero"] is True
    assert "symmetric_about_zero=True" in viewer.figure_script()


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

def test_qt_show_fit_draws_side_by_side_panels_with_shared_view():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _with_fit_channels(_tiny_mdhisto_data()).mutable_copy()
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

    data = _with_fit_channels(_tiny_mdhisto_data()).mutable_copy()
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

    data = _with_fit_channels(_tiny_1d_mdhisto_data()).mutable_copy()
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
    assert viewer.coverage_threshold_spin.toolTip()
    assert viewer.waterfall_offset_auto_check.isChecked()
    assert viewer.waterfall_offset_auto is True
    for control in (
        viewer.waterfall_source_label,
        viewer.waterfall_step_spin,
        viewer.waterfall_step_slider,
        viewer.waterfall_step_auto_check,
        viewer.waterfall_coverage_threshold_spin,
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
    viewer.waterfall_coverage_threshold_spin.setValue(0.82)

    settings = viewer.current_plot_settings()
    script = viewer.figure_script()
    assert settings["view_mode"] == "waterfall"
    assert settings["waterfall_cmap"] == "plasma"
    assert settings["waterfall_color_min"] == pytest.approx(0.1)
    assert settings["waterfall_color_max"] == pytest.approx(0.85)
    assert settings["waterfall_coverage_threshold"] == pytest.approx(0.82)
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
    assert "coverage_threshold=0.82" in script
    compile(script, "waterfall_figure.py", "exec")

    restored = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    restored.apply_plot_settings(settings)
    assert restored.view_mode_combo.currentText() == "Waterfall"
    assert restored.waterfall_cmap == "plasma"
    assert restored.waterfall_color_min == pytest.approx(0.1)
    assert restored.waterfall_color_max == pytest.approx(0.85)
    assert restored.waterfall_coverage_threshold == pytest.approx(0.82)
    assert restored.waterfall_zero_style == ":"
    assert restored.waterfall_trace_label_suffix == " at 6 K"
    assert restored.waterfall_trace_label_font_size == pytest.approx(14.0)
    assert restored.waterfall_trace_label_color == "#d62728"
    assert restored.marker_face_color == "outline"

    viewer.view_mode_combo.setCurrentIndex(0)
    assert viewer.marker_face_color == "#ff7f0e"
    viewer.view_mode_combo.setCurrentIndex(1)
    assert viewer.marker_face_color == "outline"


def test_qt_tiled_slices_exposes_third_axis_range_step_slider_and_script():
    pytest.importorskip("PySide6")

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    viewer.view_mode_combo.setCurrentText("Tiled slices")

    assert viewer._plot_layout_mode == ("tiled", 2, False)
    assert not viewer.tiled_group.isHidden()
    assert viewer.tile_dim_combo.currentText() == "DeltaE"
    assert viewer.tile_step_auto_check.isChecked()
    for control in (
        viewer.tile_dim_combo,
        viewer.tile_range_low_spin,
        viewer.tile_range_high_spin,
        viewer.tile_step_spin,
        viewer.tile_step_slider,
        viewer.tile_step_auto_check,
        viewer.show_tile_labels_check,
        viewer.tile_local_color_scales_check,
    ):
        assert control.toolTip().strip()
    assert len(viewer._tile_axes) == 2
    assert len(viewer.figure.axes) == 3
    assert len(viewer.hidden_controls) == 1
    assert all(
        axis.collections[0].norm is viewer._tile_axes[0].collections[0].norm
        for axis in viewer._tile_axes
    )
    assert all(
        axis.texts[0].get_position() == (0.97, 0.03)
        for axis in viewer._tile_axes
    )
    assert all(
        axis.texts[0].get_bbox_patch().get_alpha() == pytest.approx(0.65)
        for axis in viewer._tile_axes
    )
    assert all(
        type(axis.texts[0].get_bbox_patch().get_boxstyle()).__name__ == "Square"
        for axis in viewer._tile_axes
    )
    viewer.show_binning_title_check.setChecked(True)
    title = viewer.figure._suptitle.get_text()
    assert "ΔE" not in title
    assert "[H,-H,0]" in title
    assert "ΔE" not in viewer._binning_title_text()

    viewer.show_tile_labels_check.setChecked(False)
    assert all(not axis.texts for axis in viewer._tile_axes)

    viewer.show_tile_labels_check.setChecked(True)
    assert viewer.tile_label_decimals_spin.toolTip()
    for widget in (viewer.tile_label_prefix_edit, viewer.tile_label_unit_edit, viewer.tile_label_si_prefix_combo):
        assert widget.toolTip()
    viewer.tile_label_decimals_spin.setValue(0)
    viewer.tile_label_prefix_edit.setText("T = ")
    viewer.tile_label_unit_edit.setText("K")
    viewer.tile_label_si_prefix_combo.setCurrentText("m")
    assert all(axis.texts[0].get_text().startswith("T = ") for axis in viewer._tile_axes)
    assert all(axis.texts[0].get_text().endswith(" mK") for axis in viewer._tile_axes)
    viewer.tile_local_color_scales_check.setChecked(True)
    assert viewer._plot_layout_mode == ("tiled", 2, True)
    assert len(viewer._tile_colorbars) == len(viewer._tile_axes) == 2
    assert len(viewer.figure.axes) == 4
    assert (
        viewer._tile_axes[0].collections[0].norm
        is not viewer._tile_axes[1].collections[0].norm
    )
    assert [
        colorbar.ax.yaxis.label.get_text()
        for colorbar in viewer._tile_colorbars
    ] == ["", r"$I(\mathbf{Q},E)$ (a.u.)"]
    viewer.autoscale_check.setChecked(False)
    assert not viewer.tile_local_color_scales_check.isChecked()
    assert not viewer.tile_local_color_scales_check.isEnabled()
    assert viewer._plot_layout_mode == ("tiled", 2, False)
    viewer.autoscale_check.setChecked(True)
    viewer.tile_local_color_scales_check.setChecked(True)

    viewer.tile_step_slider.setValue(1000)
    assert not viewer.tile_step_auto_check.isChecked()
    assert len(viewer._tile_axes) == 1
    settings = viewer.current_plot_settings()
    script = viewer.figure_script()
    assert settings["view_mode"] == "tiled_slices"
    assert settings["tile_dim"] == "DeltaE"
    assert settings["show_tile_labels"] is True
    assert settings["tile_label_decimals"] == 0
    assert settings["tile_label_prefix"] == "T = "
    assert settings["tile_label_unit"] == "K"
    assert settings["tile_label_si_prefix"] == "m"
    assert "tile_label_decimals=0" in script
    assert "tile_label_prefix='T = '" in script
    assert settings["tile_local_color_scales"] is True
    assert "plot_mdhisto_tiled_slices" in script
    assert "show_tile_labels=True" in script
    assert "local_color_scales=True" in script
    compile(script, "tiled_slice_figure.py", "exec")

    restored = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2)
    restored.apply_plot_settings(settings)
    assert restored.view_mode_combo.currentText() == "Tiled slices"
    assert restored.tile_step == pytest.approx(viewer.tile_step)
    assert restored.show_tile_labels_check.isChecked()
    assert restored.tile_label_decimals_spin.value() == 0
    assert restored.tile_label_prefix_edit.text() == "T = "
    assert restored.tile_label_unit_edit.text() == "K"
    assert restored.tile_label_si_prefix_combo.currentText() == "m"
    assert restored.tile_local_color_scales_check.isChecked()


def test_qt_tiled_cursor_readout_uses_each_panels_values_and_coordinate():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.view_mode_combo.setCurrentText("Tiled slices")
    assert len(viewer._tile_axes) == len(viewer._current_tiled_slices) == 2

    readouts = []
    for axis, panel in zip(
        viewer._tile_axes,
        viewer._current_tiled_slices,
        strict=True,
    ):
        event = SimpleNamespace(
            inaxes=axis,
            xdata=float(panel.view["x_centers"][2]),
            ydata=float(panel.view["y_centers"][3]),
        )
        viewer._on_motion(event)
        readouts.append(
            (
                viewer.cursor_hkle_label.text(),
                viewer.cursor_intensity_label.text(),
            )
        )

    assert readouts == [
        (
            "(H, K, L, E) = (1.5, -1.5, 0.75, 0.25)",
            "Signal = 37 ± 1; coverage = 100.0%",
        ),
        (
            "(H, K, L, E) = (1.5, -1.5, 0.75, 0.75)",
            "Signal = 97 ± 1; coverage = 100.0%",
        ),
    ]
    assert all(axis.format_coord(1.0, 2.0) == "" for axis in viewer._tile_axes)


def test_qt_tiled_mask_channel_invalidates_after_inplace_metadata_edit(monkeypatch):
    pytest.importorskip("PySide6")
    import nfit.qt_slice_viewer as slice_viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_mdhisto_data()
    data.metadata["nfit_mask"] = np.zeros(data.shape, dtype=bool)
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=2, channel="nfit_mask")
    viewer.view_mode_combo.setCurrentText("Tiled slices")
    calls = 0
    original_prepare = slice_viewer_module.prepare_mdhisto_tiled_slices

    def tracked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        slice_viewer_module,
        "prepare_mdhisto_tiled_slices",
        tracked_prepare,
    )
    data.metadata["nfit_mask"][0, 1, 2, 3] = True

    viewer._set_cmap("magma")

    assert calls == 1
    assert any(np.any(panel.values) for panel in viewer._current_tiled_slices)


def test_qt_waterfall_half_max_stays_enabled_when_initial_range_shrinks():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = _tiny_2d_mdhisto_data_with_singletons().mutable_copy()
    data.signal[:] *= 1.0e-3
    data.errors[:] *= 1.0e-3
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


def test_qt_grouped_waterfall_rejects_cache_when_sibling_signal_is_mutable(
    monkeypatch,
):
    pytest.importorskip("PySide6")
    import nfit.qt_slice_viewer as slice_viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    first = _tiny_1d_mdhisto_data()
    second = _tiny_1d_mdhisto_data().mutable_copy()
    viewer = QtMDHistoSliceViewer([first, second])
    viewer.view_mode_combo.setCurrentIndex(1)
    calls = 0
    original_prepare = slice_viewer_module.prepare_mdhisto_waterfall

    def tracked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        slice_viewer_module,
        "prepare_mdhisto_waterfall",
        tracked_prepare,
    )

    second.signal.reshape(-1)[0] += 10.0
    viewer._set_waterfall_cmap("plasma")

    assert calls == 1


def test_qt_grouped_waterfall_reuses_traces_for_appearance_change(monkeypatch):
    pytest.importorskip("PySide6")
    import nfit.qt_slice_viewer as slice_viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(
        [_tiny_1d_mdhisto_data(), _tiny_1d_mdhisto_data()]
    )
    viewer.view_mode_combo.setCurrentIndex(1)
    calls = 0
    original_prepare = slice_viewer_module.prepare_mdhisto_waterfall

    def tracked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        slice_viewer_module,
        "prepare_mdhisto_waterfall",
        tracked_prepare,
    )

    viewer._set_waterfall_cmap("plasma")

    assert calls == 0


def test_qt_grouped_waterfall_invalidates_mutated_sibling_metadata_mask(monkeypatch):
    pytest.importorskip("PySide6")
    import nfit.qt_slice_viewer as slice_viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    datasets = []
    for _index in range(2):
        data = _tiny_1d_mdhisto_data()
        data.metadata["nfit_mask"] = np.zeros(data.shape, dtype=bool)
        datasets.append(data)
    viewer = QtMDHistoSliceViewer(datasets, channel="nfit_mask")
    viewer.view_mode_combo.setCurrentIndex(1)
    calls = 0
    original_prepare = slice_viewer_module.prepare_mdhisto_waterfall

    def tracked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        slice_viewer_module,
        "prepare_mdhisto_waterfall",
        tracked_prepare,
    )

    datasets[1].metadata["nfit_mask"].reshape(-1)[0] = True
    viewer._set_waterfall_cmap("plasma")

    assert calls == 1


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


def test_qt_waterfall_saved_names_do_not_pin_interactive_group_selection():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    datasets = [_tiny_1d_mdhisto_data() for _index in range(4)]
    viewer = QtMDHistoSliceViewer(
        datasets,
        dataset_names=["g1 a", "g1 b", "g2 a", "g2 b"],
        dataset_group_keys=["Group1", "Group1", "Group2", "Group2"],
    )
    viewer.waterfall_dataset_names = ["g1 a"]
    viewer.view_mode_combo.setCurrentIndex(1)

    assert viewer.waterfall_source_dataset_names() == ["g1 a", "g1 b"]
    assert [trace.label for trace in viewer._current_waterfall_traces] == [
        "g1 a",
        "g1 b",
    ]

    viewer.dataset_combo.setCurrentIndex(2)

    assert viewer.waterfall_source_dataset_names() == ["g2 a", "g2 b"]
    assert [trace.label for trace in viewer._current_waterfall_traces] == [
        "g2 a",
        "g2 b",
    ]
    assert viewer.marker_face_color == "none"
    assert viewer.marker_face_color_combo.currentText() == "none"
    assert all(
        line.get_markerfacecolor() == "none"
        for line in viewer.ax_image.lines
        if line.get_marker() not in {"None", "none", ""}
    )


def test_qt_waterfall_reset_and_home_follow_current_group_extent():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    low = [_tiny_1d_mdhisto_data() for _index in range(2)]
    high = []
    for _index in range(2):
        editable = _tiny_1d_mdhisto_data().mutable_copy()
        editable.signal[...] *= 50.0
        editable.errors[...] *= 50.0
        high.append(editable.immutable_copy())
    viewer = QtMDHistoSliceViewer(
        [*low, *high],
        dataset_names=["low a", "low b", "high a", "high b"],
        dataset_group_keys=["Low", "Low", "High", "High"],
    )
    viewer.view_mode_combo.setCurrentIndex(1)
    low_ylim = viewer._waterfall_default_ylim

    viewer.dataset_combo.setCurrentIndex(2)
    current_xlim = viewer._waterfall_default_xlim
    current_ylim = viewer._waterfall_default_ylim

    assert current_xlim is not None
    assert current_ylim is not None
    assert low_ylim is not None
    assert current_ylim != pytest.approx(low_ylim)

    viewer.ax_image.set_ylim(-1.0, 1.0)
    viewer._reset_view_limits("y")
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), current_ylim)

    viewer.ax_image.set_xlim(100.0, 200.0)
    viewer.ax_image.set_ylim(100.0, 200.0)
    viewer.toolbar.home()
    np.testing.assert_allclose(viewer.ax_image.get_xlim(), current_xlim)
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), current_ylim)


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
    assert viewer.image.norm.vmin == pytest.approx(viewer.vmin_spin.value())
    assert viewer.image.norm.vmax == pytest.approx(viewer.vmax_spin.value())

    viewer.autoscale_check.setChecked(True)
    viewer.vmax_spin.setValue(500.0)
    assert not viewer.model.autoscale
    assert viewer.image.norm.vmax == pytest.approx(500.0)
    assert viewer.image.norm.vmin == pytest.approx(viewer.vmin_spin.value())

    viewer.limits_combo.setCurrentText("N-sigma")

    assert viewer.autoscale_check.isChecked()
    assert viewer.model.autoscale

    viewer.limit_n_spin.setValue(2.0)
    assert viewer.model.sigma_n == pytest.approx(2.0)

    viewer.limits_combo.setCurrentText("Nth percentile")
    viewer.limit_n_spin.setValue(5.0)
    assert viewer.model.percentile_n == pytest.approx(5.0)


def test_qt_home_preserves_manual_color_limits_locked_from_zoomed_autoscale():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.toolbar.update()
    viewer.toolbar.push_current()
    full_limits = viewer.image.get_clim()

    viewer.ax_image.set_xlim(-1.1, 0.1)
    viewer.ax_image.set_ylim(-0.6, 0.1)
    viewer.toolbar.push_current()
    zoom_limits = viewer.image.get_clim()
    assert zoom_limits != pytest.approx(full_limits)

    viewer.autoscale_check.setChecked(False)
    viewer.toolbar.home()

    assert not viewer.model.autoscale
    assert viewer.image.get_clim() == pytest.approx(zoom_limits)
    assert (viewer.vmin_spin.value(), viewer.vmax_spin.value()) == pytest.approx(
        zoom_limits
    )
    assert viewer.colorbar.ax.get_ylim() == pytest.approx(zoom_limits)


def test_qt_color_scale_preserves_manual_view_and_power_gamma():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    viewer.ax_image.set_xlim(-0.3, 0.7)
    viewer.ax_image.set_ylim(-0.5, 0.5)

    viewer.scale_combo.setCurrentText("power")
    viewer.gamma_spin.setValue(0.8)
    original_limits = (viewer.image.norm.vmin, viewer.image.norm.vmax)
    original_ticks = viewer.colorbar.get_ticks().copy()
    original_midpoint_color = viewer.image.cmap(0.5)
    viewer.alpha_spin.setValue(3.0)

    np.testing.assert_allclose(viewer.ax_image.get_xlim(), (-0.3, 0.7))
    np.testing.assert_allclose(viewer.ax_image.get_ylim(), (-0.5, 0.5))
    assert not viewer.gamma_spin.isHidden()
    assert viewer.model.power_gamma == pytest.approx(0.8)
    assert viewer.model.color_alpha == pytest.approx(3.0)
    assert (viewer.image.norm.vmin, viewer.image.norm.vmax) == pytest.approx(
        original_limits
    )
    np.testing.assert_allclose(viewer.colorbar.get_ticks(), original_ticks)
    assert viewer.image.cmap(0.5) != pytest.approx(original_midpoint_color)
    assert viewer.current_plot_settings()["color_alpha"] == pytest.approx(3.0)
    assert "color_alpha=3.0" in viewer.figure_script()
    restored = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    restored.apply_plot_settings(viewer.current_plot_settings())
    assert restored.model.color_alpha == pytest.approx(3.0)
    assert restored.alpha_spin.value() == pytest.approx(3.0)


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

    data = _tiny_mdhisto_data().mutable_copy()
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


def test_qt_display_axis_steps_snap_coarsen_and_round_trip():
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)

    assert viewer.x_step_spin.value() == pytest.approx(0.8)
    assert viewer.y_step_spin.value() == pytest.approx(0.5)
    assert viewer._current_slice["signal"].shape == (4, 5)

    viewer.x_step_spin.setValue(1.3)
    viewer.y_step_spin.setValue(1.0)

    assert viewer.x_step_spin.value() == pytest.approx(1.6)
    assert viewer.y_step_spin.value() == pytest.approx(1.0)
    assert viewer._current_slice["signal"].shape == (2, 3)
    settings = viewer.current_plot_settings()
    assert settings["x_step"] == pytest.approx(1.6)
    assert settings["y_step"] == pytest.approx(1.0)
    assert "x_step=1.6" in viewer.figure_script()
    assert "y_step=1.0" in viewer.figure_script()

    restored = QtMDHistoSliceViewer(_tiny_mdhisto_data(), x_dim=3, y_dim=2)
    restored.apply_plot_settings(settings)
    assert restored.x_step_spin.value() == pytest.approx(1.6)
    assert restored.y_step_spin.value() == pytest.approx(1.0)
    assert restored._current_slice["signal"].shape == (2, 3)


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

    data = _tiny_mdhisto_data().mutable_copy()
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
    assert (
        viewer.cursor_intensity_label.text()
        == "Signal = -0.0067 ± 0.0013; coverage = 100.0%"
    )
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


def test_qt_powder_cursor_readout_tracks_q_and_energy_axes():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    source = _tiny_mdhisto_data()
    axes = list(source.axes)
    axes[3] = MDHistoAxis(
        "|Q|",
        np.linspace(1.5, 2.5, 6),
        "Å⁻¹",
        "momentum",
    )
    data = source.with_updates(
        axes=tuple(axes),
        metadata={"nfit_data_type": "powder_inelastic"},
    )
    viewer = QtMDHistoSliceViewer(data, x_dim=3, y_dim=0)
    view = viewer.slice_arrays()
    event = SimpleNamespace(
        inaxes=viewer.ax_image,
        xdata=float(view["x_centers"][1]),
        ydata=float(view["y_centers"][1]),
    )

    viewer._on_motion(event)

    assert viewer.cursor_xy_label.text() == "(x, y) = (1.8, 0.75)"
    assert not viewer.cursor_powder_qe_label.isHidden()
    assert viewer.cursor_powder_qe_label.text() == "(|Q|, E) = (1.8, 0.75)"
    assert viewer.cursor_hkle_label.isHidden()
    assert viewer.cursor_q_label.text() == "|Q| = 1.8 Å⁻¹"


def test_qt_cursor_readout_hides_crystal_coordinates_for_powder_and_magnetization():
    pytest.importorskip("PySide6")

    from nfit.dataset import PointListData
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    powder = _tiny_mdhisto_data()
    powder.metadata["nfit_data_type"] = "powder_inelastic"
    powder_viewer = QtMDHistoSliceViewer(powder, x_dim=3, y_dim=2)

    assert powder_viewer.cursor_hkle_label.isHidden()
    assert powder_viewer.cursor_powder_qe_label.isHidden()
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
    assert magnetization_viewer.cursor_powder_qe_label.isHidden()
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

    data = _tiny_1d_mdhisto_data().mutable_copy()
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


def test_qt_viewer_exports_box_profiles_and_slice_data_model_csv(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("PySide6")
    import nfit.qt_slice_viewer as viewer_module
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(
        _with_fit_channels(_tiny_mdhisto_data()),
        x_dim=3,
        y_dim=2,
    )
    viewer.hist_axes_check.setChecked(True)
    paths = {
        "Save x profile": tmp_path / "x.csv",
        "Save y profile": tmp_path / "y.csv",
        "Save displayed data": tmp_path / "data.csv",
        "Save displayed model": tmp_path / "model.csv",
    }
    monkeypatch.setattr(
        viewer_module,
        "get_save_file_name",
        lambda _parent, caption, _default, _filters: (str(paths[caption]), ""),
    )

    viewer.save_x_cut_button.click()
    viewer.save_y_cut_button.click()
    viewer.save_data_button.click()
    viewer.save_model_button.click()

    assert paths["Save x profile"].read_text().splitlines()[0] == (
        "x,intensity,uncertainty"
    )
    assert paths["Save y profile"].read_text().splitlines()[0] == (
        "y,intensity,uncertainty"
    )
    assert paths["Save displayed data"].read_text().splitlines()[0] == (
        "x,y,I,dI"
    )
    assert paths["Save displayed model"].read_text().splitlines()[0] == "x,y,I"
