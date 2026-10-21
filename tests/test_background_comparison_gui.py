from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nfit.backgrounds import subtract_aligned_background
from nfit.box_cuts import histogram_box_profiles, rotated_box_profiles
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plot_recipes import (
    new_plot_entry,
    plot_entry_from_dict,
    plot_entry_to_dict,
    render_plot,
)
from nfit.slice_comparison import default_difference_colors


def _histograms():
    axes = (
        MDHistoAxis("H", np.arange(5.), "rlu", "momentum"),
        MDHistoAxis("E", np.arange(4.), "meV", "energy"),
    )
    values = np.arange(12.).reshape(4, 3) + 10.
    original = MDHistoData(
        axes, values, np.full(values.shape, 2.), np.zeros(values.shape, bool),
        np.ones(values.shape),
    )
    background = original.with_updates(signal=values / 2 + 5, errors=np.ones(values.shape))
    result = subtract_aligned_background(original, background)
    result = result.with_updates(metadata={
        **result.metadata, "fit": result.signal + .2,
        "residual": np.full(result.shape, .2 / np.sqrt(5.)),
    })
    return original, background, result


@pytest.fixture
def viewer_factory(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewers, exceptions = [], []
    monkeypatch.setattr(sys, "excepthook", lambda *info: exceptions.append(info))

    def make(data):
        viewer = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1, cmap="viridis")
        viewers.append(viewer)
        return viewer

    yield make
    for viewer in viewers:
        viewer.window.close()
    assert not exceptions, [(str(info[0]), str(info[1])) for info in exceptions]


def _images(viewer):
    return [axis.collections[0] for axis in viewer._compare_axes]


def test_background_checkbox_availability_layout_and_shared_signal_norm(viewer_factory):
    original, background, result = _histograms()
    viewer = viewer_factory([result, original])
    assert viewer.show_background_check.isEnabled()
    assert viewer.show_background_check.toolTip()
    layout = viewer.show_background_check.parent().layout()
    model_row = layout.getItemPosition(layout.indexOf(viewer.show_fit_check))[0]
    background_row = layout.getItemPosition(layout.indexOf(viewer.show_background_check))[0]
    assert background_row > model_row
    color_layout = viewer.color_group.parent().layout()
    assert color_layout.indexOf(viewer.diff_color_group) == color_layout.indexOf(viewer.color_group) + 1
    assert viewer.diff_color_group.isHidden()

    viewer.show_background_check.setChecked(True)
    assert viewer._background_panels_active()
    assert not viewer.diff_color_group.isHidden()
    assert [axis.get_title() for axis in viewer._compare_axes] == [
        "Unsubtracted", "Background", "Difference",
    ]
    images = _images(viewer)
    np.testing.assert_allclose(images[0].get_array(), original.signal.T)
    np.testing.assert_allclose(images[1].get_array(), background.signal.T)
    np.testing.assert_allclose(images[2].get_array(), result.signal.T)
    assert images[0].norm is images[1].norm
    assert (images[0].norm.vmin, images[0].norm.vmax) == pytest.approx((10., 21.))
    assert images[2].cmap.name == "bwr"
    assert images[2].norm.vmin == -images[2].norm.vmax
    assert viewer._diff_color_controller.snapshot() == default_difference_colors()
    viewer._set_dataset_index(1)
    assert not viewer.show_background_check.isEnabled()
    assert not viewer.show_background


def test_background_colors_are_independent_and_fit_modes_are_exclusive(viewer_factory):
    viewer = viewer_factory(_histograms()[2])
    viewer.show_fit_check.setChecked(True)
    viewer.show_residual_check.setChecked(True)
    assert viewer._fit_panels_active()
    viewer.show_background_check.setChecked(True)
    assert viewer._background_panels_active()
    assert not viewer.show_fit_check.isChecked()
    assert not viewer.show_residual_check.isChecked()
    viewer.cmap_combo.setCurrentText("magma")
    viewer.diff_cmap_combo.setCurrentText("coolwarm")
    viewer.diff_vmax_spin.setValue(2.)
    images = _images(viewer)
    assert [image.cmap.name for image in images] == ["magma", "magma", "coolwarm"]
    assert (images[2].norm.vmin, images[2].norm.vmax) == pytest.approx((-2., 2.))
    assert viewer.model.autoscale
    viewer.show_fit_check.setChecked(True)
    assert viewer._fit_panels_active()
    assert not viewer.show_background_check.isChecked()
    assert viewer.diff_color_group.isHidden()
    viewer.show_fit_check.setChecked(False)
    assert not viewer._comparison_panels_active()
    viewer.show_background_check.setChecked(True)
    assert _images(viewer)[2].cmap.name == "coolwarm"


def test_background_cursor_uses_hovered_channel_and_releases_prepared_views(viewer_factory):
    viewer = viewer_factory(_histograms()[2])
    viewer.show_background_check.setChecked(True)
    for axis, title in zip(viewer._compare_axes, ("Unsubtracted", "Background", "Difference"), strict=True):
        viewer._on_motion(SimpleNamespace(inaxes=axis, xdata=.5, ydata=.5))
        assert viewer.cursor_intensity_label.text().startswith(f"{title} = ")
    viewer.show_background_check.setChecked(False)
    assert not viewer._background_comparison_views
    viewer.show_background_check.setChecked(True)
    assert viewer._background_comparison_views
    viewer.window.close()
    viewer.release_loaded_data()
    assert not viewer._background_comparison_views


def test_background_difference_replays_source_covariance_without_replacing_overlays(viewer_factory, tmp_path):
    from nfit import bin_mdevent_group, project_measured_background_mdevent
    from nfit.background_profile_queries import replay_cached_background_box_profiles
    from tests.test_mdevent_background_statistics import _directional_fixture
    from tests.test_qt_background_profiles import _wait

    sample, source = _directional_fixture(tmp_path)
    target = bin_mdevent_group(sample, lower=[-2., -2., -2., -.5],
        upper=[.5, 2., 2., .5], num_bins=[2, 2, 1, 1])
    projected = project_measured_background_mdevent(sample, source, target)
    data = subtract_aligned_background(target, projected)
    viewer = viewer_factory(data)
    viewer.show_background_check.setChecked(True)
    viewer.hist_axes_check.setChecked(True)
    controller = viewer._background_profiles
    controller._timer.setInterval(1)
    extents = (-2., .5, -2., 2.)
    viewer._set_roi_extents(extents, update_cuts=True, draw=False)
    original_overlays = tuple(viewer.ax_fit_cut.containers)
    _wait(viewer.app, lambda: controller.status in {"exact", "approximate"})
    assert controller.status == "exact", controller.reason
    assert tuple(viewer.ax_fit_cut.containers) == original_overlays
    assert "covariance replay" in viewer.ax_residual_cut.get_ylabel()
    assert viewer.current_plot_settings()["background_uncertainty"] == "replay"
    assert "'background_uncertainty': 'replay'" in viewer._figure_script()
    expected = replay_cached_background_box_profiles(data, x_dim=0, y_dim=1,
        selections=viewer.model._normalized_selections(), extents=extents,
        display_selected=np.isfinite(viewer._current_slice["signal"]))
    for actual, wanted in zip(viewer._current_x_cut, expected.x, strict=True):
        np.testing.assert_allclose(actual, wanted, equal_nan=True)
    viewer.popout_cuts_check.setChecked(True)
    _wait(viewer.app, lambda: controller.status == "exact")
    assert viewer._cut_viewers["x"].data.metadata["background_profile_uncertainty"] == "source_covariance"


def test_background_reuse_respects_mutable_original_mask_metadata(viewer_factory):
    data = _histograms()[2]
    mask = np.zeros(data.shape, dtype=bool)
    data = data.with_updates(metadata={**data.metadata, "nfit_mask": mask})
    viewer = viewer_factory(data)
    viewer.show_background_check.setChecked(True)
    assert np.isfinite(viewer._background_comparison_views["unsubtracted"]["unsubtracted"][0, 0])
    data.metadata["nfit_mask"][0, 0] = True
    viewer.update_plot(reuse_slice=True)
    assert np.isnan(viewer._background_comparison_views["unsubtracted"]["unsubtracted"][0, 0])


@pytest.mark.parametrize("angle", [0., 30.])
def test_background_box_profiles_preserve_each_channels_uncertainty(viewer_factory, angle):
    viewer = viewer_factory(_histograms()[2])
    viewer.show_background_check.setChecked(True)
    viewer.hist_axes_check.setChecked(True)
    viewer.roi_angle_spin.setValue(angle)
    viewer._set_roi_extents((.5, 3.5, .5, 2.5), update_cuts=True, draw=False)
    np.testing.assert_allclose(viewer._background_comparison_views["unsubtracted"]["unsubtracted_errors"], 2.)
    np.testing.assert_allclose(viewer._background_comparison_views["background"]["background_errors"], 1.)
    np.testing.assert_allclose(viewer._background_comparison_views["signal"]["errors"], np.sqrt(5.))
    prepare = histogram_box_profiles if angle == 0 else rotated_box_profiles
    for channel, errors, axis, label in (
        ("unsubtracted", "unsubtracted_errors", viewer.ax_fit_cut, "Unsubtracted"),
        ("background", "background_errors", viewer.ax_fit_cut, "Background"),
        ("signal", "errors", viewer.ax_residual_cut, "Difference"),
    ):
        view = viewer._background_comparison_views[channel]
        expected = prepare(view, view[channel], view[errors], viewer._roi_extents,
                           angle, channel=channel)
        container = next(item for item in axis.containers if item.get_label() == label)
        line = container.lines[0]
        np.testing.assert_allclose(line.get_ydata(), expected.x[1], equal_nan=True)
        segments = container.lines[2][0].get_segments()
        for segment, value, uncertainty in zip(segments, expected.x[1], expected.x[2], strict=True):
            if np.isfinite(value) and np.isfinite(uncertainty):
                np.testing.assert_allclose(segment[:, 1], [value - uncertainty, value + uncertainty])
        if channel == "signal":
            np.testing.assert_allclose(viewer._current_x_cut[2], expected.x[2], equal_nan=True)
            np.testing.assert_allclose(viewer._current_y_cut[2], expected.y[2], equal_nan=True)
            assert viewer._current_box_profiles.x_measurement.contract.kind == "continuous"
    assert "σ" not in viewer.ax_residual_cut.get_ylabel()
    viewer.popout_cuts_check.setChecked(True)
    viewer.app.processEvents()
    assert not viewer.ax_residual_cut.get_visible()
    viewer.popout_cuts_check.setChecked(False)
    viewer.hist_axes_check.setChecked(False)
    viewer.hist_axes_check.setChecked(True)
    assert viewer._comparison_difference_cut_active()


def test_background_view_state_and_saved_figure_script_roundtrip(viewer_factory, monkeypatch):
    original, _background, result = _histograms()
    viewer = viewer_factory([result, result.with_updates(signal=result.signal * 2), original])
    viewer.hold_view_settings_check.setChecked(False)
    viewer.show_background_check.setChecked(True)
    viewer.diff_cmap_combo.setCurrentText("coolwarm")
    viewer.diff_vmax_spin.setValue(3.)
    settings = viewer.current_plot_settings()
    saved_colors = viewer._diff_color_controller.snapshot()
    viewer._set_dataset_index(1)
    assert not viewer.show_background
    viewer._set_dataset_index(0)
    assert viewer.show_background
    assert viewer._diff_color_controller.snapshot() == saved_colors
    viewer._set_dataset_index(2)
    assert not viewer.show_background_check.isEnabled()
    viewer._set_dataset_index(0)
    assert viewer.show_background

    entry = new_plot_entry("Background comparison", "source", settings, plot_type="mdhisto_slice")
    restored = plot_entry_from_dict(json.loads(json.dumps(plot_entry_to_dict(entry))))
    target = viewer_factory(result)
    target.apply_plot_settings(restored.settings)
    assert target.show_background_check.isChecked()
    assert target._diff_color_controller.snapshot() == saved_colors

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    script = target.figure_script()
    assert "PySide" not in script
    script = script.replace("data = ...  # Replace with your MDHistoData object", "data = prepared")
    namespace = {"prepared": result}
    exec(compile(script, "background_figure.py", "exec"), namespace)
    expected = render_plot(restored, result)
    try:
        actual_maps = [axis.collections[0] for axis in namespace["fig"].axes if axis.get_title() in {
            "Unsubtracted", "Background", "Difference",
        }]
        expected_maps = [axis.collections[0] for axis in expected.axes if axis.get_title() in {
            "Unsubtracted", "Background", "Difference",
        }]
        assert len(actual_maps) == len(expected_maps) == 3
        for actual, wanted in zip(actual_maps, expected_maps, strict=True):
            np.testing.assert_allclose(actual.get_array(), wanted.get_array())
            assert actual.cmap.name == wanted.cmap.name
            assert (actual.norm.vmin, actual.norm.vmax) == pytest.approx((wanted.norm.vmin, wanted.norm.vmax))
    finally:
        plt.close(namespace["fig"])
        plt.close(expected)
