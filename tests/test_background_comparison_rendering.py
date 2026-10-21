from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.figure import Figure

from nfit.background_channels import accumulate_background_channel
from nfit.box_cuts import histogram_box_profiles, rotated_box_profiles
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plotting_core import MDHistoSliceViewer
from nfit.qt_slice_modes import FitComparisonController, PlotLayoutController


class _DifferenceColors:
    def snapshot(self):
        from nfit.slice_comparison import default_difference_colors

        return default_difference_colors()

    def apply_to_model(self, model):
        model.cmap = "bwr"
        model.color_scale = "linear"
        model.symmetric_about_zero = True
        model.autoscale = True

    def sync_limits(self, lower, upper):
        self.limits = lower, upper


def _comparison_controller(*, cuts=False, angle=0.0, popout=False):
    original = MDHistoData(
        axes=(MDHistoAxis("x", np.arange(4.), "meV", "energy"),
              MDHistoAxis("y", np.arange(3.), "r.l.u.", "momentum")),
        signal=np.array([[10., 12.], [14., 16.], [18., 20.]]),
        errors=np.full((3, 2), 2.), mask=np.zeros((3, 2), dtype=bool), num_events=np.ones((3, 2)),
    )
    contribution = np.array([[3., 5.], [7., 9.], [11., 25.]])
    result = original.with_updates(
        signal=original.signal - contribution,
        errors=np.full(original.shape, np.sqrt(5.)),
    )
    data = accumulate_background_channel(
        original, result, contribution=contribution,
        contribution_errors=np.ones(original.shape), valid=np.ones(original.shape, bool),
    )
    model = MDHistoSliceViewer(data, x_dim=0, y_dim=1, channel="signal", cmap="viridis")
    model.auto_limits = "min/max"
    viewer = SimpleNamespace(
        model=model, data=data, figure=Figure(), unmask_model=False,
        coverage_threshold=0., xcut_percent=20, ycut_percent=20,
        _roi_extents=(0., 3., 0., 2.), _roi_angle=angle,
        _current_slice=None, _current_slice_source_key=None,
        _last_plot_view_mode=None, rectangle_selector=None,
        _compare_axes=[], _compare_colorbar_axes=[],
        _diff_color_controller=_DifferenceColors(),
        popout_cuts_check=SimpleNamespace(isChecked=lambda: popout),
        canvas=SimpleNamespace(draw_idle=lambda: None),
        marker_size=4., marker_face_color="none", line_color="black",
        fit_line_color="orange", line_plot_width=1.,
        _comparison_cuts_active=lambda: cuts,
        _default_roi_extents=lambda: (0., 3., 0., 2.),
        _slice_source_key=lambda: ("retained",),
        _slice_sources_are_cacheable=lambda: True,
        _panel_ratio=lambda value: value / 100.,
    )
    for name in ("_suppress_matplotlib_coordinate_status", "_create_rectangle_selector",
                 "_sync_control_visibility", "_sync_limit_spinboxes", "_sync_view_limit_controls",
                 "_apply_figure_font_size", "_apply_axis_linewidth", "_connect_view_limit_callbacks",
                 "_clear_roi_sum_annotation", "_show_roi_sum_annotation", "_sync_export_controls",
                 "_apply_axes_ratio", "_draw_gridline_overlay"):
        setattr(viewer, name, lambda *args, **kwargs: None)
    viewer._smoothed_slice_view = lambda view, *, channel: view
    controller = FitComparisonController(viewer)
    viewer._comparison_panel_model = controller._comparison_panel_model
    viewer._ensure_fit_compare_layout = PlotLayoutController(viewer)._ensure_fit_compare_layout
    viewer._set_roi_extents = lambda extents, **kwargs: controller._update_background_compare_cuts(extents)
    return controller, viewer


def test_background_maps_use_all_channels_and_shared_combined_limits():
    controller, viewer = _comparison_controller()
    controller._draw_background_panels_view(None, None, None, (0, 1))

    images = [axis.collections[0] for axis in viewer._compare_axes]
    assert [axis.get_title() for axis in viewer._compare_axes] == [
        "Unsubtracted", "Background", "Difference",
    ]
    assert images[0].norm is images[1].norm
    assert (images[0].norm.vmin, images[0].norm.vmax) == pytest.approx((3., 25.))
    assert images[2].norm.vmin == -images[2].norm.vmax
    assert images[2].cmap.name == "bwr"
    np.testing.assert_allclose(images[0].get_array(), [[10., 14., 18.], [12., 16., 20.]])
    np.testing.assert_allclose(images[1].get_array(), [[3., 7., 11.], [5., 9., 25.]])
    assert viewer._current_slice is viewer._background_comparison_views["signal"]
    np.testing.assert_allclose(viewer._current_slice["errors"], np.sqrt(5.))


@pytest.mark.parametrize("angle", [0., 30.])
def test_background_profiles_keep_channel_uncertainties_and_physical_difference(angle):
    controller, viewer = _comparison_controller(cuts=True, angle=angle)
    controller._draw_background_panels_view(None, None, None, (0, 1))
    profiles = histogram_box_profiles if angle == 0 else rotated_box_profiles
    for channel, error_key, axis, label in (
        ("unsubtracted", "unsubtracted_errors", viewer.ax_fit_cut, "Unsubtracted"),
        ("background", "background_errors", viewer.ax_fit_cut, "Background"),
        ("signal", "errors", viewer.ax_residual_cut, "Difference"),
    ):
        view = viewer._background_comparison_views[channel]
        expected = profiles(view, view[channel], view[error_key], viewer._roi_extents,
                            angle, channel=channel)
        line = next(item for item in axis.containers if item.get_label() == label).lines[0]
        np.testing.assert_allclose(line.get_ydata(), expected.x[1], equal_nan=True)
        if channel == "signal":
            np.testing.assert_allclose(viewer._current_x_cut[2], expected.x[2], equal_nan=True)
            np.testing.assert_allclose(viewer._current_y_cut[2], expected.y[2], equal_nan=True)
            assert viewer._current_box_profiles.x_measurement.contract.kind == "continuous"
    assert "σ" not in viewer.ax_residual_cut.get_ylabel()
    if angle:
        assert viewer.ax_residual_cut.get_xlabel().startswith("Box x")


def test_background_replay_replaces_only_difference_and_preserves_overlays():
    controller, viewer = _comparison_controller(cuts=True)
    controller._draw_background_panels_view(None, None, None, (0, 1))
    overlay_lines = tuple(viewer.ax_fit_cut.lines)
    view = dict(viewer._current_slice)
    replacement = histogram_box_profiles(
        view, view["signal"], np.full_like(view["errors"], 7.), viewer._roi_extents,
    )
    controller._render_background_difference_profiles(replacement)
    assert tuple(viewer.ax_fit_cut.lines) == overlay_lines
    assert viewer._current_box_profiles is replacement
    np.testing.assert_array_equal(viewer._current_x_cut[2], replacement.x[2])


def test_background_comparison_popout_reuses_layout_and_preserves_zoom():
    controller, viewer = _comparison_controller(cuts=True, popout=True)
    controller._draw_background_panels_view((.3, 2.7), (.2, 1.8), (0, 1), (0, 1))
    for axis in viewer._compare_axes:
        assert axis.get_xlim() == pytest.approx((.3, 2.7))
        assert axis.get_ylim() == pytest.approx((.2, 1.8))
    assert not viewer.ax_fit_cut.get_visible()
    assert not viewer.ax_residual_cut.get_visible()
    assert not viewer.ax_ycut.get_visible()
    assert not viewer.ax_residual_ycut.get_visible()


def test_comparison_presentation_does_not_reverse_import_viewer():
    from nfit import qt_slice_comparison

    tree = ast.parse(Path(qt_slice_comparison.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "qt_slice_viewer" not in imports
    assert "qt_slice_modes" not in imports


@pytest.mark.parametrize("angle", [0., 30.])
def test_restored_original_pixel_survives_diagnostic_box_masks(angle):
    from nfit.slice_comparison import plot_mdhisto_background_comparison

    controller, viewer = _comparison_controller(cuts=True, angle=angle)
    original = viewer.data.with_updates(
        signal=np.full(viewer.data.shape, 10.), errors=np.full(viewer.data.shape, 2.),
        metadata={}, auxiliary_channels={},
    )
    signal = np.full(original.shape, 7.)
    errors = np.full(original.shape, np.sqrt(5.))
    mask = np.zeros(original.shape, bool)
    signal[0, 0] = errors[0, 0] = np.nan
    mask[0, 0] = True
    result = original.with_updates(signal=signal, errors=errors, mask=mask)
    viewer.data = accumulate_background_channel(
        original, result, contribution=np.full(original.shape, 3.),
        contribution_errors=np.ones(original.shape), valid=~mask,
    )
    viewer.model.data = viewer.data
    viewer._roi_extents = (0.01, .99, 0.01, .99)
    controller._draw_background_panels_view(None, None, (0, 1), (0, 1))
    line = next(item for item in viewer.ax_fit_cut.containers
                if item.get_label() == "Unsubtracted").lines[0]
    np.testing.assert_allclose(line.get_ydata(), [10.])
    assert np.isnan(viewer._current_slice["signal"][0, 0])
    assert np.isnan(viewer._current_x_cut[1]).all()

    figure = plot_mdhisto_background_comparison(
        viewer.data, x_dim=0, y_dim=1, show_histogram_axes=True,
        roi_extents=viewer._roi_extents, roi_angle=angle,
    )
    original_axis = next(axis for axis in figure.axes if axis.get_title() == "Unsubtracted")
    original_cut = next(axis for axis in figure.axes
                        if axis is not original_axis and axis.get_shared_x_axes().joined(axis, original_axis))
    np.testing.assert_allclose(original_cut.containers[0].lines[0].get_ydata(), [10.])


def test_diagnostic_profiles_discard_primary_dependencies_and_additive_payloads():
    from nfit.measurement_aggregation import (
        MEASUREMENT_STATISTICS_CHANNELS,
        MEASUREMENT_STATISTICS_KEY,
        initialize_measurement_statistics,
    )
    from nfit.measurement_contracts import MeasurementContract
    from nfit.measurement_dependencies import CountingDependencies, independent_source_dependencies
    from nfit.slice_comparison import comparison_profile_view

    _, viewer = _comparison_controller()
    view = viewer._comparison_panel_model(viewer.data, "unsubtracted").slice_arrays()
    dependencies = independent_source_dependencies(view["errors"]**2, "difference")
    view["source_dependencies"] = dependencies
    view["counting_dependencies"] = CountingDependencies(
        numerator_dependencies=dependencies, exposure_dependencies=dependencies,
    )
    contract = MeasurementContract(kind="continuous", estimator="inverse_variance_mean",
                                   quantity="intensity", value_units="counts")
    arrays, marker = initialize_measurement_statistics(view["signal"], view["errors"], contract)
    view[MEASUREMENT_STATISTICS_KEY] = marker
    view.update(zip(MEASUREMENT_STATISTICS_CHANNELS, arrays, strict=True))
    view["measurement_contract"] = contract.to_dict()
    assert comparison_profile_view(view, "signal") is view
    primary = histogram_box_profiles(view, view["signal"], view["errors"], viewer._roi_extents)
    assert primary.x_measurement.data.source_dependencies is not None

    diagnostic = comparison_profile_view(view, "unsubtracted")
    for name in ("source_dependencies", "counting_dependencies", "measurement_contract",
                 MEASUREMENT_STATISTICS_KEY, *MEASUREMENT_STATISTICS_CHANNELS):
        assert name not in diagnostic
        assert name in view
    profiles = histogram_box_profiles(
        diagnostic, diagnostic["unsubtracted"], diagnostic["errors"], viewer._roi_extents,
        channel="unsubtracted",
    )
    np.testing.assert_allclose(profiles.x[1], [11., 15., 19.])
    np.testing.assert_allclose(profiles.x[2], np.sqrt(2.))
    assert profiles.x_measurement.data.source_dependencies is None
