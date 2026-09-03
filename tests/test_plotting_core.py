import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nfit import PointData4D
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.plotting import (
    MDHistoSliceViewer,
    default_tiled_slice_step,
    gaussian_smooth_nan,
    gaussian_smooth_uncertainty,
    inverse_variance_weighted_profile,
    mdhisto_with_signal_like,
    plot_2d_map,
    plot_energy_cut,
    plot_mdhisto_auto,
    plot_mdhisto_fit_comparison,
    plot_mdhisto_fit_line_comparison,
    plot_mdhisto_line,
    plot_mdhisto_slice,
    plot_mdhisto_tiled_slices,
    plot_mdhisto_waterfall,
    plot_q_cut,
    prepare_mdhisto_tiled_slices,
    prepare_mdhisto_waterfall,
    residual_mdhisto,
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


def test_gaussian_plot_smoothing_propagates_independent_uncertainties():
    errors = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    smoothed = gaussian_smooth_uncertainty(errors, 1.0)

    assert np.isnan(smoothed[2])
    assert np.all(smoothed[[0, 1, 3, 4]] < 1.0)


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


def test_prepare_tiled_slices_defaults_to_each_value_when_fewer_than_nine():
    data = _tiny_mdhisto_data()

    panels = prepare_mdhisto_tiled_slices(
        data,
        x_dim=3,
        y_dim=2,
        tile_dim=1,
    )

    assert len(panels) == data.shape[1] == 3
    assert default_tiled_slice_step(data, 1) == pytest.approx(1.0)
    assert all(panel.values.shape == data.shape[2:] for panel in panels)
    np.testing.assert_allclose(panels[0].values, data.signal[1, 0, :, :])
    assert panels[0].label.startswith("[H,-H,0] = ")


def test_prepare_tiled_slices_auto_step_targets_nine_panels():
    axes = (
        MDHistoAxis("x", np.arange(3.0), "", "unknown"),
        MDHistoAxis("y", np.arange(3.0), "", "unknown"),
        MDHistoAxis("scan", np.arange(13.0), "K", "temperature"),
    )
    signal = np.ones((2, 2, 12), dtype=float)
    data = MDHistoData(
        axes,
        signal,
        np.ones_like(signal),
        np.zeros_like(signal, dtype=bool),
        np.ones_like(signal),
    )

    panels = prepare_mdhisto_tiled_slices(
        data,
        x_dim=0,
        y_dim=1,
        tile_dim=2,
    )

    assert len(panels) == 9


def test_plot_tiled_slices_uses_one_shared_norm_and_far_right_colorbar():
    data = _tiny_mdhisto_data()

    figure = plot_mdhisto_tiled_slices(
        data,
        x_dim=3,
        y_dim=2,
        tile_dim=1,
        tile_range=(0.5, 2.5),
    )

    axes = figure._nfit_tiled_axes
    assert len(axes) == 3
    assert len(figure.axes) == 4
    assert all(axis.collections[0].norm is axes[0].collections[0].norm for axis in axes)
    assert figure.axes[-1].get_position().x0 > max(axis.get_position().x1 for axis in axes)
    assert all(axis.texts[0].get_position() == (0.97, 0.03) for axis in axes)
    assert all(
        axis.texts[0].get_bbox_patch().get_alpha() == pytest.approx(0.65)
        for axis in axes
    )
    assert figure.axes[-1].yaxis.label.get_text() == r"$I(\mathbf{Q},E)$ (a.u.)"


def test_plot_tiled_slices_can_hide_labels_and_autoscale_each_panel():
    data = _tiny_mdhisto_data()

    figure = plot_mdhisto_tiled_slices(
        data,
        x_dim=3,
        y_dim=2,
        tile_dim=1,
        tile_range=(0.5, 2.5),
        show_tile_labels=False,
        local_color_scales=True,
    )

    axes = figure._nfit_tiled_axes
    colorbars = figure._nfit_tiled_colorbars
    assert len(colorbars) == len(axes) == 3
    assert len(figure.axes) == 6
    assert all(not axis.texts for axis in axes)
    assert all(
        axis.collections[0].norm is not axes[0].collections[0].norm
        for axis in axes[1:]
    )
    assert all(
        colorbar.ax.yaxis.label.get_text() == r"$I(\mathbf{Q},E)$ (a.u.)"
        for colorbar in colorbars
    )

    with pytest.raises(ValueError, match="requires autoscale=True"):
        plot_mdhisto_tiled_slices(
            data,
            x_dim=3,
            y_dim=2,
            tile_dim=1,
            autoscale=False,
            manual_vmin=0.0,
            manual_vmax=1.0,
            local_color_scales=True,
        )


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


def test_derived_coverage_channels_have_labels_without_auxiliary_storage():
    data = _tiny_mdhisto_data()

    coverage = MDHistoSliceViewer(
        data,
        x_dim=3,
        y_dim=2,
        channel="coverage_fraction",
    )
    coverage_mask = MDHistoSliceViewer(
        data,
        x_dim=3,
        y_dim=2,
        channel="coverage_mask",
    )

    assert "coverage_fraction" not in data.auxiliary_channels
    assert coverage._channel_label() == "Coverage (fraction)"
    assert coverage_mask._channel_label() == "Coverage mask"
    assert coverage.slice_arrays()["coverage_fraction"].shape == data.shape[2:]
    assert coverage_mask.slice_arrays()["coverage_mask"].shape == data.shape[2:]


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
    second = second.with_updates(signal=second.signal + 2.0)

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


def test_mdhisto_slice_viewer_masks_integrated_bins_below_coverage_threshold():
    data = _tiny_mdhisto_data()
    coverage = np.ones(data.shape, dtype=float)
    coverage[0, ...] = 0.5
    data = data.with_updates(
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(
                coverage,
                label="Coverage",
                unit="fraction",
            )
        }
    )
    viewer = MDHistoSliceViewer(
        data,
        x_dim=3,
        y_dim=2,
        coverage_threshold=0.8,
    )
    viewer.selections[0] = (0.25, 0.75)
    viewer.selections[1] = (1.5, 1.5)
    viewer.integrate_checks[0] = True

    masked = viewer.slice_arrays()

    np.testing.assert_allclose(masked["coverage_fraction"], 0.75)
    assert np.all(masked["coverage_mask"])
    assert np.all(np.isnan(masked["signal"]))

    viewer.coverage_threshold = 0.7
    retained = viewer.slice_arrays()

    assert not np.any(retained["coverage_mask"])
    np.testing.assert_allclose(
        retained["signal"],
        np.sum(data.signal[:, 1, :, :], axis=0),
    )


def test_waterfall_coarsening_uses_its_own_coverage_threshold():
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

    strict = prepare_mdhisto_waterfall(
        data,
        x_dim=3,
        waterfall_dim=2,
        waterfall_step=4.0,
        coverage_threshold=0.8,
    )
    permissive = prepare_mdhisto_waterfall(
        data,
        x_dim=3,
        waterfall_dim=2,
        waterfall_step=4.0,
        coverage_threshold=0.7,
    )

    assert strict
    assert np.all(np.isnan(strict[0].values))
    assert np.any(np.isfinite(permissive[0].values))


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
    editable = data.mutable_copy()
    editable.num_events[:, 1, 2, 3] = 0.0
    data = editable.immutable_copy()
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
