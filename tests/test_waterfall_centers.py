"""Requested waterfall centers, native-bin membership and GUI/script parity."""

import numpy as np
import pytest

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plotting_core import (
    default_waterfall_step,
    plot_mdhisto_waterfall,
    prepare_mdhisto_waterfall,
)


def energy_map(offset=0.0):
    centers = np.arange(0.0, 50.01, 0.5) + offset
    edges = np.r_[centers - 0.25, centers[-1] + 0.25]
    signal = np.broadcast_to(centers, (2, centers.size)).copy()
    return MDHistoData(
        (MDHistoAxis("H", np.array([-1.0, 0.0, 1.0]), "rlu", "momentum"),
         MDHistoAxis("DeltaE", edges, "meV", "energy")),
        signal, np.full(signal.shape, 2.0), np.zeros(signal.shape, bool), np.ones(signal.shape),
    )


@pytest.mark.parametrize("offset", [0.0, 0.25])
def test_requested_centers_are_independent_of_native_histogram_origin(offset):
    data = energy_map(offset)
    assert default_waterfall_step(data, 1, centered=True) == 5.0
    traces = prepare_mdhisto_waterfall(
        data, x_dim=0, waterfall_dim=1, waterfall_step=5.0,
        waterfall_center_bounds=(5.0, 45.0), waterfall_centered=True,
    )
    np.testing.assert_allclose([t.waterfall_coordinate for t in traces], np.arange(5, 46, 5))
    assert len(traces) == 9
    ax = plot_mdhisto_waterfall(data, x_dim=0, waterfall_dim=1, waterfall_step=5.0,
                              waterfall_center_bounds=(5.0, 45.0), waterfall_centered=True)
    np.testing.assert_allclose([t.waterfall_coordinate for t in ax._nfit_waterfall_traces], np.arange(5, 46, 5))
    for trace in traces:
        center = trace.waterfall_coordinate
        selected = (data.axes[1].centers >= center - 2.5) & (data.axes[1].centers < center + 2.5)
        np.testing.assert_allclose(trace.values, data.signal[:, selected].mean(axis=1))
        np.testing.assert_allclose(trace.errors, 2.0 / np.sqrt(selected.sum()))
    # Legacy API defaults retain the old edge-anchored membership and labels.
    legacy = prepare_mdhisto_waterfall(data, x_dim=0, waterfall_dim=1, waterfall_step=5.0)
    assert legacy[0].waterfall_coordinate == pytest.approx(2.25 + offset)


def test_centered_windows_assign_boundaries_once_and_skip_empty_bins():
    from nfit.measurement_waterfalls import centered_waterfall_bins

    bins = list(centered_waterfall_bins(np.array([2.5, 7.5, 12.5]), 5.0, (5.0, 15.0)))
    assert [center for center, _selected in bins] == [5.0, 10.0, 15.0]
    np.testing.assert_array_equal(np.sum([selected for _center, selected in bins], axis=0), 1)
    # Planning remains bounded by source bins even for enormous requested ranges.
    bins = list(centered_waterfall_bins(np.array([0.0, 5.0]), 5.0, (-1e12, 1e12)))
    assert [center for center, _selected in bins] == [0.0, 5.0]
    assert list(centered_waterfall_bins(np.array([0.0, 5.0]), 5.0, (100.0, 200.0))) == []


def test_centered_counting_waterfall_pools_counts_exposure_and_measured_zeros():
    from nfit.histogram_statistics import (
        EVENT_STATISTICS_KEY,
        EVENT_STATISTICS_METADATA,
        event_statistics_channels,
    )

    data = energy_map()
    exposure = np.broadcast_to(np.arange(data.shape[1]) + 1.0, data.shape).copy()
    numerator = exposure * (data.signal + 1)
    numerator[:, 10] = 0.0  # A covered, measured zero at E=5 must still contribute exposure.
    variance = numerator.copy()
    data = data.with_updates(
        signal=numerator / exposure, errors=np.sqrt(variance) / exposure,
        metadata={EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA), "zero_event_bins_are_measured": True},
        auxiliary_channels=event_statistics_channels(numerator, variance, exposure),
    )
    trace, = prepare_mdhisto_waterfall(data, x_dim=0, waterfall_dim=1, waterfall_step=5.0,
                                     waterfall_center_bounds=(5.0, 5.0), waterfall_centered=True)
    selected = (data.axes[1].centers >= 2.5) & (data.axes[1].centers < 7.5)
    expected_exposure = exposure[:, selected].sum(axis=1)
    np.testing.assert_allclose(trace.values, numerator[:, selected].sum(axis=1) / expected_exposure)
    np.testing.assert_allclose(trace.errors, np.sqrt(variance[:, selected].sum(axis=1)) / expected_exposure)
    assert trace.measurement is not None


def test_gui_requested_centers_labels_and_saved_plot_script_agree(monkeypatch):
    import matplotlib.pyplot as plt

    from nfit.plot_recipes import new_plot_entry, render_plot
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    data = energy_map()
    viewer = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1)
    try:
        viewer.view_mode_combo.setCurrentIndex(1)
        viewer.waterfall_step_spin.setValue(5.0)
        viewer.waterfall_center_min_spin.setValue(5.0)
        viewer.waterfall_center_max_spin.setValue(45.0)
        expected = np.arange(5, 46, 5)
        np.testing.assert_allclose([t.waterfall_coordinate for t in viewer._current_waterfall_traces], expected)
        assert viewer._axis_y_label.text() == "w"
        assert viewer.swap_axes_button.text() == "Swap x/w"
        assert "intensity" in viewer.y_combo.toolTip()
        assert viewer.y_min_spin.toolTip().startswith("Lower displayed limit")
        settings = viewer.current_plot_settings()
        assert settings["waterfall_centered"] is True
        entry = new_plot_entry("Energy traces", None, settings, plot_type="mdhisto_waterfall")
        figure = render_plot(entry, data)
        np.testing.assert_allclose([t.waterfall_coordinate for t in figure.axes[0]._nfit_waterfall_traces], expected)
        script = viewer.figure_script()
        assert "waterfall_centered=True" in script
        monkeypatch.setattr(plt, "show", lambda: None)
        namespace = {"test_data": data}
        exec(script.replace("data = ...", "data = test_data"), namespace)
        np.testing.assert_allclose([t.waterfall_coordinate for t in namespace["ax"]._nfit_waterfall_traces], expected)
        viewer.swap_axes_button.click()
        assert (viewer.model.x_dim, viewer.model.y_dim) == (1, 0)
        assert viewer.waterfall_center_bounds is None
        assert viewer.waterfall_step_auto
        assert viewer._current_waterfall_traces
        viewer.view_mode_combo.setCurrentIndex(0)
        assert viewer._axis_y_label.text() == "y"
        assert viewer.swap_axes_button.text() == "Swap x/y"
    finally:
        viewer.window.close()


def test_old_saved_waterfall_recipes_keep_their_grid_until_edited():
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(energy_map(), x_dim=0, y_dim=1)
    try:
        viewer.apply_plot_settings(dict(view_mode="waterfall", waterfall_step=5.0,
                                       waterfall_step_auto=False, waterfall_center_bounds=(5, 45)))
        assert viewer.waterfall_centered is False
        assert viewer._current_waterfall_traces[0].waterfall_coordinate == pytest.approx(7.25)
        viewer._set_waterfall_center_bound("min", 5.0)
        assert viewer.waterfall_centered is True
        assert viewer._current_waterfall_traces[0].waterfall_coordinate == 5.0
    finally:
        viewer.window.close()
