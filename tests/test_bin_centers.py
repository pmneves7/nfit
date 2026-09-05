import numpy as np
import pytest

from nfit import ArrayRebinSource, rebin_nd, rebin_nd_stream
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plotting import prepare_mdhisto_tiled_slices


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("fractional", [False, True])
@pytest.mark.parametrize("resolution", [{"step_size": [0.2]}, {"num_bins": [9]}])
def test_center_grid_matches_stream_and_keeps_endpoints(backend, fractional, resolution):
    centers = np.linspace(-0.2, 1.4, 9)
    values = np.arange(1., 10.)
    kwargs = dict(lower=[-0.2], upper=[1.4], backend=backend,
                  fractional=fractional, **resolution)
    result = rebin_nd(values, centers[:, None], **kwargs)
    stream = rebin_nd_stream(ArrayRebinSource(values, centers[:, None], batch_size=3), **kwargs)
    for rebinned in (result, stream):
        np.testing.assert_allclose(rebinned.bin_centers_list[0], centers, atol=1.e-14)
        np.testing.assert_allclose(rebinned.bins_list[0][[0, -1]], [-0.3, 1.5])
        np.testing.assert_allclose(rebinned.binned_data, values)


def test_step_grid_does_not_shorten_final_bin():
    result = rebin_nd([1., 2.], [[0.], [0.9]], lower=[0.], upper=[1.], step_size=[0.3])
    np.testing.assert_allclose(result.bin_centers_list[0], [0., 0.3, 0.6, 0.9], atol=1.e-15)
    np.testing.assert_allclose(np.diff(result.bins_list[0]), 0.3)


def test_equal_center_limits_with_step_make_one_finite_bin():
    result = rebin_nd([1.], [[0.]], lower=[0.], upper=[0.], step_size=[0.2])
    np.testing.assert_allclose(result.bins_list[0], [-0.1, 0.1])


def test_single_bin_integration_keeps_edge_limits():
    result = rebin_nd([1., 3.], [[-0.2], [0.2]], lower=[-0.3], upper=[0.3], num_bins=[1])
    np.testing.assert_allclose(result.bins_list[0], [-0.3, 0.3])
    np.testing.assert_allclose(result.binned_data, [2.])


def _fine_energy_grid():
    energy_edges = np.linspace(-0.3, 1.5, 19)
    axes = (MDHistoAxis("x", np.arange(3.), "", "unknown"), MDHistoAxis("y", np.arange(3.), "", "unknown"),
            MDHistoAxis("DeltaE", energy_edges, "meV", "energy"))
    signal = np.broadcast_to(np.repeat(np.arange(9.), 2), (2, 2, 18)).copy()
    return MDHistoData(axes, signal, np.ones_like(signal), np.zeros_like(signal, dtype=bool), np.ones_like(signal))


def test_tiled_energy_centers_group_fine_bins_without_offset():
    data = _fine_energy_grid()
    panels = prepare_mdhisto_tiled_slices(data, x_dim=0, y_dim=1, tile_dim=2,
                                        tile_range=(-0.2, 1.4), tile_step=0.2)
    assert len(panels) == 9
    np.testing.assert_allclose([panel.coordinate for panel in panels], np.linspace(-0.2, 1.4, 9), atol=1.e-14)
    assert panels[1].coordinate == 0.0
    for index, panel in enumerate(panels):
        np.testing.assert_allclose(panel.values, 2 * index)
        assert panel.upper - panel.lower == pytest.approx(0.2)


def test_qt_tiled_range_preserves_requested_centers_and_export(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(_fine_energy_grid(), x_dim=0, y_dim=1)
    try:
        viewer.view_mode_combo.setCurrentText("Tiled slices")
        viewer.tile_step_auto_check.setChecked(False)
        viewer.tile_step_spin.setValue(0.2)
        viewer.tile_range_low_spin.setValue(-0.2)
        viewer.tile_range_high_spin.setValue(1.4)
        assert viewer.tile_range == pytest.approx((-0.2, 1.4))
        assert viewer.tile_step == pytest.approx(0.2)
        assert viewer.tile_range_low_spin.minimum() == pytest.approx(-0.3)
        settings = viewer.current_plot_settings()
        assert settings["tile_range"] == pytest.approx((-0.2, 1.4))
        assert settings["tile_step"] == pytest.approx(0.2)
        assert len(viewer._tile_axes) == 9
        viewer.apply_plot_settings(settings)
        assert len(viewer._tile_axes) == 9
        assert viewer.tile_range == pytest.approx((-0.2, 1.4))
    finally:
        viewer.window.close()


def test_mdevent_uses_the_same_center_grid():
    from nfit.mdevent import _requested_edges

    edges = _requested_edges(np.array([-0.2]), np.array([1.4]), np.array([9]), step_size=[0.2])
    np.testing.assert_allclose(edges[0], np.linspace(-0.3, 1.5, 10), atol=1.e-14)


def test_nonuniform_edges_display_endpoint_centers_without_changing_edges():
    from nfit.project_gui import _sanitize_rebin_axis_config

    config = _sanitize_rebin_axis_config({"bin_edges": [0., 1., 3.]})
    assert config["lower"] == 0.5
    assert config["upper"] == 2.0
    assert config["bin_edges"] == [0., 1., 3.]
