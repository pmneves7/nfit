import numpy as np
import pytest
from matplotlib.colors import Normalize

from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.plot_recipes import new_plot_entry, render_plot


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer

    axes = tuple(MDHistoAxis(name, np.arange(4.), "", "unknown") for name in ("x", "y", "scan"))
    signal = np.arange(27.).reshape(3, 3, 3) - 10
    data = MDHistoData(axes, signal, np.ones_like(signal), np.zeros_like(signal, dtype=bool), np.ones_like(signal))
    result = QtMDHistoSliceViewer(data, x_dim=0, y_dim=1)
    yield result
    result.window.close()


@pytest.mark.parametrize("mode", ["slice", "tiled_slices"])
def test_restored_color_controls_match_recipe_and_can_select_defaults(viewer, mode):
    viewer.apply_plot_settings({"view_mode": mode, "cmap": "magma_r", "color_scale": "asinh",
                                "auto_limits": "N-sigma", "sigma_n": 2.5, "autoscale": True})
    assert viewer.cmap_combo.currentText() == "magma"
    assert viewer.scale_combo.currentText() == "asinh"
    assert viewer.limits_combo.currentText() == "N-sigma"
    assert viewer.limit_n_spin.value() == pytest.approx(2.5)
    assert viewer.image.get_cmap().name == "magma_r"
    entry = new_plot_entry("restored", "dataset", viewer.current_plot_settings(),
                           plot_type="mdhisto_tiled_slices" if mode == "tiled_slices" else "mdhisto_slice")
    figure = render_plot(entry, viewer.data)
    np.testing.assert_allclose(figure.axes[0].collections[0].norm([-2., 0., 3.]),
                               viewer.image.norm([-2., 0., 3.]))
    from matplotlib import pyplot as plt
    plt.close(figure)

    # These appeared selected before the fix, so choosing them emitted no signal.
    viewer.cmap_combo.setCurrentText("viridis")
    viewer.scale_combo.setCurrentText("linear")
    viewer.limits_combo.setCurrentText("Nth percentile")
    assert viewer.model.cmap == "viridis"
    assert viewer.image.get_cmap().name == "viridis_r"
    assert type(viewer.image.norm) is Normalize
    assert viewer.model.auto_limits == "Nth percentile"
    viewer.cmap_reverse_button.click()
    assert viewer.image.get_cmap().name == "viridis"


@pytest.mark.parametrize("local", [False, True])
def test_color_parameters_roundtrip_with_local_and_manual_limits(viewer, local):
    settings = {"view_mode": "tiled_slices", "cmap": "plasma", "color_scale": "power",
                "auto_limits": "Nth percentile", "percentile_n": 7., "power_gamma": 0.4,
                "sigma_n": 2., "iqr_n": 3., "autoscale": True,
                "tile_local_color_scales": local}
    viewer.apply_plot_settings(settings)
    normalized = viewer.image.norm(np.array([-2., 0., 3.]))
    saved = viewer.current_plot_settings()
    for name in ("sigma_n", "iqr_n", "percentile_n", "power_gamma"):
        assert saved[name] == settings[name]
    viewer.apply_plot_settings({**saved, "power_gamma": 2., "percentile_n": 1.})
    viewer.apply_plot_settings(saved)
    assert viewer.gamma_spin.value() == pytest.approx(0.4)
    assert viewer.limit_n_spin.value() == pytest.approx(7.)
    np.testing.assert_allclose(viewer.image.norm(np.array([-2., 0., 3.])), normalized)
    assert viewer.model.power_gamma == pytest.approx(0.4)
    assert viewer.tile_local_color_scales_check.isChecked() == local
    assert viewer.autoscale_check.isChecked()
    assert "power_gamma=0.4" in viewer.figure_script()
    entry = new_plot_entry("restored", "dataset", saved, plot_type="mdhisto_tiled_slices")
    figure = render_plot(entry, viewer.data)
    np.testing.assert_allclose(figure.axes[0].collections[0].norm([-2., 0., 3.]), normalized)
    from matplotlib import pyplot as plt
    plt.close(figure)

    viewer.apply_plot_settings({**saved, "autoscale": False, "manual_vmin": -5., "manual_vmax": 30.})
    assert not viewer.autoscale_check.isChecked()
    assert not viewer.tile_local_color_scales_check.isChecked()
    assert viewer.image.norm.vmin == pytest.approx(-5.)
    assert viewer.image.norm.vmax == pytest.approx(30.)


@pytest.mark.parametrize("mode", ["slice", "tiled_slices"])
def test_symmetric_color_limits_roundtrip_through_saved_plots(viewer, mode):
    viewer.apply_plot_settings(
        {
            "view_mode": mode,
            "symmetric_about_zero": True,
            "autoscale": True,
        }
    )
    saved = viewer.current_plot_settings()
    entry = new_plot_entry(
        "symmetric",
        "dataset",
        saved,
        plot_type=(
            "mdhisto_tiled_slices" if mode == "tiled_slices" else "mdhisto_slice"
        ),
    )

    figure = render_plot(entry, viewer.data)

    norm = figure.axes[0].collections[0].norm
    assert norm.vmin == pytest.approx(-norm.vmax)
    assert saved["symmetric_about_zero"] is True
    assert viewer.symmetric_about_zero_check.isChecked()
    from matplotlib import pyplot as plt

    plt.close(figure)
