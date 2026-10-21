"""Backend comparison colors, scientific preparation, and saved recipe parity."""

from __future__ import annotations

import ast
import inspect
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize

from nfit.background_channels import accumulate_background_channel
from nfit.box_cuts import histogram_box_profiles
from nfit.plot_recipes import (
    new_plot_entry,
    plot_entry_from_dict,
    plot_entry_to_dict,
    render_plot,
)
from nfit.plotting_core import MDHistoSliceViewer, gaussian_smooth_uncertainty
from nfit.slice_comparison import (
    apply_difference_colors,
    default_difference_colors,
    plot_mdhisto_background_comparison,
    prepare_background_comparison_models,
    prepare_background_comparison_view,
)
from tests.plotting_test_data import tiny_mdhisto_data


def test_public_background_comparison_exports_share_the_backend():
    import nfit
    from nfit import plotting

    assert nfit.plot_mdhisto_background_comparison is plot_mdhisto_background_comparison
    assert plotting.plot_mdhisto_background_comparison is plot_mdhisto_background_comparison
    assert "plot_mdhisto_background_comparison" in plotting.__all__


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data():
    original = tiny_mdhisto_data()
    contribution = np.broadcast_to(np.linspace(1, 10, 5), original.shape)
    uncertainty = np.broadcast_to(np.linspace(0.1, 0.8, 5), original.shape)
    result = original.with_updates(
        signal=original.signal - contribution,
        errors=np.sqrt(original.errors**2 + uncertainty**2),
    )
    return accumulate_background_channel(
        original, result, contribution=contribution,
        contribution_errors=uncertainty, valid=np.ones(original.shape, dtype=bool),
    )


def _image_axes(figure):
    return [axis for axis in figure.axes if axis.get_title() in {
        "Unsubtracted", "Background", "Difference",
    }]


def test_difference_color_defaults_are_fresh_and_match_standard_parameters():
    colors = default_difference_colors()
    colors["cmap"] = "inferno"
    defaults = default_difference_colors()
    standard = MDHistoSliceViewer(_data())
    assert defaults["cmap"] == "bwr"
    assert defaults["symmetric_about_zero"] is True
    for name in defaults.keys() - {"cmap", "symmetric_about_zero"}:
        assert defaults[name] == getattr(standard, name)


def test_difference_colors_validate_before_modifying_and_restore_partial_state():
    model = MDHistoSliceViewer(_data())
    apply_difference_colors(model, {"cmap": "magma_r", "cmap_reversed": True, "manual_vmin": -4, "manual_vmax": 4, "autoscale": False})
    assert model.cmap == "magma" and model.cmap_reversed
    assert model._color_limits(np.array([-1, 1])) == (-4, 4)
    before = {name: getattr(model, name) for name in default_difference_colors()}
    with pytest.raises(ValueError, match="positive"):
        apply_difference_colors(model, {"cmap": "viridis", "power_gamma": 0})
    assert before == {name: getattr(model, name) for name in before}
    apply_difference_colors(model, {"color_alpha": 2})
    assert model.cmap == "bwr" and not model.cmap_reversed


@pytest.mark.parametrize("setting, values", [
    ("color_scale", MDHistoSliceViewer.COLOR_SCALES),
    ("auto_limits", MDHistoSliceViewer.AUTO_LIMITS),
])
def test_difference_color_options_follow_model_choices(setting, values):
    model = MDHistoSliceViewer(_data())
    for value in values:
        apply_difference_colors(model, {setting: value})
        assert getattr(model, setting) == value
    for value in ("unsupported", "3-sigma", "1.5 IQR"):
        with pytest.raises(ValueError, match="unknown difference"):
            apply_difference_colors(model, {setting: value})


@pytest.mark.parametrize("cmap", ["gray", "gray_r", "magma_r", "bwr"])
def test_difference_colormap_name_matches_model_setter(cmap):
    model = MDHistoSliceViewer(_data())
    expected = MDHistoSliceViewer(_data())
    expected._set_cmap(cmap)
    apply_difference_colors(model, {"cmap": cmap})
    assert model.cmap == expected.cmap
    assert model.cmap_reversed == expected.cmap_reversed


def test_background_models_preserve_slice_selection_and_independent_colors():
    data = _data()
    base = MDHistoSliceViewer(data, x_dim=3, y_dim=2, masked=False, cmap="magma_r", color_scale="log")
    base.selections.update({0: (0.25, 0.75), 1: (0.5, 1.5)})
    base.integrate_checks.update({0: True, 1: True})
    models = prepare_background_comparison_models(base)
    assert [model.channel for model in models] == ["unsubtracted", "background", "signal"]
    for model in models:
        assert model.data is data
        assert model.selections == base.selections
        assert model.integrate_checks == base.integrate_checks
        assert not model.masked
    assert models[0].cmap == models[1].cmap == "magma"
    assert models[0].cmap_reversed and models[1].cmap_reversed
    assert models[2].cmap == "bwr" and models[2].color_scale == "linear"
    models[0].selections[0] = (9, 9)
    assert models[1].selections == base.selections


def test_public_comparison_shares_standard_normalization_and_defaults_difference():
    figure = plot_mdhisto_background_comparison(_data(), x_dim=3, y_dim=2, color_scale="log")
    axes = _image_axes(figure)
    assert [axis.get_title() for axis in axes] == ["Unsubtracted", "Background", "Difference"]
    artists = [axis.collections[0] for axis in axes]
    assert artists[0].norm is artists[1].norm
    assert isinstance(artists[0].norm, LogNorm)
    assert type(artists[2].norm) is Normalize
    assert artists[2].cmap.name == "bwr"
    assert artists[2].norm.vmin == -artists[2].norm.vmax


def test_comparison_preparation_uses_channel_uncertainty_for_smoothing():
    model = prepare_background_comparison_models(MDHistoSliceViewer(_data(), x_dim=3, y_dim=2))[1]
    raw = model.slice_arrays()
    prepared = prepare_background_comparison_view(model, smoothing_sigma_x=0.8, smoothing_sigma_y=0.3)
    np.testing.assert_allclose(
        prepared["background_errors"],
        gaussian_smooth_uncertainty(raw["background_errors"], (0.3, 0.8)),
    )


@pytest.mark.parametrize("angle", [0, 25])
def test_public_comparison_draws_matching_channel_box_profiles(angle):
    data = _data()
    extents = (-1.8, 1.8, -0.9, 0.9)
    figure = plot_mdhisto_background_comparison(
        data, x_dim=3, y_dim=2, show_histogram_axes=True,
        box_tool={"extents": extents, "angle": angle},
    )
    models = prepare_background_comparison_models(MDHistoSliceViewer(data, x_dim=3, y_dim=2))
    for index, model in enumerate(models):
        view = prepare_background_comparison_view(model)
        error_key = "errors" if model.channel == "signal" else f"{model.channel}_errors"
        expected = histogram_box_profiles(
            view, model._display_values(view), view[error_key], extents, angle,
            channel=model.channel,
        )
        # Each panel creates image, y profile, colorbar, then x profile axes.
        line = figure.axes[index * 4 + 3].lines[0]
        np.testing.assert_allclose(line.get_xdata(), expected.x[0])
        np.testing.assert_allclose(line.get_ydata(), expected.x[1], equal_nan=True)


def test_saved_recipe_roundtrip_matches_public_scientific_configuration():
    data = _data().mutable_copy()
    data.mask[1, 1, 1, 1] = True
    data = data.immutable_copy()
    selections = {0: (0.25, 0.75), 1: (0.5, 1.5)}
    checks = {0: True, 1: True}
    diff = {"cmap": "coolwarm", "autoscale": False, "manual_vmin": -25, "manual_vmax": 25}
    settings = {
        "show_background": True, "diff_color": diff, "x_dim": 3, "y_dim": 2,
        "selections": selections, "integrate_checks": checks,
        "apply_masks": False, "cmap": "magma_r", "color_scale": "log",
        "smoothing_x": 0.4, "smoothing_y": 0.2, "smoothing_fill_nans": False,
        "x_step": 1.6, "y_step": 1.0, "xlim": [-1, 1], "ylim": [-0.8, 0.8],
    }
    entry = new_plot_entry("Background", "dataset", settings, plot_type="mdhisto_slice")
    restored = plot_entry_from_dict(json.loads(json.dumps(plot_entry_to_dict(entry))))
    actual = render_plot(restored, data)
    expected = plot_mdhisto_background_comparison(
        data, x_dim=3, y_dim=2, selections=selections, integrate_checks=checks,
        masked=False, cmap="magma_r", color_scale="log", diff_color=diff,
        smoothing_sigma_x=0.4, smoothing_sigma_y=0.2, smoothing_fill_nans=False,
        x_step=1.6, y_step=1.0, xlim=(-1, 1), ylim=(-0.8, 0.8),
    )
    for actual_axis, expected_axis in zip(_image_axes(actual), _image_axes(expected), strict=True):
        actual_image, expected_image = actual_axis.collections[0], expected_axis.collections[0]
        np.testing.assert_allclose(actual_image.get_array(), expected_image.get_array())
        assert actual_image.cmap.name == expected_image.cmap.name
        assert actual_image.norm.vmin == expected_image.norm.vmin
        assert actual_image.norm.vmax == expected_image.norm.vmax
        assert actual_axis.get_xlim() == expected_axis.get_xlim()
        assert actual_axis.get_ylim() == expected_axis.get_ylim()


def test_background_comparison_requires_retained_channels_and_signal():
    with pytest.raises(ValueError, match="retained background"):
        plot_mdhisto_background_comparison(tiny_mdhisto_data())
    with pytest.raises(ValueError, match="signal channel"):
        plot_mdhisto_background_comparison(_data(), channel="errors")


def test_background_comparison_service_has_no_qt_imports():
    import nfit.slice_comparison as service

    tree = ast.parse(inspect.getsource(service))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all("qt" not in module.lower() and "project_gui" not in module for module in modules if module)
