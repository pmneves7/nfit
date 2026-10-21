"""Backend-only background comparison preparation and plotting."""

from __future__ import annotations

from typing import Any

import numpy as np

from .axes_ratio import mdhisto_axes_aspect
from .background_channels import available_background_channels
from .mdhisto import MDHistoData
from .plotting_core import (
    MDHistoSliceViewer,
    _apply_axes_linewidth,
    _draw_major_gridlines,
    _draw_mdhisto_roi_cuts,
    _normalize_mdhisto_cmap_name,
    _panel_ratio,
    _validate_gridline_modes,
    align_mdhisto_cut_axes,
    coarsen_mdhisto_view,
    draw_mdhisto_brillouin_zones,
    smooth_mdhisto_view,
)


def default_difference_colors() -> dict[str, Any]:
    """Return independent color defaults for a background-subtracted panel."""

    return {
        "cmap": "bwr",
        "color_scale": "linear",
        "cmap_reversed": False,
        "symmetric_about_zero": True,
        "auto_limits": "min/max",
        "sigma_n": 3.0,
        "iqr_n": 1.5,
        "percentile_n": 1.0,
        "power_gamma": 0.5,
        "color_alpha": 0.0,
        "autoscale": True,
        "manual_vmin": None,
        "manual_vmax": None,
    }


def apply_difference_colors(
    model: MDHistoSliceViewer, settings: dict[str, Any] | None = None,
) -> None:
    """Validate and apply a partial difference color configuration atomically.

    Omitted fields use :func:`default_difference_colors`, independently of the
    standard signal color controls. The accepted keys are model attribute names.
    """

    colors = default_difference_colors()
    unknown = set(settings or {}) - colors.keys()
    if unknown:
        raise ValueError(f"unknown difference color settings: {sorted(unknown)}")
    colors.update(settings or {})
    if colors["color_scale"] not in MDHistoSliceViewer.COLOR_SCALES:
        raise ValueError("unknown difference color scale")
    if colors["auto_limits"] not in MDHistoSliceViewer.AUTO_LIMITS:
        raise ValueError("unknown difference automatic color limits")
    for name in ("sigma_n", "iqr_n", "percentile_n", "power_gamma", "color_alpha"):
        colors[name] = float(colors[name])
        if not np.isfinite(colors[name]):
            raise ValueError(f"difference {name} must be finite")
    if colors["sigma_n"] < 0 or colors["iqr_n"] < 0:
        raise ValueError("difference sigma and IQR factors must be nonnegative")
    if not 0 <= colors["percentile_n"] <= 50:
        raise ValueError("difference percentile must be between 0 and 50")
    if colors["power_gamma"] <= 0:
        raise ValueError("difference power gamma must be positive")
    if not -20 <= colors["color_alpha"] <= 20:
        raise ValueError("difference color alpha must be between -20 and 20")
    for name in ("manual_vmin", "manual_vmax"):
        if colors[name] is not None:
            colors[name] = float(colors[name])
            if not np.isfinite(colors[name]):
                raise ValueError(f"difference {name} must be finite")
    for name in ("cmap_reversed", "symmetric_about_zero", "autoscale"):
        colors[name] = bool(colors[name])
    colors["cmap"] = _normalize_mdhisto_cmap_name(colors["cmap"])
    # Include the viewer's canonical names and other registered maps. Reversal
    # remains an independent control, as in MDHistoSliceViewer._set_cmap.
    from matplotlib import colormaps

    if colors["cmap"] not in MDHistoSliceViewer.COLORMAPS and colors["cmap"] not in colormaps:
        raise ValueError(f"unknown difference colormap {colors['cmap']!r}")
    for name, value in colors.items():
        setattr(model, name, value)


def prepare_background_comparison_models(
    base: MDHistoSliceViewer, *, diff_color: dict[str, Any] | None = None,
) -> tuple[MDHistoSliceViewer, MDHistoSliceViewer, MDHistoSliceViewer]:
    """Clone one slice configuration into unsubtracted, background, difference.

    The data and numerical payloads remain shared and immutable; selections and
    color state belong to the returned models. No Qt widgets are constructed.
    """

    if not isinstance(base.data, MDHistoData) or not {
        "background", "unsubtracted",
    }.issubset(available_background_channels(base.data)):
        raise ValueError("background comparison requires retained background channels")
    models = []
    for channel in ("unsubtracted", "background", "signal"):
        model = MDHistoSliceViewer(
            base.data, x_dim=base.x_dim, y_dim=base.y_dim, channel=channel,
            masked=base.masked, integrate=base.integrate,
            coverage_threshold=base.coverage_threshold,
        )
        model.selections.update(base.selections)
        model.integrate_checks.update(base.integrate_checks)
        for name in default_difference_colors():
            setattr(model, name, getattr(base, name))
        models.append(model)
    apply_difference_colors(models[2], diff_color)
    return tuple(models)


def prepare_background_comparison_view(
    model: MDHistoSliceViewer,
    *,
    x_step: float | None = None,
    y_step: float | None = None,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_y: float = 0.0,
    smoothing_fill_nans: bool = True,
) -> dict[str, Any]:
    """Prepare one comparison channel with its matching display uncertainty."""

    coarse = coarsen_mdhisto_view(model.slice_arrays(), x_step=x_step, y_step=y_step)
    view = smooth_mdhisto_view(
        coarse, sigma_x=smoothing_sigma_x, sigma_y=smoothing_sigma_y,
        fill_nans=smoothing_fill_nans, channel=model.channel,
    )
    return view


def comparison_profile_view(view: dict[str, Any], channel: str) -> dict[str, Any]:
    """Isolate retained diagnostic errors and support from the primary target.

    The Difference retains its exact measurement contract, additive statistics,
    and source sensitivities. Original/background channels have only their own
    displayed values and diagonal uncertainties; the primary payload cannot be
    used to estimate these independent diagnostic profiles. In particular, an
    original pixel restored after failed subtraction remains valid here.
    """
    if channel == "signal":
        return view
    if channel not in {"unsubtracted", "background"}:
        raise ValueError("comparison profiles require a retained background channel")
    values = np.asarray(view[channel], dtype=float)
    errors = np.asarray(view[f"{channel}_errors"], dtype=float)
    result = {name: view[name] for name in (
        "x_edges", "y_edges", "x_centers", "y_centers", "coverage_fraction",
        "signal_unit", "masks_applied",
    ) if name in view}
    result.update({
        "signal": values, "errors": errors,
        channel: values, f"{channel}_errors": errors,
        "mask": ~np.isfinite(values) | ~np.isfinite(errors),
    })
    return result


def plot_mdhisto_background_comparison(
    data: MDHistoData,
    *,
    x_dim: int | str = -1,
    y_dim: int | str = 0,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    coverage_threshold: float = 0.0,
    masked: bool = True,
    cmap: str = "viridis",
    color_scale: str = "linear",
    auto_limits: str = "min/max",
    symmetric_about_zero: bool = False,
    autoscale: bool = True,
    manual_vmin: float | None = None,
    manual_vmax: float | None = None,
    sigma_n: float = 3.0,
    iqr_n: float = 1.5,
    percentile_n: float = 1.0,
    power_gamma: float = 0.5,
    color_alpha: float = 0.0,
    diff_color: dict[str, Any] | None = None,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_y: float = 0.0,
    smoothing_fill_nans: bool = True,
    x_step: float | None = None,
    y_step: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 10.0,
    show_histogram_axes: bool = False,
    roi_extents: tuple[float, float, float, float] | None = None,
    roi_angle: float = 0.0,
    background_uncertainty: str = "diagonal",
    box_tool: dict[str, Any] | None = None,
    xcut_percent: float = 20.0,
    ycut_percent: float = 16.0,
    axes_linewidth: float = 1.0,
    axes_ratio: str = "fit",
    show_brillouin_zone_boundaries: bool = False,
    show_major_gridlines: bool = False,
    brillouin_zone_spacegroup: str | None = None,
    brillouin_zone_lattice_parameters: dict[str, float] | None = None,
    brillouin_zone_color: str = "#000000",
    brillouin_zone_linewidth: float = 1.5,
    brillouin_zone_alpha: float = 1.0,
    figsize: tuple[float, float] = (15.0, 6.5),
):
    """Plot unsubtracted data, retained background, and their difference.

    The first two panels share one normalization computed over both channels.
    ``diff_color`` independently configures the difference; its default is a
    linear, symmetric blue-white-red scale. Scientific slice, mask, coarsening,
    smoothing and box-profile options follow ``plot_mdhisto_slice``.
    ``channel`` must be ``'signal'`` because the difference is the saved signal.

    ``box_tool`` may supply ``extents``, ``angle``, ``show_histogram_axes`` and
    ``background_uncertainty`` as an alternative to the explicit box arguments.
    Replay covariance applies to the difference profiles; the other profiles
    use their retained channel uncertainties.
    """

    import matplotlib.pyplot as plt

    if channel != "signal":
        raise ValueError("background comparison requires the signal channel")
    box = dict(box_tool or {})
    unknown_box = set(box) - {"extents", "angle", "show_histogram_axes", "background_uncertainty"}
    if unknown_box:
        raise ValueError(f"unknown comparison box settings: {sorted(unknown_box)}")
    roi_extents = box.get("extents", roi_extents)
    roi_angle = float(box.get("angle", roi_angle))
    show_histogram_axes = bool(box.get("show_histogram_axes", show_histogram_axes))
    background_uncertainty = box.get("background_uncertainty", background_uncertainty)
    if background_uncertainty not in {"diagonal", "replay"}:
        raise ValueError("Background uncertainty must be 'diagonal' or 'replay'")
    if background_uncertainty == "replay" and (smoothing_sigma_x != 0 or smoothing_sigma_y != 0):
        raise ValueError("Background covariance replay requires an unsmoothed signal cut")
    _validate_gridline_modes(show_brillouin_zone_boundaries, show_major_gridlines)
    base = MDHistoSliceViewer(
        data, x_dim=x_dim, y_dim=y_dim, masked=masked,
        coverage_threshold=coverage_threshold, cmap=cmap, color_scale=color_scale,
        auto_limits=auto_limits, symmetric_about_zero=symmetric_about_zero,
    )
    if selections:
        base.selections.update({int(dim): tuple(value) for dim, value in selections.items()})
    if integrate_checks:
        base.integrate_checks.update({int(dim): bool(value) for dim, value in integrate_checks.items()})
    base.autoscale = bool(autoscale)
    base.manual_vmin, base.manual_vmax = manual_vmin, manual_vmax
    base.sigma_n, base.iqr_n, base.percentile_n = float(sigma_n), float(iqr_n), float(percentile_n)
    base.power_gamma, base.color_alpha = float(power_gamma), float(color_alpha)
    models = prepare_background_comparison_models(base, diff_color=diff_color)
    views = [prepare_background_comparison_view(
        model, x_step=x_step, y_step=y_step,
        smoothing_sigma_x=smoothing_sigma_x, smoothing_sigma_y=smoothing_sigma_y,
        smoothing_fill_nans=smoothing_fill_nans,
    ) for model in models]
    values = [model._display_values(view) for model, view in zip(models, views, strict=True)]
    shared_norm = models[0]._color_norm(np.concatenate([value.ravel() for value in values[:2]]))
    with plt.rc_context({"font.size": float(font_size)}):
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        outer = figure.add_gridspec(1, 3)
        first_image = None
        for index, (model, view, value, title) in enumerate(zip(
            models, views, values, ("Unsubtracted", "Background", "Difference"), strict=True,
        )):
            if show_histogram_axes:
                grid = outer[index].subgridspec(
                    2, 3, width_ratios=[1.0, _panel_ratio(ycut_percent), 0.045],
                    height_ratios=[1.0, _panel_ratio(xcut_percent)],
                )
                image_axis = figure.add_subplot(grid[0, 0], sharex=first_image, sharey=first_image)
                ycut = figure.add_subplot(grid[0, 1], sharey=image_axis)
                color_axis = figure.add_subplot(grid[0, 2])
                xcut = figure.add_subplot(grid[1, 0], sharex=image_axis)
            else:
                grid = outer[index].subgridspec(1, 2, width_ratios=[1.0, 0.045])
                image_axis = figure.add_subplot(grid[0, 0], sharex=first_image, sharey=first_image)
                color_axis = figure.add_subplot(grid[0, 1])
                xcut = ycut = None
            first_image = first_image or image_axis
            image = image_axis.pcolormesh(
                view["x_edges"], view["y_edges"], value, shading="auto",
                cmap=model._display_cmap(), norm=shared_norm if index < 2 else model._color_norm(value),
            )
            image_axis.set_title(title)
            image_axis.set_xlabel(model._axis_label(model.x_dim))
            image_axis.set_ylabel(model._axis_label(model.y_dim))
            image_axis.set_aspect(mdhisto_axes_aspect(
                data, model.x_dim, model.y_dim, axes_ratio,
                lattice_parameters=brillouin_zone_lattice_parameters,
            ), adjustable="box")
            align_mdhisto_cut_axes(image_axis, xcut, ycut)
            if xlim is not None:
                image_axis.set_xlim(*xlim)
            if ylim is not None:
                image_axis.set_ylim(*ylim)
            if show_brillouin_zone_boundaries:
                draw_mdhisto_brillouin_zones(
                    image_axis, data, x_dim=model.x_dim, y_dim=model.y_dim,
                    selections=model.selections, spacegroup=brillouin_zone_spacegroup,
                    lattice_parameters=brillouin_zone_lattice_parameters,
                    color=brillouin_zone_color, linewidth=brillouin_zone_linewidth,
                    alpha=brillouin_zone_alpha,
                )
            elif show_major_gridlines:
                _draw_major_gridlines(image_axis, color=brillouin_zone_color,
                    linewidth=brillouin_zone_linewidth, alpha=brillouin_zone_alpha)
            colorbar = figure.colorbar(image, cax=color_axis)
            colorbar.set_label(model._channel_label())
            if show_histogram_axes and roi_extents is not None:
                profile_view = comparison_profile_view(view, model.channel)
                _draw_mdhisto_roi_cuts(
                    model, profile_view, roi_extents, xcut, ycut,
                    ax_image=image_axis, roi_angle=roi_angle,
                    background_uncertainty=background_uncertainty if index == 2 else "diagonal",
                )
            _apply_axes_linewidth((image_axis, color_axis, xcut, ycut), colorbar, axes_linewidth)
    return figure
