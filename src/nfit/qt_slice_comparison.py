"""Shared comparison-panel presentation and retained-background box profiles."""

from __future__ import annotations

import numpy as np

from .box_cuts import rotated_box_profiles
from .slice_comparison import comparison_profile_view, prepare_background_comparison_models


def draw_comparison_panel(controller, index, title, model, view, norm, label):
    """Render one comparison image while preserving the viewer's drawing hooks."""
    ax = controller._compare_axes[index]
    artist = ax.pcolormesh(
        view["x_edges"], view["y_edges"], model._display_values(view),
        shading="auto", cmap=model._display_cmap(), norm=norm,
    )
    controller._viewer._apply_axes_ratio(ax)
    controller._viewer._draw_gridline_overlay(ax)
    ax.set_title(title)
    ax.set_xlabel(model._axis_label(model.x_dim))
    ax.set_ylabel(model._axis_label(model.y_dim))
    colorbar_axis = (
        controller._compare_colorbar_axes[index]
        if index < len(controller._compare_colorbar_axes) else None
    )
    colorbar = controller.figure.colorbar(
        artist, cax=colorbar_axis, ax=None if colorbar_axis is not None else ax,
    )
    colorbar.set_label(label)
    controller._compare_colorbars.append(colorbar)
    if index == 0:
        controller.image, controller.colorbar = artist, colorbar


def finish_comparison_view(
    controller, previous_xlim, previous_ylim, previous_dims, current_dims,
    *, panel_count, with_cuts, with_difference_cut, vmin, vmax,
):
    """Restore comparison zoom, selection, popout layout, and presentation."""
    if previous_xlim is not None and previous_ylim is not None and previous_dims == current_dims:
        for axis in controller._compare_axes:
            axis.set_xlim(previous_xlim)
            axis.set_ylim(previous_ylim)
    if with_cuts:
        extents = (
            controller._default_roi_extents()
            if controller._roi_extents is None or previous_dims != current_dims
            else controller._roi_extents
        )
        controller._set_roi_extents(extents, update_cuts=True, draw=False)
        if controller.popout_cuts_check.isChecked():
            controller.grid.set_height_ratios([1.0, 0.001])
            controller.grid.set_width_ratios([
                *[value for _ in range(panel_count) for value in (1.0, 0.045)],
                *([0.001] * (1 + int(with_difference_cut))),
            ])
            for axis in (controller.ax_fit_cut, controller.ax_ycut,
                         controller.ax_residual_cut, controller.ax_residual_ycut):
                if axis is not None:
                    axis.set_visible(False)
    controller._sync_control_visibility()
    controller._sync_limit_spinboxes(vmin, vmax)
    controller._last_plot_dims = current_dims
    controller._last_plot_view_mode = "slice"
    controller._sync_view_limit_controls()
    controller._apply_figure_font_size()
    controller._apply_axis_linewidth()
    controller._connect_view_limit_callbacks()
    controller.canvas.draw_idle()


def draw_background_panels(
    controller, previous_xlim, previous_ylim, previous_dims, current_dims,
    *, reuse_prepared=False,
):
    """Compare independently prepared original, background and difference views."""
    channels = ("unsubtracted", "background", "signal")
    with_cuts = controller._comparison_cuts_active()
    controller._ensure_fit_compare_layout(3, with_cuts=with_cuts, with_residual_cut=with_cuts)
    source_key = controller._viewer._slice_source_key()
    cached = getattr(controller, "_background_comparison_views", {})
    can_reuse = (
        reuse_prepared and set(cached) == set(channels)
        and getattr(controller, "_background_comparison_source_key", None) == source_key
        and controller._viewer._slice_sources_are_cacheable()
        and previous_dims == current_dims
        and getattr(controller, "_last_plot_view_mode", None) == "slice"
    )
    models = {
        model.channel: model for model in prepare_background_comparison_models(
            controller.model, diff_color=controller._diff_color_controller.snapshot(),
        )
    }
    views = {}
    for channel in channels:
        model = models[channel]
        views[channel] = (
            cached[channel] if can_reuse
            else controller._smoothed_slice_view(model.slice_arrays(), channel=channel)
        )
    finite = [np.asarray(models[channel]._display_values(views[channel]), dtype=float).ravel()
              for channel in channels[:2]]
    values = np.concatenate(finite)
    values = values[np.isfinite(values)]
    shared_norm = models["unsubtracted"]._color_norm(values)
    vmin, vmax = models["unsubtracted"]._color_limits(values)
    difference_values = models["signal"]._display_values(views["signal"])
    difference_norm = models["signal"]._color_norm(difference_values)
    difference_limits = models["signal"]._color_limits(difference_values)
    controller._background_comparison_views = views
    controller._background_comparison_source_key = source_key
    controller._current_slice = views["signal"]
    controller._current_slice_source_key = source_key
    controller.image = controller.colorbar = None
    for index, (title, channel) in enumerate(zip(
        ("Unsubtracted", "Background", "Difference"), channels, strict=True,
    )):
        draw_comparison_panel(
            controller, index, title, models[channel], views[channel],
            difference_norm if channel == "signal" else shared_norm,
            models[channel]._channel_label(),
        )
    controller._diff_color_controller.sync_limits(*difference_limits)
    finish_comparison_view(
        controller, previous_xlim, previous_ylim, previous_dims, current_dims,
        panel_count=3, with_cuts=with_cuts, with_difference_cut=with_cuts,
        vmin=vmin, vmax=vmax,
    )


def update_background_compare_cuts(controller, extents):
    """Overlay original/background profiles and retain a separate physical difference."""
    if controller.ax_fit_cut is None or controller._current_slice is None or extents is None:
        return
    views = controller._background_comparison_views
    rotated = not np.isclose(controller._roi_angle % 360, 0.0, atol=1e-10)
    profile_function = (
        rotated_box_profiles if rotated else controller._helper("histogram_box_profiles")
    )
    profiles = {}
    for channel in ("unsubtracted", "background", "signal"):
        view = comparison_profile_view(views[channel], channel)
        profiles[channel] = profile_function(
            view, np.asarray(view[channel], dtype=float),
            np.asarray(view["errors"], dtype=float), extents,
            *([controller._roi_angle] if rotated else []),
            coverage_threshold=controller.coverage_threshold, channel=channel,
        )
    difference = profiles["signal"]
    controller._current_box_profiles = difference
    controller._current_x_cut = difference.x if len(difference.x[0]) else None
    controller._current_y_cut = difference.y if len(difference.y[0]) else None
    for axis in (controller.ax_fit_cut, controller.ax_ycut,
                 controller.ax_residual_cut, controller.ax_residual_ycut):
        if axis is not None:
            axis.clear()
    controller._clear_roi_sum_annotation()
    if np.any(difference.selected):
        controller._show_roi_sum_annotation(
            views["signal"]["signal"], views["signal"]["errors"], extents,
            selected=difference.selected,
        )
    for channel, label, color in (
        ("unsubtracted", "Unsubtracted", controller.line_color),
        ("background", "Background", controller.fit_line_color),
    ):
        profile = profiles[channel]
        x_axis, y_axis = controller.ax_fit_cut, controller.ax_ycut
        style = dict(
            marker="o", linestyle="None", ms=controller.marker_size,
            mfc=controller.marker_face_color or "none", mec=color,
            color=color, ecolor=color, elinewidth=controller.line_plot_width,
            label=label,
        )
        if x_axis is not None:
            x_axis.errorbar(profile.x[0], profile.x[1], yerr=profile.x[2], **style)
        if y_axis is not None:
            y_axis.errorbar(profile.y[1], profile.y[0], xerr=profile.y[2], **style)
    x_label = controller.model._axis_label(controller.model.x_dim)
    y_label = controller.model._axis_label(controller.model.y_dim)
    if rotated:
        x_label, y_label = f"Box x · {x_label}", f"Box y · {y_label}"
    controller.ax_fit_cut.set(xlabel=x_label, ylabel=profiles["unsubtracted"].value_label)
    controller.ax_fit_cut.legend()
    if controller.ax_ycut is not None:
        controller.ax_ycut.set(xlabel="I", ylabel="")
        controller.ax_ycut.tick_params(labelleft=False)
    render_background_difference_profiles(controller, difference)
    controller._sync_export_controls()


def render_background_difference_profiles(controller, profiles):
    """Replace only difference profiles after exact background covariance replay."""
    controller._current_box_profiles = profiles
    controller._current_x_cut = profiles.x if len(profiles.x[0]) else None
    controller._current_y_cut = profiles.y if len(profiles.y[0]) else None
    style = dict(
        marker="o", linestyle="None", ms=controller.marker_size,
        mfc=controller.marker_face_color or "none", mec=controller.line_color,
        color=controller.line_color, ecolor=controller.line_color,
        elinewidth=controller.line_plot_width, label="Difference",
    )
    x_label = controller.model._axis_label(controller.model.x_dim)
    y_label = controller.model._axis_label(controller.model.y_dim)
    if not np.isclose(controller._roi_angle % 360, 0.0, atol=1e-10):
        x_label, y_label = f"Box x · {x_label}", f"Box y · {y_label}"
    label = f"Difference\n{profiles.value_label}"
    if controller.ax_residual_cut is not None:
        controller.ax_residual_cut.clear()
        controller.ax_residual_cut.errorbar(
            profiles.x[0], profiles.x[1], yerr=profiles.x[2], **style,
        )
        controller.ax_residual_cut.axhline(0.0, color="0.5", lw=1.0, zorder=1)
        controller.ax_residual_cut.set(xlabel=x_label, ylabel=label)
    if controller.ax_residual_ycut is not None:
        controller.ax_residual_ycut.clear()
        controller.ax_residual_ycut.errorbar(
            profiles.y[1], profiles.y[0], xerr=profiles.y[2], **style,
        )
        controller.ax_residual_ycut.axvline(0.0, color="0.5", lw=1.0, zorder=1)
        controller.ax_residual_ycut.set(xlabel="ΔI", ylabel="")
        controller.ax_residual_ycut.tick_params(labelleft=False)
