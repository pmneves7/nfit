from __future__ import annotations

from typing import Any

import numpy as np

from .mdhisto import MDHistoData
from .plotting_core import (
    MDHistoSliceViewer,
    default_tiled_slice_step,
    default_waterfall_offset,
    default_waterfall_step,
    draw_waterfall_traces,
    inverse_variance_weighted_profile,
    prepare_mdhisto_tiled_slices,
    prepare_mdhisto_waterfall,
    waterfall_colors,
)

_DEFAULT_HELPERS = {
    "default_tiled_slice_step": default_tiled_slice_step,
    "default_waterfall_offset": default_waterfall_offset,
    "default_waterfall_step": default_waterfall_step,
    "draw_waterfall_traces": draw_waterfall_traces,
    "inverse_variance_weighted_profile": inverse_variance_weighted_profile,
    "prepare_mdhisto_tiled_slices": prepare_mdhisto_tiled_slices,
    "prepare_mdhisto_waterfall": prepare_mdhisto_waterfall,
    "waterfall_colors": waterfall_colors,
}


class _ViewerController:
    """Forward shared viewer state while owning one rendering responsibility."""

    __slots__ = ("_helpers", "_viewer")

    def __init__(self, viewer: Any, helpers: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_viewer", viewer)
        object.__setattr__(self, "_helpers", helpers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._viewer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_helpers", "_viewer"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._viewer, name, value)

    def _helper(self, name: str) -> Any:
        """Resolve a former ``qt_slice_viewer`` global at call time.

        The live namespace keeps established monkeypatch and extension seams
        working after the rendering implementations move into this module.
        Standalone controller use falls back to this module's implementation.
        """

        if self._helpers is not None and name in self._helpers:
            return self._helpers[name]
        return _DEFAULT_HELPERS[name]


class PlotLayoutController(_ViewerController):
    def _ensure_standard_plot_layout(self) -> None:
        if self._plot_layout_mode in (None, ("standard", 1)):
            self._plot_layout_mode = ("standard", 1)
            self._suppress_matplotlib_coordinate_status()
            return
        self.figure.clear()
        self.grid = self.figure.add_gridspec(
            2,
            3,
            width_ratios=[1.0, self._panel_ratio(self.ycut_percent), 0.045],
            height_ratios=[1.0, self._panel_ratio(self.xcut_percent)],
        )
        self.ax_image = self.figure.add_subplot(self.grid[0, 0])
        self.ax_ycut = self.figure.add_subplot(self.grid[0, 1], sharey=self.ax_image)
        self.ax_colorbar = self.figure.add_subplot(self.grid[0, 2])
        self.ax_xcut = self.figure.add_subplot(self.grid[1, 0], sharex=self.ax_image)
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self._suppress_matplotlib_coordinate_status()
        self.image = None
        self.colorbar = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._tile_axes = []
        self._tile_colorbar_axes = []
        self._tile_colorbars = []
        self._plot_layout_mode = ("standard", 1)
        self._create_rectangle_selector()

    def _ensure_waterfall_layout(self) -> None:
        if self._plot_layout_mode == ("waterfall", 1):
            self.ax_image.clear()
            return
        self.figure.clear()
        self.grid = self.figure.add_gridspec(1, 1)
        self.ax_image = self.figure.add_subplot(self.grid[0, 0])
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self.image = None
        self.colorbar = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._compare_colorbar_axes = []
        self._tile_axes = []
        self._tile_colorbar_axes = []
        self._tile_colorbars = []
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("waterfall", 1)
        self._suppress_matplotlib_coordinate_status()

    def _ensure_tiled_layout(
        self,
        panel_count: int,
        *,
        local_color_scales: bool,
    ) -> None:
        self.figure.clear()
        columns = int(np.ceil(np.sqrt(panel_count)))
        rows = int(np.ceil(panel_count / columns))
        if local_color_scales:
            self.grid = self.figure.add_gridspec(
                rows,
                columns * 2,
                width_ratios=[
                    value
                    for _ in range(columns)
                    for value in (1.0, 0.055)
                ],
            )
        else:
            self.grid = self.figure.add_gridspec(
                rows,
                columns + 1,
                width_ratios=[*[1.0] * columns, 0.055],
            )
        self._tile_axes = []
        self._tile_colorbar_axes = []
        for index in range(panel_count):
            row, column = divmod(index, columns)
            plot_column = column * 2 if local_color_scales else column
            first = self._tile_axes[0] if self._tile_axes else None
            self._tile_axes.append(
                self.figure.add_subplot(
                    self.grid[row, plot_column],
                    sharex=first,
                    sharey=first,
                )
            )
            if local_color_scales:
                self._tile_colorbar_axes.append(
                    self.figure.add_subplot(self.grid[row, plot_column + 1])
                )
        self.ax_image = self._tile_axes[0]
        if local_color_scales:
            self.ax_colorbar = self._tile_colorbar_axes[0]
        else:
            self.ax_colorbar = self.figure.add_subplot(self.grid[:, -1])
            self._tile_colorbar_axes = [self.ax_colorbar]
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self.image = None
        self.colorbar = None
        self._tile_colorbars = []
        self._compare_axes = []
        self._compare_colorbars = []
        self._compare_colorbar_axes = []
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = (
            "tiled",
            panel_count,
            bool(local_color_scales),
        )
        self._suppress_matplotlib_coordinate_status()

    def _ensure_fit_compare_layout(
        self,
        panel_count: int,
        *,
        with_cuts: bool = False,
        with_residual_cut: bool = False,
    ) -> None:
        self.figure.clear()
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self.image = None
        self.colorbar = None
        self._compare_colorbars = []
        self._compare_colorbar_axes = []
        self._tile_axes = []
        self._tile_colorbar_axes = []
        self._tile_colorbars = []
        if with_cuts:
            cut_ratio = self._panel_ratio(self.xcut_percent)
            ycut_ratio = self._panel_ratio(self.ycut_percent)
            width_ratios = []
            for _index in range(panel_count):
                width_ratios.extend([1.0, 0.045])
            width_ratios.append(ycut_ratio)
            if with_residual_cut:
                width_ratios.append(ycut_ratio)
            outer = self.figure.add_gridspec(
                2,
                panel_count * 2 + 1 + int(with_residual_cut),
                width_ratios=width_ratios,
                height_ratios=[1.0, cut_ratio],
            )
            self._compare_axes = [
                self.figure.add_subplot(outer[0, index * 2]) for index in range(panel_count)
            ]
            self._compare_colorbar_axes = [
                self.figure.add_subplot(outer[0, index * 2 + 1]) for index in range(panel_count)
            ]
            self.ax_ycut = self.figure.add_subplot(outer[0, panel_count * 2], sharey=self._compare_axes[0])
            self.ax_fit_cut = self.figure.add_subplot(outer[1, 0], sharex=self._compare_axes[0])
            if with_residual_cut:
                self.ax_residual_cut = self.figure.add_subplot(
                    outer[1, 2 * (panel_count - 1)], sharex=self._compare_axes[-1]
                )
                self.ax_residual_ycut = self.figure.add_subplot(
                    outer[0, panel_count * 2 + 1], sharey=self._compare_axes[-1]
                )
            self.grid = outer
        else:
            self._compare_colorbar_axes = []
            self.grid = self.figure.add_gridspec(1, panel_count)
            self._compare_axes = [
                self.figure.add_subplot(self.grid[0, index]) for index in range(panel_count)
            ]
        self.ax_image = self._compare_axes[0]
        self._suppress_matplotlib_coordinate_status()
        if with_cuts:
            self._create_rectangle_selector()
        elif self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("fit_compare", panel_count, with_cuts, with_residual_cut)

    def _ensure_residual_1d_layout(self) -> None:
        if self._plot_layout_mode == ("residual_1d", 2):
            self.grid.set_height_ratios([1.0, self._panel_ratio(self.residual_percent)])
            self.ax_image.clear()
            self.ax_residual.clear()
            return
        self.figure.clear()
        self.grid = self.figure.add_gridspec(
            2,
            1,
            height_ratios=[1.0, self._panel_ratio(self.residual_percent)],
        )
        self.ax_image = self.figure.add_subplot(self.grid[0, 0])
        self.ax_residual = self.figure.add_subplot(self.grid[1, 0], sharex=self.ax_image)
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.ax_residual_ycut = None
        self.image = None
        self.colorbar = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._tile_axes = []
        self._tile_colorbar_axes = []
        self._tile_colorbars = []
        self._suppress_matplotlib_coordinate_status()
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("residual_1d", 2)


class TiledSliceController(_ViewerController):
    def _draw_tiled_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        """Draw coarse third-axis bins as linked 2D panels."""

        if self.tile_dim is None:
            return
        if self.tile_step_auto:
            self.tile_step = self._helper("default_tiled_slice_step")(
                self.data,
                self.tile_dim,
                self.tile_range,
            )
        self._sync_tile_step_slider()
        slices = self._helper("prepare_mdhisto_tiled_slices")(
            self.data,
            x_dim=self.model.x_dim,
            y_dim=self.model.y_dim,
            tile_dim=self.tile_dim,
            channel=self.model.channel,
            selections=self.model.selections,
            integrate_checks=self.model.integrate_checks,
            tile_range=self.tile_range,
            tile_step=self._effective_tile_step(),
            tile_label_decimals=self.tile_label_decimals,
            tile_label_prefix=self.tile_label_prefix,
            tile_label_unit=self.tile_label_unit,
            tile_label_si_prefix=self.tile_label_si_prefix,
            coverage_threshold=self.coverage_threshold,
            masked=self.model.masked,
            smoothing_sigma_x=self.smoothing_x,
            smoothing_sigma_y=self.smoothing_y,
            smoothing_fill_nans=self.smoothing_fill_nans,
        )
        self._current_tiled_slices = slices
        self._current_slice = slices[0].view
        combined = np.concatenate([panel.values.ravel() for panel in slices])
        shared_norm = self.model._color_norm(combined)
        vmin, vmax = self.model._color_limits(combined)
        local_color_scales = bool(
            self.tile_local_color_scales and self.model.autoscale
        )
        self._ensure_tiled_layout(
            len(slices),
            local_color_scales=local_color_scales,
        )
        columns = int(np.ceil(np.sqrt(len(slices))))
        rows = int(np.ceil(len(slices) / columns))
        artists = []
        for index, (axis, panel) in enumerate(
            zip(self._tile_axes, slices, strict=True)
        ):
            row, column = divmod(index, columns)
            norm = (
                self.model._color_norm(panel.values)
                if local_color_scales
                else shared_norm
            )
            artist = axis.pcolormesh(
                panel.view["x_edges"],
                panel.view["y_edges"],
                panel.values,
                shading="auto",
                cmap=self.model._display_cmap(),
                norm=norm,
            )
            self._viewer._draw_brillouin_zone_overlay(
                axis, coordinate_overrides={self.tile_dim: panel.coordinate}
            )
            if self.show_tile_labels:
                axis.text(
                    0.97,
                    0.03,
                    panel.label,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=self.font_size,
                    bbox={
                        "boxstyle": "square,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                )
            if row == rows - 1 or index + columns >= len(slices):
                axis.set_xlabel(self.model._axis_label(self.model.x_dim))
            if column == 0:
                axis.set_ylabel(self.model._axis_label(self.model.y_dim))
            artists.append(artist)
            if local_color_scales:
                colorbar = self.figure.colorbar(
                    artist,
                    cax=self._tile_colorbar_axes[index],
                )
                if column == columns - 1:
                    colorbar.set_label(self.model._channel_label())
                self._tile_colorbars.append(colorbar)
        self.image = artists[0]
        if local_color_scales:
            self.colorbar = self._tile_colorbars[0]
        else:
            self.colorbar = self.figure.colorbar(artists[-1], cax=self.ax_colorbar)
            self.colorbar.set_label(self.model._channel_label())
            self._tile_colorbars = [self.colorbar]
        preserve_limits = (
            previous_xlim is not None
            and previous_ylim is not None
            and previous_dims == current_dims
            and getattr(self, "_last_plot_view_mode", None) == "tiled"
        )
        if preserve_limits:
            self.ax_image.set_xlim(previous_xlim)
            self.ax_image.set_ylim(previous_ylim)
        self._sync_control_visibility()
        self._sync_limit_spinboxes(vmin, vmax)
        self._last_plot_dims = current_dims
        self._last_plot_view_mode = "tiled"
        self._sync_view_limit_controls()
        self._apply_figure_font_size()
        self._apply_axis_linewidth()
        self._connect_view_limit_callbacks()
        self.canvas.draw_idle()


class WaterfallController(_ViewerController):
    def _draw_waterfall_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        """Draw offset traces for the current map or compatible 1D datasets."""

        self._ensure_waterfall_layout()
        grouped = self._viewer._waterfall_uses_1d_group()
        if grouped:
            indices = self._viewer._waterfall_1d_source_indices()
            datasets = [self.datasets[index] for index in indices]
            labels = [self.dataset_names[index] for index in indices]
            x_dim = next(
                index for index, size in enumerate(self.data.shape) if size > 1
            )
            traces = self._helper("prepare_mdhisto_waterfall")(
                datasets,
                dataset_labels=labels,
                x_dim=self.data.axes[x_dim].name,
                channel=self.model.channel,
                masked=self.model.masked,
                coverage_threshold=self.waterfall_coverage_threshold,
                smoothing_sigma_x=self.smoothing_x,
                smoothing_fill_nans=self.smoothing_fill_nans,
                include_model=self.show_fit,
                unmask_model=self.unmask_model,
            )
        else:
            step_low, step_high = self._waterfall_step_limits()
            if self.waterfall_step_auto:
                self.waterfall_step = self._helper("default_waterfall_step")(
                    self.data,
                    self.model.y_dim,
                )
            self.waterfall_step = float(
                np.clip(self.waterfall_step, step_low, step_high)
            )
            self._sync_waterfall_step_slider()
            self._set_spin_silent(
                self.waterfall_step_spin,
                self.waterfall_step,
            )
            traces = self._helper("prepare_mdhisto_waterfall")(
                self.data,
                x_dim=self.model.x_dim,
                waterfall_dim=self.model.y_dim,
                channel=self.model.channel,
                selections=self._export_selections(),
                integrate_checks=self._export_integrate_checks(),
                waterfall_step=self.waterfall_step,
                coverage_threshold=self.waterfall_coverage_threshold,
                masked=self.model.masked,
                smoothing_sigma_x=self.smoothing_x,
                smoothing_sigma_waterfall=self.smoothing_y,
                smoothing_fill_nans=self.smoothing_fill_nans,
                include_model=self.show_fit,
                unmask_model=self.unmask_model,
            )
        self._current_waterfall_traces = traces
        maximum_offset = self._waterfall_offset_maximum()
        if self.waterfall_offset_auto:
            self.waterfall_offset = self._helper("default_waterfall_offset")(traces)
        self.waterfall_offset = float(
            np.clip(self.waterfall_offset, 0.0, maximum_offset)
        )
        self._sync_waterfall_offset_slider()
        self._set_spin_silent(
            self.waterfall_offset_spin,
            self.waterfall_offset,
        )
        colors = self._helper("waterfall_colors")(
            self.waterfall_cmap,
            len(traces),
            low=self.waterfall_color_min,
            high=self.waterfall_color_max,
            reverse=self.waterfall_reverse_colors,
        )
        self._helper("draw_waterfall_traces")(
            self.ax_image,
            traces,
            colors=colors,
            trace_offset=self.waterfall_offset,
            marker=self.marker,
            line_style=self.line_style,
            marker_size=self.marker_size,
            line_width=self.line_plot_width,
            marker_edge_width=self.marker_edge_width,
            marker_face=self.marker_face_color,
            show_errorbars=self.show_errorbars,
            errorbar_caps=self.show_errorbar_caps,
            errorbar_cap_size=self.errorbar_cap_size,
            show_zero_lines=self.waterfall_show_zero_lines,
            zero_line_color=self.waterfall_zero_color,
            zero_line_style=self.waterfall_zero_style,
            zero_line_width=self.waterfall_zero_width,
            show_model=self.show_fit,
            model_color=self.waterfall_model_color,
            model_line_width=self.fit_line_width,
            show_trace_labels=self.waterfall_show_trace_labels,
            trace_label_suffix=self.waterfall_trace_label_suffix,
            trace_label_font_size=self.waterfall_trace_label_font_size,
            trace_label_color=self.waterfall_trace_label_color,
        )
        self.ax_image.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_ylabel(self.model._channel_label())
        preserve_limits = (
            previous_xlim is not None
            and previous_ylim is not None
            and previous_dims == current_dims
            and getattr(self, "_last_plot_view_mode", None) == "waterfall"
        )
        self.ax_image.relim()
        self.ax_image.autoscale(enable=True, axis="both")
        default_xlim = tuple(float(value) for value in self.ax_image.get_xlim())
        default_ylim = tuple(float(value) for value in self.ax_image.get_ylim())
        defaults_changed = (
            self._waterfall_default_xlim is None
            or self._waterfall_default_ylim is None
            or not np.allclose(default_xlim, self._waterfall_default_xlim)
            or not np.allclose(default_ylim, self._waterfall_default_ylim)
        )
        self._waterfall_default_xlim = default_xlim
        self._waterfall_default_ylim = default_ylim
        if defaults_changed:
            self._set_navigation_home_to_current_view()
        if preserve_limits:
            self.ax_image.set_xlim(previous_xlim)
            self.ax_image.set_ylim(previous_ylim)
        self.image = None
        self.colorbar = None
        self._last_plot_dims = current_dims
        self._last_plot_view_mode = "waterfall"
        self._sync_control_visibility()
        self._sync_view_limit_controls()
        self._apply_figure_font_size()
        self._apply_axis_linewidth()
        self._connect_view_limit_callbacks()
        self.canvas.draw_idle()

    def _waterfall_uses_1d_group(self) -> bool:
        return (
            isinstance(self.data, MDHistoData)
            and sum(size > 1 for size in self.data.shape) == 1
        )

    def _waterfall_1d_source_indices(self) -> list[int]:
        """Return compatible 1D datasets represented by the current trace group."""

        if not self._viewer._waterfall_uses_1d_group():
            return []
        current_dim = next(index for index, size in enumerate(self.data.shape) if size > 1)
        current_axis = self.data.axes[current_dim]
        current_group_key = self.dataset_group_keys[self.dataset_index]
        current_binning = self._viewer.binning_names[self.dataset_index]
        indices = []
        for index, dataset in enumerate(self.datasets):
            if self.dataset_group_keys[index] != current_group_key:
                continue
            if self._viewer.binning_names[index] != current_binning:
                continue
            if not isinstance(dataset, MDHistoData):
                continue
            non_singleton = [
                dim for dim, size in enumerate(dataset.shape) if size > 1
            ]
            if len(non_singleton) != 1:
                continue
            axis = dataset.axes[non_singleton[0]]
            if (axis.name, axis.units) != (current_axis.name, current_axis.units):
                continue
            available_channels = {
                *MDHistoSliceViewer.CHANNELS,
                *dataset.auxiliary_channels,
            }
            for overlay in ("fit", "residual"):
                values = dataset.metadata.get(overlay)
                if isinstance(values, np.ndarray) and values.shape == dataset.shape:
                    available_channels.add(overlay)
            if self.model.channel not in available_channels:
                continue
            indices.append(index)
        return indices or [self.dataset_index]

    def waterfall_source_dataset_names(self) -> list[str]:
        """Return dataset names used by the active waterfall recipe."""

        if not self._waterfall_mode_active() or not self._viewer._waterfall_uses_1d_group():
            return [self.dataset_names[self.dataset_index]]
        return [
            self._viewer.source_dataset_names[index]
            for index in self._viewer._waterfall_1d_source_indices()
        ]


class FitComparisonController(_ViewerController):
    def _draw_fit_panels_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        """Draw side-by-side Data/Fit(/Residual) pcolor panels.

        All panels are sliced from the same dataset with identical selections
        and color settings, so data and fit are compared apples-to-apples.
        Data and fit share one color normalization; the residual panel uses
        the same colormap, scale, and manual limits but autoscales its own
        range since it is in sigma units.
        """

        panels: list[tuple[str, str]] = [
            ("Data", self.model.channel),
            ("Fit", "fit"),
        ]
        if self.show_residual and self._has_residual_channel():
            panels.append(("Residual", "residual"))

        with_cuts = self._fit_cuts_active()
        with_residual_cut = self._fit_residual_cut_active()
        self._ensure_fit_compare_layout(
            len(panels), with_cuts=with_cuts, with_residual_cut=with_residual_cut
        )
        data_model = self._viewer._comparison_panel_model(
            self.data,
            self.model.channel,
        )
        data_view = self._smoothed_slice_view(data_model.slice_arrays())
        data_values = data_model._display_values(data_view)
        shared_norm = data_model._color_norm(data_values)
        vmin, vmax = data_model._color_limits(data_values)

        self._current_slice = data_view
        self.image = None
        self.colorbar = None
        for index, (ax, (title, channel)) in enumerate(zip(self._compare_axes, panels, strict=True)):
            model = self._viewer._comparison_panel_model(self.data, channel)
            view = self._smoothed_slice_view(model.slice_arrays())
            values = model._display_values(view)
            norm = model._color_norm(values) if title == "Residual" else shared_norm
            artist = ax.pcolormesh(
                view["x_edges"],
                view["y_edges"],
                values,
                shading="auto",
                cmap=self.model._display_cmap(),
                norm=norm,
            )
            self._viewer._draw_brillouin_zone_overlay(ax)
            ax.set_title(title)
            ax.set_xlabel(model._axis_label(model.x_dim))
            ax.set_ylabel(model._axis_label(model.y_dim))
            colorbar_axis = (
                self._compare_colorbar_axes[index]
                if index < len(self._compare_colorbar_axes)
                else None
            )
            colorbar = self.figure.colorbar(
                artist,
                cax=colorbar_axis,
                ax=None if colorbar_axis is not None else ax,
            )
            colorbar.set_label(
                "Residual (sigma)" if title == "Residual" else data_model._channel_label()
            )
            self._compare_colorbars.append(colorbar)
            if title == "Data":
                self.image = artist
                self.colorbar = colorbar

        if previous_xlim is not None and previous_ylim is not None and previous_dims == current_dims:
            for axis in self._compare_axes:
                axis.set_xlim(previous_xlim)
                axis.set_ylim(previous_ylim)
        if with_cuts:
            if self._roi_extents is None or previous_dims != current_dims:
                extents = self._default_roi_extents()
            else:
                extents = self._roi_extents
            self._set_roi_extents(extents, update_cuts=True, draw=False)
        self._sync_control_visibility()
        self._sync_limit_spinboxes(vmin, vmax)
        self._last_plot_dims = current_dims
        self._last_plot_view_mode = "slice"
        self._sync_view_limit_controls()
        self._apply_figure_font_size()
        self._apply_axis_linewidth()
        self._connect_view_limit_callbacks()
        self.canvas.draw_idle()

    def _update_fit_compare_cuts(self, extents: tuple[float, float, float, float] | None) -> None:
        """Populate weighted data+fit and residual cut axes from a box.

        The data and fit cuts are overlaid on one axes (markers plus a line,
        like the 1D fit view); the residual cut, when shown, gets its own axes.
        Data and fit cuts are inverse-variance weighted means over the boxed
        rows or columns. Residual cuts retain their sigma summation.
        """

        if self.ax_fit_cut is None or self._current_slice is None or extents is None:
            return
        x0, x1, y0, y1 = extents
        data_view = self._current_slice
        x_centers = np.asarray(data_view["x_centers"], dtype=float)
        y_centers = np.asarray(data_view["y_centers"], dtype=float)
        x_mask = (x_centers >= x0) & (x_centers <= x1)
        y_mask = (y_centers >= y0) & (y_centers <= y1)
        self.ax_fit_cut.clear()
        if self.ax_ycut is not None:
            self.ax_ycut.clear()
        if self.ax_residual_cut is not None:
            self.ax_residual_cut.clear()
        if self.ax_residual_ycut is not None:
            self.ax_residual_ycut.clear()
        self._clear_roi_sum_annotation()

        if np.any(x_mask) and np.any(y_mask):
            x = x_centers[x_mask]
            y = y_centers[y_mask]
            data_z = self.model._display_values(data_view)
            errors = np.asarray(data_view.get("errors"), dtype=float)
            selected = np.ix_(y_mask, x_mask)
            if errors.shape == data_z.shape:
                self._show_roi_sum_annotation(
                    data_z[selected],
                    errors[selected],
                    extents,
                )
            x_coverage, y_coverage = self._histogram_cut_coverage(
                data_view, x_mask, y_mask
            )
            x_insufficient = x_coverage < self.coverage_threshold
            y_insufficient = y_coverage < self.coverage_threshold
            if errors.shape == data_z.shape:
                data_cut, err_cut = self._helper("inverse_variance_weighted_profile")(
                    data_z[selected], errors[selected], axis=0
                )
                data_y_cut, err_y_cut = self._helper(
                    "inverse_variance_weighted_profile"
                )(
                    data_z[selected], errors[selected], axis=1
                )
                data_cut = np.where(x_insufficient, np.nan, data_cut)
                err_cut = np.where(x_insufficient, np.nan, err_cut)
                data_y_cut = np.where(y_insufficient, np.nan, data_y_cut)
                err_y_cut = np.where(y_insufficient, np.nan, err_y_cut)
                self.ax_fit_cut.errorbar(
                    x, data_cut, yerr=err_cut, marker="o", linestyle="None",
                    ms=self.marker_size, mfc=self.marker_face_color or "none",
                    mec=self.line_color, ecolor=self.line_color, color=self.line_color,
                    elinewidth=self.line_plot_width, label="data",
                )
            else:
                data_cut = np.nansum(data_z[np.ix_(y_mask, x_mask)], axis=0)
                data_y_cut = np.nansum(data_z[np.ix_(y_mask, x_mask)], axis=1)
                data_cut = np.where(x_insufficient, np.nan, data_cut)
                data_y_cut = np.where(y_insufficient, np.nan, data_y_cut)
                err_y_cut = None
                self.ax_fit_cut.plot(
                    x, data_cut, marker="o", linestyle="None",
                    ms=self.marker_size, mfc=self.marker_face_color or "none",
                    mec=self.line_color, color=self.line_color, label="data",
                )
            fit_model = self._viewer._comparison_panel_model(self.data, "fit")
            fit_z = fit_model._display_values(self._smoothed_slice_view(fit_model.slice_arrays()))
            fit_errors = errors
            if self.unmask_model:
                raw_data_model = self._viewer._comparison_panel_model(
                    self.data,
                    self.model.channel,
                    masked=False,
                )
                raw_data_view = self._smoothed_slice_view(raw_data_model.slice_arrays())
                fit_errors = np.asarray(raw_data_view.get("errors"), dtype=float)
            if fit_errors.shape == data_z.shape:
                fit_cut, _ = self._helper("inverse_variance_weighted_profile")(
                    fit_z[selected], fit_errors[selected], axis=0
                )
                fit_y_cut, _ = self._helper("inverse_variance_weighted_profile")(
                    fit_z[selected], fit_errors[selected], axis=1
                )
            else:
                fit_cut = np.nansum(fit_z[np.ix_(y_mask, x_mask)], axis=0)
                fit_y_cut = np.nansum(fit_z[np.ix_(y_mask, x_mask)], axis=1)
            if not self.unmask_model:
                fit_cut = np.where(x_insufficient, np.nan, fit_cut)
                fit_y_cut = np.where(y_insufficient, np.nan, fit_y_cut)
            self.ax_fit_cut.plot(
                x, fit_cut, linestyle="-", marker="", color=self.fit_line_color,
                lw=self.fit_line_width, zorder=1.5, label="fit",
            )
            if self.ax_ycut is not None:
                data_plot = dict(
                    marker="o",
                    linestyle="None",
                    ms=self.marker_size,
                    mfc=self.marker_face_color or "none",
                    mec=self.line_color,
                    color=self.line_color,
                    label="data",
                )
                if err_y_cut is None:
                    self.ax_ycut.plot(data_y_cut, y, **data_plot)
                else:
                    self.ax_ycut.errorbar(
                        data_y_cut,
                        y,
                        xerr=err_y_cut,
                        ecolor=self.line_color,
                        elinewidth=self.line_plot_width,
                        **data_plot,
                    )
                self.ax_ycut.plot(
                    fit_y_cut,
                    y,
                    linestyle="-",
                    marker="",
                    color=self.fit_line_color,
                    lw=self.fit_line_width,
                    label="fit",
                )
            if self.ax_residual_cut is not None:
                residual_model = self._viewer._comparison_panel_model(
                    self.data,
                    "residual",
                )
                residual_z = residual_model._display_values(
                    self._smoothed_slice_view(residual_model.slice_arrays())
                )
                residual_cut = np.nansum(residual_z[np.ix_(y_mask, x_mask)], axis=0)
                residual_y_cut = np.nansum(residual_z[np.ix_(y_mask, x_mask)], axis=1)
                self.ax_residual_cut.axhline(0.0, color="0.5", lw=1.0, zorder=1)
                self.ax_residual_cut.plot(
                    x, residual_cut, marker="o", linestyle="None",
                    ms=self.marker_size, mfc=self.marker_face_color or "none",
                    mec=self.line_color, color=self.line_color,
                )
                if self.ax_residual_ycut is not None:
                    self.ax_residual_ycut.axvline(0.0, color="0.5", lw=1.0, zorder=1)
                    self.ax_residual_ycut.plot(
                        residual_y_cut,
                        y,
                        marker="o",
                        linestyle="None",
                        ms=self.marker_size,
                        mfc=self.marker_face_color or "none",
                        mec=self.line_color,
                        color=self.line_color,
                    )

        x_label = self.model._axis_label(self.model.x_dim)
        self.ax_fit_cut.set_ylabel("Weighted mean")
        if self.ax_residual_cut is not None:
            self.ax_residual_cut.set_ylabel("Res. (σ)")
            self.ax_residual_cut.set_xlabel(x_label)
        self.ax_fit_cut.set_xlabel(x_label)
        if self.ax_ycut is not None:
            self.ax_ycut.set_xlabel("Weighted mean")
            self.ax_ycut.set_ylabel(self.model._axis_label(self.model.y_dim))
            self.ax_ycut.tick_params(labelleft=False)
        if self.ax_residual_ycut is not None:
            self.ax_residual_ycut.set_xlabel("Res. (σ)")
            self.ax_residual_ycut.set_ylabel(self.model._axis_label(self.model.y_dim))
            self.ax_residual_ycut.tick_params(labelleft=False)

    def _comparison_panel_model(
        self,
        data: MDHistoData,
        channel: str,
        *,
        masked: bool | None = None,
    ) -> MDHistoSliceViewer:
        if masked is None:
            masked = self.model.masked and not (
                self.unmask_model and channel in {"fit", "residual"}
            )
        model = MDHistoSliceViewer(
            data,
            x_dim=self.model.x_dim,
            y_dim=self.model.y_dim,
            channel=channel,
            cmap=self.model.cmap,
            color_scale=self.model.color_scale,
            auto_limits=self.model.auto_limits,
            integrate=self.model.integrate,
            masked=masked,
            coverage_threshold=(
                0.0
                if not masked and channel in {"fit", "residual"}
                else self.coverage_threshold
            ),
        )
        model.selections.update(dict(self.model.selections))
        model.integrate_checks.update(dict(self.model.integrate_checks))
        model.cmap_reversed = bool(self.model.cmap_reversed)
        model.autoscale = self.model.autoscale
        model.manual_vmin = self.model.manual_vmin
        model.manual_vmax = self.model.manual_vmax
        model.sigma_n = self.model.sigma_n
        model.iqr_n = self.model.iqr_n
        model.percentile_n = self.model.percentile_n
        model.power_gamma = self.model.power_gamma
        model.color_alpha = self.model.color_alpha
        return model


class StandardSliceController(_ViewerController):
    def _slice_1d_channel(self, view: dict[str, np.ndarray], name: str) -> np.ndarray | None:
        """Return a fit/residual channel from the current slice as a 1D array."""

        channel_view = view
        if (
            self.unmask_model
            and name in {"fit", "residual"}
            and not getattr(self.model, "is_point_list", False)
        ):
            channel_model = self._viewer._comparison_panel_model(
                self.data,
                name,
                masked=False,
            )
            channel_view = self._smoothed_slice_view(channel_model.slice_arrays())
        values = channel_view.get(name)
        if values is None:
            return None
        flattened = np.asarray(values, dtype=float).reshape(-1)
        x = np.asarray(view["x_centers"], dtype=float)
        if flattened.size != x.size:
            flattened = np.squeeze(np.asarray(values, dtype=float))
            if flattened.size != x.size:
                return None
        return flattened

    def _draw_1d_view(self, view: dict[str, np.ndarray], values: np.ndarray) -> None:
        self.image = None
        x = np.asarray(view["x_centers"], dtype=float)
        y = np.asarray(values, dtype=float).reshape(-1)
        if y.size != x.size:
            y = np.squeeze(values)
        if self.show_fit:
            fit_values = self._viewer._slice_1d_channel(view, "fit")
            if fit_values is not None:
                # Solid connected line drawn behind the data points.
                self.ax_image.plot(
                    x,
                    fit_values,
                    linestyle="-",
                    marker="",
                    color=self.fit_line_color,
                    lw=self.fit_line_width,
                    zorder=1.5,
                    label="fit",
                )
        marker = self.marker
        linestyle = "None" if self.line_style == "none" else self.line_style
        common = {
            "marker": marker,
            "linestyle": linestyle,
            "ms": self.marker_size,
            "lw": self.line_plot_width,
            "mew": self.marker_edge_width,
            "mfc": (
                self.line_color
                if self.marker_face_color == "outline"
                else self.marker_face_color if marker else "none"
            ),
            "mec": self.line_color,
            "color": self.line_color,
        }
        # Primary and point-list channels use ``errors``; derived MDHisto
        # channels carry their independently propagated uncertainty alongside
        # the selected values.
        error_key = (
            "errors"
            if self.model.channel == "signal"
            or getattr(self.model, "is_point_list", False)
            else f"{self.model.channel}_errors"
        )
        errors = np.asarray(view.get(error_key, []), dtype=float).reshape(-1)
        has_errors = errors.size == x.size and bool(np.any(np.isfinite(errors)))
        if self.show_errorbars and has_errors:
            self.ax_image.errorbar(
                x,
                y,
                yerr=errors,
                ecolor=self.line_color,
                capsize=self.errorbar_cap_size if self.show_errorbar_caps else 0.0,
                capthick=self.line_plot_width,
                elinewidth=self.line_plot_width,
                **common,
            )
        else:
            self.ax_image.plot(x, y, **common)
        reference_lines = self.data.metadata.get("vertical_reference_lines", [])
        drew_reference = False
        if isinstance(reference_lines, list):
            for reference in reference_lines:
                if isinstance(reference, dict):
                    value = reference.get("value")
                    label = str(reference.get("label", ""))
                else:
                    value = reference
                    label = ""
                try:
                    position = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(position):
                    continue
                self.ax_image.axvline(
                    position,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.25,
                    label=label or None,
                    zorder=1.0,
                )
                drew_reference = True
        if drew_reference:
            self.ax_image.legend()
        self.ax_image.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_ylabel(self.model._channel_label())
        if self.model._is_boolean_channel():
            self.ax_image.set_ylim(0.0, 1.0)

    def _draw_1d_with_residual(self, previous_xlim, previous_dims, current_dims) -> None:
        """Draw the 1D data+fit with a residual axes below a movable separator."""

        view = self._current_slice
        values = self.model._display_values(view)
        self._ensure_residual_1d_layout()
        self._viewer._draw_1d_view(view, values)

        x = np.asarray(view["x_centers"], dtype=float)
        residual = self._viewer._slice_1d_channel(view, "residual")
        if residual is not None:
            self.ax_residual.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            common = {
                "marker": self.marker,
                "linestyle": "None",
                "ms": self.marker_size,
                "mew": self.marker_edge_width,
                "mfc": self.marker_face_color if self.marker else "none",
                "mec": self.line_color,
                "color": self.line_color,
            }
            if self.show_errorbars:
                # The residual is (data - fit) / sigma, so its uncertainty is one.
                self.ax_residual.errorbar(
                    x,
                    residual,
                    yerr=np.ones_like(x),
                    ecolor=self.line_color,
                    capsize=self.errorbar_cap_size if self.show_errorbar_caps else 0.0,
                    capthick=self.line_plot_width,
                    elinewidth=self.line_plot_width,
                    **common,
                )
            else:
                self.ax_residual.plot(x, residual, **common)
        self.ax_residual.set_ylabel("Res. (σ)")
        self.ax_residual.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_xlabel("")
        self.ax_image.tick_params(labelbottom=False)

        self._sync_control_visibility()
        if previous_xlim is not None and previous_dims == current_dims:
            self.ax_image.set_xlim(previous_xlim)
        self._last_plot_dims = current_dims
        self._last_plot_view_mode = "slice"
        self._sync_view_limit_controls()
        self._apply_figure_font_size()
        self._apply_axis_linewidth()
        self._connect_view_limit_callbacks()
        self.canvas.draw_idle()

    def _draw_2d_view(self, view: dict[str, np.ndarray], values: np.ndarray) -> None:
        norm = self.model._color_norm(values)
        self.image = self.ax_image.pcolormesh(
            view["x_edges"],
            view["y_edges"],
            values,
            shading="auto",
            cmap=self.model._display_cmap(),
            norm=norm,
        )
        self.ax_image.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_ylabel(self.model._axis_label(self.model.y_dim))
        self._viewer._draw_brillouin_zone_overlay()
        self._viewer._draw_bragg_peak_overlay()

    def _draw_bragg_peak_overlay(self) -> None:
        """Project stored HKLs into the active histogram-axis coordinates."""

        overlay = self.bragg_peak_overlay
        if overlay is None or getattr(self.model, "is_point_list", False):
            return
        overlay_dataset = overlay.get("dataset_name")
        if overlay_dataset is not None and self.dataset_names[self.dataset_index] != overlay_dataset:
            return
        if self.model.x_dim == self.model.y_dim:
            return
        axes = self.data.axes
        if axes[self.model.x_dim].kind != "momentum" or axes[self.model.y_dim].kind != "momentum":
            return
        try:
            from .analysis.coordinates import physical_axis_vectors

            vectors = physical_axis_vectors(self.data)[:, :3]
            momentum_dims = [index for index, axis in enumerate(axes) if axis.kind == "momentum"]
            if len(momentum_dims) != 3:
                return
            momentum_vectors = vectors[momentum_dims]
            axis_coordinates = np.asarray(overlay["hkl"], dtype=float) @ np.linalg.inv(momentum_vectors)
            x_column = momentum_dims.index(self.model.x_dim)
            y_column = momentum_dims.index(self.model.y_dim)
        except (ValueError, np.linalg.LinAlgError):
            return
        accepted = np.asarray(overlay["accepted"], dtype=bool)
        finite = np.all(np.isfinite(axis_coordinates), axis=1)
        accepted &= finite
        rejected = finite & ~accepted
        if np.any(accepted):
            self.ax_image.scatter(
                axis_coordinates[accepted, x_column],
                axis_coordinates[accepted, y_column],
                s=48,
                marker="o",
                facecolors="none",
                edgecolors="#2f9e68",
                linewidths=1.7,
                label="Accepted Bragg peaks",
                zorder=6,
            )
        if np.any(rejected):
            self.ax_image.scatter(
                axis_coordinates[rejected, x_column],
                axis_coordinates[rejected, y_column],
                s=48,
                marker="x",
                color="#d94b45",
                linewidths=1.7,
                label="Rejected Bragg peaks",
                zorder=6,
            )
        if np.any(accepted) or np.any(rejected):
            self.ax_image.legend(loc="upper right", fontsize=max(self.font_size - 3, 7))


class SliceModeControllers:
    """Mode-specific collaborators used by the Qt slice-viewer shell."""

    __slots__ = ("fit_comparison", "layouts", "standard", "tiled", "waterfall")

    def __init__(
        self,
        viewer: Any,
        *,
        helpers: dict[str, Any] | None = None,
    ) -> None:
        self.layouts = PlotLayoutController(viewer, helpers)
        self.tiled = TiledSliceController(viewer, helpers)
        self.waterfall = WaterfallController(viewer, helpers)
        self.fit_comparison = FitComparisonController(viewer, helpers)
        self.standard = StandardSliceController(viewer, helpers)
