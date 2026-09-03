from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .dataset import PointListData
from .mdhisto import MDHistoData
from .plotting import (
    MDHistoSliceViewer,
    TiledSlice,
    WaterfallTrace,
    default_tiled_slice_step,
    default_waterfall_offset,
    default_waterfall_step,
    draw_waterfall_traces,
    inverse_variance_weighted_profile,
    prepare_mdhisto_tiled_slices,
    prepare_mdhisto_waterfall,
    smooth_mdhisto_view,
    waterfall_absolute_max,
    waterfall_axis_display_name,
    waterfall_colors,
    waterfall_step_bounds,
)
from .qt_controls import configure_numeric_spin_boxes
from .qt_slice_controls import (
    _clear_layout,
    _compact_combobox,
    _DualRangeSlider,
    _expanding_combobox,
    _HiddenAxisControls,
    _IntegratedAxisSlider,
    _make_data_viewer_window_class,
    _make_float_spinbox,
    _make_index_slider,  # noqa: F401 - compatibility re-export
    _qt_app,
)
from .quantities import display_unit
from .slice_viewer_state import (
    _axis_components,
    _coerce_dataset_group_keys,
    _coerce_dataset_names,
    _coerce_datasets,
    _cursor_axis_hkle_vector,
    _cursor_matrix_includes_2pi,  # noqa: F401 - compatibility re-export
    _cursor_q_matrix_from_metadata,
    _cursor_q_matrix_from_oriented_lattice,
    _DatasetViewState,
    _format_coord,
    _format_value_with_uncertainty,
    _initial_display_dims,
    _nearest_index,
    _option_name,
    _point_list_q_column_name,
    _point_list_two_theta_column_name,
    _point_list_wavelength,
)

_MARKER_OPTIONS = {
    "none": "",
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "plus": "+",
    "cross": "x",
}
_LINE_STYLE_OPTIONS = {
    "none": "none",
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dash-dot": "-.",
}
_COLOR_OPTIONS = {
    "none": "none",
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "black": "#000000",
}
_WATERFALL_COLORMAPS = (
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "cividis",
    "turbo",
    "tab10",
    "Dark2",
    "Set1",
    "Blues",
    "Reds",
)
_WATERFALL_DISCRETE_COLORMAPS = {"tab10", "Dark2", "Set1"}


class QtMDHistoSliceViewer:
    """PySide6 slice viewer with an embedded Matplotlib canvas."""

    def __init__(
        self,
        data: MDHistoData | Sequence[MDHistoData],
        *,
        dataset_names: Sequence[str] | None = None,
        dataset_group_keys: Sequence[str] | None = None,
        x_dim: int | str = -1,
        y_dim: int | str = 0,
        channel: str = "signal",
        cmap: str = "viridis",
        color_scale: str = "linear",
        auto_limits: str = "min/max",
        integrate: bool = False,
        masked: bool = True,
    ) -> None:
        self.app = _qt_app()
        self.datasets = _coerce_datasets(data)
        self.dataset_names = _coerce_dataset_names(self.datasets, dataset_names)
        self.dataset_group_keys = _coerce_dataset_group_keys(
            self.datasets,
            dataset_group_keys,
        )
        self.dataset_index = 0
        self._initial_x_dim = x_dim
        self._initial_y_dim = y_dim
        self._initial_channel = channel
        self._initial_cmap = cmap
        self._initial_color_scale = color_scale
        self._initial_auto_limits = auto_limits
        self._initial_integrate = integrate
        self._initial_masked = masked
        initial_data = self.datasets[self.dataset_index]
        model_x_dim, model_y_dim = _initial_display_dims(initial_data, x_dim, y_dim)
        self.model = MDHistoSliceViewer(
            initial_data,
            x_dim=model_x_dim,
            y_dim=model_y_dim,
            channel=channel,
            cmap=cmap,
            color_scale=color_scale,
            auto_limits=auto_limits,
            integrate=integrate,
            masked=masked,
        )
        self.data = initial_data
        self.window = None
        self.canvas = None
        self.figure = None
        self.grid = None
        self.controls_scroll = None
        self.ax_image = None
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.image = None
        self.colorbar = None
        self.rectangle_selector = None
        self.dataset_combo = None
        self.axes_group = None
        self.show_fit_check = None
        self.unmask_model_check = None
        self.show_residual_check = None
        self.residual_split_slider = None
        self.residual_split_label = None
        self.fit_line_color_combo = None
        self.fit_line_width_spin = None
        self.fit_line_color_label = None
        self.fit_line_width_label = None
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self._compare_colorbar_axes = []
        self.axis_selector_widget = None
        self.x_combo = None
        self.y_combo = None
        self._axis_y_label = None
        self.x_min_spin = None
        self.x_max_spin = None
        self.y_min_spin = None
        self.y_max_spin = None
        self.x_reset_button = None
        self.y_reset_button = None
        self.toolbar = None
        self.cmap_combo = None
        self.cmap_reverse_button = None
        self.channel_combo = None
        self.apply_masks_check = None
        self.coverage_threshold_label = None
        self.coverage_threshold_spin = None
        self.scale_combo = None
        self.limits_combo = None
        self.autoscale_check = None
        self.vmin_spin = None
        self.vmax_spin = None
        self.color_group = None
        self.smoothing_group = None
        self.smoothing_x_spin = None
        self.smoothing_y_spin = None
        self.smoothing_y_label = None
        self.gamma_label = None
        self.gamma_spin = None
        self.limit_n_label = None
        self.limit_n_spin = None
        self.cursor_xy_label = None
        self.cursor_hkle_label = None
        self.cursor_q_label = None
        self.cursor_intensity_label = None
        self.roi_button = None
        self.tools_group = None
        self.show_box_check = None
        self.hist_axes_check = None
        self.xcut_percent_slider = None
        self.ycut_percent_slider = None
        self.xcut_percent_label = None
        self.ycut_percent_label = None
        self.roi_x_center_spin = None
        self.roi_x_width_spin = None
        self.roi_y_center_spin = None
        self.roi_y_width_spin = None
        self.font_size_spin = None
        self.line_width_spin = None
        self.show_binning_title_check = None
        self.line_group = None
        self.marker_combo = None
        self.line_style_combo = None
        self.marker_size_spin = None
        self.line_plot_width_spin = None
        self.marker_edge_width_spin = None
        self.marker_face_color_combo = None
        self.line_color_combo = None
        self.show_errorbars_check = None
        self.show_errorbar_caps_check = None
        self.errorbar_cap_size_spin = None
        self.copy_figure_button = None
        self.save_plot_button = None
        self.copy_script_button = None
        self.save_script_button = None
        self._save_plot_callback = None
        self._save_project_callback = None
        self._open_new_viewer_callback = None
        self._close_callback = None
        self._child_viewers = []
        self.open_new_viewer_button = None
        self.save_project_shortcut = None
        self._unmask_model_callback = None
        self.view_mode_combo = None
        self.content_stack = None
        self.volume_panel = None
        self.waterfall_group = None
        self.tiled_group = None
        self.tile_dim_combo = None
        self.tile_range_low_spin = None
        self.tile_range_high_spin = None
        self.tile_step_spin = None
        self.tile_step_slider = None
        self.tile_step_auto_check = None
        self.waterfall_source_label = None
        self.waterfall_step_label = None
        self.waterfall_step_spin = None
        self.waterfall_step_slider = None
        self.waterfall_step_auto_check = None
        self.waterfall_coverage_threshold_spin = None
        self.waterfall_offset_spin = None
        self.waterfall_offset_slider = None
        self.waterfall_offset_auto_check = None
        self.waterfall_cmap_combo = None
        self.waterfall_color_range_label = None
        self.waterfall_color_range_slider = None
        self.waterfall_reverse_check = None
        self.waterfall_zero_check = None
        self.waterfall_zero_color_combo = None
        self.waterfall_zero_style_combo = None
        self.waterfall_zero_width_spin = None
        self.waterfall_model_color_combo = None
        self.waterfall_trace_labels_check = None
        self.waterfall_trace_label_suffix_edit = None
        self.waterfall_trace_label_font_size_spin = None
        self.waterfall_trace_label_color_combo = None
        self.xcut_percent = 20
        self.ycut_percent = 16
        self.font_size = 12.0
        self.show_binning_title = False
        self.axis_linewidth = 1.5
        self.smoothing_x = 0.0
        self.smoothing_y = 0.0
        self.marker = "o"
        self.line_style = "none"
        self.marker_size = 5.0
        self.line_plot_width = 1.5
        self.marker_edge_width = 1.5
        self.marker_face_color = "none"
        self._slice_marker_face_color = "none"
        self._waterfall_marker_face_color = "none"
        self._active_plot_view_mode = 0
        self.line_color = "#1f77b4"
        self.show_errorbars = True
        self.show_errorbar_caps = False
        self.errorbar_cap_size = 3.0
        self.show_fit = bool(initial_data.metadata.get("viewer_show_fit", False))
        self.unmask_model = False
        self.show_residual = False
        self.fit_line_color = "#d62728"
        self.fit_line_width = 2.0
        self.residual_percent = 30
        self.coverage_threshold = 0.9
        self.waterfall_step = 1.0
        self.waterfall_coverage_threshold = 0.9
        self.waterfall_step_auto = True
        self.waterfall_offset = 1.0
        self.waterfall_offset_auto = True
        self.waterfall_cmap = "viridis"
        self.waterfall_color_min = 0.0
        self.waterfall_color_max = 1.0
        self.waterfall_reverse_colors = False
        self.waterfall_show_zero_lines = True
        self.waterfall_zero_color = "#7f7f7f"
        self.waterfall_zero_style = "--"
        self.waterfall_zero_width = 0.8
        self.waterfall_model_color: str | None = None
        self.waterfall_show_trace_labels = True
        self.waterfall_trace_label_suffix = ""
        self.waterfall_trace_label_font_size = 10.0
        self.waterfall_trace_label_color: str | None = None
        self.waterfall_dataset_names: list[str] | None = None
        tile_candidates = (
            [
                dim
                for dim, size in enumerate(initial_data.shape)
                if size > 1 and dim not in (model_x_dim, model_y_dim)
            ]
            if isinstance(initial_data, MDHistoData)
            else []
        )
        self.tile_dim: int | None = tile_candidates[0] if tile_candidates else None
        if self.tile_dim is None:
            self.tile_range = (0.0, 0.0)
            self.tile_step = 1.0
        else:
            tile_centers = np.asarray(initial_data.axes[self.tile_dim].centers, dtype=float)
            self.tile_range = (float(tile_centers[0]), float(tile_centers[-1]))
            self.tile_step = default_tiled_slice_step(
                initial_data,
                self.tile_dim,
                self.tile_range,
            )
        self.tile_step_auto = True
        self._current_tiled_slices: list[TiledSlice] = []
        self._tile_axes = []
        self._current_waterfall_traces: list[WaterfallTrace] = []
        self._waterfall_default_xlim: tuple[float, float] | None = None
        self._waterfall_default_ylim: tuple[float, float] | None = None
        self.hidden_layout = None
        self.hidden_controls: dict[int, _HiddenAxisControls] = {}
        self._display_axis_dims: list[int] = []
        self._current_slice: dict[str, np.ndarray] | None = None
        self._syncing_axes = False
        self._syncing_limits = False
        self._syncing_view_limits = False
        self._autoscaling_view = False
        self._syncing_roi_controls = False
        self._box_tool_has_auto_shown_hist_axes = False
        self._restoring_dataset_state = False
        self._roi_extents: tuple[float, float, float, float] | None = None
        self._view_limit_callback_ids: list[int] = []
        self.bragg_peak_overlay: dict[str, Any] | None = None
        self._dataset_states: list[_DatasetViewState | None] = [None] * len(self.datasets)
        self._dataset_states[0] = _DatasetViewState(
            model=self.model,
            show_fit=self.show_fit,
            tile_dim=self.tile_dim,
            tile_range=self.tile_range,
            tile_step=self.tile_step,
            tile_step_auto=self.tile_step_auto,
        )
        self._plot_layout_mode: tuple[Any, ...] | None = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._build()
        configure_numeric_spin_boxes(self.app)
        self.update_plot()

    def show(self) -> QtMDHistoSliceViewer:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self

    def run(self) -> int:
        self.show()
        return int(self.app.exec())

    def slice_arrays(self) -> dict[str, np.ndarray]:
        return self.model.slice_arrays()

    def replace_datasets(
        self,
        data: MDHistoData | Sequence[MDHistoData],
        *,
        dataset_names: Sequence[str] | None = None,
        dataset_group_keys: Sequence[str] | None = None,
        selected_dataset_name: str | None = None,
    ) -> None:
        """Update displayed datasets in place while preserving viewer state."""

        previous_names = list(self.dataset_names)
        if 0 <= self.dataset_index < len(self._dataset_states):
            self._dataset_states[self.dataset_index] = self._capture_dataset_state()
        previous_states = {
            name: state
            for name, state in zip(previous_names, self._dataset_states, strict=False)
            if state is not None
        }
        current_name = (
            selected_dataset_name
            or (previous_names[self.dataset_index] if 0 <= self.dataset_index < len(previous_names) else None)
        )
        self.datasets = _coerce_datasets(data)
        self.dataset_names = _coerce_dataset_names(self.datasets, dataset_names)
        self.dataset_group_keys = _coerce_dataset_group_keys(
            self.datasets,
            dataset_group_keys,
        )
        if self.volume_panel is not None:
            self.content_stack.setCurrentIndex(0)
            self.view_mode_combo.setCurrentIndex(0)
            self._close_volume_panel()
        if current_name in self.dataset_names:
            new_index = self.dataset_names.index(current_name)
        else:
            new_index = min(self.dataset_index, len(self.datasets) - 1)
        self._dataset_states = [None] * len(self.datasets)
        for index, name in enumerate(self.dataset_names):
            state = previous_states.get(name)
            if state is not None:
                state.model.data = self.datasets[index]
                state.model.refresh_metadata_channels()
                self._dataset_states[index] = state
        state = self._dataset_states[new_index]
        if state is None:
            state = self._default_dataset_state(new_index)
            self._dataset_states[new_index] = state
        self.dataset_index = new_index
        self._set_combo_items_silent(self.dataset_combo, self.dataset_names, self.dataset_names[new_index])
        self._restore_dataset_state(state)

    def set_bragg_peak_overlay(self, peaks: Any | None, *, dataset_name: str | None = None) -> None:
        """Overlay accepted and rejected Bragg reflections on 2D momentum views."""

        if peaks is None:
            self.bragg_peak_overlay = None
        else:
            try:
                hkl = np.column_stack([peaks.column(name) for name in ("H", "K", "L")])
                accepted = (
                    np.asarray(peaks.column("Accepted"), dtype=bool)
                    if "Accepted" in peaks.columns
                    else np.ones(hkl.shape[0], dtype=bool)
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError("Bragg overlay requires H, K, and L table columns") from exc
            self.bragg_peak_overlay = {
                "hkl": np.asarray(hkl, dtype=float),
                "accepted": accepted,
                "dataset_name": dataset_name,
            }
        self.update_plot()

    @property
    def image_norm(self):
        return None if self.image is None else self.image.norm

    def copy_figure_to_clipboard(self) -> None:
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())

    def copy_script_to_clipboard(self) -> None:
        from PySide6 import QtWidgets

        QtWidgets.QApplication.clipboard().setText(self.figure_script())

    def current_plot_settings(self) -> dict[str, object]:
        """Capture the current viewer state as a JSON-ready saved-plot recipe."""

        xlim = self._export_limits("x") if self.ax_image is not None else None
        ylim = self._export_limits("y") if self.ax_image is not None else None
        view_mode = (
            "volumetric"
            if self.view_mode_combo.currentIndex() == 3
            else "tiled_slices"
            if self._tiled_mode_active()
            else "waterfall"
            if self._waterfall_mode_active()
            else "slice"
        )
        dataset_name = self.dataset_combo.currentText()
        if view_mode == "volumetric" and self.volume_panel is not None:
            dataset_name = self.volume_panel.dataset_combo.currentText()
        return {
            "view_mode": view_mode,
            "dataset_name": dataset_name,
            "x_dim": self.data.axes[self.model.x_dim].name,
            "y_dim": self.data.axes[self.model.y_dim].name,
            "channel": self.model.channel,
            "selections": self._export_selections(),
            "integrate_checks": self._export_integrate_checks(),
            "cmap": self.model._effective_cmap(),
            "color_scale": self.model.color_scale,
            "auto_limits": self.model.auto_limits,
            "autoscale": self.model.autoscale,
            "manual_vmin": self.model.manual_vmin,
            "manual_vmax": self.model.manual_vmax,
            "smoothing_x": self.smoothing_x,
            "smoothing_y": self.smoothing_y,
            "xlim": xlim,
            "ylim": ylim,
            "font_size": self.font_size,
            "axis_linewidth": self.axis_linewidth,
            "show_binning_title": self.show_binning_title,
            "show_histogram_axes": bool(self.hist_axes_check and self.hist_axes_check.isChecked()),
            "roi_extents": self._roi_extents,
            "xcut_percent": self.xcut_percent,
            "ycut_percent": self.ycut_percent,
            "show_fit": self.show_fit,
            "unmask_model": self.unmask_model,
            "show_residual": self.show_residual,
            "apply_masks": self.model.masked,
            "coverage_threshold": self.coverage_threshold,
            "show_box_tool": bool(self.show_box_check and self.show_box_check.isChecked()),
            "roi_enabled": bool(self.roi_button and self.roi_button.isChecked()),
            "waterfall_step": self.waterfall_step,
            "waterfall_coverage_threshold": self.waterfall_coverage_threshold,
            "waterfall_step_auto": self.waterfall_step_auto,
            "waterfall_offset": self.waterfall_offset,
            "waterfall_offset_auto": self.waterfall_offset_auto,
            "waterfall_cmap": self.waterfall_cmap,
            "waterfall_color_min": self.waterfall_color_min,
            "waterfall_color_max": self.waterfall_color_max,
            "waterfall_reverse_colors": self.waterfall_reverse_colors,
            "waterfall_show_zero_lines": self.waterfall_show_zero_lines,
            "waterfall_zero_color": self.waterfall_zero_color,
            "waterfall_zero_style": self.waterfall_zero_style,
            "waterfall_zero_width": self.waterfall_zero_width,
            "waterfall_model_color": self.waterfall_model_color,
            "waterfall_show_trace_labels": self.waterfall_show_trace_labels,
            "waterfall_trace_label_suffix": self.waterfall_trace_label_suffix,
            "waterfall_trace_label_font_size": self.waterfall_trace_label_font_size,
            "waterfall_trace_label_color": self.waterfall_trace_label_color,
            "waterfall_dataset_names": self.waterfall_source_dataset_names(),
            "tile_dim": (
                self.data.axes[self.tile_dim].name
                if self.tile_dim is not None
                else None
            ),
            "tile_range": self.tile_range,
            "tile_step": self.tile_step,
            "tile_step_auto": self.tile_step_auto,
            "marker": self.marker,
            "line_style": self.line_style,
            "marker_size": self.marker_size,
            "line_plot_width": self.line_plot_width,
            "marker_edge_width": self.marker_edge_width,
            "marker_face_color": self.marker_face_color,
            "show_errorbars": self.show_errorbars,
            "show_errorbar_caps": self.show_errorbar_caps,
            "errorbar_cap_size": self.errorbar_cap_size,
            "line_color": self.line_color,
            "fit_line_color": self.fit_line_color,
            "fit_line_width": self.fit_line_width,
            "residual_percent": self.residual_percent,
            "figsize": tuple(self.figure.get_size_inches()) if self.figure is not None else (8.0, 6.5),
        }

    def apply_plot_settings(self, settings: dict[str, object]) -> None:
        """Restore a saved plot recipe into the interactive controls."""

        dataset_name = settings.get("dataset_name")
        if dataset_name in self.dataset_names:
            self.dataset_combo.setCurrentIndex(self.dataset_names.index(dataset_name))
        x_name = settings.get("x_dim")
        y_name = settings.get("y_dim")
        names = [axis.name for axis in self.data.axes]
        if x_name in names:
            self._set_display_dim("x", names.index(x_name))
        if y_name in names and len(names) > 1:
            self._set_display_dim("y", names.index(y_name))
        if settings.get("channel") in self.model.CHANNELS:
            self._set_channel(str(settings["channel"]))
        self.model.selections.update({int(key): tuple(value) for key, value in dict(settings.get("selections", {})).items()})
        self.model.integrate_checks.update({int(key): bool(value) for key, value in dict(settings.get("integrate_checks", {})).items()})
        effective_cmap = str(settings.get("cmap", self.model.cmap))
        self._set_cmap(effective_cmap.removesuffix("_r"))
        self.model.cmap_reversed = effective_cmap.endswith("_r")
        self._set_color_scale(str(settings.get("color_scale", self.model.color_scale)))
        self._set_auto_limits(str(settings.get("auto_limits", self.model.auto_limits)))
        self.model.manual_vmin = settings.get("manual_vmin", self.model.manual_vmin)
        self.model.manual_vmax = settings.get("manual_vmax", self.model.manual_vmax)
        self._set_autoscale(bool(settings.get("autoscale", self.model.autoscale)))
        self.smoothing_x = float(settings.get("smoothing_x", self.smoothing_x))
        self.smoothing_y = float(settings.get("smoothing_y", self.smoothing_y))
        self._set_spin_silent(self.smoothing_x_spin, self.smoothing_x)
        self._set_spin_silent(self.smoothing_y_spin, self.smoothing_y)
        self._set_font_size(float(settings.get("font_size", self.font_size)))
        self._set_axis_linewidth(float(settings.get("axis_linewidth", self.axis_linewidth)))
        self._set_show_binning_title(
            bool(settings.get("show_binning_title", self.show_binning_title)),
            redraw=False,
        )
        self._set_show_fit(bool(settings.get("show_fit", self.show_fit)))
        self._set_unmask_model(bool(settings.get("unmask_model", self.unmask_model)))
        self._set_show_residual(bool(settings.get("show_residual", self.show_residual)))
        self.model.masked = bool(settings.get("apply_masks", self.model.masked))
        self._set_checkbox_silent(self.apply_masks_check, self.model.masked)
        self._set_coverage_threshold(
            float(settings.get("coverage_threshold", self.coverage_threshold)),
            redraw=False,
        )
        self.waterfall_step = float(settings.get("waterfall_step", self.waterfall_step))
        self.waterfall_coverage_threshold = float(
            settings.get(
                "waterfall_coverage_threshold",
                self.waterfall_coverage_threshold,
            )
        )
        self.waterfall_step_auto = bool(
            settings.get("waterfall_step_auto", self.waterfall_step_auto)
        )
        self.waterfall_offset = float(
            settings.get("waterfall_offset", self.waterfall_offset)
        )
        self.waterfall_offset_auto = bool(
            settings.get("waterfall_offset_auto", self.waterfall_offset_auto)
        )
        self.waterfall_cmap = str(
            settings.get("waterfall_cmap", self.waterfall_cmap)
        )
        self.waterfall_color_min = float(
            settings.get("waterfall_color_min", self.waterfall_color_min)
        )
        self.waterfall_color_max = float(
            settings.get("waterfall_color_max", self.waterfall_color_max)
        )
        self.waterfall_reverse_colors = bool(
            settings.get(
                "waterfall_reverse_colors",
                self.waterfall_reverse_colors,
            )
        )
        self.waterfall_show_zero_lines = bool(
            settings.get(
                "waterfall_show_zero_lines",
                self.waterfall_show_zero_lines,
            )
        )
        self.waterfall_zero_color = str(
            settings.get("waterfall_zero_color", self.waterfall_zero_color)
        )
        self.waterfall_zero_style = str(
            settings.get("waterfall_zero_style", self.waterfall_zero_style)
        )
        self.waterfall_zero_width = float(
            settings.get("waterfall_zero_width", self.waterfall_zero_width)
        )
        model_color = settings.get(
            "waterfall_model_color",
            self.waterfall_model_color,
        )
        self.waterfall_model_color = None if model_color is None else str(model_color)
        self.waterfall_show_trace_labels = bool(
            settings.get(
                "waterfall_show_trace_labels",
                self.waterfall_show_trace_labels,
            )
        )
        self.waterfall_trace_label_suffix = str(
            settings.get(
                "waterfall_trace_label_suffix",
                self.waterfall_trace_label_suffix,
            )
        )
        self.waterfall_trace_label_font_size = float(
            settings.get(
                "waterfall_trace_label_font_size",
                self.waterfall_trace_label_font_size,
            )
        )
        label_color = settings.get(
            "waterfall_trace_label_color",
            self.waterfall_trace_label_color,
        )
        self.waterfall_trace_label_color = (
            None if label_color is None else str(label_color)
        )
        dataset_names = settings.get("waterfall_dataset_names")
        self.waterfall_dataset_names = (
            [str(name) for name in dataset_names]
            if isinstance(dataset_names, (list, tuple))
            else None
        )
        tile_name = settings.get("tile_dim")
        if tile_name in names:
            self.tile_dim = names.index(str(tile_name))
        tile_range = settings.get("tile_range")
        if isinstance(tile_range, (list, tuple)) and len(tile_range) == 2:
            self.tile_range = (float(tile_range[0]), float(tile_range[1]))
        self.tile_step = float(settings.get("tile_step", self.tile_step))
        self.tile_step_auto = bool(
            settings.get("tile_step_auto", self.tile_step_auto)
        )
        self.marker = str(settings.get("marker", self.marker))
        self.line_style = str(settings.get("line_style", self.line_style))
        self.marker_size = float(settings.get("marker_size", self.marker_size))
        self.line_plot_width = float(
            settings.get("line_plot_width", self.line_plot_width)
        )
        self.marker_edge_width = float(
            settings.get("marker_edge_width", self.marker_edge_width)
        )
        marker_face_color = str(
            settings.get("marker_face_color", self.marker_face_color)
        )
        self.show_errorbars = bool(
            settings.get("show_errorbars", self.show_errorbars)
        )
        self.show_errorbar_caps = bool(
            settings.get("show_errorbar_caps", self.show_errorbar_caps)
        )
        self.errorbar_cap_size = float(
            settings.get("errorbar_cap_size", self.errorbar_cap_size)
        )
        self.line_color = str(settings.get("line_color", self.line_color))
        self.fit_line_color = str(
            settings.get("fit_line_color", self.fit_line_color)
        )
        self.fit_line_width = float(
            settings.get("fit_line_width", self.fit_line_width)
        )
        self.residual_percent = int(
            settings.get("residual_percent", self.residual_percent)
        )
        mode = {
            "waterfall": 1,
            "tiled_slices": 2,
            "volumetric": 3,
        }.get(settings.get("view_mode"), 0)
        if mode == 1:
            self._waterfall_marker_face_color = marker_face_color
        else:
            self._slice_marker_face_color = marker_face_color
        if self.view_mode_combo.currentIndex() == mode:
            self.marker_face_color = marker_face_color
        self._sync_waterfall_controls()
        self._sync_tile_controls()
        self.view_mode_combo.setCurrentIndex(mode)
        self._set_combo_silent(
            self.marker_face_color_combo,
            (
                "outline"
                if self.marker_face_color == "outline"
                else _option_name(_COLOR_OPTIONS, self.marker_face_color)
            ),
        )
        self._roi_extents = settings.get("roi_extents", self._roi_extents)
        self.xcut_percent = int(settings.get("xcut_percent", self.xcut_percent))
        self.ycut_percent = int(settings.get("ycut_percent", self.ycut_percent))
        self._set_slider_silent(self.xcut_percent_slider, self.xcut_percent)
        self._set_slider_silent(self.ycut_percent_slider, self.ycut_percent)
        self._set_slider_silent(self.residual_split_slider, self.residual_percent)
        self._set_checkbox_silent(
            self.hist_axes_check,
            bool(settings.get("show_histogram_axes", self.hist_axes_check.isChecked())),
        )
        self._set_checkbox_silent(
            self.show_box_check,
            bool(settings.get("show_box_tool", self.show_box_check.isChecked())),
        )
        self._set_checkbox_silent(
            self.roi_button,
            bool(settings.get("roi_enabled", self.roi_button.isChecked())),
        )
        self.update_plot(preserve_view=False)
        if settings.get("xlim") is not None:
            self.ax_image.set_xlim(*settings["xlim"])
        if settings.get("ylim") is not None:
            self.ax_image.set_ylim(*settings["ylim"])
        self._sync_view_limit_controls()
        self._set_rectangle_selector_from_controls()
        self._sync_histogram_panel_controls()
        self._apply_histogram_axes_layout(draw=False)
        if mode == 2 and self.volume_panel is not None:
            volume_index = self.volume_panel.dataset_combo.findText(str(dataset_name))
            if volume_index >= 0:
                self.volume_panel.dataset_combo.setCurrentIndex(volume_index)
        self.canvas.draw_idle()

    def open_new_viewer(self):
        """Open an independent viewer initialized from the current view."""

        settings = self.current_plot_settings()
        if self._open_new_viewer_callback is not None:
            viewer = self._open_new_viewer_callback(settings.get("dataset_name"))
        else:
            viewer = type(self)(
                self.datasets,
                dataset_names=self.dataset_names,
                dataset_group_keys=self.dataset_group_keys,
            )
            self._child_viewers.append(viewer)
        if viewer is None:
            return None
        viewer.apply_plot_settings(settings)
        viewer.show()
        return viewer

    def set_open_new_viewer_callback(self, callback) -> None:
        """Set the project-aware factory used by :meth:`open_new_viewer`."""

        self._open_new_viewer_callback = callback

    def set_save_plot_callback(self, callback) -> None:
        """Expose project-bound saved-plot creation when a callback is supplied."""

        self._save_plot_callback = callback
        if self.save_plot_button is not None:
            self.save_plot_button.setEnabled(callback is not None)

    def set_save_project_callback(self, callback) -> None:
        """Route the standard Save shortcut to the owning project window."""

        self._save_project_callback = callback
        if self.save_project_shortcut is not None:
            self.save_project_shortcut.setEnabled(callback is not None)

    def set_unmask_model_callback(self, callback) -> None:
        """Set the project callback that rebuilds full-grid model channels."""

        self._unmask_model_callback = callback

    def set_close_callback(self, callback) -> None:
        """Notify the owning project when this viewer window closes."""

        self._close_callback = callback

    def save_script(self) -> None:
        from pathlib import Path

        from PySide6 import QtWidgets

        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save figure script",
            (
                "mdhisto_tiled_slices_figure.py"
                if self._tiled_mode_active()
                else "mdhisto_waterfall_figure.py"
                if self._waterfall_mode_active()
                else "mdhisto_slice_figure.py"
            ),
            "Python scripts (*.py);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        Path(path).write_text(self.figure_script(), encoding="utf-8")

    def figure_script(self) -> str:
        if self._waterfall_mode_active():
            return self._waterfall_figure_script()
        if self._tiled_mode_active():
            return self._tiled_figure_script()
        source_file = self.data.metadata.get("source_file") if isinstance(self.data.metadata, dict) else None
        data_line = (
            f"data = load_mantid_mdhisto_nxs({source_file!r}, copy_metadata=False)"
            if source_file
            else "data = ...  # Replace with your MDHistoData object"
        )
        if self._is_effective_1d():
            return self._line_figure_script(data_line)
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                "from nfit import load_mantid_mdhisto_nxs, plot_mdhisto_slice",
                "",
                data_line,
                "fig = plot_mdhisto_slice(",
                "    data,",
                f"    x_dim={self.data.axes[self.model.x_dim].name!r},",
                f"    y_dim={self.data.axes[self.model.y_dim].name!r},",
                f"    channel={self.model.channel!r},",
                f"    selections={self._export_selections()!r},",
                f"    integrate_checks={self._export_integrate_checks()!r},",
                f"    coverage_threshold={self.coverage_threshold!r},",
                f"    cmap={self.model._effective_cmap()!r},",
                f"    color_scale={self.model.color_scale!r},",
                f"    auto_limits={self.model.auto_limits!r},",
                f"    autoscale={self.model.autoscale!r},",
                f"    manual_vmin={self.model.manual_vmin!r},",
                f"    manual_vmax={self.model.manual_vmax!r},",
                f"    sigma_n={self.model.sigma_n!r},",
                f"    iqr_n={self.model.iqr_n!r},",
                f"    percentile_n={self.model.percentile_n!r},",
                f"    power_gamma={self.model.power_gamma!r},",
                f"    smoothing_sigma_x={self.smoothing_x!r},",
                f"    smoothing_sigma_y={self.smoothing_y!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    axes_linewidth={self.axis_linewidth!r},",
                f"    show_histogram_axes={self.hist_axes_check.isChecked()!r},",
                f"    roi_extents={self._roi_extents!r},",
                f"    xcut_percent={self.xcut_percent!r},",
                f"    ycut_percent={self.ycut_percent!r},",
                ")",
                f"fig.suptitle({self._binning_title_text()!r})",
                "plt.show()",
                "",
            ]
        )

    def _tiled_figure_script(self) -> str:
        source_file = (
            self.data.metadata.get("source_file")
            if isinstance(self.data.metadata, dict)
            else None
        )
        data_line = (
            f"data = load_mantid_mdhisto_nxs({source_file!r}, copy_metadata=False)"
            if source_file
            else "data = ...  # Replace with your MDHistoData object"
        )
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                "from nfit import load_mantid_mdhisto_nxs, plot_mdhisto_tiled_slices",
                "",
                data_line,
                "fig = plot_mdhisto_tiled_slices(",
                "    data,",
                f"    x_dim={self.data.axes[self.model.x_dim].name!r},",
                f"    y_dim={self.data.axes[self.model.y_dim].name!r},",
                f"    tile_dim={self.data.axes[self.tile_dim].name!r},",
                f"    channel={self.model.channel!r},",
                f"    selections={self._export_selections()!r},",
                f"    integrate_checks={self._export_integrate_checks()!r},",
                f"    tile_range={self.tile_range!r},",
                f"    tile_step={self.tile_step!r},",
                f"    coverage_threshold={self.coverage_threshold!r},",
                f"    masked={self.model.masked!r},",
                f"    cmap={self.model._effective_cmap()!r},",
                f"    color_scale={self.model.color_scale!r},",
                f"    auto_limits={self.model.auto_limits!r},",
                f"    autoscale={self.model.autoscale!r},",
                f"    manual_vmin={self.model.manual_vmin!r},",
                f"    manual_vmax={self.model.manual_vmax!r},",
                f"    sigma_n={self.model.sigma_n!r},",
                f"    iqr_n={self.model.iqr_n!r},",
                f"    percentile_n={self.model.percentile_n!r},",
                f"    power_gamma={self.model.power_gamma!r},",
                f"    smoothing_sigma_x={self.smoothing_x!r},",
                f"    smoothing_sigma_y={self.smoothing_y!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    axes_linewidth={self.axis_linewidth!r},",
                f"    figsize={tuple(self.figure.get_size_inches())!r},",
                ")",
                f"fig.suptitle({self._binning_title_text()!r})",
                "plt.show()",
                "",
            ]
        )

    def _waterfall_figure_script(self) -> str:
        source_indices = (
            self._waterfall_1d_source_indices()
            if self._waterfall_uses_1d_group()
            else [self.dataset_index]
        )
        loader_imports = {"plot_mdhisto_waterfall"}
        loader_expressions = []
        for index in source_indices:
            dataset = self.datasets[index]
            metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
            source = metadata.get("source_file")
            if not source:
                loader_expressions = []
                break
            if metadata.get("importer") == "powder_ins_csv":
                loader_imports.add("import_powder_ins_csv")
                loader_expressions.append(
                    f"import_powder_ins_csv({str(source)!r}, {dict(metadata.get('import_options', {}))!r})"
                )
            else:
                loader_imports.add("load_mantid_mdhisto_nxs")
                loader_expressions.append(
                    f"load_mantid_mdhisto_nxs({str(source)!r}, copy_metadata=False)"
                )
        if loader_expressions:
            loaders = ",\n    ".join(loader_expressions)
            data_lines = f"data = [\n    {loaders},\n]"
        else:
            data_lines = "data = ...  # Replace with one MDHistoData object or a list of 1D datasets"
        labels = [self.dataset_names[index] for index in source_indices]
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                f"from nfit import {', '.join(sorted(loader_imports))}",
                "",
                data_lines,
                "ax = plot_mdhisto_waterfall(",
                "    data,",
                f"    dataset_labels={labels!r},",
                f"    x_dim={self.data.axes[self.model.x_dim].name!r},",
                f"    waterfall_dim={self.data.axes[self.model.y_dim].name!r},",
                f"    channel={self.model.channel!r},",
                f"    selections={self._export_selections()!r},",
                f"    integrate_checks={self._export_integrate_checks()!r},",
                f"    waterfall_step={self.waterfall_step!r},",
                f"    coverage_threshold={self.waterfall_coverage_threshold!r},",
                f"    trace_offset={self.waterfall_offset!r},",
                f"    cmap={self.waterfall_cmap!r},",
                f"    color_range={(self.waterfall_color_min, self.waterfall_color_max)!r},",
                f"    reverse_colors={self.waterfall_reverse_colors!r},",
                f"    marker={self.marker!r},",
                f"    line_style={self.line_style!r},",
                f"    marker_size={self.marker_size!r},",
                f"    line_width={self.line_plot_width!r},",
                f"    marker_edge_width={self.marker_edge_width!r},",
                f"    marker_face={self.marker_face_color!r},",
                f"    show_errorbars={self.show_errorbars!r},",
                f"    errorbar_caps={self.show_errorbar_caps!r},",
                f"    errorbar_cap_size={self.errorbar_cap_size!r},",
                f"    show_zero_lines={self.waterfall_show_zero_lines!r},",
                f"    zero_line_color={self.waterfall_zero_color!r},",
                f"    zero_line_style={self.waterfall_zero_style!r},",
                f"    zero_line_width={self.waterfall_zero_width!r},",
                f"    show_model={self.show_fit!r},",
                f"    unmask_model={self.unmask_model!r},",
                f"    model_color={self.waterfall_model_color!r},",
                f"    model_line_width={self.fit_line_width!r},",
                f"    show_trace_labels={self.waterfall_show_trace_labels!r},",
                f"    trace_label_suffix={self.waterfall_trace_label_suffix!r},",
                f"    trace_label_font_size={self.waterfall_trace_label_font_size!r},",
                f"    trace_label_color={self.waterfall_trace_label_color!r},",
                f"    smoothing_sigma_x={self.smoothing_x!r},",
                f"    smoothing_sigma_waterfall={self.smoothing_y!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    axes_linewidth={self.axis_linewidth!r},",
                f"    figsize={tuple(self.figure.get_size_inches())!r},",
                ")",
                f"ax.figure.suptitle({self._binning_title_text()!r})",
                "plt.show()",
                "",
            ]
        )

    def _line_figure_script(self, data_line: str) -> str:
        return "\n".join(
            [
                "import matplotlib.pyplot as plt",
                "from nfit import load_mantid_mdhisto_nxs, plot_mdhisto_line",
                "",
                data_line,
                "ax = plot_mdhisto_line(",
                "    data,",
                f"    axis_dim={self.data.axes[self.model.x_dim].name!r},",
                f"    channel={self.model.channel!r},",
                f"    smoothing_sigma={self.smoothing_x!r},",
                ")",
                "for container in ax.containers:",
                "    for artist in getattr(container, 'lines', []):",
                "        if artist is not None:",
                "            try:",
                f"                artist.set_color({self.line_color!r})",
                "            except Exception:",
                "                pass",
                "for line in ax.lines:",
                f"    line.set_marker({self.marker!r})",
                f"    line.set_linestyle({'None' if self.line_style == 'none' else self.line_style!r})",
                f"    line.set_markersize({self.marker_size!r})",
                f"    line.set_linewidth({self.line_plot_width!r})",
                f"    line.set_markeredgewidth({self.marker_edge_width!r})",
                f"    line.set_markerfacecolor({self.marker_face_color!r})",
                f"    line.set_markeredgecolor({self.line_color!r})",
                f"    line.set_color({self.line_color!r})",
                f"ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width={self.axis_linewidth!r})",
                "ax.figure.set_size_inches(8.0, 6.0)",
                f"ax.figure.suptitle({self._binning_title_text()!r})",
                f"plt.rcParams.update({{'font.size': {self.font_size!r}}})",
                "plt.show()",
                "",
            ]
        )

    def _export_selections(self) -> dict[int, tuple[float, float]]:
        return {int(dim): (float(values[0]), float(values[1])) for dim, values in self.model.selections.items()}

    def _export_integrate_checks(self) -> dict[int, bool]:
        return {int(dim): bool(value) for dim, value in self.model.integrate_checks.items()}

    def _export_limits(self, axis_name: str) -> tuple[float, float]:
        values = self.ax_image.get_xlim() if axis_name == "x" else self.ax_image.get_ylim()
        return float(values[0]), float(values[1])

    def _build(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PySide6 import QtCore, QtGui, QtWidgets

        self.window = _make_data_viewer_window_class()(self)
        self.window.setWindowTitle("nfit Data Viewer")
        self.window.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        mode_bar = QtWidgets.QWidget()
        mode_layout = QtWidgets.QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(8, 6, 8, 6)
        mode_layout.addWidget(QtWidgets.QLabel("Visualization"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.setObjectName("data_viewer_mode_combo")
        self.view_mode_combo.addItems(
            ["Slice viewer", "Waterfall", "Tiled slices", "Volumetric"]
        )
        self.view_mode_combo.setToolTip(
            "Switch between standard slices, offset waterfall traces, tiled 2D slices, "
            "and volumetric rendering. Tiled-slice and volumetric modes require at least "
            "three dimensions with more than one bin."
        )
        self.view_mode_combo.currentIndexChanged.connect(self._set_view_mode)
        mode_layout.addWidget(self.view_mode_combo)
        self.open_new_viewer_button = QtWidgets.QPushButton("Open new viewer")
        self.open_new_viewer_button.setObjectName("data_viewer_open_new_button")
        self.open_new_viewer_button.setToolTip(
            "Open an independent data viewer initialized with the current dataset, "
            "visualization mode, axes, ranges, and styling."
        )
        self.open_new_viewer_button.clicked.connect(self.open_new_viewer)
        mode_layout.addWidget(self.open_new_viewer_button)
        mode_layout.addStretch(1)
        main_layout.addWidget(mode_bar)
        self.content_stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.content_stack, 1)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(8, 8, 8, 8)
        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
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
        self._suppress_matplotlib_coordinate_status()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setToolTip(
            "Interactive plot canvas. Move the cursor for coordinate readouts; use the toolbar or box tool to inspect slices."
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self.window)
        self.toolbar.setToolTip(
            "Matplotlib navigation toolbar for pan, zoom, home, configure, and save actions."
        )
        cursor_bar = QtWidgets.QWidget()
        cursor_layout = QtWidgets.QHBoxLayout(cursor_bar)
        cursor_layout.setContentsMargins(4, 0, 4, 0)
        cursor_layout.setSpacing(12)
        self.cursor_xy_label = QtWidgets.QLabel("(x, y) = (-, -)")
        self.cursor_hkle_label = QtWidgets.QLabel("(H, K, L, E) = (-, -, -, -)")
        self.cursor_q_label = QtWidgets.QLabel("|Q| = ? Å⁻¹")
        self.cursor_intensity_label = QtWidgets.QLabel("Signal = -")
        for label, width in (
            (self.cursor_xy_label, 230),
            (self.cursor_hkle_label, 350),
            (self.cursor_q_label, 150),
            (self.cursor_intensity_label, 220),
        ):
            label.setMinimumWidth(width)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        cursor_layout.addWidget(self.cursor_xy_label, 0)
        cursor_layout.addWidget(self.cursor_hkle_label, 0)
        cursor_layout.addWidget(self.cursor_q_label, 0)
        cursor_layout.addWidget(self.cursor_intensity_label, 1)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(cursor_bar)
        plot_layout.addWidget(self.canvas, 1)
        self._sync_cursor_visibility()

        controls = QtWidgets.QScrollArea()
        self.controls_scroll = controls
        controls.setWidgetResizable(True)
        controls.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls.setMinimumWidth(430)
        controls.setMaximumWidth(500)
        controls_widget = QtWidgets.QWidget()
        controls_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        controls.setWidget(controls_widget)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        dataset_group = QtWidgets.QGroupBox("Dataset")
        dataset_layout = QtWidgets.QGridLayout(dataset_group)
        dataset_layout.setHorizontalSpacing(6)
        dataset_layout.setVerticalSpacing(6)
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems(self.dataset_names)
        self.dataset_combo.setToolTip("Choose which loaded dataset is displayed in the viewer.")
        _expanding_combobox(self.dataset_combo)
        self.dataset_combo.currentIndexChanged.connect(self._set_dataset_index)
        dataset_layout.addWidget(QtWidgets.QLabel("Dataset"), 0, 0)
        dataset_layout.addWidget(self.dataset_combo, 0, 1)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(self.model.CHANNELS)
        self.channel_combo.setToolTip(
            "Choose the data channel to display, such as signal, combined_mask, file_mask, nfit_mask, fit, or residual."
        )
        _compact_combobox(self.channel_combo)
        self.channel_combo.setCurrentText(self.model.channel)
        self.channel_combo.currentTextChanged.connect(self._set_channel)
        self._sync_channel_combo()
        self.apply_masks_check = QtWidgets.QCheckBox("Apply Masks")
        self.apply_masks_check.setChecked(self.model.masked)
        self.apply_masks_check.setToolTip("Show data after applying file masks and nfit masks. Uncheck to inspect masked-out data.")
        self.apply_masks_check.toggled.connect(self._set_apply_masks)
        dataset_layout.addWidget(QtWidgets.QLabel("Channel"), 1, 0)
        dataset_layout.addWidget(self.channel_combo, 1, 1)
        dataset_layout.addWidget(self.apply_masks_check, 1, 2)
        self.show_fit_check = QtWidgets.QCheckBox("Show model")
        self.show_fit_check.setToolTip(
            "Show the current model beside the data (2D) or as a line under the data (1D). "
            "Enabled once current model channels or stored fit channels are available for this dataset."
        )
        self.show_fit_check.toggled.connect(self._set_show_fit)
        self.unmask_model_check = QtWidgets.QCheckBox("Unmask model")
        self.unmask_model_check.setObjectName("viewer_unmask_model_check")
        self.unmask_model_check.setToolTip(
            "Evaluate and draw the model outside data masks across the full plotted region. "
            "Data remain masked; residuals extend only where underlying data and uncertainties are finite."
        )
        self.unmask_model_check.toggled.connect(self._set_unmask_model)
        self.show_residual_check = QtWidgets.QCheckBox("Show residual")
        self.show_residual_check.setToolTip(
            "Also show the normalized residual: a third panel (2D) or axes below the data (1D)."
        )
        self.show_residual_check.toggled.connect(self._set_show_residual)
        dataset_layout.addWidget(self.show_fit_check, 2, 1)
        dataset_layout.addWidget(self.show_residual_check, 2, 2)
        dataset_layout.addWidget(self.unmask_model_check, 3, 1)
        coverage_label = QtWidgets.QLabel("Coverage")
        self.coverage_threshold_label = coverage_label
        self.coverage_threshold_spin = _make_float_spinbox(0.0, 1.0)
        self.coverage_threshold_spin.setObjectName("viewer_coverage_threshold")
        self.coverage_threshold_spin.setDecimals(3)
        self.coverage_threshold_spin.setSingleStep(0.05)
        self.coverage_threshold_spin.setMaximumWidth(82)
        self.coverage_threshold_spin.setValue(self.coverage_threshold)
        coverage_tooltip = (
            "Mask slice pixels and histogram-tool reductions whose measured support "
            "is below this fraction of the requested integration volume."
        )
        coverage_label.setToolTip(coverage_tooltip)
        self.coverage_threshold_spin.setToolTip(coverage_tooltip)
        self.coverage_threshold_spin.valueChanged.connect(
            self._set_coverage_threshold
        )
        dataset_layout.addWidget(coverage_label, 4, 0)
        dataset_layout.addWidget(self.coverage_threshold_spin, 4, 1)
        self.residual_split_label = QtWidgets.QLabel()
        self.residual_split_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.residual_split_slider.setMinimumWidth(80)
        self.residual_split_slider.setRange(10, 70)
        self.residual_split_slider.setSingleStep(1)
        self.residual_split_slider.setPageStep(5)
        self.residual_split_slider.setValue(self.residual_percent)
        self.residual_split_slider.setToolTip(
            "Vertical position of the separator between the data and residual axes."
        )
        self.residual_split_slider.valueChanged.connect(self._set_residual_percent)
        dataset_layout.addWidget(self.residual_split_label, 5, 0)
        dataset_layout.addWidget(self.residual_split_slider, 5, 1, 1, 2)
        dataset_layout.setColumnStretch(1, 1)
        controls_layout.addWidget(dataset_group)

        axes_group = QtWidgets.QGroupBox("Displayed axes")
        self.axes_group = axes_group
        axes_layout = QtWidgets.QGridLayout(axes_group)
        axes_layout.setHorizontalSpacing(6)
        axes_layout.setVerticalSpacing(5)
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self._sync_axis_combos(rebuild=True)
        self.x_combo.setToolTip("Choose the dataset axis shown horizontally.")
        self.y_combo.setToolTip("Choose the dataset axis shown vertically. For 1D data this control is hidden.")
        _compact_combobox(self.x_combo)
        _compact_combobox(self.y_combo)
        self.x_combo.currentIndexChanged.connect(lambda index: self._set_display_dim("x", index))
        self.y_combo.currentIndexChanged.connect(lambda index: self._set_display_dim("y", index))
        axis_selector_widget = QtWidgets.QWidget()
        self.axis_selector_widget = axis_selector_widget
        axis_selector_layout = QtWidgets.QHBoxLayout(axis_selector_widget)
        axis_selector_layout.setContentsMargins(0, 0, 0, 0)
        axis_selector_layout.setSpacing(6)
        axis_selector_layout.addWidget(QtWidgets.QLabel("x"))
        axis_selector_layout.addWidget(self.x_combo)
        self._axis_y_label = QtWidgets.QLabel("y")
        axis_selector_layout.addWidget(self._axis_y_label)
        axis_selector_layout.addWidget(self.y_combo)
        axis_selector_layout.addStretch(1)
        axes_layout.addWidget(axis_selector_widget, 0, 0, 1, 4)
        self.x_min_spin = _make_float_spinbox()
        self.x_max_spin = _make_float_spinbox()
        self.y_min_spin = _make_float_spinbox()
        self.y_max_spin = _make_float_spinbox()
        self.x_reset_button = QtWidgets.QPushButton("Reset")
        self.y_reset_button = QtWidgets.QPushButton("Reset")
        self.x_min_spin.setToolTip("Lower displayed limit for the horizontal axis.")
        self.x_max_spin.setToolTip("Upper displayed limit for the horizontal axis.")
        self.y_min_spin.setToolTip("Lower displayed limit for the vertical axis.")
        self.y_max_spin.setToolTip("Upper displayed limit for the vertical axis.")
        self.x_reset_button.setToolTip("Reset the horizontal axis limits to the full displayed data range.")
        self.y_reset_button.setToolTip("Reset the vertical axis limits to the full displayed data range.")
        self.x_reset_button.setMaximumWidth(64)
        self.y_reset_button.setMaximumWidth(64)
        self.x_min_spin.valueChanged.connect(lambda _value: self._set_view_limits("x"))
        self.x_max_spin.valueChanged.connect(lambda _value: self._set_view_limits("x"))
        self.y_min_spin.valueChanged.connect(lambda _value: self._set_view_limits("y"))
        self.y_max_spin.valueChanged.connect(lambda _value: self._set_view_limits("y"))
        self.x_reset_button.clicked.connect(lambda: self._reset_view_limits("x"))
        self.y_reset_button.clicked.connect(lambda: self._reset_view_limits("y"))
        axes_layout.addWidget(QtWidgets.QLabel("min"), 1, 1)
        axes_layout.addWidget(QtWidgets.QLabel("max"), 1, 2)
        axes_layout.addWidget(QtWidgets.QLabel("x limits"), 2, 0)
        axes_layout.addWidget(self.x_min_spin, 2, 1)
        axes_layout.addWidget(self.x_max_spin, 2, 2)
        axes_layout.addWidget(self.x_reset_button, 2, 3)
        axes_layout.addWidget(QtWidgets.QLabel("y limits"), 3, 0)
        axes_layout.addWidget(self.y_min_spin, 3, 1)
        axes_layout.addWidget(self.y_max_spin, 3, 2)
        axes_layout.addWidget(self.y_reset_button, 3, 3)
        axes_layout.setColumnStretch(1, 1)
        axes_layout.setColumnStretch(2, 1)
        controls_layout.addWidget(axes_group)

        tiled_group = QtWidgets.QGroupBox("Tiled slices")
        self.tiled_group = tiled_group
        tiled_layout = QtWidgets.QGridLayout(tiled_group)
        tiled_layout.setHorizontalSpacing(6)
        tiled_layout.setVerticalSpacing(6)
        self.tile_dim_combo = QtWidgets.QComboBox()
        self.tile_dim_combo.setToolTip(
            "Choose the third dimension whose coarse slices are arranged as panels."
        )
        _expanding_combobox(self.tile_dim_combo)
        self.tile_dim_combo.currentIndexChanged.connect(self._set_tile_dimension)
        tiled_layout.addWidget(QtWidgets.QLabel("Third dimension"), 0, 0)
        tiled_layout.addWidget(self.tile_dim_combo, 0, 1, 1, 3)

        self.tile_range_low_spin = _make_float_spinbox()
        self.tile_range_high_spin = _make_float_spinbox()
        self.tile_range_low_spin.setToolTip(
            "Lowest third-axis bin center included in the tiled panels."
        )
        self.tile_range_high_spin.setToolTip(
            "Highest third-axis bin center included in the tiled panels."
        )
        self.tile_range_low_spin.valueChanged.connect(self._set_tile_range)
        self.tile_range_high_spin.valueChanged.connect(self._set_tile_range)
        tiled_layout.addWidget(QtWidgets.QLabel("Range low"), 1, 0)
        tiled_layout.addWidget(self.tile_range_low_spin, 1, 1)
        tiled_layout.addWidget(QtWidgets.QLabel("Range high"), 1, 2)
        tiled_layout.addWidget(self.tile_range_high_spin, 1, 3)

        self.tile_step_spin = _make_float_spinbox(1.0e-9, 1.0e12)
        self.tile_step_spin.setDecimals(8)
        self.tile_step_spin.setToolTip(
            "Width of each coarse bin along the third dimension. Each bin becomes one 2D panel."
        )
        self.tile_step_spin.valueChanged.connect(self._set_tile_step)
        self.tile_step_auto_check = QtWidgets.QCheckBox("Auto (up to 9)")
        self.tile_step_auto_check.setChecked(self.tile_step_auto)
        self.tile_step_auto_check.setToolTip(
            "Choose a bin width that makes nine panels, or one panel per value when fewer than nine are available."
        )
        self.tile_step_auto_check.toggled.connect(self._set_tile_step_auto)
        tiled_layout.addWidget(QtWidgets.QLabel("Step size"), 2, 0)
        tiled_layout.addWidget(self.tile_step_spin, 2, 1)
        tiled_layout.addWidget(self.tile_step_auto_check, 2, 2, 1, 2)
        self.tile_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tile_step_slider.setRange(0, 1000)
        self.tile_step_slider.setToolTip(
            "Adjust the third-axis step from one native bin to the selected range span."
        )
        self.tile_step_slider.valueChanged.connect(self._set_tile_step_from_slider)
        tiled_layout.addWidget(self.tile_step_slider, 3, 0, 1, 4)
        tiled_layout.setColumnStretch(1, 1)
        tiled_layout.setColumnStretch(3, 1)
        controls_layout.addWidget(tiled_group)
        self._sync_tile_controls(reset_range=True)

        self.hidden_group = QtWidgets.QGroupBox("Integrated Axes")
        self.hidden_layout = QtWidgets.QVBoxLayout(self.hidden_group)
        controls_layout.addWidget(self.hidden_group)
        self._rebuild_hidden_axis_controls()

        color_group = QtWidgets.QGroupBox("Color")
        self.color_group = color_group
        color_layout = QtWidgets.QGridLayout(color_group)
        color_layout.setHorizontalSpacing(6)
        color_layout.setVerticalSpacing(6)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(self.model.COLORMAPS)
        self.cmap_combo.setToolTip("Choose the colormap used for 2D image data.")
        _compact_combobox(self.cmap_combo)
        self.cmap_combo.setCurrentText(self.model.cmap)
        self.cmap_combo.currentTextChanged.connect(self._set_cmap)
        self.cmap_reverse_button = QtWidgets.QPushButton("Reverse")
        self.cmap_reverse_button.setToolTip("Reverse the selected colormap.")
        self.cmap_reverse_button.clicked.connect(self._toggle_cmap_reverse)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(self.model.COLOR_SCALES)
        self.scale_combo.setToolTip("Choose linear, logarithmic, symmetric logarithmic, or power color scaling.")
        _compact_combobox(self.scale_combo)
        self.scale_combo.setCurrentText(self.model.color_scale)
        self.scale_combo.currentTextChanged.connect(self._set_color_scale)
        self.limits_combo = QtWidgets.QComboBox()
        self.limits_combo.addItems(self.model.AUTO_LIMITS)
        self.limits_combo.setToolTip("Choose how automatic color limits are estimated from the displayed data.")
        _compact_combobox(self.limits_combo)
        self.limits_combo.setCurrentText(self.model.auto_limits)
        self.limits_combo.currentTextChanged.connect(self._set_auto_limits)
        self.autoscale_check = QtWidgets.QCheckBox("Autoscale")
        self.autoscale_check.setChecked(self.model.autoscale)
        self.autoscale_check.setToolTip("Automatically recompute color limits when the displayed data or selection changes.")
        self.autoscale_check.toggled.connect(self._set_autoscale)
        self.vmin_spin = _make_float_spinbox()
        self.vmax_spin = _make_float_spinbox()
        self.gamma_spin = _make_float_spinbox(1.0e-6, 20.0)
        self.gamma_spin.setValue(self.model.power_gamma)
        self.limit_n_spin = _make_float_spinbox(0.0, 50.0)
        self.limit_n_spin.setValue(self._current_limit_n())
        self.vmin_spin.setToolTip("Manual lower color limit when autoscale is off.")
        self.vmax_spin.setToolTip("Manual upper color limit when autoscale is off.")
        self.gamma_spin.setToolTip("Exponent used by power color scaling.")
        self.limit_n_spin.setToolTip("Width parameter used by the selected automatic color-limit rule.")
        self.vmin_spin.valueChanged.connect(lambda value: self._set_manual_limit("vmin", value))
        self.vmax_spin.valueChanged.connect(lambda value: self._set_manual_limit("vmax", value))
        self.gamma_spin.valueChanged.connect(self._set_power_gamma)
        self.limit_n_spin.valueChanged.connect(self._set_limit_n)
        self.gamma_label = QtWidgets.QLabel("gamma")
        self.limit_n_label = QtWidgets.QLabel("N")
        color_layout.addWidget(QtWidgets.QLabel("Colormap"), 0, 0)
        color_layout.addWidget(self.cmap_combo, 0, 1)
        color_layout.addWidget(self.cmap_reverse_button, 0, 2, 1, 2)
        color_layout.addWidget(QtWidgets.QLabel("Scale"), 1, 0)
        color_layout.addWidget(self.scale_combo, 1, 1)
        color_layout.addWidget(self.gamma_label, 1, 2)
        color_layout.addWidget(self.gamma_spin, 1, 3)
        color_layout.addWidget(QtWidgets.QLabel("Auto limits"), 2, 0)
        color_layout.addWidget(self.limits_combo, 2, 1)
        color_layout.addWidget(self.limit_n_label, 2, 2)
        color_layout.addWidget(self.limit_n_spin, 2, 3)
        color_layout.addWidget(self.autoscale_check, 3, 1, 1, 3)
        color_layout.setRowMinimumHeight(4, 6)
        color_layout.addWidget(QtWidgets.QLabel("vmin"), 5, 0)
        color_layout.addWidget(self.vmin_spin, 5, 1)
        color_layout.addWidget(QtWidgets.QLabel("vmax"), 5, 2)
        color_layout.addWidget(self.vmax_spin, 5, 3)
        self.gamma_label.setVisible(self.model.color_scale == "power")
        self.gamma_spin.setVisible(self.model.color_scale == "power")
        self._sync_limit_n_visibility()
        controls_layout.addWidget(color_group)

        smoothing_group = QtWidgets.QGroupBox("Plot smoothing")
        self.smoothing_group = smoothing_group
        smoothing_layout = QtWidgets.QGridLayout(smoothing_group)
        self.smoothing_x_spin = _make_float_spinbox(0.0, 100.0)
        self.smoothing_y_spin = _make_float_spinbox(0.0, 100.0)
        self.smoothing_x_spin.setObjectName("slice_smoothing_x_spin")
        self.smoothing_y_spin.setObjectName("slice_smoothing_y_spin")
        for axis_name, spin in (("X", self.smoothing_x_spin), ("Y", self.smoothing_y_spin)):
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
            spin.setSuffix(" bins")
            spin.setToolTip(
                f"Gaussian blur sigma along displayed {axis_name}, in bin widths. "
                "This changes plotting only; fitting, dataset values, and numerical exports remain unchanged."
            )
        self.smoothing_x_spin.valueChanged.connect(lambda value: self._set_plot_smoothing("x", value))
        self.smoothing_y_spin.valueChanged.connect(lambda value: self._set_plot_smoothing("y", value))
        smoothing_layout.addWidget(QtWidgets.QLabel("X sigma"), 0, 0)
        smoothing_layout.addWidget(self.smoothing_x_spin, 0, 1)
        self.smoothing_y_label = QtWidgets.QLabel("Y sigma")
        smoothing_layout.addWidget(self.smoothing_y_label, 1, 0)
        smoothing_layout.addWidget(self.smoothing_y_spin, 1, 1)
        smoothing_layout.setColumnStretch(1, 1)
        controls_layout.addWidget(smoothing_group)

        tools_group = QtWidgets.QGroupBox("Histogram box cuts")
        self.tools_group = tools_group
        tools_layout = QtWidgets.QGridLayout(tools_group)
        self.roi_button = QtWidgets.QPushButton("Box tool")
        self.roi_button.setCheckable(True)
        self.roi_button.setToolTip(
            "Toggle the rectangle tool used to populate inverse-variance weighted x/y profile cuts "
            "with propagated error bars."
        )
        self.roi_button.toggled.connect(self._set_roi_enabled)
        self.show_box_check = QtWidgets.QCheckBox("Show box tool")
        self.show_box_check.setChecked(False)
        self.show_box_check.setToolTip("Show or hide the rectangle selection tool.")
        self.show_box_check.toggled.connect(self._set_box_tool_visible)
        self.hist_axes_check = QtWidgets.QCheckBox("Show histogram axes")
        self.hist_axes_check.setChecked(False)
        self.hist_axes_check.setToolTip(
            "Show or hide x/y profile cuts beside the image. Each point is an inverse-variance "
            "weighted mean over the selected box with its propagated standard error."
        )
        self.hist_axes_check.toggled.connect(self._set_histogram_axes_visible)
        self.xcut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ycut_percent_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.xcut_percent_slider.setToolTip("Height of the horizontal cut panel as a percentage of the figure.")
        self.ycut_percent_slider.setToolTip("Width of the vertical cut panel as a percentage of the figure.")
        for slider, value in (
            (self.xcut_percent_slider, self.xcut_percent),
            (self.ycut_percent_slider, self.ycut_percent),
        ):
            slider.setRange(5, 45)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.setValue(value)
        self.xcut_percent_label = QtWidgets.QLabel()
        self.ycut_percent_label = QtWidgets.QLabel()
        self.roi_x_center_spin = _make_float_spinbox()
        self.roi_x_width_spin = _make_float_spinbox(0.0, 1.0e12)
        self.roi_y_center_spin = _make_float_spinbox()
        self.roi_y_width_spin = _make_float_spinbox(0.0, 1.0e12)
        self.roi_x_center_spin.setToolTip("Center of the rectangle selection along the horizontal axis.")
        self.roi_x_width_spin.setToolTip("Width of the rectangle selection along the horizontal axis.")
        self.roi_y_center_spin.setToolTip("Center of the rectangle selection along the vertical axis.")
        self.roi_y_width_spin.setToolTip("Width of the rectangle selection along the vertical axis.")
        self.xcut_percent_slider.valueChanged.connect(self._set_xcut_percent)
        self.ycut_percent_slider.valueChanged.connect(self._set_ycut_percent)
        self.roi_x_center_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_x_width_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_y_center_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        self.roi_y_width_spin.valueChanged.connect(lambda _value: self._set_roi_from_controls())
        tools_layout.addWidget(self.roi_button, 0, 0, 1, 4)
        tools_layout.addWidget(self.show_box_check, 1, 0, 1, 2)
        tools_layout.addWidget(self.hist_axes_check, 1, 2, 1, 2)
        tools_layout.addWidget(QtWidgets.QLabel("x center"), 2, 0)
        tools_layout.addWidget(self.roi_x_center_spin, 2, 1)
        tools_layout.addWidget(QtWidgets.QLabel("width"), 2, 2)
        tools_layout.addWidget(self.roi_x_width_spin, 2, 3)
        tools_layout.addWidget(QtWidgets.QLabel("y center"), 3, 0)
        tools_layout.addWidget(self.roi_y_center_spin, 3, 1)
        tools_layout.addWidget(QtWidgets.QLabel("width"), 3, 2)
        tools_layout.addWidget(self.roi_y_width_spin, 3, 3)
        tools_layout.addWidget(self.xcut_percent_label, 4, 0)
        tools_layout.addWidget(self.xcut_percent_slider, 4, 1, 1, 3)
        tools_layout.addWidget(self.ycut_percent_label, 5, 0)
        tools_layout.addWidget(self.ycut_percent_slider, 5, 1, 1, 3)
        tools_layout.setColumnStretch(1, 1)
        tools_layout.setColumnStretch(3, 1)
        self._sync_histogram_panel_controls()
        controls_layout.addWidget(tools_group)

        line_group = QtWidgets.QGroupBox("Line plot")
        self.line_group = line_group
        line_layout = QtWidgets.QGridLayout(line_group)
        line_layout.setHorizontalSpacing(6)
        line_layout.setVerticalSpacing(6)
        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItems(_MARKER_OPTIONS.keys())
        self.marker_combo.setCurrentText("circle")
        self.marker_combo.setToolTip("Marker shape used for 1D line plots.")
        _compact_combobox(self.marker_combo)
        self.marker_combo.currentTextChanged.connect(self._set_marker)
        self.line_style_combo = QtWidgets.QComboBox()
        self.line_style_combo.addItems(_LINE_STYLE_OPTIONS.keys())
        self.line_style_combo.setCurrentText("none")
        self.line_style_combo.setToolTip("Line style connecting points in 1D line plots.")
        _compact_combobox(self.line_style_combo)
        self.line_style_combo.currentTextChanged.connect(self._set_line_style)
        self.marker_size_spin = _make_float_spinbox(0.0, 50.0)
        self.marker_size_spin.setValue(self.marker_size)
        self.marker_size_spin.setToolTip("Size of markers in 1D line plots.")
        self.marker_size_spin.valueChanged.connect(self._set_marker_size)
        self.line_plot_width_spin = _make_float_spinbox(0.0, 20.0)
        self.line_plot_width_spin.setValue(self.line_plot_width)
        self.line_plot_width_spin.setToolTip("Width of the plotted line in 1D line plots.")
        self.line_plot_width_spin.valueChanged.connect(self._set_line_plot_width)
        self.marker_edge_width_spin = _make_float_spinbox(0.0, 20.0)
        self.marker_edge_width_spin.setValue(self.marker_edge_width)
        self.marker_edge_width_spin.setToolTip("Width of marker outlines in 1D line plots.")
        self.marker_edge_width_spin.valueChanged.connect(self._set_marker_edge_width)
        self.marker_face_color_combo = QtWidgets.QComboBox()
        self.marker_face_color_combo.addItems(
            ["none", "outline", *[name for name in _COLOR_OPTIONS if name != "none"]]
        )
        self.marker_face_color_combo.setCurrentText("none")
        self.marker_face_color_combo.setToolTip(
            "Fill markers with no color, their outline color, or one common color."
        )
        _compact_combobox(self.marker_face_color_combo)
        self.marker_face_color_combo.currentTextChanged.connect(self._set_marker_face_color)
        self.line_color_combo = QtWidgets.QComboBox()
        self.line_color_combo.addItems(_COLOR_OPTIONS.keys())
        self.line_color_combo.setCurrentText("blue")
        self.line_color_combo.setToolTip("Line and marker-edge color for 1D line plots.")
        _compact_combobox(self.line_color_combo)
        self.line_color_combo.currentTextChanged.connect(self._set_line_color)
        self.show_errorbars_check = QtWidgets.QCheckBox("Show errorbars")
        self.show_errorbars_check.setChecked(self.show_errorbars)
        self.show_errorbars_check.setToolTip("Show uncertainty bars when the selected channel has errors.")
        self.show_errorbars_check.toggled.connect(self._set_show_errorbars)
        self.show_errorbar_caps_check = QtWidgets.QCheckBox("Endcaps")
        self.show_errorbar_caps_check.setChecked(self.show_errorbar_caps)
        self.show_errorbar_caps_check.setToolTip("Draw endcaps on error bars.")
        self.show_errorbar_caps_check.toggled.connect(self._set_show_errorbar_caps)
        self.errorbar_cap_size_spin = _make_float_spinbox(0.0, 30.0)
        self.errorbar_cap_size_spin.setValue(self.errorbar_cap_size)
        self.errorbar_cap_size_spin.setToolTip("Size of error-bar endcaps.")
        self.errorbar_cap_size_spin.valueChanged.connect(self._set_errorbar_cap_size)
        line_layout.addWidget(QtWidgets.QLabel("Marker"), 0, 0)
        line_layout.addWidget(self.marker_combo, 0, 1)
        line_layout.addWidget(QtWidgets.QLabel("Line"), 0, 2)
        line_layout.addWidget(self.line_style_combo, 0, 3)
        line_layout.addWidget(QtWidgets.QLabel("Marker size"), 1, 0)
        line_layout.addWidget(self.marker_size_spin, 1, 1)
        line_layout.addWidget(QtWidgets.QLabel("Linewidth"), 1, 2)
        line_layout.addWidget(self.line_plot_width_spin, 1, 3)
        line_layout.addWidget(QtWidgets.QLabel("Edge width"), 2, 0)
        line_layout.addWidget(self.marker_edge_width_spin, 2, 1)
        line_layout.addWidget(self.show_errorbars_check, 2, 2, 1, 2)
        line_layout.addWidget(QtWidgets.QLabel("Marker face"), 3, 0)
        line_layout.addWidget(self.marker_face_color_combo, 3, 1)
        line_layout.addWidget(QtWidgets.QLabel("Line/color"), 3, 2)
        line_layout.addWidget(self.line_color_combo, 3, 3)
        line_layout.addWidget(self.show_errorbar_caps_check, 4, 0, 1, 2)
        line_layout.addWidget(QtWidgets.QLabel("Cap size"), 4, 2)
        line_layout.addWidget(self.errorbar_cap_size_spin, 4, 3)
        self.fit_line_color_combo = QtWidgets.QComboBox()
        self.fit_line_color_combo.addItems([name for name in _COLOR_OPTIONS if name != "none"])
        self.fit_line_color_combo.setCurrentText(_option_name(_COLOR_OPTIONS, self.fit_line_color))
        self.fit_line_color_combo.setToolTip("Color used for model overlays in 1D plots.")
        _compact_combobox(self.fit_line_color_combo)
        self.fit_line_color_combo.currentTextChanged.connect(self._set_fit_line_color)
        self.fit_line_width_spin = _make_float_spinbox(0.1, 20.0)
        self.fit_line_width_spin.setValue(self.fit_line_width)
        self.fit_line_width_spin.setToolTip("Line width used for model overlays in 1D plots.")
        self.fit_line_width_spin.valueChanged.connect(self._set_fit_line_width)
        self.fit_line_color_label = QtWidgets.QLabel("Model color")
        self.fit_line_width_label = QtWidgets.QLabel("Model width")
        line_layout.addWidget(self.fit_line_color_label, 5, 0)
        line_layout.addWidget(self.fit_line_color_combo, 5, 1)
        line_layout.addWidget(self.fit_line_width_label, 5, 2)
        line_layout.addWidget(self.fit_line_width_spin, 5, 3)
        controls_layout.addWidget(line_group)

        waterfall_group = QtWidgets.QGroupBox("Waterfall traces")
        self.waterfall_group = waterfall_group
        waterfall_layout = QtWidgets.QGridLayout(waterfall_group)
        waterfall_layout.setHorizontalSpacing(6)
        waterfall_layout.setVerticalSpacing(6)
        self.waterfall_source_label = QtWidgets.QLabel()
        self.waterfall_source_label.setWordWrap(True)
        self.waterfall_source_label.setToolTip(
            "For multidimensional data, traces are coarse bins along the selected waterfall axis. "
            "For compatible 1D data, every dataset contributes one trace."
        )
        waterfall_layout.addWidget(self.waterfall_source_label, 0, 0, 1, 4)

        self.waterfall_step_label = QtWidgets.QLabel("Bin width")
        self.waterfall_step_spin = _make_float_spinbox(1.0e-9, 1.0e12)
        self.waterfall_step_spin.setDecimals(8)
        self.waterfall_step_spin.setValue(self.waterfall_step)
        self.waterfall_step_spin.setToolTip(
            "Width of each coarse bin along the waterfall axis. Bins are reduced to "
            "inverse-variance weighted mean traces with propagated one-sigma errors."
        )
        self.waterfall_step_spin.valueChanged.connect(self._set_waterfall_step)
        self.waterfall_step_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.waterfall_step_slider.setRange(0, 1000)
        self.waterfall_step_slider.setToolTip(
            "Adjust the bin width from one native waterfall-axis bin to the "
            "full waterfall-axis span."
        )
        self.waterfall_step_slider.valueChanged.connect(
            self._set_waterfall_step_from_slider
        )
        self.waterfall_step_auto_check = QtWidgets.QCheckBox("Auto (~10)")
        self.waterfall_step_auto_check.setChecked(self.waterfall_step_auto)
        self.waterfall_step_auto_check.setToolTip(
            "Choose a waterfall-axis bin width that produces approximately ten traces."
        )
        self.waterfall_step_auto_check.toggled.connect(self._set_waterfall_step_auto)
        waterfall_layout.addWidget(self.waterfall_step_label, 1, 0)
        waterfall_layout.addWidget(self.waterfall_step_spin, 1, 1)
        waterfall_layout.addWidget(self.waterfall_step_auto_check, 1, 2, 1, 2)
        waterfall_layout.addWidget(self.waterfall_step_slider, 2, 0, 1, 4)

        waterfall_coverage_label = QtWidgets.QLabel("Coverage")
        self.waterfall_coverage_threshold_spin = _make_float_spinbox(0.0, 1.0)
        self.waterfall_coverage_threshold_spin.setObjectName(
            "waterfall_coverage_threshold"
        )
        self.waterfall_coverage_threshold_spin.setDecimals(3)
        self.waterfall_coverage_threshold_spin.setSingleStep(0.05)
        self.waterfall_coverage_threshold_spin.setMaximumWidth(82)
        self.waterfall_coverage_threshold_spin.setValue(
            self.waterfall_coverage_threshold
        )
        waterfall_coverage_tooltip = (
            "Mask coarse waterfall samples whose measured support is below this "
            "fraction of the requested trace-bin volume."
        )
        waterfall_coverage_label.setToolTip(waterfall_coverage_tooltip)
        self.waterfall_coverage_threshold_spin.setToolTip(
            waterfall_coverage_tooltip
        )
        self.waterfall_coverage_threshold_spin.valueChanged.connect(
            self._set_waterfall_coverage_threshold
        )
        waterfall_layout.addWidget(waterfall_coverage_label, 3, 0)
        waterfall_layout.addWidget(
            self.waterfall_coverage_threshold_spin, 3, 1
        )

        self.waterfall_offset_spin = _make_float_spinbox(0.0, 1.0e12)
        self.waterfall_offset_spin.setDecimals(8)
        self.waterfall_offset_spin.setValue(self.waterfall_offset)
        self.waterfall_offset_spin.setToolTip(
            "Vertical offset between adjacent traces, in the displayed channel units."
        )
        self.waterfall_offset_spin.valueChanged.connect(self._set_waterfall_offset)
        self.waterfall_offset_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.waterfall_offset_slider.setRange(0, 1000)
        self.waterfall_offset_slider.setToolTip(
            "Adjust the trace offset from zero to the largest absolute data "
            "value in the prepared waterfall traces."
        )
        self.waterfall_offset_slider.valueChanged.connect(
            self._set_waterfall_offset_from_slider
        )
        self.waterfall_offset_auto_check = QtWidgets.QCheckBox("Auto (half max)")
        self.waterfall_offset_auto_check.setChecked(self.waterfall_offset_auto)
        self.waterfall_offset_auto_check.setToolTip(
            "Set the trace offset to half the largest absolute intensity among the prepared traces."
        )
        self.waterfall_offset_auto_check.toggled.connect(self._set_waterfall_offset_auto)
        waterfall_layout.addWidget(QtWidgets.QLabel("Trace offset"), 4, 0)
        waterfall_layout.addWidget(self.waterfall_offset_spin, 4, 1)
        waterfall_layout.addWidget(self.waterfall_offset_auto_check, 4, 2, 1, 2)
        waterfall_layout.addWidget(self.waterfall_offset_slider, 5, 0, 1, 4)

        self.waterfall_cmap_combo = QtWidgets.QComboBox()
        self.waterfall_cmap_combo.addItems(_WATERFALL_COLORMAPS)
        self.waterfall_cmap_combo.setCurrentText(self.waterfall_cmap)
        self.waterfall_cmap_combo.setToolTip(
            "Color sequence sampled uniformly across the displayed waterfall traces."
        )
        _compact_combobox(self.waterfall_cmap_combo)
        self.waterfall_cmap_combo.currentTextChanged.connect(self._set_waterfall_cmap)
        self.waterfall_reverse_check = QtWidgets.QCheckBox("Reverse")
        self.waterfall_reverse_check.setChecked(self.waterfall_reverse_colors)
        self.waterfall_reverse_check.setToolTip("Reverse the order of colors sampled from the sequence.")
        self.waterfall_reverse_check.toggled.connect(self._set_waterfall_reverse_colors)
        waterfall_layout.addWidget(QtWidgets.QLabel("Colors"), 6, 0)
        waterfall_layout.addWidget(self.waterfall_cmap_combo, 6, 1)
        waterfall_layout.addWidget(self.waterfall_reverse_check, 6, 2, 1, 2)

        self.waterfall_color_range_slider = _DualRangeSlider()
        self.waterfall_color_range_slider.set_values(0, 1000)
        self.waterfall_color_range_slider.setToolTip(
            "Drag the two handles to exclude colors near either end of a "
            "continuous colormap."
        )
        self.waterfall_color_range_slider.changed.connect(
            self._set_waterfall_color_range
        )
        self.waterfall_color_range_label = QtWidgets.QLabel("Color range")
        waterfall_layout.addWidget(self.waterfall_color_range_label, 7, 0)
        waterfall_layout.addWidget(
            self.waterfall_color_range_slider,
            7,
            1,
            1,
            3,
        )

        self.waterfall_zero_check = QtWidgets.QCheckBox("Zero references")
        self.waterfall_zero_check.setChecked(self.waterfall_show_zero_lines)
        self.waterfall_zero_check.setToolTip(
            "Draw a horizontal zero-intensity reference at the offset baseline of every trace."
        )
        self.waterfall_zero_check.toggled.connect(self._set_waterfall_zero_lines)
        self.waterfall_zero_color_combo = QtWidgets.QComboBox()
        self.waterfall_zero_color_combo.addItems(
            [name for name in _COLOR_OPTIONS if name != "none"]
        )
        self.waterfall_zero_color_combo.setCurrentText(
            _option_name(_COLOR_OPTIONS, self.waterfall_zero_color)
        )
        self.waterfall_zero_color_combo.setToolTip("Color of the per-trace zero reference lines.")
        _compact_combobox(self.waterfall_zero_color_combo)
        self.waterfall_zero_color_combo.currentTextChanged.connect(
            self._set_waterfall_zero_color
        )
        waterfall_layout.addWidget(self.waterfall_zero_check, 8, 0, 1, 2)
        waterfall_layout.addWidget(self.waterfall_zero_color_combo, 8, 2, 1, 2)

        self.waterfall_zero_style_combo = QtWidgets.QComboBox()
        self.waterfall_zero_style_combo.addItems(
            [name for name in _LINE_STYLE_OPTIONS if name != "none"]
        )
        self.waterfall_zero_style_combo.setCurrentText(
            _option_name(_LINE_STYLE_OPTIONS, self.waterfall_zero_style)
        )
        self.waterfall_zero_style_combo.setToolTip("Line style of the zero references.")
        _compact_combobox(self.waterfall_zero_style_combo)
        self.waterfall_zero_style_combo.currentTextChanged.connect(
            self._set_waterfall_zero_style
        )
        self.waterfall_zero_width_spin = _make_float_spinbox(0.1, 20.0)
        self.waterfall_zero_width_spin.setValue(self.waterfall_zero_width)
        self.waterfall_zero_width_spin.setToolTip("Line width of the zero references.")
        self.waterfall_zero_width_spin.valueChanged.connect(
            self._set_waterfall_zero_width
        )
        waterfall_layout.addWidget(QtWidgets.QLabel("Reference style"), 9, 0)
        waterfall_layout.addWidget(self.waterfall_zero_style_combo, 9, 1)
        waterfall_layout.addWidget(QtWidgets.QLabel("Width"), 9, 2)
        waterfall_layout.addWidget(self.waterfall_zero_width_spin, 9, 3)

        self.waterfall_model_color_combo = QtWidgets.QComboBox()
        self.waterfall_model_color_combo.addItems(
            ["match traces", *[name for name in _COLOR_OPTIONS if name != "none"]]
        )
        self.waterfall_model_color_combo.setCurrentText("match traces")
        self.waterfall_model_color_combo.setToolTip(
            "Use each trace's color for its model line, or draw every model trace in one selected color."
        )
        _compact_combobox(self.waterfall_model_color_combo)
        self.waterfall_model_color_combo.currentTextChanged.connect(
            self._set_waterfall_model_color
        )
        self.waterfall_trace_labels_check = QtWidgets.QCheckBox("Trace labels")
        self.waterfall_trace_labels_check.setChecked(self.waterfall_show_trace_labels)
        self.waterfall_trace_labels_check.setToolTip(
            "Label each multidimensional trace by its waterfall-axis center, "
            "or each 1D trace by its dataset name."
        )
        self.waterfall_trace_labels_check.toggled.connect(
            self._set_waterfall_trace_labels
        )
        waterfall_layout.addWidget(QtWidgets.QLabel("Model colors"), 10, 0)
        waterfall_layout.addWidget(self.waterfall_model_color_combo, 10, 1)
        waterfall_layout.addWidget(self.waterfall_trace_labels_check, 10, 2, 1, 2)

        self.waterfall_trace_label_suffix_edit = QtWidgets.QLineEdit()
        self.waterfall_trace_label_suffix_edit.setText(
            self.waterfall_trace_label_suffix
        )
        self.waterfall_trace_label_suffix_edit.setPlaceholderText(
            "Optional replacement for generated axis units"
        )
        self.waterfall_trace_label_suffix_edit.setToolTip(
            "For coordinate-derived traces, replace the default axis units "
            "(for example meV or r.l.u.) with this text. For grouped 1D "
            "datasets, append it verbatim to the dataset label."
        )
        self.waterfall_trace_label_suffix_edit.textChanged.connect(
            self._set_waterfall_trace_label_suffix
        )
        waterfall_layout.addWidget(QtWidgets.QLabel("Label suffix"), 11, 0)
        waterfall_layout.addWidget(
            self.waterfall_trace_label_suffix_edit,
            11,
            1,
            1,
            3,
        )

        self.waterfall_trace_label_font_size_spin = _make_float_spinbox(4.0, 48.0)
        self.waterfall_trace_label_font_size_spin.setDecimals(1)
        self.waterfall_trace_label_font_size_spin.setValue(
            self.waterfall_trace_label_font_size
        )
        self.waterfall_trace_label_font_size_spin.setToolTip(
            "Font size used only for waterfall trace labels."
        )
        self.waterfall_trace_label_font_size_spin.valueChanged.connect(
            self._set_waterfall_trace_label_font_size
        )
        self.waterfall_trace_label_color_combo = QtWidgets.QComboBox()
        self.waterfall_trace_label_color_combo.addItems(
            ["match traces", *[name for name in _COLOR_OPTIONS if name != "none"]]
        )
        self.waterfall_trace_label_color_combo.setCurrentText("match traces")
        self.waterfall_trace_label_color_combo.setToolTip(
            "Match each label to its trace, or use one common font color for all labels."
        )
        _compact_combobox(self.waterfall_trace_label_color_combo)
        self.waterfall_trace_label_color_combo.currentTextChanged.connect(
            self._set_waterfall_trace_label_color
        )
        waterfall_layout.addWidget(QtWidgets.QLabel("Label size"), 12, 0)
        waterfall_layout.addWidget(
            self.waterfall_trace_label_font_size_spin,
            12,
            1,
        )
        waterfall_layout.addWidget(QtWidgets.QLabel("Label color"), 12, 2)
        waterfall_layout.addWidget(
            self.waterfall_trace_label_color_combo,
            12,
            3,
        )

        waterfall_layout.setColumnStretch(1, 1)
        waterfall_layout.setColumnStretch(3, 1)
        controls_layout.addWidget(waterfall_group)

        figure_group = QtWidgets.QGroupBox("Figure")
        figure_layout = QtWidgets.QGridLayout(figure_group)
        figure_layout.setHorizontalSpacing(6)
        figure_layout.setVerticalSpacing(6)
        self.font_size_spin = _make_float_spinbox(4.0, 48.0)
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setValue(self.font_size)
        self.font_size_spin.setToolTip("Base font size used for axes labels, ticks, titles, and readouts.")
        self.font_size_spin.valueChanged.connect(self._set_font_size)
        self.line_width_spin = _make_float_spinbox(0.1, 10.0)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setValue(self.axis_linewidth)
        self.line_width_spin.setToolTip("Width of figure axes and frame lines.")
        self.line_width_spin.valueChanged.connect(self._set_axis_linewidth)
        self.show_binning_title_check = QtWidgets.QCheckBox("Show other-axis binning above plot")
        self.show_binning_title_check.setObjectName("show_binning_title")
        self.show_binning_title_check.setChecked(self.show_binning_title)
        self.show_binning_title_check.setToolTip(
            "Display the selected or integrated ranges of every non-plotted axis above the figure, including coordinate labels and units."
        )
        self.show_binning_title_check.toggled.connect(self._set_show_binning_title)
        self.copy_figure_button = QtWidgets.QPushButton("Copy figure")
        self.save_plot_button = QtWidgets.QPushButton("Save plot")
        self.copy_script_button = QtWidgets.QPushButton("Copy script")
        self.save_script_button = QtWidgets.QPushButton("Save script")
        for button in (
            self.copy_figure_button,
            self.save_plot_button,
            self.copy_script_button,
            self.save_script_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.copy_figure_button.setToolTip("Copy the current figure image to the clipboard.")
        self.save_plot_button.setToolTip(
            "Store the current view as an editable plot in this workspace."
        )
        self.copy_script_button.setToolTip("Copy a Python script that recreates the current viewer plot.")
        self.save_script_button.setToolTip("Save a Python script that recreates the current viewer plot.")
        self.save_plot_button.setEnabled(False)
        self.copy_figure_button.clicked.connect(self.copy_figure_to_clipboard)
        self.save_plot_button.clicked.connect(
            lambda: self._save_plot_callback() if self._save_plot_callback else None
        )
        self.copy_script_button.clicked.connect(self.copy_script_to_clipboard)
        self.save_script_button.clicked.connect(self.save_script)
        figure_layout.addWidget(QtWidgets.QLabel("Font size"), 0, 0)
        figure_layout.addWidget(self.font_size_spin, 0, 1)
        figure_layout.addWidget(QtWidgets.QLabel("Linewidth"), 0, 2)
        figure_layout.addWidget(self.line_width_spin, 0, 3)
        figure_layout.addWidget(self.show_binning_title_check, 1, 0, 1, 4)
        figure_layout.addWidget(self.copy_figure_button, 2, 0, 1, 2)
        figure_layout.addWidget(self.save_plot_button, 2, 2, 1, 2)
        figure_layout.addWidget(self.copy_script_button, 3, 0, 1, 2)
        figure_layout.addWidget(self.save_script_button, 3, 2, 1, 2)
        controls_layout.addWidget(figure_group)
        controls_layout.addStretch(1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(plot_panel)
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([970, 430])
        self.content_stack.addWidget(splitter)

        self._plot_layout_mode = ("standard", 1)
        self._create_rectangle_selector()
        self._connect_view_limit_callbacks()
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self.window)
        copy_shortcut.activated.connect(self.copy_figure_to_clipboard)
        self.save_project_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Save,
            self.window,
        )
        self.save_project_shortcut.setEnabled(False)
        self.save_project_shortcut.activated.connect(
            lambda: (
                self._save_project_callback()
                if self._save_project_callback is not None
                else None
            )
        )
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self.window)
        self.close_shortcut.activated.connect(self.window.close)
        self._sync_fit_channel_controls()
        self._sync_view_mode_availability()

    def _sync_view_mode_availability(self) -> None:
        from .qt_volume_viewer import supports_volume_view

        waterfall_available = (
            isinstance(self.data, MDHistoData)
            and sum(size > 1 for size in self.data.shape) >= 1
        )
        waterfall_item = self.view_mode_combo.model().item(1)
        if waterfall_item is not None:
            waterfall_item.setEnabled(waterfall_available)
            waterfall_item.setToolTip(
                "Stack coarse bins or compatible 1D datasets as offset traces."
                if waterfall_available
                else "Waterfall mode currently requires gridded MDHisto data."
            )
        tiled_available = (
            isinstance(self.data, MDHistoData)
            and sum(size > 1 for size in self.data.shape) >= 3
        )
        tiled_item = self.view_mode_combo.model().item(2)
        if tiled_item is not None:
            tiled_item.setEnabled(tiled_available)
            tiled_item.setToolTip(
                "Tile coarse third-axis bins as 2D slices."
                if tiled_available
                else "Tiled slices require at least three non-singleton dimensions."
            )
        volume_available = supports_volume_view(self.data)
        volume_item = self.view_mode_combo.model().item(3)
        if volume_item is not None:
            volume_item.setEnabled(volume_available)
            volume_item.setToolTip(
                "Render this dataset as a volume or isosurface."
                if volume_available
                else "3D mode requires a gridded dataset with at least three dimensions."
            )
        if (
            (not waterfall_available and self.view_mode_combo.currentIndex() == 1)
            or (not tiled_available and self.view_mode_combo.currentIndex() == 2)
            or (not volume_available and self.view_mode_combo.currentIndex() == 3)
        ):
            self.view_mode_combo.setCurrentIndex(0)

    def _set_view_mode(self, index: int) -> None:
        index = int(index)
        if index in {0, 1, 2}:
            previous = self._active_plot_view_mode
            if index != previous:
                if previous == 1:
                    self._waterfall_marker_face_color = self.marker_face_color
                else:
                    self._slice_marker_face_color = self.marker_face_color
                self.marker_face_color = (
                    self._waterfall_marker_face_color
                    if index == 1
                    else self._slice_marker_face_color
                )
                self._active_plot_view_mode = index
                self._set_combo_silent(
                    self.marker_face_color_combo,
                    (
                        "outline"
                        if self.marker_face_color == "outline"
                        else _option_name(_COLOR_OPTIONS, self.marker_face_color)
                    ),
                )
            self._sync_tile_controls()
            if index == 2 or previous == 2:
                self._rebuild_hidden_axis_controls()
            self.content_stack.setCurrentIndex(0)
            self.update_plot(preserve_view=False)
            return
        from .qt_volume_viewer import QtVolumeViewerPanel, supports_volume_view

        if not supports_volume_view(self.data):
            self.view_mode_combo.setCurrentIndex(0)
            return
        if self.volume_panel is None:
            supported = [
                (dataset, name, original_index)
                for original_index, (dataset, name) in enumerate(zip(self.datasets, self.dataset_names, strict=True))
                if supports_volume_view(dataset)
            ]
            selected = next(
                (index for index, (_dataset, _name, original) in enumerate(supported) if original == self.dataset_index),
                0,
            )
            self.volume_panel = QtVolumeViewerPanel(
                [item[0] for item in supported],
                dataset_names=[item[1] for item in supported],
                selected_index=selected,
            )
            self.content_stack.addWidget(self.volume_panel)
        self.content_stack.setCurrentWidget(self.volume_panel)

    def _close_volume_panel(self) -> None:
        panel = self.volume_panel
        if panel is None:
            return
        self.volume_panel = None
        if self.content_stack is not None:
            self.content_stack.setCurrentIndex(0)
            self.content_stack.removeWidget(panel)
        panel.setParent(None)
        panel.shutdown()
        panel.deleteLater()

    def _rebuild_hidden_axis_controls(self) -> None:
        from PySide6 import QtWidgets

        _clear_layout(self.hidden_layout)
        self.hidden_controls = {}
        if getattr(self.model, "is_point_list", False):
            self.hidden_layout.addWidget(QtWidgets.QLabel("No hidden axes."))
            return
        hidden_dims = [
            dim
            for dim, size in enumerate(self.data.shape)
            if size > 1
            and dim not in (self.model.x_dim, self.model.y_dim)
            and not (self._tiled_mode_active() and dim == self.tile_dim)
        ]
        if not hidden_dims:
            self.hidden_layout.addWidget(QtWidgets.QLabel("No hidden axes."))
            return
        for dim in hidden_dims:
            axis = self.data.axes[dim]
            centers = axis.centers
            low_value = float(np.nanmin(centers))
            high_value = float(np.nanmax(centers))
            current = self.model.selections.get(dim, (float(centers[centers.size // 2]),) * 2)

            group = QtWidgets.QGroupBox(axis.name)
            grid = QtWidgets.QGridLayout(group)
            grid.setContentsMargins(8, 4, 8, 4)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(3)
            value_spin = _make_float_spinbox(low_value, high_value)
            width_spin = _make_float_spinbox(0.0, max(high_value - low_value, 0.0))
            low_spin = _make_float_spinbox(low_value, high_value)
            high_spin = _make_float_spinbox(low_value, high_value)
            range_slider = _IntegratedAxisSlider(centers.size)
            integrate_check = QtWidgets.QCheckBox("Integrate range")
            value_spin.setToolTip(f"Center value selected along hidden axis {axis.name}.")
            width_spin.setToolTip(f"Width integrated along hidden axis {axis.name}.")
            low_spin.setToolTip(f"Lower bound of the integrated range along hidden axis {axis.name}.")
            high_spin.setToolTip(f"Upper bound of the integrated range along hidden axis {axis.name}.")
            range_slider.setToolTip(f"Drag to choose the selected or integrated range along hidden axis {axis.name}.")
            integrate_check.setToolTip(
                "Integrate over the selected range on this hidden axis instead of taking one slice position."
            )
            integrate_check.setChecked(self.model.integrate_checks.get(dim, self.model.integrate))
            midpoint = 0.5 * (float(current[0]) + float(current[1]))
            value_index = _nearest_index(centers, midpoint)
            low_index = _nearest_index(centers, float(min(current)))
            high_index = _nearest_index(centers, float(max(current)))
            if low_index == high_index and centers.size > 1:
                high_index = min(low_index + 1, centers.size - 1)
                if high_index == low_index:
                    low_index = max(low_index - 1, 0)
            controls = _HiddenAxisControls(
                integrate_check,
                value_spin,
                width_spin,
                low_spin,
                high_spin,
                range_slider,
                centers,
                True,
            )
            self.hidden_controls[dim] = controls
            self._set_hidden_controls_indices(dim, value_index, low_index, high_index)
            controls.syncing = False
            value_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "value_spin")
            )
            width_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "width_spin")
            )
            low_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "low_spin")
            )
            high_spin.valueChanged.connect(
                lambda _value, axis_dim=dim: self._sync_hidden_axis(axis_dim, "high_spin")
            )
            range_slider.changed.connect(
                lambda source, axis_dim=dim: self._sync_hidden_axis(axis_dim, source)
            )
            integrate_check.toggled.connect(
                lambda _checked, axis_dim=dim: self._sync_hidden_axis(axis_dim, "integrate")
            )
            grid.addWidget(QtWidgets.QLabel("Value"), 0, 0)
            grid.addWidget(value_spin, 0, 1)
            grid.addWidget(QtWidgets.QLabel("Width"), 0, 2)
            grid.addWidget(width_spin, 0, 3)
            grid.addWidget(QtWidgets.QLabel("Range low"), 1, 0)
            grid.addWidget(low_spin, 1, 1)
            grid.addWidget(QtWidgets.QLabel("Range high"), 1, 2)
            grid.addWidget(high_spin, 1, 3)
            grid.addWidget(range_slider, 2, 0, 1, 4)
            grid.addWidget(integrate_check, 3, 0, 1, 4)
            grid.setColumnMinimumWidth(1, 82)
            grid.setColumnMinimumWidth(3, 82)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 0)
            grid.setColumnStretch(3, 0)
            self.hidden_layout.addWidget(group)

    def _sync_hidden_axis(self, dim: int, source: str) -> None:
        controls = self.hidden_controls[dim]
        if controls.syncing:
            return
        centers = controls.centers
        value_index = controls.slider.value_index
        low_index = controls.slider.low_index
        high_index = controls.slider.high_index
        if source == "value_spin":
            new_value_index = _nearest_index(centers, float(controls.value.value()))
            value_index, low_index, high_index = self._shift_hidden_range(
                centers, new_value_index, low_index, high_index
            )
        elif source == "value":
            value_index, low_index, high_index = self._shift_hidden_range(
                centers, value_index, low_index, high_index
            )
        elif source == "width_spin":
            self._set_integrate_checked(controls, True)
            low_index, high_index = self._range_from_width(
                centers, value_index, float(controls.width.value())
            )
        elif source == "low_spin":
            self._set_integrate_checked(controls, True)
            low_index = _nearest_index(centers, float(controls.low.value()))
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "high_spin":
            self._set_integrate_checked(controls, True)
            high_index = _nearest_index(centers, float(controls.high.value()))
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "low":
            self._set_integrate_checked(controls, True)
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "high":
            self._set_integrate_checked(controls, True)
            low_index, high_index = sorted((low_index, high_index))
            value_index = int(round(0.5 * (low_index + high_index)))
        elif source == "integrate":
            pass
        self._set_hidden_controls_indices(dim, value_index, low_index, high_index)
        self.model.integrate_checks[dim] = bool(controls.integrate.isChecked())
        if controls.integrate.isChecked():
            self.model.selections[dim] = (float(controls.low.value()), float(controls.high.value()))
        else:
            value = float(controls.value.value())
            self.model.selections[dim] = (value, value)
        self.update_plot()

    def _set_hidden_controls_indices(self, dim: int, value_index: int, low_index: int, high_index: int) -> None:
        controls = self.hidden_controls[dim]
        centers = controls.centers
        value_index = int(np.clip(value_index, 0, centers.size - 1))
        low_index = int(np.clip(low_index, 0, centers.size - 1))
        high_index = int(np.clip(high_index, 0, centers.size - 1))
        low_index, high_index = sorted((low_index, high_index))
        controls.syncing = True
        try:
            controls.slider.set_indices(value_index, low_index, high_index)
            controls.slider.set_integrate_range(controls.integrate.isChecked())
            controls.value.setValue(float(centers[value_index]))
            controls.low.setValue(float(centers[low_index]))
            controls.high.setValue(float(centers[high_index]))
            controls.width.setValue(float(centers[high_index] - centers[low_index]))
        finally:
            controls.syncing = False

    def _shift_hidden_range(
        self, centers: np.ndarray, value_index: int, low_index: int, high_index: int
    ) -> tuple[int, int, int]:
        width = max(high_index - low_index, 0)
        half_low = width // 2
        half_high = width - half_low
        low_index = value_index - half_low
        high_index = value_index + half_high
        if low_index < 0:
            high_index -= low_index
            low_index = 0
        if high_index >= centers.size:
            low_index -= high_index - (centers.size - 1)
            high_index = centers.size - 1
        low_index = max(low_index, 0)
        return value_index, low_index, high_index

    def _range_from_width(self, centers: np.ndarray, value_index: int, width: float) -> tuple[int, int]:
        if centers.size <= 1:
            return 0, 0
        value = float(centers[value_index])
        low = _nearest_index(centers, value - 0.5 * max(width, 0.0))
        high = _nearest_index(centers, value + 0.5 * max(width, 0.0))
        if low == high and width > 0:
            if high < centers.size - 1:
                high += 1
            elif low > 0:
                low -= 1
        return tuple(sorted((low, high)))

    def _set_integrate_checked(self, controls: _HiddenAxisControls, checked: bool) -> None:
        was_syncing = controls.syncing
        controls.syncing = True
        try:
            controls.integrate.setChecked(checked)
        finally:
            controls.syncing = was_syncing

    def _set_display_dim(self, axis_name: str, dim: int) -> None:
        if self._syncing_axes:
            return
        if getattr(self.model, "is_point_list", False):
            if axis_name != "x" or not (0 <= int(dim) < len(self.model.point_coordinates)):
                return
            self.model.x_key = self.model.point_coordinates[int(dim)]
            self.update_plot(preserve_view=False)
            return
        if not (0 <= int(dim) < len(self._display_axis_dims)):
            return
        dim = self._display_axis_dims[int(dim)]
        if axis_name == "x" and dim == self.model.y_dim:
            self.model.x_dim, self.model.y_dim = self.model.y_dim, self.model.x_dim
            self._sync_axis_combos()
        elif axis_name == "y" and dim == self.model.x_dim:
            self.model.x_dim, self.model.y_dim = self.model.y_dim, self.model.x_dim
            self._sync_axis_combos()
        elif axis_name == "x":
            self.model.x_dim = int(dim)
        else:
            self.model.y_dim = int(dim)
        self._sync_tile_controls()
        self._rebuild_hidden_axis_controls()
        self.update_plot(preserve_view=False)

    def _set_dataset_index(self, index: int) -> None:
        index = int(index)
        if index == self.dataset_index or not (0 <= index < len(self.datasets)):
            return
        self._dataset_states[self.dataset_index] = self._capture_dataset_state()
        state = self._dataset_states[index]
        if state is None:
            state = self._default_dataset_state(index)
            self._dataset_states[index] = state
        self.dataset_index = index
        self._restore_dataset_state(state)

    def _capture_dataset_state(self) -> _DatasetViewState:
        xlim = None
        ylim = None
        if self.ax_image is not None and self._current_slice is not None:
            xlim = tuple(float(value) for value in self.ax_image.get_xlim())
            ylim = tuple(float(value) for value in self.ax_image.get_ylim())
        return _DatasetViewState(
            model=self.model,
            roi_extents=self._roi_extents,
            xlim=xlim,
            ylim=ylim,
            show_box_tool=bool(self.show_box_check.isChecked()) if self.show_box_check is not None else False,
            histogram_axes=bool(self.hist_axes_check.isChecked()) if self.hist_axes_check is not None else False,
            roi_enabled=bool(self.roi_button.isChecked()) if self.roi_button is not None else False,
            xcut_percent=int(self.xcut_percent),
            ycut_percent=int(self.ycut_percent),
            font_size=float(self.font_size),
            axis_linewidth=float(self.axis_linewidth),
            show_binning_title=bool(self.show_binning_title),
            box_tool_has_auto_shown_hist_axes=bool(self._box_tool_has_auto_shown_hist_axes),
            marker=str(self.marker),
            line_style=str(self.line_style),
            marker_size=float(self.marker_size),
            line_plot_width=float(self.line_plot_width),
            marker_edge_width=float(self.marker_edge_width),
            marker_face_color=str(self.marker_face_color),
            line_color=str(self.line_color),
            show_errorbars=bool(self.show_errorbars),
            show_errorbar_caps=bool(self.show_errorbar_caps),
            errorbar_cap_size=float(self.errorbar_cap_size),
            show_fit=bool(self.show_fit),
            unmask_model=bool(self.unmask_model),
            show_residual=bool(self.show_residual),
            fit_line_color=str(self.fit_line_color),
            fit_line_width=float(self.fit_line_width),
            residual_percent=int(self.residual_percent),
            apply_masks=bool(self.model.masked),
            cmap_reversed=bool(self.model.cmap_reversed),
            smoothing_x=float(self.smoothing_x),
            smoothing_y=float(self.smoothing_y),
            tile_dim=self.tile_dim,
            tile_range=self.tile_range,
            tile_step=float(self.tile_step),
            tile_step_auto=bool(self.tile_step_auto),
        )

    def _default_dataset_state(self, index: int) -> _DatasetViewState:
        data = self.datasets[index]
        x_dim, y_dim = self._default_display_dims_for_dataset(data)
        model = MDHistoSliceViewer(
            data,
            x_dim=x_dim,
            y_dim=y_dim,
            channel=self._initial_channel,
            cmap=self._initial_cmap,
            color_scale=self._initial_color_scale,
            auto_limits=self._initial_auto_limits,
            integrate=self._initial_integrate,
            masked=self._initial_masked,
            coverage_threshold=self.coverage_threshold,
        )
        tile_candidates = (
            [
                dim
                for dim, size in enumerate(data.shape)
                if size > 1 and dim not in (model.x_dim, model.y_dim)
            ]
            if isinstance(data, MDHistoData)
            else []
        )
        tile_dim = tile_candidates[0] if tile_candidates else None
        tile_range = (0.0, 0.0)
        tile_step = 1.0
        if tile_dim is not None:
            centers = np.asarray(data.axes[tile_dim].centers, dtype=float)
            tile_range = (float(centers[0]), float(centers[-1]))
            tile_step = default_tiled_slice_step(data, tile_dim, tile_range)
        return _DatasetViewState(
            model=model,
            apply_masks=bool(model.masked),
            show_fit=bool(data.metadata.get("viewer_show_fit", False)),
            tile_dim=tile_dim,
            tile_range=tile_range,
            tile_step=tile_step,
        )

    def _restore_dataset_state(self, state: _DatasetViewState) -> None:
        self._restoring_dataset_state = True
        try:
            self.model = state.model
            self.data = state.model.data
            self._roi_extents = state.roi_extents
            self.xcut_percent = int(state.xcut_percent)
            self.ycut_percent = int(state.ycut_percent)
            self.font_size = float(state.font_size)
            self.axis_linewidth = float(state.axis_linewidth)
            self.show_binning_title = bool(state.show_binning_title)
            self.marker = str(state.marker)
            self.line_style = str(state.line_style)
            self.marker_size = float(state.marker_size)
            self.line_plot_width = float(state.line_plot_width)
            self.marker_edge_width = float(state.marker_edge_width)
            self.marker_face_color = str(state.marker_face_color)
            if self._active_plot_view_mode == 1:
                self._waterfall_marker_face_color = self.marker_face_color
            else:
                self._slice_marker_face_color = self.marker_face_color
            self.line_color = str(state.line_color)
            self.show_errorbars = bool(state.show_errorbars)
            self.show_errorbar_caps = bool(state.show_errorbar_caps)
            self.errorbar_cap_size = float(state.errorbar_cap_size)
            self.show_fit = bool(state.show_fit)
            self.unmask_model = bool(state.unmask_model)
            self.show_residual = bool(state.show_residual)
            self.fit_line_color = str(state.fit_line_color)
            self.fit_line_width = float(state.fit_line_width)
            self.residual_percent = int(state.residual_percent)
            self.smoothing_x = float(state.smoothing_x)
            self.smoothing_y = float(state.smoothing_y)
            self.tile_dim = state.tile_dim
            self.tile_range = tuple(state.tile_range)
            self.tile_step = float(state.tile_step)
            self.tile_step_auto = bool(state.tile_step_auto)
            self.model.masked = bool(state.apply_masks)
            self.model.coverage_threshold = self.coverage_threshold
            self.model.cmap_reversed = bool(state.cmap_reversed)
            self._box_tool_has_auto_shown_hist_axes = bool(state.box_tool_has_auto_shown_hist_axes)
            self._current_slice = None
            self._last_plot_dims = None
            self._sync_axis_combos(rebuild=True)
            self._sync_tile_controls()
            self._sync_view_mode_availability()
            self._sync_channel_combo()
            self._set_combo_silent(self.cmap_combo, self.model.cmap)
            self._set_combo_silent(self.scale_combo, self.model.color_scale)
            self._set_combo_silent(self.limits_combo, self.model.auto_limits)
            self._set_checkbox_silent(self.autoscale_check, self.model.autoscale)
            self._set_checkbox_silent(self.apply_masks_check, self.model.masked)
            self._set_spin_silent(
                self.coverage_threshold_spin,
                self.coverage_threshold,
            )
            self._set_spin_silent(self.gamma_spin, self.model.power_gamma)
            self._set_spin_silent(self.limit_n_spin, self._current_limit_n())
            self._set_spin_silent(self.font_size_spin, self.font_size)
            self._set_spin_silent(self.line_width_spin, self.axis_linewidth)
            self._set_checkbox_silent(
                self.show_binning_title_check, self.show_binning_title
            )
            self._set_spin_silent(self.smoothing_x_spin, self.smoothing_x)
            self._set_spin_silent(self.smoothing_y_spin, self.smoothing_y)
            self._set_combo_silent(self.marker_combo, _option_name(_MARKER_OPTIONS, self.marker))
            self._set_combo_silent(self.line_style_combo, _option_name(_LINE_STYLE_OPTIONS, self.line_style))
            self._set_spin_silent(self.marker_size_spin, self.marker_size)
            self._set_spin_silent(self.line_plot_width_spin, self.line_plot_width)
            self._set_spin_silent(self.marker_edge_width_spin, self.marker_edge_width)
            self._set_combo_silent(
                self.marker_face_color_combo,
                (
                    "outline"
                    if self.marker_face_color == "outline"
                    else _option_name(_COLOR_OPTIONS, self.marker_face_color)
                ),
            )
            self._set_combo_silent(self.line_color_combo, _option_name(_COLOR_OPTIONS, self.line_color))
            self._set_checkbox_silent(self.show_errorbars_check, self.show_errorbars)
            self._set_checkbox_silent(self.show_errorbar_caps_check, self.show_errorbar_caps)
            self._set_spin_silent(self.errorbar_cap_size_spin, self.errorbar_cap_size)
            self._set_combo_silent(self.fit_line_color_combo, _option_name(_COLOR_OPTIONS, self.fit_line_color))
            self._set_spin_silent(self.fit_line_width_spin, self.fit_line_width)
            self._set_slider_silent(self.residual_split_slider, self.residual_percent)
            self._sync_fit_channel_controls()
            self._set_slider_silent(self.xcut_percent_slider, self.xcut_percent)
            self._set_slider_silent(self.ycut_percent_slider, self.ycut_percent)
            self._set_checkbox_silent(self.show_box_check, state.show_box_tool)
            self._set_checkbox_silent(self.hist_axes_check, state.histogram_axes)
            self._set_checkbox_silent(self.roi_button, state.roi_enabled)
            self.gamma_label.setVisible(self.model.color_scale == "power")
            self.gamma_spin.setVisible(self.model.color_scale == "power")
            self._sync_limit_n_visibility()
            self._rebuild_hidden_axis_controls()
            self._sync_control_visibility()
        finally:
            self._restoring_dataset_state = False
        self.update_plot(preserve_view=False)
        if state.xlim is not None:
            self.ax_image.set_xlim(*state.xlim)
        if state.ylim is not None:
            self.ax_image.set_ylim(*state.ylim)
        self._sync_view_limit_controls()
        self._set_rectangle_selector_from_controls()
        self._sync_histogram_panel_controls()
        self._apply_histogram_axes_layout(draw=False)
        self.canvas.draw_idle()

    def _set_combo_silent(self, combo, value: str) -> None:
        previous = combo.blockSignals(True)
        try:
            combo.setCurrentText(str(value))
        finally:
            combo.blockSignals(previous)

    def _set_combo_items_silent(self, combo, items: Sequence[str], current: str | None) -> None:
        previous = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems([str(item) for item in items])
            if current is not None:
                combo.setCurrentText(str(current))
        finally:
            combo.blockSignals(previous)

    def _set_checkbox_silent(self, checkbox, checked: bool) -> None:
        previous = checkbox.blockSignals(True)
        try:
            checkbox.setChecked(bool(checked))
        finally:
            checkbox.blockSignals(previous)

    def _set_spin_silent(self, spinbox, value: float) -> None:
        previous = spinbox.blockSignals(True)
        try:
            spinbox.setValue(float(value))
        finally:
            spinbox.blockSignals(previous)

    def _set_slider_silent(self, slider, value: int) -> None:
        previous = slider.blockSignals(True)
        try:
            slider.setValue(int(value))
        finally:
            slider.blockSignals(previous)

    def _sync_waterfall_controls(self) -> None:
        if self.waterfall_step_spin is None:
            return
        self._set_spin_silent(self.waterfall_step_spin, self.waterfall_step)
        self._set_spin_silent(
            self.waterfall_coverage_threshold_spin,
            self.waterfall_coverage_threshold,
        )
        self._set_checkbox_silent(
            self.waterfall_step_auto_check,
            self.waterfall_step_auto,
        )
        self._set_spin_silent(self.waterfall_offset_spin, self.waterfall_offset)
        self._set_checkbox_silent(
            self.waterfall_offset_auto_check,
            self.waterfall_offset_auto,
        )
        self._set_combo_silent(self.waterfall_cmap_combo, self.waterfall_cmap)
        self.waterfall_color_range_slider.set_values(
            int(round(1000.0 * self.waterfall_color_min)),
            int(round(1000.0 * self.waterfall_color_max)),
        )
        self._set_checkbox_silent(
            self.waterfall_reverse_check,
            self.waterfall_reverse_colors,
        )
        self._set_checkbox_silent(
            self.waterfall_zero_check,
            self.waterfall_show_zero_lines,
        )
        self._set_combo_silent(
            self.waterfall_zero_color_combo,
            _option_name(_COLOR_OPTIONS, self.waterfall_zero_color),
        )
        self._set_combo_silent(
            self.waterfall_zero_style_combo,
            _option_name(_LINE_STYLE_OPTIONS, self.waterfall_zero_style),
        )
        self._set_spin_silent(
            self.waterfall_zero_width_spin,
            self.waterfall_zero_width,
        )
        self._set_combo_silent(
            self.waterfall_model_color_combo,
            (
                "match traces"
                if self.waterfall_model_color is None
                else _option_name(_COLOR_OPTIONS, self.waterfall_model_color)
            ),
        )
        self._set_checkbox_silent(
            self.waterfall_trace_labels_check,
            self.waterfall_show_trace_labels,
        )
        previous = self.waterfall_trace_label_suffix_edit.blockSignals(True)
        try:
            self.waterfall_trace_label_suffix_edit.setText(
                self.waterfall_trace_label_suffix
            )
        finally:
            self.waterfall_trace_label_suffix_edit.blockSignals(previous)
        self._set_spin_silent(
            self.waterfall_trace_label_font_size_spin,
            self.waterfall_trace_label_font_size,
        )
        self._set_combo_silent(
            self.waterfall_trace_label_color_combo,
            (
                "match traces"
                if self.waterfall_trace_label_color is None
                else _option_name(
                    _COLOR_OPTIONS,
                    self.waterfall_trace_label_color,
                )
            ),
        )

    def _tile_axis_candidates(self) -> list[int]:
        if getattr(self.model, "is_point_list", False):
            return []
        return [
            dim
            for dim, size in enumerate(self.data.shape)
            if size > 1 and dim not in (self.model.x_dim, self.model.y_dim)
        ]

    def _sync_tile_controls(self, *, reset_range: bool = False) -> None:
        if self.tile_dim_combo is None:
            return
        candidates = self._tile_axis_candidates()
        if self.tile_dim not in candidates:
            self.tile_dim = candidates[0] if candidates else None
            reset_range = True
        previous = self.tile_dim_combo.blockSignals(True)
        try:
            self.tile_dim_combo.clear()
            for dim in candidates:
                self.tile_dim_combo.addItem(self.data.axes[dim].name, dim)
            if self.tile_dim is not None:
                index = self.tile_dim_combo.findData(self.tile_dim)
                self.tile_dim_combo.setCurrentIndex(max(index, 0))
        finally:
            self.tile_dim_combo.blockSignals(previous)
        enabled = self.tile_dim is not None
        for widget in (
            self.tile_dim_combo,
            self.tile_range_low_spin,
            self.tile_range_high_spin,
            self.tile_step_spin,
            self.tile_step_slider,
            self.tile_step_auto_check,
        ):
            if widget is not None:
                widget.setEnabled(enabled)
        if not enabled:
            return
        centers = np.asarray(self.data.axes[self.tile_dim].centers, dtype=float)
        minimum = float(np.min(centers))
        maximum = float(np.max(centers))
        if reset_range:
            self.tile_range = (minimum, maximum)
        else:
            low, high = sorted(map(float, self.tile_range))
            self.tile_range = (
                float(np.clip(low, minimum, maximum)),
                float(np.clip(high, minimum, maximum)),
            )
        for spin in (self.tile_range_low_spin, self.tile_range_high_spin):
            previous = spin.blockSignals(True)
            try:
                spin.setRange(minimum, maximum)
                spin.setSingleStep(max((maximum - minimum) / 100.0, 1.0e-9))
            finally:
                spin.blockSignals(previous)
        self._set_spin_silent(self.tile_range_low_spin, self.tile_range[0])
        self._set_spin_silent(self.tile_range_high_spin, self.tile_range[1])
        if self.tile_step_auto:
            self.tile_step = default_tiled_slice_step(
                self.data,
                self.tile_dim,
                self.tile_range,
            )
        self._set_checkbox_silent(self.tile_step_auto_check, self.tile_step_auto)
        self._set_spin_silent(self.tile_step_spin, self.tile_step)
        self._sync_tile_step_slider()

    def _tile_step_limits(self) -> tuple[float, float]:
        if self.tile_dim is None:
            return 1.0, 1.0
        minimum, _full_span = waterfall_step_bounds(self.data, self.tile_dim)
        centers = np.asarray(self.data.axes[self.tile_dim].centers, dtype=float)
        edges = np.asarray(self.data.axes[self.tile_dim].values, dtype=float)
        low, high = sorted(map(float, self.tile_range))
        selected = np.flatnonzero((centers >= low) & (centers <= high))
        if selected.size == 0:
            return minimum, minimum
        span = float(edges[int(selected[-1]) + 1] - edges[int(selected[0])])
        return min(minimum, span), max(minimum, span)

    def _sync_tile_step_slider(self) -> None:
        if self.tile_step_spin is None:
            return
        low, high = self._tile_step_limits()
        self.tile_step = float(np.clip(self.tile_step, low, high))
        previous = self.tile_step_spin.blockSignals(True)
        try:
            self.tile_step_spin.setRange(low, high)
            self.tile_step_spin.setValue(self.tile_step)
        finally:
            self.tile_step_spin.blockSignals(previous)
        fraction = 0.0 if high <= low else (self.tile_step - low) / (high - low)
        self._set_slider_silent(
            self.tile_step_slider,
            int(round(1000.0 * np.clip(fraction, 0.0, 1.0))),
        )

    def _set_tile_dimension(self, index: int) -> None:
        dim = self.tile_dim_combo.itemData(int(index))
        if dim is None or int(dim) == self.tile_dim:
            return
        self.tile_dim = int(dim)
        self._sync_tile_controls(reset_range=True)
        self._rebuild_hidden_axis_controls()
        if self._tiled_mode_active():
            self.update_plot(preserve_view=False)

    def _set_tile_range(self, _value: float) -> None:
        if self._restoring_dataset_state or self.tile_dim is None:
            return
        low, high = sorted(
            (
                float(self.tile_range_low_spin.value()),
                float(self.tile_range_high_spin.value()),
            )
        )
        self.tile_range = (low, high)
        if self.tile_step_auto:
            self.tile_step = default_tiled_slice_step(
                self.data,
                self.tile_dim,
                self.tile_range,
            )
        self._sync_tile_controls()
        if self._tiled_mode_active():
            self.update_plot(preserve_view=False)

    def _set_tile_step(self, value: float) -> None:
        if self._restoring_dataset_state:
            return
        low, high = self._tile_step_limits()
        self.tile_step = float(np.clip(value, low, high))
        if self.tile_step_auto_check.isChecked():
            self.tile_step_auto = False
            self._set_checkbox_silent(self.tile_step_auto_check, False)
        self._sync_tile_step_slider()
        if self._tiled_mode_active():
            self.update_plot(preserve_view=False)

    def _set_tile_step_from_slider(self, position: int) -> None:
        if self._restoring_dataset_state:
            return
        low, high = self._tile_step_limits()
        fraction = float(np.clip(position, 0, 1000)) / 1000.0
        value = low + fraction * (high - low)
        self._set_spin_silent(self.tile_step_spin, value)
        self._set_tile_step(value)

    def _set_tile_step_auto(self, enabled: bool) -> None:
        self.tile_step_auto = bool(enabled)
        if self.tile_step_auto and self.tile_dim is not None:
            self.tile_step = default_tiled_slice_step(
                self.data,
                self.tile_dim,
                self.tile_range,
            )
        self._sync_tile_step_slider()
        if not self._restoring_dataset_state and self._tiled_mode_active():
            self.update_plot(preserve_view=False)

    def _set_rectangle_selector_from_controls(self) -> None:
        if self.rectangle_selector is None:
            return
        show_box = bool(self.show_box_check.isChecked())
        active = show_box and bool(self.roi_button.isChecked())
        self.rectangle_selector.set_visible(show_box)
        self.rectangle_selector.set_active(active)
        self._set_rectangle_selector_style(active=active)

    def _default_display_dims_for_dataset(self, data: MDHistoData) -> tuple[int, int]:
        if isinstance(data, PointListData):
            return 0, 0
        if self._initial_x_dim == -1 and self._initial_y_dim == 0:
            return self._fallback_display_dims(data)
        try:
            model = MDHistoSliceViewer(
                data,
                x_dim=self._initial_x_dim,
                y_dim=self._initial_y_dim,
                channel=self._initial_channel,
                cmap=self._initial_cmap,
                color_scale=self._initial_color_scale,
                auto_limits=self._initial_auto_limits,
                integrate=self._initial_integrate,
                masked=self._initial_masked,
            )
            return model.x_dim, model.y_dim
        except Exception:
            return self._fallback_display_dims(data)

    def _fallback_display_dims(self, data: MDHistoData) -> tuple[int, int]:
        varying = [dim for dim, size in enumerate(data.shape) if size > 1]
        if len(varying) == 1:
            x_dim = varying[0]
            return x_dim, next(
                (dim for dim in range(data.signal.ndim) if dim != x_dim),
                x_dim,
            )
        if len(varying) >= 2:
            return varying[0], varying[1]
        if data.signal.ndim >= 2:
            return 0, 1
        return 0, 0

    def _sync_axis_combos(self, *, rebuild: bool = False) -> None:
        self._syncing_axes = True
        try:
            if getattr(self.model, "is_point_list", False):
                if rebuild:
                    self._display_axis_dims = list(range(len(self.model.point_coordinates)))
                    self.x_combo.clear()
                    self.y_combo.clear()
                    self.x_combo.addItems(self.model.point_coordinates)
                if self.model.x_key in self.model.point_coordinates:
                    self.x_combo.setCurrentIndex(self.model.point_coordinates.index(self.model.x_key))
                return
            if rebuild:
                self._display_axis_dims = self._non_singleton_dims()
                labels = [self.data.axes[dim].name for dim in self._display_axis_dims]
                self.x_combo.clear()
                self.y_combo.clear()
                self.x_combo.addItems(labels)
                self.y_combo.addItems(labels)
            if self.model.x_dim in self._display_axis_dims:
                self.x_combo.setCurrentIndex(self._display_axis_dims.index(self.model.x_dim))
            if self.model.y_dim in self._display_axis_dims:
                self.y_combo.setCurrentIndex(self._display_axis_dims.index(self.model.y_dim))
        finally:
            self._syncing_axes = False

    def _sync_channel_combo(self) -> None:
        if self.channel_combo is None:
            return
        if getattr(self.model, "is_point_list", False):
            items = list(self.model.point_channels)
        else:
            items = list(self.model.CHANNELS)
        self._set_combo_items_silent(self.channel_combo, items, self.model.channel)

    def _non_singleton_dims(self) -> list[int]:
        if getattr(self.model, "is_point_list", False):
            return []
        return [dim for dim, size in enumerate(self.data.shape) if size > 1]

    def _set_cmap(self, cmap: str) -> None:
        self.model.cmap = str(cmap)
        self.update_plot()

    def _set_plot_smoothing(self, axis_name: str, value: float) -> None:
        if self._restoring_dataset_state:
            return
        if axis_name == "x":
            self.smoothing_x = max(float(value), 0.0)
        else:
            self.smoothing_y = max(float(value), 0.0)
        self.update_plot()

    def _smoothed_slice_view(self, view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if getattr(self.model, "is_point_list", False):
            return view
        return smooth_mdhisto_view(
            view,
            sigma_x=self.smoothing_x,
            sigma_y=self.smoothing_y,
        )

    def _toggle_cmap_reverse(self) -> None:
        self.model.cmap_reversed = not bool(self.model.cmap_reversed)
        self.update_plot()

    def _set_channel(self, channel: str) -> None:
        if getattr(self.model, "is_point_list", False):
            if channel in self.model.point_channels:
                self.model.channel = channel
                self.update_plot(preserve_view=False)
            return
        self.model.channel = self.model._resolve_channel(channel)
        self.update_plot(preserve_view=False)

    def _set_apply_masks(self, checked: bool) -> None:
        self.model.masked = bool(checked)
        self.update_plot()

    def _set_coverage_threshold(
        self,
        value: float,
        *,
        redraw: bool = True,
    ) -> None:
        self.coverage_threshold = float(np.clip(value, 0.0, 1.0))
        self.model.coverage_threshold = self.coverage_threshold
        if self.coverage_threshold_spin is not None:
            self._set_spin_silent(
                self.coverage_threshold_spin,
                self.coverage_threshold,
            )
        if redraw:
            self.update_plot()

    def _has_fit_channel(self) -> bool:
        return self._channel_available("fit")

    def _has_residual_channel(self) -> bool:
        return self._channel_available("residual")

    def _channel_available(self, name: str) -> bool:
        if getattr(self.model, "is_point_list", False):
            return self.model.point_overlay_channel(name) is not None
        return name in self.model.CHANNELS

    def _fit_panels_active(self) -> bool:
        """Side-by-side pcolor panels are used for 2D data with a fit shown."""

        return bool(
            not self._tiled_mode_active()
            and self.show_fit
            and self._has_fit_channel()
            and not self._is_effective_1d()
        )

    def _residual_axes_active(self) -> bool:
        """A residual axes below the data is used for 1D data."""

        return bool(
            self.show_fit
            and self.show_residual
            and self._has_fit_channel()
            and self._has_residual_channel()
            and self._is_effective_1d()
        )

    def _fit_cuts_active(self) -> bool:
        """Integrated data+fit cut axes are shown when the box tool is on in 2D fit compare."""

        return bool(
            self._fit_panels_active()
            and self.hist_axes_check is not None
            and self.hist_axes_check.isChecked()
        )

    def _fit_residual_cut_active(self) -> bool:
        """A separate residual cut axes accompanies the data+fit cut when residuals are shown."""

        return bool(self._fit_cuts_active() and self.show_residual and self._has_residual_channel())

    def _sync_fit_channel_controls(self) -> None:
        if self.show_fit_check is None:
            return
        has_fit = self._has_fit_channel()
        has_residual = self._has_residual_channel()
        if not has_fit:
            self.show_fit = False
        if not (self.show_fit and has_residual):
            self.show_residual = False
        self._set_checkbox_silent(self.show_fit_check, self.show_fit)
        self._set_checkbox_silent(self.unmask_model_check, self.unmask_model)
        self._set_checkbox_silent(self.show_residual_check, self.show_residual)
        self.show_fit_check.setEnabled(has_fit)
        self.unmask_model_check.setEnabled(bool(self.show_fit and has_fit))
        self.show_residual_check.setEnabled(bool(self.show_fit and has_residual))
        residual_split = self._residual_axes_active()
        self.residual_split_label.setVisible(residual_split)
        self.residual_split_slider.setVisible(residual_split)
        self.residual_split_label.setText(f"Residual height: {self.residual_percent}%")
        fit_line = bool(
            has_fit
            and (
                self._is_effective_1d()
                or self._fit_panels_active()
                or self._waterfall_mode_active()
            )
        )
        for widget in (
            self.fit_line_color_label,
            self.fit_line_color_combo,
            self.fit_line_width_label,
            self.fit_line_width_spin,
        ):
            if widget is not None:
                widget.setEnabled(fit_line and self.show_fit)

    def _set_show_fit(self, enabled: bool) -> None:
        self.show_fit = bool(enabled)
        self._sync_fit_channel_controls()
        self.update_plot(preserve_view=False)

    def _set_unmask_model(self, enabled: bool) -> None:
        changed = self.unmask_model != bool(enabled)
        self.unmask_model = bool(enabled)
        self._sync_fit_channel_controls()
        if (
            changed
            and not self._restoring_dataset_state
            and self._unmask_model_callback is not None
        ):
            self._unmask_model_callback(self.unmask_model)
        self.update_plot(preserve_view=False)

    def _set_show_residual(self, enabled: bool) -> None:
        self.show_residual = bool(enabled)
        self._sync_fit_channel_controls()
        self.update_plot(preserve_view=False)

    def _set_fit_line_color(self, color_name: str) -> None:
        self.fit_line_color = _COLOR_OPTIONS.get(str(color_name), "#d62728")
        if self.show_fit and (
            self._is_effective_1d()
            or self._fit_cuts_active()
            or self._waterfall_mode_active()
        ):
            self.update_plot()

    def _set_fit_line_width(self, value: float) -> None:
        self.fit_line_width = float(value)
        if self.show_fit and (
            self._is_effective_1d()
            or self._fit_cuts_active()
            or self._waterfall_mode_active()
        ):
            self.update_plot()

    def _set_residual_percent(self, value: int) -> None:
        self.residual_percent = int(value)
        self.residual_split_label.setText(f"Residual height: {self.residual_percent}%")
        if self._plot_layout_mode == ("residual_1d", 2) and self.grid is not None:
            self.grid.set_height_ratios([1.0, self._panel_ratio(self.residual_percent)])
            self.canvas.draw_idle()

    def _set_color_scale(self, color_scale: str) -> None:
        self.model.color_scale = str(color_scale)
        self.gamma_label.setVisible(self.model.color_scale == "power")
        self.gamma_spin.setVisible(self.model.color_scale == "power")
        self.update_plot()

    def _set_auto_limits(self, auto_limits: str) -> None:
        self.model.auto_limits = str(auto_limits)
        self.limit_n_spin.setValue(self._current_limit_n())
        self._sync_limit_n_visibility()
        self._set_autoscale_checkbox(True)
        self.update_plot()

    def _set_autoscale(self, autoscale: bool) -> None:
        if self._syncing_limits:
            return
        self.model.autoscale = bool(autoscale)
        self.update_plot()

    def _set_manual_limit(self, which: str, value: float) -> None:
        if self._syncing_limits:
            return
        if which == "vmin":
            self.model.manual_vmin = float(value)
        else:
            self.model.manual_vmax = float(value)
        self._set_autoscale_checkbox(False)
        self.update_plot()

    def _set_power_gamma(self, gamma: float) -> None:
        self.model.power_gamma = max(float(gamma), 1.0e-12)
        if self.model.color_scale == "power":
            self.update_plot()

    def _set_limit_n(self, value: float) -> None:
        if self.model.auto_limits == "N-sigma":
            self.model.sigma_n = max(float(value), 0.0)
        elif self.model.auto_limits == "N IQR":
            self.model.iqr_n = max(float(value), 0.0)
        elif self.model.auto_limits == "Nth percentile":
            self.model.percentile_n = float(np.clip(value, 0.0, 50.0))
        if self.model.auto_limits != "min/max":
            self._set_autoscale_checkbox(True)
            self.update_plot()

    def _set_marker(self, marker_name: str) -> None:
        self.marker = _MARKER_OPTIONS.get(str(marker_name), "o")
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_line_style(self, line_style_name: str) -> None:
        self.line_style = _LINE_STYLE_OPTIONS.get(str(line_style_name), "none")
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_marker_size(self, value: float) -> None:
        self.marker_size = float(value)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_line_plot_width(self, value: float) -> None:
        self.line_plot_width = float(value)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_marker_edge_width(self, value: float) -> None:
        self.marker_edge_width = float(value)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_marker_face_color(self, color_name: str) -> None:
        self.marker_face_color = (
            "outline"
            if color_name == "outline"
            else _COLOR_OPTIONS.get(str(color_name), "#1f77b4")
        )
        if self._waterfall_mode_active():
            self._waterfall_marker_face_color = self.marker_face_color
        else:
            self._slice_marker_face_color = self.marker_face_color
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_line_color(self, color_name: str) -> None:
        self.line_color = _COLOR_OPTIONS.get(str(color_name), "#1f77b4")
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_show_errorbars(self, show_errorbars: bool) -> None:
        self.show_errorbars = bool(show_errorbars)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_show_errorbar_caps(self, show_caps: bool) -> None:
        self.show_errorbar_caps = bool(show_caps)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_errorbar_cap_size(self, value: float) -> None:
        self.errorbar_cap_size = float(value)
        if self._is_effective_1d() or self._waterfall_mode_active():
            self.update_plot()

    def _set_waterfall_step(self, value: float) -> None:
        if self._restoring_dataset_state:
            return
        low, high = self._waterfall_step_limits()
        self.waterfall_step = float(np.clip(value, low, high))
        self._sync_waterfall_step_slider()
        if self.waterfall_step_auto_check is not None and self.waterfall_step_auto_check.isChecked():
            self._set_checkbox_silent(self.waterfall_step_auto_check, False)
            self.waterfall_step_auto = False
        self.update_plot(preserve_view=False)

    def _set_waterfall_coverage_threshold(self, value: float) -> None:
        self.waterfall_coverage_threshold = float(np.clip(value, 0.0, 1.0))
        if self.waterfall_coverage_threshold_spin is not None:
            self._set_spin_silent(
                self.waterfall_coverage_threshold_spin,
                self.waterfall_coverage_threshold,
            )
        self.update_plot(preserve_view=False)

    def _set_waterfall_step_from_slider(self, position: int) -> None:
        if self._restoring_dataset_state:
            return
        low, high = self._waterfall_step_limits()
        fraction = float(np.clip(position, 0, 1000)) / 1000.0
        self._set_spin_silent(
            self.waterfall_step_spin,
            low + fraction * (high - low),
        )
        self._set_waterfall_step(self.waterfall_step_spin.value())

    def _set_waterfall_step_auto(self, enabled: bool) -> None:
        self.waterfall_step_auto = bool(enabled)
        if not self._restoring_dataset_state:
            self.update_plot(preserve_view=False)

    def _set_waterfall_offset(self, value: float) -> None:
        if self._restoring_dataset_state:
            return
        maximum = self._waterfall_offset_maximum()
        self.waterfall_offset = float(np.clip(value, 0.0, maximum))
        self._sync_waterfall_offset_slider()
        if self.waterfall_offset_auto_check is not None and self.waterfall_offset_auto_check.isChecked():
            self._set_checkbox_silent(self.waterfall_offset_auto_check, False)
            self.waterfall_offset_auto = False
        self.update_plot(preserve_view=False)

    def _set_waterfall_offset_from_slider(self, position: int) -> None:
        if self._restoring_dataset_state:
            return
        maximum = self._waterfall_offset_maximum()
        value = maximum * float(np.clip(position, 0, 1000)) / 1000.0
        self._set_spin_silent(self.waterfall_offset_spin, value)
        self._set_waterfall_offset(value)

    def _set_waterfall_offset_auto(self, enabled: bool) -> None:
        self.waterfall_offset_auto = bool(enabled)
        if not self._restoring_dataset_state:
            self.update_plot(preserve_view=False)

    def _set_waterfall_cmap(self, cmap: str) -> None:
        self.waterfall_cmap = str(cmap)
        self._sync_control_visibility()
        self.update_plot()

    def _set_waterfall_color_range(self, low: int, high: int) -> None:
        self.waterfall_color_min = float(np.clip(low, 0, 1000)) / 1000.0
        self.waterfall_color_max = float(np.clip(high, 0, 1000)) / 1000.0
        self.update_plot()

    def _set_waterfall_reverse_colors(self, enabled: bool) -> None:
        self.waterfall_reverse_colors = bool(enabled)
        self.update_plot()

    def _set_waterfall_zero_lines(self, enabled: bool) -> None:
        self.waterfall_show_zero_lines = bool(enabled)
        self.update_plot()

    def _set_waterfall_zero_color(self, color_name: str) -> None:
        self.waterfall_zero_color = _COLOR_OPTIONS.get(color_name, "#7f7f7f")
        self.update_plot()

    def _set_waterfall_zero_style(self, style_name: str) -> None:
        self.waterfall_zero_style = _LINE_STYLE_OPTIONS.get(style_name, "--")
        self.update_plot()

    def _set_waterfall_zero_width(self, value: float) -> None:
        self.waterfall_zero_width = max(float(value), 0.1)
        self.update_plot()

    def _set_waterfall_model_color(self, color_name: str) -> None:
        self.waterfall_model_color = (
            None if color_name == "match traces" else _COLOR_OPTIONS.get(color_name)
        )
        self.update_plot()

    def _set_waterfall_trace_labels(self, enabled: bool) -> None:
        self.waterfall_show_trace_labels = bool(enabled)
        self.update_plot()

    def _set_waterfall_trace_label_suffix(self, suffix: str) -> None:
        self.waterfall_trace_label_suffix = str(suffix)
        self.update_plot()

    def _set_waterfall_trace_label_font_size(self, value: float) -> None:
        self.waterfall_trace_label_font_size = max(float(value), 1.0)
        self.update_plot()

    def _set_waterfall_trace_label_color(self, color_name: str) -> None:
        self.waterfall_trace_label_color = (
            None
            if color_name == "match traces"
            else _COLOR_OPTIONS.get(color_name)
        )
        self.update_plot()

    def _waterfall_step_limits(self) -> tuple[float, float]:
        if self._waterfall_uses_1d_group():
            return 1.0, 1.0
        return waterfall_step_bounds(self.data, self.model.y_dim)

    def _waterfall_offset_maximum(self) -> float:
        maximum = waterfall_absolute_max(self._current_waterfall_traces)
        return maximum if maximum > 0.0 else 1.0

    def _sync_waterfall_step_slider(self) -> None:
        low, high = self._waterfall_step_limits()
        self.waterfall_step_spin.setRange(low, high)
        fraction = (
            0.0
            if high <= low
            else (float(self.waterfall_step) - low) / (high - low)
        )
        self._set_slider_silent(
            self.waterfall_step_slider,
            int(round(1000.0 * np.clip(fraction, 0.0, 1.0))),
        )

    def _sync_waterfall_offset_slider(self) -> None:
        maximum = self._waterfall_offset_maximum()
        previous = self.waterfall_offset_spin.blockSignals(True)
        try:
            self.waterfall_offset_spin.setRange(0.0, maximum)
        finally:
            self.waterfall_offset_spin.blockSignals(previous)
        self._set_slider_silent(
            self.waterfall_offset_slider,
            int(
                round(
                    1000.0
                    * np.clip(float(self.waterfall_offset) / maximum, 0.0, 1.0)
                )
            ),
        )

    def _set_font_size(self, value: float) -> None:
        self.font_size = float(value)
        self._apply_figure_font_size()
        self.canvas.draw_idle()

    def _set_show_binning_title(self, checked: bool, *, redraw: bool = True) -> None:
        self.show_binning_title = bool(checked)
        if self.show_binning_title_check is not None:
            self._set_checkbox_silent(
                self.show_binning_title_check, self.show_binning_title
            )
        self._apply_binning_title()
        if redraw and self.canvas is not None:
            self.canvas.draw_idle()

    def _binning_title_text(self) -> str:
        if not self.show_binning_title or getattr(self.model, "is_point_list", False):
            return ""
        parts = []
        for dim, axis in enumerate(self.data.axes):
            if dim in {self.model.x_dim, self.model.y_dim}:
                continue
            centers = np.asarray(axis.centers, dtype=float)
            if centers.size == 0:
                continue
            selection = self.model.selections.get(
                dim, (float(centers[centers.size // 2]),) * 2
            )
            low, high = sorted((float(selection[0]), float(selection[1])))
            name = (
                "ΔE"
                if axis.role in {"energy", "energy_transfer"}
                or str(axis.name).casefold().replace("_", "")
                in {"deltae", "energy", "energytransfer"}
                else str(axis.name)
            )
            unit = display_unit(axis.units)
            if unit.casefold() == "rlu":
                unit = "r.l.u."
            suffix = f" {unit}" if unit else ""
            parts.append(f"{name}=[{low:.5g},{high:.5g}]{suffix}")
        return ", ".join(parts)

    def _apply_binning_title(self) -> None:
        if self.figure is None:
            return
        title = self._binning_title_text()
        artist = self.figure.suptitle(title)
        artist.set_visible(bool(title))
        artist.set_fontsize(float(self.font_size) + 1.0)

    def _set_axis_linewidth(self, value: float) -> None:
        self.axis_linewidth = float(value)
        self._apply_axis_linewidth()
        self.canvas.draw_idle()

    def _apply_figure_font_size(self) -> None:
        if self.figure is None:
            return
        self._apply_binning_title()
        size = float(self.font_size)
        for axis in (
            self.ax_image,
            self.ax_xcut,
            self.ax_ycut,
            self.ax_colorbar,
            self.ax_residual,
            self.ax_fit_cut,
            self.ax_residual_cut,
            self.ax_residual_ycut,
        ):
            if axis is None:
                continue
            axis.title.set_fontsize(size + 1.0)
            axis.xaxis.label.set_fontsize(size)
            axis.yaxis.label.set_fontsize(size)
            axis.tick_params(axis="both", labelsize=size)
        for axis in self._compare_axes:
            axis.title.set_fontsize(size + 1.0)
            axis.xaxis.label.set_fontsize(size)
            axis.yaxis.label.set_fontsize(size)
            axis.tick_params(axis="both", labelsize=size)
        for axis in self._tile_axes[1:]:
            axis.title.set_fontsize(size + 1.0)
            axis.xaxis.label.set_fontsize(size)
            axis.yaxis.label.set_fontsize(size)
            axis.tick_params(axis="both", labelsize=size)
        if self.colorbar is not None:
            self.colorbar.ax.yaxis.label.set_fontsize(size)
            self.colorbar.ax.tick_params(labelsize=size)
        for colorbar in self._compare_colorbars:
            colorbar.ax.yaxis.label.set_fontsize(size)
            colorbar.ax.tick_params(labelsize=size)

    def _apply_axis_linewidth(self) -> None:
        width = float(self.axis_linewidth)
        for axis in (
            self.ax_image,
            self.ax_xcut,
            self.ax_ycut,
            self.ax_residual,
            self.ax_fit_cut,
            self.ax_residual_cut,
            self.ax_residual_ycut,
        ):
            if axis is None:
                continue
            for spine in axis.spines.values():
                spine.set_linewidth(width)
            axis.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=width)
        for axis in self._compare_axes:
            for spine in axis.spines.values():
                spine.set_linewidth(width)
            axis.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=width)
        for axis in self._tile_axes[1:]:
            for spine in axis.spines.values():
                spine.set_linewidth(width)
            axis.tick_params(axis="both", which="both", direction="in", top=True, right=True, width=width)
        if self.ax_colorbar is not None:
            for spine in self.ax_colorbar.spines.values():
                spine.set_linewidth(width)
            self.ax_colorbar.tick_params(axis="both", which="both", direction="in", width=width)
        if self.colorbar is not None:
            self.colorbar.outline.set_linewidth(width)
            self.colorbar.ax.tick_params(which="both", direction="in", width=width)
        for colorbar in self._compare_colorbars:
            colorbar.outline.set_linewidth(width)
            colorbar.ax.tick_params(which="both", direction="in", width=width)

    def _current_limit_n(self) -> float:
        if self.model.auto_limits == "N-sigma":
            return float(self.model.sigma_n)
        if self.model.auto_limits == "N IQR":
            return float(self.model.iqr_n)
        if self.model.auto_limits == "Nth percentile":
            return float(self.model.percentile_n)
        return 1.0

    def _sync_limit_n_visibility(self) -> None:
        visible = self.model.auto_limits != "min/max"
        self.limit_n_label.setVisible(visible)
        self.limit_n_spin.setVisible(visible)

    def _set_autoscale_checkbox(self, checked: bool) -> None:
        self._syncing_limits = True
        try:
            self.autoscale_check.setChecked(checked)
        finally:
            self._syncing_limits = False
        self.model.autoscale = bool(checked)

    def _connect_view_limit_callbacks(self) -> None:
        if self.ax_image is None:
            return
        for callback_id in self._view_limit_callback_ids:
            try:
                self.ax_image.callbacks.disconnect(callback_id)
            except Exception:
                pass
        self._view_limit_callback_ids = [
            self.ax_image.callbacks.connect("xlim_changed", lambda _axis: self._on_view_limits_changed("x")),
            self.ax_image.callbacks.connect("ylim_changed", lambda _axis: self._on_view_limits_changed("y")),
        ]

    def _on_view_limits_changed(self, axis_name: str) -> None:
        self._sync_view_limit_controls(axis_name)
        self._apply_autoscale_to_view()

    def _visible_values(self, view: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
        """Return the subset of ``values`` whose bin centers fall within the current x/y limits."""

        if self.ax_image is None:
            return values
        values = np.asarray(values, dtype=float)
        x_centers = np.asarray(view.get("x_centers"))
        y_centers = np.asarray(view.get("y_centers"))
        if values.ndim != 2 or x_centers.size != values.shape[1] or y_centers.size != values.shape[0]:
            return values
        xlo, xhi = sorted(self.ax_image.get_xlim())
        ylo, yhi = sorted(self.ax_image.get_ylim())
        x_in = (x_centers >= xlo) & (x_centers <= xhi)
        y_in = (y_centers >= ylo) & (y_centers <= yhi)
        if not x_in.any() or not y_in.any():
            return values
        return values[np.ix_(y_in, x_in)]

    def _apply_autoscale_to_view(self) -> None:
        """Recompute autoscale color limits from data visible within the current x/y limits."""

        if (
            self._autoscaling_view
            or self.ax_image is None
            or self.image is None
            or self._current_slice is None
            or not self.model.autoscale
            or self.model._is_boolean_channel()
            or self._is_effective_1d()
            or self._fit_panels_active()
            or self._tiled_mode_active()
        ):
            return
        self._autoscaling_view = True
        try:
            values = self.model._display_values(self._current_slice)
            subset = self._visible_values(self._current_slice, values)
            self.image.set_norm(self.model._color_norm(subset))
            vmin, vmax = self.model._color_limits(subset)
            if self.colorbar is not None:
                self.colorbar.update_normal(self.image)
                self.colorbar.set_label(self.model._channel_label())
            self._sync_limit_spinboxes(vmin, vmax)
            self.canvas.draw_idle()
        finally:
            self._autoscaling_view = False

    def _set_view_limits(self, axis_name: str) -> None:
        if self._syncing_view_limits or self.ax_image is None:
            return
        if axis_name == "x":
            low = float(self.x_min_spin.value())
            high = float(self.x_max_spin.value())
            if low == high:
                return
            self.ax_image.set_xlim(low, high)
            self._apply_compare_view_limits(xlim=(low, high))
        else:
            low = float(self.y_min_spin.value())
            high = float(self.y_max_spin.value())
            if low == high:
                return
            self.ax_image.set_ylim(low, high)
            self._apply_compare_view_limits(ylim=(low, high))
        self.canvas.draw_idle()

    def _reset_view_limits(self, axis_name: str) -> None:
        if self.ax_image is None:
            return
        if (
            self._waterfall_mode_active()
            and axis_name == "x"
            and self._waterfall_default_xlim is not None
        ):
            self.ax_image.set_xlim(*self._waterfall_default_xlim)
        elif (
            self._waterfall_mode_active()
            and axis_name == "y"
            and self._waterfall_default_ylim is not None
        ):
            self.ax_image.set_ylim(*self._waterfall_default_ylim)
        elif axis_name == "x":
            self.ax_image.set_xlim(*self._default_view_limits(self.model.x_dim))
            self._apply_compare_view_limits(xlim=self.ax_image.get_xlim())
        elif self._is_effective_1d():
            self.ax_image.relim()
            self.ax_image.autoscale(axis="y")
        else:
            self.ax_image.set_ylim(*self._default_view_limits(self.model.y_dim))
            self._apply_compare_view_limits(ylim=self.ax_image.get_ylim())
        self.canvas.draw_idle()

    def _set_navigation_home_to_current_view(self) -> None:
        """Make Matplotlib Home restore the plot's current data extent."""

        if self.toolbar is None:
            return
        self.toolbar.update()
        self.toolbar.push_current()

    def _sync_view_limit_controls(self, axis_name: str | None = None) -> None:
        if self.ax_image is None or self.x_min_spin is None:
            return
        self._syncing_view_limits = True
        try:
            if axis_name in (None, "x"):
                xmin, xmax = self.ax_image.get_xlim()
                self.x_min_spin.setValue(float(xmin))
                self.x_max_spin.setValue(float(xmax))
                self._set_axis_spin_steps((self.x_min_spin, self.x_max_spin), self.model.x_dim)
            if axis_name in (None, "y"):
                ymin, ymax = self.ax_image.get_ylim()
                self.y_min_spin.setValue(float(ymin))
                self.y_max_spin.setValue(float(ymax))
                self._set_axis_spin_steps((self.y_min_spin, self.y_max_spin), self.model.y_dim)
            if self._fit_panels_active():
                self._apply_compare_view_limits(
                    xlim=self.ax_image.get_xlim(),
                    ylim=self.ax_image.get_ylim(),
                )
        finally:
            self._syncing_view_limits = False

    def _apply_compare_view_limits(
        self,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        if not self._compare_axes:
            return
        for axis in self._compare_axes:
            if axis is self.ax_image:
                continue
            if xlim is not None:
                axis.set_xlim(*xlim)
            if ylim is not None:
                axis.set_ylim(*ylim)

    def _default_view_limits(self, dim: int) -> tuple[float, float]:
        if getattr(self.model, "is_point_list", False):
            values = np.asarray(self.model.data.column(self.model.x_key), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return 0.0, 1.0
            return float(np.min(finite)), float(np.max(finite))
        edges = self.model._axis_edges(dim)
        return float(edges[0]), float(edges[-1])

    def _set_axis_spin_steps(self, spinboxes, dim: int) -> None:
        low, high = self._default_view_limits(dim)
        step = max(abs(high - low) / 500.0, 1.0e-6)
        for spinbox in spinboxes:
            spinbox.setSingleStep(step)

    def _sync_control_visibility(self) -> None:
        is_line = self._is_effective_1d()
        is_point = getattr(self.model, "is_point_list", False)
        is_waterfall = self._waterfall_mode_active()
        is_tiled = self._tiled_mode_active()
        grouped_waterfall = is_waterfall and self._waterfall_uses_1d_group()
        compare_active = self._fit_panels_active()
        if self.axes_group is not None:
            self.axes_group.setVisible(
                is_waterfall or is_line or len(self._non_singleton_dims()) >= 2
            )
        if self.axis_selector_widget is not None:
            # Point data is always 1D but still lets the user pick which
            # coordinate is the x axis, so keep the selector visible.
            self.axis_selector_widget.setVisible(is_point or not is_line or is_waterfall)
        if self.y_combo is not None:
            self.y_combo.setVisible(not is_point and not grouped_waterfall)
        if self._axis_y_label is not None:
            self._axis_y_label.setVisible(not is_point and not grouped_waterfall)
        if self.hidden_group is not None:
            self.hidden_group.setVisible(not is_line)
        if self.color_group is not None:
            self.color_group.setVisible(not is_line and not is_waterfall)
        if self.smoothing_group is not None:
            self.smoothing_group.setVisible(not is_point)
        if self.smoothing_y_spin is not None:
            self.smoothing_y_spin.setVisible(not is_line and not grouped_waterfall)
        if self.smoothing_y_label is not None:
            self.smoothing_y_label.setVisible(not is_line and not grouped_waterfall)
        if self.tools_group is not None:
            # The box tool is used both in the standard 2D layout and in 2D
            # fit compare (where it drives the integrated data+fit cut).
            self.tools_group.setVisible(not is_line and not is_waterfall and not is_tiled)
        if self.line_group is not None:
            self.line_group.setVisible(
                not is_tiled and (is_line or compare_active or is_waterfall)
            )
        if self.waterfall_group is not None:
            self.waterfall_group.setVisible(is_waterfall)
        if self.tiled_group is not None:
            self.tiled_group.setVisible(is_tiled)
        for widget in (
            self.coverage_threshold_label,
            self.coverage_threshold_spin,
        ):
            if widget is not None:
                widget.setVisible(not is_point and not is_waterfall)
        if self.waterfall_source_label is not None:
            if grouped_waterfall:
                count = len(self._waterfall_1d_source_indices())
                group_key = self.dataset_group_keys[self.dataset_index]
                group_text = (
                    f" in {group_key}" if group_key and group_key != "root" else ""
                )
                self.waterfall_source_label.setText(
                    f"One trace per compatible 1D dataset{group_text} ({count} traces)"
                )
            elif is_waterfall:
                self.waterfall_source_label.setText(
                    "Coarse bins along "
                    f"{waterfall_axis_display_name(self.data.axes[self.model.y_dim].name)}"
                )
        for widget in (
            self.waterfall_step_label,
            self.waterfall_step_spin,
            self.waterfall_step_slider,
            self.waterfall_step_auto_check,
        ):
            if widget is not None:
                widget.setVisible(is_waterfall and not grouped_waterfall)
        if self.waterfall_color_range_slider is not None:
            show_color_range = (
                is_waterfall
                and self.waterfall_cmap not in _WATERFALL_DISCRETE_COLORMAPS
            )
            self.waterfall_color_range_slider.setVisible(show_color_range)
            self.waterfall_color_range_label.setVisible(show_color_range)
        for widget in (
            self.show_fit_check,
            self.unmask_model_check,
            self.show_residual_check,
        ):
            if widget is not None:
                widget.setVisible(not is_tiled and (widget is not self.show_residual_check or not is_waterfall))
        if self.residual_split_label is not None:
            self.residual_split_label.setVisible(
                not is_waterfall and not is_tiled and self._residual_axes_active()
            )
        if self.residual_split_slider is not None:
            self.residual_split_slider.setVisible(
                not is_waterfall and not is_tiled and self._residual_axes_active()
            )
        self._sync_cursor_visibility()

    def _suppress_matplotlib_coordinate_status(self) -> None:
        axes = (
            self.ax_image,
            self.ax_xcut,
            self.ax_ycut,
            self.ax_colorbar,
            self.ax_residual,
            self.ax_fit_cut,
            self.ax_residual_cut,
            self.ax_residual_ycut,
        )
        for axis in (*axes, *self._compare_axes, *self._tile_axes):
            if axis is not None:
                axis.format_coord = lambda _x, _y: ""

    def _dataset_type_text(self) -> str:
        metadata = getattr(self.data, "metadata", {}) or {}
        candidates: list[Any] = []
        if isinstance(metadata, dict):
            for key in (
                "nfit_data_type",
                "data_type",
                "dataset_type",
                "kind",
                "nfit_dataset_kind",
                "measurement_type",
                "importer",
            ):
                if key in metadata:
                    candidates.append(metadata[key])
        if isinstance(self.data, PointListData):
            candidates.extend(getattr(self.data, "coordinate_names", []))
            candidates.extend(channel.get("label", "") for channel in getattr(self.data, "channels", []))
        else:
            candidates.extend(axis.name for axis in getattr(self.data, "axes", ()))
            candidates.extend(axis.kind for axis in getattr(self.data, "axes", ()))
        return " ".join(str(value).lower() for value in candidates if value is not None)

    def _is_powder_dataset(self) -> bool:
        text = self._dataset_type_text()
        if "powder" in text:
            return True
        if isinstance(self.data, PointListData):
            coordinates = [name.lower() for name in self.data.coordinate_names]
            return any("theta" in name or name in {"q", "|q|", "q_modulus"} for name in coordinates)
        axis_names = [axis.name.lower() for axis in getattr(self.data, "axes", ())]
        has_powder_axis = any("theta" in name or name in {"q", "|q|", "q_modulus"} for name in axis_names)
        has_crystal_axis = any("[h" in name or name in {"h", "k", "l"} for name in axis_names)
        return has_powder_axis and not has_crystal_axis

    def _is_magnetization_dataset(self) -> bool:
        return "magnetization" in self._dataset_type_text() or "mpms" in self._dataset_type_text()

    def _sync_cursor_visibility(self) -> None:
        if self.cursor_hkle_label is not None:
            self.cursor_hkle_label.setVisible(not (self._is_powder_dataset() or self._is_magnetization_dataset()))
        if self.cursor_q_label is not None:
            self.cursor_q_label.setVisible(not self._is_magnetization_dataset())

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
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("waterfall", 1)
        self._suppress_matplotlib_coordinate_status()

    def _ensure_tiled_layout(self, panel_count: int) -> None:
        self.figure.clear()
        columns = int(np.ceil(np.sqrt(panel_count)))
        rows = int(np.ceil(panel_count / columns))
        self.grid = self.figure.add_gridspec(
            rows,
            columns + 1,
            width_ratios=[*[1.0] * columns, 0.055],
        )
        self._tile_axes = []
        for index in range(panel_count):
            row, column = divmod(index, columns)
            first = self._tile_axes[0] if self._tile_axes else None
            self._tile_axes.append(
                self.figure.add_subplot(
                    self.grid[row, column],
                    sharex=first,
                    sharey=first,
                )
            )
        self.ax_image = self._tile_axes[0]
        self.ax_colorbar = self.figure.add_subplot(self.grid[:, -1])
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_residual = None
        self.ax_fit_cut = None
        self.ax_residual_cut = None
        self.ax_residual_ycut = None
        self.image = None
        self.colorbar = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._compare_colorbar_axes = []
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("tiled", panel_count)
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
        self._suppress_matplotlib_coordinate_status()
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
        self._plot_layout_mode = ("residual_1d", 2)

    def _create_rectangle_selector(self) -> None:
        from matplotlib.widgets import RectangleSelector

        self.rectangle_selector = RectangleSelector(
            self.ax_image,
            self._on_rectangle,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=True,
        )
        self.rectangle_selector.set_active(False)
        if self.show_box_check is not None:
            self._set_box_tool_visible(self.show_box_check.isChecked())

    def update_plot(self, *, preserve_view: bool = True) -> None:
        self._sync_fit_channel_controls()
        previous_xlim = self.ax_image.get_xlim() if preserve_view and self._current_slice is not None else None
        previous_ylim = self.ax_image.get_ylim() if preserve_view and self._current_slice is not None else None
        previous_dims = getattr(self, "_last_plot_dims", None)
        current_dims = (self.model.x_dim, self.model.y_dim)
        self._current_slice = self._smoothed_slice_view(self.model.slice_arrays())
        view = self._current_slice
        if self._waterfall_mode_active():
            self._draw_waterfall_view(
                previous_xlim,
                previous_ylim,
                previous_dims,
                current_dims,
            )
            return
        if self._tiled_mode_active():
            self._draw_tiled_view(
                previous_xlim,
                previous_ylim,
                previous_dims,
                current_dims,
            )
            return
        if self._fit_panels_active():
            self._draw_fit_panels_view(previous_xlim, previous_ylim, previous_dims, current_dims)
            return
        if self._residual_axes_active():
            self._draw_1d_with_residual(previous_xlim, previous_dims, current_dims)
            return
        self._ensure_standard_plot_layout()
        for axis in (self.ax_image, self.ax_xcut, self.ax_ycut):
            axis.clear()
        values = self.model._display_values(view)
        vmin, vmax = self.model._color_limits(values)
        if self._is_effective_1d():
            self._draw_1d_view(view, values)
        else:
            self._draw_2d_view(view, values)
        self._sync_control_visibility()
        if previous_xlim is not None and previous_ylim is not None and previous_dims == current_dims:
            self.ax_image.set_xlim(previous_xlim)
            self.ax_image.set_ylim(previous_ylim)
        if self.image is not None and self.colorbar is None:
            self.ax_colorbar.set_visible(True)
            self.colorbar = self.figure.colorbar(self.image, cax=self.ax_colorbar)
        elif self.image is not None:
            self.ax_colorbar.set_visible(True)
            self.colorbar.update_normal(self.image)
        else:
            self.ax_colorbar.set_visible(False)
        if self.colorbar is not None and self.image is not None:
            self.colorbar.set_label(self.model._channel_label())
        self.ax_xcut.set_ylabel("Int.")
        self.ax_ycut.set_xlabel("Int.")
        self._sync_limit_spinboxes(vmin, vmax)
        self._last_plot_dims = current_dims
        self._last_plot_view_mode = "slice"
        self._sync_view_limit_controls()
        if self._roi_extents is None or previous_dims != current_dims:
            self._set_roi_extents(self._default_roi_extents(), update_cuts=False, draw=False)
        else:
            self._set_roi_extents(self._roi_extents, update_cuts=False, draw=False)
        self._apply_histogram_axes_layout(draw=False)
        self._apply_figure_font_size()
        self._apply_axis_linewidth()
        self._connect_view_limit_callbacks()
        self._apply_autoscale_to_view()
        self.canvas.draw_idle()

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
            self.tile_step = default_tiled_slice_step(
                self.data,
                self.tile_dim,
                self.tile_range,
            )
        self._sync_tile_step_slider()
        slices = prepare_mdhisto_tiled_slices(
            self.data,
            x_dim=self.model.x_dim,
            y_dim=self.model.y_dim,
            tile_dim=self.tile_dim,
            channel=self.model.channel,
            selections=self.model.selections,
            integrate_checks=self.model.integrate_checks,
            tile_range=self.tile_range,
            tile_step=self.tile_step,
            coverage_threshold=self.coverage_threshold,
            masked=self.model.masked,
            smoothing_sigma_x=self.smoothing_x,
            smoothing_sigma_y=self.smoothing_y,
        )
        self._current_tiled_slices = slices
        self._current_slice = slices[0].view
        combined = np.concatenate([panel.values.ravel() for panel in slices])
        norm = self.model._color_norm(combined)
        vmin, vmax = self.model._color_limits(combined)
        self._ensure_tiled_layout(len(slices))
        columns = int(np.ceil(np.sqrt(len(slices))))
        rows = int(np.ceil(len(slices) / columns))
        artists = []
        for index, (axis, panel) in enumerate(
            zip(self._tile_axes, slices, strict=True)
        ):
            row, column = divmod(index, columns)
            artist = axis.pcolormesh(
                panel.view["x_edges"],
                panel.view["y_edges"],
                panel.values,
                shading="auto",
                cmap=self.model._effective_cmap(),
                norm=norm,
            )
            axis.text(
                0.97,
                0.03,
                panel.label,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=self.font_size,
            )
            if row == rows - 1 or index + columns >= len(slices):
                axis.set_xlabel(self.model._axis_label(self.model.x_dim))
            if column == 0:
                axis.set_ylabel(self.model._axis_label(self.model.y_dim))
            artists.append(artist)
        self.image = artists[0]
        self.colorbar = self.figure.colorbar(artists[-1], cax=self.ax_colorbar)
        self.colorbar.set_label(self.model._channel_label())
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

    def _draw_waterfall_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        """Draw offset traces for the current map or compatible 1D datasets."""

        self._ensure_waterfall_layout()
        grouped = self._waterfall_uses_1d_group()
        if grouped:
            indices = self._waterfall_1d_source_indices()
            datasets = [self.datasets[index] for index in indices]
            labels = [self.dataset_names[index] for index in indices]
            x_dim = next(
                index for index, size in enumerate(self.data.shape) if size > 1
            )
            traces = prepare_mdhisto_waterfall(
                datasets,
                dataset_labels=labels,
                x_dim=self.data.axes[x_dim].name,
                channel=self.model.channel,
                masked=self.model.masked,
                coverage_threshold=self.waterfall_coverage_threshold,
                smoothing_sigma_x=self.smoothing_x,
                include_model=self.show_fit,
                unmask_model=self.unmask_model,
            )
        else:
            step_low, step_high = self._waterfall_step_limits()
            if self.waterfall_step_auto:
                self.waterfall_step = default_waterfall_step(
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
            traces = prepare_mdhisto_waterfall(
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
                include_model=self.show_fit,
                unmask_model=self.unmask_model,
            )
        self._current_waterfall_traces = traces
        maximum_offset = self._waterfall_offset_maximum()
        if self.waterfall_offset_auto:
            self.waterfall_offset = default_waterfall_offset(traces)
        self.waterfall_offset = float(
            np.clip(self.waterfall_offset, 0.0, maximum_offset)
        )
        self._sync_waterfall_offset_slider()
        self._set_spin_silent(
            self.waterfall_offset_spin,
            self.waterfall_offset,
        )
        colors = waterfall_colors(
            self.waterfall_cmap,
            len(traces),
            low=self.waterfall_color_min,
            high=self.waterfall_color_max,
            reverse=self.waterfall_reverse_colors,
        )
        draw_waterfall_traces(
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
        data_model = self._comparison_panel_model(self.data, self.model.channel)
        data_view = self._smoothed_slice_view(data_model.slice_arrays())
        data_values = data_model._display_values(data_view)
        shared_norm = data_model._color_norm(data_values)
        vmin, vmax = data_model._color_limits(data_values)

        self._current_slice = data_view
        self.image = None
        self.colorbar = None
        for index, (ax, (title, channel)) in enumerate(zip(self._compare_axes, panels, strict=True)):
            model = self._comparison_panel_model(self.data, channel)
            view = self._smoothed_slice_view(model.slice_arrays())
            values = model._display_values(view)
            norm = model._color_norm(values) if title == "Residual" else shared_norm
            artist = ax.pcolormesh(
                view["x_edges"],
                view["y_edges"],
                values,
                shading="auto",
                cmap=self.model._effective_cmap(),
                norm=norm,
            )
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

        if np.any(x_mask) and np.any(y_mask):
            x = x_centers[x_mask]
            y = y_centers[y_mask]
            data_z = self.model._display_values(data_view)
            errors = np.asarray(data_view.get("errors"), dtype=float)
            x_coverage, y_coverage = self._histogram_cut_coverage(
                data_view, x_mask, y_mask
            )
            x_insufficient = x_coverage < self.coverage_threshold
            y_insufficient = y_coverage < self.coverage_threshold
            if errors.shape == data_z.shape:
                selected = np.ix_(y_mask, x_mask)
                data_cut, err_cut = inverse_variance_weighted_profile(
                    data_z[selected], errors[selected], axis=0
                )
                data_y_cut, err_y_cut = inverse_variance_weighted_profile(
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
            fit_model = self._comparison_panel_model(self.data, "fit")
            fit_z = fit_model._display_values(self._smoothed_slice_view(fit_model.slice_arrays()))
            fit_errors = errors
            if self.unmask_model:
                raw_data_model = self._comparison_panel_model(
                    self.data,
                    self.model.channel,
                    masked=False,
                )
                raw_data_view = self._smoothed_slice_view(raw_data_model.slice_arrays())
                fit_errors = np.asarray(raw_data_view.get("errors"), dtype=float)
            if fit_errors.shape == data_z.shape:
                fit_cut, _ = inverse_variance_weighted_profile(
                    fit_z[selected], fit_errors[selected], axis=0
                )
                fit_y_cut, _ = inverse_variance_weighted_profile(
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
                residual_model = self._comparison_panel_model(self.data, "residual")
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
        return model

    def _is_effective_1d(self) -> bool:
        if getattr(self.model, "is_point_list", False):
            return True
        return sum(size > 1 for size in self.data.shape) == 1

    def _waterfall_mode_active(self) -> bool:
        return self.view_mode_combo is not None and self.view_mode_combo.currentIndex() == 1

    def _tiled_mode_active(self) -> bool:
        return self.view_mode_combo is not None and self.view_mode_combo.currentIndex() == 2

    def _waterfall_uses_1d_group(self) -> bool:
        return (
            isinstance(self.data, MDHistoData)
            and sum(size > 1 for size in self.data.shape) == 1
        )

    def _waterfall_1d_source_indices(self) -> list[int]:
        """Return compatible 1D datasets represented by the current trace group."""

        if not self._waterfall_uses_1d_group():
            return []
        current_dim = next(index for index, size in enumerate(self.data.shape) if size > 1)
        current_axis = self.data.axes[current_dim]
        current_group_key = self.dataset_group_keys[self.dataset_index]
        indices = []
        for index, dataset in enumerate(self.datasets):
            if self.dataset_group_keys[index] != current_group_key:
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

        if not self._waterfall_mode_active() or not self._waterfall_uses_1d_group():
            return [self.dataset_names[self.dataset_index]]
        return [
            self.dataset_names[index] for index in self._waterfall_1d_source_indices()
        ]

    def _slice_1d_channel(self, view: dict[str, np.ndarray], name: str) -> np.ndarray | None:
        """Return a fit/residual channel from the current slice as a 1D array."""

        channel_view = view
        if (
            self.unmask_model
            and name in {"fit", "residual"}
            and not getattr(self.model, "is_point_list", False)
        ):
            channel_model = self._comparison_panel_model(self.data, name, masked=False)
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
            fit_values = self._slice_1d_channel(view, "fit")
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
        self._draw_1d_view(view, values)

        x = np.asarray(view["x_centers"], dtype=float)
        residual = self._slice_1d_channel(view, "residual")
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
            cmap=self.model._effective_cmap(),
            norm=norm,
        )
        self.ax_image.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_image.set_ylabel(self.model._axis_label(self.model.y_dim))
        self._draw_bragg_peak_overlay()

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

    def _sync_limit_spinboxes(self, vmin: float, vmax: float) -> None:
        if not self.model.autoscale:
            return
        self._syncing_limits = True
        try:
            self.vmin_spin.setValue(float(vmin))
            self.vmax_spin.setValue(float(vmax))
        finally:
            self._syncing_limits = False

    def _set_roi_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.show_box_check is not None and not self.show_box_check.isChecked():
                self.show_box_check.setChecked(True)
            if self.hist_axes_check is not None and not self.hist_axes_check.isChecked():
                self.hist_axes_check.setChecked(True)
        if self.rectangle_selector is not None:
            visible = self.show_box_check is None or self.show_box_check.isChecked()
            self.rectangle_selector.set_active(bool(enabled) and visible)
            self._set_rectangle_selector_style(active=bool(enabled) and visible)
            self.canvas.draw_idle()

    def _set_box_tool_visible(self, visible: bool) -> None:
        if visible and not self._box_tool_has_auto_shown_hist_axes:
            self._box_tool_has_auto_shown_hist_axes = True
            if self.hist_axes_check is not None and not self.hist_axes_check.isChecked():
                self.hist_axes_check.setChecked(True)
        if not visible:
            self.roi_button.setChecked(False)
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_visible(bool(visible))
            self.rectangle_selector.set_active(bool(visible) and self.roi_button.isChecked())
            self._set_rectangle_selector_style(active=bool(visible) and self.roi_button.isChecked())
        self.canvas.draw_idle()

    def _set_rectangle_selector_style(self, *, active: bool) -> None:
        if self.rectangle_selector is None:
            return
        color = "#4f8bd6" if active else "#8a8a8a"
        self.rectangle_selector.set_props(
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
            alpha=1.0 if active else 0.75,
        )
        self.rectangle_selector.set_handle_props(
            markeredgecolor=color,
            markerfacecolor=color,
            alpha=1.0 if active else 0.75,
        )

    def _set_histogram_axes_visible(self, visible: bool) -> None:
        self._sync_histogram_panel_controls()
        if self._fit_panels_active():
            # Toggling the cut axes changes the fit-compare layout, so rebuild.
            self.update_plot()
            return
        self._apply_histogram_axes_layout(draw=True)

    def _set_xcut_percent(self, value: int) -> None:
        self.xcut_percent = int(value)
        self._sync_histogram_panel_controls()
        if self._fit_panels_active():
            if self.ax_fit_cut is not None and self.grid is not None:
                self.grid.set_height_ratios([1.0, self._panel_ratio(self.xcut_percent)])
                self.canvas.draw_idle()
            return
        self._apply_histogram_axes_layout(draw=True)

    def _set_ycut_percent(self, value: int) -> None:
        self.ycut_percent = int(value)
        self._sync_histogram_panel_controls()
        if self._fit_panels_active() and self._fit_cuts_active():
            self.update_plot()
            return
        self._apply_histogram_axes_layout(draw=True)

    def _sync_histogram_panel_controls(self) -> None:
        if self.hist_axes_check is None:
            return
        visible = bool(self.hist_axes_check.isChecked()) and not self._is_effective_1d()
        self.xcut_percent_slider.setEnabled(visible)
        self.ycut_percent_slider.setEnabled(visible)
        self.xcut_percent_label.setText(f"X cut height: {self.xcut_percent}%")
        self.ycut_percent_label.setText(f"Y cut width: {self.ycut_percent}%")

    def _apply_histogram_axes_layout(self, *, draw: bool) -> None:
        if self.grid is None or self.hist_axes_check is None:
            return
        # The x/y cut panels only exist in the standard 2D layout, whose grid
        # is 2 rows by 3 columns. In the 1D residual and fit-compare layouts
        # the grid has a different shape, so touching its ratios here would
        # raise; guard on the actual grid shape rather than the (possibly
        # stale) cut-axis references, which can survive a figure rebuild.
        if self._plot_layout_mode != ("standard", 1):
            return
        if self.ax_xcut is None or self.ax_ycut is None:
            return
        if getattr(self.grid, "nrows", None) != 2 or getattr(self.grid, "ncols", None) != 3:
            return
        visible = bool(self.hist_axes_check.isChecked()) and not self._is_effective_1d()
        x_ratio = self._panel_ratio(self.xcut_percent) if visible else 0.001
        y_ratio = self._panel_ratio(self.ycut_percent) if visible else 0.001
        self.grid.set_height_ratios([1.0, x_ratio])
        self.grid.set_width_ratios([1.0, y_ratio, 0.045])
        self.ax_xcut.set_visible(visible)
        self.ax_ycut.set_visible(visible)
        if draw:
            self.canvas.draw_idle()

    def _panel_ratio(self, percent: float) -> float:
        fraction = float(np.clip(percent, 1.0, 80.0)) / 100.0
        return fraction / (1.0 - fraction)

    def _set_roi_from_controls(self) -> None:
        if self._syncing_roi_controls:
            return
        x_center = float(self.roi_x_center_spin.value())
        y_center = float(self.roi_y_center_spin.value())
        x_width = max(float(self.roi_x_width_spin.value()), 0.0)
        y_width = max(float(self.roi_y_width_spin.value()), 0.0)
        extents = (
            x_center - 0.5 * x_width,
            x_center + 0.5 * x_width,
            y_center - 0.5 * y_width,
            y_center + 0.5 * y_width,
        )
        self._set_roi_extents(extents, update_cuts=True, draw=True)

    def _set_roi_extents(
        self,
        extents: tuple[float, float, float, float],
        *,
        update_cuts: bool,
        draw: bool,
    ) -> None:
        x0, x1, y0, y1 = self._normalize_roi_extents(extents)
        if self.rectangle_selector is not None:
            self.rectangle_selector.extents = (x0, x1, y0, y1)
            x0, x1, y0, y1 = self._normalize_roi_extents(self.rectangle_selector.extents)
        self._roi_extents = (x0, x1, y0, y1)
        self._sync_roi_controls_from_extents()
        if update_cuts:
            self._update_histogram_cuts_from_extents(self._roi_extents)
        if draw:
            self.canvas.draw_idle()

    def _sync_roi_controls_from_extents(self) -> None:
        if self.roi_x_center_spin is None or self._roi_extents is None:
            return
        x0, x1, y0, y1 = self._roi_extents
        self._syncing_roi_controls = True
        try:
            self._set_axis_spin_steps((self.roi_x_center_spin, self.roi_x_width_spin), self.model.x_dim)
            self._set_axis_spin_steps((self.roi_y_center_spin, self.roi_y_width_spin), self.model.y_dim)
            self.roi_x_center_spin.setValue(0.5 * (x0 + x1))
            self.roi_x_width_spin.setValue(abs(x1 - x0))
            self.roi_y_center_spin.setValue(0.5 * (y0 + y1))
            self.roi_y_width_spin.setValue(abs(y1 - y0))
        finally:
            self._syncing_roi_controls = False

    def _default_roi_extents(self) -> tuple[float, float, float, float]:
        xmin, xmax = self._default_view_limits(self.model.x_dim)
        ymin, ymax = self._default_view_limits(self.model.y_dim)
        x_width = 0.5 * abs(xmax - xmin)
        y_width = 0.5 * abs(ymax - ymin)
        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)
        return (
            x_center - 0.5 * x_width,
            x_center + 0.5 * x_width,
            y_center - 0.5 * y_width,
            y_center + 0.5 * y_width,
        )

    def _normalize_roi_extents(self, extents) -> tuple[float, float, float, float]:
        x0, x1, y0, y1 = [float(value) for value in extents]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return x0, x1, y0, y1

    def _current_roi_extents(self, click=None, release=None) -> tuple[float, float, float, float] | None:
        if self.rectangle_selector is not None:
            extents = self._normalize_roi_extents(self.rectangle_selector.extents)
            if extents[0] != extents[1] and extents[2] != extents[3]:
                return extents
        if click is None or release is None:
            return self._roi_extents
        if click.xdata is None or release.xdata is None or click.ydata is None or release.ydata is None:
            return None
        return self._normalize_roi_extents((click.xdata, release.xdata, click.ydata, release.ydata))

    def _update_histogram_cuts_from_extents(self, extents: tuple[float, float, float, float] | None) -> None:
        if self._fit_cuts_active():
            self._update_fit_compare_cuts(extents)
            self.canvas.draw_idle()
            return
        if self.ax_xcut is None or self.ax_ycut is None:
            return
        if self._current_slice is None or extents is None:
            return
        x0, x1, y0, y1 = extents
        view = self._current_slice
        x_mask = (view["x_centers"] >= x0) & (view["x_centers"] <= x1)
        y_mask = (view["y_centers"] >= y0) & (view["y_centers"] <= y1)
        self.ax_xcut.clear()
        self.ax_ycut.clear()
        if np.any(x_mask) and np.any(y_mask):
            z = self.model._display_values(view)
            errors = np.asarray(view["errors"], dtype=float)
            selected = np.ix_(y_mask, x_mask)
            x_cut, x_error = inverse_variance_weighted_profile(
                z[selected], errors[selected], axis=0
            )
            y_cut, y_error = inverse_variance_weighted_profile(
                z[selected], errors[selected], axis=1
            )
            x_coverage, y_coverage = self._histogram_cut_coverage(
                view, x_mask, y_mask
            )
            x_insufficient = x_coverage < self.coverage_threshold
            y_insufficient = y_coverage < self.coverage_threshold
            x_cut = np.where(x_insufficient, np.nan, x_cut)
            x_error = np.where(x_insufficient, np.nan, x_error)
            y_cut = np.where(y_insufficient, np.nan, y_cut)
            y_error = np.where(y_insufficient, np.nan, y_error)
            self.ax_xcut.errorbar(
                view["x_centers"][x_mask], x_cut, yerr=x_error, fmt="-", lw=1.2, capsize=0
            )
            self.ax_ycut.errorbar(
                y_cut, view["y_centers"][y_mask], xerr=y_error, fmt="-", lw=1.2, capsize=0
            )
        self.ax_xcut.set_ylabel("Weighted mean")
        self.ax_xcut.set_xlabel(self.model._axis_label(self.model.x_dim))
        self.ax_ycut.set_xlabel("Weighted mean")
        self.ax_ycut.set_ylabel(self.model._axis_label(self.model.y_dim))
        self._apply_histogram_axes_layout(draw=False)

    def _histogram_cut_coverage(
        self,
        view: dict[str, np.ndarray],
        x_mask: np.ndarray,
        y_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coverage = np.asarray(view["coverage_fraction"], dtype=float)[
            np.ix_(y_mask, x_mask)
        ]
        x_widths = np.diff(np.asarray(view["x_edges"], dtype=float))[x_mask]
        y_widths = np.diff(np.asarray(view["y_edges"], dtype=float))[y_mask]
        x_coverage = np.sum(coverage * y_widths[:, None], axis=0) / np.sum(
            y_widths
        )
        y_coverage = np.sum(coverage * x_widths[None, :], axis=1) / np.sum(
            x_widths
        )
        return x_coverage, y_coverage

    def _on_motion(self, event) -> None:
        if self._current_slice is None or event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._waterfall_mode_active():
            self._on_waterfall_motion(event)
            return
        if self._is_effective_1d():
            self._on_line_motion(event)
            return
        view = self._current_slice
        x_idx = int(np.searchsorted(view["x_edges"], event.xdata, side="right") - 1)
        y_idx = int(np.searchsorted(view["y_edges"], event.ydata, side="right") - 1)
        if not (0 <= x_idx < view["signal"].shape[1] and 0 <= y_idx < view["signal"].shape[0]):
            return
        values = self.model._display_values(view)
        coords = self._cursor_hkle(x_idx, y_idx)
        value_text, error_text = _format_value_with_uncertainty(
            float(values[y_idx, x_idx]),
            float(view["errors"][y_idx, x_idx]),
        )
        self.cursor_xy_label.setText(
            f"(x, y) = ({_format_coord(view['x_centers'][x_idx])}, {_format_coord(view['y_centers'][y_idx])})"
        )
        self.cursor_hkle_label.setText(
            "(H, K, L, E) = "
            f"({_format_coord(coords['H'])}, {_format_coord(coords['K'])}, "
            f"{_format_coord(coords['L'])}, {_format_coord(coords['E'])})"
        )
        self.cursor_q_label.setText(self._format_q_modulus(coords))
        coverage = float(view["coverage_fraction"][y_idx, x_idx])
        self.cursor_intensity_label.setText(
            f"Signal = {value_text} ± {error_text}; coverage = {coverage:.1%}"
        )

    def _on_waterfall_motion(self, event) -> None:
        candidates = []
        for trace_index, trace in enumerate(self._current_waterfall_traces):
            x = np.asarray(trace.x, dtype=float)
            if x.size == 0 or not np.any(np.isfinite(x)):
                continue
            x_index = int(np.nanargmin(np.abs(x - float(event.xdata))))
            values = np.asarray(trace.values, dtype=float)
            if x_index >= values.size or not np.isfinite(values[x_index]):
                continue
            plotted = float(values[x_index]) + trace_index * self.waterfall_offset
            candidates.append(
                (abs(plotted - float(event.ydata)), trace_index, x_index, plotted)
            )
        if not candidates:
            return
        _distance, trace_index, x_index, plotted = min(candidates)
        trace = self._current_waterfall_traces[trace_index]
        value = float(trace.values[x_index])
        errors = (
            np.asarray(trace.errors, dtype=float)
            if trace.errors is not None
            else np.array([], dtype=float)
        )
        error = float(errors[x_index]) if x_index < errors.size else np.nan
        value_text, error_text = _format_value_with_uncertainty(value, error)
        x_value = float(trace.x[x_index])
        self.cursor_xy_label.setText(
            f"(x, y+offset) = ({_format_coord(x_value)}, {_format_coord(plotted)})"
        )
        self.cursor_hkle_label.setText(f"Trace = {trace.label}")
        self.cursor_intensity_label.setText(
            f"{trace.label}: {value_text} ± {error_text}"
        )

    def _on_line_motion(self, event) -> None:
        view = self._current_slice
        x = np.asarray(view["x_centers"], dtype=float)
        if x.size == 0:
            return
        x_idx = int(np.nanargmin(np.abs(x - float(event.xdata))))
        values = np.asarray(self.model._display_values(view), dtype=float).reshape(-1)
        errors = np.asarray(view["errors"], dtype=float).reshape(-1)
        if not (0 <= x_idx < values.size):
            return
        value = float(values[x_idx])
        error = float(errors[x_idx]) if x_idx < errors.size else np.nan
        coords = self._cursor_hkle_1d(x_idx)
        value_text, error_text = _format_value_with_uncertainty(value, error)
        self.cursor_xy_label.setText(f"(x, y) = ({_format_coord(x[x_idx])}, {_format_coord(value)})")
        self.cursor_hkle_label.setText(
            "(H, K, L, E) = "
            f"({_format_coord(coords['H'])}, {_format_coord(coords['K'])}, "
            f"{_format_coord(coords['L'])}, {_format_coord(coords['E'])})"
        )
        self.cursor_q_label.setText(self._format_q_modulus(coords))
        self.cursor_intensity_label.setText(f"Signal = {value_text} ± {error_text}")

    def _format_q_modulus(self, coords: dict[str, float]) -> str:
        q = self._q_modulus_inv_angstrom(coords)
        if q is None or not np.isfinite(q):
            return "|Q| = ? Å⁻¹"
        return f"|Q| = {_format_coord(q)} Å⁻¹"

    def _q_modulus_inv_angstrom(self, coords: dict[str, float]) -> float | None:
        q_modulus = coords.get("q_modulus")
        if q_modulus is not None and np.isfinite(q_modulus):
            return abs(float(q_modulus))
        hkl = np.asarray([coords["H"], coords["K"], coords["L"]], dtype=float)
        if hkl.shape != (3,) or not np.all(np.isfinite(hkl)):
            return None
        try:
            from .fitting import (
                _as_3x3_matrix,
                _metadata_coordinate_units_are_inv_angstrom,
                _resolve_q_transform,
            )

            if _metadata_coordinate_units_are_inv_angstrom(self.data.metadata):
                q_vector = hkl
            elif "rlu_to_inv_angstrom_matrix" in self.data.metadata:
                q_vector = _as_3x3_matrix(
                    self.data.metadata["rlu_to_inv_angstrom_matrix"],
                    name="rlu_to_inv_angstrom_matrix",
                ) @ hkl
            elif "ub_matrix" in self.data.metadata:
                q_vector = _cursor_q_matrix_from_metadata(self.data.metadata, "ub_matrix") @ hkl
            elif "orientation_matrix" in self.data.metadata:
                q_vector = _cursor_q_matrix_from_metadata(self.data.metadata, "orientation_matrix") @ hkl
            elif isinstance(self.data.metadata.get("oriented_lattice"), dict):
                q_vector = _cursor_q_matrix_from_oriented_lattice(self.data.metadata["oriented_lattice"]) @ hkl
            else:
                q_vector = _resolve_q_transform(self.data) @ hkl
        except (KeyError, TypeError, ValueError):
            return None
        return float(np.linalg.norm(q_vector))

    def _cursor_hkle(self, x_idx: int, y_idx: int) -> dict[str, float]:
        hkle = np.zeros(4, dtype=float)
        has_energy = False
        hidden = self.model._normalized_selections()
        for dim, _axis in enumerate(self.data.axes):
            if dim == self.model.x_dim:
                value = self.data.axes[dim].centers[x_idx]
            elif dim == self.model.y_dim:
                value = self.data.axes[dim].centers[y_idx]
            else:
                selection = hidden.get(dim)
                if isinstance(selection, tuple):
                    value = float(np.mean(self.data.axes[dim].centers[selection[0] : selection[1] + 1]))
                elif selection is None:
                    continue
                else:
                    value = self.data.axes[dim].centers[int(selection)]
            vector = _cursor_axis_hkle_vector(self.data, dim)
            hkle += float(value) * vector
            has_energy = has_energy or bool(vector[3])
        return {"H": hkle[0], "K": hkle[1], "L": hkle[2], "E": hkle[3] if has_energy else np.nan}

    def _cursor_hkle_1d(self, x_idx: int) -> dict[str, float]:
        if getattr(self.model, "is_point_list", False):
            return self._cursor_hkle_point_list_1d(x_idx)
        hkle = np.zeros(4, dtype=float)
        has_energy = False
        hidden = self.model._normalized_selections()
        for dim, _axis in enumerate(self.data.axes):
            if dim == self.model.x_dim:
                value = self.data.axes[dim].centers[x_idx]
            else:
                selection = hidden.get(dim)
                if isinstance(selection, tuple):
                    value = float(np.mean(self.data.axes[dim].centers[selection[0] : selection[1] + 1]))
                elif selection is None:
                    value = self.data.axes[dim].centers[0]
                else:
                    value = self.data.axes[dim].centers[int(selection)]
            vector = _cursor_axis_hkle_vector(self.data, dim)
            hkle += float(value) * vector
            has_energy = has_energy or bool(vector[3])
        return {"H": hkle[0], "K": hkle[1], "L": hkle[2], "E": hkle[3] if has_energy else np.nan}

    def _cursor_hkle_point_list_1d(self, x_idx: int) -> dict[str, float]:
        coords = {"H": 0.0, "K": 0.0, "L": 0.0, "E": np.nan}
        data = self.model.data
        x = np.asarray(data.column(self.model.x_key), dtype=float)
        if x.size == 0:
            return coords
        order = np.argsort(x, kind="stable")
        if not (0 <= int(x_idx) < order.size):
            return coords
        row = int(order[int(x_idx)])
        q_modulus = self._point_list_row_q_modulus(data, row)
        if q_modulus is not None:
            coords["q_modulus"] = q_modulus
        for name in data.coordinate_names:
            components = _axis_components(name)
            if not components:
                continue
            value = float(data.column(name)[row])
            for component, coefficient in components.items():
                if component == "E":
                    coords["E"] = value
                else:
                    coords[component] += float(coefficient) * value
        return coords

    def _point_list_row_q_modulus(self, data: PointListData, row: int) -> float | None:
        q_name = _point_list_q_column_name(data)
        if q_name is not None:
            q_values = np.asarray(data.column(q_name), dtype=float)
            if 0 <= row < q_values.size and np.isfinite(q_values[row]):
                return abs(float(q_values[row]))
        two_theta_name = _point_list_two_theta_column_name(data)
        wavelength = _point_list_wavelength(data, two_theta_name)
        if two_theta_name is None or wavelength is None or wavelength <= 0.0:
            return None
        two_theta = np.asarray(data.column(two_theta_name), dtype=float)
        if not (0 <= row < two_theta.size) or not np.isfinite(two_theta[row]):
            return None
        theta = np.deg2rad(float(two_theta[row])) / 2.0
        return float(4.0 * np.pi * np.sin(theta) / wavelength)

    def _on_rectangle(self, click, release) -> None:
        extents = self._current_roi_extents(click, release)
        if extents is None:
            return
        self._set_roi_extents(extents, update_cuts=True, draw=True)
