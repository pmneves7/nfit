from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .application_preferences import (
    default_continuous_colormap,
    default_waterfall_colormap,
)
from .colormaps import (
    IMAGE_COLORMAP_GROUPS,  # noqa: F401 - compatibility re-export
    WATERFALL_COLORMAP_GROUPS,  # noqa: F401 - compatibility re-export
    WATERFALL_COLORMAPS,
    WATERFALL_DISCRETE_COLORMAPS,
    populate_qt_colormap_combo,  # noqa: F401 - compatibility re-export
)
from .dataset import PointListData
from .file_dialogs import get_save_file_name
from .mdhisto import MDHistoData
from .plotting_core import (
    MDHistoSliceViewer,
    TiledSlice,
    WaterfallTrace,
    _draw_box_sum_annotation,
    coarsen_mdhisto_view,
    default_tiled_slice_step,
    default_waterfall_offset,  # noqa: F401 - compatibility re-export
    default_waterfall_step,  # noqa: F401 - compatibility re-export
    draw_mdhisto_brillouin_zones,
    draw_waterfall_traces,  # noqa: F401 - compatibility re-export
    integrated_box_sum,
    inverse_variance_weighted_profile,
    mdhisto_view_native_step,
    prepare_mdhisto_tiled_slices,  # noqa: F401 - compatibility re-export
    prepare_mdhisto_waterfall,  # noqa: F401 - compatibility re-export
    smooth_mdhisto_view,
    waterfall_absolute_max,
    waterfall_axis_display_name,
    waterfall_colors,  # noqa: F401 - compatibility re-export
    waterfall_step_bounds,
)
from .qt_controls import configure_numeric_spin_boxes
from .qt_slice_controls import (
    _clear_layout,
    _compact_combobox,  # noqa: F401 - compatibility re-export
    _DualRangeSlider,  # noqa: F401 - compatibility re-export
    _expanding_combobox,  # noqa: F401 - compatibility re-export
    _HiddenAxisControls,
    _IntegratedAxisSlider,
    _make_data_viewer_window_class,  # noqa: F401 - compatibility re-export
    _make_float_spinbox,
    _make_index_slider,  # noqa: F401 - compatibility re-export
    _qt_app,
)
from .qt_slice_modes import SliceModeControllers
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
    "light red": "#e57373",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "black": "#000000",
}
_WATERFALL_COLORMAPS = WATERFALL_COLORMAPS
_WATERFALL_DISCRETE_COLORMAPS = WATERFALL_DISCRETE_COLORMAPS


class QtMDHistoSliceViewer:
    """PySide6 slice viewer with an embedded Matplotlib canvas."""

    def __init__(
        self,
        data: MDHistoData | Sequence[MDHistoData],
        *,
        dataset_names: Sequence[str] | None = None,
        dataset_group_keys: Sequence[str] | None = None,
        crystal_contexts: Sequence[dict[str, Any]] | None = None,
        x_dim: int | str = -1,
        y_dim: int | str = 0,
        channel: str = "signal",
        cmap: str | None = None,
        color_scale: str = "linear",
        color_alpha: float = 0.0,
        auto_limits: str = "min/max",
        integrate: bool = False,
        masked: bool = True,
    ) -> None:
        self.app = _qt_app()
        cmap = default_continuous_colormap() if cmap is None else cmap
        self.datasets = _coerce_datasets(data)
        self.dataset_names = _coerce_dataset_names(self.datasets, dataset_names)
        self.source_dataset_names = [
            str(getattr(item, "metadata", {}).get("source_dataset_name", name))
            for item, name in zip(self.datasets, self.dataset_names, strict=True)
        ]
        self.binning_names = [
            str(getattr(item, "metadata", {}).get("binning_name", "Default"))
            for item in self.datasets
        ]
        self.dataset_group_keys = _coerce_dataset_group_keys(
            self.datasets,
            dataset_group_keys,
        )
        if crystal_contexts is None:
            self.crystal_contexts = [{} for _ in self.datasets]
        else:
            self.crystal_contexts = [dict(value) for value in crystal_contexts]
            if len(self.crystal_contexts) != len(self.datasets):
                raise ValueError("crystal_contexts length must match datasets length")
        self.dataset_index = 0
        self._initial_x_dim = x_dim
        self._initial_y_dim = y_dim
        self._initial_channel = channel
        self._initial_cmap = cmap
        self._initial_color_scale = color_scale
        self._initial_color_alpha = float(color_alpha)
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
            color_alpha=color_alpha,
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
        self.binning_combo = None
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
        self.x_step_spin = None
        self.y_step_spin = None
        self.step_header_label = None
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
        self.tile_local_color_scales_check = None
        self.vmin_spin = None
        self.vmax_spin = None
        self.color_group = None
        self.smoothing_group = None
        self.smoothing_x_spin = None
        self.smoothing_y_spin = None
        self.smoothing_y_label = None
        self.smoothing_fill_nans_check = None
        self.brillouin_zone_group = None
        self.show_brillouin_zone_check = None
        self.show_major_gridlines_check = None
        self.brillouin_zone_color_combo = None
        self.brillouin_zone_linewidth_spin = None
        self.brillouin_zone_alpha_spin = None
        self.gamma_label = None
        self.gamma_spin = None
        self.alpha_label = None
        self.alpha_spin = None
        self.limit_n_label = None
        self.limit_n_spin = None
        self.cursor_xy_label = None
        self.cursor_hkle_label = None
        self.cursor_q_label = None
        self.cursor_intensity_label = None
        self.roi_button = None
        self.roi_sum_text = None
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
        self.tile_label_decimals_spin = None
        self.show_tile_labels_check = None
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
        self.save_new_plot_button = None
        self.copy_script_button = None
        self.save_script_button = None
        self._save_plot_callback = None
        self._save_new_plot_callback = None
        self._save_project_callback = None
        self._open_new_viewer_callback = None
        self._close_callback = None
        self._child_viewers = []
        self.open_new_viewer_button = None
        self.open_kpath_viewer_button = None
        self.store_plot_status_label = None
        self.save_project_shortcut = None
        self._unmask_model_callback = None
        self._brillouin_zone_context_callback = None
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
        self.smoothing_fill_nans = False
        self.show_brillouin_zone_boundaries = False
        self.show_major_gridlines = False
        self.brillouin_zone_color = "#e57373"
        self.brillouin_zone_linewidth = 1.5
        self.brillouin_zone_alpha = 1.0
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
        self.waterfall_cmap = default_waterfall_colormap()
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
        self.tile_label_decimals = 1
        self.tile_label_prefix = "{axis} = "
        self.tile_label_unit = "{unit}"
        self.tile_label_si_prefix = ""
        self.show_tile_labels = True
        self.tile_local_color_scales = False
        self._current_tiled_slices: list[TiledSlice] = []
        self._tile_axes = []
        self._tile_colorbar_axes = []
        self._tile_colorbars = []
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
        self._syncing_display_steps = False
        self._autoscaling_view = False
        self._syncing_roi_controls = False
        self._box_tool_has_auto_shown_hist_axes = False
        self._restoring_dataset_state = False
        self._roi_extents: tuple[float, float, float, float] | None = None
        self._view_limit_callback_ids: list[int] = []
        self.bragg_peak_overlay: dict[str, Any] | None = None
        self.display_step_factors: dict[int, int] = {}
        self._dataset_states: list[_DatasetViewState | None] = [None] * len(self.datasets)
        self._dataset_states[0] = _DatasetViewState(
            model=self.model,
            show_fit=self.show_fit,
            tile_dim=self.tile_dim,
            tile_range=self.tile_range,
            tile_step=self.tile_step,
            tile_step_auto=self.tile_step_auto,
            tile_label_decimals=self.tile_label_decimals,
            tile_label_prefix=self.tile_label_prefix,
            tile_label_unit=self.tile_label_unit,
            tile_label_si_prefix=self.tile_label_si_prefix,
            show_tile_labels=self.show_tile_labels,
            tile_local_color_scales=self.tile_local_color_scales,
            display_step_factors=dict(self.display_step_factors),
        )
        self._plot_layout_mode: tuple[Any, ...] | None = None
        self._compare_axes = []
        self._compare_colorbars = []
        self._mode_controllers = SliceModeControllers(self, helpers=globals())
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
        crystal_contexts: Sequence[dict[str, Any]] | None = None,
        selected_dataset_name: str | None = None,
        selected_binning_name: str | None = None,
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
        self.source_dataset_names = [
            str(getattr(item, "metadata", {}).get("source_dataset_name", name))
            for item, name in zip(self.datasets, self.dataset_names, strict=True)
        ]
        self.binning_names = [
            str(getattr(item, "metadata", {}).get("binning_name", "Default"))
            for item in self.datasets
        ]
        self.dataset_group_keys = _coerce_dataset_group_keys(
            self.datasets,
            dataset_group_keys,
        )
        if crystal_contexts is None:
            self.crystal_contexts = [{} for _ in self.datasets]
        else:
            self.crystal_contexts = [dict(value) for value in crystal_contexts]
            if len(self.crystal_contexts) != len(self.datasets):
                raise ValueError("crystal_contexts length must match datasets length")
        if self.volume_panel is not None:
            self.content_stack.setCurrentIndex(0)
            self.view_mode_combo.setCurrentIndex(0)
            self._close_volume_panel()
        if selected_dataset_name is not None:
            candidates = [
                index
                for index, source in enumerate(self.source_dataset_names)
                if source == selected_dataset_name
            ]
            preferred = next(
                (
                    index
                    for index in candidates
                    if self.binning_names[index] == selected_binning_name
                ),
                candidates[0] if candidates else None,
            )
            new_index = preferred if preferred is not None else min(self.dataset_index, len(self.datasets) - 1)
        elif current_name in self.dataset_names:
            new_index = self.dataset_names.index(current_name)
        else:
            new_index = min(self.dataset_index, len(self.datasets) - 1)
        self._dataset_states = [None] * len(self.datasets)
        for index, name in enumerate(self.dataset_names):
            state = previous_states.get(name)
            if state is not None:
                tile_dim = state.tile_dim
                if (
                    tile_dim is not None
                    and isinstance(state.model.data, MDHistoData)
                    and isinstance(self.datasets[index], MDHistoData)
                    and tile_dim < len(state.model.data.axes)
                    and tile_dim < len(self.datasets[index].axes)
                ):
                    old_axis = state.model.data.axes[tile_dim]
                    new_axis = self.datasets[index].axes[tile_dim]
                    if (
                        "metadata_dimension" in old_axis.metadata
                        and old_axis.name == new_axis.name
                        and np.allclose(state.tile_range, [old_axis.centers[0], old_axis.centers[-1]])
                    ):
                        state.tile_range = (float(new_axis.centers[0]), float(new_axis.centers[-1]))
                state.model.data = self.datasets[index]
                state.model.refresh_metadata_channels()
                self._dataset_states[index] = state
        state = self._dataset_states[new_index]
        if state is None:
            state = self._default_dataset_state(new_index)
            self._dataset_states[new_index] = state
        self.dataset_index = new_index
        self._sync_dataset_binning_combos()
        self._restore_dataset_state(state)

    def _source_dataset_options(self) -> list[str]:
        return list(dict.fromkeys(self.source_dataset_names))

    def _sync_dataset_binning_combos(self) -> None:
        if self.dataset_combo is None or self.binning_combo is None:
            return
        source = self.source_dataset_names[self.dataset_index]
        options = self._source_dataset_options()
        self._set_combo_items_silent(self.dataset_combo, options, source)
        indices = [
            index for index, value in enumerate(self.source_dataset_names) if value == source
        ]
        names = [self.binning_names[index] for index in indices]
        self._set_combo_items_silent(
            self.binning_combo, names, self.binning_names[self.dataset_index]
        )
        self.binning_combo.setEnabled(len(indices) > 1)

    def _set_dataset_selection(self, index: int) -> None:
        options = self._source_dataset_options()
        if not (0 <= int(index) < len(options)):
            return
        source = options[int(index)]
        candidates = [
            candidate
            for candidate, value in enumerate(self.source_dataset_names)
            if value == source
        ]
        if candidates:
            self._set_dataset_index(candidates[0])

    def _set_binning_selection(self, index: int) -> None:
        source = self.source_dataset_names[self.dataset_index]
        candidates = [
            candidate
            for candidate, value in enumerate(self.source_dataset_names)
            if value == source
        ]
        if 0 <= int(index) < len(candidates):
            self._set_dataset_index(candidates[int(index)])

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
        dataset_name = self.source_dataset_names[self.dataset_index]
        if view_mode == "volumetric" and self.volume_panel is not None:
            dataset_name = self.volume_panel.dataset_combo.currentText()
        return {
            "view_mode": view_mode,
            "dataset_name": dataset_name,
            "binning_name": self.binning_names[self.dataset_index],
            "x_dim": self.data.axes[self.model.x_dim].name,
            "y_dim": self.data.axes[self.model.y_dim].name,
            "channel": self.model.channel,
            "selections": self._export_selections(),
            "integrate_checks": self._export_integrate_checks(),
            "cmap": self.model._effective_cmap(),
            "color_scale": self.model.color_scale,
            "auto_limits": self.model.auto_limits,
            "sigma_n": self.model.sigma_n,
            "iqr_n": self.model.iqr_n,
            "percentile_n": self.model.percentile_n,
            "power_gamma": self.model.power_gamma,
            "color_alpha": self.model.color_alpha,
            "autoscale": self.model.autoscale,
            "manual_vmin": self.model.manual_vmin,
            "manual_vmax": self.model.manual_vmax,
            "smoothing_x": self.smoothing_x,
            "smoothing_y": self.smoothing_y,
            "smoothing_fill_nans": self.smoothing_fill_nans,
            "show_brillouin_zone_boundaries": self.show_brillouin_zone_boundaries,
            "show_major_gridlines": self.show_major_gridlines,
            "brillouin_zone_spacegroup": self._effective_brillouin_zone_context().get("spacegroup"),
            "brillouin_zone_lattice_parameters": self._effective_brillouin_zone_context().get("lattice_parameters"),
            "brillouin_zone_color": self.brillouin_zone_color,
            "brillouin_zone_linewidth": self.brillouin_zone_linewidth,
            "brillouin_zone_alpha": self.brillouin_zone_alpha,
            "xlim": xlim,
            "ylim": ylim,
            "x_step": self._current_display_step("x"),
            "y_step": self._current_display_step("y"),
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
            "tile_label_decimals": self.tile_label_decimals,
            "tile_label_prefix": self.tile_label_prefix,
            "tile_label_unit": self.tile_label_unit,
            "tile_label_si_prefix": self.tile_label_si_prefix,
            "show_tile_labels": self.show_tile_labels,
            "tile_local_color_scales": self.tile_local_color_scales,
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
        binning_name = settings.get("binning_name")
        selected = next(
            (
                index
                for index, source in enumerate(self.source_dataset_names)
                if source == dataset_name
                and (binning_name is None or self.binning_names[index] == binning_name)
            ),
            None,
        )
        if selected is not None:
            self._set_dataset_index(selected)
        x_name = settings.get("x_dim")
        y_name = settings.get("y_dim")
        names = [axis.name for axis in self.data.axes]
        if x_name in names:
            self._set_display_dim("x", names.index(x_name))
        if y_name in names and len(names) > 1:
            self._set_display_dim("y", names.index(y_name))
        if settings.get("x_step") is not None:
            self._set_display_step("x", float(settings["x_step"]), redraw=False)
        if settings.get("y_step") is not None:
            self._set_display_step("y", float(settings["y_step"]), redraw=False)
        if settings.get("channel") in self.model.CHANNELS:
            self._set_channel(str(settings["channel"]))
        self.model.selections.update({int(key): tuple(value) for key, value in dict(settings.get("selections", {})).items()})
        self.model.integrate_checks.update({int(key): bool(value) for key, value in dict(settings.get("integrate_checks", {})).items()})
        effective_cmap = str(settings.get("cmap", self.model.cmap))
        self._set_cmap(effective_cmap.removesuffix("_r"))
        self.model.cmap_reversed = effective_cmap.endswith("_r")
        self._set_color_scale(str(settings.get("color_scale", self.model.color_scale)))
        self._set_auto_limits(str(settings.get("auto_limits", self.model.auto_limits)))
        for name in (
            "sigma_n",
            "iqr_n",
            "percentile_n",
            "power_gamma",
            "color_alpha",
        ):
            if name in settings:
                setattr(self.model, name, float(settings[name]))
        self.model.manual_vmin = settings.get("manual_vmin", self.model.manual_vmin)
        self.model.manual_vmax = settings.get("manual_vmax", self.model.manual_vmax)
        self.model.autoscale = bool(settings.get("autoscale", self.model.autoscale))
        self.smoothing_x = float(settings.get("smoothing_x", self.smoothing_x))
        self.smoothing_y = float(settings.get("smoothing_y", self.smoothing_y))
        self.smoothing_fill_nans = bool(
            settings.get("smoothing_fill_nans", False)
        )
        self.show_brillouin_zone_boundaries = bool(
            settings.get("show_brillouin_zone_boundaries", False)
        )
        self.show_major_gridlines = bool(
            settings.get("show_major_gridlines", False)
        )
        if self.show_brillouin_zone_boundaries and self.show_major_gridlines:
            self.show_major_gridlines = False
        context = self._effective_brillouin_zone_context()
        if settings.get("brillouin_zone_spacegroup"):
            context["spacegroup"] = str(settings["brillouin_zone_spacegroup"])
        if settings.get("brillouin_zone_lattice_parameters"):
            context["lattice_parameters"] = dict(
                settings["brillouin_zone_lattice_parameters"]
            )
        self.crystal_contexts[self.dataset_index] = context
        self.brillouin_zone_color = str(
            settings.get("brillouin_zone_color", self.brillouin_zone_color)
        )
        self.brillouin_zone_linewidth = float(
            settings.get("brillouin_zone_linewidth", self.brillouin_zone_linewidth)
        )
        self.brillouin_zone_alpha = float(
            settings.get("brillouin_zone_alpha", self.brillouin_zone_alpha)
        )
        self._set_spin_silent(self.smoothing_x_spin, self.smoothing_x)
        self._set_spin_silent(self.smoothing_y_spin, self.smoothing_y)
        self._set_checkbox_silent(
            self.smoothing_fill_nans_check, self.smoothing_fill_nans
        )
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
        self.tile_label_decimals = max(0, min(10, int(settings.get("tile_label_decimals", self.tile_label_decimals))))
        self._set_spin_silent(self.tile_label_decimals_spin, self.tile_label_decimals)
        self.tile_label_prefix = str(settings.get("tile_label_prefix", self.tile_label_prefix))
        self.tile_label_unit = str(settings.get("tile_label_unit", self.tile_label_unit))
        self.tile_label_si_prefix = str(settings.get("tile_label_si_prefix", self.tile_label_si_prefix))
        self._sync_tile_label_controls()
        self.show_tile_labels = bool(
            settings.get("show_tile_labels", self.show_tile_labels)
        )
        self.tile_local_color_scales = bool(
            settings.get(
                "tile_local_color_scales", self.tile_local_color_scales
            )
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
        self._sync_tiled_color_controls()
        self._set_checkbox_silent(
            self.show_tile_labels_check, self.show_tile_labels
        )
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
        self._sync_color_controls()
        self._set_checkbox_silent(
            self.show_brillouin_zone_check,
            self.show_brillouin_zone_boundaries,
        )
        self._set_checkbox_silent(
            self.show_major_gridlines_check,
            self.show_major_gridlines,
        )
        bz_color_index = self.brillouin_zone_color_combo.findData(
            self.brillouin_zone_color
        )
        if bz_color_index >= 0:
            self.brillouin_zone_color_combo.setCurrentIndex(bz_color_index)
        self._set_spin_silent(
            self.brillouin_zone_linewidth_spin, self.brillouin_zone_linewidth
        )
        self._set_spin_silent(
            self.brillouin_zone_alpha_spin, self.brillouin_zone_alpha
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
                crystal_contexts=self.crystal_contexts,
            )
            self._child_viewers.append(viewer)
        if viewer is None:
            return None
        viewer.apply_plot_settings(settings)
        viewer.show()
        return viewer

    def open_kpath_viewer(self):
        """Open the current 4D HKL dataset in the dedicated path viewer."""

        from PySide6 import QtWidgets

        from .qt_brillouin_zone import prompt_brillouin_zone_context
        from .qt_kpath_viewer import QtKPathViewer

        if not isinstance(self.data, MDHistoData):
            return None
        context = self._effective_brillouin_zone_context()
        if not context.get("spacegroup") or not isinstance(
            context.get("lattice_parameters"), dict
        ):
            updated = prompt_brillouin_zone_context(
                self.window, context, require_lattice=True
            )
            if updated is None:
                return None
            context = updated
            self.crystal_contexts[self.dataset_index] = updated
            if self._brillouin_zone_context_callback is not None:
                self._brillouin_zone_context_callback(
                    self.source_dataset_names[self.dataset_index], dict(updated)
                )
        try:
            viewer = QtKPathViewer(
                self.data,
                lattice_parameters=dict(context["lattice_parameters"]),
                spacegroup=str(context["spacegroup"]),
                parent=self.window,
            )
        except (KeyError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self.window, "K-path viewer", f"Could not open the K-path viewer:\n{exc}"
            )
            return None
        self._child_viewers.append(viewer)
        return viewer.show()

    def set_open_new_viewer_callback(self, callback) -> None:
        """Set the project-aware factory used by :meth:`open_new_viewer`."""

        self._open_new_viewer_callback = callback

    def set_save_plot_callback(self, callback, *, new_plot_callback=None) -> None:
        """Expose project-bound saved-plot creation when a callback is supplied."""

        self._save_plot_callback = callback
        self._save_new_plot_callback = new_plot_callback
        if self.save_plot_button is not None:
            self.save_plot_button.setEnabled(callback is not None)
        if self.save_new_plot_button is not None:
            self.save_new_plot_button.setEnabled(new_plot_callback is not None)

    def set_saved_plot_editing(self) -> None:
        """Expose update/copy actions only for a viewer opened from a recipe."""
        self.save_plot_button.setText("Save plot")
        self.save_plot_button.setToolTip("Update the stored plot currently being edited with this view.")
        self.save_new_plot_button.show()

    def save_new_plot(self):
        """Save a separate recipe and continue editing that new plot."""
        return self._store_plot_with_callback(self._save_new_plot_callback)

    def store_plot(self):
        """Store the current view through the owning project and report it."""

        return self._store_plot_with_callback(self._save_plot_callback)

    def _store_plot_with_callback(self, callback):
        if callback is None:
            return None
        plot = callback()
        if self.store_plot_status_label is not None:
            if plot is None:
                self.store_plot_status_label.setText("Plot was not stored")
            else:
                name = str(getattr(plot, "name", "plot"))
                self.store_plot_status_label.setText(f"Stored as {name}")
        return plot

    def set_save_project_callback(self, callback) -> None:
        """Route the standard Save shortcut to the owning project window."""

        self._save_project_callback = callback
        if self.save_project_shortcut is not None:
            self.save_project_shortcut.setEnabled(callback is not None)

    def set_unmask_model_callback(self, callback) -> None:
        """Set the project callback that rebuilds full-grid model channels."""

        self._unmask_model_callback = callback

    def set_brillouin_zone_context_callback(self, callback) -> None:
        """Persist crystal information entered from the viewer when available."""

        self._brillouin_zone_context_callback = callback

    def set_close_callback(self, callback) -> None:
        """Notify the owning project when this viewer window closes."""

        self._close_callback = callback

    def save_script(self) -> None:
        from pathlib import Path


        path, _selected_filter = get_save_file_name(
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
                f"    color_alpha={self.model.color_alpha!r},",
                f"    smoothing_sigma_x={self.smoothing_x!r},",
                f"    smoothing_sigma_y={self.smoothing_y!r},",
                f"    smoothing_fill_nans={self.smoothing_fill_nans!r},",
                f"    x_step={self._current_display_step('x')!r},",
                f"    y_step={self._current_display_step('y')!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    axes_linewidth={self.axis_linewidth!r},",
                f"    show_histogram_axes={self.hist_axes_check.isChecked()!r},",
                f"    roi_extents={self._roi_extents!r},",
                f"    xcut_percent={self.xcut_percent!r},",
                f"    ycut_percent={self.ycut_percent!r},",
                f"    show_brillouin_zone_boundaries={self.show_brillouin_zone_boundaries!r},",
                f"    show_major_gridlines={self.show_major_gridlines!r},",
                f"    brillouin_zone_spacegroup={self._effective_brillouin_zone_context().get('spacegroup')!r},",
                f"    brillouin_zone_lattice_parameters={self._effective_brillouin_zone_context().get('lattice_parameters')!r},",
                f"    brillouin_zone_color={self.brillouin_zone_color!r},",
                f"    brillouin_zone_linewidth={self.brillouin_zone_linewidth!r},",
                f"    brillouin_zone_alpha={self.brillouin_zone_alpha!r},",
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
                f"    tile_step={self._effective_tile_step()!r},",
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
                f"    color_alpha={self.model.color_alpha!r},",
                f"    smoothing_sigma_x={self.smoothing_x!r},",
                f"    smoothing_sigma_y={self.smoothing_y!r},",
                f"    smoothing_fill_nans={self.smoothing_fill_nans!r},",
                f"    x_step={self._current_display_step('x')!r},",
                f"    y_step={self._current_display_step('y')!r},",
                f"    xlim={self._export_limits('x')!r},",
                f"    ylim={self._export_limits('y')!r},",
                f"    font_size={self.font_size!r},",
                f"    axes_linewidth={self.axis_linewidth!r},",
                f"    tile_label_decimals={self.tile_label_decimals!r},",
                f"    tile_label_prefix={self.tile_label_prefix!r},",
                f"    tile_label_unit={self.tile_label_unit!r},",
                f"    tile_label_si_prefix={self.tile_label_si_prefix!r},",
                f"    show_tile_labels={self.show_tile_labels!r},",
                f"    local_color_scales={self.tile_local_color_scales!r},",
                f"    show_brillouin_zone_boundaries={self.show_brillouin_zone_boundaries!r},",
                f"    show_major_gridlines={self.show_major_gridlines!r},",
                f"    brillouin_zone_spacegroup={self._effective_brillouin_zone_context().get('spacegroup')!r},",
                f"    brillouin_zone_lattice_parameters={self._effective_brillouin_zone_context().get('lattice_parameters')!r},",
                f"    brillouin_zone_color={self.brillouin_zone_color!r},",
                f"    brillouin_zone_linewidth={self.brillouin_zone_linewidth!r},",
                f"    brillouin_zone_alpha={self.brillouin_zone_alpha!r},",
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
                f"    smoothing_fill_nans={self.smoothing_fill_nans!r},",
                f"    x_step={self._current_display_step('x')!r},",
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
                f"    axis_step={self._current_display_step('x')!r},",
                f"    smoothing_sigma={self.smoothing_x!r},",
                f"    smoothing_fill_nans={self.smoothing_fill_nans!r},",
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
        from .qt_slice_builder import build_slice_viewer

        build_slice_viewer(
            self,
            marker_options=_MARKER_OPTIONS,
            line_style_options=_LINE_STYLE_OPTIONS,
            color_options=_COLOR_OPTIONS,
        )

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
        from .qt_volume_viewer import (
            QtVolumeViewerPanel,
            confirm_large_volume_view,
            supports_volume_view,
        )

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
            selected_dataset, selected_name, _original_index = supported[selected]
            if not confirm_large_volume_view(
                self.window,
                selected_dataset,
                selected_name,
            ):
                blocked = self.view_mode_combo.blockSignals(True)
                self.view_mode_combo.setCurrentIndex(self._active_plot_view_mode)
                self.view_mode_combo.blockSignals(blocked)
                self.content_stack.setCurrentIndex(0)
                return
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
        self._sync_dataset_binning_combos()

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
            smoothing_fill_nans=bool(self.smoothing_fill_nans),
            show_brillouin_zone_boundaries=bool(self.show_brillouin_zone_boundaries),
            show_major_gridlines=bool(self.show_major_gridlines),
            brillouin_zone_color=str(self.brillouin_zone_color),
            brillouin_zone_linewidth=float(self.brillouin_zone_linewidth),
            brillouin_zone_alpha=float(self.brillouin_zone_alpha),
            tile_dim=self.tile_dim,
            tile_range=self.tile_range,
            tile_step=float(self.tile_step),
            tile_step_auto=bool(self.tile_step_auto),
            tile_label_decimals=self.tile_label_decimals,
            show_tile_labels=bool(self.show_tile_labels),
            tile_label_prefix=self.tile_label_prefix,
            tile_label_unit=self.tile_label_unit,
            tile_label_si_prefix=self.tile_label_si_prefix,
            tile_local_color_scales=bool(self.tile_local_color_scales),
            display_step_factors=dict(self.display_step_factors),
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
            color_alpha=self._initial_color_alpha,
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
            self.smoothing_fill_nans = bool(state.smoothing_fill_nans)
            self.show_brillouin_zone_boundaries = bool(
                state.show_brillouin_zone_boundaries
            )
            self.show_major_gridlines = bool(state.show_major_gridlines)
            self.brillouin_zone_color = str(state.brillouin_zone_color)
            self.brillouin_zone_linewidth = float(state.brillouin_zone_linewidth)
            self.brillouin_zone_alpha = float(state.brillouin_zone_alpha)
            self.tile_dim = state.tile_dim
            self.tile_range = tuple(state.tile_range)
            self.tile_step = float(state.tile_step)
            self.tile_step_auto = bool(state.tile_step_auto)
            self.tile_label_decimals = state.tile_label_decimals
            self.tile_label_prefix = state.tile_label_prefix
            self.tile_label_unit = state.tile_label_unit
            self.tile_label_si_prefix = state.tile_label_si_prefix
            self._sync_tile_label_controls()
            self._set_spin_silent(self.tile_label_decimals_spin, self.tile_label_decimals)
            self.show_tile_labels = bool(state.show_tile_labels)
            self.tile_local_color_scales = bool(
                state.tile_local_color_scales and state.model.autoscale
            )
            self.display_step_factors = dict(state.display_step_factors or {})
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
            self._set_spin_silent(self.alpha_spin, self.model.color_alpha)
            self._set_spin_silent(self.limit_n_spin, self._current_limit_n())
            self._set_spin_silent(self.font_size_spin, self.font_size)
            self._set_spin_silent(self.line_width_spin, self.axis_linewidth)
            self._set_checkbox_silent(
                self.show_binning_title_check, self.show_binning_title
            )
            self._set_checkbox_silent(
                self.show_tile_labels_check, self.show_tile_labels
            )
            self._sync_tiled_color_controls()
            self._set_spin_silent(self.smoothing_x_spin, self.smoothing_x)
            self._set_spin_silent(self.smoothing_y_spin, self.smoothing_y)
            self._set_checkbox_silent(
                self.smoothing_fill_nans_check, self.smoothing_fill_nans
            )
            self._set_checkbox_silent(
                self.show_brillouin_zone_check,
                self.show_brillouin_zone_boundaries,
            )
            self._set_checkbox_silent(
                self.show_major_gridlines_check,
                self.show_major_gridlines,
            )
            color_index = self.brillouin_zone_color_combo.findData(
                self.brillouin_zone_color
            )
            if color_index >= 0:
                self.brillouin_zone_color_combo.setCurrentIndex(color_index)
            self._set_spin_silent(
                self.brillouin_zone_linewidth_spin,
                self.brillouin_zone_linewidth,
            )
            self._set_spin_silent(
                self.brillouin_zone_alpha_spin, self.brillouin_zone_alpha
            )
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
            self._sync_display_step_controls()
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
        axis_values = np.asarray(self.data.axes[self.tile_dim].values, dtype=float)
        minimum = float(np.min(axis_values))
        maximum = float(np.max(axis_values))
        if reset_range:
            self.tile_range = (float(np.min(centers)), float(np.max(centers)))
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

    def _effective_tile_step(self) -> float | None:
        if self.tile_step_auto and self.tile_dim is not None and "metadata_dimension" in self.data.axes[self.tile_dim].metadata:
            return None
        return self.tile_step

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

    def _set_tile_label_decimals(self, value: int) -> None:
        self.tile_label_decimals = int(value)
        if not self._restoring_dataset_state and self._tiled_mode_active():
            self.update_plot()

    def _sync_tile_label_controls(self) -> None:
        self.tile_label_prefix_edit.setText(self.tile_label_prefix)
        self.tile_label_unit_edit.setText(self.tile_label_unit)
        self._set_combo_silent(self.tile_label_si_prefix_combo, self.tile_label_si_prefix)

    def _set_tile_label_text(self, *_args) -> None:
        self.tile_label_prefix = self.tile_label_prefix_edit.text()
        self.tile_label_unit = self.tile_label_unit_edit.text()
        self.tile_label_si_prefix = self.tile_label_si_prefix_combo.currentText()
        if not self._restoring_dataset_state and self._tiled_mode_active():
            self.update_plot()

    def _set_show_tile_labels(self, checked: bool) -> None:
        self.show_tile_labels = bool(checked)
        if not self._restoring_dataset_state and self._tiled_mode_active():
            self.update_plot()

    def _set_tile_local_color_scales(self, checked: bool) -> None:
        self.tile_local_color_scales = bool(checked and self.model.autoscale)
        self._sync_tiled_color_controls()
        if not self._restoring_dataset_state and self._tiled_mode_active():
            self.update_plot(preserve_view=False)

    def _sync_tiled_color_controls(self) -> None:
        if self.tile_local_color_scales_check is None:
            return
        if not self.model.autoscale:
            self.tile_local_color_scales = False
        self._set_checkbox_silent(
            self.tile_local_color_scales_check,
            self.tile_local_color_scales,
        )
        self.tile_local_color_scales_check.setEnabled(self.model.autoscale)

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
                color_alpha=self._initial_color_alpha,
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

    def _sync_color_controls(self) -> None:
        """Reflect restored model settings without firing user-edit callbacks."""
        self._set_combo_silent(self.cmap_combo, self.model.cmap)
        self._set_combo_silent(self.scale_combo, self.model.color_scale)
        self._set_combo_silent(self.limits_combo, self.model.auto_limits)
        self._set_checkbox_silent(self.autoscale_check, self.model.autoscale)
        self._set_spin_silent(self.gamma_spin, self.model.power_gamma)
        self._set_spin_silent(self.alpha_spin, self.model.color_alpha)
        self._set_spin_silent(self.limit_n_spin, self._current_limit_n())
        self.gamma_label.setVisible(self.model.color_scale == "power")
        self.gamma_spin.setVisible(self.model.color_scale == "power")
        self._sync_limit_n_visibility()
        self._sync_tiled_color_controls()

    def _set_plot_smoothing(self, axis_name: str, value: float) -> None:
        if self._restoring_dataset_state:
            return
        if axis_name == "x":
            self.smoothing_x = max(float(value), 0.0)
        else:
            self.smoothing_y = max(float(value), 0.0)
        self.update_plot()

    def _set_show_brillouin_zone_boundaries(self, checked: bool) -> None:
        if self._restoring_dataset_state:
            return
        if checked and not self._ensure_brillouin_zone_context():
            self._set_checkbox_silent(self.show_brillouin_zone_check, False)
            return
        if checked:
            self.show_major_gridlines = False
            self._set_checkbox_silent(self.show_major_gridlines_check, False)
        self.show_brillouin_zone_boundaries = bool(checked)
        self.update_plot()

    def _set_show_major_gridlines(self, checked: bool) -> None:
        if self._restoring_dataset_state:
            return
        if checked:
            self.show_brillouin_zone_boundaries = False
            self._set_checkbox_silent(self.show_brillouin_zone_check, False)
        self.show_major_gridlines = bool(checked)
        self.update_plot()

    def _set_brillouin_zone_color(self, _index: int) -> None:
        if self._restoring_dataset_state or self.brillouin_zone_color_combo is None:
            return
        self.brillouin_zone_color = str(
            self.brillouin_zone_color_combo.currentData()
        )
        self.update_plot()

    def _set_brillouin_zone_linewidth(self, value: float) -> None:
        if self._restoring_dataset_state:
            return
        self.brillouin_zone_linewidth = float(value)
        self.update_plot()

    def _set_brillouin_zone_alpha(self, value: float) -> None:
        if self._restoring_dataset_state:
            return
        self.brillouin_zone_alpha = float(value)
        self.update_plot()

    def _effective_brillouin_zone_context(self) -> dict[str, Any]:
        context = dict(self.crystal_contexts[self.dataset_index])
        metadata = self.data.metadata if isinstance(self.data.metadata, dict) else {}
        for key in ("spacegroup", "lattice_parameters"):
            if not context.get(key) and metadata.get(key):
                context[key] = metadata[key]
        crystal = metadata.get("crystal")
        if not context.get("spacegroup") and isinstance(crystal, dict):
            context["spacegroup"] = crystal.get("spacegroup")
        return context

    def _ensure_brillouin_zone_context(self) -> bool:
        from .analysis.coordinates import rlu_to_q_matrix
        from .qt_brillouin_zone import prompt_brillouin_zone_context

        context = self._effective_brillouin_zone_context()
        metadata = dict(self.data.metadata)
        if context.get("lattice_parameters"):
            metadata["lattice_parameters"] = context["lattice_parameters"]
        try:
            rlu_to_q_matrix(metadata)
            require_lattice = False
        except (KeyError, TypeError, ValueError):
            require_lattice = True
        if context.get("spacegroup") and not require_lattice:
            return True
        updated = prompt_brillouin_zone_context(
            self.window, context, require_lattice=require_lattice
        )
        if updated is None:
            return False
        self.crystal_contexts[self.dataset_index] = updated
        if self._brillouin_zone_context_callback is not None:
            self._brillouin_zone_context_callback(
                self.source_dataset_names[self.dataset_index], dict(updated)
            )
        return True

    def _draw_gridline_overlay(
        self, ax: Any | None = None, *, coordinate_overrides: dict[int, float] | None = None
    ) -> None:
        if getattr(self.model, "is_point_list", False):
            return
        target = self.ax_image if ax is None else ax
        if target is None:
            return
        if self.show_major_gridlines:
            target.grid(
                True,
                which="major",
                axis="both",
                color=self.brillouin_zone_color,
                linewidth=self.brillouin_zone_linewidth,
                alpha=self.brillouin_zone_alpha,
            )
            return
        if not self.show_brillouin_zone_boundaries:
            return
        context = self._effective_brillouin_zone_context()
        try:
            draw_mdhisto_brillouin_zones(
                target,
                self.data,
                x_dim=self.model.x_dim,
                y_dim=self.model.y_dim,
                selections=self.model.selections,
                coordinate_overrides=coordinate_overrides,
                spacegroup=context.get("spacegroup"),
                lattice_parameters=context.get("lattice_parameters"),
                color=self.brillouin_zone_color,
                linewidth=self.brillouin_zone_linewidth,
                alpha=self.brillouin_zone_alpha,
            )
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            # Axis changes can temporarily make an enabled HKL overlay inapplicable.
            return

    def _set_smoothing_fill_nans(self, checked: bool) -> None:
        if self._restoring_dataset_state:
            return
        self.smoothing_fill_nans = bool(checked)
        self.update_plot()

    def _smoothed_slice_view(self, view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if getattr(self.model, "is_point_list", False):
            return view
        return smooth_mdhisto_view(
            coarsen_mdhisto_view(
                view,
                x_step=self._current_display_step("x"),
                y_step=self._current_display_step("y"),
            ),
            sigma_x=self.smoothing_x,
            sigma_y=self.smoothing_y,
            fill_nans=self.smoothing_fill_nans,
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
        if not autoscale:
            self.model.manual_vmin = float(self.vmin_spin.value())
            self.model.manual_vmax = float(self.vmax_spin.value())
        self.model.autoscale = bool(autoscale)
        self._sync_tiled_color_controls()
        self.update_plot()

    def _set_manual_limit(self, which: str, value: float) -> None:
        if self._syncing_limits:
            return
        if self.model.autoscale:
            self.model.manual_vmin = float(self.vmin_spin.value())
            self.model.manual_vmax = float(self.vmax_spin.value())
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

    def _set_color_alpha(self, alpha: float) -> None:
        self.model.color_alpha = float(np.clip(alpha, -20.0, 20.0))
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
        displayed_dims = {self.model.x_dim, self.model.y_dim}
        if self._tiled_mode_active() and self.tile_dim is not None:
            displayed_dims.add(self.tile_dim)
        for dim, axis in enumerate(self.data.axes):
            if dim in displayed_dims:
                continue
            centers = np.asarray(axis.centers, dtype=float)
            if centers.size == 0:
                continue
            selection = self.model.selections.get(
                dim, (float(centers[centers.size // 2]),) * 2
            )
            low, high = sorted((float(selection[0]), float(selection[1])))
            if not self.model.integrate_checks.get(dim, self.model.integrate):
                selected_bin = self.model._normalized_selections()[dim]
                edges = self.model._axis_edges(dim)
                low, high = float(edges[selected_bin]), float(edges[selected_bin + 1])
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
        for colorbar in self._tile_colorbars:
            if colorbar is self.colorbar:
                continue
            colorbar.ax.yaxis.label.set_fontsize(size)
            colorbar.ax.tick_params(labelsize=size)
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
        for colorbar in self._tile_colorbars:
            if colorbar is self.colorbar:
                continue
            colorbar.outline.set_linewidth(width)
            colorbar.ax.tick_params(which="both", direction="in", width=width)
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
        self._sync_tiled_color_controls()

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

    def _on_navigation_history_restored(self) -> None:
        """Reapply nfit state after Matplotlib restores every axes in the figure."""

        self._sync_view_limit_controls()
        if self.image is not None:
            # Matplotlib navigation history includes colorbar axes. Restoring one
            # can mutate the mappable normalization after nfit has restored the
            # data view, leaving manual limits and their controls inconsistent.
            self.update_plot(preserve_view=True)
        elif self.canvas is not None:
            self.canvas.draw_idle()

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

    def _native_display_step(self, dim: int) -> float:
        if getattr(self.model, "is_point_list", False):
            values = np.asarray(self.model.data.column(self.model.x_key), dtype=float)
            values = np.unique(values[np.isfinite(values)])
            view = {"x_centers": values, "x_edges": np.array([], dtype=float)}
            return mdhisto_view_native_step(view, "x")
        axis = self.data.axes[int(dim)]
        view = {
            "x_centers": np.asarray(axis.centers, dtype=float),
            "x_edges": np.asarray(axis.values, dtype=float),
        }
        return mdhisto_view_native_step(view, "x")

    def _current_display_step(self, axis_name: str) -> float:
        dim = self.model.x_dim if axis_name == "x" else self.model.y_dim
        native = self._native_display_step(dim)
        value = native * max(int(self.display_step_factors.get(int(dim), 1)), 1)
        return float(f"{value:.12g}")

    def _set_display_step(
        self,
        axis_name: str,
        value: float,
        *,
        redraw: bool = True,
    ) -> None:
        if self._syncing_display_steps or getattr(self.model, "is_point_list", False):
            return
        dim = self.model.x_dim if axis_name == "x" else self.model.y_dim
        native = self._native_display_step(dim)
        size = int(self.data.shape[dim])
        factor = int(np.clip(np.floor(float(value) / native + 0.5), 1, max(size, 1)))
        self.display_step_factors[int(dim)] = factor
        self._sync_display_step_controls()
        if redraw and not self._restoring_dataset_state:
            self.update_plot()

    def _sync_display_step_controls(self) -> None:
        if self.x_step_spin is None or self.y_step_spin is None:
            return
        self._syncing_display_steps = True
        try:
            for axis_name, spinbox in (
                ("x", self.x_step_spin),
                ("y", self.y_step_spin),
            ):
                dim = self.model.x_dim if axis_name == "x" else self.model.y_dim
                native = self._native_display_step(dim)
                size = (
                    len(np.unique(np.asarray(self.model.data.column(self.model.x_key))))
                    if getattr(self.model, "is_point_list", False)
                    else int(self.data.shape[dim])
                )
                previous = spinbox.blockSignals(True)
                try:
                    spinbox.setDecimals(8)
                    spinbox.setRange(native, native * max(size, 1))
                    spinbox.setSingleStep(native)
                    spinbox.setValue(self._current_display_step(axis_name))
                finally:
                    spinbox.blockSignals(previous)
        finally:
            self._syncing_display_steps = False

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
        if self.x_step_spin is not None:
            self.x_step_spin.setVisible(not is_point)
        if self.y_step_spin is not None:
            self.y_step_spin.setVisible(not is_point and not is_line and not is_waterfall)
        if self.step_header_label is not None:
            self.step_header_label.setVisible(not is_point)
        self._sync_display_step_controls()
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
        if self.brillouin_zone_group is not None:
            self.brillouin_zone_group.setVisible(
                not is_point and not is_line and not is_waterfall
            )
        if self.open_kpath_viewer_button is not None:
            axes = getattr(self.data, "axes", ())
            self.open_kpath_viewer_button.setVisible(
                isinstance(self.data, MDHistoData)
                and len(axes) == 4
                and sum(axis.kind == "momentum" for axis in axes) == 3
                and sum(
                    axis.kind == "energy" or axis.role == "energy_transfer"
                    for axis in axes
                ) == 1
            )
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
        if self.show_tile_labels_check is not None:
            self.show_tile_labels_check.setVisible(is_tiled)
            self.tile_label_decimals_spin.setVisible(is_tiled)
            self.tile_label_options.setVisible(is_tiled)
        if self.tile_local_color_scales_check is not None:
            self.tile_local_color_scales_check.setVisible(is_tiled)
            self._sync_tiled_color_controls()
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
        self._mode_controllers.layouts._ensure_standard_plot_layout()

    def _ensure_waterfall_layout(self) -> None:
        self._mode_controllers.layouts._ensure_waterfall_layout()

    def _ensure_tiled_layout(
        self,
        panel_count: int,
        *,
        local_color_scales: bool,
    ) -> None:
        self._mode_controllers.layouts._ensure_tiled_layout(
            panel_count,
            local_color_scales=local_color_scales,
        )

    def _ensure_fit_compare_layout(
        self,
        panel_count: int,
        *,
        with_cuts: bool = False,
        with_residual_cut: bool = False,
    ) -> None:
        self._mode_controllers.layouts._ensure_fit_compare_layout(
            panel_count,
            with_cuts=with_cuts,
            with_residual_cut=with_residual_cut,
        )

    def _ensure_residual_1d_layout(self) -> None:
        self._mode_controllers.layouts._ensure_residual_1d_layout()

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
        self.roi_sum_text = None
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
            self._set_roi_extents(
                self._default_roi_extents(),
                update_cuts=bool(
                    self.hist_axes_check and self.hist_axes_check.isChecked()
                ),
                draw=False,
            )
        else:
            self._set_roi_extents(
                self._roi_extents,
                update_cuts=bool(
                    self.hist_axes_check and self.hist_axes_check.isChecked()
                ),
                draw=False,
            )
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
        self._mode_controllers.tiled._draw_tiled_view(
            previous_xlim,
            previous_ylim,
            previous_dims,
            current_dims,
        )

    def _draw_waterfall_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        self._mode_controllers.waterfall._draw_waterfall_view(
            previous_xlim,
            previous_ylim,
            previous_dims,
            current_dims,
        )

    def _draw_fit_panels_view(
        self,
        previous_xlim,
        previous_ylim,
        previous_dims,
        current_dims,
    ) -> None:
        self._mode_controllers.fit_comparison._draw_fit_panels_view(
            previous_xlim,
            previous_ylim,
            previous_dims,
            current_dims,
        )

    def _update_fit_compare_cuts(
        self,
        extents: tuple[float, float, float, float] | None,
    ) -> None:
        self._mode_controllers.fit_comparison._update_fit_compare_cuts(extents)

    def _comparison_panel_model(
        self,
        data: MDHistoData,
        channel: str,
        *,
        masked: bool | None = None,
    ) -> MDHistoSliceViewer:
        return self._mode_controllers.fit_comparison._comparison_panel_model(
            data,
            channel,
            masked=masked,
        )

    def _is_effective_1d(self) -> bool:
        if getattr(self.model, "is_point_list", False):
            return True
        return sum(size > 1 for size in self.data.shape) == 1

    def _waterfall_mode_active(self) -> bool:
        return self.view_mode_combo is not None and self.view_mode_combo.currentIndex() == 1

    def _tiled_mode_active(self) -> bool:
        return self.view_mode_combo is not None and self.view_mode_combo.currentIndex() == 2

    def _waterfall_uses_1d_group(self) -> bool:
        return self._mode_controllers.waterfall._waterfall_uses_1d_group()

    def _waterfall_1d_source_indices(self) -> list[int]:
        return self._mode_controllers.waterfall._waterfall_1d_source_indices()

    def waterfall_source_dataset_names(self) -> list[str]:
        """Return dataset names used by the active waterfall recipe."""

        return self._mode_controllers.waterfall.waterfall_source_dataset_names()

    def _slice_1d_channel(
        self,
        view: dict[str, np.ndarray],
        name: str,
    ) -> np.ndarray | None:
        return self._mode_controllers.standard._slice_1d_channel(view, name)

    def _draw_1d_view(
        self,
        view: dict[str, np.ndarray],
        values: np.ndarray,
    ) -> None:
        self._mode_controllers.standard._draw_1d_view(view, values)

    def _draw_1d_with_residual(
        self,
        previous_xlim,
        previous_dims,
        current_dims,
    ) -> None:
        self._mode_controllers.standard._draw_1d_with_residual(
            previous_xlim,
            previous_dims,
            current_dims,
        )

    def _draw_2d_view(
        self,
        view: dict[str, np.ndarray],
        values: np.ndarray,
    ) -> None:
        self._mode_controllers.standard._draw_2d_view(view, values)

    def _draw_bragg_peak_overlay(self) -> None:
        self._mode_controllers.standard._draw_bragg_peak_overlay()

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
            self._clear_roi_sum_annotation()
        if self.rectangle_selector is not None:
            self.rectangle_selector.set_visible(bool(visible))
            self.rectangle_selector.set_active(bool(visible) and self.roi_button.isChecked())
            self._set_rectangle_selector_style(active=bool(visible) and self.roi_button.isChecked())
        if visible and self._roi_extents is not None:
            self._update_histogram_cuts_from_extents(self._roi_extents)
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
        if not visible:
            self._clear_roi_sum_annotation()
        elif self._roi_extents is not None:
            self._update_histogram_cuts_from_extents(self._roi_extents)
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
        self._clear_roi_sum_annotation()
        if np.any(x_mask) and np.any(y_mask):
            z = self.model._display_values(view)
            errors = np.asarray(view["errors"], dtype=float)
            selected = np.ix_(y_mask, x_mask)
            self._show_roi_sum_annotation(
                z[selected],
                errors[selected],
                extents,
            )
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

    def _clear_roi_sum_annotation(self) -> None:
        annotation = self.roi_sum_text
        self.roi_sum_text = None
        if annotation is not None:
            annotation.set_text("")

    def _show_roi_sum_annotation(
        self,
        values: np.ndarray,
        errors: np.ndarray,
        extents: tuple[float, float, float, float],
    ) -> None:
        if (
            self.ax_image is None
            or self.show_box_check is None
            or not self.show_box_check.isChecked()
            or self.hist_axes_check is None
            or not self.hist_axes_check.isChecked()
        ):
            return
        total, uncertainty, _count = integrated_box_sum(values, errors)
        self.roi_sum_text = _draw_box_sum_annotation(
            self.ax_image,
            extents,
            total,
            uncertainty,
        )

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
        if self._current_slice is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._tiled_mode_active():
            tile_index = next(
                (
                    index
                    for index, axis in enumerate(self._tile_axes)
                    if event.inaxes is axis
                ),
                None,
            )
            if tile_index is None or tile_index >= len(self._current_tiled_slices):
                return
            panel = self._current_tiled_slices[tile_index]
            coordinate_overrides = (
                {self.tile_dim: panel.coordinate}
                if self.tile_dim is not None
                else None
            )
            self._on_2d_motion(
                event,
                panel.view,
                values=panel.values,
                coordinate_overrides=coordinate_overrides,
            )
            return
        if event.inaxes != self.ax_image:
            return
        if self._waterfall_mode_active():
            self._on_waterfall_motion(event)
            return
        if self._is_effective_1d():
            self._on_line_motion(event)
            return
        self._on_2d_motion(event, self._current_slice)

    def _on_2d_motion(
        self,
        event,
        view: dict[str, np.ndarray],
        *,
        values: np.ndarray | None = None,
        coordinate_overrides: dict[int, float] | None = None,
    ) -> None:
        x_idx = int(np.searchsorted(view["x_edges"], event.xdata, side="right") - 1)
        y_idx = int(np.searchsorted(view["y_edges"], event.ydata, side="right") - 1)
        if not (0 <= x_idx < view["signal"].shape[1] and 0 <= y_idx < view["signal"].shape[0]):
            return
        display_values = (
            self.model._display_values(view)
            if values is None
            else np.asarray(values, dtype=float)
        )
        coords = self._cursor_hkle(
            x_idx,
            y_idx,
            coordinate_overrides=coordinate_overrides,
            displayed_coordinates=(
                float(view["x_centers"][x_idx]),
                float(view["y_centers"][y_idx]),
            ),
        )
        value_text, error_text = _format_value_with_uncertainty(
            float(display_values[y_idx, x_idx]),
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
        coords = self._cursor_hkle_1d(x_idx, x_value=float(x[x_idx]))
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

    def _cursor_hkle(
        self,
        x_idx: int,
        y_idx: int,
        *,
        coordinate_overrides: dict[int, float] | None = None,
        displayed_coordinates: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        hkle = np.zeros(4, dtype=float)
        has_energy = False
        hidden = self.model._normalized_selections()
        overrides = coordinate_overrides or {}
        for dim, _axis in enumerate(self.data.axes):
            if dim == self.model.x_dim:
                value = (
                    displayed_coordinates[0]
                    if displayed_coordinates is not None
                    else self.data.axes[dim].centers[x_idx]
                )
            elif dim == self.model.y_dim:
                value = (
                    displayed_coordinates[1]
                    if displayed_coordinates is not None
                    else self.data.axes[dim].centers[y_idx]
                )
            elif dim in overrides:
                value = overrides[dim]
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

    def _cursor_hkle_1d(
        self,
        x_idx: int,
        *,
        x_value: float | None = None,
    ) -> dict[str, float]:
        if getattr(self.model, "is_point_list", False):
            return self._cursor_hkle_point_list_1d(x_idx)
        hkle = np.zeros(4, dtype=float)
        has_energy = False
        hidden = self.model._normalized_selections()
        for dim, _axis in enumerate(self.data.axes):
            if dim == self.model.x_dim:
                value = (
                    float(x_value)
                    if x_value is not None
                    else self.data.axes[dim].centers[x_idx]
                )
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
