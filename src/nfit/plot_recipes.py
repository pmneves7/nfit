"""Backend plot recipes shared by saved plots, the data viewer, and scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from .mdhisto import MDHistoData
from .pipeline import PlotEntry, PlotSourceRef
from .plotting import (
    _resolve_mdhisto_dim,
    plot_mdhisto_fit_comparison,
    plot_mdhisto_line,
    plot_mdhisto_slice,
    plot_mdhisto_tiled_slices,
    plot_mdhisto_waterfall,
)

PLOT_TYPE_LABELS = {
    "mdhisto_slice": "MDHisto slice",
    "mdhisto_tiled_slices": "Tiled 2D slices",
    "mdhisto_line": "MDHisto line",
    "mdhisto_waterfall": "Waterfall",
    "fit_comparison": "Data, fit, and residual",
    "fit_covariance": "Fit covariance or correlation",
}


def new_plot_entry(
    name: str,
    dataset_id: str | None,
    settings: dict[str, Any],
    *,
    plot_type: str,
    fit_id: str | None = None,
    dataset_ids: list[str] | None = None,
) -> PlotEntry:
    """Create a plot entry using only stable source identifiers."""

    if plot_type not in PLOT_TYPE_LABELS:
        raise ValueError(f"unknown plot type {plot_type!r}")
    return PlotEntry(
        name=name,
        type=plot_type,
        sources=[
            PlotSourceRef(dataset_id=source_id, fit_id=fit_id)
            for source_id in (dataset_ids or [dataset_id])
        ],
        settings=dict(settings),
    )


def render_plot(
    entry: PlotEntry,
    data: MDHistoData | list[MDHistoData] | None = None,
    *,
    fit_entry: Any | None = None,
):
    """Return the Matplotlib figure for a saved MDHisto plot without Qt."""

    settings = dict(entry.settings)
    plot_type = entry.type
    if plot_type == "fit_covariance":
        if fit_entry is None:
            raise ValueError("fit covariance plot requires a fit result")
        figure = _render_fit_covariance(fit_entry, settings)
    elif data is None:
        raise ValueError("saved plot requires MDHisto data")
    elif plot_type == "mdhisto_waterfall":
        datasets = [data] if isinstance(data, MDHistoData) else list(data)
        ax = plot_mdhisto_waterfall(
            datasets,
            dataset_labels=settings.get("waterfall_dataset_names"),
            x_dim=settings.get("x_dim", -1),
            waterfall_dim=settings.get("y_dim", 0),
            channel=settings.get("channel", "signal"),
            selections=_selections(settings),
            integrate_checks=_integrate_checks(settings),
            waterfall_step=float(settings.get("waterfall_step", 1.0)),
            coverage_threshold=float(
                settings.get("waterfall_coverage_threshold", 0.9)
            ),
            trace_offset=float(settings.get("waterfall_offset", 1.0)),
            cmap=settings.get("waterfall_cmap", "viridis"),
            color_range=(
                float(settings.get("waterfall_color_min", 0.0)),
                float(settings.get("waterfall_color_max", 1.0)),
            ),
            reverse_colors=bool(settings.get("waterfall_reverse_colors", False)),
            marker=settings.get("marker", "o"),
            line_style=settings.get("line_style", "none"),
            marker_size=float(settings.get("marker_size", 5.0)),
            line_width=float(settings.get("line_plot_width", 1.5)),
            marker_edge_width=float(settings.get("marker_edge_width", 1.5)),
            marker_face=settings.get(
                "waterfall_marker_face",
                settings.get("marker_face_color", "none"),
            ),
            show_errorbars=bool(settings.get("show_errorbars", True)),
            errorbar_caps=bool(settings.get("show_errorbar_caps", False)),
            errorbar_cap_size=float(settings.get("errorbar_cap_size", 3.0)),
            show_zero_lines=bool(settings.get("waterfall_show_zero_lines", True)),
            zero_line_color=settings.get("waterfall_zero_color", "#7f7f7f"),
            zero_line_style=settings.get("waterfall_zero_style", "--"),
            zero_line_width=float(settings.get("waterfall_zero_width", 0.8)),
            show_model=bool(settings.get("show_fit", False)),
            unmask_model=bool(settings.get("unmask_model", False)),
            model_color=settings.get("waterfall_model_color"),
            model_line_width=float(settings.get("fit_line_width", 2.0)),
            show_trace_labels=bool(
                settings.get("waterfall_show_trace_labels", True)
            ),
            trace_label_suffix=str(
                settings.get("waterfall_trace_label_suffix", "")
            ),
            trace_label_font_size=float(
                settings.get("waterfall_trace_label_font_size", 10.0)
            ),
            trace_label_color=settings.get("waterfall_trace_label_color"),
            smoothing_sigma_x=float(settings.get("smoothing_x", 0.0)),
            smoothing_sigma_waterfall=float(settings.get("smoothing_y", 0.0)),
            smoothing_fill_nans=bool(settings.get("smoothing_fill_nans", False)),
            xlim=_pair(settings.get("xlim")),
            ylim=_pair(settings.get("ylim")),
            font_size=float(settings.get("font_size", 12.0)),
            axes_linewidth=float(settings.get("axis_linewidth", 1.5)),
            figsize=tuple(settings.get("figsize", (8.0, 6.5))),
        )
        figure = ax.figure
    elif plot_type == "mdhisto_tiled_slices":
        if not isinstance(data, MDHistoData):
            raise TypeError("tiled-slice plots require one MDHisto dataset")
        figure = plot_mdhisto_tiled_slices(
            data,
            x_dim=settings.get("x_dim", -1),
            y_dim=settings.get("y_dim", 0),
            tile_dim=settings.get("tile_dim", 1),
            channel=settings.get("channel", "signal"),
            selections=_selections(settings),
            integrate_checks=_integrate_checks(settings),
            tile_range=_pair(settings.get("tile_range")),
            tile_step=(
                None if settings.get("tile_step_auto", False)
                and "metadata_dimension" in data.axes[_resolve_mdhisto_dim(data, settings.get("tile_dim", 1))].metadata
                else float(settings.get("tile_step", 1.0))
            ),
            coverage_threshold=float(settings.get("coverage_threshold", 0.9)),
            masked=bool(settings.get("apply_masks", True)),
            cmap=settings.get("cmap", "viridis"),
            color_scale=settings.get("color_scale", "linear"),
            auto_limits=settings.get("auto_limits", "min/max"),
            autoscale=bool(settings.get("autoscale", True)),
            manual_vmin=settings.get("manual_vmin"),
            manual_vmax=settings.get("manual_vmax"),
            sigma_n=float(settings.get("sigma_n", 3.0)),
            iqr_n=float(settings.get("iqr_n", 1.5)),
            percentile_n=float(settings.get("percentile_n", 1.0)),
            power_gamma=float(settings.get("power_gamma", 0.5)),
            color_alpha=float(settings.get("color_alpha", 0.0)),
            smoothing_sigma_x=float(settings.get("smoothing_x", 0.0)),
            smoothing_sigma_y=float(settings.get("smoothing_y", 0.0)),
            smoothing_fill_nans=bool(settings.get("smoothing_fill_nans", False)),
            xlim=_pair(settings.get("xlim")),
            ylim=_pair(settings.get("ylim")),
            font_size=float(settings.get("font_size", 12.0)),
            axes_linewidth=float(settings.get("axis_linewidth", 1.5)),
            tile_label_decimals=int(settings.get("tile_label_decimals", 1)),
            tile_label_prefix=str(settings.get("tile_label_prefix", "{axis} = ")),
            tile_label_unit=str(settings.get("tile_label_unit", "{unit}")),
            tile_label_si_prefix=str(settings.get("tile_label_si_prefix", "")),
            show_tile_labels=bool(settings.get("show_tile_labels", True)),
            local_color_scales=bool(
                settings.get("tile_local_color_scales", False)
            ),
            show_brillouin_zone_boundaries=bool(
                settings.get("show_brillouin_zone_boundaries", False)
            ),
            brillouin_zone_spacegroup=settings.get("brillouin_zone_spacegroup"),
            brillouin_zone_lattice_parameters=settings.get(
                "brillouin_zone_lattice_parameters"
            ),
            brillouin_zone_color=settings.get("brillouin_zone_color", "#e57373"),
            brillouin_zone_linewidth=float(
                settings.get("brillouin_zone_linewidth", 1.0)
            ),
            brillouin_zone_alpha=float(settings.get("brillouin_zone_alpha", 0.75)),
            figsize=tuple(settings.get("figsize", (10.0, 8.0))),
        )
    elif plot_type == "mdhisto_line":
        if not isinstance(data, MDHistoData):
            raise TypeError("line plots require one MDHisto dataset")
        axis = settings.get("x_dim")
        ax = plot_mdhisto_line(
            data,
            axis_dim=axis,
            channel=settings.get("channel", "signal"),
            smoothing_sigma=float(settings.get("smoothing_x", 0.0)),
            smoothing_fill_nans=bool(settings.get("smoothing_fill_nans", False)),
        )
        figure = ax.figure
    elif plot_type == "fit_comparison":
        if not isinstance(data, MDHistoData):
            raise TypeError("fit-comparison plots require one MDHisto dataset")
        unmask_model = bool(settings.get("unmask_model", False))
        fit = _channel_data(data, "fit", unmasked=unmask_model)
        residual = _channel_data(data, "residual", unmasked=unmask_model)
        figure = plot_mdhisto_fit_comparison(
            data,
            fit,
            residual=residual,
            show_residual=bool(settings.get("show_residual", True)),
            x_dim=settings.get("x_dim", -1),
            y_dim=settings.get("y_dim", 0),
            channel=settings.get("channel", "signal"),
            selections=_selections(settings),
            integrate_checks=_integrate_checks(settings),
            cmap=settings.get("cmap", "viridis"),
            color_scale=settings.get("color_scale", "linear"),
            auto_limits=settings.get("auto_limits", "min/max"),
            show_brillouin_zone_boundaries=bool(
                settings.get("show_brillouin_zone_boundaries", False)
            ),
            brillouin_zone_spacegroup=settings.get("brillouin_zone_spacegroup"),
            brillouin_zone_lattice_parameters=settings.get(
                "brillouin_zone_lattice_parameters"
            ),
            brillouin_zone_color=settings.get("brillouin_zone_color", "#e57373"),
            brillouin_zone_linewidth=float(
                settings.get("brillouin_zone_linewidth", 1.0)
            ),
            brillouin_zone_alpha=float(settings.get("brillouin_zone_alpha", 0.75)),
            figsize=tuple(settings.get("figsize", (8.0, 6.5))),
        )
    elif plot_type == "mdhisto_slice":
        if not isinstance(data, MDHistoData):
            raise TypeError("slice plots require one MDHisto dataset")
        figure = plot_mdhisto_slice(
            data,
            x_dim=settings.get("x_dim", -1),
            y_dim=settings.get("y_dim", 0),
            channel=settings.get("channel", "signal"),
            selections=_selections(settings),
            integrate_checks=_integrate_checks(settings),
            coverage_threshold=float(settings.get("coverage_threshold", 0.9)),
            cmap=settings.get("cmap", "viridis"),
            color_scale=settings.get("color_scale", "linear"),
            auto_limits=settings.get("auto_limits", "min/max"),
            autoscale=bool(settings.get("autoscale", True)),
            manual_vmin=settings.get("manual_vmin"),
            manual_vmax=settings.get("manual_vmax"),
            sigma_n=float(settings.get("sigma_n", 3.0)),
            iqr_n=float(settings.get("iqr_n", 1.5)),
            percentile_n=float(settings.get("percentile_n", 1.0)),
            power_gamma=float(settings.get("power_gamma", 0.5)),
            color_alpha=float(settings.get("color_alpha", 0.0)),
            smoothing_sigma_x=float(settings.get("smoothing_x", 0.0)),
            smoothing_sigma_y=float(settings.get("smoothing_y", 0.0)),
            smoothing_fill_nans=bool(settings.get("smoothing_fill_nans", False)),
            xlim=_pair(settings.get("xlim")),
            ylim=_pair(settings.get("ylim")),
            font_size=float(settings.get("font_size", 12.0)),
            axes_linewidth=float(settings.get("axis_linewidth", 1.5)),
            show_histogram_axes=bool(settings.get("show_histogram_axes", False)),
            roi_extents=_quad(settings.get("roi_extents")),
            xcut_percent=float(settings.get("xcut_percent", 20.0)),
            ycut_percent=float(settings.get("ycut_percent", 16.0)),
            show_brillouin_zone_boundaries=bool(
                settings.get("show_brillouin_zone_boundaries", False)
            ),
            brillouin_zone_spacegroup=settings.get("brillouin_zone_spacegroup"),
            brillouin_zone_lattice_parameters=settings.get(
                "brillouin_zone_lattice_parameters"
            ),
            brillouin_zone_color=settings.get("brillouin_zone_color", "#e57373"),
            brillouin_zone_linewidth=float(
                settings.get("brillouin_zone_linewidth", 1.0)
            ),
            brillouin_zone_alpha=float(settings.get("brillouin_zone_alpha", 0.75)),
            figsize=tuple(settings.get("figsize", (8.0, 6.5))),
        )
    else:
        raise ValueError(f"unsupported plot type {plot_type!r}")
    _apply_presentation(figure, settings)
    return figure


def plot_script(entry: PlotEntry, *, project_path: str | Path) -> str:
    """Generate an editable backend-only Python script for a saved plot."""

    source = entry.sources[0] if entry.sources else PlotSourceRef()
    settings = pformat(entry.settings, sort_dicts=False, width=88)
    return "\n".join(
        [
            "from pathlib import Path",
            "",
            "from nfit import load_project, render_project_plot",
            "",
            f"PROJECT_PATH = Path({str(project_path)!r})",
            f"PLOT_ID = {entry.id!r}",
            f"DATASET_ID = {source.dataset_id!r}",
            f"PLOT_SETTINGS = {settings}",
            "",
            "def make_figure():",
            "    project = load_project(PROJECT_PATH)",
            "    # The project resolves masks, scale factors, derived data, and fit channels.",
            "    return render_project_plot(project, PLOT_ID, settings=PLOT_SETTINGS)",
            "",
            "if __name__ == '__main__':",
            "    figure = make_figure()",
            "    figure.savefig('plot.png', dpi=300)",
            "",
        ]
    )


def plot_entry_to_dict(entry: PlotEntry) -> dict[str, Any]:
    """Return a JSON-ready plot entry representation."""

    payload = asdict(entry)
    payload["sources"] = [asdict(source) for source in entry.sources]
    return payload


def plot_entry_from_dict(payload: dict[str, Any]) -> PlotEntry:
    """Restore a plot entry, tolerating early project files without IDs."""

    return PlotEntry(
        name=str(payload.get("name", "Plot")),
        type=str(payload.get("type", "mdhisto_slice")),
        sources=[PlotSourceRef(**source) for source in payload.get("sources", []) if isinstance(source, dict)],
        settings=dict(payload.get("settings", {})),
        renderer_version=int(payload.get("renderer_version", 1)),
        source_fingerprints=dict(payload.get("source_fingerprints", {})),
        metadata=dict(payload.get("metadata", {})),
        id=str(payload.get("id") or PlotEntry("").id),
    )


def _channel_data(
    data: MDHistoData,
    name: str,
    *,
    unmasked: bool = False,
) -> MDHistoData:
    channel = data.auxiliary_channels.get(name)
    metadata_values = data.metadata.get(name)
    if channel is None and not (
        isinstance(metadata_values, np.ndarray) and metadata_values.shape == data.shape
    ):
        raise ValueError(f"plot requires a stored {name!r} channel")
    values = channel.values if channel is not None else metadata_values
    errors = channel.errors if channel is not None else None
    return MDHistoData(
        data.axes,
        values,
        errors if errors is not None else data.errors,
        np.zeros(data.shape, dtype=bool) if unmasked else data.mask,
        data.num_events,
        metadata=dict(data.metadata),
    )


def _selections(settings: dict[str, Any]) -> dict[int, tuple[float, float]]:
    return {int(dim): tuple(values) for dim, values in dict(settings.get("selections", {})).items()}


def _integrate_checks(settings: dict[str, Any]) -> dict[int, bool]:
    return {int(dim): bool(value) for dim, value in dict(settings.get("integrate_checks", {})).items()}


def _pair(value: Any) -> tuple[float, float] | None:
    return tuple(value) if isinstance(value, (list, tuple)) and len(value) == 2 else None


def _quad(value: Any) -> tuple[float, float, float, float] | None:
    return tuple(value) if isinstance(value, (list, tuple)) and len(value) == 4 else None


def _apply_presentation(figure: Any, settings: dict[str, Any]) -> None:
    title = str(settings.get("title", "")).strip()
    if title:
        figure.axes[0].set_title(title)
    xlabel = str(settings.get("xlabel", "")).strip()
    ylabel = str(settings.get("ylabel", "")).strip()
    if xlabel:
        figure.axes[0].set_xlabel(xlabel)
    if ylabel:
        figure.axes[0].set_ylabel(ylabel)


def _render_fit_covariance(fit_entry: Any, settings: dict[str, Any]):
    """Draw a fit covariance/correlation matrix from the persisted fit result."""

    import matplotlib.pyplot as plt

    goodness = getattr(fit_entry, "goodness", {})
    covariance = goodness.get("covariance", {}) if isinstance(goodness, dict) else {}
    names = [str(name) for name in covariance.get("variables", [])]
    matrix = covariance.get("matrix")
    title = "Covariance"
    if matrix is None:
        matrix = covariance.get("correlation")
        title = "Correlation"
        if isinstance(matrix, dict):
            names = names or [str(name) for name in matrix]
            matrix = [[dict(matrix.get(row, {})).get(col, np.nan) for col in names] for row in names]
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("fit result does not contain a square covariance or correlation matrix")
    if len(names) != array.shape[0]:
        names = [f"p{index + 1}" for index in range(array.shape[0])]
    figure, axes = plt.subplots(figsize=tuple(settings.get("figsize", (7.0, 6.0))))
    limit = 1.0 if title == "Correlation" else max(1.0, float(np.nanmax(np.abs(array))))
    image = axes.imshow(array, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    axes.set_yticks(np.arange(len(names)), names)
    axes.set_title(title)
    figure.colorbar(image, ax=axes, label=title)
    return figure
