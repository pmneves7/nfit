from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .colormaps import IMAGE_COLORMAPS
from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData, mdhisto_coverage_fraction, mdhisto_measured_bins
from .quantities import display_axis_label, display_channel_label, display_unit


@dataclass(frozen=True)
class WaterfallTrace:
    """One prepared trace in a waterfall plot."""

    x: np.ndarray
    values: np.ndarray
    errors: np.ndarray | None
    model_values: np.ndarray | None
    label: str
    waterfall_coordinate: float | None = None
    waterfall_unit: str = ""


@dataclass(frozen=True)
class TiledSlice:
    """One prepared 2D panel in a tiled-slice plot."""

    view: dict[str, np.ndarray]
    values: np.ndarray
    coordinate: float
    lower: float
    upper: float
    label: str


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Return bin edges bracketing sorted 1D bin centers (for pcolormesh-free plots)."""

    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    deltas = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * deltas[0]
    edges[-1] = centers[-1] + 0.5 * deltas[-1]
    return edges


def plot_energy_cut(
    data: PointData4D,
    *,
    ax=None,
    model_values: ArrayLike | None = None,
    label: str = "data",
):
    """Plot intensity versus energy transfer with optional model overlay."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    order = np.argsort(data.E)
    ax.errorbar(
        data.E[order],
        data.intensity[order],
        yerr=data.sigma[order],
        fmt="o",
        ms=4,
        label=label,
    )
    if model_values is not None:
        model = np.asarray(model_values, dtype=float)
        ax.plot(data.E[order], model[order], "-", label="model")
    ax.set_xlabel("Energy transfer E (meV)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return ax


def plot_q_cut(
    data: PointData4D,
    coordinate: str = "H",
    *,
    ax=None,
    model_values: ArrayLike | None = None,
    label: str = "data",
):
    """Plot intensity versus one reciprocal-lattice coordinate."""

    import matplotlib.pyplot as plt

    if coordinate not in {"H", "K", "L"}:
        raise ValueError("coordinate must be one of 'H', 'K', or 'L'")
    x = getattr(data, coordinate)
    order = np.argsort(x)
    if ax is None:
        _, ax = plt.subplots()
    ax.errorbar(x[order], data.intensity[order], yerr=data.sigma[order], fmt="o", ms=4, label=label)
    if model_values is not None:
        model = np.asarray(model_values, dtype=float)
        ax.plot(x[order], model[order], "-", label="model")
    ax.set_xlabel(f"{coordinate} (RLU)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return ax


def plot_2d_map(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    *,
    ax=None,
    xlabel: str = "H (RLU)",
    ylabel: str = "K (RLU)",
    clabel: str = "Intensity",
    gridsize: int = 60,
):
    """Plot a 2D map from point-cloud values.

    The helper uses ``tricontourf`` for irregular point clouds and falls back to
    scatter if triangulation is not possible.
    """

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    try:
        triangulation = mtri.Triangulation(x_arr, y_arr)
        artist = ax.tricontourf(triangulation, v_arr, levels=gridsize)
    except Exception:
        artist = ax.scatter(x_arr, y_arr, c=v_arr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = ax.figure.colorbar(artist, ax=ax)
    cbar.set_label(clabel)
    return ax


def plot_mdhisto_slice(
    data: MDHistoData,
    *,
    x_dim: int | str = -1,
    y_dim: int | str = 0,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    coverage_threshold: float = 0.9,
    cmap: str = "viridis",
    color_scale: str = "linear",
    auto_limits: str = "min/max",
    autoscale: bool = True,
    manual_vmin: float | None = None,
    manual_vmax: float | None = None,
    sigma_n: float = 3.0,
    iqr_n: float = 1.5,
    percentile_n: float = 1.0,
    power_gamma: float = 0.5,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_y: float = 0.0,
    smoothing_fill_nans: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 10.0,
    show_histogram_axes: bool = False,
    roi_extents: tuple[float, float, float, float] | None = None,
    xcut_percent: float = 20.0,
    ycut_percent: float = 16.0,
    axes_linewidth: float = 1.0,
    figsize: tuple[float, float] = (8.0, 6.5),
):
    """Render a non-interactive MDHisto slice figure.

    This is the scripting-friendly counterpart to ``slice_viewer``. It uses the
    same slicing, integration, masking, and color normalization logic as the
    interactive viewers, but returns a Matplotlib ``Figure`` for notebooks,
    scripts, and exports.
    """

    import matplotlib.pyplot as plt

    model = MDHistoSliceViewer(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        channel=channel,
        cmap=cmap,
        color_scale=color_scale,
        auto_limits=auto_limits,
        coverage_threshold=coverage_threshold,
    )
    if selections:
        model.selections.update({int(dim): tuple(value) for dim, value in selections.items()})
    if integrate_checks:
        model.integrate_checks.update({int(dim): bool(value) for dim, value in integrate_checks.items()})
    model.autoscale = bool(autoscale)
    model.manual_vmin = manual_vmin
    model.manual_vmax = manual_vmax
    model.sigma_n = float(sigma_n)
    model.iqr_n = float(iqr_n)
    model.percentile_n = float(percentile_n)
    model.power_gamma = float(power_gamma)

    view = smooth_mdhisto_view(
        model.slice_arrays(),
        sigma_x=smoothing_sigma_x,
        sigma_y=smoothing_sigma_y,
        fill_nans=smoothing_fill_nans,
    )
    with plt.rc_context({"font.size": float(font_size)}):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        if show_histogram_axes:
            x_ratio = _panel_ratio(xcut_percent)
            y_ratio = _panel_ratio(ycut_percent)
            grid = fig.add_gridspec(
                2,
                3,
                width_ratios=[1.0, y_ratio, 0.045],
                height_ratios=[1.0, x_ratio],
            )
            ax_image = fig.add_subplot(grid[0, 0])
            ax_ycut = fig.add_subplot(grid[0, 1], sharey=ax_image)
            ax_colorbar = fig.add_subplot(grid[0, 2])
            ax_xcut = fig.add_subplot(grid[1, 0], sharex=ax_image)
        else:
            grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045])
            ax_image = fig.add_subplot(grid[0, 0])
            ax_colorbar = fig.add_subplot(grid[0, 1])
            ax_xcut = None
            ax_ycut = None

        values = model._display_values(view)
        image = ax_image.pcolormesh(
            view["x_edges"],
            view["y_edges"],
            values,
            shading="auto",
            cmap=model.cmap,
            norm=model._color_norm(values),
        )
        ax_image.set_xlabel(model._axis_label(model.x_dim))
        ax_image.set_ylabel(model._axis_label(model.y_dim))
        if xlim is not None:
            ax_image.set_xlim(*xlim)
        if ylim is not None:
            ax_image.set_ylim(*ylim)
        colorbar = fig.colorbar(image, cax=ax_colorbar)
        colorbar.set_label(model._channel_label())
        _apply_axes_linewidth((ax_image, ax_colorbar, ax_xcut, ax_ycut), colorbar, axes_linewidth)

        if show_histogram_axes and roi_extents is not None and ax_xcut is not None and ax_ycut is not None:
            _draw_mdhisto_roi_cuts(model, view, roi_extents, ax_xcut, ax_ycut)
            _apply_axes_linewidth((ax_image, ax_colorbar, ax_xcut, ax_ycut), colorbar, axes_linewidth)

    return fig


def prepare_mdhisto_tiled_slices(
    data: MDHistoData,
    *,
    x_dim: int | str = -1,
    y_dim: int | str = 0,
    tile_dim: int | str = 1,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    tile_range: tuple[float, float] | None = None,
    tile_step: float | None = None,
    tile_label_decimals: int = 1,
    tile_label_prefix: str = "{axis} = ",
    tile_label_unit: str = "{unit}",
    tile_label_si_prefix: str = "",
    coverage_threshold: float = 0.9,
    masked: bool = True,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_y: float = 0.0,
    smoothing_fill_nans: bool = True,
) -> list[TiledSlice]:
    """Prepare coarse third-axis bins as a sequence of 2D slices.

    The two displayed dimensions remain intact. Each panel integrates the
    selected bins along ``tile_dim``; any further dimensions use the ordinary
    slice-viewer point or range selections. With ``tile_step=None``, metadata
    axes produce one panel per coordinate, including irregularly spaced values.
    Label options affect presentation only: decimals are fixed decimal places,
    the prefix replaces `{axis}` with the axis name, and the unit replaces
    `{unit}` with its original unit. The SI prefix scales values relative to
    that original unit and is prepended to the label unit.
    """

    if not isinstance(data, MDHistoData):
        raise TypeError("tiled slices require one MDHistoData object")
    x_index = _resolve_mdhisto_dim(data, x_dim)
    y_index = _resolve_mdhisto_dim(data, y_dim)
    tile_index = _resolve_mdhisto_dim(data, tile_dim)
    if len({x_index, y_index, tile_index}) != 3:
        raise ValueError("x_dim, y_dim, and tile_dim must be different")
    if data.shape[tile_index] <= 1:
        raise ValueError("tile_dim must contain more than one bin")

    centers = np.asarray(data.axes[tile_index].centers, dtype=float)
    edges = np.asarray(data.axes[tile_index].values, dtype=float)
    if edges.size != centers.size + 1:
        edges = _edges_from_centers(centers)
    if tile_range is None:
        low_value, high_value = float(centers[0]), float(centers[-1])
    else:
        low_value, high_value = sorted(map(float, tile_range))
    width = (
        default_tiled_slice_step(data, tile_index, (low_value, high_value))
        if tile_step is None
        else float(tile_step)
    )
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("tile_step must be positive")
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        raise ValueError("tile_range must be finite")
    exact_coordinates = tile_step is None and "metadata_dimension" in data.axes[tile_index].metadata
    if exact_coordinates:
        selected_indices = np.flatnonzero((centers >= low_value) & (centers <= high_value))
        coordinates = centers[selected_indices]
        intervals = [(edges[i], edges[i + 1]) for i in selected_indices]
        count = len(coordinates)
    else:
        count = max(int(np.floor((high_value - low_value) / width + 1.e-10)) + 1, 1)
        coordinates = low_value + np.arange(count, dtype=float) * width
        boundaries = low_value + (np.arange(count + 1, dtype=float) - 0.5) * width
        intervals = zip(boundaries[:-1], boundaries[1:], strict=True)

    base = MDHistoSliceViewer(
        data,
        x_dim=x_index,
        y_dim=y_index,
        channel=channel,
        masked=masked,
        coverage_threshold=coverage_threshold,
    )
    if selections:
        base.selections.update(
            {int(dim): tuple(value) for dim, value in selections.items()}
        )
    if integrate_checks:
        base.integrate_checks.update(
            {int(dim): bool(value) for dim, value in integrate_checks.items()}
        )
    axis = data.axes[tile_index]
    axis_name = waterfall_axis_display_name(axis.name)
    displayed_unit = display_unit(axis.units) if axis.units else ""
    si_exponents = {"": 0, "p": -12, "n": -9, "µ": -6, "m": -3, "k": 3, "M": 6, "G": 9, "T": 12}
    if tile_label_si_prefix not in si_exponents:
        raise ValueError("unsupported tiled-label SI prefix")
    if not 0 <= tile_label_decimals <= 10:
        raise ValueError("tile_label_decimals must be between 0 and 10")
    label_prefix = tile_label_prefix.replace("{axis}", axis_name)
    label_unit = tile_label_unit.replace("{unit}", displayed_unit)
    unit_suffix = f" {tile_label_si_prefix}{label_unit}" if label_unit else ""
    slices: list[TiledSlice] = []
    for index, (low, high) in enumerate(intervals):
        tolerance = max(abs(low), abs(high), width, 1.0) * 1.e-12
        selected = (centers >= low - tolerance) & (
            (centers < high - tolerance) if index < count - 1 else (centers <= high + tolerance)
        )
        indices = np.array([selected_indices[index]]) if exact_coordinates else np.flatnonzero(selected)
        if indices.size == 0:
            continue
        panel = MDHistoSliceViewer(
            data,
            x_dim=x_index,
            y_dim=y_index,
            channel=channel,
            cmap=base.cmap,
            color_scale=base.color_scale,
            auto_limits=base.auto_limits,
            masked=masked,
            coverage_threshold=coverage_threshold,
        )
        panel.selections.update(dict(base.selections))
        panel.integrate_checks.update(dict(base.integrate_checks))
        panel.selections[tile_index] = (
            float(centers[indices[0]]),
            float(centers[indices[-1]]),
        )
        panel.integrate_checks[tile_index] = True
        view = smooth_mdhisto_view(
            panel.slice_arrays(),
            sigma_x=smoothing_sigma_x,
            sigma_y=smoothing_sigma_y,
            fill_nans=smoothing_fill_nans,
        )
        coordinate = float(coordinates[index])
        if not exact_coordinates and abs(coordinate) < width * 1.e-10:
            coordinate = 0.0
        # Adding zero suppresses negative zero after rounding (also on Python 3.10).
        label_value = round(
            coordinate / 10.0 ** si_exponents[tile_label_si_prefix], tile_label_decimals
        ) + 0.0
        slices.append(
            TiledSlice(
                view=view,
                values=np.asarray(panel._display_values(view), dtype=float),
                coordinate=coordinate,
                lower=float(edges[indices[0]]),
                upper=float(edges[indices[-1] + 1]),
                label=(
                    f"{label_prefix}"
                    f"{label_value:.{tile_label_decimals}f}"
                    f"{unit_suffix}"
                ),
            )
        )
    return slices


def default_tiled_slice_step(
    data: MDHistoData,
    tile_dim: int | str,
    tile_range: tuple[float, float] | None = None,
) -> float:
    """Choose a third-axis bin width that produces up to nine panels."""

    index = _resolve_mdhisto_dim(data, tile_dim)
    centers = np.asarray(data.axes[index].centers, dtype=float)
    edges = np.asarray(data.axes[index].values, dtype=float)
    if edges.size != centers.size + 1:
        edges = _edges_from_centers(centers)
    if tile_range is None:
        selected = np.arange(centers.size)
    else:
        low, high = sorted(map(float, tile_range))
        selected = np.flatnonzero((centers >= low) & (centers <= high))
    if selected.size == 0:
        return 1.0
    if selected.size <= 9:
        widths = np.abs(np.diff(edges))[selected]
        positive = widths[np.isfinite(widths) & (widths > 0.0)]
        if positive.size:
            return float(np.min(positive))
    span = float(centers[int(selected[-1])] - centers[int(selected[0])])
    target = max(min(int(selected.size), 9), 1)
    return span / max(target - 1, 1) if np.isfinite(span) and span > 0.0 else 1.0


def plot_mdhisto_tiled_slices(
    data: MDHistoData,
    *,
    x_dim: int | str = -1,
    y_dim: int | str = 0,
    tile_dim: int | str = 1,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    tile_range: tuple[float, float] | None = None,
    tile_step: float | None = None,
    coverage_threshold: float = 0.9,
    masked: bool = True,
    cmap: str = "viridis",
    color_scale: str = "linear",
    auto_limits: str = "min/max",
    autoscale: bool = True,
    manual_vmin: float | None = None,
    manual_vmax: float | None = None,
    sigma_n: float = 3.0,
    iqr_n: float = 1.5,
    percentile_n: float = 1.0,
    power_gamma: float = 0.5,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_y: float = 0.0,
    smoothing_fill_nans: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 10.0,
    axes_linewidth: float = 1.0,
    tile_label_decimals: int = 1,
    tile_label_prefix: str = "{axis} = ",
    tile_label_unit: str = "{unit}",
    tile_label_si_prefix: str = "",
    show_tile_labels: bool = True,
    local_color_scales: bool = False,
    figsize: tuple[float, float] = (10.0, 8.0),
):
    """Render a grid of 2D slices with global or per-panel color scales."""

    import matplotlib.pyplot as plt

    slices = prepare_mdhisto_tiled_slices(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        tile_dim=tile_dim,
        channel=channel,
        selections=selections,
        integrate_checks=integrate_checks,
        tile_range=tile_range,
        tile_step=tile_step,
        tile_label_decimals=tile_label_decimals,
        tile_label_prefix=tile_label_prefix,
        tile_label_unit=tile_label_unit,
        tile_label_si_prefix=tile_label_si_prefix,
        coverage_threshold=coverage_threshold,
        masked=masked,
        smoothing_sigma_x=smoothing_sigma_x,
        smoothing_sigma_y=smoothing_sigma_y,
        smoothing_fill_nans=smoothing_fill_nans,
    )
    if not slices:
        raise ValueError("the tiled-slice settings produced no panels")
    model = MDHistoSliceViewer(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        channel=channel,
        cmap=cmap,
        color_scale=color_scale,
        auto_limits=auto_limits,
        coverage_threshold=coverage_threshold,
    )
    model.autoscale = bool(autoscale)
    model.manual_vmin = manual_vmin
    model.manual_vmax = manual_vmax
    model.sigma_n = float(sigma_n)
    model.iqr_n = float(iqr_n)
    model.percentile_n = float(percentile_n)
    model.power_gamma = float(power_gamma)
    if local_color_scales and not model.autoscale:
        raise ValueError("local_color_scales requires autoscale=True")
    combined = np.concatenate([panel.values.ravel() for panel in slices])
    shared_norm = model._color_norm(combined)
    columns = int(np.ceil(np.sqrt(len(slices))))
    rows = int(np.ceil(len(slices) / columns))
    with plt.rc_context({"font.size": float(font_size)}):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        if local_color_scales:
            grid = fig.add_gridspec(
                rows,
                columns * 2,
                width_ratios=[value for _ in range(columns) for value in (1.0, 0.055)],
            )
        else:
            grid = fig.add_gridspec(
                rows,
                columns + 1,
                width_ratios=[*[1.0] * columns, 0.055],
            )
        axes = []
        colorbars = []
        artist = None
        for index, panel in enumerate(slices):
            row, column = divmod(index, columns)
            plot_column = column * 2 if local_color_scales else column
            ax = fig.add_subplot(
                grid[row, plot_column],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
            norm = model._color_norm(panel.values) if local_color_scales else shared_norm
            artist = ax.pcolormesh(
                panel.view["x_edges"],
                panel.view["y_edges"],
                panel.values,
                shading="auto",
                cmap=model._effective_cmap(),
                norm=norm,
            )
            if show_tile_labels:
                ax.text(
                    0.97,
                    0.03,
                    panel.label,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    bbox={
                        "boxstyle": "square,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                )
            if row == rows - 1 or index + columns >= len(slices):
                ax.set_xlabel(model._axis_label(model.x_dim))
            if column == 0:
                ax.set_ylabel(model._axis_label(model.y_dim))
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            axes.append(ax)
            if local_color_scales:
                colorbar_axis = fig.add_subplot(grid[row, plot_column + 1])
                colorbar = fig.colorbar(artist, cax=colorbar_axis)
                if column == columns - 1:
                    colorbar.set_label(model._channel_label())
                colorbars.append(colorbar)
        if not local_color_scales:
            colorbar_axis = fig.add_subplot(grid[:, -1])
            colorbar = fig.colorbar(artist, cax=colorbar_axis)
            colorbar.set_label(model._channel_label())
            colorbars.append(colorbar)
        _apply_axes_linewidth(
            (*axes, *(colorbar.ax for colorbar in colorbars)),
            None,
            axes_linewidth,
        )
        for colorbar in colorbars:
            colorbar.outline.set_linewidth(float(axes_linewidth))
        fig._nfit_tiled_slices = slices
        fig._nfit_tiled_axes = axes
        fig._nfit_tiled_colorbars = colorbars
    return fig


def plot_mdhisto_line(
    data: MDHistoData,
    *,
    axis_dim: int | str | None = None,
    channel: str = "signal",
    smoothing_sigma: float = 0.0,
    smoothing_fill_nans: bool = True,
    ax=None,
):
    """Render a one-dimensional MDHisto channel as a line plot.

    Mantid reductions often keep singleton dimensions. This helper treats the
    selected non-singleton dimension as x and indexes the remaining dimensions at
    their first bin, so a shape like ``(1, 1, 1, N)`` becomes a normal line plot.
    """

    import matplotlib.pyplot as plt

    if axis_dim is None:
        non_singleton = [dim for dim, size in enumerate(data.shape) if size > 1]
        if len(non_singleton) != 1:
            raise ValueError("axis_dim is required unless exactly one dimension is non-singleton")
        axis_index = non_singleton[0]
    else:
        axis_index = _resolve_mdhisto_dim(data, axis_dim)
    channel_name = _resolve_mdhisto_channel(channel, data)
    values = _mdhisto_channel_array(data, channel_name)
    index = [0] * data.signal.ndim
    index[axis_index] = slice(None)
    y = np.asarray(values[tuple(index)], dtype=float)
    source_finite = np.isfinite(y)
    y = gaussian_smooth_nan(y, (max(float(smoothing_sigma), 0.0),))
    if not smoothing_fill_nans:
        y = np.where(source_finite, y, np.nan)
    x = data.axes[axis_index].centers
    if ax is None:
        _, ax = plt.subplots()
    if channel_name == "signal":
        yerr = np.asarray(_mdhisto_channel_array(data, "errors")[tuple(index)], dtype=float)
        source_error_finite = np.isfinite(yerr)
        yerr = gaussian_smooth_uncertainty(
            yerr,
            (max(float(smoothing_sigma), 0.0),),
        )
        if not smoothing_fill_nans:
            yerr = np.where(source_error_finite, yerr, np.nan)
        ax.errorbar(x, y, yerr=yerr, fmt="-", lw=1.2)
    else:
        ax.plot(x, y, "-", lw=1.2)
    axis = data.axes[axis_index]
    ax.set_xlabel(
        display_axis_label(
            axis.name,
            axis.units,
            quantity_type=axis.role,
        )
    )
    channel_model = MDHistoSliceViewer(
        data,
        x_dim=axis_index,
        y_dim=_waterfall_other_axis(data, axis_index),
        channel=channel_name,
    )
    ax.set_ylabel(channel_model._channel_label())
    return ax


def prepare_mdhisto_waterfall(
    data: MDHistoData | Sequence[MDHistoData],
    *,
    dataset_labels: Sequence[str] | None = None,
    x_dim: int | str = -1,
    waterfall_dim: int | str = 0,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    waterfall_step: float | None = None,
    coverage_threshold: float = 0.9,
    masked: bool = True,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_waterfall: float = 0.0,
    smoothing_fill_nans: bool = True,
    include_model: bool = False,
    unmask_model: bool = False,
) -> list[WaterfallTrace]:
    """Prepare one multidimensional dataset or several 1D datasets as traces.

    For multidimensional data, hidden dimensions use the ordinary slice-viewer
    selections and the waterfall dimension is grouped into bins of
    ``waterfall_step``. Each profile is an inverse-variance weighted mean, so
    its units remain those of the displayed channel. For a sequence of 1D
    datasets, each dataset contributes one trace.
    """

    datasets = [data] if isinstance(data, MDHistoData) else list(data)
    if not datasets or not all(isinstance(item, MDHistoData) for item in datasets):
        raise TypeError("waterfall plots require one or more MDHistoData objects")
    labels = (
        [str(label) for label in dataset_labels]
        if dataset_labels is not None
        else [f"dataset {index + 1}" for index in range(len(datasets))]
    )
    if len(labels) != len(datasets):
        raise ValueError("dataset_labels length must match the number of datasets")

    all_one_dimensional = all(_mdhisto_non_singleton_count(item) == 1 for item in datasets)
    if len(datasets) > 1 and not all_one_dimensional:
        raise ValueError("multiple-source waterfall plots require one-dimensional datasets")
    if all_one_dimensional:
        traces: list[WaterfallTrace] = []
        expected_axis: tuple[str, str] | None = None
        for dataset, label in zip(datasets, labels, strict=True):
            axis_index = _waterfall_1d_axis(dataset, x_dim)
            axis = dataset.axes[axis_index]
            axis_identity = (axis.name, axis.units)
            if expected_axis is None:
                expected_axis = axis_identity
            elif axis_identity != expected_axis:
                raise ValueError(
                    "grouped 1D waterfall datasets must share the same x-axis name and units"
                )
            model = MDHistoSliceViewer(
                dataset,
                x_dim=axis_index,
                y_dim=_waterfall_other_axis(dataset, axis_index),
                channel=channel,
                masked=masked,
                coverage_threshold=coverage_threshold,
            )
            view = smooth_mdhisto_view(
                model.slice_arrays(),
                sigma_x=smoothing_sigma_x,
                sigma_y=0.0,
                fill_nans=smoothing_fill_nans,
            )
            values = np.asarray(model._display_values(view), dtype=float).reshape(-1)
            errors = _waterfall_channel_errors(model, view, values.shape)
            model_values = _waterfall_model_values(view, values.shape) if include_model else None
            if include_model and unmask_model and "fit" in model.CHANNELS:
                fit_model = MDHistoSliceViewer(
                    dataset,
                    x_dim=axis_index,
                    y_dim=_waterfall_other_axis(dataset, axis_index),
                    channel="fit",
                    masked=False,
                    coverage_threshold=0.0,
                )
                fit_view = smooth_mdhisto_view(
                    fit_model.slice_arrays(),
                    sigma_x=smoothing_sigma_x,
                    sigma_y=0.0,
                    fill_nans=smoothing_fill_nans,
                )
                model_values = np.asarray(
                    fit_model._display_values(fit_view),
                    dtype=float,
                ).reshape(values.shape)
            traces.append(
                WaterfallTrace(
                    x=np.asarray(view["x_centers"], dtype=float),
                    values=values,
                    errors=errors,
                    model_values=model_values,
                    label=label,
                )
            )
        return traces

    dataset = datasets[0]
    x_index = _resolve_mdhisto_dim(dataset, x_dim)
    waterfall_index = _resolve_mdhisto_dim(dataset, waterfall_dim)
    if x_index == waterfall_index:
        raise ValueError("x_dim and waterfall_dim must be different")
    model = MDHistoSliceViewer(
        dataset,
        x_dim=x_index,
        y_dim=waterfall_index,
        channel=channel,
        masked=masked,
        coverage_threshold=0.0,
    )
    if selections:
        model.selections.update(
            {int(dim): tuple(value) for dim, value in selections.items()}
        )
    if integrate_checks:
        model.integrate_checks.update(
            {int(dim): bool(value) for dim, value in integrate_checks.items()}
        )
    view = smooth_mdhisto_view(
        model.slice_arrays(),
        sigma_x=smoothing_sigma_x,
        sigma_y=smoothing_sigma_waterfall,
        fill_nans=smoothing_fill_nans,
    )
    values = np.asarray(model._display_values(view), dtype=float)
    errors = _waterfall_channel_errors(model, view, values.shape)
    model_values = _waterfall_model_values(view, values.shape) if include_model else None
    if include_model and unmask_model and "fit" in model.CHANNELS:
        fit_model = MDHistoSliceViewer(
            dataset,
            x_dim=x_index,
            y_dim=waterfall_index,
            channel="fit",
            masked=False,
            coverage_threshold=0.0,
        )
        if selections:
            fit_model.selections.update(
                {int(dim): tuple(value) for dim, value in selections.items()}
            )
        if integrate_checks:
            fit_model.integrate_checks.update(
                {int(dim): bool(value) for dim, value in integrate_checks.items()}
            )
        fit_view = smooth_mdhisto_view(
            fit_model.slice_arrays(),
            sigma_x=smoothing_sigma_x,
            sigma_y=smoothing_sigma_waterfall,
            fill_nans=smoothing_fill_nans,
        )
        model_values = np.asarray(
            fit_model._display_values(fit_view),
            dtype=float,
        )
    axis = dataset.axes[waterfall_index]
    return _coarsen_waterfall_view(
        view,
        values,
        errors,
        model_values,
        step=waterfall_step,
        coverage_threshold=coverage_threshold,
        unmask_model=unmask_model,
        axis_name=axis.name,
        axis_units=axis.units,
    )


def plot_mdhisto_waterfall(
    data: MDHistoData | Sequence[MDHistoData],
    *,
    dataset_labels: Sequence[str] | None = None,
    x_dim: int | str = -1,
    waterfall_dim: int | str = 0,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    waterfall_step: float | None = None,
    coverage_threshold: float = 0.9,
    trace_offset: float | None = None,
    cmap: str = "viridis",
    color_range: tuple[float, float] = (0.0, 1.0),
    reverse_colors: bool = False,
    marker: str = "o",
    line_style: str = "none",
    marker_size: float = 5.0,
    line_width: float = 1.5,
    marker_edge_width: float = 1.5,
    marker_face: str = "none",
    show_errorbars: bool = True,
    errorbar_caps: bool = False,
    errorbar_cap_size: float = 3.0,
    show_zero_lines: bool = True,
    zero_line_color: str = "#7f7f7f",
    zero_line_style: str = "--",
    zero_line_width: float = 0.8,
    show_model: bool = False,
    unmask_model: bool = False,
    model_color: str | None = None,
    model_line_width: float = 2.0,
    show_trace_labels: bool = True,
    trace_label_suffix: str = "",
    trace_label_font_size: float = 10.0,
    trace_label_color: str | None = None,
    smoothing_sigma_x: float = 0.0,
    smoothing_sigma_waterfall: float = 0.0,
    smoothing_fill_nans: bool = True,
    masked: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 12.0,
    axes_linewidth: float = 1.5,
    figsize: tuple[float, float] = (8.0, 6.5),
    ax=None,
):
    """Render stacked, offset traces from MDHisto data."""

    import matplotlib.pyplot as plt

    datasets = [data] if isinstance(data, MDHistoData) else list(data)
    traces = prepare_mdhisto_waterfall(
        datasets,
        dataset_labels=dataset_labels,
        x_dim=x_dim,
        waterfall_dim=waterfall_dim,
        channel=channel,
        selections=selections,
        integrate_checks=integrate_checks,
        waterfall_step=waterfall_step,
        coverage_threshold=coverage_threshold,
        masked=masked,
        smoothing_sigma_x=smoothing_sigma_x,
        smoothing_sigma_waterfall=smoothing_sigma_waterfall,
        smoothing_fill_nans=smoothing_fill_nans,
        include_model=show_model,
        unmask_model=unmask_model,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = waterfall_colors(
        cmap,
        len(traces),
        low=color_range[0],
        high=color_range[1],
        reverse=reverse_colors,
    )
    offset = (
        default_waterfall_offset(traces)
        if trace_offset is None
        else float(trace_offset)
    )
    draw_waterfall_traces(
        ax,
        traces,
        colors=colors,
        trace_offset=offset,
        marker=marker,
        line_style=line_style,
        marker_size=marker_size,
        line_width=line_width,
        marker_edge_width=marker_edge_width,
        marker_face=marker_face,
        show_errorbars=show_errorbars,
        errorbar_caps=errorbar_caps,
        errorbar_cap_size=errorbar_cap_size,
        show_zero_lines=show_zero_lines,
        zero_line_color=zero_line_color,
        zero_line_style=zero_line_style,
        zero_line_width=zero_line_width,
        show_model=show_model,
        model_color=model_color,
        model_line_width=model_line_width,
        show_trace_labels=show_trace_labels,
        trace_label_suffix=trace_label_suffix,
        trace_label_font_size=trace_label_font_size,
        trace_label_color=trace_label_color,
    )
    first = datasets[0]
    x_index = _waterfall_1d_axis(first, x_dim) if _mdhisto_non_singleton_count(first) == 1 else _resolve_mdhisto_dim(first, x_dim)
    axis = first.axes[x_index]
    ax.set_xlabel(
        display_axis_label(
            axis.name,
            axis.units,
            quantity_type=axis.role,
        )
    )
    channel_model = MDHistoSliceViewer(
        first,
        x_dim=x_index,
        y_dim=_waterfall_other_axis(first, x_index),
        channel=channel,
    )
    ax.set_ylabel(channel_model._channel_label())
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.figure.set_size_inches(*figsize)
    with plt.rc_context({"font.size": float(font_size)}):
        _apply_axes_linewidth((ax,), None, axes_linewidth)
        for item in (ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()):
            item.set_fontsize(float(font_size))
    ax._nfit_waterfall_traces = traces
    ax._nfit_waterfall_offset = offset
    return ax


def draw_waterfall_traces(
    ax,
    traces: Sequence[WaterfallTrace],
    *,
    colors: Sequence[Any],
    trace_offset: float,
    marker: str,
    line_style: str,
    marker_size: float,
    line_width: float,
    marker_edge_width: float,
    marker_face: str,
    show_errorbars: bool,
    errorbar_caps: bool,
    errorbar_cap_size: float,
    show_zero_lines: bool,
    zero_line_color: str,
    zero_line_style: str,
    zero_line_width: float,
    show_model: bool,
    model_color: str | None,
    model_line_width: float,
    show_trace_labels: bool,
    trace_label_suffix: str = "",
    trace_label_font_size: float = 10.0,
    trace_label_color: str | None = None,
) -> None:
    """Draw already prepared waterfall traces on one Matplotlib axes."""

    style = "None" if str(line_style).lower() == "none" else line_style
    for index, (trace, color) in enumerate(zip(traces, colors, strict=True)):
        baseline = float(index) * float(trace_offset)
        x = np.asarray(trace.x, dtype=float)
        values = np.asarray(trace.values, dtype=float)
        shifted = values + baseline
        finite_x = x[np.isfinite(x)]
        if show_zero_lines and finite_x.size:
            ax.plot(
                [float(np.min(finite_x)), float(np.max(finite_x))],
                [baseline, baseline],
                color=zero_line_color,
                linestyle=zero_line_style,
                linewidth=float(zero_line_width),
                zorder=0.5,
            )
        if show_model and trace.model_values is not None:
            ax.plot(
                x,
                np.asarray(trace.model_values, dtype=float) + baseline,
                color=color if model_color is None else model_color,
                linestyle="-",
                marker="",
                linewidth=float(model_line_width),
                zorder=1.5,
            )
        common = {
            "marker": marker,
            "linestyle": style,
            "ms": float(marker_size),
            "lw": float(line_width),
            "mew": float(marker_edge_width),
            "mfc": (
                color
                if marker_face == "outline"
                else marker_face if marker else "none"
            ),
            "mec": color,
            "color": color,
            "zorder": 2.0,
        }
        errors = None if trace.errors is None else np.asarray(trace.errors, dtype=float)
        if (
            show_errorbars
            and errors is not None
            and errors.shape == shifted.shape
            and np.any(np.isfinite(errors))
        ):
            ax.errorbar(
                x,
                shifted,
                yerr=errors,
                ecolor=color,
                capsize=float(errorbar_cap_size) if errorbar_caps else 0.0,
                capthick=float(line_width),
                elinewidth=float(line_width),
                **common,
            )
        else:
            ax.plot(x, shifted, **common)
        if show_trace_labels and trace.label:
            valid = np.isfinite(x) & np.isfinite(shifted)
            if np.any(valid):
                first = int(np.flatnonzero(valid)[0])
                label = f"{trace.label}{trace_label_suffix}"
                if (
                    trace.waterfall_coordinate is not None
                    and trace_label_suffix
                ):
                    label = (
                        f"{trace.waterfall_coordinate:.5g}"
                        f"{trace_label_suffix}"
                    )
                ax.annotate(
                    label,
                    (float(x[first]), float(shifted[first])),
                    xytext=(4, 5),
                    textcoords="offset points",
                    color=color if trace_label_color is None else trace_label_color,
                    fontsize=float(trace_label_font_size),
                    ha="left",
                    va="bottom",
                )


def waterfall_colors(
    cmap: str,
    count: int,
    *,
    low: float = 0.0,
    high: float = 1.0,
    reverse: bool = False,
) -> list[Any]:
    """Return colors sampled uniformly from a selected colormap interval."""

    from matplotlib import colormaps

    if count <= 0:
        return []
    name = "gray" if cmap == "grey" else str(cmap)
    color_map = colormaps.get_cmap(name)
    lower, upper = sorted(
        (
            float(np.clip(low, 0.0, 1.0)),
            float(np.clip(high, 0.0, 1.0)),
        )
    )
    positions = np.linspace(lower, upper, max(int(count), 1))
    if reverse:
        positions = positions[::-1]
    return [color_map(float(position)) for position in positions]


def waterfall_absolute_max(traces: Sequence[WaterfallTrace]) -> float:
    """Return the largest finite absolute data value across prepared traces."""

    maxima = [
        float(np.nanmax(np.abs(trace.values)))
        for trace in traces
        if np.any(np.isfinite(trace.values))
    ]
    return max(maxima, default=0.0)


def default_waterfall_offset(traces: Sequence[WaterfallTrace]) -> float:
    """Return half the largest finite absolute trace intensity."""

    maximum = waterfall_absolute_max(traces)
    return 0.5 * maximum if maximum > 0.0 else 1.0


def default_waterfall_step(data: MDHistoData, waterfall_dim: int | str) -> float:
    """Choose a bin width that yields approximately ten waterfall traces."""

    index = _resolve_mdhisto_dim(data, waterfall_dim)
    centers = np.asarray(data.axes[index].centers, dtype=float)
    if centers.size == 0:
        return 1.0
    edges = _edges_from_centers(centers)
    span = float(edges[-1] - edges[0])
    target = max(min(int(centers.size), 10), 1)
    return span / target if np.isfinite(span) and span > 0.0 else 1.0


def waterfall_step_bounds(
    data: MDHistoData,
    waterfall_dim: int | str,
) -> tuple[float, float]:
    """Return native-bin and full-span limits for waterfall bin width."""

    index = _resolve_mdhisto_dim(data, waterfall_dim)
    edges = np.asarray(data.axes[index].values, dtype=float)
    finite_edges = edges[np.isfinite(edges)]
    if finite_edges.size < 2:
        return 1.0, 1.0
    widths = np.abs(np.diff(finite_edges))
    positive = widths[np.isfinite(widths) & (widths > 0.0)]
    span = float(np.max(finite_edges) - np.min(finite_edges))
    minimum = float(np.min(positive)) if positive.size else span
    maximum = span if np.isfinite(span) and span > 0.0 else minimum
    minimum = minimum if np.isfinite(minimum) and minimum > 0.0 else maximum
    return min(minimum, maximum), max(minimum, maximum)


def _coarsen_waterfall_view(
    view: dict[str, np.ndarray],
    values: np.ndarray,
    errors: np.ndarray | None,
    model_values: np.ndarray | None,
    *,
    step: float | None,
    coverage_threshold: float,
    unmask_model: bool,
    axis_name: str,
    axis_units: str,
) -> list[WaterfallTrace]:
    x = np.asarray(view["x_centers"], dtype=float)
    y = np.asarray(view["y_centers"], dtype=float)
    y_edges = np.asarray(view["y_edges"], dtype=float)
    width = float(step) if step is not None else float(y_edges[-1] - y_edges[0]) / max(min(y.size, 10), 1)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("waterfall_step must be positive")
    start = float(y_edges[0])
    stop = float(y_edges[-1])
    boundaries = start + np.arange(max(int(np.ceil((stop - start) / width)), 1) + 1) * width
    boundaries[-1] = max(boundaries[-1], stop)
    traces: list[WaterfallTrace] = []
    for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        selected = (y >= low) & ((y < high) if index < len(boundaries) - 2 else (y <= high))
        if not np.any(selected):
            continue
        selected_values = np.asarray(values[selected, :], dtype=float)
        selected_errors = None if errors is None else np.asarray(errors[selected, :], dtype=float)
        profile, uncertainty, weights = _waterfall_weighted_profile(
            selected_values,
            selected_errors,
        )
        selected_coverage = np.asarray(
            view.get("coverage_fraction", np.ones(values.shape, dtype=float))[selected, :],
            dtype=float,
        )
        widths = np.diff(y_edges)[selected]
        coverage = np.sum(
            selected_coverage * widths[:, None],
            axis=0,
        ) / np.sum(widths)
        insufficient = ~np.isfinite(coverage) | (
            coverage < float(np.clip(coverage_threshold, 0.0, 1.0))
        )
        profile = np.where(insufficient, np.nan, profile)
        if uncertainty is not None:
            uncertainty = np.where(insufficient, np.nan, uncertainty)
        model_profile = None
        if model_values is not None:
            selected_model = np.asarray(model_values[selected, :], dtype=float)
            model_profile = _waterfall_profile_with_weights(selected_model, weights)
            if not unmask_model:
                model_profile = np.where(insufficient, np.nan, model_profile)
        coordinate = float(np.nanmean(y[selected]))
        displayed_unit = display_unit(axis_units) if axis_units else ""
        unit_suffix = f" {displayed_unit}" if displayed_unit else ""
        traces.append(
            WaterfallTrace(
                x=x,
                values=profile,
                errors=uncertainty,
                model_values=model_profile,
                label=f"{coordinate:.5g}{unit_suffix}",
                waterfall_coordinate=coordinate,
                waterfall_unit=displayed_unit,
            )
        )
    return traces


def waterfall_axis_display_name(name: str) -> str:
    """Use compact scientific notation for waterfall energy labels."""

    return (
        "ΔE"
        if str(name).strip().casefold().replace("_", "") == "deltae"
        else str(name)
    )


def _waterfall_weighted_profile(
    values: np.ndarray,
    errors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if errors is None or errors.shape != values.shape:
        return _nanmean_axis0(values), None, None
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    weights = np.zeros(values.shape, dtype=float)
    np.divide(1.0, errors, out=weights, where=valid)
    np.square(weights, out=weights)
    weight_sum = np.sum(weights, axis=0)
    weighted_sum = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    profile = np.full(weight_sum.shape, np.nan, dtype=float)
    uncertainty = np.full(weight_sum.shape, np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=profile, where=weight_sum > 0.0)
    np.divide(1.0, np.sqrt(weight_sum), out=uncertainty, where=weight_sum > 0.0)
    fallback = _nanmean_axis0(values)
    profile = np.where(np.isfinite(profile), profile, fallback)
    return profile, uncertainty, weights


def _waterfall_profile_with_weights(
    values: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    if weights is None or weights.shape != values.shape:
        return _nanmean_axis0(values)
    valid = np.isfinite(values) & (weights > 0.0)
    weight_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
    weighted_sum = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    profile = np.full(weight_sum.shape, np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=profile, where=weight_sum > 0.0)
    fallback = _nanmean_axis0(values)
    return np.where(np.isfinite(profile), profile, fallback)


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    return _nanmean_axis(values, axis=0)


def _nanmean_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """Average finite values while leaving empty reductions as NaN."""

    finite = np.isfinite(values)
    counts = np.sum(finite, axis=axis)
    totals = np.sum(np.where(finite, values, 0.0), axis=axis)
    result = np.full(counts.shape, np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _waterfall_channel_errors(
    model: MDHistoSliceViewer,
    view: dict[str, np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray | None:
    key = f"{model.channel}_errors"
    values = view.get(key, view.get("errors") if model.channel == "signal" else None)
    if values is None:
        return None
    errors = np.asarray(values, dtype=float)
    return errors.reshape(shape) if errors.size == int(np.prod(shape)) else None


def _waterfall_model_values(
    view: dict[str, np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray | None:
    values = view.get("fit")
    if values is None:
        return None
    model = np.asarray(values, dtype=float)
    return model.reshape(shape) if model.size == int(np.prod(shape)) else None


def _mdhisto_non_singleton_count(data: MDHistoData) -> int:
    return sum(size > 1 for size in data.shape)


def _waterfall_1d_axis(data: MDHistoData, requested: int | str) -> int:
    non_singleton = [index for index, size in enumerate(data.shape) if size > 1]
    if len(non_singleton) != 1:
        return _resolve_mdhisto_dim(data, requested)
    if isinstance(requested, str) and requested in [axis.name for axis in data.axes]:
        candidate = _resolve_mdhisto_dim(data, requested)
        if data.shape[candidate] > 1:
            return candidate
    if isinstance(requested, int):
        candidate = int(requested) % data.signal.ndim
        if data.shape[candidate] > 1:
            return candidate
    return non_singleton[0]


def _waterfall_other_axis(data: MDHistoData, x_dim: int) -> int:
    return next((index for index in range(data.signal.ndim) if index != x_dim), x_dim)


def plot_mdhisto_auto(data: MDHistoData, **kwargs):
    """Plot a 1D MDHisto as a line, otherwise render a 2D slice figure."""

    non_singleton = [dim for dim, size in enumerate(data.shape) if size > 1]
    if len(non_singleton) == 1:
        return plot_mdhisto_line(data, axis_dim=non_singleton[0], **kwargs)
    if len(non_singleton) < 1:
        raise ValueError("MDHisto data has no plottable non-singleton dimensions")
    kwargs.setdefault("x_dim", non_singleton[-1])
    kwargs.setdefault("y_dim", non_singleton[-2])
    return plot_mdhisto_slice(data, **kwargs)


def mdhisto_with_signal_like(
    template: MDHistoData,
    signal: ArrayLike,
    *,
    errors: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    num_events: ArrayLike | None = None,
    metadata: dict | None = None,
) -> MDHistoData:
    """Return an MDHistoData object sharing axes with ``template``."""

    signal_arr = np.asarray(signal, dtype=float)
    if signal_arr.shape != template.signal.shape:
        raise ValueError(
            f"signal shape {signal_arr.shape} does not match template shape "
            f"{template.signal.shape}"
        )
    errors_arr = np.zeros_like(signal_arr) if errors is None else np.asarray(errors, dtype=float)
    mask_arr = np.asarray(template.mask if mask is None else mask, dtype=bool)
    events_arr = np.asarray(template.num_events if num_events is None else num_events, dtype=float)
    return MDHistoData(
        axes=template.axes,
        signal=signal_arr,
        errors=errors_arr,
        mask=mask_arr,
        num_events=events_arr,
        coordinate_system=template.coordinate_system,
        visual_normalization=template.visual_normalization,
        metadata={**template.metadata, **({} if metadata is None else metadata)},
    )


def residual_mdhisto(data: MDHistoData, fit: MDHistoData) -> MDHistoData:
    """Return ``(data - fit) / error`` as an MDHistoData object."""

    if data.signal.shape != fit.signal.shape:
        raise ValueError("data and fit shapes must match")
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = (
            np.asarray(data.signal, dtype=float)
            - np.asarray(fit.signal, dtype=float)
        ) / np.asarray(data.errors, dtype=float)
    mask = np.asarray(data.mask, dtype=bool) | ~np.isfinite(residual)
    return mdhisto_with_signal_like(
        data,
        residual,
        errors=np.ones_like(residual),
        mask=mask,
        metadata={"channel": "normalized residual"},
    )


def plot_mdhisto_fit_comparison(
    data: MDHistoData,
    fit: MDHistoData,
    *,
    residual: MDHistoData | None = None,
    show_residual: bool = True,
    x_dim: int | str = -1,
    y_dim: int | str = 0,
    channel: str = "signal",
    selections: dict[int, tuple[float, float]] | None = None,
    integrate_checks: dict[int, bool] | None = None,
    cmap: str = "viridis",
    color_scale: str = "linear",
    auto_limits: str = "min/max",
    figsize: tuple[float, float] | None = None,
):
    """Plot data, fit, and optionally residual for matching MDHisto datasets."""

    non_singleton = [dim for dim, size in enumerate(data.shape) if size > 1]
    if len(non_singleton) == 1:
        return plot_mdhisto_fit_line_comparison(
            data,
            fit,
            residual=residual,
            show_residual=show_residual,
            axis_dim=non_singleton[0],
            channel=channel,
            figsize=(8.0, 5.5) if figsize is None else figsize,
        )
    return _plot_mdhisto_fit_slice_comparison(
        data,
        fit,
        residual=residual,
        show_residual=show_residual,
        x_dim=x_dim,
        y_dim=y_dim,
        channel=channel,
        selections=selections,
        integrate_checks=integrate_checks,
        cmap=cmap,
        color_scale=color_scale,
        auto_limits=auto_limits,
        figsize=figsize,
    )


def plot_mdhisto_fit_line_comparison(
    data: MDHistoData,
    fit: MDHistoData,
    *,
    residual: MDHistoData | None = None,
    show_residual: bool = True,
    axis_dim: int | str | None = None,
    channel: str = "signal",
    ax=None,
    figsize: tuple[float, float] = (8.0, 5.5),
):
    """Overlay 1D data, fit, and vertically offset residual."""

    import matplotlib.pyplot as plt

    if axis_dim is None:
        non_singleton = [dim for dim, size in enumerate(data.shape) if size > 1]
        if len(non_singleton) != 1:
            raise ValueError("axis_dim is required unless exactly one dimension is non-singleton")
        axis_index = non_singleton[0]
    else:
        axis_index = _resolve_mdhisto_dim(data, axis_dim)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    x, y, yerr = _mdhisto_1d_values(data, axis_index, channel)
    _, yfit, _ = _mdhisto_1d_values(fit, axis_index, channel)
    ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle="none", ms=5.0, mfc="none", label="data")
    ax.plot(x, yfit, "-", lw=1.5, label="fit")
    if show_residual:
        if residual is None:
            residual = residual_mdhisto(data, fit)
        _, r, _ = _mdhisto_1d_values(residual, axis_index, "signal")
        offset = _residual_offset(y, yfit)
        ax.axhline(offset, color="0.7", lw=0.8)
        ax.plot(x, r + offset, "o", ms=4.0, mfc="none", color="0.25", label="residual")
    axis = data.axes[axis_index]
    ax.set_xlabel(
        display_axis_label(
            axis.name,
            axis.units,
            quantity_type=axis.role,
        )
    )
    channel_name = _resolve_mdhisto_channel(channel, data)
    channel_model = MDHistoSliceViewer(
        data,
        x_dim=axis_index,
        y_dim=_waterfall_other_axis(data, axis_index),
        channel=channel_name,
    )
    ax.set_ylabel(channel_model._channel_label())
    ax.legend()
    return ax


def _plot_mdhisto_fit_slice_comparison(
    data: MDHistoData,
    fit: MDHistoData,
    *,
    residual: MDHistoData | None,
    show_residual: bool,
    x_dim: int | str,
    y_dim: int | str,
    channel: str,
    selections: dict[int, tuple[float, float]] | None,
    integrate_checks: dict[int, bool] | None,
    cmap: str,
    color_scale: str,
    auto_limits: str,
    figsize: tuple[float, float] | None,
):
    import matplotlib.pyplot as plt

    panels: list[tuple[str, MDHistoData, str]] = [("Data", data, channel), ("Fit", fit, channel)]
    if show_residual:
        panels.append(
            (
                "Residual",
                residual_mdhisto(data, fit) if residual is None else residual,
                "signal",
            )
        )
    fig_width = 5.0 * len(panels) if figsize is None else figsize[0]
    fig_height = 4.8 if figsize is None else figsize[1]
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.atleast_1d(axes)

    data_model = _configured_mdhisto_model(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        channel=channel,
        selections=selections,
        integrate_checks=integrate_checks,
        cmap=cmap,
        color_scale=color_scale,
        auto_limits=auto_limits,
    )
    data_view = data_model.slice_arrays()
    data_values = data_model._display_values(data_view)
    shared_norm = data_model._color_norm(data_values)

    for ax, (title, panel_data, panel_channel) in zip(axes, panels, strict=True):
        model = _configured_mdhisto_model(
            panel_data,
            x_dim=x_dim,
            y_dim=y_dim,
            channel=panel_channel,
            selections=selections,
            integrate_checks=integrate_checks,
            cmap=cmap,
            color_scale=color_scale,
            auto_limits=auto_limits,
        )
        view = model.slice_arrays()
        values = model._display_values(view)
        norm = None if title == "Residual" else shared_norm
        artist = ax.pcolormesh(
            view["x_edges"],
            view["y_edges"],
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel(model._axis_label(model.x_dim))
        ax.set_ylabel(model._axis_label(model.y_dim))
        colorbar = fig.colorbar(artist, ax=ax)
        colorbar.set_label("Residual (sigma)" if title == "Residual" else model._channel_label())
    return fig


def _configured_mdhisto_model(
    data: MDHistoData,
    *,
    x_dim: int | str,
    y_dim: int | str,
    channel: str,
    selections: dict[int, tuple[float, float]] | None,
    integrate_checks: dict[int, bool] | None,
    cmap: str,
    color_scale: str,
    auto_limits: str,
) -> MDHistoSliceViewer:
    model = MDHistoSliceViewer(
        data,
        x_dim=x_dim,
        y_dim=y_dim,
        channel=channel,
        cmap=cmap,
        color_scale=color_scale,
        auto_limits=auto_limits,
    )
    if selections:
        model.selections.update({int(dim): tuple(value) for dim, value in selections.items()})
    if integrate_checks:
        model.integrate_checks.update({int(dim): bool(value) for dim, value in integrate_checks.items()})
    return model


def _mdhisto_1d_values(
    data: MDHistoData,
    axis_index: int,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    channel_name = _resolve_mdhisto_channel(channel, data)
    values = _mdhisto_channel_array(data, channel_name)
    index = [0] * data.signal.ndim
    index[axis_index] = slice(None)
    y = np.asarray(values[tuple(index)], dtype=float)
    yerr = None
    if channel_name == "signal":
        yerr = np.asarray(_mdhisto_channel_array(data, "errors")[tuple(index)], dtype=float)
    elif (
        channel_name in data.auxiliary_channels
        and data.auxiliary_channels[channel_name].errors is not None
    ):
        yerr = np.asarray(
            data.auxiliary_channels[channel_name].errors[tuple(index)], dtype=float
        )
    return data.axes[axis_index].centers, y, yerr


def _residual_offset(data_values: np.ndarray, fit_values: np.ndarray) -> float:
    finite = np.asarray(np.concatenate([np.ravel(data_values), np.ravel(fit_values)]), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0
    span = float(np.nanmax(finite) - np.nanmin(finite))
    span = span if span > 0.0 else max(abs(float(np.nanmean(finite))), 1.0)
    return float(np.nanmin(finite) - 0.25 * span)


def _resolve_mdhisto_dim(data: MDHistoData, dim: int | str) -> int:
    if isinstance(dim, int):
        return dim % data.signal.ndim
    names = [axis.name for axis in data.axes]
    if dim not in names:
        raise ValueError(f"unknown axis {dim!r}; choose one of {names}")
    return names.index(dim)


def _resolve_mdhisto_channel(
    channel: str, data: MDHistoData | None = None
) -> str:
    normalized = MDHistoSliceViewer.CHANNEL_ALIASES.get(str(channel), str(channel))
    choices = (
        (*MDHistoSliceViewer.CHANNELS, *data.auxiliary_channels)
        if data is not None
        else MDHistoSliceViewer.CHANNELS
    )
    if normalized not in choices:
        raise ValueError(f"unknown channel {channel!r}; choose one of {choices}")
    return normalized


def _mdhisto_channel_array(data: MDHistoData, channel: str) -> np.ndarray:
    if channel == "signal":
        values = np.asarray(data.signal, dtype=float)
    elif channel == "errors":
        values = np.asarray(data.errors, dtype=float)
    elif channel == "num_events":
        values = np.asarray(data.num_events, dtype=float)
    elif channel == "combined_mask":
        return np.asarray(data.mask, dtype=float)
    elif channel in {"file_mask", "nfit_mask"}:
        return np.asarray(data.metadata.get(channel, np.zeros(data.shape, dtype=bool)), dtype=float)
    elif channel in data.auxiliary_channels:
        values = np.asarray(data.auxiliary_channels[channel].values, dtype=float)
    else:
        raise ValueError(f"unknown channel {channel!r}")
    empty = ~mdhisto_measured_bins(data)
    return np.where(np.asarray(data.mask, dtype=bool) | empty, np.nan, values)


def _panel_ratio(percent: float) -> float:
    fraction = float(np.clip(percent, 1.0, 80.0)) / 100.0
    return fraction / (1.0 - fraction)


def inverse_variance_weighted_profile(
    values: np.ndarray,
    errors: np.ndarray,
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an inverse-variance weighted profile and its standard error.

    Non-finite values, non-finite uncertainties, and non-positive
    uncertainties do not contribute to the profile.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    weights = np.zeros(values.shape, dtype=float)
    np.divide(1.0, errors, out=weights, where=valid)
    np.square(weights, out=weights)
    weight_sum = np.sum(weights, axis=axis)
    weighted_sum = np.sum(np.where(valid, values * weights, 0.0), axis=axis)
    mean = np.full(weight_sum.shape, np.nan, dtype=float)
    uncertainty = np.full(weight_sum.shape, np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=mean, where=weight_sum > 0.0)
    np.divide(1.0, np.sqrt(weight_sum), out=uncertainty, where=weight_sum > 0.0)
    return mean, uncertainty


def _draw_mdhisto_roi_cuts(
    model: MDHistoSliceViewer,
    view: dict[str, np.ndarray],
    roi_extents: tuple[float, float, float, float],
    ax_xcut,
    ax_ycut,
) -> None:
    x0, x1, y0, y1 = roi_extents
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x_mask = (view["x_centers"] >= x0) & (view["x_centers"] <= x1)
    y_mask = (view["y_centers"] >= y0) & (view["y_centers"] <= y1)
    if np.any(x_mask) and np.any(y_mask):
        z = model._display_values(view)
        errors = np.asarray(view["errors"], dtype=float)
        selected = np.ix_(y_mask, x_mask)
        x_cut, x_error = inverse_variance_weighted_profile(
            z[selected], errors[selected], axis=0
        )
        y_cut, y_error = inverse_variance_weighted_profile(
            z[selected], errors[selected], axis=1
        )
        coverage = np.asarray(view["coverage_fraction"], dtype=float)[selected]
        x_widths = np.diff(np.asarray(view["x_edges"], dtype=float))[x_mask]
        y_widths = np.diff(np.asarray(view["y_edges"], dtype=float))[y_mask]
        x_coverage = np.sum(coverage * y_widths[:, None], axis=0) / np.sum(
            y_widths
        )
        y_coverage = np.sum(coverage * x_widths[None, :], axis=1) / np.sum(
            x_widths
        )
        x_insufficient = x_coverage < model.coverage_threshold
        y_insufficient = y_coverage < model.coverage_threshold
        x_cut = np.where(x_insufficient, np.nan, x_cut)
        x_error = np.where(x_insufficient, np.nan, x_error)
        y_cut = np.where(y_insufficient, np.nan, y_cut)
        y_error = np.where(y_insufficient, np.nan, y_error)
        ax_xcut.errorbar(
            view["x_centers"][x_mask], x_cut, yerr=x_error, fmt="-", lw=1.2, capsize=0
        )
        ax_ycut.errorbar(
            y_cut, view["y_centers"][y_mask], xerr=y_error, fmt="-", lw=1.2, capsize=0
        )
    ax_xcut.set_ylabel("Weighted mean")
    ax_xcut.set_xlabel(model._axis_label(model.x_dim))
    ax_ycut.set_xlabel("Weighted mean")
    ax_ycut.set_ylabel(model._axis_label(model.y_dim))


def _apply_axes_linewidth(axes, colorbar, linewidth: float) -> None:
    width = float(linewidth)
    for axis in axes:
        if axis is None:
            continue
        for spine in axis.spines.values():
            spine.set_linewidth(width)
        is_colorbar_axis = colorbar is not None and axis is colorbar.ax
        axis.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=not is_colorbar_axis,
            right=not is_colorbar_axis,
            width=width,
        )
    if colorbar is not None:
        colorbar.outline.set_linewidth(width)
        colorbar.ax.tick_params(which="both", direction="in", width=width)


class _DropdownSelect:
    """Small Matplotlib dropdown built from buttons."""

    def __init__(self, fig, rect, label: str, options, value: str, callback) -> None:
        from matplotlib.widgets import Button

        self.fig = fig
        self.label = label
        self.options = tuple(options)
        self.value = value if value in self.options else self.options[0]
        self.callback = callback
        self.button_ax = fig.add_axes(rect)
        self.button_ax.set_zorder(20)
        self.button = Button(self.button_ax, self._button_text())
        self.button.on_clicked(lambda _event: self.toggle())
        self.option_axes = []
        self.option_buttons = []
        x0, y0, width, height = rect
        option_height = min(height, 0.032)
        for index, option in enumerate(self.options):
            axis = fig.add_axes([x0, y0 - (index + 1) * option_height, width, option_height])
            axis.set_zorder(40)
            button = Button(axis, option)
            button.on_clicked(lambda _event, choice=option: self.select(choice))
            axis.set_visible(False)
            self.option_axes.append(axis)
            self.option_buttons.append(button)

    def toggle(self) -> None:
        visible = not self.option_axes[0].get_visible()
        for axis in self.option_axes:
            axis.set_visible(visible)
        self.fig.canvas.draw_idle()

    def select(self, value: str) -> None:
        self.value = value
        self.button.label.set_text(self._button_text())
        for axis in self.option_axes:
            axis.set_visible(False)
        self.callback(value)
        self.fig.canvas.draw_idle()

    def _button_text(self) -> str:
        return f"{self.label}: {self.value} v"


class MDHistoSliceViewer:
    """Interactive Matplotlib slice viewer for binned MD histogram data.

    The viewer is intentionally modeled after the Mantid Workbench slice viewer:
    pick two displayed axes, choose point or range integration for the remaining
    axes, tune the colormap and normalization, and drag a rectangle on the image
    to generate integrated x/y cuts.
    """

    COLOR_SCALES = ("linear", "log", "symmetriclog", "asinh", "power")
    AUTO_LIMITS = ("min/max", "N-sigma", "N IQR", "Nth percentile")
    COLORMAPS = IMAGE_COLORMAPS
    CHANNELS = (
        "signal",
        "errors",
        "num_events",
        "coverage_fraction",
        "coverage_mask",
        "combined_mask",
        "file_mask",
        "nfit_mask",
    )
    CHANNEL_ALIASES = {
        "multiplicity": "num_events",
        "events": "num_events",
        "error": "errors",
        "mask": "combined_mask",
        "total_mask": "combined_mask",
    }
    CHANNEL_LABELS = {
        "signal": "Signal",
        "errors": "Error",
        "num_events": "Multiplicity",
        "coverage_fraction": "Coverage",
        "coverage_mask": "Coverage mask",
        "combined_mask": "Combined mask",
        "file_mask": "File mask",
        "nfit_mask": "Nfit mask",
    }

    def __init__(
        self,
        data: MDHistoData,
        *,
        x_dim: int | str = -1,
        y_dim: int | str = 0,
        channel: str = "signal",
        cmap: str = "viridis",
        color_scale: str = "linear",
        auto_limits: str = "min/max",
        integrate: bool = False,
        masked: bool = True,
        coverage_threshold: float = 0.9,
    ) -> None:
        self.data = data
        self.is_point_list = isinstance(data, PointListData)
        if self.is_point_list:
            coordinates = list(data.coordinate_names) or list(data.column_names)
            hidden_channels = set(data.metadata.get("viewer_hidden_channels", []))
            channels = [
                label for label in data.channel_labels if label not in hidden_channels
            ] or [
                name for name in data.column_names if name not in coordinates
            ] or list(data.column_names)
            # x-axis choices are the coordinates plus any other non-channel column
            # (e.g. derived q/d/2theta), coordinates first.
            channel_columns = set()
            for channel_def in data.channels:
                channel_columns.add(channel_def.get("value"))
                if channel_def.get("error"):
                    channel_columns.add(channel_def.get("error"))
            extra_columns = [
                name
                for name in data.column_names
                if name not in coordinates and name not in channel_columns
            ]
            self.point_coordinates = coordinates + extra_columns
            self.point_channels = channels
            self.x_key = coordinates[0]
            self.x_dim = 0
            self.y_dim = 0
            self.channel = channel if channel in channels else channels[0]
        else:
            if data.signal.ndim < 2:
                raise ValueError("slice viewer requires data with at least two dimensions")
            self.x_dim = self._resolve_dim(x_dim)
            self.y_dim = self._resolve_dim(y_dim)
            if self.x_dim == self.y_dim:
                raise ValueError("x_dim and y_dim must be different")
            self.refresh_metadata_channels()
            self.channel = self._resolve_channel(channel)
        self.cmap = str(cmap)
        self.cmap_reversed = False
        if self.cmap.endswith("_r"):
            self.cmap = self.cmap[:-2]
            self.cmap_reversed = True
        if self.cmap == "gray":
            self.cmap = "grey"
        self.color_scale = color_scale
        self.auto_limits = auto_limits
        self.power_gamma = 0.5
        self.sigma_n = 3.0
        self.iqr_n = 1.5
        self.percentile_n = 1.0
        self.autoscale = True
        self.manual_vmin: float | None = None
        self.manual_vmax: float | None = None
        self.integrate = integrate
        self.masked = masked
        self.coverage_threshold = float(np.clip(coverage_threshold, 0.0, 1.0))
        self.selections = self._default_selections()

        self.fig = None
        self.ax_image = None
        self.ax_xcut = None
        self.ax_ycut = None
        self.ax_colorbar = None
        self.image = None
        self.colorbar = None
        self.cursor_text = None
        self.slider_axes = []
        self.sliders = {}
        self.integrate_checks = {}
        self.radio_axes = []
        self.widgets = []
        self.hidden_widgets = []
        self.dropdowns = []
        self.x_radio = None
        self.y_radio = None
        self._syncing_axis_radios = False
        self.rectangle_selector = None
        self._current_slice: dict[str, np.ndarray] | None = None

    def show(self):
        """Create the interactive figure and return it."""

        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons, RectangleSelector, TextBox

        self.fig = plt.figure(figsize=(14, 9))
        self.ax_image = self.fig.add_axes([0.08, 0.24, 0.54, 0.60])
        self.ax_xcut = self.fig.add_axes([0.08, 0.08, 0.54, 0.12], sharex=self.ax_image)
        self.ax_ycut = self.fig.add_axes([0.65, 0.24, 0.08, 0.60], sharey=self.ax_image)
        self.ax_colorbar = self.fig.add_axes([0.745, 0.24, 0.018, 0.60])

        labels = [axis.name for axis in self.data.axes]
        x_radio_ax = self.fig.add_axes([0.80, 0.72, 0.08, 0.18])
        y_radio_ax = self.fig.add_axes([0.90, 0.72, 0.08, 0.18])
        x_radio = RadioButtons(x_radio_ax, labels, active=self.x_dim)
        y_radio = RadioButtons(y_radio_ax, labels, active=self.y_dim)
        self.x_radio = x_radio
        self.y_radio = y_radio
        x_radio_ax.set_title("x")
        y_radio_ax.set_title("y")
        x_radio.on_clicked(lambda label: self._set_display_dim("x", labels.index(label)))
        y_radio.on_clicked(lambda label: self._set_display_dim("y", labels.index(label)))
        self.radio_axes.extend([x_radio_ax, y_radio_ax])
        self.widgets.extend([x_radio, y_radio])

        cmap_dropdown = _DropdownSelect(
            self.fig,
            [0.80, 0.62, 0.18, 0.035],
            "Colormap",
            self.COLORMAPS,
            self.cmap,
            self._set_cmap,
        )
        scale_dropdown = _DropdownSelect(
            self.fig,
            [0.80, 0.54, 0.18, 0.035],
            "Scale",
            self.COLOR_SCALES,
            self.color_scale,
            self._set_color_scale,
        )
        limits_dropdown = _DropdownSelect(
            self.fig,
            [0.80, 0.46, 0.18, 0.035],
            "Limits",
            self.AUTO_LIMITS,
            self.auto_limits,
            self._set_auto_limits,
        )
        self.dropdowns.extend([cmap_dropdown, scale_dropdown, limits_dropdown])
        self.widgets.extend(self.dropdowns)

        check_ax = self.fig.add_axes([0.80, 0.36, 0.18, 0.04])
        checks = CheckButtons(check_ax, ["Autoscale"], [self.autoscale])
        checks.on_clicked(lambda _label: self._toggle_autoscale())
        self.radio_axes.append(check_ax)
        self.widgets.append(checks)

        vmin_ax = self.fig.add_axes([0.80, 0.29, 0.08, 0.04])
        vmax_ax = self.fig.add_axes([0.90, 0.29, 0.08, 0.04])
        vmin_box = TextBox(vmin_ax, "vmin")
        vmax_box = TextBox(vmax_ax, "vmax")
        vmin_box.on_submit(lambda text: self._set_manual_limit("vmin", text))
        vmax_box.on_submit(lambda text: self._set_manual_limit("vmax", text))
        self.radio_axes.extend([vmin_ax, vmax_ax])
        self.widgets.extend([vmin_box, vmax_box])

        self.cursor_text = self.fig.text(0.80, 0.16, "x: -\ny: -\nI: -\nerr: -", fontsize=9)
        self._build_hidden_axis_controls()
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
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.update()
        return self.fig

    def slice_arrays(self) -> dict[str, np.ndarray]:
        """Return the current 2D slice and associated axes/errors/events."""

        if getattr(self, "is_point_list", False):
            return self._point_slice_arrays()
        selections = self._normalized_selections()
        signal, variance, events, mask, coverage, coverage_mask = self._reduce_arrays(
            selections
        )

        remaining = [dim for dim in range(self.data.signal.ndim) if dim in (self.x_dim, self.y_dim)]
        y_pos = remaining.index(self.y_dim)
        x_pos = remaining.index(self.x_dim)
        signal2d = np.moveaxis(signal, (y_pos, x_pos), (0, 1))
        variance2d = np.moveaxis(variance, (y_pos, x_pos), (0, 1))
        events2d = np.moveaxis(events, (y_pos, x_pos), (0, 1))
        mask2d = np.moveaxis(mask, (y_pos, x_pos), (0, 1))
        if self.masked:
            signal2d = np.where(mask2d, np.nan, signal2d)
            variance2d = np.where(mask2d, np.nan, variance2d)

        view = {
            "x_edges": self._axis_edges(self.x_dim),
            "y_edges": self._axis_edges(self.y_dim),
            "x_centers": self.data.axes[self.x_dim].centers,
            "y_centers": self.data.axes[self.y_dim].centers,
            "signal": signal2d,
            "errors": np.sqrt(variance2d),
            "num_events": events2d,
            "combined_mask": mask2d,
            "mask": mask2d,
            "file_mask": self._slice_metadata_mask("file_mask", selections),
            "nfit_mask": self._slice_metadata_mask("nfit_mask", selections),
            "coverage_fraction": np.moveaxis(
                coverage, (y_pos, x_pos), (0, 1)
            ),
            "coverage_mask": np.moveaxis(
                coverage_mask, (y_pos, x_pos), (0, 1)
            ),
        }
        for name in self._metadata_channel_names():
            if name == "coverage_fraction":
                continue
            values2d = self._slice_metadata_channel(name, selections)
            if self.masked:
                values2d = np.where(mask2d, np.nan, values2d)
            view[name] = values2d
            errors2d = self._slice_auxiliary_channel_errors(name, selections)
            if errors2d is not None:
                if self.masked:
                    errors2d = np.where(mask2d, np.nan, errors2d)
                view[f"{name}_errors"] = errors2d
        return view

    def update(self) -> None:
        """Redraw the image using the current selections and color settings."""

        if self.fig is None or self.ax_image is None:
            return
        self._current_slice = self.slice_arrays()
        view = self._current_slice
        self.ax_image.clear()
        self.ax_xcut.clear()
        self.ax_ycut.clear()
        values = self._display_values(view)
        norm = self._color_norm(values)
        self.image = self.ax_image.pcolormesh(
            view["x_edges"],
            view["y_edges"],
            values,
            shading="auto",
            cmap=self._effective_cmap(),
            norm=norm,
        )
        self.ax_image.set_xlabel(self._axis_label(self.x_dim))
        self.ax_image.set_ylabel(self._axis_label(self.y_dim))
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.image, cax=self.ax_colorbar)
            self.colorbar.set_label(self._channel_label())
        else:
            self.colorbar.update_normal(self.image)
            self.colorbar.set_label(self._channel_label())
        self.fig.canvas.draw_idle()

    def _reduce_arrays(self, selections: dict[int, tuple[int, int] | int]):
        index = []
        reduce_axes = []
        integrated_dims: list[tuple[int, int, int, int]] = []
        output_axis = 0
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                index.append(slice(None))
                output_axis += 1
            else:
                selection = selections[dim]
                if isinstance(selection, tuple):
                    start, stop = selection
                    index.append(slice(start, stop + 1))
                    reduce_axes.append(output_axis)
                    integrated_dims.append((output_axis, dim, start, stop))
                    output_axis += 1
                else:
                    index.append(selection)

        signal = np.asarray(self.data.signal[tuple(index)], dtype=float)
        variance = np.asarray(self.data.errors[tuple(index)], dtype=float) ** 2
        events = np.asarray(self.data.num_events[tuple(index)], dtype=float)
        mask = np.asarray(self.data.mask[tuple(index)], dtype=bool)
        coverage = np.asarray(
            mdhisto_coverage_fraction(self.data)[tuple(index)],
            dtype=float,
        )
        if self.masked:
            signal = np.where(mask, np.nan, signal)
            variance = np.where(mask, np.nan, variance)
            events = np.where(mask, 0.0, events)
            coverage = np.where(mask, 0.0, coverage)
        coverage_weight = np.ones(coverage.shape, dtype=float)
        for axis, dim, start, stop in integrated_dims:
            widths = np.diff(self._axis_edges(dim))[start : stop + 1]
            shape = [1] * coverage.ndim
            shape[axis] = widths.size
            coverage_weight *= widths.reshape(shape)
        coverage_numerator = coverage * coverage_weight
        coverage_denominator = coverage_weight
        for axis in sorted(reduce_axes, reverse=True):
            signal = np.nansum(signal, axis=axis)
            variance = np.nansum(variance, axis=axis)
            events = np.nansum(events, axis=axis)
            mask = np.all(mask, axis=axis)
            coverage_numerator = np.sum(coverage_numerator, axis=axis)
            coverage_denominator = np.sum(coverage_denominator, axis=axis)
            signal, variance, mask = self._blank_empty_bins(signal, variance, events, mask)
        coverage = np.zeros(np.asarray(coverage_numerator).shape, dtype=float)
        np.divide(
            coverage_numerator,
            coverage_denominator,
            out=coverage,
            where=coverage_denominator > 0.0,
        )
        coverage = np.clip(coverage, 0.0, 1.0)
        coverage_mask = coverage < self.coverage_threshold
        mask = np.asarray(mask, dtype=bool) | coverage_mask
        signal = np.where(coverage_mask, np.nan, signal)
        variance = np.where(coverage_mask, np.nan, variance)
        signal, variance, mask = self._blank_empty_bins(signal, variance, events, mask)
        return signal, variance, events, mask, coverage, coverage_mask

    def _metadata_channel_names(self) -> tuple[str, ...]:
        """Return grid-shaped float channels stored in metadata (fit results)."""

        if getattr(self, "is_point_list", False):
            return ()
        names = []
        for name in ("fit", "residual"):
            value = self.data.metadata.get(name)
            if isinstance(value, np.ndarray) and value.shape == self.data.shape:
                names.append(name)
        names.extend(
            name
            for name in self.data.auxiliary_channels
            if name not in type(self).CHANNELS
        )
        return tuple(names)

    def refresh_metadata_channels(self) -> None:
        """Refresh instance channel lists after MDHisto metadata changes."""

        if getattr(self, "is_point_list", False):
            return
        extra_channels = self._metadata_channel_names()
        self.CHANNELS = (*type(self).CHANNELS, *extra_channels)
        self.CHANNEL_LABELS = {
            **type(self).CHANNEL_LABELS,
            **(
                {"signal": str(self.data.metadata["signal_label"])}
                if self.data.metadata.get("signal_label")
                else {}
            ),
            **(
                {
                    "fit": "Fit",
                    "residual": "Residual (sigma)",
                }
                if extra_channels
                else {}
            ),
            **{
                name: (
                    f"{channel.label} ({channel.unit})"
                    if channel.label and channel.unit
                    else channel.label or name
                )
                for name, channel in self.data.auxiliary_channels.items()
            },
        }
        if hasattr(self, "channel") and self.channel not in self.CHANNELS:
            self.channel = self._resolve_channel("signal")

    def point_overlay_channel(self, name: str) -> str | None:
        """Return the mapped point-list overlay channel for the selected data channel."""

        if not getattr(self, "is_point_list", False):
            return None
        mapping = self.data.metadata.get(f"viewer_{name}_channel_map", {})
        if isinstance(mapping, dict) and mapping:
            label = mapping.get(self.channel)
            return str(label) if label in self.data.channel_labels else None
        return name if name in self.data.channel_labels else None

    def _slice_metadata_channel(
        self, name: str, selections: dict[int, tuple[int, int] | int]
    ) -> np.ndarray:
        """Slice a grid-shaped metadata channel like the signal channel.

        Integrated axes are reduced with ``nansum`` to match how the signal
        channel accumulates over an integration range.
        """

        auxiliary = self.data.auxiliary_channels.get(name)
        values = np.asarray(
            auxiliary.values if auxiliary is not None else self.data.metadata[name],
            dtype=float,
        )
        index: list[Any] = []
        reduce_axes = []
        output_axis = 0
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                index.append(slice(None))
                output_axis += 1
            else:
                selection = selections[dim]
                if isinstance(selection, tuple):
                    start, stop = selection
                    index.append(slice(start, stop + 1))
                    reduce_axes.append(output_axis)
                    output_axis += 1
                else:
                    index.append(selection)
        out = values[tuple(index)]
        for axis in sorted(reduce_axes, reverse=True):
            out = (
                _nanmean_axis(out, axis=axis)
                if auxiliary is not None
                else np.nansum(out, axis=axis)
            )
        remaining = [dim for dim in range(self.data.signal.ndim) if dim in (self.x_dim, self.y_dim)]
        y_pos = remaining.index(self.y_dim)
        x_pos = remaining.index(self.x_dim)
        return np.moveaxis(out, (y_pos, x_pos), (0, 1))

    def _slice_auxiliary_channel_errors(
        self,
        name: str,
        selections: dict[int, tuple[int, int] | int],
    ) -> np.ndarray | None:
        """Slice and propagate an auxiliary channel's one-sigma errors."""

        auxiliary = self.data.auxiliary_channels.get(name)
        if auxiliary is None or auxiliary.errors is None:
            return None
        errors = np.asarray(auxiliary.errors, dtype=float)
        index: list[Any] = []
        reduce_axes = []
        output_axis = 0
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                index.append(slice(None))
                output_axis += 1
            else:
                selection = selections[dim]
                if isinstance(selection, tuple):
                    start, stop = selection
                    index.append(slice(start, stop + 1))
                    reduce_axes.append(output_axis)
                    output_axis += 1
                else:
                    index.append(selection)
        out = errors[tuple(index)]
        for axis in sorted(reduce_axes, reverse=True):
            count = np.sum(np.isfinite(out), axis=axis)
            variance = np.nansum(np.square(out), axis=axis)
            out = np.full(variance.shape, np.nan, dtype=float)
            np.divide(np.sqrt(variance), count, out=out, where=count > 0)
        remaining = [
            dim
            for dim in range(self.data.signal.ndim)
            if dim in (self.x_dim, self.y_dim)
        ]
        y_pos = remaining.index(self.y_dim)
        x_pos = remaining.index(self.x_dim)
        return np.moveaxis(out, (y_pos, x_pos), (0, 1))

    def _slice_metadata_mask(self, name: str, selections: dict[int, tuple[int, int] | int]) -> np.ndarray:
        mask = np.asarray(self.data.metadata.get(name, np.zeros(self.data.shape, dtype=bool)), dtype=bool)
        if mask.shape != self.data.shape:
            mask = np.zeros(self.data.shape, dtype=bool)
        index = []
        reduce_axes = []
        output_axis = 0
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                index.append(slice(None))
                output_axis += 1
            else:
                selection = selections[dim]
                if isinstance(selection, tuple):
                    start, stop = selection
                    index.append(slice(start, stop + 1))
                    reduce_axes.append(output_axis)
                    output_axis += 1
                else:
                    index.append(selection)
        out = mask[tuple(index)]
        for axis in sorted(reduce_axes, reverse=True):
            out = np.any(out, axis=axis)
        remaining = [dim for dim in range(self.data.signal.ndim) if dim in (self.x_dim, self.y_dim)]
        y_pos = remaining.index(self.y_dim)
        x_pos = remaining.index(self.x_dim)
        return np.moveaxis(out, (y_pos, x_pos), (0, 1))

    def _blank_empty_bins(
        self,
        signal: np.ndarray,
        variance: np.ndarray,
        events: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = (
            np.zeros(np.asarray(events).shape, dtype=bool)
            if bool(self.data.metadata.get("zero_event_bins_are_measured", False))
            else np.asarray(events <= 0.0)
        )
        if not np.any(empty):
            return signal, variance, mask
        return np.where(empty, np.nan, signal), np.where(empty, np.nan, variance), np.asarray(mask) | empty

    def _normalized_selections(self) -> dict[int, tuple[int, int] | int]:
        selections: dict[int, tuple[int, int] | int] = {}
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                continue
            raw = self.selections.get(dim)
            centers = self.data.axes[dim].centers
            if raw is None:
                selections[dim] = int(centers.size // 2)
            elif isinstance(raw, tuple):
                low, high = raw
                start = self._nearest_center_index(dim, min(low, high))
                stop = self._nearest_center_index(dim, max(low, high))
                if self.integrate_checks.get(dim, self.integrate):
                    selections[dim] = (start, stop)
                else:
                    selections[dim] = self._nearest_center_index(dim, 0.5 * (low + high))
            else:
                selections[dim] = int(np.clip(raw, 0, centers.size - 1))
        return selections

    def _default_selections(self) -> dict[int, tuple[float, float]]:
        if getattr(self, "is_point_list", False):
            return {}
        selections = {}
        for dim, axis in enumerate(self.data.axes):
            centers = axis.centers
            mid = float(centers[centers.size // 2])
            selections[dim] = (mid, mid)
        return selections

    def _axis_edges(self, dim: int) -> np.ndarray:
        axis = self.data.axes[dim]
        if axis.values.size == self.data.shape[dim] + 1:
            return axis.values
        centers = axis.values
        if centers.size == 1:
            return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
        deltas = np.diff(centers)
        edges = np.empty(centers.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * deltas[0]
        edges[-1] = centers[-1] + 0.5 * deltas[-1]
        return edges

    def _axis_label(self, dim: int) -> str:
        if getattr(self, "is_point_list", False):
            unit = self.data.unit(self.x_key)
            return display_axis_label(
                self.x_key,
                unit,
                quantity_type=self.data.quantity_type(self.x_key),
            )
        axis = self.data.axes[dim]
        return display_axis_label(
            axis.name,
            axis.units,
            quantity_type=axis.role,
        )

    def _resolve_channel(self, channel: str) -> str:
        normalized = self.CHANNEL_ALIASES.get(str(channel), str(channel))
        if normalized not in self.CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; choose one of {self.CHANNELS}")
        return normalized

    def _channel_label(self) -> str:
        if getattr(self, "is_point_list", False):
            unit = self._point_channel_unit()
            quantity_type = (
                self.data.channel_quantity_type(self.channel)
                if self.channel in self.data.channel_labels
                else self.data.quantity_type(self.channel)
            )
            return display_channel_label(
                self.channel,
                unit,
                quantity_type=quantity_type,
            )
        if self.channel in {
            "num_events",
            "coverage_mask",
            "combined_mask",
            "file_mask",
            "nfit_mask",
        }:
            return self.CHANNEL_LABELS.get(self.channel, self.channel)
        if self.channel == "residual":
            return self.CHANNEL_LABELS[self.channel]
        source_channel = "signal" if self.channel in {"signal", "errors", "fit"} else self.channel
        auxiliary = self.data.auxiliary_channels.get(source_channel)
        if source_channel == "coverage_fraction" and auxiliary is None:
            # Coverage is a derived viewer channel even for older/native data
            # that do not persist it as an MDHisto auxiliary channel.
            return display_channel_label(
                self.CHANNEL_LABELS[source_channel],
                "fraction",
                quantity_type="dimensionless",
            )
        quantity_type = self.data.channel_quantity_type(source_channel)
        unit = self.data.channel_unit(source_channel)
        label = (
            auxiliary.label
            if auxiliary is not None
            else self.CHANNEL_LABELS.get(source_channel, source_channel)
        )
        return display_channel_label(
            label,
            unit,
            quantity_type=quantity_type,
            uncertainty=self.channel == "errors",
        )

    def _display_values(self, view: dict[str, np.ndarray]) -> np.ndarray:
        if getattr(self, "is_point_list", False):
            return np.asarray(view["signal"], dtype=float)
        if self.channel in {"combined_mask", "file_mask", "nfit_mask"}:
            return np.asarray(view[self.channel], dtype=float)
        return np.asarray(view[self.channel], dtype=float)

    def _point_channel_unit(self) -> str:
        if self.channel in self.data.channel_labels:
            return self.data.unit(self.data.channel(self.channel)["value"])
        return self.data.unit(self.channel)

    def _point_slice_arrays(self) -> dict[str, np.ndarray]:
        x = np.asarray(self.data.column(self.x_key), dtype=float)
        if self.channel in self.data.channel_labels:
            y = np.asarray(self.data.channel_values(self.channel), dtype=float)
            errors = self.data.channel_errors(self.channel)
        else:
            y = np.asarray(self.data.column(self.channel), dtype=float)
            errors = None
        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]
        e = np.full(x.shape, np.nan) if errors is None else np.asarray(errors, dtype=float)[order]
        view = {
            "x_edges": _edges_from_centers(x),
            "y_edges": np.array([0.0, 1.0], dtype=float),
            "x_centers": x,
            "y_centers": np.array([], dtype=float),
            "signal": y,
            "errors": e,
            "num_events": np.ones_like(x),
            "combined_mask": ~np.isfinite(y),
            "mask": ~np.isfinite(y),
            "file_mask": np.zeros_like(x, dtype=bool),
            "nfit_mask": np.zeros_like(x, dtype=bool),
        }
        for name in ("fit", "residual"):
            label = self.point_overlay_channel(name)
            if label is not None:
                view[name] = np.asarray(self.data.channel_values(label), dtype=float)[order]
        return view

    def _resolve_dim(self, dim: int | str) -> int:
        if isinstance(dim, int):
            resolved = dim % self.data.signal.ndim
        else:
            names = [axis.name for axis in self.data.axes]
            if dim not in names:
                raise ValueError(f"unknown axis {dim!r}; choose one of {names}")
            resolved = names.index(dim)
        return resolved

    def _nearest_center_index(self, dim: int, value: float) -> int:
        centers = self.data.axes[dim].centers
        return int(np.nanargmin(np.abs(centers - value)))

    def _build_hidden_axis_controls(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RangeSlider

        if self.fig is None:
            return
        for axis in self.slider_axes:
            axis.remove()
        self.slider_axes = []
        self.sliders = {}
        self.integrate_checks = {}
        self.hidden_widgets = []

        hidden_dims = [dim for dim in range(self.data.signal.ndim) if dim not in (self.x_dim, self.y_dim)]
        top = 0.96
        for row, dim in enumerate(hidden_dims):
            axis = self.data.axes[dim]
            centers = axis.centers
            low = float(np.nanmin(centers))
            high = float(np.nanmax(centers))
            selection = self.selections.get(dim, (float(centers[centers.size // 2]),) * 2)
            slider_ax = self.fig.add_axes([0.08, top - row * 0.055, 0.54, 0.025])
            slider = RangeSlider(
                slider_ax,
                axis.name,
                low,
                high,
                valinit=selection,
                valfmt="%.4g",
            )
            slider.on_changed(lambda values, axis_dim=dim: self._set_axis_selection(axis_dim, values))
            check_ax = self.fig.add_axes([0.65, top - row * 0.055, 0.11, 0.025])
            check = CheckButtons(check_ax, ["range"], [self.integrate])
            check.on_clicked(lambda _label, axis_dim=dim: self._toggle_axis_integrate(axis_dim))
            self.slider_axes.extend([slider_ax, check_ax])
            self.sliders[dim] = slider
            self.integrate_checks[dim] = self.integrate
            self.hidden_widgets.extend([slider, check])
        plt.draw()

    def _set_axis_selection(self, dim: int, values: tuple[float, float]) -> None:
        self.selections[dim] = (float(values[0]), float(values[1]))
        self.update()

    def _toggle_axis_integrate(self, dim: int) -> None:
        self.integrate_checks[dim] = not self.integrate_checks.get(dim, self.integrate)
        self.update()

    def _set_display_dim(self, axis_name: str, dim: int) -> None:
        if self._syncing_axis_radios:
            return
        if axis_name == "x":
            if dim == self.y_dim:
                self.x_dim, self.y_dim = self.y_dim, self.x_dim
                self._sync_axis_radios()
                self._build_hidden_axis_controls()
                self.update()
                return
            self.x_dim = dim
        else:
            if dim == self.x_dim:
                self.x_dim, self.y_dim = self.y_dim, self.x_dim
                self._sync_axis_radios()
                self._build_hidden_axis_controls()
                self.update()
                return
            self.y_dim = dim
        self._build_hidden_axis_controls()
        self.update()

    def _sync_axis_radios(self) -> None:
        if self.x_radio is None or self.y_radio is None:
            return
        self._syncing_axis_radios = True
        try:
            self.x_radio.set_active(self.x_dim)
            self.y_radio.set_active(self.y_dim)
        finally:
            self._syncing_axis_radios = False

    def _set_cmap(self, cmap: str) -> None:
        self.cmap = "grey" if str(cmap) == "gray" else str(cmap).removesuffix("_r")
        self.update()

    def _effective_cmap(self) -> str:
        cmap = "gray" if self._is_boolean_channel() or self.cmap == "grey" else self.cmap
        return f"{cmap}_r" if self.cmap_reversed else cmap

    def _is_boolean_channel(self) -> bool:
        return self.channel in {"combined_mask", "file_mask", "nfit_mask"}

    def _set_color_scale(self, color_scale: str) -> None:
        self.color_scale = color_scale
        self.update()

    def _set_auto_limits(self, auto_limits: str) -> None:
        self.auto_limits = auto_limits
        self.update()

    def _toggle_autoscale(self) -> None:
        self.autoscale = not self.autoscale
        self.update()

    def _set_manual_limit(self, which: str, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            return
        if which == "vmin":
            self.manual_vmin = value
        else:
            self.manual_vmax = value
        self.autoscale = False
        self.update()

    def _color_norm(self, values: np.ndarray):
        import matplotlib.colors as colors

        vmin, vmax = self._color_limits(values)
        if self._is_boolean_channel():
            return colors.Normalize(vmin=0.0, vmax=1.0)
        if self.color_scale == "log":
            finite_positive = values[np.isfinite(values) & (values > 0)]
            if finite_positive.size == 0:
                return colors.Normalize(vmin=vmin, vmax=vmax)
            positive_min = max(vmin, float(np.min(finite_positive)))
            positive_max = max(vmax, positive_min * 1.01)
            return colors.LogNorm(vmin=positive_min, vmax=positive_max)
        if self.color_scale == "symmetriclog":
            finite = values[np.isfinite(values)]
            linthresh = max(np.nanstd(finite) * 0.01, 1e-12) if finite.size else 1e-12
            return colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        if self.color_scale == "asinh":
            asinh_norm = getattr(colors, "AsinhNorm", None)
            if asinh_norm is not None:
                return asinh_norm(vmin=vmin, vmax=vmax)
            return colors.Normalize(vmin=vmin, vmax=vmax)
        if self.color_scale == "power":
            gamma = max(float(self.power_gamma), 1e-12)

            def forward(values):
                scaled = (np.asarray(values) - vmin) / (vmax - vmin)
                return 1.0 - np.exp(-gamma * np.clip(scaled, 0.0, 1.0))

            def inverse(values):
                clipped = np.clip(values, 0.0, 1.0 - np.finfo(float).eps)
                return vmin - (vmax - vmin) * np.log1p(-clipped) / gamma

            return colors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax)
        return colors.Normalize(vmin=vmin, vmax=vmax)

    def _color_limits(self, values: np.ndarray) -> tuple[float, float]:
        if self._is_boolean_channel():
            return (0.0, 1.0)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)
        if not self.autoscale and self.manual_vmin is not None and self.manual_vmax is not None:
            if self.manual_vmin != self.manual_vmax:
                return (min(self.manual_vmin, self.manual_vmax), max(self.manual_vmin, self.manual_vmax))
        if self.auto_limits in {"3-sigma", "N-sigma"}:
            center = float(np.nanmean(finite))
            width = float(self.sigma_n * np.nanstd(finite))
            vmin, vmax = center - width, center + width
        elif self.auto_limits in {"1.5 IQR", "N IQR"}:
            q1, q3 = np.nanpercentile(finite, [25.0, 75.0])
            width = self.iqr_n * (q3 - q1)
            vmin, vmax = float(q1 - width), float(q3 + width)
        elif self.auto_limits == "Nth percentile":
            n = float(np.clip(self.percentile_n, 0.0, 50.0))
            vmin, vmax = np.nanpercentile(finite, [n, 100.0 - n])
            vmin, vmax = float(vmin), float(vmax)
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            pad = 1.0 if vmin == 0 else abs(vmin) * 0.01
            vmin -= pad
            vmax += pad
        return vmin, vmax

    def _on_motion(self, event) -> None:
        if self.cursor_text is None or self._current_slice is None or event.inaxes != self.ax_image:
            return
        if event.xdata is None or event.ydata is None:
            return
        view = self._current_slice
        x_idx = int(np.searchsorted(view["x_edges"], event.xdata, side="right") - 1)
        y_idx = int(np.searchsorted(view["y_edges"], event.ydata, side="right") - 1)
        if not (0 <= x_idx < view["signal"].shape[1] and 0 <= y_idx < view["signal"].shape[0]):
            return
        self.cursor_text.set_text(
            "\n".join(
                [
                    f"x: {view['x_centers'][x_idx]:.6g}",
                    f"y: {view['y_centers'][y_idx]:.6g}",
                    f"I: {view['signal'][y_idx, x_idx]:.6g}",
                    f"err: {view['errors'][y_idx, x_idx]:.6g}",
                ]
            )
        )
        self.fig.canvas.draw_idle()

    def _on_rectangle(self, click, release) -> None:
        if self._current_slice is None or self.ax_xcut is None or self.ax_ycut is None:
            return
        if click.xdata is None or release.xdata is None or click.ydata is None or release.ydata is None:
            return
        x0, x1 = sorted([click.xdata, release.xdata])
        y0, y1 = sorted([click.ydata, release.ydata])
        view = self._current_slice
        x_mask = (view["x_centers"] >= x0) & (view["x_centers"] <= x1)
        y_mask = (view["y_centers"] >= y0) & (view["y_centers"] <= y1)
        if not np.any(x_mask) or not np.any(y_mask):
            return
        z = view["signal"]
        x_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=0)
        y_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=1)

        self.ax_xcut.clear()
        self.ax_xcut.plot(view["x_centers"][x_mask], x_cut, "-", lw=1.2)
        self.ax_xcut.set_ylabel("int.")
        self.ax_xcut.set_xlabel(self._axis_label(self.x_dim))

        self.ax_ycut.clear()
        self.ax_ycut.plot(y_cut, view["y_centers"][y_mask], "-", lw=1.2)
        self.ax_ycut.set_xlabel("int.")
        self.ax_ycut.set_ylabel(self._axis_label(self.y_dim))
        self.fig.canvas.draw_idle()


def gaussian_smooth_nan(values: np.ndarray, sigma: float | Sequence[float]) -> np.ndarray:
    """Gaussian-smooth plotting values, filling gaps from finite neighbors."""

    from scipy.ndimage import gaussian_filter

    array = np.asarray(values, dtype=float)
    sigma_values = np.broadcast_to(np.asarray(sigma, dtype=float), (array.ndim,))
    if not np.any(sigma_values > 0.0):
        return array.copy()
    finite = np.isfinite(array)
    numerator = gaussian_filter(
        np.where(finite, array, 0.0),
        sigma=sigma_values,
        mode="constant",
        cval=0.0,
    )
    denominator = gaussian_filter(
        finite.astype(float),
        sigma=sigma_values,
        mode="constant",
        cval=0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = numerator / denominator
    return np.where(denominator > np.finfo(float).eps, smoothed, np.nan)


def gaussian_smooth_uncertainty(
    errors: np.ndarray,
    sigma: float | Sequence[float],
) -> np.ndarray:
    """Propagate independent one-sigma errors through normalized Gaussian smoothing."""

    from scipy.ndimage import convolve1d, gaussian_filter

    array = np.asarray(errors, dtype=float)
    sigma_values = np.broadcast_to(np.asarray(sigma, dtype=float), (array.ndim,))
    if not np.any(sigma_values > 0.0):
        return array.copy()
    finite = np.isfinite(array)
    denominator = gaussian_filter(
        finite.astype(float),
        sigma=sigma_values,
        mode="constant",
        cval=0.0,
    )
    propagated_variance = np.where(finite, np.square(array), 0.0)
    for axis, width in enumerate(sigma_values):
        if width <= 0.0:
            continue
        radius = int(4.0 * float(width) + 0.5)
        positions = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * np.square(positions / float(width)))
        kernel /= np.sum(kernel)
        propagated_variance = convolve1d(
            propagated_variance,
            np.square(kernel),
            axis=axis,
            mode="constant",
            cval=0.0,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        uncertainty = np.sqrt(propagated_variance) / denominator
    return np.where(denominator > np.finfo(float).eps, uncertainty, np.nan)


def smooth_mdhisto_view(
    view: dict[str, np.ndarray],
    *,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    fill_nans: bool = True,
) -> dict[str, np.ndarray]:
    """Return a plot-only smoothed copy of a 1D or 2D slice-view mapping."""

    result = dict(view)
    reference = np.asarray(view.get("signal"), dtype=float)
    if reference.ndim == 1:
        sigma = (max(float(sigma_x), 0.0),)
    elif reference.ndim == 2:
        sigma = (max(float(sigma_y), 0.0), max(float(sigma_x), 0.0))
    else:
        return result
    if not any(value > 0.0 for value in sigma):
        return result
    excluded = {
        "combined_mask",
        "mask",
        "file_mask",
        "nfit_mask",
        "coverage_fraction",
        "coverage_mask",
    }
    for name, values in view.items():
        array = np.asarray(values)
        if name in excluded or array.shape != reference.shape or array.dtype == bool:
            continue
        if name == "errors":
            smoothed = gaussian_smooth_uncertainty(array, sigma)
        else:
            smoothed = gaussian_smooth_nan(array, sigma)
        if not fill_nans:
            smoothed = np.where(np.isfinite(array), smoothed, np.nan)
        result[name] = smoothed
    return result
