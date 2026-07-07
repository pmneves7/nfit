from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .dataset import PointData4D
from .mdhisto import MDHistoData


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
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 10.0,
    show_histogram_axes: bool = False,
    roi_extents: tuple[float, float, float, float] | None = None,
    xcut_percent: float = 20.0,
    ycut_percent: float = 16.0,
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

    view = model.slice_arrays()
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
        ax_image.set_title("MDHisto slice")
        if xlim is not None:
            ax_image.set_xlim(*xlim)
        if ylim is not None:
            ax_image.set_ylim(*ylim)
        colorbar = fig.colorbar(image, cax=ax_colorbar)
        colorbar.set_label(model._channel_label())

        if show_histogram_axes and roi_extents is not None and ax_xcut is not None and ax_ycut is not None:
            _draw_mdhisto_roi_cuts(model, view, roi_extents, ax_xcut, ax_ycut)

    return fig


def _panel_ratio(percent: float) -> float:
    fraction = float(np.clip(percent, 1.0, 80.0)) / 100.0
    return fraction / (1.0 - fraction)


def _draw_mdhisto_roi_cuts(
    model: "MDHistoSliceViewer",
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
        x_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=0)
        y_cut = np.nansum(z[np.ix_(y_mask, x_mask)], axis=1)
        ax_xcut.plot(view["x_centers"][x_mask], x_cut, "-", lw=1.2)
        ax_ycut.plot(y_cut, view["y_centers"][y_mask], "-", lw=1.2)
    ax_xcut.set_ylabel("Int.")
    ax_xcut.set_xlabel(model._axis_label(model.x_dim))
    ax_ycut.set_xlabel("Int.")
    ax_ycut.set_ylabel(model._axis_label(model.y_dim))


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
    COLORMAPS = ("viridis", "magma", "plasma", "cividis", "turbo")
    CHANNELS = ("signal", "errors", "num_events", "mask")
    CHANNEL_ALIASES = {"multiplicity": "num_events", "events": "num_events", "error": "errors"}
    CHANNEL_LABELS = {
        "signal": "Signal",
        "errors": "Error",
        "num_events": "Multiplicity",
        "mask": "Mask",
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
    ) -> None:
        if data.signal.ndim < 2:
            raise ValueError("slice viewer requires data with at least two dimensions")
        self.data = data
        self.x_dim = self._resolve_dim(x_dim)
        self.y_dim = self._resolve_dim(y_dim)
        if self.x_dim == self.y_dim:
            raise ValueError("x_dim and y_dim must be different")
        self.cmap = cmap
        self.channel = self._resolve_channel(channel)
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

        selections = self._normalized_selections()
        signal, variance, events, mask = self._reduce_arrays(selections)

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

        return {
            "x_edges": self._axis_edges(self.x_dim),
            "y_edges": self._axis_edges(self.y_dim),
            "x_centers": self.data.axes[self.x_dim].centers,
            "y_centers": self.data.axes[self.y_dim].centers,
            "signal": signal2d,
            "errors": np.sqrt(variance2d),
            "num_events": events2d,
            "mask": mask2d,
        }

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
            cmap=self.cmap,
            norm=norm,
        )
        self.ax_image.set_xlabel(self._axis_label(self.x_dim))
        self.ax_image.set_ylabel(self._axis_label(self.y_dim))
        self.ax_image.set_title("MDHisto slice")
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
        for dim in range(self.data.signal.ndim):
            if dim in (self.x_dim, self.y_dim):
                index.append(slice(None))
            else:
                selection = selections[dim]
                if isinstance(selection, tuple):
                    start, stop = selection
                    index.append(slice(start, stop + 1))
                    reduce_axes.append(len(index) - 1)
                else:
                    index.append(selection)

        signal = np.asarray(self.data.signal[tuple(index)], dtype=float)
        variance = np.asarray(self.data.errors[tuple(index)], dtype=float) ** 2
        events = np.asarray(self.data.num_events[tuple(index)], dtype=float)
        mask = np.asarray(self.data.mask[tuple(index)], dtype=bool)
        if self.masked:
            signal = np.where(mask, np.nan, signal)
            variance = np.where(mask, np.nan, variance)
            events = np.where(mask, 0.0, events)
        for axis in sorted(reduce_axes, reverse=True):
            signal = np.nansum(signal, axis=axis)
            variance = np.nansum(variance, axis=axis)
            events = np.nansum(events, axis=axis)
            mask = np.all(mask, axis=axis)
            signal, variance, mask = self._blank_empty_bins(signal, variance, events, mask)
        signal, variance, mask = self._blank_empty_bins(signal, variance, events, mask)
        return signal, variance, events, mask

    def _blank_empty_bins(
        self,
        signal: np.ndarray,
        variance: np.ndarray,
        events: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = np.asarray(events <= 0.0)
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
        axis = self.data.axes[dim]
        return f"{axis.name} ({axis.units})" if axis.units else axis.name

    def _resolve_channel(self, channel: str) -> str:
        normalized = self.CHANNEL_ALIASES.get(str(channel), str(channel))
        if normalized not in self.CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; choose one of {self.CHANNELS}")
        return normalized

    def _channel_label(self) -> str:
        return self.CHANNEL_LABELS.get(self.channel, self.channel)

    def _display_values(self, view: dict[str, np.ndarray]) -> np.ndarray:
        if self.channel == "mask":
            return np.asarray(view[self.channel], dtype=float)
        return np.asarray(view[self.channel], dtype=float)

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
        self.cmap = cmap
        self.update()

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


def slice_viewer(data: MDHistoData, **kwargs) -> MDHistoSliceViewer:
    """Open an interactive Qt MD histogram slice viewer.

    Parameters are forwarded to :class:`metallix.qt_slice_viewer.QtMDHistoSliceViewer`.
    The returned viewer object owns the Qt window. In scripts, call
    ``viewer.run()`` to start the Qt event loop; in notebooks with ``%gui qt``,
    keeping the returned object assigned is usually sufficient.

    ``viewer = slice_viewer(data)``
    """

    from .qt_slice_viewer import QtMDHistoSliceViewer

    viewer = QtMDHistoSliceViewer(data, **kwargs)
    viewer.show()
    return viewer
