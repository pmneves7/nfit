"""Qt and Matplotlib presentation for stored fit diagnostics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .fit_results import (
    _display_fit_parameters,
    _posterior_correlation_matrix,
    _posterior_display_options,
    _sampling_result_from_dict,
)
from .fitting import SamplingResult
from .pipeline import FitTimelineEntry


def _format_number(value: Any) -> str:
    """Format a diagnostic value compactly for plots and labels."""

    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _tooltip_table_corner_buttons(table: Any, tooltip: str) -> None:
    """Give a table's internal corner/select buttons a tooltip."""

    from PySide6 import QtWidgets

    for button in table.findChildren(QtWidgets.QAbstractButton):
        if not button.toolTip():
            button.setToolTip(tooltip)


class _FitDiagnosticsPlotWindow:
    """Dedicated Matplotlib window for fit covariance and posterior diagnostics."""

    def __init__(self, fit_entry: FitTimelineEntry, parent: Any) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        self.fit_entry = fit_entry
        self.window = QtWidgets.QMainWindow(parent.window if hasattr(parent, "window") else parent)
        self.window.setWindowTitle("Fit diagnostics")
        self.close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self.window)
        self.close_shortcut.activated.connect(self.window.close)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setToolTip("Fit diagnostics from covariance estimates and stored emcee samples.")
        self.label_table = self._make_label_table()
        self.plot_label_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plot_label_splitter.setObjectName("fit_diagnostics_plot_label_splitter")
        self.plot_label_splitter.setToolTip(
            "Drag the divider to allocate space between diagnostic plots and editable parameter labels."
        )
        self.plot_label_splitter.setChildrenCollapsible(False)
        self.plot_label_splitter.addWidget(self.tabs)
        self.plot_label_splitter.addWidget(self.label_table)
        self.plot_label_splitter.setStretchFactor(0, 4)
        self.plot_label_splitter.setStretchFactor(1, 1)
        self.plot_label_splitter.setSizes([520, 180])
        layout.addWidget(self.plot_label_splitter, 1)
        self.window.setCentralWidget(central)
        self._redraw_plots()

    def _make_label_table(self) -> Any:
        from PySide6 import QtCore, QtWidgets

        names = _fit_entry_diagnostic_parameter_names(self.fit_entry)
        labels = _fit_parameter_plot_labels(self.fit_entry, names)
        table = QtWidgets.QTableWidget(len(names), 2)
        table.setObjectName("fit_diagnostics_label_table")
        table.setToolTip(
            "Edit plot labels for this fit result. Plain text or Matplotlib mathtext/LaTeX-style labels are accepted."
        )
        table.setHorizontalHeaderLabels(["Full parameter name", "Plot label"])
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(names):
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            label_item = QtWidgets.QTableWidgetItem(labels.get(name, f"p{row + 1}"))
            name_item.setToolTip(name)
            label_item.setToolTip("Editable plot label. Use p1-style names, plain text, or mathtext such as $\\Gamma$.")
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, label_item)
        table.itemChanged.connect(self._label_table_changed)
        table.setMinimumHeight(90)
        table.resizeColumnsToContents()
        _tooltip_table_corner_buttons(table, "Select all parameter-label rows.")
        return table

    def _label_table_changed(self, item: Any) -> None:
        if item.column() != 1:
            return
        labels = dict(self.fit_entry.metadata.get("parameter_labels", {}))
        full_name_item = self.label_table.item(item.row(), 0)
        if full_name_item is None:
            return
        full_name = full_name_item.text()
        labels[full_name] = item.text().strip() or _compact_diagnostic_labels([full_name])[0]
        self.fit_entry.metadata["parameter_labels"] = labels
        self._redraw_plots()

    def _redraw_plots(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        active_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        self.tabs.clear()

        covariance = _covariance_matrix_from_fit_entry(self.fit_entry)
        if covariance is not None:
            matrix, names, title = covariance
            labels = _fit_parameter_plot_labels(self.fit_entry, names)
            cov_fig = Figure(figsize=(max(7, 1.0 * len(names) + 4), max(5, 0.75 * len(names) + 3)))
            _draw_centered_matrix_heatmap(cov_fig, matrix, names, labels=labels, title=title)
            self.tabs.addTab(FigureCanvas(cov_fig), title)

        result = _sampling_result_from_dict(self.fit_entry.metadata.get("posterior_samples"))
        if result is None:
            self._restore_active_tab(active_tab)
            return

        samples = np.asarray(result.samples, dtype=float)
        names = list(result.variable_names)
        labels = _fit_parameter_plot_labels(self.fit_entry, names)
        summaries = _fit_parameter_summaries(self.fit_entry)

        trace_fig = Figure(figsize=(8, max(3, 1.4 * max(1, len(names)))))
        trace_axes = trace_fig.subplots(max(1, len(names)), 1, squeeze=False)
        for index, name in enumerate(names):
            ax = trace_axes[index, 0]
            _draw_trace_panel(
                ax,
                result,
                parameter_index=index,
                summary=summaries.get(name, {}),
            )
            ax.set_ylabel(labels.get(name, f"p{index + 1}"))
        trace_axes[-1, 0].set_xlabel("MCMC step" if result.chain is not None else "sample")
        trace_fig.subplots_adjust(left=0.18, right=0.98, bottom=0.1, top=0.95, hspace=0.28)
        self.tabs.addTab(FigureCanvas(trace_fig), "Trace")

        n = len(names)
        corner_fig = Figure(figsize=(max(6, 2.45 * max(1, n)), max(6, 2.45 * max(1, n))))
        axes = corner_fig.subplots(max(1, n), max(1, n), squeeze=False)
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if row == col:
                    _draw_corner_histogram_panel(
                        ax,
                        samples[:, col],
                        labels.get(names[col], f"p{col + 1}"),
                        summaries.get(names[col], {}),
                    )
                elif row > col:
                    _draw_corner_density_panel(ax, samples[:, col], samples[:, row])
                    _draw_corner_reference_lines(
                        ax,
                        summaries.get(names[col], {}),
                        summaries.get(names[row], {}),
                    )
                else:
                    ax.axis("off")
                _disable_axis_offset_text(ax)
                if row == n - 1:
                    ax.set_xlabel(labels.get(names[col], f"p{col + 1}"))
                if col == 0 and row > 0:
                    ax.set_ylabel(labels.get(names[row], f"p{row + 1}"))
                elif row == col and col == 0:
                    ax.set_ylabel("Count")
        corner_fig.subplots_adjust(
            left=0.18,
            right=0.98,
            bottom=0.18,
            top=0.9,
            hspace=0.48,
            wspace=0.48,
        )
        self.tabs.addTab(FigureCanvas(corner_fig), "Corner")
        self._restore_active_tab(active_tab)

    def _restore_active_tab(self, tab_text: str) -> None:
        if not tab_text:
            return
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == tab_text:
                self.tabs.setCurrentIndex(index)
                return

    def show(self) -> None:
        self.window.resize(900, 700)
        self.window.show()


def _draw_matrix_heatmap(
    ax: Any,
    matrix: Any,
    names: list[str],
    *,
    labels: dict[str, str] | None = None,
    title: str,
    colorbar_ax: Any | None = None,
) -> None:
    """Draw a labeled covariance or correlation heatmap."""

    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return
    plot_labels = labels or {
        name: label
        for name, label in zip(
            names,
            _compact_diagnostic_labels(names),
            strict=True,
        )
    }
    finite = arr[np.isfinite(arr)]
    if title.lower().startswith("correlation"):
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
    elif finite.size:
        limit = float(np.nanmax(np.abs(finite)))
        vmin, vmax = (-limit, limit) if limit > 0 else (-1.0, 1.0)
        cmap = "coolwarm"
    else:
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
    image = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels([plot_labels.get(name, f"p{index + 1}") for index, name in enumerate(names)], rotation=0)
    ax.set_yticklabels([plot_labels.get(name, f"p{index + 1}") for index, name in enumerate(names)])
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            value = arr[row, col]
            if np.isfinite(value):
                ax.text(col, row, _format_number(value), ha="center", va="center", fontsize=7)
    if colorbar_ax is None:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.figure.colorbar(image, cax=colorbar_ax)


def _draw_centered_matrix_heatmap(
    fig: Any,
    matrix: Any,
    names: list[str],
    *,
    labels: dict[str, str] | None = None,
    title: str,
) -> Any:
    """Draw the covariance/correlation matrix centered in the diagnostics tab."""

    arr = np.asarray(matrix, dtype=float)
    label_count = max(1, len(names))
    square_size = min(0.66, max(0.42, 0.10 * label_count + 0.34))
    colorbar_width = 0.028
    gap = 0.025
    total_width = square_size + gap + colorbar_width
    left = max(0.08, (1.0 - total_width) / 2.0)
    bottom = max(0.14, (1.0 - square_size) / 2.0)
    ax = fig.add_axes([left, bottom, square_size, square_size])
    colorbar_ax = fig.add_axes([left + square_size + gap, bottom, colorbar_width, square_size])
    _draw_matrix_heatmap(ax, arr, names, labels=labels, title=title, colorbar_ax=colorbar_ax)
    return ax


def _compact_diagnostic_labels(names: list[str]) -> list[str]:
    return [f"p{index + 1}" for index in range(len(names))]


def _draw_trace_panel(
    ax: Any,
    result: SamplingResult,
    *,
    parameter_index: int,
    summary: dict[str, float],
) -> None:
    """Draw one posterior trace panel, preferring raw walker chains."""

    if result.chain is not None:
        chain = np.asarray(result.chain, dtype=float)
        if chain.ndim == 3 and parameter_index < chain.shape[2]:
            steps = np.arange(chain.shape[0])
            ax.plot(steps, chain[:, :, parameter_index], linewidth=0.55, alpha=0.75)
            burn_in = int(result.metadata.get("burn_in", 0) or 0)
            if 0 < burn_in < chain.shape[0]:
                ax.axvline(
                    burn_in,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                )
                top = ax.get_ylim()[1]
                ax.annotate(
                    "burn-in",
                    xy=(burn_in, top),
                    xytext=(4, -4),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="black",
                )
            if summary.get("best") is not None:
                ax.axhline(float(summary["best"]), color="red", linewidth=0.9, alpha=0.8)
            return

    samples = np.asarray(result.samples, dtype=float)
    if samples.ndim == 2 and parameter_index < samples.shape[1]:
        ax.plot(samples[:, parameter_index], linewidth=0.7)
    if summary.get("best") is not None:
        ax.axhline(float(summary["best"]), color="red", linewidth=0.9, alpha=0.8)


def _fit_entry_diagnostic_parameter_names(fit_entry: FitTimelineEntry) -> list[str]:
    names: list[str] = []
    covariance = _covariance_matrix_from_fit_entry(fit_entry)
    if covariance is not None:
        _matrix, cov_names, _title = covariance
        names.extend(cov_names)
    samples = _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples"))
    if samples is not None:
        names.extend(samples.variable_names)
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = goodness.get("parameters") if isinstance(goodness.get("parameters"), dict) else {}
    names.extend(str(name) for name in params)
    return list(dict.fromkeys(names))


def _fit_parameter_plot_labels(
    fit_entry: FitTimelineEntry,
    names: list[str],
) -> dict[str, str]:
    stored = fit_entry.metadata.get("parameter_labels")
    labels = dict(stored) if isinstance(stored, dict) else {}
    out: dict[str, str] = {}
    defaults_changed = False
    for index, name in enumerate(names):
        label = str(labels.get(name, "")).strip()
        if not label:
            label = f"p{index + 1}"
            labels[name] = label
            defaults_changed = True
        out[name] = label
    if defaults_changed:
        fit_entry.metadata["parameter_labels"] = labels
    return out


def _fit_parameter_summaries(fit_entry: FitTimelineEntry) -> dict[str, dict[str, float]]:
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    params = _display_fit_parameters(fit_entry)
    stderr = goodness.get("stderr") if isinstance(goodness.get("stderr"), dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    posterior_params = (
        posterior.get("parameters")
        if isinstance(posterior.get("parameters"), dict)
        else {}
    )
    use_posterior_errors = bool(_posterior_display_options(fit_entry).get("use_posterior_uncertainties"))
    names = set(params) | set(stderr) | set(posterior_params)
    summaries: dict[str, dict[str, float]] = {}
    for name in names:
        summary: dict[str, float] = {}
        if name in params:
            summary["best"] = float(params[name])
        if name in stderr:
            err = float(stderr[name])
            summary["stderr"] = err
            if "best" in summary:
                summary.setdefault("low", summary["best"] - err)
                summary.setdefault("high", summary["best"] + err)
        posterior_row = posterior_params.get(name)
        if use_posterior_errors and isinstance(posterior_row, dict):
            for source, target in (("median", "median"), ("p16", "low"), ("p84", "high")):
                if source in posterior_row:
                    summary[target] = float(posterior_row[source])
        summaries[str(name)] = summary
    return summaries


def _draw_corner_histogram_panel(
    ax: Any,
    values: Any,
    name: str,
    summary: dict[str, float],
) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    ax.hist(arr, bins=40, histtype="step", color="0.1", linewidth=1.1)
    center = summary.get("best", summary.get("median"))
    low = summary.get("low")
    high = summary.get("high")
    if center is not None:
        ax.axvline(center, color="red", linewidth=1.0)
    if low is not None:
        ax.axvline(low, color="0.25", linestyle="--", linewidth=0.8)
    if high is not None:
        ax.axvline(high, color="0.25", linestyle="--", linewidth=0.8)
    ax.set_title(_corner_histogram_title(name, summary), fontsize=9, pad=8)


def _draw_corner_reference_lines(
    ax: Any,
    x_summary: dict[str, float],
    y_summary: dict[str, float],
) -> None:
    x_center = x_summary.get("best", x_summary.get("median"))
    y_center = y_summary.get("best", y_summary.get("median"))
    if x_center is not None:
        ax.axvline(x_center, color="red", linewidth=0.8, alpha=0.85)
    if y_center is not None:
        ax.axhline(y_center, color="red", linewidth=0.8, alpha=0.85)
    if x_center is not None and y_center is not None:
        ax.plot([x_center], [y_center], marker="o", color="red", markersize=3)


def _corner_histogram_title(name: str, summary: dict[str, float]) -> str:
    center = summary.get("best", summary.get("median"))
    if center is None:
        return name
    low = summary.get("low")
    high = summary.get("high")
    if low is None or high is None:
        return rf"${_mathtext_label(name)} = {_format_number(center)}$"
    plus = high - center
    minus = center - low
    return (
        rf"${_mathtext_label(name)} = {_format_number(center)}"
        rf"\,\pm^{{+{_format_number(plus)}}}_{{-{_format_number(minus)}}}$"
    )


def _mathtext_label(label: str) -> str:
    stripped = str(label).strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1]
    if re.search(r"[\\{}_^]", stripped):
        return stripped
    return stripped.replace(" ", r"\ ")


def _disable_axis_offset_text(ax: Any) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)
        if hasattr(formatter, "set_scientific"):
            formatter.set_scientific(False)


def _covariance_matrix_from_fit_entry(
    fit_entry: FitTimelineEntry,
) -> tuple[np.ndarray, list[str], str] | None:
    if _posterior_display_options(fit_entry).get("use_posterior_uncertainties"):
        posterior = _posterior_correlation_matrix(fit_entry)
        if posterior is not None:
            matrix, names = posterior
            return matrix, names, "Posterior correlation"
    goodness = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    covariance = goodness.get("covariance") if isinstance(goodness.get("covariance"), dict) else {}
    names = [str(name) for name in covariance.get("variables", [])]
    matrix = covariance.get("matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if len(names) != arr.shape[0]:
                names = [f"p{index}" for index in range(arr.shape[0])]
            return arr, names, "Covariance"
    correlation = covariance.get("correlation")
    if isinstance(correlation, dict) and correlation:
        names = names or [str(name) for name in correlation]
        arr = np.asarray(
            [
                [float(dict(correlation.get(row_name, {})).get(col_name, np.nan)) for col_name in names]
                for row_name in names
            ],
            dtype=float,
        )
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr, names, "Correlation"
    return None


def _fit_entry_has_diagnostic_plots(fit_entry: FitTimelineEntry | None) -> bool:
    if fit_entry is None:
        return False
    if _covariance_matrix_from_fit_entry(fit_entry) is not None:
        return True
    return _sampling_result_from_dict(fit_entry.metadata.get("posterior_samples")) is not None


def _draw_corner_density_panel(ax: Any, x: Any, y: Any) -> None:
    """Draw a 2D posterior panel with density shading, contours, and samples."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size == 0 or np.ptp(x_arr) == 0.0 or np.ptp(y_arr) == 0.0:
        ax.scatter(x_arr, y_arr, s=2, alpha=0.25, color="tab:blue", linewidths=0)
        return

    counts, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=48)
    counts = counts.T
    if np.any(counts > 0):
        positive = counts[counts > 0]
        image = ax.imshow(
            counts,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            cmap="Blues",
            alpha=0.6,
            interpolation="nearest",
        )
        del image
        if positive.size >= 4:
            levels = np.percentile(positive, [50.0, 75.0, 90.0])
            levels = np.unique(levels[levels > 0])
            if levels.size:
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                ax.contour(
                    x_centers,
                    y_centers,
                    counts,
                    levels=levels,
                    colors="0.15",
                    linewidths=0.8,
                    alpha=0.85,
                )
    ax.scatter(x_arr, y_arr, s=1.4, alpha=0.12, color="tab:blue", linewidths=0)
