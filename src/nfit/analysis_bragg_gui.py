from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import PointListData
from .mdhisto import MDHistoData


def _bragg_parameter_group(name: str) -> str:
    groups = {
        "peak_source": "Peak selection",
        "peak_table_dataset_id": "Peak selection",
        "peak_positions_hkl": "Peak selection",
        "include_systematic_absences": "Peak selection",
        "d_min_angstrom": "Peak selection",
        "d_max_angstrom": "Peak selection",
        "energy_min_meV": "Elastic volume",
        "energy_max_meV": "Elastic volume",
        "method": "Integration region",
        "coordinate_frame": "Integration region",
        "box_half_widths": "Integration region",
        "ellipsoid_semiaxes": "Integration region",
        "ellipsoid_rotation": "Integration region",
        "center_mode": "Integration region",
        "centroid_search_radius": "Integration region",
        "background_mode": "Background",
        "background_inner_scale": "Background",
        "background_outer_scale": "Background",
        "exclude_neighbor_regions": "Background",
        "minimum_peak_coverage": "Quality filters",
        "minimum_background_coverage": "Quality filters",
        "minimum_signal_to_noise": "Quality filters",
        "maximum_background": "Quality filters",
        "gaussian_background": "Gaussian fit",
        "gaussian_max_nfev": "Gaussian fit",
        "gaussian_fallback": "Gaussian fit",
        "subvoxel_samples": "Numerics",
    }
    return groups.get(name, "Integration region")


def _format_table_value(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e5 or abs(value) < 1.0e-4:
        return f"{value:.5g}"
    return f"{value:.6g}"


def _bragg_status_text(status: int, definitions: dict[str, Any]) -> str:
    if status == 0:
        return "Accepted"
    reasons = [
        str(definitions.get(str(bit), f"status {bit}"))
        for bit in (1, 2, 4, 8, 16)
        if status & bit
    ]
    return "; ".join(reasons) if reasons else f"Rejected ({status})"


class NumericTableItem:
    """Factory-compatible numeric QTableWidgetItem with numeric sorting."""

    def __new__(cls, text: str, value: float):
        from PySide6 import QtWidgets

        class _Item(QtWidgets.QTableWidgetItem):
            def __init__(self, label: str, number: float) -> None:
                super().__init__(label)
                self.number = number

            def __lt__(self, other: Any) -> bool:
                if hasattr(other, "number"):
                    return self.number < float(other.number)
                return super().__lt__(other)

        return _Item(text, value)


class BraggPeakDiagnosticsWidget:
    """Persisted table-driven views of Bragg integration regions and fits."""

    def __init__(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PySide6 import QtWidgets

        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("bragg_peak_diagnostics")
        self.widget.setToolTip(
            "Select a reflection to inspect three local data planes, the integration region, and axis profiles."
        )
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QtWidgets.QLabel("Run a Bragg integration to inspect individual reflections.")
        self.summary.setWordWrap(True)
        splitter = QtWidgets.QSplitter()
        self.peak_table = QtWidgets.QTableWidget()
        self.peak_table.setObjectName("bragg_diagnostic_peak_table")
        self.peak_table.setToolTip("Select a reflection to update the diagnostic figure.")
        self.peak_table.setSortingEnabled(True)
        self.peak_table.horizontalHeader().setSortIndicatorShown(True)
        self.peak_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.peak_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.peak_table.setMinimumWidth(360)
        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        splitter.addWidget(self.peak_table)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(self.summary)
        layout.addWidget(splitter, 1)
        self.source: MDHistoData | None = None
        self.table: PointListData | None = None
        self.parameters: dict[str, Any] = {}
        self.peak_table.itemSelectionChanged.connect(self._selection_changed)

    def clear(self) -> None:
        self.source = None
        self.table = None
        self.parameters = {}
        self.peak_table.clear()
        self.peak_table.setRowCount(0)
        self.peak_table.setColumnCount(0)
        self.summary.setText("Run a Bragg integration to inspect individual reflections.")
        self.figure.clear()
        self.canvas.draw_idle()
    def set_result(
        self,
        source: MDHistoData | None,
        table: PointListData,
        parameters: dict[str, Any],
    ) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        self.table = table
        self.parameters = dict(parameters)
        self.source = source
        if source is not None:
            try:
                from .analysis.bragg import bragg_volume

                self.source = bragg_volume(
                    source,
                    parameters.get("energy_min_meV"),
                    parameters.get("energy_max_meV"),
                )
            except (TypeError, ValueError):
                self.source = None
        columns = [
            name
            for name in (
                "Accepted", "H", "K", "L", "I", "dI", "I/dI", "Coverage", "BackgroundCoverage"
            )
            if name in table.columns
        ]
        self.peak_table.setSortingEnabled(False)
        self.peak_table.clear()
        self.peak_table.setRowCount(table.size)
        self.peak_table.setColumnCount(len(columns) + 1)
        self.peak_table.setHorizontalHeaderLabels([*columns, "Reason"])
        definitions = table.metadata.get("status_bits", {})
        statuses = table.column("Status") if "Status" in table.columns else np.zeros(table.size)
        for row in range(table.size):
            accepted = bool(table.column("Accepted")[row]) if "Accepted" in table.columns else True
            for column, name in enumerate(columns):
                value = float(table.column(name)[row])
                text = "Yes" if name == "Accepted" and value else "No" if name == "Accepted" else _format_table_value(value)
                item = NumericTableItem(text, value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                if not accepted:
                    item.setForeground(QtGui.QColor("#d94b45"))
                self.peak_table.setItem(row, column, item)
            reason = QtWidgets.QTableWidgetItem(_bragg_status_text(int(statuses[row]), definitions))
            reason.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            if not accepted:
                reason.setForeground(QtGui.QColor("#d94b45"))
            self.peak_table.setItem(row, len(columns), reason)
        self.peak_table.setSortingEnabled(True)
        self.peak_table.resizeColumnsToContents()
        accepted_count = int(table.metadata.get("accepted_count", 0))
        self.summary.setText(
            f"{table.size} generated reflections; {accepted_count} accepted and "
            f"{table.size - accepted_count} rejected. Select a row to inspect its region."
        )
        if table.size:
            self.peak_table.selectRow(0)
        else:
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            axis.text(0.5, 0.5, "No peaks intersect the selected volume", ha="center", va="center")
            axis.set_axis_off()
            self.canvas.draw_idle()

    def select_peak(self, source_row: int) -> None:
        for row in range(self.peak_table.rowCount()):
            item = self.peak_table.item(row, 0)
            if item is not None and item.data(0x0100) == source_row:
                self.peak_table.selectRow(row)
                return
        self._draw_peak(source_row)

    def _selection_changed(self) -> None:
        items = self.peak_table.selectedItems()
        if items:
            source_row = items[0].data(0x0100)
            if source_row is not None:
                self._draw_peak(int(source_row))

    def _draw_peak(self, row: int) -> None:
        from itertools import product

        from matplotlib.patches import Ellipse, Rectangle

        if self.table is None:
            return
        self.figure.clear()
        if self.source is None or self.source.signal.ndim != 3:
            axis = self.figure.add_subplot(111)
            axis.text(0.5, 0.5, "Input volume is not available for this saved result", ha="center", va="center")
            axis.set_axis_off()
            self.canvas.draw_idle()
            return
        from .analysis.coordinates import bin_edges, physical_axis_vectors

        source = self.source
        peak = np.array([self.table.column(name)[row] for name in ("H", "K", "L")], dtype=float)
        vectors = physical_axis_vectors(source)[:, :3]
        coordinates = peak @ np.linalg.inv(vectors)
        edges = [bin_edges(axis, size) for axis, size in zip(source.axes, source.shape, strict=True)]
        centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
        indices = [int(np.argmin(np.abs(values - coordinate))) for values, coordinate in zip(centers, coordinates, strict=True)]
        widths = np.asarray(self.table.metadata.get("integration_half_widths", [0.1, 0.1, 0.1]), dtype=float)
        if str(self.table.metadata.get("coordinate_frame")) == "hkl":
            offsets = np.asarray(list(product(*[(-value, value) for value in widths])))
            coordinate_half_widths = np.max(np.abs(offsets @ np.linalg.inv(vectors)), axis=0)
        else:
            coordinate_half_widths = np.full(3, float(np.max(widths)))
        bin_widths = np.array([float(np.median(np.diff(edge))) for edge in edges])
        coordinate_half_widths = np.maximum(coordinate_half_widths, 0.75 * bin_widths)
        accepted = bool(self.table.column("Accepted")[row])
        color = "#2f9e68" if accepted else "#d94b45"
        method = str(self.table.metadata.get("method", "ellipsoid_sum"))
        outer_scale = float(self.table.metadata.get("background_outer_scale", 1.0))
        background_mode = str(self.table.metadata.get("background_mode", "none"))
        pairs = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        for column, (x_dim, y_dim, fixed_dim) in enumerate(pairs):
            axis = self.figure.add_subplot(2, 3, column + 1)
            plane = np.take(source.signal, indices[fixed_dim], axis=fixed_dim)
            axis.pcolormesh(edges[x_dim], edges[y_dim], np.asarray(plane).T, shading="auto", cmap="viridis")
            x0, y0 = coordinates[x_dim], coordinates[y_dim]
            hx, hy = coordinate_half_widths[x_dim], coordinate_half_widths[y_dim]
            patch = (
                Rectangle((x0 - hx, y0 - hy), 2 * hx, 2 * hy, fill=False, edgecolor=color, linewidth=2)
                if method == "box_sum"
                else Ellipse((x0, y0), 2 * hx, 2 * hy, fill=False, edgecolor=color, linewidth=2)
            )
            axis.add_patch(patch)
            if background_mode == "shell" and method != "gaussian_fit":
                outer = (
                    Rectangle((x0 - outer_scale * hx, y0 - outer_scale * hy), 2 * outer_scale * hx, 2 * outer_scale * hy, fill=False, edgecolor="#f0a53a", linewidth=1.3, linestyle="--")
                    if method == "box_sum"
                    else Ellipse((x0, y0), 2 * outer_scale * hx, 2 * outer_scale * hy, fill=False, edgecolor="#f0a53a", linewidth=1.3, linestyle="--")
                )
                axis.add_patch(outer)
            axis.plot([x0], [y0], marker="+", color=color, markersize=9, markeredgewidth=1.8)
            extent = np.maximum(coordinate_half_widths * max(outer_scale, 2.5), 3 * bin_widths)
            axis.set_xlim(x0 - extent[x_dim], x0 + extent[x_dim])
            axis.set_ylim(y0 - extent[y_dim], y0 + extent[y_dim])
            axis.set_xlabel(source.axes[x_dim].name)
            axis.set_ylabel(source.axes[y_dim].name)
            axis.set_title(f"{source.axes[fixed_dim].name} = {centers[fixed_dim][indices[fixed_dim]]:.4g}")
        amplitude = float(self.table.column("FitAmplitude")[row]) if "FitAmplitude" in self.table.columns else np.nan
        baseline = float(self.table.column("FitBaseline")[row]) if "FitBaseline" in self.table.columns else np.nan
        fit_sigmas = np.array([
            float(self.table.column(name)[row]) if name in self.table.columns else np.nan
            for name in ("FitSigma1", "FitSigma2", "FitSigma3")
        ])
        for dim in range(3):
            axis = self.figure.add_subplot(2, 3, dim + 4)
            selection = list(indices)
            selection[dim] = slice(None)
            profile = np.asarray(source.signal[tuple(selection)], dtype=float)
            axis.plot(centers[dim], profile, "o", ms=3.5, color="#3478b8", label="Data")
            if np.isfinite(amplitude) and np.isfinite(baseline) and np.isfinite(fit_sigmas[dim]) and fit_sigmas[dim] > 0:
                x = np.linspace(centers[dim][0], centers[dim][-1], 400)
                fitted = baseline + amplitude * np.exp(-0.5 * ((x - coordinates[dim]) / fit_sigmas[dim]) ** 2)
                axis.plot(x, fitted, color="#d94b45", linewidth=1.8, label="Gaussian fit")
                axis.legend(fontsize=8)
            extent = max(coordinate_half_widths[dim] * max(outer_scale, 3.0), 4 * bin_widths[dim])
            axis.set_xlim(coordinates[dim] - extent, coordinates[dim] + extent)
            axis.set_xlabel(source.axes[dim].name)
            axis.set_ylabel("Intensity")
        status = int(self.table.column("Status")[row])
        reason = _bragg_status_text(status, self.table.metadata.get("status_bits", {}))
        signal_to_noise = float(self.table.column("I/dI")[row])
        self.figure.suptitle(
            f"({peak[0]:.4g}, {peak[1]:.4g}, {peak[2]:.4g})  |  {reason}  |  I/dI = {_format_table_value(signal_to_noise)}",
            color=color,
        )
        self.canvas.draw_idle()
