from __future__ import annotations

import json
from typing import Any

import numpy as np

from .analysis import (
    AnalysisContext,
    AnalysisEntry,
    AnalysisInput,
    analysis_definition,
    available_analysis_types,
    default_analysis_parameters,
)
from .analysis.artifacts import read_dataset_artifact
from .analysis.fingerprint import dataset_entry_fingerprint
from .analysis.runner import execute_to_artifacts
from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData
from .pipeline import DatasetEntry, DatasetGroup
from .quantities import display_unit


class DataPlaygroundWindow:
    """Registry-driven non-fitting analysis workbench for one project explorer."""

    def __init__(self, explorer: Any) -> None:
        from PySide6 import QtCore, QtWidgets

        self.explorer = explorer
        self.group = None
        self.parameter_widgets: dict[str, Any] = {}
        self.parameter_rows: dict[str, tuple[Any, Any]] = {}
        self._current_result_data: PointListData | MDHistoData | None = None
        self._current_result_output = None
        self.additional_input_ids: list[str] = []
        self.window = QtWidgets.QMainWindow(explorer.window)
        self.window.setWindowTitle("nfit Analysis Window")
        self.window.resize(900, 650)
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        analysis_row = QtWidgets.QHBoxLayout()
        self.analysis_combo = QtWidgets.QComboBox()
        self.analysis_combo.setToolTip("Select a saved analysis recipe or create a new one.")
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.setToolTip("Start a new analysis recipe.")
        self.duplicate_button = QtWidgets.QPushButton("Duplicate")
        self.duplicate_button.setToolTip("Duplicate the selected recipe without its previous result.")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setToolTip("Delete the selected recipe and its linked derived datasets.")
        analysis_row.addWidget(self.analysis_combo, 1)
        analysis_row.addWidget(self.new_button)
        analysis_row.addWidget(self.duplicate_button)
        analysis_row.addWidget(self.delete_button)
        layout.addLayout(analysis_row)
        selectors = QtWidgets.QHBoxLayout()
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setToolTip("Dataset used as the primary analysis input.")
        self.secondary_dataset_combo = QtWidgets.QComboBox()
        self.secondary_dataset_combo.setToolTip("Optional secondary input, such as an H, K, L peak table.")
        self.additional_inputs_button = QtWidgets.QPushButton("Select runs...")
        self.additional_inputs_button.setToolTip(
            "Choose all MDEvent rotation-angle datasets used by the angle-energy background estimator."
        )
        self.additional_inputs_button.setVisible(False)
        self.operation_combo = QtWidgets.QComboBox()
        for key in available_analysis_types():
            self.operation_combo.addItem(analysis_definition(key).label, key)
        self.operation_combo.setToolTip("Choose the dataset operation to configure.")
        self.name_edit = QtWidgets.QLineEdit("Analysis")
        self.name_edit.setToolTip("Editable name stored with this analysis recipe.")
        selectors.addWidget(self.dataset_combo, 2)
        selectors.addWidget(self.secondary_dataset_combo, 2)
        selectors.addWidget(self.additional_inputs_button, 2)
        selectors.addWidget(self.operation_combo, 2)
        selectors.addWidget(self.name_edit, 2)
        layout.addLayout(selectors)
        self.parameter_scroll = QtWidgets.QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_panel = QtWidgets.QWidget()
        self.parameter_form = QtWidgets.QFormLayout(self.parameter_panel)
        self.parameter_scroll.setWidget(self.parameter_panel)
        self.parameter_scroll.setMinimumHeight(240)
        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.setToolTip("Inspect analysis outputs, diagnostics, and provenance.")
        self.results = QtWidgets.QWidget()
        self.results.setToolTip("Latest analysis summary, output table, and artifact actions.")
        results_layout = QtWidgets.QVBoxLayout(self.results)
        results_layout.setContentsMargins(4, 4, 4, 4)
        self.result_summary = QtWidgets.QLabel("Run an analysis to inspect its outputs.")
        self.result_summary.setWordWrap(True)
        self.result_summary.setTextInteractionFlags(
            self.result_summary.textInteractionFlags()
            | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setObjectName("analysis_result_table")
        self.result_table.setToolTip(
            "Sortable analysis output. Select a Bragg reflection to update its diagnostic plots."
        )
        self.result_table.setSortingEnabled(True)
        self.result_table.horizontalHeader().setSortIndicatorShown(True)
        self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        result_actions = QtWidgets.QHBoxLayout()
        self.add_output_button = QtWidgets.QPushButton("Add to datasets")
        self.add_output_button.setToolTip(
            "Create a disabled dataset from this analysis artifact. Analysis tables are not added automatically."
        )
        self.add_output_button.setEnabled(False)
        self.add_output_button.clicked.connect(self.add_current_output_to_datasets)
        self.view_peak_overlay_button = QtWidgets.QPushButton("View input with peaks")
        self.view_peak_overlay_button.setToolTip(
            "Open the input dataset in the data viewer with accepted and rejected Bragg reflections marked."
        )
        self.view_peak_overlay_button.setEnabled(False)
        self.view_peak_overlay_button.clicked.connect(self.view_input_with_peaks)
        self.export_bragg_button = QtWidgets.QPushButton("Export .int")
        self.export_bragg_button.setToolTip(
            "Export accepted Bragg reflections as a headerless, space-delimited .int file "
            "containing H, K, L, I, and dI."
        )
        self.export_bragg_button.setEnabled(False)
        self.export_bragg_button.clicked.connect(self.export_current_bragg_int)
        result_actions.addWidget(self.add_output_button)
        result_actions.addWidget(self.view_peak_overlay_button)
        result_actions.addWidget(self.export_bragg_button)
        result_actions.addStretch(1)
        results_layout.addWidget(self.result_summary)
        results_layout.addWidget(self.result_table, 1)
        results_layout.addLayout(result_actions)
        self.diagnostics = QtWidgets.QWidget()
        self.diagnostics.setToolTip("Numerical quality summary and per-reflection integration diagnostics.")
        diagnostics_layout = QtWidgets.QVBoxLayout(self.diagnostics)
        diagnostics_layout.setContentsMargins(4, 4, 4, 4)
        self.diagnostic_summary = QtWidgets.QTextEdit()
        self.diagnostic_summary.setReadOnly(True)
        self.diagnostic_summary.setMaximumHeight(110)
        self.diagnostic_summary.setToolTip("Generated, accepted, rejected, and quality-threshold summary.")
        self.bragg_diagnostics = BraggPeakDiagnosticsWidget()
        diagnostics_layout.addWidget(self.diagnostic_summary)
        diagnostics_layout.addWidget(self.bragg_diagnostics.widget, 1)
        self.provenance = QtWidgets.QTextEdit()
        self.provenance.setReadOnly(True)
        self.provenance.setToolTip("Recipe hash and input fingerprints for this result.")
        self.result_tabs.addTab(self.results, "Results")
        self.result_tabs.addTab(self.diagnostics, "Diagnostics")
        self.result_tabs.addTab(self.provenance, "Provenance")
        self.result_tabs.setMinimumHeight(180)
        self.config_output_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.config_output_splitter.setObjectName("analysis_config_output_splitter")
        self.config_output_splitter.setToolTip(
            "Drag this divider to allocate space between the analysis configuration and its outputs."
        )
        self.config_output_splitter.setChildrenCollapsible(False)
        self.config_output_splitter.addWidget(self.parameter_scroll)
        self.config_output_splitter.addWidget(self.result_tabs)
        self.config_output_splitter.setStretchFactor(0, 3)
        self.config_output_splitter.setStretchFactor(1, 2)
        self.config_output_splitter.setSizes([390, 250])
        layout.addWidget(self.config_output_splitter, 1)
        commands = QtWidgets.QHBoxLayout()
        commands.addStretch(1)
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Run this recipe without modifying the input dataset.")
        commands.addWidget(self.run_button)
        layout.addLayout(commands)
        self.operation_combo.currentIndexChanged.connect(self._operation_changed)
        self.analysis_combo.currentIndexChanged.connect(self._select_analysis)
        self.new_button.clicked.connect(self.new_analysis)
        self.duplicate_button.clicked.connect(self.duplicate_analysis)
        self.delete_button.clicked.connect(self.delete_analysis)
        self.run_button.clicked.connect(self.run)
        self.result_table.itemSelectionChanged.connect(self._result_row_selected)
        self.additional_inputs_button.clicked.connect(self._choose_additional_inputs)
        self._rebuild_parameters()

    def show(self) -> None:
        self.window.show()

    def raise_(self) -> None:
        self.window.raise_()

    def select_group(self, group: Any, *, dataset: DatasetEntry | None = None) -> None:
        self.group = group
        self._clear_result_views()
        self._refresh_analysis_list()
        self.dataset_combo.clear()
        self.secondary_dataset_combo.clear()
        self.secondary_dataset_combo.addItem("No secondary input", None)
        entries = list(group.iter_datasets())
        for entry in entries:
            self.dataset_combo.addItem(entry.name, entry.id)
            self.secondary_dataset_combo.addItem(entry.name, entry.id)
        if dataset in entries:
            self.dataset_combo.setCurrentIndex(entries.index(dataset))
        self.additional_input_ids = [entry.id for entry in entries if entry.kind == "mdevent"]
        self._update_additional_inputs_button()

    def _refresh_analysis_list(self, selected_id: str | None = None) -> None:
        self.analysis_combo.blockSignals(True)
        self.analysis_combo.clear()
        self.analysis_combo.addItem("New analysis", None)
        if self.group is not None:
            for analysis in self.group.analyses:
                self.analysis_combo.addItem(analysis.name, analysis.id)
        index = self.analysis_combo.findData(selected_id) if selected_id else 0
        self.analysis_combo.setCurrentIndex(max(index, 0))
        self.analysis_combo.blockSignals(False)

    def _selected_analysis(self) -> AnalysisEntry | None:
        selected_id = self.analysis_combo.currentData()
        if self.group is None or selected_id is None:
            return None
        return next((item for item in self.group.analyses if item.id == selected_id), None)

    def _select_analysis(self) -> None:
        analysis = self._selected_analysis()
        if analysis is None:
            return
        self.name_edit.setText(analysis.name)
        index = self.operation_combo.findData(analysis.type)
        if index >= 0:
            self.operation_combo.setCurrentIndex(index)
        if analysis.input_dataset_ids:
            dataset_index = self.dataset_combo.findData(analysis.input_dataset_ids[0])
            if dataset_index >= 0:
                self.dataset_combo.setCurrentIndex(dataset_index)
        if len(analysis.input_dataset_ids) > 1:
            secondary_index = self.secondary_dataset_combo.findData(analysis.input_dataset_ids[1])
            if secondary_index >= 0:
                self.secondary_dataset_combo.setCurrentIndex(secondary_index)
        self.additional_input_ids = list(analysis.input_dataset_ids)
        self._update_additional_inputs_button()
        self._set_parameter_values(analysis.parameters)
        self._render_analysis_result(analysis)

    def _set_parameter_values(self, parameters: dict[str, Any]) -> None:
        from PySide6 import QtWidgets

        for name, value in parameters.items():
            widget = self.parameter_widgets.get(name)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText("" if value is None else json.dumps(value))

    def new_analysis(self) -> None:
        self.analysis_combo.setCurrentIndex(0)
        self.name_edit.setText("Analysis")
        self._rebuild_parameters()
        self._clear_result_views()

    def duplicate_analysis(self) -> None:
        source = self._selected_analysis()
        if source is None:
            return
        duplicate = AnalysisEntry(f"{source.name} copy", source.type, list(source.input_dataset_ids), dict(source.parameters))
        self.group.analyses.append(duplicate)
        self._refresh_analysis_list(duplicate.id)
        self._select_analysis()
        self.explorer._mark_dirty()
        self.explorer._refresh_tree(select_group=self.group)

    def delete_analysis(self) -> None:
        analysis = self._selected_analysis()
        if analysis is None:
            return
        if not self.explorer._delete_analysis_entry(self.group, analysis):
            return
        self._refresh_analysis_list()
        self._clear_result_views()
        self.explorer._mark_dirty()
        self.explorer._refresh_tree(select_group=self.group)

    def _rebuild_parameters(self) -> None:
        from PySide6 import QtWidgets

        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self.parameter_widgets.clear()
        self.parameter_rows.clear()
        key = self.operation_combo.currentData()
        if not key:
            return
        grouped_forms: dict[str, Any] = {}
        if key == "bragg_integration":
            for title in ("Peak selection", "Elastic volume", "Integration region", "Background", "Quality filters", "Gaussian fit", "Numerics"):
                group_box = QtWidgets.QGroupBox(title)
                group_box.setObjectName(f"bragg_parameter_group_{title.lower().replace(' ', '_')}")
                form = QtWidgets.QFormLayout(group_box)
                form.setContentsMargins(10, 8, 10, 8)
                grouped_forms[title] = form
                self.parameter_form.addRow(group_box)
        for parameter in analysis_definition(key).parameters:
            if isinstance(parameter.default, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(parameter.default)
            elif parameter.choices:
                widget = QtWidgets.QComboBox()
                for value, label in parameter.choices:
                    widget.addItem(label, value)
                widget.setCurrentIndex(max(widget.findData(parameter.default), 0))
            elif isinstance(parameter.default, int):
                widget = QtWidgets.QSpinBox()
                widget.setRange(-1_000_000_000, 1_000_000_000)
                widget.setValue(parameter.default)
            elif isinstance(parameter.default, float):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(-1.0e12, 1.0e12)
                widget.setDecimals(8)
                widget.setValue(parameter.default)
            else:
                widget = QtWidgets.QLineEdit(
                    "" if parameter.default is None else json.dumps(parameter.default)
                )
            widget.setToolTip(f"{parameter.description}\nAllowed: {parameter.allowed}\nExample: {parameter.example}")
            form = grouped_forms.get(_bragg_parameter_group(parameter.name), self.parameter_form)
            label = QtWidgets.QLabel(parameter.label)
            label.setToolTip(widget.toolTip())
            form.addRow(label, widget)
            self.parameter_widgets[parameter.name] = widget
            self.parameter_rows[parameter.name] = (label, widget)
        if key == "bragg_integration":
            for name in ("method", "background_mode", "center_mode", "peak_source"):
                widget = self.parameter_widgets.get(name)
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.currentIndexChanged.connect(self._sync_bragg_parameter_visibility)
            self._sync_bragg_parameter_visibility()

    def _operation_changed(self) -> None:
        self._rebuild_parameters()
        is_multi = self.operation_combo.currentData() == "angle_energy_background"
        self.secondary_dataset_combo.setVisible(not is_multi)
        self.additional_inputs_button.setVisible(is_multi)
        self._update_additional_inputs_button()

    def _update_additional_inputs_button(self) -> None:
        if not hasattr(self, "additional_inputs_button"):
            return
        count = len(set(self.additional_input_ids))
        self.additional_inputs_button.setText(f"Select runs... ({count})")

    def _choose_additional_inputs(self) -> None:
        from PySide6 import QtCore, QtWidgets

        if self.group is None:
            return
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle("Angle-energy background inputs")
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(
            "Select the MDEvent runs measured at different sample rotation angles."
        )
        layout.addWidget(label)
        choices = QtWidgets.QListWidget()
        choices.setToolTip(
            "Every checked run is independently reduced before the lowest-intensity fraction is averaged."
        )
        selected = set(self.additional_input_ids)
        for entry in self.group.iter_datasets():
            if entry.kind != "mdevent" and not isinstance(entry.data, PointData4D):
                continue
            item = QtWidgets.QListWidgetItem(entry.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.id)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if entry.id in selected
                else QtCore.Qt.CheckState.Unchecked
            )
            choices.addItem(item)
        layout.addWidget(choices)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.additional_input_ids = [
            str(choices.item(index).data(QtCore.Qt.ItemDataRole.UserRole))
            for index in range(choices.count())
            if choices.item(index).checkState() == QtCore.Qt.CheckState.Checked
        ]
        self._update_additional_inputs_button()

    def _parameters(self) -> dict[str, Any]:
        from PySide6 import QtWidgets

        values = default_analysis_parameters(self.operation_combo.currentData())
        for name, widget in self.parameter_widgets.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                values[name] = widget.value()
            else:
                text = widget.text().strip()
                if not text:
                    values[name] = None
                    continue
                try:
                    values[name] = json.loads(text)
                except json.JSONDecodeError:
                    values[name] = text
        return values

    def _sync_bragg_parameter_visibility(self) -> None:
        """Show only Bragg controls relevant to the selected modes."""

        from PySide6 import QtWidgets

        def choice(name: str) -> Any:
            widget = self.parameter_widgets.get(name)
            return widget.currentData() if isinstance(widget, QtWidgets.QComboBox) else None

        method = choice("method")
        background = choice("background_mode")
        center = choice("center_mode")
        source = choice("peak_source")
        visible = {
            "peak_table_dataset_id": False,
            "peak_positions_hkl": False,
            "box_half_widths": method == "box_sum",
            "ellipsoid_semiaxes": method in {"ellipsoid_sum", "gaussian_fit"},
            "ellipsoid_rotation": method == "ellipsoid_sum",
            "centroid_search_radius": center == "centroid",
            "background_inner_scale": background == "shell" and method != "gaussian_fit",
            "background_outer_scale": background == "shell" and method != "gaussian_fit",
            "exclude_neighbor_regions": background == "shell" and method != "gaussian_fit",
            "minimum_background_coverage": background == "shell" and method != "gaussian_fit",
            "gaussian_background": method == "gaussian_fit",
            "gaussian_max_nfev": method == "gaussian_fit",
            "gaussian_fallback": method == "gaussian_fit",
            "subvoxel_samples": method == "ellipsoid_sum",
        }
        if source == "table":
            self.secondary_dataset_combo.setToolTip("H, K, L peak table used by this Bragg recipe.")
        else:
            self.secondary_dataset_combo.setToolTip("Optional secondary input, such as an H, K, L peak table.")
        for name, (label, widget) in self.parameter_rows.items():
            show = visible.get(name, True)
            label.setVisible(show)
            widget.setVisible(show)

    def _clear_result_views(self) -> None:
        self._current_result_data = None
        self._current_result_output = None
        self.result_summary.setText("Run an analysis to inspect its outputs.")
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        self.diagnostic_summary.clear()
        self.bragg_diagnostics.clear()
        self.add_output_button.setEnabled(False)
        self.view_peak_overlay_button.setEnabled(False)
        self.export_bragg_button.setEnabled(False)

    def _artifact_data(self, output: Any) -> PointListData | MDHistoData | None:
        if self.explorer.project_path is None or not output.artifact_path:
            return None
        path = self.explorer.project_path.parent / output.artifact_path
        try:
            return read_dataset_artifact(path)
        except (OSError, TypeError, ValueError):
            return None

    def _render_analysis_result(self, analysis: AnalysisEntry) -> None:
        result = analysis.result
        if result is None:
            self._clear_result_views()
            return
        output_data = []
        summary = [f"Status: {result.status}"]
        for output in result.outputs:
            line = f"{output.label}: {output.kind}"
            data = self._artifact_data(output)
            if data is not None:
                output_data.append((output, data))
                line += f" ({getattr(data, 'size', 0)} rows)" if isinstance(data, PointListData) else ""
            elif output.scalar_value is not None:
                line += f" = {output.scalar_value:.6g}"
                if output.scalar_uncertainty is not None:
                    line += f" +/- {output.scalar_uncertainty:.2g}"
                if output.unit:
                    line += f" {display_unit(output.unit)}"
            summary.append(line)
        summary.extend(result.warnings)
        self.result_summary.setText("\n".join(summary))
        diagnostic_payload = dict(result.diagnostics)
        if output_data:
            output, data = output_data[0]
            self._current_result_output = output
            self._current_result_data = data
            diagnostic_payload["output_metadata"] = {
                **dict(output.metadata),
                **dict(getattr(data, "metadata", {})),
            }
            if isinstance(data, PointListData):
                self._populate_result_table(data)
            self.add_output_button.setEnabled(True)
            is_bragg = isinstance(data, PointListData) and data.metadata.get("analysis_kind") == "bragg_peak_integration"
            self.view_peak_overlay_button.setEnabled(is_bragg)
            self.export_bragg_button.setEnabled(is_bragg)
            if is_bragg:
                source = self._analysis_primary_data(analysis)
                self.bragg_diagnostics.set_result(source, data, analysis.parameters)
        else:
            self.result_table.clear()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            self.bragg_diagnostics.clear()
        self.diagnostic_summary.setPlainText(json.dumps(diagnostic_payload, indent=2, sort_keys=True))
        self.provenance.setPlainText(
            json.dumps(
                {
                    "recipe_hash": result.recipe_hash,
                    "input_fingerprints": result.input_fingerprints,
                    "created_at": result.created_at,
                    "duration_seconds": result.duration_seconds,
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _analysis_primary_data(self, analysis: AnalysisEntry) -> MDHistoData | None:
        if self.group is None or not analysis.input_dataset_ids:
            return None
        dataset = next(
            (item for item in self.group.iter_datasets() if item.id == analysis.input_dataset_ids[0]),
            None,
        )
        if dataset is None:
            return None
        data = self._primary_analysis_data(dataset)
        return data if isinstance(data, MDHistoData) else None

    def _populate_result_table(self, data: PointListData) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        preferred = [
            "Accepted", "H", "K", "L", "I", "dI", "I/dI", "Background",
            "Coverage", "BackgroundCoverage", "Status", "FitSigma1", "FitSigma2",
            "FitSigma3", "ReducedChi2",
        ]
        columns = [name for name in preferred if name in data.columns]
        columns.extend(name for name in data.column_names if name not in columns and not name.startswith("Nominal"))
        self.result_table.setSortingEnabled(False)
        self.result_table.clear()
        self.result_table.setRowCount(data.size)
        self.result_table.setColumnCount(len(columns) + (1 if "Status" in data.columns else 0))
        headers = [*columns]
        if "Status" in data.columns:
            headers.append("Rejection reason")
        self.result_table.setHorizontalHeaderLabels(headers)
        accepted = data.column("Accepted") if "Accepted" in data.columns else np.ones(data.size)
        status = data.column("Status") if "Status" in data.columns else np.zeros(data.size)
        status_bits = data.metadata.get("status_bits", {})
        for row in range(data.size):
            for column, name in enumerate(columns):
                value = float(data.column(name)[row])
                text = "Yes" if name == "Accepted" and value else "No" if name == "Accepted" else _format_table_value(value)
                item = NumericTableItem(text, value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                if not bool(accepted[row]):
                    item.setForeground(QtGui.QColor("#d94b45"))
                self.result_table.setItem(row, column, item)
            if "Status" in data.columns:
                reason = _bragg_status_text(int(status[row]), status_bits)
                item = QtWidgets.QTableWidgetItem(reason)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                if int(status[row]):
                    item.setForeground(QtGui.QColor("#d94b45"))
                self.result_table.setItem(row, len(columns), item)
        self.result_table.setSortingEnabled(True)
        self.result_table.resizeColumnsToContents()
        if data.size:
            self.result_table.selectRow(0)

    def _result_row_selected(self) -> None:
        items = self.result_table.selectedItems()
        if not items:
            return
        source_row = items[0].data(0x0100)
        if source_row is not None:
            self.bragg_diagnostics.select_peak(int(source_row))

    def add_current_output_to_datasets(self) -> bool:
        if self.group is None or self._current_result_data is None or self._current_result_output is None:
            return False
        output = self._current_result_output
        existing = next((item for item in self.group.iter_datasets() if item.id == output.dataset_id), None)
        if existing is not None:
            self.explorer._refresh_tree(select_group=self.group, select_dataset=existing)
            return True
        derived = next((node for node in self.group.subgroups if node.name == "Derived data"), None)
        if derived is None:
            derived = DatasetGroup("Derived data")
            self.group.subgroups.append(derived)
        data_type = str(output.metadata.get("data_type") or "derived_analysis")
        entry = DatasetEntry(
            _unique_output_name(output.label, self.group.dataset_names),
            self._current_result_data,
            kind="analysis",
            data_type=data_type,
            enabled=bool(output.metadata.get("fit_enabled", False)),
            metadata={
                "source_file": output.artifact_path,
                "analysis_artifact_path": output.artifact_path,
                "derived_from_analysis": {
                    "analysis_id": self._selected_analysis().id if self._selected_analysis() else "",
                    "output_key": output.key,
                },
            },
            id=output.dataset_id,
        )
        derived.datasets.append(entry)
        self.explorer._mark_dirty()
        self.explorer._refresh_tree(select_group=self.group, select_dataset=entry)
        return True

    def export_current_bragg_int(self) -> bool:
        """Export accepted reflections as headerless, whitespace-delimited intensity data."""

        from PySide6 import QtWidgets

        data = self._current_result_data
        if not isinstance(data, PointListData):
            return False
        if data.metadata.get("analysis_kind") != "bragg_peak_integration":
            return False
        required = ("H", "K", "L", "I", "dI")
        if any(name not in data.columns for name in required):
            QtWidgets.QMessageBox.warning(
                self.window,
                "Export Bragg reflections",
                "The integrated Bragg table does not contain all of H, K, L, I, and dI.",
            )
            return False
        analysis = self._selected_analysis()
        suggested_name = f"{analysis.name if analysis is not None else 'integrated_bragg_peaks'}.int"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Export integrated Bragg peaks",
            suggested_name,
            "Intensity files (*.int)",
        )
        if not filename:
            return False
        if not filename.lower().endswith(".int"):
            filename += ".int"
        accepted = (
            np.asarray(data.column("Accepted"), dtype=bool)
            if "Accepted" in data.columns
            else np.ones(data.size, dtype=bool)
        )
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                for row in np.flatnonzero(accepted):
                    hkl = [
                        int(np.rint(float(data.column(name)[row])))
                        for name in ("H", "K", "L")
                    ]
                    values = [
                        *hkl,
                        float(data.column("I")[row]),
                        float(data.column("dI")[row]),
                    ]
                    handle.write(" ".join(str(value) for value in values) + "\n")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self.window, "Export Bragg reflections", f"Could not write {filename}: {exc}"
            )
            return False
        return True

    def view_input_with_peaks(self) -> bool:
        analysis = self._selected_analysis()
        data = self._current_result_data
        if analysis is None or self.group is None or not isinstance(data, PointListData):
            return False
        if not analysis.input_dataset_ids:
            return False
        dataset = next(
            (item for item in self.group.iter_datasets() if item.id == analysis.input_dataset_ids[0]),
            None,
        )
        if dataset is None:
            return False
        viewer = self.explorer.open_slice_viewer(
            self.group,
            selected_dataset_name=dataset.name,
            use_composite=False,
        )
        if viewer is None or not hasattr(viewer, "set_bragg_peak_overlay"):
            return False
        viewer.set_bragg_peak_overlay(data, dataset_name=dataset.name)
        return True

    def run(self) -> bool:
        from PySide6 import QtWidgets

        if self.group is None or self.dataset_combo.currentIndex() < 0:
            return False
        if self.explorer.project_path is None:
            if not self.explorer.save_as():
                return False
        dataset_id = self.dataset_combo.currentData()
        dataset = next(item for item in self.group.iter_datasets() if item.id == dataset_id)
        analysis_data = self._primary_analysis_data(dataset)
        if analysis_data is None:
            message = (
                "Could not load the selected dataset for Bragg integration."
                if self.operation_combo.currentData() == "bragg_integration"
                else "Load the selected dataset before running the analysis."
            )
            QtWidgets.QMessageBox.warning(self.window, "Analysis Window", message)
            return False
        if self.operation_combo.currentData() in {
            "curie_weiss_fit", "low_temperature_heat_capacity_fit"
        }:
            # These fits consume the same derived, unit-aware physical channel
            # that the dataset viewer presents.
            from .project_gui import prepared_point_list_data

            try:
                analysis_data = prepared_point_list_data(dataset)
            except (TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self.window, "Point-data analysis", str(exc)
                )
                return False
        if not self._confirm_memory(analysis_data):
            return False
        context = self._analysis_context(dataset)
        data_fingerprint = dataset_entry_fingerprint(dataset, self.group)
        analysis_inputs = [AnalysisInput(dataset.id, dataset.name, analysis_data, context, data_fingerprint)]
        parameters = self._parameters()
        if self.operation_combo.currentData() == "bragg_integration" and parameters.get("peak_source") == "table":
            secondary_id = self.secondary_dataset_combo.currentData()
            secondary = next((item for item in self.group.iter_datasets() if item.id == secondary_id), None)
            if secondary is None or secondary.data is None:
                QtWidgets.QMessageBox.warning(self.window, "Analysis Window", "Select and load a secondary H, K, L peak-table dataset.")
                return False
            parameters["peak_table_dataset_id"] = secondary.id
            analysis_inputs.append(AnalysisInput(secondary.id, secondary.name, secondary.data, context, dataset_entry_fingerprint(secondary, self.group)))
        elif self.operation_combo.currentData() == "bose_elastic_separation":
            secondary_id = self.secondary_dataset_combo.currentData()
            secondary = next(
                (item for item in self.group.iter_datasets() if item.id == secondary_id),
                None,
            )
            if secondary is None:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Bose-Einstein elastic separation",
                    "Select the second-temperature dataset.",
                )
                return False
            secondary_data = self._primary_analysis_data(secondary)
            if secondary_data is None:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Bose-Einstein elastic separation",
                    "Could not load the second-temperature dataset.",
                )
                return False
            analysis_inputs.append(
                AnalysisInput(
                    secondary.id,
                    secondary.name,
                    secondary_data,
                    self._analysis_context(secondary),
                    dataset_entry_fingerprint(secondary, self.group),
                )
            )
        elif self.operation_combo.currentData() == "angle_energy_background":
            selected_ids = list(dict.fromkeys(self.additional_input_ids))
            if dataset.id not in selected_ids:
                selected_ids.insert(0, dataset.id)
            analysis_inputs = []
            for selected_id in selected_ids:
                selected = next(
                    (item for item in self.group.iter_datasets() if item.id == selected_id),
                    None,
                )
                if selected is None:
                    continue
                selected_data = self._primary_analysis_data(selected)
                if selected_data is None:
                    QtWidgets.QMessageBox.warning(
                        self.window,
                        "Angle-energy background",
                        f"Could not load {selected.name!r}.",
                    )
                    return False
                analysis_inputs.append(
                    AnalysisInput(
                        selected.id,
                        selected.name,
                        selected_data,
                        self._analysis_context(selected),
                        dataset_entry_fingerprint(selected, self.group),
                    )
                )
        analysis = self._selected_analysis()
        is_new = analysis is None
        if analysis is None:
            analysis = AnalysisEntry(self.name_edit.text().strip() or "Analysis", self.operation_combo.currentData(), [item.dataset_id for item in analysis_inputs], parameters)
        else:
            analysis.name = self.name_edit.text().strip() or analysis.name
            analysis.type = self.operation_combo.currentData()
            analysis.input_dataset_ids = [item.dataset_id for item in analysis_inputs]
            analysis.parameters = parameters

        def task(progress_callback):
            return execute_to_artifacts(analysis, analysis_inputs, self.explorer.project_path, progress_callback=progress_callback)

        def success(result):
            analysis.result = result
            if is_new:
                self.group.analyses.append(analysis)
            derived = next((node for node in self.group.subgroups if node.name == "Derived data"), None)
            if derived is None:
                derived = DatasetGroup("Derived data")
                self.group.subgroups.append(derived)
            derived.datasets[:] = [dataset for dataset in derived.datasets if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id") != analysis.id]
            for output in result.outputs:
                if output.kind == "dataset" and output.artifact_path and output.dataset_id:
                    path = self.explorer.project_path.parent / output.artifact_path
                    data = read_dataset_artifact(path)
                    derived.datasets.append(
                        DatasetEntry(
                            _unique_output_name(output.label, self.group.dataset_names),
                            data,
                            kind="analysis",
                            data_type=str(output.metadata.get("data_type") or "derived_analysis"),
                            enabled=bool(output.metadata.get("fit_enabled", False)),
                            metadata={
                                "source_file": output.artifact_path,
                                "analysis_artifact_path": output.artifact_path,
                                "derived_from_analysis": {
                                    "analysis_id": analysis.id,
                                    "output_key": output.key,
                                    "recipe_hash": result.recipe_hash,
                                },
                            },
                            id=output.dataset_id,
                        )
                    )
            from .project_gui import _link_group_backgrounds

            _link_group_backgrounds(self.group)
            self._render_analysis_result(analysis)
            self.explorer._mark_dirty()
            self._refresh_analysis_list(analysis.id)
            self.explorer._refresh_tree(select_group=self.group)
            return True

        def completion_summary(result):
            diagnostics = dict(getattr(result, "diagnostics", {}))
            if analysis.type != "bragg_integration":
                return []
            generated = int(diagnostics.get("generated_peaks", 0))
            integrated = int(diagnostics.get("integrated_peaks", 0))
            accepted = int(diagnostics.get("accepted_peaks", 0))
            rejected = int(diagnostics.get("rejected_peaks", 0))
            return [
                f"Generated {generated} crystallographic reflections.",
                f"Integrated {integrated}: {accepted} accepted, {rejected} rejected.",
            ]

        return self.explorer._start_background_task(
            title=f"Preparing {analysis_definition(analysis.type).label.lower()}...",
            failure_title="Analysis failed",
            task=task,
            on_success=success,
            success_message="Analysis complete.",
            close_on_success=False,
            completion_summary=completion_summary,
            progress_window_title="Analysis progress",
        )

    def _primary_analysis_data(self, dataset: DatasetEntry) -> Any | None:
        """Return primary input data, loading a lazy Bragg source when possible."""

        if dataset.data is not None:
            return dataset.data
        from .project_gui import dataset_for_slice_viewer

        try:
            loaded = dataset_for_slice_viewer(dataset)
        except (OSError, TypeError, ValueError):
            return None
        return dataset.data if dataset.data is not None else loaded

    def _analysis_context(self, dataset: DatasetEntry) -> AnalysisContext:
        return AnalysisContext(
            self.group.name,
            self.group.lattice_parameters,
            self.group.spacegroup,
            self.group.metadata.get("crystal"),
            dataset.parameters.get("temperature"),
            {"dataset_metadata": dataset.metadata},
        )

    def _confirm_memory(self, data: Any) -> bool:
        from PySide6 import QtWidgets

        signal = getattr(data, "signal", None)
        if not isinstance(signal, np.ndarray):
            return True
        try:
            import psutil
            available = int(psutil.virtual_memory().available)
        except ImportError:
            return True
        multiplier = 12 if self.operation_combo.currentData() == "bragg_integration" else 8
        estimate = int(signal.nbytes * multiplier)
        if estimate <= 0.7 * available:
            return True
        answer = QtWidgets.QMessageBox.question(
            self.window, "Analysis memory estimate",
            f"This operation may use about {estimate / 1024**3:.1f} GB, more than 70% of the currently available {available / 1024**3:.1f} GB. Continue?",
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes


def _unique_output_name(base: str, existing: list[str]) -> str:
    if base not in existing:
        return base
    index = 1
    while f"{base} {index}" in existing:
        index += 1
    return f"{base} {index}"


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
        "edge_policy": "Quality filters",
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
