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
from .pipeline import DatasetEntry, DatasetGroup
from .quantities import display_unit


class DataPlaygroundWindow:
    """Registry-driven non-fitting analysis workbench for one project explorer."""

    def __init__(self, explorer: Any) -> None:
        from PySide6 import QtWidgets

        self.explorer = explorer
        self.group = None
        self.parameter_widgets: dict[str, Any] = {}
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
        self.operation_combo = QtWidgets.QComboBox()
        for key in available_analysis_types():
            self.operation_combo.addItem(analysis_definition(key).label, key)
        self.operation_combo.setToolTip("Choose the dataset operation to configure.")
        self.name_edit = QtWidgets.QLineEdit("Analysis")
        self.name_edit.setToolTip("Editable name stored with this analysis recipe.")
        selectors.addWidget(self.dataset_combo, 2)
        selectors.addWidget(self.secondary_dataset_combo, 2)
        selectors.addWidget(self.operation_combo, 2)
        selectors.addWidget(self.name_edit, 2)
        layout.addLayout(selectors)
        self.parameter_scroll = QtWidgets.QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_panel = QtWidgets.QWidget()
        self.parameter_form = QtWidgets.QFormLayout(self.parameter_panel)
        self.parameter_scroll.setWidget(self.parameter_panel)
        layout.addWidget(self.parameter_scroll, 4)
        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.setToolTip("Inspect analysis outputs, diagnostics, and provenance.")
        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        self.results.setToolTip("Latest result, warnings, and output artifacts.")
        self.diagnostics = QtWidgets.QTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setToolTip("Numerical coverage, fit, and quality diagnostics.")
        self.provenance = QtWidgets.QTextEdit()
        self.provenance.setReadOnly(True)
        self.provenance.setToolTip("Recipe hash and input fingerprints for this result.")
        self.result_tabs.addTab(self.results, "Results")
        self.result_tabs.addTab(self.diagnostics, "Diagnostics")
        self.result_tabs.addTab(self.provenance, "Provenance")
        layout.addWidget(self.result_tabs, 2)
        commands = QtWidgets.QHBoxLayout()
        commands.addStretch(1)
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Run this recipe without modifying the input dataset.")
        commands.addWidget(self.run_button)
        layout.addLayout(commands)
        self.operation_combo.currentIndexChanged.connect(self._rebuild_parameters)
        self.analysis_combo.currentIndexChanged.connect(self._select_analysis)
        self.new_button.clicked.connect(self.new_analysis)
        self.duplicate_button.clicked.connect(self.duplicate_analysis)
        self.delete_button.clicked.connect(self.delete_analysis)
        self.run_button.clicked.connect(self.run)
        self._rebuild_parameters()

    def show(self) -> None:
        self.window.show()

    def raise_(self) -> None:
        self.window.raise_()

    def select_group(self, group: Any, *, dataset: DatasetEntry | None = None) -> None:
        self.group = group
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
        self._set_parameter_values(analysis.parameters)

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
                widget.setText(json.dumps(value))

    def new_analysis(self) -> None:
        self.analysis_combo.setCurrentIndex(0)
        self.name_edit.setText("Analysis")
        self._rebuild_parameters()

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
        self.group.analyses.remove(analysis)
        for subgroup in self.group.subgroups:
            subgroup.datasets[:] = [dataset for dataset in subgroup.datasets if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id") != analysis.id]
        self._refresh_analysis_list()
        self.results.clear()
        self.explorer._mark_dirty()
        self.explorer._refresh_tree(select_group=self.group)

    def _rebuild_parameters(self) -> None:
        from PySide6 import QtWidgets

        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self.parameter_widgets.clear()
        key = self.operation_combo.currentData()
        if not key:
            return
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
                widget = QtWidgets.QLineEdit(json.dumps(parameter.default))
            widget.setToolTip(f"{parameter.description}\nAllowed: {parameter.allowed}\nExample: {parameter.example}")
            self.parameter_form.addRow(parameter.label, widget)
            self.parameter_widgets[parameter.name] = widget

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
                try:
                    values[name] = json.loads(text)
                except json.JSONDecodeError:
                    values[name] = text
        return values

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
        if self.operation_combo.currentData() == "curie_weiss_fit":
            # Curie-Weiss fitting consumes the same derived, unit-aware
            # susceptibility that the dataset viewer presents.
            from .project_gui import prepared_point_list_data

            try:
                analysis_data = prepared_point_list_data(dataset)
            except (TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self.window, "Curie-Weiss fit", str(exc)
                )
                return False
        if not self._confirm_memory(analysis_data):
            return False
        context = AnalysisContext(self.group.name, self.group.lattice_parameters, self.group.spacegroup, self.group.metadata.get("crystal"), dataset.parameters.get("temperature"), {"dataset_metadata": dataset.metadata})
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
            lines = []
            for output in result.outputs:
                lines.append(f"{output.label}: {output.kind}")
                if output.artifact_path and output.dataset_id:
                    path = self.explorer.project_path.parent / output.artifact_path
                    data = read_dataset_artifact(path)
                    derived.datasets.append(DatasetEntry(_unique_output_name(output.label, self.group.dataset_names), data, kind="analysis", data_type="derived_analysis", metadata={"source_file": output.artifact_path, "analysis_artifact_path": output.artifact_path, "derived_from_analysis": {"analysis_id": analysis.id, "output_key": output.key, "recipe_hash": result.recipe_hash}}, id=output.dataset_id))
                elif output.scalar_value is not None:
                    lines[-1] += f" = {output.scalar_value:.6g}"
                    if output.scalar_uncertainty is not None:
                        lines[-1] += f" +/- {output.scalar_uncertainty:.2g}"
                    if output.unit:
                        lines[-1] += f" {display_unit(output.unit)}"
            self.results.setPlainText("\n".join(lines + result.warnings))
            self.diagnostics.setPlainText(json.dumps(result.diagnostics, indent=2, sort_keys=True))
            self.provenance.setPlainText(json.dumps({"recipe_hash": result.recipe_hash, "input_fingerprints": result.input_fingerprints}, indent=2, sort_keys=True))
            self.explorer._mark_dirty()
            self._refresh_analysis_list(analysis.id)
            self.explorer._refresh_tree(select_group=self.group)
            return True

        return self.explorer._start_background_task(title=f"Running {analysis.name}", failure_title="Analysis failed", task=task, on_success=success, success_message="Analysis complete")

    def _primary_analysis_data(self, dataset: DatasetEntry) -> Any | None:
        """Return primary input data, loading a lazy Bragg source when possible."""

        if dataset.data is not None:
            return dataset.data
        if self.operation_combo.currentData() != "bragg_integration":
            return None
        from .project_gui import dataset_for_slice_viewer

        try:
            loaded = dataset_for_slice_viewer(dataset)
        except (OSError, TypeError, ValueError):
            return None
        return dataset.data if dataset.data is not None else loaded

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
