from __future__ import annotations

import json
from typing import Any

import numpy as np

from .analysis import (
    AnalysisEntry,
    analysis_definition,
    available_analysis_types,
    default_analysis_parameters,
    prepare_analysis_input,
    upsert_analysis_output_dataset,
)
from .analysis.artifacts import read_project_dataset_artifact
from .analysis.runner import (
    analysis_source_choices,
    execute_to_artifacts,
    prepare_analysis_source,
)
from .analysis_bragg_gui import (
    BraggPeakDiagnosticsWidget,
    NumericTableItem,
    _bragg_parameter_group,
    _bragg_status_text,
    _format_table_value,
)
from .analysis_window_builder import build_data_playground_window
from .dataset import PointData4D, PointListData
from .mdhisto import MDHistoData
from .pipeline import DatasetEntry
from .quantities import display_unit


class DataPlaygroundWindow:
    """Registry-driven non-fitting analysis workbench for one project explorer."""

    def __init__(self, explorer: Any) -> None:
        self.explorer = explorer
        self.group = None
        self.parameter_widgets: dict[str, Any] = {}
        self.parameter_rows: dict[str, tuple[Any, Any]] = {}
        self._current_result_data: PointListData | MDHistoData | None = None
        self._current_result_output = None
        self.additional_input_ids: list[str] = []
        build_data_playground_window(
            self,
            explorer,
            available_analysis_types=available_analysis_types,
            analysis_definition=analysis_definition,
            diagnostics_factory=BraggPeakDiagnosticsWidget,
        )
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
        for label, source_id in analysis_source_choices(group):
            self.dataset_combo.addItem(label, source_id)
            self.secondary_dataset_combo.addItem(label, source_id)
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
        try:
            return read_project_dataset_artifact(
                self.explorer.project_path,
                output.artifact_path,
            )
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
            "Accepted", "H", "K", "L", "I", "dI", "I/dI",
            "FitWindowRaw", "FitWindowBackground", "FitWindowPeak", "Background",
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
        analysis = self._selected_analysis()
        if analysis is None:
            return False
        entry = upsert_analysis_output_dataset(
            self.group,
            analysis,
            output,
            self._current_result_data,
            project_path=self.explorer.project_path,
        )
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
        if self.operation_combo.currentData() in {
            "dataset_clone",
            "histogram_arithmetic",
        }:
            return self._save_live_derived_recipe()
        dataset_id = self.dataset_combo.currentData()
        try:
            primary_input = prepare_analysis_source(self.group, dataset_id)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            message = (
                f"Could not prepare the selected dataset for Bragg integration:\n{exc}"
                if self.operation_combo.currentData() == "bragg_integration"
                else f"Could not prepare the selected dataset:\n{exc}"
            )
            QtWidgets.QMessageBox.warning(self.window, "Analysis Window", message)
            return False
        analysis_data = primary_input.data
        if not self._confirm_memory(analysis_data):
            return False
        analysis_inputs = [primary_input]
        parameters = self._parameters()
        if self.operation_combo.currentData() == "bragg_integration" and parameters.get("peak_source") == "table":
            secondary_id = self.secondary_dataset_combo.currentData()
            if secondary_id is None:
                QtWidgets.QMessageBox.warning(self.window, "Analysis Window", "Select and load a secondary H, K, L peak-table dataset.")
                return False
            parameters["peak_table_dataset_id"] = str(secondary_id)
            try:
                analysis_inputs.append(prepare_analysis_source(self.group, secondary_id))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Analysis Window",
                    f"Could not prepare the secondary peak table:\n{exc}",
                )
                return False
        elif self.operation_combo.currentData() in {
            "bose_elastic_separation",
            "histogram_arithmetic",
        }:
            secondary_id = self.secondary_dataset_combo.currentData()
            if secondary_id is None:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    analysis_definition(self.operation_combo.currentData()).label,
                    "Select the right-hand input dataset or live composite.",
                )
                return False
            try:
                secondary_input = prepare_analysis_source(self.group, secondary_id)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    analysis_definition(self.operation_combo.currentData()).label,
                    f"Could not prepare the right-hand input:\n{exc}",
                )
                return False
            analysis_inputs.append(secondary_input)
        elif self.operation_combo.currentData() == "angle_energy_background":
            selected_ids = list(dict.fromkeys(self.additional_input_ids))
            if dataset_id not in selected_ids:
                selected_ids.insert(0, dataset_id)
            analysis_inputs = []
            for selected_id in selected_ids:
                selected = next(
                    (item for item in self.group.iter_datasets() if item.id == selected_id),
                    None,
                )
                if selected is None:
                    continue
                try:
                    selected_input = prepare_analysis_source(self.group, selected.id)
                except (OSError, TypeError, ValueError) as exc:
                    QtWidgets.QMessageBox.warning(
                        self.window,
                        "Angle-energy background",
                        f"Could not prepare {selected.name!r}:\n{exc}",
                    )
                    return False
                analysis_inputs.append(selected_input)
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
            output_dataset_ids = set()
            for output in result.outputs:
                if output.kind == "dataset" and output.artifact_path and output.dataset_id:
                    data = read_project_dataset_artifact(
                        self.explorer.project_path,
                        output.artifact_path,
                    )
                    entry = upsert_analysis_output_dataset(
                        self.group,
                        analysis,
                        output,
                        data,
                        project_path=self.explorer.project_path,
                    )
                    output_dataset_ids.add(entry.id)
            for node in (self.group, *self.group.iter_subgroups()):
                node.datasets[:] = [
                    dataset
                    for dataset in node.datasets
                    if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id")
                    != analysis.id
                    or dataset.id in output_dataset_ids
                ]
            from .project_history import _link_group_backgrounds

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

    def _save_live_derived_recipe(self) -> bool:
        """Save a clone/arithmetic recipe without materializing source bins."""

        from PySide6 import QtWidgets

        if self.group is None:
            return False
        operation = str(self.operation_combo.currentData())
        source_ids = [str(self.dataset_combo.currentData())]
        if operation == "histogram_arithmetic":
            secondary_id = self.secondary_dataset_combo.currentData()
            if secondary_id is None:
                QtWidgets.QMessageBox.warning(
                    self.window,
                    analysis_definition(operation).label,
                    "Select the right-hand input dataset or live composite.",
                )
                return False
            source_ids.append(str(secondary_id))
        parameters = self._parameters()
        analysis = self._selected_analysis()
        if analysis is None:
            analysis = AnalysisEntry(
                self.name_edit.text().strip() or "Derived dataset",
                operation,
                source_ids,
                parameters,
            )
        else:
            analysis.name = self.name_edit.text().strip() or analysis.name
            analysis.type = operation
            analysis.input_dataset_ids = source_ids
            analysis.parameters = parameters

        linked = next(
            (
                dataset
                for dataset in self.group.iter_datasets()
                if dataset.metadata.get("derived_from_analysis", {}).get("analysis_id")
                == analysis.id
            ),
            None,
        )
        existing_config = None
        if linked is not None:
            from .project_gui import DATASET_REBIN_KEY

            candidate = linked.parameters.get(DATASET_REBIN_KEY)
            if isinstance(candidate, dict) and candidate.get("axes"):
                existing_config = candidate

        from .project_gui import create_derived_analysis_dataset
        from .project_history import _link_group_backgrounds

        derived = create_derived_analysis_dataset(
            self.group,
            analysis,
            name=self.name_edit.text().strip() or None,
            rebin_config=existing_config,
        )
        _link_group_backgrounds(self.group)
        self.explorer._record_data_group_state_change(self.group)
        self.explorer._mark_dirty()
        self._refresh_analysis_list(analysis.id)
        self._render_analysis_result(analysis)
        self.explorer._refresh_tree(
            select_group=self.group,
            select_dataset=derived,
        )
        return True

    def _primary_analysis_data(self, dataset: DatasetEntry) -> Any | None:
        """Return the same prepared dataset used by analysis execution."""

        if self.group is None:
            return None
        try:
            return prepare_analysis_input(self.group, dataset).data
        except (OSError, TypeError, ValueError):
            return None

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
