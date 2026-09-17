"""Performance preferences and cancellable benchmark presentation."""

from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from .file_dialogs import get_save_file_name
from .performance import load_performance_settings, save_performance_settings
from .performance_benchmark import (
    BenchmarkCancelled,
    benchmark_candidates,
    benchmark_rebin,
    export_benchmark_script,
)


class _BenchmarkWorker(QtCore.QThread):
    row = QtCore.Signal(object)
    outcome = QtCore.Signal(object)

    def __init__(self, project, target, candidates, parent):
        super().__init__(parent)
        self.project = project
        self.target = target
        self.candidates = candidates
        self.cancel = threading.Event()

    def run(self):
        try:
            result = benchmark_rebin(self.project, **self.target,
                                     candidates=self.candidates,
                                     cancel=self.cancel.is_set, progress=self.row.emit)
        except BenchmarkCancelled:
            result = "Cancelled. No settings changed."
        except Exception as exc:
            result = f"Benchmark failed: {exc}"
        self.outcome.emit(result)


class BenchmarkDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, project=None, target=None, apply=None):
        super().__init__(parent)
        self.setWindowTitle("Machine calibration" if project is None else "Benchmark this rebin")
        self.resize(720, 460)
        self.project = project
        self.target = target or {}
        self.apply = apply
        self.worker = None
        self.result = None
        self.candidates = benchmark_candidates()
        memory_targets = sorted({row["max_batch_mb"] for row in self.candidates})
        worker_targets = sorted({row["workers"] for row in self.candidates})
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            f"Tests batch targets of {', '.join(map(str, memory_targets))} MiB "
            f"with worker ceilings of {', '.join(map(str, worker_targets))}. "
            "The range is scaled to this process's detected CPU and available-memory allocation. "
            "Each trial warms up once, then times two runs in a separate process. "
            "This may take several minutes and use substantial CPU and memory. "
            "Peak memory includes the whole trial process, not the live application.\n\n"
            + ("Uses representative 3-D hard and 4-D fractional grids; recommendations are defaults, not universal optima."
               if project is None else
               "Uses the full selected rebin, including its axes, masks and integration settings. "
               "The trial enables rebinning even if currently unchecked. "
               "Live data and caches are not changed. Source loading is warmed; this is not a cold-disk benchmark.")
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Batch (MiB)", "Worker ceiling", "Median (s)", "Process peak (MiB)"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setToolTip("Completed trials; times exclude warmup and process startup. Peak memory includes warmup.")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.status = QtWidgets.QLabel("Ready. Nothing is applied automatically.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton("Start benchmark")
        self.start.setToolTip("Run isolated trials without modifying data or settings.")
        self.start.clicked.connect(self._start)
        row.addWidget(self.start)
        self.apply_button = QtWidgets.QPushButton("Apply recommendation")
        self.apply_button.setToolTip("Apply only the recommended batch target and worker ceiling to this configuration or machine defaults.")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        row.addWidget(self.apply_button)
        self.export = QtWidgets.QPushButton("Save benchmark script…")
        self.export.setToolTip("Save editable Python; real rebins also save an adjacent project snapshot. Original source files remain required.")
        self.export.clicked.connect(self._export)
        row.addWidget(self.export)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setToolTip("Cancel any active trial and close without applying a recommendation.")
        self.close_button.clicked.connect(self.reject)
        row.addWidget(self.close_button)
        layout.addLayout(row)

    def _start(self):
        self.result = None
        self.table.setRowCount(0)
        self.apply_button.setEnabled(False)
        self.start.setEnabled(False)
        self.export.setEnabled(False)
        self.close_button.setText("Cancel")
        self.status.setText("Running isolated trials…")
        self.worker = _BenchmarkWorker(
            self.project, self.target, self.candidates, self
        )
        self.worker.row.connect(self._row)
        self.worker.outcome.connect(self._outcome)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _row(self, result):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, key in enumerate(("max_batch_mb", "workers", "seconds", "peak_mib")):
            value = result[key]
            text = f"{value:.3f}" if isinstance(value, float) else f"{int(value):,}"
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(text))

    def _outcome(self, result):
        if isinstance(result, dict):
            self.result = result
            recommendation = result["recommendation"]
            self.status.setText(f"Recommended: {int(recommendation['max_batch_mb']):,} MiB, "
                                f"up to {int(recommendation['workers']):,} workers. "
                                "Prefers fewer resources within 5% of the fastest result. Not yet applied.")
        else:
            self.status.setText(result)

    def _finished(self):
        self.start.setEnabled(True)
        self.export.setEnabled(True)
        self.close_button.setText("Close")
        self.apply_button.setEnabled(self.result is not None)
        if self.worker.cancel.is_set():
            super().reject()

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel.set()
            self.status.setText("Cancelling trial…")
            self.close_button.setEnabled(False)
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)

    def _apply(self):
        try:
            self.apply(self.result["recommendation"])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply recommendation", str(exc))
            return
        self.accept()

    def _export(self):
        path, _ = get_save_file_name(self, "Save benchmark script", "rebin_benchmark.py", "Python (*.py)")
        if path:
            try:
                export_benchmark_script(path, self.project, **self.target)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Export benchmark", str(exc))


class PerformancePage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        description = QtWidgets.QLabel("Defaults for new rebin configurations. Existing dataset and saved-plot settings are preserved.")
        description.setWordWrap(True)
        form.addRow(description)
        settings = load_performance_settings()
        self.batch = QtWidgets.QSpinBox()
        self.batch.setObjectName("preferences_batch_mb")
        self.batch.setRange(0, 1_048_576)
        self.batch.setSpecialValueText("Auto (192 MiB)")
        self.batch.setSuffix(" MiB")
        self.batch.setValue(settings["max_batch_mb"])
        self.batch.setToolTip("Temporary rebin batch-memory target, not a total RAM limit. Zero uses 192 MiB.")
        form.addRow("Default batch target", self.batch)
        self.workers = QtWidgets.QSpinBox()
        self.workers.setObjectName("preferences_workers")
        self.workers.setRange(0, 4096)
        self.workers.setSpecialValueText("Auto")
        self.workers.setValue(settings["workers"])
        self.workers.setToolTip("Maximum rebin workers. Auto uses the nfit CPU allocation; each rebin may use fewer workers.")
        form.addRow("Default worker ceiling", self.workers)
        self.transient_memory = QtWidgets.QSpinBox()
        self.transient_memory.setObjectName("preferences_transient_memory_percent")
        self.transient_memory.setRange(0, 80)
        self.transient_memory.setSpecialValueText("Auto (25%)")
        self.transient_memory.setSuffix("% available")
        self.transient_memory.setValue(settings["transient_memory_percent"])
        self.transient_memory.setToolTip(
            "Maximum share of currently available RAM for all cached bin results "
            "combined and for rebin working memory. Higher values retain more "
            "results and can enable more CPUs, but leave less memory for other work."
        )
        form.addRow("Total rebin memory ceiling", self.transient_memory)
        save = QtWidgets.QPushButton("Save defaults")
        save.setToolTip("Persist these defaults for new rebin configurations; no existing settings are overwritten.")
        save.clicked.connect(self._save)
        form.addRow(save)
        calibrate = QtWidgets.QPushButton("Calibrate this machine…")
        calibrate.setToolTip("Compare representative synthetic rebins. Review timing and memory before applying recommended defaults.")
        calibrate.clicked.connect(self._calibrate)
        form.addRow(calibrate)
        self.status = QtWidgets.QLabel("Changes take effect only after Save defaults or Apply recommendation.")
        self.status.setWordWrap(True)
        form.addRow(self.status)

    def _save(self):
        try:
            save_performance_settings(
                max_batch_mb=self.batch.value(),
                workers=self.workers.value(),
                transient_memory_percent=self.transient_memory.value(),
            )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save preferences", str(exc))
            return
        self.status.setText("Defaults saved. Existing rebin configurations are unchanged.")

    def _calibrate(self):
        def apply(values):
            save_performance_settings(
                **values,
                transient_memory_percent=self.transient_memory.value(),
            )
            self.batch.setValue(values["max_batch_mb"])
            self.workers.setValue(values["workers"])
            self.status.setText("Calibrated defaults saved. Existing configurations are unchanged.")
        BenchmarkDialog(self, apply=apply).exec()
