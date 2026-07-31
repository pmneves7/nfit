"""Qt progress display for observable-specific electronic mesh searches."""

from __future__ import annotations

from typing import Any


class SamplingProgressDialog:
    """Small modeless dialog that records every tested electronic mesh."""

    def __init__(self, title: str, *, parent: Any = None) -> None:
        from PySide6 import QtCore, QtWidgets

        self._running = True

        owner = self

        class _ProgressDialog(QtWidgets.QDialog):
            def closeEvent(self, event: Any) -> None:
                if owner._running:
                    event.ignore()
                else:
                    super().closeEvent(event)

        self.dialog = _ProgressDialog(parent)
        self.dialog.setObjectName("sampling_progress_dialog")
        self.dialog.setWindowTitle(title)
        self.dialog.setWindowModality(
            QtCore.Qt.WindowModality.WindowModal
        )
        self.dialog.resize(820, 360)

        layout = QtWidgets.QVBoxLayout(self.dialog)
        self.status = QtWidgets.QLabel("Preparing the first candidate mesh…")
        self.status.setObjectName("sampling_progress_status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setObjectName("sampling_progress_table")
        self.table.setHorizontalHeaderLabels(
            (
                "Iteration",
                "Mesh",
                "Maximum",
                "L²",
                "Integrated",
                "Largest metric",
                "Result",
                "Runtime",
            )
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for column in (0, 2, 3, 4, 5, 6, 7):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        layout.addWidget(self.table)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.close_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.close_button.setEnabled(False)
        buttons.rejected.connect(self.dialog.close)
        layout.addWidget(buttons)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def update(self, progress: Any) -> None:
        """Display one started or completed candidate-mesh event."""

        from PySide6 import QtWidgets

        row = int(progress.iteration) - 1
        if self.table.rowCount() <= row:
            self.table.setRowCount(row + 1)
        mesh = " × ".join(str(value) for value in progress.mesh)
        values = [
            f"{progress.iteration}/{progress.candidate_count}",
            mesh,
            "—",
            "—",
            "—",
            "—",
            "Running…" if progress.phase == "started" else "Reference",
            "—",
        ]
        comparison = progress.comparison
        if comparison is not None:
            values[2] = f"{comparison.maximum_relative_error:.3g}"
            values[3] = f"{comparison.rms_relative_error:.3g}"
            values[4] = f"{comparison.integrated_relative_error:.3g}"
            values[5] = (
                f"{max(comparison.maximum_relative_error, comparison.rms_relative_error, comparison.integrated_relative_error):.3g}"
            )
            values[6] = "Pass" if comparison.passed else "Refine"
        if progress.phase == "completed":
            values[7] = f"{progress.elapsed_seconds:.2f} s"
        for column, value in enumerate(values):
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        if progress.phase == "started":
            self.status.setText(
                f"Iteration {progress.iteration} of at most "
                f"{progress.candidate_count}: evaluating {mesh}…"
            )
        else:
            self.status.setText(
                f"Completed {mesh} in {progress.elapsed_seconds:.2f} s; "
                f"{progress.consecutive_passes}/{progress.required_passes} "
                "successive refinements currently pass."
            )
        self.table.scrollToBottom()
        QtWidgets.QApplication.processEvents()
        self.dialog.raise_()

    def finish(self, certificate: Any) -> None:
        """Mark the search complete while leaving its record visible."""

        self._running = False
        if certificate.certified:
            mesh = " × ".join(str(value) for value in certificate.chosen_mesh)
            self.status.setText(f"Certified production mesh: {mesh}.")
        else:
            self.status.setText(
                "The configured refinement budget was exhausted without "
                "certifying the requested tolerance."
            )
        self.close_button.setEnabled(True)
        self.dialog.raise_()

    def fail(self, message: str) -> None:
        """Display a failed search without discarding completed iterations."""

        self._running = False
        self.status.setText(f"Mesh search failed: {message}")
        self.close_button.setEnabled(True)
        self.dialog.raise_()
