"""Read-only statistical provenance presentation."""

from .measurement_diagnostics import measurement_diagnostics_text


def show_measurement_diagnostics(parent, data):
    """Inspect metadata without materializing a lazy histogram volume."""
    from PySide6 import QtWidgets

    text = measurement_diagnostics_text(data)
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Statistics and provenance")
    dialog.resize(650, 550)
    layout = QtWidgets.QVBoxLayout(dialog)
    report = QtWidgets.QPlainTextEdit(text)
    report.setReadOnly(True)
    report.setToolTip("Available statistical assumptions, units and provenance; numerical arrays are not loaded.")
    layout.addWidget(report)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    copy = buttons.addButton("Copy report", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
    copy.setToolTip("Copy the same metadata report returned by measurement_diagnostics_text(data).")
    copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text))
    layout.addWidget(buttons)
    dialog.exec()


def build_measurement_diagnostics_button(parent, data_provider):
    from PySide6 import QtWidgets

    button = QtWidgets.QPushButton("Statistics and provenance", parent)
    button.setObjectName("measurement_diagnostics_button")
    button.setToolTip("Inspect statistical targets, retained count statistics, exposure units, uncertainty assumptions and interval availability without reading the full volume.")
    button.clicked.connect(lambda: show_measurement_diagnostics(parent, data_provider()))
    return button
