"""Qt viewer for electronic Hamiltonian and operator matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from .electronic_matrix import ElectronicMatrixInspection
from .qt_viewer_shell import create_viewer_shell


def _display_values(
    inspection: ElectronicMatrixInspection,
    *,
    contribution: bool,
    component: str,
) -> np.ndarray:
    matrix = (
        inspection.contribution_matrix
        if contribution
        else inspection.basis_matrix
    )
    if component == "real":
        return matrix.real
    if component == "imaginary":
        return matrix.imag
    if component == "magnitude":
        return np.abs(matrix)
    if component == "phase":
        return np.angle(matrix)
    raise ValueError(f"unknown matrix display component {component!r}")


def show_electronic_matrix_catalog(
    catalog: tuple[ElectronicMatrixInspection, ...] | list[ElectronicMatrixInspection],
    *,
    parent: Any | None = None,
) -> Any:
    """Open a matrix heatmap, exact element table, and subspace decomposition."""

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtGui, QtWidgets

    entries = tuple(catalog)
    if not entries:
        raise ValueError("matrix viewer requires at least one inspection")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow(parent)
    window.setObjectName("electronic_matrix_viewer")
    window.setWindowTitle("Electronic matrix inspector")
    central, viewport_layout, settings = create_viewer_shell(
        QtWidgets,
        viewer_key="electronic_matrix",
    )
    settings.setFixedWidth(380)
    _layout = settings.layout()
    while _layout.count():
        item = _layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()

    tabs = QtWidgets.QTabWidget()
    tabs.setObjectName("electronic_matrix_tabs")
    figure = Figure(figsize=(7.0, 6.0))
    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName("electronic_matrix_canvas")
    canvas.setToolTip(
        "Heatmap of the selected complex matrix component. Row and column "
        "labels follow the ordered electronic basis."
    )
    tabs.addTab(canvas, "Heatmap")
    elements = QtWidgets.QTableWidget()
    elements.setObjectName("electronic_matrix_elements")
    elements.setToolTip(
        "Exact complex matrix elements. Values use a+bi notation and the "
        "selected basis or coefficient contribution."
    )
    elements.setEditTriggers(
        QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    )
    tabs.addTab(elements, "Matrix elements")
    viewport_layout.addWidget(tabs)
    window.setCentralWidget(central)

    selection_group = QtWidgets.QGroupBox("Matrix")
    selection_layout = QtWidgets.QFormLayout(selection_group)
    matrix_combo = QtWidgets.QComboBox()
    matrix_combo.setObjectName("electronic_matrix_selection")
    matrix_combo.setToolTip(
        "Select the resolved Hamiltonian, a named parameter, a spin operator, "
        "or a representative onsite/hopping matrix."
    )
    for index, entry in enumerate(entries):
        matrix_combo.addItem(entry.title, index)
    coefficient_mode = QtWidgets.QComboBox()
    coefficient_mode.setObjectName("electronic_matrix_coefficient_mode")
    coefficient_mode.setToolTip(
        "Show the normalized/dimensionless basis matrix or multiply it by its "
        "current energy coefficient."
    )
    coefficient_mode.addItem("Coefficient contribution", True)
    coefficient_mode.addItem("Matrix basis", False)
    component_combo = QtWidgets.QComboBox()
    component_combo.setObjectName("electronic_matrix_component")
    component_combo.setToolTip(
        "Choose which scalar component of the complex matrix appears in the heatmap."
    )
    component_combo.addItem("Real part", "real")
    component_combo.addItem("Imaginary part", "imaginary")
    component_combo.addItem("Magnitude", "magnitude")
    component_combo.addItem("Phase", "phase")
    description = QtWidgets.QLabel()
    description.setObjectName("electronic_matrix_description")
    description.setWordWrap(True)
    description.setToolTip(
        "Definition of the selected matrix and its coefficient convention."
    )
    selection_layout.addRow("Object", matrix_combo)
    selection_layout.addRow("Values", coefficient_mode)
    selection_layout.addRow("Heatmap", component_combo)
    selection_layout.addRow(description)
    _layout.addWidget(selection_group)

    decomposition_group = QtWidgets.QGroupBox("Orbital-subspace decomposition")
    decomposition_group.setObjectName("electronic_matrix_decomposition_group")
    decomposition_layout = QtWidgets.QVBoxLayout(decomposition_group)
    decomposition = QtWidgets.QTableWidget()
    decomposition.setObjectName("electronic_matrix_decomposition")
    decomposition.setToolTip(
        "Nonzero P_row A P_column blocks grouped by site, orbital manifold, "
        "and spin, with Frobenius norm and largest matrix element."
    )
    decomposition.setColumnCount(4)
    decomposition.setHorizontalHeaderLabels(
        ("Row ← column", "Shape", "Frobenius", "Maximum")
    )
    decomposition.setEditTriggers(
        QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    )
    decomposition_layout.addWidget(decomposition)
    _layout.addWidget(decomposition_group, 1)

    output_group = QtWidgets.QGroupBox("Output")
    output_layout = QtWidgets.QHBoxLayout(output_group)
    copy_matrix = QtWidgets.QPushButton("Copy matrix")
    copy_matrix.setObjectName("electronic_matrix_copy")
    copy_matrix.setToolTip(
        "Copy the selected exact complex matrix as tab-separated text."
    )
    output_layout.addWidget(copy_matrix)
    _layout.addWidget(output_group)

    state: dict[str, Any] = {"matrix": entries[0].contribution_matrix}

    def redraw() -> None:
        entry = entries[int(matrix_combo.currentData())]
        contribution = bool(coefficient_mode.currentData())
        component = str(component_combo.currentData())
        matrix = (
            entry.contribution_matrix
            if contribution
            else entry.basis_matrix
        )
        state["matrix"] = matrix
        coefficient_text = (
            f"\nCoefficient: {entry.coefficient_label} = "
            f"{entry.coefficient:g} {entry.units}"
            if entry.coefficient_label
            else ""
        )
        description.setText(entry.description + coefficient_text)
        coefficient_mode.setEnabled(bool(entry.coefficient_label))

        figure.clear()
        axis = figure.add_subplot(111)
        values = _display_values(
            entry,
            contribution=contribution,
            component=component,
        )
        if component in {"real", "imaginary"}:
            maximum = float(np.max(np.abs(values), initial=0.0)) or 1.0
            image = axis.imshow(
                values,
                cmap="RdBu_r",
                vmin=-maximum,
                vmax=maximum,
                aspect="auto",
            )
        elif component == "phase":
            image = axis.imshow(
                values,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
                aspect="auto",
            )
        else:
            image = axis.imshow(values, cmap="viridis", aspect="auto")
        if len(entry.column_labels) <= 24:
            axis.set_xticks(
                np.arange(len(entry.column_labels)),
                entry.column_labels,
                rotation=90,
                fontsize=7,
            )
        if len(entry.row_labels) <= 24:
            axis.set_yticks(
                np.arange(len(entry.row_labels)),
                entry.row_labels,
                fontsize=7,
            )
        axis.set_title(entry.title)
        figure.colorbar(image, ax=axis, shrink=0.8)
        figure.tight_layout()
        canvas.draw_idle()

        elements.clear()
        elements.setRowCount(matrix.shape[0])
        elements.setColumnCount(matrix.shape[1])
        elements.setVerticalHeaderLabels(entry.row_labels)
        elements.setHorizontalHeaderLabels(entry.column_labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text = f"{value.real:.8g}{value.imag:+.8g}i"
                elements.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        elements.resizeColumnsToContents()

        blocks = entry.subspace_blocks(contribution=contribution)
        decomposition.setRowCount(len(blocks))
        for row, block in enumerate(blocks):
            values = (
                f"{block.row_subspace} ← {block.column_subspace}",
                f"{block.shape[0]}×{block.shape[1]}",
                f"{block.frobenius_norm:.7g}",
                f"{block.maximum_magnitude:.7g}",
            )
            for column, text in enumerate(values):
                decomposition.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(text),
                )
        decomposition.resizeColumnsToContents()

    def copy_current_matrix() -> None:
        matrix = np.asarray(state["matrix"])
        text = "\n".join(
            "\t".join(
                f"{value.real:.16g}{value.imag:+.16g}j"
                for value in row
            )
            for row in matrix
        )
        QtWidgets.QApplication.clipboard().setText(text)

    matrix_combo.currentIndexChanged.connect(redraw)
    coefficient_mode.currentIndexChanged.connect(redraw)
    component_combo.currentIndexChanged.connect(redraw)
    copy_matrix.clicked.connect(copy_current_matrix)
    copy_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Copy,
        window,
    )
    copy_shortcut.activated.connect(copy_current_matrix)
    close_shortcut = QtGui.QShortcut(
        QtGui.QKeySequence.StandardKey.Close,
        window,
    )
    close_shortcut.activated.connect(window.close)
    window._nfit_application = application
    window._nfit_catalog = entries
    window._nfit_matrix_state = state
    window._nfit_copy_shortcut = copy_shortcut
    window._nfit_close_shortcut = close_shortcut
    redraw()
    window.resize(1280, 820)
    window.show()
    return window
