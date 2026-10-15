"""Shared transactional UB setup dialog and compact orientation presentation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .file_dialogs import get_open_file_name, get_save_file_name
from .ub_orientation import (
    lattice_orientation_from_ub,
    read_isaw_ub,
    read_nexus_ub,
    ub_from_lattice_orientation,
    validate_ub_matrix,
    write_isaw_ub,
)

UB_TOOLTIP = (
    "UB excludes 2π and maps column [h, k, l] to Q′ in inverse angstrom. "
    "Mantid/SNS coordinates use incident beam +z and vertical +y."
)


class UBSetupDialog:
    """Edit an isolated UB; only an accepted result changes the caller's state.

    The matrix determines the initial cell and orientation. Editing cell/u/v
    makes those fields authoritative until Calculate or Apply computes a new
    matrix. Direct matrix edits and file imports derive all other fields.
    """

    def __init__(self, parent: Any = None, *, ub: Any, lattice=None, u=None, v=None):
        from PySide6 import QtWidgets

        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("UB setup")
        self.dialog.setObjectName("ub_setup_dialog")
        self.dialog.setMinimumWidth(540)
        self.result = None
        self._updating = False
        self._pending = "matrix"
        outer = QtWidgets.QVBoxLayout(self.dialog)
        form = QtWidgets.QGridLayout()
        self.lattice_edits = {}
        for index, name in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            row, column = divmod(index, 3)
            form.addWidget(QtWidgets.QLabel(name), row, 2 * column)
            edit = QtWidgets.QLineEdit()
            edit.setMaximumWidth(112)
            edit.setObjectName(f"ub_lattice_{name}")
            edit.setToolTip("Unit-cell length in angstrom." if index < 3 else "Unit-cell angle in degrees.")
            edit.textChanged.connect(self._parameters_changed)
            form.addWidget(edit, row, 2 * column + 1)
            self.lattice_edits[name] = edit
        self.orientation_edits = {}
        for row, name in enumerate(("u", "v"), start=2):
            for component in range(3):
                form.addWidget(QtWidgets.QLabel(f"{name}{'hkl'[component]}"), row, 2 * component)
                edit = QtWidgets.QLineEdit()
                edit.setMaximumWidth(112)
                edit.setObjectName(f"ub_orientation_{name}_{component}")
                edit.setToolTip(
                    "Reciprocal-lattice components (h, k, l). u = inverse(UB) × [0,0,1] "
                    "points along the beam; v = inverse(UB) × [1,0,0] lies in the horizontal plane. "
                    "Vectors retain the Shiver/Mantid magnitude."
                )
                edit.textChanged.connect(self._parameters_changed)
                form.addWidget(edit, row, 2 * component + 1)
                self.orientation_edits[name, component] = edit
        outer.addLayout(form)
        sources = QtWidgets.QHBoxLayout()
        for text, name, tooltip, slot in (
            ("UB from NeXus", "ub_load_nexus", "Read UB metadata from a NeXus file without loading its event or histogram arrays.", self._load_nexus),
            ("UB from ISAW", "ub_load_isaw", "Read an ISAW .mat file and convert its transposed IPNS coordinates to Mantid coordinates.", self._load_isaw),
            ("Calculate UB", "ub_calculate", "Calculate UB from the cell and u/v fields. Apply also performs this calculation when those fields have changed.", self._calculate),
        ):
            button = QtWidgets.QPushButton(text)
            button.setObjectName(name)
            button.setToolTip(tooltip)
            button.clicked.connect(slot)
            sources.addWidget(button)
        outer.addLayout(sources)
        caption = QtWidgets.QLabel("UB matrix (Å⁻¹, without 2π)")
        caption.setToolTip(UB_TOOLTIP)
        outer.addWidget(caption)
        self.matrix_table = QtWidgets.QTableWidget(3, 3)
        self.matrix_table.setObjectName("ub_matrix_table")
        self.matrix_table.setToolTip(UB_TOOLTIP + " Editing the matrix updates the cell and orientation vectors.")
        self.matrix_table.horizontalHeader().setVisible(False)
        self.matrix_table.verticalHeader().setVisible(False)
        self.matrix_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.matrix_table.setFixedHeight(108)
        self.matrix_table.cellChanged.connect(self._matrix_changed)
        outer.addWidget(self.matrix_table)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("ub_setup_status")
        self.status.setWordWrap(True)
        self.status.setToolTip("The pending edit determines whether Apply validates the matrix or calculates it from the cell and u/v fields.")
        outer.addWidget(self.status)
        if ub is None:
            initial_cell = lattice or dict(a=1, b=1, c=1, alpha=90, beta=90, gamma=90)
            ub = ub_from_lattice_orientation(
                initial_cell, [1, 0, 0] if u is None else u,
                [0, 1, 0] if v is None else v,
            )
            self._set_matrix(ub)
            self.status.setText("Proposed UB from the unit cell and u/v; Apply installs this orientation. " + UB_TOOLTIP)
        else:
            self._set_matrix(ub)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Apply | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        save = buttons.addButton("Save ISAW", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        save.setObjectName("ub_save_isaw")
        save.setToolTip("Save the current valid setup to an ISAW .mat file, converting to transposed IPNS coordinates.")
        save.clicked.connect(self._save_isaw)
        apply = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply)
        apply.setToolTip("Validate and apply this matrix, cell, and orientation to the selected scope.")
        apply.clicked.connect(self._accept)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip("Close without changing the selected scope.")
        buttons.rejected.connect(self.dialog.reject)
        outer.addWidget(buttons)

    def exec(self):
        return self.dialog.exec()

    @staticmethod
    def _parameter_value(edit):
        retained = edit.property("ub_retained_value")
        return retained[0] if retained is not None and edit.text() == retained[1] else float(edit.text())

    def _lattice(self):
        return {name: self._parameter_value(edit) for name, edit in self.lattice_edits.items()}

    def _vector(self, name):
        return np.asarray([self._parameter_value(self.orientation_edits[name, index]) for index in range(3)])

    @staticmethod
    def _set_parameter(edit, value):
        text = f"{value:.8g}"
        edit.setProperty("ub_retained_value", (float(value), text))
        edit.setText(text)
        edit.setCursorPosition(0)

    def _matrix(self):
        from PySide6 import QtCore

        def value(row, column):
            item = self.matrix_table.item(row, column)
            retained = item.data(QtCore.Qt.ItemDataRole.UserRole)
            # Compact text keeps the editor readable; untouched cells retain
            # their exact numerical values, including values read from files.
            if retained is not None and item.text() == retained[1]:
                return retained[0]
            return float(item.text())

        return validate_ub_matrix([[value(row, column) for column in range(3)] for row in range(3)])

    def _parameters_changed(self, *_args):
        if self._updating:
            return
        self._pending = "parameters"
        self.matrix_table.setEnabled(False)
        self.status.setText("Cell or u/v edits pending. Calculate UB or Apply to update the matrix." + self._handedness_notice())

    def _update_parameters(self, matrix, derived=None):
        lattice, u, v = lattice_orientation_from_ub(matrix) if derived is None else derived
        self._updating = True
        try:
            for name, value in lattice.items():
                self._set_parameter(self.lattice_edits[name], value)
            for name, vector in (("u", u), ("v", v)):
                for index, value in enumerate(vector):
                    self._set_parameter(self.orientation_edits[name, index], value)
        finally:
            self._updating = False

    def _set_matrix(self, matrix):
        from PySide6 import QtCore, QtWidgets

        matrix = validate_ub_matrix(matrix)
        derived = lattice_orientation_from_ub(matrix)
        self._updating = True
        try:
            for row in range(3):
                for column in range(3):
                    value = float(matrix[row, column])
                    text = f"{value:.8g}"
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, (value, text))
                    item.setToolTip(f"UB[{row + 1}, {column + 1}] = {value:.17g} Å⁻¹")
                    self.matrix_table.setItem(row, column, item)
        finally:
            self._updating = False
        self._update_parameters(matrix, derived)
        self._pending = "matrix"
        self.matrix_table.setEnabled(True)
        self.status.setText(UB_TOOLTIP + self._handedness_notice())

    def _handedness_notice(self):
        try:
            negative = np.linalg.det(self._matrix()) < 0
        except (TypeError, ValueError):
            return ""
        return (
            " This matrix is left-handed. Calculating from cell and u/v replaces it with a right-handed orientation."
            if negative else ""
        )

    def _matrix_changed(self, *_args):
        if self._updating:
            return
        self._pending = "matrix"
        try:
            self._update_parameters(self._matrix())
            self.status.setText(UB_TOOLTIP + self._handedness_notice())
        except (TypeError, ValueError) as error:
            self.status.setText(f"Invalid UB matrix: {error}")

    def _resolved_matrix(self):
        if self._pending == "parameters":
            self._set_matrix(ub_from_lattice_orientation(self._lattice(), self._vector("u"), self._vector("v")))
        return self._matrix()

    def _warning(self, message, error):
        from PySide6 import QtWidgets

        QtWidgets.QMessageBox.warning(self.dialog, "UB setup", f"{message}:\n{error}")

    def _calculate(self):
        try:
            self._set_matrix(ub_from_lattice_orientation(self._lattice(), self._vector("u"), self._vector("v")))
            return True
        except (TypeError, ValueError) as error:
            self._warning("Could not calculate UB", error)
            return False

    def _load_isaw(self):
        path, _ = get_open_file_name(self.dialog, "Load ISAW UB", "", "ISAW matrix (*.mat);;All files (*)")
        if path:
            try:
                self._set_matrix(read_isaw_ub(path)[0])
            except (OSError, TypeError, ValueError) as error:
                self._warning("Could not read ISAW file", error)

    def _load_nexus(self):
        path, _ = get_open_file_name(self.dialog, "Load UB from NeXus", "", "NeXus/HDF5 (*.nxs *.nxs.h5 *.nx5 *.h5 *.hdf5);;All files (*)")
        if path:
            try:
                self._set_matrix(read_nexus_ub(path)[0])
            except (OSError, TypeError, ValueError) as error:
                self._warning("Could not read NeXus orientation", error)

    def _save_isaw(self):
        try:
            matrix = self._resolved_matrix()
            path, _ = get_save_file_name(self.dialog, "Save ISAW UB", "UB.mat", "ISAW matrix (*.mat)")
            if path:
                write_isaw_ub(path, matrix)
        except (OSError, TypeError, ValueError) as error:
            self._warning("Could not save ISAW file", error)

    def _accept(self):
        try:
            matrix = self._resolved_matrix()
            lattice, u, v = lattice_orientation_from_ub(matrix)
            self.result = {"ub_matrix": matrix.tolist(), "lattice_parameters": lattice, "u": u.tolist(), "v": v.tolist()}
        except (TypeError, ValueError) as error:
            self._warning("Invalid UB setup", error)
            return
        self.dialog.accept()


def ub_matrix_widget(
    ub, *, parent=None, object_name="ub", on_changed=None, on_open=None,
    allow_source=False, to_editor=validate_ub_matrix, from_editor=validate_ub_matrix,
    source_editor_ub=None,
):
    """Show canonical Mantid UB while retaining the caller's explicit frame.

    The adapters convert matrices only; callbacks receive the caller's native
    convention. ``source_editor_ub`` is an optional canonical Mantid source
    matrix supplied by the owning service, used only without an explicit UB.
    No matrix conversion is inferred from its numerical entries.
    """
    from PySide6 import QtCore, QtGui, QtWidgets

    widget = QtWidgets.QWidget(parent)
    widget.setObjectName(object_name)
    widget.setToolTip(UB_TOOLTIP)
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    display = QtWidgets.QLabel()
    display.setObjectName("ub_matrix_display" if object_name == "ub" else f"{object_name}_display")
    display.setToolTip(UB_TOOLTIP)
    display.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    display.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    current = ub

    def refresh():
        matrix = source_editor_ub if current is None else to_editor(current)
        if matrix is None:
            display.setText("UB matrix: from source" if allow_source else "UB matrix: not set")
        else:
            matrix = validate_ub_matrix(matrix)
            caption = "UB matrix (Å⁻¹, from source)" if current is None and allow_source else "UB matrix (Å⁻¹)"
            display.setText(caption + "\n" + "\n".join(" ".join(f"{value: .7g}".rjust(13) for value in row) for row in matrix))

    refresh()
    layout.addWidget(display)
    row = QtWidgets.QHBoxLayout()
    button = QtWidgets.QPushButton("UB setup…")
    button.setObjectName("open_ub_setup" if object_name == "ub" else f"{object_name}_setup")
    button.setToolTip("Edit the unit cell, u/v orientation, and UB matrix; import NeXus or ISAW orientation metadata.")

    def open_setup():
        nonlocal current
        if on_open is not None:
            on_open()
            return
        dialog = UBSetupDialog(widget, ub=source_editor_ub if current is None else to_editor(current))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.result is not None:
            value = from_editor(dialog.result["ub_matrix"]).tolist()
            if on_changed is None or on_changed(value) is not False:
                current = value
                refresh()

    button.clicked.connect(open_setup)
    row.addWidget(button)
    if allow_source:
        reset = QtWidgets.QPushButton("Use source UB")
        reset.setObjectName(f"{object_name}_source")
        reset.setToolTip("Remove the explicit UB override and use the orientation recorded in the source file.")

        def use_source():
            nonlocal current
            if on_changed is None or on_changed(None) is not False:
                current = None
                refresh()

        reset.clicked.connect(use_source)
        row.addWidget(reset)
    row.addStretch(1)
    layout.addLayout(row)
    return widget
