"""Importer-specific Qt dialogs for project dataset ingestion.

The project explorer owns workflow dispatch; this module owns the presentation
and validation details for importers with structured option forms.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .importers import inspect_powder_ins_csv

__all__ = [
    "prompt_import_choice",
    "prompt_macs_nexus_options",
    "prompt_powder_ins_csv_options",
]


def prompt_import_choice(
    parent: Any,
    *,
    title: str,
    prompt: str,
    choices: list[tuple[str, str]],
    default_index: int = 0,
    object_name: str = "import_choice",
) -> str | None:
    """Show an explicitly raised import choice dialog and return its value."""

    from PySide6 import QtCore, QtWidgets

    if not choices:
        return None
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName(f"{object_name}_dialog")
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(440)
    dialog.setWindowModality(
        QtCore.Qt.WindowModality.WindowModal
        if parent is not None
        else QtCore.Qt.WindowModality.ApplicationModal
    )
    if sys.platform.startswith("linux"):
        dialog.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel(prompt)
    label.setWordWrap(True)
    layout.addWidget(label)

    combo = QtWidgets.QComboBox()
    combo.setObjectName(object_name)
    combo.setToolTip(prompt)
    for choice_label, value in choices:
        combo.addItem(choice_label, value)
    combo.setCurrentIndex(max(0, min(int(default_index), combo.count() - 1)))
    layout.addWidget(combo)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setToolTip(
        "Continue with the selected import option."
    )
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
        "Cancel the dataset import."
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    # Static QInputDialog helpers can create an invisible modal after an
    # external GTK chooser exits under ThinLinc. Realize and raise this window
    # before entering its modal loop so the user can always interact with it.
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    application = QtWidgets.QApplication.instance()
    if application is not None:
        application.processEvents()
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return str(combo.currentData())


def prompt_macs_nexus_options(
    parent: Any,
    paths: list[str | Path],
) -> tuple[dict[str, dict[str, Any]], str] | bool:
    """Configure a batch of MACS files before expanding SPEC and DIFF."""

    from PySide6 import QtGui, QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Import NIST NCNR MACS NeXus")
    dialog.setMinimumWidth(620)
    outer = QtWidgets.QVBoxLayout(dialog)
    explanation = QtWidgets.QLabel(
        "Each file will be imported as separate SPEC (energy analyzed) and "
        "DIFF (unanalysed, elastic-coordinate approximation) datasets."
    )
    explanation.setWordWrap(True)
    outer.addWidget(explanation)
    form = QtWidgets.QFormLayout()

    group_mode = QtWidgets.QComboBox()
    group_mode.setObjectName("macs_stream_group_mode")
    group_mode.addItem("Add to compatible existing SPEC/DIFF groups", "reuse")
    group_mode.addItem("Create new SPEC/DIFF groups", "new")
    group_mode.setToolTip(
        "Choose whether this batch joins existing MACS SPEC and DIFF "
        "collections under the selected parent or starts separate collections."
    )
    form.addRow("Dataset groups", group_mode)

    a3_offset = QtWidgets.QLineEdit()
    a3_offset.setObjectName("macs_a3_offset")
    a3_offset.setValidator(QtGui.QDoubleValidator(-360.0, 360.0, 6, dialog))
    a3_offset.setPlaceholderText("use NeXus sampleState/a3Zero")
    a3_offset.setToolTip(
        "Offset added to the recorded sample A3 angle before converting detector "
        "positions to HKL. Use 66.5° for the orientation shown in the supplied "
        "DAVE/MSlice setup. Leave blank to use sampleState/a3Zero from the file."
    )
    form.addRow("A3 offset (deg)", a3_offset)

    monitor_target = QtWidgets.QDoubleSpinBox()
    monitor_target.setObjectName("macs_monitor_target")
    monitor_target.setRange(1.0, 1.0e12)
    monitor_target.setDecimals(0)
    monitor_target.setValue(1.0e6)
    monitor_target.setToolTip(
        "Normalize counts and Poisson uncertainties to this incident-monitor "
        "count. DAVE/MSlice commonly uses 1,000,000."
    )
    form.addRow("Monitor target", monitor_target)

    efficiency = QtWidgets.QCheckBox("Apply detector efficiency factors")
    efficiency.setObjectName("macs_apply_efficiency")
    efficiency.setChecked(True)
    efficiency.setToolTip(
        "Multiply each stream by the 20 per-channel correction factors stored "
        "in its NeXus detectorEfficiency field."
    )
    form.addRow("Efficiency", efficiency)

    alignment = QtWidgets.QCheckBox("Mask misset analyzer blades")
    alignment.setObjectName("macs_mask_alignment")
    alignment.setChecked(True)
    alignment.setToolTip(
        "Apply DAVE's per-scan analyzer-angle test to SPEC only. DIFF is not "
        "energy analyzed and does not inherit this mask."
    )
    form.addRow("Analyzer alignment", alignment)

    tolerance = QtWidgets.QDoubleSpinBox()
    tolerance.setObjectName("macs_alignment_tolerance")
    tolerance.setRange(0.0, 10.0)
    tolerance.setDecimals(3)
    tolerance.setValue(1.0)
    tolerance.setToolTip(
        "Maximum analyzer two-theta deviation beyond the closest blade. DAVE "
        "uses 1° for files that record analyzer two-theta."
    )
    form.addRow("Alignment tolerance (deg)", tolerance)

    dead = QtWidgets.QCheckBox("Detect unresponsive SPEC analyzer channels")
    dead.setObjectName("macs_detect_dead")
    dead.setChecked(True)
    dead.setToolTip(
        "Mask a SPEC channel when its nonzero-count occupancy is a robust low "
        "outlier across at least 20 scan points. The decision and occupancy are "
        "recorded in dataset provenance."
    )
    form.addRow("Dead-channel test", dead)

    manual = QtWidgets.QLineEdit()
    manual.setObjectName("macs_manual_channels")
    manual.setPlaceholderText("for example: 19")
    manual.setToolTip(
        "Optional comma-separated 1-based SPEC analyzer channels to mask. "
        "These do not mask the corresponding DIFF detectors."
    )
    form.addRow("Additional SPEC masks", manual)
    outer.addLayout(form)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    outer.addWidget(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return False
    offset_text = a3_offset.text().strip()
    shared = {
        "a3_offset_deg": float(offset_text) if offset_text else None,
        "monitor_target": float(monitor_target.value()),
        "apply_detector_efficiency": bool(efficiency.isChecked()),
        "mask_misaligned_analyzers": bool(alignment.isChecked()),
        "analyzer_alignment_tolerance_deg": float(tolerance.value()),
        "detect_dead_analyzers": bool(dead.isChecked()),
        "masked_analyzer_channels": manual.text().strip(),
    }
    options = {str(Path(path)): copy.deepcopy(shared) for path in paths}
    return options, str(group_mode.currentData())


def prompt_powder_ins_csv_options(
    parent: Any,
    paths: list[str | Path],
) -> dict[str, dict[str, Any]] | bool:
    """Configure a batch of digitized powder INS cuts and maps."""

    from PySide6 import QtCore, QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Import powder INS CSV")
    dialog.resize(980, 520)
    outer = QtWidgets.QVBoxLayout(dialog)

    common = QtWidgets.QGroupBox("Signal convention")
    form = QtWidgets.QFormLayout(common)
    observable = QtWidgets.QComboBox()
    observable.setObjectName("powder_csv_observable")
    observable.addItem("Scattering cross section / intensity", "cross_section")
    observable.addItem("Dynamical susceptibility χ″", "chi_double_prime")
    observable.setToolTip(
        "Choose what the digitized y values represent. Temperature then "
        "allows nfit to create the paired cross-section or χ″ channel."
    )
    form.addRow("Imported quantity", observable)

    units = QtWidgets.QComboBox()
    units.setObjectName("powder_csv_units")
    units.setToolTip(
        "Unit before the amount-of-sample denominator. Use normalized "
        "intensity for published 1/meV data that are not cross sections."
    )
    form.addRow("Signal units", units)

    basis = QtWidgets.QComboBox()
    basis.setObjectName("powder_csv_basis")
    for label, value in (
        ("Unknown / none", "unknown"),
        ("Per formula unit", "per_formula_unit"),
        ("Per atom or magnetic ion", "per_magnetic_ion"),
        ("Per unit cell", "per_unit_cell"),
    ):
        basis.addItem(label, value)
    basis.setToolTip(
        "Denominator already present in the digitized values. This labels "
        "the data and does not apply a hidden numerical rescaling."
    )
    form.addRow("Normalization", basis)

    atom_label = QtWidgets.QLineEdit()
    atom_label.setObjectName("powder_csv_atom_label")
    atom_label.setPlaceholderText("V")
    atom_label.setToolTip(
        "Atom or ion named by a per-atom denominator, for example V. "
        "Leave blank to display 'per magnetic ion'."
    )
    form.addRow("Atom / ion label", atom_label)

    kinematic = QtWidgets.QComboBox()
    kinematic.setObjectName("powder_csv_kinematic")
    kinematic.addItem("Removed upstream (S(Q,E)-like)", "removed")
    kinematic.addItem("k_f/k_i remains in the values", "included")
    kinematic.setToolTip(
        "Choose Removed when the published ordinate is (k_i/k_f)d²σ/dΩdE, "
        "as in Tomiyasu et al. Included data later require fixed Ei or Ef."
    )
    form.addRow("k_f/k_i state", kinematic)

    map_sigma = QtWidgets.QDoubleSpinBox()
    map_sigma.setObjectName("powder_csv_map_sigma")
    map_sigma.setRange(1.0e-12, 1.0e12)
    map_sigma.setDecimals(6)
    map_sigma.setValue(1.0)
    map_sigma.setToolTip(
        "Uniform one-sigma uncertainty assigned only to digitized maps, "
        "which do not contain an error layer. It controls fit weighting."
    )
    form.addRow("Map σ", map_sigma)
    outer.addWidget(common)

    table = QtWidgets.QTableWidget(len(paths), 6)
    table.setObjectName("powder_csv_conditions")
    table.setHorizontalHeaderLabels(
        [
            "File",
            "Layout / cut",
            "Temperature (K)",
            "Fixed Q or E",
            "Quantity",
            "Units",
        ]
    )
    table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
    for column in (1, 2, 3, 4, 5):
        table.horizontalHeader().setSectionResizeMode(
            column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
    row_controls: list[tuple[Path, str, Any, Any, Any, Any, Any]] = []

    def populate_units(
        combo: Any,
        representation: str,
        *,
        current: str | None = None,
    ) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("Arbitrary", "arbitrary")
            if representation == "cross_section":
                combo.addItem("1/meV", "1/meV")
                combo.addItem("mbarn/sr/meV", "mbarn/sr/meV")
                combo.addItem("barn/sr/meV", "barn/sr/meV")
            else:
                combo.addItem("μ_B²/meV", "mu_B^2/meV")
                combo.addItem("spin²/meV", "spin^2/meV")
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    for row, raw_path in enumerate(paths):
        path = Path(raw_path)
        inspection = inspect_powder_ins_csv(path)
        filename = QtWidgets.QTableWidgetItem(path.name)
        filename.setFlags(filename.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        filename.setToolTip(str(path))
        table.setItem(row, 0, filename)

        cut = QtWidgets.QComboBox()
        cut.setObjectName(f"powder_csv_cut_type_{row}")
        cut.setToolTip("For a 1D file, specify whether x is Q at fixed E or E at fixed Q.")
        if inspection["layout"] == "matrix_q_energy":
            cut.addItem("Q-E map", "matrix_q_energy")
            cut.setEnabled(False)
        else:
            cut.addItem("Constant E (x = Q)", "constant_energy")
            cut.addItem("Constant Q (x = E)", "constant_q")
        table.setCellWidget(row, 1, cut)

        temperature = QtWidgets.QLineEdit()
        temperature.setObjectName(f"powder_csv_temperature_{row}")
        temperature.setPlaceholderText("required")
        temperature.setToolTip(
            "Sample temperature in kelvin for this file. It is required "
            "because digitizer CSV files contain no experimental metadata."
        )
        table.setCellWidget(row, 2, temperature)

        fixed = QtWidgets.QLineEdit()
        fixed.setObjectName(f"powder_csv_fixed_value_{row}")
        fixed.setPlaceholderText("meV" if inspection["layout"] == "cut" else "not applicable")
        fixed.setEnabled(inspection["layout"] == "cut")
        fixed.setToolTip(
            "Fixed E in meV for a constant-E cut, or fixed Q in Å⁻¹ for a constant-Q cut."
        )
        table.setCellWidget(row, 3, fixed)
        cut.currentIndexChanged.connect(
            lambda _index, selector=cut, editor=fixed: editor.setPlaceholderText(
                "meV" if selector.currentData() == "constant_energy" else "Å⁻¹"
            )
        )
        row_observable = QtWidgets.QComboBox()
        row_observable.setObjectName(f"powder_csv_observable_{row}")
        row_observable.addItem("Intensity / cross section", "cross_section")
        row_observable.addItem("χ″", "chi_double_prime")
        row_observable.setToolTip(
            "Physical quantity digitized in this file. This per-file choice "
            "allows one batch to contain intensity and χ″ datasets."
        )
        table.setCellWidget(row, 4, row_observable)

        row_units = QtWidgets.QComboBox()
        row_units.setObjectName(f"powder_csv_units_{row}")
        row_units.setToolTip(
            "Signal units for this file, before the common amount-of-sample denominator."
        )
        populate_units(row_units, "cross_section")
        row_observable.currentIndexChanged.connect(
            lambda _index, source=row_observable, target=row_units: populate_units(
                target, str(source.currentData()), current=str(target.currentData())
            )
        )
        table.setCellWidget(row, 5, row_units)
        row_controls.append(
            (
                path,
                inspection["layout"],
                cut,
                temperature,
                fixed,
                row_observable,
                row_units,
            )
        )
    outer.addWidget(table, 1)

    note = QtWidgets.QLabel(
        "Matrix CSVs use Q across columns and energy down rows. "
        "Three-column cuts use x, signal, and one-sigma uncertainty."
    )
    note.setWordWrap(True)
    note.setToolTip(
        "The importer recognizes the plot digitizer's y\\x matrix header; "
        "it does not infer temperatures or fixed conditions from filenames."
    )
    outer.addWidget(note)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
        | QtWidgets.QDialogButtonBox.StandardButton.Ok
    )
    outer.addWidget(buttons)
    result: dict[str, dict[str, Any]] = {}

    def rebuild_units() -> None:
        current = units.currentData()
        populate_units(
            units,
            str(observable.currentData()),
            current=None if current is None else str(current),
        )

    def apply_default_observable() -> None:
        rebuild_units()
        for *_prefix, row_observable, row_units in row_controls:
            row_observable.setCurrentIndex(row_observable.findData(observable.currentData()))
            populate_units(
                row_units,
                str(row_observable.currentData()),
                current=str(units.currentData()),
            )

    def apply_default_units() -> None:
        for *_prefix, row_observable, row_units in row_controls:
            index = row_units.findData(units.currentData())
            if index >= 0 and row_observable.currentData() == observable.currentData():
                row_units.setCurrentIndex(index)

    def basis_suffix() -> str:
        value = str(basis.currentData())
        if value == "per_formula_unit":
            return "/f.u."
        if value == "per_unit_cell":
            return "/unit cell"
        if value == "per_magnetic_ion":
            label = atom_label.text().strip()
            return f"/{label}" if label else "/magnetic ion"
        return ""

    def accept() -> None:
        result.clear()
        try:
            for (
                path,
                layout,
                cut,
                temperature,
                fixed,
                row_observable,
                row_units,
            ) in row_controls:
                temperature_K = float(temperature.text())
                if not np.isfinite(temperature_K) or temperature_K <= 0.0:
                    raise ValueError(f"{path.name}: enter a positive temperature")
                cut_type = str(cut.currentData()) if layout == "cut" else ""
                fixed_value = None
                if layout == "cut":
                    fixed_value = float(fixed.text())
                    if not np.isfinite(fixed_value):
                        raise ValueError(f"{path.name}: enter a finite fixed value")
                base_unit = str(row_units.currentData())
                source_unit = base_unit if base_unit == "arbitrary" else base_unit + basis_suffix()
                result[str(path)] = {
                    "layout": layout,
                    "cut_type": cut_type,
                    "fixed_value": fixed_value,
                    "temperature_K": temperature_K,
                    "source_representation": str(row_observable.currentData()),
                    "source_unit": source_unit,
                    "fit_representation": str(row_observable.currentData()),
                    "normalization_basis": str(basis.currentData()),
                    "normalization_label": atom_label.text().strip(),
                    "kf_ki_state": str(kinematic.currentData()),
                    "default_uncertainty": float(map_sigma.value()),
                }
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(dialog, "Import powder INS CSV", str(exc))
            return
        dialog.accept()

    observable.currentIndexChanged.connect(apply_default_observable)
    units.currentIndexChanged.connect(apply_default_units)
    basis.currentIndexChanged.connect(
        lambda _index: atom_label.setEnabled(basis.currentData() == "per_magnetic_ion")
    )
    basis.setCurrentIndex(basis.findData("unknown"))
    atom_label.setEnabled(False)
    apply_default_observable()
    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return False
    return result
