"""Focused editors for declared physical conventions in analysis recipes."""
from __future__ import annotations

import copy
import math
from dataclasses import MISSING, fields
from typing import Any

from PySide6 import QtCore, QtWidgets

from .analysis import default_analysis_parameters
from .analysis.corrections import SpectralConvention


class SpectralConventionEditor(QtWidgets.QWidget):
    """Edit a saved INS convention without inferring calibration or corrections."""

    valueChanged = QtCore.Signal()

    def __init__(self, value: dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("spectral_convention_editor")
        self.setToolTip("Declare the quantity, units, normalization, and corrections already present in the input signal.")
        self._loading = False
        self._defaults = default_analysis_parameters("spectral_conversion")["spectral_convention"]
        self._fields = {field.name for field in fields(SpectralConvention)}
        self._required_fields = {field.name for field in fields(SpectralConvention) if field.default is MISSING}
        self.controls: dict[str, QtWidgets.QWidget] = {}
        self._rows: dict[str, tuple[QtWidgets.QFormLayout, QtWidgets.QWidget]] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        main = QtWidgets.QFormLayout()
        layout.addLayout(main)
        self.representation = self._combo(main, "representation", "Quantity", (
            ("Measured intensity", "measured_intensity"),
            ("Cross section", "cross_section"),
            ("Scattering function S(Q,E)", "s_qw"),
            ("Dynamic susceptibility χ″", "chi_double_prime"),
        ), "Choose what the input values represent. Changing this declaration does not convert the data.")
        self.unit = QtWidgets.QComboBox()
        self.unit.setEditable(True)
        self._row(main, "unit", "Input units", self.unit,
                  "Choose or type the recorded input units. Declaring units does not calibrate measured counts; provide the conversion scale separately.")
        self.normalization_basis = self._combo(main, "normalization_basis", "Normalization", (
            ("Unknown", "unknown"), ("Per magnetic ion", "per_magnetic_ion"),
            ("Per formula unit", "per_formula_unit"), ("Per unit cell", "per_unit_cell"),
        ), "State the basis used by the input intensity. Absolute integrals need a known basis and, when applicable, the number of magnetic ions per basis.")
        self.corrections_button = QtWidgets.QToolButton()
        self.corrections_button.setObjectName("spectral_convention_corrections_toggle")
        self.corrections_button.setText("Corrections and scale")
        self.corrections_button.setToolTip("Show correction states, fixed neutron energy, moment units, and the existing intensity scale declaration.")
        self.corrections_button.setCheckable(True)
        self.corrections_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.corrections_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        layout.addWidget(self.corrections_button)
        self.corrections = QtWidgets.QWidget()
        self.corrections.setObjectName("spectral_convention_corrections")
        correction_form = QtWidgets.QFormLayout(self.corrections)
        correction_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.corrections)
        states = (("Included in input", "included"), ("Already removed", "removed"))
        for name, label, tip in (
            ("form_factor_state", "Form factor F²", "Is the magnetic form-factor squared still included in the input? Its correction value is supplied separately."),
            ("polarization_state", "Polarization", "Is the neutron polarization factor still included in the input? Its correction assumption is supplied separately."),
            ("bose_state", "Bose population", "Is the thermal population factor still included in the input? Temperature is supplied separately."),
            ("kf_ki_state", "Neutron kf/ki", "Is the final-to-incident neutron wavevector ratio still included? If included, declare fixed incident or final energy below."),
        ):
            self._combo(correction_form, name, label, states, tip)
        self.fixed_energy = self._combo(correction_form, "fixed_energy", "Fixed energy", (
            ("Not specified", None), ("Incident Ei", "incident_energy_meV"), ("Final Ef", "final_energy_meV"),
        ), "Choose the fixed neutron energy for the kf/ki correction. Provide either incident or final energy, in meV.")
        self.energy = self._number(correction_form, "energy", "Energy (meV)",
            "Positive fixed incident or final neutron energy in meV. Leave blank when it is not specified.")
        self.moment_unit = self._combo(correction_form, "moment_unit", "Moment units", (
            ("Magnetic moment μB²", "mu_B_squared"), ("Spin squared", "spin_squared"),
        ), "Declare whether magnetic spectra use squared Bohr magnetons or dimensionless spin squared. This affects the magnetic cross-section factor.")
        self.g_factor = self._number(correction_form, "g_factor", "Landé g factor",
            "Positive dimensionless Landé g factor for spin-to-moment model conversion or QFI from moment units. Leave blank if not specified; nfit never assumes g = 2.")
        self.absolute_scale = QtWidgets.QCheckBox("Input has an absolute scale")
        self._row(correction_form, "absolute_scale", "Scale", self.absolute_scale,
            "Declare an existing absolute calibration. Checking this box does not calibrate counts or determine a scale factor.")
        self.magnetic_ions_per_basis = self._number(correction_form, "magnetic_ions_per_basis", "Magnetic ions per basis",
            "Positive number of magnetic ions per formula unit or unit cell, for conversion to a per-ion result. Leave blank if unknown.")
        for control in self.controls.values():
            if isinstance(control, QtWidgets.QComboBox):
                control.currentIndexChanged.connect(self._changed)
                if control.isEditable():
                    control.currentTextChanged.connect(self._changed)
            elif isinstance(control, QtWidgets.QLineEdit):
                control.textChanged.connect(self._changed)
            elif isinstance(control, QtWidgets.QCheckBox):
                control.toggled.connect(self._changed)
        self.corrections_button.toggled.connect(self._expand_corrections)
        self.setValue(copy.deepcopy(self._defaults) if value is None else value)
        self._expand_corrections(False)

    def _row(self, form, name, label, control, tooltip):
        control.setObjectName(f"spectral_convention_{name}")
        control.setToolTip(tooltip)
        form.addRow(label, control)
        self.controls[name] = control
        self._rows[name] = (form, control)
        return control

    def _combo(self, form, name, label, choices, tooltip):
        control = QtWidgets.QComboBox()
        for text, key in choices:
            control.addItem(text, key)
        return self._row(form, name, label, control, tooltip)

    def _number(self, form, name, label, tooltip):
        control = QtWidgets.QLineEdit()
        control.setPlaceholderText("Not specified")
        control.setMaximumWidth(160)
        return self._row(form, name, label, control, tooltip)

    def _expand_corrections(self, expanded):
        self.corrections.setVisible(expanded)
        self.corrections_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    def _show_row(self, name, visible):
        form, control = self._rows[name]
        control.setVisible(visible)
        form.labelForField(control).setVisible(visible)

    def _refresh_visibility(self):
        representation = self.representation.currentData()
        suggestions = {
            "measured_intensity": ("counts", "arbitrary units"),
            "cross_section": ("barn/(sr meV)", "mbarn/(sr meV)"),
            "s_qw": ("mu_B^2/meV", "spin^2/meV"),
            "chi_double_prime": ("mu_B^2/meV", "spin^2/meV"),
        }.get(representation, ())
        current = self.unit.currentText()
        blocker = QtCore.QSignalBlocker(self.unit)
        self.unit.clear()
        self.unit.addItems(suggestions)
        self.unit.setEditText(current)
        del blocker
        self._show_row("fixed_energy", self.controls["kf_ki_state"].currentData() == "included")
        self._show_row("energy", self.controls["kf_ki_state"].currentData() == "included" and self.fixed_energy.currentData() is not None)
        self._show_row("g_factor", True)
        self._show_row("magnetic_ions_per_basis", self.normalization_basis.currentData() in {"per_formula_unit", "per_unit_cell"})

    def _changed(self, *_):
        if not self._loading:
            self._refresh_visibility()
            self.valueChanged.emit()

    @staticmethod
    def _optional_number(control, label):
        text = control.text().strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a finite positive number or blank") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{label} must be a finite positive number or blank")
        return number

    def _read_controls(self):
        result = {
            name: self.controls[name].currentData()
            for name in ("representation", "normalization_basis", "moment_unit", "form_factor_state", "polarization_state", "bose_state", "kf_ki_state")
        }
        result.update(
            unit=self.unit.currentText(),
            absolute_scale=self.absolute_scale.isChecked(),
            g_factor=self._optional_number(self.g_factor, "Landé g factor"),
            magnetic_ions_per_basis=self._optional_number(self.magnetic_ions_per_basis, "Magnetic ions per basis"),
            incident_energy_meV=None, final_energy_meV=None,
        )
        energy_key = self.fixed_energy.currentData()
        number = self._optional_number(self.energy, "Fixed neutron energy")
        if energy_key is not None:
            result[energy_key] = number
        return result

    def value(self) -> dict[str, Any]:
        """Return a validated convention, preserving saved extension fields."""
        current = self._read_controls()
        SpectralConvention.from_dict(current)
        result = copy.deepcopy(self._saved)
        for name, value in current.items():
            if value != self._initial[name] or (name not in result and name in self._required_fields):
                result[name] = value
        return result

    def setValue(self, value: dict[str, Any]) -> None:
        """Restore declarations without inferring corrections or calibration."""
        if not isinstance(value, dict):
            raise ValueError("spectral convention must be a dictionary")
        effective = {**self._defaults, "incident_energy_meV": None, "final_energy_meV": None, **{name: item for name, item in value.items() if name in self._fields}}
        SpectralConvention.from_dict(effective)
        for name in ("g_factor", "magnetic_ions_per_basis", "incident_energy_meV", "final_energy_meV"):
            number = effective[name]
            if number is not None and (not math.isfinite(float(number)) or float(number) <= 0):
                raise ValueError(f"{name} must be finite and positive or None")
        self._loading = True
        try:
            for name in ("representation", "normalization_basis", "moment_unit", "form_factor_state", "polarization_state", "bose_state", "kf_ki_state"):
                self.controls[name].setCurrentIndex(self.controls[name].findData(effective[name]))
            self.unit.setEditText(effective["unit"])
            self.absolute_scale.setChecked(effective["absolute_scale"])
            for name in ("g_factor", "magnetic_ions_per_basis"):
                self.controls[name].setText("" if effective[name] is None else str(effective[name]))
            energy_key = next((name for name in ("incident_energy_meV", "final_energy_meV") if effective[name] is not None), None)
            self.fixed_energy.setCurrentIndex(self.fixed_energy.findData(energy_key))
            self.energy.setText("" if energy_key is None else str(effective[energy_key]))
            self._refresh_visibility()
            self._saved = copy.deepcopy(value)
            self._initial = self._read_controls()
        finally:
            self._loading = False
