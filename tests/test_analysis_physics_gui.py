"""Physical-convention controls preserve saved declarations and optional values."""
import copy

import pytest

from nfit.analysis import default_analysis_parameters


@pytest.fixture
def editor(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.analysis_physics_gui import SpectralConventionEditor

    application = widgets.QApplication.instance() or widgets.QApplication([])
    result = SpectralConventionEditor()
    yield result
    result.close()
    application.processEvents()


def convention(**changes):
    result = default_analysis_parameters("spectral_conversion")["spectral_convention"]
    result.update(changes)
    return result


@pytest.mark.parametrize("changes", [
    {},
    {"representation": "cross_section", "unit": "mbarn/(sr meV)", "normalization_basis": "per_formula_unit", "magnetic_ions_per_basis": 2., "absolute_scale": True},
    {"representation": "chi_double_prime", "unit": "mu_B^2/meV", "form_factor_state": "removed", "polarization_state": "removed", "bose_state": "removed", "normalization_basis": "per_magnetic_ion", "absolute_scale": True, "g_factor": 2.3},
    {"representation": "s_qw", "unit": "spin^2/meV", "moment_unit": "spin_squared", "g_factor": 2.1, "kf_ki_state": "included", "incident_energy_meV": 35., "final_energy_meV": None},
    {"representation": "cross_section", "unit": "barn/(sr meV)/unit cell", "normalization_basis": "per_unit_cell", "magnetic_ions_per_basis": 4., "kf_ki_state": "included", "incident_energy_meV": None, "final_energy_meV": 14.7},
])
def test_scalar_conventions_roundtrip_exactly_without_inference(editor, changes):
    saved = convention(**changes)
    editor.setValue(saved)
    assert editor.value() == saved
    assert editor.unit.currentText() == saved["unit"]


def test_default_comes_from_registry_and_does_not_add_unspecified_energy(editor):
    expected = default_analysis_parameters("spectral_conversion")["spectral_convention"]
    assert editor.value() == expected
    assert "incident_energy_meV" not in editor.value()
    assert "final_energy_meV" not in editor.value()
    assert editor.g_factor.text() == ""
    assert editor.magnetic_ions_per_basis.text() == ""
    assert not editor.absolute_scale.isChecked()


def test_unknown_saved_fields_are_preserved_and_isolated(editor):
    saved = convention(future_convention={"version": 2, "coefficients": [1, 2]})
    expected = copy.deepcopy(saved)
    editor.setValue(saved)
    saved["future_convention"]["coefficients"].append(3)
    assert editor.value() == expected
    returned = editor.value()
    returned["future_convention"]["coefficients"].append(4)
    assert editor.value() == expected
    editor.unit.setEditText("recorded counts per monitor")
    assert editor.value()["future_convention"] == expected["future_convention"]


def test_optional_numbers_stay_none_and_clear_without_replacement(editor):
    saved = convention(g_factor=2., magnetic_ions_per_basis=3., incident_energy_meV=None, final_energy_meV=None)
    editor.setValue(saved)
    editor.g_factor.clear()
    editor.magnetic_ions_per_basis.clear()
    result = editor.value()
    assert result["g_factor"] is None
    assert result["magnetic_ions_per_basis"] is None
    assert result["incident_energy_meV"] is None
    assert result["final_energy_meV"] is None


def test_editing_quantity_does_not_change_declared_units_or_calibrate(editor):
    editor.unit.setEditText("measured detector counts")
    editor.representation.setCurrentIndex(editor.representation.findData("cross_section"))
    result = editor.value()
    assert result["unit"] == "measured detector counts"
    assert result["representation"] == "cross_section"
    assert result["absolute_scale"] is False
    assert "does not calibrate" in editor.unit.toolTip()
    assert "does not calibrate" in editor.absolute_scale.toolTip()


def test_controls_have_tooltips_and_corrections_expand_conditionally(editor):
    assert editor.toolTip()
    assert all(control.toolTip() for control in editor.controls.values())
    assert editor.corrections_button.toolTip()
    editor.show()
    assert editor.corrections.isHidden()
    editor.corrections_button.setChecked(True)
    assert not editor.corrections.isHidden()
    assert editor.fixed_energy.isHidden()
    assert editor.energy.isHidden()
    assert not editor.g_factor.isHidden()
    assert editor.magnetic_ions_per_basis.isHidden()
    editor.controls["kf_ki_state"].setCurrentIndex(editor.controls["kf_ki_state"].findData("included"))
    assert not editor.fixed_energy.isHidden()
    editor.fixed_energy.setCurrentIndex(editor.fixed_energy.findData("final_energy_meV"))
    assert not editor.energy.isHidden()
    editor.energy.setText("14.7")
    assert editor.value()["final_energy_meV"] == 14.7
    editor.moment_unit.setCurrentIndex(editor.moment_unit.findData("spin_squared"))
    assert not editor.g_factor.isHidden()
    editor.normalization_basis.setCurrentIndex(editor.normalization_basis.findData("per_formula_unit"))
    assert not editor.magnetic_ions_per_basis.isHidden()


@pytest.mark.parametrize("field", ["g_factor", "magnetic_ions_per_basis", "energy"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "-2", "0", "not a number"])
def test_optional_physical_numbers_must_be_finite_positive_or_blank(editor, field, invalid):
    if field == "energy":
        editor.fixed_energy.setCurrentIndex(editor.fixed_energy.findData("incident_energy_meV"))
    editor.controls[field].setText(invalid)
    with pytest.raises(ValueError, match="finite positive"):
        editor.value()


@pytest.mark.parametrize("field", ["g_factor", "magnetic_ions_per_basis", "incident_energy_meV", "final_energy_meV"])
def test_nonfinite_saved_optional_numbers_are_rejected(editor, field):
    with pytest.raises(ValueError, match="finite"):
        editor.setValue(convention(**{field: float("nan")}))


def test_empty_units_and_missing_fixed_energy_have_useful_validation(editor):
    editor.unit.setEditText(" ")
    with pytest.raises(ValueError, match="unit must be explicit"):
        editor.value()
    editor.setValue(convention())
    editor.controls["kf_ki_state"].setCurrentIndex(editor.controls["kf_ki_state"].findData("included"))
    with pytest.raises(ValueError, match="requires incident_energy_meV or final_energy_meV"):
        editor.value()


def test_programmatic_restore_is_quiet_but_user_edits_emit_change(editor):
    emitted = []
    editor.valueChanged.connect(lambda: emitted.append(True))
    editor.setValue(convention(unit="custom detector counts"))
    assert not emitted
    editor.unit.setEditText("custom counts per monitor")
    assert emitted


def test_restore_preserves_optional_key_presence_and_original_scalar_types(editor):
    saved = convention(g_factor=2, magnetic_ions_per_basis=3, unit="  detector counts  ")
    editor.setValue(saved)
    actual = editor.value()
    assert actual == saved
    assert type(actual["g_factor"]) is int
    assert type(actual["magnetic_ions_per_basis"]) is int
    assert actual["unit"] == "  detector counts  "
    editor.setValue({})
    assert editor.value() == convention()


@pytest.mark.parametrize("moment_unit", ["mu_B_squared", "spin_squared"])
def test_g_factor_remains_editable_for_both_moment_conventions(editor, moment_unit):
    editor.setValue(convention(moment_unit=moment_unit))
    editor.corrections_button.setChecked(True)
    assert not editor.g_factor.isHidden()
    editor.g_factor.setText("2.4")
    assert editor.value()["g_factor"] == 2.4
    assert "QFI from moment units" in editor.g_factor.toolTip()
    assert "never assumes" in editor.g_factor.toolTip()
