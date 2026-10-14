"""Saved spectral declarations agree across conversion entry points."""

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit.analysis import SpectralConvention
from nfit.analysis.spectral import convert_spectral_representation
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.scientific_compatibility import validate_spectral_source_settings
from nfit.spectral_channels import (
    default_spectral_channel_config,
    normalized_spectral_channel_config,
    updated_spectral_channel_config,
    with_paired_spectral_channels,
)
from nfit.spectral_metadata import normalization_basis_suffix


def histogram(metadata=None):
    return MDHistoData(
        axes=(MDHistoAxis("Q", [0., 1., 2.], "1/angstrom", "momentum"),
              MDHistoAxis("E", [1., 2., 3.], "meV", "energy")),
        signal=np.full((2, 2), 2.), errors=np.full((2, 2), .1),
        mask=np.zeros((2, 2), dtype=bool), num_events=np.ones((2, 2)),
        metadata=metadata or {},
    )


@pytest.mark.parametrize("target", ["cross_section", "chi_double_prime"])
@pytest.mark.parametrize("source_representation", ["cross_section", "chi_double_prime"])
@pytest.mark.parametrize("moment,g", [("mu_B_squared", None), ("spin_squared", 2.4)])
@pytest.mark.parametrize("geometry", [{"incident_energy_meV": 8.}, {"final_energy_meV": 5.}])
def test_analysis_and_paired_channels_keep_their_saved_schema_and_physical_scale(target, source_representation, moment, g, geometry):
    susceptibility_unit = "mu_B^2/meV" if moment == "mu_B_squared" else "spin^2/meV"
    source_unit = "barn/sr/meV" if source_representation == "cross_section" else susceptibility_unit
    source_state = "included" if source_representation == "cross_section" else "removed"
    source = SpectralConvention(
        source_representation, source_unit, "per_formula_unit", 3., moment, g,
        source_state, source_state, source_state, "included", True, **geometry,
    )
    data = histogram({"spectral_convention": source.to_dict()})
    analysis = convert_spectral_representation(
        data, convention=source, target_representation=target, temperature_K=25., polarization=2.,
    )
    config = default_spectral_channel_config()
    config.update(source_representation=source_representation, source_unit=source_unit, fit_representation=target,
                  moment_unit=moment, g_factor=g, kf_ki_state="included", **geometry)
    paired = with_paired_spectral_channels(data, config, temperature_K=25.)
    analysis_unit = "barn/(sr meV)" if target == "cross_section" else susceptibility_unit
    paired_unit = "mbarn/sr/meV/f.u." if target == "cross_section" else susceptibility_unit + "/f.u."
    state = "included" if target == "cross_section" else "removed"
    # Literal schema guards against introducing new keys, dropping None-valued
    # fields, or treating reference Ei/Ef as a susceptibility correction.
    expected = {
        "representation": target, "unit": analysis_unit,
        "normalization_basis": "per_formula_unit", "magnetic_ions_per_basis": 3.,
        "moment_unit": moment, "g_factor": g,
        "form_factor_state": state, "polarization_state": state, "bose_state": state,
        "kf_ki_state": "included", "absolute_scale": True,
        "incident_energy_meV": geometry.get("incident_energy_meV"),
        "final_energy_meV": geometry.get("final_energy_meV"),
    }
    assert analysis.metadata["spectral_convention"] == expected
    assert paired.metadata["spectral_convention"] == {**expected, "unit": paired_unit}
    multiplier = 1000. if target == "cross_section" else 1.
    np.testing.assert_allclose(paired.signal, analysis.signal * multiplier)
    np.testing.assert_allclose(paired.errors, analysis.errors * multiplier)
    for converted in (analysis, paired):
        assert converted.metadata["signal_unit"] == converted.metadata["spectral_convention"]["unit"]
    assert analysis.metadata["spectral_observable"] == {
        **expected, "source_representation": target, "source_unit": analysis_unit,
        "fit_representation": target, "polarization_mode": "custom_scalar",
        "polarization_scalar": 2.,
        "quantity_type": "differential_cross_section" if target == "cross_section" else "dynamic_susceptibility",
        "temperature_K": 25.,
    }
    assert analysis.metadata["spectral_conversion"] == {
        "source_convention": source.to_dict(), "temperature_K": 25.,
        "scale_signal_per_barn_sr_meV": 1.,
    }
    assert paired.metadata["spectral_observable"] == {
        **config, **expected, "unit": paired_unit, "fit_representation": target,
        "quantity_type": analysis.metadata["signal_quantity_type"], "temperature_K": 25.,
    }


@pytest.mark.parametrize("temperature", [25., np.array([[10.], [25.]])])
def test_temperature_and_array_polarization_records_preserve_entry_point_shapes(temperature):
    source = SpectralConvention(
        "cross_section", "barn/sr/meV", "per_formula_unit", None, "mu_B_squared", 2.,
        "included", "included", "included", "removed", True,
    )
    analysis = convert_spectral_representation(
        histogram(), convention=source, target_representation="chi_double_prime",
        temperature_K=temperature, polarization=np.array([[1.], [2.]]),
    )
    expected_temperature = 25. if np.ndim(temperature) == 0 else [[10.], [25.]]
    assert analysis.metadata["temperature"] == expected_temperature
    assert analysis.metadata["spectral_observable"]["temperature_K"] == expected_temperature
    assert analysis.metadata["spectral_observable"]["polarization_mode"] == "custom_array"
    assert analysis.metadata["spectral_observable"]["polarization_scalar"] == [[1.], [2.]]
    paired = with_paired_spectral_channels(histogram(), {}, temperature_K=temperature)
    assert paired.metadata["temperature"] == expected_temperature
    # Paired-channel observable metadata deliberately keeps the caller's array.
    np.testing.assert_array_equal(paired.metadata["spectral_observable"]["temperature_K"], temperature)


@pytest.mark.parametrize("basis,ions", [("per_formula_unit", 4.), ("per_unit_cell", None)])
def test_ion_count_inheritance_requires_the_same_basis(basis, ions):
    data = histogram({"spectral_convention": {
        "normalization_basis": "per_formula_unit", "magnetic_ions_per_basis": 4.,
    }})
    paired = with_paired_spectral_channels(data, {"normalization_basis": basis}, temperature_K=None)
    target = paired.metadata["spectral_convention"]
    assert target["magnetic_ions_per_basis"] == ions
    assert target["absolute_scale"] is False
    assert "temperature" not in paired.metadata
    assert paired.metadata["spectral_observable"]["temperature_K"] is None


@pytest.mark.parametrize("unit,representation", [
    ("mbarn/sr/meV/f.u.", "cross_section"), ("barn/sr/meV/f.u.", "cross_section"),
    ("1/meV/f.u.", "cross_section"), ("mu_B^2/meV/f.u.", "chi_double_prime"),
    ("spin^2/meV/f.u.", "chi_double_prime"),
])
def test_basis_and_label_edits_preserve_existing_unit_prefixes(unit, representation):
    config = {"source_unit": unit, "source_representation": representation}
    prefix = unit.removesuffix("/f.u.")
    edited = updated_spectral_channel_config(config, "normalization_basis", "per_magnetic_ion")
    assert edited["source_unit"] == prefix + "/magnetic ion"
    edited = updated_spectral_channel_config(edited, "normalization_label", " Fe ")
    assert edited["source_unit"] == prefix + "/Fe"
    edited = updated_spectral_channel_config(edited, "normalization_basis", "per_unit_cell")
    assert edited["source_unit"] == prefix + "/unit cell"
    # Existing label edits rewrite the suffix even when the basis is a cell.
    edited = updated_spectral_channel_config(edited, "normalization_label", "")
    assert edited["source_unit"] == prefix + "/magnetic ion"


@pytest.mark.parametrize("basis,label,suffix", [
    ("per_formula_unit", "", "/f.u."), ("per_unit_cell", "", "/unit cell"),
    ("unknown", "", ""), ("per_magnetic_ion", " Fe ", "/Fe"),
    ("per_magnetic_ion", " ", "/magnetic ion"),
])
def test_basis_display_and_saved_unit_suffixes_agree(basis, label, suffix):
    assert normalization_basis_suffix(basis, label) == suffix
    assert normalization_basis_suffix(basis, label, spaced=True) == (" / " + suffix[1:] if suffix else "")


@pytest.mark.parametrize("suffix,accepted", [("/Fe", True), ("/magnetic ion", True), ("", True), ("/Co", False), ("/f.u.", False)])
def test_compatibility_only_strips_the_declared_basis_and_label(suffix, accepted):
    config = normalized_spectral_channel_config({
        "source_representation": "chi_double_prime", "source_unit": "mu_B^2/meV" + suffix,
        "normalization_basis": "per_magnetic_ion", "normalization_label": "Fe",
    })
    data = histogram({"signal_quantity_type": "dynamic_susceptibility", "signal_unit": "mu_B^2/meV/Fe",
                      "spectral_convention": {"representation": "chi_double_prime", "unit": "mu_B^2/meV/Fe",
                                              "normalization_basis": "per_magnetic_ion", "normalization_label": "Fe"}})
    if accepted:
        validate_spectral_source_settings(data, config)
    else:
        with pytest.raises(ValueError, match="source unit"):
            validate_spectral_source_settings(data, config)


def test_metadata_service_is_shared_and_has_no_conversion_or_gui_dependencies():
    import nfit.analysis.spectral as analysis
    import nfit.scientific_compatibility as compatibility
    import nfit.spectral_channels as channels
    import nfit.spectral_metadata as metadata

    assert analysis.target_spectral_record is channels.target_spectral_record is metadata.target_spectral_record
    assert compatibility.unit_without_normalization_basis is metadata.unit_without_normalization_basis
    assert channels.normalization_basis_suffix is metadata.normalization_basis_suffix
    tree = ast.parse(Path(metadata.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(imports) == {"__future__", "collections.abc", "typing", "quantities"}
