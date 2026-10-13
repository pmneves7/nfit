import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from nfit.analysis.data_reduction import combine_aligned_histograms
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from nfit.project_composites import (
    _cached_composite_dataset_data,
    _composite_scope,
    composite_dataset_data,
    data_group_composite_config,
)
from nfit.scientific_compatibility import (
    SCIENTIFIC_IDENTITY_KEY,
    scientific_data_identity,
    validate_scientific_compatibility,
)


def histogram(value=10., *, convention=None, metadata=None):
    axes = tuple(MDHistoAxis(name, np.array([-.5, .5]), "rlu", "momentum")
                 for name in ("H", "K", "L")) + (
        MDHistoAxis("DeltaE", np.array([.5, 1.5]), "meV", "energy"),
    )
    declaration = {
        "representation": "cross_section", "unit": "barn/(sr meV)",
        "normalization_basis": "per_formula_unit", "magnetic_ions_per_basis": 2.,
        "moment_unit": "mu_B_squared", "g_factor": 2.,
        "form_factor_state": "included", "polarization_state": "included",
        "bose_state": "included", "kf_ki_state": "removed", "absolute_scale": True,
        **(convention or {}),
    }
    return MDHistoData(
        axes, np.full((1, 1, 1, 1), value), np.ones((1, 1, 1, 1)),
        np.zeros((1, 1, 1, 1), bool), np.ones((1, 1, 1, 1)),
        metadata={"signal_semantics": "density", "signal_unit": declaration["unit"],
                  "signal_quantity_type": "differential_cross_section",
                  "spectral_convention": declaration, **(metadata or {})},
    )


def group_for(entries, *, root=None, node=None):
    group = DataGroup("sources", datasets=entries) if root is None else _composite_scope(root, node)
    config = data_group_composite_config(group)
    config.update(enabled=True, fractional=False, minimum_coverage=0., mean_weighting="uniform")
    if "axes" not in config:
        config["axes"] = [{"name": name, "variable": name, "units": "rlu" if index < 3 else "meV",
                           "vector": np.eye(4)[index].tolist(), "lower": -.5 if index < 3 else .5,
                           "upper": .5 if index < 3 else 1.5, "num_bins": 1}
                          for index, name in enumerate(("H", "K", "L", "E"))]
    for index, axis in enumerate(config["axes"]):
        axis.update(bin_edges=[-.5, .5] if index < 3 else [.5, 1.5], mode="edges",
                    auto_lower=False, auto_upper=False)
    return group


@pytest.mark.parametrize("change", [
    {"unit": "mbarn/(sr meV)"},
    {"normalization_basis": "per_magnetic_ion"},
    {"magnetic_ions_per_basis": 1.},
    {"form_factor_state": "removed"},
    {"polarization_state": "removed"},
    {"bose_state": "removed"},
    {"kf_ki_state": "included", "incident_energy_meV": 15.},
    {"absolute_scale": False},
])
def test_flat_composites_and_arithmetic_reject_physical_mismatches(change):
    first, other = histogram(), histogram(convention=change)
    group = group_for([DatasetEntry("left", first), DatasetEntry("right", other)])
    with pytest.raises(ValueError, match="incompatible scientific units or normalization"):
        composite_dataset_data(group)
    with pytest.raises(ValueError, match="incompatible scientific units or normalization"):
        combine_aligned_histograms(first, other)


def test_known_and_unknown_units_cannot_be_combined():
    first = histogram()
    unknown = first.with_updates(metadata={"signal_semantics": "density"})
    with pytest.raises(ValueError, match="unit"):
        validate_scientific_compatibility([(None, first), (None, unknown)])
    validate_scientific_compatibility([(None, unknown), (None, unknown)])


def test_different_energy_coordinate_units_cannot_be_pooled_without_conversion():
    first = histogram()
    other = first.with_updates(axes=(*first.axes[:-1], MDHistoAxis("DeltaE", np.array([.5, 1.5]), "eV", "energy")))
    group = group_for([DatasetEntry("meV", first), DatasetEntry("eV", other)])
    with pytest.raises(ValueError, match="energy_units"):
        composite_dataset_data(group)


def test_corrected_susceptibility_can_combine_with_different_reference_energies():
    from nfit.analysis import SpectralConvention
    from nfit.analysis.spectral import convert_spectral_representation
    from nfit.cross_section import cross_section_from_chipp, kf_over_ki

    entries = []
    for name, energy in (("HYSPEC Ei15", {"incident_energy_meV": 15.}),
                         ("MACS Ef3.7", {"final_energy_meV": 3.7}),
                         ("MACS Ef5", {"final_energy_meV": 5.})):
        template = histogram(convention={"kf_ki_state": "included", **energy})
        ratio = kf_over_ki(1., **energy)
        signal = float(cross_section_from_chipp(3., 1., 10., kf_ki=ratio))
        data = template.with_updates(signal=np.full(template.shape, signal))
        converted = convert_spectral_representation(data,
            convention=SpectralConvention.from_dict(data.metadata["spectral_convention"]),
            target_representation="chi_double_prime", temperature_K=10.)
        entries.append(DatasetEntry(name, converted))
    result = composite_dataset_data(group_for(entries))
    np.testing.assert_allclose(result.signal, 3.)
    assert scientific_data_identity(result)["kf_ki_state"] == "removed"


def test_canonical_sources_ignore_different_original_entry_calibrations():
    entries = [DatasetEntry(name, histogram(value, convention=energy),
                   parameters={"spectral_channels": {"source_unit": "arbitrary",
                       "signal_per_mbarn": calibration, "kf_ki_state": "included", **energy}})
               for name, value, calibration, energy in (
                   ("HYSPEC Ei15", 10., 2., {"incident_energy_meV": 15.}),
                   ("MACS Ef3.7", 20., 7., {"final_energy_meV": 3.7}),
                   ("MACS Ef5", 30., 12., {"final_energy_meV": 5.}))]
    np.testing.assert_allclose(composite_dataset_data(group_for(entries)).signal, 20.)


def test_composite_conversion_matches_actual_susceptibility_source_settings():
    data = histogram(3., convention={"representation": "chi_double_prime", "unit": "mu_B^2/meV",
        "form_factor_state": "removed", "polarization_state": "removed", "bose_state": "removed"},
        metadata={"signal_quantity_type": "dynamic_susceptibility", "temperature": 10.})
    group = group_for([DatasetEntry("corrected response", data, parameters={"temperature": 10.})])
    config = data_group_composite_config(group)
    config["spectral_channels"] = {"enabled": True, "source_representation": "cross_section"}
    with pytest.raises(ValueError, match="source settings do not match"):
        composite_dataset_data(group)
    config["spectral_channels"] = {"enabled": True, "source_representation": "chi_double_prime",
        "source_unit": "mu_B^2/meV/f.u.", "fit_representation": "chi_double_prime",
        "normalization_basis": "per_formula_unit", "moment_unit": "mu_B_squared"}
    result = composite_dataset_data(group)
    np.testing.assert_allclose(result.signal, 3.)
    np.testing.assert_allclose(result.errors, 1.)


def test_paired_composite_conversion_cannot_remove_bose_factor_twice():
    group = group_for([DatasetEntry("corrected cross", histogram(convention={"bose_state": "removed"}),
                                   parameters={"temperature": 10.})])
    data_group_composite_config(group)["spectral_channels"] = {
        "enabled": True, "source_representation": "cross_section", "source_unit": "barn/sr/meV/f.u.",
        "normalization_basis": "per_formula_unit", "fit_representation": "chi_double_prime"}
    with pytest.raises(ValueError, match="Bose correction was already removed"):
        composite_dataset_data(group)


def test_arbitrary_units_with_basis_suffix_do_not_claim_a_known_physical_scale():
    from nfit.scientific_compatibility import validate_spectral_source_settings

    data = histogram(convention={"unit": "arb. units/f.u.", "absolute_scale": False})
    assert scientific_data_identity(data)["unit"] is None
    validate_spectral_source_settings(data, {"source_representation": "cross_section", "source_unit": "arbitrary"})


def test_unconverted_sources_require_same_calibration_and_monitor_units():
    first = histogram().with_updates(metadata={"signal_unit": "counts/monitor=10000"})
    second = first.with_updates(metadata={"signal_unit": "counts/monitor=20000"})
    with pytest.raises(ValueError, match="unit"):
        composite_dataset_data(group_for([DatasetEntry("one", first), DatasetEntry("two", second)]))
    entries = [DatasetEntry(name, first, parameters={"spectral_channels": {
        "source_unit": "arbitrary", "signal_per_mbarn": calibration,
        "normalization_basis": "per_formula_unit"}})
        for name, calibration in (("one", 2.), ("two", 3.))]
    with pytest.raises(ValueError, match="signal_per_mbarn"):
        composite_dataset_data(group_for(entries))


def test_mixed_included_direct_and_indirect_kinematics_rejected_before_pooling():
    entries = [DatasetEntry(name, histogram(convention={"kf_ki_state": "included", **energy}))
               for name, energy in (("HYSPEC Ei15", {"incident_energy_meV": 15.}),
                                    ("MACS Ef3.7", {"final_energy_meV": 3.7}),
                                    ("MACS Ef5", {"final_energy_meV": 5.}))]
    with pytest.raises(ValueError, match="source-specific k_f/k_i removal"):
        composite_dataset_data(group_for(entries))


def test_canonical_absolute_sources_combine_across_instrument_and_energy_provenance():
    entries = [DatasetEntry(name, histogram(value, convention=energy,
                   metadata={"instrument": name.split()[0], "normalization": normalization}),
                   metadata={"instrument": name.split()[0], "normalization": normalization})
               for name, value, energy, normalization in (
                   ("HYSPEC Ei15", 10., {"incident_energy_meV": 15.}, "proton_charge"),
                   ("MACS Ef3.7", 20., {"final_energy_meV": 3.7}, "monitor"),
                   ("MACS Ef5", 30., {"final_energy_meV": 5.}, "monitor"))]
    data = composite_dataset_data(group_for(entries))
    np.testing.assert_allclose(data.signal, 20.)
    np.testing.assert_allclose(data.errors, 1. / np.sqrt(3.))
    identity = scientific_data_identity(data)
    assert identity["unit"] == "barn/sr/meV"
    assert identity["normalization_basis"] == "per_formula_unit"
    assert identity["kf_ki_state"] == "removed"
    assert "incident_energy_meV" not in identity
    assert data.metadata[SCIENTIFIC_IDENTITY_KEY] == identity


def test_nested_groups_retain_compatibility_and_independent_backgrounds_and_masks():
    children = []
    backgrounds = []
    for index, (name, energy) in enumerate((
        ("HYSPEC Ei15", {"incident_energy_meV": 15.}),
        ("MACS Ef3.7", {"final_energy_meV": 3.7}),
        ("MACS Ef5", {"final_energy_meV": 5.}),
    )):
        data = histogram(10. * (index + 1), convention=energy)
        sample = DatasetEntry(name, data)
        sample.masks.append(MaskSpec("own HKL region", parameters={"H": [1., 2.]}))
        background = DatasetGroup(f"{name} background", datasets=[
            DatasetEntry("background", histogram(index + 1., convention=energy))], enabled=False)
        child = DatasetGroup(name, datasets=[sample], backgrounds=[
            BackgroundSpec("own background", source_group_id=background.id, source_group=background)])
        children.append(child)
        backgrounds.append(background)
    root = DataGroup("mixed instrument", subgroups=[*children, *backgrounds])
    for child in [*children, *backgrounds]:
        group_for([], root=root, node=child)
    parent_config = data_group_composite_config(root)
    parent_config.update(copy.deepcopy(data_group_composite_config(_composite_scope(root, children[0]))))
    result = composite_dataset_data(root)
    np.testing.assert_allclose(result.signal, 18.)
    assert "background" in result.auxiliary_channels
    np.testing.assert_allclose(result.auxiliary_channels["background"].values, 2.)
    assert all(child.datasets[0].masks for child in children)
    np.testing.assert_allclose(children[0].datasets[0].data.signal, 10.)
    children[2].datasets[0].replace_data(histogram(30., convention={"normalization_basis": "per_unit_cell"}))
    with pytest.raises(ValueError, match="normalization_basis"):
        _cached_composite_dataset_data(root)


def test_cached_composite_does_not_hide_changed_source_units():
    entries = [DatasetEntry("one", histogram()), DatasetEntry("two", histogram())]
    group = group_for(entries)
    _cached_composite_dataset_data(group)
    entries[1].replace_data(histogram(convention={"unit": "mbarn/(sr meV)"}))
    with pytest.raises(ValueError, match="unit"):
        _cached_composite_dataset_data(group)


def test_each_histogram_applies_its_own_background_and_mask_before_pooling():
    first = DatasetEntry("HYSPEC Ei15", histogram(10., convention={"incident_energy_meV": 15.}))
    second = DatasetEntry("MACS Ef3.7", histogram(20., convention={"final_energy_meV": 3.7}))
    left_background = DatasetEntry("HYSPEC background", histogram(1.))
    right_background = DatasetEntry("MACS background", histogram(4.))
    first.backgrounds.append(BackgroundSpec("left", source_dataset_id=left_background.id, source_entry=left_background))
    second.backgrounds.append(BackgroundSpec("right", source_dataset_id=right_background.id, source_entry=right_background))
    group = group_for([first, second, left_background, right_background])
    result = _cached_composite_dataset_data(group)
    np.testing.assert_allclose(result.signal, 12.5)
    np.testing.assert_allclose(result.errors, 1.)
    np.testing.assert_allclose(result.auxiliary_channels["background"].values, 2.5)
    # A source-owned mask removes only that contribution, while its source
    # arrays and the other source's independent background stay unchanged.
    first.masks.append(MaskSpec("exclude reflection", parameters={"H": [-.1, .1]}))
    masked = _cached_composite_dataset_data(group)
    np.testing.assert_allclose(masked.signal, 16.)
    np.testing.assert_allclose(masked.auxiliary_channels["background"].values, 4.)
    np.testing.assert_allclose(first.data.signal, 10.)
    first.masks.clear()
    right_background.replace_data(histogram(6.))
    changed = _cached_composite_dataset_data(group)
    np.testing.assert_allclose(changed.signal, 11.5)


def test_source_background_physical_units_must_match():
    sample = DatasetEntry("sample", histogram())
    background = DatasetEntry("background", histogram(convention={"unit": "mbarn/(sr meV)"}))
    sample.backgrounds.append(BackgroundSpec("background", source_dataset_id=background.id, source_entry=background))
    with pytest.raises(ValueError, match="incompatible scientific units"):
        composite_dataset_data(group_for([sample]))


def test_two_temperature_scientific_combination_requires_compatible_units():
    from nfit.analysis.data_reduction import separate_bose_elastic

    with pytest.raises(ValueError, match="normalization_basis"):
        separate_bose_elastic(histogram(), histogram(convention={"normalization_basis": "per_unit_cell"}),
                             first_temperature_K=5., second_temperature_K=20.)


@pytest.mark.parametrize("kind", ["mdevent", "raw_dgs_nexus", "point"])
def test_unsupported_native_source_backgrounds_are_not_silently_ignored(kind):
    from nfit.dataset import PointData4D

    source = DatasetEntry("source", PointData4D([0.], [0.], [0.], [1.], [10.], [1.]), kind=kind)
    source.backgrounds.append(BackgroundSpec("needs group replay", source_dataset_id="background"))
    group = group_for([source])
    with pytest.raises(ValueError, match="Attach the background to its dataset group"):
        composite_dataset_data(group)


@pytest.mark.parametrize("kind", ["mdevent", "raw_dgs_nexus"])
@pytest.mark.parametrize("field", ["normalization", "signal_per_mbarn"])
def test_lazy_native_entries_reject_different_declared_normalizations_before_reduction(kind, field, monkeypatch):
    import nfit.project_composites as composites

    entries = []
    for index in (1, 2):
        if field == "normalization":
            metadata = {"normalization": f"monitor target {index}"}
            parameters = {}
        else:
            metadata = {}
            parameters = {"spectral_channels": {"source_unit": "arbitrary", "signal_per_mbarn": index}}
        entries.append(DatasetEntry(str(index), None, kind=kind, metadata=metadata, parameters=parameters))
    child = DatasetGroup("native", datasets=entries)
    root = DataGroup("workspace", subgroups=[child])
    scope = group_for([], root=root, node=child)
    monkeypatch.setattr(composites, "bin_mdevent_group", lambda *a, **k: pytest.fail("must validate before native numerical pooling"))
    monkeypatch.setattr(composites, "_raw_dgs_composite_histogram", lambda *a, **k: pytest.fail("must validate before native numerical pooling"))
    with pytest.raises(ValueError, match=field):
        composite_dataset_data(scope)


def test_lazy_entry_declarations_are_checked_without_loading_payloads():
    from nfit.project_composites import _composite_cache_signature

    entries = [DatasetEntry("known", None, kind="mdhisto", data_type="single_crystal_inelastic",
                            metadata={"signal_unit": "barn/sr/meV"}),
               DatasetEntry("unknown", None, kind="mdhisto", data_type="single_crystal_inelastic")]
    group = group_for(entries)
    with pytest.raises(ValueError, match="unit"):
        _composite_cache_signature(group)


def test_compatibility_service_is_gui_independent():
    import nfit.scientific_compatibility as service

    tree = ast.parse(Path(service.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("gui" in name or name in {"project_data", "project_composites"} for name in imports)
