"""Physical conversions retain their scientific and statistical identity."""
from dataclasses import replace

import numpy as np
import pytest

from nfit import coarsen_measurement_histogram
from nfit.analysis import SpectralConvention
from nfit.analysis.spectral import (
    convert_spectral_representation,
    spectral_energy_reduce,
    spectral_kernel,
)
from nfit.background_channels import background_channel
from nfit.cross_section import KB_MEV_PER_K, bose_denominator, cross_section_from_chipp
from nfit.fit_config import FitDatasetInput, compile_fit_problem
from nfit.fitting import evaluate_problem_model
from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
    selected_event_statistics,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_aggregation import selected_measurement_statistics
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import CountingDependencies, independent_source_dependencies
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.pipeline import ModelComponentSpec
from nfit.spectral_channels import default_spectral_channel_config, with_paired_spectral_channels
from nfit.spin_fluctuations import local_relaxational_chipp


def histogram(values=((1.7, 1.7),), *, metadata=None, channels=None, dependencies=None, counting=None, errors=None, mask=None):
    values = np.asarray(values, dtype=float)
    return MDHistoData(
        axes=(MDHistoAxis("Q", np.arange(values.shape[0]+1), "1/angstrom", "momentum"),
              MDHistoAxis("DeltaE", np.arange(values.shape[1]+1)*2., "meV", "energy")),
        signal=values, errors=np.full(values.shape, .2) if errors is None else errors,
        mask=np.zeros(values.shape, bool) if mask is None else mask,
        num_events=np.ones(values.shape), metadata=metadata or {}, auxiliary_channels=channels or {},
        source_dependencies=dependencies, counting_dependencies=counting,
    )


def convention(representation="chi_double_prime", unit="mu_B^2/meV", *, states="removed", kinematic=False):
    return SpectralConvention(representation, unit, "per_formula_unit", 2., "mu_B_squared", 2.,
        states, states, states, "included" if kinematic else "removed", True,
        incident_energy_meV=3. if kinematic else None)


def convert(data, source=None, target="chi_double_prime", **kwargs):
    return convert_spectral_representation(data, convention=source or convention(),
        target_representation=target, temperature_K=kwargs.pop("temperature_K", 25.), **kwargs)


def test_corrected_chipp_identity_and_uncertainties_are_unchanged():
    data = histogram()
    actual = convert(data, form_factor_sq=.8, polarization=2.)
    np.testing.assert_array_equal(actual.signal, data.signal)
    np.testing.assert_array_equal(actual.errors, data.errors)


def test_structure_factor_to_chipp_removes_only_existing_projection():
    data = histogram(((3., 3.),))
    source = replace(convention("s_qw"), bose_state="included")
    actual = convert(data, source, form_factor_sq=.8, polarization=2.)
    expected_factor = np.pi * (1-np.exp(-np.array([[1., 3.]])/(KB_MEV_PER_K*25.)))
    np.testing.assert_allclose(actual.signal, data.signal*expected_factor)
    np.testing.assert_allclose(actual.errors, data.errors*expected_factor)


@pytest.mark.parametrize("states", ["included", "removed"])
def test_cross_section_target_records_the_factors_it_actually_contains(states):
    data = histogram()
    source = convention("cross_section", "barn/(sr meV)", states=states)
    direct = convert(data, source, form_factor_sq=.8, polarization=2.)
    cross = convert(data, source, "cross_section", form_factor_sq=.8, polarization=2.)
    result = convert(cross, SpectralConvention.from_dict(cross.metadata["spectral_convention"]),
        form_factor_sq=.8, polarization=2.)
    np.testing.assert_allclose(result.signal, direct.signal)
    np.testing.assert_allclose(result.errors, direct.errors)


def counted_data(*, uncertain=False):
    counts = np.array([[4., 9.]])
    exposure = np.ones(counts.shape)
    dep = independent_source_dependencies(counts, "counts")
    bundle = None
    if uncertain:
        bundle = CountingDependencies(numerator_dependencies=dep,
            exposure_dependencies=independent_source_dependencies(np.full(counts.shape,.04), "exposure"))
    contract = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="intensity",
        value_units="counts", exposure_units="N", dependence="shared_sources", normalizer="uncertain" if uncertain else "known")
    return histogram(counts, errors=np.sqrt(counts+counts**2*.04 if uncertain else counts), dependencies=dep, counting=bundle,
        metadata={"measurement_contract":contract.to_dict(), EVENT_STATISTICS_KEY:dict(EVENT_STATISTICS_METADATA)},
        channels=event_statistics_channels(counts, counts, exposure))


@pytest.mark.parametrize("uncertain", [False, True])
def test_absolute_calibration_preserves_count_statistics_and_dependencies(uncertain):
    data = counted_data(uncertain=uncertain)
    actual = convert(data, convention("measured_intensity", "counts", states="included"),
        "cross_section", scale=2.)
    statistics = selected_event_statistics(actual)
    assert statistics is not None
    np.testing.assert_allclose(statistics[0], [[2., 4.5]])
    np.testing.assert_allclose(statistics[1], [[1., 2.25]])
    assert actual.metadata["measurement_contract"]["value_units"] == "barn/(sr meV)"
    assert actual.source_dependencies is not None
    assert (actual.counting_dependencies is not None) == uncertain
    np.testing.assert_allclose(actual.source_dependencies.variance(), actual.errors**2)
    pooled = coarsen_measurement_histogram(actual, ([0.,1.], [0.,4.]))
    assert pooled.signal.item() == pytest.approx(3.25)
    expected_variance = 3.25/4. + (.08*(3.25/2.)**2 if uncertain else 0.)
    assert pooled.errors.item() == pytest.approx(np.sqrt(expected_variance))


def test_deterministic_background_is_a_derived_target_with_preserved_sensitivities():
    data = counted_data(uncertain=True)
    actual = convert(data, convention("measured_intensity", "counts", states="included"),
        "cross_section", scale=2., background=2.)
    np.testing.assert_allclose(actual.signal, [[1., 3.5]])
    np.testing.assert_allclose(actual.errors, data.errors/2.)
    assert actual.source_dependencies is not None and actual.counting_dependencies is None
    np.testing.assert_allclose(actual.source_dependencies.variance(), actual.errors**2)
    assert "measurement_contract" not in actual.metadata
    assert EVENT_STATISTICS_KEY not in actual.metadata
    assert actual.metadata["measurement_target_required"]
    np.testing.assert_allclose(background_channel(actual, "unsubtracted").values, data.signal/2.)
    np.testing.assert_allclose(actual.auxiliary_channels["background"].values, 1.)


def test_continuous_mean_retains_its_shifted_sufficient_statistics():
    contract = MeasurementContract(kind="continuous", estimator="uniform_mean", quantity="intensity", value_units="U")
    data = histogram(((1., 3.),), metadata={"measurement_contract":contract.to_dict()})
    prepared = coarsen_measurement_histogram(data, ([0.,1.], [0.,2.,4.]))
    actual = convert(prepared, convention("measured_intensity", "counts", states="included"),
        "cross_section", scale=2., background=1.)
    assert selected_measurement_statistics(actual) is not None
    pooled = coarsen_measurement_histogram(actual, ([0.,1.], [0.,4.]))
    assert pooled.signal.item() == pytest.approx(.5)
    assert pooled.errors.item() == pytest.approx(np.sqrt(2)*.1/2)


def test_existing_background_channels_receive_the_same_physical_multiplier():
    from nfit.mdhisto import MDHistoChannel

    data = histogram(((4., 4.),), metadata={"background_channels_version":1},
        channels={"background":MDHistoChannel(np.array([[2.,2.]]), np.zeros((1,2)), unit="counts")})
    actual = convert(data, convention("measured_intensity", "counts", states="included"),
        "cross_section", scale=2.)
    np.testing.assert_allclose(actual.auxiliary_channels["background"].values, [[1.,1.]])
    assert actual.auxiliary_channels["background"].unit == "barn/sr/meV"
    np.testing.assert_allclose(background_channel(actual, "unsubtracted").values, [[3.,3.]])


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("selected", ["cross_section", "chi_double_prime"])
def test_inaccessible_masked_cell_does_not_block_the_paired_source_view(absolute, selected):
    data = histogram(mask=np.array([[False,True]]))
    config = default_spectral_channel_config()
    config.update(kf_ki_state="included", incident_energy_meV=3., fit_representation=selected)
    if absolute:
        config["source_unit"] = "mbarn/sr/meV"
    result = with_paired_spectral_channels(data, config, temperature_K=25.)
    assert np.isfinite(result.signal[0,0])
    assert result.mask[0,1]
    if selected == "cross_section":
        np.testing.assert_array_equal(result.signal, data.signal)
    else:
        assert np.isnan(result.signal[0,1])


def test_analysis_inaccessible_cell_is_masked_without_losing_valid_cells():
    data = histogram()
    source = convention("cross_section", "barn/(sr meV)", states="included", kinematic=True)
    actual = convert(data, source)
    assert np.isfinite(actual.signal[0,0]) and not actual.mask[0,0]
    assert np.isnan(actual.signal[0,1]) and actual.mask[0,1]


def test_zero_energy_forward_conversion_masks_missing_slope_but_keeps_source_chipp():
    data = histogram()
    data = data.with_updates(axes=(data.axes[0], MDHistoAxis("DeltaE", [-1.,1.,3.], "meV", "energy")),
        signal=np.array([[0.,1.]]))
    cross = convert(data, target="cross_section")
    assert cross.mask[0,0] and np.isnan(cross.signal[0,0])
    identity = convert(data)
    assert not identity.mask[0,0] and identity.signal[0,0] == 0.


@pytest.mark.parametrize("bad_temperature", [np.nan, np.inf, -1., [20.,np.nan]])
def test_bose_conversion_rejects_invalid_temperature(bad_temperature):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert(histogram(), convention("cross_section", "barn/(sr meV)", states="included"),
            temperature_K=bad_temperature)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        bose_denominator([1.,3.], bad_temperature)


def test_temperature_array_conversion_and_reduction_match_individual_samples():
    data = histogram(((1.,1.),(1.,1.)))
    temperature = np.array([[0.],[25.]])
    source = convention("cross_section", "barn/(sr meV)", states="included")
    actual = convert(data, source, temperature_K=temperature)
    for row, value in enumerate((0.,25.)):
        expected = convert(histogram(), source, temperature_K=value)
        np.testing.assert_allclose(actual.signal[row], expected.signal[0]/1.7)
    assert actual.metadata["spectral_conversion"]["temperature_K"] == [[0.],[25.]]
    result = spectral_energy_reduce(actual, kernel="qfi", convention=convention(),
        temperature_K=temperature, energy_max_meV=4.)
    expected = np.sum(actual.signal*spectral_kernel("qfi", np.array([[1.,3.]]), temperature)*2., axis=1)/4.
    np.testing.assert_allclose(result.data.column("Quantum Fisher information"), expected)


def test_converted_chipp_replaces_stale_fit_observable():
    energy=np.array([1.,3.])
    expected=4.*local_relaxational_chipp(energy, chi_loc=1.5, gamma=2.5)
    cross=1000*cross_section_from_chipp(expected, energy, 20., polarization=2.)
    data=histogram(cross[None,:], metadata={"temperature":20., "spectral_observable": {
        "fit_representation":"cross_section", "moment_unit":"mu_B_squared", "g_factor":2.,
        "unit":"mbarn/sr/meV", "polarization_mode":"isotropic_single_component"}})
    actual=convert(data, convention("cross_section", "mbarn/(sr meV)", states="included"),
        temperature_K=20., polarization=2.)
    assert actual.metadata["spectral_observable"]["fit_representation"] == "chi_double_prime"
    points=prepare_histogram_fit_points(actual)
    component=ModelComponentSpec("loc", "local_relaxational", parameters={"chi_loc":1.5,"gamma":2.5}, fit_parameters={})
    compiled=compile_fit_problem([component], [FitDatasetInput("data",points)])
    parameters={spec.name:spec.value for spec in compiled.problem.parameter_specs}
    np.testing.assert_allclose(evaluate_problem_model(compiled.problem,"data",parameters),expected)


def test_conversion_service_stays_independent_of_gui_coordinators():
    import ast
    from pathlib import Path

    import nfit.spectral_conversion as service

    tree=ast.parse(Path(service.__file__).read_text())
    imports=[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert all("qt" not in name.lower() and "gui" not in name.lower() for name in imports)
    assert "project_data" not in imports


@pytest.mark.parametrize("representation,unit", [
    ("chi_double_prime", "mu_B^2/meV"), ("s_qw", "mu_B^2/meV"),
    ("cross_section", "barn/(sr meV)"),
])
def test_absolute_conversion_requires_an_absolute_source_declaration(representation, unit):
    source = replace(convention(representation, unit), absolute_scale=False)
    with pytest.raises(ValueError, match="absolute intensity scale"):
        convert(histogram(), source)


def test_absolute_conversion_requires_an_explicit_sample_basis():
    source = replace(convention("measured_intensity", "counts", states="included"), normalization_basis="unknown")
    with pytest.raises(ValueError, match="known normalization basis"):
        convert(histogram(), source)


def test_cached_bose_factor_cannot_bypass_temperature_validation():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cross_section_from_chipp([1.], [1.], np.nan, bose_denominator_values=[1.])


def test_pointwise_polarization_is_preserved_and_never_approximated_for_fitting():
    from nfit.spectral_channels import spectral_polarization_factor

    data = histogram()
    actual = convert(data, convention("cross_section", "barn/(sr meV)", states="included"),
        polarization=np.array([[1.,2.]]))
    scalar = convert(data, convention("cross_section", "barn/(sr meV)", states="included"))
    np.testing.assert_allclose(actual.signal, scalar.signal/[[1.,2.]])
    assert actual.metadata["spectral_observable"]["polarization_scalar"] == [[1.,2.]]
    with pytest.raises(ValueError, match="array polarization.*not supported"):
        spectral_polarization_factor(actual.metadata["spectral_observable"])


def test_explicit_conversion_temperature_replaces_stale_imported_scalar():
    data = histogram(metadata={"temperature":10.})
    actual = convert(data, convention("cross_section", "barn/(sr meV)", states="included"),
        temperature_K=25.)
    assert actual.metadata["temperature"] == 25.
    assert prepare_histogram_fit_points(actual).temperature == 25.


@pytest.mark.parametrize("source_representation,target_representation", [
    ("cross_section", "chi_double_prime"), ("chi_double_prime", "cross_section"),
])
def test_paired_primary_convention_is_authoritative_for_public_reconversion(source_representation, target_representation):
    source = convention(source_representation,
        "barn/(sr meV)" if source_representation == "cross_section" else "mu_B^2/meV",
        states="included" if source_representation == "cross_section" else "removed")
    data = histogram(metadata={"spectral_convention":source.to_dict()})
    config = default_spectral_channel_config()
    config.update(source_representation=source_representation, fit_representation=target_representation,
        source_unit="barn/sr/meV/f.u." if source_representation == "cross_section" else "mu_B^2/meV/f.u.")
    paired = with_paired_spectral_channels(data, config, temperature_K=25.)
    recorded = SpectralConvention.from_dict(paired.metadata["spectral_convention"])
    assert recorded.representation == target_representation
    assert recorded.unit == paired.metadata["signal_unit"]
    assert recorded.magnetic_ions_per_basis == 2.
    expected_state = "removed" if target_representation == "chi_double_prime" else "included"
    assert recorded.form_factor_state == recorded.polarization_state == recorded.bose_state == expected_state
    actual = convert(paired, recorded, target_representation, polarization=2.)
    expected_factor = .001 if target_representation == "cross_section" else 1.
    np.testing.assert_allclose(actual.signal, paired.signal * expected_factor)
    np.testing.assert_allclose(actual.errors, paired.errors * expected_factor)
