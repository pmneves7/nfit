"""Sparse primitive identity, propagation, bounded storage and archive reuse."""

import numpy as np
import pytest

from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.backgrounds import (
    background_with_user_mask_zeros,
    subtract_aligned_background,
    subtract_powder_background,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_dependencies import (
    CountingDependencies,
    SourceDependencies,
    SourceReplayRequired,
    combine_source_dependencies,
    independent_source_dependencies,
    project_source_dependencies,
    ratio_source_dependencies,
)
from nfit.pipeline import DatasetEntry
from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file


def histogram(signal=(4., 6.), dependencies=None, errors=None, metadata=None):
    signal = np.asarray(signal)
    if errors is None:
        errors = np.ones(signal.shape) if dependencies is None else np.sqrt(dependencies.variance())
    return MDHistoData(axes=(MDHistoAxis("x", np.arange(len(signal)+1), "K", "unknown"),),
        signal=signal, errors=np.asarray(errors), mask=np.zeros(signal.shape, dtype=bool),
        num_events=np.array([5, 8]), metadata={"signal_semantics": "density", **(metadata or {})},
        source_dependencies=dependencies)


def test_fractional_copies_reunite_before_squaring():
    primitive = independent_source_dependencies(np.array([4.]), "event")
    split = project_source_dependencies(primitive, [0, 0], [0, 1], [.5, .5], (2,))
    np.testing.assert_array_equal(split.variance(), [1, 1])
    pooled = project_source_dependencies(split, [0, 1], [0, 0], [1, 1], (1,))
    np.testing.assert_array_equal(pooled.variance(), [4])
    copies = project_source_dependencies(primitive, [0, 0], [0, 1], [1, 1], (2,))
    pooled = project_source_dependencies(copies, [0, 1], [0, 0], [1, 1], (1,))
    np.testing.assert_array_equal(pooled.variance(), [16])


def test_signed_sources_cancel_and_independent_sources_add():
    a = independent_source_dependencies(np.array([4., 9.]), "sample")
    canceled = combine_source_dependencies((a, a), (1, -1))
    np.testing.assert_array_equal(canceled.variance(), [0, 0])
    b = independent_source_dependencies(np.array([4., 9.]), "other")
    np.testing.assert_array_equal(combine_source_dependencies((a, b), (1, -2)).variance(), [20, 45])
    inconsistent = independent_source_dependencies(np.array([4., 10.]), "sample")
    with pytest.raises(ValueError, match="Inconsistent variance"):
        combine_source_dependencies((a, inconsistent), (1, 1))


def test_shared_calibration_term_does_not_average_away():
    dependence = SourceDependencies(shape=(2,), source_ids=("count0", "count1", "calibration"),
        source_variances=np.array([1., 1., .04]), observation_indices=np.array([0, 0, 1, 1]),
        source_indices=np.array([0, 2, 1, 2]), coefficients=np.array([1., 10., 1., 10.]))
    mean = project_source_dependencies(dependence, [0, 1], [0, 0], [.5, .5], (1,))
    assert mean.variance().item() == pytest.approx(.5 + 4)


def test_source_variance_multiplication_avoids_intermediate_overflow():
    dependencies = SourceDependencies(shape=(1,), source_ids=("scaled",), source_variances=np.array([1e-300]),
        observation_indices=np.array([0]), source_indices=np.array([0]), coefficients=np.array([1e300]))
    assert dependencies.variance().item() == pytest.approx(1e300)


def uncertain_count_data():
    from nfit.histogram_statistics import (
        EVENT_STATISTICS_KEY,
        EVENT_STATISTICS_METADATA,
        event_statistics_channels,
    )
    from nfit.measurement_contracts import MeasurementContract

    numerator = np.array([10., 0.])
    exposure = np.array([2., 2.])
    bundle = CountingDependencies(numerator_dependencies=independent_source_dependencies(numerator, "counts"),
        exposure_dependencies=independent_source_dependencies(np.array([.04, .04]), "exposure"))
    primary = ratio_source_dependencies(bundle, numerator, exposure)
    contract = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="rate", value_units="1",
        exposure_units="s", normalizer="uncertain", dependence="shared_sources")
    return MDHistoData(axes=(MDHistoAxis("x", np.array([0., 1., 2.]), "1", "unknown"),),
        signal=numerator/exposure, errors=np.sqrt(primary.variance()), mask=np.zeros(2, bool), num_events=numerator,
        metadata={EVENT_STATISTICS_KEY: EVENT_STATISTICS_METADATA, "measurement_contract": contract.to_dict(),
                  "signal_semantics": "density"},
        auxiliary_channels=event_statistics_channels(numerator, numerator, exposure), counting_dependencies=bundle)


def test_uncertain_exposure_bundle_and_shared_ratio_cancellation(tmp_path):
    from nfit.histogram_statistics import selected_event_statistics

    data = uncertain_count_data()
    np.testing.assert_allclose(data.errors**2, [2.75, 0])
    np.testing.assert_array_equal(selected_event_statistics(data, (slice(0, 1),))[0], [10])
    assert data.metadata["counting_uncertainty"] == "delta_method"
    assert data.source_dependencies is not None
    path = tmp_path / "uncertain.npz"
    write_dataset_artifact(data, path)
    restored = read_dataset_artifact(path)
    assert restored.counting_dependencies is not None
    np.testing.assert_allclose(restored.errors**2, [2.75, 0])
    assert restored.with_updates(signal=restored.signal+1).counting_dependencies is None
    c = SourceDependencies(shape=(1,), source_ids=("calibration",), source_variances=np.array([.01]),
        observation_indices=np.array([0]), source_indices=np.array([0]), coefficients=np.array([10.]))
    n = SourceDependencies(shape=(1,), source_ids=("calibration",), source_variances=np.array([.01]),
        observation_indices=np.array([0]), source_indices=np.array([0]), coefficients=np.array([2.]))
    shared = ratio_source_dependencies(CountingDependencies(numerator_dependencies=c, exposure_dependencies=n), np.array([10.]), np.array([2.]))
    np.testing.assert_array_equal(shared.variance(), [0])


@pytest.mark.parametrize("artifact", [True, False])
def test_point_data_archive_retains_payload_and_sources(tmp_path, artifact):
    from nfit.dataset import PointData4D

    sources = independent_source_dependencies(np.array([4., 9.]), "points")
    points = PointData4D(H=[1, 2], K=[0, 0], L=[0, 0], E=[3, 4], intensity=[4, 5], sigma=[2, 3],
        temperature=[5, 10], magnetic_field=[0, 0, 1], normalization_denominator=[2, 4],
        measurement_payload={"measurement_value_numerator": [8, 20]}, source_dependencies=sources)
    path = tmp_path / "points.npz"
    if artifact:
        write_dataset_artifact(points, path)
        restored = read_dataset_artifact(path, memory_map=True)
    else:
        save_dataset_file(DatasetEntry(name="points", data=points), path, use_view=False)
        restored, _ = _load_nfit_dataset_file(path)
    np.testing.assert_array_equal(restored.intensity, points.intensity)
    np.testing.assert_array_equal(restored.temperature, points.temperature)
    np.testing.assert_array_equal(restored.normalization_denominator, [2, 4])
    np.testing.assert_array_equal(restored.measurement_payload["measurement_value_numerator"], [8, 20])
    np.testing.assert_array_equal(restored.source_dependencies.variance(), [4, 9])


def test_mask_selection_and_weighted_direct_staged_equivalence():
    original = independent_source_dependencies(np.array([4., 9., 16.]), "source")
    selected = project_source_dependencies(original, [0, 2], [0, 0], [.2, .8], (1,))
    assert selected.variance().item() == pytest.approx(.2**2*4 + .8**2*16)
    stage = project_source_dependencies(original, [0, 1, 2], [0, 1, 1], [1, .5, .5], (2,))
    staged = project_source_dependencies(stage, [0, 1], [0, 0], [.5, .5], (1,))
    direct = project_source_dependencies(original, [0, 1, 2], [0, 0, 0], [.5, .25, .25], (1,))
    np.testing.assert_array_equal(staged.variance(), direct.variance())


def test_invalid_payloads_and_budget_require_replay():
    original = independent_source_dependencies(np.array([4., 9.]), "source")
    with pytest.raises(SourceReplayRequired, match="budget"):
        project_source_dependencies(original, [0, 0, 0], [0, 0, 0], [1, 1, 1], (1,), max_bytes=16)
    with pytest.raises(SourceReplayRequired):
        project_source_dependencies(None, [0], [0], [1], (1,))
    with pytest.raises(ValueError, match="reproduce"):
        histogram(dependencies=original, errors=[1, 1])
    with pytest.raises(ValueError, match="assignment"):
        project_source_dependencies(original, [4], [0], [1], (1,))
    with pytest.raises(SourceReplayRequired, match="budget"):
        SourceDependencies(shape=(10**12,), source_ids=(), source_variances=np.array([]),
            observation_indices=np.array([], dtype=int), source_indices=np.array([], dtype=int), coefficients=np.array([]))


def test_container_preserves_masks_and_invalidates_primary_changes():
    dependencies = independent_source_dependencies(np.array([4., 9.]), "source")
    data = histogram(dependencies=dependencies)
    assert data.with_updates(mask=np.array([True, False])).source_dependencies is dependencies
    assert data.with_updates(signal=data.signal + 1).source_dependencies is None
    assert data.mutable_copy().source_dependencies is None
    assert not dependencies.coefficients.flags.writeable
    with pytest.raises(ValueError):
        dependencies.coefficients[0] = 2


def test_primary_replacement_clears_inherited_measurement_declarations():
    from nfit.measurement_aggregation import (
        initialize_measurement_statistics,
        measurement_statistics_channels,
    )
    from nfit.measurement_contracts import MeasurementContract

    contract = MeasurementContract(kind="continuous", estimator="uniform_mean", quantity="response", value_units="1")
    arrays, marker = initialize_measurement_statistics([4, 6], [1, 1], contract)
    data = histogram(metadata={"measurement_contract": contract.to_dict(), "measurement_statistics": marker})
    data = data.with_updates(auxiliary_channels=measurement_statistics_channels(*arrays))
    changed = data.with_updates(signal=data.signal + 1)
    assert "measurement_contract" not in changed.metadata
    assert changed.metadata["measurement_target_required"]
    assert "measurement_statistics" not in changed.metadata
    assert "measurement_value_numerator" not in changed.auxiliary_channels
    explicit = data.with_updates(signal=data.signal.copy(), metadata=data.metadata,
                                 auxiliary_channels=data.auxiliary_channels)
    assert explicit.metadata["measurement_contract"] == contract.to_dict()
    working = data.mutable_copy()
    assert "measurement_contract" not in working.metadata
    assert "measurement_statistics" not in working.metadata
    assert working.metadata["measurement_target_required"]
    assert "measurement_value_numerator" not in working.auxiliary_channels


@pytest.mark.parametrize("mapped", [False, True])
def test_project_artifact_roundtrip_retains_sources(tmp_path, monkeypatch, mapped):
    from nfit.analysis import artifacts

    dependencies = independent_source_dependencies(np.array([4., 9.]), "source")
    data = histogram(dependencies=dependencies)
    path = tmp_path / "sources.npz"
    write_dataset_artifact(data, path)
    monkeypatch.setattr(artifacts, "_MAPPED_MEMBER_MIN_BYTES", 0)
    restored = read_dataset_artifact(path, memory_map=mapped)
    assert restored.source_dependencies.source_ids == dependencies.source_ids
    np.testing.assert_array_equal(restored.source_dependencies.variance(), dependencies.variance())
    assert not restored.source_dependencies.coefficients.flags.writeable
    if mapped:
        from nfit.mapped_archive import array_storage_nbytes

        assert array_storage_nbytes(restored.source_dependencies).mapped > 0


def test_portable_dataset_roundtrip_retains_sources(tmp_path):
    dependencies = independent_source_dependencies(np.array([4., 9.]), "source")
    data = histogram(dependencies=dependencies)
    path = tmp_path / "sources.npz"
    save_dataset_file(DatasetEntry(name="source", data=data), path, use_view=False)
    restored, _ = _load_nfit_dataset_file(path)
    np.testing.assert_array_equal(restored.source_dependencies.variance(), [4, 9])
    with np.load(path, allow_pickle=False) as archive:
        assert "source_dependencies_coefficients" in archive
        assert "source_dependencies" not in str(archive["metadata_json"])


def test_aligned_background_cancellation_and_independent_fallback():
    source = independent_source_dependencies(np.array([4., 9.]), "source")
    data = histogram(dependencies=source)
    result = subtract_aligned_background(data, data)
    np.testing.assert_array_equal(result.signal, [0, 0])
    np.testing.assert_array_equal(result.errors, [0, 0])
    np.testing.assert_array_equal(result.num_events, data.num_events)
    assert result.metadata["background_uncertainty_dependence"] == "tracked_sources"
    independent = subtract_aligned_background(histogram(), histogram())
    np.testing.assert_allclose(independent.errors, np.sqrt(2))
    assert independent.metadata["background_uncertainty_dependence"] == "assumed_independent"
    with pytest.raises(SourceReplayRequired):
        subtract_aligned_background(data, histogram())


def test_explicit_background_mask_zeros_source_sensitivities():
    source = independent_source_dependencies(np.array([4., 9.]), "source")
    background = histogram(dependencies=source, metadata={"nfit_mask": [True, False]})
    masked = background_with_user_mask_zeros(background)
    np.testing.assert_array_equal(masked.source_dependencies.variance(), [0, 9])
    assert masked.num_events[0] == 0
    np.testing.assert_array_equal(background.source_dependencies.variance(), [4, 9])


def test_background_interpolation_retains_reused_primitive_variance():
    axes = (MDHistoAxis("|Q|", np.array([0., 1., 2.]), "A^-1", "momentum"),
            MDHistoAxis("DeltaE", np.array([0., 1., 2.]), "meV", "energy"))
    background_sources = independent_source_dependencies(np.ones((2, 2))*4, "background")
    background = MDHistoData(axes=axes, signal=np.ones((2, 2))*3, errors=np.ones((2, 2))*2,
        mask=np.zeros((2, 2), bool), num_events=np.ones((2, 2)), source_dependencies=background_sources)
    sample_axes = (MDHistoAxis("|Q|", np.array([.75, 1.25]), "A^-1", "momentum"),
                   MDHistoAxis("DeltaE", np.array([.75, 1.25]), "meV", "energy"))
    sample = MDHistoData(axes=sample_axes, signal=np.array([[7.]]), errors=np.array([[1.]]),
        mask=np.zeros((1, 1), bool), num_events=np.ones((1, 1)),
        source_dependencies=independent_source_dependencies(np.ones((1, 1)), "sample"))
    once = subtract_powder_background(sample, background)
    np.testing.assert_allclose(once.signal, [[4.]])
    np.testing.assert_allclose(once.errors**2, [[2.]])  # 1 + four (1/4)^2 * 4.
    twice = subtract_powder_background(once, background)
    np.testing.assert_allclose(twice.signal, [[1.]])
    np.testing.assert_allclose(twice.errors**2, [[5.]])  # Reused background variance: 1 + 4*1.


def test_bragg_linear_integrals_retain_common_source_and_shell_cancellation():
    from nfit.analysis.bragg import integrate_bragg_peaks

    shape = (5, 5, 5)
    sources = SourceDependencies(shape=shape, source_ids=("common",), source_variances=np.array([1.]),
        observation_indices=np.arange(np.prod(shape)), source_indices=np.zeros(np.prod(shape), dtype=int),
        coefficients=np.ones(np.prod(shape)))
    axes = tuple(MDHistoAxis(name, np.arange(6.), "rlu", "momentum") for name in ("H", "K", "L"))
    data = MDHistoData(axes=axes, signal=np.full(shape, 7.), errors=np.ones(shape), mask=np.zeros(shape, bool),
        num_events=np.ones(shape), source_dependencies=sources,
        metadata={"signal_semantics": "density", "lattice_parameters": {"a": 2*np.pi, "b": 2*np.pi, "c": 2*np.pi}})
    result = integrate_bragg_peaks(data, [[2.5, 2.5, 2.5]], method="box_sum", box_half_widths=[.5]*3)
    assert result.column("dI")[0] == pytest.approx(1)
    subtracted = integrate_bragg_peaks(data, [[2.5, 2.5, 2.5]], method="box_sum", box_half_widths=[.5]*3,
        background_mode="shell", background_inner_scale=1, background_outer_scale=2)
    assert subtracted.column("I")[0] == pytest.approx(0)
    assert subtracted.column("dI")[0] == pytest.approx(0, abs=1e-14)
    assert "source replay required" in subtracted.metadata["cross_reflection_covariance"]
    with pytest.raises(SourceReplayRequired, match="correlated likelihood"):
        integrate_bragg_peaks(data, [[2.5, 2.5, 2.5]], method="gaussian_fit", ellipsoid_semiaxes=[.5]*3)


def test_elastic_bragg_volume_preserves_covariance_and_real_contributions():
    from nfit.analysis.bragg import _elastic_reduce

    shape = (1, 1, 1, 2)
    sources = SourceDependencies(shape=shape, source_ids=("common",), source_variances=np.array([1.]),
        observation_indices=np.array([0, 1]), source_indices=np.array([0, 0]), coefficients=np.array([1., 1.]))
    axes = tuple(MDHistoAxis(name, np.array([0., 1.]), "rlu", "momentum") for name in ("H", "K", "L"))
    axes += (MDHistoAxis("DeltaE", np.array([0., 1., 2.]), "meV", "energy"),)
    data = MDHistoData(axes=axes, signal=np.full(shape, 7.), errors=np.ones(shape), mask=np.zeros(shape, bool),
        num_events=np.array([[[[3., 5.]]]]), source_dependencies=sources,
        metadata={"signal_semantics": "density"})
    volume = _elastic_reduce(data, 0, 2)
    assert volume.signal.item() == 14
    assert volume.errors.item() == 2
    assert volume.num_events.item() == 8
    assert volume.metadata["measurement_contract"]["estimator"] == "linear_sum"


def continuous_histogram(estimator="uniform_mean"):
    from nfit.measurement_aggregation import (
        initialize_measurement_statistics,
        measurement_statistics_channels,
    )
    from nfit.measurement_contracts import MeasurementContract

    contract = MeasurementContract(kind="continuous", estimator=estimator, quantity="response", value_units="1")
    stats, marker = initialize_measurement_statistics([2., 6.], [1., 2.], contract)
    return histogram(signal=[2., 6.], errors=[1., 2.], metadata={"measurement_contract": contract.to_dict(),
        "measurement_statistics": marker}).with_updates(num_events=np.zeros(2),
            auxiliary_channels=measurement_statistics_channels(*stats))


def test_continuous_support_uses_measurements_not_event_contributions():
    from nfit.mdhisto import mdhisto_coverage_fraction, mdhisto_measured_bins

    data = continuous_histogram()
    np.testing.assert_array_equal(mdhisto_measured_bins(data), [True, True])
    np.testing.assert_array_equal(mdhisto_coverage_fraction(data), [1, 1])
    masked = data.with_updates(mask=np.array([True, False]))
    np.testing.assert_array_equal(mdhisto_measured_bins(masked), [False, True])
    np.testing.assert_array_equal(mdhisto_coverage_fraction(masked), [1, 1])  # Geometric support precedes masks.
    plain = histogram().with_updates(num_events=np.zeros(2))
    np.testing.assert_array_equal(mdhisto_measured_bins(plain), [False, False])
    np.testing.assert_array_equal(mdhisto_coverage_fraction(plain), [0, 0])


def test_background_derivation_drops_stale_statistics_and_keeps_support():
    from nfit.mdhisto import mdhisto_measured_bins
    from nfit.measurement_aggregation import MEASUREMENT_STATISTICS_CHANNELS

    continuous = continuous_histogram()
    result = subtract_aligned_background(continuous, continuous)
    assert "measurement_contract" not in result.metadata
    assert "measurement_statistics" not in result.metadata
    assert not any(name in result.auxiliary_channels for name in MEASUREMENT_STATISTICS_CHANNELS)
    assert result.metadata["measurement_target_required"]
    assert result.metadata["measurement_derivation"]["sample_contract"] == continuous.metadata["measurement_contract"]
    np.testing.assert_array_equal(mdhisto_measured_bins(result), [True, True])
    counted = uncertain_count_data()
    result = subtract_aligned_background(counted, counted)
    assert "event_statistics" not in result.metadata
    assert "normalization_denominator" not in result.auxiliary_channels
    assert result.source_dependencies is not None
    assert result.counting_dependencies is None
    np.testing.assert_array_equal(result.source_dependencies.variance(), [0, 0])
    legacy = subtract_aligned_background(histogram(), histogram())
    assert "measurement_target_required" not in legacy.metadata


def test_native_legacy_background_keeps_sample_exposure_without_stale_count_statistics():
    from nfit.histogram_statistics import EVENT_STATISTICS_METADATA, event_statistics_channels

    data = histogram(signal=[2, 3], errors=[1, 1], metadata={"event_statistics": EVENT_STATISTICS_METADATA,
        "normalization_denominator": np.array([2., 4.])})
    data = data.with_updates(auxiliary_channels=event_statistics_channels(np.array([4., 12.]), np.array([4., 16.]), np.array([2., 4.])))
    result = subtract_aligned_background(data, data)
    assert "measurement_target_required" not in result.metadata
    assert "event_statistics" not in result.metadata
    np.testing.assert_array_equal(result.metadata["normalization_denominator"], [2, 4])
    np.testing.assert_array_equal(result.auxiliary_channels["normalization_denominator"].values, [2, 4])
    assert "event_signal_numerator" not in result.auxiliary_channels


def test_metadata_stack_fails_explicitly_before_dropping_measurement_dependencies():
    from nfit.metadata_dimensions import MetadataDimension, stack_metadata_histograms

    declared = continuous_histogram()
    with pytest.raises(SourceReplayRequired, match="Metadata stacking"):
        stack_metadata_histograms(declared, {(0,): declared},
            [MetadataDimension("Temperature", "parameters/temperature", "K")], [np.array([5.])])


def test_spectral_conversion_preserves_count_dependencies_and_updates_quantity():
    from nfit.histogram_statistics import selected_event_statistics
    from nfit.spectral_channels import with_paired_spectral_channels

    data = uncertain_count_data().with_updates(axes=(MDHistoAxis("DeltaE", np.array([-2., 0., 2.]), "meV", "energy"),))
    converted = with_paired_spectral_channels(data, {"enabled": True, "fit_representation": "chi_double_prime"}, temperature_K=5)
    assert converted.counting_dependencies is not None
    assert selected_event_statistics(converted) is not None
    np.testing.assert_allclose(converted.source_dependencies.variance(), converted.errors**2)
    assert "measurement_contract" not in converted.metadata
    assert converted.metadata["measurement_target_required"]
    assert converted.metadata["measurement_derivation"]["quantity"] == "dynamic_susceptibility"
    assert converted.metadata["measurement_derivation"]["value_units"] == converted.metadata["signal_unit"]
    assert converted.signal[0] < 0


def test_deterministic_scaling_preserves_count_bundle_and_precision_statistics():
    from nfit.histogram_statistics import selected_event_statistics
    from nfit.measurement_aggregation import (
        normalized_measurement_statistics,
        selected_measurement_statistics,
    )
    from nfit.measurement_scaling import scale_measurement_data

    counted = uncertain_count_data()
    scaled = scale_measurement_data(counted, 2)
    np.testing.assert_allclose(scaled.signal, counted.signal*2)
    np.testing.assert_allclose(scaled.source_dependencies.variance(), counted.source_dependencies.variance()*4)
    assert scaled.counting_dependencies is not None
    assert selected_event_statistics(scaled) is not None
    continuous = continuous_histogram("inverse_variance_mean")
    scaled = scale_measurement_data(continuous, [2., 4.])
    stats = selected_measurement_statistics(scaled)
    assert stats is not None
    np.testing.assert_allclose(scaled.signal, [4, 24])
    np.testing.assert_allclose(scaled.errors, [2, 8])
    pooled = tuple(np.array([np.sum(a)]) for a in stats)
    signal, variance = normalized_measurement_statistics(*pooled)
    assert signal.item() == pytest.approx((4/4+24/64)/(1/4+1/64))
    assert variance.item() == pytest.approx(1/(1/4+1/64))
    assert scaled.metadata["measurement_statistics"]["precision_scale"] == 2


def test_exact_zero_scaling_preserves_payload_and_removes_poisson_certificate():
    from nfit.composite_scaling import scale_composite_data
    from nfit.histogram_statistics import selected_event_statistics
    from nfit.measurement_likelihoods import PoissonCountModel
    from nfit.measurement_scaling import scale_measurement_data
    from nfit.project_view_data import _apply_dataset_scale

    counted = uncertain_count_data().with_updates(metadata={**uncertain_count_data().metadata,
        "poisson_count_model": PoissonCountModel(constant_weight=1, provenance="synthetic").to_dict()})
    zero = scale_composite_data(counted, 0, _apply_dataset_scale)
    np.testing.assert_array_equal(zero.signal, [0, 0])
    np.testing.assert_array_equal(zero.errors, [0, 0])
    assert zero.source_dependencies is not None
    assert selected_event_statistics(zero) is not None
    assert "poisson_count_model" not in zero.metadata
    doubled = scale_measurement_data(counted, 2)
    assert doubled.metadata["poisson_count_model"]["constant_weight"] == 2
    varying = scale_measurement_data(counted, [2, 3])
    assert "poisson_count_model" not in varying.metadata
    precision = scale_measurement_data(continuous_histogram("inverse_variance_mean"), 0)
    np.testing.assert_array_equal(precision.signal, [0, 0])
    assert precision.metadata["measurement_statistics"]["deterministic_zero_scale"]
    with pytest.raises(SourceReplayRequired, match="exact-constraint"):
        scale_measurement_data(continuous_histogram("inverse_variance_mean"), [0, 2])


def test_project_dataset_copy_preserves_immutable_point_statistics():
    from nfit.measurement_fit_data import prepare_histogram_fit_points
    from nfit.pipeline import DataGroup
    from nfit.project_gui import copy_dataset_to_group
    original = prepare_histogram_fit_points(uncertain_count_data())
    group = DataGroup("copy target")
    copied = copy_dataset_to_group(DatasetEntry("source", original), group).data
    assert copied.measurement_payload is not None
    assert copied.source_dependencies is original.source_dependencies
    assert copied.metadata is not original.metadata
    np.testing.assert_array_equal(copied.measurement_payload["event_signal_numerator"], original.measurement_payload["event_signal_numerator"])
    assert not copied.measurement_payload["event_signal_numerator"].flags.writeable
    assert not copied.source_dependencies.coefficients.flags.writeable
