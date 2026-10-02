"""Analytic one-bin contracts, without instruments or external engines."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nfit import MeasurementContract, SourceTerm, estimate_measurement_bin


def contract(kind, estimator, **kwargs):
    return MeasurementContract(kind=kind, estimator=estimator, quantity="response", value_units="U", **kwargs)


def counting(**kwargs):
    return contract("counting", "exposure_pool", exposure_units="s", **kwargs)


def sampled(estimator="coordinate_mean", **kwargs):
    return contract("sampled_function", estimator, coordinate_units="K", interpolation="linear", **kwargs)


@pytest.mark.parametrize("model", [
    counting(),
    counting(normalizer="uncertain", dependence="shared_sources"),
    contract("continuous", "inverse_variance_mean"),
    sampled(),
    sampled("coordinate_integral", partial_support="covered_only"),
    contract("linear_reconstruction", "linear_sum", dependence="shared_sources"),
])
def test_complete_contract_json_round_trip(model):
    assert MeasurementContract.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    with pytest.raises(FrozenInstanceError):
        model.estimator = "other"


@pytest.mark.parametrize("changes", [
    {"kind": "instrument_name"}, {"estimator": "uniform_mean"},
    {"exposure_units": None}, {"value_units": ""}, {"dependence": "unknown"},
    {"normalizer": "ignored"}, {"interpolation": "linear"},
    {"missing": "zero_fill"}, {"coordinate_units": "K"},
])
def test_invalid_count_contract_rejected(changes):
    fields = counting().to_dict()
    fields.update(changes)
    with pytest.raises(ValueError):
        MeasurementContract.from_dict(fields)


def test_reject_unknown_schema_and_fields():
    for changes in ({"version": 2}, {"version": True}, {"hidden_recipe": 1}):
        with pytest.raises(ValueError):
            MeasurementContract.from_dict(counting().to_dict() | changes)


def test_counts_pool_before_division_and_keep_covered_zeros():
    result = estimate_measurement_bin(counting(), [4, 0, 999], [4, 0, 999], exposure=[1, 9, 0])
    assert result.value == 0.4
    assert result.variance == 0.04
    assert (result.numerator, result.numerator_variance, result.exposure) == (4, 4, 10)
    assert (result.included, result.excluded) == (2, 1)
    assert result.coverage_fraction is None  # Exposure does not infer geometric coverage.
    assert result.uncertainty == "conditional_on_exposure"
    empty = estimate_measurement_bin(counting(), [0], [0], exposure=[10])
    assert empty.measured and empty.standard_error == 0


def test_count_heterogeneity_is_exposure_average_not_uniform_mean():
    result = estimate_measurement_bin(counting(), [2, 90], [2, 90], exposure=[1, 9])
    assert result.value == 9.2  # Exposure average of intensities 2 and 10, not 6.
    assert result.variance == pytest.approx(0.92)


def test_all_unexposed_is_missing_and_masks_remove_exposure():
    missing = estimate_measurement_bin(counting(), [0], [0], exposure=[0])
    assert not missing.measured and np.isnan(missing.value) and missing.support == 0
    masked = estimate_measurement_bin(counting(), [4, 0], [4, 0], exposure=[1, 9], mask=[False, True])
    assert masked.value == 4 and masked.exposure == 1


def test_uncertain_independent_exposure_ratio_delta_variance():
    result = estimate_measurement_bin(counting(normalizer="uncertain"), [10], [10],
        exposure=[2], exposure_variance=[0.04])
    assert result.value == 5
    assert result.variance == pytest.approx(2.75)
    assert result.exposure_variance == 0.04 and result.uncertainty == "delta_method"


def test_shared_normalizer_does_not_average_away_and_joint_terms_cancel():
    source = SourceTerm("common exposure calibration", 1, 0.04)
    result = estimate_measurement_bin(counting(normalizer="uncertain", dependence="shared_sources"),
        [10, 10], [10, 10], exposure=[2, 2], exposure_variance=[0.04, 0.04],
        source_terms=[[SourceTerm("count1", 1, 10)], [SourceTerm("count2", 1, 10)]],
        exposure_source_terms=[[source], [source]])
    assert result.exposure_variance == pytest.approx(0.16)
    assert result.variance == pytest.approx(1.5)  # Counting1.25 + shared calibration0.25.
    cancelled = estimate_measurement_bin(counting(normalizer="uncertain", dependence="shared_sources"),
        [10], [1], exposure=[2], exposure_variance=[0.04],
        source_terms=[[SourceTerm("same scale", 10, 0.01)]],
        exposure_source_terms=[[SourceTerm("same scale", 2, 0.01)]])
    assert cancelled.value == 5 and cancelled.variance == 0


def test_independent_continuous_mean_and_heterogeneity_diagnostic():
    result = estimate_measurement_bin(contract("continuous", "inverse_variance_mean"), [2, 10], [1, 9])
    assert result.value == pytest.approx(2.8)
    assert result.variance == pytest.approx(0.9)
    assert result.chi2 == pytest.approx(6.4) and result.degrees_of_freedom == 1
    uniform = estimate_measurement_bin(contract("continuous", "uniform_mean"), [2, 10], [1, 9])
    assert uniform.value == 6 and uniform.variance == 2.5
    assert uniform.chi2 is None


def test_continuous_mean_does_not_double_weight_exposure():
    with pytest.raises(ValueError, match="Exposure inputs"):
        estimate_measurement_bin(contract("continuous", "inverse_variance_mean"), [2], [1], exposure=[9])
    with pytest.raises(ValueError, match="positive variances"):
        estimate_measurement_bin(contract("continuous", "inverse_variance_mean"), [0], [0])
    with pytest.raises(ValueError, match="requires independent"):
        contract("continuous", "inverse_variance_mean", dependence="shared_sources")


def test_coordinate_mean_integrates_sampling_width_and_preserves_node_covariance():
    mean = estimate_measurement_bin(sampled(), [0, 1, 4], [1, 1, 1], coordinates=[0, 1, 4], interval=(0, 4))
    # Integral coefficients(.5,2,1.5); the shared middle node enters once with weight2.
    assert mean.value == 2 and mean.variance == pytest.approx(6.5 / 16)
    assert mean.coverage_fraction == 1 and mean.support == 4
    integral = estimate_measurement_bin(sampled("coordinate_integral"), [0, 1, 4], [1, 1, 1],
        coordinates=[0, 1, 4], interval=(0, 4))
    assert integral.value == 8 and integral.variance == 6.5 and integral.units == "(U)*(K)"
    assert np.mean([0, 1, 4]) != mean.value


def test_coordinate_clip_and_no_extrapolation_or_gap_bridging():
    clipped = estimate_measurement_bin(sampled(), [0, 10], [1, 1], coordinates=[0, 10], interval=(2, 4))
    assert clipped.value == 3 and clipped.variance == pytest.approx(0.7**2 + 0.3**2)
    rejected = estimate_measurement_bin(sampled(), [0, 1, 4, 5], [1]*4,
        coordinates=[0, 1, 4, 5], interval=(0, 5), mask=[False, False, True, False])
    assert not rejected.measured and rejected.coverage_fraction == 0.2
    partial = estimate_measurement_bin(sampled(partial_support="covered_only"), [0, 1, 4, 5], [1]*4,
        coordinates=[0, 1, 4, 5], interval=(0, 5), mask=[False, False, True, False])
    assert partial.value == 0.5 and partial.coverage_fraction == 0.2
    assert (partial.included, partial.excluded) == (2, 2)
    extrapolation = estimate_measurement_bin(sampled(), [0, 1], [1, 1], coordinates=[0, 1], interval=(-1, 1))
    assert not extrapolation.measured and extrapolation.coverage_fraction == 0.5


def test_signed_shared_background_cancels_covariance_without_poisson_assumption():
    model = contract("linear_reconstruction", "linear_sum", dependence="shared_sources")
    terms = [SourceTerm("background", 1, 9)]
    result = estimate_measurement_bin(model, [-2, -2], [9, 9], coefficients=[1, -1], source_terms=[terms, terms])
    assert result.value == 0 and result.variance == 0 and result.measured
    copies = estimate_measurement_bin(model, [-2, -2], [9, 9], coefficients=[1, 1], source_terms=[terms, terms])
    assert copies.value == -4 and copies.variance == 36  # Not the independent variance18.


def test_partial_shared_sources_reconstruction():
    result = estimate_measurement_bin(contract("linear_reconstruction", "linear_sum", dependence="shared_sources"),
        [3, 5], [5, 8], coefficients=[1, 1], source_terms=[
            [SourceTerm("a", 1, 1), SourceTerm("background", -1, 4)],
            [SourceTerm("b", 1, 4), SourceTerm("background", -1, 4)],
        ])
    assert result.value == 8 and result.variance == 21  #1+4+4*4, not5+8.


def test_source_terms_must_match_diagonal_and_identity():
    model = contract("linear_reconstruction", "linear_sum", dependence="shared_sources")
    with pytest.raises(ValueError, match="requires source terms"):
        estimate_measurement_bin(model, [1], [1], coefficients=[1])
    with pytest.raises(ValueError, match="reproduce"):
        estimate_measurement_bin(model, [1], [2], coefficients=[1], source_terms=[[SourceTerm("a", 1, 1)]])
    with pytest.raises(ValueError, match="Inconsistent variance"):
        estimate_measurement_bin(model, [1, 1], [1, 2], coefficients=[1, 1],
            source_terms=[[SourceTerm("a", 1, 1)], [SourceTerm("a", 1, 2)]])


def test_reconstruction_requires_missing_terms_by_default():
    model = contract("linear_reconstruction", "linear_sum")
    assert model.missing == "reject"
    with pytest.raises(ValueError, match="rejects missing"):
        estimate_measurement_bin(model, [10, np.nan], [1, 1], coefficients=[1, -1])
    partial = estimate_measurement_bin(contract("linear_reconstruction", "linear_sum", missing="omit"),
        [10, np.nan], [1, 1], coefficients=[1, -1])
    assert partial.value == 10 and partial.excluded == 1


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_count_ratio_variance_handles_finite_extreme_scales(scale):
    result = estimate_measurement_bin(counting(), [scale], [scale], exposure=[scale])
    assert result.value == 1
    assert result.variance == pytest.approx(1 / scale, rel=1e-14, abs=0)


def test_missing_policy_and_input_arrays_immutable():
    values, variance, exposure = (np.array([1., np.nan]), np.array([1., 1.]), np.array([1., 2.]))
    for array in (values, variance, exposure):
        array.setflags(write=False)
    result = estimate_measurement_bin(counting(), values, variance, exposure=exposure)
    assert result.value == 1 and result.exposure == 1
    with pytest.raises(ValueError, match="rejects missing"):
        estimate_measurement_bin(counting(missing="reject"), values, variance, exposure=exposure)


def test_repeated_sampling_known_gaussian_errors():
    rng = np.random.default_rng(321)
    errors = np.array([1., 2., 4.])
    draws = rng.normal(7, errors, size=(4000, 3))
    model = contract("continuous", "inverse_variance_mean")
    estimates = [estimate_measurement_bin(model, draw, errors**2) for draw in draws]
    assert np.mean([r.value for r in estimates]) == pytest.approx(7, abs=0.04)
    assert np.var([r.value for r in estimates], ddof=1) == pytest.approx(estimates[0].variance, rel=0.06)
