"""Analytic and repeated-sampling acceptance for continuous measurements.

Temperature is in K and susceptibility is in emu/mol. These references assume
supplied Gaussian variances and piecewise linear interpolation; they do not
claim interpolation error is included or choose an instrument-specific default.
"""

import numpy as np
import pytest

from nfit import (
    MeasurementContract,
    SourceDependencies,
    bin_measurement_points,
    coarsen_measurement_histogram,
    estimate_measurement_bin,
    estimate_measurement_region,
)
from nfit.dataset import PointData4D


def _contract(kind="continuous", estimator="uniform_mean", **kwargs):
    return MeasurementContract(
        kind=kind, estimator=estimator, quantity="susceptibility",
        value_units="emu/mol", **kwargs,
    )


def _coordinate_contract(**kwargs):
    return _contract(
        "sampled_function", "coordinate_mean", coordinate_units="K",
        interpolation="linear", **kwargs,
    )


def _points(temperature, response, variance, contract, **kwargs):
    zeros = np.zeros(len(temperature))
    return PointData4D(
        temperature, zeros, zeros, zeros, response, np.sqrt(variance),
        metadata={"measurement_contract": contract.to_dict()}, **kwargs,
    )


def _edges(temperature):
    return (np.asarray(temperature), *(np.array([-.5, .5]) for _ in range(3)))


def _trapezoid_weights(temperature):
    """Independent reference for a full-node interval, in coordinate units."""
    widths = np.diff(temperature)
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def _check_gaussian_sampling(draws, expected_mean, expected_variance):
    # Fixed seeds and six standard errors bound sampling noise, rather than
    # treating an exact Monte Carlo realization as a numerical golden file.
    n = len(draws)
    assert abs(np.mean(draws) - expected_mean) < 6 * np.sqrt(expected_variance / n)
    assert abs(np.var(draws, ddof=1) / expected_variance - 1) < 6 * np.sqrt(2 / (n - 1))
    coverage = np.mean(np.abs(draws - expected_mean) <= 1.959963984540054 * np.sqrt(expected_variance))
    assert abs(coverage - .95) < 6 * np.sqrt(.95 * .05 / n)


def test_dense_temperature_sampling_has_coordinate_target_not_sampling_density_target():
    temperature = np.r_[0., np.linspace(1, 2, 101), 8., 14., 20.]
    susceptibility = 2 + .3 * temperature
    variance = (.05 + .005 * temperature)**2
    coordinate = _coordinate_contract()
    data = _points(temperature, susceptibility, variance, coordinate)
    result = bin_measurement_points(data, _edges([0, 20]), source_namespace="chi:scan-A")
    weights = _trapezoid_weights(temperature) / 20
    assert result.signal.item() == pytest.approx(5.)
    assert result.errors.item()**2 == pytest.approx(np.dot(weights**2, variance))
    assert result.axes[0].units == "K"
    assert result.metadata["measurement_contract"]["value_units"] == "emu/mol"
    mean = estimate_measurement_bin(_contract(), susceptibility, variance)
    precision = estimate_measurement_bin(_contract(estimator="inverse_variance_mean"), susceptibility, variance)
    assert mean.value == pytest.approx(np.mean(susceptibility))
    assert precision.value == pytest.approx(np.average(susceptibility, weights=1 / variance))
    assert abs(mean.value - result.signal.item()) > 2
    assert abs(precision.value - result.signal.item()) > 2
    # Supplied errors do not establish that a temperature-varying response is
    # a common value: the model's disagreement is deliberately conspicuous.
    assert precision.chi2 / precision.degrees_of_freedom > 10


def test_repeated_sampling_coordinate_and_common_response_uncertainties():
    rng = np.random.default_rng(606001)
    temperature = np.r_[0., np.linspace(1, 2, 10), 10., 20.]
    variance = (.1 + .01 * temperature)**2
    truth = 2 + .3 * temperature
    weights = _trapezoid_weights(temperature) / 20
    simulated = truth + rng.normal(size=(2048, len(temperature))) * np.sqrt(variance)
    estimates = [estimate_measurement_bin(
        _coordinate_contract(), draw, variance, coordinates=temperature, interval=(0, 20),
    ) for draw in simulated]
    expected_variance = np.dot(weights**2, variance)
    np.testing.assert_allclose([estimate.variance for estimate in estimates], expected_variance)
    _check_gaussian_sampling(np.array([estimate.value for estimate in estimates]), 5., expected_variance)

    # A separate common-response experiment has heterogeneous supplied
    # uncertainties, independent of the dense-temperature interval target.
    common_variance = np.array([.01, .04, .16, .64])
    repeated = 3.7 + rng.normal(size=(2048, 4)) * np.sqrt(common_variance)
    estimates = [estimate_measurement_bin(
        _contract(estimator="inverse_variance_mean"), draw, common_variance,
    ) for draw in repeated]
    expected_variance = 1 / np.sum(1 / common_variance)
    np.testing.assert_allclose([estimate.variance for estimate in estimates], expected_variance)
    _check_gaussian_sampling(np.array([estimate.value for estimate in estimates]), 3.7, expected_variance)


def test_final_temperature_cut_retains_shared_boundary_nodes():
    temperature = np.array([0., 1., 2., 4., 8., 12., 20.])
    variance = np.array([1., 4., 1., 9., 4., 16., 9.]) * .001
    truth = 2 + .3 * temperature
    data = _points(temperature, truth, variance, _coordinate_contract())
    intervals = np.array([0., 2., 8., 20.])
    histogram = bin_measurement_points(data, _edges(intervals), source_namespace="chi:scan-B")
    # Interval means become one full-coordinate mean by weighting interval
    # widths. The shared boundary observations must be added before squaring.
    weights = (np.diff(intervals) / 20).reshape(histogram.signal.shape)
    result = estimate_measurement_region(
        histogram.signal, histogram.errors, weights=weights,
        source_dependencies=histogram.source_dependencies,
        contract=_contract("linear_reconstruction", "linear_sum", dependence="shared_sources"),
    )
    original_weights = _trapezoid_weights(temperature) / 20
    assert result.value == pytest.approx(5.)
    assert result.variance == pytest.approx(np.dot(original_weights**2, variance))
    assert result.variance > np.sum((weights * histogram.errors)**2)
    rng = np.random.default_rng(606002)
    primitive_draws = truth + rng.normal(size=(4096, len(temperature))) * np.sqrt(variance)
    _check_gaussian_sampling(primitive_draws @ original_weights, result.value, result.variance)


@pytest.mark.parametrize("policy", ["reject", "covered_only"])
def test_masked_temperature_node_removes_neighboring_support_without_zero_fill(policy):
    temperature = np.array([0., 1., 2., 5., 8.])
    truth = 2 + .3 * temperature
    # Missing node at 2 K removes 1--2 and 2--5 K, leaving [0,1]+[5,8].
    data = _points(temperature, truth, np.ones(5), _coordinate_contract(partial_support=policy),
                   mask=[True, True, False, True, True])
    result = bin_measurement_points(data, _edges([0, 8]), source_namespace="chi:scan-C")
    assert result.auxiliary_channels["coverage_fraction"].values.item() == .5
    if policy == "reject":
        assert result.mask.item() and np.isnan(result.signal.item())
    else:
        # Exact linear mean over the measured support, not an eight-K interval
        # integral with the missing four K silently filled with zero.
        weights = np.array([.5, .5, 0., 1.5, 1.5]) / 4
        assert not result.mask.item()
        assert result.signal.item() == pytest.approx(weights @ truth)
        assert result.errors.item()**2 == pytest.approx(weights @ weights)


def test_repeated_temperature_coordinates_need_an_explicit_measurement_model():
    data = _points([0, 1, 1, 2], [1, 2, 4, 5], [1, 1, 9, 1], _coordinate_contract())
    with pytest.raises(ValueError, match="combine repeated nodes explicitly"):
        bin_measurement_points(data, _edges([0, 2]), source_namespace="chi:repeat")
    repeated = estimate_measurement_bin(
        _contract(estimator="inverse_variance_mean"), [2, 4], [1, 9],
    )
    combined = _points([0, 1, 2], [1, repeated.value, 5], [1, repeated.variance, 1], _coordinate_contract())
    result = bin_measurement_points(combined, _edges([0, 2]), source_namespace="chi:repeat-combined")
    assert result.signal.item() == pytest.approx(.25 * 1 + .5 * 2.2 + .25 * 5)
    assert result.errors.item()**2 == pytest.approx(.25**2 * 2 + .5**2 * .9)


def test_sparse_coordinate_interpolation_does_not_claim_curvature_uncertainty():
    # Deliberately outside the piecewise-linear truth assumed above: extremely
    # precise nodes cannot establish the response between a 2--10 K gap.
    temperature = np.array([0., 1., 2., 10.])
    susceptibility = temperature**2 / 100
    variance = np.full(4, 1e-12)
    data = _points(temperature, susceptibility, variance, _coordinate_contract())
    result = bin_measurement_points(data, _edges([0, 10]), source_namespace="chi:curved")
    weights = _trapezoid_weights(temperature) / 10
    assert result.signal.item() == pytest.approx(weights @ susceptibility)
    assert result.errors.item()**2 == pytest.approx(weights**2 @ variance)
    analytic_quadratic_mean = 1 / 3
    assert abs(result.signal.item() - analytic_quadratic_mean) > 1000 * result.errors.item()
    assert result.metadata["measurement_contract"]["interpolation"] == "linear"


@pytest.mark.parametrize("shared_background", [False, True])
def test_uneven_observation_counts_and_background_covariance_survive_final_cut(shared_background):
    # Three observations in the first cell and one in the fringe. Combining
    # cell means uniformly would change the target and its uncertainty.
    temperature = np.array([.1, .2, .3, .9])
    values = np.array([3., 4., 5., 8.])
    sample_variance = np.array([1., 4., 9., 16.]) * .01
    background_variance = .25
    n = len(values)
    if shared_background:
        matrix = np.column_stack((np.eye(n), -np.ones(n)))
        primitive_variance = np.r_[sample_variance, background_variance]
    else:
        matrix = np.column_stack((np.eye(n), -np.eye(n)))
        primitive_variance = np.r_[sample_variance, np.full(n, background_variance)]
    rows, cols = np.nonzero(matrix)
    dependencies = SourceDependencies(
        shape=(n,), source_ids=tuple(f"primitive-{index}" for index in range(len(primitive_variance))),
        source_variances=primitive_variance, observation_indices=rows,
        source_indices=cols, coefficients=matrix[rows, cols],
    )
    contract = _contract(dependence="shared_sources")
    data = _points(temperature, values, sample_variance + background_variance,
                   contract, source_dependencies=dependencies)
    fine = bin_measurement_points(data, _edges([0, .5, 1]))
    staged = coarsen_measurement_histogram(fine, _edges([0, 1]))
    direct = bin_measurement_points(data, _edges([0, 1]))
    target_weights = np.full(n, 1 / n)
    source_coefficients = target_weights @ matrix
    expected_variance = np.dot(source_coefficients**2, primitive_variance)
    for result in (direct, staged):
        assert result.signal.item() == pytest.approx(5.)
        assert result.errors.item()**2 == pytest.approx(expected_variance)
        assert result.num_events.item() == n
    cut = estimate_measurement_region(
        fine.signal, fine.errors, weights=np.array([.75, .25]).reshape(fine.signal.shape),
        source_dependencies=fine.source_dependencies,
        contract=_contract("linear_reconstruction", "linear_sum", dependence="shared_sources"),
    )
    assert cut.value == pytest.approx(5.)
    assert cut.variance == pytest.approx(expected_variance)
    if shared_background:
        assert cut.variance == pytest.approx(sample_variance.sum() / n**2 + background_variance)
    else:
        assert cut.variance == pytest.approx((sample_variance.sum() + n * background_variance) / n**2)
    rng = np.random.default_rng(606010 + shared_background)
    primitive_draws = rng.normal(size=(4096, len(primitive_variance))) * np.sqrt(primitive_variance)
    _check_gaussian_sampling(5. + primitive_draws @ source_coefficients, cut.value, cut.variance)
