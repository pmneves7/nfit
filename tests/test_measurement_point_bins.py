"""Original-point binning preserves declared sufficient statistics and sources."""
import numpy as np
import pytest

from nfit.dataset import PointData4D
from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
)
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import SourceDependencies, SourceReplayRequired
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.measurement_point_bins import bin_measurement_points
from nfit.measurement_rebinning import coarsen_measurement_histogram


def contract(kind="continuous", estimator="uniform_mean", **kwargs):
    return MeasurementContract(kind=kind, estimator=estimator, quantity="response", value_units="U", **kwargs)


def points(x, y, sigma, model, **kwargs):
    n = len(x)
    return PointData4D(x, np.zeros(n), np.zeros(n), np.zeros(n), y, sigma,
                       metadata={"measurement_contract": model.to_dict()}, **kwargs)


def edges(x):
    return (np.asarray(x), *([np.array([-0.5, 0.5])]*3))


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_direct_staged_and_prepared_points_preserve_observations(estimator):
    data = points([0.1, 0.3, 0.7], [1, 3, 11], [1, 2, 4], contract(estimator=estimator))
    direct = bin_measurement_points(data, edges([0, 1]))
    first = bin_measurement_points(data, edges([0, .5, 1]))
    staged = coarsen_measurement_histogram(first, edges([0, 1]))
    prepared = bin_measurement_points(prepare_histogram_fit_points(first), edges([0, 1]))
    for result in (staged, prepared):
        np.testing.assert_allclose(result.signal, direct.signal)
        np.testing.assert_allclose(result.errors, direct.errors)
        np.testing.assert_array_equal(result.num_events, direct.num_events)
    assert direct.num_events.item() == 3


def test_counting_zeros_exposure_and_no_invented_event_counts():
    model = contract("counting", "exposure_pool", exposure_units="s")
    data = points([.1, .3], [4, 0], [2, 0], model,
        normalization_denominator=[1, 9], measurement_payload={
            EVENT_SIGNAL_NUMERATOR: [4, 0], EVENT_VARIANCE_NUMERATOR: [4, 0], NORMALIZATION_DENOMINATOR: [1, 9]})
    result = bin_measurement_points(data, edges([0, 1]))
    assert result.signal.item() == .4 and result.errors.item() == .2
    assert not result.mask.item() and result.num_events.item() == 0
    assert result.metadata["num_events_semantics"] == "unavailable"
    np.testing.assert_allclose(result.auxiliary_channels[NORMALIZATION_DENOMINATOR].values, 10)
    staged = bin_measurement_points(prepare_histogram_fit_points(result), edges([0, 1]))
    np.testing.assert_allclose(staged.signal, result.signal)
    np.testing.assert_allclose(staged.errors, result.errors)
    with pytest.raises(SourceReplayRequired, match="explicit numerator"):
        bin_measurement_points(data.with_updates(measurement_payload=None), edges([0, 1]))


def test_exact_bin_boundaries_and_masked_missing():
    data = points([0, .5, 1, 1.1], [0, 2, 4, 999], [0, 1, 1, 1], contract())
    result = bin_measurement_points(data, edges([0, .5, 1]))
    np.testing.assert_array_equal(result.signal.ravel(), [0, 3])
    np.testing.assert_array_equal(result.num_events.ravel(), [1, 2])
    assert not np.any(result.mask)
    missing = bin_measurement_points(data.with_updates(mask=[False]*4), edges([0, 1]))
    assert np.isnan(missing.signal.item()) and missing.mask.item()


def test_shared_sources_do_not_average_away_and_survive_staging():
    model = contract(dependence="shared_sources")
    dependencies = SourceDependencies(shape=(2,), source_ids=("shared",), source_variances=np.array([9.]),
        observation_indices=np.array([0, 1]), source_indices=np.array([0, 0]), coefficients=np.array([1., 1.]))
    data = points([.1, .7], [2, 4], [3, 3], model, source_dependencies=dependencies)
    direct = bin_measurement_points(data, edges([0, 1]))
    first = bin_measurement_points(data, edges([0, .5, 1]))
    staged = coarsen_measurement_histogram(first, edges([0, 1]))
    assert direct.signal.item() == 3 and direct.errors.item() == 3
    np.testing.assert_allclose(staged.errors, direct.errors)
    assert direct.source_dependencies.source_ids == ("shared",)


def test_sampled_nodes_integrate_coordinate_and_share_adjacent_bin_covariance():
    model = contract("sampled_function", "coordinate_mean", coordinate_units="K", interpolation="linear")
    data = points([0, 1, 4], [0, 1, 4], [1, 1, 1], model)
    data = data.with_updates(metadata={**data.metadata, "measurement_source_id": "sample-1"})
    direct = bin_measurement_points(data, edges([0, 4]), coordinate_dimension=0)
    assert direct.signal.item() == 2
    assert direct.errors.item()**2 == pytest.approx(6.5/16)
    assert direct.axes[0].units == "K"
    repeated = bin_measurement_points(data, edges([0, 4]), coordinate_dimension=0)
    assert repeated.source_dependencies.source_ids == direct.source_dependencies.source_ids
    fine = bin_measurement_points(data, edges([0, 1, 4]), coordinate_dimension=0)
    np.testing.assert_allclose(fine.signal.ravel(), [.5, 2.5])
    # The central original node occurs in both bins; identities are retained.
    payload = fine.source_dependencies
    assert set(payload.source_indices[payload.observation_indices == 0]) & set(payload.source_indices[payload.observation_indices == 1])
    with pytest.raises(ValueError, match="source replay"):
        coarsen_measurement_histogram(fine, edges([0, 4]))
    masked = bin_measurement_points(data.with_updates(mask=[True, False, True]), edges([0, 4]), coordinate_dimension=0)
    assert masked.mask.item() and masked.auxiliary_channels["coverage_fraction"].values.item() == 0


def test_sampled_partial_support_and_multidimensional_rejection():
    model = contract("sampled_function", "coordinate_integral", coordinate_units="K", interpolation="linear", partial_support="covered_only")
    data = points([0, 1, 4, 5], [0, 1, 4, 5], [1]*4, model, mask=[True, True, False, True])
    data = data.with_updates(metadata={**data.metadata, "measurement_source_id": "sample-2"})
    result = bin_measurement_points(data, edges([0, 5]))
    assert result.signal.item() == .5
    assert result.auxiliary_channels["coverage_fraction"].values.item() == .2
    assert result.metadata["measurement_contract"]["estimator"] == "coordinate_integral"
    with pytest.raises(SourceReplayRequired, match="Multidimensional"):
        bin_measurement_points(data.with_updates(K=[0, 0, 1, 1]), edges([0, 5]), coordinate_dimension=0)


def test_project_dispatch_uses_requested_physical_basis_and_grid():
    from nfit.project_rebinning import _rebin_point_data
    data = points([0, 1], [2, 6], [1, 1], contract())
    config = {"fractional": False, "axes": [{"name": "H", "lower": 0, "upper": .5, "num_bins": 1, "step_size": 1,
        "mode": "edges", "bin_edges": [0, .5], "vector": [2, 0, 0, 0]},
        *[{"lower": -.5, "upper": .5, "num_bins": 1, "mode": "edges", "bin_edges": [-.5, .5]} for _ in range(3)]]}
    result = _rebin_point_data(data, config)
    assert result.signal.item() == 4
    assert result.metadata["rebin"]["vectors"][0] == [2, 0, 0, 0]
    assert result.metadata["measurement_contract"]["estimator"] == "uniform_mean"
    config["axes"][0]["fractional"] = True
    with pytest.raises(SourceReplayRequired, match="fractional"):
        _rebin_point_data(data, config)


def test_sampled_namespace_required_and_coordinate_ids_survive_subset():
    model = contract("sampled_function", "coordinate_mean", coordinate_units="K", interpolation="linear")
    data = points([0, 1, 4], [0, 1, 4], [1]*3, model)
    with pytest.raises(SourceReplayRequired, match="stable source_namespace"):
        bin_measurement_points(data, edges([0, 4]))
    full = bin_measurement_points(data, edges([0, 4]), source_namespace="acquisition-A:chi")
    subset = bin_measurement_points(data.subset([False, True, True]), edges([1, 4]), source_namespace="acquisition-A:chi")
    assert set(subset.source_dependencies.source_ids) <= set(full.source_dependencies.source_ids)


def test_sampled_project_route_resolves_units_and_file_namespace():
    from nfit.project_rebinning import _rebin_point_data
    model = contract("sampled_function", "coordinate_mean", coordinate_units="K", interpolation="linear")
    data = points([0, 1, 4], [0, 1, 4], [1]*3, model)
    data = data.with_updates(metadata={**data.metadata, "source_file": "/data/acquisition-A.dat"})
    config = {"fractional": False, "axes": [{"mode": "edges", "bin_edges": [0, 4]},
        *[{"mode": "edges", "bin_edges": [-.5, .5]} for _ in range(3)]]}
    result = _rebin_point_data(data, config)
    assert result.signal.item() == 2
    assert result.axes[0].units == "K" and result.axes[0].kind == "unknown"
    assert all(source.startswith("/data/acquisition-A.dat:node:") for source in result.source_dependencies.source_ids)


def test_count_payload_corruption_and_uncertain_exposure_rejected():
    model = contract("counting", "exposure_pool", exposure_units="s")
    data = points([.5], [1], [1], model, measurement_payload={EVENT_SIGNAL_NUMERATOR: [2],
        EVENT_VARIANCE_NUMERATOR: [1], NORMALIZATION_DENOMINATOR: [1]})
    with pytest.raises(ValueError, match="does not reproduce"):
        bin_measurement_points(data, edges([0, 1]))
    with pytest.raises(SourceReplayRequired, match="uncertain-exposure"):
        bin_measurement_points(data, edges([0, 1]), contract=contract("counting", "exposure_pool", exposure_units="s", normalizer="uncertain"))


def test_shared_counting_pool_preserves_numerator_covariance():
    model = contract("counting", "exposure_pool", exposure_units="s", dependence="shared_sources")
    dependencies = SourceDependencies(shape=(2,), source_ids=("same-count",), source_variances=np.array([4.]),
        observation_indices=np.array([0, 1]), source_indices=np.array([0, 0]), coefficients=np.array([1., 1.]))
    data = points([.1, .7], [4, 4], [2, 2], model, source_dependencies=dependencies,
        measurement_payload={EVENT_SIGNAL_NUMERATOR: [4, 4], EVENT_VARIANCE_NUMERATOR: [4, 4], NORMALIZATION_DENOMINATOR: [1, 1]})
    result = bin_measurement_points(data, edges([0, 1]))
    assert result.signal.item() == 4 and result.errors.item() == 2
    assert result.auxiliary_channels[EVENT_VARIANCE_NUMERATOR].values.item() == 16
    assert result.metadata["event_statistics"]["covariance"] == "tracked_sources"
