"""Exact staged estimates and explicit replay requirements for declared grids."""
from dataclasses import replace

import numpy as np
import pytest

from nfit import (
    coarsen_measurement_histogram,
    combine_measurement_histograms,
    histogram_box_profiles,
)
from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_aggregation import selected_measurement_statistics
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import (
    independent_source_dependencies,
    project_source_dependencies,
)
from nfit.plotting_core import MDHistoSliceViewer
from nfit.project_rebinning import _default_rebin_axes, _rebin_mdhisto_data


def data_for(estimator):
    contract = MeasurementContract(kind="continuous", estimator=estimator, quantity="susceptibility", value_units="emu/mol")
    return MDHistoData(axes=(MDHistoAxis("T", np.arange(5.), "K", "unknown"),
                            MDHistoAxis("field", np.array([0., 1.]), "T", "unknown")),
        signal=np.array([1., 4., 10., 20.])[:, None], errors=np.array([1., 2., 3., 4.])[:, None],
        mask=np.zeros((4, 1), bool), num_events=np.zeros((4, 1)),
        metadata={"measurement_contract": contract.to_dict(), "zero_event_bins_are_measured": True})


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_direct_staged_histogram_and_final_profile_agree(estimator):
    data = data_for(estimator)
    direct = coarsen_measurement_histogram(data, ([0, 4], [0, 1]))
    intermediate = coarsen_measurement_histogram(data, ([0, 1, 3, 4], [0, 1]))
    staged = coarsen_measurement_histogram(intermediate, ([0, 4], [0, 1]))
    np.testing.assert_allclose(direct.signal, staged.signal)
    np.testing.assert_allclose(direct.errors, staged.errors)
    np.testing.assert_allclose(selected_measurement_statistics(direct), selected_measurement_statistics(staged))
    view = MDHistoSliceViewer(intermediate, x_dim=1, y_dim=0).slice_arrays()
    profile = histogram_box_profiles(view, view["signal"], view["errors"], (0, 1, 0, 4)).x_measurement
    assert profile.data.signal.item() == pytest.approx(direct.signal.item())
    assert profile.data.errors.item() == pytest.approx(direct.errors.item())
    assert profile.data.num_events.item() == 0


def test_project_rebin_uses_contract_not_legacy_mean_choice():
    data = data_for("inverse_variance_mean")
    axes = _default_rebin_axes(data)
    for axis, edges in zip(axes, ([0, 4], [0, 1]), strict=True):
        axis.update(mode="edges", bin_edges=edges, lower=edges[0], upper=edges[-1])
    result = _rebin_mdhisto_data(data, {"axes": axes, "mean_weighting": "uniform", "fractional": False})
    expected = coarsen_measurement_histogram(data, ([0, 4], [0, 1]))
    np.testing.assert_allclose(result.signal, expected.signal)
    np.testing.assert_allclose(result.errors, expected.errors)
    assert result.metadata["rebin"]["statistical_assignment"] == "aligned_complete_cells"
    with pytest.raises(ValueError, match="replay"):
        _rebin_mdhisto_data(data, {"axes": axes, "fractional": True})


def test_declared_grid_planning_does_not_allocate_a_dummy_histogram(monkeypatch):
    import nfit.project_rebinning as rebinning
    data = data_for("uniform_mean")
    axes = _default_rebin_axes(data)
    for axis, edge in zip(axes, ([0, 4], [0, 1]), strict=True):
        axis.update(mode="edges", bin_edges=edge, lower=edge[0], upper=edge[-1])
    def unexpected_histogram(*args, **kwargs):
        raise AssertionError("Grid planning should not accumulate a second histogram")
    monkeypatch.setattr(rebinning, "rebin_nd", unexpected_histogram)
    result = rebinning._rebin_mdhisto_data(data, {"axes": axes, "fractional": False})
    assert result.signal.item() == 8.75


def test_split_cells_and_missing_observations_require_explicit_treatment():
    data = data_for("uniform_mean")
    with pytest.raises(ValueError, match="replay"):
        coarsen_measurement_histogram(data, ([0, .5, 4], [0, 1]))
    missing = data.with_updates(mask=np.array([False, True, False, False])[:, None])
    contract = replace(MeasurementContract.from_dict(data.metadata["measurement_contract"]), missing="reject")
    with pytest.raises(ValueError, match="rejects"):
        coarsen_measurement_histogram(missing, ([0, 4], [0, 1]), contract=contract)


def test_shared_count_copies_reunite_with_source_variance():
    primitive = independent_source_dependencies(np.ones((1,)), "one-event")
    dependencies = project_source_dependencies(primitive, [0, 0], [0, 1], [1, 1], (2, 1))
    contract = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="intensity", value_units="U",
        exposure_units="N", dependence="shared_sources")
    ones = np.ones((2, 1))
    data = MDHistoData(axes=(MDHistoAxis("y", np.arange(3.), "", "unknown"), MDHistoAxis("x", [0, 1], "", "unknown")),
        signal=ones, errors=ones, mask=np.zeros((2, 1), bool), num_events=ones,
        metadata={"measurement_contract": contract.to_dict(), EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA)},
        auxiliary_channels=event_statistics_channels(ones, ones, ones), source_dependencies=dependencies)
    output = coarsen_measurement_histogram(data, ([0, 2], [0, 1]))
    assert output.signal.item() == output.errors.item() == 1
    view = MDHistoSliceViewer(data, x_dim=1, y_dim=0).slice_arrays()
    profile = histogram_box_profiles(view, view["signal"], view["errors"], (0, 1, 0, 2)).x_measurement
    assert profile.data.errors.item() == 1
    assert profile.data.source_dependencies is not None


def test_uncertain_exposure_of_zero_count_cell_survives_pooling():
    from nfit import CountingDependencies
    from nfit.measurement_dependencies import SourceDependencies
    c, n = np.array([[4.], [0.]]), np.array([[1.], [9.]])
    numerator = independent_source_dependencies(c, "counts")
    # Exposure uncertainty in the empty second cell affects the pooled rate,
    # although its original zero rate has zero first-order ratio variance.
    exposure = SourceDependencies(shape=(2, 1), source_ids=("calibration",), source_variances=np.array([.01]),
        observation_indices=np.array([1]), source_indices=np.array([0]), coefficients=np.array([9.]))
    bundle = CountingDependencies(numerator_dependencies=numerator, exposure_dependencies=exposure)
    contract = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="rate", value_units="U", exposure_units="N",
        normalizer="uncertain", dependence="shared_sources")
    data = MDHistoData(axes=(MDHistoAxis("y", [0, 1, 2], "", "unknown"), MDHistoAxis("x", [0, 1], "", "unknown")),
        signal=c/n, errors=np.sqrt(c)/n, mask=np.zeros((2, 1), bool), num_events=c,
        metadata={"measurement_contract": contract.to_dict(), EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA)},
        auxiliary_channels=event_statistics_channels(c, c, n), counting_dependencies=bundle)
    output = coarsen_measurement_histogram(data, ([0, 2], [0, 1]))
    assert output.signal.item() == .4
    assert output.errors.item()**2 == pytest.approx(.04 + (.4/10)**2*.81)
    assert output.counting_dependencies is not None
    from nfit.histogram_statistics import selected_event_statistics
    assert selected_event_statistics(output) is not None


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_independently_prepared_means_reconcile_weights_in_composites(estimator):
    from nfit.pipeline import DataGroup, DatasetEntry
    from nfit.project_data import _composite_mdhisto_data
    data = data_for(estimator)
    a = coarsen_measurement_histogram(data, ([0, 2], [0, 1]))
    # Shift the second coordinate grid onto the first physical interval. Its
    # independent precision reference differs, so naive W pooling is incorrect.
    b = coarsen_measurement_histogram(data, ([2, 4], [0, 1]))
    b = replace(b, axes=a.axes)
    combined = combine_measurement_histograms([a, b])
    direct = coarsen_measurement_histogram(data, ([0, 4], [0, 1]))
    np.testing.assert_allclose(combined.signal, direct.signal)
    np.testing.assert_allclose(combined.errors, direct.errors)
    datasets = [DatasetEntry("a", data=a), DatasetEntry("b", data=b)]
    axes = _default_rebin_axes(a)
    for axis, edge in zip(axes, ([0, 2], [0, 1]), strict=True):
        axis.update(mode="edges", bin_edges=edge, lower=edge[0], upper=edge[-1])
    project = _composite_mdhisto_data(DataGroup("measurements", datasets=datasets),
        {"axes": axes, "fractional": False}, datasets=datasets)
    np.testing.assert_allclose(project.signal, direct.signal)
    np.testing.assert_allclose(project.errors, direct.errors)
