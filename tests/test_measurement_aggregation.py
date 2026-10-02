import numpy as np
import pytest

from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_aggregation import (
    MEASUREMENT_STATISTICS_KEY,
    initialize_measurement_statistics,
    measurement_statistics_channels,
    normalized_measurement_statistics,
    pool_measurement_statistics,
    rescale_measurement_statistics,
    selected_measurement_statistics,
)
from nfit.measurement_contracts import MeasurementContract
from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view, prepare_mdhisto_waterfall


def contract(estimator):
    return MeasurementContract(kind="continuous", estimator=estimator, quantity="susceptibility", value_units="emu/mol")


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_additive_generic_statistics_preserve_direct_staged_means(estimator):
    declaration = contract(estimator)
    values = np.array([1., 3., 8.])
    errors = np.array([1., 2., 3.])
    stats, marker = initialize_measurement_statistics(values, errors, declaration)
    direct = pool_measurement_statistics(*stats, (0,))
    first = pool_measurement_statistics(*(x[:2] for x in stats), (0,))
    staged = pool_measurement_statistics(*(np.array([a, b[-1]]) for a, b in zip(first, stats, strict=True)), (0,))
    np.testing.assert_allclose(direct, staged)
    if estimator == "uniform_mean":
        expected = np.mean(values), np.sum(errors**2) / 9
    else:
        weights = 1 / errors**2
        expected = np.average(values, weights=weights), 1 / weights.sum()
    np.testing.assert_allclose(normalized_measurement_statistics(*direct), expected)
    assert marker["estimator"] == estimator
    assert direct[-1] == 3


def test_precision_weights_remain_finite_in_small_units_and_reconcile_sources():
    declaration = contract("inverse_variance_mean")
    a, am = initialize_measurement_statistics([1.], [1e-155], declaration)
    b, bm = initialize_measurement_statistics([3.], [2e-155], declaration)
    b, bm = rescale_measurement_statistics(b, bm, am["precision_scale"])
    stats = tuple(x + y for x, y in zip(a, b, strict=True))
    value, variance = normalized_measurement_statistics(*stats)
    np.testing.assert_allclose(value, [1.4])
    assert np.isfinite(variance).all() and variance[0] > 0
    assert bm["precision_scale"] == am["precision_scale"]


def generic_histogram(estimator, *, with_statistics=True):
    values = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
    errors = np.array([[[1., 2., 1.], [2., 1., 2.]], [[2., 1., 2.], [1., 2., 1.]]])
    declaration = contract(estimator)
    stats, marker = initialize_measurement_statistics(values, errors, declaration)
    metadata = {"measurement_contract": declaration.to_dict()}
    if with_statistics:
        metadata[MEASUREMENT_STATISTICS_KEY] = marker
    return MDHistoData(tuple(MDHistoAxis(name, np.arange(n+1), "", "unknown")
                            for name, n in zip(("hidden", "y", "x"), values.shape, strict=True)),
                       values, errors, np.zeros(values.shape, bool), np.zeros(values.shape),
                       metadata=metadata, auxiliary_channels=measurement_statistics_channels(*stats) if with_statistics else {})


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
@pytest.mark.parametrize("with_statistics", [False, True])
def test_hidden_axis_slice_and_display_coarsening_preserve_statistics(estimator, with_statistics):
    hist = generic_histogram(estimator, with_statistics=with_statistics)
    viewer = MDHistoSliceViewer(hist, x_dim=2, y_dim=1, integrate=True)
    viewer.selections[0] = (0., 2.)
    view = viewer.slice_arrays()
    coarse = coarsen_mdhisto_view(view, x_step=3, y_step=2)
    stats, _ = initialize_measurement_statistics(hist.signal, hist.errors, contract(estimator))
    expected = normalized_measurement_statistics(*pool_measurement_statistics(*stats, (0, 1, 2)))
    np.testing.assert_allclose(coarse["signal"], [[expected[0]]])
    np.testing.assert_allclose(coarse["errors"]**2, [[expected[1]]])
    assert coarse["measurement_observation_count"].item() == 12
    assert view[MEASUREMENT_STATISTICS_KEY]["estimator"] == estimator
    assert view["measurement_contract"] == hist.metadata["measurement_contract"]


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_waterfall_retains_prepared_mean_payload(estimator):
    hist = generic_histogram(estimator)
    traces = prepare_mdhisto_waterfall(hist, x_dim=2, waterfall_dim=1,
        selections={0: (0., 2.)}, integrate_checks={0: True}, waterfall_step=2.)
    assert len(traces) == 1
    profile = traces[0].measurement
    assert profile is not None
    assert profile.contract.estimator == estimator
    assert selected_measurement_statistics(profile.data) is not None
    np.testing.assert_allclose(profile.data.auxiliary_channels["measurement_observation_count"].values, [4, 4, 4])


def test_count_waterfall_measured_zeros_contribute_exposure():
    c = np.array([[4., 0.], [0., 0.]])
    n = np.array([[1., 1.], [9., 9.]])
    hist = MDHistoData((MDHistoAxis("y", [0, 1, 2], "", "unknown"), MDHistoAxis("x", [0, 1, 2], "", "unknown")),
        c/n, np.sqrt(c)/n, np.zeros(c.shape, bool), c,
        metadata={EVENT_STATISTICS_KEY: EVENT_STATISTICS_METADATA, "zero_event_bins_are_measured": True},
        auxiliary_channels=event_statistics_channels(c, c, n))
    trace, = prepare_mdhisto_waterfall(hist, x_dim=1, waterfall_dim=0, waterfall_step=2.)
    np.testing.assert_allclose(trace.values, [.4, 0.])
    np.testing.assert_allclose(trace.errors, [.2, 0.])
    assert trace.measurement is not None
    np.testing.assert_allclose(trace.measurement.data.auxiliary_channels["normalization_denominator"].values, [10, 10])


def test_shared_uniform_slice_coarsening_and_waterfall_reunite_source():
    from nfit.measurement_dependencies import SourceDependencies
    declaration = MeasurementContract(kind="continuous", estimator="uniform_mean", quantity="response", value_units="1", dependence="shared_sources")
    dependencies = SourceDependencies(shape=(2, 2, 2), source_ids=("measurement",), source_variances=np.array([4.]),
        observation_indices=np.arange(8), source_indices=np.zeros(8, dtype=np.int64), coefficients=np.ones(8))
    hist = MDHistoData(tuple(MDHistoAxis(name, [0, 1, 2], "", "unknown") for name in ("hidden", "y", "x")),
        np.full((2, 2, 2), 3.), np.full((2, 2, 2), 2.), np.zeros((2, 2, 2), bool), np.ones((2, 2, 2)),
        metadata={"measurement_contract": declaration.to_dict()}, source_dependencies=dependencies)
    viewer = MDHistoSliceViewer(hist, x_dim=2, y_dim=1, integrate=True)
    viewer.selections[0] = (0., 2.)
    view = viewer.slice_arrays()
    np.testing.assert_allclose(view["errors"], 2.)
    assert view["source_dependencies"].shape == (2, 2)
    coarse = coarsen_mdhisto_view(view, x_step=2., y_step=2.)
    np.testing.assert_allclose(coarse["errors"], 2.)
    traces = prepare_mdhisto_waterfall(hist, x_dim=2, waterfall_dim=1, selections={0: (0., 2.)}, integrate_checks={0: True}, waterfall_step=2.)
    np.testing.assert_allclose(traces[0].errors, 2.)
    assert traces[0].measurement.data.source_dependencies is not None


def test_uncertain_exposure_slices_and_display_pool_primitive_numerator_and_exposure():
    from nfit.measurement_dependencies import (
        CountingDependencies,
        independent_source_dependencies,
        ratio_source_dependencies,
    )
    c = np.array([[[10.,0.]], [[0.,0.]]])
    n = np.full(c.shape, 2.)
    bundle = CountingDependencies(numerator_dependencies=independent_source_dependencies(c, "counts"),
        exposure_dependencies=independent_source_dependencies(np.full(c.shape,.04), "normalization"))
    primary = ratio_source_dependencies(bundle,c,n)
    declaration = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="rate", value_units="1", exposure_units="s",
                                      normalizer="uncertain", dependence="shared_sources")
    hist = MDHistoData(tuple(MDHistoAxis(name,np.arange(size+1),"","unknown") for name,size in zip(("hidden","y","x"),c.shape,strict=True)),
        c/n, np.sqrt(primary.variance()), np.zeros(c.shape,bool), c,
        metadata={EVENT_STATISTICS_KEY:EVENT_STATISTICS_METADATA,"measurement_contract":declaration.to_dict(),"zero_event_bins_are_measured":True},
        auxiliary_channels=event_statistics_channels(c,c,n),counting_dependencies=bundle)
    viewer = MDHistoSliceViewer(hist,x_dim=2,y_dim=1,integrate=True)
    viewer.selections[0] = (0.,2.)
    view = viewer.slice_arrays()
    assert view["counting_dependencies"].shape == (1,2)
    coarse = coarsen_mdhisto_view(view,x_step=2.)
    expected_variance = 10/8**2 + 10**2*.16/8**4
    np.testing.assert_allclose(coarse["signal"],[[1.25]])
    np.testing.assert_allclose(coarse["errors"]**2,[[expected_variance]])
    np.testing.assert_allclose(coarse["event_variance_numerator"],[[10.]])
    trace, = prepare_mdhisto_waterfall(hist,x_dim=2,waterfall_dim=1,selections={0:(0.,2.)},integrate_checks={0:True},x_step=2.,waterfall_step=1.)
    np.testing.assert_allclose(trace.values,[1.25])
    np.testing.assert_allclose(trace.errors**2,[expected_variance])
    assert trace.measurement.data.counting_dependencies is not None


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
def test_declared_model_projection_uses_same_observation_weights(estimator):
    hist = generic_histogram(estimator)
    prediction = np.arange(hist.signal.size).reshape(hist.shape)*3.
    hist = hist.with_updates(metadata={**hist.metadata,"fit":prediction,"residual":np.ones(hist.shape)})
    viewer = MDHistoSliceViewer(hist,x_dim=2,y_dim=1,integrate=True)
    viewer.selections[0] = (0.,2.)
    view = viewer.slice_arrays()
    coarse = coarsen_mdhisto_view(view,x_step=3.,y_step=2.)
    weights = hist.auxiliary_channels["measurement_weight"].values
    np.testing.assert_allclose(coarse["fit"],[[np.sum(prediction*weights)/np.sum(weights)]])
    np.testing.assert_allclose(coarse["residual"],(coarse["signal"]-coarse["fit"])/coarse["errors"])


def test_derived_view_requires_new_target_before_aggregation():
    hist = generic_histogram("uniform_mean")
    metadata = {key:value for key,value in hist.metadata.items() if key not in {"measurement_contract",MEASUREMENT_STATISTICS_KEY}}
    metadata.update(measurement_target_required=True, measurement_derivation={"operation":"subtract"})
    hist = hist.with_updates(metadata=metadata, auxiliary_channels={})
    viewer = MDHistoSliceViewer(hist,x_dim=2,y_dim=1,integrate=True)
    viewer.selections[0] = (0.,2.)
    with pytest.raises(ValueError, match="statistical target"):
        viewer.slice_arrays()
    viewer.integrate = False
    viewer.selections[0] = 0.
    view = viewer.slice_arrays()
    assert view["measurement_target_required"]
    with pytest.raises(ValueError, match="statistical target"):
        coarsen_mdhisto_view(view,x_step=3.)
    with pytest.raises(ValueError, match="statistical target"):
        prepare_mdhisto_waterfall(hist,x_dim=2,waterfall_dim=1,selections={0:(0.,0.)},waterfall_step=2.)


def test_reject_missing_policy_enforced_for_retained_statistics():
    from dataclasses import replace
    hist = generic_histogram("uniform_mean")
    declaration = replace(contract("uniform_mean"),missing="reject")
    mask = np.array(hist.mask)
    mask[0,0,0] = True
    hist = hist.with_updates(metadata={**hist.metadata,"measurement_contract":declaration.to_dict()},mask=mask)
    viewer = MDHistoSliceViewer(hist,x_dim=2,y_dim=1,integrate=True)
    viewer.selections[0] = (0.,2.)
    with pytest.raises(ValueError, match="rejects missing"):
        viewer.slice_arrays()


def test_waterfall_partial_coverage_uses_geometric_width_and_mask():
    c, n = np.array([[4.,4.],[0.,0.]]), np.ones((2,2))
    hist = MDHistoData((MDHistoAxis("y",[0.,1.,4.],"","unknown"), MDHistoAxis("x",[0.,1.,2.],"","unknown")),
        c/n,np.sqrt(c)/n,np.zeros(c.shape,bool),c,
        metadata={EVENT_STATISTICS_KEY:EVENT_STATISTICS_METADATA,"zero_event_bins_are_measured":True},
        auxiliary_channels=event_statistics_channels(c,c,n))
    from nfit.mdhisto import MDHistoChannel
    channels = dict(hist.auxiliary_channels)
    channels["coverage_fraction"] = MDHistoChannel(np.array([[.2,.2],[.6,.6]]))
    hist = hist.with_updates(auxiliary_channels=channels)
    trace, = prepare_mdhisto_waterfall(hist,x_dim=1,waterfall_dim=0,waterfall_step=4.)
    np.testing.assert_allclose(trace.measurement.data.auxiliary_channels["coverage_fraction"].values,[.5,.5])
    hist = hist.with_updates(mask=np.array([[False,False],[True,True]]))
    trace, = prepare_mdhisto_waterfall(hist,x_dim=1,waterfall_dim=0,waterfall_step=4.,coverage_threshold=.1)
    np.testing.assert_allclose(trace.measurement.data.auxiliary_channels["coverage_fraction"].values,[.05,.05])
    assert np.all(trace.measurement.data.mask) and np.all(np.isnan(trace.values))


def test_sampled_histogram_can_display_without_aggregating_interpolated_cells():
    from nfit.dataset import PointData4D
    from nfit.measurement_point_bins import bin_measurement_points
    declaration = MeasurementContract(kind="sampled_function",estimator="coordinate_mean", quantity="susceptibility",value_units="emu/mol",
        coordinate_units="K",interpolation="linear")
    points = PointData4D([0.,1.,2.],np.zeros(3),np.zeros(3),np.zeros(3),[0.,2.,4.],np.ones(3),
        metadata={"measurement_contract":declaration.to_dict(),"measurement_source_id":"temperature-scan"})
    hist = bin_measurement_points(points,([0.,1.,2.],[-1.,1.],[-1.,1.],[-1.,1.]))
    view = MDHistoSliceViewer(hist,x_dim=0,y_dim=1).slice_arrays()
    np.testing.assert_allclose(view["signal"],[[1.,3.]])
    np.testing.assert_allclose(view["source_dependencies"].variance(),view["errors"]**2)
    from nfit.measurement_fit_data import prepare_histogram_fit_points
    prepared = prepare_histogram_fit_points(hist)
    np.testing.assert_allclose(prepared.H,[.5,1.5])
    assert prepared.metadata["measurement_reduction"]["assignment"] == "piecewise_linear_original_nodes"
    assert not prepared.metadata["fit_coordinate_slots"]["H"]["physical_momentum_energy"]
    with pytest.raises(ValueError,match="source replay"):
        coarsen_mdhisto_view(view,x_step=2.)


def test_declared_smoothing_is_preview_and_cannot_reuse_independent_statistics():
    from nfit.measurement_waterfalls import prepare_waterfall_measurement_profile
    from nfit.plotting_core import smooth_mdhisto_view
    hist = generic_histogram("uniform_mean")
    view = MDHistoSliceViewer(hist,x_dim=2,y_dim=1).slice_arrays()
    smoothed = smooth_mdhisto_view(view,sigma_x=1.)
    assert smoothed["measurement_target_required"]
    assert smoothed["measurement_derivation"]["operation"] == "smoothing_preview"
    assert "measurement_contract" not in smoothed and MEASUREMENT_STATISTICS_KEY not in smoothed
    with pytest.raises(ValueError,match="statistical target"):
        prepare_waterfall_measurement_profile(smoothed,smoothed["signal"],smoothed["errors"],np.ones(2,bool))
    # Plot previews keep historical diagonal smoothing without claiming a
    # retained scientific profile that can be aggregated or fitted again.
    traces = prepare_mdhisto_waterfall(hist,x_dim=2,waterfall_dim=1,waterfall_step=2.,smoothing_sigma_x=1.)
    assert all(trace.measurement is None for trace in traces)
