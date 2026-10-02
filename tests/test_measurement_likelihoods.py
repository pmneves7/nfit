import numpy as np
import pytest

from nfit.dataset import PointData4D
from nfit.fitting import (
    FitDataset,
    FitProblem,
    OptimizationConfig,
    ParameterSpec,
    fit_problem_least_squares,
)
from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import SourceDependencies, SourceReplayRequired
from nfit.measurement_fit_data import prepare_histogram_fit_points
from nfit.measurement_likelihoods import (
    PoissonCountModel,
    fit_measurement_residuals,
    poisson_count_statistics,
    poisson_deviance_residuals,
)


def counting_histogram():
    c = np.array([4., 0.])
    n = np.array([1., 9.])
    contract = MeasurementContract(kind="counting", estimator="exposure_pool", quantity="rate", value_units="1/s", exposure_units="s")
    return MDHistoData((MDHistoAxis("DeltaE", [0., 1., 2.], "meV", "energy"),), c/n, np.sqrt(c)/n,
        np.zeros(2, bool), c, auxiliary_channels=event_statistics_channels(c,c,n),
        metadata={EVENT_STATISTICS_KEY: EVENT_STATISTICS_METADATA, "zero_event_bins_are_measured": True,
                  "measurement_contract": contract.to_dict(),
                  "poisson_count_model": PoissonCountModel(constant_weight=1., provenance="independent detector counts with audited unit weights").to_dict()})


def test_fit_preparation_valid_subset_mask_and_replacement_preserve_payload():
    hist = counting_histogram()
    points = prepare_histogram_fit_points(hist)
    assert points.metadata["measurement_contract"] == hist.metadata["measurement_contract"]
    assert points.metadata["fit_measurement_provenance"]["sufficient_statistics"] == "validated"
    np.testing.assert_array_equal(points.normalization_denominator, [1,9])
    assert not points.measurement_payload["event_signal_numerator"].flags.writeable
    np.testing.assert_array_equal(points.valid(require_positive_sigma=False).measurement_payload["event_signal_numerator"], [4,0])
    np.testing.assert_array_equal(points.subset([False,True]).normalization_denominator, [9])
    changed = points.with_updates(intensity=points.intensity*2, sigma=points.sigma*2)
    assert changed.measurement_payload is None and "poisson_count_model" not in changed.metadata
    np.testing.assert_array_equal(points.with_updates(mask=[True,False]).measurement_payload["event_signal_numerator"], [4,0])
    mutable = points.mutable_copy()
    assert mutable.measurement_payload is None and mutable.source_dependencies is None
    assert "poisson_count_model" not in mutable.metadata


def test_poisson_likelihood_keeps_zeros_and_recovers_exposure_pooled_mle():
    points = prepare_histogram_fit_points(counting_histogram())
    problem = FitProblem([FitDataset("counts", points, likelihood="poisson_deviance")],
        lambda data, params: np.full(data.size, params["rate"]), [ParameterSpec("rate", 1., min=.001, max=10.)])
    result = fit_problem_least_squares(problem, config=OptimizationConfig(kwargs={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}))
    np.testing.assert_allclose(result.params["rate"], .4, rtol=1e-5)
    assert result.dataset_sizes["counts"] == 2
    assert result.dataset_likelihoods == {"counts": "poisson_deviance"}
    np.testing.assert_allclose(result.stderr["rate"], .2, rtol=1e-5)
    assert result.covariance_interpretation == "expected_fisher_asymptotic"
    with pytest.raises(ValueError, match="Fisher"):
        fit_problem_least_squares(problem, config=OptimizationConfig(covariance_mode="residual"))


def test_gaussian_default_keeps_historical_positive_error_selection():
    points = prepare_histogram_fit_points(counting_histogram())
    problem = FitProblem([FitDataset("counts", points)], lambda data, params: np.full(data.size, params["rate"]),
                        [ParameterSpec("rate", 4., vary=False)])
    result = fit_problem_least_squares(problem)
    assert result.dataset_sizes["counts"] == 1
    assert result.dataset_likelihoods == {"counts": "gaussian"}


def test_poisson_admissibility_never_infers_primitives_from_integer_values():
    points = prepare_histogram_fit_points(counting_histogram())
    metadata = dict(points.metadata)
    metadata.pop("poisson_count_model")
    with pytest.raises(ValueError, match="audited"):
        poisson_count_statistics(points.with_updates(metadata=metadata))
    metadata = {**points.metadata, "symmetry_operations_hkl": [np.eye(3).tolist(), (-np.eye(3)).tolist()]}
    with pytest.raises(ValueError, match="Symmetry"):
        poisson_count_statistics(points.with_updates(metadata=metadata))
    payload = dict(points.measurement_payload)
    payload["event_variance_numerator"] = np.array([8.,0.])
    with pytest.raises(ValueError, match="constant-weight"):
        poisson_count_statistics(points.with_updates(measurement_payload=payload))
    metadata = {**points.metadata, "background_subtractions": ["reference"]}
    with pytest.raises(ValueError, match="background"):
        poisson_count_statistics(points.with_updates(metadata=metadata))


def test_poisson_deviance_zero_near_equality_and_wide_dynamic_range():
    np.testing.assert_allclose(poisson_deviance_residuals([0,4], [2,4]), [-2,0])
    result = poisson_deviance_residuals([1.,1e10], [1e100,1e10+1])
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[1], -1e-5, rtol=1e-6)
    assert poisson_deviance_residuals([0.], [0.])[0] == 0


def shared_points(*, singular=False):
    dependencies = SourceDependencies(shape=(2,), source_ids=("shared","a","b"), source_variances=np.array([1.,1.,1.]),
        observation_indices=np.array([0,0,1,1]) if not singular else np.array([0,1]),
        source_indices=np.array([0,1,0,2]) if not singular else np.array([0,0]),
        coefficients=np.ones(4) if not singular else np.ones(2))
    sigma = np.sqrt(dependencies.variances())
    declaration = MeasurementContract(kind="continuous", estimator="uniform_mean", quantity="response", value_units="1", dependence="shared_sources")
    return PointData4D([0,1],[0,0],[0,0],[0,0], [1,3], sigma,
                       source_dependencies=dependencies, metadata={"measurement_contract": declaration.to_dict()})


def test_optional_gls_preserves_shared_uncertainty_and_rejects_singular_reuse():
    points = shared_points()
    with pytest.raises(SourceReplayRequired, match="GLS"):
        fit_measurement_residuals(points, [0,0], "gaussian")
    problem = FitProblem([FitDataset("shared",points,likelihood="gaussian_gls")],
        lambda data, params: np.full(data.size,params["response"]), [ParameterSpec("response",0.)])
    result = fit_problem_least_squares(problem)
    np.testing.assert_allclose(result.params["response"],2.,atol=1e-7)
    np.testing.assert_allclose(result.stderr["response"],np.sqrt(1.5),rtol=1e-6)
    with pytest.raises(SourceReplayRequired, match="Singular"):
        fit_measurement_residuals(shared_points(singular=True), [0,0], "gaussian_gls")
    selected = points.subset([False,True])
    np.testing.assert_allclose(selected.source_dependencies.variances(),[2.])


def test_separately_fitted_datasets_cannot_reuse_primitives():
    points = shared_points()
    problem = FitProblem([FitDataset("a", points.subset([True,False]), likelihood="gaussian_gls"),
                          FitDataset("b", points.subset([False,True]), likelihood="gaussian_gls")],
        lambda data, params: np.full(data.size, params["response"]), [ParameterSpec("response",0.)])
    with pytest.raises(SourceReplayRequired, match="joint GLS"):
        fit_problem_least_squares(problem)


def test_public_fit_rebin_uses_retained_count_statistics():
    from nfit.fitting import rebin_point_data
    points = prepare_histogram_fit_points(counting_histogram())
    edges = [[-1.,1.], [-1.,1.], [-1.,1.], [0.,2.]]
    rebinned = rebin_point_data(points, bin_edges=edges, fractional=False)
    np.testing.assert_allclose(rebinned.intensity, [.4])
    np.testing.assert_allclose(rebinned.sigma, [.2])
    np.testing.assert_allclose(rebinned.normalization_denominator, [10.])
    with pytest.raises(SourceReplayRequired, match="physical cells"):
        rebin_point_data(points, bin_edges=edges)


def test_histogram_poisson_residuals_and_labels_recompute_at_display_grid():
    from nfit.measurement_likelihoods import histogram_view_residuals
    from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view
    hist = counting_histogram()
    view = {"signal":hist.signal[None,:],"errors":hist.errors[None,:],"fit":np.full((1,2),.4),"residual":np.zeros((1,2)),
            "event_signal_numerator":np.array([[4.,0.]]),"event_variance_numerator":np.array([[4.,0.]]),
            "normalization_denominator":np.array([[1.,9.]]),"fit_likelihood":"poisson_deviance","poisson_count_model":hist.metadata["poisson_count_model"],
            "x_edges":np.array([0.,1.,2.]),"x_centers":np.array([.5,1.5]),"y_edges":np.array([0.,1.]),"y_centers":np.array([.5]),
            "mask":np.zeros((1,2),bool),"num_events":np.array([[4.,0.]]),EVENT_STATISTICS_KEY:EVENT_STATISTICS_METADATA}
    residual = histogram_view_residuals(view)
    np.testing.assert_allclose(residual[0,1],-np.sqrt(7.2))
    coarse = coarsen_mdhisto_view(view,x_step=2.)
    np.testing.assert_allclose(coarse["residual"],[[0.]])
    hist = hist.with_updates(metadata={**hist.metadata,"fit":np.full(hist.shape,.4),"residual":np.zeros(hist.shape),"fit_likelihood":"poisson_deviance","residual_label":"Poisson deviance residual"})
    two_dimensional = MDHistoData((MDHistoAxis("singleton", [0.,1.], "", "unknown"), *hist.axes),
        hist.signal[None,:], hist.errors[None,:], hist.mask[None,:], hist.num_events[None,:],
        metadata={**hist.metadata,"fit":np.full((1,2),.4),"residual":np.zeros((1,2))},
        auxiliary_channels=event_statistics_channels(np.array([[4.,0.]]), np.array([[4.,0.]]), np.array([[1.,9.]])))
    viewer = MDHistoSliceViewer(two_dimensional)
    assert viewer.CHANNEL_LABELS["residual"] == "Poisson deviance residual"


def test_compiled_fit_metadata_and_project_recipe_preserve_likelihood(tmp_path):
    from nfit.fit_config import FitDatasetInput, compile_fit_problem
    from nfit.pipeline import DataGroup, DatasetEntry, ModelComponentSpec
    from nfit.project_gui import (
        NfitProject,
        fit_dataset_inputs,
        import_dataset_paths,
        load_project,
        save_dataset_file,
        save_project,
    )
    hist = counting_histogram()
    component = ModelComponentSpec(name="constant",type="constant_background",parameters={"constant":.4},fit_parameters={"constant":False})
    inputs = [FitDatasetInput(name="counts",data=prepare_histogram_fit_points(hist),metadata={"fit_likelihood":"poisson_deviance"})]
    compiled = compile_fit_problem([component],inputs)
    assert compiled.problem.datasets[0].likelihood == "poisson_deviance"
    dataset = DatasetEntry("counts",hist,kind="mdhisto",parameters={"fit_likelihood":"poisson_deviance"})
    portable_path = tmp_path/"counts.npz"
    save_dataset_file(dataset, portable_path, use_view=False)
    group = DataGroup("g")
    import_dataset_paths(group, [portable_path])
    group.datasets[0].parameters["fit_likelihood"] = "poisson_deviance"
    project = NfitProject([group])
    path = tmp_path/"likelihood.nfit"
    save_project(project,path)
    restored = load_project(path)
    replay_inputs,_ = fit_dataset_inputs(restored.data_groups[0])
    assert replay_inputs[0].metadata["fit_likelihood"] == "poisson_deviance"
    assert replay_inputs[0].data.measurement_payload is not None
