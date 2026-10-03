from types import SimpleNamespace

import numpy as np
import pytest

from nfit.histogram_statistics import (
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
    poisson_rate_interval,
)
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_dependencies import (
    CountingDependencies,
    SourceDependencies,
    independent_source_dependencies,
)
from nfit.measurement_diagnostics import (
    CONFIDENCE_CHANNELS,
    available_confidence_channels,
    measurement_diagnostics,
    poisson_interval_channels,
)
from nfit.measurement_likelihoods import PoissonCountModel
from nfit.measurement_smoothing import smooth_count_histogram
from nfit.plotting_core import MDHistoSliceViewer, coarsen_mdhisto_view, smooth_mdhisto_view


def counts_histogram(shape=(2, 2, 2), *, dependencies=False):
    c = np.arange(np.prod(shape), dtype=float).reshape(shape)
    n = np.arange(1, c.size + 1, dtype=float).reshape(shape)
    metadata = {
        "event_statistics": dict(EVENT_STATISTICS_METADATA), "zero_event_bins_are_measured": True,
        "measurement_contract": MeasurementContract(kind="counting", estimator="exposure_pool",
            quantity="rate", value_units="1/s", exposure_units="s").to_dict(),
        "poisson_count_model": PoissonCountModel(constant_weight=1., provenance="synthetic independent integer primitives").to_dict(),
    }
    bundle = None
    if dependencies:
        bundle = CountingDependencies(
            numerator_dependencies=independent_source_dependencies(c, "counts"),
            exposure_dependencies=SourceDependencies(shape=shape, source_ids=(), source_variances=np.empty(0),
                observation_indices=np.empty(0, int), source_indices=np.empty(0, int), coefficients=np.empty(0)),
        )
    return MDHistoData(tuple(MDHistoAxis(f"axis{i}", np.arange(size + 1, dtype=float), "1", "edges") for i, size in enumerate(shape)),
        c / n, np.sqrt(c) / n, np.zeros(shape, bool), c,
        metadata=metadata, auxiliary_channels=event_statistics_channels(c, c, n), counting_dependencies=bundle)


def test_intervals_pool_counts_after_hidden_axis_integration_and_coarsening():
    data = counts_histogram()
    viewer = MDHistoSliceViewer(data, x_dim=0, y_dim=1, channel="confidence_upper")
    viewer.selections[2] = (0., 2.)
    viewer.integrate_checks[2] = True
    view = viewer.slice_arrays()
    expected = poisson_rate_interval(view["event_signal_numerator"], view["normalization_denominator"])
    np.testing.assert_allclose(view["confidence_upper"], expected[1])
    coarse = coarsen_mdhisto_view(view, x_step=2., y_step=2.)
    expected = poisson_rate_interval(np.array([[28.]]), np.array([[36.]]))
    np.testing.assert_allclose(coarse["confidence_upper"], expected[1])
    assert coarse["confidence_upper"].shape == (1, 1)
    assert "68.27%" in viewer._channel_label()


def test_covered_zero_interval_is_positive_and_unexposed_masked_cells_stay_missing():
    data = counts_histogram((2, 2))
    view = MDHistoSliceViewer(data).slice_arrays()
    assert view["confidence_lower"][0, 0] == 0
    assert view["confidence_upper"][0, 0] > 0
    view["mask"] = np.array([[True, False], [False, False]])
    assert np.isnan(poisson_interval_channels(view)["confidence_upper"][0, 0])
    view["normalization_denominator"] = np.zeros((2, 2))
    assert np.isnan(poisson_interval_channels(view)["confidence_upper"]).all()


def test_no_intervals_inferred_for_weighted_symmetrized_or_shared_counts():
    data = counts_histogram((2, 2))
    metadata = dict(data.metadata)
    metadata.pop("poisson_count_model")
    assert not available_confidence_channels(data.with_updates(metadata=metadata))
    metadata = {**data.metadata, "symmetry_operations_hkl": [np.eye(3), -np.eye(3)]}
    assert not available_confidence_channels(data.with_updates(metadata=metadata))
    assert not available_confidence_channels(counts_histogram((2, 2), dependencies=True))
    view = MDHistoSliceViewer(data).slice_arrays()
    view["event_variance_numerator"] = view["event_variance_numerator"] * 2
    with pytest.raises(ValueError, match="constant-weight"):
        poisson_interval_channels(view)
    smoothed = smooth_mdhisto_view(MDHistoSliceViewer(data).slice_arrays(), sigma_x=1.)
    assert all(name not in smoothed for name in (*CONFIDENCE_CHANNELS, "poisson_count_model"))


def test_metadata_report_does_not_read_lazy_arrays():
    class ArrayTrap:
        shape = (2, 2)

        def __array__(self, *args, **kwargs):
            raise AssertionError("metadata report loaded an array")

    data = counts_histogram((2, 2))
    lazy = SimpleNamespace(metadata=data.metadata, shape=data.shape,
        source_dependencies=None, counting_dependencies=None,
        auxiliary_channels={name: SimpleNamespace(values=ArrayTrap(), label=channel.label, unit=channel.unit)
                            for name, channel in data.auxiliary_channels.items()})
    report = measurement_diagnostics(lazy)
    assert report["confidence_intervals"]["channels"] == list(CONFIDENCE_CHANNELS)
    assert report["measurement_contract"]["exposure_units"] == "s"


def test_corelli_report_discloses_correlations_and_normalization_without_loading_arrays():
    class LazyTrap:
        def __array__(self, *args, **kwargs):
            raise AssertionError("CORELLI diagnostic loaded lazy numerical data")

    reconstruction = {
        "method": "correlation_chopper_finite_energy",
        "channel_covariance": "Energy channels reuse measured events; stored errors are diagonal only.",
        "normalization_limit": "Charge and duty-cycle normalization does not apply 4D trajectory normalization.",
        "metadata_exposure_uah": LazyTrap(),
    }
    lazy = SimpleNamespace(metadata={"corelli_reconstruction": reconstruction},
        signal=LazyTrap(), errors=LazyTrap(), num_events=LazyTrap(),
        source_dependencies=None, counting_dependencies=None, auxiliary_channels={})
    report = measurement_diagnostics(lazy)
    assert report["corelli_reconstruction"]["channel_covariance"] == reconstruction["channel_covariance"]
    assert report["corelli_reconstruction"]["normalization_limit"] == reconstruction["normalization_limit"]
    assert report["corelli_reconstruction"]["metadata_exposure_uah"] == "[LazyTrap payload omitted]"
    assert isinstance(report["confidence_intervals"], str)


def test_scientific_smoothing_matches_explicit_linear_covariance_and_later_pooling():
    data = counts_histogram((3,), dependencies=True)
    changed = smooth_count_histogram(data, .6, truncate=2.)
    weights = np.array([[1, np.exp(-.5 / .6**2), np.exp(-2 / .6**2)],
                        [np.exp(-.5 / .6**2), 1, np.exp(-.5 / .6**2)],
                        [np.exp(-2 / .6**2), np.exp(-.5 / .6**2), 1]])
    weights /= 1 + 2 * np.exp(-.5 / .6**2) + 2 * np.exp(-2 / .6**2)
    c = weights @ np.arange(3.)
    n = weights @ np.arange(1., 4.)
    operator = weights / n[:, None]
    covariance = operator @ np.diag(np.arange(3.)) @ operator.T
    np.testing.assert_allclose(changed.signal, c/n)
    np.testing.assert_allclose(changed.errors**2, covariance.diagonal())
    assert covariance[0, 1] > 0
    from nfit.measurement_regions import estimate_measurement_region
    estimate = estimate_measurement_region(changed.signal, changed.errors,
        weights=n/n.sum(), source_dependencies=changed.source_dependencies)
    total_weight = weights.sum(axis=0)
    expected_error = np.sqrt(np.sum(total_weight**2 * np.arange(3.))) / n.sum()
    np.testing.assert_allclose(estimate.standard_error, expected_error)
    assert "poisson_count_model" not in changed.metadata
    assert changed.auxiliary_channels["normalization_denominator"].unit == data.auxiliary_channels["normalization_denominator"].unit
    np.testing.assert_array_equal(data.signal, [0, .5, 2/3])


def test_scientific_smoothing_requires_dependencies_and_bounds_work():
    from nfit.measurement_dependencies import SourceReplayRequired
    with pytest.raises(SourceReplayRequired, match="dependencies"):
        smooth_count_histogram(counts_histogram((3,)), 1.)
    with pytest.raises(SourceReplayRequired, match="budget"):
        smooth_count_histogram(counts_histogram((3,), dependencies=True), 1., max_bytes=1)
    data = counts_histogram((3,), dependencies=True).with_updates(mask=[True, False, False])
    kept = smooth_count_histogram(data, .6)
    filled = smooth_count_histogram(data, .6, fill_missing=True)
    assert kept.mask[0] and not filled.mask[0]
    assert np.isfinite(filled.signal[0])
    assert filled.auxiliary_channels["coverage_fraction"].values[0] < 1


def test_qt_diagnostics_tooltips_and_interval_display_remain_unsmoothed():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = QtMDHistoSliceViewer(counts_histogram((2, 2)))
    button = viewer.window.findChild(QtWidgets.QPushButton, "measurement_diagnostics_button")
    assert button is not None and button.toolTip()
    viewer.model.channel = "confidence_upper"
    viewer.smoothing_x = 1.
    view = viewer.model.slice_arrays()
    displayed = viewer._smoothed_slice_view(view)
    np.testing.assert_allclose(displayed["confidence_upper"], view["confidence_upper"])
    assert viewer.channel_combo.findData("confidence_upper") >= 0
    viewer.window.close()
    assert app is not None
