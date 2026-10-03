"""Synthetic continuous-wave acceptance, without certifying an instrument.

The HB2A table is already reduced 2theta/I/dI data. Supplied dI is the
observation uncertainty; intensity is not evidence of a raw Poisson count.
The separate exposure example declares genuine independent synthetic counts.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from nfit import (
    DataGroup,
    MDHistoAxis,
    MeasurementContract,
    PointData4D,
    PointListData,
    PoissonCountModel,
    bin_measurement_points,
    coarsen_measurement_histogram,
    estimate_measurement_region,
    fit_data_bundle,
    import_dataset_paths,
    import_hb2a_powder,
    poisson_interval_channels,
)
from nfit.histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
)
from nfit.measurement_diagnostics import available_confidence_channels
from nfit.project_point_lists import point_list_config, prepared_point_list_data


def _continuous_contract(estimator):
    return MeasurementContract(
        kind="continuous", estimator=estimator,
        quantity="reduced powder intensity", value_units="arbitrary intensity units",
    )


def _output_axes(edges, varying_dimension, name, units):
    return tuple(
        MDHistoAxis(name if dim == varying_dimension else f"constant-{dim}",
                    edge, units if dim == varying_dimension else "1", "unknown")
        for dim, edge in enumerate(edges)
    )


@pytest.mark.parametrize("estimator", ["uniform_mean", "inverse_variance_mean"])
@pytest.mark.parametrize("wavelength", [None, 2.41])
def test_reduced_hb2a_public_import_prepare_bin_and_final_cut(tmp_path, estimator, wavelength):
    # Repeats at 10.1 degrees are distinct observations. The last row is an
    # excluded fringe with missing dI; the 11.8-degree measured zero has a
    # nonzero supplied uncertainty and must remain in the estimate.
    angles = np.array([10., 10.1, 10.1, 10.9, 11.8, 12.2, 13.9])
    intensity = np.array([2., 4., 6., 8., 0., 5., 999.])
    errors = np.array([.5, 1., 2., 1.5, .7, 3., np.nan])
    path = tmp_path / "synthetic-hb2a.dat"
    np.savetxt(path, np.column_stack((angles, intensity, errors)), fmt="%.17g")
    imported = import_hb2a_powder(path)
    assert isinstance(imported, PointListData)
    np.testing.assert_array_equal(imported.channel_values("I"), intensity)
    np.testing.assert_allclose(imported.channel_errors("I"), errors, equal_nan=True)
    assert "measurement_contract" not in imported.metadata
    assert "poisson_count_model" not in imported.metadata

    group = DataGroup("Synthetic CW acceptance")
    entry = import_dataset_paths(group, [path], data_type="powder_elastic")[0]
    if wavelength is not None:
        point_list_config(entry)["wavelength"] = {"value": wavelength, "two_theta": "2theta"}
    prepared = prepared_point_list_data(entry)
    bundle = fit_data_bundle(group, entry, force_rebin=False)
    assert bundle is not None
    points = bundle.points
    np.testing.assert_allclose(points.intensity, intensity)
    np.testing.assert_allclose(points.sigma, errors, equal_nan=True)
    np.testing.assert_array_equal(points.mask, [True] * 6 + [False])
    assert points.mask[4]  # Supplied uncertainty makes the measured zero valid.
    assert points.measurement_payload is None
    assert "poisson_count_model" not in points.metadata
    angular_edges = np.array([9.9, 10.5, 11.5, 13., 14.])
    varying_dimension = 3 if wavelength is None else 0
    if wavelength is None:
        coordinate_edges = angular_edges
        name, units = "2theta", "deg"
        np.testing.assert_allclose(points.E, angles)
        np.testing.assert_array_equal(prepared.coordinate_names, ["2theta"])
    else:
        coordinate_edges = 4 * np.pi * np.sin(np.deg2rad(angular_edges) / 2) / wavelength
        name, units = "q", "1/angstrom"
        expected_q = 4 * np.pi * np.sin(np.deg2rad(angles) / 2) / wavelength
        np.testing.assert_allclose(prepared.column("q"), expected_q)
        np.testing.assert_allclose(points.H, expected_q)
        assert points.metadata["powder_q_modulus_axis"]
        assert points.metadata["coordinate_units"] == units
    edges = tuple(coordinate_edges if dim == varying_dimension else np.array([-.5, .5])
                  for dim in range(4))
    histogram = bin_measurement_points(
        points, edges, contract=_continuous_contract(estimator),
        axes=_output_axes(edges, varying_dimension, name, units),
    )
    weights = np.ones(6) if estimator == "uniform_mean" else 1 / errors[:6]**2
    weights /= weights.sum()
    expected_mean = weights @ intensity[:6]
    expected_variance = weights**2 @ errors[:6]**2
    coarse_edges = tuple(coordinate_edges[[0, -1]] if dim == varying_dimension else edge
                         for dim, edge in enumerate(edges))
    coarse = coarsen_measurement_histogram(histogram, coarse_edges)
    assert coarse.signal.item() == pytest.approx(expected_mean)
    assert coarse.errors.item()**2 == pytest.approx(expected_variance)
    assert coarse.num_events.item() == 6
    assert histogram.mask.ravel()[-1]  # No valid observation in the last cell.
    np.testing.assert_array_equal(histogram.num_events.ravel(), [3, 1, 2, 0])
    assert not available_confidence_channels(histogram)
    assert EVENT_SIGNAL_NUMERATOR not in histogram.auxiliary_channels
    assert histogram.metadata["num_events_semantics"] == "original_measurement_observations"
    # Final means pool the original observations, not the three occupied cell
    # averages. Select all represented cells with their estimator mass.
    cell_mass = np.array([
        np.sum(weights[:3]), weights[3], np.sum(weights[4:]), 0.,
    ]).reshape(histogram.signal.shape)
    cut = estimate_measurement_region(
        histogram.signal, histogram.errors, weights=cell_mass, mask=histogram.mask,
        contract=MeasurementContract(
            kind="linear_reconstruction", estimator="linear_sum",
            quantity="pooled powder response", value_units="arbitrary intensity units",
        ),
    )
    assert cut.value == pytest.approx(expected_mean)
    assert cut.variance == pytest.approx(expected_variance)


def test_declared_cw_counts_pool_covered_zero_and_keep_unexposed_fringe_missing():
    # Generic synthetic CW counts/exposures; this is not inferred from HB2A's
    # already reduced intensities and does not assert a specific raw importer.
    numerator = np.array([4., 0., 99.])
    exposure = np.array([1., 9., 0.])
    values = np.divide(numerator, exposure, out=np.full(3, np.nan), where=exposure > 0)
    errors = np.divide(np.sqrt(numerator), exposure, out=np.full(3, np.nan), where=exposure > 0)
    contract = MeasurementContract(
        kind="counting", estimator="exposure_pool", quantity="count rate",
        value_units="counts/s", exposure_units="s",
    )
    points = PointData4D(
        [10.1, 10.9, 11.9], np.zeros(3), np.zeros(3), np.zeros(3), values, errors,
        mask=[True, True, False], metadata={
            "measurement_contract": contract.to_dict(),
            "poisson_count_model": PoissonCountModel(
                constant_weight=1., provenance="synthetic independent CW integer counts",
            ).to_dict(),
        }, measurement_payload={
            EVENT_SIGNAL_NUMERATOR: numerator,
            EVENT_VARIANCE_NUMERATOR: numerator,
            NORMALIZATION_DENOMINATOR: exposure,
            "num_events": numerator,
        },
    )
    edges = (np.array([10, 10.5, 11, 12]), *(np.array([-.5, .5]) for _ in range(3)))
    histogram = bin_measurement_points(
        points, edges, axes=_output_axes(edges, 0, "2theta", "deg"),
    )
    assert not histogram.mask.ravel()[1]
    assert histogram.signal.ravel()[1] == histogram.errors.ravel()[1] == 0
    assert histogram.mask.ravel()[2]

    def interval_view(data):
        return {
            **data.metadata, "signal": data.signal, "errors": data.errors, "mask": data.mask,
            **{key: data.auxiliary_channels[key].values for key in (
                EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR,
            )},
        }

    bounds = poisson_interval_channels(interval_view(histogram), confidence=.95)
    assert bounds["confidence_lower"].ravel()[1] == 0
    assert bounds["confidence_upper"].ravel()[1] == pytest.approx(-np.log(.025) / 9)
    assert np.isnan(bounds["confidence_upper"].ravel()[2])
    coarse = coarsen_measurement_histogram(
        histogram, (np.array([10, 11, 12]), *edges[1:]),
    )
    assert coarse.signal.ravel()[0] == .4
    assert coarse.errors.ravel()[0] == .2
    assert coarse.auxiliary_channels[NORMALIZATION_DENOMINATOR].values.ravel()[0] == 10
    bounds = poisson_interval_channels(interval_view(coarse), confidence=.95)
    assert bounds["confidence_lower"].ravel()[0] == pytest.approx(chi2.ppf(.025, 8) / 20)
    assert bounds["confidence_upper"].ravel()[0] == pytest.approx(chi2.ppf(.975, 10) / 20)
