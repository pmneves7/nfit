"""Retain the deposited exposure when fractional point means are rebinned."""

import numpy as np
import pytest

from nfit.dataset import PointData4D
from nfit.fitting import rebin_point_data
from nfit.histogram_reduction import normalization_denominator
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.measurement_contracts import MeasurementContract
from nfit.project_rebinning import _rebin_mdhisto_data, _rebin_point_data


def _points(coordinates=(0.0, 1.0), *, exposure=True):
    return PointData4D(
        coordinates,
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        [2.0, 10.0],
        [1.0, 3.0],
        normalization_denominator=[2.0, 6.0] if exposure else None,
    )


def _edges(h_edges):
    return [h_edges, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]


def _config(h_edges, *, weighting="uniform", fractional=True):
    return {
        "mean_weighting": weighting,
        "minimum_coverage": 0.0,
        "axes": [
            {
                "mode": "edges",
                "bin_edges": edges,
                "fractional": fractional,
                "lower": edges[0],
                "upper": edges[-1],
            }
            for edges in _edges(h_edges)
        ],
    }


@pytest.mark.parametrize("entry_point", ["project", "scripting"])
def test_fractional_point_mean_retains_deposited_exposure(entry_point):
    source = _points((0.25, 0.75))
    if entry_point == "project":
        result = _rebin_point_data(source, _config([-0.5, 0.5, 1.5]))
        signal = result.signal.ravel()
        errors = result.errors.ravel()
        denominator = normalization_denominator(result).ravel()
        channel = result.auxiliary_channels["normalization_denominator"]
        assert channel.unit == "arbitrary normalization units"
        assert result.metadata["rebin"]["weighted_by_normalization_denominator"]
        assert not result.metadata["normalization_denominator"].flags.writeable
    else:
        result = rebin_point_data(source, bin_edges=_edges([-0.5, 0.5, 1.5]))
        signal, errors = result.intensity, result.sigma
        denominator = result.normalization_denominator

    np.testing.assert_allclose(denominator, [3.0, 5.0])
    np.testing.assert_allclose(signal, [6.0, 9.2])
    np.testing.assert_allclose(errors, np.sqrt([22.5, 182.5]) / [3.0, 5.0])
    assert not denominator.flags.writeable
    np.testing.assert_allclose(source.normalization_denominator, [2.0, 6.0])


@pytest.mark.parametrize("entry_point", ["project", "scripting"])
def test_exposure_weighted_point_mean_survives_staged_rebin(entry_point):
    source = _points()
    if entry_point == "project":
        intermediate = _rebin_point_data(source, _config([-0.5, 0.5, 1.5]))
        staged = _rebin_mdhisto_data(intermediate, _config([-0.5, 1.5]))
        direct = _rebin_point_data(source, _config([-0.5, 1.5]))
        staged_signal, staged_error = staged.signal, staged.errors
        direct_signal, direct_error = direct.signal, direct.errors
        denominator = normalization_denominator(staged)
    else:
        intermediate = rebin_point_data(source, bin_edges=_edges([-0.5, 0.5, 1.5]))
        staged = rebin_point_data(intermediate, bin_edges=_edges([-0.5, 1.5]))
        direct = rebin_point_data(source, bin_edges=_edges([-0.5, 1.5]))
        staged_signal, staged_error = staged.intensity, staged.sigma
        direct_signal, direct_error = direct.intensity, direct.sigma
        denominator = staged.normalization_denominator

    np.testing.assert_allclose(staged_signal, direct_signal)
    np.testing.assert_allclose(staged_error, direct_error)
    np.testing.assert_allclose(staged_signal, 8.0)
    np.testing.assert_allclose(denominator, 8.0)


@pytest.mark.parametrize("entry_point", ["project", "scripting"])
def test_inverse_variance_point_mean_does_not_retain_weights_as_exposure(entry_point):
    source = _points()
    # A source-shaped compatibility field must not survive as output exposure.
    source = source.with_updates(
        metadata={"normalization_denominator": np.ones((1, 1, 1, 1))}
    )
    if entry_point == "project":
        result = _rebin_point_data(
            source, _config([-0.5, 1.5], weighting="inverse_variance")
        )
        assert normalization_denominator(result) is None
        assert "normalization_denominator" not in result.auxiliary_channels
    else:
        result = rebin_point_data(
            source,
            bin_edges=_edges([-0.5, 1.5]),
            mean_weighting="inverse_variance",
        )
        assert result.normalization_denominator is None
    assert "normalization_denominator" not in result.metadata


def test_unnormalized_point_sum_does_not_retain_exposure_for_a_mean():
    result = rebin_point_data(
        _points(), bin_edges=_edges([-0.5, 1.5]), normalize=False
    )
    assert result.normalization_denominator is None


@pytest.mark.parametrize("entry_point", ["project", "scripting"])
def test_point_mean_without_source_exposure_does_not_create_exposure(entry_point):
    source = _points(exposure=False)
    if entry_point == "project":
        result = _rebin_point_data(source, _config([-0.5, 1.5]))
        assert normalization_denominator(result) is None
    else:
        result = rebin_point_data(source, bin_edges=_edges([-0.5, 1.5]))
        assert result.normalization_denominator is None


@pytest.mark.parametrize("legacy_fractional", [False, True])
def test_declared_histogram_rejects_current_per_axis_fractional_assignment(legacy_fractional):
    contract = MeasurementContract(
        kind="continuous", estimator="uniform_mean", quantity="intensity", value_units="U"
    )
    source = MDHistoData(
        axes=tuple(
            MDHistoAxis(name, edges, "", "unknown")
            for name, edges in zip(("H", "K", "L", "E"), _edges([-0.5, 0.5, 1.5]), strict=True)
        ),
        signal=np.array([2.0, 10.0]).reshape(2, 1, 1, 1),
        errors=np.ones((2, 1, 1, 1)),
        mask=np.zeros((2, 1, 1, 1), dtype=bool),
        num_events=np.ones((2, 1, 1, 1)),
        metadata={"measurement_contract": contract.to_dict()},
    )
    config = _config([-0.5, 1.5], fractional=False)
    config["fractional"] = legacy_fractional
    # Explicit per-axis choices take precedence over the legacy global flag.
    allowed = _rebin_mdhisto_data(source, config)
    np.testing.assert_allclose(allowed.signal, 6.0)
    config["axes"][0]["fractional"] = True
    with pytest.raises(ValueError, match="source replay for fractional"):
        _rebin_mdhisto_data(source, config)
