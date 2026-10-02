"""Synthetic checks for the native GetEi reference and optional arithmetic."""

import math

import numpy as np
import pytest

from nfit import raw_dgs_monitors as monitors


def test_reference_negative_roundoff_is_nan_without_python_domain_failure():
    # Empty neighboring cells and almost equal spacings: expanded reference
    # subtraction is negative by roundoff, while the squared coefficient is
    # nonnegative. This must not become an Ei/T0 fallback exception.
    backward = np.float64(1.4550783633161066)
    forward = np.float64(1.455078363316107)
    centre_error = 18.587976302691597
    for trailing in (False, True):
        reference = monitors._monitor_derivative_uncertainty(
            0., centre_error, 0., backward, forward,
            trailing=trailing, variance_policy="mantid",
        )
        stable = monitors._monitor_derivative_uncertainty(
            0., centre_error, 0., backward, forward,
            trailing=trailing, variance_policy="stable",
        )
        assert math.isnan(reference)
        assert np.isfinite(stable) and stable >= 0.
        assert stable == pytest.approx(2.0636799272435e-15, rel=1e-14)


def test_reference_nan_stops_peak_tail_extension(monkeypatch):
    # C++ comparisons against NaN stop this loop after one step. Replacing NaN
    # with zero would extend through two extra empty bins and shift the moment.
    x = np.arange(9, dtype=float)
    y = np.array([0., 10., 20., 100., 40., 10., 0., 0., 0.])
    errors = np.zeros_like(y)
    ordinary = monitors._mantid_getei_peak_region(x, y, errors)

    def sparse_tail_error(*args, trailing, variance_policy):
        return math.nan if trailing else 0.

    monkeypatch.setattr(monitors, "_monitor_derivative_uncertainty", sparse_tail_error)
    reference = monitors._mantid_getei_peak_region(x, y, errors)
    assert reference is not None and ordinary is not None
    assert reference[0][-1] == 5.
    assert ordinary[0][-1] == 7.
    assert np.isfinite(reference[2]) and reference[2] > 0.


def test_distribution_rebin_preserves_reference_reciprocal_rounding():
    # Constant unit count density stays unit density. The frozen binary64
    # standard error follows HistogramData::rebinFrequencies: sqrt(sum(E*E*
    # overlap*old_width)) multiplied by reciprocal new width, not division.
    counts = np.ones(2)
    edges = np.array([0., 1., 2.])
    target = np.array([.13, 1.7])
    signal, error = monitors._rebin_monitor_histogram(
        counts, edges, target, distribution=True,
    )
    assert signal[0] == 1.
    assert float(error[0]).hex() == "0x1.989ed8184f57fp-1"
    assert error[0] == pytest.approx(1 / np.sqrt(1.57))


def test_monitor_rebin_accumulates_input_errors_separately():
    counts = np.array([3., 3., 17.])
    edges = np.arange(4, dtype=float)
    target = np.array([0., .75, 2.25, 3.])
    signal, error = monitors._rebin_monitor_histogram(counts, edges, target)
    np.testing.assert_allclose(signal, [2.25, 8., 12.75])
    np.testing.assert_allclose(error**2, signal)
    assert signal.sum() == counts.sum()


def test_stable_derivative_is_explicit_and_reference_is_default(monkeypatch):
    x = np.arange(-30., 31.)
    y = 1000 * np.exp(-x**2 / 50.)
    errors = np.sqrt(y)
    calls = []
    original = monitors._monitor_derivative_uncertainty

    def capture(*args, **kwargs):
        calls.append(kwargs["variance_policy"])
        return original(*args, **kwargs)

    monkeypatch.setattr(monitors, "_monitor_derivative_uncertainty", capture)
    assert monitors._mantid_getei_peak_region(x, y, errors) is not None
    assert calls and set(calls) == {"mantid"}
    calls.clear()
    assert monitors._mantid_getei_peak_region(x, y, errors, variance_policy="stable") is not None
    assert calls and set(calls) == {"stable"}


@pytest.mark.parametrize("policy", ["unknown", "MANTID", None])
def test_monitor_peak_rejects_unrecognized_policy_before_fitting(policy):
    values = np.arange(3.)
    with pytest.raises(ValueError, match="monitor_variance_policy"):
        monitors._mantid_getei_v2_peak(values, 1., 60., variance_policy=policy)
    with pytest.raises(ValueError, match="monitor_variance_policy"):
        monitors._mantid_getei_peak_region(values, values, values, variance_policy=policy)
