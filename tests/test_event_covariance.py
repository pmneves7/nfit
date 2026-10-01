"""Copies retain shared-event uncertainty inside and across bin boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from nfit.event_covariance import accumulate_copy_covariance


def test_same_event_copies_in_one_bin_do_not_create_independent_counts():
    indices = np.array([[0], [0], [0]])
    source_variance = np.array([4.0])
    independent = np.array([12.0])
    result = accumulate_copy_covariance(indices, source_variance, independent)
    np.testing.assert_equal(independent, [36.0])
    assert result == {"within_bin_pairs_corrected": 3, "cross_bin_pairs_unrepresented": 0, "cross_terms_added": 24.0}
    np.testing.assert_equal(indices, [[0], [0], [0]])
    np.testing.assert_equal(source_variance, [4.0])


def test_copies_on_opposite_sides_of_a_boundary_keep_cross_bin_limitation_explicit():
    # One event contributes once to each bin. Marginal variances are already
    # correct; their sum cannot represent the variance of integrating both.
    independent = np.array([4.0, 4.0])
    result = accumulate_copy_covariance(np.array([[0], [1]]), [4.0], independent)
    np.testing.assert_equal(independent, [4.0, 4.0])
    assert result["cross_bin_pairs_unrepresented"] == 1
    diagonal_only_integrated_variance = independent.sum()
    original_event_integrated_variance = (1 + 1)**2 * 4.0
    assert diagonal_only_integrated_variance == 8.0
    assert original_event_integrated_variance == 16.0


def test_event_identity_masking_and_run_variance_scale_are_preserved():
    indices = np.array([[0, 0, -1], [0, 1, 0], [-1, 1, 0]])
    source_variance = np.array([4.0, 9.0, 16.0])
    # Independent-copy diagonal: bin0 = 4+9+4+16+16; bin1 =9+9.
    output = np.array([[49.0, 18.0]])
    result = accumulate_copy_covariance(indices, source_variance, output)
    np.testing.assert_equal(output, [[89.0, 36.0]])
    assert result["within_bin_pairs_corrected"] == 3
    assert result["cross_bin_pairs_unrepresented"] == 2
    assert result["cross_terms_added"] == 58.0


def test_signed_or_fractional_copy_coefficients_use_covariance_cross_terms():
    indices = np.array([[0, 1], [0, 1]])
    coefficients = np.array([[1.0, 0.25], [-1.0, 0.75]])
    output = np.array([8.0, 4.0 * (0.25**2 + 0.75**2)])
    accumulate_copy_covariance(indices, [4.0, 4.0], output, coefficients=coefficients)
    np.testing.assert_allclose(output, [0.0, 4.0])


def test_empty_or_single_copy_batches_need_no_correction():
    output = np.zeros(2)
    assert accumulate_copy_covariance(np.empty((3, 0), dtype=int), [], output)["within_bin_pairs_corrected"] == 0
    assert accumulate_copy_covariance(np.array([[1]]), [2.0], output)["cross_bin_pairs_unrepresented"] == 0
    np.testing.assert_equal(output, [0.0, 0.0])


@pytest.mark.parametrize("indices,variance,coefficients", [
    (np.array([0, 0]), [1.0], None),
    (np.array([[0.0]]), [1.0], None),
    (np.array([[2]]), [1.0], None),
    (np.array([[-2]]), [1.0], None),
    (np.array([[0]]), [-1.0], None),
    (np.array([[0]]), [np.nan], None),
    (np.array([[0]]), [1.0], np.array([[np.inf]])),
    (np.array([[0]]), [1.0], np.ones((2, 1))),
])
def test_invalid_copy_descriptions_are_rejected(indices, variance, coefficients):
    with pytest.raises(ValueError):
        accumulate_copy_covariance(indices, variance, np.zeros(2), coefficients=coefficients)


def test_noncontiguous_or_immutable_accumulators_are_rejected():
    with pytest.raises(ValueError):
        accumulate_copy_covariance(np.array([[0], [0]]), [1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        accumulate_copy_covariance(np.array([[0], [0]]), [1.0], np.zeros(4)[::2])
    output = np.zeros(2)
    output.setflags(write=False)
    with pytest.raises(ValueError):
        accumulate_copy_covariance(np.array([[0], [0]]), [1.0], output)
