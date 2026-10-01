"""Sparse diagonal corrections for copies of the same measured event.

Source events are independent; transformed copies of one event are perfectly
correlated. The caller supplies the exact bin assignments used to accumulate
the copies. This module adds their within-bin cross terms without constructing
a dense covariance matrix. Covariance between different output bins remains
unrepresented and must not be inferred to vanish.
"""

from __future__ import annotations

import numpy as np


def accumulate_copy_covariance(
    copy_bin_indices,
    source_variances,
    variance_sum,
    *,
    coefficients=None,
) -> dict[str, int | float]:
    """Add same-event, same-bin covariance to an existing variance accumulator.

    ``copy_bin_indices`` has shape ``(copies, source_events)``. A bin index of
    ``-1`` excludes that copy. ``source_variances`` is the untransformed event
    variance, including any shared event-weight or run-scale factor squared.
    Optional ``coefficients`` has the same shape as the indices and represents
    the signed multiplier applied to each copy; absent coefficients are one.

    The accumulator initially contains the independent-copy diagonal terms.
    For each same-bin pair this adds ``2 * a * b * source_variance``. Thus
    two unit copies contribute ``4*v`` rather than ``2*v`` in their shared
    bin, while opposite signed copies cancel. No source arrays are mutated.

    Returned counts distinguish corrected within-bin pairs from cross-bin
    pairs whose covariance cannot be stored in this diagonal accumulator.
    ``cross_terms_added`` records the added numerator variance without scanning
    the output grid, allowing callers to retain the original event-weight RMS.
    Use exact assignments from the signal kernel: independently recomputing
    assignments at floating-point boundaries can apply cross terms wrongly.
    """
    indices = np.asarray(copy_bin_indices)
    variance = np.asarray(source_variances, dtype=float)
    if not isinstance(variance_sum, np.ndarray):
        raise ValueError("variance accumulator must be an array, not copied input storage")
    output = np.asarray(variance_sum)
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("copy bin indices must be a two-dimensional integer array")
    if variance.shape != (indices.shape[1],) or not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("source variances must be finite nonnegative values, one per event")
    if not np.issubdtype(output.dtype, np.floating) or not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError("variance accumulator must be writable contiguous floating-point storage")
    if np.any(indices < -1) or np.any(indices >= output.size):
        raise ValueError("copy bin index is outside the accumulator")
    if coefficients is not None:
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != indices.shape or not np.all(np.isfinite(coefficients)):
            raise ValueError("copy coefficients must be finite and match the index array")
    flat = output.reshape(-1)
    within_pairs = cross_pairs = 0
    cross_terms_added = 0.0
    for a in range(indices.shape[0]):
        for b in range(a + 1, indices.shape[0]):
            valid = (indices[a] >= 0) & (indices[b] >= 0)
            same = valid & (indices[a] == indices[b])
            within_pairs += int(np.count_nonzero(same))
            cross_pairs += int(np.count_nonzero(valid & ~same))
            if not np.any(same):
                continue
            correction = 2 * variance[same]
            if coefficients is not None:
                correction *= coefficients[a, same] * coefficients[b, same]
            cross_terms_added += float(np.sum(correction))
            np.add.at(flat, indices[a, same], correction)
    return {
        "within_bin_pairs_corrected": within_pairs,
        "cross_bin_pairs_unrepresented": cross_pairs,
        "cross_terms_added": cross_terms_added,
    }
