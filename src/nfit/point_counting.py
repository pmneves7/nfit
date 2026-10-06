"""Retain additive counting statistics in the shared point histogram kernel."""

from __future__ import annotations

import numpy as np

from .histogram_statistics import EVENT_STATISTICS_METADATA, event_statistics_channels
from .measurement_contracts import MeasurementContract
from .rebin import rebin_nd, rebin_nd_symmetry


def is_counting_measurement(metadata):
    """An explicit counting contract, rather than an instrument-name heuristic."""
    declaration = metadata.get("measurement_contract")
    if declaration is None:
        return False
    contract = MeasurementContract.from_dict(declaration)
    return contract.kind == "counting" and contract.estimator == "exposure_pool" and contract.normalizer == "known"


def counting_point_channels(result, coordinates, raw_counts, symmetry, kwargs, metadata):
    """Preserve numerator, propagated variance, exposure and measured counts.

    The shared uniform kernel has already accumulated alpha*C, alpha²*V and
    alpha*N through physical-edge or fractional assignment. Its denominator
    is exposure, including known response corrections. A lightweight second
    pass retains raw neutron counts separately from observation multiplicity.
    Neighboring-bin covariance is not represented. Symmetry copies follow the
    existing independent-copy diagonal convention.
    """
    if raw_counts is None:
        raise ValueError("Counting point histograms require retained raw counts")
    raw_counts = np.asarray(raw_counts, float)
    if raw_counts.shape != (len(coordinates),) or np.any(~np.isfinite(raw_counts) | (raw_counts < 0)):
        raise ValueError("Raw counts must be finite, nonnegative and match point coordinates")
    exposure = np.asarray(result._normalization, float)
    numerator = np.where(exposure > 0, result.binned_data * exposure, 0.)
    variance = np.where(exposure > 0, np.square(result.binned_data_errs * exposure), 0.)
    count_kwargs = {**kwargs, "data_weights": None, "data_errs": None,
                    "normalize": False, "minimum_samples": 0, "progress_callback": None}
    counts = (rebin_nd(raw_counts, coordinates, **count_kwargs) if symmetry is None else
              rebin_nd_symmetry(raw_counts, coordinates, symmetry, **count_kwargs))
    channels = event_statistics_channels(numerator, variance, exposure)
    fractional = any(result._fractional_axes)
    metadata.update(event_statistics=dict(EVENT_STATISTICS_METADATA), zero_event_bins_are_measured=True,
                    num_events_semantics="retained_neutron_counts",
                    measurement_reduction={"version": 1,
                        "assignment": "fractional" if fractional else "nearest_physical_bin"})
    if fractional or symmetry is not None:
        metadata.pop("poisson_count_model", None)
        metadata["poisson_count_model_invalidation"] = "Fractional assignment or symmetry copies are not independent integer count bins"
    return channels, counts.binned_data
