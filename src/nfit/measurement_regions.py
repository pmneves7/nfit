"""Linear region estimates with explicit targets and source sensitivities."""
from __future__ import annotations

import numpy as np

from .measurement_contracts import MeasurementContract
from .measurement_statistics import MeasurementEstimate


def estimate_measurement_region(values, errors, *, weights=None, mask=None,
                                source_dependencies=None, contract=None):
    """Estimate a stated linear sum/integral without changing it into a mean.

    ``weights`` are the deterministic coefficients of the linear target (for
    example cell-volume overlaps). Errors and value units refer to the input
    quantity; the contract declares output units including any coordinate units.
    Missing observations follow the declared policy. Shared sources require the
    corresponding sparse payload; no covariance is inferred from event counts.
    """
    values, errors = np.broadcast_arrays(np.asarray(values, float), np.asarray(errors, float))
    if contract is None:
        contract = MeasurementContract(kind="linear_reconstruction", estimator="linear_sum",
            quantity="sum of values", value_units="unspecified units", missing="omit")
    if contract.kind != "linear_reconstruction":
        raise ValueError("Region estimates require a linear reconstruction target")
    coefficients = np.ones(values.shape) if weights is None else np.broadcast_to(np.asarray(weights, float), values.shape)
    if np.any(~np.isfinite(coefficients)):
        raise ValueError("Region coefficients must be finite")
    selected = coefficients != 0
    valid = np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    if mask is not None:
        valid &= ~np.broadcast_to(np.asarray(mask, bool), values.shape)
    if contract.missing == "reject" and np.any(selected & ~valid):
        raise ValueError("The contract rejects missing region observations")
    valid &= selected
    included = int(np.count_nonzero(valid))
    if not included:
        return MeasurementEstimate(np.nan, np.nan, contract.output_units, 0, 0, int(np.count_nonzero(selected)))
    value = float(np.sum(values[valid]*coefficients[valid]))
    if source_dependencies is not None:
        from .measurement_dependencies import project_source_dependencies
        if source_dependencies.shape != values.shape:
            raise ValueError("Region source dependencies must match observations")
        source_dependencies.validate_variances(errors**2, mask=~valid)
        variance = project_source_dependencies(source_dependencies, np.flatnonzero(valid),
            np.zeros(included, dtype=int), coefficients[valid], (1,)).variance().item()
    else:
        if contract.dependence == "shared_sources":
            raise ValueError("Shared region uncertainty requires source dependencies or source replay")
        variance = float(np.sum((errors[valid]*coefficients[valid])**2))
    return MeasurementEstimate(value, variance, contract.output_units, included, included,
        int(np.count_nonzero(selected))-included)
