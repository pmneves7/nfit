"""Additive sufficient statistics for explicitly declared continuous means."""
from __future__ import annotations

import numpy as np

from .measurement_contracts import MeasurementContract

MEASUREMENT_STATISTICS_KEY = "measurement_statistics"
MEASUREMENT_VALUE_NUMERATOR = "measurement_value_numerator"
MEASUREMENT_VARIANCE_NUMERATOR = "measurement_variance_numerator"
MEASUREMENT_WEIGHT = "measurement_weight"
MEASUREMENT_OBSERVATION_COUNT = "measurement_observation_count"
MEASUREMENT_STATISTICS_CHANNELS = (MEASUREMENT_VALUE_NUMERATOR, MEASUREMENT_VARIANCE_NUMERATOR,
                                   MEASUREMENT_WEIGHT, MEASUREMENT_OBSERVATION_COUNT)


def validate_measurement_selection(contract, values, errors, *, mask=None, weight=None):
    """Enforce an explicit reject-missing policy on selected observations."""
    if contract is None or contract.missing != "reject":
        return
    valid = np.isfinite(values) & np.isfinite(errors) & (np.asarray(errors) >= 0)
    if contract.estimator == "inverse_variance_mean":
        valid &= np.asarray(errors) > 0
    if mask is not None:
        valid &= ~np.asarray(mask, bool)
    if weight is not None:
        valid &= np.isfinite(weight) & (np.asarray(weight) > 0)
    if not np.all(valid):
        raise ValueError("The contract rejects missing, masked or unexposed observations")


def initialize_measurement_statistics(values, errors, contract: MeasurementContract, *, mask=None):
    """Return four additive arrays and their marker without inferring a target.

    Precision weights share a finite reference sigma; they are proportional to
    inverse variance, not an absolute exposure. This scale must be reconciled
    before combining independently initialized sources.
    """
    if contract.kind != "continuous" or contract.dependence != "independent":
        raise ValueError("Continuous independent means require their measurement payload or source replay")
    values, errors = np.broadcast_arrays(np.asarray(values, float), np.asarray(errors, float))
    valid = np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    if mask is not None:
        valid &= ~np.asarray(mask, bool)
    marker = {"version": 1, "estimator": contract.estimator, "precision_scale": None}
    if contract.estimator == "inverse_variance_mean":
        valid &= errors > 0
        scale = float(np.min(errors[valid])) if np.any(valid) else 1.0
        marker["precision_scale"] = scale
        weight = np.zeros(values.shape)
        np.divide(scale, errors, out=weight, where=valid)
        weight **= 2
    else:
        weight = valid.astype(float)
    if contract.missing == "reject" and not np.all(valid):
        raise ValueError("The contract rejects missing or masked observations")
    numerator = np.where(valid, weight * np.where(valid, values, 0), 0)
    weighted_error = weight * np.where(valid, errors, 0)
    return (numerator, weighted_error ** 2, weight, valid.astype(float)), marker


def normalized_measurement_statistics(numerator, variance, weight, observation_count=None):
    """Finalize S/W and Q/W², retaining measured zero-variance estimates."""
    numerator, variance, weight = np.broadcast_arrays(numerator, variance, weight)
    valid = np.isfinite(numerator) & np.isfinite(variance) & (variance >= 0) & np.isfinite(weight) & (weight > 0)
    signal = np.full(numerator.shape, np.nan)
    result_variance = np.full(numerator.shape, np.nan)
    np.divide(numerator, weight, out=signal, where=valid)
    np.divide(variance, weight, out=result_variance, where=valid)
    np.divide(result_variance, weight, out=result_variance, where=valid)
    return signal, result_variance


def pool_measurement_statistics(numerator, variance, weight, observation_count, axes, *, mask=None):
    """Pool compatible additive payloads across the specified array axes."""
    arrays = np.broadcast_arrays(numerator, variance, weight, observation_count)
    valid = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & (arrays[1] >= 0) & np.isfinite(arrays[2]) & (arrays[2] > 0)
    if mask is not None:
        valid &= ~np.asarray(mask, bool)
    return tuple(np.sum(np.where(valid, array, 0), axis=tuple(axes)) for array in arrays)


def measurement_statistics_channels(numerator, variance, weight, observation_count):
    """Build immutable generic measurement channels."""
    from .mdhisto import MDHistoChannel
    return {name: MDHistoChannel(array, label=name.replace("_", " "))
            for name, array in zip(MEASUREMENT_STATISTICS_CHANNELS,
                                   (numerator, variance, weight, observation_count), strict=True)}


def selected_measurement_statistics(data, selection=(...,), *, validate=True):
    """Read a selected payload and verify it reproduces the primary estimate."""
    marker = data.metadata.get(MEASUREMENT_STATISTICS_KEY)
    if not isinstance(marker, dict) or marker.get("version") != 1:
        return None
    if any(name not in data.auxiliary_channels or data.auxiliary_channels[name].values.shape != data.shape
           for name in MEASUREMENT_STATISTICS_CHANNELS):
        return None
    declaration = data.metadata.get("measurement_contract")
    if declaration is not None and marker.get("estimator") != MeasurementContract.from_dict(declaration).estimator:
        raise ValueError("Stored sufficient statistics belong to a different estimator; replay sources")
    stats = tuple(np.asarray(data.auxiliary_channels[name].values[selection], float)
                  for name in MEASUREMENT_STATISTICS_CHANNELS)
    if validate:
        signal, variance = normalized_measurement_statistics(*stats)
        if not (np.allclose(signal, data.signal[selection], rtol=1e-12, atol=0, equal_nan=True)
                and np.allclose(variance, np.square(data.errors[selection]), rtol=1e-12, atol=0, equal_nan=True)):
            return None
    return stats


def rescale_measurement_statistics(stats, marker, target_precision_scale):
    """Reconcile scaled precisions before pooling separately prepared sources."""
    if marker.get("estimator") != "inverse_variance_mean":
        return stats, dict(marker)
    current = float(marker["precision_scale"])
    target = float(target_precision_scale)
    if not np.isfinite(target) or target <= 0 or target > current:
        raise ValueError("Choose a positive common precision scale no larger than any source scale")
    ratio = (target / current) ** 2
    numerator, variance, weight, count = stats
    updated = {**marker, "precision_scale": target}
    return (numerator * ratio, (variance * ratio) * ratio, weight * ratio, count), updated


def reduce_source_dependencies(dependencies, selection, weights, axes, output_weight):
    """Project a selected slice's weighted mean with bounded selected-grid indices."""
    from .measurement_dependencies import project_source_dependencies
    vectors = []
    kept_dimensions = []
    for dim, (size, item) in enumerate(zip(dependencies.shape, selection, strict=True)):
        if isinstance(item, slice):
            vectors.append(np.arange(size)[item])
            kept_dimensions.append(dim)
        else:
            vectors.append(np.array([int(item)]))
    selected_shape = np.asarray(weights).shape
    coordinates = np.meshgrid(*vectors, indexing="ij", sparse=True)
    input_indices = np.ravel_multi_index(tuple(np.broadcast_arrays(*coordinates)), dependencies.shape)
    input_indices = input_indices.reshape(selected_shape)
    output_shape = np.asarray(output_weight).shape
    surviving = [dim for dim in range(len(selected_shape)) if dim not in axes]
    selected_grids = np.indices(selected_shape, sparse=True)
    outputs = np.ravel_multi_index(tuple(np.broadcast_arrays(*(selected_grids[dim] for dim in surviving))), output_shape)
    outputs = np.broadcast_to(outputs, selected_shape)
    denominators = np.asarray(output_weight).reshape(tuple(selected_shape[dim] if dim in surviving else 1 for dim in range(len(selected_shape))))
    coefficients = np.zeros(selected_shape)
    np.divide(weights, denominators, out=coefficients, where=(np.asarray(weights) > 0) & (denominators > 0))
    valid = coefficients.reshape(-1) != 0
    return project_source_dependencies(dependencies, input_indices.reshape(-1)[valid], outputs.reshape(-1)[valid],
                                       coefficients.reshape(-1)[valid], output_shape)


def coarsen_source_dependencies(dependencies, weights, axis, factor, output_weight):
    """Project complete cells into display blocks using their mean coefficients."""
    from .measurement_dependencies import project_source_dependencies
    shape = list(dependencies.shape)
    shape[axis] = (shape[axis] + factor - 1) // factor
    grids = np.indices(dependencies.shape, sparse=True)
    coordinates = [grid // factor if dim == axis else grid for dim, grid in enumerate(grids)]
    outputs = np.ravel_multi_index(tuple(np.broadcast_arrays(*coordinates)), tuple(shape))
    weights = np.asarray(weights)
    denominators = np.asarray(output_weight).reshape(-1)[outputs]
    coefficients = np.zeros(weights.shape)
    np.divide(weights, denominators, out=coefficients, where=(weights != 0) & (denominators > 0))
    valid = coefficients.reshape(-1) != 0
    return project_source_dependencies(dependencies, np.flatnonzero(valid), outputs.reshape(-1)[valid],
                                       coefficients.reshape(-1)[valid], tuple(shape))
