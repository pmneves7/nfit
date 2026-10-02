"""Contract-aware physical binning of original point measurements."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from .dataset import PointData4D
from .histogram_statistics import (
    EVENT_SIGNAL_NUMERATOR,
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    EVENT_VARIANCE_NUMERATOR,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
    normalized_event_statistics,
)
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from .measurement_aggregation import (
    MEASUREMENT_STATISTICS_CHANNELS,
    MEASUREMENT_STATISTICS_KEY,
    initialize_measurement_statistics,
    measurement_statistics_channels,
    normalized_measurement_statistics,
)
from .measurement_contracts import MeasurementContract
from .measurement_dependencies import (
    SourceDependencies,
    SourceReplayRequired,
    project_source_dependencies,
)
from .measurement_statistics import _interval_weights, estimate_measurement_bin


def bin_measurement_points(data: PointData4D, bin_edges, *, contract=None,
                           vectors=None, axes=None, coordinate_dimension=None, source_namespace=None) -> MDHistoData:
    """Bin declared counts or continuous observations without averaging averages.

    Known-exposure counting requires explicit C,V,N ``measurement_payload``;
    rates alone cannot reconstruct event statistics. Uniform and independent
    inverse-variance means retain additive statistics through staged binning.
    ``source_namespace`` or metadata ``measurement_source_id`` must identify
    the same original observable whenever sampled nodes are replayed. Node IDs
    use physical coordinates, so reordering and subsets preserve their identity.
    Shared sensitivities refer to the input primary signal and are projected
    using the complete output-estimator coefficients.

    A sampled function must have one varying coordinate and constant remaining
    coordinates. Original nodes, including masked nodes, define piecewise linear
    support. Replaying those original points is required for a different grid.
    Uppermost physical bin edges are included. No event counts are invented.
    """
    if not isinstance(data, PointData4D):
        raise TypeError("data must be PointData4D")
    if contract is None:
        declaration = data.metadata.get("measurement_contract")
        if declaration is None:
            raise ValueError("Point binning requires an explicit measurement contract")
        contract = MeasurementContract.from_dict(declaration)
    if not isinstance(contract, MeasurementContract):
        raise TypeError("contract must be a MeasurementContract")
    if contract.kind not in {"counting", "continuous", "sampled_function"} or contract.normalizer != "known":
        raise SourceReplayRequired("This point contract requires its original reconstruction or uncertain-exposure replay")
    edges = tuple(np.asarray(edge, float) for edge in bin_edges)
    if len(edges) != 4 or any(edge.ndim != 1 or len(edge) < 2 or np.any(~np.isfinite(edge))
                              or np.any(np.diff(edge) <= 0) for edge in edges):
        raise ValueError("Point bin edges must be four finite, strictly increasing arrays")
    shape = tuple(len(edge)-1 for edge in edges)
    basis = np.eye(4) if vectors is None else np.asarray(vectors, float)
    if basis.shape != (4, 4) or np.any(~np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("Point coordinate vectors must form a finite invertible 4D basis")
    coordinates = np.column_stack(data.coordinates()) @ np.linalg.inv(basis)
    default_axes = axes is None
    if axes is None:
        axes = tuple(MDHistoAxis(name, edge, "rlu" if dim < 3 else "meV",
                                "momentum" if dim < 3 else "energy", metadata={"vector": basis[dim].tolist()})
                     for dim, (name, edge) in enumerate(zip(("H", "K", "L", "DeltaE"), edges, strict=True)))
    else:
        axes = tuple(axes)
        if len(axes) != 4 or any(not np.array_equal(axis.values, edge) for axis, edge in zip(axes, edges, strict=True)):
            raise ValueError("Output axes must match the requested physical bin edges")
    if contract.kind == "sampled_function":
        if data.metadata.get("measurement_reduction", {}).get("assignment") == "piecewise_linear_original_nodes":
            raise SourceReplayRequired("Sampled histogram means are not original nodes; replay original point sources")
        return _sampled_points(data, contract, coordinates, edges, axes, coordinate_dimension, source_namespace, default_axes)
    if source_namespace is not None:
        raise ValueError("source_namespace applies only to sampled functions")
    if coordinate_dimension is not None:
        raise ValueError("coordinate_dimension applies only to sampled functions")
    dependencies = data.source_dependencies
    if contract.dependence == "shared_sources" and dependencies is None:
        raise SourceReplayRequired("Shared-source point binning requires tracked source sensitivities")
    if np.any((data.sigma < 0) & data.mask):
        raise ValueError("Point uncertainties must be nonnegative")
    if dependencies is not None:
        dependencies.validate_variances(data.sigma**2, mask=~data.mask)
    metadata = copy.deepcopy(data.metadata)
    payload = data.measurement_payload or {}
    if contract.kind == "counting":
        names = (EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR, NORMALIZATION_DENOMINATOR)
        if any(name not in payload for name in names):
            raise SourceReplayRequired("Counting points require explicit numerator, variance and exposure payloads")
        statistics = tuple(np.asarray(payload[name], float) for name in names)
        signal, variance = normalized_event_statistics(*statistics)
        if not (np.allclose(signal[data.mask], data.intensity[data.mask], rtol=1e-12, atol=0, equal_nan=True)
                and np.allclose(variance[data.mask], data.sigma[data.mask]**2, rtol=1e-12, atol=0, equal_nan=True)):
            raise ValueError("Counting payload does not reproduce the point signal and uncertainty")
        if np.any((statistics[0] < 0) & data.mask):
            raise ValueError("Counting numerators must be nonnegative")
        metadata[EVENT_STATISTICS_KEY] = {**EVENT_STATISTICS_METADATA,
            "covariance": "tracked_sources" if dependencies is not None else "diagonal_only"}
        finalizer, channels = normalized_event_statistics, event_statistics_channels
    else:
        stored = metadata.get(MEASUREMENT_STATISTICS_KEY)
        if stored is not None:
            if not isinstance(stored, dict) or stored.get("version") != 1 or stored.get("estimator") != contract.estimator:
                raise SourceReplayRequired("Stored point statistics belong to another estimator; replay original observations")
            if any(name not in payload for name in MEASUREMENT_STATISTICS_CHANNELS):
                raise SourceReplayRequired("Stored continuous point statistics are incomplete")
            statistics = tuple(np.asarray(payload[name], float) for name in MEASUREMENT_STATISTICS_CHANNELS)
            signal, variance = normalized_measurement_statistics(*statistics)
            if not (np.allclose(signal[data.mask], data.intensity[data.mask], rtol=1e-12, atol=0, equal_nan=True)
                    and np.allclose(variance[data.mask], data.sigma[data.mask]**2, rtol=1e-12, atol=0, equal_nan=True)):
                raise ValueError("Continuous payload does not reproduce the point signal and uncertainty")
        else:
            statistics, marker = initialize_measurement_statistics(data.intensity, data.sigma,
                replace(contract, dependence="independent", missing="omit"), mask=~data.mask)
            metadata[MEASUREMENT_STATISTICS_KEY] = marker
        metadata.pop(EVENT_STATISTICS_KEY, None)
        finalizer, channels = normalized_measurement_statistics, measurement_statistics_channels
    indices = [_axis_indices(coordinates[:, dim], edge) for dim, edge in enumerate(edges)]
    selected = np.logical_and.reduce([index >= 0 for index in indices])
    valid = (data.mask & np.isfinite(data.intensity) & np.isfinite(data.sigma) & (data.sigma >= 0)
             & np.isfinite(statistics[2]) & (statistics[2] > 0))
    valid &= np.logical_and.reduce([np.isfinite(array) for array in statistics]) & (statistics[1] >= 0)
    if contract.kind == "continuous" and contract.estimator == "inverse_variance_mean":
        valid &= data.sigma > 0
    if contract.missing == "reject" and np.any(selected & ~valid):
        raise ValueError("The contract rejects missing, masked or unexposed observations")
    included = np.flatnonzero(selected & valid)
    output = np.ravel_multi_index(tuple(index[included] for index in indices), shape)
    size = int(np.prod(shape))
    pooled = tuple(np.bincount(output, weights=array[included], minlength=size).reshape(shape) for array in statistics)
    signal, variance = finalizer(*pooled)
    output_dependencies = None
    if dependencies is not None:
        coefficients = statistics[2][included] / pooled[2].ravel()[output]
        output_dependencies = project_source_dependencies(dependencies, included, output, coefficients, shape)
        variance = np.where(pooled[2] > 0, output_dependencies.variance(), np.nan)
        pooled = (pooled[0], np.where(pooled[2] > 0, (variance * pooled[2]) * pooled[2], 0), *pooled[2:])
    auxiliary = channels(*pooled)
    auxiliary["coverage_fraction"] = MDHistoChannel((pooled[2] > 0).astype(float), label="Measured support", unit="fraction")
    if contract.kind == "continuous":
        events = pooled[3]
        metadata["num_events_semantics"] = "original_measurement_observations"
    else:
        raw_counts = payload.get("num_events")
        if raw_counts is not None and np.any(~np.isfinite(raw_counts) | (np.asarray(raw_counts) < 0)):
            raise ValueError("Retained event counts must be finite and nonnegative")
        events = (np.zeros(shape) if raw_counts is None else
                  np.bincount(output, weights=np.asarray(raw_counts)[included], minlength=size).reshape(shape))
        metadata["num_events_semantics"] = "unavailable" if raw_counts is None else "retained_event_counts"
    metadata.update(measurement_contract=contract.to_dict(), signal_unit=contract.value_units, zero_event_bins_are_measured=True,
        measurement_reduction={"version": 1, "assignment": "nearest_physical_bin"})
    return MDHistoData(axes, signal, np.sqrt(variance), ~np.isfinite(signal) | ~np.isfinite(variance), events,
        metadata=metadata, auxiliary_channels=auxiliary, source_dependencies=output_dependencies,
        coordinate_system=data.metadata.get("coordinate_system"), visual_normalization=data.metadata.get("visual_normalization"))


def _axis_indices(values, edge):
    indices = np.searchsorted(edge, values, side="right")-1
    indices[values == edge[-1]] = len(edge)-2
    indices[~np.isfinite(values) | (values < edge[0]) | (values > edge[-1])] = -1
    return indices


def _sampled_points(data, contract, coordinates, edges, axes, dimension, source_namespace, default_axes):
    varying = np.flatnonzero(np.ptp(coordinates, axis=0) > 0)
    if dimension is None:
        if len(varying) != 1:
            raise SourceReplayRequired("Sampled function binning requires one varying coordinate or explicit coordinate_dimension")
        dimension = int(varying[0])
    if type(dimension) is not int or not 0 <= dimension < 4:
        raise ValueError("coordinate_dimension must be an integer from zero to three")
    if len(data.H) < 2 or np.any(~np.isfinite(coordinates)) or any(dim != dimension for dim in varying):
        raise SourceReplayRequired("Multidimensional sampled interpolation requires original-source replay")
    if default_axes:
        axes = tuple(replace(axis, units=contract.coordinate_units, kind="unknown",
                    name=str(data.metadata.get("coordinate_name", "coordinate")), metadata={**axis.metadata, "variable": "coordinate"})
                    if dim == dimension else axis for dim, axis in enumerate(axes))
    elif axes[dimension].units != contract.coordinate_units:
        raise ValueError("Sampled coordinate axis units must match the declared coordinate units")
    if np.any((data.sigma < 0) & data.mask):
        raise ValueError("Point uncertainties must be nonnegative")
    order = np.argsort(coordinates[:, dimension], kind="stable")
    x = coordinates[order, dimension]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Sampled function coordinates must be distinct; combine repeated nodes explicitly")
    fixed = [None if dim == dimension else int(_axis_indices(coordinates[:1, dim], edge)[0])
             for dim, edge in enumerate(edges)]
    shape = tuple(len(edge)-1 for edge in edges)
    signal = np.full(shape, np.nan)
    variance = np.full(shape, np.nan)
    coverage = np.zeros(shape)
    support = np.zeros(shape)
    observations = np.zeros(shape)
    dependencies = data.source_dependencies
    if dependencies is None:
        if contract.dependence == "shared_sources":
            raise SourceReplayRequired("Shared sampled nodes require tracked source sensitivities")
        namespace = source_namespace if source_namespace is not None else data.metadata.get("measurement_source_id")
        if not isinstance(namespace, str) or not namespace.strip():
            raise SourceReplayRequired("Sampled nodes require an explicit stable source_namespace or measurement_source_id")
        # Coordinates name nodes, so sorting or selecting a subset preserves IDs.
        node_ids = tuple(namespace + ":node:" + ",".join(float(value).hex() for value in row)
                         for row in np.column_stack(data.coordinates()))
        dependencies = SourceDependencies(shape=(data.size,), source_ids=node_ids,
            source_variances=np.where(np.isfinite(data.sigma), data.sigma**2, 0),
            observation_indices=np.arange(data.size), source_indices=np.arange(data.size), coefficients=np.ones(data.size))
    dependencies.validate_variances(data.sigma**2, mask=~data.mask)
    inputs, outputs, coefficients = [], [], []
    if all(index is None or index >= 0 for index in fixed):
        for index, (lower, upper) in enumerate(zip(edges[dimension][:-1], edges[dimension][1:], strict=True)):
            position = tuple(index if dim == dimension else fixed[dim] for dim in range(4))
            result = estimate_measurement_bin(replace(contract, dependence="independent"),
                data.intensity[order], data.sigma[order]**2, mask=~data.mask[order], coordinates=x, interval=(lower, upper))
            signal[position], variance[position] = result.value, result.variance
            support[position], coverage[position], observations[position] = result.support, result.coverage_fraction, result.included
            if result.measured:
                valid = data.mask[order] & np.isfinite(data.intensity[order]) & np.isfinite(data.sigma[order]) & (data.sigma[order] >= 0)
                weights, length, _coverage = _interval_weights(x, valid, (lower, upper))
                if contract.estimator == "coordinate_mean":
                    weights /= length
                contributing = np.flatnonzero(weights)
                inputs.extend(order[contributing])
                outputs.extend([np.ravel_multi_index(position, shape)]*len(contributing))
                coefficients.extend(weights[contributing])
    output_dependencies = project_source_dependencies(dependencies, np.asarray(inputs, np.int64),
        np.asarray(outputs, np.int64), np.asarray(coefficients, float), shape)
    variance = np.where(np.isfinite(signal), output_dependencies.variance(), np.nan)
    metadata = copy.deepcopy(data.metadata)
    metadata.pop(EVENT_STATISTICS_KEY, None)
    metadata.pop(MEASUREMENT_STATISTICS_KEY, None)
    metadata.update(measurement_contract=replace(contract, dependence="shared_sources").to_dict(), signal_unit=contract.output_units,
        zero_event_bins_are_measured=True, num_events_semantics="interpolating_original_nodes",
        measurement_reduction={"version": 1, "assignment": "piecewise_linear_original_nodes", "coordinate_dimension": dimension,
                               "different_grid_requires": "original_point_source_replay"})
    auxiliary = {"coverage_fraction": MDHistoChannel(coverage, label="Coordinate coverage", unit="fraction"),
                 "coordinate_support": MDHistoChannel(support, label="Covered coordinate length", unit=contract.coordinate_units)}
    return MDHistoData(axes, signal, np.sqrt(variance), ~np.isfinite(signal), observations,
        metadata=metadata, auxiliary_channels=auxiliary, source_dependencies=output_dependencies)
