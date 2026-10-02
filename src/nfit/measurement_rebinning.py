"""Exact aligned histogram coarsening under explicit statistical contracts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .histogram_statistics import (
    EVENT_STATISTICS_KEY,
    event_statistics_channels,
    normalized_event_statistics,
    selected_event_statistics,
)
from .mdhisto import MDHistoChannel, MDHistoData, mdhisto_coverage_fraction
from .measurement_contracts import MeasurementContract
from .measurement_dependencies import SourceReplayRequired


def coarsen_measurement_histogram(data: MDHistoData, bin_edges, *, contract=None,
                                  minimum_coverage=0.0) -> MDHistoData:
    """Combine complete source cells without assuming a subcell distribution.

    Output edges must coincide with source edges. Refinement, rotated grids and
    crossings require replay of the source observations. Covered zero counts
    remain observations; exposure and geometric coverage are distinct. Explicit
    continuous means retain their additive weights rather than averaging means.
    """
    from .measurement_aggregation import (
        MEASUREMENT_STATISTICS_KEY,
        initialize_measurement_statistics,
        measurement_statistics_channels,
        normalized_measurement_statistics,
        selected_measurement_statistics,
    )
    if contract is None:
        declaration = data.metadata.get("measurement_contract")
        if declaration is None:
            raise ValueError("Aligned measurement coarsening requires an explicit contract")
        contract = MeasurementContract.from_dict(declaration)
    if not isinstance(contract, MeasurementContract):
        raise TypeError("contract must be a MeasurementContract")
    bundle = getattr(data, "counting_dependencies", None)
    if (contract.normalizer != "known" and bundle is None) or contract.kind not in {"counting", "continuous"}:
        raise SourceReplayRequired("This contract requires source replay rather than histogram coarsening")
    dependencies = getattr(data, "source_dependencies", None)
    if contract.dependence != "independent" and dependencies is None:
        raise SourceReplayRequired("Shared-source coarsening requires source dependencies or source replay")
    if not np.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between zero and one")
    edges = tuple(np.asarray(edge, dtype=float) for edge in bin_edges)
    if len(edges) != len(data.axes):
        raise ValueError("Output edges must match source dimensions")
    maps, widths = [], []
    for axis, edge in zip(data.axes, edges, strict=True):
        if len(axis.values) != data.shape[len(maps)] + 1:
            raise SourceReplayRequired("Aligned coarsening requires physical source-cell edges, not only centers; replay source observations")
        if edge.ndim != 1 or len(edge) < 2 or np.any(~np.isfinite(edge)) or np.any(np.diff(edge) <= 0):
            raise ValueError("Output edges must be finite and increasing")
        source = axis.values
        matches = np.searchsorted(source, edge)
        tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(edge))
        candidates = np.clip(matches, 0, len(source)-1)
        previous = np.clip(matches-1, 0, len(source)-1)
        candidates = np.where(np.abs(source[previous]-edge) < np.abs(source[candidates]-edge), previous, candidates)
        if np.any(np.abs(source[candidates]-edge) > tolerance):
            raise SourceReplayRequired("Output edges split source cells; replay source observations for this grid")
        mapping = np.searchsorted(edge, axis.centers, side="right")-1
        mapping[(axis.centers < edge[0]) | (axis.centers >= edge[-1])] = -1
        maps.append(mapping)
        widths.append(np.diff(source))
    shape = tuple(len(edge)-1 for edge in edges)
    metadata = dict(data.metadata)
    from .histogram_statistics import has_event_statistics
    from .measurement_aggregation import rescale_measurement_statistics
    counting = contract.kind == "counting"
    if counting:
        if not has_event_statistics(data):
            raise ValueError("Counting coarsening requires validated numerator, variance and exposure")
        finalize, channels, length = normalized_event_statistics, event_statistics_channels, 3
    else:
        finalize, channels, length = normalized_measurement_statistics, measurement_statistics_channels, 4
    stored = not counting and isinstance(metadata.get(MEASUREMENT_STATISTICS_KEY), dict)
    precision_scale = None
    batch_size = 100_000
    if not counting and not stored and contract.estimator == "inverse_variance_mean":
        precision_scale = np.inf
        for start in range(0, data.signal.size, batch_size):
            sigma = data.errors.reshape(-1)[start:start+batch_size]
            okay = np.isfinite(sigma) & (sigma > 0) & ~data.mask.reshape(-1)[start:start+batch_size]
            if np.any(okay):
                precision_scale = min(precision_scale, float(np.min(sigma[okay])))
        if not np.isfinite(precision_scale):
            precision_scale = 1.0
    size = int(np.prod(shape))
    sums = [np.zeros(size) for _ in range(length)]
    support, events = np.zeros(size), np.zeros(size)
    dependency_inputs, dependency_outputs, dependency_weights = [], [], []
    for start in range(0, data.signal.size, batch_size):
        flat = np.arange(start, min(start+batch_size, data.signal.size))
        source_indices = np.unravel_index(flat, data.shape)
        out_indices = tuple(mapping[index] for mapping,index in zip(maps,source_indices,strict=True))
        selected = np.logical_and.reduce([index >= 0 for index in out_indices])
        if counting:
            statistics = selected_event_statistics(data, source_indices)
        elif stored:
            statistics = selected_measurement_statistics(data, source_indices)
        else:
            statistics, marker = initialize_measurement_statistics(
                data.signal[source_indices], data.errors[source_indices],
                replace(contract, missing="omit", dependence="independent"), mask=data.mask[source_indices])
            if precision_scale is not None:
                statistics, marker = rescale_measurement_statistics(statistics, marker, precision_scale)
            metadata[MEASUREMENT_STATISTICS_KEY] = marker
        if statistics is None:
            raise ValueError("Stored measurement statistics do not reproduce the primary estimates; replay sources")
        weight = np.asarray(statistics[2])
        valid = selected & ~data.mask[source_indices] & np.isfinite(data.signal[source_indices]) & np.isfinite(data.errors[source_indices]) & (weight > 0)
        if contract.missing == "reject" and np.any(selected & ~valid):
            raise ValueError("The contract rejects missing, masked or unexposed observations")
        output = np.ravel_multi_index(tuple(index[valid] for index in out_indices), shape)
        for accumulator,array in zip(sums,statistics,strict=True):
            accumulator += np.bincount(output, weights=np.asarray(array)[valid], minlength=size)
        source_volume = np.ones(flat.shape)
        for width,index in zip(widths,source_indices,strict=True):
            source_volume *= width[index]
        support += np.bincount(output, weights=source_volume[valid]*np.asarray(mdhisto_coverage_fraction(data,source_indices))[valid], minlength=size)
        events += np.bincount(output, weights=data.num_events[source_indices][valid], minlength=size)
        if dependencies is not None or bundle is not None:
            dependency_inputs.append(flat[valid])
            dependency_outputs.append(output)
            dependency_weights.append(weight[valid])
    pooled = tuple(array.reshape(shape) for array in sums)
    signal, variance = finalize(*pooled)
    output_volume = np.ones(shape)
    for dim,edge in enumerate(edges):
        axis_shape = [1]*len(shape)
        axis_shape[dim] = len(edge)-1
        output_volume *= np.diff(edge).reshape(axis_shape)
    coverage = np.clip(support.reshape(shape)/output_volume, 0, 1)
    events = events.reshape(shape)
    output_dependencies = None
    output_counting_dependencies = None
    if bundle is not None:
        from .measurement_dependencies import (
            CountingDependencies,
            project_source_dependencies,
            ratio_source_dependencies,
        )
        inputs, output = map(np.concatenate, (dependency_inputs, dependency_outputs))
        output_counting_dependencies = CountingDependencies(
            numerator_dependencies=project_source_dependencies(bundle.numerator_dependencies, inputs, output, np.ones(len(inputs)), shape),
            exposure_dependencies=project_source_dependencies(bundle.exposure_dependencies, inputs, output, np.ones(len(inputs)), shape))
        output_dependencies = ratio_source_dependencies(output_counting_dependencies, pooled[0], pooled[2])
        variance = np.where(pooled[2] > 0, output_dependencies.variance(), np.nan)
        pooled = (pooled[0], output_counting_dependencies.numerator_dependencies.variance(), pooled[2])
    elif dependencies is not None:
        from .measurement_dependencies import project_source_dependencies
        inputs, output, weights = map(np.concatenate, (dependency_inputs, dependency_outputs, dependency_weights))
        coefficients = weights/pooled[2].reshape(-1)[output]
        output_dependencies = project_source_dependencies(dependencies, inputs, output, coefficients, shape)
        variance = np.where(pooled[2] > 0, output_dependencies.variance(), np.nan)
        pooled = (pooled[0], np.where(pooled[2] > 0, variance*pooled[2]**2, 0), *pooled[2:])
    auxiliary = channels(*pooled)
    auxiliary["coverage_fraction"] = MDHistoChannel(coverage, label="Coverage", unit="fraction")
    metadata.update(measurement_contract=contract.to_dict(), signal_unit=contract.value_units, zero_event_bins_are_measured=True,
        measurement_reduction={"version": 1, "assignment": "aligned_complete_cells"})
    if contract.kind == "continuous":
        metadata.pop(EVENT_STATISTICS_KEY, None)
    axes = tuple(replace(axis, values=edge, metadata={key:value for key,value in axis.metadata.items() if key != "discrete_centers"}) for axis,edge in zip(data.axes,edges,strict=True))
    return MDHistoData(axes=axes, signal=signal, errors=np.sqrt(variance),
        mask=~np.isfinite(signal) | ~np.isfinite(variance) | (coverage < minimum_coverage),
        num_events=events, metadata=metadata, auxiliary_channels=auxiliary,
        coordinate_system=data.coordinate_system, visual_normalization=data.visual_normalization,
        source_dependencies=output_dependencies, counting_dependencies=output_counting_dependencies)


def combine_measurement_histograms(datasets) -> MDHistoData:
    """Pool compatible aligned payloads, reconciling scaled Gaussian precision.

    Contracts and physical grids must agree. Shared primitive IDs combine before
    squaring. Mixed tracked/untracked inputs require source replay rather than an
    invented independence assumption. This does not sum divided intensities.
    """
    from .measurement_aggregation import (
        MEASUREMENT_STATISTICS_KEY,
        initialize_measurement_statistics,
        measurement_statistics_channels,
        normalized_measurement_statistics,
        rescale_measurement_statistics,
        selected_measurement_statistics,
    )
    from .measurement_dependencies import (
        CountingDependencies,
        combine_source_dependencies,
        project_source_dependencies,
        ratio_source_dependencies,
    )
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("At least one measurement histogram is required")
    first = datasets[0]
    contract = MeasurementContract.from_dict(first.metadata.get("measurement_contract", {}))
    from .project_dataset_io import _json_safe_value
    for data in datasets:
        if data.metadata.get("measurement_contract") != contract.to_dict():
            raise ValueError("Combining measurements requires matching explicit contracts and units")
        if len(data.axes) != len(first.axes) or any(
            axis.units != original.units or axis.kind != original.kind or axis.name != original.name
            or _json_safe_value(axis.metadata) != _json_safe_value(original.metadata)
            or not np.array_equal(axis.values, original.values)
            for axis,original in zip(data.axes,first.axes,strict=True)):
            raise ValueError("Combining measurements requires the same physical grid; rebin/replay sources first")
    if contract.kind not in {"counting", "continuous"}:
        raise ValueError("This combination target requires original source replay")
    counting = contract.kind == "counting"
    statistics, markers, valid = [], [], []
    for data in datasets:
        if counting:
            stats = selected_event_statistics(data)
            marker = None
        else:
            stats = selected_measurement_statistics(data)
            marker = data.metadata.get(MEASUREMENT_STATISTICS_KEY)
            if stats is None:
                if marker is not None:
                    raise ValueError("Stored measurement statistics are stale; replay sources")
                stats,marker = initialize_measurement_statistics(data.signal, data.errors,
                    replace(contract, dependence="independent", missing="omit"), mask=data.mask)
        if stats is None:
            raise ValueError("Combination requires validated additive measurement statistics")
        good = ~data.mask & np.isfinite(data.signal) & np.isfinite(data.errors) & (stats[2] > 0)
        if contract.missing == "reject" and not np.all(good):
            raise ValueError("The contract rejects missing composite observations")
        statistics.append(stats)
        markers.append(marker)
        valid.append(good)
    if not counting and contract.estimator == "inverse_variance_mean":
        zero_scaled = [bool(marker.get("deterministic_zero_scale")) for marker in markers]
        if any(zero_scaled) and not all(zero_scaled):
            raise SourceReplayRequired("Combining exact-zero and finite precision measurements requires an exact-constraint target or source replay")
        scale = min(float(marker["precision_scale"]) for marker in markers)
        statistics = [rescale_measurement_statistics(stats,marker,scale)[0]
                      for stats,marker in zip(statistics,markers,strict=True)]
        markers = [{**marker, "precision_scale": scale} for marker in markers]
    accumulators = [np.zeros(first.shape) for _ in statistics[0]]
    for stats,good in zip(statistics,valid,strict=True):
        for target,array in zip(accumulators,stats,strict=True):
            target += np.where(good,array,0)
    pooled = tuple(accumulators)
    finalize = normalized_event_statistics if counting else normalized_measurement_statistics
    signal, variance = finalize(*pooled)
    dependencies = [data.source_dependencies for data in datasets]
    bundles = [data.counting_dependencies for data in datasets]
    output_dependencies = output_bundle = None
    if any(bundle is not None for bundle in bundles):
        if not counting or not all(bundle is not None for bundle in bundles):
            raise ValueError("Mixed uncertain exposure models require source replay")
        projected = []
        for bundle,good in zip(bundles,valid,strict=True):
            indices = np.flatnonzero(good)
            projected.append(CountingDependencies(
                numerator_dependencies=project_source_dependencies(bundle.numerator_dependencies, indices,indices,np.ones(len(indices)),first.shape),
                exposure_dependencies=project_source_dependencies(bundle.exposure_dependencies,indices,indices,np.ones(len(indices)),first.shape)))
        output_bundle = CountingDependencies(
            numerator_dependencies=combine_source_dependencies([bundle.numerator_dependencies for bundle in projected],np.ones(len(projected))),
            exposure_dependencies=combine_source_dependencies([bundle.exposure_dependencies for bundle in projected],np.ones(len(projected))))
        output_dependencies = ratio_source_dependencies(output_bundle,pooled[0],pooled[2])
        variance = np.where(pooled[2] > 0, output_dependencies.variance(),np.nan)
        pooled = (pooled[0], output_bundle.numerator_dependencies.variance(),pooled[2])
    elif any(payload is not None for payload in dependencies):
        if not all(payload is not None for payload in dependencies):
            raise ValueError("Mixed tracked and untracked measurements require source replay")
        projected = []
        for payload,stats,good in zip(dependencies,statistics,valid,strict=True):
            indices = np.flatnonzero(good)
            factors = stats[2].reshape(-1)[indices]/pooled[2].reshape(-1)[indices]
            projected.append(project_source_dependencies(payload,indices,indices,factors,first.shape))
        output_dependencies = combine_source_dependencies(projected,np.ones(len(projected)))
        variance = np.where(pooled[2] > 0, output_dependencies.variance(),np.nan)
        pooled = (pooled[0],np.where(pooled[2] > 0,variance*pooled[2]**2,0),*pooled[2:])
    elif contract.dependence != "independent" or contract.normalizer != "known":
        raise ValueError("This contract requires source dependencies or source replay")
    if output_dependencies is None:
        from .source_lineage import validate_independent_source_lineage

        validate_independent_source_lineage(*datasets)
    channels = event_statistics_channels(*pooled) if counting else measurement_statistics_channels(*pooled)
    coverage = np.maximum.reduce([np.where(good,mdhisto_coverage_fraction(data),0)
        for data,good in zip(datasets,valid,strict=True)])
    channels["coverage_fraction"] = MDHistoChannel(coverage,label="Coverage",unit="fraction")
    metadata = dict(first.metadata)
    from .source_lineage import merge_source_lineage_metadata

    metadata.update(merge_source_lineage_metadata(*datasets))
    metadata.update(measurement_combination={"version": 1,"sources": len(datasets)},zero_event_bins_are_measured=True)
    if not counting:
        metadata[MEASUREMENT_STATISTICS_KEY] = markers[0]
    metadata["coverage_combination"] = "maximum_source_fraction; overlap_geometry_not_reconstructed"
    events = np.zeros(first.shape)
    for data,good in zip(datasets,valid,strict=True):
        events += np.where(good,data.num_events,0)
    return MDHistoData(first.axes, signal,np.sqrt(variance),~np.isfinite(signal),events,
        coordinate_system=first.coordinate_system,visual_normalization=first.visual_normalization,
        metadata=metadata,auxiliary_channels=channels,source_dependencies=output_dependencies,counting_dependencies=output_bundle)
