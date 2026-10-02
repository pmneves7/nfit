"""Prepared strip profiles for histogram waterfalls, independent of plotting."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .histogram_statistics import (
    EVENT_STATISTICS_CHANNELS,
    EVENT_STATISTICS_KEY,
    NORMALIZATION_DENOMINATOR,
    normalized_event_statistics,
    pool_event_statistics,
)
from .measurement_aggregation import (
    MEASUREMENT_STATISTICS_CHANNELS,
    MEASUREMENT_STATISTICS_KEY,
    initialize_measurement_statistics,
    normalized_measurement_statistics,
    pool_measurement_statistics,
    reduce_source_dependencies,
)
from .measurement_contracts import MeasurementContract


def prepare_waterfall_measurement_profile(view, values, errors, selected):
    """Prepare a declared waterfall strip with additive statistics."""
    from .histogram_statistics import event_statistics_channels
    from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
    from .measurement_aggregation import measurement_statistics_channels
    from .measurement_profiles import MeasurementProfile

    mask = np.asarray(view.get("mask", np.zeros(values.shape, bool)), bool)
    selection_mask = ~np.broadcast_to(selected[:, None], values.shape)
    if view.get("masks_applied", True):
        selection_mask = selection_mask | mask
    declaration = view.get("measurement_contract")
    if declaration is None and view.get("measurement_target_required"):
        raise ValueError("Waterfall aggregation of derived measurements requires an explicit statistical target")
    is_count = isinstance(view.get(EVENT_STATISTICS_KEY), dict) and all(
        name in view for name in (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR))
    contract = (MeasurementContract.from_dict(declaration) if declaration is not None else
                MeasurementContract(kind="counting", estimator="exposure_pool", quantity="intensity",
                    value_units=view.get("signal_unit", "unspecified intensity units"),
                    exposure_units=view.get("exposure_unit", "arbitrary normalization units")))
    dependencies = view.get("source_dependencies")
    bundle = view.get("counting_dependencies")
    if (contract.dependence != "independent" and dependencies is None) or (contract.normalizer != "known" and bundle is None):
        raise ValueError("Waterfall aggregation requires a dependency payload or source replay")
    if contract.kind == "counting":
        if not is_count:
            raise ValueError("Counting waterfalls require validated C,V,N payloads")
        arrays = tuple(np.asarray(view[name], float) for name in (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR))
        stats = pool_event_statistics(*arrays, (0,), mask=selection_mask)
        signal, variance = normalized_event_statistics(*stats)
        channels = event_statistics_channels(*stats)
        weights = np.where(~selection_mask, arrays[2], 0)
        metadata = {EVENT_STATISTICS_KEY: dict(view[EVENT_STATISTICS_KEY]), "zero_event_bins_are_measured": True}
    elif contract.kind == "continuous":
        if isinstance(view.get(MEASUREMENT_STATISTICS_KEY), dict) and all(name in view for name in MEASUREMENT_STATISTICS_CHANNELS):
            arrays = tuple(np.asarray(view[name], float) for name in MEASUREMENT_STATISTICS_CHANNELS)
            marker = dict(view[MEASUREMENT_STATISTICS_KEY])
            expected_signal, expected_variance = normalized_measurement_statistics(*arrays)
            visible = ~selection_mask
            if marker.get("version") != 1 or marker.get("estimator") != contract.estimator or not (
                np.allclose(expected_signal[visible], values[visible], rtol=1e-12, atol=0, equal_nan=True)
                and np.allclose(expected_variance[visible], np.square(errors)[visible], rtol=1e-12, atol=0, equal_nan=True)):
                raise ValueError("Stored waterfall statistics do not reproduce the declared observations; replay sources")
        else:
            arrays, marker = initialize_measurement_statistics(values, errors, replace(contract, dependence="independent", missing="omit"), mask=selection_mask)
        stats = pool_measurement_statistics(*arrays, (0,), mask=selection_mask)
        signal, variance = normalized_measurement_statistics(*stats)
        channels = measurement_statistics_channels(*stats)
        weights = np.where(~selection_mask, arrays[2], 0)
        metadata = {MEASUREMENT_STATISTICS_KEY: marker, "zero_event_bins_are_measured": True}
    elif contract.kind == "sampled_function" and np.count_nonzero(selected) == 1:
        row = int(np.flatnonzero(selected)[0])
        signal, variance = values[row].copy(), np.square(errors[row])
        signal = np.where(selection_mask[row], np.nan, signal)
        variance = np.where(selection_mask[row], np.nan, variance)
        channels, metadata = {}, {"zero_event_bins_are_measured":True}
        weights = (~selection_mask).astype(float)
        stats = None
    else:
        raise ValueError("This waterfall target requires its measurement payload or source replay")
    if contract.missing == "reject" and np.any(np.broadcast_to(selected[:, None], values.shape) & (selection_mask | ~np.isfinite(values) | ~np.isfinite(errors) | (weights <= 0))):
        raise ValueError("The contract rejects missing, masked or unexposed observations")
    output_dependencies = None
    output_bundle = None
    if bundle is not None:
        from .measurement_dependencies import CountingDependencies, ratio_source_dependencies
        valid_weights = np.where((weights > 0) & np.isfinite(values) & np.isfinite(errors), 1., 0.)
        ones = np.ones(signal.shape)
        output_bundle = CountingDependencies(
            numerator_dependencies=reduce_source_dependencies(bundle.numerator_dependencies, (slice(None), slice(None)), valid_weights, (0,), ones),
            exposure_dependencies=reduce_source_dependencies(bundle.exposure_dependencies, (slice(None), slice(None)), valid_weights, (0,), ones))
        output_dependencies = ratio_source_dependencies(output_bundle, stats[0], stats[2])
        variance = np.where(stats[2] > 0, output_dependencies.variance(), np.nan)
        stats = (stats[0], output_bundle.numerator_dependencies.variance(), stats[2])
        channels = event_statistics_channels(*stats)
    elif dependencies is not None:
        dependencies.validate_variances(np.square(errors), mask=mask if view.get("masks_applied", True) else None)
        output_dependencies = reduce_source_dependencies(dependencies, (slice(None), slice(None)), weights, (0,),
            np.ones(signal.shape) if stats is None else stats[2])
        variance = output_dependencies.variance()
        if stats is not None:
            stats = (stats[0], variance*stats[2]**2, *stats[2:])
            channels = event_statistics_channels(*stats) if contract.kind == "counting" else measurement_statistics_channels(*stats)
    metadata.update(measurement_contract=contract.to_dict(), num_events_semantics=view.get("num_events_semantics", "unspecified_source_multiplicity"),
                    signal_semantics="density", signal_unit=contract.value_units)
    coverage = np.asarray(view.get("coverage_fraction", np.ones(values.shape)), float)
    widths = np.diff(np.asarray(view["y_edges"], float))[:,None]
    good = ~selection_mask & (weights > 0) & np.isfinite(values) & np.isfinite(errors)
    covered = np.where(good & np.isfinite(coverage), np.clip(coverage,0,1), 0)
    requested_width = np.sum(widths[selected])
    fraction = np.sum(covered*widths, axis=0)/requested_width
    channels["coverage_fraction"] = MDHistoChannel(fraction, label="Geometric coverage", unit="1")
    data = MDHistoData((MDHistoAxis("profile", view["x_edges"], "", "unknown"),), signal, np.sqrt(variance),
                       ~np.isfinite(signal), np.sum(np.where(~selection_mask, view["num_events"], 0), axis=0),
                       metadata=metadata, auxiliary_channels=channels, source_dependencies=output_dependencies, counting_dependencies=output_bundle)
    return MeasurementProfile(data, contract), weights[selected, :]
