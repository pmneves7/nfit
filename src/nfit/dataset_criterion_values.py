"""Bounded, unsubtracted per-acquisition diagnostics for dataset selection."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .cache_utils import dataset_content_signature
from .metadata_dimensions import metadata_channel_values

CRITERION_VALUE_VERSION = 1
_METADATA_MAX_BYTES = 32 * 1024**2


def value_signature(group, dataset, parameters):
    """Identify a scalar diagnostic independently of exclusion thresholds."""
    channel = parameters.get("channel", "elastic")
    if channel == "elastic":
        from .dgs_background_sources import raw_source
        from .raw_dgs_cache import reduction_signature
        from .reduction_recipes import effective_reduction_config

        if not raw_source(group):
            raise ValueError("Elastic intensity currently requires native raw DGS runs")
        identity = reduction_signature(dataset, effective_reduction_config(group, dataset))
        recipe = [channel, parameters.get("energy_min", -0.5), parameters.get("energy_max", 0.5)]
    elif channel == "metadata":
        source = "/".join(
            part.strip()
            for part in str(parameters.get("source", "metadata/incident_energy"))
            .replace(">", "/")
            .split("/")
            if part.strip()
        )
        recipe = [channel, source, parameters.get("statistic", "time_average")]
        identity = dataset_content_signature(dataset)
        if source.startswith(("metadata/", "parameters/")) or source == "temperature":
            values, unit = _numeric_channel(dataset, source)
            identity = [identity, values.shape, unit, hashlib.sha256(values.tobytes()).hexdigest()]
    else:
        raise ValueError(f"Unknown dataset diagnostic channel: {channel}")
    return hashlib.sha256(
        json.dumps([CRITERION_VALUE_VERSION, identity, recipe], sort_keys=True).encode()
    ).hexdigest()


def criterion_value(group, dataset, parameters, *, progress_callback=None):
    """Calculate one diagnostic; never bin, subtract, or change source arrays."""
    if parameters.get("channel", "elastic") == "metadata":
        return _metadata_value(dataset, parameters)
    return _elastic_value(group, dataset, parameters, progress_callback)


def _elastic_value(group, dataset, parameters, progress_callback):
    from . import raw_dgs
    from .dgs_background_sources import prepared_raw_source, raw_source
    from .raw_dgs_cache import iter_cached_event_chunks

    if not raw_source(group):
        raise ValueError("Elastic intensity currently requires native raw DGS runs")
    low, high = float(parameters.get("energy_min", -0.5)), float(parameters.get("energy_max", 0.5))
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Elastic energy limits must be finite and increasing (meV)")
    with prepared_raw_source(group, dataset) as (config, info, norm, archive, geometry, setup, _):
        ei = float(config.get("incident_energy_override") or info.incident_energy)
        bounds = raw_dgs._energy_transfer_bounds(config, ei)
        if low < bounds[0] or high > bounds[1]:
            raise ValueError(f"{dataset.name}: elastic window exceeds its reduction energy domain")
        exposure = float(norm["charge"]) * float(np.sum(norm["solid"][norm["solid"] > 0]))
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"{dataset.name}: no positive detector/beam normalization")
        # Preserve the full reduction domain before applying the diagnostic
        # interval: filtering earlier can change float32 boundary membership.
        chunks = (
            iter_cached_event_chunks(archive, 32 * 1024**2)
            if archive is not None
            else raw_dgs._iter_reduced_event_chunks(
                info, config, geometry, ei, bounds, 32 * 1024**2, hyspec_preprocessing=setup
            )
        )
        counts, variance, events, processed = 0.0, 0.0, 0, 0
        for block, raw_count in chunks:
            selected = (block[:, 3] >= low) & (block[:, 3] < high)
            counts += float(np.sum(block[selected, 4]))
            variance += float(np.sum(block[selected, 5]))
            events += int(np.count_nonzero(selected))
            processed += raw_count
            if progress_callback is not None:
                progress_callback(
                    dict(
                        stage="dataset_criterion_events",
                        iteration=processed,
                        total=info.event_count,
                        message=f"Reading elastic intensity: {dataset.name}",
                    )
                )
    return dict(
        value=counts / exposure,
        error=np.sqrt(variance) / exposure,
        units="relative integrated intensity",
        events=events,
        numerator=counts,
        normalization=exposure,
    )


def _numeric_channel(dataset, source):
    try:
        if not source.startswith(("metadata/", "parameters/")) and source != "temperature":
            import h5py

            filename = dataset.metadata.get("source_file") or getattr(
                dataset.data, "metadata", {}
            ).get("source_file")
            if filename:
                with h5py.File(filename, "r") as handle:
                    channel = handle[source]
                    if channel.ndim > 1 or channel.size * 8 > _METADATA_MAX_BYTES:
                        raise ValueError(
                            "Choose a scalar or a one-dimensional metadata log of at most 32 MiB"
                        )
        return metadata_channel_values(dataset, source)
    except (KeyError, TypeError, OSError, AttributeError) as error:
        raise ValueError(
            f"{dataset.name}: cannot read metadata channel {source!r}: {error}"
        ) from error


def _metadata_value(dataset, parameters):
    source = "/".join(
        part.strip()
        for part in str(parameters.get("source", "metadata/incident_energy"))
        .replace(">", "/")
        .split("/")
        if part.strip()
    )
    values, units = _numeric_channel(dataset, source)
    if values.ndim > 1:
        raise ValueError("Dataset diagnostics require scalar or one-dimensional metadata")
    statistic = parameters.get("statistic", "time_average")
    if statistic == "min":
        value = float(np.min(values))
    elif statistic == "max":
        value = float(np.max(values))
    elif statistic == "time_average":
        value = (
            float(values.ravel()[0]) if values.size == 1 else _time_average(dataset, source, values)
        )
    else:
        raise ValueError("Metadata statistic must be min, max, or time_average")
    return dict(value=value, units=units)


def _time_average(dataset, source, values):
    import h5py

    from .raw_dgs_goniometer import step_log_mean
    from .raw_dgs_pulses import iso_timestamp_ns, nexus_timestamps

    filename = dataset.metadata.get("source_file") or getattr(dataset.data, "metadata", {}).get(
        "source_file"
    )
    if not filename or source.startswith(("metadata/", "parameters/")):
        raise ValueError("Time averaging a series requires a timestamped NeXus log")
    with h5py.File(filename, "r") as handle:
        channel = handle[source]
        times = channel.parent.get("time")
        entry = handle[source.split("/")[0]]
        if times is None or "start_time" not in entry or "end_time" not in entry:
            raise ValueError("Time averaging requires log timestamps and run start/end times")
        if times.ndim != 1 or times.size != values.size:
            raise ValueError("Log timestamps must be one-dimensional and match the value count")
        stamps = nexus_timestamps(times)
        if stamps is None:
            raise ValueError("Log timestamps need an explicit absolute time origin")
        start = iso_timestamp_ns(entry["start_time"][()])
        stop = iso_timestamp_ns(entry["end_time"][()])
        if stop <= start:
            raise ValueError("Run end time must follow its start time")
        return float(step_log_mean(stamps, values, np.array([[start, stop]], dtype=np.int64)))
