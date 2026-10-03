"""Bounded covariance replay for cached directional-background fields.

Recipes retain acquisition transforms, never an event payload or a covariance
matrix. Explicit final profiles stream the original observations on demand.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np

from .event_bin_indices import flat_bin_indices
from .histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    NORMALIZATION_DENOMINATOR,
    event_statistics_channels,
)
from .mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from .measurement_contracts import MeasurementContract
from .measurement_dependencies import SourceReplayRequired
from .source_lineage import tracked_source_ids

CACHED_BACKGROUND_REPLAY = "cached_background_replay"
CACHED_BACKGROUND_REPLAY_VERSION = 1
REPLAY_INDEPENDENT_VARIANCE = "background_replay_independent_variance"
MAX_RECIPE_BYTES = 16 * 1024**2
MAX_QUERY_CACHE_BYTES = 16 * 1024**2
_QUERY_CACHE = OrderedDict()
_QUERY_CACHE_LOCK = RLock()
_QUERY_CACHE_BYTES = 0


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def packed_replay_mask(mask):
    """Encode a grid/detector mask without retaining or serializing its array."""
    if mask is None:
        return None
    return base64.b64encode(np.packbits(np.asarray(mask, bool).ravel())).decode("ascii")


def _unpack(mask, size):
    if mask is None:
        return None
    packed = np.frombuffer(base64.b64decode(mask, validate=True), np.uint8)
    if packed.size != (size + 7) // 8:
        raise SourceReplayRequired("Invalid cached background replay mask")
    return np.unpackbits(packed, count=size).astype(bool)


def _file_identity(path):
    try:
        stat = Path(path).stat()
    except OSError as error:
        raise SourceReplayRequired(f"Background replay source is unavailable: {path}") from error
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def background_replay_source_recipe(
    source, *, workspace_path, frame, gonio, shape, dtype, content_sha256,
    transforms, weights, detector_ids, accepted_ids, excluded_bins,
):
    """Capture the complete resolved event projection after its source stream."""
    path = str(Path(source.metadata["source_file"]).expanduser().resolve())
    ids = np.asarray(detector_ids, np.int64)
    acceptance, exclusions, acceptance_indices, exclusion_indices = [], [], [], []
    for accepted, excluded in zip(accepted_ids, excluded_bins, strict=True):
        flags = packed_replay_mask(np.isin(ids, accepted))
        if flags not in acceptance:
            acceptance.append(flags)
        acceptance_indices.append(acceptance.index(flags))
        flags = packed_replay_mask(excluded)
        if flags not in exclusions:
            exclusions.append(flags)
        exclusion_indices.append(exclusions.index(flags))
    return {"source_file": path, "file_identity": _file_identity(path),
            "source_lineage_ids": list(tracked_source_ids(source)),
            "workspace_path": workspace_path, "event_shape": list(shape),
            "event_dtype": str(dtype), "event_sha256": content_sha256,
            "experiment_index": int(source.metadata["mdevent_experiment_index"]),
            "frame": frame, "goniometer": np.asarray(gonio).tolist(),
            "transforms": [np.asarray(inverse).tolist() for inverse, _ in transforms],
            "energy_bounds": [np.asarray(bounds).tolist() for _, bounds in transforms],
            "weights": np.asarray(weights).tolist(), "detector_ids": ids.tolist(),
            "acceptance": acceptance, "acceptance_indices": acceptance_indices,
            "exclusions": exclusions, "exclusion_indices": exclusion_indices}


def background_replay_payload(data, sources):
    """Declare exposure pooling of a cached measured-background field."""
    payload = {"version": CACHED_BACKGROUND_REPLAY_VERSION,
               "target": "background_exposure_weighted_field_mean",
               "shape": list(data.shape), "edges": [axis.values.tolist() for axis in data.axes],
               "weight_channel": NORMALIZATION_DENOMINATOR,
               "independent_variance_channel": REPLAY_INDEPENDENT_VARIANCE,
               "terms": [{"scale": 1., "denominator_channel": NORMALIZATION_DENOMINATOR,
                          "sources": sources, "excluded": None}],
               "normalizer": "known", "calibration_uncertainty": "not_represented"}
    return payload


def _validate_size(payload):
    if len(json.dumps(payload, separators=(",", ":"))) > MAX_RECIPE_BYTES:
        raise SourceReplayRequired("Cached background recipe exceeds its storage budget; use direct final-grid replay")


def background_replay_metadata(metadata, payload):
    """Attach optional bounded provenance without blocking histogram creation."""
    metadata = dict(metadata)
    try:
        _validate_size(payload)
    except SourceReplayRequired as error:
        metadata.pop(CACHED_BACKGROUND_REPLAY, None)
        metadata["background_replay_unavailable_reason"] = str(error)
        metadata["background_profile_uncertainty"] = "diagonal_approximation_replay_recipe_budget_exceeded"
    else:
        metadata[CACHED_BACKGROUND_REPLAY] = payload
        metadata.pop("background_replay_unavailable_reason", None)
        metadata["background_profile_uncertainty"] = "diagonal_approximation_until_source_replay"
    return metadata


def _has_replay(data):
    payload = data.metadata.get(CACHED_BACKGROUND_REPLAY)
    return isinstance(payload, dict) and payload.get("version") == CACHED_BACKGROUND_REPLAY_VERSION


def replay_represents_background_overlap(data, background):
    """Permit tracked reuse only when all shared primitives are replay terms."""
    if not (_has_replay(data) and _has_replay(background)):
        return False
    overlap = set(tracked_source_ids(data)) & set(tracked_source_ids(background))
    payload = data.metadata[CACHED_BACKGROUND_REPLAY]
    if overlap & set(payload.get("independent_source_ids", [])):
        return False
    def represented(source):
        return {identity for term in source.metadata[CACHED_BACKGROUND_REPLAY]["terms"]
                for recipe in term["sources"] for identity in recipe.get("source_lineage_ids", [])}
    return overlap <= represented(data) & represented(background)


def exclude_background_replay_cells(data, excluded, metadata):
    """Explicitly zeroed background cells carry no primitive sensitivity."""
    if not _has_replay(data):
        return metadata
    payload = copy.deepcopy(data.metadata[CACHED_BACKGROUND_REPLAY])
    for term in payload["terms"]:
        previous = _unpack(term.get("excluded"), data.signal.size)
        flags = np.asarray(excluded, bool).ravel()
        term["excluded"] = packed_replay_mask(flags if previous is None else flags | previous)
    return {**metadata, CACHED_BACKGROUND_REPLAY: payload}


def background_replay_subtraction(data, background, result, scale):
    """Retain the target and covariance recipe through aligned subtraction.

    The target is a sample-exposure-weighted mean of the *subtracted field*,
    not separate pooling of sample and background normalizations.
    """
    if not _has_replay(background):
        return result
    if background.metadata[CACHED_BACKGROUND_REPLAY].get("sample_uncertainty") is not None:
        # A subtracted operand has its own independent sample primitives; its
        # background replay terms alone cannot certify their signed reuse.
        metadata = dict(result.metadata)
        metadata.pop(CACHED_BACKGROUND_REPLAY, None)
        metadata["background_replay_unavailable_reason"] = (
            "An already-subtracted background operand contains independent sample "
            "primitives; replay all components to represent their covariance"
        )
        metadata["background_profile_uncertainty"] = "diagonal_approximation_untracked_background_component"
        return result.with_updates(metadata=metadata)
    if _has_replay(data):
        payload = copy.deepcopy(data.metadata[CACHED_BACKGROUND_REPLAY])
        independent = data.auxiliary_channels[REPLAY_INDEPENDENT_VARIANCE].values
    else:
        channel = data.auxiliary_channels.get(NORMALIZATION_DENOMINATOR)
        if channel is None:
            # Unnormalized/other measurement targets require a separate explicit
            # declaration; an intensity alone does not define acquisition weight.
            metadata = dict(result.metadata)
            metadata["background_profile_uncertainty"] = "diagonal_approximation_replay_target_unavailable"
            return result.with_updates(metadata=metadata)
        payload = copy.deepcopy(background.metadata[CACHED_BACKGROUND_REPLAY])
        payload["terms"] = []
        payload["independent_source_ids"] = list(tracked_source_ids(data))
        independent = np.square(data.errors)
    channels = dict(result.auxiliary_channels)
    channel = data.auxiliary_channels.get(NORMALIZATION_DENOMINATOR)
    if channel is None:
        raise SourceReplayRequired("Cached subtraction requires the sample's exposure payload")
    channels[NORMALIZATION_DENOMINATOR] = channel
    channels[REPLAY_INDEPENDENT_VARIANCE] = MDHistoChannel(
        independent, label="Independent sample variance under its recorded reduction policy"
    )
    background_payload = background.metadata[CACHED_BACKGROUND_REPLAY]
    for term in background_payload["terms"]:
        name = f"background_replay_exposure_{len(payload['terms'])}"
        channels[name] = background.auxiliary_channels[term["denominator_channel"]]
        payload["terms"].append({**copy.deepcopy(term), "scale": -float(scale) * term["scale"],
                                 "denominator_channel": name})
    payload.update(target="sample_exposure_weighted_subtracted_field_mean",
                   weight_channel=NORMALIZATION_DENOMINATOR,
                   independent_variance_channel=REPLAY_INDEPENDENT_VARIANCE,
                   sample_uncertainty="recorded_histogram_variance")
    metadata = background_replay_metadata(result.metadata, payload)
    metadata["normalization_denominator"] = channel.values
    return result.with_updates(metadata=metadata, auxiliary_channels=channels)


def merge_background_replay_partitions(partitions, metadata, channels):
    """Retain per-source output masks when independently masked runs pool."""
    if not partitions or not all(_has_replay(data) for data in partitions):
        return metadata, channels
    payload = copy.deepcopy(partitions[0].metadata[CACHED_BACKGROUND_REPLAY])
    payload["terms"] = []
    for data in partitions:
        excluded = packed_replay_mask(data.mask)
        for term in data.metadata[CACHED_BACKGROUND_REPLAY]["terms"]:
            payload["terms"].append({**copy.deepcopy(term), "excluded": excluded})
    metadata = background_replay_metadata(metadata, payload)
    channels = {**channels, REPLAY_INDEPENDENT_VARIANCE: MDHistoChannel(
        np.zeros(partitions[0].shape), label="Independent variance outside replay sources")}
    return metadata, channels


def _validate_payload(data):
    if not _has_replay(data):
        reason = data.metadata.get("background_replay_unavailable_reason", "recompute the background or replay directly on the final grid")
        raise SourceReplayRequired(f"Exact cached background uncertainty requires a retained replay recipe: {reason}")
    payload = data.metadata[CACHED_BACKGROUND_REPLAY]
    if tuple(payload["shape"]) != data.shape or any(
        not np.array_equal(axis.values, edges)
        for axis, edges in zip(data.axes, payload["edges"], strict=True)
    ):
        raise SourceReplayRequired("Cached background grid changed; replay its sources on the requested grid")
    for name in (payload["weight_channel"], payload["independent_variance_channel"],
                 *(term["denominator_channel"] for term in payload["terms"])):
        if name not in data.auxiliary_channels:
            raise SourceReplayRequired("Cached background replay is missing its exposure/variance payload")
    return payload


def _projection_variance(data, payload, selected, indices, weights, bins, progress_callback, max_batch_bytes):
    """Stream primitive coefficients, combining all signed reuse before squaring."""
    import h5py

    shape = data.shape
    size = data.signal.size
    edges = [np.asarray(axis.values, float) for axis in data.axes]
    grouped = {}
    for term in payload["terms"]:
        denominator = data.auxiliary_channels[term["denominator_channel"]].values.ravel()
        excluded = _unpack(term.get("excluded"), size)
        included = selected.ravel() & np.isfinite(denominator) & (denominator > 0)
        if excluded is not None:
            included &= ~excluded
        coefficient = np.zeros(size)
        coefficient[included] = float(term["scale"]) * weights.ravel()[included] / denominator[included]
        for source in term["sources"]:
            key = (source["source_file"], source["workspace_path"])
            grouped.setdefault(key, []).append((source, coefficient))
    variance = np.zeros(bins)
    receipts = []
    for (path, workspace_path), terms in grouped.items():
        identity = _file_identity(path)
        for source, _ in terms:
            # Device/inode can change when moving an unchanged source. Size and
            # timestamp are fast stale checks; streamed bytes are authoritative.
            if any(identity[key] != source["file_identity"][key] for key in ("size", "mtime_ns")):
                raise SourceReplayRequired(f"Background source changed since this histogram was cached: {path}")
        transform_count = sum(len(source["weights"]) for source, _ in terms)
        rows = max(1, min(100_000, int(max_batch_bytes) // max(128 * transform_count, 1)))
        prepared = []
        for source, coefficient in terms:
            ids = np.asarray(source["detector_ids"], np.int64)
            accepted = [_unpack(mask, len(ids)) for mask in source["acceptance"]]
            excluded = [_unpack(mask, size) for mask in source["exclusions"]]
            prepared.append((source, coefficient, ids, accepted, excluded))
        digest = hashlib.sha256()
        with h5py.File(path, "r") as handle:
            values = handle[f"{workspace_path}/event_data/event_data"]
            if any(list(values.shape) != source["event_shape"] or str(values.dtype) != source["event_dtype"]
                   for source, _ in terms):
                raise SourceReplayRequired("Background source event layout changed")
            for start in range(0, values.shape[0], rows):
                raw = np.asarray(values[start:start + rows])
                digest.update(raw.tobytes())
                block = np.asarray(raw, float)
                # Each original event remains one primitive even when multiple
                # replay terms or sample angles use it with different signs.
                locations, coefficients = [], []
                for source, coefficient, ids, accepted, excluded in prepared:
                    belongs = block[:, 2].astype(np.int64) == source["experiment_index"]
                    if np.any(belongs & (block[:, 3] != 0)):
                        raise SourceReplayRequired("Background replay requires one goniometer per experiment")
                    lab = block[:, 5:8]
                    if source["frame"] == "QSample":
                        lab = lab @ np.asarray(source["goniometer"]).T
                    for column, (inverse, bounds, weight) in enumerate(zip(
                        source["transforms"], source["energy_bounds"], source["weights"], strict=True,
                    )):
                        coordinates = np.column_stack((lab @ np.asarray(inverse).T, block[:, 8]))
                        flat = flat_bin_indices(coordinates, edges, shape)
                        valid = belongs & (flat >= 0) & (block[:, 8] >= bounds[0]) & (block[:, 8] <= bounds[1])
                        accepted_ids = ids[accepted[source["acceptance_indices"][column]]]
                        valid &= np.isin(block[:, 4].astype(np.int64), accepted_ids)
                        exclusion = excluded[source["exclusion_indices"][column]]
                        if exclusion is not None:
                            valid &= ~exclusion[np.maximum(flat, 0)]
                        valid &= selected.ravel()[np.maximum(flat, 0)]
                        final = np.full(len(block), -1, np.int64)
                        final[valid] = indices.ravel()[flat[valid]]
                        gain = np.zeros(len(block))
                        gain[valid] = weight * coefficient[flat[valid]]
                        locations.append(final)
                        coefficients.append(gain)
                if locations:
                    locations = np.asarray(locations).T
                    coefficients = np.asarray(coefficients).T
                    order = np.argsort(locations, axis=1, kind="stable")
                    ordered = np.take_along_axis(locations, order, axis=1)
                    gains = np.take_along_axis(coefficients, order, axis=1)
                    first = np.ones(ordered.shape, bool)
                    first[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
                    starts = np.flatnonzero(first.ravel())
                    combined = np.add.reduceat(gains.ravel(), starts)
                    output = ordered.ravel()[starts]
                    row = starts // ordered.shape[1]
                    valid = (output >= 0) & (combined != 0)
                    source_variance = block[row[valid], 1]
                    if np.any(~np.isfinite(source_variance) | (source_variance < 0)):
                        raise SourceReplayRequired("Background source has invalid primitive variance")
                    np.add.at(variance, output[valid], source_variance * combined[valid]**2)
                if progress_callback is not None:
                    progress_callback({"stage": "cached_background_covariance", "iteration": min(start + rows, values.shape[0]),
                                       "total": values.shape[0], "message": "Replaying background covariance for the explicit final profile"})
            actual_digest = digest.hexdigest()
            if any(actual_digest != source["event_sha256"] for source, _ in terms):
                raise SourceReplayRequired("Background event contents differ from the cached histogram")
            receipts.append({"source_file": path, "event_sha256": actual_digest})
    return variance, receipts


def clear_cached_background_profile_queries():
    """Release the bounded numerical cache for explicit final-profile queries."""
    global _QUERY_CACHE_BYTES
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE.clear()
        _QUERY_CACHE_BYTES = 0


def replay_cached_background_profile(
    data, *, selected, indices, edges, centers=None, coverage_weights=None,
    coverage_threshold=0., progress_callback=None, max_batch_bytes=64 * 1024**2,
):
    """Prepare an exact background-covariance profile of a cached original grid.

    ``selected`` and ``indices`` have the full histogram shape. Each selected
    cached cell belongs to one final profile bin. Masks and missing background
    coverage are omitted explicitly; covered zeros retain their exposure.
    The mean weights are background exposure for a standalone background and
    sample exposure for an already-subtracted field. This differs from reducing
    sample/background components separately on a new physical final grid.

    Original events are loaded only for this explicit operation. Repeated source
    observations are combined before squaring coefficients. Sample uncertainty
    keeps its recorded histogram variance; shared calibration/exposure uncertainty
    and unavailable sample cross-bin covariance are not reconstructed. Final
    profile bins can share observations: only their marginal variances are
    returned, so subsequent coarsening or joint fitting requires source replay
    or an explicit joint dependency payload.
    """
    from .measurement_profiles import MeasurementProfile

    global _QUERY_CACHE_BYTES
    payload = _validate_payload(data)
    selected, indices = np.asarray(selected, bool), np.asarray(indices)
    edges = np.asarray(edges, float)
    if selected.shape != data.shape or indices.shape != data.shape or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Selection and integer profile assignments must match the original cached grid")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Profile edges must be finite and increasing")
    bins = len(edges) - 1
    if np.any(selected & ((indices < 0) | (indices >= bins))):
        raise ValueError("Selected profile assignments lie outside its bins")
    if not np.isfinite(coverage_threshold) or not 0 <= coverage_threshold <= 1:
        raise ValueError("Coverage threshold must be between zero and one")
    weights = data.auxiliary_channels[payload["weight_channel"]].values
    independent = data.auxiliary_channels[payload["independent_variance_channel"]].values
    good = selected & ~data.mask & np.isfinite(data.signal) & np.isfinite(weights) & (weights > 0)
    good &= np.isfinite(independent) & (independent >= 0)
    def pool(values, valid=good):
        return np.bincount(indices[valid], weights=np.asarray(values)[valid], minlength=bins)
    exposure = pool(weights)
    numerator = pool(np.where(good, data.signal, 0) * weights)
    independent_variance = pool(independent * weights**2)
    files = sorted({source["source_file"] for term in payload["terms"] for source in term["sources"]})
    identities = [(path, _file_identity(path)) for path in files]
    signature = hashlib.sha256()
    signature.update(_digest({"recipe": payload, "sources": identities}).encode())
    for array in (good, indices, np.where(good, weights, 0)):
        signature.update(np.ascontiguousarray(array).tobytes())
    for term in payload["terms"]:
        signature.update(np.ascontiguousarray(
            data.auxiliary_channels[term["denominator_channel"]].values[good]
        ).tobytes())
    signature.update(str(bins).encode())
    cache_key = signature.hexdigest()
    with _QUERY_CACHE_LOCK:
        cached = _QUERY_CACHE.get(cache_key)
        if cached is not None:
            _QUERY_CACHE.move_to_end(cache_key)
    cache_hit = cached is not None
    if cached is None:
        source_variance, receipts = _projection_variance(
            data, payload, good, indices, weights, bins, progress_callback, max_batch_bytes,
        )
        source_variance.setflags(write=False)
        cached = (source_variance, receipts)
        cache_size = source_variance.nbytes + len(json.dumps(receipts))
        if cache_size <= MAX_QUERY_CACHE_BYTES:
            with _QUERY_CACHE_LOCK:
                previous = _QUERY_CACHE.pop(cache_key, None)
                if previous is not None:
                    _QUERY_CACHE_BYTES -= previous[0].nbytes + len(json.dumps(previous[1]))
                while _QUERY_CACHE and _QUERY_CACHE_BYTES + cache_size > MAX_QUERY_CACHE_BYTES:
                    previous = _QUERY_CACHE.popitem(last=False)[1]
                    _QUERY_CACHE_BYTES -= previous[0].nbytes + len(json.dumps(previous[1]))
                _QUERY_CACHE[cache_key] = cached
                _QUERY_CACHE_BYTES += cache_size
    source_variance, receipts = cached
    variance_numerator = independent_variance + source_variance
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = numerator / exposure
        variance = variance_numerator / exposure**2
    geometric = np.broadcast_to(1. if coverage_weights is None else coverage_weights, data.shape)
    if np.any(~np.isfinite(geometric) | (geometric <= 0)):
        raise ValueError("Positive finite coverage weights must match the cached grid")
    support = pool(geometric, selected)
    coverage = np.zeros(bins)
    np.divide(pool(geometric), support, out=coverage, where=support > 0)
    mask = (exposure <= 0) | ~np.isfinite(signal) | (coverage < coverage_threshold)
    contract = MeasurementContract(kind="counting", estimator="exposure_pool",
        quantity=payload["target"].replace("_", " "),
        value_units=str(data.metadata.get("signal_unit") or "arbitrary intensity units"),
        exposure_units=data.auxiliary_channels[payload["weight_channel"]].unit or "arbitrary normalization units",
        dependence="shared_sources")
    metadata = {"measurement_contract": contract.to_dict(),
                EVENT_STATISTICS_KEY: {**EVENT_STATISTICS_METADATA,
                                       "covariance": "within_final_bins_only_requires_source_replay_for_aggregation"},
                "profile_covariance": "replayed_background_sources",
                "measurement_target": payload["target"],
                "background_profile_uncertainty": "source_covariance",
                "cross_profile_bin_covariance": "not_retained_requires_source_replay",
                "missing_background_policy": "omit_missing_coverage_and_masked_cells",
                "zero_event_bins_are_measured": True, "num_events_semantics": "sample_event_contributions" if "subtracted" in payload["target"] else "replayed_cached_cell_event_contributions",
                "cached_background_profile": {"version": 1, "recipe_signature": cache_key,
                    "sources": receipts, "query_cache_hit": cache_hit,
                    "normalizer": "known", "calibration_uncertainty": "not_represented",
                    "sample_uncertainty": payload.get("sample_uncertainty", "not_applicable"),
                    "variance": "observed_primitive_variance",
                    "confidence_intervals": "not_estimated_from_marginal_zero_count_display_bounds"}}
    channels = event_statistics_channels(numerator, variance_numerator, exposure)
    channels["coverage_fraction"] = MDHistoChannel(coverage, label="Coverage", unit="fraction")
    axis_metadata = {} if centers is None else {"discrete_centers": np.asarray(centers, float).tolist()}
    output = MDHistoData(axes=(MDHistoAxis("Profile", edges, "", "unknown", metadata=axis_metadata),),
                         signal=signal, errors=np.sqrt(variance), mask=mask,
                         num_events=pool(data.num_events), metadata=metadata, auxiliary_channels=channels)
    return MeasurementProfile(output, contract)
