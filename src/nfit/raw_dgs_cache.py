"""Dataset-owned, streamed reduced-event caches for direct-geometry runs.

Event chunks remain on disk. The project stores each run as a separate nested
NPZ, with uncompressed float64 blocks for bounded reads. No event arrays
are loaded when cache references are bound on project open.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .array_archive import array_archive_writer
from .data_workspace import bind_data_workspace, record_temporary_file, temporary_data_directory
from .dgs_reduction_policy import (
    DEFAULT_EVENT_PRECISION_POLICY,
    DEFAULT_MONITOR_VARIANCE_POLICY,
    resolved_dgs_reduction_policies,
)
from .project_archive import (
    REDUCED_EVENT_ASSET_ROOT,
    ArchiveMember,
    dataset_artifact_member,
    open_project_artifact,
)
from .raw_dgs_geometry_precision import DGS_HE3_EFFICIENCY_CONVENTION

_CACHE_KEY = "raw_dgs_reduction_cache"
RAW_DGS_REDUCTION_VERSION = 8
_EVENT_ROW_BYTES = 6 * 8
_EVENT_BLOCK_ROWS = (32 * 1024**2 + _EVENT_ROW_BYTES - 1) // _EVENT_ROW_BYTES
_REDUCTION_DEFAULTS = {
    "incident_energy_override": None,
    "t0_override": None,
    "energy_min_fraction": -0.95,
    "energy_max_fraction": 0.95,
    "bad_pulse_threshold": 95.0,
    "ki_kf_normalization": True,
    "he3_detector_efficiency_correction": True,
    "monitor_variance_policy": DEFAULT_MONITOR_VARIANCE_POLICY,
    "event_precision_policy": DEFAULT_EVENT_PRECISION_POLICY,
    "hyspec_default_mask": True,
    "hyspec_tof_crop": True,
    "hyspec_tank_offset_override": None,
}


def _file_signature(path, previous=None):
    if not path:
        return None
    source = Path(path).resolve()
    try:
        stat = source.stat()
    except FileNotFoundError:
        if previous and previous[0] == str(source):
            return previous
        raise
    return [str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def reduction_signature(dataset, config):
    """Identify source/calibration inputs, independently of any output grid."""
    previous = dataset.metadata.get(_CACHE_KEY, {}).get("signature")
    try:
        previous = json.loads(previous) if previous else []
    except (TypeError, ValueError):
        previous = []
    previous_files = previous[1:4] if len(previous) >= 4 else [None] * 3
    settings = {key: [key in config, config.get(key, default)] for key, default in _REDUCTION_DEFAULTS.items()}
    provenance = dataset.metadata.get("resolved_reduction", {}).get("provenance", {})
    instrument = str(provenance.get("instrument_name") or dataset.metadata.get("instrument_name") or "unknown").upper()
    if instrument not in {"UNKNOWN", "HYSPEC", ""}:
        # This setting cannot affect another instrument. Retain the cached
        # representation (including its historical absence), so introducing
        # the HYSPEC checkbox does not invalidate unchanged non-HYSPEC runs.
        old_settings = previous[4] if len(previous) == 6 and isinstance(previous[4], dict) else {}
        if "hyspec_default_mask" in old_settings:
            settings["hyspec_default_mask"] = old_settings["hyspec_default_mask"]
        else:
            settings.pop("hyspec_default_mask", None)
    policies = resolved_dgs_reduction_policies(config)
    for key in ("monitor_variance_policy", "event_precision_policy"):
        settings[key] = policies[key]
    # Only legacy high-precision He-3 corrections changed. Preserve signatures
    # for the unchanged Mantid path and for reductions without this correction.
    settings.update(reduction_convention_signature(config))
    return json.dumps(
        [
            RAW_DGS_REDUCTION_VERSION,
            _file_signature(dataset.metadata["source_file"], previous_files[0]),
            _file_signature(config.get("normalization_file"), previous_files[1]),
            _file_signature(config.get("mask_file"), previous_files[2]),
            settings,
            config.get("kf_ki_normalization", True),
        ],
        sort_keys=True,
    )


def reduction_convention_signature(config):
    """Invalidate only legacy high-precision DGS He-3 reductions and binnings."""

    policies = resolved_dgs_reduction_policies(config)
    if policies["event_precision_policy"] == "high_precision" and config.get(
        "he3_detector_efficiency_correction", True
    ):
        return {"he3_efficiency_convention": DGS_HE3_EFFICIENCY_CONVENTION}
    return {}


@dataclass(frozen=True)
class _ReducedEventCache:
    signature: str
    content: Path | ArchiveMember
    # Own staging files until the project adopts a saved archive reference.
    staging: Any = None
    disk_bytes: int = 0

    def __deepcopy__(self, memo):
        return self

    @contextmanager
    def open(self):
        if isinstance(self.content, ArchiveMember):
            with open_project_artifact(self.content.path, self.content.member) as stream:
                with np.load(stream, allow_pickle=False) as archive:
                    yield archive
        else:
            with np.load(self.content, allow_pickle=False) as archive:
                yield archive


def cached_reduction(dataset, signature):
    cache = getattr(dataset, "_raw_dgs_reduction_cache", None)
    return cache if cache is not None and cache.signature == signature else None


def cache_event_chunks(dataset, signature, header, normalization, chunks):
    """Write and yield chunks; publish a cache only after complete reduction."""
    owner = dataset.metadata.get("_project_path") or dataset.metadata.get("source_file")
    staging = temporary_data_directory(owner, prefix="nfit-reduced-events-")
    path = Path(staging.name) / "events.npz"
    chunk_count = 0
    # Column storage makes the coordinate/weight columns contiguous. Buffering
    # preserves event order and bounds storage without repeated bank headers.
    buffer = np.empty((_EVENT_BLOCK_ROWS, 6), dtype=np.float64, order="F")
    filled = 0
    pending_raw_count = 0
    try:
        with path.open("wb") as stream, array_archive_writer(
            stream, max_member_bytes=buffer.nbytes, compressed=False,
        ) as write:
            for key, value in normalization.items():
                write(key, value)
            for events, raw_count in chunks:
                if not len(events):
                    pending_raw_count += raw_count
                position = 0
                while position < len(events):
                    end = min(len(events), position + _EVENT_BLOCK_ROWS - filled)
                    buffer[filled:filled + end - position] = events[position:end]
                    pending_raw_count += raw_count * end // len(events) - raw_count * position // len(events)
                    filled += end - position
                    position = end
                    if filled == _EVENT_BLOCK_ROWS:
                        write(f"events_{chunk_count}", buffer)
                        write(f"raw_count_{chunk_count}", pending_raw_count)
                        chunk_count += 1
                        filled = pending_raw_count = 0
                yield events, raw_count
            if filled or pending_raw_count:
                write(f"events_{chunk_count}", np.asfortranarray(buffer[:filled]))
                write(f"raw_count_{chunk_count}", pending_raw_count)
                chunk_count += 1
            write("header_json", json.dumps({**header, "chunk_count": chunk_count}))
        size = path.stat().st_size
        record_temporary_file(path, size)
        dataset._raw_dgs_reduction_cache = _ReducedEventCache(signature, path, staging, size)
        dataset.metadata[_CACHE_KEY] = {
            "version": RAW_DGS_REDUCTION_VERSION,
            "signature": signature,
            "member": _cache_member(dataset),
        }
    except BaseException:
        staging.cleanup()
        raise


def iter_cached_event_chunks(archive, max_batch_bytes):
    header = json.loads(str(archive["header_json"].item()))
    rows = max(1, int(max_batch_bytes) // _EVENT_ROW_BYTES)
    for index in range(header["chunk_count"]):
        events = archive[f"events_{index}"]
        raw_count = int(archive[f"raw_count_{index}"].item())
        # Empty chunks still advance the raw-event progress count.
        if not len(events):
            yield events, raw_count
        else:
            for start in range(0, len(events), rows):
                stop = min(start + rows, len(events))
                progress_count = raw_count * stop // len(events) - raw_count * start // len(events)
                yield events[start:stop], progress_count


def _cache_member(dataset):
    # Reuse the archive codec's dataset-ID validation.
    relative = Path(dataset_artifact_member(dataset.id)).parts[-2]
    return str(REDUCED_EVENT_ASSET_ROOT / relative / "events.npz")


def reduced_event_cache_info(dataset):
    """Return cache metadata without reading reduced events, or None."""
    payload = dataset.metadata.get(_CACHE_KEY)
    return dict(payload) if isinstance(payload, dict) else None


def clear_reduced_event_cache(dataset):
    """Discard one run's reduced-event cache; the next binning regenerates it."""
    dataset._raw_dgs_reduction_cache = None
    dataset.metadata.pop(_CACHE_KEY, None)


def project_reduced_event_artifacts(project):
    """Collect cache references; no numerical payloads are materialized."""
    artifacts = {}
    for group in project.data_groups:
        for dataset in group.iter_datasets():
            cache = getattr(dataset, "_raw_dgs_reduction_cache", None)
            if cache is not None:
                member = _cache_member(dataset)
                dataset.metadata[_CACHE_KEY]["member"] = member
                artifacts[member] = cache.content
    return artifacts


def bind_project_reduced_event_caches(project, path):
    """Attach lazy archive references after loading or saving a project."""
    for group in project.data_groups:
        bind_data_workspace(group, path)
        for dataset in group.iter_datasets():
            payload = dataset.metadata.get(_CACHE_KEY)
            if isinstance(payload, dict) and payload.get("version") == RAW_DGS_REDUCTION_VERSION:
                dataset._raw_dgs_reduction_cache = _ReducedEventCache(
                    str(payload["signature"]),
                    ArchiveMember(Path(path), str(payload["member"])),
                )
