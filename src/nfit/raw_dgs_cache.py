"""Dataset-owned, streamed reduced-event caches for direct-geometry runs.

Event chunks remain on disk. The project stores each run as a separate nested
NPZ, with uncompressed float64 arrays for fast, bounded reads. No event arrays
are loaded when cache references are bound on project open.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .project_archive import (
    REDUCED_EVENT_ASSET_ROOT,
    ArchiveMember,
    dataset_artifact_member,
    open_project_artifact,
)

_CACHE_KEY = "raw_dgs_reduction_cache"
_CACHE_VERSION = 1
_REDUCTION_DEFAULTS = {
    "incident_energy_override": None,
    "t0_override": None,
    "energy_min_fraction": -0.95,
    "energy_max_fraction": 0.95,
    "bad_pulse_threshold": 95.0,
    "ki_kf_normalization": True,
    "he3_detector_efficiency_correction": True,
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
    return json.dumps(
        [
            _CACHE_VERSION,
            _file_signature(dataset.metadata["source_file"], previous_files[0]),
            _file_signature(config.get("normalization_file"), previous_files[1]),
            _file_signature(config.get("mask_file"), previous_files[2]),
            {key: [key in config, config.get(key, default)] for key, default in _REDUCTION_DEFAULTS.items()},
            config.get("kf_ki_normalization", True),
        ],
        sort_keys=True,
    )


@dataclass(frozen=True)
class _ReducedEventCache:
    signature: str
    content: Path | ArchiveMember
    # Own staging files until the project adopts a saved archive reference.
    staging: Any = None

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


def _write_array(archive, name, value):
    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def cache_event_chunks(dataset, signature, header, normalization, chunks):
    """Write and yield chunks; publish a cache only after complete reduction."""
    staging = tempfile.TemporaryDirectory(prefix="nfit-reduced-events-")
    path = Path(staging.name) / "events.npz"
    chunk_count = 0
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for key, value in normalization.items():
                _write_array(archive, key, value)
            for events, raw_count in chunks:
                _write_array(archive, f"events_{chunk_count}", events)
                _write_array(archive, f"raw_count_{chunk_count}", raw_count)
                chunk_count += 1
                yield events, raw_count
            _write_array(archive, "header_json", json.dumps({**header, "chunk_count": chunk_count}))
        dataset._raw_dgs_reduction_cache = _ReducedEventCache(signature, path, staging)
        dataset.metadata[_CACHE_KEY] = {
            "version": _CACHE_VERSION,
            "signature": signature,
            "member": _cache_member(dataset),
        }
    except BaseException:
        staging.cleanup()
        raise


def iter_cached_event_chunks(archive, max_batch_bytes):
    header = json.loads(str(archive["header_json"].item()))
    rows = max(1, int(max_batch_bytes) // (5 * 8))
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
        for dataset in group.iter_datasets():
            payload = dataset.metadata.get(_CACHE_KEY)
            if isinstance(payload, dict) and payload.get("version") == _CACHE_VERSION:
                dataset._raw_dgs_reduction_cache = _ReducedEventCache(
                    str(payload["signature"]),
                    ArchiveMember(Path(path), str(payload["member"])),
                )
