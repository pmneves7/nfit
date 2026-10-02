"""Bounded source identity and conservative reuse checks for tracked imports.

These records describe acquisition overlap, not a numerical covariance matrix.
They let newly tracked source aliases fail closed when a consumer would treat
them as independent. Unmarked legacy datasets retain their existing behavior.
No project containers, source readers or GUI modules are needed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

from .measurement_dependencies import SourceReplayRequired

SOURCE_SELECTION_LINEAGE_KEY = "source_selection_lineage"
SOURCE_LINEAGE_KEY = "source_lineage"
MAX_TRACKED_SOURCE_IDS = 100_000
MAX_TRACKED_SOURCE_BYTES = 16 * 1024**2


def source_identity(path, *, experiment_index=None, importer=None, stream=None) -> str:
    """Identify a physical file/experiment or a registered independent stream."""
    identity = str(Path(path).expanduser().resolve())
    if experiment_index is not None:
        identity += f"#experiment={experiment_index}"
    if stream:
        identity += f"#importer={quote(str(importer or ''), safe='')}&stream={quote(str(stream), safe='')}"
    return identity


def _metadata(source):
    metadata = source if isinstance(source, Mapping) else getattr(source, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("tracked source metadata must be a mapping")
    return metadata


def _bounded_unique(values):
    result = []
    seen = set()
    byte_count = 0
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("tracked source identities must be nonempty strings")
        if value in seen:
            continue
        byte_count += len(value.encode("utf-8"))
        if len(result) >= MAX_TRACKED_SOURCE_IDS or byte_count > MAX_TRACKED_SOURCE_BYTES:
            raise SourceReplayRequired("Tracked source identities exceed their metadata budget; prepare a smaller source selection")
        seen.add(value)
        result.append(value)
    return tuple(result)


def tracked_source_ids(source) -> tuple[str, ...]:
    """Read explicit identities only; no identity is inferred for legacy data."""
    metadata = _metadata(source)
    values = ()
    aggregate = metadata.get(SOURCE_LINEAGE_KEY)
    if aggregate is not None:
        if not isinstance(aggregate, Mapping) or aggregate.get("version") != 1:
            raise ValueError("unsupported tracked source lineage record")
        identities = aggregate.get("source_ids")
        if not isinstance(identities, (list, tuple)):
            raise ValueError("tracked source lineage requires a list of source identities")
        values = identities
    single = metadata.get(SOURCE_SELECTION_LINEAGE_KEY)
    if single is not None:
        if not isinstance(single, Mapping) or single.get("version") != 1:
            raise ValueError("unsupported source selection lineage record")
        identity = single.get("source_identity")
        from itertools import chain

        values = chain(values, (identity,))
    return _bounded_unique(values)


def _payload(identities):
    return {SOURCE_LINEAGE_KEY: {"version": 1, "source_ids": list(identities),
                                "overlap_policy": "conservative_acquisition_overlap"}} if identities else {}


def merge_source_lineage_metadata(*sources) -> dict:
    """Union tracked acquisition identities for arithmetic and derived data."""
    return _payload(_bounded_unique(identity for source in sources for identity in tracked_source_ids(source)))


def source_lineage_metadata(datasets, *, include_disabled=False) -> dict:
    """Collect marked input identities without I/O.

    Preparation services skip disabled entries by default. Reducers can pass
    ``include_disabled=True`` when the supplied list is the exact set they
    processed. Zero weights are retained conservatively because some source
    adapters do not interpret them as absence of acquisition data.
    """
    return _payload(_bounded_unique(identity for dataset in datasets
                                    if include_disabled or getattr(dataset, "enabled", True)
                                    for identity in tracked_source_ids(dataset)))


def with_source_lineage(data, source_metadata):
    """Attach entry context to immutable data while retaining its statistics."""
    lineage = merge_source_lineage_metadata(data, source_metadata)
    if not lineage:
        return data
    metadata = {**data.metadata, **lineage}
    return data.with_updates(metadata=metadata)


def validate_source_selection_combination(datasets) -> None:
    """Reject independent coaddition of tracked aliases of one acquisition."""
    seen = set()
    for dataset in datasets:
        if not getattr(dataset, "enabled", True) :
            continue
        identities = set(tracked_source_ids(dataset))
        overlap = seen & identities
        if overlap:
            identity = min(overlap)
            raise SourceReplayRequired(
                f"Selected groups reuse {identity!r}. Select a group individually or "
                "combine unique original sources; merging these aliases requires "
                "a path that propagates shared-source covariance."
            )
        seen.update(identities)


def validate_fit_source_lineage(datasets) -> None:
    """Reject tracked acquisition overlap between separate fitted data blocks.

    Disjoint selections of a tracked acquisition are conservatively rejected;
    an acquisition record alone cannot prove independent primitive counts or
    calibration. Separate GLS blocks do not provide joint cross-block covariance.
    """
    seen = set()
    for name, points in datasets:
        identities = set(tracked_source_ids(points))
        overlap = seen & identities
        if overlap:
            identity = min(overlap)
            raise SourceReplayRequired(
                f"Dataset {name!r} reuses tracked source {identity!r} from another fit dataset. "
                "Fit one selection or unique original sources; joint shared-source covariance "
                "across these fit blocks is not represented."
            )
        seen.update(identities)


def validate_independent_source_lineage(*sources) -> None:
    """Require disjoint tracked acquisitions for independent-variance arithmetic."""
    seen = set()
    for source in sources:
        identities = set(tracked_source_ids(source))
        if seen & identities:
            raise SourceReplayRequired(
                "Independent input variances do not represent overlapping tracked sources. "
                "Use unique original sources or a branch that propagates their joint covariance."
            )
        seen.update(identities)
