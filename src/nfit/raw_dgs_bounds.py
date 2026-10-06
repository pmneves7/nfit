"""Bounded physical-coordinate extents of native direct-geometry events."""

from collections import OrderedDict

import numpy as np

from .dgs_background_sources import reduced_event_stream
from .raw_dgs_cache import reduction_signature
from .reduction_recipes import effective_reduction_config

_BOUNDS_CACHE = OrderedDict()
_MAX_BOUNDS_CACHE_ENTRIES = 16


def raw_dgs_coordinate_bounds(
    group,
    *,
    datasets=None,
    basis=None,
    symmetry_operations=None,
    max_batch_bytes=192 * 1024 * 1024,
    powder=False,
    progress_callback=None,
):
    """Return selected reduced-event extents in r.l.u./meV or Å⁻¹/meV.

    Reduce fresh sources or reuse their per-run caches, keeping only bounded
    chunks in memory. Bounds are physical coordinates before histogram-edge
    rounding. The small bounds cache includes source/calibration identity,
    effective reduction settings, UB, output basis and symmetry.
    """
    from .raw_dgs import _goniometer, _symmetry_matrices

    if group.metadata.get("raw_dgs", {}).get("format") != "raw-direct-geometry-nexus":
        raise ValueError("coordinate bounds require a native direct-geometry group")
    selected = [
        d
        for d in (group.datasets if datasets is None else datasets)
        if d.enabled and d.fit_weight > 0
    ]
    if not selected:
        raise ValueError("coordinate bounds require selected native DGS runs")
    basis = np.eye(4) if basis is None else np.asarray(basis, float)
    if basis.shape != (4, 4) or not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 4:
        raise ValueError("native event bounds require a finite invertible 4D basis")
    operations = _symmetry_matrices(symmetry_operations)
    inverse = np.linalg.inv(basis)
    settings = [effective_reduction_config(group, d) for d in selected]
    key = (
        tuple(
            (reduction_signature(d, c), tuple(np.asarray(c["ub_matrix"]).ravel()))
            for d, c in zip(selected, settings, strict=True)
        ),
        tuple(basis.ravel()),
        tuple(tuple(op.ravel()) for op in operations),
        bool(powder),
    )
    if key in _BOUNDS_CACHE:
        _BOUNDS_CACHE.move_to_end(key)
        return list(_BOUNDS_CACHE[key])
    lower, upper = np.full(2 if powder else 4, np.inf), np.full(2 if powder else 4, -np.inf)
    rows = max(1, min(1_000_000, int(max_batch_bytes) // 192))
    total = sum(int(d.metadata["event_count"]) for d in selected)
    completed = 0
    collection_name = str(group.name)
    message = (
        "Finding native DGS event coordinate bounds\n"
        f"{collection_name} · {len(selected):,} run{'s' if len(selected) != 1 else ''}"
    )
    for dataset in selected:
        with reduced_event_stream(group, dataset, rows=rows) as (config, info, _norm, chunks):
            gonio = _goniometer(info.omega, info.phi, info.chi)
            hkl_transform = np.linalg.inv(2 * np.pi * np.asarray(config["ub_matrix"])).T
            for events, raw_count in chunks:
                if len(events):
                    q_lab, energy = events[:, :3], events[:, 3]
                    if powder:
                        coordinates = [np.column_stack((np.linalg.norm(q_lab, axis=1), energy))]
                    else:
                        hkl = (q_lab @ gonio) @ hkl_transform
                        coordinates = (
                            np.column_stack((hkl @ op.T, energy)) @ inverse for op in operations
                        )
                    for values in coordinates:
                        finite = values[np.all(np.isfinite(values), axis=1)]
                        if len(finite):
                            lower = np.minimum(lower, finite.min(axis=0))
                            upper = np.maximum(upper, finite.max(axis=0))
                completed += raw_count
                if progress_callback is not None:
                    progress_callback(
                        dict(
                            stage="raw_dgs_coordinate_bounds",
                            iteration=completed,
                            total=total,
                            message=message,
                            source_collection_id=group.id,
                            source_collection_name=collection_name,
                            source_run_count=len(selected),
                        )
                    )
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("no finite reduced events are available for automatic native bounds")
    result = tuple(zip(lower.tolist(), upper.tolist(), strict=True))
    _BOUNDS_CACHE[key] = result
    while len(_BOUNDS_CACHE) > _MAX_BOUNDS_CACHE_ENTRIES:
        _BOUNDS_CACHE.popitem(last=False)
    return list(result)
