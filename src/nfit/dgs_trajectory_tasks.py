"""Pool literally identical detector trajectories before normalization batches.

This service changes only repeated exposure work. Event numerators, variances,
counts and source membership are not inputs. Callers must first check complete
detector identity, direction, normalization and mask payloads for equality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _task_key(payload, endpoints):
    try:
        matrix = np.asarray(payload[0])
        energy, charge = float(payload[1]), float(payload[3])
        bounds = np.asarray(payload[2])
        if (
            matrix.shape != (3, 3)
            or matrix.dtype.kind != "f"
            or not np.all(np.isfinite(matrix))
            or bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or not math.isfinite(energy)
            or energy <= 0.0
            or not math.isfinite(charge)
            or charge <= 0.0
        ):
            return None
        # A conservative energy-domain key suffices: equal Ei and these exact
        # endpoints imply equal compiled requested-box wavevector clipping.
        # Leave square-root arithmetic and further saturated equivalences to
        # the authoritative kernel rather than reproducing them here.
        low = max(float(np.min(bounds)), endpoints[0])
        high = min(float(np.max(bounds)), endpoints[1])
        if not math.isfinite(low) or not math.isfinite(high):
            return None
        return (
            matrix.dtype.str,
            matrix.tobytes(),
            energy.hex(),
            float(low).hex(),
            float(high).hex(),
        )
    except (IndexError, TypeError, ValueError, OverflowError):
        # Leave malformed inputs to the existing reducer's validation path.
        return None


def pool_trajectory_tasks(
    payloads: Sequence[tuple[Any, ...]],
    energy_edges,
    *,
    checked_shared_geometry: bool = False,
) -> list[tuple[Any, ...]]:
    """Pool positive charges for equal compiled affine/Ei/energy-window tasks.

    Energies are in meV. Rows begin with
    ``(inverse_matrix, Ei, energy_bounds, charge)`` and may carry
    caller-owned trailing values. Preserve the first row's bounds and trailing
    values, and first-occurrence task order. Call separately for each completely
    checked detector geometry; a mixed or unchecked collection is unchanged.

    Use only in the compiled requested-box-clipping path; the legacy Python
    fallback does not use the same preclipping. Only the sums of positive finite
    charges change arithmetic order. Exposure
    is linear in charge, so equality is to floating-point accumulation precision,
    rather than a promise of bitwise identical final normalization.
    """
    original = list(payloads)
    if not checked_shared_geometry or len(original) < 2:
        return original
    edges = np.asarray(energy_edges)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        return original
    endpoints = (float(edges[0]), float(edges[-1]))
    groups: dict[tuple, list[int]] = {}
    for index, payload in enumerate(original):
        key = _task_key(payload, endpoints)
        if key is not None:
            groups.setdefault(key, []).append(index)
    replacements = {}
    consumed = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        try:
            charge = math.fsum(float(original[index][3]) for index in indices)
        except OverflowError:
            continue
        if not math.isfinite(charge):
            continue
        first = indices[0]
        row = original[first]
        replacements[first] = (*row[:3], charge, *row[4:])
        consumed.update(indices[1:])
    return [replacements.get(index, row) for index, row in enumerate(original)
            if index not in consumed]
