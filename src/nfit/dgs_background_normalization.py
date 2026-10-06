"""Bounded exposure groups for measured-background trajectory integration.

Each event source keeps its original reconstruction and covariance recipe.
Only positive exposure work is pooled, after literal detector payload checks.
The supplied numerical reducer owns clipping, task pooling and integration.
"""

from __future__ import annotations

import hashlib

import numpy as np

MAX_REPLAY_NORMALIZATION_BATCH_TASKS = 250_000
NORMALIZATION_PROGRESS_UNITS_PER_ANGLE = 1_000


def _array_identity(array):
    values = np.asarray(array)
    return values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest()


def _same_array(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.array_equal(first, second)
    )


class ReplayExposureBatch:
    """Collect source/angle exposure recipes without retaining event arrays.

    Hashes select candidate matches; complete arrays confirm identity. Distinct
    masks and detector geometries remain distinct. Flush between sources when
    the estimated recipe storage reaches its allowance, bounding additional
    storage by the allowance plus one source's existing preparation payload.
    """

    def __init__(self, max_batch_bytes):
        self.limit_bytes = max(1, min(32 * 1024**2, int(max_batch_bytes) // 8))
        self.pending_bytes = 0
        self.completed_contributions = 0
        self.groups = {}

    @property
    def full(self):
        return self.pending_bytes >= self.limit_bytes

    def add(self, detectors, payloads, excluded):
        mask_key = None if excluded is None else _array_identity(excluded)
        candidates = self.groups.setdefault(mask_key, [])
        group = next(
            (g for g in candidates if excluded is None or _same_array(excluded, g["excluded"])),
            None,
        )
        if group is None:
            group = dict(
                excluded=excluded, detectors=[], payloads=[], identities={}, contributions=0
            )
            candidates.append(group)
        indices = []
        for detector in detectors:
            identity = tuple(_array_identity(array) for array in detector)
            matches = group["identities"].setdefault(identity, [])
            index = next(
                (
                    i
                    for i in matches
                    if all(
                        _same_array(a, b)
                        for a, b in zip(detector, group["detectors"][i], strict=True)
                    )
                ),
                None,
            )
            if index is None:
                index = len(group["detectors"])
                group["detectors"].append(detector)
                matches.append(index)
                self.pending_bytes += sum(np.asarray(a).nbytes for a in detector)
            indices.append(index)
        group["payloads"].extend((*p[:4], indices[p[4]]) for p in payloads)
        group["contributions"] += 1
        self.pending_bytes += 256 * len(payloads)

    def integrate(self, output, normalize, edges, shape, *, sources, angles, progress_callback):
        """Add pooled exposure to the existing denominator, then release recipes."""
        total = sources * angles * NORMALIZATION_PROGRESS_UNITS_PER_ANGLE
        for candidates in self.groups.values():
            for group in candidates:
                completed = self.completed_contributions
                contributions = group["contributions"]

                def report(event, *, completed=completed, contributions=contributions):
                    if progress_callback is None:
                        return
                    fraction = min(
                        1.0, max(0.0, event.get("iteration", 0) / max(1, event.get("total", 0)))
                    )
                    units = completed + fraction * contributions
                    run = min(sources, max(1, int(np.ceil(units / max(1, angles)))))
                    angle = min(angles, max(1, int(np.ceil(units - (run - 1) * angles))))
                    progress_callback(
                        {
                            **event,
                            "stage": "mdevent_background_normalization",
                            "iteration": int(round(units * NORMALIZATION_PROGRESS_UNITS_PER_ANGLE)),
                            "total": total,
                            "background_run": run,
                            "background_runs_total": sources,
                            "sample_angle": angle,
                            "sample_angles_total": angles,
                            "pooled_background_exposure": True,
                            "message": f"normalizing pooled trajectories for {sources} background runs: "
                            f"{event.get('message', 'integrating detector trajectories')}",
                        }
                    )

                norm = np.asarray(
                    normalize(
                        group["detectors"],
                        group["payloads"],
                        edges,
                        shape,
                        progress_callback=report,
                        max_batch_tasks=MAX_REPLAY_NORMALIZATION_BATCH_TASKS,
                    )
                ).ravel()
                if group["excluded"] is None:
                    output += norm
                else:
                    included = ~group["excluded"]
                    output[included] += norm[included]
                self.completed_contributions += contributions
        self.groups.clear()
        self.pending_bytes = 0
