"""Benchmark the compiled measured-background replay kernel against NumPy.

The synthetic workload models a 722-angle scan with directional QLab events.
It exercises detector acceptance, per-angle energy coverage, excluded output
bins, correlated copies, varied event weights, unknown detectors, non-finite
coordinates, and exact outer-bin edges.  File I/O, trajectory normalization,
and input/flag preparation are deliberately outside the reported replay time.

Run from the repository root with the project's environment::

    PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
      benchmarks/benchmark_measured_background_replay.py

The first compiled call is reported separately as
``jit_or_cache_warmup_seconds``; it may compile or restore Numba's disk cache.
All compiled timings after that call are warmed-JIT measurements.  The default
10,000 events are processed in the same bounded row chunks as production.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from nfit import _mdevent_background_numba as compiled
from nfit import _parallel
from nfit.mdevent_background import MAX_REPLAY_TRANSFORM_TASKS, _replay_numpy_block


@dataclass(frozen=True)
class Fixture:
    block: np.ndarray
    lab: np.ndarray
    transforms: list[tuple[np.ndarray, np.ndarray]]
    accepted_ids: list[np.ndarray]
    exclusions_numpy: list[np.ndarray | None]
    inverses: np.ndarray
    energy_bounds: np.ndarray
    weights: np.ndarray
    accepted_compiled: object
    exclusions_compiled: object
    detector_indices: np.ndarray
    edges: tuple[np.ndarray, ...]
    shape: tuple[int, int, int, int]


def make_fixture(events: int, transforms: int, seed: int) -> Fixture:
    """Create deterministic directional events spanning many distinct bins."""

    rng = np.random.default_rng(seed)
    shape = (48, 48, 12, 32)
    edges = (
        np.linspace(-3.2, 3.2, shape[0] + 1),
        np.linspace(-3.2, 3.2, shape[1] + 1),
        np.linspace(-1.5, 1.5, shape[2] + 1),
        np.linspace(-12.0, 36.0, shape[3] + 1),
    )
    angles = np.linspace(0.0, 2.0 * np.pi, transforms, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    inverses = np.zeros((transforms, 3, 3), dtype=np.float64)
    inverses[:, 0, 0] = cosine
    inverses[:, 0, 1] = sine
    inverses[:, 1, 0] = -sine
    inverses[:, 1, 1] = cosine
    inverses[:, 2, 2] = 1.0
    lower = -10.0 + 2.0 * np.sin(angles * 3.0)
    upper = 34.0 - 2.0 * np.cos(angles * 5.0)
    energy_bounds = np.column_stack((lower, upper))
    weights = rng.uniform(0.15, 1.85, transforms) / transforms

    detector_ids = 1_000 + 2 * np.arange(96, dtype=np.int64)
    acceptance = []
    accepted_ids = []
    exclusions_numpy: list[np.ndarray | None] = []
    output_size = int(np.prod(shape))
    for index in range(transforms):
        # Rotate a sparse detector outage pattern through the angle sequence.
        flags = ((np.arange(detector_ids.size) + index) % 17 != 0)
        acceptance.append(flags)
        accepted_ids.append(detector_ids[flags])
        if index % 37 == 0:
            excluded = np.zeros(output_size, dtype=bool)
            excluded[(index * 104729) % output_size :: 997] = True
            exclusions_numpy.append(excluded)
        else:
            exclusions_numpy.append(None)

    block = np.zeros((events, 9), dtype=np.float64)
    block[:, 0] = rng.lognormal(mean=0.0, sigma=0.55, size=events)
    block[:, 1] = rng.lognormal(mean=-0.2, sigma=0.7, size=events)
    block[:, 4] = rng.choice(detector_ids, size=events)
    radius = rng.uniform(0.15, 3.0, events)
    azimuth = rng.uniform(-np.pi, np.pi, events)
    block[:, 5] = radius * np.cos(azimuth)
    block[:, 6] = radius * np.sin(azimuth)
    block[:, 7] = rng.uniform(-1.35, 1.35, events)
    block[:, 8] = rng.uniform(-11.5, 35.5, events)
    # Deterministic rejection and boundary cases are part of every normal run.
    if events >= 8:
        block[0, 4] = -999  # unknown detector
        block[1, 5] = np.nan
        block[2, 8] = np.nan
        block[3, 5:8] = (edges[0][0], 0.0, 0.0)
        block[4, 5:8] = (edges[0][-1], 0.0, 0.0)
        block[5, 8] = edges[3][0]
        block[6, 8] = edges[3][-1]
        block[7, 0] = 0.0
    lab = np.ascontiguousarray(block[:, 5:8])
    sorted_ids = np.sort(detector_ids)
    indices = np.searchsorted(sorted_ids, block[:, 4].astype(np.int64))
    known = indices < sorted_ids.size
    known &= sorted_ids[np.minimum(indices, sorted_ids.size - 1)] == block[:, 4]
    indices[~known] = -1
    exclusion_arrays = [
        np.empty(0, dtype=bool) if value is None else value
        for value in exclusions_numpy
    ]
    accepted_compiled, exclusions_compiled = compiled.prepare_replay_flags(
        acceptance, exclusion_arrays, output_size=output_size
    )
    transforms_list = [
        (inverses[index], energy_bounds[index]) for index in range(transforms)
    ]
    return Fixture(
        block, lab, transforms_list, accepted_ids, exclusions_numpy,
        np.ascontiguousarray(inverses), np.ascontiguousarray(energy_bounds),
        np.ascontiguousarray(weights), accepted_compiled, exclusions_compiled,
        np.ascontiguousarray(indices), edges, shape,
    )


def empty_outputs(fixture: Fixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.zeros(int(np.prod(fixture.shape))) for _ in range(3))  # type: ignore[return-value]


def replay(fixture: Fixture, *, backend: str, workers: int) -> tuple[float, tuple[np.ndarray, ...], int]:
    outputs = empty_outputs(fixture)
    rows = max(1, min(100_000, MAX_REPLAY_TRANSFORM_TASKS // len(fixture.transforms)))
    used_workers = 1
    started = perf_counter()
    for start in range(0, fixture.block.shape[0], rows):
        stop = min(start + rows, fixture.block.shape[0])
        if backend == "numpy":
            _replay_numpy_block(
                fixture.block[start:stop], fixture.lab[start:stop], fixture.transforms,
                fixture.accepted_ids, fixture.exclusions_numpy, fixture.edges,
                fixture.shape, fixture.weights, *outputs,
            )
        else:
            used_workers = compiled.accumulate_replayed_events(
                fixture.lab[start:stop], fixture.block[start:stop, 8],
                fixture.detector_indices[start:stop], fixture.block[start:stop, 0],
                fixture.block[start:stop, 1], fixture.inverses, fixture.energy_bounds,
                fixture.weights, fixture.accepted_compiled, fixture.exclusions_compiled,
                fixture.edges, np.asarray(fixture.shape, dtype=np.int64), *outputs,
                workers=workers,
            )
    return perf_counter() - started, outputs, used_workers


def parity(reference: tuple[np.ndarray, ...], candidate: tuple[np.ndarray, ...]) -> dict[str, object]:
    names = ("numerator", "variance", "events")
    checks = {}
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        checks[name] = {
            "exact": bool(np.array_equal(expected, actual)),
            "max_abs_error": float(np.max(np.abs(expected - actual), initial=0.0)),
            "sum": float(actual.sum()),
        }
    checks["allclose"] = bool(all(
        np.allclose(expected, actual, rtol=2e-14, atol=1e-14)
        for expected, actual in zip(reference, candidate, strict=True)
    ))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--transforms", type=int, default=722)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if min(args.events, args.transforms, args.repeats) < 1:
        parser.error("events, transforms, and repeats must be positive")

    preparation_started = perf_counter()
    fixture = make_fixture(args.events, args.transforms, args.seed)
    preparation_seconds = perf_counter() - preparation_started
    rows = max(1, min(100_000, MAX_REPLAY_TRANSFORM_TASKS // args.transforms))
    requested_parallel = _parallel.bounded_worker_count(
        4 if args.parallel_workers is None else args.parallel_workers
    )

    # Compile or restore every relevant specialization with a tiny event slice.
    # Keeping all transforms here ensures the scratch-allocation regime matches.
    warm_fixture = Fixture(
        fixture.block[: min(16, args.events)], fixture.lab[: min(16, args.events)],
        fixture.transforms, fixture.accepted_ids, fixture.exclusions_numpy,
        fixture.inverses, fixture.energy_bounds, fixture.weights,
        fixture.accepted_compiled, fixture.exclusions_compiled,
        fixture.detector_indices[: min(16, args.events)], fixture.edges, fixture.shape,
    )
    warmup_seconds, _, _ = replay(warm_fixture, backend="compiled", workers=1)

    reference_runs = [replay(fixture, backend="numpy", workers=1) for _ in range(args.repeats)]
    single_runs = [replay(fixture, backend="compiled", workers=1) for _ in range(args.repeats)]
    parallel_runs = [
        replay(fixture, backend="compiled", workers=requested_parallel)
        for _ in range(args.repeats)
    ]
    reference_seconds = statistics.median(run[0] for run in reference_runs)
    single_seconds = statistics.median(run[0] for run in single_runs)
    parallel_seconds = statistics.median(run[0] for run in parallel_runs)
    reference = reference_runs[-1][1]
    single = single_runs[-1][1]
    parallel = parallel_runs[-1][1]
    result = {
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "numba": __import__("numba").__version__, "platform": platform.platform(),
            "logical_cpus": _parallel.detect_cpu_budget(),
            "nfit_cpu_limit": _parallel.bounded_worker_count(None),
        },
        "workload": {
            "events": args.events, "transforms": args.transforms,
            "event_transform_tasks": args.events * args.transforms,
            "shape": fixture.shape, "output_bins": int(np.prod(fixture.shape)),
            "production_task_cap": MAX_REPLAY_TRANSFORM_TASKS,
            "rows_per_call": min(rows, args.events),
            "calls_per_run": (args.events + rows - 1) // rows,
            "scratch_bytes_per_task": compiled.REPLAY_SCRATCH_BYTES_PER_TASK,
            "estimated_max_scratch_bytes": (
                min(rows, args.events)
                * args.transforms
                * compiled.REPLAY_SCRATCH_BYTES_PER_TASK
            ),
            "repeats": args.repeats,
            "timing_excludes": ["fixture/flag preparation", "JIT compilation", "I/O", "normalization"],
        },
        "preparation_seconds": preparation_seconds,
        "jit_or_cache_warmup_seconds": warmup_seconds,
        "timings": {
            "numpy_reference_seconds": reference_seconds,
            "compiled_1_cpu_seconds": single_seconds,
            "compiled_parallel_seconds": parallel_seconds,
            "compiled_parallel_workers": parallel_runs[-1][2],
            "compiled_1_cpu_speedup_vs_numpy": reference_seconds / single_seconds,
            "compiled_parallel_speedup_vs_numpy": reference_seconds / parallel_seconds,
            "parallel_speedup_vs_compiled_1_cpu": single_seconds / parallel_seconds,
        },
        "parity": {
            "compiled_1_cpu": parity(reference, single),
            "compiled_parallel": parity(reference, parallel),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["parity"]["compiled_1_cpu"]["allclose"] or not result["parity"]["compiled_parallel"]["allclose"]:
        raise SystemExit("compiled replay did not match the NumPy reference")


if __name__ == "__main__":
    main()
