"""Exploratory detector-trajectory accumulator lifecycles.

This module compares production's persistent worker buffers with the previous
per-batch lifecycle and experimental slab ownership. The previous lifecycle
is reconstructed from the shared geometry kernel, so numerical trajectory
rules have one source.

The helpers accept the same argument order as
``nfit._mdevent_numba.trajectory_normalization``.  A batch is that argument
tuple with its four run arrays (positions 3 through 6) already sliced.
"""

from __future__ import annotations

import inspect
import math
import textwrap
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import get_num_threads, njit, set_num_threads

from nfit import _mdevent_numba


@dataclass(frozen=True)
class DensePlan:
    """Memory and traffic implied by one dense-private accumulator plan."""

    output_bins: int
    workers: int
    batches: int

    @property
    def grid_bytes(self) -> int:
        return self.output_bins * np.dtype(np.float64).itemsize

    @property
    def partial_bytes(self) -> int:
        return self.workers * self.grid_bytes

    @property
    def baseline_zero_reduce_bytes(self) -> int:
        # Logical bytes swept; actual physical traffic depends on allocation
        # and operating-system zero-page behavior.
        return 2 * self.batches * self.partial_bytes

    @property
    def persistent_zero_reduce_bytes(self) -> int:
        return 2 * self.partial_bytes


def bounded_dense_workers(
    output_bins: int,
    *,
    cpu_limit: int,
    workspace_bytes: int,
    live_dense_grids: int = 5,
    reduction_grids: int = 1,
) -> int:
    """Plan workers after reserving live and final-reduction dense grids.

    ``live_dense_grids=5`` models data, variance, event count, the Python result
    accumulator, and one returned/reduction grid. This analytical helper can
    compare reservations; production now reserves its four other live grids.
    """

    grid = max(1, int(output_bins) * np.dtype(np.float64).itemsize)
    reserved = max(0, int(live_dense_grids) + int(reduction_grids)) * grid
    partial_budget = max(0, int(workspace_bytes) - reserved)
    return max(1, min(max(1, int(cpu_limit)), partial_budget // grid))


def _persistent_kernel():
    """Compile the production geometry kernel with caller-owned work arrays."""

    return _mdevent_numba.accumulate_trajectory_normalization


_TRAJECTORY_ACCUMULATE = _persistent_kernel()


@njit(nogil=True, parallel=True)
def _eager_partial(workers: int, output_size: int) -> np.ndarray:
    """Allocate private grids through Numba's eager zeroing path."""

    return np.zeros((workers, output_size), dtype=np.float64)


def _per_batch_baseline_kernel():
    """Reconstruct the v0.97 per-call allocation from shared geometry source."""

    dispatcher = _mdevent_numba.accumulate_trajectory_normalization
    source = textwrap.dedent(inspect.getsource(dispatcher.py_func))
    source = source[source.index("def accumulate_trajectory_normalization") :]
    source = source.replace(
        "def accumulate_trajectory_normalization(\n    partial, scratch, ",
        "def trajectory_normalization_per_batch(\n    ",
        1,
    )
    body_start = "):\n    detectors = theta.size\n"
    if body_start not in source:
        raise RuntimeError(
            "production trajectory signature changed; update the baseline transform"
        )
    allocations = (
        "):\n"
        "    workers = get_num_threads()\n"
        "    output_size = shape[0] * shape[1] * shape[2] * shape[3]\n"
        "    partial = np.zeros((workers, output_size), dtype=np.float64)\n"
        "    max_intersections = edge0.size + edge1.size + edge2.size + edge3.size + 2\n"
        "    scratch = np.empty((workers, max_intersections), dtype=np.float64)\n"
        "    detectors = theta.size\n"
    )
    source = source.replace(body_start, allocations, 1)
    source += "    return np.sum(partial, axis=0)\n"
    namespace = {
        "np": np,
        "get_num_threads": _mdevent_numba.get_num_threads,
        "get_thread_id": _mdevent_numba.get_thread_id,
        "prange": _mdevent_numba.prange,
        "ENERGY_TO_K2": _mdevent_numba.ENERGY_TO_K2,
    }
    exec(compile(source, "<trajectory-v097-baseline>", "exec"), namespace)
    return njit(fastmath=False, nogil=True, parallel=True)(
        namespace["trajectory_normalization_per_batch"]
    )


_TRAJECTORY_PER_BATCH = _per_batch_baseline_kernel()


def _serial_slab_kernel():
    """Compile the production kernel for one caller-owned axis-0 slab."""

    dispatcher = _mdevent_numba.accumulate_trajectory_normalization
    source = textwrap.dedent(inspect.getsource(dispatcher.py_func))
    source = source[source.index("def accumulate_trajectory_normalization") :]
    source = source.replace(
        "def accumulate_trajectory_normalization(\n",
        "def trajectory_normalization_slab(\n    owns_upper_edge,\n",
        1,
    )
    source = source.replace("    for task in prange(total):\n", "    for task in range(total):\n", 1)
    source = source.replace("        thread = get_thread_id()\n", "        thread = 0\n", 1)
    boundary_assignment = "            if coordinate0 == edge0[-1]:\n"
    if boundary_assignment not in source:
        raise RuntimeError(
            "production trajectory boundary handling changed; update slab ownership"
        )
    source = source.replace(
        boundary_assignment,
        "            if coordinate0 == edge0[-1] and not owns_upper_edge:\n"
        "                continue\n"
        + boundary_assignment,
        1,
    )
    namespace = {
        "np": np,
        "ENERGY_TO_K2": _mdevent_numba.ENERGY_TO_K2,
    }
    exec(compile(source, "<trajectory-axis0-slab-candidate>", "exec"), namespace)
    return njit(fastmath=False, nogil=True, parallel=False)(
        namespace["trajectory_normalization_slab"]
    )


_TRAJECTORY_SLAB = _serial_slab_kernel()


def _workspace_shape(args: Sequence[np.ndarray], workers: int) -> tuple[int, int]:
    edge0, edge1, edge2, edge3, shape = args[7:12]
    output_size = math.prod(int(value) for value in shape)
    intersections = edge0.size + edge1.size + edge2.size + edge3.size + 2
    return output_size, intersections


def run_persistent_dense(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> np.ndarray:
    """Accumulate all batches into persistent private grids, then reduce once."""

    batches = list(batches)
    if not batches:
        raise ValueError("at least one trajectory batch is required")
    selected = max(1, min(int(workers), get_num_threads()))
    output_size, intersections = _workspace_shape(batches[0], selected)
    partial = np.zeros((selected, output_size), dtype=np.float64)
    scratch = np.empty((selected, intersections), dtype=np.float64)
    previous = get_num_threads()
    set_num_threads(selected)
    try:
        for args in batches:
            current_size, current_intersections = _workspace_shape(args, selected)
            if (current_size, current_intersections) != (output_size, intersections):
                raise ValueError("all batches must use the same output grid")
            _TRAJECTORY_ACCUMULATE(partial, scratch, *args)
    finally:
        set_num_threads(previous)
    return np.sum(partial, axis=0)


def run_persistent_eager(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> np.ndarray:
    """Accumulate persistently after eagerly zeroing private grids in Numba."""

    batches = list(batches)
    if not batches:
        raise ValueError("at least one trajectory batch is required")
    selected = max(1, min(int(workers), get_num_threads()))
    output_size, intersections = _workspace_shape(batches[0], selected)
    previous = get_num_threads()
    set_num_threads(selected)
    try:
        partial = _eager_partial(selected, output_size)
        scratch = np.empty((selected, intersections), dtype=np.float64)
        for args in batches:
            current_size, current_intersections = _workspace_shape(args, selected)
            if (current_size, current_intersections) != (output_size, intersections):
                raise ValueError("all batches must use the same output grid")
            _TRAJECTORY_ACCUMULATE(partial, scratch, *args)
    finally:
        set_num_threads(previous)
    return np.sum(partial, axis=0)


def run_production(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> np.ndarray:
    """Exercise the actual accumulator, including its automatic allocation policy."""

    batches = list(batches)
    if not batches:
        raise ValueError("at least one trajectory batch is required")
    from nfit.mdevent import _trajectory_eager_partial

    selected = max(1, min(int(workers), get_num_threads()))
    output_size, _ = _workspace_shape(batches[0], selected)
    accumulator = _mdevent_numba.trajectory_normalization_accumulator(
        *batches[0][7:12], workers=selected,
        eager=_trajectory_eager_partial(output_size, selected),
    )
    for args in batches:
        accumulator.accumulate(*args)
    return accumulator.result()


def run_baseline_dense(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> np.ndarray:
    """Run the unchanged production allocation/reduction lifecycle per batch."""

    result = None
    previous = get_num_threads()
    set_num_threads(max(1, min(int(workers), previous)))
    try:
        for args in batches:
            current = _TRAJECTORY_PER_BATCH(*args)
            if result is None:
                result = np.array(current, copy=True)
            else:
                result += current
    finally:
        set_num_threads(previous)
    if result is None:
        raise ValueError("at least one trajectory batch is required")
    return result


def _axis0_ranges(axis_bins: int, workers: int) -> list[tuple[int, int]]:
    count = min(max(1, int(workers)), max(1, int(axis_bins)))
    return [
        (axis_bins * index // count, axis_bins * (index + 1) // count)
        for index in range(count)
    ]


def run_axis0_slabs(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> np.ndarray:
    """Replay trajectories over disjoint axis-0 slabs in native threads.

    Each worker owns one contiguous first-axis view of the single global output.
    The slab's edge array clips trajectories before internal boundary scanning.
    This removes dense worker-private grids at the cost of replaying geometry
    once per slab.
    """

    batches = [tuple(args) for args in batches]
    if not batches:
        raise ValueError("at least one trajectory batch is required")
    shape = np.asarray(batches[0][11], dtype=np.int64)
    if shape.shape != (4,):
        raise ValueError("trajectory output shape must have four dimensions")
    axis0_edges = np.asarray(batches[0][7], dtype=np.float64)
    output = np.zeros(tuple(int(value) for value in shape), dtype=np.float64)
    ranges = _axis0_ranges(int(shape[0]), workers)

    def accumulate_slab(bounds: tuple[int, int]) -> None:
        start, stop = bounds
        local_shape = shape.copy()
        local_shape[0] = stop - start
        local_edges = axis0_edges[start : stop + 1]
        local_output = output[start:stop].reshape(1, -1)
        intersections = sum(
            int(np.asarray(value).size)
            for value in (local_edges, *batches[0][8:11])
        ) + 2
        scratch = np.empty((1, intersections), dtype=np.float64)
        for original in batches:
            if not np.array_equal(original[7], axis0_edges):
                raise ValueError("all batches must use the same axis-0 edges")
            local = list(original)
            local[7] = local_edges
            local[11] = local_shape
            _TRAJECTORY_SLAB(stop == int(shape[0]), local_output, scratch, *local)

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        list(executor.map(accumulate_slab, ranges))
    return output.reshape(-1)


def compare_dense_lifecycles(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> dict[str, float | int | bool]:
    """Time both lifecycles and verify numerical equivalence."""

    batches = list(batches)
    started = time.perf_counter()
    baseline = run_baseline_dense(batches, workers=workers)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    persistent = run_persistent_dense(batches, workers=workers)
    persistent_seconds = time.perf_counter() - started
    output_size, _ = _workspace_shape(batches[0], workers)
    plan = DensePlan(output_size, workers, len(batches))
    return {
        "output_bins": output_size,
        "workers": workers,
        "batches": len(batches),
        "baseline_seconds": baseline_seconds,
        "persistent_seconds": persistent_seconds,
        "speedup": baseline_seconds / max(persistent_seconds, np.finfo(float).tiny),
        "equivalent": bool(np.allclose(baseline, persistent, rtol=2e-13, atol=0.0)),
        "max_abs_difference": float(np.max(np.abs(baseline - persistent))),
        "partial_bytes": plan.partial_bytes,
        "baseline_minimum_zero_reduce_bytes": plan.baseline_zero_reduce_bytes,
        "persistent_minimum_zero_reduce_bytes": plan.persistent_zero_reduce_bytes,
    }


def compare_axis0_slabs(
    batches: Iterable[Sequence[np.ndarray]], *, workers: int
) -> dict[str, float | int | bool]:
    """Compare slab ownership with the unchanged dense baseline."""

    batches = list(batches)
    started = time.perf_counter()
    baseline = run_baseline_dense(batches, workers=workers)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    slabbed = run_axis0_slabs(batches, workers=workers)
    slab_seconds = time.perf_counter() - started
    difference = np.abs(baseline - slabbed)
    return {
        "output_bins": int(baseline.size),
        "workers": workers,
        "batches": len(batches),
        "slabs": len(_axis0_ranges(int(np.asarray(batches[0][11])[0]), workers)),
        "baseline_seconds": baseline_seconds,
        "slab_seconds": slab_seconds,
        "speedup": baseline_seconds / max(slab_seconds, np.finfo(float).tiny),
        "equivalent": bool(np.allclose(baseline, slabbed, rtol=2e-13, atol=0.0)),
        "max_abs_difference": float(np.max(difference)),
        "different_bins": int(np.count_nonzero(difference)),
        "slab_output_bytes": int(baseline.nbytes),
    }


def slab_owner_cost(output_bins: int, workers: int, batches: int) -> dict[str, int]:
    """Model a one-output-grid slab-owner candidate before implementing it.

    A slab owner removes worker-private grids, but each owner must currently
    replay every trajectory because a trajectory can cross multiple flat-index
    slabs.  The model makes that severe compute tradeoff explicit.
    """

    return {
        "dense_bytes": int(output_bins) * np.dtype(np.float64).itemsize,
        "trajectory_replay_factor": max(1, int(workers)),
        "kernel_passes": max(1, int(workers)) * max(1, int(batches)),
    }
