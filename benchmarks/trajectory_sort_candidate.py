"""Benchmark the detector-trajectory intersection sort in isolation.

This deliberately uses the production grid density but does not allocate the
four-dimensional output histogram.  It measures only intersection generation
and ordering, which keeps the benchmark small enough to run on a laptop.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit


@njit(cache=True)
def _fill_intersections(edges, trajectory: int, scratch):
    """Fill one deterministic, finite, production-density crossing list."""
    low = 0.35 + (trajectory % 19) * 0.004
    high = 2.85 - (trajectory % 23) * 0.003
    count = 2
    scratch[0] = low
    scratch[1] = high
    # Affine transforms model (qin - boundary) / qout for three Q axes.
    for dim in range(3):
        qout = (1.35 + 0.17 * dim) * (-1.0 if (trajectory + dim) & 1 else 1.0)
        qin = -0.8 + ((trajectory * (17 + dim * 6)) % 101) / 100.0 * 1.6
        for boundary in edges[dim]:
            value = (qin - boundary) / qout
            if low < value < high:
                scratch[count] = value
                count += 1
    # Monotone nonlinear values model sqrt((Ei - DeltaE) / constant).
    phase = (trajectory % 29) * 0.001
    for boundary in edges[3]:
        value = 0.45 + 2.25 * np.sqrt(boundary) + phase
        if low < value < high:
            scratch[count] = value
            count += 1
    return count


@njit(cache=True)
def _insertion_benchmark(edges, trajectories: int):
    scratch = np.empty(
        edges[0].size + edges[1].size + edges[2].size + edges[3].size + 2,
        dtype=np.float64,
    )
    checksum = 0.0
    count_sum = 0
    maximum = 0
    for trajectory in range(trajectories):
        count = _fill_intersections(edges, trajectory, scratch)
        for i in range(1, count):
            value = scratch[i]
            j = i - 1
            while j >= 0 and scratch[j] > value:
                scratch[j + 1] = scratch[j]
                j -= 1
            scratch[j + 1] = value
        for i in range(count):
            checksum += scratch[i] * (i + 1)
        count_sum += count
        maximum = max(maximum, count)
    return checksum, count_sum, maximum


@njit(cache=True)
def _array_sort_benchmark(edges, trajectories: int):
    scratch = np.empty(
        edges[0].size + edges[1].size + edges[2].size + edges[3].size + 2,
        dtype=np.float64,
    )
    checksum = 0.0
    count_sum = 0
    maximum = 0
    for trajectory in range(trajectories):
        count = _fill_intersections(edges, trajectory, scratch)
        scratch[:count].sort()
        for i in range(count):
            checksum += scratch[i] * (i + 1)
        count_sum += count
        maximum = max(maximum, count)
    return checksum, count_sum, maximum


@njit(cache=True)
def _sorted_samples(edges, trajectories: int, use_array_sort: bool):
    width = edges[0].size + edges[1].size + edges[2].size + edges[3].size + 2
    output = np.full((trajectories, width), np.nan, dtype=np.float64)
    counts = np.empty(trajectories, dtype=np.int64)
    scratch = np.empty(width, dtype=np.float64)
    for trajectory in range(trajectories):
        count = _fill_intersections(edges, trajectory, scratch)
        if use_array_sort:
            scratch[:count].sort()
        else:
            for i in range(1, count):
                value = scratch[i]
                j = i - 1
                while j >= 0 and scratch[j] > value:
                    scratch[j + 1] = scratch[j]
                    j -= 1
                scratch[j + 1] = value
        output[trajectory, :count] = scratch[:count]
        counts[trajectory] = count
    return output, counts


def _timed(function, edges, trajectories: int, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function(edges, trajectories)
        samples.append(time.perf_counter() - started)
    return np.asarray(samples), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    edges = (
        np.linspace(-4.0, 4.0, 162),
        np.linspace(-4.0, 4.0, 162),
        np.linspace(-5.5, 5.5, 222),
        np.linspace(0.0, 1.0, 82),
    )
    # Compile both candidates before timing.
    insertion_reference = _insertion_benchmark(edges, 1)
    array_sort_reference = _array_sort_benchmark(edges, 1)
    if insertion_reference != array_sort_reference:
        raise AssertionError((insertion_reference, array_sort_reference))
    insertion_samples, insertion_counts = _sorted_samples(edges, 512, False)
    array_sort_samples, array_sort_counts = _sorted_samples(edges, 512, True)
    bitwise_equal = np.array_equal(insertion_counts, array_sort_counts) and np.array_equal(
        insertion_samples, array_sort_samples, equal_nan=True
    )
    if not bitwise_equal:
        raise AssertionError("sorted intersections differ")

    insertion_times, insertion = _timed(
        _insertion_benchmark, edges, args.trajectories, args.repeats
    )
    sort_times, array_sort = _timed(
        _array_sort_benchmark, edges, args.trajectories, args.repeats
    )
    if insertion != array_sort:
        raise AssertionError((insertion, array_sort))

    print(f"trajectories={args.trajectories:,}")
    print(f"mean_intersections={insertion[1] / args.trajectories:.2f}")
    print(f"max_intersections={insertion[2]}")
    print(f"checksum={insertion[0]:.17g}")
    print(f"bitwise_equal={bitwise_equal}")
    print(f"insertion_seconds={np.median(insertion_times):.6f}")
    print(f"array_sort_seconds={np.median(sort_times):.6f}")
    print(f"speedup={np.median(insertion_times) / np.median(sort_times):.3f}x")


if __name__ == "__main__":
    main()
