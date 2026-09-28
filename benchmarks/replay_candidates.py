"""Experimental replay layouts and deterministic, disjoint-energy scatter.

The transpose and energy-scatter candidates remain outside production. The
predicted variant uses the production index predictor; the baseline explicitly
disables it. Run with
PYTHONPATH=src:benchmarks and the nfit interpreter. Timings include allocation,
transposition and accumulation, but exclude JIT, I/O and normalization.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from dataclasses import replace
from time import perf_counter

import numpy as np
from benchmark_measured_background_replay import make_fixture
from numba import get_thread_id, njit, prange, set_num_threads

from nfit import _mdevent_background_numba as base


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def combine_contiguous(raw, weights, unique, combined, slots_by_worker):
    for event in prange(raw.shape[0]):
        slots = slots_by_worker[get_thread_id()]
        slots[:] = -1
        count = 0
        for transform in range(raw.shape[1]):
            flat = raw[event, transform]
            if flat < 0:
                continue
            slot = base._hash_slot(flat, slots.size - 1)
            while slots[slot] >= 0 and unique[event, slots[slot]] != flat:
                slot = (slot + 1) & (slots.size - 1)
            collision = slots[slot]
            if collision < 0:
                unique[event, count] = flat
                combined[event, count] = weights[transform]
                slots[slot] = count
                count += 1
            else:
                combined[event, collision] += weights[transform]


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def scatter_energy(unique, combined, signal, source_variance, numerator, variance,
                   events, energy_bins, workers):
    # Momentum transforms leave energy unchanged: every copy of one event
    # belongs to the same worker. Within each destination bin, source-event
    # order is unchanged. No atomics or private output grids are needed.
    for owner in prange(workers):
        for event in range(unique.shape[0]):
            first = unique[event, 0]
            if first < 0 or (first % energy_bins) % workers != owner:
                continue
            for column in range(unique.shape[1]):
                flat = unique[event, column]
                if flat < 0:
                    break
                weight = combined[event, column]
                numerator[flat] += signal[event] * weight
                variance[flat] += source_variance[event] * weight * weight
                events[flat] += 1.0


def replay(fixture, workers, layout, scatter):
    set_num_threads(workers)
    outputs = tuple(np.zeros(int(np.prod(fixture.shape))) for _ in range(3))
    rows = max(1, min(100_000, 1_000_000 // len(fixture.weights)))
    timings = dict(allocation=0., mapping=0., collision=0., scatter=0.)
    started = perf_counter()
    for start in range(0, len(fixture.lab), rows):
        stop = min(start + rows, len(fixture.lab))
        n, t = stop-start, len(fixture.weights)
        tick = perf_counter()
        raw = np.full((t, n), -1, dtype=np.int64)
        unique = np.full((n, t), -1, dtype=np.int64)
        combined = np.empty((n, t))
        slots = np.empty((workers, 1 << (2*t-1).bit_length()), dtype=np.int64)
        energies = np.ascontiguousarray(fixture.block[start:stop, 8])
        signal = np.ascontiguousarray(fixture.block[start:stop, 0])
        variance = np.ascontiguousarray(fixture.block[start:stop, 1])
        timings['allocation'] += perf_counter()-tick
        tick = perf_counter()
        prediction = (base._regular_axes(fixture.edges[:3]),) if layout == "predicted" else ()
        base._map_transforms(fixture.lab[start:stop], energies,
            fixture.detector_indices[start:stop], fixture.inverses, fixture.energy_bounds,
            fixture.accepted_compiled, fixture.exclusions_compiled, *fixture.edges[:3],
            base._energy_bin_indices(energies, fixture.edges[3]),
            np.asarray(fixture.shape), raw, *prediction)
        timings['mapping'] += perf_counter()-tick
        tick = perf_counter()
        if layout == 'transpose':
            combine_contiguous(np.ascontiguousarray(raw.T), fixture.weights, unique, combined, slots)
        else:
            base._combine_collisions(raw, fixture.weights, unique, combined, slots)
        timings['collision'] += perf_counter()-tick
        tick = perf_counter()
        if scatter == 'energy':
            scatter_energy(unique, combined, signal, variance, *outputs, fixture.shape[3], workers)
        else:
            base._scatter(unique, combined, signal, variance, *outputs)
        timings['scatter'] += perf_counter()-tick
    timings['total'] = perf_counter()-started
    return timings, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=4000)
    parser.add_argument('--transforms', type=int, default=722)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--spatial-bins', type=int, default=48)
    parser.add_argument('--nonuniform', action='store_true')
    args = parser.parse_args()
    if min(args.events, args.transforms, args.workers, args.repeats, args.spatial_bins) < 1:
        parser.error('workload sizes and workers must be positive')
    args.workers = base.effective_workers(args.workers)
    fixture = make_fixture(args.events, args.transforms, 20260928,
                           shape=(args.spatial_bins, args.spatial_bins, 12, 32))
    if args.nonuniform:
        edges = tuple(edge**3 / max(abs(edge))**2 for edge in fixture.edges[:3])
        fixture = replace(fixture, edges=(*edges, fixture.edges[3]))
    modes = [('binary', 'serial'), ('transpose', 'serial'), ('binary', 'energy'), ('transpose', 'energy'), ('predicted', 'serial')]
    reference = None
    for layout, scatter in modes:
        _, out = replay(fixture, args.workers, layout, scatter)
        if reference is None:
            reference = out
        for expected, actual in zip(reference, out, strict=True):
            assert expected.tobytes() == actual.tobytes(), (layout, scatter)
    samples = {mode: [] for mode in modes}
    # Interleave alternatives so cache warmth and machine drift affect all.
    for repeat in range(args.repeats):
        for mode in (modes if repeat % 2 == 0 else modes[::-1]):
            timings, out = replay(fixture, args.workers, *mode)
            for expected, actual in zip(reference, out, strict=True):
                assert expected.tobytes() == actual.tobytes(), mode
            samples[mode].append(timings)
    result = dict(workload=vars(args), bitwise_equal=True, timings={},
                  environment=dict(platform=platform.platform(), python=platform.python_version(),
                                   numpy=np.__version__, numba=__import__('numba').__version__))
    for mode, runs in samples.items():
        result['timings']['/'.join(mode)] = {
            key: statistics.median(run[key] for run in runs) for key in runs[0]
        }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
