"""Compare ordered NumPy and compiled neutron-event accumulation."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from nfit import mdevent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=1_000_000)
    parser.add_argument('--span', type=float, default=4.0)
    parser.add_argument('--shape', default='64,64,64,48')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(','))
    if len(shape) not in (2, 4) or min(shape) < 1 or args.events < 1 or args.repeats < 1:
        parser.error('use two or four positive dimensions and positive event/repeat counts')
    backend = mdevent._MDEVENT_NUMBA
    if backend is None:
        parser.error('Numba neutron-event backend is unavailable')
    rng = np.random.default_rng(37)
    edges = tuple(np.linspace(-1, 1, size + 1) for size in shape)
    coordinates = rng.uniform(-args.span, args.span, (args.events, len(shape)))
    inside = min(10_000, args.events)
    coordinates[:inside] = rng.uniform(-.99, .99, (inside, len(shape)))
    weights = rng.uniform(.1, 3, args.events)
    arrays = [np.zeros(shape) for _ in range(3)]

    def run(compiled):
        mdevent._MDEVENT_NUMBA = backend if compiled else None
        mdevent._accumulate_discrete_event_coordinates(
            coordinates, weights, None, edges, shape, *arrays,
        )

    try:
        run(False)
        expected = [array.copy() for array in arrays]
        for array in arrays:
            array.fill(0)
        run(True)  # compilation is excluded from the timed repetitions
        for actual, reference in zip(arrays, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        del expected
        rows = {}
        for label, compiled in [('numpy', False), ('numba', True)]:
            times = []
            for _ in range(args.repeats):
                for array in arrays:
                    array.fill(0)
                start = time.perf_counter()
                run(compiled)
                times.append(time.perf_counter() - start)
            rows[label] = {'seconds': times, 'median_seconds': statistics.median(times)}
        print(json.dumps({'events': args.events, 'shape': shape, 'span': args.span,
                          'exact_output': True, 'measurements': rows}))
    finally:
        mdevent._MDEVENT_NUMBA = backend


if __name__ == '__main__':
    main()
