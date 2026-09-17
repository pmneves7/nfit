"""Measure lossless artifact I/O and histogram slicing in an isolated directory.

Run with the nfit environment, for example::

    python benchmarks/benchmark_large_arrays.py --workers 1,4

The script does not load, modify, or save user projects or preferences.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from nfit._parallel import num_threads, thread_budget
from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.plotting_core import MDHistoSliceViewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="96,80,64,24")
    parser.add_argument("--workers", default="1,4")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 4 or min(shape) < 1 or args.repeats < 1:
        parser.error("provide four positive dimensions and at least one repetition")
    rng = np.random.default_rng(831)
    signal = rng.random(shape)
    signal[signal < 0.85] = np.nan
    errors = np.sqrt(signal)
    mask = ~np.isfinite(signal)
    events = np.where(mask, 0.0, 1.0)
    coverage = np.where(mask, 0.0, 0.95)
    for array in (signal, errors, mask, events, coverage):
        array.setflags(write=False)
    axes = tuple(
        MDHistoAxis(
            name, np.arange(size + 1, dtype=float),
            "meV" if name == "E" else "rlu",
            "energy" if name == "E" else "momentum",
        )
        for name, size in zip(("H", "K", "L", "E"), shape, strict=True)
    )
    data = MDHistoData(
        axes, signal, errors, mask, events,
        auxiliary_channels={"coverage_fraction": MDHistoChannel(coverage)},
    )
    viewer = MDHistoSliceViewer(data, x_dim=0, y_dim=3)
    slice_times = []
    for _ in range(args.repeats):
        started = perf_counter()
        viewer.slice_arrays()
        slice_times.append(perf_counter() - started)
    rows = []
    with tempfile.TemporaryDirectory(prefix="nfit-array-benchmark-") as temporary:
        path = Path(temporary) / "data.npz"
        for workers in map(int, args.workers.split(",")):
            writes, reads = [], []
            with thread_budget(workers):
                ceiling = num_threads()
                for _ in range(args.repeats):
                    started = perf_counter()
                    write_dataset_artifact(data, path)
                    writes.append(perf_counter() - started)
                    started = perf_counter()
                    restored = read_dataset_artifact(path)
                    reads.append(perf_counter() - started)
                    for name in ("signal", "errors", "mask", "num_events"):
                        np.testing.assert_equal(getattr(restored, name), getattr(data, name))
                    np.testing.assert_equal(
                        restored.auxiliary_channels["coverage_fraction"].values, coverage,
                    )
                    del restored
            rows.append({
                "requested_workers": workers, "cpu_ceiling": ceiling,
                "write_seconds": statistics.median(writes),
                "read_seconds": statistics.median(reads),
                "compressed_bytes": path.stat().st_size,
            })
    print(json.dumps({
        "shape": shape, "bins": int(np.prod(shape)),
        "slice_seconds": statistics.median(slice_times), "artifacts": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
