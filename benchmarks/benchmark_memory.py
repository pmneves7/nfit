"""Measure cache reuse and histogram rebinning in one fresh process per invocation.

Select baseline/current code with PYTHONPATH. Inputs and result artifacts are
explicit paths; this harness never opens or saves a project or edits preferences.
Peak RSS includes imports, input construction/loading, and warm-up. Timings do not.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from nfit._parallel import thread_budget
from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.cache_utils import array_payload_nbytes
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.project_rebinning import _default_rebin_axes, _rebin_mdhisto_data
from nfit.rebin_cache import RebinCache, RebinCacheBudget


def fixture(shape: tuple[int, ...], *, sparse: bool = True) -> MDHistoData:
    rng = np.random.default_rng(32969)
    signal = rng.random(shape)
    if sparse:
        signal[signal < 0.85] = np.nan
    errors = np.sqrt(signal)
    mask = ~np.isfinite(signal)
    events = np.where(mask, 0.0, 1.0)
    coverage = np.where(mask, 0.0, 0.95)
    normalization = np.where(mask, 0.0, 2.0)
    for array in (signal, errors, mask, events, coverage, normalization):
        array.setflags(write=False)
    axes = tuple(
        MDHistoAxis(
            name, np.linspace(-1, 1, size + 1),
            "meV" if name == "E" else "rlu",
            "energy" if name == "E" else "momentum",
            metadata={"variable": name},
        )
        for name, size in zip(("H", "K", "L", "E"), shape, strict=True)
    )
    return MDHistoData(
        axes, signal, errors, mask, events,
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(coverage),
            "normalization_denominator": MDHistoChannel(normalization),
        },
    )


def configuration(data: MDHistoData, symmetry: bool) -> dict:
    config = {
        "axes": _default_rebin_axes(data), "fractional": False,
        "max_batch_mb": 32, "workers": 4, "mean_weighting": "uniform",
    }
    for axis, size in zip(config["axes"], data.shape, strict=True):
        axis.update(mode="bins", num_bins=max(2, size // 2), auto_lower=True, auto_upper=True,
                    auto_step_size=True, auto_step_size_value=axis["step_size"])
    if symmetry:
        config["symmetry"] = {"mode": "space_group", "expression": "P -1"}
    return config


def peak_rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--shape", default="96,80,64,24")
    parser.add_argument("--operation", choices=("cache", "rebin", "symmetry"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.split(",")))
    if len(shape) != 4 or min(shape) < 1 or args.workers < 1:
        parser.error("provide four positive dimensions and a positive worker count")
    with thread_budget(args.workers):
        if args.operation != "cache":
            # Exceed both the streaming and Numba thresholds with valid data;
            # a tiny sparse fixture would only start asynchronous compilation.
            warm = fixture((32, 32, 16, 16), sparse=False)
            _rebin_mdhisto_data(warm, configuration(warm, args.operation == "symmetry"))
            del warm
            gc.collect()
        data = read_dataset_artifact(args.input) if args.input else fixture(shape)
        row = {
            "operation": args.operation, "shape": list(data.shape),
            "input": str(args.input) if args.input else "synthetic SEQUOIA-shaped sparse float64",
            "input_payload_bytes": array_payload_nbytes((data,)),
            "workers": args.workers, "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(),
        }
        if args.operation == "cache":
            budget = RebinCacheBudget()
            cache = RebinCache(budget)
            budget.configure(max(8 * 1024**2, row["input_payload_bytes"] // 2))
            cache["bin"] = ("unchanged-signature", data)
            cache._compress_resident("bin", max_bytes=budget.limit_bytes)
            del data
            gc.collect()
            restored = []
            times = []
            for _ in range(4):
                start = perf_counter()
                value = cache.get("bin")
                times.append(perf_counter() - start)
                if value is None:
                    raise RuntimeError("fixture did not fit in compressed cache budget")
                restored.append(value[1])
            row.update(
                lookup_seconds=times,
                unique_payload_bytes=array_payload_nbytes(restored),
                same_container=all(value is restored[0] for value in restored),
                accounted_bytes=budget.total_bytes(),
            )
            result = restored[0]
        else:
            config = configuration(data, args.operation == "symmetry")
            started = perf_counter()
            result = _rebin_mdhisto_data(data, config)
            row["seconds"] = perf_counter() - started
            row["output_shape"] = list(result.shape)
        row["peak_rss_bytes"] = peak_rss()
        row["result_payload_bytes"] = array_payload_nbytes((result,))
        if args.output:
            write_dataset_artifact(result, args.output)
        print(json.dumps(row))


if __name__ == "__main__":
    main()
