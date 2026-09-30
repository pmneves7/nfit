"""Compare discrete, mixed, and fractional histogram rebins without saving data.

Run each worker/batch configuration in a fresh process. Use --project and
--member to measure an embedded cached histogram, or --shape for a synthetic
fixture. The first invocation includes kernel startup; later repeats measure
resident kernels. Loading, result summaries, and garbage collection are
outside the timed region.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

from benchmark_rebin_crossover import (
    configuration,
    environment,
    fixture,
    numerical_payload_bytes,
    parse_shape,
    peak_rss_bytes,
    result_stats,
)

from nfit import project_rebinning as rebinning
from nfit._parallel import thread_budget
from nfit.analysis.artifacts import read_project_dataset_artifact
from nfit.mdhisto import MDHistoData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--member", help="histogram archive member inside --project")
    parser.add_argument("--shape", type=parse_shape, default=(96, 80, 64, 24))
    parser.add_argument("--occupancy", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--modes", nargs="+", choices=("discrete", "mixed", "fractional"),
                        default=["discrete", "mixed", "fractional"])
    parser.add_argument("--coarsen", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-mb", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if bool(args.project) != bool(args.member):
        parser.error("--project and --member must be supplied together")
    if min(args.coarsen, args.workers, args.batch_mb, args.repeats) < 1:
        parser.error("coarsen, workers, batch-mb, and repeats must be positive")
    started = perf_counter()
    data = (
        read_project_dataset_artifact(args.project, args.member)
        if args.project else fixture(args.shape, occupancy=args.occupancy, chunk_points=1_000_000)
    )
    if not isinstance(data, MDHistoData):
        parser.error("--member must reference an MDHistoData artifact")
    runtime = environment()
    runtime["numba"] = getattr(sys.modules.get("numba"), "__version__", None)
    print(json.dumps({"environment": runtime, "shape": data.shape,
        "input_payload_bytes": numerical_payload_bytes(data),
        "normalization_weighted": "normalization_denominator" in data.auxiliary_channels,
        "load_seconds": perf_counter() - started, "project": str(args.project),
        "member": args.member}), flush=True)
    with thread_budget(args.workers):
        for mode in args.modes:
            config = configuration(data, operation="regular", coarsen=args.coarsen)
            config.update(workers=args.workers, max_batch_mb=args.batch_mb)
            flags = [mode == "fractional" or (mode == "mixed" and axis.kind != "energy")
                     for axis in data.axes]
            source_vectors = rebinning._mdhisto_rebin_source_axis_vectors(data)
            for axis_config, fractional, vector in zip(config["axes"], flags, source_vectors, strict=True):
                axis_config.update(fractional=fractional, auto_lower=False,
                                   auto_upper=False, auto_step_size=False)
                if vector is not None:
                    axis_config["vector"] = vector.tolist()
            details = {}

            def progress(event, details=details):
                details.update({key: event[key] for key in
                    ("backend", "workers", "parallel_strategy", "batch_size") if key in event})

            for repeat in range(args.repeats):
                gc.collect()
                started = perf_counter()
                result = rebinning._rebin_mdhisto_data(data, config, progress_callback=progress)
                seconds = perf_counter() - started
                row = {"mode": mode, "fractional_axes": flags, "repeat": repeat,
                    "seconds": seconds, "workers_requested": args.workers,
                    "batch_mb": args.batch_mb, "coarsen": args.coarsen,
                    "output_shape": result.shape, "dispatch": details,
                    "stats": result_stats(result), "peak_rss_bytes": peak_rss_bytes()}
                print(json.dumps(row), flush=True)
                del result


if __name__ == "__main__":
    main()
