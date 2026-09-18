"""Measure the dense/streaming MDHisto rebin crossover in a fresh process.

Each invocation measures exactly one path, so ``ru_maxrss`` is comparable
between ``--path dense`` and ``--path stream``.  The fixture deliberately owns
five float64 arrays and one boolean mask (41 bytes per source bin); therefore
``--shape 256,192,128,40`` has a numerical source payload of about 10.3 GB.

Examples (run each command in a fresh process)::

    PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
      benchmarks/benchmark_rebin_crossover.py --shape 96,80,64,24 \
      --path dense --operation regular --occupancy sparse
    PYTHONPATH=src /Users/pmneves/anaconda3/envs/nfit/bin/python \
      benchmarks/benchmark_rebin_crossover.py --shape 256,192,128,40 \
      --path stream --operation fractional --occupancy dense

Input construction and optional artifact writing are reported separately and
are excluded from ``rebin_seconds``.  Peak RSS is a process-lifetime high-water
mark, so it includes imports, matching-path warm-up, and the source payload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import socket
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from nfit import project_rebinning as rebinning
from nfit._parallel import num_threads, thread_budget
from nfit.analysis.artifacts import write_dataset_artifact
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData

MISSING_FRACTIONS = {"sparse": 0.85, "dense": 0.30}


def parse_shape(value: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be four comma-separated integers") from error
    if len(shape) != 4 or min(shape) < 1:
        raise argparse.ArgumentTypeError("shape must be four positive dimensions")
    return shape  # type: ignore[return-value]


def fixture(shape: tuple[int, ...], *, occupancy: str, chunk_points: int) -> MDHistoData:
    """Build the full payload in slabs, avoiding a second grid-sized RNG temporary."""

    signal = np.empty(shape, dtype=np.float64)
    errors = np.empty(shape, dtype=np.float64)
    mask = np.empty(shape, dtype=bool)
    events = np.empty(shape, dtype=np.float64)
    coverage = np.empty(shape, dtype=np.float64)
    normalization = np.empty(shape, dtype=np.float64)
    rng = np.random.default_rng(32969)
    missing_fraction = MISSING_FRACTIONS[occupancy]
    flat = tuple(array.reshape(-1) for array in (signal, errors, mask, events, coverage, normalization))
    for start in range(0, signal.size, chunk_points):
        stop = min(signal.size, start + chunk_points)
        values, sigma, flags, counts, covered, denominator = (
            array[start:stop] for array in flat
        )
        rng.random(values.size, out=values)
        invalid = values < missing_fraction
        values[invalid] = np.nan
        np.sqrt(values, out=sigma)
        flags[...] = invalid
        counts.fill(1.0)
        counts[invalid] = 0.0
        covered.fill(0.95)
        covered[invalid] = 0.0
        denominator.fill(2.0)
        denominator[invalid] = 0.0
    for array in (signal, errors, mask, events, coverage, normalization):
        array.setflags(write=False)
    axes = tuple(
        MDHistoAxis(
            name, np.linspace(-1.0, 1.0, size + 1),
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


def configuration(data: MDHistoData, *, operation: str, coarsen: int) -> dict:
    config = {
        "axes": rebinning._default_rebin_axes(data),
        "fractional": operation in {"fractional", "symmetry"},
        "max_batch_mb": 32,
        "workers": 1,
        "mean_weighting": "uniform",
        "minimum_coverage": 0.0,
    }
    for axis, size in zip(config["axes"], data.shape, strict=True):
        axis.update(
            mode="bins", num_bins=max(1, size // coarsen), auto_lower=True,
            auto_upper=True, auto_step_size=True, auto_step_size_value=axis["step_size"],
        )
    if operation == "symmetry":
        config["symmetry"] = {"mode": "space_group", "expression": "P -1"}
    return config


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def numerical_payload_bytes(data: MDHistoData) -> int:
    """Count numerical arrays only, excluding small axes and Python metadata."""

    arrays = [data.signal, data.errors, data.mask, data.num_events]
    arrays.extend(channel.values for channel in data.auxiliary_channels.values())
    arrays.extend(
        channel.errors for channel in data.auxiliary_channels.values()
        if channel.errors is not None
    )
    return sum(array.nbytes for array in arrays)


def result_stats(data: MDHistoData) -> dict[str, float | int]:
    signal = np.asarray(data.signal)
    return {
        "finite_signal_bins": int(np.count_nonzero(np.isfinite(signal))),
        "masked_bins": int(np.count_nonzero(data.mask)),
        "signal_nan_sum": float(np.nansum(signal)),
        "error_nan_sum": float(np.nansum(data.errors)),
        "events_sum": float(np.sum(data.num_events)),
        "coverage_sum": float(np.sum(data.auxiliary_channels["coverage_fraction"].values)),
    }


def numerical_digest(data: MDHistoData, *, normalize_nan: bool) -> str:
    """Hash all numerical channels in bounded slabs for cross-process checks."""

    digest = hashlib.sha256()
    arrays = (
        ("signal", data.signal), ("errors", data.errors), ("mask", data.mask),
        ("num_events", data.num_events),
        *( (f"aux:{name}", channel.values)
           for name, channel in sorted(data.auxiliary_channels.items()) ),
    )
    for name, array in arrays:
        values = np.asarray(array).reshape(-1)
        digest.update(f"{name}:{values.dtype.str}:{values.shape}".encode())
        for start in range(0, values.size, 1_000_000):
            chunk = values[start:start + 1_000_000]
            if normalize_nan and np.issubdtype(chunk.dtype, np.floating) and np.any(np.isnan(chunk)):
                chunk = chunk.copy()
                chunk[np.isnan(chunk)] = np.nan
            digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def environment() -> dict[str, object]:
    thread_variables = {
        name: os.environ[name]
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        if name in os.environ
    }
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        physical_memory = None
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "logical_cpus": os.cpu_count(), "physical_memory_bytes": physical_memory,
        "thread_environment": thread_variables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, default=(96, 80, 64, 24))
    parser.add_argument("--path", choices=("dense", "stream", "auto"), required=True)
    parser.add_argument("--operation", choices=("regular", "fractional", "symmetry"), required=True)
    parser.add_argument("--occupancy", choices=tuple(MISSING_FRACTIONS), default="sparse")
    parser.add_argument("--coarsen", type=int, default=2, help="source/output bin ratio; use 1 for identity")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-points", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, help="optional .npz artifact, written after timing")
    args = parser.parse_args()
    if args.coarsen < 1 or args.workers < 1 or args.chunk_points < 1:
        parser.error("coarsen, workers, and chunk-points must be positive")
    threshold = {"dense": sys.maxsize, "stream": 0, "auto": rebinning.MDHISTO_STREAM_MIN_POINTS}[args.path]
    # The warm-up performs precisely the same selected path and operation.  It is
    # intentionally before fixture timing, while RSS still reports its cost.
    with thread_budget(args.workers):
        effective_workers = num_threads()
        original_threshold = rebinning.MDHISTO_STREAM_MIN_POINTS
        rebinning.MDHISTO_STREAM_MIN_POINTS = threshold
        try:
            # At least 314k valid points even for sparse occupancy; this also
            # crosses the 100k Numba threshold in one streamed 256-MB batch.
            warm = fixture((32, 32, 32, 64), occupancy=args.occupancy, chunk_points=args.chunk_points)
            warm_config = configuration(warm, operation=args.operation, coarsen=2)
            warm_config["max_batch_mb"] = 256
            rebinning._rebin_mdhisto_data(warm, warm_config)
            del warm
            gc.collect()
            started = perf_counter()
            data = fixture(args.shape, occupancy=args.occupancy, chunk_points=args.chunk_points)
            construction_seconds = perf_counter() - started
            config = configuration(data, operation=args.operation, coarsen=args.coarsen)
            streamed = rebinning._mdhisto_streaming_supported(data, config, config["axes"])
            started = perf_counter()
            result = rebinning._rebin_mdhisto_data(data, config)
            rebin_seconds = perf_counter() - started
        finally:
            rebinning.MDHISTO_STREAM_MIN_POINTS = original_threshold
    output_seconds = None
    if args.output:
        if args.output.suffix != ".npz":
            parser.error("--output must name an .npz file")
        started = perf_counter()
        write_dataset_artifact(result, args.output)
        output_seconds = perf_counter() - started
    row = {
        "path_requested": args.path,
        "path_actual": "stream" if streamed else "dense",
        "rebin_backend": "mdhisto_stream" if streamed else "mdhisto_dense",
        "stream_threshold_forced": threshold,
        "operation": args.operation, "fractional": bool(config["fractional"]),
        "symmetry": config.get("symmetry"), "occupancy": args.occupancy,
        "valid_fraction_target": 1.0 - MISSING_FRACTIONS[args.occupancy],
        "shape": list(data.shape), "coarsen": args.coarsen, "workers_requested": args.workers,
        "workers_effective": effective_workers, "input_payload_bytes": numerical_payload_bytes(data),
        "fixture_seed": 32969, "source_digest": numerical_digest(data, normalize_nan=False),
        "construction_seconds": construction_seconds, "rebin_seconds": rebin_seconds,
        "peak_rss_bytes": peak_rss_bytes(), "output_shape": list(result.shape),
        "result_payload_bytes": numerical_payload_bytes(result), "result_stats": result_stats(result),
        "result_digest_normalized_nan": numerical_digest(result, normalize_nan=True),
        "output": str(args.output) if args.output else None, "output_seconds": output_seconds,
        "environment": environment(),
    }
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
