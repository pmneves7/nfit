"""Benchmark adaptive N-D rebinning at representative source sizes.

The 50-million-point case is opt-in because its source arrays alone require
multiple gigabytes. Example:

    python benchmarks/benchmark_rebin.py --sizes 100 500000 50000000
"""

from __future__ import annotations

import argparse
import resource
import sys

import numpy as np

from nfit import rebin_nd


def benchmark(size: int, ndim: int, backend: str, batch_mb: int, workers: int | None, strategy: str, bins_per_dim: int, fractional: bool, coordinate_span: float) -> None:
    rng = np.random.default_rng(20260712)
    coords = rng.random((size, ndim), dtype=np.float64) * coordinate_span
    data = rng.normal(size=size)
    errors = rng.uniform(0.5, 1.5, size=size)
    result = rebin_nd(
        data,
        coords,
        data_errs=errors,
        lower=np.zeros(ndim),
        upper=np.ones(ndim),
        num_bins=[bins_per_dim] * ndim,
        fractional=fractional,
        max_batch_bytes=batch_mb * 1024 * 1024,
        backend=backend,
        workers=workers,
        parallel_strategy=strategy,
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_rss / 1024.0
    print(
        f"points={size:,} dims={ndim} bins={bins_per_dim}^{ndim} fractional={fractional} "
        f"requested={backend}/{strategy} resolved={result.resolved_backend}/"
        f"{result.resolved_parallel_strategy} workers={result.resolved_workers} "
        f"prepare={result.timings['prepare']:.3f}s "
        f"accumulate={result.timings['accumulate']:.3f}s "
        f"normalize={result.timings['normalize']:.3f}s "
        f"total={result.timings['total']:.3f}s "
        f"throughput={size / max(result.timings['accumulate'], 1e-12) / 1e6:.2f}Mpoints/s "
        f"peak_rss~={peak_mb:.0f}MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500_000])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[4])
    parser.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto")
    parser.add_argument("--strategy", choices=["auto", "serial", "dense", "sparse"], default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--bins-per-dim", type=int, default=24)
    parser.add_argument("--nonfractional", action="store_true")
    parser.add_argument("--coordinate-span", type=float, default=1.0)
    parser.add_argument("--batch-mb", type=int, default=192)
    args = parser.parse_args()
    for ndim in args.dimensions:
        for size in args.sizes:
            benchmark(
                size, ndim, args.backend, args.batch_mb, args.workers,
                args.strategy, args.bins_per_dim, not args.nonfractional,
                args.coordinate_span,
            )


if __name__ == "__main__":
    main()
