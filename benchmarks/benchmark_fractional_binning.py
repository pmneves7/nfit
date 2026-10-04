"""Time complete warm fractional point rebins against an optional Git revision.

Source generation and JIT warmup are outside the timers. Preparation, bounded
accumulation, output allocation, and finalization are inside. No Mantid or
instrument files are used; the curved cloud represents a rotating area detector.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
from threadpoolctl import threadpool_limits

from nfit import rebin as current


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reference(revision, directory):
    root = Path(__file__).resolve().parents[1]
    for filename in ("rebin.py", "_rebin_numba.py"):
        contents = subprocess.check_output(
            ["git", "show", f"{revision}:src/nfit/{filename}"], cwd=root,
        )
        (directory / filename).write_bytes(contents)
    kernel = _load_module("nfit._fractional_benchmark_reference_numba", directory / "_rebin_numba.py")
    module = _load_module("nfit._fractional_benchmark_reference", directory / "rebin.py")
    module._NUMBA_REBIN = kernel
    return module


def _curved_detector(size):
    """Map detector centers to elastic Q and rotate them about the sample y axis."""
    side = max(2, int(np.sqrt(size / 16)))
    pixels = np.linspace(-0.3, 0.3, side)
    x, y = np.meshgrid(pixels, pixels, indexing="ij")
    outgoing = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    q_lab = np.array([0.0, 0.0, 1.0]) - outgoing
    clouds = []
    for angle in np.linspace(-0.8, 0.8, 16):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        clouds.append(q_lab @ rotation)
    return np.concatenate(clouds)


def _cases(size):
    rng = np.random.default_rng(7211)
    for name, ndim in (("uniform_3d", 3), ("uniform_4d", 4), ("mixed_4d", 4),
                       ("nonuniform_3d", 3), ("integrated_7d", 7),
                       ("curved_detector_3d", 3), ("streamed_3d", 3)):
        coords = _curved_detector(size) if name == "curved_detector_3d" else rng.uniform(-1.0, 1.0, (size, ndim))
        count = coords.shape[0]
        kwargs = dict(
            data_errs=rng.uniform(0.5, 2.0, count), data_weights=rng.uniform(0.5, 2.0, count),
            lower=[-1.0] * ndim, upper=[1.0] * ndim, num_bins=[24] * ndim,
            fractional=True, backend="numba", mean_weighting="uniform",
        )
        if name == "mixed_4d":
            kwargs["fractional_axes"] = [True, True, True, False]
        elif name == "nonuniform_3d":
            kwargs["bin_edges"] = [np.linspace(-1.0, 1.0, 25) ** 3 for _ in range(ndim)]
        elif name == "integrated_7d":
            kwargs["num_bins"] = [24, 24, 24, 1, 1, 1, 1]
        elif name == "curved_detector_3d":
            kwargs.update(lower=[-0.4] * 3, upper=[0.4] * 3, num_bins=[80] * 3)
        yield name, rng.normal(size=count), coords, kwargs


def _run(module, name, data, coords, kwargs, workers):
    options = dict(kwargs, workers=workers)
    if name == "streamed_3d":
        errors = options.pop("data_errs")
        weights = options.pop("data_weights")
        source = module.ArrayRebinSource(
            data, coords, data_errs=errors, data_weights=weights, batch_size=100_000,
        )
        return module.rebin_nd_stream(source, **options)
    return module.rebin_nd(data, coords, **options)


def benchmark(reference, size, repeats, workers):
    modules = {"current": current} if reference is None else {"reference": reference, "current": current}
    rows = []
    with threadpool_limits(limits=1):
        for name, data, coords, kwargs in _cases(size):
            for worker_count in workers:
                # Specialize both paths before timing, then check the complete
                # numerical output on the exact same source and output grid.
                prepared = {label: _run(module, name, data, coords, kwargs, worker_count)
                            for label, module in modules.items()}
                if reference is not None:
                    for field in ("binned_data", "binned_data_errs", "n_samples", "_normalization"):
                        np.testing.assert_allclose(
                            getattr(prepared["current"], field), getattr(prepared["reference"], field),
                            rtol=1e-12, atol=1e-12, equal_nan=True,
                        )
                timings = {label: [] for label in modules}
                labels = list(modules)
                for iteration in range(repeats):
                    for label in (labels if iteration % 2 == 0 else list(reversed(labels))):
                        started = perf_counter()
                        _run(modules[label], name, data, coords, kwargs, worker_count)
                        timings[label].append(perf_counter() - started)
                medians = {label: float(np.median(values)) for label, values in timings.items()}
                row = dict(
                    case=name, points=data.size, dimensions=coords.shape[1], workers=worker_count,
                    seconds=timings, median_seconds=medians,
                    resolved_backends={label: result.resolved_backend for label, result in prepared.items()},
                )
                if reference is not None:
                    row["speedup"] = medians["reference"] / medians["current"]
                rows.append(row)
                print(json.dumps(row), file=sys.stderr, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Git revision whose two generic rebin modules are timed")
    parser.add_argument("--points", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    if args.points < 1 or args.repeats < 1 or min(args.workers) < 1:
        parser.error("points, repeats, and workers must be positive")
    with tempfile.TemporaryDirectory(prefix="nfit-fractional-code-") as directory:
        reference = None if args.reference is None else _reference(args.reference, Path(directory))
        results = benchmark(reference, args.points, args.repeats, args.workers)
    print(json.dumps(dict(
        reference=args.reference, platform=platform.platform(), machine=platform.machine(),
        python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
        points_requested=args.points, repeats=args.repeats,
        scope="warm full public calls; source generation, JIT compilation and scientific I/O excluded",
        parity="all outputs allclose rtol=atol=1e-12; NaN masks equal" if reference else None,
        results=results,
    ), indent=2))


if __name__ == "__main__":
    main()
