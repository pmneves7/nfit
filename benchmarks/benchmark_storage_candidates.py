"""Compare isolated, lossless 4-D storage candidates for SEQUOIA-like grids.

This exploratory benchmark does not implement an nfit storage backend. Its
synthetic fixture has float64 ``(H, K, L, E)`` signal, error, and event-count
arrays. Every candidate/repeat uses a fresh process and is checked exactly.
Reads are expressly post-write OS-cache-affected, never cold-cache claims.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ORIENTATIONS = ((0, 1), (0, 3), (2, 3))
HDF5_FILTERS = ("gzip", "lzf")


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import h5py
        result.update(h5py=h5py.__version__, hdf5=h5py.version.hdf5_version)
    except ImportError:
        result["h5py"] = "unavailable"
    try:
        import blosc2
        result["blosc2"] = blosc2.__version__
    except ImportError:
        result["blosc2"] = "unavailable (not benchmarked)"
    return result


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    shape = tuple(int(item) for item in value.split(","))
    if len(shape) != 4 or min(shape) < 2:
        raise argparse.ArgumentTypeError("shape must contain four integers of at least 2")
    return shape  # type: ignore[return-value]


def _fixture(shape: tuple[int, int, int, int], sparsity: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260917)
    signal = rng.standard_normal(shape)
    signal[rng.random(shape) < sparsity] = 0.0
    return {"signal": signal, "errors": np.sqrt(np.abs(signal)),
            "num_events": np.where(signal == 0.0, 0.0, 1.0)}


def _chunks(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    chunk = list(shape)
    while int(np.prod(chunk)) * np.dtype("float64").itemsize > 1024 * 1024:
        index = max(range(4), key=lambda i: chunk[i])
        chunk[index] = max(1, (chunk[index] + 1) // 2)
    return tuple(chunk)  # type: ignore[return-value]


def _blocks(chunks: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(max(1, size // 2) for size in chunks)  # type: ignore[return-value]


def _artifact_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file())


def _timed(callable_):
    started = perf_counter()
    result = callable_()
    return perf_counter() - started, result


def _operations(store, shape: tuple[int, int, int, int], expected: np.ndarray) -> dict[str, float]:
    full_seconds, full = _timed(lambda: np.array(store, copy=True))
    np.testing.assert_array_equal(full, expected)
    del full
    plane_seconds: dict[str, float] = {}
    for axes in ORIENTATIONS:
        selectors: list[object] = [size // 2 for size in shape]
        selectors[axes[0]] = selectors[axes[1]] = slice(None)
        selector = tuple(selectors)
        seconds, plane = _timed(lambda selector=selector: np.array(store[selector], copy=True))
        np.testing.assert_array_equal(plane, expected[selector])
        plane_seconds[f"{axes[0]}-{axes[1]}"] = seconds
    roi = tuple(slice(size // 4, 3 * size // 4) for size in shape)
    seconds, reduced = _timed(lambda: float(np.sum(store[roi], dtype=np.float64)))
    np.testing.assert_allclose(
        reduced, float(np.sum(expected[roi], dtype=np.float64)), rtol=0,
        atol=np.finfo(np.float64).eps * expected[roi].size,
    )
    return {"full_read_seconds": full_seconds, "roi_reduce_seconds": seconds,
            **{f"thin_plane_{key}_seconds": value for key, value in plane_seconds.items()}}


def _load_input(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _check_all(store: dict[str, object], expected: dict[str, np.ndarray]) -> None:
    for name, values in expected.items():
        np.testing.assert_array_equal(np.array(store[name], copy=True), values)


def _blosc_params(blosc2, candidate: str):
    if "lz4_bitshuffle" in candidate:
        return blosc2.CParams(
            codec=blosc2.Codec.LZ4, clevel=1,
            filters=[blosc2.Filter.NOFILTER] * 5 + [blosc2.Filter.BITSHUFFLE],
        )
    return blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5)


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    expected = _load_input(Path(args.input))
    signal, artifact, candidate = expected["signal"], Path(args.artifact), args.worker
    if candidate == "resident_numpy":
        construction_seconds, store = _timed(lambda: {name: np.array(values, copy=True) for name, values in expected.items()})
        write_seconds, artifact_bytes = 0.0, sum(value.nbytes for value in expected.values())
        operation = _operations(store["signal"], signal.shape, signal)
        _check_all(store, expected)
    elif candidate == "npy_memmap":
        def write_npy():
            artifact.mkdir()
            for name, values in expected.items():
                np.save(artifact / f"{name}.npy", values)
        write_seconds, _ = _timed(write_npy)
        artifact_bytes = _artifact_bytes(artifact)
        construction_seconds, store = _timed(lambda: {name: np.load(artifact / f"{name}.npy", mmap_mode="r") for name in expected})
        operation = _operations(store["signal"], signal.shape, signal)
        _check_all(store, expected)
    elif candidate.startswith("hdf5_"):
        import h5py
        compression, chunks = candidate.removeprefix("hdf5_"), _chunks(signal.shape)
        def write_hdf5():
            with h5py.File(artifact, "w") as handle:
                for name, values in expected.items():
                    handle.create_dataset(name, data=values, chunks=chunks, compression=compression, shuffle=True)
        write_seconds, _ = _timed(write_hdf5)
        artifact_bytes = _artifact_bytes(artifact)
        def open_hdf5():
            handle = h5py.File(artifact, "r")
            return handle, {name: handle[name] for name in expected}
        construction_seconds, (handle, store) = _timed(open_hdf5)
        try:
            operation = _operations(store["signal"], signal.shape, signal)
            _check_all(store, expected)
        finally:
            handle.close()
    elif candidate.startswith("blosc2_"):
        import blosc2
        chunks = _chunks(signal.shape)
        params = _blosc_params(blosc2, candidate)
        if candidate.endswith("_memory"):
            construction_seconds, store = _timed(lambda: {
                name: blosc2.asarray(values, chunks=chunks, blocks=_blocks(chunks), cparams=params)
                for name, values in expected.items()
            })
            write_seconds, artifact_bytes = 0.0, 0
            operation = _operations(store["signal"], signal.shape, signal)
            _check_all(store, expected)
            raw_bytes = sum(value.nbytes for value in expected.values())
            compressed_payload_bytes = int(sum(array.schunk.cbytes for array in store.values()))
            return {"candidate": candidate, "construction_seconds": construction_seconds,
                    "write_seconds": write_seconds, "artifact_bytes": artifact_bytes,
                    "compression_ratio_vs_raw": None,
                    "compressed_payload_bytes": compressed_payload_bytes,
                    "compressed_payload_ratio_vs_raw": raw_bytes / compressed_payload_bytes,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "peak_rss_interpretation": "process high-water includes input and exact-check temporaries; not retained-store incremental memory",
                    "storage_location": "memory buffer (no persisted artifact)",
                    "access_cache_state": "in-memory compressed store (not a disk-cache measurement)", **operation}

        def write_blosc2():
            artifact.mkdir()
            for name, values in expected.items():
                blosc2.asarray(values, chunks=chunks, blocks=_blocks(chunks), cparams=params,
                               urlpath=str(artifact / f"{name}.b2"), mode="w")
        write_seconds, _ = _timed(write_blosc2)
        artifact_bytes = _artifact_bytes(artifact)
        construction_seconds, store = _timed(lambda: {name: blosc2.open(artifact / f"{name}.b2") for name in expected})
        operation = _operations(store["signal"], signal.shape, signal)
        _check_all(store, expected)
    else:
        raise ValueError(f"unknown candidate {candidate}")
    raw_bytes = sum(value.nbytes for value in expected.values())
    return {"candidate": candidate, "construction_seconds": construction_seconds, "write_seconds": write_seconds,
            "artifact_bytes": artifact_bytes, "compression_ratio_vs_raw": raw_bytes / artifact_bytes,
            "peak_rss_bytes": _peak_rss_bytes(),
            "access_cache_state": "post_write_os_cache_affected (not a cold-cache measurement)", **operation}


def _child_command(candidate: str, input_path: Path, artifact: Path, blosc2_path: Path | None) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", candidate,
               "--input", str(input_path), "--artifact", str(artifact)]
    return command + (["--blosc2-path", str(blosc2_path)] if blosc2_path else [])


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    candidates = ["resident_numpy", "npy_memmap"]
    try:
        import h5py  # noqa: F401
    except ImportError:
        pass
    else:
        candidates.extend(f"hdf5_{name}" for name in HDF5_FILTERS)
    try:
        import blosc2  # noqa: F401
    except ImportError:
        pass
    else:
        candidates.extend(("blosc2_zstd", "blosc2_lz4_bitshuffle", "blosc2_lz4_bitshuffle_memory"))
    with tempfile.TemporaryDirectory(prefix="nfit-storage-candidates-") as temporary:
        directory = Path(temporary)
        input_path = directory / "synthetic_sequoia_arrays.npz"
        if args.input_fixture:
            arrays = _load_input(args.input_fixture)
            if set(arrays) != {"signal", "errors", "num_events"}:
                raise ValueError("input fixture must contain signal, errors, and num_events arrays")
            if len({values.shape for values in arrays.values()}) != 1 or arrays["signal"].ndim != 4:
                raise ValueError("input fixture arrays must have one shared four-dimensional shape")
            np.savez(input_path, **arrays)
            input_kind = f"provided SEQUOIA fixture: {args.input_fixture.name}"
        else:
            arrays = _fixture(args.shape, args.sparsity)
            np.savez(input_path, **arrays)
            input_kind = "synthetic SEQUOIA-shaped fixture"
        raw_bytes = sum(value.nbytes for value in arrays.values())
        shape = arrays["signal"].shape
        zero_fraction = float(np.mean(arrays["signal"] == 0.0))
        nan_fraction = float(np.mean(~np.isfinite(arrays["signal"])))
        del arrays
        results = []
        for candidate in candidates:
            repeats = []
            for repeat in range(args.repeats):
                artifact = directory / f"{candidate}-{repeat}.{'h5' if candidate.startswith('hdf5') else 'store'}"
                environment = os.environ.copy()
                if args.blosc2_path:
                    environment["PYTHONPATH"] = str(args.blosc2_path) + os.pathsep + environment.get("PYTHONPATH", "")
                completed = subprocess.run(_child_command(candidate, input_path, artifact, args.blosc2_path), text=True, capture_output=True, env=environment)
                if completed.returncode:
                    raise RuntimeError(f"{candidate} worker failed:\n{completed.stderr or completed.stdout}")
                repeats.append(json.loads(completed.stdout))
            results.append({key: (statistics.median([row[key] for row in repeats]) if isinstance(repeats[0][key], (int, float)) else repeats[0][key]) for key in repeats[0]})
    return {"purpose": "exploratory benchmark; no production storage backend",
            "input": {"kind": f"{input_kind}; 4-D float64 signal plus errors and event counts", "shape": shape,
                      "axis_order": (["dimension_0", "dimension_1", "dimension_2", "dimension_3"]
                                     if args.input_fixture else ["H", "K", "L", "E"]),
                      "arrays": ["signal", "errors", "num_events"],
                      "sparsity_zero_fraction": zero_fraction, "signal_nan_fraction": nan_fraction,
                      "raw_bytes": raw_bytes},
            "measurement": {"repeats": args.repeats, "process_isolation": "one fresh process per candidate and repeat", "read_cache_label": "post-write OS-cache-affected; cache was not cleared",
                            "operations": ["construction/open", "full materialized signal read", "2-D thin signal planes (0-1, 0-3, 2-3)", "central signal ROI float64 sum", "exact full check of all arrays"]},
            "environment": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                            "thread_env": {name: os.environ.get(name, "unset") for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "versions": _versions()}, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, default=(64, 64, 48, 24))
    parser.add_argument("--sparsity", type=float, default=0.70)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--input-fixture", type=Path, help="existing NPZ with signal, errors, and num_events")
    parser.add_argument("--blosc2-path", type=Path, help="temporary optional Blosc2 installation")
    parser.add_argument("--worker", choices=(
        "resident_numpy", "npy_memmap", "hdf5_gzip", "hdf5_lzf", "blosc2_zstd",
        "blosc2_lz4_bitshuffle", "blosc2_lz4_bitshuffle_memory",
    ))
    parser.add_argument("--input")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    if args.blosc2_path:
        sys.path.insert(0, str(args.blosc2_path))
    if not 0 <= args.sparsity <= 1 or args.repeats < 1:
        parser.error("sparsity must be in [0, 1] and repeats must be positive")
    if args.worker:
        if not args.input or not args.artifact:
            parser.error("worker mode requires --input and --artifact")
        print(json.dumps(_run_worker(args)))
        return
    report = _run_parent(args)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
