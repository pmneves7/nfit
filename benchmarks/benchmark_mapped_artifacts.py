"""Benchmark resident and disk-backed reads of a large nfit histogram artifact.

Preparation and reading are deliberately separate commands.  The default
fixture owns five float64 grids and one boolean mask, so shape
``256,192,128,40`` contains about 10.3 GB of numerical data::

    python benchmarks/benchmark_mapped_artifacts.py --prepare /tmp/large.npz
    python benchmarks/benchmark_mapped_artifacts.py --artifact /tmp/large.npz

The comparison command starts each reader in a fresh process.  Read timings are
post-write/OS-cache-dependent; the benchmark does not claim cold-cache results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from nfit.analysis.artifacts import (
    dataset_artifact_from_payload,
    read_dataset_artifact,
    write_dataset_artifact,
)
from nfit.mapped_archive import array_storage_nbytes, read_mapped_array_archive
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData

DEFAULT_SHAPE = (256, 192, 128, 40)
ARRAY_NAMES = (
    "signal", "errors", "mask", "num_events",
    "coverage_fraction", "normalization_denominator",
)
HASH_CHUNK_BYTES = 32 * 1024**2


def _parse_shape(text: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must contain four integers") from error
    if len(shape) != 4 or min(shape) < 1:
        raise argparse.ArgumentTypeError("shape must contain four positive integers")
    return shape  # type: ignore[return-value]


def _fixture(shape: tuple[int, int, int, int], chunk_points: int) -> MDHistoData:
    arrays = [np.empty(shape, dtype=np.float64) for _ in range(5)]
    mask = np.empty(shape, dtype=bool)
    flat = [array.reshape(-1) for array in arrays]
    flat_mask = mask.reshape(-1)
    rng = np.random.default_rng(32969)
    for start in range(0, flat_mask.size, chunk_points):
        stop = min(flat_mask.size, start + chunk_points)
        values = flat[0][start:stop]
        rng.random(values.size, out=values)
        invalid = values < 0.85
        np.sqrt(values, out=flat[1][start:stop])
        flat[2][start:stop].fill(1.0)
        flat[3][start:stop].fill(0.95)
        flat[4][start:stop].fill(2.0)
        for array in flat:
            array[start:stop][invalid] = 0.0
        values[invalid] = np.nan
        flat[1][start:stop][invalid] = np.nan
        flat_mask[start:stop] = invalid
    for array in (*arrays, mask):
        array.setflags(write=False)
    axes = tuple(
        MDHistoAxis(name, np.linspace(-1.0, 1.0, size + 1), unit, kind)
        for name, size, unit, kind in zip(
            ("H", "K", "L", "E"), shape,
            ("rlu", "rlu", "rlu", "meV"),
            ("momentum", "momentum", "momentum", "energy"), strict=True,
        )
    )
    return MDHistoData(
        axes, arrays[0], arrays[1], mask, arrays[2],
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(arrays[3]),
            "normalization_denominator": MDHistoChannel(arrays[4]),
        },
    )


def _arrays(data: MDHistoData) -> dict[str, np.ndarray]:
    return {
        "signal": data.signal,
        "errors": data.errors,
        "mask": data.mask,
        "num_events": data.num_events,
        "coverage_fraction": data.auxiliary_channels["coverage_fraction"].values,
        "normalization_denominator": data.auxiliary_channels[
            "normalization_denominator"
        ].values,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _array_hashes(data: MDHistoData) -> dict[str, str]:
    result = {}
    for name, array in _arrays(data).items():
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        rows = max(1, HASH_CHUNK_BYTES // max(1, array[0:1].nbytes))
        for start in range(0, array.shape[0], rows):
            digest.update(np.ascontiguousarray(array[start:start + rows]).view(np.uint8))
        result[name] = digest.hexdigest()
    return result


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _memory() -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "rss_bytes": None, "pss_bytes": None, "anonymous_bytes": None,
        "file_bytes": None, "peak_rss_bytes": _peak_rss_bytes(),
    }
    path = Path("/proc/self/smaps_rollup")
    if not path.exists():
        return values
    fields: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.split()
        if parts and parts[0].isdigit():
            fields[name] = int(parts[0]) * 1024
    rss = fields.get("Rss")
    anonymous = fields.get("Anonymous")
    values.update(
        rss_bytes=rss, pss_bytes=fields.get("Pss"), anonymous_bytes=anonymous,
        file_bytes=(None if rss is None or anonymous is None else max(0, rss - anonymous)),
    )
    return values


def _timed_operation(callable_) -> dict[str, Any]:
    before = _memory()
    started = perf_counter()
    result = callable_()
    seconds = perf_counter() - started
    after = _memory()
    return {"seconds": seconds, "result": float(result), "memory_before": before,
            "memory_after": after}


def _operations(data: MDHistoData) -> dict[str, dict[str, Any]]:
    signal = data.signal
    center = tuple(size // 2 for size in signal.shape)
    plane = (center[0], slice(None), slice(None), center[3])
    roi = tuple(
        slice(max(0, midpoint - min(32, size // 2)),
              min(size, midpoint + min(32, size - size // 2)))
        for midpoint, size in zip(center, signal.shape, strict=True)
    )
    first = _timed_operation(lambda: np.nansum(signal[plane], dtype=np.float64))
    warm = _timed_operation(lambda: np.nansum(signal[plane], dtype=np.float64))
    bounded_roi = _timed_operation(lambda: np.nansum(signal[roi], dtype=np.float64))
    first_full = _timed_operation(lambda: np.nansum(signal, dtype=np.float64))
    warm_full_rows = [
        _timed_operation(lambda: np.nansum(signal, dtype=np.float64))
        for _ in range(3)
    ]
    warm_full = {
        "seconds": statistics.median(row["seconds"] for row in warm_full_rows),
        "result": warm_full_rows[-1]["result"],
        "memory_before": warm_full_rows[0]["memory_before"],
        "memory_after": warm_full_rows[-1]["memory_after"],
        "repeat_seconds": [row["seconds"] for row in warm_full_rows],
    }
    return {"first_plane_sum": first, "warm_plane_sum": warm,
            "bounded_roi_sum": bounded_roi, "first_full_sum": first_full,
            "warm_full_sum_median": warm_full}


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    target = args.prepare.resolve()
    started = perf_counter()
    data = _fixture(args.shape, args.chunk_points)
    construction_seconds = perf_counter() - started
    raw_bytes = sum(array.nbytes for array in _arrays(data).values())
    expected_hashes = _array_hashes(data)
    started = perf_counter()
    write_dataset_artifact(data, target)
    write_seconds = perf_counter() - started
    archive_hash = _file_hash(target)
    manifest = target.with_suffix(target.suffix + ".benchmark.json")
    report = {
        "mode": "prepare", "artifact": str(target), "shape": list(args.shape),
        "raw_bytes": raw_bytes, "artifact_bytes": target.stat().st_size,
        "construction_seconds": construction_seconds, "write_seconds": write_seconds,
        "array_hashes": expected_hashes, "archive_sha256": archive_hash,
    }
    manifest.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    path = args.artifact.resolve()
    source_before = path.stat()
    memory_before = _memory()
    started = perf_counter()
    if args.worker == "resident":
        data = read_dataset_artifact(path)
    else:
        payload = read_mapped_array_archive(
            path, mapped_min_bytes=args.mapped_min_bytes,
            temp_dir=args.temp_dir,
        )
        data = dataset_artifact_from_payload(payload)
    open_seconds = perf_counter() - started
    memory_after_open = _memory()
    storage = array_storage_nbytes(data)
    operations = _operations(data)
    started = perf_counter()
    hashes = _array_hashes(data)
    hash_seconds = perf_counter() - started
    source_after = path.stat()
    return {
        "reader": args.worker, "open_seconds": open_seconds,
        "hostname": socket.gethostname(),
        "nfit_num_threads": os.environ.get("NFIT_NUM_THREADS"),
        "memory_before_open": memory_before, "memory_after_open": memory_after_open,
        "storage_bytes": {"heap": storage.heap, "mapped": storage.mapped,
                          "total": storage.total},
        "operations": operations, "exact_array_hashes": hashes,
        "exact_hash_seconds": hash_seconds,
        "source_stat_unchanged": (
            source_before.st_size == source_after.st_size
            and source_before.st_mtime_ns == source_after.st_mtime_ns
            and source_before.st_ino == source_after.st_ino
        ),
    }


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    path = args.artifact.resolve()
    manifest_path = path.with_suffix(path.suffix + ".benchmark.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for reader in ("resident", "mapped"):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--artifact", str(path),
            "--worker", reader, "--mapped-min-bytes", str(args.mapped_min_bytes),
        ]
        if args.temp_dir:
            command.extend(("--temp-dir", str(args.temp_dir)))
        environment = os.environ.copy()
        if reader == "mapped":
            environment["NFIT_NUM_THREADS"] = str(args.mapped_workers)
        completed = subprocess.run(
            command, capture_output=True, text=True, env=environment
        )
        if completed.returncode:
            raise RuntimeError(f"{reader} worker failed:\n{completed.stderr}")
        row = json.loads(completed.stdout)
        if row["exact_array_hashes"] != manifest["array_hashes"]:
            raise RuntimeError(f"{reader} reader produced different array contents")
        if not row["source_stat_unchanged"]:
            raise RuntimeError(f"{reader} reader modified the source archive")
        results.append(row)
    archive_hash = _file_hash(path)
    if archive_hash != manifest["archive_sha256"]:
        raise RuntimeError("source archive hash changed during benchmark")
    return {
        "purpose": "resident-versus-mapped nfit artifact crossover benchmark",
        "input": {key: manifest[key] for key in (
            "artifact", "shape", "raw_bytes", "artifact_bytes", "archive_sha256"
        )},
        "measurement": {
            "process_isolation": "one fresh process per reader",
            "cache_state": "OS-cache-dependent; cache is not cleared",
            "decode_concurrency": (
                "resident reader uses nfit's configured bounded parallel NPZ inflation; "
                f"mapped reader uses {args.mapped_workers} bounded inflation worker(s)"
            ),
            "mapped_min_bytes": args.mapped_min_bytes,
            "mapped_workers": args.mapped_workers,
            "hash_chunk_bytes": HASH_CHUNK_BYTES,
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "cpu_count": os.cpu_count()},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--artifact", type=Path)
    parser.add_argument("--shape", type=_parse_shape, default=DEFAULT_SHAPE)
    parser.add_argument("--chunk-points", type=int, default=1_000_000)
    parser.add_argument("--mapped-min-bytes", type=int, default=32 * 1024**2)
    parser.add_argument("--mapped-workers", type=int, default=4)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--worker", choices=("resident", "mapped"), help=argparse.SUPPRESS)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.chunk_points < 1 or args.mapped_min_bytes < 0 or args.mapped_workers < 1:
        parser.error(
            "chunk-points and mapped-workers must be positive; "
            "mapped-min-bytes must be nonnegative"
        )
    if args.worker and not args.artifact:
        parser.error("worker mode requires --artifact")
    report = _prepare(args) if args.prepare else (
        _worker(args) if args.worker else _compare(args)
    )
    rendered = json.dumps(report, indent=None if args.worker else 2, sort_keys=True)
    print(rendered)
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
