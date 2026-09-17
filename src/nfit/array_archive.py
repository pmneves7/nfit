"""Bounded parallel compression for ordinary NumPy NPZ archives.

Large members use independent DEFLATE blocks in one standard ZIP stream.
Readers need no nfit-specific codec. Small payloads retain NumPy's writer.
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, BinaryIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .performance import operation_worker_count

_CHUNK_BYTES = 4 * 1024**2
_PARALLEL_MIN_BYTES = 32 * 1024**2


def _deflate_block(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    # FULL_FLUSH ends a byte-aligned block without setting the final-block bit.
    # Independent blocks can therefore be concatenated and decoded normally.
    return compressor.compress(data) + compressor.flush(zlib.Z_FULL_FLUSH)


class _ParallelDeflater:
    def __init__(self, executor: ThreadPoolExecutor, workers: int) -> None:
        self.executor = executor
        self.workers = workers
        self.pending: deque[Future[bytes]] = deque()

    def compress(self, data: Any) -> bytes:
        raw = memoryview(data).cast("B")
        completed = []
        for start in range(0, len(raw), _CHUNK_BYTES):
            if len(self.pending) >= self.workers:
                completed.append(self.pending.popleft().result())
            # NumPy may reuse its iteration buffer after write() returns.
            block = bytes(raw[start : start + _CHUNK_BYTES])
            self.pending.append(self.executor.submit(_deflate_block, block))
        return b"".join(completed)

    def flush(self) -> bytes:
        completed = [future.result() for future in self.pending]
        self.pending.clear()
        completed.append(zlib.compressobj(wbits=-15).flush())
        return b"".join(completed)


def write_array_archive(destination: BinaryIO, payload: Mapping[str, Any]) -> None:
    """Write a lossless, pickle-free NPZ using the shared CPU and RAM limits."""

    arrays = {name: np.asanyarray(value) for name, value in payload.items()}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("object arrays cannot be saved in an nfit array archive")
    largest = max((array.nbytes for array in arrays.values()), default=0)
    workers = min(
        max(1, (largest + _CHUNK_BYTES - 1) // _CHUNK_BYTES),
        operation_worker_count(
            largest,
            bytes_per_worker=4 * _CHUNK_BYTES,
            min_parallel_bytes=_PARALLEL_MIN_BYTES,
        ),
    )
    if workers <= 1:
        np.savez_compressed(destination, **arrays)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        with ZipFile(destination, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
            for name, array in arrays.items():
                with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                    # zipfile owns headers, CRCs, sizes and ZIP64. Its compressor
                    # is the sole private adapter here; fall back to its ordinary
                    # compressor if a future Python changes the writer interface.
                    if array.nbytes >= _PARALLEL_MIN_BYTES and hasattr(stream, "_compressor"):
                        stream._compressor = _ParallelDeflater(executor, workers)
                    np.lib.format.write_array(stream, array, allow_pickle=False)
