"""Bounded resident and compressed in-memory caches for computed binnings."""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .analysis.artifacts import (
    _payload,
    dataset_artifact_from_payload,
    read_dataset_artifact,
    read_project_dataset_artifact,
)
from .cache_utils import array_payload_nbytes
from .dataset import PointListData
from .mdhisto import MDHistoData

_CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class _DiskBinning:
    signature: str
    path: Path
    project_member: str | None = None
    owned: bool = True

    def restore(self) -> MDHistoData | PointListData:
        if self.project_member is None:
            return read_dataset_artifact(self.path)
        return read_project_dataset_artifact(self.path, self.project_member)


class CompressedBinning:
    """Chunk-compressed NPZ payload held entirely in memory."""

    def __init__(
        self, arrays: dict[str, tuple[str, tuple[int, ...], tuple[bytes, ...]]]
    ):
        self.arrays = arrays
        self.nbytes = sum(
            len(chunk)
            for _dtype, _shape, chunks in arrays.values()
            for chunk in chunks
        )

    @classmethod
    def from_data(
        cls, data: MDHistoData | PointListData, *, max_bytes: int
    ) -> CompressedBinning | None:
        arrays = {}
        used = 0
        for name, value in _payload(data).items():
            array = np.asarray(value)
            if not array.flags.c_contiguous or array.dtype.hasobject:
                return None
            raw = memoryview(array).cast("B")
            chunks = []
            for offset in range(0, len(raw), _CHUNK_BYTES):
                chunk = zlib.compress(raw[offset : offset + _CHUNK_BYTES], level=1)
                used += len(chunk)
                if used > max_bytes:
                    return None
                chunks.append(chunk)
            arrays[name] = (array.dtype.str, tuple(array.shape), tuple(chunks))
        return cls(arrays)

    def _array(self, name: str) -> np.ndarray:
        dtype_text, shape, chunks = self.arrays[name]
        array = np.empty(shape, dtype=np.dtype(dtype_text))
        raw = memoryview(array).cast("B")
        offset = 0
        for chunk in chunks:
            decoded = zlib.decompress(chunk)
            raw[offset : offset + len(decoded)] = decoded
            offset += len(decoded)
        if offset != len(raw):
            raise ValueError(f"compressed binning member {name!r} has the wrong size")
        return array

    def restore(self) -> MDHistoData | PointListData:
        """Reconstruct the normal immutable nfit data container."""

        return dataset_artifact_from_payload(
            {name: self._array(name) for name in self.arrays}
        )

    def write_npz(self, destination: str | Path) -> None:
        """Write a standard compressed NPZ only to the user's chosen path."""

        target = Path(destination)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with ZipFile(
                target,
                mode="x",
                compression=ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as archive:
                created = True
                for name, (dtype_text, shape, chunks) in self.arrays.items():
                    with archive.open(f"{name}.npy", mode="w", force_zip64=True) as stream:
                        np.lib.format.write_array_header_2_0(
                            stream,
                            {
                                "descr": np.lib.format.dtype_to_descr(np.dtype(dtype_text)),
                                "fortran_order": False,
                                "shape": shape,
                            },
                        )
                        for chunk in chunks:
                            stream.write(zlib.decompress(chunk))
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise


class RebinCacheBudget:
    """Enforce one memory and recency budget across result-cache stages."""

    def __init__(self) -> None:
        self.limit_bytes: int | None = None
        self._caches: list[RebinCache] = []
        self._clock = 0

    def register(self, cache: RebinCache) -> None:
        if not any(existing is cache for existing in self._caches):
            self._caches.append(cache)

    def configure(self, total_bytes: int | None) -> None:
        self.limit_bytes = (
            None if total_bytes is None else max(int(total_bytes), 0)
        )

    def next_tick(self) -> int:
        self._clock += 1
        return self._clock

    def total_bytes(self) -> int:
        resident_values = tuple(
            value
            for cache in self._caches
            for value in OrderedDict.values(cache)
        )
        return array_payload_nbytes(resident_values) + sum(
            cache._compressed_bytes for cache in self._caches
        )

    def enforce(self) -> None:
        if self.limit_bytes is None:
            return
        while self.total_bytes() > self.limit_bytes:
            candidates = [
                (*candidate, cache)
                for cache in self._caches
                if (candidate := cache._oldest_candidate()) is not None
            ]
            if not candidates:
                return
            _tick, kind, key, cache = min(candidates, key=lambda item: item[0])
            before = self.total_bytes()
            if kind == "resident":
                cache._compress_resident(key, max_bytes=self.limit_bytes)
                if self.total_bytes() >= before and key in cache._compressed:
                    cache._discard_compressed(key)
            else:
                cache._discard_compressed(key)


SHARED_REBIN_CACHE_BUDGET = RebinCacheBudget()


class RebinCache(OrderedDict):
    """Keep recent arrays and older compressed results within one RAM budget.

    A GUI may inject before_discard to offer an explicit compressed NPZ save
    before the oldest compressed result is discarded. Scripting workflows use
    the bounded cache without Qt or a prompt.
    """

    def __init__(self, budget: RebinCacheBudget | None = None):
        super().__init__()
        self._budget = budget or RebinCacheBudget()
        self._budget.register(self)
        self._compressed: OrderedDict[Any, tuple[str, CompressedBinning]] = OrderedDict()
        self._disk: OrderedDict[Any, _DiskBinning] = OrderedDict()
        self._compressed_bytes = 0
        self._resident_ticks: dict[Any, int] = {}
        self._compressed_ticks: dict[Any, int] = {}
        self._labels: dict[Any, str] = {}
        self.before_discard: (
            Callable[[Any, CompressedBinning], str | Path | None] | None
        ) = None

    def set_label(self, key: Any, label: str) -> None:
        """Associate a human-readable bin name with a process-local cache key."""

        self._labels[key] = label

    @property
    def _compressed_limit(self) -> int:
        """Compatibility view of the single shared cache limit."""

        return int(self._budget.limit_bytes or 0)

    def configure_budget(self, total_bytes: int | None) -> None:
        """Set the single allowance shared by every attached result cache."""

        self._budget.configure(total_bytes)

    def enforce_budget(self) -> None:
        self._budget.enforce()

    def __setitem__(self, key, value):
        self._drop_disk(key)
        old = self._compressed.pop(key, None)
        if old is not None:
            self._compressed_bytes -= old[1].nbytes
            self._compressed_ticks.pop(key, None)
        super().__setitem__(key, value)
        self._resident_ticks[key] = self._budget.next_tick()

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().get(key, default)
        stored = self._compressed.get(key)
        if stored is not None:
            self._compressed.move_to_end(key)
            self._compressed_ticks[key] = self._budget.next_tick()
            signature, artifact = stored
            return signature, artifact.restore()
        disk = self._disk.get(key)
        if disk is None:
            return default
        self._disk.move_to_end(key)
        try:
            return disk.signature, disk.restore()
        except (KeyError, OSError, ValueError):
            self._drop_disk(key)
            return default

    def __contains__(self, key):
        return (
            super().__contains__(key)
            or key in self._compressed
            or key in self._disk
        )

    def has_signature(self, key, signature):
        """Check freshness without decompressing a potentially large grid."""

        resident = super().get(key)
        if resident is not None:
            return resident[0] == signature
        stored = self._compressed.get(key)
        if stored is not None:
            return stored[0] == signature
        disk = self._disk.get(key)
        return disk is not None and disk.signature == signature and disk.path.exists()

    def move_to_end(self, key, last=True):
        if super().__contains__(key):
            super().move_to_end(key, last=last)
            self._resident_ticks[key] = self._budget.next_tick()
        elif key in self._compressed:
            self._compressed.move_to_end(key, last=last)
            self._compressed_ticks[key] = self._budget.next_tick()
        elif key in self._disk:
            self._disk.move_to_end(key, last=last)

    def _drop_disk(self, key: Any) -> None:
        disk = self._disk.pop(key, None)
        if disk is not None and disk.owned:
            try:
                disk.path.unlink(missing_ok=True)
            except OSError:
                pass

    def set_project_backing(
        self,
        key: Any,
        *,
        signature: str,
        project_path: str | Path,
        member: str,
    ) -> None:
        """Replace a session spill with a lazy reference to a saved project member."""

        if key not in self._disk:
            return
        self._drop_disk(key)
        self._disk[key] = _DiskBinning(
            signature=str(signature),
            path=Path(project_path),
            project_member=str(member),
            owned=False,
        )

    def clear_disk_cache(self) -> None:
        """Forget disk-backed entries and remove only nfit-owned session files."""

        for key in tuple(self._disk):
            self._drop_disk(key)

    def clear_project_backing(self, project_path: str | Path) -> None:
        """Forget lazy members owned by one project archive without deleting it."""

        target = Path(project_path)
        for key, disk in tuple(self._disk.items()):
            if not disk.owned and disk.path == target:
                self._drop_disk(key)

    def _oldest_candidate(self) -> tuple[int, str, Any] | None:
        candidates = [
            (tick, "resident", key) for key, tick in self._resident_ticks.items()
        ]
        candidates.extend(
            (tick, "compressed", key)
            for key, tick in self._compressed_ticks.items()
        )
        return min(candidates, default=None, key=lambda item: item[0])

    def _compress_resident(self, key: Any, *, max_bytes: int) -> None:
        if not super().__contains__(key):
            return
        signature, data = OrderedDict.__getitem__(self, key)
        OrderedDict.__delitem__(self, key)
        tick = self._resident_ticks.pop(key, None)
        if tick is None:
            tick = self._budget.next_tick()
        if max_bytes > 0 and isinstance(data, (MDHistoData, PointListData)):
            artifact = CompressedBinning.from_data(data, max_bytes=max_bytes)
            if artifact is not None:
                self._compressed[key] = (signature, artifact)
                self._compressed_bytes += artifact.nbytes
                self._compressed_ticks[key] = tick
                return
        self._labels.pop(key, None)

    def _discard_compressed(self, key: Any) -> None:
        stored = self._compressed.get(key)
        if stored is None:
            return
        signature, artifact = stored
        destination = None
        if self.before_discard is not None:
            destination = self.before_discard(self._labels.get(key, key), artifact)
        self._compressed.pop(key)
        self._compressed_bytes -= artifact.nbytes
        self._compressed_ticks.pop(key, None)
        if destination is None:
            self._labels.pop(key, None)
        else:
            self._disk[key] = _DiskBinning(
                signature=signature,
                path=Path(destination),
            )

    def popitem(self, last=True):
        key = next(reversed(self)) if last else next(iter(self))
        result = key, OrderedDict.__getitem__(self, key)
        self._compress_resident(key, max_bytes=self._compressed_limit)
        self._budget.enforce()
        return result

    def clear(self):
        """Clear resident, compressed, and disk-backed cache tiers."""

        super().clear()
        self._compressed.clear()
        self._compressed_bytes = 0
        self._resident_ticks.clear()
        self._compressed_ticks.clear()
        self.clear_disk_cache()
        self._labels.clear()
