"""Bounded resident and compressed in-memory caches for computed binnings."""

from __future__ import annotations

import weakref
import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np

from .analysis.artifacts import (
    _payload,
    dataset_artifact_from_payload,
    read_dataset_artifact,
    read_project_dataset_artifact,
)
from .dataset import PointListData
from .mapped_archive import array_storage_nbytes
from .mdhisto import MDHistoData

_CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class _DiskBinning:
    signature: str
    path: Path
    project_member: str | None = None
    owned: bool = True
    project_identity: tuple[int, int, int, int] | None = None

    def available(self) -> bool:
        if not self.path.exists():
            return False
        if self.project_member is None or self.project_identity is None:
            return True
        try:
            stat = self.path.stat()
        except OSError:
            return False
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ) == self.project_identity

    def restore(self) -> MDHistoData | PointListData:
        if self.project_member is None:
            return read_dataset_artifact(self.path, memory_map=None)
        if not self.available():
            raise OSError("the project archive changed since this cache was registered")
        return read_project_dataset_artifact(
            self.path, self.project_member, memory_map=None
        )


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

        arrays = {name: self._array(name) for name in self.arrays}
        for array in arrays.values():
            array.setflags(write=False)
        return dataset_artifact_from_payload(arrays)

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
        self._borrowed: dict[int, weakref.ReferenceType[Any]] = {}
        self._borrowed_lock = RLock()

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
        with self._borrowed_lock:
            borrowed_values = tuple(
                value
                for reference in self._borrowed.values()
                if (value := reference()) is not None
            )
        # File-backed pages can be reclaimed by the OS. Charge only owned
        # heap allocations, deduplicating views of the same backing arrays.
        return array_storage_nbytes((resident_values, borrowed_values)).heap + sum(
            cache._compressed_bytes for cache in self._caches
        )

    def register_borrowed(self, value: MDHistoData | PointListData) -> None:
        """Account for a decoded compressed result until its last user releases it."""

        identity = id(value)

        def released(reference: weakref.ReferenceType[Any]) -> None:
            with self._borrowed_lock:
                current = self._borrowed.get(identity)
                if current is reference:
                    self._borrowed.pop(identity, None)

        reference = weakref.ref(value, released)
        with self._borrowed_lock:
            current = self._borrowed.get(identity)
            if current is None or current() is not value:
                self._borrowed[identity] = reference

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
        self._project_backings: dict[Any, _DiskBinning] = {}
        self._compressed_bytes = 0
        self._resident_ticks: dict[Any, int] = {}
        self._compressed_ticks: dict[Any, int] = {}
        self._labels: dict[Any, str] = {}
        self._decode_lock = RLock()
        self._decoded: dict[
            Any, tuple[str, weakref.ReferenceType[MDHistoData | PointListData]]
        ] = {}
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

        if self._budget is SHARED_REBIN_CACHE_BUDGET:
            # Preferences may change while nfit is running. Do not let module
            # import-time cache constants compete to set this shared limit.
            from .cache_utils import scientific_cache_budget_bytes

            self._budget.configure(scientific_cache_budget_bytes())
        else:
            self._budget.configure(total_bytes)

    def enforce_budget(self) -> None:
        self._budget.enforce()

    def __setitem__(self, key, value):
        with self._decode_lock:
            self._decoded.pop(key, None)
            self._drop_disk(key)
            signature = value[0] if isinstance(value, tuple) and value else None
            backing = self._project_backings.get(key)
            if (
                backing is not None
                and (backing.signature != signature or not backing.available())
            ):
                self._project_backings.pop(key, None)
            old = self._compressed.pop(key, None)
            if old is not None:
                self._compressed_bytes -= old[1].nbytes
                self._compressed_ticks.pop(key, None)
            super().__setitem__(key, value)
            self._resident_ticks[key] = self._budget.next_tick()
            if (
                signature is not None
                and len(value) > 1
                and isinstance(value[1], (MDHistoData, PointListData))
            ):
                self._remember_decoded(key, signature, value[1])

    def _remember_decoded(
        self, key: Any, signature: str, value: MDHistoData | PointListData
    ) -> None:
        """Weakly retain one canonical decoded value without extending its life."""

        cache_ref = weakref.ref(self)

        def released(reference, *, cache_key=key):
            cache = cache_ref()
            if cache is None:
                return
            with cache._decode_lock:
                current = cache._decoded.get(cache_key)
                if current is not None and current[1] is reference:
                    cache._decoded.pop(cache_key, None)

        reference = weakref.ref(value, released)
        self._decoded[key] = (signature, reference)
        self._budget.register_borrowed(value)

    def get(self, key, default=None):
        if self._budget is SHARED_REBIN_CACHE_BUDGET:
            self.configure_budget(None)
        with self._decode_lock:
            if super().__contains__(key):
                result = super().get(key, default)
            else:
                result = None
            decoded = self._decoded.get(key)
            if result is None and decoded is not None:
                value = decoded[1]()
                if value is not None:
                    if key in self._compressed:
                        self._compressed.move_to_end(key)
                        self._compressed_ticks[key] = self._budget.next_tick()
                    result = decoded[0], value
                else:
                    self._decoded.pop(key, None)
            stored = self._compressed.get(key) if result is None else None
            if result is None and stored is not None:
                self._compressed.move_to_end(key)
                self._compressed_ticks[key] = self._budget.next_tick()
                signature, artifact = stored
                value = artifact.restore()
                self._remember_decoded(key, signature, value)
                result = signature, value
            disk = self._disk.get(key) if result is None else None
            if result is None and disk is not None:
                self._disk.move_to_end(key)
                try:
                    restored = disk.restore()
                except (BadZipFile, EOFError, KeyError, OSError, ValueError, zlib.error):
                    self._drop_disk(key)
                    if self._project_backings.get(key) is disk:
                        self._project_backings.pop(key, None)
                    return default
                result = (disk.signature, restored)
                self[key] = result
                self.move_to_end(key)
            if result is None:
                return default
        if self._budget is SHARED_REBIN_CACHE_BUDGET or disk is not None:
            self._budget.enforce()
        return result

    def peek_resident(self, key: Any, default: Any = None) -> Any:
        """Return only an already-decoded value without changing cache recency."""

        return OrderedDict.get(self, key, default)

    def __contains__(self, key):
        decoded = self._decoded.get(key)
        return (
            super().__contains__(key)
            or key in self._compressed
            or key in self._disk
            or (decoded is not None and decoded[1]() is not None)
        )

    def has_signature(self, key, signature):
        """Check freshness without decompressing a potentially large grid."""

        resident = super().get(key)
        if resident is not None:
            return resident[0] == signature
        stored = self._compressed.get(key)
        if stored is not None:
            return stored[0] == signature
        decoded = self._decoded.get(key)
        if decoded is not None and decoded[1]() is not None:
            return decoded[0] == signature
        disk = self._disk.get(key)
        return (
            disk is not None
            and disk.signature == signature
            and disk.available()
        )

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
        lazy: bool = False,
    ) -> None:
        """Record a reusable saved artifact, optionally as the active lazy tier."""

        with self._decode_lock:
            self._set_project_backing_locked(
                key,
                signature=signature,
                project_path=project_path,
                member=member,
                lazy=lazy,
            )

    def _set_project_backing_locked(
        self,
        key: Any,
        *,
        signature: str,
        project_path: str | Path,
        member: str,
        lazy: bool,
    ) -> None:

        path = Path(project_path)
        stat = path.stat()
        backing = _DiskBinning(
            signature=str(signature),
            path=path,
            project_member=str(member),
            owned=False,
            project_identity=(
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            ),
        )
        self._project_backings[key] = backing
        if lazy:
            self._decoded.pop(key, None)
            if super().__contains__(key):
                OrderedDict.__delitem__(self, key)
                self._resident_ticks.pop(key, None)
            old = self._compressed.pop(key, None)
            if old is not None:
                self._compressed_bytes -= old[1].nbytes
                self._compressed_ticks.pop(key, None)
            self._drop_disk(key)
            self._disk[key] = backing
        elif key in self._disk:
            self._drop_disk(key)
            self._disk[key] = backing
        elif not super().__contains__(key) and key not in self._compressed:
            self._disk[key] = backing

    def project_backing(
        self, key: Any, signature: str
    ) -> tuple[Path, str] | None:
        """Return a current saved member usable without restoring its arrays."""

        backing = self._project_backings.get(key)
        if (
            backing is None
            or backing.signature != signature
            or backing.project_member is None
            or not backing.available()
        ):
            if backing is not None and not backing.available():
                self._project_backings.pop(key, None)
                if self._disk.get(key) is backing:
                    self._drop_disk(key)
            return None
        return backing.path, backing.project_member

    def clear_disk_cache(self) -> None:
        """Forget disk-backed entries and remove only nfit-owned session files."""

        for key in tuple(self._disk):
            self._drop_disk(key)

    def clear_project_backing(self, project_path: str | Path) -> None:
        """Forget lazy members owned by one project archive without deleting it."""

        target = Path(project_path)
        for key, backing in tuple(self._project_backings.items()):
            if backing.path == target:
                self._project_backings.pop(key, None)
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
        with self._decode_lock:
            self._compress_resident_locked(key, max_bytes=max_bytes)

    def _compress_resident_locked(self, key: Any, *, max_bytes: int) -> None:
        if not super().__contains__(key):
            return
        signature, data = OrderedDict.__getitem__(self, key)
        OrderedDict.__delitem__(self, key)
        tick = self._resident_ticks.pop(key, None)
        if tick is None:
            tick = self._budget.next_tick()
        backing = self._project_backings.get(key)
        if backing is not None and backing.signature == signature and backing.available():
            self._disk[key] = backing
            return
        if max_bytes > 0 and isinstance(data, (MDHistoData, PointListData)):
            artifact = CompressedBinning.from_data(data, max_bytes=max_bytes)
            if artifact is not None:
                self._compressed[key] = (signature, artifact)
                self._compressed_bytes += artifact.nbytes
                self._compressed_ticks[key] = tick
                return
        self._labels.pop(key, None)

    def _discard_compressed(self, key: Any) -> None:
        with self._decode_lock:
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
                backing = self._project_backings.get(key)
                if (
                    backing is not None
                    and backing.signature == signature
                    and backing.available()
                ):
                    self._disk[key] = backing
                else:
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

        with self._decode_lock:
            super().clear()
            self._compressed.clear()
            self._compressed_bytes = 0
            self._resident_ticks.clear()
            self._compressed_ticks.clear()
            self._decoded.clear()
            self.clear_disk_cache()
            self._project_backings.clear()
            self._labels.clear()
