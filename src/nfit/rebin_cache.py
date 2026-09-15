"""Bounded resident and compressed in-memory caches for computed binnings."""

import zlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .analysis.artifacts import _payload, dataset_artifact_from_payload
from .dataset import PointListData
from .mdhisto import MDHistoData

_CHUNK_BYTES = 8 * 1024**2


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
    ) -> "CompressedBinning | None":
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


class RebinCache(OrderedDict):
    """Keep recent arrays and older compressed results within RAM budgets.

    A GUI may inject before_discard to offer an explicit compressed NPZ save
    before the oldest compressed result is discarded. Scripting workflows use
    the bounded cache without Qt or a prompt.
    """

    def __init__(self):
        super().__init__()
        self._compressed: OrderedDict[Any, tuple[str, CompressedBinning]] = OrderedDict()
        self._compressed_bytes = 0
        self._compressed_limit = 0
        self._labels: dict[Any, str] = {}
        self.before_discard: Callable[[Any, CompressedBinning], None] | None = None

    def set_label(self, key: Any, label: str) -> None:
        """Associate a human-readable bin name with a process-local cache key."""

        self._labels[key] = label

    def configure_budget(self, total_bytes: int | None) -> int | None:
        """Reserve one quarter of the cache allowance for compressed entries."""

        if total_bytes is None:
            self._compressed_limit = 0
            return None
        total = max(int(total_bytes), 0)
        self._compressed_limit = total // 4
        return total - self._compressed_limit

    def __setitem__(self, key, value):
        old = self._compressed.pop(key, None)
        if old is not None:
            self._compressed_bytes -= old[1].nbytes
        super().__setitem__(key, value)

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().get(key, default)
        stored = self._compressed.get(key)
        if stored is None:
            return default
        self._compressed.move_to_end(key)
        signature, artifact = stored
        return signature, artifact.restore()

    def __contains__(self, key):
        return super().__contains__(key) or key in self._compressed

    def has_signature(self, key, signature):
        """Check freshness without decompressing a potentially large grid."""

        resident = super().get(key)
        if resident is not None:
            return resident[0] == signature
        stored = self._compressed.get(key)
        return stored is not None and stored[0] == signature

    def move_to_end(self, key, last=True):
        if super().__contains__(key):
            super().move_to_end(key, last=last)
        elif key in self._compressed:
            self._compressed.move_to_end(key, last=last)

    def popitem(self, last=True):
        key = next(reversed(self)) if last else next(iter(self))
        signature, data = self[key]
        result = super().popitem(last=last)
        if (
            self._compressed_limit > 0
            and isinstance(data, (MDHistoData, PointListData))
        ):
            artifact = CompressedBinning.from_data(
                data, max_bytes=self._compressed_limit
            )
            if artifact is not None:
                self._compressed[key] = (signature, artifact)
                self._compressed_bytes += artifact.nbytes
                while self._compressed_bytes > self._compressed_limit:
                    oldest_key, (_old_signature, oldest) = next(
                        iter(self._compressed.items())
                    )
                    if self.before_discard is not None:
                        self.before_discard(
                            self._labels.get(oldest_key, oldest_key), oldest
                        )
                    self._compressed.pop(oldest_key)
                    self._compressed_bytes -= oldest.nbytes
                    self._labels.pop(oldest_key, None)
            else:
                self._labels.pop(key, None)
        else:
            self._labels.pop(key, None)
        return result

    def clear(self):
        super().clear()
        self._compressed.clear()
        self._compressed_bytes = 0
        self._labels.clear()
