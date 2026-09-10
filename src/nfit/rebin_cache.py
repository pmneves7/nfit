"""Bounded in-memory binnings with temporary disk overflow for reuse and saving."""

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from .dataset import PointListData
from .mdhisto import MDHistoData


class RebinCache(OrderedDict):
    """Retain evicted numerical results until invalidated or explicitly cleared.

    Only resident arrays count against the existing LRU memory budget. Overflow
    uses uncompressed temporary artifacts to avoid compression work during a
    reduction. Clearing the cache removes its temporary files too.
    """

    def __init__(self):
        super().__init__()
        self._directory = None
        self._overflow = {}

    def __setitem__(self, key, value):
        previous = self._overflow.pop(key, None)
        if previous is not None:
            previous[1].unlink(missing_ok=True)
        super().__setitem__(key, value)

    def get(self, key, default=None):
        if key in self:
            return super().get(key, default)
        stored = self._overflow.get(key)
        if stored is None:
            return default
        signature, path = stored
        return signature, read_dataset_artifact(path)

    def has_signature(self, key, signature):
        """Check freshness without loading a potentially large disk payload."""
        resident = super().get(key)
        if resident is not None:
            return resident[0] == signature
        stored = self._overflow.get(key)
        return stored is not None and stored[0] == signature and stored[1].exists()

    def move_to_end(self, key, last=True):
        if key in self:
            super().move_to_end(key, last=last)

    def popitem(self, last=True):
        key = next(reversed(self)) if last else next(iter(self))
        signature, data = self[key]
        if isinstance(data, (MDHistoData, PointListData)):
            if self._directory is None:
                self._directory = TemporaryDirectory(prefix="nfit-rebin-overflow-")
            path = Path(self._directory.name) / f"{uuid4().hex}.npz"
            write_dataset_artifact(data, path, compressed=False)
            self._overflow[key] = (signature, path)
        return super().popitem(last=last)

    def clear(self):
        super().clear()
        self._overflow.clear()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
