"""Dataset catalog and on-demand payload sequence for interactive viewers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, overload

from .dataset import PointListData
from .mdhisto import MDHistoData

ViewerDataset = MDHistoData | PointListData


class ViewerLoadCancelled(Exception):
    """A deferred viewer load stopped at the user's request."""


@dataclass(frozen=True)
class ViewerDatasetDescriptor:
    """Lightweight catalog entry for a viewer dataset payload."""

    name: str
    source_dataset_name: str = ""
    binning_name: str = "Default"
    binning_id: str = "fit"
    group_key: str = ""
    crystal_context: Mapping[str, Any] = field(default_factory=dict)
    source_dataset_id: str = ""
    source_scope_id: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "source_dataset_name",
            str(self.source_dataset_name or self.name),
        )
        object.__setattr__(self, "binning_name", str(self.binning_name or "Default"))
        object.__setattr__(self, "binning_id", str(self.binning_id or "fit"))
        object.__setattr__(self, "group_key", str(self.group_key))
        object.__setattr__(self, "crystal_context", dict(self.crystal_context))
        object.__setattr__(self, "source_dataset_id", str(self.source_dataset_id))


class DeferredViewerDatasets(Sequence[ViewerDataset]):
    """A descriptor-backed sequence that materializes payloads on indexing.

    ``loader`` is called once per index until that index is explicitly released.
    Catalog inspection through :attr:`descriptors` never invokes it.
    """

    def __init__(
        self,
        descriptors: Sequence[ViewerDatasetDescriptor],
        loader: Callable[[int], ViewerDataset],
        *,
        initial_index: int = 0,
        revision: object | None = None,
        release_callback: Callable[[int | None], None] | None = None,
    ) -> None:
        self._descriptors = tuple(descriptors)
        if not self._descriptors:
            raise ValueError("DeferredViewerDatasets requires at least one descriptor")
        if not 0 <= int(initial_index) < len(self._descriptors):
            raise IndexError("initial_index is outside the dataset catalog")
        self._loader = loader
        self._initial_index = int(initial_index)
        self._revision = revision if revision is not None else object()
        self._cache: dict[int, ViewerDataset] = {}
        self._lock = RLock()
        self._load_locks = [RLock() for _ in self._descriptors]
        self._generations = [0 for _ in self._descriptors]
        self._release_callback = release_callback

    @property
    def descriptors(self) -> tuple[ViewerDatasetDescriptor, ...]:
        return self._descriptors

    @property
    def initial_index(self) -> int:
        return self._initial_index

    @property
    def revision(self) -> object:
        return self._revision

    @property
    def cached_indices(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._cache))

    def cached_items(self) -> tuple[tuple[int, ViewerDataset], ...]:
        """Return loaded payloads without accessing any unloaded catalog entry."""

        with self._lock:
            return tuple(sorted(self._cache.items()))

    def release(self, index: int | None = None) -> None:
        """Release this catalog's references, retaining descriptors and its loader.

        A callback lets an owning wrapper release its underlying catalog too.
        Existing viewers or other catalogs may still hold the immutable data.
        """

        normalized = None if index is None else self._normalize_index(index)
        with self._lock:
            if normalized is None:
                self._cache.clear()
                self._generations = [value + 1 for value in self._generations]
            else:
                self._cache.pop(normalized, None)
                self._generations[normalized] += 1
        if self._release_callback is not None:
            self._release_callback(normalized)

    def is_loaded(self, index: int) -> bool:
        with self._lock:
            return self._normalize_index(index) in self._cache

    def cached(self, index: int) -> ViewerDataset | None:
        with self._lock:
            return self._cache.get(self._normalize_index(index))

    def cache_identity(self) -> tuple[object, tuple[tuple[int, int], ...]]:
        """Identify the catalog and loaded payloads without loading new ones."""

        return self._revision, tuple(
            (index, id(payload)) for index, payload in self.cached_items()
        )

    def __len__(self) -> int:
        return len(self._descriptors)

    @overload
    def __getitem__(self, index: int) -> ViewerDataset: ...

    @overload
    def __getitem__(self, index: slice) -> list[ViewerDataset]: ...

    def __getitem__(self, index: int | slice) -> ViewerDataset | list[ViewerDataset]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = self._normalize_index(index)
        with self._lock:
            cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        with self._load_locks[normalized]:
            with self._lock:
                cached = self._cache.get(normalized)
                generation = self._generations[normalized]
            if cached is not None:
                return cached
            payload = self._loader(normalized)
            if not isinstance(payload, (MDHistoData, PointListData)):
                raise TypeError("viewer dataset loader must return MDHistoData or PointListData")
            with self._lock:
                if self._generations[normalized] == generation:
                    self._cache[normalized] = payload
            return payload

    def __iter__(self) -> Iterator[ViewerDataset]:
        for index in range(len(self)):
            yield self[index]

    def _normalize_index(self, index: int) -> int:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError("viewer dataset index out of range")
        return normalized
