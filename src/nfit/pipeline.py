from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .analysis.core import AnalysisEntry


DataTransformAny = Callable[[Any], Any]


@dataclass
class MaskSpec:
    """Serializable analysis mask configuration attached to a dataset.

    ``type`` names a registered mask kind and ``parameters`` stores only
    JSON-like values needed to rebuild the mask in a script or GUI.
    """

    name: str
    type: str = "coordinate_range"
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    invert: bool = False
    additive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundSpec:
    """Serializable scaled background attached to a dataset.

    ``source_dataset_id`` or ``source_group_id`` is the stable project
    reference. Runtime links are restored after loading and deliberately
    omitted from project serialization. ``projection`` is ``"center"`` for
    direct powder interpolation or ``"sample_trajectories"`` for MDEvent
    powder backgrounds forward-projected through the sample acceptance, or
    ``"measured_events"`` to replay measured lab-frame background events at
    the sample angles while retaining detector-direction dependence.
    """

    name: str
    source_dataset_id: str = ""
    scale: float = 1.0
    enabled: bool = True
    interpolation: str = "linear"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_group_id: str | None = None
    source_entry: DatasetEntry | None = field(default=None, repr=False, compare=False)
    source_group: DatasetGroup | None = field(default=None, repr=False, compare=False)
    projection: str = "center"


@dataclass
class ModelComponentSpec:
    """Serializable model component configuration attached to a data group.

    Fit-flexibility fields
    ----------------------
    ``fit_parameters`` marks which parameters the optimizer may vary.
    ``sharing`` maps a parameter name to ``{"mode": ..., "groups": {...}}``
    where mode is ``"global"`` (one shared value), ``"per_dataset"`` (one
    independent value per dataset), or ``"grouped"`` (datasets sharing a tie
    key in ``groups`` share one value; unlisted datasets get their own).
    Parameters without a ``sharing`` entry default to global sharing.
    ``limits`` maps a parameter name to ``[min, max]`` bounds where either side
    may be ``None``. ``constraints``
    holds inequalities such as ``{"parameter": "c0", "op": ">=",
    "reference": ...}`` or exact derived relationships such as
    ``{"parameter": "c0", "op": "=", "expression": "10 - `other.c0`"}``.
    ``applies_to`` restricts the component to the named datasets
    (``None`` means all compatible datasets).
    """

    name: str
    type: str = "constant_background"
    parameters: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    fit_parameters: dict[str, bool] = field(default_factory=dict)
    sharing: dict[str, dict[str, Any]] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    applies_to: list[str] | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitTimelineEntry:
    """Serializable GUI fit-history node for one data group.

    ``channels`` stores per-dataset fitted-model channels computed when the
    fit ran, keyed by dataset name. Each entry holds ``"fit"`` and
    ``"residual"`` float arrays aligned with the dataset's view (grid-shaped
    for MDHisto data, one value per point for point lists) so viewers never
    need to re-evaluate a potentially expensive model.
    """

    name: str
    kind: str = "result"
    snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    duration_seconds: float | None = None
    optimizer: str = "least_squares"
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    goodness: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    children: list[FitTimelineEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class PlotSourceRef:
    """A stable project object used by a saved plotting recipe."""

    dataset_id: str | None = None
    fit_id: str | None = None


@dataclass
class PlotEntry:
    """A serializable, GUI-independent figure recipe owned by a workspace."""

    name: str
    type: str = "mdhisto_slice"
    sources: list[PlotSourceRef] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    renderer_version: int = 1
    source_fingerprints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class DatasetEntry:
    """One dataset plus flexible experimental metadata.

    ``data`` may be point data, MDHisto data, bulk susceptibility, or a future
    domain-specific container. Fitting currently requires entries whose prepared
    data are :class:`PointData4D`.
    """

    name: str
    data: Any
    kind: str = ""
    data_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    masks: list[MaskSpec] = field(default_factory=list)
    backgrounds: list[BackgroundSpec] = field(default_factory=list)
    enabled: bool = True
    fit_weight: float = 1.0
    scale_factor: float = 1.0
    scale_factor_vary: bool = False
    scale_factor_group: str | None = None
    transforms: Sequence[DataTransformAny] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: uuid4().hex)
    _data_revision: int = field(default=0, init=False, repr=False, compare=False)
    _data_matches_source: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _source_data_identity: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.data = _immutable_data(self.data)
        self._data_matches_source = bool(self.metadata.get("source_file"))
        self._source_data_identity = (
            id(self.data)
            if self.data is not None and self._data_matches_source
            else None
        )

    def prepared(self) -> Any:
        """Return data after applying this dataset's transforms."""

        prepared = self.data
        for transform in self.transforms:
            prepared = transform(prepared)
        return prepared

    def copy(self, **changes: Any) -> DatasetEntry:
        """Return an independent entry with a new stable identifier."""

        changes.pop("id", None)
        copied = replace(self, id=uuid4().hex, **changes)
        if "data" in changes:
            copied._data_matches_source = False
            copied._source_data_identity = None
        else:
            copied._data_matches_source = self.data_matches_source
            copied._source_data_identity = (
                id(copied.data)
                if copied.data is not None and copied._data_matches_source
                else None
            )
        return copied

    def __deepcopy__(self, memo: dict[int, Any]) -> DatasetEntry:
        """Deep-copy project configuration while preserving source provenance."""

        copied = object.__new__(type(self))
        memo[id(self)] = copied
        for item in fields(self):
            setattr(copied, item.name, copy.deepcopy(getattr(self, item.name), memo))
        copied.data = _immutable_data(copied.data)
        copied._data_matches_source = self.data_matches_source
        copied._source_data_identity = (
            id(copied.data)
            if copied.data is not None and copied._data_matches_source
            else None
        )
        return copied

    @property
    def data_revision(self) -> int:
        """Process-local revision incremented by :meth:`replace_data`."""

        return self._data_revision

    @property
    def data_matches_source(self) -> bool:
        """Whether the loaded data still represent ``metadata['source_file']``."""

        if not self._data_matches_source:
            return False
        return (
            self.data is None
            or (
                self._source_data_identity is not None
                and id(self.data) == self._source_data_identity
            )
        )

    @property
    def data_cache_token(self) -> tuple[int, int]:
        """Cheap cache token that also detects unsupported direct replacement."""

        return self._data_revision, id(self.data)

    def replace_data(self, data: Any, *, source_backed: bool = False) -> Any:
        """Install canonical immutable data and invalidate dependent caches.

        ``source_backed=True`` is reserved for importer and lazy-loading paths
        whose arrays exactly represent the current ``source_file``. Developer
        edits should keep the default ``False`` so analysis fingerprints hash
        the replacement arrays rather than the original file.
        """

        canonical = _immutable_data(data)
        self.data = canonical
        self._data_revision += 1
        self._data_matches_source = bool(source_backed)
        self._source_data_identity = (
            id(canonical)
            if canonical is not None and self._data_matches_source
            else None
        )
        return canonical

    def unload_data(self) -> None:
        """Drop loaded arrays while retaining the file-backed project entry."""

        self.replace_data(None, source_backed=True)


def _immutable_data(data: Any) -> Any:
    if data is None:
        return None
    converter = getattr(data, "immutable_copy", None)
    return converter() if callable(converter) else data


@dataclass
class DatasetGroup:
    """A nested group of datasets sharing masks, backgrounds, and configuration.

    Groups may nest via ``subgroups``. A disabled group and all of its
    descendants are omitted from fitting without changing the descendants'
    individual enabled states. ``masks`` on a group apply to every descendant
    dataset. ``backgrounds`` are applied once to the group's composite, after
    its enabled datasets have been combined. Fit weights and scale factors are
    *not* stored here; the GUI edits those in bulk across a group's descendant
    datasets.
    """

    name: str
    datasets: list[DatasetEntry] = field(default_factory=list)
    subgroups: list[DatasetGroup] = field(default_factory=list)
    enabled: bool = True
    masks: list[MaskSpec] = field(default_factory=list)
    backgrounds: list[BackgroundSpec] = field(default_factory=list)
    resolution: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)

    def iter_datasets(self) -> Iterator[DatasetEntry]:
        """Yield every dataset in this group and its nested subgroups."""

        yield from self.datasets
        for subgroup in self.subgroups:
            yield from subgroup.iter_datasets()

    def iter_subgroups(self) -> Iterator[DatasetGroup]:
        """Yield this group's subgroups recursively."""

        for subgroup in self.subgroups:
            yield subgroup
            yield from subgroup.iter_subgroups()


@dataclass
class DataGroup:
    """Collection of related datasets and shared sample metadata.

    ``datasets`` are the group's direct datasets; ``subgroups`` hold nested
    dataset groups. ``masks`` apply to every dataset in the group and its
    subgroups. Models and fit history live only at this top level.
    """

    name: str
    datasets: list[DatasetEntry] = field(default_factory=list)
    subgroups: list[DatasetGroup] = field(default_factory=list)
    masks: list[MaskSpec] = field(default_factory=list)
    backgrounds: list[BackgroundSpec] = field(default_factory=list)
    lattice_parameters: dict[str, float] | None = None
    spacegroup: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    models: dict[str, ModelComponentSpec] = field(default_factory=dict)
    fits: list[FitTimelineEntry] = field(default_factory=list)
    active_fit_path: list[int] | None = None
    analyses: list[AnalysisEntry] = field(default_factory=list)
    plots: list[PlotEntry] = field(default_factory=list)

    def iter_datasets(self) -> Iterator[DatasetEntry]:
        """Yield every dataset in this group and its nested subgroups."""

        yield from self.datasets
        for subgroup in self.subgroups:
            yield from subgroup.iter_datasets()

    def iter_subgroups(self) -> Iterator[DatasetGroup]:
        """Yield all nested dataset groups recursively."""

        for subgroup in self.subgroups:
            yield subgroup
            yield from subgroup.iter_subgroups()

    def add_dataset(self, dataset: DatasetEntry, *, into: DatasetGroup | None = None) -> None:
        """Add a dataset (optionally into a subgroup), requiring globally unique names."""

        if dataset.name in self.dataset_names:
            raise ValueError(f"duplicate dataset name {dataset.name!r}")
        (into or self).datasets.append(dataset)

    @property
    def dataset_names(self) -> list[str]:
        """All dataset names in the group tree, in traversal order."""

        return [dataset.name for dataset in self.iter_datasets()]

    def get_dataset(self, name: str) -> DatasetEntry:
        """Return a dataset by name from anywhere in the group tree."""

        for dataset in self.iter_datasets():
            if dataset.name == name:
                return dataset
        raise KeyError(f"unknown dataset {name!r}")

    def select(self, names: Iterable[str] | None = None) -> list[DatasetEntry]:
        """Return enabled datasets in requested order, or all enabled datasets when omitted."""

        if names is None:
            return [dataset for dataset in self.iter_datasets() if dataset.enabled]
        return [dataset for dataset in (self.get_dataset(name) for name in names) if dataset.enabled]

    def data_sequence(self, names: Iterable[str] | None = None) -> list[Any]:
        """Return raw data objects for viewer and plotting helpers."""

        return [entry.data for entry in self.select(names)]

    def prepared_sequence(self, names: Iterable[str] | None = None) -> list[Any]:
        """Return transformed data objects for analysis or plotting."""

        return [entry.prepared() for entry in self.select(names)]
