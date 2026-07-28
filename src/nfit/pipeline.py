from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from .dataset import PointData4D
from .fitting import (
    FitDataset,
    FitProblem,
    FitResult,
    ModelFunction,
    ModelSpec,
    OptimizationConfig,
    ParameterBinding,
    ParameterSpec,
    fit_problem_least_squares,
)

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

    ``source_dataset_id`` is the stable project reference. ``source_entry`` is
    relinked at runtime after loading and is deliberately omitted from project
    serialization.
    """

    name: str
    source_dataset_id: str
    scale: float = 1.0
    enabled: bool = True
    interpolation: str = "linear"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_entry: DatasetEntry | None = field(default=None, repr=False, compare=False)


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
    models: dict[str, FitModelSession | ModelComponentSpec] = field(default_factory=dict)
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

    def add_model(self, model: FitModelSession) -> None:
        """Associate a model session with this group."""

        if model.name in self.models:
            raise ValueError(f"duplicate model name {model.name!r}")
        self.models[model.name] = model

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


@dataclass
class FitHistoryEntry:
    """One completed optimization attempt in a model session."""

    index: int
    result: FitResult
    parameter_specs_before: tuple[ParameterSpec, ...]
    parameter_specs_after: tuple[ParameterSpec, ...]
    dataset_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitModelSession:
    """Model, fit settings, dataset overrides, and iterative fit history."""

    name: str
    model: ModelSpec | ModelFunction
    parameter_specs: list[ParameterSpec]
    optimizer: OptimizationConfig = field(default_factory=OptimizationConfig)
    resolution_by_dataset: dict[str, Any] = field(default_factory=dict)
    parameter_bindings_by_dataset: dict[str, dict[str, ParameterBinding]] = field(
        default_factory=dict
    )
    weights_by_dataset: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[FitHistoryEntry] = field(default_factory=list)

    def build_problem(
        self,
        data_group: DataGroup,
        *,
        dataset_names: Iterable[str] | None = None,
    ) -> FitProblem:
        """Build a :class:`FitProblem` from selected point-data entries."""

        selected = data_group.select(dataset_names)
        fit_datasets: list[FitDataset] = []
        for entry in selected:
            weight = float(self.weights_by_dataset.get(entry.name, entry.fit_weight))
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"dataset {entry.name!r} fit weight must be finite and non-negative"
                )
            if weight == 0.0:
                continue
            prepared = entry.prepared()
            if not isinstance(prepared, PointData4D):
                raise TypeError(
                    f"dataset {entry.name!r} prepared to {type(prepared).__name__}; "
                    "FitModelSession currently requires PointData4D"
                )
            scale = float(getattr(entry, "scale_factor", 1.0) or 1.0)
            if scale != 1.0:
                prepared = replace(
                    prepared,
                    intensity=np.asarray(prepared.intensity, dtype=float) * scale,
                    sigma=np.asarray(prepared.sigma, dtype=float) * abs(scale),
                )
            fit_datasets.append(
                FitDataset(
                    name=entry.name,
                    data=prepared,
                    weight=weight,
                    resolution=self._dataset_lookup(self.resolution_by_dataset, entry.name),
                    parameter_bindings=self.parameter_bindings_by_dataset.get(entry.name, {}),
                    metadata={**entry.metadata, "parameters": dict(entry.parameters)},
                )
            )
        if not fit_datasets:
            raise ValueError("no enabled positive-weight dataset is available for fitting")
        return FitProblem(
            datasets=fit_datasets,
            model=self.model,
            parameter_specs=tuple(self.parameter_specs),
            description=self.name,
            metadata=dict(self.metadata),
        )

    def fit(
        self,
        data_group: DataGroup,
        *,
        dataset_names: Iterable[str] | None = None,
        update_parameters: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> FitResult:
        """Run one fit, append it to history, and optionally update parameters."""

        before = tuple(self.parameter_specs)
        problem = self.build_problem(data_group, dataset_names=dataset_names)
        selected_names = tuple(dataset.name for dataset in problem.datasets)
        result = fit_problem_least_squares(problem, config=self.optimizer)
        after = (
            tuple(_updated_parameter_specs(before, result.params))
            if update_parameters
            else before
        )
        if update_parameters:
            self.parameter_specs = list(after)
        self.history.append(
            FitHistoryEntry(
                index=len(self.history),
                result=result,
                parameter_specs_before=before,
                parameter_specs_after=after,
                dataset_names=selected_names,
                metadata={} if metadata is None else dict(metadata),
            )
        )
        return result

    def rollback(self, history_index: int, *, use_after: bool = True) -> None:
        """Restore parameter values from a previous history entry."""

        entry = self.history[history_index]
        specs = entry.parameter_specs_after if use_after else entry.parameter_specs_before
        self.parameter_specs = list(specs)

    def current_parameters(self) -> dict[str, float]:
        """Return current parameter values keyed by name."""

        return {spec.name: float(spec.value) for spec in self.parameter_specs}

    @staticmethod
    def _dataset_lookup(mapping: dict[str, Any], dataset_name: str) -> Any:
        if dataset_name in mapping:
            return mapping[dataset_name]
        return mapping.get("*")


def _updated_parameter_specs(
    specs: Sequence[ParameterSpec],
    params: dict[str, float],
) -> list[ParameterSpec]:
    return [replace(spec, value=float(params.get(spec.name, spec.value))) for spec in specs]
