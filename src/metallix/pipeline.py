from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

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


DataTransformAny = Callable[[Any], Any]


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
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    transforms: Sequence[DataTransformAny] = field(default_factory=tuple)

    def prepared(self) -> Any:
        """Return data after applying this dataset's transforms."""

        prepared = self.data
        for transform in self.transforms:
            prepared = transform(prepared)
        return prepared


@dataclass
class DataGroup:
    """Collection of related datasets and shared sample metadata."""

    name: str
    datasets: list[DatasetEntry] = field(default_factory=list)
    lattice_parameters: dict[str, float] | None = None
    spacegroup: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    models: dict[str, "FitModelSession"] = field(default_factory=dict)

    def add_dataset(self, dataset: DatasetEntry) -> None:
        """Add a dataset, requiring names to stay unique."""

        if dataset.name in self.dataset_names:
            raise ValueError(f"duplicate dataset name {dataset.name!r}")
        self.datasets.append(dataset)

    def add_model(self, model: "FitModelSession") -> None:
        """Associate a model session with this group."""

        if model.name in self.models:
            raise ValueError(f"duplicate model name {model.name!r}")
        self.models[model.name] = model

    @property
    def dataset_names(self) -> list[str]:
        """Dataset names in group order."""

        return [dataset.name for dataset in self.datasets]

    def get_dataset(self, name: str) -> DatasetEntry:
        """Return a dataset by name."""

        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"unknown dataset {name!r}")

    def select(self, names: Iterable[str] | None = None) -> list[DatasetEntry]:
        """Return datasets in requested order, or all datasets when omitted."""

        if names is None:
            return list(self.datasets)
        return [self.get_dataset(name) for name in names]

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
            prepared = entry.prepared()
            if not isinstance(prepared, PointData4D):
                raise TypeError(
                    f"dataset {entry.name!r} prepared to {type(prepared).__name__}; "
                    "FitModelSession currently requires PointData4D"
                )
            fit_datasets.append(
                FitDataset(
                    name=entry.name,
                    data=prepared,
                    weight=float(self.weights_by_dataset.get(entry.name, 1.0)),
                    resolution=self._dataset_lookup(self.resolution_by_dataset, entry.name),
                    parameter_bindings=self.parameter_bindings_by_dataset.get(entry.name, {}),
                    metadata={**entry.metadata, "parameters": dict(entry.parameters)},
                )
            )
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

        selected_names = tuple(entry.name for entry in data_group.select(dataset_names))
        before = tuple(self.parameter_specs)
        problem = self.build_problem(data_group, dataset_names=selected_names)
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
