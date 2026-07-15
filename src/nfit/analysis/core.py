from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias
from uuid import uuid4

from ..dataset import PointData4D, PointListData
from ..mdhisto import MDHistoData


def new_analysis_id() -> str:
    """Return a stable, JSON-friendly analysis identifier."""

    return uuid4().hex


@dataclass
class AnalysisOutputRef:
    key: str
    label: str
    kind: str
    artifact_path: str | None = None
    dataset_id: str | None = None
    scalar_value: float | None = None
    scalar_uncertainty: float | None = None
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResultRecord:
    recipe_hash: str
    input_fingerprints: dict[str, str]
    outputs: list[AnalysisOutputRef]
    status: str
    created_at: str
    duration_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AnalysisEntry:
    name: str
    type: str
    input_dataset_ids: list[str]
    parameters: dict[str, Any]
    id: str = field(default_factory=new_analysis_id)
    operation_version: int = 1
    enabled: bool = True
    result: AnalysisResultRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisContext:
    data_group_name: str
    lattice_parameters: dict[str, float] | None
    spacegroup: str | None
    crystal: dict[str, Any] | None
    temperature_K: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


AnalysisData: TypeAlias = MDHistoData | PointListData | PointData4D


@dataclass
class AnalysisInput:
    dataset_id: str
    dataset_name: str
    data: AnalysisData
    context: AnalysisContext
    fingerprint: str


@dataclass
class DatasetOutput:
    data: AnalysisData
    label: str
    data_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableOutput:
    data: PointListData
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalarOutput:
    value: float
    uncertainty: float | None
    unit: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


AnalysisOutput: TypeAlias = DatasetOutput | TableOutput | ScalarOutput


@dataclass
class AnalysisExecution:
    outputs: dict[str, AnalysisOutput]
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
