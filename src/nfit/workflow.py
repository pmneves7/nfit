"""Dependency-aware, human-readable workflow script generation.

The workflow graph records scientific operations and stable object IDs, not GUI
events.  Renderers select the dependency closure of one target and emit ordinary
Python using public nfit APIs.  Dataset reconstruction is the first supported
vertical slice; analysis, fit, and plot nodes will use the same graph contract.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pprint import pformat
from typing import Any

from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from .project_io import NfitProject

WORKFLOW_SCHEMA_VERSION = 1
_NODE_STAGES = {
    "source_dataset": "load_sources",
    "prepared_dataset": "prepare_datasets",
    "analysis": "run_analyses",
    "fit": "run_fits",
    "posterior": "run_fits",
    "model_evaluation": "run_fits",
    "plot": "make_plots",
}


class WorkflowValidationError(ValueError):
    """Raised when a workflow graph is incomplete, cyclic, or ambiguous."""


@dataclass(frozen=True)
class WorkflowOutput:
    """One stable object produced by a workflow node."""

    key: str
    object_id: str
    value_kind: str


@dataclass(frozen=True)
class WorkflowNode:
    """One versioned scientific operation in a workflow graph."""

    id: str
    kind: str
    operation: str
    operation_version: int = 1
    dependencies: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[WorkflowOutput, ...] = ()

    @property
    def stage(self) -> str:
        """Return the generated-script stage containing this operation."""

        try:
            return _NODE_STAGES[self.kind]
        except KeyError as exc:
            raise WorkflowValidationError(
                f"workflow node {self.id!r} has unknown kind {self.kind!r}"
            ) from exc


@dataclass(frozen=True)
class WorkflowPlan:
    """A validated dependency graph and one or more selected target nodes."""

    nodes: tuple[WorkflowNode, ...]
    targets: tuple[str, ...]
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate IDs, dependencies, JSON-safe config, outputs, and cycles."""

        if self.schema_version != WORKFLOW_SCHEMA_VERSION:
            raise WorkflowValidationError(
                f"unsupported workflow schema version {self.schema_version}"
            )
        by_id: dict[str, WorkflowNode] = {}
        output_ids: dict[str, str] = {}
        for node in self.nodes:
            if not node.id:
                raise WorkflowValidationError("workflow node IDs cannot be empty")
            if node.id in by_id:
                raise WorkflowValidationError(f"duplicate workflow node ID {node.id!r}")
            if node.operation_version < 1:
                raise WorkflowValidationError(
                    f"workflow node {node.id!r} has invalid operation version"
                )
            _ = node.stage
            try:
                json.dumps(node.config)
            except (TypeError, ValueError) as exc:
                raise WorkflowValidationError(
                    f"workflow node {node.id!r} has non-JSON configuration"
                ) from exc
            by_id[node.id] = node
            for output in node.outputs:
                if output.object_id in output_ids:
                    raise WorkflowValidationError(
                        f"workflow output object ID {output.object_id!r} is produced "
                        f"by both {output_ids[output.object_id]!r} and {node.id!r}"
                    )
                output_ids[output.object_id] = node.id
        for target in self.targets:
            if target not in by_id:
                raise WorkflowValidationError(f"unknown workflow target {target!r}")
        for node in self.nodes:
            missing = [dependency for dependency in node.dependencies if dependency not in by_id]
            if missing:
                raise WorkflowValidationError(
                    f"workflow node {node.id!r} has missing dependencies: "
                    + ", ".join(repr(item) for item in missing)
                )
        self.topological_nodes()

    def dependency_closure(self) -> set[str]:
        """Return target node IDs and every transitive dependency."""

        by_id = {node.id: node for node in self.nodes}
        missing_targets = [target for target in self.targets if target not in by_id]
        if missing_targets:
            raise WorkflowValidationError(
                "unknown workflow target(s): "
                + ", ".join(repr(item) for item in missing_targets)
            )
        selected: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in selected:
                return
            selected.add(node_id)
            for dependency in by_id[node_id].dependencies:
                if dependency not in by_id:
                    raise WorkflowValidationError(
                        f"workflow node {node_id!r} has missing dependency {dependency!r}"
                    )
                visit(dependency)

        for target in self.targets:
            visit(target)
        return selected

    def topological_nodes(self) -> tuple[WorkflowNode, ...]:
        """Return the selected dependency closure in stable execution order."""

        by_id = {node.id: node for node in self.nodes}
        selected = self.dependency_closure()
        state: dict[str, int] = {}
        ordered: list[WorkflowNode] = []

        def visit(node_id: str, trail: tuple[str, ...]) -> None:
            status = state.get(node_id, 0)
            if status == 2:
                return
            if status == 1:
                cycle = " -> ".join((*trail, node_id))
                raise WorkflowValidationError(f"workflow dependency cycle: {cycle}")
            state[node_id] = 1
            node = by_id[node_id]
            for dependency in node.dependencies:
                if dependency in selected:
                    visit(dependency, (*trail, node_id))
            state[node_id] = 2
            ordered.append(node)

        for target in self.targets:
            visit(target, ())
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation used by tests and future bundles."""

        return {
            "schema_version": self.schema_version,
            "targets": list(self.targets),
            "nodes": [
                {
                    **asdict(node),
                    "dependencies": list(node.dependencies),
                    "outputs": [asdict(output) for output in node.outputs],
                }
                for node in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkflowPlan:
        """Restore and validate a plan from :meth:`to_dict` output."""

        plan = cls(
            nodes=tuple(
                WorkflowNode(
                    id=str(node["id"]),
                    kind=str(node["kind"]),
                    operation=str(node["operation"]),
                    operation_version=int(node.get("operation_version", 1)),
                    dependencies=tuple(str(item) for item in node.get("dependencies", [])),
                    config=dict(node.get("config", {})),
                    outputs=tuple(
                        WorkflowOutput(
                            key=str(output["key"]),
                            object_id=str(output["object_id"]),
                            value_kind=str(output["value_kind"]),
                        )
                        for output in node.get("outputs", [])
                    ),
                )
                for node in payload.get("nodes", [])
            ),
            targets=tuple(str(item) for item in payload.get("targets", [])),
            schema_version=int(payload.get("schema_version", 0)),
        )
        plan.validate()
        return plan


def dataset_workflow_plan(project: NfitProject, dataset_id: str) -> WorkflowPlan:
    """Build the dependency graph required to reconstruct one ordinary dataset.

    The first implementation slice supports source-backed, non-composite
    datasets plus dataset-attached powder backgrounds.  Unsupported derived or
    grouped-reduction sources fail explicitly rather than producing a script
    that only appears reproducible.
    """

    group, dataset = _find_dataset(project, dataset_id)
    source_datasets = _dataset_background_closure(group, dataset)
    nodes: list[WorkflowNode] = []
    for source_dataset in source_datasets:
        nodes.append(_source_node(group, source_dataset))
    target_dependencies = [f"source:{dataset.id}"]
    target_dependencies.extend(
        f"source:{background.source_dataset_id}"
        for background in dataset.backgrounds
        if background.enabled
    )
    nodes.append(
        WorkflowNode(
            id=f"dataset:{dataset.id}",
            kind="prepared_dataset",
            operation="dataset_for_slice_viewer",
            dependencies=tuple(dict.fromkeys(target_dependencies)),
            config={
                "dataset_id": dataset.id,
                "inherited_masks": [
                    _mask_spec(mask) for mask in _effective_masks(group, dataset)
                ],
            },
            outputs=(WorkflowOutput("data", dataset.id, "prepared_dataset"),),
        )
    )
    plan = WorkflowPlan(tuple(nodes), (f"dataset:{dataset.id}",))
    plan.validate()
    return plan


def render_workflow_script(
    plan: WorkflowPlan,
    *,
    source_root: str | Path | None = None,
) -> str:
    """Render a readable, GUI-free Python script for a supported plan."""

    plan.validate()
    ordered = plan.topological_nodes()
    unsupported = [
        node
        for node in ordered
        if node.kind not in {"source_dataset", "prepared_dataset"}
    ]
    if unsupported:
        kinds = ", ".join(sorted({node.kind for node in unsupported}))
        raise NotImplementedError(
            f"workflow script rendering is not implemented for: {kinds}"
        )
    source_nodes = [node for node in ordered if node.kind == "source_dataset"]
    prepared_nodes = [node for node in ordered if node.kind == "prepared_dataset"]
    root = _source_root(source_nodes, source_root)
    source_specs = {
        node.config["dataset_id"]: {
            **copy.deepcopy(node.config),
            "path": _path_for_script(node.config["path"], root),
        }
        for node in source_nodes
    }
    prepared_specs = {
        node.config["dataset_id"]: copy.deepcopy(node.config)
        for node in prepared_nodes
    }
    target_ids = [
        output.object_id
        for node in prepared_nodes
        if node.id in plan.targets
        for output in node.outputs
        if output.value_kind == "prepared_dataset"
    ]
    expected_version = _nfit_version()
    return _dataset_script_text(
        expected_version=expected_version,
        root=root,
        source_specs=source_specs,
        prepared_specs=prepared_specs,
        target_ids=target_ids,
    )


def dataset_workflow_script(
    project: NfitProject,
    dataset_id: str,
    *,
    source_root: str | Path | None = None,
) -> str:
    """Return an editable script rebuilding one dataset from source files."""

    return render_workflow_script(
        dataset_workflow_plan(project, dataset_id),
        source_root=source_root,
    )


def _source_node(group: DataGroup, dataset: DatasetEntry) -> WorkflowNode:
    if dataset.metadata.get("derived_from_analysis"):
        raise WorkflowValidationError(
            f"dataset {dataset.name!r} is derived from an analysis; export the "
            "analysis workflow instead"
        )
    source = str(dataset.metadata.get("source_file", "")).strip()
    if not source:
        raise WorkflowValidationError(
            f"dataset {dataset.name!r} has no reproducible source_file"
        )
    parent = _dataset_parent(group, dataset)
    grouped_reduction = dataset.kind in {"raw_dgs_nexus", "mdevent"} or (
        isinstance(parent, DatasetGroup)
        and any(key in parent.metadata for key in ("raw_dgs", "mdevent"))
    )
    if grouped_reduction:
        raise WorkflowValidationError(
            f"dataset {dataset.name!r} belongs to a grouped reduction; grouped "
            "raw-DGS and MDEvent workflow rendering is not implemented yet"
        )
    metadata = {
        str(key): copy.deepcopy(value)
        for key, value in dataset.metadata.items()
        if key
        not in {
            "source_file",
            "import_status",
            "importer",
            "import_options",
            "export_file",
        }
    }
    path = Path(source).expanduser()
    fingerprint = None
    if path.exists():
        stat = path.stat()
        fingerprint = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    config = {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "path": source,
        "kind": dataset.kind,
        "data_type": dataset.data_type,
        "importer": dataset.metadata.get("importer"),
        "import_options": copy.deepcopy(dataset.metadata.get("import_options")),
        "metadata": metadata,
        "parameters": copy.deepcopy(dataset.parameters),
        "enabled": bool(dataset.enabled),
        "fit_weight": float(dataset.fit_weight),
        "scale_factor": float(dataset.scale_factor),
        "scale_factor_vary": bool(dataset.scale_factor_vary),
        "masks": [_mask_spec(mask) for mask in dataset.masks],
        "backgrounds": [_background_spec(item) for item in dataset.backgrounds],
        "source_fingerprint": fingerprint,
    }
    return WorkflowNode(
        id=f"source:{dataset.id}",
        kind="source_dataset",
        operation="dataset_entry_from_path",
        config=config,
        outputs=(
            WorkflowOutput(
                "dataset",
                f"source-entry:{dataset.id}",
                "dataset_entry",
            ),
        ),
    )


def _find_dataset(
    project: NfitProject, dataset_id: str
) -> tuple[DataGroup, DatasetEntry]:
    for group in project.data_groups:
        for dataset in group.iter_datasets():
            if dataset.id == dataset_id:
                return group, dataset
    raise KeyError(f"unknown dataset ID {dataset_id!r}")


def _dataset_background_closure(
    group: DataGroup, target: DatasetEntry
) -> list[DatasetEntry]:
    by_id = {dataset.id: dataset for dataset in group.iter_datasets()}
    ordered: list[DatasetEntry] = []
    state: dict[str, int] = {}

    def visit(dataset: DatasetEntry, trail: tuple[str, ...]) -> None:
        status = state.get(dataset.id, 0)
        if status == 2:
            return
        if status == 1:
            names = " -> ".join((*trail, dataset.name))
            raise WorkflowValidationError(f"background dependency cycle: {names}")
        state[dataset.id] = 1
        for background in dataset.backgrounds:
            if not background.enabled:
                continue
            source = by_id.get(background.source_dataset_id)
            if source is None:
                raise WorkflowValidationError(
                    f"background {background.name!r} on dataset {dataset.name!r} "
                    "refers to a missing dataset"
                )
            visit(source, (*trail, dataset.name))
        state[dataset.id] = 2
        ordered.append(dataset)

    visit(target, ())
    return ordered


def _dataset_parent(group: DataGroup, target: DatasetEntry) -> DataGroup | DatasetGroup | None:
    def visit(node: DataGroup | DatasetGroup):
        if target in node.datasets:
            return node
        for subgroup in node.subgroups:
            found = visit(subgroup)
            if found is not None:
                return found
        return None

    return visit(group)


def _effective_masks(group: DataGroup, target: DatasetEntry) -> list[MaskSpec]:
    def visit(node: DataGroup | DatasetGroup) -> list[MaskSpec] | None:
        if target in node.datasets:
            return list(node.masks)
        for subgroup in node.subgroups:
            found = visit(subgroup)
            if found is not None:
                return [*node.masks, *found]
        return None

    return visit(group) or []


def _mask_spec(mask: MaskSpec) -> dict[str, Any]:
    return {
        "name": mask.name,
        "type": mask.type,
        "parameters": copy.deepcopy(mask.parameters),
        "enabled": bool(mask.enabled),
        "invert": bool(mask.invert),
        "additive": bool(mask.additive),
        "metadata": copy.deepcopy(mask.metadata),
    }


def _background_spec(background: BackgroundSpec) -> dict[str, Any]:
    return {
        "name": background.name,
        "source_dataset_id": background.source_dataset_id,
        "scale": float(background.scale),
        "enabled": bool(background.enabled),
        "interpolation": background.interpolation,
        "metadata": copy.deepcopy(background.metadata),
    }


def _source_root(
    source_nodes: list[WorkflowNode], requested: str | Path | None
) -> Path:
    if requested is not None:
        return Path(requested).expanduser().resolve()
    paths = [
        Path(str(node.config["path"])).expanduser().resolve()
        for node in source_nodes
    ]
    if not paths:
        return Path.cwd()
    common = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return common


def _path_for_script(path: str, root: Path) -> str:
    source = Path(path).expanduser()
    try:
        return str(source.resolve().relative_to(root))
    except ValueError:
        return str(source.resolve())


def _nfit_version() -> str:
    try:
        return version("nfit")
    except PackageNotFoundError:
        return "unknown"


def _dataset_script_text(
    *,
    expected_version: str,
    root: Path,
    source_specs: dict[str, dict[str, Any]],
    prepared_specs: dict[str, dict[str, Any]],
    target_ids: list[str],
) -> str:
    source_text = pformat(source_specs, sort_dicts=False, width=96)
    prepared_text = pformat(prepared_specs, sort_dicts=False, width=96)
    targets_text = pformat(target_ids, sort_dicts=False, width=96)
    return f'''"""Rebuild an nfit dataset workflow from its original source files.

Generated configuration is ordinary Python: edit SOURCE_ROOT, paths, masks,
rebin settings, scales, or other dictionaries before running as needed.
"""

from __future__ import annotations

import copy
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nfit import (
    BackgroundSpec,
    MaskSpec,
    dataset_entry_from_path,
    dataset_for_slice_viewer,
)

EXPECTED_NFIT_VERSION = {expected_version!r}
SOURCE_ROOT = Path({str(root)!r})

SOURCE_DATASETS = {source_text}

PREPARED_DATASETS = {prepared_text}

TARGET_DATASET_IDS = {targets_text}


def check_nfit_version():
    """Warn when the runtime differs from the version that wrote this script."""
    try:
        installed = version("nfit")
    except PackageNotFoundError:
        installed = "unknown"
    if EXPECTED_NFIT_VERSION != "unknown" and installed != EXPECTED_NFIT_VERSION:
        warnings.warn(
            f"Generated with nfit {{EXPECTED_NFIT_VERSION}}, running with {{installed}}.",
            stacklevel=2,
        )


def source_path(path_text):
    """Resolve one editable source path against SOURCE_ROOT."""
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else SOURCE_ROOT / path


def validate_source(path, expected):
    """Warn if a source file changed since script generation."""
    if not expected:
        return
    stat = path.stat()
    current = {{"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}}
    if current != expected:
        warnings.warn(f"Source file changed: {{path}}", stacklevel=2)


def load_sources():
    """Load source-backed DatasetEntry objects without opening the GUI."""
    datasets = {{}}
    for dataset_id, spec in SOURCE_DATASETS.items():
        path = source_path(spec["path"])
        validate_source(path, spec.get("source_fingerprint"))
        entry = dataset_entry_from_path(
            path,
            data_type=spec["data_type"],
            importer_name=spec.get("importer"),
            importer_options=copy.deepcopy(spec.get("import_options")),
        )
        entry.id = dataset_id
        entry.name = spec["name"]
        entry.kind = spec["kind"]
        entry.metadata.update(copy.deepcopy(spec["metadata"]))
        entry.parameters = copy.deepcopy(spec["parameters"])
        entry.enabled = bool(spec["enabled"])
        entry.fit_weight = float(spec["fit_weight"])
        entry.scale_factor = float(spec["scale_factor"])
        entry.scale_factor_vary = bool(spec["scale_factor_vary"])
        entry.masks = [MaskSpec(**copy.deepcopy(item)) for item in spec["masks"]]
        entry.backgrounds = [
            BackgroundSpec(**copy.deepcopy(item)) for item in spec["backgrounds"]
        ]
        datasets[dataset_id] = entry
    for entry in datasets.values():
        for background in entry.backgrounds:
            background.source_entry = datasets.get(background.source_dataset_id)
    return datasets


def prepare_datasets(datasets):
    """Apply masks, rebinning, backgrounds, scales, and channel conversions."""
    prepared = {{}}
    for dataset_id, spec in PREPARED_DATASETS.items():
        inherited_masks = [
            MaskSpec(**copy.deepcopy(item)) for item in spec["inherited_masks"]
        ]
        prepared[dataset_id] = dataset_for_slice_viewer(
            datasets[dataset_id],
            extra_masks=inherited_masks,
            force_rebin=True,
            force_masks=True,
        )
    return prepared


def build_workflow():
    """Run every stage needed by the selected target dataset."""
    check_nfit_version()
    datasets = load_sources()
    prepared = prepare_datasets(datasets)
    return datasets, prepared


def main():
    _datasets, prepared = build_workflow()
    for dataset_id in TARGET_DATASET_IDS:
        data = prepared[dataset_id]
        shape = getattr(data, "shape", (getattr(data, "size", 0),))
        print(f"Prepared {{dataset_id}} with shape {{shape}}")


if __name__ == "__main__":
    main()
'''


def workflow_variable_name(label: str, *, fallback: str = "item") -> str:
    """Return a readable Python identifier for future explicit renderers."""

    value = re.sub(r"\W+", "_", str(label).strip()).strip("_").lower()
    if not value or value[0].isdigit():
        value = f"{fallback}_{value}" if value else fallback
    return value
