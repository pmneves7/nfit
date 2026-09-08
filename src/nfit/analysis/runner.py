from __future__ import annotations

import hashlib
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..project_archive import replace_analysis_artifacts
from .artifacts import output_data, write_dataset_artifact
from .core import (
    AnalysisContext,
    AnalysisEntry,
    AnalysisInput,
    AnalysisOutputRef,
    AnalysisResultRecord,
    ScalarOutput,
)
from .fingerprint import dataset_entry_fingerprint, recipe_hash
from .registry import (
    analysis_definition,
    default_analysis_parameters,
    run_analysis_operation,
)

COMPOSITE_ANALYSIS_SOURCE_PREFIX = "group-composite:"


def composite_analysis_source_id(group, node) -> str:
    """Return the stable analysis-input reference for a live group composite."""

    suffix = "root" if node is group else node.id
    return f"{COMPOSITE_ANALYSIS_SOURCE_PREFIX}{suffix}"


def analysis_source_choices(group) -> list[tuple[str, str]]:
    """Return dataset and enabled-composite choices for the Analysis Window."""

    from ..project_data import _composite_scope, data_group_composite_enabled

    choices = [(dataset.name, dataset.id) for dataset in group.iter_datasets()]
    for node in (group, *group.iter_subgroups()):
        scope = _composite_scope(group, node)
        if data_group_composite_enabled(scope):
            choices.append(
                (f"{node.name} [live composite]", composite_analysis_source_id(group, node))
            )
    return choices


def prepare_analysis_source(group, source_id: str, *, progress_callback=None) -> AnalysisInput:
    """Prepare either an ordinary dataset or a live group-composite input."""

    if not str(source_id).startswith(COMPOSITE_ANALYSIS_SOURCE_PREFIX):
        dataset = next(
            (candidate for candidate in group.iter_datasets() if candidate.id == source_id),
            None,
        )
        if dataset is None:
            raise KeyError(f"analysis refers to missing dataset ID {source_id}")
        return prepare_analysis_input(
            group,
            dataset,
            progress_callback=progress_callback,
        )

    from ..project_data import (
        _cached_composite_dataset_data,
        _composite_cache_signature,
        _composite_scope,
        data_group_composite_enabled,
    )

    suffix = str(source_id)[len(COMPOSITE_ANALYSIS_SOURCE_PREFIX) :]
    node = group if suffix == "root" else next(
        (candidate for candidate in group.iter_subgroups() if candidate.id == suffix),
        None,
    )
    if node is None:
        raise KeyError(f"analysis refers to missing group composite {source_id}")
    scope = _composite_scope(group, node)
    if not data_group_composite_enabled(scope):
        raise ValueError(f"group composite {node.name!r} is disabled")
    data = _cached_composite_dataset_data(
        scope,
        force_rebin=True,
        progress_callback=progress_callback,
    )
    if data is None:
        raise ValueError(f"could not prepare group composite {node.name!r}")
    signature = _composite_cache_signature(scope)
    context = AnalysisContext(
        group.name,
        group.lattice_parameters,
        group.spacegroup,
        group.metadata.get("crystal"),
        None,
        {"source_group": node.name, "source_group_id": suffix},
    )
    return AnalysisInput(
        str(source_id),
        f"{node.name} composite",
        data,
        context,
        hashlib.sha256(signature.encode("utf-8")).hexdigest(),
    )


def prepare_analysis_input(group, dataset, *, progress_callback=None) -> AnalysisInput:
    """Build one analysis input from the canonical prepared dataset state.

    Analyses consume the same masks, rebinning, backgrounds, scale, and
    representation used by the data viewer.  Imports are local to keep this
    runner independent of Qt initialization.
    """

    from ..project_data import dataset_for_slice_viewer, effective_dataset_masks

    data = dataset_for_slice_viewer(
        dataset,
        extra_masks=effective_dataset_masks(group, dataset),
        force_rebin=True,
        force_masks=True,
        progress_callback=progress_callback,
    )
    if data is None:
        raise ValueError(f"could not prepare dataset {dataset.name!r}")
    context = AnalysisContext(
        group.name,
        group.lattice_parameters,
        group.spacegroup,
        group.metadata.get("crystal"),
        dataset.parameters.get("temperature"),
        {"dataset_metadata": dataset.metadata},
    )
    return AnalysisInput(
        dataset.id,
        dataset.name,
        data,
        context,
        dataset_entry_fingerprint(dataset, group),
    )


def prepare_analysis_inputs(
    group,
    analysis: AnalysisEntry,
    *,
    progress_callback=None,
) -> list[AnalysisInput]:
    """Prepare the ordered project datasets selected by an analysis recipe."""

    return [
        prepare_analysis_source(
            group,
            dataset_id,
            progress_callback=progress_callback,
        )
        for dataset_id in analysis.input_dataset_ids
    ]


def run_project_analysis(
    group,
    analysis: AnalysisEntry,
    *,
    progress_callback=None,
    cancel_callback=None,
):
    """Run an analysis recipe against its canonical prepared project inputs."""

    inputs = prepare_analysis_inputs(
        group,
        analysis,
        progress_callback=progress_callback,
    )
    parameters = {
        **default_analysis_parameters(analysis.type),
        **analysis.parameters,
    }
    return run_analysis_operation(
        analysis.type,
        inputs,
        parameters,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def execute_to_artifacts(
    analysis: AnalysisEntry,
    inputs: list[AnalysisInput],
    project_path: str | Path,
    *,
    progress_callback=None,
    cancel_callback=None,
) -> AnalysisResultRecord:
    """Run an analysis and atomically publish all of its artifact outputs."""

    definition = analysis_definition(analysis.type)
    analysis.operation_version = definition.version
    parameters = {**default_analysis_parameters(analysis.type), **analysis.parameters}
    current_recipe = recipe_hash(analysis.type, definition.version, parameters, analysis.input_dataset_ids)
    started = time.monotonic()
    execution = run_analysis_operation(
        analysis.type, inputs, parameters,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    refs: list[AnalysisOutputRef] = []
    existing_dataset_ids = {
        output.key: output.dataset_id
        for output in (analysis.result.outputs if analysis.result is not None else [])
        if output.dataset_id
    }
    with tempfile.TemporaryDirectory(prefix="nfit-analysis-") as temporary:
        staging = Path(temporary)
        artifact_payloads: dict[str, Path] = {}
        for key, output in execution.outputs.items():
            if isinstance(output, ScalarOutput):
                refs.append(AnalysisOutputRef(key, output.label, "scalar", scalar_value=output.value, scalar_uncertainty=output.uncertainty, unit=output.unit, metadata=output.metadata))
                continue
            filename = f"{key}.npz"
            artifact_payloads[filename] = staging / filename
            write_dataset_artifact(output_data(output), artifact_payloads[filename])
            metadata = dict(output.metadata)
            data_type = getattr(output, "data_type", "")
            if data_type:
                metadata.setdefault("data_type", data_type)
            refs.append(AnalysisOutputRef(key, output.label, "table" if output.__class__.__name__ == "TableOutput" else "dataset", artifact_path=filename, dataset_id=existing_dataset_ids.get(key, uuid4().hex), metadata=metadata))
        artifact_paths = replace_analysis_artifacts(
            project_path,
            analysis.id,
            artifact_payloads,
        )
    for ref in refs:
        if ref.artifact_path:
            ref.artifact_path = artifact_paths[ref.artifact_path]
    return AnalysisResultRecord(
        recipe_hash=current_recipe,
        input_fingerprints={item.dataset_id: item.fingerprint for item in inputs},
        outputs=refs, status="success", created_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=time.monotonic() - started,
        warnings=execution.warnings, diagnostics=execution.diagnostics,
    )


def analysis_is_fresh(analysis: AnalysisEntry, inputs: list[AnalysisInput]) -> bool:
    if analysis.result is None or analysis.result.status != "success":
        return False
    definition = analysis_definition(analysis.type)
    parameters = {**default_analysis_parameters(analysis.type), **analysis.parameters}
    expected = recipe_hash(analysis.type, definition.version, parameters, analysis.input_dataset_ids)
    return analysis.result.recipe_hash == expected and analysis.result.input_fingerprints == {item.dataset_id: item.fingerprint for item in inputs}


def upsert_analysis_output_dataset(
    group,
    analysis: AnalysisEntry,
    output: AnalysisOutputRef,
    data,
    *,
    project_path: str | Path | None,
):
    """Add or refresh a linked derived dataset without resetting user settings.

    A rerun replaces only the immutable numerical payload and provenance owned
    by the analysis. Dataset-local masks, rebinning, symmetry, enabled state,
    fit weight, and scale settings remain independently editable.
    """

    from ..pipeline import DatasetEntry, DatasetGroup

    if not output.dataset_id or not output.artifact_path:
        raise ValueError("analysis output does not contain a dataset artifact")
    existing = next(
        (dataset for dataset in group.iter_datasets() if dataset.id == output.dataset_id),
        None,
    )
    provenance = {
        "analysis_id": analysis.id,
        "output_key": output.key,
        "recipe_hash": analysis.result.recipe_hash if analysis.result is not None else "",
    }
    source_metadata = {
        "source_file": output.artifact_path,
        "analysis_artifact_path": output.artifact_path,
        "analysis_output_metadata": dict(output.metadata),
        "derived_from_analysis": provenance,
    }
    if project_path is not None:
        source_metadata["_project_path"] = str(project_path)
    data_type = str(output.metadata.get("data_type") or "derived_analysis")
    if existing is not None:
        existing.replace_data(data, source_backed=True)
        existing.kind = "analysis"
        existing.data_type = data_type
        existing.metadata.update(source_metadata)
        return existing

    derived = next(
        (node for node in group.subgroups if node.name == "Derived data"),
        None,
    )
    if derived is None:
        derived = DatasetGroup("Derived data")
        group.subgroups.append(derived)
    base = str(output.label).strip() or "Derived dataset"
    name = base
    index = 2
    existing_names = set(group.dataset_names)
    while name in existing_names:
        name = f"{base} {index}"
        index += 1
    entry = DatasetEntry(
        name,
        data,
        kind="analysis",
        data_type=data_type,
        enabled=bool(output.metadata.get("fit_enabled", False)),
        fit_weight=float(output.metadata.get("fit_weight", 0.0)),
        metadata=source_metadata,
        id=output.dataset_id,
    )
    derived.datasets.append(entry)
    return entry
