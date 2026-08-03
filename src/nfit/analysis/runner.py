from __future__ import annotations

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


def prepare_analysis_input(group, dataset, *, progress_callback=None) -> AnalysisInput:
    """Build one analysis input from the canonical prepared dataset state.

    Analyses consume the same masks, rebinning, backgrounds, scale, and
    representation used by the data viewer.  Imports are local to keep this
    runner independent of Qt initialization.
    """

    from ..project_gui import dataset_for_slice_viewer, effective_dataset_masks

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

    datasets = {dataset.id: dataset for dataset in group.iter_datasets()}
    missing = [
        dataset_id
        for dataset_id in analysis.input_dataset_ids
        if dataset_id not in datasets
    ]
    if missing:
        raise KeyError(
            "analysis refers to missing dataset ID(s): " + ", ".join(missing)
        )
    return [
        prepare_analysis_input(
            group,
            datasets[dataset_id],
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
