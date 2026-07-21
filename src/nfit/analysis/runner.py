from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import analysis_asset_root, output_data, write_dataset_artifact
from .core import (
    AnalysisEntry,
    AnalysisInput,
    AnalysisOutputRef,
    AnalysisResultRecord,
    ScalarOutput,
)
from .fingerprint import recipe_hash
from .registry import analysis_definition, default_analysis_parameters, run_analysis_operation


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
    parameters = {**default_analysis_parameters(analysis.type), **analysis.parameters}
    current_recipe = recipe_hash(analysis.type, definition.version, parameters, analysis.input_dataset_ids)
    started = time.monotonic()
    execution = run_analysis_operation(
        analysis.type, inputs, parameters,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    root = analysis_asset_root(project_path) / analysis.id
    staging = root.parent / f".{analysis.id}-{uuid4().hex}.staging"
    backup = root.parent / f".{analysis.id}-{uuid4().hex}.backup"
    staging.mkdir(parents=True, exist_ok=False)
    refs: list[AnalysisOutputRef] = []
    try:
        for key, output in execution.outputs.items():
            if isinstance(output, ScalarOutput):
                refs.append(AnalysisOutputRef(key, output.label, "scalar", scalar_value=output.value, scalar_uncertainty=output.uncertainty, unit=output.unit, metadata=output.metadata))
                continue
            artifact = staging / f"{key}.npz"
            write_dataset_artifact(output_data(output), artifact)
            metadata = dict(output.metadata)
            data_type = getattr(output, "data_type", "")
            if data_type:
                metadata.setdefault("data_type", data_type)
            refs.append(AnalysisOutputRef(key, output.label, "table" if output.__class__.__name__ == "TableOutput" else "dataset", artifact_path=artifact.name, dataset_id=uuid4().hex, metadata=metadata))
        if root.exists():
            root.replace(backup)
        try:
            staging.replace(root)
        except Exception:
            if backup.exists():
                backup.replace(root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        for ref in refs:
            if ref.artifact_path:
                ref.artifact_path = str((Path(project_path).name + "-assets") / Path("analyses") / analysis.id / Path(ref.artifact_path).name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
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
