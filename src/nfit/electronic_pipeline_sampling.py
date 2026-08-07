"""Dataset-domain convergence for complete electronic-response pipelines.

The low-level Lindhard certificate in :mod:`nfit.electronic_sampling` is useful
for an explicitly declared ``(Q, E, T)`` inspection domain.  This module adds
the complementary project-facing contract: evaluate the complete compiled
tight-binding -> Lindhard -> RPA observable on deterministic representatives of
the datasets that will actually be fitted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PointData4D
from .electronic_sampling import (
    SamplingCertificate,
    SamplingComparison,
    SamplingPolicy,
    SamplingProgress,
    automatic_mesh_ladder,
)
from .electronic_structure import electronic_model_from_component
from .fit_config import (
    FitDatasetInput,
    compile_fit_problem,
    model_supports_data_type,
)
from .fitting import evaluate_problem_model
from .model_registry import model_definition, serialize_model_component
from .pipeline import ModelComponentSpec


@dataclass(frozen=True)
class PipelineDatasetDomain:
    """Recorded deterministic representative domain for one fitted dataset."""

    name: str
    data_type: str
    total_points: int
    valid_points: int
    sampled_indices: tuple[int, ...]
    coordinate_ranges: Mapping[str, tuple[float, float]]
    domain_digest: str

    @property
    def sampled_points(self) -> int:
        return len(self.sampled_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "total_points": self.total_points,
            "valid_points": self.valid_points,
            "sampled_points": self.sampled_points,
            "sampled_indices": list(self.sampled_indices),
            "coordinate_ranges": {
                key: list(value) for key, value in self.coordinate_ranges.items()
            },
            "domain_digest": self.domain_digest,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_array_digest(digest: Any, label: str, values: Any) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _dataset_domain_digest(
    dataset: FitDatasetInput,
    valid_indices: np.ndarray,
) -> str:
    data = dataset.data
    digest = hashlib.sha256()
    digest.update(dataset.name.encode("utf-8"))
    digest.update(dataset.data_type.encode("utf-8"))
    _update_array_digest(digest, "valid_indices", valid_indices)
    for name in ("H", "K", "L", "E"):
        _update_array_digest(digest, name, getattr(data, name)[valid_indices])
    if isinstance(data.temperature, np.ndarray):
        _update_array_digest(
            digest,
            "temperature",
            data.temperature[valid_indices],
        )
    else:
        scalar_temperature = (
            None if data.temperature is None else float(data.temperature)
        )
        digest.update(
            json.dumps(
                {"temperature": scalar_temperature},
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        )
    if isinstance(data.magnetic_field, np.ndarray) and data.magnetic_field.ndim == 2:
        _update_array_digest(
            digest,
            "magnetic_field",
            data.magnetic_field[valid_indices],
        )
    elif data.magnetic_field is not None:
        _update_array_digest(digest, "magnetic_field", data.magnetic_field)
    digest.update(
        json.dumps(
            _json_safe(data.metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _feature_columns(data: PointData4D, valid_indices: np.ndarray) -> dict[str, np.ndarray]:
    columns = {
        name: np.asarray(getattr(data, name)[valid_indices], dtype=float)
        for name in ("H", "K", "L", "E")
    }
    if isinstance(data.temperature, np.ndarray):
        columns["temperature_K"] = np.asarray(
            data.temperature[valid_indices], dtype=float
        )
    if isinstance(data.magnetic_field, np.ndarray) and data.magnetic_field.ndim == 2:
        selected = np.asarray(data.magnetic_field[valid_indices], dtype=float)
        for axis, label in enumerate(("field_x_T", "field_y_T", "field_z_T")):
            columns[label] = selected[:, axis]
    return columns


def _coordinate_ranges(
    data: PointData4D,
    valid_indices: np.ndarray,
) -> dict[str, tuple[float, float]]:
    columns = _feature_columns(data, valid_indices)
    if data.temperature is not None and not isinstance(data.temperature, np.ndarray):
        value = float(data.temperature)
        columns["temperature_K"] = np.full(1, value, dtype=float)
    if (
        isinstance(data.magnetic_field, np.ndarray)
        and data.magnetic_field.shape == (3,)
    ):
        for value, label in zip(
            data.magnetic_field,
            ("field_x_T", "field_y_T", "field_z_T"),
            strict=True,
        ):
            columns[label] = np.full(1, float(value), dtype=float)
    return {
        name: (float(np.min(values)), float(np.max(values)))
        for name, values in columns.items()
    }


def _representative_indices(
    data: PointData4D,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    limit = int(max_points)
    if limit < 1:
        raise ValueError("max_points_per_dataset must be positive")
    valid_indices = np.flatnonzero(data.valid_mask())
    if valid_indices.size == 0:
        raise ValueError("dataset has no finite unmasked positive-uncertainty points")
    if valid_indices.size <= limit:
        return valid_indices, valid_indices

    columns = _feature_columns(data, valid_indices)
    candidate_count = min(valid_indices.size, max(4096, 64 * limit))
    candidate_positions = np.linspace(
        0,
        valid_indices.size - 1,
        candidate_count,
        dtype=np.int64,
    )
    # Explicitly add every coordinate extremum.  Evenly spaced flattened-grid
    # candidates alone can miss a thin temperature, field, or reciprocal-space
    # boundary in sparse point lists.
    extrema = []
    for values in columns.values():
        extrema.extend((int(np.argmin(values)), int(np.argmax(values))))
    candidate_positions = np.unique(
        np.concatenate((candidate_positions, np.asarray(extrema, dtype=np.int64)))
    )

    feature_values = []
    for values in columns.values():
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper > lower:
            feature_values.append(
                (values[candidate_positions] - lower) / (upper - lower)
            )
    if not feature_values:
        chosen_positions = np.linspace(
            0,
            valid_indices.size - 1,
            limit,
            dtype=np.int64,
        )
        return valid_indices[np.unique(chosen_positions)], valid_indices

    features = np.column_stack(feature_values)
    extrema_indices = np.searchsorted(
        candidate_positions,
        np.unique(np.asarray(extrema, dtype=np.int64)),
    )
    center_distance = np.sum((features - 0.5) ** 2, axis=1)
    if extrema_indices.size <= limit:
        # Preserve all independent coordinate boundaries whenever the point
        # budget permits. This keeps a thin temperature, field, energy, or Q
        # boundary from disappearing merely because it is sparse in storage
        # order.
        selected = [int(value) for value in extrema_indices]
    else:
        # An unusually small user budget cannot hold every boundary. Choose a
        # deterministic space-filling subset of extrema rather than favoring
        # the coordinate-column order.
        extrema_features = features[extrema_indices]
        selected_within_extrema = [
            int(np.argmax(center_distance[extrema_indices]))
        ]
        minimum_extrema_distance = np.sum(
            (
                extrema_features
                - extrema_features[selected_within_extrema[0]]
            )
            ** 2,
            axis=1,
        )
        while len(selected_within_extrema) < limit:
            minimum_extrema_distance[
                np.asarray(selected_within_extrema, dtype=np.intp)
            ] = -1.0
            next_index = int(np.argmax(minimum_extrema_distance))
            selected_within_extrema.append(next_index)
            distance = np.sum(
                (extrema_features - extrema_features[next_index]) ** 2,
                axis=1,
            )
            minimum_extrema_distance = np.minimum(
                minimum_extrema_distance,
                distance,
            )
        selected = [
            int(extrema_indices[index]) for index in selected_within_extrema
        ]
    minimum_distance = np.min(
        np.sum(
            (features[:, np.newaxis, :] - features[selected][np.newaxis, :, :])
            ** 2,
            axis=2,
        ),
        axis=1,
    )
    while len(selected) < min(limit, candidate_positions.size):
        minimum_distance[np.asarray(selected, dtype=np.intp)] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum((features - features[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    chosen = np.sort(valid_indices[candidate_positions[np.asarray(selected)]])
    return chosen, valid_indices


def _subset_point_data(data: PointData4D, indices: np.ndarray) -> PointData4D:
    temperature: float | np.ndarray | None
    if isinstance(data.temperature, np.ndarray):
        temperature = data.temperature[indices]
    else:
        temperature = data.temperature
    magnetic_field: np.ndarray | None
    if isinstance(data.magnetic_field, np.ndarray) and data.magnetic_field.ndim == 2:
        magnetic_field = data.magnetic_field[indices]
    elif data.magnetic_field is None:
        magnetic_field = None
    else:
        magnetic_field = np.asarray(data.magnetic_field, dtype=float)
    metadata = copy.deepcopy(data.metadata)
    metadata["electronic_sampling_representative_indices"] = indices.tolist()
    return PointData4D(
        H=data.H[indices],
        K=data.K[indices],
        L=data.L[indices],
        E=data.E[indices],
        intensity=data.intensity[indices],
        sigma=data.sigma[indices],
        mask=np.ones(indices.size, dtype=bool),
        temperature=temperature,
        magnetic_field=magnetic_field,
        metadata=metadata,
    )


def sample_pipeline_datasets(
    datasets: Sequence[FitDatasetInput],
    *,
    max_points_per_dataset: int = 32,
) -> tuple[tuple[FitDatasetInput, ...], tuple[PipelineDatasetDomain, ...]]:
    """Return deterministic representative points from arbitrary fit datasets."""

    sampled: list[FitDatasetInput] = []
    domains: list[PipelineDatasetDomain] = []
    names: set[str] = set()
    for dataset in datasets:
        if dataset.name in names:
            raise ValueError(f"duplicate dataset name {dataset.name!r}")
        names.add(dataset.name)
        selected, valid = _representative_indices(
            dataset.data,
            max_points=max_points_per_dataset,
        )
        sampled.append(
            FitDatasetInput(
                name=dataset.name,
                data=_subset_point_data(dataset.data, selected),
                weight=dataset.weight,
                data_type=dataset.data_type,
                metadata=copy.deepcopy(dataset.metadata),
                scale_value=dataset.scale_value,
                scale_vary=dataset.scale_vary,
                scale_group=dataset.scale_group,
            )
        )
        domains.append(
            PipelineDatasetDomain(
                name=dataset.name,
                data_type=dataset.data_type,
                total_points=dataset.data.size,
                valid_points=int(valid.size),
                sampled_indices=tuple(int(value) for value in selected),
                coordinate_ranges=_coordinate_ranges(dataset.data, valid),
                domain_digest=_dataset_domain_digest(dataset, valid),
            )
        )
    return tuple(sampled), tuple(domains)


def _component_mapping(components: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    values = list(
        components.values() if isinstance(components, Mapping) else components
    )
    result = {str(component.name): component for component in values}
    if len(result) != len(values):
        raise ValueError("electronic pipeline components must have unique names")
    return result


def _reference_names(component: Any) -> tuple[str, ...]:
    definition = model_definition(str(component.type))
    return tuple(
        str(component.config.get(field, "")).strip()
        for field in definition.component_reference_fields
        if str(component.config.get(field, "")).strip()
    )


def electronic_pipeline_components(
    lindhard_component: Any,
    components: Mapping[str, Any] | Sequence[Any],
) -> tuple[Any, ...]:
    """Return the selected Lindhard component's upstream and dressing closure."""

    if getattr(lindhard_component, "type", None) != "lindhard":
        raise TypeError("pipeline sampling requires a Lindhard component")
    by_name = _component_mapping(components)
    selected_name = str(lindhard_component.name)
    if by_name.get(selected_name) is not lindhard_component:
        raise ValueError("the selected Lindhard component is not in components")
    if not bool(getattr(lindhard_component, "enabled", True)):
        raise ValueError("the selected Lindhard component is disabled")

    included: set[str] = set()

    def include_upstream(name: str) -> None:
        if name in included:
            return
        try:
            component = by_name[name]
        except KeyError as exc:
            raise ValueError(
                f"electronic pipeline references missing component {name!r}"
            ) from exc
        if not bool(getattr(component, "enabled", True)):
            raise ValueError(
                f"electronic pipeline references disabled component {name!r}"
            )
        included.add(name)
        references = list(_reference_names(component))
        if component is lindhard_component and not references:
            candidates = [
                candidate.name
                for candidate in by_name.values()
                if candidate.type == "tight_binding"
                and bool(getattr(candidate, "enabled", True))
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "an automatic Lindhard source requires exactly one enabled "
                    "tight-binding component"
                )
            references = candidates
        for reference in references:
            include_upstream(reference)

    include_upstream(selected_name)
    changed = True
    while changed:
        changed = False
        for name, component in by_name.items():
            if name in included or not bool(getattr(component, "enabled", True)):
                continue
            definition = model_definition(str(component.type))
            if not definition.consumes_referenced_observables:
                continue
            if any(reference in included for reference in _reference_names(component)):
                include_upstream(name)
                changed = True
    return tuple(component for name, component in by_name.items() if name in included)


def _component_state_payload(component: Any, *, selected_lindhard: str) -> dict[str, Any]:
    payload = copy.deepcopy(serialize_model_component(component, purpose="workflow"))
    config = payload.get("config", {})
    if isinstance(config, dict):
        config.pop("response_sampling_certificate", None)
        config.pop("response_q_interpolation_certificates", None)
        if str(payload.get("name")) == selected_lindhard:
            config.pop("response_mesh", None)
    return payload


def electronic_pipeline_state_digest(
    lindhard_component: Any,
    components: Mapping[str, Any] | Sequence[Any],
) -> str:
    """Digest all physical state relevant to one compiled response pipeline."""

    closure = electronic_pipeline_components(lindhard_component, components)
    return _stable_digest(
        [
            _component_state_payload(
                component,
                selected_lindhard=str(lindhard_component.name),
            )
            for component in closure
        ]
    )


def _configured_components(
    closure: Sequence[Any],
    *,
    lindhard_name: str,
    mesh: tuple[int, ...],
) -> tuple[Any, ...]:
    configured = tuple(
        ModelComponentSpec(
            **serialize_model_component(component, purpose="workflow")
        )
        for component in closure
    )
    selected = next(
        component for component in configured if component.name == lindhard_name
    )
    selected.config["response_mesh"] = list(mesh)
    if not str(selected.config.get("electronic_component", "")).strip():
        sources = [
            component.name
            for component in configured
            if component.type == "tight_binding" and component.enabled
        ]
        if len(sources) != 1:
            raise ValueError(
                "an automatic Lindhard source requires exactly one enabled "
                "tight-binding component"
            )
        selected.config["electronic_component"] = sources[0]
    selected.config["response_sampling_mode"] = "manual"
    selected.config["response_sampling_certificate"] = {}
    selected.config["response_q_evaluation"] = "direct"
    selected.config["response_q_interpolation_rtol"] = 0.0
    selected.config["response_q_interpolation_atol"] = 0.0
    selected.config["response_q_interpolation_mesh"] = []
    selected.config["response_q_interpolation_certificates"] = {}
    return configured


def _compiled_predictions(
    closure: Sequence[Any],
    datasets: Sequence[FitDatasetInput],
    *,
    lindhard_name: str,
    mesh: tuple[int, ...],
) -> dict[str, np.ndarray]:
    configured = _configured_components(
        closure,
        lindhard_name=lindhard_name,
        mesh=mesh,
    )
    compiled = compile_fit_problem(
        configured,
        datasets,
        description=f"electronic pipeline convergence at {mesh}",
    )
    params = {
        spec.name: float(spec.value) for spec in compiled.problem.parameter_specs
    }
    predictions = {
        dataset.name: np.asarray(
            evaluate_problem_model(compiled.problem, dataset.name, params),
            dtype=float,
        )
        for dataset in compiled.problem.datasets
    }
    for name, values in predictions.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"electronic pipeline produced nonfinite values for dataset {name!r} "
                f"on mesh {mesh}"
            )
    return predictions


def _potentially_applicable_datasets(
    closure: Sequence[Any],
    datasets: Sequence[FitDatasetInput],
) -> tuple[FitDatasetInput, ...]:
    observable_components = [
        component
        for component in closure
        if model_definition(str(component.type)).data_types
    ]
    return tuple(
        dataset
        for dataset in datasets
        if any(
            model_supports_data_type(component.type, dataset.data_type)
            and (
                component.applies_to is None
                or dataset.name in component.applies_to
            )
            for component in observable_components
        )
    )


def _prediction_comparison(
    coarse: Mapping[str, np.ndarray],
    fine: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> tuple[SamplingComparison, dict[str, Any]]:
    if set(coarse) != set(fine):
        raise ValueError("pipeline convergence datasets changed between meshes")
    details: dict[str, Any] = {}
    maxima = []
    rms_values = []
    for name in sorted(fine):
        coarse_values = np.asarray(coarse[name], dtype=float)
        fine_values = np.asarray(fine[name], dtype=float)
        if coarse_values.shape != fine_values.shape:
            raise ValueError(
                f"pipeline convergence shape changed for dataset {name!r}"
            )
        delta = np.abs(fine_values - coarse_values)
        maximum_absolute = float(np.max(delta)) if delta.size else 0.0
        scale = max(float(np.max(np.abs(fine_values))), 1.0e-12)
        fine_norm = float(np.linalg.norm(fine_values.ravel()))
        maximum_relative = maximum_absolute / scale
        rms_relative = float(np.linalg.norm(delta.ravel())) / max(
            fine_norm,
            1.0e-12,
        )
        passed = max(maximum_relative, rms_relative) <= float(tolerance)
        maxima.append(maximum_relative)
        rms_values.append(rms_relative)
        details[name] = {
            "points": int(fine_values.size),
            "maximum_absolute_change": maximum_absolute,
            "normalization_scale": scale,
            "maximum_relative_error": maximum_relative,
            "rms_relative_error": rms_relative,
            "passed": passed,
        }
    maximum = max(maxima, default=0.0)
    rms = max(rms_values, default=0.0)
    return (
        SamplingComparison(
            coarse_mesh=(),
            fine_mesh=(),
            maximum_relative_error=maximum,
            rms_relative_error=rms,
            integrated_relative_error=0.0,
            passed=max(maximum, rms) <= float(tolerance),
        ),
        details,
    )


def certify_electronic_pipeline_sampling(
    lindhard_component: Any,
    components: Mapping[str, Any] | Sequence[Any],
    datasets: Sequence[FitDatasetInput],
    *,
    seed_mesh: Sequence[int] | None = None,
    policy: SamplingPolicy | None = None,
    max_refinements: int = 7,
    max_mesh_points: int = 500_000,
    max_points_per_dataset: int = 32,
    progress_callback: Any | None = None,
) -> SamplingCertificate:
    """Certify a fixed k mesh against complete dataset-facing observables.

    Arbitrary experimental-Q interpolation is disabled during this calculation,
    so the certificate isolates integration-mesh error.  Broadening, powder
    orientation count, backend, and all physical component parameters remain
    fixed at their declared project values.
    """

    from .electronic_sampling import sampling_policy

    selected_policy = sampling_policy() if policy is None else policy
    closure = electronic_pipeline_components(lindhard_component, components)
    source_name = str(
        lindhard_component.config.get("electronic_component", "")
    ).strip()
    sources = [
        component
        for component in closure
        if component.type == "tight_binding"
        and (not source_name or component.name == source_name)
    ]
    source = sources[0] if len(sources) == 1 else None
    if source is None:
        raise ValueError(
            "pipeline sampling requires an explicit enabled tight-binding source"
        )
    model = electronic_model_from_component(source)
    base_mesh = (
        lindhard_component.config.get("response_mesh", [16] * model.dimension)
        if seed_mesh is None
        else seed_mesh
    )
    meshes = automatic_mesh_ladder(
        model,
        base_mesh,
        policy=selected_policy,
        max_refinements=max_refinements,
        max_mesh_points=max_mesh_points,
    )
    candidate_datasets = _potentially_applicable_datasets(closure, datasets)
    if not candidate_datasets:
        raise ValueError(
            "no enabled prepared dataset is modeled by the selected electronic pipeline"
        )
    sampled, domains = sample_pipeline_datasets(
        candidate_datasets,
        max_points_per_dataset=max_points_per_dataset,
    )
    # Compile once to remove datasets that this electronic pipeline does not
    # model.  This lets mixed projects contain arbitrary unrelated data types
    # without weakening or preventing the electronic certificate.
    probe_mesh = meshes[0] if meshes else tuple(int(value) for value in base_mesh)
    probe = compile_fit_problem(
        _configured_components(
            closure,
            lindhard_name=str(lindhard_component.name),
            mesh=probe_mesh,
        ),
        sampled,
        description="electronic pipeline convergence domain",
    )
    applicable = {dataset.name for dataset in probe.problem.datasets}
    sampled = tuple(dataset for dataset in sampled if dataset.name in applicable)
    domains = tuple(domain for domain in domains if domain.name in applicable)
    if not sampled:
        raise ValueError(
            "no enabled prepared dataset is modeled by the selected electronic pipeline"
        )

    attempted: list[tuple[int, ...]] = []
    comparisons: list[SamplingComparison] = []
    comparison_details: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    previous: dict[str, np.ndarray] | None = None
    previous_mesh: tuple[int, ...] | None = None
    passing_run = 0
    chosen: tuple[int, ...] | None = None
    stopping_reason = (
        "mesh_budget_excluded_all_candidates"
        if not meshes
        else "mesh_ladder_exhausted"
    )
    for iteration, mesh in enumerate(meshes, start=1):
        if progress_callback is not None:
            progress_callback(
                SamplingProgress(
                    observable="electronic_pipeline",
                    iteration=iteration,
                    candidate_count=len(meshes),
                    mesh=mesh,
                    phase="started",
                    consecutive_passes=passing_run,
                    required_passes=selected_policy.consecutive_passes,
                )
            )
        started = time.perf_counter()
        current = _compiled_predictions(
            closure,
            sampled,
            lindhard_name=str(lindhard_component.name),
            mesh=mesh,
        )
        elapsed = time.perf_counter() - started
        attempted.append(mesh)
        timings.append({"mesh": list(mesh), "elapsed_seconds": elapsed})
        comparison: SamplingComparison | None = None
        if previous is not None and previous_mesh is not None:
            raw, details = _prediction_comparison(
                previous,
                current,
                tolerance=selected_policy.relative_tolerance,
            )
            comparison = SamplingComparison(
                coarse_mesh=previous_mesh,
                fine_mesh=mesh,
                maximum_relative_error=raw.maximum_relative_error,
                rms_relative_error=raw.rms_relative_error,
                integrated_relative_error=0.0,
                passed=raw.passed,
            )
            comparisons.append(comparison)
            comparison_details.append(
                {
                    "coarse_mesh": list(previous_mesh),
                    "fine_mesh": list(mesh),
                    "datasets": details,
                }
            )
            passing_run = passing_run + 1 if comparison.passed else 0
        if progress_callback is not None:
            progress_callback(
                SamplingProgress(
                    observable="electronic_pipeline",
                    iteration=iteration,
                    candidate_count=len(meshes),
                    mesh=mesh,
                    phase="completed",
                    elapsed_seconds=elapsed,
                    comparison=comparison,
                    consecutive_passes=passing_run,
                    required_passes=selected_policy.consecutive_passes,
                )
            )
        if comparison is not None and passing_run >= selected_policy.consecutive_passes:
            chosen = previous_mesh
            stopping_reason = "required_consecutive_passes_reached"
            break
        remaining = len(meshes) - iteration
        passes_needed = selected_policy.consecutive_passes - passing_run
        if comparison is not None and remaining < passes_needed:
            stopping_reason = "insufficient_remaining_refinements"
            break
        previous = current
        previous_mesh = mesh

    state_digest = electronic_pipeline_state_digest(
        lindhard_component,
        components,
    )
    domain_payload = {
        "kind": "fit_datasets",
        "dataset_count": len(domains),
        "sampled_points": sum(domain.sampled_points for domain in domains),
        "datasets": [domain.to_dict() for domain in domains],
    }
    input_digest = _stable_digest(
        {
            "observable": "electronic_pipeline",
            "pipeline_state_digest": state_digest,
            "domains": [domain.to_dict() for domain in domains],
            "mesh_shift": lindhard_component.config.get(
                "response_mesh_shift", [0.5] * model.dimension
            ),
            "symmetry": lindhard_component.config.get("response_symmetry", "auto"),
            "policy": selected_policy.to_dict(),
        }
    )
    return SamplingCertificate(
        observable="electronic_pipeline",
        status="certified" if chosen is not None else "budget_exhausted",
        policy=selected_policy,
        chosen_mesh=chosen,
        attempted_meshes=tuple(attempted),
        comparisons=tuple(comparisons),
        input_digest=input_digest,
        domain=domain_payload,
        provenance={
            "comparison": (
                "successive complete compiled TB-Lindhard-RPA observables; "
                "every applicable dataset must pass"
            ),
            "pipeline_components": [
                {"name": component.name, "type": component.type}
                for component in closure
            ],
            "pipeline_state_digest": state_digest,
            "model_digest": model.content_digest,
            "source_component": source.name,
            "lindhard_component": lindhard_component.name,
            "broadening_meV": float(
                lindhard_component.parameters.get("broadening", 5.0)
            ),
            "chemical_potential_mode": str(
                lindhard_component.config.get(
                    "chemical_potential_mode", "source"
                )
            ),
            "chemical_potential_meV": float(
                source.config.get("chemical_potential_meV", 0.0)
            ),
            "filling_per_cell": (
                float(lindhard_component.config.get("filling_per_cell", 1.0))
                if str(
                    lindhard_component.config.get(
                        "chemical_potential_mode", "source"
                    )
                )
                == "filling"
                else None
            ),
            "mesh_shift": list(
                lindhard_component.config.get(
                    "response_mesh_shift", [0.5] * model.dimension
                )
            ),
            "symmetry": str(
                lindhard_component.config.get("response_symmetry", "auto")
            ),
            "q_evaluation_during_certificate": "direct",
            "representative_selection": (
                "deterministic farthest-point coverage of H, K, L, E, "
                "temperature, and magnetic field"
            ),
            "max_points_per_dataset": int(max_points_per_dataset),
            "dataset_comparisons": comparison_details,
            "mesh_timings": timings,
            "stopping_reason": stopping_reason,
        },
    )
