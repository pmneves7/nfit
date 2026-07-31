"""Observable-specific automatic sampling for electronic calculations.

Sampling is certified before a fit and resolves to an ordinary concrete mesh.
The optimizer never changes the mesh, so automatic selection cannot make the
objective discontinuous.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .electronic_structure import (
    DensityOfStatesResult,
    ElectronicModel,
    density_of_states,
    k_mesh,
)

SamplingAccuracy = Literal["preview", "standard", "high", "custom"]

_ACCURACY_TOLERANCES = {
    "preview": 5.0e-2,
    "standard": 1.0e-2,
    "high": 2.0e-3,
}


@dataclass(frozen=True)
class SamplingPolicy:
    """Accuracy and stopping rules for an automatic mesh search."""

    accuracy: SamplingAccuracy = "standard"
    relative_tolerance: float = 1.0e-2
    consecutive_passes: int = 2
    refinement_factor: float = 1.3

    def __post_init__(self) -> None:
        accuracy = str(self.accuracy).strip().lower()
        if accuracy not in {"preview", "standard", "high", "custom"}:
            raise ValueError(
                "sampling accuracy must be preview, standard, high, or custom"
            )
        tolerance = float(self.relative_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("sampling relative tolerance must be positive")
        if int(self.consecutive_passes) < 1:
            raise ValueError("sampling consecutive_passes must be positive")
        factor = float(self.refinement_factor)
        if not np.isfinite(factor) or factor <= 1.0:
            raise ValueError("sampling refinement_factor must exceed one")
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "consecutive_passes", int(self.consecutive_passes))
        object.__setattr__(self, "refinement_factor", factor)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy."""

        return {
            "accuracy": self.accuracy,
            "relative_tolerance": self.relative_tolerance,
            "consecutive_passes": self.consecutive_passes,
            "refinement_factor": self.refinement_factor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SamplingPolicy:
        """Reconstruct a policy from serialized values."""

        return cls(
            accuracy=str(payload.get("accuracy", "custom")),
            relative_tolerance=float(
                payload.get("relative_tolerance", 1.0e-2)
            ),
            consecutive_passes=int(payload.get("consecutive_passes", 2)),
            refinement_factor=float(payload.get("refinement_factor", 1.3)),
        )


def sampling_policy(
    accuracy: SamplingAccuracy | str = "standard",
    *,
    relative_tolerance: float | None = None,
    consecutive_passes: int = 2,
    refinement_factor: float = 1.3,
) -> SamplingPolicy:
    """Return one named accuracy policy or an explicit custom policy."""

    name = str(accuracy).strip().lower()
    if name == "custom":
        if relative_tolerance is None:
            raise ValueError("custom sampling accuracy requires a tolerance")
        tolerance = float(relative_tolerance)
    else:
        try:
            tolerance = _ACCURACY_TOLERANCES[name]
        except KeyError as exc:
            raise ValueError(
                "sampling accuracy must be preview, standard, high, or custom"
            ) from exc
        if relative_tolerance is not None:
            raise ValueError(
                "relative_tolerance is accepted only for custom sampling"
            )
    return SamplingPolicy(
        accuracy=name,
        relative_tolerance=tolerance,
        consecutive_passes=consecutive_passes,
        refinement_factor=refinement_factor,
    )


@dataclass(frozen=True)
class SamplingComparison:
    """Error measured between two successive meshes."""

    coarse_mesh: tuple[int, ...]
    fine_mesh: tuple[int, ...]
    maximum_relative_error: float
    rms_relative_error: float
    integrated_relative_error: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible comparison."""

        return {
            "coarse_mesh": list(self.coarse_mesh),
            "fine_mesh": list(self.fine_mesh),
            "maximum_relative_error": self.maximum_relative_error,
            "rms_relative_error": self.rms_relative_error,
            "integrated_relative_error": self.integrated_relative_error,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SamplingComparison:
        """Reconstruct a comparison from serialized values."""

        return cls(
            coarse_mesh=tuple(int(value) for value in payload["coarse_mesh"]),
            fine_mesh=tuple(int(value) for value in payload["fine_mesh"]),
            maximum_relative_error=float(payload["maximum_relative_error"]),
            rms_relative_error=float(payload["rms_relative_error"]),
            integrated_relative_error=float(
                payload.get("integrated_relative_error", 0.0)
            ),
            passed=bool(payload["passed"]),
        )


@dataclass(frozen=True)
class SamplingProgress:
    """Progress from one candidate mesh in an automatic sampling search."""

    observable: str
    iteration: int
    candidate_count: int
    mesh: tuple[int, ...]
    phase: Literal["started", "completed"]
    elapsed_seconds: float = 0.0
    comparison: SamplingComparison | None = None
    consecutive_passes: int = 0
    required_passes: int = 2


@dataclass(frozen=True)
class SamplingCertificate:
    """Serializable record of an observable-specific mesh search."""

    observable: str
    status: Literal["certified", "budget_exhausted"]
    policy: SamplingPolicy
    chosen_mesh: tuple[int, ...] | None
    attempted_meshes: tuple[tuple[int, ...], ...]
    comparisons: tuple[SamplingComparison, ...]
    input_digest: str
    domain: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def certified(self) -> bool:
        """Whether the requested tolerance was met."""

        return self.status == "certified" and self.chosen_mesh is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible certificate."""

        return {
            "observable": self.observable,
            "status": self.status,
            "policy": self.policy.to_dict(),
            "chosen_mesh": (
                None if self.chosen_mesh is None else list(self.chosen_mesh)
            ),
            "attempted_meshes": [
                list(mesh) for mesh in self.attempted_meshes
            ],
            "comparisons": [
                comparison.to_dict() for comparison in self.comparisons
            ],
            "input_digest": self.input_digest,
            "domain": dict(self.domain),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SamplingCertificate:
        """Reconstruct a certificate from serialized values."""

        chosen = payload.get("chosen_mesh")
        return cls(
            observable=str(payload["observable"]),
            status=str(payload["status"]),
            policy=SamplingPolicy.from_dict(payload["policy"]),
            chosen_mesh=(
                None
                if chosen is None
                else tuple(int(value) for value in chosen)
            ),
            attempted_meshes=tuple(
                tuple(int(value) for value in mesh)
                for mesh in payload.get("attempted_meshes", ())
            ),
            comparisons=tuple(
                SamplingComparison.from_dict(item)
                for item in payload.get("comparisons", ())
            ),
            input_digest=str(payload["input_digest"]),
            domain=dict(payload.get("domain", {})),
            provenance=dict(payload.get("provenance", {})),
        )


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _periodic_shape(
    model: ElectronicModel,
    shape: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(int(value) for value in shape)
    if len(values) == model.dimension:
        selected = values
    elif len(values) == 3:
        selected = tuple(values[axis] for axis in model.periodic_axes)
    else:
        raise ValueError(
            "mesh shape must match the periodic dimensionality or have three axes"
        )
    if any(value < 1 for value in selected):
        raise ValueError("mesh sizes must be positive")
    return selected


def automatic_mesh_ladder(
    model: ElectronicModel,
    seed_shape: Sequence[int],
    *,
    policy: SamplingPolicy,
    max_refinements: int,
    max_mesh_points: int,
    minimum_axis_size: int = 4,
) -> tuple[tuple[int, ...], ...]:
    """Build anisotropic candidate meshes with nearly uniform physical spacing."""

    refinements = int(max_refinements)
    point_limit = int(max_mesh_points)
    if refinements < policy.consecutive_passes + 1:
        raise ValueError(
            "max_refinements must permit the requested consecutive comparisons"
        )
    if point_limit < 1:
        raise ValueError("max_mesh_points must be positive")
    seed = _periodic_shape(model, seed_shape)
    reciprocal = np.asarray(model.reciprocal_lattice, dtype=float)
    lengths = np.asarray(
        [
            np.linalg.norm(reciprocal[:, axis])
            for axis in model.periodic_axes
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("reciprocal lattice vectors must be finite and nonzero")
    ratios = lengths / np.max(lengths)
    initial_long_axis = max(
        int(minimum_axis_size),
        int(round(max(seed) / 2.0)),
    )
    meshes: list[tuple[int, ...]] = []
    for level in range(refinements):
        long_axis = initial_long_axis * policy.refinement_factor**level
        raw = np.maximum(
            int(minimum_axis_size),
            np.ceil(long_axis * ratios).astype(int),
        )
        # Even grids are convenient for common half-reciprocal transfers and
        # keep the ladder deterministic without imposing a crystallographic
        # assumption.
        shape = tuple(int(value + value % 2) for value in raw)
        if meshes:
            shape = tuple(
                max(value, previous + 2)
                for value, previous in zip(shape, meshes[-1], strict=True)
            )
        if int(np.prod(shape, dtype=np.int64)) > point_limit:
            break
        meshes.append(shape)
    return tuple(meshes)


def _array_comparison(
    coarse: np.ndarray,
    fine: np.ndarray,
    *,
    tolerance: float,
    integrated_axis: np.ndarray | None = None,
) -> tuple[float, float, float, bool]:
    coarse_array = np.asarray(coarse)
    fine_array = np.asarray(fine)
    if coarse_array.shape != fine_array.shape:
        raise ValueError("convergence arrays must have matching shapes")
    delta = np.abs(fine_array - coarse_array)
    scale = max(float(np.max(np.abs(fine_array))), 1.0e-12)
    maximum = float(np.max(delta)) / scale
    fine_norm = float(np.linalg.norm(fine_array.ravel()))
    rms = float(np.linalg.norm(delta.ravel())) / max(fine_norm, 1.0e-12)
    integrated = 0.0
    if integrated_axis is not None:
        axis = np.asarray(integrated_axis, dtype=float)
        spacing = np.diff(axis)

        def integrate(values: np.ndarray) -> float:
            return float(
                np.sum(
                    0.5
                    * (values[..., 1:] + values[..., :-1])
                    * spacing
                )
            )

        numerator = float(
            np.sum(
                [
                    integrate(np.abs(channel))
                    for channel in delta.reshape(-1, delta.shape[-1])
                ]
            )
        )
        denominator = float(
            np.sum(
                [
                    integrate(np.abs(channel))
                    for channel in np.abs(fine_array).reshape(
                        -1, fine_array.shape[-1]
                    )
                ]
            )
        )
        integrated = numerator / max(denominator, 1.0e-12)
    passed = max(maximum, rms, integrated) <= float(tolerance)
    return maximum, rms, integrated, passed


def _certificate_from_evaluations(
    *,
    observable: str,
    policy: SamplingPolicy,
    meshes: Sequence[tuple[int, ...]],
    evaluate: Callable[[tuple[int, ...]], np.ndarray],
    input_digest: str,
    domain: Mapping[str, Any],
    provenance: Mapping[str, Any],
    integrated_axis: np.ndarray | None = None,
    progress_callback: Callable[[SamplingProgress], None] | None = None,
) -> SamplingCertificate:
    attempted: list[tuple[int, ...]] = []
    comparisons: list[SamplingComparison] = []
    previous: np.ndarray | None = None
    previous_mesh: tuple[int, ...] | None = None
    passing_run = 0
    chosen: tuple[int, ...] | None = None
    candidate_count = len(meshes)
    for iteration, mesh in enumerate(meshes, start=1):
        if progress_callback is not None:
            progress_callback(
                SamplingProgress(
                    observable=observable,
                    iteration=iteration,
                    candidate_count=candidate_count,
                    mesh=mesh,
                    phase="started",
                    consecutive_passes=passing_run,
                    required_passes=policy.consecutive_passes,
                )
            )
        started = time.perf_counter()
        values = np.asarray(evaluate(mesh))
        elapsed = time.perf_counter() - started
        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"{observable} convergence produced nonfinite values on {mesh}"
            )
        attempted.append(mesh)
        comparison: SamplingComparison | None = None
        if previous is not None and previous_mesh is not None:
            maximum, rms, integrated, passed = _array_comparison(
                previous,
                values,
                tolerance=policy.relative_tolerance,
                integrated_axis=integrated_axis,
            )
            comparison = SamplingComparison(
                coarse_mesh=previous_mesh,
                fine_mesh=mesh,
                maximum_relative_error=maximum,
                rms_relative_error=rms,
                integrated_relative_error=integrated,
                passed=passed,
            )
            comparisons.append(comparison)
            passing_run = passing_run + 1 if passed else 0
        if progress_callback is not None:
            progress_callback(
                SamplingProgress(
                    observable=observable,
                    iteration=iteration,
                    candidate_count=candidate_count,
                    mesh=mesh,
                    phase="completed",
                    elapsed_seconds=elapsed,
                    comparison=comparison,
                    consecutive_passes=passing_run,
                    required_passes=policy.consecutive_passes,
                )
            )
        if comparison is not None:
            if passing_run >= policy.consecutive_passes:
                chosen = mesh
                break
        previous = values
        previous_mesh = mesh
    status = "certified" if chosen is not None else "budget_exhausted"
    return SamplingCertificate(
        observable=observable,
        status=status,
        policy=policy,
        chosen_mesh=chosen,
        attempted_meshes=tuple(attempted),
        comparisons=tuple(comparisons),
        input_digest=input_digest,
        domain=dict(domain),
        provenance=dict(provenance),
    )


def certify_dos_sampling(
    model: ElectronicModel,
    energy_meV: ArrayLike,
    *,
    seed_mesh: Sequence[int],
    policy: SamplingPolicy | None = None,
    max_refinements: int = 7,
    max_mesh_points: int = 2_000_000,
    symmetry: Literal["auto", "full", "reduced"] = "auto",
    broadening_meV: float = 5.0,
    method: Literal["gaussian", "tetrahedron"] = "gaussian",
    chemical_potential_meV: float = 0.0,
    projections: Mapping[str, Sequence[int]] | None = None,
    backend: str | None = "auto",
    workers: int | None = 0,
    max_batch_bytes: int = 256 * 1024**2,
    progress_callback: Callable[[SamplingProgress], None] | None = None,
) -> SamplingCertificate:
    """Select and certify a fixed DOS mesh on one declared energy domain."""

    selected_policy = sampling_policy() if policy is None else policy
    energy = np.asarray(energy_meV, dtype=float)
    if (
        energy.ndim != 1
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(np.diff(energy) <= 0.0)
    ):
        raise ValueError(
            "DOS certification requires a strictly increasing finite energy grid"
        )
    selected_method = str(method).strip().lower()
    selected_symmetry = str(symmetry).strip().lower()
    if selected_method == "tetrahedron":
        selected_symmetry = "full"
    if projections and selected_symmetry == "reduced":
        raise ValueError("projected DOS cannot require symmetry reduction")
    meshes = automatic_mesh_ladder(
        model,
        seed_mesh,
        policy=selected_policy,
        max_refinements=max_refinements,
        max_mesh_points=max_mesh_points,
    )
    labels = tuple(sorted((projections or {}).keys()))

    def evaluate(shape: tuple[int, ...]) -> np.ndarray:
        mesh = k_mesh(
            model,
            shape,
            symmetry="full" if projections else selected_symmetry,
        )
        result: DensityOfStatesResult = density_of_states(
            model,
            mesh,
            energy,
            broadening_meV=broadening_meV,
            method=selected_method,
            chemical_potential_meV=chemical_potential_meV,
            projections=projections,
            backend=backend,
            workers=workers,
            max_batch_bytes=max_batch_bytes,
        )
        return np.stack(
            [
                result.total_per_meV_cell,
                *(result.projected_per_meV_cell[label] for label in labels),
            ]
        )

    digest = _stable_digest(
        {
            "observable": "density_of_states",
            "model_digest": model.content_digest,
            "energy_meV": energy.tolist(),
            "broadening_meV": float(broadening_meV),
            "method": selected_method,
            "chemical_potential_meV": float(chemical_potential_meV),
            "projections": {
                label: [int(index) for index in indices]
                for label, indices in sorted((projections or {}).items())
            },
            "symmetry": selected_symmetry,
        }
    )
    return _certificate_from_evaluations(
        observable="density_of_states",
        policy=selected_policy,
        meshes=meshes,
        evaluate=evaluate,
        input_digest=digest,
        domain={
            "energy_min_meV": float(energy[0]),
            "energy_max_meV": float(energy[-1]),
            "energy_points": int(energy.size),
            "channels": ["total", *labels],
        },
        provenance={
            "comparison": (
                "successive total and projected DOS curves on one fixed energy grid"
            ),
            "model_digest": model.content_digest,
            "mesh_spacing": "approximately uniform physical reciprocal spacing",
            "symmetry": selected_symmetry,
            "method": selected_method,
            "broadening_meV": float(broadening_meV),
            "chemical_potential_meV": float(chemical_potential_meV),
        },
        integrated_axis=energy,
        progress_callback=progress_callback,
    )


def certify_lindhard_sampling(
    model: ElectronicModel,
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    *,
    seed_mesh: Sequence[int],
    temperature_K: float,
    broadening_meV: float,
    chemical_potential_meV: float = 0.0,
    filling_per_cell: float | None = None,
    policy: SamplingPolicy | None = None,
    max_refinements: int = 7,
    max_mesh_points: int = 500_000,
    mesh_shift: Sequence[float] | None = None,
    symmetry: Literal["auto", "full", "reduced"] = "auto",
    backend: str | None = "auto",
    workers: int | None = 0,
    max_batch_bytes: int = 256 * 1024**2,
    transition_max_batch_bytes: int | None = None,
    transition_backend: Literal["auto", "numpy", "numba"] = "auto",
    progress_callback: Callable[[SamplingProgress], None] | None = None,
) -> SamplingCertificate:
    """Select a fixed mesh from the full complex bare spin response."""

    from .electronic_response import (
        ElectronicResponseCache,
        bare_spin_susceptibility,
        chemical_potential_for_filling,
        response_k_mesh,
    )

    selected_policy = sampling_policy() if policy is None else policy
    q = np.asarray(q_reduced, dtype=float)
    energy = np.asarray(energy_meV, dtype=float)
    if q.ndim == 1:
        q = np.broadcast_to(q, (energy.size, 3))
    if (
        q.shape != (energy.size, 3)
        or energy.ndim != 1
        or energy.size < 1
        or np.any(~np.isfinite(q))
        or np.any(~np.isfinite(energy))
    ):
        raise ValueError(
            "Lindhard certification requires paired finite Q and energy points"
        )
    meshes = automatic_mesh_ladder(
        model,
        seed_mesh,
        policy=selected_policy,
        max_refinements=max_refinements,
        max_mesh_points=max_mesh_points,
    )
    cache = ElectronicResponseCache()

    def evaluate(shape: tuple[int, ...]) -> np.ndarray:
        mesh = response_k_mesh(
            model,
            shape,
            q,
            shift=mesh_shift,
            symmetry=symmetry,
        )
        mu = (
            float(chemical_potential_meV)
            if filling_per_cell is None
            else chemical_potential_for_filling(
                model,
                k_mesh(
                    model,
                    shape,
                    shift=mesh_shift,
                    symmetry="full",
                ),
                float(filling_per_cell),
                temperature_K=float(temperature_K),
                backend=backend,
                workers=workers,
                max_batch_bytes=max_batch_bytes,
                cache=cache,
            )
        )
        result = bare_spin_susceptibility(
            model,
            q,
            energy,
            mesh,
            temperature_K=float(temperature_K),
            chemical_potential_meV=mu,
            broadening_meV=float(broadening_meV),
            backend=backend,
            workers=workers,
            max_batch_bytes=max_batch_bytes,
            transition_max_batch_bytes=transition_max_batch_bytes,
            transition_backend=transition_backend,
            q_evaluation="direct",
            cache=cache,
        )
        return np.asarray(result.values_per_meV_cell)

    digest = _stable_digest(
        {
            "observable": "lindhard",
            "model_digest": model.content_digest,
            "q_reduced": q.tolist(),
            "energy_meV": energy.tolist(),
            "temperature_K": float(temperature_K),
            "broadening_meV": float(broadening_meV),
            "chemical_potential_meV": float(chemical_potential_meV),
            "filling_per_cell": (
                None if filling_per_cell is None else float(filling_per_cell)
            ),
            "mesh_shift": (
                None
                if mesh_shift is None
                else [float(value) for value in mesh_shift]
            ),
            "symmetry": str(symmetry),
        }
    )
    return _certificate_from_evaluations(
        observable="lindhard",
        policy=selected_policy,
        meshes=meshes,
        evaluate=evaluate,
        input_digest=digest,
        domain={
            "points": int(energy.size),
            "energy_min_meV": float(np.min(energy)),
            "energy_max_meV": float(np.max(energy)),
            "temperature_K": float(temperature_K),
            "q_min_reduced": np.min(q, axis=0).tolist(),
            "q_max_reduced": np.max(q, axis=0).tolist(),
            "tensor": "complex Cartesian spin susceptibility",
        },
        provenance={
            "comparison": (
                "successive full complex Cartesian spin-response tensors"
            ),
            "model_digest": model.content_digest,
            "mesh_spacing": "approximately uniform physical reciprocal spacing",
            "q_evaluation_during_certificate": "direct",
            "symmetry": str(symmetry),
            "broadening_meV": float(broadening_meV),
            "chemical_potential_mode": (
                "source" if filling_per_cell is None else "filling"
            ),
            "filling_per_cell": (
                None if filling_per_cell is None else float(filling_per_cell)
            ),
        },
        progress_callback=progress_callback,
    )
