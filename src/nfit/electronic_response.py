"""Bare multiband Lindhard susceptibilities and magnetic projections."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cache_utils import lru_store
from .cross_section import KB_MEV_PER_K
from .electronic_backends import (
    ElectronicBackend,
    ElectronicEigensystem,
    evaluate_eigensystem,
)
from .electronic_structure import ElectronicModel, WavevectorSampling

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _readonly(value: ArrayLike, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass
class ElectronicResponseCache:
    """Bounded in-memory cache of immutable electronic eigensystems."""

    max_bytes: int = 512 * 1024**2
    max_entries: int = 64
    hits: int = 0
    misses: int = 0
    _entries: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def __post_init__(self) -> None:
        if int(self.max_bytes) < 0 or int(self.max_entries) < 0:
            raise ValueError("response cache limits must be nonnegative")
        self.max_bytes = int(self.max_bytes)
        self.max_entries = int(self.max_entries)

    def get(self, key: tuple[Any, ...]) -> ElectronicEigensystem | None:
        """Return and refresh one cached eigensystem."""

        result = self._entries.get(key)
        if result is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return result

    def put(
        self,
        key: tuple[Any, ...],
        value: ElectronicEigensystem,
    ) -> None:
        """Store one eigensystem within count and numerical-byte limits."""

        lru_store(
            self._entries,
            key,
            value,
            self.max_entries,
            max_array_bytes=self.max_bytes,
        )

    def clear(self) -> None:
        """Remove cached arrays and reset hit/miss counters."""

        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def entries(self) -> int:
        return len(self._entries)


def _coordinate_digest(coordinates: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(coordinates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cached_eigensystem(
    model: ElectronicModel,
    coordinates: np.ndarray,
    *,
    eigenvectors: bool,
    backend: ElectronicBackend | str | None,
    workers: int | None,
    max_batch_bytes: int,
    cache: ElectronicResponseCache | None,
) -> ElectronicEigensystem:
    key = (
        model.content_digest,
        _coordinate_digest(coordinates),
        bool(eigenvectors),
        None if backend is None else str(backend),
        workers,
        int(max_batch_bytes),
    )
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
        if not eigenvectors:
            richer_key = (*key[:2], True, *key[3:])
            cached = cache.get(richer_key)
            if cached is not None:
                return cached
    result = evaluate_eigensystem(
        model,
        coordinates,
        eigenvectors=eigenvectors,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
    )
    if cache is not None:
        cache.put(key, result)
    return result


@dataclass(frozen=True)
class ElectronicOperatorBasis:
    """Ordered one-particle operators used to construct a response matrix."""

    labels: tuple[str, ...]
    matrices: ComplexArray
    conjugate_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrices = _readonly(self.matrices, np.complex128)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("operator matrices must have shape (n, basis, basis)")
        if len(self.labels) != matrices.shape[0] or len(set(self.labels)) != len(
            self.labels
        ):
            raise ValueError("operator labels must be unique and match the matrices")
        conjugates = self.conjugate_indices or tuple(range(len(self.labels)))
        if (
            len(conjugates) != len(self.labels)
            or any(index < 0 or index >= len(self.labels) for index in conjugates)
            or any(conjugates[conjugates[index]] != index for index in conjugates)
        ):
            raise ValueError("operator conjugate indices must define an involution")
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "conjugate_indices", tuple(conjugates))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def size(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible operator-basis description."""

        return {
            "labels": list(self.labels),
            "matrices_real": self.matrices.real.tolist(),
            "matrices_imag": self.matrices.imag.tolist(),
            "conjugate_indices": list(self.conjugate_indices),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ElectronicOperatorBasis:
        """Reconstruct an operator basis from :meth:`to_dict` output."""

        matrices = np.asarray(payload["matrices_real"], dtype=float) + 1.0j * np.asarray(
            payload["matrices_imag"], dtype=float
        )
        return cls(
            labels=tuple(str(value) for value in payload["labels"]),
            matrices=matrices,
            conjugate_indices=tuple(
                int(value) for value in payload["conjugate_indices"]
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SusceptibilityResult:
    """Complex susceptibility evaluated at paired wavevector-energy points."""

    q_reduced: FloatArray
    energy_meV: FloatArray
    values_per_meV_cell: ComplexArray
    operator_labels: tuple[str, ...]
    conjugate_indices: tuple[int, ...]
    model_digest: str
    temperature_K: float
    chemical_potential_meV: float
    broadening_meV: float
    Q_reduced: FloatArray | None = None
    response_kind: str = "bare"
    normalization: str = "per primitive cell"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        q = _readonly(self.q_reduced, float)
        energy = _readonly(self.energy_meV, float)
        values = _readonly(self.values_per_meV_cell, np.complex128)
        transferred = q if self.Q_reduced is None else _readonly(self.Q_reduced, float)
        count = len(self.operator_labels)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("q_reduced must have shape (n, 3)")
        if energy.shape != (q.shape[0],):
            raise ValueError("energy_meV must contain one value per wavevector")
        if values.shape != (q.shape[0], count, count):
            raise ValueError(
                "susceptibility values must have shape (n, operators, operators)"
            )
        if len(self.conjugate_indices) != count:
            raise ValueError("conjugate map must match the operator basis")
        if transferred.shape != q.shape:
            raise ValueError("Q_reduced must match q_reduced")
        object.__setattr__(self, "q_reduced", q)
        object.__setattr__(self, "Q_reduced", transferred)
        object.__setattr__(self, "energy_meV", energy)
        object.__setattr__(self, "values_per_meV_cell", values)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def chi_prime(self) -> FloatArray:
        return np.asarray(self.values_per_meV_cell.real)

    @property
    def chi_double_prime(self) -> FloatArray:
        return np.asarray(self.values_per_meV_cell.imag)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result including complex response values."""

        return {
            "q_reduced": self.q_reduced.tolist(),
            "Q_reduced": self.Q_reduced.tolist(),
            "energy_meV": self.energy_meV.tolist(),
            "values_real": self.values_per_meV_cell.real.tolist(),
            "values_imag": self.values_per_meV_cell.imag.tolist(),
            "operator_labels": list(self.operator_labels),
            "conjugate_indices": list(self.conjugate_indices),
            "model_digest": self.model_digest,
            "temperature_K": self.temperature_K,
            "chemical_potential_meV": self.chemical_potential_meV,
            "broadening_meV": self.broadening_meV,
            "response_kind": self.response_kind,
            "normalization": self.normalization,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SusceptibilityResult:
        """Reconstruct a result from :meth:`to_dict` output."""

        values = np.asarray(payload["values_real"], dtype=float) + 1.0j * np.asarray(
            payload["values_imag"], dtype=float
        )
        return cls(
            q_reduced=payload["q_reduced"],
            Q_reduced=payload.get("Q_reduced"),
            energy_meV=payload["energy_meV"],
            values_per_meV_cell=values,
            operator_labels=tuple(str(value) for value in payload["operator_labels"]),
            conjugate_indices=tuple(
                int(value) for value in payload["conjugate_indices"]
            ),
            model_digest=str(payload["model_digest"]),
            temperature_K=float(payload["temperature_K"]),
            chemical_potential_meV=float(payload["chemical_potential_meV"]),
            broadening_meV=float(payload["broadening_meV"]),
            response_kind=str(payload.get("response_kind", "bare")),
            normalization=str(
                payload.get("normalization", "per primitive cell")
            ),
            provenance=dict(payload.get("provenance", {})),
        )


@dataclass(frozen=True)
class ResponseConvergenceResult:
    """Mesh and broadening convergence of a complex scalar response."""

    mesh_shapes: tuple[tuple[int, ...], ...]
    broadenings_meV: FloatArray
    q_reduced: FloatArray
    energy_meV: FloatArray
    values_per_meV_cell: ComplexArray
    mesh_max_absolute_error: FloatArray
    mesh_max_relative_error: FloatArray
    broadening_max_absolute_change: FloatArray
    broadening_max_relative_change: FloatArray
    reference_mesh_index: int
    reference_broadening_index: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        broadenings = _readonly(self.broadenings_meV, float)
        q = _readonly(self.q_reduced, float)
        energy = _readonly(self.energy_meV, float)
        values = _readonly(self.values_per_meV_cell, np.complex128)
        mesh_absolute = _readonly(self.mesh_max_absolute_error, float)
        mesh_relative = _readonly(self.mesh_max_relative_error, float)
        broad_absolute = _readonly(self.broadening_max_absolute_change, float)
        broad_relative = _readonly(self.broadening_max_relative_change, float)
        expected = (len(self.mesh_shapes), broadenings.size, energy.size)
        if values.shape != expected or q.shape != (energy.size, 3):
            raise ValueError("convergence values must match mesh, broadening, and points")
        if mesh_absolute.shape != expected[:2] or mesh_relative.shape != expected[:2]:
            raise ValueError("mesh convergence metrics must match mesh and broadening")
        if broad_absolute.shape != expected[:2] or broad_relative.shape != expected[:2]:
            raise ValueError(
                "broadening convergence metrics must match mesh and broadening"
            )
        if not 0 <= int(self.reference_mesh_index) < len(self.mesh_shapes):
            raise ValueError("reference_mesh_index is out of range")
        if not 0 <= int(self.reference_broadening_index) < broadenings.size:
            raise ValueError("reference_broadening_index is out of range")
        object.__setattr__(self, "broadenings_meV", broadenings)
        object.__setattr__(self, "q_reduced", q)
        object.__setattr__(self, "energy_meV", energy)
        object.__setattr__(self, "values_per_meV_cell", values)
        object.__setattr__(self, "mesh_max_absolute_error", mesh_absolute)
        object.__setattr__(self, "mesh_max_relative_error", mesh_relative)
        object.__setattr__(
            self,
            "broadening_max_absolute_change",
            broad_absolute,
        )
        object.__setattr__(
            self,
            "broadening_max_relative_change",
            broad_relative,
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible convergence record."""

        return {
            "mesh_shapes": [list(shape) for shape in self.mesh_shapes],
            "broadenings_meV": self.broadenings_meV.tolist(),
            "q_reduced": self.q_reduced.tolist(),
            "energy_meV": self.energy_meV.tolist(),
            "values_real": self.values_per_meV_cell.real.tolist(),
            "values_imag": self.values_per_meV_cell.imag.tolist(),
            "mesh_max_absolute_error": self.mesh_max_absolute_error.tolist(),
            "mesh_max_relative_error": self.mesh_max_relative_error.tolist(),
            "broadening_max_absolute_change": (
                self.broadening_max_absolute_change.tolist()
            ),
            "broadening_max_relative_change": (
                self.broadening_max_relative_change.tolist()
            ),
            "reference_mesh_index": int(self.reference_mesh_index),
            "reference_broadening_index": int(self.reference_broadening_index),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResponseConvergenceResult:
        """Reconstruct a convergence result from :meth:`to_dict` output."""

        values = np.asarray(payload["values_real"], dtype=float) + 1.0j * np.asarray(
            payload["values_imag"], dtype=float
        )
        return cls(
            mesh_shapes=tuple(
                tuple(int(value) for value in shape)
                for shape in payload["mesh_shapes"]
            ),
            broadenings_meV=payload["broadenings_meV"],
            q_reduced=payload["q_reduced"],
            energy_meV=payload["energy_meV"],
            values_per_meV_cell=values,
            mesh_max_absolute_error=payload["mesh_max_absolute_error"],
            mesh_max_relative_error=payload["mesh_max_relative_error"],
            broadening_max_absolute_change=payload[
                "broadening_max_absolute_change"
            ],
            broadening_max_relative_change=payload[
                "broadening_max_relative_change"
            ],
            reference_mesh_index=int(payload["reference_mesh_index"]),
            reference_broadening_index=int(
                payload["reference_broadening_index"]
            ),
            provenance=dict(payload.get("provenance", {})),
        )


def response_convergence_scan(
    model: ElectronicModel,
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    *,
    mesh_shapes: Sequence[Sequence[int]],
    broadenings_meV: Sequence[float],
    temperature_K: float,
    chemical_potential_meV: float = 0.0,
    mesh_shift: Sequence[float] | None = None,
    relative_floor: float = 1.0e-12,
    reference_mesh_index: int = -1,
    reference_broadening_index: int = -1,
    backend: ElectronicBackend | str | None = "numpy",
    workers: int | None = 1,
    max_batch_bytes: int = 256 * 1024**2,
    transition_max_batch_bytes: int | None = None,
    cache: ElectronicResponseCache | None = None,
) -> ResponseConvergenceResult:
    """Evaluate mesh and broadening convergence as separate numerical axes."""

    from .electronic_structure import k_mesh

    shapes = tuple(tuple(int(value) for value in shape) for shape in mesh_shapes)
    if not shapes or any(not shape or any(value < 1 for value in shape) for shape in shapes):
        raise ValueError("mesh_shapes must contain positive mesh dimensions")
    broadenings = np.asarray(broadenings_meV, dtype=float)
    if (
        broadenings.ndim != 1
        or broadenings.size < 1
        or np.any(~np.isfinite(broadenings))
        or np.any(broadenings <= 0.0)
    ):
        raise ValueError("broadenings_meV must contain positive finite values")
    floor = float(relative_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    Q, energy = _point_inputs(q_reduced, energy_meV)
    mesh_reference = int(reference_mesh_index) % len(shapes)
    broadening_reference = int(reference_broadening_index) % broadenings.size
    response_cache = cache or ElectronicResponseCache()
    values = np.empty(
        (len(shapes), broadenings.size, energy.size),
        dtype=np.complex128,
    )
    records = []
    for mesh_index, shape in enumerate(shapes):
        mesh = k_mesh(model, shape, shift=mesh_shift, symmetry="full")
        mesh_records = []
        for broadening_index, broadening in enumerate(broadenings):
            response = bare_spin_susceptibility(
                model,
                Q,
                energy,
                mesh,
                temperature_K=temperature_K,
                chemical_potential_meV=chemical_potential_meV,
                broadening_meV=float(broadening),
                backend=backend,
                workers=workers,
                max_batch_bytes=max_batch_bytes,
                transition_max_batch_bytes=transition_max_batch_bytes,
                cache=response_cache,
            )
            values[mesh_index, broadening_index] = isotropic_spin_component(
                response
            )
            mesh_records.append(dict(response.provenance))
        records.append(mesh_records)

    mesh_reference_values = values[mesh_reference][None, :, :]
    mesh_delta = np.abs(values - mesh_reference_values)
    mesh_absolute = np.max(mesh_delta, axis=2)
    mesh_relative = np.max(
        mesh_delta / np.maximum(np.abs(mesh_reference_values), floor),
        axis=2,
    )
    broad_reference_values = values[:, broadening_reference][:, None, :]
    broad_delta = np.abs(values - broad_reference_values)
    broad_absolute = np.max(broad_delta, axis=2)
    broad_relative = np.max(
        broad_delta / np.maximum(np.abs(broad_reference_values), floor),
        axis=2,
    )
    return ResponseConvergenceResult(
        mesh_shapes=shapes,
        broadenings_meV=broadenings,
        q_reduced=np.mod(Q, 1.0),
        energy_meV=energy,
        values_per_meV_cell=values,
        mesh_max_absolute_error=mesh_absolute,
        mesh_max_relative_error=mesh_relative,
        broadening_max_absolute_change=broad_absolute,
        broadening_max_relative_change=broad_relative,
        reference_mesh_index=mesh_reference,
        reference_broadening_index=broadening_reference,
        provenance={
            "model_digest": model.content_digest,
            "temperature_K": float(temperature_K),
            "chemical_potential_meV": float(chemical_potential_meV),
            "mesh_shift": (
                None if mesh_shift is None else [float(value) for value in mesh_shift]
            ),
            "relative_floor": floor,
            "response_records": records,
            "comparison": {
                "mesh": "each mesh versus reference mesh at fixed broadening",
                "broadening": (
                    "each broadening versus reference broadening at fixed mesh"
                ),
            },
        },
    )


def orbital_pair_operator_basis(
    model: ElectronicModel,
    basis_indices: tuple[int, ...] | list[int] | None = None,
) -> ElectronicOperatorBasis:
    """Return an ordered ``|a><b|`` basis for all or selected basis states."""

    indices = (
        tuple(range(model.n_basis))
        if basis_indices is None
        else tuple(int(value) for value in basis_indices)
    )
    if len(indices) != len(set(indices)) or any(
        value < 0 or value >= model.n_basis for value in indices
    ):
        raise ValueError("basis_indices must contain unique valid basis indices")
    labels = []
    matrices = []
    conjugates = []
    for local_a, a in enumerate(indices):
        state_a = model.basis[a]
        for local_b, b in enumerate(indices):
            state_b = model.basis[b]
            matrix = np.zeros((model.n_basis, model.n_basis), dtype=np.complex128)
            matrix[a, b] = 1.0
            matrices.append(matrix)
            labels.append(f"{state_a.label}←{state_b.label}")
            conjugates.append(local_b * len(indices) + local_a)
    return ElectronicOperatorBasis(
        tuple(labels),
        np.asarray(matrices),
        tuple(conjugates),
        metadata={
            "kind": "ordered_orbital_pairs",
            "multiplication": "chi_AB=<O_A O_B^dagger>",
            "basis_indices": list(indices),
        },
    )


def electron_filling(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    *,
    chemical_potential_meV: float,
    temperature_K: float,
    backend: ElectronicBackend | str | None = "numpy",
    workers: int | None = 1,
    max_batch_bytes: int = 256 * 1024**2,
    cache: ElectronicResponseCache | None = None,
) -> float:
    """Return electrons per primitive cell for a chemical potential."""

    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("electron filling requires an integration mesh")
    eigensystem = _cached_eigensystem(
        model,
        mesh.reduced_coordinates,
        eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
        cache=cache,
    )
    spin_degeneracy = int(
        model.provenance.get(
            "implicit_spin_degeneracy",
            2 if model.spin_operators is None else 1,
        )
    )
    occupations = _fermi_function(
        eigensystem.eigenvalues,
        chemical_potential_meV,
        temperature_K,
    )
    return float(
        spin_degeneracy
        * np.einsum("k,kn->", np.asarray(mesh.weights), occupations)
    )


def chemical_potential_for_filling(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    filling_per_cell: float,
    *,
    temperature_K: float,
    backend: ElectronicBackend | str | None = "numpy",
    workers: int | None = 1,
    max_batch_bytes: int = 256 * 1024**2,
    cache: ElectronicResponseCache | None = None,
) -> float:
    """Solve the finite-temperature chemical potential for a target filling."""

    from scipy.optimize import brentq

    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("filling-based chemical potential requires temperature > 0")
    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("filling-based chemical potential requires an integration mesh")
    eigensystem = _cached_eigensystem(
        model,
        mesh.reduced_coordinates,
        eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
        cache=cache,
    )
    degeneracy = int(
        model.provenance.get(
            "implicit_spin_degeneracy",
            2 if model.spin_operators is None else 1,
        )
    )
    capacity = float(degeneracy * model.n_basis)
    target = float(filling_per_cell)
    if not np.isfinite(target) or not 0.0 < target < capacity:
        raise ValueError(
            f"filling_per_cell must lie strictly between 0 and {capacity:g}"
        )
    margin = max(100.0 * KB_MEV_PER_K * temperature, 1.0)
    lower = float(np.min(eigensystem.eigenvalues) - margin)
    upper = float(np.max(eigensystem.eigenvalues) + margin)
    weights = np.asarray(mesh.weights)

    def residual(mu: float) -> float:
        occupations = _fermi_function(
            eigensystem.eigenvalues,
            mu,
            temperature,
        )
        filling = degeneracy * np.einsum("k,kn->", weights, occupations)
        return float(filling - target)

    return float(brentq(residual, lower, upper, xtol=1.0e-11, rtol=1.0e-13))


def _fermi_function(
    energy_meV: np.ndarray,
    chemical_potential_meV: float,
    temperature_K: float,
) -> np.ndarray:
    shifted = np.asarray(energy_meV, dtype=float) - float(chemical_potential_meV)
    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature_K must be finite and nonnegative")
    if temperature == 0.0:
        return np.where(shifted < 0.0, 1.0, np.where(shifted > 0.0, 0.0, 0.5))
    argument = np.clip(shifted / (KB_MEV_PER_K * temperature), -700.0, 700.0)
    return 1.0 / (np.exp(argument) + 1.0)


def _static_equal_energy_limit(
    energy_meV: np.ndarray,
    occupation: np.ndarray,
    *,
    chemical_potential_meV: float,
    temperature_K: float,
    broadening_meV: float,
) -> np.ndarray:
    if temperature_K > 0.0:
        return occupation * (1.0 - occupation) / (
            KB_MEV_PER_K * temperature_K
        )
    offset = energy_meV - chemical_potential_meV
    return broadening_meV / (
        np.pi * (offset * offset + broadening_meV * broadening_meV)
    )


def _point_inputs(
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    q = np.asarray(q_reduced, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    energy = np.asarray(energy_meV, dtype=float)
    if energy.ndim == 0:
        energy = np.full(q.shape[0], float(energy))
    q, energy_column = np.broadcast_arrays(q, energy[:, None])
    q = np.asarray(q[:, :3], dtype=float)
    energy = np.asarray(energy_column[:, 0], dtype=float)
    if (
        q.ndim != 2
        or q.shape[1] != 3
        or energy.shape != (q.shape[0],)
        or np.any(~np.isfinite(q))
        or np.any(~np.isfinite(energy))
    ):
        raise ValueError("q and energy must define finite paired response points")
    return q, energy


def response_k_mesh(
    model: ElectronicModel,
    shape: Sequence[int],
    q_reduced: ArrayLike,
    *,
    shift: Sequence[float] | None = None,
    symmetry: Literal["auto", "full", "reduced"] = "auto",
    operator_kind: str = "implicit_isotropic_spin",
) -> WavevectorSampling:
    """Return a fail-closed little-group mesh for a response calculation.

    Reduction is certified only for an nfit-built, implicit-spin model and a
    physical isotropic-spin response. Every retained operation fixes every
    requested transferred wavevector modulo a reciprocal-lattice vector.
    """

    from dataclasses import replace

    from .electronic_structure import k_mesh

    policy = str(symmetry).strip().lower()
    if policy not in {"auto", "full", "reduced"}:
        raise ValueError("response symmetry policy must be auto, full, or reduced")
    full = k_mesh(model, shape, shift=shift, symmetry="full")
    if policy == "full":
        return replace(
            full,
            provenance={
                **dict(full.provenance),
                "response_symmetry_reduction": {
                    "policy": "full",
                    "applied": False,
                    "reason": "full mesh requested",
                },
            },
        )

    def unchanged(reason: str) -> WavevectorSampling:
        if policy == "reduced":
            raise ValueError(
                f"response symmetry reduction is not certified: {reason}"
            )
        return replace(
            full,
            provenance={
                **dict(full.provenance),
                "response_symmetry_reduction": {
                    "policy": "auto",
                    "applied": False,
                    "reason": reason,
                },
            },
        )

    if model.dimension != 3:
        return unchanged("only three-dimensional response meshes are reduced")
    if operator_kind != "implicit_isotropic_spin" or model.spin_operators is not None:
        return unchanged(
            "the requested operator basis is not certified as implicit isotropic spin"
        )
    if int(model.provenance.get("implicit_spin_degeneracy", 2)) != 2:
        return unchanged("the implicit spin degeneracy is not two")
    metadata = model.provenance.get("reciprocal_symmetry", {})
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("certified_by") != "nfit_orbital_builder"
    ):
        return unchanged("model has no nfit-certified reciprocal symmetry")
    q_input = np.asarray(q_reduced, dtype=float)
    point_count = 1 if q_input.ndim == 1 else q_input.shape[0]
    Q, _energy = _point_inputs(q_input, np.zeros(point_count))
    reduced_q = np.mod(Q, 1.0)
    candidates: list[np.ndarray] = []
    signs = (1, -1) if bool(metadata.get("includes_time_reversal")) else (1,)
    for raw_rotation in metadata.get("rotations", ()):
        rotation = np.asarray(raw_rotation, dtype=np.int64)
        if rotation.shape != (3, 3):
            return unchanged("certified rotation metadata is invalid")
        for sign in signs:
            transform = sign * rotation
            mapped_q = reduced_q @ transform.T
            if np.allclose(
                mapped_q - reduced_q,
                np.rint(mapped_q - reduced_q),
                atol=1.0e-9,
            ):
                candidates.append(transform)
    unique_candidates = []
    for candidate in candidates:
        if not any(np.array_equal(candidate, item) for item in unique_candidates):
            unique_candidates.append(candidate)
    if len(unique_candidates) < 2:
        return unchanged("the requested wavevectors have no nontrivial certified little group")

    mesh_shape = np.asarray(full.mesh_shape, dtype=np.int64)
    mesh_shift = np.asarray(full.shift, dtype=float)
    indices = np.indices(tuple(mesh_shape), dtype=np.int64).reshape(3, -1).T
    coordinates = (indices + mesh_shift[None, :]) / mesh_shape[None, :]
    representatives = np.arange(indices.shape[0], dtype=np.int64)
    compatible = 0
    for transform in unique_candidates:
        transformed = coordinates @ transform.T
        transformed_indices = (
            transformed * mesh_shape[None, :] - mesh_shift[None, :]
        )
        rounded = np.rint(transformed_indices).astype(np.int64)
        if not np.allclose(transformed_indices, rounded, atol=1.0e-9):
            continue
        wrapped = np.mod(rounded, mesh_shape[None, :])
        flat = np.ravel_multi_index(wrapped.T, tuple(mesh_shape))
        representatives = np.minimum(representatives, flat)
        compatible += 1
    if compatible < 2:
        return unchanged("mesh shape or shift is incompatible with the little group")
    unique, counts = np.unique(representatives, return_counts=True)
    if np.any(representatives[unique] != unique):
        return unchanged("certified little-group operations do not close on this mesh")
    return WavevectorSampling(
        "mesh",
        full.reduced_coordinates[unique],
        weights=counts.astype(float) / float(indices.shape[0]),
        mesh_shape=full.mesh_shape,
        shift=full.shift,
        provenance={
            **dict(full.provenance),
            "response_symmetry_reduction": {
                "policy": policy,
                "applied": True,
                "operator_kind": operator_kind,
                "full_size": int(indices.shape[0]),
                "irreducible_size": int(unique.size),
                "little_group_operation_count": int(compatible),
                "q_reduced": reduced_q.tolist(),
            },
        },
    )


def bare_lindhard_susceptibility(
    model: ElectronicModel,
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    mesh: WavevectorSampling,
    operators: ElectronicOperatorBasis,
    *,
    temperature_K: float,
    chemical_potential_meV: float = 0.0,
    broadening_meV: float = 1.0,
    operator_matrices_by_point: ArrayLike | None = None,
    backend: ElectronicBackend | str | None = "numpy",
    workers: int | None = 1,
    max_batch_bytes: int = 256 * 1024**2,
    transition_max_batch_bytes: int | None = None,
    cache: ElectronicResponseCache | None = None,
) -> SusceptibilityResult:
    """Evaluate the causal generalized Lindhard response.

    The response is evaluated at paired ``(q, E)`` points. The integration mesh
    may be weighted; its provenance records whether it is a full mesh or a
    certified response little-group reduction.
    """

    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("Lindhard susceptibility requires an integration mesh")
    if operators.matrices.shape[1:] != (model.n_basis, model.n_basis):
        raise ValueError("operator matrices must match the electronic basis")
    eta = float(broadening_meV)
    mu = float(chemical_potential_meV)
    temperature = float(temperature_K)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("broadening_meV must be finite and positive")
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_meV must be finite")
    transferred_q, energy = _point_inputs(q_reduced, energy_meV)
    q = np.mod(transferred_q, 1.0)
    point_operators = (
        np.broadcast_to(
            np.asarray(operators.matrices)[None, ...],
            (q.shape[0], *operators.matrices.shape),
        )
        if operator_matrices_by_point is None
        else np.asarray(operator_matrices_by_point, dtype=np.complex128)
    )
    if point_operators.shape != (
        q.shape[0],
        operators.size,
        model.n_basis,
        model.n_basis,
    ):
        raise ValueError(
            "point-dependent operators must have shape "
            "(n_points, n_operators, n_basis, n_basis)"
        )

    k = np.asarray(mesh.reduced_coordinates, dtype=float)
    weights = np.asarray(mesh.weights, dtype=float)
    transition_budget = int(
        max_batch_bytes
        if transition_max_batch_bytes is None
        else transition_max_batch_bytes
    )
    if transition_budget < 1:
        raise ValueError("transition_max_batch_bytes must be positive")
    cache_hits_before = 0 if cache is None else cache.hits
    cache_misses_before = 0 if cache is None else cache.misses
    base = _cached_eigensystem(
        model,
        k,
        eigenvectors=True,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
        cache=cache,
    )
    if base.eigenvectors is None:  # pragma: no cover - defensive
        raise RuntimeError("Lindhard response requires band eigenvectors")
    occupation_k = _fermi_function(base.eigenvalues, mu, temperature)
    result = np.empty(
        (q.shape[0], operators.size, operators.size),
        dtype=np.complex128,
    )
    unique_q, inverse = np.unique(q, axis=0, return_inverse=True)
    execution_records = []
    transition_batch_sizes = []
    for q_index, q_value in enumerate(unique_q):
        point_indices = np.flatnonzero(inverse == q_index)
        reference_operators = point_operators[point_indices[0]]
        if not np.allclose(
            point_operators[point_indices],
            reference_operators[None, ...],
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError("equal q points must use equal operator matrices")
        shifted = _cached_eigensystem(
            model,
            k + q_value[None, :],
            eigenvectors=True,
            backend=backend,
            workers=workers,
            max_batch_bytes=max_batch_bytes,
            cache=cache,
        )
        if shifted.eigenvectors is None:  # pragma: no cover - defensive
            raise RuntimeError("Lindhard response requires band eigenvectors")
        execution_records.append(dict(shifted.provenance))
        occupation_q = _fermi_function(shifted.eigenvalues, mu, temperature)
        bands = model.n_basis
        per_k_bytes = max(
            1,
            16 * operators.size * bands * bands
            + 64 * bands * bands
            + 32 * bands,
        )
        transition_batch = max(
            1,
            min(k.shape[0], transition_budget // per_k_bytes),
        )
        transition_batch_sizes.append(transition_batch)
        for point_index in point_indices:
            result[point_index] = 0.0
        for start in range(0, k.shape[0], transition_batch):
            stop = min(start + transition_batch, k.shape[0])
            base_values = base.eigenvalues[start:stop]
            shifted_values = shifted.eigenvalues[start:stop]
            occupation_base = occupation_k[start:stop]
            occupation_shifted = occupation_q[start:stop]
            delta_energy = (
                base_values[:, :, None] - shifted_values[:, None, :]
            )
            occupation_difference = (
                occupation_base[:, :, None] - occupation_shifted[:, None, :]
            )
            matrix_elements = np.einsum(
                "kan,Aab,kbm->kAnm",
                base.eigenvectors[start:stop].conj(),
                reference_operators,
                shifted.eigenvectors[start:stop],
                optimize=True,
            )
            equal = np.abs(delta_energy) <= 1.0e-10
            static_limit = _static_equal_energy_limit(
                0.5 * (base_values[:, :, None] + shifted_values[:, None, :]),
                0.5
                * (
                    occupation_base[:, :, None]
                    + occupation_shifted[:, None, :]
                ),
                chemical_potential_meV=mu,
                temperature_K=temperature,
                broadening_meV=eta,
            )
            for point_index in point_indices:
                transferred = energy[point_index]
                kernel = -occupation_difference / (
                    transferred + delta_energy + 1.0j * eta
                )
                if abs(transferred) <= 1.0e-14:
                    kernel = np.where(equal, static_limit, kernel)
                result[point_index] += np.einsum(
                    "k,knm,kAnm,kBnm->AB",
                    weights[start:stop],
                    kernel,
                    matrix_elements,
                    matrix_elements.conj(),
                    optimize=True,
                )

    return SusceptibilityResult(
        q_reduced=q,
        Q_reduced=transferred_q,
        energy_meV=energy,
        values_per_meV_cell=result,
        operator_labels=operators.labels,
        conjugate_indices=operators.conjugate_indices,
        model_digest=model.content_digest,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        broadening_meV=eta,
        provenance={
            "formula": "generalized_lindhard",
            "sign": "-(f_nk-f_mkq)/(E+e_nk-e_mkq+i eta)",
            "operator_basis": dict(operators.metadata),
            "extended_zone_Q_reduced": transferred_q.tolist(),
            "mesh": mesh.to_dict(),
            "base_execution": dict(base.provenance),
            "shifted_execution": execution_records,
            "precision": "float64/complex128",
            "symmetry": dict(mesh.provenance).get(
                "response_symmetry_reduction",
                {
                    "policy": "full",
                    "applied": False,
                    "reason": "no response reduction metadata",
                },
            ),
            "approximation": "finite lifetime broadening only",
            "transition_batch_size": (
                min(transition_batch_sizes) if transition_batch_sizes else 0
            ),
            "transition_max_batch_bytes": transition_budget,
            "cache": {
                "enabled": cache is not None,
                "hits": (
                    0 if cache is None else cache.hits - cache_hits_before
                ),
                "misses": (
                    0 if cache is None else cache.misses - cache_misses_before
                ),
                "entries": 0 if cache is None else cache.entries,
            },
        },
    )


def spin_operator_matrices(
    model: ElectronicModel,
    q_reduced: ArrayLike,
) -> tuple[ElectronicOperatorBasis, ComplexArray, float]:
    """Return position-phased spin operators and an implicit-spin prefactor."""

    q = np.asarray(q_reduced, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_reduced must have shape (n, 3)")
    phase = np.exp(
        2.0j * np.pi * q @ np.asarray(model.orbital_centers, dtype=float).T
    )
    if model.spin_operators is None:
        degeneracy = int(model.provenance.get("implicit_spin_degeneracy", 2))
        if degeneracy != 2:
            raise ValueError(
                "a model without explicit spin operators must declare "
                "implicit_spin_degeneracy=2 for a physical spin response"
            )
        identity = np.eye(model.n_basis, dtype=np.complex128)
        matrices = phase[:, None, :, None] * identity[None, None, :, :]
        basis = ElectronicOperatorBasis(
            ("spin",),
            identity[None, ...],
            metadata={"kind": "implicit_isotropic_spin"},
        )
        return basis, matrices, degeneracy / 4.0
    matrices = (
        phase[:, None, :, None]
        * np.asarray(model.spin_operators, dtype=np.complex128)[None, ...]
    )
    basis = ElectronicOperatorBasis(
        ("Sx", "Sy", "Sz"),
        np.asarray(model.spin_operators),
        metadata={"kind": "cartesian_spin", "position_phases": True},
    )
    return basis, matrices, 1.0


def bare_spin_susceptibility(
    model: ElectronicModel,
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    mesh: WavevectorSampling,
    **kwargs: Any,
) -> SusceptibilityResult:
    """Return the physical Cartesian spin susceptibility per primitive cell."""

    Q, energy = _point_inputs(q_reduced, energy_meV)
    basis, matrices, prefactor = spin_operator_matrices(model, Q)
    response = bare_lindhard_susceptibility(
        model,
        Q,
        energy,
        mesh,
        basis,
        operator_matrices_by_point=matrices,
        **kwargs,
    )
    if model.spin_operators is not None:
        return response
    scalar = prefactor * response.values_per_meV_cell[:, 0, 0]
    tensor = np.zeros((Q.shape[0], 3, 3), dtype=np.complex128)
    tensor[:, 0, 0] = scalar
    tensor[:, 1, 1] = scalar
    tensor[:, 2, 2] = scalar
    return SusceptibilityResult(
        q_reduced=response.q_reduced,
        Q_reduced=response.Q_reduced,
        energy_meV=response.energy_meV,
        values_per_meV_cell=tensor,
        operator_labels=("Sx", "Sy", "Sz"),
        conjugate_indices=(0, 1, 2),
        model_digest=response.model_digest,
        temperature_K=response.temperature_K,
        chemical_potential_meV=response.chemical_potential_meV,
        broadening_meV=response.broadening_meV,
        provenance={
            **dict(response.provenance),
            "operator_basis": {
                "kind": "cartesian_spin",
                "implicit_spin_trace_factor": prefactor,
            },
        },
    )


def neutron_spin_contraction(
    response: SusceptibilityResult,
    reciprocal_lattice: ArrayLike,
) -> ComplexArray:
    """Contract a Cartesian spin response with the neutron polarization tensor."""

    if response.operator_labels != ("Sx", "Sy", "Sz"):
        raise ValueError("neutron contraction requires a Cartesian spin response")
    reciprocal = np.asarray(reciprocal_lattice, dtype=float)
    if reciprocal.shape != (3, 3):
        raise ValueError("reciprocal_lattice must have shape (3, 3)")
    q_cartesian = response.Q_reduced @ reciprocal.T
    norm = np.linalg.norm(q_cartesian, axis=1)
    unit = np.zeros_like(q_cartesian)
    nonzero = norm > 1.0e-14
    unit[nonzero] = q_cartesian[nonzero] / norm[nonzero, None]
    projector = np.broadcast_to(np.eye(3), (q_cartesian.shape[0], 3, 3)).copy()
    projector[nonzero] -= np.einsum(
        "pa,pb->pab",
        unit[nonzero],
        unit[nonzero],
    )
    projector[~nonzero] *= 2.0 / 3.0
    return np.einsum(
        "pab,pab->p",
        projector,
        response.values_per_meV_cell,
        optimize=True,
    )


def isotropic_spin_component(response: SusceptibilityResult) -> ComplexArray:
    """Return one Cartesian component, ``Tr(chi_s)/3``."""

    if response.operator_labels != ("Sx", "Sy", "Sz"):
        raise ValueError("isotropic component requires a Cartesian spin response")
    return np.trace(response.values_per_meV_cell, axis1=1, axis2=2) / 3.0
