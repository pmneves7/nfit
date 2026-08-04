"""Bare multiband Lindhard susceptibilities and magnetic projections."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _parallel
from .cache_utils import array_digest, array_payload_nbytes, readonly_array
from .cross_section import KB_MEV_PER_K
from .electronic_backends import (
    ElectronicBackend,
    ElectronicEigensystem,
    evaluate_eigensystem,
)
from .electronic_structure import ElectronicModel, WavevectorSampling

try:
    from . import _electronic_cupy as _CUPY_RESPONSE_BACKEND
except Exception:  # pragma: no cover - CuPy or a GPU is unavailable
    _CUPY_RESPONSE_BACKEND = None

try:
    from . import _electronic_numba as _NUMBA_RESPONSE_BACKEND
except Exception:  # pragma: no cover - Numba is unavailable
    _NUMBA_RESPONSE_BACKEND = None

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - declared dependency is unavailable
    _threadpool_limits = None

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_NUMBA_TRANSITION_MIN_WORK = 25_000_000
_Q_PARALLEL_MIN_WORK = 25_000_000
_INTERPOLATION_GEOMETRY_LOCK = RLock()


@lru_cache(maxsize=16)
def _q_executor(workers: int) -> ThreadPoolExecutor:
    """Return a reusable executor for independent transferred wavevectors."""

    return ThreadPoolExecutor(
        max_workers=int(workers),
        thread_name_prefix="nfit-response-q",
    )


def _group_inverse_indices(inverse: np.ndarray) -> list[np.ndarray]:
    """Return point indices grouped by a dense inverse-index array.

    ``np.unique(..., return_inverse=True)`` numbers groups densely from zero.
    Sorting once avoids the quadratic repeated ``inverse == group`` scans that
    are prohibitive for full multidimensional neutron volumes.
    """

    inverse = np.asarray(inverse, dtype=np.intp)
    if inverse.size == 0:
        return []
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return list(np.split(order, boundaries))


def _q_eigensystem_batch_size(
    *,
    mesh_points: int,
    basis_size: int,
    q_workers: int,
    max_batch_bytes: int,
) -> int:
    """Return a bounded number of shifted-q eigensystems per worker task."""

    # One shifted point retains a Hamiltonian while it is diagonalized, its
    # eigenvectors, and its eigenvalues. Divide the configured response budget
    # across concurrent q workers so their combined live arrays remain bounded.
    per_q_bytes = max(
        1,
        int(mesh_points)
        * (32 * int(basis_size) ** 2 + 8 * int(basis_size)),
    )
    worker_budget = max(1, int(max_batch_bytes) // max(1, int(q_workers)))
    # Small Hermitian matrices are fastest in modest leading-dimension batches
    # on both Apple Accelerate and OpenBLAS. Larger batches increase live memory
    # and were slower in the representative 4-orbital response benchmark.
    return max(1, min(4, worker_budget // per_q_bytes))


def _split_shifted_eigensystems(
    values: ElectronicEigensystem,
    q_values: np.ndarray,
    mesh_points: int,
) -> dict[str, ElectronicEigensystem]:
    """Split one flattened eigensystem batch into digest-keyed q views."""

    result: dict[str, ElectronicEigensystem] = {}
    for local_index, q_value in enumerate(q_values):
        start = local_index * mesh_points
        stop = start + mesh_points
        result[array_digest(q_value, np.float64)] = ElectronicEigensystem(
            eigenvalues=values.eigenvalues[start:stop],
            eigenvectors=(
                None
                if values.eigenvectors is None
                else values.eigenvectors[start:stop]
            ),
            provenance={
                **dict(values.provenance),
                "q_batch_size": int(q_values.shape[0]),
                "q_batch_index": int(local_index),
            },
        )
    return result


@dataclass
class ElectronicResponseCache:
    """Bounded in-memory cache of immutable electronic-response intermediates."""

    max_bytes: int = 512 * 1024**2
    max_entries: int = 64
    hits: int = 0
    misses: int = 0
    _entries: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _device_entries: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _entry_bytes: dict[tuple[Any, ...], int] = field(
        default_factory=dict,
        repr=False,
    )
    _device_entry_bytes: dict[tuple[Any, ...], int] = field(
        default_factory=dict,
        repr=False,
    )
    _host_bytes: int = field(default=0, repr=False)
    _device_bytes: int = field(default=0, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        if int(self.max_bytes) < 0 or int(self.max_entries) < 0:
            raise ValueError("response cache limits must be nonnegative")
        self.max_bytes = int(self.max_bytes)
        self.max_entries = int(self.max_entries)

    def get(self, key: tuple[Any, ...]) -> Any | None:
        """Return and refresh one cached response object."""

        with self._lock:
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
        value: Any,
    ) -> None:
        """Store one immutable object within count and numerical-byte limits."""

        with self._lock:
            self._host_bytes = self._store(
                self._entries,
                self._entry_bytes,
                key,
                value,
                current_bytes=self._host_bytes,
            )

    def get_device(self, key: tuple[Any, ...]) -> Any | None:
        """Return one accelerator-resident intermediate without host transfer."""

        with self._lock:
            result = self._device_entries.get(key)
            if result is None:
                self.misses += 1
                return None
            self._device_entries.move_to_end(key)
            self.hits += 1
            return result

    def put_device(self, key: tuple[Any, ...], value: Any) -> None:
        """Store a bounded accelerator-resident intermediate."""

        with self._lock:
            self._device_bytes = self._store(
                self._device_entries,
                self._device_entry_bytes,
                key,
                value,
                current_bytes=self._device_bytes,
            )

    def _store(
        self,
        entries: OrderedDict,
        entry_bytes: dict[tuple[Any, ...], int],
        key: tuple[Any, ...],
        value: Any,
        *,
        current_bytes: int,
    ) -> int:
        """Insert one entry with constant-time total-byte accounting."""

        previous_bytes = entry_bytes.pop(key, 0)
        if key in entries:
            del entries[key]
        total = current_bytes - previous_bytes
        size = _response_cache_payload_nbytes(value)
        if self.max_entries == 0 or self.max_bytes == 0 or size > self.max_bytes:
            return total
        entries[key] = value
        entry_bytes[key] = size
        total += size
        while entries and (
            len(entries) > self.max_entries or total > self.max_bytes
        ):
            oldest, _discarded = entries.popitem(last=False)
            total -= entry_bytes.pop(oldest)
        return total

    def clear(self) -> None:
        """Remove cached arrays and reset hit/miss counters."""

        with self._lock:
            self._entries.clear()
            self._device_entries.clear()
            self._entry_bytes.clear()
            self._device_entry_bytes.clear()
            self._host_bytes = 0
            self._device_bytes = 0
            self.hits = 0
            self.misses = 0

    @property
    def entries(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def device_entries(self) -> int:
        """Number of accelerator-resident cached intermediates."""

        with self._lock:
            return len(self._device_entries)

    @property
    def host_bytes(self) -> int:
        """Conservative numerical bytes retained in the host cache."""

        with self._lock:
            return self._host_bytes

    @property
    def device_bytes(self) -> int:
        """Conservative numerical bytes retained in the accelerator cache."""

        with self._lock:
            return self._device_bytes


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
        "eigensystem",
        model.content_digest,
        array_digest(coordinates, np.float64),
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
            richer_key = (*key[:3], True, *key[4:])
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
        matrices = readonly_array(self.matrices, np.complex128)
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
        q = readonly_array(self.q_reduced, float)
        energy = readonly_array(self.energy_meV, float)
        values = readonly_array(self.values_per_meV_cell, np.complex128)
        transferred = q if self.Q_reduced is None else readonly_array(self.Q_reduced, float)
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


def _response_cache_payload_nbytes(value: Any) -> int:
    """Size known response objects without traversing scalar provenance."""

    if isinstance(value, ElectronicEigensystem):
        return int(value.eigenvalues.nbytes) + (
            0 if value.eigenvectors is None else int(value.eigenvectors.nbytes)
        )
    if isinstance(value, SusceptibilityResult):
        arrays = (
            value.q_reduced,
            value.Q_reduced,
            value.energy_meV,
            value.values_per_meV_cell,
        )
        distinct = {id(array): array for array in arrays if array is not None}
        return sum(int(array.nbytes) for array in distinct.values())
    return array_payload_nbytes(value)


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
        broadenings = readonly_array(self.broadenings_meV, float)
        q = readonly_array(self.q_reduced, float)
        energy = readonly_array(self.energy_meV, float)
        values = readonly_array(self.values_per_meV_cell, np.complex128)
        mesh_absolute = readonly_array(self.mesh_max_absolute_error, float)
        mesh_relative = readonly_array(self.mesh_max_relative_error, float)
        broad_absolute = readonly_array(self.broadening_max_absolute_change, float)
        broad_relative = readonly_array(self.broadening_max_relative_change, float)
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
    transition_backend: Literal["auto", "numpy", "numba"] = "auto",
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
                transition_backend=transition_backend,
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


def _commensurate_mesh_shift(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    q_reduced: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> NDArray[np.int64] | None:
    """Return the integer mesh shift carrying ``k`` to ``k + q``, or ``None``.

    Split out of :func:`_commensurate_mesh_permutation` so that testing whether
    a wavevector is commensurate costs a few comparisons instead of building
    and discarding a full ``(n_k,)`` permutation.
    """

    q = np.asarray(q_reduced, dtype=float)
    local_q = q[list(model.periodic_axes)]
    if np.allclose(local_q, np.rint(local_q), rtol=0.0, atol=tolerance):
        return np.zeros(len(model.periodic_axes), dtype=np.int64)
    shape = tuple(int(value) for value in mesh.mesh_shape)
    if (
        not shape
        or len(shape) != model.dimension
        or int(np.prod(shape)) != mesh.reduced_coordinates.shape[0]
        or mesh.provenance.get("provider") != "uniform"
    ):
        return None
    shifts_float = local_q * np.asarray(shape, dtype=float)
    shifts = np.rint(shifts_float).astype(np.int64)
    if not np.allclose(shifts_float, shifts, rtol=0.0, atol=tolerance):
        return None
    return shifts


def _commensurate_mesh_permutation(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    q_reduced: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> NDArray[np.int64] | None:
    """Map ``k`` to ``k + q`` on a complete uniform periodic mesh."""

    shifts = _commensurate_mesh_shift(model, mesh, q_reduced, tolerance=tolerance)
    if shifts is None:
        return None
    if not np.any(shifts):
        return np.arange(mesh.reduced_coordinates.shape[0], dtype=np.int64)
    shape = tuple(int(value) for value in mesh.mesh_shape)
    indices = np.indices(shape, dtype=np.int64).reshape(len(shape), -1).T
    shifted = np.mod(indices + shifts[None, :], np.asarray(shape)[None, :])
    return np.ravel_multi_index(shifted.T, shape)


def _commensurate_point_flags(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    q_reduced: np.ndarray,
) -> NDArray[np.bool_]:
    """Return, per response point, whether its wavevector is on the mesh.

    Response points overwhelmingly repeat a handful of wavevectors (one per
    energy of a constant-Q cut), so the test is evaluated once per distinct
    wavevector and broadcast back.
    """

    reduced = np.mod(np.asarray(q_reduced, dtype=float), 1.0)
    unique, inverse = np.unique(reduced, axis=0, return_inverse=True)
    flags = np.asarray(
        [
            _commensurate_mesh_shift(model, mesh, value) is not None
            for value in unique
        ],
        dtype=bool,
    )
    return flags[np.asarray(inverse, dtype=np.intp).ravel()]


def _permuted_eigensystem(
    base: ElectronicEigensystem,
    permutation: NDArray[np.int64],
    q_reduced: np.ndarray,
) -> ElectronicEigensystem:
    values = np.asarray(base.eigenvalues)[permutation]
    values.setflags(write=False)
    vectors = (
        None
        if base.eigenvectors is None
        else np.asarray(base.eigenvectors)[permutation]
    )
    if vectors is not None:
        vectors.setflags(write=False)
    return ElectronicEigensystem(
        eigenvalues=values,
        eigenvectors=vectors,
        provenance={
            **dict(base.provenance),
            "resolved_backend": "commensurate_permutation",
            "commensurate_q_reduced": np.asarray(q_reduced, dtype=float).tolist(),
            "source_backend": dict(base.provenance).get("resolved_backend"),
            "approximation": "none",
        },
    )


def _mesh_symmetry_record(mesh: WavevectorSampling) -> dict[str, Any]:
    """Return the mesh's response-reduction record, or the full-mesh default."""

    return dict(mesh.provenance).get(
        "response_symmetry_reduction",
        {
            "policy": "full",
            "applied": False,
            "reason": "no response reduction metadata",
        },
    )


def _cache_record(
    cache: ElectronicResponseCache | None,
    *,
    hits_before: int,
    misses_before: int,
    include_device: bool = False,
) -> dict[str, Any]:
    """Return the per-call cache statistics recorded in response provenance."""

    record: dict[str, Any] = {
        "enabled": cache is not None,
        "hits": 0 if cache is None else cache.hits - hits_before,
        "misses": 0 if cache is None else cache.misses - misses_before,
        "entries": 0 if cache is None else cache.entries,
    }
    if include_device:
        record["device_entries"] = 0 if cache is None else cache.device_entries
    return record


def _q_evaluation_record(
    *,
    commensurate_permutation_count: int,
    direct_shift_count: int,
) -> dict[str, Any]:
    """Return the exact-evaluation record shared by every non-interpolated path."""

    return {
        "policy": "exact",
        "commensurate_permutation_count": int(commensurate_permutation_count),
        "direct_shift_count": int(direct_shift_count),
        "approximation": "none",
    }


def _base_response_provenance(
    operators: ElectronicOperatorBasis,
    mesh: WavevectorSampling,
    transferred_q: np.ndarray,
    *,
    transition_budget: int,
) -> dict[str, Any]:
    """Return the provenance keys every ``_bare_lindhard_direct`` path shares."""

    return {
        "formula": "generalized_lindhard",
        "sign": "-(f_nk-f_mkq)/(E+e_nk-e_mkq+i eta)",
        "operator_basis": dict(operators.metadata),
        "extended_zone_Q_reduced": transferred_q.tolist(),
        "mesh": mesh.to_dict(),
        "precision": "float64/complex128",
        "symmetry": _mesh_symmetry_record(mesh),
        "approximation": "finite lifetime broadening only",
        "transition_max_batch_bytes": int(transition_budget),
    }


def _validate_transition_backend(requested: str) -> None:
    """Reject an unknown transition-backend name before any work is done."""

    _resolved_transition_backend(requested, work=0, n_operators=1)


def _resolved_transition_backend(
    requested: str,
    *,
    work: int,
    n_operators: int,
) -> str:
    """Choose the exact CPU transition-contraction implementation."""

    choice = str(requested).strip().lower()
    if choice not in {"auto", "numpy", "numba"}:
        raise ValueError(
            "transition_backend must be auto, numpy, or numba"
        )
    if choice == "numpy":
        return "numpy"
    if choice == "numba":
        return "numba" if _NUMBA_RESPONSE_BACKEND is not None else "numpy"
    if (
        _NUMBA_RESPONSE_BACKEND is not None
        and n_operators > 1
        and int(work) >= _NUMBA_TRANSITION_MIN_WORK
    ):
        return "numba"
    return "numpy"


def _ordered_orbital_pair_indices(
    model: ElectronicModel,
    operators: ElectronicOperatorBasis,
) -> tuple[int, ...] | None:
    """Return certified ``|a><b|`` indices for the factorized contraction."""

    if dict(operators.metadata).get("kind") != "ordered_orbital_pairs":
        return None
    try:
        indices = tuple(
            int(value)
            for value in dict(operators.metadata)["basis_indices"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    size = len(indices)
    if (
        size < 1
        or operators.size != size * size
        or len(set(indices)) != size
        or any(value < 0 or value >= model.n_basis for value in indices)
    ):
        return None
    if np.count_nonzero(operators.matrices) != size * size:
        return None
    for local_a, a in enumerate(indices):
        for local_b, b in enumerate(indices):
            if operators.matrices[local_a * size + local_b, a, b] != 1.0:
                return None
    return indices


def _factorized_orbital_pair_contraction(
    weights: np.ndarray,
    energy: np.ndarray,
    broadening: float,
    delta_energy: np.ndarray,
    occupation_difference: np.ndarray,
    equal_energy: np.ndarray,
    static_limit: np.ndarray,
    base_eigenvectors: np.ndarray,
    shifted_eigenvectors: np.ndarray,
    basis_indices: tuple[int, ...],
) -> np.ndarray:
    """Contract an ordered orbital-pair response without dense operators."""

    kernel = -occupation_difference[None, ...] / (
        energy[:, None, None, None]
        + delta_energy[None, ...]
        + 1.0j * broadening
    )
    static_rows = np.abs(energy) <= 1.0e-14
    if np.any(static_rows):
        kernel[static_rows] = np.where(
            equal_energy[None, ...],
            static_limit[None, ...],
            kernel[static_rows],
        )
    selected = list(basis_indices)
    base = base_eigenvectors[:, selected, :]
    shifted = shifted_eigenvectors[:, selected, :]
    base_density = np.einsum(
        "kan,kcn->knac",
        base.conj(),
        base,
        optimize=True,
    )
    shifted_density = np.einsum(
        "kbm,kdm->kmbd",
        shifted,
        shifted.conj(),
        optimize=True,
    )
    intermediate = np.einsum(
        "eknm,kmbd,k->eknbd",
        kernel,
        shifted_density,
        weights,
        optimize=True,
    )
    response = np.einsum(
        "eknbd,knac->eacbd",
        intermediate,
        base_density,
        optimize=True,
    )
    size = len(basis_indices)
    return response.transpose(0, 1, 3, 2, 4).reshape(
        energy.size,
        size * size,
        size * size,
    )


def _bare_lindhard_direct(
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
    transition_backend: Literal["auto", "numpy", "numba"] = "auto",
    cache: ElectronicResponseCache | None = None,
    factorize_ordered_pairs: bool = False,
    precomputed_base: ElectronicEigensystem | None = None,
    allow_q_parallel: bool = True,
    cache_completed_response: bool = True,
    precomputed_shifted: Mapping[str, ElectronicEigensystem] | None = None,
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
    constant_operators = operator_matrices_by_point is None
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
    orbital_pair_indices = (
        _ordered_orbital_pair_indices(model, operators)
        if factorize_ordered_pairs
        else None
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
    transition_backend_request = str(transition_backend).strip().lower()
    _validate_transition_backend(transition_backend_request)
    response_cache_key = (
        "bare_lindhard",
        model.content_digest,
        array_digest(k, np.float64),
        array_digest(weights, np.float64),
        array_digest(transferred_q, np.float64),
        array_digest(energy, np.float64),
        (
            "constant"
            if constant_operators
            else array_digest(point_operators, np.complex128)
        ),
        (
            array_digest(operators.matrices, np.complex128)
            if constant_operators
            else "point-dependent"
        ),
        operators.labels,
        operators.conjugate_indices,
        temperature,
        mu,
        eta,
        None if backend is None else str(backend),
        workers,
        int(max_batch_bytes),
        transition_budget,
        transition_backend_request,
        bool(factorize_ordered_pairs),
        bool(allow_q_parallel),
    )
    cache_hits_before = 0 if cache is None else cache.hits
    cache_misses_before = 0 if cache is None else cache.misses
    if cache is not None and cache_completed_response:
        cached_response = cache.get(response_cache_key)
        if cached_response is not None:
            return cached_response
    requested_backend = (
        "" if backend is None else str(backend).strip().lower()
    )
    if requested_backend == "cupy" and _CUPY_RESPONSE_BACKEND is not None:
        unique_q = np.unique(q, axis=0)
        permutations = [
            _commensurate_mesh_permutation(model, mesh, q_value)
            for q_value in unique_q
        ]
        gpu_values, gpu_record = _CUPY_RESPONSE_BACKEND.evaluate_lindhard(
            model,
            k,
            weights,
            q,
            energy,
            point_operators,
            temperature_K=temperature,
            chemical_potential_meV=mu,
            broadening_meV=eta,
            max_batch_bytes=int(max_batch_bytes),
            transition_max_batch_bytes=transition_budget,
            permutations=permutations,
            cache=cache,
        )
        response = SusceptibilityResult(
            q_reduced=q,
            Q_reduced=transferred_q,
            energy_meV=energy,
            values_per_meV_cell=gpu_values,
            operator_labels=operators.labels,
            conjugate_indices=operators.conjugate_indices,
            model_digest=model.content_digest,
            temperature_K=temperature,
            chemical_potential_meV=mu,
            broadening_meV=eta,
            provenance={
                **_base_response_provenance(
                    operators,
                    mesh,
                    transferred_q,
                    transition_budget=transition_budget,
                ),
                "base_execution": dict(gpu_record["base_execution"]),
                "shifted_execution": list(gpu_record["shifted_execution"]),
                "transition_batch_size": int(
                    gpu_record["transition_batch_size"]
                ),
                "energy_batch_size": int(gpu_record["energy_batch_size"]),
                "q_evaluation": _q_evaluation_record(
                    commensurate_permutation_count=gpu_record[
                        "commensurate_permutation_count"
                    ],
                    direct_shift_count=gpu_record["direct_shift_count"],
                ),
                "cache": _cache_record(
                    cache,
                    hits_before=cache_hits_before,
                    misses_before=cache_misses_before,
                    include_device=True,
                ),
                "response_execution": "cupy_end_to_end",
                "transition_backend": "cupy",
                "host_transfer": "completed susceptibility only",
            },
        )
        if cache is not None and cache_completed_response:
            cache.put(response_cache_key, response)
        return response
    base = (
        precomputed_base
        if precomputed_base is not None
        else _cached_eigensystem(
            model,
            k,
            eigenvectors=True,
            backend=backend,
            workers=workers,
            max_batch_bytes=max_batch_bytes,
            cache=cache,
        )
    )
    if base.eigenvectors is None:  # pragma: no cover - defensive
        raise RuntimeError("Lindhard response requires band eigenvectors")
    occupation_k = _fermi_function(base.eigenvalues, mu, temperature)
    result = np.empty(
        (q.shape[0], operators.size, operators.size),
        dtype=np.complex128,
    )
    operator_keys = (
        None
        if constant_operators
        else np.asarray(
            [array_digest(value, np.complex128) for value in point_operators],
            dtype=object,
        )
    )
    unique_q, inverse = np.unique(q, axis=0, return_inverse=True)
    total_workers = (
        _parallel.num_threads()
        if workers is None or int(workers) == 0
        else max(1, int(workers))
    )
    q_parallel_work = (
        int(q.shape[0])
        * int(k.shape[0])
        * model.n_basis
        * model.n_basis
        * operators.size
        * operators.size
    )
    q_worker_count = min(total_workers, unique_q.shape[0])
    if (
        allow_q_parallel
        and q_worker_count > 1
        and q_parallel_work >= _Q_PARALLEL_MIN_WORK
    ):
        point_groups = _group_inverse_indices(inverse)
        inner_workers = max(1, total_workers // q_worker_count)
        q_batch_size = _q_eigensystem_batch_size(
            mesh_points=k.shape[0],
            basis_size=model.n_basis,
            q_workers=q_worker_count,
            max_batch_bytes=max_batch_bytes,
        )
        q_batch_size = min(
            q_batch_size,
            max(1, int(np.ceil(len(point_groups) / q_worker_count))),
        )
        q_group_batches = [
            point_groups[start : start + q_batch_size]
            for start in range(0, len(point_groups), q_batch_size)
        ]

        def evaluate_q_batch(groups: list[np.ndarray]) -> SusceptibilityResult:
            indices = np.concatenate(groups)
            batch_q = np.unique(q[indices], axis=0)
            direct_q = np.asarray(
                [
                    value
                    for value in batch_q
                    if _commensurate_mesh_permutation(model, mesh, value) is None
                ],
                dtype=float,
            ).reshape(-1, 3)
            shifted_by_q: dict[str, ElectronicEigensystem] = {}
            if direct_q.size:
                coordinates = (
                    k[None, :, :] + direct_q[:, None, :]
                ).reshape(-1, 3)
                shifted_batch = evaluate_eigensystem(
                    model,
                    coordinates,
                    eigenvectors=True,
                    backend=backend,
                    workers=inner_workers,
                    max_batch_bytes=max_batch_bytes,
                )
                shifted_by_q = _split_shifted_eigensystems(
                    shifted_batch,
                    direct_q,
                    k.shape[0],
                )
            return _bare_lindhard_direct(
                model,
                transferred_q[indices],
                energy[indices],
                mesh,
                operators,
                temperature_K=temperature,
                chemical_potential_meV=mu,
                broadening_meV=eta,
                operator_matrices_by_point=(
                    None
                    if constant_operators
                    else point_operators[indices]
                ),
                backend=backend,
                workers=inner_workers,
                max_batch_bytes=max_batch_bytes,
                transition_max_batch_bytes=transition_budget,
                transition_backend=transition_backend_request,
                cache=cache,
                factorize_ordered_pairs=factorize_ordered_pairs,
                precomputed_base=base,
                allow_q_parallel=False,
                cache_completed_response=False,
                precomputed_shifted=shifted_by_q,
            )

        limiter = (
            _threadpool_limits(
                limits=inner_workers,
                user_api="blas",
            )
            if _threadpool_limits is not None
            else contextlib.nullcontext()
        )
        with limiter:
            q_responses = list(
                _q_executor(q_worker_count).map(
                    evaluate_q_batch,
                    q_group_batches,
                )
            )
        combined_values = np.empty(
            (q.shape[0], operators.size, operators.size),
            dtype=np.complex128,
        )
        for groups, q_response in zip(
            q_group_batches,
            q_responses,
            strict=True,
        ):
            indices = np.concatenate(groups)
            combined_values[indices] = q_response.values_per_meV_cell
        reference_provenance = dict(q_responses[0].provenance)
        transition_backends = {
            str(item.provenance.get("transition_backend", "unknown"))
            for item in q_responses
        }
        response = SusceptibilityResult(
            q_reduced=q,
            Q_reduced=transferred_q,
            energy_meV=energy,
            values_per_meV_cell=combined_values,
            operator_labels=operators.labels,
            conjugate_indices=operators.conjugate_indices,
            model_digest=model.content_digest,
            temperature_K=temperature,
            chemical_potential_meV=mu,
            broadening_meV=eta,
            provenance={
                **reference_provenance,
                "extended_zone_Q_reduced": transferred_q.tolist(),
                "shifted_execution": [
                    record
                    for item in q_responses
                    for record in item.provenance.get(
                        "shifted_execution",
                        [],
                    )
                ],
                "transition_batch_size": min(
                    int(item.provenance["transition_batch_size"])
                    for item in q_responses
                ),
                "energy_batch_size": min(
                    int(item.provenance["energy_batch_size"])
                    for item in q_responses
                ),
                "transition_backend": (
                    next(iter(transition_backends))
                    if len(transition_backends) == 1
                    else "mixed"
                ),
                "q_evaluation": _q_evaluation_record(
                    commensurate_permutation_count=sum(
                        int(
                            item.provenance["q_evaluation"][
                                "commensurate_permutation_count"
                            ]
                        )
                        for item in q_responses
                    ),
                    direct_shift_count=sum(
                        int(item.provenance["q_evaluation"]["direct_shift_count"])
                        for item in q_responses
                    ),
                ),
                "q_parallel_execution": {
                    "applied": True,
                    "q_workers": q_worker_count,
                    "inner_workers": inner_workers,
                    "unique_q": int(unique_q.shape[0]),
                    "q_batch_size": int(q_batch_size),
                    "q_batches": int(len(q_group_batches)),
                    "work_estimate": q_parallel_work,
                },
                "cache": _cache_record(
                    cache,
                    hits_before=cache_hits_before,
                    misses_before=cache_misses_before,
                ),
            },
        )
        if cache is not None and cache_completed_response:
            cache.put(response_cache_key, response)
        return response
    execution_records = []
    transition_batch_sizes = []
    energy_batch_sizes = []
    commensurate_q_count = 0
    resolved_transition_backends: set[str] = set()
    for q_index, q_value in enumerate(unique_q):
        point_indices = np.flatnonzero(inverse == q_index)
        operator_groups = (
            [point_indices]
            if operator_keys is None
            else [
                point_indices[operator_keys[point_indices] == key]
                for key in np.unique(operator_keys[point_indices])
            ]
        )
        permutation = _commensurate_mesh_permutation(model, mesh, q_value)
        if permutation is None:
            shifted = (
                None
                if precomputed_shifted is None
                else precomputed_shifted.get(
                    array_digest(q_value, np.float64)
                )
            )
            if shifted is None:
                shifted = _cached_eigensystem(
                    model,
                    k + q_value[None, :],
                    eigenvectors=True,
                    backend=backend,
                    workers=workers,
                    max_batch_bytes=max_batch_bytes,
                    cache=cache,
                )
        else:
            shifted = _permuted_eigensystem(base, permutation, q_value)
            commensurate_q_count += 1
        if shifted.eigenvectors is None:  # pragma: no cover - defensive
            raise RuntimeError("Lindhard response requires band eigenvectors")
        execution_records.append(dict(shifted.provenance))
        occupation_q = _fermi_function(shifted.eigenvalues, mu, temperature)
        bands = model.n_basis
        factorized_pairs = (
            orbital_pair_indices is not None
            and transition_backend_request == "auto"
        )
        # Bound unconditionally: the per-energy byte estimate below reads it
        # inside a conditional expression, which is easy to break when edited.
        pair_size = 0 if orbital_pair_indices is None else len(orbital_pair_indices)
        if factorized_pairs:
            per_k_bytes = max(
                1,
                32 * bands * pair_size * pair_size
                + 64 * bands * bands
                + 32 * bands,
            )
        else:
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
            batch_k = stop - start
            fixed_bytes = per_k_bytes * batch_k
            per_energy_bytes = max(
                1,
                16
                * batch_k
                * (
                    bands * bands
                    + (
                        bands * pair_size * pair_size
                        if factorized_pairs
                        else 0
                    )
                ),
            )
            available = max(per_energy_bytes, transition_budget - fixed_bytes)
            for operator_group in operator_groups:
                matrix_elements = None
                if not factorized_pairs:
                    reference_operators = point_operators[operator_group[0]]
                    matrix_elements = np.einsum(
                        "kan,Aab,kbm->kAnm",
                        base.eigenvectors[start:stop].conj(),
                        reference_operators,
                        shifted.eigenvectors[start:stop],
                        optimize=True,
                    )
                energy_batch = max(
                    1,
                    min(operator_group.size, available // per_energy_bytes),
                )
                energy_batch_sizes.append(energy_batch)
                for energy_start in range(0, operator_group.size, energy_batch):
                    batch_indices = operator_group[
                        energy_start : energy_start + energy_batch
                    ]
                    transferred = energy[batch_indices]
                    if factorized_pairs:
                        resolved_transition_backends.add(
                            "numpy_orbital_pair_factorized"
                        )
                        result[batch_indices] += (
                            _factorized_orbital_pair_contraction(
                                weights[start:stop],
                                transferred,
                                eta,
                                delta_energy,
                                occupation_difference,
                                equal,
                                static_limit,
                                base.eigenvectors[start:stop],
                                shifted.eigenvectors[start:stop],
                                orbital_pair_indices,
                            )
                        )
                        continue
                    transition_work = (
                        int(batch_indices.size)
                        * operators.size
                        * operators.size
                        * batch_k
                        * bands
                        * bands
                    )
                    resolved_transition = _resolved_transition_backend(
                        transition_backend_request,
                        work=transition_work,
                        n_operators=operators.size,
                    )
                    resolved_transition_backends.add(resolved_transition)
                    if resolved_transition == "numba":
                        worker_count = (
                            _parallel.num_threads()
                            if workers is None or int(workers) == 0
                            else max(1, int(workers))
                        )
                        _NUMBA_RESPONSE_BACKEND.initialize_num_threads(
                            worker_count
                        )
                        result[batch_indices] += (
                            _NUMBA_RESPONSE_BACKEND.contract_lindhard(
                                weights[start:stop],
                                transferred,
                                eta,
                                delta_energy,
                                occupation_difference,
                                equal,
                                static_limit,
                                matrix_elements,
                                parallel=worker_count > 1,
                            )
                        )
                    else:
                        kernel = -occupation_difference[None, ...] / (
                            transferred[:, None, None, None]
                            + delta_energy[None, ...]
                            + 1.0j * eta
                        )
                        static_rows = np.abs(transferred) <= 1.0e-14
                        if np.any(static_rows):
                            kernel[static_rows] = np.where(
                                equal[None, ...],
                                static_limit[None, ...],
                                kernel[static_rows],
                            )
                        result[batch_indices] += np.einsum(
                            "k,eknm,kAnm,kBnm->eAB",
                            weights[start:stop],
                            kernel,
                            matrix_elements,
                            matrix_elements.conj(),
                            optimize=True,
                        )

    response = SusceptibilityResult(
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
            **_base_response_provenance(
                operators,
                mesh,
                transferred_q,
                transition_budget=transition_budget,
            ),
            "base_execution": dict(base.provenance),
            "shifted_execution": execution_records,
            "transition_batch_size": (
                min(transition_batch_sizes) if transition_batch_sizes else 0
            ),
            "energy_batch_size": (
                min(energy_batch_sizes) if energy_batch_sizes else 0
            ),
            "transition_backend": (
                next(iter(resolved_transition_backends))
                if len(resolved_transition_backends) == 1
                else "mixed"
            ),
            "q_evaluation": _q_evaluation_record(
                commensurate_permutation_count=commensurate_q_count,
                direct_shift_count=len(unique_q) - commensurate_q_count,
            ),
            "q_parallel_execution": {
                "applied": False,
                "q_workers": 1,
                "inner_workers": total_workers,
                "unique_q": int(unique_q.shape[0]),
                "work_estimate": q_parallel_work,
            },
            "cache": _cache_record(
                cache,
                hits_before=cache_hits_before,
                misses_before=cache_misses_before,
            ),
        },
    )
    if cache is not None and cache_completed_response:
        cache.put(response_cache_key, response)
    return response


def _q_mesh_candidates(
    mesh_shape: Sequence[int],
    initial_shape: Sequence[int] | None,
) -> tuple[tuple[int, ...], ...]:
    full = tuple(int(value) for value in mesh_shape)
    divisors = [
        tuple(value for value in range(1, size + 1) if size % value == 0)
        for size in full
    ]
    if initial_shape is None:
        current = tuple(
            max(value for value in axis_divisors if value <= min(8, size))
            for size, axis_divisors in zip(full, divisors, strict=True)
        )
    else:
        current = tuple(int(value) for value in initial_shape)
        if len(current) != len(full):
            raise ValueError("q_interpolation_mesh must match the model dimension")
        if any(
            value < 1 or size % value
            for value, size in zip(current, full, strict=True)
        ):
            raise ValueError(
                "q_interpolation_mesh sizes must be positive divisors of the "
                "integration mesh"
            )
    candidates = [current]
    while current != full:
        current = tuple(
            next(
                (value for value in axis_divisors if value > old),
                size,
            )
            for old, size, axis_divisors in zip(
                current, full, divisors, strict=True
            )
        )
        candidates.append(current)
    return tuple(candidates)


def _periodic_linear_stencils(
    model: ElectronicModel,
    q_reduced: np.ndarray,
    mesh_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node coordinates, parent indices, and periodic linear weights."""

    shape = np.asarray(tuple(int(value) for value in mesh_shape), dtype=np.int64)
    nodes: list[np.ndarray] = []
    parents: list[int] = []
    weights: list[float] = []
    for point_index, q_value in enumerate(np.mod(q_reduced, 1.0)):
        choices = []
        for local_axis, axis in enumerate(model.periodic_axes):
            coordinate = q_value[axis] * shape[local_axis]
            lower = int(np.floor(coordinate + 1.0e-13))
            fraction = float(coordinate - np.floor(coordinate))
            if fraction <= 1.0e-12 or 1.0 - fraction <= 1.0e-12:
                choices.append(((lower % shape[local_axis], 1.0),))
            else:
                choices.append(
                    (
                        (lower % shape[local_axis], 1.0 - fraction),
                        ((lower + 1) % shape[local_axis], fraction),
                    )
                )
        for selection in product(*choices):
            node = np.array(q_value, copy=True)
            weight = 1.0
            for local_axis, axis in enumerate(model.periodic_axes):
                node[axis] = selection[local_axis][0] / shape[local_axis]
                weight *= selection[local_axis][1]
            nodes.append(node)
            parents.append(point_index)
            weights.append(weight)
    return (
        np.asarray(nodes, dtype=float),
        np.asarray(parents, dtype=np.int64),
        np.asarray(weights, dtype=float),
    )


def _periodic_linear_stencil_indices(
    model: ElectronicModel,
    q_reduced: np.ndarray,
    mesh_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return vectorized flat node indices and weights for every q point."""

    shape = np.asarray(tuple(int(value) for value in mesh_shape), dtype=np.int64)
    coordinates = (
        np.mod(q_reduced[:, model.periodic_axes], 1.0) * shape[None, :]
    )
    floors = np.floor(coordinates)
    lower = floors.astype(np.int64) % shape[None, :]
    fraction = coordinates - floors
    fraction = np.where(
        (fraction <= 1.0e-12) | (1.0 - fraction <= 1.0e-12),
        0.0,
        fraction,
    )
    corners = np.asarray(
        list(product((0, 1), repeat=len(model.periodic_axes))),
        dtype=np.int64,
    )
    node_indices = (
        lower[:, None, :] + corners[None, :, :]
    ) % shape[None, None, :]
    axis_weights = np.where(
        corners[None, :, :] == 0,
        1.0 - fraction[:, None, :],
        fraction[:, None, :],
    )
    weights = np.prod(axis_weights, axis=2)
    flat = np.ravel_multi_index(
        tuple(node_indices[:, :, axis].ravel() for axis in range(shape.size)),
        tuple(shape),
    ).reshape(q_reduced.shape[0], corners.shape[0])
    return flat, weights


def _interpolated_lindhard_response(
    model: ElectronicModel,
    transferred_q: np.ndarray,
    energy: np.ndarray,
    mesh: WavevectorSampling,
    operators: ElectronicOperatorBasis,
    point_operators: np.ndarray | None,
    *,
    q_mesh_shape: Sequence[int],
    direct_kwargs: Mapping[str, Any],
) -> SusceptibilityResult:
    values = np.zeros(
        (transferred_q.shape[0], operators.size, operators.size),
        dtype=np.complex128,
    )
    max_stencils = 2 ** len(model.periodic_axes)
    point_operator_bytes = (
        0
        if point_operators is None
        else 16 * operators.size * model.n_basis * model.n_basis
    )
    per_point_bytes = max(
        1,
        max_stencils
        * (
            24
            + 8
            + 8
            + 16 * operators.size * operators.size
            + point_operator_bytes
        ),
    )
    memory_budget = max(
        1,
        int(
            direct_kwargs.get("transition_max_batch_bytes")
            or direct_kwargs.get("max_batch_bytes", 256 * 1024**2)
        ),
    )
    point_batch = max(
        1,
        min(
            transferred_q.shape[0],
            65_536,
            memory_budget // per_point_bytes,
        ),
    )
    evaluated_stencils = 0
    unique_nodes: set[tuple[float, ...]] = set()
    reference_response: SusceptibilityResult | None = None
    for start in range(0, transferred_q.shape[0], point_batch):
        stop = min(start + point_batch, transferred_q.shape[0])
        nodes, parents, interpolation_weights = _periodic_linear_stencils(
            model,
            transferred_q[start:stop],
            q_mesh_shape,
        )
        chunk_operators = (
            None
            if point_operators is None
            else point_operators[start:stop][parents]
        )
        node_energy = energy[start:stop][parents]
        response_inverse: np.ndarray | None = None
        evaluation_nodes = nodes
        evaluation_energy = node_energy
        if chunk_operators is None:
            # Interpolation commonly maps many source points onto the same
            # commensurate node and energy. Evaluate each pair once, then
            # scatter it back to the stencil rows. This is essential for a
            # million-point volume, where the raw stencil can contain several
            # million duplicate response requests.
            pairs = np.column_stack((nodes, node_energy))
            unique_pairs, response_inverse = np.unique(
                pairs,
                axis=0,
                return_inverse=True,
            )
            evaluation_nodes = unique_pairs[:, :3]
            evaluation_energy = unique_pairs[:, 3]
        node_response = _bare_lindhard_direct(
            model,
            evaluation_nodes,
            evaluation_energy,
            mesh,
            operators,
            operator_matrices_by_point=chunk_operators,
            cache_completed_response=False,
            **direct_kwargs,
        )
        np.add.at(
            values[start:stop],
            parents,
            interpolation_weights[:, None, None]
            * (
                node_response.values_per_meV_cell
                if response_inverse is None
                else node_response.values_per_meV_cell[response_inverse]
            ),
        )
        evaluated_stencils += int(nodes.shape[0])
        unique_nodes.update(map(tuple, np.unique(nodes, axis=0)))
        reference_response = node_response
    if reference_response is None:  # pragma: no cover - validated nonempty input
        raise RuntimeError("interpolated response contains no points")
    return SusceptibilityResult(
        q_reduced=np.mod(transferred_q, 1.0),
        Q_reduced=transferred_q,
        energy_meV=energy,
        values_per_meV_cell=values,
        operator_labels=operators.labels,
        conjugate_indices=operators.conjugate_indices,
        model_digest=model.content_digest,
        temperature_K=reference_response.temperature_K,
        chemical_potential_meV=reference_response.chemical_potential_meV,
        broadening_meV=reference_response.broadening_meV,
        provenance={
            **dict(reference_response.provenance),
            "approximation": "finite lifetime broadening and q interpolation",
            "q_interpolation": {
                "method": "periodic_linear",
                "mesh_shape": list(q_mesh_shape),
                "requested_points": int(transferred_q.shape[0]),
                "evaluated_stencil_points": evaluated_stencils,
                "unique_stencil_q": len(unique_nodes),
                "point_batch_size": int(point_batch),
                "point_batches": int(
                    np.ceil(transferred_q.shape[0] / point_batch)
                ),
            },
        },
    )


def _interpolation_certificate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a digest-bearing immutable-record payload for q interpolation."""

    result = dict(payload)
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    result["certificate_digest"] = hashlib.sha256(encoded).hexdigest()
    return result


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
    transition_backend: Literal["auto", "numpy", "numba"] = "auto",
    cache: ElectronicResponseCache | None = None,
    q_evaluation: Literal[
        "auto", "direct", "commensurate", "interpolated"
    ] = "auto",
    q_interpolation_rtol: float | None = None,
    q_interpolation_atol: float | None = None,
    q_interpolation_mesh: Sequence[int] | None = None,
    q_validation_points: int = 8,
) -> SusceptibilityResult:
    """Evaluate a direct, commensurate, or validated interpolated response.

    ``auto`` is exact unless at least one interpolation tolerance is supplied.
    Commensurate points always use the exact periodic mesh permutation.
    """

    policy = str(q_evaluation).strip().lower()
    if policy not in {"auto", "direct", "commensurate", "interpolated"}:
        raise ValueError(
            "q_evaluation must be auto, direct, commensurate, or interpolated"
        )
    rtol = 0.0 if q_interpolation_rtol is None else float(q_interpolation_rtol)
    atol = 0.0 if q_interpolation_atol is None else float(q_interpolation_atol)
    if (
        not np.isfinite(rtol)
        or not np.isfinite(atol)
        or rtol < 0.0
        or atol < 0.0
    ):
        raise ValueError("q interpolation tolerances must be finite and nonnegative")
    if int(q_validation_points) < 1:
        raise ValueError("q_validation_points must be positive")
    Q, energy = _point_inputs(q_reduced, energy_meV)
    constant_operators = operator_matrices_by_point is None
    point_operators = (
        np.broadcast_to(
            np.asarray(operators.matrices)[None, ...],
            (Q.shape[0], *operators.matrices.shape),
        )
        if operator_matrices_by_point is None
        else np.asarray(operator_matrices_by_point, dtype=np.complex128)
    )
    if point_operators.shape != (
        Q.shape[0],
        operators.size,
        model.n_basis,
        model.n_basis,
    ):
        raise ValueError(
            "point-dependent operators must have shape "
            "(n_points, n_operators, n_basis, n_basis)"
        )
    direct_kwargs = {
        "temperature_K": temperature_K,
        "chemical_potential_meV": chemical_potential_meV,
        "broadening_meV": broadening_meV,
        "backend": backend,
        "workers": workers,
        "max_batch_bytes": max_batch_bytes,
        "transition_max_batch_bytes": transition_max_batch_bytes,
        "transition_backend": transition_backend,
        "cache": cache,
        "factorize_ordered_pairs": operator_matrices_by_point is None,
    }

    def selected_operators(indices: np.ndarray) -> np.ndarray | None:
        return None if constant_operators else point_operators[indices]
    commensurate = _commensurate_point_flags(model, mesh, Q)
    if policy == "commensurate" and not np.all(commensurate):
        first = int(np.flatnonzero(~commensurate)[0])
        raise ValueError(
            f"response point {first} is not commensurate with the integration mesh"
        )
    interpolation_enabled = (rtol > 0.0 or atol > 0.0) and policy in {
        "auto",
        "interpolated",
    }
    if policy == "interpolated" and not interpolation_enabled:
        raise ValueError(
            "interpolated q evaluation requires a positive relative or "
            "absolute interpolation tolerance"
        )
    if policy in {"direct", "commensurate"} or not interpolation_enabled:
        return _bare_lindhard_direct(
            model,
            Q,
            energy,
            mesh,
            operators,
            operator_matrices_by_point=(
                None if constant_operators else point_operators
            ),
            **direct_kwargs,
        )
    if np.all(commensurate):
        return _bare_lindhard_direct(
            model,
            Q,
            energy,
            mesh,
            operators,
            operator_matrices_by_point=(
                None if constant_operators else point_operators
            ),
            **direct_kwargs,
        )
    if (
        not mesh.mesh_shape
        or int(np.prod(mesh.mesh_shape)) != mesh.reduced_coordinates.shape[0]
        or dict(mesh.provenance).get("provider") != "uniform"
    ):
        if policy == "interpolated":
            raise ValueError(
                "q interpolation requires a complete uniform integration mesh"
            )
        return _bare_lindhard_direct(
            model,
            Q,
            energy,
            mesh,
            operators,
            operator_matrices_by_point=(
                None if constant_operators else point_operators
            ),
            **direct_kwargs,
        )

    exact_indices = np.flatnonzero(commensurate)
    approximate_indices = np.flatnonzero(~commensurate)
    validation_count = min(int(q_validation_points), approximate_indices.size)
    validation_local = np.unique(
        np.linspace(
            0,
            approximate_indices.size - 1,
            validation_count,
            dtype=int,
        )
    )
    validation_indices = approximate_indices[validation_local]
    validation = _bare_lindhard_direct(
        model,
        Q[validation_indices],
        energy[validation_indices],
        mesh,
        operators,
        operator_matrices_by_point=selected_operators(validation_indices),
        **direct_kwargs,
    )
    accepted: SusceptibilityResult | None = None
    validation_absolute = np.inf
    validation_relative = np.inf
    selected_shape: tuple[int, ...] | None = None
    attempts: list[dict[str, Any]] = []
    for candidate in _q_mesh_candidates(mesh.mesh_shape, q_interpolation_mesh):
        validation_trial = _interpolated_lindhard_response(
            model,
            Q[validation_indices],
            energy[validation_indices],
            mesh,
            operators,
            selected_operators(validation_indices),
            q_mesh_shape=candidate,
            direct_kwargs=direct_kwargs,
        )
        trial = validation_trial.values_per_meV_cell
        reference = validation.values_per_meV_cell
        difference = np.abs(trial - reference)
        reference_scale = np.maximum(
            np.max(np.abs(reference), axis=(1, 2), keepdims=True),
            max(atol, np.finfo(float).tiny),
        )
        validation_absolute = float(np.max(difference))
        validation_relative = float(np.max(difference / reference_scale))
        passed = bool(
            np.all(difference <= atol + rtol * reference_scale)
        )
        attempts.append(
            {
                "mesh_shape": list(candidate),
                "maximum_absolute_error": validation_absolute,
                "maximum_relative_error": validation_relative,
                "passed": passed,
            }
        )
        if passed:
            selected_shape = candidate
            break
    if selected_shape is not None:
        accepted = _interpolated_lindhard_response(
            model,
            Q[approximate_indices],
            energy[approximate_indices],
            mesh,
            operators,
            selected_operators(approximate_indices),
            q_mesh_shape=selected_shape,
            direct_kwargs=direct_kwargs,
        )
    else:
        if policy == "interpolated":
            raise ValueError(
                "q interpolation did not satisfy the requested tolerance; "
                f"maximum absolute error {validation_absolute:g}, maximum "
                f"relative error {validation_relative:g}"
            )
        accepted = _bare_lindhard_direct(
            model,
            Q[approximate_indices],
            energy[approximate_indices],
            mesh,
            operators,
            operator_matrices_by_point=selected_operators(
                approximate_indices
            ),
            **direct_kwargs,
        )

    exact = (
        None
        if exact_indices.size == 0
        else _bare_lindhard_direct(
            model,
            Q[exact_indices],
            energy[exact_indices],
            mesh,
            operators,
            operator_matrices_by_point=selected_operators(exact_indices),
            **direct_kwargs,
        )
    )
    values = np.empty(
        (Q.shape[0], operators.size, operators.size),
        dtype=np.complex128,
    )
    values[approximate_indices] = accepted.values_per_meV_cell
    if exact is not None:
        values[exact_indices] = exact.values_per_meV_cell
    used_interpolation = selected_shape is not None
    certificate = _interpolation_certificate(
        {
            "schema_version": 1,
            "status": "certified" if used_interpolation else "exact_fallback",
            "certified_quantity": "bare_lindhard_operator_tensor",
            "model_digest": model.content_digest,
            "integration_mesh_shape": list(mesh.mesh_shape or ()),
            "selected_interpolation_mesh_shape": (
                None if selected_shape is None else list(selected_shape)
            ),
            "relative_tolerance": rtol,
            "absolute_tolerance": atol,
            "validation_method": "deterministic direct-response comparison",
            "relative_error_normalization": (
                "maximum direct-response magnitude per validation point"
            ),
            "validation_point_count": int(validation_indices.size),
            "validation_q_reduced": Q[validation_indices].tolist(),
            "validation_energy_meV": energy[validation_indices].tolist(),
            "maximum_absolute_error": validation_absolute,
            "maximum_relative_error": validation_relative,
            "attempts": attempts,
        }
    )
    return SusceptibilityResult(
        q_reduced=np.mod(Q, 1.0),
        Q_reduced=Q,
        energy_meV=energy,
        values_per_meV_cell=values,
        operator_labels=operators.labels,
        conjugate_indices=operators.conjugate_indices,
        model_digest=model.content_digest,
        temperature_K=accepted.temperature_K,
        chemical_potential_meV=accepted.chemical_potential_meV,
        broadening_meV=accepted.broadening_meV,
        provenance={
            **dict(accepted.provenance),
            "approximation": (
                "finite lifetime broadening and validated q interpolation"
                if used_interpolation
                else "finite lifetime broadening only"
            ),
            "q_evaluation": {
                "requested_policy": policy,
                "resolved_policy": (
                    "validated_interpolation" if used_interpolation else "exact_fallback"
                ),
                "exact_commensurate_points": int(exact_indices.size),
                "interpolated_points": (
                    int(approximate_indices.size) if used_interpolation else 0
                ),
                "direct_off_mesh_points": (
                    0 if used_interpolation else int(approximate_indices.size)
                ),
                "interpolation_method": (
                    "periodic_linear" if used_interpolation else None
                ),
                "selected_mesh_shape": (
                    None if selected_shape is None else list(selected_shape)
                ),
                "attempted_mesh_shapes": [
                    attempt["mesh_shape"] for attempt in attempts
                ],
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
                "validation_points": validation_indices.tolist(),
                "validation_max_absolute_error": validation_absolute,
                "validation_max_relative_error": validation_relative,
            },
            "q_interpolation_certificate": certificate,
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


def _implicit_spin_density_basis(model: ElectronicModel) -> ElectronicOperatorBasis:
    matrices = np.zeros(
        (model.n_basis, model.n_basis, model.n_basis),
        dtype=np.complex128,
    )
    indices = np.arange(model.n_basis)
    matrices[indices, indices, indices] = 1.0
    return ElectronicOperatorBasis(
        tuple(f"density_{index}" for index in indices),
        matrices,
        metadata={"kind": "orbital_density_interpolation"},
    )


def _without_q_interpolation_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(kwargs)
    for name in (
        "q_evaluation",
        "q_interpolation_rtol",
        "q_interpolation_atol",
        "q_interpolation_mesh",
        "q_validation_points",
    ):
        result.pop(name, None)
    return result


def _direct_implicit_spin_scalar(
    model: ElectronicModel,
    Q: np.ndarray,
    energy: np.ndarray,
    mesh: WavevectorSampling,
    kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, SusceptibilityResult, float]:
    basis, matrices, prefactor = spin_operator_matrices(model, Q)
    response = bare_lindhard_susceptibility(
        model,
        Q,
        energy,
        mesh,
        basis,
        operator_matrices_by_point=matrices,
        q_evaluation="direct",
        **_without_q_interpolation_kwargs(kwargs),
    )
    return prefactor * response.values_per_meV_cell[:, 0, 0], response, prefactor


def _interpolated_implicit_spin_scalar(
    model: ElectronicModel,
    Q: np.ndarray,
    energy: np.ndarray,
    mesh: WavevectorSampling,
    *,
    q_mesh_shape: Sequence[int],
    kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate an orbital-density tensor and contract phases by chunk."""

    density_basis = _implicit_spin_density_basis(model)
    degeneracy = int(model.provenance.get("implicit_spin_degeneracy", 2))
    if degeneracy != 2:
        raise ValueError("implicit spin interpolation requires spin degeneracy 2")
    prefactor = degeneracy / 4.0
    max_stencils = 2 ** len(model.periodic_axes)
    operators = density_basis.size
    per_point_bytes = max(
        1,
        max_stencils
        * (40 + 16 * operators * operators + 16 * operators),
    )
    memory_budget = int(
        kwargs.get("transition_max_batch_bytes")
        or kwargs.get("max_batch_bytes", 256 * 1024**2)
    )
    point_batch = max(
        1,
        min(Q.shape[0], 65_536, max(1, memory_budget) // per_point_bytes),
    )
    mesh_shape = tuple(int(value) for value in q_mesh_shape)
    unique_energy, energy_inverse = np.unique(energy, return_inverse=True)
    node_count = int(np.prod(mesh_shape))
    lookup_bytes = (
        node_count
        * unique_energy.size
        * operators
        * operators
        * 16
    )
    use_global_lookup = Q.shape[0] > point_batch and lookup_bytes <= memory_budget
    cached_geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    geometry_cache_hit = False
    response_cache = kwargs.get("cache")
    geometry_bytes = Q.shape[0] * (
        max_stencils * (np.dtype(np.int32).itemsize + np.dtype(float).itemsize)
        + operators * np.dtype(np.complex128).itemsize
    )
    if (
        use_global_lookup
        and isinstance(response_cache, ElectronicResponseCache)
        and geometry_bytes <= response_cache.max_bytes
    ):
        centers = np.asarray(model.orbital_centers, dtype=float)
        geometry_key = (
            "periodic_linear_orbital_density_geometry",
            array_digest(Q, np.float64),
            mesh_shape,
            tuple(model.periodic_axes),
            array_digest(centers, np.float64),
        )
        with _INTERPOLATION_GEOMETRY_LOCK:
            cached = response_cache.get(geometry_key)
            if cached is not None:
                cached_geometry = cached
                geometry_cache_hit = True
            else:
                flat_geometry = np.empty(
                    (Q.shape[0], max_stencils),
                    dtype=np.int32,
                )
                weight_geometry = np.empty(
                    (Q.shape[0], max_stencils),
                    dtype=float,
                )
                phase_geometry = np.empty(
                    (Q.shape[0], operators),
                    dtype=np.complex128,
                )
                for start in range(0, Q.shape[0], point_batch):
                    stop = min(start + point_batch, Q.shape[0])
                    flat, weights = _periodic_linear_stencil_indices(
                        model,
                        Q[start:stop],
                        mesh_shape,
                    )
                    flat_geometry[start:stop] = flat
                    weight_geometry[start:stop] = weights
                    phase_geometry[start:stop] = np.exp(
                        2.0j * np.pi * Q[start:stop] @ centers.T
                    )
                for item in (
                    flat_geometry,
                    weight_geometry,
                    phase_geometry,
                ):
                    item.setflags(write=False)
                cached_geometry = (
                    flat_geometry,
                    weight_geometry,
                    phase_geometry,
                )
                response_cache.put(geometry_key, cached_geometry)
    response_lookup: np.ndarray | None = None
    if use_global_lookup:
        node_indices = np.asarray(
            list(product(*(range(value) for value in mesh_shape))),
            dtype=float,
        )
        node_coordinates = np.zeros((node_count, 3), dtype=float)
        for local_axis, axis in enumerate(model.periodic_axes):
            node_coordinates[:, axis] = (
                node_indices[:, local_axis] / mesh_shape[local_axis]
            )
        evaluation_q = np.repeat(
            node_coordinates,
            unique_energy.size,
            axis=0,
        )
        evaluation_energy = np.tile(unique_energy, node_count)
        lookup_response = bare_lindhard_susceptibility(
            model,
            evaluation_q,
            evaluation_energy,
            mesh,
            density_basis,
            operator_matrices_by_point=None,
            q_evaluation="direct",
            **_without_q_interpolation_kwargs(kwargs),
        )
        response_lookup = lookup_response.values_per_meV_cell.reshape(
            node_count,
            unique_energy.size,
            operators,
            operators,
        )
    scalar = np.zeros(Q.shape[0], dtype=np.complex128)
    unique_nodes: set[tuple[float, ...]] = set()
    unique_node_count = 0
    evaluated_stencils = 0
    evaluated_pairs = 0
    for start in range(0, Q.shape[0], point_batch):
        stop = min(start + point_batch, Q.shape[0])
        if response_lookup is not None:
            if cached_geometry is None:
                flat_nodes, weights = _periodic_linear_stencil_indices(
                    model,
                    Q[start:stop],
                    q_mesh_shape,
                )
                phases = np.exp(
                    2.0j
                    * np.pi
                    * Q[start:stop]
                    @ np.asarray(model.orbital_centers, dtype=float).T
                )
            else:
                flat_nodes = cached_geometry[0][start:stop]
                weights = cached_geometry[1][start:stop]
                phases = cached_geometry[2][start:stop]
            stencil_response = response_lookup[
                flat_nodes,
                energy_inverse[start:stop, None],
            ]
            scalar[start:stop] = prefactor * np.einsum(
                "pa,psab,pb,ps->p",
                phases,
                stencil_response,
                phases.conj(),
                weights,
                optimize=True,
            )
            evaluated_stencils += int(flat_nodes.size)
            continue
        flat_nodes, compact_weights = _periodic_linear_stencil_indices(
            model,
            Q[start:stop],
            q_mesh_shape,
        )
        stencil_count = flat_nodes.shape[1]
        parents = np.repeat(
            np.arange(stop - start, dtype=np.int64),
            stencil_count,
        )
        weights = compact_weights.ravel()
        nodes = np.repeat(
            np.mod(Q[start:stop], 1.0),
            stencil_count,
            axis=0,
        )
        integer_nodes = np.column_stack(
            np.unravel_index(flat_nodes.ravel(), mesh_shape)
        )
        for local_axis, axis in enumerate(model.periodic_axes):
            nodes[:, axis] = (
                integer_nodes[:, local_axis] / mesh_shape[local_axis]
            )
        node_energy = energy[start:stop][parents]
        pairs = np.column_stack((nodes, node_energy))
        unique_pairs, inverse = np.unique(
            pairs,
            axis=0,
            return_inverse=True,
        )
        node_response = bare_lindhard_susceptibility(
            model,
            unique_pairs[:, :3],
            unique_pairs[:, 3],
            mesh,
            density_basis,
            operator_matrices_by_point=None,
            q_evaluation="direct",
            **_without_q_interpolation_kwargs(kwargs),
        )
        stencil_response = node_response.values_per_meV_cell[inverse]
        evaluated_pairs += int(unique_pairs.shape[0])
        phases = np.exp(
            2.0j
            * np.pi
            * Q[start:stop]
            @ np.asarray(model.orbital_centers, dtype=float).T
        )
        stencil_values = prefactor * np.einsum(
            "sa,sab,sb->s",
            phases[parents],
            stencil_response,
            phases[parents].conj(),
            optimize=True,
        )
        np.add.at(scalar[start:stop], parents, weights * stencil_values)
        unique_nodes.update(map(tuple, np.unique(nodes, axis=0)))
        evaluated_stencils += int(nodes.shape[0])
    if response_lookup is not None:
        evaluated_pairs = node_count * unique_energy.size
        unique_node_count = node_count
    else:
        unique_node_count = len(unique_nodes)
    return scalar, {
        "method": "periodic_linear_orbital_density",
        "mesh_shape": list(q_mesh_shape),
        "requested_points": int(Q.shape[0]),
        "evaluated_stencil_points": evaluated_stencils,
        "evaluated_unique_node_energy_pairs": evaluated_pairs,
        "unique_stencil_q": unique_node_count,
        "point_batch_size": int(point_batch),
        "point_batches": int(np.ceil(Q.shape[0] / point_batch)),
        "global_node_energy_lookup": bool(response_lookup is not None),
        "lookup_bytes": int(lookup_bytes if response_lookup is not None else 0),
        "geometry_cache_hit": geometry_cache_hit,
        "geometry_cache_bytes": int(
            geometry_bytes if cached_geometry is not None else 0
        ),
        "extended_zone_phase_contraction": "exact",
    }


def _certified_implicit_spin_interpolation(
    model: ElectronicModel,
    Q: np.ndarray,
    energy: np.ndarray,
    mesh: WavevectorSampling,
    kwargs: Mapping[str, Any],
) -> SusceptibilityResult | None:
    """Return certified interpolated spin response, or ``None`` for auto fallback."""

    policy = str(kwargs.get("q_evaluation", "auto")).strip().lower()
    rtol = float(kwargs.get("q_interpolation_rtol") or 0.0)
    atol = float(kwargs.get("q_interpolation_atol") or 0.0)
    commensurate = _commensurate_point_flags(model, mesh, Q)
    approximate_indices = np.flatnonzero(~commensurate)
    if approximate_indices.size == 0:
        return None
    validation_count = min(
        int(kwargs.get("q_validation_points", 8)),
        approximate_indices.size,
    )
    validation_local = np.unique(
        np.linspace(
            0,
            approximate_indices.size - 1,
            validation_count,
            dtype=int,
        )
    )
    validation_indices = approximate_indices[validation_local]
    reference, reference_response, prefactor = _direct_implicit_spin_scalar(
        model,
        Q[validation_indices],
        energy[validation_indices],
        mesh,
        kwargs,
    )
    selected_shape: tuple[int, ...] | None = None
    attempts: list[dict[str, Any]] = []
    validation_absolute = np.inf
    validation_relative = np.inf
    for candidate in _q_mesh_candidates(
        mesh.mesh_shape or (),
        kwargs.get("q_interpolation_mesh"),
    ):
        trial, _record = _interpolated_implicit_spin_scalar(
            model,
            Q[validation_indices],
            energy[validation_indices],
            mesh,
            q_mesh_shape=candidate,
            kwargs=kwargs,
        )
        difference = np.abs(trial - reference)
        scale = max(
            float(np.max(np.abs(reference))),
            atol,
            np.finfo(float).tiny,
        )
        validation_absolute = float(np.max(difference))
        validation_relative = float(validation_absolute / scale)
        passed = bool(np.all(difference <= atol + rtol * scale))
        attempts.append(
            {
                "mesh_shape": list(candidate),
                "maximum_absolute_error": validation_absolute,
                "maximum_relative_error": validation_relative,
                "passed": passed,
            }
        )
        if passed:
            selected_shape = candidate
            break
    if selected_shape is None:
        if policy == "interpolated":
            raise ValueError(
                "q interpolation did not satisfy the requested tolerance; "
                f"maximum absolute error {validation_absolute:g}, maximum "
                f"relative error {validation_relative:g}"
            )
        exact_scalar, exact_response, exact_prefactor = (
            _direct_implicit_spin_scalar(
                model,
                Q,
                energy,
                mesh,
                kwargs,
            )
        )
        exact_tensor = np.zeros(
            (Q.shape[0], 3, 3),
            dtype=np.complex128,
        )
        exact_tensor[:, 0, 0] = exact_scalar
        exact_tensor[:, 1, 1] = exact_scalar
        exact_tensor[:, 2, 2] = exact_scalar
        fallback_certificate = _interpolation_certificate(
            {
                "schema_version": 1,
                "status": "exact_fallback",
                "certified_quantity": "bare_cartesian_spin_susceptibility",
                "model_digest": model.content_digest,
                "integration_mesh_shape": list(mesh.mesh_shape or ()),
                "selected_interpolation_mesh_shape": None,
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
                "validation_method": (
                    "deterministic direct physical spin-response comparison"
                ),
                "validation_point_count": int(validation_indices.size),
                "validation_q_reduced": Q[validation_indices].tolist(),
                "validation_energy_meV": energy[validation_indices].tolist(),
                "maximum_absolute_error": validation_absolute,
                "maximum_relative_error": validation_relative,
                "attempts": attempts,
            }
        )
        return SusceptibilityResult(
            q_reduced=exact_response.q_reduced,
            Q_reduced=exact_response.Q_reduced,
            energy_meV=exact_response.energy_meV,
            values_per_meV_cell=exact_tensor,
            operator_labels=("Sx", "Sy", "Sz"),
            conjugate_indices=(0, 1, 2),
            model_digest=exact_response.model_digest,
            temperature_K=exact_response.temperature_K,
            chemical_potential_meV=exact_response.chemical_potential_meV,
            broadening_meV=exact_response.broadening_meV,
            provenance={
                **dict(exact_response.provenance),
                "q_evaluation": {
                    "requested_policy": policy,
                    "resolved_policy": "exact_fallback",
                    "exact_commensurate_points": int(np.count_nonzero(commensurate)),
                    "interpolated_points": 0,
                    "direct_off_mesh_points": int(approximate_indices.size),
                    "interpolation_method": None,
                    "selected_mesh_shape": None,
                    "attempted_mesh_shapes": [
                        attempt["mesh_shape"] for attempt in attempts
                    ],
                    "relative_tolerance": rtol,
                    "absolute_tolerance": atol,
                    "validation_points": validation_indices.tolist(),
                    "validation_max_absolute_error": validation_absolute,
                    "validation_max_relative_error": validation_relative,
                },
                "q_interpolation_certificate": fallback_certificate,
                "operator_basis": {
                    "kind": "cartesian_spin",
                    "implicit_spin_trace_factor": exact_prefactor,
                },
            },
        )
    approximate, interpolation_record = _interpolated_implicit_spin_scalar(
        model,
        Q[approximate_indices],
        energy[approximate_indices],
        mesh,
        q_mesh_shape=selected_shape,
        kwargs=kwargs,
    )
    scalar = np.empty(Q.shape[0], dtype=np.complex128)
    scalar[approximate_indices] = approximate
    exact_indices = np.flatnonzero(commensurate)
    if exact_indices.size:
        exact, _exact_response, _exact_prefactor = _direct_implicit_spin_scalar(
            model,
            Q[exact_indices],
            energy[exact_indices],
            mesh,
            kwargs,
        )
        scalar[exact_indices] = exact
    tensor = np.zeros((Q.shape[0], 3, 3), dtype=np.complex128)
    tensor[:, 0, 0] = scalar
    tensor[:, 1, 1] = scalar
    tensor[:, 2, 2] = scalar
    certificate = _interpolation_certificate(
        {
            "schema_version": 1,
            "status": "certified",
            "certified_quantity": "bare_cartesian_spin_susceptibility",
            "model_digest": model.content_digest,
            "integration_mesh_shape": list(mesh.mesh_shape or ()),
            "selected_interpolation_mesh_shape": list(selected_shape),
            "relative_tolerance": rtol,
            "absolute_tolerance": atol,
            "validation_method": (
                "deterministic direct physical spin-response comparison"
            ),
            "relative_error_normalization": (
                "maximum direct spin-response magnitude over validation points"
            ),
            "validation_point_count": int(validation_indices.size),
            "validation_q_reduced": Q[validation_indices].tolist(),
            "validation_energy_meV": energy[validation_indices].tolist(),
            "maximum_absolute_error": validation_absolute,
            "maximum_relative_error": validation_relative,
            "attempts": attempts,
        }
    )
    return SusceptibilityResult(
        q_reduced=np.mod(Q, 1.0),
        Q_reduced=Q,
        energy_meV=energy,
        values_per_meV_cell=tensor,
        operator_labels=("Sx", "Sy", "Sz"),
        conjugate_indices=(0, 1, 2),
        model_digest=model.content_digest,
        temperature_K=reference_response.temperature_K,
        chemical_potential_meV=reference_response.chemical_potential_meV,
        broadening_meV=reference_response.broadening_meV,
        provenance={
            **dict(reference_response.provenance),
            "approximation": (
                "finite lifetime broadening and validated q interpolation"
            ),
            "q_evaluation": {
                "requested_policy": policy,
                "resolved_policy": "validated_interpolation",
                "exact_commensurate_points": int(exact_indices.size),
                "interpolated_points": int(approximate_indices.size),
                "direct_off_mesh_points": 0,
                "interpolation_method": "periodic_linear_orbital_density",
                "selected_mesh_shape": list(selected_shape),
                "attempted_mesh_shapes": [
                    attempt["mesh_shape"] for attempt in attempts
                ],
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
                "validation_points": validation_indices.tolist(),
                "validation_max_absolute_error": validation_absolute,
                "validation_max_relative_error": validation_relative,
            },
            "q_interpolation": interpolation_record,
            "q_interpolation_certificate": certificate,
            "operator_basis": {
                "kind": "cartesian_spin",
                "implicit_spin_trace_factor": prefactor,
                "interpolation_basis": "orbital_density",
                "extended_zone_phase_contraction": "exact",
            },
        },
    )


def bare_spin_susceptibility(
    model: ElectronicModel,
    q_reduced: ArrayLike,
    energy_meV: ArrayLike,
    mesh: WavevectorSampling,
    **kwargs: Any,
) -> SusceptibilityResult:
    """Return the physical Cartesian spin susceptibility per primitive cell."""

    Q, energy = _point_inputs(q_reduced, energy_meV)
    policy = str(kwargs.get("q_evaluation", "auto")).strip().lower()
    interpolation_tolerance = max(
        float(kwargs.get("q_interpolation_rtol") or 0.0),
        float(kwargs.get("q_interpolation_atol") or 0.0),
    )
    use_orbital_interpolation = (
        model.spin_operators is None
        and policy in {"auto", "interpolated"}
        and interpolation_tolerance > 0.0
        and not np.all(_commensurate_point_flags(model, mesh, Q))
    )
    if use_orbital_interpolation:
        interpolated = _certified_implicit_spin_interpolation(
            model,
            Q,
            energy,
            mesh,
            kwargs,
        )
        if interpolated is not None:
            return interpolated
    basis, matrices, prefactor = spin_operator_matrices(model, Q)
    if use_orbital_interpolation:
        kwargs = dict(kwargs)
        kwargs["q_evaluation"] = "direct"
        kwargs["q_interpolation_rtol"] = 0.0
        kwargs["q_interpolation_atol"] = 0.0
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
