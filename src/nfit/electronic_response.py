"""Bare multiband Lindhard susceptibilities and magnetic projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cross_section import KB_MEV_PER_K
from .electronic_backends import ElectronicBackend, evaluate_eigensystem
from .electronic_structure import ElectronicModel, WavevectorSampling

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _readonly(value: ArrayLike, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
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


def orbital_pair_operator_basis(model: ElectronicModel) -> ElectronicOperatorBasis:
    """Return the complete ordered ``|a><b|`` operator basis."""

    labels = []
    matrices = []
    conjugates = []
    for a, state_a in enumerate(model.basis):
        for b, state_b in enumerate(model.basis):
            matrix = np.zeros((model.n_basis, model.n_basis), dtype=np.complex128)
            matrix[a, b] = 1.0
            matrices.append(matrix)
            labels.append(f"{state_a.label}←{state_b.label}")
            conjugates.append(b * model.n_basis + a)
    return ElectronicOperatorBasis(
        tuple(labels),
        np.asarray(matrices),
        tuple(conjugates),
        metadata={
            "kind": "ordered_orbital_pairs",
            "multiplication": "chi_AB=<O_A O_B^dagger>",
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
) -> float:
    """Return electrons per primitive cell for a chemical potential."""

    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("electron filling requires an integration mesh")
    eigensystem = evaluate_eigensystem(
        model,
        mesh.reduced_coordinates,
        eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
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
) -> float:
    """Solve the finite-temperature chemical potential for a target filling."""

    from scipy.optimize import brentq

    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("filling-based chemical potential requires temperature > 0")
    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("filling-based chemical potential requires an integration mesh")
    eigensystem = evaluate_eigensystem(
        model,
        mesh.reduced_coordinates,
        eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
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
) -> SusceptibilityResult:
    """Evaluate the causal generalized Lindhard response.

    The response is evaluated at paired ``(q, E)`` points. The integration mesh
    may be weighted, but Phase 4 callers use a full uniform mesh as the
    reference calculation.
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
    base = evaluate_eigensystem(
        model,
        k,
        eigenvectors=True,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
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
        shifted = evaluate_eigensystem(
            model,
            k + q_value[None, :],
            eigenvectors=True,
            backend=backend,
            workers=workers,
            max_batch_bytes=max_batch_bytes,
        )
        if shifted.eigenvectors is None:  # pragma: no cover - defensive
            raise RuntimeError("Lindhard response requires band eigenvectors")
        execution_records.append(dict(shifted.provenance))
        occupation_q = _fermi_function(shifted.eigenvalues, mu, temperature)
        delta_energy = (
            base.eigenvalues[:, :, None] - shifted.eigenvalues[:, None, :]
        )
        occupation_difference = (
            occupation_k[:, :, None] - occupation_q[:, None, :]
        )
        matrix_elements = np.einsum(
            "kan,Aab,kbm->kAnm",
            base.eigenvectors.conj(),
            reference_operators,
            shifted.eigenvectors,
            optimize=True,
        )
        equal = np.abs(delta_energy) <= 1.0e-10
        static_limit = _static_equal_energy_limit(
            0.5
            * (
                base.eigenvalues[:, :, None]
                + shifted.eigenvalues[:, None, :]
            ),
            0.5
            * (
                occupation_k[:, :, None]
                + occupation_q[:, None, :]
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
            result[point_index] = np.einsum(
                "k,knm,kAnm,kBnm->AB",
                weights,
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
            "symmetry": "full_mesh_reference",
            "approximation": "finite lifetime broadening only",
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
