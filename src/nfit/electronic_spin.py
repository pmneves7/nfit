"""Optional spin and spin-orbit structure for tight-binding models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .electronic_resolvers import resolve_manifold_symmetry_representation
from .electronic_structure import BasisState, ElectronicModel
from .harmonics import complex_harmonic_angular_momentum, real_harmonic_transform

ComplexArray = NDArray[np.complex128]

SPIN_TREATMENTS = ("auto", "implicit", "collinear", "spinor")
SOC_PRESCRIPTIONS = ("auto", "atomic", "projected", "effective")


def _complex_matrix_to_data(values: ArrayLike) -> list[Any]:
    array = np.asarray(values, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _complex_matrix_from_data(values: Any) -> ComplexArray:
    array = np.asarray(values)
    if array.ndim >= 1 and array.shape[-1] == 2:
        return np.asarray(array[..., 0] + 1j * array[..., 1], dtype=np.complex128)
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True)
class SpinOrbitTerm:
    """One onsite ``lambda L.S`` coefficient attached to an orbital manifold."""

    identifier: str
    label: str
    manifold_label: str
    value_meV: float = 0.0
    bounds_meV: tuple[float | None, float | None] = (None, None)
    fit: bool = False
    prescription: Literal["auto", "atomic", "projected", "effective"] = "auto"
    orbital_operators: ComplexArray | None = None

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.label.strip():
            raise ValueError("SOC terms require nonempty identifiers and labels")
        if not self.manifold_label.strip():
            raise ValueError("SOC terms require a manifold label")
        if self.prescription not in SOC_PRESCRIPTIONS:
            raise ValueError(
                f"SOC prescription must be one of {SOC_PRESCRIPTIONS}"
            )
        if not np.isfinite(self.value_meV):
            raise ValueError("SOC coupling must be finite")
        lower, upper = self.bounds_meV
        if any(value is not None and not np.isfinite(value) for value in (lower, upper)):
            raise ValueError("SOC bounds must be finite or None")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("SOC lower bound must be below its upper bound")
        if self.orbital_operators is not None:
            operators = np.asarray(self.orbital_operators, dtype=np.complex128)
            if (
                operators.ndim != 3
                or operators.shape[0] != 3
                or operators.shape[1] != operators.shape[2]
            ):
                raise ValueError("orbital_operators must have shape (3, n, n)")
            for operator in operators:
                if not np.allclose(operator, operator.conj().T, atol=1e-10):
                    raise ValueError("orbital angular-momentum operators must be Hermitian")
            frozen = operators.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, "orbital_operators", frozen)
        if self.prescription == "effective" and self.orbital_operators is None:
            raise ValueError(
                "effective SOC requires explicit orbital angular-momentum operators"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "manifold_label": self.manifold_label,
            "value_meV": float(self.value_meV),
            "bounds_meV": list(self.bounds_meV),
            "fit": bool(self.fit),
            "prescription": self.prescription,
            "orbital_operators": (
                None
                if self.orbital_operators is None
                else _complex_matrix_to_data(self.orbital_operators)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpinOrbitTerm:
        raw_bounds = payload.get("bounds_meV", (None, None))
        raw_operators = payload.get("orbital_operators")
        return cls(
            identifier=str(payload["identifier"]),
            label=str(payload.get("label", payload["identifier"])),
            manifold_label=str(payload["manifold_label"]),
            value_meV=float(payload.get("value_meV", 0.0)),
            bounds_meV=(
                None if raw_bounds[0] is None else float(raw_bounds[0]),
                None if raw_bounds[1] is None else float(raw_bounds[1]),
            ),
            fit=bool(payload.get("fit", False)),
            prescription=str(payload.get("prescription", "auto")),
            orbital_operators=(
                None
                if raw_operators is None
                else _complex_matrix_from_data(raw_operators)
            ),
        )


def spin_orbit_term(
    manifold_label: str,
    *,
    value_meV: float = 0.0,
    bounds_meV: tuple[float | None, float | None] = (None, None),
    fit: bool = False,
    prescription: str = "auto",
    orbital_operators: ArrayLike | None = None,
) -> SpinOrbitTerm:
    """Construct a stable SOC term for one named manifold."""

    manifold = str(manifold_label)
    return SpinOrbitTerm(
        identifier=f"{manifold}:soc:lambda",
        label=f"{manifold} λ L·S",
        manifold_label=manifold,
        value_meV=float(value_meV),
        bounds_meV=bounds_meV,
        fit=bool(fit),
        prescription=str(prescription),
        orbital_operators=(
            None
            if orbital_operators is None
            else np.asarray(orbital_operators, dtype=np.complex128)
        ),
    )


def spin_half_operators(n_orbitals: int = 1) -> ComplexArray:
    """Return ``Sx, Sy, Sz`` for orbital-major ``(a↑,a↓,...)`` ordering."""

    count = int(n_orbitals)
    if count < 1:
        raise ValueError("n_orbitals must be positive")
    pauli = np.asarray(
        (
            ((0.0, 1.0), (1.0, 0.0)),
            ((0.0, -1.0j), (1.0j, 0.0)),
            ((1.0, 0.0), (0.0, -1.0)),
        ),
        dtype=np.complex128,
    )
    return np.asarray(
        [np.kron(np.eye(count), matrix / 2.0) for matrix in pauli],
        dtype=np.complex128,
    )


def manifold_orbital_operators(
    manifold: Any,
    *,
    prescription: str = "auto",
    explicit_operators: ArrayLike | None = None,
    frame_cartesian: ArrayLike | None = None,
) -> ComplexArray:
    """Return projected ``Lx, Ly, Lz`` in crystal Cartesian components."""

    selected = str(prescription)
    if selected not in SOC_PRESCRIPTIONS:
        raise ValueError(f"SOC prescription must be one of {SOC_PRESCRIPTIONS}")
    if explicit_operators is not None:
        local = np.asarray(explicit_operators, dtype=np.complex128)
        if local.shape != (3, manifold.dimension, manifold.dimension):
            raise ValueError(
                "explicit orbital operators must match the manifold dimension"
            )
    else:
        if manifold.basis_kind not in {"real_harmonic", "complex_harmonic"}:
            raise ValueError(
                f"manifold {manifold.label!r} requires explicit effective "
                "orbital operators for SOC"
            )
        if selected == "effective":
            raise ValueError("effective SOC requires explicit orbital operators")
        complete = manifold.dimension == 2 * int(manifold.l) + 1
        if selected == "atomic" and not complete:
            raise ValueError(
                f"atomic SOC requires the complete l={manifold.l} shell; use "
                "the projected prescription for this truncated subspace"
            )
        transform = np.asarray(manifold.harmonic_transform, dtype=np.complex128)
        full = complex_harmonic_angular_momentum(int(manifold.l))
        if manifold.basis_kind == "real_harmonic":
            real_transform = real_harmonic_transform(int(manifold.l))
            full = np.asarray(
                [
                    real_transform.conj().T
                    @ operator
                    @ real_transform
                    for operator in full
                ]
            )
        local = np.asarray(
            [transform.conj().T @ operator @ transform for operator in full],
            dtype=np.complex128,
        )
    frame = np.asarray(
        manifold.local_frame if frame_cartesian is None else frame_cartesian,
        dtype=float,
    )
    if frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8):
        raise ValueError("SOC local frame must be an orthonormal 3x3 matrix")
    crystal = np.einsum("ij,jab->iab", frame, local, optimize=True)
    return np.asarray(crystal, dtype=np.complex128)


def spin_orbit_matrix(orbital_operators: ArrayLike) -> ComplexArray:
    """Return dimensionless ``L.S`` in orbital-major spinor ordering."""

    orbital = np.asarray(orbital_operators, dtype=np.complex128)
    if orbital.ndim != 3 or orbital.shape[0] != 3:
        raise ValueError("orbital_operators must have shape (3, n, n)")
    spin = spin_half_operators(1)
    return sum(
        np.kron(orbital[axis], spin[axis])
        for axis in range(3)
    )


def resolved_spin_treatment(
    requested: str,
    soc_terms: Sequence[SpinOrbitTerm | Mapping[str, Any]] = (),
) -> str:
    """Resolve efficient automatic spin handling from active model terms."""

    treatment = str(requested)
    if treatment not in SPIN_TREATMENTS:
        raise ValueError(f"spin treatment must be one of {SPIN_TREATMENTS}")
    has_soc = bool(soc_terms)
    if treatment == "auto":
        return "spinor" if has_soc else "implicit"
    if treatment == "implicit" and has_soc:
        raise ValueError("implicit spin cannot represent spin-orbit coupling")
    if treatment == "collinear" and has_soc:
        raise ValueError("collinear spin cannot represent L.S spin mixing")
    return treatment


def _append_origin_if_needed(
    model: ElectronicModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    translations = np.asarray(model.translations, dtype=np.int64)
    blocks = np.asarray(model.hamiltonian_blocks, dtype=np.complex128)
    weights = np.asarray(model.interpolation_weights, dtype=float)
    parameters = {
        name: np.asarray(values, dtype=np.complex128)
        for name, values in model.parameter_blocks.items()
    }
    if np.any(np.all(translations == 0, axis=1)):
        return translations, blocks, weights, parameters
    translations = np.vstack((translations, np.zeros((1, 3), dtype=np.int64)))
    blocks = np.concatenate(
        (blocks, np.zeros((1, model.n_basis, model.n_basis), dtype=np.complex128))
    )
    weights = np.concatenate((weights, np.ones(1)))
    parameters = {
        name: np.concatenate(
            (
                values,
                np.zeros(
                    (1, model.n_basis, model.n_basis),
                    dtype=np.complex128,
                ),
            )
        )
        for name, values in parameters.items()
    }
    return translations, blocks, weights, parameters


def lift_electronic_model_spin(
    model: ElectronicModel,
    *,
    treatment: str = "implicit",
    manifolds: Sequence[Any] = (),
    soc_terms: Sequence[SpinOrbitTerm | Mapping[str, Any]] = (),
) -> ElectronicModel:
    """Return an implicit, collinear, or full-spinor electronic model."""

    terms = tuple(
        item if isinstance(item, SpinOrbitTerm) else SpinOrbitTerm.from_dict(item)
        for item in soc_terms
    )
    resolved = resolved_spin_treatment(treatment, terms)
    provenance = {
        **dict(model.provenance),
        "requested_spin_treatment": str(treatment),
        "resolved_spin_treatment": resolved,
        "implicit_spin_degeneracy": 2 if resolved == "implicit" else 1,
    }
    if resolved == "implicit":
        return replace(model, provenance=provenance)

    translations, blocks, weights, parameter_blocks = _append_origin_if_needed(model)
    lifted_blocks = np.asarray(
        [np.kron(block, np.eye(2)) for block in blocks],
        dtype=np.complex128,
    )
    lifted_parameters = {
        name: np.asarray(
            [np.kron(block, np.eye(2)) for block in values],
            dtype=np.complex128,
        )
        for name, values in parameter_blocks.items()
    }
    basis = tuple(
        BasisState(
            label=f"{state.label}:{spin}",
            site=state.site,
            species=state.species,
            orbital=state.orbital,
            correlated_shell=state.correlated_shell,
            spin=spin,
            metadata=dict(state.metadata),
        )
        for state in model.basis
        for spin in ("up", "down")
    )
    centers = np.repeat(np.asarray(model.orbital_centers), 2, axis=0)
    values = dict(model.parameter_values)
    if terms:
        manifold_lookup = {item.label: item for item in manifolds}
        origin = int(np.flatnonzero(np.all(translations == 0, axis=1))[0])
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, state in enumerate(model.basis):
            manifold_label = str(state.metadata.get("manifold", ""))
            grouped.setdefault((state.site, manifold_label), []).append(index)
        for term in terms:
            if term.identifier in values:
                raise ValueError(f"duplicate electronic parameter {term.identifier!r}")
            manifold = manifold_lookup.get(term.manifold_label)
            if manifold is None:
                raise ValueError(
                    f"SOC term refers to missing manifold {term.manifold_label!r}"
                )
            selector = np.zeros_like(lifted_blocks)
            matched = False
            for (_site, manifold_label), orbital_indices in grouped.items():
                if manifold_label != term.manifold_label:
                    continue
                matched = True
                frame = np.asarray(
                    model.basis[orbital_indices[0]].metadata.get(
                        "local_frame_cartesian",
                        manifold.local_frame,
                    ),
                    dtype=float,
                )
                orbital = manifold_orbital_operators(
                    manifold,
                    prescription=term.prescription,
                    explicit_operators=term.orbital_operators,
                    frame_cartesian=frame,
                )
                local_soc = spin_orbit_matrix(orbital)
                spinor_indices = np.asarray(
                    [
                        2 * orbital_index + spin_index
                        for orbital_index in orbital_indices
                        for spin_index in (0, 1)
                    ],
                    dtype=int,
                )
                selector[origin][np.ix_(spinor_indices, spinor_indices)] += (
                    local_soc / weights[origin]
                )
            if not matched:
                raise ValueError(
                    f"SOC manifold {term.manifold_label!r} has no resolved basis states"
                )
            values[term.identifier] = float(term.value_meV)
            lifted_parameters[term.identifier] = selector
    lifted = ElectronicModel(
        direct_lattice=model.direct_lattice,
        basis=basis,
        translations=translations,
        hamiltonian_blocks=lifted_blocks,
        interpolation_weights=weights,
        orbital_centers=centers,
        periodic_axes=model.periodic_axes,
        parameter_values=values,
        parameter_blocks=lifted_parameters,
        spin_operators=spin_half_operators(model.n_basis),
        energy_zero_meV=model.energy_zero_meV,
        provenance=provenance,
        fourier_gauge=model.fourier_gauge,
    )
    if resolved == "spinor":
        validate_spinor_time_reversal(lifted)
    return lifted


def time_reversal_unitary(n_orbitals: int) -> ComplexArray:
    """Return the unitary part of ``Theta = U K`` for spin one-half."""

    return np.kron(
        np.eye(int(n_orbitals), dtype=np.complex128),
        np.asarray(((0.0, 1.0), (-1.0, 0.0)), dtype=np.complex128),
    )


def spinor_time_reversal_residual(
    model: ElectronicModel,
    reduced_k: ArrayLike = ((0.0, 0.0, 0.0), (0.173, 0.271, 0.319)),
) -> float:
    """Return the maximum residual of ``H(k)=U H(-k)* U†``."""

    if model.n_basis % 2:
        raise ValueError("a spinor model must have an even basis dimension")
    unitary = time_reversal_unitary(model.n_basis // 2)
    wavevectors = np.atleast_2d(np.asarray(reduced_k, dtype=float))
    positive = model.hamiltonian(wavevectors)
    negative = model.hamiltonian(-wavevectors)
    transformed = np.einsum(
        "ab,kbc,cd->kad",
        unitary,
        negative.conj(),
        unitary.conj().T,
        optimize=True,
    )
    scale = max(1.0, float(np.max(np.abs(positive))))
    return float(np.max(np.abs(positive - transformed)) / scale)


def validate_spinor_time_reversal(
    model: ElectronicModel,
    *,
    tolerance: float = 1.0e-8,
) -> None:
    """Raise when a nominal nonmagnetic spinor model violates time reversal."""

    residual = spinor_time_reversal_residual(model)
    if residual > float(tolerance):
        raise ValueError(
            f"spinor Hamiltonian violates time reversal (residual {residual:.3g})"
        )


def spinor_rotation_representation(rotation_cartesian: ArrayLike) -> ComplexArray:
    """Return the spin-half representation of a spatial orthogonal operation."""

    from scipy.spatial.transform import Rotation

    operation = np.asarray(rotation_cartesian, dtype=float)
    if operation.shape != (3, 3) or not np.allclose(
        operation.T @ operation,
        np.eye(3),
        atol=1.0e-8,
    ):
        raise ValueError("rotation_cartesian must be an orthogonal 3x3 matrix")
    axial_rotation = np.linalg.det(operation) * operation
    rotation_vector = Rotation.from_matrix(axial_rotation).as_rotvec()
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-14:
        return np.eye(2, dtype=np.complex128)
    axis = rotation_vector / angle
    pauli = 2.0 * spin_half_operators(1)
    generator = sum(axis[index] * pauli[index] for index in range(3))
    return (
        np.cos(angle / 2.0) * np.eye(2)
        - 1.0j * np.sin(angle / 2.0) * generator
    )


def spinor_manifold_representation(
    manifold: Any,
    rotation_cartesian: ArrayLike,
) -> ComplexArray:
    """Return the double-group action ``D_orbital tensor D_1/2``."""

    orbital = resolve_manifold_symmetry_representation(
        manifold,
        rotation_cartesian,
    )
    return np.kron(
        orbital,
        spinor_rotation_representation(rotation_cartesian),
    )
