"""Interaction vertices and RPA dressings for electronic susceptibilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .electronic_response import SusceptibilityResult
from .electronic_structure import ElectronicModel, electronic_energy_to_meV

ComplexArray = NDArray[np.complex128]


def _readonly(value: ArrayLike, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InteractionVertex:
    """Interaction matrix in the ordered operator basis of a susceptibility."""

    operator_labels: tuple[str, ...]
    values_meV: ComplexArray
    channel: str
    kind: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly(self.values_meV, np.complex128)
        size = len(self.operator_labels)
        if values.ndim not in {2, 3} or values.shape[-2:] != (size, size):
            raise ValueError(
                "interaction values must have shape (operators, operators) or "
                "(points, operators, operators)"
            )
        if len(set(self.operator_labels)) != size:
            raise ValueError("interaction operator labels must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("interaction values must be finite")
        object.__setattr__(self, "values_meV", values)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical interaction description."""

        return {
            "operator_labels": list(self.operator_labels),
            "values_real_meV": self.values_meV.real.tolist(),
            "values_imag_meV": self.values_meV.imag.tolist(),
            "channel": self.channel,
            "kind": self.kind,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InteractionVertex:
        """Reconstruct a vertex from :meth:`to_dict` output."""

        values = np.asarray(payload["values_real_meV"], dtype=float) + 1.0j * np.asarray(
            payload["values_imag_meV"], dtype=float
        )
        return cls(
            operator_labels=tuple(str(value) for value in payload["operator_labels"]),
            values_meV=values,
            channel=str(payload["channel"]),
            kind=str(payload["kind"]),
            provenance=dict(payload.get("provenance", {})),
        )


def scalar_stoner_vertex(
    operator_labels: Sequence[str],
    interaction: float,
    *,
    energy_unit: str = "eV",
) -> InteractionVertex:
    """Return an isotropic scalar Stoner vertex in a declared operator basis."""

    value_meV = float(electronic_energy_to_meV(interaction, energy_unit))
    if not np.isfinite(value_meV):
        raise ValueError("Stoner interaction must be finite")
    labels = tuple(str(label) for label in operator_labels)
    return InteractionVertex(
        operator_labels=labels,
        values_meV=value_meV * np.eye(len(labels), dtype=np.complex128),
        channel="spin",
        kind="scalar_stoner",
        provenance={
            "input_energy_unit": energy_unit,
            "input_interaction": float(interaction),
        },
    )


def matrix_interaction_vertex(
    operator_labels: Sequence[str],
    values: ArrayLike,
    *,
    energy_unit: str = "eV",
    channel: str = "spin",
) -> InteractionVertex:
    """Return a user-supplied static interaction matrix."""

    canonical = electronic_energy_to_meV(values, energy_unit)
    return InteractionVertex(
        operator_labels=tuple(str(label) for label in operator_labels),
        values_meV=np.asarray(canonical, dtype=np.complex128),
        channel=str(channel),
        kind="user_matrix",
        provenance={"input_energy_unit": energy_unit},
    )


def correlated_basis_indices(
    model: ElectronicModel,
    shells: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Return spin-independent basis states selected by correlated-shell labels."""

    if model.spin_operators is not None:
        raise ValueError(
            "the standard Hubbard-Hund preset currently requires an implicit-spin "
            "normal-state electronic model; use a matrix vertex for spinor models"
        )
    requested = {
        str(value).strip() for value in (shells or ()) if str(value).strip()
    }
    available = {
        state.correlated_shell
        for state in model.basis
        if state.correlated_shell
    }
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"unknown correlated shell {sorted(unknown)[0]!r}; available shells are "
            f"{sorted(available)!r}"
        )
    selected = tuple(
        index
        for index, state in enumerate(model.basis)
        if state.correlated_shell
        and (not requested or state.correlated_shell in requested)
    )
    if not selected:
        raise ValueError(
            "Hubbard-Hund dressing requires at least one basis state with a "
            "correlated_shell label"
        )
    return selected


def hubbard_hund_spin_vertex(
    model: ElectronicModel,
    basis_indices: Sequence[int],
    *,
    U: float,
    U_prime: float | None = None,
    J_H: float = 0.0,
    J_pair: float | None = None,
    rotationally_invariant: bool = True,
    energy_unit: str = "eV",
) -> InteractionVertex:
    """Build the local multiorbital Hubbard-Hund spin-channel vertex.

    The operator order is ``|l1><l2|`` with the second pair ``|l3><l4|``.
    Interactions connect orbitals only within the same site-attached
    ``correlated_shell``.
    """

    indices = tuple(int(value) for value in basis_indices)
    if len(indices) != len(set(indices)) or any(
        value < 0 or value >= model.n_basis for value in indices
    ):
        raise ValueError("basis_indices must contain unique valid basis indices")
    states = tuple(model.basis[index] for index in indices)
    if any(not state.correlated_shell for state in states):
        raise ValueError("every Hubbard-Hund basis state needs correlated_shell")
    if rotationally_invariant:
        resolved_J_pair = float(J_H)
        resolved_U_prime = float(U) - 2.0 * float(J_H)
    else:
        if U_prime is None or J_pair is None:
            raise ValueError(
                "independent Hubbard-Hund parameters require U_prime and J_pair"
            )
        resolved_U_prime = float(U_prime)
        resolved_J_pair = float(J_pair)
    input_values = np.asarray(
        [U, resolved_U_prime, J_H, resolved_J_pair],
        dtype=float,
    )
    if np.any(~np.isfinite(input_values)):
        raise ValueError("Hubbard-Hund interactions must be finite")
    U_meV, Up_meV, J_meV, Jp_meV = np.asarray(
        electronic_energy_to_meV(input_values, energy_unit),
        dtype=float,
    )

    pairs = tuple((a, b) for a in range(len(indices)) for b in range(len(indices)))
    labels = tuple(
        f"{states[a].label}←{states[b].label}" for a, b in pairs
    )
    values = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)

    def same_shell(*local_indices: int) -> bool:
        keys = {
            (states[index].site, states[index].correlated_shell)
            for index in local_indices
        }
        return len(keys) == 1

    for row, (l1, l2) in enumerate(pairs):
        for column, (l3, l4) in enumerate(pairs):
            if not same_shell(l1, l2, l3, l4):
                continue
            if l1 == l2 == l3 == l4:
                values[row, column] = U_meV
            elif l1 == l3 and l2 == l4 and l1 != l2:
                values[row, column] = Up_meV
            elif l1 == l2 and l3 == l4 and l1 != l3:
                values[row, column] = J_meV
            elif l1 == l4 and l2 == l3 and l1 != l2:
                values[row, column] = Jp_meV

    return InteractionVertex(
        operator_labels=labels,
        values_meV=values,
        channel="spin",
        kind="hubbard_hund",
        provenance={
            "input_energy_unit": energy_unit,
            "basis_indices": list(indices),
            "correlated_shells": sorted(
                {state.correlated_shell for state in states}
            ),
            "site_local": True,
            "rotationally_invariant": bool(rotationally_invariant),
            "U_meV": U_meV,
            "U_prime_meV": Up_meV,
            "J_H_meV": J_meV,
            "J_pair_meV": Jp_meV,
            "pair_order": "(l1,l2),(l3,l4)",
        },
    )


def rpa_dress_susceptibility(
    bare: SusceptibilityResult,
    vertex: InteractionVertex,
    *,
    singular_tolerance: float = 1.0e-12,
) -> SusceptibilityResult:
    """Apply ``chi = (I - chi0 Gamma)^-1 chi0`` point by point."""

    if bare.operator_labels != vertex.operator_labels:
        raise ValueError(
            "interaction operator labels and order must exactly match the bare response"
        )
    threshold = float(singular_tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("singular_tolerance must be finite and positive")
    values = np.asarray(bare.values_per_meV_cell)
    matrices = np.asarray(vertex.values_meV)
    if matrices.ndim == 2:
        matrices = np.broadcast_to(matrices, values.shape)
    if matrices.shape != values.shape:
        raise ValueError("point-dependent interaction vertex must match the response")
    identity = np.eye(values.shape[-1], dtype=np.complex128)
    dressed = np.empty_like(values)
    minimum_singular_values = np.empty(values.shape[0], dtype=float)
    condition_numbers = np.empty(values.shape[0], dtype=float)
    for index, (chi0, gamma) in enumerate(zip(values, matrices, strict=True)):
        denominator = identity - chi0 @ gamma
        singular_values = np.linalg.svd(denominator, compute_uv=False)
        minimum_singular_values[index] = float(singular_values[-1])
        condition_numbers[index] = float(
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0.0
            else np.inf
        )
        scale = max(1.0, float(singular_values[0]))
        if singular_values[-1] <= threshold * scale:
            raise np.linalg.LinAlgError(
                "RPA denominator is singular within the configured tolerance "
                f"at response point {index}"
            )
        dressed[index] = np.linalg.solve(denominator, chi0)
    return SusceptibilityResult(
        q_reduced=bare.q_reduced,
        Q_reduced=bare.Q_reduced,
        energy_meV=bare.energy_meV,
        values_per_meV_cell=dressed,
        operator_labels=bare.operator_labels,
        conjugate_indices=bare.conjugate_indices,
        model_digest=bare.model_digest,
        temperature_K=bare.temperature_K,
        chemical_potential_meV=bare.chemical_potential_meV,
        broadening_meV=bare.broadening_meV,
        response_kind=f"rpa_{vertex.kind}",
        normalization=bare.normalization,
        provenance={
            **dict(bare.provenance),
            "interaction": vertex.to_dict(),
            "dressing": {
                "equation": "chi=(I-chi0 Gamma)^-1 chi0",
                "multiplication_order": "I-chi0@Gamma",
                "channel": vertex.channel,
                "singular_tolerance": threshold,
                "minimum_singular_value": minimum_singular_values.tolist(),
                "condition_number": condition_numbers.tolist(),
            },
        },
    )


def project_implicit_spin_response(
    response: SusceptibilityResult,
    model: ElectronicModel,
    basis_indices: Sequence[int],
) -> SusceptibilityResult:
    """Project an orbital-pair response onto isotropic physical spin."""

    if model.spin_operators is not None:
        raise ValueError("implicit-spin projection requires a spin-independent model")
    indices = tuple(int(value) for value in basis_indices)
    size = len(indices)
    expected_labels = tuple(
        f"{model.basis[a].label}←{model.basis[b].label}"
        for a in indices
        for b in indices
    )
    if response.operator_labels != expected_labels:
        raise ValueError("response does not use the expected orbital-pair basis")
    phases = np.exp(
        2.0j
        * np.pi
        * response.Q_reduced
        @ np.asarray(model.orbital_centers, dtype=float)[list(indices)].T
    )
    coefficients = np.zeros((response.q_reduced.shape[0], size * size), complex)
    for local_index in range(size):
        coefficients[:, local_index * size + local_index] = phases[:, local_index]
    scalar = 0.5 * np.einsum(
        "pA,pAB,pB->p",
        coefficients,
        response.values_per_meV_cell,
        coefficients.conj(),
        optimize=True,
    )
    tensor = np.zeros((len(scalar), 3, 3), dtype=np.complex128)
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
        response_kind=response.response_kind,
        normalization=response.normalization,
        provenance={
            **dict(response.provenance),
            "projection": {
                "kind": "implicit_isotropic_spin",
                "basis_indices": list(indices),
                "spin_trace_factor": 0.5,
                "position_phases": True,
            },
        },
    )
