"""Structure-first orbital and onsite builder for tight-binding models.

The builder stores editable orbital manifolds and symmetry-allowed onsite
terms separately from the resolved :class:`~nfit.electronic_structure.ElectronicModel`.
Analytic orbitals are represented in a declared local frame. Numerical or
custom bases remain valid electronic bases but do not acquire an inferred
symmetry character.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystal import (
    expand_crystal_sites,
    lattice_vectors,
    site_symmetry_operations,
    validate_crystal,
)
from .electronic_structure import (
    BasisState,
    ElectronicModel,
    build_electronic_model,
    electronic_energy_to_meV,
    normalize_electronic_energy_unit,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

ORBITAL_PRESETS = (
    "effective",
    "s",
    "p",
    "d",
    "t2g",
    "eg",
    "a1g_t2g",
    "eg_prime_t2g",
    "t2g_trigonal",
    "f",
    "custom",
)


class OrbitalSymmetryError(ValueError):
    """Raised when a requested orbital subspace is not closed under symmetry."""


def _complex_matrix_to_data(values: ArrayLike) -> list[list[list[float]]]:
    array = np.asarray(values, dtype=np.complex128)
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in array
    ]


def _complex_matrix_from_data(values: Any) -> ComplexArray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 2:
        return np.asarray(array[..., 0] + 1j * array[..., 1], dtype=np.complex128)
    return np.asarray(values, dtype=np.complex128)


def _frame(values: ArrayLike) -> FloatArray:
    frame = np.asarray(values, dtype=float)
    if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
        raise ValueError("local_frame must be a finite 3x3 matrix")
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8):
        raise ValueError("local_frame columns must be orthonormal")
    if np.linalg.det(frame) < 1.0 - 1e-8:
        raise ValueError("local_frame must be right-handed")
    return frame


@dataclass(frozen=True)
class OrbitalManifold:
    """One ordered orbital subspace attached to a crystallographic site.

    ``harmonic_transform`` maps the selected orbitals into the complete
    spherical-harmonic shell in nfit's documented order. It is absent for an
    effective scalar or a custom numerical basis. ``local_frame`` stores its
    orthonormal axes as columns in crystal Cartesian coordinates.
    """

    site_label: str
    label: str
    basis_kind: Literal[
        "effective_scalar",
        "real_harmonic",
        "complex_harmonic",
        "custom",
        "wannier",
    ]
    orbitals: tuple[str, ...]
    l: int | None = None
    irrep: str = ""
    degeneracy_groups: tuple[tuple[str, ...], ...] = ()
    local_frame: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    spin_basis: str = "spinless"
    correlated_shell: str = ""
    symmetry_mode: Literal["automatic", "none"] = "automatic"
    harmonic_transform: ComplexArray | None = None
    preset: str = "custom"

    def __post_init__(self) -> None:
        allowed_basis_kinds = {
            "effective_scalar",
            "real_harmonic",
            "complex_harmonic",
            "custom",
            "wannier",
        }
        if self.basis_kind not in allowed_basis_kinds:
            raise ValueError(
                f"basis_kind must be one of {sorted(allowed_basis_kinds)}"
            )
        if self.symmetry_mode not in {"automatic", "none"}:
            raise ValueError("symmetry_mode must be 'automatic' or 'none'")
        if not self.site_label.strip() or not self.label.strip():
            raise ValueError("orbital manifolds require nonempty site and manifold labels")
        if not self.orbitals or len(set(self.orbitals)) != len(self.orbitals):
            raise ValueError("orbital labels must be nonempty and unique within a manifold")
        _frame(self.local_frame)
        if self.spin_basis != "spinless":
            raise ValueError("Stage 3.2 supports spinless manifolds; spin expansion is Stage 3.5")
        if self.basis_kind in {"real_harmonic", "complex_harmonic"}:
            if self.l is None or int(self.l) < 0:
                raise ValueError("spherical-harmonic manifolds require l >= 0")
            transform = np.asarray(self.harmonic_transform, dtype=np.complex128)
            expected = (2 * int(self.l) + 1, len(self.orbitals))
            if transform.shape != expected:
                raise ValueError(
                    f"harmonic_transform must have shape {expected} for this manifold"
                )
            gram = transform.conj().T @ transform
            if not np.allclose(gram, np.eye(len(self.orbitals)), atol=1e-8):
                raise ValueError("harmonic_transform columns must be orthonormal")
            frozen = transform.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, "harmonic_transform", frozen)
        elif self.harmonic_transform is not None:
            raise ValueError("only spherical-harmonic bases use harmonic_transform")
        if self.basis_kind in {"custom", "wannier"} and self.symmetry_mode != "none":
            raise ValueError(
                "custom and Wannier bases require symmetry_mode='none' unless "
                "explicit representation matrices are added in a later stage"
            )
        known = set(self.orbitals)
        used: set[str] = set()
        for group in self.degeneracy_groups:
            if not group or any(name not in known for name in group):
                raise ValueError("degeneracy groups must contain declared orbitals")
            if used.intersection(group):
                raise ValueError("an orbital may occur in at most one degeneracy group")
            used.update(group)

    @property
    def dimension(self) -> int:
        return len(self.orbitals)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "site_label": self.site_label,
            "label": self.label,
            "basis_kind": self.basis_kind,
            "orbitals": list(self.orbitals),
            "l": self.l,
            "irrep": self.irrep,
            "degeneracy_groups": [list(group) for group in self.degeneracy_groups],
            "local_frame": np.asarray(self.local_frame, dtype=float).tolist(),
            "spin_basis": self.spin_basis,
            "correlated_shell": self.correlated_shell,
            "symmetry_mode": self.symmetry_mode,
            "preset": self.preset,
        }
        if self.harmonic_transform is not None:
            payload["harmonic_transform"] = _complex_matrix_to_data(
                self.harmonic_transform
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OrbitalManifold:
        transform = payload.get("harmonic_transform")
        return cls(
            site_label=str(payload["site_label"]),
            label=str(payload["label"]),
            basis_kind=str(payload["basis_kind"]),
            orbitals=tuple(str(value) for value in payload["orbitals"]),
            l=None if payload.get("l") is None else int(payload["l"]),
            irrep=str(payload.get("irrep", "")),
            degeneracy_groups=tuple(
                tuple(str(value) for value in group)
                for group in payload.get("degeneracy_groups", ())
            ),
            local_frame=tuple(
                tuple(float(value) for value in row)
                for row in payload.get("local_frame", np.eye(3))
            ),
            spin_basis=str(payload.get("spin_basis", "spinless")),
            correlated_shell=str(payload.get("correlated_shell", "")),
            symmetry_mode=str(payload.get("symmetry_mode", "automatic")),
            harmonic_transform=(
                None if transform is None else _complex_matrix_from_data(transform)
            ),
            preset=str(payload.get("preset", "custom")),
        )


@dataclass(frozen=True)
class OnsiteInvariant:
    """One normalized Hermitian onsite matrix and its editable coefficient."""

    identifier: str
    label: str
    site_label: str
    basis_labels: tuple[str, ...]
    matrix: ComplexArray
    kind: Literal["onsite_energy", "onsite_hybridization"]
    value_meV: float = 0.0
    bounds_meV: tuple[float | None, float | None] = (None, None)
    fit: bool = False
    source: str = "site_symmetry"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        n = len(self.basis_labels)
        if matrix.shape != (n, n) or not np.allclose(
            matrix, matrix.conj().T, atol=1e-9
        ):
            raise ValueError("onsite invariant matrix must be Hermitian and match its basis")
        if not np.isfinite(float(self.value_meV)):
            raise ValueError("onsite values must be finite")
        low, high = self.bounds_meV
        if any(
            value is not None and not np.isfinite(float(value))
            for value in (low, high)
        ):
            raise ValueError("onsite bounds must be finite or None")
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError("onsite lower bound must be below its upper bound")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "matrix", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "site_label": self.site_label,
            "basis_labels": list(self.basis_labels),
            "matrix": _complex_matrix_to_data(self.matrix),
            "kind": self.kind,
            "value_meV": float(self.value_meV),
            "bounds_meV": [
                None if value is None else float(value) for value in self.bounds_meV
            ],
            "fit": bool(self.fit),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OnsiteInvariant:
        bounds = payload.get("bounds_meV", (None, None))
        return cls(
            identifier=str(payload["identifier"]),
            label=str(payload["label"]),
            site_label=str(payload["site_label"]),
            basis_labels=tuple(str(value) for value in payload["basis_labels"]),
            matrix=_complex_matrix_from_data(payload["matrix"]),
            kind=str(payload["kind"]),
            value_meV=float(payload.get("value_meV", 0.0)),
            bounds_meV=(
                None if bounds[0] is None else float(bounds[0]),
                None if bounds[1] is None else float(bounds[1]),
            ),
            fit=bool(payload.get("fit", False)),
            source=str(payload.get("source", "site_symmetry")),
        )


def _real_harmonic_layout(l: int) -> tuple[tuple[str, int], ...]:
    if l == 0:
        return (("m0", 0),)
    if l == 1:
        return (("cos", 1), ("sin", 1), ("m0", 0))
    if l == 2:
        return (("sin", 2), ("sin", 1), ("cos", 1), ("cos", 2), ("m0", 0))
    return (("m0", 0),) + tuple(
        entry for m in range(1, l + 1) for entry in (("cos", m), ("sin", m))
    )


def _real_harmonic_names(l: int) -> tuple[str, ...]:
    if l == 0:
        return ("s",)
    if l == 1:
        return ("p_x", "p_y", "p_z")
    if l == 2:
        return ("d_xy", "d_yz", "d_zx", "d_x2_y2", "d_z2")
    shell = {3: "f"}.get(l, f"l{l}")
    return tuple(
        f"{shell}_{kind}{m}" if kind != "m0" else f"{shell}_m0"
        for kind, m in _real_harmonic_layout(l)
    )


def _selection(rows: int, indices: Sequence[int]) -> ComplexArray:
    result = np.zeros((rows, len(indices)), dtype=np.complex128)
    for column, row in enumerate(indices):
        result[int(row), column] = 1.0
    return result


def orbital_manifold_preset(
    site_label: str,
    preset: str,
    *,
    label: str | None = None,
    local_frame: ArrayLike | None = None,
    correlated_shell: str = "",
) -> OrbitalManifold:
    """Create a built-in manifold in the crystal frame unless another is given."""

    key = str(preset).strip().lower()
    if key not in ORBITAL_PRESETS:
        raise ValueError(f"unknown orbital preset {preset!r}")
    frame = np.eye(3) if local_frame is None else _frame(local_frame)
    default_label = f"{site_label}_{key}"
    if key == "effective":
        return OrbitalManifold(
            site_label=str(site_label),
            label=label or default_label,
            basis_kind="effective_scalar",
            orbitals=("effective",),
            local_frame=tuple(map(tuple, frame)),
            correlated_shell=correlated_shell,
            preset=key,
        )
    if key == "custom":
        return OrbitalManifold(
            site_label=str(site_label),
            label=label or default_label,
            basis_kind="custom",
            orbitals=("orbital",),
            local_frame=tuple(map(tuple, frame)),
            correlated_shell=correlated_shell,
            symmetry_mode="none",
            preset=key,
        )
    if key in {"s", "p", "d", "f"}:
        l = {"s": 0, "p": 1, "d": 2, "f": 3}[key]
        names = _real_harmonic_names(l)
        return OrbitalManifold(
            site_label=str(site_label),
            label=label or default_label,
            basis_kind="real_harmonic",
            l=l,
            orbitals=names,
            irrep=key,
            local_frame=tuple(map(tuple, frame)),
            correlated_shell=correlated_shell,
            harmonic_transform=np.eye(2 * l + 1, dtype=np.complex128),
            preset=key,
        )
    names_d = _real_harmonic_names(2)
    if key == "t2g":
        indices = (0, 1, 2)
        names = tuple(names_d[index] for index in indices)
        transform = _selection(5, indices)
        irrep = "t2g"
        groups = ()
    elif key == "eg":
        indices = (3, 4)
        names = tuple(names_d[index] for index in indices)
        transform = _selection(5, indices)
        irrep = "eg"
        groups = ()
    else:
        trigonal = np.zeros((5, 3), dtype=np.complex128)
        trigonal[:3, 0] = 1.0 / np.sqrt(3.0)
        trigonal[:3, 1] = np.asarray([2.0, -1.0, -1.0]) / np.sqrt(6.0)
        trigonal[:3, 2] = np.asarray([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        if key == "a1g_t2g":
            names = ("a1g",)
            transform = trigonal[:, :1]
            irrep = "a1g"
            groups = ()
        elif key == "eg_prime_t2g":
            names = ("eg_prime_1", "eg_prime_2")
            transform = trigonal[:, 1:]
            irrep = "eg_prime"
            groups = ()
        else:
            names = ("a1g", "eg_prime_1", "eg_prime_2")
            transform = trigonal
            irrep = "a1g + eg_prime"
            groups = ()
    return OrbitalManifold(
        site_label=str(site_label),
        label=label or default_label,
        basis_kind="real_harmonic",
        l=2,
        orbitals=names,
        irrep=irrep,
        degeneracy_groups=groups,
        local_frame=tuple(map(tuple, frame)),
        correlated_shell=correlated_shell,
        harmonic_transform=transform,
        preset=key,
    )


def _sphere_points(count: int) -> FloatArray:
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    phi = np.pi * (1.0 + np.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def _complex_harmonic_values(l: int, points: FloatArray) -> ComplexArray:
    from scipy import special

    z = np.clip(points[:, 2], -1.0, 1.0)
    polar = np.arccos(z)
    azimuth = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    columns = []
    for m in range(-l, l + 1):
        if hasattr(special, "sph_harm_y"):
            columns.append(special.sph_harm_y(l, m, polar, azimuth))
        else:  # scipy < 1.15
            columns.append(special.sph_harm(m, l, azimuth, polar))
    return np.column_stack(columns)


def _real_transform(l: int) -> ComplexArray:
    size = 2 * l + 1
    result = np.zeros((size, size), dtype=np.complex128)
    for column, (kind, m) in enumerate(_real_harmonic_layout(l)):
        if kind == "m0":
            result[l, column] = 1.0
        elif kind == "cos":
            result[l + m, column] = (-1) ** m / np.sqrt(2.0)
            result[l - m, column] = 1.0 / np.sqrt(2.0)
        else:
            result[l + m, column] = -1j * (-1) ** m / np.sqrt(2.0)
            result[l - m, column] = 1j / np.sqrt(2.0)
    return result


def spherical_harmonic_representation(
    l: int,
    rotation: ArrayLike,
    *,
    basis_kind: Literal["real_harmonic", "complex_harmonic"] = "real_harmonic",
) -> ComplexArray:
    """Return the active O(3) representation in nfit's harmonic convention."""

    operation = np.asarray(rotation, dtype=float)
    if operation.shape != (3, 3) or not np.allclose(
        operation.T @ operation, np.eye(3), atol=1e-8
    ):
        raise ValueError("rotation must be an orthogonal 3x3 matrix")
    points = _sphere_points(max(32, 6 * (2 * int(l) + 1)))
    values = _complex_harmonic_values(int(l), points)
    transformed = _complex_harmonic_values(int(l), points @ operation)
    if basis_kind == "real_harmonic":
        transform = _real_transform(int(l))
        values = values @ transform
        transformed = transformed @ transform
    representation = np.linalg.lstsq(values, transformed, rcond=None)[0]
    left, _singular, right = np.linalg.svd(representation)
    representation = left @ right
    representation[np.abs(representation) < 1e-12] = 0.0
    if basis_kind == "real_harmonic":
        representation = np.real_if_close(representation, tol=1000)
    return np.asarray(representation, dtype=np.complex128)


def manifold_symmetry_representation(
    manifold: OrbitalManifold | Mapping[str, Any],
    rotation_cartesian: ArrayLike,
    *,
    closure_tolerance: float = 1e-8,
) -> ComplexArray:
    """Return one site's symmetry matrix, rejecting non-closed subspaces."""

    item = (
        manifold
        if isinstance(manifold, OrbitalManifold)
        else OrbitalManifold.from_dict(manifold)
    )
    if item.symmetry_mode != "automatic":
        raise OrbitalSymmetryError(
            f"manifold {item.label!r} has no automatic symmetry representation"
        )
    if item.basis_kind == "effective_scalar":
        return np.ones((1, 1), dtype=np.complex128)
    frame = _frame(item.local_frame)
    rotation = np.asarray(rotation_cartesian, dtype=float)
    local_rotation = frame.T @ rotation @ frame
    full = spherical_harmonic_representation(
        int(item.l), local_rotation, basis_kind=item.basis_kind
    )
    transform = np.asarray(item.harmonic_transform, dtype=np.complex128)
    image = full @ transform
    projected = transform @ (transform.conj().T @ image)
    residual = float(np.linalg.norm(image - projected))
    if residual > closure_tolerance * max(1.0, float(np.linalg.norm(image))):
        raise OrbitalSymmetryError(
            f"orbital manifold {item.label!r} is not closed under the site "
            f"symmetry (closure residual {residual:.3g}); add the missing "
            "orbitals or set symmetry_mode='none'"
        )
    result = transform.conj().T @ image
    left, _singular, right = np.linalg.svd(result)
    return left @ right


def _block_diagonal(blocks: Sequence[ComplexArray]) -> ComplexArray:
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=np.complex128)
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        result[offset:stop, offset:stop] = block
        offset = stop
    return result


def _hermitian_basis(size: int) -> tuple[ComplexArray, ...]:
    result: list[ComplexArray] = []
    for row in range(size):
        matrix = np.zeros((size, size), dtype=np.complex128)
        matrix[row, row] = 1.0
        result.append(matrix)
    for row in range(size):
        for column in range(row + 1, size):
            real = np.zeros((size, size), dtype=np.complex128)
            real[row, column] = real[column, row] = 1.0 / np.sqrt(2.0)
            result.append(real)
            imaginary = np.zeros((size, size), dtype=np.complex128)
            imaginary[row, column] = 1j / np.sqrt(2.0)
            imaginary[column, row] = -1j / np.sqrt(2.0)
            result.append(imaginary)
    return tuple(result)


def _matrix_coefficients(
    matrix: ComplexArray, basis: Sequence[ComplexArray]
) -> FloatArray:
    return np.asarray(
        [float(np.vdot(item, matrix).real) for item in basis], dtype=float
    )


def _null_space(constraints: FloatArray, dimension: int) -> FloatArray:
    if constraints.size == 0:
        return np.eye(dimension)
    _left, singular, right = np.linalg.svd(constraints, full_matrices=True)
    tolerance = max(
        1e-10,
        max(constraints.shape)
        * np.finfo(float).eps
        * (singular[0] if singular.size else 1.0),
    )
    rank = int(np.sum(singular > tolerance))
    return right[rank:].T


def onsite_invariants(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    site_label: str,
) -> tuple[OnsiteInvariant, ...]:
    """Generate the symmetry-allowed Hermitian onsite basis at one site."""

    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
        if (
            item.site_label
            if isinstance(item, OrbitalManifold)
            else str(item.get("site_label", ""))
        )
        == str(site_label)
    )
    if not items:
        raise ValueError(f"no orbital manifolds are attached to site {site_label!r}")
    labels = tuple(
        f"{item.label}:{orbital}" for item in items for orbital in item.orbitals
    )
    basis = _hermitian_basis(len(labels))
    constraints: list[FloatArray] = []
    representations: list[ComplexArray] = []
    automatic = all(item.symmetry_mode == "automatic" for item in items)
    if automatic:
        for operation in site_symmetry_operations(crystal, site_label):
            blocks = [
                manifold_symmetry_representation(
                    item, operation["rotation_cartesian"]
                )
                for item in items
            ]
            representation = _block_diagonal(blocks)
            representations.append(representation)
            action = np.column_stack(
                [
                    _matrix_coefficients(
                        representation @ candidate @ representation.conj().T,
                        basis,
                    )
                    for candidate in basis
                ]
            )
            constraints.append(action - np.eye(len(basis)))
    else:
        # With no representation data, retain declared diagonal onsite groups
        # only. No symmetry or hybridization is inferred.
        off_diagonal_start = len(labels)
        for index in range(off_diagonal_start, len(basis)):
            row = np.zeros(len(basis))
            row[index] = 1.0
            constraints.append(row[None, :])

    global_index: dict[tuple[str, str], int] = {}
    cursor = 0
    for item in items:
        for orbital in item.orbitals:
            global_index[(item.label, orbital)] = cursor
            cursor += 1
        for group in item.degeneracy_groups:
            anchor = global_index[(item.label, group[0])]
            projector = np.zeros((len(labels), len(labels)), dtype=np.complex128)
            for orbital in group:
                group_index = global_index[(item.label, orbital)]
                projector[group_index, group_index] = 1.0
            if any(
                not np.allclose(
                    representation @ projector @ representation.conj().T,
                    projector,
                    atol=1e-8,
                )
                for representation in representations
            ):
                raise OrbitalSymmetryError(
                    f"declared degeneracy group {list(group)!r} in manifold "
                    f"{item.label!r} is not closed under the site symmetry"
                )
            for orbital in group[1:]:
                row = np.zeros(len(basis))
                row[anchor] = 1.0
                row[global_index[(item.label, orbital)]] = -1.0
                constraints.append(row[None, :])

    stacked = (
        np.vstack(constraints)
        if constraints
        else np.zeros((0, len(basis)), dtype=float)
    )
    null = _null_space(stacked, len(basis))
    if null.shape[1] == 0:
        raise OrbitalSymmetryError(
            f"site {site_label!r} has no onsite matrix satisfying the selected constraints"
        )
    projector = null @ null.T
    vectors: list[FloatArray] = []
    for index in range(len(basis)):
        vector = projector[:, index].copy()
        for previous in vectors:
            vector -= previous * float(previous @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            vector /= norm
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0:
                vector *= -1.0
            vectors.append(vector)
    result: list[OnsiteInvariant] = []
    energy_index = hybrid_index = 0
    for vector in vectors:
        matrix = sum(
            (
                coefficient * candidate
                for coefficient, candidate in zip(vector, basis, strict=True)
            ),
            start=np.zeros_like(basis[0]),
        )
        diagonal = np.diag(np.diag(matrix))
        if np.linalg.norm(matrix - diagonal) < 1e-9:
            energy_index += 1
            kind = "onsite_energy"
            short_label = f"epsilon_{energy_index}"
        else:
            hybrid_index += 1
            kind = "onsite_hybridization"
            short_label = f"v_{hybrid_index}"
        signature = hashlib.sha256(
            json.dumps(
                {
                    "site": str(site_label),
                    "basis": labels,
                    "matrix_real": np.round(matrix.real, 10).tolist(),
                    "matrix_imag": np.round(matrix.imag, 10).tolist(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        result.append(
            OnsiteInvariant(
                identifier=f"{site_label}:onsite:{signature}",
                label=f"{site_label} {short_label}",
                site_label=str(site_label),
                basis_labels=labels,
                matrix=matrix,
                kind=kind,
                source=(
                    "site_symmetry" if automatic else "declared_degeneracy"
                ),
            )
        )
    return tuple(result)


def generate_onsite_terms(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    *,
    previous: Sequence[OnsiteInvariant | Mapping[str, Any]] = (),
) -> tuple[OnsiteInvariant, ...]:
    """Generate all site onsite bases while preserving stable editable values."""

    validate_crystal(crystal)
    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    if len({item.label for item in items}) != len(items):
        raise ValueError("orbital manifold labels must be unique")
    sites = {str(site["label"]) for site in crystal.get("sites", [])}
    missing = sorted({item.site_label for item in items} - sites)
    if missing:
        raise ValueError(f"orbital manifolds refer to missing crystal sites {missing}")
    old = {
        item.identifier: item
        if isinstance(item, OnsiteInvariant)
        else OnsiteInvariant.from_dict(item)
        for item in previous
    }
    ordered_sites = [
        str(site["label"])
        for site in crystal.get("sites", [])
        if any(item.site_label == str(site["label"]) for item in items)
    ]
    result: list[OnsiteInvariant] = []
    for site_label in ordered_sites:
        for generated in onsite_invariants(crystal, items, site_label):
            prior = old.get(generated.identifier)
            if prior is not None:
                generated = OnsiteInvariant(
                    identifier=generated.identifier,
                    label=generated.label,
                    site_label=generated.site_label,
                    basis_labels=generated.basis_labels,
                    matrix=generated.matrix,
                    kind=generated.kind,
                    value_meV=prior.value_meV,
                    bounds_meV=prior.bounds_meV,
                    fit=prior.fit,
                    source=generated.source,
                )
            result.append(generated)
    return tuple(result)


def build_orbital_electronic_model(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    onsite_terms: Sequence[OnsiteInvariant | Mapping[str, Any]],
    *,
    periodic_axes: Sequence[int] = (0, 1, 2),
) -> ElectronicModel:
    """Resolve a structure-first onsite model into canonical tight-binding data."""

    validate_crystal(crystal)
    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    terms = tuple(
        item if isinstance(item, OnsiteInvariant) else OnsiteInvariant.from_dict(item)
        for item in onsite_terms
    )
    if not items:
        raise ValueError("add at least one orbital manifold before resolving the model")
    site_order = [
        str(site["label"])
        for site in crystal.get("sites", [])
        if any(item.site_label == str(site["label"]) for item in items)
    ]
    basis_states: list[BasisState] = []
    centers: list[tuple[float, float, float]] = []
    site_blocks: list[tuple[str, int, int, tuple[OrbitalManifold, ...]]] = []
    for site_label in site_order:
        site_manifolds = tuple(item for item in items if item.site_label == site_label)
        expanded = expand_crystal_sites(crystal, [site_label])
        for site in expanded:
            start = len(basis_states)
            for manifold in site_manifolds:
                for orbital in manifold.orbitals:
                    basis_states.append(
                        BasisState(
                            label=f"{site.label}:{manifold.label}:{orbital}",
                            site=site.label,
                            species=site.element,
                            orbital=orbital,
                            correlated_shell=manifold.correlated_shell,
                            spin="",
                            metadata={
                                "manifold": manifold.label,
                                "preset": manifold.preset,
                            },
                        )
                    )
                    centers.append(site.position)
            site_blocks.append(
                (site_label, start, len(basis_states), site_manifolds)
            )
    size = len(basis_states)
    zero = np.zeros((size, size), dtype=np.complex128)
    parameter_values: dict[str, float] = {}
    parameter_hoppings: dict[str, dict[tuple[int, int, int], ComplexArray]] = {}
    terms_by_site: dict[str, list[OnsiteInvariant]] = {}
    for term in terms:
        terms_by_site.setdefault(term.site_label, []).append(term)
    for site_label, start, stop, site_manifolds in site_blocks:
        expected = tuple(
            f"{item.label}:{orbital}"
            for item in site_manifolds
            for orbital in item.orbitals
        )
        for term in terms_by_site.get(site_label, []):
            if term.basis_labels != expected:
                raise ValueError(
                    f"onsite term {term.identifier!r} does not match the current orbital basis"
                )
            selector = parameter_hoppings.setdefault(
                term.identifier, {(0, 0, 0): np.zeros_like(zero)}
            )[(0, 0, 0)]
            selector[start:stop, start:stop] = term.matrix
            parameter_values[term.identifier] = term.value_meV
    model = build_electronic_model(
        direct_lattice=lattice_vectors(crystal["lattice"]),
        basis=basis_states,
        hoppings={(0, 0, 0): zero},
        orbital_centers=np.asarray(centers, dtype=float),
        periodic_axes=tuple(int(axis) for axis in periodic_axes),
        parameter_values=parameter_values,
        parameter_hoppings=parameter_hoppings,
        energy_unit="meV",
        provenance={
            "source": "nfit_orbital_builder",
            "spacegroup": str(crystal.get("spacegroup", "P 1")),
            "manifold_count": len(items),
            "onsite_term_count": len(terms),
            "hopping_stage": "not_configured",
        },
    )
    return model


def _component_manifolds(component: Any) -> tuple[OrbitalManifold, ...]:
    return tuple(
        OrbitalManifold.from_dict(item)
        for item in component.config.get("orbital_manifolds", ())
    )


def _component_terms(component: Any) -> tuple[OnsiteInvariant, ...]:
    return tuple(
        OnsiteInvariant.from_dict(item)
        for item in component.config.get("onsite_terms", ())
    )


def resolve_tight_binding_builder(component: Any) -> ElectronicModel:
    """Regenerate and install canonical ``model_data`` from builder records."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("the orbital builder requires a tight_binding component")
    model = build_orbital_electronic_model(
        component.config["crystal"],
        _component_manifolds(component),
        _component_terms(component),
        periodic_axes=component.config.get("periodic_axes") or (0, 1, 2),
    )
    component.config["source_path"] = ""
    component.config["model_data"] = model.to_dict()
    component.config["model_digest"] = model.content_digest
    projections: dict[str, list[int]] = {}
    for index, state in enumerate(model.basis):
        manifold_label = str(state.metadata.get("manifold", ""))
        projections.setdefault(manifold_label, []).append(index)
    component.config["projection_groups"] = projections
    return model


def set_tight_binding_orbital_manifolds(
    component: Any,
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    *,
    generate_terms: bool = True,
) -> tuple[OrbitalManifold, ...]:
    """Install validated manifold records and optionally regenerate onsite terms."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("orbital manifolds require a tight_binding component")
    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    if len({item.label for item in items}) != len(items):
        raise ValueError("orbital manifold labels must be unique")
    before = deepcopy(component.config)
    try:
        component.config["orbital_manifolds"] = [item.to_dict() for item in items]
        if generate_terms:
            generated = generate_onsite_terms(
                component.config["crystal"],
                items,
                previous=_component_terms(component),
            )
            component.config["onsite_terms"] = [
                item.to_dict() for item in generated
            ]
        if items and component.config.get("onsite_terms"):
            resolve_tight_binding_builder(component)
        else:
            component.config["onsite_terms"] = []
            component.config["model_data"] = {}
            component.config["model_digest"] = ""
    except Exception:
        component.config.clear()
        component.config.update(before)
        raise
    return items


def add_tight_binding_orbital_manifold(
    component: Any,
    manifold: OrbitalManifold | Mapping[str, Any],
) -> OrbitalManifold:
    """Append one manifold through the same validated component boundary."""

    item = (
        manifold
        if isinstance(manifold, OrbitalManifold)
        else OrbitalManifold.from_dict(manifold)
    )
    set_tight_binding_orbital_manifolds(
        component, (*_component_manifolds(component), item)
    )
    return item


def remove_tight_binding_orbital_manifold(component: Any, label: str) -> None:
    """Remove one named manifold and regenerate the remaining onsite model."""

    items = tuple(
        item for item in _component_manifolds(component) if item.label != str(label)
    )
    if len(items) == len(_component_manifolds(component)):
        raise KeyError(f"unknown orbital manifold {label!r}")
    set_tight_binding_orbital_manifolds(component, items)


def regenerate_tight_binding_onsite_terms(
    component: Any,
) -> tuple[OnsiteInvariant, ...]:
    """Regenerate symmetry invariants and preserve stable values and bounds."""

    before = deepcopy(component.config)
    try:
        terms = generate_onsite_terms(
            component.config["crystal"],
            _component_manifolds(component),
            previous=_component_terms(component),
        )
        component.config["onsite_terms"] = [item.to_dict() for item in terms]
        if terms:
            resolve_tight_binding_builder(component)
    except Exception:
        component.config.clear()
        component.config.update(before)
        raise
    return terms


def set_tight_binding_onsite_term(
    component: Any,
    identifier: str,
    *,
    value: float | None = None,
    energy_unit: str | None = None,
    lower: float | None | Literal["unchanged"] = "unchanged",
    upper: float | None | Literal["unchanged"] = "unchanged",
    fit: bool | None = None,
) -> OnsiteInvariant:
    """Update one onsite coefficient, bounds, or future fit-selection flag."""

    terms = list(_component_terms(component))
    for index, term in enumerate(terms):
        if term.identifier != str(identifier):
            continue
        unit = normalize_electronic_energy_unit(
            energy_unit or component.config.get("electronic_energy_unit", "eV")
        )
        bounds = list(term.bounds_meV)
        if lower != "unchanged":
            bounds[0] = (
                None
                if lower is None
                else float(electronic_energy_to_meV(lower, unit))
            )
        if upper != "unchanged":
            bounds[1] = (
                None
                if upper is None
                else float(electronic_energy_to_meV(upper, unit))
            )
        updated = OnsiteInvariant(
            identifier=term.identifier,
            label=term.label,
            site_label=term.site_label,
            basis_labels=term.basis_labels,
            matrix=term.matrix,
            kind=term.kind,
            value_meV=(
                term.value_meV
                if value is None
                else float(electronic_energy_to_meV(value, unit))
            ),
            bounds_meV=(bounds[0], bounds[1]),
            fit=term.fit if fit is None else bool(fit),
            source=term.source,
        )
        terms[index] = updated
        before = deepcopy(component.config)
        try:
            component.config["onsite_terms"] = [item.to_dict() for item in terms]
            resolve_tight_binding_builder(component)
        except Exception:
            component.config.clear()
            component.config.update(before)
            raise
        return updated
    raise KeyError(f"unknown onsite term {identifier!r}")


def configure_tight_binding_builder(
    component: Any,
    *,
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    onsite_terms: Sequence[OnsiteInvariant | Mapping[str, Any]] = (),
) -> ElectronicModel:
    """Rebuild a component from editable high-level builder configuration."""

    before = deepcopy(component.config)
    try:
        set_tight_binding_orbital_manifolds(
            component, manifolds, generate_terms=True
        )
        if onsite_terms:
            parsed = tuple(
                item
                if isinstance(item, OnsiteInvariant)
                else OnsiteInvariant.from_dict(item)
                for item in onsite_terms
            )
            supplied = {item.identifier: item for item in parsed}
            generated = list(_component_terms(component))
            if set(supplied) != {item.identifier for item in generated}:
                raise ValueError(
                    "stored onsite terms do not match the invariants regenerated "
                    "from the crystal and orbital manifolds"
                )
            merged = []
            for term in generated:
                source = supplied[term.identifier]
                if not np.allclose(source.matrix, term.matrix, atol=1e-9):
                    raise ValueError(
                        f"stored matrix for onsite term {term.identifier!r} changed"
                    )
                merged.append(
                    OnsiteInvariant(
                        identifier=term.identifier,
                        label=term.label,
                        site_label=term.site_label,
                        basis_labels=term.basis_labels,
                        matrix=term.matrix,
                        kind=term.kind,
                        value_meV=source.value_meV,
                        bounds_meV=source.bounds_meV,
                        fit=source.fit,
                        source=term.source,
                    )
                )
            component.config["onsite_terms"] = [item.to_dict() for item in merged]
        return resolve_tight_binding_builder(component)
    except Exception:
        component.config.clear()
        component.config.update(before)
        raise
