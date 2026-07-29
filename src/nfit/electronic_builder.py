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
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystal import (
    Bond,
    BondOrbit,
    BondSymmetry,
    CrystalSite,
    bond_stabilizer_symmetries,
    cartesian_rotation,
    expand_crystal_sites,
    generate_spatial_bond_orbits,
    lattice_vectors,
    orbits_from_config,
    orbits_to_config,
    site_symmetry_operations,
    sites_to_config,
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
    site_point_group: str = ""
    submanifold_id: str = ""

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
            "site_point_group": self.site_point_group,
            "submanifold_id": self.submanifold_id,
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
            site_point_group=str(payload.get("site_point_group", "")),
            submanifold_id=str(payload.get("submanifold_id", "")),
        )


@dataclass(frozen=True)
class SiteSymmetrySubmanifold:
    """One symmetry-closed subspace of a complete local harmonic shell."""

    identifier: str
    label: str
    site_label: str
    point_group: str
    shell: str
    l: int
    irrep_label: str
    orbitals: tuple[str, ...]
    harmonic_transform: ComplexArray

    def __post_init__(self) -> None:
        transform = np.asarray(self.harmonic_transform, dtype=np.complex128)
        expected = (2 * int(self.l) + 1, len(self.orbitals))
        if transform.shape != expected:
            raise ValueError(
                f"harmonic_transform must have shape {expected} for this submanifold"
            )
        if not np.allclose(
            transform.conj().T @ transform,
            np.eye(len(self.orbitals)),
            atol=1e-8,
        ):
            raise ValueError("submanifold transform columns must be orthonormal")
        frozen = transform.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "harmonic_transform", frozen)

    @property
    def dimension(self) -> int:
        return len(self.orbitals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "site_label": self.site_label,
            "point_group": self.point_group,
            "shell": self.shell,
            "l": int(self.l),
            "irrep_label": self.irrep_label,
            "orbitals": list(self.orbitals),
            "harmonic_transform": _complex_matrix_to_data(self.harmonic_transform),
        }


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


@dataclass(frozen=True)
class HoppingInvariant:
    """One real coefficient multiplying a symmetry-covariant hopping matrix."""

    identifier: str
    label: str
    orbit_label: str
    distance_angstrom: float
    representative_bond: Bond
    basis_i: tuple[str, ...]
    basis_j: tuple[str, ...]
    matrix: ComplexArray
    value_meV: float = 0.0
    bounds_meV: tuple[float | None, float | None] = (None, None)
    fit: bool = False
    source: str = "space_group_covariance"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.shape != (len(self.basis_i), len(self.basis_j)):
            raise ValueError(
                "hopping matrix shape must match the endpoint orbital bases"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("hopping matrices must be finite")
        if not np.isfinite(float(self.value_meV)):
            raise ValueError("hopping values must be finite")
        low, high = self.bounds_meV
        if any(
            value is not None and not np.isfinite(float(value))
            for value in (low, high)
        ):
            raise ValueError("hopping bounds must be finite or None")
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError("hopping lower bound must be below its upper bound")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "matrix", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "orbit_label": self.orbit_label,
            "distance_angstrom": float(self.distance_angstrom),
            "representative_bond": {
                "site_i": int(self.representative_bond.site_i),
                "site_j": int(self.representative_bond.site_j),
                "offset": list(self.representative_bond.offset),
            },
            "basis_i": list(self.basis_i),
            "basis_j": list(self.basis_j),
            "matrix": _complex_matrix_to_data(self.matrix),
            "value_meV": float(self.value_meV),
            "bounds_meV": [
                None if value is None else float(value)
                for value in self.bounds_meV
            ],
            "fit": bool(self.fit),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoppingInvariant:
        bond = payload["representative_bond"]
        bounds = payload.get("bounds_meV", (None, None))
        return cls(
            identifier=str(payload["identifier"]),
            label=str(payload["label"]),
            orbit_label=str(payload["orbit_label"]),
            distance_angstrom=float(payload["distance_angstrom"]),
            representative_bond=Bond(
                int(bond["site_i"]),
                int(bond["site_j"]),
                tuple(int(value) for value in bond["offset"]),
            ),
            basis_i=tuple(str(value) for value in payload["basis_i"]),
            basis_j=tuple(str(value) for value in payload["basis_j"]),
            matrix=_complex_matrix_from_data(payload["matrix"]),
            value_meV=float(payload.get("value_meV", 0.0)),
            bounds_meV=(
                None if bounds[0] is None else float(bounds[0]),
                None if bounds[1] is None else float(bounds[1]),
            ),
            fit=bool(payload.get("fit", False)),
            source=str(payload.get("source", "space_group_covariance")),
        )


@dataclass(frozen=True)
class HoppingGeneration:
    """Expanded sites, spatial bond orbits, and allowed hopping coefficients."""

    sites: tuple[CrystalSite, ...]
    orbits: tuple[BondOrbit, ...]
    terms: tuple[HoppingInvariant, ...]


def hopping_endpoint_orbitals(
    term: HoppingInvariant | Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return active ``(from_j, to_i)`` orbitals of one hopping matrix basis."""

    item = (
        term
        if isinstance(term, HoppingInvariant)
        else HoppingInvariant.from_dict(term)
    )
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    support = np.abs(item.matrix) > threshold
    from_j = tuple(
        label
        for label, active in zip(item.basis_j, np.any(support, axis=0), strict=True)
        if active
    )
    to_i = tuple(
        label
        for label, active in zip(item.basis_i, np.any(support, axis=1), strict=True)
        if active
    )
    return from_j, to_i


def _compact_orbital_group(labels: Sequence[str]) -> str:
    groups: dict[str, list[str]] = {}
    for label in labels:
        manifold, separator, orbital = str(label).partition(":")
        groups.setdefault(manifold, []).append(orbital if separator else manifold)
    parts = []
    for manifold, orbitals in groups.items():
        shown = orbitals[:3]
        suffix = f",+{len(orbitals) - 3}" if len(orbitals) > 3 else ""
        parts.append(f"{manifold}[{','.join(shown)}{suffix}]")
    return "+".join(parts)


def _descriptive_hopping_label(
    orbit_label: str,
    index: int,
    basis_i: Sequence[str],
    basis_j: Sequence[str],
    matrix: ArrayLike,
) -> str:
    support = np.abs(np.asarray(matrix)) > 1e-10
    from_j = tuple(
        label
        for label, active in zip(basis_j, np.any(support, axis=0), strict=True)
        if active
    )
    to_i = tuple(
        label
        for label, active in zip(basis_i, np.any(support, axis=1), strict=True)
        if active
    )
    return (
        f"{orbit_label} t{index}: "
        f"{_compact_orbital_group(to_i)} ← {_compact_orbital_group(from_j)}"
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
    raise AssertionError("unreachable orbital preset")


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


def site_point_group_symbol(
    crystal: Mapping[str, Any],
    site_label: str,
) -> str:
    """Identify the crystallographic point group that fixes one site."""

    import gemmi

    operations = site_symmetry_operations(crystal, site_label)
    gemmi_operations = []
    for operation in operations:
        item = gemmi.Op()
        item.rot = np.rint(
            np.asarray(operation["rotation_fractional"], dtype=float) * gemmi.Op.DEN
        ).astype(int).tolist()
        item.tran = [0, 0, 0]
        gemmi_operations.append(item)
    group = gemmi.GroupOps(gemmi_operations)
    group.add_missing_elements()
    identified = gemmi.find_spacegroup_by_ops(group)
    if identified is None:
        return f"order-{len(operations)} site group"
    return str(identified.point_group_hm())


def _canonical_subspace_basis(projector: ComplexArray) -> ComplexArray:
    dimension = int(round(float(np.trace(projector).real)))
    columns: list[ComplexArray] = []
    for index in range(projector.shape[0]):
        vector = projector[:, index].copy()
        for previous in columns:
            vector -= previous * np.vdot(previous, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        phase = vector[pivot] / abs(vector[pivot])
        vector /= phase
        columns.append(vector)
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise OrbitalSymmetryError("could not construct a stable basis for a site subspace")
    return np.column_stack(columns)


def _operation_order(rotation: ArrayLike, *, maximum: int = 12) -> int | None:
    matrix = np.asarray(rotation, dtype=float)
    product = np.eye(3)
    for order in range(1, maximum + 1):
        product = product @ matrix
        if np.allclose(product, np.eye(3), atol=1e-7):
            return order
    return None


def _trigonal_irrep_label(
    point_group: str,
    l: int,
    projector: ComplexArray,
    operations: Sequence[Mapping[str, Any]],
    representations: Sequence[ComplexArray],
) -> str:
    """Return conventional trigonal labels when the character test is unique."""

    dimension = int(round(float(np.trace(projector).real)))
    group = str(point_group).replace(" ", "")
    supported = {"3", "-3", "32", "3m", "-3m"}
    if group not in supported:
        return ""
    parity = ""
    if group in {"-3", "-3m"}:
        parity = "g" if int(l) % 2 == 0 else "u"
    if dimension == 2:
        return f"E{parity}"
    if dimension != 1:
        return ""
    if group in {"3", "-3"}:
        return f"A{parity}"

    secondary_character = None
    for operation, representation in zip(
        operations,
        representations,
        strict=True,
    ):
        rotation = np.asarray(operation["rotation_cartesian"], dtype=float)
        if _operation_order(rotation) != 2:
            continue
        determinant = float(np.linalg.det(rotation))
        if group in {"32", "-3m"} and determinant < 0.0:
            continue
        if group == "3m" and determinant > 0.0:
            continue
        secondary_character = float(
            np.trace(projector @ representation).real
        )
        break
    if secondary_character is None:
        return ""
    subscript = "1" if secondary_character > 0.0 else "2"
    return f"A{subscript}{parity}"


def site_symmetry_harmonic_submanifolds(
    crystal: Mapping[str, Any],
    site_label: str,
    shell: Literal["s", "p", "d", "f"],
    *,
    local_frame: ArrayLike | None = None,
) -> tuple[SiteSymmetrySubmanifold, ...]:
    """Split a complete harmonic shell into site-symmetry-closed subspaces.

    The decomposition is obtained from the actual stabilizer of ``site_label``.
    It therefore adapts to the crystal and local frame rather than assuming a
    named cubic or trigonal crystal-field splitting.
    """

    key = str(shell).strip().lower()
    if key not in {"s", "p", "d", "f"}:
        raise ValueError("site-symmetry submanifolds require an s, p, d, or f shell")
    frame = np.eye(3) if local_frame is None else _frame(local_frame)
    l = {"s": 0, "p": 1, "d": 2, "f": 3}[key]
    names = _real_harmonic_names(l)
    complete = orbital_manifold_preset(
        site_label,
        key,
        local_frame=frame,
    )
    operations = tuple(site_symmetry_operations(crystal, site_label))
    representations = tuple(
        manifold_symmetry_representation(
            complete,
            operation["rotation_cartesian"],
        )
        for operation in operations
    )
    point_group = site_point_group_symbol(crystal, site_label)
    size = 2 * l + 1

    # The Reynolds average of a deterministic real-symmetric operator lies in
    # the commutant of the site representation. Its eigenspaces are therefore
    # symmetry closed. A group that acts only by scalar matrices imposes no
    # meaningful orbital splitting, so it leaves the complete shell intact.
    scalar_action = all(
        np.allclose(
            representation,
            np.trace(representation) * np.eye(size) / size,
            atol=1e-8,
        )
        for representation in representations
    )
    if scalar_action:
        eigenvalues = np.zeros(size)
        eigenvectors = np.eye(size, dtype=np.complex128)
    else:
        row, column = np.indices((size, size))
        seed = (
            np.cos((row + 1) * (column + 2))
            + np.sin(np.sqrt(2.0) * (row + 2) * (column + 1))
        )
        seed = 0.5 * (seed + seed.T) + np.diag(
            np.sqrt(3.0) * np.arange(size)
        )
        invariant = sum(
            representation @ seed @ representation.conj().T
            for representation in representations
        ) / len(representations)
        invariant = 0.5 * (invariant + invariant.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(invariant)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    clusters: list[list[int]] = []
    for index, value in enumerate(eigenvalues):
        if not clusters or abs(value - eigenvalues[clusters[-1][0]]) > tolerance:
            clusters.append([index])
        else:
            clusters[-1].append(index)

    candidates: list[tuple[tuple[Any, ...], ComplexArray]] = []
    for cluster in clusters:
        vectors = eigenvectors[:, cluster]
        projector = vectors @ vectors.conj().T
        projector[np.abs(projector) < 1e-11] = 0.0
        weights = np.real(np.diag(projector))
        dominant = tuple(
            int(index)
            for index in np.argsort(-weights)[: len(cluster)]
        )
        candidates.append(((min(dominant), len(cluster), dominant), projector))
    candidates.sort(key=lambda item: item[0])

    prepared: list[dict[str, Any]] = []
    for index, (_sort_key, projector) in enumerate(candidates, start=1):
        diagonal = np.real(np.diag(projector))
        selected = tuple(
            int(item)
            for item in np.flatnonzero(np.isclose(diagonal, 1.0, atol=1e-8))
        )
        selector = (
            len(selected) == int(round(float(np.trace(projector).real)))
            and np.allclose(
                projector,
                np.diag(np.isclose(diagonal, 1.0, atol=1e-8).astype(float)),
                atol=1e-8,
            )
        )
        transform = (
            _selection(size, selected)
            if selector
            else _canonical_subspace_basis(projector)
        )
        orbital_names = (
            tuple(names[item] for item in selected)
            if selector
            else tuple(f"{key}_subspace_{index}_{item + 1}" for item in range(transform.shape[1]))
        )
        contributions = [
            (names[item], float(diagonal[item]) / transform.shape[1])
            for item in np.argsort(-diagonal)
            if diagonal[item] > 1e-6
        ]
        description = ", ".join(
            f"{name} {100.0 * weight:.0f}%"
            for name, weight in contributions[:3]
        )
        digest = hashlib.sha256(
            np.round(projector, decimals=10).tobytes()
        ).hexdigest()[:12]
        identifier = f"{key}:{point_group}:{digest}"
        prepared.append(
            {
                "identifier": identifier,
                "index": index,
                "transform": transform,
                "orbitals": orbital_names,
                "description": description,
                "irrep_label": _trigonal_irrep_label(
                    point_group,
                    l,
                    projector,
                    operations,
                    representations,
                ),
            }
        )

    irrep_counts: dict[str, int] = {}
    for item in prepared:
        irrep = str(item["irrep_label"])
        if irrep:
            irrep_counts[irrep] = irrep_counts.get(irrep, 0) + 1
    irrep_seen: dict[str, int] = {}
    result: list[SiteSymmetrySubmanifold] = []
    for item in prepared:
        irrep = str(item["irrep_label"])
        if irrep:
            irrep_seen[irrep] = irrep_seen.get(irrep, 0) + 1
            irrep_text = irrep
            if irrep_counts[irrep] > 1:
                irrep_text += (
                    f" copy {irrep_seen[irrep]}/{irrep_counts[irrep]}"
                )
        else:
            irrep_text = f"subspace {item['index']}"
        label = (
            f"{point_group} {irrep_text} "
            f"(dimension {item['transform'].shape[1]}; "
            f"{item['description']})"
        )
        result.append(
            SiteSymmetrySubmanifold(
                identifier=str(item["identifier"]),
                label=label,
                site_label=str(site_label),
                point_group=point_group,
                shell=key,
                l=l,
                irrep_label=irrep,
                orbitals=tuple(item["orbitals"]),
                harmonic_transform=np.asarray(item["transform"]),
            )
        )
    return tuple(result)


def orbital_manifold_from_site_symmetry(
    crystal: Mapping[str, Any],
    site_label: str,
    shell: Literal["s", "p", "d", "f"],
    submanifold_id: str,
    *,
    label: str | None = None,
    local_frame: ArrayLike | None = None,
    correlated_shell: str = "",
) -> OrbitalManifold:
    """Create one orbital manifold selected from a site's harmonic subspaces."""

    options = site_symmetry_harmonic_submanifolds(
        crystal,
        site_label,
        shell,
        local_frame=local_frame,
    )
    selected = next(
        (item for item in options if item.identifier == str(submanifold_id)),
        None,
    )
    if selected is None:
        raise ValueError(
            f"unknown {shell!r} submanifold {submanifold_id!r} at "
            f"site {site_label!r}"
        )
    frame = np.eye(3) if local_frame is None else _frame(local_frame)
    default_label = f"{site_label}_{shell}_subspace"
    return OrbitalManifold(
        site_label=str(site_label),
        label=label or default_label,
        basis_kind="real_harmonic",
        orbitals=selected.orbitals,
        l=selected.l,
        irrep=selected.label,
        local_frame=tuple(map(tuple, frame)),
        correlated_shell=correlated_shell,
        harmonic_transform=selected.harmonic_transform,
        preset="site_symmetry",
        site_point_group=selected.point_group,
        submanifold_id=selected.identifier,
    )


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


@dataclass(frozen=True)
class _ExpandedOrbitalSite:
    representative_label: str
    site: CrystalSite
    manifolds: tuple[OrbitalManifold, ...]
    generator_cartesian: FloatArray

    @property
    def basis_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{manifold.label}:{orbital}"
            for manifold in self.manifolds
            for orbital in manifold.orbitals
        )


def _expanded_orbital_sites(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold],
    *,
    expected_sites: Sequence[CrystalSite] | None = None,
) -> tuple[_ExpandedOrbitalSite, ...]:
    ordered_labels = [
        str(site["label"])
        for site in crystal.get("sites", ())
        if any(item.site_label == str(site["label"]) for item in manifolds)
    ]
    contexts: list[_ExpandedOrbitalSite] = []
    seen: set[tuple[float, float, float]] = set()
    for representative in ordered_labels:
        site_manifolds = tuple(
            item for item in manifolds if item.site_label == representative
        )
        for site in expand_crystal_sites(crystal, [representative]):
            key = tuple(site.position)
            if key in seen:
                continue
            seen.add(key)
            generator = (
                np.eye(3)
                if site.rotation is None
                else cartesian_rotation(site.rotation, crystal["lattice"])
            )
            contexts.append(
                _ExpandedOrbitalSite(
                    representative_label=representative,
                    site=site,
                    manifolds=site_manifolds,
                    generator_cartesian=np.asarray(generator, dtype=float),
                )
            )
    if expected_sites is not None:
        actual = tuple(context.site.position for context in contexts)
        expected = tuple(site.position for site in expected_sites)
        if actual != expected:
            raise OrbitalSymmetryError(
                "expanded orbital sites do not match the generated bond-site order"
            )
    return tuple(contexts)


def _site_mapping_representation(
    source: _ExpandedOrbitalSite,
    target: _ExpandedOrbitalSite,
    rotation_cartesian: ArrayLike,
) -> FloatArray:
    if source.representative_label != target.representative_label:
        raise OrbitalSymmetryError(
            "a space-group operation mapped between inequivalent orbital sites"
        )
    if tuple(item.label for item in source.manifolds) != tuple(
        item.label for item in target.manifolds
    ):
        raise OrbitalSymmetryError(
            "symmetry-related sites do not carry the same ordered manifolds"
        )
    relative = (
        target.generator_cartesian.T
        @ np.asarray(rotation_cartesian, dtype=float)
        @ source.generator_cartesian
    )
    blocks: list[ComplexArray] = []
    for manifold in source.manifolds:
        if manifold.symmetry_mode == "none":
            if not np.allclose(relative, np.eye(3), atol=1e-8):
                raise OrbitalSymmetryError(
                    f"manifold {manifold.label!r} has no representation for "
                    "a nontrivial hopping symmetry operation"
                )
            block = np.eye(manifold.dimension, dtype=np.complex128)
        else:
            block = manifold_symmetry_representation(manifold, relative)
        blocks.append(block)
    representation = _block_diagonal(blocks)
    if np.max(np.abs(representation.imag), initial=0.0) > 1e-8:
        raise OrbitalSymmetryError(
            "Stage 3.3 hopping generation requires a real spinless orbital "
            "representation; complex/spinor hopping follows in Stage 3.5"
        )
    return np.asarray(representation.real, dtype=float)


def _mapped_hopping_matrix(
    contexts: Sequence[_ExpandedOrbitalSite],
    representative: Bond,
    member: Bond,
    symmetry: BondSymmetry,
    matrix: ArrayLike,
    crystal: Mapping[str, Any],
) -> FloatArray:
    rotation = cartesian_rotation(symmetry.rotation, crystal["lattice"])
    source_i = contexts[representative.site_i]
    source_j = contexts[representative.site_j]
    complex_values = np.asarray(matrix, dtype=np.complex128)
    if np.max(np.abs(complex_values.imag), initial=0.0) > 1e-8:
        raise OrbitalSymmetryError(
            "Stage 3.3 generated hopping matrices must be real"
        )
    values = np.asarray(complex_values.real, dtype=float)
    if not symmetry.reverses:
        target_i = contexts[member.site_i]
        target_j = contexts[member.site_j]
        left = _site_mapping_representation(source_i, target_i, rotation)
        right = _site_mapping_representation(source_j, target_j, rotation)
        return left @ values @ right.T
    target_for_j = contexts[member.site_i]
    target_for_i = contexts[member.site_j]
    left = _site_mapping_representation(source_j, target_for_j, rotation)
    right = _site_mapping_representation(source_i, target_for_i, rotation)
    return left @ values.T @ right.T


def _canonical_real_subspace_basis(projector: FloatArray) -> FloatArray:
    dimension = int(round(float(np.trace(projector))))
    columns: list[FloatArray] = []
    for index in range(projector.shape[0]):
        vector = projector[:, index].copy()
        for previous in columns:
            vector -= previous * float(previous @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        vector /= norm
        lead = int(np.argmax(np.abs(vector)))
        if vector[lead] < 0.0:
            vector *= -1.0
        columns.append(vector)
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise OrbitalSymmetryError(
            "could not construct a stable symmetry-allowed hopping basis"
        )
    return np.column_stack(columns)


def hopping_invariants(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    sites: Sequence[CrystalSite],
    orbit: BondOrbit,
) -> tuple[HoppingInvariant, ...]:
    """Generate the real spinless hopping basis allowed on one bond orbit."""

    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    contexts = _expanded_orbital_sites(
        crystal,
        items,
        expected_sites=sites,
    )
    representative = orbit.bonds[0]
    context_i = contexts[representative.site_i]
    context_j = contexts[representative.site_j]
    rows = len(context_i.basis_labels)
    columns = len(context_j.basis_labels)
    size = rows * columns
    constraints: list[FloatArray] = []
    for symmetry in bond_stabilizer_symmetries(
        crystal,
        sites,
        representative,
    ):
        action = np.column_stack(
            [
                _mapped_hopping_matrix(
                    contexts,
                    representative,
                    representative,
                    symmetry,
                    np.eye(size)[:, index].reshape(rows, columns),
                    crystal,
                ).reshape(size)
                for index in range(size)
            ]
        )
        constraints.append(action - np.eye(size))
    stacked = (
        np.vstack(constraints)
        if constraints
        else np.zeros((0, size), dtype=float)
    )
    null = _null_space(stacked, size)
    if null.shape[1] == 0:
        return ()
    projector = null @ null.T
    basis = _canonical_real_subspace_basis(projector)
    result = []
    for index in range(basis.shape[1]):
        matrix = basis[:, index].reshape(rows, columns)
        digest_payload = {
            "orbit": orbit.label,
            "bond": [
                representative.site_i,
                representative.site_j,
                *representative.offset,
            ],
            "basis_i": context_i.basis_labels,
            "basis_j": context_j.basis_labels,
            "matrix": np.round(matrix, decimals=10).tolist(),
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        result.append(
            HoppingInvariant(
                identifier=f"{orbit.label}:hopping:{digest}",
                label=_descriptive_hopping_label(
                    orbit.label,
                    index + 1,
                    context_i.basis_labels,
                    context_j.basis_labels,
                    matrix,
                ),
                orbit_label=orbit.label,
                distance_angstrom=orbit.distance_angstrom,
                representative_bond=representative,
                basis_i=context_i.basis_labels,
                basis_j=context_j.basis_labels,
                matrix=np.asarray(matrix, dtype=np.complex128),
                source="spinless_time_reversal_space_group",
            )
        )
    return tuple(result)


def generate_hopping_terms(
    crystal: Mapping[str, Any],
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    cutoff_angstrom: float,
    *,
    previous: Sequence[HoppingInvariant | Mapping[str, Any]] = (),
) -> HoppingGeneration:
    """Generate spatial orbits and symmetry-allowed hopping coefficients."""

    validate_crystal(crystal)
    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    active_labels = [
        str(site["label"])
        for site in crystal.get("sites", ())
        if any(item.site_label == str(site["label"]) for item in items)
    ]
    if not active_labels:
        raise ValueError("add at least one orbital manifold before generating hoppings")
    sites, orbits = generate_spatial_bond_orbits(
        crystal,
        active_labels,
        float(cutoff_angstrom),
    )
    old = {
        item.identifier: item
        if isinstance(item, HoppingInvariant)
        else HoppingInvariant.from_dict(item)
        for item in previous
    }
    terms: list[HoppingInvariant] = []
    for orbit in orbits:
        for generated in hopping_invariants(crystal, items, sites, orbit):
            prior = old.get(generated.identifier)
            if prior is not None:
                generated = HoppingInvariant(
                    identifier=generated.identifier,
                    label=generated.label,
                    orbit_label=generated.orbit_label,
                    distance_angstrom=generated.distance_angstrom,
                    representative_bond=generated.representative_bond,
                    basis_i=generated.basis_i,
                    basis_j=generated.basis_j,
                    matrix=generated.matrix,
                    value_meV=prior.value_meV,
                    bounds_meV=prior.bounds_meV,
                    fit=prior.fit,
                    source=generated.source,
                )
            terms.append(generated)
    return HoppingGeneration(
        sites=tuple(sites),
        orbits=tuple(orbits),
        terms=tuple(terms),
    )


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
    hopping_orbits: Sequence[BondOrbit | Mapping[str, Any]] = (),
    hopping_terms: Sequence[HoppingInvariant | Mapping[str, Any]] = (),
    periodic_axes: Sequence[int] = (0, 1, 2),
) -> ElectronicModel:
    """Resolve structure-first onsite and hopping terms into canonical data."""

    validate_crystal(crystal)
    items = tuple(
        item if isinstance(item, OrbitalManifold) else OrbitalManifold.from_dict(item)
        for item in manifolds
    )
    terms = tuple(
        item if isinstance(item, OnsiteInvariant) else OnsiteInvariant.from_dict(item)
        for item in onsite_terms
    )
    orbits = tuple(
        item if isinstance(item, BondOrbit) else orbits_from_config([item])[0]
        for item in hopping_orbits
    )
    hopping_items = tuple(
        item
        if isinstance(item, HoppingInvariant)
        else HoppingInvariant.from_dict(item)
        for item in hopping_terms
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
                generator = cartesian_rotation(
                    (
                        np.eye(3)
                        if site.rotation is None
                        else np.asarray(site.rotation, dtype=float)
                    ),
                    crystal["lattice"],
                )
                local_frame_cartesian = generator @ np.asarray(
                    manifold.local_frame,
                    dtype=float,
                )
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
                                "local_frame_cartesian": (
                                    local_frame_cartesian.tolist()
                                ),
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

    contexts = _expanded_orbital_sites(crystal, items)
    if len(contexts) != len(site_blocks):
        raise OrbitalSymmetryError(
            "expanded hopping sites do not match the resolved electronic basis"
        )
    ranges = tuple((start, stop) for _label, start, stop, _items in site_blocks)
    orbit_lookup = {orbit.label: orbit for orbit in orbits}
    for term in hopping_items:
        orbit = orbit_lookup.get(term.orbit_label)
        if orbit is None:
            raise ValueError(
                f"hopping term {term.identifier!r} refers to missing orbit "
                f"{term.orbit_label!r}"
            )
        if orbit.operations is None or len(orbit.operations) != len(orbit.bonds):
            raise ValueError(
                f"hopping orbit {orbit.label!r} lacks symmetry mapping operations"
            )
        if term.representative_bond != orbit.bonds[0]:
            raise ValueError(
                f"hopping term {term.identifier!r} no longer matches its "
                "representative bond"
            )
        representative = orbit.bonds[0]
        if term.basis_i != contexts[representative.site_i].basis_labels:
            raise ValueError(
                f"hopping term {term.identifier!r} no longer matches endpoint i"
            )
        if term.basis_j != contexts[representative.site_j].basis_labels:
            raise ValueError(
                f"hopping term {term.identifier!r} no longer matches endpoint j"
            )
        selector_blocks = parameter_hoppings.setdefault(term.identifier, {})
        for member, symmetry in zip(
            orbit.bonds,
            orbit.operations,
            strict=True,
        ):
            matrix = _mapped_hopping_matrix(
                contexts,
                representative,
                member,
                symmetry,
                term.matrix,
                crystal,
            )
            vector = tuple(int(value) for value in member.offset)
            partner = tuple(-value for value in vector)
            start_i, stop_i = ranges[member.site_i]
            start_j, stop_j = ranges[member.site_j]
            block = selector_blocks.setdefault(vector, np.zeros_like(zero))
            block[start_i:stop_i, start_j:stop_j] += matrix
            partner_block = selector_blocks.setdefault(
                partner,
                np.zeros_like(zero),
            )
            partner_block[start_j:stop_j, start_i:stop_i] += matrix.T
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
            "hopping_orbit_count": len(orbits),
            "hopping_term_count": len(hopping_items),
            "hopping_stage": (
                "symmetry_generated" if hopping_items else "not_configured"
            ),
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


def _component_hopping_terms(component: Any) -> tuple[HoppingInvariant, ...]:
    return tuple(
        HoppingInvariant.from_dict(item)
        for item in component.config.get("hopping_terms", ())
    )


def _component_soc_terms(component: Any) -> tuple[Any, ...]:
    from .electronic_spin import SpinOrbitTerm

    return tuple(
        SpinOrbitTerm.from_dict(item)
        for item in component.config.get("soc_terms", ())
    )


def _component_hopping_candidates(
    component: Any,
) -> tuple[HoppingInvariant, ...]:
    return tuple(
        HoppingInvariant.from_dict(item)
        for item in component.config.get("hopping_candidates", ())
    )


def _component_hopping_orbits(component: Any) -> tuple[BondOrbit, ...]:
    return tuple(orbits_from_config(component.config.get("spatial_orbits", ())))


def tight_binding_parameter_terms(
    component: Any,
) -> tuple[Any, ...]:
    """Return active named Hamiltonian terms in stable builder order."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("tight-binding parameters require a tight_binding component")
    terms = (
        *_component_terms(component),
        *_component_hopping_terms(component),
        *_component_soc_terms(component),
    )
    names = [term.identifier for term in terms]
    if len(names) != len(set(names)):
        raise ValueError("tight-binding parameter identifiers must be unique")
    return terms


def tight_binding_parameter_names(component: Any) -> tuple[str, ...]:
    """Return stable optimizer names for active onsite and hopping terms."""

    return tuple(term.identifier for term in tight_binding_parameter_terms(component))


def tight_binding_parameter_labels(component: Any) -> dict[str, str]:
    """Return stable parameter identifier -> descriptive builder label."""

    return {
        term.identifier: term.label
        for term in tight_binding_parameter_terms(component)
    }


def _builder_state_snapshot(component: Any) -> dict[str, Any]:
    return {
        name: deepcopy(getattr(component, name))
        for name in (
            "config",
            "parameters",
            "fit_parameters",
            "limits",
            "sharing",
            "metadata",
        )
    }


def _restore_builder_state(component: Any, state: Mapping[str, Any]) -> None:
    for name, values in state.items():
        mapping = getattr(component, name)
        mapping.clear()
        mapping.update(values)


def _term_with_parameter_state(
    term: Any,
    component: Any,
) -> Any:
    name = term.identifier
    value = float(component.parameters.get(name, term.value_meV))
    raw_bounds = component.limits.get(name, term.bounds_meV)
    bounds = (
        tuple(raw_bounds)
        if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 2
        else term.bounds_meV
    )
    return replace(
        term,
        value_meV=value,
        bounds_meV=(
            None if bounds[0] in (None, "") else float(bounds[0]),
            None if bounds[1] in (None, "") else float(bounds[1]),
        ),
        fit=bool(component.fit_parameters.get(name, term.fit)),
    )


def _adopt_term_parameter_state(
    component: Any,
    terms: Sequence[Any],
) -> None:
    for term in terms:
        component.parameters[term.identifier] = float(term.value_meV)
        component.fit_parameters[term.identifier] = bool(term.fit)
        if term.bounds_meV == (None, None):
            component.limits.pop(term.identifier, None)
        else:
            component.limits[term.identifier] = list(term.bounds_meV)
        component.sharing.setdefault(
            term.identifier,
            {"mode": "global", "groups": {}},
        )


def reconcile_tight_binding_parameters(component: Any) -> tuple[str, ...]:
    """Synchronize builder terms with common parameter, fit, and sharing state.

    The common :class:`ModelComponentSpec` mappings are authoritative after a
    parameter has been installed. Term records retain a mirrored copy so that
    the high-level builder remains independently serializable and readable.
    """

    terms = tight_binding_parameter_terms(component)
    names = tuple(term.identifier for term in terms)
    keep = set(names)
    metadata = component.metadata if isinstance(component.metadata, dict) else {}
    previous = {
        str(name)
        for name in metadata.get("tight_binding_parameter_ids", ())
    }
    for mapping in (
        component.parameters,
        component.fit_parameters,
        component.limits,
        component.sharing,
    ):
        if isinstance(mapping, dict):
            for name in previous - keep:
                mapping.pop(name, None)

    for term in terms:
        name = term.identifier
        component.parameters.setdefault(name, float(term.value_meV))
        component.fit_parameters.setdefault(name, bool(term.fit))
        if name not in component.limits and term.bounds_meV != (None, None):
            component.limits[name] = list(term.bounds_meV)
        entry = component.sharing.get(name)
        if not isinstance(entry, dict) or entry.get("mode") not in {
            "global",
            "per_dataset",
            "grouped",
        }:
            component.sharing[name] = {"mode": "global", "groups": {}}
        else:
            entry.setdefault("groups", {})

    updated_onsite = [
        _term_with_parameter_state(term, component)
        for term in _component_terms(component)
    ]
    updated_hoppings = [
        _term_with_parameter_state(term, component)
        for term in _component_hopping_terms(component)
    ]
    updated_soc = [
        _term_with_parameter_state(term, component)
        for term in _component_soc_terms(component)
    ]
    component.config["onsite_terms"] = [term.to_dict() for term in updated_onsite]
    component.config["hopping_terms"] = [
        term.to_dict() for term in updated_hoppings
    ]
    component.config["soc_terms"] = [term.to_dict() for term in updated_soc]
    active_hoppings = {term.identifier: term for term in updated_hoppings}
    component.config["hopping_candidates"] = [
        active_hoppings.get(term.identifier, term).to_dict()
        for term in _component_hopping_candidates(component)
    ]
    metadata["tight_binding_parameter_ids"] = list(names)
    component.metadata = metadata
    return names


def set_tight_binding_parameter_state(
    component: Any,
    *,
    values_meV: Mapping[str, float] | None = None,
    fit_parameters: Mapping[str, bool] | None = None,
    limits_meV: Mapping[str, Sequence[float | None]] | None = None,
    sharing: Mapping[str, Mapping[str, Any]] | None = None,
) -> ElectronicModel:
    """Update common tight-binding parameter state and rebuild the model.

    Values and bounds at this API boundary are canonical meV. The specialized
    term setters remain the display-unit boundary used by the GUI.
    """

    before = _builder_state_snapshot(component)
    try:
        return _set_tight_binding_parameter_state_unchecked(
            component,
            values_meV=values_meV,
            fit_parameters=fit_parameters,
            limits_meV=limits_meV,
            sharing=sharing,
        )
    except Exception:
        _restore_builder_state(component, before)
        raise


def set_tight_binding_spin_treatment(
    component: Any,
    treatment: str,
) -> ElectronicModel:
    """Select automatic, implicit, collinear, or full-spinor handling."""

    from .electronic_spin import SPIN_TREATMENTS

    selected = str(treatment)
    if selected not in SPIN_TREATMENTS:
        raise ValueError(f"spin treatment must be one of {SPIN_TREATMENTS}")
    before = _builder_state_snapshot(component)
    try:
        component.config["spin_treatment"] = selected
        return resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise


def set_tight_binding_soc_term(
    component: Any,
    manifold_label: str,
    *,
    enabled: bool = True,
    value: float | None = None,
    energy_unit: str | None = None,
    lower: float | None | Literal["unchanged"] = "unchanged",
    upper: float | None | Literal["unchanged"] = "unchanged",
    fit: bool | None = None,
    prescription: str | None = None,
    orbital_operators: ArrayLike | None = None,
) -> ElectronicModel:
    """Add, update, or remove onsite SOC for one spatial manifold."""

    from .electronic_spin import SpinOrbitTerm, spin_orbit_term

    label = str(manifold_label)
    manifolds = {item.label: item for item in _component_manifolds(component)}
    if label not in manifolds:
        raise KeyError(f"unknown orbital manifold {label!r}")
    terms = list(_component_soc_terms(component))
    existing = next(
        (term for term in terms if term.manifold_label == label),
        None,
    )
    before = _builder_state_snapshot(component)
    try:
        if not enabled:
            component.config["soc_terms"] = [
                term.to_dict()
                for term in terms
                if term.manifold_label != label
            ]
            reconcile_tight_binding_parameters(component)
            return resolve_tight_binding_builder(component)
        unit = normalize_electronic_energy_unit(
            energy_unit or component.config.get("electronic_energy_unit", "eV")
        )
        if existing is None:
            existing = spin_orbit_term(label)
            terms.append(existing)
        bounds = list(existing.bounds_meV)
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
        updated = SpinOrbitTerm(
            identifier=existing.identifier,
            label=existing.label,
            manifold_label=existing.manifold_label,
            value_meV=(
                existing.value_meV
                if value is None
                else float(electronic_energy_to_meV(value, unit))
            ),
            bounds_meV=(bounds[0], bounds[1]),
            fit=existing.fit if fit is None else bool(fit),
            prescription=(
                existing.prescription
                if prescription is None
                else str(prescription)
            ),
            orbital_operators=(
                existing.orbital_operators
                if orbital_operators is None
                else np.asarray(orbital_operators, dtype=np.complex128)
            ),
        )
        terms = [
            updated if term.manifold_label == label else term
            for term in terms
        ]
        component.config["soc_terms"] = [term.to_dict() for term in terms]
        _adopt_term_parameter_state(component, (updated,))
        reconcile_tight_binding_parameters(component)
        return resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise


def configure_tight_binding_spin(
    component: Any,
    *,
    treatment: str = "auto",
    soc_terms: Sequence[Mapping[str, Any] | Any] = (),
) -> ElectronicModel:
    """Install serialized spin/SOC builder state and resolve the model."""

    from .electronic_spin import SPIN_TREATMENTS, SpinOrbitTerm

    selected = str(treatment)
    if selected not in SPIN_TREATMENTS:
        raise ValueError(f"spin treatment must be one of {SPIN_TREATMENTS}")
    parsed = tuple(
        item if isinstance(item, SpinOrbitTerm) else SpinOrbitTerm.from_dict(item)
        for item in soc_terms
    )
    known = {item.label for item in _component_manifolds(component)}
    missing = sorted({item.manifold_label for item in parsed} - known)
    if missing:
        raise ValueError(f"SOC terms refer to missing manifolds {missing}")
    before = _builder_state_snapshot(component)
    try:
        component.config["spin_treatment"] = selected
        component.config["soc_terms"] = [item.to_dict() for item in parsed]
        _adopt_term_parameter_state(component, parsed)
        reconcile_tight_binding_parameters(component)
        return resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise


def _set_tight_binding_parameter_state_unchecked(
    component: Any,
    *,
    values_meV: Mapping[str, float] | None,
    fit_parameters: Mapping[str, bool] | None,
    limits_meV: Mapping[str, Sequence[float | None]] | None,
    sharing: Mapping[str, Mapping[str, Any]] | None,
) -> ElectronicModel:
    names = set(reconcile_tight_binding_parameters(component))
    supplied = set(values_meV or ())
    supplied.update(fit_parameters or ())
    supplied.update(limits_meV or ())
    supplied.update(sharing or ())
    unknown = supplied - names
    if unknown:
        raise KeyError(f"unknown tight-binding parameter {sorted(unknown)[0]!r}")

    for name, value in (values_meV or {}).items():
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("tight-binding parameter values must be finite")
        component.parameters[name] = numeric
    for name, selected in (fit_parameters or {}).items():
        component.fit_parameters[name] = bool(selected)
    for name, raw in (limits_meV or {}).items():
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise ValueError("tight-binding limits must contain lower and upper")
        bounds = [
            None if value is None else float(value)
            for value in raw
        ]
        if any(value is not None and not np.isfinite(value) for value in bounds):
            raise ValueError("tight-binding limits must be finite or None")
        if (
            bounds[0] is not None
            and bounds[1] is not None
            and bounds[0] >= bounds[1]
        ):
            raise ValueError("tight-binding lower limits must be below upper limits")
        if bounds == [None, None]:
            component.limits.pop(name, None)
        else:
            component.limits[name] = bounds
    for name, raw in (sharing or {}).items():
        if not isinstance(raw, Mapping):
            raise ValueError("tight-binding sharing entries must be mappings")
        mode = str(raw.get("mode", "global"))
        if mode not in {"global", "per_dataset", "grouped"}:
            raise ValueError(f"unsupported tight-binding sharing mode {mode!r}")
        groups = raw.get("groups", {})
        if not isinstance(groups, Mapping):
            raise ValueError("tight-binding sharing groups must be a mapping")
        component.sharing[name] = {
            "mode": mode,
            "groups": {
                str(dataset): str(group)
                for dataset, group in groups.items()
            },
        }
    reconcile_tight_binding_parameters(component)
    return resolve_tight_binding_builder(component)


def _install_hopping_generation(
    component: Any,
    generation: HoppingGeneration,
) -> None:
    active_identifiers = {
        item.identifier for item in _component_hopping_terms(component)
    }
    component.config["site_positions"] = sites_to_config(generation.sites)
    component.config["expanded_crystal_sites"] = [
        {
            "label": site.label,
            "position": list(site.position),
            "element": site.element,
            "rotation": (
                None
                if site.rotation is None
                else [list(row) for row in site.rotation]
            ),
        }
        for site in generation.sites
    ]
    component.config["spatial_orbits"] = orbits_to_config(generation.orbits)
    component.config["hopping_candidates"] = [
        item.to_dict() for item in generation.terms
    ]
    component.config["hopping_terms"] = [
        item.to_dict()
        for item in generation.terms
        if item.identifier in active_identifiers
    ]


def resolve_tight_binding_builder(component: Any) -> ElectronicModel:
    """Regenerate and install canonical ``model_data`` from builder records."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("the orbital builder requires a tight_binding component")
    reconcile_tight_binding_parameters(component)
    manifolds = _component_manifolds(component)
    model = build_orbital_electronic_model(
        component.config["crystal"],
        manifolds,
        _component_terms(component),
        hopping_orbits=_component_hopping_orbits(component),
        hopping_terms=_component_hopping_terms(component),
        periodic_axes=component.config.get("periodic_axes") or (0, 1, 2),
    )
    from .electronic_spin import lift_electronic_model_spin

    model = lift_electronic_model_spin(
        model,
        treatment=str(component.config.get("spin_treatment", "auto")),
        manifolds=manifolds,
        soc_terms=_component_soc_terms(component),
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
    before = _builder_state_snapshot(component)
    try:
        component.config["orbital_manifolds"] = [item.to_dict() for item in items]
        retained_soc = [
            term
            for term in _component_soc_terms(component)
            if term.manifold_label in {item.label for item in items}
        ]
        component.config["soc_terms"] = [
            term.to_dict() for term in retained_soc
        ]
        if generate_terms:
            generated = generate_onsite_terms(
                component.config["crystal"],
                items,
                previous=_component_terms(component),
            )
            component.config["onsite_terms"] = [
                item.to_dict() for item in generated
            ]
        cutoff = float(component.config.get("hopping_cutoff_angstrom", 0.0))
        if items and cutoff > 0.0:
            hopping = generate_hopping_terms(
                component.config["crystal"],
                items,
                cutoff,
                previous=_component_hopping_terms(component),
            )
            _install_hopping_generation(component, hopping)
        if items and component.config.get("onsite_terms"):
            resolve_tight_binding_builder(component)
        else:
            component.config["onsite_terms"] = []
            component.config["hopping_candidates"] = []
            component.config["hopping_terms"] = []
            component.config["spatial_orbits"] = []
            component.config["site_positions"] = []
            component.config["expanded_crystal_sites"] = []
            component.config["model_data"] = {}
            component.config["model_digest"] = ""
            reconcile_tight_binding_parameters(component)
    except Exception:
        _restore_builder_state(component, before)
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

    before = _builder_state_snapshot(component)
    try:
        terms = generate_onsite_terms(
            component.config["crystal"],
            _component_manifolds(component),
            previous=_component_terms(component),
        )
        component.config["onsite_terms"] = [item.to_dict() for item in terms]
        cutoff = float(component.config.get("hopping_cutoff_angstrom", 0.0))
        if cutoff > 0.0:
            generation = generate_hopping_terms(
                component.config["crystal"],
                _component_manifolds(component),
                cutoff,
                previous=_component_hopping_terms(component),
            )
            _install_hopping_generation(component, generation)
        if terms:
            resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise
    return terms


def regenerate_tight_binding_hopping_terms(
    component: Any,
    cutoff_angstrom: float | None = None,
) -> HoppingGeneration:
    """Regenerate hopping orbits and matrices, preserving stable coefficients."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("hopping generation requires a tight_binding component")
    cutoff = float(
        component.config.get("hopping_cutoff_angstrom", 0.0)
        if cutoff_angstrom is None
        else cutoff_angstrom
    )
    if cutoff <= 0.0:
        raise ValueError("hopping cutoff must be positive")
    before = _builder_state_snapshot(component)
    try:
        generation = generate_hopping_terms(
            component.config["crystal"],
            _component_manifolds(component),
            cutoff,
            previous=_component_hopping_terms(component),
        )
        component.config["hopping_cutoff_angstrom"] = cutoff
        _install_hopping_generation(component, generation)
        resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise
    return generation


def add_tight_binding_hopping_term(
    component: Any,
    identifier: str,
) -> HoppingInvariant:
    """Activate one generated hopping candidate in the resolved Hamiltonian."""

    key = str(identifier)
    active = list(_component_hopping_terms(component))
    for term in active:
        if term.identifier == key:
            return term
    candidate = next(
        (
            term
            for term in _component_hopping_candidates(component)
            if term.identifier == key
        ),
        None,
    )
    if candidate is None:
        raise KeyError(f"unknown hopping candidate {identifier!r}")
    before = _builder_state_snapshot(component)
    try:
        component.config["hopping_terms"] = [
            item.to_dict() for item in (*active, candidate)
        ]
        resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise
    return candidate


def remove_tight_binding_hopping_term(
    component: Any,
    identifier: str,
) -> None:
    """Remove one active hopping term while retaining it as a suggestion."""

    key = str(identifier)
    active = list(_component_hopping_terms(component))
    retained = [term for term in active if term.identifier != key]
    if len(retained) == len(active):
        raise KeyError(f"unknown active hopping term {identifier!r}")
    before = _builder_state_snapshot(component)
    try:
        removed = next(term for term in active if term.identifier == key)
        candidates = list(_component_hopping_candidates(component))
        candidates = [
            removed if term.identifier == key else term for term in candidates
        ]
        if not any(term.identifier == key for term in candidates):
            candidates.append(removed)
        component.config["hopping_candidates"] = [
            item.to_dict() for item in candidates
        ]
        component.config["hopping_terms"] = [
            item.to_dict() for item in retained
        ]
        resolve_tight_binding_builder(component)
    except Exception:
        _restore_builder_state(component, before)
        raise


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
        before = _builder_state_snapshot(component)
        try:
            _adopt_term_parameter_state(component, (updated,))
            component.config["onsite_terms"] = [item.to_dict() for item in terms]
            resolve_tight_binding_builder(component)
        except Exception:
            _restore_builder_state(component, before)
            raise
        return updated
    raise KeyError(f"unknown onsite term {identifier!r}")


def set_tight_binding_hopping_term(
    component: Any,
    identifier: str,
    *,
    value: float | None = None,
    energy_unit: str | None = None,
    lower: float | None | Literal["unchanged"] = "unchanged",
    upper: float | None | Literal["unchanged"] = "unchanged",
    fit: bool | None = None,
) -> HoppingInvariant:
    """Update one hopping coefficient, bounds, or future fit-selection flag."""

    terms = list(_component_hopping_terms(component))
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
        updated = HoppingInvariant(
            identifier=term.identifier,
            label=term.label,
            orbit_label=term.orbit_label,
            distance_angstrom=term.distance_angstrom,
            representative_bond=term.representative_bond,
            basis_i=term.basis_i,
            basis_j=term.basis_j,
            matrix=term.matrix,
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
        before = _builder_state_snapshot(component)
        try:
            _adopt_term_parameter_state(component, (updated,))
            candidates = [
                updated if item.identifier == updated.identifier else item
                for item in _component_hopping_candidates(component)
            ]
            if not any(
                item.identifier == updated.identifier for item in candidates
            ):
                candidates.append(updated)
            component.config["hopping_candidates"] = [
                item.to_dict() for item in candidates
            ]
            component.config["hopping_terms"] = [
                item.to_dict() for item in terms
            ]
            resolve_tight_binding_builder(component)
        except Exception:
            _restore_builder_state(component, before)
            raise
        return updated
    raise KeyError(f"unknown hopping term {identifier!r}")


def configure_tight_binding_builder(
    component: Any,
    *,
    manifolds: Sequence[OrbitalManifold | Mapping[str, Any]],
    onsite_terms: Sequence[OnsiteInvariant | Mapping[str, Any]] = (),
    hopping_cutoff_angstrom: float | None = None,
    hopping_terms: Sequence[HoppingInvariant | Mapping[str, Any]] = (),
    spin_treatment: str = "auto",
    soc_terms: Sequence[Mapping[str, Any] | Any] = (),
) -> ElectronicModel:
    """Rebuild a component from editable high-level builder configuration."""

    before = _builder_state_snapshot(component)
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
            _adopt_term_parameter_state(component, merged)
        if hopping_cutoff_angstrom is not None:
            generation = regenerate_tight_binding_hopping_terms(
                component,
                hopping_cutoff_angstrom,
            )
            if hopping_terms:
                parsed_hoppings = tuple(
                    item
                    if isinstance(item, HoppingInvariant)
                    else HoppingInvariant.from_dict(item)
                    for item in hopping_terms
                )
                supplied_hoppings = {
                    item.identifier: item for item in parsed_hoppings
                }
                generated_hoppings = list(generation.terms)
                generated_identifiers = {
                    item.identifier for item in generated_hoppings
                }
                if not set(supplied_hoppings).issubset(generated_identifiers):
                    raise ValueError(
                        "stored active hopping terms do not match the candidates "
                        "regenerated from the crystal and orbital manifolds"
                    )
                merged_hoppings = []
                for term in generated_hoppings:
                    source = supplied_hoppings.get(term.identifier)
                    if source is None:
                        continue
                    if not np.allclose(source.matrix, term.matrix, atol=1e-9):
                        raise ValueError(
                            f"stored matrix for hopping term "
                            f"{term.identifier!r} changed"
                        )
                    merged_hoppings.append(
                        HoppingInvariant(
                            identifier=term.identifier,
                            label=term.label,
                            orbit_label=term.orbit_label,
                            distance_angstrom=term.distance_angstrom,
                            representative_bond=term.representative_bond,
                            basis_i=term.basis_i,
                            basis_j=term.basis_j,
                            matrix=term.matrix,
                            value_meV=source.value_meV,
                            bounds_meV=source.bounds_meV,
                            fit=source.fit,
                            source=term.source,
                        )
                    )
                component.config["hopping_terms"] = [
                    item.to_dict() for item in merged_hoppings
                ]
                _adopt_term_parameter_state(component, merged_hoppings)
                candidate_values = {
                    item.identifier: item for item in merged_hoppings
                }
                component.config["hopping_candidates"] = [
                    candidate_values.get(item.identifier, item).to_dict()
                    for item in generated_hoppings
                ]
        return configure_tight_binding_spin(
            component,
            treatment=spin_treatment,
            soc_terms=soc_terms,
        )
    except Exception:
        _restore_builder_state(component, before)
        raise
