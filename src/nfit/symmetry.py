"""Reciprocal-space symmetry specifications used by the rebinners.

The public operations are expressed on direct fractional coordinates, following
the Jones-faithful notation used by CIF and Mantid.  They are converted to the
dual action on Miller indices before any data are transformed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .fitting import direct_basis_from_lattice_parameters, reciprocal_basis_from_lattice_parameters

SymmetryMode = Literal["none", "space_group", "point_group", "operations", "generators"]


@dataclass(frozen=True)
class ReciprocalSymmetryOperation:
    """One linear operation on column-vector HKL coordinates."""

    matrix_hkl: np.ndarray
    label: str


@dataclass(frozen=True)
class SymmetrySpec:
    """User-facing rebin symmetry specification."""

    mode: SymmetryMode = "none"
    expression: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and bool(self.expression.strip())


def symmetry_spec_from_config(config: dict[str, Any] | None) -> SymmetrySpec:
    """Read and sanitize a persisted rebin symmetry configuration."""

    payload = {} if not isinstance(config, dict) else config
    mode = str(payload.get("mode", "none"))
    if mode not in {"none", "space_group", "point_group", "operations", "generators"}:
        mode = "none"
    return SymmetrySpec(mode=mode, expression=str(payload.get("expression", "")))


def symmetry_config(spec: SymmetrySpec) -> dict[str, str]:
    return {"mode": spec.mode, "expression": spec.expression}


def resolve_symmetry(
    spec: SymmetrySpec,
    *,
    lattice_parameters: dict[str, Any] | None = None,
) -> tuple[ReciprocalSymmetryOperation, ...]:
    """Resolve a symmetry specification to unique HKL operations.

    Space-group translations are intentionally discarded: reciprocal-space
    intensity coordinates transform only under the associated point group.
    """

    if not spec.enabled:
        return (_identity_operation(),)
    if spec.mode == "space_group":
        return _space_group_operations(spec.expression)
    if spec.mode == "point_group":
        return _point_group_operations(spec.expression)
    if spec.mode == "operations":
        return _explicit_operations(spec.expression)
    if spec.mode == "generators":
        return _generator_operations(spec.expression, lattice_parameters)
    raise ValueError(f"unsupported symmetry mode {spec.mode!r}")


def symmetry_preview(
    spec: SymmetrySpec,
    *,
    lattice_parameters: dict[str, Any] | None = None,
) -> str:
    operations = resolve_symmetry(spec, lattice_parameters=lattice_parameters)
    labels = ", ".join(operation.label for operation in operations[:4])
    suffix = "" if len(operations) <= 4 else ", ..."
    return f"{len(operations)} operation{'s' if len(operations) != 1 else ''}: {labels}{suffix}"


def transform_hkl(hkl: np.ndarray, operations: Iterable[ReciprocalSymmetryOperation]) -> Iterable[np.ndarray]:
    """Yield transformed ``(n, 3)`` HKL arrays without copying all images."""

    values = np.asarray(hkl, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("HKL coordinates must have shape (n, 3)")
    for operation in operations:
        yield values @ np.asarray(operation.matrix_hkl, dtype=float).T


def _identity_operation() -> ReciprocalSymmetryOperation:
    return ReciprocalSymmetryOperation(np.eye(3), "x,y,z")


def _require_gemmi():
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("rebin symmetry requires the optional 'gemmi' package") from exc
    return gemmi


def _space_group_operations(expression: str) -> tuple[ReciprocalSymmetryOperation, ...]:
    gemmi = _require_gemmi()
    name = expression.strip()
    group = None
    if name.isdigit():
        try:
            group = gemmi.get_spacegroup_reference_setting(int(name))
        except (RuntimeError, ValueError):
            pass
    else:
        group = gemmi.find_spacegroup_by_name(name)
    if group is None:
        raise ValueError(f"unknown space group {expression!r}")
    den = float(gemmi.Op.DEN)
    operations = []
    for operation in group.operations():
        direct = np.asarray(operation.rot, dtype=float) / den
        reciprocal = np.linalg.inv(direct).T
        operations.append(ReciprocalSymmetryOperation(reciprocal, operation.triplet()))
    return _unique_operations(operations)


_POINT_GROUP_ALIASES = {
    "-3m1": "P -3 1 m",
    "-31m": "P -3 m 1",
    "3m1": "P 3 m 1",
    "31m": "P 3 1 m",
    "312": "P 3 1 2",
    "321": "P 3 2 1",
    "-6m2": "P -6 m 2",
    "-62m": "P -6 2 m",
}


def _point_group_operations(expression: str) -> tuple[ReciprocalSymmetryOperation, ...]:
    gemmi = _require_gemmi()
    normalized = expression.replace(" ", "").lower()
    alias = _POINT_GROUP_ALIASES.get(normalized)
    if alias is not None:
        return _space_group_operations(alias)
    candidates = [group for group in gemmi.spacegroup_table() if group.point_group_hm().replace(" ", "").lower() == normalized]
    if not candidates:
        raise ValueError(f"unknown crystallographic point group {expression!r}")
    # The Gemmi table is ordered by International Tables number; the first
    # entry supplies the conventional orientation for a bare point-group name.
    return _space_group_operations(candidates[0].hm)


def _explicit_operations(expression: str) -> tuple[ReciprocalSymmetryOperation, ...]:
    gemmi = _require_gemmi()
    pieces = [piece.strip() for piece in expression.split(";") if piece.strip()]
    if not pieces:
        raise ValueError("provide one or more semicolon-separated symmetry operations")
    operations = []
    for piece in pieces:
        try:
            operation = gemmi.Op(piece)
        except RuntimeError as exc:
            raise ValueError(f"invalid symmetry operation {piece!r}") from exc
        direct = np.asarray(operation.rot, dtype=float) / float(gemmi.Op.DEN)
        operations.append(ReciprocalSymmetryOperation(np.linalg.inv(direct).T, operation.triplet()))
    return _unique_operations(operations)


def _generator_operations(
    expression: str,
    lattice_parameters: dict[str, Any] | None,
) -> tuple[ReciprocalSymmetryOperation, ...]:
    if not isinstance(lattice_parameters, dict):
        raise ValueError("geometric symmetry generators require lattice parameters")
    generators = []
    for text in [part.strip() for part in expression.split(";") if part.strip()]:
        rotation = _parse_rotation_generator(text, lattice_parameters)
        if rotation is None:
            rotation = _parse_mirror_generator(text, lattice_parameters)
        if rotation is None:
            raise ValueError(
                "generators use rotate(order=N, axis=[u,v,w]) or mirror(plane=(h,k,l))"
            )
        generators.append(rotation)
    if not generators:
        raise ValueError("provide one or more geometric generators")
    matrices = [np.eye(3)]
    pending = [np.eye(3)]
    while pending:
        current = pending.pop()
        for generator in generators:
            candidate = generator @ current
            if not any(np.allclose(candidate, known, atol=1e-10) for known in matrices):
                matrices.append(candidate)
                pending.append(candidate)
                if len(matrices) > 96:
                    raise ValueError("generator closure exceeded 96 operations; it is not a finite point group")
    return _unique_operations(
        ReciprocalSymmetryOperation(matrix, _matrix_label(matrix)) for matrix in matrices
    )


def _parse_rotation_generator(text: str, lattice: dict[str, Any]) -> np.ndarray | None:
    match = re.fullmatch(
        r"rotate\(\s*order\s*=\s*(\d+)\s*,\s*axis\s*=\s*\[([^]]+)\]\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    order = int(match.group(1))
    if order < 2:
        raise ValueError("rotation order must be at least two")
    axis = _triple(match.group(2), "rotation axis")
    direct = _direct_basis(lattice)
    cartesian_axis = direct @ axis
    norm = float(np.linalg.norm(cartesian_axis))
    if norm == 0.0:
        raise ValueError("rotation axis must be nonzero")
    unit = cartesian_axis / norm
    cross = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    angle = 2.0 * np.pi / order
    cartesian = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    reciprocal = _reciprocal_basis(lattice)
    return np.linalg.inv(reciprocal) @ cartesian @ reciprocal


def _parse_mirror_generator(text: str, lattice: dict[str, Any]) -> np.ndarray | None:
    match = re.fullmatch(
        r"mirror\(\s*plane\s*=\s*\(([^)]+)\)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    plane = _triple(match.group(1), "mirror plane")
    reciprocal = _reciprocal_basis(lattice)
    normal = reciprocal @ plane
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("mirror plane must be nonzero")
    unit = normal / norm
    cartesian = np.eye(3) - 2.0 * np.outer(unit, unit)
    return np.linalg.inv(reciprocal) @ cartesian @ reciprocal


def _triple(text: str, label: str) -> np.ndarray:
    try:
        values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{label} must contain three numeric values") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain three finite numeric values")
    return values


def _direct_basis(lattice: dict[str, Any]) -> np.ndarray:
    return direct_basis_from_lattice_parameters(**_lattice_kwargs(lattice))


def _reciprocal_basis(lattice: dict[str, Any]) -> np.ndarray:
    return reciprocal_basis_from_lattice_parameters(**_lattice_kwargs(lattice))


def _lattice_kwargs(lattice: dict[str, Any]) -> dict[str, float]:
    try:
        return {name: float(lattice[name]) for name in ("a", "b", "c", "alpha", "beta", "gamma")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("lattice parameters require a, b, c, alpha, beta, and gamma") from exc


def _unique_operations(operations: Iterable[ReciprocalSymmetryOperation]) -> tuple[ReciprocalSymmetryOperation, ...]:
    unique: list[ReciprocalSymmetryOperation] = []
    for operation in operations:
        matrix = np.asarray(operation.matrix_hkl, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("symmetry operation is not a finite 3x3 matrix")
        if not any(np.allclose(matrix, other.matrix_hkl, atol=1e-10) for other in unique):
            unique.append(ReciprocalSymmetryOperation(matrix, operation.label))
    unique.sort(key=lambda operation: (not np.allclose(operation.matrix_hkl, np.eye(3)), operation.label))
    return tuple(unique)


def _matrix_label(matrix: np.ndarray) -> str:
    return "[" + "; ".join(
        ", ".join(f"{value:.6g}" for value in row) for row in matrix
    ) + "]"
