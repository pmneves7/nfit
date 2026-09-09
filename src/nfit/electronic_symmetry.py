"""Orbital symmetry representations shared by electronic model layers.

This module intentionally depends only on the harmonic conventions.  Builder
records and spin lifting can therefore use the same representation machinery
without importing one another.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonics import real_harmonic_transform

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class OrbitalSymmetryError(ValueError):
    """Raised when a requested orbital subspace is not closed under symmetry."""


def _manifold_value(manifold: Any, name: str) -> Any:
    if isinstance(manifold, Mapping):
        if name == "symmetry_mode":
            return manifold.get(name, "automatic")
        if name == "local_frame":
            return manifold.get(name, np.eye(3))
        return manifold[name]
    return getattr(manifold, name)


def _harmonic_transform(manifold: Any) -> ComplexArray:
    values = np.asarray(_manifold_value(manifold, "harmonic_transform"))
    if isinstance(manifold, Mapping) and values.ndim == 3 and values.shape[-1] == 2:
        values = values[..., 0] + 1.0j * values[..., 1]
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
        transform = real_harmonic_transform(int(l))
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
    manifold: Any,
    rotation_cartesian: ArrayLike,
    *,
    closure_tolerance: float = 1e-8,
) -> ComplexArray:
    """Return one manifold's symmetry matrix, rejecting non-closed subspaces."""

    symmetry_mode = str(_manifold_value(manifold, "symmetry_mode"))
    label = str(_manifold_value(manifold, "label"))
    if symmetry_mode != "automatic":
        raise OrbitalSymmetryError(
            f"manifold {label!r} has no automatic symmetry representation"
        )
    basis_kind = str(_manifold_value(manifold, "basis_kind"))
    if basis_kind == "effective_scalar":
        return np.ones((1, 1), dtype=np.complex128)
    frame = _frame(_manifold_value(manifold, "local_frame"))
    rotation = np.asarray(rotation_cartesian, dtype=float)
    local_rotation = frame.T @ rotation @ frame
    angular_momentum = int(_manifold_value(manifold, "l"))
    full = spherical_harmonic_representation(
        angular_momentum,
        local_rotation,
        basis_kind=basis_kind,
    )
    transform = _harmonic_transform(manifold)
    image = full @ transform
    projected = transform @ (transform.conj().T @ image)
    residual = float(np.linalg.norm(image - projected))
    if residual > closure_tolerance * max(1.0, float(np.linalg.norm(image))):
        raise OrbitalSymmetryError(
            f"orbital manifold {label!r} is not closed under the site "
            f"symmetry (closure residual {residual:.3g}); add the missing "
            "orbitals or set symmetry_mode='none'"
        )
    result = transform.conj().T @ image
    left, _singular, right = np.linalg.svd(result)
    return left @ right
