from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..fitting import reciprocal_basis_from_lattice_parameters
from ..mdhisto import MDHistoAxis, MDHistoData, mdhisto_measured_bins
from .core import AnalysisContext


def bin_edges(axis: MDHistoAxis, size: int) -> np.ndarray:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return values.copy()
    if values.size != size:
        raise ValueError(f"axis {axis.name!r} is incompatible with dimension size {size}")
    if size == 1:
        width = float(axis.metadata.get("bin_width", 1.0))
        return np.array([values[0] - width / 2.0, values[0] + width / 2.0])
    edges = np.empty(size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def bin_widths(axis: MDHistoAxis, size: int) -> np.ndarray:
    widths = np.diff(bin_edges(axis, size))
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError(f"axis {axis.name!r} must have increasing finite bin edges")
    return widths


def physical_axis_vectors(data: MDHistoData) -> np.ndarray:
    vectors = np.zeros((len(data.axes), 4), dtype=float)
    stored = data.metadata.get("rebin", {}).get("vectors") if isinstance(data.metadata.get("rebin"), dict) else None
    for index, axis in enumerate(data.axes):
        if isinstance(stored, (list, tuple)) and index < len(stored):
            candidate = np.asarray(stored[index], dtype=float).reshape(-1)
            if candidate.size in (3, 4):
                vectors[index, : candidate.size] = candidate
                continue
        projection = _projection_vector(axis.name)
        if projection is not None:
            vectors[index, :3] = projection
            continue
        role = axis.role
        component = {"h": 0, "k": 1, "l": 2, "energy_transfer": 3}.get(role)
        if component is not None:
            vectors[index, component] = 1.0
    return vectors


def physical_coordinate_arrays(data: MDHistoData) -> dict[str, np.ndarray]:
    mesh = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    hkle = np.zeros((*data.shape, 4), dtype=float)
    for values, vector in zip(mesh, physical_axis_vectors(data), strict=True):
        hkle += values[..., None] * vector
    result = {"H": hkle[..., 0], "K": hkle[..., 1], "L": hkle[..., 2]}
    for axis, values in zip(data.axes, mesh, strict=True):
        if axis.role == "q_modulus":
            result["q_modulus"] = values
    if np.any(physical_axis_vectors(data)[:, 3]):
        result["E"] = hkle[..., 3]
    return result


def rlu_to_q_matrix(
    data_metadata: dict[str, Any], context: AnalysisContext | None = None
) -> np.ndarray:
    sources = [data_metadata]
    oriented = data_metadata.get("oriented_lattice")
    if isinstance(oriented, dict):
        sources.append(oriented)
    for source in sources:
        for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
            if key in source:
                matrix = np.asarray(source[key], dtype=float)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise ValueError(f"{key} must be a finite 3x3 matrix")
                includes = key == "rlu_to_inv_angstrom_matrix" or bool(
                    source.get(f"{key}_includes_2pi", source.get("include_2pi", False))
                )
                return matrix if includes else 2.0 * np.pi * matrix
    lattice = data_metadata.get("lattice_parameters")
    if not isinstance(lattice, dict) and context is not None:
        lattice = context.lattice_parameters
    if not isinstance(lattice, dict):
        raise ValueError("an attached lattice or UB matrix is required")
    return reciprocal_basis_from_lattice_parameters(
        a=float(lattice["a"]), b=float(lattice["b"]), c=float(lattice["c"]),
        alpha=float(lattice.get("alpha", 90.0)), beta=float(lattice.get("beta", 90.0)),
        gamma=float(lattice.get("gamma", 90.0)), include_2pi=True,
    )


def q_bin_volume(data: MDHistoData, context: AnalysisContext | None = None) -> np.ndarray:
    vectors = physical_axis_vectors(data)
    momentum_dims = [i for i, axis in enumerate(data.axes) if axis.kind == "momentum"]
    if len(momentum_dims) != 3:
        raise ValueError("Q-volume integration requires exactly three momentum axes")
    hkl_vectors = vectors[momentum_dims, :3]
    q_vectors = hkl_vectors @ rlu_to_q_matrix(data.metadata, context).T
    jacobian = abs(float(np.linalg.det(q_vectors)))
    if not np.isfinite(jacobian) or jacobian <= 1e-14:
        raise ValueError("momentum axes are not linearly independent")
    volume = np.ones(data.shape, dtype=float) * jacobian
    for dim in momentum_dims:
        shape = [1] * data.signal.ndim
        shape[dim] = data.shape[dim]
        volume *= bin_widths(data.axes[dim], data.shape[dim]).reshape(shape)
    return volume


def energy_bin_widths(data: MDHistoData) -> np.ndarray:
    dimensions = [i for i, axis in enumerate(data.axes) if axis.kind == "energy" or axis.role == "energy_transfer"]
    if len(dimensions) != 1:
        raise ValueError("expected exactly one energy-transfer axis")
    dim = dimensions[0]
    shape = [1] * data.signal.ndim
    shape[dim] = data.shape[dim]
    return bin_widths(data.axes[dim], data.shape[dim]).reshape(shape)


def measured_mask(data: MDHistoData) -> np.ndarray:
    return mdhisto_measured_bins(data) & np.isfinite(data.signal) & np.isfinite(data.errors)


def q_modulus_for_spectral(
    data: MDHistoData, context: AnalysisContext | None = None
) -> np.ndarray:
    """Return a Q array with singleton non-momentum dimensions for broadcasting."""

    momentum_dims = [i for i, axis in enumerate(data.axes) if axis.kind == "momentum"]
    if not momentum_dims:
        raise ValueError("a momentum axis is required for form-factor correction")
    shape = [1] * data.signal.ndim
    q_axes = [dim for dim in momentum_dims if data.axes[dim].role == "q_modulus"]
    if q_axes:
        if len(momentum_dims) != 1:
            raise ValueError("|Q| cannot be mixed with independent HKL momentum axes")
        dim = q_axes[0]
        shape[dim] = data.shape[dim]
        return data.axes[dim].centers.reshape(shape)
    mesh = np.meshgrid(*(data.axes[dim].centers for dim in momentum_dims), indexing="ij")
    hkl = np.zeros((*mesh[0].shape, 3), dtype=float)
    vectors = physical_axis_vectors(data)
    for values, dim in zip(mesh, momentum_dims, strict=True):
        hkl += values[..., None] * vectors[dim, :3]
    q = np.linalg.norm(hkl @ rlu_to_q_matrix(data.metadata, context).T, axis=-1)
    for dim in momentum_dims:
        shape[dim] = data.shape[dim]
    return q.reshape(shape)


def signal_semantics(data: MDHistoData) -> str:
    semantics = str(data.metadata.get("signal_semantics", "unknown"))
    if semantics not in {"density", "bin_integral", "unknown"}:
        raise ValueError(f"unknown signal_semantics {semantics!r}")
    return semantics


def _projection_vector(name: str) -> np.ndarray | None:
    match = re.search(r"\[([^\]]+)\]", str(name))
    if match is None:
        return None
    pieces = [piece.strip().replace(" ", "") for piece in match.group(1).split(",")]
    if len(pieces) != 3:
        return None
    values = []
    for piece in pieces:
        if piece in {"", "0"}:
            values.append(0.0)
        else:
            coefficient = re.sub(r"[HKL]", "", piece)
            values.append(float({"": "1", "+": "1", "-": "-1"}.get(coefficient, coefficient)))
    return np.asarray(values, dtype=float)
