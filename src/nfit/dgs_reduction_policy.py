"""Numerical conventions shared by direct-geometry event importers.

The Mantid defaults reproduce its discrete event histogram and independent-copy
variance conventions. Alternative conventions are explicit scientific choices.
This service has no importer or GUI dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_MONITOR_VARIANCE_POLICY = "mantid"
MONITOR_VARIANCE_POLICIES = (
    ("mantid", "Mantid monitor variance"),
    ("stable", "Stable monitor variance"),
)
DEFAULT_EVENT_PRECISION_POLICY = "mantid"
EVENT_PRECISION_POLICIES = (
    ("mantid", "Mantid event precision"),
    ("high_precision", "High event precision"),
)
DEFAULT_SYMMETRY_VARIANCE_POLICY = "independent_copies"
SYMMETRY_VARIANCE_POLICIES = (
    ("independent_copies", "Independent symmetry copies (Mantid)"),
    ("within_bin_covariance", "Correlated copies within each final bin"),
)
DGS_REDUCTION_POLICY_VERSION = 2

# Mantid v6.16 PhysicalConstants (SI units). Keep these coefficients distinct:
# ConvertToMD uses NeutronEToKSqr, while MDNorm constructs energyToK in the
# stated multiplication order. Cancelling reciprocals can change final ulps.
MANTID_PLANCK_J_S = 6.62606896e-34
MANTID_NEUTRON_MASS_KG = 1.674927211e-27
MANTID_MEV_J = 1.602176487e-22
_HBAR = MANTID_PLANCK_J_S / (2 * math.pi)
ENERGY_TO_K2 = 1e20 * _HBAR * _HBAR / (2 * MANTID_NEUTRON_MASS_KG * MANTID_MEV_J)
ENERGY_TO_K = (8 * math.pi * math.pi * MANTID_NEUTRON_MASS_KG * MANTID_MEV_J * 1e-20
               / (MANTID_PLANCK_J_S * MANTID_PLANCK_J_S))


def _validated(value, name, choices):
    if value not in {key for key, _label in choices}:
        allowed = ", ".join(repr(key) for key, _label in choices)
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def validated_monitor_variance_policy(value: str) -> str:
    return _validated(value, "monitor_variance_policy", MONITOR_VARIANCE_POLICIES)


def validated_event_precision_policy(value: str) -> str:
    return _validated(value, "event_precision_policy", EVENT_PRECISION_POLICIES)


def validated_symmetry_variance_policy(value: str) -> str:
    return _validated(value, "symmetry_variance_policy", SYMMETRY_VARIANCE_POLICIES)


def resolved_dgs_reduction_policies(config: Mapping[str, Any], *, include_monitor=True):
    """Resolve effective defaults for persistence, cache signatures, and scripts."""
    policies = {
        "event_precision_policy": validated_event_precision_policy(
            config.get("event_precision_policy", DEFAULT_EVENT_PRECISION_POLICY)
        ),
        "symmetry_variance_policy": validated_symmetry_variance_policy(
            config.get("symmetry_variance_policy", DEFAULT_SYMMETRY_VARIANCE_POLICY)
        ),
    }
    if include_monitor:
        policies["monitor_variance_policy"] = validated_monitor_variance_policy(
            config.get("monitor_variance_policy", DEFAULT_MONITOR_VARIANCE_POLICY)
        )
    return policies


def dgs_histogram_edges(edges, policy=DEFAULT_EVENT_PRECISION_POLICY):
    """Return physical edges with Mantid's float32 dimension arithmetic.

    Nonuniform grids have no BinMD equivalent and retain their explicit edges.
    """
    policy = validated_event_precision_policy(policy)
    result = tuple(np.asarray(edge, dtype=float) for edge in edges)
    if policy == "high_precision":
        return result
    result = tuple(
        _mantid_dimension_edges(edge)
        if _uniform(edge) else edge
        for edge in result
    )
    if any(np.any(np.diff(edge) <= 0) for edge in result):
        raise ValueError("Mantid event precision cannot distinguish the requested bin edges; use high_precision")
    return result


def _mantid_dimension_edges(edge):
    lower, upper = map(np.float32, _mantid_limits(edge))
    width = (upper - lower) / np.float32(len(edge) - 1)
    return (np.arange(len(edge), dtype=np.float32) * width + lower).astype(float)


def _mantid_extent(value):
    # MDNorm::getBinParameters streams OutputExtents with default precision
    # before passing the string to BinMD::setPropertyValue (six significant
    # digits). Unlike coord_t conversion, this rounds the double extent first.
    return float(format(value, ".6g"))


def _mantid_binning_extents(edges):
    return tuple(
        np.linspace(*_mantid_limits(edge), len(edge))
        for edge in edges
    )


def _mantid_limits(edge):
    lower, upper = float(edge[0]), float(edge[-1])
    if len(edge) > 2:
        # MDNorm stepped recipes first store dimMin in coord_t; their upper
        # extent is dimMin + nsteps*step in double before string serialization.
        width = (upper - lower) / (len(edge) - 1)
        lower = float(np.float32(lower))
        upper = lower + (len(edge) - 1) * width
    return _mantid_extent(lower), _mantid_extent(upper)


def _uniform(edge):
    widths = np.diff(edge)
    return bool(np.allclose(widths, widths[0], rtol=1e-10, atol=1e-13))


def dgs_uses_mantid_trajectory_grid(edges, policy=DEFAULT_EVENT_PRECISION_POLICY):
    """Whether requested axes have a uniform Mantid normalization equivalent."""
    return validated_event_precision_policy(policy) == "mantid" and all(
        _uniform(np.asarray(edge)) for edge in edges
    )


def dgs_trajectory_bin_indices(coordinates, edges, shape):
    """Match MDHistoWorkspace indexing for rounded normalization positions."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    flat = np.zeros(len(coordinates), dtype=np.int64)
    valid = np.ones(len(coordinates), dtype=bool)
    for dim, edge in enumerate(edges):
        origin = np.float32(edge[0])
        width = np.float32(np.float32(edge[1]) - origin)
        relative = coordinates[:, dim] - origin
        fractional = relative / width
        okay = np.isfinite(fractional) & (relative >= 0) & (fractional < shape[dim])
        indices = np.where(okay, fractional, 0).astype(np.int64)
        valid &= okay
        flat = flat * shape[dim] + indices
    return np.where(valid, flat, -1)


def _float32_product(left, right):
    """Mantid Matrix<float> sums products sequentially, without BLAS."""
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for index in range(left.shape[1]):
        result += left[:, index, None] * right[index, None, :]
    return result


def _mantid_float32_inverse(matrix):
    """Match Matrix<float>::Invert's float LU and double back substitution.

    Only the small affine basis uses this compatibility arithmetic. Event
    batches remain vectorized. A LAPACK float inverse can differ by an ulp at
    bin boundaries, so it cannot substitute for this specified convention.
    """
    lu = np.array(matrix, dtype=np.float32, copy=True)
    size = len(lu)
    scales = 1.0 / np.max(np.abs(lu).astype(float), axis=1)
    pivots = []
    for column in range(size):
        for row in range(column):
            value = float(lu[row, column])
            for k in range(row):
                value -= float(lu[row, k] * lu[k, column])
            lu[row, column] = value
        largest = -1.0
        pivot = column
        for row in range(column, size):
            value = float(lu[row, column])
            for k in range(column):
                value -= float(lu[row, k] * lu[k, column])
            lu[row, column] = value
            candidate = scales[row] * abs(value)
            if candidate >= largest:
                largest, pivot = candidate, row
        if pivot != column:
            lu[[pivot, column]] = lu[[column, pivot]]
            scales[pivot] = scales[column]
        pivots.append(pivot)
        if lu[column, column] == 0:
            lu[column, column] = 1e-14
        if column != size - 1:
            lu[column + 1:, column] *= np.float32(1.0 / float(lu[column, column]))
    inverse = np.empty_like(lu)
    for column in range(size):
        values = np.eye(size, dtype=float)[:, column].copy()
        first_nonzero = -1
        for row, pivot in enumerate(pivots):
            value = values[pivot]
            values[pivot] = values[row]
            if first_nonzero != -1:
                for k in range(first_nonzero, row):
                    value -= float(lu[row, k]) * values[k]
            elif value != 0:
                first_nonzero = row
            values[row] = value
        for row in range(size - 1, -1, -1):
            value = values[row]
            for k in range(row + 1, size):
                value -= float(lu[row, k]) * values[k]
            values[row] = value / float(lu[row, row])
        inverse[:, column] = values
    return inverse


def _mantid_affine(basis_columns, edges):
    # MDNorm serializes each basis component using stringstream's default six
    # significant digits before BinMD parses the basis into coord_t (float32).
    basis = np.asarray([[float(format(value, ".6g")) if abs(value) > 1e-10 else 0. for value in row]
                        for row in basis_columns], dtype=np.float32)
    origin = np.zeros(len(edges), dtype=np.float32)
    for index, edge in enumerate(edges):
        origin += (basis[:, index].astype(float) * float(np.float32(edge[0]))).astype(np.float32)
    gram = _float32_product(basis.T, basis)
    inverse = _float32_product(_mantid_float32_inverse(gram), basis.T)
    offset = _float32_product(inverse, origin[:, None])[:, 0]
    # SlicingAlgorithm keeps unnormalized basis vectors but forms two double
    # scales from their float32 norm, then divides their float32 VMD copies.
    # Cancelling those scales algebraically changes boundary rounding.
    transform_scaling = 1.0 / np.sqrt(np.diag(gram)).astype(float)
    binning_scaling = np.asarray([
        (len(edge) - 1) / ((edge[-1] - edge[0]) / scale)
        for edge, scale in zip(edges, transform_scaling, strict=True)
    ])
    scaling = binning_scaling.astype(np.float32) / transform_scaling.astype(np.float32)
    return inverse * scaling[:, None], -offset * scaling


def prepare_dgs_event_projector(ub, basis, operation, edges,
                                policy=DEFAULT_EVENT_PRECISION_POLICY):
    """Prepare one scientific projection and reuse it across all event batches.

    The returned callable accepts ``(q_sample, energy)`` and returns coordinates
    and accumulator edges. Preparation depends on UB, axes, symmetry operation,
    requested grid and precision policy, never on instrument geometry.
    """
    policy = validated_event_precision_policy(policy)
    basis = np.asarray(basis, dtype=float)
    operation = np.asarray(operation, dtype=float)
    if policy == "high_precision":
        hkl_transform = np.linalg.inv(2 * np.pi * np.asarray(ub)).T
        inverse_basis = np.linalg.inv(basis)
        def project(q_sample, energy):
            hkl = np.asarray(q_sample) @ hkl_transform
            return np.column_stack((hkl @ operation.T, energy)) @ inverse_basis, tuple(edges)
        return project
    columns = np.zeros((4, 4))
    columns[:3, :3] = 2 * np.pi * np.asarray(ub) @ np.linalg.inv(operation) @ basis[:3, :3].T
    columns[3, 3] = 1
    if not all(_uniform(np.asarray(edge)) for edge in edges):
        # Explicit nonuniform grids retain their boundaries; round the physical
        # projection using the same affine convention without uniformizing it.
        matrix, offset = _mantid_affine(columns, tuple(np.array([0., 1.]) for _ in edges))
        accumulation_edges = tuple(edges)
    else:
        matrix, offset = _mantid_affine(columns, _mantid_binning_extents(edges))
        accumulation_edges = tuple(np.arange(len(edge), dtype=float) for edge in edges)
    def project(q_sample, energy):
        inputs = np.column_stack((q_sample, energy)).astype(np.float32)
        coordinates = np.zeros_like(inputs)
        for index in range(4):
            coordinates += inputs[:, index, None] * matrix[None, :, index]
        coordinates += offset
        for index, edge in enumerate(accumulation_edges):
            coordinates[coordinates[:, index] >= edge[-1], index] = np.nan
        return coordinates, accumulation_edges
    return project


def dgs_event_coordinates(q_sample, energy, ub, basis, operation, edges,
                          policy=DEFAULT_EVENT_PRECISION_POLICY):
    """Project QSample events onto physical or Mantid fractional-bin axes.

    Mantid uses float32 affine arithmetic, six-significant-digit MDNorm basis
    strings and an exclusive upper limit. Returned integer edges prevent a
    second physical-coordinate transform in the event accumulator. For repeated
    batches use :func:`prepare_dgs_event_projector` to prepare the transform once.
    """
    return prepare_dgs_event_projector(ub, basis, operation, edges, policy)(q_sample, energy)


def dgs_powder_coordinates(q_modulus, energy, edges, policy=DEFAULT_EVENT_PRECISION_POLICY):
    """Apply the event precision convention to an aligned |Q|, energy grid."""
    policy = validated_event_precision_policy(policy)
    coordinates = np.column_stack((q_modulus, energy))
    if policy == "high_precision":
        return coordinates, tuple(edges)
    coordinates = coordinates.astype(np.float32)
    if all(_uniform(np.asarray(edge)) for edge in edges):
        for index, edge in enumerate(_mantid_binning_extents(edges)):
            width = (np.float32(edge[-1]) - np.float32(edge[0])) / np.float32(len(edge) - 1)
            scale = np.float32(1.) / width
            coordinates[:, index] = (coordinates[:, index] - np.float32(edge[0])) * scale
        edges = tuple(np.arange(len(edge), dtype=float) for edge in edges)
    for index, edge in enumerate(edges):
        coordinates[coordinates[:, index] >= edge[-1], index] = np.nan
    return coordinates, tuple(edges)
