"""Numerical grid configuration and rebinning services for project datasets.

This module is deliberately independent of project loading, caches, derived-data
recipes, and Qt.  The project_data module re-exports its established helper API.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dataset import PointData4D, PointListData
from .mdhisto import (
    MDHistoAxis,
    MDHistoChannel,
    MDHistoData,
    mdhisto_coverage_fraction,
    mdhisto_measured_bins,
)
from .project_coordinates import (
    _clean_axis_weight,
    _identity_vector,
    _mdhisto_coordinate_grids,
    _mdhisto_rebin_axis_vector,
    _mdhisto_rebin_source_axis_vectors,
)
from .rebin import _uniform_center_edges, rebin_nd, rebin_nd_symmetry
from .symmetry import resolve_symmetry, symmetry_spec_from_config

DEFAULT_REBIN_MAX_BATCH_MB = 192
DEFAULT_MINIMUM_COVERAGE = 0.9
DEFAULT_MINIMUM_SAMPLES = 0.0
REBIN_RESOLUTION_MODE_KEY = "resolution_mode"
REBIN_AXIS_MODES = frozenset({"discrete", "step", "bins", "edges", "tolerance"})

_COMPATIBILITY_NAMESPACE: Mapping[str, Any] | None = None


def configure_rebinning_compatibility(namespace: Mapping[str, Any] | None) -> None:
    """Use live facade constants while preserving standalone service imports."""

    global _COMPATIBILITY_NAMESPACE
    _COMPATIBILITY_NAMESPACE = namespace


def _compatibility_value(name: str, default: Any) -> Any:
    if _COMPATIBILITY_NAMESPACE is None:
        return default
    return _COMPATIBILITY_NAMESPACE.get(name, default)


def _parameter_to_text(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _rebin_symmetry_operations(
    config: dict[str, Any],
    lattice_parameters: dict[str, Any] | None = None,
) -> tuple:
    payload = config.get("symmetry")
    spec = symmetry_spec_from_config(payload)
    stored_lattice = payload.get("lattice_parameters") if isinstance(payload, dict) else None
    lattice = lattice_parameters if lattice_parameters is not None else stored_lattice
    return resolve_symmetry(spec, lattice_parameters=lattice)


def _rebin_symmetry_count(
    config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None
) -> int:
    try:
        return len(_rebin_symmetry_operations(config, lattice_parameters))
    except (ImportError, ValueError):
        return 1


def _rebin_symmetry_matrices(
    config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None
) -> tuple[np.ndarray, ...] | None:
    operations = _rebin_symmetry_operations(config, lattice_parameters)
    if len(operations) == 1 and np.allclose(operations[0].matrix_hkl, np.eye(3)):
        return None
    return tuple(np.asarray(operation.matrix_hkl, dtype=float) for operation in operations)


def _rebin_symmetry_metadata(
    config: dict[str, Any], lattice_parameters: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    spec = symmetry_spec_from_config(config.get("symmetry"))
    if not spec.enabled:
        return None
    operations = _rebin_symmetry_operations(config, lattice_parameters)
    return {
        "mode": spec.mode,
        "expression": spec.expression,
        "operation_count": len(operations),
        "operations_hkl": [
            np.asarray(operation.matrix_hkl, dtype=float).tolist() for operation in operations
        ],
        "labels": [operation.label for operation in operations],
        "energy_unchanged": True,
    }


def _rebin_mean_weighting(config: dict[str, Any]) -> str:
    value = config.get("mean_weighting")
    if value == "normalization":
        return "uniform"
    return str(value) if value in {"inverse_variance", "uniform"} else "uniform"


def _rebin_axis_mode(config: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    """Return one axis's grid-construction mode, including legacy migration."""

    mode = str(axis.get("mode", "")).casefold()
    axis_modes = _compatibility_value("REBIN_AXIS_MODES", REBIN_AXIS_MODES)
    if mode in axis_modes:
        return mode
    if axis.get("bin_edges") is not None:
        return "edges"
    resolution_key = _compatibility_value(
        "REBIN_RESOLUTION_MODE_KEY", REBIN_RESOLUTION_MODE_KEY
    )
    return "bins" if config.get(resolution_key) == "bins" else "step"


def _rebin_axis_fractional(
    config: Mapping[str, Any], axis: Mapping[str, Any]
) -> bool:
    """Return whether a physical axis distributes points between adjacent bins."""

    if _rebin_axis_mode(config, axis) in {"discrete", "tolerance"}:
        return False
    value = axis.get("fractional")
    if isinstance(value, bool):
        return value
    legacy = config.get("fractional")
    return bool(legacy) if isinstance(legacy, bool) else True


def _migrate_rebin_axis_modes(config: dict[str, Any]) -> None:
    """Persist the per-axis mode representation used by current projects."""

    axes = config.get("axes")
    if not isinstance(axes, list):
        return
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        axis["mode"] = _rebin_axis_mode(config, axis)
        axis["fractional"] = _rebin_axis_fractional(config, axis)
    config.pop("fractional", None)


def _rebin_fractional_axes(
    config: Mapping[str, Any], axes_config: Sequence[Mapping[str, Any]]
) -> list[bool]:
    """Return the independently configured assignment behavior for each axis."""

    return [_rebin_axis_fractional(config, axis) for axis in axes_config]


def _rebin_minimum_coverage(config: dict[str, Any]) -> float:
    default = _compatibility_value("DEFAULT_MINIMUM_COVERAGE", DEFAULT_MINIMUM_COVERAGE)
    try:
        value = float(config.get("minimum_coverage", default))
    except (TypeError, ValueError):
        return float(default)
    return float(np.clip(value, 0.0, 1.0))


def _rebin_minimum_samples(config: dict[str, Any]) -> float:
    """Return the minimum effective sample contribution for an output bin."""

    default = _compatibility_value("DEFAULT_MINIMUM_SAMPLES", DEFAULT_MINIMUM_SAMPLES)
    try:
        value = float(config.get("minimum_samples", default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value) or value < 0.0:
        return float(default)
    return value


def _rebin_max_batch_mb(config: dict[str, Any]) -> int:
    default = _compatibility_value("DEFAULT_REBIN_MAX_BATCH_MB", DEFAULT_REBIN_MAX_BATCH_MB)
    try:
        return max(int(config.get("max_batch_mb", default)), 1)
    except (TypeError, ValueError):
        return int(default)


def _rebin_max_batch_bytes(config: dict[str, Any]) -> int:
    from .performance import transient_rebin_memory_limit_bytes

    requested = _rebin_max_batch_mb(config) * 1024 * 1024
    return min(requested, transient_rebin_memory_limit_bytes())


def _rebin_max_parallel_bytes() -> int:
    """Return the machine ceiling for thread-private rebin accumulators."""

    from .performance import transient_rebin_memory_limit_bytes

    return transient_rebin_memory_limit_bytes()


def _rebin_resolution_mode(config: dict[str, Any]) -> str:
    """Return the active resolution control mode for a rebin panel."""

    resolution_key = _compatibility_value(
        "REBIN_RESOLUTION_MODE_KEY", REBIN_RESOLUTION_MODE_KEY
    )
    return "bins" if config.get(resolution_key) == "bins" else "step"


def _rebin_axis_bound_is_auto(axis: Mapping[str, Any], key: str) -> bool:
    """Return whether a saved bound still matches its last automatic value."""

    if key not in {"lower", "upper"} or not bool(axis.get(f"auto_{key}", False)):
        return False
    try:
        value = float(axis[key])
        automatic = float(axis.get(f"auto_{key}_value", value))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isclose(value, automatic))


def _resolve_auto_rebin_axes(
    axes_config: Sequence[dict[str, Any]],
    data_bounds: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Resolve blank limits and align uniform bins so zero is a bin center."""

    if len(axes_config) != len(data_bounds):
        raise ValueError("automatic rebin bounds must match the coordinate dimensions")
    resolved: list[dict[str, Any]] = []
    for axis_config, (data_lower, data_upper) in zip(axes_config, data_bounds, strict=True):
        axis = _sanitize_rebin_axis_config(dict(axis_config))
        if axis.get("bin_edges") is not None:
            resolved.append(axis)
            continue
        auto_lower = _rebin_axis_bound_is_auto(axis, "lower")
        auto_upper = _rebin_axis_bound_is_auto(axis, "upper")
        if not (auto_lower or auto_upper):
            resolved.append(axis)
            continue
        step = float(axis.get("step_size", 0.0) or 0.0)
        auto_step = bool(axis.get("auto_step_size", False)) and np.isclose(
            step,
            float(axis.get("auto_step_size_value", step)),
        )
        if auto_step:
            target_lower = float(data_lower) if auto_lower else float(axis["lower"])
            target_upper = float(data_upper) if auto_upper else float(axis["upper"])
            width = target_upper - target_lower
            if np.isfinite(width) and width > 0.0:
                step = width / max(int(axis.get("num_bins", 1)) - 1, 1)
        if not np.isfinite(step) or step <= 0.0:
            width = float(data_upper) - float(data_lower)
            step = width / max(int(axis.get("num_bins", 1)) - 1, 1)
        if not np.isfinite(step) or step <= 0.0:
            step = 1.0
        # Uniform edges at (n + 1/2)*step put an integer multiple of the
        # step, including zero, at every bin center.
        # Choose half-step edges strictly outside the finite data interval.
        # The resulting centers are integer multiples of ``step``. Strict
        # containment also keeps samples exactly on half-step boundaries from
        # collapsing into the same final bin.
        aligned_lower = (math.ceil(float(data_lower) / step - 0.5) - 0.5) * step
        aligned_upper = (math.floor(float(data_upper) / step - 0.5) + 1.5) * step
        if aligned_upper <= aligned_lower:
            aligned_upper = aligned_lower + step
        if auto_lower:
            axis["lower"] = aligned_lower + step / 2
        if auto_upper:
            axis["upper"] = aligned_upper - step / 2
        axis["step_size"] = step
        axis["num_bins"] = max(
            int(round((float(axis["upper"]) - float(axis["lower"])) / step)) + 1, 1
        )
        resolved.append(axis)
    return resolved


def _cluster_coordinate_centers(values: Any, tolerance: float) -> np.ndarray:
    """Cluster finite coordinates while keeping every member within tolerance."""

    finite = np.sort(np.asarray(values, dtype=float).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot determine discrete bins without finite coordinates")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("axis tolerance must be finite and nonnegative")
    if tolerance == 0.0:
        return np.unique(finite)
    centers: list[float] = []
    start = 0
    running_sum = float(finite[0])
    count = 1
    for stop in range(1, finite.size):
        candidate_sum = running_sum + float(finite[stop])
        candidate_count = count + 1
        candidate_mean = candidate_sum / candidate_count
        if (
            max(
                candidate_mean - float(finite[start]),
                float(finite[stop]) - candidate_mean,
            )
            <= tolerance + 1e-12
        ):
            running_sum = candidate_sum
            count = candidate_count
            continue
        centers.append(running_sum / count)
        start = stop
        running_sum = float(finite[stop])
        count = 1
    centers.append(running_sum / count)
    return np.asarray(centers, dtype=float)


def _coordinate_center_edges(centers: Any, *, singleton_half_width: float = 0.5) -> np.ndarray:
    """Build nearest-center boundaries for an ordered set of centers."""

    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0.0):
        raise ValueError("discrete bin centers must be finite and strictly increasing")
    if values.size == 1:
        half_width = max(float(singleton_half_width), np.finfo(float).eps)
        return np.asarray([values[0] - half_width, values[0] + half_width])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.r_[
        values[0] - (values[1] - values[0]) / 2.0,
        midpoints,
        values[-1] + (values[-1] - values[-2]) / 2.0,
    ]


def _resolve_data_driven_rebin_axes(
    config: Mapping[str, Any],
    axes_config: Sequence[dict[str, Any]],
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    """Resolve Discrete/Tolerance axes from the projected source coordinates."""

    values = np.asarray(coordinates, dtype=float).reshape(-1, len(axes_config))
    resolved: list[dict[str, Any]] = []
    for index, axis_config in enumerate(axes_config):
        axis = _sanitize_rebin_axis_config(dict(axis_config))
        mode = _rebin_axis_mode(config, axis)
        axis["mode"] = mode
        if mode not in {"discrete", "tolerance"}:
            resolved.append(axis)
            continue
        tolerance = 0.0 if mode == "discrete" else float(axis["tolerance"])
        candidates = axis.get("candidate_centers", values[:, index])
        centers = _cluster_coordinate_centers(candidates, tolerance)
        edges = _coordinate_center_edges(
            centers,
            singleton_half_width=tolerance if tolerance > 0.0 else 0.5,
        )
        axis.update(
            {
                "bin_edges": edges.tolist(),
                "resolved_centers": centers.tolist(),
                "lower": float(centers[0]),
                "upper": float(centers[-1]),
                "num_bins": int(centers.size),
                "step_size": (
                    float(np.median(np.diff(centers)))
                    if centers.size > 1
                    else max(2.0 * tolerance, 1.0)
                ),
            }
        )
        resolved.append(axis)
    return resolved


def _axis_mode_coordinates_with_symmetry(
    config: Mapping[str, Any],
    projected_coordinates: np.ndarray,
    *,
    physical_coordinates: np.ndarray | None = None,
    symmetry: Sequence[np.ndarray] | None = None,
    output_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Return all projected coordinates needed to derive dynamic axis grids."""

    axes = config.get("axes", [])
    if not any(
        isinstance(axis, Mapping) and _rebin_axis_mode(config, axis) in {"discrete", "tolerance"}
        for axis in axes
    ):
        return np.asarray(projected_coordinates, dtype=float)
    if symmetry is None:
        return np.asarray(projected_coordinates, dtype=float)
    if physical_coordinates is None or output_basis is None:
        raise ValueError("dynamic axis modes require projected symmetry coordinates")
    physical = np.asarray(physical_coordinates, dtype=float)
    inverse_basis = np.linalg.inv(np.asarray(output_basis, dtype=float))
    ndim = physical.shape[-1]
    projected = []
    for operation in symmetry:
        transform = np.eye(ndim, dtype=float)
        transform[:3, :3] = np.asarray(operation, dtype=float).T
        projected.append(physical @ transform @ inverse_basis)
    return np.concatenate(projected, axis=0)


def _finite_coordinate_bounds(coordinates: np.ndarray) -> list[tuple[float, float]]:
    values = np.asarray(coordinates, dtype=float).reshape(-1, coordinates.shape[-1])
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.size == 0:
        raise ValueError("cannot determine automatic limits without finite coordinates")
    return [
        (float(np.min(values[:, index])), float(np.max(values[:, index])))
        for index in range(values.shape[1])
    ]


def _symmetry_projected_coordinate_bounds(
    physical_coordinates: np.ndarray,
    symmetry: Sequence[np.ndarray],
    output_basis: np.ndarray,
) -> list[tuple[float, float]]:
    """Return output-coordinate bounds over every reciprocal-symmetry image."""

    physical = np.asarray(physical_coordinates, dtype=float)
    basis = np.asarray(output_basis, dtype=float)
    ndim = physical.shape[-1]
    if ndim < 3 or basis.shape != (ndim, ndim):
        raise ValueError("symmetry auto limits require HKL as the first three coordinates")
    inverse_basis = np.linalg.inv(basis)
    lower = np.full(ndim, np.inf, dtype=float)
    upper = np.full(ndim, -np.inf, dtype=float)
    for operation in symmetry:
        transform = np.eye(ndim, dtype=float)
        transform[:3, :3] = np.asarray(operation, dtype=float).T
        projected = physical @ transform @ inverse_basis
        operation_bounds = _finite_coordinate_bounds(projected)
        lower = np.minimum(lower, [bound[0] for bound in operation_bounds])
        upper = np.maximum(upper, [bound[1] for bound in operation_bounds])
    return list(zip(lower.tolist(), upper.tolist(), strict=True))


def _rebin_grid_kwargs(config: dict[str, Any], axes_config: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a mixed uniform/explicit-edge grid for the public rebinner."""

    # Step axes retain constant-width indexing. Bins axes are materialized as
    # explicit edges so the one-bin lower/upper interval convention and exact
    # requested count remain valid when modes are mixed across dimensions.
    result: dict[str, Any] = {"step_size": [float(axis["step_size"]) for axis in axes_config]}
    bin_edges = [
        (
            _uniform_center_edges(
                axis["lower"], axis["upper"], count=int(axis["num_bins"])
            ).tolist()
            if _rebin_axis_mode(config, axis) == "bins"
            else axis.get("bin_edges")
            if _rebin_axis_mode(config, axis) in {"edges", "discrete", "tolerance"}
            else None
        )
        for axis in axes_config
    ]
    if any(edges is not None for edges in bin_edges):
        result["bin_edges"] = bin_edges
    return result


def _composite_rebin_step_sizes(config: dict[str, Any]) -> list[float] | None:
    """Return composite step sizes only when the composite is in Step mode."""

    if _rebin_resolution_mode(config) != "step":
        return None
    return [
        float(axis["step_size"])
        for axis in (_sanitize_rebin_axis_config(axis) for axis in config.get("axes", []))
    ]


def _composite_rebin_bin_edges(
    config: dict[str, Any],
) -> list[list[float] | None] | None:
    """Return per-axis explicit edges, retaining ``None`` for uniform axes."""

    axes = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    edges = [axis.get("bin_edges") for axis in axes]
    return edges if any(values is not None for values in edges) else None


def _default_rebin_axes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, PointListData):
        axes = []
        coordinate_names = list(data.coordinate_names) or list(data.column_names)[:1]
        for name in coordinate_names:
            values = np.asarray(data.column(name), dtype=float)
            finite = values[np.isfinite(values)]
            lower = float(np.min(finite)) if finite.size else 0.0
            upper = float(np.max(finite)) if finite.size else 1.0
            num_bins = max(int(np.unique(finite).size), 1) if finite.size else 1
            num_bins = min(num_bins, 200)
            axes.append(
                {
                    "name": name,
                    "units": data.unit(name),
                    "lower": lower,
                    "upper": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
                }
            )
        return axes
    if isinstance(data, MDHistoData):
        axes: list[dict[str, Any]] = []
        ndim = len(data.axes)
        for index, (axis, size) in enumerate(zip(data.axes, data.shape, strict=True)):
            lower, upper = float(axis.centers[0]), float(axis.centers[-1])
            num_bins = max(int(size), 1)
            native_lower, native_upper = _axis_bounds(axis, size)
            vector = _mdhisto_rebin_axis_vector(axis, index, ndim)
            variable = _mdhisto_rebin_axis_variable(axis, index, ndim)
            axes.append(
                {
                    "name": _rebin_axis_name(variable, vector),
                    "variable": variable,
                    "units": axis.units,
                    "vector": vector,
                    "lower": lower,
                    "upper": upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": lower,
                    "auto_upper_value": upper,
                    "num_bins": num_bins,
                    "step_size": (native_upper - native_lower)
                    if size == 1
                    else _step_size_from_bounds(lower, upper, num_bins),
                }
            )
        return axes
    if isinstance(data, PointData4D):
        axes = []
        for index, (name, units, values) in enumerate(
            (
                ("H", "rlu", data.H),
                ("K", "rlu", data.K),
                ("L", "rlu", data.L),
                ("E", "meV", data.E),
            )
        ):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            lower = float(np.min(finite)) if finite.size else 0.0
            upper = float(np.max(finite)) if finite.size else 1.0
            # Point clouds commonly contain a distinct floating-point value at
            # nearly every observation. Treating every unique value as a grid
            # coordinate would create an unusably large Cartesian product
            # (MACS HKLE scans are a representative case). Keep the automatic
            # preview below the normal two-million-bin safety threshold; users
            # can then choose instrument-appropriate step sizes explicitly.
            num_bins = min(max(int(np.unique(finite).size), 1), 32) if finite.size else 1
            axes.append(
                {
                    "name": name,
                    "variable": name,
                    "units": units,
                    "vector": _identity_vector(index, 4),
                    "lower": lower,
                    "upper": upper,
                    "auto_lower": True,
                    "auto_upper": True,
                    "auto_lower_value": lower,
                    "auto_upper_value": upper,
                    "num_bins": num_bins,
                    "step_size": _step_size_from_bounds(lower, upper, num_bins),
                    "auto_step_size": True,
                    "auto_step_size_value": _step_size_from_bounds(lower, upper, num_bins),
                }
            )
        return axes
    return []


def _axis_bounds(axis: MDHistoAxis, size: int) -> tuple[float, float]:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return float(values[0]), float(values[-1])
    centers = axis.centers
    if centers.size == 0:
        return 0.0, 1.0
    if centers.size == 1:
        return float(centers[0] - 0.5), float(centers[0] + 0.5)
    step = float(np.nanmedian(np.diff(centers)))
    return float(centers[0] - 0.5 * step), float(centers[-1] + 0.5 * step)


def _step_size_from_bounds(lower: float, upper: float, num_bins: int) -> float:
    if num_bins <= 0:
        return 0.0
    if upper == lower:
        return 1.0
    return float((float(upper) - float(lower)) / float(max(num_bins - 1, 1)))


def _num_bins_from_step_size(lower: Any, upper: Any, step_size: float) -> int:
    width = abs(float(upper) - float(lower))
    if width == 0.0:
        return 1
    return max(int(np.floor(width / float(step_size) + 1.0e-10)) + 1, 1)


def _normalize_rebin_bin_edges(value: Any) -> list[float]:
    """Validate one axis's optional explicit, potentially nonuniform edges."""

    edges = np.asarray(value, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin edges must be a one-dimensional list with at least two values")
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin edges must be finite and strictly increasing")
    return edges.tolist()


def _sanitize_rebin_axis_config(axis_config: dict[str, Any]) -> dict[str, Any]:
    lower = float(axis_config.get("lower", 0.0))
    upper = float(axis_config.get("upper", lower))
    num_bins = max(int(axis_config.get("num_bins", 1)), 1)
    try:
        step_size = float(axis_config.get("step_size"))
    except (TypeError, ValueError):
        step_size = 0.0
    if not np.isfinite(step_size) or step_size <= 0.0:
        step_size = _step_size_from_bounds(lower, upper, num_bins)
    sanitized = {
        **axis_config,
        "lower": lower,
        "upper": upper,
        "num_bins": num_bins,
        "step_size": step_size,
    }
    mode = str(axis_config.get("mode", "")).casefold()
    if mode:
        axis_modes = _compatibility_value("REBIN_AXIS_MODES", REBIN_AXIS_MODES)
        sanitized["mode"] = mode if mode in axis_modes else "step"
    if "tolerance" in axis_config or mode == "tolerance":
        try:
            tolerance = float(axis_config.get("tolerance", step_size))
        except (TypeError, ValueError):
            tolerance = step_size
        sanitized["tolerance"] = (
            tolerance if np.isfinite(tolerance) and tolerance > 0.0 else step_size
        )
    explicit_edges = axis_config.get("bin_edges")
    if explicit_edges is not None and not (
        isinstance(explicit_edges, str) and not explicit_edges.strip()
    ):
        edges = _normalize_rebin_bin_edges(explicit_edges)
        sanitized["bin_edges"] = edges
        sanitized["lower"] = (edges[0] + edges[1]) / 2 if len(edges) > 2 else edges[0]
        sanitized["upper"] = (edges[-2] + edges[-1]) / 2 if len(edges) > 2 else edges[-1]
        sanitized["num_bins"] = len(edges) - 1
        sanitized["step_size"] = (edges[-1] - edges[0]) / (len(edges) - 1)
    else:
        sanitized.pop("bin_edges", None)
    vector = axis_config.get("vector")
    if isinstance(vector, (list, tuple, np.ndarray)):
        sanitized["vector"] = [_clean_axis_weight(component) for component in vector]
    return sanitized


def _rebin_axis_vector(axis_config: dict[str, Any], index: int, ndim: int) -> np.ndarray:
    """Return the projection vector for a rebin output axis, falling back to identity."""

    vector = axis_config.get("vector")
    if isinstance(vector, (list, tuple, np.ndarray)) and len(vector) == ndim:
        candidate = np.asarray(vector, dtype=float)
        if np.all(np.isfinite(candidate)):
            return candidate
    return np.asarray(_identity_vector(index if 0 <= index < ndim else 0, ndim), dtype=float)


def _mdhisto_rebin_axis_variable(axis: MDHistoAxis, index: int, ndim: int) -> str:
    """Return the scalar variable used to parameterize one rebin axis row."""

    role_variables = {
        "h": "H",
        "k": "K",
        "l": "L",
        "energy": "E",
        "energy_transfer": "E",
    }
    if axis.role in role_variables:
        return role_variables[axis.role]
    if str(axis.name).casefold().replace("_", "") in {
        "deltae",
        "energy",
        "energytransfer",
    }:
        return "E"
    symbols = re.findall(r"[HKL]", str(axis.name).upper())
    if symbols:
        return symbols[0]
    if ndim == 4 and 0 <= index < 4:
        return ("H", "K", "L", "E")[index]
    return str(axis.name or f"Axis {index + 1}")


def _rebin_axis_name(variable: str, vector: Any) -> str:
    """Generate the plotted axis name from its variable and HKLE direction."""

    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size == 4 and str(variable).upper() == "E":
        coefficient = _clean_axis_weight(values[3])
        if coefficient == 1.0:
            return "DeltaE"
        return f"{coefficient:g}DeltaE"
    if values.size == 4:
        symbol = str(variable).upper()
        terms = []
        for value in values[:3]:
            coefficient = _clean_axis_weight(value)
            if coefficient == 0.0:
                terms.append("0")
            elif coefficient == 1.0:
                terms.append(symbol)
            elif coefficient == -1.0:
                terms.append(f"-{symbol}")
            else:
                terms.append(f"{coefficient:g}{symbol}")
        return f"[{','.join(terms)}]"
    return str(variable)


def _validate_mdhisto_rebin_basis(axes_config: list[dict[str, Any]], ndim: int) -> np.ndarray:
    """Validate and return the output basis represented by rebin axis rows."""

    basis = np.vstack(
        [
            _rebin_axis_vector(axis_config, index, ndim)
            for index, axis_config in enumerate(axes_config)
        ]
    )
    if basis.shape != (ndim, ndim) or not np.all(np.isfinite(basis)):
        raise ValueError("coordinate axes must form a finite square basis")
    if ndim == 4:
        for index, (axis_config, vector) in enumerate(zip(axes_config, basis, strict=True)):
            variable = str(axis_config.get("variable", ("H", "K", "L", "E")[index])).upper()
            if variable == "E":
                if np.any(vector[:3] != 0.0) or vector[3] == 0.0:
                    raise ValueError(
                        f"axis row {index + 1} is the energy variable and must contain only a nonzero E component"
                    )
            elif vector[3] != 0.0:
                raise ValueError(
                    f"axis row {index + 1} is a momentum variable; momentum and energy components cannot be mixed"
                )
    if np.linalg.matrix_rank(basis) != ndim:
        rank = int(np.linalg.matrix_rank(basis))
        raise ValueError(
            "coordinate axis vectors must form an invertible basis; "
            f"this matrix has rank {rank} rather than {ndim}"
        )
    return basis


def _momentum_rebin_vector_text(axis_config: dict[str, Any], index: int) -> str:
    """Format one GUI momentum-basis row without exposing the energy column."""

    return _parameter_to_text(_rebin_axis_vector(axis_config, index, 4)[:3].tolist())


def _momentum_rebin_axis_indices(axes_config: Sequence[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, axis in enumerate(axes_config)
        if str(axis.get("variable", ("H", "K", "L", "E")[index])).upper() != "E"
    ]


def _momentum_rebin_matrix(axes_config: Sequence[dict[str, Any]]) -> list[list[float]]:
    indices = _momentum_rebin_axis_indices(axes_config)
    if len(axes_config) != 4 or len(indices) != 3:
        raise ValueError(
            "HKLE rebinning requires three momentum coordinates and one energy coordinate"
        )
    return [_rebin_axis_vector(axes_config[index], index, 4)[:3].tolist() for index in indices]


def _momentum_coordinate_variables(matrix: np.ndarray) -> list[str]:
    """Choose concise H/K/L scalar labels for momentum-matrix rows."""

    symbols = ["H", "K", "L"]
    variables: list[str | None] = [None, None, None]
    used: set[str] = set()
    for row, values in enumerate(matrix):
        nonzero = np.flatnonzero(~np.isclose(values, 0.0))
        if nonzero.size != 1:
            continue
        symbol = symbols[int(nonzero[0])]
        if symbol not in used:
            variables[row] = symbol
            used.add(symbol)
    remaining = iter(symbol for symbol in symbols if symbol not in used)
    return [value if value is not None else next(remaining) for value in variables]


def _rebin_config_basis_bounds(
    axes_config: list[dict[str, Any]],
    candidate_basis: np.ndarray,
) -> list[tuple[float, float]]:
    """Transform the configured output box into a replacement coordinate basis."""

    current_basis = _validate_mdhisto_rebin_basis(axes_config, 4)
    endpoints = [(float(axis["lower"]), float(axis["upper"])) for axis in axes_config]
    projected_corners = np.asarray(
        [
            [endpoints[index][bit] for index, bit in enumerate(bits)]
            for bits in np.ndindex(*(2 for _ in endpoints))
        ],
        dtype=float,
    )
    physical_corners = projected_corners @ current_basis
    new_coordinates = physical_corners @ np.linalg.inv(candidate_basis)
    return [
        (float(np.min(new_coordinates[:, index])), float(np.max(new_coordinates[:, index])))
        for index in range(4)
    ]


def _point_data_rebin_basis_bounds(
    data: PointData4D,
    axes_config: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return exact finite point-cloud bounds in the requested output basis."""

    basis = _validate_mdhisto_rebin_basis(axes_config, 4)
    coordinates = np.column_stack(data.coordinates())
    coordinates = coordinates[np.all(np.isfinite(coordinates), axis=1)]
    if coordinates.size == 0:
        raise ValueError("point dataset has no finite HKLE coordinates")
    projected = coordinates @ np.linalg.inv(basis)
    bounds: list[tuple[float, float]] = []
    for index in range(4):
        lower = float(np.min(projected[:, index]))
        upper = float(np.max(projected[:, index]))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        bounds.append((lower, upper))
    return bounds


def _update_rebin_momentum_basis(
    axes_config: list[dict[str, Any]],
    index: int,
    momentum_vector: Sequence[float],
    *,
    data: MDHistoData | PointData4D | None = None,
) -> None:
    """Apply one row of the user-facing 3x3 momentum coordinate block."""

    if len(axes_config) != 4 or not (0 <= index < 4):
        raise ValueError("momentum-coordinate editing requires four HKLE axes")
    variable = str(axes_config[index].get("variable", ("H", "K", "L", "E")[index])).upper()
    if variable == "E":
        raise ValueError("the energy coordinate is fixed and is not part of the 3x3 momentum block")
    vector = np.asarray(momentum_vector, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("a momentum coordinate row must contain three finite H, K, and L values")
    matrix = _momentum_rebin_matrix(axes_config)
    matrix[_momentum_rebin_axis_indices(axes_config).index(index)] = vector.tolist()
    _update_rebin_momentum_matrix(axes_config, matrix, data=data)


def _update_rebin_momentum_matrix(
    axes_config: list[dict[str, Any]],
    momentum_matrix: Sequence[Sequence[float]],
    *,
    data: MDHistoData | PointData4D | None = None,
) -> None:
    """Replace the complete 3x3 momentum block atomically."""

    matrix = np.asarray(momentum_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("the momentum-coordinate matrix must be a finite 3x3 array")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 3:
        raise ValueError(
            "the momentum-coordinate matrix must be invertible; "
            f"this matrix has rank {rank} rather than 3"
        )
    candidate = [dict(axis) for axis in axes_config]
    indices = _momentum_rebin_axis_indices(candidate)
    if len(candidate) != 4 or len(indices) != 3:
        raise ValueError(
            "HKLE rebinning requires three momentum coordinates and one energy coordinate"
        )
    for axis_index, axis in enumerate(candidate):
        axis.setdefault("variable", ("H", "K", "L", "E")[axis_index])
    variables = _momentum_coordinate_variables(matrix)
    for row, axis_index in enumerate(indices):
        candidate[axis_index]["variable"] = variables[row]
        candidate[axis_index]["vector"] = [
            *[_clean_axis_weight(component) for component in matrix[row]],
            0.0,
        ]
    basis = _validate_mdhisto_rebin_basis(candidate, 4)
    if isinstance(data, MDHistoData):
        bounds = _mdhisto_rebin_basis_bounds(data, candidate)
    elif isinstance(data, PointData4D):
        bounds = _point_data_rebin_basis_bounds(data, candidate)
    else:
        bounds = _rebin_config_basis_bounds(axes_config, basis)
    for axis_config, (lower, upper) in zip(candidate, bounds, strict=True):
        previous_step = float(axis_config.get("step_size", 0.0) or 0.0)
        axis_config.pop("bin_edges", None)
        axis_config["lower"] = lower
        axis_config["upper"] = upper
        if bool(axis_config.get("auto_lower", False)):
            axis_config["auto_lower_value"] = lower
        if bool(axis_config.get("auto_upper", False)):
            axis_config["auto_upper_value"] = upper
        if previous_step > 0.0 and np.isfinite(previous_step):
            axis_config["num_bins"] = _num_bins_from_step_size(lower, upper, previous_step)
        axis_config["name"] = _rebin_axis_name(
            str(axis_config.get("variable", "")), axis_config.get("vector", [])
        )
        axis_config.update(_sanitize_rebin_axis_config(axis_config))
    axes_config[:] = candidate


def _mdhisto_rebin_basis_transform(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
) -> np.ndarray:
    """Map source bin coordinates into coordinates of the requested basis."""

    ndim = len(data.axes)
    output_basis = _validate_mdhisto_rebin_basis(axes_config, ndim)
    source_vectors = _mdhisto_rebin_source_axis_vectors(data)
    if any(vector is None for vector in source_vectors):
        raise ValueError("source axes do not define a complete coordinate basis")
    source_basis = np.vstack(source_vectors)
    if source_basis.shape != (ndim, ndim) or np.linalg.matrix_rank(source_basis) != ndim:
        raise ValueError("source coordinate axes must be linearly independent")
    return source_basis @ np.linalg.inv(output_basis)


def _mdhisto_rebin_basis_bounds(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return a bounding box containing the full source grid in a new basis."""

    transform = _mdhisto_rebin_basis_transform(data, axes_config)
    source_bounds = [
        _axis_bounds(axis, size) for axis, size in zip(data.axes, data.shape, strict=True)
    ]
    bounds = []
    for output_index in range(transform.shape[1]):
        lower = 0.0
        upper = 0.0
        for (source_lower, source_upper), coefficient in zip(
            source_bounds, transform[:, output_index], strict=True
        ):
            endpoints = (coefficient * source_lower, coefficient * source_upper)
            lower += min(endpoints)
            upper += max(endpoints)
        bounds.append((float(lower), float(upper)))
    return bounds


def _update_mdhisto_rebin_basis(
    data: MDHistoData,
    axes_config: list[dict[str, Any]],
    index: int,
    vector: list[float],
) -> None:
    """Apply one basis edit, regenerating names and full-data output bounds."""

    candidate = [dict(axis) for axis in axes_config]
    candidate[index]["vector"] = list(vector)
    _validate_mdhisto_rebin_basis(candidate, len(data.axes))
    bounds = _mdhisto_rebin_basis_bounds(data, candidate)
    for axis_config, (lower, upper) in zip(candidate, bounds, strict=True):
        previous_step = float(axis_config.get("step_size", 0.0) or 0.0)
        axis_config["lower"] = lower
        axis_config["upper"] = upper
        if previous_step > 0.0 and np.isfinite(previous_step):
            axis_config["num_bins"] = _num_bins_from_step_size(lower, upper, previous_step)
        axis_config["name"] = _rebin_axis_name(
            str(axis_config.get("variable", "")), axis_config.get("vector", [])
        )
        axis_config.update(_sanitize_rebin_axis_config(axis_config))
    axes_config[:] = candidate


def _mdhisto_rebin_component(
    data: MDHistoData,
    source_grids: list[np.ndarray],
    axis_config: dict[str, Any],
    index: int,
) -> np.ndarray:
    ndim = len(data.axes)
    vector = _rebin_axis_vector(axis_config, index, ndim)
    component = np.zeros(data.shape, dtype=float)
    if vector.size == 4:
        source_vectors = _mdhisto_rebin_source_axis_vectors(data)
        for source_vector, grid in zip(source_vectors, source_grids, strict=True):
            if source_vector is not None and np.allclose(vector, source_vector):
                return np.asarray(grid, dtype=float)
        coords = _mdhisto_coordinate_grids(data)
        if all(name in coords for name in ("H", "K", "L", "E")):
            for weight, name in zip(vector, ("H", "K", "L", "E"), strict=True):
                if weight:
                    component = component + float(weight) * coords[name]
            return component
    for weight, grid in zip(vector, source_grids, strict=True):
        if weight:
            component = component + float(weight) * grid
    return component


def _mdhisto_axis_edges(axis: MDHistoAxis, size: int) -> np.ndarray:
    values = np.asarray(axis.values, dtype=float)
    if values.size == size + 1:
        return values
    if values.size == size:
        if size == 1:
            return np.asarray([values[0] - 0.5, values[0] + 0.5])
        edges = np.empty(size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return edges
    raise ValueError(f"axis {axis.name!r} does not define {size} bins")


def _mdhisto_cell_volumes(data: MDHistoData) -> np.ndarray:
    """Return native hypervolumes represented by MDHisto bin centers."""

    volume = np.ones(data.shape, dtype=float)
    for dim, (axis, size) in enumerate(zip(data.axes, data.shape, strict=True)):
        edges = _mdhisto_axis_edges(axis, size)
        widths = np.diff(edges)
        shape = [1] * data.signal.ndim
        shape[dim] = size
        volume *= widths.reshape(shape)
    return np.abs(volume)


def _output_bin_volumes(bins_list: Sequence[np.ndarray]) -> np.ndarray:
    shape = tuple(len(values) - 1 for values in bins_list)
    volume = np.ones(shape, dtype=float)
    for dim, values in enumerate(bins_list):
        widths = np.abs(np.diff(np.asarray(values, dtype=float)))
        reshape = [1] * len(shape)
        reshape[dim] = widths.size
        volume *= widths.reshape(reshape)
    return volume


def _rebin_mdhisto_coverage(
    data: MDHistoData,
    coords: np.ndarray,
    config: dict[str, Any],
    axes_config: Sequence[dict[str, Any]],
    bins_list: Sequence[np.ndarray],
    *,
    symmetry: Sequence[np.ndarray] | None = None,
    output_axes: np.ndarray | None = None,
) -> np.ndarray:
    """Map measured native-bin hypervolume into requested output bins."""

    coordinate_ndim = int(np.asarray(coords).shape[-1])
    source_ndim = data.signal.ndim
    source_coverage = mdhisto_coverage_fraction(data)
    usable = (
        np.isfinite(data.signal)
        & np.isfinite(data.errors)
        & ~np.asarray(data.mask, dtype=bool)
        & mdhisto_measured_bins(data)
    )
    source_coverage = np.where(usable, source_coverage, 0.0)
    source_volume = _mdhisto_cell_volumes(data)
    vectors = np.asarray(
        [
            _rebin_axis_vector(axis_config, index, source_ndim)
            for index, axis_config in enumerate(axes_config[:source_ndim])
        ],
        dtype=float,
    )
    mapping = (
        _mdhisto_rebin_basis_transform(data, list(axes_config[:source_ndim]))
        if source_ndim == 4
        else vectors.T
    )
    if mapping.shape == (data.signal.ndim, data.signal.ndim):
        source_volume = source_volume * abs(float(np.linalg.det(mapping)))
    if (
        symmetry is None
        and coordinate_ndim == source_ndim
        and mapping.shape == (source_ndim, source_ndim)
        and np.allclose(mapping, np.eye(source_ndim))
    ):
        covered_volume = source_coverage
        for dim, (axis, size, output_edges) in enumerate(
            zip(data.axes, data.shape, bins_list, strict=True)
        ):
            source_edges = _mdhisto_axis_edges(axis, size)
            output_edges = np.asarray(output_edges, dtype=float)
            overlap = np.maximum(
                0.0,
                np.minimum(output_edges[1:, None], source_edges[None, 1:])
                - np.maximum(output_edges[:-1, None], source_edges[None, :-1]),
            )
            covered_volume = np.tensordot(
                overlap,
                covered_volume,
                axes=(1, dim),
            )
            covered_volume = np.moveaxis(covered_volume, 0, dim)
        output_volume = _output_bin_volumes(bins_list)
        coverage = np.zeros(output_volume.shape, dtype=float)
        np.divide(
            covered_volume,
            output_volume,
            out=coverage,
            where=output_volume > 0.0,
        )
        return np.clip(coverage, 0.0, 1.0)
    weights = (source_volume * source_coverage).ravel()
    kwargs = dict(
        data_weights=weights,
        lower=[float(np.asarray(values)[0]) for values in bins_list],
        upper=[float(np.asarray(values)[-1]) for values in bins_list],
        **_rebin_grid_kwargs(config, list(axes_config)),
        # Coverage is geometric even when the signal reducer uses nearest-bin
        # assignment. Cloud-in-cell deposition avoids assigning an entire
        # rotated native voxel to whichever output bin contains its center.
        fractional=True,
        fractional_axes=[True] * source_ndim + [False] * (coordinate_ndim - source_ndim),
        normalize=False,
        mean_weighting="uniform",
        max_batch_bytes=_rebin_max_batch_bytes(config),
        max_parallel_bytes=_rebin_max_parallel_bytes(),
    )
    # Reuse the resolved edges, rather than interpreting them as centers again.
    kwargs["bin_edges"] = bins_list
    coverage_rebin = (
        rebin_nd_symmetry(
            np.ones(data.signal.size, dtype=float),
            np.asarray(coords, dtype=float).reshape(-1, coordinate_ndim),
            symmetry,
            axes=output_axes,
            **kwargs,
        )
        if symmetry is not None
        else rebin_nd(
            np.ones(data.signal.size, dtype=float),
            np.asarray(coords, dtype=float).reshape(-1, coordinate_ndim),
            axes=output_axes,
            **kwargs,
        )
    )
    if coverage_rebin.binned_data is None:
        raise RuntimeError("coverage rebinning did not produce binned data")
    spatial_volume = _output_bin_volumes(bins_list[:source_ndim])
    output_volume = np.broadcast_to(
        spatial_volume.reshape(spatial_volume.shape + (1,) * (coordinate_ndim - source_ndim)),
        tuple(len(values) - 1 for values in bins_list),
    )
    coverage = np.zeros(output_volume.shape, dtype=float)
    np.divide(
        np.asarray(coverage_rebin.binned_data, dtype=float),
        output_volume,
        out=coverage,
        where=output_volume > 0.0,
    )
    return np.clip(coverage, 0.0, 1.0)


def _rebin_mdhisto_data(
    data: MDHistoData,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData:
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "projecting source coordinates and applying masks",
            }
        )
    if any("metadata_dimension" in axis.metadata for axis in data.axes):
        raise ValueError(
            "Rebin the original collection to preserve its discrete metadata dimensions."
        )
    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    if len(axes_config) != len(data.axes):
        axes_config = _default_rebin_axes(data)

    ndim = len(data.axes)
    source_grids = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    if ndim == 4:
        transform = _mdhisto_rebin_basis_transform(data, axes_config)
        projected = []
        for output_index in range(ndim):
            component = np.zeros(data.shape, dtype=float)
            for source_grid, coefficient in zip(
                source_grids, transform[:, output_index], strict=True
            ):
                if coefficient:
                    component = component + float(coefficient) * source_grid
            projected.append(component)
    else:
        projected = [
            _mdhisto_rebin_component(data, list(source_grids), axis_config, index)
            for index, axis_config in enumerate(axes_config)
        ]
    coords = np.stack(projected, axis=-1)
    symmetry = _rebin_symmetry_matrices(config, data.metadata.get("lattice_parameters"))
    output_axes = None
    if symmetry is not None:
        if ndim != 4:
            raise ValueError("rebin symmetry requires a four-dimensional HKLE dataset")
        physical = _mdhisto_coordinate_grids(data)
        if not all(name in physical for name in ("H", "K", "L", "E")):
            raise ValueError(
                "rebin symmetry requires reconstructable H, K, L, and energy coordinates"
            )
        coords = np.stack([physical[name] for name in ("H", "K", "L", "E")], axis=-1)
        output_axes = _validate_mdhisto_rebin_basis(axes_config, ndim)
        data_bounds = _symmetry_projected_coordinate_bounds(coords, symmetry, output_axes)
    else:
        data_bounds = _finite_coordinate_bounds(coords)
    axes_config = _resolve_auto_rebin_axes(axes_config, data_bounds)
    valid = np.isfinite(data.signal) & np.isfinite(data.errors) & ~data.mask
    normalization_channel = data.auxiliary_channels.get("normalization_denominator")
    normalization_values = None
    if normalization_channel is not None:
        normalization_values = np.asarray(normalization_channel.values, dtype=float)
        valid &= np.isfinite(normalization_values) & (normalization_values > 0.0)
    if data.num_events is not None:
        valid &= mdhisto_measured_bins(data)
    signal = data.signal[valid]
    errors = data.errors[valid]
    coords_valid = coords[valid]
    if signal.size == 0:
        raise ValueError("no valid data points remain before rebinning")
    projected_valid = (
        coords_valid @ np.linalg.inv(output_axes) if output_axes is not None else coords_valid
    )
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        projected_valid,
        physical_coordinates=coords_valid if symmetry is not None else None,
        symmetry=symmetry,
        output_basis=output_axes,
    )
    axes_config = _resolve_data_driven_rebin_axes(config, axes_config, mode_coordinates)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    kwargs = dict(
        data_errs=errors,
        data_weights=(None if normalization_values is None else normalization_values[valid]),
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        max_parallel_bytes=_rebin_max_parallel_bytes(),
        progress_callback=progress_callback,
    )
    result = (
        rebin_nd_symmetry(signal, coords_valid, symmetry, axes=output_axes, **kwargs)
        if symmetry is not None
        else rebin_nd(signal, coords_valid, **kwargs)
    )
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None:
        raise RuntimeError("rebinning did not produce binned data")
    if result.bins_list is None:
        raise RuntimeError("rebinning did not produce bins")
    vectors = [
        _rebin_axis_vector(axis_config, index, ndim).tolist()
        for index, axis_config in enumerate(axes_config)
    ]
    rebinned_axes = tuple(
        MDHistoAxis(
            name=str(axis_config.get("name") or source_axis.name),
            values=np.asarray(bins, dtype=float),
            units=str(
                axis_config.get("units")
                if axis_config.get("units") is not None
                else source_axis.units
            ),
            kind=source_axis.kind,
            frame=source_axis.frame,
            path=source_axis.path,
            metadata={
                **dict(source_axis.metadata),
                "variable": str(axis_config.get("variable", "")),
                **(
                    {"discrete_centers": axis_config["resolved_centers"]}
                    if axis_config.get("resolved_centers") is not None
                    else {}
                ),
            },
        )
        for source_axis, axis_config, bins in zip(
            data.axes, axes_config, result.bins_list, strict=True
        )
    )
    mask = ~np.isfinite(result.binned_data) | ~np.isfinite(result.binned_data_errs)
    mask |= result.n_samples <= 0.0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_coverage",
                "message": "calculating geometric coverage for output bins",
            }
        )
    coverage = _rebin_mdhisto_coverage(
        data,
        coords,
        config,
        axes_config,
        result.bins_list,
        symmetry=symmetry,
        output_axes=output_axes,
    )
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata = dict(data.metadata)
    metadata["signal_semantics"] = "density"
    metadata["signal_semantics_source"] = "nfit_normalized_rebin"
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    metadata["rebin"] = {
        "lower": lower,
        "upper": upper,
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "bin_edges": [np.asarray(edges, dtype=float).tolist() for edges in result.bins_list],
        "vectors": vectors,
        "fractional_axes": _rebin_fractional_axes(config, axes_config),
        "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
        "normalize": True,
        "mean_weighting": _rebin_mean_weighting(config),
        "minimum_coverage": _rebin_minimum_coverage(config),
        "minimum_samples": _rebin_minimum_samples(config),
        "max_batch_mb": _rebin_max_batch_mb(config),
        "max_batch_bytes": _rebin_max_batch_bytes(config),
        "weighted_by_normalization_denominator": normalization_values is not None,
    }
    symmetry_metadata = _rebin_symmetry_metadata(config, data.metadata.get("lattice_parameters"))
    if symmetry_metadata is not None:
        metadata["rebin"]["symmetry"] = symmetry_metadata
    auxiliary_channels = {
        "coverage_fraction": MDHistoChannel(
            coverage,
            label="Coverage",
            unit="fraction",
        )
    }
    if (
        normalization_values is not None
        and _rebin_mean_weighting(config) == "uniform"
        and result._normalization is not None
    ):
        auxiliary_channels["normalization_denominator"] = MDHistoChannel(
            np.asarray(result._normalization, dtype=float),
            label="Combined detector-trajectory normalization",
            unit="arbitrary normalization units",
        )
    return MDHistoData(
        axes=rebinned_axes,
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=data.coordinate_system,
        visual_normalization=data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=auxiliary_channels,
    )


def _rebin_point_data(
    data: PointData4D,
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> MDHistoData:
    source = data.valid(require_positive_sigma=False)
    if source.size == 0:
        raise ValueError("no valid data points remain before rebinning")
    normalization_weighted = source.normalization_denominator is not None
    return _point_data_histogram(
        np.column_stack(source.coordinates()),
        np.asarray(source.intensity, dtype=float),
        np.asarray(source.sigma, dtype=float),
        config,
        data_weights=(
            None
            if source.normalization_denominator is None
            else np.asarray(source.normalization_denominator, dtype=float)
        ),
        source_metadata=data.metadata,
        coordinate_system=data.metadata.get("coordinate_system"),
        visual_normalization=data.metadata.get("visual_normalization"),
        metadata_updates={
            "weighted_by_normalization_denominator": normalization_weighted,
        },
        progress_callback=progress_callback,
    )


def _point_data_histogram(
    coordinates: np.ndarray,
    signal: np.ndarray,
    errors: np.ndarray,
    config: dict[str, Any],
    *,
    data_weights: np.ndarray | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    coordinate_system: int | None = None,
    visual_normalization: int | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> MDHistoData:
    """Bin physical HKLE plus optional metadata coordinates into a histogram."""

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_sources",
                "message": "projecting point coordinates and resolving per-axis grids",
            }
        )

    axes_config = [_sanitize_rebin_axis_config(axis) for axis in config.get("axes", [])]
    physical_coordinates = np.asarray(coordinates, dtype=float)
    if physical_coordinates.ndim != 2 or physical_coordinates.shape[1] < 4:
        raise ValueError("point-data rebinning requires HKLE coordinate columns")
    if len(axes_config) != physical_coordinates.shape[1]:
        raise ValueError("point-data rebin axes must match the coordinate columns")
    for index, axis_config in enumerate(axes_config[:4]):
        axis_config.setdefault("variable", ("H", "K", "L", "E")[index])
    physical_basis = _validate_mdhisto_rebin_basis(axes_config[:4], 4)
    basis = np.eye(physical_coordinates.shape[1], dtype=float)
    basis[:4, :4] = physical_basis
    projected_coordinates = physical_coordinates @ np.linalg.inv(basis)
    metadata = copy.deepcopy(dict(source_metadata or {}))
    symmetry = _rebin_symmetry_matrices(config, metadata.get("lattice_parameters"))
    data_bounds = (
        _symmetry_projected_coordinate_bounds(physical_coordinates, symmetry, basis)
        if symmetry is not None
        else _finite_coordinate_bounds(projected_coordinates)
    )
    axes_config = _resolve_auto_rebin_axes(axes_config, data_bounds)
    mode_coordinates = _axis_mode_coordinates_with_symmetry(
        config,
        projected_coordinates,
        physical_coordinates=physical_coordinates,
        symmetry=symmetry,
        output_basis=basis,
    )
    axes_config = _resolve_data_driven_rebin_axes(config, axes_config, mode_coordinates)
    lower = [axis["lower"] for axis in axes_config]
    upper = [axis["upper"] for axis in axes_config]
    kwargs = dict(
        data_errs=np.asarray(errors, dtype=float),
        data_weights=(None if data_weights is None else np.asarray(data_weights, dtype=float)),
        axes=basis,
        lower=lower,
        upper=upper,
        **_rebin_grid_kwargs(config, axes_config),
        fractional=bool(config.get("fractional", False)),
        fractional_axes=_rebin_fractional_axes(config, axes_config),
        normalize=True,
        mean_weighting=_rebin_mean_weighting(config),
        minimum_samples=_rebin_minimum_samples(config),
        max_batch_bytes=_rebin_max_batch_bytes(config),
        max_parallel_bytes=_rebin_max_parallel_bytes(),
        progress_callback=progress_callback,
    )
    result = (
        rebin_nd_symmetry(
            np.asarray(signal, dtype=float),
            physical_coordinates,
            symmetry,
            **kwargs,
        )
        if symmetry is not None
        else rebin_nd(
            np.asarray(signal, dtype=float),
            physical_coordinates,
            **kwargs,
        )
    )
    if (
        result.binned_data is None
        or result.binned_data_errs is None
        or result.n_samples is None
        or result.bins_list is None
    ):
        raise RuntimeError("point-data rebinning did not produce binned data")
    mask = (
        ~np.isfinite(result.binned_data)
        | ~np.isfinite(result.binned_data_errs)
        | (result.n_samples <= 0.0)
    )
    coverage = np.asarray(result.n_samples > 0.0, dtype=float)
    coverage_mask = coverage < _rebin_minimum_coverage(config)
    mask |= coverage_mask
    metadata.update(copy.deepcopy(dict(metadata_updates or {})))
    metadata["signal_semantics"] = "density"
    metadata["signal_semantics_source"] = "nfit_normalized_rebin"
    metadata["coverage_mask_count"] = int(np.count_nonzero(coverage_mask))
    metadata["rebin"] = {
        "lower": lower,
        "upper": upper,
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "bin_edges": [np.asarray(edges, dtype=float).tolist() for edges in result.bins_list],
        "vectors": basis.tolist(),
        "fractional_axes": _rebin_fractional_axes(config, axes_config),
        "axis_modes": [_rebin_axis_mode(config, axis) for axis in axes_config],
        "normalize": True,
        "mean_weighting": _rebin_mean_weighting(config),
        "minimum_coverage": _rebin_minimum_coverage(config),
        "minimum_samples": _rebin_minimum_samples(config),
        "max_batch_mb": _rebin_max_batch_mb(config),
        "max_batch_bytes": _rebin_max_batch_bytes(config),
    }
    for key in (
        "weighted_by_fit_weight",
        "weighted_by_normalization_denominator",
    ):
        if key in metadata:
            metadata["rebin"][key] = bool(metadata[key])
    symmetry_metadata = _rebin_symmetry_metadata(config, metadata.get("lattice_parameters"))
    if symmetry_metadata is not None:
        metadata["rebin"]["symmetry"] = symmetry_metadata
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "rebin_output",
                "message": "building the rebinned dataset and output channels",
            }
        )
    return MDHistoData(
        axes=tuple(
            MDHistoAxis(
                name=str(
                    axis_config.get("name")
                    or (("H", "K", "L", "DeltaE")[index] if index < 4 else f"Axis {index + 1}")
                ),
                values=np.asarray(edges, dtype=float),
                units=str(
                    axis_config.get("units")
                    or ("rlu" if index < 3 else "meV" if index == 3 else "")
                ),
                kind=(
                    "momentum"
                    if index < 3
                    else "energy_transfer"
                    if index == 3
                    else "unknown"
                ),
                frame="HKL" if index < 3 else "General Frame",
                metadata={
                    "variable": str(
                        axis_config.get(
                            "variable",
                            ("H", "K", "L", "E")[index]
                            if index < 4
                            else axis_config.get("name", f"Axis {index + 1}"),
                        )
                    ),
                    **(
                        {
                            "metadata_dimension": copy.deepcopy(
                                axis_config["metadata_dimension"]
                            ),
                            "interpolation": "none",
                        }
                        if axis_config.get("metadata_dimension") is not None
                        else {}
                    ),
                    **(
                        {"discrete_centers": axis_config["resolved_centers"]}
                        if axis_config.get("resolved_centers") is not None
                        else {}
                    ),
                },
            )
            for index, (axis_config, edges) in enumerate(
                zip(axes_config, result.bins_list, strict=True)
            )
        ),
        signal=np.asarray(result.binned_data, dtype=float),
        errors=np.asarray(result.binned_data_errs, dtype=float),
        mask=np.asarray(mask, dtype=bool),
        num_events=np.asarray(result.n_samples, dtype=float),
        coordinate_system=coordinate_system,
        visual_normalization=visual_normalization,
        metadata=metadata,
        auxiliary_channels={
            "coverage_fraction": MDHistoChannel(
                coverage,
                label="Coverage",
                unit="fraction",
            )
        },
    )
