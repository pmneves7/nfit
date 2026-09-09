"""Coordinate reconstruction and projection helpers for project data services."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoAxis, MDHistoData

COORDINATE_RANGE_AXIS_PREFIX = "axis_"
COORDINATE_RANGE_PARAMETER_NAMES = ("H", "K", "L", "E")


def _parse_parameter_text(text: str) -> Any:
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped


def _mdhisto_rebin_axis_vector(axis: MDHistoAxis, index: int, ndim: int) -> list[float]:
    """Return a default rebin vector that follows the displayed MDHisto axis."""

    if ndim == 4:
        role = axis.role
        role_vectors = {
            "h": [1.0, 0.0, 0.0, 0.0],
            "k": [0.0, 1.0, 0.0, 0.0],
            "l": [0.0, 0.0, 1.0, 0.0],
            "energy": [0.0, 0.0, 0.0, 1.0],
            "energy_transfer": [0.0, 0.0, 0.0, 1.0],
        }
        if role in role_vectors:
            return role_vectors[role]
        projection = _axis_projection_vector(axis.name)
        if projection is not None:
            return [_clean_axis_weight(value) for value in [*projection.tolist(), 0.0]]
    return _identity_vector(index, ndim)


def _mdhisto_rebin_source_axis_vectors(data: MDHistoData) -> list[np.ndarray | None]:
    vectors: list[np.ndarray | None] = []
    if len(data.axes) != 4:
        return [None for _axis in data.axes]
    for axis in data.axes:
        vectors.append(np.asarray(_mdhisto_rebin_axis_vector(axis, len(vectors), len(data.axes)), dtype=float))
    return vectors


def _resolve_projected_axis_grid(name: Any, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    """Resolve a projected axis name (or vector) to a coordinate grid."""

    if isinstance(name, str):
        key = name.strip()
        if key in coords:
            return coords[key]
        if key.upper() in coords:
            return coords[key.upper()]
        projection = _axis_projection_vector(key)
        if projection is not None:
            return _project_hkl_vector(projection, coords)
        parsed = _parse_parameter_text(key)
        if isinstance(parsed, (list, tuple)):
            return _project_hkle_vector(parsed, coords)
        return None
    if isinstance(name, (list, tuple, np.ndarray)):
        return _project_hkle_vector(name, coords)
    return None


def _project_hkl_vector(vector: np.ndarray, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    if not {"H", "K", "L"}.issubset(coords):
        return None
    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    return hkl @ np.asarray(vector, dtype=float)


def _project_hkle_vector(vector: Any, coords: dict[str, np.ndarray]) -> np.ndarray | None:
    try:
        weights = np.asarray(vector, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(weights)):
        return None
    if weights.shape == (3,):
        return _project_hkl_vector(weights, coords)
    if weights.shape == (4,) and {"H", "K", "L", "E"}.issubset(coords):
        hkle = np.stack([coords["H"], coords["K"], coords["L"], coords["E"]], axis=-1)
        return hkle @ weights
    return None


def _mdhisto_coordinate_range_axis_grids(
    data: MDHistoData,
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    specs = _coordinate_range_axis_specs(data)
    if not specs:
        return {}
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    source_vectors = _mdhisto_rebin_source_axis_vectors(data)
    coords: dict[str, np.ndarray] = {}
    for index, spec in enumerate(specs):
        key = f"{COORDINATE_RANGE_AXIS_PREFIX}{index}"
        vector = _coordinate_axis_vector(parameters.get(key), len(data.axes))
        if vector is None:
            vector = np.asarray(spec["vector"], dtype=float)
        matched_grid = next(
            (
                axis_grid
                for source_vector, axis_grid in zip(source_vectors, axis_values, strict=True)
                if source_vector is not None and np.allclose(vector, source_vector)
            ),
            None,
        )
        if matched_grid is not None:
            coords[str(spec["name"])] = np.asarray(matched_grid, dtype=float)
            continue
        values = np.zeros(data.shape, dtype=float)
        for axis_weight, axis_grid in zip(vector, axis_values, strict=True):
            if axis_weight:
                values = values + float(axis_weight) * axis_grid
        coords[str(spec["name"])] = values
    return coords


def _coordinate_range_axis_specs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, MDHistoData):
        return _mdhisto_coordinate_range_axis_specs(data)
    if isinstance(data, PointData4D):
        names = ["H", "K", "L", "E"]
        return [
            {"name": name, "vector": _identity_vector(index, len(names))}
            for index, name in enumerate(names)
        ]
    return []


def _mdhisto_coordinate_range_axis_specs(data: MDHistoData) -> list[dict[str, Any]]:
    ndim = len(data.axes)
    role_names = {"h": "H", "k": "K", "l": "L", "energy_transfer": "E"}
    assigned = {
        role_names[axis.role]
        for axis in data.axes
        if axis.role in role_names
    }
    remaining_names = iter(name for name in COORDINATE_RANGE_PARAMETER_NAMES if name not in assigned)
    specs = []
    for index, axis in enumerate(data.axes):
        name = role_names.get(axis.role)
        if name is None:
            name = next(remaining_names, str(axis.name or f"Axis {index}"))
        specs.append(
            {
                "name": name,
                "vector": _mdhisto_rebin_axis_vector(axis, index, ndim),
            }
        )
    return specs


def _identity_vector(index: int, ndim: int) -> list[float]:
    return [1.0 if axis_index == index else 0.0 for axis_index in range(ndim)]


def _clean_axis_weight(value: Any) -> float:
    """Round a coordinate-axis weight to drop floating-point noise (e.g. 0.5000000000000001 -> 0.5)."""

    rounded = round(float(value), 10)
    return rounded + 0.0


def _coordinate_axis_vector(value: Any, ndim: int) -> np.ndarray | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if not _is_vector_length(value, ndim):
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _is_vector_length(value: Any, length: int) -> bool:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == length


def _mdhisto_coordinate_grids(data: MDHistoData) -> dict[str, np.ndarray]:
    """Return physical HKLE grids, reconstructing them from any rebinned basis."""

    shape = data.shape
    axis_values = np.meshgrid(*(axis.centers for axis in data.axes), indexing="ij")
    coords: dict[str, np.ndarray] = {}
    hkle = np.zeros((*shape, 4), dtype=float)
    hkle_contributions = 0
    for index, (axis, values) in enumerate(zip(data.axes, axis_values, strict=True)):
        if "metadata_dimension" in axis.metadata:
            continue
        role = axis.role
        if role == "q_modulus":
            coords["q_modulus"] = values
            continue
        vector = _mdhisto_axis_coordinate_vector(data, index)
        if vector is None:
            vector = {
                "h": np.array([1.0, 0.0, 0.0, 0.0]),
                "k": np.array([0.0, 1.0, 0.0, 0.0]),
                "l": np.array([0.0, 0.0, 1.0, 0.0]),
                "energy_transfer": np.array([0.0, 0.0, 0.0, 1.0]),
            }.get(role)
        if vector is not None:
            hkle += values[..., np.newaxis] * vector
            hkle_contributions += 1
    if hkle_contributions:
        coords.update({"H": hkle[..., 0], "K": hkle[..., 1], "L": hkle[..., 2]})
        if len(data.axes) >= 4 or np.any(hkle[..., 3] != 0.0):
            coords["E"] = hkle[..., 3]
    return coords


def _mdhisto_axis_coordinate_vector(data: MDHistoData, index: int) -> np.ndarray | None:
    """Return an output-axis vector in physical HKLE coordinates when known."""

    rebin = data.metadata.get("rebin") if isinstance(data.metadata, dict) else None
    vectors = rebin.get("vectors") if isinstance(rebin, dict) else None
    if isinstance(vectors, (list, tuple)) and 0 <= index < len(vectors):
        try:
            vector = np.asarray(vectors[index], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            vector = np.asarray([], dtype=float)
        if vector.size in {3, 4} and np.all(np.isfinite(vector)):
            return np.pad(vector, (0, 4 - vector.size))
    projection = _axis_projection_vector(data.axes[index].name)
    if projection is not None:
        return np.pad(projection, (0, 1))
    return None


def _axis_projection_vector(name: str) -> np.ndarray | None:
    match = re.search(r"\[([^\]]+)\]", str(name))
    if match is None:
        return None
    pieces = [piece.strip() for piece in match.group(1).split(",")]
    if len(pieces) != 3:
        return None
    return np.asarray([_projection_piece_value(piece) for piece in pieces], dtype=float)


def _projection_piece_value(piece: str) -> float:
    cleaned = piece.replace(" ", "")
    if cleaned in {"", "0"}:
        return 0.0
    if cleaned in {"H", "K", "L"}:
        return 1.0
    if cleaned in {"-H", "-K", "-L"}:
        return -1.0
    cleaned = re.sub(r"[HKL]", "", cleaned)
    if cleaned in {"", "+"}:
        return 1.0
    if cleaned == "-":
        return -1.0
    return float(cleaned)
