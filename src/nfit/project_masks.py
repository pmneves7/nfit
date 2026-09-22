"""GUI-independent mask evaluation and application for project datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import PointData4D
from .mdhisto import MDHistoData
from .pipeline import DatasetEntry, MaskSpec
from .project_coordinates import (
    COORDINATE_RANGE_PARAMETER_NAMES,
    _mdhisto_coordinate_grids,
    _mdhisto_coordinate_range_axis_grids,
    _parse_parameter_text,
    _resolve_projected_axis_grid,
)


def _point_data_with_nfit_masks(
    dataset: DatasetEntry,
    data: PointData4D,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> PointData4D:
    reject = ~np.asarray(data.mask, dtype=bool)
    nfit_mask = _nfit_mask_for_point_data(dataset, data, extra_masks=extra_masks)
    combined_reject = reject | nfit_mask
    if not np.any(combined_reject) and np.all(np.asarray(data.mask, dtype=bool)):
        return data
    metadata = dict(data.metadata)
    metadata["nfit_mask_count"] = int(np.count_nonzero(nfit_mask))
    metadata["combined_mask_count"] = int(np.count_nonzero(combined_reject))
    temperature = data.temperature.copy() if isinstance(data.temperature, np.ndarray) else data.temperature
    return PointData4D(
        H=data.H.copy(),
        K=data.K.copy(),
        L=data.L.copy(),
        E=data.E.copy(),
        intensity=data.intensity.copy(),
        sigma=data.sigma.copy(),
        mask=~combined_reject,
        temperature=temperature,
        magnetic_field=None if data.magnetic_field is None else np.array(data.magnetic_field),
        metadata=metadata,
    )


def _nfit_mask_for_point_data(
    dataset: DatasetEntry,
    data: PointData4D,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> np.ndarray:
    combined = np.zeros(data.size, dtype=bool)
    for mask in [*(extra_masks or []), *dataset.masks]:
        if not mask.enabled:
            continue
        mask_values = _evaluate_point_data_mask(data, mask)
        if mask.invert:
            mask_values = ~mask_values
        if mask.additive:
            combined &= ~mask_values
        else:
            combined |= mask_values
    return combined


def _evaluate_point_data_mask(data: PointData4D, mask: MaskSpec) -> np.ndarray:
    if mask.type == "coordinate_range":
        return _point_data_coordinate_range_mask(data, mask.parameters)
    if mask.type == "energy_q_range":
        return _point_data_energy_q_range_mask(data, mask.parameters)
    if mask.type == "phonon_cone":
        return _point_data_phonon_cone_mask(data, mask.parameters)
    if mask.type == "box":
        return _point_data_box_mask(data, mask.parameters)
    if mask.type == "ellipsoid":
        return _point_data_ellipsoid_mask(data, mask.parameters)
    return np.zeros(data.size, dtype=bool)


def _point_data_coordinate_range_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.size, dtype=bool)
    active = False
    coords = _point_data_coordinate_values(data)
    for name in COORDINATE_RANGE_PARAMETER_NAMES:
        bounds = _exclusion_parameter_range(parameters.get(name))
        if bounds is None or name not in coords:
            continue
        active = True
        reject &= _values_in_range(coords[name], bounds)
    return reject if active else np.zeros(data.size, dtype=bool)


def _point_data_energy_q_range_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.size, dtype=bool)
    energy = _exclusion_parameter_range(parameters.get("energy"))
    q_modulus = _exclusion_parameter_range(parameters.get("q_modulus"))
    if energy is None and q_modulus is None:
        return np.zeros(data.size, dtype=bool)
    if energy is not None:
        reject &= _values_in_range(np.asarray(data.E, dtype=float), energy)
    if q_modulus is not None:
        reject &= _values_in_range(_point_data_q_modulus(data), q_modulus)
    return reject


def _point_data_box_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    resolved = _point_data_projected_region_inputs(data, parameters, extent_key="width")
    if resolved is None:
        return np.zeros(data.size, dtype=bool)
    values, center, width = resolved
    half_widths = [0.5 * value for value in width]
    if any(half_width <= 0.0 for half_width in half_widths):
        return np.zeros(data.size, dtype=bool)
    reject = np.ones(data.size, dtype=bool)
    for value, coordinate, half_width in zip(values, center, half_widths, strict=True):
        reject &= np.abs(value - coordinate) <= half_width
    return reject


def _point_data_ellipsoid_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    resolved = _point_data_projected_region_inputs(data, parameters, extent_key="radii")
    if resolved is None:
        return np.zeros(data.size, dtype=bool)
    values, center, radii = resolved
    if any(radius <= 0.0 for radius in radii):
        return np.zeros(data.size, dtype=bool)
    scaled_square = np.zeros(data.size, dtype=float)
    for value, coordinate, radius in zip(values, center, radii, strict=True):
        scaled = (value - coordinate) / radius
        scaled_square = scaled_square + scaled * scaled
    return scaled_square <= 1.0


def _point_data_projected_region_inputs(
    data: PointData4D,
    parameters: dict[str, Any],
    *,
    extent_key: str,
) -> tuple[list[np.ndarray], list[float], list[float]] | None:
    axes = _mask_axis_names(parameters)
    if not axes:
        return None
    center = _parameter_float_sequence(parameters.get("center"))
    extent = _parameter_float_sequence(parameters.get(extent_key))
    if center is None or extent is None:
        return None
    if not len(axes) == len(center) == len(extent):
        return None
    coords = _point_data_coordinate_values(data)
    values: list[np.ndarray] = []
    for name in axes:
        value = _resolve_projected_axis_grid(name, coords)
        if value is None:
            return None
        values.append(value)
    return values, center, extent


def _point_data_phonon_cone_mask(data: PointData4D, parameters: dict[str, Any]) -> np.ndarray:
    slope = _parameter_float(parameters.get("slope"))
    if slope is None or slope <= 0.0:
        return np.zeros(data.size, dtype=bool)
    centers = _coordinate_centers(parameters.get("center"), 3)
    if centers is None:
        return np.zeros(data.size, dtype=bool)
    radius = max(_parameter_float(parameters.get("radius")) or 0.0, 0.0)
    q_vectors = _point_data_q_vectors(data)
    cone_radius = np.abs(np.asarray(data.E, dtype=float)) / slope + radius
    masked = np.zeros(data.size, dtype=bool)
    matrix = _mdhisto_q_matrix(data.metadata)
    centers_q = centers if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata) else centers @ matrix.T
    for center_q in centers_q:
        masked |= np.linalg.norm(q_vectors - center_q, axis=-1) <= cone_radius
    return masked


def _point_data_coordinate_values(data: PointData4D) -> dict[str, np.ndarray]:
    return {
        "H": np.asarray(data.H, dtype=float),
        "K": np.asarray(data.K, dtype=float),
        "L": np.asarray(data.L, dtype=float),
        "E": np.asarray(data.E, dtype=float),
    }


def _point_data_q_vectors(data: PointData4D) -> np.ndarray:
    hkl = np.column_stack([data.H, data.K, data.L])
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        return hkl
    return hkl @ _mdhisto_q_matrix(data.metadata).T


def _point_data_q_modulus(data: PointData4D) -> np.ndarray:
    return np.linalg.norm(_point_data_q_vectors(data), axis=1)


def _mdhisto_with_nfit_masks(
    dataset: DatasetEntry,
    *,
    data: MDHistoData | None = None,
    extra_masks: list[MaskSpec] | None = None,
) -> MDHistoData:
    data = dataset.data if data is None else data
    if not isinstance(data, MDHistoData):
        raise TypeError("dataset does not contain MDHistoData")
    masks = tuple([*(extra_masks or []), *dataset.masks])
    if not any(mask.enabled for mask in masks):
        return _mdhisto_with_file_mask_only(data)
    file_mask = np.asarray(data.mask, dtype=bool)
    nfit_mask = _nfit_mask_for_mdhisto(dataset, data, extra_masks=extra_masks)
    combined_mask = file_mask if not np.any(nfit_mask) else file_mask | nfit_mask
    # ``MDHistoData`` owns immutable arrays.  Mark these newly evaluated masks
    # immutable before construction so they can also serve as metadata channels
    # without a second full-volume copy.
    nfit_mask.setflags(write=False)
    if combined_mask is not file_mask:
        combined_mask.setflags(write=False)
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask
    metadata["nfit_mask"] = nfit_mask
    metadata["nfit_mask_count"] = int(np.count_nonzero(nfit_mask))
    metadata["file_mask_count"] = int(np.count_nonzero(file_mask))
    metadata["combined_mask_count"] = int(np.count_nonzero(combined_mask))
    metadata.pop("mask_application_pending", None)
    return MDHistoData(
        axes=data.axes,
        signal=data.signal,
        errors=data.errors,
        mask=combined_mask,
        num_events=data.num_events,
        coordinate_system=data.coordinate_system,
        visual_normalization=data.visual_normalization,
        metadata=metadata,
        auxiliary_channels=data.auxiliary_channels,
    )


def _mdhisto_with_file_mask_only(data: MDHistoData) -> MDHistoData:
    """Return a view that exposes the imported file mask without a live mask channel."""

    file_mask = np.asarray(data.mask, dtype=bool)
    metadata = dict(data.metadata)
    metadata["file_mask"] = file_mask
    metadata.pop("nfit_mask", None)
    metadata["nfit_mask_count"] = 0
    metadata["file_mask_count"] = int(np.count_nonzero(file_mask))
    metadata["combined_mask_count"] = metadata["file_mask_count"]
    metadata.pop("mask_application_pending", None)
    return data.with_updates(metadata=metadata)


def _nfit_mask_for_mdhisto(
    dataset: DatasetEntry,
    data: MDHistoData,
    *,
    extra_masks: list[MaskSpec] | None = None,
) -> np.ndarray:
    combined = np.zeros(data.shape, dtype=bool)
    # Ancestor group masks apply first, then the dataset's own masks.
    for mask in [*(extra_masks or []), *dataset.masks]:
        if not mask.enabled:
            continue
        mask_values = _evaluate_mdhisto_mask(data, mask)
        if mask.invert:
            mask_values = ~mask_values
        if mask.additive:
            combined &= ~mask_values
        else:
            combined |= mask_values
    return combined


def _evaluate_mdhisto_mask(data: MDHistoData, mask: MaskSpec) -> np.ndarray:
    if mask.type == "coordinate_range":
        return _mdhisto_coordinate_range_mask(data, mask.parameters)
    if mask.type == "energy_q_range":
        return _mdhisto_energy_q_range_mask(data, mask.parameters)
    if mask.type == "phonon_cone":
        return _mdhisto_phonon_cone_mask(data, mask.parameters)
    if mask.type == "box":
        return _mdhisto_box_mask(data, mask.parameters)
    if mask.type == "ellipsoid":
        return _mdhisto_ellipsoid_mask(data, mask.parameters)
    return np.zeros(data.shape, dtype=bool)


def _mdhisto_box_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    """Mask a projected box, mirroring :func:`nfit.fitting.mask_out_box`.

    ``axes`` names the projected coordinates, ``center`` locates the box, and
    ``width`` gives the full extent along each axis. A point is masked when it
    falls inside the box along every listed axis; the zero-width default (see
    MASK_TYPE_DEFINITIONS) therefore masks nothing.
    """

    resolved = _mdhisto_projected_region_inputs(data, parameters, extent_key="width")
    if resolved is None:
        return np.zeros(data.shape, dtype=bool)
    grids, center, width = resolved
    half_widths = [0.5 * value for value in width]
    if any(half_width <= 0.0 for half_width in half_widths):
        return np.zeros(data.shape, dtype=bool)
    reject = np.ones(data.shape, dtype=bool)
    for grid, coordinate, half_width in zip(grids, center, half_widths, strict=True):
        reject &= np.abs(grid - coordinate) <= half_width
    return reject


def _mdhisto_ellipsoid_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    """Mask a projected ellipsoid, mirroring :func:`nfit.fitting.mask_out_ellipsoid`.

    ``axes`` names the projected coordinates, ``center`` locates the ellipsoid,
    and ``radii`` gives the radius along each axis. The zero-radius default
    masks nothing.
    """

    resolved = _mdhisto_projected_region_inputs(data, parameters, extent_key="radii")
    if resolved is None:
        return np.zeros(data.shape, dtype=bool)
    grids, center, radii = resolved
    if any(radius <= 0.0 for radius in radii):
        return np.zeros(data.shape, dtype=bool)
    scaled_square = np.zeros(data.shape, dtype=float)
    for grid, coordinate, radius in zip(grids, center, radii, strict=True):
        scaled = (grid - coordinate) / radius
        scaled_square = scaled_square + scaled * scaled
    return scaled_square <= 1.0


def _mdhisto_projected_region_inputs(
    data: MDHistoData,
    parameters: dict[str, Any],
    *,
    extent_key: str,
) -> tuple[list[np.ndarray], list[float], list[float]] | None:
    """Resolve the projected axis grids, center, and extent for box/ellipsoid masks.

    Returns ``None`` when the parameters are incomplete, mismatched in length,
    or name coordinates that cannot be projected onto the dataset axes, so the
    caller can fall back to masking nothing.
    """

    axes = _mask_axis_names(parameters)
    if not axes:
        return None
    center = _parameter_float_sequence(parameters.get("center"))
    extent = _parameter_float_sequence(parameters.get(extent_key))
    if center is None or extent is None:
        return None
    if not len(axes) == len(center) == len(extent):
        return None
    # Range-axis grids supply projected/leftover axis names, but the true
    # reciprocal coordinates (H/K/L/E) must win when a name refers to them.
    coords = _mdhisto_coordinate_range_axis_grids(data, parameters)
    coords.update(_mdhisto_coordinate_grids(data))
    grids: list[np.ndarray] = []
    for name in axes:
        grid = _resolve_projected_axis_grid(name, coords)
        if grid is None:
            return None
        grids.append(grid)
    return grids, center, extent


def _mask_axis_names(parameters: dict[str, Any]) -> list[Any]:
    axes = parameters.get("axes")
    if isinstance(axes, str):
        axes = _parse_parameter_text(axes)
    if not isinstance(axes, (list, tuple)):
        return []
    return list(axes)


def _parameter_float_sequence(value: Any) -> list[float] | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    result: list[float] = []
    for item in value:
        number = _parameter_float(item)
        if number is None:
            return None
        result.append(number)
    return result


def _mdhisto_phonon_cone_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    slope = _parameter_float(parameters.get("slope"))
    if slope is None or slope <= 0.0:
        # A non-positive slope leaves the starter mask inert (see MASK_TYPE_DEFINITIONS).
        return np.zeros(data.shape, dtype=bool)
    centers = _coordinate_centers(parameters.get("center"), 3)
    if centers is None:
        return np.zeros(data.shape, dtype=bool)
    coords = _mdhisto_coordinate_grids(data)
    if not {"H", "K", "L", "E"}.issubset(coords):
        return np.zeros(data.shape, dtype=bool)
    radius = _parameter_float(parameters.get("radius")) or 0.0
    radius = max(radius, 0.0)

    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        q_vectors = hkl
        centers_q = centers
    else:
        matrix = _mdhisto_q_matrix(data.metadata)
        q_vectors = np.einsum("ij,...j->...i", matrix, hkl)
        centers_q = centers @ matrix.T

    cone_radius = np.abs(coords["E"]) / slope + radius
    masked = np.zeros(data.shape, dtype=bool)
    for center_q in centers_q:
        masked |= np.linalg.norm(q_vectors - center_q, axis=-1) <= cone_radius
    return masked


def _coordinate_centers(value: Any, ndim: int) -> np.ndarray | None:
    """Parse one coordinate vector or a nonempty list of coordinate vectors."""

    if isinstance(value, str):
        value = _parse_parameter_text(value)
    try:
        centers = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if centers.shape == (ndim,):
        centers = centers.reshape(1, ndim)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != ndim:
        return None
    return centers if np.all(np.isfinite(centers)) else None


def _parameter_float(value: Any) -> float | None:
    if isinstance(value, str):
        value = _parse_parameter_text(value)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mdhisto_coordinate_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.shape, dtype=bool)
    active = False
    coords = _mdhisto_coordinate_grids(data)
    coords.update(_mdhisto_coordinate_range_axis_grids(data, parameters))
    for name in COORDINATE_RANGE_PARAMETER_NAMES:
        bounds = _exclusion_parameter_range(parameters.get(name))
        if bounds is None or name not in coords:
            continue
        active = True
        reject &= _values_in_range(coords[name], bounds)
    return reject if active else np.zeros(data.shape, dtype=bool)


def _mdhisto_energy_q_range_mask(data: MDHistoData, parameters: dict[str, Any]) -> np.ndarray:
    reject = np.ones(data.shape, dtype=bool)
    energy = _exclusion_parameter_range(parameters.get("energy"))
    q_modulus = _exclusion_parameter_range(parameters.get("q_modulus"))
    coords = _mdhisto_coordinate_grids(data)
    if energy is None and q_modulus is None:
        return np.zeros(data.shape, dtype=bool)
    if energy is not None:
        if "E" not in coords:
            if not _range_is_unrestricted(energy):
                return np.zeros(data.shape, dtype=bool)
        else:
            reject &= _values_in_range(coords["E"], energy)
    if q_modulus is not None:
        q_values = coords.get("q_modulus")
        if q_values is None:
            q_values = _mdhisto_q_modulus_grid(data, coords)
        reject &= _values_in_range(q_values, q_modulus)
    return reject


def _parameter_range(value: Any) -> tuple[float | None, float | None] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = _parse_parameter_text(value)
        if value in (None, ""):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lower = None if value[0] in (None, "") else float(value[0])
    upper = None if value[1] in (None, "") else float(value[1])
    return lower, upper


def _exclusion_parameter_range(value: Any) -> tuple[float | None, float | None] | None:
    """Return an active exclusion interval; [0, 0] is the neutral GUI default."""

    bounds = _parameter_range(value)
    if bounds is None:
        return None
    if bounds[0] == 0.0 and bounds[1] == 0.0:
        return None
    return bounds


def _values_in_range(values: np.ndarray, bounds: tuple[float | None, float | None]) -> np.ndarray:
    lower, upper = bounds
    selected = np.ones(values.shape, dtype=bool)
    if lower is not None:
        selected &= values >= lower - _range_tolerance(lower)
    if upper is not None:
        selected &= values <= upper + _range_tolerance(upper)
    return selected


def _range_is_unrestricted(bounds: tuple[float | None, float | None]) -> bool:
    lower, upper = bounds
    return lower is None and upper is None


def _range_tolerance(value: float) -> float:
    return max(1.0, abs(float(value))) * 1.0e-12


def _mdhisto_q_modulus_grid(data: MDHistoData, coords: dict[str, np.ndarray]) -> np.ndarray:
    if not {"H", "K", "L"}.issubset(coords):
        raise ValueError("|Q| masks require H, K, and L coordinates")
    hkl = np.stack([coords["H"], coords["K"], coords["L"]], axis=-1)
    if _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(data.metadata):
        q_vectors = hkl
    else:
        matrix = _mdhisto_q_matrix(data.metadata)
        q_vectors = np.einsum("ij,...j->...i", matrix, hkl)
    return np.linalg.norm(q_vectors, axis=-1)


def _metadata_coordinate_units_are_inv_angstrom_for_mdhisto(metadata: dict[str, Any]) -> bool:
    candidates = [metadata.get("coordinate_units"), metadata.get("momentum_units"), metadata.get("q_units")]
    return any("angstrom" in str(value).lower() and "rlu" not in str(value).lower() for value in candidates if value)


def _mdhisto_q_matrix(metadata: dict[str, Any]) -> np.ndarray:
    for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
        if key in metadata:
            matrix = np.asarray(metadata[key], dtype=float)
            return matrix if key == "rlu_to_inv_angstrom_matrix" or _matrix_includes_2pi(metadata, key) else 2.0 * np.pi * matrix
    oriented_lattice = metadata.get("oriented_lattice")
    if isinstance(oriented_lattice, dict):
        for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
            if key in oriented_lattice:
                matrix = np.asarray(oriented_lattice[key], dtype=float)
                return matrix if key == "rlu_to_inv_angstrom_matrix" or _matrix_includes_2pi(oriented_lattice, key) else 2.0 * np.pi * matrix
    lattice = metadata.get("lattice_parameters")
    if isinstance(lattice, dict):
        from .fitting import reciprocal_basis_from_lattice_parameters

        return reciprocal_basis_from_lattice_parameters(
            a=float(lattice["a"]),
            b=float(lattice["b"]),
            c=float(lattice["c"]),
            alpha=float(lattice.get("alpha", 90.0)),
            beta=float(lattice.get("beta", 90.0)),
            gamma=float(lattice.get("gamma", 90.0)),
            include_2pi=bool(lattice.get("include_2pi", True)),
        )
    raise ValueError("|Q| masks require inverse-angstrom coordinates or an attached lattice/UB matrix")


def _matrix_includes_2pi(metadata: dict[str, Any], key: str) -> bool:
    for flag_key in (f"{key}_includes_2pi", "q_matrix_includes_2pi", "include_2pi", "includes_2pi"):
        if flag_key in metadata:
            return bool(metadata[flag_key])
    lattice = metadata.get("lattice_parameters")
    return bool(isinstance(lattice, dict) and lattice.get("include_2pi"))
