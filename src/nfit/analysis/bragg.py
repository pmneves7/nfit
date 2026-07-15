from __future__ import annotations

from itertools import product

import numpy as np

from ..dataset import PointListData
from ..mdhisto import MDHistoData
from .coordinates import (
    bin_edges,
    measured_mask,
    physical_axis_vectors,
    q_bin_volume,
    rlu_to_q_matrix,
    signal_semantics,
)


def generate_bragg_peaks(
    data: MDHistoData,
    spacegroup: str,
    *,
    include_systematic_absences: bool = False,
    d_min_angstrom: float | None = None,
    d_max_angstrom: float | None = None,
) -> np.ndarray:
    """Generate deterministic integer HKLs intersecting direct-axis data bounds."""

    import gemmi

    vectors = physical_axis_vectors(data)[:, :3]
    if data.signal.ndim != 3 or abs(float(np.linalg.det(vectors))) <= 1e-12:
        raise ValueError("automatic peak generation requires three independent HKL projections")
    coordinate_bounds = [(float(bin_edges(axis, size)[0]), float(bin_edges(axis, size)[-1])) for axis, size in zip(data.axes, data.shape, strict=True)]
    corners = np.asarray(
        [np.asarray(values) @ vectors for values in product(*[(lo, hi) for lo, hi in coordinate_bounds])]
    )
    bounds = list(zip(corners.min(axis=0), corners.max(axis=0), strict=True))
    ranges = [range(int(np.floor(lo)), int(np.ceil(hi)) + 1) for lo, hi in bounds]
    group = gemmi.find_spacegroup_by_name(spacegroup)
    if group is None:
        raise ValueError(f"unknown space group {spacegroup!r}")
    operations = group.operations()
    qmatrix = rlu_to_q_matrix(data.metadata)
    peaks = []
    for h in ranges[0]:
        for k in ranges[1]:
            for l in ranges[2]:
                if (h, k, l) == (0, 0, 0):
                    continue
                if not include_systematic_absences and operations.is_systematically_absent([h, k, l]):
                    continue
                q = float(np.linalg.norm(qmatrix @ np.array([h, k, l])))
                d = 2 * np.pi / q if q > 0 else np.inf
                if d_min_angstrom is not None and d < d_min_angstrom:
                    continue
                if d_max_angstrom is not None and d > d_max_angstrom:
                    continue
                peaks.append((q, h, k, l))
    peaks.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return np.asarray([[h, k, l] for _q, h, k, l in peaks], dtype=float).reshape(-1, 3)


def integrate_bragg_peaks(
    data: MDHistoData,
    peaks_hkl: np.ndarray,
    *,
    method: str = "box_sum",
    box_half_widths=(0.1, 0.1, 0.1),
    ellipsoid_semiaxes=(0.1, 0.1, 0.1),
    ellipsoid_rotation=None,
    subvoxel_samples: int = 3,
    gaussian_max_nfev: int = 1000,
    energy_min_meV: float | None = None,
    energy_max_meV: float | None = None,
    background_mode: str = "none",
    background_inner_scale: float = 1.5,
    background_outer_scale: float = 2.0,
    minimum_peak_coverage: float = 0.9,
    minimum_background_coverage: float = 0.7,
    edge_policy: str = "reject",
    exclude_neighbor_regions: bool = True,
    center_mode: str = "nominal",
    centroid_search_radius: float = 0.1,
    coordinate_frame: str = "hkl",
    progress_callback=None,
    cancel_callback=None,
    gaussian_background: str = "constant",
) -> PointListData:
    """Integrate axis-aligned HKL boxes with optional surrounding shell background."""

    if data.signal.ndim == 4:
        if energy_min_meV is None or energy_max_meV is None:
            raise ValueError("4D Bragg integration requires an explicit elastic energy window")
        data = _elastic_reduce(data, energy_min_meV, energy_max_meV)
    if data.signal.ndim != 3 or len([axis for axis in data.axes if axis.kind == "momentum"]) != 3:
        raise ValueError("Bragg box integration requires a three-dimensional momentum histogram")
    semantics = signal_semantics(data)
    if semantics == "unknown":
        raise ValueError("Bragg integration requires signal_semantics metadata")
    vectors = physical_axis_vectors(data)[:, :3]
    if abs(float(np.linalg.det(vectors))) <= 1e-12:
        raise ValueError("momentum axes must be independent linear HKL projections")
    components = []
    direct_axes = True
    for vector in vectors:
        nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
        if len(nonzero) != 1 or not np.isclose(vector[nonzero[0]], 1.0):
            direct_axes = False
            components.append(0)
            continue
        components.append(int(nonzero[0]))
    direct_axes = direct_axes and sorted(components) == [0, 1, 2]
    if method not in {"box_sum", "ellipsoid_sum", "gaussian_fit"}:
        raise ValueError("unknown Bragg integration method")
    half = np.asarray(box_half_widths if method == "box_sum" else ellipsoid_semiaxes, dtype=float)
    if half.shape != (3,) or np.any(half <= 0):
        raise ValueError("box_half_widths must contain three positive values")
    if background_mode not in {"none", "shell"}:
        raise ValueError("background_mode must be 'none' or 'shell'")
    if background_mode == "shell" and not (1 <= background_inner_scale < background_outer_scale):
        raise ValueError("background scales must satisfy 1 <= inner < outer")
    if edge_policy not in {"reject", "report_partial"}:
        raise ValueError("edge_policy must be 'reject' or 'report_partial'")
    if center_mode not in {"nominal", "centroid"}:
        raise ValueError("center_mode must be 'nominal' or 'centroid'")
    if coordinate_frame not in {"hkl", "q_angstrom_inverse"}:
        raise ValueError("coordinate_frame must be 'hkl' or 'q_angstrom_inverse'")
    if ellipsoid_rotation is None:
        ellipsoid_rotation = np.eye(3)
    qmatrix = rlu_to_q_matrix(data.metadata)
    all_edges = [bin_edges(axis, size) for axis, size in zip(data.axes, data.shape, strict=True)]
    all_measured = measured_mask(data)
    all_qvolume = q_bin_volume(data)
    rows = []
    peak_array = np.asarray(peaks_hkl, dtype=float).reshape(-1, 3)
    for peak_index, peak in enumerate(peak_array):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Bragg integration cancelled")
        if progress_callback is not None:
            progress_callback({"stage": "bragg_integration", "completed": peak_index, "total": len(peak_array)})
        if center_mode == "centroid":
            centroid_slices = _region_slices(all_edges, vectors, peak, np.full(3, centroid_search_radius), np.eye(3), 1.0, coordinate_frame, qmatrix, "ellipsoid_sum")
            centroid_edges = [edge[selection.start:selection.stop + 1] for edge, selection in zip(all_edges, centroid_slices, strict=True)]
            centroid_data = _slice_mdhisto(data, centroid_slices, all_edges)
            peak = _centroid_center(centroid_data, centroid_edges, vectors, peak, centroid_search_radius, all_measured[centroid_slices], qmatrix if coordinate_frame != "hkl" else None)
        extent_scale = 3.0 if method == "gaussian_fit" else (background_outer_scale if background_mode == "shell" else 1.0)
        slices = _region_slices(all_edges, vectors, peak, half, ellipsoid_rotation, extent_scale, coordinate_frame, qmatrix, method)
        local = _slice_mdhisto(data, slices, all_edges)
        edges = [edge[selection.start:selection.stop + 1] for edge, selection in zip(all_edges, slices, strict=True)]
        measured = all_measured[slices]
        qvolume = all_qvolume[slices]
        if method == "gaussian_fit":
            rows.append(_fit_gaussian(local, edges, vectors, peak, half, measured, minimum_peak_coverage, gaussian_max_nfev, coordinate_frame, gaussian_background))
            continue
        def region_fraction(scale, region_edges=edges, region_peak=peak):
            if coordinate_frame == "hkl" and direct_axes:
                if method == "box_sum":
                    return _box_fraction(region_edges, components, region_peak, half * scale)
                return _ellipsoid_fraction(
                    region_edges,
                    components,
                    region_peak,
                    half * scale,
                    ellipsoid_rotation,
                    subvoxel_samples,
                )
            return _sample_region_fraction(
                region_edges,
                vectors,
                region_peak,
                half * scale,
                ellipsoid_rotation,
                subvoxel_samples,
                None if coordinate_frame == "hkl" else qmatrix,
                method,
            )
        peak_fraction = region_fraction(1.0)
        peak_total = _integration_weights(data, peak_fraction, qvolume, semantics)
        peak_measured = np.where(measured, peak_total, 0.0)
        peak_volume = float(np.sum(peak_measured))
        geometric_peak_volume = float(np.sum(peak_total))
        coverage = peak_volume / geometric_peak_volume if geometric_peak_volume > 0 else 0.0
        raw = float(np.sum(local.signal * peak_measured))
        raw_var = float(np.sum((local.errors * peak_measured) ** 2))
        background = 0.0
        background_var = 0.0
        background_coverage = 1.0
        if background_mode == "shell":
            outer = region_fraction(background_outer_scale)
            inner = region_fraction(background_inner_scale)
            shell_fraction = np.clip(outer - inner, 0.0, 1.0)
            if exclude_neighbor_regions:
                for neighbor in np.asarray(peaks_hkl, dtype=float).reshape(-1, 3):
                    if np.allclose(neighbor, peak):
                        continue
                    neighbor_fraction = (
                        (_box_fraction(edges, components, neighbor, half) if method == "box_sum" else _ellipsoid_fraction(edges, components, neighbor, half, ellipsoid_rotation, subvoxel_samples))
                        if coordinate_frame == "hkl" and direct_axes
                        else _sample_region_fraction(edges, vectors, neighbor, half, ellipsoid_rotation, subvoxel_samples, None if coordinate_frame == "hkl" else qmatrix, method)
                    )
                    shell_fraction *= 1.0 - neighbor_fraction
            shell_total = _integration_weights(data, shell_fraction, qvolume, semantics)
            shell_measured = np.where(measured, shell_total, 0.0)
            shell_volume = float(np.sum(shell_measured))
            geometric_shell_volume = float(np.sum(shell_total))
            background_coverage = shell_volume / geometric_shell_volume if geometric_shell_volume > 0 else 0.0
            if shell_volume > 0:
                shell_sum = float(np.sum(local.signal * shell_measured))
                shell_var = float(np.sum((local.errors * shell_measured) ** 2))
                ratio = peak_volume / shell_volume
                background = shell_sum * ratio
                background_var = shell_var * ratio**2
        intensity = raw - background
        sigma = np.sqrt(raw_var + background_var)
        accepted = coverage >= minimum_peak_coverage and background_coverage >= minimum_background_coverage
        status = 0.0 if accepted else 1.0
        if not accepted and edge_policy == "reject":
            intensity = sigma = np.nan
        rows.append((*peak, intensity, sigma, background, intensity / sigma if sigma > 0 else np.nan, coverage, background_coverage, status))
    if progress_callback is not None:
        progress_callback({"stage": "bragg_integration", "completed": len(peak_array), "total": len(peak_array)})
    columns = dict(zip(("H", "K", "L", "I", "dI", "Background", "I/dI", "Coverage", "BackgroundCoverage", "Status"), np.asarray(rows, dtype=float).T, strict=True))
    return PointListData(columns, coordinate_names=["H", "K", "L"], channels=[{"label": "Integrated intensity", "value": "I", "error": "dI"}, {"label": "Background", "value": "Background", "error": None}, {"label": "I/dI", "value": "I/dI", "error": None}], metadata={"method": method, "background_mode": background_mode})


def _box_fraction(edges, components, center, half):
    factors = []
    for axis_edges, component in zip(edges, components, strict=True):
        overlap = np.maximum(0.0, np.minimum(axis_edges[1:], center[component] + half[component]) - np.maximum(axis_edges[:-1], center[component] - half[component]))
        factors.append(overlap / np.diff(axis_edges))
    return factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]


def _integration_weights(data, fraction, qvolume, semantics):
    return fraction * qvolume if semantics == "density" else fraction


def _ellipsoid_fraction(edges, components, center, semiaxes, rotation, samples):
    if samples < 1 or samples % 2 != 1:
        raise ValueError("subvoxel_samples must be a positive odd integer")
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("ellipsoid_rotation must be an orthonormal 3x3 matrix")
    coordinates = []
    offsets = (np.arange(samples) + 0.5) / samples
    for axis_edges in edges:
        coordinates.append(axis_edges[:-1, None] + np.diff(axis_edges)[:, None] * offsets)
    fraction = np.zeros(tuple(len(edge) - 1 for edge in edges), dtype=float)
    for i in range(samples):
        for j in range(samples):
            for k in range(samples):
                physical = np.zeros((*fraction.shape, 3), dtype=float)
                physical[..., components[0]] = coordinates[0][:, i, None, None]
                physical[..., components[1]] = coordinates[1][None, :, j, None]
                physical[..., components[2]] = coordinates[2][None, None, :, k]
                normalized = (physical - center) @ rotation / semiaxes
                fraction += np.sum(normalized**2, axis=-1) <= 1.0
    return fraction / samples**3


def _fit_gaussian(data, edges, vectors, center, initial_sigma, measured, minimum_coverage, max_nfev, coordinate_frame, background_mode):
    from scipy.optimize import least_squares

    if signal_semantics(data) != "density":
        raise ValueError("gaussian_fit requires density-valued signal")
    centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    grids = np.meshgrid(*centers, indexing="ij")
    xyz = np.zeros((*data.shape, 3), dtype=float)
    for grid, vector in zip(grids, vectors, strict=True):
        xyz += grid[..., None] * vector
    qmatrix = rlu_to_q_matrix(data.metadata)
    fit_xyz = xyz if coordinate_frame == "hkl" else xyz @ qmatrix.T
    fit_center = center if coordinate_frame == "hkl" else center @ qmatrix.T
    region = np.all(np.abs(fit_xyz - fit_center) <= 3.0 * initial_sigma, axis=-1)
    valid = region & measured & np.isfinite(data.signal) & np.isfinite(data.errors) & (data.errors > 0)
    coverage = float(np.count_nonzero(valid) / max(np.count_nonzero(region), 1))
    if np.count_nonzero(valid) < 8:
        return (*center, np.nan, np.nan, np.nan, np.nan, coverage, 1.0, 2.0)
    x = fit_xyz[valid]
    y = data.signal[valid]
    error = data.errors[valid]
    background0 = float(np.nanmedian(y))
    amplitude0 = max(float(np.nanmax(y) - background0), np.finfo(float).eps)
    if background_mode not in {"constant", "linear"}:
        raise ValueError("gaussian_background must be 'constant' or 'linear'")
    p0 = np.r_[amplitude0, fit_center, np.log(initial_sigma), background0]
    if background_mode == "linear":
        p0 = np.r_[p0, np.zeros(3)]

    def residual(p):
        amplitude, fitted_center, log_sigma, background = p[0], p[1:4], p[4:7], p[7]
        local_background = background
        if background_mode == "linear":
            local_background = local_background + (x - fitted_center) @ p[8:11]
        model = local_background + amplitude * np.exp(-0.5 * np.sum(((x - fitted_center) / np.exp(log_sigma)) ** 2, axis=1))
        return (model - y) / error

    lower = np.r_[0.0, fit_center - 2 * initial_sigma, np.log(initial_sigma / 10), -np.inf]
    upper = np.r_[np.inf, fit_center + 2 * initial_sigma, np.log(initial_sigma * 10), np.inf]
    if background_mode == "linear":
        lower = np.r_[lower, [-np.inf] * 3]
        upper = np.r_[upper, [np.inf] * 3]
    fit = least_squares(residual, p0, bounds=(lower, upper), max_nfev=int(max_nfev))
    amplitude = fit.x[0]
    sigma = np.exp(fit.x[4:7])
    jacobian = abs(float(np.linalg.det(qmatrix))) if coordinate_frame == "hkl" else 1.0
    gaussian_volume = (2 * np.pi) ** 1.5 * float(np.prod(sigma)) * jacobian
    intensity = amplitude * gaussian_volume
    background = fit.x[7] * gaussian_volume
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    dof = max(fit.fun.size - fit.x.size, 1)
    covariance *= float(np.sum(fit.fun**2) / dof)
    gradient = np.zeros(fit.x.size)
    gradient[0] = gaussian_volume
    gradient[4:7] = intensity
    uncertainty = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
    status = 0.0 if fit.success and coverage >= minimum_coverage else 3.0
    fitted_center = fit.x[1:4] if coordinate_frame == "hkl" else fit.x[1:4] @ np.linalg.inv(qmatrix).T
    return (*fitted_center, intensity, uncertainty, background, intensity / uncertainty if uncertainty > 0 else np.nan, coverage, 1.0, status)


def _elastic_reduce(data: MDHistoData, lower: float, upper: float) -> MDHistoData:
    dimensions = [i for i, axis in enumerate(data.axes) if axis.kind == "energy" or axis.role == "energy_transfer"]
    if len(dimensions) != 1 or upper <= lower:
        raise ValueError("invalid elastic energy axis or window")
    dim = dimensions[0]
    edges = bin_edges(data.axes[dim], data.shape[dim])
    overlaps = np.maximum(0.0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
    fractions = overlaps / np.diff(edges)
    shape = [1] * data.signal.ndim
    shape[dim] = data.shape[dim]
    weights = overlaps if signal_semantics(data) == "density" else fractions
    weights = weights.reshape(shape)
    valid = measured_mask(data) & (weights > 0)
    signal = np.sum(np.where(valid, data.signal * weights, 0.0), axis=dim)
    errors = np.sqrt(np.sum(np.where(valid, (data.errors * weights) ** 2, 0.0), axis=dim))
    requested = upper - lower
    coverage = np.sum(np.where(valid, overlaps.reshape(shape), 0.0), axis=dim) / requested
    axes = tuple(axis for i, axis in enumerate(data.axes) if i != dim)
    metadata = {**data.metadata, "elastic_energy_window_meV": [lower, upper], "energy_coverage": coverage.tolist()}
    return MDHistoData(axes, signal, errors, coverage <= 0, np.where(coverage > 0, 1.0, 0.0), coordinate_system=data.coordinate_system, visual_normalization=data.visual_normalization, metadata=metadata)


def _centroid_center(data, edges, vectors, center, radius, measured, transform=None):
    if radius <= 0:
        raise ValueError("centroid_search_radius must be positive")
    centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    grids = np.meshgrid(*centers, indexing="ij")
    xyz = np.zeros((*data.shape, 3), dtype=float)
    for grid, vector in zip(grids, vectors, strict=True):
        xyz += grid[..., None] * vector
    selection_xyz = xyz if transform is None else xyz @ transform.T
    selection_center = center if transform is None else center @ transform.T
    region = np.sum(((selection_xyz - selection_center) / radius) ** 2, axis=-1) <= 1.0
    valid = region & measured & np.isfinite(data.signal)
    if not np.any(valid):
        return center
    baseline = float(np.nanmin(data.signal[valid]))
    weights = np.maximum(data.signal[valid] - baseline, 0.0)
    if np.sum(weights) <= 0:
        return center
    return np.sum(xyz[valid] * weights[:, None], axis=0) / np.sum(weights)


def _sample_region_fraction(edges, vectors, center, widths, rotation, samples, qmatrix, method):
    if samples < 1 or samples % 2 != 1:
        raise ValueError("subvoxel_samples must be a positive odd integer")
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("ellipsoid_rotation must be an orthonormal 3x3 matrix")
    coordinates = []
    offsets = (np.arange(samples) + 0.5) / samples
    for axis_edges in edges:
        coordinates.append(axis_edges[:-1, None] + np.diff(axis_edges)[:, None] * offsets)
    shape = tuple(len(edge) - 1 for edge in edges)
    fraction = np.zeros(shape, dtype=float)
    for i in range(samples):
        for j in range(samples):
            for k in range(samples):
                hkl = (
                    coordinates[0][:, i, None, None, None] * vectors[0]
                    + coordinates[1][None, :, j, None, None] * vectors[1]
                    + coordinates[2][None, None, :, k, None] * vectors[2]
                )
                delta_hkl = hkl - center
                delta = delta_hkl if qmatrix is None else delta_hkl @ qmatrix.T
                if method == "box_sum":
                    inside = np.all(np.abs(delta) <= widths, axis=-1)
                else:
                    inside = np.sum(((delta @ rotation) / widths) ** 2, axis=-1) <= 1.0
                fraction += inside
    return fraction / samples**3


def _region_slices(edges, vectors, center, widths, rotation, scale, coordinate_frame, qmatrix, method):
    inverse_vectors = np.linalg.inv(np.asarray(vectors, dtype=float))
    center_coordinates = np.asarray(center) @ inverse_vectors
    frame_extent = np.asarray(widths, dtype=float) * scale
    if method != "box_sum":
        frame_extent = frame_extent @ np.abs(np.asarray(rotation, dtype=float).T)
    if coordinate_frame == "hkl":
        coordinate_extent = frame_extent @ np.abs(inverse_vectors)
    else:
        coordinate_extent = frame_extent @ np.abs(np.linalg.inv(qmatrix).T @ inverse_vectors)
    selections = []
    for axis_edges, value, extent in zip(edges, center_coordinates, coordinate_extent, strict=True):
        start = int(np.searchsorted(axis_edges, value - extent, side="right") - 1)
        stop = int(np.searchsorted(axis_edges, value + extent, side="left"))
        start = max(0, min(start, len(axis_edges) - 2))
        stop = max(start + 1, min(stop, len(axis_edges) - 1))
        selections.append(slice(start, stop))
    return tuple(selections)


def _slice_mdhisto(data, slices, edges):
    from ..mdhisto import MDHistoAxis

    axes = tuple(
        MDHistoAxis(axis.name, edge[selection.start:selection.stop + 1], axis.units, axis.kind, axis.frame, axis.path, axis.metadata)
        for axis, edge, selection in zip(data.axes, edges, slices, strict=True)
    )
    return MDHistoData(axes, data.signal[slices], data.errors[slices], data.mask[slices], data.num_events[slices], coordinate_system=data.coordinate_system, visual_normalization=data.visual_normalization, metadata=data.metadata)
