from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import PointListData
from .mdhisto import MDHistoData
from .plotting_core import MDHistoSliceViewer


@dataclass
class _DatasetViewState:
    model: MDHistoSliceViewer
    roi_extents: tuple[float, float, float, float] | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    show_box_tool: bool = False
    histogram_axes: bool = False
    roi_enabled: bool = False
    xcut_percent: int = 20
    ycut_percent: int = 16
    font_size: float = 12.0
    axis_linewidth: float = 1.5
    show_binning_title: bool = False
    box_tool_has_auto_shown_hist_axes: bool = False
    marker: str = "o"
    line_style: str = "none"
    marker_size: float = 5.0
    line_plot_width: float = 1.5
    marker_edge_width: float = 1.5
    marker_face_color: str = "none"
    line_color: str = "#1f77b4"
    show_errorbars: bool = True
    show_errorbar_caps: bool = False
    errorbar_cap_size: float = 3.0
    show_fit: bool = False
    unmask_model: bool = False
    show_residual: bool = False
    fit_line_color: str = "#d62728"
    fit_line_width: float = 2.0
    residual_percent: int = 30
    apply_masks: bool = True
    cmap_reversed: bool = False
    smoothing_x: float = 0.0
    smoothing_y: float = 0.0
    smoothing_fill_nans: bool = False
    show_brillouin_zone_boundaries: bool = False
    show_major_gridlines: bool = False
    brillouin_zone_color: str = "#e57373"
    brillouin_zone_linewidth: float = 1.5
    brillouin_zone_alpha: float = 1.0
    tile_dim: int | None = None
    tile_range: tuple[float, float] = (0.0, 0.0)
    tile_step: float = 1.0
    tile_step_auto: bool = True
    tile_label_decimals: int = 1
    tile_label_prefix: str = "{axis} = "
    tile_label_unit: str = "{unit}"
    tile_label_si_prefix: str = ""
    show_tile_labels: bool = True
    tile_local_color_scales: bool = False
    display_step_factors: dict[int, int] | None = None


def _coerce_datasets(data: MDHistoData | Sequence[MDHistoData]) -> list[MDHistoData]:
    if isinstance(data, (MDHistoData, PointListData)):
        return [data]
    datasets = list(data)
    if not datasets:
        raise ValueError("QtMDHistoSliceViewer requires at least one dataset")
    if not all(isinstance(dataset, (MDHistoData, PointListData)) for dataset in datasets):
        raise TypeError("all datasets must be MDHistoData or PointListData instances")
    return datasets


def _coerce_dataset_names(
    datasets: Sequence[MDHistoData], names: Sequence[str] | None
) -> list[str]:
    if names is not None:
        labels = [str(name) for name in names]
        if len(labels) != len(datasets):
            raise ValueError("dataset_names length must match datasets length")
        return labels
    labels = []
    for index, dataset in enumerate(datasets):
        source = dataset.metadata.get("source_file") if isinstance(dataset.metadata, dict) else None
        labels.append(str(source) if source else f"dataset {index + 1}")
    return labels


def _coerce_dataset_group_keys(
    datasets: Sequence[MDHistoData],
    keys: Sequence[str] | None,
) -> list[str]:
    if keys is None:
        return [""] * len(datasets)
    labels = [str(key) for key in keys]
    if len(labels) != len(datasets):
        raise ValueError("dataset_group_keys length must match datasets length")
    return labels


def _normalized_column_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("|", "")
        .replace("modulus", "")
    )


def _point_list_q_column_name(data: PointListData) -> str | None:
    for name in data.column_names:
        normalized = _normalized_column_name(name)
        if normalized in {"q", "qangstrom1", "qangstrom^-1", "qinverseangstrom"}:
            return name
    return None


def _point_list_two_theta_column_name(data: PointListData) -> str | None:
    for name in data.column_names:
        normalized = _normalized_column_name(name).replace("θ", "theta")
        if normalized in {"2theta", "twotheta"}:
            return name
    return None


def _point_list_wavelength(data: PointListData, two_theta_name: str | None) -> float | None:
    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    candidates = []
    wavelength = metadata.get("wavelength")
    if isinstance(wavelength, dict):
        configured_two_theta = wavelength.get("two_theta")
        if configured_two_theta not in (None, "", two_theta_name):
            return None
        candidates.append(wavelength.get("value"))
    for key in ("neutron_wavelength", "incident_wavelength", "lambda", "wavelength"):
        if key in metadata and not isinstance(metadata[key], dict):
            candidates.append(metadata[key])
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return None


def _cursor_q_matrix_from_metadata(metadata: dict[str, Any], key: str) -> np.ndarray:
    from .fitting import _as_3x3_matrix

    matrix = _as_3x3_matrix(metadata[key], name=key)
    if _cursor_matrix_includes_2pi(metadata, key):
        return matrix
    return 2.0 * np.pi * matrix


def _cursor_q_matrix_from_oriented_lattice(oriented_lattice: dict[str, Any]) -> np.ndarray:
    from .fitting import _as_3x3_matrix

    if "rlu_to_inv_angstrom_matrix" in oriented_lattice:
        return _as_3x3_matrix(
            oriented_lattice["rlu_to_inv_angstrom_matrix"],
            name="oriented_lattice.rlu_to_inv_angstrom_matrix",
        )
    for key in ("ub_matrix", "orientation_matrix"):
        if key in oriented_lattice:
            matrix = _as_3x3_matrix(oriented_lattice[key], name=f"oriented_lattice.{key}")
            if _cursor_matrix_includes_2pi(oriented_lattice, key):
                return matrix
            return 2.0 * np.pi * matrix
    raise ValueError("oriented_lattice does not contain a cursor Q matrix")


def _cursor_matrix_includes_2pi(metadata: dict[str, Any], key: str) -> bool:
    for flag_key in (
        f"{key}_includes_2pi",
        "q_matrix_includes_2pi",
        "include_2pi",
        "includes_2pi",
    ):
        if flag_key in metadata:
            return bool(metadata[flag_key])
    lattice_parameters = metadata.get("lattice_parameters")
    if isinstance(lattice_parameters, dict) and "include_2pi" in lattice_parameters:
        return bool(lattice_parameters["include_2pi"])
    return False


def _initial_display_dims(
    data: MDHistoData, x_dim: int | str, y_dim: int | str
) -> tuple[int | str, int | str]:
    if isinstance(data, PointListData):
        return 0, 0
    if x_dim != -1 or y_dim != 0:
        return x_dim, y_dim
    varying = [dim for dim, size in enumerate(data.shape) if size > 1]
    if len(varying) == 1:
        x_index = varying[0]
        return x_index, next(
            (dim for dim in range(data.signal.ndim) if dim != x_index),
            x_index,
        )
    if len(varying) >= 2:
        return varying[0], varying[1]
    if data.signal.ndim >= 2:
        return 0, 1
    return 0, 0


def _format_coord(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(float(value)) < 1.0e-12:
        return "0"
    return f"{float(value):.5g}"


def _format_value_with_uncertainty(value: float, error: float) -> tuple[str, str]:
    if not np.isfinite(value) or not np.isfinite(error) or error == 0.0:
        return _format_coord(value), _format_coord(error)
    error_text = f"{abs(float(error)):.2g}"
    if "e" in error_text or "E" in error_text:
        return f"{float(value):.2e}", error_text
    decimals = len(error_text.partition(".")[2])
    return f"{float(value):.{decimals}f}", error_text


def _option_name(options: dict[str, str], value: str) -> str:
    for name, option_value in options.items():
        if option_value == value:
            return name
    return next(iter(options))


def _nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - float(value))))


def _axis_components(name: str) -> dict[str, float]:
    text = name.strip()
    lower = text.lower()
    if "deltae" in lower or lower in {"e", "energy"}:
        return {"E": 1.0}
    if text.startswith("[") and text.endswith("]"):
        parts = [part.strip() for part in text[1:-1].split(",")]
        components: dict[str, float] = {}
        for component, part in zip(("H", "K", "L"), parts, strict=False):
            coefficient = _component_coefficient(part, component)
            if coefficient != 0.0:
                components[component] = coefficient
        return components
    if text in {"H", "K", "L"}:
        return {text: 1.0}
    return {}


def _cursor_axis_hkle_vector(data: MDHistoData, dimension: int) -> np.ndarray:
    """Return one displayed axis's physical HKLE direction vector.

    Rebinned histograms save their output-axis basis in ``metadata['rebin']``.
    That basis is authoritative because a human-readable label such as
    ``[L,L,-2L]`` cannot reliably encode every numeric coefficient.
    """

    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    rebin = metadata.get("rebin")
    vectors = rebin.get("vectors") if isinstance(rebin, dict) else None
    if isinstance(vectors, (list, tuple)) and 0 <= dimension < len(vectors):
        try:
            vector = np.asarray(vectors[dimension], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            vector = np.asarray([], dtype=float)
        if vector.size in {3, 4} and np.all(np.isfinite(vector)):
            return np.pad(vector, (0, 4 - vector.size))

    vector = np.zeros(4, dtype=float)
    for component, coefficient in _axis_components(data.axes[dimension].name).items():
        vector[{"H": 0, "K": 1, "L": 2, "E": 3}[component]] = coefficient
    return vector


def _component_coefficient(part: str, component: str) -> float:
    cleaned = part.replace(" ", "")
    if cleaned in {"0", "0.0", ""}:
        return 0.0
    if cleaned in {"H", "K", "L"}:
        return 1.0
    if cleaned in {"-H", "-K", "-L"}:
        return -1.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0
