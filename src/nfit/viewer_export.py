"""Text exports for prepared data-viewer profiles, maps, and waterfalls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .plotting_core import WaterfallTrace


def save_viewer_columns_csv(
    path: str | Path,
    columns: Mapping[str, np.ndarray],
) -> Path:
    """Write equal-length numeric columns as a UTF-8 CSV file."""

    if not columns:
        raise ValueError("at least one export column is required")
    names = [str(name) for name in columns]
    if any(
        not name or any(character in name for character in ",\r\n")
        for name in names
    ):
        raise ValueError(
            "CSV column names must be non-empty and cannot contain commas or newlines"
        )
    arrays = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in columns.values()
    ]
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all export columns must have the same length")
    destination = Path(path).expanduser()
    np.savetxt(
        destination,
        np.column_stack(arrays),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.17g",
    )
    return destination


def save_profile_csv(
    path: str | Path,
    coordinate: np.ndarray,
    intensity: np.ndarray,
    uncertainty: np.ndarray,
    *,
    coordinate_name: str,
) -> Path:
    """Save one profile as coordinate, intensity, and uncertainty columns."""

    return save_viewer_columns_csv(
        path,
        {
            coordinate_name: coordinate,
            "intensity": intensity,
            "uncertainty": uncertainty,
        },
    )


def save_grid_csv(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> Path:
    """Save a rectangular 2D view in tidy ``x,y,I[,dI]`` form."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    values = np.asarray(intensity, dtype=float)
    expected = (y_values.size, x_values.size)
    if values.shape != expected:
        raise ValueError(
            f"intensity shape {values.shape} does not match y/x grid {expected}"
        )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    columns: dict[str, np.ndarray] = {
        "x": x_grid.reshape(-1),
        "y": y_grid.reshape(-1),
        "I": values.reshape(-1),
    }
    if uncertainty is not None:
        errors = np.asarray(uncertainty, dtype=float)
        if errors.shape != expected:
            raise ValueError(
                f"uncertainty shape {errors.shape} does not match y/x grid {expected}"
            )
        columns["dI"] = errors.reshape(-1)
    return save_viewer_columns_csv(path, columns)


def save_waterfall_csv(
    path: str | Path,
    traces: Sequence[WaterfallTrace],
    *,
    model: bool = False,
) -> Path:
    """Save unshifted waterfall traces in tidy ``x,y,I[,dI]`` form.

    Multidimensional waterfalls use their physical waterfall-axis coordinate
    for ``y``. Grouped one-dimensional datasets use their zero-based trace
    index because they do not have a shared physical waterfall coordinate.
    """

    if not traces:
        raise ValueError("the waterfall contains no traces to export")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    has_uncertainty = not model and any(
        trace.errors is not None for trace in traces
    )
    for index, trace in enumerate(traces):
        x_values = np.asarray(trace.x, dtype=float).reshape(-1)
        source = trace.model_values if model else trace.values
        if source is None:
            raise ValueError(
                "the displayed waterfall does not contain model values"
            )
        values = np.asarray(source, dtype=float).reshape(-1)
        if values.shape != x_values.shape:
            raise ValueError(
                "waterfall x and intensity arrays must have matching shapes"
            )
        coordinate = (
            float(index)
            if trace.waterfall_coordinate is None
            else float(trace.waterfall_coordinate)
        )
        x_parts.append(x_values)
        y_parts.append(np.full(x_values.shape, coordinate, dtype=float))
        value_parts.append(values)
        if has_uncertainty:
            errors = (
                np.full(x_values.shape, np.nan, dtype=float)
                if trace.errors is None
                else np.asarray(trace.errors, dtype=float).reshape(-1)
            )
            if errors.shape != x_values.shape:
                raise ValueError(
                    "waterfall x and uncertainty arrays must have matching shapes"
                )
            error_parts.append(errors)
    columns: dict[str, np.ndarray] = {
        "x": np.concatenate(x_parts),
        "y": np.concatenate(y_parts),
        "I": np.concatenate(value_parts),
    }
    if has_uncertainty:
        columns["dI"] = np.concatenate(error_parts)
    return save_viewer_columns_csv(path, columns)
