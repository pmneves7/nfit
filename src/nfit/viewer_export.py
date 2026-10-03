"""Text exports for prepared data-viewer profiles, maps, and waterfalls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .histogram_statistics import EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR
from .measurement_aggregation import MEASUREMENT_STATISTICS_CHANNELS, MEASUREMENT_STATISTICS_KEY
from .measurement_dependencies import (
    CountingDependencies,
    counting_dependency_archive_payload,
    source_dependency_archive_payload,
)
from .measurement_profiles import MeasurementProfile
from .plotting_core import WaterfallTrace
from .project_dataset_io import _json_safe_value


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


def save_measurement_profile_csv(
    path: str | Path, profile: MeasurementProfile, *, coordinate_name: str,
    coordinate_unit: str = "", include_statistics: bool = True,
    require_exact_background_uncertainty: bool = False,
) -> Path:
    """Export a prepared profile and its versioned declaration in ``.csv.json``.

    Extended columns retain available C,V,N, contributions, mask and coverage.
    ``include_statistics=False`` preserves the historical three-column CSV;
    the JSON sidecar still records its estimator and uncertainty assumptions.
    """
    if require_exact_background_uncertainty:
        uncertainty = profile.data.metadata.get("background_profile_uncertainty")
        if uncertainty and uncertainty != "source_covariance":
            from .measurement_dependencies import SourceReplayRequired

            raise SourceReplayRequired("This profile retains approximate background uncertainty; replay its sources before exporting")
    coordinate, values, errors = profile.arrays
    columns = {coordinate_name: coordinate, "intensity": values, "uncertainty": errors}
    data = profile.data
    if include_statistics:
        columns.update({name: data.auxiliary_channels[name].values
                        for name in (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR, *MEASUREMENT_STATISTICS_CHANNELS)
                        if name in data.auxiliary_channels})
        columns.update(num_events=data.num_events, mask=data.mask,
                       coverage_fraction=data.auxiliary_channels["coverage_fraction"].values)
    destination = save_viewer_columns_csv(path, columns)
    declaration = {"version": 1, "measurement_contract": profile.contract.to_dict(),
        "coordinate": {"name": coordinate_name, "unit": coordinate_unit,
                       "edges": data.axes[0].values.tolist()},
        "metadata": data.metadata, "columns": list(columns)}
    _save_measurement_declaration(destination, declaration, {"profile": data.source_dependencies, "counting": data.counting_dependencies})
    return destination


def _save_measurement_declaration(destination, declaration, dependencies):
    payload, descriptors = {}, {}
    for name, dependency in dependencies.items():
        if dependency is not None:
            members = (counting_dependency_archive_payload(dependency) if isinstance(dependency, CountingDependencies)
                       else source_dependency_archive_payload(dependency))
            prefix = name + "_"
            payload.update({prefix+key: value for key,value in members.items()})
            descriptors[name] = {"prefix": prefix, "shape": dependency.shape}
    if payload:
        archive = destination.with_suffix(destination.suffix + ".sources.npz")
        np.savez(archive, **payload)
        declaration["source_dependencies"] = {"file": archive.name, "payloads": descriptors}
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(_json_safe_value(declaration), indent=2) + "\n", encoding="utf-8")


def save_measurement_grid_csv(path, view, *, channel="signal", coordinate_units=("", ""),
                              values=None, errors=None, include_statistics=True):
    """Export a prepared slice, retained statistics, units and source factors.

    Diagnostic channels and model predictions do not acquire an observation
    counting contract. The legacy array-only ``save_grid_csv`` stays available.
    """
    values = np.asarray(view[channel] if values is None else values)
    x, y = np.meshgrid(view["x_centers"], view["y_centers"])
    if values.shape != x.shape:
        raise ValueError("Prepared grid values must match the displayed coordinates")
    columns = {"x": x, "y": y, "I": values}
    if errors is not None or channel == "signal":
        columns["dI"] = view["errors"] if errors is None else errors
    for name in ("mask", "coverage_fraction", "num_events"):
        if include_statistics and name in view:
            columns[name] = view[name]
    metadata = {key: view[key] for key in ("num_events_semantics", "masks_applied", "signal_unit") if key in view}
    dependencies = {}
    if channel == "signal":
        for name in (*EVENT_STATISTICS_CHANNELS, NORMALIZATION_DENOMINATOR, *MEASUREMENT_STATISTICS_CHANNELS):
            if name in view:
                columns[name] = view[name]
        metadata.update({key: view[key] for key in ("measurement_contract", "event_statistics", MEASUREMENT_STATISTICS_KEY) if key in view})
        dependencies["grid"] = view.get("source_dependencies")
        dependencies["counting"] = view.get("counting_dependencies")
    destination = save_viewer_columns_csv(path, columns)
    _save_measurement_declaration(destination, {"version": 1, "channel": channel,
        "metadata": metadata, "coordinate_units": coordinate_units,
        "edges": {axis:view[f"{axis}_edges"] for axis in ("x", "y")}, "columns": list(columns)}, dependencies)
    return destination


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
    prepared = [getattr(trace, "measurement", None) for trace in traces]
    declarations, dependencies = {}, {}
    if not model and any(item is not None for item in prepared):
        names = ("num_events", "mask", "coverage_fraction", *EVENT_STATISTICS_CHANNELS,
                 NORMALIZATION_DENOMINATOR, *MEASUREMENT_STATISTICS_CHANNELS)
        for name in names:
            parts = []
            available = False
            for trace,item in zip(traces,prepared,strict=True):
                values = None
                if item is not None:
                    if name in {"num_events", "mask"}:
                        values = getattr(item.data, name)
                    elif name in item.data.auxiliary_channels:
                        values = item.data.auxiliary_channels[name].values
                available |= values is not None
                parts.append(np.full(len(trace.x), np.nan) if values is None else values)
            if available:
                columns[name] = np.concatenate(parts)
        for index,item in enumerate(prepared):
            if item is not None:
                declarations[str(index)] = {"contract": item.contract.to_dict(), "metadata": item.data.metadata}
                dependencies[f"trace{index}"] = item.data.source_dependencies
                dependencies[f"trace{index}_counting"] = item.data.counting_dependencies
    destination = save_viewer_columns_csv(path, columns)
    if declarations:
        _save_measurement_declaration(destination, {"version": 1, "traces": declarations, "columns": list(columns)}, dependencies)
    return destination
