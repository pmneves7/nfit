"""Shared Qt helpers for rebin settings and resolved-grid summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .mdhisto import MDHistoData
from .project_rebinning import (
    _cluster_coordinate_centers,
    _coordinate_center_edges,
    _rebin_axis_fractional,
    _rebin_axis_mode,
    _sanitize_rebin_axis_config,
)
from .rebin import _uniform_center_edges

PHYSICAL_REBIN_MODE_CHOICES = (
    ("Discrete", "discrete"),
    ("Step", "step"),
    ("Bins", "bins"),
    ("Edges", "edges"),
    ("Tolerance", "tolerance"),
)

METADATA_REBIN_MODE_CHOICES = PHYSICAL_REBIN_MODE_CHOICES


def add_rebin_mode_items(combo: Any, *, metadata: bool = False) -> None:
    """Populate a grid-construction mode selector."""

    choices = METADATA_REBIN_MODE_CHOICES if metadata else PHYSICAL_REBIN_MODE_CHOICES
    for title, value in choices:
        combo.addItem(title, value)


def add_rebin_assignment_items(combo: Any) -> None:
    """Populate a point-assignment selector independently of grid construction."""

    combo.addItem("Fractional", True)
    combo.addItem("Discrete", False)


def rebin_bin_information_widget(
    config: Mapping[str, Any],
    *,
    data: Any | None,
    object_prefix: str,
    estimated_seconds: float | None = None,
    compressed_disk_bytes: int | None = None,
) -> Any:
    """Build a compact, expandable description of a requested or cached grid."""

    from PySide6 import QtCore, QtWidgets

    widget = QtWidgets.QWidget()
    widget.setObjectName(f"{object_prefix}_bin_information")
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)

    exact, shape, total_bins, payload_bytes = rebin_memory_estimate(config, data)
    axes = _axis_information(config, data if exact else None)
    qualifier = "Resolved cached grid" if exact else "Configured grid estimate"
    summary = QtWidgets.QLabel(
        f"{qualifier}: {shape or '-'} · {total_bins:,} bins"
    )
    summary.setObjectName(f"{object_prefix}_bin_shape")
    summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    summary.setToolTip(
        "The exact cached output is shown after a rebin. Before then, dynamic Discrete and "
        "Tolerance counts are estimates from the saved configuration."
    )
    layout.addWidget(summary)

    size_parts = [
        f"Numeric payload: {'exactly' if exact else 'about'} {_format_bytes(payload_bytes)} in memory",
        _compressed_disk_size_text(compressed_disk_bytes),
    ]
    if estimated_seconds is not None and np.isfinite(estimated_seconds):
        size_parts.append(f"estimated rebin time: {estimated_seconds:.1f} s")
    size = QtWidgets.QLabel(" · ".join(size_parts))
    size.setObjectName(f"{object_prefix}_bin_size")
    size.setWordWrap(True)
    size.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    size.setToolTip(
        "The memory value counts the output numerical arrays. Compressed disk size is the "
        "actual binning artifact stored in the current nfit project; a dash means this "
        "binning is not currently embedded in the saved project."
    )
    layout.addWidget(size)

    tree = QtWidgets.QTreeWidget()
    tree.setObjectName(f"{object_prefix}_bin_tree")
    tree.setColumnCount(5)
    tree.setHeaderLabels(["Axis / values", "Grid", "Mode", "Bins", "Limits"])
    tree.setAlternatingRowColors(True)
    tree.setRootIsDecorated(True)
    tree.setToolTip(
        "Expand an axis to inspect its centers and edges. Very long arrays are abbreviated "
        "at the middle so this panel remains responsive."
    )
    for name, mode, fractional, count, centers, edges, units in axes:
        center_limits = _limits_text(centers, units)
        edge_limits = _limits_text(edges, units)
        limits = center_limits or edge_limits or "resolved during rebin"
        parent = QtWidgets.QTreeWidgetItem(
            [name, mode.title(), "Fractional" if fractional else "Discrete", f"{count:,}", limits]
        )
        if centers is not None:
            parent.addChild(
                QtWidgets.QTreeWidgetItem(
                    ["Centers", "", "", f"{len(centers):,}", _format_values(centers, units)]
                )
            )
        if edges is not None:
            parent.addChild(
                QtWidgets.QTreeWidgetItem(
                    ["Edges", "", "", f"{len(edges):,}", _format_values(edges, units)]
                )
            )
        tree.addTopLevelItem(parent)
    tree.header().setStretchLastSection(True)
    tree.resizeColumnToContents(0)
    tree.resizeColumnToContents(1)
    tree.resizeColumnToContents(2)
    tree.resizeColumnToContents(3)
    layout.addWidget(tree, 1)
    return widget


def rebin_memory_estimate(
    config: Mapping[str, Any], data: Any | None = None
) -> tuple[bool, tuple[int, ...], int, int]:
    """Return exact/cached state, shape, bins, and numerical payload bytes."""

    exact = isinstance(data, MDHistoData)
    axes = _axis_information(config, data if exact else None)
    shape = tuple(item[3] for item in axes)
    total_bins = int(np.prod(shape, dtype=np.int64)) if shape else 0
    payload_bytes = _mdhisto_payload_bytes(data) if exact else total_bins * 33
    return exact, shape, total_bins, payload_bytes


def rebin_memory_estimate_label(
    config: Mapping[str, Any],
    *,
    data: Any | None,
    object_prefix: str,
    compressed_disk_bytes: int | None = None,
) -> Any:
    """Build the memory/disk estimate shown beside editable rebin settings."""

    from PySide6 import QtCore, QtWidgets

    label = QtWidgets.QLabel(
        rebin_memory_estimate_text(
            config,
            data=data,
            compressed_disk_bytes=compressed_disk_bytes,
        )
    )
    label.setObjectName(f"{object_prefix}_memory_estimate")
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setToolTip(
        "This estimates the numerical output arrays from the configured grid, or reports "
        "the exact cached payload. Peak rebin memory can be higher because source and worker "
        "arrays also exist. Compressed disk size is shown only when the current binning is "
        "embedded in the saved nfit project."
    )
    return label


def rebin_memory_estimate_text(
    config: Mapping[str, Any],
    *,
    data: Any | None,
    compressed_disk_bytes: int | None = None,
) -> str:
    """Return the concise memory and disk-size description for a rebin."""

    exact, _shape, _total_bins, payload_bytes = rebin_memory_estimate(config, data)
    return (
        f"{'Cached result memory' if exact else 'Estimated result memory'}: "
        f"{'exactly ' if exact else 'about '}{_format_bytes(payload_bytes)} · "
        f"{_compressed_disk_size_text(compressed_disk_bytes)}"
    )


def _compressed_disk_size_text(size: int | None) -> str:
    """Format an exact saved-project artifact size or an unavailable marker."""

    return f"Compressed disk size: {_format_bytes(size) if size is not None else '-'}"


def _axis_information(
    config: Mapping[str, Any], data: MDHistoData | None
) -> list[tuple[str, str, bool, int, np.ndarray | None, np.ndarray | None, str]]:
    configured = [
        _sanitize_rebin_axis_config(axis)
        for axis in config.get("axes", [])
        if isinstance(axis, Mapping)
    ]
    if data is not None:
        modes = list(data.metadata.get("rebin", {}).get("axis_modes", []))
        assignments = list(data.metadata.get("rebin", {}).get("fractional_axes", []))
        information = []
        for index, axis in enumerate(data.axes):
            mode = (
                str(modes[index])
                if index < len(modes)
                else "metadata"
                if "metadata_dimension" in axis.metadata
                else _rebin_axis_mode(config, configured[index])
                if index < len(configured)
                else "resolved"
            )
            information.append(
                (
                    axis.name,
                    mode,
                    bool(assignments[index])
                    if index < len(assignments)
                    else _rebin_axis_fractional(config, configured[index])
                    if index < len(configured)
                    else False,
                    int(data.shape[index]),
                    np.asarray(axis.centers, dtype=float),
                    np.asarray(axis.values, dtype=float),
                    axis.units,
                )
            )
        return information

    information = []
    for index, axis in enumerate(configured):
        mode = _rebin_axis_mode(config, axis)
        edges = _configured_edges(axis, mode)
        centers = None
        if axis.get("resolved_centers") is not None:
            centers = np.asarray(axis["resolved_centers"], dtype=float)
        elif axis.get("candidate_centers") is not None and mode in {
            "discrete",
            "tolerance",
        }:
            centers = _cluster_coordinate_centers(
                axis["candidate_centers"],
                0.0 if mode == "discrete" else float(axis["tolerance"]),
            )
            edges = _coordinate_center_edges(
                centers,
                singleton_half_width=(
                    float(axis["tolerance"]) if mode == "tolerance" else 0.5
                ),
            )
        elif edges is not None:
            centers = (edges[:-1] + edges[1:]) / 2.0
        count = int(
            len(centers)
            if centers is not None
            else len(edges) - 1
            if edges is not None
            else max(int(axis.get("num_bins", 1) or 1), 1)
        )
        information.append(
            (
                str(axis.get("name", f"Axis {index + 1}")),
                mode,
                _rebin_axis_fractional(config, axis),
                count,
                centers,
                edges,
                str(axis.get("units", "")),
            )
        )
    return information


def _configured_edges(axis: Mapping[str, Any], mode: str) -> np.ndarray | None:
    try:
        if mode == "edges":
            values = np.asarray(axis.get("bin_edges"), dtype=float)
            return values if values.ndim == 1 and values.size >= 2 else None
        if mode == "step":
            return _uniform_center_edges(
                float(axis["lower"]),
                float(axis["upper"]),
                step=float(axis["step_size"]),
            )
        if mode == "bins":
            return _uniform_center_edges(
                float(axis["lower"]),
                float(axis["upper"]),
                count=max(int(axis["num_bins"]), 1),
            )
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _mdhisto_payload_bytes(data: MDHistoData) -> int:
    arrays = [data.signal, data.errors, data.mask, data.num_events]
    for channel in data.auxiliary_channels.values():
        arrays.append(channel.values)
        if channel.errors is not None:
            arrays.append(channel.errors)
    arrays.extend(axis.values for axis in data.axes)
    return sum(int(np.asarray(array).nbytes) for array in arrays)


def _limits_text(values: Sequence[float] | np.ndarray | None, units: str) -> str:
    if values is None or len(values) == 0:
        return ""
    suffix = f" {units}" if units else ""
    return f"{float(values[0]):.6g} to {float(values[-1]):.6g}{suffix}"


def _format_values(values: Sequence[float] | np.ndarray, units: str) -> str:
    array = np.asarray(values, dtype=float)
    text = np.array2string(
        array,
        precision=7,
        separator=", ",
        threshold=128,
        edgeitems=12,
        max_line_width=10_000,
    )
    return text + (f" {units}" if units else "")


def _format_bytes(value: int) -> str:
    size = float(max(value, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"
