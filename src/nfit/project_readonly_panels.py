"""Read-only Qt detail panels with explicit presentation capabilities.

Contexts carry only the live helpers needed by each panel. Builders neither
prepare datasets nor reach into the project explorer's module namespace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import PointListData
from .pipeline import DatasetEntry


@dataclass(frozen=True)
class DatasetMetadataPanelContext:
    """Metadata inspection and the explorer's overridable tree builder."""

    metadata_mapping: Callable[[DatasetEntry], Mapping[str, Any]]
    metadata_tree: Callable[..., Any]


@dataclass(frozen=True)
class AnalysisTablePanelContext:
    """Live scalar formatting for persisted analysis values."""

    format_number: Callable[[float], str]


@dataclass(frozen=True)
class MetadataTreePanelContext:
    """Live bounded metadata-item rendering."""

    add_item: Callable[[Any, str, Any], Any]


def _details_group_box(self, title: str, lines: list[str]) -> Any:
    from PySide6 import QtCore, QtWidgets

    group_box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    text = "\n".join(lines) if lines else "-"
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(label)
    return group_box


def _dataset_metadata_group_box(
    self, dataset: DatasetEntry, *, context: DatasetMetadataPanelContext,
) -> Any:
    metadata = dict(context.metadata_mapping(dataset))
    if dataset.parameters:
        metadata["Parameters"] = dict(dataset.parameters)
    return context.metadata_tree(
        "Metadata",
        metadata,
        object_name="dataset_metadata_tree",
        empty_text="No additional metadata.",
    )


def _analysis_table_group_box(
    self, data: PointListData, *, context: AnalysisTablePanelContext,
) -> Any:
    from PySide6 import QtCore, QtGui, QtWidgets

    box = QtWidgets.QGroupBox("Results table")
    layout = QtWidgets.QVBoxLayout(box)
    table = QtWidgets.QTableWidget(data.size, len(data.column_names))
    table.setObjectName("analysis_output_table")
    table.setToolTip("Persisted analysis values. Rejected Bragg reflections are shown in red.")
    table.setHorizontalHeaderLabels(data.column_names)
    accepted = data.column("Accepted") if "Accepted" in data.columns else np.ones(data.size)
    for row in range(data.size):
        for column, name in enumerate(data.column_names):
            value = float(data.column(name)[row])
            item = QtWidgets.QTableWidgetItem(context.format_number(value) if np.isfinite(value) else "-")
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            if not bool(accepted[row]):
                item.setForeground(QtGui.QColor("#d94b45"))
            table.setItem(row, column, item)
    table.resizeColumnsToContents()
    table.setMinimumHeight(180)
    table.setMaximumHeight(420)
    layout.addWidget(table)
    return box


def _metadata_tree_group_box(
    self,
    title: str,
    mapping: dict[str, Any],
    *,
    object_name: str,
    empty_text: str,
    context: MetadataTreePanelContext,
) -> Any:
    from PySide6 import QtCore, QtWidgets

    group_box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(group_box)
    layout.setContentsMargins(10, 8, 10, 8)
    if not mapping:
        label = QtWidgets.QLabel(empty_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        return group_box

    tree = QtWidgets.QTreeWidget()
    tree.setObjectName(object_name)
    tree.setToolTip("Expandable metadata table. Expand rows to inspect nested fields and hover values for full text.")
    tree.setColumnCount(2)
    tree.setHeaderLabels(["Field", "Value"])
    tree.setRootIsDecorated(True)
    tree.setAlternatingRowColors(True)
    tree.setUniformRowHeights(True)
    tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
    tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tree.setMinimumHeight(120)
    tree.setMaximumHeight(260)
    tree.header().setStretchLastSection(True)
    tree.header().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    for key in sorted(mapping):
        item = context.add_item(tree, str(key), mapping[key])
        if key == "Parameters":
            item.setExpanded(True)
    for index in range(min(4, tree.topLevelItemCount())):
        tree.topLevelItem(index).setExpanded(True)
    tree.resizeColumnToContents(0)
    layout.addWidget(tree)
    return group_box
