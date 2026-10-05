"""Flat, sortable presentation of project resources and their lifecycle controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from .project_resources import ResourceRow, ResourceSnapshot

_GB_BYTES = 1_000_000_000
_MIB_BYTES = 1024**2
_SORT_ROLE = QtCore.Qt.ItemDataRole.UserRole
_COLUMNS = (
    ("Name", "name"),
    ("Type", "kind"),
    ("State", "state"),
    ("RAM (GB)", "ram_bytes"),
    ("Freed on unload (GB)", "reclaimable_bytes"),
    ("Project cache (GB)", "project_disk_bytes"),
    ("Temporary cache (GB)", "temporary_disk_bytes"),
    ("Used by", "users"),
)
_BYTE_FIELDS = frozenset(
    ("ram_bytes", "reclaimable_bytes", "project_disk_bytes", "temporary_disk_bytes")
)
_COLUMN_TOOLTIPS = (
    "Resource name. Rows are independent of the dataset-group hierarchy.",
    "The kind of numerical data or cache represented by this row.",
    "Whether this resource is loaded, cached, or must be recomputed.",
    "Memory held by this resource. Shared arrays also appear in their other owners' rows.",
    "Memory that can be released by unloading this row after its users are detached.",
    "Space occupied by this cache in the saved nfit project file.",
    "Space occupied by this resource in temporary scientific storage.",
    "Viewers and operations currently using this resource. See the details below.",
)


class ResourceTableModel(QtCore.QAbstractTableModel):
    """Read-only inventory; constructing or sorting it never loads scientific data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: tuple[ResourceRow, ...] = ()

    def replace_rows(self, rows: Sequence[ResourceRow]) -> None:
        self.beginResetModel()
        self.rows = tuple(rows)
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(_COLUMNS)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        row = self.rows[index.row()]
        field = _COLUMNS[index.column()][1]
        value = getattr(row, field)
        if role == _SORT_ROLE:
            return len(value) if field == "users" else value
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if field in _BYTE_FIELDS:
                return f"{value / _GB_BYTES:.3f}"
            return ", ".join(value) if field == "users" else value
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and field in _BYTE_FIELDS:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if field in _BYTE_FIELDS:
                return f"{value:,} bytes. {_COLUMN_TOOLTIPS[index.column()]}"
            return row.reason or _COLUMN_TOOLTIPS[index.column()]
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation != QtCore.Qt.Orientation.Horizontal or not 0 <= section < len(_COLUMNS):
            return super().headerData(section, orientation, role)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return _COLUMNS[section][0]
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return _COLUMN_TOOLTIPS[section]
        return None


class ResourceSortProxyModel(QtCore.QSortFilterProxyModel):
    """Compare Python integers directly across Qt's mixed-width QVariant types."""

    def lessThan(self, left, right):
        first = self.sourceModel().data(left, _SORT_ROLE)
        second = self.sourceModel().data(right, _SORT_ROLE)
        if isinstance(first, int) and isinstance(second, int):
            return first < second
        return str(first).casefold() < str(second).casefold()


class ResourceManagerWindow(QtWidgets.QDialog):
    """Manage resident whole datasets through GUI-independent service callbacks.

    Callbacks may call ``set_busy(True)`` before starting an asynchronous operation
    and ``set_busy(False)`` after it finishes. Inventory refreshes are paused while
    busy. The snapshot callback must return metadata without decoding data arrays.
    """

    def __init__(
        self,
        *,
        snapshot: Callable[[], ResourceSnapshot],
        unload: Callable[[tuple[str, ...]], object],
        load: Callable[[tuple[str, ...]], object],
        delete: Callable[[tuple[str, ...]], object],
        apply_limits: Callable[[int, int], object],
        set_temporary_directory: Callable[[str], object],
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("resource_manager")
        self.setWindowTitle("nfit Resource Manager")
        self.resize(1360, 760)
        self._snapshot = snapshot
        self._unload = unload
        self._load = load
        self._delete = delete
        self._apply_limits = apply_limits
        self._set_temporary_directory = set_temporary_directory
        self._busy = False
        self._limits_dirty = False
        self._directory_dirty = False
        self._last_snapshot = None

        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("resource_manager_summary")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.summary.setToolTip(
            "Accounted RAM counts shared numerical storage once. Reserved RAM is held "
            "for operations before allocation. Process RAM also includes Python and GUI overhead."
        )
        layout.addWidget(self.summary)

        self.model = ResourceTableModel(self)
        self.proxy = ResourceSortProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(_SORT_ROLE)
        self.proxy.setSortCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.table = QtWidgets.QTableView()
        self.table.setObjectName("resource_manager_table")
        self.table.setToolTip(
            "Click a column heading to sort ascending or descending. Use Shift-click "
            "for a range and Ctrl-click (Command-click on macOS) for individual rows."
        )
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((245, 165, 170, 100, 175, 155, 180, 140)):
            title = _COLUMNS[column][0]
            header.resizeSection(column, max(width, header.fontMetrics().horizontalAdvance(title) + 36))
        header.setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        actions = QtWidgets.QHBoxLayout()
        self.unload_button = self._button(
            "Remove from memory", "resource_manager_unload",
            "Release the selected resources from RAM while retaining reloadable data "
            "and settings. Viewers using them may need to be closed or detached.",
            lambda: self._run_action(self._unload, "Remove from memory"),
        )
        self.load_button = self._button(
            "Load into memory", "resource_manager_load",
            "Load the selected resources into RAM. The operation is admitted only "
            "when the RAM budget can cover its memory use.",
            lambda: self._run_action(self._load, "Load into memory"),
        )
        self.delete_button = self._button(
            "Delete cache from project", "resource_manager_delete",
            "Remove the selected saved caches on the next successful project save, "
            "after confirmation. Loaded cubes stay in RAM; source files and dataset "
            "settings remain available.",
            self._confirm_delete,
        )
        for button in (self.unload_button, self.load_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch()
        self.refresh_button = self._button(
            "Refresh", "resource_manager_refresh",
            "Refresh resource metadata without loading scientific data.", self.refresh,
        )
        actions.addWidget(self.refresh_button)
        layout.addLayout(actions)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setObjectName("resource_manager_details")
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        self.details.setToolTip(
            "Selected resources, their parents and children, viewers or operations using "
            "them, and reasons an action is unavailable."
        )
        layout.addWidget(self.details)

        settings_group = QtWidgets.QGroupBox("Budgets and temporary storage")
        settings = QtWidgets.QGridLayout(settings_group)
        self.cpu_limit = QtWidgets.QSpinBox()
        self.cpu_limit.setObjectName("resource_manager_cpu_limit")
        self.cpu_limit.setRange(0, 8192)
        self.cpu_limit.setSpecialValueText("Auto")
        self.cpu_limit.setToolTip(
            "Maximum CPU workers for scientific operations. Auto uses the application default. "
            "This limit is shared by open projects in this nfit installation."
        )
        self.ram_limit_gb = QtWidgets.QDoubleSpinBox()
        self.ram_limit_gb.setObjectName("resource_manager_ram_limit_gb")
        self.ram_limit_gb.setDecimals(3)
        self.ram_limit_gb.setRange(0, 1_048_576 * _MIB_BYTES / _GB_BYTES)
        self.ram_limit_gb.setSingleStep(1.0)
        self.ram_limit_gb.setSpecialValueText("Auto")
        self.ram_limit_gb.setSuffix(" GB")
        self.ram_limit_gb.setToolTip(
            "Scientific RAM budget in decimal gigabytes (1 GB = 1,000,000,000 bytes). "
            "Auto uses the application default. This budget is shared by open projects."
        )
        self.cpu_limit.valueChanged.connect(self._mark_limits_dirty)
        self.ram_limit_gb.valueChanged.connect(self._mark_limits_dirty)
        self.apply_button = self._button(
            "Apply budgets", "resource_manager_apply_limits",
            "Save CPU and RAM limits for this nfit installation. Reducing the budget "
            "does not silently unload data; use Remove from memory to release resources.",
            self._save_limits,
        )
        settings.addWidget(QtWidgets.QLabel("CPU limit"), 0, 0)
        settings.addWidget(self.cpu_limit, 0, 1)
        settings.addWidget(QtWidgets.QLabel("RAM limit"), 0, 2)
        settings.addWidget(self.ram_limit_gb, 0, 3)
        settings.addWidget(self.apply_button, 0, 4)

        self.temporary_directory = QtWidgets.QLineEdit()
        self.temporary_directory.setObjectName("resource_manager_temporary_directory")
        self.temporary_directory.setPlaceholderText("Alongside project or source file")
        self.temporary_directory.setToolTip(
            "Location for future temporary scientific storage. Changing it does not move "
            "existing open files. Leave empty to use the default beside the project file."
        )
        self.temporary_directory.textEdited.connect(self._mark_directory_dirty)
        self.browse_button = self._button(
            "Browse…", "resource_manager_browse_temporary_directory",
            "Choose a directory for future temporary scientific storage.", self._browse_directory,
        )
        self.directory_button = self._button(
            "Apply location", "resource_manager_apply_temporary_directory",
            "Use this directory for future temporary allocations. Existing files stay "
            "at their current location until released.", self._save_directory,
        )
        settings.addWidget(QtWidgets.QLabel("Temporary storage"), 1, 0)
        settings.addWidget(self.temporary_directory, 1, 1, 1, 3)
        settings.addWidget(self.browse_button, 1, 4)
        settings.addWidget(self.directory_button, 2, 4)
        self.temporary_usage = QtWidgets.QLabel()
        self.temporary_usage.setObjectName("resource_manager_temporary_usage")
        self.temporary_usage.setToolTip(
            "Disk usage of temporary scientific storage tracked by nfit. Saved project "
            "caches appear in the separate Project cache table column."
        )
        settings.addWidget(self.temporary_usage, 2, 1, 1, 3)
        layout.addWidget(settings_group)

        self.status = QtWidgets.QLabel()
        self.status.setObjectName("resource_manager_status")
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.status.setToolTip("The current resource operation or the most recent error.")
        layout.addWidget(self.status)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(3000)
        self._timer.timeout.connect(self._automatic_refresh)
        self.refresh()

    def _button(self, text, name, tooltip, callback):
        button = QtWidgets.QPushButton(text)
        button.setObjectName(name)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def selected_rows(self) -> tuple[ResourceRow, ...]:
        indexes = sorted(self.table.selectionModel().selectedRows(), key=lambda index: index.row())
        return tuple(self.model.rows[self.proxy.mapToSource(index).row()] for index in indexes)

    def selected_keys(self) -> tuple[str, ...]:
        return tuple(row.key for row in self.selected_rows())

    def select_keys(self, keys: Sequence[str]) -> None:
        """Select existing rows by stable identity without changing table sorting."""
        wanted = set(keys)
        selection = QtCore.QItemSelection()
        for index, row in enumerate(self.model.rows):
            if row.key in wanted:
                mapped = self.proxy.mapFromSource(self.model.index(index, 0))
                selection.select(mapped, mapped)
        self.table.selectionModel().select(
            selection,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        self._selection_changed()

    def refresh(self) -> None:
        """Refresh inventory metadata, preserving selections and unsaved editor values."""
        self._refresh(quiet=False)

    def _automatic_refresh(self) -> None:
        self._refresh(quiet=True)

    def _refresh(self, *, quiet: bool) -> None:
        if self._busy:
            return
        keys = self.selected_keys()
        try:
            snapshot = self._snapshot()
        except Exception as error:
            self._report_error("Refresh resources", error, quiet=quiet)
            return
        self._last_snapshot = snapshot
        if tuple(snapshot.rows) != self.model.rows:
            current = self.table.currentIndex()
            current_key = (
                self.model.rows[self.proxy.mapToSource(current).row()].key
                if current.isValid() else None
            )
            vertical = self.table.verticalScrollBar().value()
            horizontal = self.table.horizontalScrollBar().value()
            self.model.replace_rows(snapshot.rows)
            self.select_keys(keys)
            if current_key is not None:
                for index, row in enumerate(self.model.rows):
                    if row.key == current_key:
                        self.table.selectionModel().setCurrentIndex(
                            self.proxy.mapFromSource(self.model.index(index, 0)),
                            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                        )
                        break
            self.table.verticalScrollBar().setValue(vertical)
            self.table.horizontalScrollBar().setValue(horizontal)
        self._selection_changed()
        self.summary.setText(
            f"Accounted RAM {snapshot.used_bytes / _GB_BYTES:.3f} GB  |  "
            f"Reserved {snapshot.reserved_bytes / _GB_BYTES:.3f} GB  |  "
            f"Budget {snapshot.limit_bytes / _GB_BYTES:.3f} GB  |  "
            f"Process RAM {snapshot.process_bytes / _GB_BYTES:.3f} GB"
        )
        self.temporary_usage.setText(
            f"Temporary disk usage: {snapshot.temporary_disk_bytes / _GB_BYTES:.3f} GB"
        )
        if not self._limits_dirty:
            blockers = [QtCore.QSignalBlocker(self.cpu_limit), QtCore.QSignalBlocker(self.ram_limit_gb)]
            self.cpu_limit.setValue(snapshot.cpu_limit)
            self.ram_limit_gb.setValue(snapshot.ram_limit_mb * _MIB_BYTES / _GB_BYTES)
            del blockers
        if not self._directory_dirty:
            self.temporary_directory.setText(snapshot.temporary_directory)

    def set_busy(self, busy: bool) -> None:
        """Prevent conflicting resource actions while an admitted operation is running."""
        self._busy = bool(busy)
        self.table.setEnabled(not self._busy)
        for widget in (
            self.refresh_button, self.cpu_limit, self.ram_limit_gb, self.apply_button,
            self.temporary_directory, self.browse_button, self.directory_button,
        ):
            widget.setEnabled(not self._busy)
        self.status.setText("Resource operation in progress…" if self._busy else "")
        self._selection_changed()
        if not self._busy:
            self.refresh()

    def _selection_changed(self, *_args) -> None:
        rows = self.selected_rows()
        enabled = bool(rows) and not self._busy
        self.unload_button.setEnabled(enabled and all(row.can_unload for row in rows))
        self.load_button.setEnabled(enabled and all(row.can_load for row in rows))
        self.delete_button.setEnabled(enabled and all(row.can_delete for row in rows))
        if not rows:
            self.details.setPlainText("Select one or more rows to view their relationships and available actions.")
            return
        descriptions = []
        for row in rows:
            parts = [f"{row.name} ({row.kind}) — {row.state}"]
            for title, values in (
                ("Parents", row.parents), ("Children", row.children), ("Used by", row.users),
            ):
                parts.append(f"{title}: " + (", ".join(values) or "None"))
            if row.reason:
                parts.append(row.reason)
            descriptions.append("\n".join(parts))
        self.details.setPlainText("\n\n".join(descriptions))

    def _run_action(self, callback, title, *, keys=None) -> None:
        if self._busy:
            return
        keys = self.selected_keys() if keys is None else keys
        if not keys:
            return
        try:
            callback(keys)
        except Exception as error:
            self._report_error(title, error)
            return
        if not self._busy:
            self.refresh()

    def _confirm_delete(self) -> None:
        keys = self.selected_keys()
        if self._busy or not keys:
            return
        rows = self.selected_rows()
        names = "\n".join(row.name for row in rows[:12])
        if len(rows) > 12:
            names += f"\n… and {len(rows) - 12} more"
        answer = QtWidgets.QMessageBox.question(
            self, "Delete cache from project?",
            "Are you sure you want to delete the selected cached data from the project?\n\n"
            + names
            + "\n\nSaved caches will be removed from the project file on the next successful save. "
            "Loaded cubes remain available in memory. Source files and dataset settings "
            "are retained. Data may need to be reduced or binned again.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._run_action(self._delete, "Delete cache from project", keys=keys)

    def _mark_limits_dirty(self, *_args) -> None:
        self._limits_dirty = True

    def _mark_directory_dirty(self, *_args) -> None:
        self._directory_dirty = True

    def _save_limits(self) -> None:
        try:
            self._apply_limits(
                self.cpu_limit.value(), round(self.ram_limit_gb.value() * _GB_BYTES / _MIB_BYTES),
            )
        except Exception as error:
            self._report_error("Apply budgets", error)
            return
        self._limits_dirty = False
        self.refresh()

    def _browse_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Temporary scientific storage", self.temporary_directory.text(),
        )
        if path:
            self.temporary_directory.setText(path)
            self._directory_dirty = True

    def _save_directory(self) -> None:
        try:
            self._set_temporary_directory(self.temporary_directory.text().strip())
        except Exception as error:
            self._report_error("Apply temporary storage location", error)
            return
        self._directory_dirty = False
        self.refresh()

    def _report_error(self, title, error, *, quiet=False) -> None:
        self.status.setText(str(error))
        if not quiet:
            QtWidgets.QMessageBox.warning(self, title, str(error))
