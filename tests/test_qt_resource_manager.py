from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    kind: str = "Histogram"
    state: str = "In RAM; saved in project"
    ram_bytes: int = 0
    reclaimable_bytes: int = 0
    project_disk_bytes: int = 0
    temporary_disk_bytes: int = 0
    users: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    can_unload: bool = True
    can_load: bool = False
    can_delete: bool = True
    reason: str = ""


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit import qt_resource_manager as gui

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rows = (
        Row("a", "Alpha", ram_bytes=9_000_000_000, users=("Viewer 1",), parents=("Source A",)),
        Row("b", "Beta", ram_bytes=80_000_000_000, children=("Background",)),
        Row("c", "Cold", state="Saved in project", can_unload=False, can_load=True),
        Row("d", "Delta", ram_bytes=700_000_000, can_delete=False, reason="Only copy of unsaved data."),
    )
    snapshot = SimpleNamespace(
        rows=rows, used_bytes=89_700_000_000, reserved_bytes=5_000_000_000,
        limit_bytes=100_000_000_000, process_bytes=92_000_000_000,
        temporary_directory="/chosen/scientific/work", temporary_disk_bytes=2_000_000_000,
        cpu_limit=8, ram_limit_mb=100_000,
    )
    state = SimpleNamespace(snapshot=snapshot, calls=[], warnings=[], snapshot_calls=0)

    def inventory():
        state.snapshot_calls += 1
        return state.snapshot

    def limits(cpu, ram_mb):
        state.calls.append(("limits", cpu, ram_mb))
        snapshot.cpu_limit = cpu
        snapshot.ram_limit_mb = ram_mb

    def directory(path):
        state.calls.append(("directory", path))
        snapshot.temporary_directory = path

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *args: state.warnings.append(args),
    )
    value = gui.ResourceManagerWindow(
        snapshot=inventory,
        unload=lambda keys: state.calls.append(("unload", keys)),
        load=lambda keys: state.calls.append(("load", keys)),
        delete=lambda keys: state.calls.append(("delete", keys)),
        apply_limits=limits, set_temporary_directory=directory,
    )
    yield value, state, gui, app
    value.close()


def test_table_sorts_bytes_numerically_and_preserves_selection(manager):
    value, state, _gui, _app = manager
    from PySide6 import QtCore

    value.select_keys(("a", "b"))
    value.table.sortByColumn(3, QtCore.Qt.SortOrder.AscendingOrder)
    assert [value.proxy.index(i, 0).data() for i in range(4)] == ["Cold", "Delta", "Alpha", "Beta"]
    assert set(value.selected_keys()) == {"a", "b"}
    value.table.sortByColumn(3, QtCore.Qt.SortOrder.DescendingOrder)
    assert value.proxy.index(0, 0).data() == "Beta"
    assert value.proxy.index(0, 3).data(QtCore.Qt.ItemDataRole.UserRole) == 80_000_000_000
    state.snapshot.rows = tuple(replace(row, ram_bytes=row.ram_bytes + 1) for row in state.snapshot.rows)
    value.refresh()
    assert set(value.selected_keys()) == {"a", "b"}
    assert value.proxy.index(0, 0).data() == "Beta"


def test_shift_and_control_mouse_click_select_multiple_rows(manager):
    value, _state, _gui, app = manager
    from PySide6 import QtCore, QtTest

    value.show()
    app.processEvents()

    def click(row, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
        point = value.table.visualRect(value.proxy.index(row, 0)).center()
        QtTest.QTest.mouseClick(value.table.viewport(), QtCore.Qt.MouseButton.LeftButton, modifier, point)

    click(0)
    click(1, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert set(value.selected_keys()) == {"a", "b"}
    click(3, QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert set(value.selected_keys()) >= {"b", "c", "d"}


def test_actions_use_all_selected_keys_and_show_relationships(manager):
    value, state, _gui, _app = manager
    value.select_keys(("a", "b"))
    assert value.unload_button.isEnabled()
    assert not value.load_button.isEnabled()
    value.unload_button.click()
    assert state.calls[-1] == ("unload", ("a", "b"))
    assert "Parents: Source A" in value.details.toPlainText()
    assert "Children: Background" in value.details.toPlainText()
    assert "Used by: Viewer 1" in value.details.toPlainText()
    value.select_keys(("c",))
    assert value.load_button.isEnabled()
    assert not value.unload_button.isEnabled()
    value.load_button.click()
    assert state.calls[-1] == ("load", ("c",))
    value.select_keys(("a", "d"))
    assert not value.delete_button.isEnabled()
    assert "Only copy of unsaved data." in value.details.toPlainText()


def test_deletion_requires_explicit_confirmation_and_defaults_to_no(manager, monkeypatch):
    value, state, _gui, _app = manager
    from PySide6 import QtWidgets

    dialogs = []
    answer = QtWidgets.QMessageBox.StandardButton.No

    def question(*args):
        dialogs.append(args)
        return answer

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)
    value.select_keys(("a", "b"))
    value.delete_button.click()
    assert not state.calls
    assert dialogs[-1][-1] == QtWidgets.QMessageBox.StandardButton.No
    assert "Are you sure" in dialogs[-1][2]
    assert "Alpha" in dialogs[-1][2] and "Beta" in dialogs[-1][2]
    assert "next successful save" in dialogs[-1][2]
    assert "Loaded cubes remain available in memory" in dialogs[-1][2]
    answer = QtWidgets.QMessageBox.StandardButton.Yes
    value.delete_button.click()
    assert state.calls[-1] == ("delete", ("a", "b"))


def test_busy_guard_pauses_refresh_and_disables_actions(manager):
    value, state, _gui, _app = manager
    value.select_keys(("a",))
    calls = state.snapshot_calls
    value.set_busy(True)
    value.refresh()
    assert state.snapshot_calls == calls
    assert not value.unload_button.isEnabled()
    assert not value.table.isEnabled()
    assert not value.apply_button.isEnabled()
    value.set_busy(False)
    assert state.snapshot_calls == calls + 1
    assert value.unload_button.isEnabled()


def test_async_callbacks_can_keep_window_busy_until_completion(manager):
    value, state, _gui, _app = manager
    value.select_keys(("c",))
    value._load = lambda _keys: value.set_busy(True)
    calls = state.snapshot_calls
    value.load_button.click()
    assert value._busy
    assert state.snapshot_calls == calls
    value.set_busy(False)
    assert state.snapshot_calls == calls + 1


def test_gb_budget_roundtrip_preserves_unsaved_edits(manager):
    value, state, _gui, _app = manager
    assert value.ram_limit_gb.value() == pytest.approx(104.858)
    value.cpu_limit.setValue(4)
    value.ram_limit_gb.setValue(4.295)
    value.refresh()
    assert value.cpu_limit.value() == 4
    assert value.ram_limit_gb.value() == 4.295
    value.apply_button.click()
    assert state.calls[-1] == ("limits", 4, 4096)
    assert not value._limits_dirty
    assert "Accounted RAM 89.700 GB" in value.summary.text()
    assert "Temporary disk usage: 2.000 GB" == value.temporary_usage.text()


def test_temporary_location_changes_only_through_callback(manager, monkeypatch):
    value, state, _gui, _app = manager
    from PySide6 import QtWidgets

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: "/new/work")
    value.browse_button.click()
    assert value.temporary_directory.text() == "/new/work"
    assert state.snapshot.temporary_directory == "/chosen/scientific/work"
    value.refresh()
    assert value.temporary_directory.text() == "/new/work"
    value.directory_button.click()
    assert state.calls[-1] == ("directory", "/new/work")


def test_errors_do_not_discard_selection_and_periodic_errors_are_nonmodal(manager):
    value, state, _gui, _app = manager
    value.select_keys(("a",))

    def fail(_keys=None):
        raise MemoryError("Opening this will exceed the RAM budget.")

    value._unload = fail
    value.unload_button.click()
    assert value.selected_keys() == ("a",)
    assert len(state.warnings) == 1
    assert "RAM budget" in value.status.text()
    value._snapshot = fail
    value._automatic_refresh()
    assert len(state.warnings) == 1


def test_all_interactive_controls_and_headers_have_tooltips(manager):
    value, _state, _gui, _app = manager
    from PySide6 import QtCore, QtWidgets

    for widget in value.findChildren(QtWidgets.QWidget):
        if isinstance(widget, (
            QtWidgets.QPushButton, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox,
            QtWidgets.QLineEdit, QtWidgets.QTableView, QtWidgets.QPlainTextEdit,
        )):
            # Spin boxes' private line editors inherit their owning control's tooltip.
            if isinstance(widget, QtWidgets.QLineEdit) and isinstance(widget.parent(), QtWidgets.QAbstractSpinBox):
                continue
            assert widget.toolTip(), widget.objectName()
    for column in range(value.model.columnCount()):
        assert value.model.headerData(column, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.ToolTipRole)


def test_timer_is_active_only_while_window_is_visible(manager):
    value, _state, _gui, app = manager
    assert not value._timer.isActive()
    value.show()
    app.processEvents()
    assert value._timer.isActive()
    value.hide()
    assert not value._timer.isActive()


def test_default_table_headers_fit_and_columns_remain_resizable(manager):
    value, _state, _gui, app = manager
    from PySide6 import QtCore, QtWidgets

    value.show()
    app.processEvents()
    header = value.table.horizontalHeader()
    for column in range(value.model.columnCount()):
        title = value.model.headerData(column, QtCore.Qt.Orientation.Horizontal)
        assert header.sectionSize(column) >= header.fontMetrics().horizontalAdvance(title) + 22
        assert header.sectionResizeMode(column) == QtWidgets.QHeaderView.ResizeMode.Interactive
    assert value.table.columnWidth(1) >= value.table.fontMetrics().horizontalAdvance("Shared model cache") + 16
    assert value.table.horizontalScrollBar().maximum() == 0
