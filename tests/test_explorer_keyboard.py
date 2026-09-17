"""Platform tree shortcuts use the same actions as the explorer menu."""

from types import SimpleNamespace

import pytest

from nfit import project_gui


@pytest.mark.parametrize("key_name", ["Key_Delete", "Key_Backspace"])
def test_mac_tree_delete_keys_dispatch_the_selection_action(monkeypatch, key_name):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtGui = pytest.importorskip("PySide6.QtGui")
    monkeypatch.setattr(project_gui.platform, "system", lambda: "Darwin")
    app = project_gui._qt_app()
    calls = []
    tree = project_gui._make_project_tree_class()(
        SimpleNamespace(delete_selected=lambda: calls.append("delete"))
    )
    event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        getattr(QtCore.Qt.Key, key_name),
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    app.sendEvent(tree, event)
    assert calls == ["delete"]
    assert event.isAccepted()
    tree.close()


def test_backspace_in_tree_name_editor_does_not_delete_item(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = project_gui._qt_app()
    calls = []
    tree = project_gui._make_project_tree_class()(
        SimpleNamespace(delete_selected=lambda: calls.append("delete"))
    )
    item = QtWidgets.QTreeWidgetItem(tree, ["Dataset"])
    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
    tree.setCurrentItem(item)
    tree.show()
    tree.editItem(item, 0)
    app.processEvents()
    editor = tree.findChild(QtWidgets.QLineEdit)
    assert editor is not None
    editor.deselect()
    editor.setCursorPosition(len(editor.text()))
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Backspace)
    assert editor.text() == "Datase"
    assert calls == []
    tree.close()
