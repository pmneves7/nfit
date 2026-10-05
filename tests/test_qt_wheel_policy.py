import ast
from pathlib import Path

import pytest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from nfit.qt_controls import configure_gui_input_policy

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_gui_input_policy(application)
    return application


def wheel(app, widget, *, delta=-120, focused=False, control=False):
    from PySide6 import QtCore, QtGui

    widget.show()
    if focused:
        widget.setFocus()
    app.processEvents()
    point = QtCore.QPoint(5, 5)
    event = QtGui.QWheelEvent(QtCore.QPointF(point), QtCore.QPointF(widget.mapToGlobal(point)),
        QtCore.QPoint(), QtCore.QPoint(0, delta), QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ControlModifier if control else QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate, False)
    app.sendEvent(widget, event)


@pytest.mark.parametrize("kind", ["spin", "double", "date", "combo", "editable", "slider", "dial", "tabs", "custom"])
@pytest.mark.parametrize("focused", [False, True])
def test_wheel_never_changes_control_or_custom_plot(app, kind, focused):
    from PySide6 import QtCore, QtWidgets

    if kind == "spin":
        widget = QtWidgets.QSpinBox()
        widget.setValue(5)
        state = widget.value
    elif kind == "double":
        widget = QtWidgets.QDoubleSpinBox()
        widget.setValue(5.)
        state = widget.value
    elif kind == "date":
        widget = QtWidgets.QDateEdit(QtCore.QDate(2026, 1, 1))
        state = widget.date
    elif kind in {"combo", "editable"}:
        widget = QtWidgets.QComboBox()
        widget.addItems(["one", "two", "three"])
        widget.setCurrentIndex(1)
        widget.setEditable(kind == "editable")
        state = widget.currentIndex
    elif kind in {"slider", "dial"}:
        widget = QtWidgets.QSlider() if kind == "slider" else QtWidgets.QDial()
        widget.setValue(50)
        state = widget.value
    elif kind == "tabs":
        widget = QtWidgets.QTabBar()
        widget.addTab("one")
        widget.addTab("two")
        state = widget.currentIndex
    else:
        class Plot(QtWidgets.QWidget):
            value = 0

            def wheelEvent(self, event):
                self.value += 1

        widget = Plot()
        def state():
            return widget.value
    original = state()
    try:
        wheel(app, widget, focused=focused)
        wheel(app, widget, delta=120, focused=focused, control=True)
        if kind == "editable":
            wheel(app, widget.lineEdit(), focused=focused)
        if kind in {"spin", "double", "date"}:
            wheel(app, widget.findChild(QtWidgets.QLineEdit), focused=focused)
        assert state() == original
    finally:
        widget.close()


def test_control_wheel_scrolls_surrounding_panel_without_editing(app):
    from PySide6 import QtWidgets

    area = QtWidgets.QScrollArea()
    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    spin = QtWidgets.QSpinBox()
    spin.setValue(5)
    layout.addWidget(spin)
    layout.addStretch()
    content.setMinimumHeight(1500)
    area.setWidget(content)
    area.resize(300, 200)
    area.show()
    app.processEvents()
    try:
        wheel(app, spin, focused=True)
        assert spin.value() == 5
        assert area.verticalScrollBar().value() > 0
        before = area.verticalScrollBar().value()
        wheel(app, area.viewport())
        assert area.verticalScrollBar().value() > before
    finally:
        area.close()


def test_list_scroll_and_keyboard_edits_remain_available(app):
    from PySide6 import QtCore, QtTest, QtWidgets

    listing = QtWidgets.QListWidget()
    listing.addItems([str(i) for i in range(100)])
    listing.resize(200, 100)
    listing.show()
    spin = QtWidgets.QSpinBox()
    spin.setValue(5)
    try:
        wheel(app, listing.viewport())
        assert listing.verticalScrollBar().value() > 0
        spin.show()
        spin.setFocus()
        QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Up)
        assert spin.value() == 6
    finally:
        listing.close()
        spin.close()


def test_all_real_qt_application_factories_install_shared_policy():
    root = Path(__file__).resolve().parents[1] / "src/nfit"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n.func for n in ast.walk(node) if isinstance(n, ast.Call)]
            if not any(isinstance(c, ast.Attribute) and c.attr == "QApplication" for c in calls):
                continue
            assert any(isinstance(c, ast.Name) and c.id in {
                "configure_numeric_spin_boxes", "configure_gui_input_policy"
            } for c in calls), (path.name, node.name)


def test_ctrl_wheel_scrolls_text_without_changing_font_size(app):
    from PySide6 import QtWidgets

    text = QtWidgets.QTextEdit()
    text.setReadOnly(True)
    text.setPlainText("\n".join(str(i) for i in range(100)))
    text.resize(200, 100)
    text.show()
    app.processEvents()
    original = text.font().pointSizeF()
    try:
        wheel(app, text.viewport(), control=True)
        assert text.font().pointSizeF() == original
        assert text.verticalScrollBar().value() > 0
    finally:
        text.close()


def test_open_dropdown_list_can_scroll_without_selecting_an_item(app):
    from PySide6 import QtWidgets

    combo = QtWidgets.QComboBox()
    combo.addItems([str(i) for i in range(100)])
    combo.setCurrentIndex(50)
    combo.setMaxVisibleItems(5)
    combo.show()
    combo.showPopup()
    app.processEvents()
    try:
        before = combo.view().verticalScrollBar().value()
        wheel(app, combo.view().viewport())
        assert combo.currentIndex() == 50
        assert combo.view().verticalScrollBar().value() > before
    finally:
        combo.hidePopup()
        combo.close()


def test_native_window_wheel_reaches_scroll_panel_but_not_editor(app):
    from PySide6 import QtCore, QtGui, QtWidgets

    area = QtWidgets.QScrollArea()
    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    spin = QtWidgets.QSpinBox()
    spin.setValue(5)
    layout.addWidget(spin)
    layout.addStretch()
    content.setMinimumHeight(1500)
    area.setWidget(content)
    area.resize(300, 200)
    area.show()
    app.processEvents()
    window = area.windowHandle()
    global_point = spin.mapToGlobal(QtCore.QPoint(5, 5))
    event = QtGui.QWheelEvent(QtCore.QPointF(window.mapFromGlobal(global_point)),
        QtCore.QPointF(global_point), QtCore.QPoint(), QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate, False)
    try:
        app.sendEvent(window, event)
        assert spin.value() == 5
        assert area.verticalScrollBar().value() > 0
    finally:
        area.close()
