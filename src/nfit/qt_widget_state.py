"""Preserve presentation state while rebuilding Qt editor controls.

This module knows nothing about project data. Callers decide whether the same
object is still selected; restored state never includes scientific values.
"""

from __future__ import annotations

from contextlib import contextmanager


def _locator(root, widget):
    """Use named controls when available, with a local path for Qt internals."""
    from PySide6 import QtWidgets

    if widget is root:
        return ()
    if widget is None or not root.isAncestorOf(widget):
        return None
    name = widget.objectName()
    if name and len(root.findChildren(QtWidgets.QWidget, name)) == 1:
        return (("name", name),)
    parent = widget.parentWidget()
    parent_path = _locator(root, parent)
    if parent_path is None:
        return None
    siblings = [child for child in parent.children() if type(child) is type(widget)]
    return (*parent_path, ("child", type(widget).__name__, siblings.index(widget)))


def _resolve(root, path):
    from PySide6 import QtWidgets

    if path is None:
        return None
    widget = root
    for part in path:
        if part[0] == "name":
            widget = widget.findChild(QtWidgets.QWidget, part[1])
        else:
            children = [child for child in widget.children() if type(child).__name__ == part[1]]
            widget = children[part[2]] if part[2] < len(children) else None
        if widget is None:
            return None
    return widget


@contextmanager
def preserve_widget_state(root, *, enabled=True, is_current=lambda: True,
                          scroll_position=None, focus_object_name=None):
    """Retain tabs, scrolling and the focused field across an editor rebuild.

    Delayed layout passes may clamp scroll bars. Restore them again afterwards,
    without stealing focus from a later user action or replaying an old refresh.
    """
    from PySide6 import QtCore, QtWidgets
    from shiboken6 import isValid

    from .qt_operation_guard import close_widget_popups

    close_widget_popups(root)

    generation = getattr(root, "_nfit_state_generation", 0) + 1
    root._nfit_state_generation = generation
    # A full panel refresh supersedes pending restores from its smaller editors.
    for child in root.findChildren(QtWidgets.QWidget):
        if hasattr(child, "_nfit_state_generation"):
            child._nfit_state_generation += 1
    if not enabled:
        yield
        return

    widgets = [root, *root.findChildren(QtWidgets.QWidget)]
    tabs = [
        (_locator(root, widget), widget.tabText(widget.currentIndex()))
        for widget in widgets
        if isinstance(widget, QtWidgets.QTabWidget) and widget.currentIndex() >= 0
    ]
    scrolls = [
        (_locator(root, widget), widget.horizontalScrollBar().value(),
         widget.verticalScrollBar().value())
        for widget in widgets if isinstance(widget, QtWidgets.QAbstractScrollArea)
    ]
    if scroll_position is not None and isinstance(root, QtWidgets.QAbstractScrollArea):
        scrolls = [(path, x, scroll_position if path == () else y) for path, x, y in scrolls]
    focused = root.window().focusWidget()
    focus_path = _locator(root, focused)
    if focused is not None and focus_path:
        # An unnamed field's sibling index is not a durable editing identity.
        # Only named controls (including a named spin/combo's internal editor)
        # may transfer focus to a replacement widget.
        owner = focused.parentWidget()
        named_field = (bool(focused.objectName()) and not focused.objectName().startswith("qt_")
                       and focus_path[0][0] == "name")
        named_owner = (
            isinstance(owner, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox))
            and bool(owner.objectName()) and _locator(root, owner)[0][0] == "name"
        )
        if not (named_field or named_owner):
            focus_path = None
    if focus_object_name:
        focus_path = (("name", focus_object_name),)
    text_state = None
    focused_inside = focused is not None and root.isAncestorOf(focused)
    if focused_inside and isinstance(focused, QtWidgets.QLineEdit):
        text_state = (focused.cursorPosition(), focused.selectionStart(), len(focused.selectedText()))

    # Removing a focused editor can emit editingFinished while its replacement
    # is only half built. Its triggering edit has already been handled by the
    # caller; do not turn widget teardown into another scientific edit.
    blocked = []
    if focused_inside:
        controls = [focused]
        if isinstance(focused.parentWidget(), QtWidgets.QAbstractSpinBox):
            controls.append(focused.parentWidget())
        blocked = [(control, control.blockSignals(True)) for control in controls]
    try:
        yield
    finally:
        for control, previous in blocked:
            if isValid(control) and root.isAncestorOf(control):
                control.blockSignals(previous)

    def valid():
        return (isValid(root) and getattr(root, "_nfit_state_generation", None) == generation
                and is_current())

    if not valid():
        return
    for path, title in tabs:
        widget = _resolve(root, path)
        if isinstance(widget, QtWidgets.QTabWidget):
            index = next((i for i in range(widget.count()) if widget.tabText(i) == title), -1)
            if index >= 0:
                blocker = QtCore.QSignalBlocker(widget)
                widget.setCurrentIndex(index)
                del blocker
    if root.layout() is not None:
        root.layout().activate()
    fallback_focus = root.window().focusWidget()
    focus_restored = False
    restored_focus = None

    def focus_changed():
        current = root.window().focusWidget()
        if focus_restored:
            return current is not restored_focus
        teardown_cleared = current is None and (
            fallback_focus is focused or fallback_focus is None or not isValid(fallback_focus)
        )
        return current is not fallback_focus and not teardown_cleared

    def restore_focus():
        nonlocal focus_restored, restored_focus
        if focus_restored or focus_changed():
            return
        replacement = _resolve(root, focus_path)
        if (focus_path is None and focused is not None and isValid(focused)
                and root.isAncestorOf(focused)):
            replacement = focused
        if replacement is not None and replacement.isEnabled() and replacement.isVisibleTo(root):
            focus_restored = True
            restored_focus = replacement
            replacement.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            if text_state is not None and isinstance(replacement, QtWidgets.QLineEdit):
                cursor, start, length = text_state
                if start >= 0 and length:
                    replacement.setSelection(start, length)
                else:
                    replacement.setCursorPosition(cursor)

    def restore_scrolls():
        if not valid() or focus_changed():
            return
        # Newly inserted controls may become visible only after a layout pass.
        # Restore focus once, and only if the user has not focused another field.
        restore_focus()
        for path, x, y in scrolls:
            widget = _resolve(root, path)
            if isinstance(widget, QtWidgets.QAbstractScrollArea):
                widget.horizontalScrollBar().setValue(x)
                widget.verticalScrollBar().setValue(y)

    restore_scrolls()
    QtCore.QTimer.singleShot(0, restore_scrolls)
    QtCore.QTimer.singleShot(0, lambda: QtCore.QTimer.singleShot(0, restore_scrolls))
