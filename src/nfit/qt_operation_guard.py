"""Keep user input out of editors while GUI operations replace shared state."""

from __future__ import annotations

from functools import wraps


def close_widget_popups(root=None) -> None:
    """Dismiss combo/menu popups before their controls or windows are rebuilt."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    widgets = app.allWidgets() if root is None else root.findChildren(QtWidgets.QWidget)
    for widget in widgets:
        if isinstance(widget, QtWidgets.QComboBox):
            widget.hidePopup()
        elif isinstance(widget, QtWidgets.QMenu) and widget.isVisible():
            widget.hide()


def close_operation_window(window) -> None:
    """Allow an operation to retire a viewer without accepting user close input."""
    from PySide6.QtCore import QObject
    from shiboken6 import isValid

    if not isinstance(window, QObject):
        window.close()
        return
    previous = window.property("nfit_operation_close")
    window.setProperty("nfit_operation_close", True)
    try:
        window.close()
    finally:
        if isValid(window):
            window.setProperty("nfit_operation_close", previous)


def _application_filter(app):
    from PySide6 import QtCore, QtGui, QtWidgets

    existing = getattr(app, "_nfit_operation_input_filter", None)
    if existing is not None:
        return existing

    class InputFilter(QtCore.QObject):
        def __init__(self):
            super().__init__(app)
            self.depth = 0
            self.input_types = {
                getattr(QtCore.QEvent.Type, name) for name in (
                    "MouseButtonPress", "MouseButtonRelease", "MouseButtonDblClick",
                    "MouseMove", "Wheel", "KeyPress", "KeyRelease", "Shortcut",
                    "ShortcutOverride", "ContextMenu", "TouchBegin", "TouchUpdate",
                    "TouchEnd", "TabletPress", "TabletMove", "TabletRelease",
                    "DragEnter", "DragMove", "Drop", "Close", "InputMethod",
                )
            }

        def eventFilter(self, watched, event):
            if not self.depth or event.type() not in self.input_types:
                return False
            # Progress windows and modal questions remain interactive. Modeless
            # viewers/editors (including windows created during the operation)
            # do not. Walk QObject ancestry so shortcuts and popup views follow
            # the same rule as ordinary widgets.
            owner = watched
            if isinstance(owner, QtGui.QWindow):
                owner = next((
                    widget for widget in app.topLevelWidgets()
                    if widget.windowHandle() == watched
                ), watched)
            while owner is not None:
                if event.type() == QtCore.QEvent.Type.Close and owner.property("nfit_operation_close"):
                    return False
                if isinstance(owner, QtWidgets.QDialog) and (
                    owner.property("nfit_operation_dialog") or owner.isModal()
                ):
                    return False
                owner = owner.parent()
            if event.type() == QtCore.QEvent.Type.Close:
                event.ignore()
            else:
                event.accept()
            return True

    result = InputFilter()
    app._nfit_operation_input_filter = result
    return result


class GuiOperationGuard:
    """Nestable, GUI-thread-only input barrier; paints and progress still run.

    No enabled states or scientific values are changed. Discard queued input
    before releasing the final guard, rather than replaying clicks against
    controls that may now represent different data.
    """

    def __init__(self):
        from PySide6 import QtCore, QtWidgets

        self.app = QtWidgets.QApplication.instance()
        self.filter = None
        if self.app is None:
            return
        if QtCore.QThread.currentThread() != self.app.thread():
            raise RuntimeError("GUI operation guards must run on the GUI thread")
        self.filter = _application_filter(self.app)
        if not self.filter.depth:
            close_widget_popups()
            self.app.installEventFilter(self.filter)
        self.filter.depth += 1

    def close(self):
        if self.filter is None:
            return
        guard_filter, self.filter = self.filter, None
        try:
            if guard_filter.depth == 1:
                self.app.processEvents()
        finally:
            guard_filter.depth -= 1
            if not guard_filter.depth:
                self.app.removeEventFilter(guard_filter)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def guarded_gui_operation(function):
    """Protect a synchronous operation, including its final GUI refresh."""
    @wraps(function)
    def guarded(*args, **kwargs):
        with GuiOperationGuard():
            return function(*args, **kwargs)
    return guarded
