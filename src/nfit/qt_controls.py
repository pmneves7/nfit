"""Application-wide Qt input safeguards used by nfit's GUI windows."""

from __future__ import annotations

from typing import Any

COMPACT_SCALAR_FIELD_WIDTH = 130
COMPACT_SHORT_TEXT_FIELD_WIDTH = 220


def constrain_input_width(widget: Any, maximum: int) -> Any:
    """Keep a short scalar or symbolic input from consuming a whole form row."""

    widget.setMaximumWidth(int(maximum))
    return widget


def configure_gui_input_policy(app: Any) -> None:
    """Reserve wheel input for scrolling, never editing values or plot views.

    Applies to current and future widgets, regardless of keyboard focus.
    Wheel input over other widgets is forwarded to a surrounding scroll area.
    """

    from PySide6 import QtCore, QtWidgets

    policy = getattr(app, "_nfit_numeric_spin_box_policy", None)
    if policy is None:

        class GuiInputPolicy(QtCore.QObject):
            def eventFilter(self, watched: Any, event: Any) -> bool:
                if (
                    event.type() == QtCore.QEvent.Type.Wheel
                    and isinstance(watched, QtWidgets.QWidget)
                ):
                    if _is_scroll_surface(watched):
                        # QTextEdit uses Ctrl+wheel to alter its font size.
                        # Scroll gestures must remain scrolling even with Ctrl.
                        event.setModifiers(
                            event.modifiers() & ~QtCore.Qt.KeyboardModifier.ControlModifier
                        )
                        return False
                    _forward_wheel_to_scroll_area(watched, event)
                    return True
                if (
                    event.type() == QtCore.QEvent.Type.Show
                    and isinstance(watched, QtWidgets.QAbstractSpinBox)
                ):
                    _hide_spin_box_buttons(watched)
                return super().eventFilter(watched, event)

        policy = GuiInputPolicy(app)
        app.installEventFilter(policy)
        app._nfit_numeric_spin_box_policy = policy

    for widget in app.allWidgets():
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            _hide_spin_box_buttons(widget)


def _hide_spin_box_buttons(spin_box: Any) -> None:
    from PySide6 import QtWidgets

    spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)


# Preserve the previous public entry point used by existing GUI factories.
configure_numeric_spin_boxes = configure_gui_input_policy


def _is_scroll_surface(widget: Any) -> bool:
    from PySide6 import QtWidgets

    if isinstance(widget, (QtWidgets.QAbstractScrollArea, QtWidgets.QScrollBar, QtWidgets.QMenu)):
        return True
    if not isinstance(widget, QtWidgets.QWidget):
        return False
    parent = widget.parentWidget()
    return isinstance(parent, QtWidgets.QAbstractScrollArea) and parent.viewport() is widget


def _forward_wheel_to_scroll_area(widget: Any, event: Any) -> None:
    from PySide6 import QtCore, QtGui, QtWidgets

    current = widget.parentWidget() if isinstance(widget, QtWidgets.QWidget) else None
    while current is not None:
        if isinstance(current, QtWidgets.QAbstractScrollArea):
            viewport = current.viewport()
            forwarded = QtGui.QWheelEvent(
                QtCore.QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
                event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                event.buttons(), event.modifiers(), event.phase(), event.inverted(),
                event.source(), event.pointingDevice(),
            )
            QtWidgets.QApplication.sendEvent(viewport, forwarded)
            return
        current = current.parentWidget()
