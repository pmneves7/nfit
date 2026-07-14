"""Application-wide Qt input safeguards used by nfit's GUI windows."""

from __future__ import annotations

from typing import Any


def configure_numeric_spin_boxes(app: Any) -> None:
    """Prevent accidental mouse changes in every current and future spin box."""

    from PySide6 import QtCore, QtWidgets

    policy = getattr(app, "_nfit_numeric_spin_box_policy", None)
    if policy is None:

        class NumericSpinBoxPolicy(QtCore.QObject):
            def eventFilter(self, watched: Any, event: Any) -> bool:
                if event.type() == QtCore.QEvent.Type.Wheel and _spin_box_for(watched) is not None:
                    return True
                if (
                    event.type() == QtCore.QEvent.Type.Show
                    and isinstance(watched, QtWidgets.QAbstractSpinBox)
                ):
                    _hide_spin_box_buttons(watched)
                return super().eventFilter(watched, event)

        policy = NumericSpinBoxPolicy(app)
        app.installEventFilter(policy)
        app._nfit_numeric_spin_box_policy = policy

    for widget in app.allWidgets():
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            _hide_spin_box_buttons(widget)


def _hide_spin_box_buttons(spin_box: Any) -> None:
    from PySide6 import QtWidgets

    spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)


def _spin_box_for(widget: Any) -> Any | None:
    from PySide6 import QtWidgets

    current = widget if isinstance(widget, QtWidgets.QWidget) else None
    while current is not None:
        if isinstance(current, QtWidgets.QAbstractSpinBox):
            return current
        current = current.parentWidget()
    return None
