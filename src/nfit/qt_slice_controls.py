from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .qt_branding import configure_application_icon
from .qt_controls import configure_numeric_spin_boxes


@dataclass
class _HiddenAxisControls:
    integrate: Any
    value: Any
    width: Any
    low: Any
    high: Any
    slider: Any
    centers: np.ndarray
    syncing: bool = False


def _qt_app():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    configure_application_icon(app)
    configure_numeric_spin_boxes(app)
    return app


def _make_data_viewer_window_class():
    from PySide6 import QtWidgets

    class DataViewerWindow(QtWidgets.QMainWindow):
        def __init__(self, viewer):
            super().__init__()
            self.viewer = viewer

        def closeEvent(self, event):
            self.viewer._close_volume_panel()
            self.viewer._close_cut_viewers()
            super().closeEvent(event)
            callback = self.viewer._close_callback
            if callback is not None:
                self.viewer._close_callback = None
                callback()

    return DataViewerWindow


def _make_float_spinbox(low: float = -1.0e12, high: float = 1.0e12):
    from PySide6 import QtWidgets

    spin = _ClampingDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(6)
    spin.setSingleStep(max((high - low) / 200.0, 0.001) if np.isfinite(high - low) else 0.001)
    spin.setKeyboardTracking(False)
    spin.setMaximumWidth(104)
    spin.setMinimumWidth(82)
    spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
    return spin


def _compact_combobox(combo) -> None:
    from PySide6 import QtWidgets

    combo.setMaximumWidth(132)
    combo.setMinimumWidth(84)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)


def _expanding_combobox(combo) -> None:
    from PySide6 import QtWidgets

    combo.setMinimumWidth(180)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


def _make_index_slider(size: int):
    from PySide6 import QtCore, QtWidgets

    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, max(int(size) - 1, 0))
    slider.setSingleStep(1)
    slider.setPageStep(max(int(size) // 20, 1))
    return slider


def _qt_classes():
    from PySide6 import QtCore, QtGui, QtWidgets

    return QtCore, QtGui, QtWidgets


QtCore, QtGui, QtWidgets = _qt_classes()


class _ClampingDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Accept complete numeric text outside the range; clamp on commit."""

    def _number(self, text):
        text = text.strip()
        if self.prefix() and text.startswith(self.prefix()):
            text = text[len(self.prefix()):]
        if self.suffix() and text.endswith(self.suffix()):
            text = text[:-len(self.suffix())]
        value, valid = self.locale().toDouble(text.strip())
        return value, valid and np.isfinite(value)

    def validate(self, text, position):
        _value, valid = self._number(text)
        if valid:
            return QtGui.QValidator.State.Acceptable, text, position
        if text.strip() in {"", "+", "-", self.locale().decimalPoint()}:
            return QtGui.QValidator.State.Intermediate, text, position
        return super().validate(text, position)

    def valueFromText(self, text):
        value, valid = self._number(text)
        if valid:
            return float(np.clip(value, self.minimum(), self.maximum()))
        return super().valueFromText(text)


class _DualRangeSlider(QtWidgets.QWidget):
    changed = QtCore.Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.low_value = 0
        self.high_value = 1000
        self._dragging: str | None = None
        self.setMinimumHeight(28)
        self.setMouseTracking(True)

    def set_values(self, low: int, high: int) -> None:
        self.low_value, self.high_value = sorted(
            (
                int(np.clip(low, 0, 1000)),
                int(np.clip(high, 0, 1000)),
            )
        )
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 0, -10, 0)
        y = rect.center().y()
        painter.setPen(QtGui.QPen(QtGui.QColor("#707070"), 3))
        painter.drawLine(rect.left(), y, rect.right(), y)
        low_x = self._value_to_x(self.low_value)
        high_x = self._value_to_x(self.high_value)
        painter.setPen(QtGui.QPen(QtGui.QColor("#4f8bd6"), 5))
        painter.drawLine(low_x, y, high_x, y)
        self._draw_handle(painter, low_x, y)
        self._draw_handle(painter, high_x, y)

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        x = float(event.position().x())
        self._dragging = (
            "low"
            if abs(x - self._value_to_x(self.low_value))
            <= abs(x - self._value_to_x(self.high_value))
            else "high"
        )
        self._set_dragged_value(x)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            self._set_dragged_value(float(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:
        del event
        self._dragging = None

    def _set_dragged_value(self, x: float) -> None:
        value = self._x_to_value(x)
        if self._dragging == "low":
            self.low_value = min(value, self.high_value)
        elif self._dragging == "high":
            self.high_value = max(value, self.low_value)
        self.update()
        self.changed.emit(self.low_value, self.high_value)

    def _value_to_x(self, value: int) -> float:
        rect = self.rect().adjusted(10, 0, -10, 0)
        return float(rect.left() + rect.width() * int(value) / 1000.0)

    def _x_to_value(self, x: float) -> int:
        rect = self.rect().adjusted(10, 0, -10, 0)
        if rect.width() <= 0:
            return 0
        fraction = (float(x) - rect.left()) / rect.width()
        return int(np.clip(round(1000.0 * fraction), 0, 1000))

    def _draw_handle(self, painter, x: float, y: float) -> None:
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#80b7ff")))
        painter.setPen(QtGui.QPen(QtGui.QColor("#303030"), 1))
        painter.drawEllipse(QtCore.QPointF(x, y), 6.0, 6.0)


class _IntegratedAxisSlider(QtWidgets.QWidget):
    changed = QtCore.Signal(str)

    def __init__(self, size: int, parent=None) -> None:
        super().__init__(parent)
        self.size = max(int(size), 1)
        self.value_index = 0
        self.low_index = 0
        self.high_index = 0
        self.integrate_range = False
        self._dragging: str | None = None
        self.setMinimumHeight(42)
        self.setMouseTracking(True)

    def set_indices(self, value_index: int, low_index: int, high_index: int) -> None:
        self.value_index = int(np.clip(value_index, 0, self.size - 1))
        self.low_index = int(np.clip(low_index, 0, self.size - 1))
        self.high_index = int(np.clip(high_index, 0, self.size - 1))
        self.low_index, self.high_index = sorted((self.low_index, self.high_index))
        self.update()

    def set_integrate_range(self, integrate_range: bool) -> None:
        self.integrate_range = bool(integrate_range)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self._track_rect()
        y = rect.center().y()
        painter.setPen(QtGui.QPen(QtGui.QColor("#7f7f7f"), 3))
        painter.drawLine(rect.left(), y, rect.right(), y)
        if self.integrate_range:
            low_x = self._index_to_x(self.low_index)
            high_x = self._index_to_x(self.high_index)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4f8bd6"), 5))
            painter.drawLine(low_x, y, high_x, y)
            self._draw_handle(painter, low_x, y, QtGui.QColor("#80b7ff"))
            self._draw_handle(painter, high_x, y, QtGui.QColor("#80b7ff"))
        self._draw_handle(painter, self._index_to_x(self.value_index), y, QtGui.QColor("#f4d35e"))

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._dragging = self._nearest_handle(event.position().x())
        self._set_dragged_index(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            self._set_dragged_index(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        del event
        self._dragging = None

    def _set_dragged_index(self, x: float) -> None:
        index = self._x_to_index(float(x))
        if self._dragging == "value":
            width = self.high_index - self.low_index
            half_low = width // 2
            half_high = width - half_low
            low = index - half_low
            high = index + half_high
            if low < 0:
                high -= low
                low = 0
            if high >= self.size:
                low -= high - (self.size - 1)
                high = self.size - 1
            self.set_indices(index, max(low, 0), high)
            self.changed.emit("value")
        elif self._dragging == "low":
            self.set_indices(int(round(0.5 * (index + self.high_index))), index, self.high_index)
            self.changed.emit("low")
        elif self._dragging == "high":
            self.set_indices(int(round(0.5 * (self.low_index + index))), self.low_index, index)
            self.changed.emit("high")

    def _nearest_handle(self, x: float) -> str:
        candidates = {"value": abs(x - self._index_to_x(self.value_index))}
        if self.integrate_range:
            candidates["low"] = abs(x - self._index_to_x(self.low_index))
            candidates["high"] = abs(x - self._index_to_x(self.high_index))
        return min(candidates, key=candidates.get)

    def _track_rect(self):
        margins = 16
        return self.rect().adjusted(margins, 0, -margins, 0)

    def _index_to_x(self, index: int) -> float:
        rect = self._track_rect()
        if self.size <= 1:
            return float(rect.center().x())
        return float(rect.left() + (rect.width() * index / (self.size - 1)))

    def _x_to_index(self, x: float) -> int:
        rect = self._track_rect()
        if self.size <= 1 or rect.width() <= 0:
            return 0
        fraction = (x - rect.left()) / rect.width()
        return int(np.clip(round(fraction * (self.size - 1)), 0, self.size - 1))

    def _draw_handle(self, painter, x: float, y: float, color) -> None:
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor("#303030"), 1))
        painter.drawEllipse(QtCore.QPointF(x, y), 6.0, 6.0)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)
