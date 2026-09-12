"""Small dependency-free dual-handle range control for analysis parameters."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QVBoxLayout, QWidget


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 10_000
        self._lower = 0
        self._upper = 10_000
        self._active_handle = ""
        self.setMinimumHeight(24)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self) -> QSize:
        return QSize(220, 28)

    def setRange(self, minimum: int, maximum: int) -> None:
        if maximum <= minimum:
            raise ValueError("Range slider maximum must be greater than its minimum")
        self._minimum, self._maximum = int(minimum), int(maximum)
        self.setValues(self._lower, self._upper)

    def setValues(self, lower: int, upper: int) -> None:
        lower = max(self._minimum, min(int(lower), self._maximum))
        upper = max(lower, min(int(upper), self._maximum))
        changed = (lower, upper) != (self._lower, self._upper)
        self._lower, self._upper = lower, upper
        self.update()
        if changed:
            self.rangeChanged.emit(lower, upper)

    def values(self) -> tuple[int, int]:
        return self._lower, self._upper

    def _x_for_value(self, value: int) -> int:
        margin = 9
        span = max(1, self.width() - 2 * margin)
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return round(margin + fraction * span)

    def _value_for_x(self, x: int) -> int:
        margin = 9
        span = max(1, self.width() - 2 * margin)
        fraction = max(0.0, min(1.0, (x - margin) / span))
        return round(self._minimum + fraction * (self._maximum - self._minimum))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.height() // 2
        left = self._x_for_value(self._minimum)
        right = self._x_for_value(self._maximum)
        lower = self._x_for_value(self._lower)
        upper = self._x_for_value(self._upper)
        painter.setPen(QPen(QColor("#6b7078"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(left, center, right, center)
        painter.setPen(QPen(self.palette().highlight().color(), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(lower, center, upper, center)
        painter.setPen(QPen(QColor("#20242a"), 1))
        painter.setBrush(self.palette().highlight().color())
        painter.drawEllipse(QPoint(lower, center), 7, 7)
        painter.drawEllipse(QPoint(upper, center), 7, 7)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        lower_distance = abs(event.position().x() - self._x_for_value(self._lower))
        upper_distance = abs(event.position().x() - self._x_for_value(self._upper))
        self._active_handle = "lower" if lower_distance <= upper_distance else "upper"
        self._move_active(round(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_handle:
            self._move_active(round(event.position().x()))
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._active_handle = ""
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() not in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            super().keyPressEvent(event)
            return
        if not self._active_handle:
            self._active_handle = "lower"
        step = -1 if event.key() == Qt.Key.Key_Left else 1
        if self._active_handle == "lower":
            self.setValues(min(self._upper, self._lower + step), self._upper)
        else:
            self.setValues(self._lower, max(self._lower, self._upper + step))
        event.accept()

    def _move_active(self, x: int) -> None:
        value = self._value_for_x(x)
        if self._active_handle == "lower":
            self.setValues(min(value, self._upper), self._upper)
        else:
            self.setValues(self._lower, max(value, self._lower))


class MetricRangeControl(QWidget):
    """Float range slider paired with exact lower/upper numeric inputs."""

    rangeChanged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._domain = (0.0, 1.0)
        self.slider = RangeSlider()
        self.lower = QDoubleSpinBox()
        self.upper = QDoubleSpinBox()
        for spinbox in (self.lower, self.upper):
            spinbox.setDecimals(9)
            spinbox.setKeyboardTracking(False)
        spin_layout = QHBoxLayout()
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.addWidget(self.lower)
        spin_layout.addWidget(self.upper)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.slider)
        layout.addLayout(spin_layout)
        self.slider.rangeChanged.connect(self._slider_changed)
        self.lower.valueChanged.connect(self._spin_changed)
        self.upper.valueChanged.connect(self._spin_changed)
        self.set_domain(0, 1)

    def set_domain(self, minimum: float, maximum: float, *, suffix: str = "") -> None:
        minimum, maximum = float(minimum), float(maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            minimum, maximum = 0.0, 1.0
        if maximum <= minimum:
            pad = max(1e-6, abs(minimum) * 0.01)
            minimum, maximum = max(0.0, minimum - pad), maximum + pad
        old_values = self.values()
        self._domain = (minimum, maximum)
        step = max((maximum - minimum) / 1000, 10 ** -self.lower.decimals())
        for spinbox in (self.lower, self.upper):
            spinbox.blockSignals(True)
            spinbox.setRange(minimum, maximum)
            spinbox.setSingleStep(step)
            spinbox.setSuffix(suffix)
            spinbox.blockSignals(False)
        if old_values == (0.0, 1.0):
            old_values = (minimum, maximum)
        self.set_values(
            max(minimum, min(old_values[0], maximum)),
            max(minimum, min(old_values[1], maximum)),
        )

    def set_values(self, lower: float, upper: float) -> None:
        minimum, maximum = self._domain
        lower = max(minimum, min(float(lower), maximum))
        upper = max(lower, min(float(upper), maximum))
        for spinbox, value in ((self.lower, lower), (self.upper, upper)):
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValues(self._to_slider(lower), self._to_slider(upper))
        self.slider.blockSignals(False)

    def values(self) -> tuple[float, float]:
        return self.lower.value(), self.upper.value()

    def _to_slider(self, value: float) -> int:
        minimum, maximum = self._domain
        return round(10_000 * (value - minimum) / (maximum - minimum))

    def _from_slider(self, value: int) -> float:
        minimum, maximum = self._domain
        return minimum + (maximum - minimum) * value / 10_000

    def _slider_changed(self, lower: int, upper: int) -> None:
        self.set_values(self._from_slider(lower), self._from_slider(upper))
        self.rangeChanged.emit(*self.values())

    def _spin_changed(self) -> None:
        lower, upper = self.values()
        if self.sender() is self.lower and lower > upper:
            upper = lower
        elif self.sender() is self.upper and upper < lower:
            lower = upper
        self.set_values(lower, upper)
        self.rangeChanged.emit(lower, upper)
