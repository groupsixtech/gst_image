"""Layered full-coordinate image canvas and drawing interactions."""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


def _qimage_bgr(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        return QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


class ImageCanvas(QGraphicsView):
    line_finished = Signal(str, object, object)
    rectangle_finished = Signal(str, object, object)
    polygon_finished = Signal(str, object)
    brush_stroke = Signal(str, object)
    brush_radius_adjust_requested = Signal(int)
    scene_clicked = Signal(object)
    coordinates_changed = Signal(object)

    LINE_TOOLS: ClassVar = {"calibrate", "measure", "split"}
    RECT_TOOLS: ClassVar = {"include", "exclude", "analysis_box"}
    BRUSH_TOOLS: ClassVar = {"seed", "seed_eraser", "mask_brush", "mask_eraser"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#16191d"))
        self.setMouseTracking(True)
        self._tool = "pan"
        self._source_size = (1, 1)
        self._preview_size = (1, 1)
        self._display_rect = QRectF(0, 0, 1, 1)
        self._base_item: QGraphicsPixmapItem | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._annotation_item: QGraphicsPathItem | None = None
        self._roi_item: QGraphicsPathItem | None = None
        self._start: QPointF | None = None
        self._temporary: QGraphicsItem | None = None
        self._polygon: list[QPointF] = []
        self._brush_points: list[QPointF] = []
        self._brush_radius = 12
        self._brush_cursor_item: QGraphicsEllipseItem | None = None

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    @property
    def preview_size(self) -> tuple[int, int]:
        return self._preview_size

    @property
    def current_tool(self) -> str:
        return self._tool

    @property
    def display_rect(self) -> QRectF:
        return QRectF(self._display_rect)

    def set_tool(self, tool: str) -> None:
        self._clear_temporary()
        self._polygon.clear()
        self._tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor
        )
        if self._brush_cursor_item is not None:
            self._brush_cursor_item.setVisible(tool in self.BRUSH_TOOLS)

    def set_brush_radius(self, radius: int) -> None:
        """Set the source-pixel brush radius used by the on-canvas outline."""
        self._brush_radius = max(1, int(radius))
        if self._brush_cursor_item is not None and self._brush_cursor_item.isVisible():
            center = self._brush_cursor_item.rect().center()
            self._position_brush_cursor(center)

    def set_image(
        self,
        preview_bgr: np.ndarray,
        full_size: tuple[int, int],
        source_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.scene().clear()
        self._temporary = None
        self._start = None
        self._brush_points.clear()
        self._preview_size = (preview_bgr.shape[1], preview_bgr.shape[0])
        self._source_size = full_size
        left, top, right, bottom = source_rect or (0, 0, full_size[0], full_size[1])
        if right <= left or bottom <= top:
            raise ValueError("Display source rectangle must have a positive size")
        self._display_rect = QRectF(left, top, right - left, bottom - top)
        pixmap = QPixmap.fromImage(_qimage_bgr(preview_bgr))
        self._base_item = self.scene().addPixmap(pixmap)
        self._base_item.setZValue(0)
        transform = QTransform.fromScale(
            (right - left) / preview_bgr.shape[1],
            (bottom - top) / preview_bgr.shape[0],
        )
        self._base_item.setTransform(transform)
        self._base_item.setPos(left, top)
        self.scene().setSceneRect(self._display_rect)
        self._overlay_item = None
        self._annotation_item = self.scene().addPath(QPainterPath(), QPen(QColor("white"), 2))
        self._annotation_item.setZValue(20)
        roi_pen = QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine)
        self._roi_item = self.scene().addPath(QPainterPath(), roi_pen)
        self._roi_item.setZValue(19)
        brush_pen = QPen(QColor("#ff2d95"), 2, Qt.PenStyle.SolidLine)
        brush_pen.setCosmetic(True)
        self._brush_cursor_item = self.scene().addEllipse(QRectF(), brush_pen)
        self._brush_cursor_item.setZValue(40)
        self._brush_cursor_item.setVisible(False)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _values_in_display_rect(self, values: np.ndarray) -> np.ndarray:
        """Crop full-source overlays when the canvas is displaying an ROI."""
        values = np.asarray(values)
        display_width = round(self._display_rect.width())
        display_height = round(self._display_rect.height())
        if values.shape[:2] == (display_height, display_width):
            return values
        source_width, source_height = self._source_size
        if values.shape[:2] == (source_height, source_width):
            left = max(0, round(self._display_rect.left()))
            top = max(0, round(self._display_rect.top()))
            right = min(source_width, left + display_width)
            bottom = min(source_height, top + display_height)
            return values[top:bottom, left:right]
        return values

    def _position_overlay(self, item: QGraphicsPixmapItem) -> None:
        item.setTransform(self._base_item.transform())
        item.setPos(self._base_item.pos())
        item.setZValue(10)

    def set_mask_overlay(
        self,
        mask: np.ndarray | None,
        color: tuple[int, int, int] = (255, 215, 0),
        opacity: float = 0.45,
    ) -> None:
        if self._base_item is None:
            return
        if self._overlay_item is not None:
            self.scene().removeItem(self._overlay_item)
            self._overlay_item = None
        if mask is None:
            return
        width, height = self._preview_size
        visible = self._values_in_display_rect(mask)
        small = cv2.resize(
            visible.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[small > 0, :3] = color
        rgba[small > 0, 3] = round(255 * opacity)
        image = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._overlay_item = self.scene().addPixmap(QPixmap.fromImage(image))
        self._position_overlay(self._overlay_item)

    def set_label_overlay(
        self,
        labels: np.ndarray | None,
        colors: dict[int, tuple[int, int, int]],
        opacity: float = 0.45,
        *,
        align_to_source: bool = False,
    ) -> None:
        """Display labels in either the current-view or full-source coordinate space."""
        if labels is None:
            self.set_mask_overlay(None)
            return
        if align_to_source:
            small = np.asarray(labels, dtype=np.int32)
            height, width = small.shape[:2]
        else:
            width, height = self._preview_size
            visible = self._values_in_display_rect(labels)
            small = cv2.resize(
                visible.astype(np.int32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for label, color in colors.items():
            member = small == label
            rgba[member, :3] = color
            rgba[member, 3] = round(255 * opacity)
        if self._overlay_item is not None:
            self.scene().removeItem(self._overlay_item)
        image = QImage(
            rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888
        ).copy()
        self._overlay_item = self.scene().addPixmap(QPixmap.fromImage(image))
        if align_to_source:
            source_width, source_height = self._source_size
            self._overlay_item.setTransform(
                QTransform.fromScale(source_width / width, source_height / height)
            )
            self._overlay_item.setPos(0, 0)
            self._overlay_item.setZValue(10)
        else:
            self._position_overlay(self._overlay_item)

    def set_particle_group_overlay(
        self,
        labels: np.ndarray,
        label_groups: dict[int, int],
        colors: dict[int, tuple[int, int, int]],
        opacity: float = 0.55,
    ) -> None:
        width, height = self._preview_size
        visible_labels = self._values_in_display_rect(labels)
        small_labels = cv2.resize(
            visible_labels.astype(np.int32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        lookup = np.zeros(max(1, int(small_labels.max()) + 1), dtype=np.uint16)
        for label, group in label_groups.items():
            if 0 <= label < lookup.size:
                lookup[label] = group
        self.set_label_overlay(lookup[small_labels], colors, opacity)

    def set_composite_overlays(
        self,
        layers: list[
            tuple[
                np.ndarray,
                tuple[int, int, int] | dict[int, tuple[int, int, int]],
                float,
            ]
        ],
    ) -> None:
        """Composite visible binary/instance and multi-class layers in preview space."""
        if self._base_item is None:
            return
        if self._overlay_item is not None:
            self.scene().removeItem(self._overlay_item)
            self._overlay_item = None
        if not layers:
            return
        width, height = self._preview_size
        premultiplied = np.zeros((height, width, 3), dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        for values, colors, opacity in layers:
            visible = self._values_in_display_rect(values)
            small = cv2.resize(
                visible.astype(np.int32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            layer_alpha = float(np.clip(opacity, 0, 1))
            if isinstance(colors, tuple):
                selections = [((small > 0), colors)]
            else:
                maximum = max(0, int(small.max()))
                lookup = np.zeros((maximum + 1, 3), dtype=np.uint8)
                present = np.zeros(maximum + 1, dtype=bool)
                for label, color in colors.items():
                    if 0 <= label <= maximum:
                        lookup[label] = color
                        present[label] = True
                safe = np.clip(small, 0, maximum)
                selected = (small >= 0) & (small <= maximum) & present[safe]
                selections = [(selected, lookup[safe])]
            for selected, selected_colors in selections:
                if not np.any(selected):
                    continue
                rgb = (
                    np.asarray(selected_colors, dtype=np.float32)
                    if isinstance(selected_colors, tuple)
                    else selected_colors[selected].astype(np.float32)
                )
                premultiplied[selected] = (
                    rgb * layer_alpha
                    + premultiplied[selected] * (1 - layer_alpha)
                )
                alpha[selected] = layer_alpha + alpha[selected] * (1 - layer_alpha)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        occupied = alpha > 0
        rgba[occupied, :3] = np.clip(
            premultiplied[occupied] / alpha[occupied, None], 0, 255
        ).astype(np.uint8)
        rgba[:, :, 3] = np.rint(255 * alpha).astype(np.uint8)
        image = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._overlay_item = self.scene().addPixmap(QPixmap.fromImage(image))
        self._position_overlay(self._overlay_item)

    def set_annotations(self, lines: list[tuple[QPointF, QPointF, QColor]]) -> None:
        path = QPainterPath()
        for start, end, _ in lines:
            path.moveTo(start)
            path.lineTo(end)
        if self._annotation_item is not None:
            self._annotation_item.setPath(path)

    def set_roi_outlines(self, polygons: list[list[QPointF]]) -> None:
        path = QPainterPath()
        for points in polygons:
            if len(points) < 2:
                continue
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        if self._roi_item is not None:
            self._roi_item.setPath(path)

    def wheelEvent(self, event) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mouseMoveEvent(self, event) -> None:
        point = self.mapToScene(event.position().toPoint())
        self.coordinates_changed.emit(point)
        if self._brush_cursor_item is not None and self._tool in self.BRUSH_TOOLS:
            self._position_brush_cursor(point)
            self._brush_cursor_item.setVisible(self._display_rect.contains(point))
        if self._start is not None and self._tool in self.LINE_TOOLS:
            self._draw_line(self._start, point)
        elif self._start is not None and self._tool in self.RECT_TOOLS:
            self._draw_rectangle(self._start, point)
        elif self._brush_points and self._tool in self.BRUSH_TOOLS:
            self._brush_points.append(point)
            self.brush_stroke.emit(self._tool, list(self._brush_points[-2:]))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._brush_cursor_item is not None:
            self._brush_cursor_item.setVisible(False)
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._tool in self.BRUSH_TOOLS:
            if event.key() == Qt.Key.Key_BracketLeft:
                self.brush_radius_adjust_requested.emit(-1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_BracketRight:
                self.brush_radius_adjust_requested.emit(1)
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        point = self.mapToScene(event.position().toPoint())
        if self._brush_cursor_item is not None and self._tool in self.BRUSH_TOOLS:
            self._position_brush_cursor(point)
            self._brush_cursor_item.setVisible(self._display_rect.contains(point))
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool in self.LINE_TOOLS | self.RECT_TOOLS:
                self._start = point
            elif self._tool == "polygon":
                self._polygon.append(point)
                self._draw_polygon()
            elif self._tool in self.BRUSH_TOOLS:
                self._brush_points = [point]
                self.brush_stroke.emit(self._tool, [point])
            elif self._tool in {"select", "eyedropper"}:
                self.scene_clicked.emit(point)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self._start is not None and self._tool in self.LINE_TOOLS:
                start = self._start
                self._start = None
                self._clear_temporary()
                self.line_finished.emit(self._tool, start, point)
            elif self._start is not None and self._tool in self.RECT_TOOLS:
                start = self._start
                self._start = None
                self._clear_temporary()
                self.rectangle_finished.emit(self._tool, start, point)
            elif self._brush_points:
                self._brush_points = []
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._tool == "polygon" and len(self._polygon) >= 3:
            points = list(self._polygon)
            self._polygon.clear()
            self._clear_temporary()
            self.polygon_finished.emit("include", points)
        super().mouseDoubleClickEvent(event)

    def _clear_temporary(self) -> None:
        if self._temporary is not None and self._temporary.scene() is not None:
            self.scene().removeItem(self._temporary)
        self._temporary = None

    def _position_brush_cursor(self, center: QPointF) -> None:
        if self._brush_cursor_item is None:
            return
        radius = self._brush_radius
        self._brush_cursor_item.setRect(
            center.x() - radius,
            center.y() - radius,
            radius * 2,
            radius * 2,
        )

    def _draw_line(self, start: QPointF, end: QPointF) -> None:
        self._clear_temporary()
        path = QPainterPath(start)
        path.lineTo(end)
        item = self.scene().addPath(path, QPen(QColor("white"), 2))
        item.setZValue(30)
        self._temporary = item

    def _draw_rectangle(self, start: QPointF, end: QPointF) -> None:
        self._clear_temporary()
        rectangle = QRectF(start, end).normalized()
        item = self.scene().addRect(rectangle, QPen(QColor("white"), 2))
        item.setZValue(30)
        self._temporary = item

    def _draw_polygon(self) -> None:
        self._clear_temporary()
        path = QPainterPath(self._polygon[0])
        for point in self._polygon[1:]:
            path.lineTo(point)
        item = self.scene().addPath(path, QPen(QColor("white"), 2))
        item.setZValue(30)
        self._temporary = item
