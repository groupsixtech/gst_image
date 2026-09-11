"""Layered full-coordinate image canvas and drawing interactions."""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
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
    scene_clicked = Signal(object)
    coordinates_changed = Signal(object)

    LINE_TOOLS: ClassVar = {"calibrate", "measure", "split"}
    RECT_TOOLS: ClassVar = {"include", "exclude", "analysis_box"}
    BRUSH_TOOLS: ClassVar = {"seed", "mask_brush", "mask_eraser"}

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
        self._base_item: QGraphicsPixmapItem | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._annotation_item: QGraphicsPathItem | None = None
        self._roi_item: QGraphicsPathItem | None = None
        self._start: QPointF | None = None
        self._temporary: QGraphicsItem | None = None
        self._polygon: list[QPointF] = []
        self._brush_points: list[QPointF] = []

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    @property
    def preview_size(self) -> tuple[int, int]:
        return self._preview_size

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

    def set_image(self, preview_bgr: np.ndarray, full_size: tuple[int, int]) -> None:
        self.scene().clear()
        self._preview_size = (preview_bgr.shape[1], preview_bgr.shape[0])
        self._source_size = full_size
        pixmap = QPixmap.fromImage(_qimage_bgr(preview_bgr))
        self._base_item = self.scene().addPixmap(pixmap)
        self._base_item.setZValue(0)
        transform = QTransform.fromScale(
            full_size[0] / preview_bgr.shape[1], full_size[1] / preview_bgr.shape[0]
        )
        self._base_item.setTransform(transform)
        self.scene().setSceneRect(QRectF(0, 0, full_size[0], full_size[1]))
        self._overlay_item = None
        self._annotation_item = self.scene().addPath(QPainterPath(), QPen(QColor("white"), 2))
        self._annotation_item.setZValue(20)
        roi_pen = QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine)
        self._roi_item = self.scene().addPath(QPainterPath(), roi_pen)
        self._roi_item.setZValue(19)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

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
        small = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
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
        self._overlay_item.setTransform(self._base_item.transform())
        self._overlay_item.setZValue(10)

    def set_label_overlay(
        self, labels: np.ndarray | None, colors: dict[int, tuple[int, int, int]], opacity: float = 0.45
    ) -> None:
        if labels is None:
            self.set_mask_overlay(None)
            return
        width, height = self._preview_size
        small = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
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
        self._overlay_item.setTransform(self._base_item.transform())
        self._overlay_item.setZValue(10)

    def set_particle_group_overlay(
        self,
        labels: np.ndarray,
        label_groups: dict[int, int],
        colors: dict[int, tuple[int, int, int]],
        opacity: float = 0.55,
    ) -> None:
        width, height = self._preview_size
        small_labels = cv2.resize(
            labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST
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
            small = cv2.resize(
                np.asarray(values).astype(np.int32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            selections = (
                ((small > 0), colors),
            ) if isinstance(colors, tuple) else (
                (small == label, color) for label, color in colors.items()
            )
            layer_alpha = float(np.clip(opacity, 0, 1))
            for selected, color in selections:
                if not np.any(selected):
                    continue
                premultiplied[selected] = (
                    np.asarray(color, dtype=np.float32) * layer_alpha
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
        self._overlay_item.setTransform(self._base_item.transform())
        self._overlay_item.setZValue(10)

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
        if self._start is not None and self._tool in self.LINE_TOOLS:
            self._draw_line(self._start, point)
        elif self._start is not None and self._tool in self.RECT_TOOLS:
            self._draw_rectangle(self._start, point)
        elif self._brush_points and self._tool in self.BRUSH_TOOLS:
            self._brush_points.append(point)
            self.brush_stroke.emit(self._tool, list(self._brush_points[-2:]))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        point = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool in self.LINE_TOOLS | self.RECT_TOOLS:
                self._start = point
            elif self._tool == "polygon":
                self._polygon.append(point)
                self._draw_polygon()
            elif self._tool in self.BRUSH_TOOLS:
                self._brush_points = [point]
                self.brush_stroke.emit(self._tool, [point])
            elif self._tool == "select":
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
