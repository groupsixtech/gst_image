from types import SimpleNamespace

import cv2
import numpy as np
from gst_image_app import mainwindow as mainwindow_module
from gst_image_app.mainwindow import MainWindow, _analyze_region_preview, _analyze_source
from PySide6.QtCore import QPointF, Qt

from gst_image.analysis.calibration import create_measurement
from gst_image.models import ROI, Calibration, Point, ROIKind, SegmentationRecipe, ThresholdMethod


def _source(tmp_path, *, shape=(40, 60), bgr=(30, 60, 90)):
    path = tmp_path / "source.png"
    image = np.empty((*shape, 3), dtype=np.uint8)
    image[:] = bgr
    assert cv2.imwrite(str(path), image)
    return path


def _rectangle(name="Analysis box", kind=ROIKind.ANALYSIS_BOX):
    return ROI(
        name=name,
        kind=kind,
        shape="rectangle",
        points=[Point(x=10, y=8), Point(x=30, y=24)],
    )


def test_full_resolution_roi_display_uses_native_pixels(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window.manifest.rois.append(_rectangle())
    window._refresh_rois()
    window.rois_list.item(0).setSelected(True)

    assert window.show_full_resolution_roi()
    assert window.canvas.preview_size == (21, 17)
    assert window.canvas.display_rect.left() == 10
    assert window.canvas.display_rect.top() == 8

    window.overview_resolution.setValue(100)
    assert window.show_overview()
    assert window.canvas.preview_size == (60, 40)

    window.overview_resolution.setValue(50)
    assert window.show_overview()
    assert window.canvas.preview_size == (30, 20)

    window.rois_list.item(0).setSelected(True)
    window.roi_resolution.setValue(50)
    assert window.show_full_resolution_roi()
    assert window.canvas.preview_size == (10, 8)
    window._set_dirty(False)


def test_eyedropper_reads_native_source_for_threshold_and_class_color(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path, bgr=(30, 60, 90)))
    window.resize(900, 600)
    window.show()
    window.canvas.set_tool("eyedropper")

    viewport_point = window.canvas.mapFromScene(QPointF(20, 20))
    qtbot.mouseClick(
        window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point
    )
    assert window.manual_threshold.value() == 66
    assert window.method.currentData() == ThresholdMethod.MANUAL

    window.eyedropper_target.setCurrentIndex(
        window.eyedropper_target.findData("binary_color")
    )
    qtbot.mouseClick(
        window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point
    )
    selected_class = next(
        item for item in window.manifest.classes if item.id == window.binary_class.currentData()
    )
    assert selected_class.color == "#5a3c1e"
    window._set_dirty(False)


def test_measurement_text_includes_calibrated_and_pixel_lengths(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window.manifest.calibration = Calibration(mm_per_pixel=0.1)
    measurement = create_measurement(
        "Measurement 1", (0, 0), (3, 4), window.manifest.calibration
    )
    window.manifest.measurements.append(measurement)
    window._refresh_measurements()

    assert window.measurements_list.item(1).text() == "Measurement 1: 0.5 mm (5.000 px)"
    window.measurements_list.item(1).setSelected(True)
    window.delete_selected_measurements()
    assert not window.manifest.measurements
    assert window.canvas._annotation_item.path().elementCount() == 0
    window._set_dirty(False)


def test_deleting_roi_removes_outline_and_dependent_layers(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle()
    window.manifest.rois.append(roi)
    layer = mainwindow_module.SegmentationLayer(
        name="ROI particles", kind="instances", scope_roi_ids=[roi.id]
    )
    window.manifest.layers.append(layer)
    window.layer_masks[layer.id] = np.ones((40, 60), dtype=np.uint8)
    window._refresh_all()
    window.rois_list.item(0).setSelected(True)

    window.delete_selected_rois()
    assert not window.manifest.rois
    assert not window.manifest.layers
    assert layer.id not in window.layer_masks
    assert window.canvas._roi_item.path().elementCount() == 0
    window._set_dirty(False)


def test_final_analysis_and_roi_preview_keep_native_resolution(monkeypatch, tmp_path):
    path = _source(tmp_path, shape=(37, 53))
    captured = []

    def fake_segment(image, domain, recipe, calibration, **_kwargs):
        captured.append((image.shape[:2], domain.shape, recipe.sauvola_window_px))
        return SimpleNamespace(mask=np.zeros(domain.shape, np.uint8), summary={})

    monkeypatch.setattr(mainwindow_module, "segment_particles", fake_segment)
    recipe = SegmentationRecipe(sauvola_window_px=101)
    _analyze_source(path, [], [], recipe, None, progress=lambda *_: None, cancelled=lambda: False)
    assert captured[-1] == ((37, 53), (37, 53), 101)

    roi = _rectangle()
    native_region = np.zeros((17, 21, 3), dtype=np.uint8)
    _analyze_region_preview(
        native_region,
        (10, 8, 31, 25),
        [roi],
        [roi.id],
        recipe,
        None,
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    assert captured[-1] == ((17, 21), (17, 21), 101)
