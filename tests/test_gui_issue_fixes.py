from types import SimpleNamespace

import cv2
import numpy as np
from gst_image_app import mainwindow as mainwindow_module
from gst_image_app.mainwindow import (
    MainWindow,
    _analyze_region_preview,
    _analyze_source,
    _analyze_source_region,
)
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence

from gst_image.analysis.calibration import create_measurement
from gst_image.models import (
    ROI,
    Calibration,
    Point,
    ROIKind,
    SegmentationLayer,
    SegmentationRecipe,
    ThresholdMethod,
)


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
    assert window.manual_threshold_high.value() == 66
    assert window.manual_threshold_low.value() == 0
    assert window.method.currentData() == ThresholdMethod.MANUAL

    window.eyedropper_target.setCurrentIndex(
        window.eyedropper_target.findData("threshold_low")
    )
    qtbot.mouseClick(
        window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point
    )
    assert window.manual_threshold_low.value() == 66
    assert window.manual_threshold_high.value() == 66

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


def test_class_seed_overlay_stays_in_source_coordinates_in_roi_view(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window.manifest.rois.append(_rectangle())
    window._refresh_rois()
    window.rois_list.item(0).setSelected(True)
    assert window.show_full_resolution_roi()

    window._activate_tool("seed")
    window._paint_training([QPointF(20, 16)])

    assert window.training_labels[4, 5] > 0
    overlay = window.canvas._overlay_item
    assert overlay is not None
    assert overlay.pixmap().size().toTuple() == (15, 10)
    assert overlay.pos() == QPointF(0, 0)
    assert overlay.transform().m11() == 4
    assert overlay.transform().m22() == 4
    assert overlay.mapToScene(QPointF(5, 4)) == QPointF(20, 16)
    window._set_dirty(False)


def test_bracket_shortcuts_adjust_active_mask_brush_radius(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window.show()
    window._set_dirty(False)
    window._activate_tool("mask_brush")
    window.brush_radius.setValue(12)
    window.activateWindow()
    window.canvas.setFocus()

    qtbot.keyClick(window.canvas, Qt.Key.Key_BracketRight)
    assert window.brush_radius.value() == 13
    qtbot.keyClick(window.canvas, Qt.Key.Key_BracketLeft)
    assert window.brush_radius.value() == 12


def test_middle_mouse_temporarily_pans_without_changing_active_tool(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.canvas.set_image(np.zeros((400, 600, 3), np.uint8), (600, 400))
    window._activate_tool("select")
    window.canvas.scale(3, 3)
    window.canvas.centerOn(300, 200)
    qtbot.wait(50)

    start = QPoint(250, 180)
    before = (
        window.canvas.horizontalScrollBar().value(),
        window.canvas.verticalScrollBar().value(),
    )
    qtbot.mousePress(
        window.canvas.viewport(), Qt.MouseButton.MiddleButton, pos=start
    )
    assert window.canvas.viewport().cursor().shape() == Qt.CursorShape.ClosedHandCursor
    qtbot.mouseMove(window.canvas.viewport(), pos=start + QPoint(40, 25))
    qtbot.mouseRelease(
        window.canvas.viewport(),
        Qt.MouseButton.MiddleButton,
        pos=start + QPoint(40, 25),
    )

    after = (
        window.canvas.horizontalScrollBar().value(),
        window.canvas.verticalScrollBar().value(),
    )
    assert after != before
    assert window.canvas.current_tool == "select"
    assert window.canvas.viewport().cursor().shape() == Qt.CursorShape.CrossCursor


def test_shift_a_triggers_preview_action(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    triggered = []
    window.preview_action.triggered.connect(lambda *_args: triggered.append(True))

    assert window.preview_action.shortcut() == QKeySequence("Shift+A")
    window.activateWindow()
    window.canvas.setFocus()
    qtbot.wait(50)
    qtbot.keyClick(window.canvas, Qt.Key.Key_A, Qt.KeyboardModifier.ShiftModifier)

    assert len(triggered) == 1


def test_particle_run_uses_and_preserves_selected_assisted_class(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    particle_class = next(
        item for item in window.manifest.classes if item.preset == "particle"
    )
    weld_class = next(item for item in window.manifest.classes if item.preset == "weld")

    window.seed_class.setCurrentIndex(window.seed_class.findData(weld_class.id))
    assert window.binary_class.currentData() == weld_class.id
    window.seed_class.setCurrentIndex(window.seed_class.findData(particle_class.id))
    assert window.binary_class.currentData() == particle_class.id

    started = []
    window.thread_pool = SimpleNamespace(start=started.append)
    window.run_segmentation()

    assert len(started) == 1
    recipe = started[0].args[3]
    assert recipe.target_class_id == particle_class.id
    window.worker = None
    window.run_recipe = recipe
    window.run_scope_ids = []
    labels = np.zeros((40, 60), dtype=np.int32)
    result = SimpleNamespace(mask=labels > 0, labels=labels, particles=[], summary={})
    window._particle_result(
        (labels.astype(np.uint8), np.ones_like(labels, bool), result, (0, 0, 60, 40))
    )

    layer = next(item for item in window.manifest.layers if item.kind == "instances")
    assert layer.class_id == particle_class.id
    assert window.seed_class.currentData() == particle_class.id
    assert window.binary_class.currentData() == particle_class.id
    window._set_dirty(False)


def test_mask_brush_and_eraser_edit_selected_result_with_immediate_overlay(
    qtbot, tmp_path
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window._set_dirty(False)
    layer = SegmentationLayer(name="Editable particles", kind="instances")
    labels = np.zeros((40, 60), dtype=np.int32)
    window.manifest.layers.append(layer)
    window.layer_masks[layer.id] = labels
    window.current_layer_id = layer.id
    window.current_labels = labels
    window.current_mask = np.zeros_like(labels, dtype=np.uint8)

    window._activate_tool("mask_brush")
    window._brush_stroke("mask_brush", [QPointF(20, 16)])
    assert window.current_mask[16, 20] == 1
    assert window.canvas._overlay_item is not None

    window._activate_tool("mask_eraser")
    window._brush_stroke("mask_eraser", [QPointF(20, 16)])
    assert window.current_mask[16, 20] == 0
    window.mask_commit_timer.stop()
    window._set_dirty(False)


def test_seed_eraser_is_persisted_and_replayed(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))

    window._paint_training([QPointF(20, 16)])
    assert window.training_labels[4, 5] > 0
    window._paint_training([QPointF(20, 16)], erase=True)
    assert window.training_labels[4, 5] == 0
    assert window.manifest.training_strokes[-1].erase

    window.training_labels.fill(0)
    window._restore_training_strokes()
    assert window.training_labels[4, 5] == 0
    window._set_dirty(False)


def test_region_training_warns_with_safe_overview_resolution(
    qtbot, tmp_path, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    window._set_dirty(False)
    monkeypatch.setattr(mainwindow_module, "REGION_CLASSIFICATION_MAX_PIXELS", 100)
    messages = []
    monkeypatch.setattr(
        mainwindow_module.QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    window.train_regions()

    assert window.worker is None
    assert len(messages) == 1
    title, message = messages[0]
    assert title == "Region-classification overview is too large"
    assert "15 x 10 pixels" in message
    assert "20% or lower" in message
    assert "Selected-ROI resolution does not affect" in message

    window._activate_tool("select")
    qtbot.keyClick(window.canvas, Qt.Key.Key_BracketRight)
    assert window.brush_radius.value() == 12


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
    payload = _analyze_source_region(
        path,
        (10, 8, 31, 25),
        [roi],
        [roi.id],
        recipe,
        None,
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    assert captured[-1] == ((17, 21), (17, 21), 101)
    assert payload[3] == (10, 8, 31, 25)

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


def test_final_selected_roi_run_dispatches_only_native_roi(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle()
    window.manifest.rois.append(roi)
    window._refresh_rois()
    window.rois_list.item(0).setSelected(True)
    window.selected_roi_only.setChecked(True)
    started = []
    window.thread_pool = SimpleNamespace(start=started.append)

    window.run_segmentation()

    assert len(started) == 1
    worker = started[0]
    assert worker.function is _analyze_source_region
    assert worker.args[1] == (10, 8, 31, 25)
    assert worker.args[3] == [roi.id]
    assert "21 × 17 pixels" in window.preview_status.text()
    window.worker = None
    window._set_dirty(False)


def test_selected_roi_result_is_placed_in_full_source_coordinates(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _source(tmp_path)
    window._load_new_image(path)
    roi = _rectangle()
    window.manifest.rois.append(roi)
    recipe = SegmentationRecipe(
        target_class_id=window.binary_class.currentData(),
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=255,
        illumination_correction=False,
        gaussian_blur_sigma=0,
        open_radius_px=0,
        close_radius_px=0,
        min_particle_area_px=1,
        split_touching=False,
    )
    payload = _analyze_source_region(
        path,
        (10, 8, 31, 25),
        [roi],
        [roi.id],
        recipe,
        None,
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    window.run_recipe = recipe
    window.run_scope_ids = [roi.id]

    window._particle_result(payload)

    assert window.current_labels.shape == (40, 60)
    assert not np.any(window.current_labels[:8])
    assert not np.any(window.current_labels[:, :10])
    assert np.any(window.current_labels[8:25, 10:31])
    assert window.particles[0].centroid_x_px == 20
    assert window.particles[0].centroid_y_px == 16
    summary = window.manifest.runs[-1].summary
    assert summary["analysis_scope"] == "selected_roi"
    assert summary["processed_width_px"] == 21
    assert summary["source_width_px"] == 60
    window._set_dirty(False)
