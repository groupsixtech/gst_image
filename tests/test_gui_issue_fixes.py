from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from gst_image_app import mainwindow as mainwindow_module
from gst_image_app.mainwindow import (
    MainWindow,
    _analyze_preview,
    _analyze_region_preview,
    _analyze_source,
    _analyze_source_region,
    _classify_source_region,
    _particle_volume_pie_data,
    _region_training_labels,
)
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from gst_image.analysis.calibration import create_measurement
from gst_image.analysis.particles import measure_particles
from gst_image.models import (
    ROI,
    Calibration,
    ParticleGroup,
    ParticleGrouping,
    Point,
    RegionAnalysis,
    ROIKind,
    SegmentationLayer,
    SegmentationRecipe,
    ThresholdMethod,
    TrainingStroke,
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


def test_o_and_p_select_mask_brush_and_eraser(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.canvas.setFocus()
    qtbot.wait(50)

    assert window.tool_actions["mask_brush"].shortcut() == QKeySequence("O")
    assert window.tool_actions["mask_eraser"].shortcut() == QKeySequence("P")
    qtbot.keyClick(window.canvas, Qt.Key.Key_O)
    assert window.canvas.current_tool == "mask_brush"
    assert window.tool_actions["mask_brush"].isChecked()
    qtbot.keyClick(window.canvas, Qt.Key.Key_P)
    assert window.canvas.current_tool == "mask_eraser"
    assert window.tool_actions["mask_eraser"].isChecked()


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
    roi = _rectangle()
    window.manifest.rois.append(roi)
    window._refresh_rois()
    window.rois_list.item(0).setSelected(True)
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
    assert title == "Region-classification ROI is too large"
    assert "21 x 17 pixels" in message
    assert "Selected ROI resolution to 54% or lower" in message

    window._activate_tool("select")
    qtbot.keyClick(window.canvas, Qt.Key.Key_BracketRight)
    assert window.brush_radius.value() == 12


def test_region_training_uses_only_selected_analysis_box(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle("Classifier box")
    window.manifest.rois.append(roi)
    class_ids = [item.id for item in window._trainable_classes()[:2]]
    window.manifest.training_strokes.extend(
        [
            TrainingStroke(
                class_id=class_ids[0],
                points=[Point(x=14, y=12)],
                radius_px=1,
            ),
            TrainingStroke(
                class_id=class_ids[1],
                points=[Point(x=26, y=20)],
                radius_px=1,
            ),
            TrainingStroke(
                class_id=class_ids[0],
                points=[Point(x=50, y=35)],
                radius_px=1,
            ),
        ]
    )
    window._refresh_rois()
    window.rois_list.item(0).setSelected(True)
    started = []
    window.thread_pool = SimpleNamespace(start=started.append)

    window.train_regions()

    assert len(started) == 1
    worker = started[0]
    assert worker.function is _classify_source_region
    assert worker.args[1] == (10, 8, 31, 25)
    assert worker.args[2] == 100
    assert worker.args[3].shape == (17, 21)
    assert set(np.unique(worker.args[3])) == {0, 1, 2}
    assert worker.args[5] == roi.id

    local_labels = np.ones((17, 21), dtype=np.int32)
    local_labels[:, 11:] = 2
    window._region_result(
        (RegionAnalysis(local_labels, [1, 2], {"training_pixels": 10}), worker.args[1], roi.id)
    )
    layer = next(item for item in window.manifest.layers if item.kind == "multiclass")
    saved = window.layer_masks[layer.id]
    assert layer.scope_roi_ids == [roi.id]
    assert saved.shape == (40, 60)
    assert not np.any(saved[:8])
    assert not np.any(saved[:, :10])
    assert set(np.unique(saved[8:25, 10:31])) == {1, 2}
    window.worker = None
    window._set_dirty(False)


def test_region_training_label_rasterization_excludes_other_boxes():
    bounds = (10, 8, 31, 25)
    strokes = [
        TrainingStroke(
            class_id="inside", points=[Point(x=15, y=12)], radius_px=1
        ),
        TrainingStroke(
            class_id="outside", points=[Point(x=50, y=35)], radius_px=1
        ),
    ]

    labels = _region_training_labels(strokes, ["inside", "outside"], bounds, (17, 21))

    assert np.any(labels == 1)
    assert not np.any(labels == 2)


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
    started = []
    window.thread_pool = SimpleNamespace(start=started.append)

    window.run_segmentation()

    assert len(started) == 1
    worker = started[0]
    assert worker.function is _analyze_source_region
    assert worker.args[1] == (10, 8, 31, 25)
    assert worker.args[3] == [roi.id]
    assert roi.recipe_id == worker.args[4].id
    assert worker.args[4].name == f"{roi.name} recipe"
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


def test_volume_fraction_pie_uses_group_area_and_analysis_remainder():
    buckets = [
        ("Fine", "#112233", [SimpleNamespace(area_px=10), SimpleNamespace(area_px=15)]),
        ("Coarse", "#445566", [SimpleNamespace(area_px=35)]),
    ]

    labels, areas, colors, title = _particle_volume_pie_data(buckets, 100)

    assert labels == ["Fine", "Coarse", "Matrix / unshown"]
    assert areas == [25, 35, 40]
    assert colors == ["#112233", "#445566", "#30343b"]
    assert title == "Estimated volume fraction\n(area basis)"


def test_copy_current_roi_mask_places_native_binary_crop_on_clipboard(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle("Copy ROI")
    layer = SegmentationLayer(
        name="Copy particles", kind="instances", scope_roi_ids=[roi.id]
    )
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[10:13, 12:16] = 1
    window.manifest.rois.append(roi)
    window.manifest.layers.append(layer)
    window.layer_masks[layer.id] = labels
    window.current_layer_id = layer.id
    window.current_labels = labels
    window.current_mask = (labels > 0).astype(np.uint8)
    window._refresh_all()
    window.rois_list.item(0).setSelected(True)

    window.copy_current_roi_mask()

    image = QApplication.clipboard().image()
    assert (image.width(), image.height()) == (21, 17)
    assert image.pixelColor(0, 0).red() == 0
    assert image.pixelColor(2, 2).red() == 255
    assert "Copy ROI" in window.statusBar().currentMessage()
    window._set_dirty(False)


def test_save_particle_grouping_overlay_writes_native_roi_image(
    qtbot, tmp_path, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle("Grouping ROI")
    layer = SegmentationLayer(
        name="Grouped particles", kind="instances", scope_roi_ids=[roi.id]
    )
    labels = np.zeros((40, 60), dtype=np.int32)
    cv2.circle(labels, (20, 16), 4, 1, cv2.FILLED)
    particles = measure_particles(labels)
    grouping = ParticleGrouping(
        name="Presentation groups",
        source_layer_id=layer.id,
        groups=[
            ParticleGroup(
                name="Red particles",
                color="#ff0000",
                size_unit="px",
                size_min=0,
                size_max=100,
                circularity_min=0,
                circularity_max=1,
            )
        ],
    )
    window.manifest.rois.append(roi)
    window.manifest.layers.append(layer)
    window.manifest.particle_records[layer.id] = particles
    window.layer_masks[layer.id] = labels
    window.current_layer_id = layer.id
    window.current_labels = labels
    window.current_mask = (labels > 0).astype(np.uint8)
    window.particles = particles
    window._refresh_all()
    window._preview_particle_grouping(grouping)
    target = tmp_path / "grouped-roi.png"
    monkeypatch.setattr(
        mainwindow_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), "PNG images (*.png)"),
    )

    saved = window.save_particle_grouping_overlay()

    assert saved == target
    image = cv2.imread(str(target))
    assert image.shape[:2] == (17, 21)
    assert image[8, 10, 2] > image[8, 10, 1]
    window._set_dirty(False)


def test_selecting_roi_switches_its_particle_layer_and_saved_scheme(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    first_roi = _rectangle("ROI one")
    second_roi = ROI(
        name="ROI two",
        kind=ROIKind.ANALYSIS_BOX,
        shape="rectangle",
        points=[Point(x=32, y=8), Point(x=50, y=24)],
    )
    class_id = window.binary_class.currentData()
    first_layer = SegmentationLayer(
        name="Particles — ROI one",
        class_id=class_id,
        kind="instances",
        scope_roi_ids=[first_roi.id],
    )
    second_layer = SegmentationLayer(
        name="Particles — ROI two",
        class_id=class_id,
        kind="instances",
        scope_roi_ids=[second_roi.id],
    )
    first_labels = np.zeros((40, 60), dtype=np.int32)
    second_labels = np.zeros((40, 60), dtype=np.int32)
    cv2.circle(first_labels, (20, 16), 3, 1, cv2.FILLED)
    cv2.circle(second_labels, (40, 16), 4, 1, cv2.FILLED)
    first_scheme = ParticleGrouping(
        name="ROI one scheme",
        source_layer_id=first_layer.id,
        groups=[
            ParticleGroup(
                name="One",
                size_unit="px",
                size_min=0,
                size_max=100,
                circularity_min=0,
                circularity_max=1,
            )
        ],
    )
    second_scheme = ParticleGrouping(
        name="ROI two scheme",
        source_layer_id=second_layer.id,
        groups=[
            ParticleGroup(
                name="Two",
                size_unit="px",
                size_min=0,
                size_max=100,
                circularity_min=0,
                circularity_max=1,
            )
        ],
    )
    window.manifest.rois.extend([first_roi, second_roi])
    window.manifest.layers.extend([first_layer, second_layer])
    window.layer_masks[first_layer.id] = first_labels
    window.layer_masks[second_layer.id] = second_labels
    window.manifest.particle_records[first_layer.id] = measure_particles(first_labels)
    window.manifest.particle_records[second_layer.id] = measure_particles(second_labels)
    window.manifest.particle_groupings.extend([first_scheme, second_scheme])
    window.manifest.active_particle_groupings = {
        first_layer.id: first_scheme.id,
        second_layer.id: second_scheme.id,
    }
    window._refresh_all()

    second_item = window.rois_list.item(1)
    second_item.setSelected(True)
    window._roi_selected(second_item)
    assert window.current_layer_id == second_layer.id
    assert window.grouping_panel.scheme_combo.currentData() == second_scheme.id
    assert window.grouping_preview_definition.id == second_scheme.id
    window._refresh_all()
    assert [
        item.data(Qt.ItemDataRole.UserRole)
        for item in window.rois_list.selectedItems()
    ] == [second_roi.id]

    window.rois_list.clearSelection()
    first_item = window.rois_list.item(0)
    first_item.setSelected(True)
    window._roi_selected(first_item)
    assert window.current_layer_id == first_layer.id
    assert window.grouping_panel.scheme_combo.currentData() == first_scheme.id
    assert window.grouping_preview_definition.id == first_scheme.id
    window._set_dirty(False)


def test_cellpose_worker_runs_each_analysis_box_and_skips_the_rest_of_the_overview(
    tmp_path, monkeypatch
):
    path = _source(tmp_path, shape=(40, 60))
    first = _rectangle(name="Box A")
    second = ROI(
        name="Box B",
        kind=ROIKind.ANALYSIS_BOX,
        shape="rectangle",
        points=[Point(x=40, y=4), Point(x=54, y=18)],
    )
    include = _rectangle(name="Include", kind=ROIKind.INCLUDE)
    captured = {}

    def fake_inference(source, domain, recipe, calibration, **kwargs):
        captured.update(domain=domain, regions=kwargs["regions"])
        return SimpleNamespace(
            labels=np.zeros(domain.shape, dtype=np.int32),
            particles=[],
            summary={},
            source_bounds_px=(0, 0, 1, 1),
            region_bounds_px=list(kwargs["regions"]),
            cellpose_version="x",
            torch_version="x",
            model_sha256="x",
            model_cache_path="x",
        )

    monkeypatch.setattr(mainwindow_module, "run_cellpose_inference", fake_inference)
    recipe = SimpleNamespace(modality="biological", device="cpu")

    mainwindow_module._analyze_cellpose_source(
        path,
        [first, second, include],
        [],
        recipe,
        None,
        False,
        progress=lambda value, message: None,
        cancelled=lambda: False,
    )

    # Both Analysis boxes are analysed; the Include ROI does not widen the scope.
    assert captured["regions"] == [(10, 8, 31, 25), (40, 4, 55, 19)]
    assert not captured["domain"][0, 0]
    assert captured["domain"][8, 10] and captured["domain"][4, 40]


def test_cellpose_worker_refuses_to_run_without_an_analysis_box(tmp_path):
    path = _source(tmp_path)
    with pytest.raises(ValueError, match="Analysis box"):
        mainwindow_module._analyze_cellpose_source(
            path,
            [_rectangle(name="Include", kind=ROIKind.INCLUDE)],
            [],
            SimpleNamespace(modality="biological", device="cpu"),
            None,
            False,
            progress=lambda value, message: None,
            cancelled=lambda: False,
        )


def _auto_accept_dialogs(monkeypatch):
    """Accept modal dialogs without an event loop by swapping the class the GUI builds."""

    class _Accepted(mainwindow_module.QDialog):
        def exec(self):
            return mainwindow_module.QDialog.DialogCode.Accepted

    monkeypatch.setattr(mainwindow_module, "QDialog", _Accepted)


def test_cellpose_dialog_defaults_to_gpu_and_needs_no_licence_or_download_ticks(
    qtbot, tmp_path, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    monkeypatch.setattr(mainwindow_module, "_cuda_is_available", lambda: True)
    _auto_accept_dialogs(monkeypatch)

    recipe = window._cellpose_recipe_dialog([_rectangle()])

    # Both former checkboxes are gone; the run is licensed and may fetch weights.
    assert recipe.device == "gpu"
    assert recipe.noncommercial_license_accepted
    assert recipe.diameter_px == 46
    assert (recipe.flow_threshold, recipe.cellprob_threshold) == (0.4, 0.0)
    window._set_dirty(False)


def test_cellpose_dialog_falls_back_to_cpu_without_a_visible_cuda_device(
    qtbot, tmp_path, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    monkeypatch.setattr(mainwindow_module, "_cuda_is_available", lambda: False)
    _auto_accept_dialogs(monkeypatch)

    assert window._cellpose_recipe_dialog([_rectangle()]).device == "cpu"
    window._set_dirty(False)


def test_cellpose_dialog_lists_every_analysis_box_it_will_segment(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path, shape=(40, 60)))
    small = ROI(
        name="Small",
        kind=ROIKind.ANALYSIS_BOX,
        shape="rectangle",
        points=[Point(x=2, y=2), Point(x=6, y=6)],
    )

    widget = window._cellpose_scope_list([small, _rectangle(name="Large")])

    # Largest box first, each with its own pixel bounds.
    assert widget.count() == 2
    assert widget.item(0).text().startswith("Large — 21 × 17 px at (10, 8)")
    assert widget.item(1).text().startswith("Small — 5 × 5 px at (2, 2)")
    window._set_dirty(False)


def test_analysis_scope_follows_roi_selection_without_a_preview_resolution_control(
    qtbot, tmp_path
):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle()
    window.manifest.rois.append(roi)
    window._refresh_rois()

    # The redundant controls are gone; scope is implied by the ROI selection.
    assert not hasattr(window, "preview_resolution")
    assert not hasattr(window, "selected_roi_only")
    assert "whole image" in window.analysis_scope_label.text()

    started = []
    window.thread_pool = SimpleNamespace(start=started.append)
    window.run_segmentation()
    assert started[0].function is _analyze_source
    assert started[0].args[2] == []
    window.worker = None

    window.rois_list.item(0).setSelected(True)
    window._roi_selected(window.rois_list.item(0))
    assert roi.name in window.analysis_scope_label.text()

    started.clear()
    window.run_segmentation()
    assert started[0].function is _analyze_source_region
    assert started[0].args[3] == [roi.id]
    window.worker = None
    window._set_dirty(False)


def test_preview_scope_and_resolution_follow_the_roi_selection(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(_source(tmp_path))
    roi = _rectangle()
    window.manifest.rois.append(roi)
    window._refresh_rois()
    window.roi_resolution.setValue(100)
    started = []
    window.thread_pool = SimpleNamespace(start=started.append)

    # No selection previews the overview at the overview resolution.
    window.run_preview_segmentation()
    assert started[-1].function is _analyze_preview
    assert started[-1].args[3] == []
    window.worker = None

    window.rois_list.item(0).setSelected(True)
    window.run_preview_segmentation()
    assert started[-1].function is _analyze_region_preview
    assert started[-1].args[1] == (10, 8, 31, 25)
    assert started[-1].args[3] == [roi.id]
    window.worker = None
    window._set_dirty(False)


def test_group_statistics_table_totals_and_sorts_by_column(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.grouping_panel
    panel.set_statistics(
        {
            "Coarse": {"count": 3, "count_percent": 30.0, "area_px": 300.0,
                       "area_fraction": 0.03, "size": {"mean": 9.0, "median": 8.0},
                       "circularity": {"mean": 0.5}},
            "Fine": {"count": 7, "count_percent": 70.0, "area_px": 100.0,
                     "area_fraction": 0.01, "size": {"mean": 2.0, "median": 1.5},
                     "circularity": {"mean": 0.9}},
        }
    )
    table = panel.statistics_table

    # A bold, non-summable-columns-blank total row closes the table.
    assert table.rowCount() == 3
    assert [table.item(2, column).text() for column in range(5)] == [
        "Total",
        "10",
        "100",
        "400",
        "4",
    ]
    assert [table.item(2, column).text() for column in range(5, 8)] == ["", "", ""]
    assert table.item(2, 0).font().bold()
    assert [table.item(row, 0).text() for row in range(2)] == ["Coarse", "Fine"]

    # Numeric columns sort by value, ascending then descending, total stays last.
    panel._statistics_header_clicked(1)
    assert [table.item(row, 0).text() for row in range(3)] == ["Coarse", "Fine", "Total"]
    panel._statistics_header_clicked(1)
    assert [table.item(row, 0).text() for row in range(3)] == ["Fine", "Coarse", "Total"]
    panel._statistics_header_clicked(3)
    assert [table.item(row, 0).text() for row in range(3)] == ["Fine", "Coarse", "Total"]
