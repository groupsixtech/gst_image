from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from gst_image_app.mainwindow import MainWindow
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QMessageBox

from gst_image.analysis.weld_dilution import fit_surface_reference
from gst_image.models import Calibration, Point
from gst_image.project import load_project, save_project, validate_project


@pytest.fixture
def workspace(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    source = tmp_path / "panorama.png"
    image = np.full((100, 180, 3), 180, np.uint8)
    image[25:75, 20:160] = 80
    cv2.imwrite(str(source), image)
    window = MainWindow()
    qtbot.addWidget(window)
    window._load_new_image(source)
    window.show()
    window.dilution.show()
    yield window, window.dilution
    window.dilution.cancel()
    window.thread_pool.waitForDone(10000)
    window.dirty = False


def install_envelope(window, controller):
    mask = np.zeros((100, 180), np.uint8)
    mask[25:75, 20:160] = 1
    domain = np.ones_like(mask)
    controller._replace_working_masks(mask, domain)
    controller.line_finished("weld_line", QPointF(2, 49.5), QPointF(178, 49.5))
    return mask


def test_line_fit_flip_edit_and_no_particle_records(workspace):
    window, c = workspace
    install_envelope(window, c)
    assert c.draft.reference.span_px == 176
    c.flip.setChecked(True)
    assert c.draft.reference.normal.y == -1
    c.points.item(0, 1).setText("48.5")
    assert c.draft.reference.rms_residual_px < 1e-8
    before = c.envelope.copy()
    c.edit_envelope([QPointF(80, 25)], erase=True)
    assert not np.array_equal(c.envelope, before)
    assert window.manifest.particle_records == {}
    window.undo_stack.undo()
    assert np.array_equal(c.envelope, before)
    window.undo_stack.redo()
    assert not np.array_equal(c.envelope, before)


def test_calculate_review_stale_calibration_and_roundtrip(workspace, qtbot, tmp_path):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    qtbot.waitUntil(lambda: c.run is not None, timeout=15000)
    assert c.run.summary.dilution_percent == 50
    c.confirm()
    assert c.run.review_status == "confirmed"
    window.manifest.calibration = Calibration(mm_per_pixel=0.01)
    window._set_dirty()
    assert c.run.outdated and c.run.review_status == "pending"
    assert not c.draft.segmentation_stale
    c.calculate()
    qtbot.waitUntil(lambda: not c.run.outdated, timeout=15000)
    assert c.run.summary.depth_mm.mean == pytest.approx(0.25)
    root = save_project(tmp_path / "gui.gstproj", window.manifest, window.layer_masks)
    assert validate_project(root) == []
    loaded, masks = load_project(root)
    window.manifest, window.layer_masks = loaded, masks
    window._set_dirty(False)
    assert c.run.summary.dilution_percent == 50
    assert c.draft.reference == loaded.dilution_drafts[0].reference


def test_preview_then_native_candidates_select_and_calculate(workspace, qtbot):
    window, c = workspace
    c.sigma.setValue(2)
    c.window_size.setValue(51)
    c.segment(preview=True)
    qtbot.waitUntil(lambda: c.preview_result is not None, timeout=15000)
    assert not window.manifest.dilution_runs
    c.segment()
    qtbot.waitUntil(lambda: c.candidates is not None, timeout=15000)
    assert c.envelope.shape == (100, 180)
    ys, xs = np.nonzero(c.candidates)
    c._tool("weld_select")
    c.scene_clicked(QPointF(float(xs[len(xs) // 2]), float(ys[len(ys) // 2])))
    assert c.envelope.any()
    assert c.draft.component_seeds
    c.line_finished("weld_line", QPointF(2, 49.5), QPointF(178, 49.5))
    c.calculate()
    qtbot.waitUntil(lambda: c.run is not None, timeout=15000)
    assert not window.manifest.particle_records


def test_plot_selection_crop_alignment_and_polygons(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    qtbot.waitUntil(lambda: c.run is not None, timeout=15000)
    c.plot_picked(SimpleNamespace(ind=[80]))
    assert c.selected_line == 80
    roi_count = len(window.manifest.rois)
    points = [QPointF(20, 25), QPointF(40, 25), QPointF(40, 40)]
    window.canvas.polygon_finished.emit("weld_polygon_remove", points)
    assert len(window.manifest.rois) == roi_count
    assert c.run.outdated
    crop = np.full((50, 80, 3), 100, np.uint8)
    window._set_canvas_image(crop, (40, 20, 120, 70))
    c.render()
    assert window.canvas._overlay_item.pos() == QPointF(40, 20)
    assert window.canvas._overlay_item.pixmap().width() == 80


def test_cancelled_or_superseded_result_is_not_committed(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    c.cancel()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert window.manifest.dilution_runs == []
    c.calculate()
    c.line_finished("weld_line", QPointF(0, 40), QPointF(179, 40))
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert window.manifest.dilution_runs == []


def test_multiple_named_analyses_and_layer_deletion(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    qtbot.waitUntil(lambda: c.run is not None, timeout=15000)
    first = c.draft
    run_id, layer_id = c.run.id, c.run.envelope_layer_id
    c.new_draft()
    assert c.draft.id != first.id
    c.analysis_choice.setCurrentIndex(c.analysis_choice.findData(first.id))
    assert c.run.id == run_id
    removed = window._remove_layers({layer_id})
    assert layer_id in removed
    assert first.last_run_id is None
    assert not window.manifest.dilution_runs


def test_fit_rejects_degenerate_points_and_keeps_valid_tilt(workspace):
    _, c = workspace
    c.set_points([Point(x=1, y=1), Point(x=1, y=1)])
    c.fit_points()
    assert c.draft.reference is None
    points = [Point(x=10, y=10), Point(x=40, y=15), Point(x=100, y=25)]
    c.set_points(points)
    c.fit_points()
    assert c.draft.reference == fit_surface_reference(points)


def test_workspace_screenshot_and_standard_brush_routes_to_weld(workspace, qtbot, tmp_path):
    from matplotlib.font_manager import findfont
    from PySide6.QtGui import QFont, QFontDatabase

    window, c = workspace
    font_id = QFontDatabase.addApplicationFont(findfont("DejaVu Sans"))
    window.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    install_envelope(window, c)
    window.manifest.calibration = Calibration(mm_per_pixel=0.02)
    window._set_dirty()
    c.calculate()
    qtbot.waitUntil(lambda: c.run is not None, timeout=15000)
    c.result_tabs.setCurrentIndex(2)
    c.stage.setCurrentIndex(0)
    c.render()
    qtbot.wait(100)
    assert window.grab().save(str(tmp_path / "weld_workspace.png"))
    before = c.envelope.copy()
    window._brush_stroke("mask_eraser", [QPointF(70, 25)])
    assert not np.array_equal(c.envelope, before)
    assert not window.manifest.particle_records
    assert c.run.outdated


def test_brush_shortcut_matches_cursor_and_mask_radius(workspace):
    window, c = workspace
    install_envelope(window, c)
    c._tool("weld_brush")
    previous = c.radius.value()
    window._adjust_brush_radius(1)
    assert c.radius.value() == previous + 1
    assert window.canvas._brush_radius == c.radius.value()


def test_surface_points_draft_and_component_undo(workspace):
    window, c = workspace
    c._tool("weld_points")
    c.scene_clicked(QPointF(3, 4))
    assert c.draft.surface_points == [Point(x=3, y=4)]
    assert c.draft.reference is None
    mask = install_envelope(window, c)
    c.candidates = mask.astype(np.int32)
    c.envelope[:] = 0
    c._tool("weld_select")
    c.scene_clicked(QPointF(30, 40))
    assert c.envelope.any() and c.draft.component_seeds
    window.undo_stack.undo()
    assert not c.envelope.any() and c.draft.component_seeds == []
    window.undo_stack.redo()
    assert c.envelope.any() and c.draft.component_seeds


def test_blurred_view_restores_original_and_blocks_provisional_edits(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.preview_blur()
    qtbot.waitUntil(lambda: c.worker is None and c.blurred_preview is not None, timeout=15000)
    c.stage.setCurrentIndex(0)
    c.segment()
    qtbot.waitUntil(lambda: c.worker is None and c.candidates is not None, timeout=15000)
    original = window.canvas._base_item.pixmap().toImage()
    c.stage.setCurrentIndex(1)
    assert c.blurred_preview is not None
    assert window.canvas._base_item.pixmap().toImage() != original
    c.stage.setCurrentIndex(0)
    assert window.canvas._base_item.pixmap().toImage() == original
    c.segment(preview=True)
    qtbot.waitUntil(lambda: c.worker is None and c.preview_result is not None, timeout=15000)
    before = c.envelope.copy()
    c._tool("weld_select")
    c.scene_clicked(QPointF(30, 40))
    assert np.array_equal(c.envelope, before)
    c.edit_envelope([QPointF(30, 40)])
    assert np.array_equal(c.envelope, before)
    c._tool("weld_brush")
    assert c.preview_result is None
    assert window.canvas._base_item.pixmap().toImage() == original


def test_blur_preview_is_independent_of_segmentation(workspace, qtbot, monkeypatch):
    window, c = workspace

    def unexpected_segmentation(*args, **kwargs):
        raise AssertionError("Blur preview must not threshold or create candidates")

    monkeypatch.setattr(
        "gst_image_app.weld_dilution.segment_weld_envelope", unexpected_segmentation,
    )
    # Blur inspection needs neither a surface reference nor a selected analysis ROI.
    c.scope.setCurrentIndex(1)
    original = window.canvas._base_item.pixmap().toImage()
    c.blur_button.click()
    qtbot.waitUntil(lambda: c.worker is None and c.blurred_preview is not None, timeout=15000)
    assert "Blur-only preview" in c.status.text()
    assert c.stage.currentIndex() == 1
    assert c.candidates is None and c.preview_result is None
    assert c.envelope is None and not window.manifest.dilution_runs
    assert not c.annotation_items
    gray = cv2.imread(str(window.current_path), cv2.IMREAD_GRAYSCALE)
    expected = cv2.GaussianBlur(gray, (81, 81), 10, borderType=cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(c.blurred_preview, expected)
    assert window.canvas._base_item.pixmap().toImage() != original
    blur = c.blurred_preview
    c.c.setValue(7)
    assert c.blurred_preview is blur  # Threshold controls do not alter the blur stage.
    c.stage.setCurrentIndex(0)
    assert window.canvas._base_item.pixmap().toImage() == original
    c.stage.setCurrentIndex(1)
    c.sigma.setValue(15)
    assert c.blurred_preview is None
    assert window.canvas._base_item.pixmap().toImage() == original
    assert "Preview blur" in c.status.text()


def test_superseded_blur_preview_is_discarded(workspace, qtbot, monkeypatch):
    import threading

    _, c = workspace
    started, release = threading.Event(), threading.Event()

    def delayed_blur(*args, **kwargs):
        started.set()
        release.wait(5)
        return np.zeros((100, 180), np.uint8)

    monkeypatch.setattr("gst_image_app.weld_dilution.blurred_weld_preview", delayed_blur)
    c.preview_blur()
    try:
        qtbot.waitUntil(started.is_set, timeout=15000)
        c.sigma.setValue(20)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert c.blurred_preview is None
    assert c.stage.currentIndex() == 0


def test_local_blur_brush_undo_and_saved_recipe(workspace, qtbot, tmp_path):
    window, c = workspace
    install_envelope(window, c)
    before = c.envelope.copy()
    c.sigma.setValue(0)
    c.blur_radius.setValue(15)
    c.brush_sigma.setValue(8)
    c._tool("weld_blur_brush")
    c.brush_stroke("weld_blur_brush", [QPointF(30, 25), QPointF(100, 25)])
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert c.blurred_preview is not None
    assert len(c.draft.recipe.blur_strokes) == 1 and c.draft.segmentation_stale
    assert window.canvas.current_tool == "weld_blur_brush"
    np.testing.assert_array_equal(c.envelope, before)
    painted = c.blurred_preview.copy()
    c.c.setValue(5)
    assert len(c.draft.recipe.blur_strokes) == 1
    root = save_project(tmp_path / "brush.gstproj", window.manifest, window.layer_masks)
    loaded, _ = load_project(root)
    assert loaded.dilution_drafts[0].recipe.blur_strokes == c.draft.recipe.blur_strokes
    window.undo_stack.undo()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert c.draft.recipe.blur_strokes == []
    assert not np.array_equal(c.blurred_preview, painted)
    window.undo_stack.redo()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    np.testing.assert_array_equal(c.blurred_preview, painted)
    assert not window.manifest.particle_records


def test_native_blur_detail_retains_source_resolution(workspace, qtbot):
    window, c = workspace
    # Exceed the old 2400 px preview cap to exercise native-detail mapping.
    image = np.random.default_rng(8).integers(0, 256, (600, 2600, 3), dtype=np.uint8)
    cv2.imwrite(str(window.current_path), image)
    window._load_new_image(window.current_path)
    c.show()
    c.preview_blur()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert c.blurred_preview.shape == (600, 2600)
    c.inspect_blur_detail(QPointF(1300, 300))
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert window.canvas.transform().m11() == 1
    shown = window.canvas._base_item.pixmap().toImage()
    assert shown.width() == shown.height() == 512
    assert shown.pixelColor(256, 256).red() == c.blurred_preview[300, 1300]
    c.stage.setCurrentIndex(0)
    assert window.canvas._base_item.pixmap().toImage() != shown
    c.stage.setCurrentIndex(1)
    assert window.canvas._base_item.pixmap().toImage() == shown


def test_source_revision_change_discards_pending_result(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    window.manifest.source_revision += 1
    window._set_dirty()
    qtbot.waitUntil(lambda: not c.jobs, timeout=15000)
    assert not window.manifest.dilution_runs
    assert c.draft.segmentation_stale


def test_selected_scope_exclusions_and_deletion(workspace, qtbot):
    window, c = workspace
    window._rectangle_finished("analysis_box", QPointF(10, 10), QPointF(170, 90))
    window._rectangle_finished("exclude", QPointF(80, 40), QPointF(90, 50))
    window.rois_list.item(0).setSelected(True)
    c.scope.setCurrentIndex(1)
    c.segment()
    qtbot.waitUntil(lambda: c.worker is None and c.candidates is not None, timeout=15000)
    assert not c.domain[0, 0] and not c.domain[45, 85]
    assert c.domain[30, 30]
    assert c.draft.scope_roi_ids == [window.manifest.rois[0].id]
    ids = {c.draft.envelope_layer_id, c.draft.domain_layer_id}
    window.delete_selected_rois()
    assert not ids.intersection(window.layer_masks)
    assert c.draft.envelope_layer_id is None


def test_historical_layer_displays_its_snapshot_and_cannot_edit_it(workspace, qtbot):
    window, c = workspace
    install_envelope(window, c)
    c.calculate()
    qtbot.waitUntil(lambda: c.worker is None and c.run is not None, timeout=15000)
    first = c.run
    before = window.layer_masks[first.envelope_layer_id].copy()
    c.line_finished("weld_line", QPointF(0, 35), QPointF(179, 35))
    c.calculate()
    qtbot.waitUntil(lambda: c.worker is None and c.run.id != first.id, timeout=15000)
    latest_id = c.run.id
    window._refresh_layers()
    from PySide6.QtCore import Qt

    item = next(
        window.layers_list.item(i)
        for i in range(window.layers_list.count())
        if window.layers_list.item(i).data(Qt.ItemDataRole.UserRole) == first.envelope_layer_id
    )
    window.show_selected_layer(item)
    assert c.run.id == first.id
    assert c.display_reference.origin.y == 49.5
    assert c.points.item(0, 1).text() == "49.5"
    c.edit_envelope([QPointF(80, 25)], erase=True)
    assert np.array_equal(window.layer_masks[first.envelope_layer_id], before)
    c.leave_history()
    assert c.run.id == latest_id
    assert c.points.item(0, 1).text() == "35"


def test_invalid_surface_text_cannot_corrupt_saved_draft(workspace, tmp_path):
    window, c = workspace
    install_envelope(window, c)
    c.points.item(0, 0).setText("nan")
    assert c.draft.reference is None
    assert all(np.isfinite(p.x) for p in c.draft.surface_points)
    path = save_project(tmp_path / "invalid-input.gstproj", window.manifest, window.layer_masks)
    loaded, _ = load_project(path)
    assert loaded.dilution_drafts[0].reference is None


def test_changed_source_file_cannot_create_a_result_with_old_hash(workspace, qtbot):
    window, c = workspace
    cv2.imwrite(str(window.current_path), np.full((100, 180, 3), 20, np.uint8))
    c.segment()
    qtbot.waitUntil(lambda: c.worker is None, timeout=15000)
    assert "Source image changed" in c.status.text()
    assert c.candidates is None
    assert c.draft.envelope_layer_id is None
