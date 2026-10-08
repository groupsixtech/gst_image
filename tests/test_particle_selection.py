import numpy as np
import pytest
from gst_image_app.mainwindow import MainWindow
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget

from gst_image.analysis.particles import measure_particles
from gst_image.models import ProjectManifest, SegmentationLayer


@pytest.fixture
def particle_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    labels = np.zeros((60, 80), np.int32)
    labels[10:20, 10:20] = 3
    labels[30:50, 50:70] = 17
    layer = SegmentationLayer(name="Particles", kind="instances")
    window.manifest = ProjectManifest(
        name="selection",
        source_path="source.png",
        source_sha256="0" * 64,
        image_width=80,
        image_height=60,
        layers=[layer],
        particle_records={layer.id: measure_particles(labels)},
    )
    window.current_layer_id = layer.id
    window.current_labels = labels
    window.layer_masks[layer.id] = labels
    window.particles = window.manifest.particle_records[layer.id]
    window.canvas.set_image(np.zeros((60, 80, 3), np.uint8), (80, 60))
    window._refresh_all()
    window._render_visible_layers()
    return window


@pytest.mark.parametrize("distinct_colors", [False, True])
def test_mouse_and_keyboard_selection_highlight_only_selected_particle(
    qtbot, particle_window, distinct_colors
):
    window = particle_window
    window.distinct_particle_colors_action.setChecked(distinct_colors)
    original_labels = window.current_labels.copy()
    original_manifest = window.manifest.model_dump()
    original_overlay = window.canvas._overlay_item.pixmap().toImage()
    table = window.particle_table
    tabs = table.parentWidget().parentWidget().parentWidget()
    assert isinstance(tabs, QTabWidget)
    tabs.setCurrentWidget(table.parentWidget())
    window.show()
    qtbot.mouseClick(
        table.viewport(), Qt.MouseButton.LeftButton,
        pos=table.visualItemRect(table.item(0, 2)).center(),
    )
    highlight = window.canvas._particle_highlight_item
    image = highlight.pixmap().toImage()
    assert image.pixelColor(15, 15).getRgb() == (255, 0, 255, 210)
    assert image.pixelColor(10, 10).getRgb() == (255, 255, 255, 255)
    assert image.pixelColor(60, 40).alpha() == 0
    assert highlight.zValue() > window.canvas._overlay_item.zValue()

    qtbot.keyClick(table, Qt.Key.Key_Down)
    image = window.canvas._particle_highlight_item.pixmap().toImage()
    assert image.pixelColor(15, 15).alpha() == 0
    assert image.pixelColor(60, 40).getRgb() == (255, 0, 255, 210)
    assert window.canvas._overlay_item.pixmap().toImage() == original_overlay
    np.testing.assert_array_equal(window.current_labels, original_labels)
    assert window.manifest.model_dump() == original_manifest
    table.clearSelection()
    assert window.canvas._particle_highlight_item is None


def test_highlight_tracks_label_through_refresh_and_filtering(particle_window):
    window = particle_window
    table = window.particle_table
    table.selectRow(1)
    window.grouping_preview.particles.reverse()
    window._refresh_particles()
    assert table.selectionModel().selectedRows()[0].row() == 0
    assert table.item(0, 0).text() == "17"
    assert window.canvas._particle_highlight_item is not None

    window.grouping_preview.filtered_labels.add(17)
    window.grouping_preview_definition.show_filtered = False
    window._refresh_particles()
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "3"
    assert not table.selectionModel().selectedRows()
    assert window.canvas._particle_highlight_item is None


def test_switching_layers_clears_selection_even_with_same_labels(particle_window):
    window = particle_window
    window.particle_table.selectRow(0)
    other_layer = SegmentationLayer(name="Other particles", kind="instances")
    window.manifest.layers.append(other_layer)
    window.manifest.particle_records[other_layer.id] = window.particles
    window.layer_masks[other_layer.id] = window.current_labels.copy()
    window._refresh_layers()
    window.show_selected_layer(window.layers_list.item(1))
    assert not window.particle_table.selectionModel().selectedRows()
    assert window.canvas._particle_highlight_item is None


def test_highlight_aligns_with_scaled_roi_and_survives_overlay_changes(particle_window):
    window = particle_window
    window.particle_table.selectRow(1)
    window._set_canvas_image(np.zeros((20, 20, 3), np.uint8), (40, 20, 80, 60))
    highlight = window.canvas._particle_highlight_item
    assert highlight.pos() == window.canvas._base_item.pos()
    assert highlight.transform() == window.canvas._base_item.transform()
    assert highlight.pixmap().toImage().pixelColor(10, 10).getRgb() == (255, 0, 255, 210)
    assert highlight.pixmap().toImage().pixelColor(2, 2).alpha() == 0

    window.manifest.layers[0].visible = False
    window._render_visible_layers()
    assert window.canvas._overlay_item is None
    assert window.canvas._particle_highlight_item is not None
    window._set_canvas_image(np.zeros((60, 80, 3), np.uint8))
    image = window.canvas._particle_highlight_item.pixmap().toImage()
    assert image.pixelColor(60, 40).getRgb() == (255, 0, 255, 210)
