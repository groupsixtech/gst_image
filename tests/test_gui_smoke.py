import cv2
import numpy as np
from gst_image_app.mainwindow import MainWindow
from gst_image_app.range_slider import MetricRangeControl
from PySide6.QtWidgets import QApplication

from gst_image.analysis.particles import measure_particles
from gst_image.models import (
    MorphologicalGradient,
    MorphologicalInput,
    ProjectManifest,
    SegmentationLayer,
    SegmentationMethod,
    SegmentationRecipe,
    ThresholdMethod,
)


def test_main_window_constructs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert "GST Image" in window.windowTitle()
    assert window.preview_button.text() == "Preview"
    assert window.run_button.text() == "Run particles — full resolution"
    assert window.window_size_slider.value() == window.window_size.value()


def test_analysis_sliders_and_spin_boxes_stay_synchronized(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.manual_threshold_high_slider.setValue(211)
    assert window.manual_threshold_high.value() == 211
    window.manual_threshold_low_slider.setValue(220)
    assert window.manual_threshold_low.value() == 220
    assert window.manual_threshold_high.value() == 220
    window.manual_threshold_high_slider.setValue(50)
    assert window.manual_threshold_high.value() == 50
    assert window.manual_threshold_low.value() == 50
    window.ball_radius.setValue(333)
    assert window.ball_radius_slider.value() == 333
    assert (window.overview_resolution.minimum(), window.overview_resolution.maximum()) == (
        0,
        100,
    )
    assert (window.roi_resolution.minimum(), window.roi_resolution.maximum()) == (0, 100)


def test_dual_range_slider_and_exact_fields_stay_synchronized(qtbot):
    control = MetricRangeControl()
    qtbot.addWidget(control)
    control.set_domain(0, 20, suffix=" px")
    control.slider.setValues(2500, 7500)
    assert control.lower.value() == 5
    assert control.upper.value() == 15
    control.lower.setValue(8)
    assert control.slider.values()[0] == 4000


def test_particle_grouping_panel_previews_filters_colors_and_saves(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    labels = np.zeros((60, 80), np.int32)
    cv2.circle(labels, (20, 30), 5, 1, cv2.FILLED)
    cv2.circle(labels, (55, 30), 10, 2, cv2.FILLED)
    particles = measure_particles(labels)
    manifest = ProjectManifest(
        name="group GUI",
        source_path="source.png",
        source_sha256="0" * 64,
        image_width=80,
        image_height=60,
    )
    layer = SegmentationLayer(name="Particles", kind="instances")
    manifest.layers.append(layer)
    manifest.particle_records[layer.id] = particles
    window.manifest = manifest
    window.current_layer_id = layer.id
    window.current_labels = labels.copy()
    window.current_mask = (labels > 0).astype(np.uint8)
    window.layer_masks[layer.id] = labels.copy()
    window.particles = particles
    window.canvas.set_image(np.zeros((60, 80, 3), np.uint8), (80, 60))
    window._refresh_all()

    panel = window.grouping_panel
    assert panel.isEnabled()
    panel.size_bins.setValue(2)
    panel.circularity_bins.setValue(1)
    panel._generate_groups()
    qtbot.wait(150)

    assert window.grouping_preview is not None
    assert len(panel.current_grouping().groups) == 2
    assert len({group.color for group in panel.current_grouping().groups}) == 2
    panel.copy_group_table()
    copied_groups = QApplication.clipboard().text().splitlines()
    assert copied_groups[0].startswith("On\tColor\tName")
    assert len(copied_groups) == 3
    assert all(line.startswith("Yes\t#") for line in copied_groups[1:])
    panel.copy_statistics_table()
    copied_statistics = QApplication.clipboard().text().splitlines()
    assert copied_statistics[0].startswith("Group\tCount\tCount %")
    assert len(copied_statistics) == 5
    overlay = window.canvas._overlay_item.pixmap().toImage()
    left_color = overlay.pixelColor(20, 30)
    right_color = overlay.pixelColor(55, 30)
    assert left_color != right_color
    panel.filter_box.setChecked(True)
    panel.filter_size.setChecked(True)
    maximum = max(particle.equivalent_radius_px for particle in particles)
    panel.filter_size_range.set_values(maximum - 0.1, maximum)
    panel._filter_changed()
    qtbot.wait(150)
    assert len(window.grouping_preview.included_labels) == 1
    filtered_overlay = window.canvas._overlay_item.pixmap().toImage()
    assert filtered_overlay.pixelColor(20, 30).alpha() == 0
    assert filtered_overlay.pixelColor(55, 30).alpha() > 0

    grouping = panel.current_grouping()
    window._save_particle_grouping(grouping)
    assert manifest.active_particle_groupings[layer.id] == grouping.id
    assert manifest.particle_groupings[0].name == grouping.name
    assert np.array_equal(window.layer_masks[layer.id], labels)
    window._set_dirty(False)


def test_color_plots_include_volume_fraction_pie(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    window = MainWindow()
    qtbot.addWidget(window)
    labels = np.zeros((40, 60), np.int32)
    cv2.circle(labels, (20, 20), 5, 1, cv2.FILLED)
    window.manifest = ProjectManifest(
        name="plot GUI",
        source_path="source.png",
        source_sha256="0" * 64,
        image_width=60,
        image_height=40,
    )
    window.particles = measure_particles(labels)
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: None)

    figure = window.plot_particles()

    assert len(figure.axes) == 3
    assert figure.axes[2].get_title() == "Particle area share"


def test_threshold_method_shows_only_applicable_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    def select(method):
        window.method.setCurrentIndex(window.method.findData(method))

    select(ThresholdMethod.SAUVOLA)
    assert not window.sauvola_window_control.isHidden()
    assert not window.sauvola_k_control.isHidden()
    assert window.gaussian_block_control.isHidden()
    assert window.gaussian_c_control.isHidden()
    assert window.manual_threshold_low_control.isHidden()
    assert window.manual_threshold_high_control.isHidden()

    select(ThresholdMethod.ADAPTIVE_GAUSSIAN)
    assert window.sauvola_window_control.isHidden()
    assert not window.gaussian_block_control.isHidden()
    assert not window.gaussian_c_control.isHidden()
    assert window.manual_threshold_low_control.isHidden()
    assert window.manual_threshold_high_control.isHidden()

    select(ThresholdMethod.OTSU)
    assert window.sauvola_window_control.isHidden()
    assert window.gaussian_block_control.isHidden()
    assert window.manual_threshold_low_control.isHidden()
    assert window.manual_threshold_high_control.isHidden()

    select(ThresholdMethod.MANUAL)
    assert not window.manual_threshold_low_control.isHidden()
    assert not window.manual_threshold_high_control.isHidden()
    assert not window.gaussian_blur_control.isHidden()

    window.segmentation_method.setCurrentIndex(
        window.segmentation_method.findData(SegmentationMethod.MORPHOLOGICAL_WATERSHED)
    )
    assert not window.morphological_input.isHidden()
    assert not window.morphological_gradient.isHidden()
    assert not window.morphological_gradient_control.isHidden()
    assert not window.morphological_tolerance_control.isHidden()
    assert window.split_touching.isHidden()
    window.morphological_input.setCurrentIndex(
        window.morphological_input.findData(MorphologicalInput.BORDER)
    )
    assert window.morphological_gradient.isHidden()
    assert window.morphological_gradient_control.isHidden()


def test_all_exposed_parameters_are_copied_to_recipe(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.channel.setCurrentIndex(window.channel.findData("red"))
    window.method.setCurrentIndex(window.method.findData(ThresholdMethod.ADAPTIVE_GAUSSIAN))
    window.gaussian_block.setValue(55)
    window.gaussian_c.setValue(4.5)
    window.manual_threshold_low.setValue(20)
    window.manual_threshold_high.setValue(200)
    window.gaussian_blur_sigma.setValue(1.25)
    window.open_radius.setValue(3)
    window.close_radius.setValue(4)
    window.watershed_distance.setValue(11)
    window.segmentation_method.setCurrentIndex(
        window.segmentation_method.findData(SegmentationMethod.MORPHOLOGICAL_WATERSHED)
    )
    window.morphological_gradient.setCurrentIndex(
        window.morphological_gradient.findData(MorphologicalGradient.INTERNAL)
    )
    window.morphological_tolerance.setValue(17)
    window.fill_holes.setChecked(True)

    recipe = window._recipe_from_controls()
    assert recipe.channel == "red"
    assert recipe.gaussian_block_px == 55
    assert recipe.gaussian_c == 4.5
    assert recipe.manual_threshold_low == 20
    assert recipe.manual_threshold_high == 200
    assert recipe.gaussian_blur_sigma == 1.25
    assert recipe.open_radius_px == 3
    assert recipe.close_radius_px == 4
    assert recipe.watershed_min_distance_px == 11
    assert recipe.segmentation_method == SegmentationMethod.MORPHOLOGICAL_WATERSHED
    assert recipe.morphological_gradient == MorphologicalGradient.INTERNAL
    assert recipe.morphological_tolerance == 17
    assert recipe.fill_holes


def test_section_defaults_reset_only_their_parameters(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.channel.setCurrentIndex(window.channel.findData("red"))
    window.segmentation_method.setCurrentIndex(
        window.segmentation_method.findData(SegmentationMethod.MORPHOLOGICAL_WATERSHED)
    )
    window.method.setCurrentIndex(window.method.findData(ThresholdMethod.MANUAL))
    window.polarity.setCurrentIndex(1)
    window.manual_threshold_high.setValue(231)
    window.manual_threshold_low.setValue(50)
    window.window_size.setValue(333)
    window.morphological_input.setCurrentIndex(
        window.morphological_input.findData(MorphologicalInput.BORDER)
    )
    window.morphological_gradient_radius.setValue(8)
    window.morphological_tolerance.setValue(44)
    window.morphological_connectivity.setCurrentIndex(
        window.morphological_connectivity.findData(4)
    )
    window.morphological_calculate_dams.setChecked(False)
    window.gaussian_blur_sigma.setValue(4.2)
    window.illumination.setChecked(False)
    window.ball_radius.setValue(999)
    window.open_radius.setValue(9)
    window.close_radius.setValue(10)
    window.fill_holes.setChecked(True)
    window.min_area.setValue(800)
    window.max_area.setValue(900)
    window.split_touching.setChecked(False)
    window.watershed_distance.setValue(42)
    window.tile_size.setValue(4096)
    window.brush_radius.setValue(33)
    window.overview_resolution.setValue(80)
    window.roi_resolution.setValue(40)
    window.preview_resolution.setCurrentIndex(1)
    window.selected_roi_only.setChecked(True)
    header_state = (
        window.binary_class.currentData(),
        window.channel.currentData(),
        window.segmentation_method.currentData(),
        window.method.currentData(),
        window.polarity.currentData(),
        window.overview_resolution.value(),
        window.roi_resolution.value(),
        window.preview_resolution.currentData(),
        window.selected_roi_only.isChecked(),
    )

    window.restore_threshold_defaults()
    assert window.manual_threshold_low.value() == 0
    assert window.manual_threshold_high.value() == 127
    assert window.window_size.value() == 333
    assert window.open_radius.value() == 9
    window.restore_morphological_defaults()
    window.restore_preblur_defaults()
    window.restore_rolling_ball_defaults()
    window.restore_open_close_defaults()
    window.restore_flood_fill_defaults()
    window.restore_particle_filter_defaults()
    window.restore_watershed_defaults()
    window.restore_tile_defaults()
    window.restore_brush_defaults()

    defaults = SegmentationRecipe()
    assert window.morphological_input.currentData() == defaults.morphological_input
    assert (
        window.morphological_gradient_radius.value()
        == defaults.morphological_gradient_radius_px
    )
    assert window.morphological_tolerance.value() == defaults.morphological_tolerance
    assert (
        window.morphological_connectivity.currentData()
        == defaults.morphological_connectivity
    )
    assert window.morphological_calculate_dams.isChecked()
    assert window.gaussian_blur_sigma.value() == defaults.gaussian_blur_sigma
    assert window.illumination.isChecked()
    assert window.ball_radius.value() == defaults.rolling_ball_radius_px
    assert window.open_radius.value() == defaults.open_radius_px
    assert window.close_radius.value() == defaults.close_radius_px
    assert window.fill_holes.isChecked() == defaults.fill_holes
    assert window.min_area.value() == defaults.min_particle_area_px
    assert window.max_area.value() == 0
    assert window.split_touching.isChecked() == defaults.split_touching
    assert window.watershed_distance.value() == defaults.watershed_min_distance_px
    assert window.tile_size.value() == defaults.tile_size_px
    assert window.brush_radius.value() == 12
    assert not hasattr(window, "restore_defaults_button")
    assert header_state == (
        window.binary_class.currentData(),
        window.channel.currentData(),
        window.segmentation_method.currentData(),
        window.method.currentData(),
        window.polarity.currentData(),
        window.overview_resolution.value(),
        window.roi_resolution.value(),
        window.preview_resolution.currentData(),
        window.selected_roi_only.isChecked(),
    )
