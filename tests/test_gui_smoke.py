from gst_image_app.mainwindow import MainWindow

from gst_image.models import (
    MorphologicalGradient,
    MorphologicalInput,
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


def test_restore_analysis_defaults_resets_recipe_and_resolution_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.method.setCurrentIndex(window.method.findData(ThresholdMethod.MANUAL))
    window.manual_threshold_high.setValue(231)
    window.manual_threshold_low.setValue(50)
    window.open_radius.setValue(9)
    window.fill_holes.setChecked(True)
    window.overview_resolution.setValue(80)
    window.roi_resolution.setValue(40)

    window.restore_analysis_defaults()
    recipe = window._recipe_from_controls()
    ignored = {"id", "name", "target_class_id"}
    assert recipe.model_dump(exclude=ignored) == SegmentationRecipe().model_dump(
        exclude=ignored
    )
    assert window.overview_resolution.value() == 25
    assert window.roi_resolution.value() == 100
