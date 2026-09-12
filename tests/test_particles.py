
import cv2
import numpy as np
import pytest

from gst_image.analysis.groups import (
    assign_particle_groups,
    evaluate_particle_grouping,
    generate_particle_groups,
    particle_group_statistics,
    validate_particle_groups,
)
from gst_image.analysis.particles import (
    measure_particles,
    merge_particle_labels,
    segment_particles,
    split_particle_by_line,
)
from gst_image.analysis.preprocess import threshold_array
from gst_image.models import (
    Calibration,
    MorphologicalGradient,
    MorphologicalInput,
    ParticleCriteria,
    ParticleGroup,
    ParticleGrouping,
    ParticlePolarity,
    ParticleSizeMetric,
    SegmentationMethod,
    SegmentationRecipe,
    ThresholdMethod,
)


def test_manual_segmentation_and_radius_measurement():
    image = np.full((180, 240), 210, np.uint8)
    gold = np.zeros_like(image)
    cv2.circle(image, (70, 90), 20, 40, cv2.FILLED)
    cv2.circle(image, (170, 90), 30, 40, cv2.FILLED)
    cv2.circle(gold, (70, 90), 20, 1, cv2.FILLED)
    cv2.circle(gold, (170, 90), 30, 1, cv2.FILLED)
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=99,
        illumination_correction=False,
        min_particle_area_px=100,
        split_touching=False,
        open_radius_px=0,
        close_radius_px=0,
        gaussian_blur_sigma=0,
        tile_size_px=256,
    )
    result = segment_particles(image, np.ones_like(image, bool), recipe)
    assert result.summary["particle_count"] == 2
    assert abs(result.summary["area_fraction"] - np.mean(gold)) < 0.001
    radii = sorted(item.equivalent_radius_px for item in result.particles)
    assert radii[0] == pytest.approx(20, abs=0.6)
    assert radii[1] == pytest.approx(30, abs=0.6)


def test_local_threshold_handles_gradient_and_stitch_step():
    height, width = 240, 480
    background = np.tile(np.linspace(145, 220, width, dtype=np.float32), (height, 1))
    background[:, width // 2 :] -= 18
    image = background.astype(np.uint8)
    gold = np.zeros_like(image)
    for center in ((80, 70), (200, 160), (320, 70), (410, 165)):
        cv2.circle(gold, center, 14, 1, cv2.FILLED)
        cv2.circle(image, center, 14, max(10, int(image[center[1], center[0]]) - 85), cv2.FILLED)
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.SAUVOLA,
        sauvola_window_px=61,
        sauvola_k=0.12,
        rolling_ball_radius_px=75,
        min_particle_area_px=200,
        max_particle_area_px=1000,
        split_touching=False,
        open_radius_px=1,
        close_radius_px=1,
        tile_size_px=256,
    )
    result = segment_particles(image, np.ones_like(image, bool), recipe)
    intersection = np.count_nonzero(result.mask & (gold > 0))
    union = np.count_nonzero(result.mask | (gold > 0))
    assert intersection / union >= 0.80
    assert result.summary["particle_count"] == 4


@pytest.mark.parametrize("method", list(ThresholdMethod))
def test_every_threshold_method_produces_a_binary_selection(method):
    image = np.tile(np.linspace(20, 230, 31, dtype=np.uint8), (31, 1))
    recipe = SegmentationRecipe(
        threshold_method=method,
        manual_threshold_low=0,
        manual_threshold_high=124,
        sauvola_window_px=9,
        gaussian_block_px=9,
        gaussian_c=2,
        gaussian_blur_sigma=0,
    )
    selected = threshold_array(image, recipe, global_otsu_threshold=125)
    assert selected.dtype == bool
    assert selected.shape == image.shape
    assert np.any(selected)
    assert np.any(~selected)


@pytest.mark.parametrize("polarity", list(ParticlePolarity))
def test_manual_threshold_selects_an_inclusive_band_independent_of_polarity(polarity):
    image = np.array([[9, 10, 15, 20, 21]], dtype=np.uint8)
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=10,
        manual_threshold_high=20,
        polarity=polarity,
        gaussian_blur_sigma=0,
    )
    assert threshold_array(image, recipe).tolist() == [
        [False, True, True, True, False]
    ]


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1, 20), (10, 256), (21, 20)],
)
def test_manual_threshold_rejects_invalid_ranges(low, high):
    with pytest.raises(ValueError, match="manual_threshold|Manual threshold"):
        SegmentationRecipe(manual_threshold_low=low, manual_threshold_high=high)


def test_flood_fill_option_fills_enclosed_particle_holes():
    image = np.full((100, 100), 220, np.uint8)
    cv2.circle(image, (50, 50), 25, 30, 8)
    common = {
        "threshold_method": ThresholdMethod.MANUAL,
        "manual_threshold_low": 0,
        "manual_threshold_high": 99,
        "illumination_correction": False,
        "gaussian_blur_sigma": 0,
        "open_radius_px": 0,
        "close_radius_px": 0,
        "split_touching": False,
        "tile_size_px": 256,
    }
    unfilled = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(fill_holes=False, **common),
    )
    filled = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(fill_holes=True, **common),
    )
    assert not unfilled.mask[50, 50]
    assert filled.mask[50, 50]
    assert np.count_nonzero(filled.mask) > np.count_nonzero(unfilled.mask)


def test_morphological_closing_joins_narrow_threshold_gaps():
    image = np.full((100, 100), 220, np.uint8)
    cv2.rectangle(image, (15, 35), (44, 65), 30, cv2.FILLED)
    cv2.rectangle(image, (47, 35), (76, 65), 30, cv2.FILLED)
    common = {
        "threshold_method": ThresholdMethod.MANUAL,
        "manual_threshold_low": 0,
        "manual_threshold_high": 99,
        "illumination_correction": False,
        "gaussian_blur_sigma": 0,
        "open_radius_px": 0,
        "split_touching": False,
        "tile_size_px": 256,
    }
    separate = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(close_radius_px=0, **common),
    )
    closed = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(close_radius_px=2, **common),
    )
    assert separate.summary["particle_count"] == 2
    assert closed.summary["particle_count"] == 1


@pytest.mark.parametrize("gradient", list(MorphologicalGradient))
def test_morphological_watershed_segments_threshold_foreground(gradient):
    image = np.full((120, 180), 220, np.uint8)
    cv2.circle(image, (55, 60), 24, 35, cv2.FILLED)
    cv2.circle(image, (125, 60), 20, 65, cv2.FILLED)
    recipe = SegmentationRecipe(
        segmentation_method=SegmentationMethod.MORPHOLOGICAL_WATERSHED,
        morphological_input=MorphologicalInput.OBJECT,
        morphological_gradient=gradient,
        morphological_gradient_radius_px=2,
        morphological_tolerance=5,
        morphological_connectivity=4,
        morphological_calculate_dams=True,
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=99,
        illumination_correction=False,
        gaussian_blur_sigma=0,
        open_radius_px=0,
        close_radius_px=0,
        min_particle_area_px=100,
        tile_size_px=256,
    )
    result = segment_particles(image, np.ones_like(image, bool), recipe)
    assert result.summary["particle_count"] == 2
    assert set(np.unique(result.labels)) == {0, 1, 2}


def test_boundary_particles_contribute_area_but_are_flagged():
    labels = np.zeros((50, 50), np.int32)
    cv2.circle(labels, (0, 25), 10, 1, cv2.FILLED)
    cv2.circle(labels, (30, 25), 7, 2, cv2.FILLED)
    records = measure_particles(labels, Calibration(mm_per_pixel=0.01), np.ones_like(labels, bool))
    assert records[0].border_touching
    assert not records[1].border_touching
    assert records[1].area_mm2 == pytest.approx(records[1].area_px * 0.0001)


def test_ordered_group_assignment():
    labels = np.zeros((80, 80), np.int32)
    cv2.circle(labels, (20, 40), 5, 1, cv2.FILLED)
    cv2.circle(labels, (55, 40), 10, 2, cv2.FILLED)
    records = measure_particles(labels, Calibration(mm_per_pixel=0.1))
    groups = [
        ParticleGroup(name="small", radius_min=0, radius_max=0.75),
        ParticleGroup(name="large", radius_min=0.75, radius_max=2),
    ]
    assigned = assign_particle_groups(records, groups)
    assert [item.group for item in assigned] == ["small", "large"]

    at_boundary = records[0].model_copy(
        update={"equivalent_radius_mm": 0.75, "circularity": 0.5}
    )
    at_final_upper = records[0].model_copy(
        update={"equivalent_radius_mm": 2.0, "circularity": 1.0}
    )
    assigned = assign_particle_groups([at_boundary, at_final_upper], groups)
    assert [item.group for item in assigned] == ["large", "large"]


def test_saved_grouping_filters_then_partitions_by_size_and_circularity():
    labels = np.zeros((80, 80), np.int32)
    cv2.circle(labels, (20, 40), 5, 1, cv2.FILLED)
    cv2.circle(labels, (55, 40), 10, 2, cv2.FILLED)
    records = measure_particles(labels, Calibration(mm_per_pixel=0.1))
    records[0] = records[0].model_copy(update={"circularity": 0.4})
    records[1] = records[1].model_copy(update={"circularity": 0.9})
    groups = generate_particle_groups(
        [0, 0.75, 2],
        [0, 0.5, 1],
        size_metric=ParticleSizeMetric.EQUIVALENT_RADIUS,
        size_unit="mm",
    )
    grouping = ParticleGrouping(
        name="Filtered grid",
        source_layer_id="particles",
        filter_criteria=ParticleCriteria(
            size_unit="mm", size_min=0.4, size_max=2, circularity_min=0.3
        ),
        groups=groups,
    )

    result = evaluate_particle_grouping(records, grouping)

    assert result.filtered_labels == set()
    assert result.unclassified_labels == set()
    assert [particle.group for particle in result.particles] == [
        "Size 1 / Circularity 1",
        "Size 2 / Circularity 2",
    ]
    filtered = grouping.model_copy(
        update={
            "filter_criteria": ParticleCriteria(
                size_unit="mm", size_min=0.4, circularity_min=0.5
            )
        }
    )
    filtered_result = evaluate_particle_grouping(records, filtered)
    assert filtered_result.filtered_labels == {1}
    assert filtered_result.included_labels == {2}
    assert records[0].group == "Unclassified"


def test_group_validation_rejects_overlap_but_allows_adjacent_ranges():
    adjacent = [
        ParticleGroup(name="Small", size_unit="px", size_min=0, size_max=5),
        ParticleGroup(name="Large", size_unit="px", size_min=5, size_max=10),
    ]
    validate_particle_groups(adjacent)
    with pytest.raises(ValueError, match="overlap"):
        validate_particle_groups(
            adjacent
            + [ParticleGroup(name="Overlap", size_unit="px", size_min=4, size_max=6)]
        )


def test_group_statistics_include_distribution_and_analysis_denominator():
    labels = np.zeros((50, 50), np.int32)
    cv2.circle(labels, (15, 25), 4, 1, cv2.FILLED)
    cv2.circle(labels, (35, 25), 7, 2, cv2.FILLED)
    records = measure_particles(labels)
    grouping = ParticleGrouping(
        groups=[
            ParticleGroup(
                name="All", size_unit="px", size_min=0, size_max=20,
                circularity_min=0, circularity_max=1,
            )
        ]
    )
    result = evaluate_particle_grouping(records, grouping)
    statistics = particle_group_statistics(
        records, grouping, result, analyzed_pixels=2500
    )

    assert statistics["All"]["count"] == 2
    assert statistics["All"]["count_percent"] == 100
    assert statistics["All"]["area_fraction"] == pytest.approx(
        sum(particle.area_px for particle in records) / 2500
    )
    assert statistics["All"]["size"]["median"] is not None
    assert statistics["All"]["circularity"]["mean"] is not None


def test_physical_morphology_requires_calibration():
    image = np.full((30, 30), 200, np.uint8)
    recipe = SegmentationRecipe(
        illumination_correction=False,
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=99,
        open_radius_mm=0.01,
        split_touching=False,
    )
    with pytest.raises(ValueError, match="calibration"):
        segment_particles(image, np.ones_like(image, bool), recipe)


def test_watershed_split_and_manual_label_edits_preserve_instances():
    image = np.full((160, 200), 210, np.uint8)
    cv2.circle(image, (85, 80), 30, 35, cv2.FILLED)
    cv2.circle(image, (115, 80), 30, 35, cv2.FILLED)
    recipe = SegmentationRecipe(
        illumination_correction=False,
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=99,
        min_particle_area_px=200,
        split_touching=True,
        watershed_min_distance_px=15,
        open_radius_px=0,
        close_radius_px=0,
        tile_size_px=256,
    )
    result = segment_particles(image, np.ones_like(image, bool), recipe)
    assert result.summary["particle_count"] == 2

    merged = merge_particle_labels(result.labels, [1, 2])
    assert int(merged.max()) == 1
    split = split_particle_by_line(merged, 1, (100, 40), (100, 120), width=2)
    assert int(split.max()) == 2


def test_tiled_processing_preserves_particle_crossing_tile_boundary():
    image = np.full((300, 520), 210, np.uint8)
    cv2.circle(image, (256, 150), 35, 30, cv2.FILLED)
    common = {
        "illumination_correction": False,
        "threshold_method": ThresholdMethod.MANUAL,
        "manual_threshold_low": 0,
        "manual_threshold_high": 99,
        "min_particle_area_px": 100,
        "split_touching": False,
        "open_radius_px": 0,
        "close_radius_px": 0,
    }
    tiled = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(tile_size_px=256, **common),
    )
    untiled = segment_particles(
        image,
        np.ones_like(image, bool),
        SegmentationRecipe(tile_size_px=1024, **common),
    )
    assert tiled.summary["particle_count"] == 1
    assert np.array_equal(tiled.mask, untiled.mask)
    assert tiled.summary["area_fraction"] == pytest.approx(
        untiled.summary["area_fraction"], abs=0.001
    )


def test_particle_segmentation_can_be_cancelled():
    image = np.full((30, 30), 200, np.uint8)
    with pytest.raises(InterruptedError, match="cancelled"):
        segment_particles(
            image,
            np.ones_like(image, bool),
            SegmentationRecipe(illumination_correction=False),
            cancelled=lambda: True,
        )
