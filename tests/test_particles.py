
import cv2
import numpy as np
import pytest

from gst_image.analysis.groups import assign_particle_groups
from gst_image.analysis.particles import (
    measure_particles,
    merge_particle_labels,
    segment_particles,
    split_particle_by_line,
)
from gst_image.models import Calibration, ParticleGroup, SegmentationRecipe, ThresholdMethod


def test_manual_segmentation_and_radius_measurement():
    image = np.full((180, 240), 210, np.uint8)
    gold = np.zeros_like(image)
    cv2.circle(image, (70, 90), 20, 40, cv2.FILLED)
    cv2.circle(image, (170, 90), 30, 40, cv2.FILLED)
    cv2.circle(gold, (70, 90), 20, 1, cv2.FILLED)
    cv2.circle(gold, (170, 90), 30, 1, cv2.FILLED)
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold=100,
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


def test_physical_morphology_requires_calibration():
    image = np.full((30, 30), 200, np.uint8)
    recipe = SegmentationRecipe(
        illumination_correction=False,
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold=100,
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
        manual_threshold=100,
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
        "manual_threshold": 100,
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
