import math

import pytest

from gst_image.analysis.calibration import create_measurement
from gst_image.models import (
    Calibration,
    ParticleCriteria,
    ParticleGroup,
    ParticleRecord,
    ParticleSizeMetric,
    RegionClassifierRecipe,
    SegmentationRecipe,
)


def particle(radius_px=5.0, radius_mm=0.5, circularity=0.8):
    return ParticleRecord(
        label=1,
        area_px=math.pi * radius_px**2,
        area_mm2=math.pi * radius_mm**2,
        equivalent_radius_px=radius_px,
        equivalent_radius_mm=radius_mm,
        equivalent_diameter_px=2 * radius_px,
        perimeter_crofton_px=2 * math.pi * radius_px,
        circularity=circularity,
        solidity=1,
        eccentricity=0,
        major_axis_px=2 * radius_px,
        minor_axis_px=2 * radius_px,
        feret_diameter_max_px=2 * radius_px,
        centroid_x_px=10,
        centroid_y_px=10,
    )


def test_calibration_and_line_measurement():
    calibration = Calibration.from_reference((0, 0), (200, 0), 2.0, "mm")
    assert calibration.mm_per_pixel == pytest.approx(0.01)
    measurement = create_measurement("width", (10, 10), (10, 60), calibration)
    assert measurement.length_px == pytest.approx(50)
    assert measurement.length_mm == pytest.approx(0.5)


def test_micrometre_calibration_is_canonical_mm():
    calibration = Calibration.from_reference((0, 0), (100, 0), 500, "µm")
    assert calibration.mm_per_pixel == pytest.approx(0.005)


def test_group_ranges_are_half_open_except_final_group():
    value = particle(radius_mm=0.5)
    first = ParticleGroup(name="small", radius_min=0, radius_max=0.5)
    last = ParticleGroup(name="large", radius_min=0.5, radius_max=1)
    assert not first.matches(value)
    assert last.matches(value, final=True)
    endpoint = particle(radius_mm=1)
    assert last.matches(endpoint, final=True)


def test_particle_size_criteria_support_radius_diameter_and_area_units():
    value = particle(radius_px=5, radius_mm=0.5)
    assert ParticleCriteria(size_unit="px").size_value(value) == 5
    assert ParticleCriteria(
        size_metric=ParticleSizeMetric.EQUIVALENT_DIAMETER, size_unit="px"
    ).size_value(value) == 10
    assert ParticleCriteria(
        size_metric=ParticleSizeMetric.AREA, size_unit="mm"
    ).size_value(value) == pytest.approx(math.pi * 0.5**2)


def test_recipe_normalizes_even_windows():
    recipe = SegmentationRecipe(sauvola_window_px=100, gaussian_block_px=20)
    assert recipe.sauvola_window_px == 101
    assert recipe.gaussian_block_px == 21


def test_invalid_group_and_feature_ranges_are_rejected():
    with pytest.raises(ValueError, match="Maximum radius"):
        ParticleGroup(name="invalid", radius_min=2, radius_max=1)
    with pytest.raises(ValueError, match="Maximum feature sigma"):
        RegionClassifierRecipe(sigma_min=4, sigma_max=2)
