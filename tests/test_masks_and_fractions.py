import numpy as np
import pytest

from gst_image.analysis.fractions import compute_area_fractions, compute_project_fractions
from gst_image.analysis.masks import build_analysis_mask, rasterize_roi, suggest_specimen_mask
from gst_image.models import (
    ROI,
    Point,
    ProjectManifest,
    ROIKind,
    SegmentationLayer,
)


def rectangle(name, kind, first, last):
    return ROI(
        name=name,
        kind=kind,
        shape="rectangle",
        points=[Point(x=first[0], y=first[1]), Point(x=last[0], y=last[1])],
    )


def test_mask_precedence():
    include = rectangle("include", ROIKind.INCLUDE, (2, 2), (8, 8))
    exclude = rectangle("exclude", ROIKind.EXCLUDE, (5, 5), (8, 8))
    mask = build_analysis_mask((10, 10), [include, exclude])
    assert mask[3, 3]
    assert not mask[6, 6]
    assert not mask[0, 0]


def test_empty_roi_selection_uses_full_specimen_minus_exclusions():
    include = rectangle("include", ROIKind.INCLUDE, (2, 2), (4, 4))
    exclude = rectangle("exclude", ROIKind.EXCLUDE, (7, 7), (9, 9))
    mask = build_analysis_mask((10, 10), [include, exclude], include_ids=set())
    assert mask[0, 0]
    assert mask[3, 3]
    assert not mask[8, 8]


def test_polygon_rasterization():
    roi = ROI(
        name="triangle",
        points=[Point(x=1, y=1), Point(x=8, y=1), Point(x=1, y=8)],
    )
    mask = rasterize_roi((10, 10), roi)
    assert mask[2, 2]
    assert not mask[9, 9]


def test_area_fraction_uses_domain_denominator():
    domain = np.zeros((10, 10), bool)
    domain[:, :5] = True
    phase = np.zeros_like(domain)
    phase[:5, :5] = True
    result = compute_area_fractions({"phase": phase}, domain)
    assert result.entries[0].area_fraction == pytest.approx(0.5)
    assert result.entries[0].estimated_volume_fraction_percent == pytest.approx(50)


def test_specimen_suggestion_removes_black_canvas_and_small_scale_bar():
    image = np.zeros((100, 200), np.uint8)
    image[20:80, 20:180] = 150
    image[90:93, 150:190] = 255
    mask = suggest_specimen_mask(image)
    assert mask[50, 100]
    assert not mask[0, 0]
    assert not mask[91, 170]


def test_project_fractions_expand_multiclass_layer(tmp_path):
    source = tmp_path / "placeholder.png"
    source.write_bytes(b"placeholder")
    manifest = ProjectManifest(
        name="fractions",
        source_path=str(source),
        source_sha256="0" * 64,
        image_width=4,
        image_height=2,
    )
    weld, haz = manifest.classes[:2]
    regions = SegmentationLayer(
        name="Zones", kind="multiclass", class_value_map={1: weld.id, 2: haz.id}
    )
    domain = SegmentationLayer(name="domain", kind="domain", visible=False)
    manifest.layers.extend([regions, domain])
    labels = np.array([[1, 1, 2, 2], [1, 2, 2, 0]], dtype=np.uint8)
    result = compute_project_fractions(
        manifest, {regions.id: labels, domain.id: np.ones_like(labels)}
    )
    fractions = {entry.layer: entry.area_fraction for entry in result.entries}
    assert fractions["Zones: Weld"] == pytest.approx(3 / 8)
    assert fractions["Zones: HAZ"] == pytest.approx(4 / 8)
