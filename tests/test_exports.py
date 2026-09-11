import cv2
import numpy as np

from gst_image.export import create_overlay, export_analysis
from gst_image.models import ProjectManifest, SegmentationLayer


def test_multiclass_overlay_and_fraction_export(tmp_path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full((2, 4), 100, np.uint8))
    manifest = ProjectManifest(
        name="export",
        source_path=str(source),
        source_sha256="0" * 64,
        image_width=4,
        image_height=2,
    )
    weld, haz = manifest.classes[:2]
    zones = SegmentationLayer(
        name="Zones",
        kind="multiclass",
        opacity=1,
        class_value_map={1: weld.id, 2: haz.id},
    )
    domain = SegmentationLayer(name="Analysis domain", kind="domain", visible=False)
    manifest.layers.extend([zones, domain])
    labels = np.array([[1, 1, 2, 2], [1, 2, 2, 0]], dtype=np.uint8)
    masks = {zones.id: labels, domain.id: np.ones_like(labels)}

    overlay = create_overlay(np.full((2, 4, 3), 100, np.uint8), manifest, masks)
    assert tuple(overlay[0, 0]) != tuple(overlay[0, 2])

    from gst_image.analysis import compute_project_fractions

    fractions = compute_project_fractions(manifest, masks)
    destination = export_analysis(tmp_path / "exported", manifest, masks, fractions=fractions)
    text = (destination / "fractions.csv").read_text(encoding="utf-8")
    assert "Zones: Weld" in text
    assert "Zones: HAZ" in text
