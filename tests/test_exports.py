import cv2
import numpy as np

from gst_image.analysis.particles import measure_particles
from gst_image.export import create_overlay, export_analysis
from gst_image.models import (
    ROI,
    AnalysisRun,
    ParticleGroup,
    ParticleGrouping,
    Point,
    ProjectManifest,
    ROIKind,
    SegmentationLayer,
    SegmentationRecipe,
)


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


def test_saved_particle_grouping_exports_rules_assignments_statistics_and_overlay(tmp_path):
    image = np.full((30, 40, 3), 80, np.uint8)
    labels = np.zeros((30, 40), np.int32)
    cv2.circle(labels, (10, 15), 4, 1, cv2.FILLED)
    cv2.circle(labels, (28, 15), 7, 2, cv2.FILLED)
    particles = measure_particles(labels)
    manifest = ProjectManifest(
        name="groups",
        source_path=str(tmp_path / "source.png"),
        source_sha256="0" * 64,
        image_width=40,
        image_height=30,
    )
    roi = ROI(
        name="Export box",
        kind=ROIKind.ANALYSIS_BOX,
        shape="rectangle",
        points=[Point(x=4, y=5), Point(x=35, y=25)],
    )
    particle_class = next(item for item in manifest.classes if item.preset == "particle")
    layer = SegmentationLayer(
        name="Particles",
        kind="instances",
        class_id=particle_class.id,
        scope_roi_ids=[roi.id],
    )
    run = AnalysisRun(
        recipe=SegmentationRecipe(),
        layer_ids=[layer.id],
        summary={"analyzed_pixels": 1200},
    )
    layer.source_run_id = run.id
    grouping = ParticleGrouping(
        name="Two sizes",
        source_layer_id=layer.id,
        groups=[
            ParticleGroup(
                name="Small",
                color="#ff0000",
                size_unit="px",
                size_min=0,
                size_max=5,
                circularity_min=0,
                circularity_max=1,
            ),
            ParticleGroup(
                name="Large",
                color="#00ff00",
                size_unit="px",
                size_min=5,
                size_max=20,
                circularity_min=0,
                circularity_max=1,
            ),
        ],
    )
    manifest.rois.append(roi)
    manifest.layers.append(layer)
    manifest.runs.append(run)
    manifest.particle_records[layer.id] = particles
    manifest.particle_groupings.append(grouping)
    manifest.active_particle_groupings[layer.id] = grouping.id
    original_labels = labels.copy()

    destination = export_analysis(
        tmp_path / "exported-groups",
        manifest,
        {layer.id: labels},
        source_image=image,
    )

    assert (destination / "particle_grouping_definitions.csv").exists()
    assert (destination / "particle_group_assignments.csv").exists()
    assert (destination / "particle_group_statistics.csv").exists()
    assert "Small" in (destination / "particle_group_statistics.csv").read_text(
        encoding="utf-8"
    )
    overlay_files = list(destination.glob("particle_grouping_Two_sizes_*.png"))
    assert overlay_files
    grouped_overlay = cv2.imread(str(overlay_files[0]))
    assert grouped_overlay.shape[:2] == (21, 32)
    assert grouped_overlay[10, 6, 2] > grouped_overlay[10, 6, 1]
    assert grouped_overlay[10, 24, 1] > grouped_overlay[10, 24, 2]
    roi_files = list(destination.glob("roi_analysis_box_Export_box_*.png"))
    assert len(roi_files) == 1
    assert cv2.imread(str(roi_files[0])).shape[:2] == (21, 32)
    segmentation_files = list(destination.glob("segmentation_overlay_Particles_*.png"))
    assert len(segmentation_files) == 1
    assert cv2.imread(str(segmentation_files[0])).shape[:2] == (21, 32)
    assert np.array_equal(labels, original_labels)
