
import cv2
import numpy as np

from gst_image.cli import analyze_image, main
from gst_image.image_io import sha256_file
from gst_image.models import ProjectManifest, SegmentationLayer, SegmentationRecipe, ThresholdMethod
from gst_image.project import (
    load_project,
    relink_source,
    resolve_source,
    save_project,
    validate_project,
)


def test_project_roundtrip_and_source_hash_validation(tmp_path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full((20, 30), 128, np.uint8))
    manifest = ProjectManifest(
        name="roundtrip",
        source_path=str(source),
        source_sha256=sha256_file(source),
        image_width=30,
        image_height=20,
    )
    layer = SegmentationLayer(name="phase")
    manifest.layers.append(layer)
    mask = np.zeros((20, 30), np.uint8)
    mask[:, :10] = 1
    project = save_project(tmp_path / "roundtrip.gstproj", manifest, {layer.id: mask})
    loaded, masks = load_project(project)
    assert loaded.project_id == manifest.project_id
    assert np.array_equal(masks[layer.id], mask)
    assert (project / "previews" / "source.jpg").exists()
    assert validate_project(project) == []
    cv2.imwrite(str(source), np.full((20, 30), 129, np.uint8))
    assert any("SHA-256" in issue for issue in validate_project(project))
    relink_source(loaded, source)
    assert loaded.source_revision == 2
    assert loaded.layers == []


def test_cli_analysis_creates_reopenable_project(tmp_path):
    source = tmp_path / "particles.png"
    image = np.full((100, 140), 210, np.uint8)
    cv2.circle(image, (70, 50), 15, 30, cv2.FILLED)
    cv2.imwrite(str(source), image)
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold=100,
        illumination_correction=False,
        min_particle_area_px=50,
        split_touching=False,
        open_radius_px=0,
        close_radius_px=0,
        tile_size_px=256,
    )
    output = tmp_path / "analysis.gstproj"
    analyze_image(source, output, recipe, 0.01)
    manifest, masks = load_project(output)
    assert manifest.runs[-1].summary["particle_count"] == 1
    assert len(masks) == 2
    assert (output / "results" / "particles.csv").exists()
    assert main(["validate-project", str(output)]) == 0
    exported = tmp_path / "exported"
    assert main(["export", str(output), "--output", str(exported)]) == 0
    assert (exported / "fractions.csv").exists()


def test_portable_project_prefers_its_unchanged_source_copy(tmp_path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full((8, 10), 100, np.uint8))
    manifest = ProjectManifest(
        name="portable",
        source_path=str(source),
        source_sha256=sha256_file(source),
        image_width=10,
        image_height=8,
    )
    project = save_project(tmp_path / "portable.gstproj", manifest, portable=True)
    cv2.imwrite(str(source), np.full((8, 10), 200, np.uint8))
    assert resolve_source(project, manifest) == project / manifest.portable_source
    assert validate_project(project) == []
