
import json

import cv2
import numpy as np
import pytest

from gst_image.cli import _recipe, analyze_image, main
from gst_image.image_io import sha256_file
from gst_image.models import (
    AnalysisRun,
    ParticlePolarity,
    ProjectManifest,
    SegmentationLayer,
    SegmentationRecipe,
    ThresholdMethod,
)
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
        manual_threshold_low=0,
        manual_threshold_high=99,
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


@pytest.mark.parametrize(
    ("threshold", "polarity", "expected"),
    [
        (100, ParticlePolarity.DARK, (0, 99)),
        (100.25, ParticlePolarity.DARK, (0, 100)),
        (100, ParticlePolarity.BRIGHT, (101, 255)),
        (100.25, ParticlePolarity.BRIGHT, (101, 255)),
        (0, ParticlePolarity.DARK, (0, 0)),
        (255, ParticlePolarity.BRIGHT, (255, 255)),
    ],
)
def test_legacy_manual_threshold_recipe_migrates_to_inclusive_band(
    threshold, polarity, expected
):
    recipe = SegmentationRecipe.model_validate(
        {"manual_threshold": threshold, "polarity": polarity}
    )
    assert (recipe.manual_threshold_low, recipe.manual_threshold_high) == expected
    assert "manual_threshold" not in recipe.model_dump()


def test_cli_loads_legacy_standalone_recipe_json(tmp_path):
    path = tmp_path / "legacy-recipe.json"
    path.write_text(
        json.dumps({"manual_threshold": 120, "polarity": "bright"}),
        encoding="utf-8",
    )
    recipe = _recipe(str(path))
    assert (recipe.manual_threshold_low, recipe.manual_threshold_high) == (121, 255)


def test_version_one_project_migrates_all_recipes_without_changing_masks(tmp_path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full((12, 16), 100, np.uint8))
    recipe = SegmentationRecipe(
        threshold_method=ThresholdMethod.MANUAL,
        manual_threshold_low=0,
        manual_threshold_high=100,
    )
    layer = SegmentationLayer(name="particles")
    manifest = ProjectManifest(
        name="legacy",
        source_path=str(source),
        source_sha256=sha256_file(source),
        image_width=16,
        image_height=12,
        recipes=[recipe],
        layers=[layer],
        runs=[AnalysisRun(recipe=recipe, layer_ids=[layer.id])],
    )
    mask = np.zeros((12, 16), np.uint8)
    mask[:, 4:8] = 1
    project = save_project(tmp_path / "legacy.gstproj", manifest, {layer.id: mask})
    manifest_path = project / "project.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    for recipe_payload in (payload["recipes"][0], payload["runs"][0]["recipe"]):
        recipe_payload.pop("manual_threshold_low")
        recipe_payload.pop("manual_threshold_high")
        recipe_payload["manual_threshold"] = 100.25
        recipe_payload["polarity"] = "dark"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, masks = load_project(project)
    assert loaded.schema_version == 2
    assert (loaded.recipes[0].manual_threshold_low, loaded.recipes[0].manual_threshold_high) == (
        0,
        100,
    )
    assert (
        loaded.runs[0].recipe.manual_threshold_low,
        loaded.runs[0].recipe.manual_threshold_high,
    ) == (0, 100)
    assert np.array_equal(masks[layer.id], mask)

    save_project(project, loaded, masks, source_override=source)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert "manual_threshold" not in saved["recipes"][0]
    assert "manual_threshold" not in saved["runs"][0]["recipe"]
