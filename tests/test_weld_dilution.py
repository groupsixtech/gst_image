import cv2
import numpy as np
import pytest

from gst_image.analysis.weld_dilution import (
    append_dilution_run,
    blurred_weld_preview,
    envelope_from_seeds,
    fit_surface_reference,
    measure_weld_dilution,
    scaled_weld_recipe,
    segment_weld_envelope,
)
from gst_image.cli import main
from gst_image.image_io import sha256_file
from gst_image.models import (
    Calibration,
    Point,
    ProjectManifest,
    WeldBlurStroke,
    WeldDilutionDraft,
    WeldDilutionRecipe,
    WeldSampling,
)
from gst_image.project import load_project, relink_source, save_project, validate_project


def test_independent_blur_matches_segmentation_blur_and_handles_cancellation():
    image = np.random.default_rng(12).integers(0, 256, (270, 530), dtype=np.uint8)
    recipe = WeldDilutionRecipe(gaussian_sigma_px=10, tile_size_px=256)
    blurred = blurred_weld_preview(image, recipe)
    segmented = segment_weld_envelope(
        image, np.ones_like(image, bool), recipe, return_blurred=True,
    )
    np.testing.assert_array_equal(blurred, segmented.blurred)
    np.testing.assert_array_equal(
        blurred, cv2.GaussianBlur(image, (81, 81), 10, borderType=cv2.BORDER_REPLICATE),
    )
    with pytest.raises(InterruptedError):
        blurred_weld_preview(image, recipe, cancelled=lambda: True)


def test_local_blur_changes_only_painted_pixels_and_replays_in_tiles():
    image = np.random.default_rng(6).integers(0, 256, (300, 600), dtype=np.uint8)
    stroke = WeldBlurStroke(
        points=[Point(x=130.5, y=60.5), Point(x=500.5, y=210.5)], radius_px=24, sigma_px=5,
    )
    recipe = WeldDilutionRecipe(
        gaussian_sigma_px=0, blur_strokes=[stroke], tile_size_px=128, window_px=31,
    )
    preview = blurred_weld_preview(image, recipe, max_size=None)
    assert not np.array_equal(preview, image)
    np.testing.assert_array_equal(preview[:20], image[:20])
    assert preview[135, 315] == cv2.GaussianBlur(
        image, (41, 41), 5, borderType=cv2.BORDER_REPLICATE,
    )[135, 315]
    domain = np.ones_like(image, bool)
    tiled = segment_weld_envelope(image, domain, recipe, return_blurred=True)
    recipe.tile_size_px = 2048
    whole = segment_weld_envelope(image, domain, recipe, return_blurred=True)
    np.testing.assert_array_equal(preview, whole.blurred)
    np.testing.assert_array_equal(tiled.blurred, whole.blurred)
    np.testing.assert_array_equal(tiled.candidates, whole.candidates)
    scaled = scaled_weld_recipe(recipe, 0.5).blur_strokes[0]
    assert scaled.radius_px == 12 and scaled.sigma_px == 2.5
    assert scaled.points[0].x == stroke.points[0].x / 2


def rectangle():
    envelope = np.zeros((40, 60), bool)
    envelope[10:30, 10:50] = True
    return envelope, np.ones_like(envelope)


def test_exact_pixel_cell_areas_and_volume():
    envelope, domain = rectangle()
    ref = fit_surface_reference([(0, 19.5), (59, 19.5)])
    summary, lines = measure_weld_dilution(
        envelope,
        domain,
        ref,
        Calibration(mm_per_pixel=0.1),
        WeldSampling(weld_length_mm=10),
    )
    assert summary.dilution_percent == pytest.approx(50)
    assert summary.penetration_area_px2 == 400
    assert summary.envelope_area_mm2 == pytest.approx(8)
    assert summary.envelope_volume_mm3 == pytest.approx(80)
    assert summary.depth_px.count == 40
    assert summary.depth_px.mean == 10
    assert summary.height_px.mean == 10
    assert summary.depth_mm.mean == 1
    assert summary.quality_flags == []
    assert any(p.status == "no_weld" and p.depth_px is None for p in lines)


def test_fractional_horizon_conserves_area_and_spacing_independence():
    envelope, domain = rectangle()
    ref = fit_surface_reference([(0, 20), (59, 20)])
    first, _ = measure_weld_dilution(envelope, domain, ref)
    second, _ = measure_weld_dilution(envelope, domain, ref, sampling=WeldSampling(spacing=2.3))
    assert first.penetration_area_px2 == pytest.approx(380)
    assert first.envelope_area_px2 == first.penetration_area_px2 + first.reinforcement_area_px2
    assert first.dilution_percent == second.dilution_percent


def test_endpoint_order_fit_flip_and_tilt():
    points = [(0, 1), (10, 6), (20, 11)]
    ref = fit_surface_reference(points)
    reverse = fit_surface_reference(points[::-1])
    assert ref.normal.x == pytest.approx(reverse.normal.x)
    assert ref.normal.y == pytest.approx(reverse.normal.y)
    assert ref.rms_residual_px < 1e-12
    assert ref.span_px == pytest.approx(np.sqrt(500))
    envelope, domain = rectangle()
    a, _ = measure_weld_dilution(envelope, domain, ref)
    b, _ = measure_weld_dilution(envelope, domain, fit_surface_reference(points, flipped=True))
    assert a.penetration_area_px2 + b.penetration_area_px2 == pytest.approx(envelope.sum())


def test_calibration_scaling_and_physical_spacing():
    envelope, domain = rectangle()
    ref = fit_surface_reference([(0, 20), (50, 20)])
    a, _ = measure_weld_dilution(envelope, domain, ref, Calibration(mm_per_pixel=0.1))
    b, lines = measure_weld_dilution(
        envelope,
        domain,
        ref,
        Calibration(mm_per_pixel=0.2),
        WeldSampling(spacing=200, spacing_unit="µm"),
    )
    assert a.dilution_percent == b.dilution_percent
    assert b.depth_mm.mean == 2 * a.depth_mm.mean
    assert b.envelope_area_mm2 == 4 * a.envelope_area_mm2
    assert len(lines) == 60
    with pytest.raises(ValueError, match="calibration"):
        measure_weld_dilution(envelope, domain, ref, sampling=WeldSampling(weld_length_mm=2))


@pytest.mark.parametrize(
    "case,status",
    [("gap", "no_contact"), ("multiple", "multiple"), ("clipped", "clipped"), ("zero", "valid")],
)
def test_tie_line_invalid_and_zero_cases(case, status):
    envelope, domain = rectangle()
    if case == "gap":
        envelope[10:22] = False
    if case == "multiple":
        envelope[14:16] = False
    if case == "clipped":
        domain[25:] = False
    ref = fit_surface_reference(
        [(0, 9.5 if case == "zero" else 19.5), (59, 9.5 if case == "zero" else 19.5)]
    )
    summary, lines = measure_weld_dilution(envelope, domain, ref)
    sample = next(p for p in lines if p.x == 20)
    assert sample.status == status
    if case == "zero":
        assert sample.height_px == 0
        assert summary.height_px.mean == 0
    else:
        assert sample.depth_px is None
    if case == "clipped":
        assert "partial_section" in summary.quality_flags


def test_invalid_geometry_and_empty_envelope():
    with pytest.raises(ValueError):
        fit_surface_reference([(1, 2), (1, 2)])
    with pytest.raises(ValueError):
        fit_surface_reference([(np.nan, 2), (1, 2)])
    envelope, domain = rectangle()
    with pytest.raises(ValueError, match="empty"):
        measure_weld_dilution(envelope * 0, domain, fit_surface_reference([(0, 2), (3, 2)]))


@pytest.mark.parametrize("method", ["adaptive_gaussian", "sauvola"])
@pytest.mark.parametrize("polarity", ["dark", "bright"])
def test_tiled_segmentation_gradient_blur_and_holes(method, polarity):
    h, w = 180, 340
    image = np.tile(np.linspace(120, 220, w), (h, 1))
    image[45:135, 40:300] -= 65
    image[70:90, 90:105] += 65
    image = image.astype(np.uint8)
    if polarity == "bright":
        image = 255 - image
    domain = np.ones((h, w), bool)
    domain[80:90, 180:190] = False
    settings = {
        "threshold_method": method,
        "polarity": polarity,
        "gaussian_sigma_px": 8,
        "window_px": 51,
        "close_radius_px": 2,
        "open_radius_px": 1,
    }
    small = segment_weld_envelope(image, domain, WeldDilutionRecipe(**settings, tile_size_px=64))
    large = segment_weld_envelope(image, domain, WeldDilutionRecipe(**settings, tile_size_px=2048))
    assert np.array_equal(small.candidates > 0, large.candidates > 0)
    assert not np.any(small.candidates[~domain])
    assert np.any(small.candidates)


def test_component_selection_and_cancellation():
    labels = np.zeros((30, 30), np.int32)
    labels[3:12, 3:12] = 1
    labels[6:8, 6:8] = 0
    labels[20:25, 20:25] = 2
    domain = np.ones_like(labels, bool)
    domain[7, 7] = False
    mask = envelope_from_seeds(labels, [Point(x=4, y=4)], domain)
    assert mask[6, 6] and not mask[7, 7] and not mask[22, 22]
    with pytest.raises(ValueError, match="no longer"):
        envelope_from_seeds(labels, [Point(x=0, y=0)], domain)
    with pytest.raises(InterruptedError):
        segment_weld_envelope(
            np.ones_like(labels, np.uint8), domain, WeldDilutionRecipe(), cancelled=lambda: True
        )


def make_project(tmp_path):
    envelope, domain = rectangle()
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.full(envelope.shape, 150, np.uint8))
    manifest = ProjectManifest(
        name="weld",
        source_path=str(source),
        source_sha256=sha256_file(source),
        image_width=60,
        image_height=40,
        calibration=Calibration(mm_per_pixel=0.1),
    )
    draft = WeldDilutionDraft(reference=fit_surface_reference([(0, 20), (59, 20)]))
    summary, lines = measure_weld_dilution(envelope, domain, draft.reference, manifest.calibration)
    masks = {}
    run = append_dilution_run(manifest, masks, draft, envelope, domain, summary, lines)
    manifest.dilution_drafts.append(draft)
    return manifest, masks, run


def test_project_roundtrip_cli_exports_and_source_lifecycle(tmp_path):
    manifest, masks, run = make_project(tmp_path)
    path = save_project(tmp_path / "input.gstproj", manifest, masks)
    before = (path / "project.json").read_bytes()
    assert validate_project(path) == []
    loaded, restored = load_project(path)
    assert loaded.dilution_runs[0] == run
    assert np.array_equal(restored[run.envelope_layer_id], masks[run.envelope_layer_id])
    output = tmp_path / "replay"
    assert main(["dilution", str(path), "--run-id", run.id, "--output", str(output)]) == 0
    replay, _ = load_project(output / "dilution.gstproj")
    assert replay.dilution_runs[0].summary == run.summary
    assert validate_project(output / "dilution.gstproj") == []
    export = output / "dilution" / replay.dilution_runs[0].id
    for filename in (
        "summary.csv",
        "summary.json",
        "tie_lines.csv",
        "depth_histogram.csv",
        "profile.png",
        "histogram.png",
        "envelope.tif",
        "divided_regions.tif",
        "annotated_panorama.png",
        "provenance.json",
        "recipe_reference.json",
    ):
        assert (export / filename).stat().st_size > 0
    assert (path / "project.json").read_bytes() == before
    relink_source(loaded, manifest.source_path)
    assert not loaded.dilution_runs and not loaded.dilution_drafts


def test_schema_6_migration_and_ownership_validation(tmp_path):
    import json

    manifest, masks, _run = make_project(tmp_path)
    path = save_project(tmp_path / "broken.gstproj", manifest, masks)
    payload = json.loads((path / "project.json").read_text())
    payload["layers"][0]["source_run_id"] = "missing"
    (path / "project.json").write_text(json.dumps(payload))
    assert any("Dilution" in issue for issue in validate_project(path))
    payload["schema_version"] = 6
    payload.pop("dilution_drafts")
    payload.pop("dilution_runs")
    payload["layers"] = []
    (path / "project.json").write_text(json.dumps(payload))
    loaded, _ = load_project(path)
    assert loaded.schema_version == 7
    assert loaded.dilution_runs == loaded.dilution_drafts == []


def test_tilted_tie_lines_are_perpendicular_and_uniform():
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    ref = fit_surface_reference([(0, 0), (39, 39)])
    summary, lines = measure_weld_dilution(
        mask, np.ones_like(mask), ref, sampling=WeldSampling(spacing=np.sqrt(2))
    )
    assert summary.dilution_percent == pytest.approx(50)
    assert np.diff([p.position_px for p in lines]) == pytest.approx(np.sqrt(2))
    for line in lines:
        if line.status == "valid":
            expected = min(line.x - 9.5, 29.5 - line.x) * np.sqrt(2)
            assert line.depth_px == pytest.approx(expected)
            assert line.height_px == pytest.approx(expected)


def test_cli_resegment_discards_manual_corrections_and_review(tmp_path):
    source = tmp_path / "weld.png"
    image = np.full((80, 140), 160, np.uint8)
    image[20:60, 20:120] = 70
    cv2.imwrite(str(source), image)
    domain = np.ones_like(image, bool)
    recipe = WeldDilutionRecipe(
        gaussian_sigma_px=2, window_px=51,
        blur_strokes=[WeldBlurStroke(
            points=[Point(x=60, y=40)], radius_px=18, sigma_px=8,
        )],
    )
    candidates = segment_weld_envelope(image, domain, recipe).candidates
    ys, xs = np.nonzero(candidates)
    seed = Point(x=int(xs[len(xs) // 2]), y=int(ys[len(ys) // 2]))
    expected = envelope_from_seeds(candidates, [seed], domain)
    edited = expected.copy()
    edited[3:6, 3:6] = True
    draft = WeldDilutionDraft(
        reference=fit_surface_reference([(0, 39.5), (139, 39.5)]),
        recipe=recipe,
        component_seeds=[seed],
        manually_edited=True,
    )
    manifest = ProjectManifest(
        name="test",
        source_path=str(source),
        source_sha256=sha256_file(source),
        image_width=140,
        image_height=80,
        dilution_drafts=[draft],
    )
    summary, lines = measure_weld_dilution(edited, domain, draft.reference)
    masks = {}
    run = append_dilution_run(manifest, masks, draft, edited, domain, summary, lines)
    run.review_status = "confirmed"
    path = save_project(tmp_path / "source.gstproj", manifest, masks)
    output = tmp_path / "regenerated"
    assert (
        main(["dilution", str(path), "--run-id", run.id, "--output", str(output), "--resegment"])
        == 0
    )
    result, masks = load_project(output / "dilution.gstproj")
    replay = result.dilution_runs[0]
    assert not replay.manually_edited and replay.review_status == "pending"
    assert np.array_equal(masks[replay.envelope_layer_id], expected)


def test_saved_run_masks_are_immutable_snapshots(tmp_path):
    manifest, masks, run = make_project(tmp_path)
    saved = masks[run.envelope_layer_id].copy()
    mask, domain = rectangle()
    draft = manifest.dilution_drafts[0]
    summary, lines = measure_weld_dilution(mask, domain, draft.reference)
    append_dilution_run(manifest, masks, draft, mask, domain, summary, lines)
    mask[:] = False
    assert np.array_equal(masks[run.envelope_layer_id], saved)
    assert len(manifest.dilution_runs) == 2
