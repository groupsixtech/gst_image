import hashlib
import json

import cv2
import numpy as np
import pytest

from gst_image import cli
from gst_image.analysis.cellpose_inference import run_cellpose_inference
from gst_image.models import (
    Calibration,
    CellposeInferenceRecipe,
    CellposeInferenceRun,
    ProjectManifest,
    SegmentationLayer,
)
from gst_image.project import load_project, save_project, validate_project


class _CellposeModel:
    def __init__(self, weights, labels, cancelled=None):
        self.pretrained_model = str(weights)
        self.labels = labels
        self.cancelled = cancelled
        self.received = None
        self.kwargs = None
        self.calls = []

    def eval(self, image, **kwargs):
        self.received = image.copy()
        self.kwargs = kwargs
        self.calls.append(image.copy())
        if self.cancelled is not None:
            self.cancelled[0] = True
        labels = self.labels(image) if callable(self.labels) else self.labels
        return labels, [], [], []


def test_cellpose_restores_native_coordinates_clips_domain_and_records_settings(tmp_path):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"stock test weights")
    domain = np.zeros((6, 8), dtype=bool)
    domain[1:5, 1:3] = True
    domain[1:5, 4:6] = True
    labels = np.ones((4, 5), dtype=np.int32)
    model = _CellposeModel(weights, labels)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[1, 1] = (10, 20, 30)
    recipe = CellposeInferenceRecipe(
        modality="metallography",
        diameter_px=40,
        cellprob_threshold=-0.5,
        flow_threshold=0.6,
        min_size_px=3,
        tile_overlap=0.2,
        noncommercial_license_accepted=True,
    )

    result = run_cellpose_inference(image, domain, recipe, model=model)

    assert model.received.shape == (4, 5, 3)
    # Cellpose reads RGB; the BGR source pixel arrives channel-reversed.
    assert tuple(model.received[0, 0]) == (30, 20, 10)
    assert model.kwargs == {
        "diameter": 40,
        "flow_threshold": 0.6,
        "cellprob_threshold": -0.5,
        "min_size": 3,
        "tile_overlap": 0.2,
    }
    assert result.source_bounds_px == (1, 1, 6, 5)
    assert not np.any(result.labels[~domain])
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert len(result.particles) == 2
    assert result.summary["modality"] == "metallography"
    assert result.model_sha256 == hashlib.sha256(weights.read_bytes()).hexdigest()


def _boxed_domain(shape, boxes):
    domain = np.zeros(shape, dtype=bool)
    for left, top, right, bottom in boxes:
        domain[top:bottom, left:right] = True
    return domain


@pytest.mark.parametrize("resolution", [100, 50])
def test_cellpose_runs_once_per_analysis_box_instead_of_the_whole_overview(tmp_path, resolution):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    boxes = [(1, 1, 5, 5), (12, 2, 16, 6)]
    domain = _boxed_domain((10, 20), boxes)
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    model = _CellposeModel(weights, lambda crop: np.ones(crop.shape[:2], dtype=np.int32))
    recipe = CellposeInferenceRecipe(
        resolution_percent=resolution, noncommercial_license_accepted=True
    )

    result = run_cellpose_inference(image, domain, recipe, regions=boxes, model=model)

    size = 4 * resolution // 100
    assert [crop.shape for crop in model.calls] == [(size, size, 3)] * 2
    assert result.input_dimensions_px == [(size, size)] * 2
    assert result.summary["input_pixels"] == 2 * size**2
    assert result.region_bounds_px == boxes
    assert result.source_bounds_px == (1, 1, 16, 6)
    assert result.summary["analysis_regions"] == 2
    # Each box yields its own instance, and nothing outside the boxes is labelled.
    assert len(result.particles) == 2
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert not np.any(result.labels[~domain])
    assert result.summary["analyzed_pixels"] == 32


@pytest.mark.parametrize("resolution", [100, 50])
def test_cellpose_regions_clip_to_the_domain_and_merge_where_boxes_overlap(tmp_path, resolution):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    domain = _boxed_domain((10, 10), [(2, 2, 8, 8)])
    model = _CellposeModel(weights, lambda crop: np.zeros(crop.shape[:2], dtype=np.int32))
    recipe = CellposeInferenceRecipe(
        resolution_percent=resolution, noncommercial_license_accepted=True
    )

    result = run_cellpose_inference(
        np.zeros((10, 10, 3), dtype=np.uint8),
        domain,
        recipe,
        # Two overlapping requests plus one wholly outside the domain.
        regions=[(0, 0, 6, 10), (4, 0, 10, 10), (0, 9, 2, 10)],
        model=model,
    )

    assert result.region_bounds_px == [(2, 2, 8, 8)]
    size = 6 * resolution // 100
    assert [crop.shape for crop in model.calls] == [(size, size, 3)]


def test_cellpose_rejects_regions_that_miss_the_analysis_domain(tmp_path):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    recipe = CellposeInferenceRecipe(noncommercial_license_accepted=True)
    with pytest.raises(ValueError, match="No analysis region"):
        run_cellpose_inference(
            np.zeros((10, 10, 3), dtype=np.uint8),
            _boxed_domain((10, 10), [(0, 0, 3, 3)]),
            recipe,
            regions=[(5, 5, 9, 9)],
            model=_CellposeModel(weights, np.zeros((4, 4), dtype=np.int32)),
        )


def test_cellpose_cache_path_follows_the_installed_cellpose_constant(monkeypatch, tmp_path):
    from gst_image.analysis import cellpose_inference

    class _Models:
        MODEL_DIR = tmp_path / "cellpose-models"

    monkeypatch.setattr(cellpose_inference, "_cellpose_models", lambda: _Models)

    assert cellpose_inference.cellpose_cache_path() == tmp_path / "cellpose-models" / "cpsam_v2"


@pytest.mark.parametrize("resolution", [100, 50])
def test_cellpose_requires_acknowledgement_and_discards_late_cancelled_result(tmp_path, resolution):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    labels = np.ones((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="non-commercial"):
        run_cellpose_inference(image, None, CellposeInferenceRecipe(), model=_CellposeModel(weights, labels))

    cancelled = [False]
    recipe = CellposeInferenceRecipe(
        resolution_percent=resolution, noncommercial_license_accepted=True
    )
    with pytest.raises(InterruptedError, match="cancelled"):
        run_cellpose_inference(
            image,
            None,
            recipe,
            model=_CellposeModel(weights, labels, cancelled),
            cancelled=lambda: cancelled[0],
        )


def test_cellpose_gpu_recipe_checks_cuda_before_running_model(tmp_path, monkeypatch):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    calls = []
    monkeypatch.setattr(
        "gst_image.analysis.cellpose_inference._require_cuda_gpu", lambda: calls.append("checked")
    )
    recipe = CellposeInferenceRecipe(device="gpu", noncommercial_license_accepted=True)
    result = run_cellpose_inference(
        np.zeros((4, 4, 3), dtype=np.uint8),
        None,
        recipe,
        model=_CellposeModel(weights, np.ones((4, 4), dtype=np.int32)),
    )

    assert calls == ["checked"]
    assert result.summary["device"] == "gpu"


def test_cellpose_provenance_roundtrips_and_project_validation_checks_owned_layers(tmp_path):
    recipe = CellposeInferenceRecipe(noncommercial_license_accepted=True)
    layer = SegmentationLayer(name="Cells", kind="instances", review_status="pending")
    domain = SegmentationLayer(name="Cellpose domain", kind="domain", review_status="pending")
    run = CellposeInferenceRun(
        recipe=recipe,
        layer_ids=[layer.id, domain.id],
        summary={"analyzed_pixels": 64},
        model_sha256="a" * 64,
        model_cache_path="C:/cellpose/cpsam_v2",
    )
    layer.source_run_id = run.id
    domain.source_run_id = run.id
    manifest = ProjectManifest(
        name="cellpose provenance",
        source_path=str(tmp_path / "missing.png"),
        source_sha256="0" * 64,
        image_width=8,
        image_height=8,
        layers=[layer, domain],
        cellpose_inference_recipes=[recipe],
        cellpose_inference_runs=[run],
    )

    project = save_project(tmp_path / "cellpose.gstproj", manifest)
    loaded, _ = load_project(project)

    assert loaded.schema_version == 7
    assert loaded.cellpose_inference_runs[0].recipe.model_id == "cpsam_v2"
    assert validate_project(project, verify_hash=False) == [
        f"Source image is missing: {tmp_path / 'missing.png'}"
    ]


def test_cellpose_cli_requires_explicit_licence_flag_and_builds_recipe(monkeypatch, tmp_path):
    captured = {}

    def fake_inference(input_path, output, recipe, mm_per_pixel, **kwargs):
        captured.update(
            input_path=input_path,
            output=output,
            recipe=recipe,
            mm_per_pixel=mm_per_pixel,
            kwargs=kwargs,
        )

    monkeypatch.setattr(cli, "infer_cellpose_image", fake_inference)
    args = cli.build_parser().parse_args(
        [
            "infer-cellpose",
            "cells.png",
            "--output",
            str(tmp_path / "cells.gstproj"),
            "--modality",
            "metallography",
            "--device",
            "gpu",
            "--accept-cellpose-noncommercial-license",
            "--allow-model-download",
            "--diameter",
            "42",
            "--resolution-percent",
            "50",
        ]
    )

    assert args.handler(args) == 0
    assert captured["recipe"].modality == "metallography"
    assert captured["recipe"].device == "gpu"
    assert captured["recipe"].diameter_px == 42
    assert captured["recipe"].resolution_percent == 50
    assert captured["recipe"].noncommercial_license_accepted
    assert captured["kwargs"]["allow_model_download"]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_reduced_cellpose_preserves_calibration_coordinates_and_tuning(tmp_path, monkeypatch, device):
    monkeypatch.setattr("gst_image.analysis.cellpose_inference._require_cuda_gpu", lambda: None)
    image = np.empty((16, 20, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    domain = _boxed_domain(image.shape[:2], [(4, 2, 16, 14)])
    small_labels = np.zeros((6, 6), dtype=np.int32)
    small_labels[1:5, 1:3] = 7
    small_labels[1:5, 3:5] = 8  # Touching instances must remain separate.
    calibration = Calibration(mm_per_pixel=0.025)
    original_calibration = calibration.model_dump()
    results = []
    for resolution, labels in [(100, small_labels.repeat(2, 0).repeat(2, 1)), (50, small_labels)]:
        model = _CellposeModel(tmp_path / "weights", labels)
        recipe = CellposeInferenceRecipe(
            resolution_percent=resolution, device=device, diameter_px=46, min_size_px=15,
            noncommercial_license_accepted=True,
        )
        progress = []
        result = run_cellpose_inference(
            image, domain, recipe, calibration, model=model,
            progress=lambda value, message, messages=progress: messages.append(message),
        )
        results.append(result)
        assert tuple(model.received[0, 0]) == (30, 20, 10)
        assert model.kwargs["diameter"] == 46
        assert model.kwargs["min_size"] == 15
        assert result.summary["analyzed_pixels"] == 144
        assert result.summary["resolution_percent"] == resolution
        assert any(f"at {resolution}%" in message for message in progress)
    assert results[1].input_dimensions_px == [(6, 6)]
    assert results[1].summary["input_pixels"] == results[0].summary["input_pixels"] / 4
    np.testing.assert_array_equal(results[0].labels, results[1].labels)
    assert results[0].particles == results[1].particles
    assert len(results[1].particles) == 2
    first = results[1].particles[0]
    assert first.area_px == 32
    assert first.area_mm2 == pytest.approx(32 * 0.025**2)
    assert first.equivalent_diameter_mm == pytest.approx(2 * np.sqrt(32 / np.pi) * 0.025)
    assert (first.centroid_x_px, first.centroid_y_px) == (7.5, 7.5)
    assert calibration.model_dump() == original_calibration


def test_reduced_cellpose_restores_odd_dimensions_then_clips_and_splits(tmp_path):
    domain = _boxed_domain((10, 14), [(3, 2, 12, 9)])
    domain[2:9, 7] = False  # A thin exclusion must survive coarse inference.
    model = _CellposeModel(tmp_path / "weights", lambda crop: np.ones(crop.shape[:2], np.int32))
    result = run_cellpose_inference(
        np.zeros((10, 14, 3), np.uint8), domain,
        CellposeInferenceRecipe(resolution_percent=50, noncommercial_license_accepted=True),
        model=model,
    )
    assert model.received.shape == (4, 4, 3)
    assert result.labels.dtype == np.int32
    np.testing.assert_array_equal(result.labels > 0, domain)
    assert len(result.particles) == 2
    assert all(p.border_touching for p in result.particles)
    assert result.summary["area_fraction"] == 1


def test_reduced_cellpose_tiny_crop_still_has_one_input_pixel(tmp_path):
    model = _CellposeModel(tmp_path / "weights", np.ones((1, 1), np.int32))
    result = run_cellpose_inference(
        np.zeros((2, 3, 3), np.uint8), None,
        CellposeInferenceRecipe(resolution_percent=1, noncommercial_license_accepted=True),
        model=model,
    )
    assert model.received.shape == (1, 1, 3)
    assert result.labels.shape == (2, 3)
    assert result.particles[0].area_px == 6


@pytest.mark.parametrize("value", [0, 101, -1, 50.5])
def test_cellpose_recipe_rejects_invalid_resolution(value):
    with pytest.raises(ValueError, match="resolution_percent"):
        CellposeInferenceRecipe(resolution_percent=value)


@pytest.mark.parametrize("value", ["0", "101", "-1"])
def test_cellpose_cli_rejects_invalid_resolution_before_inference(value, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid resolution must not start inference")

    monkeypatch.setattr(cli, "infer_cellpose_image", unexpected)
    args = cli.build_parser().parse_args([
        "infer-cellpose", "unused.png", "--output", "unused.gstproj",
        "--resolution-percent", value, "--accept-cellpose-noncommercial-license",
    ])
    with pytest.raises(ValueError, match="resolution_percent"):
        args.handler(args)


def test_reduced_cellpose_cli_roundtrip_export_and_legacy_migration(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((12, 16, 3), 100, np.uint8))
    monkeypatch.setattr(cli, "suggest_specimen_mask", lambda gray: np.ones(gray.shape, bool))
    model = _CellposeModel(tmp_path / "weights", lambda crop: np.ones(crop.shape[:2], np.int32))
    monkeypatch.setattr(
        cli, "run_cellpose_inference",
        lambda *args, **kwargs: run_cellpose_inference(*args, model=model, **kwargs),
    )
    args = cli.build_parser().parse_args([
        "infer-cellpose", str(source), "--output", str(tmp_path / "result.gstproj"),
        "--resolution-percent", "50", "--mm-per-pixel", "0.1",
        "--accept-cellpose-noncommercial-license",
    ])
    assert args.handler(args) == 0
    root = tmp_path / "result.gstproj"
    manifest, masks = load_project(root)
    run = manifest.cellpose_inference_runs[0]
    assert run.recipe.resolution_percent == 50
    assert run.input_dimensions_px == [(8, 6)]
    assert run.summary["input_pixels"] == 48
    assert run.summary["analyzed_pixels"] == 192
    assert run.review_status == "pending"
    assert all(layer.review_status == "pending" for layer in manifest.layers)
    assert all(mask.shape == (12, 16) for mask in masks.values())
    assert validate_project(root) == []
    provenance = json.loads((root / "results/provenance.json").read_text(encoding="utf-8"))
    assert provenance["cellpose_inference_runs"][0]["input_dimensions_px"] == [[8, 6]]
    recipes = json.loads((root / "results/recipes.json").read_text(encoding="utf-8"))
    assert recipes["cellpose_inference"][0]["resolution_percent"] == 50
    particle = next(iter(manifest.particle_records.values()))[0]
    assert particle.area_mm2 == pytest.approx(1.92)

    # Old recipes mean native resolution, both standalone and embedded in history.
    payload = manifest.model_dump(mode="json")
    payload["schema_version"] = 5
    payload["cellpose_inference_recipes"][0].pop("resolution_percent")
    old_run = payload["cellpose_inference_runs"][0]
    old_run["recipe"].pop("resolution_percent")
    old_run.pop("input_dimensions_px")
    for key in ("resolution_percent", "input_pixels"):
        old_run["summary"].pop(key)
    old_run["summary"]["analysis_resolution"] = "original"
    (root / "project.json").write_text(json.dumps(payload), encoding="utf-8")
    migrated, masks = load_project(root)
    assert migrated.schema_version == 7
    assert migrated.cellpose_inference_recipes[0].resolution_percent == 100
    assert migrated.cellpose_inference_runs[0].recipe.resolution_percent == 100
    assert migrated.cellpose_inference_runs[0].input_dimensions_px == []
    save_project(root, migrated, masks)
    reloaded, _ = load_project(root)
    assert reloaded.cellpose_inference_runs == migrated.cellpose_inference_runs


def test_clip_and_split_keeps_touching_instances_apart_and_splits_severed_ones():
    from gst_image.analysis.cellpose_inference import _clip_and_split_labels

    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:5, 1:3] = 7  # touches label 8 along a shared edge
    labels[1:5, 3:5] = 8
    labels[5, 1:5] = 9  # severed in two by the domain below
    domain = np.ones((6, 6), dtype=bool)
    domain[5, 3] = False

    result = _clip_and_split_labels(labels, domain)

    # Touching labels stay distinct; the severed one becomes two instances.
    assert sorted(np.bincount(result.ravel())[1:].tolist()) == [1, 2, 8, 8]
    assert result[1, 1] != result[1, 3]
    assert result[5, 1] != result[5, 4]
    assert result[5, 3] == 0


def test_clip_and_split_cost_does_not_scale_with_the_instance_count():
    import time

    from gst_image.analysis.cellpose_inference import _clip_and_split_labels

    shape = (600, 600)
    domain = np.ones(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    count = 0
    for top in range(0, shape[0] - 10, 10):
        for left in range(0, shape[1] - 10, 10):
            count += 1
            labels[top + 1 : top + 9, left + 1 : left + 9] = count

    started = time.perf_counter()
    result = _clip_and_split_labels(labels, domain)
    elapsed = time.perf_counter() - started

    assert int(result.max()) == count == 3481
    # The per-instance loop this replaced needed minutes for this many instances.
    assert elapsed < 2.0
