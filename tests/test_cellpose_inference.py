import hashlib

import numpy as np
import pytest

from gst_image import cli
from gst_image.analysis.cellpose_inference import run_cellpose_inference
from gst_image.models import (
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


def test_cellpose_runs_once_per_analysis_box_instead_of_the_whole_overview(tmp_path):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    boxes = [(1, 1, 5, 5), (12, 2, 16, 6)]
    domain = _boxed_domain((10, 20), boxes)
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    model = _CellposeModel(weights, lambda crop: np.ones(crop.shape[:2], dtype=np.int32))
    recipe = CellposeInferenceRecipe(noncommercial_license_accepted=True)

    result = run_cellpose_inference(image, domain, recipe, regions=boxes, model=model)

    assert [crop.shape for crop in model.calls] == [(4, 4, 3), (4, 4, 3)]
    assert result.region_bounds_px == boxes
    assert result.source_bounds_px == (1, 1, 16, 6)
    assert result.summary["analysis_regions"] == 2
    # Each box yields its own instance, and nothing outside the boxes is labelled.
    assert len(result.particles) == 2
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert not np.any(result.labels[~domain])
    assert result.summary["analyzed_pixels"] == 32


def test_cellpose_regions_clip_to_the_domain_and_merge_where_boxes_overlap(tmp_path):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    domain = _boxed_domain((10, 10), [(2, 2, 8, 8)])
    model = _CellposeModel(weights, lambda crop: np.zeros(crop.shape[:2], dtype=np.int32))
    recipe = CellposeInferenceRecipe(noncommercial_license_accepted=True)

    result = run_cellpose_inference(
        np.zeros((10, 10, 3), dtype=np.uint8),
        domain,
        recipe,
        # Two overlapping requests plus one wholly outside the domain.
        regions=[(0, 0, 6, 10), (4, 0, 10, 10), (0, 9, 2, 10)],
        model=model,
    )

    assert result.region_bounds_px == [(2, 2, 8, 8)]
    assert [crop.shape for crop in model.calls] == [(6, 6, 3)]


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


def test_cellpose_requires_acknowledgement_and_discards_late_cancelled_result(tmp_path):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    labels = np.ones((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="non-commercial"):
        run_cellpose_inference(image, None, CellposeInferenceRecipe(), model=_CellposeModel(weights, labels))

    cancelled = [False]
    recipe = CellposeInferenceRecipe(noncommercial_license_accepted=True)
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

    assert loaded.schema_version == 5
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
        ]
    )

    assert args.handler(args) == 0
    assert captured["recipe"].modality == "metallography"
    assert captured["recipe"].device == "gpu"
    assert captured["recipe"].diameter_px == 42
    assert captured["recipe"].noncommercial_license_accepted
    assert captured["kwargs"]["allow_model_download"]


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
