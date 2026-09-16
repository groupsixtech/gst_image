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

    def eval(self, image, **kwargs):
        self.received = image.copy()
        self.kwargs = kwargs
        if self.cancelled is not None:
            self.cancelled[0] = True
        return self.labels, [], [], []


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
    assert tuple(model.received[0, 0]) == (10, 20, 30)
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
