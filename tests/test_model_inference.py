import hashlib
import json

import numpy as np
import pytest

from gst_image.analysis.model_inference import ModelPack, run_model_inference
from gst_image.models import ModelInferenceRun, ProjectManifest
from gst_image.project import load_project, save_project


class _Input:
    name = "image"


class _Session:
    def __init__(self, outputs):
        self.outputs = outputs
        self.received = None

    def get_inputs(self):
        return [_Input()]

    def run(self, names, feed):
        self.received = feed["image"]
        return [self.outputs[name](self.received) for name in names]


def _pack(tmp_path, outputs, *, tile_size=32, overlap=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test weights")
    manifest = {
        "model_id": "etched-optical-v1",
        "model_version": "1.0.0",
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "input": {"channel_order": "RGB", "mean": [0, 0, 0], "std": [1, 1, 1]},
        "tiling": {"tile_size_px": tile_size, "overlap_px": overlap},
        "outputs": outputs,
        "semantic_classes": {"0": "Background", "1": "Ferrite", "2": "Martensite"},
        "training_data_summary": "reviewed etched optical crops",
        "validation_metrics": {"macro_dice": 0.91},
        "license": "Proprietary validation data; BSD runtime",
        "attribution": "GST Image lab model",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ModelPack.open(tmp_path)


def test_model_pack_validates_weights_and_inference_normalizes_rgb(tmp_path):
    outputs = [
        {"name": "semantic", "kind": "semantic", "activation": "softmax"},
        {"name": "foreground", "kind": "particle_foreground"},
        {"name": "boundary", "kind": "particle_boundary"},
    ]
    pack = _pack(tmp_path, outputs)

    def semantic(input_tensor):
        _, _, height, width = input_tensor.shape
        result = np.zeros((1, 3, height, width), dtype=np.float32)
        result[:, 1] = 1
        return result

    def particle(input_tensor):
        _, _, height, width = input_tensor.shape
        result = np.zeros((1, 1, height, width), dtype=np.float32)
        result[:, :, 8:24, 8:24] = 0.9
        return result

    session = _Session({"semantic": semantic, "foreground": particle, "boundary": particle})
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)  # BGR source pixel
    domain = np.ones((32, 32), dtype=bool)
    domain[0, 0] = False
    result = run_model_inference(image, domain, pack, session=session)

    assert np.allclose(session.received[0, :, 0, 0], [30 / 255, 20 / 255, 10 / 255])
    assert result.semantic_labels[1, 1] == 1
    assert result.semantic_labels[0, 0] == 0
    assert result.particle_labels is not None
    assert not np.any(result.particle_labels[~domain])
    assert result.summary["model_sha256"] == pack.model_sha256


def test_tiled_model_output_is_blended_and_cancellation_is_observed(tmp_path):
    outputs = [{"name": "semantic", "kind": "semantic", "activation": "softmax"}]
    pack = _pack(tmp_path, outputs)

    def semantic(input_tensor):
        _, _, height, width = input_tensor.shape
        result = np.zeros((1, 3, height, width), dtype=np.float32)
        result[:, 2] = 1
        return result

    session = _Session({"semantic": semantic})
    result = run_model_inference(
        np.zeros((40, 48, 3), dtype=np.uint8),
        None,
        pack,
        session=session,
    )
    assert np.all(result.semantic_labels == 2)
    with pytest.raises(InterruptedError, match="cancelled"):
        run_model_inference(
            np.zeros((32, 32, 3), dtype=np.uint8),
            None,
            pack,
            session=session,
            cancelled=lambda: True,
        )


def test_model_pack_detects_corrupt_weight_file(tmp_path):
    pack = _pack(tmp_path, [{"name": "semantic", "kind": "semantic"}])
    (tmp_path / "model.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        ModelPack.open(pack.root)


def test_model_inference_provenance_roundtrips(tmp_path):
    source = tmp_path / "source.png"
    pack = _pack(tmp_path / "pack", [{"name": "semantic", "kind": "semantic"}])
    recipe = pack.default_recipe()
    run = ModelInferenceRun(recipe=recipe, review_status="pending", summary={"macro_dice": 0.91})
    manifest = ProjectManifest(
        name="model provenance",
        source_path=str(source),
        source_sha256="0" * 64,
        image_width=10,
        image_height=10,
        model_inference_recipes=[recipe],
        model_inference_runs=[run],
    )
    project = save_project(tmp_path / "model.gstproj", manifest)
    loaded, _ = load_project(project)
    assert loaded.schema_version == 4
    assert loaded.model_inference_runs[0].recipe.model_sha256 == pack.model_sha256
