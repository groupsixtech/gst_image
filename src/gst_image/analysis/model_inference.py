"""Optional, offline ONNX inference for reviewed micrograph model packs.

The module deliberately imports ONNX Runtime only when a model is opened.  This
keeps the threshold and assisted-classification workflows usable without the ML
extra installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from gst_image.analysis.particles import measure_particles
from gst_image.models import Calibration, ModelInferenceRecipe, ParticleRecord

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


class _SessionInput(Protocol):
    name: str


class InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class ModelInputSpec(BaseModel):
    channel_order: Literal["RGB", "BGR", "GRAY"] = "RGB"
    mean: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    scale: float = Field(default=255.0, gt=0)

    @model_validator(mode="after")
    def validate_channels(self) -> ModelInputSpec:
        expected = 1 if self.channel_order == "GRAY" else 3
        if len(self.mean) != expected or len(self.std) != expected:
            raise ValueError(f"{self.channel_order} input requires {expected} mean/std values")
        if any(value <= 0 for value in self.std):
            raise ValueError("Input normalization standard deviations must be positive")
        return self


class ModelTilingSpec(BaseModel):
    tile_size_px: int = Field(default=512, ge=32)
    overlap_px: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ModelTilingSpec:
        if self.overlap_px >= self.tile_size_px:
            raise ValueError("Tile overlap must be smaller than tile size")
        return self


class ModelOutputSpec(BaseModel):
    name: str
    kind: Literal["semantic", "particle_foreground", "particle_boundary"]
    activation: Literal["softmax", "sigmoid", "none"] = "none"


class ModelPackManifest(BaseModel):
    """The stable JSON contract for a separately installed model pack."""

    format_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    architecture: Literal["unet_2d"] = "unet_2d"
    compatible_app_version: str = Field(default="0.1.0", min_length=1)
    model_file: str = "model.onnx"
    model_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    input: ModelInputSpec = Field(default_factory=ModelInputSpec)
    tiling: ModelTilingSpec = Field(default_factory=ModelTilingSpec)
    outputs: list[ModelOutputSpec]
    semantic_classes: dict[int, str] = Field(default_factory=dict)
    training_data_summary: str = ""
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    license: str = ""
    attribution: str = ""

    @model_validator(mode="after")
    def validate_outputs(self) -> ModelPackManifest:
        kinds = [output.kind for output in self.outputs]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Each output kind may occur only once")
        if "semantic" in kinds and not self.semantic_classes:
            raise ValueError("A semantic output requires semantic_classes")
        if any(value < 0 for value in self.semantic_classes):
            raise ValueError("Semantic class values must be non-negative")
        return self

    @property
    def output_by_kind(self) -> dict[str, ModelOutputSpec]:
        return {output.kind: output for output in self.outputs}


@dataclass(slots=True)
class ModelPack:
    root: Path
    manifest: ModelPackManifest
    model_path: Path
    model_sha256: str

    @classmethod
    def open(cls, path: str | Path) -> ModelPack:
        root = Path(path)
        if root.is_file():
            root = root.parent
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model pack manifest is missing: {manifest_path}")
        try:
            manifest = ModelPackManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Model pack manifest is not valid JSON: {manifest_path}") from error
        model_path = root / manifest.model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights are missing: {model_path}")
        digest = _sha256_file(model_path)
        if manifest.model_sha256 and digest.lower() != manifest.model_sha256.lower():
            raise ValueError("Model weights do not match the SHA-256 recorded in manifest.json")
        return cls(root=root, manifest=manifest, model_path=model_path, model_sha256=digest)

    def default_recipe(self) -> ModelInferenceRecipe:
        return ModelInferenceRecipe(
            name=f"{self.manifest.model_id} {self.manifest.model_version}",
            model_id=self.manifest.model_id,
            model_version=self.manifest.model_version,
            model_sha256=self.model_sha256,
        )


@dataclass(slots=True)
class ModelInferenceResult:
    semantic_labels: np.ndarray | None
    semantic_confidence: np.ndarray | None
    particle_probability: np.ndarray | None
    boundary_probability: np.ndarray | None
    particle_labels: np.ndarray | None
    particles: list[ParticleRecord]
    summary: dict[str, float | int | str]


def runtime_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def create_inference_session(pack: ModelPack) -> InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Model inference requires the optional ML extra: pip install gst-image[ml]"
        ) from error
    return ort.InferenceSession(str(pack.model_path), providers=["CPUExecutionProvider"])


def run_model_inference(
    image: np.ndarray,
    analysis_mask: np.ndarray | None,
    pack: ModelPack,
    recipe: ModelInferenceRecipe | None = None,
    calibration: Calibration | None = None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    session: InferenceSession | None = None,
) -> ModelInferenceResult:
    """Run a validated pack over native-resolution overlapping image tiles.

    Models consume ``NCHW`` float arrays. Semantic outputs are class logits or
    probabilities; particle heads are single-channel logits/probabilities.
    """
    pack = pack if isinstance(pack, ModelPack) else ModelPack.open(pack)
    recipe = recipe or pack.default_recipe()
    _validate_recipe(pack, recipe)
    source = np.asarray(image)
    if source.ndim not in (2, 3):
        raise ValueError("Model inference image must be grayscale or BGR/RGB")
    height, width = source.shape[:2]
    domain = (
        np.ones((height, width), dtype=bool)
        if analysis_mask is None
        else np.asarray(analysis_mask, dtype=bool)
    )
    if domain.shape != (height, width) or not np.any(domain):
        raise ValueError("Analysis mask must match the image and contain at least one pixel")
    active_session = session or create_inference_session(pack)
    inputs = active_session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Model pack ONNX graph must expose exactly one input")
    output_specs = pack.manifest.output_by_kind
    output_names = [spec.name for spec in pack.manifest.outputs]
    accumulators = _OutputAccumulators.create(output_specs, (height, width))
    tile_size = pack.manifest.tiling.tile_size_px
    starts_y = _tile_starts(height, tile_size, pack.manifest.tiling.overlap_px)
    starts_x = _tile_starts(width, tile_size, pack.manifest.tiling.overlap_px)
    total = len(starts_y) * len(starts_x)
    window = _tile_window(tile_size)
    for index, (top, left) in enumerate(
        ((top, left) for top in starts_y for left in starts_x), start=1
    ):
        _check_cancelled(cancelled)
        tile, valid_height, valid_width = _tile(source, top, left, tile_size)
        tensor = _prepare_input(tile, pack.manifest.input)
        raw_outputs = active_session.run(output_names, {inputs[0].name: tensor})
        if len(raw_outputs) != len(output_names):
            raise ValueError("Model returned a different number of outputs than the model pack declares")
        for spec, raw in zip(pack.manifest.outputs, raw_outputs, strict=True):
            values = _decode_output(raw, spec, tile_size)
            accumulators.add(
                spec.kind,
                values[:, :valid_height, :valid_width],
                top,
                left,
                window[:valid_height, :valid_width],
            )
        _report(progress, index / total * 0.8, f"Running model tile {index} of {total}")
    _check_cancelled(cancelled)
    outputs = accumulators.finish()
    semantic_labels, semantic_confidence = _semantic_result(
        outputs.get("semantic"),
        domain,
        recipe.semantic_confidence_threshold,
        probabilities=output_specs.get("semantic", ModelOutputSpec(name="", kind="semantic")).activation
        == "softmax",
    )
    particle_probability = _first_channel(outputs.get("particle_foreground"))
    boundary_probability = _first_channel(outputs.get("particle_boundary"))
    labels = None
    particles: list[ParticleRecord] = []
    if particle_probability is not None:
        _report(progress, 0.86, "Separating model particles")
        labels = _particle_instances(
            particle_probability,
            boundary_probability,
            domain,
            recipe,
        )
        particles = measure_particles(labels, calibration, domain)
    _check_cancelled(cancelled)
    summary: dict[str, float | int | str] = {
        "model_id": pack.manifest.model_id,
        "model_version": pack.manifest.model_version,
        "model_sha256": pack.model_sha256,
        "analyzed_pixels": int(np.count_nonzero(domain)),
        "analysis_resolution": "original",
        "runtime_version": runtime_version(),
    }
    if semantic_labels is not None:
        for value, name in pack.manifest.semantic_classes.items():
            summary[f"class_pixels_{name}"] = int(np.count_nonzero(semantic_labels == value))
    if labels is not None:
        segmented = int(np.count_nonzero(labels))
        summary.update(
            particle_count=len(particles),
            segmented_pixels=segmented,
            area_fraction=segmented / int(np.count_nonzero(domain)),
            estimated_volume_fraction_percent=100 * segmented / int(np.count_nonzero(domain)),
        )
    _report(progress, 1.0, "Model inference complete")
    return ModelInferenceResult(
        semantic_labels=semantic_labels,
        semantic_confidence=semantic_confidence,
        particle_probability=particle_probability,
        boundary_probability=boundary_probability,
        particle_labels=labels,
        particles=particles,
        summary=summary,
    )


@dataclass(slots=True)
class _OutputAccumulators:
    values: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]

    @classmethod
    def create(
        cls, specs: dict[str, ModelOutputSpec], shape: tuple[int, int]
    ) -> _OutputAccumulators:
        height, width = shape
        channels = {"semantic": None, "particle_foreground": 1, "particle_boundary": 1}
        return cls(
            values={
                kind: np.zeros((count or 0, height, width), dtype=np.float32)
                for kind, count in channels.items()
                if kind in specs
            },
            weights={
                kind: np.zeros((height, width), dtype=np.float32) for kind in specs
            },
        )

    def add(
        self, kind: str, values: np.ndarray, top: int, left: int, window: np.ndarray
    ) -> None:
        if self.values[kind].shape[0] == 0:
            self.values[kind] = np.zeros(
                (values.shape[0], *self.weights[kind].shape), dtype=np.float32
            )
        bottom, right = top + values.shape[1], left + values.shape[2]
        self.values[kind][:, top:bottom, left:right] += values * window
        self.weights[kind][top:bottom, left:right] += window

    def finish(self) -> dict[str, np.ndarray]:
        return {
            kind: values / np.maximum(self.weights[kind][None, :, :], 1e-6)
            for kind, values in self.values.items()
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_recipe(pack: ModelPack, recipe: ModelInferenceRecipe) -> None:
    from gst_image import __version__

    if pack.manifest.compatible_app_version != __version__:
        raise ValueError(
            f"Model pack requires GST Image {pack.manifest.compatible_app_version}; "
            f"this installation is {__version__}"
        )
    if recipe.model_id != pack.manifest.model_id or recipe.model_version != pack.manifest.model_version:
        raise ValueError("Inference recipe does not match this model pack")
    if recipe.model_sha256.lower() != pack.model_sha256.lower():
        raise ValueError("Inference recipe does not match this model pack's weights")


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _tile(image: np.ndarray, top: int, left: int, tile_size: int) -> tuple[np.ndarray, int, int]:
    bottom, right = min(image.shape[0], top + tile_size), min(image.shape[1], left + tile_size)
    values = image[top:bottom, left:right]
    valid_height, valid_width = values.shape[:2]
    pad_bottom, pad_right = tile_size - valid_height, tile_size - valid_width
    if pad_bottom or pad_right:
        border = cv2.BORDER_REFLECT_101 if min(values.shape[:2]) > 1 else cv2.BORDER_REPLICATE
        values = cv2.copyMakeBorder(values, 0, pad_bottom, 0, pad_right, border)
    return values, valid_height, valid_width


def _prepare_input(tile: np.ndarray, spec: ModelInputSpec) -> np.ndarray:
    if spec.channel_order == "GRAY":
        values = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
        values = values[:, :, None]
    else:
        if tile.ndim == 2:
            values = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        else:
            values = tile[:, :, :3]
        if spec.channel_order == "RGB":
            values = values[:, :, ::-1]
    normalized = values.astype(np.float32) / spec.scale
    normalized = (normalized - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
        spec.std, dtype=np.float32
    )
    return np.moveaxis(normalized, -1, 0)[None, :, :, :].astype(np.float32, copy=False)


def _decode_output(raw: np.ndarray, spec: ModelOutputSpec, tile_size: int) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    elif values.ndim == 3:
        pass
    elif values.ndim == 2:
        values = values[None, :, :]
    else:
        raise ValueError(f"Model output {spec.name!r} must have shape NCHW or CHW")
    if values.shape[-2:] != (tile_size, tile_size):
        raise ValueError(
            f"Model output {spec.name!r} has shape {values.shape[-2:]}; expected {tile_size} x {tile_size}"
        )
    if spec.kind != "semantic" and values.shape[0] != 1:
        raise ValueError(f"Model output {spec.name!r} must have one channel")
    if spec.activation == "sigmoid":
        values = 1 / (1 + np.exp(-values))
    elif spec.activation == "softmax":
        values = _softmax(values)
    return values


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


def _tile_window(size: int) -> np.ndarray:
    axis = np.hanning(size).astype(np.float32) + 0.1
    return np.outer(axis, axis).astype(np.float32)


def _semantic_result(
    values: np.ndarray | None,
    domain: np.ndarray,
    confidence_threshold: float,
    *,
    probabilities: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if values is None:
        return None, None
    class_probabilities = values if probabilities else _softmax(values)
    confidence = class_probabilities.max(axis=0)
    labels = class_probabilities.argmax(axis=0).astype(np.int32)
    labels[(confidence < confidence_threshold) | ~domain] = 0
    confidence[~domain] = 0
    return labels, confidence.astype(np.float32)


def _first_channel(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return values[0].astype(np.float32, copy=False)


def _particle_instances(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray | None,
    domain: np.ndarray,
    recipe: ModelInferenceRecipe,
) -> np.ndarray:
    foreground = (foreground_probability >= recipe.particle_confidence_threshold) & domain
    if not np.any(foreground):
        return np.zeros(foreground.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(foreground)
    coordinates = peak_local_max(
        distance,
        labels=foreground,
        min_distance=recipe.watershed_min_distance_px,
        exclude_border=False,
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if coordinates.size:
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        markers, _ = ndi.label(markers > 0)
    else:
        markers, _ = ndi.label(foreground)
    if boundary_probability is None:
        elevation = -distance
    else:
        # Confident boundary pixels become watershed ridges without removing
        # them from the foreground mask (and therefore without eroding measured area).
        elevation = np.where(
            boundary_probability >= recipe.boundary_confidence_threshold,
            1.0,
            boundary_probability,
        )
    labels = watershed(elevation, markers, mask=foreground).astype(np.int32)
    return _filter_labels(labels, recipe.min_particle_area_px, recipe.max_particle_area_px)


def _filter_labels(labels: np.ndarray, minimum: int, maximum: int | None) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=np.int32)
    next_label = 1
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    for value, count in zip(values, counts, strict=True):
        if count < minimum or (maximum is not None and count > maximum):
            continue
        result[labels == value] = next_label
        next_label += 1
    return result


def _report(progress: Progress | None, value: float, message: str) -> None:
    if progress:
        progress(value, message)


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
