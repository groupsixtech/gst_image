"""Optional, local Cellpose-SAM v2 instance inference.

Cellpose is imported only when this module is asked to run a model.  The stock
weights are never bundled: callers must explicitly permit Cellpose's first-run
download when the local cache is empty.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from scipy import ndimage as ndi

from gst_image.analysis.particles import measure_particles
from gst_image.models import Calibration, CellposeInferenceRecipe, ParticleRecord

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

STOCK_MODEL_ID = "cpsam_v2"
STOCK_MODEL_LICENSE = "CC-BY-NC"
STOCK_MODEL_ATTRIBUTION = "Cellpose-SAM; Pachitariu, Rariden, and Stringer (2025)"


class CellposeModel(Protocol):
    pretrained_model: str | Path

    def eval(self, image: np.ndarray, **kwargs: Any) -> tuple[Any, ...]: ...


@dataclass(slots=True)
class CellposeInferenceResult:
    labels: np.ndarray
    particles: list[ParticleRecord]
    summary: dict[str, float | int | str]
    source_bounds_px: tuple[int, int, int, int]
    cellpose_version: str
    torch_version: str
    model_sha256: str
    model_cache_path: str


def cellpose_cache_path() -> Path:
    """Return the expected cache path without creating it or downloading weights."""
    models = _cellpose_models()
    return Path(models.MODELS_DIR) / STOCK_MODEL_ID


def run_cellpose_inference(
    image: np.ndarray,
    analysis_mask: np.ndarray | None,
    recipe: CellposeInferenceRecipe,
    calibration: Calibration | None = None,
    *,
    allow_model_download: bool = False,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    model: CellposeModel | None = None,
) -> CellposeInferenceResult:
    """Run local stock Cellpose-SAM, restore full coordinates, and measure instances."""
    if not recipe.noncommercial_license_accepted:
        raise ValueError("Cellpose-SAM requires acknowledgement of its non-commercial licence")
    _check_cancelled(cancelled)
    source = np.asarray(image)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("Cellpose inference requires a grayscale or three-channel image")
    height, width = source.shape[:2]
    domain = (
        np.ones((height, width), dtype=bool)
        if analysis_mask is None
        else np.asarray(analysis_mask, dtype=bool)
    )
    if domain.shape != (height, width) or not np.any(domain):
        raise ValueError("Analysis mask must match the image and contain at least one pixel")
    top, left, bottom, right = _domain_bounds(domain)
    _report(progress, 0.03, "Preparing Cellpose-SAM input")
    active_model, cache_path = _get_model(allow_model_download, model)
    crop = source[top:bottom, left:right, :3]
    _check_cancelled(cancelled)
    _report(progress, 0.1, "Running local Cellpose-SAM v2")
    started = time.perf_counter()
    response = active_model.eval(
        crop,
        diameter=recipe.diameter_px,
        flow_threshold=recipe.flow_threshold,
        cellprob_threshold=recipe.cellprob_threshold,
        min_size=recipe.min_size_px,
        tile_overlap=recipe.tile_overlap,
    )
    elapsed_seconds = time.perf_counter() - started
    # Cellpose does not offer a cancellation callback for eval; never save a late result.
    _check_cancelled(cancelled)
    local_labels = _labels_from_response(response, crop.shape[:2])
    labels = np.zeros((height, width), dtype=np.int32)
    labels[top:bottom, left:right] = local_labels
    _report(progress, 0.84, "Clipping Cellpose instances to analysis domain")
    labels = _clip_and_split_labels(labels, domain)
    particles = measure_particles(labels, calibration, domain)
    _check_cancelled(cancelled)
    segmented_pixels = int(np.count_nonzero(labels))
    analyzed_pixels = int(np.count_nonzero(domain))
    resolved_path = _resolved_model_path(active_model, cache_path)
    summary: dict[str, float | int | str] = {
        "model_id": STOCK_MODEL_ID,
        "modality": recipe.modality,
        "analyzed_pixels": analyzed_pixels,
        "analysis_resolution": "original",
        "particle_count": len(particles),
        "segmented_pixels": segmented_pixels,
        "area_fraction": segmented_pixels / analyzed_pixels,
        "estimated_volume_fraction_percent": 100 * segmented_pixels / analyzed_pixels,
        "elapsed_seconds": elapsed_seconds,
    }
    _report(progress, 1.0, "Cellpose-SAM inference complete")
    return CellposeInferenceResult(
        labels=labels,
        particles=particles,
        summary=summary,
        source_bounds_px=(left, top, right, bottom),
        cellpose_version=_distribution_version("cellpose"),
        torch_version=_torch_version(),
        model_sha256=_sha256_file(resolved_path) if resolved_path.is_file() else "not-recorded",
        model_cache_path=str(resolved_path),
    )


def _cellpose_models() -> Any:
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError(
            "Cellpose inference requires the optional extra: pip install gst-image[cellpose]"
        ) from error
    return models


def _get_model(
    allow_model_download: bool, model: CellposeModel | None
) -> tuple[CellposeModel, Path]:
    if model is not None:
        path = _resolved_model_path(model, Path("not-recorded"))
        return model, path
    models = _cellpose_models()
    cache_path = Path(models.MODELS_DIR) / STOCK_MODEL_ID
    if not cache_path.is_file() and not allow_model_download:
        raise FileNotFoundError(
            "Stock Cellpose-SAM v2 weights are not cached. Explicitly allow the first-run model "
            "download; no image data will be uploaded."
        )
    try:
        return models.CellposeModel(pretrained_model=STOCK_MODEL_ID, gpu=False), cache_path
    except Exception as error:
        if not cache_path.is_file():
            raise RuntimeError(f"Unable to obtain local Cellpose-SAM v2 weights: {error}") from error
        raise


def _labels_from_response(response: tuple[Any, ...], expected_shape: tuple[int, int]) -> np.ndarray:
    if not response:
        raise ValueError("Cellpose returned no mask output")
    labels = np.asarray(response[0])
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.shape != expected_shape:
        raise ValueError(
            f"Cellpose returned labels of shape {labels.shape}, expected {expected_shape}"
        )
    return labels.astype(np.int32, copy=False)


def _domain_bounds(domain: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(domain)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _clip_and_split_labels(labels: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Clip labels and retain disconnected fragments as distinct instances."""
    clipped = np.where(domain, labels, 0).astype(np.int32, copy=False)
    result = np.zeros_like(clipped, dtype=np.int32)
    next_label = 1
    for value in np.unique(clipped[clipped > 0]):
        pieces, count = ndi.label(clipped == value)
        for piece in range(1, count + 1):
            result[pieces == piece] = next_label
            next_label += 1
    return result


def _resolved_model_path(model: CellposeModel, fallback: Path) -> Path:
    value = getattr(model, "pretrained_model", fallback)
    return Path(value) if value else fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _torch_version() -> str:
    try:
        import torch
    except ImportError:
        return "not-installed"
    return str(torch.__version__)


def _report(progress: Progress | None, value: float, message: str) -> None:
    if progress:
        progress(value, message)


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
