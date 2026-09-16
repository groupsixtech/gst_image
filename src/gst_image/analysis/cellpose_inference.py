"""Optional, local Cellpose-SAM v2 instance inference.

Cellpose is imported only when this module is asked to run a model.  The stock
weights are never bundled: callers must explicitly permit Cellpose's first-run
download when the local cache is empty.

Evaluation mirrors the upstream Cellpose application: the stock ``cpsam_v2``
backbone is called with Cellpose's own defaults for everything the recipe does
not name, so a region analysed here matches the same region analysed in the
Cellpose GUI.  Unlike the GUI, inference is confined to the caller's regions
(the project's Analysis boxes) instead of the whole overview image.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from collections.abc import Callable, Iterable, Sequence
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
Bounds = tuple[int, int, int, int]

STOCK_MODEL_ID = "cpsam_v2"
STOCK_MODEL_LICENSE = "CC-BY-NC"
STOCK_MODEL_ATTRIBUTION = "Cellpose-SAM; Pachitariu, Rariden, and Stringer (2025)"
# Cellpose rescales by 30 / diameter, so an unset diameter is the app's native-scale default.
CELLPOSE_NATIVE_DIAMETER_PX = 30.0

_EVAL_SPAN = (0.1, 0.84)


class CellposeModel(Protocol):
    pretrained_model: str | Path

    def eval(self, image: np.ndarray, **kwargs: Any) -> tuple[Any, ...]: ...


@dataclass(slots=True)
class CellposeInferenceResult:
    labels: np.ndarray
    particles: list[ParticleRecord]
    summary: dict[str, float | int | str]
    source_bounds_px: Bounds
    region_bounds_px: list[Bounds]
    cellpose_version: str
    torch_version: str
    model_sha256: str
    model_cache_path: str


def cellpose_cache_path() -> Path:
    """Return the expected cache path without creating it or downloading weights."""
    return _cellpose_model_dir(_cellpose_models()) / STOCK_MODEL_ID


def run_cellpose_inference(
    image: np.ndarray,
    analysis_mask: np.ndarray | None,
    recipe: CellposeInferenceRecipe,
    calibration: Calibration | None = None,
    *,
    regions: Iterable[Bounds] | None = None,
    allow_model_download: bool = False,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    model: CellposeModel | None = None,
) -> CellposeInferenceResult:
    """Run local stock Cellpose-SAM per region, restore full coordinates, and measure instances."""
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
    if recipe.device == "gpu":
        _require_cuda_gpu()
    boxes = _resolve_regions(regions, domain, (height, width))
    _report(progress, 0.03, "Preparing Cellpose-SAM input")
    active_model, cache_path = _get_model(
        allow_model_download, model, use_gpu=recipe.device == "gpu"
    )
    eval_kwargs = _eval_kwargs(recipe)
    labels = np.zeros((height, width), dtype=np.int32)
    offset = 0
    started = time.perf_counter()
    for index, (left, top, right, bottom) in enumerate(boxes):
        _check_cancelled(cancelled)
        _report(
            progress,
            _span_fraction(index, len(boxes)),
            f"Running local Cellpose-SAM v2 on analysis region {index + 1} of {len(boxes)}",
        )
        crop = _cellpose_input(source[top:bottom, left:right])
        response = active_model.eval(crop, **eval_kwargs)
        # Cellpose does not offer a cancellation callback for eval; never save a late result.
        _check_cancelled(cancelled)
        local = _labels_from_response(response, crop.shape[:2])
        # Clip per region so post-processing costs scale with the boxes, not the overview.
        local = _clip_and_split_labels(local, domain[top:bottom, left:right])
        peak = int(local.max())
        if peak == 0:
            continue
        window = labels[top:bottom, left:right]
        labels[top:bottom, left:right] = np.where(local > 0, local + offset, window)
        offset += peak
    elapsed_seconds = time.perf_counter() - started
    _report(progress, _EVAL_SPAN[1], "Measuring Cellpose instances")
    particles = measure_particles(labels, calibration, domain)
    _check_cancelled(cancelled)
    segmented_pixels = int(np.count_nonzero(labels))
    analyzed_pixels = int(np.count_nonzero(_region_domain(domain, boxes)))
    resolved_path = _resolved_model_path(active_model, cache_path)
    summary: dict[str, float | int | str] = {
        "model_id": STOCK_MODEL_ID,
        "modality": recipe.modality,
        "device": recipe.device,
        "analysis_regions": len(boxes),
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
        source_bounds_px=_union(boxes),
        region_bounds_px=list(boxes),
        cellpose_version=_distribution_version("cellpose"),
        torch_version=_torch_version(),
        model_sha256=_sha256_file(resolved_path) if resolved_path.is_file() else "not-recorded",
        model_cache_path=str(resolved_path),
    )


def _eval_kwargs(recipe: CellposeInferenceRecipe) -> dict[str, Any]:
    """Name only what the recipe controls so Cellpose's own defaults apply elsewhere."""
    return {
        "diameter": recipe.diameter_px,
        "flow_threshold": recipe.flow_threshold,
        "cellprob_threshold": recipe.cellprob_threshold,
        "min_size": recipe.min_size_px,
        "tile_overlap": recipe.tile_overlap,
    }


def _cellpose_input(crop: np.ndarray) -> np.ndarray:
    """Hand Cellpose RGB, the channel order the Cellpose app reads images in."""
    return np.ascontiguousarray(crop[:, :, :3][:, :, ::-1])


def _cellpose_models() -> Any:
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError(
            "Cellpose inference requires the optional extra: pip install gst-image[cellpose]"
        ) from error
    return models


def _cellpose_model_dir(models: Any) -> Path:
    """Resolve the weight cache across Cellpose releases that renamed the constant."""
    for attribute in ("MODEL_DIR", "MODELS_DIR"):
        value = getattr(models, attribute, None)
        if value:
            return Path(value)
    return Path.home() / ".cellpose" / "models"


def _get_model(
    allow_model_download: bool, model: CellposeModel | None, *, use_gpu: bool
) -> tuple[CellposeModel, Path]:
    if model is not None:
        path = _resolved_model_path(model, Path("not-recorded"))
        return model, path
    models = _cellpose_models()
    cache_path = _cellpose_model_dir(models) / STOCK_MODEL_ID
    if not cache_path.is_file() and not allow_model_download:
        raise FileNotFoundError(
            "Stock Cellpose-SAM v2 weights are not cached. Explicitly allow the first-run model "
            "download; no image data will be uploaded."
        )
    try:
        return models.CellposeModel(pretrained_model=STOCK_MODEL_ID, gpu=use_gpu), cache_path
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


def _resolve_regions(
    regions: Iterable[Bounds] | None, domain: np.ndarray, shape: tuple[int, int]
) -> list[Bounds]:
    """Clip requested regions to the domain, dropping empties and merging overlaps."""
    if regions is None:
        return [_domain_bounds(domain)]
    height, width = shape
    clipped: list[Bounds] = []
    for left, top, right, bottom in regions:
        x0, x1 = sorted((int(left), int(right)))
        y0, y1 = sorted((int(top), int(bottom)))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        window = domain[y0:y1, x0:x1]
        if not np.any(window):
            continue
        inner_left, inner_top, inner_right, inner_bottom = _domain_bounds(window)
        clipped.append((x0 + inner_left, y0 + inner_top, x0 + inner_right, y0 + inner_bottom))
    if not clipped:
        raise ValueError("No analysis region overlaps the analysis domain")
    return _merge_regions(clipped)


def _merge_regions(boxes: list[Bounds]) -> list[Bounds]:
    """Fuse overlapping regions so no instance is segmented twice along a shared edge."""
    merged = list(boxes)
    fused = True
    while fused:
        fused = False
        for first in range(len(merged)):
            for second in range(first + 1, len(merged)):
                if _intersects(merged[first], merged[second]):
                    merged[first] = _union((merged[first], merged[second]))
                    del merged[second]
                    fused = True
                    break
            if fused:
                break
    return sorted(merged, key=lambda box: (box[1], box[0]))


def _intersects(first: Bounds, second: Bounds) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def _union(boxes: Sequence[Bounds]) -> Bounds:
    lefts, tops, rights, bottoms = zip(*boxes, strict=True)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _region_domain(domain: np.ndarray, boxes: Sequence[Bounds]) -> np.ndarray:
    """Restrict the domain to the pixels Cellpose was actually asked to look at."""
    covered = np.zeros_like(domain, dtype=bool)
    for left, top, right, bottom in boxes:
        covered[top:bottom, left:right] = True
    return domain & covered


def _domain_bounds(domain: np.ndarray) -> Bounds:
    ys, xs = np.nonzero(domain)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _span_fraction(index: int, total: int) -> float:
    start, end = _EVAL_SPAN
    return start + (end - start) * index / max(1, total)


def _clip_and_split_labels(labels: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Clip labels and retain disconnected fragments as distinct instances.

    Connectivity is resolved per label inside that label's bounding box, so the total
    work tracks the image area rather than the instance count.  Labelling the whole
    foreground in one pass would be wrong here: two fragments of one instance can be
    bridged by a neighbouring instance's pixels and would wrongly fuse.
    """
    clipped = np.where(domain, labels, 0).astype(np.int32, copy=False)
    result = np.zeros_like(clipped, dtype=np.int32)
    next_label = 1
    for value, window in enumerate(ndi.find_objects(clipped), start=1):
        if window is None:
            continue
        piece = clipped[window] == value
        count, pieces = cv2.connectedComponents(
            piece.view(np.uint8), connectivity=4, ltype=cv2.CV_32S
        )
        target = result[window]
        target[piece] = pieces[piece] + (next_label - 1)
        next_label += count - 1
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


def _require_cuda_gpu() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU Cellpose requires a CUDA-enabled PyTorch installation; install it before gst-image[cellpose]"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available to PyTorch. Install the CUDA wheel matching the NVIDIA driver, "
            "then verify torch.cuda.is_available() is True."
        )


def _report(progress: Progress | None, value: float, message: str) -> None:
    if progress:
        progress(value, message)


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
