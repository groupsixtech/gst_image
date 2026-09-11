"""Assisted multi-class region segmentation from analyst training strokes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import multiscale_basic_features
from skimage.filters import sobel
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier

from gst_image.models import RegionAnalysis, RegionClassifierRecipe

REGION_CLASSIFICATION_MAX_PIXELS = 4_500_000


def snap_region_boundaries(
    image: np.ndarray, labels: np.ndarray, band_px: int = 3
) -> np.ndarray:
    """Snap classifier boundaries to local image gradients within a narrow band."""
    if band_px <= 0:
        return labels
    gray = np.mean(image[:, :, :3], axis=2) if image.ndim == 3 else image
    boundary = ndi.maximum_filter(labels, size=3) != ndi.minimum_filter(labels, size=3)
    band = ndi.binary_dilation(boundary, iterations=band_px)
    markers = labels.astype(np.int32, copy=True)
    markers[band] = 0
    if len(np.unique(markers[markers > 0])) < 2:
        return labels
    refined = watershed(sobel(gray), markers=markers, mask=np.ones(labels.shape, dtype=bool))
    return refined.astype(labels.dtype, copy=False)


def classify_regions(
    preview: np.ndarray,
    training_strokes: np.ndarray,
    recipe: RegionClassifierRecipe | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RegionAnalysis:
    """Train a deterministic classifier from a positive-integer stroke label image."""
    recipe = recipe or RegionClassifierRecipe()
    if preview.shape[:2] != training_strokes.shape:
        raise ValueError("Training strokes must match the preview dimensions")
    class_ids = sorted(int(value) for value in np.unique(training_strokes) if value > 0)
    if len(class_ids) < 2:
        raise ValueError("Paint training strokes for at least two region classes")
    pixel_count = preview.shape[0] * preview.shape[1]
    if pixel_count > REGION_CLASSIFICATION_MAX_PIXELS:
        height, width = preview.shape[:2]
        raise ValueError(
            f"The region-classification overview is {width} x {height} pixels "
            f"({pixel_count / 1_000_000:.2f} MP), above the 4.5 MP limit. "
            "Reduce Overview resolution, click Show overview, and train again."
        )
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
    if progress:
        progress(0.05, "Computing region features")
    channel_axis = -1 if preview.ndim == 3 else None
    features = multiscale_basic_features(
        preview,
        intensity=recipe.intensity_features,
        edges=recipe.edge_features,
        texture=recipe.texture_features,
        sigma_min=recipe.sigma_min,
        sigma_max=recipe.sigma_max,
        channel_axis=channel_axis,
    )
    selected = training_strokes > 0
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
    if progress:
        progress(0.45, "Training region classifier")
    classifier = RandomForestClassifier(
        n_estimators=recipe.n_estimators,
        max_depth=recipe.max_depth,
        max_samples=recipe.max_samples,
        # The caller already runs this work off the UI thread. A single sklearn
        # worker also keeps replay behavior stable on restricted lab workstations.
        n_jobs=1,
        random_state=recipe.random_seed,
        class_weight="balanced_subsample",
    )
    classifier.fit(features[selected], training_strokes[selected])
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
    if progress:
        progress(0.75, "Classifying region pixels")
    labels = classifier.predict(features.reshape(-1, features.shape[-1])).reshape(
        training_strokes.shape
    )
    labels = snap_region_boundaries(preview, labels, recipe.edge_snap_band_px)
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")
    counts = {str(class_id): int(np.count_nonzero(labels == class_id)) for class_id in class_ids}
    summary: dict[str, Any] = {
        "class_pixels": counts,
        "training_pixels": int(np.count_nonzero(selected)),
        "random_seed": recipe.random_seed,
    }
    if progress:
        progress(1.0, "Region classification complete")
    return RegionAnalysis(labels.astype(np.int32), class_ids, summary)
