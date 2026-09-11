"""Channel selection, illumination flattening, and threshold implementations."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.restoration import rolling_ball

from gst_image.models import ParticlePolarity, SegmentationRecipe, ThresholdMethod


def to_gray(image: np.ndarray, channel: str = "gray") -> np.ndarray:
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a grayscale or three-channel image")
    key = channel.lower()
    if key == "gray":
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    if key in {"b", "blue"}:
        return image[:, :, 0]
    if key in {"g", "green"}:
        return image[:, :, 1]
    if key in {"r", "red"}:
        return image[:, :, 2]
    lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB)
    indexes = {"l": 0, "lab_l": 0, "a": 1, "lab_a": 1, "lab_b": 2}
    if key not in indexes:
        raise ValueError(f"Unsupported analysis channel: {channel}")
    return lab[:, :, indexes[key]]


def estimate_illumination_field(
    gray: np.ndarray,
    recipe: SegmentationRecipe,
    analysis_mask: np.ndarray | None = None,
    max_dimension: int = 768,
) -> np.ndarray:
    """Estimate a smooth full-size illumination field on a coarse image."""
    height, width = gray.shape
    scale = min(1.0, max_dimension / max(height, width))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    working = gray.copy()
    if analysis_mask is not None and np.any(analysis_mask):
        fill = int(np.median(gray[analysis_mask]))
        working[~analysis_mask] = fill
    small = cv2.resize(working, small_size, interpolation=cv2.INTER_AREA)
    radius = max(3, round(recipe.rolling_ball_radius_px * scale))
    if recipe.polarity == ParticlePolarity.DARK:
        inverse_background = rolling_ball(255 - small, radius=radius, workers=-1)
        small_field = 255.0 - inverse_background.astype(np.float32)
    else:
        small_field = rolling_ball(small, radius=radius, workers=-1).astype(np.float32)
    field = cv2.resize(small_field, (width, height), interpolation=cv2.INTER_CUBIC)
    positive = field[field > 1]
    floor = float(np.percentile(positive, 2)) if positive.size else 1.0
    return np.maximum(field, max(1.0, floor)).astype(np.float32)


def flatten_illumination(gray: np.ndarray, field: np.ndarray) -> np.ndarray:
    target = float(np.median(field[field > 0]))
    corrected = gray.astype(np.float32) * target / np.maximum(field, 1.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def threshold_array(
    gray: np.ndarray,
    recipe: SegmentationRecipe,
    global_otsu_threshold: float | None = None,
) -> np.ndarray:
    if recipe.gaussian_blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), recipe.gaussian_blur_sigma)
    dark = recipe.polarity == ParticlePolarity.DARK
    if recipe.threshold_method == ThresholdMethod.MANUAL:
        return (gray >= recipe.manual_threshold_low) & (
            gray <= recipe.manual_threshold_high
        )
    if recipe.threshold_method == ThresholdMethod.SAUVOLA:
        threshold = threshold_sauvola(
            gray,
            window_size=recipe.sauvola_window_px,
            k=recipe.sauvola_k,
            r=128,
        )
        return gray < threshold if dark else gray > threshold
    if recipe.threshold_method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
        mode = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        return (
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                mode,
                recipe.gaussian_block_px,
                recipe.gaussian_c,
            )
            > 0
        )
    threshold = global_otsu_threshold
    if threshold is None:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return gray < threshold if dark else gray > threshold
