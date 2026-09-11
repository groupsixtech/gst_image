"""ROI rasterization and specimen-domain masking."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from scipy import ndimage as ndi

from gst_image.models import ROI, ROIKind


def rasterize_roi(shape: tuple[int, int], roi: ROI) -> np.ndarray:
    height, width = shape
    points = [(p.x, p.y) for p in roi.points]
    mask = np.zeros((height, width), dtype=np.uint8)
    if roi.shape == "rectangle":
        (x1, y1), (x2, y2) = points[:2]
        left, right = sorted((round(x1), round(x2)))
        top, bottom = sorted((round(y1), round(y2)))
        cv2.rectangle(
            mask,
            (max(0, left), max(0, top)),
            (min(width - 1, right), min(height - 1, bottom)),
            1,
            thickness=cv2.FILLED,
        )
    else:
        polygon = np.asarray(points, dtype=np.float64)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def build_analysis_mask(
    shape: tuple[int, int],
    rois: Iterable[ROI] = (),
    specimen_mask: np.ndarray | None = None,
    include_ids: set[str] | None = None,
) -> np.ndarray:
    """Apply valid specimen ∩ inclusion scopes − exclusions."""
    rois = list(rois)
    domain = (
        np.ones(shape, dtype=bool)
        if specimen_mask is None
        else np.asarray(specimen_mask, dtype=bool).copy()
    )
    inclusions = [
        roi
        for roi in rois
        if roi.kind in {ROIKind.INCLUDE, ROIKind.ANALYSIS_BOX}
        and (include_ids is None or roi.id in include_ids)
    ]
    if inclusions:
        included = np.zeros(shape, dtype=bool)
        for roi in inclusions:
            included |= rasterize_roi(shape, roi)
        domain &= included
    for roi in rois:
        if roi.kind == ROIKind.EXCLUDE:
            domain &= ~rasterize_roi(shape, roi)
    return domain


def suggest_specimen_mask(gray: np.ndarray, black_threshold: int = 5) -> np.ndarray:
    """Suggest a domain mask while rejecting black stitch canvas and small annotations."""
    if gray.ndim != 2:
        raise ValueError("Specimen mask requires a grayscale image")
    valid = gray > black_threshold
    if np.mean(valid) > 0.995:
        return np.ones_like(valid)
    radius = max(2, round(min(gray.shape) * 0.002))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    valid = ndi.binary_fill_holes(valid)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return valid
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    minimum = max(64, int(largest * 0.02))
    keep = np.zeros_like(valid)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            keep[labels == label] = True
    return keep
