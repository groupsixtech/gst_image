"""Accuracy metrics for synthetic and expert-approved segmentation masks."""

from __future__ import annotations

import math

import numpy as np


def dice_score(predicted: np.ndarray, expected: np.ndarray) -> float:
    predicted_mask = np.asarray(predicted, dtype=bool)
    expected_mask = np.asarray(expected, dtype=bool)
    if predicted_mask.shape != expected_mask.shape:
        raise ValueError("Compared masks must have the same shape")
    denominator = int(np.count_nonzero(predicted_mask)) + int(
        np.count_nonzero(expected_mask)
    )
    if denominator == 0:
        return 1.0
    intersection = int(np.count_nonzero(predicted_mask & expected_mask))
    return 2 * intersection / denominator


def _instance_matches(
    predicted: np.ndarray, expected: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int, float]], int, int]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    predicted = np.asarray(predicted, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.int64)
    if predicted.shape != expected.shape or predicted.ndim != 2:
        raise ValueError("Instance label images must be same-shaped two-dimensional arrays")
    predicted_ids = np.unique(predicted[predicted > 0])
    expected_ids = np.unique(expected[expected > 0])
    predicted_areas = dict(zip(*np.unique(predicted[predicted > 0], return_counts=True), strict=True))
    expected_areas = dict(zip(*np.unique(expected[expected > 0], return_counts=True), strict=True))
    overlap = (predicted > 0) & (expected > 0)
    candidates: list[tuple[float, int, int]] = []
    if np.any(overlap):
        pairs, intersections = np.unique(
            np.column_stack((predicted[overlap], expected[overlap])),
            axis=0,
            return_counts=True,
        )
        for (predicted_id, expected_id), intersection in zip(
            pairs, intersections, strict=True
        ):
            union = (
                predicted_areas[int(predicted_id)]
                + expected_areas[int(expected_id)]
                - int(intersection)
            )
            iou = int(intersection) / union
            if iou >= iou_threshold:
                candidates.append((iou, int(predicted_id), int(expected_id)))
    matched_predicted: set[int] = set()
    matched_expected: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, predicted_id, expected_id in sorted(candidates, reverse=True):
        if predicted_id in matched_predicted or expected_id in matched_expected:
            continue
        matched_predicted.add(predicted_id)
        matched_expected.add(expected_id)
        matches.append((predicted_id, expected_id, iou))
    return matches, len(predicted_ids), len(expected_ids)


def particle_instance_f1(
    predicted: np.ndarray, expected: np.ndarray, iou_threshold: float = 0.5
) -> float:
    matches, predicted_count, expected_count = _instance_matches(
        predicted, expected, iou_threshold
    )
    denominator = predicted_count + expected_count
    return 1.0 if denominator == 0 else 2 * len(matches) / denominator


def evaluate_particle_segmentation(
    predicted: np.ndarray,
    expected: np.ndarray,
    analysis_mask: np.ndarray | None = None,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return the metrics used by the particle-segmentation acceptance gates."""
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)
    if predicted.shape != expected.shape:
        raise ValueError("Compared label images must have the same shape")
    domain = (
        np.ones(predicted.shape, dtype=bool)
        if analysis_mask is None
        else np.asarray(analysis_mask, dtype=bool)
    )
    if domain.shape != predicted.shape or not np.any(domain):
        raise ValueError("Analysis mask must match and contain at least one pixel")
    predicted_fraction = np.count_nonzero((predicted > 0) & domain) / np.count_nonzero(domain)
    expected_fraction = np.count_nonzero((expected > 0) & domain) / np.count_nonzero(domain)
    matches, predicted_count, expected_count = _instance_matches(
        predicted * domain, expected * domain, iou_threshold
    )
    predicted_areas = np.bincount((predicted * domain).astype(np.int64).ravel())
    expected_areas = np.bincount((expected * domain).astype(np.int64).ravel())
    radius_errors = [
        abs(
            math.sqrt(predicted_areas[predicted_id] / math.pi)
            - math.sqrt(expected_areas[expected_id] / math.pi)
        )
        / math.sqrt(expected_areas[expected_id] / math.pi)
        for predicted_id, expected_id, _ in matches
        if expected_areas[expected_id] > 0
    ]
    denominator = predicted_count + expected_count
    return {
        "area_fraction_error_percentage_points": 100
        * abs(predicted_fraction - expected_fraction),
        "instance_f1_at_iou": 1.0 if denominator == 0 else 2 * len(matches) / denominator,
        "median_equivalent_radius_relative_error": (
            float(np.median(radius_errors)) if radius_errors else float("nan")
        ),
        "matched_instances": len(matches),
        "predicted_instances": predicted_count,
        "expected_instances": expected_count,
    }
