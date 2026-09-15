import cv2
import numpy as np
import pytest

from gst_image.evaluation import (
    dice_score,
    evaluate_model_release,
    evaluate_particle_segmentation,
    particle_instance_f1,
)


def test_particle_acceptance_metrics_for_small_boundary_difference():
    expected = np.zeros((150, 160), np.int32)
    predicted = np.zeros_like(expected)
    cv2.circle(expected, (35, 50), 12, 1, cv2.FILLED)
    cv2.circle(expected, (85, 50), 16, 2, cv2.FILLED)
    cv2.circle(predicted, (35, 50), 11, 1, cv2.FILLED)
    cv2.circle(predicted, (85, 50), 15, 2, cv2.FILLED)

    metrics = evaluate_particle_segmentation(predicted, expected)
    assert metrics["instance_f1_at_iou"] == 1
    assert metrics["area_fraction_error_percentage_points"] < 1
    assert metrics["median_equivalent_radius_relative_error"] < 0.1
    assert particle_instance_f1(predicted, expected) == 1
    assert dice_score(predicted > 0, expected > 0) > 0.9


def test_evaluation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        dice_score(np.zeros((2, 2)), np.zeros((3, 3)))


def test_model_release_metrics_require_expert_review_after_passing_gates():
    semantic = np.array([[0, 1], [2, 2]], dtype=np.int32)
    particles = np.array([[0, 1], [0, 1]], dtype=np.int32)
    release = evaluate_model_release(semantic, semantic, particles, particles)

    assert release["metrics_pass"]
    assert release["expert_review_required"]
    assert release["ready_for_expert_review"]
