"""Public analysis API."""

from .calibration import create_measurement
from .cellpose_inference import run_cellpose_inference
from .fractions import compute_area_fractions, compute_project_fractions
from .groups import (
    assign_particle_groups,
    evaluate_particle_grouping,
    generate_particle_groups,
    particle_group_statistics,
    particle_group_summary,
    validate_particle_groups,
)
from .masks import build_analysis_mask, rasterize_roi, suggest_specimen_mask
from .model_inference import ModelPack, ModelPackManifest, run_model_inference
from .particles import measure_particles, segment_particles
from .regions import classify_regions

__all__ = [
    "ModelPack",
    "ModelPackManifest",
    "assign_particle_groups",
    "build_analysis_mask",
    "classify_regions",
    "compute_area_fractions",
    "compute_project_fractions",
    "create_measurement",
    "evaluate_particle_grouping",
    "generate_particle_groups",
    "measure_particles",
    "particle_group_statistics",
    "particle_group_summary",
    "rasterize_roi",
    "run_cellpose_inference",
    "run_model_inference",
    "segment_particles",
    "suggest_specimen_mask",
    "validate_particle_groups",
]
