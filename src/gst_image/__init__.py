"""Framework-neutral weld microstructure image analysis."""

__version__ = "0.1.0"

from .analysis import (
    assign_particle_groups,
    classify_regions,
    compute_area_fractions,
    compute_project_fractions,
    measure_particles,
    run_model_inference,
    segment_particles,
)
from .evaluation import (
    ModelAcceptanceGates,
    dice_score,
    evaluate_model_release,
    evaluate_particle_segmentation,
    macro_dice_score,
    particle_instance_f1,
)
from .export import export_analysis
from .models import (
    ROI,
    AnalysisRun,
    Calibration,
    ClassDefinition,
    Measurement,
    ModelInferenceRecipe,
    ModelInferenceRun,
    ParticleCriteria,
    ParticleGroup,
    ParticleGrouping,
    ParticleRecord,
    ParticleSizeMetric,
    ProjectManifest,
    RegionClassifierRecipe,
    SegmentationLayer,
    SegmentationRecipe,
)
from .project import load_project, relink_source, save_project, validate_project

__all__ = [
    "ROI",
    "AnalysisRun",
    "Calibration",
    "ClassDefinition",
    "Measurement",
    "ModelAcceptanceGates",
    "ModelInferenceRecipe",
    "ModelInferenceRun",
    "ParticleCriteria",
    "ParticleGroup",
    "ParticleGrouping",
    "ParticleRecord",
    "ParticleSizeMetric",
    "ProjectManifest",
    "RegionClassifierRecipe",
    "SegmentationLayer",
    "SegmentationRecipe",
    "assign_particle_groups",
    "classify_regions",
    "compute_area_fractions",
    "compute_project_fractions",
    "dice_score",
    "evaluate_model_release",
    "evaluate_particle_segmentation",
    "export_analysis",
    "load_project",
    "macro_dice_score",
    "measure_particles",
    "particle_instance_f1",
    "relink_source",
    "run_model_inference",
    "save_project",
    "segment_particles",
    "validate_project",
]
