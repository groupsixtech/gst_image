"""Persistent and runtime models used by both the GUI and CLI."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_SCHEMA_VERSION = 5


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Point(BaseModel):
    x: float
    y: float


class Calibration(BaseModel):
    """Isotropic image calibration stored canonically as millimetres per pixel."""

    mm_per_pixel: float = Field(gt=0)
    reference_start: Point | None = None
    reference_end: Point | None = None
    reference_length: float | None = Field(default=None, gt=0)
    reference_unit: str = "mm"

    @classmethod
    def from_reference(
        cls,
        start: tuple[float, float],
        end: tuple[float, float],
        length: float,
        unit: str = "mm",
    ) -> Calibration:
        if length <= 0:
            raise ValueError("Reference length must be positive")
        unit_key = unit.strip().lower().replace("μ", "u")
        factors = {"mm": 1.0, "um": 0.001, "µm": 0.001}
        if unit_key not in factors:
            raise ValueError("Calibration unit must be mm or µm")
        pixels = math.dist(start, end)
        if pixels <= 0:
            raise ValueError("Calibration endpoints must be distinct")
        return cls(
            mm_per_pixel=length * factors[unit_key] / pixels,
            reference_start=Point(x=start[0], y=start[1]),
            reference_end=Point(x=end[0], y=end[1]),
            reference_length=length,
            reference_unit="µm" if unit_key in {"um", "µm"} else "mm",
        )


class ROIKind(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    ANALYSIS_BOX = "analysis_box"


class ROI(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    kind: ROIKind = ROIKind.INCLUDE
    shape: str = "polygon"
    points: list[Point]
    recipe_id: str | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> ROI:
        minimum = 2 if self.shape == "rectangle" else 3
        if len(self.points) < minimum:
            raise ValueError(f"{self.shape} ROI requires at least {minimum} points")
        return self


class ClassDefinition(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    color: str = "#ffd400"
    preset: str | None = None
    enabled: bool = True


def default_classes() -> list[ClassDefinition]:
    return [
        ClassDefinition(name="Weld", color="#ef5350", preset="weld"),
        ClassDefinition(name="HAZ", color="#ffb300", preset="haz"),
        ClassDefinition(name="Base Material", color="#42a5f5", preset="base"),
        ClassDefinition(name="Particle", color="#ffee58", preset="particle"),
        ClassDefinition(name="Phase", color="#66bb6a", preset="phase"),
        ClassDefinition(name="Excluded Area", color="#9e9e9e", preset="excluded"),
    ]


class ThresholdMethod(StrEnum):
    SAUVOLA = "sauvola"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    OTSU = "otsu"
    MANUAL = "manual"


class SegmentationMethod(StrEnum):
    THRESHOLD = "threshold"
    MORPHOLOGICAL_WATERSHED = "morphological_watershed"


class MorphologicalInput(StrEnum):
    OBJECT = "object"
    BORDER = "border"


class MorphologicalGradient(StrEnum):
    MORPHOLOGICAL = "morphological"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ParticlePolarity(StrEnum):
    DARK = "dark"
    BRIGHT = "bright"


class ParticleSizeMetric(StrEnum):
    EQUIVALENT_RADIUS = "equivalent_radius"
    EQUIVALENT_DIAMETER = "equivalent_diameter"
    AREA = "area"


class SegmentationRecipe(BaseModel):
    id: str = Field(default_factory=_id)
    name: str = "Particle segmentation"
    target_class_id: str | None = None
    channel: str = "gray"
    segmentation_method: SegmentationMethod = SegmentationMethod.THRESHOLD
    polarity: ParticlePolarity = ParticlePolarity.DARK
    illumination_correction: bool = True
    rolling_ball_radius_px: int = Field(default=151, ge=3)
    threshold_method: ThresholdMethod = ThresholdMethod.SAUVOLA
    manual_threshold_low: int = Field(default=0, ge=0, le=255)
    manual_threshold_high: int = Field(default=127, ge=0, le=255)
    sauvola_window_px: int = Field(default=101, ge=3)
    sauvola_k: float = Field(default=0.2, ge=-1, le=1)
    gaussian_block_px: int = Field(default=101, ge=3)
    gaussian_c: float = 2.0
    gaussian_blur_sigma: float = Field(default=0.8, ge=0)
    fill_holes: bool = False
    open_radius_px: int = Field(default=1, ge=0)
    close_radius_px: int = Field(default=1, ge=0)
    open_radius_mm: float | None = Field(default=None, ge=0)
    close_radius_mm: float | None = Field(default=None, ge=0)
    min_particle_area_px: int = Field(default=25, ge=0)
    max_particle_area_px: int | None = Field(default=None, ge=1)
    min_particle_area_mm2: float | None = Field(default=None, ge=0)
    max_particle_area_mm2: float | None = Field(default=None, ge=0)
    split_touching: bool = True
    watershed_min_distance_px: int = Field(default=7, ge=1)
    watershed_min_distance_mm: float | None = Field(default=None, gt=0)
    morphological_input: MorphologicalInput = MorphologicalInput.OBJECT
    morphological_gradient: MorphologicalGradient = MorphologicalGradient.MORPHOLOGICAL
    morphological_gradient_radius_px: int = Field(default=1, ge=1)
    morphological_tolerance: int = Field(default=10, ge=1, le=255)
    morphological_connectivity: Literal[4, 8] = 4
    morphological_calculate_dams: bool = True
    tile_size_px: int = Field(default=2048, ge=256)
    exclude_border_particles_from_size_stats: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_manual_threshold(cls, data: Any) -> Any:
        """Translate the legacy one-sided threshold into an inclusive 8-bit band."""
        if not isinstance(data, dict) or "manual_threshold" not in data:
            return data
        values = dict(data)
        threshold = float(values.pop("manual_threshold"))
        polarity = ParticlePolarity(values.get("polarity", ParticlePolarity.DARK))
        if polarity == ParticlePolarity.DARK:
            low, high = 0, math.ceil(threshold) - 1
        else:
            low, high = math.floor(threshold) + 1, 255
        values.setdefault("manual_threshold_low", max(0, min(255, low)))
        values.setdefault("manual_threshold_high", max(0, min(255, high)))
        return values

    @model_validator(mode="after")
    def normalize_windows(self) -> SegmentationRecipe:
        for field_name in ("sauvola_window_px", "gaussian_block_px"):
            value = getattr(self, field_name)
            if value % 2 == 0:
                setattr(self, field_name, value + 1)
        if (
            self.max_particle_area_px is not None
            and self.max_particle_area_px < self.min_particle_area_px
        ):
            raise ValueError("Maximum particle area must be at least the minimum")
        if self.manual_threshold_low > self.manual_threshold_high:
            raise ValueError("Manual threshold lower bound must not exceed the upper bound")
        return self


class RegionClassifierRecipe(BaseModel):
    id: str = Field(default_factory=_id)
    name: str = "Assisted region classification"
    sigma_min: float = Field(default=1.0, gt=0)
    sigma_max: float = Field(default=16.0, gt=0)
    intensity_features: bool = True
    edge_features: bool = True
    texture_features: bool = True
    n_estimators: int = Field(default=80, ge=10)
    max_depth: int | None = Field(default=12, ge=2)
    max_samples: float = Field(default=0.15, gt=0, le=1)
    random_seed: int = 0
    edge_snap_band_px: int = Field(default=3, ge=0, le=32)

    @model_validator(mode="after")
    def validate_sigmas(self) -> RegionClassifierRecipe:
        if self.sigma_max < self.sigma_min:
            raise ValueError("Maximum feature sigma must be at least the minimum")
        return self


class ModelInferenceRecipe(BaseModel):
    """Reproducible settings for a reviewed, packaged ONNX segmentation model."""

    id: str = Field(default_factory=_id)
    name: str = "Model inference"
    model_id: str
    model_version: str
    model_sha256: str = Field(min_length=64, max_length=64)
    semantic_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    particle_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    boundary_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    watershed_min_distance_px: int = Field(default=7, ge=1)
    min_particle_area_px: int = Field(default=1, ge=1)
    max_particle_area_px: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_particle_bounds(self) -> ModelInferenceRecipe:
        if (
            self.max_particle_area_px is not None
            and self.max_particle_area_px < self.min_particle_area_px
        ):
            raise ValueError("Maximum particle area must be at least the minimum")
        return self


class ModelInferenceRun(BaseModel):
    """Immutable provenance for an ML result that is awaiting or passed review."""

    id: str = Field(default_factory=_id)
    created_at: datetime = Field(default_factory=_now)
    recipe: ModelInferenceRecipe
    layer_ids: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    source_bounds_px: tuple[int, int, int, int] | None = None
    runtime_version: str = "not-recorded"
    review_status: Literal["pending", "confirmed"] = "pending"
    reviewed_at: datetime | None = None
    reviewer: str | None = None


class CellposeInferenceRecipe(BaseModel):
    """Reproducible settings for the locally cached stock Cellpose-SAM model."""

    id: str = Field(default_factory=_id)
    name: str = "Cellpose-SAM v2 inference"
    model_id: Literal["cpsam_v2"] = "cpsam_v2"
    modality: Literal["biological", "metallography"] = "biological"
    device: Literal["cpu", "gpu"] = "cpu"
    diameter_px: float | None = Field(default=None, gt=0)
    cellprob_threshold: float = Field(default=0.0, ge=-10, le=10)
    flow_threshold: float = Field(default=0.4, gt=0, le=10)
    min_size_px: int = Field(default=15, ge=0)
    tile_overlap: float = Field(default=0.1, ge=0, lt=1)
    noncommercial_license_accepted: bool = False


class CellposeInferenceRun(BaseModel):
    """Immutable provenance for a locally run Cellpose-SAM result."""

    id: str = Field(default_factory=_id)
    created_at: datetime = Field(default_factory=_now)
    recipe: CellposeInferenceRecipe
    layer_ids: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    source_bounds_px: tuple[int, int, int, int] | None = None
    cellpose_version: str = "not-recorded"
    torch_version: str = "not-recorded"
    model_sha256: str = "not-recorded"
    model_cache_path: str = "not-recorded"
    license_name: str = "CC-BY-NC"
    attribution: str = "Cellpose-SAM; Pachitariu, Rariden, and Stringer (2025)"
    license_acknowledged_at: datetime | None = None
    review_status: Literal["pending", "confirmed"] = "pending"
    reviewed_at: datetime | None = None
    reviewer: str | None = None


class SegmentationLayer(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    class_id: str | None = None
    kind: str = "binary"
    mask_path: str | None = None
    source_run_id: str | None = None
    scope_roi_ids: list[str] = Field(default_factory=list)
    visible: bool = True
    opacity: float = Field(default=0.45, ge=0, le=1)
    class_value_map: dict[int, str] = Field(default_factory=dict)
    review_status: Literal["not_required", "pending", "confirmed"] = "not_required"


class ParticleRecord(BaseModel):
    label: int
    area_px: float
    area_mm2: float | None = None
    equivalent_radius_px: float
    equivalent_radius_mm: float | None = None
    equivalent_diameter_px: float
    equivalent_diameter_mm: float | None = None
    perimeter_crofton_px: float
    perimeter_crofton_mm: float | None = None
    circularity: float
    solidity: float
    eccentricity: float
    major_axis_px: float
    major_axis_mm: float | None = None
    minor_axis_px: float
    minor_axis_mm: float | None = None
    feret_diameter_max_px: float
    feret_diameter_max_mm: float | None = None
    centroid_x_px: float
    centroid_y_px: float
    centroid_x_mm: float | None = None
    centroid_y_mm: float | None = None
    border_touching: bool = False
    group: str = "Unclassified"


class ParticleCriteria(BaseModel):
    size_metric: ParticleSizeMetric = ParticleSizeMetric.EQUIVALENT_RADIUS
    size_unit: Literal["mm", "px"] = "mm"
    size_min: float | None = Field(default=None, ge=0)
    size_max: float | None = Field(default=None, ge=0)
    circularity_min: float | None = Field(default=None, ge=0, le=1)
    circularity_max: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_radius_fields(cls, data: Any) -> Any:
        """Accept project-v2 radius fields and the former public constructor."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "radius_unit" in values:
            values.setdefault("size_unit", values.pop("radius_unit"))
        if "radius_min" in values:
            values.setdefault("size_min", values.pop("radius_min"))
        if "radius_max" in values:
            values.setdefault("size_max", values.pop("radius_max"))
        return values

    @model_validator(mode="after")
    def validate_bounds(self) -> ParticleCriteria:
        if (
            self.size_min is not None
            and self.size_max is not None
            and self.size_max < self.size_min
        ):
            label = (
                "radius"
                if self.size_metric == ParticleSizeMetric.EQUIVALENT_RADIUS
                else "size"
            )
            raise ValueError(f"Maximum {label} must be at least the minimum")
        if (
            self.circularity_min is not None
            and self.circularity_max is not None
            and self.circularity_max < self.circularity_min
        ):
            raise ValueError("Maximum circularity must be at least the minimum")
        return self

    def size_value(self, particle: ParticleRecord) -> float | None:
        calibrated = self.size_unit == "mm"
        if self.size_metric == ParticleSizeMetric.EQUIVALENT_RADIUS:
            return (
                particle.equivalent_radius_mm
                if calibrated
                else particle.equivalent_radius_px
            )
        if self.size_metric == ParticleSizeMetric.EQUIVALENT_DIAMETER:
            return (
                particle.equivalent_diameter_mm
                if calibrated
                else particle.equivalent_diameter_px
            )
        return particle.area_mm2 if calibrated else particle.area_px

    def matches(
        self,
        particle: ParticleRecord,
        *,
        include_size_upper: bool = True,
        include_circularity_upper: bool = True,
    ) -> bool:
        size = self.size_value(particle)
        if (self.size_min is not None or self.size_max is not None) and size is None:
            return False
        if (
            self.size_min is not None
            and size is not None
            and size < self.size_min
            and not math.isclose(size, self.size_min, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return False
        if self.size_max is not None and size is not None:
            if (
                include_size_upper
                and size > self.size_max
                and not math.isclose(size, self.size_max, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return False
            if not include_size_upper and size >= self.size_max:
                return False
        if (
            self.circularity_min is not None
            and particle.circularity < self.circularity_min
            and not math.isclose(
                particle.circularity,
                self.circularity_min,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            return False
        if self.circularity_max is not None:
            if (
                include_circularity_upper
                and particle.circularity > self.circularity_max
                and not math.isclose(
                    particle.circularity,
                    self.circularity_max,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                return False
            if not include_circularity_upper and particle.circularity >= self.circularity_max:
                return False
        return True

    @property
    def radius_unit(self) -> Literal["mm", "px"]:
        return self.size_unit

    @property
    def radius_min(self) -> float | None:
        return self.size_min

    @property
    def radius_max(self) -> float | None:
        return self.size_max


class ParticleGroup(ParticleCriteria):
    id: str = Field(default_factory=_id)
    name: str
    color: str = Field(default="#ffee58", pattern=r"^#[0-9A-Fa-f]{6}$")
    enabled: bool = True

    def matches(
        self,
        particle: ParticleRecord,
        *,
        final: bool = False,
        include_size_upper: bool | None = None,
        include_circularity_upper: bool | None = None,
    ) -> bool:
        return self.enabled and super().matches(
            particle,
            include_size_upper=(
                final if include_size_upper is None else include_size_upper
            ),
            include_circularity_upper=(
                final
                if include_circularity_upper is None
                else include_circularity_upper
            ),
        )


class ParticleGrouping(BaseModel):
    id: str = Field(default_factory=_id)
    name: str = "Particle grouping"
    source_layer_id: str | None = None
    filter_criteria: ParticleCriteria | None = None
    groups: list[ParticleGroup] = Field(default_factory=list)
    show_filtered: bool = False
    show_unclassified: bool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Measurement(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    start: Point
    end: Point
    length_px: float
    length_mm: float | None = None
    created_at: datetime = Field(default_factory=_now)


class AnalysisRun(BaseModel):
    id: str = Field(default_factory=_id)
    created_at: datetime = Field(default_factory=_now)
    recipe: SegmentationRecipe
    layer_ids: list[str] = Field(default_factory=list)
    result_files: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 0
    status: str = "complete"


class EditEvent(BaseModel):
    id: str = Field(default_factory=_id)
    created_at: datetime = Field(default_factory=_now)
    action: str
    target_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TrainingStroke(BaseModel):
    class_id: str
    points: list[Point]
    radius_px: float = Field(default=12, gt=0)
    erase: bool = False


class ProjectManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PROJECT_SCHEMA_VERSION
    project_id: str = Field(default_factory=_id)
    name: str
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    source_path: str
    source_sha256: str
    source_revision: int = 1
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    application_version: str = "0.1.0"
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    calibration: Calibration | None = None
    classes: list[ClassDefinition] = Field(default_factory=default_classes)
    rois: list[ROI] = Field(default_factory=list)
    recipes: list[SegmentationRecipe] = Field(default_factory=list)
    region_recipes: list[RegionClassifierRecipe] = Field(default_factory=list)
    model_inference_recipes: list[ModelInferenceRecipe] = Field(default_factory=list)
    cellpose_inference_recipes: list[CellposeInferenceRecipe] = Field(default_factory=list)
    layers: list[SegmentationLayer] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    # Retained for loading/API compatibility. New GUI work is stored as layer-scoped schemes.
    groups: list[ParticleGroup] = Field(default_factory=list)
    particle_groupings: list[ParticleGrouping] = Field(default_factory=list)
    active_particle_groupings: dict[str, str] = Field(default_factory=dict)
    runs: list[AnalysisRun] = Field(default_factory=list)
    model_inference_runs: list[ModelInferenceRun] = Field(default_factory=list)
    cellpose_inference_runs: list[CellposeInferenceRun] = Field(default_factory=list)
    edits: list[EditEvent] = Field(default_factory=list)
    training_strokes: list[TrainingStroke] = Field(default_factory=list)
    particle_records: dict[str, list[ParticleRecord]] = Field(default_factory=dict)
    portable_source: str | None = None


class FractionEntry(BaseModel):
    layer: str
    segmented_pixels: int
    analyzed_pixels: int
    area_fraction: float
    estimated_volume_fraction_percent: float


class FractionResults(BaseModel):
    entries: list[FractionEntry]


@dataclass(slots=True)
class ParticleAnalysis:
    mask: np.ndarray
    labels: np.ndarray
    particles: list[ParticleRecord]
    summary: dict[str, Any]
    corrected_preview: np.ndarray | None = None


@dataclass(slots=True)
class RegionAnalysis:
    labels: np.ndarray
    class_ids: list[int]
    summary: dict[str, Any]
