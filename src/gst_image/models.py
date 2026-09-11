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

PROJECT_SCHEMA_VERSION = 2


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


class ParticleGroup(BaseModel):
    name: str
    color: str = "#ffee58"
    radius_unit: Literal["mm", "px"] = "mm"
    radius_min: float | None = Field(default=None, ge=0)
    radius_max: float | None = Field(default=None, ge=0)
    circularity_min: float | None = Field(default=None, ge=0)
    circularity_max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ParticleGroup:
        if (
            self.radius_min is not None
            and self.radius_max is not None
            and self.radius_max < self.radius_min
        ):
            raise ValueError("Maximum radius must be at least the minimum")
        if (
            self.circularity_min is not None
            and self.circularity_max is not None
            and self.circularity_max < self.circularity_min
        ):
            raise ValueError("Maximum circularity must be at least the minimum")
        return self

    def matches(self, particle: ParticleRecord, *, final: bool = False) -> bool:
        radius = (
            particle.equivalent_radius_mm
            if self.radius_unit == "mm"
            else particle.equivalent_radius_px
        )
        if radius is None:
            return False
        if self.radius_min is not None and radius < self.radius_min:
            return False
        if self.radius_max is not None:
            if final and radius > self.radius_max:
                return False
            if not final and radius >= self.radius_max:
                return False
        if self.circularity_min is not None and particle.circularity < self.circularity_min:
            return False
        if self.circularity_max is not None:
            if final and particle.circularity > self.circularity_max:
                return False
            if not final and particle.circularity >= self.circularity_max:
                return False
        return True


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
    layers: list[SegmentationLayer] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    groups: list[ParticleGroup] = Field(default_factory=list)
    runs: list[AnalysisRun] = Field(default_factory=list)
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
