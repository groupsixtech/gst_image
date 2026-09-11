"""Particle segmentation, instance separation, and morphology measurement."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import h_minima
from skimage.segmentation import watershed

from gst_image.analysis.preprocess import (
    estimate_illumination_field,
    flatten_illumination,
    threshold_array,
    to_gray,
)
from gst_image.models import (
    Calibration,
    MorphologicalGradient,
    MorphologicalInput,
    ParticleAnalysis,
    ParticleRecord,
    SegmentationMethod,
    SegmentationRecipe,
    ThresholdMethod,
)

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _report(callback: Progress | None, value: float, message: str) -> None:
    if callback:
        callback(value, message)


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Analysis cancelled")


def _area_limits(
    recipe: SegmentationRecipe, calibration: Calibration | None
) -> tuple[int, int | None]:
    minimum = recipe.min_particle_area_px
    maximum = recipe.max_particle_area_px
    if recipe.min_particle_area_mm2 is not None:
        if calibration is None:
            raise ValueError("Physical particle-area filters require image calibration")
        minimum = math.ceil(recipe.min_particle_area_mm2 / calibration.mm_per_pixel**2)
    if recipe.max_particle_area_mm2 is not None:
        if calibration is None:
            raise ValueError("Physical particle-area filters require image calibration")
        maximum = math.floor(recipe.max_particle_area_mm2 / calibration.mm_per_pixel**2)
    if maximum is not None and maximum < minimum:
        raise ValueError("Maximum particle area must be at least the minimum")
    return minimum, maximum


def _physical_recipe(
    recipe: SegmentationRecipe, calibration: Calibration | None
) -> SegmentationRecipe:
    updates: dict[str, int] = {}
    for physical, pixels, minimum in (
        ("open_radius_mm", "open_radius_px", 0),
        ("close_radius_mm", "close_radius_px", 0),
        ("watershed_min_distance_mm", "watershed_min_distance_px", 1),
    ):
        value = getattr(recipe, physical)
        if value is not None:
            if calibration is None:
                raise ValueError("Physical morphology settings require image calibration")
            updates[pixels] = max(minimum, round(value / calibration.mm_per_pixel))
    return recipe.model_copy(update=updates) if updates else recipe


def _otsu_from_preview(
    gray: np.ndarray, field: np.ndarray | None, analysis_mask: np.ndarray
) -> float:
    height, width = gray.shape
    scale = min(1.0, 2048 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    preview = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    if field is not None:
        preview_field = cv2.resize(field, size, interpolation=cv2.INTER_AREA)
        preview = flatten_illumination(preview, preview_field)
    domain = cv2.resize(
        analysis_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    values = preview[domain]
    if values.size == 0:
        raise ValueError("Analysis mask is empty")
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_OTSU)
    return float(threshold)


def _threshold_tiled(
    gray: np.ndarray,
    field: np.ndarray | None,
    analysis_mask: np.ndarray,
    recipe: SegmentationRecipe,
    progress: Progress | None,
    cancelled: Cancelled | None,
) -> np.ndarray:
    height, width = gray.shape
    output = np.zeros_like(gray, dtype=np.uint8)
    if recipe.threshold_method == ThresholdMethod.SAUVOLA:
        window_halo = recipe.sauvola_window_px // 2
    elif recipe.threshold_method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
        window_halo = recipe.gaussian_block_px // 2
    else:
        window_halo = 0
    morphology_halo = 2 * max(recipe.open_radius_px, recipe.close_radius_px)
    halo = max(8, window_halo + morphology_halo + 4)
    tile = recipe.tile_size_px
    total = math.ceil(height / tile) * math.ceil(width / tile)
    global_otsu = (
        _otsu_from_preview(gray, field, analysis_mask)
        if recipe.threshold_method == ThresholdMethod.OTSU
        else None
    )
    kernel_open = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * recipe.open_radius_px + 1,) * 2
        )
        if recipe.open_radius_px
        else None
    )
    kernel_close = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * recipe.close_radius_px + 1,) * 2
        )
        if recipe.close_radius_px
        else None
    )
    index = 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            _check_cancelled(cancelled)
            y2, x2 = min(height, y + tile), min(width, x + tile)
            ey1, ex1 = max(0, y - halo), max(0, x - halo)
            ey2, ex2 = min(height, y2 + halo), min(width, x2 + halo)
            tile_gray = gray[ey1:ey2, ex1:ex2]
            if field is not None:
                tile_gray = flatten_illumination(tile_gray, field[ey1:ey2, ex1:ex2])
            local = threshold_array(tile_gray, recipe, global_otsu)
            if kernel_open is not None:
                local = cv2.morphologyEx(
                    local.astype(np.uint8), cv2.MORPH_OPEN, kernel_open
                ) > 0
            if kernel_close is not None:
                local = cv2.morphologyEx(
                    local.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close
                ) > 0
            core = local[y - ey1 : y2 - ey1, x - ex1 : x2 - ex1]
            output[y:y2, x:x2] = core & analysis_mask[y:y2, x:x2]
            index += 1
            _report(progress, 0.15 + 0.45 * index / total, "Thresholding image tiles")
    return output


def _filter_and_label(mask: np.ndarray, minimum: int, maximum: int | None) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep[1:] = areas[1:] >= minimum
    if maximum is not None:
        keep[1:] &= areas[1:] <= maximum
    filtered = keep[labels].astype(np.uint8)
    _, relabeled = cv2.connectedComponents(filtered, connectivity=8)
    return relabeled.astype(np.int32, copy=False)


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background islands by flood-filling background connected to the border."""
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    background = (padded == 0).astype(np.uint8)
    flooded = background.copy()
    cv2.floodFill(flooded, None, (0, 0), 2, flags=4)
    holes = flooded == 1
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def _morphological_watershed_labels(
    gray: np.ndarray,
    field: np.ndarray | None,
    mask: np.ndarray,
    recipe: SegmentationRecipe,
    progress: Progress | None,
    cancelled: Cancelled | None,
) -> np.ndarray:
    """ImageJ-style extended-minima watershed constrained to threshold foreground."""
    _report(progress, 0.62, "Preparing morphological watershed")
    working = flatten_illumination(gray, field) if field is not None else gray
    if recipe.gaussian_blur_sigma > 0:
        working = cv2.GaussianBlur(working, (0, 0), recipe.gaussian_blur_sigma)
    if recipe.morphological_input == MorphologicalInput.OBJECT:
        radius = recipe.morphological_gradient_radius_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        eroded = cv2.erode(working, kernel)
        dilated = cv2.dilate(working, kernel)
        if recipe.morphological_gradient == MorphologicalGradient.INTERNAL:
            surface = cv2.subtract(working, eroded)
        elif recipe.morphological_gradient == MorphologicalGradient.EXTERNAL:
            surface = cv2.subtract(dilated, working)
        else:
            surface = cv2.subtract(dilated, eroded)
    else:
        surface = working
    connectivity = ndi.generate_binary_structure(
        2, 1 if recipe.morphological_connectivity == 4 else 2
    )
    minima = h_minima(
        surface,
        recipe.morphological_tolerance,
        footprint=connectivity,
    )
    minima &= np.asarray(mask, dtype=bool)
    markers, marker_count = ndi.label(minima, structure=connectivity)
    if marker_count == 0:
        markers, _ = ndi.label(mask, structure=connectivity)
    _check_cancelled(cancelled)
    _report(progress, 0.70, "Flooding morphological watershed basins")
    labels = watershed(
        surface,
        markers=markers,
        mask=np.asarray(mask, dtype=bool),
        connectivity=connectivity,
        watershed_line=recipe.morphological_calculate_dams,
    )
    return labels.astype(np.int32, copy=False)


def _compact_instance_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber existing instances without joining adjacent labeled regions."""
    values = np.unique(labels)
    values = values[values > 0]
    if values.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(values[-1]) + 1, dtype=np.int32)
    lookup[values] = np.arange(1, values.size + 1, dtype=np.int32)
    return lookup[labels]


def _filter_instance_labels(
    labels: np.ndarray, minimum: int, maximum: int | None
) -> np.ndarray:
    """Filter labeled instances by area while preserving watershed boundaries."""
    areas = np.bincount(labels.ravel())
    keep = areas >= minimum
    keep[0] = False
    if maximum is not None:
        keep &= areas <= maximum
    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    return _compact_instance_labels(filtered)


def _split_touching_components(
    labels: np.ndarray,
    minimum_area: int,
    min_distance: int,
    progress: Progress | None,
    cancelled: Cancelled | None,
) -> np.ndarray:
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (labels > 0).astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for component in range(1, count):
        _check_cancelled(cancelled)
        x, y, width, height, area = stats[component]
        source = components[y : y + height, x : x + width] == component
        if area < max(2 * minimum_area, math.pi * min_distance**2):
            result[y : y + height, x : x + width][source] = next_label
            next_label += 1
            continue
        distance = ndi.distance_transform_edt(source)
        peaks = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=max(1.0, float(distance.max()) * 0.35),
            labels=source,
            exclude_border=False,
        )
        if len(peaks) < 2:
            result[y : y + height, x : x + width][source] = next_label
            next_label += 1
        else:
            markers = np.zeros(source.shape, dtype=np.int32)
            for marker_id, (row, column) in enumerate(peaks, 1):
                markers[row, column] = marker_id
            split = watershed(-distance, markers, mask=source)
            target = result[y : y + height, x : x + width]
            for split_id in range(1, int(split.max()) + 1):
                member = split == split_id
                if np.count_nonzero(member) >= minimum_area:
                    target[member] = next_label
                    next_label += 1
        if component % 100 == 0:
            _report(progress, 0.6 + 0.15 * component / max(1, count - 1), "Splitting clusters")
    return result


def measure_particles(
    labels: np.ndarray,
    calibration: Calibration | None = None,
    analysis_mask: np.ndarray | None = None,
) -> list[ParticleRecord]:
    """Measure labeled two-dimensional particle sections."""
    if labels.ndim != 2:
        raise ValueError("Particle labels must be a two-dimensional array")
    boundary = None
    if analysis_mask is not None:
        domain = np.asarray(analysis_mask, dtype=bool)
        inner = ndi.binary_erosion(domain, structure=np.ones((3, 3)), border_value=0)
        boundary = domain & ~inner
    scale = calibration.mm_per_pixel if calibration else None
    records: list[ParticleRecord] = []
    for region in regionprops(labels):
        area = float(region.area)
        perimeter = float(region.perimeter_crofton)
        circularity = 4 * math.pi * area / perimeter**2 if perimeter else 0.0
        try:
            feret = float(region.feret_diameter_max)
        except (ValueError, AttributeError):
            feret = float(region.axis_major_length)
        touching = False
        if boundary is not None:
            touching = bool(np.any(boundary[region.slice] & region.image))
        equivalent_radius = math.sqrt(area / math.pi)
        records.append(
            ParticleRecord(
                label=int(region.label),
                area_px=area,
                area_mm2=area * scale**2 if scale else None,
                equivalent_radius_px=equivalent_radius,
                equivalent_radius_mm=equivalent_radius * scale if scale else None,
                equivalent_diameter_px=2 * equivalent_radius,
                equivalent_diameter_mm=2 * equivalent_radius * scale if scale else None,
                perimeter_crofton_px=perimeter,
                perimeter_crofton_mm=perimeter * scale if scale else None,
                circularity=float(np.clip(circularity, 0, 1)),
                solidity=float(region.solidity),
                eccentricity=float(region.eccentricity),
                major_axis_px=float(region.axis_major_length),
                major_axis_mm=float(region.axis_major_length) * scale if scale else None,
                minor_axis_px=float(region.axis_minor_length),
                minor_axis_mm=float(region.axis_minor_length) * scale if scale else None,
                feret_diameter_max_px=feret,
                feret_diameter_max_mm=feret * scale if scale else None,
                centroid_x_px=float(region.centroid[1]),
                centroid_y_px=float(region.centroid[0]),
                centroid_x_mm=float(region.centroid[1]) * scale if scale else None,
                centroid_y_mm=float(region.centroid[0]) * scale if scale else None,
                border_touching=touching,
            )
        )
    return records


def _summary(
    particles: list[ParticleRecord],
    mask: np.ndarray,
    analysis_mask: np.ndarray,
    recipe: SegmentationRecipe,
    calibration: Calibration | None,
) -> dict[str, Any]:
    eligible = [
        p
        for p in particles
        if not (recipe.exclude_border_particles_from_size_stats and p.border_touching)
    ]
    radii = np.asarray(
        [
            p.equivalent_radius_mm
            if calibration is not None
            else p.equivalent_radius_px
            for p in eligible
        ],
        dtype=float,
    )
    segmented = int(np.count_nonzero(mask))
    analyzed = int(np.count_nonzero(analysis_mask))
    result: dict[str, Any] = {
        "particle_count": len(particles),
        "size_statistics_count": len(eligible),
        "border_particle_count": sum(p.border_touching for p in particles),
        "segmented_pixels": segmented,
        "analyzed_pixels": analyzed,
        "area_fraction": segmented / analyzed if analyzed else 0.0,
        "estimated_volume_fraction_percent": 100 * segmented / analyzed if analyzed else 0.0,
        "number_density_per_pixel2": len(eligible) / analyzed if analyzed else 0.0,
        "radius_unit": "mm" if calibration else "px",
    }
    if calibration and analyzed:
        result["analyzed_area_mm2"] = analyzed * calibration.mm_per_pixel**2
        result["segmented_area_mm2"] = segmented * calibration.mm_per_pixel**2
        result["number_density_per_mm2"] = len(eligible) / result["analyzed_area_mm2"]
    if radii.size:
        result.update(
            radius_mean=float(np.mean(radii)),
            radius_std=float(np.std(radii, ddof=1)) if radii.size > 1 else 0.0,
            radius_min=float(np.min(radii)),
            radius_q1=float(np.percentile(radii, 25)),
            radius_median=float(np.median(radii)),
            radius_q3=float(np.percentile(radii, 75)),
            radius_max=float(np.max(radii)),
        )
    circularities = np.asarray([particle.circularity for particle in eligible], dtype=float)
    if circularities.size:
        result.update(
            circularity_mean=float(np.mean(circularities)),
            circularity_std=(
                float(np.std(circularities, ddof=1)) if circularities.size > 1 else 0.0
            ),
            circularity_q1=float(np.percentile(circularities, 25)),
            circularity_median=float(np.median(circularities)),
            circularity_q3=float(np.percentile(circularities, 75)),
        )
    return result


def segment_particles(
    image: np.ndarray,
    analysis_mask: np.ndarray | None,
    recipe: SegmentationRecipe,
    calibration: Calibration | None = None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> ParticleAnalysis:
    """Run illumination-aware, tiled particle segmentation and instance measurement."""
    recipe = _physical_recipe(recipe, calibration)
    gray = to_gray(image, recipe.channel)
    domain = (
        np.ones(gray.shape, dtype=bool)
        if analysis_mask is None
        else np.asarray(analysis_mask, dtype=bool)
    )
    if domain.shape != gray.shape or not np.any(domain):
        raise ValueError("Analysis mask must match the image and contain at least one pixel")
    minimum, maximum = _area_limits(recipe, calibration)
    _report(progress, 0.02, "Estimating illumination")
    field = (
        estimate_illumination_field(gray, recipe, domain)
        if recipe.illumination_correction
        else None
    )
    _check_cancelled(cancelled)
    raw_mask = _threshold_tiled(gray, field, domain, recipe, progress, cancelled)
    if recipe.fill_holes:
        _report(progress, 0.60, "Flood-filling enclosed holes")
        raw_mask = _fill_binary_holes(raw_mask) & domain
    if recipe.segmentation_method == SegmentationMethod.MORPHOLOGICAL_WATERSHED:
        labels = _morphological_watershed_labels(
            gray, field, raw_mask, recipe, progress, cancelled
        )
        labels = _filter_instance_labels(labels, minimum, maximum)
    else:
        _report(progress, 0.62, "Filtering connected particles")
        labels = _filter_and_label(raw_mask, minimum, maximum)
        if recipe.split_touching:
            labels = _split_touching_components(
                labels, minimum, recipe.watershed_min_distance_px, progress, cancelled
            )
            labels = _filter_instance_labels(labels, minimum, maximum)
    _check_cancelled(cancelled)
    mask = labels > 0
    _report(progress, 0.82, "Measuring particles")
    particles = measure_particles(labels, calibration, domain)
    summary = _summary(particles, mask, domain, recipe, calibration)
    preview_scale = min(1.0, 1600 / max(gray.shape))
    preview_size = (
        max(1, round(gray.shape[1] * preview_scale)),
        max(1, round(gray.shape[0] * preview_scale)),
    )
    corrected_preview = cv2.resize(gray, preview_size, interpolation=cv2.INTER_AREA)
    if field is not None:
        corrected_preview = flatten_illumination(
            corrected_preview,
            cv2.resize(field, preview_size, interpolation=cv2.INTER_AREA),
        )
    _report(progress, 1.0, "Analysis complete")
    return ParticleAnalysis(mask, labels, particles, summary, corrected_preview)


def merge_particle_labels(labels: np.ndarray, selected: list[int]) -> np.ndarray:
    selected = sorted({int(value) for value in selected if value > 0})
    if len(selected) < 2:
        return labels.copy()
    result = labels.copy()
    keep = selected[0]
    result[np.isin(result, selected)] = keep
    return _compact_instance_labels(result)


def split_particle_by_line(
    labels: np.ndarray,
    particle_label: int,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int = 1,
) -> np.ndarray:
    result = labels.copy()
    target = result == particle_label
    cut = np.zeros(result.shape, dtype=np.uint8)
    cv2.line(cut, tuple(map(round, start)), tuple(map(round, end)), 1, max(1, width))
    target[cut > 0] = False
    count, pieces = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)
    result[result == particle_label] = 0
    next_label = int(result.max()) + 1
    for piece in range(1, count):
        result[pieces == piece] = next_label
        next_label += 1
    return _compact_instance_labels(result)
