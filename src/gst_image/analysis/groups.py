"""Non-destructive particle filtering, grouping, and group statistics."""

from __future__ import annotations

import colorsys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from gst_image.models import (
    ParticleCriteria,
    ParticleGroup,
    ParticleGrouping,
    ParticleRecord,
    ParticleSizeMetric,
)


@dataclass(slots=True)
class ParticleGroupingResult:
    """Derived assignments for one particle layer and one grouping scheme."""

    particles: list[ParticleRecord]
    assignments: dict[int, str | None]
    included_labels: set[int]
    filtered_labels: set[int]
    unclassified_labels: set[int]
    group_names: dict[int, str]


def assign_particle_groups(
    particles: list[ParticleRecord], groups: list[ParticleGroup]
) -> list[ParticleRecord]:
    """Compatibility wrapper for the original ordered group API."""
    assigned: list[ParticleRecord] = []
    enabled = [group for group in groups if group.enabled]
    for particle in particles:
        name = "Unclassified"
        for index, group in enumerate(enabled):
            if group.matches(particle, final=index == len(enabled) - 1):
                name = group.name
                break
        assigned.append(particle.model_copy(update={"group": name}))
    return assigned


def _outer_upper(groups: Iterable[ParticleGroup], attribute: str) -> float | None:
    values = [getattr(group, attribute) for group in groups]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _same_upper(value: float | None, outer: float | None) -> bool:
    if value is None:
        return outer is None
    return outer is not None and bool(np.isclose(value, outer, rtol=1e-12, atol=1e-12))


def evaluate_particle_grouping(
    particles: list[ParticleRecord],
    grouping: ParticleGrouping,
    *,
    copy_records: bool = True,
) -> ParticleGroupingResult:
    """Apply a reversible filter, then assign included particles to colored groups."""
    enabled = [group for group in grouping.groups if group.enabled]
    outer_size = _outer_upper(enabled, "size_max")
    outer_circularity = _outer_upper(enabled, "circularity_max")
    count = len(particles)
    labels = np.fromiter((particle.label for particle in particles), dtype=np.int64, count=count)
    circularities = np.fromiter(
        (particle.circularity for particle in particles), dtype=float, count=count
    )
    size_cache: dict[tuple[ParticleSizeMetric, str], np.ndarray] = {}

    def size_values(criteria: ParticleCriteria) -> np.ndarray:
        key = (criteria.size_metric, criteria.size_unit)
        if key not in size_cache:
            if criteria.size_metric == ParticleSizeMetric.EQUIVALENT_RADIUS:
                attribute = (
                    "equivalent_radius_mm"
                    if criteria.size_unit == "mm"
                    else "equivalent_radius_px"
                )
            elif criteria.size_metric == ParticleSizeMetric.EQUIVALENT_DIAMETER:
                attribute = (
                    "equivalent_diameter_mm"
                    if criteria.size_unit == "mm"
                    else "equivalent_diameter_px"
                )
            else:
                attribute = "area_mm2" if criteria.size_unit == "mm" else "area_px"
            size_cache[key] = np.fromiter(
                (
                    np.nan if (value := getattr(particle, attribute)) is None else value
                    for particle in particles
                ),
                dtype=float,
                count=count,
            )
        return size_cache[key]

    def criteria_mask(
        criteria: ParticleCriteria,
        *,
        include_size_upper: bool = True,
        include_circularity_upper: bool = True,
    ) -> np.ndarray:
        selected = np.ones(count, dtype=bool)
        if criteria.size_min is not None or criteria.size_max is not None:
            sizes = size_values(criteria)
            selected &= np.isfinite(sizes)
            if criteria.size_min is not None:
                selected &= (sizes >= criteria.size_min) | np.isclose(
                    sizes, criteria.size_min, rtol=1e-9, atol=1e-12
                )
            if criteria.size_max is not None:
                selected &= (
                    (sizes <= criteria.size_max)
                    | np.isclose(sizes, criteria.size_max, rtol=1e-9, atol=1e-12)
                    if include_size_upper
                    else sizes < criteria.size_max
                )
        if criteria.circularity_min is not None:
            selected &= (circularities >= criteria.circularity_min) | np.isclose(
                circularities,
                criteria.circularity_min,
                rtol=1e-9,
                atol=1e-12,
            )
        if criteria.circularity_max is not None:
            selected &= (
                (circularities <= criteria.circularity_max)
                | np.isclose(
                    circularities,
                    criteria.circularity_max,
                    rtol=1e-9,
                    atol=1e-12,
                )
                if include_circularity_upper
                else circularities < criteria.circularity_max
            )
        return selected

    included_mask = (
        criteria_mask(grouping.filter_criteria)
        if grouping.filter_criteria
        else np.ones(count, dtype=bool)
    )
    assigned_indexes = np.full(count, -1, dtype=np.int32)
    for index, group in enumerate(enabled):
        matches = criteria_mask(
            group,
            include_size_upper=_same_upper(group.size_max, outer_size),
            include_circularity_upper=_same_upper(
                group.circularity_max, outer_circularity
            ),
        )
        matches &= included_mask & (assigned_indexes < 0)
        assigned_indexes[matches] = index

    assignments: dict[int, str | None] = {}
    included = set(labels[included_mask].tolist())
    filtered = set(labels[~included_mask].tolist())
    unclassified = set(labels[included_mask & (assigned_indexes < 0)].tolist())
    updated: list[ParticleRecord] = []
    group_names: dict[int, str] = {}

    for position, particle in enumerate(particles):
        if not included_mask[position]:
            assignments[particle.label] = None
            group_names[particle.label] = "Filtered out"
            updated.append(
                particle.model_copy(update={"group": "Filtered out"})
                if copy_records
                else particle
            )
            continue
        group_index = int(assigned_indexes[position])
        if group_index < 0:
            assignments[particle.label] = None
            group_names[particle.label] = "Unclassified"
            updated.append(
                particle.model_copy(update={"group": "Unclassified"})
                if copy_records
                else particle
            )
        else:
            match = enabled[group_index]
            assignments[particle.label] = match.id
            group_names[particle.label] = match.name
            updated.append(
                particle.model_copy(update={"group": match.name})
                if copy_records
                else particle
            )

    return ParticleGroupingResult(
        particles=updated,
        assignments=assignments,
        included_labels=included,
        filtered_labels=filtered,
        unclassified_labels=unclassified,
        group_names=group_names,
    )


def _intervals_overlap(
    first_min: float | None,
    first_max: float | None,
    second_min: float | None,
    second_max: float | None,
) -> bool:
    low = max(
        float("-inf") if first_min is None else first_min,
        float("-inf") if second_min is None else second_min,
    )
    high = min(
        float("inf") if first_max is None else first_max,
        float("inf") if second_max is None else second_max,
    )
    return low < high


def validate_particle_groups(groups: list[ParticleGroup]) -> None:
    """Reject ambiguous group definitions while allowing adjacent range boundaries."""
    enabled = [group for group in groups if group.enabled]
    names = [group.name.strip().casefold() for group in enabled]
    if any(not name for name in names):
        raise ValueError("Every enabled particle group requires a name")
    if len(names) != len(set(names)):
        raise ValueError("Particle group names must be unique")
    for index, first in enumerate(enabled):
        for second in enabled[index + 1 :]:
            same_measure = (
                first.size_metric == second.size_metric
                and first.size_unit == second.size_unit
            )
            if not same_measure:
                raise ValueError("All groups in a scheme must use the same size metric and unit")
            if _intervals_overlap(
                first.size_min, first.size_max, second.size_min, second.size_max
            ) and _intervals_overlap(
                first.circularity_min,
                first.circularity_max,
                second.circularity_min,
                second.circularity_max,
            ):
                raise ValueError(
                    f"Particle groups {first.name!r} and {second.name!r} overlap"
                )


def generate_particle_groups(
    size_edges: list[float],
    circularity_edges: list[float],
    *,
    size_metric: ParticleSizeMetric = ParticleSizeMetric.EQUIVALENT_RADIUS,
    size_unit: str = "px",
    colors: list[str] | None = None,
) -> list[ParticleGroup]:
    """Generate a disjoint one-dimensional grouping or two-dimensional grid."""
    if len(size_edges) < 2 or len(circularity_edges) < 2:
        raise ValueError("Grouping edges require at least a minimum and maximum")
    if any(right <= left for left, right in pairwise(size_edges)):
        raise ValueError("Size grouping boundaries must be strictly increasing")
    if any(
        right <= left
        for left, right in pairwise(circularity_edges)
    ):
        raise ValueError("Circularity grouping boundaries must be strictly increasing")
    palette = list(colors) if colors else [
        "#42a5f5",
        "#ef5350",
        "#66bb6a",
        "#ffb300",
        "#ab47bc",
        "#26c6da",
        "#ec407a",
        "#8d6e63",
        "#7e57c2",
        "#9ccc65",
        "#ffa726",
        "#26a69a",
    ]
    requested_colors = (len(size_edges) - 1) * (len(circularity_edges) - 1)
    color_index = len(palette)
    while len(palette) < requested_colors:
        red, green, blue = colorsys.hsv_to_rgb(
            (color_index * 0.61803398875) % 1.0, 0.68, 0.9
        )
        color_index += 1
        candidate = f"#{round(255 * red):02x}{round(255 * green):02x}{round(255 * blue):02x}"
        if candidate not in palette:
            palette.append(candidate)
    groups: list[ParticleGroup] = []
    size_count = len(size_edges) - 1
    circularity_count = len(circularity_edges) - 1
    for size_index, (size_min, size_max) in enumerate(
        pairwise(size_edges), start=1
    ):
        for circularity_index, (circularity_min, circularity_max) in enumerate(
            pairwise(circularity_edges), start=1
        ):
            if size_count > 1 and circularity_count > 1:
                name = f"Size {size_index} / Circularity {circularity_index}"
            elif size_count > 1:
                name = f"Size {size_index}"
            elif circularity_count > 1:
                name = f"Circularity {circularity_index}"
            else:
                name = "Included particles"
            groups.append(
                ParticleGroup(
                    name=name,
                    color=palette[len(groups) % len(palette)],
                    size_metric=size_metric,
                    size_unit=size_unit,
                    size_min=float(size_min),
                    size_max=float(size_max),
                    circularity_min=float(circularity_min),
                    circularity_max=float(circularity_max),
                )
            )
    return groups


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "min": float(np.min(data)),
        "q1": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "q3": float(np.percentile(data, 75)),
        "max": float(np.max(data)),
    }


def particle_group_statistics(
    particles: list[ParticleRecord],
    grouping: ParticleGrouping,
    result: ParticleGroupingResult | None = None,
    *,
    analyzed_pixels: int | None = None,
    exclude_border_from_size: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return display/export statistics for every group plus unmatched categories."""
    result = result or evaluate_particle_grouping(particles, grouping)
    groups_by_id = {group.id: group for group in grouping.groups if group.enabled}
    buckets: dict[str, list[ParticleRecord]] = {group.id: [] for group in groups_by_id.values()}
    buckets["__unclassified__"] = []
    buckets["__filtered__"] = []
    for particle in result.particles:
        if particle.label in result.filtered_labels:
            buckets["__filtered__"].append(particle)
        else:
            buckets[result.assignments[particle.label] or "__unclassified__"].append(particle)

    included_count = len(result.included_labels)
    criteria = next(iter(groups_by_id.values()), grouping.filter_criteria)
    summary: dict[str, dict[str, Any]] = {}
    for bucket_id, records in buckets.items():
        if bucket_id == "__filtered__":
            name, color, included_bucket = "Filtered out", "#616161", False
        elif bucket_id == "__unclassified__":
            name, color, included_bucket = "Unclassified", "#9e9e9e", True
        else:
            group = groups_by_id[bucket_id]
            name, color, included_bucket = group.name, group.color, True
        eligible = [
            particle
            for particle in records
            if not (exclude_border_from_size and particle.border_touching)
        ]
        metric = groups_by_id.get(bucket_id, criteria)
        sizes = (
            [value for particle in eligible if (value := metric.size_value(particle)) is not None]
            if metric is not None
            else []
        )
        circularities = [particle.circularity for particle in eligible]
        area = float(sum(particle.area_px for particle in records))
        physical_areas = [
            particle.area_mm2 for particle in records if particle.area_mm2 is not None
        ]
        fraction = area / analyzed_pixels if analyzed_pixels else None
        summary[name] = {
            "group_id": None if bucket_id.startswith("__") else bucket_id,
            "color": color,
            "included": included_bucket,
            "count": len(records),
            "count_percent": (
                100 * len(records) / included_count
                if included_bucket and included_count
                else 0.0
            ),
            "area_px": area,
            "area_mm2": float(sum(physical_areas)) if physical_areas else None,
            "area_fraction": fraction,
            "estimated_volume_fraction_percent": 100 * fraction if fraction is not None else None,
            "border_touching_count": sum(particle.border_touching for particle in records),
            "size_statistics_count": len(sizes),
            "size": _distribution(sizes),
            "circularity": _distribution(circularities),
        }
    return summary


def particle_group_summary(particles: list[ParticleRecord]) -> dict[str, dict[str, Any]]:
    """Compatibility summary for records carrying the legacy group-name field."""
    counts = Counter(p.group for p in particles)
    areas: Counter[str] = Counter()
    for particle in particles:
        areas[particle.group] += particle.area_px
    return {
        name: {"count": count, "area_px": float(areas[name])}
        for name, count in sorted(counts.items())
    }
