"""Ordered, exclusive particle grouping."""

from __future__ import annotations

from collections import Counter
from typing import Any

from gst_image.models import ParticleGroup, ParticleRecord


def assign_particle_groups(
    particles: list[ParticleRecord], groups: list[ParticleGroup]
) -> list[ParticleRecord]:
    assigned: list[ParticleRecord] = []
    for particle in particles:
        name = "Unclassified"
        for index, group in enumerate(groups):
            if group.matches(particle, final=index == len(groups) - 1):
                name = group.name
                break
        assigned.append(particle.model_copy(update={"group": name}))
    return assigned


def particle_group_summary(particles: list[ParticleRecord]) -> dict[str, dict[str, Any]]:
    counts = Counter(p.group for p in particles)
    areas: Counter[str] = Counter()
    for particle in particles:
        areas[particle.group] += particle.area_px
    return {
        name: {"count": count, "area_px": float(areas[name])}
        for name, count in sorted(counts.items())
    }

