"""Portable tabular, raster, and provenance exports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from gst_image.analysis.groups import evaluate_particle_grouping, particle_group_statistics
from gst_image.models import (
    ROI,
    FractionResults,
    ParticleGrouping,
    ParticleRecord,
    ProjectManifest,
    SegmentationLayer,
)


def _hex_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        return (0, 255, 255)
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "layer"


def _roi_bounds(
    roi: ROI, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Return a clipped, half-open source-image bounding box for one ROI."""
    xs = [point.x for point in roi.points]
    ys = [point.y for point in roi.points]
    left = min(image_width - 1, max(0, int(np.floor(min(xs)))))
    top = min(image_height - 1, max(0, int(np.floor(min(ys)))))
    right = min(image_width, int(np.ceil(max(xs))) + 1)
    bottom = min(image_height, int(np.ceil(max(ys))) + 1)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _scope_bounds(
    manifest: ProjectManifest, scope_roi_ids: Sequence[str]
) -> tuple[int, int, int, int]:
    """Return the union bounds of a layer's saved ROI scope, or the full image."""
    scoped = [roi for roi in manifest.rois if roi.id in scope_roi_ids]
    if not scoped:
        return 0, 0, manifest.image_width, manifest.image_height
    bounds = [
        _roi_bounds(roi, manifest.image_width, manifest.image_height) for roi in scoped
    ]
    return (
        min(value[0] for value in bounds),
        min(value[1] for value in bounds),
        max(value[2] for value in bounds),
        max(value[3] for value in bounds),
    )


def create_segmentation_layer_overlay(
    image: np.ndarray,
    manifest: ProjectManifest,
    layer: SegmentationLayer,
    values: np.ndarray,
) -> np.ndarray:
    """Render one segmentation layer over an equally sized source image."""
    visible_layer = layer.model_copy(update={"visible": True})
    isolated_manifest = manifest.model_copy(update={"layers": [visible_layer]})
    return create_overlay(image, isolated_manifest, {layer.id: values})


def create_overlay(
    image: np.ndarray,
    manifest: ProjectManifest,
    masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image[:, :, :3].copy()
    classes = {item.id: item for item in manifest.classes}
    for layer in manifest.layers:
        if not layer.visible or layer.id not in masks:
            continue
        layer_values = np.asarray(masks[layer.id])
        if layer_values.shape != overlay.shape[:2]:
            raise ValueError(f"Layer {layer.name!r} does not match the source image")
        for top in range(0, overlay.shape[0], 512):
            bottom = min(overlay.shape[0], top + 512)
            value_block = layer_values[top:bottom]
            overlay_block = overlay[top:bottom]
            if layer.kind == "multiclass":
                selections = (
                    (value_block == value, classes[class_id].color)
                    for value, class_id in layer.class_value_map.items()
                    if class_id in classes
                )
            else:
                color = (
                    classes[layer.class_id].color
                    if layer.class_id in classes
                    else "#ffff00"
                )
                selections = ((value_block > 0, color),)
            for selected, hex_color in selections:
                if not np.any(selected):
                    continue
                pixels = overlay_block[selected].astype(np.float32)
                pixels *= 1 - layer.opacity
                pixels += np.asarray(_hex_bgr(hex_color), dtype=np.float32) * layer.opacity
                overlay_block[selected] = np.clip(pixels, 0, 255).astype(np.uint8)
    return overlay


def create_particle_group_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    particles: Sequence[ParticleRecord],
    grouping: ParticleGrouping,
    *,
    opacity: float = 0.6,
) -> np.ndarray:
    """Color an instance layer with one saved grouping without changing its labels."""
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image[:, :, :3].copy()
    labels = np.asarray(labels)
    if labels.shape != overlay.shape[:2]:
        raise ValueError("Particle labels do not match the source image")
    result = evaluate_particle_grouping(
        list(particles), grouping, copy_records=False
    )
    maximum = max(0, int(labels.max()))
    colors = np.zeros((maximum + 1, 3), dtype=np.uint8)
    visible = np.zeros(maximum + 1, dtype=bool)
    group_colors = {group.id: _hex_bgr(group.color) for group in grouping.groups}
    for particle in result.particles:
        if not 0 <= particle.label <= maximum:
            continue
        if particle.label in result.filtered_labels:
            if grouping.show_filtered:
                colors[particle.label] = _hex_bgr("#616161")
                visible[particle.label] = True
            continue
        group_id = result.assignments.get(particle.label)
        if group_id in group_colors:
            colors[particle.label] = group_colors[group_id]
            visible[particle.label] = True
        elif grouping.show_unclassified:
            colors[particle.label] = _hex_bgr("#9e9e9e")
            visible[particle.label] = True
    safe = np.clip(labels.astype(np.int64), 0, maximum)
    selected = (labels >= 0) & (labels <= maximum) & visible[safe]
    if np.any(selected):
        pixels = overlay[selected].astype(np.float32)
        pixels *= 1 - opacity
        pixels += colors[safe[selected]].astype(np.float32) * opacity
        overlay[selected] = np.clip(pixels, 0, 255).astype(np.uint8)
    return overlay


def _export_particle_groupings(
    destination: Path,
    manifest: ProjectManifest,
    masks: Mapping[str, np.ndarray],
    source_image: np.ndarray | None,
) -> None:
    definitions: list[dict] = []
    assignments: list[dict] = []
    statistics_rows: list[dict] = []
    layers = {layer.id: layer for layer in manifest.layers}
    runs = {
        **{run.id: run for run in manifest.runs},
        **{run.id: run for run in manifest.model_inference_runs},
    }
    for grouping in manifest.particle_groupings:
        layer_id = grouping.source_layer_id
        if layer_id is None or layer_id not in manifest.particle_records:
            continue
        layer = layers.get(layer_id)
        records = manifest.particle_records[layer_id]
        result = evaluate_particle_grouping(records, grouping, copy_records=False)
        filter_values = (
            grouping.filter_criteria.model_dump(mode="json")
            if grouping.filter_criteria
            else {}
        )
        for group in grouping.groups:
            definitions.append(
                {
                    "grouping_id": grouping.id,
                    "grouping": grouping.name,
                    "analysis_layer_id": layer_id,
                    "analysis_layer": layer.name if layer else layer_id,
                    "group_id": group.id,
                    "group": group.name,
                    "color": group.color,
                    "enabled": group.enabled,
                    "size_metric": group.size_metric.value,
                    "size_unit": group.size_unit,
                    "size_min": group.size_min,
                    "size_max": group.size_max,
                    "circularity_min": group.circularity_min,
                    "circularity_max": group.circularity_max,
                    "filter_size_min": filter_values.get("size_min"),
                    "filter_size_max": filter_values.get("size_max"),
                    "filter_size_metric": filter_values.get("size_metric"),
                    "filter_size_unit": filter_values.get("size_unit"),
                    "filter_circularity_min": filter_values.get("circularity_min"),
                    "filter_circularity_max": filter_values.get("circularity_max"),
                    "show_filtered": grouping.show_filtered,
                    "show_unclassified": grouping.show_unclassified,
                }
            )
        if not grouping.groups:
            definitions.append(
                {
                    "grouping_id": grouping.id,
                    "grouping": grouping.name,
                    "analysis_layer_id": layer_id,
                    "analysis_layer": layer.name if layer else layer_id,
                    "group_id": None,
                    "group": None,
                    "color": None,
                    "enabled": None,
                    "size_metric": filter_values.get("size_metric"),
                    "size_unit": filter_values.get("size_unit"),
                    "size_min": None,
                    "size_max": None,
                    "circularity_min": None,
                    "circularity_max": None,
                    "filter_size_min": filter_values.get("size_min"),
                    "filter_size_max": filter_values.get("size_max"),
                    "filter_size_metric": filter_values.get("size_metric"),
                    "filter_size_unit": filter_values.get("size_unit"),
                    "filter_circularity_min": filter_values.get("circularity_min"),
                    "filter_circularity_max": filter_values.get("circularity_max"),
                    "show_filtered": grouping.show_filtered,
                    "show_unclassified": grouping.show_unclassified,
                }
            )
        names = {group.id: group.name for group in grouping.groups}
        for particle in result.particles:
            group_id = result.assignments.get(particle.label)
            filtered = particle.label in result.filtered_labels
            assignments.append(
                {
                    "grouping_id": grouping.id,
                    "grouping": grouping.name,
                    "analysis_layer_id": layer_id,
                    "analysis_layer": layer.name if layer else layer_id,
                    "particle_label": particle.label,
                    "included": not filtered,
                    "filtered": filtered,
                    "group_id": group_id,
                    "group": (
                        "Filtered out"
                        if filtered
                        else names.get(group_id, "Unclassified")
                    ),
                }
            )
        run = runs.get(layer.source_run_id) if layer else None
        group_statistics = particle_group_statistics(
            records,
            grouping,
            result,
            analyzed_pixels=run.summary.get("analyzed_pixels") if run else None,
            exclude_border_from_size=(
                getattr(run.recipe, "exclude_border_particles_from_size_stats", True)
                if run
                else True
            ),
        )
        for name, values in group_statistics.items():
            statistics_rows.append(
                {
                    "grouping_id": grouping.id,
                    "grouping": grouping.name,
                    "analysis_layer_id": layer_id,
                    "analysis_layer": layer.name if layer else layer_id,
                    "group_id": values["group_id"],
                    "group": name,
                    "color": values["color"],
                    "included": values["included"],
                    "count": values["count"],
                    "count_percent": values["count_percent"],
                    "area_px": values["area_px"],
                    "area_mm2": values["area_mm2"],
                    "area_fraction": values["area_fraction"],
                    "estimated_volume_fraction_percent": values[
                        "estimated_volume_fraction_percent"
                    ],
                    "border_touching_count": values["border_touching_count"],
                    "size_statistics_count": values["size_statistics_count"],
                    **{f"size_{key}": value for key, value in values["size"].items()},
                    **{
                        f"circularity_{key}": value
                        for key, value in values["circularity"].items()
                    },
                }
            )
        if source_image is not None and layer_id in masks:
            left, top, right, bottom = _scope_bounds(
                manifest, layer.scope_roi_ids if layer else []
            )
            grouped_overlay = create_particle_group_overlay(
                source_image[top:bottom, left:right],
                np.asarray(masks[layer_id])[top:bottom, left:right],
                records,
                grouping,
            )
            filename = _safe_filename(f"particle_grouping_{grouping.name}_{grouping.id[:8]}")
            cv2.imwrite(str(destination / f"{filename}.png"), grouped_overlay)
    if definitions:
        pd.DataFrame(definitions).to_csv(
            destination / "particle_grouping_definitions.csv", index=False
        )
    if assignments:
        pd.DataFrame(assignments).to_csv(
            destination / "particle_group_assignments.csv", index=False
        )
    if statistics_rows:
        pd.DataFrame(statistics_rows).to_csv(
            destination / "particle_group_statistics.csv", index=False
        )


def _export_roi_and_segmentation_images(
    destination: Path,
    manifest: ProjectManifest,
    masks: Mapping[str, np.ndarray],
    source_image: np.ndarray | None,
) -> None:
    """Write native source ROI crops and crop-aware visual overlays for result layers."""
    if source_image is None:
        return
    for roi in manifest.rois:
        left, top, right, bottom = _roi_bounds(
            roi, manifest.image_width, manifest.image_height
        )
        filename = _safe_filename(
            f"roi_{roi.kind.value}_{roi.name}_{roi.id[:8]}"
        )
        cv2.imwrite(
            str(destination / f"{filename}.png"),
            source_image[top:bottom, left:right],
        )
    for layer in manifest.layers:
        if layer.kind == "domain" or layer.id not in masks:
            continue
        left, top, right, bottom = _scope_bounds(manifest, layer.scope_roi_ids)
        overlay = create_segmentation_layer_overlay(
            source_image[top:bottom, left:right],
            manifest,
            layer,
            np.asarray(masks[layer.id])[top:bottom, left:right],
        )
        filename = _safe_filename(
            f"segmentation_overlay_{layer.name}_{layer.id[:8]}"
        )
        cv2.imwrite(str(destination / f"{filename}.png"), overlay)


def export_analysis(
    output: str | Path,
    manifest: ProjectManifest,
    masks: Mapping[str, np.ndarray],
    *,
    particles: Sequence[ParticleRecord] = (),
    fractions: FractionResults | None = None,
    source_image: np.ndarray | None = None,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "provenance.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    (destination / "recipes.json").write_text(
        json.dumps(
            {
                "segmentation": [item.model_dump(mode="json") for item in manifest.recipes],
                "region_classification": [
                    item.model_dump(mode="json") for item in manifest.region_recipes
                ],
                "model_inference": [
                    item.model_dump(mode="json") for item in manifest.model_inference_recipes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if manifest.recipes:
        (destination / "recipe.json").write_text(
            manifest.recipes[-1].model_dump_json(indent=2), encoding="utf-8"
        )
    for layer in manifest.layers:
        if layer.id in masks:
            tifffile.imwrite(
                destination / f"{_safe_filename(layer.name)}_{layer.id[:8]}_mask.tif",
                np.asarray(masks[layer.id]),
                compression="zlib",
                metadata={"axes": "YX"},
            )
    persistent_particles = [
        {
            "analysis_layer_id": layer_id,
            "analysis_layer": next(
                (layer.name for layer in manifest.layers if layer.id == layer_id), layer_id
            ),
            **particle.model_dump(),
        }
        for layer_id, records in manifest.particle_records.items()
        for particle in records
    ]
    particle_rows = persistent_particles or [particle.model_dump() for particle in particles]
    if particle_rows:
        table = pd.DataFrame(particle_rows)
        table.to_csv(destination / "particles.csv", index=False)
        histogram_groups = (
            table.groupby("analysis_layer_id", dropna=False)
            if "analysis_layer_id" in table
            else [("", table)]
        )
        radius_histograms = []
        circularity_histograms = []
        for layer_id, layer_table in histogram_groups:
            radii = layer_table["equivalent_radius_mm"].dropna()
            unit = "mm"
            if radii.empty:
                radii = layer_table["equivalent_radius_px"]
                unit = "px"
            counts, edges = np.histogram(radii.to_numpy(), bins="auto")
            radius_histograms.extend(
                {
                    "analysis_layer_id": layer_id,
                    "unit": unit,
                    "bin_left": left,
                    "bin_right": right,
                    "count": int(count),
                }
                for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
            )
            counts, edges = np.histogram(
                layer_table["circularity"].to_numpy(), bins=np.linspace(0, 1, 21)
            )
            circularity_histograms.extend(
                {
                    "analysis_layer_id": layer_id,
                    "bin_left": left,
                    "bin_right": right,
                    "count": int(count),
                }
                for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
            )
        pd.DataFrame(radius_histograms).to_csv(
            destination / "radius_histogram.csv", index=False
        )
        pd.DataFrame(circularity_histograms).to_csv(
            destination / "circularity_histogram.csv", index=False
        )
        group_table = (
            table.groupby(
                (["analysis_layer_id", "analysis_layer"] if "analysis_layer_id" in table else [])
                + ["group"],
                dropna=False,
            )
            .agg(count=("label", "count"), area_px=("area_px", "sum"))
            .reset_index()
        )
        runs = {
            **{run.id: run for run in manifest.runs},
            **{run.id: run for run in manifest.model_inference_runs},
        }
        denominators = {
            layer.id: runs[layer.source_run_id].summary.get("analyzed_pixels")
            for layer in manifest.layers
            if layer.source_run_id in runs
        }
        if "analysis_layer_id" in group_table:
            group_table["analyzed_pixels"] = group_table["analysis_layer_id"].map(
                denominators
            )
            valid = group_table["analyzed_pixels"].fillna(0) > 0
            group_table.loc[valid, "area_fraction"] = (
                group_table.loc[valid, "area_px"]
                / group_table.loc[valid, "analyzed_pixels"]
            )
        else:
            analyzed_pixels = next(
                (
                    run.summary.get("analyzed_pixels")
                    for run in reversed(manifest.runs)
                    if run.summary.get("analyzed_pixels")
                ),
                None,
            )
            if analyzed_pixels:
                group_table["area_fraction"] = group_table["area_px"] / analyzed_pixels
        if "area_fraction" in group_table:
            group_table["estimated_volume_fraction_percent"] = (
                100 * group_table["area_fraction"]
            )
        group_table.to_csv(destination / "particle_groups.csv", index=False)
    _export_roi_and_segmentation_images(destination, manifest, masks, source_image)
    _export_particle_groupings(destination, manifest, masks, source_image)
    pd.DataFrame([item.model_dump() for item in manifest.measurements]).to_csv(
        destination / "measurements.csv", index=False
    )
    if fractions is not None:
        pd.DataFrame([entry.model_dump() for entry in fractions.entries]).to_csv(
            destination / "fractions.csv", index=False
        )
    summaries = {
        **{run.id: run.summary for run in manifest.runs},
        **{run.id: run.summary for run in manifest.model_inference_runs},
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    summary_rows = [
        {
            "analysis_run_id": run.id,
            "created_at": run.created_at,
            "analysis_kind": "rule_based",
            **run.summary,
        }
        for run in manifest.runs
    ] + [
        {
            "analysis_run_id": run.id,
            "created_at": run.created_at,
            "analysis_kind": "model_inference",
            "model_id": run.recipe.model_id,
            "model_version": run.recipe.model_version,
            "model_sha256": run.recipe.model_sha256,
            "review_status": run.review_status,
            **run.summary,
        }
        for run in manifest.model_inference_runs
    ]
    pd.DataFrame(summary_rows).to_csv(destination / "analysis_summary.csv", index=False)
    if source_image is not None and masks:
        overlay = create_overlay(source_image, manifest, masks)
        cv2.imwrite(str(destination / "overlay.png"), overlay)
    return destination
