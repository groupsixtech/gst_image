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

from gst_image.models import FractionResults, ParticleRecord, ProjectManifest


def _hex_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        return (0, 255, 255)
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "layer"


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
        runs = {run.id: run for run in manifest.runs}
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
    pd.DataFrame([item.model_dump() for item in manifest.measurements]).to_csv(
        destination / "measurements.csv", index=False
    )
    if fractions is not None:
        pd.DataFrame([entry.model_dump() for entry in fractions.entries]).to_csv(
            destination / "fractions.csv", index=False
        )
    summaries = {run.id: run.summary for run in manifest.runs}
    (destination / "analysis_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    summary_rows = [
        {"analysis_run_id": run.id, "created_at": run.created_at, **run.summary}
        for run in manifest.runs
    ]
    pd.DataFrame(summary_rows).to_csv(destination / "analysis_summary.csv", index=False)
    if source_image is not None and masks:
        overlay = create_overlay(source_image, manifest, masks)
        cv2.imwrite(str(destination / "overlay.png"), overlay)
    return destination
