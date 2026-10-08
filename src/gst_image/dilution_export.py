"""Weld dilution tables, plots, native masks and annotated overview exports."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gst_image.analysis.weld_dilution import divided_weld_mask, line_bounds
from gst_image.image_io import load_preview


def tie_line_table(run):
    rows = [line.model_dump() for line in run.tie_lines]
    table = pd.DataFrame(rows)
    table.insert(0, "run_id", run.id)
    if run.calibration:
        for name in ("position", "depth", "height", "thickness"):
            table[f"{name}_mm"] = table[f"{name}_px"] * run.calibration.mm_per_pixel
    return table


def plot_dilution(figure, run, *, histogram=False, unit=None, bins=None):
    figure.clear()
    axis = figure.add_subplot(111)
    unit = unit or run.sampling.display_unit
    if not run.calibration:
        unit = "px"
    factor = (
        run.calibration.mm_per_pixel * (1000 if unit == "µm" else 1)
        if run.calibration and unit != "px"
        else 1
    )
    for attr, label, color in (
        ("depth_px", "Penetration", "#ef8c28"),
        ("height_px", "Reinforcement", "#00bcd4"),
    ):
        if histogram:
            values = [getattr(p, attr) * factor for p in run.tie_lines if p.status == "valid"]
            axis.hist(
                values,
                bins=bins or run.sampling.histogram_bins,
                label=label,
                color=color,
                alpha=0.6,
            )
        else:
            values = [
                getattr(p, attr) * factor if p.status == "valid" else np.nan for p in run.tie_lines
            ]
            axis.plot(
                [p.position_px * factor for p in run.tie_lines],
                values,
                color=color,
                label=label,
                picker=5,
            )
    axis.set_xlabel(f"{'Length' if histogram else 'Position along horizon'} ({unit})")
    axis.set_ylabel("Count" if histogram else f"Length ({unit})")
    axis.legend()
    status = "OUTDATED" if run.outdated else run.review_status.upper()
    axis.set_title(f"{run.name} — {status}")
    figure.tight_layout()
    return axis


def annotated_dilution_overview(image, envelope, run):
    image = image.copy()
    h, w = envelope.shape
    sx, sy = image.shape[1] / w, image.shape[0] / h
    # Divide at source resolution before nearest-neighbour display scaling.
    labels = cv2.resize(
        divided_weld_mask(envelope, run.reference),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    for value, color in ((1, (40, 140, 239)), (2, (212, 188, 0))):
        selected = labels == value
        image[selected] = (image[selected] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
    ref = run.reference
    o = np.array([ref.origin.x, ref.origin.y])
    t = np.array([ref.tangent.x, ref.tangent.y])
    n = np.array([ref.normal.x, ref.normal.y])

    def pixel(p):
        return tuple(np.rint(p * [sx, sy]).astype(int))

    limits = line_bounds(o, t, envelope.shape)
    if limits:
        cv2.line(image, pixel(o + t * limits[0]), pixel(o + t * limits[1]), (255, 255, 255), 1)
    valid = [p for p in run.tie_lines if p.status == "valid"]
    for p in valid[:: max(1, len(valid) // 40)]:
        q = np.array([p.x, p.y])
        cv2.line(image, pixel(q - n * p.height_px), pixel(q + n * p.depth_px), (240, 240, 240), 1)
    text = (
        f"Cross-sectional dilution: {run.summary.dilution_percent:.3f}% | "
        f"{'OUTDATED' if run.outdated else run.review_status}"
    )
    if run.summary.quality_flags:
        text += " | " + ", ".join(run.summary.quality_flags)
    cv2.putText(image, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(image, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    length_px = max(1, w / 10)
    end = (20 + round(length_px * sx), image.shape[0] - 25)
    cv2.line(image, (20, end[1]), end, (255, 255, 255), 3)
    scale_label = (
        f"{length_px * run.calibration.mm_per_pixel:.4g} mm"
        if run.calibration
        else f"{length_px:.4g} px"
    )
    cv2.putText(
        image, scale_label, (20, end[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    return image


def export_dilution(output, manifest, masks, *, source_image=None, runs=None):
    destination = Path(output)
    runs = manifest.dilution_runs if runs is None else runs
    if not runs:
        return destination
    if source_image is None:
        source = Path(manifest.source_path)
        if source.exists():
            source_image, _ = load_preview(source, color=True)
    for run in runs:
        root = destination / "dilution" / run.id
        root.mkdir(parents=True, exist_ok=True)
        envelope = masks[run.envelope_layer_id] > 0
        summary = {
            "run_id": run.id,
            "name": run.name,
            "review_status": run.review_status,
            "outdated": run.outdated,
            **run.summary.model_dump(mode="json"),
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.json_normalize(summary).to_csv(root / "summary.csv", index=False)
        tie_line_table(run).to_csv(root / "tie_lines.csv", index=False)
        (root / "recipe_reference.json").write_text(
            json.dumps(
                {
                    "reference": run.reference.model_dump(mode="json"),
                    "recipe": run.recipe.model_dump(mode="json"),
                    "sampling": run.sampling.model_dump(mode="json"),
                    "calibration": run.calibration.model_dump(mode="json")
                    if run.calibration
                    else None,
                    "component_seeds": [p.model_dump() for p in run.component_seeds],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "provenance.json").write_text(
            json.dumps(
                {
                    "run": run.model_dump(mode="json"),
                    "application_version": manifest.application_version,
                    "dependency_versions": manifest.dependency_versions,
                    "edits": [e.model_dump(mode="json") for e in manifest.edits],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for name, values in (
            ("envelope", envelope.astype(np.uint8)),
            ("divided_regions", divided_weld_mask(envelope, run.reference)),
            ("domain", masks[run.domain_layer_id]),
        ):
            tifffile.imwrite(root / f"{name}.tif", values, compression="zlib")
        for attr in ("depth", "height"):
            values = [getattr(p, f"{attr}_px") for p in run.tie_lines if p.status == "valid"]
            counts, edges = np.histogram(values, bins=run.sampling.histogram_bins)
            data = {"lower_px": edges[:-1], "upper_px": edges[1:], "count": counts}
            if run.calibration:
                data.update(
                    lower_mm=edges[:-1] * run.calibration.mm_per_pixel,
                    upper_mm=edges[1:] * run.calibration.mm_per_pixel,
                )
            pd.DataFrame(data).to_csv(root / f"{attr}_histogram.csv", index=False)
        for histogram, name in ((False, "profile"), (True, "histogram")):
            figure = Figure(figsize=(10, 4))
            FigureCanvasAgg(figure)
            plot_dilution(figure, run, histogram=histogram)
            figure.savefig(root / f"{name}.png", dpi=150)
        if source_image is not None:
            image = annotated_dilution_overview(source_image, envelope, run)
            if not cv2.imwrite(str(root / "annotated_panorama.png"), image):
                raise OSError("Could not write annotated weld panorama")
    return destination
