"""Command-line analysis, batch, export, and project validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from gst_image.analysis import (
    assign_particle_groups,
    build_analysis_mask,
    compute_area_fractions,
    compute_project_fractions,
    segment_particles,
    suggest_specimen_mask,
)
from gst_image.export import export_analysis
from gst_image.image_io import iter_images, load_image, sha256_file
from gst_image.models import (
    AnalysisRun,
    Calibration,
    ProjectManifest,
    SegmentationLayer,
    SegmentationRecipe,
)
from gst_image.project import (
    load_project,
    project_path,
    resolve_source,
    save_project,
    validate_project,
)


def _recipe(path: str | None) -> SegmentationRecipe:
    if path is None:
        return SegmentationRecipe()
    return SegmentationRecipe.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _progress(value: float, message: str) -> None:
    print(f"[{value:6.1%}] {message}", file=sys.stderr)


def analyze_image(
    image_path: Path,
    output: Path,
    recipe: SegmentationRecipe,
    mm_per_pixel: float | None,
    *,
    portable: bool = False,
    force: bool = False,
) -> Path:
    target = project_path(output)
    if target.exists() and not force:
        raise FileExistsError(f"Project already exists (use --force to update it): {target}")
    gray = load_image(image_path)
    specimen = suggest_specimen_mask(gray)
    analysis_mask = build_analysis_mask(gray.shape, specimen_mask=specimen)
    calibration = Calibration(mm_per_pixel=mm_per_pixel) if mm_per_pixel else None
    result = segment_particles(
        gray, analysis_mask, recipe, calibration, progress=_progress
    )
    manifest = ProjectManifest(
        name=image_path.stem,
        source_path=str(image_path.resolve()),
        source_sha256=sha256_file(image_path),
        image_width=gray.shape[1],
        image_height=gray.shape[0],
        calibration=calibration,
        recipes=[recipe],
    )
    particle_class = next(item for item in manifest.classes if item.preset == "particle")
    layer = SegmentationLayer(name="Particles", class_id=particle_class.id, kind="instances")
    domain_layer = SegmentationLayer(name="Analysis domain", kind="domain", visible=False)
    manifest.layers.append(layer)
    manifest.layers.append(domain_layer)
    result.particles = assign_particle_groups(result.particles, manifest.groups)
    manifest.particle_records[layer.id] = result.particles
    run = AnalysisRun(
        recipe=recipe,
        layer_ids=[layer.id, domain_layer.id],
        summary=result.summary,
    )
    layer.source_run_id = run.id
    manifest.runs.append(run)
    masks = {layer.id: result.labels, domain_layer.id: analysis_mask.astype("uint8")}
    root = save_project(target, manifest, masks, portable=portable)
    fractions = compute_area_fractions({layer.name: result.mask}, analysis_mask)
    export_analysis(
        root / "results",
        manifest,
        masks,
        particles=result.particles,
        fractions=fractions,
    )
    print(root)
    return root


def _calibration_table(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    result: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if "image" not in row or "mm_per_pixel" not in row:
                raise ValueError("Calibration CSV requires image and mm_per_pixel columns")
            result[row["image"]] = float(row["mm_per_pixel"])
    return result


def _cmd_analyze(args: argparse.Namespace) -> int:
    analyze_image(
        Path(args.input),
        Path(args.output),
        _recipe(args.recipe),
        args.mm_per_pixel,
        portable=args.portable,
        force=args.force,
    )
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    source = Path(args.input)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    calibrations = _calibration_table(args.calibration_csv)
    project_scale = None
    if args.calibration_project:
        calibration_manifest, _ = load_project(args.calibration_project, load_masks=False)
        if calibration_manifest.calibration is None:
            raise ValueError("Calibration project does not contain a scale calibration")
        project_scale = calibration_manifest.calibration.mm_per_pixel
    failures = 0
    for image in iter_images(source):
        scale = calibrations.get(
            image.name,
            calibrations.get(str(image), args.mm_per_pixel or project_scale),
        )
        print(f"Analyzing {image}", file=sys.stderr)
        try:
            analyze_image(
                image,
                destination / f"{image.stem}.gstproj",
                _recipe(args.recipe),
                scale,
                portable=args.portable,
                force=args.force,
            )
        except Exception as error:  # continue a batch while reporting each failure
            failures += 1
            print(f"ERROR: {image}: {error}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_export(args: argparse.Namespace) -> int:
    issues = validate_project(args.project)
    if issues:
        raise ValueError("Project validation failed: " + "; ".join(issues))
    manifest, masks = load_project(args.project)
    source = load_image(resolve_source(args.project, manifest), color=True) if args.overlay else None
    particles = [
        particle
        for records in manifest.particle_records.values()
        for particle in records
    ]
    fractions = compute_project_fractions(manifest, masks)
    export_analysis(
        args.output,
        manifest,
        masks,
        particles=particles,
        fractions=fractions,
        source_image=source,
    )
    print(Path(args.output).resolve())
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    issues = validate_project(args.project, verify_hash=not args.no_hash)
    if issues:
        for issue in issues:
            print(f"INVALID: {issue}")
        return 1
    print("Project is valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gst-image-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one micrograph")
    analyze.add_argument("input")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--recipe")
    analyze.add_argument("--mm-per-pixel", type=float)
    analyze.add_argument("--portable", action="store_true")
    analyze.add_argument("--force", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    batch = subparsers.add_parser("batch", help="Analyze every image in a folder")
    batch.add_argument("input")
    batch.add_argument("--output", required=True)
    batch.add_argument("--recipe")
    batch.add_argument("--calibration-csv")
    batch.add_argument("--calibration-project")
    batch.add_argument("--mm-per-pixel", type=float)
    batch.add_argument("--portable", action="store_true")
    batch.add_argument("--force", action="store_true")
    batch.set_defaults(handler=_cmd_batch)

    export = subparsers.add_parser("export", help="Export a saved project")
    export.add_argument("project")
    export.add_argument("--output", required=True)
    export.add_argument("--overlay", action="store_true")
    export.set_defaults(handler=_cmd_export)

    validate = subparsers.add_parser("validate-project", help="Verify project files and source hash")
    validate.add_argument("project")
    validate.add_argument("--no-hash", action="store_true")
    validate.set_defaults(handler=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, FileExistsError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
