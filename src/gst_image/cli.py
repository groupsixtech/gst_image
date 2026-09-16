"""Command-line analysis, batch, export, and project validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from gst_image.analysis import (
    ModelPack,
    assign_particle_groups,
    build_analysis_mask,
    compute_area_fractions,
    compute_project_fractions,
    run_cellpose_inference,
    run_model_inference,
    segment_particles,
    suggest_specimen_mask,
)
from gst_image.export import export_analysis
from gst_image.image_io import iter_images, load_image, sha256_file
from gst_image.models import (
    AnalysisRun,
    Calibration,
    CellposeInferenceRecipe,
    CellposeInferenceRun,
    ClassDefinition,
    ModelInferenceRun,
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


def _class_for_name(manifest: ProjectManifest, name: str, index: int) -> ClassDefinition:
    existing = next(
        (item for item in manifest.classes if item.name.casefold() == name.casefold()), None
    )
    if existing:
        return existing
    colors = ("#66bb6a", "#42a5f5", "#ab47bc", "#ef5350", "#ffb300")
    result = ClassDefinition(name=name, color=colors[index % len(colors)])
    manifest.classes.append(result)
    return result


def infer_model_image(
    image_path: Path,
    output: Path,
    model_pack_path: Path,
    mm_per_pixel: float | None,
    *,
    portable: bool = False,
    force: bool = False,
) -> Path:
    """Create a pending-review project from an offline model pack."""
    target = project_path(output)
    if target.exists() and not force:
        raise FileExistsError(f"Project already exists (use --force to update it): {target}")
    pack = ModelPack.open(model_pack_path)
    image = load_image(image_path, color=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    domain = build_analysis_mask(gray.shape, specimen_mask=suggest_specimen_mask(gray))
    calibration = Calibration(mm_per_pixel=mm_per_pixel) if mm_per_pixel else None
    result = run_model_inference(image, domain, pack, calibration=calibration, progress=_progress)
    recipe = pack.default_recipe()
    manifest = ProjectManifest(
        name=image_path.stem,
        source_path=str(image_path.resolve()),
        source_sha256=sha256_file(image_path),
        image_width=image.shape[1],
        image_height=image.shape[0],
        calibration=calibration,
        model_inference_recipes=[recipe],
    )
    layers: list[SegmentationLayer] = []
    masks: dict[str, np.ndarray] = {}
    if result.semantic_labels is not None:
        class_map = {
            value: _class_for_name(manifest, name, value).id
            for value, name in pack.manifest.semantic_classes.items()
            if value != 0 and name.casefold() != "background"
        }
        semantic_layer = SegmentationLayer(
            name=f"{pack.manifest.model_id} phases",
            kind="multiclass",
            class_value_map=class_map,
            review_status="pending",
        )
        layers.append(semantic_layer)
        masks[semantic_layer.id] = result.semantic_labels
    if result.particle_labels is not None:
        particle_class = _class_for_name(manifest, "Particle", 0)
        particle_layer = SegmentationLayer(
            name=f"{pack.manifest.model_id} particles",
            class_id=particle_class.id,
            kind="instances",
            review_status="pending",
        )
        layers.append(particle_layer)
        masks[particle_layer.id] = result.particle_labels
        manifest.particle_records[particle_layer.id] = result.particles
    domain_layer = SegmentationLayer(
        name="Model analysis domain", kind="domain", visible=False, review_status="pending"
    )
    layers.append(domain_layer)
    masks[domain_layer.id] = domain.astype("uint8")
    run = ModelInferenceRun(
        recipe=recipe,
        layer_ids=[layer.id for layer in layers],
        summary=result.summary,
        source_bounds_px=(0, 0, image.shape[1], image.shape[0]),
        runtime_version=result.summary["runtime_version"],
    )
    for layer in layers:
        layer.source_run_id = run.id
    manifest.layers.extend(layers)
    manifest.model_inference_runs.append(run)
    root = save_project(target, manifest, masks, portable=portable)
    fractions = compute_project_fractions(manifest, masks)
    export_analysis(root / "results", manifest, masks, particles=result.particles, fractions=fractions)
    print(root)
    return root


def _cmd_infer_model(args: argparse.Namespace) -> int:
    infer_model_image(
        Path(args.input),
        Path(args.output),
        Path(args.model_pack),
        args.mm_per_pixel,
        portable=args.portable,
        force=args.force,
    )
    return 0


def _cmd_confirm_model_run(args: argparse.Namespace) -> int:
    manifest, masks = load_project(args.project)
    run = next(
        (
            item
            for item in [*manifest.model_inference_runs, *manifest.cellpose_inference_runs]
            if item.id == args.run
        ),
        None,
    )
    if run is None:
        raise ValueError(f"Model inference run was not found: {args.run}")
    run.review_status = "confirmed"
    run.reviewed_at = datetime.now(UTC)
    run.reviewer = args.reviewer
    for layer in manifest.layers:
        if layer.source_run_id == run.id:
            layer.review_status = "confirmed"
    save_project(args.project, manifest, masks)
    print(f"Confirmed model run {run.id}")
    return 0


def infer_cellpose_image(
    image_path: Path,
    output: Path,
    recipe: CellposeInferenceRecipe,
    mm_per_pixel: float | None,
    *,
    allow_model_download: bool = False,
    portable: bool = False,
    force: bool = False,
) -> Path:
    """Create a pending-review project from local Cellpose-SAM v2 inference."""
    target = project_path(output)
    if target.exists() and not force:
        raise FileExistsError(f"Project already exists (use --force to update it): {target}")
    image = load_image(image_path, color=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    domain = build_analysis_mask(gray.shape, specimen_mask=suggest_specimen_mask(gray))
    calibration = Calibration(mm_per_pixel=mm_per_pixel) if mm_per_pixel else None
    result = run_cellpose_inference(
        image,
        domain,
        recipe,
        calibration,
        allow_model_download=allow_model_download,
        progress=_progress,
    )
    output_class = "Cell" if recipe.modality == "biological" else "Particle"
    manifest = ProjectManifest(
        name=image_path.stem,
        source_path=str(image_path.resolve()),
        source_sha256=sha256_file(image_path),
        image_width=image.shape[1],
        image_height=image.shape[0],
        calibration=calibration,
        cellpose_inference_recipes=[recipe],
    )
    instance_layer = SegmentationLayer(
        name=f"Cellpose-SAM v2 {output_class.lower()}s",
        class_id=_class_for_name(manifest, output_class, 0).id,
        kind="instances",
        review_status="pending",
    )
    domain_layer = SegmentationLayer(
        name="Cellpose analysis domain", kind="domain", visible=False, review_status="pending"
    )
    run = CellposeInferenceRun(
        recipe=recipe,
        layer_ids=[instance_layer.id, domain_layer.id],
        summary=result.summary,
        source_bounds_px=result.source_bounds_px,
        region_bounds_px=result.region_bounds_px,
        cellpose_version=result.cellpose_version,
        torch_version=result.torch_version,
        model_sha256=result.model_sha256,
        model_cache_path=result.model_cache_path,
        license_acknowledged_at=datetime.now(UTC),
    )
    instance_layer.source_run_id = run.id
    domain_layer.source_run_id = run.id
    manifest.layers.extend([instance_layer, domain_layer])
    manifest.cellpose_inference_runs.append(run)
    manifest.particle_records[instance_layer.id] = result.particles
    masks = {instance_layer.id: result.labels, domain_layer.id: domain.astype("uint8")}
    root = save_project(target, manifest, masks, portable=portable)
    fractions = compute_project_fractions(manifest, masks)
    export_analysis(root / "results", manifest, masks, particles=result.particles, fractions=fractions)
    print(root)
    return root


def _cmd_infer_cellpose(args: argparse.Namespace) -> int:
    if not args.accept_cellpose_noncommercial_license:
        raise ValueError(
            "Cellpose-SAM v2 is CC-BY-NC; pass --accept-cellpose-noncommercial-license to proceed"
        )
    recipe = CellposeInferenceRecipe(
        modality=args.modality,
        device=args.device,
        diameter_px=args.diameter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        min_size_px=args.min_size,
        tile_overlap=args.tile_overlap,
        noncommercial_license_accepted=True,
    )
    infer_cellpose_image(
        Path(args.input),
        Path(args.output),
        recipe,
        args.mm_per_pixel,
        allow_model_download=args.allow_model_download,
        portable=args.portable,
        force=args.force,
    )
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

    infer_model = subparsers.add_parser(
        "infer-model", help="Run a validated local ONNX model pack; output remains pending review"
    )
    infer_model.add_argument("input")
    infer_model.add_argument("--model-pack", required=True)
    infer_model.add_argument("--output", required=True)
    infer_model.add_argument("--mm-per-pixel", type=float)
    infer_model.add_argument("--portable", action="store_true")
    infer_model.add_argument("--force", action="store_true")
    infer_model.set_defaults(handler=_cmd_infer_model)

    infer_cellpose = subparsers.add_parser(
        "infer-cellpose", help="Run local stock Cellpose-SAM v2; output remains pending review"
    )
    infer_cellpose.add_argument("input")
    infer_cellpose.add_argument("--output", required=True)
    infer_cellpose.add_argument("--modality", choices=("biological", "metallography"), default="biological")
    infer_cellpose.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    infer_cellpose.add_argument("--mm-per-pixel", type=float)
    infer_cellpose.add_argument("--diameter", type=float)
    infer_cellpose.add_argument("--cellprob-threshold", type=float, default=0.0)
    infer_cellpose.add_argument("--flow-threshold", type=float, default=0.4)
    infer_cellpose.add_argument("--min-size", type=int, default=15)
    infer_cellpose.add_argument("--tile-overlap", type=float, default=0.1)
    infer_cellpose.add_argument("--allow-model-download", action="store_true")
    infer_cellpose.add_argument("--accept-cellpose-noncommercial-license", action="store_true")
    infer_cellpose.add_argument("--portable", action="store_true")
    infer_cellpose.add_argument("--force", action="store_true")
    infer_cellpose.set_defaults(handler=_cmd_infer_cellpose)

    confirm_model = subparsers.add_parser(
        "confirm-model-run", help="Record expert review for a pending model result"
    )
    confirm_model.add_argument("project")
    confirm_model.add_argument("--run", required=True)
    confirm_model.add_argument("--reviewer", required=True)
    confirm_model.set_defaults(handler=_cmd_confirm_model_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError, FileExistsError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
