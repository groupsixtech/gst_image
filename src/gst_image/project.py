"""Versioned `.gstproj` directory persistence and source verification."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image

from gst_image import __version__
from gst_image.image_io import load_preview, sha256_file
from gst_image.models import PROJECT_SCHEMA_VERSION, ProjectManifest

MANIFEST_NAME = "project.json"
DEPENDENCIES = (
    "numpy",
    "opencv-python",
    "Pillow",
    "pydantic",
    "PySide6",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "tifffile",
    "onnxruntime",
    "cellpose",
    "torch",
)


def project_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".gstproj":
        target = target.with_name(target.name + ".gstproj")
    return target


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _migrate(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("schema_version", 1))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Project schema {version} is newer than this application supports")
    if version < 3 and payload.get("groups") and not payload.get("particle_groupings"):
        instance_layers = [
            layer for layer in payload.get("layers", []) if layer.get("kind") == "instances"
        ]
        if not instance_layers:
            instance_layers = [{"id": None}]
        groupings = []
        active: dict[str, str] = {}
        for layer in instance_layers:
            grouping_id = str(uuid.uuid4())
            groupings.append(
                {
                    "id": grouping_id,
                    "name": "Migrated particle grouping",
                    "source_layer_id": layer.get("id"),
                    "groups": [dict(group) for group in payload["groups"]],
                }
            )
            if layer.get("id"):
                active[layer["id"]] = grouping_id
        payload["particle_groupings"] = groupings
        payload["active_particle_groupings"] = active
        payload["groups"] = []
    if version < 4:
        # Model inference was introduced as optional metadata. Older projects have no
        # model-derived results, so empty collections retain their exact meaning.
        payload.setdefault("model_inference_recipes", [])
        payload.setdefault("model_inference_runs", [])
        for layer in payload.get("layers", []):
            layer.setdefault("review_status", "not_required")
    if version < 5:
        payload.setdefault("cellpose_inference_recipes", [])
        payload.setdefault("cellpose_inference_runs", [])
    # SegmentationRecipe's pre-validator translates legacy manual_threshold values wherever
    # recipes occur, including recipes embedded in run history.
    payload["schema_version"] = PROJECT_SCHEMA_VERSION
    return payload


def save_project(
    path: str | Path,
    manifest: ProjectManifest,
    masks: Mapping[str, np.ndarray] | None = None,
    *,
    portable: bool = False,
    source_override: str | Path | None = None,
) -> Path:
    root = project_path(path)
    masks_dir = root / "masks"
    results_dir = root / "results"
    previews_dir = root / "previews"
    for directory in (root, masks_dir, results_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest.schema_version = PROJECT_SCHEMA_VERSION
    manifest.application_version = __version__
    manifest.dependency_versions = dependency_versions()
    manifest.updated_at = datetime.now(UTC)
    source_for_preview = Path(source_override or manifest.source_path)
    if source_for_preview.exists():
        preview, _ = load_preview(source_for_preview, (1200, 800))
        cv2.imwrite(str(previews_dir / "source.jpg"), preview)
    if masks:
        layer_by_id = {layer.id: layer for layer in manifest.layers}
        for layer_id, values in masks.items():
            relative = Path("masks") / f"{layer_id}.tif"
            tifffile.imwrite(
                root / relative,
                np.asarray(values),
                compression="zlib",
                metadata={"axes": "YX"},
            )
            if layer_id in layer_by_id:
                layer_by_id[layer_id].mask_path = relative.as_posix()
    if portable:
        source = source_for_preview
        if not source.exists():
            raise FileNotFoundError(f"Cannot create portable project; source is missing: {source}")
        portable_dir = root / "source"
        portable_dir.mkdir(exist_ok=True)
        destination = portable_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        manifest.portable_source = destination.relative_to(root).as_posix()
    temporary = root / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(root / MANIFEST_NAME)
    return root


def load_project(
    path: str | Path, *, load_masks: bool = True
) -> tuple[ProjectManifest, dict[str, np.ndarray]]:
    root = project_path(path)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Project manifest not found: {manifest_path}")
    payload = _migrate(json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest = ProjectManifest.model_validate(payload)
    masks: dict[str, np.ndarray] = {}
    if load_masks:
        for layer in manifest.layers:
            if layer.mask_path:
                mask_path = root / layer.mask_path
                if mask_path.exists():
                    masks[layer.id] = tifffile.imread(mask_path)
    return manifest, masks


def resolve_source(root: str | Path, manifest: ProjectManifest) -> Path:
    project_root = project_path(root)
    if manifest.portable_source:
        portable = project_root / manifest.portable_source
        if portable.exists():
            return portable
    source = Path(manifest.source_path)
    if source.exists():
        return source
    return source


def validate_project(path: str | Path, *, verify_hash: bool = True) -> list[str]:
    root = project_path(path)
    manifest, _ = load_project(root, load_masks=False)
    issues: list[str] = []
    source = resolve_source(root, manifest)
    if not source.exists():
        issues.append(f"Source image is missing: {source}")
    elif verify_hash:
        actual = sha256_file(source)
        if actual != manifest.source_sha256:
            issues.append(
                "Source SHA-256 differs from the recorded revision; relink or create a new source revision"
            )
    for layer in manifest.layers:
        if layer.mask_path and not (root / layer.mask_path).exists():
            issues.append(f"Layer mask is missing: {layer.mask_path}")
    layer_ids = {layer.id for layer in manifest.layers}
    groupings = {grouping.id: grouping for grouping in manifest.particle_groupings}
    for grouping in groupings.values():
        if grouping.source_layer_id and grouping.source_layer_id not in layer_ids:
            issues.append(
                f"Particle grouping {grouping.name!r} references a missing layer: "
                f"{grouping.source_layer_id}"
            )
    for layer_id, grouping_id in manifest.active_particle_groupings.items():
        grouping = groupings.get(grouping_id)
        if layer_id not in layer_ids:
            issues.append(f"Active particle grouping references a missing layer: {layer_id}")
        elif grouping is None:
            issues.append(f"Active particle grouping is missing: {grouping_id}")
        elif grouping.source_layer_id != layer_id:
            issues.append(
                f"Active particle grouping {grouping.name!r} belongs to a different layer"
            )
    model_runs = {
        **{run.id: run for run in manifest.model_inference_runs},
        **{run.id: run for run in manifest.cellpose_inference_runs},
    }
    for run in model_runs.values():
        for layer_id in run.layer_ids:
            if layer_id not in layer_ids:
                issues.append(
                    f"Model inference run {run.id} references a missing layer: {layer_id}"
                )
                continue
            layer = next(layer for layer in manifest.layers if layer.id == layer_id)
            if layer.source_run_id != run.id:
                issues.append(
                    f"Model inference run {run.id} does not own layer {layer.name!r}"
                )
    return issues


def relink_source(
    manifest: ProjectManifest,
    new_source: str | Path,
    *,
    invalidate_results: bool = True,
) -> ProjectManifest:
    """Create a verified source revision and invalidate results tied to old pixels."""
    source = Path(new_source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as image:
        if image.size != (manifest.image_width, manifest.image_height):
            raise ValueError(
                "Relinked image dimensions differ from the project; create a new project instead"
            )
    manifest.source_path = str(source)
    manifest.source_sha256 = sha256_file(source)
    manifest.source_revision += 1
    manifest.portable_source = None
    if invalidate_results:
        manifest.layers.clear()
        manifest.runs.clear()
        manifest.model_inference_runs.clear()
        manifest.model_inference_recipes.clear()
        manifest.cellpose_inference_runs.clear()
        manifest.cellpose_inference_recipes.clear()
        manifest.particle_records.clear()
        manifest.particle_groupings.clear()
        manifest.active_particle_groupings.clear()
    return manifest
