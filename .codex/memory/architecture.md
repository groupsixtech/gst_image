# Architecture and Persistence

Read [Repository Guidelines](../../AGENTS.md) first for conventions and commands.
This note maps the codebase for agents making architecture-level changes.

## Layering and entry points

GST Image is a Windows-focused desktop/CLI application for traceable weld-image
analysis. The intended dependency direction is:

```text
PySide6 GUI (app/gst_image_app) ─┐
CLI (src/gst_image/cli.py) ──────┼─> core package (src/gst_image)
                                └─> project/export/image I/O
```

Keep reusable scientific behavior in `src/gst_image/`; `app/gst_image_app/`
should coordinate widgets, canvas input, background work, and persistence—not
reimplement algorithms. `pyproject.toml` exposes `gst-image` for
`gst_image_app.main:main` and `gst-image-cli` for `gst_image.cli:main`.

| Area | Primary code | Responsibility |
| --- | --- | --- |
| Data contracts | `models.py` | Pydantic persisted models and small runtime result dataclasses. |
| Particle pipeline | `analysis/particles.py`, `preprocess.py` | Segmentation, labels, morphology measurements. |
| Domains and ROIs | `analysis/masks.py` | Rasterization and valid analysis domain construction. |
| Region ML | `analysis/regions.py` | Stroke-trained multi-class overview classification. |
| ONNX model inference | `analysis/model_inference.py` | Validated local model packs and tiled, native-resolution CPU inference. |
| Cellpose inference | `analysis/cellpose_inference.py` | Local stock `cpsam_v2` instance inference within Analysis boxes. |
| Grouping/fractions | `analysis/groups.py`, `fractions.py` | Post-analysis particle groups and area estimates. |
| Files/results | `project.py`, `export.py`, `image_io.py` | Project lifecycle, output files, and large-image loading. |
| Desktop UI | `mainwindow.py`, `canvas.py`, `workers.py` | Application state, native-coordinate canvas, threaded jobs. |

## Core runtime flows

### Particle analysis

1. GUI or CLI loads an image with `image_io`; CLI first makes a specimen mask.
2. `build_analysis_mask()` combines the specimen domain, include/analysis ROIs,
   and exclude ROIs.
3. `segment_particles()` returns binary mask, integer instance labels, records,
   summary, and a corrected preview.
4. The caller creates an instance `SegmentationLayer`, matching domain layer,
   `AnalysisRun`, and `particle_records[layer.id]` entry.
5. `save_project()` persists the manifest and masks; `export_analysis()` writes
   user-facing result files.

### Assisted region classification

The GUI maintains source-coordinate training strokes. It generates an overview,
rescales strokes to it, runs `classify_regions()` in a worker, then maps labels
back to source dimensions as a multiclass layer. It is separate from particle
preview/selected-ROI analysis.

### Optional model inference

`ModelPack.open()` validates a separately installed ONNX pack's manifest and
weights hash before `run_model_inference()` blends overlapping native-resolution
tiles. The current session explicitly uses `CPUExecutionProvider`; do not imply
that installing CUDA or `onnxruntime-gpu` accelerates this path. Its semantic
and/or particle layers, plus their domain layer, are linked to one pending-review
`ModelInferenceRun`.

`run_cellpose_inference()` lazy-loads only the stock `cpsam_v2` model. GUI runs
operate on project Analysis boxes (one selected box, otherwise all boxes), use
the normal valid domain, restore each crop to source coordinates, and then
measure labels. It converts OpenCV BGR input to RGB for Cellpose. Preserve the
explicit first-download permission, CC-BY-NC acknowledgement, GPU availability
check, and cancellation behavior; a cancelled run must not save a late result.
See [model inference and review](inference-and-review.md) for the complete
contract and change checklist.

### GUI execution

`MainWindow` owns mutable application state (manifest, active layer, masks,
source/preview images, and canvas state). Long operations are dispatched through
`FunctionWorker`/`QThreadPool`; core operations accept progress and cancellation
callbacks. Do not run segmentation or classification on the UI thread.

## Coordinate and image invariants

- Source-image pixels are canonical. ROI points, measurements, mask edits, and
  training strokes are stored in native source coordinates.
- Displayed overviews and selected ROI crops are derived views. A selected-ROI
  result must be positioned back into a full-source-sized mask before saving or
  overlaying.
- Images use OpenCV BGR color ordering; core grayscale operations use 8-bit
  arrays. Label images are positive `int32` instance/class IDs; zero is background.
- Analysis domain is `specimen ∩ (included ROIs, when any) − excluded ROIs`.
  Never calculate fraction or border status against a different domain.

## Persistent project contract

A `.gstproj` is a directory. `project.json` is the Pydantic `ProjectManifest`;
`masks/<layer-id>.tif` contains zlib-compressed arrays; `previews/source.jpg`
is a reduced source preview; `results/` can hold exported analysis. Portable
projects also copy the source to `source/`.

`PROJECT_SCHEMA_VERSION` is 5. `load_project()` migrates older group data to
layer-scoped `particle_groupings` and initializes optional ONNX/Cellpose inference
collections; `SegmentationRecipe` independently migrates a
legacy one-sided `manual_threshold`. Add a forward migration in `_migrate()` and
tests whenever changing persisted meaning. `ProjectManifest` forbids unknown
fields, so schema changes require explicit model updates.

Projects record source SHA-256, dimensions, dependency versions, source revision,
recipes, rule-based/ONNX/Cellpose runs, edits, layers, and particle records. `validate_project()` detects
missing masks, missing/changed source images, and broken grouping references.
`relink_source()` requires identical dimensions and invalidates result-dependent
state by default; preserve that traceability behavior.

## Source and result ownership

Each result layer can reference `source_run_id`; each particle grouping is tied
to `source_layer_id`. Domain layers use the same `scope_roi_ids` as the analysis
layer. When deleting or replacing a layer/ROI, clean up dependent masks, records,
groups, and runs consistently—see the GUI tests covering ROI deletion and layer
selection. For ONNX and Cellpose, every run-owned result and domain layer must
both appear in `run.layer_ids` and point back to that run with `source_run_id`;
they start as `review_status="pending"` and are confirmed together. This
bidirectional ownership is validated and exported as provenance.
