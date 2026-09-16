# Reviewed ONNX model packs

GST Image can run a locally installed model pack through **Run reviewed model
pack…** or `gst-image-cli infer-model`. Model results are always saved as
**pending review**. Inspect the native-resolution overlay and measurements, then
record the reviewer in the application or with `confirm-model-run` before using
the layer as a validated measurement result.

The base application does not require ONNX Runtime. Install the optional runtime
on Windows with:

```powershell
python -m pip install -e ".[ml]"
```

## Pack layout

```text
etched-optical-v1/
  manifest.json
  model.onnx
```

`manifest.json` is the portable contract. It records the model identity and
SHA-256, RGB/BGR/gray input normalization, native-resolution tiling, ONNX output
names, semantic class values, dataset summary, validation metrics, compatible
application version, licence, and attribution. The model file hash is checked
before inference.

The ONNX graph must have one NCHW image input. Each declared output must be NCHW
or CHW at the pack tile size:

- `semantic`: one channel per class, in the same order as `semantic_classes`.
  Declare `softmax` activation for probability output; use `none` for logits.
- `particle_foreground`: one probability channel, or one logit channel with
  `sigmoid` activation declared.
- `particle_boundary`: optional probability channel, or a logit channel with
  `sigmoid` activation, used as the watershed landscape.

The core blends overlapping tile probabilities, clips all predictions to the
project's analysis domain, and retains the existing watershed and particle
measurement path for particle labels.

## Local Cellpose-SAM v2 instances

GST Image also provides an optional **Run Cellpose-SAM v2…** workflow for 2D
biological-cell or experimental metallography-particle instances. It creates a
pending-review instance layer, so the normal overlay, brush, split/merge,
grouping, measurement, export, and reviewer-confirmation workflow still applies.
Cellpose does not create a phase-map layer.

Install the separate runtime when needed:

```powershell
python -m pip install -e ".[cellpose]"
```

The workflow runs the stock `cpsam_v2` model locally on CPU. It never uploads a
source image. If the model is not already in Cellpose's local cache, the user
must explicitly allow a one-time weight download in the dialog (or supply
`--allow-model-download` at the CLI); GST Image never bundles the weights.

Stock Cellpose-SAM weights are CC-BY-NC. The dialog requires a non-commercial
use acknowledgement, and every run records the acknowledgement, Cellpose/Torch
versions, local weight SHA-256 and cache path, timing, settings, source bounds,
and attribution in the project. Do not use this stock workflow for commercial
work without separate rights from the maintainers.

Select **Biological cells** or **Metallography particles** before running. The
choice controls the result class name and provenance only. Metallography output
is experimental until it has passed the held-out gold-mask and expert-review
gates documented below. The current specimen/include/exclude analysis domain is
always enforced: Cellpose labels are restored in full source coordinates and
clipped to that domain before measurement.

The advanced controls retain Cellpose defaults: automatic diameter, cell
probability threshold `0.0`, flow threshold `0.4`, minimum size `15 px`, and
tile overlap `0.1`. Cellpose may finish its current model evaluation before a
cancellation request is observed; GST Image discards the late result rather
than saving it.

For scripted use:

```powershell
gst-image-cli infer-cellpose input.png --output result.gstproj --modality biological `
  --accept-cellpose-noncommercial-license --allow-model-download
```

## Maintainer workflow

Train outside the desktop application on reviewed source-image groups, not
random adjacent tiles. Keep source groups separate for training, validation, and
held-out testing. Each release candidate needs semantic masks for reportable
phases and individual labels for particles, including difficult etching,
illumination, stitching, scale-bar, and small/touching-particle examples.

Compare held-out predictions against the threshold/watershed and random-forest
baselines. The default gates are macro Dice at least 0.90, instance F1 at IoU
0.5 at least 0.85, area-fraction error no more than five percentage points, and
median equivalent-radius relative error no more than 10%, followed by expert
overlay/distribution review. Use `gst_image.evaluation.evaluate_model_release`
to calculate the quantitative gate; it deliberately reports that expert review
is still required. Record the final values in `validation_metrics`.
