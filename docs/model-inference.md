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

The workflow runs the stock `cpsam_v2` model locally. The dialog preselects
**NVIDIA CUDA GPU** when PyTorch can see a CUDA device and falls back to **CPU**
when it cannot; either can be chosen per run, or set with `--device` at the CLI.
It never uploads a source image. Missing weights are fetched once from Cellpose
on demand — the GUI no longer asks per run, while the CLI still requires
`--allow-model-download`. GST Image never bundles the weights.

Stock Cellpose-SAM weights are CC-BY-NC, which the dialog states and every GUI
run records as a non-commercial acknowledgement; the CLI still requires
`--accept-cellpose-noncommercial-license` explicitly. Each run records that
acknowledgement, Cellpose/Torch versions, local weight SHA-256 and cache path,
timing, settings, the bounds of every analysed region, and attribution in the
project. Do not use this stock workflow for commercial work without separate
rights from the maintainers.

### Analysis boxes define the scope

In the GUI, Cellpose-SAM runs **only inside Analysis boxes**, never across the
whole overview. Draw at least one Analysis box first; the button reports that
requirement instead of running. Each box is evaluated as its own native-resolution
image, exactly as if that box had been opened in the Cellpose app, and its labels
are then restored to full source coordinates. Overlapping boxes are fused into one
region first, so no instance is cut along a shared edge. Tick **Selected ROI only**
with a single Analysis box selected to run just that box.

The dialog lists the boxes the run will segment, largest first, with each box's
name, pixel size and origin — check that list before pressing OK to confirm the
intended box is in scope.

Because every box is normalised on its own, a box is reproducible on its own
terms: moving a box changes its result, and a box measured here matches the same
crop measured in the Cellpose app. Exclude ROIs and the specimen mask still apply
— labels are clipped to that domain before measurement, and `analyzed_pixels`
counts only domain pixels inside the boxes.

The `infer-cellpose` CLI has no project ROIs, so it analyses the whole image.

Select **Biological cells** or **Metallography particles** before running. The
choice controls the result class name and provenance only. Metallography output
is experimental until it has passed the held-out gold-mask and expert-review
gates documented below.

### Diameter must match the working resolution

Cellpose-SAM is scale-sensitive: it rescales the region by `30 / diameter`, so the
default diameter asks the model to treat particles as roughly 30 source pixels
across. Analysis boxes run at **native** source resolution, which is often a
higher magnification than a downscaled export tested in the Cellpose app — and
the same settings then under-segment the largest particles, carving them up or
missing them entirely while small particles look fine.

Measured on `test/img/analysis-regions/26-0011_100_6_roi1.png`, which is a 0.645×
export of a region of `26-0011_100_6_stitch.jpg`:

| run | masks | area fraction | largest mask (export px) |
| --- | --- | --- | --- |
| 0.645× export, default diameter | 371 | 58.4% | 57 660 |
| native resolution, default diameter | 378 | 48.7% | 24 753 |
| native resolution, diameter `46` | 379 | 58.2% | 56 854 |

`30 / 0.645 ≈ 46`, so the dialog's Diameter defaults to **46 px** — tuned for
native-resolution Analysis boxes on stitched metallographs rather than for
Cellpose's own `30`. Dial it to 0 for "Cellpose default (30 px)", which applies no
rescaling and reproduces the Cellpose app on an image at its own scale. If a
downscaled image segments well in the Cellpose app, set Diameter to 30 divided by
that scale. A mismatch shows up as a dropped area fraction with an unchanged
particle count.

The remaining advanced controls retain Cellpose defaults: cell probability
threshold `0.0`, flow threshold `0.4`, minimum size `15 px`, and tile overlap
`0.1`. Everything the dialog does not name — normalisation, batch
size, resampling, tiling — is left at Cellpose's own defaults. Cellpose may finish
its current model evaluation before a cancellation request is observed; GST Image
discards the late result rather than saving it.

### NVIDIA CUDA GPU setup (Windows)

Install a CUDA-enabled PyTorch wheel that matches the workstation's NVIDIA driver
using the command generated by the official [PyTorch selector](https://pytorch.org/get-started/locally/),
then install GST Image's Cellpose extra:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[cellpose-gpu]"
python -c "import torch; print(torch.cuda.is_available())"
```

`cu118` is only an example; select the CUDA build compatible with the target
workstation rather than copying it blindly. The `cellpose-gpu` extra installs
Cellpose but intentionally does not choose a CUDA wheel for you. Choose **NVIDIA CUDA GPU** in the
Cellpose dialog or add `--device gpu` to `infer-cellpose`. GST Image refuses a
GPU run when PyTorch cannot see a CUDA device, rather than silently falling back
to CPU. Windows AMD GPU/ROCm Cellpose acceleration is not supported upstream.

For scripted use:

```powershell
gst-image-cli infer-cellpose input.png --output result.gstproj --modality biological `
  --device gpu --accept-cellpose-noncommercial-license --allow-model-download
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
