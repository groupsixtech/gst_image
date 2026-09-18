# Cellpose-SAM v2: local instance segmentation and custom training

GST Image can run the stock Cellpose-SAM v2 (`cpsam_v2`) model locally to turn a
2-D image into an **instance-label layer**: each detected cell or particle gets its own label.
That makes the normal GST Image workflow available afterwards: inspect the overlay, correct
isolated masks, split or merge instances, measure them, group them, export them, and record a
reviewer.

This is distinct from threshold-based particle segmentation and assisted region classification.
Cellpose finds individual objects; it does not make a Weld/HAZ/Base phase map. Choose
**Biological cells** or **Metallography particles (experimental)** before a run. That choice
sets the result class and provenance; it does not switch the stock model or train it for a new
material.

> **Current GST Image limit:** the application runs only the stock `cpsam_v2` checkpoint. It
> cannot train, fine-tune, import, or select a custom Cellpose checkpoint. [Custom Cellpose
> training](#custom-cellpose-training-outside-gst-image) is performed in Cellpose itself, outside
> GST Image.

## Before the first run

Install the optional runtime in the same virtual environment as GST Image:

```powershell
python -m pip install -e ".[cellpose]"
```

The first run downloads the stock weights to Cellpose's local cache; GST Image never uploads the
source image and does not bundle the weights. The GUI permits that one-time weight download when
you start the run. The CLI requires `--allow-model-download` because it has no interactive dialog.

The stock Cellpose-SAM weights are **CC-BY-NC**. The GUI records a non-commercial acknowledgement
with every run; the CLI requires `--accept-cellpose-noncommercial-license`. Do not use this stock
workflow for commercial work unless you have separately obtained the required rights from the
Cellpose maintainers.

For a CUDA-capable NVIDIA GPU, first install the PyTorch wheel appropriate for the workstation's
driver from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install
the GST Image extra:

```powershell
python -m pip install -e ".[cellpose-gpu]"
python -c "import torch; print(torch.cuda.is_available())"
```

The `cellpose-gpu` extra intentionally does not choose a CUDA wheel. The dialog selects **NVIDIA
CUDA GPU** when PyTorch can see one, otherwise it selects **CPU**. A requested GPU run fails rather
than silently changing to CPU. Windows AMD/ROCm acceleration is not supported by this workflow.

## Run Cellpose in GST Image

1. Open a source image, then draw one or more **Analysis boxes** around representative regions.
   Cellpose has no preview mode in GST Image and never runs over the unboxed overview.
2. Add **Exclude boxes** over scale bars, text, seams, and other invalid content. The specimen mask
   and exclusions still limit the final analysis domain.
3. Click **Run Cellpose-SAM v2...** in the Analysis panel. The dialog lists every Analysis box it
   will process, largest first, including its native pixel dimensions and origin. Check that list
   before proceeding.
4. Select the result type and device, enter an initial diameter, and leave the other controls at
   their defaults for the first comparison. If exactly one Analysis box is selected in the ROIs
   list, only that box is run; otherwise all Analysis boxes are run.
5. Inspect the saved, native-resolution instance overlay. Turn on **Colour each particle
   individually** to find unwanted merges or splits. Use the normal mask brush/eraser and
   split/merge tools only for isolated, understood defects; retune or choose another method when
   the problem is systematic.
6. The result remains **pending review**. With either its instance or matching domain layer
   selected, click **Confirm selected model result...** in **Layers** and enter the reviewer once
   the overlay, scope, count, area fraction, and size distribution are acceptable. Save the
   project and retain its exported provenance with reported measurements.

Each Analysis box is passed to Cellpose as a separate native-resolution crop and restored to
source coordinates. Overlapping boxes are fused before inference so an object is not segmented
twice at a shared edge. After inference GST Image clips labels to the valid domain and splits any
disconnected fragments before measurement. Thus `analyzed_pixels` is only the valid pixels inside
the processed boxes, not the entire overview.

## Command-line usage

`infer-cellpose` creates a new pending-review `.gstproj`. Unlike the GUI, it has no project ROIs:
it runs against the whole image after GST Image's specimen-domain suggestion. Use the GUI when the
scope needs hand-drawn Analysis or Exclude boxes.

```powershell
gst-image-cli infer-cellpose input.tif --output cellpose-result.gstproj `
  --modality biological --device cpu --diameter 30 `
  --accept-cellpose-noncommercial-license --allow-model-download
```

Use `--mm-per-pixel` when calibrated sizes are needed, `--portable` to copy the source into the
new project, and `--force` only when replacing an existing output project is intentional. Run
`gst-image-cli infer-cellpose --help` for the complete command contract.

## What the controls mean

GST Image passes the settings below directly to Cellpose and otherwise leaves Cellpose's own
evaluation defaults in effect. It converts the source BGR image to RGB and provides its first
three channels to the model.

| Control | Default | Effect and tuning direction |
| --- | --- | --- |
| **Diameter (px)** | GUI: 46; CLI: unset | Typical object diameter at the **native source resolution**. Cellpose rescales by `30 / diameter`. Use `0` in the GUI or omit `--diameter` in the CLI to leave Cellpose's default in control. Increase it when the native objects are larger than the scale that segmented well; decrease it when they are smaller. |
| **Cell probability threshold** | `0.0` | Pixels above this model score form candidate objects. Lower it to recover missed, weak objects; raise it to suppress false positives in dim or ambiguous background. |
| **Flow threshold** | `0.4` | Maximum permitted disagreement between a recovered object's flows and model-predicted flows. Raise it when valid objects are rejected; lower it when unstable or implausible shapes are accepted. |
| **Minimum mask size (px)** | `15` | Removes tiny masks after inference. Use it only for features below the measurement definition, not as the primary way to hide widespread false positives. |
| **Tile overlap** | `0.1` | Overlap used while Cellpose evaluates tiled input. Keep the default initially; change only after verifying a repeatable tile-boundary issue. |

Cellpose-SAM is fairly size tolerant, but its default scale is centred near 30 px. GST Image's GUI
uses 46 px as its initial diameter because its native stitched-metallograph test crop was 1/0.645
larger than a downscaled crop that worked at 30 px. That is an initial metallography convenience,
not a universal physical size. If a downscaled image at scale `s` segmented correctly in Cellpose,
try `30 / s` for the same native source crop. For example, a 0.645x export suggests `30 / 0.645`,
or about 46 px. Verify the proposed diameter on representative small, typical, large, touching,
and difficult objects before reporting values.

## A defensible tuning and review loop

1. Start with one clean, representative Analysis box and the default probability, flow, size, and
   overlap settings. Set diameter from an observed object size in native pixels.
2. Compare the same checkpoints after each single change: typical objects, smallest reportable
   objects, largest objects, touching objects, lighting extremes, and exclusion boundaries.
3. Correct the parameter that matches the failure: scale for systematic size mismatch, cell
   probability for missing/background candidates, flow for shape-quality acceptance, or minimum
   size for genuinely irrelevant specks. Do not use sparse manual corrections to disguise a
   systematic model failure.
4. Run an independent box or held-out image before confirming the result. Review the domain as
   well as labels: objects clipped by an exclusion or box edge can remain valid fraction pixels
   but affect size statistics as border-touching particles.
5. Export the masks, `recipes.json`, `provenance.json`, and summary together. GST Image records
   the Cellpose and Torch versions, settings, model-cache path and SHA-256 when available,
   analysed bounds, timing, licence acknowledgement, and reviewer status.

## Custom Cellpose training outside GST Image

Use custom training when the stock model repeatedly fails for a stable image style even after
appropriate cropping and tuning. It is a separate Cellpose workflow, not a button in GST Image.
The upstream [Cellpose GUI guide](https://cellpose.readthedocs.io/en/latest/gui.html#training-your-own-cellpose-model)
and [training reference](https://cellpose.readthedocs.io/en/latest/train.html) are the authoritative
instructions; the procedure below keeps the training data and validation scientifically useful.

### Prepare labels and splits

1. Split by independent source image, specimen, acquisition session, or batch **before** creating
   crops. Do not put adjacent tiles from one source in both training and test sets; that produces
   overoptimistic validation.
2. For each source image, make an instance-label mask of exactly the same width and height:
   background is `0` and each separate object is a positive integer (`1`, `2`, ...). Objects must
   not overlap. GST Image exports instance masks as TIFF, but rename/place each reviewed export
   beside its matching source carefully and verify alignment before using it as a Cellpose label.
3. Include the real variation that matters: small and large objects, touching objects, weak
   contrast, etching or illumination changes, seams, and common artifacts. Exclude invalid image
   areas rather than labelling annotations or scale bars as objects.
4. Reserve a held-out test folder that is never used to choose training settings. Save the data
   split, annotation rules, source identifiers, Cellpose version, checkpoint, and exact command.

### Fine-tune with upstream Cellpose

For interactive, human-in-the-loop work, install Cellpose's GUI in a separate environment and run:

```powershell
python -m pip install "cellpose[gui]"
python -m cellpose
```

Open images from a folder of one coherent image style, run the built-in
`cpsam` model, fix masks, save the resulting `_seg.npy`, then choose **Models > Train new model...**
(`Ctrl+T`). Start from the built-in `cpsam` model and keep all annotations intended for one custom
model in the same folder. Cellpose uses the labelled images in that folder during training; moving
or omitting earlier labels changes their weighting in later training.

For a scripted TIFF dataset, upstream Cellpose expects matching names such as:

```text
train/field_001.tif
train/field_001_masks.tif
test/field_101.tif
test/field_101_masks.tif
```

Its documented fine-tuning command is:

```powershell
python -m cellpose --train --dir C:\cellpose-data\train --test_dir C:\cellpose-data\test `
  --learning_rate 0.00001 --weight_decay 0.1 --n_epochs 100 --train_batch_size 1
```

Use absolute paths. Cellpose saves the checkpoint in the training directory's `models/` folder.
GPU training is generally much faster, but do not let speed replace held-out evaluation. Because
Cellpose normalizes each training image and crop independently, test on full images or crops that
match the intended deployment scale and normalization, not only on attractive training-style
patches.

### Validate and deploy deliberately

Review instance overlap, count, area fraction, size distribution, and the hard cases on held-out
sources. For GST Image model-pack releases, the repository's default quantitative gates are macro
Dice at least 0.90, instance F1 at IoU 0.5 at least 0.85, area-fraction error no more than five
percentage points, and median equivalent-radius relative error no more than 10%, followed by
expert review. Those are sensible starting checks for a Cellpose experiment as well, but passing
them does not make a custom Cellpose checkpoint runnable in GST Image.

Use the custom checkpoint in Cellpose or another compatible deployment environment. Supporting a
custom Cellpose checkpoint in GST Image would require a product change with an explicit model
selection, provenance, licence, validation, and review path; do not replace the stock cache file
to work around that boundary.
