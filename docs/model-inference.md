# Reviewed ONNX model packs

GST Image can run a locally installed model pack through **Run reviewed model
pack...** or `gst-image-cli infer-model`. Model results are always saved as
**pending review**. Inspect the native-resolution overlay and measurements, then
record the reviewer in the application or with `confirm-model-run` before using
the layer as a validated measurement result.

This page describes the ONNX-pack workflow. For local stock Cellpose-SAM v2
instances and external Cellpose fine-tuning, see
[Cellpose-SAM v2: local instance segmentation and custom training](cellpose.md).

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

## ONNX model-pack maintainer workflow

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
