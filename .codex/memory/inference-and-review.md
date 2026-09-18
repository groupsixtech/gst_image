# Model Inference and Review

Read [Repository Guidelines](../../AGENTS.md) first. This note captures the
durable contract for the optional model-inference work added in recent history;
it is deliberately narrower than the user guides in `docs/`.

## ONNX model packs

The application does not bundle or download pretrained ONNX weights. A user or
model maintainer supplies a local pack with `manifest.json` and the declared
ONNX model file. `ModelPack.open()` in
[`analysis/model_inference.py`](../../src/gst_image/analysis/model_inference.py)
validates the manifest, file presence, and optional SHA-256 before inference.

- The supported manifest contract is format version 1, `unet_2d`, one NCHW
  image input, declared NCHW/CHW outputs, input normalization, tiling, class
  mappings, and output names/types. Do not make a generic `.onnx` file appear
  compatible by guessing its outputs or preprocessing.
- The `ml` extra installs ONNX Runtime, but `create_inference_session()`
  explicitly requests `CPUExecutionProvider`. CUDA, PyTorch, and
  `onnxruntime-gpu` are not a performance path unless that core contract is
  deliberately changed and tested.
- Inference blends overlapping native-resolution tiles, clips predictions to
  the analysis domain, and uses the existing watershed/measurement path for
  particle labels. Maintain cancellation checks in the tile loop.
- Model training, ONNX export, pack assembly, and held-out validation happen
  outside GST Image. See `docs/model-inference.md` for the release gates and
  user-facing installation workflow.

## Cellpose-SAM v2

[`analysis/cellpose_inference.py`](../../src/gst_image/analysis/cellpose_inference.py)
intentionally supports only Cellpose's stock `cpsam_v2` checkpoint. It is a
local, lazy-loaded dependency; its weights are cached by Cellpose and the first
download must be explicitly allowed. The recipe records CC-BY-NC acceptance.

- Do not add a custom checkpoint selector or imply that GST Image trains or
  imports a fine-tuned Cellpose model. Custom training stays in upstream
  Cellpose until an explicit product contract supports importing it.
- GUI runs require Analysis boxes: one selected box is the scope, otherwise all
  Analysis boxes are processed. Inference uses original-resolution crops, turns
  OpenCV BGR into RGB, returns labels to source coordinates, clips them to the
  valid domain, and splits disconnected fragments before measurement.
- `CellposeInferenceRecipe` defaults to CPU; the GUI may select NVIDIA GPU when
  PyTorch exposes one. A GPU recipe must call the CUDA availability check and
  fail clearly when unavailable; it must never quietly run on CPU. A
  CUDA-compatible PyTorch wheel is selected by the workstation setup, not the
  package metadata.
- Cellpose evaluation cannot be interrupted mid-call. Check cancellation before
  and after it so a late completed result is discarded rather than persisted.

See `docs/cellpose.md` for licences, installation, parameters, and the
user-facing training discussion.

## Review, persistence, and exports

Both inference paths create result layers and a matching domain layer linked by
`source_run_id` to a `ModelInferenceRun` or `CellposeInferenceRun`. The run's
`layer_ids` must link back to precisely those layers. New model-derived layers
start `pending`; confirmation records a reviewer and confirms every layer from
that run. `confirm-model-run` supports both run types.

Project schema 4 introduced ONNX collections and schema 5 introduced Cellpose
collections; `project._migrate()` must preserve old-project meaning when these
contracts change. `validate_project()` checks run/layer ownership. Exports must
include model recipes, run provenance (including review status), and correct
domain denominators for model-derived particle summaries.

## Change and test checklist

An inference change commonly spans all of these locations:

1. Pydantic recipe/run model and core inference implementation.
2. GUI dialog/worker and its explicit scope, download/licence, device, and
   cancellation behavior.
3. CLI parser and a reopenable pending-review project.
4. Project migration/validation plus export/provenance fields.
5. Focused synthetic tests in `tests/test_model_inference.py` or
   `tests/test_cellpose_inference.py`, then project/CLI and GUI regression
   coverage.

Use synthetic image arrays and `tmp_path`; large source imagery and generated
export folders were intentionally removed from version control. Keep source
coordinates, analysis-domain clipping, label separation, review state, and
model provenance as explicit assertions.
