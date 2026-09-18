# GST Image

GST Image is a Windows-focused desktop and command-line application for traceable
segmentation and measurement of weld macrographs and microstructures. It supports
particle/phase area fraction, assisted Weld/HAZ/Base classification, calibrated line
measurements, per-particle morphology, radius/circularity grouping, and reopenable projects.

The reported volume fraction is the two-dimensional segmented area fraction used as a
stereological estimate. Equivalent particle radii are radii of equal-area circles on the
observed section; neither value is a direct three-dimensional reconstruction.

## Install

GST Image is Windows-focused and supports Python 3.12 through 3.14. Use 64-bit
Python on a 64-bit Windows installation. The base application, rule-based
segmentation, assisted region classification, and export do not need an NVIDIA
GPU, CUDA, PyTorch, Cellpose, or ONNX Runtime.

### Base application

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-lock.txt` records the exact Windows/Python 3.14 environment used for the
verified release when a fully pinned installation is required.

### Optional reviewed ONNX model packs (CPU)

Install the `ml` extra to use **Run reviewed model pack...** or
`gst-image-cli infer-model`:

```powershell
python -m pip install -e ".[ml]"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

This installs the CPU `onnxruntime` package. GST Image currently creates ONNX sessions with
`CPUExecutionProvider`, so an NVIDIA driver, CUDA, cuDNN, PyTorch, and `onnxruntime-gpu` are **not
required and will not accelerate ONNX-pack inference in this release**. ONNX Runtime's Windows
wheel requires the Microsoft Visual C++ 2019 runtime; install the current x64
[Visual C++ Redistributable](https://aka.ms/vc14/vc_redist.x64.exe) if it is absent or ONNX Runtime
reports a missing MSVC DLL. The extra supplies the runtime, not a model: obtain a reviewed model
pack containing `manifest.json` and `model.onnx` as described in the
[ONNX model-pack guide](docs/model-inference.md).

### ONNX model weights: download, install, and validate

An ONNX model pack is required to run `infer-model`; GST Image does not bundle pretrained ONNX
weights and does not download them on demand. A pack is a local directory, not a package to install
with `pip`:

```text
model-packs/
  approved-pack/
    manifest.json
    model.onnx
```

`manifest.json` is required alongside the weights. It declares the model identity, input channel
order and normalization, tile geometry, output tensors, semantic class values, and
validation/provenance; it should also record the model SHA-256. GST Image currently accepts its
documented `unet_2d` NCHW semantic and/or
particle-output contract. A generic Hugging Face ONNX file, detector, classifier, or transformer
is **not** directly loadable merely because it has an `.onnx` extension.

Find candidate pretrained weights on the [Hugging Face ONNX model search](https://huggingface.co/models?library=onnx), then read the model card, licence, training-data
description, input/output definition, and held-out metrics before downloading. Prefer a publisher
who supplies a GST Image-compatible pack. Otherwise, an owner must first verify that the ONNX graph
meets the contract, create a correct `manifest.json`, and validate the result; do not invent output
names, normalization, or class mappings to make an unrelated model appear compatible.

To download a reviewed public or authorized private Hugging Face repository into a local pack
directory, install the Hub CLI in the active virtual environment and pin a reviewed revision:

```powershell
python -m pip install --upgrade huggingface_hub
hf auth login                         # required for private or gated repositories
hf download ORG_OR_USER/MODEL_REPO --dry-run
hf download ORG_OR_USER/MODEL_REPO --revision COMMIT_OR_TAG --local-dir .\model-packs\approved-pack
Get-FileHash .\model-packs\approved-pack\model.onnx -Algorithm SHA256
gst-image-cli infer-model input.tif --model-pack .\model-packs\approved-pack --output model-result.gstproj
```

Replace the uppercase placeholders with the repository and exact revision approved for the work.
The `--dry-run` step reveals download size before transfer. Record the resulting revision and
SHA-256 in the pack's `manifest.json` and analysis record. For a gated model, accept its terms in
the browser before `hf auth login`; never place an access token in a project, model pack, or shell
history.

For a custom model, train and export a compatible `unet_2d` ONNX graph outside GST Image, create
the complete pack manifest, calculate/record the SHA-256, and validate it on held-out source-image
groups before loading it. This is an integration and validation step, not a file rename. The
minimum release gates and required manifest fields are in the [ONNX model-pack guide]
(docs/model-inference.md).

For metallographic micrographs, the most applicable training data is usually your own
expert-annotated images acquired with the same preparation, etching, microscope, resolution, and
measurement definition. Useful public starting points to investigate - not drop-in GST Image model
packs - include:

| Dataset | Potential fit | Important limitation |
| --- | --- | --- |
| [MetalDAM](https://huggingface.co/datasets/Voxel51/OD_MetalDAM) | SEM additive-manufacturing microstructures with semantic masks for matrix, austenite, martensite/austenite, precipitates, and defects. | Class definitions, alloys, magnifications, and annotations may not match this project's measurement definition. |
| [EMVista](https://huggingface.co/datasets/InnovatorLab/EMVista) | Electron-microscopy microstructure benchmark with instance-level annotations; useful for investigating instance-boundary training/evaluation. | Confirm downloadable label format, licence, and domain match before using it for training. |
| [SWXD-derived weld-defect data](https://huggingface.co/datasets/AI4Manufacturing/103) | Weld-defect masks from digitized X-ray films. | Appropriate only for radiographic weld-defect work, not optical/SEM phase segmentation. |

Keep samples from the same specimen, image, session, or stitch together in one split. Public data
can help initialize a training experiment, but it cannot replace held-out validation on the target
material and imaging workflow.

### Optional Cellpose-SAM v2 (CPU)

Install the Cellpose extra to use local stock Cellpose-SAM instance segmentation on the CPU:

```powershell
python -m pip install -e ".[cellpose]"
python -c "import cellpose, torch; print(cellpose.__version__); print(torch.__version__); print(torch.cuda.is_available())"
```

Cellpose requires PyTorch; this installation reuses an existing PyTorch install or installs one in
the active virtual environment. `False` from `torch.cuda.is_available()` is expected unless you
have deliberately installed a CUDA-enabled PyTorch wheel. The stock model weights download to
Cellpose's local cache only when an inference run first needs them; source images are not uploaded.

### Optional NVIDIA GPU acceleration for Cellpose

GPU acceleration is available only for Cellpose on a CUDA-capable NVIDIA GPU. It is not used by
the current ONNX-pack path. Complete the following in order:

1. Confirm that Windows can see the NVIDIA GPU, then download and install the current driver for
   that exact GPU from [NVIDIA's driver page](https://www.nvidia.com/drivers). Restart if the
   installer asks. In a new PowerShell window, run `nvidia-smi`; it must show the GPU and driver.
2. In the active `.venv`, use the [PyTorch Start Locally selector](https://pytorch.org/get-started/locally/)
   with **Stable**, **Windows**, **Pip**, **Python**, and the CUDA build supported by the driver.
   Copy its generated command exactly. For example, if the selector/driver combination supports
   CUDA 12.6:

   ```powershell
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

   Do not copy that example blindly: choose a CUDA wheel no newer than the CUDA version reported
   by `nvidia-smi`, and prefer the selector over a manually guessed version.
3. Install the GPU Cellpose extra after PyTorch, then verify that the same virtual environment can
   see the GPU:

   ```powershell
   python -m pip install -e ".[cellpose-gpu]"
   python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
   ```

For this prebuilt-PyTorch-wheel workflow, the NVIDIA **driver** is required, but the full CUDA
Toolkit, cuDNN, Visual Studio, and `nvcc` are not. PyTorch supplies its required CUDA user-space
libraries. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) only when you
are developing CUDA code, building PyTorch from source, or compiling a CUDA extension; NVIDIA's
current Windows Toolkit installer does not install the driver for you, so keep the driver step
separate.

If the verification command prints `False`, do not select **NVIDIA CUDA GPU** in GST Image. First
update/check the NVIDIA driver, repeat the PyTorch-selector installation in the active environment,
and verify again. GST Image deliberately fails a requested GPU Cellpose run rather than silently
falling back to CPU. Windows AMD/ROCm acceleration is not supported by this integration.

## Run

```powershell
gst-image
gst-image-cli --help
pytest
```

Thin launchers are also provided under `bin/` for environments where console scripts are
not on `PATH`.

## GUI training documentation

New GUI users should begin with the [GST Image training guide](docs/README.md). It provides a
linked onboarding path, full panel/tool reference, segmentation tuning guidance, assisted
region-classification training, Cellpose-SAM usage and external custom-training guidance, result
review, grouping, export, and troubleshooting.

## Cellpose workflow and licensing

GST Image can run the stock Cellpose-SAM v2 (`cpsam_v2`) model locally for 2-D instance
segmentation. `cpsam_v2` is the only Cellpose checkpoint GST Image selects; its first use downloads
the stock weights into Cellpose's local cache and creates a pending-review cell or particle instance
layer. The stock weights are CC-BY-NC, so separate rights are required for commercial use.

The current upstream Cellpose model family also includes these checkpoints, but GST Image does not
select or load them:

| Upstream Cellpose checkpoint | Description | Use in GST Image |
| --- | --- | --- |
| `cpsam_v2` | Current Cellpose-SAM model using a SAM-ViTL backbone with an upstream low-contrast training fix. | **Supported stock model.** |
| `cpdino` | CellposeDINO model using the larger DINOv3-ViTL backbone. | Not supported. Run it in upstream Cellpose. |
| `cpdino-vitb` | Smaller CellposeDINO model using the DINOv3-ViTB backbone. | Not supported. Run it in upstream Cellpose. |
| `cpsam` | Original Cellpose-SAM model using a SAM-ViTL backbone. | Not supported. Run it in upstream Cellpose. |

Upstream Cellpose downloads a built-in checkpoint the first time it is selected. Outside GST Image,
choose a model in the Cellpose GUI, run
`python -m cellpose --pretrained_model cpdino`, or construct
`models.CellposeModel(pretrained_model="cpdino")` in Python. See the current
[Cellpose model reference](https://cellpose.readthedocs.io/en/latest/models.html) for details.

### Add a custom model to upstream Cellpose

Custom Cellpose models are added to **upstream Cellpose**, not GST Image. After fine-tuning a model
from the Cellpose GUI or CLI, either register the file for the GUI or provide its full path:

```powershell
python -m cellpose --add_model C:\cellpose-data\train\models\my_model
python -m cellpose --pretrained_model C:\cellpose-data\train\models\my_model
```

The first command makes the checkpoint available under the Cellpose GUI's custom-model selection;
the second runs it directly. In Python, use
`models.CellposeModel(pretrained_model=r"C:\cellpose-data\train\models\my_model")`. Train from
the built-in `cpsam` model with reviewed instance labels, keep all intended training images in the
same training folder, and test on held-out source groups. GST Image cannot import, select, or run
a custom Cellpose checkpoint; use upstream Cellpose for that result or complete a separate,
validated ONNX-pack integration. See the [Cellpose-SAM usage, tuning, and training guide]
(docs/cellpose.md) for the full fine-tuning workflow.

## Typical GUI workflow

1. Open a micrograph or an existing `.gstproj` directory.
2. Draw over the scale bar, enter its known length, and choose mm or µm.
3. Add inclusion/analysis rectangles or polygons and exclusion regions as needed.
4. Preview and run the particle recipe. Overview and selected-ROI resolutions are controlled
   independently from 0-100%; 100% uses native pixels, and both affect display and preview only.
   The ROIs tab selection implies the scope: select exactly one Include/Analysis box to preview
   and analyze that box alone, or leave the list unselected to work over the whole image. The
   **Run particles — full resolution** action always uses original-resolution pixels; scoped to a
   box it loads and processes only that native ROI crop, otherwise it processes the complete
   source image. ROI results are positioned back into full-source coordinates for overlays and
   export. Press `Shift+A` to run the same
   **Preview** action from the keyboard. The **Segmentation class** and **Assisted region
   classes** selections stay linked, and the saved particle layer is assigned to that class.
   Use the eyedropper to set either endpoint of a manual threshold range or a
   segmentation-class color directly from a native-resolution 5 × 5 source sample.
5. Inspect the overlay and particle table; select the saved result layer before using its
   mask brush/eraser or split/merge corrections. Press `O` for **Mask brush** or `P` for
   **Mask eraser**. While a class seed, seed eraser, mask brush, or mask eraser is active, use
   `[` and `]` to decrease or increase its source-pixel radius. The pink canvas outline shows
   the active brush footprint. Hold the middle mouse button and drag to pan temporarily without
   changing the active tool. In the Layers tab, **Copy ROI mask image** copies the active
   result as a native-resolution black-and-white ROI crop suitable for pasting into another
   application. Toggle **Colour each particle individually** (Layers tab or **View**, `Shift+C`)
   to give every label its own colour while judging which particles need splitting or joining,
   and toggle it off for the standard mask.
6. Open **Particle groups** with an instance layer selected to filter and partition measured
   particles by equivalent radius, equivalent diameter, area, circularity, or a combination.
   The dual-handle range bars update the image, particle table, plots, and per-group statistics
   without rerunning segmentation. Generate equal-width size groups, circularity groups, or a
   two-dimensional size/circularity grid, then edit group names, colors, and exact bounds before
   saving the scheme to the project. **Copy group table** and **Copy statistics** place the
   complete tables on the clipboard as tab-separated text for Notepad, Excel, and similar
   applications. **Open color plots** includes size, circularity, and an area-based estimated
   volume-fraction pie chart in the active ROI's group colors. Its save button writes the
   combined view, each individual graph as a PNG, and a reloadable Matplotlib Figure pickle.
   **Save grouping overlay image…** writes the active Analysis box at native resolution with its
   grouping colours.
7. For macro zones, follow the [assisted region-classification guide](docs/region-classification.md)
   to select one Analysis box, paint and erase example strokes for Weld, HAZ, Base, or custom
   classes, and train the classifier only within that box at the Selected ROI resolution.
8. For biological cells or experimental metallography particles, follow the
   [Cellpose-SAM guide](docs/cellpose.md): draw Analysis boxes, run the local stock model at native
   resolution, tune its instance settings, and confirm the reviewed result. GST Image's Cellpose
   integration is inference-only; custom-model training remains an external Cellpose workflow.
9. Save the project and export masks, native ROI source images, per-layer segmentation overlays,
   measurements, fractions, particle CSVs, saved grouping definitions, particle assignments,
   group statistics, and crop-aware color grouping overlays.

Analysis values have linked sliders and numeric fields for quick tuning and exact entry.
Calibrated measurements are shown in both physical and pixel units. Layers, ROIs,
measurements/calibration, and particle groups can be removed with their tab's **Delete
selected** action; deleting an ROI also removes analysis layers derived from that ROI.

Threshold-specific controls change with the selected method:

| Method | Method-specific controls |
| --- | --- |
| Sauvola | Window size and k |
| Adaptive Gaussian | Block size and constant C |
| Otsu | None; the threshold is calculated automatically from the analysis domain |
| Manual | Inclusive lower and upper intensity thresholds, each settable with the eyedropper |

Analysis channel, polarity, Gaussian pre-blur, optional illumination correction, morphology,
particle-area filtering, watershed splitting, and tile size are shared settings and remain
available when applicable. The illumination radius is hidden when correction is disabled, and
watershed distance is hidden when splitting is disabled.

Manual thresholding selects processed 8-bit channel values inside the inclusive lower/upper
range. Particle polarity does not invert this manual band, but it still guides optional
rolling-ball illumination correction. Version-1 projects and standalone recipe JSON files with
a one-sided `manual_threshold` are migrated to the equivalent dark or bright range. The legacy
Dark/0 and Bright/255 empty selections are represented by the nearest one-value range because
an ordered 0-255 interval cannot express an empty selection.

Binary closing and optional border-seeded flood filling are available after every threshold
method. Closing joins nearby foreground and seals narrow gaps; flood filling fills enclosed
background holes after all tiles have been assembled, so tile boundaries do not create false
holes. Each parameter section has its own restore button for threshold-method settings,
morphological watershed, pre-blur, rolling-ball correction, opening/closing, flood filling,
particle filters, touching-particle watershed, tile size, and brush size. These actions leave
the class, channel, segmentation/threshold methods, polarity, preview mode, ROI scope, and
overview/ROI resolutions unchanged.

Particle grouping filters are post-analysis and non-destructive. They only control which
measured instances are included in a saved grouping result; the original instance labels and
segmentation mask remain unchanged. This is different from the segmentation recipe's minimum
and maximum particle-area filters, which discard components while segmentation is running.
When both size and circularity filtering are enabled, both conditions must match. Adjacent
generated groups use non-overlapping boundaries, with the outermost upper boundary included.
Filtered and unclassified particles can independently be hidden or shown in grey.

Every saved grouping is tied to its source particle layer. Its statistics include counts,
count percentage, total area, analyzed-domain area/estimated volume fraction, border-particle
count, and size/circularity distributions. Saved schemes are exported to
`particle_grouping_definitions.csv`, `particle_group_assignments.csv`, and
`particle_group_statistics.csv`; native Analysis-box grouping overlays are exported as PNG files.
Selected-ROI recipes and particle layers are likewise tied to their Analysis box. Clicking an
ROI restores its particle recipe and activates that ROI's most recent matching result layer,
particle records, mask, and saved grouping scheme, allowing several ROIs to be analyzed and
edited independently in one open project.

The **Morphological watershed** segmentation option is based on the ImageJ/MorphoLibJ
[Morphological Segmentation](https://imagej.net/plugins/morphological-segmentation) pipeline.
It supports object or border inputs, morphological/internal/external gradients, gradient
radius, extended-minima tolerance, 4/8 connectivity, and optional watershed dams. In GST Image,
the watershed is constrained to the foreground selected by the active threshold method so
particle area fractions retain their foreground/background meaning.

## CLI examples

```powershell
gst-image-cli analyze test/img/example.jpg --output analysis.gstproj --mm-per-pixel 0.001
gst-image-cli batch test/img --output test-output --recipe recipe.json --calibration-csv scales.csv
gst-image-cli infer-cellpose input.tif --output cellpose-result.gstproj --modality biological --device cpu --accept-cellpose-noncommercial-license --allow-model-download
gst-image-cli export analysis.gstproj --output exported
gst-image-cli validate-project analysis.gstproj
```

Physical size filters require a calibration. A source image is referenced by path and
SHA-256; changed inputs are reported and derived results are not silently reused.

## Repository layout

- `src/gst_image/` contains Qt-independent models, algorithms, project I/O, and CLI code.
- `app/gst_image_app/` contains the PySide6 interface.
- `bin/` contains development launchers.
- `tests/` contains synthetic unit/integration tests and opt-in large-image tests.
- `temp/` retains the original Tkinter prototype for reference.

See `BENCHMARKS.md` for the reproducible full-resolution memory/runtime benchmark.
