# GST Image

GST Image is a Windows-focused desktop and command-line application for traceable
segmentation and measurement of weld macrographs and microstructures. It supports
particle/phase area fraction, assisted Weld/HAZ/Base classification, calibrated line
measurements, per-particle morphology, radius/circularity grouping, and reopenable projects.

The reported volume fraction is the two-dimensional segmented area fraction used as a
stereological estimate. Equivalent particle radii are radii of equal-area circles on the
observed section; neither value is a direct three-dimensional reconstruction.

## Install

Python 3.12 through 3.14 is supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-lock.txt` records the exact Windows/Python 3.14 environment used for the
verified release when a fully pinned installation is required.

## Run

```powershell
gst-image
gst-image-cli --help
pytest
```

Thin launchers are also provided under `bin/` for environments where console scripts are
not on `PATH`.

## Typical GUI workflow

1. Open a micrograph or an existing `.gstproj` directory.
2. Draw over the scale bar, enter its known length, and choose mm or µm.
3. Add inclusion/analysis rectangles or polygons and exclusion regions as needed.
4. Preview and run the particle recipe. Overview and selected-ROI resolutions are controlled
   independently from 0-100%; 100% uses native pixels. Use **Fast overview** for whole-image
   tuning, or select one Include/Analysis box and choose **Selected ROI** to limit memory. The
   **Run particles — full resolution** action always analyzes the original source image.
   Use the eyedropper to set either endpoint of a manual threshold range or a
   segmentation-class color directly from a native-resolution 5 × 5 source sample.
5. Inspect the overlay and particle table; use mask brush/eraser and split/merge corrections.
6. For macro zones, paint example strokes for Weld, HAZ, Base, or custom classes and train
   the assisted classifier.
7. Save the project and export masks, overlays, measurements, fractions, and particle CSVs.

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
holes. **Restore analysis defaults** resets the full recipe, brush setting, preview mode, and
both resolution controls (25% overview and 100% selected ROI).

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
