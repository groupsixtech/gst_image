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
4. Preview and run the particle recipe. Use local Sauvola thresholding for stitched images,
   or choose Otsu, adaptive Gaussian, or manual thresholding.
5. Inspect the overlay and particle table; use mask brush/eraser and split/merge corrections.
6. For macro zones, paint example strokes for Weld, HAZ, Base, or custom classes and train
   the assisted classifier.
7. Save the project and export masks, overlays, measurements, fractions, and particle CSVs.

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
