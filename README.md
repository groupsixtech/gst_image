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

## GUI training documentation

New GUI users should begin with the [GST Image training guide](docs/README.md). It provides a
linked onboarding path, full panel/tool reference, segmentation tuning guidance, assisted
region-classification training, result review, grouping, export, and troubleshooting.

## Typical GUI workflow

1. Open a micrograph or an existing `.gstproj` directory.
2. Draw over the scale bar, enter its known length, and choose mm or µm.
3. Add inclusion/analysis rectangles or polygons and exclusion regions as needed.
4. Preview and run the particle recipe. Overview and selected-ROI resolutions are controlled
   independently from 0-100%; 100% uses native pixels. Use **Fast overview** for whole-image
   tuning, or select one Include/Analysis box and choose **Selected ROI** to limit memory. The
   **Run particles — full resolution** action always uses original-resolution pixels. With
   **Analyze selected ROI/box only** enabled, it loads and processes only the selected native
   ROI crop; otherwise it processes the complete source image. ROI results are positioned back
   into full-source coordinates for overlays and export. Press `Shift+A` to run the same
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
   application.
6. Open **Particle groups** with an instance layer selected to filter and partition measured
   particles by equivalent radius, equivalent diameter, area, circularity, or a combination.
   The dual-handle range bars update the image, particle table, plots, and per-group statistics
   without rerunning segmentation. Generate equal-width size groups, circularity groups, or a
   two-dimensional size/circularity grid, then edit group names, colors, and exact bounds before
   saving the scheme to the project. **Copy group table** and **Copy statistics** place the
   complete tables on the clipboard as tab-separated text for Notepad, Excel, and similar
   applications. **Open color plots** includes size, circularity, and an area-based estimated
   volume-fraction pie chart in the active ROI's group colors.
7. For macro zones, follow the [assisted region-classification guide](docs/region-classification.md)
   to paint and erase example strokes for Weld, HAZ, Base, or custom classes and train the
   classifier.
8. Save the project and export masks, overlays, measurements, fractions, particle CSVs, saved
   grouping definitions, particle assignments, group statistics, and color grouping overlays.

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
`particle_group_statistics.csv`; color grouping overlays are exported as PNG files.
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
