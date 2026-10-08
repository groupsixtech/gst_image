# Weld dilution from a cross-sectional panorama

Open **Analysis → Weld Dilution**. The guided workspace measures a flat-surface
bead or combined overlay relative to one straight original coupon surface.
Source pixels are never rotated to align the horizon. Multiple named analyses
can coexist in a project.

## Procedure

1. Open the panorama and use **Calibrate** on a known reference distance. Without
   calibration, area ratios and pixel lengths are still available. Physical
   spacing and estimated volumes require calibration.
2. Choose **Full panorama**, or select an Include/Analysis ROI in the ROIs tab
   and choose **Selected ROI**. Existing exclusion ROIs apply in either case.
   The specimen domain rejects black stitching canvas. Its outline is green.
3. Draw a line along a visible segment of the unmelted coupon surface. For a
   longer reference span, use **Add surface points** on several visible surface
   segments. Edit their source X/Y coordinates in the table to move them, or
   delete selected rows. The fit uses orthogonal least squares. Check its RMS
   residual, span, and projected dashed line across the entire image. A short
   reference span makes extrapolation more sensitive to point placement.
4. Check the orange arrow: it must point into the substrate. **Flip substrate
   side** reverses it. Drawing the endpoints in reverse order does not change
   the substrate side.
5. In **Gaussian blur**, choose the grayscale/color channel and sigma (default
   10 source pixels), then click **Preview blur**. This operation performs only
   Gaussian smoothing; it does not threshold, generate components, or alter an
   envelope. Inspect the blurred image and use **Show original** to compare.
   Adjust sigma and preview again until microstructural detail is sufficiently
   suppressed without moving the fusion boundary excessively. Smoothing runs on
   native pixels, and the preview retains native detail. No surface reference
   or ROI selection is required to inspect the blur. On a large panorama, sigma
   10 can represent less than one screen pixel at fit-to-window scale: check the
   blur feedback next to the controls. Click **Inspect native detail (click image)**,
   then click the region of interest to view a 512-pixel patch at 1:1. Switch
   between **Original + envelope** and **Blurred preview** to compare; **Return to
   panorama** restores the overview.
6. In **Adaptive segmentation**, choose dark/bright polarity and adaptive Gaussian
   or Sauvola thresholding. Starting settings are window 201 source pixels, C=2,
   closing radius 3 px, and opening off. Click **Preview segmentation** to inspect
   provisional threshold candidates. Increase the window when local thresholding
   captures only a boundary band. Threshold settings preserve the blur preview;
   changing channel or sigma invalidates it. Full-resolution segmentation applies
   the selected native blur settings once before thresholding. These settings are
   editable starting values, not a validated universal weld recipe.
7. **Generate full-resolution envelope**, then click the weld's candidate
   components. Gray shows candidates; orange marks selected components. Clicking
   again deselects a component. Selection supports Undo/Redo. The largest object
   is not automatically selected.
8. Inspect the original image. Use brush/erase or polygon add/remove to correct
   the crown and fusion boundary. Polygons finish with a double-click. Brush
   radius is in source pixels. Corrections use the application's Undo/Redo
   stack and do not produce particle records. Internal holes are filled when
   calculating the outer envelope; explicit domain exclusions remain excluded.
   The HAZ and surrounding unmelted material must remain outside the envelope.
9. Set perpendicular tie-line spacing (default one source pixel), display units,
   and histogram bins. Optionally enter a weld length in millimetres; zero means
   no volume estimate. Click **Calculate**.
10. Inspect Summary, Statistics, Tie-lines, the depth/height profile and
    histograms in the **Dilution** results tab. Selecting a table row or clicking
    a plotted profile highlights that tie-line in magenta. Invalid stations
    appear as gaps in the profile. Zooming changes only displayed tie-line
    density, never measurement density.
11. **Confirm reviewed** records review of the current result. Save the project
    and export. Geometry, calibration, envelope or measurement changes mark the
    previous result outdated and clear review. Segmentation/domain changes also
    require regenerating the envelope. Other measurement changes can be
    recalculated without rerunning segmentation.

Close the Weld Dilution dock to return to the normal Analysis controls.

### Local blur brush

Set **Blur brush radius (source px)** and **Local sigma (source px)**, then choose
**Paint local blur** and drag over a region requiring more smoothing. Each stroke
refreshes the blur preview in the background. Set local sigma above the global
sigma to see additional smoothing. The brush changes preprocessing for subsequent
segmentation; it does not paint the accepted weld envelope or modify the source file.

Painted pixels use the greater of global sigma and the last stroke's sigma.
Overlapping strokes replace the local strength rather than repeatedly blurring an
already blurred image. Outside strokes, global smoothing applies. Inspect stroke
boundaries before segmentation. Use Undo/Redo or **Clear local blur strokes** to
revise the painted regions. Painting marks existing segmentation/results outdated:
regenerate candidates before measuring. Stroke coordinates, radius and sigma are
saved in the recipe and replayed by CLI `--resegment`.

## Meaning of the measurements

The primary result is **Cross-sectional dilution (%)**:

`100 × penetration area / (penetration area + reinforcement area)`.

Penetration is fused area on the substrate side of the reference surface;
reinforcement is fused area on its opposite side. Both are weld metal. This is
a geometric measurement, not a chemical-composition measurement. The outer
envelope includes internal pores and dark microstructural features. There is no
pore-subtracted solid-metal result in this version.

Area is measured from native pixel cells. A pixel crossed by the reference line
contributes proportionally to both sides. The colored raster overlay assigns
each pixel one display label and is therefore not itself the fractional-area
calculation. Penetration plus reinforcement always equals envelope area. Tie-line
spacing does not affect dilution.

Lengths are measured perpendicular to the reference, using exact intersections
with the boundary of the accepted pixel cells. Positions increase along the
fitted tangent from the image boundary; uniformly spaced stations start half a
spacing inside that boundary. If spacing exceeds the image span, one station is
placed at its midpoint. A valid station has one continuous envelope interval
that contacts the horizon. Real zero penetration/reinforcement is retained.

Statuses:

| Status | Meaning | Included in length statistics? |
| --- | --- | --- |
| `valid` | One complete interval contacting the horizon | Yes |
| `no_weld` | No envelope intersection | No |
| `no_contact` | Envelope exists on the normal but misses the horizon | No |
| `multiple` | Disconnected envelope intervals | No |
| `clipped` | An endpoint reaches an image/domain boundary | No |

Clipping takes priority over other ambiguous statuses. Invalid lengths are blank,
not zero. Statistics include count, mean, median, population standard deviation,
minimum, maximum, P5 and P95. All accepted envelope pixels contribute to area,
including those at stations with invalid profile lengths. A `partial_section`
flag means the envelope touches an image/domain boundary; the displayed ratio is
for the measured section only. `no_valid_tie_lines` does not invalidate the area
calculation, but requires reviewing the mask and reference before interpreting
depth statistics. The optional local ratio `depth / (depth + height)` is never
averaged to calculate overall dilution.

With calibration, area is converted by `(mm/px)²`. Optional volume is area times
entered weld length, with the explicit assumption **constant cross-section along
the entered length**. One section does not reconstruct a varying weld in 3D.

## Saved projects, CLI and exports

Schema 7 adds optional dilution drafts and completed run snapshots. Earlier
projects migrate with empty dilution collections. Each completed run owns a
native envelope mask and matching domain mask, fitted geometry, calibration,
recipe, selected-component seeds, measurement settings, summary, tie-lines,
review state, and source revision. Later draft edits do not mutate saved masks.
Clicking a saved run's layer opens its inputs, overlay, and results as a read-only
snapshot. **Return to current draft** resumes editing the latest working state.
Relinking the source invalidates dilution state along with other analyses.

Replay a saved run into an empty output directory:

```powershell
gst-image-cli dilution "coupon.gstproj" --run-id "RUN-ID" --output "replay"
```

Find run IDs in `project.json` under `dilution_runs`, or in the export directory.
Default replay measures the saved final mask using the saved run's calibration
and inputs. It writes `replay/dilution.gstproj` and exports; the input project is
unchanged. A confirmed, current input retains its review status on exact replay.

Add `--resegment` to regenerate candidates from the saved recipe and domain and
select them using saved component seeds. This discards manual boundary edits and
creates a pending-review result. Replay fails clearly if seeds are absent, miss
the image, or no longer hit a candidate. It does not guess another component.

GUI Export and ordinary project Export both include a `dilution/<run-id>/`
directory containing summary CSV/JSON, tie-line CSV, depth/height histogram CSVs,
profile/histogram PNGs, native envelope/domain/divided-region TIFFs, recipe and
reference JSON, provenance, and an annotated panorama when the source is
available. Run IDs, units, review state, quality flags, and volume assumptions
remain explicit. Unreviewed/outdated exports are labeled accordingly.

## Limits and verification

The workflow supports flat original surfaces and a combined bead/overlay envelope.
Curved/grooved surfaces, individual-pass attribution, automatic horizon detection,
batch processing, and multi-section volume integration are not implemented.
Segmentation presets need validation against operator-reviewed representative
panoramas before production interpretation. A speed benchmark is not an accuracy
validation.

Focused tests: `pytest tests/test_weld_dilution.py tests/test_weld_dilution_gui.py`.
Optional performance check:

```powershell
python tests/benchmark_weld_dilution.py "test/img/1199_1_stitch.jpg"
```

Scientific background: [geometric weld dilution](https://www.mdpi.com/2075-4701/12/9/1506)
and [OpenCV adaptive thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).
