# Getting started: first complete analysis

This walkthrough produces a defensible particle/phase result while teaching the main workflow.
For a control-by-control reference, continue to [Workspace and drawing tools](workspace-and-tools.md)
and [Particle segmentation](particle-segmentation.md).

## Before you begin

Decide what counts as the foreground: for example, dark pores, bright precipitates, or a phase
with a distinct colour channel. Use an image with enough resolution that the smallest meaningful
object is several pixels across. Do not expect a threshold to distinguish visually identical
materials.

## The workflow

1. Choose **File > Open image…** (`Ctrl+O`) and select PNG, JPEG, BMP, TIFF, or a project with
   **File > Open project…** (`Ctrl+Shift+O`). An image opens as an overview rather than forcing a
   full-resolution load into the display.
2. If physical sizes or densities matter, select **Calibrate** on the toolbar. Drag exactly along
   a known scale-bar length, enter that length, and choose `mm` or `µm`. GST Image converts the
   answer to millimetres per pixel internally. Calibration affects physical measurements and
   physical units; it does not change the pixels or segmentation.
3. Define where calculation is valid. Draw an **Analysis box** around the area to measure. Add an
   **Exclude box** over labels, scale bars, image seams, or damage that must not enter the
   denominator. Use a polygon when the specimen boundary is irregular. The actual analysis domain
   is specimen area intersected with included/analysis areas, minus exclusion areas.
4. In the Analysis panel, set **Segmentation class**, **Analysis channel**, and **Particle
   polarity**. For dark inclusions, start with `Grayscale`, `Dark`, `Threshold particles`, and
   `Sauvola`.
5. Tick **Analyze selected ROI/box only**, select exactly one Include or Analysis ROI in the
   right-hand ROIs tab, and choose **Selected ROI at full resolution** as the preview mode. Click
   **Preview**. Alternatively, use **Fast overview** to tune a whole image rapidly. The temporary
   yellow preview overlay is not a layer or saved result.
6. Compare the overlay to the source at several locations: ordinary objects, the smallest object
   you care about, bright/dark background variation, and touching objects. Adjust only one
   relevant setting, preview again, and record the reason. The tuning order and examples are in
   [Segmentation methods and tuning](segmentation-methods-and-tuning.md).
7. When the preview status says parameters are confirmed, click **Run particles — full
   resolution**. The full run always works on original pixels. With selected-ROI analysis it only
   loads and processes that source crop; otherwise it processes the entire source.
8. Inspect **Summary**, **Particles**, and **Layers**. Use the [review and correction guide]
   (review-correction-grouping-export.md) for hand corrections or grouping.
9. Choose **File > Save project…** (`Ctrl+S`). A `.gstproj` is a directory even though it has a
   file-like name. Use **Save portable project as…** when another person needs the source image
   copied into the project.
10. Choose **File > Export…**, select an output folder, and keep the generated provenance,
    recipe, masks, tables, native ROI source crops, and segmentation/grouping overlay images
    together with any reported results.

## Preview versus final run

Preview is a test result at overview or selected-ROI display resolution. It lets you find wrong
polarity, unwanted background, and bad object separation cheaply. A confirmed preview only means
the controls and scope have not changed since that preview; it does not certify scientific
accuracy. The final run uses the same recipe on native source pixels, creates an instance layer
and matching analysis-domain layer, calculates measurements, and can be saved/exported.

Changing a segmentation control, analysis scope, or preview resolution invalidates confirmation.
Run Preview again before relying on the full-resolution button.

## Practical example: dark pores

Suppose a polished cross-section has dark, roughly circular pores and an unevenly lit background.
Draw an Analysis box around clean material and exclude its scale bar. Select `Dark`, leave
**Rolling-ball correction** on, use Sauvola with its defaults, and preview the ROI at full
resolution. If faint pores are missing, lower Sauvola `k` a little; if surface texture becomes
pores, raise `k` or use a larger window. Then apply the smallest **Opening radius** that removes
isolated speckles and set **Minimum area** just below the smallest pore that should count. If
joined pores remain, keep watershed splitting on and reduce its minimum distance cautiously.
