# Particle segmentation: Analysis panel reference

Particle segmentation turns pixels into a binary foreground mask and then into separate labelled
objects for measurement. The recommended order is: choose channel/polarity, control illumination,
tune the threshold, clean obvious pixel noise, set object-size rules, and finally separate
touching objects. For explanations and examples of each method, see
[Segmentation methods and tuning](segmentation-methods-and-tuning.md).

## Scope, display, and run controls

| Control | Meaning | How to use it |
| --- | --- | --- |
| **Segmentation class** | The project class assigned to the created instance layer/overlay colour. | Choose Particle, Phase, or a custom class. It does not teach a classifier or change threshold pixels. |
| **Analysis channel** | Pixel data used to segment: Grayscale, Blue, Green, Red, Lab lightness, Lab a, or Lab b. | Compare channels on a representative ROI; choose the one where target and background are most distinct. Lab `a` separates green↔magenta tendencies; Lab `b` separates blue↔yellow tendencies. |
| **Segmentation method** | `Threshold particles` or `Morphological watershed`. | Start with Threshold particles for most work. Morphological watershed is a different instance-separation route for boundary-rich images; it still starts from the selected threshold foreground. |
| **Threshold method** | Sauvola, Adaptive Gaussian, Otsu, or Manual. | Select the family appropriate to illumination and contrast. Its extra controls appear below. |
| **Particle polarity** | Whether desired pixels are Dark or Bright relative to background. | Correct polarity is essential for Sauvola, Adaptive Gaussian, and Otsu. Manual uses its numeric band exactly, but polarity still guides rolling-ball correction. |
| **Overview resolution** | Percent of native width and height shown/used for a fast overview preview and region training. | 25% is a practical large-image start. It changes display/preview detail, not final-run resolution. |
| **Selected ROI resolution** | Percent of native pixels loaded when displaying or previewing one selected Include/Analysis ROI. | Set 100% to tune a critical ROI at native detail; lower it only when a very large ROI is slow. |
| **Preview resolution** | `Fast overview` or `Selected ROI at full resolution`. | Use overview for broad behaviour, then native ROI for final tuning. The latter requires exactly one selected Include or Analysis box. |
| **Eyedropper sets** | Destination for Eyedropper clicks. | Select manual high/low to sample intensity limits or a class-colour target to sample colour. |
| **Analyze selected ROI/box only** | Limits final analysis to one selected Include/Analysis ROI. | Tick it and select exactly one ROI in the ROIs tab. Exclude ROIs still apply. Result labels are placed back in source coordinates. |
| **Show selected ROI** | Loads the selected ROI for display at Selected ROI resolution. | Use before inspecting/tuning fine detail. |
| **Show overview** | Reloads the whole image at Overview resolution. | Use to return to context or place/select an Analysis box. |
| **Preview** | Runs a temporary segmentation and shows a mask overlay/summary. | Tune here; `Shift+A` is the shortcut. It never creates a saved layer. |
| **Run particles — full resolution** | Runs the recipe on original source pixels and creates saved instance/domain layers. | Use only after verifying a preview. It is computationally heavier. |
| **Cancel** | Requests cancellation of the current background work. | Use if scope/recipe is clearly wrong; wait for controls to re-enable. |

All numeric values have a linked slider and spin box. The slider is for quick exploration; type a
value in the box for an exact, reproducible setting. Each **Restore … default** resets only that
section, not the class, image, ROI selection, or unrelated settings.

## Threshold-specific controls

| Visible for | Parameter | What it changes |
| --- | --- | --- |
| Sauvola | **Sauvola window (px)** | Width of the local neighbourhood that estimates local background. It is always made odd. Larger sees slower illumination change; smaller follows local changes but can react to texture/noise. |
| Sauvola | **Sauvola k** | Sensitivity/offset of the local decision. With the Dark setting, increasing positive `k` generally lowers the local threshold and makes the foreground stricter (fewer dark pixels). With Bright, it generally raises it and is likewise stricter. Confirm with preview because local contrast matters. |
| Adaptive Gaussian | **Adaptive block size (px)** | Odd-sized neighbourhood whose weighted Gaussian average becomes the local reference. Larger responds to broad illumination gradients; smaller follows local variation. |
| Adaptive Gaussian | **Adaptive constant C** | Value subtracted from the local Gaussian reference before comparison. Increasing `C` makes Dark selection more generous and Bright selection stricter; decreasing does the reverse. |
| Manual | **Manual lower threshold** and **Manual upper threshold** | Inclusive 8-bit channel-value band from 0 (black) to 255 (white). Pixels inside it are foreground. Polarity does not invert this band. Use Eyedropper on clean target/background patches, then widen/narrow deliberately. |
| Otsu | No extra control | Computes one global cut-off from a reduced analysis-domain preview, then applies that same value consistently across tiles. |

## Shared cleanup and measurement controls

| Control | What it changes | Good use / caution |
| --- | --- | --- |
| **Gaussian pre-blur sigma (px)** | Softens the selected channel before thresholding and watershed surfaces. | Raise slightly to suppress pixel-level noise or weak scratches. Too much blurs small objects together and moves boundaries. `0` disables it. |
| **Rolling-ball correction** | Estimates a smooth background illumination field and divides it out before thresholding. | Enable for vignetting/shading. Disable only when the real target is so broad that it may be mistaken for background. |
| **Background radius (px)** | Scale of rolling-ball background estimation. | Set larger than the widest feature you wish to retain, and roughly comparable to/larger than the lighting variation. Too small can flatten genuine large features; too large may leave shading. |
| **Opening radius (px)** | Elliptical erosion then dilation, applied after thresholding. | Removes isolated specks and thin bridges smaller than the radius. Use the smallest effective value; it can delete small targets. `0` disables. |
| **Closing radius (px)** | Elliptical dilation then erosion, applied after thresholding. | Closes small gaps and smooths tiny holes. It can join nearby objects, increasing later watershed work. `0` disables. |
| **Flood-fill enclosed holes** | Fills background islands not connected to the analysis-domain border after tiles are assembled. | Enable when a particle should be solid but thresholding leaves internal holes. Do not enable when internal cavities are meaningful background. |
| **Minimum area (px²)** | Removes connected objects smaller than this before measurement. | Set from a known smallest meaningful object; this is an analysis rule, not merely display filtering. |
| **Maximum area (px²)** | Removes objects above the limit; `No maximum` disables it. | Use to reject scale-bar remnants/large artifacts only after they are excluded where possible. Do not use it to hide ordinary segmentation failures. |
| **Watershed split touching particles** | Applies distance-transform watershed to threshold components. | Enable for touching roughly blob-like particles. Disable if each connected foreground component is intentionally one object. |
| **Watershed minimum distance (px)** | Minimum spacing between candidate particle centres/peaks. | Increase to prevent over-splitting; decrease to separate smaller touching particles. It is meaningful only when split touching particles is enabled. |
| **Tile size (px)** | Working tile edge length for thresholding large images. | Changes memory/speed trade-off, not desired scientific scale. Leave 2048 unless hardware or performance requires a change. |
| **Brush radius (px)** | Source-pixel radius for seed/mask brushes and approximate split-line thickness. | Match it to the correction scale, not the displayed zoom. `[` and `]` adjust it while a brush tool is active. |

## Morphological watershed controls

These appear when **Segmentation method** is `Morphological watershed`. The active threshold
still determines which foreground pixels are eligible. The watershed then divides that foreground
by “flooding” an intensity or boundary surface from robust local minima.

| Control | Plain-language meaning | Tuning direction |
| --- | --- | --- |
| **Morphological input** | `Object image` constructs a boundary-emphasising surface from the image; `Border image` uses the working intensity image directly. | Begin with Object image for visible particle edges. Try Border image when intensity itself already represents a good basin/boundary surface. |
| **Gradient type** | With Object input: `Morphological` combines outside and inside edge change; `Internal` emphasizes change within foreground-side boundaries; `External` emphasizes outside-side boundaries. | Start Morphological. Internal can favour dark/bright detail inside objects; External can favour contrast just outside them. Compare previews. |
| **Morphological gradient radius (px)** | Size of the structuring element used to observe intensity change around an edge. | Increase modestly for broader/noisier edges; keep 1 for fine boundaries. Too large moves/merges edge evidence. |
| **Morphological tolerance** | Depth a local minimum must have to become a seed. | Increase to remove shallow/noisy seeds and create fewer, larger basins. Decrease to retain more seeds and split more aggressively. |
| **Morphological connectivity** | Whether pixels touching only at a corner can connect (`8`) or only edge-sharing pixels connect (`4`). | Start 4 for conservative diagonal separation; try 8 when diagonal pixels should be one continuous region. |
| **Calculate watershed dams** | Leaves a zero-valued dividing line between neighbouring watershed basins. | Keep enabled when a physical boundary should separate labels. Disable only if zero-width dams create undesirable gaps in the foreground representation. |

For algorithm choice, limitations, and method-specific experiments, continue to
[Segmentation methods and tuning](segmentation-methods-and-tuning.md).
