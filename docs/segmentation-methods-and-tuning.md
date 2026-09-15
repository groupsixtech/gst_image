# Segmentation methods and systematic tuning

Use this page after you know the Analysis-panel controls in
[Particle segmentation](particle-segmentation.md). Segmentation is not “find the perfect slider.”
It is a repeatable decision about which pixels represent the material of interest, followed by a
separate decision about which connected pixels are individual objects.

## How the particle pipeline works

1. GST Image converts the chosen image channel to 8-bit values (0 black, 255 white).
2. If requested, it estimates slow background lighting with scikit-image rolling-ball correction
   on a coarse copy, scales the field back to source dimensions, and flattens illumination.
3. It thresholds the image in overlapping tiles using the selected method. Overlap avoids seams
   from local windows and morphology. The analysis domain is enforced in every tile.
4. It optionally opens/closes the binary result and fills enclosed holes.
5. It either labels connected foreground, optionally splitting touching objects by a distance
   watershed, or performs the ImageJ-style morphological watershed route.
6. It filters instance areas, measures surviving instances with scikit-image `regionprops`, and
   calculates area fraction against the saved analysis-domain layer.

The strongest workflow is therefore: validate foreground first, then clean only clear artifacts,
then validate object count/shape. Do not use a size filter to compensate for a wrong threshold.

## Choose a threshold family

| Method | What it does | Best cases | Main disadvantages |
| --- | --- | --- | --- |
| **Sauvola** | Calculates a cut-off separately at each pixel from local mean and contrast (scikit-image `threshold_sauvola`). | Uneven illumination, gradual etching/polishing variation, or targets that are locally distinct but vary across the image. | Can react to texture, scratches, and very small local windows; result may be less intuitive than one global number. |
| **Adaptive Gaussian** | Uses an OpenCV Gaussian-weighted local average minus constant `C` as the cut-off. | Local lighting variation where a smooth, weighted neighbourhood gives stable results. | Also susceptible to texture; `C` changes direction with polarity, which is easy to misread. |
| **Otsu** | Finds one intensity cut-off that best separates two histogram populations. GST Image computes it from a reduced, in-domain preview for consistent tiles. | Evenly lit images with a clearly bimodal histogram: e.g., dark pores on uniform bright matrix. | Cannot accommodate real gradients or overlapping target/background intensities; a dominant background can bias it. |
| **Manual** | Keeps exactly the inclusive 0–255 band entered by the user. | Stable acquisition conditions, a well-separated channel, or a validated SOP requiring explicit limits. | Does not adapt to illumination; a range that works in one field can fail in another. |

### Threshold choice flow

Start with **Otsu** only when the background is uniform and the desired feature makes a clean,
image-wide dark/bright population. Start with **Sauvola** when the specimen has lighting or
contrast drift. Try **Adaptive Gaussian** when Sauvola follows fine texture too much or when a
Gaussian weighted local reference is visually steadier. Use **Manual** when you can defend fixed
channel-value limits from known standards or controls.

No threshold method recognises material identity. If two regions have the same selected-channel
values, thresholding cannot distinguish them. For broad Weld/HAZ/Base zones with mixed texture,
use supervised [assisted region classification](region-classification.md) instead.

## Method-specific tuning

### Sauvola: the usual uneven-lighting choice

Sauvola computes a local threshold from a window centred on each pixel. The **window** is the
physical neighbourhood that defines “local background”; **k** adjusts how much local contrast
changes the decision. GST Image uses `r=128` for the 8-bit dynamic range.

- Begin with window 101 px, `k=0.2`, and rolling-ball correction enabled.
- Set the window several times wider than the target diameter, but smaller than the scale of
  lighting/etch contrast change. For 10-pixel particles under 500-pixel shading, try 81–151 px.
- For Dark targets, raise positive `k` when too much background is selected; lower it if real
  dark targets disappear. For Bright targets the strictness direction is analogous: higher `k`
  is generally more selective. Change by 0.02–0.05, not by large jumps.
- If one texture grain becomes foreground in a small window but not a larger neighbourhood,
  increase window before adding opening. If the image’s local lighting changes faster than the
  window can follow, reduce it.

**Example:** dark precipitates are 12–30 px across, and the left edge is darker than the right.
Choose Dark/Sauvola, rolling ball 151 px, window 101 px, `k=0.20`. At the left edge matrix pits
are selected: raise `k` to 0.25. At the right edge genuine faint precipitates then vanish: try a
smaller 81-px window before lowering `k` again. Choose the setting that works at multiple
representative locations, not just the prettiest patch.

### Adaptive Gaussian: a locally weighted alternative

Adaptive Gaussian subtracts **C** from a Gaussian-weighted local average. With Dark polarity,
foreground is lower than this local reference, so increasing `C` makes the selected range more
generous; with Bright polarity, increasing `C` makes it stricter.

- Start with block 101 px and `C=2`.
- Make block size several times target width. Increase it if texture is treated as background
  drift; decrease it only when lighting varies on a smaller scale.
- Adjust `C` in steps of 1–3 while watching known target/background pixels. For Dark targets,
  decrease `C` to reject more and increase it to recover dim targets. Reverse that practical
  direction for Bright targets.
- Keep pre-blur low (0–1 px) initially. Adaptive methods plus heavy blur can turn weak bridges
  into real connections.

**Example:** bright inclusions lie on a smoothly varying dark field. Start Bright, block 101,
`C=2`. Too much mottled matrix appears: raise `C` to 5. Small inclusions disappear: return to 3
and add only a 0.5 px pre-blur if single-pixel noise remains.

### Otsu: simple global separation

Otsu tries every possible global intensity division and chooses the one that best separates the
distribution into two classes. It is automatic, not magic: if a dark illumination corner creates
a third population, its one threshold must compromise.

- First use an Analysis box/exclusions to remove scale bars, labels, black stitch canvas, and
  unrelated material. These pixels otherwise distort the histogram.
- Turn on rolling-ball correction if the field is visibly shaded, then preview.
- Select Dark or Bright correctly. There is no Otsu slider in the UI; improve the input/channel,
  domain, or illumination rather than searching for a hidden threshold setting.

**Example:** dark circular pores on a uniformly polished bright field, no tonal drift. With Dark
and Otsu, the preview captures pores cleanly but also selects the black scale bar. Exclude the
scale bar and preview again. Do not set a maximum area merely to suppress it—the denominator and
threshold selection would still be contaminated without an exclusion.

### Manual: explicit, reproducible band selection

Manual selects pixels whose processed chosen-channel values are between **lower** and **upper**,
inclusive. It is not “below threshold” unless the range is 0 through your upper value, nor
“above threshold” unless the range starts near your lower value through 255.

- Use **Eyedropper sets > Manual upper threshold** and click clean foreground/background to learn
  practical values. The 5×5 native sample reduces one-pixel noise sensitivity.
- For dark features, start lower=0, then set upper just above the darker target values but below
  normal background. For bright features, start upper=255 and set lower just below target values.
- If the useful target values occupy a middle colour/channel band, use both endpoints to exclude
  black and white artifacts.
- Validate values on several fields and document the chosen limits. Illumination correction can
  improve consistency, but manual bounds still refer to the corrected selected channel.

**Example:** a Lab `b` channel renders yellow phase as 160–205 while matrix is 105–140 and a
bright label is 240+. Choose Manual lower=155 and upper=210. Exclude the label separately. This
is more faithful than treating every value above 155 as phase.

## Separating touching objects

Thresholding tells GST Image “these pixels are foreground”; it does not automatically know where
one touching particle ends. There are two choices.

### Standard distance-transform watershed

With **Threshold particles** and **Watershed split touching particles**, GST Image calculates each
foreground pixel’s distance from background (SciPy Euclidean distance transform). Particle centres
are local distance peaks found with scikit-image `peak_local_max`; scikit-image watershed floods
outward from those peaks to divide a connected component.

- **Minimum distance** is the centre-to-centre spacing required between peaks. A larger number
  accepts fewer peaks and gives fewer/larger objects. A smaller number finds closer peaks and
  can split a single irregular object into too many objects.
- It only attempts splitting on sufficiently large components, which prevents nonsense splits of
  tiny speckles.
- This is advantageous for roughly round, touching particles with a visible neck. It is less
  reliable for elongated fibres, irregular cracks, or clusters whose centres are not distinct.

**Tuning example:** two 20-px-radius circles touch. Starting distance 7 yields three labels due to
surface roughness: increase to 10 or reduce pre-threshold noise. Starting distance 15 leaves one
label: try 10 then 7. Do not first use Closing, which may create stronger bridges.

### Morphological watershed

Morphological watershed follows the ImageJ/MorphoLibJ-style idea of forming a landscape from
object/border intensity change, removing shallow minima with `h_minima`, then flooding from the
remaining minima. The scikit-image watershed is constrained to threshold foreground, so it does
not invent area outside the binary mask.

Use it when visible edges/boundaries inside a foreground cluster are more informative than round
object centres: for example, particles with thin dark boundary valleys or irregular adjacent
phases. It can be advantageous over distance watershed for non-circular shapes. It is
disadvantageous when edges are noisy/weak or when threshold foreground is already fragmentary:
then it can produce many arbitrary basins.

Start Object image + Morphological gradient + radius 1 + tolerance 10 + 4-connectivity + dams.
Change tolerance first. If there are many false pieces, increase it (e.g. 10→20→35). If known
adjacent objects remain merged, lower it (10→6→3). Only then experiment with gradient radius/type
or 8-connectivity. See the detailed control meanings in [Particle segmentation]
(particle-segmentation.md#morphological-watershed-controls).

## Systematic tuning protocol

1. **Write the question.** State target class, smallest meaningful object, expected bright/dark
   appearance, whether touching objects are separate, and whether result is count, area fraction,
   or both.
2. **Calibrate and scope first.** Calibrate if physical thresholds matter. Exclude labels/scale
   bars and select a representative Analysis box. A bad denominator cannot be repaired later.
3. **Pick channel and polarity.** Look for maximal target/background contrast. Record the choice.
4. **Fix illumination before threshold.** Toggle rolling-ball correction and set a radius larger
   than targets. Compare several fields; do not tune threshold around avoidable shading.
5. **Tune one threshold family at a time.** Hold cleanup and watershed at conservative/default
   values. Make a small parameter step, preview, and compare the same checkpoints.
6. **Clean minimally.** Add a little pre-blur or opening only for demonstrable noise; closing or
   flood-fill only for demonstrable gaps/holes.
7. **Set area limits from the measurement definition.** Minimum area should reject objects below
   your scientifically meaningful size, not merely make counts look tidy. Use an ROI/exclusion
   rather than maximum area to hide an obvious artifact.
8. **Tune separation last.** Inspect cluster regions and isolated objects. Alter only watershed
   spacing/tolerance, then verify it has not changed expected single objects.
9. **Confirm at native ROI resolution.** Check representative easy, difficult, and boundary
   locations. Run full resolution only when the unchanged preview is confirmed.
10. **Save the recipe with the project and export provenance.** Repeatability is part of a useful
    measurement.

For a formal study, reserve representative images that were not used for tuning, make a manual
reference mask/instance set for them, and compare results using Dice/instance matching where
available in your workflow. This guards against tuning only to a favoured image.
