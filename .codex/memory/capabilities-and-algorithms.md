# Capabilities and Algorithms

Read [Repository Guidelines](../../AGENTS.md) for contribution rules. This note
summarizes current behavior and the implementation choices agents must preserve.

## User-facing capabilities

- Open large micrographs or reopen `.gstproj` directories; optional portable
  projects include their source image.
- Calibrate pixels from a measured scale line; physical settings/measurements
  are stored canonically in millimetres per pixel.
- Define polygon or rectangle include, exclude, and Analysis Box ROIs; suggest a
  specimen domain that rejects black stitch canvas and small annotations.
- Segment particle/phase layers, inspect and manually brush, erase, split, or
  merge instance masks; measure particle morphology and fractions.
- Train Weld/HAZ/Base/custom region classes from painted samples, then edit the
  multiclass result mask.
- Run reviewed local ONNX model packs or optional local stock Cellpose-SAM v2
  instance inference; model-derived layers remain pending review until confirmed.
- Filter and group particles by equivalent radius, diameter, area, circularity,
  or combined criteria without changing the original label image.
- Export masks, overlays, CSV summaries/tables/histograms, recipes, and complete
  provenance; use CLI `analyze`, `batch`, `infer-model`, `infer-cellpose`, `export`,
  and `validate-project`.

The reported “volume fraction” is a two-dimensional segmented-area fraction used
as a stereological estimate, not a 3D reconstruction. Equivalent radii are
equal-area-circle radii on the observed section.

## Particle segmentation pipeline

`segment_particles()` in `analysis/particles.py` owns the scientific workflow:

1. Select grayscale, B/G/R, or Lab channel with `to_gray()`.
2. Optionally estimate a coarse rolling-ball illumination field and flatten it.
3. Threshold in tiles with a halo, using Sauvola, Adaptive Gaussian, Otsu, or an
   inclusive manual 8-bit range. Otsu derives one threshold from a bounded
   preview of the analysis domain for tile consistency.
4. Optionally apply elliptical opening/closing and flood-fill enclosed holes;
   every tile is clipped to the analysis mask.
5. Build instances via connected components and optional distance-transform
   watershed splitting, or use the ImageJ-style morphological-watershed path.
6. Filter instances by pixel or calibrated physical area, measure them, and
   compute fractions/density/size statistics.

The normal separation watershed uses SciPy Euclidean distance transform,
`peak_local_max`, and scikit-image watershed. The morphological variant computes
internal/external/full morphological gradient surfaces, extended minima
(`h_minima`), and constrained watershed basins. It must stay constrained to the
threshold foreground so foreground area remains meaningful.

`measure_particles()` uses `skimage.measure.regionprops`: area, Crofton
perimeter, circularity, solidity, eccentricity, axes, Feret diameter, centroid,
equivalent size, calibration conversions, and analysis-domain border contact.
By default border-touching particles remain in the layer/fraction but are omitted
from size statistics.

## Regions, ROIs, and grouping

Region classification computes scikit-image multi-scale intensity, edge, and
texture features, then fits a deterministic scikit-learn `RandomForestClassifier`
with `random_state` and one worker. Its labels are optionally snapped to Sobel
edges using marker watershed. It requires at least two painted class IDs and
enforces a 4.5 MP overview cap to control feature-memory use.

ROI masks are rasterized with OpenCV. Fractions always use the matching domain
layer, including multiclass `class_value_map` values separately.

Grouping is post-analysis and non-destructive. Group intervals are validated to
avoid overlap; adjacent generated intervals use exclusive upper bounds except
the outermost final bound. Group statistics report counts, area/fraction, border
counts, and size/circularity distributions. Keep grouping tied to its source
instance layer, never only a global particle list.

## Libraries and their intended roles

| Library | Use in this repository |
| --- | --- |
| NumPy | Array representation and numerical summaries. |
| OpenCV | Image decode/encode, colors, morphology, connected components, drawing. |
| Pillow | Large-image dimensions, scaled loading, region crop, portable image handling. |
| SciPy | Distance transforms, binary morphology, connected labels. |
| scikit-image | Thresholding, rolling ball, features, region properties, watershed. |
| scikit-learn | Random-forest region classifier. |
| Pydantic | Validation, JSON-compatible persisted models, migrations at model boundaries. |
| tifffile | Compressed TIFF project masks and exported masks. |
| pandas | CSV exports and aggregate tables. |
| PySide6 | Desktop widgets, graphics canvas, workers, undo commands. |
| matplotlib | GUI plots for particle distributions/group summaries. |
| Cellpose (optional) | Local Cellpose-SAM v2 labelled-instance inference on CPU or a CUDA-capable NVIDIA GPU; stock weights are explicitly downloaded to Cellpose's cache and require a CC-BY-NC acknowledgement. |

## Evaluation and exports

`evaluation.py` offers binary Dice, greedy one-to-one instance matching at IoU,
instance F1, area-fraction error, and relative equivalent-radius error. Synthetic
tests run by default; `tests/gold/README.md` describes how to add approved masks.

`export_analysis()` writes `provenance.json`, recipe JSON, TIFF masks, particle
and grouping CSVs, fractions, summary JSON/CSV, histograms, and optional source
overlays. Preserve source layer IDs and unit fields in new exports so outputs are
traceable across multiple analysis layers.
