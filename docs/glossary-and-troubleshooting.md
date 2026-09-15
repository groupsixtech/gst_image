# Glossary and troubleshooting

Use this page as a quick reference alongside the training workflow in [Getting started]
(getting-started.md).

## Plain-language glossary

| Term | Meaning in GST Image |
| --- | --- |
| **Analysis domain** | The pixels allowed in the denominator: specimen mask ∩ included/analysis ROIs − excluded ROIs. Pixels outside it cannot contribute to result fraction or measurement. |
| **ROI** | Region of interest. Include and Analysis ROIs add valid area; Exclude ROIs remove invalid area. An Analysis box also carries a saved particle recipe. |
| **Foreground / background** | Pixels selected as target / not selected as target by thresholding. Foreground is not necessarily white or bright; polarity decides expected relation. |
| **Binary mask** | A two-value image: zero background, nonzero foreground. |
| **Instance label** | An integer image where every separate particle has its own positive number; zero is background. This is needed for count and per-particle morphology. |
| **Multiclass label** | An integer image where values represent named region classes rather than separate particles. |
| **Area fraction** | Segmented foreground area divided by valid analysis-domain area. |
| **Estimated volume fraction** | Area fraction × 100, presented as a 2-D stereological estimate rather than a direct 3-D measurement. |
| **Equivalent radius/diameter** | Radius/diameter of a circle with the same measured area as an object. It is a convenient size measure, not a promise of roundness. |
| **Circularity** | `4π × area / perimeter²`, limited to 0–1. A smooth circle approaches 1; long, rough, or jagged shapes are lower. |
| **Solidity** | Object area divided by convex-hull area. A concave, branched, or holed object has lower solidity. |
| **Feret diameter** | Greatest distance between two parallel tangents of an object; a maximum calliper-like width. |
| **Border-touching particle** | An object meeting the edge of the valid analysis domain. It remains in fraction by default but is excluded from size statistics by default. |
| **Recipe** | Persisted segmentation or classifier settings. Save/export it with reported measurements. |
| **Preview** | Temporary test run, not a saved/exportable analysis layer. |
| **Watershed** | A way of splitting a connected foreground region into labelled basins, analogous to water filling a landscape from several low points. |

## Symptom → likely cause → action

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Nearly everything is selected. | Wrong polarity, too-permissive threshold, local window too small, or strong uncorrected shading. | Check polarity/channel first; enable/tune rolling ball; make Sauvola/Adaptive setting stricter; preview representative locations. |
| Almost nothing is selected. | Wrong polarity or overly strict threshold/range. | Check polarity; reduce Sauvola `k` for Dark targets, reduce Adaptive `C` for Bright targets, or widen Manual range. |
| Works in one side but fails in another. | Illumination/etching variation exceeds a global method. | Use rolling-ball correction and Sauvola/Adaptive Gaussian; choose a window/block matched to variation; ensure ROI excludes foreign image areas. |
| Scale bar, text, or black canvas is selected. | It is in the analysis domain and has target-like intensity. | Draw an Exclude box before tuning. Do not rely on a maximum-area filter alone. |
| Tiny specks inflate count. | Threshold sees noise/texture. | Improve channel/threshold first; then add the smallest useful pre-blur/opening and set minimum area from a documented smallest meaningful object. |
| True small particles disappear. | Opening, blur, or minimum area is too aggressive. | Reduce/disable cleanup temporarily and lower the size limit; retune threshold before accepting noise. |
| Neighbouring particles are one label. | Threshold/closing built a bridge or watershed separation is too conservative. | Reduce unnecessary Closing; enable touching-particle watershed; lower its minimum distance gradually; use manual Split for isolated cases. |
| One particle becomes many labels. | Distance watershed finds noisy centres or morphological watershed has too many minima. | Increase watershed minimum distance; reduce upstream noise; for morphological watershed increase tolerance. |
| Masks have holes that should be solid. | Threshold misses internal pixels. | First verify intensity; then use modest Closing or Flood-fill enclosed holes only if cavities are not scientifically meaningful. |
| Particle result differs after manually painting. | Painting/relabeling changes mask geometry and measurements. | Expected. Review values; save the project. If edits are numerous, improve/re-run the recipe instead. |
| “Select exactly one Include or Analysis box.” | A selected-ROI operation needs one selected ROI of the right type. | In ROIs, select one Include or Analysis box; then Show selected ROI, preview, or run. |
| “Source changed” or source missing. | The source no longer matches the saved checksum/path. | Use Relink source as new revision. This clears derived analysis deliberately; do not relink casually. |
| Region classifier needs more classes. | Seeds contain fewer than two positive class labels. | Paint clean examples for at least two classes, then train. |
| Region classification exceeds 4.5 MP. | Feature computation has a memory guard. | Reduce Overview resolution, click Show overview, then train. Selected ROI resolution is unrelated. |
| Region prediction is wrong only where appearance changes. | Training examples do not cover that variation. | Add clean strokes for the correct class in that appearance and retrain; do not simply add more copies of existing examples. |
| Grouping seems to change the mask. | Grey/coloured overlay may be confused with segmentation editing. | Grouping is non-destructive. Confirm by toggling grouping/layer; only recipe filters and manual edits change instance labels. |

## Quality-control checklist before reporting

- Is the source image identified and unchanged from the project’s recorded revision?
- Is calibration valid and the scale bar drawn accurately, if physical units are reported?
- Does the analysis domain exclude labels, scale bars, seams, and other invalid pixels?
- Was the recipe checked at native resolution on easy and difficult representative areas?
- Are threshold, cleanup, size filters, and watershed choices consistent with the stated measurement
  definition?
- Were border particles handled consistently in size statistics and described in the method?
- Were manual edits sparse, reviewed, and retained in the saved project?
- Are exported recipe/provenance files retained alongside any numbers, masks, figures, or CSVs?

If a result will support a critical engineering, research, or quality decision, have a qualified
reviewer inspect the source/overlay/domain and retain a documented validation sample. The software
helps make work traceable; it does not replace materials-domain judgement.
