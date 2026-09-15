# Review, correction, grouping, and export

This page starts after a full-resolution particle run or assisted-region training has created a
saved layer. A preview cannot be manually corrected, grouped, or exported as a saved result.

## Review a completed particle run

1. In **Layers**, click the instance layer you just created and adjust opacity so both source and
   mask edges are visible. Toggle the layer checkbox on/off to find systematic false positives or
   missing targets.
2. Check more than the attractive centre of the ROI: image edges, lighting extremes, smallest
   accepted targets, largest targets, and touching clusters.
3. Compare **Summary** with your expectations. The denominator is the matching hidden domain layer,
   which includes specimen/inclusion scope minus exclusions. `Area fraction` is foreground area /
   domain area; `estimated volume fraction` is that fraction × 100 as a 2-D stereological estimate.
4. Scan **Particles** for unusual area, circularity, solidity, border flag, and group. Border
   objects normally stay in area fraction but are excluded from default size statistics so partial
   objects do not bias size distributions.
5. Decide whether an issue is systematic or isolated. Systematic issues should trigger recipe/ROI
   changes and a re-run. Reserve manual edits for sparse, well-understood local corrections.

## Correct result layers

### Particle instances

Select the particle instance layer first. **Mask brush** (`O`) paints foreground and **Mask eraser**
(`P`) removes it. Use a brush radius close to the feature width and inspect the result at high zoom.
These edits are undoable/redoable and are saved with the project; they change the result mask, so
they should be treated as part of the analysis record, not cosmetic markup.

Use **Select** to click particle instances. Then:

- **Merge selected** merges two or more selected labels into one instance. Use it when watershed
  split one physical particle too aggressively.
- **Split** requires exactly one selected particle. Draw across its narrow connection. The split
  removes a line with a width related to brush radius, labels the separated pieces, and leaves the
  rest unchanged. Use when two true particles are joined.

After a correction, recheck size/count/area values. If dozens of similar corrections are needed,
go back to thresholding, cleanup, or watershed settings rather than continuing manual repair.

### Multiclass region layers

Select the multiclass layer. Choose a desired class in **Assisted region classes**, then use
**Mask brush** to assign that class locally; use **Mask eraser** to clear labels. A correction does
not automatically retrain the classifier—paint new **Class seed** strokes and train again when the
same error recurs elsewhere. See [assisted region classification](region-classification.md).

## Particle groups: non-destructive post-analysis categories

Grouping partitions already measured particles by size, circularity, or both. It does **not**
change instance labels, mask pixels, segmentation area fraction, or the original particle layer.
This is fundamentally different from recipe **Minimum/Maximum area**, which removes objects during
segmentation.

Open the **Particle groups** tab with an instance layer selected. The panel includes:

| Panel control | Purpose | How to use it |
| --- | --- | --- |
| **Grouping name / saved grouping selector** | Identifies a grouping scheme associated with this exact source layer. | Save named alternatives such as `ASTM size bins` and `round-particle subset`; schemes from other layers do not apply accidentally. |
| **Analysis filter (non-destructive)** | Limits which particles contribute to an active grouping preview/statistics by size and/or circularity. | Turn it on to focus analysis, not to edit segmentation. When both size and circularity bounds are present, a particle must satisfy both. |
| **Size metric** | Equivalent radius, equivalent diameter, or area. | Choose the metric used by filter/groups. Equivalent sizes compare area to a same-area circle; they are not literal calliper dimensions. |
| **Unit** | Pixel or millimetre values, where calibration permits mm. | Use mm for comparable calibrated studies; use px only when calibration is unavailable or pixels are the intended criterion. Area units square the selected length unit. |
| **Size and circularity bounds / dual-handle bars** | Inclusive range limits for filter or selected group. | Drag for exploration; enter exact values for reported bins. Circularity ranges from 0 to 1. |
| **Show filtered / show unclassified** | Displays filtered and unassigned labels in grey in the overlay/table. | Keep visible while auditing; hide only for a cleaner presentation. |
| **Generate / replace groups** | Builds regular bins based on selected generation mode. | Use it to draft equal-width size groups, circularity groups, or a size×circularity grid; review/edit the generated boundaries. |
| **Add group / Remove group** | Adds or removes a manually defined group. | Use manual groups for protocol-defined, irregular, or named ranges. |
| **Set color** | Changes selected group overlay/plot colour. | Use high-contrast, consistent colours; colour does not alter membership. |
| **Copy group table** | Copies group definitions as tab-separated data. | Paste into a spreadsheet or audit note. |
| **Open color plots** | Opens size/circularity plots and area-based estimated-volume-fraction pie view in group colours. | Use for exploratory review; export tables for reporting. |
| **Copy statistics** | Copies active grouping summary as tab-separated data. | Paste values into a review record or spreadsheet. |
| **Save / Delete grouping** | Persists or removes the grouping scheme for the source instance layer. | Save after validation; delete only a scheme, not the segmentation layer. |

Generated adjacent groups do not overlap: their shared upper/lower boundary is assigned once, with
the outermost final upper boundary included. A particle matching no enabled group is
**Unclassified**. Filtered-out particles are distinct from unclassified particles: the former fail
the grouping filter, while the latter passed it but match no group.

### Grouping example

With calibrated particles, choose equivalent diameter in mm. Create filter bounds 0.010–0.200 mm
to set aside dust and rare artifacts non-destructively. Generate three equal-width groups, then
rename them `Fine`, `Medium`, and `Coarse`; check exact boundaries and set colours. Open the colour
plots and inspect whether grey particles are unexpectedly common. Save the scheme. The original
segmentation retains every instance, while grouping statistics report included group counts, area,
area fraction, estimated volume fraction, border count, and size/circularity distribution.

## Save, reopen, and export

**Save project…** keeps the manifest, recipes, ROIs, calibration, strokes, result layers, edits,
particle records, and saved grouping schemes. A normal project references its source and checks its
SHA-256 checksum. If the source changes, create a revision with **Relink source as new revision…**;
derived output is intentionally invalidated rather than silently mixed with new pixels. A portable
project additionally copies its source and is the better hand-off format.

**Export…** writes to a selected directory. Typical outputs include:

| Output | Why to retain it |
| --- | --- |
| `provenance.json` | Full project manifest: source identity, runs, layers, classes, and traceability. |
| `recipes.json` and latest `recipe.json` | Segmentation and region-classification settings needed to reproduce work. |
| `*_mask.tif` | Compressed native-resolution layer masks; instance/multiclass labels are values, not merely pictures. |
| Particle/summary/fraction CSV and JSON files | Numeric measurements, area fractions, and run summaries for analysis/reporting. |
| Overlay PNG | Human-readable visual check of layer placement. |
| `particle_grouping_definitions.csv` | Exact grouping/filter bounds, colours, and source layer. |
| `particle_group_assignments.csv` | Per-particle inclusion/group membership. |
| `particle_group_statistics.csv` | Group count, area, area fraction, estimated fraction, border count, and distributions. |
| Grouping overlay PNGs | Colour-coded visual audit of each saved grouping. |

Keep exported recipe/provenance next to any CSV used in a report. A CSV without its analysis domain,
calibration, recipe, and source identity is difficult to reproduce or defend.
