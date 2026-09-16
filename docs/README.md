# GST Image training guide

This guide is the entry point for first-time users of GST Image. Work through the pages in the
order below once; thereafter, use the linked topic pages as a reference while analysing an image.

## Start here

1. [Getting started](getting-started.md) — open an image, understand the workspace, calibrate it,
   create a safe analysis area, preview, run, save, and export.
2. [Workspace and drawing tools](workspace-and-tools.md) — every menu, toolbar button, canvas
   gesture, and Results and layers panel tab.
3. [Particle segmentation](particle-segmentation.md) — the complete Analysis-panel reference,
   including what each control changes in the mask.
4. [Segmentation methods and tuning](segmentation-methods-and-tuning.md) — choose an algorithm,
   understand its parameters, and tune it systematically with worked examples.
5. [Assisted region classification](region-classification.md) — paint examples, train the
   supervised classifier, improve its result, and understand its scikit-image/scikit-learn steps.
6. [Reviewed ML model workflows](model-inference.md) — run validated ONNX phase/particle packs
   or local Cellpose-SAM instances, preserve provenance, and record expert review.
7. [Review, correction, grouping, and export](review-correction-grouping-export.md) — inspect
   results, repair masks, separate/merge particles, group measurements, and deliver traceable
   files.
8. [Glossary and troubleshooting](glossary-and-troubleshooting.md) — plain-language definitions,
   common failure symptoms, and recovery actions.

## A sensible first exercise

Use an image whose features of interest are visibly darker or brighter than their surroundings.
Open it, calibrate it if you know the scale, put an **Analysis box** around a representative area,
and follow the first example in [Segmentation methods and tuning](segmentation-methods-and-tuning.md).
Run a preview before a full-resolution result. A preview is not saved and is the safe place to
experiment.

## Scope and terminology

GST Image measures **two-dimensional image sections**. Its reported “estimated volume fraction”
is the segmented area fraction used as a stereological estimate; it is not a 3-D reconstruction.
Similarly, equivalent radius and diameter describe an equal-area circle, not necessarily the
object’s literal radius or diameter. See the [glossary](glossary-and-troubleshooting.md).

The application stores source-coordinate ROIs, strokes, masks, recipes, measurements, and run
provenance in a `.gstproj` project directory. Saving early and often makes analysis reproducible.
