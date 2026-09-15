# Assisted region classification

Assisted region classification is the tool for broad named regions—such as Weld, HAZ, Base
Material, or custom zones—when a single bright/dark threshold is not enough. You paint a small
number of pixels that you know belong to each class. GST Image learns the visual patterns around
those examples and labels one selected Analysis box. This is **supervised machine learning**:
the human supplies the teaching labels and must review the predicted result.

For navigation and brush mechanics, see [Workspace and drawing tools](workspace-and-tools.md).
For small discrete particle/phase objects, use [particle segmentation](particle-segmentation.md)
instead; it creates individually measured instances.

## What the human must provide

The algorithm does not know what “Weld” or “HAZ” means. You must:

1. Define/choose at least **two** enabled classes under **Assisted region classes**. Default
   classes include Weld, HAZ, Base Material, Particle, Phase, and an excluded-area definition;
   **Add class** creates a named, coloured custom class and **Rename class** renames the selected
   class.
2. Paint trustworthy examples for at least two different classes. A stroke is a positive assertion
   that all painted pixels belong to its class, so avoid uncertain boundaries, scale bars, labels,
   glare, and mixed-material pixels.
3. Cover normal variation: different parts of the zone, different brightness/texture, and more
   than one image location. Several small strokes are usually better than one huge stroke.
4. Review the prediction and add/correct examples where it fails. Training is iterative; a perfect
   first run is not expected.

## Training workflow

1. Draw and select exactly one **Analysis box** around the region to classify. Set **Selected ROI
   resolution** in Analysis; 100% uses the native source crop. The resulting crop must be 4.5
   megapixels or less, so use a lower percentage for a large box. **Show selected ROI** is useful
   for inspecting the pixels but is not required before training.
2. Under **Assisted region classes**, choose a class. Use **Add class** and its colour chooser if
   the default names do not match your work.
3. Activate **Class seed** on the toolbar. Paint clean, interior examples of that class. The
   semi-transparent class-coloured overlay shows the supplied labels.
4. Choose another class and repeat. At least two class IDs must have painted pixels.
5. Activate **Seed eraser** to remove an incorrect training stroke. This edits teaching labels
   only; it does not erase a completed prediction.
6. Click **Train / update region classifier**. It loads and computes features only for the
   selected Analysis box, then creates/updates that box's independently scoped multiclass result
   layer.
7. Inspect boundaries and interiors while toggling the layer/opacity in Layers. Add examples in
   places it wrongly predicts, retrain, and compare again.
8. When a local exception remains, select the multiclass result layer, choose the desired class,
   and use **Mask brush** to paint a correction or **Mask eraser** to clear it. Save the project.

Seeds are stored in original image coordinates and replayed when either display resolution
changes. At training time only strokes intersecting the selected Analysis box are rasterized into
its local crop. The brush size is in source pixels; use `[` and `]` to adjust it. Larger source
images can make a given brush appear small on a reduced view, which is expected.

## What GST Image learns

The implementation computes scikit-image `multiscale_basic_features` for every working-resolution
pixel in the selected Analysis box. The feature stack contains enabled combinations of:

| Feature family | Plain-language idea | Why it helps |
| --- | --- | --- |
| **Intensity** | Brightness/colour at multiple blur scales. | Distinguishes consistently lighter/darker or coloured regions. |
| **Edges** | How sharply nearby pixel values change. | Helps locate boundaries and distinguish smooth zones from edge-rich ones. |
| **Texture** | Local pattern/variation at multiple sizes. | Helps separate regions that have similar average brightness but different grain/structure. |

The feature scales run from `sigma_min=1` to `sigma_max=16` pixels in the persisted classifier
recipe. A small sigma notices fine detail; a large sigma describes broader texture. The current GUI
does not expose recipe controls for these advanced values, so the standard interactive workflow is
to improve representative seed coverage rather than attempt hidden parameter tuning.

GST Image then fits scikit-learn’s `RandomForestClassifier` to just the painted pixels. A random
forest is a collection of many simple decision trees: each tree asks feature questions such as
“is this neighbourhood bright and smooth at this scale?”; their combined vote assigns a class. It
uses 80 trees, maximum depth 12, a 15% per-tree training sample, balanced class weighting, one
worker, and a fixed random seed of 0. Those choices make a run reproducible from the same image,
recipe, and strokes, but they do not remove the need for representative supervision.

Finally, GST Image predicts every pixel in the selected Analysis box. It may “snap” class boundaries within a narrow
3-pixel band to strong Sobel image gradients using scikit-image watershed. This can align a
coarse machine-learning boundary to a visible edge; it cannot create a real boundary where the
image offers no edge evidence. The crop labels are mapped back into full-source coordinates as a
multiclass layer whose pixels outside that Analysis box remain clear. Other Analysis boxes retain
their own independent classification layers.

## Best practices for supervised training

- **Teach interiors before boundaries.** First use unambiguous centre areas. If a transition zone
  is genuinely gradual, do not demand a sharp classification boundary that the image cannot show.
- **Balance classes.** Do not paint thousands of Base pixels and a tiny dot of Weld. The model has
  balanced tree weighting, but examples still need to cover each class’s variation.
- **Sample variation, not repetition.** Add strokes at different brightnesses/textures within each
  class and across the image. A long stroke over one uniform patch adds less information than
  short strokes over distinct-looking patches.
- **Keep contamination out.** If a class stroke crosses a scale bar, crack, annotation, or another
  zone, erase it. The model will otherwise learn that contamination as part of the class.
- **Use a hold-out visual check.** Leave some representative areas unpainted and judge prediction
  there. If all checks are inside the painted examples, you only know it remembered its lessons.
- **Iterate by failure type.** If an entire HAZ brightness subtype becomes Base, add HAZ samples of
  that subtype. If only a thin edge is wrong, first see whether boundary snapping/visible edge is
  adequate, then use a small mask correction if needed.
- **Do not mix objectives.** A region classifier labels zones; it does not count individual pores.
  Run particle segmentation separately inside an appropriate ROI if both results are required.

## Worked example: Weld, HAZ, and Base

Open a macrograph, draw an Analysis box around the macro zone, and select it. Start at a Selected
ROI resolution that keeps the box below 4.5 MP. Select Weld, activate Class seed, and paint five
short strokes well inside visually different weld areas. Select HAZ and paint five strokes around
several parts of the transition band. Select Base Material and paint five in both bright and darker
base areas. Train. Suppose darker HAZ is labelled Weld: add two interior HAZ strokes in that darker
appearance, not on the boundary, and retrain. Suppose a tiny label remains misclassified: exclude
it from any particle analysis and use Mask brush/eraser on the region result if it matters to the
displayed zone map. Save the project so the seeds, result, class map, scope, and recipe are auditable.

## Limits and common messages

- **“Paint training strokes for at least two region classes”** — paint at least one stroke for a
  second class inside the selected Analysis box; several samples per class are strongly
  recommended.
- **No Analysis box selected** — select exactly one ROI whose kind is Analysis box. Include and
  Exclude ROIs are not region-classification scopes.
- **Selected Analysis box above 4.5 MP** — reduce Selected ROI resolution to the percentage in the
  warning and train again. This changes classifier working resolution, not the saved source
  coordinates of strokes or results.
- **No editable mask is selected / nothing to paint** — run/training must produce a result first,
  then select a non-domain result layer before Mask brush or Mask eraser can alter it.
- **Prediction changes after a new seed** — expected. A new seed changes the training set and the
  random forest’s learned decision boundary. Keep a saved project or exported result before a
  material revision if comparison is important.

Unlike a deep neural network, this workflow does not download a pre-trained model, need a GPU, or
persist a separately reusable model file. It trains a deterministic per-project classifier from
your painted labels and records the recipe/strokes needed to reproduce the layer.
