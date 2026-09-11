# Assisted region classification

The assisted classifier turns a few painted examples into a complete Weld/HAZ/Base (or
custom-class) layer. It operates on the whole-image overview, then maps the result back to
the source-image dimensions. It does not use the selected-ROI preview.

## Workflow

1. Set **Overview resolution** in the Analysis panel and click **Show overview**. The overview
   must contain no more than 4.5 million pixels. The default 25% setting is a good starting
   point for a large stitched image.
2. Choose a class under **Assisted region classes**, activate **Class seed**, and paint several
   representative areas well inside that region. Repeat for at least two classes. Samples
   distributed across the image are generally more useful than one large stroke.
3. Use **Seed eraser** to remove mistaken training paint. This changes only the examples used
   for the next training run; it does not edit a completed result layer.
4. Adjust any active brush with `[` and `]`. The radius is measured in source-image pixels,
   and the pink outline shows its footprint at the current zoom.
5. Click **Train / update region classifier**. When it finishes, the result appears as the
   **Weld regions** layer.
6. To correct the completed result directly, select **Weld regions** in the Layers tab, choose
   the desired class, and use **Mask brush**. Use **Mask eraser** to clear labels. These tools
   require an editable result layer; they do not edit a temporary particle preview.
7. Save the project. Class-seed and seed-eraser strokes are saved in source coordinates and
   replayed if the overview resolution changes.

## The 4.5 MP overview limit

Region classification computes a multi-scale feature stack, which uses substantially more
memory than the displayed RGB image. To keep that work reliable, the classifier limits the
overview to 4.5 megapixels. If the current overview is larger, the application reports its
dimensions and recommends a safe percentage.

Set **Overview resolution** to the recommended percentage or lower, click **Show overview**,
and train again. Existing class seeds are rescaled automatically. Lowering **Selected ROI
resolution** will not help because region classification uses the whole-image overview.

## Common messages

- **Paint training strokes for at least two region classes**: choose a second class and add
  representative Class seed strokes before training.
- **Region-classification overview is too large**: reduce Overview resolution as described
  above and click Show overview before training again.
- **No editable mask is selected**: run particle analysis or region classification, then
  select its non-domain result in the Layers tab before using Mask brush or Mask eraser.
- **Nothing appears while painting**: verify that the layer is visible and its opacity is
  above zero. Class seed paint is shown only while Class seed or Seed eraser is active.
