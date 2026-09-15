# Gold-mask validation set

Synthetic gold masks are exercised by the normal test suite. Before segmentation accuracy
is formally accepted for lab reporting, add metallurgist-approved crops and masks here for:

- isolated round and angular particles;
- touching particle clusters;
- strong illumination gradients and stitch steps;
- Weld, HAZ, and Base boundaries with fixed training strokes;
- fusion-line dilution analysis boxes; and
- black canvas and scale-bar exclusions.
- phase masks with the approved class-value map and individual particle labels for every
  candidate model-pack release; split these by original micrograph/acquisition condition, not
  adjacent crops.

The approval record should include the source SHA-256, crop rectangle in full-image
coordinates, annotator, review date, and intended class definition. Original images under
`test/img` must not be modified.

Use `gst_image.evaluation.evaluate_particle_segmentation` for the documented area-fraction,
instance-F1-at-IoU, and equivalent-radius error gates, and `macro_dice_score` for phase classes.
