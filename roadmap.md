## 

### Learning-based micrograph segmentation roadmap

#### Summary
Adopt a staged, offline-first workflow: a custom 2D U-Net semantic model for phase maps, followed by a boundary-aware particle head whose foreground/boundary probabilities feed the existing watershed and measurement pipeline. Train with reviewed etched-optical micrograph crops; ship only versioned, validated model packs.
- Keep thresholding/watershed and the current scribble-trained random forest as fast, explainable baselines.
- Use [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as a training/benchmark reference, not the bundled runtime; it self-configures supervised semantic-segmentation pipelines. nnU-Net
- Do not make Cellpose, Omnipose, or SAM-derived models the primary automatic workflow: use them only to accelerate annotation experiments. Their biological pretraining is not evidence of reliability on etched metallography. [Cellpose](https://github.com/MouseLand/cellpose), [microSAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)
- Keep StarDist as a later candidate for nearly star-convex particles; it requires complete instance labels and is shape-constrained. [StarDist](https://github.com/stardist/stardist)
Implementation changes
- Add an optional ML inference extra using ONNX Runtime CPU; the main application continues to install and run without ML dependencies.
- Define a portable model-pack contract: manifest.json, ONNX weights, class map, required preprocessing/channel order, tile size/overlap, output-head definitions, model/version/hash, compatible app version, training-data summary, validation metrics, and license/attribution.
- Add persisted ModelInferenceRecipe and ModelInferenceRun records. They must capture model-pack hash, preprocessing, confidence thresholds, ROI/domain, native-image coordinates, runtime version, and review/confirmation status.
- Run inference in a cancellable worker over padded, overlapping native-resolution ROI tiles; blend probabilities, enforce the existing domain mask, and place semantic labels/particle instances back into full-source coordinates.
- Produce separate non-destructive layers:
  - multiclass phase layer from the semantic U-Net;
  - particle foreground, boundary, and watershed-derived instance layer for existing morphology/grouping/export logic.
- Require review before a model result becomes a confirmed measurement layer; preserve manual brush/split/merge corrections and export their provenance.
Data and model workflow
- Create lab-approved semantic and instance gold masks for representative source micrographs, including illumination/stitch variation, etch variation, difficult phase boundaries, touching particles, small particles, labels/scale bars, and exclusions.
- Split by original micrograph/acquisition condition—not by adjacent tiles—into train, validation, and held-out test sets.
- Train a compact RGB 2D U-Net with phase softmax output plus particle foreground and boundary/distance outputs. Use class-balanced semantic loss and boundary-aware particle loss; retain the current watershed as the final instance separator.
- Benchmark against the current threshold/watershed and random-forest workflows. Only package a model after held-out validation and metallurgist review.
- Treat Mask R-CNN as a later experimental alternative for irregular, overlapping, multi-class particles; it has demonstrated materials-science applicability but is heavier and demands full instance annotations. [materials instance-segmentation study](https://arxiv.org/abs/2101.01585)
Test and acceptance plan
- Unit-test model-pack validation, absent/corrupt weights, schema migration, tile blending, channel normalization, confidence handling, cancellation, and full-image coordinate restoration.
- Add approved semantic and instance gold fixtures; evaluate macro Dice per phase, particle instance F1 at IoU 0.5, area-fraction error, and equivalent-radius error using the existing evaluation API.
- Default release gates: macro Dice ≥0.90 for reportable phase classes, instance F1 ≥0.85, absolute area-fraction error ≤5 percentage points, median equivalent-radius error ≤10%, plus metallurgist approval of held-out overlays and distributions.
- Verify CPU inference on a declared Windows reference workstation; run it off the UI thread and record elapsed time in model provenance.
Assumptions
- Initial scope is etched RGB optical metallography, Windows CPU, offline use, permissively licensed dependencies, and maintainer-trained models delivered as separate local packs.
- Training is external to the end-user application in v1; users choose validated local packs and review inference results, but do not train models in-app.
- The scientific class definitions and approved annotations—not pretrained biological weights—are the source of truth for model behavior.