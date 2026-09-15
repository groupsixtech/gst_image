# Modification Playbook

Read [Repository Guidelines](../../AGENTS.md) for style, commands, and review
expectations. Use this note for the technical change path.

## Choose the smallest owning layer

| Change | Usual edit locations |
| --- | --- |
| New segmentation parameter or result field | `models.py`, core analysis, GUI/CLI wiring, persistence/export tests. |
| New image algorithm | `analysis/`, then focused synthetic tests; keep GUI orchestration thin. |
| New GUI control or shortcut | `mainwindow.py`/component, canvas behavior, `pytest-qt` regression test. |
| New grouping metric | `ParticleCriteria`, `analysis/groups.py`, grouping panel, exports, migrations if persisted. |
| New project field/file | models, `project.py`, validation, migration, round-trip tests. |
| New CLI behavior | `cli.py`, command parser, end-to-end project/export test, README if user-visible. |

Avoid embedding scientific logic in `MainWindow`. The CLI should be able to use
the core behavior directly, and core functions should validate inputs rather than
depending on widget state.

## Change procedure

1. Find an adjacent model, algorithm, and test before coding. Preserve native
   coordinate arrays and layer/source-run links.
2. Update the Pydantic contract first for persisted settings/results. Give new
   settings conservative defaults and validate impossible combinations.
3. Put processing in `src/gst_image/analysis/` or a core module. Accept progress
   and cancellation callbacks for substantial work; check cancellation in loops.
4. Wire GUI and CLI separately. GUI analysis must use `FunctionWorker`; CLI must
   create a valid project, domain layer, run, and masks.
5. Update exports, project migration/validation, and documentation when their
   output or meaning changes.
6. Write the smallest regression test that demonstrates the requested behavior,
   then run focused tests, full tests, and Ruff.

## Tests and commands

```powershell
pytest tests/test_particles.py
pytest tests/test_project_and_cli.py
pytest tests/test_gui_smoke.py
pytest
pytest -m large
ruff check src app tests
```

`tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`, so GUI tests must work
without a visible display. Use `qtbot` to add widgets and wait for asynchronous
UI effects. Use `tmp_path` for images/projects/exports and synthetic NumPy/OpenCV
images for algorithm tests. Add `@pytest.mark.large` only when a test needs
supplied full-resolution imagery.

For scientific changes, test both expected results and invariants: no pixels
outside the domain, correct calibrated units, stable label handling, correct
border status, and unchanged behavior for equivalent tiled/untiled cases where
applicable. For project changes, test save/load, `validate_project()`, and any
old JSON migration. For GUI bugs, add a direct user-interaction regression test.

## Large-image and correctness guardrails

- Do not load a full source image merely to display it. Use `load_preview()`,
  `load_scaled_image()`, or `load_region()` appropriately.
- Tiled thresholding depends on a sufficient halo for local threshold windows
  and morphology. Preserve it when adding neighborhood operations.
- Illumination estimation intentionally runs at a capped coarse resolution;
  raising that cap changes memory and benchmark characteristics.
- Maintain BGR/OpenCV versus RGB/Pillow conversion boundaries. A color-channel
  regression can silently alter segmentation.
- Preserve analysis-mask intersection through thresholding, labels, metrics,
  fractions, and exports. A layer mask alone is not its denominator.
- Reindex labels only when needed and never merge distinct touching watershed
  regions by relabeling connected foreground.
- Avoid unseeded ML/random processing; classification has a persisted seed for
  reproducibility.

## Persistence and export checklist

When a new result is persisted, ensure it is represented in `ProjectManifest`,
serialized with JSON-compatible values, tied to a layer/run where appropriate,
included in validation, and exported with provenance. Do not silently reuse a
project after source pixels change: hashes and `relink_source()` intentionally
invalidate derived results.

Run `gst-image-cli validate-project <project.gstproj>` after manual project
testing. For output changes, inspect generated CSV headers, TIFF masks, and
overlay alignment—not just that a file exists.

## Performance baseline

`BENCHMARKS.md` records the verified large-image baseline and command. It uses a
157 MP image, tiled Sauvola, rolling-ball correction, and no watershed, with a
sub-4 GiB target on the recorded environment. Re-run the benchmark for changes
that affect loading, illumination correction, thresholding, morphology, or array
retention; record hardware and dependency versions with meaningful comparisons.
