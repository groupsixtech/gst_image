# Repository Guidelines

For deeper technical context before making a non-trivial change, see the
[agent memory index](.codex/memory/README.md). This guide remains the source for
repository-wide working conventions.

## Project Structure & Module Organization

`src/gst_image/` contains the Qt-independent domain models, image-processing
algorithms, project persistence, exports, and CLI. Keep new analysis features in
`src/gst_image/analysis/` and keep GUI-specific work in `app/gst_image_app/`.
`tests/` holds unit, integration, and offscreen GUI tests; `test/` contains
checked-in sample inputs and expected export artifacts. Use `docs/` for user or
workflow documentation and `bin/` for PowerShell launchers. The legacy Tkinter
prototype in `temp/` is reference-only.

## Build, Test, and Development Commands

Create an environment and install development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

- `pytest` runs the normal suite defined in `pyproject.toml`.
- `pytest -m large` runs opt-in, full-resolution image tests.
- `ruff check src app tests` checks Python style and likely errors.
- `gst-image` launches the PySide6 application; `gst-image-cli --help` lists CLI actions.
- `python tests/benchmark_large_image.py "test/img/1199_1_stitch.jpg"` runs the documented performance benchmark when that sample is available.

## Coding Style & Naming Conventions

Target Python 3.12+, use four-space indentation, and keep lines within Ruff's
100-character limit. Use `snake_case` for functions, variables, and modules;
`PascalCase` for classes; and explicit, domain-oriented names such as
`SegmentationRecipe` or `measure_particles`. Preserve the separation between
GUI code and reusable processing code. Ruff ignores broad-exception warning
`BLE001`, but handle errors deliberately and include useful context.

## Testing Guidelines

Write pytest tests as `tests/test_<feature>.py`, with test functions named
`test_<behavior>`. Prefer synthetic images, `tmp_path`, and focused assertions;
use `qtbot` for widget behavior. Mark tests needing supplied full-size imagery
with `@pytest.mark.large`. Run the focused file first, then `pytest` before a
pull request. No coverage threshold is configured, so add regression tests for
each fixed defect or changed analysis rule.

## Commit & Pull Request Guidelines

Recent history uses concise imperative summaries, commonly `feat: ...`,
`fixes`, or `test`; prefer clear conventional-style messages such as
`feat: add particle grouping export` or `fix: retain masks on project reload`.
Keep commits scoped. Pull requests should explain the behavior change, link the
relevant issue, state the tests run, and include screenshots for GUI or overlay
changes. Do not commit generated `.gstproj` outputs, caches, or local virtual
environments.
