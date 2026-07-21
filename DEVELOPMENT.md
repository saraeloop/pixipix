# PixiPix development

## Environment

PixiPix v0.1 is intentionally locked to Python 3.12. Install `uv`, then run:

```bash
uv sync
uv run python --version
```

## Quality gates

Run the complete sequence before requesting review:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build
```

Use `uv run ruff format .` to apply formatting. Targeted pytest paths are useful while
developing, but they do not replace the full suite. Do not commit generated `dist/`,
caches, local environments, temporary extraction outputs, or locked `docs-internal`.

The source distribution intentionally includes tests, neutral synthetic fixtures, the
repository ignore rules, `.python-version`, `uv.lock`, README, development guidance, and
license material so a source release remains auditable and testable. Repository-only
brand assets and branded examples are excluded. The wheel intentionally contains only
the runtime package and license metadata. Neither artifact may contain internal planning
files, caches, environments, build output, machine-local absolute paths, or restricted
repository-only artwork.

## Architecture map

```text
Typer CLI adapter
  -> inspect / extract / scale / pixelize / align stage orchestration
      -> typed domain models + deterministic stage functions
          -> image, metadata, serialization, handoff, and filesystem helpers
              -> Pillow / NumPy / standard library
```

`cli.py` only parses, invokes, renders, and maps errors. `config.py` owns semantic
validation and configuration hashes. `models.py` contains public typed contracts;
mutable image buffers have explicit ownership and never enter frozen metadata.
`imageio.py` owns PNG and mask behavior. `stages/extract.py` owns flood fill, filtering,
ordering, frame creation, extraction metadata, and safe publication. `serialization.py`
is the only deterministic JSON writer.

`stages/scale.py` owns geometric and channel rounding, sheet-level factor calculation,
premultiplied-alpha BOX resampling, exact reference targets, override warnings, and
scale metadata. Geometry and channel rounding are deliberately separate helpers.
Premultiplication uses unquantized float32 channels, Pillow `F`-mode BOX is the single
resampling pass, un-premultiplication uses float64, and final channels are quantized
half-away-from-zero once. Fully opaque input uses the mathematically equivalent native
RGBA BOX path so it remains byte-equivalent to ordinary BOX output. Transparent output
RGB is always normalized to zero.

`stages/pixelize.py` owns bottom-left cell-grid preparation, top/right remainder
handling, representative selection, alpha policy, and logical-space metadata. Padding
is added only above and to the right; cropping removes only those edges. The logical
array retains conventional top-left row storage while its partition contract is
explicitly bottom-left anchored.

`stages/align.py` owns logical-space fixed-canvas geometry and composition. Canvas
coordinates have a top-left origin and describe pixel boundaries. Horizontal placement
is left `0`, center `floor((canvas_width - input_width) / 2)`, or right
`canvas_width - input_width`; vertical placement is the corresponding top/center rule or
`effective_baseline_y - input_height` for bottom anchors. Bottom baselines default to the
canvas height. Explicit offsets apply after base placement, and clipping is calculated
from the final placement.

Alignment derives exact per-edge overflow plus explicit visible source and destination
rectangles before copying. Empty rectangles are always `(0, 0, 0, 0)`. Composition
allocates a new transparent-black RGBA canvas and copies only the explicit visible slice;
it performs no resize, resampling, color conversion, alpha conversion, or input mutation.
The `error` policy aggregates findings and fails before publication, `warn` publishes one
warning per clipped frame, and `allow` publishes findings without clipping warnings.

`stages/io.py` owns strict prior-stage validation and the generic atomic publication
path used by scale, pixelize, and align. It validates ownership and schema markers, frame order,
unique names and safe relative paths, RGBA mode, declared dimensions, optional declared
hashes, and exact frame-directory contents. Publication builds in a temporary sibling,
validates the complete payload, atomically replaces verified same-stage output under
`--force`, and restores the prior output after rename failure where possible.

Stage metadata is the process boundary. `scale` consumes successful schema-1 `extract`
metadata; `pixelize` consumes successful schema-1 `scale` metadata; `align` consumes
successful, semantically coherent schema-1 `pixelize` metadata. These stages preserve
metadata frame order and record a typed prior-stage identity. Current extraction
metadata does not declare artifact hashes, so new stages validate hashes when present
but do not invent a second hash policy.

Expected domain errors are rendered without tracebacks. This milestone has no public
debug flag; unexpected failures return exit code 4 with internal details suppressed.

When adding a stage, keep source-pixel and logical-pixel coordinates explicit, accept
typed input plus validated immutable configuration, return typed results, centralize
serialization, and make stage order recoverable from versioned metadata rather than
filesystem enumeration. Do not create empty future-stage modules.

Algorithm-focused tests live in `tests/unit/test_scale.py`, `tests/unit/test_pixelize.py`,
and `tests/unit/test_align.py`. Configuration matrices live in
`tests/unit/test_pipeline_config.py` and `tests/unit/test_alignment_config.py`; stage
handoff, atomic publication, CLI workflow, and separate-process determinism are exercised
under `tests/integration/`. Run targeted
tests while iterating, then the complete quality gate above. Distribution verification
must also install the built wheel into an isolated environment and exercise
`extract -> scale -> pixelize -> align` through the installed console script.

## Asset distribution enforcement

`CONTRIBUTING.md` defines contributor obligations and the three asset tiers.
`ASSET-LICENSES.md` is the authoritative rights statement and declares the repository-only
directory prefixes in a delimited machine-checkable block.

Hatch excludes every declared prefix from wheels and source distributions. Release
inspection reads the same manifest, rejects normalized archive paths beneath any declared
prefix, and requires the sdist's manifest to match the repository copy. Tests verify that
the manifest prefixes remain covered by Hatch configuration and by release inspection.
The generic `assets/` directory is not globally reserved, so future redistributable
resources may be packaged deliberately without weakening the `assets/brand/` boundary.

Prefer tiny arrays or programmatically generated images for tests. Do not add visual
material whose provenance, permitted use, or distribution status is unclear.

## Uncommitted review workflow

Implementation work should be reviewed after tests, adversarial diff inspection, and a
clear `git status --short`. Do not commit, push, or publish a pull request until the
working tree has been explicitly approved.
