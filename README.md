<p align="center">
  <img src="https://raw.githubusercontent.com/saraeloop/pixipix/main/assets/brand/pixipix-logo.png" alt="PixiPix logo" width="400">
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/pixipix.svg" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.12-18181b" alt="Python 3.12">
  <img src="https://img.shields.io/github/stars/saraeloop/pixipix?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/badge/license-Apache%202.0-64748b" alt="Apache 2.0 license">
</p>

<p align="center"><strong>Tiny poses in. Tidy pixels out.</strong></p>

PixiPix is a deterministic, local-first command-line tool for extracting isolated
visual frames from PNG source sheets, scaling them with one shared geometric ruler,
converting configured pseudo-pixel cells into true logical RGBA pixels, and placing
those frames on deterministic fixed-size canvases.

PixiPix is content- and source-agnostic. It never recognizes subjects or
infers what a frame depicts, and it never asks how a sheet was made:
hand-drawn, rendered, scanned, exported, procedurally generated, and
AI-generated raster inputs all work identically.

> **Note:**
> PixiPix is in active development. APIs and output contracts may change
> during the alpha series.

## Example

Pixi, the official PixiPix mascot, demonstrates the complete current
pipeline on a real multi-frame raster source:

```text
inspect → extract → scale → pixelize → align
```

See [`examples/pixi-demo/`](examples/pixi-demo/).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- PNG source images

PixiPix performs no network requests, telemetry, cloud uploads, or AI inference.

## Install

Clone the repository and create the locked environment:

```bash
uv sync
uv run pixipix --help
```

To build an installable wheel:

```bash
uv build
uv tool install dist/pixipix-0.1.0a4-py3-none-any.whl
pixipix --help
```

## Quick start

Create `pixipix.toml`:

```toml
[project]
name = "sprite-source"
strict = true

[source]
format = "png"
expected_components = 2

[background]
mode = "alpha"
alpha_threshold = 8

[extract]
connectivity = 8
minimum_area = 4
maximum_area = 10000
padding = 1
row_tolerance = 2

[frames]
names = ["frame-a", "frame-b"]

[scale]
mode = "reference-frame-width"
reference_frame = "frame-a"
target_size = 24

[pixelize]
source_cell_size = 6
representative = "alpha-weighted-majority"
alpha_policy = "binary"
alpha_threshold = 128
remainder_policy = "pad-transparent"

[output]
frame_width = 48
frame_height = 48
anchor = "bottom-center"
baseline_y = 44
clip_policy = "error"

[frame_offsets.frame-b]
dx = 1
dy = -1
```

Inspect the source without writing files:

```bash
uv run pixipix inspect source.png --config pixipix.toml
```

Extract the configured frames:

```bash
uv run pixipix extract source.png \
  --config pixipix.toml \
  --output build/extracted

uv run pixipix scale build/extracted \
  --config pixipix.toml \
  --output build/scaled

uv run pixipix pixelize build/scaled \
  --config pixipix.toml \
  --output build/pixelized

uv run pixipix align build/pixelized \
  --config pixipix.toml \
  --output build/aligned
```

The accepted component count must match both `source.expected_components` and the
number of configured frame names. PixiPix fails instead of guessing when they differ.

## Commands

### `pixipix inspect`

```text
pixipix inspect INPUT --config CONFIG
```

Reports deterministic facts including:

- source dimensions, input mode, and alpha presence
- normalized RGBA mode
- selected background behavior and foreground bounds
- candidate, accepted, and rejected components
- component bounds, areas, rejection reasons, and deterministic order
- configured frame-name assignments when counts match
- configured source cell size when present

`inspect` never infers a source cell size and does not write output artifacts.

### `pixipix extract`

```text
pixipix extract INPUT --config CONFIG --output OUTPUT [--force] [--show-warnings]
```

Writes one RGBA PNG per accepted component plus versioned `stage.json` metadata. The
output is staged and validated before it is published.

### `pixipix scale`

```text
pixipix scale INPUT_DIR --config CONFIG --output OUTPUT [--force] [--show-warnings]
```

Consumes a valid extraction-stage directory and applies one global scale factor to
every frame in source pixel space. Reference modes derive the factor from one named
frame and set its configured width or height target exactly. Optional, explicit
per-frame multipliers are recorded and always produce warnings. Scaling uses BOX over
float32 premultiplied RGBA channels, then deterministically un-premultiplies and
normalizes transparent pixels to prevent dark fringes.

### `pixipix pixelize`

```text
pixipix pixelize INPUT_DIR --config CONFIG --output OUTPUT [--force] [--show-warnings]
```

Consumes a valid scale-stage directory and emits one logical RGBA pixel per configured
source cell. The grid is anchored at bottom-left: incomplete space belongs to the top
and right edges. Output is always at 1× logical resolution and is not aligned,
palette-locked, or packed.

### `pixipix align`

```text
pixipix align INPUT_DIR --config CONFIG --output OUTPUT [--force] [--show-warnings]
```

Consumes valid pixelize-stage output and places every logical RGBA frame on the same
configured transparent-black canvas. Alignment copies visible pixels exactly; it never
resizes, resamples, recolors, or changes alpha. Placement, per-edge overflow, and visible
source/destination rectangles are recorded in versioned metadata.

### Warnings and automation

Successful write-stage warnings are printed to stderr while the existing success message remains
on stdout. By default, each command prints only warnings created by that stage; pass
`--show-warnings` to include the complete inherited warning history in stored order.
`stage.json` remains the structured source of truth for warning data.

Warnings do not turn a successful command into a failure: exit code `0` remains the
authoritative success signal. Scripts that treat any stderr output as failure may need
adjustment.

### Other commands

```bash
pixipix --help
pixipix version
python -m pixipix
```

## Configuration reference

PixiPix parses TOML strictly. Unknown keys, unsupported sections, invalid values,
duplicate names, unsafe filenames, and inconsistent counts are configuration errors.

### Source limits

```toml
[source]
format = "png"             # only png is supported
expected_components = 2    # optional, but recommended
max_width = 4096
max_height = 4096
max_pixels = 16777216
max_components = 128
```

The limits are checked before expensive image allocations where possible.
`max_pixels` has a fixed ceiling of 16,777,216 pixels, and `max_components` may not
exceed `max_pixels`; configuration cannot silently disable those resource bounds.
RGB, RGBA, indexed, grayscale, and grayscale-alpha PNGs are accepted and normalized to
an owned `uint8` RGBA buffer. Malformed, truncated, and decoder-limit inputs fail as
unsupported input.

### Background modes

Transparent source:

```toml
[background]
mode = "alpha"
alpha_threshold = 8
```

Known solid color:

```toml
[background]
mode = "explicit-color"
color = "#f4e46a"
tolerance = 0.02
alpha_threshold = 8
```

Color sampled from all four corners:

```toml
[background]
mode = "corner-color"
tolerance = 0.02
alpha_threshold = 8
sample_corners = true
```

Color modes use normalized maximum per-channel distance. The largest absolute channel
difference divided by 255 must be less than or equal to `tolerance`. Six-digit colors
compare RGB; eight-digit colors compare RGBA. Corner mode compares RGBA samples and
fails if the corners disagree beyond tolerance. Pixels below `alpha_threshold` are
always background.

### Extraction

```toml
[extract]
connectivity = 8     # 4 or 8
minimum_area = 4
maximum_area = 10000 # optional
padding = 1
row_tolerance = 2
```

Components smaller than `minimum_area` or larger than `maximum_area` remain visible in
inspection and stage metadata as rejected components.

### Frame names

```toml
[frames]
names = ["frame-a", "frame-b"]
```

Names are preserved unchanged in metadata and assigned only after deterministic
ordering and count validation. Path separators, traversal and absolute-path syntax,
controls, surrounding whitespace, trailing dots, Windows-reserved basenames, and
overlong filenames are rejected. Filename normalization is deterministic, and
normalized filenames must remain unique even on case-insensitive filesystems.

### Global scale

Choose exactly one scale mode:

```toml
[scale]
mode = "explicit-factor"
factor = 0.75
```

```toml
[scale]
mode = "reference-frame-width" # or reference-frame-height
reference_frame = "frame-a"
target_size = 24                # logical pixels
```

Reference modes use `target_size × pixelize.source_cell_size` as the exact source-space
target. All other dimensions use the same factor with round-half-away-from-zero;
non-empty dimensions remain at least one source pixel. A reference frame cannot have
an override. An exceptional non-reference correction is explicit and warned:

```toml
[frame_overrides.frame-b]
scale_multiplier = 0.96
```

PixiPix never infers or suggests per-frame normalization.

### Logical pixelization

```toml
[pixelize]
source_cell_size = 6
representative = "alpha-weighted-majority"
alpha_policy = "binary"
alpha_threshold = 128
remainder_policy = "pad-transparent"
```

`source_cell_size` is required by `pixelize` and by reference scaling. Representative
strategies are exact RGBA `majority` with first row-major tie-breaking, locked `center`
sampling, and the default `alpha-weighted-majority`, which ignores transparent RGB and
weights visible RGB groups by alpha. Alpha is either `binary` at the configured
inclusive threshold or explicitly `preserve`d as the selected-color opacity.

Remainder policies are `pad-transparent` (minimal top/right transparent-black padding),
`error`, and `crop-with-warning` (top/right incomplete strips only). Cropping that would
reduce a non-empty frame to zero dimensions is rejected.

### Fixed-canvas alignment

```toml
[output]
frame_width = 48
frame_height = 48
anchor = "bottom-center"
baseline_y = 44
clip_policy = "error"

[frame_offsets.frame-b]
dx = 2
dy = -1
```

`frame_width`, `frame_height`, and `anchor` are required by `align`. Supported anchors
are `top-left`, `top-center`, `top-right`, `center-left`, `center`, `center-right`,
`bottom-left`, `bottom-center`, and `bottom-right`.

Canvas coordinates use a top-left origin and refer to pixel boundaries. For bottom
anchors, `baseline_y` is the boundary where the input frame's bottom edge lands before
offsets; it defaults to `frame_height` and may range from zero through `frame_height`,
inclusive. Non-bottom anchors reject `baseline_y`.

Center placement uses mathematical floor:

```text
floor((canvas_size - input_size) / 2)
```

An odd remainder leaves the extra transparent pixel on the right or bottom. The same
floor rule applies to negative differences when an input is larger than its canvas.
Explicit integer frame offsets apply after anchor placement. A declared offset must
change at least one axis; `{dx = 0, dy = 0}` is rejected during configuration validation,
and every valid declared offset contributes one deterministic alignment warning when
alignment metadata is published.

Clipping is evaluated after offsets. Policies are:

- `error` (default): aggregate every clipped frame and publish nothing
- `warn`: publish, record findings, and add one warning per clipped frame
- `allow`: publish and record findings without clipping warnings

All policies retain exact `leftOverflow`, `topOverflow`, `rightOverflow`, and
`bottomOverflow` counts. Metadata also records visible source and destination rectangles
as `x`, `y`, `width`, and `height`; every empty rectangle uses the canonical all-zero
representation.

## Output

```text
build/extracted/
├── .pixipix-output
├── frames/
│   ├── frame-a.png
│   └── frame-b.png
└── stage.json

build/scaled/
├── .pixipix-output
├── frames/
│   ├── frame-a.png
│   └── frame-b.png
└── stage.json

build/pixelized/
├── .pixipix-output
├── frames/
│   ├── frame-a.png
│   └── frame-b.png
└── stage.json

build/aligned/
├── .pixipix-output
├── frames/
│   ├── frame-a.png
│   └── frame-b.png
└── stage.json
```

`stage.json` records:

- schema and PixiPix versions
- exact-source and effective-configuration SHA-256 hashes
- normalized source and background-removal facts
- candidate, accepted, and rejected components
- ordered frame names and relative paths
- original and padded source bounds
- component areas and deterministic source order
- warnings and successful status

Public artifacts contain no timestamps, absolute machine paths, or temporary paths.
Scale metadata records prior-stage identity, config hashes, the shared global factor,
reference measurements, overrides, effective frame factors, dimensions, and warnings.
Pixelize metadata records prior-stage identity, the bottom-left grid origin, cell size,
selection and alpha policies, per-frame top/right padding or crop, logical dimensions,
and warnings. Align metadata records the canvas, anchor, configured/effective baseline,
clipping policy, offsets, final placement, exact overflow, explicit visible rectangles,
and warnings. Frame order always comes from `stage.json`, never directory enumeration.

## Output safety

PixiPix does not merge new files into stale output:

- a non-empty output directory is rejected by default
- an existing empty output directory is safely replaced without requiring `--force`
- `--force` replaces only output with a valid ownership marker plus a coherent,
  successful `stage.json` contract and all referenced RGBA PNG frames
- unowned directories, symlink targets, and untrusted symlink parents are not
  destructively replaced; the root-owned standard `/tmp` alias is supported
- output is built in a temporary sibling directory
- a previous owned output is restored if atomic publication fails where practical

## Determinism

Component discovery scans pixels in row-major order. Four-connectivity visits up,
left, right, then down. Eight-connectivity visits up-left, up, up-right, left, right,
down-left, down, then down-right.

Accepted components are first sorted by top, left, ascending area, and discovery index.
Each row is anchored to the top coordinate of its first component; another component
joins the first existing row whose anchor differs by at most `row_tolerance` (inclusive).
The fixed anchor prevents pairwise chaining from merging distant rows. Rows are ordered
by anchor top; components within a row use left, ascending area, then discovery index.

JSON uses sorted keys, two-space indentation, UTF-8, finite numbers, Unix-style relative
paths, and exactly one trailing newline. PNG output is RGBA, excludes source metadata,
uses explicit compression settings, and zeroes RGB channels for fully transparent
pixels.

Geometric and channel quantization use separate round-half-away-from-zero helpers.
Scale BOX filtering operates on float32 premultiplied red, green, blue, and alpha
channels; un-premultiplication occurs in float64 and channels are quantized once.
Fully opaque input uses the equivalent native RGBA BOX path for ordinary BOX identity.
Pixel representatives use fixed row-major tie rules. Sequential `scale` then `pixelize`
is the canonical path; no fused implementation exists.

## Exit codes

| Code | Meaning                        |
| ---: | ------------------------------ |
|  `0` | Success                        |
|  `1` | Processing or output failure   |
|  `2` | Configuration failure          |
|  `3` | Unsupported or malformed input |
|  `4` | Unexpected internal error      |

Expected domain failures do not print tracebacks.

## Current limitations

- one PNG source sheet per command
- strict configuration and one shared scale factor per extracted sheet
- component filtering by minimum and optional maximum area
- explicit source-cell size; no source-cell inference or automatic frame normalization
- explicit fixed canvas and placement; no automatic canvas, anchor, baseline, or offset
  inference
- no palette processing, recoloring, atlas packing, final manifest/report, animation
  generation, or editor integration
- no end-to-end `build` command yet

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor setup, project conventions,
fixture requirements, and the full verification workflow.

PixiPix is licensed under the
[Apache License 2.0](https://github.com/saraeloop/pixipix/blob/main/LICENSE).
