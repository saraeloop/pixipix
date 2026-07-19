# PixiPix

**Tiny poses in. Tidy pixels out.**

PixiPix is a deterministic, local-first command-line tool for extracting isolated
visual frames from PNG source sheets. It removes a configured background, finds
connected foreground components, applies explicit frame names, and writes cropped
RGBA images with versioned metadata.

PixiPix is content-agnostic. It operates on pixels and geometry only: it does not
recognize subjects, infer frame meaning, or assign semantic names.

> [!NOTE]
> The current release provides inspection and extraction. Scaling, pixel-grid
> conversion, alignment, palette processing, atlas packing, and editor export are not
> implemented yet.

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
uv tool install dist/pixipix-0.1.0-py3-none-any.whl
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
pixipix extract INPUT --config CONFIG --output OUTPUT [--force]
```

Writes one RGBA PNG per accepted component plus versioned `stage.json` metadata. The
output is staged and validated before it is published.

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

### Optional inspection value

```toml
[pixelize]
source_cell_size = 6
```

The current release reports this configured value during inspection but does not use
it to transform pixels.

## Output

```text
build/extracted/
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

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Success |
| `1` | Processing or output failure |
| `2` | Configuration failure |
| `3` | Unsupported or malformed input |
| `4` | Unexpected internal error |

Expected domain failures do not print tracebacks.

## Current limitations

- one PNG source sheet per command
- strict extraction only
- component filtering by minimum and optional maximum area
- no scaling, pixel-grid conversion, alignment, palette processing, recoloring, packing,
  final manifest/report, animation generation, or editor integration
- no end-to-end `build` command yet

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture, fixture provenance, and the full
verification workflow.

PixiPix is licensed under the [Apache License 2.0](LICENSE).
