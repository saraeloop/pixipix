# Pixi pipeline example

<p align="center">
  <img src="../../assets/brand/pixipix-mascot-logo.png" alt="Pixi mascot and PixiPix wordmark" width="400">
</p>

## Purpose

Pixi is the official PixiPix mascot and the first official real-world pipeline
example.

Pixi receives no special handling. The same pipeline and contracts apply to any valid
raster sheet. Pixi remains example content only: the core pipeline contains no
Pixi-specific names, branches, assumptions, or behavior.

This example runs the complete current pipeline:

```text
inspect → extract → scale → pixelize → align
```

> **Preview images:** Source-sheet and aligned-output previews will be added soon. The
> example is already runnable using the commands below.

## What this example demonstrates

The source sheet is one large PNG containing thirty differently sized Pixi poses
drawn in a chunky pixel-art style, where each _apparent_ pixel is really a block
of roughly 4×4 image pixels. The pipeline turns that one sheet into thirty
clean, same-sized, true logical-pixel frames. Stage by stage:

**`inspect` looks before anything is written.** It reports what PixiPix sees in
the sheet: 31 disconnected regions of visible pixels ("components"), their
sizes, and how they will be ordered. Nothing is guessed and nothing is
produced — this is how you calibrate a config before committing to it.

**`extract` cuts the sheet into frames.** Each connected region of visible
pixels becomes one frame. Thirty regions are large enough to be accepted as
poses; one — a small crescent moon floating above the sleeping pose — falls
below the configured minimum area and is rejected. PixiPix follows pixels, not
meaning: it cannot know the moon "belongs to" the sleeper, so the config
decides. The thirty accepted frames are named in reading order, left to right,
top to bottom.

**`scale` shrinks everything with one shared ruler.** A single scale factor is
computed from one reference pose and applied to every frame identically. Small
poses stay small relative to big ones; the sheet's relative geometry is
preserved under one shared factor, subject to deterministic integer rounding.
PixiPix deliberately never normalizes frames individually —
if two poses were drawn at different sizes, the output honestly keeps them at
different sizes.

**`pixelize` collapses fake pixels into real ones.** The art's 4×4 blocks each
become exactly one logical RGBA pixel, chosen deterministically from the
pixels in that cell. A pose that occupied ~180×190 image pixels becomes a true
~45×48 pixel sprite. This is the step that converts enlarged pixel-style
artwork into true logical-pixel output.

**`align` puts every frame on the same stage.** Each logical frame is placed
on an identical transparent 64×64 canvas, bottom-centered on a shared baseline
four pixels above the canvas bottom — so every standing pose has its feet on
the same floor. Four flying poses are explicitly nudged upward with configured
offsets so bottom-alignment doesn't ground them. Pixels are copied exactly;
alignment never resizes or recolors anything.

The result is thirty deterministic 64×64 frames that downstream consumers can
interchange without performing their own geometry repair. Running the same
pipeline twice produces byte-identical output trees.

## Artwork provenance and usage

`pixi-demo-sheet.png` is the official PixiPix mascot example asset, created for this
repository and curated by saraeloop.

The sheet, Pixi mascot, PixiPix logo, and generated example outputs are reserved brand
assets. The complete example remains available in repository checkouts for local use and
is excluded from wheels and source distributions. See the authoritative
[asset rights and distribution manifest](../../ASSET-LICENSES.md) before copying,
processing, or otherwise using these assets.

Source integrity: the checked-in image is the supplied example artwork without
modification or regeneration.

SHA-256: `1fed0f9849070d0b9e89bf8708c36e3ce70521452a90e180b289268f9f2e27e2`.

## Source contract

- Source: `pixi-demo-sheet.png`, a `1194×1317` RGBA PNG with a transparent background.
- Extraction: 8-connected alpha foreground with `alpha_threshold = 8`.
- Components: 31 candidates, 30 accepted poses, and one rejected detached effect.
- Ordering: accepted components are named in deterministic six-column, five-row reading
  order.
- Area calibration: `minimum_area = 1500` keeps every full Pixi pose in this sheet while
  rejecting the detached crescent moon, whose area is 1,339 source pixels.
- Logical sampling: each logical pixel represents a configured `4×4` source-pixel cell.
- Scaling: one shared factor is derived from `pixi-stand-cube`, whose height is normalized
  to 48 logical pixels.
- Alignment: every logical frame is placed on a transparent `64×64` canvas with a shared
  bottom baseline at `y = 60`.

The `minimum_area` value is specific to this source. Users must inspect and recalibrate
the threshold for their own artwork; PixiPix does not infer whether a disconnected
effect belongs to a nearby frame.

## Frame names

The stable names below follow the source's deterministic reading order:

| Order | Name                | Visible state or action              |
| ----: | ------------------- | ------------------------------------ |
|     1 | `pixi-stand-cube`   | Standing with wand and cube          |
|     2 | `pixi-wave-wand`    | Winking with raised wand             |
|     3 | `pixi-point-cube`   | Holding a cube while pointing        |
|     4 | `pixi-walk-side`    | Walking in side profile              |
|     5 | `pixi-stand-back`   | Standing rear view                   |
|     6 | `pixi-cast`         | Casting with raised wand             |
|     7 | `pixi-fly-cube`     | Flying while carrying a cube         |
|     8 | `pixi-sit`          | Sitting                              |
|     9 | `pixi-leap`         | Leaping with raised wand             |
|    10 | `pixi-read-map`     | Reading a map or scroll              |
|    11 | `pixi-fly-wand`     | Flying with wand extended            |
|    12 | `pixi-hold-cube`    | Holding a cube with closed eyes      |
|    13 | `pixi-fly`          | Flying horizontally                  |
|    14 | `pixi-crawl`        | Crawling                             |
|    15 | `pixi-nap`          | Napping on the ground                |
|    16 | `pixi-hide-wall`    | Peeking from behind a wall           |
|    17 | `pixi-confused`     | Confused with question mark          |
|    18 | `pixi-cheer`        | Cheering with raised fist            |
|    19 | `pixi-heart`        | Holding a heart                      |
|    20 | `pixi-ready`        | Ready stance with raised fists       |
|    21 | `pixi-magnify`      | Looking through a magnifier          |
|    22 | `pixi-cry`          | Crying                               |
|    23 | `pixi-present-wand` | Presenting a wand                    |
|    24 | `pixi-point`        | Pointing upward while holding a cube |
|    25 | `pixi-magic-circle` | Casting inside a magic circle        |
|    26 | `pixi-cast-back`    | Rear-view crescent spell cast        |
|    27 | `pixi-read-book`    | Reading an open book                 |
|    28 | `pixi-cookie`       | Eating a cookie                      |
|    29 | `pixi-laptop`       | Working at a laptop                  |
|    30 | `pixi-sleep`        | Sleeping in bed                      |

Some poses are narratively close—particularly the casting, flying, pointing, and
sleeping variants. Their names use visible props, orientation, or posture to remain
stable without claiming semantic recognition by PixiPix. `pixi-nap` distinguishes the
ground pose from the final `pixi-sleep` bed pose.

## Alpha policy

The example was run with both supported policies at `alpha_threshold = 128`:

- `preserve` retains readable body and hair outlines, translucent wings, and
  semi-transparent edge pixels at the small logical size.
- `binary` introduces conspicuous transparent gaps through the body, hair, and wings
  because selected source cells below the threshold become fully transparent.

The committed configuration therefore uses `alpha_policy = "preserve"`. This is an
observed choice for this source, not special mascot handling.

## Alignment offsets

All frames share `baseline_y = 60`. Four clearly airborne poses use explicit vertical
offsets so bottom alignment does not make them appear grounded:

| Frame           |    Offset | Reason                                            |
| --------------- | --------: | ------------------------------------------------- |
| `pixi-fly-cube` | `dy = -4` | Preserve visible hovering while carrying the cube |
| `pixi-leap`     | `dy = -4` | Keep the bent-leg leap above the baseline         |
| `pixi-fly-wand` | `dy = -4` | Preserve the airborne wand pose                   |
| `pixi-fly`      | `dy = -6` | Preserve the stronger horizontal flying pose      |

These offsets are configured explicitly; PixiPix does not infer them. Each produces the
expected `PX_ALIGN_OFFSET_001` warning, and none causes clipping.

## Run commands

Run every command from the repository root:

```bash
uv run pixipix inspect \
  examples/pixi-demo/pixi-demo-sheet.png \
  --config examples/pixi-demo/pixipix.toml

uv run pixipix run \
  examples/pixi-demo/pixi-demo-sheet.png \
  --config examples/pixi-demo/pixipix.toml \
  --output build/pixi-demo/run
```

`pixipix run` is the simplest complete-demo path. It executes Extract → Scale →
Pixelize → Align and publishes one run root containing inspectable `extract/`, `scale/`,
`pixelize/`, and `align/` stage trees.

The individual commands remain available for explicit stage-by-stage inspection and
debugging:

```bash
uv run pixipix extract \
  examples/pixi-demo/pixi-demo-sheet.png \
  --config examples/pixi-demo/pixipix.toml \
  --output build/pixi-demo/extracted

uv run pixipix scale \
  build/pixi-demo/extracted \
  --config examples/pixi-demo/pixipix.toml \
  --output build/pixi-demo/scaled

uv run pixipix pixelize \
  build/pixi-demo/scaled \
  --config examples/pixi-demo/pixipix.toml \
  --output build/pixi-demo/pixelized

uv run pixipix align \
  build/pixi-demo/pixelized \
  --config examples/pixi-demo/pixipix.toml \
  --output build/pixi-demo/aligned
```

Generated output belongs under the ignored `build/` directory and must not be committed.

Generated frames contain the Pixi artwork and remain subject to the
[authoritative asset rights statement](../../ASSET-LICENSES.md).

## What to look at afterward

After `pixipix run` completes:

- Open `build/pixi-demo/run/align/frames/` and page through the thirty PNGs —
  every frame is 64×64, every standing pose shares the same floor, and the
  four flying poses hover above it.
- Compare `pixi-stand-cube.png` with its source region: the same pose and
  visual palette, converted into true logical pixels.
- Open `build/pixi-demo/run/extract/stage.json` and find the rejected component —
  the crescent moon, listed with its area and rejection reason rather than
  silently discarded.
- Open `build/pixi-demo/run/align/stage.json` for each frame's recorded
  placement: base position, offset, final position, and overflow (all zeros —
  nothing clipped).
- Run the pipeline a second time into a different output directory and diff
  the two trees: every byte is identical.

## Expected result

- Candidate components: 31.
- Accepted frames: 30.
- Rejected components: one detached crescent moon with area 1,339, rejected as
  `below-minimum-area`.
- Extracted source-frame range: `132–222` pixels wide and `119–231` pixels high.
- Shared scale factor: `0.8458149779735683` — one ratio applied to every
  frame, preserving relative geometry without per-frame normalization.
- Scaled frame range: `112–188` pixels wide and `101–195` pixels high.
- Logical frame range: `28–47` pixels wide and `26–49` pixels high — the
  frames genuinely differ in size until alignment, by design.
- Aligned output: 30 frames, each exactly `64×64`.
- Baseline: `y = 60` — a shared floor four pixels above the canvas bottom.
- Explicit offsets: four, listed above.
- Warnings: four expected `PX_ALIGN_OFFSET_001` warnings and no unexpected warnings.
- Clipping findings: zero.

## Honest limitation

Connected-component extraction follows pixels, not narrative meaning. Detached effects
such as punctuation, sparkles, sleep marks, or moons can become independent candidates.
For this source, the crescent moon above `pixi-sleep` is the only separate rejected
component. Other visible decorative marks are part of accepted connected components and
remain with those poses.

The area filter is an explicit, reproducible choice rather than semantic attachment.
PixiPix does not infer which detached effect belongs to which frame, and this example
does not add component grouping or mascot-specific behavior.
