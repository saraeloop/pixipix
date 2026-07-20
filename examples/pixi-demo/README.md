# Pixi pipeline example

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

## Artwork provenance and usage

`pixi-demo-sheet.png` is the official PixiPix mascot example asset, created for this
repository and curated by saraeloop.

The artwork is **not** licensed under the repository's
[Apache License 2.0](../../LICENSE). That license applies to the PixiPix software and
other materials explicitly distributed under it.

This sheet is included solely so users can run the example pipeline locally. It may be
copied and processed for that purpose. Generated example outputs may be created for
local evaluation, but the source artwork and generated derivatives may not be
redistributed, reused, modified for unrelated purposes, or incorporated into another
project without permission.

The PixiPix name, PixiPix logo, and Pixi mascot are PixiPix brand assets.

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

Generated frames contain the Pixi artwork and carry the same reserved rights as
the source sheet: they are for local use and must not be redistributed.

## Expected result

- Candidate components: 31.
- Accepted frames: 30.
- Rejected components: one detached crescent moon with area 1,339, rejected as
  `below-minimum-area`.
- Extracted source-frame range: `132–222` pixels wide and `119–231` pixels high.
- Shared scale factor: `0.8458149779735683`.
- Scaled frame range: `112–188` pixels wide and `101–195` pixels high.
- Logical frame range: `28–47` pixels wide and `26–49` pixels high.
- Aligned output: 30 frames, each exactly `64×64`.
- Baseline: `y = 60`.
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
