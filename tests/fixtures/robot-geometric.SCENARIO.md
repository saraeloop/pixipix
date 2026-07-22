# Installed-distribution smoke scenario

This static contract was established from the source checkout before the installed
distribution smoke assertions were implemented. The fixture is the neutral,
programmatically generated `robot-geometric.png`, configured by
`robot-geometric.toml`.

- Inspection finds exactly 2 candidate components and accepts exactly 2, in the
  deterministic frame order `idle`, `signal`; it rejects 0 components.
- Extraction publishes `frames/idle.png` at 8x10 and `frames/signal.png` at 9x10.
- Identity scaling (`explicit-factor`, factor `1.0`) preserves those dimensions.
- Pixelization uses source cell size 4 and `pad-transparent`. It publishes `idle` at
  2x3 logical pixels (2 top-padding pixels) and `signal` at 3x3 logical pixels
  (2 top-padding and 3 right-padding pixels).
- Alignment is the final stage (`align`) and publishes successful schema-1 metadata
  on a 4x4 canvas. Its frames remain ordered as `idle`, `signal`, with final files
  `frames/idle.png`, `frames/signal.png`.
- Both final files are PNG images in exact `RGBA` mode at exactly 4x4 pixels.
- Extract, scale, pixelize, and align each publish exactly 0 warnings; the expected
  warning-code sequence for every stage is therefore empty.
- Alignment reports no clipping findings. The final placements are `(1, 0)` for
  `idle` and `(0, 0)` for `signal`.

PNG encoder bytes are not part of this contract; metadata identities, order,
dimensions, mode, stage publication, and warning facts are.
