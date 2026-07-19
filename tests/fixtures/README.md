# Synthetic fixture provenance

All PNG files in this directory are original geometric test data created for PixiPix by
the repository owner in 2026. They contain only programmatically drawn rectangles and
pixels, no third-party or model-generated artwork. They are distributed under the
repository's Apache-2.0 license and may be redistributed, modified, and used in
automated tests.

Run `uv run python tests/fixtures/generate_fixtures.py` to reproduce their exact pixels.
Pillow encoder bytes can vary if the locked dependency changes; pixel content is the
fixture contract.

- `transparent-multi.png`: two alpha-backed components plus one noise pixel.
- `solid-background.png`: two components on a uniform RGB background.
- `connectivity.png`: diagonally adjacent pixels for four/eight-connectivity tests.
- `multi-row.png`: components arranged across two rows.
- `robot-geometric.png`: a non-animal geometric asset used by CLI verification.
