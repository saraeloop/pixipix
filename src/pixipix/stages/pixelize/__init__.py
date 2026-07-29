"""Compatibility facade for deterministic logical pixel conversion."""

from __future__ import annotations

# Same-name aliases are explicit mypy re-exports; Ruff otherwise splits them one per statement.
# Keep owner groups intact; tests/architecture/test_import_compatibility.py locks this surface.
# isort: off
from .api import publish_pixelize as publish_pixelize
from .execution import (
    PixelizeRun as PixelizeRun,
    PreparedCellGrid as PreparedCellGrid,
    apply_alpha_policy as apply_alpha_policy,
    pixelize_prepared_grid as pixelize_prepared_grid,
    pixelize_stage as pixelize_stage,
    prepare_cell_grid as prepare_cell_grid,
    representative_pixel as representative_pixel,
    round_channel_half_away_from_zero as round_channel_half_away_from_zero,
)
from .planning import (
    MAX_PREPARED_PIXELS as MAX_PREPARED_PIXELS,
    CellGridProjection as CellGridProjection,
    PixelizeStagePlan as PixelizeStagePlan,
    project_cell_grid as project_cell_grid,
    project_pixelize_resources as project_pixelize_resources,
    project_pixelize_stage as project_pixelize_stage,
)
# isort: on
