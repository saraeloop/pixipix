"""Compatibility facade for deterministic global source-space scaling."""

from __future__ import annotations

# Same-name aliases are explicit mypy re-exports; Ruff otherwise splits them one per statement.
# Keep owner groups intact; tests/architecture/test_import_compatibility.py locks this surface.
# isort: off
from .api import publish_scale as publish_scale
from .execution import (
    ScaleRun as ScaleRun,
    premultiplied_box_resize as premultiplied_box_resize,
    scale_stage as scale_stage,
)
from .geometry import (
    round_channel_half_away_from_zero as round_channel_half_away_from_zero,
    round_half_away_from_zero as round_half_away_from_zero,
    transformed_dimension as transformed_dimension,
)
from .planning import (
    MAX_TRANSFORMED_PIXELS as MAX_TRANSFORMED_PIXELS,
    ScaleStagePlan as ScaleStagePlan,
    project_scale_resources as project_scale_resources,
    project_scale_stage as project_scale_stage,
)
# isort: on
