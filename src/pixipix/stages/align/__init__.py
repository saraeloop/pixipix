"""Compatibility facade for deterministic fixed-canvas alignment."""

from __future__ import annotations

# Same-name aliases are explicit mypy re-exports; Ruff otherwise splits them one per statement.
# Keep owner groups intact; tests/architecture/test_import_compatibility.py locks this surface.
# isort: off
from .api import publish_align as publish_align
from .execution import (
    AlignmentRun as AlignmentRun,
    align_stage as align_stage,
    compose_aligned_canvas as compose_aligned_canvas,
)
from .geometry import (
    EMPTY_RECTANGLE as EMPTY_RECTANGLE,
    calculate_alignment_frame as calculate_alignment_frame,
    mathematical_floor_center as mathematical_floor_center,
)
from .planning import (
    AlignmentStagePlan as AlignmentStagePlan,
    clipping_finding as clipping_finding,
    project_align_resources as project_align_resources,
    project_align_stage as project_align_stage,
)
# isort: on
