"""Supported inspection and extraction-compute boundaries."""

from __future__ import annotations

from pathlib import Path

from pixipix.config import LoadedConfig
from pixipix.errors import ProcessingError
from pixipix.models import ExtractionResult, ExtractionRun, InspectionResult
from pixipix.resources import enforce_resource_policy

from .analysis import _analyze
from .execution import _materialize_frame_crop
from .planning import project_extract_resources, project_extracted_frames


def inspect_source(input_path: Path, loaded: LoadedConfig) -> InspectionResult:
    analysis = _analyze(input_path, loaded)
    assignments: tuple[str, ...] | None = None
    if len(analysis.ordered) == len(loaded.config.frames.names):
        assignments = loaded.config.frames.names
    return InspectionResult(
        source=analysis.source.metadata,
        background=analysis.background,
        candidates=analysis.component_map.components,
        accepted=analysis.accepted,
        rejected=analysis.rejected,
        ordered=analysis.ordered,
        frame_assignments=assignments,
        configured_source_cell_size=loaded.config.pixelize.source_cell_size,
    )


def extract_source(input_path: Path, loaded: LoadedConfig) -> ExtractionRun:
    analysis = _analyze(input_path, loaded)
    config = loaded.config
    accepted_count = len(analysis.ordered)
    expected = config.source.expected_components
    if expected is not None and accepted_count != expected:
        raise ProcessingError(
            "PX_EXTRACT_002",
            "extract",
            f"accepted component count {accepted_count} does not match expected count {expected}",
            path=input_path.name,
            remediation="inspect component bounds and adjust extraction filters",
        )
    if accepted_count != len(config.frames.names):
        raise ProcessingError(
            "PX_EXTRACT_003",
            "name",
            f"accepted component count {accepted_count} does not match "
            f"frame-name count {len(config.frames.names)}",
            path=input_path.name,
            remediation="inspect component bounds and update thresholds or frame names",
        )

    height, width = analysis.mask.shape
    frames = project_extracted_frames(analysis, loaded)
    projection = project_extract_resources(width * height, frames)
    enforce_resource_policy(projection, config.resources)
    images = tuple(
        _materialize_frame_crop(analysis, component, frame)
        for component, frame in zip(analysis.ordered, frames, strict=True)
    )
    result = ExtractionResult(
        source=analysis.source.metadata,
        background=analysis.background,
        candidates=analysis.component_map.components,
        accepted=analysis.accepted,
        rejected=analysis.rejected,
        ordered=analysis.ordered,
        frames=frames,
    )
    return ExtractionRun(result=result, frame_images=images)
