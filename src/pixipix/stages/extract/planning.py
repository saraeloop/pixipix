"""Allocation-free Extract geometry and resource projection."""

from __future__ import annotations

from pathlib import PurePosixPath

from pixipix.config import LoadedConfig
from pixipix.models import ExtractedFrame, Rect
from pixipix.resources import ResourceProjection

from .analysis import _Analysis


def _padded_bounds(bounds: Rect, padding: int, width: int, height: int) -> Rect:
    return Rect(
        left=max(0, bounds.left - padding),
        top=max(0, bounds.top - padding),
        right=min(width, bounds.right + padding),
        bottom=min(height, bounds.bottom + padding),
    )


def project_extract_resources(
    source_area: int,
    frames: tuple[ExtractedFrame, ...],
) -> ResourceProjection:
    """Project the locked extract explicit-buffer formula."""

    frame_areas = tuple(frame.padded_bounds.width * frame.padded_bounds.height for frame in frames)
    aggregate_output = sum(frame_areas)
    largest_frame = max(frame_areas, default=0)
    materialization = 9 * source_area + 4 * aggregate_output + largest_frame
    publication = 4 * aggregate_output + 5 * largest_frame
    return ResourceProjection(
        "extract",
        source_area,
        aggregate_output,
        max(materialization, publication),
    )


def project_extracted_frames(
    analysis: _Analysis,
    loaded: LoadedConfig,
) -> tuple[ExtractedFrame, ...]:
    """Project every padded/clipped frame without materializing a crop."""

    config = loaded.config
    height, width = analysis.mask.shape
    return tuple(
        ExtractedFrame(
            name=name,
            relative_path=PurePosixPath("frames") / filename,
            source_order=source_order,
            discovery_index=component.discovery_index,
            component_area=component.area,
            original_bounds=component.bounds,
            padded_bounds=_padded_bounds(
                component.bounds,
                config.extract.padding,
                width,
                height,
            ),
        )
        for source_order, (component, name, filename) in enumerate(
            zip(
                analysis.ordered,
                config.frames.names,
                config.frames.filenames,
                strict=True,
            )
        )
    )
