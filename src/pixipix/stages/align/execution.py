"""Execution of admitted alignment plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixipix import __version__
from pixipix.config import LoadedConfig
from pixipix.models import AlignmentFrame, AlignmentStageMetadata, UInt8Image
from pixipix.stages.align.planning import AlignmentStagePlan, _require_output_config
from pixipix.stages.io import LoadedStageInput, OutputFrameImage


@dataclass(slots=True)
class AlignmentRun:
    metadata: AlignmentStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


def compose_aligned_canvas(pixels: UInt8Image, frame: AlignmentFrame) -> UInt8Image:
    """Copy only the explicit visible rectangle onto a new transparent canvas."""

    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("alignment input must be uint8 RGBA")
    if pixels.shape[:2] != (frame.input_height, frame.input_width):
        raise ValueError("alignment input dimensions do not match placement metadata")
    canvas = np.zeros((frame.output_height, frame.output_width, 4), dtype=np.uint8)
    source = frame.visible_source_rectangle
    destination = frame.visible_destination_rectangle
    if source.width == 0 or source.height == 0:
        return canvas
    canvas[
        destination.y : destination.y + destination.height,
        destination.x : destination.x + destination.width,
    ] = pixels[
        source.y : source.y + source.height,
        source.x : source.x + source.width,
    ]
    return canvas


def align_stage(
    stage: LoadedStageInput,
    loaded: LoadedConfig,
    plan: AlignmentStagePlan,
) -> AlignmentRun:
    """Execute one admitted alignment plan without recomputing placement."""

    output = _require_output_config(loaded)
    frame_images = tuple(
        OutputFrameImage(
            source.relative_path,
            compose_aligned_canvas(source.pixels, metadata_frame),
        )
        for source, metadata_frame in zip(stage.frames, plan.frames, strict=True)
    )
    metadata = AlignmentStageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="align",
        status="successful",
        prior_stage=stage.identity,
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        canvas_width=output.frame_width,
        canvas_height=output.frame_height,
        anchor=output.anchor,
        configured_baseline_y=output.baseline_y,
        effective_baseline_y=output.effective_baseline_y,
        clipping_policy=output.clip_policy,
        clipping_findings=plan.clipping_findings,
        frames=plan.frames,
        warnings=plan.warnings,
    )
    return AlignmentRun(metadata, frame_images)
