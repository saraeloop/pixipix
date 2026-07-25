"""Deterministic fixed-canvas placement in logical pixel space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from pixipix import __version__
from pixipix.config import LoadedConfig, OutputConfig
from pixipix.errors import AlignmentClippingError, ConfigurationError, UnsupportedInputError
from pixipix.models import (
    AlignmentClippingFinding,
    AlignmentFrame,
    AlignmentRectangle,
    AlignmentStageMetadata,
    ProcessingWarning,
    UInt8Image,
)
from pixipix.resources import ResourceProjection, enforce_resource_policy
from pixipix.stages.io import (
    LoadedStageInput,
    OutputFrameImage,
    ValidatedStageInput,
    decode_stage_input,
    publish_stage_output,
    validate_stage_input,
    validate_stage_output_target,
)

EMPTY_RECTANGLE = AlignmentRectangle(0, 0, 0, 0)


@dataclass(slots=True)
class AlignmentRun:
    metadata: AlignmentStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


@dataclass(frozen=True, slots=True)
class AlignmentStagePlan:
    frames: tuple[AlignmentFrame, ...]
    clipping_findings: tuple[AlignmentClippingFinding, ...]
    warnings: tuple[ProcessingWarning, ...]
    projection: ResourceProjection


def mathematical_floor_center(canvas_size: int, input_size: int) -> int:
    """Return floor((canvas_size - input_size) / 2) using exact integers.

    Python integer floor division is mathematical floor, including for negative
    odd differences; it therefore cannot truncate negative halves toward zero.
    """

    return (canvas_size - input_size) // 2


def _axis_parts(anchor: str) -> tuple[str, str]:
    if anchor == "center":
        return "center", "center"
    vertical, horizontal = anchor.split("-", maxsplit=1)
    return horizontal, vertical


def calculate_alignment_frame(
    *,
    name: str,
    relative_path: PurePosixPath,
    source_order: int,
    input_width: int,
    input_height: int,
    output: OutputConfig,
    dx: int = 0,
    dy: int = 0,
) -> AlignmentFrame:
    """Calculate placement, overflow, and explicit visible rectangles."""

    if input_width <= 0 or input_height <= 0:
        raise ValueError("alignment input dimensions must be positive")
    horizontal, vertical = _axis_parts(output.anchor)
    if horizontal == "left":
        base_x = 0
    elif horizontal == "center":
        base_x = mathematical_floor_center(output.frame_width, input_width)
    else:
        base_x = output.frame_width - input_width
    if vertical == "top":
        base_y = 0
    elif vertical == "center":
        base_y = mathematical_floor_center(output.frame_height, input_height)
    else:
        if output.effective_baseline_y is None:
            raise ValueError("bottom anchors require an effective baseline")
        base_y = output.effective_baseline_y - input_height

    final_x = base_x + dx
    final_y = base_y + dy
    left_overflow = max(0, -final_x)
    top_overflow = max(0, -final_y)
    right_overflow = max(0, final_x + input_width - output.frame_width)
    bottom_overflow = max(0, final_y + input_height - output.frame_height)
    clipped = any((left_overflow, top_overflow, right_overflow, bottom_overflow))

    source_left = max(0, -final_x)
    source_top = max(0, -final_y)
    source_right = min(input_width, output.frame_width - final_x)
    source_bottom = min(input_height, output.frame_height - final_y)
    visible_width = max(0, source_right - source_left)
    visible_height = max(0, source_bottom - source_top)
    if visible_width == 0 or visible_height == 0:
        source_rectangle = EMPTY_RECTANGLE
        destination_rectangle = EMPTY_RECTANGLE
    else:
        source_rectangle = AlignmentRectangle(
            source_left,
            source_top,
            visible_width,
            visible_height,
        )
        destination_rectangle = AlignmentRectangle(
            max(0, final_x),
            max(0, final_y),
            visible_width,
            visible_height,
        )
    return AlignmentFrame(
        name=name,
        relative_path=relative_path,
        source_order=source_order,
        input_width=input_width,
        input_height=input_height,
        base_x=base_x,
        base_y=base_y,
        offset_dx=dx,
        offset_dy=dy,
        final_x=final_x,
        final_y=final_y,
        left_overflow=left_overflow,
        top_overflow=top_overflow,
        right_overflow=right_overflow,
        bottom_overflow=bottom_overflow,
        clipped=clipped,
        visible_source_rectangle=source_rectangle,
        visible_destination_rectangle=destination_rectangle,
        output_width=output.frame_width,
        output_height=output.frame_height,
    )


def clipping_finding(frame: AlignmentFrame) -> AlignmentClippingFinding:
    return AlignmentClippingFinding(
        frame_name=frame.name,
        source_order=frame.source_order,
        left_overflow=frame.left_overflow,
        top_overflow=frame.top_overflow,
        right_overflow=frame.right_overflow,
        bottom_overflow=frame.bottom_overflow,
        visible_source_rectangle=frame.visible_source_rectangle,
        visible_destination_rectangle=frame.visible_destination_rectangle,
    )


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


def _require_output_config(loaded: LoadedConfig) -> OutputConfig:
    output = loaded.config.output
    if output is None:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_001",
            'align command requires a complete "output" configuration section',
        )
    return output


def _validate_config_handoff(
    stage: ValidatedStageInput | LoadedStageInput,
    loaded: LoadedConfig,
) -> None:
    if stage.identity.effective_config_sha256 != loaded.effective_config_sha256:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_009",
            "configuration does not match the pixelize-stage effective configuration",
            remediation="run pixelize and align with the same validated configuration",
        )
    input_names = tuple(frame.name for frame in stage.frames)
    if input_names != loaded.config.frames.names:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_010",
            "configured frame order does not match pixelize metadata",
        )
    input_filenames = tuple(frame.relative_path.name for frame in stage.frames)
    if input_filenames != loaded.config.frames.filenames:
        raise UnsupportedInputError(
            "PX_STAGE_015",
            "prior-stage frame paths do not match configured frame identities",
        )


def project_align_resources(
    frames: tuple[AlignmentFrame, ...],
    canvas_width: int,
    canvas_height: int,
) -> ResourceProjection:
    """Project the locked align explicit-buffer formula."""

    aggregate_input = sum(frame.input_width * frame.input_height for frame in frames)
    canvas_area = canvas_width * canvas_height
    aggregate_output = len(frames) * canvas_area
    return ResourceProjection(
        "align",
        aggregate_input,
        aggregate_output,
        4 * aggregate_input + 4 * aggregate_output + 5 * canvas_area,
    )


def project_align_stage(
    stage: ValidatedStageInput,
    loaded: LoadedConfig,
) -> AlignmentStagePlan:
    """Project placements, warnings, and resources without decoding pixels."""

    output = _require_output_config(loaded)
    _validate_config_handoff(stage, loaded)
    offsets = {item.frame_name: (item.dx, item.dy) for item in loaded.config.frame_offsets}
    metadata_frames = tuple(
        calculate_alignment_frame(
            name=frame.name,
            relative_path=frame.relative_path,
            source_order=frame.source_order,
            input_width=frame.dimensions.width,
            input_height=frame.dimensions.height,
            output=output,
            dx=offsets.get(frame.name, (0, 0))[0],
            dy=offsets.get(frame.name, (0, 0))[1],
        )
        for frame in stage.frames
    )
    findings = tuple(clipping_finding(frame) for frame in metadata_frames if frame.clipped)
    if findings and output.clip_policy == "error":
        raise AlignmentClippingError(findings)

    warnings: list[ProcessingWarning] = list(stage.warnings)
    warnings.extend(
        ProcessingWarning(
            code="PX_ALIGN_OFFSET_001",
            stage="align",
            message=(
                f'frame "{item.frame_name}" uses explicit alignment offset '
                f"dx={item.dx}, dy={item.dy}; placement is user-managed"
            ),
        )
        for item in loaded.config.frame_offsets
    )
    if output.clip_policy == "warn":
        warnings.extend(
            ProcessingWarning(
                code="PX_ALIGN_CLIP_002",
                stage="align",
                message=(
                    f'frame "{item.frame_name}" clipped left={item.left_overflow}, '
                    f"top={item.top_overflow}, right={item.right_overflow}, "
                    f"bottom={item.bottom_overflow}"
                ),
            )
            for item in findings
        )
    return AlignmentStagePlan(
        metadata_frames,
        findings,
        tuple(warnings),
        project_align_resources(
            metadata_frames,
            output.frame_width,
            output.frame_height,
        ),
    )


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


def publish_align(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> AlignmentStageMetadata:
    validate_stage_output_target(output, "align", force=force)
    validated = validate_stage_input(input_dir, "pixelize")
    plan = project_align_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = align_stage(stage, loaded, plan)
    publish_stage_output(output, "align", run.metadata, run.frame_images, force=force)
    return run.metadata
