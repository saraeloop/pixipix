"""Allocation-light alignment planning and resource projection."""

from __future__ import annotations

from dataclasses import dataclass

from pixipix.config import LoadedConfig, OutputConfig
from pixipix.errors import AlignmentClippingError, ConfigurationError, UnsupportedInputError
from pixipix.models import AlignmentClippingFinding, AlignmentFrame, ProcessingWarning
from pixipix.pipeline.input import ValidatedStageInput
from pixipix.resources import ResourceProjection
from pixipix.stages.align.geometry import calculate_alignment_frame


@dataclass(frozen=True, slots=True)
class AlignmentStagePlan:
    frames: tuple[AlignmentFrame, ...]
    clipping_findings: tuple[AlignmentClippingFinding, ...]
    warnings: tuple[ProcessingWarning, ...]
    projection: ResourceProjection


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


def _require_output_config(loaded: LoadedConfig) -> OutputConfig:
    output = loaded.config.output
    if output is None:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_001",
            'align command requires a complete "output" configuration section',
        )
    return output


def _validate_config_handoff(
    stage: ValidatedStageInput,
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
