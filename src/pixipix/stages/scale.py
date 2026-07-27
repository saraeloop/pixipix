"""Deterministic global source-space scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pixipix import __version__
from pixipix.config import LoadedConfig, ScaleConfig
from pixipix.errors import ConfigurationError, ProcessingError, UnsupportedInputError
from pixipix.models import (
    Dimensions,
    ProcessingWarning,
    ScaleFrame,
    ScaleOverrideMetadata,
    ScaleStageMetadata,
    UInt8Image,
)
from pixipix.pipeline.input import (
    LoadedStageInput,
    ValidatedStageInput,
    decode_stage_input,
    validate_stage_input,
)
from pixipix.pipeline.publication import (
    OutputFrameImage,
    publish_stage_output,
    validate_stage_output_target,
)
from pixipix.resources import ResourceProjection, enforce_resource_policy

MAX_TRANSFORMED_PIXELS = 16_777_216


@dataclass(slots=True)
class ScaleRun:
    metadata: ScaleStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


@dataclass(frozen=True, slots=True)
class ScaleStagePlan:
    frames: tuple[ScaleFrame, ...]
    global_factor: float
    source_reference_measurement: int | None
    exact_target_source_measurement: int | None
    warnings: tuple[ProcessingWarning, ...]
    projection: ResourceProjection


def round_half_away_from_zero(value: float) -> int:
    """Round one finite geometric value with halves away from zero."""

    if not math.isfinite(value):
        raise ValueError("geometric value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    return magnitude if value >= 0 else -magnitude


def round_channel_half_away_from_zero(value: float) -> int:
    """Quantize one finite channel value separately from geometry rules."""

    if not math.isfinite(value):
        raise ValueError("channel value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    rounded = magnitude if value >= 0 else -magnitude
    return min(255, max(0, rounded))


def transformed_dimension(source_dimension: int, factor: float) -> int:
    if source_dimension <= 0:
        raise ValueError("source dimension must be positive")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("scale factor must be finite and positive")
    return max(1, round_half_away_from_zero(source_dimension * factor))


def _resize_float_channel(channel: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    contiguous = np.ascontiguousarray(channel, dtype=np.float32)
    with Image.fromarray(contiguous, mode="F") as image:
        resized = image.resize(size, resample=Image.Resampling.BOX)
        return np.array(resized, dtype=np.float64, copy=True)


def premultiplied_box_resize(pixels: UInt8Image, size: tuple[int, int]) -> UInt8Image:
    """Resize RGBA using float32 premultiplied channels and BOX filtering.

    Premultiplication is performed without integer quantization. Pillow's ``F``
    mode supplies the locked float32 BOX pass. Un-premultiplication occurs in
    float64 and is quantized once with deterministic channel rounding.
    """

    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("scale input must be uint8 RGBA")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("scale output dimensions must be positive")
    source = np.array(pixels, dtype=np.uint8, copy=True)
    if np.all(source[:, :, 3] == 255):
        with Image.fromarray(source, mode="RGBA") as opaque_image:
            opaque_resized = opaque_image.resize(size, resample=Image.Resampling.BOX)
            return np.array(opaque_resized, dtype=np.uint8, copy=True)
    alpha = source[:, :, 3].astype(np.float32)
    premultiplied = source[:, :, :3].astype(np.float32) * (alpha[:, :, None] / 255.0)
    resized_alpha = _resize_float_channel(alpha, size)
    resized_premultiplied = np.stack(
        tuple(_resize_float_channel(premultiplied[:, :, channel], size) for channel in range(3)),
        axis=2,
    )
    output = np.zeros((height, width, 4), dtype=np.uint8)
    for row in range(height):
        for column in range(width):
            alpha_value = round_channel_half_away_from_zero(float(resized_alpha[row, column]))
            output[row, column, 3] = alpha_value
            if alpha_value == 0 or resized_alpha[row, column] <= 0:
                continue
            for channel in range(3):
                straight = (
                    float(resized_premultiplied[row, column, channel])
                    * 255.0
                    / float(resized_alpha[row, column])
                )
                output[row, column, channel] = round_channel_half_away_from_zero(straight)
    return output


def _require_scale_config(loaded: LoadedConfig) -> ScaleConfig:
    config = loaded.config.scale
    if config is None:
        raise ConfigurationError(
            "PX_SCALE_CONFIG_001",
            'scale command requires a complete "scale" configuration section',
        )
    return config


def _validate_config_handoff(
    stage: ValidatedStageInput | LoadedStageInput, loaded: LoadedConfig
) -> None:
    if stage.identity.effective_config_sha256 != loaded.effective_config_sha256:
        raise ConfigurationError(
            "PX_SCALE_CONFIG_002",
            "configuration does not match the extraction-stage effective configuration",
            remediation="run extract and scale with the same validated configuration",
        )
    input_names = tuple(frame.name for frame in stage.frames)
    if input_names != loaded.config.frames.names:
        raise ConfigurationError(
            "PX_SCALE_CONFIG_003",
            "configured frame order does not match extraction metadata",
        )
    input_filenames = tuple(frame.relative_path.name for frame in stage.frames)
    if input_filenames != loaded.config.frames.filenames:
        raise UnsupportedInputError(
            "PX_STAGE_015",
            "prior-stage frame paths do not match configured frame identities",
        )


def _global_factor(
    config: ScaleConfig,
    stage: ValidatedStageInput | LoadedStageInput,
    source_cell_size: int | None,
) -> tuple[float, int | None, int | None]:
    if config.mode == "explicit-factor":
        if config.factor is None:
            raise ConfigurationError("PX_SCALE_CONFIG_001", "explicit scale factor is missing")
        return config.factor, None, None
    if source_cell_size is None or config.reference_frame is None or config.target_size is None:
        raise ConfigurationError(
            "PX_SCALE_CONFIG_001",
            "reference scaling requires reference frame, target size, and source cell size",
        )
    reference = next(
        (frame for frame in stage.frames if frame.name == config.reference_frame), None
    )
    if reference is None:
        available = ", ".join(frame.name for frame in stage.frames)
        raise ConfigurationError(
            "PX_SCALE_CONFIG_004",
            f'reference frame "{config.reference_frame}" is absent from extraction metadata',
            remediation=f"choose one of: {available}",
        )
    source_measurement = (
        reference.dimensions.width
        if config.mode == "reference-frame-width"
        else reference.dimensions.height
    )
    exact_target = config.target_size * source_cell_size
    return exact_target / source_measurement, source_measurement, exact_target


def project_scale_resources(frames: tuple[ScaleFrame, ...]) -> ResourceProjection:
    """Project the locked scale explicit-buffer formula."""

    aggregate_input = sum(
        frame.input_dimensions.width * frame.input_dimensions.height for frame in frames
    )
    aggregate_output = sum(
        frame.output_dimensions.width * frame.output_dimensions.height for frame in frames
    )
    transient = max(
        (
            max(
                36 * input_area,
                24 * input_area + 28 * output_area,
                20 * input_area + 56 * output_area,
                5 * output_area,
            )
            for frame in frames
            for input_area, output_area in (
                (
                    frame.input_dimensions.width * frame.input_dimensions.height,
                    frame.output_dimensions.width * frame.output_dimensions.height,
                ),
            )
        ),
        default=0,
    )
    return ResourceProjection(
        "scale",
        aggregate_input,
        aggregate_output,
        4 * aggregate_input + 4 * aggregate_output + transient,
    )


def project_scale_stage(
    stage: ValidatedStageInput,
    loaded: LoadedConfig,
) -> ScaleStagePlan:
    """Project all scale geometry and policy inputs without decoding pixels."""

    config = _require_scale_config(loaded)
    _validate_config_handoff(stage, loaded)
    try:
        global_factor, source_measurement, exact_target = _global_factor(
            config, stage, loaded.config.pixelize.source_cell_size
        )
    except (OverflowError, ValueError) as error:
        raise ProcessingError(
            "PX_SCALE_002",
            "scale",
            "configured scale geometry exceeds the numeric safety limit",
            remediation="reduce the configured scale factor, target size, or source cell size",
        ) from error
    overrides = {item.frame_name: item.scale_multiplier for item in loaded.config.frame_overrides}
    warnings = stage.warnings + tuple(
        ProcessingWarning(
            code="PX_SCALE_OVERRIDE_001",
            stage="scale",
            message=(
                f'frame "{item.frame_name}" uses explicit scale multiplier '
                f"{item.scale_multiplier}; cross-frame consistency is user-managed"
            ),
        )
        for item in loaded.config.frame_overrides
    )
    metadata_frames: list[ScaleFrame] = []
    for frame in stage.frames:
        multiplier = overrides.get(frame.name, 1.0)
        effective = global_factor * multiplier
        try:
            width = transformed_dimension(frame.dimensions.width, effective)
            height = transformed_dimension(frame.dimensions.height, effective)
        except (OverflowError, ValueError) as error:
            raise ProcessingError(
                "PX_SCALE_002",
                "scale",
                "transformed dimensions exceed the numeric safety limit",
                frame=frame.name,
                remediation="reduce the configured scale factor or frame multiplier",
            ) from error
        if frame.name == config.reference_frame:
            if exact_target is None:
                raise ProcessingError("PX_SCALE_001", "scale", "reference target is missing")
            if config.mode == "reference-frame-width":
                width = exact_target
            else:
                height = exact_target
        if width * height > MAX_TRANSFORMED_PIXELS:
            raise ProcessingError(
                "PX_SCALE_002",
                "scale",
                f"transformed dimensions {width}x{height} exceed the safety limit",
                frame=frame.name,
                remediation="reduce the configured scale factor or target size",
            )
        metadata_frames.append(
            ScaleFrame(
                name=frame.name,
                relative_path=frame.relative_path,
                source_order=frame.source_order,
                input_dimensions=frame.dimensions,
                output_dimensions=Dimensions(width, height),
                scale_multiplier=multiplier,
                effective_factor=effective,
            )
        )
    frames = tuple(metadata_frames)
    return ScaleStagePlan(
        frames,
        global_factor,
        source_measurement,
        exact_target,
        warnings,
        project_scale_resources(frames),
    )


def scale_stage(
    stage: LoadedStageInput,
    loaded: LoadedConfig,
    plan: ScaleStagePlan,
) -> ScaleRun:
    """Execute one admitted scale plan without recomputing geometry."""

    config = _require_scale_config(loaded)
    output_frames = tuple(
        OutputFrameImage(
            source.relative_path,
            premultiplied_box_resize(
                source.pixels,
                (
                    metadata.output_dimensions.width,
                    metadata.output_dimensions.height,
                ),
            ),
        )
        for source, metadata in zip(stage.frames, plan.frames, strict=True)
    )
    metadata = ScaleStageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="scale",
        status="successful",
        prior_stage=stage.identity,
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        scale_mode=config.mode,
        global_factor=plan.global_factor,
        reference_frame=config.reference_frame,
        source_reference_measurement=plan.source_reference_measurement,
        exact_target_source_measurement=plan.exact_target_source_measurement,
        logical_target_size=config.target_size,
        source_cell_size=loaded.config.pixelize.source_cell_size,
        configured_frame_overrides=tuple(
            ScaleOverrideMetadata(item.frame_name, item.scale_multiplier)
            for item in loaded.config.frame_overrides
        ),
        frames=plan.frames,
        warnings=plan.warnings,
    )
    return ScaleRun(metadata, output_frames)


def publish_scale(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> ScaleStageMetadata:
    validate_stage_output_target(output, "scale", force=force)
    validated = validate_stage_input(input_dir, "extract")
    plan = project_scale_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = scale_stage(stage, loaded, plan)
    publish_stage_output(output, "scale", run.metadata, run.frame_images, force=force)
    return run.metadata
