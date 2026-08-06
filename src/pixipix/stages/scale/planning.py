"""Allocation-light scale planning and resource projection."""

from __future__ import annotations

from dataclasses import dataclass

from pixipix._scale_geometry import transformed_dimension
from pixipix.config import LoadedConfig, ScaleConfig
from pixipix.errors import ConfigurationError, ProcessingError, UnsupportedInputError
from pixipix.models import Dimensions, ProcessingWarning, ScaleFrame
from pixipix.pipeline.input import ValidatedStageInput
from pixipix.resources import ResourceProjection

MAX_TRANSFORMED_PIXELS = 16_777_216


@dataclass(frozen=True, slots=True)
class ScaleStagePlan:
    frames: tuple[ScaleFrame, ...]
    global_factor: float
    source_reference_measurement: int | None
    exact_target_source_measurement: int | None
    warnings: tuple[ProcessingWarning, ...]
    projection: ResourceProjection


def _require_scale_config(loaded: LoadedConfig) -> ScaleConfig:
    config = loaded.config.scale
    if config is None:
        raise ConfigurationError(
            "PX_SCALE_CONFIG_001",
            'scale command requires a complete "scale" configuration section',
        )
    return config


def _validate_config_handoff(stage: ValidatedStageInput, loaded: LoadedConfig) -> None:
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
    stage: ValidatedStageInput,
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
