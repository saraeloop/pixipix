"""Pixelize-stage projection and resource planning."""

from __future__ import annotations

from dataclasses import dataclass

from pixipix.config import LoadedConfig, PixelizeConfig
from pixipix.errors import ConfigurationError, ProcessingError, UnsupportedInputError
from pixipix.models import Dimensions, PixelizeFrame, ProcessingWarning
from pixipix.pipeline.input import ValidatedStageInput
from pixipix.resources import ResourceProjection

MAX_PREPARED_PIXELS = 16_777_216


@dataclass(frozen=True, slots=True)
class CellGridProjection:
    input_dimensions: Dimensions
    prepared_dimensions: Dimensions
    logical_output_dimensions: Dimensions
    top_padding: int
    right_padding: int
    top_crop: int
    right_crop: int
    warning: ProcessingWarning | None


@dataclass(frozen=True, slots=True)
class PixelizeStagePlan:
    frames: tuple[PixelizeFrame, ...]
    cell_grids: tuple[CellGridProjection, ...]
    warnings: tuple[ProcessingWarning, ...]
    projection: ResourceProjection


def project_cell_grid(
    dimensions: Dimensions,
    cell_size: int,
    policy: str,
    frame_name: str,
) -> CellGridProjection:
    """Project bottom-left preparation and logical geometry without pixels."""

    width = dimensions.width
    height = dimensions.height
    if cell_size <= 0:
        raise ValueError("cell size must be positive")
    if width <= 0 or height <= 0:
        raise ProcessingError(
            "PX_PIXELIZE_001", "pixelize", "input frame is empty", frame=frame_name
        )
    width_remainder = width % cell_size
    height_remainder = height % cell_size
    if policy == "error" and (width_remainder or height_remainder):
        raise ProcessingError(
            "PX_PIXELIZE_REMAINDER_001",
            "pixelize",
            (
                f"input dimensions {width}x{height} are not divisible by cell size "
                f"{cell_size}; width remainder {width_remainder}, height remainder "
                f"{height_remainder}"
            ),
            frame=frame_name,
            remediation="use pad-transparent or crop-with-warning, or adjust scale geometry",
        )
    if policy == "pad-transparent":
        right_padding = (-width) % cell_size
        top_padding = (-height) % cell_size
        prepared_width = width + right_padding
        prepared_height = height + top_padding
        if prepared_width * prepared_height > MAX_PREPARED_PIXELS:
            raise ProcessingError(
                "PX_PIXELIZE_002",
                "pixelize",
                f"prepared dimensions {prepared_width}x{prepared_height} exceed the safety limit",
                frame=frame_name,
                remediation="reduce source_cell_size or adjust scale geometry",
            )
        return CellGridProjection(
            dimensions,
            Dimensions(prepared_width, prepared_height),
            Dimensions(prepared_width // cell_size, prepared_height // cell_size),
            top_padding,
            right_padding,
            0,
            0,
            None,
        )
    if policy == "crop-with-warning":
        top_crop = height_remainder
        right_crop = width_remainder
        prepared_height = height - top_crop
        prepared_width = width - right_crop
        if prepared_width <= 0 or prepared_height <= 0:
            raise ProcessingError(
                "PX_PIXELIZE_REMAINDER_002",
                "pixelize",
                f"top/right crop would reduce non-empty frame {width}x{height} to zero dimensions",
                frame=frame_name,
                remediation="use pad-transparent or reduce source_cell_size",
            )
        warning = None
        if top_crop or right_crop:
            warning = ProcessingWarning(
                code="PX_PIXELIZE_CROP_001",
                stage="pixelize",
                message=(
                    f'frame "{frame_name}" cropped top={top_crop}, right={right_crop} '
                    f"from {width}x{height} to {prepared_width}x{prepared_height}"
                ),
            )
        return CellGridProjection(
            dimensions,
            Dimensions(prepared_width, prepared_height),
            Dimensions(prepared_width // cell_size, prepared_height // cell_size),
            0,
            0,
            top_crop,
            right_crop,
            warning,
        )
    if policy == "error":
        return CellGridProjection(
            dimensions,
            dimensions,
            Dimensions(width // cell_size, height // cell_size),
            0,
            0,
            0,
            0,
            None,
        )
    raise ValueError(f"unsupported remainder policy: {policy}")


def _require_pixelize_config(loaded: LoadedConfig) -> tuple[PixelizeConfig, int]:
    config = loaded.config.pixelize
    if config.source_cell_size is None:
        raise ConfigurationError(
            "PX_PIXELIZE_CONFIG_001",
            'pixelize command requires "pixelize.source_cell_size"',
        )
    return config, config.source_cell_size


def _validate_config_handoff(
    stage: ValidatedStageInput,
    loaded: LoadedConfig,
    cell_size: int,
) -> None:
    if stage.identity.effective_config_sha256 != loaded.effective_config_sha256:
        raise ConfigurationError(
            "PX_PIXELIZE_CONFIG_002",
            "configuration does not match the scale-stage effective configuration",
            remediation="run scale and pixelize with the same validated configuration",
        )
    input_names = tuple(frame.name for frame in stage.frames)
    if input_names != loaded.config.frames.names:
        raise ConfigurationError(
            "PX_PIXELIZE_CONFIG_003", "configured frame order does not match scale metadata"
        )
    input_filenames = tuple(frame.relative_path.name for frame in stage.frames)
    if input_filenames != loaded.config.frames.filenames:
        raise UnsupportedInputError(
            "PX_STAGE_015",
            "prior-stage frame paths do not match configured frame identities",
        )
    declared_cell_size = stage.metadata.get("sourceCellSize")
    if declared_cell_size != cell_size:
        raise ConfigurationError(
            "PX_PIXELIZE_CONFIG_004",
            f"configured source cell size {cell_size} does not match scale metadata "
            f"{declared_cell_size!r}",
        )


def project_pixelize_resources(
    frames: tuple[PixelizeFrame, ...],
    cell_size: int,
) -> ResourceProjection:
    """Project the locked pixelize explicit-buffer formula."""

    aggregate_input = sum(
        frame.input_dimensions.width * frame.input_dimensions.height for frame in frames
    )
    aggregate_output = sum(
        frame.logical_output_dimensions.width * frame.logical_output_dimensions.height
        for frame in frames
    )
    transient = max(
        (
            max(
                4 * input_area + 4 * prepared_area,
                4 * prepared_area + 4 * cell_size * cell_size,
                5 * logical_area,
            )
            for frame in frames
            for input_area, prepared_area, logical_area in (
                (
                    frame.input_dimensions.width * frame.input_dimensions.height,
                    frame.prepared_dimensions.width * frame.prepared_dimensions.height,
                    frame.logical_output_dimensions.width * frame.logical_output_dimensions.height,
                ),
            )
        ),
        default=0,
    )
    return ResourceProjection(
        "pixelize",
        aggregate_input,
        aggregate_output,
        4 * aggregate_input + 4 * aggregate_output + transient,
    )


def project_pixelize_stage(
    stage: ValidatedStageInput,
    loaded: LoadedConfig,
) -> PixelizeStagePlan:
    """Project pixelize geometry and warnings without decoding frame PNGs."""

    config, cell_size = _require_pixelize_config(loaded)
    _validate_config_handoff(stage, loaded, cell_size)
    metadata_frames: list[PixelizeFrame] = []
    warnings = list(stage.warnings)
    cell_grids: list[CellGridProjection] = []
    for frame in stage.frames:
        projected = project_cell_grid(
            frame.dimensions,
            cell_size,
            config.remainder_policy,
            frame.name,
        )
        if projected.warning is not None:
            warnings.append(projected.warning)
        metadata_frames.append(
            PixelizeFrame(
                name=frame.name,
                relative_path=frame.relative_path,
                source_order=frame.source_order,
                input_dimensions=frame.dimensions,
                prepared_dimensions=projected.prepared_dimensions,
                top_padding=projected.top_padding,
                right_padding=projected.right_padding,
                top_crop=projected.top_crop,
                right_crop=projected.right_crop,
                logical_output_dimensions=projected.logical_output_dimensions,
            )
        )
        cell_grids.append(projected)
    frames = tuple(metadata_frames)
    return PixelizeStagePlan(
        frames,
        tuple(cell_grids),
        tuple(warnings),
        project_pixelize_resources(frames, cell_size),
    )
