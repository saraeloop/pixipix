"""Deterministic bottom-left-anchored logical pixel conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pixipix import __version__
from pixipix.config import LoadedConfig, PixelizeConfig
from pixipix.errors import ConfigurationError, ProcessingError, UnsupportedInputError
from pixipix.models import (
    Dimensions,
    PixelizeFrame,
    PixelizeStageMetadata,
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
from pixipix.stages.scale import round_channel_half_away_from_zero

MAX_PREPARED_PIXELS = 16_777_216


@dataclass(slots=True)
class PreparedCellGrid:
    pixels: UInt8Image
    top_padding: int
    right_padding: int
    top_crop: int
    right_crop: int
    warning: ProcessingWarning | None


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


@dataclass(slots=True)
class PixelizeRun:
    metadata: PixelizeStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


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


def prepare_cell_grid(
    pixels: UInt8Image,
    cell_size: int,
    policy: str,
    frame_name: str,
    *,
    projection: CellGridProjection | None = None,
) -> PreparedCellGrid:
    """Prepare a bottom-left grid by changing only the top and right edges."""

    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("pixelize input must be uint8 RGBA")
    height, width, _ = pixels.shape
    projected = projection or project_cell_grid(
        Dimensions(width, height), cell_size, policy, frame_name
    )
    if (width, height) != (
        projected.input_dimensions.width,
        projected.input_dimensions.height,
    ):
        raise ValueError("pixelize input dimensions do not match projected geometry")
    if policy == "pad-transparent":
        prepared = np.pad(
            np.array(pixels, dtype=np.uint8, copy=True),
            ((projected.top_padding, 0), (0, projected.right_padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return PreparedCellGrid(
            np.asarray(prepared, dtype=np.uint8),
            projected.top_padding,
            projected.right_padding,
            0,
            0,
            projected.warning,
        )
    if policy == "crop-with-warning":
        prepared = np.array(
            pixels[
                projected.top_crop : height,
                0 : projected.prepared_dimensions.width,
            ],
            dtype=np.uint8,
            copy=True,
        )
        return PreparedCellGrid(
            prepared,
            0,
            0,
            projected.top_crop,
            projected.right_crop,
            projected.warning,
        )
    if policy == "error":
        return PreparedCellGrid(
            np.array(pixels, dtype=np.uint8, copy=True),
            0,
            0,
            0,
            0,
            projected.warning,
        )
    raise AssertionError("project_cell_grid accepted an unsupported policy")


def _majority(cell: UInt8Image) -> tuple[int, int, int, int]:
    counts: dict[tuple[int, int, int, int], int] = {}
    first: dict[tuple[int, int, int, int], int] = {}
    for index, raw in enumerate(cell.reshape(-1, 4)):
        value = tuple(int(channel) for channel in raw)
        rgba = (value[0], value[1], value[2], value[3])
        counts[rgba] = counts.get(rgba, 0) + 1
        first.setdefault(rgba, index)
    return max(counts, key=lambda value: (counts[value], -first[value]))


def _center(cell: UInt8Image) -> tuple[int, int, int, int]:
    cell_size = cell.shape[0]
    coordinate = (cell_size - 1) // 2
    value = cell[coordinate, coordinate]
    return tuple(int(channel) for channel in value)  # type: ignore[return-value]


def _alpha_weighted_majority(cell: UInt8Image) -> tuple[int, int, int, int]:
    weights: dict[tuple[int, int, int], int] = {}
    alpha_squares: dict[tuple[int, int, int], int] = {}
    first: dict[tuple[int, int, int], int] = {}
    for index, raw in enumerate(cell.reshape(-1, 4)):
        red, green, blue, alpha = (int(channel) for channel in raw)
        if alpha == 0:
            continue
        rgb = (red, green, blue)
        weights[rgb] = weights.get(rgb, 0) + alpha
        alpha_squares[rgb] = alpha_squares.get(rgb, 0) + alpha * alpha
        first.setdefault(rgb, index)
    if not weights:
        return (0, 0, 0, 0)
    selected = max(weights, key=lambda value: (weights[value], -first[value]))
    representative_alpha = round_channel_half_away_from_zero(
        alpha_squares[selected] / weights[selected]
    )
    return (*selected, representative_alpha)


def representative_pixel(cell: UInt8Image, strategy: str) -> tuple[int, int, int, int]:
    if strategy == "majority":
        return _majority(cell)
    if strategy == "center":
        return _center(cell)
    if strategy == "alpha-weighted-majority":
        return _alpha_weighted_majority(cell)
    raise ValueError(f"unsupported representative strategy: {strategy}")


def apply_alpha_policy(
    rgba: tuple[int, int, int, int], policy: str, threshold: int
) -> tuple[int, int, int, int]:
    red, green, blue, alpha = rgba
    if policy == "binary":
        alpha = 255 if alpha >= threshold else 0
    elif policy != "preserve":
        raise ValueError(f"unsupported alpha policy: {policy}")
    if alpha == 0:
        return (0, 0, 0, 0)
    return (red, green, blue, alpha)


def pixelize_prepared_grid(
    pixels: UInt8Image,
    cell_size: int,
    strategy: str,
    alpha_policy: str,
    alpha_threshold: int,
) -> UInt8Image:
    height, width, _ = pixels.shape
    if width % cell_size or height % cell_size:
        raise ValueError("prepared grid must be exactly divisible by cell size")
    output = np.zeros((height // cell_size, width // cell_size, 4), dtype=np.uint8)
    for logical_row in range(output.shape[0]):
        top = logical_row * cell_size
        for logical_column in range(output.shape[1]):
            left = logical_column * cell_size
            cell = pixels[top : top + cell_size, left : left + cell_size]
            rgba = representative_pixel(cell, strategy)
            output[logical_row, logical_column] = apply_alpha_policy(
                rgba, alpha_policy, alpha_threshold
            )
    return output


def _require_pixelize_config(loaded: LoadedConfig) -> tuple[PixelizeConfig, int]:
    config = loaded.config.pixelize
    if config.source_cell_size is None:
        raise ConfigurationError(
            "PX_PIXELIZE_CONFIG_001",
            'pixelize command requires "pixelize.source_cell_size"',
        )
    return config, config.source_cell_size


def _validate_config_handoff(
    stage: ValidatedStageInput | LoadedStageInput,
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


def pixelize_stage(
    stage: LoadedStageInput,
    loaded: LoadedConfig,
    plan: PixelizeStagePlan,
) -> PixelizeRun:
    """Execute one admitted pixelize plan without recomputing geometry."""

    config, cell_size = _require_pixelize_config(loaded)
    output_frames: list[OutputFrameImage] = []
    for frame, projected in zip(stage.frames, plan.cell_grids, strict=True):
        prepared = prepare_cell_grid(
            frame.pixels,
            cell_size,
            config.remainder_policy,
            frame.name,
            projection=projected,
        )
        output = pixelize_prepared_grid(
            prepared.pixels,
            cell_size,
            config.representative,
            config.alpha_policy,
            config.alpha_threshold,
        )
        output_frames.append(OutputFrameImage(frame.relative_path, output))
    metadata = PixelizeStageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="pixelize",
        status="successful",
        prior_stage=stage.identity,
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        source_cell_size=cell_size,
        cell_grid_origin="bottom-left",
        representative=config.representative,
        alpha_policy=config.alpha_policy,
        alpha_threshold=config.alpha_threshold,
        remainder_policy=config.remainder_policy,
        frames=plan.frames,
        warnings=plan.warnings,
    )
    return PixelizeRun(metadata, tuple(output_frames))


def publish_pixelize(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> PixelizeStageMetadata:
    validate_stage_output_target(output, "pixelize", force=force)
    validated = validate_stage_input(input_dir, "scale")
    plan = project_pixelize_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = pixelize_stage(stage, loaded, plan)
    publish_stage_output(output, "pixelize", run.metadata, run.frame_images, force=force)
    return run.metadata
