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
from pixipix.stages.io import (
    LoadedStageInput,
    OutputFrameImage,
    load_stage_input,
    publish_stage_output,
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


@dataclass(slots=True)
class PixelizeRun:
    metadata: PixelizeStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


def prepare_cell_grid(
    pixels: UInt8Image, cell_size: int, policy: str, frame_name: str
) -> PreparedCellGrid:
    """Prepare a bottom-left grid by changing only the top and right edges."""

    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("pixelize input must be uint8 RGBA")
    if cell_size <= 0:
        raise ValueError("cell size must be positive")
    height, width, _ = pixels.shape
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
        right_padding = 0 if width_remainder == 0 else cell_size - width_remainder
        top_padding = 0 if height_remainder == 0 else cell_size - height_remainder
        prepared_width = width + right_padding
        prepared_height = height + top_padding
        if prepared_width * prepared_height > MAX_PREPARED_PIXELS:
            raise ProcessingError(
                "PX_PIXELIZE_002",
                "pixelize",
                (f"prepared dimensions {prepared_width}x{prepared_height} exceed the safety limit"),
                frame=frame_name,
                remediation="reduce source_cell_size or adjust scale geometry",
            )
        prepared = np.pad(
            np.array(pixels, dtype=np.uint8, copy=True),
            ((top_padding, 0), (0, right_padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return PreparedCellGrid(
            np.asarray(prepared, dtype=np.uint8),
            top_padding,
            right_padding,
            0,
            0,
            None,
        )
    if policy == "crop-with-warning":
        top_crop = height_remainder
        right_crop = width_remainder
        cropped_height = height - top_crop
        cropped_width = width - right_crop
        if cropped_width <= 0 or cropped_height <= 0:
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
                    f"from {width}x{height} to {cropped_width}x{cropped_height}"
                ),
            )
        prepared = np.array(pixels[top_crop:height, 0:cropped_width], dtype=np.uint8, copy=True)
        return PreparedCellGrid(prepared, 0, 0, top_crop, right_crop, warning)
    if policy == "error":
        return PreparedCellGrid(np.array(pixels, dtype=np.uint8, copy=True), 0, 0, 0, 0, None)
    raise ValueError(f"unsupported remainder policy: {policy}")


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


def _validate_config_handoff(stage: LoadedStageInput, loaded: LoadedConfig, cell_size: int) -> None:
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


def pixelize_stage(stage: LoadedStageInput, loaded: LoadedConfig) -> PixelizeRun:
    """Convert validated scale frames into one RGBA pixel per source cell."""

    config, cell_size = _require_pixelize_config(loaded)
    _validate_config_handoff(stage, loaded, cell_size)
    metadata_frames: list[PixelizeFrame] = []
    output_frames: list[OutputFrameImage] = []
    warnings = list(stage.warnings)
    for frame in stage.frames:
        prepared = prepare_cell_grid(frame.pixels, cell_size, config.remainder_policy, frame.name)
        if prepared.warning is not None:
            warnings.append(prepared.warning)
        output = pixelize_prepared_grid(
            prepared.pixels,
            cell_size,
            config.representative,
            config.alpha_policy,
            config.alpha_threshold,
        )
        prepared_height, prepared_width, _ = prepared.pixels.shape
        logical_height, logical_width, _ = output.shape
        metadata_frames.append(
            PixelizeFrame(
                name=frame.name,
                relative_path=frame.relative_path,
                source_order=frame.source_order,
                input_dimensions=frame.dimensions,
                prepared_dimensions=Dimensions(prepared_width, prepared_height),
                top_padding=prepared.top_padding,
                right_padding=prepared.right_padding,
                top_crop=prepared.top_crop,
                right_crop=prepared.right_crop,
                logical_output_dimensions=Dimensions(logical_width, logical_height),
            )
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
        frames=tuple(metadata_frames),
        warnings=tuple(warnings),
    )
    return PixelizeRun(metadata, tuple(output_frames))


def publish_pixelize(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> PixelizeStageMetadata:
    stage = load_stage_input(input_dir, "scale")
    run = pixelize_stage(stage, loaded)
    publish_stage_output(output, "pixelize", run.metadata, run.frame_images, force=force)
    return run.metadata
