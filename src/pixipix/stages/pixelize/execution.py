"""Pixelize-stage numerical execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixipix.config import LoadedConfig
from pixipix.models import Dimensions, PixelizeStageMetadata, ProcessingWarning, UInt8Image
from pixipix.pipeline.input import LoadedStageInput
from pixipix.pipeline.publication import OutputFrameImage
from pixipix.stages.scale import (
    round_channel_half_away_from_zero as round_channel_half_away_from_zero,
)

from .metadata import build_pixelize_metadata
from .planning import (
    CellGridProjection,
    PixelizeStagePlan,
    _require_pixelize_config,
    project_cell_grid,
)


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
    metadata = build_pixelize_metadata(stage, loaded, plan, config, cell_size)
    return PixelizeRun(metadata, tuple(output_frames))
