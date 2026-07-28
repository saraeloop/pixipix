"""Execution of admitted scale plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from pixipix.config import LoadedConfig
from pixipix.models import ScaleStageMetadata, UInt8Image
from pixipix.pipeline.input import LoadedStageInput
from pixipix.pipeline.publication import OutputFrameImage

from .geometry import round_channel_half_away_from_zero
from .metadata import build_scale_metadata
from .planning import ScaleStagePlan, _require_scale_config


@dataclass(slots=True)
class ScaleRun:
    metadata: ScaleStageMetadata
    frame_images: tuple[OutputFrameImage, ...]


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
    metadata = build_scale_metadata(stage, loaded, plan, config)
    return ScaleRun(metadata, output_frames)
