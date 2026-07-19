"""Deterministic PNG codec boundary and foreground-mask generation."""

from __future__ import annotations

import warnings
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixipix.config import BackgroundConfig, SourceConfig
from pixipix.errors import ProcessingError, UnsupportedInputError
from pixipix.models import (
    BackgroundSummary,
    BoolMask,
    Rect,
    SourceImage,
    SourceImageMetadata,
    UInt8Image,
)

PNG_SAVE_OPTIONS: dict[str, object] = {
    "compress_level": 9,
    "optimize": False,
}


def load_source(path: Path, config: SourceConfig) -> SourceImage:
    """Load one PNG and take ownership of a normalized RGBA uint8 array."""

    if path.suffix.lower() != ".png":
        raise UnsupportedInputError(
            "PX_INPUT_001",
            "only PNG input is supported",
            path=path.name,
            remediation="convert the source to PNG",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise UnsupportedInputError(
                        "PX_INPUT_001",
                        "input extension is PNG but content is not PNG",
                        path=path.name,
                    )
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise UnsupportedInputError(
                        "PX_INPUT_003", "source image is empty", path=path.name
                    )
                if (
                    width > config.max_width
                    or height > config.max_height
                    or width * height > config.max_pixels
                ):
                    raise UnsupportedInputError(
                        "PX_INPUT_004",
                        f"source dimensions {width}x{height} exceed configured safety limits",
                        path=path.name,
                        remediation="reduce the image size within the fixed safety ceiling",
                    )
                input_mode = image.mode
                has_alpha = "A" in image.getbands() or "transparency" in image.info
                image.load()
                rgba = image.convert("RGBA")
                pixels = np.array(rgba, dtype=np.uint8, copy=True)
    except UnsupportedInputError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise UnsupportedInputError(
            "PX_INPUT_004",
            "source dimensions exceed decoder safety limits",
            path=path.name,
            remediation="reduce the image size within the fixed safety ceiling",
        ) from error
    except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as error:
        raise UnsupportedInputError(
            "PX_INPUT_002",
            "unable to decode PNG input",
            path=path.name,
            remediation="verify that the file is a readable, well-formed PNG",
        ) from error
    metadata = SourceImageMetadata(
        path=PurePosixPath(path.name),
        width=width,
        height=height,
        input_mode=input_mode,
        has_alpha=has_alpha,
    )
    return SourceImage(metadata=metadata, pixels=pixels)


def _rgba_hex(color: UInt8Image) -> str:
    values = color.tolist()
    return "#" + "".join(f"{int(value):02x}" for value in values)


def _rgba_color(value: str) -> UInt8Image:
    return np.frombuffer(bytes.fromhex(value[1:]), dtype=np.uint8).copy()


def _matching_color_mask(
    pixels: UInt8Image, color: UInt8Image, tolerance: float, compare_alpha: bool
) -> BoolMask:
    channel_count = 4 if compare_alpha else 3
    source = pixels[:, :, :channel_count].astype(np.int16)
    target = color[:channel_count].astype(np.int16)
    distance = np.max(np.abs(source - target), axis=2)
    return np.asarray(distance <= tolerance * 255.0, dtype=np.bool_)


def _foreground_bounds(mask: BoolMask) -> Rect | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return Rect(
        left=int(columns.min()),
        top=int(rows.min()),
        right=int(columns.max()) + 1,
        bottom=int(rows.max()) + 1,
    )


def generate_foreground_mask(
    source: SourceImage, config: BackgroundConfig
) -> tuple[BoolMask, BackgroundSummary]:
    """Return a new boolean mask without mutating source pixels.

    Color modes use normalized maximum per-channel distance: the largest
    absolute channel difference divided by 255 must be within tolerance.
    Six-digit explicit colors compare RGB; eight-digit colors compare RGBA.
    Corner-color mode compares RGBA and selects the top-left sample.
    """

    pixels = source.pixels
    selected_color: str | None = None
    if config.mode == "alpha":
        mask = np.asarray(pixels[:, :, 3] >= config.alpha_threshold, dtype=np.bool_)
    elif config.mode == "explicit-color":
        if config.color is None:
            raise ProcessingError(
                "PX_BACKGROUND_001", "mask", "explicit background color is missing"
            )
        color = _rgba_color(config.color)
        selected_color = _rgba_hex(color)
        mask = np.logical_and(
            pixels[:, :, 3] >= config.alpha_threshold,
            np.logical_not(
                _matching_color_mask(pixels, color, config.tolerance, config.compare_alpha)
            ),
        )
    else:
        corners = np.stack(
            (
                pixels[0, 0],
                pixels[0, -1],
                pixels[-1, 0],
                pixels[-1, -1],
            )
        ).astype(np.uint8)
        selected = corners[0]
        differences = np.max(np.abs(corners.astype(np.int16) - selected.astype(np.int16)), axis=1)
        if np.any(differences > config.tolerance * 255.0):
            raise ProcessingError(
                "PX_BACKGROUND_002",
                "mask",
                "sampled corner colors disagree beyond configured tolerance",
                path=source.metadata.path.as_posix(),
                remediation="use explicit-color mode or provide a source with consistent corners",
            )
        selected_color = _rgba_hex(selected)
        mask = np.logical_and(
            pixels[:, :, 3] >= config.alpha_threshold,
            np.logical_not(_matching_color_mask(pixels, selected, config.tolerance, True)),
        )
    mask = np.asarray(mask, dtype=np.bool_)
    touches = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
    summary = BackgroundSummary(
        mode=config.mode,
        selected_color=selected_color,
        tolerance=config.tolerance,
        pixels_removed=int(mask.size - np.count_nonzero(mask)),
        foreground_touches_boundary=touches,
        foreground_bounds=_foreground_bounds(mask),
    )
    return mask, summary


def normalized_transparent_rgb(pixels: UInt8Image) -> UInt8Image:
    output = np.array(pixels, dtype=np.uint8, copy=True)
    transparent = output[:, :, 3] == 0
    output[transparent, :3] = 0
    return output


def write_png(path: Path, pixels: UInt8Image) -> None:
    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ProcessingError("PX_PNG_001", "encode", "PNG buffer must be uint8 RGBA")
    normalized = np.ascontiguousarray(normalized_transparent_rgb(pixels))
    with Image.fromarray(normalized, mode="RGBA") as image:
        image.save(path, format="PNG", **PNG_SAVE_OPTIONS)
