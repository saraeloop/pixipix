from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pixipix.models import UInt8Image


def write_rgba(path: Path, pixels: UInt8Image) -> None:
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")


def write_rgb(path: Path, pixels: UInt8Image) -> None:
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def transparent_sheet() -> UInt8Image:
    pixels = np.zeros((8, 14, 4), dtype=np.uint8)
    pixels[1:4, 1:4] = (20, 40, 60, 255)
    pixels[1:3, 8:12] = (120, 80, 40, 200)
    pixels[6, 13] = (255, 0, 0, 255)  # deterministic rejected noise
    return pixels


def extraction_config(
    *,
    names: tuple[str, ...] = ("idle", "signal"),
    background: str = 'mode = "alpha"\nalpha_threshold = 8',
    minimum_area: int = 2,
    maximum_area: int | None = None,
    padding: int = 1,
    expected: int | None = 2,
) -> str:
    expected_line = f"expected_components = {expected}\n" if expected is not None else ""
    maximum_line = f"maximum_area = {maximum_area}\n" if maximum_area is not None else ""
    quoted_names = ", ".join(f'"{name}"' for name in names)
    return (
        "[project]\n"
        'name = "test"\n'
        "strict = true\n\n"
        "[source]\n"
        f"{expected_line}"
        "max_width = 64\n"
        "max_height = 64\n"
        "max_pixels = 4096\n"
        "max_components = 16\n\n"
        "[background]\n"
        f"{background}\n\n"
        "[extract]\n"
        "connectivity = 8\n"
        f"minimum_area = {minimum_area}\n"
        f"{maximum_line}"
        f"padding = {padding}\n"
        "row_tolerance = 2\n\n"
        "[frames]\n"
        f"names = [{quoted_names}]\n"
    )


def write_config(path: Path, content: str | None = None) -> None:
    path.write_text(content or extraction_config(), encoding="utf-8")


def pipeline_config(
    *,
    names: tuple[str, ...] = ("idle", "signal"),
    scale: str = 'mode = "explicit-factor"\nfactor = 1.0',
    pixelize: str = (
        "source_cell_size = 2\n"
        'representative = "alpha-weighted-majority"\n'
        'alpha_policy = "binary"\n'
        "alpha_threshold = 128\n"
        'remainder_policy = "pad-transparent"'
    ),
    overrides: str = "",
    padding: int = 0,
) -> str:
    return (
        extraction_config(names=names, expected=len(names), padding=padding)
        + "\n[scale]\n"
        + scale
        + "\n\n[pixelize]\n"
        + pixelize
        + ("\n\n" + overrides if overrides else "")
        + "\n"
    )
