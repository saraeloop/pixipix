"""Regenerate the project-owned synthetic PNG fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write(root: Path, name: str, pixels: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> None:
    Image.fromarray(pixels).save(root / name, format="PNG")


def main() -> None:
    root = Path(__file__).parent

    transparent = np.zeros((10, 16, 4), dtype=np.uint8)
    transparent[1:5, 1:5] = (30, 60, 90, 255)
    transparent[2:6, 10:15] = (100, 140, 180, 220)
    transparent[9, 15] = (255, 0, 0, 255)
    _write(root, "transparent-multi.png", transparent)

    solid = np.full((10, 16, 3), (240, 220, 80), dtype=np.uint8)
    solid[1:5, 1:5] = (30, 60, 90)
    solid[2:6, 10:15] = (100, 140, 180)
    _write(root, "solid-background.png", solid)

    connectivity = np.zeros((3, 3, 4), dtype=np.uint8)
    connectivity[0, 0] = (255, 255, 255, 255)
    connectivity[1, 1] = (255, 255, 255, 255)
    _write(root, "connectivity.png", connectivity)

    rows = np.zeros((12, 16, 4), dtype=np.uint8)
    rows[1:3, 8:10] = (80, 80, 80, 255)
    rows[2:5, 1:4] = (100, 100, 100, 255)
    rows[8:11, 2:5] = (120, 120, 120, 255)
    rows[7:10, 11:15] = (140, 140, 140, 255)
    _write(root, "multi-row.png", rows)

    robot = np.zeros((12, 20, 4), dtype=np.uint8)
    robot[2:9, 1:7] = (55, 75, 95, 255)
    robot[1:3, 3:5] = (110, 210, 235, 255)
    robot[3:10, 12:19] = (75, 95, 115, 255)
    robot[2:4, 14:17] = (255, 180, 70, 255)
    _write(root, "robot-geometric.png", robot)


if __name__ == "__main__":
    main()
