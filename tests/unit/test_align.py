from __future__ import annotations

from pathlib import PurePosixPath
from typing import cast

import numpy as np
import pytest

from pixipix.config import Anchor, OutputConfig
from pixipix.models import AlignmentFrame, AlignmentRectangle
from pixipix.serialization import to_json_data
from pixipix.stages.align import (
    EMPTY_RECTANGLE,
    calculate_alignment_frame,
    compose_aligned_canvas,
    mathematical_floor_center,
)


def _output(
    *,
    width: int = 10,
    height: int = 8,
    anchor: str = "top-left",
    baseline: int | None = None,
) -> OutputConfig:
    effective = (height if baseline is None else baseline) if anchor.startswith("bottom-") else None
    return OutputConfig(
        width,
        height,
        cast(Anchor, anchor),
        baseline,
        effective,
        "error",
    )


def _frame(
    *,
    width: int = 4,
    height: int = 3,
    output: OutputConfig | None = None,
    dx: int = 0,
    dy: int = 0,
) -> AlignmentFrame:
    return calculate_alignment_frame(
        name="frame-a",
        relative_path=PurePosixPath("frames/frame-a.png"),
        source_order=0,
        input_width=width,
        input_height=height,
        output=output or _output(),
        dx=dx,
        dy=dy,
    )


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("top-left", (0, 0)),
        ("top-center", (3, 0)),
        ("top-right", (6, 0)),
        ("center-left", (0, 2)),
        ("center", (3, 2)),
        ("center-right", (6, 2)),
        ("bottom-left", (0, 5)),
        ("bottom-center", (3, 5)),
        ("bottom-right", (6, 5)),
    ],
)
def test_all_nine_anchor_origins(anchor: str, expected: tuple[int, int]) -> None:
    frame = _frame(output=_output(anchor=anchor))
    assert (frame.base_x, frame.base_y) == expected
    assert (frame.final_x, frame.final_y) == expected
    assert not frame.clipped


def test_odd_remainder_and_negative_centering_use_mathematical_floor() -> None:
    odd = _frame(width=4, height=3, output=_output(width=11, height=10, anchor="center"))
    assert (odd.base_x, odd.base_y) == (3, 3)
    assert odd.visible_destination_rectangle == AlignmentRectangle(3, 3, 4, 3)
    oversized = _frame(width=13, height=11, output=_output(width=10, height=10, anchor="center"))
    assert mathematical_floor_center(10, 13) == -2
    assert mathematical_floor_center(10, 11) == -1
    assert (oversized.base_x, oversized.base_y) == (-2, -1)
    assert (
        oversized.left_overflow,
        oversized.top_overflow,
        oversized.right_overflow,
        oversized.bottom_overflow,
    ) == (2, 1, 1, 0)


def test_default_explicit_and_zero_bottom_baselines() -> None:
    default = _frame(width=6, height=4, output=_output(width=20, height=15, anchor="bottom-right"))
    explicit = _frame(
        width=20,
        height=30,
        output=_output(width=48, height=48, anchor="bottom-center", baseline=44),
    )
    zero = _frame(
        width=8,
        height=10,
        output=_output(width=32, height=32, anchor="bottom-right", baseline=0),
    )
    assert (default.base_x, default.base_y) == (14, 11)
    assert (explicit.base_x, explicit.base_y) == (14, 14)
    assert (zero.base_x, zero.base_y) == (24, -10)
    assert zero.top_overflow == 10
    assert zero.visible_source_rectangle == EMPTY_RECTANGLE
    assert zero.visible_destination_rectangle == EMPTY_RECTANGLE


def test_offsets_apply_after_base_and_can_cause_clipping() -> None:
    frame = _frame(
        width=8,
        height=6,
        output=_output(width=12, height=12, anchor="bottom-center"),
        dx=4,
    )
    assert (frame.base_x, frame.base_y) == (2, 6)
    assert (frame.final_x, frame.final_y) == (6, 6)
    assert frame.right_overflow == 2
    assert frame.visible_source_rectangle == AlignmentRectangle(0, 0, 6, 6)
    assert frame.visible_destination_rectangle == AlignmentRectangle(6, 6, 6, 6)


def test_extreme_offsets_use_exact_unbounded_integer_geometry() -> None:
    magnitude = 9_223_372_036_854_775_807
    positive = _frame(width=4, height=3, dx=magnitude, dy=magnitude)
    negative = _frame(width=4, height=3, dx=-magnitude, dy=-magnitude)
    assert positive.right_overflow == magnitude + 4 - positive.output_width
    assert positive.bottom_overflow == magnitude + 3 - positive.output_height
    assert negative.left_overflow == magnitude
    assert negative.top_overflow == magnitude
    assert positive.visible_source_rectangle == negative.visible_source_rectangle == EMPTY_RECTANGLE
    assert (
        positive.visible_destination_rectangle
        == negative.visible_destination_rectangle
        == EMPTY_RECTANGLE
    )


@pytest.mark.parametrize(
    ("dx", "dy", "overflow", "source", "destination"),
    [
        (-2, 3, (2, 0, 0, 0), (2, 0, 4, 5), (0, 3, 4, 5)),
        (3, -2, (0, 2, 0, 0), (0, 2, 5, 4), (3, 0, 5, 4)),
        (7, 2, (0, 0, 3, 0), (0, 0, 3, 6), (7, 2, 3, 6)),
        (2, 8, (0, 0, 0, 4), (0, 0, 6, 2), (2, 8, 6, 2)),
        (-3, -2, (3, 2, 0, 0), (3, 2, 5, 6), (0, 0, 5, 6)),
    ],
)
def test_exact_clipping_rectangles(
    dx: int,
    dy: int,
    overflow: tuple[int, int, int, int],
    source: tuple[int, int, int, int],
    destination: tuple[int, int, int, int],
) -> None:
    frame = _frame(
        width=8 if dx < 0 and dy < 0 else (5 if dy < 0 else 6),
        height=8 if dx < 0 and dy < 0 else (5 if dx < 0 else 6),
        output=_output(width=10 if dx != -2 else 12, height=10 if dx != -2 else 12),
        dx=dx,
        dy=dy,
    )
    assert (
        frame.left_overflow,
        frame.top_overflow,
        frame.right_overflow,
        frame.bottom_overflow,
    ) == overflow
    assert frame.visible_source_rectangle == AlignmentRectangle(*source)
    assert frame.visible_destination_rectangle == AlignmentRectangle(*destination)


@pytest.mark.parametrize(
    ("dx", "dy"),
    [(-4, 0), (10, 0), (0, -4), (0, 10), (-4, -4), (20, 20)],
)
def test_every_fully_offcanvas_direction_uses_identical_empty_rectangles(dx: int, dy: int) -> None:
    frame = _frame(
        width=4,
        height=4,
        output=_output(width=10, height=10),
        dx=dx,
        dy=dy,
    )
    assert frame.visible_source_rectangle == EMPTY_RECTANGLE
    assert frame.visible_destination_rectangle == EMPTY_RECTANGLE
    assert to_json_data(frame.visible_source_rectangle) == {
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
    }


def test_exact_boundary_tangent_has_zero_visible_area() -> None:
    left = _frame(width=4, height=4, output=_output(width=10, height=10), dx=-4)
    right = _frame(width=4, height=4, output=_output(width=10, height=10), dx=10)
    assert left.visible_source_rectangle == right.visible_source_rectangle == EMPTY_RECTANGLE
    assert left.left_overflow == 4
    assert right.right_overflow == 4


def test_composition_copies_exact_asymmetric_pixels_without_mutation() -> None:
    pixels = np.array(
        [
            [[1, 2, 3, 255], [4, 5, 6, 128], [90, 80, 70, 0]],
            [[7, 8, 9, 64], [10, 11, 12, 255], [13, 14, 15, 255]],
        ],
        dtype=np.uint8,
    )
    before = pixels.copy()
    frame = _frame(
        width=3,
        height=2,
        output=_output(width=5, height=4, anchor="bottom-right"),
    )
    canvas = compose_aligned_canvas(pixels, frame)
    assert canvas.shape == (4, 5, 4)
    assert np.array_equal(canvas[2:4, 2:5], pixels)
    assert not canvas[:2].any()
    assert not canvas[:, :2].any()
    assert canvas[2, 4].tolist() == [90, 80, 70, 0]
    assert np.array_equal(pixels, before)
    canvas[2, 2] = (255, 255, 255, 255)
    assert np.array_equal(pixels, before)


def test_composition_copies_only_explicit_visible_region() -> None:
    pixels = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    frame = _frame(width=4, height=4, output=_output(width=3, height=3), dx=-1, dy=-1)
    canvas = compose_aligned_canvas(pixels, frame)
    assert np.array_equal(canvas, pixels[1:4, 1:4])
