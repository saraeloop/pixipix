"""Deterministic fixed-canvas placement geometry."""

from __future__ import annotations

from pathlib import PurePosixPath

from pixipix.config import OutputConfig
from pixipix.models import AlignmentFrame, AlignmentRectangle

EMPTY_RECTANGLE = AlignmentRectangle(0, 0, 0, 0)


def mathematical_floor_center(canvas_size: int, input_size: int) -> int:
    """Return floor((canvas_size - input_size) / 2) using exact integers.

    Python integer floor division is mathematical floor, including for negative
    odd differences; it therefore cannot truncate negative halves toward zero.
    """

    return (canvas_size - input_size) // 2


def _axis_parts(anchor: str) -> tuple[str, str]:
    if anchor == "center":
        return "center", "center"
    vertical, horizontal = anchor.split("-", maxsplit=1)
    return horizontal, vertical


def calculate_alignment_frame(
    *,
    name: str,
    relative_path: PurePosixPath,
    source_order: int,
    input_width: int,
    input_height: int,
    output: OutputConfig,
    dx: int = 0,
    dy: int = 0,
) -> AlignmentFrame:
    """Calculate placement, overflow, and explicit visible rectangles."""

    if input_width <= 0 or input_height <= 0:
        raise ValueError("alignment input dimensions must be positive")
    horizontal, vertical = _axis_parts(output.anchor)
    if horizontal == "left":
        base_x = 0
    elif horizontal == "center":
        base_x = mathematical_floor_center(output.frame_width, input_width)
    else:
        base_x = output.frame_width - input_width
    if vertical == "top":
        base_y = 0
    elif vertical == "center":
        base_y = mathematical_floor_center(output.frame_height, input_height)
    else:
        if output.effective_baseline_y is None:
            raise ValueError("bottom anchors require an effective baseline")
        base_y = output.effective_baseline_y - input_height

    final_x = base_x + dx
    final_y = base_y + dy
    left_overflow = max(0, -final_x)
    top_overflow = max(0, -final_y)
    right_overflow = max(0, final_x + input_width - output.frame_width)
    bottom_overflow = max(0, final_y + input_height - output.frame_height)
    clipped = any((left_overflow, top_overflow, right_overflow, bottom_overflow))

    source_left = max(0, -final_x)
    source_top = max(0, -final_y)
    source_right = min(input_width, output.frame_width - final_x)
    source_bottom = min(input_height, output.frame_height - final_y)
    visible_width = max(0, source_right - source_left)
    visible_height = max(0, source_bottom - source_top)
    if visible_width == 0 or visible_height == 0:
        source_rectangle = EMPTY_RECTANGLE
        destination_rectangle = EMPTY_RECTANGLE
    else:
        source_rectangle = AlignmentRectangle(
            source_left,
            source_top,
            visible_width,
            visible_height,
        )
        destination_rectangle = AlignmentRectangle(
            max(0, final_x),
            max(0, final_y),
            visible_width,
            visible_height,
        )
    return AlignmentFrame(
        name=name,
        relative_path=relative_path,
        source_order=source_order,
        input_width=input_width,
        input_height=input_height,
        base_x=base_x,
        base_y=base_y,
        offset_dx=dx,
        offset_dy=dy,
        final_x=final_x,
        final_y=final_y,
        left_overflow=left_overflow,
        top_overflow=top_overflow,
        right_overflow=right_overflow,
        bottom_overflow=bottom_overflow,
        clipped=clipped,
        visible_source_rectangle=source_rectangle,
        visible_destination_rectangle=destination_rectangle,
        output_width=output.frame_width,
        output_height=output.frame_height,
    )
