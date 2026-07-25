from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import pytest

from pixipix.config import Anchor, OutputConfig, load_config
from pixipix.errors import AlignmentClippingError, ResourcePolicyError
from pixipix.models import AlignmentFrame, AlignmentRectangle
from pixipix.resources import (
    ResourceFinding,
    ResourcePolicy,
    ResourceProjection,
    enforce_resource_policy,
    resource_findings,
)
from pixipix.serialization import to_json_data
from pixipix.stages.align import (
    EMPTY_RECTANGLE,
    calculate_alignment_frame,
    compose_aligned_canvas,
    mathematical_floor_center,
    project_align_resources,
    project_align_stage,
)
from pixipix.stages.io import validate_stage_input
from tests.helpers import alignment_config, write_config, write_declared_pixelize_stage


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


@pytest.mark.parametrize(
    ("config_text", "dimensions"),
    [
        (
            alignment_config(
                names=("frame-a",),
                width=10,
                height=8,
                anchor="bottom-center",
                baseline_y=6,
                clip_policy="warn",
            ),
            (3, 2),
        ),
        (
            alignment_config(
                names=("frame-a",),
                width=11,
                height=10,
                anchor="center",
                clip_policy="warn",
            ),
            (4, 3),
        ),
        (
            alignment_config(
                names=("frame-a",),
                width=10,
                height=8,
                anchor="top-left",
                clip_policy="warn",
                offsets="[frame_offsets.frame-a]\ndx = 2\ndy = -1",
            ),
            (4, 3),
        ),
        (
            alignment_config(
                names=("frame-a",),
                width=10,
                height=10,
                anchor="center",
                clip_policy="warn",
            ),
            (13, 11),
        ),
    ],
    ids=["bottom-baseline", "center-odd", "declared-offset", "overflow"],
)
def test_alignment_stage_plan_stays_synchronized_with_placement_helper(
    tmp_path: Path,
    config_text: str,
    dimensions: tuple[int, int],
) -> None:
    config_path = tmp_path / "project.toml"
    write_config(config_path, config_text)
    loaded = load_config(config_path)
    pixelized = tmp_path / "pixelized"
    write_declared_pixelize_stage(pixelized, loaded, (dimensions,))
    validated = validate_stage_input(pixelized, "pixelize")

    plan = project_align_stage(validated, loaded)
    output = loaded.config.output
    assert output is not None
    offset = loaded.config.frame_offsets[0] if loaded.config.frame_offsets else None
    expected = calculate_alignment_frame(
        name="frame-a",
        relative_path=PurePosixPath("frames/frame-a.png"),
        source_order=0,
        input_width=dimensions[0],
        input_height=dimensions[1],
        output=output,
        dx=offset.dx if offset is not None else 0,
        dy=offset.dy if offset is not None else 0,
    )

    assert plan.frames == (expected,)
    assert (
        plan.frames[0].final_x,
        plan.frames[0].final_y,
        plan.frames[0].left_overflow,
        plan.frames[0].top_overflow,
        plan.frames[0].right_overflow,
        plan.frames[0].bottom_overflow,
        plan.frames[0].visible_source_rectangle,
        plan.frames[0].visible_destination_rectangle,
    ) == (
        expected.final_x,
        expected.final_y,
        expected.left_overflow,
        expected.top_overflow,
        expected.right_overflow,
        expected.bottom_overflow,
        expected.visible_source_rectangle,
        expected.visible_destination_rectangle,
    )


def test_alignment_clipping_policy_precedes_or_reaches_resource_enforcement(
    tmp_path: Path,
) -> None:
    plans = {}
    for policy in ("error", "warn"):
        root = tmp_path / policy
        root.mkdir()
        config_path = root / "project.toml"
        write_config(
            config_path,
            alignment_config(
                names=("frame-a",),
                width=1,
                height=1,
                anchor="center",
                clip_policy=policy,
            )
            + "\n[resources]\nmax_aggregate_input_pixels = 1\n",
        )
        loaded = load_config(config_path)
        pixelized = root / "pixelized"
        write_declared_pixelize_stage(pixelized, loaded, ((10, 10),))
        validated = validate_stage_input(pixelized, "pixelize")
        if policy == "error":
            with pytest.raises(AlignmentClippingError):
                project_align_stage(validated, loaded)
        else:
            plans[policy] = (project_align_stage(validated, loaded), loaded)

    warn_plan, warn_loaded = plans["warn"]
    assert len(warn_plan.clipping_findings) == 1
    assert warn_plan.clipping_findings[0].frame_name == "frame-a"
    with pytest.raises(ResourcePolicyError):
        enforce_resource_policy(
            warn_plan.projection,
            warn_loaded.config.resources,
        )


def test_align_modeled_byte_boundary_admits_equality_and_refuses_plus_one() -> None:
    equality_output = _output(width=4000, height=4000)
    equality_frames = tuple(
        calculate_alignment_frame(
            name=f"frame-{index}",
            relative_path=PurePosixPath(f"frames/frame-{index}.png"),
            source_order=index,
            input_width=2500,
            input_height=2800,
            output=equality_output,
            dx=0,
            dy=0,
        )
        for index in range(10)
    )
    plus_one_output = _output(width=13, height=1_230_769)
    plus_one_sizes = ((2800, 2700),) * 9 + ((1319, 1486),)
    plus_one_frames = tuple(
        calculate_alignment_frame(
            name=f"frame-{index}",
            relative_path=PurePosixPath(f"frames/frame-{index}.png"),
            source_order=index,
            input_width=width,
            input_height=height,
            output=plus_one_output,
            dx=0,
            dy=0,
        )
        for index, (width, height) in enumerate(plus_one_sizes)
    )
    policy = ResourcePolicy(
        max_aggregate_input_pixels=150_000_000,
        max_aggregate_output_pixels=160_000_000,
        max_modeled_peak_live_bytes=1_000_000_000,
    )

    equality = project_align_resources(equality_frames, 4000, 4000)
    plus_one = project_align_resources(plus_one_frames, 13, 1_230_769)

    assert equality == ResourceProjection(
        "align",
        70_000_000,
        160_000_000,
        1_000_000_000,
    )
    assert resource_findings(equality, policy) == ()
    enforce_resource_policy(equality, policy)
    assert plus_one == ResourceProjection(
        "align",
        70_000_034,
        159_999_970,
        1_000_000_001,
    )
    assert resource_findings(plus_one, policy) == (
        ResourceFinding(
            "modeled_peak_live_bytes",
            1_000_000_001,
            1_000_000_000,
        ),
    )
    with pytest.raises(ResourcePolicyError):
        enforce_resource_policy(plus_one, policy)
