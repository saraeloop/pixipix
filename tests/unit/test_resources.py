from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import PurePosixPath

import pytest

from pixipix.errors import ResourcePolicyError
from pixipix.models import (
    AlignmentFrame,
    AlignmentRectangle,
    Dimensions,
    ExtractedFrame,
    PixelizeFrame,
    Rect,
    ScaleFrame,
)
from pixipix.resources import (
    ResourceFinding,
    ResourcePolicy,
    ResourceProjection,
    enforce_resource_policy,
    resource_findings,
)
from pixipix.stages.align import project_align_resources
from pixipix.stages.extract import project_extract_resources
from pixipix.stages.pixelize import project_pixelize_resources
from pixipix.stages.scale import project_scale_resources


def _path(index: int) -> PurePosixPath:
    return PurePosixPath("frames") / f"f{index}.png"


def _extracted_frame(index: int, area: int) -> ExtractedFrame:
    bounds = Rect(0, 0, area, 1)
    return ExtractedFrame(f"f{index}", _path(index), index, index, area, bounds, bounds)


def _scale_frame(
    index: int, input_size: tuple[int, int], output_size: tuple[int, int]
) -> ScaleFrame:
    return ScaleFrame(
        f"f{index}",
        _path(index),
        index,
        Dimensions(*input_size),
        Dimensions(*output_size),
        1.0,
        1.0,
    )


def _pixelize_frame(
    index: int,
    input_size: tuple[int, int],
    prepared_size: tuple[int, int],
    logical_size: tuple[int, int],
) -> PixelizeFrame:
    return PixelizeFrame(
        f"f{index}",
        _path(index),
        index,
        Dimensions(*input_size),
        Dimensions(*prepared_size),
        0,
        0,
        0,
        0,
        Dimensions(*logical_size),
    )


def _alignment_frame(index: int, size: tuple[int, int], canvas: tuple[int, int]) -> AlignmentFrame:
    empty = AlignmentRectangle(0, 0, 0, 0)
    return AlignmentFrame(
        f"f{index}",
        _path(index),
        index,
        size[0],
        size[1],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        empty,
        empty,
        canvas[0],
        canvas[1],
    )


def test_resource_models_are_frozen_and_defaults_are_complete() -> None:
    policy = ResourcePolicy()
    projection = ResourceProjection("scale", 1, 2, 3)
    finding = ResourceFinding("aggregate_input_pixels", 2, 1)

    assert (
        policy.max_aggregate_input_pixels,
        policy.max_aggregate_output_pixels,
        policy.max_modeled_peak_live_bytes,
    ) == (50_000_000, 60_000_000, 1_000_000_000)
    with pytest.raises(FrozenInstanceError):
        projection.aggregate_input_pixels = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        finding.limit = 2  # type: ignore[misc]


def test_comparison_admits_equality_and_reports_plus_one_in_fixed_order() -> None:
    policy = ResourcePolicy(50, 60, 100)
    assert resource_findings(ResourceProjection("scale", 50, 60, 100), policy) == ()

    projection = ResourceProjection("scale", 51, 61, 101)
    assert resource_findings(projection, policy) == (
        ResourceFinding("aggregate_input_pixels", 51, 50),
        ResourceFinding("aggregate_output_pixels", 61, 60),
        ResourceFinding("modeled_peak_live_bytes", 101, 100),
    )

    with pytest.raises(ResourcePolicyError) as raised:
        enforce_resource_policy(projection, policy)
    error = raised.value
    assert error.stage == "scale"
    assert error.projection == projection
    assert error.policy == policy
    assert error.findings == resource_findings(projection, policy)
    assert str(error) == (
        "PX_RESOURCE_001 [scale] aggregate resource policy exceeded: "
        "aggregate input pixels 51/50; aggregate output pixels 61/60; "
        "modeled peak live bytes under the explicit-buffer model 101/100. "
        "Remediation: reduce frame count or dimensions, adjust transformation or canvas "
        "settings, or raise the configured budget within its allowed cap when the execution "
        "environment can support it"
    )


def test_extract_formula_known_floor_and_scenario_k() -> None:
    floor_areas = (46_535,) * 22 + (16_433,)
    floor = project_extract_resources(
        1_572_498,
        tuple(_extracted_frame(index, area) for index, area in enumerate(floor_areas)),
    )
    checkpoint = project_extract_resources(
        240,
        (_extracted_frame(0, 80), _extracted_frame(1, 90)),
    )

    assert floor == ResourceProjection("extract", 1_572_498, 1_040_203, 18_359_829)
    assert checkpoint == ResourceProjection("extract", 240, 170, 2_930)
    assert project_extract_resources(10, ()) == ResourceProjection("extract", 10, 0, 90)


def test_scale_formula_heavy_and_scenarios_f_g_h() -> None:
    heavy = project_scale_resources(
        tuple(_scale_frame(index, (512, 512), (768, 768)) for index in range(64))
    )
    scenario_f = project_scale_resources(
        (
            *(_scale_frame(index, (500, 500), (1000, 1000)) for index in range(60)),
            _scale_frame(60, (1, 1), (1, 1)),
        )
    )
    scenario_g = project_scale_resources((_scale_frame(0, (4096, 4096), (4096, 4096)),))
    scenario_h = project_scale_resources(
        tuple(_scale_frame(index, (1000, 1000), (1200, 1200)) for index in range(100))
    )

    assert heavy == ResourceProjection("scale", 16_777_216, 37_748_736, 256_376_832)
    assert scenario_f == ResourceProjection("scale", 15_000_001, 60_000_001, 361_000_008)
    assert scenario_g == ResourceProjection("scale", 16_777_216, 16_777_216, 1_409_286_144)
    assert scenario_h == ResourceProjection("scale", 100_000_000, 144_000_000, 1_076_640_000)
    assert project_scale_resources(()) == ResourceProjection("scale", 0, 0, 0)


def test_pixelize_formula_heavy_and_scenario_e() -> None:
    heavy = project_pixelize_resources(
        tuple(_pixelize_frame(index, (768, 768), (768, 768), (192, 192)) for index in range(64)),
        4,
    )
    scenario_e = project_pixelize_resources(
        tuple(
            _pixelize_frame(index, (4094, 4094), (4094, 4094), (2047, 2047)) for index in range(4)
        ),
        2,
    )

    assert heavy == ResourceProjection("pixelize", 37_748_736, 2_359_296, 165_150_720)
    assert scenario_e == ResourceProjection("pixelize", 67_043_344, 16_760_836, 469_303_408)
    assert project_pixelize_resources((), 1) == ResourceProjection("pixelize", 0, 0, 0)


def test_align_formula_exact_byte_equality_and_plus_one() -> None:
    equality_canvas = (4000, 4000)
    equality = project_align_resources(
        tuple(_alignment_frame(index, (2500, 2800), equality_canvas) for index in range(10)),
        *equality_canvas,
    )
    plus_one_canvas = (13, 1_230_769)
    plus_one = project_align_resources(
        (
            *(_alignment_frame(index, (2800, 2700), plus_one_canvas) for index in range(9)),
            _alignment_frame(9, (1319, 1486), plus_one_canvas),
        ),
        *plus_one_canvas,
    )

    assert equality == ResourceProjection("align", 70_000_000, 160_000_000, 1_000_000_000)
    assert plus_one == ResourceProjection("align", 70_000_034, 159_999_970, 1_000_000_001)
    assert project_align_resources((), 1, 1) == ResourceProjection("align", 0, 0, 5)


def test_paired_frame_formulas_select_one_largest_transient() -> None:
    extract = project_extract_resources(
        10,
        (_extracted_frame(0, 3), _extracted_frame(1, 7)),
    )
    scale = project_scale_resources(
        (
            _scale_frame(0, (10, 10), (1, 1)),
            _scale_frame(1, (1, 1), (10, 10)),
        )
    )
    pixelize = project_pixelize_resources(
        (
            _pixelize_frame(0, (10, 10), (10, 10), (5, 5)),
            _pixelize_frame(1, (2, 2), (4, 4), (2, 2)),
        ),
        2,
    )

    assert extract == ResourceProjection("extract", 10, 10, 137)
    assert scale == ResourceProjection("scale", 101, 101, 6_428)
    assert pixelize == ResourceProjection("pixelize", 104, 29, 1_332)
