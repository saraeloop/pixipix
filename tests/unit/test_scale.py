from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixipix.config import load_config
from pixipix.resources import ResourceProjection
from pixipix.stages.io import validate_stage_input
from pixipix.stages.scale import (
    premultiplied_box_resize,
    project_scale_stage,
    round_channel_half_away_from_zero,
    round_half_away_from_zero,
    transformed_dimension,
)
from tests.helpers import pipeline_config, write_config, write_declared_extract_stage


def test_round_half_away_from_zero_boundaries() -> None:
    assert [round_half_away_from_zero(value) for value in (1.4, 1.5, 1.6)] == [1, 2, 2]
    assert [round_half_away_from_zero(value) for value in (-1.4, -1.5, -1.6)] == [-1, -2, -2]
    assert round_half_away_from_zero(2.5) == 3
    assert round_half_away_from_zero(0.5) == 1


def test_tiny_nonempty_dimension_stays_nonempty() -> None:
    assert transformed_dimension(1, 0.01) == 1
    assert transformed_dimension(3, 0.5) == 2


@pytest.mark.parametrize(
    ("scale_text", "dimensions", "overrides"),
    [
        (
            'mode = "explicit-factor"\nfactor = 1.5',
            ((3, 5), (8, 2)),
            "",
        ),
        (
            'mode = "reference-frame-width"\nreference_frame = "a"\ntarget_size = 5',
            ((3, 5), (8, 2)),
            "",
        ),
        (
            'mode = "reference-frame-height"\nreference_frame = "a"\ntarget_size = 5',
            ((3, 3), (8, 2)),
            "",
        ),
        (
            'mode = "explicit-factor"\nfactor = 0.01',
            ((1, 3), (2, 1)),
            "[frame_overrides.b]\nscale_multiplier = 2.0",
        ),
    ],
    ids=["explicit", "reference-width", "reference-height", "override-and-minimum-one"],
)
def test_scale_stage_plan_dimensions_stay_synchronized_with_transform_helper(
    tmp_path: Path,
    scale_text: str,
    dimensions: tuple[tuple[int, int], ...],
    overrides: str,
) -> None:
    config_path = tmp_path / "project.toml"
    write_config(
        config_path,
        pipeline_config(
            names=("a", "b"),
            scale=scale_text,
            overrides=overrides,
        ),
    )
    loaded = load_config(config_path)
    extracted = tmp_path / "extracted"
    write_declared_extract_stage(extracted, loaded, dimensions)
    validated = validate_stage_input(extracted, "extract")

    plan = project_scale_stage(validated, loaded)
    scale_config = loaded.config.scale
    assert scale_config is not None
    for source, planned in zip(validated.frames, plan.frames, strict=True):
        expected_width = transformed_dimension(
            source.dimensions.width,
            planned.effective_factor,
        )
        expected_height = transformed_dimension(
            source.dimensions.height,
            planned.effective_factor,
        )
        if source.name == scale_config.reference_frame:
            assert scale_config.target_size is not None
            assert loaded.config.pixelize.source_cell_size is not None
            exact_target = scale_config.target_size * loaded.config.pixelize.source_cell_size
            if scale_config.mode == "reference-frame-width":
                expected_width = exact_target
            else:
                expected_height = exact_target
        assert (
            planned.output_dimensions.width,
            planned.output_dimensions.height,
        ) == (expected_width, expected_height)
    if scale_config.factor == 0.01:
        assert all(
            frame.output_dimensions.width == frame.output_dimensions.height == 1
            for frame in plan.frames
        )


def test_scale_plan_resource_transient_preserves_each_input_output_pair(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "project.toml"
    write_config(
        config_path,
        pipeline_config(
            names=("input-heavy", "output-heavy", "paired-peak"),
            scale='mode = "explicit-factor"\nfactor = 1.0',
            overrides=(
                "[frame_overrides.input-heavy]\n"
                "scale_multiplier = 0.001\n\n"
                "[frame_overrides.output-heavy]\n"
                "scale_multiplier = 32.0"
            ),
        ),
    )
    loaded = load_config(config_path)
    extracted = tmp_path / "extracted"
    write_declared_extract_stage(
        extracted,
        loaded,
        ((1000, 1), (1, 1), (30, 30)),
    )
    validated = validate_stage_input(extracted, "extract")

    plan = project_scale_stage(validated, loaded)
    paired_areas = tuple(
        (
            frame.input_dimensions.width * frame.input_dimensions.height,
            frame.output_dimensions.width * frame.output_dimensions.height,
        )
        for frame in plan.frames
    )
    transients = tuple(
        max(
            36 * input_area,
            24 * input_area + 28 * output_area,
            20 * input_area + 56 * output_area,
            5 * output_area,
        )
        for input_area, output_area in paired_areas
    )

    assert paired_areas == ((1000, 1), (1, 1024), (900, 900))
    assert transients == (36_000, 57_364, 68_400)
    assert transients.index(max(transients)) == 2
    assert plan.projection == ResourceProjection("scale", 1_901, 1_925, 83_704)


def test_channel_rounding_is_separate_and_clamped() -> None:
    assert round_channel_half_away_from_zero(127.5) == 128
    assert round_channel_half_away_from_zero(-2.0) == 0
    assert round_channel_half_away_from_zero(300.0) == 255


def test_premultiplied_resize_has_no_black_fringe() -> None:
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (240, 30, 10, 255)
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output[0, 0].tolist() == [240, 30, 10, 64]


def test_hidden_transparent_rgb_does_not_survive() -> None:
    pixels = np.array(
        [[[255, 0, 255, 0], [0, 200, 80, 255]]],
        dtype=np.uint8,
    )
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output[0, 0].tolist() == [0, 200, 80, 128]


def test_alpha_zero_is_transparent_black_and_input_is_not_mutated() -> None:
    pixels = np.full((2, 2, 4), (10, 20, 30, 0), dtype=np.uint8)
    before = pixels.copy()
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output.tolist() == [[[0, 0, 0, 0]]]
    assert np.array_equal(pixels, before)


def test_opaque_input_matches_ordinary_box() -> None:
    pixels = np.array(
        [
            [[10, 20, 30, 255], [30, 40, 50, 255]],
            [[50, 60, 70, 255], [70, 80, 90, 255]],
        ],
        dtype=np.uint8,
    )
    expected = np.asarray(Image.fromarray(pixels, mode="RGBA").resize((1, 1), Image.Resampling.BOX))
    assert np.array_equal(premultiplied_box_resize(pixels, (1, 1)), expected)


def test_random_opaque_inputs_are_byte_equivalent_to_native_box() -> None:
    generator = np.random.default_rng(7)
    for source_width, source_height in ((2, 3), (3, 5), (7, 4)):
        pixels = generator.integers(0, 256, (source_height, source_width, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        for target in ((1, 1), (2, 2), (5, 3), (9, 8)):
            expected = np.asarray(
                Image.fromarray(pixels, mode="RGBA").resize(target, Image.Resampling.BOX)
            )
            assert np.array_equal(premultiplied_box_resize(pixels, target), expected)


def test_resize_up_down_and_repeat_are_deterministic() -> None:
    pixels = np.array([[[20, 40, 60, 200], [80, 100, 120, 100]]], dtype=np.uint8)
    up = premultiplied_box_resize(pixels, (5, 3))
    down_first = premultiplied_box_resize(up, (1, 1))
    down_second = premultiplied_box_resize(up, (1, 1))
    assert up.shape == (3, 5, 4)
    assert np.array_equal(down_first, down_second)
