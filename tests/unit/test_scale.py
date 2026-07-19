from __future__ import annotations

import numpy as np
from PIL import Image

from pixipix.stages.scale import (
    premultiplied_box_resize,
    round_channel_half_away_from_zero,
    round_half_away_from_zero,
    transformed_dimension,
)


def test_round_half_away_from_zero_boundaries() -> None:
    assert [round_half_away_from_zero(value) for value in (1.4, 1.5, 1.6)] == [1, 2, 2]
    assert [round_half_away_from_zero(value) for value in (-1.4, -1.5, -1.6)] == [-1, -2, -2]
    assert round_half_away_from_zero(2.5) == 3
    assert round_half_away_from_zero(0.5) == 1


def test_tiny_nonempty_dimension_stays_nonempty() -> None:
    assert transformed_dimension(1, 0.01) == 1
    assert transformed_dimension(3, 0.5) == 2


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
