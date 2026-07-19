from __future__ import annotations

import numpy as np
import pytest

from pixipix.errors import ProcessingError
from pixipix.stages.pixelize import (
    apply_alpha_policy,
    pixelize_prepared_grid,
    prepare_cell_grid,
    representative_pixel,
)


def test_scenario_i_padding_exactly() -> None:
    pixels = np.ones((19, 25, 4), dtype=np.uint8)
    prepared = prepare_cell_grid(pixels, 6, "pad-transparent", "shape")
    assert prepared.right_padding == 5
    assert prepared.top_padding == 5
    assert prepared.pixels.shape == (24, 30, 4)
    assert not prepared.pixels[:5].any()
    assert not prepared.pixels[:, 25:].any()
    assert np.array_equal(prepared.pixels[5:, :25], pixels)
    logical = pixelize_prepared_grid(prepared.pixels, 6, "center", "preserve", 128)
    assert logical.shape == (4, 5, 4)


@pytest.mark.parametrize(
    ("shape", "top", "right"),
    [((18, 24, 4), 0, 0), ((18, 25, 4), 0, 5), ((19, 24, 4), 5, 0)],
)
def test_padding_remainder_axes(shape: tuple[int, int, int], top: int, right: int) -> None:
    prepared = prepare_cell_grid(np.ones(shape, dtype=np.uint8), 6, "pad-transparent", "f")
    assert (prepared.top_padding, prepared.right_padding) == (top, right)


def test_bottom_left_anchor_preserves_bottom_and_changes_only_top_right() -> None:
    pixels = np.zeros((3, 3, 4), dtype=np.uint8)
    pixels[-1, 0] = (10, 20, 30, 255)
    prepared = prepare_cell_grid(pixels, 2, "pad-transparent", "f")
    assert prepared.pixels[-1, 0].tolist() == [10, 20, 30, 255]
    assert prepared.pixels[0].tolist() == [[0, 0, 0, 0]] * 4


def test_error_policy_reports_remainders() -> None:
    with pytest.raises(ProcessingError, match="width remainder 1, height remainder 1"):
        prepare_cell_grid(np.zeros((3, 3, 4), dtype=np.uint8), 2, "error", "idle")


def test_crop_removes_top_and_right_only_and_warns() -> None:
    pixels = np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4)
    prepared = prepare_cell_grid(pixels, 3, "crop-with-warning", "idle")
    assert (prepared.top_crop, prepared.right_crop) == (2, 1)
    assert prepared.pixels.shape == (3, 6, 4)
    assert np.array_equal(prepared.pixels, pixels[2:, :6])
    assert prepared.warning is not None


def test_crop_to_zero_is_rejected() -> None:
    with pytest.raises(ProcessingError, match="zero dimensions"):
        prepare_cell_grid(np.zeros((2, 5, 4), dtype=np.uint8), 6, "crop-with-warning", "f")


def test_padding_rejects_unsafe_prepared_dimensions_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_pad(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe padding reached allocation")

    monkeypatch.setattr(np, "pad", fail_pad)
    with pytest.raises(ProcessingError, match="PX_PIXELIZE_002"):
        prepare_cell_grid(
            np.zeros((1, 1, 4), dtype=np.uint8),
            20_000_000,
            "pad-transparent",
            "f",
        )


def test_majority_rgba_distinction_and_first_tie() -> None:
    cell = np.array(
        [
            [[1, 2, 3, 10], [1, 2, 3, 20]],
            [[1, 2, 3, 20], [1, 2, 3, 10]],
        ],
        dtype=np.uint8,
    )
    assert representative_pixel(cell, "majority") == (1, 2, 3, 10)


def test_center_odd_and_even_convention() -> None:
    odd = np.arange(3 * 3 * 4, dtype=np.uint8).reshape(3, 3, 4)
    even = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    assert representative_pixel(odd, "center") == tuple(int(value) for value in odd[1, 1])
    assert representative_pixel(even, "center") == tuple(int(value) for value in even[1, 1])


def test_alpha_weighted_visible_color_beats_noise_and_uses_selected_alpha() -> None:
    cell = np.array(
        [
            [[200, 10, 30, 10], [4, 80, 120, 200]],
            [[200, 10, 30, 10], [4, 80, 120, 100]],
        ],
        dtype=np.uint8,
    )
    assert representative_pixel(cell, "alpha-weighted-majority") == (4, 80, 120, 167)


def test_alpha_weighted_tie_first_no_synthesized_rgb_and_all_transparent() -> None:
    tied = np.array([[[9, 8, 7, 100], [1, 2, 3, 100]]], dtype=np.uint8)
    assert representative_pixel(tied, "alpha-weighted-majority") == (9, 8, 7, 100)
    transparent = np.full((2, 2, 4), (200, 100, 50, 0), dtype=np.uint8)
    assert representative_pixel(transparent, "alpha-weighted-majority") == (0, 0, 0, 0)


def test_alpha_policies_boundaries_and_transparent_normalization() -> None:
    assert apply_alpha_policy((10, 20, 30, 127), "binary", 128) == (0, 0, 0, 0)
    assert apply_alpha_policy((10, 20, 30, 128), "binary", 128) == (10, 20, 30, 255)
    assert apply_alpha_policy((10, 20, 30, 255), "binary", 128) == (10, 20, 30, 255)
    assert apply_alpha_policy((10, 20, 30, 127), "preserve", 128) == (10, 20, 30, 127)
    assert apply_alpha_policy((10, 20, 30, 0), "preserve", 128) == (0, 0, 0, 0)


def test_sparse_opaque_coverage_may_remain_opaque() -> None:
    cell = np.zeros((2, 2, 4), dtype=np.uint8)
    cell[0, 0] = (70, 80, 90, 255)
    selected = representative_pixel(cell, "alpha-weighted-majority")
    assert apply_alpha_policy(selected, "preserve", 128) == (70, 80, 90, 255)
