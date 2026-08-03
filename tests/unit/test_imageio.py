# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

import pixipix.imageio as imageio
from pixipix.config import BackgroundConfig, SourceConfig
from pixipix.errors import ProcessingError, UnsupportedInputError
from pixipix.imageio import generate_foreground_mask, load_source, write_png
from pixipix.models import SourceImage, UInt8Image
from tests.helpers import write_rgb, write_rgba


class _ImageIoPillowProxy:
    def __init__(self, open_image: Callable[..., object]) -> None:
        self.open = open_image

    def __getattr__(self, name: str) -> object:
        return getattr(Image, name)


def test_rgb_png_normalizes_to_rgba(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    pixels = np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8)
    write_rgb(path, pixels)

    source = load_source(path, SourceConfig())

    assert source.metadata.input_mode == "RGB"
    assert source.metadata.has_alpha is False
    assert source.pixels.shape == (2, 3, 4)
    assert np.all(source.pixels[:, :, 3] == 255)


def test_rgba_png_remains_semantically_identical(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    pixels = np.array([[[1, 2, 3, 4], [5, 6, 7, 255]]], dtype=np.uint8)
    write_rgba(path, pixels)

    source = load_source(path, SourceConfig())

    assert source.metadata.has_alpha is True
    assert np.array_equal(source.pixels, pixels)
    assert source.pixels.flags.owndata


@pytest.mark.parametrize(
    ("mode", "values", "has_alpha"),
    [
        ("L", np.array([[10, 20]], dtype=np.uint8), False),
        ("LA", np.array([[[10, 0], [20, 200]]], dtype=np.uint8), True),
        ("P", np.array([[0, 1]], dtype=np.uint8), True),
    ],
)
def test_supported_png_modes_normalize_intentionally(
    tmp_path: Path, mode: str, values: np.ndarray, has_alpha: bool
) -> None:
    path = tmp_path / f"{mode}.png"
    image = Image.fromarray(values, mode=mode)
    if mode == "P":
        image.putpalette([0, 0, 0, 50, 60, 70] + [0, 0, 0] * 254)
        image.info["transparency"] = 0
    image.save(path, format="PNG")

    source = load_source(path, SourceConfig())

    assert source.metadata.input_mode == mode
    assert source.metadata.has_alpha is has_alpha
    assert source.pixels.dtype == np.uint8
    assert source.pixels.shape == (1, 2, 4)
    assert source.pixels.flags.owndata


def test_malformed_png_and_unsupported_extension_fail_stably(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.png"
    malformed.write_bytes(b"not-png")
    with pytest.raises(UnsupportedInputError, match="PX_INPUT_002"):
        load_source(malformed, SourceConfig())

    unsupported = tmp_path / "bad.jpg"
    unsupported.write_bytes(b"anything")
    with pytest.raises(UnsupportedInputError, match="PX_INPUT_001"):
        load_source(unsupported, SourceConfig())

    valid = tmp_path / "truncated.png"
    write_rgba(valid, np.zeros((2, 2, 4), dtype=np.uint8))
    encoded = valid.read_bytes()
    valid.write_bytes(encoded[: len(encoded) // 2])
    with pytest.raises(UnsupportedInputError, match="PX_INPUT_002"):
        load_source(valid, SourceConfig())


def test_dimension_limit_fails_before_array_conversion(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    write_rgba(path, np.zeros((2, 3, 4), dtype=np.uint8))

    with pytest.raises(UnsupportedInputError, match="PX_INPUT_004"):
        load_source(path, SourceConfig(max_width=2))


def test_pillow_decompression_bomb_is_mapped_to_stable_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bomb.png"
    write_rgba(path, np.zeros((2, 2, 4), dtype=np.uint8))
    foundational_max_pixels = Image.MAX_IMAGE_PIXELS
    foundational_open = Image.open
    foundational_bomb_error = Image.DecompressionBombError

    def raise_decompression_bomb(_path: Path) -> None:
        raise foundational_bomb_error("simulated oversized PNG")

    with monkeypatch.context() as scoped:
        scoped.setattr(imageio, "Image", _ImageIoPillowProxy(raise_decompression_bomb))

        with pytest.raises(UnsupportedInputError) as raised:
            load_source(path, SourceConfig())

        assert raised.value.code == "PX_INPUT_004"
        assert type(raised.value.__cause__) is foundational_bomb_error
        assert str(raised.value.__cause__) == "simulated oversized PNG"
        assert foundational_max_pixels == Image.MAX_IMAGE_PIXELS
        assert Image.open is foundational_open
        assert Image.DecompressionBombError is foundational_bomb_error

    assert imageio.Image is Image
    assert foundational_max_pixels == Image.MAX_IMAGE_PIXELS
    assert Image.open is foundational_open


def _source_from_pixels(tmp_path: Path, pixels: UInt8Image) -> tuple[Path, SourceImage]:
    path = tmp_path / "source.png"
    write_rgba(path, pixels)
    return path, load_source(path, SourceConfig())


def test_alpha_background_mode_and_no_source_mutation(tmp_path: Path) -> None:
    pixels = np.array([[[7, 8, 9, 0], [1, 2, 3, 8], [4, 5, 6, 255]]], dtype=np.uint8)
    _, source = _source_from_pixels(tmp_path, pixels)
    before = source.pixels.copy()

    mask, summary = generate_foreground_mask(source, BackgroundConfig(alpha_threshold=8))

    assert mask.tolist() == [[False, True, True]]
    assert summary.pixels_removed == 1
    assert summary.foreground_touches_boundary is True
    assert np.array_equal(source.pixels, before)


def test_explicit_color_uses_documented_max_channel_tolerance(tmp_path: Path) -> None:
    pixels = np.array(
        [
            [
                [100, 100, 100, 255],
                [105, 96, 100, 255],
                [120, 100, 100, 255],
                [120, 100, 100, 0],
            ]
        ],
        dtype=np.uint8,
    )
    _, source = _source_from_pixels(tmp_path, pixels)
    config = BackgroundConfig(mode="explicit-color", tolerance=5 / 255, color="#646464ff")

    mask, summary = generate_foreground_mask(source, config)

    assert mask.tolist() == [[False, False, True, False]]
    assert summary.selected_color == "#646464ff"


def test_corner_color_agreement_and_disagreement(tmp_path: Path) -> None:
    pixels = np.full((3, 3, 4), (20, 30, 40, 255), dtype=np.uint8)
    pixels[1, 1] = (200, 100, 50, 255)
    _, source = _source_from_pixels(tmp_path, pixels)

    mask, summary = generate_foreground_mask(
        source,
        BackgroundConfig(mode="corner-color", tolerance=0.0),
    )
    assert int(mask.sum()) == 1
    assert summary.foreground_touches_boundary is False

    pixels[2, 2] = (21, 30, 40, 255)
    _, disagreeing = _source_from_pixels(tmp_path, pixels)
    with pytest.raises(ProcessingError, match="PX_BACKGROUND_002"):
        generate_foreground_mask(
            disagreeing,
            BackgroundConfig(mode="corner-color", tolerance=0.0),
        )


def test_tiny_corner_image_and_explicit_alpha_comparison(tmp_path: Path) -> None:
    pixels = np.array([[[10, 20, 30, 100]]], dtype=np.uint8)
    _, source = _source_from_pixels(tmp_path, pixels)

    tiny_mask, _ = generate_foreground_mask(
        source, BackgroundConfig(mode="corner-color", alpha_threshold=8)
    )
    rgb_mask, _ = generate_foreground_mask(
        source,
        BackgroundConfig(
            mode="explicit-color", color="#0a141eff", compare_alpha=False, alpha_threshold=8
        ),
    )
    rgba_mask, _ = generate_foreground_mask(
        source,
        BackgroundConfig(
            mode="explicit-color", color="#0a141eff", compare_alpha=True, alpha_threshold=8
        ),
    )

    assert tiny_mask.tolist() == [[False]]
    assert rgb_mask.tolist() == [[False]]
    assert rgba_mask.tolist() == [[True]]


def test_repeat_png_write_is_byte_identical_and_normalizes_transparent_rgb(tmp_path: Path) -> None:
    pixels = np.array([[[200, 100, 50, 0], [1, 2, 3, 255]]], dtype=np.uint8)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    write_png(first, pixels)
    write_png(second, pixels)

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        decoded = np.asarray(image.convert("RGBA"))
    assert decoded[0, 0].tolist() == [0, 0, 0, 0]
    assert image.info == {}
    assert pixels[0, 0].tolist() == [200, 100, 50, 0]

    data = first.read_bytes()
    chunks: list[str] = []
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunks.append(data[offset + 4 : offset + 8].decode("ascii"))
        offset += length + 12
    assert chunks == ["IHDR", "IDAT", "IEND"]


def test_png_writer_accepts_noncontiguous_rgba_and_rejects_malformed_buffers(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((2, 4, 4), dtype=np.uint8)[:, ::2, :]
    assert not pixels.flags.c_contiguous
    write_png(tmp_path / "valid.png", pixels)

    for malformed in (
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint16),
        np.zeros((2, 2), dtype=np.uint8),
    ):
        with pytest.raises(ProcessingError, match="PX_PNG_001"):
            write_png(tmp_path / "invalid.png", cast(UInt8Image, malformed))
