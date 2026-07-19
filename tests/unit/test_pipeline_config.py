from __future__ import annotations

from pathlib import Path

import pytest

from pixipix.config import LoadedConfig, load_config
from pixipix.errors import ConfigurationError
from tests.helpers import pipeline_config, write_config


def _load(tmp_path: Path, content: str) -> LoadedConfig:
    path = tmp_path / "pipeline.toml"
    write_config(path, content)
    return load_config(path)


@pytest.mark.parametrize(
    ("scale", "mode", "factor"),
    [
        ('mode = "explicit-factor"\nfactor = 0.75', "explicit-factor", 0.75),
        (
            'mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 4',
            "reference-frame-width",
            None,
        ),
        (
            'mode = "reference-frame-height"\nreference_frame = "signal"\ntarget_size = 3',
            "reference-frame-height",
            None,
        ),
    ],
)
def test_valid_scale_modes(tmp_path: Path, scale: str, mode: str, factor: float | None) -> None:
    loaded = _load(tmp_path, pipeline_config(scale=scale))
    assert loaded.config.scale is not None
    assert loaded.config.scale.mode == mode
    assert loaded.config.scale.factor == factor


@pytest.mark.parametrize(
    "scale",
    [
        'mode = "explicit-factor"',
        'mode = "explicit-factor"\nfactor = 1.0\nreference_frame = "idle"',
        'mode = "explicit-factor"\nfactor = 1.0\ntarget_size = 4',
        'mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 4\nfactor = 1.0',
        'mode = "reference-frame-width"\ntarget_size = 4',
        'mode = "reference-frame-width"\nreference_frame = "idle"',
        'mode = "reference-frame-width"\nreference_frame = "missing"\ntarget_size = 4',
        'mode = "reference-frame-height"\nreference_frame = "idle"\ntarget_size = 0',
        'mode = "unknown"\nfactor = 1.0',
        'mode = "explicit-factor"\nfactor = 1.0\nunknown = true',
    ],
)
def test_invalid_scale_contracts(tmp_path: Path, scale: str) -> None:
    with pytest.raises(ConfigurationError):
        _load(tmp_path, pipeline_config(scale=scale))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_nonpositive_or_nonfinite_factor_fails(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigurationError, match="PX_CONFIG_024"):
        _load(tmp_path, pipeline_config(scale=f'mode = "explicit-factor"\nfactor = {value}'))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_override_fails(tmp_path: Path, value: str) -> None:
    overrides = f"[frame_overrides.signal]\nscale_multiplier = {value}"
    with pytest.raises(ConfigurationError, match="PX_CONFIG_024"):
        _load(tmp_path, pipeline_config(overrides=overrides))


def test_override_validation_and_reference_prohibition(tmp_path: Path) -> None:
    valid = _load(
        tmp_path,
        pipeline_config(overrides="[frame_overrides.signal]\nscale_multiplier = 0.96"),
    )
    assert valid.config.frame_overrides[0].frame_name == "signal"

    reference = 'mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 4'
    with pytest.raises(ConfigurationError, match="PX_CONFIG_030"):
        _load(
            tmp_path,
            pipeline_config(
                scale=reference,
                overrides="[frame_overrides.idle]\nscale_multiplier = 0.96",
            ),
        )
    with pytest.raises(ConfigurationError, match="PX_CONFIG_028"):
        _load(
            tmp_path,
            pipeline_config(overrides="[frame_overrides.missing]\nscale_multiplier = 1.0"),
        )
    with pytest.raises(ConfigurationError, match="PX_CONFIG_003"):
        _load(
            tmp_path,
            pipeline_config(overrides="[frame_overrides.signal]\nunknown = 1"),
        )


@pytest.mark.parametrize(
    "pixelize",
    [
        "source_cell_size = 0",
        'source_cell_size = 2\nrepresentative = "mean"',
        'source_cell_size = 2\nalpha_policy = "soft"',
        "source_cell_size = 2\nalpha_threshold = 256",
        'source_cell_size = 2\nremainder_policy = "left-pad"',
        "source_cell_size = 2\nunknown = true",
    ],
)
def test_invalid_pixelize_contracts(tmp_path: Path, pixelize: str) -> None:
    with pytest.raises(ConfigurationError):
        _load(tmp_path, pipeline_config(pixelize=pixelize))


def test_pixelize_defaults_and_canonical_hash(tmp_path: Path) -> None:
    minimal = pipeline_config(pixelize="source_cell_size = 6")
    loaded = _load(tmp_path, minimal)
    assert loaded.config.pixelize.representative == "alpha-weighted-majority"
    assert loaded.config.pixelize.alpha_policy == "binary"
    assert loaded.config.pixelize.alpha_threshold == 128
    assert loaded.config.pixelize.remainder_policy == "pad-transparent"

    second_path = tmp_path / "second.toml"
    write_config(second_path, "# comment\n" + minimal.replace(" = ", "="))
    second = load_config(second_path)
    assert loaded.source_config_sha256 != second.source_config_sha256
    assert loaded.effective_config_sha256 == second.effective_config_sha256


def test_reference_mode_requires_cell_size(tmp_path: Path) -> None:
    content = pipeline_config(
        scale='mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 4',
        pixelize="",
    )
    with pytest.raises(ConfigurationError, match="PX_CONFIG_034"):
        _load(tmp_path, content)


@pytest.mark.parametrize(
    ("scale", "pixelize"),
    [
        ('mode = "explicit-factor"\nfactor = true', "source_cell_size = 2"),
        (
            'mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = true',
            "source_cell_size = 2",
        ),
        ('mode = "explicit-factor"\nfactor = 1.0', "source_cell_size = true"),
        (
            'mode = "explicit-factor"\nfactor = 1.0',
            "source_cell_size = 2\nalpha_threshold = true",
        ),
    ],
)
def test_booleans_are_not_accepted_as_pipeline_numbers(
    tmp_path: Path, scale: str, pixelize: str
) -> None:
    with pytest.raises(ConfigurationError):
        _load(tmp_path, pipeline_config(scale=scale, pixelize=pixelize))
