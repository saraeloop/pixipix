from __future__ import annotations

from pathlib import Path

import pytest

from pixipix.config import MAX_SOURCE_PIXELS, LoadedConfig, load_config
from pixipix.errors import ConfigurationError
from pixipix.serialization import to_json_data
from tests.helpers import alignment_config, pipeline_config, write_config


def _load(tmp_path: Path, content: str, name: str = "config.toml") -> LoadedConfig:
    path = tmp_path / name
    write_config(path, content)
    return load_config(path)


@pytest.mark.parametrize(
    "anchor",
    [
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center",
        "center-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ],
)
def test_all_nine_anchors_validate(tmp_path: Path, anchor: str) -> None:
    loaded = _load(tmp_path, alignment_config(anchor=anchor))
    assert loaded.config.output is not None
    assert loaded.config.output.anchor == anchor
    expected_baseline = 8 if anchor.startswith("bottom-") else None
    assert loaded.config.output.effective_baseline_y == expected_baseline


@pytest.mark.parametrize("policy", ["error", "warn", "allow"])
def test_all_clipping_policies_validate(tmp_path: Path, policy: str) -> None:
    loaded = _load(tmp_path, alignment_config(clip_policy=policy))
    assert loaded.config.output is not None
    assert loaded.config.output.clip_policy == policy


def test_omitted_and_explicit_error_have_same_effective_config_and_hash(tmp_path: Path) -> None:
    omitted = _load(tmp_path, alignment_config(clip_policy=None), "omitted.toml")
    explicit = _load(tmp_path, alignment_config(clip_policy="error"), "explicit.toml")
    assert omitted.config.output is not None
    assert explicit.config.output is not None
    assert omitted.config.output.clip_policy == "error"
    assert explicit.config.output.clip_policy == "error"
    assert omitted.config == explicit.config
    assert to_json_data(omitted.config) == to_json_data(explicit.config)
    assert omitted.effective_config_sha256 == explicit.effective_config_sha256
    assert omitted.source_config_sha256 != explicit.source_config_sha256


@pytest.mark.parametrize("baseline", [0, 8])
def test_bottom_baseline_inclusive_bounds(tmp_path: Path, baseline: int) -> None:
    loaded = _load(tmp_path, alignment_config(baseline_y=baseline))
    assert loaded.config.output is not None
    assert loaded.config.output.baseline_y == baseline
    assert loaded.config.output.effective_baseline_y == baseline


@pytest.mark.parametrize("anchor", ["top-left", "top-center", "center", "center-right"])
def test_baseline_rejected_for_nonbottom_anchor(tmp_path: Path, anchor: str) -> None:
    with pytest.raises(ConfigurationError, match="PX_ALIGN_CONFIG_005"):
        _load(tmp_path, alignment_config(anchor=anchor, baseline_y=4))


@pytest.mark.parametrize(
    "output",
    [
        'frame_width = 0\nframe_height = 8\nanchor = "center"',
        'frame_width = 8\nframe_height = -1\nanchor = "center"',
        'frame_width = true\nframe_height = 8\nanchor = "center"',
        'frame_width = 8\nframe_height = true\nanchor = "center"',
        'frame_width = 4097\nframe_height = 4097\nanchor = "center"',
        'frame_width = 8\nframe_height = 8\nanchor = "middle"',
        'frame_width = 8\nframe_height = 8\nanchor = "center"\nclip_policy = "silent"',
        'frame_width = 8\nframe_height = 8\nanchor = "center"\nauto_fit = true',
        'frame_width = 8\nframe_height = 8\nanchor = "bottom-center"\nbaseline_y = -1',
        'frame_width = 8\nframe_height = 8\nanchor = "bottom-center"\nbaseline_y = 9',
        'frame_width = 8\nframe_height = 8\nanchor = "bottom-center"\nbaseline_y = true',
    ],
)
def test_invalid_output_contracts(tmp_path: Path, output: str) -> None:
    with pytest.raises(ConfigurationError):
        _load(tmp_path, pipeline_config(output=output))


@pytest.mark.parametrize(
    "output",
    [
        'frame_height = 8\nanchor = "center"',
        'frame_width = 8\nanchor = "center"',
        "frame_width = 8\nframe_height = 8",
    ],
)
def test_required_output_fields(tmp_path: Path, output: str) -> None:
    with pytest.raises(ConfigurationError, match="PX_ALIGN_CONFIG_001"):
        _load(tmp_path, pipeline_config(output=output))


@pytest.mark.parametrize(
    ("width", "height"),
    [(4096, 4096), (MAX_SOURCE_PIXELS, 1), (1, MAX_SOURCE_PIXELS)],
)
def test_canvas_safety_limit_accepts_exact_boundary_and_extreme_axes(
    tmp_path: Path, width: int, height: int
) -> None:
    loaded = _load(tmp_path, alignment_config(width=width, height=height))
    assert loaded.config.output is not None
    assert loaded.config.output.frame_width * loaded.config.output.frame_height == MAX_SOURCE_PIXELS


@pytest.mark.parametrize(
    ("width", "height"),
    [(MAX_SOURCE_PIXELS + 1, 1), (1, MAX_SOURCE_PIXELS + 1)],
)
def test_canvas_safety_limit_rejects_one_pixel_above_maximum(
    tmp_path: Path, width: int, height: int
) -> None:
    with pytest.raises(ConfigurationError, match="PX_ALIGN_CONFIG_012"):
        _load(tmp_path, alignment_config(width=width, height=height))


def test_valid_offsets_are_typed_in_frame_order(tmp_path: Path) -> None:
    offsets = "[frame_offsets.signal]\ndx = 0\ndy = -1\n\n[frame_offsets.idle]\ndx = 2\ndy = 0"
    loaded = _load(tmp_path, alignment_config(offsets=offsets))
    assert [(item.frame_name, item.dx, item.dy) for item in loaded.config.frame_offsets] == [
        ("idle", 2, 0),
        ("signal", 0, -1),
    ]


@pytest.mark.parametrize(
    "offsets",
    [
        "[frame_offsets.idle]\ndx = 0\ndy = 0",
        "[frame_offsets.missing]\ndx = 1\ndy = 0",
        '[frame_offsets.idle]\ndx = "1"\ndy = 0',
        '[frame_offsets.idle]\ndx = 1\ndy = "0"',
        "[frame_offsets.idle]\ndx = true\ndy = 0",
        "[frame_offsets.idle]\ndx = 1\ndy = false",
        "[frame_offsets.idle]\ndx = 1",
        "[frame_offsets.idle]\ndy = 1",
        "[frame_offsets.idle]\ndx = 1\ndy = 0\nunknown = 2",
    ],
)
def test_invalid_frame_offsets(tmp_path: Path, offsets: str) -> None:
    with pytest.raises(ConfigurationError):
        _load(tmp_path, alignment_config(offsets=offsets))


def test_effective_config_serialization_is_canonical_and_stable(tmp_path: Path) -> None:
    first = _load(
        tmp_path,
        alignment_config(
            width=48,
            height=48,
            baseline_y=44,
            offsets="[frame_offsets.signal]\ndx = 2\ndy = -1",
        ),
        "first.toml",
    )
    second_content = "# formatting only\n" + alignment_config(
        width=48,
        height=48,
        baseline_y=44,
        offsets="[frame_offsets.signal]\ndx=2\ndy=-1",
    ).replace(" = ", "=")
    second = _load(tmp_path, second_content, "second.toml")
    data = to_json_data(first.config)
    assert isinstance(data, dict)
    assert data["output"] == {
        "anchor": "bottom-center",
        "baselineY": 44,
        "clipPolicy": "error",
        "effectiveBaselineY": 44,
        "frameHeight": 48,
        "frameWidth": 48,
    }
    assert first.effective_config_sha256 == second.effective_config_sha256
