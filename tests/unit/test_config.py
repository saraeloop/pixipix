from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pixipix.config import MAX_SOURCE_PIXELS, load_config
from pixipix.errors import ConfigurationError
from pixipix.resources import (
    DEFAULT_MAX_AGGREGATE_INPUT_PIXELS,
    DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS,
    DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES,
    MAX_AGGREGATE_INPUT_PIXELS_CAP,
    MAX_AGGREGATE_OUTPUT_PIXELS_CAP,
    MAX_MODELED_PEAK_LIVE_BYTES_CAP,
    ResourcePolicy,
)
from tests.helpers import extraction_config, write_config


def test_valid_extraction_config_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "project.toml"
    source = extraction_config() + "\n[pixelize]\nsource_cell_size = 6\n"
    path.write_text(source, encoding="utf-8")

    loaded = load_config(path)

    assert loaded.config.frames.names == ("idle", "signal")
    assert loaded.config.pixelize.source_cell_size == 6
    assert loaded.source_config_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert len(loaded.effective_config_sha256) == 64


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (extraction_config() + "\nunknown = 1\n", "PX_CONFIG_003"),
        (
            extraction_config().replace("max_width = 64", "max_width = 64\nfuture = 1"),
            "PX_CONFIG_003",
        ),
        (extraction_config(background='mode = "guess"'), "PX_CONFIG_010"),
        (extraction_config(background='mode = "alpha"\nalpha_threshold = 0'), "PX_CONFIG_005"),
        (extraction_config().replace("connectivity = 8", "connectivity = 6"), "PX_CONFIG_015"),
        (extraction_config().replace("minimum_area = 2", "minimum_area = -1"), "PX_CONFIG_005"),
        (extraction_config(names=("same", "same")), "PX_CONFIG_018"),
        (extraction_config(names=("one",), expected=2), "PX_CONFIG_021"),
        (extraction_config(names=("../escape", "safe")), "PX_CONFIG_020"),
        (extraction_config().replace("[source]", '[source]\nformat = "jpeg"'), "PX_CONFIG_009"),
        (extraction_config() + '\n[align]\nanchor = "bottom-left"\n', "PX_CONFIG_003"),
        (extraction_config().replace("strict = true", "strict = false"), "PX_CONFIG_022"),
        (
            extraction_config().replace(
                "max_pixels = 4096", f"max_pixels = {MAX_SOURCE_PIXELS + 1}"
            ),
            "PX_CONFIG_005",
        ),
        (
            extraction_config().replace("max_components = 16", "max_components = 5000"),
            "PX_CONFIG_023",
        ),
    ],
)
def test_invalid_configurations_fail_stably(tmp_path: Path, content: str, code: str) -> None:
    path = tmp_path / "project.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=code):
        load_config(path)


def test_effective_hash_ignores_comments_and_formatting(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    write_config(first)
    second.write_text("# comment\n" + extraction_config().replace(" = ", "="), encoding="utf-8")

    first_loaded = load_config(first)
    second_loaded = load_config(second)

    assert first_loaded.source_config_sha256 != second_loaded.source_config_sha256
    assert first_loaded.effective_config_sha256 == second_loaded.effective_config_sha256


def test_filename_mapping_is_deterministic_and_collision_safe(tmp_path: Path) -> None:
    path = tmp_path / "project.toml"
    write_config(path, extraction_config(names=("idle pose", "signal!")))

    loaded = load_config(path)

    assert loaded.config.frames.names == ("idle pose", "signal!")
    assert loaded.config.frames.filenames == ("idle_pose.png", "signal.png")

    write_config(path, extraction_config(names=("A", "a")))
    with pytest.raises(ConfigurationError, match="PX_CONFIG_019"):
        load_config(path)

    write_config(path, extraction_config(names=("K", "\N{KELVIN SIGN}")))
    with pytest.raises(ConfigurationError, match="PX_CONFIG_019"):
        load_config(path)


@pytest.mark.parametrize(
    "name",
    (
        "CON",
        "nul.txt",
        "COM1",
        "trailing.",
        " leading",
        "trailing ",
        "a:b",
        "zero\u200bwidth",
        "x" * 121,
    ),
)
def test_frame_names_are_cross_platform_safe(tmp_path: Path, name: str) -> None:
    path = tmp_path / "project.toml"
    write_config(path, extraction_config(names=(name,), expected=1))

    with pytest.raises(ConfigurationError, match="PX_CONFIG_020"):
        load_config(path)


def test_color_semantics_are_canonical_and_list_order_is_preserved(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    third = tmp_path / "third.toml"
    write_config(
        first,
        extraction_config(background='mode = "explicit-color"\ncolor = "#AABBCCFF"'),
    )
    write_config(
        second,
        extraction_config(background='color="#aabbccFF"\nmode="explicit-color"'),
    )
    write_config(
        third,
        extraction_config(
            names=("signal", "idle"),
            background='mode = "explicit-color"\ncolor = "#aabbccff"',
        ),
    )

    first_loaded = load_config(first)
    second_loaded = load_config(second)
    third_loaded = load_config(third)

    assert first_loaded.config.background.color == "#aabbccff"
    assert first_loaded.source_config_sha256 != second_loaded.source_config_sha256
    assert first_loaded.effective_config_sha256 == second_loaded.effective_config_sha256
    assert first_loaded.effective_config_sha256 != third_loaded.effective_config_sha256


def test_negative_zero_has_one_canonical_numeric_representation(tmp_path: Path) -> None:
    first = tmp_path / "positive.toml"
    second = tmp_path / "negative.toml"
    write_config(
        first,
        extraction_config(background='mode = "explicit-color"\ncolor = "#010203"\ntolerance = 0.0'),
    )
    write_config(
        second,
        extraction_config(
            background='mode = "explicit-color"\ncolor = "#010203"\ntolerance = -0.0'
        ),
    )

    assert load_config(first).effective_config_sha256 == load_config(second).effective_config_sha256


def test_resource_defaults_and_explicit_defaults_have_one_effective_identity(
    tmp_path: Path,
) -> None:
    omitted = tmp_path / "omitted.toml"
    explicit = tmp_path / "explicit.toml"
    write_config(omitted)
    write_config(
        explicit,
        extraction_config()
        + (
            "\n[resources]\n"
            "max_aggregate_input_pixels = 50000000\n"
            "max_aggregate_output_pixels = 60000000\n"
            "max_modeled_peak_live_bytes = 1000000000\n"
        ),
    )

    omitted_loaded = load_config(omitted)
    explicit_loaded = load_config(explicit)

    assert omitted_loaded.config.resources == ResourcePolicy()
    assert explicit_loaded.config.resources == ResourcePolicy()
    assert omitted_loaded.source_config_sha256 != explicit_loaded.source_config_sha256
    assert omitted_loaded.effective_config_sha256 == explicit_loaded.effective_config_sha256


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("max_aggregate_input_pixels", 1, ResourcePolicy(max_aggregate_input_pixels=1)),
        (
            "max_aggregate_input_pixels",
            MAX_AGGREGATE_INPUT_PIXELS_CAP,
            ResourcePolicy(max_aggregate_input_pixels=MAX_AGGREGATE_INPUT_PIXELS_CAP),
        ),
        (
            "max_aggregate_input_pixels",
            DEFAULT_MAX_AGGREGATE_INPUT_PIXELS + 1,
            ResourcePolicy(max_aggregate_input_pixels=DEFAULT_MAX_AGGREGATE_INPUT_PIXELS + 1),
        ),
        ("max_aggregate_output_pixels", 1, ResourcePolicy(max_aggregate_output_pixels=1)),
        (
            "max_aggregate_output_pixels",
            MAX_AGGREGATE_OUTPUT_PIXELS_CAP,
            ResourcePolicy(max_aggregate_output_pixels=MAX_AGGREGATE_OUTPUT_PIXELS_CAP),
        ),
        (
            "max_aggregate_output_pixels",
            DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS + 1,
            ResourcePolicy(max_aggregate_output_pixels=DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS + 1),
        ),
        ("max_modeled_peak_live_bytes", 1, ResourcePolicy(max_modeled_peak_live_bytes=1)),
        (
            "max_modeled_peak_live_bytes",
            MAX_MODELED_PEAK_LIVE_BYTES_CAP,
            ResourcePolicy(max_modeled_peak_live_bytes=MAX_MODELED_PEAK_LIVE_BYTES_CAP),
        ),
        (
            "max_modeled_peak_live_bytes",
            DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES + 1,
            ResourcePolicy(max_modeled_peak_live_bytes=DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES + 1),
        ),
    ],
)
def test_resource_policy_accepts_positive_values_through_each_cap(
    tmp_path: Path,
    key: str,
    value: int,
    expected: ResourcePolicy,
) -> None:
    path = tmp_path / "project.toml"
    write_config(path, extraction_config() + f"\n[resources]\n{key} = {value}\n")

    assert load_config(path).config.resources == expected


@pytest.mark.parametrize(
    ("key", "cap"),
    [
        ("max_aggregate_input_pixels", MAX_AGGREGATE_INPUT_PIXELS_CAP),
        ("max_aggregate_output_pixels", MAX_AGGREGATE_OUTPUT_PIXELS_CAP),
        ("max_modeled_peak_live_bytes", MAX_MODELED_PEAK_LIVE_BYTES_CAP),
    ],
)
@pytest.mark.parametrize(
    "raw_kind",
    ["boolean", "float", "string", "array", "table"],
)
def test_resource_policy_rejects_non_integer_values_with_exact_error(
    tmp_path: Path,
    key: str,
    cap: int,
    raw_kind: str,
) -> None:
    path = tmp_path / "project.toml"
    defaults = {
        "max_aggregate_input_pixels": DEFAULT_MAX_AGGREGATE_INPUT_PIXELS,
        "max_aggregate_output_pixels": DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS,
        "max_modeled_peak_live_bytes": DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES,
    }
    raw = {
        "boolean": "true",
        "float": f"{defaults[key]}.0",
        "string": f'"{defaults[key]}"',
        "array": "[1]",
        "table": "{ value = 1 }",
    }[raw_kind]
    write_config(path, extraction_config() + f"\n[resources]\n{key} = {raw}\n")

    with pytest.raises(ConfigurationError) as raised:
        load_config(path)

    assert str(raised.value) == (
        f'PX_CONFIG_035 [config] "resources.{key}" must be an integer. '
        f"Remediation: use a positive integer no greater than {cap}"
    )


@pytest.mark.parametrize(
    ("key", "cap"),
    [
        ("max_aggregate_input_pixels", MAX_AGGREGATE_INPUT_PIXELS_CAP),
        ("max_aggregate_output_pixels", MAX_AGGREGATE_OUTPUT_PIXELS_CAP),
        ("max_modeled_peak_live_bytes", MAX_MODELED_PEAK_LIVE_BYTES_CAP),
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_resource_policy_rejects_non_positive_values_with_exact_error(
    tmp_path: Path,
    key: str,
    cap: int,
    value: int,
) -> None:
    path = tmp_path / "project.toml"
    write_config(path, extraction_config() + f"\n[resources]\n{key} = {value}\n")

    with pytest.raises(ConfigurationError) as raised:
        load_config(path)

    assert str(raised.value) == (
        f'PX_CONFIG_036 [config] "resources.{key}" must be greater than zero. '
        f"Remediation: use a positive integer no greater than {cap}"
    )


@pytest.mark.parametrize(
    ("key", "cap"),
    [
        ("max_aggregate_input_pixels", MAX_AGGREGATE_INPUT_PIXELS_CAP),
        ("max_aggregate_output_pixels", MAX_AGGREGATE_OUTPUT_PIXELS_CAP),
        ("max_modeled_peak_live_bytes", MAX_MODELED_PEAK_LIVE_BYTES_CAP),
    ],
)
def test_resource_policy_rejects_values_above_cap_with_exact_error(
    tmp_path: Path,
    key: str,
    cap: int,
) -> None:
    path = tmp_path / "project.toml"
    value = cap + 1
    write_config(path, extraction_config() + f"\n[resources]\n{key} = {value}\n")

    with pytest.raises(ConfigurationError) as raised:
        load_config(path)

    assert str(raised.value) == (
        f'PX_CONFIG_037 [config] "resources.{key}" value {value} '
        f"exceeds the maximum allowed value {cap}. "
        f"Remediation: use a positive integer no greater than {cap}"
    )


def test_resource_policy_rejects_unknown_keys_and_non_table_sections(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.toml"
    non_table = tmp_path / "non-table.toml"
    write_config(unknown, extraction_config() + "\n[resources]\nfuture_budget = 1\n")
    write_config(non_table, "resources = 1\n" + extraction_config())

    with pytest.raises(ConfigurationError, match="PX_CONFIG_003"):
        load_config(unknown)
    with pytest.raises(ConfigurationError, match="PX_CONFIG_002"):
        load_config(non_table)


def test_toml_null_is_rejected_before_resource_policy_parsing(tmp_path: Path) -> None:
    path = tmp_path / "project.toml"
    write_config(
        path,
        extraction_config() + "\n[resources]\nmax_aggregate_input_pixels = null\n",
    )

    with pytest.raises(ConfigurationError, match="PX_CONFIG_001"):
        load_config(path)


def test_changed_resource_policy_changes_effective_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    write_config(first)
    write_config(
        second,
        extraction_config() + "\n[resources]\nmax_aggregate_output_pixels = 60000001\n",
    )

    assert load_config(first).effective_config_sha256 != load_config(second).effective_config_sha256
