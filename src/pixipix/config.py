"""Strict milestone-scoped TOML configuration parsing."""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

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
from pixipix.serialization import canonical_json_bytes

type RgbaColor = str
type BackgroundMode = Literal["alpha", "corner-color", "explicit-color"]
type ScaleMode = Literal["explicit-factor", "reference-frame-width", "reference-frame-height"]
type RepresentativeStrategy = Literal["majority", "center", "alpha-weighted-majority"]
type AlphaPolicy = Literal["binary", "preserve"]
type RemainderPolicy = Literal["pad-transparent", "error", "crop-with-warning"]
type Anchor = Literal[
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]
type ClipPolicy = Literal["error", "warn", "allow"]

MAX_SOURCE_PIXELS = 16_777_216
MAX_FRAME_FILENAME_BYTES = 120
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str | None = None
    strict: bool = True


@dataclass(frozen=True, slots=True)
class SourceConfig:
    format: Literal["png"] = "png"
    expected_components: int | None = None
    max_width: int = 4096
    max_height: int = 4096
    max_pixels: int = 16_777_216
    max_components: int = 128


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    mode: BackgroundMode = "alpha"
    alpha_threshold: int = 8
    tolerance: float = 0.0
    color: RgbaColor | None = None
    compare_alpha: bool = False
    sample_corners: bool = True


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    connectivity: Literal[4, 8] = 8
    minimum_area: int = 1
    maximum_area: int | None = None
    padding: int = 0
    row_tolerance: int = 0


@dataclass(frozen=True, slots=True)
class FramesConfig:
    names: tuple[str, ...]
    filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    mode: ScaleMode
    factor: float | None = None
    reference_frame: str | None = None
    target_size: int | None = None


@dataclass(frozen=True, slots=True)
class FrameScaleOverride:
    frame_name: str
    scale_multiplier: float


@dataclass(frozen=True, slots=True)
class PixelizeConfig:
    source_cell_size: int | None = None
    representative: RepresentativeStrategy = "alpha-weighted-majority"
    alpha_policy: AlphaPolicy = "binary"
    alpha_threshold: int = 128
    remainder_policy: RemainderPolicy = "pad-transparent"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    frame_width: int
    frame_height: int
    anchor: Anchor
    baseline_y: int | None
    effective_baseline_y: int | None
    clip_policy: ClipPolicy = "error"


@dataclass(frozen=True, slots=True)
class FrameOffset:
    frame_name: str
    dx: int
    dy: int


@dataclass(frozen=True, slots=True)
class PixiPixConfig:
    project: ProjectConfig
    resources: ResourcePolicy
    source: SourceConfig
    background: BackgroundConfig
    extract: ExtractConfig
    frames: FramesConfig
    scale: ScaleConfig | None
    frame_overrides: tuple[FrameScaleOverride, ...]
    pixelize: PixelizeConfig
    output: OutputConfig | None
    frame_offsets: tuple[FrameOffset, ...]


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: PixiPixConfig
    source_config_sha256: str
    effective_config_sha256: str


def _table(value: object, section: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(
            "PX_CONFIG_002", f'configuration section "{section}" must be a table'
        )
    return cast(dict[str, object], value)


def _reject_unknown(table: dict[str, object], allowed: set[str], section: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        dotted = f"{section}.{unknown[0]}" if section else unknown[0]
        raise ConfigurationError(
            "PX_CONFIG_003",
            f'unknown configuration key "{dotted}"',
            remediation="remove the key or use a section supported by this milestone",
        )


def _integer(
    table: dict[str, object],
    key: str,
    default: int | None,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int | None:
    value = table.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("PX_CONFIG_004", f'"{key}" must be an integer')
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ConfigurationError("PX_CONFIG_005", f'"{key}" must be at least {minimum}{upper}')
    return value


def _boolean(table: dict[str, object], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError("PX_CONFIG_006", f'"{key}" must be a boolean')
    return value


def _resource_integer(
    table: dict[str, object],
    key: str,
    default: int,
    cap: int,
) -> int:
    value = table.get(key, default)
    dotted = f"resources.{key}"
    remediation = f"use a positive integer no greater than {cap}"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            "PX_CONFIG_035",
            f'"{dotted}" must be an integer',
            remediation=remediation,
        )
    if value <= 0:
        raise ConfigurationError(
            "PX_CONFIG_036",
            f'"{dotted}" must be greater than zero',
            remediation=remediation,
        )
    if value > cap:
        raise ConfigurationError(
            "PX_CONFIG_037",
            f'"{dotted}" value {value} exceeds the maximum allowed value {cap}',
            remediation=remediation,
        )
    return value


def _parse_resources(root: dict[str, object]) -> ResourcePolicy:
    table = _table(root.get("resources", {}), "resources")
    allowed = {
        "max_aggregate_input_pixels",
        "max_aggregate_output_pixels",
        "max_modeled_peak_live_bytes",
    }
    _reject_unknown(table, allowed, "resources")
    return ResourcePolicy(
        max_aggregate_input_pixels=_resource_integer(
            table,
            "max_aggregate_input_pixels",
            DEFAULT_MAX_AGGREGATE_INPUT_PIXELS,
            MAX_AGGREGATE_INPUT_PIXELS_CAP,
        ),
        max_aggregate_output_pixels=_resource_integer(
            table,
            "max_aggregate_output_pixels",
            DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS,
            MAX_AGGREGATE_OUTPUT_PIXELS_CAP,
        ),
        max_modeled_peak_live_bytes=_resource_integer(
            table,
            "max_modeled_peak_live_bytes",
            DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES,
            MAX_MODELED_PEAK_LIVE_BYTES_CAP,
        ),
    )


def _string(table: dict[str, object], key: str, default: str | None) -> str | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("PX_CONFIG_007", f'"{key}" must be a non-empty string')
    return value.strip()


def _positive_finite(table: dict[str, object], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("PX_CONFIG_024", f'"{key}" must be numeric')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigurationError("PX_CONFIG_024", f'"{key}" must be finite and greater than zero')
    return number


def _parse_color(value: object) -> tuple[RgbaColor, bool]:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", value) is None
    ):
        raise ConfigurationError(
            "PX_CONFIG_008", 'background color must use "#RRGGBB" or "#RRGGBBAA"'
        )
    compare_alpha = len(value) == 9
    digits = value[1:].lower()
    if not compare_alpha:
        digits += "ff"
    return f"#{digits}", compare_alpha


def _filename_for(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    if name != name.strip() or normalized != normalized.strip() or normalized.endswith("."):
        raise ConfigurationError(
            "PX_CONFIG_020",
            f'frame name "{name}" has unsafe leading or trailing whitespace or punctuation',
        )
    if (
        normalized in {"", ".", ".."}
        or any(char in normalized for char in ("/", "\\", "\0", ":"))
        or any(unicodedata.category(char) in {"Cc", "Cf"} for char in normalized)
    ):
        raise ConfigurationError(
            "PX_CONFIG_020",
            f'frame name "{name}" is unsafe for output filename generation',
            remediation="remove path syntax and control characters",
        )
    mapped = "".join(char if char.isalnum() or char in "-_." else "_" for char in normalized)
    mapped = re.sub(r"_+", "_", mapped).strip("._")
    if not mapped:
        raise ConfigurationError("PX_CONFIG_020", f'frame name "{name}" has no safe filename')
    if mapped.partition(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ConfigurationError(
            "PX_CONFIG_020", f'frame name "{name}" maps to a reserved Windows filename'
        )
    filename = f"{mapped}.png"
    if len(filename.encode("utf-8")) > MAX_FRAME_FILENAME_BYTES:
        raise ConfigurationError(
            "PX_CONFIG_020",
            f'frame name "{name}" produces a filename longer than '
            f"{MAX_FRAME_FILENAME_BYTES} UTF-8 bytes",
        )
    return filename


def _parse_project(root: dict[str, object]) -> ProjectConfig:
    table = _table(root.get("project", {}), "project")
    _reject_unknown(table, {"name", "strict"}, "project")
    strict = _boolean(table, "strict", True)
    if not strict:
        raise ConfigurationError(
            "PX_CONFIG_022", "permissive mode is not supported by the extraction milestone"
        )
    return ProjectConfig(name=_string(table, "name", None), strict=strict)


def _parse_source(root: dict[str, object]) -> SourceConfig:
    table = _table(root.get("source", {}), "source")
    _reject_unknown(
        table,
        {
            "format",
            "expected_components",
            "max_width",
            "max_height",
            "max_pixels",
            "max_components",
        },
        "source",
    )
    source_format = _string(table, "format", "png")
    if source_format != "png":
        raise ConfigurationError("PX_CONFIG_009", f'unsupported source format "{source_format}"')
    return SourceConfig(
        format="png",
        expected_components=_integer(table, "expected_components", None, minimum=0),
        max_width=cast(int, _integer(table, "max_width", 4096, minimum=1, maximum=65_535)),
        max_height=cast(int, _integer(table, "max_height", 4096, minimum=1, maximum=65_535)),
        max_pixels=cast(
            int,
            _integer(table, "max_pixels", MAX_SOURCE_PIXELS, minimum=1, maximum=MAX_SOURCE_PIXELS),
        ),
        max_components=cast(
            int,
            _integer(table, "max_components", 128, minimum=1, maximum=MAX_SOURCE_PIXELS),
        ),
    )


def _parse_background(root: dict[str, object]) -> BackgroundConfig:
    table = _table(root.get("background", {}), "background")
    _reject_unknown(
        table, {"mode", "alpha_threshold", "tolerance", "color", "sample_corners"}, "background"
    )
    mode = _string(table, "mode", "alpha")
    if mode not in {"alpha", "corner-color", "explicit-color"}:
        raise ConfigurationError("PX_CONFIG_010", f'unsupported background mode "{mode}"')
    alpha_threshold = cast(int, _integer(table, "alpha_threshold", 8, minimum=1, maximum=255))
    tolerance_value = table.get("tolerance", 0.0)
    if isinstance(tolerance_value, bool) or not isinstance(tolerance_value, (int, float)):
        raise ConfigurationError("PX_CONFIG_011", '"background.tolerance" must be numeric')
    tolerance = float(tolerance_value)
    if not 0.0 <= tolerance <= 1.0:
        raise ConfigurationError("PX_CONFIG_011", '"background.tolerance" must be between 0 and 1')
    if tolerance == 0.0:
        tolerance = 0.0
    raw_color = table.get("color")
    color: RgbaColor | None = None
    compare_alpha = False
    if raw_color is not None:
        color, compare_alpha = _parse_color(raw_color)
    sample_corners = _boolean(table, "sample_corners", True)
    if mode == "explicit-color" and color is None:
        raise ConfigurationError("PX_CONFIG_012", 'explicit-color mode requires "background.color"')
    if mode != "explicit-color" and color is not None:
        raise ConfigurationError(
            "PX_CONFIG_013", '"background.color" is only valid in explicit-color mode'
        )
    if mode == "corner-color" and not sample_corners:
        raise ConfigurationError("PX_CONFIG_014", "corner-color mode requires corner sampling")
    return BackgroundConfig(
        mode=cast(BackgroundMode, mode),
        alpha_threshold=alpha_threshold,
        tolerance=tolerance,
        color=color,
        compare_alpha=compare_alpha,
        sample_corners=sample_corners,
    )


def _parse_extract(root: dict[str, object]) -> ExtractConfig:
    table = _table(root.get("extract", {}), "extract")
    _reject_unknown(
        table,
        {"connectivity", "minimum_area", "maximum_area", "padding", "row_tolerance"},
        "extract",
    )
    connectivity = _integer(table, "connectivity", 8, minimum=4, maximum=8)
    if connectivity not in {4, 8}:
        raise ConfigurationError("PX_CONFIG_015", '"extract.connectivity" must be 4 or 8')
    minimum = cast(int, _integer(table, "minimum_area", 1, minimum=1))
    maximum = _integer(table, "maximum_area", None, minimum=1)
    if maximum is not None and maximum < minimum:
        raise ConfigurationError(
            "PX_CONFIG_016", '"extract.maximum_area" must be at least minimum_area'
        )
    return ExtractConfig(
        connectivity=cast(Literal[4, 8], connectivity),
        minimum_area=minimum,
        maximum_area=maximum,
        padding=cast(int, _integer(table, "padding", 0, minimum=0)),
        row_tolerance=cast(int, _integer(table, "row_tolerance", 0, minimum=0)),
    )


def _parse_frames(root: dict[str, object]) -> FramesConfig:
    table = _table(root.get("frames", {}), "frames")
    _reject_unknown(table, {"names"}, "frames")
    raw_names = table.get("names")
    if not isinstance(raw_names, list) or not raw_names:
        raise ConfigurationError("PX_CONFIG_017", '"frames.names" must be a non-empty array')
    if not all(isinstance(name, str) and name.strip() for name in raw_names):
        raise ConfigurationError("PX_CONFIG_017", "every frame name must be a non-empty string")
    names = tuple(cast(str, name) for name in raw_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigurationError("PX_CONFIG_018", f'duplicate frame name "{duplicates[0]}"')
    filenames = tuple(_filename_for(name) for name in names)
    folded = tuple(filename.casefold() for filename in filenames)
    if len(set(folded)) != len(folded):
        raise ConfigurationError("PX_CONFIG_019", "frame names produce colliding output filenames")
    return FramesConfig(names=names, filenames=filenames)


def _parse_scale(root: dict[str, object], frames: FramesConfig) -> ScaleConfig | None:
    raw = root.get("scale")
    if raw is None:
        return None
    table = _table(raw, "scale")
    _reject_unknown(table, {"mode", "factor", "reference_frame", "target_size"}, "scale")
    mode = _string(table, "mode", None)
    if mode not in {"explicit-factor", "reference-frame-width", "reference-frame-height"}:
        raise ConfigurationError("PX_CONFIG_025", f'unsupported scale mode "{mode}"')
    factor_present = "factor" in table
    reference_present = "reference_frame" in table
    target_present = "target_size" in table
    if mode == "explicit-factor":
        if not factor_present:
            raise ConfigurationError(
                "PX_CONFIG_026", 'explicit-factor mode requires "scale.factor"'
            )
        if reference_present or target_present:
            raise ConfigurationError(
                "PX_CONFIG_026",
                'explicit-factor mode forbids "scale.reference_frame" and "scale.target_size"',
            )
        return ScaleConfig(mode="explicit-factor", factor=_positive_finite(table, "factor"))
    if factor_present:
        raise ConfigurationError("PX_CONFIG_026", 'reference scale modes forbid "scale.factor"')
    reference = _string(table, "reference_frame", None)
    if reference is None or not target_present:
        raise ConfigurationError(
            "PX_CONFIG_026",
            'reference scale modes require "scale.reference_frame" and "scale.target_size"',
        )
    if reference not in frames.names:
        available = ", ".join(frames.names)
        raise ConfigurationError(
            "PX_CONFIG_027",
            f'scale reference frame "{reference}" is not configured',
            remediation=f"choose one of: {available}",
        )
    target = cast(int, _integer(table, "target_size", None, minimum=1))
    return ScaleConfig(mode=cast(ScaleMode, mode), reference_frame=reference, target_size=target)


def _parse_frame_overrides(
    root: dict[str, object], frames: FramesConfig, scale: ScaleConfig | None
) -> tuple[FrameScaleOverride, ...]:
    root_table = _table(root.get("frame_overrides", {}), "frame_overrides")
    unknown_frames = sorted(set(root_table) - set(frames.names))
    if unknown_frames:
        raise ConfigurationError(
            "PX_CONFIG_028",
            f'frame override names unknown frame "{unknown_frames[0]}"',
        )
    overrides: list[FrameScaleOverride] = []
    for frame_name in frames.names:
        if frame_name not in root_table:
            continue
        table = _table(root_table[frame_name], f"frame_overrides.{frame_name}")
        _reject_unknown(table, {"scale_multiplier"}, f"frame_overrides.{frame_name}")
        if "scale_multiplier" not in table:
            raise ConfigurationError(
                "PX_CONFIG_029",
                f'frame override "{frame_name}" requires "scale_multiplier"',
            )
        if scale is not None and frame_name == scale.reference_frame:
            raise ConfigurationError(
                "PX_CONFIG_030",
                f'scale reference frame "{frame_name}" cannot declare a scale override',
                remediation=(
                    f"remove frame_overrides.{frame_name}.scale_multiplier or choose "
                    "another reference frame"
                ),
            )
        overrides.append(
            FrameScaleOverride(
                frame_name=frame_name,
                scale_multiplier=_positive_finite(table, "scale_multiplier"),
            )
        )
    return tuple(overrides)


def _parse_pixelize(root: dict[str, object]) -> PixelizeConfig:
    table = _table(root.get("pixelize", {}), "pixelize")
    _reject_unknown(
        table,
        {
            "source_cell_size",
            "representative",
            "alpha_policy",
            "alpha_threshold",
            "remainder_policy",
        },
        "pixelize",
    )
    representative = _string(table, "representative", "alpha-weighted-majority")
    if representative not in {"majority", "center", "alpha-weighted-majority"}:
        raise ConfigurationError(
            "PX_CONFIG_031", f'unsupported pixelize representative "{representative}"'
        )
    alpha_policy = _string(table, "alpha_policy", "binary")
    if alpha_policy not in {"binary", "preserve"}:
        raise ConfigurationError(
            "PX_CONFIG_032", f'unsupported pixelize alpha policy "{alpha_policy}"'
        )
    remainder_policy = _string(table, "remainder_policy", "pad-transparent")
    if remainder_policy not in {"pad-transparent", "error", "crop-with-warning"}:
        raise ConfigurationError(
            "PX_CONFIG_033", f'unsupported pixelize remainder policy "{remainder_policy}"'
        )
    return PixelizeConfig(
        source_cell_size=_integer(table, "source_cell_size", None, minimum=1),
        representative=cast(RepresentativeStrategy, representative),
        alpha_policy=cast(AlphaPolicy, alpha_policy),
        alpha_threshold=cast(int, _integer(table, "alpha_threshold", 128, minimum=0, maximum=255)),
        remainder_policy=cast(RemainderPolicy, remainder_policy),
    )


def _alignment_integer(
    table: dict[str, object],
    key: str,
    default: int | None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = table.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("PX_ALIGN_CONFIG_011", f'"{key}" must be an integer')
    if minimum is not None and value < minimum:
        raise ConfigurationError("PX_ALIGN_CONFIG_012", f'"{key}" must be at least {minimum}')
    if maximum is not None and value > maximum:
        raise ConfigurationError("PX_ALIGN_CONFIG_012", f'"{key}" must be at most {maximum}')
    return value


def _parse_output(root: dict[str, object]) -> OutputConfig | None:
    raw = root.get("output")
    if raw is None:
        return None
    table = _table(raw, "output")
    _reject_unknown(
        table,
        {"frame_width", "frame_height", "anchor", "baseline_y", "clip_policy"},
        "output",
    )
    missing = [key for key in ("frame_width", "frame_height", "anchor") if key not in table]
    if missing:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_001", f'output configuration requires "{missing[0]}"'
        )
    frame_width = cast(
        int,
        _alignment_integer(table, "frame_width", None, minimum=1, maximum=MAX_SOURCE_PIXELS),
    )
    frame_height = cast(
        int,
        _alignment_integer(table, "frame_height", None, minimum=1, maximum=MAX_SOURCE_PIXELS),
    )
    if frame_width * frame_height > MAX_SOURCE_PIXELS:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_002",
            f"output canvas {frame_width}x{frame_height} exceeds the safety limit",
        )
    anchor = _string(table, "anchor", None)
    anchors = set(get_args(Anchor.__value__))
    if anchor is None or anchor not in anchors:
        raise ConfigurationError("PX_ALIGN_CONFIG_003", f'unsupported output anchor "{anchor}"')
    clip_policy = _string(table, "clip_policy", "error")
    clip_policies = set(get_args(ClipPolicy.__value__))
    if clip_policy not in clip_policies:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_004", f'unsupported output clipping policy "{clip_policy}"'
        )
    baseline_y = _alignment_integer(
        table,
        "baseline_y",
        None,
        minimum=0,
        maximum=frame_height,
    )
    is_bottom = anchor.startswith("bottom-")
    if baseline_y is not None and not is_bottom:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_005", '"baseline_y" is permitted only with bottom anchors'
        )
    effective_baseline_y = frame_height if is_bottom and baseline_y is None else baseline_y
    return OutputConfig(
        frame_width=frame_width,
        frame_height=frame_height,
        anchor=cast(Anchor, anchor),
        baseline_y=baseline_y,
        effective_baseline_y=effective_baseline_y,
        clip_policy=cast(ClipPolicy, clip_policy),
    )


def _parse_frame_offsets(root: dict[str, object], frames: FramesConfig) -> tuple[FrameOffset, ...]:
    root_table = _table(root.get("frame_offsets", {}), "frame_offsets")
    unknown_frames = sorted(set(root_table) - set(frames.names))
    if unknown_frames:
        raise ConfigurationError(
            "PX_ALIGN_CONFIG_006",
            f'frame offset names unknown frame "{unknown_frames[0]}"',
        )
    offsets: list[FrameOffset] = []
    for frame_name in frames.names:
        if frame_name not in root_table:
            continue
        table = _table(root_table[frame_name], f"frame_offsets.{frame_name}")
        _reject_unknown(table, {"dx", "dy"}, f"frame_offsets.{frame_name}")
        missing = [key for key in ("dx", "dy") if key not in table]
        if missing:
            raise ConfigurationError(
                "PX_ALIGN_CONFIG_007",
                f'frame offset "{frame_name}" requires "{missing[0]}"',
            )
        dx = cast(int, _alignment_integer(table, "dx", None))
        dy = cast(int, _alignment_integer(table, "dy", None))
        if dx == 0 and dy == 0:
            raise ConfigurationError(
                "PX_ALIGN_CONFIG_008",
                f'frame offset "{frame_name}" must change at least one axis',
            )
        offsets.append(FrameOffset(frame_name, dx, dy))
    return tuple(offsets)


def load_config(path: Path) -> LoadedConfig:
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            "PX_CONFIG_001", "unable to read configuration", path=path.name
        ) from error
    try:
        raw = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(
            "PX_CONFIG_001", f"invalid TOML configuration: {error}", path=path.name
        ) from error
    root = _table(raw, "root")
    _reject_unknown(
        root,
        {
            "project",
            "resources",
            "source",
            "background",
            "extract",
            "frames",
            "scale",
            "frame_overrides",
            "pixelize",
            "output",
            "frame_offsets",
        },
        "",
    )
    frames = _parse_frames(root)
    scale = _parse_scale(root, frames)
    pixelize = _parse_pixelize(root)
    if scale is not None and scale.mode != "explicit-factor" and pixelize.source_cell_size is None:
        raise ConfigurationError(
            "PX_CONFIG_034",
            'reference scale modes require "pixelize.source_cell_size"',
        )
    config = PixiPixConfig(
        project=_parse_project(root),
        resources=_parse_resources(root),
        source=_parse_source(root),
        background=_parse_background(root),
        extract=_parse_extract(root),
        frames=frames,
        scale=scale,
        frame_overrides=_parse_frame_overrides(root, frames, scale),
        pixelize=pixelize,
        output=_parse_output(root),
        frame_offsets=_parse_frame_offsets(root, frames),
    )
    expected = config.source.expected_components
    if expected is not None and expected != len(config.frames.names):
        raise ConfigurationError(
            "PX_CONFIG_021",
            "source.expected_components must equal the configured frame-name count",
        )
    if config.source.max_components > config.source.max_pixels:
        raise ConfigurationError(
            "PX_CONFIG_023", "source.max_components must not exceed source.max_pixels"
        )
    if len(config.frames.names) > config.source.max_components:
        raise ConfigurationError(
            "PX_CONFIG_023", "frame-name count must not exceed source.max_components"
        )
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    effective_hash = hashlib.sha256(canonical_json_bytes(config, pretty=False)).hexdigest()
    return LoadedConfig(config, source_hash, effective_hash)
