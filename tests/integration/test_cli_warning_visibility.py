from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np

from tests.helpers import write_config, write_rgba

SCALE_WARNING = (
    b'pixipix: warning [scale] PX_SCALE_OVERRIDE_001: frame "signal" uses explicit scale '
    b"multiplier 1.0; cross-frame consistency is user-managed\n"
)
IDLE_CROP_WARNING = (
    b'pixipix: warning [pixelize] PX_PIXELIZE_CROP_001: frame "idle" cropped top=1, '
    b"right=2 from 8x10 to 6x9\n"
)
SIGNAL_CROP_WARNING = (
    b'pixipix: warning [pixelize] PX_PIXELIZE_CROP_001: frame "signal" cropped top=1, '
    b"right=0 from 9x10 to 9x9\n"
)
ALIGN_OFFSET_WARNING = (
    b'pixipix: warning [align] PX_ALIGN_OFFSET_001: frame "signal" uses explicit '
    b"alignment offset dx=1, dy=0; placement is user-managed\n"
)
ALIGN_OFFSET_TWO_WARNING = (
    b'pixipix: warning [align] PX_ALIGN_OFFSET_001: frame "signal" uses explicit '
    b"alignment offset dx=2, dy=0; placement is user-managed\n"
)
ALIGN_CLIP_WARNING = (
    b'pixipix: warning [align] PX_ALIGN_CLIP_002: frame "signal" clipped left=0, top=0, '
    b"right=1, bottom=0\n"
)
UNKNOWN_WARNING = (
    b"pixipix: warning [quantize] PX_FUTURE_001: future-stage warning preserved verbatim\n"
)
FULL_HISTORY = SCALE_WARNING + IDLE_CROP_WARNING + SIGNAL_CROP_WARNING + ALIGN_OFFSET_WARNING
FAILURE = (
    b"PX_ALIGN_CLIP_001 [align] alignment clips 2 frame(s): idle: left=1, top=1, "
    b"right=0, bottom=1; signal: left=1, top=1, right=0, bottom=1. Remediation: "
    b"increase the canvas, adjust explicit offsets, or choose warn/allow\n"
)
SCALE_STRUCTURED = {
    "code": "PX_SCALE_OVERRIDE_001",
    "message": (
        'frame "signal" uses explicit scale multiplier 1.0; cross-frame consistency is user-managed'
    ),
    "stage": "scale",
}
IDLE_CROP_STRUCTURED = {
    "code": "PX_PIXELIZE_CROP_001",
    "message": 'frame "idle" cropped top=1, right=2 from 8x10 to 6x9',
    "stage": "pixelize",
}
SIGNAL_CROP_STRUCTURED = {
    "code": "PX_PIXELIZE_CROP_001",
    "message": 'frame "signal" cropped top=1, right=0 from 9x10 to 9x9',
    "stage": "pixelize",
}
ALIGN_OFFSET_STRUCTURED = {
    "code": "PX_ALIGN_OFFSET_001",
    "message": (
        'frame "signal" uses explicit alignment offset dx=1, dy=0; placement is user-managed'
    ),
    "stage": "align",
}
ANSI_ESCAPE = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


def _console() -> Path:
    console = Path(sys.executable).with_name("pixipix")
    assert console.is_file()
    return console


def _config(
    *,
    remainder_policy: str = "crop-with-warning",
    frame_width: int = 4,
    frame_height: int = 4,
    clip_policy: str = "warn",
    offset_dx: int | None = 1,
) -> str:
    offset = (
        f"\n[frame_offsets.signal]\ndx = {offset_dx}\ndy = 0\n" if offset_dx is not None else "\n"
    )
    return (
        "[project]\n"
        'name = "full-warning-lineage"\n'
        "strict = true\n\n"
        "[source]\n"
        'format = "png"\n'
        "expected_components = 2\n"
        "max_width = 64\n"
        "max_height = 64\n"
        "max_pixels = 4096\n"
        "max_components = 16\n\n"
        "[background]\n"
        'mode = "alpha"\n'
        "alpha_threshold = 8\n\n"
        "[extract]\n"
        "connectivity = 8\n"
        "minimum_area = 4\n"
        "padding = 1\n"
        "row_tolerance = 2\n\n"
        "[frames]\n"
        'names = ["idle", "signal"]\n\n'
        "[scale]\n"
        'mode = "explicit-factor"\n'
        "factor = 1.0\n\n"
        "[frame_overrides.signal]\n"
        "scale_multiplier = 1.0\n\n"
        "[pixelize]\n"
        "source_cell_size = 3\n"
        'representative = "alpha-weighted-majority"\n'
        'alpha_policy = "binary"\n'
        "alpha_threshold = 128\n"
        f'remainder_policy = "{remainder_policy}"\n\n'
        "[output]\n"
        f"frame_width = {frame_width}\n"
        f"frame_height = {frame_height}\n"
        'anchor = "center"\n'
        f'clip_policy = "{clip_policy}"\n'
        f"{offset}"
    )


def _prepare(root: Path, config_text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[5:13, 5:11] = (40, 80, 120, 255)
    pixels[5:13, 30:37] = (160, 100, 40, 255)
    write_rgba(root / "source.png", pixels)
    write_config(root / "scenario.toml", config_text)


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [_console(), *arguments],
        cwd=root,
        capture_output=True,
        check=False,
    )


def _run_to_pixelize(root: Path, config_text: str) -> dict[str, subprocess.CompletedProcess[bytes]]:
    _prepare(root, config_text)
    results = {
        "extract": _run(
            root,
            "extract",
            "source.png",
            "--config",
            "scenario.toml",
            "--output",
            "extracted",
        )
    }
    results["scale"] = _run(
        root,
        "scale",
        "extracted",
        "--config",
        "scenario.toml",
        "--output",
        "scaled",
    )
    results["pixelize"] = _run(
        root,
        "pixelize",
        "scaled",
        "--config",
        "scenario.toml",
        "--output",
        "pixelized",
    )
    assert all(result.returncode == 0 for result in results.values())
    return results


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _warnings(stage: Path) -> list[dict[str, str]]:
    metadata = json.loads((stage / "stage.json").read_text(encoding="utf-8"))
    return cast(list[dict[str, str]], metadata["warnings"])


def _plain_help(stdout: bytes) -> str:
    return ANSI_ESCAPE.sub(b"", stdout).decode()


def test_write_command_help_includes_flag_and_inspect_excludes_it(tmp_path: Path) -> None:
    for command in ("extract", "scale", "pixelize", "align"):
        result = _run(tmp_path, command, "--help")
        assert result.returncode == 0
        normalized_help = " ".join(_plain_help(result.stdout).split())
        assert "--show-warnings" in normalized_help
        assert "Show inherited warnings in addition to" in normalized_help
        assert "warnings created by this stage." in normalized_help

    inspect_help = _run(tmp_path, "inspect", "--help")
    assert inspect_help.returncode == 0
    assert "--show-warnings" not in _plain_help(inspect_help.stdout)

    _prepare(tmp_path / "project", _config())
    rejected = _run(
        tmp_path / "project",
        "inspect",
        "source.png",
        "--config",
        "scenario.toml",
        "--show-warnings",
    )
    assert rejected.returncode == 2
    assert rejected.stdout == b""
    assert b"show-warnings" in rejected.stderr
    assert b"candidate components:" not in rejected.stderr


def test_scale_and_pixelize_default_warnings_match_calibrated_literals(tmp_path: Path) -> None:
    results = _run_to_pixelize(tmp_path, _config())

    assert results["extract"].stdout == b"extracted 2 frame(s) to extracted\n"
    assert results["extract"].stderr == b""
    assert results["scale"].stdout == b"scaled 2 frame(s) to scaled\n"
    assert results["scale"].stderr == SCALE_WARNING
    assert _warnings(tmp_path / "scaled") == [SCALE_STRUCTURED]
    assert results["pixelize"].stdout == b"pixelized 2 frame(s) to pixelized\n"
    assert results["pixelize"].stderr == IDLE_CROP_WARNING + SIGNAL_CROP_WARNING
    assert _warnings(tmp_path / "pixelized") == [
        SCALE_STRUCTURED,
        IDLE_CROP_STRUCTURED,
        SIGNAL_CROP_STRUCTURED,
    ]
    assert (tmp_path / "pixelized" / "stage.json").is_file()


def test_align_default_and_full_history_are_deterministic_and_metadata_invariant(
    tmp_path: Path,
) -> None:
    default_root = tmp_path / "default"
    show_roots = (tmp_path / "show-a", tmp_path / "show-b")
    _run_to_pixelize(default_root, _config())
    for root in show_roots:
        _run_to_pixelize(root, _config())

    default = _run(
        default_root,
        "align",
        "pixelized",
        "--config",
        "scenario.toml",
        "--output",
        "aligned",
    )
    shown = [
        _run(
            root,
            "align",
            "pixelized",
            "--config",
            "scenario.toml",
            "--output",
            "aligned",
            "--show-warnings",
        )
        for root in show_roots
    ]

    assert default.returncode == shown[0].returncode == shown[1].returncode == 0
    assert (
        default.stdout == shown[0].stdout == shown[1].stdout == (b"aligned 2 frame(s) to aligned\n")
    )
    assert default.stderr == ALIGN_OFFSET_WARNING
    assert shown[0].stderr == shown[1].stderr == FULL_HISTORY
    assert _artifact_bytes(default_root / "aligned") == _artifact_bytes(show_roots[0] / "aligned")
    assert _artifact_bytes(show_roots[0] / "aligned") == _artifact_bytes(show_roots[1] / "aligned")

    default_metadata = json.loads(
        (default_root / "aligned" / "stage.json").read_text(encoding="utf-8")
    )
    shown_metadata = json.loads(
        (show_roots[0] / "aligned" / "stage.json").read_text(encoding="utf-8")
    )
    assert default_metadata["warnings"] == shown_metadata["warnings"]
    assert default_metadata["sourceConfigSha256"] == shown_metadata["sourceConfigSha256"]
    assert default_metadata["effectiveConfigSha256"] == shown_metadata["effectiveConfigSha256"]
    assert default_metadata["priorStage"] == shown_metadata["priorStage"]
    assert default_metadata["warnings"] == [
        SCALE_STRUCTURED,
        IDLE_CROP_STRUCTURED,
        SIGNAL_CROP_STRUCTURED,
        ALIGN_OFFSET_STRUCTURED,
    ]


def test_align_offset_and_clipping_warnings_keep_stored_order(tmp_path: Path) -> None:
    _run_to_pixelize(
        tmp_path,
        _config(remainder_policy="pad-transparent", offset_dx=2),
    )

    result = _run(
        tmp_path,
        "align",
        "pixelized",
        "--config",
        "scenario.toml",
        "--output",
        "aligned",
    )

    assert result.returncode == 0
    assert result.stdout == b"aligned 2 frame(s) to aligned\n"
    assert result.stderr == ALIGN_OFFSET_TWO_WARNING + ALIGN_CLIP_WARNING
    assert [warning["code"] for warning in _warnings(tmp_path / "aligned")] == [
        "PX_SCALE_OVERRIDE_001",
        "PX_ALIGN_OFFSET_001",
        "PX_ALIGN_CLIP_002",
    ]


def test_extract_flag_is_warning_free_and_artifact_equivalent_today(tmp_path: Path) -> None:
    roots = (tmp_path / "default", tmp_path / "shown")
    for root in roots:
        _prepare(root, _config())

    default = _run(
        roots[0],
        "extract",
        "source.png",
        "--config",
        "scenario.toml",
        "--output",
        "extracted",
    )
    shown = _run(
        roots[1],
        "extract",
        "source.png",
        "--config",
        "scenario.toml",
        "--output",
        "extracted",
        "--show-warnings",
    )

    assert default.returncode == shown.returncode == 0
    assert default.stdout == shown.stdout == b"extracted 2 frame(s) to extracted\n"
    assert default.stderr == shown.stderr == b""
    assert _warnings(roots[0] / "extracted") == _warnings(roots[1] / "extracted") == []
    assert _artifact_bytes(roots[0] / "extracted") == _artifact_bytes(roots[1] / "extracted")


def test_unknown_stage_warning_is_test_only_carry_forward_and_renders_under_flag(
    tmp_path: Path,
) -> None:
    """Mutating a published stage is test-only and unsupported user behavior."""

    _run_to_pixelize(tmp_path, _config())
    mutated_stage = tmp_path / "pixelized-mutated"
    shutil.copytree(tmp_path / "pixelized", mutated_stage)
    metadata_path = mutated_stage / "stage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    synthetic = {
        "code": "PX_FUTURE_001",
        "stage": "quantize",
        "message": "future-stage warning preserved verbatim",
    }
    metadata["warnings"].append(synthetic)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    default = _run(
        tmp_path,
        "align",
        "pixelized-mutated",
        "--config",
        "scenario.toml",
        "--output",
        "aligned-default",
    )
    shown = _run(
        tmp_path,
        "align",
        "pixelized-mutated",
        "--config",
        "scenario.toml",
        "--output",
        "aligned-shown",
        "--show-warnings",
    )

    assert default.returncode == shown.returncode == 0
    assert default.stderr == ALIGN_OFFSET_WARNING
    assert shown.stderr == (
        SCALE_WARNING
        + IDLE_CROP_WARNING
        + SIGNAL_CROP_WARNING
        + UNKNOWN_WARNING
        + ALIGN_OFFSET_WARNING
    )
    republished = _warnings(tmp_path / "aligned-shown")
    assert republished[3] == synthetic
    assert [warning["stage"] for warning in republished] == [
        "scale",
        "pixelize",
        "pixelize",
        "quantize",
        "align",
    ]


def test_alignment_failure_prints_no_warning_lines_or_output(tmp_path: Path) -> None:
    config = _config(
        remainder_policy="pad-transparent",
        frame_width=2,
        frame_height=2,
        clip_policy="error",
        offset_dx=None,
    )
    _run_to_pixelize(tmp_path, config)

    for index, extra in enumerate(((), ("--show-warnings",))):
        output = f"aligned-{index}"
        result = _run(
            tmp_path,
            "align",
            "pixelized",
            "--config",
            "scenario.toml",
            "--output",
            output,
            *extra,
        )
        assert result.returncode == 1
        assert result.stdout == b""
        assert result.stderr == FAILURE
        assert b"pixipix: warning" not in result.stderr
        assert not (tmp_path / output).exists()
