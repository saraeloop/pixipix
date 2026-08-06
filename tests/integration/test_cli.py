from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import zlib
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from typer.testing import CliRunner

import pixipix.cli as cli_module
from pixipix.cli import app
from tests.helpers import (
    extraction_config,
    pipeline_config,
    resource_scenario_e,
    resource_scenario_f,
    resource_scenario_h,
    transparent_sheet,
    write_config,
    write_rgba,
)

runner = CliRunner()
_CLI_SUBPROCESS_TIMEOUT_SECONDS = 10


def _console_script() -> Path:
    script = Path(sys.executable).with_name("pixipix.exe" if os.name == "nt" else "pixipix")
    assert script.is_file()
    return script


def _project(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, transparent_sheet())
    write_config(config)
    return image, config


def _write_decompression_bomb_png(path: Path) -> None:
    write_rgba(path, np.zeros((1, 1, 4), dtype=np.uint8))
    encoded = bytearray(path.read_bytes())
    assert encoded[12:16] == b"IHDR"
    encoded[16:24] = struct.pack(">II", 1_000_000, 1_000_000)
    encoded[29:33] = struct.pack(">I", zlib.crc32(encoded[12:29]))
    path.write_bytes(encoded)


def test_help_and_version_commands() -> None:
    help_result = runner.invoke(app, ["--help"])
    module_equivalent = runner.invoke(app, [])
    version_result = runner.invoke(app, ["version"])

    assert distribution_version("pixipix") == "0.1.1"
    assert help_result.exit_code == 0
    assert "Tiny poses in. Tidy pixels out." in help_result.output
    assert module_equivalent.exit_code == 0
    assert "Tiny poses in. Tidy pixels out." in module_equivalent.output
    assert version_result.exit_code == 0
    assert version_result.output == f"PixiPix {distribution_version('pixipix')}\n"


def test_actual_console_and_module_entry_points_agree() -> None:
    console = subprocess.run(
        [_console_script(), "--help"], capture_output=True, text=True, check=False
    )
    module = subprocess.run(
        [sys.executable, "-m", "pixipix"], capture_output=True, text=True, check=False
    )
    version = subprocess.run(
        [_console_script(), "version"], capture_output=True, text=True, check=False
    )

    assert console.returncode == module.returncode == version.returncode == 0
    assert "Tiny poses in. Tidy pixels out." in console.stdout
    assert "Tiny poses in. Tidy pixels out." in module.stdout
    assert version.stdout == f"PixiPix {distribution_version('pixipix')}\n"
    assert console.stderr == module.stderr == version.stderr == ""


def test_inspect_success_is_deterministic_and_writes_nothing(tmp_path: Path) -> None:
    image, config = _project(tmp_path)

    first = runner.invoke(app, ["inspect", str(image), "--config", str(config)])
    second = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert first.exit_code == 0
    assert first.output == second.output
    assert "candidate components: 3" in first.output
    assert "reasons=below-minimum-area" in first.output
    assert "bounds=(13,6)-(14,7) area=1 discovery=2" in first.output
    assert "configured source cell size: not configured" in first.output
    assert not (tmp_path / "stage.json").exists()


def test_inspect_count_mismatch_is_explicit_and_still_writes_nothing(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(config, extraction_config(names=("only",), expected=None))

    result = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert result.exit_code == 0
    assert "frame assignments: unavailable (component/name count mismatch)" in result.output
    assert not (tmp_path / "stage.json").exists()
    assert not (tmp_path / "output").exists()


def test_extract_success(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "output"

    result = runner.invoke(
        app,
        ["extract", str(image), "--config", str(config), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert "extracted 2 frame(s)" in result.output
    assert (output / "stage.json").is_file()


def test_scale_and_pixelize_help_and_success(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(config, pipeline_config())
    extracted = tmp_path / "extracted"
    scaled = tmp_path / "scaled"
    pixelized = tmp_path / "pixelized"
    assert runner.invoke(app, ["scale", "--help"]).exit_code == 0
    assert runner.invoke(app, ["pixelize", "--help"]).exit_code == 0
    assert (
        runner.invoke(
            app, ["extract", str(image), "--config", str(config), "--output", str(extracted)]
        ).exit_code
        == 0
    )
    scale_result = runner.invoke(
        app, ["scale", str(extracted), "--config", str(config), "--output", str(scaled)]
    )
    pixel_result = runner.invoke(
        app,
        ["pixelize", str(scaled), "--config", str(config), "--output", str(pixelized)],
    )
    assert scale_result.exit_code == 0
    assert "scaled 2 frame(s)" in scale_result.output
    assert pixel_result.exit_code == 0
    assert "pixelized 2 frame(s)" in pixel_result.output


def test_pixelize_wrong_prior_stage_has_unsupported_exit(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(config, pipeline_config())
    extracted = tmp_path / "extracted"
    runner.invoke(app, ["extract", str(image), "--config", str(config), "--output", str(extracted)])
    result = runner.invoke(
        app,
        [
            "pixelize",
            str(extracted),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "pixelized"),
        ],
    )
    assert result.exit_code == 3
    assert "PX_STAGE_003" in result.output
    assert "Traceback" not in result.output


def test_actual_subprocess_reports_scale_geometry_incoherence_before_publication(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    extracted = tmp_path / "extracted"
    scaled = tmp_path / "scaled"
    output = tmp_path / "pixelized"
    write_rgba(image, transparent_sheet())
    write_config(config, pipeline_config())
    commands: tuple[tuple[str | Path, ...], ...] = (
        ("extract", image, "--config", config, "--output", extracted),
        ("scale", extracted, "--config", config, "--output", scaled),
    )
    for command in commands:
        result = subprocess.run(
            [_console_script(), *command],
            capture_output=True,
            check=False,
            timeout=_CLI_SUBPROCESS_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stderr
    stage_path = scaled / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["outputDimensions"]["width"] = 4
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = subprocess.run(
        [
            _console_script(),
            "pixelize",
            scaled,
            "--config",
            config,
            "--output",
            output,
        ],
        capture_output=True,
        check=False,
        timeout=_CLI_SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == 3
    assert result.stdout == b""
    assert result.stderr == (
        b'PX_STAGE_009 [load] scale frame "idle" output dimensions 4x3 '
        b"do not match declared scale geometry 3x3.\n"
    )
    assert b"Traceback" not in result.stderr
    assert not output.exists()
    assert list(tmp_path.glob(".pixelized.pixipix-build-*")) == []
    assert list(tmp_path.glob(".pixelized.pixipix-backup-*")) == []


def test_extreme_finite_scale_factor_is_a_processing_failure(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(
        config,
        pipeline_config(scale='mode = "explicit-factor"\nfactor = 1e308'),
    )
    extracted = tmp_path / "extracted"
    assert (
        runner.invoke(
            app, ["extract", str(image), "--config", str(config), "--output", str(extracted)]
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app,
        [
            "scale",
            str(extracted),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "scaled"),
        ],
    )

    assert result.exit_code == 1
    assert "PX_SCALE_002" in result.output
    assert "Traceback" not in result.output


def test_configuration_failure_exit_code(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    config = tmp_path / "invalid.toml"
    write_rgba(image, np.zeros((1, 1, 4), dtype=np.uint8))
    config.write_text("[unknown]\nvalue = 1\n", encoding="utf-8")

    result = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert result.exit_code == 2
    assert "PX_CONFIG_003" in result.output
    assert "Traceback" not in result.output


def test_px_input_001_unsupported_input_exit_code(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    image = tmp_path / "source.jpg"
    write_config(config, extraction_config(names=("one",), expected=1))
    image.write_bytes(b"not-an-image")

    result = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert result.exit_code == 3
    assert re.findall(r"\bPX_INPUT_\d{3}\b", result.stderr) == ["PX_INPUT_001"]


def test_actual_cli_reports_px_input_004_for_decoder_safety_limit(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    source = tmp_path / "decoder-bomb.png"
    output = tmp_path / "output"
    write_config(config)
    _write_decompression_bomb_png(source)

    console_script = _console_script().resolve()
    expected_console_script = (
        Path(sys.executable).with_name("pixipix.exe" if os.name == "nt" else "pixipix").resolve()
    )
    assert console_script == expected_console_script
    assert console_script.is_file()
    if os.name != "nt":
        assert os.access(console_script, os.X_OK)

    result = subprocess.run(
        [
            console_script,
            "extract",
            source,
            "--config",
            config,
            "--output",
            output,
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )

    stderr_text = result.stderr.decode("utf-8")
    stderr_lines = stderr_text.splitlines()

    assert result.returncode == 3
    assert result.stdout == b""
    assert stderr_text.endswith("\n")
    assert stderr_text.count("\n") == 1
    assert len(stderr_lines) == 1
    error_line = stderr_lines[0]
    assert error_line
    assert re.findall(r"\bPX_INPUT_\d{3}\b", error_line) == ["PX_INPUT_004"]
    assert "PX_INPUT_004 [load]" in error_line
    assert "source dimensions exceed decoder safety limits" in error_line
    assert "reduce the image size within the fixed safety ceiling" in error_line
    assert "Traceback" not in error_line
    assert "UnsupportedInputError" not in error_line

    repository_root = Path(__file__).resolve().parents[2]
    canonical_module_paths = (
        repository_root / "src" / "pixipix" / "cli.py",
        repository_root / "src" / "pixipix" / "imageio.py",
    )
    assert str(repository_root) not in error_line
    assert str(tmp_path.resolve()) not in error_line
    assert "src/pixipix" not in error_line
    assert "src\\pixipix" not in error_line
    assert "tests/" not in error_line
    assert "tests\\" not in error_line
    assert all(
        str(module_path.resolve()) not in error_line for module_path in canonical_module_paths
    )
    assert re.search(r'\bFile "[^"]+", line \d+', error_line) is None
    assert re.search(r"(?<![\w.])/(?:[^/\s]+/)*[^/\s]+", error_line) is None
    assert re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]", error_line) is None

    assert b"extracted " not in result.stdout
    assert "extracted " not in error_line
    assert not output.exists()
    assert not tuple(tmp_path.glob(f".{output.name}.pixipix-*"))


def test_processing_failure_exit_code(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(config, extraction_config(names=("one",), expected=None))

    result = runner.invoke(
        app,
        [
            "extract",
            str(image),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 1
    assert "PX_EXTRACT_003" in result.output


def test_unexpected_internal_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, config = _project(tmp_path)

    def explode(_image: Path, _config: object) -> NoReturn:
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(cli_module, "inspect_source", explode)
    result = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert result.exit_code == 4
    assert "PX_INTERNAL_001" in result.output
    assert "sensitive internal detail" not in result.output
    assert "Traceback" not in result.output


def test_actual_subprocess_maps_configuration_error_without_traceback(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    config = tmp_path / "invalid.toml"
    write_rgba(image, np.zeros((1, 1, 4), dtype=np.uint8))
    config.write_text("[unknown]\nvalue = 1\n", encoding="utf-8")

    result = subprocess.run(
        [_console_script(), "inspect", image, "--config", config],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "PX_CONFIG_003" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_resource_policy_precedes_missing_stage_input(tmp_path: Path) -> None:
    config = tmp_path / "invalid-resource.toml"
    output = tmp_path / "scaled"
    write_config(
        config,
        pipeline_config() + "\n[resources]\nmax_aggregate_input_pixels = 150000001\n",
    )

    result = subprocess.run(
        [
            _console_script(),
            "scale",
            tmp_path / "missing-stage",
            "--config",
            config,
            "--output",
            output,
        ],
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == (
        b'PX_CONFIG_037 [config] "resources.max_aggregate_input_pixels" value '
        b"150000001 exceeds the maximum allowed value 150000000. Remediation: use "
        b"a positive integer no greater than 150000000\n"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("scenario", "command", "expected"),
    [
        (
            resource_scenario_e,
            "pixelize",
            (
                b"PX_RESOURCE_001 [pixelize] aggregate resource policy exceeded: "
                b"aggregate input pixels 67043344/50000000. Remediation: reduce frame "
                b"count or dimensions, adjust transformation or canvas settings, or raise "
                b"the configured budget within its allowed cap when the execution "
                b"environment can support it\n"
            ),
        ),
        (
            resource_scenario_f,
            "scale",
            (
                b"PX_RESOURCE_001 [scale] aggregate resource policy exceeded: aggregate "
                b"output pixels 60000001/60000000. Remediation: reduce frame count or "
                b"dimensions, adjust transformation or canvas settings, or raise the "
                b"configured budget within its allowed cap when the execution environment "
                b"can support it\n"
            ),
        ),
        (
            resource_scenario_h,
            "scale",
            (
                b"PX_RESOURCE_001 [scale] aggregate resource policy exceeded: aggregate "
                b"input pixels 100000000/50000000; aggregate output pixels "
                b"144000000/120000000; modeled peak live bytes under the explicit-buffer "
                b"model 1076640000/1000000000. Remediation: reduce frame count or "
                b"dimensions, adjust transformation or canvas settings, or raise the "
                b"configured budget within its allowed cap when the execution environment "
                b"can support it\n"
            ),
        ),
    ],
)
def test_resource_refusal_cli_bytes_are_exact(
    tmp_path: Path,
    scenario: object,
    command: str,
    expected: bytes,
) -> None:
    build_scenario = scenario
    assert callable(build_scenario)
    config, input_root, output = build_scenario(tmp_path)

    result = subprocess.run(
        [_console_script(), command, input_root, "--config", config, "--output", output],
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == b""
    assert result.stderr == expected
    assert result.stderr.endswith(b"\n") and not result.stderr.endswith(b"\n\n")
    assert b"warning" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("scenario", [resource_scenario_f, resource_scenario_h])
def test_equivalent_resource_refusals_are_byte_deterministic(
    tmp_path: Path,
    scenario: object,
) -> None:
    results: list[subprocess.CompletedProcess[bytes]] = []
    outputs: list[Path] = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        build_scenario = scenario
        assert callable(build_scenario)
        config, input_root, output = build_scenario(root)
        results.append(
            subprocess.run(
                [
                    _console_script(),
                    "scale",
                    input_root,
                    "--config",
                    config,
                    "--output",
                    output,
                ],
                capture_output=True,
                check=False,
            )
        )
        outputs.append(output)

    assert [result.returncode for result in results] == [1, 1]
    assert [result.stdout for result in results] == [b"", b""]
    assert results[0].stderr == results[1].stderr
    assert not any(output.exists() for output in outputs)
