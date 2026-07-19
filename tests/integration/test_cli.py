from __future__ import annotations

import subprocess
import sys
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
    transparent_sheet,
    write_config,
    write_rgba,
)

runner = CliRunner()


def _console_script() -> Path:
    script = Path(sys.executable).with_name("pixipix")
    assert script.is_file()
    return script


def _project(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, transparent_sheet())
    write_config(config)
    return image, config


def test_help_and_version_commands() -> None:
    help_result = runner.invoke(app, ["--help"])
    module_equivalent = runner.invoke(app, [])
    version_result = runner.invoke(app, ["version"])

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


def test_unsupported_input_exit_code(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    image = tmp_path / "source.jpg"
    write_config(config, extraction_config(names=("one",), expected=1))
    image.write_bytes(b"not-an-image")

    result = runner.invoke(app, ["inspect", str(image), "--config", str(config)])

    assert result.exit_code == 3
    assert "PX_INPUT_001" in result.output


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
