"""Smoke-test an installed PixiPix wheel outside the working project."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

SMOKE_STAGES = ("inspect", "extract", "scale", "pixelize", "align")
type SmokeStage = Literal["inspect", "extract", "scale", "pixelize", "align"]
type WriteStage = Literal["extract", "scale", "pixelize", "align"]


@dataclass(frozen=True, slots=True)
class FixtureContract:
    """Static facts locked in ``robot-geometric.SCENARIO.md``."""

    candidate_count: int = 2
    accepted_count: int = 2
    rejected_count: int = 0
    frame_names: tuple[str, ...] = ("idle", "signal")
    frame_paths: tuple[str, ...] = ("frames/idle.png", "frames/signal.png")
    final_stage: str = "align"
    schema_version: int = 1
    canvas_width: int = 4
    canvas_height: int = 4
    png_mode: str = "RGBA"
    warning_codes: tuple[str, ...] = ()


FIXTURE_CONTRACT = FixtureContract()


class SmokeFailure(RuntimeError):
    """Raised when installed-distribution proof or validation fails."""


def _render_command(command: Sequence[str | Path]) -> list[str]:
    return [str(part) for part in command]


def _process_details(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _run_setup(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = _render_command(command)
    result = subprocess.run(
        rendered,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SmokeFailure(
            f"distribution smoke setup command failed ({result.returncode}): "
            f"{' '.join(rendered)}\n{_process_details(result)}"
        )
    return result


def _run_stage(
    stage: SmokeStage, command: Sequence[str | Path], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    rendered = _render_command(command)
    result = subprocess.run(rendered, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SmokeFailure(
            f"distribution smoke failed during {stage} ({result.returncode}): "
            f"{' '.join(rendered)}\n{_process_details(result)}"
        )
    print(f"distribution smoke completed stage {stage}")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _validate_installed_location(
    *, interpreter: Path, console: Path, module_path: Path, environment: Path, repository: Path
) -> None:
    interpreter = interpreter.absolute()
    console = console.absolute()
    module_path = module_path.resolve()
    environment = environment.resolve()
    repository = repository.resolve()
    source_tree = repository / "src"
    if not _inside(interpreter, environment):
        raise SmokeFailure("isolated interpreter path is outside the isolated environment")
    if not _inside(console, environment):
        raise SmokeFailure("isolated console-script path is outside the isolated environment")
    if _inside(module_path, repository) or _inside(module_path, source_tree):
        raise SmokeFailure("installed pixipix module resolved inside the repository checkout")
    if not _inside(module_path, environment):
        raise SmokeFailure("installed pixipix module path is outside the isolated environment")


def _prove_installed_module(environment: Path, repository: Path, console: Path) -> Path:
    if "PYTHONPATH" in os.environ:
        raise SmokeFailure("repository-derived PYTHONPATH must be absent during installed smoke")
    module = importlib.import_module("pixipix")
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise SmokeFailure("installed pixipix module did not expose __file__")
    module_path = Path(raw_path).resolve()
    interpreter = Path(sys.executable).absolute()
    environment = environment.resolve()
    active_prefix = Path(sys.prefix).resolve()
    if active_prefix != environment:
        raise SmokeFailure("isolated interpreter sys.prefix does not match the smoke environment")
    _validate_installed_location(
        interpreter=interpreter,
        console=console,
        module_path=module_path,
        environment=environment,
        repository=repository,
    )
    print(f"isolated interpreter: {interpreter}")
    print(f"isolated environment prefix: {active_prefix}")
    print(f"isolated console script: {console.absolute()}")
    print(f"resolved pixipix.__file__: {module_path}")
    print("installed module inside isolated environment: true")
    print("installed module outside repository checkout: true")
    return module_path


def _validate_inspection(stdout: str) -> None:
    expected_lines = {
        f"candidate components: {FIXTURE_CONTRACT.candidate_count}",
        f"accepted components: {FIXTURE_CONTRACT.accepted_count}",
        f"rejected components: {FIXTURE_CONTRACT.rejected_count}",
        "frame assignments: 0=idle, 1=signal",
        "configured source cell size: 4",
    }
    actual_lines = set(stdout.splitlines())
    missing = sorted(expected_lines - actual_lines)
    if missing:
        raise SmokeFailure(f"installed fixture inspection is missing expected fact: {missing[0]}")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeFailure(f"{label} is missing or invalid JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise SmokeFailure(f"{label} must contain a JSON object")
    return cast(dict[str, object], value)


def _validate_stage_publication(stage: WriteStage, output: Path) -> None:
    from pixipix.stages.io import _valid_owned_output, load_stage_input

    if not _valid_owned_output(output, stage):
        raise SmokeFailure(f"{stage} stage failed installed production publication validation")
    if stage == "extract":
        loaded = load_stage_input(output, "extract")
    elif stage == "scale":
        loaded = load_stage_input(output, "scale")
    elif stage == "pixelize":
        loaded = load_stage_input(output, "pixelize")
    else:
        loaded = None
    if loaded is not None:
        names = tuple(frame.name for frame in loaded.frames)
        if names != FIXTURE_CONTRACT.frame_names:
            raise SmokeFailure(f"{stage} stage frame identities or order do not match contract")
        warning_codes = tuple(warning.code for warning in loaded.warnings)
        if warning_codes != FIXTURE_CONTRACT.warning_codes:
            raise SmokeFailure(f"{stage} stage warnings do not match contract")
    print(f"installed production publication validation passed for {stage}")


def _validate_final_output(output: Path) -> None:
    from PIL import Image, UnidentifiedImageError

    marker = _load_json_object(output / ".pixipix-output", "aligned ownership marker")
    if marker != {"owner": "pixipix", "schemaVersion": 1, "stage": "align"}:
        raise SmokeFailure("aligned ownership marker does not match the publication contract")
    metadata = _load_json_object(output / "stage.json", "aligned stage metadata")
    expected_metadata = {
        "schemaVersion": FIXTURE_CONTRACT.schema_version,
        "stage": FIXTURE_CONTRACT.final_stage,
        "status": "successful",
        "canvasWidth": FIXTURE_CONTRACT.canvas_width,
        "canvasHeight": FIXTURE_CONTRACT.canvas_height,
        "warnings": [],
        "clippingFindings": [],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise SmokeFailure(
                f"aligned stage metadata field {key!r} is {metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    raw_frames = metadata.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != len(FIXTURE_CONTRACT.frame_names):
        raise SmokeFailure("aligned stage metadata has an unexpected final frame count")
    declared_paths: list[str] = []
    for order, (raw_frame, expected_name, expected_path) in enumerate(
        zip(
            raw_frames,
            FIXTURE_CONTRACT.frame_names,
            FIXTURE_CONTRACT.frame_paths,
            strict=True,
        )
    ):
        if not isinstance(raw_frame, dict):
            raise SmokeFailure("aligned stage metadata frame entry must be an object")
        frame = cast(dict[str, object], raw_frame)
        expected_fields = {
            "name": expected_name,
            "sourceOrder": order,
            "relativePath": expected_path,
            "outputWidth": FIXTURE_CONTRACT.canvas_width,
            "outputHeight": FIXTURE_CONTRACT.canvas_height,
            "clipped": False,
        }
        for key, expected in expected_fields.items():
            if frame.get(key) != expected:
                raise SmokeFailure(
                    f"aligned frame {expected_name!r} field {key!r} does not match contract"
                )
        declared_paths.append(expected_path)
        frame_path = output / expected_path
        if not frame_path.is_file() or frame_path.is_symlink():
            raise SmokeFailure(f"expected aligned frame is missing or unsafe: {expected_path}")
        try:
            with Image.open(frame_path) as image:
                image.load()
                if image.format != "PNG":
                    raise SmokeFailure(f"aligned frame is not PNG: {expected_path}")
                if image.mode != FIXTURE_CONTRACT.png_mode:
                    raise SmokeFailure(
                        f"aligned frame mode does not match contract: {expected_path}"
                    )
                if image.size != (
                    FIXTURE_CONTRACT.canvas_width,
                    FIXTURE_CONTRACT.canvas_height,
                ):
                    raise SmokeFailure(
                        f"aligned frame dimensions do not match contract: {expected_path}"
                    )
        except SmokeFailure:
            raise
        except (UnidentifiedImageError, OSError) as error:
            raise SmokeFailure(f"unable to decode aligned PNG: {expected_path}") from error
    if tuple(declared_paths) != FIXTURE_CONTRACT.frame_paths:
        raise SmokeFailure("aligned frame paths or deterministic order do not match contract")
    frames_root = output / "frames"
    actual_paths = tuple(
        sorted(path.relative_to(output).as_posix() for path in frames_root.iterdir())
    )
    if actual_paths != tuple(sorted(FIXTURE_CONTRACT.frame_paths)):
        raise SmokeFailure("aligned frame directory contains missing or undeclared files")
    root_entries = {path.name for path in output.iterdir()}
    if root_entries != {".pixipix-output", "frames", "stage.json"}:
        raise SmokeFailure("aligned stage directory contains undeclared artifacts")
    print("final aligned metadata and PNG validation passed")


def _stage_tree_manifest(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _stage_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, byte_length, sha256 in _stage_tree_manifest(root):
        digest.update(f"{relative}\0{byte_length}\0{sha256}\n".encode())
    return digest.hexdigest()


def _validate_installed_run(run_root: Path, manual_outputs: Mapping[str, Path]) -> None:
    marker = _load_json_object(run_root / ".pixipix-run", "run ownership marker")
    if marker != {"kind": "run", "owner": "pixipix", "schemaVersion": 1}:
        raise SmokeFailure("run ownership marker does not match the publication contract")
    if {path.name for path in run_root.iterdir()} != {
        ".pixipix-run",
        "extract",
        "scale",
        "pixelize",
        "align",
    }:
        raise SmokeFailure("run root does not contain exactly the authoritative stage set")
    for stage in ("extract", "scale", "pixelize", "align"):
        run_stage = run_root / stage
        manual_stage = manual_outputs[stage]
        _validate_stage_publication(stage, run_stage)
        if _stage_tree_manifest(run_stage) != _stage_tree_manifest(manual_stage):
            raise SmokeFailure(f"installed RUN {stage} tree differs from the manual sequence")
        if _stage_tree_sha256(run_stage) != _stage_tree_sha256(manual_stage):
            raise SmokeFailure(f"installed RUN {stage} tree hash differs from manual execution")
    _validate_final_output(run_root / "align")
    print("installed RUN stage-tree parity validation passed")


def _validate_installed_warning_visibility(
    *, console: Path, image: Path, config: Path, working_directory: Path
) -> None:
    warning_config = working_directory / "warning.toml"
    warning_config.write_bytes(
        config.read_bytes() + b"\n[frame_overrides.signal]\nscale_multiplier = 1.0\n"
    )
    output_root = working_directory / "warning-output"
    extracted = output_root / "extracted"
    scaled = output_root / "scaled"
    extract_result = _run_setup(
        [console, "extract", image, "--config", warning_config, "--output", extracted],
        cwd=working_directory,
    )
    scale_result = _run_setup(
        [console, "scale", extracted, "--config", warning_config, "--output", scaled],
        cwd=working_directory,
    )
    expected_warning = (
        'pixipix: warning [scale] PX_SCALE_OVERRIDE_001: frame "signal" uses explicit '
        "scale multiplier 1.0; cross-frame consistency is user-managed\n"
    )
    if extract_result.stderr != "":
        raise SmokeFailure("installed warning fixture extraction wrote unexpected stderr")
    if scale_result.stdout != f"scaled 2 frame(s) to {scaled}\n":
        raise SmokeFailure("installed warning fixture scale stdout does not match contract")
    if scale_result.stderr != expected_warning:
        raise SmokeFailure("installed warning fixture scale stderr does not match contract")
    metadata = _load_json_object(scaled / "stage.json", "warning fixture scale metadata")
    if metadata.get("warnings") != [
        {
            "code": "PX_SCALE_OVERRIDE_001",
            "message": (
                'frame "signal" uses explicit scale multiplier 1.0; '
                "cross-frame consistency is user-managed"
            ),
            "stage": "scale",
        }
    ]:
        raise SmokeFailure("installed warning fixture metadata does not match contract")
    print("installed CLI warning visibility validation passed")


def _validate_installed_resource_identity(config: Path, working_directory: Path) -> Path:
    from pixipix.config import load_config
    from pixipix.resources import ResourcePolicy

    explicit = working_directory / "explicit-resources.toml"
    explicit.write_bytes(
        config.read_bytes()
        + (
            b"\n[resources]\n"
            b"max_aggregate_input_pixels = 50000000\n"
            b"max_aggregate_output_pixels = 60000000\n"
            b"max_modeled_peak_live_bytes = 1000000000\n"
        )
    )
    omitted_loaded = load_config(config)
    explicit_loaded = load_config(explicit)
    if omitted_loaded.config.resources != ResourcePolicy():
        raise SmokeFailure("omitted installed resource policy does not resolve to defaults")
    if explicit_loaded.config.resources != ResourcePolicy():
        raise SmokeFailure("explicit installed resource policy does not resolve to defaults")
    if omitted_loaded.source_config_sha256 == explicit_loaded.source_config_sha256:
        raise SmokeFailure(
            "omitted and explicit resource defaults unexpectedly share source identity"
        )
    if omitted_loaded.effective_config_sha256 != explicit_loaded.effective_config_sha256:
        raise SmokeFailure("omitted and explicit resource defaults differ in effective identity")
    print("installed resource default identity validation passed")
    return explicit


def _write_resource_refusal_fixture(working_directory: Path) -> tuple[Path, Path, Path]:
    from pixipix import __version__
    from pixipix.config import load_config

    config = working_directory / "resource-refusal.toml"
    config.write_text(
        (
            "[project]\n"
            'name = "installed-resource-refusal"\n'
            "strict = true\n\n"
            "[resources]\n"
            "max_aggregate_input_pixels = 50000000\n"
            "max_aggregate_output_pixels = 60000000\n"
            "max_modeled_peak_live_bytes = 1000000000\n\n"
            "[source]\n"
            'format = "png"\n'
            "expected_components = 1\n"
            "max_width = 4096\n"
            "max_height = 4096\n"
            "max_pixels = 16777216\n"
            "max_components = 1\n\n"
            "[background]\n"
            'mode = "alpha"\n'
            "alpha_threshold = 8\n\n"
            "[extract]\n"
            "connectivity = 8\n"
            "minimum_area = 1\n"
            "padding = 0\n"
            "row_tolerance = 0\n\n"
            "[frames]\n"
            'names = ["ceiling"]\n\n'
            "[scale]\n"
            'mode = "explicit-factor"\n'
            "factor = 1.0\n\n"
            "[pixelize]\n"
            "source_cell_size = 4\n"
            'representative = "alpha-weighted-majority"\n'
            'alpha_policy = "binary"\n'
            "alpha_threshold = 128\n"
            'remainder_policy = "pad-transparent"\n'
        ),
        encoding="utf-8",
    )
    loaded = load_config(config)
    input_root = working_directory / "resource-refusal-extract"
    frames_root = input_root / "frames"
    frames_root.mkdir(parents=True)
    (input_root / ".pixipix-output").write_text(
        json.dumps({"owner": "pixipix", "schemaVersion": 1, "stage": "extract"}),
        encoding="utf-8",
    )
    (frames_root / "ceiling.png").write_bytes(b"decoder sentinel: not a PNG")
    metadata = {
        "schemaVersion": 1,
        "pixipixVersion": __version__,
        "stage": "extract",
        "status": "successful",
        "sourceConfigSha256": loaded.source_config_sha256,
        "effectiveConfigSha256": loaded.effective_config_sha256,
        "frames": [
            {
                "name": "ceiling",
                "relativePath": "frames/ceiling.png",
                "sourceOrder": 0,
                "paddedBounds": {
                    "left": 0,
                    "top": 0,
                    "right": 4096,
                    "bottom": 4096,
                },
            }
        ],
        "warnings": [],
    }
    (input_root / "stage.json").write_text(json.dumps(metadata), encoding="utf-8")
    output = working_directory / "resource-refusal-output"
    return config, input_root, output


def _validate_installed_resource_refusal(*, console: Path, working_directory: Path) -> None:
    config, input_root, output = _write_resource_refusal_fixture(working_directory)
    result = subprocess.run(
        _render_command([console, "scale", input_root, "--config", config, "--output", output]),
        cwd=working_directory,
        capture_output=True,
        check=False,
    )
    expected_stderr = (
        b"PX_RESOURCE_001 [scale] aggregate resource policy exceeded: modeled peak "
        b"live bytes under the explicit-buffer model 1409286144/1000000000. "
        b"Remediation: reduce frame count or dimensions, adjust transformation or "
        b"canvas settings, or raise the configured budget within its allowed cap when "
        b"the execution environment can support it\n"
    )
    if result.returncode != 1:
        raise SmokeFailure(f"installed resource refusal exited {result.returncode}, expected 1")
    if result.stdout != b"":
        raise SmokeFailure("installed resource refusal wrote unexpected stdout")
    if result.stderr != expected_stderr:
        raise SmokeFailure("installed resource refusal stderr does not match contract")
    if output.exists():
        raise SmokeFailure("installed resource refusal unexpectedly published output")
    print("installed metadata-only resource refusal validation passed")


def _stage_command(
    stage: SmokeStage,
    *,
    console: Path,
    image: Path,
    config: Path,
    outputs: Mapping[str, Path],
) -> list[str | Path]:
    if stage == "inspect":
        return [console, stage, image, "--config", config]
    if stage == "extract":
        source = image
    else:
        prior = SMOKE_STAGES[SMOKE_STAGES.index(stage) - 1]
        source = outputs[prior]
    return [console, stage, source, "--config", config, "--output", outputs[stage]]


def _run_installed_pipeline(
    *, repository: Path, fixture_dir: Path, environment: Path, console: Path
) -> int:
    repository = repository.resolve()
    working_directory = Path.cwd().resolve()
    if _inside(working_directory, repository):
        raise SmokeFailure("installed smoke working directory must be outside the repository")
    _prove_installed_module(environment, repository, console)
    from pixipix.pipeline.run import run_pipeline

    if run_pipeline.__module__ != "pixipix.pipeline.run":
        raise SmokeFailure("installed Python RUN API does not resolve to its authoritative owner")
    expected_version = importlib.metadata.version("pixipix")

    help_result = _run_setup([console, "--help"], cwd=working_directory)
    if "Tiny poses in. Tidy pixels out." not in help_result.stdout:
        raise SmokeFailure("console help output is missing the PixiPix product statement")
    run_help_result = _run_setup([console, "run", "--help"], cwd=working_directory)
    if "Extract" not in run_help_result.stdout or "Align" not in run_help_result.stdout:
        raise SmokeFailure("installed RUN help is missing the complete stage order")
    version_result = _run_setup([console, "version"], cwd=working_directory)
    if version_result.stdout.strip() != f"PixiPix {expected_version}":
        raise SmokeFailure("console version does not match installed distribution metadata")
    module_result = _run_setup([sys.executable, "-m", "pixipix"], cwd=working_directory)
    if "Tiny poses in. Tidy pixels out." not in module_result.stdout:
        raise SmokeFailure(
            "python -m pixipix is not equivalent to the installed console entry point"
        )

    image = fixture_dir / "robot-geometric.png"
    config = fixture_dir / "robot-geometric.toml"
    if not image.is_file() or not config.is_file():
        raise SmokeFailure("copied robot smoke-test fixture is incomplete")
    config = _validate_installed_resource_identity(config, working_directory)
    _validate_installed_resource_refusal(
        console=console,
        working_directory=working_directory,
    )
    output_root = working_directory / "smoke-output"
    outputs = {
        "extract": output_root / "extracted",
        "scale": output_root / "scaled",
        "pixelize": output_root / "pixelized",
        "align": output_root / "aligned",
    }
    for raw_stage in SMOKE_STAGES:
        stage = cast(SmokeStage, raw_stage)
        result = _run_stage(
            stage,
            _stage_command(
                stage,
                console=console,
                image=image,
                config=config,
                outputs=outputs,
            ),
            cwd=working_directory,
        )
        if stage == "inspect":
            _validate_inspection(result.stdout)
        else:
            _validate_stage_publication(stage, outputs[stage])
    _validate_final_output(outputs["align"])
    run_root = working_directory / "installed-run"
    run_result = _run_setup(
        [console, "run", image, "--config", config, "--output", run_root],
        cwd=working_directory,
    )
    if run_result.stdout != f"completed run with 2 frame(s) at {run_root}\n":
        raise SmokeFailure("installed RUN success output does not match the CLI contract")
    if run_result.stderr:
        raise SmokeFailure("installed canonical RUN wrote unexpected warnings")
    print("distribution smoke completed operation run")
    _validate_installed_run(run_root, outputs)
    _validate_installed_warning_visibility(
        console=console,
        image=image,
        config=config,
        working_directory=working_directory,
    )
    print(f"distribution smoke test passed for pixipix {expected_version}")
    return 0


def _isolated_paths(environment: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        scripts = environment / "Scripts"
        return scripts / "python.exe", scripts / "pixipix.exe"
    binaries = environment / "bin"
    return binaries / "python", binaries / "pixipix"


def _sanitized_environment(environment: Path) -> dict[str, str]:
    sanitized = dict(os.environ)
    sanitized.pop("PYTHONPATH", None)
    sanitized["PYTHONNOUSERSITE"] = "1"
    sanitized["VIRTUAL_ENV"] = str(environment)
    return sanitized


def _run_artifact_smoke(artifact: Path) -> int:
    repository = Path(__file__).resolve().parents[1]
    artifact = artifact.resolve()
    if not artifact.is_file() or artifact.suffix != ".whl":
        raise SmokeFailure(f"smoke artifact must be one existing wheel: {artifact}")
    uv = shutil.which("uv")
    if uv is None:
        raise SmokeFailure("uv is required to create the isolated smoke environment")
    with tempfile.TemporaryDirectory(prefix="pixipix-distribution-smoke-") as temporary:
        root = Path(temporary).resolve()
        if _inside(root, repository.resolve()):
            raise SmokeFailure("isolated smoke root must be outside the repository")
        environment = root / "venv"
        working_directory = root / "work"
        fixture_dir = root / "fixture"
        working_directory.mkdir()
        fixture_dir.mkdir()
        for filename in ("robot-geometric.png", "robot-geometric.toml"):
            shutil.copy2(repository / "tests" / "fixtures" / filename, fixture_dir / filename)
        sanitized = _sanitized_environment(environment)
        _run_setup(
            [uv, "venv", "--python", sys.executable, environment],
            cwd=root,
            environment=sanitized,
        )
        interpreter, console = _isolated_paths(environment)
        _run_setup(
            [uv, "pip", "install", "--python", interpreter, artifact],
            cwd=root,
            environment=sanitized,
        )
        if not interpreter.is_file() or not console.is_file():
            raise SmokeFailure(
                "isolated artifact installation did not create required entry points"
            )
        command: list[str | Path] = [
            interpreter,
            Path(__file__).resolve(),
            "--installed-run",
            "--repository",
            repository,
            "--fixture-dir",
            fixture_dir,
            "--environment",
            environment,
            "--console",
            console,
        ]
        result = subprocess.run(
            _render_command(command),
            cwd=working_directory,
            env=sanitized,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
            )
        return result.returncode


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--installed-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixture-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--environment", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--console", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create an isolated install or execute the installed validation payload."""

    args = _command_parser().parse_args(argv)
    try:
        if args.installed_run:
            required = (args.repository, args.fixture_dir, args.environment, args.console)
            if any(value is None for value in required):
                raise SmokeFailure("installed smoke invocation is missing internal path arguments")
            return _run_installed_pipeline(
                repository=args.repository,
                fixture_dir=args.fixture_dir,
                environment=args.environment,
                console=args.console,
            )
        if args.artifact is None:
            raise SmokeFailure("smoke verification requires --artifact WHEEL")
        return _run_artifact_smoke(args.artifact)
    except SmokeFailure as error:
        print(f"distribution smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
