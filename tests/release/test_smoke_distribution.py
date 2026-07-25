from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from scripts.smoke_distribution import (
    FIXTURE_CONTRACT,
    SMOKE_STAGES,
    SmokeFailure,
    _run_stage,
    _validate_final_output,
    _validate_installed_location,
    _validate_installed_resource_identity,
    _validate_installed_resource_refusal,
    _write_resource_refusal_fixture,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "smoke_distribution.py"


@dataclass(frozen=True, slots=True)
class BuiltArtifacts:
    direct_wheel: Path
    sdist: Path
    rebuilt_wheel: Path


def _run(command: list[str | Path], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    return subprocess.run(rendered, cwd=cwd, capture_output=True, text=True, check=False)


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


@pytest.fixture(scope="session")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> BuiltArtifacts:
    root = tmp_path_factory.mktemp("distribution-smoke-artifacts")
    direct = root / "direct"
    rebuilt = root / "rebuilt"
    direct.mkdir()
    rebuilt.mkdir()
    wheel_result = _run(
        ["uv", "build", "--wheel", "--no-sources", "--out-dir", direct],
        cwd=PROJECT_ROOT,
    )
    assert wheel_result.returncode == 0, wheel_result.stderr
    sdist_result = _run(
        ["uv", "build", "--sdist", "--no-sources", "--out-dir", direct],
        cwd=PROJECT_ROOT,
    )
    assert sdist_result.returncode == 0, sdist_result.stderr
    sdist = _single(direct, "*.tar.gz")
    rebuilt_result = _run(
        ["uv", "build", "--wheel", "--no-sources", "--out-dir", rebuilt, sdist],
        cwd=PROJECT_ROOT,
    )
    assert rebuilt_result.returncode == 0, rebuilt_result.stderr
    return BuiltArtifacts(
        direct_wheel=_single(direct, "*.whl"),
        sdist=sdist,
        rebuilt_wheel=_single(rebuilt, "*.whl"),
    )


def _run_smoke(wheel: Path) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, SMOKE_SCRIPT, "--artifact", wheel], cwd=PROJECT_ROOT)


def _write_final_stage(root: Path) -> None:
    frames = root / "frames"
    frames.mkdir(parents=True)
    (root / ".pixipix-output").write_text(
        json.dumps({"owner": "pixipix", "schemaVersion": 1, "stage": "align"}),
        encoding="utf-8",
    )
    metadata_frames: list[dict[str, object]] = []
    for order, (name, relative_path) in enumerate(
        zip(FIXTURE_CONTRACT.frame_names, FIXTURE_CONTRACT.frame_paths, strict=True)
    ):
        Image.new("RGBA", (4, 4)).save(root / relative_path, format="PNG")
        metadata_frames.append(
            {
                "name": name,
                "sourceOrder": order,
                "relativePath": relative_path,
                "outputWidth": 4,
                "outputHeight": 4,
                "clipped": False,
            }
        )
    metadata = {
        "schemaVersion": 1,
        "stage": "align",
        "status": "successful",
        "canvasWidth": 4,
        "canvasHeight": 4,
        "warnings": [],
        "clippingFindings": [],
        "frames": metadata_frames,
    }
    (root / "stage.json").write_text(json.dumps(metadata), encoding="utf-8")


def _record_hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _corrupt_align_implementation(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as wheel:
        infos = wheel.infolist()
        files = {info.filename: wheel.read(info) for info in infos}
    align_path = "pixipix/stages/align.py"
    original = files[align_path]
    needle = b"def compose_aligned_canvas("
    replacement = b"def corrupted_compose_aligned_canvas("
    assert original.count(needle) == 1
    files[align_path] = original.replace(needle, replacement, 1)

    record_path = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(files[record_path].decode("utf-8"))))
    for row in rows:
        if row[0] == align_path:
            row[1] = _record_hash(files[align_path])
            row[2] = str(len(files[align_path]))
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[record_path] = stream.getvalue().encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as wheel:
        for info in infos:
            wheel.writestr(info, files[info.filename])


def test_fixture_contract_is_static_and_configuration_reaches_align() -> None:
    scenario = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.SCENARIO.md"
    config_path = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert scenario.is_file()
    assert FIXTURE_CONTRACT.candidate_count == 2
    assert FIXTURE_CONTRACT.accepted_count == 2
    assert FIXTURE_CONTRACT.rejected_count == 0
    assert FIXTURE_CONTRACT.frame_names == ("idle", "signal")
    assert FIXTURE_CONTRACT.frame_paths == ("frames/idle.png", "frames/signal.png")
    assert FIXTURE_CONTRACT.final_stage == "align"
    assert FIXTURE_CONTRACT.schema_version == 1
    assert (FIXTURE_CONTRACT.canvas_width, FIXTURE_CONTRACT.canvas_height) == (4, 4)
    assert FIXTURE_CONTRACT.png_mode == "RGBA"
    assert FIXTURE_CONTRACT.warning_codes == ()
    assert config["scale"] == {"mode": "explicit-factor", "factor": 1.0}
    assert config["pixelize"]["source_cell_size"] == 4
    assert config["pixelize"]["remainder_policy"] == "pad-transparent"
    assert config["output"] == {
        "frame_width": 4,
        "frame_height": 4,
        "anchor": "center",
        "clip_policy": "error",
    }


def test_canonical_smoke_sequence_is_complete_and_ordered() -> None:
    assert SMOKE_STAGES == ("inspect", "extract", "scale", "pixelize", "align")


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_installed_artifact_runs_complete_pipeline(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)

    result = _run_smoke(wheel)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert [
        line.removeprefix("distribution smoke completed stage ")
        for line in result.stdout.splitlines()
        if line.startswith("distribution smoke completed stage ")
    ] == list(SMOKE_STAGES)
    assert "resolved pixipix.__file__:" in result.stdout
    assert "installed module inside isolated environment: true" in result.stdout
    assert "installed module outside repository checkout: true" in result.stdout
    assert "installed production publication validation passed for align" in result.stdout
    assert "final aligned metadata and PNG validation passed" in result.stdout
    assert "installed CLI warning visibility validation passed" in result.stdout
    assert "installed resource default identity validation passed" in result.stdout
    assert "installed metadata-only resource refusal validation passed" in result.stdout
    assert "distribution smoke test passed for pixipix" in result.stdout


def test_installed_resource_smoke_contracts_are_safe_and_exact(tmp_path: Path) -> None:
    config = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.toml"
    console = Path(sys.executable).with_name("pixipix")

    explicit = _validate_installed_resource_identity(config, tmp_path)
    _validate_installed_resource_refusal(
        console=console,
        working_directory=tmp_path,
    )

    assert explicit.name == "explicit-resources.toml"
    assert not (tmp_path / "resource-refusal-output").exists()
    assert (tmp_path / "resource-refusal-extract" / "frames" / "ceiling.png").stat().st_size < 64


def test_resource_refusal_fixture_uses_explicit_policy_a(tmp_path: Path) -> None:
    config, input_root, output = _write_resource_refusal_fixture(tmp_path)
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))

    assert parsed["resources"] == {
        "max_aggregate_input_pixels": 50_000_000,
        "max_aggregate_output_pixels": 60_000_000,
        "max_modeled_peak_live_bytes": 1_000_000_000,
    }
    assert parsed["frames"] == {"names": ["ceiling"]}
    assert (input_root / "frames" / "ceiling.png").stat().st_size < 64
    assert not output.exists()


def test_repository_source_resolution_fails_installed_location_proof(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    interpreter = environment / "bin" / "python"
    console = environment / "bin" / "pixipix"
    repository_module = PROJECT_ROOT / "src" / "pixipix" / "__init__.py"

    with pytest.raises(SmokeFailure, match="resolved inside the repository checkout"):
        _validate_installed_location(
            interpreter=interpreter,
            console=console,
            module_path=repository_module,
            environment=environment,
            repository=PROJECT_ROOT,
        )


def test_final_validation_rejects_missing_frame(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    (tmp_path / FIXTURE_CONTRACT.frame_paths[0]).unlink()

    with pytest.raises(SmokeFailure, match="expected aligned frame is missing"):
        _validate_final_output(tmp_path)


def test_final_validation_rejects_undeclared_frame(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    Image.new("RGBA", (4, 4)).save(tmp_path / "frames" / "extra.png", format="PNG")

    with pytest.raises(SmokeFailure, match="missing or undeclared files"):
        _validate_final_output(tmp_path)


def test_final_validation_rejects_invalid_metadata(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    (tmp_path / "stage.json").write_text("{", encoding="utf-8")

    with pytest.raises(SmokeFailure, match="aligned stage metadata is missing or invalid JSON"):
        _validate_final_output(tmp_path)


def test_failed_command_reports_exact_stage_and_process_output(tmp_path: Path) -> None:
    with pytest.raises(SmokeFailure, match="distribution smoke failed during pixelize") as captured:
        _run_stage(
            "pixelize",
            [sys.executable, "-c", "import sys; print('stage output'); sys.exit(7)"],
            cwd=tmp_path,
        )

    assert "stage output" in str(captured.value)
    assert "(7)" in str(captured.value)


def test_corrupted_installed_artifact_fails_at_align(
    tmp_path: Path, built_artifacts: BuiltArtifacts
) -> None:
    corrupted = tmp_path / built_artifacts.direct_wheel.name
    _corrupt_align_implementation(built_artifacts.direct_wheel, corrupted)

    result = _run_smoke(corrupted)
    combined = result.stdout + result.stderr

    assert result.returncode != 0
    assert "distribution smoke completed stage inspect" in result.stdout
    assert "distribution smoke completed stage extract" in result.stdout
    assert "distribution smoke completed stage scale" in result.stdout
    assert "distribution smoke completed stage pixelize" in result.stdout
    assert "distribution smoke completed stage align" not in result.stdout
    assert "distribution smoke failed during align" in combined
    assert "PX_INTERNAL_001" in combined
