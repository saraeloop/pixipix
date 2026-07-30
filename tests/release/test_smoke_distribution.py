from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
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
    _isolated_paths,
    _run_stage,
    _sanitized_environment,
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


def _run(
    command: list[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    return subprocess.run(
        rendered,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


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


def _validate_prior_stage_artifacts(output: Path, stage: str) -> None:
    assert json.loads((output / ".pixipix-output").read_text(encoding="utf-8")) == {
        "owner": "pixipix",
        "schemaVersion": 1,
        "stage": stage,
    }
    metadata = json.loads((output / "stage.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == stage
    assert metadata["status"] == "successful"
    assert tuple(frame["name"] for frame in metadata["frames"]) == FIXTURE_CONTRACT.frame_names
    assert (
        tuple(frame["relativePath"] for frame in metadata["frames"]) == FIXTURE_CONTRACT.frame_paths
    )
    for relative_path in FIXTURE_CONTRACT.frame_paths:
        with Image.open(output / relative_path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGBA"


def _run_corrupted_installed_pipeline(
    wheel: Path,
    root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = root / "venv"
    working_directory = root / "work"
    fixture_directory = root / "fixture"
    root.mkdir()
    working_directory.mkdir()
    fixture_directory.mkdir()
    for filename in ("robot-geometric.png", "robot-geometric.toml"):
        shutil.copy2(PROJECT_ROOT / "tests" / "fixtures" / filename, fixture_directory / filename)

    sanitized = _sanitized_environment(environment)
    sanitized.pop("PYTHONHOME", None)
    creation = _run(
        ["uv", "venv", "--python", sys.executable, environment],
        cwd=root,
        environment=sanitized,
    )
    assert creation.returncode == 0, creation.stderr
    interpreter, console = _isolated_paths(environment)
    installation = _run(
        ["uv", "pip", "install", "--python", interpreter, wheel],
        cwd=root,
        environment=sanitized,
    )
    assert installation.returncode == 0, installation.stderr

    imported = _run(
        [
            interpreter,
            "-c",
            (
                "import pathlib; import pixipix.stages.align as align; "
                "print(pathlib.Path(align.__file__).resolve())"
            ),
        ],
        cwd=working_directory,
        environment=sanitized,
    )
    assert imported.returncode == 0, imported.stderr
    assert str(environment.resolve()) in imported.stdout
    assert str(PROJECT_ROOT.resolve()) not in imported.stdout

    help_result = _run([console, "--help"], cwd=working_directory, environment=sanitized)
    assert help_result.returncode == 0, help_result.stderr
    assert "Tiny poses in. Tidy pixels out." in help_result.stdout

    image = fixture_directory / "robot-geometric.png"
    config = fixture_directory / "robot-geometric.toml"
    output_root = working_directory / "smoke-output"
    outputs = {
        "extract": output_root / "extracted",
        "scale": output_root / "scaled",
        "pixelize": output_root / "pixelized",
        "align": output_root / "aligned",
    }
    inspect_result = _run(
        [console, "inspect", image, "--config", config],
        cwd=working_directory,
        environment=sanitized,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr

    source = image
    for stage in ("extract", "scale", "pixelize"):
        output = outputs[stage]
        result = _run(
            [console, stage, source, "--config", config, "--output", output],
            cwd=working_directory,
            environment=sanitized,
        )
        assert result.returncode == 0, result.stderr
        _validate_prior_stage_artifacts(output, stage)
        source = output

    result = _run(
        [console, "align", source, "--config", config, "--output", outputs["align"]],
        cwd=working_directory,
        environment=sanitized,
    )
    assert not outputs["align"].exists()
    return result


def _corrupt_align_implementation(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as wheel:
        infos = wheel.infolist()
        files = {info.filename: wheel.read(info) for info in infos}
    align_path = "pixipix/stages/align/execution.py"
    original = files[align_path]
    needle = b"    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:\n"
    replacement = (
        b'    raise RuntimeError("simulated installed align execution corruption")\n' + needle
    )
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
def test_wheel_contains_exact_extract_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    stages_source = PROJECT_ROOT / "src" / "pixipix" / "stages" / "__init__.py"
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract"
    expected_members = {
        f"pixipix/stages/extract/{name}": source_root / name
        for name in (
            "__init__.py",
            "analysis.py",
            "api.py",
            "execution.py",
            "metadata.py",
            "planning.py",
            "publication.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert "pixipix/stages/__init__.py" in members
        assert {member for member in members if member.startswith("pixipix/stages/extract")} == set(
            expected_members
        )
        assert archive.read("pixipix/stages/__init__.py") == stages_source.read_bytes()
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/extract.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_extract_first_import_preserves_compatibility_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / f"{artifact_name}-extract-import"
    working_directory.mkdir()
    code = (
        "import inspect, pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix.stages.extract as extract; "
        "import pixipix.stages.extract.analysis as analysis; "
        "import pixipix.stages.extract.api as api; "
        "import pixipix.stages.extract.execution as execution; "
        "import pixipix.stages.extract.metadata as metadata; "
        "import pixipix.stages.extract.planning as planning; "
        "import pixipix.stages.extract.publication as publication; "
        "import pixipix.models as models; "
        "import pixipix.resources as resources; "
        "expected = ("
        "'ComponentMap', 'label_components', 'filter_components', 'order_components', "
        "'inspect_source', 'extract_source', 'project_extract_resources', "
        "'project_extracted_frames', 'publish_extraction'); "
        "assert all(hasattr(extract, name) for name in expected); "
        "assert pathlib.Path(extract.__file__).name == '__init__.py'; "
        "assert extract.__spec__.submodule_search_locations is not None; "
        "assert extract.ComponentMap is analysis.ComponentMap; "
        "assert extract.label_components is analysis.label_components; "
        "assert extract.filter_components is analysis.filter_components; "
        "assert extract.order_components is analysis.order_components; "
        "assert extract.inspect_source is api.inspect_source; "
        "assert extract.extract_source is api.extract_source; "
        "assert extract.project_extract_resources is planning.project_extract_resources; "
        "assert extract.project_extracted_frames is planning.project_extracted_frames; "
        "assert extract.publish_extraction is publication.publish_extraction; "
        "assert analysis.ComponentMap.__module__ == 'pixipix.stages.extract.analysis'; "
        "assert analysis._Analysis.__module__ == 'pixipix.stages.extract.analysis'; "
        "assert api.inspect_source.__module__ == 'pixipix.stages.extract.api'; "
        "assert api.extract_source.__module__ == 'pixipix.stages.extract.api'; "
        "assert planning.project_extract_resources.__module__ "
        "== 'pixipix.stages.extract.planning'; "
        "assert publication.publish_extraction.__module__ "
        "== 'pixipix.stages.extract.publication'; "
        "assert api.InspectionResult is models.InspectionResult; "
        "assert api.ExtractionRun is models.ExtractionRun; "
        "assert api.ExtractionResult is models.ExtractionResult; "
        "assert execution.ExtractedFrame is models.ExtractedFrame; "
        "assert execution.FrameImage is models.FrameImage; "
        "assert planning.ExtractedFrame is models.ExtractedFrame; "
        "assert planning.ResourceProjection is resources.ResourceProjection; "
        "assert callable(metadata._stage_metadata); "
        "assert callable(publication._valid_owned_output); "
        "assert not hasattr(extract, '_Analysis'); "
        "assert not hasattr(extract, '_analyze'); "
        "assert not hasattr(extract, '_padded_bounds'); "
        "assert not hasattr(extract, '_materialize_frame_crop'); "
        "assert not hasattr(extract, '_valid_frame_png'); "
        "assert not hasattr(extract, '_validate_staged_output'); "
        "assert not hasattr(extract, '_validate_output_location'); "
        "assert not hasattr(extract, 'np'); "
        "assert not hasattr(extract, 'Image'); "
        "assert not hasattr(extract, 'load_source'); "
        "assert not hasattr(extract, 'write_png'); "
        "assert str(inspect.signature(extract.publish_extraction)) == "
        "\"(input_path: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'ExtractionResult'\"; "
        "assert sys.modules['pixipix.stages.extract'] is extract; "
        "assert 'pixipix.stages.extract.__init__' not in sys.modules; "
        "print(pathlib.Path(extract.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_align_package_member_only(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    expected_align_members = {
        "pixipix/stages/align/__init__.py",
        "pixipix/stages/align/api.py",
        "pixipix/stages/align/execution.py",
        "pixipix/stages/align/geometry.py",
        "pixipix/stages/align/planning.py",
    }
    assert {member for member in members if member.startswith("pixipix/stages/align")} == (
        expected_align_members
    )
    assert "pixipix/stages/align.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_scale_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale"
    expected_members = {
        f"pixipix/stages/scale/{name}": source_root / name
        for name in (
            "__init__.py",
            "api.py",
            "execution.py",
            "geometry.py",
            "metadata.py",
            "planning.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {member for member in members if member.startswith("pixipix/stages/scale")} == set(
            expected_members
        )
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/scale.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_pixelize_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize"
    expected_members = {
        f"pixipix/stages/pixelize/{name}": source_root / name
        for name in (
            "__init__.py",
            "api.py",
            "execution.py",
            "metadata.py",
            "planning.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {
            member for member in members if member.startswith("pixipix/stages/pixelize")
        } == set(expected_members)
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/pixelize.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_scale_and_pixelize_imports_work_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / f"{artifact_name}-scale-import"
    working_directory.mkdir()
    code = (
        "import pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix.stages.pixelize as pixelize; "
        "import pixipix.stages.pixelize.api as pixelize_api; "
        "import pixipix.stages.pixelize.execution as pixelize_execution; "
        "import pixipix.stages.pixelize.metadata as pixelize_metadata; "
        "import pixipix.stages.pixelize.planning as pixelize_planning; "
        "import pixipix.stages.scale as scale; "
        "import pixipix.stages.scale.api as api; "
        "import pixipix.stages.scale.execution as execution; "
        "import pixipix.stages.scale.geometry as geometry; "
        "import pixipix.stages.scale.metadata as metadata; "
        "import pixipix.stages.scale.planning as planning; "
        "assert pathlib.Path(pixelize.__file__).name == '__init__.py'; "
        "assert pixelize.__spec__.submodule_search_locations is not None; "
        "assert pixelize.publish_pixelize is pixelize_api.publish_pixelize; "
        "assert pixelize.PreparedCellGrid is pixelize_execution.PreparedCellGrid; "
        "assert pixelize.CellGridProjection is pixelize_planning.CellGridProjection; "
        "assert pixelize.PixelizeRun is pixelize_execution.PixelizeRun; "
        "assert pixelize.PixelizeStagePlan is pixelize_planning.PixelizeStagePlan; "
        "assert pixelize.project_cell_grid is pixelize_planning.project_cell_grid; "
        "assert pixelize.prepare_cell_grid is pixelize_execution.prepare_cell_grid; "
        "assert pixelize.representative_pixel is pixelize_execution.representative_pixel; "
        "assert pixelize.apply_alpha_policy is pixelize_execution.apply_alpha_policy; "
        "assert pixelize.pixelize_prepared_grid "
        "is pixelize_execution.pixelize_prepared_grid; "
        "assert pixelize.project_pixelize_resources "
        "is pixelize_planning.project_pixelize_resources; "
        "assert pixelize.project_pixelize_stage is pixelize_planning.project_pixelize_stage; "
        "assert pixelize.pixelize_stage is pixelize_execution.pixelize_stage; "
        "assert pixelize.MAX_PREPARED_PIXELS is pixelize_planning.MAX_PREPARED_PIXELS; "
        "assert pixelize.PixelizeStagePlan.__module__ "
        "== 'pixipix.stages.pixelize.planning'; "
        "assert pixelize.PixelizeRun.__module__ "
        "== 'pixipix.stages.pixelize.execution'; "
        "assert pixelize.publish_pixelize.__module__ == 'pixipix.stages.pixelize.api'; "
        "assert pixelize.round_channel_half_away_from_zero "
        "is pixelize_execution.round_channel_half_away_from_zero "
        "is scale.round_channel_half_away_from_zero; "
        "assert callable(pixelize_metadata.build_pixelize_metadata); "
        "assert not hasattr(pixelize, 'decode_stage_input'); "
        "assert not hasattr(pixelize, 'np'); "
        "assert not hasattr(pixelize, 'Image'); "
        "assert not hasattr(pixelize, '_require_pixelize_config'); "
        "assert not hasattr(pixelize, '_validate_config_handoff'); "
        "assert not hasattr(pixelize, 'build_pixelize_metadata'); "
        "assert scale.publish_scale is api.publish_scale; "
        "assert scale.ScaleRun is execution.ScaleRun; "
        "assert scale.scale_stage is execution.scale_stage; "
        "assert scale.premultiplied_box_resize is execution.premultiplied_box_resize; "
        "assert scale.ScaleStagePlan is planning.ScaleStagePlan; "
        "assert scale.MAX_TRANSFORMED_PIXELS is planning.MAX_TRANSFORMED_PIXELS; "
        "assert scale.project_scale_stage is planning.project_scale_stage; "
        "assert scale.project_scale_resources is planning.project_scale_resources; "
        "assert scale.round_half_away_from_zero is geometry.round_half_away_from_zero; "
        "assert scale.transformed_dimension is geometry.transformed_dimension; "
        "assert scale.round_channel_half_away_from_zero "
        "is geometry.round_channel_half_away_from_zero; "
        "assert callable(metadata.build_scale_metadata); "
        "assert scale.ScaleStagePlan.__module__ == 'pixipix.stages.scale.planning'; "
        "assert scale.publish_scale.__module__ == 'pixipix.stages.scale.api'; "
        "assert scale.round_channel_half_away_from_zero.__module__ "
        "== 'pixipix.stages.scale.geometry'; "
        "assert not hasattr(scale, 'decode_stage_input'); "
        "assert not hasattr(scale, 'Image'); "
        "assert not hasattr(scale, 'np'); "
        "assert not hasattr(scale, '_require_scale_config'); "
        "assert not hasattr(scale, '_resize_float_channel'); "
        "assert not hasattr(scale, 'build_scale_metadata'); "
        "assert sys.modules['pixipix.stages.pixelize'] is pixelize; "
        "assert 'pixipix.stages.pixelize.__init__' not in sys.modules; "
        "assert sys.modules['pixipix.stages.scale'] is scale; "
        "assert 'pixipix.stages.scale.__init__' not in sys.modules; "
        "print(pathlib.Path(scale.__file__).resolve()); "
        "print(pathlib.Path(pixelize.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_shared_pipeline_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    expected_members = {
        "pixipix/pipeline/__init__.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "__init__.py",
        "pixipix/pipeline/artifacts.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "artifacts.py",
        "pixipix/pipeline/input.py": PROJECT_ROOT / "src" / "pixipix" / "pipeline" / "input.py",
        "pixipix/pipeline/publication.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "publication.py",
        "pixipix/stages/io.py": PROJECT_ROOT / "src" / "pixipix" / "stages" / "io.py",
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {member for member in members if member.startswith("pixipix/pipeline/")} == set(
            expected_members
        ) - {"pixipix/stages/io.py"}
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_shared_pipeline_imports_work_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / artifact_name
    working_directory.mkdir()
    code = (
        "import pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix.pipeline.input as pipeline_input; "
        "import pixipix.pipeline.publication as publication; "
        "import pixipix.stages.io as stage_io; "
        "assert stage_io.load_stage_input is pipeline_input.load_stage_input; "
        "assert stage_io._valid_owned_output is publication._valid_owned_output; "
        "print(pathlib.Path(pipeline_input.__file__).resolve()); "
        "print(pathlib.Path(publication.__file__).resolve()); "
        "print(pathlib.Path(stage_io.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


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


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_corrupted_installed_artifact_fails_at_align(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    corrupted = tmp_path / wheel.name
    _corrupt_align_implementation(wheel, corrupted)

    result = _run_corrupted_installed_pipeline(corrupted, tmp_path / "installed")

    assert result.returncode == 4
    assert result.stdout == ""
    assert "PX_INTERNAL_001" in result.stderr
    assert "Traceback" not in result.stderr
