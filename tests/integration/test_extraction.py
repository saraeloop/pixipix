from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixipix.config import load_config
from pixipix.errors import ProcessingError, ResourcePolicyError
from pixipix.models import (
    ExtractionResult,
    ExtractionRun,
    FrameImage,
    InspectionResult,
    StageMetadata,
)
from pixipix.stages import extract as extract_stage
from pixipix.stages.extract import inspect_source, publish_extraction
from tests.helpers import extraction_config, transparent_sheet, write_config, write_rgba


def _project(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "robot.png"
    config = tmp_path / "robot.toml"
    write_rgba(image, transparent_sheet())
    write_config(config)
    return image, config


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_inspection_is_typed_and_does_not_write_artifacts(tmp_path: Path) -> None:
    image, config = _project(tmp_path)

    result = inspect_source(image, load_config(config))

    assert type(result) is InspectionResult
    assert result.source.width == 14
    assert len(result.candidates) == 3
    assert len(result.accepted) == 2
    assert len(result.rejected) == 1
    assert result.frame_assignments == ("idle", "signal")
    assert result.configured_source_cell_size is None
    assert not (tmp_path / "stage.json").exists()


def test_extract_writes_component_only_rgba_frames_and_metadata(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "output"

    result = publish_extraction(image, load_config(config), output)

    assert type(result) is ExtractionResult
    assert [frame.name for frame in result.frames] == ["idle", "signal"]
    assert (output / ".pixipix-output").is_file()
    metadata = json.loads((output / "stage.json").read_text(encoding="utf-8"))
    assert metadata["schemaVersion"] == 1
    assert metadata["pixipixVersion"] == distribution_version("pixipix")
    assert metadata["stage"] == "extract"
    assert metadata["status"] == "successful"
    assert [frame["name"] for frame in metadata["frames"]] == ["idle", "signal"]
    assert len(metadata["candidateComponents"]) == 3
    assert len(metadata["acceptedComponents"]) == 2
    assert metadata["orderedComponents"] == [
        metadata["acceptedComponents"][0],
        metadata["acceptedComponents"][1],
    ]
    assert metadata["rejectedComponents"][0]["reasons"] == ["below-minimum-area"]
    rendered = (output / "stage.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in rendered
    assert "timestamp" not in rendered.lower()

    with Image.open(output / "frames" / "idle.png") as first:
        first_pixels = np.asarray(first.convert("RGBA"))
        assert first.mode == "RGBA"
    assert first_pixels.shape == (5, 5, 4)
    assert first_pixels[0, 0].tolist() == [0, 0, 0, 0]
    assert np.count_nonzero(first_pixels[:, :, 3]) == 9

    with Image.open(output / "frames" / "signal.png") as second:
        second_pixels = np.asarray(second.convert("RGBA"))
    assert set(np.unique(second_pixels[:, :, 3]).tolist()) == {0, 200}

    frame_paths = {frame["relativePath"] for frame in metadata["frames"]}
    assert frame_paths == {
        path.relative_to(output).as_posix() for path in (output / "frames").glob("*.png")
    }


def test_extract_run_and_frame_image_identities_are_exact(tmp_path: Path) -> None:
    image, config = _project(tmp_path)

    run = extract_stage.extract_source(image, load_config(config))

    assert type(run) is ExtractionRun
    assert type(run.result) is ExtractionResult
    assert all(type(frame) is FrameImage for frame in run.frame_images)


def test_repeated_extractions_are_byte_identical(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    loaded = load_config(config)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = publish_extraction(image, loaded, first)
    second_result = publish_extraction(image, loaded, second)

    assert _artifact_bytes(first) == _artifact_bytes(second)
    assert first_result.ordered == second_result.ordered
    assert first_result.frames == second_result.frames


def test_separate_process_extractions_are_byte_identical(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    console = Path(sys.executable).with_name("pixipix")
    first = tmp_path / "process-first"
    second = tmp_path / "process-second"

    for output in (first, second):
        result = subprocess.run(
            [console, "extract", image, "--config", config, "--output", output],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert _artifact_bytes(first) == _artifact_bytes(second)


def test_count_mismatch_stops_before_output(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    write_config(config, extraction_config(names=("only",), expected=None))
    output = tmp_path / "output"

    with pytest.raises(ProcessingError, match="PX_EXTRACT_003"):
        publish_extraction(image, load_config(config), output)

    assert not output.exists()


def test_padded_crop_masks_neighboring_accepted_and_rejected_components(tmp_path: Path) -> None:
    pixels = np.zeros((5, 8, 4), dtype=np.uint8)
    pixels[1:3, 1:3] = (10, 20, 30, 255)
    pixels[1, 4] = (200, 10, 10, 255)
    image = tmp_path / "neighbors.png"
    config = tmp_path / "neighbors.toml"
    write_rgba(image, pixels)
    write_config(
        config,
        extraction_config(names=("kept",), expected=1, minimum_area=2, padding=3),
    )

    result = publish_extraction(image, load_config(config), tmp_path / "output")

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    with Image.open(tmp_path / "output" / "frames" / "kept.png") as frame:
        rendered = np.asarray(frame.convert("RGBA"))
    assert rendered[1, 4].tolist() == [0, 0, 0, 0]
    assert np.count_nonzero(rendered[:, :, 3]) == 4


def test_synthetic_non_animal_asset_processes(tmp_path: Path) -> None:
    pixels = np.zeros((8, 16, 4), dtype=np.uint8)
    pixels[1:6, 1:5] = (50, 70, 90, 255)
    pixels[2:7, 10:15] = (80, 120, 160, 255)
    image = tmp_path / "geometric-robot.png"
    config = tmp_path / "robot.toml"
    write_rgba(image, pixels)
    write_config(config, extraction_config(minimum_area=4, padding=0))

    result = publish_extraction(image, load_config(config), tmp_path / "robot-output")

    assert [frame.name for frame in result.frames] == ["idle", "signal"]
    schema = (tmp_path / "robot-output" / "stage.json").read_text(encoding="utf-8").lower()
    for forbidden in ("feline", "cat", "fur", "collar", "paw", "tail", "breed"):
        assert forbidden not in schema


def test_failed_staging_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "output"

    def fail_write(_path: Path, _pixels: object) -> None:
        raise ProcessingError("PX_TEST", "encode", "simulated failure")

    monkeypatch.setattr(extract_stage, "write_png", fail_write)
    with pytest.raises(ProcessingError, match="PX_TEST"):
        publish_extraction(image, load_config(config), output)

    assert not output.exists()
    assert list(tmp_path.glob(".output.pixipix-build-*")) == []


def test_admitted_extract_uses_package_crop_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "output"

    class ExtractCropReached(Exception):
        pass

    def mark_crop(_analysis: object, _component: object, _frame: object) -> None:
        raise ExtractCropReached("extract package crop binding reached")

    monkeypatch.setattr(extract_stage, "_materialize_frame_crop", mark_crop)

    with pytest.raises(
        ExtractCropReached,
        match="extract package crop binding reached",
    ) as raised:
        publish_extraction(image, load_config(config), output)

    traceback_names = tuple(entry.name for entry in raised.traceback)
    assert "extract_source" in traceback_names
    assert traceback_names[-1] == "mark_crop"
    assert not output.exists()
    assert list(tmp_path.glob(".output.pixipix-build-*")) == []


def test_nonempty_foreign_directory_is_never_replaced(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "foreign"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("important", encoding="utf-8")

    with pytest.raises(ProcessingError, match="PX_OUTPUT_002"):
        publish_extraction(image, load_config(config), output)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_extraction(image, load_config(config), output, force=True)

    assert keep.read_text(encoding="utf-8") == "important"


def test_marker_alone_or_corrupt_stage_never_authorizes_replacement(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    loaded = load_config(config)
    output = tmp_path / "forged"
    output.mkdir()
    marker = output / ".pixipix-output"
    marker.write_text('{"owner":"pixipix","schemaVersion":1,"stage":"extract"}\n', encoding="utf-8")
    keep = output / "keep.txt"
    keep.write_text("important", encoding="utf-8")

    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_extraction(image, loaded, output, force=True)

    publish_extraction(image, loaded, tmp_path / "valid")
    valid_stage = (tmp_path / "valid" / "stage.json").read_text(encoding="utf-8")
    (output / "stage.json").write_text(
        valid_stage.replace('"successful"', '"failed"'), encoding="utf-8"
    )
    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_extraction(image, loaded, output, force=True)
    assert keep.read_text(encoding="utf-8") == "important"


def test_existing_empty_output_is_replaced_without_force(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "empty"
    output.mkdir()

    publish_extraction(image, load_config(config), output)

    assert (output / "stage.json").is_file()


def test_target_created_during_staging_is_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "raced"
    original_validate = extract_stage._validate_staged_output

    def create_foreign_target(root: Path, metadata: StageMetadata) -> None:
        original_validate(root, metadata)
        output.mkdir()
        (output / "keep.txt").write_text("important", encoding="utf-8")

    monkeypatch.setattr(extract_stage, "_validate_staged_output", create_foreign_target)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_002"):
        publish_extraction(image, load_config(config), output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "important"
    assert list(tmp_path.glob(".raced.pixipix-build-*")) == []


def test_owned_output_can_be_replaced_with_force(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "owned"
    loaded = load_config(config)
    publish_extraction(image, loaded, output)
    stale = output / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    publish_extraction(image, loaded, output, force=True)

    assert not stale.exists()
    assert (output / "stage.json").is_file()
    assert list(tmp_path.glob(".owned.pixipix-backup-*")) == []


def test_symlink_output_is_rejected(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "linked"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProcessingError, match="PX_OUTPUT_004"):
        publish_extraction(image, load_config(config), output, force=True)


def test_extract_resource_checkpoint_precedes_crop_and_preserves_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures"
    image = fixture_root / "robot-geometric.png"
    fixture_config = fixture_root / "robot-geometric.toml"
    admitted_config = tmp_path / "admitted.toml"
    refused_config = tmp_path / "refused.toml"
    output = tmp_path / "owned"
    admitted_config.write_bytes(fixture_config.read_bytes())
    refused_config.write_text(
        fixture_config.read_text(encoding="utf-8")
        + "\n[resources]\nmax_aggregate_output_pixels = 169\n",
        encoding="utf-8",
    )
    publish_extraction(image, load_config(admitted_config), output)
    original = _artifact_bytes(output)

    def fail_crop(_analysis: object, _component: object, _frame: object) -> None:
        raise AssertionError("frame crop must not materialize before resource admission")

    monkeypatch.setattr(extract_stage, "_materialize_frame_crop", fail_crop)
    with pytest.raises(ResourcePolicyError) as raised:
        publish_extraction(image, load_config(refused_config), output, force=True)

    projection = raised.value.projection
    assert raised.value.policy == load_config(refused_config).config.resources
    assert (
        projection.aggregate_input_pixels,
        projection.aggregate_output_pixels,
        projection.modeled_peak_live_bytes,
    ) == (240, 170, 2_930)
    assert tuple(finding.kind for finding in raised.value.findings) == ("aggregate_output_pixels",)
    assert capsys.readouterr() == ("", "")
    assert _artifact_bytes(output) == original
    assert list(tmp_path.glob(".owned.pixipix-build-*")) == []


def test_different_admitting_resource_policies_preserve_extraction_results_and_pngs(
    tmp_path: Path,
) -> None:
    image, default_config = _project(tmp_path)
    raised_config = tmp_path / "raised.toml"
    write_config(
        raised_config,
        extraction_config() + "\n[resources]\nmax_aggregate_output_pixels = 60000001\n",
    )
    default_loaded = load_config(default_config)
    raised_loaded = load_config(raised_config)
    default_output = tmp_path / "default-output"
    raised_output = tmp_path / "raised-output"

    default_result = publish_extraction(image, default_loaded, default_output)
    raised_result = publish_extraction(image, raised_loaded, raised_output)

    assert default_loaded.effective_config_sha256 != raised_loaded.effective_config_sha256
    assert default_result == raised_result
    assert {path.name: path.read_bytes() for path in (default_output / "frames").iterdir()} == {
        path.name: path.read_bytes() for path in (raised_output / "frames").iterdir()
    }
    default_stage_bytes = (default_output / "stage.json").read_bytes()
    raised_stage_bytes = (raised_output / "stage.json").read_bytes()
    assert default_stage_bytes != raised_stage_bytes
    default_stage = json.loads(default_stage_bytes)
    raised_stage = json.loads(raised_stage_bytes)
    assert default_stage.pop("sourceConfigSha256") == default_loaded.source_config_sha256
    assert raised_stage.pop("sourceConfigSha256") == raised_loaded.source_config_sha256
    assert default_stage.pop("effectiveConfigSha256") == default_loaded.effective_config_sha256
    assert raised_stage.pop("effectiveConfigSha256") == raised_loaded.effective_config_sha256
    assert default_stage == raised_stage


def test_symlink_parent_and_dangerous_repository_root_are_rejected(tmp_path: Path) -> None:
    image, config = _project(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ProcessingError, match="PX_OUTPUT_004"):
        publish_extraction(image, load_config(config), linked_parent / "output")
    with pytest.raises(ProcessingError, match="PX_OUTPUT_007"):
        publish_extraction(image, load_config(config), Path.cwd(), force=True)


def test_root_owned_standard_tmp_alias_is_allowed_when_present() -> None:
    if not Path("/tmp").is_symlink():
        pytest.skip("platform /tmp is not a symlink")

    extract_stage._validate_output_location(Path("/tmp/pixipix-validation-probe"))


def test_previous_output_is_restored_on_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, config = _project(tmp_path)
    output = tmp_path / "owned"
    loaded = load_config(config)
    publish_extraction(image, loaded, output)
    original = _artifact_bytes(output)
    real_replace = Path.replace

    def fail_new_publication(self: Path, target: Path) -> Path:
        if self.name.startswith(".owned.pixipix-build-") and target == output:
            raise OSError("simulated rename failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_publication)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005"):
        publish_extraction(image, loaded, output, force=True)

    assert _artifact_bytes(output) == original
    assert list(tmp_path.glob(".owned.pixipix-backup-*")) == []
