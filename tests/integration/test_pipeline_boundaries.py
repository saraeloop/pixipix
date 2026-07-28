from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import pytest

import pixipix.pipeline.input as pipeline_input
import pixipix.stages.scale as scale_stage
import pixipix.stages.scale.api as scale_api
from pixipix.config import load_config
from pixipix.errors import PixiPixError, ProcessingError, UnsupportedInputError
from pixipix.pipeline.input import LoadedStageInput, load_stage_input
from pixipix.pipeline.publication import OutputFrameImage, publish_stage_output
from pixipix.serialization import write_json as serialization_write_json
from tests.helpers import (
    pipeline_config,
    write_config,
    write_declared_extract_stage,
    write_declared_scale_stage,
    write_rgba,
)

type CombinedInvalidCase = Literal[
    "marker_and_stage_json",
    "schema_and_missing_frame",
    "identity_and_resource",
    "dimensions_and_png",
    "stage_metadata_and_png",
    "extra_file_and_png",
    "hash_and_png",
]
type LoadEquivalenceCase = Literal[
    "valid",
    "metadata_failure",
    "malformed_png",
    "dimension_mismatch",
    "hash_mismatch",
]


def _metadata(root: Path) -> dict[str, object]:
    value = json.loads((root / "stage.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_metadata(root: Path, value: dict[str, object]) -> None:
    (root / "stage.json").write_text(json.dumps(value), encoding="utf-8")


def _error_signature(error: PixiPixError) -> tuple[object, ...]:
    return (
        type(error),
        error.code,
        error.stage,
        error.message,
        error.path,
        error.frame,
        error.remediation,
        type(error.__cause__) if error.__cause__ is not None else None,
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "marker_and_stage_json",
            (
                "PX_STAGE_002",
                "required stage file is not valid JSON",
                ".pixipix-output",
                "JSONDecodeError",
            ),
        ),
        (
            "schema_and_missing_frame",
            ("PX_STAGE_005", "unsupported stage metadata schema 99", "input", None),
        ),
        (
            "identity_and_resource",
            ("PX_STAGE_008", "stage metadata has invalid effective config hash", None, None),
        ),
        (
            "dimensions_and_png",
            ("PX_STAGE_007", "invalid frame width in stage metadata", None, None),
        ),
        (
            "stage_metadata_and_png",
            ("PX_STAGE_009", "scale metadata has invalid scale mode", None, None),
        ),
        (
            "extra_file_and_png",
            ("PX_STAGE_014", "stage directory contains undeclared artifacts", None, None),
        ),
        (
            "hash_and_png",
            (
                "PX_STAGE_013",
                "declared frame hash does not match bytes",
                "frames/one.png",
                None,
            ),
        ),
    ],
)
def test_combined_invalid_inputs_preserve_first_error_and_avoid_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: CombinedInvalidCase,
    expected: tuple[str, str, str | None, str | None],
) -> None:
    config = tmp_path / "project.toml"
    config_text = pipeline_config(names=("one",))
    if case == "identity_and_resource":
        config_text += "\n[resources]\nmax_aggregate_input_pixels = 1\n"
    write_config(config, config_text)
    loaded = load_config(config)
    root = tmp_path / "input"
    expected_stage: Literal["extract", "scale"] = "extract"
    if case == "stage_metadata_and_png":
        write_declared_scale_stage(root, loaded, ((2, 2),), ((2, 2),), factor=1.0)
        expected_stage = "scale"
    else:
        write_declared_extract_stage(root, loaded, ((2, 2),))

    metadata = _metadata(root)
    frame_path = root / "frames" / "one.png"
    if case == "marker_and_stage_json":
        (root / ".pixipix-output").write_text("{", encoding="utf-8")
        (root / "stage.json").write_text("{", encoding="utf-8")
    elif case == "schema_and_missing_frame":
        metadata["schemaVersion"] = 99
        _write_metadata(root, metadata)
        frame_path.unlink()
    elif case == "identity_and_resource":
        metadata["effectiveConfigSha256"] = "not-a-hash"
        _write_metadata(root, metadata)
    elif case == "dimensions_and_png":
        frames = metadata["frames"]
        assert isinstance(frames, list)
        frame = frames[0]
        assert isinstance(frame, dict)
        bounds = frame["paddedBounds"]
        assert isinstance(bounds, dict)
        bounds["right"] = 0
        _write_metadata(root, metadata)
    elif case == "stage_metadata_and_png":
        metadata["scaleMode"] = "invalid"
        _write_metadata(root, metadata)
    elif case == "extra_file_and_png":
        (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    elif case == "hash_and_png":
        frames = metadata["frames"]
        assert isinstance(frames, list)
        frame = frames[0]
        assert isinstance(frame, dict)
        frame["sha256"] = "0" * 64
        _write_metadata(root, metadata)

    decode_calls: list[str] = []

    def fail_image_open(_path: Path) -> object:
        decode_calls.append("pipeline.input.Image.open")
        raise AssertionError("decode must not run before metadata admission")

    def fail_scale_decode(_validated: object) -> None:
        decode_calls.append("pixipix.stages.scale.api.decode_stage_input")
        raise AssertionError("stage decode must not run before metadata admission")

    monkeypatch.setattr("pixipix.pipeline.input.Image.open", fail_image_open)
    monkeypatch.setattr(scale_api, "decode_stage_input", fail_scale_decode)

    with pytest.raises(UnsupportedInputError) as captured:
        if case == "identity_and_resource":
            scale_stage.publish_scale(root, loaded, tmp_path / "output")
        else:
            load_stage_input(root, expected_stage)

    code, message, path, cause_name = expected
    error = captured.value
    assert error.code == code
    assert error.stage == "load"
    assert error.message == message
    assert error.path == (root.name if path == "input" else path)
    assert error.frame is None
    assert error.remediation is None
    assert (type(error.__cause__).__name__ if error.__cause__ is not None else None) == cause_name
    assert decode_calls == []
    assert not (tmp_path / "output").exists()


def _capture_error(operation: Callable[[], object]) -> PixiPixError:
    try:
        operation()
    except PixiPixError as error:
        return error
    raise AssertionError("operation unexpectedly succeeded")


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "metadata_failure",
        "malformed_png",
        "dimension_mismatch",
        "hash_mismatch",
    ],
)
def test_load_stage_input_matches_explicit_validate_then_decode(
    tmp_path: Path,
    case: LoadEquivalenceCase,
) -> None:
    config = tmp_path / "project.toml"
    write_config(config, pipeline_config(names=("one",)))
    loaded = load_config(config)
    root = tmp_path / "input"
    declared_dimensions = ((1, 1),) if case == "dimension_mismatch" else ((2, 2),)
    write_declared_extract_stage(root, loaded, declared_dimensions)
    frame_path = root / "frames" / "one.png"
    if case in {"valid", "dimension_mismatch", "hash_mismatch"}:
        write_rgba(frame_path, np.zeros((2, 2, 4), dtype=np.uint8))
    metadata = _metadata(root)
    if case == "metadata_failure":
        metadata["schemaVersion"] = 99
        _write_metadata(root, metadata)
    elif case == "hash_mismatch":
        frames = metadata["frames"]
        assert isinstance(frames, list)
        frame = frames[0]
        assert isinstance(frame, dict)
        frame["sha256"] = "0" * 64
        _write_metadata(root, metadata)

    def explicit() -> LoadedStageInput:
        return pipeline_input.decode_stage_input(
            pipeline_input.validate_stage_input(root, "extract")
        )

    def wrapped() -> LoadedStageInput:
        return pipeline_input.load_stage_input(root, "extract")

    if case == "valid":
        explicit_result = explicit()
        wrapped_result = wrapped()
        assert type(explicit_result) is type(wrapped_result) is pipeline_input.LoadedStageInput
        assert explicit_result.identity == wrapped_result.identity
        assert explicit_result.metadata == wrapped_result.metadata
        assert explicit_result.warnings == wrapped_result.warnings
        assert tuple(frame.name for frame in explicit_result.frames) == ("one",)
        assert tuple(frame.name for frame in wrapped_result.frames) == ("one",)
        explicit_pixels = explicit_result.frames[0].pixels
        wrapped_pixels = wrapped_result.frames[0].pixels
        assert np.array_equal(explicit_pixels, wrapped_pixels)
        assert explicit_pixels.flags.owndata
        assert wrapped_pixels.flags.owndata
        assert not np.shares_memory(explicit_pixels, wrapped_pixels)
        return

    explicit_error = _capture_error(explicit)
    wrapped_error = _capture_error(wrapped)
    assert _error_signature(explicit_error) == _error_signature(wrapped_error)
    assert str(explicit_error) == str(wrapped_error)


def _publication_payload() -> tuple[dict[str, object], tuple[OutputFrameImage, ...]]:
    metadata: dict[str, object] = {
        "schemaVersion": 1,
        "stage": "scale",
        "status": "successful",
        "frames": [
            {
                "name": "one",
                "relativePath": "frames/one.png",
                "sourceOrder": 0,
                "outputDimensions": {"width": 1, "height": 1},
            }
        ],
    }
    pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    return metadata, (OutputFrameImage(PurePosixPath("frames/one.png"), pixels),)


def test_publication_write_json_patch_affects_execution_and_cleans_fresh_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scaled"
    metadata, frames = _publication_payload()
    observed: list[str] = []

    def fail_stage_json(path: Path, value: object) -> None:
        observed.append(path.name)
        if path.name == "stage.json":
            raise OSError("write-json sentinel")
        serialization_write_json(path, value)

    monkeypatch.setattr("pixipix.pipeline.publication.write_json", fail_stage_json)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005") as captured:
        publish_stage_output(output, "scale", metadata, frames)

    assert observed == [".pixipix-output", "stage.json"]
    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "write-json sentinel"
    assert not output.exists()
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []


def test_publication_path_replace_patch_affects_fresh_output_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scaled"
    metadata, frames = _publication_payload()
    real_replace = Path.replace
    observed: list[tuple[str, Path]] = []

    def fail_final_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".scaled.pixipix-build-") and target == output:
            observed.append((self.name, target))
            raise OSError("replace sentinel")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_final_replace)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005") as captured:
        publish_stage_output(output, "scale", metadata, frames)

    assert len(observed) == 1
    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "replace sentinel"
    assert not output.exists()
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []
    assert list(tmp_path.glob(".scaled.pixipix-backup-*")) == []


def test_publication_backup_replace_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scaled"
    metadata, frames = _publication_payload()
    publish_stage_output(output, "scale", metadata, frames)
    original = {
        path.relative_to(output): path.read_bytes()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    real_replace = Path.replace
    observed: list[tuple[Path, str]] = []

    def fail_backup_replace(self: Path, target: Path) -> Path:
        if self == output and target.name == "previous":
            observed.append((self, target.parent.name))
            raise OSError("backup sentinel")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_backup_replace)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005") as captured:
        publish_stage_output(output, "scale", metadata, frames, force=True)

    assert len(observed) == 1
    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "backup sentinel"
    assert {
        path.relative_to(output): path.read_bytes()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    } == original
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []
    assert list(tmp_path.glob(".scaled.pixipix-backup-*")) == []
