# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
import pytest
from PIL import Image

import pixipix.imageio as imageio
import pixipix.pipeline.input as pipeline_input
import pixipix.pipeline.publication as pipeline_publication
import pixipix.serialization as serialization
import pixipix.stages.align.api as align_api
import pixipix.stages.extract.analysis as extract_analysis
import pixipix.stages.extract.execution as extract_execution
import pixipix.stages.pixelize.api as pixelize_api
import pixipix.stages.pixelize.execution as pixelize_execution
import pixipix.stages.scale as scale_stage
import pixipix.stages.scale.api as scale_api
import pixipix.stages.scale.execution as scale_execution
from pixipix.config import LoadedConfig, SourceConfig, load_config
from pixipix.errors import PixiPixError, ProcessingError, UnsupportedInputError
from pixipix.pipeline.input import LoadedStageInput, load_stage_input
from pixipix.pipeline.publication import OutputFrameImage, publish_stage_output
from pixipix.serialization import write_json as serialization_write_json
from pixipix.stages.align import publish_align
from pixipix.stages.pixelize import publish_pixelize
from tests.helpers import (
    alignment_config,
    pipeline_config,
    write_config,
    write_declared_extract_stage,
    write_declared_pixelize_stage,
    write_declared_scale_stage,
    write_rgba,
)

type CombinedInvalidCase = Literal[
    "marker_and_stage_json",
    "schema_and_missing_frame",
    "identity_and_resource",
    "dimensions_and_png",
    "stage_metadata_and_png",
    "geometry_and_png",
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


class _DelegatingPillowProxy:
    def __init__(self, open_image: Callable[..., object]) -> None:
        self.open = open_image
        self.DecompressionBombWarning = Image.DecompressionBombWarning
        self.DecompressionBombError = Image.DecompressionBombError

    def __getattr__(self, name: str) -> object:
        return getattr(Image, name)


class _PipelineInputNumpyProxy:
    def __init__(self, array: Callable[..., object]) -> None:
        self.array = array

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


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
            "geometry_and_png",
            (
                "PX_STAGE_009",
                'scale frame "one" output dimensions 3x2 do not match declared scale geometry 2x2',
                None,
                None,
            ),
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
    if case in {"stage_metadata_and_png", "geometry_and_png"}:
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
    elif case in {"stage_metadata_and_png", "geometry_and_png"}:
        if case == "stage_metadata_and_png":
            metadata["scaleMode"] = "invalid"
        frames = metadata["frames"]
        assert isinstance(frames, list)
        frame = frames[0]
        assert isinstance(frame, dict)
        output_dimensions = frame["outputDimensions"]
        assert isinstance(output_dimensions, dict)
        output_dimensions["width"] = 3
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

    real_pipeline_image = pipeline_input.Image
    foundational_open = Image.open
    proxy = _DelegatingPillowProxy(fail_image_open)
    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline_input, "Image", proxy)
        scoped.setattr(scale_api, "decode_stage_input", fail_scale_decode)

        with pytest.raises(UnsupportedInputError) as captured:
            if case == "identity_and_resource":
                scale_stage.publish_scale(root, loaded, tmp_path / "output")
            else:
                load_stage_input(root, expected_stage)
        assert Image.open is foundational_open
        assert proxy.DecompressionBombWarning is Image.DecompressionBombWarning
        assert proxy.DecompressionBombError is Image.DecompressionBombError

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
    assert pipeline_input.Image is real_pipeline_image
    assert Image.open is foundational_open


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
    monkeypatch: pytest.MonkeyPatch,
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
        real_array = np.array
        foundational_open = Image.open
        real_pipeline_numpy = pipeline_input.np
        real_pipeline_image = pipeline_input.Image
        neighboring_numpy = (
            scale_execution.np,
            pixelize_execution.np,
            extract_analysis.np,
            extract_execution.np,
        )
        open_calls: list[object] = []
        array_calls: list[dict[str, object]] = []

        def record_open(*args: object, **kwargs: object) -> object:
            open_calls.append(args[0])
            return cast(Any, foundational_open)(*args, **kwargs)

        def record_array(*args: object, **kwargs: object) -> object:
            array_calls.append(kwargs)
            return cast(Any, real_array)(*args, **kwargs)

        proxy = _DelegatingPillowProxy(record_open)
        with monkeypatch.context() as scoped:
            scoped.setattr(pipeline_input, "Image", proxy)
            scoped.setattr(pipeline_input, "np", _PipelineInputNumpyProxy(record_array))
            imageio.load_source(frame_path, SourceConfig())
            assert open_calls == []

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
            assert len(open_calls) == 2
            assert array_calls == [
                {"dtype": np.uint8, "copy": True},
                {"dtype": np.uint8, "copy": True},
            ]
            assert (
                scale_execution.np,
                pixelize_execution.np,
                extract_analysis.np,
                extract_execution.np,
            ) == neighboring_numpy
            assert np.array is real_array
            assert Image.open is foundational_open
            assert proxy.DecompressionBombWarning is Image.DecompressionBombWarning
            assert proxy.DecompressionBombError is Image.DecompressionBombError

        assert pipeline_input.np is real_pipeline_numpy
        assert pipeline_input.Image is real_pipeline_image
        assert np.array is real_array
        assert Image.open is foundational_open
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


def _admitted_stage_input(
    tmp_path: Path,
    stage: Literal["extract", "scale", "pixelize"],
) -> tuple[LoadedConfig, Path]:
    config = tmp_path / f"{stage}.toml"
    write_config(config, alignment_config(names=("one",), width=4, height=4))
    loaded = load_config(config)
    root = tmp_path / stage
    if stage == "extract":
        write_declared_extract_stage(root, loaded, ((2, 2),))
        dimensions = (2, 2)
    elif stage == "scale":
        write_declared_scale_stage(root, loaded, ((2, 2),), ((2, 2),), factor=1.0)
        dimensions = (2, 2)
    else:
        write_declared_pixelize_stage(root, loaded, ((1, 1),))
        dimensions = (1, 1)
    write_rgba(
        root / "frames" / "one.png",
        np.zeros((dimensions[1], dimensions[0], 4), dtype=np.uint8),
    )
    return loaded, root


def test_shared_publication_ignores_late_imageio_write_png_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scaled"
    metadata, frames = _publication_payload()
    assert "pixipix.pipeline.publication" in sys.modules
    retained_write_png = pipeline_publication.write_png
    calls = 0

    def wrong_owner_write_png(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("late imageio.write_png patch intercepted shared publication")

    monkeypatch.setattr(imageio, "write_png", wrong_owner_write_png)

    publish_stage_output(output, "scale", metadata, frames)

    assert output.is_dir()
    assert calls == 0
    assert pipeline_publication.write_png is retained_write_png


def test_shared_publication_ignores_late_serialization_write_json_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "scaled"
    metadata, frames = _publication_payload()
    assert "pixipix.pipeline.publication" in sys.modules
    retained_write_json = pipeline_publication.write_json
    calls = 0

    def wrong_owner_write_json(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("late serialization.write_json patch intercepted publication")

    monkeypatch.setattr(serialization, "write_json", wrong_owner_write_json)

    publish_stage_output(output, "scale", metadata, frames)

    assert output.is_dir()
    assert calls == 0
    assert pipeline_publication.write_json is retained_write_json


def test_align_ignores_late_shared_decoder_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, pixelized = _admitted_stage_input(tmp_path, "pixelize")
    output = tmp_path / "aligned"
    assert "pixipix.stages.align.api" in sys.modules
    retained_decoder = align_api.decode_stage_input
    calls = 0

    def wrong_owner_decode(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("late shared decoder patch intercepted Align")

    monkeypatch.setattr(pipeline_input, "decode_stage_input", wrong_owner_decode)

    metadata = publish_align(pixelized, loaded, output)

    assert metadata.stage == "align"
    assert output.is_dir()
    assert calls == 0
    assert align_api.decode_stage_input is retained_decoder


def test_scale_ignores_late_shared_decoder_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, extracted = _admitted_stage_input(tmp_path, "extract")
    output = tmp_path / "scaled"
    assert "pixipix.stages.scale.api" in sys.modules
    retained_decoder = scale_api.decode_stage_input
    calls = 0

    def wrong_owner_decode(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("late shared decoder patch intercepted Scale")

    monkeypatch.setattr(pipeline_input, "decode_stage_input", wrong_owner_decode)

    metadata = scale_stage.publish_scale(extracted, loaded, output)

    assert metadata.stage == "scale"
    assert output.is_dir()
    assert calls == 0
    assert scale_api.decode_stage_input is retained_decoder


def test_pixelize_ignores_late_shared_decoder_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, scaled = _admitted_stage_input(tmp_path, "scale")
    output = tmp_path / "pixelized"
    assert "pixipix.stages.pixelize.api" in sys.modules
    retained_decoder = pixelize_api.decode_stage_input
    calls = 0

    def wrong_owner_decode(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("late shared decoder patch intercepted Pixelize")

    monkeypatch.setattr(pipeline_input, "decode_stage_input", wrong_owner_decode)

    metadata = publish_pixelize(scaled, loaded, output)

    assert metadata.stage == "pixelize"
    assert output.is_dir()
    assert calls == 0
    assert pixelize_api.decode_stage_input is retained_decoder


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
    foundational_open = Image.open
    real_publication_image = pipeline_publication.Image
    neighboring_images = (pipeline_input.Image, imageio.Image)
    observed: list[tuple[str, Path]] = []
    opened: list[object] = []

    def fail_final_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".scaled.pixipix-build-") and target == output:
            observed.append((self.name, target))
            raise OSError("replace sentinel")
        return real_replace(self, target)

    def record_open(*args: object, **kwargs: object) -> object:
        opened.append(args[0])
        return cast(Any, foundational_open)(*args, **kwargs)

    unrelated_source = tmp_path / "unrelated-source"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_source.write_text("delegated", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_final_replace)
        scoped.setattr(
            pipeline_publication,
            "Image",
            _DelegatingPillowProxy(record_open),
        )
        assert unrelated_source.replace(unrelated_target) == unrelated_target
        assert unrelated_target.read_text(encoding="utf-8") == "delegated"
        with pytest.raises(ProcessingError, match="PX_OUTPUT_005") as captured:
            publish_stage_output(output, "scale", metadata, frames)
        assert len(opened) == 1
        assert (pipeline_input.Image, imageio.Image) == neighboring_images
        assert Image.open is foundational_open

    assert len(observed) == 1
    assert observed[0][0].startswith(".scaled.pixipix-build-")
    assert observed[0][1] == output
    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "replace sentinel"
    assert not output.exists()
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []
    assert list(tmp_path.glob(".scaled.pixipix-backup-*")) == []
    assert Path.replace is real_replace
    assert pipeline_publication.Image is real_publication_image
    assert Image.open is foundational_open


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
