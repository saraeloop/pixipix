from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pixipix.stages.io as stage_io
from pixipix.config import load_config
from pixipix.errors import (
    AlignmentClippingError,
    ConfigurationError,
    ExitCode,
    ProcessingError,
    ResourcePolicyError,
    UnsupportedInputError,
)
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale
from tests.helpers import (
    alignment_config,
    pipeline_config,
    transparent_sheet,
    write_config,
    write_declared_pixelize_stage,
    write_rgba,
)


def _pipeline(
    tmp_path: Path,
    *,
    config_text: str | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    extracted = tmp_path / "extracted"
    scaled = tmp_path / "scaled"
    pixelized = tmp_path / "pixelized"
    write_rgba(image, transparent_sheet())
    write_config(config, config_text or alignment_config())
    loaded = load_config(config)
    publish_extraction(image, loaded, extracted)
    publish_scale(extracted, loaded, scaled)
    publish_pixelize(scaled, loaded, pixelized)
    return image, config, extracted, scaled, pixelized


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_pixelize_to_align_output_and_metadata_contract(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    aligned = tmp_path / "aligned"
    metadata = publish_align(pixelized, load_config(config), aligned)

    assert metadata.stage == "align"
    assert metadata.prior_stage.stage == "pixelize"
    assert (metadata.canvas_width, metadata.canvas_height) == (8, 8)
    assert metadata.anchor == "bottom-center"
    assert metadata.configured_baseline_y is None
    assert metadata.effective_baseline_y == 8
    assert metadata.clipping_policy == "error"
    assert metadata.clipping_findings == ()
    assert [(item.name, item.input_width, item.input_height) for item in metadata.frames] == [
        ("idle", 2, 2),
        ("signal", 2, 1),
    ]
    assert [(item.final_x, item.final_y) for item in metadata.frames] == [(3, 6), (3, 7)]
    assert (aligned / ".pixipix-output").read_text(encoding="utf-8") == (
        '{\n  "owner": "pixipix",\n  "schemaVersion": 1,\n  "stage": "align"\n}\n'
    )
    rendered = json.loads((aligned / "stage.json").read_text(encoding="utf-8"))
    assert rendered["frames"][0]["visibleSourceRectangle"] == {
        "height": 2,
        "width": 2,
        "x": 0,
        "y": 0,
    }
    assert rendered["frames"][0]["visibleDestinationRectangle"] == {
        "height": 2,
        "width": 2,
        "x": 3,
        "y": 6,
    }
    assert str(tmp_path) not in json.dumps(rendered)
    for frame in metadata.frames:
        with Image.open(aligned / frame.relative_path) as image:
            image.load()
            assert image.mode == "RGBA"
            assert image.size == (8, 8)


def test_alignment_exact_pixels_padding_and_no_vertical_inversion(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    aligned = tmp_path / "aligned"
    metadata = publish_align(pixelized, load_config(config), aligned)
    for frame in metadata.frames:
        with Image.open(pixelized / frame.relative_path) as source_image:
            source = np.array(source_image, dtype=np.uint8, copy=True)
        with Image.open(aligned / frame.relative_path) as output_image:
            output = np.array(output_image, dtype=np.uint8, copy=True)
        destination = frame.visible_destination_rectangle
        assert np.array_equal(
            output[
                destination.y : destination.y + destination.height,
                destination.x : destination.x + destination.width,
            ],
            source,
        )
        occupied = np.zeros(output.shape[:2], dtype=np.bool_)
        occupied[
            destination.y : destination.y + destination.height,
            destination.x : destination.x + destination.width,
        ] = True
        assert not output[~occupied].any()


def test_offsets_and_warning_order_follow_prior_frame_order(tmp_path: Path) -> None:
    config_text = alignment_config(
        width=4,
        height=3,
        clip_policy="warn",
        offsets=("[frame_offsets.signal]\ndx = 2\ndy = 0\n\n[frame_offsets.idle]\ndx = 0\ndy = -1"),
    )
    _, config, _, _, pixelized = _pipeline(tmp_path, config_text=config_text)
    stage_path = pixelized / "stage.json"
    prior = json.loads(stage_path.read_text(encoding="utf-8"))
    prior["warnings"] = [{"code": "PX_PRIOR_001", "stage": "pixelize", "message": "prior warning"}]
    stage_path.write_text(json.dumps(prior), encoding="utf-8")

    metadata = publish_align(pixelized, load_config(config), tmp_path / "aligned")
    assert [warning.code for warning in metadata.warnings] == [
        "PX_PRIOR_001",
        "PX_ALIGN_OFFSET_001",
        "PX_ALIGN_OFFSET_001",
        "PX_ALIGN_CLIP_002",
    ]
    assert [warning.message.split('"')[1] for warning in metadata.warnings[1:3]] == [
        "idle",
        "signal",
    ]
    assert [item.frame_name for item in metadata.clipping_findings] == ["signal"]
    assert [(item.offset_dx, item.offset_dy) for item in metadata.frames] == [(0, -1), (2, 0)]


def test_error_policy_aggregates_all_frames_and_publishes_nothing(tmp_path: Path) -> None:
    config_text = alignment_config(width=1, height=1, anchor="center", clip_policy="error")
    _, config, _, _, pixelized = _pipeline(tmp_path, config_text=config_text)
    output = tmp_path / "aligned"
    with pytest.raises(AlignmentClippingError) as captured:
        publish_align(pixelized, load_config(config), output)
    assert [item.frame_name for item in captured.value.findings] == ["idle", "signal"]
    assert not output.exists()
    assert list(tmp_path.glob(".aligned.pixipix-build-*")) == []


def test_align_requires_output_configuration(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path, config_text=pipeline_config())
    with pytest.raises(ConfigurationError, match="PX_ALIGN_CONFIG_001"):
        publish_align(pixelized, load_config(config), tmp_path / "aligned")


@pytest.mark.parametrize(("policy", "warning_count"), [("warn", 2), ("allow", 0)])
def test_warn_and_allow_publish_findings_for_fully_clipped_frames(
    tmp_path: Path, policy: str, warning_count: int
) -> None:
    config_text = alignment_config(
        width=2,
        height=2,
        anchor="top-left",
        clip_policy=policy,
        offsets=(
            "[frame_offsets.idle]\ndx = 20\ndy = 20\n\n[frame_offsets.signal]\ndx = -20\ndy = -20"
        ),
    )
    _, config, _, _, pixelized = _pipeline(tmp_path, config_text=config_text)
    output = tmp_path / "aligned"
    metadata = publish_align(pixelized, load_config(config), output)
    assert len(metadata.clipping_findings) == 2
    assert (
        sum(warning.code == "PX_ALIGN_CLIP_002" for warning in metadata.warnings) == warning_count
    )
    for frame in metadata.frames:
        assert frame.visible_source_rectangle.width == 0
        with Image.open(output / frame.relative_path) as image:
            assert not np.asarray(image, dtype=np.uint8).any()


def test_wrong_prior_stage_and_tampered_pixelize_semantics_are_rejected(tmp_path: Path) -> None:
    _, config, _, scaled, pixelized = _pipeline(tmp_path)
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_003"):
        publish_align(scaled, load_config(config), tmp_path / "wrong")

    metadata_path = pixelized / "stage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["preparedDimensions"]["width"] += 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_016"):
        publish_align(pixelized, load_config(config), tmp_path / "tampered")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schemaVersion", 99),
        ("status", "failed"),
        ("cellGridOrigin", "top-left"),
        ("representative", "mean"),
        ("alphaPolicy", "soft"),
        ("remainderPolicy", "silent"),
    ],
)
def test_pixelize_stage_contract_tampering_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    metadata_path = pixelized / "stage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        publish_align(pixelized, load_config(config), tmp_path / "aligned")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("marker-schema", True, id="boolean-marker-schema"),
        pytest.param("marker-schema", 1.0, id="floating-marker-schema"),
        pytest.param("metadata-schema", True, id="boolean-metadata-schema"),
        pytest.param("metadata-schema", 1.0, id="floating-metadata-schema"),
        pytest.param("prior-schema", True, id="boolean-prior-stage-schema"),
        pytest.param("prior-schema", 1.0, id="floating-prior-stage-schema"),
        pytest.param("source-order", False, id="boolean-source-order"),
        pytest.param("source-order", 0.0, id="floating-source-order"),
    ],
)
def test_noninteger_stage_contract_fields_are_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    if field == "marker-schema":
        path = pixelized / ".pixipix-output"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schemaVersion"] = value
    else:
        path = pixelized / "stage.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if field == "metadata-schema":
            document["schemaVersion"] = value
        elif field == "prior-schema":
            document["priorStage"]["schemaVersion"] = value
        else:
            document["frames"][0]["sourceOrder"] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(UnsupportedInputError):
        publish_align(pixelized, load_config(config), tmp_path / "aligned")


def test_swapped_equal_size_pixelize_paths_are_rejected(tmp_path: Path) -> None:
    pixels = np.zeros((4, 10, 4), dtype=np.uint8)
    pixels[0:4, 0:4] = (255, 0, 0, 255)
    pixels[0:4, 6:10] = (0, 255, 0, 255)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, pixels)
    write_config(config, alignment_config())
    loaded = load_config(config)
    extracted = tmp_path / "extracted"
    scaled = tmp_path / "scaled"
    pixelized = tmp_path / "pixelized"
    publish_extraction(image, loaded, extracted)
    publish_scale(extracted, loaded, scaled)
    publish_pixelize(scaled, loaded, pixelized)
    metadata_path = pixelized / "stage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first = metadata["frames"][0]["relativePath"]
    metadata["frames"][0]["relativePath"] = metadata["frames"][1]["relativePath"]
    metadata["frames"][1]["relativePath"] = first
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_015"):
        publish_align(pixelized, loaded, tmp_path / "aligned")


def test_undeclared_extra_frame_and_non_rgba_frame_are_rejected(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    (pixelized / "frames" / "extra.png").write_bytes(b"extra")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_014"):
        publish_align(pixelized, load_config(config), tmp_path / "extra-output")
    (pixelized / "frames" / "extra.png").unlink()
    frame_path = pixelized / "frames" / "idle.png"
    with Image.open(frame_path) as image:
        image.convert("RGB").save(frame_path, format="PNG")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_012"):
        publish_align(pixelized, load_config(config), tmp_path / "rgb-output")


def test_foreign_output_and_owned_force_replacement(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    loaded = load_config(config)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("important", encoding="utf-8")
    with pytest.raises(ProcessingError, match="PX_OUTPUT_002"):
        publish_align(pixelized, loaded, foreign)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_align(pixelized, loaded, foreign, force=True)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "important"

    owned = tmp_path / "aligned"
    publish_align(pixelized, loaded, owned)
    (owned / "stale.txt").write_text("stale", encoding="utf-8")
    publish_align(pixelized, loaded, owned, force=True)
    assert not (owned / "stale.txt").exists()


def test_malformed_owned_marker_cannot_authorize_force_replacement(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    loaded = load_config(config)
    output = tmp_path / "aligned"
    publish_align(pixelized, loaded, output)
    sentinel = output / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")
    marker_path = output / ".pixipix-output"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["schemaVersion"] = True
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_align(pixelized, loaded, output, force=True)
    assert sentinel.read_text(encoding="utf-8") == "important"


def test_align_output_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    target = tmp_path / "foreign-target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")
    output = tmp_path / "aligned-link"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_004"):
        publish_align(pixelized, load_config(config), output, force=True)
    assert sentinel.read_text(encoding="utf-8") == "important"


def test_align_publication_failure_restores_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    loaded = load_config(config)
    output = tmp_path / "aligned"
    publish_align(pixelized, loaded, output)
    original = _artifact_bytes(output)
    real_replace = Path.replace

    def fail_new_publication(self: Path, target: Path) -> Path:
        if self.name.startswith(".aligned.pixipix-build-") and target == output:
            raise OSError("simulated rename failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_publication)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005"):
        publish_align(pixelized, loaded, output, force=True)
    assert _artifact_bytes(output) == original
    assert list(tmp_path.glob(".aligned.pixipix-backup-*")) == []


def test_align_staged_write_failure_cleans_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    output = tmp_path / "aligned"

    def fail_write(_path: Path, _pixels: object) -> None:
        raise ProcessingError("PX_TEST", "encode", "simulated failure")

    monkeypatch.setattr(stage_io, "write_png", fail_write)
    with pytest.raises(ProcessingError, match="PX_TEST"):
        publish_align(pixelized, load_config(config), output)
    assert not output.exists()
    assert list(tmp_path.glob(".aligned.pixipix-build-*")) == []


def test_align_publication_setup_failure_is_processing_error(tmp_path: Path) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ProcessingError, match="PX_OUTPUT_005") as captured:
        publish_align(
            pixelized,
            load_config(config),
            blocked_parent / "aligned",
        )
    assert captured.value.exit_code is ExitCode.PROCESSING_FAILURE


def test_prior_png_header_is_validated_before_pixel_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)

    class MismatchedPng:
        format = "PNG"
        mode = "RGBA"
        size = (100_000, 100_000)

        def __enter__(self) -> MismatchedPng:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def load(self) -> None:
            raise AssertionError("pixel decode occurred before header validation")

    monkeypatch.setattr("pixipix.stages.io.Image.open", lambda _path: MismatchedPng())
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_012"):
        publish_align(pixelized, load_config(config), tmp_path / "aligned")


def test_prior_png_decompression_bomb_is_unsupported_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)

    def fail_open(_path: Path) -> object:
        raise Image.DecompressionBombError("simulated oversized PNG")

    monkeypatch.setattr("pixipix.stages.io.Image.open", fail_open)
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_012"):
        publish_align(pixelized, load_config(config), tmp_path / "aligned")


def test_align_repeated_processes_ignore_mtime_and_are_byte_identical(tmp_path: Path) -> None:
    config_text = alignment_config(
        width=3,
        height=2,
        clip_policy="warn",
        offsets="[frame_offsets.signal]\ndx = 1\ndy = 0",
    )
    _, config, _, _, pixelized = _pipeline(tmp_path, config_text=config_text)
    console = Path(sys.executable).with_name("pixipix")
    outputs = (tmp_path / "align-a", tmp_path / "align-b")
    for index, output in enumerate(outputs):
        for frame in (pixelized / "frames").iterdir():
            os.utime(frame, (1_000 + index, 1_000 + index))
        result = subprocess.run(
            [console, "align", pixelized, "--config", config, "--output", output],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert _artifact_bytes(outputs[0]) == _artifact_bytes(outputs[1])
    combined = b"".join(_artifact_bytes(outputs[0]).values())
    assert str(tmp_path).encode() not in combined
    for forbidden in (b"timestamp", b"hostname", b"username"):
        assert forbidden not in combined.lower()


def test_align_cli_help_and_expected_failures_have_stable_exits(tmp_path: Path) -> None:
    console = Path(sys.executable).with_name("pixipix")
    help_result = subprocess.run(
        [console, "align", "--help"], capture_output=True, text=True, check=False
    )
    module_help = subprocess.run(
        [sys.executable, "-m", "pixipix", "align", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == module_help.returncode == 0
    assert "fixed canvas" in help_result.stdout
    assert "fixed canvas" in module_help.stdout

    _, config, _, scaled, pixelized = _pipeline(tmp_path / "valid")
    wrong = subprocess.run(
        [console, "align", scaled, "--config", config, "--output", tmp_path / "wrong"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong.returncode == 3
    assert "PX_STAGE_003" in wrong.stderr
    assert "Traceback" not in wrong.stderr

    invalid_config = tmp_path / "invalid.toml"
    write_config(
        invalid_config,
        alignment_config(offsets="[frame_offsets.idle]\ndx = 0\ndy = 0"),
    )
    invalid = subprocess.run(
        [
            console,
            "align",
            pixelized,
            "--config",
            invalid_config,
            "--output",
            tmp_path / "invalid-output",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "PX_ALIGN_CONFIG_008" in invalid.stderr
    assert "Traceback" not in invalid.stderr

    clipping_root = tmp_path / "clipping"
    _, clipping_config, _, _, clipping_input = _pipeline(
        clipping_root,
        config_text=alignment_config(width=1, height=1, anchor="center"),
    )
    clipping = subprocess.run(
        [
            console,
            "align",
            clipping_input,
            "--config",
            clipping_config,
            "--output",
            clipping_root / "aligned",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert clipping.returncode == 1
    assert "PX_ALIGN_CLIP_001" in clipping.stderr
    assert "idle" in clipping.stderr and "signal" in clipping.stderr
    assert "Traceback" not in clipping.stderr


def test_warning_only_clipping_reaches_resource_refusal_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "project.toml"
    write_config(
        config,
        alignment_config(width=1, height=1, clip_policy="warn")
        + "\n[resources]\nmax_aggregate_input_pixels = 1\n",
    )
    loaded = load_config(config)
    pixelized = tmp_path / "pixelized"
    output = tmp_path / "aligned"
    write_declared_pixelize_stage(pixelized, loaded, ((10, 10), (10, 10)))

    def fail_decode(_validated: object) -> None:
        raise AssertionError("decoder must not run for a resource refusal")

    monkeypatch.setattr("pixipix.stages.align.decode_stage_input", fail_decode)
    with pytest.raises(ResourcePolicyError) as raised:
        publish_align(pixelized, loaded, output)

    assert raised.value.projection.stage == "align"
    assert raised.value.policy == loaded.config.resources
    assert raised.value.projection.aggregate_input_pixels == 200
    assert tuple(finding.kind for finding in raised.value.findings) == ("aggregate_input_pixels",)
    assert capsys.readouterr() == ("", "")
    assert not output.exists()


def test_align_execution_uses_module_decoder_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, _, _, pixelized = _pipeline(tmp_path)
    output = tmp_path / "aligned"

    class PatchedDecoderUsed(Exception):
        pass

    def fail_decode(_validated: object) -> None:
        raise PatchedDecoderUsed

    monkeypatch.setattr("pixipix.stages.align.decode_stage_input", fail_decode)
    with pytest.raises(PatchedDecoderUsed):
        publish_align(pixelized, load_config(config), output)

    assert not output.exists()
