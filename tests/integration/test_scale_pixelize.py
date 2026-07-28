from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pixipix.pipeline.publication as pipeline_publication
from pixipix.config import load_config
from pixipix.errors import (
    ConfigurationError,
    ProcessingError,
    ResourcePolicyError,
    UnsupportedInputError,
)
from pixipix.serialization import write_json
from pixipix.stages.extract import publish_extraction
from pixipix.stages.io import load_stage_input
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale
from tests.helpers import (
    pipeline_config,
    resource_scenario_e,
    resource_scenario_f,
    resource_scenario_g,
    resource_scenario_h,
    transparent_sheet,
    write_config,
    write_declared_extract_stage,
    write_declared_scale_stage,
    write_rgba,
)


def _project(tmp_path: Path, *, config_text: str | None = None) -> tuple[Path, Path]:
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, transparent_sheet())
    write_config(config, config_text or pipeline_config())
    return image, config


def _extract(tmp_path: Path, config_text: str | None = None) -> tuple[Path, Path, Path]:
    image, config = _project(tmp_path, config_text=config_text)
    extracted = tmp_path / "extracted"
    publish_extraction(image, load_config(config), extracted)
    return image, config, extracted


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_reference_width_scale_and_pixelize_pipeline(tmp_path: Path) -> None:
    config_text = pipeline_config(
        scale='mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 2'
    )
    _, config, extracted = _extract(tmp_path, config_text)
    scaled = tmp_path / "scaled"
    pixelized = tmp_path / "pixelized"

    scale_metadata = publish_scale(extracted, load_config(config), scaled)
    pixelize_metadata = publish_pixelize(scaled, load_config(config), pixelized)

    assert scale_metadata.global_factor == 4 / 3
    assert scale_metadata.source_reference_measurement == 3
    assert scale_metadata.exact_target_source_measurement == 4
    assert [
        (item.output_dimensions.width, item.output_dimensions.height)
        for item in scale_metadata.frames
    ] == [
        (4, 4),
        (5, 3),
    ]
    assert [
        (item.logical_output_dimensions.width, item.logical_output_dimensions.height)
        for item in pixelize_metadata.frames
    ] == [
        (2, 2),
        (3, 2),
    ]
    assert (pixelize_metadata.frames[1].top_padding, pixelize_metadata.frames[1].right_padding) == (
        1,
        1,
    )
    assert pixelize_metadata.cell_grid_origin == "bottom-left"
    for root in (scaled, pixelized):
        rendered = (root / "stage.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in rendered
        assert "timestamp" not in rendered.lower()


def test_reference_height_is_exact_and_shared_factor_preserves_geometry(tmp_path: Path) -> None:
    config_text = pipeline_config(
        scale='mode = "reference-frame-height"\nreference_frame = "signal"\ntarget_size = 2'
    )
    _, config, extracted = _extract(tmp_path, config_text)
    metadata = publish_scale(extracted, load_config(config), tmp_path / "scaled")
    assert metadata.global_factor == 2.0
    assert [
        (frame.output_dimensions.width, frame.output_dimensions.height) for frame in metadata.frames
    ] == [
        (6, 6),
        (8, 4),
    ]
    assert {frame.effective_factor for frame in metadata.frames} == {2.0}


def test_explicit_override_is_warned_and_recorded(tmp_path: Path) -> None:
    config_text = pipeline_config(overrides="[frame_overrides.signal]\nscale_multiplier = 0.5")
    _, config, extracted = _extract(tmp_path, config_text)
    metadata = publish_scale(extracted, load_config(config), tmp_path / "scaled")
    assert metadata.frames[0].effective_factor == 1.0
    assert metadata.frames[1].effective_factor == 0.5
    assert metadata.frames[1].output_dimensions.width == 2
    assert metadata.configured_frame_overrides[0].frame_name == "signal"
    assert [warning.code for warning in metadata.warnings] == ["PX_SCALE_OVERRIDE_001"]


def test_pixelize_carries_forward_scale_warnings(tmp_path: Path) -> None:
    config_text = pipeline_config(overrides="[frame_overrides.signal]\nscale_multiplier = 0.5")
    _, config, extracted = _extract(tmp_path, config_text)
    scaled = tmp_path / "scaled"
    publish_scale(extracted, load_config(config), scaled)

    metadata = publish_pixelize(scaled, load_config(config), tmp_path / "pixelized")

    assert [warning.code for warning in metadata.warnings] == ["PX_SCALE_OVERRIDE_001"]


def test_metadata_order_not_filename_order(tmp_path: Path) -> None:
    config_text = pipeline_config(names=("zeta", "alpha"))
    _, config, extracted = _extract(tmp_path, config_text)
    scaled = tmp_path / "scaled"
    metadata = publish_scale(extracted, load_config(config), scaled)
    assert [frame.name for frame in metadata.frames] == ["zeta", "alpha"]
    assert [frame.relative_path.name for frame in metadata.frames] == ["zeta.png", "alpha.png"]


@pytest.mark.parametrize("prior_stage", ["extract", "scale"])
def test_swapped_equal_size_frame_paths_are_rejected(tmp_path: Path, prior_stage: str) -> None:
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    pixels = np.zeros((4, 8, 4), dtype=np.uint8)
    pixels[1:3, 1:3] = (255, 0, 0, 255)
    pixels[1:3, 5:7] = (0, 255, 0, 255)
    write_rgba(image, pixels)
    write_config(config, pipeline_config())
    loaded = load_config(config)
    extracted = tmp_path / "extracted"
    scaled = tmp_path / "scaled"
    publish_extraction(image, loaded, extracted)
    input_root = extracted
    if prior_stage == "scale":
        publish_scale(extracted, loaded, scaled)
        input_root = scaled
    stage_path = input_root / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    first_path = metadata["frames"][0]["relativePath"]
    metadata["frames"][0]["relativePath"] = metadata["frames"][1]["relativePath"]
    metadata["frames"][1]["relativePath"] = first_path
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_015"):
        if prior_stage == "extract":
            publish_scale(extracted, loaded, scaled)
        else:
            publish_pixelize(scaled, loaded, tmp_path / "pixelized")


def test_wrong_prior_stage_is_rejected_without_output(tmp_path: Path) -> None:
    _, config, extracted = _extract(tmp_path)
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_003"):
        publish_pixelize(extracted, load_config(config), tmp_path / "pixelized")
    assert not (tmp_path / "pixelized").exists()


@pytest.mark.parametrize("tamper", ["schema", "duplicate", "unsafe", "dimension", "missing"])
def test_stage_handoff_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    _, _, extracted = _extract(tmp_path)
    stage_path = extracted / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    if tamper == "schema":
        metadata["schemaVersion"] = 99
    elif tamper == "duplicate":
        metadata["frames"][1]["name"] = metadata["frames"][0]["name"]
    elif tamper == "unsafe":
        metadata["frames"][0]["relativePath"] = "../outside.png"
    elif tamper == "dimension":
        metadata["frames"][0]["paddedBounds"]["right"] += 1
    else:
        (extracted / metadata["frames"][0]["relativePath"]).unlink()
    if tamper != "missing":
        stage_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        load_stage_input(extracted, "extract")


def test_declared_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    _, _, extracted = _extract(tmp_path)
    stage_path = extracted / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["sha256"] = "0" * 64
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_013"):
        load_stage_input(extracted, "extract")


@pytest.mark.parametrize(
    ("field", "value"),
    [("globalFactor", 0), ("scaleMode", "independent"), ("priorStage", {})],
)
def test_incomplete_or_invalid_scale_metadata_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    _, config, extracted = _extract(tmp_path)
    scaled = tmp_path / "scaled"
    publish_scale(extracted, load_config(config), scaled)
    stage_path = scaled / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    metadata[field] = value
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_009"):
        load_stage_input(scaled, "scale")


def test_explicit_scale_metadata_rejects_reference_only_fields(tmp_path: Path) -> None:
    _, config, extracted = _extract(tmp_path)
    scaled = tmp_path / "scaled"
    publish_scale(extracted, load_config(config), scaled)
    stage_path = scaled / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "referenceFrame": "idle",
            "sourceReferenceMeasurement": 3,
            "exactTargetSourceMeasurement": 4,
            "logicalTargetSize": 2,
        }
    )
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_009"):
        load_stage_input(scaled, "scale")


@pytest.mark.parametrize("tamper", ["prior-hash", "effective-factor", "unknown-override"])
def test_internally_inconsistent_scale_metadata_is_rejected(tmp_path: Path, tamper: str) -> None:
    _, config, extracted = _extract(tmp_path)
    scaled = tmp_path / "scaled"
    publish_scale(extracted, load_config(config), scaled)
    stage_path = scaled / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    if tamper == "prior-hash":
        metadata["priorStage"]["effectiveConfigSha256"] = "0" * 64
    elif tamper == "effective-factor":
        metadata["frames"][0]["effectiveFactor"] = 2.0
    else:
        metadata["configuredFrameOverrides"] = [{"frameName": "missing", "scaleMultiplier": 1.0}]
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_009"):
        load_stage_input(scaled, "scale")


def test_reference_scale_metadata_requires_exact_target_identity(tmp_path: Path) -> None:
    config_text = pipeline_config(
        scale='mode = "reference-frame-width"\nreference_frame = "idle"\ntarget_size = 2'
    )
    _, config, extracted = _extract(tmp_path, config_text)
    scaled = tmp_path / "scaled"
    publish_scale(extracted, load_config(config), scaled)
    stage_path = scaled / "stage.json"
    metadata = json.loads(stage_path.read_text(encoding="utf-8"))
    metadata["exactTargetSourceMeasurement"] += 1
    stage_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_009"):
        load_stage_input(scaled, "scale")


def test_symlink_input_parent_is_rejected(tmp_path: Path) -> None:
    _, _, extracted = _extract(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(UnsupportedInputError, match="PX_STAGE_001"):
        load_stage_input(linked / extracted.name, "extract")


def test_foreign_output_refused_and_owned_force_replaces(tmp_path: Path) -> None:
    _, config, extracted = _extract(tmp_path)
    loaded = load_config(config)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("important", encoding="utf-8")
    with pytest.raises(ProcessingError, match="PX_OUTPUT_002"):
        publish_scale(extracted, loaded, foreign)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_003"):
        publish_scale(extracted, loaded, foreign, force=True)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "important"

    owned = tmp_path / "scaled"
    publish_scale(extracted, loaded, owned)
    (owned / "stale.txt").write_text("stale", encoding="utf-8")
    publish_scale(extracted, loaded, owned, force=True)
    assert not (owned / "stale.txt").exists()


def test_scale_and_pixelize_repeated_processes_are_byte_identical(tmp_path: Path) -> None:
    _, config, extracted = _extract(tmp_path)
    console = Path(sys.executable).with_name("pixipix")
    scale_roots = (tmp_path / "scale-a", tmp_path / "scale-b")
    pixel_roots = (tmp_path / "pixel-a", tmp_path / "pixel-b")
    for scaled, pixelized in zip(scale_roots, pixel_roots, strict=True):
        scale_result = subprocess.run(
            [console, "scale", extracted, "--config", config, "--output", scaled],
            capture_output=True,
            text=True,
            check=False,
        )
        assert scale_result.returncode == 0, scale_result.stderr
        pixel_result = subprocess.run(
            [console, "pixelize", scaled, "--config", config, "--output", pixelized],
            capture_output=True,
            text=True,
            check=False,
        )
        assert pixel_result.returncode == 0, pixel_result.stderr
    assert _artifact_bytes(scale_roots[0]) == _artifact_bytes(scale_roots[1])
    assert _artifact_bytes(pixel_roots[0]) == _artifact_bytes(pixel_roots[1])


def test_new_stage_publication_failure_restores_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, extracted = _extract(tmp_path)
    loaded = load_config(config)
    output = tmp_path / "scaled"
    publish_scale(extracted, loaded, output)
    original = _artifact_bytes(output)
    real_replace = Path.replace

    def fail_new_publication(self: Path, target: Path) -> Path:
        if self.name.startswith(".scaled.pixipix-build-") and target == output:
            raise OSError("simulated rename failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_publication)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_005"):
        publish_scale(extracted, loaded, output, force=True)
    assert _artifact_bytes(output) == original
    assert list(tmp_path.glob(".scaled.pixipix-backup-*")) == []


def test_new_stage_failed_staging_cleans_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, extracted = _extract(tmp_path)
    output = tmp_path / "scaled"

    def fail_write(_path: Path, _pixels: object) -> None:
        raise ProcessingError("PX_TEST", "encode", "simulated failure")

    monkeypatch.setattr(pipeline_publication, "write_png", fail_write)
    with pytest.raises(ProcessingError, match="PX_TEST"):
        publish_scale(extracted, load_config(config), output)
    assert not output.exists()
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []


@pytest.mark.parametrize(
    ("scenario", "stage", "projection", "finding_kinds"),
    [
        (
            resource_scenario_e,
            "pixelize",
            (67_043_344, 16_760_836, 469_303_408),
            ("aggregate_input_pixels",),
        ),
        (
            resource_scenario_f,
            "scale",
            (15_000_001, 60_000_001, 361_000_008),
            ("aggregate_output_pixels",),
        ),
        (
            resource_scenario_g,
            "scale",
            (16_777_216, 16_777_216, 1_409_286_144),
            ("modeled_peak_live_bytes",),
        ),
        (
            resource_scenario_h,
            "scale",
            (100_000_000, 144_000_000, 1_076_640_000),
            (
                "aggregate_input_pixels",
                "aggregate_output_pixels",
                "modeled_peak_live_bytes",
            ),
        ),
    ],
)
def test_metadata_only_resource_scenarios_refuse_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scenario: object,
    stage: str,
    projection: tuple[int, int, int],
    finding_kinds: tuple[str, ...],
) -> None:
    build_scenario = scenario
    assert callable(build_scenario)
    config, input_root, output = build_scenario(tmp_path)
    loaded = load_config(config)

    def fail_decode(_validated: object) -> None:
        raise AssertionError("decoder must not run for refused declarations")

    if stage == "pixelize":
        monkeypatch.setattr("pixipix.stages.pixelize.decode_stage_input", fail_decode)
    else:
        monkeypatch.setattr("pixipix.stages.scale.api.decode_stage_input", fail_decode)

    def operation() -> object:
        if stage == "pixelize":
            return publish_pixelize(input_root, loaded, output)
        return publish_scale(input_root, loaded, output)

    with pytest.raises(ResourcePolicyError) as raised:
        operation()

    error = raised.value
    assert error.projection.stage == stage
    assert error.policy == loaded.config.resources
    assert (
        error.projection.aggregate_input_pixels,
        error.projection.aggregate_output_pixels,
        error.projection.modeled_peak_live_bytes,
    ) == projection
    assert tuple(finding.kind for finding in error.findings) == finding_kinds
    assert capsys.readouterr() == ("", "")
    assert not output.exists()


def test_admitted_metadata_with_malformed_png_reaches_decoder(tmp_path: Path) -> None:
    config = tmp_path / "admitted.toml"
    write_config(
        config,
        pipeline_config(
            names=("tiny",),
            scale='mode = "explicit-factor"\nfactor = 1.0',
        ),
    )
    loaded = load_config(config)
    input_root = tmp_path / "admitted-extract"
    output = tmp_path / "admitted-output"
    write_declared_extract_stage(input_root, loaded, ((1, 1),))

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_011"):
        publish_scale(input_root, loaded, output)

    assert not output.exists()


def test_scale_per_image_projection_precedes_aggregate_policy(tmp_path: Path) -> None:
    config = tmp_path / "scale-per-image.toml"
    write_config(
        config,
        pipeline_config(
            names=("large",),
            scale='mode = "explicit-factor"\nfactor = 2.0',
        )
        + "\n[resources]\nmax_aggregate_input_pixels = 1\n",
    )
    loaded = load_config(config)
    input_root = tmp_path / "extract"
    write_declared_extract_stage(input_root, loaded, ((3000, 3000),))

    with pytest.raises(ProcessingError, match="PX_SCALE_002"):
        publish_scale(input_root, loaded, tmp_path / "scaled")


def test_pixelize_per_image_projection_precedes_aggregate_policy(tmp_path: Path) -> None:
    config = tmp_path / "pixelize-per-image.toml"
    write_config(
        config,
        pipeline_config(
            names=("large",),
            scale='mode = "explicit-factor"\nfactor = 1.0',
            pixelize=(
                "source_cell_size = 3\n"
                'representative = "alpha-weighted-majority"\n'
                'alpha_policy = "binary"\n'
                "alpha_threshold = 128\n"
                'remainder_policy = "pad-transparent"'
            ),
        )
        + "\n[resources]\nmax_aggregate_input_pixels = 1\n",
    )
    loaded = load_config(config)
    input_root = tmp_path / "scaled"
    write_declared_scale_stage(
        input_root,
        loaded,
        ((4096, 4094),),
        ((4096, 4094),),
        factor=1.0,
    )

    with pytest.raises(ProcessingError, match="PX_PIXELIZE_002"):
        publish_pixelize(input_root, loaded, tmp_path / "pixelized")


def test_mixed_resource_policy_lineage_fails_identity_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_config = tmp_path / "original.toml"
    changed_config = tmp_path / "changed.toml"
    write_config(original_config, pipeline_config(names=("tiny",)))
    write_config(
        changed_config,
        pipeline_config(names=("tiny",))
        + "\n[resources]\nmax_aggregate_output_pixels = 60000001\n",
    )
    original = load_config(original_config)
    changed = load_config(changed_config)
    input_root = tmp_path / "extract"
    write_declared_extract_stage(input_root, original, ((1, 1),))

    def fail_decode(_validated: object) -> None:
        raise AssertionError("decoder must not run for an identity refusal")

    monkeypatch.setattr("pixipix.stages.scale.api.decode_stage_input", fail_decode)
    with pytest.raises(ConfigurationError, match="PX_SCALE_CONFIG_002"):
        publish_scale(input_root, changed, tmp_path / "scaled")


def test_over_budget_dimensions_do_not_override_schema_precedence(tmp_path: Path) -> None:
    config, input_root, output = resource_scenario_e(tmp_path)
    metadata_path = input_root / "stage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schemaVersion"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UnsupportedInputError, match="PX_STAGE_005"):
        publish_pixelize(input_root, load_config(config), output)

    assert not output.exists()


def test_raised_byte_budget_admits_scenario_g_to_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "raised.toml"
    write_config(
        config,
        pipeline_config(
            names=("large",),
            scale='mode = "explicit-factor"\nfactor = 1.0',
        )
        + "\n[resources]\nmax_modeled_peak_live_bytes = 1500000000\n",
    )
    loaded = load_config(config)
    input_root = tmp_path / "extract"
    write_declared_extract_stage(input_root, loaded, ((4096, 4096),))

    class DecodeReached(RuntimeError):
        pass

    def mark_decode(_validated: object) -> None:
        raise DecodeReached

    monkeypatch.setattr("pixipix.stages.scale.api.decode_stage_input", mark_decode)
    with pytest.raises(DecodeReached):
        publish_scale(input_root, loaded, tmp_path / "scaled")


def test_internal_staged_marker_validation_is_a_processing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, extracted = _extract(tmp_path)
    output = tmp_path / "scaled"

    def corrupt_marker(path: Path, value: object) -> None:
        write_json(path, value)
        if path.name == ".pixipix-output":
            path.write_text("{", encoding="utf-8")

    monkeypatch.setattr(pipeline_publication, "write_json", corrupt_marker)
    with pytest.raises(ProcessingError, match="PX_OUTPUT_006"):
        publish_scale(extracted, load_config(config), output)
    assert not output.exists()
    assert list(tmp_path.glob(".scaled.pixipix-build-*")) == []


def test_output_frames_are_rgba_and_logical_resolution(tmp_path: Path) -> None:
    _, config, extracted = _extract(tmp_path)
    scaled = tmp_path / "scaled"
    pixelized = tmp_path / "pixelized"
    publish_scale(extracted, load_config(config), scaled)
    publish_pixelize(scaled, load_config(config), pixelized)
    with Image.open(scaled / "frames" / "idle.png") as scale_image:
        assert scale_image.mode == "RGBA"
        assert scale_image.size == (3, 3)
    with Image.open(pixelized / "frames" / "idle.png") as logical:
        assert logical.mode == "RGBA"
        assert logical.size == (2, 2)
        assert np.asarray(logical).shape == (2, 2, 4)
