from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

import pixipix.cli as cli_module
import pixipix.pipeline.run as pipeline_run
from pixipix.cli import app
from pixipix.config import LoadedConfig, load_config
from pixipix.errors import ProcessingError, ResourcePolicyError
from pixipix.models import (
    AlignmentStageMetadata,
    ExtractionResult,
    PixelizeStageMetadata,
    ScaleStageMetadata,
)
from pixipix.pipeline.publication import validate_stage_output_target
from pixipix.pipeline.run import PipelineRunResult, run_pipeline
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.io import load_stage_input
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale
from tests.helpers import (
    alignment_config,
    pipeline_config,
    transparent_sheet,
    write_config,
    write_rgba,
)

RUN_STAGES = ("extract", "scale", "pixelize", "align")
runner = CliRunner()


def _fixture(tmp_path: Path, *, config_text: str | None = None) -> tuple[Path, Path, LoadedConfig]:
    source = tmp_path / "source.png"
    config = tmp_path / "pixipix.toml"
    write_rgba(source, transparent_sheet())
    write_config(config, config_text or alignment_config())
    return source, config, load_config(config)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, content in _tree_bytes(root).items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


def _metadata(root: Path, stage: str) -> dict[str, object]:
    value = json.loads((root / stage / "stage.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_metadata(root: Path, stage: str, value: dict[str, object]) -> None:
    (root / stage / "stage.json").write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _residue(parent: Path, output_name: str) -> tuple[str, ...]:
    prefixes = (
        f".{output_name}.pixipix-run-build-",
        f".{output_name}.pixipix-run-backup-",
    )
    return tuple(sorted(path.name for path in parent.iterdir() if path.name.startswith(prefixes)))


def test_python_run_api_uses_exact_stage_order_one_config_and_inspectable_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    order: list[str] = []

    def extract(
        input_path: Path, actual: LoadedConfig, stage_output: Path, *, force: bool = False
    ) -> ExtractionResult:
        order.append("extract")
        assert actual is loaded
        return publish_extraction(input_path, actual, stage_output, force=force)

    def scale(
        input_dir: Path, actual: LoadedConfig, stage_output: Path, *, force: bool = False
    ) -> ScaleStageMetadata:
        order.append("scale")
        assert actual is loaded
        return publish_scale(input_dir, actual, stage_output, force=force)

    def pixelize(
        input_dir: Path, actual: LoadedConfig, stage_output: Path, *, force: bool = False
    ) -> PixelizeStageMetadata:
        order.append("pixelize")
        assert actual is loaded
        return publish_pixelize(input_dir, actual, stage_output, force=force)

    def align(
        input_dir: Path, actual: LoadedConfig, stage_output: Path, *, force: bool = False
    ) -> AlignmentStageMetadata:
        order.append("align")
        assert actual is loaded
        return publish_align(input_dir, actual, stage_output, force=force)

    monkeypatch.setattr(pipeline_run, "publish_extraction", extract)
    monkeypatch.setattr(pipeline_run, "publish_scale", scale)
    monkeypatch.setattr(pipeline_run, "publish_pixelize", pixelize)
    monkeypatch.setattr(pipeline_run, "publish_align", align)

    result = run_pipeline(source, loaded, output)

    assert isinstance(result, PipelineRunResult)
    assert result.output_root == output
    assert order == list(RUN_STAGES)
    assert result.warnings == result.align.warnings
    assert {path.name for path in output.iterdir()} == {".pixipix-run", *RUN_STAGES}
    assert load_stage_input(output / "extract", "extract").identity.effective_config_sha256 == (
        loaded.effective_config_sha256
    )
    assert load_stage_input(output / "scale", "scale").identity.effective_config_sha256 == (
        loaded.effective_config_sha256
    )
    assert load_stage_input(output / "pixelize", "pixelize").identity.effective_config_sha256 == (
        loaded.effective_config_sha256
    )
    validate_stage_output_target(output / "align", "align", force=True)
    assert _residue(tmp_path, output.name) == ()


def test_manual_sequence_and_run_have_exact_stage_tree_parity_and_deterministic_rerun(
    tmp_path: Path,
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    manual = tmp_path / "manual"
    manual.mkdir()
    publish_extraction(source, loaded, manual / "extract")
    publish_scale(manual / "extract", loaded, manual / "scale")
    publish_pixelize(manual / "scale", loaded, manual / "pixelize")
    publish_align(manual / "pixelize", loaded, manual / "align")

    first = tmp_path / "run-first"
    second = tmp_path / "run-second"
    first_result = run_pipeline(source, loaded, first)
    second_result = run_pipeline(source, loaded, second)

    for stage in RUN_STAGES:
        manual_tree = _tree_bytes(manual / stage)
        assert _tree_bytes(first / stage) == manual_tree
        assert _tree_bytes(second / stage) == manual_tree
        assert _tree_sha256(first / stage) == _tree_sha256(manual / stage)
        assert _tree_sha256(second / stage) == _tree_sha256(manual / stage)
    assert first_result.warnings == second_result.warnings
    assert _tree_bytes(first) == _tree_bytes(second)


def test_stage_tree_parity_evidence_detects_one_byte_mutation(tmp_path: Path) -> None:
    source, _config, loaded = _fixture(tmp_path)
    manual = tmp_path / "manual"
    run = tmp_path / "run"
    manual.mkdir()
    publish_extraction(source, loaded, manual / "extract")
    publish_scale(manual / "extract", loaded, manual / "scale")
    publish_pixelize(manual / "scale", loaded, manual / "pixelize")
    publish_align(manual / "pixelize", loaded, manual / "align")
    run_pipeline(source, loaded, run)
    before = _tree_sha256(run / "scale")

    metadata_path = run / "scale" / "stage.json"
    metadata_path.write_bytes(
        metadata_path.read_bytes().replace(b'"stage": "scale"', b'"stage": "scalz"')
    )

    assert _tree_sha256(run / "scale") != before
    assert _tree_bytes(run / "scale") != _tree_bytes(manual / "scale")


@pytest.mark.parametrize("failed_stage", RUN_STAGES)
def test_each_stage_failure_publishes_no_run_and_cleans_temporary_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "failed-run"

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise ProcessingError(f"PX_TEST_{failed_stage.upper()}", failed_stage, "injected failure")

    publisher = f"publish_{'extraction' if failed_stage == 'extract' else failed_stage}"
    monkeypatch.setattr(pipeline_run, publisher, fail)

    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output)

    assert captured.value.code == f"PX_TEST_{failed_stage.upper()}"
    assert not output.exists()
    assert _residue(tmp_path, output.name) == ()


def test_staged_run_validation_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(pipeline_run, "_valid_completed_run", lambda _root: False)

    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output)

    assert captured.value.code == "PX_OUTPUT_006"
    assert not output.exists()
    assert _residue(tmp_path, output.name) == ()


def test_run_cleanup_never_removes_an_unrelated_similar_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    unrelated = tmp_path / ".run.pixipix-run-build-user-owned"
    unrelated.mkdir()
    keep = unrelated / "keep.txt"
    keep.write_text("important", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise ProcessingError("PX_TEST_EXTRACT", "extract", "injected failure")

    monkeypatch.setattr(pipeline_run, "publish_extraction", fail)
    with pytest.raises(ProcessingError):
        run_pipeline(source, loaded, output)

    assert keep.read_text(encoding="utf-8") == "important"
    assert unrelated.is_dir()


def test_existing_valid_run_survives_failed_force_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)
    before = _tree_bytes(output)

    def fail(
        _input_dir: Path, _loaded: LoadedConfig, _output: Path, *, force: bool = False
    ) -> NoReturn:
        del force
        raise ProcessingError("PX_TEST_ALIGN", "align", "injected failure")

    monkeypatch.setattr(pipeline_run, "publish_align", fail)
    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output, force=True)

    assert captured.value.code == "PX_TEST_ALIGN"
    assert _tree_bytes(output) == before
    assert _residue(tmp_path, output.name) == ()


def test_top_level_atomic_replacement_failure_restores_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)
    before = _tree_bytes(output)
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if path.name.startswith(f".{output.name}.pixipix-run-build-") and target == output:
            raise OSError("injected final publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output, force=True)

    assert captured.value.code == "PX_OUTPUT_005"
    assert _tree_bytes(output) == before
    assert _residue(tmp_path, output.name) == ()


def test_top_level_backup_move_failure_preserves_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)
    before = _tree_bytes(output)
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if path == output and target.name == "previous":
            raise OSError("injected backup move failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output, force=True)

    assert captured.value.code == "PX_OUTPUT_005"
    assert _tree_bytes(output) == before
    assert _residue(tmp_path, output.name) == ()


def test_failed_top_level_restore_retains_recoverable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)
    before = _tree_bytes(output)
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if path.name.startswith(f".{output.name}.pixipix-run-build-") and target == output:
            raise OSError("injected publication failure")
        if path.name == "previous" and target == output:
            raise OSError("injected restore failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output, force=True)

    backups = tuple(tmp_path.glob(f".{output.name}.pixipix-run-backup-*"))
    assert captured.value.code == "PX_OUTPUT_005"
    assert not output.exists()
    assert len(backups) == 1
    assert _tree_bytes(backups[0] / "previous") == before
    assert tuple(tmp_path.glob(f".{output.name}.pixipix-run-build-*")) == ()


def test_run_force_replacement_requires_valid_run_ownership(tmp_path: Path) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)

    with pytest.raises(ProcessingError) as without_force:
        run_pipeline(source, loaded, output)
    assert without_force.value.code == "PX_OUTPUT_002"

    original = _tree_bytes(output)
    pixels = transparent_sheet()
    pixels[1, 1] = (200, 10, 20, 255)
    write_rgba(source, pixels)
    run_pipeline(source, loaded, output, force=True)
    assert _tree_bytes(output) != original

    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ProcessingError) as unowned_error:
        run_pipeline(source, loaded, unowned, force=True)
    assert unowned_error.value.code == "PX_OUTPUT_003"
    assert (unowned / "notes.txt").read_text(encoding="utf-8") == "keep"

    (output / "align" / "stage.json").unlink()
    with pytest.raises(ProcessingError) as malformed:
        run_pipeline(source, loaded, output, force=True)
    assert malformed.value.code == "PX_OUTPUT_003"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-run-marker",
        "marker-only",
        "missing-stage",
        "extra-user-file",
        "unsuccessful-stage",
        "mismatched-effective-identity",
        "incoherent-prior-stage",
        "missing-frame",
        "malformed-extract-metadata",
        "malformed-warning-lineage",
        "incoherent-warning-stage",
        "incoherent-align-handoff",
    ),
)
def test_force_rejects_every_malformed_complete_run_category(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)

    if mutation == "missing-run-marker":
        (output / ".pixipix-run").unlink()
    elif mutation == "marker-only":
        for stage in RUN_STAGES:
            shutil.rmtree(output / stage)
    elif mutation == "missing-stage":
        shutil.rmtree(output / "pixelize")
    elif mutation == "extra-user-file":
        (output / "keep.txt").write_text("important", encoding="utf-8")
    elif mutation == "unsuccessful-stage":
        metadata = _metadata(output, "align")
        metadata["status"] = "failed"
        _write_metadata(output, "align", metadata)
    elif mutation == "mismatched-effective-identity":
        metadata = _metadata(output, "align")
        metadata["effectiveConfigSha256"] = "0" * 64
        _write_metadata(output, "align", metadata)
    elif mutation == "incoherent-prior-stage":
        metadata = _metadata(output, "pixelize")
        prior = metadata["priorStage"]
        assert isinstance(prior, dict)
        prior["stage"] = "extract"
        _write_metadata(output, "pixelize", metadata)
    elif mutation == "missing-frame":
        (output / "scale" / "frames" / "idle.png").unlink()
    elif mutation == "malformed-extract-metadata":
        metadata = _metadata(output, "extract")
        del metadata["background"]
        _write_metadata(output, "extract", metadata)
    elif mutation == "malformed-warning-lineage":
        metadata = _metadata(output, "align")
        metadata["warnings"] = {}
        _write_metadata(output, "align", metadata)
    elif mutation == "incoherent-warning-stage":
        metadata = _metadata(output, "align")
        warnings = metadata["warnings"]
        assert isinstance(warnings, list)
        warnings.append({"code": "PX_TEST_WARNING", "stage": "extract", "message": "wrong owner"})
        _write_metadata(output, "align", metadata)
    elif mutation == "incoherent-align-handoff":
        metadata = _metadata(output, "align")
        frames = metadata["frames"]
        assert isinstance(frames, list) and isinstance(frames[0], dict)
        width = frames[0]["inputWidth"]
        assert type(width) is int
        frames[0]["inputWidth"] = width + 1
        _write_metadata(output, "align", metadata)
    else:
        raise AssertionError(f"unhandled mutation {mutation}")

    before = _tree_bytes(output)
    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output, force=True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert _tree_bytes(output) == before
    assert _residue(tmp_path, output.name) == ()


def test_stage_owned_child_cannot_authorize_run_root_replacement(tmp_path: Path) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, loaded, output)
    child = output / "align"
    before = _tree_bytes(child)

    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, child, force=True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert _tree_bytes(child) == before


def test_empty_and_unsafe_run_targets_follow_publication_path_policy(tmp_path: Path) -> None:
    source, _config, loaded = _fixture(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    run_pipeline(source, loaded, empty)
    assert (empty / ".pixipix-run").is_file()

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProcessingError) as linked_error:
        run_pipeline(source, loaded, linked, force=True)
    assert linked_error.value.code == "PX_OUTPUT_004"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ProcessingError) as parent_error:
        run_pipeline(source, loaded, linked_parent / "run")
    assert parent_error.value.code == "PX_OUTPUT_004"
    assert tuple(outside.iterdir()) == ()
    assert tuple(real_parent.iterdir()) == ()


def test_destination_created_before_final_revalidation_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _config, loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    original_validator = pipeline_run._valid_completed_run
    staged_validations = 0

    def validate(root: Path) -> bool:
        nonlocal staged_validations
        result = original_validator(root)
        if root != output and result:
            staged_validations += 1
            output.mkdir()
            (output / "keep.txt").write_text("important", encoding="utf-8")
        return result

    monkeypatch.setattr(pipeline_run, "_valid_completed_run", validate)

    with pytest.raises(ProcessingError) as captured:
        run_pipeline(source, loaded, output)

    assert staged_validations == 1
    assert captured.value.code == "PX_OUTPUT_002"
    assert (output / "keep.txt").read_text(encoding="utf-8") == "important"
    assert _residue(tmp_path, output.name) == ()


def test_stage_resource_refusal_preserves_run_publish_or_nothing(tmp_path: Path) -> None:
    config_text = alignment_config() + (
        "\n[resources]\n"
        "max_aggregate_input_pixels = 50000000\n"
        "max_aggregate_output_pixels = 1\n"
        "max_modeled_peak_live_bytes = 1000000000\n"
    )
    source, _config, loaded = _fixture(tmp_path, config_text=config_text)
    output = tmp_path / "run"

    with pytest.raises(ResourcePolicyError) as captured:
        run_pipeline(source, loaded, output)

    assert captured.value.stage == "extract"
    assert not output.exists()
    assert _residue(tmp_path, output.name) == ()


@pytest.mark.parametrize(
    ("expected_stage", "config_text"),
    (
        (
            "extract",
            alignment_config() + "\n[resources]\nmax_aggregate_output_pixels = 1\n",
        ),
        (
            "scale",
            pipeline_config(
                scale='mode = "explicit-factor"\nfactor = 2.0',
                output=(
                    'frame_width = 8\nframe_height = 8\nanchor = "bottom-center"\n'
                    'clip_policy = "error"'
                ),
            )
            + "\n[resources]\nmax_aggregate_output_pixels = 30\n",
        ),
        (
            "align",
            alignment_config() + "\n[resources]\nmax_aggregate_output_pixels = 100\n",
        ),
    ),
)
def test_reachable_stage_resource_refusals_preserve_existing_run(
    tmp_path: Path,
    expected_stage: str,
    config_text: str,
) -> None:
    source, config, base_loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    run_pipeline(source, base_loaded, output)
    before = _tree_bytes(output)
    write_config(config, config_text)

    with pytest.raises(ResourcePolicyError) as captured:
        run_pipeline(source, load_config(config), output, force=True)

    assert captured.value.stage == expected_stage
    assert _tree_bytes(output) == before
    assert _residue(tmp_path, output.name) == ()


def test_cli_run_help_success_warning_order_and_failure_taxonomy(tmp_path: Path) -> None:
    config_text = pipeline_config(
        scale='mode = "explicit-factor"\nfactor = 1.0',
        pixelize=(
            'source_cell_size = 2\nrepresentative = "alpha-weighted-majority"\n'
            'alpha_policy = "binary"\nalpha_threshold = 128\n'
            'remainder_policy = "crop-with-warning"'
        ),
        overrides="[frame_overrides.signal]\nscale_multiplier = 1.0",
        output=(
            'frame_width = 8\nframe_height = 8\nanchor = "bottom-center"\nclip_policy = "error"'
        ),
        offsets="[frame_offsets.signal]\ndx = 1\ndy = 0",
    )
    source, config, _loaded = _fixture(tmp_path, config_text=config_text)
    output = tmp_path / "run"

    help_result = runner.invoke(app, ["run", "--help"])
    result = runner.invoke(
        app,
        ["run", str(source), "--config", str(config), "--output", str(output)],
    )

    assert help_result.exit_code == 0
    assert "Extract" in help_result.stdout
    assert "Scale" in help_result.stdout
    assert "Pixelize" in help_result.stdout
    assert "Align" in help_result.stdout
    assert result.exit_code == 0
    assert result.stdout == f"completed run with 2 frame(s) at {output}\n"
    warning_lines = result.stderr_bytes.splitlines()
    assert len(warning_lines) == 3
    assert b"[scale] PX_SCALE_OVERRIDE_001" in warning_lines[0]
    assert b"[pixelize] PX_PIXELIZE_CROP_001" in warning_lines[1]
    assert b"[align] PX_ALIGN_OFFSET_001" in warning_lines[2]
    assert result.exit_code == 0
    assert [
        item.stage for item in run_pipeline(source, load_config(config), tmp_path / "api").warnings
    ] == [
        "scale",
        "pixelize",
        "align",
    ]

    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text("not = [valid", encoding="utf-8")
    config_failure = runner.invoke(
        app,
        [
            "run",
            str(source),
            "--config",
            str(invalid_config),
            "--output",
            str(tmp_path / "config-failure"),
        ],
    )
    missing_input = runner.invoke(
        app,
        [
            "run",
            str(tmp_path / "missing.png"),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "input-failure"),
        ],
    )
    processing_failure = runner.invoke(
        app,
        ["run", str(source), "--config", str(config), "--output", str(output)],
    )
    assert processing_failure.exit_code == 1
    assert processing_failure.stdout == ""
    assert b"PX_OUTPUT_002" in processing_failure.stderr_bytes
    assert config_failure.exit_code == 2
    assert missing_input.exit_code == 3
    assert b"Traceback" not in (
        processing_failure.stderr_bytes + config_failure.stderr_bytes + missing_input.stderr_bytes
    )
    assert not (tmp_path / "config-failure").exists()
    assert not (tmp_path / "input-failure").exists()


def test_cli_run_delegates_once_to_python_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config, _loaded = _fixture(tmp_path)
    output = tmp_path / "run"
    calls: list[tuple[Path, Path, bool]] = []
    authoritative = run_pipeline

    def delegated(
        input_path: Path,
        loaded: LoadedConfig,
        destination: Path,
        *,
        force: bool = False,
    ) -> PipelineRunResult:
        calls.append((input_path, destination, force))
        return authoritative(input_path, loaded, destination, force=force)

    monkeypatch.setattr(cli_module, "run_pipeline", delegated)
    result = runner.invoke(
        app,
        ["run", str(source), "--config", str(config), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert calls == [(source, output, False)]
    assert (output / "align" / "stage.json").is_file()


def test_cli_run_unexpected_failure_uses_internal_error_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config, _loaded = _fixture(tmp_path)
    output = tmp_path / "run"

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(cli_module, "run_pipeline", fail)
    result = runner.invoke(
        app,
        ["run", str(source), "--config", str(config), "--output", str(output)],
    )

    assert result.exit_code == 4
    assert result.stdout == ""
    assert result.stderr_bytes == (
        b"PX_INTERNAL_001 [internal] unexpected internal error. Remediation: report the defect; "
        b"tracebacks are intentionally disabled in this milestone.\n"
    )
    assert b"private failure detail" not in result.stderr_bytes
    assert not output.exists()
    assert _residue(tmp_path, output.name) == ()


def test_public_run_artifacts_do_not_leak_temporary_or_absolute_paths(tmp_path: Path) -> None:
    distinctive = tmp_path / "machine-private-parent"
    distinctive.mkdir()
    source, _config, loaded = _fixture(distinctive)
    output = distinctive / "public-run"

    run_pipeline(source, loaded, output)

    public_bytes = b"\n".join(_tree_bytes(output).values())
    assert b"pixipix-run-build" not in public_bytes
    assert b"pixipix-run-backup" not in public_bytes
    assert str(distinctive).encode() not in public_bytes
    assert (output / ".pixipix-run").read_bytes() == (
        b'{\n  "kind": "run",\n  "owner": "pixipix",\n  "schemaVersion": 1\n}\n'
    )
