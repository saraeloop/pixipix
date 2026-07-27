"""Imperative orchestration for the align stage."""

from __future__ import annotations

from pathlib import Path

from pixipix.config import LoadedConfig
from pixipix.models import AlignmentStageMetadata
from pixipix.resources import enforce_resource_policy
from pixipix.stages.align.execution import align_stage
from pixipix.stages.align.planning import project_align_stage
from pixipix.stages.io import (
    decode_stage_input,
    publish_stage_output,
    validate_stage_input,
    validate_stage_output_target,
)


def publish_align(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> AlignmentStageMetadata:
    validate_stage_output_target(output, "align", force=force)
    validated = validate_stage_input(input_dir, "pixelize")
    plan = project_align_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = align_stage(stage, loaded, plan)
    publish_stage_output(output, "align", run.metadata, run.frame_images, force=force)
    return run.metadata
