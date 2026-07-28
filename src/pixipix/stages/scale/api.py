"""Imperative orchestration for the scale stage."""

from __future__ import annotations

from pathlib import Path

from pixipix.config import LoadedConfig
from pixipix.models import ScaleStageMetadata
from pixipix.pipeline.input import decode_stage_input, validate_stage_input
from pixipix.pipeline.publication import publish_stage_output, validate_stage_output_target
from pixipix.resources import enforce_resource_policy

from .execution import scale_stage
from .planning import project_scale_stage


def publish_scale(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> ScaleStageMetadata:
    validate_stage_output_target(output, "scale", force=force)
    validated = validate_stage_input(input_dir, "extract")
    plan = project_scale_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = scale_stage(stage, loaded, plan)
    publish_stage_output(output, "scale", run.metadata, run.frame_images, force=force)
    return run.metadata
