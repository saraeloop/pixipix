"""Pixelize-stage orchestration."""

from __future__ import annotations

from pathlib import Path

from pixipix.config import LoadedConfig
from pixipix.models import PixelizeStageMetadata
from pixipix.pipeline.input import decode_stage_input, validate_stage_input
from pixipix.pipeline.publication import publish_stage_output, validate_stage_output_target
from pixipix.resources import enforce_resource_policy

from .execution import pixelize_stage
from .planning import project_pixelize_stage


def publish_pixelize(
    input_dir: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> PixelizeStageMetadata:
    validate_stage_output_target(output, "pixelize", force=force)
    validated = validate_stage_input(input_dir, "scale")
    plan = project_pixelize_stage(validated, loaded)
    enforce_resource_policy(plan.projection, loaded.config.resources)
    stage = decode_stage_input(validated)
    run = pixelize_stage(stage, loaded, plan)
    publish_stage_output(output, "pixelize", run.metadata, run.frame_images, force=force)
    return run.metadata
