"""Authoritative whole-pipeline application orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pixipix.config import LoadedConfig
from pixipix.errors import UnsupportedInputError
from pixipix.models import (
    AlignmentStageMetadata,
    ExtractionResult,
    PixelizeStageMetadata,
    ProcessingWarning,
    ScaleStageMetadata,
)
from pixipix.pipeline.artifacts import StageName, _read_json_object
from pixipix.pipeline.input import ValidatedStageInput, validate_stage_input
from pixipix.pipeline.publication import _valid_owned_output, publish_run_output
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.extract.metadata import _valid_owned_extract_metadata
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale

RUN_STAGES: tuple[StageName, ...] = ("extract", "scale", "pixelize", "align")
type DownstreamStage = Literal["scale", "pixelize", "align"]


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    output_root: Path
    extract: ExtractionResult
    scale: ScaleStageMetadata
    pixelize: PixelizeStageMetadata
    align: AlignmentStageMetadata
    warnings: tuple[ProcessingWarning, ...]


def _valid_warning_lineage(metadata: dict[StageName, dict[str, object]]) -> bool:
    previous: list[object] = []
    for stage in RUN_STAGES:
        warnings = metadata[stage].get("warnings")
        if not isinstance(warnings, list):
            return False
        if any(
            not isinstance(item, dict)
            or set(item) != {"code", "stage", "message"}
            or any(not isinstance(item.get(key), str) or not item.get(key) for key in item)
            for item in warnings
        ):
            return False
        if warnings[: len(previous)] != previous:
            return False
        if any(item["stage"] != stage for item in warnings[len(previous) :]):
            return False
        previous = warnings
    return True


def _valid_frame_handoff(
    prior: ValidatedStageInput,
    current_metadata: dict[str, object],
    current_stage: DownstreamStage,
) -> bool:
    current_frames = current_metadata.get("frames")
    if not isinstance(current_frames, list) or len(current_frames) != len(prior.frames):
        return False
    for prior_frame, current_frame in zip(prior.frames, current_frames, strict=True):
        if not isinstance(current_frame, dict):
            return False
        if (
            current_frame.get("name") != prior_frame.name
            or current_frame.get("relativePath") != prior_frame.relative_path.as_posix()
            or current_frame.get("sourceOrder") != prior_frame.source_order
        ):
            return False
        if current_stage == "align":
            width = current_frame.get("inputWidth")
            height = current_frame.get("inputHeight")
        else:
            dimensions = current_frame.get("inputDimensions")
            if not isinstance(dimensions, dict):
                return False
            width = dimensions.get("width")
            height = dimensions.get("height")
        if (
            type(width) is not int
            or type(height) is not int
            or width != prior_frame.dimensions.width
            or height != prior_frame.dimensions.height
        ):
            return False
    return True


def _valid_completed_run(root: Path) -> bool:
    try:
        metadata: dict[StageName, dict[str, object]] = {
            stage: _read_json_object(root / stage / "stage.json", "PX_RUN") for stage in RUN_STAGES
        }
        if not _valid_owned_output(root / "extract", "extract", _valid_owned_extract_metadata):
            return False
        if not all(_valid_owned_output(root / stage, stage) for stage in RUN_STAGES[1:]):
            return False
        validated_extract = validate_stage_input(root / "extract", "extract")
        validated_scale = validate_stage_input(root / "scale", "scale")
        validated_pixelize = validate_stage_input(root / "pixelize", "pixelize")
        if not _valid_warning_lineage(metadata):
            return False
        if not all(
            (
                _valid_frame_handoff(validated_extract, metadata["scale"], "scale"),
                _valid_frame_handoff(validated_scale, metadata["pixelize"], "pixelize"),
                _valid_frame_handoff(validated_pixelize, metadata["align"], "align"),
            )
        ):
            return False
        identity_fields = (
            tuple(document.get("sourceConfigSha256") for document in metadata.values()),
            tuple(document.get("effectiveConfigSha256") for document in metadata.values()),
            tuple(document.get("pixipixVersion") for document in metadata.values()),
        )
        for values in identity_fields:
            if not all(isinstance(value, str) and value for value in values):
                return False
            if len(set(values)) != 1:
                return False
        for prior_stage, stage in pairwise(RUN_STAGES):
            prior = metadata[stage].get("priorStage")
            if not isinstance(prior, dict):
                return False
            if prior != {
                "stage": prior_stage,
                "schemaVersion": metadata[prior_stage].get("schemaVersion"),
                "pixipixVersion": metadata[prior_stage].get("pixipixVersion"),
                "effectiveConfigSha256": metadata[prior_stage].get("effectiveConfigSha256"),
            }:
                return False
        return True
    except (UnsupportedInputError, OSError):
        return False


def run_pipeline(
    input_path: Path,
    loaded: LoadedConfig,
    output: Path,
    *,
    force: bool = False,
) -> PipelineRunResult:
    """Execute Extract → Scale → Pixelize → Align and publish one complete run."""

    def build(root: Path) -> PipelineRunResult:
        extracted = publish_extraction(input_path, loaded, root / "extract")
        scaled = publish_scale(root / "extract", loaded, root / "scale")
        pixelized = publish_pixelize(root / "scale", loaded, root / "pixelize")
        aligned = publish_align(root / "pixelize", loaded, root / "align")
        return PipelineRunResult(
            output_root=output,
            extract=extracted,
            scale=scaled,
            pixelize=pixelized,
            align=aligned,
            warnings=aligned.warnings,
        )

    return publish_run_output(output, build, _valid_completed_run, force=force)
