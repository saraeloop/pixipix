"""Extract-specific orchestration and adaptation for shared publication."""

from __future__ import annotations

from pathlib import Path

from pixipix.config import LoadedConfig
from pixipix.models import ExtractionResult
from pixipix.pipeline.publication import (
    OutputFrameImage,
    publish_stage_output,
    validate_stage_output_target,
)

from .api import extract_source
from .metadata import _stage_metadata, _valid_owned_extract_metadata


def publish_extraction(
    input_path: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> ExtractionResult:
    validate_stage_output_target(
        output,
        "extract",
        force=force,
        owned_metadata_validator=_valid_owned_extract_metadata,
    )
    run = extract_source(input_path, loaded)
    metadata = _stage_metadata(run, loaded)
    frames = tuple(
        OutputFrameImage(
            relative_path=frame.metadata.relative_path,
            pixels=frame.pixels,
        )
        for frame in run.frame_images
    )
    publish_stage_output(
        output,
        "extract",
        metadata,
        frames,
        force=force,
        owned_metadata_validator=_valid_owned_extract_metadata,
    )
    return run.result
