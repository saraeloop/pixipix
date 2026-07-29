"""Pixelize-stage metadata construction."""

from __future__ import annotations

from pixipix import __version__
from pixipix.config import LoadedConfig, PixelizeConfig
from pixipix.models import PixelizeStageMetadata
from pixipix.pipeline.input import LoadedStageInput

from .planning import PixelizeStagePlan


def build_pixelize_metadata(
    stage: LoadedStageInput,
    loaded: LoadedConfig,
    plan: PixelizeStagePlan,
    config: PixelizeConfig,
    cell_size: int,
) -> PixelizeStageMetadata:
    return PixelizeStageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="pixelize",
        status="successful",
        prior_stage=stage.identity,
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        source_cell_size=cell_size,
        cell_grid_origin="bottom-left",
        representative=config.representative,
        alpha_policy=config.alpha_policy,
        alpha_threshold=config.alpha_threshold,
        remainder_policy=config.remainder_policy,
        frames=plan.frames,
        warnings=plan.warnings,
    )
