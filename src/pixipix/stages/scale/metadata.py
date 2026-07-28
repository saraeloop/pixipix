"""Pure scale metadata construction."""

from __future__ import annotations

from pixipix import __version__
from pixipix.config import LoadedConfig, ScaleConfig
from pixipix.models import ScaleOverrideMetadata, ScaleStageMetadata
from pixipix.pipeline.input import LoadedStageInput

from .planning import ScaleStagePlan


def build_scale_metadata(
    stage: LoadedStageInput,
    loaded: LoadedConfig,
    plan: ScaleStagePlan,
    config: ScaleConfig,
) -> ScaleStageMetadata:
    return ScaleStageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="scale",
        status="successful",
        prior_stage=stage.identity,
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        scale_mode=config.mode,
        global_factor=plan.global_factor,
        reference_frame=config.reference_frame,
        source_reference_measurement=plan.source_reference_measurement,
        exact_target_source_measurement=plan.exact_target_source_measurement,
        logical_target_size=config.target_size,
        source_cell_size=loaded.config.pixelize.source_cell_size,
        configured_frame_overrides=tuple(
            ScaleOverrideMetadata(item.frame_name, item.scale_multiplier)
            for item in loaded.config.frame_overrides
        ),
        frames=plan.frames,
        warnings=plan.warnings,
    )
