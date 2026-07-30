"""Pure metadata construction for Extract."""

from __future__ import annotations

from pixipix import __version__
from pixipix.config import LoadedConfig
from pixipix.models import ExtractionRun, StageMetadata


def _stage_metadata(run: ExtractionRun, loaded: LoadedConfig) -> StageMetadata:
    result = run.result
    return StageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="extract",
        status="successful",
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        source=result.source,
        background=result.background,
        candidate_components=result.candidates,
        accepted_components=result.accepted,
        rejected_components=result.rejected,
        ordered_components=result.ordered,
        frames=result.frames,
        warnings=result.warnings,
    )
