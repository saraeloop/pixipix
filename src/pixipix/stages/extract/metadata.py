"""Pure metadata construction for Extract."""

from __future__ import annotations

from pixipix import __version__
from pixipix.config import LoadedConfig
from pixipix.models import ExtractionRun, StageMetadata


def _valid_owned_extract_metadata(metadata: dict[str, object]) -> bool:
    """Validate Extract-only metadata facts before an owned target is authorized."""

    required_container_types = {
        "source": dict,
        "background": dict,
        "candidateComponents": list,
        "acceptedComponents": list,
        "rejectedComponents": list,
        "orderedComponents": list,
        "warnings": list,
    }
    if any(
        not isinstance(metadata.get(key), expected)
        for key, expected in required_container_types.items()
    ):
        return False
    for key in ("sourceConfigSha256", "effectiveConfigSha256"):
        value = metadata.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False
    return isinstance(metadata.get("pixipixVersion"), str)


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
