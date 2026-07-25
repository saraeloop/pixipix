"""Deterministic aggregate-resource policy and admission semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type ResourceStage = Literal["extract", "scale", "pixelize", "align"]
type ResourceFindingKind = Literal[
    "aggregate_input_pixels",
    "aggregate_output_pixels",
    "modeled_peak_live_bytes",
]

DEFAULT_MAX_AGGREGATE_INPUT_PIXELS = 50_000_000
DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS = 60_000_000
DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES = 1_000_000_000

MAX_AGGREGATE_INPUT_PIXELS_CAP = 150_000_000
MAX_AGGREGATE_OUTPUT_PIXELS_CAP = 160_000_000
MAX_MODELED_PEAK_LIVE_BYTES_CAP = 2_000_000_000

RESOURCE_FINDING_ORDER: tuple[ResourceFindingKind, ...] = (
    "aggregate_input_pixels",
    "aggregate_output_pixels",
    "modeled_peak_live_bytes",
)


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Complete normalized aggregate-resource policy."""

    max_aggregate_input_pixels: int = DEFAULT_MAX_AGGREGATE_INPUT_PIXELS
    max_aggregate_output_pixels: int = DEFAULT_MAX_AGGREGATE_OUTPUT_PIXELS
    max_modeled_peak_live_bytes: int = DEFAULT_MAX_MODELED_PEAK_LIVE_BYTES


@dataclass(frozen=True, slots=True)
class ResourceProjection:
    """Exact stage projection under the locked explicit-buffer model."""

    stage: ResourceStage
    aggregate_input_pixels: int
    aggregate_output_pixels: int
    modeled_peak_live_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceFinding:
    """One exceeded aggregate-resource budget."""

    kind: ResourceFindingKind
    computed: int
    limit: int


type ResourceFindings = tuple[ResourceFinding, ...]


def resource_findings(
    projection: ResourceProjection,
    policy: ResourcePolicy,
) -> ResourceFindings:
    """Return every exceeded budget in the locked deterministic order."""

    values = {
        "aggregate_input_pixels": (
            projection.aggregate_input_pixels,
            policy.max_aggregate_input_pixels,
        ),
        "aggregate_output_pixels": (
            projection.aggregate_output_pixels,
            policy.max_aggregate_output_pixels,
        ),
        "modeled_peak_live_bytes": (
            projection.modeled_peak_live_bytes,
            policy.max_modeled_peak_live_bytes,
        ),
    }
    return tuple(
        ResourceFinding(kind, computed, limit)
        for kind in RESOURCE_FINDING_ORDER
        for computed, limit in (values[kind],)
        if computed > limit
    )


def enforce_resource_policy(
    projection: ResourceProjection,
    policy: ResourcePolicy,
) -> None:
    """Raise one structured processing error when any budget is exceeded."""

    findings = resource_findings(projection, policy)
    if findings:
        from pixipix.errors import ResourcePolicyError

        raise ResourcePolicyError(projection, policy, findings)
