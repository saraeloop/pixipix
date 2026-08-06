"""Foundational deterministic scale geometry."""

from __future__ import annotations

import math


def round_half_away_from_zero(value: float) -> int:
    """Round one finite geometric value with halves away from zero."""

    if not math.isfinite(value):
        raise ValueError("geometric value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    return magnitude if value >= 0 else -magnitude


def transformed_dimension(source_dimension: int, factor: float) -> int:
    if source_dimension <= 0:
        raise ValueError("source dimension must be positive")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("scale factor must be finite and positive")
    return max(1, round_half_away_from_zero(source_dimension * factor))
