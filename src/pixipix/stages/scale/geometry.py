"""Deterministic scale geometry and channel quantization."""

from __future__ import annotations

import math


def round_half_away_from_zero(value: float) -> int:
    """Round one finite geometric value with halves away from zero."""

    if not math.isfinite(value):
        raise ValueError("geometric value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    return magnitude if value >= 0 else -magnitude


def round_channel_half_away_from_zero(value: float) -> int:
    """Quantize one finite channel value separately from geometry rules."""

    if not math.isfinite(value):
        raise ValueError("channel value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    rounded = magnitude if value >= 0 else -magnitude
    return min(255, max(0, rounded))


def transformed_dimension(source_dimension: int, factor: float) -> int:
    if source_dimension <= 0:
        raise ValueError("source dimension must be positive")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("scale factor must be finite and positive")
    return max(1, round_half_away_from_zero(source_dimension * factor))
