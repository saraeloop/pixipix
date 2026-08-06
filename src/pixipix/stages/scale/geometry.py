"""Deterministic scale geometry and channel quantization."""

from __future__ import annotations

import math

from pixipix._scale_geometry import (
    round_half_away_from_zero as round_half_away_from_zero,
)
from pixipix._scale_geometry import (
    transformed_dimension as transformed_dimension,
)


def round_channel_half_away_from_zero(value: float) -> int:
    """Quantize one finite channel value separately from geometry rules."""

    if not math.isfinite(value):
        raise ValueError("channel value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    rounded = magnitude if value >= 0 else -magnitude
    return min(255, max(0, rounded))
