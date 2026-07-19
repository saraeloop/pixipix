"""Typed source-space domain models for the extraction milestone.

Rectangles use zero-based source-pixel coordinates with half-open right and
bottom edges: ``[left, right) x [top, bottom)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import numpy as np
import numpy.typing as npt

UInt8Image = npt.NDArray[np.uint8]
BoolMask = npt.NDArray[np.bool_]
type RejectionReason = Literal["below-minimum-area", "above-maximum-area"]


@dataclass(frozen=True, slots=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class SourceImageMetadata:
    path: PurePosixPath
    width: int
    height: int
    input_mode: str
    has_alpha: bool
    normalized_mode: Literal["RGBA"] = "RGBA"


@dataclass(slots=True)
class SourceImage:
    """Ownership-controlled decoded pixels; callers must treat pixels as read-only."""

    metadata: SourceImageMetadata
    pixels: UInt8Image


@dataclass(frozen=True, slots=True)
class Component:
    discovery_index: int
    bounds: Rect
    area: int


@dataclass(frozen=True, slots=True)
class RejectedComponent:
    component: Component
    reasons: tuple[RejectionReason, ...]


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    name: str
    relative_path: PurePosixPath
    source_order: int
    discovery_index: int
    component_area: int
    original_bounds: Rect
    padded_bounds: Rect


@dataclass(slots=True)
class FrameImage:
    metadata: ExtractedFrame
    pixels: UInt8Image


@dataclass(frozen=True, slots=True)
class BackgroundSummary:
    mode: Literal["alpha", "corner-color", "explicit-color"]
    selected_color: str | None
    tolerance: float
    pixels_removed: int
    foreground_touches_boundary: bool
    foreground_bounds: Rect | None


@dataclass(frozen=True, slots=True)
class ProcessingWarning:
    code: str
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    source: SourceImageMetadata
    background: BackgroundSummary
    candidates: tuple[Component, ...]
    accepted: tuple[Component, ...]
    rejected: tuple[RejectedComponent, ...]
    ordered: tuple[Component, ...]
    frames: tuple[ExtractedFrame, ...]
    warnings: tuple[ProcessingWarning, ...] = ()


@dataclass(slots=True)
class ExtractionRun:
    result: ExtractionResult
    frame_images: tuple[FrameImage, ...]


@dataclass(frozen=True, slots=True)
class InspectionResult:
    source: SourceImageMetadata
    background: BackgroundSummary
    candidates: tuple[Component, ...]
    accepted: tuple[Component, ...]
    rejected: tuple[RejectedComponent, ...]
    ordered: tuple[Component, ...]
    frame_assignments: tuple[str, ...] | None
    configured_source_cell_size: int | None


@dataclass(frozen=True, slots=True)
class StageMetadata:
    schema_version: int
    pixipix_version: str
    stage: Literal["extract"]
    status: Literal["successful"]
    source_config_sha256: str
    effective_config_sha256: str
    source: SourceImageMetadata
    background: BackgroundSummary
    candidate_components: tuple[Component, ...]
    accepted_components: tuple[Component, ...]
    rejected_components: tuple[RejectedComponent, ...]
    ordered_components: tuple[Component, ...]
    frames: tuple[ExtractedFrame, ...]
    warnings: tuple[ProcessingWarning, ...]


@dataclass(frozen=True, slots=True)
class OutputMarker:
    schema_version: int
    owner: Literal["pixipix"]
    stage: Literal["extract"]
