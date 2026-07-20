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
    stage: Literal["extract", "scale", "pixelize", "align"]


@dataclass(frozen=True, slots=True)
class Dimensions:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PriorStageIdentity:
    stage: Literal["extract", "scale", "pixelize"]
    schema_version: int
    pixipix_version: str
    effective_config_sha256: str


@dataclass(frozen=True, slots=True)
class ScaleFrame:
    name: str
    relative_path: PurePosixPath
    source_order: int
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    scale_multiplier: float
    effective_factor: float


@dataclass(frozen=True, slots=True)
class ScaleOverrideMetadata:
    frame_name: str
    scale_multiplier: float


@dataclass(frozen=True, slots=True)
class ScaleStageMetadata:
    schema_version: int
    pixipix_version: str
    stage: Literal["scale"]
    status: Literal["successful"]
    prior_stage: PriorStageIdentity
    source_config_sha256: str
    effective_config_sha256: str
    scale_mode: Literal["explicit-factor", "reference-frame-width", "reference-frame-height"]
    global_factor: float
    reference_frame: str | None
    source_reference_measurement: int | None
    exact_target_source_measurement: int | None
    logical_target_size: int | None
    source_cell_size: int | None
    configured_frame_overrides: tuple[ScaleOverrideMetadata, ...]
    frames: tuple[ScaleFrame, ...]
    warnings: tuple[ProcessingWarning, ...]


@dataclass(frozen=True, slots=True)
class PixelizeFrame:
    name: str
    relative_path: PurePosixPath
    source_order: int
    input_dimensions: Dimensions
    prepared_dimensions: Dimensions
    top_padding: int
    right_padding: int
    top_crop: int
    right_crop: int
    logical_output_dimensions: Dimensions


@dataclass(frozen=True, slots=True)
class PixelizeStageMetadata:
    schema_version: int
    pixipix_version: str
    stage: Literal["pixelize"]
    status: Literal["successful"]
    prior_stage: PriorStageIdentity
    source_config_sha256: str
    effective_config_sha256: str
    source_cell_size: int
    cell_grid_origin: Literal["bottom-left"]
    representative: Literal["majority", "center", "alpha-weighted-majority"]
    alpha_policy: Literal["binary", "preserve"]
    alpha_threshold: int
    remainder_policy: Literal["pad-transparent", "error", "crop-with-warning"]
    frames: tuple[PixelizeFrame, ...]
    warnings: tuple[ProcessingWarning, ...]


@dataclass(frozen=True, slots=True)
class AlignmentRectangle:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AlignmentClippingFinding:
    frame_name: str
    source_order: int
    left_overflow: int
    top_overflow: int
    right_overflow: int
    bottom_overflow: int
    visible_source_rectangle: AlignmentRectangle
    visible_destination_rectangle: AlignmentRectangle


@dataclass(frozen=True, slots=True)
class AlignmentFrame:
    name: str
    relative_path: PurePosixPath
    source_order: int
    input_width: int
    input_height: int
    base_x: int
    base_y: int
    offset_dx: int
    offset_dy: int
    final_x: int
    final_y: int
    left_overflow: int
    top_overflow: int
    right_overflow: int
    bottom_overflow: int
    clipped: bool
    visible_source_rectangle: AlignmentRectangle
    visible_destination_rectangle: AlignmentRectangle
    output_width: int
    output_height: int


@dataclass(frozen=True, slots=True)
class AlignmentStageMetadata:
    schema_version: int
    pixipix_version: str
    stage: Literal["align"]
    status: Literal["successful"]
    prior_stage: PriorStageIdentity
    source_config_sha256: str
    effective_config_sha256: str
    canvas_width: int
    canvas_height: int
    anchor: Literal[
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center",
        "center-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ]
    configured_baseline_y: int | None
    effective_baseline_y: int | None
    clipping_policy: Literal["error", "warn", "allow"]
    clipping_findings: tuple[AlignmentClippingFinding, ...]
    frames: tuple[AlignmentFrame, ...]
    warnings: tuple[ProcessingWarning, ...]
