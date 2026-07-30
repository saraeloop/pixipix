"""Source decoding and deterministic component analysis for Extract."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pixipix.config import ExtractConfig, LoadedConfig
from pixipix.errors import ProcessingError
from pixipix.imageio import generate_foreground_mask, load_source
from pixipix.models import (
    BackgroundSummary,
    BoolMask,
    Component,
    Rect,
    RejectedComponent,
    RejectionReason,
    SourceImage,
)

LabelMap = npt.NDArray[np.int32]

FOUR_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (0, -1),  # left
    (0, 1),  # right
    (1, 0),  # down
)
EIGHT_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, -1),  # up-left
    (-1, 0),  # up
    (-1, 1),  # up-right
    (0, -1),  # left
    (0, 1),  # right
    (1, -1),  # down-left
    (1, 0),  # down
    (1, 1),  # down-right
)


@dataclass(slots=True)
class ComponentMap:
    components: tuple[Component, ...]
    labels: LabelMap


@dataclass(slots=True)
class _Analysis:
    source: SourceImage
    mask: BoolMask
    component_map: ComponentMap
    accepted: tuple[Component, ...]
    rejected: tuple[RejectedComponent, ...]
    ordered: tuple[Component, ...]
    background: BackgroundSummary


def label_components(mask: BoolMask, connectivity: int, max_components: int) -> ComponentMap:
    """Label foreground using row-major discovery and ADR-locked neighbor order."""

    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    neighbors = FOUR_NEIGHBORS if connectivity == 4 else EIGHT_NEIGHBORS
    components: list[Component] = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column] != 0:
                continue
            discovery_index = len(components)
            if discovery_index >= max_components:
                raise ProcessingError(
                    "PX_EXTRACT_001",
                    "extract",
                    f"candidate component count exceeds configured limit {max_components}",
                    remediation="raise max_components explicitly or remove foreground noise",
                )
            label = discovery_index + 1
            pending: deque[tuple[int, int]] = deque([(row, column)])
            labels[row, column] = label
            area = 0
            left = column
            right = column
            top = row
            bottom = row
            while pending:
                current_row, current_column = pending.popleft()
                area += 1
                left = min(left, current_column)
                right = max(right, current_column)
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                for row_offset, column_offset in neighbors:
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and labels[next_row, next_column] == 0
                    ):
                        labels[next_row, next_column] = label
                        pending.append((next_row, next_column))
            components.append(
                Component(
                    discovery_index=discovery_index,
                    bounds=Rect(left=left, top=top, right=right + 1, bottom=bottom + 1),
                    area=area,
                )
            )
    return ComponentMap(tuple(components), labels)


def filter_components(
    components: tuple[Component, ...], config: ExtractConfig
) -> tuple[tuple[Component, ...], tuple[RejectedComponent, ...]]:
    accepted: list[Component] = []
    rejected: list[RejectedComponent] = []
    for component in components:
        reasons: list[RejectionReason] = []
        if component.area < config.minimum_area:
            reasons.append("below-minimum-area")
        if config.maximum_area is not None and component.area > config.maximum_area:
            reasons.append("above-maximum-area")
        if reasons:
            rejected.append(RejectedComponent(component, tuple(reasons)))
        else:
            accepted.append(component)
    return tuple(accepted), tuple(rejected)


def order_components(
    components: tuple[Component, ...], row_tolerance: int
) -> tuple[Component, ...]:
    """Group by row top, then apply the locked deterministic reading order."""

    by_top = sorted(
        components,
        key=lambda item: (
            item.bounds.top,
            item.bounds.left,
            item.area,
            item.discovery_index,
        ),
    )
    rows: list[tuple[int, list[Component]]] = []
    for component in by_top:
        matching_row = next(
            (row for row in rows if abs(component.bounds.top - row[0]) <= row_tolerance),
            None,
        )
        if matching_row is None:
            rows.append((component.bounds.top, [component]))
        else:
            matching_row[1].append(component)
    ordered: list[Component] = []
    for _, row in sorted(rows, key=lambda item: item[0]):
        ordered.extend(
            sorted(
                row,
                key=lambda item: (item.bounds.left, item.area, item.discovery_index),
            )
        )
    return tuple(ordered)


def _analyze(input_path: Path, loaded: LoadedConfig) -> _Analysis:
    config = loaded.config
    source = load_source(input_path, config.source)
    mask, background = generate_foreground_mask(source, config.background)
    component_map = label_components(
        mask, config.extract.connectivity, config.source.max_components
    )
    accepted, rejected = filter_components(component_map.components, config.extract)
    ordered = order_components(accepted, config.extract.row_tolerance)
    return _Analysis(source, mask, component_map, accepted, rejected, ordered, background)
