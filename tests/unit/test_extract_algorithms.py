from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from pixipix.config import ExtractConfig
from pixipix.errors import ProcessingError
from pixipix.models import Component, Rect
from pixipix.stages.extract import filter_components, label_components, order_components


def test_four_and_eight_connectivity() -> None:
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)

    four = label_components(mask, 4, 8)
    eight = label_components(mask, 8, 8)

    assert [component.area for component in four.components] == [1, 1]
    assert [component.discovery_index for component in four.components] == [0, 1]
    assert [component.bounds for component in four.components] == [
        Rect(0, 0, 1, 1),
        Rect(1, 1, 2, 2),
    ]
    assert len(eight.components) == 1
    assert eight.components[0].area == 2
    assert eight.components[0].bounds == Rect(0, 0, 2, 2)


def test_row_major_discovery_and_exact_area_bounds() -> None:
    mask = np.zeros((5, 7), dtype=np.bool_)
    mask[0, 5] = True
    mask[2:4, 1:4] = True

    result = label_components(mask, 4, 8)

    assert result.components == (
        Component(0, Rect(5, 0, 6, 1), 1),
        Component(1, Rect(1, 2, 4, 4), 6),
    )
    assert result.labels[0, 5] == 1
    assert result.labels[2, 1] == 2


def test_empty_foreground_is_safe() -> None:
    result = label_components(np.zeros((3, 4), dtype=np.bool_), 8, 8)
    assert result.components == ()
    assert not result.labels.any()


def test_component_limit_fails_stably() -> None:
    mask = np.array([[True, False, True]], dtype=np.bool_)
    with pytest.raises(ProcessingError, match="PX_EXTRACT_001"):
        label_components(mask, 4, 1)


def test_checkerboard_component_limit_boundary_and_one_pixel_components() -> None:
    mask = np.indices((5, 5)).sum(axis=0) % 2 == 0

    result = label_components(np.asarray(mask, dtype=np.bool_), 4, 13)

    assert len(result.components) == 13
    assert all(component.area == 1 for component in result.components)
    assert [component.discovery_index for component in result.components] == list(range(13))
    with pytest.raises(ProcessingError, match="PX_EXTRACT_001"):
        label_components(np.asarray(mask, dtype=np.bool_), 4, 12)


def test_full_foreground_holes_and_thin_diagonals_have_exact_areas() -> None:
    full = np.ones((4, 6), dtype=np.bool_)
    full[1:3, 2:4] = False
    component = label_components(full, 4, 1).components[0]
    assert component.area == 20
    assert component.bounds == Rect(0, 0, 6, 4)

    diagonal = np.eye(8, dtype=np.bool_)
    assert len(label_components(diagonal, 4, 8).components) == 8
    connected = label_components(diagonal, 8, 1).components
    assert len(connected) == 1
    assert connected[0].area == 8


def test_area_filter_retains_rejected_components_and_reasons() -> None:
    components = (
        Component(0, Rect(0, 0, 1, 1), 1),
        Component(1, Rect(2, 0, 4, 2), 4),
        Component(2, Rect(5, 0, 8, 3), 9),
    )

    accepted, rejected = filter_components(
        components, ExtractConfig(minimum_area=2, maximum_area=8)
    )

    assert accepted == (components[1],)
    assert [item.component for item in rejected] == [components[0], components[2]]
    assert rejected[0].reasons == ("below-minimum-area",)
    assert rejected[1].reasons == ("above-maximum-area",)


def test_area_filter_boundaries_are_inclusive() -> None:
    minimum = Component(0, Rect(0, 0, 2, 1), 2)
    maximum = Component(1, Rect(0, 1, 4, 3), 8)

    accepted, rejected = filter_components(
        (minimum, maximum), ExtractConfig(minimum_area=2, maximum_area=8)
    )

    assert accepted == (minimum, maximum)
    assert rejected == ()


def test_reading_order_rows_left_to_right_and_tie_breakers() -> None:
    components = (
        Component(3, Rect(8, 10, 10, 12), 4),
        Component(2, Rect(1, 11, 3, 13), 4),
        Component(1, Rect(6, 1, 7, 2), 1),
        Component(0, Rect(0, 0, 2, 2), 4),
        Component(5, Rect(1, 10, 2, 11), 1),
    )

    ordered = order_components(components, row_tolerance=1)

    assert [component.discovery_index for component in ordered] == [0, 1, 5, 2, 3]
    assert order_components(components, row_tolerance=1) == ordered


def test_row_tolerance_boundary_is_inclusive() -> None:
    first = Component(0, Rect(9, 0, 10, 1), 1)
    at_boundary = Component(1, Rect(0, 2, 1, 3), 1)

    assert order_components((first, at_boundary), 2) == (at_boundary, first)
    assert order_components((first, at_boundary), 1) == (first, at_boundary)


def test_reading_order_is_permutation_independent_and_does_not_chain_rows() -> None:
    anchor = Component(10, Rect(9, 0, 10, 1), 1)
    near_anchor = Component(11, Rect(8, 2, 9, 3), 1)
    near_only_by_chain = Component(12, Rect(0, 4, 1, 5), 1)
    components = (anchor, near_anchor, near_only_by_chain)

    expected = (near_anchor, anchor, near_only_by_chain)
    for permutation in permutations(components):
        assert order_components(permutation, 2) == expected


def test_row_grouping_uses_top_anchor_not_overlap_or_component_height() -> None:
    tall = Component(0, Rect(5, 0, 7, 20), 40)
    short = Component(1, Rect(0, 3, 1, 4), 1)
    later = Component(2, Rect(2, 6, 3, 7), 1)

    assert order_components((later, short, tall), 3) == (short, tall, later)


def test_area_is_ascending_tie_breaker_before_discovery_index() -> None:
    larger = Component(0, Rect(1, 0, 3, 2), 4)
    smaller = Component(9, Rect(1, 0, 2, 1), 1)

    assert order_components((larger, smaller), 0) == (smaller, larger)
