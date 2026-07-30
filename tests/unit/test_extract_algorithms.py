from __future__ import annotations

from collections.abc import Callable
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pixipix.stages.extract as extract_stage
import pixipix.stages.extract.analysis as extract_analysis
import pixipix.stages.extract.api as extract_api
import pixipix.stages.extract.execution as extract_execution
import pixipix.stages.extract.planning as extract_planning
import pixipix.stages.extract.publication as extract_publication
from pixipix.config import ExtractConfig, load_config
from pixipix.errors import ProcessingError
from pixipix.models import Component, ExtractedFrame, Rect
from pixipix.resources import ResourceProjection
from pixipix.stages.extract import (
    filter_components,
    label_components,
    order_components,
    project_extract_resources,
    project_extracted_frames,
)
from tests.helpers import extraction_config, write_config, write_rgba


class _ExtractNumpyProxy:
    def __init__(self, zeros: Callable[..., object]) -> None:
        self.zeros = zeros

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


class _ExtractExecutionNumpyProxy:
    def __init__(self, array: Callable[..., object]) -> None:
        self.array = array

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


def test_four_and_eight_connectivity() -> None:
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)

    four = label_components(mask, 4, 8)
    eight = label_components(mask, 8, 8)

    assert type(four) is extract_stage.ComponentMap
    assert type(eight) is extract_stage.ComponentMap
    assert [component.area for component in four.components] == [1, 1]
    assert [component.discovery_index for component in four.components] == [0, 1]
    assert [component.bounds for component in four.components] == [
        Rect(0, 0, 1, 1),
        Rect(1, 1, 2, 2),
    ]
    assert len(eight.components) == 1
    assert eight.components[0].area == 2
    assert eight.components[0].bounds == Rect(0, 0, 2, 2)


def test_labeling_uses_extract_package_numpy_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.ones((1, 1), dtype=np.bool_)
    global_zeros = np.zeros

    class ExtractZerosReached(Exception):
        pass

    def mark_zeros(*_args: object, **_kwargs: object) -> None:
        raise ExtractZerosReached("extract package np.zeros reached")

    proxy = _ExtractNumpyProxy(mark_zeros)
    assert proxy.int32 is np.int32
    monkeypatch.setattr(extract_analysis, "np", proxy)

    with pytest.raises(
        ExtractZerosReached,
        match=r"extract package np\.zeros reached",
    ) as raised:
        label_components(mask, 4, 1)

    traceback_names = tuple(entry.name for entry in raised.traceback)
    assert "label_components" in traceback_names
    assert traceback_names[-1] == "mark_zeros"
    assert np.zeros is global_zeros


def test_staged_png_validation_uses_extract_publication_pillow_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(b"not decoded because the package binding raises")

    class ExtractImageOpenReached(Exception):
        pass

    def mark_open(_path: Path) -> None:
        raise ExtractImageOpenReached("extract publication Image.open reached")

    monkeypatch.setattr(extract_publication, "Image", SimpleNamespace(open=mark_open))

    with pytest.raises(
        ExtractImageOpenReached,
        match=r"extract publication Image\.open reached",
    ) as raised:
        extract_publication._valid_frame_png(candidate)

    traceback_names = tuple(entry.name for entry in raised.traceback)
    assert "_valid_frame_png" in traceback_names
    assert traceback_names[-1] == "mark_open"


def test_analysis_uses_authoritative_source_decoder_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "project.toml"
    write_config(config)
    loaded = load_config(config)

    class ExtractSourceDecoderReached(Exception):
        pass

    def mark_load_source(_path: Path, _config: object) -> None:
        raise ExtractSourceDecoderReached("extract analysis load_source reached")

    monkeypatch.setattr(extract_analysis, "load_source", mark_load_source)

    with pytest.raises(
        ExtractSourceDecoderReached,
        match="extract analysis load_source reached",
    ) as raised:
        extract_stage.inspect_source(tmp_path / "source.png", loaded)

    traceback_names = tuple(entry.name for entry in raised.traceback)
    assert "_analyze" in traceback_names
    assert traceback_names[-1] == "mark_load_source"


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


def test_projected_extract_bounds_and_formula_inputs_stay_synchronized_without_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.zeros((10, 12, 4), dtype=np.uint8)
    pixels[0:2, 0:1] = (10, 20, 30, 255)
    pixels[3:5, 5:7] = (40, 50, 60, 255)
    pixels[8:10, 10:12] = (70, 80, 90, 255)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, pixels)
    write_config(
        config,
        extraction_config(
            names=("edge-top-left", "middle", "edge-bottom-right"),
            minimum_area=1,
            padding=2,
            expected=3,
        ),
    )
    loaded = load_config(config)
    analysis = extract_analysis._analyze(image, loaded)
    crop_calls = 0

    assert type(analysis) is extract_analysis._Analysis

    def record_crop(*_args: object) -> None:
        nonlocal crop_calls
        crop_calls += 1

    monkeypatch.setattr(extract_api, "_materialize_frame_crop", record_crop)
    frames = project_extracted_frames(analysis, loaded)

    assert all(type(frame) is ExtractedFrame for frame in frames)
    assert tuple(frame.original_bounds for frame in frames) == (
        Rect(0, 0, 1, 2),
        Rect(5, 3, 7, 5),
        Rect(10, 8, 12, 10),
    )
    assert (
        tuple(frame.padded_bounds for frame in frames)
        == tuple(
            extract_planning._padded_bounds(component.bounds, 2, 12, 10)
            for component in analysis.ordered
        )
        == (
            Rect(0, 0, 3, 4),
            Rect(3, 1, 9, 7),
            Rect(8, 6, 12, 10),
        )
    )
    frame_areas = tuple(frame.padded_bounds.width * frame.padded_bounds.height for frame in frames)
    assert (10 * 12, sum(frame_areas), max(frame_areas)) == (120, 64, 36)
    projection = project_extract_resources(120, frames)
    assert type(projection) is ResourceProjection
    assert projection == ResourceProjection(
        "extract",
        120,
        64,
        1_372,
    )
    assert crop_calls == 0


def test_frame_projection_uses_extract_package_bounds_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.zeros((3, 3, 4), dtype=np.uint8)
    pixels[1, 1] = (10, 20, 30, 255)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, pixels)
    write_config(
        config,
        extraction_config(names=("only",), minimum_area=1, padding=1, expected=1),
    )
    loaded = load_config(config)
    analysis = extract_analysis._analyze(image, loaded)

    class ExtractBoundsReached(Exception):
        pass

    def mark_bounds(_bounds: Rect, _padding: int, _width: int, _height: int) -> None:
        raise ExtractBoundsReached("extract planning _padded_bounds reached")

    monkeypatch.setattr(extract_planning, "_padded_bounds", mark_bounds)

    with pytest.raises(
        ExtractBoundsReached,
        match="extract planning _padded_bounds reached",
    ) as raised:
        project_extracted_frames(analysis, loaded)

    traceback_names = tuple(entry.name for entry in raised.traceback)
    assert "project_extracted_frames" in traceback_names
    assert traceback_names[-1] == "mark_bounds"


def test_crop_materialization_uses_execution_numpy_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.zeros((3, 3, 4), dtype=np.uint8)
    pixels[1, 1] = (10, 20, 30, 200)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, pixels)
    write_config(
        config,
        extraction_config(names=("only",), minimum_area=1, padding=1, expected=1),
    )
    loaded = load_config(config)
    analysis = extract_analysis._analyze(image, loaded)
    frame = project_extracted_frames(analysis, loaded)[0]
    component = analysis.ordered[0]
    real_array: Callable[..., object] = np.array
    calls: list[dict[str, object]] = []

    def record_array(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return real_array(*args, **kwargs)

    proxy = _ExtractExecutionNumpyProxy(record_array)
    assert proxy.uint8 is np.uint8
    assert vars(extract_analysis)["np"] is np
    monkeypatch.setattr(extract_execution, "np", proxy)

    materialized = extract_execution._materialize_frame_crop(analysis, component, frame)

    assert calls == [{"dtype": np.uint8, "copy": True}]
    assert materialized.pixels[1, 1].tolist() == [10, 20, 30, 200]
    assert materialized.pixels[0, 0].tolist() == [0, 0, 0, 0]
    assert np.array is real_array
    assert vars(extract_analysis)["np"] is np


def test_analysis_and_execution_numpy_bindings_are_cross_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.zeros((3, 3, 4), dtype=np.uint8)
    pixels[1, 1] = (10, 20, 30, 200)
    image = tmp_path / "source.png"
    config = tmp_path / "project.toml"
    write_rgba(image, pixels)
    write_config(
        config,
        extraction_config(names=("only",), minimum_area=1, padding=1, expected=1),
    )
    loaded = load_config(config)
    analysis = extract_analysis._analyze(image, loaded)
    frame = project_extracted_frames(analysis, loaded)[0]
    component = analysis.ordered[0]
    mask = np.ones((1, 1), dtype=np.bool_)
    global_zeros = np.zeros
    global_array = np.array

    def fail_zeros(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("analysis np.zeros must not affect execution")

    with monkeypatch.context() as scoped:
        scoped.setattr(extract_analysis, "np", _ExtractNumpyProxy(fail_zeros))
        materialized = extract_execution._materialize_frame_crop(analysis, component, frame)
        assert materialized.pixels[1, 1].tolist() == [10, 20, 30, 200]
        assert vars(extract_execution)["np"] is np

    def fail_array(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("execution np.array must not affect analysis")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            extract_execution,
            "np",
            _ExtractExecutionNumpyProxy(fail_array),
        )
        component_map = label_components(mask, 4, 1)
        assert len(component_map.components) == 1
        assert vars(extract_analysis)["np"] is np

    assert vars(extract_analysis)["np"] is np
    assert vars(extract_execution)["np"] is np
    assert np.zeros is global_zeros
    assert np.array is global_array


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
