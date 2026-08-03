# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

import pixipix.imageio as imageio
import pixipix.pipeline.input as pipeline_input
import pixipix.pipeline.publication as pipeline_publication
import pixipix.stages.extract.analysis as extract_analysis
import pixipix.stages.extract.execution as extract_execution
import pixipix.stages.extract.publication as extract_publication
import pixipix.stages.pixelize.execution as pixelize_execution
import pixipix.stages.scale.execution as scale_execution
import pixipix.stages.scale.geometry as scale_geometry
import pixipix.stages.scale.metadata as scale_metadata
import pixipix.stages.scale.planning as scale_planning
from pixipix.config import LoadedConfig, ScaleConfig, load_config
from pixipix.errors import ConfigurationError
from pixipix.models import Dimensions, ScaleStageMetadata, UInt8Image
from pixipix.pipeline.input import (
    InputStageFrame,
    LoadedStageInput,
)
from pixipix.pipeline.publication import OutputFrameImage
from pixipix.resources import ResourceProjection
from pixipix.stages.io import validate_stage_input
from pixipix.stages.scale import (
    ScaleRun,
    ScaleStagePlan,
    premultiplied_box_resize,
    project_scale_stage,
    round_channel_half_away_from_zero,
    round_half_away_from_zero,
    scale_stage,
    transformed_dimension,
)
from tests.helpers import pipeline_config, write_config, write_declared_extract_stage


class _ScaleNumpyProxy:
    def __init__(self, stack: Callable[..., object]) -> None:
        self.stack = stack

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


def _patch_scale_stack(
    monkeypatch: pytest.MonkeyPatch,
    stack: Callable[..., object],
) -> None:
    monkeypatch.setattr(scale_execution, "np", _ScaleNumpyProxy(stack))


class _ScaleImageProxy:
    def __init__(self, fromarray: Callable[..., object]) -> None:
        self.fromarray = fromarray

    def __getattr__(self, name: str) -> object:
        return getattr(Image, name)


def _loaded_scale_case(
    tmp_path: Path,
) -> tuple[LoadedStageInput, LoadedConfig, ScaleStagePlan]:
    config_path = tmp_path / "project.toml"
    write_config(
        config_path,
        pipeline_config(
            names=("a",),
            scale='mode = "explicit-factor"\nfactor = 1.0',
        ),
    )
    loaded = load_config(config_path)
    extracted = tmp_path / "extracted"
    write_declared_extract_stage(extracted, loaded, ((2, 2),))
    validated = validate_stage_input(extracted, "extract")
    plan = project_scale_stage(validated, loaded)
    frames = tuple(
        InputStageFrame(
            frame.name,
            frame.relative_path,
            frame.source_order,
            frame.dimensions,
            np.full(
                (frame.dimensions.height, frame.dimensions.width, 4),
                255,
                dtype=np.uint8,
            ),
        )
        for frame in validated.frames
    )
    return (
        LoadedStageInput(
            validated.identity,
            frames,
            validated.metadata,
            validated.warnings,
        ),
        loaded,
        plan,
    )


def test_round_half_away_from_zero_boundaries() -> None:
    assert [round_half_away_from_zero(value) for value in (1.4, 1.5, 1.6)] == [1, 2, 2]
    assert [round_half_away_from_zero(value) for value in (-1.4, -1.5, -1.6)] == [-1, -2, -2]
    assert round_half_away_from_zero(2.5) == 3
    assert round_half_away_from_zero(0.5) == 1


def test_tiny_nonempty_dimension_stays_nonempty() -> None:
    assert transformed_dimension(1, 0.01) == 1
    assert transformed_dimension(3, 0.5) == 2


@pytest.mark.parametrize(
    ("scale_text", "dimensions", "overrides"),
    [
        (
            'mode = "explicit-factor"\nfactor = 1.5',
            ((3, 5), (8, 2)),
            "",
        ),
        (
            'mode = "reference-frame-width"\nreference_frame = "a"\ntarget_size = 5',
            ((3, 5), (8, 2)),
            "",
        ),
        (
            'mode = "reference-frame-height"\nreference_frame = "a"\ntarget_size = 5',
            ((3, 3), (8, 2)),
            "",
        ),
        (
            'mode = "explicit-factor"\nfactor = 0.01',
            ((1, 3), (2, 1)),
            "[frame_overrides.b]\nscale_multiplier = 2.0",
        ),
    ],
    ids=["explicit", "reference-width", "reference-height", "override-and-minimum-one"],
)
def test_scale_stage_plan_dimensions_stay_synchronized_with_transform_helper(
    tmp_path: Path,
    scale_text: str,
    dimensions: tuple[tuple[int, int], ...],
    overrides: str,
) -> None:
    config_path = tmp_path / "project.toml"
    write_config(
        config_path,
        pipeline_config(
            names=("a", "b"),
            scale=scale_text,
            overrides=overrides,
        ),
    )
    loaded = load_config(config_path)
    extracted = tmp_path / "extracted"
    write_declared_extract_stage(extracted, loaded, dimensions)
    validated = validate_stage_input(extracted, "extract")

    plan = project_scale_stage(validated, loaded)
    scale_config = loaded.config.scale
    assert scale_config is not None
    for source, planned in zip(validated.frames, plan.frames, strict=True):
        expected_width = transformed_dimension(
            source.dimensions.width,
            planned.effective_factor,
        )
        expected_height = transformed_dimension(
            source.dimensions.height,
            planned.effective_factor,
        )
        if source.name == scale_config.reference_frame:
            assert scale_config.target_size is not None
            assert loaded.config.pixelize.source_cell_size is not None
            exact_target = scale_config.target_size * loaded.config.pixelize.source_cell_size
            if scale_config.mode == "reference-frame-width":
                expected_width = exact_target
            else:
                expected_height = exact_target
        assert (
            planned.output_dimensions.width,
            planned.output_dimensions.height,
        ) == (expected_width, expected_height)
    if scale_config.factor == 0.01:
        assert all(
            frame.output_dimensions.width == frame.output_dimensions.height == 1
            for frame in plan.frames
        )


def test_scale_plan_resource_transient_preserves_each_input_output_pair(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "project.toml"
    write_config(
        config_path,
        pipeline_config(
            names=("input-heavy", "output-heavy", "paired-peak"),
            scale='mode = "explicit-factor"\nfactor = 1.0',
            overrides=(
                "[frame_overrides.input-heavy]\n"
                "scale_multiplier = 0.001\n\n"
                "[frame_overrides.output-heavy]\n"
                "scale_multiplier = 32.0"
            ),
        ),
    )
    loaded = load_config(config_path)
    extracted = tmp_path / "extracted"
    write_declared_extract_stage(
        extracted,
        loaded,
        ((1000, 1), (1, 1), (30, 30)),
    )
    validated = validate_stage_input(extracted, "extract")

    plan = project_scale_stage(validated, loaded)
    paired_areas = tuple(
        (
            frame.input_dimensions.width * frame.input_dimensions.height,
            frame.output_dimensions.width * frame.output_dimensions.height,
        )
        for frame in plan.frames
    )
    transients = tuple(
        max(
            36 * input_area,
            24 * input_area + 28 * output_area,
            20 * input_area + 56 * output_area,
            5 * output_area,
        )
        for input_area, output_area in paired_areas
    )

    assert paired_areas == ((1000, 1), (1, 1024), (900, 900))
    assert transients == (36_000, 57_364, 68_400)
    assert transients.index(max(transients)) == 2
    assert plan.projection == ResourceProjection("scale", 1_901, 1_925, 83_704)


def test_scale_stage_consumes_planned_dimensions_without_geometry_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, loaded, plan = _loaded_scale_case(tmp_path)
    planned_frame = replace(plan.frames[0], output_dimensions=Dimensions(1, 1))
    admitted_plan = replace(plan, frames=(planned_frame,))

    def fail_recomputation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("execution recomputed admitted scale geometry")

    monkeypatch.setattr(scale_geometry, "transformed_dimension", fail_recomputation)
    monkeypatch.setattr(scale_planning, "_global_factor", fail_recomputation)
    monkeypatch.setattr(scale_planning, "project_scale_resources", fail_recomputation)

    run = scale_stage(stage, loaded, admitted_plan)

    assert run.frame_images[0].pixels.shape == (1, 1, 4)
    assert run.metadata.frames == admitted_plan.frames


def test_scale_stage_preserves_config_transform_metadata_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, loaded, plan = _loaded_scale_case(tmp_path)
    events: list[str] = []
    real_require = scale_planning._require_scale_config
    real_resize = scale_execution.premultiplied_box_resize
    real_metadata = scale_metadata.build_scale_metadata
    real_run = scale_execution.ScaleRun

    def record_require(value: LoadedConfig) -> ScaleConfig:
        events.append("config")
        return real_require(value)

    def record_resize(pixels: UInt8Image, size: tuple[int, int]) -> UInt8Image:
        events.append("transform")
        return real_resize(pixels, size)

    def record_metadata(
        stage_value: LoadedStageInput,
        loaded_value: LoadedConfig,
        plan_value: ScaleStagePlan,
        config: ScaleConfig,
    ) -> ScaleStageMetadata:
        events.append("metadata")
        return real_metadata(stage_value, loaded_value, plan_value, config)

    def record_run(
        metadata: ScaleStageMetadata,
        frame_images: tuple[OutputFrameImage, ...],
    ) -> ScaleRun:
        events.append("run")
        return real_run(metadata, frame_images)

    monkeypatch.setattr(scale_execution, "_require_scale_config", record_require)
    monkeypatch.setattr(scale_execution, "premultiplied_box_resize", record_resize)
    monkeypatch.setattr(scale_execution, "build_scale_metadata", record_metadata)
    monkeypatch.setattr(scale_execution, "ScaleRun", record_run)

    run = scale_stage(stage, loaded, plan)

    assert events == ["config", "transform", "metadata", "run"]
    assert run.metadata == scale_metadata.build_scale_metadata(
        stage,
        loaded,
        plan,
        real_require(loaded),
    )
    assert type(run) is real_run


def test_missing_scale_config_fails_before_execution_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, loaded, plan = _loaded_scale_case(tmp_path)
    missing_scale = replace(loaded, config=replace(loaded.config, scale=None))

    def fail_late_work(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("missing scale config reached transformation or metadata")

    monkeypatch.setattr(scale_execution, "Image", _ScaleImageProxy(fail_late_work))
    monkeypatch.setattr(scale_execution, "_resize_float_channel", fail_late_work)
    monkeypatch.setattr(scale_execution, "premultiplied_box_resize", fail_late_work)
    monkeypatch.setattr(scale_execution, "build_scale_metadata", fail_late_work)

    with pytest.raises(ConfigurationError) as raised:
        scale_stage(stage, missing_scale, plan)

    assert raised.value.code == "PX_SCALE_CONFIG_001"


def test_channel_rounding_is_separate_and_clamped() -> None:
    assert round_channel_half_away_from_zero(127.5) == 128
    assert round_channel_half_away_from_zero(-2.0) == 0
    assert round_channel_half_away_from_zero(300.0) == 255


def test_premultiplied_resize_has_no_black_fringe() -> None:
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (240, 30, 10, 255)
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output[0, 0].tolist() == [240, 30, 10, 64]


def test_hidden_transparent_rgb_does_not_survive() -> None:
    pixels = np.array(
        [[[255, 0, 255, 0], [0, 200, 80, 255]]],
        dtype=np.uint8,
    )
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output[0, 0].tolist() == [0, 200, 80, 128]


def test_alpha_zero_is_transparent_black_and_input_is_not_mutated() -> None:
    pixels = np.full((2, 2, 4), (10, 20, 30, 0), dtype=np.uint8)
    before = pixels.copy()
    output = premultiplied_box_resize(pixels, (1, 1))
    assert output.tolist() == [[[0, 0, 0, 0]]]
    assert np.array_equal(pixels, before)


def test_opaque_input_matches_ordinary_box() -> None:
    pixels = np.array(
        [
            [[10, 20, 30, 255], [30, 40, 50, 255]],
            [[50, 60, 70, 255], [70, 80, 90, 255]],
        ],
        dtype=np.uint8,
    )
    expected = np.asarray(Image.fromarray(pixels, mode="RGBA").resize((1, 1), Image.Resampling.BOX))
    assert np.array_equal(premultiplied_box_resize(pixels, (1, 1)), expected)


def test_execution_image_binding_affects_real_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.full((2, 2, 4), 255, dtype=np.uint8)
    global_fromarray = Image.fromarray
    neighboring_images = (
        pipeline_input.Image,
        pipeline_publication.Image,
        extract_publication.Image,
        imageio.Image,
    )
    modes: list[object] = []

    def record_fromarray(*args: object, **kwargs: object) -> object:
        modes.append(kwargs.get("mode"))
        return cast(Any, global_fromarray)(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(scale_execution, "Image", _ScaleImageProxy(record_fromarray))
        output = premultiplied_box_resize(pixels, (1, 1))

        assert modes == ["RGBA"]
        assert output.tolist() == [[[255, 255, 255, 255]]]
        assert Image.fromarray is global_fromarray
        assert (
            pipeline_input.Image,
            pipeline_publication.Image,
            extract_publication.Image,
            imageio.Image,
        ) == neighboring_images

    assert scale_execution.Image is Image
    assert Image.fromarray is global_fromarray


def test_execution_image_binding_affects_transparent_float_channel_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.array([[[20, 40, 60, 128], [80, 100, 120, 0]]], dtype=np.uint8)
    global_fromarray = Image.fromarray
    neighboring_images = (
        pipeline_input.Image,
        pipeline_publication.Image,
        extract_publication.Image,
        imageio.Image,
    )
    modes: list[object] = []

    def record_float_fromarray(*args: object, **kwargs: object) -> object:
        if kwargs.get("mode") != "F":
            raise AssertionError("transparent resize reached a non-float Pillow path")
        modes.append(kwargs["mode"])
        return cast(Any, global_fromarray)(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(scale_execution, "Image", _ScaleImageProxy(record_float_fromarray))
        output = premultiplied_box_resize(pixels, (1, 1))

        assert modes == ["F", "F", "F", "F"]
        assert output.shape == (1, 1, 4)
        assert Image.fromarray is global_fromarray
        assert (
            pipeline_input.Image,
            pipeline_publication.Image,
            extract_publication.Image,
            imageio.Image,
        ) == neighboring_images

    assert scale_execution.Image is Image
    assert Image.fromarray is global_fromarray


def test_opaque_fast_path_bypasses_premultiplied_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.full((2, 2, 4), 255, dtype=np.uint8)

    def fail_stack(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("opaque input reached the premultiplied path")

    _patch_scale_stack(monkeypatch, fail_stack)
    output = premultiplied_box_resize(pixels, (1, 1))

    assert output.tolist() == [[[255, 255, 255, 255]]]


def test_transparent_input_reaches_premultiplied_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.array([[[20, 40, 60, 128], [80, 100, 120, 0]]], dtype=np.uint8)
    global_stack = np.stack

    class PremultipliedPathReached(Exception):
        pass

    def mark_stack(*_args: object, **_kwargs: object) -> None:
        raise PremultipliedPathReached("scale premultiplied np.stack reached")

    neighboring_numpy = (
        pixelize_execution.np,
        extract_analysis.np,
        extract_execution.np,
        pipeline_input.np,
    )
    with monkeypatch.context() as scoped:
        _patch_scale_stack(scoped, mark_stack)
        assert np.stack is global_stack
        with pytest.raises(
            PremultipliedPathReached,
            match=r"scale premultiplied np\.stack reached",
        ) as raised:
            premultiplied_box_resize(pixels, (1, 1))
        traceback_names = tuple(entry.name for entry in raised.traceback)
        assert "premultiplied_box_resize" in traceback_names
        assert traceback_names[-1] == "mark_stack"
        assert (
            pixelize_execution.np,
            extract_analysis.np,
            extract_execution.np,
            pipeline_input.np,
        ) == neighboring_numpy
        assert np.stack is global_stack

    assert scale_execution.np is np
    assert np.stack is global_stack


def test_random_opaque_inputs_are_byte_equivalent_to_native_box() -> None:
    generator = np.random.default_rng(7)
    for source_width, source_height in ((2, 3), (3, 5), (7, 4)):
        pixels = generator.integers(0, 256, (source_height, source_width, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        for target in ((1, 1), (2, 2), (5, 3), (9, 8)):
            expected = np.asarray(
                Image.fromarray(pixels, mode="RGBA").resize(target, Image.Resampling.BOX)
            )
            assert np.array_equal(premultiplied_box_resize(pixels, target), expected)


def test_resize_up_down_and_repeat_are_deterministic() -> None:
    pixels = np.array([[[20, 40, 60, 200], [80, 100, 120, 100]]], dtype=np.uint8)
    up = premultiplied_box_resize(pixels, (5, 3))
    down_first = premultiplied_box_resize(up, (1, 1))
    down_second = premultiplied_box_resize(up, (1, 1))
    assert up.shape == (3, 5, 4)
    assert np.array_equal(down_first, down_second)
