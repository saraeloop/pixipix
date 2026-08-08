from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Literal, cast

import pytest

from pixipix.config import load_config
from pixipix.stages.align import AlignmentStagePlan, project_align_stage
from pixipix.stages.io import validate_stage_input
from pixipix.stages.pixelize import PixelizeStagePlan, project_pixelize_stage
from pixipix.stages.scale import ScaleStagePlan, project_scale_stage
from tests.helpers import (
    alignment_config,
    write_config,
    write_declared_extract_stage,
    write_declared_pixelize_stage,
    write_declared_scale_stage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CompatibilityModule = Literal[
    "pixipix.pipeline.input",
    "pixipix.pipeline.publication",
    "pixipix.stages.extract",
    "pixipix.stages.extract.analysis",
    "pixipix.stages.extract.api",
    "pixipix.stages.extract.execution",
    "pixipix.stages.extract.planning",
    "pixipix.stages.extract.publication",
    "pixipix.stages.scale",
    "pixipix.stages.scale.api",
    "pixipix.stages.pixelize",
    "pixipix.stages.pixelize.api",
    "pixipix.stages.align",
    "pixipix.stages.align.api",
    "pixipix.stages.io",
]
SymbolKind = Literal["function", "class", "value"]
Classification = Literal["public", "internal", "private-but-consumed"]


@dataclass(frozen=True, slots=True)
class CompatibilitySymbol:
    module: CompatibilityModule
    name: str
    consumers: tuple[str, ...]
    classification: Classification
    kind: SymbolKind
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class PhysicalLayoutAssumption:
    path: str
    consumer: str
    reason: str


def _symbol(
    module: CompatibilityModule,
    name: str,
    consumers: tuple[str, ...],
    classification: Classification,
    kind: SymbolKind = "function",
    signature: str | None = None,
) -> CompatibilitySymbol:
    return CompatibilitySymbol(module, name, consumers, classification, kind, signature)


MATRIX = (
    _symbol(
        "pixipix.stages.extract",
        "ComponentMap",
        ("tests",),
        "private-but-consumed",
        "class",
        "(components: 'tuple[Component, ...]', labels: 'LabelMap') -> None",
    ),
    _symbol(
        "pixipix.stages.extract.analysis",
        "_Analysis",
        ("tests",),
        "private-but-consumed",
        "class",
        "(source: 'SourceImage', mask: 'BoolMask', component_map: 'ComponentMap', "
        "accepted: 'tuple[Component, ...]', rejected: 'tuple[RejectedComponent, ...]', "
        "ordered: 'tuple[Component, ...]', background: 'BackgroundSummary') -> None",
    ),
    _symbol(
        "pixipix.stages.extract",
        "inspect_source",
        ("production", "tests"),
        "internal",
        signature="(input_path: 'Path', loaded: 'LoadedConfig') -> 'InspectionResult'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "publish_extraction",
        ("production", "tests"),
        "internal",
        signature="(input_path: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'ExtractionResult'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "filter_components",
        ("tests",),
        "private-but-consumed",
        signature="(components: 'tuple[Component, ...]', config: 'ExtractConfig') "
        "-> 'tuple[tuple[Component, ...], tuple[RejectedComponent, ...]]'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "label_components",
        ("tests",),
        "private-but-consumed",
        signature="(mask: 'BoolMask', connectivity: 'int', max_components: 'int') "
        "-> 'ComponentMap'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "order_components",
        ("tests",),
        "private-but-consumed",
        signature="(components: 'tuple[Component, ...]', row_tolerance: 'int') "
        "-> 'tuple[Component, ...]'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "project_extract_resources",
        ("tests",),
        "private-but-consumed",
        signature="(source_area: 'int', frames: 'tuple[ExtractedFrame, ...]') "
        "-> 'ResourceProjection'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "project_extracted_frames",
        ("tests",),
        "private-but-consumed",
        signature="(analysis: '_Analysis', loaded: 'LoadedConfig') -> 'tuple[ExtractedFrame, ...]'",
    ),
    _symbol(
        "pixipix.stages.extract.analysis",
        "_analyze",
        ("tests",),
        "private-but-consumed",
        signature="(input_path: 'Path', loaded: 'LoadedConfig') -> '_Analysis'",
    ),
    _symbol(
        "pixipix.stages.extract.api",
        "_materialize_frame_crop",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(analysis: '_Analysis', component: 'Component', frame: 'ExtractedFrame') "
        "-> 'FrameImage'",
    ),
    _symbol(
        "pixipix.stages.extract.execution",
        "_materialize_frame_crop",
        ("tests",),
        "private-but-consumed",
        signature="(analysis: '_Analysis', component: 'Component', frame: 'ExtractedFrame') "
        "-> 'FrameImage'",
    ),
    _symbol(
        "pixipix.stages.extract.planning",
        "_padded_bounds",
        ("tests",),
        "private-but-consumed",
        signature="(bounds: 'Rect', padding: 'int', width: 'int', height: 'int') -> 'Rect'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "extract_source",
        ("tests",),
        "internal",
        signature="(input_path: 'Path', loaded: 'LoadedConfig') -> 'ExtractionRun'",
    ),
    _symbol(
        "pixipix.stages.extract.analysis",
        "load_source",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(path: 'Path', config: 'SourceConfig') -> 'SourceImage'",
    ),
    _symbol(
        "pixipix.stages.extract.analysis",
        "np",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
    ),
    _symbol(
        "pixipix.stages.extract.execution",
        "np",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
    ),
    _symbol(
        "pixipix.stages.scale",
        "ScaleStagePlan",
        ("architecture",),
        "internal",
        "class",
        "(frames: 'tuple[ScaleFrame, ...]', global_factor: 'float', "
        "source_reference_measurement: 'int | None', "
        "exact_target_source_measurement: 'int | None', "
        "warnings: 'tuple[ProcessingWarning, ...]', "
        "projection: 'ResourceProjection') -> None",
    ),
    _symbol(
        "pixipix.stages.scale",
        "ScaleRun",
        ("compatibility",),
        "internal",
        "class",
        "(metadata: 'ScaleStageMetadata', frame_images: 'tuple[OutputFrameImage, ...]') -> None",
    ),
    _symbol(
        "pixipix.stages.scale",
        "MAX_TRANSFORMED_PIXELS",
        ("compatibility",),
        "internal",
        "value",
    ),
    _symbol(
        "pixipix.stages.scale",
        "scale_stage",
        ("compatibility",),
        "internal",
        signature=(
            "(stage: 'LoadedStageInput', loaded: 'LoadedConfig', plan: 'ScaleStagePlan') "
            "-> 'ScaleRun'"
        ),
    ),
    _symbol(
        "pixipix.stages.scale",
        "publish_scale",
        ("production", "tests"),
        "internal",
        signature="(input_dir: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'ScaleStageMetadata'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "premultiplied_box_resize",
        ("tests",),
        "private-but-consumed",
        signature="(pixels: 'UInt8Image', size: 'tuple[int, int]') -> 'UInt8Image'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "project_scale_stage",
        ("tests",),
        "private-but-consumed",
        signature="(stage: 'ValidatedStageInput', loaded: 'LoadedConfig') -> 'ScaleStagePlan'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "project_scale_resources",
        ("tests",),
        "private-but-consumed",
        signature="(frames: 'tuple[ScaleFrame, ...]') -> 'ResourceProjection'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "round_channel_half_away_from_zero",
        ("production", "tests"),
        "private-but-consumed",
        signature="(value: 'float') -> 'int'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "round_half_away_from_zero",
        ("tests",),
        "private-but-consumed",
        signature="(value: 'float') -> 'int'",
    ),
    _symbol(
        "pixipix.stages.scale",
        "transformed_dimension",
        ("tests",),
        "private-but-consumed",
        signature="(source_dimension: 'int', factor: 'float') -> 'int'",
    ),
    _symbol(
        "pixipix.stages.scale.api",
        "decode_stage_input",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "PreparedCellGrid",
        ("compatibility",),
        "internal",
        "class",
        "(pixels: 'UInt8Image', top_padding: 'int', right_padding: 'int', "
        "top_crop: 'int', right_crop: 'int', warning: 'ProcessingWarning | None') -> None",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "CellGridProjection",
        ("compatibility",),
        "internal",
        "class",
        "(input_dimensions: 'Dimensions', prepared_dimensions: 'Dimensions', "
        "logical_output_dimensions: 'Dimensions', top_padding: 'int', "
        "right_padding: 'int', top_crop: 'int', right_crop: 'int', "
        "warning: 'ProcessingWarning | None') -> None",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "PixelizeRun",
        ("compatibility",),
        "internal",
        "class",
        "(metadata: 'PixelizeStageMetadata', frame_images: 'tuple[OutputFrameImage, ...]') -> None",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "PixelizeStagePlan",
        ("architecture",),
        "internal",
        "class",
        "(frames: 'tuple[PixelizeFrame, ...]', "
        "cell_grids: 'tuple[CellGridProjection, ...]', "
        "warnings: 'tuple[ProcessingWarning, ...]', "
        "projection: 'ResourceProjection') -> None",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "MAX_PREPARED_PIXELS",
        ("compatibility",),
        "internal",
        "value",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "pixelize_stage",
        ("compatibility",),
        "internal",
        signature=(
            "(stage: 'LoadedStageInput', loaded: 'LoadedConfig', plan: 'PixelizeStagePlan') "
            "-> 'PixelizeRun'"
        ),
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "apply_alpha_policy",
        ("tests",),
        "private-but-consumed",
        signature="(rgba: 'tuple[int, int, int, int]', policy: 'str', threshold: 'int') "
        "-> 'tuple[int, int, int, int]'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "pixelize_prepared_grid",
        ("tests",),
        "private-but-consumed",
        signature="(pixels: 'UInt8Image', cell_size: 'int', strategy: 'str', "
        "alpha_policy: 'str', alpha_threshold: 'int') -> 'UInt8Image'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "prepare_cell_grid",
        ("tests",),
        "private-but-consumed",
        signature="(pixels: 'UInt8Image', cell_size: 'int', policy: 'str', "
        "frame_name: 'str', *, projection: 'CellGridProjection | None' = None) "
        "-> 'PreparedCellGrid'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "project_cell_grid",
        ("tests",),
        "private-but-consumed",
        signature="(dimensions: 'Dimensions', cell_size: 'int', policy: 'str', "
        "frame_name: 'str') -> 'CellGridProjection'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "project_pixelize_resources",
        ("tests",),
        "private-but-consumed",
        signature="(frames: 'tuple[PixelizeFrame, ...]', cell_size: 'int') -> 'ResourceProjection'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "project_pixelize_stage",
        ("tests",),
        "private-but-consumed",
        signature="(stage: 'ValidatedStageInput', loaded: 'LoadedConfig') -> 'PixelizeStagePlan'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "publish_pixelize",
        ("production", "tests"),
        "internal",
        signature="(input_dir: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'PixelizeStageMetadata'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "representative_pixel",
        ("tests",),
        "private-but-consumed",
        signature="(cell: 'UInt8Image', strategy: 'str') -> 'tuple[int, int, int, int]'",
    ),
    _symbol(
        "pixipix.stages.pixelize",
        "round_channel_half_away_from_zero",
        ("production", "tests"),
        "private-but-consumed",
        signature="(value: 'float') -> 'int'",
    ),
    _symbol(
        "pixipix.stages.pixelize.api",
        "decode_stage_input",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol(
        "pixipix.stages.align",
        "AlignmentStagePlan",
        ("architecture",),
        "internal",
        "class",
        "(frames: 'tuple[AlignmentFrame, ...]', "
        "clipping_findings: 'tuple[AlignmentClippingFinding, ...]', "
        "warnings: 'tuple[ProcessingWarning, ...]', "
        "projection: 'ResourceProjection') -> None",
    ),
    _symbol("pixipix.stages.align", "EMPTY_RECTANGLE", ("tests",), "private-but-consumed", "value"),
    _symbol(
        "pixipix.stages.align",
        "calculate_alignment_frame",
        ("tests",),
        "private-but-consumed",
        signature="(*, name: 'str', relative_path: 'PurePosixPath', source_order: 'int', "
        "input_width: 'int', input_height: 'int', output: 'OutputConfig', "
        "dx: 'int' = 0, dy: 'int' = 0) -> 'AlignmentFrame'",
    ),
    _symbol(
        "pixipix.stages.align",
        "compose_aligned_canvas",
        ("tests",),
        "private-but-consumed",
        signature="(pixels: 'UInt8Image', frame: 'AlignmentFrame') -> 'UInt8Image'",
    ),
    _symbol(
        "pixipix.stages.align",
        "mathematical_floor_center",
        ("tests",),
        "private-but-consumed",
        signature="(canvas_size: 'int', input_size: 'int') -> 'int'",
    ),
    _symbol(
        "pixipix.stages.align",
        "project_align_resources",
        ("tests",),
        "private-but-consumed",
        signature="(frames: 'tuple[AlignmentFrame, ...]', canvas_width: 'int', "
        "canvas_height: 'int') -> 'ResourceProjection'",
    ),
    _symbol(
        "pixipix.stages.align",
        "project_align_stage",
        ("tests",),
        "private-but-consumed",
        signature="(stage: 'ValidatedStageInput', loaded: 'LoadedConfig') -> 'AlignmentStagePlan'",
    ),
    _symbol(
        "pixipix.stages.align",
        "publish_align",
        ("production", "tests"),
        "internal",
        signature="(input_dir: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'AlignmentStageMetadata'",
    ),
    _symbol(
        "pixipix.stages.align.api",
        "decode_stage_input",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol("pixipix.pipeline.input", "InputStageFrame", ("facade",), "internal", "class"),
    _symbol("pixipix.pipeline.input", "LoadedStageInput", ("production",), "internal", "class"),
    _symbol("pixipix.pipeline.input", "ValidatedStageFrame", ("facade",), "internal", "class"),
    _symbol("pixipix.pipeline.input", "ValidatedStageInput", ("production",), "internal", "class"),
    _symbol(
        "pixipix.pipeline.input",
        "validate_stage_input",
        ("production",),
        "internal",
        signature=(
            "(root: 'Path', expected_stage: \"Literal['extract', 'scale', 'pixelize']\") "
            "-> 'ValidatedStageInput'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.input",
        "decode_stage_input",
        ("production",),
        "internal",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol(
        "pixipix.pipeline.input",
        "load_stage_input",
        ("facade",),
        "internal",
        signature=(
            "(root: 'Path', expected_stage: \"Literal['extract', 'scale', 'pixelize']\") "
            "-> 'LoadedStageInput'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.input",
        "Image",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
    ),
    _symbol(
        "pixipix.pipeline.input",
        "np",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "OutputFrameImage",
        ("production",),
        "internal",
        "class",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "validate_stage_output_target",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: 'StageName', *, force: 'bool' = False, "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "publish_stage_output",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: 'StageName', metadata: 'object', "
            "frames: 'tuple[OutputFrameImage, ...]', *, force: 'bool' = False, "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "publish_run_output",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', build_run: 'RunBuilder[T]', "
            "complete_run_validator: 'CompleteRunValidator', *, force: 'bool' = False) -> 'T'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "_valid_owned_output",
        ("facade",),
        "private-but-consumed",
        signature=(
            "(path: 'Path', stage: 'StageName', "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'bool'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "_validate_output_location",
        ("tests",),
        "private-but-consumed",
        signature="(output: 'Path') -> 'None'",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "_validate_staged",
        ("monkeypatch",),
        "private-but-consumed",
        signature=(
            "(root: 'Path', stage: 'StageName', metadata: 'object', "
            "frames: 'tuple[OutputFrameImage, ...]') -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "_remove_tree",
        ("tests",),
        "private-but-consumed",
        signature="(path: 'Path', parent: 'Path', prefix: 'str') -> 'bool'",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "Image",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "write_json",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(path: 'Path', value: 'object') -> 'None'",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "write_png",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(path: 'Path', pixels: 'UInt8Image') -> 'None'",
    ),
    _symbol("pixipix.stages.io", "InputStageFrame", ("facade",), "internal", "class"),
    _symbol("pixipix.stages.io", "ValidatedStageInput", ("production",), "internal", "class"),
    _symbol("pixipix.stages.io", "LoadedStageInput", ("production",), "internal", "class"),
    _symbol("pixipix.stages.io", "ValidatedStageFrame", ("facade",), "internal", "class"),
    _symbol("pixipix.stages.io", "OutputFrameImage", ("production",), "internal", "class"),
    _symbol(
        "pixipix.stages.io",
        "validate_stage_input",
        ("production", "tests"),
        "internal",
        signature=(
            "(root: 'Path', expected_stage: \"Literal['extract', 'scale', 'pixelize']\") "
            "-> 'ValidatedStageInput'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "decode_stage_input",
        ("production",),
        "internal",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol(
        "pixipix.stages.io",
        "load_stage_input",
        ("smoke", "tests"),
        "internal",
        signature=(
            "(root: 'Path', expected_stage: \"Literal['extract', 'scale', 'pixelize']\") "
            "-> 'LoadedStageInput'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "validate_stage_output_target",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: 'StageName', *, force: 'bool' = False, "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "publish_stage_output",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: 'StageName', metadata: 'object', "
            "frames: 'tuple[OutputFrameImage, ...]', *, force: 'bool' = False, "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "_valid_owned_output",
        ("smoke",),
        "private-but-consumed",
        signature=(
            "(path: 'Path', stage: 'StageName', "
            "owned_metadata_validator: 'OwnedMetadataValidator | None' = None) -> 'bool'"
        ),
    ),
)

PHYSICAL_LAYOUT = (
    PhysicalLayoutAssumption(
        "pixipix/stages/align/execution.py",
        "tests/release/test_smoke_distribution.py",
        "installed-wheel corruption member selected by exact archive path",
    ),
)


def _current_consumers() -> list[tuple[Path, str]]:
    consumers: list[tuple[Path, str]] = []
    for root_name in ("src", "tests", "scripts"):
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative.startswith("tests/architecture/"):
                continue
            if relative.startswith("src/pixipix/stages/align/"):
                continue
            category = (
                "production"
                if relative.startswith("src/")
                else "smoke"
                if relative.startswith("scripts/")
                else "tests"
            )
            consumers.append((path, category))
    return consumers


def _consumed_symbols() -> set[tuple[str, str]]:
    targets = {entry.module for entry in MATRIX}
    consumed: set[tuple[str, str]] = set()
    for path, _category in _current_consumers():
        if path == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in targets:
                consumed.update((node.module, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == "pixipix.stages":
                for alias in node.names:
                    module = f"pixipix.stages.{alias.name}"
                    if module in targets:
                        aliases[alias.asname or alias.name] = module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in targets:
                        aliases[alias.asname or alias.name.rsplit(".", 1)[-1]] = alias.name
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                consumed.add((aliases[node.value.id], node.attr))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in aliases
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                consumed.add((aliases[node.args[0].id], node.args[1].value))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in targets:
                    continue
                for module in sorted(targets, key=len, reverse=True):
                    prefix = module + "."
                    if node.value.startswith(prefix):
                        consumed.add((module, node.value.removeprefix(prefix).split(".", 1)[0]))
                        break
    return consumed


type PatchAuthorityRegistries = Mapping[str, frozenset[str]]


DECLARED_PATCH_SEAMS = frozenset(
    {
        "pixipix.pipeline.publication.write_json",
        "pixipix.pipeline.publication.write_png",
        "pixipix.pipeline.publication._validate_staged",
        "pixipix.stages.align.api.decode_stage_input",
        "pixipix.stages.extract.analysis.load_source",
        "pixipix.stages.extract.api._materialize_frame_crop",
        "pixipix.stages.pixelize.api.decode_stage_input",
        "pixipix.stages.scale.api.decode_stage_input",
    }
)

OWNER_LOCAL_DEPENDENCIES = frozenset(
    {
        "pixipix.imageio.Image",
        "pixipix.pipeline.input.Image",
        "pixipix.pipeline.input.np",
        "pixipix.pipeline.publication.Image",
        "pixipix.stages.extract.analysis.np",
        "pixipix.stages.extract.execution.np",
        "pixipix.stages.pixelize.execution.np",
        "pixipix.stages.scale.execution.Image",
        "pixipix.stages.scale.execution.np",
    }
)

BROAD_NECESSARY_SEAMS = frozenset({"pathlib.Path.replace"})

DELIBERATE_NON_SEAMS = frozenset(
    {
        "pixipix.pipeline.publication._prepare_target",
        "pixipix.pipeline.publication._remove_tree",
    }
)

EXPECTED_DECLARED_PATCH_SEAMS = frozenset(
    {
        "pixipix.pipeline.publication.write_json",
        "pixipix.pipeline.publication.write_png",
        "pixipix.pipeline.publication._validate_staged",
        "pixipix.stages.align.api.decode_stage_input",
        "pixipix.stages.extract.analysis.load_source",
        "pixipix.stages.extract.api._materialize_frame_crop",
        "pixipix.stages.pixelize.api.decode_stage_input",
        "pixipix.stages.scale.api.decode_stage_input",
    }
)

EXPECTED_OWNER_LOCAL_DEPENDENCIES = frozenset(
    {
        "pixipix.imageio.Image",
        "pixipix.pipeline.input.Image",
        "pixipix.pipeline.input.np",
        "pixipix.pipeline.publication.Image",
        "pixipix.stages.extract.analysis.np",
        "pixipix.stages.extract.execution.np",
        "pixipix.stages.pixelize.execution.np",
        "pixipix.stages.scale.execution.Image",
        "pixipix.stages.scale.execution.np",
    }
)

EXPECTED_BROAD_NECESSARY_SEAMS = frozenset({"pathlib.Path.replace"})

EXPECTED_DELIBERATE_NON_SEAMS = frozenset(
    {
        "pixipix.pipeline.publication._prepare_target",
        "pixipix.pipeline.publication._remove_tree",
    }
)

PATCH_AUTHORITY_REGISTRIES: PatchAuthorityRegistries = MappingProxyType(
    {
        "broad necessary seam": BROAD_NECESSARY_SEAMS,
        "declared seam": DECLARED_PATCH_SEAMS,
        "deliberate non-seam": DELIBERATE_NON_SEAMS,
        "owner-local dependency": OWNER_LOCAL_DEPENDENCIES,
    }
)

EXPECTED_PATCH_AUTHORITY_REGISTRIES: PatchAuthorityRegistries = MappingProxyType(
    {
        "broad necessary seam": EXPECTED_BROAD_NECESSARY_SEAMS,
        "declared seam": EXPECTED_DECLARED_PATCH_SEAMS,
        "deliberate non-seam": EXPECTED_DELIBERATE_NON_SEAMS,
        "owner-local dependency": EXPECTED_OWNER_LOCAL_DEPENDENCIES,
    }
)

PATCH_AUTHORITY_CLASSIFICATIONS = MappingProxyType(
    {
        **{binding: "declared seam" for binding in DECLARED_PATCH_SEAMS},
        **{binding: "owner-local dependency" for binding in OWNER_LOCAL_DEPENDENCIES},
        **{binding: "broad necessary seam" for binding in BROAD_NECESSARY_SEAMS},
        **{binding: "deliberate non-seam" for binding in DELIBERATE_NON_SEAMS},
    }
)

EXPECTED_PATCH_CLASSIFICATIONS = MappingProxyType(
    {
        **{binding: "declared seam" for binding in EXPECTED_DECLARED_PATCH_SEAMS},
        **{binding: "owner-local dependency" for binding in EXPECTED_OWNER_LOCAL_DEPENDENCIES},
        **{binding: "broad necessary seam" for binding in EXPECTED_BROAD_NECESSARY_SEAMS},
        **{binding: "deliberate non-seam" for binding in EXPECTED_DELIBERATE_NON_SEAMS},
    }
)

_EXPECTED_REGISTRY_NAMES = frozenset(
    {
        "broad necessary seam",
        "declared seam",
        "deliberate non-seam",
        "owner-local dependency",
    }
)
_REPRESENTATIVE_IMPLEMENTATION_PROBE = "pixipix.stages.align.api.validate_stage_input"
_REPRESENTATIVE_HARNESS_PATCH = "pixipix.cli.inspect_source"
_WRONG_OWNER_SUBSTITUTION = "pixipix.pipeline.input.decode_stage_input"
_FOUNDATIONAL_LIBRARY_SUBSTITUTION = "PIL.Image"
_KNOWN_INVALID_CLASSIFICATIONS = {
    _REPRESENTATIVE_IMPLEMENTATION_PROBE: "implementation probe",
    _REPRESENTATIVE_HARNESS_PATCH: "harness-only patch",
    _WRONG_OWNER_SUBSTITUTION: "consumer runtime owner",
    _FOUNDATIONAL_LIBRARY_SUBSTITUTION: "consumer-owned module binding",
}


def _resolve_patch_binding(binding: str) -> object:
    parts = binding.split(".")
    for boundary in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[boundary:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(f"unable to import patch binding {binding}")


def _registry_members(registries: PatchAuthorityRegistries) -> dict[str, frozenset[str]]:
    return dict(registries)


def _registry_classifications(
    members: Mapping[str, frozenset[str]],
    *,
    expected_classifications: Mapping[str, str],
) -> dict[str, str]:
    classifications_by_binding: dict[str, list[str]] = {}
    for classification, bindings in members.items():
        for binding in bindings:
            classifications_by_binding.setdefault(binding, []).append(classification)
    for binding, classifications in sorted(classifications_by_binding.items()):
        if len(classifications) > 1:
            actual = ", ".join(sorted(classifications))
            expected = expected_classifications.get(binding, "one compatible registry")
            raise _patch_registry_error(
                registry=actual,
                binding=binding,
                actual=actual,
                expected=expected,
                remediation="classify the binding in exactly one compatible stable registry",
            )
    return {
        binding: classifications[0]
        for binding, classifications in classifications_by_binding.items()
    }


def _patch_registry_error(
    *,
    registry: str,
    binding: str,
    actual: str,
    expected: str,
    remediation: str,
) -> AssertionError:
    return AssertionError(
        f"registry={registry}; binding={binding}; actual classification={actual}; "
        f"expected classification={expected}; remediation={remediation}"
    )


def _validate_patch_authority_registries(
    candidate_registries: PatchAuthorityRegistries,
    expected_registries: PatchAuthorityRegistries,
    expected_classifications: Mapping[str, str],
) -> None:
    candidate_members = _registry_members(candidate_registries)
    expected_members = _registry_members(expected_registries)
    if frozenset(candidate_members) != _EXPECTED_REGISTRY_NAMES:
        raise AssertionError("candidate patch-authority registry names are incomplete or unknown")
    if frozenset(expected_members) != _EXPECTED_REGISTRY_NAMES:
        raise AssertionError("expected patch-authority registry names are incomplete or unknown")

    candidate_classifications = _registry_classifications(
        candidate_members,
        expected_classifications=expected_classifications,
    )
    derived_expected_classifications = _registry_classifications(
        expected_members,
        expected_classifications=expected_classifications,
    )
    if dict(expected_classifications) != derived_expected_classifications:
        all_expected_bindings = sorted(
            set(expected_classifications) | set(derived_expected_classifications)
        )
        for binding in all_expected_bindings:
            supplied = expected_classifications.get(binding, "unregistered")
            derived = derived_expected_classifications.get(binding, "unregistered")
            if supplied != derived:
                raise _patch_registry_error(
                    registry="expected contract",
                    binding=binding,
                    actual=supplied,
                    expected=derived,
                    remediation="derive classifications from the independent expected literals",
                )
        raise AssertionError("expected patch classifications do not match expected registries")

    all_bindings = sorted(set(candidate_classifications) | set(derived_expected_classifications))
    for binding in all_bindings:
        actual = candidate_classifications.get(binding, "omitted")
        required_classification = expected_classifications.get(binding)
        if required_classification is None:
            required_classification = _KNOWN_INVALID_CLASSIFICATIONS.get(binding, "unregistered")
        if required_classification != actual:
            registry = actual if actual != "omitted" else required_classification
            raise _patch_registry_error(
                registry=registry,
                binding=binding,
                actual=actual,
                expected=required_classification,
                remediation="use the locked execution-effective binding and classification",
            )

    for binding in sorted(candidate_classifications):
        assert _resolve_patch_binding(binding) is not None


def test_matrix_covers_every_current_consumed_stage_symbol() -> None:
    protected = {(entry.module, entry.name) for entry in MATRIX}
    missing = sorted(_consumed_symbols() - protected)
    assert not missing, f"consumed stage symbols missing from compatibility matrix: {missing}"


def test_every_compatibility_symbol_imports_with_expected_shape() -> None:
    for entry in MATRIX:
        module: ModuleType = importlib.import_module(entry.module)
        value = getattr(module, entry.name)
        if entry.kind == "function":
            assert entry.signature is not None
            assert inspect.isfunction(value), f"{entry.module}.{entry.name} is not a function"
        elif entry.kind == "class":
            assert inspect.isclass(value), f"{entry.module}.{entry.name} is not a class"
        if entry.signature is not None:
            assert str(inspect.signature(value)) == entry.signature


def test_stable_patch_authority_registries_are_exact_and_resolve() -> None:
    _validate_patch_authority_registries(
        PATCH_AUTHORITY_REGISTRIES,
        EXPECTED_PATCH_AUTHORITY_REGISTRIES,
        EXPECTED_PATCH_CLASSIFICATIONS,
    )
    assert PATCH_AUTHORITY_REGISTRIES == EXPECTED_PATCH_AUTHORITY_REGISTRIES
    assert PATCH_AUTHORITY_REGISTRIES is not EXPECTED_PATCH_AUTHORITY_REGISTRIES
    assert DECLARED_PATCH_SEAMS is not EXPECTED_DECLARED_PATCH_SEAMS
    assert OWNER_LOCAL_DEPENDENCIES is not EXPECTED_OWNER_LOCAL_DEPENDENCIES
    assert BROAD_NECESSARY_SEAMS is not EXPECTED_BROAD_NECESSARY_SEAMS
    assert DELIBERATE_NON_SEAMS is not EXPECTED_DELIBERATE_NON_SEAMS
    assert PATCH_AUTHORITY_CLASSIFICATIONS == EXPECTED_PATCH_CLASSIFICATIONS
    assert PATCH_AUTHORITY_CLASSIFICATIONS is not EXPECTED_PATCH_CLASSIFICATIONS
    assert len(DECLARED_PATCH_SEAMS) == 8
    assert len(OWNER_LOCAL_DEPENDENCIES) == 9
    assert {"pathlib.Path.replace"} == BROAD_NECESSARY_SEAMS
    assert len(DELIBERATE_NON_SEAMS) == 2


def test_candidate_registry_copies_cannot_mutate_expected_contract() -> None:
    expected_before = _registry_members(EXPECTED_PATCH_AUTHORITY_REGISTRIES)
    expected_classifications_before = dict(EXPECTED_PATCH_CLASSIFICATIONS)

    missing_declared = set(DECLARED_PATCH_SEAMS)
    missing_declared.remove("pixipix.stages.align.api.decode_stage_input")
    missing_candidate = dict(PATCH_AUTHORITY_REGISTRIES)
    missing_candidate["declared seam"] = frozenset(missing_declared)
    with pytest.raises(AssertionError, match="actual classification=omitted"):
        _validate_patch_authority_registries(
            missing_candidate,
            EXPECTED_PATCH_AUTHORITY_REGISTRIES,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )

    extra_declared = set(DECLARED_PATCH_SEAMS)
    extra_declared.add("pixipix.stages.align.api.validate_stage_input")
    extra_candidate = dict(PATCH_AUTHORITY_REGISTRIES)
    extra_candidate["declared seam"] = frozenset(extra_declared)
    with pytest.raises(AssertionError, match="expected classification=implementation probe"):
        _validate_patch_authority_registries(
            extra_candidate,
            EXPECTED_PATCH_AUTHORITY_REGISTRIES,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )

    missing_registry = dict(PATCH_AUTHORITY_REGISTRIES)
    missing_registry.pop("broad necessary seam")
    with pytest.raises(AssertionError, match="registry names are incomplete or unknown"):
        _validate_patch_authority_registries(
            missing_registry,
            EXPECTED_PATCH_AUTHORITY_REGISTRIES,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )

    unknown_registry = dict(PATCH_AUTHORITY_REGISTRIES)
    unknown_registry["unknown registry"] = frozenset()
    with pytest.raises(AssertionError, match="registry names are incomplete or unknown"):
        _validate_patch_authority_registries(
            unknown_registry,
            EXPECTED_PATCH_AUTHORITY_REGISTRIES,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )

    assert _registry_members(EXPECTED_PATCH_AUTHORITY_REGISTRIES) == expected_before
    assert dict(EXPECTED_PATCH_CLASSIFICATIONS) == expected_classifications_before


def test_expected_registry_contract_is_independently_controlled() -> None:
    candidate_before = _registry_members(PATCH_AUTHORITY_REGISTRIES)
    expected_before = _registry_members(EXPECTED_PATCH_AUTHORITY_REGISTRIES)
    expected_classifications_before = dict(EXPECTED_PATCH_CLASSIFICATIONS)
    _validate_patch_authority_registries(
        PATCH_AUTHORITY_REGISTRIES,
        EXPECTED_PATCH_AUTHORITY_REGISTRIES,
        EXPECTED_PATCH_CLASSIFICATIONS,
    )

    omitted_binding = "pixipix.stages.align.api.decode_stage_input"
    modified_expected_declared = set(EXPECTED_DECLARED_PATCH_SEAMS)
    modified_expected_declared.remove(omitted_binding)
    modified_expected = dict(EXPECTED_PATCH_AUTHORITY_REGISTRIES)
    modified_expected["declared seam"] = frozenset(modified_expected_declared)
    modified_expected_classifications = dict(EXPECTED_PATCH_CLASSIFICATIONS)
    modified_expected_classifications.pop(omitted_binding)

    with pytest.raises(AssertionError) as raised:
        _validate_patch_authority_registries(
            PATCH_AUTHORITY_REGISTRIES,
            modified_expected,
            modified_expected_classifications,
        )
    diagnostic = str(raised.value)
    assert f"binding={omitted_binding}" in diagnostic
    assert "actual classification=declared seam" in diagnostic
    assert "expected classification=unregistered" in diagnostic

    assert _registry_members(PATCH_AUTHORITY_REGISTRIES) == candidate_before
    assert _registry_members(EXPECTED_PATCH_AUTHORITY_REGISTRIES) == expected_before
    assert dict(EXPECTED_PATCH_CLASSIFICATIONS) == expected_classifications_before

    missing_expected_registry = dict(EXPECTED_PATCH_AUTHORITY_REGISTRIES)
    missing_expected_registry.pop("broad necessary seam")
    with pytest.raises(AssertionError, match="registry names are incomplete or unknown"):
        _validate_patch_authority_registries(
            PATCH_AUTHORITY_REGISTRIES,
            missing_expected_registry,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )


PatchRegistryMutation = Literal[
    "omit-declared",
    "add-non-seam",
    "add-probe",
    "add-harness",
    "substitute-wrong-owner",
    "substitute-foundational-library",
    "omit-dependency",
    "omit-non-seam",
    "double-classify",
]


def _mutated_patch_registries(mutation: PatchRegistryMutation) -> PatchAuthorityRegistries:
    declared = set(DECLARED_PATCH_SEAMS)
    dependencies = set(OWNER_LOCAL_DEPENDENCIES)
    broad = set(BROAD_NECESSARY_SEAMS)
    non_seams = set(DELIBERATE_NON_SEAMS)
    if mutation == "omit-declared":
        declared.remove("pixipix.stages.align.api.decode_stage_input")
    elif mutation == "add-non-seam":
        declared.add("pixipix.pipeline.publication._remove_tree")
    elif mutation == "add-probe":
        declared.add(_REPRESENTATIVE_IMPLEMENTATION_PROBE)
    elif mutation == "add-harness":
        declared.add(_REPRESENTATIVE_HARNESS_PATCH)
    elif mutation == "substitute-wrong-owner":
        declared.remove("pixipix.stages.align.api.decode_stage_input")
        declared.add(_WRONG_OWNER_SUBSTITUTION)
    elif mutation == "substitute-foundational-library":
        dependencies.remove("pixipix.stages.scale.execution.Image")
        dependencies.add(_FOUNDATIONAL_LIBRARY_SUBSTITUTION)
    elif mutation == "omit-dependency":
        dependencies.remove("pixipix.stages.scale.execution.Image")
    elif mutation == "omit-non-seam":
        non_seams.remove("pixipix.pipeline.publication._remove_tree")
    elif mutation == "double-classify":
        dependencies.add("pixipix.pipeline.publication.write_png")
    return {
        "broad necessary seam": frozenset(broad),
        "declared seam": frozenset(declared),
        "deliberate non-seam": frozenset(non_seams),
        "owner-local dependency": frozenset(dependencies),
    }


@pytest.mark.parametrize(
    ("mutation", "binding", "expected"),
    [
        ("omit-declared", "pixipix.stages.align.api.decode_stage_input", "declared seam"),
        (
            "add-non-seam",
            "pixipix.pipeline.publication._remove_tree",
            "deliberate non-seam",
        ),
        ("add-probe", _REPRESENTATIVE_IMPLEMENTATION_PROBE, "implementation probe"),
        ("add-harness", _REPRESENTATIVE_HARNESS_PATCH, "harness-only patch"),
        ("substitute-wrong-owner", _WRONG_OWNER_SUBSTITUTION, "consumer runtime owner"),
        (
            "substitute-foundational-library",
            _FOUNDATIONAL_LIBRARY_SUBSTITUTION,
            "consumer-owned module binding",
        ),
        ("omit-dependency", "pixipix.stages.scale.execution.Image", "owner-local dependency"),
        (
            "omit-non-seam",
            "pixipix.pipeline.publication._remove_tree",
            "deliberate non-seam",
        ),
        ("double-classify", "pixipix.pipeline.publication.write_png", "declared seam"),
    ],
)
def test_stable_patch_registry_rejects_invalid_mutation(
    mutation: PatchRegistryMutation,
    binding: str,
    expected: str,
) -> None:
    with pytest.raises(AssertionError) as raised:
        _validate_patch_authority_registries(
            _mutated_patch_registries(mutation),
            EXPECTED_PATCH_AUTHORITY_REGISTRIES,
            EXPECTED_PATCH_CLASSIFICATIONS,
        )

    diagnostic = str(raised.value)
    assert f"binding={binding}" in diagnostic
    assert "registry=" in diagnostic
    assert "actual classification=" in diagnostic
    assert f"expected classification={expected}" in diagnostic
    assert "remediation=" in diagnostic


def test_stages_io_facade_reexports_exact_pipeline_objects() -> None:
    facade = importlib.import_module("pixipix.stages.io")
    pipeline_input = importlib.import_module("pixipix.pipeline.input")
    publication = importlib.import_module("pixipix.pipeline.publication")
    owners = {
        "InputStageFrame": pipeline_input,
        "LoadedStageInput": pipeline_input,
        "ValidatedStageFrame": pipeline_input,
        "ValidatedStageInput": pipeline_input,
        "decode_stage_input": pipeline_input,
        "load_stage_input": pipeline_input,
        "validate_stage_input": pipeline_input,
        "OutputFrameImage": publication,
        "_valid_owned_output": publication,
        "publish_stage_output": publication,
        "validate_stage_output_target": publication,
    }
    for name, owner in owners.items():
        assert getattr(facade, name) is getattr(owner, name)
    assert not hasattr(facade, "Image")
    assert not hasattr(facade, "write_json")
    assert not hasattr(facade, "write_png")


def test_stages_io_facade_is_static_and_definition_free() -> None:
    facade_path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "io.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert [
        (node.level, node.module, tuple(alias.name for alias in node.names)) for node in imports
    ] == [
        (
            2,
            "pipeline.input",
            (
                "InputStageFrame",
                "LoadedStageInput",
                "ValidatedStageFrame",
                "ValidatedStageInput",
                "decode_stage_input",
                "load_stage_input",
                "validate_stage_input",
            ),
        ),
        (
            2,
            "pipeline.publication",
            (
                "OutputFrameImage",
                "_valid_owned_output",
                "publish_stage_output",
                "validate_stage_output_target",
            ),
        ),
    ]
    assert all(alias.asname == alias.name for node in imports for alias in node.names)
    assert all(isinstance(node, (ast.Expr, ast.ImportFrom)) for node in tree.body)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def test_align_consumer_import_path_resolves_to_package_member() -> None:
    module = importlib.import_module("pixipix.stages.align")
    assert module.__file__ is not None
    module_file = Path(module.__file__).resolve()

    assert module.__spec__ is not None
    assert module.__spec__.submodule_search_locations is not None
    assert module_file.relative_to(PROJECT_ROOT).as_posix() == (
        "src/pixipix/stages/align/__init__.py"
    )
    assert not (PROJECT_ROOT / "src" / "pixipix" / "stages" / "align.py").exists()


def test_align_facade_reexports_exact_internal_objects() -> None:
    facade = importlib.import_module("pixipix.stages.align")
    api = importlib.import_module("pixipix.stages.align.api")
    execution = importlib.import_module("pixipix.stages.align.execution")
    geometry = importlib.import_module("pixipix.stages.align.geometry")
    planning = importlib.import_module("pixipix.stages.align.planning")

    owners = {
        "AlignmentRun": execution,
        "AlignmentStagePlan": planning,
        "EMPTY_RECTANGLE": geometry,
        "align_stage": execution,
        "calculate_alignment_frame": geometry,
        "clipping_finding": planning,
        "compose_aligned_canvas": execution,
        "mathematical_floor_center": geometry,
        "project_align_resources": planning,
        "project_align_stage": planning,
        "publish_align": api,
    }
    for name, owner in owners.items():
        assert getattr(facade, name) is getattr(owner, name)
    assert not hasattr(facade, "decode_stage_input")
    assert api.decode_stage_input is not None


def test_align_facade_is_relative_grouped_and_definition_free() -> None:
    facade_path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "align" / "__init__.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
    relative_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level
    ]
    expected = {
        "api": ("publish_align",),
        "execution": ("AlignmentRun", "align_stage", "compose_aligned_canvas"),
        "geometry": (
            "EMPTY_RECTANGLE",
            "calculate_alignment_frame",
            "mathematical_floor_center",
        ),
        "planning": (
            "AlignmentStagePlan",
            "clipping_finding",
            "project_align_resources",
            "project_align_stage",
        ),
    }

    assert {
        node.module: tuple(alias.name for alias in node.names) for node in relative_imports
    } == expected
    assert all(node.level == 1 for node in relative_imports)
    assert all(alias.asname == alias.name for node in relative_imports for alias in node.names)
    assert all(isinstance(node, (ast.Expr, ast.ImportFrom)) for node in tree.body)
    expressions = [node for node in tree.body if isinstance(node, ast.Expr)]
    assert len(expressions) == 1
    assert isinstance(expressions[0].value, ast.Constant)
    assert isinstance(expressions[0].value.value, str)

    definitions = 0
    for path in sorted((facade_path.parent).glob("*.py")):
        module_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions += sum(
            isinstance(node, ast.ClassDef) and node.name == "AlignmentStagePlan"
            for node in module_tree.body
        )
    assert definitions == 1


def test_installed_smoke_private_import_and_physical_member_are_explicit() -> None:
    smoke = (PROJECT_ROOT / "scripts" / "smoke_distribution.py").read_text(encoding="utf-8")
    release_path = PROJECT_ROOT / "tests" / "release" / "test_smoke_distribution.py"
    release_tree = ast.parse(
        release_path.read_text(encoding="utf-8"),
        filename=str(release_path),
    )
    assert "from pixipix.stages.io import _valid_owned_output, load_stage_input" in smoke
    for assumption in PHYSICAL_LAYOUT:
        assigned_paths = {
            node.value.value
            for node in ast.walk(release_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "align_path"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        assert assigned_paths == {assumption.path}
        assert assumption.consumer == "tests/release/test_smoke_distribution.py"


def test_matrix_does_not_promote_consumed_stage_symbols_to_public_api() -> None:
    assert all(entry.classification != "public" for entry in MATRIX)


def test_extract_facade_reexports_exact_internal_objects_and_omits_old_bindings() -> None:
    facade = importlib.import_module("pixipix.stages.extract")
    analysis = importlib.import_module("pixipix.stages.extract.analysis")
    api = importlib.import_module("pixipix.stages.extract.api")
    planning = importlib.import_module("pixipix.stages.extract.planning")
    publication = importlib.import_module("pixipix.stages.extract.publication")
    owners = {
        "ComponentMap": analysis,
        "extract_source": api,
        "filter_components": analysis,
        "inspect_source": api,
        "label_components": analysis,
        "order_components": analysis,
        "project_extract_resources": planning,
        "project_extracted_frames": planning,
        "publish_extraction": publication,
    }

    for name, owner in owners.items():
        assert getattr(facade, name) is getattr(owner, name)

    for name in (
        "_Analysis",
        "_analyze",
        "_padded_bounds",
        "_materialize_frame_crop",
    ):
        assert not hasattr(facade, name), f"migrated private facade binding remains: {name}"
    for name in (
        "_valid_marker",
        "_valid_owned_output",
        "_valid_frame_png",
        "_validate_staged_output",
        "_validate_output_location",
        "_prepare_target",
        "_remove_temporary_tree",
    ):
        assert not hasattr(publication, name), f"duplicate publication owner remains: {name}"
    for name in (
        "np",
        "Image",
        "load_source",
        "generate_foreground_mask",
        "write_png",
        "write_json",
        "to_json_data",
        "enforce_resource_policy",
    ):
        assert not hasattr(facade, name), f"infrastructure facade binding remains: {name}"
    assert not hasattr(facade, "ExtractStagePlan")


def test_extract_facade_is_relative_grouped_and_definition_free() -> None:
    facade_path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract" / "__init__.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    expected = {
        "analysis": (
            "ComponentMap",
            "filter_components",
            "label_components",
            "order_components",
        ),
        "api": ("extract_source", "inspect_source"),
        "planning": ("project_extract_resources", "project_extracted_frames"),
        "publication": ("publish_extraction",),
    }

    assert {node.module: tuple(alias.name for alias in node.names) for node in imports} == expected
    assert all(node.level == 1 for node in imports)
    assert all(alias.asname == alias.name for node in imports for alias in node.names)
    assert all(isinstance(node, (ast.Expr, ast.ImportFrom)) for node in tree.body)
    expressions = [node for node in tree.body if isinstance(node, ast.Expr)]
    assert len(expressions) == 1
    assert isinstance(expressions[0].value, ast.Constant)
    assert isinstance(expressions[0].value.value, str)

    package = facade_path.parent
    owner_counts: dict[str, int] = {}
    for path in sorted(package.glob("*.py")):
        module_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in module_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                owner_counts[node.name] = owner_counts.get(node.name, 0) + 1
    assert owner_counts["ComponentMap"] == 1
    assert owner_counts["_Analysis"] == 1
    assert all(count == 1 for count in owner_counts.values())


def test_extract_package_preserves_module_and_foundational_type_identities() -> None:
    package = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract"
    extract = importlib.import_module("pixipix.stages.extract")
    analysis = importlib.import_module("pixipix.stages.extract.analysis")
    api = importlib.import_module("pixipix.stages.extract.api")
    planning = importlib.import_module("pixipix.stages.extract.planning")
    publication = importlib.import_module("pixipix.stages.extract.publication")
    extract_file = extract.__file__

    assert extract_file is not None
    assert Path(extract_file).resolve() == (package / "__init__.py").resolve()
    assert extract.__spec__ is not None
    assert extract.__spec__.submodule_search_locations is not None
    assert analysis.ComponentMap.__module__ == "pixipix.stages.extract.analysis"
    assert analysis._Analysis.__module__ == "pixipix.stages.extract.analysis"
    assert api.inspect_source.__module__ == "pixipix.stages.extract.api"
    assert api.extract_source.__module__ == "pixipix.stages.extract.api"
    assert planning.project_extract_resources.__module__ == "pixipix.stages.extract.planning"
    assert publication.publish_extraction.__module__ == "pixipix.stages.extract.publication"
    assert sys.modules["pixipix.stages.extract"] is extract
    assert "pixipix.stages.extract.__init__" not in sys.modules

    code = (
        "import pathlib, sys; "
        "import pixipix.models as models; "
        "import pixipix.resources as resources; "
        "import pixipix.stages.extract as extract; "
        "import pixipix.stages.extract.analysis as analysis; "
        "import pixipix.stages.extract.api as api; "
        "import pixipix.stages.extract.execution as execution; "
        "import pixipix.stages.extract.metadata as metadata; "
        "import pixipix.stages.extract.planning as planning; "
        "import pixipix.stages.extract.publication as publication; "
        "assert pathlib.Path(extract.__file__).name == '__init__.py'; "
        "assert extract.ComponentMap is analysis.ComponentMap; "
        "assert extract.inspect_source is api.inspect_source; "
        "assert extract.extract_source is api.extract_source; "
        "assert extract.project_extract_resources is planning.project_extract_resources; "
        "assert extract.publish_extraction is publication.publish_extraction; "
        "assert api.ExtractionRun is models.ExtractionRun; "
        "assert api.ExtractionResult is models.ExtractionResult; "
        "assert execution.ExtractedFrame is models.ExtractedFrame; "
        "assert execution.FrameImage is models.FrameImage; "
        "assert planning.ExtractedFrame is models.ExtractedFrame; "
        "assert planning.ResourceProjection is resources.ResourceProjection; "
        "assert sys.modules['pixipix.stages.extract'] is extract; "
        "assert 'pixipix.stages.extract.__init__' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_stage_plan_exports_are_the_exact_types_constructed_by_planners(
    tmp_path: Path,
) -> None:
    config = tmp_path / "pixipix.toml"
    write_config(config, alignment_config())
    loaded = load_config(config)

    extract_root = tmp_path / "extract"
    write_declared_extract_stage(extract_root, loaded, ((4, 4), (6, 4)))
    scale_plan = project_scale_stage(
        validate_stage_input(extract_root, "extract"),
        loaded,
    )
    assert type(scale_plan) is ScaleStagePlan

    scale_root = tmp_path / "scale"
    write_declared_scale_stage(
        scale_root,
        loaded,
        ((4, 4), (6, 4)),
        ((4, 4), (6, 4)),
        factor=1.0,
    )
    pixelize_plan = project_pixelize_stage(
        validate_stage_input(scale_root, "scale"),
        loaded,
    )
    assert type(pixelize_plan) is PixelizeStagePlan

    pixelize_root = tmp_path / "pixelize"
    write_declared_pixelize_stage(pixelize_root, loaded, ((2, 2), (3, 2)))
    align_plan = project_align_stage(
        validate_stage_input(pixelize_root, "pixelize"),
        loaded,
    )
    assert type(align_plan) is AlignmentStagePlan


def test_pixelize_facade_reexports_exact_internal_objects() -> None:
    package = PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize"
    pixelize = importlib.import_module("pixipix.stages.pixelize")
    api = importlib.import_module("pixipix.stages.pixelize.api")
    execution = importlib.import_module("pixipix.stages.pixelize.execution")
    metadata = importlib.import_module("pixipix.stages.pixelize.metadata")
    planning = importlib.import_module("pixipix.stages.pixelize.planning")
    scale = importlib.import_module("pixipix.stages.scale")
    geometry = importlib.import_module("pixipix.stages.scale.geometry")
    pixelize_file = pixelize.__file__

    assert pixelize_file is not None
    assert Path(pixelize_file).resolve() == (package / "__init__.py").resolve()
    assert pixelize.__spec__ is not None
    assert pixelize.__spec__.submodule_search_locations is not None
    owners = {
        "publish_pixelize": api,
        "PreparedCellGrid": execution,
        "CellGridProjection": planning,
        "PixelizeRun": execution,
        "PixelizeStagePlan": planning,
        "project_cell_grid": planning,
        "prepare_cell_grid": execution,
        "representative_pixel": execution,
        "apply_alpha_policy": execution,
        "pixelize_prepared_grid": execution,
        "project_pixelize_resources": planning,
        "project_pixelize_stage": planning,
        "pixelize_stage": execution,
        "MAX_PREPARED_PIXELS": planning,
        "round_channel_half_away_from_zero": execution,
    }
    for name, owner in owners.items():
        assert getattr(pixelize, name) is getattr(owner, name)
    for infrastructure_name in (
        "decode_stage_input",
        "np",
        "_require_pixelize_config",
        "_validate_config_handoff",
        "_majority",
        "_center",
        "_alpha_weighted_majority",
        "build_pixelize_metadata",
        "Image",
    ):
        assert not hasattr(pixelize, infrastructure_name)
    assert str(inspect.signature(metadata.build_pixelize_metadata)) == (
        "(stage: 'LoadedStageInput', loaded: 'LoadedConfig', plan: 'PixelizeStagePlan', "
        "config: 'PixelizeConfig', cell_size: 'int') -> 'PixelizeStageMetadata'"
    )
    assert str(inspect.signature(planning._validate_config_handoff)) == (
        "(stage: 'ValidatedStageInput', loaded: 'LoadedConfig', cell_size: 'int') -> 'None'"
    )
    assert (
        pixelize.round_channel_half_away_from_zero
        is execution.round_channel_half_away_from_zero
        is vars(scale)["round_channel_half_away_from_zero"]
        is vars(geometry)["round_channel_half_away_from_zero"]
    )
    assert sys.modules["pixipix.stages.pixelize"] is pixelize
    assert "pixipix.stages.pixelize.__init__" not in sys.modules

    code = (
        "import pathlib, sys; "
        "import pixipix.stages.pixelize as pixelize; "
        "import pixipix.stages.pixelize.api as api; "
        "import pixipix.stages.pixelize.execution as execution; "
        "import pixipix.stages.pixelize.metadata as metadata; "
        "import pixipix.stages.pixelize.planning as planning; "
        "import pixipix.stages.scale as scale; "
        "import pixipix.stages.scale.geometry as geometry; "
        "assert pathlib.Path(pixelize.__file__).name == '__init__.py'; "
        "assert pixelize.__spec__.submodule_search_locations is not None; "
        "assert pixelize.PixelizeStagePlan is planning.PixelizeStagePlan; "
        "assert pixelize.PixelizeRun is execution.PixelizeRun; "
        "assert pixelize.publish_pixelize is api.publish_pixelize; "
        "assert pixelize.PixelizeStagePlan.__module__ "
        "== 'pixipix.stages.pixelize.planning'; "
        "assert pixelize.PixelizeRun.__module__ "
        "== 'pixipix.stages.pixelize.execution'; "
        "assert pixelize.publish_pixelize.__module__ == 'pixipix.stages.pixelize.api'; "
        "assert pixelize.round_channel_half_away_from_zero "
        "is execution.round_channel_half_away_from_zero "
        "is scale.round_channel_half_away_from_zero "
        "is geometry.round_channel_half_away_from_zero; "
        "assert callable(metadata.build_pixelize_metadata); "
        "assert not hasattr(pixelize, 'decode_stage_input'); "
        "assert not hasattr(pixelize, 'np'); "
        "assert not hasattr(pixelize, '_require_pixelize_config'); "
        "assert not hasattr(pixelize, '_validate_config_handoff'); "
        "assert not hasattr(pixelize, 'build_pixelize_metadata'); "
        "assert sys.modules['pixipix.stages.pixelize'] is pixelize; "
        "assert 'pixipix.stages.pixelize.__init__' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pixelize_facade_is_relative_grouped_and_definition_free() -> None:
    facade_path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize" / "__init__.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
    relative_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level
    ]
    expected = {
        "api": ("publish_pixelize",),
        "execution": (
            "PixelizeRun",
            "PreparedCellGrid",
            "apply_alpha_policy",
            "pixelize_prepared_grid",
            "pixelize_stage",
            "prepare_cell_grid",
            "representative_pixel",
            "round_channel_half_away_from_zero",
        ),
        "planning": (
            "MAX_PREPARED_PIXELS",
            "CellGridProjection",
            "PixelizeStagePlan",
            "project_cell_grid",
            "project_pixelize_resources",
            "project_pixelize_stage",
        ),
    }

    assert {
        node.module: tuple(alias.name for alias in node.names) for node in relative_imports
    } == expected
    assert all(node.level == 1 for node in relative_imports)
    assert all(alias.asname == alias.name for node in relative_imports for alias in node.names)
    assert all(isinstance(node, (ast.Expr, ast.ImportFrom)) for node in tree.body)
    expressions = [node for node in tree.body if isinstance(node, ast.Expr)]
    assert len(expressions) == 1
    assert isinstance(expressions[0].value, ast.Constant)
    assert isinstance(expressions[0].value.value, str)


def test_scale_facade_reexports_exact_internal_objects() -> None:
    package = PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale"
    foundational_geometry = importlib.import_module("pixipix._scale_geometry")
    pipeline_input = importlib.import_module("pixipix.pipeline.input")
    scale = importlib.import_module("pixipix.stages.scale")
    api = importlib.import_module("pixipix.stages.scale.api")
    execution = importlib.import_module("pixipix.stages.scale.execution")
    geometry = importlib.import_module("pixipix.stages.scale.geometry")
    metadata = importlib.import_module("pixipix.stages.scale.metadata")
    planning = importlib.import_module("pixipix.stages.scale.planning")
    pixelize = importlib.import_module("pixipix.stages.pixelize")
    scale_file = scale.__file__

    assert scale_file is not None
    assert Path(scale_file).resolve() == (package / "__init__.py").resolve()
    owners = {
        "publish_scale": api,
        "ScaleRun": execution,
        "scale_stage": execution,
        "premultiplied_box_resize": execution,
        "ScaleStagePlan": planning,
        "MAX_TRANSFORMED_PIXELS": planning,
        "project_scale_stage": planning,
        "project_scale_resources": planning,
        "round_half_away_from_zero": geometry,
        "transformed_dimension": geometry,
        "round_channel_half_away_from_zero": geometry,
    }
    for name, owner in owners.items():
        assert getattr(scale, name) is getattr(owner, name)
    assert geometry.round_half_away_from_zero is foundational_geometry.round_half_away_from_zero
    assert geometry.transformed_dimension is foundational_geometry.transformed_dimension
    assert pipeline_input.transformed_dimension is foundational_geometry.transformed_dimension
    assert planning.transformed_dimension is foundational_geometry.transformed_dimension
    for infrastructure_name in (
        "decode_stage_input",
        "Image",
        "np",
        "_require_scale_config",
        "_resize_float_channel",
        "build_scale_metadata",
    ):
        assert not hasattr(scale, infrastructure_name)
    assert str(inspect.signature(metadata.build_scale_metadata)) == (
        "(stage: 'LoadedStageInput', loaded: 'LoadedConfig', plan: 'ScaleStagePlan', "
        "config: 'ScaleConfig') -> 'ScaleStageMetadata'"
    )
    assert str(inspect.signature(planning._validate_config_handoff)) == (
        "(stage: 'ValidatedStageInput', loaded: 'LoadedConfig') -> 'None'"
    )
    assert str(inspect.signature(planning._global_factor)) == (
        "(config: 'ScaleConfig', stage: 'ValidatedStageInput', "
        "source_cell_size: 'int | None') -> 'tuple[float, int | None, int | None]'"
    )
    assert (
        vars(pixelize)["round_channel_half_away_from_zero"]
        is vars(scale)["round_channel_half_away_from_zero"]
    )
    assert sys.modules["pixipix.stages.scale"] is scale
    assert "pixipix.stages.scale.__init__" not in sys.modules

    code = (
        "import sys; "
        "import pixipix.stages.pixelize as pixelize; "
        "import pixipix.stages.scale as scale; "
        "import pixipix.stages.scale.api as api; "
        "import pixipix.stages.scale.execution as execution; "
        "import pixipix.stages.scale.geometry as geometry; "
        "import pixipix.stages.scale.planning as planning; "
        "assert pixelize.round_channel_half_away_from_zero "
        "is scale.round_channel_half_away_from_zero; "
        "assert scale.ScaleRun is execution.ScaleRun; "
        "assert scale.ScaleStagePlan is planning.ScaleStagePlan; "
        "assert scale.publish_scale is api.publish_scale; "
        "assert scale.round_channel_half_away_from_zero "
        "is geometry.round_channel_half_away_from_zero; "
        "assert scale.ScaleStagePlan.__module__ == 'pixipix.stages.scale.planning'; "
        "assert scale.publish_scale.__module__ == 'pixipix.stages.scale.api'; "
        "assert scale.round_channel_half_away_from_zero.__module__ "
        "== 'pixipix.stages.scale.geometry'; "
        "assert not hasattr(scale, 'decode_stage_input'); "
        "assert not hasattr(scale, 'Image'); "
        "assert not hasattr(scale, 'np'); "
        "assert not hasattr(scale, '_require_scale_config'); "
        "assert not hasattr(scale, '_resize_float_channel'); "
        "assert not hasattr(scale, 'build_scale_metadata'); "
        "assert sys.modules['pixipix.stages.scale'] is scale; "
        "assert 'pixipix.stages.scale.__init__' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_scale_facade_is_relative_grouped_and_definition_free() -> None:
    facade_path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale" / "__init__.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
    relative_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level
    ]
    expected = {
        "api": ("publish_scale",),
        "execution": ("ScaleRun", "premultiplied_box_resize", "scale_stage"),
        "geometry": (
            "round_channel_half_away_from_zero",
            "round_half_away_from_zero",
            "transformed_dimension",
        ),
        "planning": (
            "MAX_TRANSFORMED_PIXELS",
            "ScaleStagePlan",
            "project_scale_resources",
            "project_scale_stage",
        ),
    }

    assert {
        node.module: tuple(alias.name for alias in node.names) for node in relative_imports
    } == expected
    assert all(node.level == 1 for node in relative_imports)
    assert all(alias.asname == alias.name for node in relative_imports for alias in node.names)
    assert all(isinstance(node, (ast.Expr, ast.ImportFrom)) for node in tree.body)
    assert all(
        isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for node in tree.body
        if isinstance(node, ast.Expr)
    )
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))
        for node in tree.body
    )


# M3.5 Slice 11: complete compatibility-facade and installed-manifest contract.
CompatibilityExportKind = Literal["function", "class", "integer", "singleton"]
CompatibilityAssertionRule = Literal[
    "identity",
    "value-exact-type-ast",
]
AllPosture = Literal["present-exact", "absent-by-design"]


@dataclass(frozen=True, slots=True)
class FacadeExportContract:
    name: str
    direct_owner: str
    final_owner: str
    kind: CompatibilityExportKind
    rule: CompatibilityAssertionRule


@dataclass(frozen=True, slots=True)
class FacadeSurfaceContract:
    module: str
    source: str
    package: str
    exports: tuple[FacadeExportContract, ...]
    named_non_exports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FacadeCandidate:
    module: str
    exports: tuple[tuple[str, str], ...]
    semantic_exports: tuple[ObservedFacadeExport, ...]
    renamed_imports: tuple[str, ...]
    wildcard_imports: tuple[str, ...]
    owner_module_imports: tuple[str, ...]
    unaliased_public_imports: tuple[str, ...]
    public_functions: tuple[str, ...]
    public_classes: tuple[str, ...]
    public_assignments: tuple[str, ...]
    all_present: bool
    all_value: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class CompatibilityCandidate:
    root_version_present: bool
    root_all_present: bool
    root_all_value: tuple[str, ...] | None
    root_public_leaks: tuple[str, ...]
    stages: FacadeCandidate
    facades: tuple[FacadeCandidate, ...]
    posture_claims: tuple[tuple[str, AllPosture], ...]
    aggregate_claim: tuple[int, int, int]
    underscore_claim: tuple[str, ...]
    permanent_aliases: tuple[str, ...]
    temporary_test_local: tuple[str, ...]
    removable_test_local: tuple[str, ...]
    runtime_failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedFacadeExport:
    name: str
    direct_owner: str
    final_owner: str
    kind: CompatibilityExportKind
    rule: CompatibilityAssertionRule
    assertion_result: bool


def _export(
    name: str,
    direct_owner: str,
    final_owner: str,
    kind: CompatibilityExportKind,
    rule: CompatibilityAssertionRule = "identity",
) -> FacadeExportContract:
    return FacadeExportContract(name, direct_owner, final_owner, kind, rule)


EXPECTED_FACADE_SURFACES = (
    FacadeSurfaceContract(
        "pixipix.stages.io",
        "src/pixipix/stages/io.py",
        "pixipix.stages",
        (
            _export("InputStageFrame", "pixipix.pipeline.input", "pixipix.pipeline.input", "class"),
            _export(
                "LoadedStageInput", "pixipix.pipeline.input", "pixipix.pipeline.input", "class"
            ),
            _export(
                "ValidatedStageFrame",
                "pixipix.pipeline.input",
                "pixipix.pipeline.input",
                "class",
            ),
            _export(
                "ValidatedStageInput",
                "pixipix.pipeline.input",
                "pixipix.pipeline.input",
                "class",
            ),
            _export(
                "decode_stage_input",
                "pixipix.pipeline.input",
                "pixipix.pipeline.input",
                "function",
            ),
            _export(
                "load_stage_input",
                "pixipix.pipeline.input",
                "pixipix.pipeline.input",
                "function",
            ),
            _export(
                "validate_stage_input",
                "pixipix.pipeline.input",
                "pixipix.pipeline.input",
                "function",
            ),
            _export(
                "OutputFrameImage",
                "pixipix.pipeline.publication",
                "pixipix.pipeline.publication",
                "class",
            ),
            _export(
                "_valid_owned_output",
                "pixipix.pipeline.publication",
                "pixipix.pipeline.publication",
                "function",
            ),
            _export(
                "publish_stage_output",
                "pixipix.pipeline.publication",
                "pixipix.pipeline.publication",
                "function",
            ),
            _export(
                "validate_stage_output_target",
                "pixipix.pipeline.publication",
                "pixipix.pipeline.publication",
                "function",
            ),
        ),
        ("Image", "write_json", "write_png"),
    ),
    FacadeSurfaceContract(
        "pixipix.stages.extract",
        "src/pixipix/stages/extract/__init__.py",
        "pixipix.stages.extract",
        (
            _export(
                "ComponentMap",
                "pixipix.stages.extract.analysis",
                "pixipix.stages.extract.analysis",
                "class",
            ),
            _export(
                "filter_components",
                "pixipix.stages.extract.analysis",
                "pixipix.stages.extract.analysis",
                "function",
            ),
            _export(
                "label_components",
                "pixipix.stages.extract.analysis",
                "pixipix.stages.extract.analysis",
                "function",
            ),
            _export(
                "order_components",
                "pixipix.stages.extract.analysis",
                "pixipix.stages.extract.analysis",
                "function",
            ),
            _export(
                "extract_source",
                "pixipix.stages.extract.api",
                "pixipix.stages.extract.api",
                "function",
            ),
            _export(
                "inspect_source",
                "pixipix.stages.extract.api",
                "pixipix.stages.extract.api",
                "function",
            ),
            _export(
                "project_extract_resources",
                "pixipix.stages.extract.planning",
                "pixipix.stages.extract.planning",
                "function",
            ),
            _export(
                "project_extracted_frames",
                "pixipix.stages.extract.planning",
                "pixipix.stages.extract.planning",
                "function",
            ),
            _export(
                "publish_extraction",
                "pixipix.stages.extract.publication",
                "pixipix.stages.extract.publication",
                "function",
            ),
        ),
        (
            "Image",
            "enforce_resource_policy",
            "generate_foreground_mask",
            "load_source",
            "np",
            "to_json_data",
            "write_json",
            "write_png",
        ),
    ),
    FacadeSurfaceContract(
        "pixipix.stages.scale",
        "src/pixipix/stages/scale/__init__.py",
        "pixipix.stages.scale",
        (
            _export(
                "publish_scale",
                "pixipix.stages.scale.api",
                "pixipix.stages.scale.api",
                "function",
            ),
            _export(
                "ScaleRun",
                "pixipix.stages.scale.execution",
                "pixipix.stages.scale.execution",
                "class",
            ),
            _export(
                "premultiplied_box_resize",
                "pixipix.stages.scale.execution",
                "pixipix.stages.scale.execution",
                "function",
            ),
            _export(
                "scale_stage",
                "pixipix.stages.scale.execution",
                "pixipix.stages.scale.execution",
                "function",
            ),
            _export(
                "round_channel_half_away_from_zero",
                "pixipix.stages.scale.geometry",
                "pixipix.stages.scale.geometry",
                "function",
            ),
            _export(
                "round_half_away_from_zero",
                "pixipix.stages.scale.geometry",
                "pixipix._scale_geometry",
                "function",
            ),
            _export(
                "transformed_dimension",
                "pixipix.stages.scale.geometry",
                "pixipix._scale_geometry",
                "function",
            ),
            _export(
                "MAX_TRANSFORMED_PIXELS",
                "pixipix.stages.scale.planning",
                "pixipix.stages.scale.planning",
                "integer",
                "value-exact-type-ast",
            ),
            _export(
                "ScaleStagePlan",
                "pixipix.stages.scale.planning",
                "pixipix.stages.scale.planning",
                "class",
            ),
            _export(
                "project_scale_resources",
                "pixipix.stages.scale.planning",
                "pixipix.stages.scale.planning",
                "function",
            ),
            _export(
                "project_scale_stage",
                "pixipix.stages.scale.planning",
                "pixipix.stages.scale.planning",
                "function",
            ),
        ),
        ("Image", "build_scale_metadata", "decode_stage_input", "np"),
    ),
    FacadeSurfaceContract(
        "pixipix.stages.pixelize",
        "src/pixipix/stages/pixelize/__init__.py",
        "pixipix.stages.pixelize",
        (
            _export(
                "publish_pixelize",
                "pixipix.stages.pixelize.api",
                "pixipix.stages.pixelize.api",
                "function",
            ),
            _export(
                "PixelizeRun",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "class",
            ),
            _export(
                "PreparedCellGrid",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "class",
            ),
            _export(
                "apply_alpha_policy",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "function",
            ),
            _export(
                "pixelize_prepared_grid",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "function",
            ),
            _export(
                "pixelize_stage",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "function",
            ),
            _export(
                "prepare_cell_grid",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "function",
            ),
            _export(
                "representative_pixel",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.pixelize.execution",
                "function",
            ),
            _export(
                "round_channel_half_away_from_zero",
                "pixipix.stages.pixelize.execution",
                "pixipix.stages.scale.geometry",
                "function",
            ),
            _export(
                "MAX_PREPARED_PIXELS",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "integer",
                "value-exact-type-ast",
            ),
            _export(
                "CellGridProjection",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "class",
            ),
            _export(
                "PixelizeStagePlan",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "class",
            ),
            _export(
                "project_cell_grid",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "function",
            ),
            _export(
                "project_pixelize_resources",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "function",
            ),
            _export(
                "project_pixelize_stage",
                "pixipix.stages.pixelize.planning",
                "pixipix.stages.pixelize.planning",
                "function",
            ),
        ),
        ("Image", "build_pixelize_metadata", "decode_stage_input", "np"),
    ),
    FacadeSurfaceContract(
        "pixipix.stages.align",
        "src/pixipix/stages/align/__init__.py",
        "pixipix.stages.align",
        (
            _export(
                "publish_align",
                "pixipix.stages.align.api",
                "pixipix.stages.align.api",
                "function",
            ),
            _export(
                "AlignmentRun",
                "pixipix.stages.align.execution",
                "pixipix.stages.align.execution",
                "class",
            ),
            _export(
                "align_stage",
                "pixipix.stages.align.execution",
                "pixipix.stages.align.execution",
                "function",
            ),
            _export(
                "compose_aligned_canvas",
                "pixipix.stages.align.execution",
                "pixipix.stages.align.execution",
                "function",
            ),
            _export(
                "EMPTY_RECTANGLE",
                "pixipix.stages.align.geometry",
                "pixipix.stages.align.geometry",
                "singleton",
                "identity",
            ),
            _export(
                "calculate_alignment_frame",
                "pixipix.stages.align.geometry",
                "pixipix.stages.align.geometry",
                "function",
            ),
            _export(
                "mathematical_floor_center",
                "pixipix.stages.align.geometry",
                "pixipix.stages.align.geometry",
                "function",
            ),
            _export(
                "AlignmentStagePlan",
                "pixipix.stages.align.planning",
                "pixipix.stages.align.planning",
                "class",
            ),
            _export(
                "clipping_finding",
                "pixipix.stages.align.planning",
                "pixipix.stages.align.planning",
                "function",
            ),
            _export(
                "project_align_resources",
                "pixipix.stages.align.planning",
                "pixipix.stages.align.planning",
                "function",
            ),
            _export(
                "project_align_stage",
                "pixipix.stages.align.planning",
                "pixipix.stages.align.planning",
                "function",
            ),
        ),
        (
            "decode_stage_input",
            "enforce_resource_policy",
            "publish_stage_output",
            "validate_stage_input",
        ),
    ),
)

EXPECTED_ROOT_BINDINGS = ("__version__",)
EXPECTED_ROOT_NON_EXPORTS = ("version",)
EXPECTED_STAGES_NON_EXPORTS = (
    "publish_align",
    "publish_extraction",
    "publish_pixelize",
    "publish_scale",
)
EXPECTED_POSTURES: tuple[tuple[str, AllPosture], ...] = (
    ("pixipix", "present-exact"),
    ("pixipix.stages", "absent-by-design"),
    ("pixipix.stages.align", "absent-by-design"),
    ("pixipix.stages.extract", "absent-by-design"),
    ("pixipix.stages.io", "absent-by-design"),
    ("pixipix.stages.pixelize", "absent-by-design"),
    ("pixipix.stages.scale", "absent-by-design"),
)
EXPECTED_INTENTIONAL_UNDERSCORES = ("pixipix.stages.io._valid_owned_output",)
PERMANENT_COMPATIBILITY = "permanent production compatibility"

# These bounded semantic-role and owner registries are deliberately independent
# from EXPECTED_FACADE_SURFACES. They identify where observation is permitted;
# they do not provide expected per-export owners, kinds, rules, or results.
IDENTITY_SINGLETON_SYMBOLS = frozenset({"pixipix.stages.align.EMPTY_RECTANGLE"})
IMMUTABLE_VALUE_SYMBOLS = frozenset(
    {
        "pixipix.stages.pixelize.MAX_PREPARED_PIXELS",
        "pixipix.stages.scale.MAX_TRANSFORMED_PIXELS",
    }
)
PRODUCTION_OWNER_MODULES = (
    "pixipix._scale_geometry",
    "pixipix.pipeline.input",
    "pixipix.pipeline.publication",
    "pixipix.stages.align.api",
    "pixipix.stages.align.execution",
    "pixipix.stages.align.geometry",
    "pixipix.stages.align.planning",
    "pixipix.stages.extract.analysis",
    "pixipix.stages.extract.api",
    "pixipix.stages.extract.planning",
    "pixipix.stages.extract.publication",
    "pixipix.stages.pixelize.api",
    "pixipix.stages.pixelize.execution",
    "pixipix.stages.pixelize.planning",
    "pixipix.stages.scale.api",
    "pixipix.stages.scale.execution",
    "pixipix.stages.scale.geometry",
    "pixipix.stages.scale.planning",
)


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _literal_all_value(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...] | None:
    value = node.value
    if value is None:
        return None
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError):
        return None
    if not isinstance(literal, list) or not all(isinstance(item, str) for item in literal):
        return None
    return tuple(literal)


def _is_production_module(target: str) -> bool:
    if not (target == "pixipix" or target.startswith("pixipix.")):
        return False
    try:
        return importlib.util.find_spec(target) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def _module_defines_name(module: ModuleType, name: str) -> bool:
    source_name = getattr(module, "__file__", None)
    if source_name is None:
        return False
    tree = ast.parse(Path(source_name).read_text(encoding="utf-8"), filename=source_name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and name in _assigned_names(node):
            return True
    return False


def _observed_kind(binding: str, value: object) -> CompatibilityExportKind:
    if inspect.isfunction(value):
        return "function"
    if inspect.isclass(value):
        return "class"
    if binding in IMMUTABLE_VALUE_SYMBOLS:
        if type(value) is not int:
            raise AssertionError(
                f"surface={binding.rsplit('.', 1)[0]} symbol={binding.rsplit('.', 1)[1]} "
                f"actual kind={type(value).__name__} expected kind=integer "
                "remediation=restore exact immutable integer"
            )
        return "integer"
    if binding in IDENTITY_SINGLETON_SYMBOLS:
        return "singleton"
    raise AssertionError(
        f"surface={binding.rsplit('.', 1)[0]} symbol={binding.rsplit('.', 1)[1]} "
        f"actual kind={type(value).__name__} expected kind=unambiguous observed semantic role "
        "remediation=classify the semantic role explicitly"
    )


def _observed_rule(kind: CompatibilityExportKind) -> CompatibilityAssertionRule:
    if kind in {"function", "class", "singleton"}:
        return "identity"
    if kind == "integer":
        return "value-exact-type-ast"
    raise AssertionError(f"unhandled observed compatibility kind: {kind}")


def _observed_final_owner(name: str, kind: CompatibilityExportKind, value: object) -> str:
    matches: list[str] = []
    for module_name in PRODUCTION_OWNER_MODULES:
        module = importlib.import_module(module_name)
        if not _module_defines_name(module, name) or not hasattr(module, name):
            continue
        owner_value = getattr(module, name)
        if kind == "integer":
            matches.extend(
                [module_name]
                if type(owner_value) is int and type(value) is int and owner_value == value
                else []
            )
        elif owner_value is value:
            matches.append(module_name)
    if len(matches) != 1:
        raise AssertionError(
            f"symbol={name} actual final-owner matches={matches} expected matches=one "
            "remediation=restore one canonical definition owner"
        )
    return matches[0]


def _observe_export(
    facade_module: str,
    name: str,
    direct_owner: str,
) -> ObservedFacadeExport:
    facade = importlib.import_module(facade_module)
    direct = importlib.import_module(direct_owner)
    facade_value = getattr(facade, name)
    direct_value = getattr(direct, name)
    binding = f"{facade_module}.{name}"
    kind = _observed_kind(binding, facade_value)
    rule = _observed_rule(kind)
    final_owner = _observed_final_owner(name, kind, facade_value)
    final_value = getattr(importlib.import_module(final_owner), name)
    assertion_result = facade_value is direct_value and (
        facade_value is final_value
        if kind != "integer"
        else type(facade_value) is int and type(final_value) is int and facade_value == final_value
    )
    return ObservedFacadeExport(
        name=name,
        direct_owner=direct_owner,
        final_owner=final_owner,
        kind=kind,
        rule=rule,
        assertion_result=assertion_result,
    )


def _scan_facade_source(
    module: str,
    source: Path,
    package: str,
) -> FacadeCandidate:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    exports: list[tuple[str, str]] = []
    renamed: list[str] = []
    wildcards: list[str] = []
    owner_modules: list[str] = []
    unaliased: list[str] = []
    public_functions: list[str] = []
    public_classes: list[str] = []
    public_assignments: list[str] = []
    all_present = False
    all_value: tuple[str, ...] | None = None

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            relative_name = "." * node.level + (node.module or "")
            owner = (
                importlib.util.resolve_name(relative_name, package) if node.level else node.module
            )
            assert owner is not None
            for alias in node.names:
                bound = alias.asname or alias.name
                rendered = f"{owner}.{alias.name} as {bound}"
                absolute_target = f"{owner}.{alias.name}"
                if alias.name == "*":
                    wildcards.append(owner)
                elif not bound.startswith("_") and _is_production_module(absolute_target):
                    owner_modules.append(rendered)
                elif alias.asname == alias.name:
                    exports.append((alias.name, owner))
                elif alias.asname is not None:
                    renamed.append(rendered)
                elif not bound.startswith("_"):
                    unaliased.append(rendered)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if not bound.startswith("_"):
                    owner_modules.append(f"{alias.name} as {bound}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public_functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                public_classes.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            if "__all__" in names:
                all_present = True
                all_value = _literal_all_value(node)
            public_assignments.extend(
                name for name in names if not name.startswith("_") and name != "__all__"
            )

    return FacadeCandidate(
        module=module,
        exports=tuple(sorted(exports)),
        semantic_exports=(),
        renamed_imports=tuple(sorted(renamed)),
        wildcard_imports=tuple(sorted(wildcards)),
        owner_module_imports=tuple(sorted(owner_modules)),
        unaliased_public_imports=tuple(sorted(unaliased)),
        public_functions=tuple(sorted(public_functions)),
        public_classes=tuple(sorted(public_classes)),
        public_assignments=tuple(sorted(public_assignments)),
        all_present=all_present,
        all_value=all_value,
    )


def _root_contract_state() -> tuple[bool, bool, tuple[str, ...] | None, tuple[str, ...]]:
    source = PROJECT_ROOT / "src" / "pixipix" / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    version_present = False
    all_present = False
    all_value: tuple[str, ...] | None = None
    public_leaks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                allowed_metadata_helper = (
                    node.level == 0
                    and node.module == "importlib.metadata"
                    and alias.name == "version"
                    and bound == "version"
                )
                if not allowed_metadata_helper and not bound.startswith("_"):
                    public_leaks.append(f"import {node.module}.{alias.name} as {bound}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if not bound.startswith("_"):
                    public_leaks.append(f"import {alias.name} as {bound}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_leaks.append(f"definition {node.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            version_present |= "__version__" in names
            if "__all__" in names:
                all_present = True
                all_value = _literal_all_value(node)
            public_leaks.extend(f"assignment {name}" for name in names if not name.startswith("_"))
    return version_present, all_present, all_value, tuple(sorted(public_leaks))


def _observed_postures(
    root_all_present: bool,
    stages: FacadeCandidate,
    facades: tuple[FacadeCandidate, ...],
) -> tuple[tuple[str, AllPosture], ...]:
    entries: list[tuple[str, AllPosture]] = [
        ("pixipix", "present-exact" if root_all_present else "absent-by-design"),
        (
            "pixipix.stages",
            "present-exact" if stages.all_present else "absent-by-design",
        ),
    ]
    entries.extend(
        (
            facade.module,
            "present-exact" if facade.all_present else "absent-by-design",
        )
        for facade in facades
    )
    return tuple(sorted(entries))


def derive_compatibility_candidate() -> CompatibilityCandidate:
    root_version, root_all_present, root_all_value, root_public_leaks = _root_contract_state()
    stages = _scan_facade_source(
        "pixipix.stages",
        PROJECT_ROOT / "src" / "pixipix" / "stages" / "__init__.py",
        "pixipix.stages",
    )
    scanned_facades = tuple(
        _scan_facade_source(
            surface.module,
            PROJECT_ROOT / surface.source,
            surface.package,
        )
        for surface in EXPECTED_FACADE_SURFACES
    )
    facades = tuple(
        replace(
            facade,
            semantic_exports=tuple(
                _observe_export(facade.module, name, direct_owner)
                for name, direct_owner in facade.exports
            ),
        )
        for facade in scanned_facades
    )
    aliases = tuple(
        sorted(
            f"{facade.module}.{name}"
            for facade in facades
            for name, _direct_owner in facade.exports
        )
    )
    deliberate = sum(len(facade.exports) for facade in facades)
    public = sum(not name.startswith("_") for facade in facades for name, _owner in facade.exports)
    underscores = tuple(
        sorted(
            f"{facade.module}.{name}"
            for facade in facades
            for name, _owner in facade.exports
            if name.startswith("_")
        )
    )
    return CompatibilityCandidate(
        root_version_present=root_version,
        root_all_present=root_all_present,
        root_all_value=root_all_value,
        root_public_leaks=root_public_leaks,
        stages=stages,
        facades=facades,
        posture_claims=_observed_postures(root_all_present, stages, facades),
        aggregate_claim=(deliberate, public, len(underscores)),
        underscore_claim=underscores,
        permanent_aliases=aliases,
        temporary_test_local=(),
        removable_test_local=(),
    )


def _expected_aliases(
    surfaces: tuple[FacadeSurfaceContract, ...] | None = None,
) -> tuple[str, ...]:
    expected_surfaces = EXPECTED_FACADE_SURFACES if surfaces is None else surfaces
    return tuple(
        sorted(
            f"{surface.module}.{export.name}"
            for surface in expected_surfaces
            for export in surface.exports
        )
    )


def _runtime_export_errors(candidate: CompatibilityCandidate) -> list[str]:
    errors: list[str] = []
    for facade in candidate.facades:
        for export in facade.semantic_exports:
            if not export.assertion_result:
                errors.append(
                    f"surface={facade.module} symbol={export.name} actual assertion=false "
                    "expected assertion=true remediation=restore canonical direct/final identity"
                )
    root = importlib.import_module("pixipix")
    root_version = getattr(root, "__version__", None)
    if type(root_version) is not str or root_version != distribution_version("pixipix"):
        errors.append(
            "surface=pixipix symbol=__version__ actual value/type differs "
            "expected value=distribution metadata str remediation=restore root binding"
        )
    return errors


def validate_compatibility_candidate(
    candidate: CompatibilityCandidate,
    *,
    expected_surfaces: tuple[FacadeSurfaceContract, ...] | None = None,
    expected_postures: tuple[tuple[str, AllPosture], ...] | None = None,
) -> None:
    errors: list[str] = []
    expected_contract = EXPECTED_FACADE_SURFACES if expected_surfaces is None else expected_surfaces
    expected_posture_contract = (
        EXPECTED_POSTURES if expected_postures is None else expected_postures
    )
    expected_by_module = {surface.module: surface for surface in expected_contract}
    candidate_surfaces = {surface.module: surface for surface in candidate.facades}
    if frozenset(candidate_surfaces) != frozenset(expected_by_module):
        errors.append(
            "surface=facades symbol=set actual form="
            f"{sorted(candidate_surfaces)} expected form={sorted(expected_by_module)} "
            "remediation=restore the five audited facades"
        )
    for module in sorted(frozenset(candidate_surfaces) | frozenset(expected_by_module)):
        actual = candidate_surfaces.get(module)
        expected = expected_by_module.get(module)
        if actual is None or expected is None:
            continue
        expected_exports = tuple(
            sorted((export.name, export.direct_owner) for export in expected.exports)
        )
        if actual.exports != expected_exports:
            errors.append(
                f"surface={module} symbol=exports actual form={actual.exports} "
                f"expected form={expected_exports} remediation=restore exact same-name imports"
            )
        expected_semantics = {export.name: export for export in expected.exports}
        observed_semantics = {export.name: export for export in actual.semantic_exports}
        if frozenset(observed_semantics) != frozenset(expected_semantics):
            errors.append(
                f"surface={module} symbol=semantic-set actual form={sorted(observed_semantics)} "
                f"expected form={sorted(expected_semantics)} "
                "remediation=derive semantics for every observed export"
            )
        for name in sorted(frozenset(observed_semantics) & frozenset(expected_semantics)):
            observed = observed_semantics[name]
            expected_export = expected_semantics[name]
            for field in ("direct_owner", "final_owner", "kind", "rule"):
                observed_value = getattr(observed, field)
                expected_value = getattr(expected_export, field)
                if observed_value != expected_value:
                    errors.append(
                        f"surface={module} symbol={name} field={field} "
                        f"actual={observed_value} expected={expected_value} "
                        "remediation=restore independently observed compatibility semantics"
                    )
        forbidden_forms = (
            ("renamed import", actual.renamed_imports),
            ("wildcard import", actual.wildcard_imports),
            ("owner-module import", actual.owner_module_imports),
            ("unaliased public import", actual.unaliased_public_imports),
            ("public function", actual.public_functions),
            ("public class", actual.public_classes),
            ("public assignment", actual.public_assignments),
        )
        for form, values in forbidden_forms:
            if values:
                errors.append(
                    f"surface={module} symbol={values[0]} actual form={form} "
                    "expected form=explicit same-name re-export "
                    "remediation=remove the facade-source leak"
                )
        if actual.all_present:
            errors.append(
                f"surface={module} symbol=__all__ actual form=present "
                "expected form=absent-by-design remediation=remove stage-facade __all__"
            )
        export_names = {name for name, _owner in actual.exports}
        for name in expected.named_non_exports:
            if name in export_names:
                errors.append(
                    f"surface={module} symbol={name} actual classification=deliberate export "
                    "expected classification=bounded non-export remediation=remove explicit export"
                )

    if candidate.stages.exports:
        errors.append(
            "surface=pixipix.stages symbol=exports actual form="
            f"{candidate.stages.exports} expected form=empty "
            "remediation=remove explicit namespace exports"
        )
    if candidate.stages.all_present:
        errors.append(
            "surface=pixipix.stages symbol=__all__ actual form=present "
            "expected form=absent-by-design remediation=remove namespace __all__"
        )
    if not candidate.root_version_present:
        errors.append(
            "surface=pixipix symbol=__version__ actual form=missing "
            "expected form=documented root binding remediation=restore root version"
        )
    if candidate.root_public_leaks:
        errors.append(
            "surface=pixipix symbol=public-source-binding actual form="
            f"{candidate.root_public_leaks} expected form=metadata helper plus documented root "
            "binding remediation=remove undocumented root binding"
        )
    if not candidate.root_all_present or candidate.root_all_value != ("__version__",):
        errors.append(
            "surface=pixipix symbol=__all__ actual form="
            f"{candidate.root_all_value if candidate.root_all_present else 'missing'} "
            "expected form=['__version__'] remediation=restore exact root __all__"
        )
    if candidate.posture_claims != expected_posture_contract:
        errors.append(
            "surface=all symbol=__all__ actual classification="
            f"{candidate.posture_claims} expected classification={expected_posture_contract} "
            "remediation=restore dual posture"
        )
    if candidate.aggregate_claim != (57, 56, 1):
        errors.append(
            "surface=facades symbol=aggregate actual form="
            f"{candidate.aggregate_claim} expected form=(57, 56, 1) "
            "remediation=restore locked facade census"
        )
    if candidate.underscore_claim != EXPECTED_INTENTIONAL_UNDERSCORES:
        errors.append(
            "surface=facades symbol=underscore actual form="
            f"{candidate.underscore_claim} expected form={EXPECTED_INTENTIONAL_UNDERSCORES} "
            "remediation=restore intentional underscore export"
        )
    expected_aliases = _expected_aliases(expected_contract)
    if candidate.permanent_aliases != expected_aliases:
        errors.append(
            "surface=facades symbol=classifications actual classification="
            f"{candidate.permanent_aliases} expected classification={expected_aliases} "
            "remediation=classify every production alias as permanent"
        )
    if candidate.temporary_test_local:
        errors.append(
            "surface=facades symbol=temporary actual classification="
            f"{candidate.temporary_test_local} expected classification=empty "
            "remediation=remove unexpected temporary scaffolding"
        )
    if candidate.removable_test_local:
        errors.append(
            "surface=facades symbol=removable actual classification="
            f"{candidate.removable_test_local} expected classification=empty "
            "remediation=production aliases are not removable"
        )
    for failure in candidate.runtime_failures:
        errors.append(
            f"surface=facades symbol={failure} actual form=runtime replacement "
            "expected form=canonical identity remediation=restore direct binding"
        )
    observed_singletons = frozenset(
        f"{facade.module}.{export.name}"
        for facade in candidate.facades
        for export in facade.semantic_exports
        if export.kind == "singleton"
    )
    observed_immutables = frozenset(
        f"{facade.module}.{export.name}"
        for facade in candidate.facades
        for export in facade.semantic_exports
        if export.kind == "integer"
    )
    if observed_singletons != IDENTITY_SINGLETON_SYMBOLS:
        errors.append(
            f"surface=facades symbol=singleton-registry actual={sorted(observed_singletons)} "
            f"expected={sorted(IDENTITY_SINGLETON_SYMBOLS)} "
            "remediation=restore exact singleton coverage"
        )
    if observed_immutables != IMMUTABLE_VALUE_SYMBOLS:
        errors.append(
            f"surface=facades symbol=immutable-registry actual={sorted(observed_immutables)} "
            f"expected={sorted(IMMUTABLE_VALUE_SYMBOLS)} "
            "remediation=restore exact immutable coverage"
        )
    errors.extend(_runtime_export_errors(candidate))
    if errors:
        raise AssertionError("\n".join(sorted(errors)))


def _manifest_surface(
    surface: FacadeSurfaceContract,
    candidate: FacadeCandidate,
) -> dict[str, object]:
    exports = [
        {
            "assertion_rule": export.rule,
            "canonical_direct_owner": export.direct_owner,
            "canonical_final_owner": export.final_owner,
            "compatibility_classification": PERMANENT_COMPATIBILITY,
            "name": export.name,
            "process_local_assertion": export.assertion_result,
            "symbol_kind": export.kind,
        }
        for export in sorted(candidate.semantic_exports, key=lambda item: item.name)
    ]
    return {
        "__all__": {"posture": "absent-by-design", "value": None},
        "bounded_named_non_exports": {
            name: name not in {export_name for export_name, _owner in candidate.exports}
            for name in sorted(surface.named_non_exports)
        },
        "documented_root_bindings": [],
        "explicit_same_name_exports": exports,
        "intentional_underscore_exports": [
            export.name
            for export in sorted(candidate.semantic_exports, key=lambda item: item.name)
            if export.name.startswith("_")
        ],
        "surface": surface.module,
    }


def build_checkout_compatibility_manifest() -> dict[str, object]:
    candidate = derive_compatibility_candidate()
    validate_compatibility_candidate(candidate)
    candidate_by_module = {surface.module: surface for surface in candidate.facades}
    root = importlib.import_module("pixipix")
    root_assertion = type(root.__version__) is str and root.__version__ == distribution_version(
        "pixipix"
    )
    root_kind = "string" if type(root.__version__) is str else type(root.__version__).__name__
    root_rule = "value-exact-type-metadata" if root_kind == "string" else "unclassified"
    surfaces: list[dict[str, object]] = [
        {
            "__all__": {"posture": "present-exact", "value": ["__version__"]},
            "bounded_named_non_exports": {"version": True},
            "documented_root_bindings": [
                {
                    "assertion_rule": root_rule,
                    "canonical_direct_owner": "distribution metadata",
                    "canonical_final_owner": "pixipix",
                    "compatibility_classification": PERMANENT_COMPATIBILITY,
                    "name": "__version__",
                    "process_local_assertion": root_assertion,
                    "symbol_kind": root_kind,
                }
            ],
            "explicit_same_name_exports": [],
            "intentional_underscore_exports": [],
            "surface": "pixipix",
        },
        {
            "__all__": {"posture": "absent-by-design", "value": None},
            "bounded_named_non_exports": {
                name: True for name in sorted(EXPECTED_STAGES_NON_EXPORTS)
            },
            "documented_root_bindings": [],
            "explicit_same_name_exports": [],
            "intentional_underscore_exports": [],
            "surface": "pixipix.stages",
        },
    ]
    surfaces.extend(
        _manifest_surface(surface, candidate_by_module[surface.module])
        for surface in sorted(EXPECTED_FACADE_SURFACES, key=lambda item: item.module)
    )
    return {"surfaces": surfaces}


def compatibility_contract_payload(
    expected_surfaces: tuple[FacadeSurfaceContract, ...] | None = None,
    expected_postures: tuple[tuple[str, AllPosture], ...] | None = None,
) -> dict[str, object]:
    surfaces = EXPECTED_FACADE_SURFACES if expected_surfaces is None else expected_surfaces
    postures = EXPECTED_POSTURES if expected_postures is None else expected_postures
    return {
        "facades": [
            {
                "module": surface.module,
                "named_non_exports": list(sorted(surface.named_non_exports)),
                "package": surface.package,
                "exports": [
                    {
                        "direct_owner": export.direct_owner,
                        "final_owner": export.final_owner,
                        "kind": export.kind,
                        "name": export.name,
                        "rule": export.rule,
                    }
                    for export in sorted(surface.exports, key=lambda item: item.name)
                ],
            }
            for surface in sorted(surfaces, key=lambda item: item.module)
        ],
        "expected_postures": [list(entry) for entry in postures],
        "identity_singletons": sorted(IDENTITY_SINGLETON_SYMBOLS),
        "immutable_values": sorted(IMMUTABLE_VALUE_SYMBOLS),
        "owner_modules": list(PRODUCTION_OWNER_MODULES),
        "scanner_cases": [
            {
                "exports": [],
                "owner_modules": [["analysis", "pixipix.stages.extract"]],
                "package": "pixipix.stages.extract",
                "source": "from . import analysis as analysis\n",
            },
            {
                "exports": [],
                "owner_modules": [["analysis", "pixipix.stages.extract"]],
                "package": "pixipix.stages.extract",
                "source": ("from pixipix.stages.extract import analysis as analysis\n"),
            },
            {
                "exports": [["ComponentMap", "pixipix.stages.extract.analysis"]],
                "owner_modules": [],
                "package": "pixipix.stages.extract",
                "source": "from .analysis import ComponentMap as ComponentMap\n",
            },
            {
                "exports": [["ComponentMap", "pixipix.stages.extract.analysis"]],
                "owner_modules": [],
                "package": "pixipix.stages.extract",
                "source": (
                    "from pixipix.stages.extract.analysis import (\n"
                    "    ComponentMap as ComponentMap,\n"
                    ")\n"
                ),
            },
            {
                "exports": [["ComponentMap", "pixipix.stages.extract.analysis"]],
                "owner_modules": [],
                "package": "pixipix.stages.pixelize",
                "source": "from ..extract.analysis import ComponentMap as ComponentMap\n",
            },
        ],
        "stages_non_exports": list(sorted(EXPECTED_STAGES_NON_EXPORTS)),
    }


def installed_compatibility_manifest_program(
    contract_payload: Mapping[str, object] | None = None,
) -> str:
    payload = json.dumps(
        compatibility_contract_payload() if contract_payload is None else contract_payload,
        sort_keys=True,
    )
    return f"""
import ast
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import pathlib
import sys

contract = json.loads({payload!r})

def module_source(module):
    loaded = importlib.import_module(module)
    source = pathlib.Path(loaded.__file__).resolve()
    return loaded, ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

def is_production_module(target):
    if not (target == "pixipix" or target.startswith("pixipix.")):
        return False
    try:
        return importlib.util.find_spec(target) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False

def classified_imports(tree, package):
    result = []
    owner_modules = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        relative = "." * node.level + (node.module or "")
        owner = importlib.util.resolve_name(relative, package) if node.level else node.module
        for alias in node.names:
            bound = alias.asname or alias.name
            target = owner + "." + alias.name
            if not bound.startswith("_") and is_production_module(target):
                owner_modules.append((alias.name, owner))
            elif alias.asname == alias.name:
                result.append((alias.name, owner))
    return sorted(result), sorted(owner_modules)

def same_name_exports(tree, package):
    return classified_imports(tree, package)[0]

def root_public_leaks(tree):
    result = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                allowed = (
                    node.level == 0
                    and node.module == "importlib.metadata"
                    and alias.name == "version"
                    and bound == "version"
                )
                if not allowed and not bound.startswith("_"):
                    result.append(bound)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if not bound.startswith("_"):
                    result.append(bound)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                result.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            result.extend(
                target.id
                for target in targets
                if isinstance(target, ast.Name) and not target.id.startswith("_")
            )
    return sorted(result)

def assigned_names(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]

def module_defines_name(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and name in assigned_names(node):
            return True
    return False

def observed_kind(binding, value):
    if inspect.isfunction(value):
        return "function"
    if inspect.isclass(value):
        return "class"
    if binding in contract["immutable_values"]:
        assert type(value) is int, (
            "surface=" + binding.rsplit(".", 1)[0] + " symbol=" + binding.rsplit(".", 1)[1]
            + " actual kind=" + type(value).__name__ + " expected kind=integer"
        )
        return "integer"
    if binding in contract["identity_singletons"]:
        return "singleton"
    raise AssertionError(
        "surface=" + binding.rsplit(".", 1)[0] + " symbol=" + binding.rsplit(".", 1)[1]
        + " actual kind=" + type(value).__name__
        + " expected kind=unambiguous observed semantic role"
    )

def observed_rule(kind):
    if kind in {{"function", "class", "singleton"}}:
        return "identity"
    if kind == "integer":
        return "value-exact-type-ast"
    raise AssertionError("unhandled observed kind: " + kind)

def observed_final_owner(name, kind, value):
    matches = []
    for owner_name in contract["owner_modules"]:
        owner, owner_tree = module_source(owner_name)
        if not module_defines_name(owner_tree, name) or not hasattr(owner, name):
            continue
        owner_value = getattr(owner, name)
        if kind == "integer":
            if type(owner_value) is int and type(value) is int and owner_value == value:
                matches.append(owner_name)
        elif owner_value is value:
            matches.append(owner_name)
    assert len(matches) == 1, (
        "symbol=" + name + " actual final-owner matches=" + str(matches)
        + " expected matches=one remediation=restore one canonical definition owner"
    )
    return matches[0]

def require_equal(surface, symbol, field, actual, expected):
    assert actual == expected, (
        "surface=" + surface + " symbol=" + symbol + " field=" + field
        + " actual=" + str(actual) + " expected=" + str(expected)
        + " remediation=restore independently observed compatibility semantics"
    )

for scanner_case in contract["scanner_cases"]:
    scanner_tree = ast.parse(scanner_case["source"])
    scanner_exports, scanner_owner_modules = classified_imports(
        scanner_tree, scanner_case["package"]
    )
    require_equal(
        "installed-scanner",
        "synthetic-import",
        "exports",
        scanner_exports,
        [tuple(entry) for entry in scanner_case["exports"]],
    )
    require_equal(
        "installed-scanner",
        "synthetic-import",
        "owner_modules",
        scanner_owner_modules,
        [tuple(entry) for entry in scanner_case["owner_modules"]],
    )

root, root_tree = module_source("pixipix")
stages, stages_tree = module_source("pixipix.stages")
root_assertion = (
    type(root.__version__) is str
    and root.__version__ == importlib.metadata.version("pixipix")
)
root_kind = "string" if type(root.__version__) is str else type(root.__version__).__name__
root_rule = "value-exact-type-metadata" if root_kind == "string" else "unclassified"
assert root_assertion
assert root.__all__ == ["__version__"]
assert root_public_leaks(root_tree) == []
assert not any(
    isinstance(node, (ast.Assign, ast.AnnAssign))
    and any(isinstance(target, ast.Name) and target.id == "__all__"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
    for node in stages_tree.body
)
assert same_name_exports(stages_tree, "pixipix.stages") == []
assert classified_imports(stages_tree, "pixipix.stages")[1] == []
observed_postures = [
    ["pixipix", "present-exact"],
    ["pixipix.stages", "absent-by-design"],
]
observed_aliases = []
observed_singletons = []
observed_immutables = []

surfaces = [
    {{
        "__all__": {{"posture": "present-exact", "value": ["__version__"]}},
        "bounded_named_non_exports": {{"version": True}},
        "documented_root_bindings": [{{
            "assertion_rule": root_rule,
            "canonical_direct_owner": "distribution metadata",
            "canonical_final_owner": "pixipix",
            "compatibility_classification": {PERMANENT_COMPATIBILITY!r},
            "name": "__version__",
            "process_local_assertion": root_assertion,
            "symbol_kind": root_kind,
        }}],
        "explicit_same_name_exports": [],
        "intentional_underscore_exports": [],
        "surface": "pixipix",
    }},
    {{
        "__all__": {{"posture": "absent-by-design", "value": None}},
        "bounded_named_non_exports": {{
            name: True for name in contract["stages_non_exports"]
        }},
        "documented_root_bindings": [],
        "explicit_same_name_exports": [],
        "intentional_underscore_exports": [],
        "surface": "pixipix.stages",
    }},
]

for surface in contract["facades"]:
    facade, tree = module_source(surface["module"])
    actual_imports, owner_modules = classified_imports(tree, surface["package"])
    assert not owner_modules, (
        "surface=" + surface["module"] + " symbol=" + str(owner_modules[0])
        + " actual form=owner-module import expected form=symbol re-export "
        + "remediation=remove explicit owner-module exposure"
    )
    expected_imports = sorted(
        (entry["name"], entry["direct_owner"]) for entry in surface["exports"]
    )
    assert actual_imports == expected_imports
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "__all__"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
        for node in tree.body
    )
    observed_postures.append([surface["module"], "absent-by-design"])
    rows = []
    expected_by_name = {{entry["name"]: entry for entry in surface["exports"]}}
    for name, direct_name in actual_imports:
        entry = expected_by_name[name]
        direct = importlib.import_module(direct_name)
        facade_value = getattr(facade, name)
        direct_value = getattr(direct, name)
        kind = observed_kind(surface["module"] + "." + name, facade_value)
        rule = observed_rule(kind)
        binding = surface["module"] + "." + name
        observed_aliases.append(binding)
        if kind == "singleton":
            observed_singletons.append(binding)
        elif kind == "integer":
            observed_immutables.append(binding)
        final_owner = observed_final_owner(name, kind, facade_value)
        final = importlib.import_module(final_owner)
        final_value = getattr(final, name)
        assertion_result = facade_value is direct_value and (
            facade_value is final_value
            if kind != "integer"
            else type(facade_value) is int
            and type(final_value) is int
            and facade_value == final_value
        )
        require_equal(surface["module"], name, "direct_owner", direct_name, entry["direct_owner"])
        require_equal(surface["module"], name, "final_owner", final_owner, entry["final_owner"])
        require_equal(surface["module"], name, "kind", kind, entry["kind"])
        require_equal(surface["module"], name, "rule", rule, entry["rule"])
        require_equal(surface["module"], name, "assertion", assertion_result, True)
        rows.append({{
            "assertion_rule": rule,
            "canonical_direct_owner": direct_name,
            "canonical_final_owner": final_owner,
            "compatibility_classification": {PERMANENT_COMPATIBILITY!r},
            "name": name,
            "process_local_assertion": assertion_result,
            "symbol_kind": kind,
        }})
    export_names = {{name for name, _direct_owner in actual_imports}}
    surfaces.append({{
        "__all__": {{"posture": "absent-by-design", "value": None}},
        "bounded_named_non_exports": {{
            name: name not in export_names for name in surface["named_non_exports"]
        }},
        "documented_root_bindings": [],
        "explicit_same_name_exports": rows,
        "intentional_underscore_exports": sorted(
            name for name in export_names if name.startswith("_")
        ),
        "surface": surface["module"],
    }})

require_equal(
    "all",
    "__all__",
    "posture",
    sorted(observed_postures),
    sorted(contract["expected_postures"]),
)
expected_aliases = sorted(
    surface["module"] + "." + entry["name"]
    for surface in contract["facades"]
    for entry in surface["exports"]
)
require_equal("facades", "aliases", "permanence", sorted(observed_aliases), expected_aliases)
require_equal(
    "facades",
    "singletons",
    "semantic-role registry",
    sorted(observed_singletons),
    sorted(contract["identity_singletons"]),
)
require_equal(
    "facades",
    "immutables",
    "semantic-role registry",
    sorted(observed_immutables),
    sorted(contract["immutable_values"]),
)

loaded_module_paths = {{}}
modules_without_files = []
for module_name, loaded_module in sorted(sys.modules.items()):
    if module_name != "pixipix" and not module_name.startswith("pixipix."):
        continue
    module_file = getattr(loaded_module, "__file__", None)
    if module_file is None:
        modules_without_files.append(module_name)
    else:
        loaded_module_paths[module_name] = str(pathlib.Path(module_file).resolve())

print(json.dumps(
    {{"manifest": {{"surfaces": sorted(surfaces, key=lambda row: row["surface"])}},
      "module_paths": loaded_module_paths,
      "modules_without_files": modules_without_files}},
    sort_keys=True,
))
""".strip()


def _replace_facade(
    candidate: CompatibilityCandidate,
    module: str,
    replacement: FacadeCandidate,
) -> CompatibilityCandidate:
    return replace(
        candidate,
        facades=tuple(
            replacement if facade.module == module else facade for facade in candidate.facades
        ),
    )


def _mutated_compatibility_candidate(mutation: str) -> CompatibilityCandidate:
    candidate = copy.deepcopy(derive_compatibility_candidate())
    first = next(facade for facade in candidate.facades if facade.module == "pixipix.stages.io")
    if mutation == "omit-export":
        return _replace_facade(candidate, first.module, replace(first, exports=first.exports[1:]))
    if mutation == "add-public-export":
        return _replace_facade(
            candidate,
            first.module,
            replace(
                first,
                exports=tuple(
                    sorted((*first.exports, ("unexpected_export", "pixipix.pipeline.input")))
                ),
            ),
        )
    if mutation == "owner-substitution":
        name, _owner = first.exports[0]
        return _replace_facade(
            candidate,
            first.module,
            replace(
                first,
                exports=((name, "pixipix.pipeline.artifacts"), *first.exports[1:]),
            ),
        )
    if mutation == "wrapper-replacement":
        return replace(candidate, runtime_failures=("pixipix.stages.io.InputStageFrame",))
    if mutation == "wildcard-import":
        return _replace_facade(
            candidate,
            first.module,
            replace(first, wildcard_imports=("pixipix.pipeline.input",)),
        )
    if mutation == "owner-module-exposure":
        return _replace_facade(
            candidate,
            first.module,
            replace(
                first,
                owner_module_imports=("pixipix.pipeline.input as pipeline_input",),
            ),
        )
    if mutation == "renamed-public-alias":
        return _replace_facade(
            candidate,
            first.module,
            replace(
                first,
                renamed_imports=("pixipix.pipeline.input.decode_stage_input as decode",),
            ),
        )
    if mutation == "public-body":
        return _replace_facade(
            candidate,
            first.module,
            replace(first, public_functions=("compatibility_wrapper",)),
        )
    if mutation == "root-all-removed":
        return replace(candidate, root_all_present=False, root_all_value=None)
    if mutation == "root-all-corrupted":
        return replace(candidate, root_all_value=("__version__", "version"))
    if mutation == "stage-all-introduced":
        return replace(
            candidate,
            stages=replace(candidate.stages, all_present=True, all_value=()),
        )
    if mutation == "root-posture-confused":
        postures = tuple(
            (surface, "absent-by-design" if surface == "pixipix" else posture)
            for surface, posture in candidate.posture_claims
        )
        return replace(candidate, posture_claims=postures)
    if mutation == "stage-posture-confused":
        postures = tuple(
            (
                surface,
                "present-exact" if surface == "pixipix.stages.io" else posture,
            )
            for surface, posture in candidate.posture_claims
        )
        return replace(candidate, posture_claims=postures)
    if mutation == "aggregate-lower":
        return replace(candidate, aggregate_claim=(56, 55, 1))
    if mutation == "aggregate-higher":
        return replace(candidate, aggregate_claim=(58, 57, 1))
    if mutation == "underscore-omitted":
        return replace(candidate, underscore_claim=())
    if mutation == "underscore-added":
        return replace(
            candidate,
            underscore_claim=(*EXPECTED_INTENTIONAL_UNDERSCORES, "pixipix.stages.io._extra"),
        )
    if mutation == "underscore-substituted":
        return replace(candidate, underscore_claim=("pixipix.stages.io._replacement",))
    if mutation == "production-removable":
        return replace(
            candidate,
            removable_test_local=("pixipix.stages.io.InputStageFrame",),
        )
    raise AssertionError(f"unknown mutation: {mutation}")


def test_complete_compatibility_candidate_matches_independent_contract() -> None:
    candidate = derive_compatibility_candidate()

    validate_compatibility_candidate(candidate)

    manifest = build_checkout_compatibility_manifest()
    surfaces = manifest["surfaces"]
    assert isinstance(surfaces, list)
    export_rows = sum(
        len(surface["explicit_same_name_exports"])
        for surface in surfaces
        if isinstance(surface, dict)
    )
    assert export_rows == 57
    assert candidate.aggregate_claim == (57, 56, 1)
    assert candidate.underscore_claim == EXPECTED_INTENTIONAL_UNDERSCORES
    assert len(candidate.permanent_aliases) == 57
    assert candidate.temporary_test_local == ()
    assert candidate.removable_test_local == ()


def test_expected_compatibility_contract_is_independent_of_candidate() -> None:
    expected_before = EXPECTED_FACADE_SURFACES
    candidate = derive_compatibility_candidate()
    mutated = replace(candidate, permanent_aliases=())

    with pytest.raises(AssertionError):
        validate_compatibility_candidate(mutated)

    assert EXPECTED_FACADE_SURFACES is expected_before
    assert len(_expected_aliases()) == 57
    assert candidate.permanent_aliases != mutated.permanent_aliases
    assert candidate.posture_claims is not EXPECTED_POSTURES
    for observed, expected in zip(
        candidate.facades,
        EXPECTED_FACADE_SURFACES,
        strict=True,
    ):
        assert id(observed.semantic_exports) != id(expected.exports)


def _mutated_expected_semantic(
    field: Literal["rule", "kind", "final_owner"],
    value: str,
) -> tuple[FacadeSurfaceContract, ...]:
    surfaces: list[FacadeSurfaceContract] = []
    for surface in EXPECTED_FACADE_SURFACES:
        exports: list[FacadeExportContract] = []
        for export in surface.exports:
            if export.name == "MAX_TRANSFORMED_PIXELS":
                if field == "rule":
                    export = replace(export, rule=cast(CompatibilityAssertionRule, value))
                elif field == "kind":
                    export = replace(export, kind=cast(CompatibilityExportKind, value))
                else:
                    export = replace(export, final_owner=value)
            exports.append(export)
        surfaces.append(replace(surface, exports=tuple(exports)))
    return tuple(surfaces)


def _mutated_observed_semantic(
    candidate: CompatibilityCandidate,
    field: Literal["rule", "kind", "final_owner"],
    value: str,
) -> CompatibilityCandidate:
    facades: list[FacadeCandidate] = []
    for facade in candidate.facades:
        semantics: list[ObservedFacadeExport] = []
        for export in facade.semantic_exports:
            if export.name == "MAX_TRANSFORMED_PIXELS":
                if field == "rule":
                    export = replace(export, rule=cast(CompatibilityAssertionRule, value))
                elif field == "kind":
                    export = replace(export, kind=cast(CompatibilityExportKind, value))
                else:
                    export = replace(export, final_owner=value)
            semantics.append(export)
        facades.append(replace(facade, semantic_exports=tuple(semantics)))
    return replace(candidate, facades=tuple(facades))


@pytest.mark.parametrize(
    ("side", "field", "value", "diagnostic"),
    [
        ("expected", "rule", "identity", "field=rule"),
        ("expected", "kind", "singleton", "field=kind"),
        ("expected", "final_owner", "pixipix.stages.scale.execution", "field=final_owner"),
        ("observed", "rule", "identity", "field=rule"),
        ("observed", "kind", "singleton", "field=kind"),
        ("observed", "final_owner", "pixipix.stages.scale.execution", "field=final_owner"),
    ],
)
def test_semantic_contract_rejects_independent_inversion(
    side: str,
    field: Literal["rule", "kind", "final_owner"],
    value: str,
    diagnostic: str,
) -> None:
    candidate = derive_compatibility_candidate()
    with pytest.raises(AssertionError, match=diagnostic):
        if side == "expected":
            validate_compatibility_candidate(
                candidate,
                expected_surfaces=_mutated_expected_semantic(field, value),
            )
        else:
            validate_compatibility_candidate(_mutated_observed_semantic(candidate, field, value))


def _mutated_installed_payload(field: str, value: str) -> dict[str, object]:
    payload = copy.deepcopy(compatibility_contract_payload())
    facades = cast(list[dict[str, object]], payload["facades"])
    exports = next(
        cast(list[dict[str, str]], surface["exports"])
        for surface in facades
        if surface["module"] == "pixipix.stages.scale"
    )
    target = next(export for export in exports if export["name"] == "MAX_TRANSFORMED_PIXELS")
    target[field] = value
    return payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule", "identity"),
        ("kind", "singleton"),
        ("final_owner", "pixipix.stages.scale.execution"),
    ],
)
def test_installed_manifest_rejects_false_semantic_payload(field: str, value: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            installed_compatibility_manifest_program(_mutated_installed_payload(field, value)),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"field={field}" in result.stderr


def test_installed_manifest_rejects_false_posture_payload() -> None:
    payload = copy.deepcopy(compatibility_contract_payload())
    postures = cast(list[list[str]], payload["expected_postures"])
    root_posture = next(entry for entry in postures if entry[0] == "pixipix")
    root_posture[1] = "absent-by-design"

    result = subprocess.run(
        [sys.executable, "-B", "-c", installed_compatibility_manifest_program(payload)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "field=posture" in result.stderr


def test_expected_posture_mutation_does_not_change_observed_posture() -> None:
    candidate = derive_compatibility_candidate()
    mutated_expected = tuple(
        (
            module,
            "absent-by-design" if module == "pixipix" else posture,
        )
        for module, posture in EXPECTED_POSTURES
    )

    assert candidate.posture_claims == EXPECTED_POSTURES
    with pytest.raises(AssertionError, match="restore dual posture"):
        validate_compatibility_candidate(candidate, expected_postures=mutated_expected)
    assert candidate.posture_claims == EXPECTED_POSTURES


def test_permanent_candidates_are_derived_from_observed_exports() -> None:
    candidate = derive_compatibility_candidate()
    observed_aliases = tuple(
        sorted(
            f"{facade.module}.{name}"
            for facade in candidate.facades
            for name, _direct_owner in facade.exports
        )
    )

    assert candidate.permanent_aliases == observed_aliases
    assert candidate.permanent_aliases is not _expected_aliases()


@pytest.mark.parametrize(
    ("source", "owner_module", "export"),
    [
        ("from . import analysis as analysis\n", True, False),
        (
            "from pixipix.stages.extract import analysis as analysis\n",
            True,
            False,
        ),
        ("from .analysis import ComponentMap as ComponentMap\n", False, True),
        (
            "from pixipix.stages.extract.analysis import (\n    ComponentMap as ComponentMap,\n)\n",
            False,
            True,
        ),
        ("from ..extract.analysis import ComponentMap as ComponentMap\n", False, True),
    ],
)
def test_facade_scanner_distinguishes_modules_from_symbols(
    tmp_path: Path,
    source: str,
    owner_module: bool,
    export: bool,
) -> None:
    facade_source = tmp_path / "facade.py"
    facade_source.write_text(source, encoding="utf-8")

    candidate = _scan_facade_source(
        "pixipix.stages.extract.synthetic",
        facade_source,
        "pixipix.stages.extract",
    )

    assert bool(candidate.owner_module_imports) is owner_module
    assert bool(candidate.exports) is export


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("omit-export", "expected form"),
        ("add-public-export", "unexpected_export"),
        ("owner-substitution", "pipeline.artifacts"),
        ("wrapper-replacement", "runtime replacement"),
        ("wildcard-import", "wildcard import"),
        ("owner-module-exposure", "owner-module import"),
        ("renamed-public-alias", "renamed import"),
        ("public-body", "public function"),
        ("root-all-removed", "surface=pixipix symbol=__all__"),
        ("root-all-corrupted", "surface=pixipix symbol=__all__"),
        ("stage-all-introduced", "surface=pixipix.stages symbol=__all__"),
        ("root-posture-confused", "restore dual posture"),
        ("stage-posture-confused", "restore dual posture"),
        ("aggregate-lower", "actual form=(56, 55, 1)"),
        ("aggregate-higher", "actual form=(58, 57, 1)"),
        ("underscore-omitted", "symbol=underscore"),
        ("underscore-added", "symbol=underscore"),
        ("underscore-substituted", "symbol=underscore"),
        ("production-removable", "production aliases are not removable"),
    ],
)
def test_compatibility_contract_rejects_upstream_mutation(
    mutation: str,
    diagnostic: str,
) -> None:
    with pytest.raises(AssertionError) as raised:
        validate_compatibility_candidate(_mutated_compatibility_candidate(mutation))

    message = str(raised.value)
    assert diagnostic in message
    assert "surface=" in message
    assert "symbol=" in message
    assert "remediation=" in message
