from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Literal

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
        "pixipix.stages.extract.publication",
        "_validate_staged_output",
        ("tests",),
        "private-but-consumed",
        signature="(root: 'Path', metadata: 'StageMetadata') -> 'None'",
    ),
    _symbol(
        "pixipix.stages.extract.publication",
        "_validate_output_location",
        ("tests",),
        "private-but-consumed",
        signature="(output: 'Path') -> 'None'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "extract_source",
        ("tests",),
        "internal",
        signature="(input_path: 'Path', loaded: 'LoadedConfig') -> 'ExtractionRun'",
    ),
    _symbol(
        "pixipix.stages.extract.publication",
        "_valid_frame_png",
        ("tests",),
        "private-but-consumed",
        signature="(path: 'Path', expected_size: 'tuple[int, int] | None' = None) -> 'bool'",
    ),
    _symbol(
        "pixipix.stages.extract.analysis",
        "load_source",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(path: 'Path', config: 'SourceConfig') -> 'SourceImage'",
    ),
    _symbol(
        "pixipix.stages.extract.publication",
        "Image",
        ("monkeypatch",),
        "private-but-consumed",
        "value",
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
        "pixipix.stages.extract.publication",
        "write_png",
        ("monkeypatch",),
        "private-but-consumed",
        signature="(path: 'Path', pixels: 'UInt8Image') -> 'None'",
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
            "(output: 'Path', stage: \"Literal['scale', 'pixelize', 'align']\", "
            "*, force: 'bool' = False) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "publish_stage_output",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: \"Literal['scale', 'pixelize', 'align']\", "
            "metadata: 'object', frames: 'tuple[OutputFrameImage, ...]', "
            "*, force: 'bool' = False) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "_valid_owned_output",
        ("facade",),
        "private-but-consumed",
        signature="(path: 'Path', stage: 'StageName') -> 'bool'",
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
            "(output: 'Path', stage: \"Literal['scale', 'pixelize', 'align']\", "
            "*, force: 'bool' = False) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "publish_stage_output",
        ("production",),
        "internal",
        signature=(
            "(output: 'Path', stage: \"Literal['scale', 'pixelize', 'align']\", "
            "metadata: 'object', frames: 'tuple[OutputFrameImage, ...]', "
            "*, force: 'bool' = False) -> 'None'"
        ),
    ),
    _symbol(
        "pixipix.stages.io",
        "_valid_owned_output",
        ("smoke",),
        "private-but-consumed",
        signature="(path: 'Path', stage: 'StageName') -> 'bool'",
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
        "pixipix.stages.align.api.decode_stage_input",
        "pixipix.stages.extract.analysis.load_source",
        "pixipix.stages.extract.api._materialize_frame_crop",
        "pixipix.stages.extract.publication._validate_staged_output",
        "pixipix.stages.extract.publication.write_png",
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
        "pixipix.stages.extract.publication.Image",
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
        "pixipix.stages.extract.publication._prepare_target",
        "pixipix.stages.extract.publication._remove_temporary_tree",
        "pixipix.stages.extract.publication._valid_frame_png",
    }
)

EXPECTED_DECLARED_PATCH_SEAMS = frozenset(
    {
        "pixipix.pipeline.publication.write_json",
        "pixipix.pipeline.publication.write_png",
        "pixipix.stages.align.api.decode_stage_input",
        "pixipix.stages.extract.analysis.load_source",
        "pixipix.stages.extract.api._materialize_frame_crop",
        "pixipix.stages.extract.publication._validate_staged_output",
        "pixipix.stages.extract.publication.write_png",
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
        "pixipix.stages.extract.publication.Image",
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
        "pixipix.stages.extract.publication._prepare_target",
        "pixipix.stages.extract.publication._remove_temporary_tree",
        "pixipix.stages.extract.publication._valid_frame_png",
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
    assert len(DECLARED_PATCH_SEAMS) == 9
    assert len(OWNER_LOCAL_DEPENDENCIES) == 10
    assert {"pathlib.Path.replace"} == BROAD_NECESSARY_SEAMS
    assert len(DELIBERATE_NON_SEAMS) == 5


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
        declared.add("pixipix.stages.extract.publication._valid_frame_png")
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
        non_seams.remove("pixipix.stages.extract.publication._valid_frame_png")
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
            "pixipix.stages.extract.publication._valid_frame_png",
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
            "pixipix.stages.extract.publication._valid_frame_png",
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
        "_valid_frame_png",
        "_validate_staged_output",
        "_validate_output_location",
    ):
        assert not hasattr(facade, name), f"migrated private facade binding remains: {name}"
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
