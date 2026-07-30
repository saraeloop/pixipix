from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

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
    "pixipix.stages.scale",
    "pixipix.stages.scale.api",
    "pixipix.stages.pixelize",
    "pixipix.stages.pixelize.api",
    "pixipix.stages.align",
    "pixipix.stages.align.api",
    "pixipix.stages.io",
]
SymbolKind = Literal["function", "class", "value"]
Classification = Literal["public", "internal", "private-but-consumed", "monkeypatch-sensitive"]


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
        "pixipix.stages.extract",
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
        "pixipix.stages.extract",
        "_analyze",
        ("tests",),
        "private-but-consumed",
        signature="(input_path: 'Path', loaded: 'LoadedConfig') -> '_Analysis'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "_materialize_frame_crop",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(analysis: '_Analysis', component: 'Component', frame: 'ExtractedFrame') "
        "-> 'FrameImage'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "_padded_bounds",
        ("tests",),
        "private-but-consumed",
        signature="(bounds: 'Rect', padding: 'int', width: 'int', height: 'int') -> 'Rect'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "_validate_staged_output",
        ("tests",),
        "private-but-consumed",
        signature="(root: 'Path', metadata: 'StageMetadata') -> 'None'",
    ),
    _symbol(
        "pixipix.stages.extract",
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
        "pixipix.stages.extract",
        "_valid_frame_png",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(path: 'Path', expected_size: 'tuple[int, int] | None' = None) -> 'bool'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "load_source",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(path: 'Path', config: 'SourceConfig') -> 'SourceImage'",
    ),
    _symbol(
        "pixipix.stages.extract",
        "Image",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        "value",
    ),
    _symbol(
        "pixipix.stages.extract",
        "np",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        "value",
    ),
    _symbol(
        "pixipix.stages.extract",
        "write_png",
        ("monkeypatch",),
        "monkeypatch-sensitive",
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
        "monkeypatch-sensitive",
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
        "monkeypatch-sensitive",
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
        "monkeypatch-sensitive",
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
        "monkeypatch-sensitive",
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
        "write_json",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(path: 'Path', value: 'object') -> 'None'",
    ),
    _symbol(
        "pixipix.pipeline.publication",
        "write_png",
        ("monkeypatch",),
        "monkeypatch-sensitive",
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


def _resolve_monkeypatch_target(target: str) -> object:
    module_name, _, attribute_path = target.partition(":")
    value: object = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    return value


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


def test_monkeypatch_sensitive_bindings_resolve_at_current_paths() -> None:
    targets = (
        "pixipix.pipeline.input:Image.open",
        "pixipix.pipeline.publication:write_json",
        "pixipix.pipeline.publication:write_png",
        "pixipix.stages.extract:Image.open",
        "pixipix.stages.extract:np.zeros",
        "pixipix.stages.extract:load_source",
        "pixipix.stages.extract:write_png",
        "pixipix.stages.extract:_materialize_frame_crop",
        "pixipix.stages.scale.api:decode_stage_input",
        "pixipix.stages.pixelize.api:decode_stage_input",
        "pixipix.stages.align.api:decode_stage_input",
    )
    for target in targets:
        assert _resolve_monkeypatch_target(target) is not None


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


def test_extract_package_preserves_module_binding_and_type_identities() -> None:
    package = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract"
    extract = importlib.import_module("pixipix.stages.extract")
    imageio = importlib.import_module("pixipix.imageio")
    models = importlib.import_module("pixipix.models")
    resources = importlib.import_module("pixipix.resources")
    pillow_image = importlib.import_module("PIL.Image")
    numpy = importlib.import_module("numpy")
    extract_file = extract.__file__

    assert extract_file is not None
    assert Path(extract_file).resolve() == (package / "__init__.py").resolve()
    assert extract.__spec__ is not None
    assert extract.__spec__.submodule_search_locations is not None
    defined_names = (
        "ComponentMap",
        "_Analysis",
        "label_components",
        "filter_components",
        "order_components",
        "_analyze",
        "inspect_source",
        "_padded_bounds",
        "project_extract_resources",
        "project_extracted_frames",
        "_materialize_frame_crop",
        "extract_source",
        "_stage_metadata",
        "_valid_marker",
        "_frame_path",
        "_valid_frame_png",
        "_validate_staged_payload",
        "_validate_staged_output",
        "_valid_owned_output",
        "_is_trusted_system_tmp_alias",
        "_validate_output_location",
        "_prepare_target",
        "_remove_temporary_tree",
        "publish_extraction",
    )
    for name in defined_names:
        assert vars(extract)[name].__module__ == "pixipix.stages.extract"
    imported_owners = {
        "ExtractedFrame": models,
        "ExtractionResult": models,
        "ExtractionRun": models,
        "FrameImage": models,
        "InspectionResult": models,
        "ResourceProjection": resources,
        "load_source": imageio,
        "write_png": imageio,
    }
    for name, owner in imported_owners.items():
        assert vars(extract)[name] is vars(owner)[name]
    assert vars(extract)["Image"] is pillow_image
    assert vars(extract)["np"] is numpy
    assert vars(extract)["FOUR_NEIGHBORS"] == ((-1, 0), (0, -1), (0, 1), (1, 0))
    assert vars(extract)["EIGHT_NEIGHBORS"] == (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    assert sys.modules["pixipix.stages.extract"] is extract
    assert "pixipix.stages.extract.__init__" not in sys.modules

    code = (
        "import pathlib, sys; "
        "import pixipix.stages.extract as extract; "
        "import pixipix.imageio as imageio; "
        "import pixipix.models as models; "
        "import pixipix.resources as resources; "
        "assert pathlib.Path(extract.__file__).name == '__init__.py'; "
        "assert extract.__spec__.submodule_search_locations is not None; "
        "assert extract.ComponentMap.__module__ == 'pixipix.stages.extract'; "
        "assert extract._Analysis.__module__ == 'pixipix.stages.extract'; "
        "assert extract.inspect_source.__module__ == 'pixipix.stages.extract'; "
        "assert extract.extract_source.__module__ == 'pixipix.stages.extract'; "
        "assert extract.publish_extraction.__module__ == 'pixipix.stages.extract'; "
        "assert extract.ExtractionRun is models.ExtractionRun; "
        "assert extract.ExtractionResult is models.ExtractionResult; "
        "assert extract.FrameImage is models.FrameImage; "
        "assert extract.ResourceProjection is resources.ResourceProjection; "
        "assert extract.load_source is imageio.load_source; "
        "assert extract.write_png is imageio.write_png; "
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
