from __future__ import annotations

import ast
import importlib
import inspect
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
StageModule = Literal[
    "pixipix.stages.extract",
    "pixipix.stages.scale",
    "pixipix.stages.pixelize",
    "pixipix.stages.align",
    "pixipix.stages.io",
]
SymbolKind = Literal["function", "class", "value"]
Classification = Literal["public", "internal", "private-but-consumed", "monkeypatch-sensitive"]


@dataclass(frozen=True, slots=True)
class CompatibilitySymbol:
    module: StageModule
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
    module: StageModule,
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
        "pixipix.stages.scale",
        "decode_stage_input",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
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
        "pixipix.stages.align",
        "decode_stage_input",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(validated: 'ValidatedStageInput') -> 'LoadedStageInput'",
    ),
    _symbol("pixipix.stages.io", "ValidatedStageInput", ("production",), "internal", "class"),
    _symbol("pixipix.stages.io", "LoadedStageInput", ("production",), "internal", "class"),
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
    _symbol("pixipix.stages.io", "Image", ("monkeypatch",), "monkeypatch-sensitive", "value"),
    _symbol(
        "pixipix.stages.io",
        "write_json",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(path: 'Path', value: 'object') -> 'None'",
    ),
    _symbol(
        "pixipix.stages.io",
        "write_png",
        ("monkeypatch",),
        "monkeypatch-sensitive",
        signature="(path: 'Path', pixels: 'UInt8Image') -> 'None'",
    ),
)

PHYSICAL_LAYOUT = (
    PhysicalLayoutAssumption(
        "pixipix/stages/align.py",
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
                for module in targets:
                    prefix = module + "."
                    if node.value.startswith(prefix):
                        consumed.add((module, node.value.removeprefix(prefix).split(".", 1)[0]))
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
        "pixipix.stages.io:Image.open",
        "pixipix.stages.io:write_json",
        "pixipix.stages.io:write_png",
        "pixipix.stages.extract:write_png",
        "pixipix.stages.extract:_materialize_frame_crop",
        "pixipix.stages.scale:decode_stage_input",
        "pixipix.stages.pixelize:decode_stage_input",
        "pixipix.stages.align:decode_stage_input",
    )
    for target in targets:
        assert _resolve_monkeypatch_target(target) is not None


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
