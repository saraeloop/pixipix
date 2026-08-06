from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONAL_MODULES = {
    "pixipix._scale_geometry",
    "pixipix.config",
    "pixipix.errors",
    "pixipix.imageio",
    "pixipix.models",
    "pixipix.resources",
    "pixipix.serialization",
}
PIPELINE_MODULES = {
    "pixipix.pipeline",
    "pixipix.pipeline.artifacts",
    "pixipix.pipeline.input",
    "pixipix.pipeline.publication",
}
PIPELINE_ALLOWED_PIXIPIX_DEPENDENCIES = {
    "pixipix.pipeline": set(),
    "pixipix.pipeline.artifacts": {
        "pixipix.errors",
        "pixipix.models",
    },
    "pixipix.pipeline.input": {
        "pixipix._scale_geometry",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.pipeline.artifacts",
    },
    "pixipix.pipeline.publication": {
        "pixipix.errors",
        "pixipix.imageio",
        "pixipix.models",
        "pixipix.pipeline.artifacts",
        "pixipix.serialization",
    },
}
STAGE_IMPLEMENTATIONS = {
    "pixipix.stages.extract",
    "pixipix.stages.scale",
    "pixipix.stages.pixelize",
    "pixipix.stages.align",
}
STAGE_PUBLISHERS = {
    "publish_extraction",
    "publish_scale",
    "publish_pixelize",
    "publish_align",
}
EXTRACT_MODULES = {
    "pixipix.stages.extract",
    "pixipix.stages.extract.analysis",
    "pixipix.stages.extract.api",
    "pixipix.stages.extract.execution",
    "pixipix.stages.extract.metadata",
    "pixipix.stages.extract.planning",
    "pixipix.stages.extract.publication",
}
EXTRACT_ALLOWED_INTERNAL_SYMBOLS = {
    ("pixipix.stages.extract", "pixipix.stages.extract.analysis"): {
        "ComponentMap",
        "filter_components",
        "label_components",
        "order_components",
    },
    ("pixipix.stages.extract", "pixipix.stages.extract.api"): {
        "extract_source",
        "inspect_source",
    },
    ("pixipix.stages.extract", "pixipix.stages.extract.planning"): {
        "project_extract_resources",
        "project_extracted_frames",
    },
    ("pixipix.stages.extract", "pixipix.stages.extract.publication"): {
        "publish_extraction",
    },
    ("pixipix.stages.extract.api", "pixipix.stages.extract.analysis"): {
        "_analyze",
    },
    ("pixipix.stages.extract.api", "pixipix.stages.extract.execution"): {
        "_materialize_frame_crop",
    },
    ("pixipix.stages.extract.api", "pixipix.stages.extract.planning"): {
        "project_extract_resources",
        "project_extracted_frames",
    },
    ("pixipix.stages.extract.execution", "pixipix.stages.extract.analysis"): {
        "_Analysis",
    },
    ("pixipix.stages.extract.planning", "pixipix.stages.extract.analysis"): {
        "_Analysis",
    },
    ("pixipix.stages.extract.publication", "pixipix.stages.extract.api"): {
        "extract_source",
    },
    ("pixipix.stages.extract.publication", "pixipix.stages.extract.metadata"): {
        "_stage_metadata",
        "_valid_owned_extract_metadata",
    },
}
EXTRACT_ALLOWED_PIXIPIX_DEPENDENCIES = {
    "pixipix.stages.extract": {
        "pixipix.stages.extract.analysis",
        "pixipix.stages.extract.api",
        "pixipix.stages.extract.planning",
        "pixipix.stages.extract.publication",
    },
    "pixipix.stages.extract.analysis": {
        "pixipix.config",
        "pixipix.errors",
        "pixipix.imageio",
        "pixipix.models",
    },
    "pixipix.stages.extract.api": {
        "pixipix.config",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.resources",
        "pixipix.stages.extract.analysis",
        "pixipix.stages.extract.execution",
        "pixipix.stages.extract.planning",
    },
    "pixipix.stages.extract.execution": {
        "pixipix.models",
        "pixipix.stages.extract.analysis",
    },
    "pixipix.stages.extract.metadata": {
        "pixipix",
        "pixipix.config",
        "pixipix.models",
    },
    "pixipix.stages.extract.planning": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.resources",
        "pixipix.stages.extract.analysis",
    },
    "pixipix.stages.extract.publication": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.publication",
        "pixipix.stages.extract.api",
        "pixipix.stages.extract.metadata",
    },
}
EXTRACT_ALLOWED_EXTERNAL_IMPORTS = {
    "pixipix.stages.extract": set(),
    "pixipix.stages.extract.analysis": {
        "__future__",
        "collections",
        "dataclasses",
        "numpy",
        "numpy.typing",
        "pathlib",
    },
    "pixipix.stages.extract.api": {"__future__", "pathlib"},
    "pixipix.stages.extract.execution": {"__future__", "numpy"},
    "pixipix.stages.extract.metadata": {"__future__"},
    "pixipix.stages.extract.planning": {"__future__", "pathlib"},
    "pixipix.stages.extract.publication": {"__future__", "pathlib"},
}
EXTRACT_ALLOWED_ROOT_IMPORTS: set[tuple[str, str, tuple[str, ...]]] = {
    (
        "pixipix.stages.extract.metadata",
        "pixipix",
        ("__version__",),
    ),
}
EXTRACT_ALLOWED_PIPELINE_SYMBOLS = {
    ("pixipix.stages.extract.publication", "pixipix.pipeline.publication"): {
        "OutputFrameImage",
        "publish_stage_output",
        "validate_stage_output_target",
    },
}
ALIGN_MODULES = {
    "pixipix.stages.align",
    "pixipix.stages.align.api",
    "pixipix.stages.align.execution",
    "pixipix.stages.align.geometry",
    "pixipix.stages.align.planning",
}
ALIGN_ALLOWED_INTERNAL_SYMBOLS = {
    ("pixipix.stages.align", "pixipix.stages.align.api"): {"publish_align"},
    ("pixipix.stages.align", "pixipix.stages.align.execution"): {
        "AlignmentRun",
        "align_stage",
        "compose_aligned_canvas",
    },
    ("pixipix.stages.align", "pixipix.stages.align.geometry"): {
        "EMPTY_RECTANGLE",
        "calculate_alignment_frame",
        "mathematical_floor_center",
    },
    ("pixipix.stages.align", "pixipix.stages.align.planning"): {
        "AlignmentStagePlan",
        "clipping_finding",
        "project_align_resources",
        "project_align_stage",
    },
    ("pixipix.stages.align.api", "pixipix.stages.align.execution"): {"align_stage"},
    ("pixipix.stages.align.api", "pixipix.stages.align.planning"): {"project_align_stage"},
    ("pixipix.stages.align.execution", "pixipix.stages.align.planning"): {
        "AlignmentStagePlan",
        "_require_output_config",
    },
    ("pixipix.stages.align.planning", "pixipix.stages.align.geometry"): {
        "calculate_alignment_frame"
    },
}
ALIGN_ALLOWED_PIXIPIX_DEPENDENCIES = {
    "pixipix.stages.align": {
        "pixipix.stages.align.api",
        "pixipix.stages.align.execution",
        "pixipix.stages.align.geometry",
        "pixipix.stages.align.planning",
    },
    "pixipix.stages.align.api": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.resources",
        "pixipix.stages.align.execution",
        "pixipix.stages.align.planning",
    },
    "pixipix.stages.align.execution": {
        "pixipix",
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.stages.align.planning",
    },
    "pixipix.stages.align.geometry": {
        "pixipix.config",
        "pixipix.models",
    },
    "pixipix.stages.align.planning": {
        "pixipix.config",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.resources",
        "pixipix.stages.align.geometry",
    },
}
ALIGN_ALLOWED_EXTERNAL_IMPORTS = {
    "pixipix.stages.align": {"__future__"},
    "pixipix.stages.align.api": {"__future__", "pathlib"},
    "pixipix.stages.align.execution": {"__future__", "dataclasses", "numpy"},
    "pixipix.stages.align.geometry": {"__future__", "pathlib"},
    "pixipix.stages.align.planning": {"__future__", "dataclasses"},
}
ALIGN_ALLOWED_ROOT_IMPORTS: set[tuple[str, str, tuple[str, ...]]] = {
    (
        "pixipix.stages.align.execution",
        "pixipix",
        ("__version__",),
    ),
}
ALIGN_ALLOWED_PIPELINE_SYMBOLS = {
    ("pixipix.stages.align.api", "pixipix.pipeline.input"): {
        "decode_stage_input",
        "validate_stage_input",
    },
    ("pixipix.stages.align.api", "pixipix.pipeline.publication"): {
        "publish_stage_output",
        "validate_stage_output_target",
    },
    ("pixipix.stages.align.execution", "pixipix.pipeline.input"): {
        "LoadedStageInput",
    },
    ("pixipix.stages.align.execution", "pixipix.pipeline.publication"): {
        "OutputFrameImage",
    },
    ("pixipix.stages.align.planning", "pixipix.pipeline.input"): {
        "ValidatedStageInput",
    },
    ("pixipix.stages.align.api", "pixipix.resources"): {
        "enforce_resource_policy",
    },
    ("pixipix.stages.align.planning", "pixipix.resources"): {
        "ResourceProjection",
    },
}
SCALE_MODULES = {
    "pixipix.stages.scale",
    "pixipix.stages.scale.api",
    "pixipix.stages.scale.execution",
    "pixipix.stages.scale.geometry",
    "pixipix.stages.scale.metadata",
    "pixipix.stages.scale.planning",
}
SCALE_ALLOWED_INTERNAL_SYMBOLS = {
    ("pixipix.stages.scale", "pixipix.stages.scale.api"): {
        "publish_scale",
    },
    ("pixipix.stages.scale", "pixipix.stages.scale.execution"): {
        "ScaleRun",
        "premultiplied_box_resize",
        "scale_stage",
    },
    ("pixipix.stages.scale", "pixipix.stages.scale.geometry"): {
        "round_channel_half_away_from_zero",
        "round_half_away_from_zero",
        "transformed_dimension",
    },
    ("pixipix.stages.scale", "pixipix.stages.scale.planning"): {
        "MAX_TRANSFORMED_PIXELS",
        "ScaleStagePlan",
        "project_scale_resources",
        "project_scale_stage",
    },
    ("pixipix.stages.scale.api", "pixipix.stages.scale.execution"): {
        "scale_stage",
    },
    ("pixipix.stages.scale.api", "pixipix.stages.scale.planning"): {
        "project_scale_stage",
    },
    ("pixipix.stages.scale.execution", "pixipix.stages.scale.geometry"): {
        "round_channel_half_away_from_zero",
    },
    ("pixipix.stages.scale.execution", "pixipix.stages.scale.metadata"): {
        "build_scale_metadata",
    },
    ("pixipix.stages.scale.execution", "pixipix.stages.scale.planning"): {
        "ScaleStagePlan",
        "_require_scale_config",
    },
    ("pixipix.stages.scale.metadata", "pixipix.stages.scale.planning"): {
        "ScaleStagePlan",
    },
}
SCALE_ALLOWED_PIXIPIX_DEPENDENCIES = {
    "pixipix.stages.scale": {
        "pixipix.stages.scale.api",
        "pixipix.stages.scale.execution",
        "pixipix.stages.scale.geometry",
        "pixipix.stages.scale.planning",
    },
    "pixipix.stages.scale.api": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.resources",
        "pixipix.stages.scale.execution",
        "pixipix.stages.scale.planning",
    },
    "pixipix.stages.scale.execution": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.stages.scale.geometry",
        "pixipix.stages.scale.metadata",
        "pixipix.stages.scale.planning",
    },
    "pixipix.stages.scale.geometry": {"pixipix._scale_geometry"},
    "pixipix.stages.scale.metadata": {
        "pixipix",
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.stages.scale.planning",
    },
    "pixipix.stages.scale.planning": {
        "pixipix._scale_geometry",
        "pixipix.config",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.resources",
    },
}
SCALE_ALLOWED_EXTERNAL_IMPORTS = {
    "pixipix.stages.scale": {"__future__"},
    "pixipix.stages.scale.api": {"__future__", "pathlib"},
    "pixipix.stages.scale.execution": {"__future__", "dataclasses", "numpy", "PIL"},
    "pixipix.stages.scale.geometry": {"__future__", "math"},
    "pixipix.stages.scale.metadata": {"__future__"},
    "pixipix.stages.scale.planning": {"__future__", "dataclasses"},
}
SCALE_ALLOWED_ROOT_IMPORTS: set[tuple[str, str, tuple[str, ...]]] = {
    (
        "pixipix.stages.scale.metadata",
        "pixipix",
        ("__version__",),
    ),
}
SCALE_ALLOWED_PIPELINE_SYMBOLS = {
    ("pixipix.stages.scale.api", "pixipix.pipeline.input"): {
        "decode_stage_input",
        "validate_stage_input",
    },
    ("pixipix.stages.scale.api", "pixipix.pipeline.publication"): {
        "publish_stage_output",
        "validate_stage_output_target",
    },
    ("pixipix.stages.scale.execution", "pixipix.pipeline.input"): {
        "LoadedStageInput",
    },
    ("pixipix.stages.scale.execution", "pixipix.pipeline.publication"): {
        "OutputFrameImage",
    },
    ("pixipix.stages.scale.metadata", "pixipix.pipeline.input"): {
        "LoadedStageInput",
    },
    ("pixipix.stages.scale.planning", "pixipix.pipeline.input"): {
        "ValidatedStageInput",
    },
    ("pixipix.stages.scale.api", "pixipix.resources"): {
        "enforce_resource_policy",
    },
    ("pixipix.stages.scale.planning", "pixipix.resources"): {
        "ResourceProjection",
    },
}
PIXELIZE_MODULES = {
    "pixipix.stages.pixelize",
    "pixipix.stages.pixelize.api",
    "pixipix.stages.pixelize.execution",
    "pixipix.stages.pixelize.metadata",
    "pixipix.stages.pixelize.planning",
}
PIXELIZE_ALLOWED_INTERNAL_SYMBOLS = {
    ("pixipix.stages.pixelize", "pixipix.stages.pixelize.api"): {
        "publish_pixelize",
    },
    ("pixipix.stages.pixelize", "pixipix.stages.pixelize.execution"): {
        "PixelizeRun",
        "PreparedCellGrid",
        "apply_alpha_policy",
        "pixelize_prepared_grid",
        "pixelize_stage",
        "prepare_cell_grid",
        "representative_pixel",
        "round_channel_half_away_from_zero",
    },
    ("pixipix.stages.pixelize", "pixipix.stages.pixelize.planning"): {
        "MAX_PREPARED_PIXELS",
        "CellGridProjection",
        "PixelizeStagePlan",
        "project_cell_grid",
        "project_pixelize_resources",
        "project_pixelize_stage",
    },
    ("pixipix.stages.pixelize.api", "pixipix.stages.pixelize.execution"): {
        "pixelize_stage",
    },
    ("pixipix.stages.pixelize.api", "pixipix.stages.pixelize.planning"): {
        "project_pixelize_stage",
    },
    ("pixipix.stages.pixelize.execution", "pixipix.stages.pixelize.metadata"): {
        "build_pixelize_metadata",
    },
    ("pixipix.stages.pixelize.execution", "pixipix.stages.pixelize.planning"): {
        "CellGridProjection",
        "PixelizeStagePlan",
        "_require_pixelize_config",
        "project_cell_grid",
    },
    ("pixipix.stages.pixelize.metadata", "pixipix.stages.pixelize.planning"): {
        "PixelizeStagePlan",
    },
}
PIXELIZE_ALLOWED_PIXIPIX_DEPENDENCIES = {
    "pixipix.stages.pixelize": {
        "pixipix.stages.pixelize.api",
        "pixipix.stages.pixelize.execution",
        "pixipix.stages.pixelize.planning",
    },
    "pixipix.stages.pixelize.api": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.resources",
        "pixipix.stages.pixelize.execution",
        "pixipix.stages.pixelize.planning",
    },
    "pixipix.stages.pixelize.execution": {
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.pipeline.publication",
        "pixipix.stages.pixelize.metadata",
        "pixipix.stages.pixelize.planning",
        "pixipix.stages.scale",
    },
    "pixipix.stages.pixelize.metadata": {
        "pixipix",
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.stages.pixelize.planning",
    },
    "pixipix.stages.pixelize.planning": {
        "pixipix.config",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.resources",
    },
}
PIXELIZE_ALLOWED_EXTERNAL_IMPORTS = {
    "pixipix.stages.pixelize": {"__future__"},
    "pixipix.stages.pixelize.api": {"__future__", "pathlib"},
    "pixipix.stages.pixelize.execution": {"__future__", "dataclasses", "numpy"},
    "pixipix.stages.pixelize.metadata": {"__future__"},
    "pixipix.stages.pixelize.planning": {"__future__", "dataclasses"},
}
PIXELIZE_ALLOWED_ROOT_IMPORTS: set[tuple[str, str, tuple[str, ...]]] = {
    (
        "pixipix.stages.pixelize.metadata",
        "pixipix",
        ("__version__",),
    ),
}
PIXELIZE_ALLOWED_PIPELINE_SYMBOLS = {
    ("pixipix.stages.pixelize.api", "pixipix.pipeline.input"): {
        "decode_stage_input",
        "validate_stage_input",
    },
    ("pixipix.stages.pixelize.api", "pixipix.pipeline.publication"): {
        "publish_stage_output",
        "validate_stage_output_target",
    },
    ("pixipix.stages.pixelize.execution", "pixipix.pipeline.input"): {
        "LoadedStageInput",
    },
    ("pixipix.stages.pixelize.execution", "pixipix.pipeline.publication"): {
        "OutputFrameImage",
    },
    ("pixipix.stages.pixelize.metadata", "pixipix.pipeline.input"): {
        "LoadedStageInput",
    },
    ("pixipix.stages.pixelize.planning", "pixipix.pipeline.input"): {
        "ValidatedStageInput",
    },
    ("pixipix.stages.pixelize.api", "pixipix.resources"): {
        "enforce_resource_policy",
    },
    ("pixipix.stages.pixelize.planning", "pixipix.resources"): {
        "ResourceProjection",
    },
}
PIPELINE_ALLOWED_INTERNAL_SYMBOLS = {
    ("pixipix.pipeline.input", "pixipix.pipeline.artifacts"): {
        "_dimensions",
        "_is_output_marker",
        "_is_untrusted_path_component",
        "_is_schema_version_one",
        "_positive_dimension",
        "_read_json_object",
        "_safe_frame_relative",
    },
    ("pixipix.pipeline.publication", "pixipix.pipeline.artifacts"): {
        "StageName",
        "_dimensions",
        "_is_output_marker",
        "_is_untrusted_path_component",
        "_is_schema_version_one",
        "_read_json_object",
        "_safe_frame_relative",
    },
}
CHANNEL_ROUNDING_SYMBOL = "round_channel_half_away_from_zero"
CHANNEL_ROUNDING_OWNER = "pixipix.stages.scale.geometry"
CHANNEL_ROUNDING_CONSUMER = "pixipix.stages.pixelize.execution"
CHANNEL_ROUNDING_CALL_SITE = "_alpha_weighted_majority"
CHANNEL_ROUNDING_FUNCTION_SOURCE = """
def round_channel_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("channel value must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    rounded = magnitude if value >= 0 else -magnitude
    return min(255, max(0, rounded))
"""


@dataclass(frozen=True, slots=True)
class ImportEdge:
    importer: str
    imported: str
    names: tuple[str, ...]
    line: int
    hard: bool
    renamed: bool
    hidden: bool
    type_only: bool


@dataclass(frozen=True, slots=True)
class StageDependencyContract:
    name: str
    modules: set[str]
    allowed_internal_symbols: dict[tuple[str, str], set[str]]
    allowed_pixipix_dependencies: dict[str, set[str]]
    allowed_external_imports: dict[str, set[str]] | None
    allowed_root_imports: set[tuple[str, str, tuple[str, ...]]]
    allowed_shared_symbols: dict[tuple[str, str], set[str]]


EXTRACT_DEPENDENCY_CONTRACT = StageDependencyContract(
    name="extract",
    modules=EXTRACT_MODULES,
    allowed_internal_symbols=EXTRACT_ALLOWED_INTERNAL_SYMBOLS,
    allowed_pixipix_dependencies=EXTRACT_ALLOWED_PIXIPIX_DEPENDENCIES,
    allowed_external_imports=EXTRACT_ALLOWED_EXTERNAL_IMPORTS,
    allowed_root_imports=EXTRACT_ALLOWED_ROOT_IMPORTS,
    allowed_shared_symbols=EXTRACT_ALLOWED_PIPELINE_SYMBOLS,
)
ALIGN_DEPENDENCY_CONTRACT = StageDependencyContract(
    name="align",
    modules=ALIGN_MODULES,
    allowed_internal_symbols=ALIGN_ALLOWED_INTERNAL_SYMBOLS,
    allowed_pixipix_dependencies=ALIGN_ALLOWED_PIXIPIX_DEPENDENCIES,
    allowed_external_imports=ALIGN_ALLOWED_EXTERNAL_IMPORTS,
    allowed_root_imports=ALIGN_ALLOWED_ROOT_IMPORTS,
    allowed_shared_symbols=ALIGN_ALLOWED_PIPELINE_SYMBOLS,
)
SCALE_DEPENDENCY_CONTRACT = StageDependencyContract(
    name="scale",
    modules=SCALE_MODULES,
    allowed_internal_symbols=SCALE_ALLOWED_INTERNAL_SYMBOLS,
    allowed_pixipix_dependencies=SCALE_ALLOWED_PIXIPIX_DEPENDENCIES,
    allowed_external_imports=SCALE_ALLOWED_EXTERNAL_IMPORTS,
    allowed_root_imports=SCALE_ALLOWED_ROOT_IMPORTS,
    allowed_shared_symbols=SCALE_ALLOWED_PIPELINE_SYMBOLS,
)
PIXELIZE_DEPENDENCY_CONTRACT = StageDependencyContract(
    name="pixelize",
    modules=PIXELIZE_MODULES,
    allowed_internal_symbols=PIXELIZE_ALLOWED_INTERNAL_SYMBOLS,
    allowed_pixipix_dependencies=PIXELIZE_ALLOWED_PIXIPIX_DEPENDENCIES,
    allowed_external_imports=PIXELIZE_ALLOWED_EXTERNAL_IMPORTS,
    allowed_root_imports=PIXELIZE_ALLOWED_ROOT_IMPORTS,
    allowed_shared_symbols=PIXELIZE_ALLOWED_PIPELINE_SYMBOLS,
)
PIPELINE_DEPENDENCY_CONTRACT = StageDependencyContract(
    name="pipeline",
    modules=PIPELINE_MODULES,
    allowed_internal_symbols=PIPELINE_ALLOWED_INTERNAL_SYMBOLS,
    allowed_pixipix_dependencies=PIPELINE_ALLOWED_PIXIPIX_DEPENDENCIES,
    allowed_external_imports=None,
    allowed_root_imports=set(),
    allowed_shared_symbols={},
)
STAGE_DEPENDENCY_CONTRACTS = (
    EXTRACT_DEPENDENCY_CONTRACT,
    ALIGN_DEPENDENCY_CONTRACT,
    SCALE_DEPENDENCY_CONTRACT,
    PIXELIZE_DEPENDENCY_CONTRACT,
)
GOVERNED_PRODUCTION_MODULES = set().union(
    PIPELINE_MODULES,
    *(contract.modules for contract in STAGE_DEPENDENCY_CONTRACTS),
)
GOVERNED_DYNAMIC_TARGETS = set().union(
    *(contract.modules for contract in STAGE_DEPENDENCY_CONTRACTS),
    PIPELINE_MODULES,
    {
        target
        for contract in STAGE_DEPENDENCY_CONTRACTS
        for _importer, target in contract.allowed_shared_symbols
    },
    {"pixipix.stages"},
)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, importer: str, package: str) -> None:
        self.importer = importer
        self.package = package
        self.edges: list[ImportEdge] = []
        self._scope_depth = 0
        self._type_checking_depth = 0
        self._hidden_depth = 0
        self._type_checking_names: set[str] = set()
        self._typing_module_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_depth += 1
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_depth += 1
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1
        self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        is_type_checking = self._is_type_checking_test(node.test)
        if is_type_checking:
            self._type_checking_depth += 1
        self._hidden_depth += 1
        for statement in node.body:
            self.visit(statement)
        self._hidden_depth -= 1
        if is_type_checking:
            self._type_checking_depth -= 1
        self._hidden_depth += 1
        for statement in node.orelse:
            self.visit(statement)
        self._hidden_depth -= 1

    def _is_type_checking_test(self, node: ast.expr) -> bool:
        direct_binding = (isinstance(node, ast.Name) and node.id in self._type_checking_names) or (
            isinstance(node, ast.Attribute)
            and node.attr == "TYPE_CHECKING"
            and isinstance(node.value, ast.Name)
            and node.value.id in self._typing_module_names
        )
        return direct_binding or (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.And)
            and any(self._is_type_checking_test(value) for value in node.values)
        )

    def _type_binding_kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in self._type_checking_names:
                return "type_checking"
            if node.id in self._typing_module_names:
                return "typing_module"
        return None

    def _rebind_type_names(self, names: set[str], value: ast.expr | None) -> None:
        binding_kind = self._type_binding_kind(value) if value is not None else None
        self._type_checking_names.difference_update(names)
        self._typing_module_names.difference_update(names)
        if binding_kind == "type_checking":
            self._type_checking_names.update(names)
        elif binding_kind == "typing_module":
            self._typing_module_names.update(names)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        self._rebind_type_names(names, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            self._rebind_type_names({node.target.id}, node.value)

    def visit_Try(self, node: ast.Try) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_With(self, node: ast.With) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_Match(self, node: ast.Match) -> None:
        self._hidden_depth += 1
        self.generic_visit(node)
        self._hidden_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            binding_name = alias.asname or alias.name.partition(".")[0]
            self._rebind_type_names({binding_name}, None)
            if alias.name == "typing":
                self._typing_module_names.add(binding_name)
            self.edges.append(
                ImportEdge(
                    self.importer,
                    alias.name,
                    (),
                    node.lineno,
                    self._scope_depth == 0 and self._type_checking_depth == 0,
                    alias.asname is not None and alias.asname != alias.name,
                    self._hidden_depth > 0,
                    self._type_checking_depth > 0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported = node.module or ""
        if node.level:
            imported = importlib.util.resolve_name(
                "." * node.level + imported,
                self.package,
            )
        for alias in node.names:
            binding_name = alias.asname or alias.name
            self._rebind_type_names({binding_name}, None)
            if imported == "typing" and alias.name == "TYPE_CHECKING":
                self._type_checking_names.add(binding_name)
        self.edges.append(
            ImportEdge(
                self.importer,
                imported,
                tuple(alias.name for alias in node.names),
                node.lineno,
                self._scope_depth == 0 and self._type_checking_depth == 0,
                any(
                    alias.asname is not None and alias.asname != alias.name for alias in node.names
                ),
                self._hidden_depth > 0,
                self._type_checking_depth > 0,
            )
        )


class _DynamicImportCollector(ast.NodeVisitor):
    def __init__(self, importer: str) -> None:
        self.importer = importer
        self.import_names = {"__import__"}
        self.importlib_names: set[str] = set()
        self.calls: list[tuple[str, int]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            binding_name = alias.asname or alias.name.partition(".")[0]
            self._rebind_import_names({binding_name}, None)
            if alias.name == "importlib":
                self.importlib_names.add(binding_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            binding_name = alias.asname or alias.name
            self._rebind_import_names({binding_name}, None)
            if node.module == "importlib" and alias.name == "import_module":
                self.import_names.add(binding_name)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        binding_kind = self._dynamic_binding_kind(node.value)
        self.visit(node.value)
        self._rebind_import_names(names, binding_kind)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            binding_kind = self._dynamic_binding_kind(node.value)
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self._rebind_import_names({node.target.id}, binding_kind)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self._is_dynamic_importer(node.func)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.calls.append((node.args[0].value, node.lineno))
        self.generic_visit(node)

    def _is_dynamic_importer(self, node: ast.expr) -> bool:
        return (isinstance(node, ast.Name) and node.id in self.import_names) or (
            isinstance(node, ast.Attribute)
            and node.attr == "import_module"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.importlib_names
        )

    def _dynamic_binding_kind(self, node: ast.expr) -> str | None:
        if self._is_dynamic_importer(node):
            return "import_function"
        if isinstance(node, ast.Name) and node.id in self.importlib_names:
            return "importlib_module"
        return None

    def _rebind_import_names(self, names: set[str], binding_kind: str | None) -> None:
        self.import_names.difference_update(names)
        self.importlib_names.difference_update(names)
        if binding_kind == "import_function":
            self.import_names.update(names)
        elif binding_kind == "importlib_module":
            self.importlib_names.update(names)


def _dynamic_import_violations(
    importer: str,
    tree: ast.AST,
    governed_targets: set[str] | None = None,
) -> list[str]:
    collector = _DynamicImportCollector(importer)
    collector.visit(tree)
    return [
        f"{importer} dynamically imports {target}; governed architecture dependencies "
        f"must be static and top-level (production line {line})"
        for target, line in collector.calls
        if governed_targets is None
        or any(
            target == governed or target.startswith(governed + ".") for governed in governed_targets
        )
    ]


def _production_files() -> list[Path]:
    return sorted(
        path
        for path in (PROJECT_ROOT / "src" / "pixipix").rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _module(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT / "src").with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package(path: Path) -> str:
    module = _module(path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _edges() -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for path in _production_files():
        collector = _ImportCollector(_module(path), _package(path))
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        edges.extend(collector.edges)
    return edges


def _collect_source(importer: str, package: str, source: str) -> list[ImportEdge]:
    collector = _ImportCollector(importer, package)
    collector.visit(ast.parse(source))
    return collector.edges


def _target_modules(edge: ImportEdge, modules: set[str]) -> set[str]:
    targets = {edge.imported} if edge.imported in modules else set()
    targets.update(
        candidate
        for name in edge.names
        if name != "*"
        and (candidate := f"{edge.imported}.{name}" if edge.imported else name) in modules
    )
    return targets


def _stage_root(module: str) -> str | None:
    return next(
        (
            stage
            for stage in STAGE_IMPLEMENTATIONS
            if module == stage or module.startswith(stage + ".")
        ),
        None,
    )


def _cross_stage_edges(
    edges: list[ImportEdge],
    modules: set[str],
) -> list[tuple[ImportEdge, str]]:
    return [
        (edge, target)
        for edge in edges
        for target in _target_modules(edge, modules)
        if (importer_root := _stage_root(edge.importer)) is not None
        and (target_root := _stage_root(target)) is not None
        and importer_root != target_root
    ]


def _canonical_json_violations(module: str, tree: ast.AST) -> list[str]:
    json_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "json"
    }
    json_functions = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "json"
        for alias in node.names
        if alias.name in {"dump", "dumps"}
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in json_aliases
            and node.func.attr in {"dump", "dumps"}
        ):
            violations.append(
                f"canonical serialization: {module} calls json.{node.func.attr} "
                f"at production line {node.lineno}"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in json_functions
        ):
            violations.append(
                f"canonical serialization: {module} calls imported json.{node.func.id} "
                f"at production line {node.lineno}"
            )
    return violations


def _assert_no_hard_cycle(edges: list[ImportEdge], modules: set[str]) -> None:
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for edge in edges:
        if edge.hard:
            graph[edge.importer].update(_target_modules(edge, modules))

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            start = visiting.index(module)
            cycle = [*visiting[start:], module]
            raise AssertionError("hard import cycle: " + " -> ".join(cycle))
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(modules):
        visit(module)


def _failure(rule: str, edge: ImportEdge) -> str:
    names = f" symbols={edge.names}" if edge.names else ""
    return f"{rule}: {edge.importer} imports {edge.imported}{names} at production line {edge.line}"


def _channel_rounding_import_failure(reason: str, edge: ImportEdge) -> str:
    return f"{reason} {_failure('channel-rounding dependency', edge)}"


def _channel_rounding_calls(
    module: str,
    tree: ast.AST,
) -> list[tuple[str, str | None, int]]:
    calls: list[tuple[str, str | None, int]] = []
    function_stack: list[str] = []

    class CallCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == CHANNEL_ROUNDING_SYMBOL:
                calls.append(
                    (
                        module,
                        function_stack[-1] if function_stack else None,
                        node.lineno,
                    )
                )
            self.generic_visit(node)

    CallCollector().visit(tree)
    return calls


def _function_body_without_docstring(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.dump(
        ast.Module(body=body, type_ignores=[]),
        include_attributes=False,
    )


def _annotation_dump(annotation: ast.expr | None) -> str | None:
    return None if annotation is None else ast.dump(annotation, include_attributes=False)


def _channel_rounding_definition_contract(
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str | None, list[ast.expr], str]:
    return (
        ast.dump(definition.args, include_attributes=False),
        _annotation_dump(definition.returns),
        definition.decorator_list,
        _function_body_without_docstring(definition),
    )


def _channel_rounding_definitions(
    trees: dict[str, ast.Module],
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    return [
        (module, node)
        for module, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == CHANNEL_ROUNDING_SYMBOL
    ]


def _channel_rounding_exact_body_duplicates(
    trees: dict[str, ast.Module],
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, str, int]]:
    owner_body = _function_body_without_docstring(definition)
    return [
        (module, node.name, node.lineno)
        for module, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node is not definition
        and _function_body_without_docstring(node) == owner_body
    ]


def _assert_channel_rounding_call_contract(
    calls: list[tuple[str, str | None, int]],
) -> None:
    scale_calls = [call for call in calls if call[0].startswith("pixipix.stages.scale")]
    assert len(scale_calls) == 2 and all(
        module == "pixipix.stages.scale.execution" and function == "premultiplied_box_resize"
        for module, function, _line in scale_calls
    ), (
        "Channel-rounding Scale-internal invocation inventory changed. ADR-004 "
        f"records two calls in premultiplied_box_resize; found {scale_calls}"
    )

    non_scale_calls = [call for call in calls if not call[0].startswith("pixipix.stages.scale")]
    assert non_scale_calls, (
        "Channel-rounding ownership contract changed: the approved non-Scale "
        f"production call in {CHANNEL_ROUNDING_CONSUMER}."
        f"{CHANNEL_ROUNDING_CALL_SITE} is missing. Reopen ADR-004 before accepting "
        "the change."
    )
    consumer_modules = {module for module, _function, _line in non_scale_calls}
    assert consumer_modules == {CHANNEL_ROUNDING_CONSUMER}, (
        "Channel-rounding ownership contract changed: new production consumer "
        f"detected {sorted(consumer_modules)}. Reopen ADR-004 before expanding "
        "this dependency."
    )
    assert len(non_scale_calls) == 1, (
        "Channel-rounding ownership contract changed: new production call site "
        f"detected {non_scale_calls}. Reopen ADR-004 before adding another call."
    )
    call_module, call_owner, _call_line = non_scale_calls[0]
    assert (call_module, call_owner) == (
        CHANNEL_ROUNDING_CONSUMER,
        CHANNEL_ROUNDING_CALL_SITE,
    ), (
        "Channel-rounding call-site owner changed. ADR-004 requires the sole "
        "non-Scale call to remain in "
        f"{CHANNEL_ROUNDING_CONSUMER}.{CHANNEL_ROUNDING_CALL_SITE}; found "
        f"{call_module}.{call_owner}. Reconfirm or reopen ADR-004 before accepting "
        "the changed call-site owner."
    )


def _governed_dependency_violations(
    contract: StageDependencyContract,
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    violations: list[str] = []
    internal_symbols: dict[tuple[str, str], set[str]] = {}
    root_imports: set[tuple[str, str, tuple[str, ...]]] = set()
    for edge in edges:
        if edge.importer not in contract.modules:
            continue
        if (
            contract.allowed_external_imports is not None
            and not edge.imported.startswith("pixipix")
            and edge.imported not in contract.allowed_external_imports[edge.importer]
        ):
            violations.append(_failure(f"{contract.name} external capability", edge))
        if edge.imported == "pixipix":
            root_import = (edge.importer, edge.imported, edge.names)
            root_imports.add(root_import)
            if root_import not in contract.allowed_root_imports:
                violations.append(_failure(f"{contract.name} package-root capability", edge))
        for target in _target_modules(edge, modules):
            key = (edge.importer, target)
            allowed_symbols = contract.allowed_internal_symbols.get(key)
            if target in contract.modules:
                key = (edge.importer, target)
                internal_symbols.setdefault(key, set()).update(edge.names)
            if allowed_symbols is None:
                allowed_symbols = contract.allowed_shared_symbols.get(key)
            if allowed_symbols is not None:
                violations.extend(
                    _governed_import_form_violations(contract.name, edge, target, allowed_symbols)
                )
            elif target in contract.modules:
                violations.append(_failure(f"{contract.name} internal dependency direction", edge))
            if (
                target.startswith("pixipix")
                and target not in contract.allowed_pixipix_dependencies[edge.importer]
            ):
                violations.append(_failure(f"{contract.name} layer capability", edge))
    return violations, internal_symbols, root_imports


def _governed_import_form_violations(
    contract_name: str,
    edge: ImportEdge,
    target: str,
    allowed_symbols: set[str],
) -> list[str]:
    violations: list[str] = []
    if edge.imported != target:
        violations.append(
            f"{contract_name} governed dependency: {edge.importer} must import approved "
            f"symbols directly from {target}; package-root access conceals the exact capability"
        )
    if not edge.names:
        violations.append(
            f"{contract_name} governed dependency: {edge.importer} may import only approved "
            f"symbols from {target}, not the whole module"
        )
    elif "*" in edge.names or set(edge.names) - allowed_symbols:
        violations.append(
            _failure(f"{contract_name} exact governed-symbol capability for {target}", edge)
        )
    if edge.renamed:
        violations.append(
            f"{contract_name} governed dependency: {edge.importer} must import governed "
            f"symbols from {target} under their canonical names so the architecture edge "
            "remains directly auditable"
        )
    if edge.hidden:
        placement = "type-only" if edge.type_only else "runtime-hidden"
        violations.append(
            f"{contract_name} governed dependency: {edge.importer} must keep the governed "
            f"dependency on {target} as a top-level static import; {placement} placement "
            "conceals the architecture edge"
        )
    return violations


def _governed_capability_exactness_violations(
    contract: StageDependencyContract,
    edges: list[ImportEdge],
    modules: set[str],
) -> list[str]:
    declared = {
        key: set(symbols)
        for key, symbols in (
            *contract.allowed_internal_symbols.items(),
            *contract.allowed_shared_symbols.items(),
        )
    }
    live: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        if edge.importer not in contract.modules:
            continue
        for target in _target_modules(edge, modules):
            key = (edge.importer, target)
            if target in contract.modules or key in declared:
                live.setdefault(key, set()).update(edge.names)

    violations: list[str] = []
    for key in sorted(set(declared) | set(live)):
        importer, target = key
        permitted = declared.get(key, set())
        imported = live.get(key, set())
        for symbol in sorted(imported - permitted):
            violations.append(
                f"{contract.name} capability exactness: {importer} imports {symbol} from "
                f"{target}, but the governed contract does not permit it"
            )
        for symbol in sorted(permitted - imported):
            violations.append(
                f"{contract.name} capability exactness: {importer} permits {symbol} from "
                f"{target}, but no production governed import or documented compatibility "
                "allowance uses it. Remove the unused permission or document the exception"
            )
    return violations


def _scale_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    return _governed_dependency_violations(SCALE_DEPENDENCY_CONTRACT, edges, modules)


def _extract_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    result = _governed_dependency_violations(EXTRACT_DEPENDENCY_CONTRACT, edges, modules)
    violations, _internal_symbols, _root_imports = result
    for edge in edges:
        if edge.importer != "pixipix.stages.extract.api":
            continue
        for target in _target_modules(edge, modules):
            if target == "pixipix.stages.extract.publication":
                violations.append(
                    "Extract API must not depend on publication; the approved exception is "
                    "one-way publication → API via extract_source. "
                    + _failure("extract reverse dependency", edge)
                )
    return result


def _pixelize_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    result = _governed_dependency_violations(PIXELIZE_DEPENDENCY_CONTRACT, edges, modules)
    violations, _internal_symbols, _root_imports = result
    for edge in edges:
        if edge.importer not in PIXELIZE_MODULES:
            continue
        for target in _target_modules(edge, modules):
            if target == "pixipix.stages.scale":
                if edge.names != (CHANNEL_ROUNDING_SYMBOL,):
                    if edge.imported == "pixipix.stages" and edge.names == ("scale",):
                        reason = (
                            "Pixelize must import the approved channel-rounding symbol "
                            "from the Scale facade directly; package-root Scale access "
                            "conceals the exact capability."
                        )
                    elif edge.imported == "pixipix.stages.scale" and not edge.names:
                        reason = (
                            "Pixelize may not import the whole Scale module; the "
                            "approved dependency exposes one exact facade symbol."
                        )
                    elif edge.imported == "pixipix.stages.scale" and edge.names == ("*",):
                        reason = (
                            "Pixelize may not use a wildcard Scale import; the approved "
                            "dependency exposes only round_channel_half_away_from_zero."
                        )
                    else:
                        reason = (
                            "Pixelize may import only round_channel_half_away_from_zero "
                            "from the Scale compatibility facade."
                        )
                    violations.append(
                        _channel_rounding_import_failure(
                            reason,
                            edge,
                        )
                    )
                if edge.names == (CHANNEL_ROUNDING_SYMBOL,) and edge.renamed:
                    violations.append(
                        _channel_rounding_import_failure(
                            "Pixelize must import round_channel_half_away_from_zero "
                            "from the Scale facade under its canonical name so the "
                            "approved edge remains directly auditable.",
                            edge,
                        )
                    )
                if edge.names == (CHANNEL_ROUNDING_SYMBOL,) and edge.hidden:
                    violations.append(
                        _channel_rounding_import_failure(
                            "The approved Pixelize-to-Scale dependency must remain a "
                            "top-level static import; hidden or TYPE_CHECKING imports "
                            "conceal the architecture edge.",
                            edge,
                        )
                    )
            if target.startswith("pixipix.stages.scale."):
                violations.append(
                    _channel_rounding_import_failure(
                        "Pixelize may depend on the Scale compatibility facade, not "
                        "Scale's internal modules such as geometry.",
                        edge,
                    )
                )
    return result


def _align_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    return _governed_dependency_violations(ALIGN_DEPENDENCY_CONTRACT, edges, modules)


def _pipeline_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    return _governed_dependency_violations(PIPELINE_DEPENDENCY_CONTRACT, edges, modules)


def test_only_cli_imports_stage_command_publishers() -> None:
    modules = {_module(path) for path in _production_files()}
    facade_reexports = {
        (
            "pixipix.stages.extract",
            "pixipix.stages.extract.publication",
            ("publish_extraction",),
        ),
        (
            "pixipix.stages.align",
            "pixipix.stages.align.api",
            ("publish_align",),
        ),
        (
            "pixipix.stages.scale",
            "pixipix.stages.scale.api",
            ("publish_scale",),
        ),
        (
            "pixipix.stages.pixelize",
            "pixipix.stages.pixelize.api",
            ("publish_pixelize",),
        ),
    }
    violations = [
        edge
        for edge in _edges()
        if any(_stage_root(target) for target in _target_modules(edge, modules))
        and STAGE_PUBLISHERS.intersection(edge.names)
        and edge.importer != "pixipix.cli"
        and (edge.importer, edge.imported, edge.names) not in facade_reexports
    ]
    assert not violations, "\n".join(
        _failure("stage publisher orchestration", edge) for edge in violations
    )


def test_foundational_modules_do_not_import_stages() -> None:
    modules = {_module(path) for path in _production_files()}
    violations = [
        edge
        for edge in _edges()
        if edge.importer in FOUNDATIONAL_MODULES
        and any(
            target == "pixipix.stages" or target.startswith("pixipix.stages.")
            for target in _target_modules(edge, modules)
        )
    ]
    assert not violations, "\n".join(
        _failure("foundational dependency direction", edge) for edge in violations
    )


def test_foundational_modules_do_not_import_pipeline() -> None:
    modules = {_module(path) for path in _production_files()}
    violations = [
        edge
        for edge in _edges()
        if edge.importer in FOUNDATIONAL_MODULES
        and any(target in PIPELINE_MODULES for target in _target_modules(edge, modules))
    ]
    assert not violations, "\n".join(
        _failure("foundational pipeline dependency direction", edge) for edge in violations
    )


def test_stages_io_does_not_import_stage_implementations() -> None:
    modules = {_module(path) for path in _production_files()}
    violations = [
        edge
        for edge in _edges()
        if edge.importer == "pixipix.stages.io"
        and any(_stage_root(target) for target in _target_modules(edge, modules))
    ]
    assert not violations, "\n".join(
        _failure("shared stage I/O direction", edge) for edge in violations
    )


def test_pipeline_module_set_and_dependency_direction_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    actual_modules = {
        module
        for module in modules
        if module == "pixipix.pipeline" or module.startswith("pixipix.pipeline.")
    }
    assert actual_modules == PIPELINE_MODULES, (
        "pipeline module set differs: "
        f"missing={sorted(PIPELINE_MODULES - actual_modules)}, "
        f"unexpected={sorted(actual_modules - PIPELINE_MODULES)}"
    )

    violations, internal_symbols, root_imports = _pipeline_dependency_violations(
        _edges(),
        modules,
    )
    assert not violations, "\n".join(violations)
    assert not root_imports
    assert internal_symbols == PIPELINE_ALLOWED_INTERNAL_SYMBOLS, (
        "pipeline internal dependency graph differs: "
        f"missing={sorted(set(PIPELINE_ALLOWED_INTERNAL_SYMBOLS) - set(internal_symbols))}, "
        f"unexpected={sorted(set(internal_symbols) - set(PIPELINE_ALLOWED_INTERNAL_SYMBOLS))}, "
        f"symbols={internal_symbols}"
    )


def test_verified_platform_alias_policy_has_one_owner_and_no_blanket_resolution(
    tmp_path: Path,
) -> None:
    owner = PROJECT_ROOT / "src" / "pixipix" / "pipeline" / "artifacts.py"
    production = _production_files()
    owner_module = importlib.import_module("pixipix.pipeline.artifacts")
    consumers = tuple(
        importlib.import_module(module)
        for module in ("pixipix.pipeline.input", "pixipix.pipeline.publication")
    )
    owner_functions = {
        value
        for value in vars(owner_module).values()
        if inspect.isfunction(value) and value.__module__ == owner_module.__name__
    }
    shared_functions = set.intersection(
        owner_functions,
        *(
            {
                value
                for value in vars(module).values()
                if inspect.isfunction(value) and value in owner_functions
            }
            for module in consumers
        ),
    )
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    redirect = tmp_path / "redirect"
    redirect.symlink_to(target, target_is_directory=True)
    escape = tmp_path / "safe" / ".." / "escape"

    def behaves_as_redirect_classifier(runtime_function: object) -> bool:
        if not inspect.isfunction(runtime_function):
            return False
        runtime_callable = cast(Callable[[Path], object], runtime_function)
        signature = inspect.signature(runtime_callable)
        parameter = next(iter(signature.parameters.values()))
        return bool(
            len(signature.parameters) == 1
            and parameter.annotation in {Path, "Path"}
            and signature.return_annotation in {bool, "bool"}
            and runtime_callable(ordinary) is False
            and runtime_callable(redirect) is True
            and runtime_callable(escape) is True
        )

    classifiers: list[Callable[[Path], bool]] = []
    for runtime_function in shared_functions:
        if behaves_as_redirect_classifier(runtime_function):
            classifiers.append(runtime_function)
    assert len(classifiers) == 1
    classifier = classifiers[0]
    for module in consumers:
        local_classifiers = [
            value
            for value in vars(module).values()
            if inspect.isfunction(value)
            and value.__module__ == module.__name__
            and behaves_as_redirect_classifier(value)
        ]
        assert not local_classifiers, (
            f"{module.__name__} defines a second execution-effective redirect classifier"
        )

    expected_aliases = {
        Path("/tmp"): (Path("private/tmp"), Path("/private/tmp")),
        Path("/var"): (Path("private/var"), Path("/private/var")),
    }
    alias_authorities = [
        value
        for value in vars(owner_module).values()
        if isinstance(value, dict)
        and any(key in expected_aliases for key in value)
        and all(
            isinstance(key, Path)
            and isinstance(targets, tuple)
            and len(targets) == 2
            and all(isinstance(target_path, Path) for target_path in targets)
            for key, targets in value.items()
        )
    ]
    assert alias_authorities == [expected_aliases]

    tree = ast.parse(owner.read_text(encoding="utf-8"), filename=str(owner))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reachable = {classifier.__name__}
    pending = [classifier.__name__]
    while pending:
        function_node = functions[pending.pop()]
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_name = node.func.id
                if called_name in functions and called_name not in reachable:
                    reachable.add(called_name)
                    pending.append(called_name)
    forbidden_calls = {
        call.func.attr
        for function_name in reachable
        for call in ast.walk(functions[function_name])
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"resolve", "realpath", "startswith"}
    }
    assert not forbidden_calls

    alias_policy_literals = {
        "/var",
        "/tmp",
        "private/var",
        "private/tmp",
        "/private/var",
        "/private/tmp",
    }
    for path in production:
        if path == owner:
            continue
        candidate_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        duplicated_facts = {
            node.value
            for node in ast.walk(candidate_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in alias_policy_literals
        }
        assert not duplicated_facts, f"{path} duplicates alias policy facts: {duplicated_facts}"


def test_governed_modules_do_not_dynamically_import_architecture_capabilities() -> None:
    violations: list[str] = []
    for path in _production_files():
        importer = _module(path)
        if importer in GOVERNED_PRODUCTION_MODULES:
            violations.extend(
                _dynamic_import_violations(
                    importer,
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
                    GOVERNED_DYNAMIC_TARGETS,
                )
            )
    assert not violations, "\n".join(violations)


def test_governed_capability_maps_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    edges = _edges()
    violations = [
        violation
        for contract in (*STAGE_DEPENDENCY_CONTRACTS, PIPELINE_DEPENDENCY_CONTRACT)
        for violation in _governed_capability_exactness_violations(contract, edges, modules)
    ]
    assert not violations, "\n".join(violations)


def test_governed_capability_exactness_rejects_missing_and_unused_permissions() -> None:
    modules = {_module(path) for path in _production_files()}
    edges = _edges()
    for contract in (*STAGE_DEPENDENCY_CONTRACTS, PIPELINE_DEPENDENCY_CONTRACT):
        governed = {
            **contract.allowed_internal_symbols,
            **contract.allowed_shared_symbols,
        }
        key = sorted(governed)[0]
        symbol = sorted(governed[key])[0]

        governed[key].remove(symbol)
        try:
            missing_violations = _governed_capability_exactness_violations(
                contract,
                edges,
                modules,
            )
        finally:
            governed[key].add(symbol)
        assert any(
            f"imports {symbol} from {key[1]}" in violation and "does not permit it" in violation
            for violation in missing_violations
        ), f"{contract.name} accepted removal of live permission {key!r} {symbol!r}"

        unused_symbol = "__slice_9_unused_capability"
        governed[key].add(unused_symbol)
        try:
            unused_violations = _governed_capability_exactness_violations(
                contract,
                edges,
                modules,
            )
        finally:
            governed[key].remove(unused_symbol)
        assert any(
            f"permits {unused_symbol} from {key[1]}" in violation
            and "no production governed import" in violation
            for violation in unused_violations
        ), f"{contract.name} accepted unused permission {key!r} {unused_symbol!r}"


def test_stage_implementations_do_not_import_stages_io_facade() -> None:
    violations = [
        edge
        for edge in _edges()
        if _stage_root(edge.importer) is not None and edge.imported == "pixipix.stages.io"
    ]
    assert not violations, "\n".join(
        _failure("stage implementation compatibility-facade dependency", edge)
        for edge in violations
    )


def test_extract_internal_module_set_and_dependency_direction_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    actual_modules = {
        module
        for module in modules
        if module == "pixipix.stages.extract" or module.startswith("pixipix.stages.extract.")
    }
    assert actual_modules == EXTRACT_MODULES, (
        "extract internal module set differs: "
        f"missing={sorted(EXTRACT_MODULES - actual_modules)}, "
        f"unexpected={sorted(actual_modules - EXTRACT_MODULES)}"
    )
    assert not (PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract.py").exists()

    violations, internal_symbols, root_imports = _extract_dependency_violations(
        _edges(),
        modules,
    )

    assert not violations, "\n".join(violations)
    assert root_imports == EXTRACT_ALLOWED_ROOT_IMPORTS
    assert internal_symbols == EXTRACT_ALLOWED_INTERNAL_SYMBOLS, (
        "extract internal dependency graph differs: "
        f"missing={sorted(set(EXTRACT_ALLOWED_INTERNAL_SYMBOLS) - set(internal_symbols))}, "
        f"unexpected={sorted(set(internal_symbols) - set(EXTRACT_ALLOWED_INTERNAL_SYMBOLS))}, "
        f"symbols={internal_symbols}"
    )
    assert (
        "pixipix.stages.extract.publication",
        "pixipix.stages.extract.api",
    ) in internal_symbols, (
        "Extract-specific approved exception missing: publication must import "
        "api.extract_source; this does not authorize api -> publication"
    )


def test_extract_publication_is_only_a_shared_lifecycle_adapter() -> None:
    path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract" / "publication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert [function.name for function in functions] == ["publish_extraction"]
    calls = [
        call.func.id
        for call in ast.walk(functions[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]
    assert calls == [
        "validate_stage_output_target",
        "extract_source",
        "_stage_metadata",
        "tuple",
        "publish_stage_output",
        "OutputFrameImage",
    ]
    forbidden_generic_owners = {
        "_valid_marker",
        "_valid_owned_output",
        "_valid_frame_png",
        "_validate_output_location",
        "_validate_staged_output",
        "_prepare_target",
        "_remove_temporary_tree",
    }
    assert forbidden_generic_owners.isdisjoint(
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )


def test_extract_dependency_rule_rejects_layer_and_module_bypasses() -> None:
    modules = {"pixipix", *EXTRACT_MODULES}
    cases = (
        (
            "pixipix.stages.extract.api",
            "pixipix.stages.extract",
            "from .publication import publish_extraction",
        ),
        (
            "pixipix.stages.extract.api",
            "pixipix.stages.extract",
            "from .publication import publish_extraction as publish",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "from .api import extract_source as run_extract",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "def load():\n    from .api import extract_source",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "if True:\n    from .api import extract_source",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "try:\n    from .api import extract_source\nexcept ImportError:\n    pass",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from .api import extract_source",
        ),
        (
            "pixipix.stages.extract.api",
            "pixipix.stages.extract",
            "def load():\n    from .publication import publish_extraction",
        ),
        (
            "pixipix.stages.extract.api",
            "pixipix.stages.extract",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from .publication import publish_extraction",
        ),
        (
            "pixipix.stages.extract.metadata",
            "pixipix.stages.extract",
            "from .api import extract_source",
        ),
        (
            "pixipix.stages.extract.publication",
            "pixipix.stages.extract",
            "from .analysis import _Analysis",
        ),
        (
            "pixipix.stages.extract.planning",
            "pixipix.stages.extract",
            "import pixipix.stages.extract.analysis as analysis",
        ),
        (
            "pixipix.stages.extract.planning",
            "pixipix.stages.extract",
            "from .analysis import *",
        ),
        (
            "pixipix.stages.extract.api",
            "pixipix.stages.extract",
            "from pixipix.stages.extract import ComponentMap",
        ),
        (
            "pixipix.stages.extract.metadata",
            "pixipix.stages.extract",
            "from pixipix import cli",
        ),
    )
    for importer, package, source in cases:
        violations, _symbols, _root_imports = _extract_dependency_violations(
            _collect_source(importer, package, source),
            modules,
        )
        assert violations, f"extract dependency bypass was accepted: {source}"


def test_extract_dependency_rule_rejects_dynamic_imports() -> None:
    cases = (
        "__import__('pixipix.stages.extract.analysis')",
        "load = __import__\nload('pixipix.stages.extract.analysis')",
        "import importlib\nimportlib.import_module('pixipix.stages.extract.analysis')",
        "from importlib import import_module as load\nload('pixipix.stages.extract.analysis')",
    )
    for source in cases:
        violations = _dynamic_import_violations(
            "pixipix.stages.extract.api",
            ast.parse(source),
        )
        assert violations, f"extract dynamic import bypass was accepted: {source}"


def test_scale_internal_module_set_and_dependency_direction_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    actual_modules = {
        module
        for module in modules
        if module == "pixipix.stages.scale" or module.startswith("pixipix.stages.scale.")
    }
    assert actual_modules == SCALE_MODULES, (
        "scale internal module set differs: "
        f"missing={sorted(SCALE_MODULES - actual_modules)}, "
        f"unexpected={sorted(actual_modules - SCALE_MODULES)}"
    )
    assert not (PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale.py").exists()

    violations, internal_symbols, root_imports = _scale_dependency_violations(
        _edges(),
        modules,
    )

    assert not violations, "\n".join(violations)
    assert root_imports == SCALE_ALLOWED_ROOT_IMPORTS
    assert internal_symbols == SCALE_ALLOWED_INTERNAL_SYMBOLS, (
        "scale internal dependency graph differs: "
        f"missing={sorted(set(SCALE_ALLOWED_INTERNAL_SYMBOLS) - set(internal_symbols))}, "
        f"unexpected={sorted(set(internal_symbols) - set(SCALE_ALLOWED_INTERNAL_SYMBOLS))}, "
        f"symbols={internal_symbols}"
    )


def test_scale_api_orchestration_order_is_exact() -> None:
    path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale" / "api.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "publish_scale"
    )
    calls = [
        call.func.id
        for statement in function.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]
    assert calls == [
        "validate_stage_output_target",
        "validate_stage_input",
        "project_scale_stage",
        "enforce_resource_policy",
        "decode_stage_input",
        "scale_stage",
        "publish_stage_output",
    ]


def test_pixelize_internal_module_set_and_dependency_direction_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    actual_modules = {
        module
        for module in modules
        if module == "pixipix.stages.pixelize" or module.startswith("pixipix.stages.pixelize.")
    }
    assert actual_modules == PIXELIZE_MODULES, (
        "pixelize internal module set differs: "
        f"missing={sorted(PIXELIZE_MODULES - actual_modules)}, "
        f"unexpected={sorted(actual_modules - PIXELIZE_MODULES)}"
    )
    assert not (PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize.py").exists()

    violations, internal_symbols, root_imports = _pixelize_dependency_violations(
        _edges(),
        modules,
    )

    assert not violations, "\n".join(violations)
    assert root_imports == PIXELIZE_ALLOWED_ROOT_IMPORTS
    assert internal_symbols == PIXELIZE_ALLOWED_INTERNAL_SYMBOLS, (
        "pixelize internal dependency graph differs: "
        f"missing={sorted(set(PIXELIZE_ALLOWED_INTERNAL_SYMBOLS) - set(internal_symbols))}, "
        f"unexpected={sorted(set(internal_symbols) - set(PIXELIZE_ALLOWED_INTERNAL_SYMBOLS))}, "
        f"symbols={internal_symbols}"
    )


def test_pixelize_api_orchestration_order_is_exact() -> None:
    path = PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize" / "api.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "publish_pixelize"
    )
    calls = [
        call.func.id
        for statement in function.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]
    assert calls == [
        "validate_stage_output_target",
        "validate_stage_input",
        "project_pixelize_stage",
        "enforce_resource_policy",
        "decode_stage_input",
        "pixelize_stage",
        "publish_stage_output",
    ]


def test_scale_dependency_rule_rejects_module_and_root_import_bypasses() -> None:
    modules = {"pixipix", *SCALE_MODULES}
    cases = (
        (
            "pixipix.stages.scale.execution",
            "pixipix.stages.scale",
            "import pixipix.stages.scale.planning as planning",
        ),
        (
            "pixipix.stages.scale.execution",
            "pixipix.stages.scale",
            "from .planning import project_scale_stage as planner",
        ),
        (
            "pixipix.stages.scale.execution",
            "pixipix.stages.scale",
            "import pixipix.stages.scale.geometry as geometry",
        ),
        (
            "pixipix.stages.scale.metadata",
            "pixipix.stages.scale",
            "from pixipix import cli",
        ),
        (
            "pixipix.stages.scale.metadata",
            "pixipix.stages.scale",
            "import pixipix",
        ),
    )
    for importer, package, source in cases:
        violations, _symbols, _root_imports = _scale_dependency_violations(
            _collect_source(importer, package, source),
            modules,
        )
        assert violations, f"scale dependency bypass was accepted: {source}"


def test_pixelize_dependency_rule_rejects_module_root_and_scale_bypasses() -> None:
    modules = {"pixipix", *PIXELIZE_MODULES, *SCALE_MODULES}
    cases = (
        (
            "pixipix.stages.pixelize.execution",
            "pixipix.stages.pixelize",
            "import pixipix.stages.pixelize.planning as planning",
        ),
        (
            "pixipix.stages.pixelize.execution",
            "pixipix.stages.pixelize",
            "from .planning import project_pixelize_stage as planner",
        ),
        (
            "pixipix.stages.pixelize.metadata",
            "pixipix.stages.pixelize",
            "from pixipix import cli",
        ),
        (
            "pixipix.stages.pixelize.metadata",
            "pixipix.stages.pixelize",
            "import pixipix",
        ),
        (
            "pixipix.stages.pixelize.execution",
            "pixipix.stages.pixelize",
            "import pixipix.stages.scale as scale",
        ),
        (
            "pixipix.stages.pixelize.execution",
            "pixipix.stages.pixelize",
            "from pixipix.stages.scale.geometry import round_channel_half_away_from_zero",
        ),
        (
            "pixipix.stages.pixelize.execution",
            "pixipix.stages.pixelize",
            "from pixipix.stages.scale import publish_scale",
        ),
    )
    for importer, package, source in cases:
        violations, _symbols, _root_imports = _pixelize_dependency_violations(
            _collect_source(importer, package, source),
            modules,
        )
        assert violations, f"pixelize dependency bypass was accepted: {source}"


def test_channel_rounding_ownership_contract_is_exact() -> None:
    calls: list[tuple[str, str | None, int]] = []
    trees: dict[str, ast.Module] = {}
    for path in _production_files():
        module = _module(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[module] = tree
        calls.extend(_channel_rounding_calls(module, tree))

    definitions = _channel_rounding_definitions(trees)
    assert len(definitions) == 1, (
        "Channel-rounding ownership changed. ADR-004 requires exactly one "
        f"{CHANNEL_ROUNDING_SYMBOL} implementation; found "
        f"{[(module, node.lineno) for module, node in definitions]}"
    )
    owner, definition = definitions[0]
    assert owner == CHANNEL_ROUNDING_OWNER, (
        "Channel-rounding ownership changed. ADR-004 requires one implementation "
        f"owned by {CHANNEL_ROUNDING_OWNER}; found {owner}"
    )

    expected_definition = ast.parse(CHANNEL_ROUNDING_FUNCTION_SOURCE).body[0]
    assert isinstance(expected_definition, ast.FunctionDef)
    assert _channel_rounding_definition_contract(
        definition
    ) == _channel_rounding_definition_contract(expected_definition), (
        "Channel-rounding semantic scope changed. ADR-004 requires finite float "
        "quantization with half-away rounding and [0,255] clamping; reopen ADR-004 "
        "before accepting a semantic expansion."
    )

    exact_body_duplicates = _channel_rounding_exact_body_duplicates(
        trees,
        definition,
    )
    assert not exact_body_duplicates, (
        "Duplicate channel-rounding implementation detected. ADR-004 requires one "
        f"Scale-owned implementation; found exact-body duplicates {exact_body_duplicates}"
    )

    _assert_channel_rounding_call_contract(calls)


@pytest.mark.parametrize(
    ("source", "expected_count"),
    [
        (
            "def round_channel_half_away_from_zero(value):\n    return value",
            1,
        ),
        (
            "def outer():\n    def round_channel_half_away_from_zero(value):\n        return value",
            1,
        ),
        (
            "class Quantizer:\n"
            "    def round_channel_half_away_from_zero(self, value):\n"
            "        return value",
            1,
        ),
        (
            "async def round_channel_half_away_from_zero(value):\n    return value",
            1,
        ),
        ("round_channel_half_away_from_zero = lambda value: value", 0),
        ("round_channel_half_away_from_zero = another_binding", 0),
        (
            "from owner import helper as round_channel_half_away_from_zero",
            0,
        ),
    ],
)
def test_channel_rounding_definition_inventory_distinguishes_implementations(
    source: str,
    expected_count: int,
) -> None:
    definitions = _channel_rounding_definitions({"pixipix.stages.fixture": ast.parse(source)})
    assert len(definitions) == expected_count


@pytest.mark.parametrize(
    ("source", "expected_duplicate"),
    [
        (
            CHANNEL_ROUNDING_FUNCTION_SOURCE.replace(
                "round_channel_half_away_from_zero",
                "copied_rounding",
                1,
            ),
            True,
        ),
        (
            CHANNEL_ROUNDING_FUNCTION_SOURCE.replace(
                "def round_channel_half_away_from_zero(value: float) -> int:\n",
                'def copied_rounding(value: float) -> int:\n    """Different documentation."""\n',
            ),
            True,
        ),
        (
            "\n"
            + CHANNEL_ROUNDING_FUNCTION_SOURCE.replace(
                "round_channel_half_away_from_zero",
                "copied_rounding",
                1,
            ).replace("value: float", "value : float")
            + "\n",
            True,
        ),
        (
            "def copied_rounding(value: float) -> int:\n"
            "    if not math.isfinite(value):\n"
            '        raise ValueError("channel value must be finite")\n'
            "    magnitude = math.floor(abs(value) + 0.5)\n"
            "    rounded = -magnitude if value < 0 else magnitude\n"
            "    return max(0, min(255, rounded))",
            False,
        ),
        (
            "def clamp_index(value: float) -> int:\n    return max(0, min(255, int(value)))",
            False,
        ),
        (
            "def outer():\n"
            "    def copied_rounding(value: float) -> int:\n"
            "        if not math.isfinite(value):\n"
            '            raise ValueError("channel value must be finite")\n'
            "        magnitude = math.floor(abs(value) + 0.5)\n"
            "        rounded = magnitude if value >= 0 else -magnitude\n"
            "        return min(255, max(0, rounded))",
            True,
        ),
    ],
)
def test_channel_rounding_duplicate_inventory_has_exact_body_scope(
    source: str,
    expected_duplicate: bool,
) -> None:
    owner_tree = ast.parse(CHANNEL_ROUNDING_FUNCTION_SOURCE)
    owner = owner_tree.body[0]
    assert isinstance(owner, ast.FunctionDef)
    duplicates = _channel_rounding_exact_body_duplicates(
        {
            CHANNEL_ROUNDING_OWNER: owner_tree,
            "pixipix.stages.fixture": ast.parse(source),
        },
        owner,
    )
    assert bool(duplicates) is expected_duplicate


@pytest.mark.parametrize(
    "module",
    [
        "pixipix.stages.scale",
        "pixipix.stages.scale.api",
        "pixipix.stages.scale.execution",
        "pixipix.stages.scale.geometry",
        "pixipix.stages.scale.metadata",
        "pixipix.stages.scale.planning",
        "pixipix.stages.scale.future_sibling",
    ],
)
def test_channel_rounding_call_inventory_scans_every_scale_module(
    module: str,
) -> None:
    calls = _channel_rounding_calls(
        module,
        ast.parse(
            "def outer():\n    def nested():\n        round_channel_half_away_from_zero(1.0)"
        ),
    )
    assert calls == [(module, "nested", 3)]


@pytest.mark.parametrize(
    ("non_scale_calls", "expected_message"),
    [
        ([], "approved non-Scale production call"),
        (
            [
                (
                    CHANNEL_ROUNDING_CONSUMER,
                    CHANNEL_ROUNDING_CALL_SITE,
                    10,
                ),
                (
                    CHANNEL_ROUNDING_CONSUMER,
                    CHANNEL_ROUNDING_CALL_SITE,
                    11,
                ),
            ],
            "new production call site",
        ),
        (
            [(CHANNEL_ROUNDING_CONSUMER, "other_function", 10)],
            "call-site owner changed",
        ),
        (
            [("pixipix.stages.pixelize.planning", "other_function", 10)],
            "new production consumer",
        ),
        (
            [(CHANNEL_ROUNDING_CONSUMER, "_weighted_alpha_majority", 10)],
            "call-site owner changed",
        ),
    ],
)
def test_channel_rounding_call_contract_failures_are_specific(
    non_scale_calls: list[tuple[str, str | None, int]],
    expected_message: str,
) -> None:
    scale_calls = [
        (
            "pixipix.stages.scale.execution",
            "premultiplied_box_resize",
            1,
        ),
        (
            "pixipix.stages.scale.execution",
            "premultiplied_box_resize",
            2,
        ),
    ]
    with pytest.raises(AssertionError, match=expected_message):
        _assert_channel_rounding_call_contract([*scale_calls, *non_scale_calls])


def test_channel_rounding_import_contract_accepts_canonical_facade_symbol() -> None:
    modules = {"pixipix", *PIXELIZE_MODULES, *SCALE_MODULES}
    edges = _collect_source(
        CHANNEL_ROUNDING_CONSUMER,
        "pixipix.stages.pixelize",
        "from pixipix.stages.scale import round_channel_half_away_from_zero",
    )
    violations, _symbols, _root_imports = _pixelize_dependency_violations(
        edges,
        modules,
    )
    assert not violations, "\n".join(violations)
    assert len(edges) == 1
    edge = edges[0]
    assert (
        edge.imported,
        edge.names,
        edge.renamed,
        edge.hidden,
        edge.hard,
    ) == (
        "pixipix.stages.scale",
        (CHANNEL_ROUNDING_SYMBOL,),
        False,
        False,
        True,
    )


@pytest.mark.parametrize(
    ("source", "expected_message"),
    [
        (
            "from pixipix.stages.scale import round_channel_half_away_from_zero as round_channel",
            "under its canonical name",
        ),
        (
            "from pixipix.stages.scale.geometry import round_channel_half_away_from_zero",
            "not Scale's internal modules",
        ),
        (
            "from pixipix.stages.scale.geometry import "
            "round_channel_half_away_from_zero as round_channel",
            "not Scale's internal modules",
        ),
        (
            "import pixipix.stages.scale",
            "may not import the whole Scale module",
        ),
        (
            "import pixipix.stages.scale as scale",
            "may not import the whole Scale module",
        ),
        (
            "from pixipix.stages import scale",
            "package-root Scale access",
        ),
        (
            "from pixipix.stages.scale import *",
            "may not use a wildcard Scale import",
        ),
        (
            "def hidden():\n    from pixipix.stages.scale import round_channel_half_away_from_zero",
            "top-level static import",
        ),
        (
            "def outer():\n"
            "    def inner():\n"
            "        from pixipix.stages.scale import "
            "round_channel_half_away_from_zero",
            "top-level static import",
        ),
        (
            "if condition:\n    from pixipix.stages.scale import round_channel_half_away_from_zero",
            "top-level static import",
        ),
        (
            "try:\n"
            "    from pixipix.stages.scale import "
            "round_channel_half_away_from_zero\n"
            "except ImportError:\n"
            "    pass",
            "top-level static import",
        ),
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from pixipix.stages.scale import "
            "round_channel_half_away_from_zero",
            "top-level static import",
        ),
        (
            "from typing import TYPE_CHECKING as TC\n"
            "if TC:\n"
            "    from pixipix.stages.scale import "
            "round_channel_half_away_from_zero",
            "top-level static import",
        ),
    ],
)
def test_channel_rounding_import_contract_rejects_static_bypasses(
    source: str,
    expected_message: str,
) -> None:
    modules = {"pixipix", *PIXELIZE_MODULES, *SCALE_MODULES}
    violations, _symbols, _root_imports = _pixelize_dependency_violations(
        _collect_source(
            CHANNEL_ROUNDING_CONSUMER,
            "pixipix.stages.pixelize",
            source,
        ),
        modules,
    )
    assert any(expected_message in violation for violation in violations), (
        "channel-rounding import bypass did not fail for the intended reason: "
        f"source={source!r}, violations={violations}"
    )


def test_pixelize_dependency_rule_rejects_dynamic_imports() -> None:
    cases = (
        "__import__('pixipix.stages.scale')",
        "alias = __import__\nalias('pixipix.stages.scale')",
        "import importlib as il\nil.import_module('pixipix.stages.scale')",
        "from importlib import import_module as load\nload('pixipix.stages.scale')",
    )
    for source in cases:
        violations = _dynamic_import_violations(
            "pixipix.stages.pixelize.execution",
            ast.parse(source),
        )
        assert violations, f"pixelize dynamic import bypass was accepted: {source}"


@pytest.mark.parametrize(
    ("source", "dynamic", "approved"),
    [
        ("from .planning import project_align_stage", False, True),
        ("import pixipix.stages.align.planning", False, False),
        ("from .planning import *", False, False),
        ("from .planning import project_align_stage as project", False, False),
        ("def load():\n    from .planning import project_align_stage", False, False),
        ("__import__('pixipix.stages.align.planning')", True, False),
    ],
)
def test_align_generalized_governed_import_forms(
    source: str,
    dynamic: bool,
    approved: bool,
) -> None:
    importer = "pixipix.stages.align.api"
    if dynamic:
        violations = _dynamic_import_violations(
            importer,
            ast.parse(source),
            GOVERNED_DYNAMIC_TARGETS,
        )
    else:
        violations, _symbols, _roots = _align_dependency_violations(
            _collect_source(importer, "pixipix.stages.align", source),
            {_module(path) for path in _production_files()},
        )
    assert bool(violations) is not approved


@pytest.mark.parametrize(
    ("source", "dynamic"),
    [
        ("from .planning import project_scale_stage as project", False),
        ("if enabled:\n    from .planning import project_scale_stage", False),
        ("import importlib as il\nil.import_module('pixipix.stages.scale.planning')", True),
        ("__import__('pixipix.stages.scale')", True),
    ],
)
def test_scale_generalized_governed_import_forms(source: str, dynamic: bool) -> None:
    importer = "pixipix.stages.scale.api"
    if dynamic:
        violations = _dynamic_import_violations(
            importer,
            ast.parse(source),
            GOVERNED_DYNAMIC_TARGETS,
        )
    else:
        violations, _symbols, _roots = _scale_dependency_violations(
            _collect_source(importer, "pixipix.stages.scale", source),
            {_module(path) for path in _production_files()},
        )
    assert violations, f"Scale governed import bypass was accepted: {source}"


@pytest.mark.parametrize(
    ("source", "dynamic"),
    [
        ("from .planning import project_pixelize_stage as project", False),
        (
            "try:\n    from .planning import project_pixelize_stage\nexcept ImportError:\n    pass",
            False,
        ),
        ("__import__('pixipix.stages.pixelize.planning')", True),
    ],
)
def test_pixelize_generalized_internal_import_forms(source: str, dynamic: bool) -> None:
    importer = "pixipix.stages.pixelize.api"
    if dynamic:
        violations = _dynamic_import_violations(
            importer,
            ast.parse(source),
            GOVERNED_DYNAMIC_TARGETS,
        )
    else:
        violations, _symbols, _roots = _pixelize_dependency_violations(
            _collect_source(importer, "pixipix.stages.pixelize", source),
            {_module(path) for path in _production_files()},
        )
    assert violations, f"Pixelize governed import bypass was accepted: {source}"


@pytest.mark.parametrize(
    ("importer", "package", "source", "dynamic"),
    [
        (
            "pixipix.stages.align.api",
            "pixipix.stages.align",
            "from pixipix.pipeline.input import validate_stage_input as validate",
            False,
        ),
        (
            "pixipix.stages.scale.api",
            "pixipix.stages.scale",
            "def load():\n    from pixipix.resources import enforce_resource_policy",
            False,
        ),
        (
            "pixipix.stages.pixelize.api",
            "pixipix.stages.pixelize",
            "__import__('pixipix.pipeline.publication')",
            True,
        ),
        (
            "pixipix.pipeline.input",
            "pixipix.pipeline",
            "from importlib import import_module as load\nload('pixipix.pipeline.artifacts')",
            True,
        ),
    ],
)
def test_shared_pipeline_generalized_import_forms(
    importer: str,
    package: str,
    source: str,
    dynamic: bool,
) -> None:
    if dynamic:
        violations = _dynamic_import_violations(
            importer,
            ast.parse(source),
            GOVERNED_DYNAMIC_TARGETS,
        )
    else:
        contract = next(
            contract for contract in STAGE_DEPENDENCY_CONTRACTS if importer in contract.modules
        )
        violations, _symbols, _roots = _governed_dependency_violations(
            contract,
            _collect_source(importer, package, source),
            {_module(path) for path in _production_files()},
        )
    assert violations, f"shared pipeline governed import bypass was accepted: {source}"


@pytest.mark.parametrize(
    "source",
    [
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .planning import project_align_stage",
        "from typing import TYPE_CHECKING as CHECK_TYPES\n"
        "if CHECK_TYPES:\n"
        "    if nested:\n"
        "        from .planning import project_align_stage",
    ],
)
def test_type_checking_bindings_are_hidden_but_not_hard(source: str) -> None:
    edges = _collect_source(
        "pixipix.stages.align.api",
        "pixipix.stages.align",
        source,
    )
    edge = next(edge for edge in edges if edge.imported == "pixipix.stages.align.planning")
    assert edge.hidden is True
    assert edge.hard is False
    assert edge.type_only is True
    violations, _symbols, _roots = _align_dependency_violations(
        edges,
        {_module(path) for path in _production_files()},
    )
    assert any("type-only placement" in violation for violation in violations)

    module_alias_edges = _collect_source(
        "pixipix.stages.align.api",
        "pixipix.stages.align",
        "import typing as type_hints\n"
        "if type_hints.TYPE_CHECKING:\n"
        "    from .planning import project_align_stage",
    )
    module_alias_edge = next(
        edge for edge in module_alias_edges if edge.imported == "pixipix.stages.align.planning"
    )
    assert (module_alias_edge.hidden, module_alias_edge.hard) == (True, False)

    _assert_no_hard_cycle(
        [
            edge,
            *_collect_source(
                "pixipix.stages.align.planning",
                "pixipix.stages.align",
                "from .api import publish_align",
            ),
        ],
        {"pixipix.stages.align.api", "pixipix.stages.align.planning"},
    )


@pytest.mark.parametrize(
    ("source", "expected_hard"),
    [
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING and condition:\n"
            "    from .planning import project_align_stage",
            False,
        ),
        (
            "from typing import TYPE_CHECKING\n"
            "if condition and TYPE_CHECKING:\n"
            "    from .planning import project_align_stage",
            False,
        ),
        (
            "from typing import TYPE_CHECKING as TC\n"
            "if TC and condition:\n"
            "    from .planning import project_align_stage",
            False,
        ),
        (
            "import typing as type_hints\n"
            "if type_hints.TYPE_CHECKING and condition:\n"
            "    from .planning import project_align_stage",
            False,
        ),
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING or condition:\n"
            "    from .planning import project_align_stage",
            True,
        ),
        (
            "from typing import TYPE_CHECKING\n"
            "if not TYPE_CHECKING:\n"
            "    from .planning import project_align_stage",
            True,
        ),
        (
            "from typing import TYPE_CHECKING as TC\n"
            "TC = True\n"
            "if TC:\n"
            "    from .planning import project_align_stage",
            True,
        ),
        (
            "from typing import TYPE_CHECKING as TC\n"
            "TC: bool = True\n"
            "if TC:\n"
            "    from .planning import project_align_stage",
            True,
        ),
        (
            "import typing as type_hints\n"
            "type_hints = runtime_flags\n"
            "if type_hints.TYPE_CHECKING:\n"
            "    from .planning import project_align_stage",
            True,
        ),
    ],
)
def test_type_checking_compound_guards_and_shadowing(
    source: str,
    expected_hard: bool,
) -> None:
    edges = _collect_source(
        "pixipix.stages.align.api",
        "pixipix.stages.align",
        source,
    )
    edge = next(edge for edge in edges if edge.imported == "pixipix.stages.align.planning")
    assert (edge.hidden, edge.hard) == (True, expected_hard)
    assert edge.type_only is not expected_hard

    violations, _symbols, _roots = _align_dependency_violations(
        edges,
        {_module(path) for path in _production_files()},
    )
    placement = "runtime-hidden" if expected_hard else "type-only"
    assert any(f"{placement} placement" in violation for violation in violations)


def test_shadowed_type_checking_edge_reaches_hard_cycle_graph() -> None:
    edges = _collect_source(
        "pixipix.stages.align.api",
        "pixipix.stages.align",
        "from typing import TYPE_CHECKING as TC\n"
        "if TC:\n"
        "    from .planning import project_align_stage\n"
        "TC = True\n"
        "if TC:\n"
        "    from .planning import project_align_stage",
    )
    planning_edges = [edge for edge in edges if edge.imported == "pixipix.stages.align.planning"]
    assert [edge.hard for edge in planning_edges] == [False, True]
    assert [edge.type_only for edge in planning_edges] == [True, False]

    reverse_edge = _collect_source(
        "pixipix.stages.align.planning",
        "pixipix.stages.align",
        "from .api import publish_align",
    )[0]
    modules = {"pixipix.stages.align.api", "pixipix.stages.align.planning"}
    _assert_no_hard_cycle([planning_edges[0], reverse_edge], modules)
    with pytest.raises(AssertionError, match="hard import cycle"):
        _assert_no_hard_cycle([*planning_edges, reverse_edge], modules)


@pytest.mark.parametrize(
    ("source", "rejected"),
    [
        (
            "import importlib as loader\nloader.import_module('pixipix.stages.align.planning')",
            True,
        ),
        ("loader.import_module('pixipix.stages.align.planning')", False),
    ],
)
def test_dynamic_import_receiver_resolution_is_precise(source: str, rejected: bool) -> None:
    violations = _dynamic_import_violations(
        "pixipix.stages.align.api",
        ast.parse(source),
        GOVERNED_DYNAMIC_TARGETS,
    )
    assert bool(violations) is rejected
    if rejected:
        assert "dynamically imports pixipix.stages.align.planning" in violations[0]

    ordinary_alias_edges = _collect_source(
        "pixipix.stages.scale.execution",
        "pixipix.stages.scale",
        "import numpy as array_library",
    )
    ordinary_alias_violations, _symbols, _roots = _scale_dependency_violations(
        ordinary_alias_edges,
        {_module(path) for path in _production_files()},
    )
    assert not ordinary_alias_violations


@pytest.mark.parametrize(
    ("source", "rejected"),
    [
        (
            "import importlib as il\nil.import_module('pixipix.stages.scale')\nil = custom_loader",
            True,
        ),
        (
            "import importlib as il\nil = custom_loader\nil.import_module('pixipix.stages.scale')",
            False,
        ),
        (
            "from importlib import import_module as load\n"
            "load('pixipix.stages.scale')\n"
            "load = custom_loader",
            True,
        ),
        (
            "from importlib import import_module as load\n"
            "load = custom_loader\n"
            "load('pixipix.stages.scale')",
            False,
        ),
        (
            "import importlib as il\n"
            "il: Loader = custom_loader\n"
            "il.import_module('pixipix.stages.scale')",
            False,
        ),
        (
            "from importlib import import_module as load\n"
            "load: Callable[..., object] = custom_loader\n"
            "load('pixipix.stages.scale')",
            False,
        ),
        (
            "load = __import__\nload('pixipix.stages.scale')",
            True,
        ),
        (
            "loader = custom_loader\nloader.import_module('pixipix.stages.scale')",
            False,
        ),
    ],
)
def test_dynamic_import_alias_rebinding_is_source_ordered(
    source: str,
    rejected: bool,
) -> None:
    violations = _dynamic_import_violations(
        "pixipix.stages.align.api",
        ast.parse(source),
        GOVERNED_DYNAMIC_TARGETS,
    )
    assert bool(violations) is rejected


def test_only_locked_stage_to_stage_rounding_edge_exists() -> None:
    modules = {_module(path) for path in _production_files()}
    stage_edges = _cross_stage_edges(_edges(), modules)
    details = "\n".join(
        _failure("stage-to-stage dependency", edge) for edge, _target in stage_edges
    ) or (
        "stage-to-stage dependency: expected exactly "
        "pixipix.stages.pixelize.execution -> "
        "pixipix.stages.scale.round_channel_half_away_from_zero"
    )
    assert len(stage_edges) == 1, details
    edge, target = stage_edges[0]
    assert (edge.importer, target, edge.names, edge.hard) == (
        "pixipix.stages.pixelize.execution",
        "pixipix.stages.scale",
        ("round_channel_half_away_from_zero",),
        True,
    ), _failure("exact rounding dependency", edge)


def test_resource_projection_formulas_remain_stage_local() -> None:
    definitions: dict[str, set[str]] = {}
    for path in _production_files():
        module = _module(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions[module] = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    for stage in ("extract", "scale", "pixelize", "align"):
        function = f"project_{stage}_resources"
        owners = {
            module
            for module, functions in definitions.items()
            if function in functions
            and (
                module == f"pixipix.stages.{stage}" or module.startswith(f"pixipix.stages.{stage}.")
            )
        }
        assert owners, (
            f"stage-local resource formula: {function} is missing from "
            f"pixipix.stages.{stage} or its package modules"
        )
        assert function not in definitions["pixipix.resources"], (
            f"stage-local resource formula: pixipix.resources defines {function}; "
            f"it must remain owned by pixipix.stages.{stage}"
        )

    modules = {_module(path) for path in _production_files()}
    resource_error_edge = next(
        edge
        for edge in _edges()
        if edge.importer == "pixipix.resources"
        and edge.imported == "pixipix.errors"
        and edge.names == ("ResourcePolicyError",)
    )
    assert resource_error_edge.hidden is True
    assert resource_error_edge.hard is False
    assert resource_error_edge.type_only is False
    for contract in (*STAGE_DEPENDENCY_CONTRACTS, PIPELINE_DEPENDENCY_CONTRACT):
        violations, _symbols, _roots = _governed_dependency_violations(
            contract,
            [resource_error_edge],
            modules,
        )
        assert not violations, (
            "the generalized governed-edge rule rejected the legitimate local "
            "resources-to-errors cycle break"
        )

    governed_local_edge = _collect_source(
        "pixipix.stages.align.api",
        "pixipix.stages.align",
        "def load():\n    from .planning import project_align_stage",
    )[-1]
    assert (governed_local_edge.hidden, governed_local_edge.type_only) == (True, False)
    governed_violations, _symbols, _roots = _align_dependency_violations(
        [governed_local_edge],
        modules,
    )
    assert any("runtime-hidden placement" in violation for violation in governed_violations)


def test_canonical_json_writes_remain_in_serialization_module() -> None:
    violations: list[str] = []
    for path in _production_files():
        module = _module(path)
        if module == "pixipix.serialization":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_canonical_json_violations(module, tree))
    assert not violations, "\n".join(violations)


def test_only_module_entrypoint_imports_cli() -> None:
    modules = {_module(path) for path in _production_files()}
    violations = [
        edge
        for edge in _edges()
        if "pixipix.cli" in _target_modules(edge, modules) and edge.importer != "pixipix.__main__"
    ]
    assert not violations, "\n".join(
        _failure("CLI dependency direction", edge) for edge in violations
    )
    entrypoint_edges = [
        edge
        for edge in _edges()
        if edge.importer == "pixipix.__main__" and "pixipix.cli" in _target_modules(edge, modules)
    ]
    assert len(entrypoint_edges) == 1, (
        "CLI dependency direction: pixipix.__main__ must import exactly one pixipix.cli edge"
    )
    assert entrypoint_edges[0].names == ("main",), _failure(
        "CLI dependency direction requires only cli.main",
        entrypoint_edges[0],
    )


def test_production_import_graph_has_no_hard_cycle() -> None:
    modules = {_module(path) for path in _production_files()}
    _assert_no_hard_cycle(_edges(), modules)


def test_align_internal_module_set_and_dependency_direction_are_exact() -> None:
    modules = {_module(path) for path in _production_files()}
    actual_modules = {
        module
        for module in modules
        if module == "pixipix.stages.align" or module.startswith("pixipix.stages.align.")
    }
    assert actual_modules == ALIGN_MODULES, (
        "align internal module set differs: "
        f"missing={sorted(ALIGN_MODULES - actual_modules)}, "
        f"unexpected={sorted(actual_modules - ALIGN_MODULES)}"
    )

    violations, internal_symbols, root_imports = _align_dependency_violations(
        _edges(),
        modules,
    )

    assert not violations, "\n".join(violations)
    assert root_imports == ALIGN_ALLOWED_ROOT_IMPORTS
    assert internal_symbols == ALIGN_ALLOWED_INTERNAL_SYMBOLS, (
        "align internal dependency graph differs: "
        f"missing={sorted(set(ALIGN_ALLOWED_INTERNAL_SYMBOLS) - set(internal_symbols))}, "
        f"unexpected={sorted(set(internal_symbols) - set(ALIGN_ALLOWED_INTERNAL_SYMBOLS))}, "
        f"symbols={internal_symbols}"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pixipix.stages.scale", "pixipix.stages.scale"),
        ("import pixipix.stages.scale as scale", "pixipix.stages.scale"),
        ("from pixipix.stages import scale", "pixipix.stages.scale"),
        (
            "from pixipix.stages.scale import round_channel_half_away_from_zero",
            "pixipix.stages.scale",
        ),
        ("from .scale import round_channel_half_away_from_zero", "pixipix.stages.scale"),
    ],
)
def test_import_target_resolution_covers_supported_syntax(
    source: str,
    expected: str,
) -> None:
    modules = {"pixipix.stages", *STAGE_IMPLEMENTATIONS}
    edge = _collect_source("pixipix.stages.pixelize.execution", "pixipix.stages", source)[0]
    assert expected in _target_modules(edge, modules)


def test_package_init_relative_import_resolves_below_package() -> None:
    edge = _collect_source(
        "pixipix.stages.align",
        "pixipix.stages.align",
        "from .api import publish_align",
    )[0]
    assert edge.imported == "pixipix.stages.align.api"


@pytest.mark.parametrize(
    "source",
    [
        "from pixipix.stages.scale import round_channel_half_away_from_zero, publish_scale",
        "from pixipix.stages.scale import *",
        "import pixipix.stages.scale",
        "from pixipix.stages import scale",
        "from pixipix.stages.align import publish_align",
    ],
)
def test_rounding_exception_rejects_broader_cross_stage_imports(source: str) -> None:
    modules = {"pixipix.stages", *STAGE_IMPLEMENTATIONS}
    edges = _collect_source("pixipix.stages.pixelize.execution", "pixipix.stages", source)
    cross_stage = _cross_stage_edges(edges, modules)
    assert len(cross_stage) == 1
    edge, target = cross_stage[0]
    assert (target, edge.names) != (
        "pixipix.stages.scale",
        ("round_channel_half_away_from_zero",),
    )
    assert "stage-to-stage dependency" in _failure("stage-to-stage dependency", edge)


def test_rounding_exception_rejects_reverse_dependency() -> None:
    modules = {"pixipix.stages", *STAGE_IMPLEMENTATIONS}
    edges = _collect_source(
        "pixipix.stages.scale",
        "pixipix.stages",
        "from pixipix.stages.pixelize import project_cell_grid",
    )
    assert _cross_stage_edges(edges, modules)


def test_cycle_detection_expands_symbols_imported_from_packages() -> None:
    modules = {"pixipix", "pixipix.config", "pixipix.resources"}
    edges = [
        *_collect_source(
            "pixipix.config",
            "pixipix",
            "from pixipix import resources",
        ),
        *_collect_source(
            "pixipix.resources",
            "pixipix",
            "from pixipix import config",
        ),
    ]
    with pytest.raises(AssertionError, match="hard import cycle"):
        _assert_no_hard_cycle(edges, modules)


@pytest.mark.parametrize(
    "source",
    [
        "import json\njson.dump({}, stream)",
        "import json as codec\ncodec.dumps({})",
        "from json import dumps\ndumps({})",
        "from json import dumps as render\nrender({})",
        "def render():\n    import json\n    return json.dumps({})",
    ],
)
def test_serialization_rule_catches_aliases_and_function_local_imports(source: str) -> None:
    violations = _canonical_json_violations("pixipix.example", ast.parse(source))
    assert violations
    assert "pixipix.example" in violations[0]


@pytest.mark.parametrize(
    "source",
    [
        "import json\njson.load(stream)",
        "import json as codec\ncodec.loads('{}')",
        "from json import loads\nloads('{}')",
    ],
)
def test_serialization_rule_allows_json_parsing(source: str) -> None:
    assert not _canonical_json_violations("pixipix.example", ast.parse(source))
