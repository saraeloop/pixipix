from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONAL_MODULES = {
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
ALIGN_MODULES = {
    "pixipix.stages.align",
    "pixipix.stages.align.api",
    "pixipix.stages.align.execution",
    "pixipix.stages.align.geometry",
    "pixipix.stages.align.planning",
}
ALIGN_ALLOWED_INTERNAL_EDGES = {
    ("pixipix.stages.align", "pixipix.stages.align.api"),
    ("pixipix.stages.align", "pixipix.stages.align.execution"),
    ("pixipix.stages.align", "pixipix.stages.align.geometry"),
    ("pixipix.stages.align", "pixipix.stages.align.planning"),
    ("pixipix.stages.align.api", "pixipix.stages.align.execution"),
    ("pixipix.stages.align.api", "pixipix.stages.align.planning"),
    ("pixipix.stages.align.execution", "pixipix.stages.align.planning"),
    ("pixipix.stages.align.planning", "pixipix.stages.align.geometry"),
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
    ("pixipix.stages.scale.planning", "pixipix.stages.scale.geometry"): {
        "transformed_dimension",
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
    "pixipix.stages.scale.geometry": set(),
    "pixipix.stages.scale.metadata": {
        "pixipix",
        "pixipix.config",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.stages.scale.planning",
    },
    "pixipix.stages.scale.planning": {
        "pixipix.config",
        "pixipix.errors",
        "pixipix.models",
        "pixipix.pipeline.input",
        "pixipix.resources",
        "pixipix.stages.scale.geometry",
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
SCALE_ALLOWED_ROOT_IMPORTS = {
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
PIXELIZE_ALLOWED_ROOT_IMPORTS = {
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
}


@dataclass(frozen=True, slots=True)
class ImportEdge:
    importer: str
    imported: str
    names: tuple[str, ...]
    line: int
    hard: bool


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, importer: str, package: str) -> None:
        self.importer = importer
        self.package = package
        self.edges: list[ImportEdge] = []
        self._scope_depth = 0
        self._type_checking_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        is_type_checking = isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        if is_type_checking:
            self._type_checking_depth += 1
        for statement in node.body:
            self.visit(statement)
        if is_type_checking:
            self._type_checking_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.edges.append(
                ImportEdge(
                    self.importer,
                    alias.name,
                    (),
                    node.lineno,
                    self._scope_depth == 0 and self._type_checking_depth == 0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported = node.module or ""
        if node.level:
            imported = importlib.util.resolve_name(
                "." * node.level + imported,
                self.package,
            )
        self.edges.append(
            ImportEdge(
                self.importer,
                imported,
                tuple(alias.name for alias in node.names),
                node.lineno,
                self._scope_depth == 0 and self._type_checking_depth == 0,
            )
        )


class _DynamicImportCollector(ast.NodeVisitor):
    def __init__(self, importer: str) -> None:
        self.importer = importer
        self.import_names = {"__import__"}
        self.violations: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "importlib":
            self.import_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.import_names:
            self.import_names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dynamic = (isinstance(node.func, ast.Name) and node.func.id in self.import_names) or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
        )
        if dynamic:
            self.violations.append(
                f"dynamic import: {self.importer} calls a dynamic importer at production "
                f"line {node.lineno}"
            )
        self.generic_visit(node)


def _dynamic_import_violations(importer: str, tree: ast.AST) -> list[str]:
    collector = _DynamicImportCollector(importer)
    collector.visit(tree)
    return collector.violations


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


def _scale_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    violations: list[str] = []
    internal_symbols: dict[tuple[str, str], set[str]] = {}
    root_imports: set[tuple[str, str, tuple[str, ...]]] = set()
    for edge in edges:
        if edge.importer not in SCALE_MODULES:
            continue
        if (
            not edge.imported.startswith("pixipix")
            and edge.imported not in SCALE_ALLOWED_EXTERNAL_IMPORTS[edge.importer]
        ):
            violations.append(_failure("scale external capability", edge))
        if edge.imported == "pixipix":
            root_import = (edge.importer, edge.imported, edge.names)
            root_imports.add(root_import)
            if root_import not in SCALE_ALLOWED_ROOT_IMPORTS:
                violations.append(_failure("scale package-root capability", edge))
        if edge.imported.startswith("pixipix.pipeline.") and (
            set(edge.names)
            - SCALE_ALLOWED_PIPELINE_SYMBOLS.get((edge.importer, edge.imported), set())
        ):
            violations.append(_failure("scale shared pipeline capability", edge))
        for target in _target_modules(edge, modules):
            if target in SCALE_MODULES:
                key = (edge.importer, target)
                internal_symbols.setdefault(key, set()).update(edge.names)
                allowed_symbols = SCALE_ALLOWED_INTERNAL_SYMBOLS.get(key)
                if allowed_symbols is None or not edge.names or set(edge.names) - allowed_symbols:
                    violations.append(_failure("scale internal dependency direction", edge))
            if (
                target.startswith("pixipix")
                and target not in SCALE_ALLOWED_PIXIPIX_DEPENDENCIES[edge.importer]
            ):
                violations.append(_failure("scale layer capability", edge))
    return violations, internal_symbols, root_imports


def _pixelize_dependency_violations(
    edges: list[ImportEdge],
    modules: set[str],
) -> tuple[list[str], dict[tuple[str, str], set[str]], set[tuple[str, str, tuple[str, ...]]]]:
    violations: list[str] = []
    internal_symbols: dict[tuple[str, str], set[str]] = {}
    root_imports: set[tuple[str, str, tuple[str, ...]]] = set()
    for edge in edges:
        if edge.importer not in PIXELIZE_MODULES:
            continue
        if (
            not edge.imported.startswith("pixipix")
            and edge.imported not in PIXELIZE_ALLOWED_EXTERNAL_IMPORTS[edge.importer]
        ):
            violations.append(_failure("pixelize external capability", edge))
        if edge.imported == "pixipix":
            root_import = (edge.importer, edge.imported, edge.names)
            root_imports.add(root_import)
            if root_import not in PIXELIZE_ALLOWED_ROOT_IMPORTS:
                violations.append(_failure("pixelize package-root capability", edge))
        if edge.imported.startswith("pixipix.pipeline.") and (
            set(edge.names)
            - PIXELIZE_ALLOWED_PIPELINE_SYMBOLS.get((edge.importer, edge.imported), set())
        ):
            violations.append(_failure("pixelize shared pipeline capability", edge))
        for target in _target_modules(edge, modules):
            if target in PIXELIZE_MODULES:
                key = (edge.importer, target)
                internal_symbols.setdefault(key, set()).update(edge.names)
                allowed_symbols = PIXELIZE_ALLOWED_INTERNAL_SYMBOLS.get(key)
                if allowed_symbols is None or not edge.names or set(edge.names) - allowed_symbols:
                    violations.append(_failure("pixelize internal dependency direction", edge))
            if target == "pixipix.stages.scale" and edge.names != (
                "round_channel_half_away_from_zero",
            ):
                violations.append(_failure("pixelize scale capability", edge))
            if (
                target.startswith("pixipix")
                and target not in PIXELIZE_ALLOWED_PIXIPIX_DEPENDENCIES[edge.importer]
            ):
                violations.append(_failure("pixelize layer capability", edge))
    return violations, internal_symbols, root_imports


def test_only_cli_imports_stage_command_publishers() -> None:
    modules = {_module(path) for path in _production_files()}
    facade_reexports = {
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

    violations: list[str] = []
    for edge in _edges():
        if edge.importer not in PIPELINE_MODULES:
            continue
        for target in _target_modules(edge, modules):
            if (
                target.startswith("pixipix")
                and target not in PIPELINE_ALLOWED_PIXIPIX_DEPENDENCIES[edge.importer]
            ):
                violations.append(_failure("pipeline dependency direction", edge))
    assert not violations, "\n".join(violations)


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


def test_extract_package_is_one_monolithic_implementation_module() -> None:
    package = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract"
    source_files = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    )

    assert source_files == ["__init__.py"]
    assert not (package.parent / "extract.py").exists()
    assert _module(package / "__init__.py") == "pixipix.stages.extract"


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
    dynamic_imports: list[str] = []
    for path in _production_files():
        importer = _module(path)
        if importer in PIXELIZE_MODULES:
            dynamic_imports.extend(
                _dynamic_import_violations(
                    importer,
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
                )
            )

    assert not violations, "\n".join(violations)
    assert not dynamic_imports, "\n".join(dynamic_imports)
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


def test_pixelize_dependency_rule_rejects_dynamic_imports() -> None:
    cases = (
        "__import__('pixipix.stages.scale')",
        "load = __import__\nload('pixipix.stages.scale')",
        "import importlib\nimportlib.import_module('pixipix.stages.scale')",
        "from importlib import import_module as load\nload('pixipix.stages.scale')",
    )
    for source in cases:
        violations = _dynamic_import_violations(
            "pixipix.stages.pixelize.execution",
            ast.parse(source),
        )
        assert violations, f"pixelize dynamic import bypass was accepted: {source}"


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

    violations: list[str] = []
    internal_edges: set[tuple[str, str]] = set()
    for edge in _edges():
        if edge.importer not in ALIGN_MODULES:
            continue
        if (
            not edge.imported.startswith("pixipix")
            and edge.imported not in ALIGN_ALLOWED_EXTERNAL_IMPORTS[edge.importer]
        ):
            violations.append(_failure("align external capability", edge))
        if edge.imported.startswith("pixipix.pipeline.") and (
            set(edge.names)
            - ALIGN_ALLOWED_PIPELINE_SYMBOLS.get((edge.importer, edge.imported), set())
        ):
            violations.append(_failure("align shared pipeline capability", edge))
        for target in _target_modules(edge, modules):
            if target in ALIGN_MODULES:
                internal_edges.add((edge.importer, target))
                if (edge.importer, target) not in ALIGN_ALLOWED_INTERNAL_EDGES:
                    violations.append(_failure("align internal dependency direction", edge))
            if (
                target.startswith("pixipix")
                and target not in ALIGN_ALLOWED_PIXIPIX_DEPENDENCIES[edge.importer]
            ):
                violations.append(_failure("align layer capability", edge))

    assert not violations, "\n".join(violations)
    assert internal_edges == ALIGN_ALLOWED_INTERNAL_EDGES, (
        "align internal dependency graph differs: "
        f"missing={sorted(ALIGN_ALLOWED_INTERNAL_EDGES - internal_edges)}, "
        f"unexpected={sorted(internal_edges - ALIGN_ALLOWED_INTERNAL_EDGES)}"
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
