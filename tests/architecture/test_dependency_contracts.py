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


def test_only_cli_imports_stage_command_publishers() -> None:
    modules = {_module(path) for path in _production_files()}
    facade_reexport = (
        "pixipix.stages.align",
        "pixipix.stages.align.api",
        ("publish_align",),
    )
    violations = [
        edge
        for edge in _edges()
        if any(_stage_root(target) for target in _target_modules(edge, modules))
        and STAGE_PUBLISHERS.intersection(edge.names)
        and edge.importer != "pixipix.cli"
        and (edge.importer, edge.imported, edge.names) != facade_reexport
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


def test_only_locked_stage_to_stage_rounding_edge_exists() -> None:
    modules = {_module(path) for path in _production_files()}
    stage_edges = _cross_stage_edges(_edges(), modules)
    details = "\n".join(
        _failure("stage-to-stage dependency", edge) for edge, _target in stage_edges
    ) or (
        "stage-to-stage dependency: expected exactly "
        "pixipix.stages.pixelize -> "
        "pixipix.stages.scale.round_channel_half_away_from_zero"
    )
    assert len(stage_edges) == 1, details
    edge, target = stage_edges[0]
    assert (edge.importer, target, edge.names, edge.hard) == (
        "pixipix.stages.pixelize",
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
    edge = _collect_source("pixipix.stages.pixelize", "pixipix.stages", source)[0]
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
    edges = _collect_source("pixipix.stages.pixelize", "pixipix.stages", source)
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
