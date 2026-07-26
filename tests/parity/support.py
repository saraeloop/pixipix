"""Deterministic capture and comparison support for the post-M3 parity baseline."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

BASELINE_COMMIT = "aace7d9ac5fd4ba43c3315afd2f8eceb582d9020"
MANIFEST_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path(__file__).with_name("baseline") / "post-m3.json"
API_PROBE_PATH = Path(__file__).with_name("capture_api.py")
CANONICAL_RUNTIME_FIELDS = (
    "implementation",
    "pythonVersion",
    "platformIdentifier",
)


class ParityError(RuntimeError):
    """A deterministic parity capture or comparison failure."""


@dataclass(frozen=True, slots=True)
class Lineage:
    identifier: str
    image: str
    config: str


LINEAGES = (
    Lineage("pixi", "examples/pixi-demo/pixi-demo-sheet.png", "examples/pixi-demo/pixipix.toml"),
    Lineage(
        "robot",
        "tests/fixtures/robot-geometric.png",
        "tests/fixtures/robot-geometric.toml",
    ),
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _encoded(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _artifact_records(root: Path) -> tuple[list[dict[str, object]], str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    records: list[dict[str, object]] = []
    tree = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        relative_bytes = relative.encode("utf-8")
        tree.update(relative_bytes)
        tree.update(b"\0")
        tree.update(content)
        records.append(
            {
                "path": relative,
                "byteLength": len(content),
                "sha256": sha256_bytes(content),
            }
        )
    return records, tree.hexdigest()


def _runtime_environment(source_root: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    if source_root is None:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(source_root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _environment(
    source_root: Path,
    python: Path,
    *,
    import_root: Path,
    production_source_root: Path | None,
) -> dict[str, object]:
    probe = (
        "import json,platform,sys;"
        "from importlib.metadata import version;"
        "from pathlib import Path;"
        "import numpy,PIL,pixipix;"
        "print(json.dumps({"
        "'implementation':platform.python_implementation(),"
        "'pythonVersion':platform.python_version(),"
        "'platformIdentifier':sys.platform+'-'+platform.machine().lower(),"
        "'numpy':numpy.__version__,"
        "'pillow':PIL.__version__,"
        "'typer':version('typer'),"
        "'pixipixFile':str(Path(pixipix.__file__).resolve())"
        "},sort_keys=True))"
    )
    result = subprocess.run(
        [python, "-B", "-c", probe],
        cwd=source_root,
        env=_runtime_environment(production_source_root),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ParityError("source-isolation probe failed")
    data = cast(dict[str, object], json.loads(result.stdout))
    imported = Path(cast(str, data.pop("pixipixFile")))
    try:
        imported.relative_to(import_root.resolve())
    except ValueError as error:
        raise ParityError(
            "production import did not resolve from the selected isolated location"
        ) from error
    data["sourceImportIsolation"] = True
    return data


def capture_environment(
    source_root: Path,
    *,
    python: Path | None = None,
    import_root: Path | None = None,
) -> dict[str, object]:
    interpreter = python or Path(sys.executable)
    production_source_root = source_root if import_root is None else None
    allowed_import_root = source_root if import_root is None else import_root
    return _environment(
        source_root,
        interpreter,
        import_root=allowed_import_root,
        production_source_root=production_source_root,
    )


def canonical_runtime_mismatch(
    expected_environment: object,
    actual_environment: dict[str, object],
) -> str | None:
    if not isinstance(expected_environment, dict):
        raise ParityError("canonical parity environment is missing or invalid")
    differences = [
        (
            field,
            expected_environment.get(field),
            actual_environment.get(field),
        )
        for field in CANONICAL_RUNTIME_FIELDS
        if expected_environment.get(field) != actual_environment.get(field)
    ]
    if not differences:
        return None
    details = ", ".join(
        f"{field} expected={expected!r} actual={actual!r}"
        for field, expected, actual in differences
    )
    return f"exact-byte parity requires the canonical runtime: {details}"


def _copy_inputs(source_root: Path, execution_root: Path) -> dict[str, str]:
    fixture_hashes: dict[str, str] = {}
    inputs = execution_root / "inputs"
    inputs.mkdir()
    for lineage in LINEAGES:
        for relative in (lineage.image, lineage.config):
            source = source_root / relative
            content = source.read_bytes()
            target = inputs / Path(relative).name
            target.write_bytes(content)
            fixture_hashes[relative] = sha256_bytes(content)
    return dict(sorted(fixture_hashes.items()))


def _warning_lines(stderr: bytes) -> list[str]:
    return [
        _encoded(line)
        for line in stderr.splitlines(keepends=True)
        if line.startswith(b"pixipix: warning ")
    ]


def _run_cli_case(
    *,
    case_id: str,
    arguments: list[str],
    python: Path,
    source_root: Path,
    production_source_root: Path | None,
    execution_root: Path,
    output_root: Path | None = None,
) -> dict[str, object]:
    result = subprocess.run(
        [python, "-B", "-m", "pixipix", *arguments],
        cwd=execution_root,
        env=_runtime_environment(production_source_root),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ParityError(f"canonical CLI case failed: {case_id}")
    artifacts: list[dict[str, object]] = []
    tree_hash: str | None = None
    if output_root is not None:
        artifacts, tree_hash = _artifact_records(execution_root / output_root)
    contracts = ["cli-byte-parity"]
    if artifacts:
        contracts.append("persisted-artifact-parity")
    return {
        "caseId": case_id,
        "kind": "cli-byte-parity",
        "parityContracts": contracts,
        "exitCode": result.returncode,
        "stdoutBase64": _encoded(result.stdout),
        "stderrBase64": _encoded(result.stderr),
        "warningLinesBase64": _warning_lines(result.stderr),
        "artifacts": artifacts,
        "stageTreeSha256": tree_hash,
    }


def _run_lineage(
    lineage: Lineage,
    *,
    python: Path,
    source_root: Path,
    production_source_root: Path | None,
    execution_root: Path,
) -> list[dict[str, object]]:
    image = f"inputs/{Path(lineage.image).name}"
    config = f"inputs/{Path(lineage.config).name}"
    prefix = Path("cli") / lineage.identifier
    outputs = {stage: prefix / stage for stage in ("extract", "scale", "pixelize", "align")}
    cases = [
        _run_cli_case(
            case_id=f"{lineage.identifier}.cli.inspect",
            arguments=["inspect", image, "--config", config],
            python=python,
            source_root=source_root,
            production_source_root=production_source_root,
            execution_root=execution_root,
        )
    ]
    commands = (
        ("extract", image),
        ("scale", outputs["extract"].as_posix()),
        ("pixelize", outputs["scale"].as_posix()),
        ("align", outputs["pixelize"].as_posix()),
    )
    for stage, input_path in commands:
        output = outputs[stage]
        cases.append(
            _run_cli_case(
                case_id=f"{lineage.identifier}.cli.{stage}",
                arguments=[
                    stage,
                    input_path,
                    "--config",
                    config,
                    "--output",
                    output.as_posix(),
                ],
                python=python,
                source_root=source_root,
                production_source_root=production_source_root,
                execution_root=execution_root,
                output_root=output,
            )
        )
    return cases


def _run_api_probe(
    *,
    python: Path,
    source_root: Path,
    production_source_root: Path | None,
    execution_root: Path,
    api_probe_path: Path,
) -> list[dict[str, object]]:
    result = subprocess.run(
        [python, "-B", api_probe_path, execution_root],
        cwd=execution_root,
        env=_runtime_environment(production_source_root),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ParityError("direct Python API capture failed")
    cases = cast(list[dict[str, object]], json.loads(result.stdout))
    for case in cases:
        raw_output = case.pop("outputRelative", None)
        if raw_output is None:
            case["parityContracts"] = ["structural-api-parity"]
            case["artifacts"] = []
            case["stageTreeSha256"] = None
            continue
        output = execution_root / cast(str, raw_output)
        records, tree_hash = _artifact_records(output)
        case["parityContracts"] = [
            "structural-api-parity",
            "persisted-artifact-parity",
        ]
        case["artifacts"] = records
        case["stageTreeSha256"] = tree_hash
    return cases


def capture_behavior(
    source_root: Path,
    execution_root: Path,
    *,
    python: Path | None = None,
    import_root: Path | None = None,
    api_probe_path: Path = API_PROBE_PATH,
) -> dict[str, object]:
    if execution_root.exists() and any(execution_root.iterdir()):
        raise ParityError("parity execution root must be empty")
    execution_root.mkdir(parents=True, exist_ok=True)
    interpreter = python or Path(sys.executable)
    production_source_root = source_root if import_root is None else None
    environment = capture_environment(
        source_root,
        python=interpreter,
        import_root=import_root,
    )
    fixtures = _copy_inputs(source_root, execution_root)
    cases: list[dict[str, object]] = []
    for lineage in LINEAGES:
        cases.extend(
            _run_lineage(
                lineage,
                python=interpreter,
                source_root=source_root,
                production_source_root=production_source_root,
                execution_root=execution_root,
            )
        )
    cases.extend(
        _run_api_probe(
            python=interpreter,
            source_root=source_root,
            production_source_root=production_source_root,
            execution_root=execution_root,
            api_probe_path=api_probe_path,
        )
    )
    return {
        "environment": environment,
        "uvLockSha256": sha256_bytes((source_root / "uv.lock").read_bytes()),
        "fixtureSha256": fixtures,
        "cases": cases,
    }


def baseline_manifest(behavior: dict[str, object]) -> dict[str, object]:
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "baselineProductionCommit": BASELINE_COMMIT,
        "baselineCleanWorktreeAttestation": True,
        **behavior,
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, object]:
    if not path.is_file():
        raise ParityError("post-M3 parity baseline is missing")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ParityError("post-M3 parity baseline must contain one JSON object")
    manifest = cast(dict[str, object], value)
    if manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ParityError("post-M3 parity baseline schema is unsupported")
    if manifest.get("baselineProductionCommit") != BASELINE_COMMIT:
        raise ParityError("post-M3 parity baseline records the wrong production commit")
    if manifest.get("baselineCleanWorktreeAttestation") is not True:
        raise ParityError("post-M3 parity baseline lacks a clean-worktree attestation")
    return manifest


def compare_behavior(expected: dict[str, object], actual: dict[str, object]) -> None:
    for field in ("environment", "uvLockSha256", "fixtureSha256"):
        if expected.get(field) != actual.get(field):
            raise ParityError(f"parity {field} differs from the declared canonical baseline")
    expected_cases = _case_map(expected)
    actual_cases = _case_map(actual)
    missing_cases = sorted(expected_cases.keys() - actual_cases.keys())
    unexpected_cases = sorted(actual_cases.keys() - expected_cases.keys())
    if missing_cases or unexpected_cases:
        raise ParityError(
            f"parity cases differ: missing={missing_cases}, unexpected={unexpected_cases}"
        )
    failures: list[str] = []
    for case_id in sorted(expected_cases):
        failures.extend(_compare_case(case_id, expected_cases[case_id], actual_cases[case_id]))
    if failures:
        raise ParityError("parity mismatch:\n" + "\n".join(failures))


def _case_map(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ParityError("parity manifest cases must be a list")
    cases: dict[str, dict[str, object]] = {}
    for raw in raw_cases:
        if not isinstance(raw, dict) or not isinstance(raw.get("caseId"), str):
            raise ParityError("parity manifest contains an invalid case")
        case = cast(dict[str, object], raw)
        identifier = cast(str, case["caseId"])
        if identifier in cases:
            raise ParityError(f"parity manifest repeats case {identifier}")
        cases[identifier] = case
    return cases


def _artifact_map(case: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_artifacts = case.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ParityError("parity case artifacts must be a list")
    result: dict[str, dict[str, object]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ParityError("parity case contains an invalid artifact")
        artifact = cast(dict[str, object], raw)
        result[cast(str, artifact["path"])] = artifact
    return result


def _compare_case(
    case_id: str,
    expected: dict[str, object],
    actual: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    expected_artifacts = _artifact_map(expected)
    actual_artifacts = _artifact_map(actual)
    missing = sorted(expected_artifacts.keys() - actual_artifacts.keys())
    unexpected = sorted(actual_artifacts.keys() - expected_artifacts.keys())
    changed = sorted(
        path
        for path in expected_artifacts.keys() & actual_artifacts.keys()
        if expected_artifacts[path] != actual_artifacts[path]
    )
    if missing:
        failures.append(f"{case_id}: missing artifacts {missing}")
    if unexpected:
        failures.append(f"{case_id}: unexpected artifacts {unexpected}")
    if changed:
        failures.append(f"{case_id}: changed artifacts {changed}")
    excluded = {"artifacts"}
    changed_fields = sorted(
        key
        for key in expected.keys() | actual.keys()
        if key not in excluded and expected.get(key) != actual.get(key)
    )
    if changed_fields:
        failures.append(f"{case_id}: changed fields {changed_fields}")
    return failures
