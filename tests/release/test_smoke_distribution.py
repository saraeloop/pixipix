from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pytest
from PIL import Image

from scripts.smoke_distribution import (
    FIXTURE_CONTRACT,
    SMOKE_STAGES,
    SmokeFailure,
    _isolated_paths,
    _run_stage,
    _sanitized_environment,
    _validate_final_output,
    _validate_installed_location,
    _validate_installed_resource_identity,
    _validate_installed_resource_refusal,
    _write_resource_refusal_fixture,
)
from tests.architecture.test_import_compatibility import (
    build_checkout_compatibility_manifest,
    compatibility_contract_payload,
    installed_compatibility_manifest_program,
)
from tests.parity.support import (
    PROJECT_ROOT as PARITY_PROJECT_ROOT,
)
from tests.parity.support import (
    capture_behavior,
    capture_environment,
    compare_behavior,
    load_release_baseline,
    require_canonical_runtime,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "smoke_distribution.py"
EXTERNAL_RELEASE_CANDIDATE_ENV = "PIXIPIX_RELEASE_CANDIDATE_DIR"
DET_CONTRACT_SNIPPETS = (
    "Within a supported, verified PixiPix execution environment",
    (
        "same PixiPix version produce the same artifact bytes, the same metadata, and the "
        "same warning order"
    ),
    (
        "Cross-platform byte equality is not claimed unless it is separately established by "
        "an explicit parity authority"
    ),
)


@dataclass(frozen=True, slots=True)
class BuiltArtifacts:
    direct_wheel: Path
    sdist: Path
    rebuilt_wheel: Path


type InstalledArtifactName = Literal["direct_wheel", "rebuilt_wheel"]
type InstalledAliasName = Literal["var", "tmp"]


def _is_darwin_runtime() -> bool:
    return sys.platform == "darwin"


@dataclass(frozen=True, slots=True)
class InstalledAliasCase:
    artifact: InstalledArtifactName
    alias: InstalledAliasName


EXPECTED_INSTALLED_ALIAS_MATRIX = frozenset(
    {
        ("direct_wheel", "var"),
        ("direct_wheel", "tmp"),
        ("rebuilt_wheel", "var"),
        ("rebuilt_wheel", "tmp"),
    }
)
INSTALLED_ALIAS_CASES = (
    InstalledAliasCase("direct_wheel", "var"),
    InstalledAliasCase("direct_wheel", "tmp"),
    InstalledAliasCase("rebuilt_wheel", "var"),
    InstalledAliasCase("rebuilt_wheel", "tmp"),
)


def _run(
    command: list[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    return subprocess.run(
        rendered,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def _external_release_candidate() -> tuple[Path, Path] | None:
    raw_directory = os.environ.get(EXTERNAL_RELEASE_CANDIDATE_ENV)
    if raw_directory is None:
        return None
    assert raw_directory, f"{EXTERNAL_RELEASE_CANDIDATE_ENV} must not be empty"
    candidate = Path(raw_directory)
    assert candidate.is_absolute(), f"{EXTERNAL_RELEASE_CANDIDATE_ENV} must be absolute"
    assert not candidate.is_symlink(), f"{EXTERNAL_RELEASE_CANDIDATE_ENV} must not be a symlink"
    assert candidate.is_dir(), f"{EXTERNAL_RELEASE_CANDIDATE_ENV} must be a directory"
    resolved = candidate.resolve(strict=True)
    assert not resolved.is_relative_to(PROJECT_ROOT.resolve()), (
        f"{EXTERNAL_RELEASE_CANDIDATE_ENV} must be outside the source checkout"
    )
    entries = tuple(sorted(resolved.iterdir()))
    assert len(entries) == 2, "external release candidate must contain exactly two artifacts"
    assert all(path.is_file() and not path.is_symlink() for path in entries), (
        "external release candidate artifacts must be regular files"
    )
    wheels = tuple(path for path in entries if path.suffix == ".whl")
    sdists = tuple(path for path in entries if path.name.endswith(".tar.gz"))
    assert len(wheels) == 1, "external release candidate must contain exactly one wheel"
    assert len(sdists) == 1, "external release candidate must contain exactly one sdist"
    return wheels[0], sdists[0]


@pytest.fixture(scope="session")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> BuiltArtifacts:
    root = tmp_path_factory.mktemp("distribution-smoke-artifacts")
    rebuilt = root / "rebuilt"
    rebuilt.mkdir()
    external = _external_release_candidate()
    if external is None:
        direct = root / "direct"
        direct.mkdir()
        wheel_result = _run(
            ["uv", "build", "--wheel", "--no-sources", "--out-dir", direct],
            cwd=PROJECT_ROOT,
        )
        assert wheel_result.returncode == 0, wheel_result.stderr
        sdist_result = _run(
            ["uv", "build", "--sdist", "--no-sources", "--out-dir", direct],
            cwd=PROJECT_ROOT,
        )
        assert sdist_result.returncode == 0, sdist_result.stderr
        direct_wheel = _single(direct, "*.whl")
        sdist = _single(direct, "*.tar.gz")
    else:
        direct_wheel, sdist = external
    rebuilt_result = _run(
        ["uv", "build", "--wheel", "--no-sources", "--out-dir", rebuilt, sdist],
        cwd=PROJECT_ROOT,
    )
    assert rebuilt_result.returncode == 0, rebuilt_result.stderr
    return BuiltArtifacts(
        direct_wheel=direct_wheel,
        sdist=sdist,
        rebuilt_wheel=_single(rebuilt, "*.whl"),
    )


def test_external_release_candidate_mode_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    wheel = candidate / "pixipix-0.1.1-py3-none-any.whl"
    sdist = candidate / "pixipix-0.1.1.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    monkeypatch.setenv(EXTERNAL_RELEASE_CANDIDATE_ENV, str(candidate))

    assert _external_release_candidate() == (wheel.resolve(), sdist.resolve())

    extra = candidate / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(AssertionError, match="exactly two artifacts"):
        _external_release_candidate()


@pytest.mark.parametrize("case", ["empty", "relative", "missing"])
def test_external_release_candidate_mode_rejects_invalid_directory(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_directory = {
        "empty": "",
        "relative": "relative",
        "missing": str(tmp_path / "missing"),
    }[case]
    monkeypatch.setenv(EXTERNAL_RELEASE_CANDIDATE_ENV, raw_directory)

    with pytest.raises(AssertionError):
        _external_release_candidate()


def test_external_release_candidate_mode_rejects_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EXTERNAL_RELEASE_CANDIDATE_ENV, str(PROJECT_ROOT))

    with pytest.raises(AssertionError, match="outside the source checkout"):
        _external_release_candidate()


def test_external_release_candidate_mode_rejects_symlinked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "pixipix.whl").write_bytes(b"wheel")
    (candidate / "pixipix.tar.gz").write_bytes(b"sdist")
    symlink = tmp_path / "candidate-symlink"
    symlink.symlink_to(candidate, target_is_directory=True)
    monkeypatch.setenv(EXTERNAL_RELEASE_CANDIDATE_ENV, str(symlink))

    with pytest.raises(AssertionError, match="must not be a symlink"):
        _external_release_candidate()


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("pixipix.zip", "pixipix.tar.gz"), "exactly one wheel"),
        (
            ("pixipix-a.whl", "pixipix-b.whl", "pixipix.tar.gz"),
            "exactly two artifacts",
        ),
    ],
)
def test_external_release_candidate_mode_rejects_ambiguous_artifacts(
    names: tuple[str, ...],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in names:
        (candidate / name).write_bytes(b"artifact")
    monkeypatch.setenv(EXTERNAL_RELEASE_CANDIDATE_ENV, str(candidate))

    with pytest.raises(AssertionError, match=message):
        _external_release_candidate()


def test_distributions_ship_the_corrected_determinism_contract(
    built_artifacts: BuiltArtifacts,
) -> None:
    packaged_texts: list[str] = []
    for wheel in (built_artifacts.direct_wheel, built_artifacts.rebuilt_wheel):
        with zipfile.ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            assert len(metadata) == 1
            packaged_texts.append(archive.read(metadata[0]).decode("utf-8"))
    with tarfile.open(built_artifacts.sdist, "r:gz") as archive:
        pkg_info = [member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")]
        assert len(pkg_info) == 1
        root_readme_name = str(PurePosixPath(pkg_info[0].name).parent / "README.md")
        readme = [member for member in archive.getmembers() if member.name == root_readme_name]
        assert len(readme) == 1
        packaged_pkg_info = archive.extractfile(pkg_info[0])
        packaged_readme = archive.extractfile(readme[0])
        assert packaged_pkg_info is not None and packaged_readme is not None
        packaged_texts.extend(
            [
                packaged_pkg_info.read().decode("utf-8"),
                packaged_readme.read().decode("utf-8"),
            ]
        )

    for text in packaged_texts:
        normalized = " ".join(text.split())
        assert all(snippet in normalized for snippet in DET_CONTRACT_SNIPPETS)


def _run_smoke(wheel: Path) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, SMOKE_SCRIPT, "--artifact", wheel], cwd=PROJECT_ROOT)


@dataclass(frozen=True, slots=True)
class InstalledCompatibilityResult:
    manifest: dict[str, object]
    module_paths: dict[str, str]
    modules_without_files: tuple[str, ...]
    environment: Path
    interpreter: Path
    working_directory: Path


@dataclass(frozen=True, slots=True)
class ArchivePolicyResult:
    member_count: int
    required_missing: tuple[str, ...]
    forbidden_matches: tuple[str, ...]
    source_tests_present: bool = False
    tests_helpers_present: bool = False


REQUIRED_FACADE_MEMBERS = frozenset(
    {
        "pixipix/__init__.py",
        "pixipix/stages/__init__.py",
        "pixipix/stages/io.py",
        "pixipix/stages/align/__init__.py",
        "pixipix/stages/extract/__init__.py",
        "pixipix/stages/pixelize/__init__.py",
        "pixipix/stages/scale/__init__.py",
    }
)
REQUIRED_SDIST_FACADE_MEMBERS = frozenset(f"src/{member}" for member in REQUIRED_FACADE_MEMBERS)
ALLOWED_FILELESS_PIXIPIX_MODULES: frozenset[str] = frozenset()


def _validate_installed_module_paths(
    module_paths: dict[str, str],
    modules_without_files: tuple[str, ...],
    environment: Path,
) -> None:
    assert module_paths, "installed compatibility inventory is empty"
    assert frozenset(modules_without_files) == ALLOWED_FILELESS_PIXIPIX_MODULES, (
        "installed compatibility inventory contains unclassified fileless modules: "
        f"{modules_without_files}"
    )
    environment_root = environment.resolve()
    checkout_root = PROJECT_ROOT.resolve()
    build_root = (PROJECT_ROOT / "build").resolve()
    dist_root = (PROJECT_ROOT / "dist").resolve()
    for module, raw_path in sorted(module_paths.items()):
        path = Path(raw_path).resolve()
        assert path.is_relative_to(environment_root), (
            f"module={module} actual path={path} expected root={environment_root} "
            "remediation=install every canonical owner inside the isolated environment"
        )
        assert not path.is_relative_to(checkout_root), module
        assert not path.is_relative_to(build_root), module
        assert not path.is_relative_to(dist_root), module


def _installed_compatibility_result(
    wheel: Path,
    root: Path,
) -> InstalledCompatibilityResult:
    environment = root / "venv"
    working_directory = root / "work"
    root.mkdir()
    working_directory.mkdir()
    sanitized = _sanitized_environment(environment)
    sanitized.pop("PYTHONHOME", None)
    sanitized.pop("PYTHONPATH", None)
    creation = _run(
        ["uv", "venv", "--python", sys.executable, environment],
        cwd=root,
        environment=sanitized,
    )
    assert creation.returncode == 0, creation.stderr
    interpreter, _console = _isolated_paths(environment)
    installation = _run(
        ["uv", "pip", "install", "--python", interpreter, wheel],
        cwd=root,
        environment=sanitized,
    )
    assert installation.returncode == 0, installation.stderr
    execution = _run(
        [
            interpreter,
            "-I",
            "-c",
            installed_compatibility_manifest_program(),
        ],
        cwd=working_directory,
        environment=sanitized,
    )
    assert execution.returncode == 0, execution.stderr
    parsed = json.loads(execution.stdout)
    assert isinstance(parsed, dict)
    manifest = parsed.get("manifest")
    module_paths = parsed.get("module_paths")
    modules_without_files = parsed.get("modules_without_files")
    assert isinstance(manifest, dict)
    assert isinstance(module_paths, dict)
    assert isinstance(modules_without_files, list)
    typed_paths = {str(module): str(path) for module, path in module_paths.items()}
    typed_without_files = tuple(sorted(str(module) for module in modules_without_files))
    _validate_installed_module_paths(typed_paths, typed_without_files, environment)
    return InstalledCompatibilityResult(
        manifest=cast(dict[str, object], manifest),
        module_paths=typed_paths,
        modules_without_files=typed_without_files,
        environment=environment,
        interpreter=interpreter,
        working_directory=working_directory,
    )


def _locked_installed_release_environment(
    wheel: Path,
    root: Path,
) -> InstalledCompatibilityResult:
    environment = root / "venv"
    working_directory = root / "work"
    root.mkdir()
    working_directory.mkdir()
    sanitized = _sanitized_environment(environment)
    sanitized.pop("PYTHONHOME", None)
    sanitized.pop("PYTHONPATH", None)
    sanitized["UV_PROJECT_ENVIRONMENT"] = str(environment)
    creation = _run(
        ["uv", "venv", "--python", sys.executable, environment],
        cwd=root,
        environment=sanitized,
    )
    assert creation.returncode == 0, creation.stderr
    interpreter, _console = _isolated_paths(environment)
    synchronization = _run(
        [
            "uv",
            "sync",
            "--locked",
            "--all-groups",
            "--no-install-project",
            "--active",
        ],
        cwd=PROJECT_ROOT,
        environment=sanitized,
    )
    assert synchronization.returncode == 0, synchronization.stderr
    installation = _run(
        ["uv", "pip", "install", "--python", interpreter, "--no-deps", wheel],
        cwd=root,
        environment=sanitized,
    )
    assert installation.returncode == 0, installation.stderr
    execution = _run(
        [interpreter, "-I", "-c", installed_compatibility_manifest_program()],
        cwd=working_directory,
        environment=sanitized,
    )
    assert execution.returncode == 0, execution.stderr
    parsed = json.loads(execution.stdout)
    assert isinstance(parsed, dict)
    manifest = parsed.get("manifest")
    module_paths = parsed.get("module_paths")
    modules_without_files = parsed.get("modules_without_files")
    assert isinstance(manifest, dict)
    assert isinstance(module_paths, dict)
    assert isinstance(modules_without_files, list)
    typed_paths = {str(module): str(path) for module, path in module_paths.items()}
    typed_without_files = tuple(sorted(str(module) for module in modules_without_files))
    _validate_installed_module_paths(typed_paths, typed_without_files, environment)
    return InstalledCompatibilityResult(
        manifest=cast(dict[str, object], manifest),
        module_paths=typed_paths,
        modules_without_files=typed_without_files,
        environment=environment,
        interpreter=interpreter,
        working_directory=working_directory,
    )


@pytest.fixture(scope="session")
def installed_compatibility_results(
    tmp_path_factory: pytest.TempPathFactory,
    built_artifacts: BuiltArtifacts,
) -> dict[str, InstalledCompatibilityResult]:
    root = tmp_path_factory.mktemp("installed-compatibility-manifests")
    return {
        artifact_name: _installed_compatibility_result(
            cast(Path, getattr(built_artifacts, artifact_name)),
            root / artifact_name,
        )
        for artifact_name in ("direct_wheel", "rebuilt_wheel")
    }


def _installed_alias_boundary_program() -> str:
    return (
        "import json, pathlib, pixipix, sys; "
        "from pixipix.pipeline.publication import validate_stage_output_target; "
        "module = pathlib.Path(pixipix.__file__).resolve(); "
        "environment = pathlib.Path(sys.argv[2]).resolve(); "
        "checkout = pathlib.Path(sys.argv[3]).resolve(); "
        "assert module.is_relative_to(environment); "
        "assert not module.is_relative_to(checkout); "
        "assert validate_stage_output_target.__module__ == 'pixipix.pipeline.publication'; "
        "root = pathlib.Path(sys.argv[1]); "
        "output = root / 'nonexistent-output'; "
        "assert not output.exists(); "
        "validate_stage_output_target(output, 'extract'); "
        "assert not output.exists(); "
        "print(json.dumps({'module': str(module), 'output': output.name}, sort_keys=True))"
    )


def test_installed_alias_case_matrix_is_exact() -> None:
    actual = {(case.artifact, case.alias) for case in INSTALLED_ALIAS_CASES}

    assert actual == EXPECTED_INSTALLED_ALIAS_MATRIX
    assert len(INSTALLED_ALIAS_CASES) == len(EXPECTED_INSTALLED_ALIAS_MATRIX)


@pytest.mark.parametrize(
    "case",
    INSTALLED_ALIAS_CASES,
    ids=lambda case: f"{case.artifact}-{case.alias}",
)
def test_installed_artifacts_accept_real_alias_nonexistent_output(
    installed_compatibility_results: dict[str, InstalledCompatibilityResult],
    case: InstalledAliasCase,
) -> None:
    if not _is_darwin_runtime():
        pytest.skip("real macOS installed alias boundary requires Darwin")
    alias = Path(f"/{case.alias}")
    canonical_alias = Path(f"/private/{case.alias}")
    if (
        not alias.is_symlink()
        or alias.readlink() != Path(f"private/{case.alias}")
        or not canonical_alias.is_dir()
        or not os.path.samefile(alias, canonical_alias)
    ):
        pytest.skip(f"host does not expose verified {alias} system alias")
    base = Path(tempfile.gettempdir()) if case.alias == "var" else alias
    lexical = Path(tempfile.mkdtemp(prefix="pixipix-installed-alias-", dir=base))
    try:
        assert alias in (lexical, *lexical.parents)
        installed = installed_compatibility_results[case.artifact]
        result = _run(
            [
                installed.interpreter,
                "-I",
                "-c",
                _installed_alias_boundary_program(),
                lexical,
                installed.environment,
                PROJECT_ROOT,
            ],
            cwd=installed.working_directory,
            environment=_sanitized_environment(installed.environment),
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["output"] == "nonexistent-output"
        module = Path(payload["module"])
        assert module.is_relative_to(installed.environment.resolve())
        assert not module.is_relative_to(PROJECT_ROOT.resolve())
        assert not (lexical / "nonexistent-output").exists()
    finally:
        shutil.rmtree(lexical, ignore_errors=True)


def _wheel_members(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        return tuple(sorted(archive.namelist()))


def _sdist_members(sdist: Path) -> tuple[str, ...]:
    with tarfile.open(sdist, "r:gz") as archive:
        return tuple(sorted(archive.getnames()))


def _normalize_archive_member(member: str) -> tuple[str, str | None]:
    if member in {"", "."}:
        return member, "malformed path: empty member"
    if "\\" in member:
        return member, "malformed path: backslash separator"
    if member.startswith("/"):
        return member, "malformed path: absolute member"
    if len(member) >= 2 and member[0].isalpha() and member[1] == ":":
        return member, "malformed path: drive-prefixed member"
    normalized = member
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = tuple(part for part in normalized.split("/") if part)
    if ".." in parts:
        return member, "malformed path: parent traversal"
    return "/".join(parts), None


def _wheel_forbidden_reason(member: str) -> str | None:
    normalized, malformed = _normalize_archive_member(member)
    if malformed is not None:
        return malformed
    parts = tuple(part for part in normalized.split("/") if part)
    lowered = tuple(part.lower() for part in parts)
    filename = lowered[-1] if lowered else ""
    if "tests" in lowered:
        return "tests package"
    if filename == "tests.py":
        return "test module"
    if filename == "test_helpers.py" or "testing" in lowered:
        return "test helper"
    if filename == "agents.md" or "docs-internal" in lowered:
        return "local authority"
    if filename == "slice-12-input.md" or any(
        part in {"acceptance", "acceptance-evidence"} for part in lowered
    ):
        return "acceptance evidence"
    if any(part in {"reports", "local-reports"} for part in lowered):
        return "internal report"
    if any(
        part in {"build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in lowered
    ):
        return "build or cache residue"
    if filename.endswith((".pyc", ".pyo")) or filename == ".ds_store":
        return "generated residue"
    if any(part in {"smoke-output", "pixi-output", "temporary-output"} for part in lowered):
        return "temporary output"
    return None


def _relative_sdist_member(member: str) -> str:
    _prefix, separator, relative = member.partition("/")
    return relative if separator else ""


def _sdist_forbidden_reason(relative: str) -> str | None:
    normalized, malformed = _normalize_archive_member(relative)
    if malformed is not None:
        return malformed
    parts = tuple(part for part in normalized.split("/") if part)
    lowered = tuple(part.lower() for part in parts)
    filename = lowered[-1] if lowered else ""
    if filename == "agents.md" or "docs-internal" in lowered:
        return "local authority"
    if filename == "slice-12-input.md" or any(
        part in {"acceptance", "acceptance-evidence"} for part in lowered
    ):
        return "acceptance evidence"
    if ".venv" in lowered or any(
        part in {"build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in lowered
    ):
        return "build or cache residue"
    if "local-reports" in lowered or ("reports" in lowered and "internal" in lowered):
        return "local report"
    if filename.endswith((".pyc", ".pyo")) or filename == ".ds_store":
        return "generated residue"
    if any(part in {"smoke-output", "pixi-output", "temporary-output"} for part in lowered):
        return "temporary output"
    return None


def _validate_wheel_member_policy(
    members: tuple[str, ...],
    artifact: str,
) -> ArchivePolicyResult:
    normalized_members = tuple(
        normalized
        for member in members
        for normalized, malformed in [_normalize_archive_member(member)]
        if malformed is None
    )
    member_set = frozenset(normalized_members)
    missing = tuple(sorted(REQUIRED_FACADE_MEMBERS - member_set))
    forbidden = tuple(
        sorted(
            f"{member}: {_wheel_forbidden_reason(member)}"
            for member in members
            if _wheel_forbidden_reason(member) is not None
        )
    )
    errors: list[str] = []
    if missing:
        errors.append(
            f"artifact={artifact} member={missing[0]} actual=missing expected=present "
            "remediation=restore required facade packaging"
        )
    if forbidden:
        errors.append(
            f"artifact={artifact} member={forbidden[0]} actual=forbidden expected=absent "
            "remediation=exclude test/local/generated installed content"
        )
    if errors:
        raise AssertionError("\n".join(errors))
    return ArchivePolicyResult(len(members), missing, forbidden)


def _validate_sdist_member_policy(members: tuple[str, ...]) -> ArchivePolicyResult:
    normalized_members = tuple(
        normalized
        for member in members
        for normalized, malformed in [_normalize_archive_member(member)]
        if malformed is None
    )
    malformed_members = tuple(
        sorted(
            f"{member}: {malformed}"
            for member in members
            for _normalized, malformed in [_normalize_archive_member(member)]
            if malformed is not None
        )
    )
    relatives = tuple(sorted(_relative_sdist_member(member) for member in normalized_members))
    relative_set = frozenset(relatives)
    missing = tuple(sorted(REQUIRED_SDIST_FACADE_MEMBERS - relative_set))
    policy_forbidden = tuple(
        f"{member}: {_sdist_forbidden_reason(member)}"
        for member in relatives
        if _sdist_forbidden_reason(member) is not None
    )
    forbidden = tuple(sorted((*malformed_members, *policy_forbidden)))
    errors: list[str] = []
    if missing:
        errors.append(
            f"artifact=sdist member={missing[0]} actual=missing expected=present "
            "remediation=restore required facade source"
        )
    if forbidden:
        errors.append(
            f"artifact=sdist member={forbidden[0]} actual=forbidden expected=absent "
            "remediation=exclude local/generated source content"
        )
    if errors:
        raise AssertionError("\n".join(errors))
    return ArchivePolicyResult(
        member_count=len(members),
        required_missing=missing,
        forbidden_matches=forbidden,
        source_tests_present=any(
            member == "tests" or member.startswith("tests/") for member in relatives
        ),
        tests_helpers_present="tests/helpers.py" in relative_set,
    )


def _validate_manifest_equality(
    expected: dict[str, object],
    actual: dict[str, object],
    context: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"context={context} symbol=manifest actual=differs expected=exact equality "
            "remediation=restore installed compatibility surface"
        )


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_installed_compatibility_manifest_matches_checkout(
    installed_compatibility_results: dict[str, InstalledCompatibilityResult],
    artifact_name: str,
) -> None:
    checkout = build_checkout_compatibility_manifest()
    installed = installed_compatibility_results[artifact_name]

    _validate_manifest_equality(checkout, installed.manifest, artifact_name)

    assert installed.module_paths
    assert all(str(PROJECT_ROOT.resolve()) not in path for path in installed.module_paths.values())


def test_three_context_compatibility_manifests_are_exact(
    installed_compatibility_results: dict[str, InstalledCompatibilityResult],
) -> None:
    checkout = build_checkout_compatibility_manifest()
    direct = installed_compatibility_results["direct_wheel"].manifest
    rebuilt = installed_compatibility_results["rebuilt_wheel"].manifest

    assert checkout == direct == rebuilt


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("rule", "identity", "field=rule"),
        ("kind", "singleton", "field=kind"),
        (
            "final_owner",
            ".".join(("pixipix", "stages", "scale", "execution")),
            "field=final_owner",
        ),
    ],
)
def test_installed_environment_rejects_false_semantic_payload(
    installed_compatibility_results: dict[str, InstalledCompatibilityResult],
    field: str,
    value: str,
    diagnostic: str,
) -> None:
    installed = installed_compatibility_results["direct_wheel"]
    payload = json.loads(json.dumps(compatibility_contract_payload()))
    scale = next(
        surface for surface in payload["facades"] if surface["module"] == "pixipix.stages.scale"
    )
    target = next(
        export for export in scale["exports"] if export["name"] == "MAX_TRANSFORMED_PIXELS"
    )
    target[field] = value
    sanitized = _sanitized_environment(installed.environment)
    sanitized.pop("PYTHONHOME", None)
    sanitized.pop("PYTHONPATH", None)

    execution = _run(
        [
            installed.interpreter,
            "-I",
            "-c",
            installed_compatibility_manifest_program(cast(dict[str, object], payload)),
        ],
        cwd=installed.working_directory,
        environment=sanitized,
    )

    assert execution.returncode != 0
    assert diagnostic in execution.stderr


def test_installed_environment_rejects_false_posture_payload(
    installed_compatibility_results: dict[str, InstalledCompatibilityResult],
) -> None:
    installed = installed_compatibility_results["direct_wheel"]
    payload = json.loads(json.dumps(compatibility_contract_payload()))
    root = next(entry for entry in payload["expected_postures"] if entry[0] == "pixipix")
    root[1] = "absent-by-design"
    sanitized = _sanitized_environment(installed.environment)
    sanitized.pop("PYTHONHOME", None)
    sanitized.pop("PYTHONPATH", None)

    execution = _run(
        [
            installed.interpreter,
            "-I",
            "-c",
            installed_compatibility_manifest_program(cast(dict[str, object], payload)),
        ],
        cwd=installed.working_directory,
        environment=sanitized,
    )

    assert execution.returncode != 0
    assert "field=posture" in execution.stderr


def test_installed_module_inventory_rejects_external_owner(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    installed_module = environment / "lib" / "python" / "site-packages" / "pixipix" / "__init__.py"
    external_owner = tmp_path / "outside" / "pixipix" / "planning.py"
    installed_module.parent.mkdir(parents=True)
    external_owner.parent.mkdir(parents=True)
    installed_module.touch()
    external_owner.touch()

    external_binding = ".".join(("pixipix", "stages", "scale", "planning"))
    with pytest.raises(AssertionError, match=external_binding):
        _validate_installed_module_paths(
            {
                "pixipix": str(installed_module),
                external_binding: str(external_owner),
            },
            (),
            environment,
        )


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_complete_wheel_member_policy(
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = cast(Path, getattr(built_artifacts, artifact_name))

    result = _validate_wheel_member_policy(_wheel_members(wheel), artifact_name)

    assert result.member_count > len(REQUIRED_FACADE_MEMBERS)
    assert result.required_missing == ()
    assert result.forbidden_matches == ()


def test_fresh_sdist_member_policy(built_artifacts: BuiltArtifacts) -> None:
    result = _validate_sdist_member_policy(_sdist_members(built_artifacts.sdist))

    assert result.member_count > len(REQUIRED_FACADE_MEMBERS)
    assert result.required_missing == ()
    assert result.forbidden_matches == ()
    assert result.source_tests_present


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("direct-missing-facade", "artifact=direct_wheel"),
        ("rebuilt-missing-facade", "artifact=rebuilt_wheel"),
        ("checkout-only-surface", "context=direct_wheel"),
        ("owner-map-disagreement", "context=rebuilt_wheel"),
        ("wheel-top-level-tests", "tests package"),
        ("wheel-nested-tests", "tests package"),
        ("wheel-test-helper", "test helper"),
        ("wheel-local-authority", "local authority"),
        ("sdist-local-authority", "local authority"),
        ("sdist-generated-content", "build or cache residue"),
    ],
)
def test_packaging_contract_rejects_upstream_mutation(
    mutation: str,
    diagnostic: str,
) -> None:
    baseline_wheel = tuple(sorted(REQUIRED_FACADE_MEMBERS | {"pixipix-0.1.dist-info/METADATA"}))
    baseline_sdist = tuple(
        sorted(
            f"pixipix-0.1/{member}"
            for member in REQUIRED_SDIST_FACADE_MEMBERS | {"tests/helpers.py", "pyproject.toml"}
        )
    )
    checkout = build_checkout_compatibility_manifest()
    with pytest.raises(AssertionError) as raised:
        if mutation in {"direct-missing-facade", "rebuilt-missing-facade"}:
            members = tuple(member for member in baseline_wheel if member != "pixipix/stages/io.py")
            artifact = "direct_wheel" if mutation.startswith("direct") else "rebuilt_wheel"
            _validate_wheel_member_policy(members, artifact)
        elif mutation in {"checkout-only-surface", "owner-map-disagreement"}:
            altered = json.loads(json.dumps(checkout))
            surfaces = altered["surfaces"]
            assert isinstance(surfaces, list)
            if mutation == "checkout-only-surface":
                surfaces.pop()
                context = "direct_wheel"
            else:
                first_facade = next(
                    surface for surface in surfaces if surface["explicit_same_name_exports"]
                )
                first_facade["explicit_same_name_exports"][0]["canonical_final_owner"] = (
                    "pixipix.pipeline.artifacts"
                )
                context = "rebuilt_wheel"
            _validate_manifest_equality(checkout, altered, context)
        elif mutation.startswith("wheel-"):
            bad_member = {
                "wheel-top-level-tests": "tests/test_surface.py",
                "wheel-nested-tests": "pixipix/tests/test_surface.py",
                "wheel-test-helper": "pixipix/test_helpers.py",
                "wheel-local-authority": "pixipix/docs-internal/report.md",
            }[mutation]
            _validate_wheel_member_policy(
                tuple(sorted((*baseline_wheel, bad_member))),
                "direct_wheel",
            )
        else:
            bad_member = {
                "sdist-local-authority": "pixipix-0.1/AGENTS.md",
                "sdist-generated-content": "pixipix-0.1/build/generated.py",
            }[mutation]
            _validate_sdist_member_policy(tuple(sorted((*baseline_sdist, bad_member))))

    message = str(raised.value)
    assert diagnostic in message
    assert "actual=" in message
    assert "expected=" in message
    assert "remediation=" in message


@pytest.mark.parametrize(
    ("member", "expected_reason"),
    [
        (r"tests\helper.py", "malformed path: backslash separator"),
        (r"pixipix\tests\helper.py", "malformed path: backslash separator"),
        ("pixipix/../tests/helpers.py", "malformed path: parent traversal"),
        ("/absolute/tests/helpers.py", "malformed path: absolute member"),
        (r"C:" + r"\tests\helpers.py", "malformed path: backslash separator"),
        ("tests/helpers.py", "tests package"),
        ("pkg/tests/helpers.py", "tests package"),
        ("pixipix/tests.py", "test module"),
        ("pixipix/test_helpers.py", "test helper"),
        ("pixipix/testing/helpers.py", "test helper"),
    ],
)
def test_wheel_policy_rejects_malformed_and_test_support_paths(
    member: str,
    expected_reason: str,
) -> None:
    assert _wheel_forbidden_reason(member) == expected_reason


@pytest.mark.parametrize(
    "member",
    [
        "pixipix/contest.py",
        "pixipix/latest.py",
        "pixipix/attestation.py",
    ],
)
def test_wheel_policy_allows_legitimate_test_substrings(member: str) -> None:
    assert _wheel_forbidden_reason(member) is None


def test_wheel_policy_collapses_harmless_leading_current_directory() -> None:
    assert _wheel_forbidden_reason("./tests/helpers.py") == "tests package"


@pytest.mark.parametrize(
    ("relative", "expected_reason"),
    [
        ("AGENTS.md", "local authority"),
        ("docs-internal/report.md", "local authority"),
        ("SLICE-12-INPUT.md", "acceptance evidence"),
        ("acceptance-evidence/report.json", "acceptance evidence"),
        ("acceptance/report.json", "acceptance evidence"),
        ("local-reports/report.md", "local report"),
        ("reports/internal/report.md", "local report"),
        (".venv/file", "build or cache residue"),
        ("build/output", "build or cache residue"),
        ("dist/package.whl", "build or cache residue"),
        ("src/pixipix/__pycache__/module.pyc", "build or cache residue"),
        (".DS_Store", "generated residue"),
    ],
)
def test_sdist_policy_rejects_bounded_local_generated_paths(
    relative: str,
    expected_reason: str,
) -> None:
    assert _sdist_forbidden_reason(relative) == expected_reason


@pytest.mark.parametrize(
    "relative",
    [
        "docs/report.md",
        "tests/helpers.py",
        "tests/test_surface.py",
    ],
)
def test_sdist_policy_allows_public_docs_and_source_tests(relative: str) -> None:
    assert _sdist_forbidden_reason(relative) is None


def test_archive_policy_rejects_correction_regression_members() -> None:
    baseline_wheel = tuple(sorted(REQUIRED_FACADE_MEMBERS | {"pixipix-0.1.dist-info/METADATA"}))
    baseline_sdist = tuple(
        sorted(
            f"pixipix-0.1/{member}"
            for member in REQUIRED_SDIST_FACADE_MEMBERS | {"tests/helpers.py", "pyproject.toml"}
        )
    )
    attacks = (
        ("wheel", r"tests\helper.py", "malformed path"),
        ("wheel", "pixipix/../tests/helpers.py", "parent traversal"),
        ("wheel", "pixipix/tests.py", "test module"),
        ("sdist", "pixipix-0.1/acceptance-evidence/report.json", "acceptance evidence"),
        ("sdist", "pixipix-0.1/reports/internal/report.md", "local report"),
    )
    for artifact, member, diagnostic in attacks:
        with pytest.raises(AssertionError, match=diagnostic):
            if artifact == "wheel":
                _validate_wheel_member_policy((*baseline_wheel, member), "direct_wheel")
            else:
                _validate_sdist_member_policy((*baseline_sdist, member))


@pytest.mark.parametrize("member", ["", "."])
def test_archive_normalization_rejects_empty_members(member: str) -> None:
    assert _normalize_archive_member(member) == (
        member,
        "malformed path: empty member",
    )


@pytest.mark.parametrize(
    ("artifact", "member"),
    [
        ("wheel", ""),
        ("wheel", "."),
        ("sdist", ""),
        ("sdist", "."),
    ],
)
def test_complete_archive_policy_rejects_empty_members(
    artifact: str,
    member: str,
) -> None:
    baseline_wheel = tuple(sorted(REQUIRED_FACADE_MEMBERS | {"pixipix-0.1.dist-info/METADATA"}))
    baseline_sdist = tuple(
        sorted(
            f"pixipix-0.1/{entry}"
            for entry in REQUIRED_SDIST_FACADE_MEMBERS | {"tests/helpers.py", "pyproject.toml"}
        )
    )

    with pytest.raises(AssertionError, match="malformed path: empty member"):
        if artifact == "wheel":
            _validate_wheel_member_policy((*baseline_wheel, member), "direct_wheel")
        else:
            _validate_sdist_member_policy((*baseline_sdist, member))


def test_complete_sdist_policy_preserves_public_documentation() -> None:
    members = tuple(
        sorted(
            f"pixipix-0.1/{entry}"
            for entry in REQUIRED_SDIST_FACADE_MEMBERS
            | {"docs/report.md", "tests/helpers.py", "pyproject.toml"}
        )
    )

    result = _validate_sdist_member_policy(members)

    assert result.forbidden_matches == ()


def _write_final_stage(root: Path) -> None:
    frames = root / "frames"
    frames.mkdir(parents=True)
    (root / ".pixipix-output").write_text(
        json.dumps({"owner": "pixipix", "schemaVersion": 1, "stage": "align"}),
        encoding="utf-8",
    )
    metadata_frames: list[dict[str, object]] = []
    for order, (name, relative_path) in enumerate(
        zip(FIXTURE_CONTRACT.frame_names, FIXTURE_CONTRACT.frame_paths, strict=True)
    ):
        Image.new("RGBA", (4, 4)).save(root / relative_path, format="PNG")
        metadata_frames.append(
            {
                "name": name,
                "sourceOrder": order,
                "relativePath": relative_path,
                "outputWidth": 4,
                "outputHeight": 4,
                "clipped": False,
            }
        )
    metadata = {
        "schemaVersion": 1,
        "stage": "align",
        "status": "successful",
        "canvasWidth": 4,
        "canvasHeight": 4,
        "warnings": [],
        "clippingFindings": [],
        "frames": metadata_frames,
    }
    (root / "stage.json").write_text(json.dumps(metadata), encoding="utf-8")


def _record_hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _validate_prior_stage_artifacts(output: Path, stage: str) -> None:
    assert json.loads((output / ".pixipix-output").read_text(encoding="utf-8")) == {
        "owner": "pixipix",
        "schemaVersion": 1,
        "stage": stage,
    }
    metadata = json.loads((output / "stage.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == stage
    assert metadata["status"] == "successful"
    assert tuple(frame["name"] for frame in metadata["frames"]) == FIXTURE_CONTRACT.frame_names
    assert (
        tuple(frame["relativePath"] for frame in metadata["frames"]) == FIXTURE_CONTRACT.frame_paths
    )
    for relative_path in FIXTURE_CONTRACT.frame_paths:
        with Image.open(output / relative_path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGBA"


def _run_corrupted_installed_pipeline(
    wheel: Path,
    root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = root / "venv"
    working_directory = root / "work"
    fixture_directory = root / "fixture"
    root.mkdir()
    working_directory.mkdir()
    fixture_directory.mkdir()
    for filename in ("robot-geometric.png", "robot-geometric.toml"):
        shutil.copy2(PROJECT_ROOT / "tests" / "fixtures" / filename, fixture_directory / filename)

    sanitized = _sanitized_environment(environment)
    sanitized.pop("PYTHONHOME", None)
    creation = _run(
        ["uv", "venv", "--python", sys.executable, environment],
        cwd=root,
        environment=sanitized,
    )
    assert creation.returncode == 0, creation.stderr
    interpreter, console = _isolated_paths(environment)
    installation = _run(
        ["uv", "pip", "install", "--python", interpreter, wheel],
        cwd=root,
        environment=sanitized,
    )
    assert installation.returncode == 0, installation.stderr

    imported = _run(
        [
            interpreter,
            "-c",
            (
                "import pathlib; import pixipix.stages.align as align; "
                "print(pathlib.Path(align.__file__).resolve())"
            ),
        ],
        cwd=working_directory,
        environment=sanitized,
    )
    assert imported.returncode == 0, imported.stderr
    assert str(environment.resolve()) in imported.stdout
    assert str(PROJECT_ROOT.resolve()) not in imported.stdout

    help_result = _run([console, "--help"], cwd=working_directory, environment=sanitized)
    assert help_result.returncode == 0, help_result.stderr
    assert "Tiny poses in. Tidy pixels out." in help_result.stdout

    image = fixture_directory / "robot-geometric.png"
    config = fixture_directory / "robot-geometric.toml"
    output_root = working_directory / "smoke-output"
    outputs = {
        "extract": output_root / "extracted",
        "scale": output_root / "scaled",
        "pixelize": output_root / "pixelized",
        "align": output_root / "aligned",
    }
    inspect_result = _run(
        [console, "inspect", image, "--config", config],
        cwd=working_directory,
        environment=sanitized,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr

    source = image
    for stage in ("extract", "scale", "pixelize"):
        output = outputs[stage]
        result = _run(
            [console, stage, source, "--config", config, "--output", output],
            cwd=working_directory,
            environment=sanitized,
        )
        assert result.returncode == 0, result.stderr
        _validate_prior_stage_artifacts(output, stage)
        source = output

    result = _run(
        [console, "align", source, "--config", config, "--output", outputs["align"]],
        cwd=working_directory,
        environment=sanitized,
    )
    assert not outputs["align"].exists()
    return result


def _corrupt_align_implementation(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as wheel:
        infos = wheel.infolist()
        files = {info.filename: wheel.read(info) for info in infos}
    align_path = "pixipix/stages/align/execution.py"
    original = files[align_path]
    needle = b"    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:\n"
    replacement = (
        b'    raise RuntimeError("simulated installed align execution corruption")\n' + needle
    )
    assert original.count(needle) == 1
    files[align_path] = original.replace(needle, replacement, 1)

    record_path = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(files[record_path].decode("utf-8"))))
    for row in rows:
        if row[0] == align_path:
            row[1] = _record_hash(files[align_path])
            row[2] = str(len(files[align_path]))
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[record_path] = stream.getvalue().encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as wheel:
        for info in infos:
            wheel.writestr(info, files[info.filename])


def test_fixture_contract_is_static_and_configuration_reaches_align() -> None:
    scenario = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.SCENARIO.md"
    config_path = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert scenario.is_file()
    assert FIXTURE_CONTRACT.candidate_count == 2
    assert FIXTURE_CONTRACT.accepted_count == 2
    assert FIXTURE_CONTRACT.rejected_count == 0
    assert FIXTURE_CONTRACT.frame_names == ("idle", "signal")
    assert FIXTURE_CONTRACT.frame_paths == ("frames/idle.png", "frames/signal.png")
    assert FIXTURE_CONTRACT.final_stage == "align"
    assert FIXTURE_CONTRACT.schema_version == 1
    assert (FIXTURE_CONTRACT.canvas_width, FIXTURE_CONTRACT.canvas_height) == (4, 4)
    assert FIXTURE_CONTRACT.png_mode == "RGBA"
    assert FIXTURE_CONTRACT.warning_codes == ()
    assert config["scale"] == {"mode": "explicit-factor", "factor": 1.0}
    assert config["pixelize"]["source_cell_size"] == 4
    assert config["pixelize"]["remainder_policy"] == "pad-transparent"
    assert config["output"] == {
        "frame_width": 4,
        "frame_height": 4,
        "anchor": "center",
        "clip_policy": "error",
    }


def test_canonical_smoke_sequence_is_complete_and_ordered() -> None:
    assert SMOKE_STAGES == ("inspect", "extract", "scale", "pixelize", "align")


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_extract_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    stages_source = PROJECT_ROOT / "src" / "pixipix" / "stages" / "__init__.py"
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "extract"
    expected_members = {
        f"pixipix/stages/extract/{name}": source_root / name
        for name in (
            "__init__.py",
            "analysis.py",
            "api.py",
            "execution.py",
            "metadata.py",
            "planning.py",
            "publication.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert "pixipix/stages/__init__.py" in members
        assert {member for member in members if member.startswith("pixipix/stages/extract")} == set(
            expected_members
        )
        assert archive.read("pixipix/stages/__init__.py") == stages_source.read_bytes()
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/extract.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_extract_first_import_preserves_compatibility_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / f"{artifact_name}-extract-import"
    working_directory.mkdir()
    code = (
        "import inspect, pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix.stages.extract as extract; "
        "import pixipix.stages.extract.analysis as analysis; "
        "import pixipix.stages.extract.api as api; "
        "import pixipix.stages.extract.execution as execution; "
        "import pixipix.stages.extract.metadata as metadata; "
        "import pixipix.stages.extract.planning as planning; "
        "import pixipix.stages.extract.publication as publication; "
        "import pixipix.pipeline.publication as pipeline_publication; "
        "import pixipix.models as models; "
        "import pixipix.resources as resources; "
        "expected = ("
        "'ComponentMap', 'label_components', 'filter_components', 'order_components', "
        "'inspect_source', 'extract_source', 'project_extract_resources', "
        "'project_extracted_frames', 'publish_extraction'); "
        "assert all(hasattr(extract, name) for name in expected); "
        "assert pathlib.Path(extract.__file__).name == '__init__.py'; "
        "assert extract.__spec__.submodule_search_locations is not None; "
        "assert extract.ComponentMap is analysis.ComponentMap; "
        "assert extract.label_components is analysis.label_components; "
        "assert extract.filter_components is analysis.filter_components; "
        "assert extract.order_components is analysis.order_components; "
        "assert extract.inspect_source is api.inspect_source; "
        "assert extract.extract_source is api.extract_source; "
        "assert extract.project_extract_resources is planning.project_extract_resources; "
        "assert extract.project_extracted_frames is planning.project_extracted_frames; "
        "assert extract.publish_extraction is publication.publish_extraction; "
        "assert analysis.ComponentMap.__module__ == 'pixipix.stages.extract.analysis'; "
        "assert analysis._Analysis.__module__ == 'pixipix.stages.extract.analysis'; "
        "assert api.inspect_source.__module__ == 'pixipix.stages.extract.api'; "
        "assert api.extract_source.__module__ == 'pixipix.stages.extract.api'; "
        "assert planning.project_extract_resources.__module__ "
        "== 'pixipix.stages.extract.planning'; "
        "assert publication.publish_extraction.__module__ "
        "== 'pixipix.stages.extract.publication'; "
        "assert api.InspectionResult is models.InspectionResult; "
        "assert api.ExtractionRun is models.ExtractionRun; "
        "assert api.ExtractionResult is models.ExtractionResult; "
        "assert execution.ExtractedFrame is models.ExtractedFrame; "
        "assert execution.FrameImage is models.FrameImage; "
        "assert planning.ExtractedFrame is models.ExtractedFrame; "
        "assert planning.ResourceProjection is resources.ResourceProjection; "
        "assert callable(metadata._stage_metadata); "
        "assert callable(metadata._valid_owned_extract_metadata); "
        "assert callable(pipeline_publication._valid_owned_output); "
        "assert publication.publish_stage_output is "
        "pipeline_publication.publish_stage_output; "
        "assert publication.validate_stage_output_target is "
        "pipeline_publication.validate_stage_output_target; "
        "assert not hasattr(publication, '_valid_owned_output'); "
        "assert not hasattr(publication, '_valid_frame_png'); "
        "assert not hasattr(publication, '_validate_staged_output'); "
        "assert not hasattr(publication, '_validate_output_location'); "
        "assert not hasattr(publication, '_prepare_target'); "
        "assert not hasattr(publication, '_remove_temporary_tree'); "
        "assert not hasattr(extract, '_Analysis'); "
        "assert not hasattr(extract, '_analyze'); "
        "assert not hasattr(extract, '_padded_bounds'); "
        "assert not hasattr(extract, '_materialize_frame_crop'); "
        "assert not hasattr(extract, '_valid_frame_png'); "
        "assert not hasattr(extract, '_validate_staged_output'); "
        "assert not hasattr(extract, '_validate_output_location'); "
        "assert not hasattr(extract, 'np'); "
        "assert not hasattr(extract, 'Image'); "
        "assert not hasattr(extract, 'load_source'); "
        "assert not hasattr(extract, 'write_png'); "
        "assert str(inspect.signature(extract.publish_extraction)) == "
        "\"(input_path: 'Path', loaded: 'LoadedConfig', output: 'Path', "
        "*, force: 'bool' = False) -> 'ExtractionResult'\"; "
        "assert sys.modules['pixipix.stages.extract'] is extract; "
        "assert 'pixipix.stages.extract.__init__' not in sys.modules; "
        "print(pathlib.Path(extract.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_align_package_member_only(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    expected_align_members = {
        "pixipix/stages/align/__init__.py",
        "pixipix/stages/align/api.py",
        "pixipix/stages/align/execution.py",
        "pixipix/stages/align/geometry.py",
        "pixipix/stages/align/planning.py",
    }
    assert {member for member in members if member.startswith("pixipix/stages/align")} == (
        expected_align_members
    )
    assert "pixipix/stages/align.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_scale_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "scale"
    expected_members = {
        f"pixipix/stages/scale/{name}": source_root / name
        for name in (
            "__init__.py",
            "api.py",
            "execution.py",
            "geometry.py",
            "metadata.py",
            "planning.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {member for member in members if member.startswith("pixipix/stages/scale")} == set(
            expected_members
        )
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/scale.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_pixelize_package_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    source_root = PROJECT_ROOT / "src" / "pixipix" / "stages" / "pixelize"
    expected_members = {
        f"pixipix/stages/pixelize/{name}": source_root / name
        for name in (
            "__init__.py",
            "api.py",
            "execution.py",
            "metadata.py",
            "planning.py",
        )
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {
            member for member in members if member.startswith("pixipix/stages/pixelize")
        } == set(expected_members)
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()
    assert "pixipix/stages/pixelize.py" not in members


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_scale_and_pixelize_imports_work_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / f"{artifact_name}-scale-import"
    working_directory.mkdir()
    code = (
        "import pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix._scale_geometry as foundational_geometry; "
        "import pixipix.pipeline.input as pipeline_input; "
        "import pixipix.stages.pixelize as pixelize; "
        "import pixipix.stages.pixelize.api as pixelize_api; "
        "import pixipix.stages.pixelize.execution as pixelize_execution; "
        "import pixipix.stages.pixelize.metadata as pixelize_metadata; "
        "import pixipix.stages.pixelize.planning as pixelize_planning; "
        "import pixipix.stages.scale as scale; "
        "import pixipix.stages.scale.api as api; "
        "import pixipix.stages.scale.execution as execution; "
        "import pixipix.stages.scale.geometry as geometry; "
        "import pixipix.stages.scale.metadata as metadata; "
        "import pixipix.stages.scale.planning as planning; "
        "assert pathlib.Path(pixelize.__file__).name == '__init__.py'; "
        "assert pixelize.__spec__.submodule_search_locations is not None; "
        "assert pixelize.publish_pixelize is pixelize_api.publish_pixelize; "
        "assert pixelize.PreparedCellGrid is pixelize_execution.PreparedCellGrid; "
        "assert pixelize.CellGridProjection is pixelize_planning.CellGridProjection; "
        "assert pixelize.PixelizeRun is pixelize_execution.PixelizeRun; "
        "assert pixelize.PixelizeStagePlan is pixelize_planning.PixelizeStagePlan; "
        "assert pixelize.project_cell_grid is pixelize_planning.project_cell_grid; "
        "assert pixelize.prepare_cell_grid is pixelize_execution.prepare_cell_grid; "
        "assert pixelize.representative_pixel is pixelize_execution.representative_pixel; "
        "assert pixelize.apply_alpha_policy is pixelize_execution.apply_alpha_policy; "
        "assert pixelize.pixelize_prepared_grid "
        "is pixelize_execution.pixelize_prepared_grid; "
        "assert pixelize.project_pixelize_resources "
        "is pixelize_planning.project_pixelize_resources; "
        "assert pixelize.project_pixelize_stage is pixelize_planning.project_pixelize_stage; "
        "assert pixelize.pixelize_stage is pixelize_execution.pixelize_stage; "
        "assert pixelize.MAX_PREPARED_PIXELS is pixelize_planning.MAX_PREPARED_PIXELS; "
        "assert pixelize.PixelizeStagePlan.__module__ "
        "== 'pixipix.stages.pixelize.planning'; "
        "assert pixelize.PixelizeRun.__module__ "
        "== 'pixipix.stages.pixelize.execution'; "
        "assert pixelize.publish_pixelize.__module__ == 'pixipix.stages.pixelize.api'; "
        "assert pixelize.round_channel_half_away_from_zero "
        "is pixelize_execution.round_channel_half_away_from_zero "
        "is scale.round_channel_half_away_from_zero; "
        "assert callable(pixelize_metadata.build_pixelize_metadata); "
        "assert not hasattr(pixelize, 'decode_stage_input'); "
        "assert not hasattr(pixelize, 'np'); "
        "assert not hasattr(pixelize, 'Image'); "
        "assert not hasattr(pixelize, '_require_pixelize_config'); "
        "assert not hasattr(pixelize, '_validate_config_handoff'); "
        "assert not hasattr(pixelize, 'build_pixelize_metadata'); "
        "assert scale.publish_scale is api.publish_scale; "
        "assert scale.ScaleRun is execution.ScaleRun; "
        "assert scale.scale_stage is execution.scale_stage; "
        "assert scale.premultiplied_box_resize is execution.premultiplied_box_resize; "
        "assert scale.ScaleStagePlan is planning.ScaleStagePlan; "
        "assert scale.MAX_TRANSFORMED_PIXELS is planning.MAX_TRANSFORMED_PIXELS; "
        "assert scale.project_scale_stage is planning.project_scale_stage; "
        "assert scale.project_scale_resources is planning.project_scale_resources; "
        "assert scale.round_half_away_from_zero is geometry.round_half_away_from_zero; "
        "assert scale.transformed_dimension is geometry.transformed_dimension; "
        "assert scale.round_half_away_from_zero "
        "is foundational_geometry.round_half_away_from_zero; "
        "assert scale.transformed_dimension is foundational_geometry.transformed_dimension; "
        "assert pipeline_input.transformed_dimension "
        "is foundational_geometry.transformed_dimension; "
        "assert planning.transformed_dimension is foundational_geometry.transformed_dimension; "
        "assert scale.round_channel_half_away_from_zero "
        "is geometry.round_channel_half_away_from_zero; "
        "assert callable(metadata.build_scale_metadata); "
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
        "assert sys.modules['pixipix.stages.pixelize'] is pixelize; "
        "assert 'pixipix.stages.pixelize.__init__' not in sys.modules; "
        "assert sys.modules['pixipix.stages.scale'] is scale; "
        "assert 'pixipix.stages.scale.__init__' not in sys.modules; "
        "print(pathlib.Path(scale.__file__).resolve()); "
        "print(pathlib.Path(pixelize.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_contains_exact_shared_pipeline_members(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    expected_members = {
        "pixipix/pipeline/__init__.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "__init__.py",
        "pixipix/pipeline/artifacts.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "artifacts.py",
        "pixipix/pipeline/input.py": PROJECT_ROOT / "src" / "pixipix" / "pipeline" / "input.py",
        "pixipix/pipeline/publication.py": PROJECT_ROOT
        / "src"
        / "pixipix"
        / "pipeline"
        / "publication.py",
        "pixipix/stages/io.py": PROJECT_ROOT / "src" / "pixipix" / "stages" / "io.py",
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert {member for member in members if member.startswith("pixipix/pipeline/")} == set(
            expected_members
        ) - {"pixipix/stages/io.py"}
        for member, source in expected_members.items():
            assert archive.read(member) == source.read_bytes()


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_wheel_shared_pipeline_imports_work_outside_checkout(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    working_directory = tmp_path / artifact_name
    working_directory.mkdir()
    code = (
        "import pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pixipix.pipeline.artifacts as artifacts; "
        "import pixipix.pipeline.input as pipeline_input; "
        "import pixipix.pipeline.publication as publication; "
        "import pixipix.stages.io as stage_io; "
        "assert stage_io.load_stage_input is pipeline_input.load_stage_input; "
        "assert stage_io._valid_owned_output is publication._valid_owned_output; "
        "assert pipeline_input._is_untrusted_path_component is "
        "artifacts._is_untrusted_path_component; "
        "assert publication._is_untrusted_path_component is "
        "artifacts._is_untrusted_path_component; "
        "assert not artifacts._is_untrusted_path_component(pathlib.Path('/ordinary')); "
        "print(pathlib.Path(pipeline_input.__file__).resolve()); "
        "print(pathlib.Path(publication.__file__).resolve()); "
        "print(pathlib.Path(stage_io.__file__).resolve())"
    )

    result = _run(
        [sys.executable, "-I", "-c", code, wheel.resolve()],
        cwd=working_directory,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel.resolve()) in result.stdout
    assert str(PROJECT_ROOT.resolve()) not in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_installed_artifact_runs_complete_pipeline(
    built_artifacts: BuiltArtifacts, artifact_name: str
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)

    result = _run_smoke(wheel)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert [
        line.removeprefix("distribution smoke completed stage ")
        for line in result.stdout.splitlines()
        if line.startswith("distribution smoke completed stage ")
    ] == list(SMOKE_STAGES)
    assert "resolved pixipix.__file__:" in result.stdout
    assert "installed module inside isolated environment: true" in result.stdout
    assert "installed module outside repository checkout: true" in result.stdout
    assert "installed production publication validation passed for align" in result.stdout
    assert "final aligned metadata and PNG validation passed" in result.stdout
    assert "installed CLI warning visibility validation passed" in result.stdout
    assert "installed resource default identity validation passed" in result.stdout
    assert "installed metadata-only resource refusal validation passed" in result.stdout
    assert "distribution smoke test passed for pixipix" in result.stdout


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_installed_artifact_matches_active_release_authority(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    external = _external_release_candidate()
    if external is not None:
        direct_wheel, sdist = external
        assert built_artifacts.direct_wheel == direct_wheel
        assert built_artifacts.sdist == sdist
        if artifact_name == "direct_wheel":
            assert wheel == direct_wheel
        else:
            assert wheel == built_artifacts.rebuilt_wheel
            assert wheel != direct_wheel
    installed = _locked_installed_release_environment(wheel, tmp_path / artifact_name)
    expected = load_release_baseline()
    actual_environment = capture_environment(
        PARITY_PROJECT_ROOT,
        python=installed.interpreter,
        import_root=installed.environment,
    )
    require_canonical_runtime(expected.get("environment"), actual_environment)

    actual = capture_behavior(
        PARITY_PROJECT_ROOT,
        tmp_path / artifact_name / "capture",
        python=installed.interpreter,
        import_root=installed.environment,
    )

    compare_behavior(expected, actual)


def test_installed_resource_smoke_contracts_are_safe_and_exact(tmp_path: Path) -> None:
    config = PROJECT_ROOT / "tests" / "fixtures" / "robot-geometric.toml"
    console = Path(sys.executable).with_name("pixipix")

    explicit = _validate_installed_resource_identity(config, tmp_path)
    _validate_installed_resource_refusal(
        console=console,
        working_directory=tmp_path,
    )

    assert explicit.name == "explicit-resources.toml"
    assert not (tmp_path / "resource-refusal-output").exists()
    assert (tmp_path / "resource-refusal-extract" / "frames" / "ceiling.png").stat().st_size < 64


def test_resource_refusal_fixture_uses_explicit_policy_a(tmp_path: Path) -> None:
    config, input_root, output = _write_resource_refusal_fixture(tmp_path)
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))

    assert parsed["resources"] == {
        "max_aggregate_input_pixels": 50_000_000,
        "max_aggregate_output_pixels": 60_000_000,
        "max_modeled_peak_live_bytes": 1_000_000_000,
    }
    assert parsed["frames"] == {"names": ["ceiling"]}
    assert (input_root / "frames" / "ceiling.png").stat().st_size < 64
    assert not output.exists()


def test_repository_source_resolution_fails_installed_location_proof(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    interpreter = environment / "bin" / "python"
    console = environment / "bin" / "pixipix"
    repository_module = PROJECT_ROOT / "src" / "pixipix" / "__init__.py"

    with pytest.raises(SmokeFailure, match="resolved inside the repository checkout"):
        _validate_installed_location(
            interpreter=interpreter,
            console=console,
            module_path=repository_module,
            environment=environment,
            repository=PROJECT_ROOT,
        )


def test_final_validation_rejects_missing_frame(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    (tmp_path / FIXTURE_CONTRACT.frame_paths[0]).unlink()

    with pytest.raises(SmokeFailure, match="expected aligned frame is missing"):
        _validate_final_output(tmp_path)


def test_final_validation_rejects_undeclared_frame(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    Image.new("RGBA", (4, 4)).save(tmp_path / "frames" / "extra.png", format="PNG")

    with pytest.raises(SmokeFailure, match="missing or undeclared files"):
        _validate_final_output(tmp_path)


def test_final_validation_rejects_invalid_metadata(tmp_path: Path) -> None:
    _write_final_stage(tmp_path)
    (tmp_path / "stage.json").write_text("{", encoding="utf-8")

    with pytest.raises(SmokeFailure, match="aligned stage metadata is missing or invalid JSON"):
        _validate_final_output(tmp_path)


def test_failed_command_reports_exact_stage_and_process_output(tmp_path: Path) -> None:
    with pytest.raises(SmokeFailure, match="distribution smoke failed during pixelize") as captured:
        _run_stage(
            "pixelize",
            [sys.executable, "-c", "import sys; print('stage output'); sys.exit(7)"],
            cwd=tmp_path,
        )

    assert "stage output" in str(captured.value)
    assert "(7)" in str(captured.value)


@pytest.mark.parametrize("artifact_name", ["direct_wheel", "rebuilt_wheel"])
def test_corrupted_installed_artifact_fails_at_align(
    tmp_path: Path,
    built_artifacts: BuiltArtifacts,
    artifact_name: str,
) -> None:
    wheel = getattr(built_artifacts, artifact_name)
    assert isinstance(wheel, Path)
    corrupted = tmp_path / wheel.name
    _corrupt_align_implementation(wheel, corrupted)

    result = _run_corrupted_installed_pipeline(corrupted, tmp_path / "installed")

    assert result.returncode == 4
    assert result.stdout == ""
    assert "PX_INTERNAL_001" in result.stderr
    assert "Traceback" not in result.stderr
