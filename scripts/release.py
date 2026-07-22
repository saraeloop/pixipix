"""Validate release tags and built Python distributions.

This module deliberately uses only the Python standard library so it can run in
the locked development environment and in a minimal CI verification context.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = re.compile(
    r"^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?$"
)
FORBIDDEN_COMPONENTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs-internal",
}
FORBIDDEN_SECRET_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
LOCAL_PATH_PATTERNS = (
    re.compile(rb"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n]+\\"),
)
EXPECTED_SDIST_FILES = {
    ".gitignore",
    ".python-version",
    "ASSET-LICENSES.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "scripts/release.py",
    "scripts/smoke_distribution.py",
    "tests/fixtures/robot-geometric.SCENARIO.md",
    "tests/fixtures/robot-geometric.png",
    "tests/fixtures/robot-geometric.toml",
    "uv.lock",
}
RESTRICTED_PATHS_START = "<!-- pixipix:restricted-distribution-paths:start -->"
RESTRICTED_PATHS_END = "<!-- pixipix:restricted-distribution-paths:end -->"


class ReleaseValidationError(ValueError):
    """Raised when a release input violates a repository contract."""


@dataclass(frozen=True)
class ProjectMetadata:
    """Authoritative project metadata loaded from ``pyproject.toml``."""

    name: str
    version: str
    requires_python: str
    license_file: str


@dataclass(frozen=True)
class Archive:
    """In-memory view of regular files in a distribution archive."""

    path: Path
    files: dict[str, bytes]


def _required_string(table: dict[str, object], key: str, source: Path) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{source}: project.{key} must be a non-empty string")
    return value


def load_project_metadata(project_file: Path) -> ProjectMetadata:
    """Read release metadata from the committed PEP 621 project table."""

    try:
        data = tomllib.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseValidationError(
            f"cannot read authoritative metadata {project_file}: {error}"
        ) from error

    project = data.get("project")
    if not isinstance(project, dict):
        raise ReleaseValidationError(f"{project_file}: missing [project] table")

    license_value = project.get("license")
    if not isinstance(license_value, dict):
        raise ReleaseValidationError(f"{project_file}: project.license must name a license file")
    license_file = license_value.get("file")
    if not isinstance(license_file, str) or not license_file:
        raise ReleaseValidationError(
            f"{project_file}: project.license.file must be a non-empty string"
        )

    return ProjectMetadata(
        name=_required_string(project, "name", project_file),
        version=_required_string(project, "version", project_file),
        requires_python=_required_string(project, "requires-python", project_file),
        license_file=license_file,
    )


def validate_release_tag(tag: str, project_file: Path) -> str:
    """Require ``v<version>`` to equal the committed project version exactly."""

    metadata = load_project_metadata(project_file)
    if not tag.startswith("v"):
        raise ReleaseValidationError(f"release tag {tag!r} must start with exactly one leading 'v'")
    if RELEASE_TAG.fullmatch(tag) is None:
        raise ReleaseValidationError(
            f"release tag {tag!r} is malformed; expected vMAJOR.MINOR.PATCH "
            "with optional aN, bN, or rcN"
        )

    tag_version = tag.removeprefix("v")
    if tag_version != metadata.version:
        raise ReleaseValidationError(
            f"release tag/package version mismatch: tag {tag!r} maps to {tag_version!r}, "
            f"but {project_file} declares {metadata.version!r}"
        )
    return tag_version


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _normalize_member_name(archive: Path, name: str) -> str:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name:
        raise ReleaseValidationError(f"{archive}: unsafe archive member path {name!r}")
    if not member.parts:
        raise ReleaseValidationError(f"{archive}: empty archive member path")

    lowered_parts = {part.lower() for part in member.parts}
    forbidden = sorted(lowered_parts & FORBIDDEN_COMPONENTS)
    if forbidden:
        raise ReleaseValidationError(
            f"{archive}: forbidden member component {forbidden[0]!r} in {name!r}"
        )

    filename = member.name.lower()
    if filename in FORBIDDEN_SECRET_NAMES or any(
        filename.endswith(suffix) for suffix in FORBIDDEN_SECRET_SUFFIXES
    ):
        raise ReleaseValidationError(
            f"{archive}: secret-like file {name!r} must not be distributed"
        )
    return member.as_posix()


def _validate_member_content(archive: Path, name: str, content: bytes, project_root: Path) -> None:
    local_root = str(project_root.resolve()).encode()
    if local_root and local_root in content:
        raise ReleaseValidationError(f"{archive}: member {name!r} leaks the local repository path")
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(content):
            raise ReleaseValidationError(
                f"{archive}: member {name!r} contains an absolute home path"
            )


def _read_wheel(path: Path, project_root: Path) -> Archive:
    files: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(path) as wheel:
            for info in wheel.infolist():
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ReleaseValidationError(
                        f"{path}: symbolic link member {info.filename!r} is forbidden"
                    )
                member_name = _normalize_member_name(path, info.filename)
                if member_name in files:
                    raise ReleaseValidationError(
                        f"{path}: duplicate normalized archive member {member_name!r}"
                    )
                content = wheel.read(info)
                _validate_member_content(path, member_name, content, project_root)
                files[member_name] = content
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseValidationError(f"cannot inspect wheel {path}: {error}") from error
    return Archive(path=path, files=files)


def _read_sdist(path: Path, project_root: Path) -> Archive:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(path, mode="r:gz") as sdist:
            for info in sdist.getmembers():
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ReleaseValidationError(
                        f"{path}: non-regular member {info.name!r} is forbidden"
                    )
                member_name = _normalize_member_name(path, info.name)
                if member_name in files:
                    raise ReleaseValidationError(
                        f"{path}: duplicate normalized archive member {member_name!r}"
                    )
                extracted = sdist.extractfile(info)
                if extracted is None:
                    raise ReleaseValidationError(f"{path}: cannot read member {info.name!r}")
                content = extracted.read()
                _validate_member_content(path, member_name, content, project_root)
                files[member_name] = content
    except (OSError, tarfile.TarError) as error:
        raise ReleaseValidationError(
            f"cannot inspect source distribution {path}: {error}"
        ) from error
    return Archive(path=path, files=files)


def _parse_metadata(content: bytes, source: str) -> Message:
    try:
        return BytesParser(policy=default).parsebytes(content)
    except Exception as error:
        raise ReleaseValidationError(
            f"cannot parse package metadata from {source}: {error}"
        ) from error


def _require_metadata(message: Message, key: str, expected: str, source: str) -> None:
    actual = message.get(key)
    if actual != expected:
        raise ReleaseValidationError(
            f"{source}: metadata {key} is {actual!r}, expected {expected!r}"
        )


def _specifier_parts(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _runtime_files(project_root: Path, package_name: str) -> set[str]:
    source_root = project_root / "src" / package_name
    if not source_root.is_dir():
        raise ReleaseValidationError(f"missing runtime package directory {source_root}")
    return {
        path.relative_to(project_root / "src").as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and not (set(path.parts) & FORBIDDEN_COMPONENTS)
    }


def _single_file(directory: Path, pattern: str, kind: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ReleaseValidationError(
            f"{directory}: expected exactly one {kind} matching {pattern!r}, found {len(matches)}"
        )
    return matches[0]


def load_restricted_distribution_prefixes(manifest_path: Path) -> tuple[str, ...]:
    """Load canonical repository-only directory prefixes from the asset manifest."""

    try:
        manifest = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseValidationError(
            f"cannot read asset manifest {manifest_path}: {error}"
        ) from error

    if manifest.count(RESTRICTED_PATHS_START) != 1 or manifest.count(RESTRICTED_PATHS_END) != 1:
        raise ReleaseValidationError(
            f"{manifest_path}: expected exactly one restricted distribution path block"
        )
    _, block_and_end = manifest.split(RESTRICTED_PATHS_START, maxsplit=1)
    block, _ = block_and_end.split(RESTRICTED_PATHS_END, maxsplit=1)

    prefixes: list[str] = []
    for line in block.splitlines():
        declared = line.strip()
        if not declared:
            continue
        member = PurePosixPath(declared)
        normalized = member.as_posix()
        canonical = f"{normalized.rstrip('/')}/"
        if (
            member.is_absolute()
            or ".." in member.parts
            or "\\" in declared
            or normalized in {"", "."}
            or declared != canonical
        ):
            raise ReleaseValidationError(
                f"{manifest_path}: restricted distribution path {declared!r} "
                "must be a canonical relative directory ending in '/'"
            )
        prefixes.append(canonical)

    if not prefixes:
        raise ReleaseValidationError(
            f"{manifest_path}: restricted distribution path block is empty"
        )
    if len(prefixes) != len(set(prefixes)):
        raise ReleaseValidationError(f"{manifest_path}: duplicate restricted distribution path")
    return tuple(prefixes)


def _restricted_distribution_members(
    names: set[str], prefixes: tuple[str, ...], *, archive_root: str | None = None
) -> list[str]:
    restricted: list[str] = []
    for name in names:
        member = PurePosixPath(name)
        if archive_root is not None:
            if not member.parts or member.parts[0] != archive_root:
                continue
            relative = PurePosixPath(*member.parts[1:]).as_posix()
        else:
            relative = member.as_posix()
        relative_parts = PurePosixPath(relative).parts
        candidates = {
            PurePosixPath(*relative_parts[index:]).as_posix().casefold()
            for index in range(len(relative_parts))
        }
        if any(
            candidate == prefix.removesuffix("/").casefold()
            or candidate.startswith(prefix.casefold())
            for candidate in candidates
            for prefix in prefixes
        ):
            restricted.append(name)
    return sorted(restricted)


def inspect_distributions(dist_dir: Path, project_root: Path) -> tuple[Path, Path]:
    """Validate the exact wheel and sdist intended for artifact upload."""

    metadata = load_project_metadata(project_root / "pyproject.toml")
    restricted_prefixes = load_restricted_distribution_prefixes(project_root / "ASSET-LICENSES.md")
    normalized_name = _normalized_distribution_name(metadata.name)
    wheel_path = _single_file(dist_dir, "*.whl", "wheel")
    sdist_path = _single_file(dist_dir, "*.tar.gz", "source distribution")
    expected_wheel_prefix = f"{normalized_name}-{metadata.version}-"
    expected_sdist_name = f"{normalized_name}-{metadata.version}.tar.gz"
    if not wheel_path.name.startswith(expected_wheel_prefix):
        raise ReleaseValidationError(
            f"wheel filename {wheel_path.name!r} does not contain expected project/version "
            f"{metadata.name!r} {metadata.version!r}"
        )
    if sdist_path.name != expected_sdist_name:
        raise ReleaseValidationError(
            f"sdist filename {sdist_path.name!r} is not expected {expected_sdist_name!r}"
        )

    wheel = _read_wheel(wheel_path, project_root)
    sdist = _read_sdist(sdist_path, project_root)
    restricted_wheel = _restricted_distribution_members(set(wheel.files), restricted_prefixes)
    if restricted_wheel:
        raise ReleaseValidationError(
            f"{wheel_path}: restricted repository-only asset path "
            f"{restricted_wheel[0]!r} must not be distributed"
        )
    dist_info = f"{normalized_name}-{metadata.version}.dist-info"
    metadata_member = f"{dist_info}/METADATA"
    if metadata_member not in wheel.files:
        raise ReleaseValidationError(f"{wheel_path}: missing {metadata_member}")
    wheel_metadata = _parse_metadata(wheel.files[metadata_member], metadata_member)
    _require_metadata(wheel_metadata, "Name", metadata.name, metadata_member)
    _require_metadata(wheel_metadata, "Version", metadata.version, metadata_member)
    requires_python = wheel_metadata.get("Requires-Python")
    if requires_python is None or _specifier_parts(requires_python) != _specifier_parts(
        metadata.requires_python
    ):
        raise ReleaseValidationError(
            f"{metadata_member}: Requires-Python is {requires_python!r}, "
            f"expected {metadata.requires_python!r}"
        )

    license_path = project_root / metadata.license_file
    try:
        expected_license = license_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ReleaseValidationError(
            f"cannot read declared license {license_path}: {error}"
        ) from error
    actual_license = wheel_metadata.get("License")
    if actual_license is None or actual_license.split() != expected_license.split():
        raise ReleaseValidationError(
            f"{metadata_member}: License does not match {metadata.license_file}"
        )
    if metadata.license_file not in wheel_metadata.get_all("License-File", []):
        raise ReleaseValidationError(
            f"{metadata_member}: License-File does not declare {metadata.license_file!r}"
        )
    license_member = f"{dist_info}/licenses/{metadata.license_file}"
    if (
        wheel.files.get(license_member, b"").decode("utf-8", errors="replace").strip()
        != expected_license
    ):
        raise ReleaseValidationError(
            f"{wheel_path}: packaged license {license_member!r} is missing or incorrect"
        )
    entry_points_member = f"{dist_info}/entry_points.txt"
    entry_points = wheel.files.get(entry_points_member, b"").decode("utf-8", errors="replace")
    if "pixipix = pixipix.cli:main" not in entry_points:
        raise ReleaseValidationError(
            f"{wheel_path}: console entry point is missing from {entry_points_member!r}"
        )

    expected_runtime = _runtime_files(project_root, metadata.name)
    missing_runtime = sorted(expected_runtime - set(wheel.files))
    if missing_runtime:
        raise ReleaseValidationError(f"{wheel_path}: missing runtime member {missing_runtime[0]!r}")
    unexpected_tests = sorted(name for name in wheel.files if "tests" in PurePosixPath(name).parts)
    if unexpected_tests:
        raise ReleaseValidationError(
            f"{wheel_path}: wheel unexpectedly contains tests: {unexpected_tests[0]!r}"
        )

    sdist_root = f"{normalized_name}-{metadata.version}"
    if not sdist.files or any(PurePosixPath(name).parts[0] != sdist_root for name in sdist.files):
        raise ReleaseValidationError(f"{sdist_path}: every member must be rooted at {sdist_root!r}")
    restricted_sdist = _restricted_distribution_members(
        set(sdist.files), restricted_prefixes, archive_root=sdist_root
    )
    if restricted_sdist:
        raise ReleaseValidationError(
            f"{sdist_path}: restricted repository-only asset path "
            f"{restricted_sdist[0]!r} must not be distributed"
        )
    expected_sdist = {f"{sdist_root}/{name}" for name in EXPECTED_SDIST_FILES}
    expected_sdist.update(f"{sdist_root}/src/{name}" for name in expected_runtime)
    missing_sdist = sorted(expected_sdist - set(sdist.files))
    if missing_sdist:
        raise ReleaseValidationError(f"{sdist_path}: missing expected member {missing_sdist[0]!r}")
    manifest_member = f"{sdist_root}/ASSET-LICENSES.md"
    expected_manifest = (project_root / "ASSET-LICENSES.md").read_bytes()
    if sdist.files.get(manifest_member) != expected_manifest:
        raise ReleaseValidationError(
            f"{sdist_path}: packaged asset manifest {manifest_member!r} is missing or incorrect"
        )
    if not any(name.startswith(f"{sdist_root}/tests/") for name in sdist.files):
        raise ReleaseValidationError(f"{sdist_path}: source distribution must contain tests")

    pkg_info_member = f"{sdist_root}/PKG-INFO"
    if pkg_info_member not in sdist.files:
        raise ReleaseValidationError(f"{sdist_path}: missing {pkg_info_member}")
    sdist_metadata = _parse_metadata(sdist.files[pkg_info_member], pkg_info_member)
    _require_metadata(sdist_metadata, "Name", metadata.name, pkg_info_member)
    _require_metadata(sdist_metadata, "Version", metadata.version, pkg_info_member)

    print(f"verified wheel: {wheel_path.name} ({len(wheel.files)} files)")
    print(f"verified sdist: {sdist_path.name} ({len(sdist.files)} files)")
    return wheel_path, sdist_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_wheels(direct_dir: Path, rebuilt_dir: Path) -> tuple[str, str]:
    """Require the direct and sdist-built wheel bytes to match exactly."""

    direct = _single_file(direct_dir, "*.whl", "direct wheel")
    rebuilt = _single_file(rebuilt_dir, "*.whl", "wheel rebuilt from sdist")
    direct_hash = _sha256(direct)
    rebuilt_hash = _sha256(rebuilt)
    print(f"direct wheel sha256:       {direct_hash}")
    print(f"sdist-built wheel sha256: {rebuilt_hash}")
    if direct.read_bytes() != rebuilt.read_bytes():
        raise ReleaseValidationError(
            "direct and sdist-built wheels are not byte-identical; release reproducibility failed"
        )
    return direct_hash, rebuilt_hash


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tag_parser = subparsers.add_parser(
        "validate-tag", help="compare a release tag to project metadata"
    )
    tag_parser.add_argument("--tag", required=True)
    tag_parser.add_argument("--project-file", type=Path, default=PROJECT_ROOT / "pyproject.toml")

    inspect_parser = subparsers.add_parser("inspect-dist", help="inspect one wheel and one sdist")
    inspect_parser.add_argument("--dist-dir", type=Path, default=PROJECT_ROOT / "dist")
    inspect_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    compare_parser = subparsers.add_parser(
        "compare-wheels", help="compare direct and sdist-built wheels"
    )
    compare_parser.add_argument("--direct-dir", type=Path, required=True)
    compare_parser.add_argument("--rebuilt-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a release verification command."""

    parser = _command_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-tag":
            version = validate_release_tag(args.tag, args.project_file)
            print(f"release tag {args.tag!r} matches authoritative package version {version!r}")
        elif args.command == "inspect-dist":
            inspect_distributions(args.dist_dir, args.project_root)
        elif args.command == "compare-wheels":
            compare_wheels(args.direct_dir, args.rebuilt_dir)
        else:  # pragma: no cover - argparse rejects unknown subcommands.
            parser.error(f"unknown command {args.command!r}")
    except ReleaseValidationError as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
