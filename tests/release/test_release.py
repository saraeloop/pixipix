from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.release import (
    EXPECTED_SDIST_FILES,
    ReleaseValidationError,
    compare_wheels,
    inspect_distributions,
    load_project_metadata,
    load_restricted_distribution_prefixes,
    validate_release_tag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DESCRIPTION = "Fixture deterministic sprite pipeline."
FIXTURE_AUTHOR = "Fixture Author"
FIXTURE_REPOSITORY_URL = "https://example.invalid/pixipix-fixture"


def _write_project(root: Path, version: str = "0.1.0") -> Path:
    license_text = "Apache License 2.0\n"
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "pixipix"',
                f'version = "{version}"',
                f'description = "{FIXTURE_DESCRIPTION}"',
                'requires-python = ">=3.12,<3.13"',
                'license = { file = "LICENSE" }',
                f'authors = [{{ name = "{FIXTURE_AUTHOR}" }}]',
                "",
                "[project.urls]",
                f'Repository = "{FIXTURE_REPOSITORY_URL}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "ASSET-LICENSES.md", root / "ASSET-LICENSES.md")
    package = root / "src" / "pixipix"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""fixture package"""\n', encoding="utf-8")
    return pyproject


def _core_metadata(
    version: str,
    *,
    name: str = "pixipix",
    summary: str | None = FIXTURE_DESCRIPTION,
    author: str | None = FIXTURE_AUTHOR,
    project_urls: tuple[str, ...] = (f"Repository, {FIXTURE_REPOSITORY_URL}",),
    requires_python: str = "<3.13,>=3.12",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> bytes:
    headers = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        "License: Apache License 2.0",
        "License-File: LICENSE",
        f"Requires-Python: {requires_python}",
    ]
    if summary is not None:
        headers.append(f"Summary: {summary}")
    if author is not None:
        headers.append(f"Author: {author}")
    headers.extend(f"Project-URL: {value}" for value in project_urls)
    headers.extend(f"{key}: {value}" for key, value in extra_headers)
    return ("\n".join(headers) + "\n\n").encode()


def _write_wheel(
    path: Path,
    version: str,
    *,
    metadata_version: str | None = None,
    requires_python: str = "<3.13,>=3.12",
    extra: dict[str, bytes] | None = None,
) -> None:
    dist_info = f"pixipix-{version}.dist-info"
    files = {
        "pixipix/__init__.py": b'"""fixture package"""\n',
        f"{dist_info}/METADATA": _core_metadata(
            metadata_version or version, requires_python=requires_python
        ),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\npixipix = pixipix.cli:main\n",
        f"{dist_info}/licenses/LICENSE": b"Apache License 2.0\n",
        f"{dist_info}/RECORD": b"",
    }
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_sdist(
    path: Path,
    project: Path,
    version: str,
    *,
    drop: set[str] | None = None,
    extra: dict[str, bytes] | None = None,
) -> None:
    root = f"pixipix-{version}"
    members = {name: f"fixture {name}\n".encode() for name in EXPECTED_SDIST_FILES}
    members.update(
        {
            "ASSET-LICENSES.md": (project / "ASSET-LICENSES.md").read_bytes(),
            "LICENSE": (project / "LICENSE").read_bytes(),
            "pyproject.toml": (project / "pyproject.toml").read_bytes(),
            "src/pixipix/__init__.py": b'"""fixture package"""\n',
            "examples/robot/README.md": b"# Robot\n",
            "PKG-INFO": _core_metadata(version),
        }
    )
    for name in drop or set():
        members.pop(name, None)
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))


def _distributions(
    tmp_path: Path,
    *,
    version: str = "0.1.0",
    metadata_version: str | None = None,
    requires_python: str = "<3.13,>=3.12",
    wheel_extra: dict[str, bytes] | None = None,
    sdist_drop: set[str] | None = None,
    sdist_extra: dict[str, bytes] | None = None,
) -> tuple[Path, Path]:
    project = tmp_path / "project"
    dist = tmp_path / "dist"
    project.mkdir()
    dist.mkdir()
    _write_project(project, version)
    _write_wheel(
        dist / f"pixipix-{version}-py3-none-any.whl",
        version,
        metadata_version=metadata_version,
        requires_python=requires_python,
        extra=wheel_extra,
    )
    _write_sdist(
        dist / f"pixipix-{version}.tar.gz",
        project,
        version,
        drop=sdist_drop,
        extra=sdist_extra,
    )
    return project, dist


@pytest.mark.parametrize(
    ("tag", "version"),
    [("v0.1.0a1", "0.1.0a1"), ("v0.1.0", "0.1.0")],
)
def test_release_tag_matches_authoritative_version(tmp_path: Path, tag: str, version: str) -> None:
    project_file = _write_project(tmp_path, version)

    assert validate_release_tag(tag, project_file) == version


def test_release_tag_requires_exactly_one_leading_v(tmp_path: Path) -> None:
    project_file = _write_project(tmp_path)

    with pytest.raises(ReleaseValidationError, match="exactly one leading 'v'"):
        validate_release_tag("0.1.0", project_file)
    with pytest.raises(ReleaseValidationError, match="malformed"):
        validate_release_tag("vv0.1.0", project_file)


@pytest.mark.parametrize("tag", ["v1", "v1.2", "v01.2.3", "v1.2.3-alpha.1", "v1.2.3.4"])
def test_malformed_release_tags_fail(tmp_path: Path, tag: str) -> None:
    project_file = _write_project(tmp_path)

    with pytest.raises(ReleaseValidationError, match="malformed"):
        validate_release_tag(tag, project_file)


def test_mismatched_release_tag_fails_clearly(tmp_path: Path) -> None:
    project_file = _write_project(tmp_path, "0.1.0")

    with pytest.raises(ReleaseValidationError, match="tag/package version mismatch"):
        validate_release_tag("v0.1.0a1", project_file)


def test_package_version_is_read_from_selected_authoritative_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_project = _write_project(first, "1.2.3")
    second_project = _write_project(second, "9.8.7")

    assert validate_release_tag("v1.2.3", first_project) == "1.2.3"
    assert validate_release_tag("v9.8.7", second_project) == "9.8.7"


def test_tag_validator_cli_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    project_file = _write_project(tmp_path, "0.1.0")
    script = Path(__file__).resolve().parents[2] / "scripts" / "release.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            "validate-tag",
            "--tag",
            "v0.1.0a1",
            "--project-file",
            project_file,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "tag/package version mismatch" in result.stderr


def test_distribution_inspection_accepts_expected_metadata_and_contents(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path)

    wheel, sdist = inspect_distributions(dist, project)

    assert wheel.suffix == ".whl"
    assert sdist.name.endswith(".tar.gz")


def _metadata_override(
    artifact: str, content: bytes
) -> tuple[dict[str, bytes] | None, dict[str, bytes] | None]:
    if artifact == "wheel":
        return {"pixipix-0.1.0.dist-info/METADATA": content}, None
    return None, {"PKG-INFO": content}


def test_current_project_metadata_authority_is_exact() -> None:
    metadata = load_project_metadata(PROJECT_ROOT / "pyproject.toml")

    assert metadata.name == "pixipix"
    assert metadata.version == "0.1.1"
    assert metadata.description == (
        "Deterministic sprite sheet pipeline: extract, scale, pixelize, and align "
        "raster art into production-ready RGBA frames."
    )
    assert metadata.author == "Saraeloop"
    assert metadata.repository_url == "https://github.com/pixipixhq/pixipix"


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("field", "content", "error"),
    [
        (
            "Summary",
            _core_metadata("0.1.0", summary="Incorrect adversarial description"),
            "metadata Summary",
        ),
        (
            "Author",
            _core_metadata("0.1.0", author="Adversary"),
            "metadata Author",
        ),
        (
            "Repository",
            _core_metadata(
                "0.1.0",
                project_urls=("Repository, https://example.invalid/adversary",),
            ),
            "metadata Project-URL Repository",
        ),
    ],
)
def test_distribution_inspection_rejects_governed_metadata_divergence(
    tmp_path: Path, artifact: str, field: str, content: bytes, error: str
) -> None:
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match=error):
        inspect_distributions(dist, project)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("field", "content", "error"),
    [
        ("Summary", _core_metadata("0.1.0", summary=None), "metadata Summary is missing"),
        ("Author", _core_metadata("0.1.0", author=None), "metadata Author is missing"),
        (
            "Repository",
            _core_metadata("0.1.0", project_urls=()),
            "metadata Project-URL Repository is missing",
        ),
    ],
)
def test_distribution_inspection_rejects_missing_governed_metadata(
    tmp_path: Path, artifact: str, field: str, content: bytes, error: str
) -> None:
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match=error):
        inspect_distributions(dist, project)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("case", "content", "error"),
    [
        (
            "duplicate Summary",
            _core_metadata("0.1.0", extra_headers=(("Summary", FIXTURE_DESCRIPTION),)),
            "metadata Summary occurs 2 times",
        ),
        (
            "duplicate Author",
            _core_metadata("0.1.0", extra_headers=(("Author", FIXTURE_AUTHOR),)),
            "metadata Author occurs 2 times",
        ),
        (
            "duplicate Repository",
            _core_metadata(
                "0.1.0",
                project_urls=(
                    f"Repository, {FIXTURE_REPOSITORY_URL}",
                    f"Repository, {FIXTURE_REPOSITORY_URL}",
                ),
            ),
            "metadata Project-URL Repository occurs 2 times",
        ),
        (
            "conflicting Repository",
            _core_metadata(
                "0.1.0",
                project_urls=(
                    f"Repository, {FIXTURE_REPOSITORY_URL}",
                    "Repository, https://example.invalid/adversary",
                ),
            ),
            "metadata Project-URL Repository occurs 2 times",
        ),
        (
            "malformed Project-URL",
            _core_metadata("0.1.0", project_urls=("Repository without a URL",)),
            "metadata Project-URL is malformed",
        ),
        (
            "right URL under wrong label",
            _core_metadata("0.1.0", project_urls=(f"Source, {FIXTURE_REPOSITORY_URL}",)),
            "metadata Project-URL Repository is missing",
        ),
    ],
)
def test_distribution_inspection_rejects_ambiguous_governed_metadata(
    tmp_path: Path, artifact: str, case: str, content: bytes, error: str
) -> None:
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match=error):
        inspect_distributions(dist, project)


@pytest.mark.parametrize(
    "summary",
    [
        "Deterministic palette-lock sprite pipeline.",
        "Fixture deterministic sprite pipeline without align.",
        "Fixture deterministic sprite pipeline producing RGB frames.",
    ],
)
@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_distribution_inspection_rejects_inaccurate_summary_variants(
    tmp_path: Path, artifact: str, summary: str
) -> None:
    content = _core_metadata("0.1.0", summary=summary)
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match="metadata Summary"):
        inspect_distributions(dist, project)


@pytest.mark.parametrize(
    "content",
    [
        _core_metadata("0.1.0", summary="Shared wrong Summary"),
        _core_metadata("0.1.0", author="Shared wrong Author"),
        _core_metadata(
            "0.1.0",
            project_urls=("Repository, https://example.invalid/shared-wrong",),
        ),
    ],
)
def test_distribution_inspection_rejects_shared_wrong_metadata(
    tmp_path: Path, content: bytes
) -> None:
    wheel_extra, _ = _metadata_override("wheel", content)
    _, sdist_extra = _metadata_override("sdist", content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match="metadata"):
        inspect_distributions(dist, project)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_distribution_inspection_preserves_unrelated_project_urls(
    tmp_path: Path, artifact: str
) -> None:
    content = _core_metadata(
        "0.1.0",
        project_urls=(
            "Documentation, https://example.invalid/docs",
            f"Repository, {FIXTURE_REPOSITORY_URL}",
        ),
    )
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    inspect_distributions(dist, project)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("content", "error"),
    [
        (_core_metadata("0.1.0", name="wrong-name"), "metadata Name"),
        (_core_metadata("0.1.1"), "metadata Version"),
    ],
)
def test_distribution_inspection_preserves_name_and_version_governance(
    tmp_path: Path, artifact: str, content: bytes, error: str
) -> None:
    wheel_extra, sdist_extra = _metadata_override(artifact, content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra, sdist_extra=sdist_extra)

    with pytest.raises(ReleaseValidationError, match=error):
        inspect_distributions(dist, project)


def test_inspect_dist_cli_rejects_governed_metadata_without_traceback(tmp_path: Path) -> None:
    content = _core_metadata("0.1.0", summary="Incorrect adversarial description")
    wheel_extra, _ = _metadata_override("wheel", content)
    project, dist = _distributions(tmp_path, wheel_extra=wheel_extra)
    script = PROJECT_ROOT / "scripts" / "release.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            "inspect-dist",
            "--dist-dir",
            dist,
            "--project-root",
            project,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "metadata Summary" in result.stderr
    assert "Traceback" not in result.stderr


def test_project_metadata_rejects_ambiguous_authors(tmp_path: Path) -> None:
    project_file = _write_project(tmp_path)
    project_file.write_text(
        project_file.read_text(encoding="utf-8").replace(
            f'authors = [{{ name = "{FIXTURE_AUTHOR}" }}]',
            'authors = [{ name = "First" }, { name = "Second" }]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="exactly one author"):
        load_project_metadata(project_file)


@pytest.mark.parametrize(
    ("artifact", "path"),
    [
        ("wheel", "assets/brand/pixipix-logo.png"),
        ("wheel", "./assets/brand/pixipix-logo.png"),
        ("wheel", "pixipix-0.1.0a5/assets/brand/pixipix-logo.png"),
        ("wheel", "pixipix-0.1.0a5/./assets/brand/pixipix-logo.png"),
        ("wheel", "Assets/Brand/future-logo.WEBP"),
        ("wheel", "examples/pixi-demo/docs/future-preview.webp"),
        ("wheel", "./examples/pixi-demo/docs/future-preview.webp"),
        ("wheel", "pixipix-0.1.0a5/examples/pixi-demo/docs/future-preview.webp"),
        ("sdist", "assets/brand/future-brand-image.svg"),
        ("sdist", "./assets/brand/pixipix-logo.png"),
        ("sdist", "Assets/Brand/future-logo.WEBP"),
        ("sdist", "examples/pixi-demo/docs/future-preview.webp"),
        ("sdist", "./examples/pixi-demo/docs/future-preview.webp"),
    ],
)
def test_distribution_inspection_rejects_repository_only_assets(
    tmp_path: Path, artifact: str, path: str
) -> None:
    extra = {path: b"restricted\n"}
    project, dist = _distributions(
        tmp_path,
        version="0.1.0a5",
        wheel_extra=extra if artifact == "wheel" else None,
        sdist_extra=extra if artifact == "sdist" else None,
    )

    with pytest.raises(ReleaseValidationError, match="restricted repository-only asset path"):
        inspect_distributions(dist, project)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "path",
    [
        "assets/brand/../palettes/allowed.json",
        "assets\\brand\\pixipix-logo.png",
    ],
)
def test_distribution_inspection_rejects_unsafe_non_posix_asset_paths(
    tmp_path: Path, artifact: str, path: str
) -> None:
    extra = {path: b"unsafe\n"}
    project, dist = _distributions(
        tmp_path,
        wheel_extra=extra if artifact == "wheel" else None,
        sdist_extra=extra if artifact == "sdist" else None,
    )

    with pytest.raises(ReleaseValidationError, match="unsafe archive member path"):
        inspect_distributions(dist, project)


def test_asset_manifest_drives_build_exclusions_and_release_inspection(tmp_path: Path) -> None:
    prefixes = load_restricted_distribution_prefixes(PROJECT_ROOT / "ASSET-LICENSES.md")
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_excludes = set(pyproject["tool"]["hatch"]["build"]["exclude"])

    assert prefixes
    assert "/assets" not in build_excludes
    assert {f"/{prefix.removesuffix('/')}" for prefix in prefixes} <= build_excludes

    for prefix in prefixes:
        future_member = f"{prefix}nested/future-preview.webp"
        for artifact in ("wheel", "sdist"):
            extra = {future_member: b"restricted\n"}
            case_root = tmp_path / f"{artifact}-{prefix.replace('/', '-')}"
            case_root.mkdir()
            project, dist = _distributions(
                case_root,
                wheel_extra=extra if artifact == "wheel" else None,
                sdist_extra=extra if artifact == "sdist" else None,
            )
            with pytest.raises(
                ReleaseValidationError, match="restricted repository-only asset path"
            ):
                inspect_distributions(dist, project)


def test_distribution_inspection_requires_authoritative_asset_manifest(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path)
    sdist_path = dist / "pixipix-0.1.0.tar.gz"
    replacement = tmp_path / "replacement.tar.gz"

    with tarfile.open(sdist_path, "r:gz") as source, tarfile.open(replacement, "w:gz") as target:
        for info in source.getmembers():
            extracted = source.extractfile(info)
            content = extracted.read() if extracted is not None else b""
            if info.name.endswith("/ASSET-LICENSES.md"):
                content = b"stale manifest\n"
                info.size = len(content)
            target.addfile(info, io.BytesIO(content))
    replacement.replace(sdist_path)

    with pytest.raises(ReleaseValidationError, match="packaged asset manifest"):
        inspect_distributions(dist, project)


def test_distribution_contract_keeps_legal_files_and_neutral_fixture() -> None:
    assert {"ASSET-LICENSES.md", "LICENSE", "tests/fixtures/robot-geometric.png"} <= (
        EXPECTED_SDIST_FILES
    )


def test_repository_checkout_keeps_complete_pixi_example_and_brand_assets() -> None:
    expected = {
        "assets/brand/pixipix-logo.png",
        "assets/brand/pixipix-mascot-logo.png",
        "examples/pixi-demo/README.md",
        "examples/pixi-demo/pixi-demo-sheet.png",
        "examples/pixi-demo/pixipix.toml",
    }

    assert all((PROJECT_ROOT / relative).is_file() for relative in expected)


def test_distribution_inspection_requires_exact_counts(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path)
    shutil.copy2(
        dist / "pixipix-0.1.0-py3-none-any.whl",
        dist / "pixipix-0.1.0-1-py3-none-any.whl",
    )

    with pytest.raises(ReleaseValidationError, match="exactly one wheel"):
        inspect_distributions(dist, project)


def test_distribution_inspection_rejects_metadata_version_mismatch(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path, metadata_version="0.1.1")

    with pytest.raises(ReleaseValidationError, match="metadata Version"):
        inspect_distributions(dist, project)


def test_distribution_inspection_rejects_python_requirement_mismatch(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path, requires_python=">=3.11")

    with pytest.raises(ReleaseValidationError, match="Requires-Python"):
        inspect_distributions(dist, project)


def test_distribution_inspection_rejects_forbidden_members(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path, wheel_extra={"docs-internal/PLAN.md": b"private\n"})

    with pytest.raises(ReleaseValidationError, match="forbidden member component"):
        inspect_distributions(dist, project)


def test_distribution_inspection_rejects_missing_expected_members(tmp_path: Path) -> None:
    project, dist = _distributions(tmp_path, sdist_drop={"uv.lock"})

    with pytest.raises(ReleaseValidationError, match=r"missing expected member.*uv.lock"):
        inspect_distributions(dist, project)


def test_distribution_inspection_rejects_absolute_home_path_leakage(tmp_path: Path) -> None:
    leaked_path = b"built at /" + b"Users/alice/work/pixipix\n"
    project, dist = _distributions(tmp_path, wheel_extra={"pixipix/build_info.txt": leaked_path})

    with pytest.raises(ReleaseValidationError, match="absolute home path"):
        inspect_distributions(dist, project)


def test_compare_wheels_reports_identical_bytes(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    rebuilt = tmp_path / "rebuilt"
    direct.mkdir()
    rebuilt.mkdir()
    _write_wheel(direct / "pixipix-0.1.0-py3-none-any.whl", "0.1.0")
    shutil.copy2(
        direct / "pixipix-0.1.0-py3-none-any.whl",
        rebuilt / "pixipix-0.1.0-py3-none-any.whl",
    )

    direct_hash, rebuilt_hash = compare_wheels(direct, rebuilt)

    assert direct_hash == rebuilt_hash


def test_compare_wheels_rejects_different_bytes(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    rebuilt = tmp_path / "rebuilt"
    direct.mkdir()
    rebuilt.mkdir()
    _write_wheel(direct / "pixipix-0.1.0-py3-none-any.whl", "0.1.0")
    _write_wheel(
        rebuilt / "pixipix-0.1.0-py3-none-any.whl",
        "0.1.0",
        extra={"pixipix/extra.py": b"different\n"},
    )

    with pytest.raises(ReleaseValidationError, match="not byte-identical"):
        compare_wheels(direct, rebuilt)
