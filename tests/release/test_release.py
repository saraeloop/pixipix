from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release import (
    EXPECTED_SDIST_FILES,
    ReleaseValidationError,
    compare_wheels,
    inspect_distributions,
    validate_release_tag,
)


def _write_project(root: Path, version: str = "0.1.0") -> Path:
    license_text = "Apache License 2.0\n"
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "pixipix"',
                f'version = "{version}"',
                'requires-python = ">=3.12,<3.13"',
                'license = { file = "LICENSE" }',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    package = root / "src" / "pixipix"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""fixture package"""\n', encoding="utf-8")
    return pyproject


def _core_metadata(version: str, *, requires_python: str = "<3.13,>=3.12") -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: pixipix\n"
        f"Version: {version}\n"
        "License: Apache License 2.0\n"
        "License-File: LICENSE\n"
        f"Requires-Python: {requires_python}\n"
        "\n"
    ).encode()


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
