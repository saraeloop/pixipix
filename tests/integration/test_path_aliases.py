from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from typer.testing import CliRunner

from pixipix.cli import app
from pixipix.config import load_config
from pixipix.errors import ProcessingError, UnsupportedInputError
from pixipix.pipeline.input import validate_stage_input
from pixipix.pipeline.publication import validate_stage_output_target
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale
from tests.helpers import alignment_config, transparent_sheet, write_config, write_rgba

type AliasName = Literal["var", "tmp"]
type AliasBoundary = Literal["direct", "cli"]


@contextmanager
def _real_alias_root(alias_name: AliasName) -> Iterator[tuple[Path, Path, Path]]:
    if sys.platform != "darwin":
        pytest.skip("real macOS alias boundary requires Darwin")
    alias = Path(f"/{alias_name}")
    canonical_alias = Path(f"/private/{alias_name}")
    raw_target = Path(f"private/{alias_name}")
    if (
        not alias.is_symlink()
        or alias.readlink() != raw_target
        or not canonical_alias.is_dir()
        or not os.path.samefile(alias, canonical_alias)
    ):
        pytest.skip(f"host does not expose verified {alias} system alias")
    base = Path(tempfile.gettempdir()) if alias == Path("/var") else alias
    lexical = Path(tempfile.mkdtemp(prefix="pixipix-path-alias-", dir=base))
    if alias not in (lexical, *lexical.parents):
        shutil.rmtree(lexical)
        pytest.skip(f"writable temporary root is not lexically beneath {alias}")
    canonical = Path("/private").joinpath(*lexical.parts[1:])
    assert os.path.samefile(lexical, canonical)
    try:
        yield alias, lexical, canonical
    finally:
        shutil.rmtree(lexical, ignore_errors=True)


@pytest.fixture(params=("var", "tmp"))
def real_alias_root(request: pytest.FixtureRequest) -> Iterator[tuple[Path, Path, Path]]:
    with _real_alias_root(request.param) as root:
        yield root


def _artifact_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_direct_pipeline(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir()
    source = root / "source.png"
    config = root / "project.toml"
    write_rgba(source, transparent_sheet())
    write_config(config, alignment_config())
    loaded = load_config(config)
    extracted = root / "extracted"
    scaled = root / "scaled"
    pixelized = root / "pixelized"
    aligned = root / "aligned"
    publish_extraction(source, loaded, extracted)
    publish_scale(extracted, loaded, scaled)
    publish_pixelize(scaled, loaded, pixelized)
    publish_align(pixelized, loaded, aligned)
    return extracted, scaled, pixelized, aligned


def _exercise_direct_alias_boundary(lexical: Path, canonical: Path) -> None:
    lexical_outputs = _run_direct_pipeline(lexical / "lexical")
    canonical_outputs = _run_direct_pipeline(canonical / "canonical")

    assert [_artifact_digest(path) for path in lexical_outputs] == [
        _artifact_digest(path) for path in canonical_outputs
    ]
    lexical_source = lexical / "lexical" / "source.png"
    lexical_config = lexical / "lexical" / "project.toml"
    publish_extraction(
        lexical_source,
        load_config(lexical_config),
        lexical_outputs[0],
        force=True,
    )


def _exercise_cli_alias_boundary(lexical: Path, _canonical: Path) -> None:
    root = lexical / "cli"
    root.mkdir()
    source = root / "source.png"
    config = root / "project.toml"
    write_rgba(source, transparent_sheet())
    write_config(config, alignment_config())
    outputs = tuple(root / stage for stage in ("extract", "scale", "pixelize", "align"))
    commands = (
        ("extract", source, outputs[0]),
        ("scale", outputs[0], outputs[1]),
        ("pixelize", outputs[1], outputs[2]),
        ("align", outputs[2], outputs[3]),
    )
    runner = CliRunner()
    for command, source_path, output in commands:
        result = runner.invoke(
            app,
            [command, str(source_path), "--config", str(config), "--output", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.is_dir()


@dataclass(frozen=True, slots=True)
class AliasBoundaryCase:
    alias: AliasName
    boundary: AliasBoundary
    exercise: Callable[[Path, Path], None]


EXPECTED_REAL_ALIAS_NAMES = frozenset[AliasName]({"var", "tmp"})
EXPECTED_REAL_ALIAS_BOUNDARIES = frozenset[AliasBoundary]({"direct", "cli"})
EXPECTED_ALIAS_BOUNDARY_MATRIX = frozenset(
    {
        ("var", "direct"),
        ("var", "cli"),
        ("tmp", "direct"),
        ("tmp", "cli"),
    }
)
ALIAS_BOUNDARY_EXERCISES: dict[AliasBoundary, Callable[[Path, Path], None]] = {
    "direct": _exercise_direct_alias_boundary,
    "cli": _exercise_cli_alias_boundary,
}
REAL_ALIAS_BOUNDARY_CASES = (
    AliasBoundaryCase("var", "direct", _exercise_direct_alias_boundary),
    AliasBoundaryCase("var", "cli", _exercise_cli_alias_boundary),
    AliasBoundaryCase("tmp", "direct", _exercise_direct_alias_boundary),
    AliasBoundaryCase("tmp", "cli", _exercise_cli_alias_boundary),
)


def test_real_alias_boundary_matrix_is_exact_and_execution_connected() -> None:
    assert {case.alias for case in REAL_ALIAS_BOUNDARY_CASES} == EXPECTED_REAL_ALIAS_NAMES
    assert {case.boundary for case in REAL_ALIAS_BOUNDARY_CASES} == EXPECTED_REAL_ALIAS_BOUNDARIES
    assert {(case.alias, case.boundary) for case in REAL_ALIAS_BOUNDARY_CASES} == (
        EXPECTED_ALIAS_BOUNDARY_MATRIX
    )
    assert len(REAL_ALIAS_BOUNDARY_CASES) == len(EXPECTED_ALIAS_BOUNDARY_MATRIX)
    assert all(
        case.exercise is ALIAS_BOUNDARY_EXERCISES[case.boundary]
        for case in REAL_ALIAS_BOUNDARY_CASES
    )


@pytest.mark.parametrize(
    "case",
    REAL_ALIAS_BOUNDARY_CASES,
    ids=lambda case: f"{case.alias}-{case.boundary}",
)
def test_required_real_alias_boundaries(case: AliasBoundaryCase) -> None:
    with _real_alias_root(case.alias) as (_alias, lexical, canonical):
        case.exercise(lexical, canonical)


def test_initial_and_final_output_validation_use_the_same_alias_policy(
    real_alias_root: tuple[Path, Path, Path],
) -> None:
    _alias, lexical, _canonical = real_alias_root
    root = lexical / "revalidation"
    root.mkdir()
    source = root / "source.png"
    config = root / "project.toml"
    output = root / "output"
    write_rgba(source, transparent_sheet())
    write_config(config)
    validate_stage_output_target(output, "extract")
    publish_extraction(source, load_config(config), output)

    assert (output / "stage.json").is_file()
    validate_stage_output_target(output, "extract", force=True)


def test_user_redirects_beneath_verified_alias_remain_rejected(
    real_alias_root: tuple[Path, Path, Path],
) -> None:
    _alias, lexical, _canonical = real_alias_root
    real = lexical / "real"
    real.mkdir()
    source = real / "source.png"
    config = real / "project.toml"
    artifact = real / "artifact"
    write_rgba(source, transparent_sheet())
    write_config(config)
    loaded = load_config(config)
    publish_extraction(source, loaded, artifact)
    linked = lexical / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(UnsupportedInputError) as input_error:
        validate_stage_input(linked / artifact.name, "extract")
    assert (input_error.value.code, input_error.value.stage, input_error.value.exit_code) == (
        "PX_STAGE_001",
        "load",
        3,
    )
    assert "/private/" not in str(input_error.value)
    with pytest.raises(ProcessingError) as output_error:
        validate_stage_output_target(linked / "output", "extract")
    assert (output_error.value.code, output_error.value.stage, output_error.value.exit_code) == (
        "PX_OUTPUT_004",
        "publish",
        1,
    )
    assert "/private/" not in str(output_error.value)
    assert not (real / "output").exists()
    with pytest.raises(ProcessingError, match="PX_OUTPUT_004"):
        validate_stage_output_target(lexical / "safe" / ".." / "escape", "extract")


def test_frame_redirect_beneath_verified_alias_remains_rejected(
    real_alias_root: tuple[Path, Path, Path],
) -> None:
    _alias, lexical, _canonical = real_alias_root
    root = lexical / "frame-redirect"
    root.mkdir()
    source = root / "source.png"
    config = root / "project.toml"
    artifact = root / "artifact"
    write_rgba(source, transparent_sheet())
    write_config(config)
    publish_extraction(source, load_config(config), artifact)
    frame = next((artifact / "frames").iterdir())
    retained = root / frame.name
    frame.replace(retained)
    frame.symlink_to(retained)

    with pytest.raises(UnsupportedInputError) as error:
        validate_stage_input(artifact, "extract")
    assert error.value.code == "PX_STAGE_011"
