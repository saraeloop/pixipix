from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pixipix.pipeline.artifacts import (
    _is_untrusted_path_component,
)

REAL_IS_SYMLINK = Path.is_symlink
REAL_LSTAT = Path.lstat
REAL_STAT = Path.stat
REAL_READLINK = os.readlink


@dataclass(frozen=True, slots=True)
class SyntheticAliasFacts:
    alias: Path
    raw_target: str
    absolute_target: Path
    link_is_symlink: bool = True
    link_uid: int = 0
    namespace_uid: int = 0
    namespace_mode: int = stat.S_IFDIR | 0o755
    target_is_symlink: bool = False
    target_uid: int = 0
    target_mode: int = stat.S_IFDIR | 0o755


VALID_VAR_FACTS = SyntheticAliasFacts(Path("/var"), "private/var", Path("/private/var"))
VALID_TMP_FACTS = SyntheticAliasFacts(
    Path("/tmp"),
    "private/tmp",
    Path("/private/tmp"),
    target_mode=stat.S_IFDIR | 0o1777,
)


@dataclass(frozen=True, slots=True)
class AliasPredicateCase:
    predicate: str
    facts: SyntheticAliasFacts
    trusted: bool


EXPECTED_ALIAS_PREDICATES = frozenset(
    {
        "valid-var",
        "valid-tmp-sticky",
        "var-absolute-target",
        "tmp-absolute-target",
        "exact-name",
        "link-uid",
        "raw-target",
        "namespace-uid",
        "namespace-directory",
        "namespace-group-writability",
        "namespace-other-writability",
        "target-symlink",
        "target-uid",
        "target-directory",
        "var-target-group-writability",
        "var-target-other-writability",
        "var-sticky-refusal",
        "tmp-sticky-requirement",
    }
)
ALIAS_PREDICATE_CASES = (
    AliasPredicateCase("valid-var", VALID_VAR_FACTS, True),
    AliasPredicateCase("valid-tmp-sticky", VALID_TMP_FACTS, True),
    AliasPredicateCase(
        "var-absolute-target", replace(VALID_VAR_FACTS, raw_target="/private/var"), True
    ),
    AliasPredicateCase(
        "tmp-absolute-target", replace(VALID_TMP_FACTS, raw_target="/private/tmp"), True
    ),
    AliasPredicateCase("exact-name", replace(VALID_VAR_FACTS, alias=Path("/variant")), False),
    AliasPredicateCase("link-uid", replace(VALID_VAR_FACTS, link_uid=501), False),
    AliasPredicateCase("raw-target", replace(VALID_VAR_FACTS, raw_target="private/tmp"), False),
    AliasPredicateCase("namespace-uid", replace(VALID_VAR_FACTS, namespace_uid=501), False),
    AliasPredicateCase(
        "namespace-directory",
        replace(VALID_VAR_FACTS, namespace_mode=stat.S_IFREG | 0o755),
        False,
    ),
    AliasPredicateCase(
        "namespace-group-writability",
        replace(VALID_VAR_FACTS, namespace_mode=stat.S_IFDIR | 0o775),
        False,
    ),
    AliasPredicateCase(
        "namespace-other-writability",
        replace(VALID_VAR_FACTS, namespace_mode=stat.S_IFDIR | 0o757),
        False,
    ),
    AliasPredicateCase("target-symlink", replace(VALID_VAR_FACTS, target_is_symlink=True), False),
    AliasPredicateCase("target-uid", replace(VALID_VAR_FACTS, target_uid=501), False),
    AliasPredicateCase(
        "target-directory",
        replace(VALID_VAR_FACTS, target_mode=stat.S_IFREG | 0o755),
        False,
    ),
    AliasPredicateCase(
        "var-target-group-writability",
        replace(VALID_VAR_FACTS, target_mode=stat.S_IFDIR | 0o775),
        False,
    ),
    AliasPredicateCase(
        "var-target-other-writability",
        replace(VALID_VAR_FACTS, target_mode=stat.S_IFDIR | 0o757),
        False,
    ),
    AliasPredicateCase(
        "var-sticky-refusal",
        replace(VALID_VAR_FACTS, target_mode=stat.S_IFDIR | 0o1777),
        False,
    ),
    AliasPredicateCase(
        "tmp-sticky-requirement",
        replace(VALID_TMP_FACTS, target_mode=stat.S_IFDIR | 0o0777),
        False,
    ),
)


def _install_synthetic_alias_facts(
    monkeypatch: pytest.MonkeyPatch, facts: SyntheticAliasFacts
) -> None:
    monkeypatch.setattr("pixipix.pipeline.artifacts.sys.platform", "darwin")
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: (
            facts.link_is_symlink
            if path == facts.alias
            else facts.target_is_symlink
            if path == facts.absolute_target
            else REAL_IS_SYMLINK(path)
        ),
    )
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: (
            SimpleNamespace(st_uid=facts.link_uid) if path == facts.alias else REAL_LSTAT(path)
        ),
    )

    def stat_facts(path: Path, *, follow_symlinks: bool = True) -> os.stat_result | SimpleNamespace:
        if path == facts.alias.parent:
            return SimpleNamespace(st_uid=facts.namespace_uid, st_mode=facts.namespace_mode)
        if path == facts.absolute_target:
            return SimpleNamespace(st_uid=facts.target_uid, st_mode=facts.target_mode)
        return REAL_STAT(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat_facts)
    monkeypatch.setattr(
        "pixipix.pipeline.artifacts.os.readlink",
        lambda path: facts.raw_target if path == facts.alias else REAL_READLINK(path),
    )


def test_alias_predicate_matrix_is_exact() -> None:
    assert {case.predicate for case in ALIAS_PREDICATE_CASES} == EXPECTED_ALIAS_PREDICATES
    assert len(ALIAS_PREDICATE_CASES) == len(EXPECTED_ALIAS_PREDICATES)


@pytest.mark.parametrize("case", ALIAS_PREDICATE_CASES, ids=lambda case: case.predicate)
def test_darwin_alias_trust_predicates_are_independently_enforced(
    monkeypatch: pytest.MonkeyPatch,
    case: AliasPredicateCase,
) -> None:
    _install_synthetic_alias_facts(monkeypatch, case.facts)

    assert _is_untrusted_path_component(case.facts.alias) is not case.trusted


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real Darwin system aliases")
@pytest.mark.parametrize(
    ("alias", "raw_target", "canonical"),
    [
        (Path("/var"), Path("private/var"), Path("/private/var")),
        (Path("/tmp"), Path("private/tmp"), Path("/private/tmp")),
    ],
)
def test_real_darwin_aliases_are_verified_without_resolving_descendants(
    alias: Path, raw_target: Path, canonical: Path
) -> None:
    assert alias.is_symlink()
    assert alias.readlink() == raw_target
    assert canonical.is_dir() and not canonical.is_symlink()
    assert os.path.samefile(alias, canonical)
    assert not _is_untrusted_path_component(alias)


def test_user_controlled_redirect_and_lexical_escape_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    assert _is_untrusted_path_component(linked)
    assert _is_untrusted_path_component(tmp_path / "safe" / ".." / "escape")
    assert not _is_untrusted_path_component(tmp_path / "ordinary")


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real Darwin system aliases")
def test_darwin_alias_trust_is_platform_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pixipix.pipeline.artifacts.sys.platform", "linux")

    assert _is_untrusted_path_component(Path("/var"))
    assert not _is_untrusted_path_component(Path("/tmp"))


def test_synthetic_windows_reparse_and_nonexistent_leaf_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("C:/candidate")
    monkeypatch.setattr("pixipix.pipeline.artifacts._runtime_os_name", lambda: "nt")
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(Path, "is_junction", lambda _path: False)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_file_attributes=0x400),
    )
    monkeypatch.setattr("pixipix.pipeline.artifacts.os.path.lexists", lambda _path: True)
    assert _is_untrusted_path_component(path)

    monkeypatch.setattr("pixipix.pipeline.artifacts.os.path.lexists", lambda _path: False)
    assert not _is_untrusted_path_component(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reparse behavior unexecuted")
def test_windows_redirects_fail_closed_when_detectable(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(ordinary, target_is_directory=True)

    assert not _is_untrusted_path_component(ordinary)
    assert _is_untrusted_path_component(linked)
