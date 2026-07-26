from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from tests.parity import generate_baseline as generator
from tests.parity import support
from tests.parity.support import BASELINE_COMMIT, ParityError, load_baseline


def test_generation_rejects_wrong_commit_and_dirty_checkout() -> None:
    with pytest.raises(ParityError, match="exact commit"):
        generator.validate_baseline_state("0" * 40, "")
    with pytest.raises(ParityError, match="clean"):
        generator.validate_baseline_state(BASELINE_COMMIT, "?? unexpected")


def test_generation_has_no_update_or_bless_mode() -> None:
    options = {
        option for action in generator.build_parser()._actions for option in action.option_strings
    }
    assert not options.intersection({"--approve", "--bless", "--update", "--accept"})


def test_generation_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "baseline.json"
    destination.write_text("preserve", encoding="utf-8")
    with pytest.raises(ParityError, match="overwrite"):
        generator.generate_baseline(destination)
    assert destination.read_text(encoding="utf-8") == "preserve"


def test_generation_never_removes_a_preexisting_worktree_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "existing"
    checkout.mkdir()
    calls: list[tuple[str, ...]] = []

    def fake_git(_repository: Path, *arguments: str) -> CompletedProcess[bytes]:
        calls.append(arguments)
        return CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(generator, "_git", fake_git)
    with pytest.raises(ParityError, match="pre-existing"):
        generator._add_baseline_worktree(tmp_path, checkout)
    assert calls == []


def test_partial_worktree_creation_failure_attempts_exact_checkout_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "partial"
    calls: list[tuple[str, ...]] = []

    def fake_git(_repository: Path, *arguments: str) -> CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[:3] == ("worktree", "add", "--detach"):
            checkout.mkdir()
            return CompletedProcess(arguments, 1, b"", b"partial failure")
        return CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(generator, "_git", fake_git)
    with pytest.raises(ParityError, match="unable to create"):
        generator._add_baseline_worktree(tmp_path, checkout)
    assert calls[-1] == ("worktree", "remove", "--force", str(checkout))


def test_isolated_environment_is_locked_noneditable_offline_and_external(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    environment_root = tmp_path / "environment"
    interpreter = environment_root / "bin" / "python"
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def fake_run(
        arguments: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        check: bool,
    ) -> CompletedProcess[bytes]:
        assert capture_output is True
        assert check is False
        calls.append((arguments, cwd, env))
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
        return CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setenv("PYTHONHOME", "/tmp/conflicting-python-home")
    monkeypatch.setenv("PYTHONPATH", "/tmp/conflicting-python-path")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert generator.prepare_isolated_environment(checkout, environment_root) == interpreter
    arguments, cwd, environment = calls[0]
    assert cwd == checkout
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(environment_root)
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert arguments[:4] == ["uv", "sync", "--project", str(checkout)]
    assert "--locked" in arguments
    assert "--all-groups" in arguments
    assert "--no-editable" in arguments
    assert "--offline" in arguments


def test_installed_capture_removes_caller_python_import_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/conflicting-caller-import")
    environment = support._runtime_environment(None)
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize("fail_inside", [False, True])
def test_isolated_worktree_cleanup_runs_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_inside: bool,
) -> None:
    calls: list[tuple[str, ...]] = []
    interpreter = tmp_path / "environment" / "bin" / "python"

    def fake_git(_repository: Path, *arguments: str) -> CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[:3] == ("worktree", "add", "--detach"):
            Path(arguments[3]).mkdir()
        if arguments[:2] == ("rev-parse", "HEAD"):
            return CompletedProcess(arguments, 0, (BASELINE_COMMIT + "\n").encode("ascii"), b"")
        return CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(generator, "_git", fake_git)
    monkeypatch.setattr(
        generator,
        "prepare_isolated_environment",
        lambda _checkout, _environment: interpreter,
    )
    if fail_inside:
        with (
            pytest.raises(RuntimeError, match="probe failure"),
            generator.isolated_baseline_worktree(tmp_path),
        ):
            raise RuntimeError("probe failure")
    else:
        with generator.isolated_baseline_worktree(tmp_path):
            pass
    removals = [call for call in calls if call[:3] == ("worktree", "remove", "--force")]
    assert len(removals) == 1
    status_checks = [call for call in calls if call[:2] == ("status", "--porcelain")]
    assert len(status_checks) == (2 if fail_inside else 3)


def test_tracked_baseline_records_exact_clean_authority() -> None:
    baseline = load_baseline()
    assert baseline["baselineProductionCommit"] == BASELINE_COMMIT
    assert baseline["baselineCleanWorktreeAttestation"] is True
    environment = baseline["environment"]
    assert isinstance(environment, dict)
    assert environment["sourceImportIsolation"] is True
