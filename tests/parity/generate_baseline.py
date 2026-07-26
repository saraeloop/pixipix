"""Explicit isolated generator for the immutable post-M3 parity baseline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tests.parity.support import (
    API_PROBE_PATH,
    BASELINE_COMMIT,
    PROJECT_ROOT,
    ParityError,
    baseline_manifest,
    canonical_json_bytes,
    capture_behavior,
)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", repository, *arguments],
        capture_output=True,
        check=False,
    )


def validate_baseline_state(head: str, status: str) -> None:
    if head != BASELINE_COMMIT:
        raise ParityError(f"baseline checkout must be exact commit {BASELINE_COMMIT}")
    if status:
        raise ParityError("baseline checkout must be clean")


def verify_checkout(checkout: Path) -> None:
    head = _git(checkout, "rev-parse", "HEAD")
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if head.returncode != 0 or status.returncode != 0:
        raise ParityError("unable to inspect isolated baseline checkout")
    validate_baseline_state(
        head.stdout.decode("ascii").strip(),
        status.stdout.decode("utf-8").strip(),
    )


def _add_baseline_worktree(repository: Path, checkout: Path) -> None:
    if checkout.exists():
        raise ParityError("refusing to replace a pre-existing worktree path")
    added = _git(repository, "worktree", "add", "--detach", str(checkout), BASELINE_COMMIT)
    if added.returncode != 0:
        _git(repository, "worktree", "remove", "--force", str(checkout))
        raise ParityError("unable to create isolated M3 baseline worktree")


def prepare_isolated_environment(checkout: Path, environment_root: Path) -> Path:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["UV_PROJECT_ENVIRONMENT"] = str(environment_root)
    synchronized = subprocess.run(
        [
            "uv",
            "sync",
            "--project",
            str(checkout),
            "--locked",
            "--all-groups",
            "--no-editable",
            "--offline",
            "--python",
            sys.executable,
        ],
        cwd=checkout,
        env=environment,
        capture_output=True,
        check=False,
    )
    if synchronized.returncode != 0:
        raise ParityError("unable to create isolated locked baseline environment")
    interpreter = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        raise ParityError("isolated baseline interpreter is missing")
    return interpreter


@contextmanager
def isolated_baseline_worktree(
    repository: Path,
) -> Iterator[tuple[Path, Path, Path, Path]]:
    with tempfile.TemporaryDirectory(prefix="pixipix-m3-baseline-") as temporary:
        temporary_root = Path(temporary)
        checkout = temporary_root / "source"
        execution = temporary_root / "execution"
        environment = temporary_root / "environment"
        _add_baseline_worktree(repository, checkout)
        try:
            verify_checkout(checkout)
            interpreter = prepare_isolated_environment(checkout, environment)
            verify_checkout(checkout)
            yield checkout, execution, interpreter, environment
            verify_checkout(checkout)
        finally:
            removed = _git(repository, "worktree", "remove", "--force", str(checkout))
            if removed.returncode != 0:
                raise ParityError("unable to remove isolated M3 baseline worktree")


def generate_baseline(destination: Path, repository: Path = PROJECT_ROOT) -> None:
    if destination.exists():
        raise ParityError("refusing to overwrite an existing parity manifest")
    with isolated_baseline_worktree(repository) as (
        checkout,
        execution,
        interpreter,
        environment,
    ):
        behavior = capture_behavior(
            checkout,
            execution,
            python=interpreter,
            import_root=environment,
            api_probe_path=API_PROBE_PATH,
        )
        manifest = baseline_manifest(behavior)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_json_bytes(manifest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a new comparison file from the exact clean M3 authority commit."
    )
    parser.add_argument("destination", type=Path)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    generate_baseline(arguments.destination)


if __name__ == "__main__":
    main()
