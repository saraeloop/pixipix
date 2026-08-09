"""Generate the bounded v0.2.0 stable-release parity authority."""

from __future__ import annotations

import subprocess
import tempfile
import tomllib
from importlib.metadata import version
from pathlib import Path

from tests.parity.support import (
    PROJECT_ROOT,
    V0_1_1_AUTHORITY,
    V0_1_1_AUTHORITY_SHA256,
    V0_2_0_AUTHORITY,
    canonical_json_bytes,
    capture_behavior,
    capture_environment,
    load_release_baseline,
    release_authority_manifest,
    require_canonical_runtime,
    sha256_bytes,
)


def main() -> None:
    if V0_2_0_AUTHORITY.path.exists():
        raise RuntimeError("v0.2.0 parity authority already exists")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if head != V0_2_0_AUTHORITY.preparation_commit:
        raise RuntimeError("v0.2.0 authority must be captured from its preparation commit")
    runtime_diff = subprocess.run(
        ["git", "diff", "--exit-code", V0_2_0_AUTHORITY.preparation_commit, "--", "src/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if runtime_diff.returncode != 0:
        raise RuntimeError("v0.2.0 authority capture must not include runtime source changes")
    if sha256_bytes(V0_1_1_AUTHORITY.path.read_bytes()) != V0_1_1_AUTHORITY_SHA256:
        raise RuntimeError("v0.1.1 parity authority differs from its immutable hash")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"]["version"] != V0_2_0_AUTHORITY.version:
        raise RuntimeError("project metadata does not identify the v0.2.0 release candidate")
    if version("pixipix") != V0_2_0_AUTHORITY.version:
        raise RuntimeError("installed PixiPix does not identify the v0.2.0 release candidate")

    historical = load_release_baseline(authority=V0_1_1_AUTHORITY)
    require_canonical_runtime(
        historical["environment"],
        capture_environment(PROJECT_ROOT),
    )
    with tempfile.TemporaryDirectory(prefix="pixipix-v0.2.0-authority-") as temporary:
        behavior = capture_behavior(PROJECT_ROOT, Path(temporary) / "execution")
    manifest = release_authority_manifest(behavior, V0_2_0_AUTHORITY)
    V0_2_0_AUTHORITY.path.write_bytes(canonical_json_bytes(manifest))


if __name__ == "__main__":
    main()
