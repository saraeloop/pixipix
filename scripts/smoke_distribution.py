"""Smoke-test an installed PixiPix distribution outside the working project."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _run(command: Sequence[str | Path]) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    result = subprocess.run(rendered, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(rendered)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    """Exercise entry points plus inspect/extract against a committed fixture."""

    repository = Path(__file__).resolve().parents[1]
    image = repository / "tests" / "fixtures" / "robot-geometric.png"
    config = repository / "tests" / "fixtures" / "robot-geometric.toml"
    if not image.is_file() or not config.is_file():
        raise RuntimeError("committed robot smoke-test fixture is incomplete")

    console = shutil.which("pixipix")
    if console is None:
        raise RuntimeError("installed distribution did not provide the pixipix console entry point")
    expected_version = importlib.metadata.version("pixipix")

    help_result = _run([console, "--help"])
    if "Tiny poses in. Tidy pixels out." not in help_result.stdout:
        raise RuntimeError("console help output is missing the PixiPix product statement")
    version_result = _run([console, "version"])
    if version_result.stdout.strip() != f"PixiPix {expected_version}":
        raise RuntimeError("console version does not match installed distribution metadata")
    module_result = _run([sys.executable, "-m", "pixipix"])
    if "Tiny poses in. Tidy pixels out." not in module_result.stdout:
        raise RuntimeError("python -m pixipix is not equivalent to the console entry point")

    inspect_result = _run([console, "inspect", image, "--config", config])
    if "candidate components: 2" not in inspect_result.stdout:
        raise RuntimeError("fixture inspection did not find the expected two components")

    with tempfile.TemporaryDirectory(prefix=".pixipix-release-smoke-", dir=repository) as temporary:
        output = Path(temporary) / "extracted"
        _run([console, "extract", image, "--config", config, "--output", output])
        expected_outputs = {
            output / ".pixipix-output",
            output / "frames" / "idle.png",
            output / "frames" / "signal.png",
            output / "stage.json",
        }
        missing = sorted(str(path) for path in expected_outputs if not path.is_file())
        if missing:
            raise RuntimeError(f"installed distribution extraction is missing {missing[0]}")

    print(f"distribution smoke test passed for pixipix {expected_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
