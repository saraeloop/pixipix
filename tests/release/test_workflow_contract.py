from __future__ import annotations

import re
from pathlib import Path

from scripts.smoke_distribution import SMOKE_STAGES

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "publish.yml"
RELEASE_ARTIFACT = "python-package-distributions"
ACTION_REFERENCE = re.compile(
    r"(?m)^\s*uses:\s+(?P<action>[^@\s]+)@(?P<sha>[0-9a-f]{40})\s+#\s+(?P<version>v\S+)\s*$"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(text: str, name: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [\w-]+:\n|\Z)", text)
    assert match is not None, f"missing workflow job {name!r}"
    return match.group("body")


def test_trusted_publisher_filename_and_environment_contract() -> None:
    text = _workflow_text()
    publish = _job(text, "publish")

    assert WORKFLOW.name == "publish.yml"
    assert re.search(r"(?m)^    environment:\n      name: pypi$", publish)
    assert "https://pypi.org/project/pixipix/" in publish


def test_only_tag_gated_publish_job_receives_oidc_permission() -> None:
    text = _workflow_text()
    build = _job(text, "build")
    canonical = _job(text, "canonical")
    publish = _job(text, "publish")

    assert "id-token: write" not in build
    assert "id-token: write" not in canonical
    assert text.count("id-token: write") == 1
    assert "id-token: write" in publish
    assert "github.event_name == 'push'" in publish
    assert "github.ref_type == 'tag'" in publish
    assert "startsWith(github.ref, 'refs/tags/v')" in publish
    assert "needs: build" in canonical
    assert "needs: canonical" in publish


def test_publish_job_only_downloads_and_publishes_verified_artifact() -> None:
    text = _workflow_text()
    build = _job(text, "build")
    canonical = _job(text, "canonical")
    publish = _job(text, "publish")

    assert "actions/download-artifact@" in publish
    assert f"name: {RELEASE_ARTIFACT}" in publish
    assert "uv build" not in publish
    assert "actions/upload-artifact@" not in publish
    assert "sha256sum" in publish
    assert "skip-existing: false" in publish
    assert "packages-dir: dist/" in publish
    assert text.count("actions/upload-artifact@") == 1
    assert "actions/upload-artifact@" in build
    assert "actions/upload-artifact@" not in canonical


def test_workflow_has_no_publish_credentials_or_untrusted_target_trigger() -> None:
    text = _workflow_text()

    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert re.search(r"(?m)^\s*(?:password|username|api-token|token)\s*:", text) is None
    assert "write-all" not in text


def test_every_action_is_pinned_to_an_expected_immutable_release() -> None:
    text = _workflow_text()
    references = {
        (match.group("action"), match.group("sha"), match.group("version"))
        for match in ACTION_REFERENCE.finditer(text)
    }

    assert references == {
        ("actions/checkout", "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0", "v7.0.0"),
        ("actions/setup-python", "a309ff8b426b58ec0e2a45f0f869d46889d02405", "v6.2.0"),
        ("astral-sh/setup-uv", "11f9893b081a58869d3b5fccaea48c9e9e46f990", "v8.3.2"),
        ("actions/upload-artifact", "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f", "v7.0.0"),
        ("actions/download-artifact", "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", "v8.0.1"),
        (
            "pypa/gh-action-pypi-publish",
            "cef221092ed1bacb1cc03d23a2d87d1d172e277b",
            "v1.14.0",
        ),
    }
    assert len(ACTION_REFERENCE.findall(text)) == 9


def test_pr_and_manual_runs_build_without_reaching_publish() -> None:
    text = _workflow_text()
    build = _job(text, "build")

    assert re.search(r"(?m)^  pull_request:$", text)
    assert re.search(r"(?m)^  workflow_dispatch:$", text)
    assert "uv sync --locked --all-groups" in build
    assert "uv run ruff format --check ." in build
    assert "uv run ruff check ." in build
    assert "uv run mypy src tests" in build
    assert "uv run pytest" in build
    assert "fetch-depth: 0" in build
    assert (
        "--deselect=tests/parity/test_parity.py::"
        "test_current_behavior_matches_explicit_v0_1_1_authority" in build
    )
    assert (
        "--deselect=tests/parity/test_parity.py::"
        "test_active_release_gate_requires_the_canonical_runtime" in build
    )
    for artifact_name in ("direct_wheel", "rebuilt_wheel"):
        deselection = (
            "--deselect='tests/release/test_smoke_distribution.py::"
            "test_installed_artifact_matches_active_release_authority"
            f"[{artifact_name}]'"
        )
        assert build.count(deselection) == 1
    assert "inspect-dist" in build
    assert "compare-wheels" in build
    assert build.count("scripts/smoke_distribution.py") == 2
    assert build.count("--artifact") == 2
    assert "--artifact dist/*.whl" in build
    assert '--artifact "$RUNNER_TEMP"/wheel-from-sdist/*.whl' in build
    assert "--with dist/*.tar.gz" not in build
    assert f"name: {RELEASE_ARTIFACT}" in build
    assert "sha256sum dist/*.whl dist/*.tar.gz" in build


def test_canonical_release_authority_runs_on_exact_runtime() -> None:
    text = _workflow_text()
    canonical = _job(text, "canonical")

    assert "runs-on: macos-15" in canonical
    assert 'python-version: "3.12.12"' in canonical
    assert "uv python install 3.12.12" in canonical
    assert '("3.12.12", "darwin", "arm64")' in canonical
    assert "test_current_behavior_matches_explicit_v0_1_1_authority" in canonical
    assert "test_active_release_gate_requires_the_canonical_runtime" in canonical
    for artifact_name in ("direct_wheel", "rebuilt_wheel"):
        node = (
            "'tests/release/test_smoke_distribution.py::"
            "test_installed_artifact_matches_active_release_authority"
            f"[{artifact_name}]'"
        )
        assert canonical.count(node) == 1


def test_canonical_verifies_the_exact_publish_candidate() -> None:
    text = _workflow_text()
    build = _job(text, "build")
    canonical = _job(text, "canonical")
    publish = _job(text, "publish")

    assert canonical.count("actions/download-artifact@") == 1
    assert f"name: {RELEASE_ARTIFACT}" in canonical
    assert "path: ${{ runner.temp }}/release-candidate" in canonical
    assert "PIXIPIX_RELEASE_CANDIDATE_DIR: ${{ runner.temp }}/release-candidate" in canonical
    assert "uv build --wheel" not in canonical
    assert "uv build --sdist" not in canonical
    assert 'wheels=("$RUNNER_TEMP"/release-candidate/*.whl)' in canonical
    assert 'sdists=("$RUNNER_TEMP"/release-candidate/*.tar.gz)' in canonical
    assert "shasum -a 256" in canonical

    for job in (build, canonical, publish):
        assert f"name: {RELEASE_ARTIFACT}" in job
    assert publish.count("actions/download-artifact@") == 1
    assert "sha256sum" in publish


def test_distribution_smoke_uses_complete_structured_stage_contract() -> None:
    assert SMOKE_STAGES == ("inspect", "extract", "scale", "pixelize", "align")
