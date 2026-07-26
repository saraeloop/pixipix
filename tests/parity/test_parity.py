from __future__ import annotations

import base64
import copy
import subprocess
from pathlib import Path

import pytest

from tests.parity.support import (
    BASELINE_PATH,
    PROJECT_ROOT,
    ParityError,
    _artifact_records,
    canonical_runtime_mismatch,
    capture_behavior,
    capture_environment,
    compare_behavior,
    load_baseline,
    sha256_bytes,
)


def _repository_state() -> tuple[bytes, bytes]:
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    staged = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return status, staged


def _require_canonical_runtime(
    expected_environment: object,
    actual_environment: dict[str, object],
) -> None:
    mismatch = canonical_runtime_mismatch(expected_environment, actual_environment)
    if mismatch is not None:
        pytest.skip(mismatch)


def test_current_behavior_matches_immutable_post_m3_baseline(tmp_path: Path) -> None:
    expected = load_baseline()
    _require_canonical_runtime(
        expected.get("environment"),
        capture_environment(PROJECT_ROOT),
    )
    baseline_before = sha256_bytes(BASELINE_PATH.read_bytes())
    repository_before = _repository_state()

    actual = capture_behavior(PROJECT_ROOT, tmp_path / "execution")

    compare_behavior(expected, actual)
    assert sha256_bytes(BASELINE_PATH.read_bytes()) == baseline_before
    assert _repository_state() == repository_before


def test_noncanonical_runtime_skips_before_behavior_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = load_baseline()
    environment = baseline["environment"]
    assert isinstance(environment, dict)
    noncanonical = {
        **environment,
        "pythonVersion": "3.12.13",
        "platformIdentifier": "linux-x86_64",
    }

    monkeypatch.setattr(
        "tests.parity.test_parity.capture_environment",
        lambda _source_root: noncanonical,
    )

    def fail_capture(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("behavior capture must not run outside the canonical runtime")

    monkeypatch.setattr("tests.parity.test_parity.capture_behavior", fail_capture)
    with pytest.raises(pytest.skip.Exception, match="linux-x86_64"):
        test_current_behavior_matches_immutable_post_m3_baseline(tmp_path)


def test_baseline_cases_capture_complete_ordered_contracts() -> None:
    baseline = load_baseline()
    environment = baseline["environment"]
    assert isinstance(environment, dict)
    noncanonical = {**environment, "pythonVersion": "3.12.13"}
    with pytest.raises(pytest.skip.Exception, match="pythonVersion"):
        _require_canonical_runtime(environment, noncanonical)
    dependency_drift = {**environment, "numpy": "different"}
    _require_canonical_runtime(environment, dependency_drift)
    cases = baseline["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 16
    for raw_case in cases:
        assert isinstance(raw_case, dict)
        case = raw_case
        assert case["kind"] in {"cli-byte-parity", "structural-api-parity"}
        contracts = case["parityContracts"]
        assert isinstance(contracts, list)
        assert case["kind"] in contracts
        artifacts = case["artifacts"]
        assert isinstance(artifacts, list)
        paths = [artifact["path"] for artifact in artifacts]
        assert paths == sorted(paths)
        for artifact in artifacts:
            assert set(artifact) == {"path", "byteLength", "sha256"}
            assert isinstance(artifact["byteLength"], int)
            assert artifact["byteLength"] > 0
            assert isinstance(artifact["sha256"], str)
            assert len(artifact["sha256"]) == 64
        if artifacts:
            assert "persisted-artifact-parity" in contracts
            assert ".pixipix-output" in paths
            assert "stage.json" in paths
            assert any(path.startswith("frames/") and path.endswith(".png") for path in paths)
            assert isinstance(case["stageTreeSha256"], str)
        if case["kind"] == "cli-byte-parity":
            assert case["exitCode"] == 0
            base64.b64decode(case["stdoutBase64"], validate=True)
            base64.b64decode(case["stderrBase64"], validate=True)
            warning_lines = case["warningLinesBase64"]
            assert isinstance(warning_lines, list)
            for line in warning_lines:
                assert base64.b64decode(line, validate=True).startswith(b"pixipix: warning ")
        else:
            assert "repr" not in case


def test_pixi_warning_order_is_frozen() -> None:
    baseline = load_baseline()
    cases = baseline["cases"]
    assert isinstance(cases, list)
    align = next(case for case in cases if case["caseId"] == "pixi.cli.align")
    warnings = [base64.b64decode(line) for line in align["warningLinesBase64"]]
    assert [line.split(b'"', 2)[1].decode("utf-8") for line in warnings] == [
        "pixi-fly-cube",
        "pixi-leap",
        "pixi-fly-wand",
        "pixi-fly",
    ]


def test_parity_mismatch_reports_missing_unexpected_and_changed_artifacts() -> None:
    expected = load_baseline()
    actual = copy.deepcopy(expected)
    cases = actual["cases"]
    assert isinstance(cases, list)
    case = next(item for item in cases if item["caseId"] == "robot.cli.align")
    artifacts = case["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0]["sha256"] = "0" * 64
    removed = artifacts.pop(1)
    artifacts.append(
        {
            "path": "frames/unexpected.png",
            "byteLength": 1,
            "sha256": "1" * 64,
        }
    )
    with pytest.raises(ParityError) as failure:
        compare_behavior(expected, actual)
    message = str(failure.value)
    assert "changed artifacts ['.pixipix-output']" in message
    assert f"missing artifacts ['{removed['path']}']" in message
    assert "unexpected artifacts ['frames/unexpected.png']" in message


def test_artifact_discovery_includes_hidden_nested_and_zero_byte_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage"
    (root / "nested").mkdir(parents=True)
    (root / ".hidden").write_bytes(b"marker")
    (root / "nested" / "empty.bin").write_bytes(b"")
    records, _tree_hash = _artifact_records(root)
    assert [(record["path"], record["byteLength"]) for record in records] == [
        (".hidden", 6),
        ("nested/empty.bin", 0),
    ]


def test_baseline_can_be_verified_from_a_read_only_copy(tmp_path: Path) -> None:
    copy_path = tmp_path / "post-m3.json"
    copy_path.write_bytes(BASELINE_PATH.read_bytes())
    copy_path.chmod(0o444)
    assert load_baseline(copy_path) == load_baseline()
