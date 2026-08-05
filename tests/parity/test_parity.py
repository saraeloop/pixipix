from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from tests.parity.support import (
    BASELINE_PATH,
    HISTORICAL_BASELINE_SHA256,
    PROJECT_ROOT,
    RELEASE_BASELINE_PATH,
    RELEASE_FIELD_CLASSIFICATION,
    RELEASE_VERSION,
    ParityError,
    _artifact_records,
    canonical_runtime_mismatch,
    capture_behavior,
    capture_environment,
    compare_behavior,
    load_baseline,
    load_release_baseline,
    sha256_bytes,
)

HISTORICAL_VERSION = "0.1.0a4"


def _case_map(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = manifest["cases"]
    assert isinstance(cases, list)
    return {cast(str, case["caseId"]): case for case in cases}


def _artifact_map(case: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = case["artifacts"]
    assert isinstance(artifacts, list)
    return {cast(str, artifact["path"]): artifact for artifact in artifacts}


def _normalize_release_version(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                HISTORICAL_VERSION
                if key == "pixipixVersion" and item == RELEASE_VERSION
                else _normalize_release_version(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_release_version(item) for item in value]
    return value


def _assert_release_transition(
    historical: dict[str, object],
    release: dict[str, object],
) -> None:
    assert release["environment"] == historical["environment"]
    assert release["fixtureSha256"] == historical["fixtureSha256"]
    assert release["uvLockSha256"] != historical["uvLockSha256"]
    historical_cases = _case_map(historical)
    release_cases = _case_map(release)
    assert list(release_cases) == list(historical_cases)
    for case_id, historical_case in historical_cases.items():
        release_case = release_cases[case_id]
        historical_artifacts = _artifact_map(historical_case)
        release_artifacts = _artifact_map(release_case)
        assert list(release_artifacts) == list(historical_artifacts)
        for path, historical_artifact in historical_artifacts.items():
            if path != "stage.json":
                assert release_artifacts[path] == historical_artifact
        historical_behavior = {
            key: item
            for key, item in historical_case.items()
            if key not in {"artifacts", "stageTreeSha256"}
        }
        release_behavior = {
            key: item
            for key, item in release_case.items()
            if key not in {"artifacts", "stageTreeSha256"}
        }
        assert _normalize_release_version(release_behavior) == historical_behavior


def _assert_stage_artifacts_are_version_only(
    execution_root: Path,
    historical: dict[str, object],
    release: dict[str, object],
) -> None:
    historical_cases = _case_map(historical)
    release_cases = _case_map(release)
    for case_id, release_case in release_cases.items():
        release_artifacts = _artifact_map(release_case)
        if "stage.json" not in release_artifacts:
            continue
        lineage, boundary, stage = case_id.split(".", maxsplit=2)
        if boundary == "cli":
            output = execution_root / "cli" / lineage / stage
        else:
            assert case_id.startswith("robot.api.publish-")
            api_stage = stage.removeprefix("publish-")
            if api_stage == "extraction":
                api_stage = "extract"
            output = execution_root / "api" / "robot" / api_stage
        stage_path = output / "stage.json"
        current = stage_path.read_bytes()
        current_token = f'"pixipixVersion": "{RELEASE_VERSION}"'.encode()
        historical_token = f'"pixipixVersion": "{HISTORICAL_VERSION}"'.encode()
        occurrences = current.count(current_token)
        assert occurrences in {1, 2}
        normalized_stage = current.replace(current_token, historical_token)
        assert current_token not in normalized_stage
        historical_artifact = _artifact_map(historical_cases[case_id])["stage.json"]
        assert len(normalized_stage) == historical_artifact["byteLength"]
        assert sha256_bytes(normalized_stage) == historical_artifact["sha256"]
        tree = hashlib.sha256()
        for path in sorted(path for path in output.rglob("*") if path.is_file()):
            relative = path.relative_to(output).as_posix()
            content = normalized_stage if relative == "stage.json" else path.read_bytes()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(content)
        assert tree.hexdigest() == historical_cases[case_id]["stageTreeSha256"]


def _classify_historical_leaf(path: tuple[str, ...]) -> str:
    if path[0] in {
        "baselineCleanWorktreeAttestation",
        "baselineProductionCommit",
        "environment",
        "fixtureSha256",
        "schemaVersion",
        "uvLockSha256",
    }:
        return "repositoryToolchainIdentity"
    assert path[0] == "cases"
    if "pixipixVersion" in path:
        return "releaseIdentity"
    if "artifacts" in path or path[-1] == "stageTreeSha256":
        return "artifactIdentity"
    return "behavioral"


def _classified_leaves(
    value: object,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    if isinstance(value, dict):
        result: list[tuple[tuple[str, ...], str]] = []
        for key, item in value.items():
            result.extend(_classified_leaves(item, (*path, key)))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_classified_leaves(item, (*path, str(index))))
        return result
    return [(path, _classify_historical_leaf(path))]


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


def test_current_behavior_matches_explicit_v0_1_0_authority(tmp_path: Path) -> None:
    expected = load_release_baseline()
    historical = load_baseline()
    _require_canonical_runtime(
        expected.get("environment"),
        capture_environment(PROJECT_ROOT),
    )
    historical_before = sha256_bytes(BASELINE_PATH.read_bytes())
    release_before = sha256_bytes(RELEASE_BASELINE_PATH.read_bytes())
    repository_before = _repository_state()
    execution_root = tmp_path / "execution"

    actual = capture_behavior(PROJECT_ROOT, execution_root)

    compare_behavior(expected, actual)
    _assert_stage_artifacts_are_version_only(execution_root, historical, expected)
    assert sha256_bytes(BASELINE_PATH.read_bytes()) == historical_before
    assert sha256_bytes(RELEASE_BASELINE_PATH.read_bytes()) == release_before
    assert _repository_state() == repository_before


def test_v0_1_0_authority_preserves_historical_behavior_and_provenance() -> None:
    historical = load_baseline()
    release = load_release_baseline()

    assert sha256_bytes(BASELINE_PATH.read_bytes()) == HISTORICAL_BASELINE_SHA256
    assert release["releaseVersion"] == RELEASE_VERSION
    assert release["historicalBaseline"] == {
        "path": BASELINE_PATH.name,
        "sha256": HISTORICAL_BASELINE_SHA256,
    }
    assert release["fieldClassification"] == RELEASE_FIELD_CLASSIFICATION
    _assert_release_transition(historical, release)


def test_every_historical_baseline_leaf_has_one_explicit_classification() -> None:
    classified = _classified_leaves(load_baseline())
    categories = {category for _path, category in classified}

    assert classified
    assert categories == {
        "artifactIdentity",
        "behavioral",
        "releaseIdentity",
        "repositoryToolchainIdentity",
    }


def test_release_candidate_lock_transition_changes_only_root_version() -> None:
    old = b'name = "pixipix"\nversion = "0.1.0a4"\nsource = { editable = "." }'
    new = b'name = "pixipix"\nversion = "0.1.0"\nsource = { editable = "." }'
    current = (PROJECT_ROOT / "uv.lock").read_bytes()
    assert current.count(new) == 1
    historical = current.replace(new, old)

    assert sha256_bytes(historical) == load_baseline()["uvLockSha256"]
    assert sha256_bytes(current) == load_release_baseline()["uvLockSha256"]


def test_release_authority_rejects_behavioral_and_identity_mutations() -> None:
    expected = load_release_baseline()

    attacks: dict[str, dict[str, object]] = {}
    lock = copy.deepcopy(expected)
    lock["uvLockSha256"] = "0" * 64
    attacks["unrelated lock change"] = lock
    png = copy.deepcopy(expected)
    _artifact_map(_case_map(png)["robot.cli.extract"])["frames/idle.png"]["sha256"] = "0" * 64
    attacks["PNG change"] = png
    tree = copy.deepcopy(expected)
    _case_map(tree)["robot.cli.align"]["stageTreeSha256"] = "0" * 64
    attacks["tree hash change"] = tree
    warning = copy.deepcopy(expected)
    _case_map(warning)["pixi.cli.align"]["warningLinesBase64"] = []
    attacks["warning change"] = warning
    error = copy.deepcopy(expected)
    error_record = _case_map(error)["robot.api.resource-policy-error"]["error"]
    assert isinstance(error_record, dict)
    error_record["code"] = "PX_RESOURCE_999"
    attacks["error code change"] = error
    order = copy.deepcopy(expected)
    result = _case_map(order)["robot.api.publish-scale"]["result"]
    assert isinstance(result, dict)
    frames = result["frames"]
    assert isinstance(frames, list)
    frames.reverse()
    attacks["frame order change"] = order
    prerelease = copy.deepcopy(expected)
    prerelease_result = _case_map(prerelease)["robot.api.publish-scale"]["result"]
    assert isinstance(prerelease_result, dict)
    prerelease_result["pixipixVersion"] = HISTORICAL_VERSION
    attacks["prerelease reintroduction"] = prerelease

    for attack in attacks.values():
        with pytest.raises(ParityError):
            compare_behavior(expected, attack)


@pytest.mark.parametrize("version", [HISTORICAL_VERSION, "0.1.1"])
def test_release_authority_rejects_wrong_version(
    version: str,
    tmp_path: Path,
) -> None:
    authority = copy.deepcopy(load_release_baseline())
    authority["releaseVersion"] = version
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(ParityError, match="wrong release version"):
        load_release_baseline(path)


def test_release_transition_rejects_historical_mutation() -> None:
    historical = copy.deepcopy(load_baseline())
    release = load_release_baseline()
    _artifact_map(_case_map(historical)["robot.cli.extract"])["frames/idle.png"]["sha256"] = (
        "0" * 64
    )

    with pytest.raises(AssertionError):
        _assert_release_transition(historical, release)


def test_release_authority_rejects_missing_behavioral_fields(tmp_path: Path) -> None:
    authority = copy.deepcopy(load_release_baseline())
    del authority["cases"]
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(ParityError, match="fields are incomplete"):
        load_release_baseline(path)


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
        test_current_behavior_matches_explicit_v0_1_0_authority(tmp_path)


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
