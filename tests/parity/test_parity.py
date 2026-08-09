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
    ACTIVE_RELEASE_AUTHORITY,
    BASELINE_PATH,
    HISTORICAL_BASELINE_SHA256,
    PROJECT_ROOT,
    RELEASE_AUTHORITIES,
    RELEASE_BASELINE_PATH,
    RELEASE_FIELD_CLASSIFICATION,
    RELEASE_VERSION,
    RUN_CASE_IDS,
    RUN_STAGES,
    V0_1_0_AUTHORITY,
    V0_1_0_AUTHORITY_SHA256,
    V0_1_1_AUTHORITY,
    V0_1_1_AUTHORITY_SHA256,
    V0_2_0_AUTHORITY,
    V0_2_0_RELEASE_FIELD_CLASSIFICATION,
    ParityError,
    ReleaseAuthority,
    _artifact_records,
    canonical_runtime_mismatch,
    capture_behavior,
    capture_environment,
    compare_behavior,
    load_baseline,
    load_release_baseline,
    normalize_pixipix_version,
    require_canonical_runtime,
    sha256_bytes,
)

POST_M3_VERSION = "0.1.0a4"
V0_1_1_PREVIOUS_VERSION = "0.1.0"
PREVIOUS_RELEASE_VERSION = "0.1.1"


def _case_map(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = manifest["cases"]
    assert isinstance(cases, list)
    return {cast(str, case["caseId"]): case for case in cases}


def _artifact_map(case: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = case["artifacts"]
    assert isinstance(artifacts, list)
    return {cast(str, artifact["path"]): artifact for artifact in artifacts}


def _assert_release_transition(
    historical: dict[str, object],
    release: dict[str, object],
    *,
    historical_version: str,
    release_version: str,
    added_case_ids: tuple[str, ...] = (),
) -> None:
    assert release["environment"] == historical["environment"]
    assert release["fixtureSha256"] == historical["fixtureSha256"]
    assert release["uvLockSha256"] != historical["uvLockSha256"]
    historical_cases = _case_map(historical)
    release_cases = _case_map(release)
    assert [case_id for case_id in release_cases if case_id not in added_case_ids] == list(
        historical_cases
    )
    assert tuple(case_id for case_id in release_cases if case_id not in historical_cases) == (
        added_case_ids
    )
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
        normalized_behavior, _replacements = normalize_pixipix_version(
            release_behavior,
            current_version=release_version,
            historical_version=historical_version,
        )
        assert normalized_behavior == historical_behavior


def _stage_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _assert_stage_artifacts_are_version_only(
    execution_root: Path,
    historical: dict[str, object],
    release: dict[str, object],
    *,
    historical_version: str,
    release_version: str,
) -> None:
    historical_cases = _case_map(historical)
    release_cases = _case_map(release)
    for case_id in historical_cases:
        release_case = release_cases[case_id]
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
        current = json.loads(stage_path.read_bytes())
        normalized_value, replacements = normalize_pixipix_version(
            current,
            current_version=release_version,
            historical_version=historical_version,
        )
        assert replacements in {1, 2}
        normalized_stage = _stage_json_bytes(normalized_value)
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


def test_current_behavior_matches_explicit_v0_2_0_authority(tmp_path: Path) -> None:
    expected = load_release_baseline()
    historical = load_release_baseline(authority=V0_1_1_AUTHORITY)
    _require_canonical_runtime(
        expected.get("environment"),
        capture_environment(PROJECT_ROOT),
    )
    post_m3_before = sha256_bytes(BASELINE_PATH.read_bytes())
    historical_before = sha256_bytes(V0_1_1_AUTHORITY.path.read_bytes())
    release_before = sha256_bytes(RELEASE_BASELINE_PATH.read_bytes())
    repository_before = _repository_state()
    execution_root = tmp_path / "execution"

    actual = capture_behavior(PROJECT_ROOT, execution_root)

    compare_behavior(expected, actual)
    _assert_stage_artifacts_are_version_only(
        execution_root,
        historical,
        expected,
        historical_version=PREVIOUS_RELEASE_VERSION,
        release_version=RELEASE_VERSION,
    )
    assert sha256_bytes(BASELINE_PATH.read_bytes()) == post_m3_before
    assert sha256_bytes(V0_1_1_AUTHORITY.path.read_bytes()) == historical_before
    assert sha256_bytes(RELEASE_BASELINE_PATH.read_bytes()) == release_before
    assert _repository_state() == repository_before


def test_active_release_gate_requires_the_canonical_runtime() -> None:
    expected = load_release_baseline()

    require_canonical_runtime(expected.get("environment"), capture_environment(PROJECT_ROOT))


def test_release_authority_roster_and_active_selection_are_exact() -> None:
    assert RELEASE_AUTHORITIES == (V0_1_0_AUTHORITY, V0_1_1_AUTHORITY, V0_2_0_AUTHORITY)
    assert ACTIVE_RELEASE_AUTHORITY is V0_2_0_AUTHORITY


def test_release_gate_rejects_noncanonical_runtime_without_a_skip() -> None:
    expected = load_release_baseline()["environment"]
    assert isinstance(expected, dict)
    noncanonical = {**expected, "platformIdentifier": "linux-x86_64"}

    with pytest.raises(ParityError, match="canonical runtime"):
        require_canonical_runtime(expected, noncanonical)


def test_v0_2_0_authority_preserves_v0_1_1_behavior_and_adds_run() -> None:
    historical = load_release_baseline(authority=V0_1_1_AUTHORITY)
    release = load_release_baseline()

    assert sha256_bytes(V0_1_1_AUTHORITY.path.read_bytes()) == V0_1_1_AUTHORITY_SHA256
    assert release["releaseVersion"] == RELEASE_VERSION
    assert release["historicalBaseline"] == {
        "path": V0_1_1_AUTHORITY.path.name,
        "sha256": V0_1_1_AUTHORITY_SHA256,
    }
    assert release["fieldClassification"] == V0_2_0_RELEASE_FIELD_CLASSIFICATION
    _assert_release_transition(
        historical,
        release,
        historical_version=PREVIOUS_RELEASE_VERSION,
        release_version=RELEASE_VERSION,
        added_case_ids=RUN_CASE_IDS,
    )


def test_v0_2_0_authority_certifies_run_as_manual_byte_parity() -> None:
    cases = _case_map(load_release_baseline())
    assert tuple(case_id for case_id in cases if case_id in RUN_CASE_IDS) == RUN_CASE_IDS

    marker: dict[str, object] | None = None
    for lineage in ("pixi", "robot"):
        run = cases[f"{lineage}.cli.run"]
        run_artifacts = _artifact_map(run)
        assert set(run_artifacts) == {
            ".pixipix-run",
            *(
                f"{stage}/{path}"
                for stage in RUN_STAGES
                for path in _artifact_map(cases[f"{lineage}.cli.{stage}"])
            ),
        }
        for stage in RUN_STAGES:
            manual = cases[f"{lineage}.cli.{stage}"]
            manual_artifacts = _artifact_map(manual)
            run_stage_artifacts = {
                path.removeprefix(f"{stage}/"): {
                    **artifact,
                    "path": path.removeprefix(f"{stage}/"),
                }
                for path, artifact in run_artifacts.items()
                if path.startswith(f"{stage}/")
            }
            assert run_stage_artifacts == manual_artifacts
            tree_hashes = run["runStageTreeSha256"]
            assert isinstance(tree_hashes, dict)
            assert tree_hashes[stage] == manual["stageTreeSha256"]
        assert run["warningLinesBase64"] == cases[f"{lineage}.cli.align"]["warningLinesBase64"]
        if marker is None:
            marker = run_artifacts[".pixipix-run"]
        else:
            assert run_artifacts[".pixipix-run"] == marker

    assert marker is not None
    marker_length = marker["byteLength"]
    assert isinstance(marker_length, int)
    assert marker_length > 0
    api = cases["robot.api.run"]
    assert api == {
        "caseId": "robot.api.run",
        "kind": "structural-api-parity",
        "stage": "run",
        "returnType": "PipelineRunResult",
        "runPipelineOwner": "pixipix.pipeline.run",
        "resultTypeOwner": "pixipix.pipeline.run",
        "stageOrder": list(RUN_STAGES),
        "warnings": [],
        "outputExists": True,
        "packageRootExport": False,
        "pipelineRootExport": False,
        "parityContracts": ["structural-api-parity"],
        "artifacts": [],
        "stageTreeSha256": None,
    }


def test_v0_1_1_authority_preserves_v0_1_0_behavior_and_provenance() -> None:
    historical = load_release_baseline(authority=V0_1_0_AUTHORITY)
    release = load_release_baseline(authority=V0_1_1_AUTHORITY)

    assert sha256_bytes(V0_1_0_AUTHORITY.path.read_bytes()) == V0_1_0_AUTHORITY_SHA256
    assert release["releaseVersion"] == V0_1_1_AUTHORITY.version
    assert release["historicalBaseline"] == {
        "path": V0_1_0_AUTHORITY.path.name,
        "sha256": V0_1_0_AUTHORITY_SHA256,
    }
    assert release["fieldClassification"] == RELEASE_FIELD_CLASSIFICATION
    _assert_release_transition(
        historical,
        release,
        historical_version=V0_1_1_PREVIOUS_VERSION,
        release_version=V0_1_1_AUTHORITY.version,
    )


def test_v0_1_0_authority_preserves_post_m3_behavior_and_provenance() -> None:
    historical = load_baseline()
    release = load_release_baseline(authority=V0_1_0_AUTHORITY)

    assert sha256_bytes(BASELINE_PATH.read_bytes()) == HISTORICAL_BASELINE_SHA256
    assert release["releaseVersion"] == V0_1_0_AUTHORITY.version
    assert release["historicalBaseline"] == {
        "path": BASELINE_PATH.name,
        "sha256": HISTORICAL_BASELINE_SHA256,
    }
    assert release["fieldClassification"] == RELEASE_FIELD_CLASSIFICATION
    _assert_release_transition(
        historical,
        release,
        historical_version=POST_M3_VERSION,
        release_version=V0_1_0_AUTHORITY.version,
    )


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


def _assert_lock_transition_changes_only_root_version(
    *,
    authority: ReleaseAuthority,
    historical_authority: ReleaseAuthority,
    old_version: str,
    new_version: str,
    current: bytes,
) -> None:
    historical = subprocess.run(
        ["git", "show", f"{authority.preparation_commit}:uv.lock"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    old = f'name = "pixipix"\nversion = "{old_version}"\nsource = {{ editable = "." }}'.encode()
    new = f'name = "pixipix"\nversion = "{new_version}"\nsource = {{ editable = "." }}'.encode()
    assert historical.count(old) == 1
    expected = historical.replace(old, new)
    assert current == expected
    assert (
        sha256_bytes(historical)
        == load_release_baseline(authority=historical_authority)["uvLockSha256"]
    )
    assert sha256_bytes(current) == load_release_baseline(authority=authority)["uvLockSha256"]


def test_historical_authority_hashes_remain_immutable() -> None:
    assert sha256_bytes(BASELINE_PATH.read_bytes()) == HISTORICAL_BASELINE_SHA256
    assert sha256_bytes(V0_1_0_AUTHORITY.path.read_bytes()) == V0_1_0_AUTHORITY_SHA256
    assert sha256_bytes(V0_1_1_AUTHORITY.path.read_bytes()) == V0_1_1_AUTHORITY_SHA256


def test_v0_1_1_lock_transition_changes_only_root_version() -> None:
    _assert_lock_transition_changes_only_root_version(
        authority=V0_1_1_AUTHORITY,
        historical_authority=V0_1_0_AUTHORITY,
        old_version="0.1.0",
        new_version="0.1.1",
        current=subprocess.run(
            ["git", "show", "5c2e5b794860523d1fde21350ddd8da5f173f442:uv.lock"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout,
    )


def test_release_candidate_lock_transition_changes_only_root_version() -> None:
    _assert_lock_transition_changes_only_root_version(
        authority=V0_2_0_AUTHORITY,
        historical_authority=V0_1_1_AUTHORITY,
        old_version="0.1.1",
        new_version="0.2.0",
        current=(PROJECT_ROOT / "uv.lock").read_bytes(),
    )


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
    metadata = copy.deepcopy(expected)
    metadata_result = _case_map(metadata)["robot.api.publish-scale"]["result"]
    assert isinstance(metadata_result, dict)
    metadata_frames = metadata_result["frames"]
    assert isinstance(metadata_frames, list)
    first_frame = metadata_frames[0]
    assert isinstance(first_frame, dict)
    output_dimensions = first_frame["outputDimensions"]
    assert isinstance(output_dimensions, dict)
    width = output_dimensions["width"]
    assert isinstance(width, int)
    output_dimensions["width"] = width + 1
    attacks["metadata change"] = metadata
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
    prerelease_result["pixipixVersion"] = PREVIOUS_RELEASE_VERSION
    attacks["prerelease reintroduction"] = prerelease

    for attack in attacks.values():
        with pytest.raises(ParityError):
            compare_behavior(expected, attack)


@pytest.mark.parametrize("version", [PREVIOUS_RELEASE_VERSION, "0.2.1"])
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
    historical = copy.deepcopy(load_release_baseline(authority=V0_1_1_AUTHORITY))
    release = load_release_baseline()
    _artifact_map(_case_map(historical)["robot.cli.extract"])["frames/idle.png"]["sha256"] = (
        "0" * 64
    )

    with pytest.raises(AssertionError):
        _assert_release_transition(
            historical,
            release,
            historical_version=PREVIOUS_RELEASE_VERSION,
            release_version=RELEASE_VERSION,
            added_case_ids=RUN_CASE_IDS,
        )


def test_structural_version_normalization_does_not_touch_unrelated_values() -> None:
    value = {
        "pixipixVersion": RELEASE_VERSION,
        "note": f"release {RELEASE_VERSION}",
        "nested": {"otherVersion": RELEASE_VERSION},
    }

    normalized, replacements = normalize_pixipix_version(
        value,
        current_version=RELEASE_VERSION,
        historical_version=PREVIOUS_RELEASE_VERSION,
    )

    assert replacements == 1
    assert normalized == {
        "pixipixVersion": PREVIOUS_RELEASE_VERSION,
        "note": f"release {RELEASE_VERSION}",
        "nested": {"otherVersion": RELEASE_VERSION},
    }

    globally_replaced = json.loads(
        json.dumps(value).replace(RELEASE_VERSION, PREVIOUS_RELEASE_VERSION)
    )
    assert globally_replaced != normalized
    assert globally_replaced["note"] == f"release {PREVIOUS_RELEASE_VERSION}"


def test_release_authority_rejects_missing_behavioral_fields(tmp_path: Path) -> None:
    authority = copy.deepcopy(load_release_baseline())
    del authority["cases"]
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(ParityError, match="fields are incomplete"):
        load_release_baseline(path)


@pytest.mark.parametrize("case_id", RUN_CASE_IDS)
def test_v0_2_0_authority_rejects_missing_run_evidence(
    case_id: str,
    tmp_path: Path,
) -> None:
    authority = copy.deepcopy(load_release_baseline())
    cases = authority["cases"]
    assert isinstance(cases, list)
    authority["cases"] = [
        case for case in cases if isinstance(case, dict) and case.get("caseId") != case_id
    ]
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(ParityError, match="RUN evidence is incomplete"):
        load_release_baseline(path)


def test_release_authority_rejects_wrong_historical_baseline(tmp_path: Path) -> None:
    authority = copy.deepcopy(load_release_baseline())
    authority["historicalBaseline"] = {
        "path": "post-m3.json",
        "sha256": HISTORICAL_BASELINE_SHA256,
    }
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(ParityError, match="wrong historical provenance"):
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
        test_current_behavior_matches_explicit_v0_2_0_authority(tmp_path)


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
