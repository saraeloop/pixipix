"""Standalone direct-API probe loaded against a selected PixiPix source checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

from pixipix.config import load_config
from pixipix.errors import PixiPixError, ResourcePolicyError
from pixipix.resources import ResourcePolicy, ResourceProjection, enforce_resource_policy
from pixipix.serialization import to_json_data
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale


def _identity(output: Path) -> dict[str, object]:
    metadata = cast(dict[str, object], json.loads((output / "stage.json").read_bytes()))
    return {
        "sourceConfigSha256": metadata["sourceConfigSha256"],
        "effectiveConfigSha256": metadata["effectiveConfigSha256"],
    }


def _success_case(identifier: str, stage: str, result: object, output: Path) -> dict[str, object]:
    return {
        "caseId": identifier,
        "kind": "structural-api-parity",
        "stage": stage,
        "returnType": type(result).__name__,
        "result": to_json_data(result),
        "publishedIdentity": _identity(output),
        "resourceProjectionAttached": hasattr(result, "projection"),
        "outputRelative": output.as_posix(),
    }


def _error_case(root: Path, loaded_config: Path) -> dict[str, object]:
    output = Path("api/robot/missing-input-output")
    try:
        publish_scale(Path("api/robot/missing-input"), load_config(loaded_config), output)
    except PixiPixError as error:
        return {
            "caseId": "robot.api.publish-scale-missing-input",
            "kind": "structural-api-parity",
            "stage": "scale",
            "exceptionType": type(error).__name__,
            "error": to_json_data(error),
            "outputExists": (root / output).exists(),
        }
    raise AssertionError("missing input unexpectedly published")


def _resource_error_case() -> dict[str, object]:
    projection = ResourceProjection("scale", 2, 3, 4)
    policy = ResourcePolicy(1, 2, 3)
    try:
        enforce_resource_policy(projection, policy)
    except ResourcePolicyError as error:
        return {
            "caseId": "robot.api.resource-policy-error",
            "kind": "structural-api-parity",
            "stage": "scale",
            "exceptionType": type(error).__name__,
            "error": to_json_data(error),
            "projection": to_json_data(error.projection),
            "policy": to_json_data(error.policy),
            "findings": to_json_data(error.findings),
            "outputExists": False,
        }
    raise AssertionError("resource policy unexpectedly admitted")


def _run_case(loaded_config: Path) -> dict[str, object]:
    import pixipix
    import pixipix.pipeline as pipeline_package
    from pixipix.pipeline.run import PipelineRunResult, run_pipeline

    output = Path("api/robot/run")
    result = run_pipeline(
        Path("inputs/robot-geometric.png"),
        load_config(loaded_config),
        output,
    )
    assert isinstance(result, PipelineRunResult)
    stage_order = [
        cast(
            dict[str, object],
            json.loads((output / stage / "stage.json").read_bytes()),
        )["stage"]
        for stage in ("extract", "scale", "pixelize", "align")
    ]
    return {
        "caseId": "robot.api.run",
        "kind": "structural-api-parity",
        "stage": "run",
        "returnType": type(result).__name__,
        "runPipelineOwner": run_pipeline.__module__,
        "resultTypeOwner": PipelineRunResult.__module__,
        "stageOrder": stage_order,
        "warnings": to_json_data(result.warnings),
        "outputExists": output.is_dir(),
        "packageRootExport": hasattr(pixipix, "run_pipeline"),
        "pipelineRootExport": hasattr(pipeline_package, "run_pipeline"),
    }


def main() -> None:
    root = Path(sys.argv[1])
    image = Path("inputs/robot-geometric.png")
    config = Path("inputs/robot-geometric.toml")
    loaded = load_config(config)
    outputs = {
        "extract": Path("api/robot/extract"),
        "scale": Path("api/robot/scale"),
        "pixelize": Path("api/robot/pixelize"),
        "align": Path("api/robot/align"),
    }
    extracted = publish_extraction(image, loaded, outputs["extract"])
    scaled = publish_scale(outputs["extract"], loaded, outputs["scale"])
    pixelized = publish_pixelize(outputs["scale"], loaded, outputs["pixelize"])
    aligned = publish_align(outputs["pixelize"], loaded, outputs["align"])
    cases = [
        _success_case("robot.api.publish-extraction", "extract", extracted, outputs["extract"]),
        _success_case("robot.api.publish-scale", "scale", scaled, outputs["scale"]),
        _success_case("robot.api.publish-pixelize", "pixelize", pixelized, outputs["pixelize"]),
        _success_case("robot.api.publish-align", "align", aligned, outputs["align"]),
        _error_case(root, config),
        _resource_error_case(),
    ]
    if len(sys.argv) == 3 and sys.argv[2] == "--include-run":
        cases.append(_run_case(config))
    print(json.dumps(cases, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
