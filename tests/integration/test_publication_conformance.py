from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
import pytest
from PIL import Image

import pixipix.pipeline.publication as pipeline_publication
from pixipix.config import LoadedConfig, load_config
from pixipix.errors import ProcessingError
from pixipix.pipeline.publication import OutputFrameImage
from pixipix.stages.align import publish_align
from pixipix.stages.extract import publish_extraction
from pixipix.stages.pixelize import publish_pixelize
from pixipix.stages.scale import publish_scale
from tests.helpers import pipeline_config, write_config, write_rgba

type PublicationStage = Literal["extract", "scale", "pixelize", "align"]
type DimensionMutation = Literal[
    "valid-rectangular",
    "width-only",
    "height-only",
    "both-axes",
]
type Mutation = Literal[
    "invalid-marker",
    "missing-stage",
    "malformed-stage",
    "failed-status",
    "stage-mismatch",
    "missing-frame",
    "extra-frame",
    "unsafe-frame-path",
    "duplicate-frame",
    "symlinked-frame",
    "wrong-mode",
    "png-dimension-mismatch",
    "metadata-dimension-mismatch",
    "corrupt-png",
]

STAGES: tuple[PublicationStage, ...] = ("extract", "scale", "pixelize", "align")


@dataclass(frozen=True, slots=True)
class DimensionAuthority:
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True, slots=True)
class DimensionCase:
    stage: PublicationStage
    mutation: DimensionMutation


DIMENSION_AUTHORITY = DimensionAuthority(width=3, height=2)
DIMENSION_EXPECTED_DELTAS: dict[DimensionMutation, tuple[int, int]] = {
    "valid-rectangular": (0, 0),
    "width-only": (1, 0),
    "height-only": (0, 1),
    "both-axes": (1, 1),
}
DIMENSION_EXPECTED_LEDGER: dict[DimensionMutation, tuple[PublicationStage, ...]] = {
    "valid-rectangular": ("extract", "scale", "pixelize", "align"),
    "width-only": ("extract", "scale", "pixelize", "align"),
    "height-only": ("extract", "scale", "pixelize", "align"),
    "both-axes": ("extract", "scale", "pixelize", "align"),
}
DIMENSION_CASES: tuple[DimensionCase, ...] = (
    DimensionCase("extract", "valid-rectangular"),
    DimensionCase("scale", "valid-rectangular"),
    DimensionCase("pixelize", "valid-rectangular"),
    DimensionCase("align", "valid-rectangular"),
    DimensionCase("extract", "width-only"),
    DimensionCase("scale", "width-only"),
    DimensionCase("pixelize", "width-only"),
    DimensionCase("align", "width-only"),
    DimensionCase("extract", "height-only"),
    DimensionCase("scale", "height-only"),
    DimensionCase("pixelize", "height-only"),
    DimensionCase("align", "height-only"),
    DimensionCase("extract", "both-axes"),
    DimensionCase("scale", "both-axes"),
    DimensionCase("pixelize", "both-axes"),
    DimensionCase("align", "both-axes"),
)
EXPECTED_STAGE_PUBLISHERS: tuple[tuple[PublicationStage, Callable[..., object]], ...] = (
    ("extract", publish_extraction),
    ("scale", publish_scale),
    ("pixelize", publish_pixelize),
    ("align", publish_align),
)
MUTATIONS: tuple[Mutation, ...] = (
    "invalid-marker",
    "missing-stage",
    "malformed-stage",
    "failed-status",
    "stage-mismatch",
    "missing-frame",
    "extra-frame",
    "unsafe-frame-path",
    "duplicate-frame",
    "symlinked-frame",
    "wrong-mode",
    "png-dimension-mismatch",
    "metadata-dimension-mismatch",
    "corrupt-png",
)
EXTRACT_METADATA_MUTATIONS: tuple[tuple[str, object], ...] = (
    ("source", []),
    ("background", []),
    ("candidateComponents", {}),
    ("acceptedComponents", {}),
    ("rejectedComponents", {}),
    ("orderedComponents", {}),
    ("warnings", {}),
    ("sourceConfigSha256", None),
    ("sourceConfigSha256", "0" * 63),
    ("sourceConfigSha256", "A" * 64),
    ("sourceConfigSha256", "g" * 64),
    ("effectiveConfigSha256", None),
    ("effectiveConfigSha256", "0" * 63),
    ("effectiveConfigSha256", "A" * 64),
    ("effectiveConfigSha256", "g" * 64),
    ("pixipixVersion", []),
)


@dataclass(frozen=True, slots=True)
class PublicationCase:
    stage: PublicationStage
    output: Path
    publish: Callable[[bool], object]


def _metadata(root: Path) -> dict[str, object]:
    value = json.loads((root / "stage.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _write_metadata(root: Path, value: dict[str, object]) -> None:
    (root / "stage.json").write_text(json.dumps(value), encoding="utf-8")


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _temporary_siblings(output: Path) -> tuple[Path, ...]:
    return tuple(output.parent.glob(f".{output.name}.pixipix-*-*"))


def _frame_dimensions(stage: PublicationStage, width: int, height: int) -> dict[str, object]:
    if stage == "extract":
        return {
            "paddedBounds": {
                "left": 10,
                "top": 20,
                "right": 10 + width,
                "bottom": 20 + height,
            }
        }
    if stage == "scale":
        return {"outputDimensions": {"width": width, "height": height}}
    if stage == "pixelize":
        return {"logicalOutputDimensions": {"width": width, "height": height}}
    return {"outputWidth": width, "outputHeight": height}


def _expected_dimensions(mutation: DimensionMutation) -> tuple[int, int]:
    width_delta, height_delta = DIMENSION_EXPECTED_DELTAS[mutation]
    return (
        DIMENSION_AUTHORITY.width + width_delta,
        DIMENSION_AUTHORITY.height + height_delta,
    )


def _test_owned_declared_dimensions(
    stage: PublicationStage, frame: dict[str, object]
) -> tuple[int, int]:
    if stage == "extract":
        bounds = frame["paddedBounds"]
        assert isinstance(bounds, dict)
        return (
            cast(int, bounds["right"]) - cast(int, bounds["left"]),
            cast(int, bounds["bottom"]) - cast(int, bounds["top"]),
        )
    if stage == "scale":
        dimensions = frame["outputDimensions"]
        assert isinstance(dimensions, dict)
        return cast(int, dimensions["width"]), cast(int, dimensions["height"])
    if stage == "pixelize":
        dimensions = frame["logicalOutputDimensions"]
        assert isinstance(dimensions, dict)
        return cast(int, dimensions["width"]), cast(int, dimensions["height"])
    return cast(int, frame["outputWidth"]), cast(int, frame["outputHeight"])


def _write_rectangular_owned_output(
    root: Path,
    case: DimensionCase,
) -> None:
    expected_width, expected_height = _expected_dimensions(case.mutation)
    frames = root / "frames"
    frames.mkdir(parents=True)
    (root / ".pixipix-output").write_text(
        json.dumps({"owner": "pixipix", "schemaVersion": 1, "stage": case.stage}),
        encoding="utf-8",
    )
    frame = {
        "sourceOrder": 0,
        "relativePath": "frames/rectangular.png",
        **_frame_dimensions(case.stage, expected_width, expected_height),
    }
    _write_metadata(
        root,
        {
            "schemaVersion": 1,
            "stage": case.stage,
            "status": "successful",
            "frames": [frame],
        },
    )
    Image.new("RGBA", DIMENSION_AUTHORITY.size, (1, 2, 3, 4)).save(
        frames / "rectangular.png", format="PNG"
    )


def _tiny_project(tmp_path: Path) -> tuple[Path, LoadedConfig]:
    source = tmp_path / "source.png"
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[1:3, 1:3] = (40, 80, 120, 255)
    write_rgba(source, pixels)
    config = tmp_path / "pixipix.toml"
    write_config(
        config,
        pipeline_config(
            names=("one",),
            pixelize=(
                "source_cell_size = 1\n"
                'representative = "center"\n'
                'alpha_policy = "preserve"\n'
                "alpha_threshold = 1\n"
                'remainder_policy = "pad-transparent"'
            ),
            output=('frame_width = 2\nframe_height = 2\nanchor = "center"\nclip_policy = "error"'),
        ),
    )
    return source, load_config(config)


def _publication_case(
    tmp_path: Path, stage: PublicationStage, *, publish_target: bool = True
) -> PublicationCase:
    source, loaded = _tiny_project(tmp_path)
    extracted = tmp_path / "prerequisite-extract"
    scaled = tmp_path / "prerequisite-scale"
    pixelized = tmp_path / "prerequisite-pixelize"
    output = tmp_path / f"owned-{stage}"

    if stage == "extract":

        def publish(force: bool) -> object:
            return publish_extraction(source, loaded, output, force=force)

    else:
        publish_extraction(source, loaded, extracted)
        if stage == "scale":

            def publish(force: bool) -> object:
                return publish_scale(extracted, loaded, output, force=force)

        else:
            publish_scale(extracted, loaded, scaled)
            if stage == "pixelize":

                def publish(force: bool) -> object:
                    return publish_pixelize(scaled, loaded, output, force=force)

            else:
                publish_pixelize(scaled, loaded, pixelized)

                def publish(force: bool) -> object:
                    return publish_align(pixelized, loaded, output, force=force)

    case = PublicationCase(stage=stage, output=output, publish=publish)
    if publish_target:
        case.publish(False)
    return case


def _first_frame(root: Path) -> Path:
    frames = _metadata(root)["frames"]
    assert isinstance(frames, list) and frames
    frame = frames[0]
    assert isinstance(frame, dict)
    relative = frame["relativePath"]
    assert isinstance(relative, str)
    return root / relative


def _mutate(root: Path, stage: PublicationStage, mutation: Mutation) -> None:
    metadata = _metadata(root)
    frames = metadata["frames"]
    assert isinstance(frames, list) and frames
    first = frames[0]
    assert isinstance(first, dict)
    frame_path = _first_frame(root)

    if mutation == "invalid-marker":
        marker = root / ".pixipix-output"
        marker.write_text('{"owner":"foreign","schemaVersion":1}', encoding="utf-8")
    elif mutation == "missing-stage":
        (root / "stage.json").unlink()
    elif mutation == "malformed-stage":
        (root / "stage.json").write_text("{", encoding="utf-8")
    elif mutation == "failed-status":
        metadata["status"] = "failed"
        _write_metadata(root, metadata)
    elif mutation == "stage-mismatch":
        metadata["stage"] = "align" if stage != "align" else "scale"
        _write_metadata(root, metadata)
    elif mutation == "missing-frame":
        frame_path.unlink()
    elif mutation == "extra-frame":
        shutil.copyfile(frame_path, root / "frames" / "extra.png")
    elif mutation == "unsafe-frame-path":
        first["relativePath"] = "frames/../escape.png"
        _write_metadata(root, metadata)
    elif mutation == "duplicate-frame":
        frames.append(dict(first))
        _write_metadata(root, metadata)
    elif mutation == "symlinked-frame":
        safe = root.parent / f"{stage}-safe.png"
        shutil.copyfile(frame_path, safe)
        frame_path.unlink()
        frame_path.symlink_to(safe)
    elif mutation == "wrong-mode":
        Image.new("RGB", (2, 2), (1, 2, 3)).save(frame_path, format="PNG")
    elif mutation == "png-dimension-mismatch":
        Image.new("RGBA", (1, 1), (1, 2, 3, 4)).save(frame_path, format="PNG")
    elif mutation == "metadata-dimension-mismatch":
        if stage == "extract":
            bounds = first["paddedBounds"]
            assert isinstance(bounds, dict)
            bounds["right"] = cast(int, bounds["right"]) + 1
        elif stage == "scale":
            dimensions = first["outputDimensions"]
            assert isinstance(dimensions, dict)
            dimensions["width"] = cast(int, dimensions["width"]) + 1
        elif stage == "pixelize":
            dimensions = first["logicalOutputDimensions"]
            assert isinstance(dimensions, dict)
            dimensions["width"] = cast(int, dimensions["width"]) + 1
        else:
            first["outputWidth"] = cast(int, first["outputWidth"]) + 1
        _write_metadata(root, metadata)
    else:
        frame_path.write_bytes(b"not a PNG")


@contextmanager
def _temporary_decoder_limit(limit: int) -> Iterator[None]:
    original = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = limit
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = original


def test_publication_adapter_roster_is_exact_and_unique() -> None:
    assert STAGES == ("extract", "scale", "pixelize", "align")
    assert len(STAGES) == len(set(STAGES)) == 4
    assert tuple(stage for stage, _publisher in EXPECTED_STAGE_PUBLISHERS) == STAGES
    assert len({publisher for _stage, publisher in EXPECTED_STAGE_PUBLISHERS}) == 4


@pytest.mark.parametrize(("stage", "expected_publisher"), EXPECTED_STAGE_PUBLISHERS)
def test_publication_case_maps_each_stage_to_its_real_publisher(
    tmp_path: Path,
    stage: PublicationStage,
    expected_publisher: Callable[..., object],
) -> None:
    case = _publication_case(tmp_path, stage, publish_target=False)
    publisher_names = {
        "publish_extraction",
        "publish_scale",
        "publish_pixelize",
        "publish_align",
    }
    referenced_publishers = {
        case.publish.__globals__[name]
        for name in case.publish.__code__.co_names
        if name in publisher_names
    }

    assert case.stage == stage
    assert referenced_publishers == {expected_publisher}
    case.publish(False)
    assert _metadata(case.output)["stage"] == stage


@pytest.mark.parametrize("stage", STAGES)
def test_valid_owned_target_and_stale_root_are_force_replaced(
    tmp_path: Path, stage: PublicationStage
) -> None:
    case = _publication_case(tmp_path, stage)
    stale = case.output / "permitted-stale-root.txt"
    stale.write_text("stale", encoding="utf-8")

    result = case.publish(True)

    assert result is not None
    assert not stale.exists()
    assert _metadata(case.output)["stage"] == stage
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize("stage", STAGES)
def test_existing_empty_directory_is_replaced_without_force(
    tmp_path: Path, stage: PublicationStage
) -> None:
    case = _publication_case(tmp_path, stage, publish_target=False)
    case.output.mkdir()

    case.publish(False)

    assert _metadata(case.output)["stage"] == stage


@pytest.mark.parametrize("stage", STAGES)
def test_unowned_nonempty_directory_is_preserved(tmp_path: Path, stage: PublicationStage) -> None:
    case = _publication_case(tmp_path, stage, publish_target=False)
    case.output.mkdir()
    keep = case.output / "keep.txt"
    keep.write_text("important", encoding="utf-8")

    with pytest.raises(ProcessingError) as without_force:
        case.publish(False)
    with pytest.raises(ProcessingError) as with_force:
        case.publish(True)

    assert without_force.value.code == "PX_OUTPUT_002"
    assert with_force.value.code == "PX_OUTPUT_003"
    assert keep.read_text(encoding="utf-8") == "important"


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_invalid_owned_target_is_rejected_without_mutation(
    tmp_path: Path, stage: PublicationStage, mutation: Mutation
) -> None:
    case = _publication_case(tmp_path, stage)
    _mutate(case.output, stage, mutation)
    before = _artifact_bytes(case.output)

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert captured.value.stage == "publish"
    assert _artifact_bytes(case.output) == before
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(("limit", "expected_exception"), ((3, "warning"), (1, "error")))
def test_decoder_safety_findings_reject_tiny_owned_png_and_restore_global(
    tmp_path: Path,
    stage: PublicationStage,
    limit: int,
    expected_exception: str,
) -> None:
    case = _publication_case(tmp_path, stage)
    frame_path = _first_frame(case.output)
    with Image.open(frame_path) as image:
        assert image.size == (2, 2)
    original = Image.MAX_IMAGE_PIXELS
    before = _artifact_bytes(case.output)

    with _temporary_decoder_limit(limit):
        expected_type = (
            Image.DecompressionBombWarning
            if expected_exception == "warning"
            else Image.DecompressionBombError
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with pytest.raises(expected_type), Image.open(frame_path):
                pass
        with pytest.raises(ProcessingError) as captured:
            case.publish(True)
        assert captured.value.code == "PX_OUTPUT_003", expected_exception

    assert original == Image.MAX_IMAGE_PIXELS
    assert _artifact_bytes(case.output) == before
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    EXTRACT_METADATA_MUTATIONS,
    ids=(
        "source-dict",
        "background-dict",
        "candidate-components-list",
        "accepted-components-list",
        "rejected-components-list",
        "ordered-components-list",
        "warnings-list",
        "source-hash-string",
        "source-hash-length",
        "source-hash-lowercase",
        "source-hash-hex",
        "effective-hash-string",
        "effective-hash-length",
        "effective-hash-lowercase",
        "effective-hash-hex",
        "version-string",
    ),
)
def test_each_extract_specific_metadata_invariant_is_required(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    case = _publication_case(tmp_path, "extract")
    metadata = _metadata(case.output)
    metadata[field] = invalid_value
    _write_metadata(case.output, metadata)
    before = _artifact_bytes(case.output)

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert captured.value.stage == "publish"
    assert _artifact_bytes(case.output) == before
    assert _temporary_siblings(case.output) == ()


def test_extract_specific_metadata_is_revalidated_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _publication_case(tmp_path, "extract")
    original_validate = pipeline_publication._validate_staged
    concurrent_bytes: dict[str, bytes] = {}

    def invalidate_extract_metadata(
        root: Path,
        stage: PublicationStage,
        metadata: object,
        frames: tuple[OutputFrameImage, ...],
    ) -> None:
        original_validate(root, stage, metadata, frames)
        existing = _metadata(case.output)
        existing["warnings"] = {}
        _write_metadata(case.output, existing)
        concurrent_bytes.update(_artifact_bytes(case.output))

    monkeypatch.setattr(
        pipeline_publication,
        "_validate_staged",
        invalidate_extract_metadata,
    )

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert captured.value.stage == "publish"
    assert _artifact_bytes(case.output) == concurrent_bytes
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize(("limit", "expected_exception"), ((3, "warning"), (1, "error")))
def test_staged_decoder_safety_rejects_tiny_png_and_restores_global(
    tmp_path: Path,
    limit: int,
    expected_exception: str,
) -> None:
    output = tmp_path / "staged-decoder-output"
    frame = OutputFrameImage(
        relative_path=PurePosixPath("frames/tiny.png"),
        pixels=np.zeros((2, 2, 4), dtype=np.uint8),
    )
    original = Image.MAX_IMAGE_PIXELS

    with _temporary_decoder_limit(limit):
        with pytest.raises(ProcessingError) as captured:
            pipeline_publication.publish_stage_output(
                output,
                "scale",
                {"probe": expected_exception},
                (frame,),
            )
        assert captured.value.code == "PX_OUTPUT_006"
        assert captured.value.stage == "publish"

    assert original == Image.MAX_IMAGE_PIXELS
    assert not output.exists()
    assert _temporary_siblings(output) == ()


def test_dimension_authority_is_exact_independent_and_consumed(tmp_path: Path) -> None:
    assert DimensionAuthority(width=3, height=2) == DIMENSION_AUTHORITY
    assert DIMENSION_AUTHORITY.width != DIMENSION_AUTHORITY.height
    assert DIMENSION_AUTHORITY.size == (3, 2)
    for authority_function in (
        _expected_dimensions,
        _frame_dimensions,
        _write_rectangular_owned_output,
    ):
        assert "_dimensions" not in authority_function.__code__.co_names

    for stage in DIMENSION_EXPECTED_LEDGER["valid-rectangular"]:
        case = DimensionCase(stage, "valid-rectangular")
        output = tmp_path / f"dimension-authority-{stage}"
        _write_rectangular_owned_output(output, case)
        with Image.open(output / "frames" / "rectangular.png") as image:
            assert image.size == DIMENSION_AUTHORITY.size == (3, 2)
        frames = _metadata(output)["frames"]
        assert isinstance(frames, list) and len(frames) == 1
        frame = frames[0]
        assert isinstance(frame, dict)
        assert _test_owned_declared_dimensions(stage, frame) == (3, 2)


def test_dimension_category_ledger_is_exact() -> None:
    assert tuple(DIMENSION_EXPECTED_LEDGER) == (
        "valid-rectangular",
        "width-only",
        "height-only",
        "both-axes",
    )
    assert len(DIMENSION_CASES) == 16
    assert len(set(DIMENSION_CASES)) == 16

    for mutation, expected_stages in DIMENSION_EXPECTED_LEDGER.items():
        category_cases = tuple(case for case in DIMENSION_CASES if case.mutation == mutation)
        actual_stages = tuple(case.stage for case in category_cases)
        assert len(category_cases) == 4
        assert actual_stages == expected_stages
        assert len(set(actual_stages)) == 4
        assert "extract" in actual_stages
        assert all(case.mutation == mutation for case in category_cases)

        width_delta, height_delta = DIMENSION_EXPECTED_DELTAS[mutation]
        assert _expected_dimensions(mutation) == (
            DIMENSION_AUTHORITY.width + width_delta,
            DIMENSION_AUTHORITY.height + height_delta,
        )


@pytest.mark.parametrize(
    "case",
    DIMENSION_CASES,
    ids=lambda case: f"{case.mutation}-{case.stage}",
)
def test_owned_rectangular_dimensions_enforce_each_axis_independently(
    tmp_path: Path,
    case: DimensionCase,
) -> None:
    output = tmp_path / f"rectangular-{case.stage}-{case.mutation}"
    _write_rectangular_owned_output(output, case)
    with Image.open(output / "frames" / "rectangular.png") as image:
        assert image.size == DIMENSION_AUTHORITY.size == (3, 2)
    frames = _metadata(output)["frames"]
    assert isinstance(frames, list) and len(frames) == 1
    frame = frames[0]
    assert isinstance(frame, dict)
    assert _test_owned_declared_dimensions(case.stage, frame) == _expected_dimensions(case.mutation)

    if case.mutation == "valid-rectangular":
        pipeline_publication.validate_stage_output_target(output, case.stage, force=True)
    else:
        with pytest.raises(ProcessingError) as captured:
            pipeline_publication.validate_stage_output_target(output, case.stage, force=True)
        assert captured.value.code == "PX_OUTPUT_003"
        assert captured.value.stage == "publish"


@pytest.mark.parametrize("stage", STAGES)
def test_destination_appearing_before_final_revalidation_is_preserved(
    tmp_path: Path,
    stage: PublicationStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _publication_case(tmp_path, stage, publish_target=False)
    original_validate = pipeline_publication._validate_staged

    def create_foreign_target(
        root: Path,
        validated_stage: PublicationStage,
        metadata: object,
        frames: tuple[OutputFrameImage, ...],
    ) -> None:
        original_validate(root, validated_stage, metadata, frames)
        case.output.mkdir()
        (case.output / "keep.txt").write_text("important", encoding="utf-8")

    monkeypatch.setattr(pipeline_publication, "_validate_staged", create_foreign_target)

    with pytest.raises(ProcessingError) as captured:
        case.publish(False)

    assert captured.value.code == "PX_OUTPUT_002"
    assert (case.output / "keep.txt").read_text(encoding="utf-8") == "important"
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize("stage", STAGES)
def test_destination_changed_before_final_revalidation_is_preserved(
    tmp_path: Path,
    stage: PublicationStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _publication_case(tmp_path, stage)
    original_validate = pipeline_publication._validate_staged

    def invalidate_target(
        root: Path,
        validated_stage: PublicationStage,
        metadata: object,
        frames: tuple[OutputFrameImage, ...],
    ) -> None:
        original_validate(root, validated_stage, metadata, frames)
        (case.output / ".pixipix-output").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline_publication, "_validate_staged", invalidate_target)

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    assert captured.value.code == "PX_OUTPUT_003"
    assert (case.output / ".pixipix-output").read_text(encoding="utf-8") == "{}"
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize("stage", STAGES)
def test_staged_validation_failure_publishes_nothing(
    tmp_path: Path,
    stage: PublicationStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _publication_case(tmp_path, stage, publish_target=False)

    def reject_staging(*_args: object, **_kwargs: object) -> None:
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged output rejected")

    monkeypatch.setattr(pipeline_publication, "_validate_staged", reject_staging)

    with pytest.raises(ProcessingError) as captured:
        case.publish(False)

    assert captured.value.code == "PX_OUTPUT_006"
    assert not case.output.exists()
    assert _temporary_siblings(case.output) == ()


@pytest.mark.parametrize("stage", STAGES)
def test_replacement_failure_restores_previous_output(
    tmp_path: Path,
    stage: PublicationStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _publication_case(tmp_path, stage)
    before = _artifact_bytes(case.output)
    real_replace = Path.replace

    def fail_new_publication(self: Path, target: Path) -> Path:
        if self.name.startswith(f".{case.output.name}.pixipix-build-") and target == case.output:
            raise OSError("simulated publication failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_publication)

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    assert captured.value.code == "PX_OUTPUT_005"
    assert _artifact_bytes(case.output) == before
    assert _temporary_siblings(case.output) == ()


def test_failed_restore_retains_recoverable_backup_and_reports_atomic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _publication_case(tmp_path, "extract")
    before = _artifact_bytes(case.output)
    real_replace = Path.replace

    def fail_publication_and_restore(self: Path, target: Path) -> Path:
        if self.name.startswith(f".{case.output.name}.pixipix-build-"):
            raise OSError("simulated publication failure")
        if self.name == "previous" and target == case.output:
            raise OSError("simulated restore failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_restore)

    with pytest.raises(ProcessingError) as captured:
        case.publish(True)

    backups = tuple(case.output.parent.glob(f".{case.output.name}.pixipix-backup-*"))
    assert captured.value.code == "PX_OUTPUT_005"
    assert not case.output.exists()
    assert len(backups) == 1
    assert _artifact_bytes(backups[0] / "previous") == before
    assert tuple(case.output.parent.glob(f".{case.output.name}.pixipix-build-*")) == ()


def test_cleanup_rejects_symlinked_or_unauthorized_roots(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    keep = outside / "keep.txt"
    keep.write_text("important", encoding="utf-8")
    linked = parent / ".output.pixipix-build-linked"
    linked.symlink_to(outside, target_is_directory=True)
    wrong_prefix = parent / "unrelated"
    wrong_prefix.mkdir()

    assert not pipeline_publication._remove_tree(linked, parent, ".output.pixipix-build-")
    assert not pipeline_publication._remove_tree(wrong_prefix, parent, ".output.pixipix-build-")
    assert keep.read_text(encoding="utf-8") == "important"
    assert linked.is_symlink()
    assert wrong_prefix.is_dir()
