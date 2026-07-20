"""Validated stage handoff and ownership-aware atomic publication."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
import tempfile
import warnings as python_warnings
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixipix.errors import ProcessingError, UnsupportedInputError
from pixipix.imageio import write_png
from pixipix.models import (
    Dimensions,
    OutputMarker,
    PriorStageIdentity,
    ProcessingWarning,
    UInt8Image,
)
from pixipix.serialization import to_json_data, write_json

type StageName = Literal["extract", "scale", "pixelize", "align"]


@dataclass(slots=True)
class InputStageFrame:
    name: str
    relative_path: PurePosixPath
    source_order: int
    dimensions: Dimensions
    pixels: UInt8Image


@dataclass(slots=True)
class LoadedStageInput:
    identity: PriorStageIdentity
    frames: tuple[InputStageFrame, ...]
    metadata: dict[str, object]
    warnings: tuple[ProcessingWarning, ...]


@dataclass(slots=True)
class OutputFrameImage:
    relative_path: PurePosixPath
    pixels: UInt8Image


def _is_schema_version_one(value: object) -> bool:
    return type(value) is int and value == 1


def _is_output_marker(value: dict[str, object], stage: StageName) -> bool:
    return (
        set(value) == {"owner", "schemaVersion", "stage"}
        and value.get("owner") == "pixipix"
        and _is_schema_version_one(value.get("schemaVersion"))
        and value.get("stage") == stage
    )


def _read_json_object(path: Path, code: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise UnsupportedInputError(
            code, "required stage file is missing or unsafe", path=path.name
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UnsupportedInputError(
            code, "required stage file is not valid JSON", path=path.name
        ) from error
    if not isinstance(value, dict):
        raise UnsupportedInputError(
            code, "required stage file must contain a JSON object", path=path.name
        )
    return cast(dict[str, object], value)


def _safe_frame_relative(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise UnsupportedInputError("PX_STAGE_006", "frame path must be a safe relative POSIX path")
    relative = PurePosixPath(raw)
    if (
        relative.as_posix() != raw
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "frames"
        or relative.suffix.lower() != ".png"
    ):
        raise UnsupportedInputError("PX_STAGE_006", f'unsafe frame path "{raw}"')
    return relative


def _positive_dimension(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedInputError("PX_STAGE_007", f"invalid {label} in stage metadata")
    return value


def _dimensions(frame: dict[str, object], stage: str) -> Dimensions:
    if stage == "extract":
        bounds = frame.get("paddedBounds")
        if not isinstance(bounds, dict):
            raise UnsupportedInputError("PX_STAGE_007", "extract frame lacks padded bounds")
        left = bounds.get("left")
        right = bounds.get("right")
        top = bounds.get("top")
        bottom = bounds.get("bottom")
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (left, right, top, bottom)
        ):
            raise UnsupportedInputError("PX_STAGE_007", "extract frame bounds are invalid")
        width = cast(int, right) - cast(int, left)
        height = cast(int, bottom) - cast(int, top)
        return Dimensions(
            _positive_dimension(width, "frame width"),
            _positive_dimension(height, "frame height"),
        )
    if stage == "scale":
        raw = frame.get("outputDimensions")
        label = "scale"
    elif stage == "pixelize":
        raw = frame.get("logicalOutputDimensions")
        label = "pixelize"
    else:
        raw = {
            "width": frame.get("outputWidth"),
            "height": frame.get("outputHeight"),
        }
        label = "align"
    if not isinstance(raw, dict):
        raise UnsupportedInputError("PX_STAGE_007", f"{label} frame lacks output dimensions")
    return Dimensions(
        _positive_dimension(raw.get("width"), "frame width"),
        _positive_dimension(raw.get("height"), "frame height"),
    )


def _sha256_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UnsupportedInputError("PX_STAGE_008", f"stage metadata has invalid {label}")
    return value


def _positive_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedInputError("PX_STAGE_009", f"scale metadata has invalid {label}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise UnsupportedInputError("PX_STAGE_009", f"scale metadata has invalid {label}")
    return number


def _processing_warnings(metadata: dict[str, object]) -> tuple[ProcessingWarning, ...]:
    raw_warnings = metadata.get("warnings")
    if not isinstance(raw_warnings, list):
        raise UnsupportedInputError("PX_STAGE_008", "stage metadata has invalid warnings")
    warnings: list[ProcessingWarning] = []
    for item in raw_warnings:
        if not isinstance(item, dict) or set(item) != {"code", "stage", "message"}:
            raise UnsupportedInputError("PX_STAGE_008", "stage metadata has invalid warning entry")
        code = item.get("code")
        stage = item.get("stage")
        message = item.get("message")
        if (
            not isinstance(code, str)
            or not code
            or not isinstance(stage, str)
            or not stage
            or not isinstance(message, str)
            or not message
        ):
            raise UnsupportedInputError("PX_STAGE_008", "stage metadata has invalid warning entry")
        warnings.append(ProcessingWarning(code, stage, message))
    return tuple(warnings)


def _validate_scale_metadata(
    metadata: dict[str, object], raw_frames: list[object], effective_hash: str
) -> None:
    prior = metadata.get("priorStage")
    if (
        not isinstance(prior, dict)
        or prior.get("stage") != "extract"
        or not _is_schema_version_one(prior.get("schemaVersion"))
        or not isinstance(prior.get("pixipixVersion"), str)
        or not prior.get("pixipixVersion")
    ):
        raise UnsupportedInputError(
            "PX_STAGE_009", "scale metadata has invalid prior-stage identity"
        )
    prior_hash = _sha256_string(prior.get("effectiveConfigSha256"), "prior effective config hash")
    if prior_hash != effective_hash:
        raise UnsupportedInputError(
            "PX_STAGE_009", "scale metadata has inconsistent configuration identity"
        )
    mode = metadata.get("scaleMode")
    if mode not in {
        "explicit-factor",
        "reference-frame-width",
        "reference-frame-height",
    }:
        raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid scale mode")
    global_factor = _positive_finite_number(metadata.get("globalFactor"), "global factor")
    cell_size = metadata.get("sourceCellSize")
    if cell_size is not None and (
        isinstance(cell_size, bool) or not isinstance(cell_size, int) or cell_size <= 0
    ):
        raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid source cell size")
    raw_overrides = metadata.get("configuredFrameOverrides")
    if not isinstance(raw_overrides, list):
        raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid override list")
    frame_values: dict[str, tuple[int, int, int, int, float, float]] = {}
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid frame entry")
        name = raw_frame.get("name")
        if not isinstance(name, str) or not name or name in frame_values:
            raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid frame name")
        input_dimensions = raw_frame.get("inputDimensions")
        if not isinstance(input_dimensions, dict):
            raise UnsupportedInputError("PX_STAGE_009", "scale frame lacks input dimensions")
        input_width = _positive_dimension(input_dimensions.get("width"), "input frame width")
        input_height = _positive_dimension(input_dimensions.get("height"), "input frame height")
        output_dimensions = raw_frame.get("outputDimensions")
        if not isinstance(output_dimensions, dict):
            raise UnsupportedInputError("PX_STAGE_009", "scale frame lacks output dimensions")
        output_width = _positive_dimension(output_dimensions.get("width"), "output frame width")
        output_height = _positive_dimension(output_dimensions.get("height"), "output frame height")
        multiplier = _positive_finite_number(raw_frame.get("scaleMultiplier"), "frame multiplier")
        effective = _positive_finite_number(
            raw_frame.get("effectiveFactor"), "effective frame factor"
        )
        expected_effective = global_factor * multiplier
        if not math.isfinite(expected_effective) or effective != expected_effective:
            raise UnsupportedInputError(
                "PX_STAGE_009", "scale frame factor is inconsistent with the global factor"
            )
        frame_values[name] = (
            input_width,
            input_height,
            output_width,
            output_height,
            multiplier,
            effective,
        )

    overrides: dict[str, float] = {}
    for item in raw_overrides:
        if not isinstance(item, dict) or set(item) != {"frameName", "scaleMultiplier"}:
            raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid override entry")
        frame_name = item.get("frameName")
        if (
            not isinstance(frame_name, str)
            or frame_name not in frame_values
            or frame_name in overrides
        ):
            raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid override frame")
        overrides[frame_name] = _positive_finite_number(
            item.get("scaleMultiplier"), "override multiplier"
        )
    for name, values in frame_values.items():
        if values[4] != overrides.get(name, 1.0):
            raise UnsupportedInputError(
                "PX_STAGE_009", "scale metadata frame multiplier disagrees with overrides"
            )

    reference_fields = (
        metadata.get("referenceFrame"),
        metadata.get("sourceReferenceMeasurement"),
        metadata.get("exactTargetSourceMeasurement"),
        metadata.get("logicalTargetSize"),
    )
    if mode == "explicit-factor":
        if any(value is not None for value in reference_fields):
            raise UnsupportedInputError(
                "PX_STAGE_009", "explicit scale metadata carries reference-only fields"
            )
        return

    reference_frame, source_measurement, exact_target, logical_target = reference_fields
    if not isinstance(reference_frame, str) or reference_frame not in frame_values:
        raise UnsupportedInputError("PX_STAGE_009", "scale metadata has invalid reference frame")
    if cell_size is None:
        raise UnsupportedInputError(
            "PX_STAGE_009", "reference scale metadata lacks source cell size"
        )
    source_measurement = _positive_dimension(source_measurement, "source reference measurement")
    exact_target = _positive_dimension(exact_target, "exact target source measurement")
    logical_target = _positive_dimension(logical_target, "logical target size")
    if exact_target != logical_target * cell_size:
        raise UnsupportedInputError(
            "PX_STAGE_009", "reference scale metadata has inconsistent exact target"
        )
    reference_values = frame_values[reference_frame]
    axis = 0 if mode == "reference-frame-width" else 1
    if (
        source_measurement != reference_values[axis]
        or exact_target != reference_values[axis + 2]
        or reference_values[4] != 1.0
        or reference_frame in overrides
    ):
        raise UnsupportedInputError(
            "PX_STAGE_009", "reference scale metadata violates the exact reference target"
        )
    try:
        expected_global = exact_target / source_measurement
    except OverflowError as error:
        raise UnsupportedInputError(
            "PX_STAGE_009", "reference scale metadata has an unrepresentable global factor"
        ) from error
    if global_factor != expected_global:
        raise UnsupportedInputError(
            "PX_STAGE_009", "reference scale metadata has inconsistent global factor"
        )


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnsupportedInputError("PX_STAGE_016", f"pixelize metadata has invalid {label}")
    return value


def _validate_pixelize_metadata(
    metadata: dict[str, object], raw_frames: list[object], effective_hash: str
) -> None:
    prior = metadata.get("priorStage")
    if (
        not isinstance(prior, dict)
        or prior.get("stage") != "scale"
        or not _is_schema_version_one(prior.get("schemaVersion"))
        or not isinstance(prior.get("pixipixVersion"), str)
        or not prior.get("pixipixVersion")
        or _sha256_string(prior.get("effectiveConfigSha256"), "prior effective config hash")
        != effective_hash
    ):
        raise UnsupportedInputError(
            "PX_STAGE_016", "pixelize metadata has invalid prior-stage identity"
        )
    cell_size = _positive_dimension(metadata.get("sourceCellSize"), "source cell size")
    if metadata.get("cellGridOrigin") != "bottom-left":
        raise UnsupportedInputError("PX_STAGE_016", "pixelize metadata has invalid grid origin")
    if metadata.get("representative") not in {
        "majority",
        "center",
        "alpha-weighted-majority",
    }:
        raise UnsupportedInputError("PX_STAGE_016", "pixelize metadata has invalid representative")
    if metadata.get("alphaPolicy") not in {"binary", "preserve"}:
        raise UnsupportedInputError("PX_STAGE_016", "pixelize metadata has invalid alpha policy")
    alpha_threshold = metadata.get("alphaThreshold")
    if (
        isinstance(alpha_threshold, bool)
        or not isinstance(alpha_threshold, int)
        or not 0 <= alpha_threshold <= 255
    ):
        raise UnsupportedInputError("PX_STAGE_016", "pixelize metadata has invalid alpha threshold")
    policy = metadata.get("remainderPolicy")
    if policy not in {"pad-transparent", "error", "crop-with-warning"}:
        raise UnsupportedInputError(
            "PX_STAGE_016", "pixelize metadata has invalid remainder policy"
        )
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise UnsupportedInputError("PX_STAGE_016", "pixelize metadata has invalid frame entry")
        input_dimensions = raw_frame.get("inputDimensions")
        prepared_dimensions = raw_frame.get("preparedDimensions")
        output_dimensions = raw_frame.get("logicalOutputDimensions")
        if not all(
            isinstance(item, dict)
            for item in (input_dimensions, prepared_dimensions, output_dimensions)
        ):
            raise UnsupportedInputError("PX_STAGE_016", "pixelize frame has incomplete dimensions")
        input_dimensions = cast(dict[str, object], input_dimensions)
        prepared_dimensions = cast(dict[str, object], prepared_dimensions)
        output_dimensions = cast(dict[str, object], output_dimensions)
        input_width = _positive_dimension(input_dimensions.get("width"), "input frame width")
        input_height = _positive_dimension(input_dimensions.get("height"), "input frame height")
        prepared_width = _positive_dimension(
            prepared_dimensions.get("width"), "prepared frame width"
        )
        prepared_height = _positive_dimension(
            prepared_dimensions.get("height"), "prepared frame height"
        )
        logical_width = _positive_dimension(output_dimensions.get("width"), "logical frame width")
        logical_height = _positive_dimension(
            output_dimensions.get("height"), "logical frame height"
        )
        top_padding = _nonnegative_integer(raw_frame.get("topPadding"), "top padding")
        right_padding = _nonnegative_integer(raw_frame.get("rightPadding"), "right padding")
        top_crop = _nonnegative_integer(raw_frame.get("topCrop"), "top crop")
        right_crop = _nonnegative_integer(raw_frame.get("rightCrop"), "right crop")
        if (
            prepared_width != logical_width * cell_size
            or prepared_height != logical_height * cell_size
            or prepared_width != input_width + right_padding - right_crop
            or prepared_height != input_height + top_padding - top_crop
        ):
            raise UnsupportedInputError(
                "PX_STAGE_016", "pixelize frame dimensions are internally inconsistent"
            )
        expected_right_remainder = input_width % cell_size
        expected_top_remainder = input_height % cell_size
        if policy == "pad-transparent":
            expected_right_padding = (-input_width) % cell_size
            expected_top_padding = (-input_height) % cell_size
            coherent = (
                right_padding == expected_right_padding
                and top_padding == expected_top_padding
                and right_crop == 0
                and top_crop == 0
            )
        elif policy == "crop-with-warning":
            coherent = (
                right_padding == 0
                and top_padding == 0
                and right_crop == expected_right_remainder
                and top_crop == expected_top_remainder
            )
        else:
            coherent = (
                right_padding == top_padding == right_crop == top_crop == 0
                and expected_right_remainder == expected_top_remainder == 0
            )
        if not coherent:
            raise UnsupportedInputError(
                "PX_STAGE_016", "pixelize frame remainder metadata is inconsistent"
            )


def load_stage_input(
    root: Path, expected_stage: Literal["extract", "scale", "pixelize"]
) -> LoadedStageInput:
    """Load frames strictly in metadata order after validating the full handoff."""

    for candidate in (root, *root.parents):
        if candidate.is_symlink() and not _trusted_tmp_alias(candidate):
            raise UnsupportedInputError(
                "PX_STAGE_001", "stage input path and parents must not be symlinks", path=root.name
            )
    if root.is_symlink() or not root.is_dir():
        raise UnsupportedInputError(
            "PX_STAGE_001", "stage input must be a real directory", path=root.name
        )
    marker = _read_json_object(root / ".pixipix-output", "PX_STAGE_002")
    if not _is_output_marker(marker, expected_stage):
        actual = marker.get("stage")
        raise UnsupportedInputError(
            "PX_STAGE_003",
            f'expected prior stage "{expected_stage}" but found "{actual}"',
            path=root.name,
            remediation=f"run the {expected_stage} stage immediately before this command",
        )
    metadata = _read_json_object(root / "stage.json", "PX_STAGE_004")
    schema = metadata.get("schemaVersion")
    if not _is_schema_version_one(schema):
        raise UnsupportedInputError(
            "PX_STAGE_005", f"unsupported stage metadata schema {schema!r}", path=root.name
        )
    actual_stage = metadata.get("stage")
    if actual_stage != expected_stage or metadata.get("status") != "successful":
        raise UnsupportedInputError(
            "PX_STAGE_003",
            f'expected successful prior stage "{expected_stage}" but found "{actual_stage}"',
            path=root.name,
        )
    version = metadata.get("pixipixVersion")
    if not isinstance(version, str) or not version:
        raise UnsupportedInputError("PX_STAGE_008", "stage metadata lacks PixiPix version")
    effective_hash = _sha256_string(metadata.get("effectiveConfigSha256"), "effective config hash")
    _sha256_string(metadata.get("sourceConfigSha256"), "source config hash")
    if expected_stage == "scale":
        required = {
            "priorStage",
            "scaleMode",
            "globalFactor",
            "referenceFrame",
            "sourceReferenceMeasurement",
            "exactTargetSourceMeasurement",
            "logicalTargetSize",
            "sourceCellSize",
            "configuredFrameOverrides",
            "warnings",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise UnsupportedInputError(
                "PX_STAGE_009", f'scale metadata lacks required field "{missing[0]}"'
            )
    elif expected_stage == "pixelize":
        required = {
            "priorStage",
            "sourceCellSize",
            "cellGridOrigin",
            "representative",
            "alphaPolicy",
            "alphaThreshold",
            "remainderPolicy",
            "warnings",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise UnsupportedInputError(
                "PX_STAGE_016", f'pixelize metadata lacks required field "{missing[0]}"'
            )
    raw_frames = metadata.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise UnsupportedInputError(
            "PX_STAGE_010", "stage metadata requires a non-empty frame order"
        )
    warnings = _processing_warnings(metadata)
    if expected_stage == "scale":
        _validate_scale_metadata(metadata, raw_frames, effective_hash)
    elif expected_stage == "pixelize":
        _validate_pixelize_metadata(metadata, raw_frames, effective_hash)
    frames_root = root / "frames"
    if frames_root.is_symlink() or not frames_root.is_dir():
        raise UnsupportedInputError("PX_STAGE_011", "stage frame directory is missing or unsafe")
    frames: list[InputStageFrame] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    expected_paths: set[Path] = set()
    for source_order, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise UnsupportedInputError("PX_STAGE_010", "stage frame entry must be an object")
        frame = cast(dict[str, object], item)
        name = frame.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise UnsupportedInputError(
                "PX_STAGE_010", "stage frame names must be non-empty and unique"
            )
        raw_source_order = frame.get("sourceOrder")
        if type(raw_source_order) is not int or raw_source_order != source_order:
            raise UnsupportedInputError(
                "PX_STAGE_010", "stage frame order is missing or non-contiguous"
            )
        relative = _safe_frame_relative(frame.get("relativePath"))
        folded = relative.as_posix().casefold()
        if folded in seen_paths:
            raise UnsupportedInputError("PX_STAGE_010", "stage frame paths must be unique")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise UnsupportedInputError(
                "PX_STAGE_011", "declared frame is missing or unsafe", path=relative.as_posix()
            )
        dimensions = _dimensions(frame, expected_stage)
        try:
            with python_warnings.catch_warnings():
                python_warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    if (
                        image.format != "PNG"
                        or image.mode != "RGBA"
                        or image.size != (dimensions.width, dimensions.height)
                    ):
                        raise UnsupportedInputError(
                            "PX_STAGE_012",
                            "frame PNG mode or dimensions do not match metadata",
                            path=relative.as_posix(),
                        )
                    image.load()
                    pixels = np.array(image, dtype=np.uint8, copy=True)
        except UnsupportedInputError:
            raise
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise UnsupportedInputError(
                "PX_STAGE_012",
                "frame PNG dimensions exceed decoder safety limits",
                path=relative.as_posix(),
            ) from error
        except (UnidentifiedImageError, OSError) as error:
            raise UnsupportedInputError(
                "PX_STAGE_011", "unable to decode declared frame", path=relative.as_posix()
            ) from error
        declared_hash = frame.get("sha256")
        if declared_hash is not None:
            expected_hash = _sha256_string(declared_hash, "frame hash")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise UnsupportedInputError(
                    "PX_STAGE_013",
                    "declared frame hash does not match bytes",
                    path=relative.as_posix(),
                )
        frames.append(InputStageFrame(name, relative, source_order, dimensions, pixels))
        seen_names.add(name)
        seen_paths.add(folded)
        expected_paths.add(path)
    entries = list(frames_root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or set(entries) != expected_paths
    ):
        raise UnsupportedInputError(
            "PX_STAGE_014", "frame directory contents do not match stage metadata"
        )
    root_entries = {entry.name for entry in root.iterdir()}
    if root_entries != {".pixipix-output", "frames", "stage.json"}:
        raise UnsupportedInputError("PX_STAGE_014", "stage directory contains undeclared artifacts")
    identity = PriorStageIdentity(expected_stage, 1, version, effective_hash)
    return LoadedStageInput(identity, tuple(frames), metadata, warnings)


def _trusted_tmp_alias(path: Path) -> bool:
    if path != Path("/tmp") or not path.is_symlink():
        return False
    try:
        link = path.lstat()
        resolved = path.resolve(strict=True)
        target = resolved.stat()
    except OSError:
        return False
    return bool(
        link.st_uid == 0
        and target.st_uid == 0
        and stat.S_ISDIR(target.st_mode)
        and target.st_mode & stat.S_ISVTX
    )


def _validate_output_location(output: Path) -> None:
    resolved = output.resolve(strict=False)
    if resolved in {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}:
        raise ProcessingError(
            "PX_OUTPUT_007",
            "publish",
            "refusing to use a dangerous output location",
            path=output.name or ".",
        )
    for candidate in (output, *output.parents):
        if candidate.is_symlink() and not _trusted_tmp_alias(candidate):
            raise ProcessingError(
                "PX_OUTPUT_004",
                "publish",
                "output path and untrusted existing parents must not be symlinks",
                path=output.name,
            )


def _valid_owned_output(path: Path, stage: StageName) -> bool:
    try:
        marker = _read_json_object(path / ".pixipix-output", "PX_STAGE")
        metadata = _read_json_object(path / "stage.json", "PX_STAGE")
        if not _is_output_marker(marker, stage):
            return False
        if (
            not _is_schema_version_one(metadata.get("schemaVersion"))
            or metadata.get("stage") != stage
            or metadata.get("status") != "successful"
        ):
            return False
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            return False
        seen: set[str] = set()
        for order, item in enumerate(frames):
            if (
                not isinstance(item, dict)
                or type(item.get("sourceOrder")) is not int
                or item.get("sourceOrder") != order
            ):
                return False
            relative = _safe_frame_relative(item.get("relativePath"))
            if relative.as_posix().casefold() in seen:
                return False
            frame_path = path.joinpath(*relative.parts)
            if frame_path.is_symlink() or not frame_path.is_file():
                return False
            if stage == "extract":
                dimensions = _dimensions(item, "extract")
            else:
                dimensions = _dimensions(item, stage)
            try:
                with python_warnings.catch_warnings():
                    python_warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(frame_path) as image:
                        if (
                            image.format != "PNG"
                            or image.mode != "RGBA"
                            or image.size != (dimensions.width, dimensions.height)
                        ):
                            return False
                        image.load()
            except (
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
                UnidentifiedImageError,
                OSError,
            ):
                return False
            seen.add(relative.as_posix().casefold())
        return True
    except (UnsupportedInputError, OSError):
        return False


def _prepare_target(output: Path, force: bool, stage: StageName) -> None:
    _validate_output_location(output)
    if not output.exists():
        return
    if not output.is_dir():
        raise ProcessingError(
            "PX_OUTPUT_001",
            "publish",
            "output path exists and is not a directory",
            path=output.name,
        )
    if next(output.iterdir(), None) is None:
        return
    if not force:
        raise ProcessingError(
            "PX_OUTPUT_002",
            "publish",
            "non-empty output directory is rejected without --force",
            path=output.name,
        )
    if not _valid_owned_output(output, stage):
        raise ProcessingError(
            "PX_OUTPUT_003",
            "publish",
            "--force may replace only a valid PixiPix-owned output for this stage",
            path=output.name,
        )


def _remove_tree(path: Path, parent: Path, prefix: str) -> bool:
    if path.parent != parent or not path.name.startswith(prefix) or path.is_symlink():
        return False
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return True


def _validate_staged(
    root: Path, stage: StageName, metadata: object, frames: tuple[OutputFrameImage, ...]
) -> None:
    try:
        marker = _read_json_object(root / ".pixipix-output", "PX_STAGE")
    except UnsupportedInputError as error:
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged ownership marker is invalid"
        ) from error
    if not _is_output_marker(marker, stage):
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged ownership marker is invalid")
    stage_path = root / "stage.json"
    try:
        rendered = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged metadata is invalid") from error
    if rendered != to_json_data(metadata):
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged metadata does not match stage results"
        )
    expected: set[Path] = set()
    for frame in frames:
        try:
            relative = _safe_frame_relative(frame.relative_path.as_posix())
        except UnsupportedInputError as error:
            raise ProcessingError(
                "PX_OUTPUT_006", "publish", "staged frame path is invalid"
            ) from error
        path = root.joinpath(*relative.parts)
        expected.add(path)
        try:
            with python_warnings.catch_warnings():
                python_warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    size = (frame.pixels.shape[1], frame.pixels.shape[0])
                    if image.format != "PNG" or image.mode != "RGBA" or image.size != size:
                        raise ProcessingError(
                            "PX_OUTPUT_006",
                            "publish",
                            "staged frame is invalid",
                            path=relative.as_posix(),
                        )
                    image.load()
        except (
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
        ) as error:
            raise ProcessingError(
                "PX_OUTPUT_006", "publish", "staged frame is invalid", path=relative.as_posix()
            ) from error
    actual = set((root / "frames").iterdir())
    if actual != expected or any(path.is_symlink() or not path.is_file() for path in actual):
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged frame files do not match metadata"
        )


def publish_stage_output(
    output: Path,
    stage: Literal["scale", "pixelize", "align"],
    metadata: object,
    frames: tuple[OutputFrameImage, ...],
    *,
    force: bool = False,
) -> None:
    """Publish a complete typed stage output through a temporary sibling."""

    parent = output.parent
    build_prefix = f".{output.name}.pixipix-build-"
    backup_prefix = f".{output.name}.pixipix-backup-"
    build_root: Path | None = None
    backup_root: Path | None = None
    previous: Path | None = None
    try:
        _prepare_target(output, force, stage)
        parent.mkdir(parents=True, exist_ok=True)
        _prepare_target(output, force, stage)
        build_root = Path(tempfile.mkdtemp(prefix=build_prefix, dir=parent))
        frames_root = build_root / "frames"
        frames_root.mkdir()
        write_json(build_root / ".pixipix-output", OutputMarker(1, "pixipix", stage))
        for frame in frames:
            relative = _safe_frame_relative(frame.relative_path.as_posix())
            write_png(build_root.joinpath(*relative.parts), frame.pixels)
        write_json(build_root / "stage.json", metadata)
        _validate_staged(build_root, stage, metadata, frames)
        _prepare_target(output, force, stage)
        if output.exists():
            backup_root = Path(tempfile.mkdtemp(prefix=backup_prefix, dir=parent))
            previous = backup_root / "previous"
            output.replace(previous)
        try:
            build_root.replace(output)
        except OSError as error:
            if previous is not None and previous.exists() and not output.exists():
                with suppress(OSError):
                    previous.replace(output)
            raise ProcessingError(
                "PX_OUTPUT_005",
                "publish",
                "atomic output publication failed",
                path=output.name,
                remediation="verify destination permissions and retry",
            ) from error
        if backup_root is not None and _remove_tree(backup_root, parent, backup_prefix):
            backup_root = None
    except ProcessingError:
        raise
    except OSError as error:
        raise ProcessingError(
            "PX_OUTPUT_005", "publish", f"unable to write {stage} output", path=output.name
        ) from error
    finally:
        if build_root is not None and build_root.exists():
            _remove_tree(build_root, parent, build_prefix)
        if backup_root is not None and backup_root.exists():
            if previous is not None and previous.exists() and not output.exists():
                with suppress(OSError):
                    previous.replace(output)
            if backup_root.exists() and output.exists():
                _remove_tree(backup_root, parent, backup_prefix)
