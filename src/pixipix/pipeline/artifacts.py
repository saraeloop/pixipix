"""Narrow persisted-artifact contract primitives shared by pipeline lifecycles."""

from __future__ import annotations

import json
import stat
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pixipix.errors import UnsupportedInputError
from pixipix.models import Dimensions

type StageName = Literal["extract", "scale", "pixelize", "align"]


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
