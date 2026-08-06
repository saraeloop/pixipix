"""Narrow persisted-artifact contract primitives shared by pipeline lifecycles."""

from __future__ import annotations

import json
import os
import stat
import sys
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


_DARWIN_SYSTEM_ALIASES = {
    Path("/tmp"): (Path("private/tmp"), Path("/private/tmp")),
    Path("/var"): (Path("private/var"), Path("/private/var")),
}


def _runtime_os_name() -> str:
    return os.name


def _runtime_platform() -> str:
    return sys.platform


def _root_owned_directory(path: Path, *, sticky_allowed: bool) -> bool:
    if path.is_symlink():
        return False
    try:
        target = path.stat()
    except OSError:
        return False
    if target.st_uid != 0 or not stat.S_ISDIR(target.st_mode):
        return False
    writable = target.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return bool(not writable or (sticky_allowed and target.st_mode & stat.S_ISVTX))


def _trusted_darwin_system_alias(path: Path) -> bool:
    expected = _DARWIN_SYSTEM_ALIASES.get(path)
    if _runtime_platform() != "darwin" or expected is None or not path.is_symlink():
        return False
    relative_target, absolute_target = expected
    try:
        link = path.lstat()
        raw_target = Path(os.readlink(path))
        namespace = path.parent.stat()
    except OSError:
        return False
    if (
        link.st_uid != 0
        or namespace.st_uid != 0
        or not stat.S_ISDIR(namespace.st_mode)
        or namespace.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or raw_target not in {relative_target, absolute_target}
    ):
        return False
    return _root_owned_directory(
        absolute_target,
        sticky_allowed=path == Path("/tmp"),
    )


def _trusted_legacy_posix_tmp_alias(path: Path) -> bool:
    """Preserve the pre-PATH exact-/tmp exception outside Darwin."""

    if _runtime_os_name() != "posix" or _runtime_platform() == "darwin" or path != Path("/tmp"):
        return False
    if not path.is_symlink():
        return False
    try:
        link = path.lstat()
        target = path.stat()
    except OSError:
        return False
    return bool(
        link.st_uid == 0
        and target.st_uid == 0
        and stat.S_ISDIR(target.st_mode)
        and target.st_mode & stat.S_ISVTX
    )


def _is_untrusted_path_component(path: Path) -> bool:
    """Classify lexical escapes and redirecting filesystem components fail-closed."""

    if ".." in path.parts:
        return True
    try:
        redirecting = path.is_symlink()
        if _runtime_os_name() == "nt" and (redirecting or os.path.lexists(path)):
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            redirecting = redirecting or path.is_junction() or bool(attributes & reparse_flag)
    except OSError:
        return True
    if not redirecting:
        return False
    return not (_trusted_darwin_system_alias(path) or _trusted_legacy_posix_tmp_alias(path))
