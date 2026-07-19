"""Central deterministic JSON serialization for public artifacts and hashes."""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePath, PurePosixPath

from pixipix.errors import ProcessingError


def _camel_case(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


def to_json_data(value: object) -> object:
    """Convert typed values to JSON data while enforcing artifact invariants."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProcessingError("PX_SERIALIZE_001", "serialize", "JSON numbers must be finite")
        return value
    if isinstance(value, Enum):
        return to_json_data(value.value)
    if isinstance(value, (Path, PurePath)):
        if value.is_absolute() or value.drive:
            raise ProcessingError(
                "PX_SERIALIZE_002",
                "serialize",
                "absolute paths are forbidden in public artifacts",
            )
        posix = PurePosixPath(value.as_posix())
        if posix.is_absolute() or ".." in posix.parts:
            raise ProcessingError(
                "PX_SERIALIZE_002",
                "serialize",
                "absolute and parent-traversing paths are forbidden in public artifacts",
            )
        return posix.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            _camel_case(field.name): to_json_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProcessingError(
                    "PX_SERIALIZE_003", "serialize", "JSON object keys must be strings"
                )
            output[key] = to_json_data(item)
        return output
    if isinstance(value, (list, tuple)):
        return [to_json_data(item) for item in value]
    raise ProcessingError(
        "PX_SERIALIZE_004",
        "serialize",
        f"unsupported public artifact value: {type(value).__name__}",
    )


def canonical_json_bytes(value: object, *, pretty: bool = True) -> bytes:
    if pretty:
        rendered = json.dumps(
            to_json_data(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    else:
        rendered = json.dumps(
            to_json_data(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"{rendered}\n".encode()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))
