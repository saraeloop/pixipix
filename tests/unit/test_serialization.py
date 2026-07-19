from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from pixipix.errors import ProcessingError
from pixipix.models import Rect
from pixipix.serialization import canonical_json_bytes


def test_canonical_json_is_sorted_utf8_and_newline_terminated() -> None:
    rendered = canonical_json_bytes({"z": "café", "a": Rect(1, 2, 3, 4)})

    assert (
        rendered
        == (
            '{\n  "a": {\n    "bottom": 4,\n    "left": 1,\n    "right": 3,\n'
            '    "top": 2\n  },\n  "z": "café"\n}\n'
        ).encode()
    )
    assert not rendered.endswith(b"\n\n")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ProcessingError, match="PX_SERIALIZE_001"):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_absolute_paths() -> None:
    with pytest.raises(ProcessingError, match="PX_SERIALIZE_002"):
        canonical_json_bytes({"path": Path("/tmp/private.png")})
    assert b"frames/a.png" in canonical_json_bytes({"path": PurePosixPath("frames/a.png")})


@pytest.mark.parametrize(
    "path",
    (PureWindowsPath("C:/private.png"), PureWindowsPath("C:private.png"), PurePosixPath("../x")),
)
def test_canonical_json_rejects_cross_platform_or_traversing_paths(path: object) -> None:
    with pytest.raises(ProcessingError, match="PX_SERIALIZE_002"):
        canonical_json_bytes({"path": path})


def test_canonical_json_is_independent_of_dictionary_insertion_order() -> None:
    first = {"b": 2, "a": 1, "items": ["second", "first"]}
    second = {"items": ["second", "first"], "a": 1, "b": 2}

    assert canonical_json_bytes(first, pretty=False) == canonical_json_bytes(second, pretty=False)
    assert canonical_json_bytes({"items": [1, 2]}) != canonical_json_bytes({"items": [2, 1]})


def test_canonical_json_rejects_sets() -> None:
    with pytest.raises(ProcessingError, match="PX_SERIALIZE_004"):
        canonical_json_bytes({"values": {1, 2}})
