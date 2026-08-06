"""Ownership-aware atomic publication for downstream stage outputs."""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings as python_warnings
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from pixipix.errors import ProcessingError, UnsupportedInputError
from pixipix.imageio import write_png
from pixipix.models import OutputMarker, UInt8Image
from pixipix.serialization import to_json_data, write_json

from .artifacts import (
    StageName,
    _dimensions,
    _is_output_marker,
    _is_schema_version_one,
    _is_untrusted_path_component,
    _read_json_object,
    _safe_frame_relative,
)


@dataclass(slots=True)
class OutputFrameImage:
    relative_path: PurePosixPath
    pixels: UInt8Image


type OwnedMetadataValidator = Callable[[dict[str, object]], bool]


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
        if _is_untrusted_path_component(candidate):
            raise ProcessingError(
                "PX_OUTPUT_004",
                "publish",
                "output path and untrusted existing parents must not be symlinks",
                path=output.name,
            )


def _valid_owned_output(
    path: Path,
    stage: StageName,
    owned_metadata_validator: OwnedMetadataValidator | None = None,
) -> bool:
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
        if owned_metadata_validator is not None and not owned_metadata_validator(metadata):
            return False
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            return False
        frames_root = path / "frames"
        if frames_root.is_symlink() or not frames_root.is_dir():
            return False
        seen: set[str] = set()
        expected: set[Path] = set()
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
            expected.add(frame_path)
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
        actual = set(frames_root.iterdir())
        return actual == expected and all(
            not frame_path.is_symlink() and frame_path.is_file() for frame_path in actual
        )
    except (UnsupportedInputError, OSError):
        return False


def _prepare_target(
    output: Path,
    force: bool,
    stage: StageName,
    owned_metadata_validator: OwnedMetadataValidator | None = None,
) -> None:
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
    if not _valid_owned_output(output, stage, owned_metadata_validator):
        raise ProcessingError(
            "PX_OUTPUT_003",
            "publish",
            "--force may replace only a valid PixiPix-owned output for this stage",
            path=output.name,
        )


def validate_stage_output_target(
    output: Path,
    stage: StageName,
    *,
    force: bool = False,
    owned_metadata_validator: OwnedMetadataValidator | None = None,
) -> None:
    """Validate a stage output target without creating or mutating it."""

    _prepare_target(output, force, stage, owned_metadata_validator)


def _remove_tree(path: Path, parent: Path, prefix: str) -> bool:
    if (
        path.parent != parent
        or not path.name.startswith(prefix)
        or any(_is_untrusted_path_component(candidate) for candidate in (path, *path.parents))
    ):
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
    stage: StageName,
    metadata: object,
    frames: tuple[OutputFrameImage, ...],
    *,
    force: bool = False,
    owned_metadata_validator: OwnedMetadataValidator | None = None,
) -> None:
    """Publish a complete typed stage output through a temporary sibling."""

    parent = output.parent
    build_prefix = f".{output.name}.pixipix-build-"
    backup_prefix = f".{output.name}.pixipix-backup-"
    build_root: Path | None = None
    backup_root: Path | None = None
    previous: Path | None = None
    try:
        _prepare_target(output, force, stage, owned_metadata_validator)
        parent.mkdir(parents=True, exist_ok=True)
        _prepare_target(output, force, stage, owned_metadata_validator)
        build_root = Path(tempfile.mkdtemp(prefix=build_prefix, dir=parent))
        frames_root = build_root / "frames"
        frames_root.mkdir()
        write_json(build_root / ".pixipix-output", OutputMarker(1, "pixipix", stage))
        for frame in frames:
            relative = _safe_frame_relative(frame.relative_path.as_posix())
            write_png(build_root.joinpath(*relative.parts), frame.pixels)
        write_json(build_root / "stage.json", metadata)
        _validate_staged(build_root, stage, metadata, frames)
        _prepare_target(output, force, stage, owned_metadata_validator)
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
