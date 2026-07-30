"""Extract-specific publication, validation, replacement, and rollback."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from pixipix.config import LoadedConfig
from pixipix.errors import ProcessingError
from pixipix.imageio import write_png
from pixipix.models import ExtractionResult, OutputMarker, StageMetadata
from pixipix.serialization import to_json_data, write_json

from .api import extract_source
from .metadata import _stage_metadata


def _valid_marker(path: Path) -> bool:
    marker = path / ".pixipix-output"
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(value == {"owner": "pixipix", "schemaVersion": 1, "stage": "extract"})


def _frame_path(root: Path, relative: PurePosixPath) -> Path:
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "frames"
        or not relative.name.endswith(".png")
    ):
        raise ProcessingError("PX_OUTPUT_006", "publish", f'unsafe staged frame path "{relative}"')
    return root.joinpath(*relative.parts)


def _valid_frame_png(path: Path, expected_size: tuple[int, int] | None = None) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return bool(
                image.format == "PNG"
                and image.mode == "RGBA"
                and (expected_size is None or image.size == expected_size)
            )
    except (UnidentifiedImageError, OSError):
        return False


def _validate_staged_payload(root: Path, metadata: StageMetadata, stage_path: Path) -> None:
    if not _valid_marker(root) or stage_path.is_symlink() or not stage_path.is_file():
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged output is incomplete")
    try:
        rendered = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged metadata is not valid JSON"
        ) from error
    if rendered != to_json_data(metadata):
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged metadata does not match extraction results"
        )
    expected_paths: set[Path] = set()
    for frame in metadata.frames:
        path = _frame_path(root, frame.relative_path)
        expected_paths.add(path)
        expected_size = (frame.padded_bounds.width, frame.padded_bounds.height)
        if not _valid_frame_png(path, expected_size):
            raise ProcessingError(
                "PX_OUTPUT_006", "publish", f'invalid staged frame "{frame.relative_path}"'
            )
    frames_root = root / "frames"
    if frames_root.is_symlink() or not frames_root.is_dir():
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged frame directory is invalid")
    actual_paths = {path for path in frames_root.iterdir() if path.is_file()}
    if any(path.is_symlink() or not path.is_file() for path in frames_root.iterdir()):
        raise ProcessingError("PX_OUTPUT_006", "publish", "staged frame directory is invalid")
    if actual_paths != expected_paths:
        raise ProcessingError(
            "PX_OUTPUT_006", "publish", "staged frame files do not match metadata"
        )


def _validate_staged_output(root: Path, metadata: StageMetadata) -> None:
    _validate_staged_payload(root, metadata, root / "stage.json")


def _valid_owned_output(path: Path) -> bool:
    """Require a coherent successful contract; a marker alone never grants ownership."""

    if not _valid_marker(path):
        return False
    stage_path = path / "stage.json"
    if stage_path.is_symlink() or not stage_path.is_file():
        return False
    try:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(stage, dict) or any(
        stage.get(key) != value
        for key, value in (
            ("schemaVersion", 1),
            ("stage", "extract"),
            ("status", "successful"),
        )
    ):
        return False
    required_container_types = {
        "source": dict,
        "background": dict,
        "candidateComponents": list,
        "acceptedComponents": list,
        "rejectedComponents": list,
        "orderedComponents": list,
        "warnings": list,
    }
    if any(
        not isinstance(stage.get(key), expected)
        for key, expected in required_container_types.items()
    ):
        return False
    for key in ("sourceConfigSha256", "effectiveConfigSha256"):
        value = stage.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False
    if not isinstance(stage.get("pixipixVersion"), str):
        return False
    frames = stage.get("frames")
    if not isinstance(frames, list) or not frames:
        return False
    seen: set[str] = set()
    for source_order, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("sourceOrder") != source_order:
            return False
        raw_relative = frame.get("relativePath")
        if not isinstance(raw_relative, str) or "\\" in raw_relative:
            return False
        relative = PurePosixPath(raw_relative)
        if relative.as_posix() != raw_relative:
            return False
        try:
            frame_path = _frame_path(path, relative)
        except ProcessingError:
            return False
        folded = relative.as_posix().casefold()
        if folded in seen or not _valid_frame_png(frame_path):
            return False
        seen.add(folded)
    return True


def _is_trusted_system_tmp_alias(path: Path) -> bool:
    """Allow the root-owned `/tmp` alias used by platforms such as macOS."""

    if path != Path("/tmp") or not path.is_symlink():
        return False
    try:
        link_info = path.lstat()
        resolved = path.resolve(strict=True)
        target_info = resolved.stat()
    except OSError:
        return False
    return bool(
        link_info.st_uid == 0
        and target_info.st_uid == 0
        and stat.S_ISDIR(target_info.st_mode)
        and target_info.st_mode & stat.S_ISVTX
    )


def _validate_output_location(output: Path) -> None:
    resolved = output.resolve(strict=False)
    dangerous = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in dangerous:
        raise ProcessingError(
            "PX_OUTPUT_007",
            "publish",
            "refusing to use a dangerous output location",
            path=output.name or ".",
        )
    for candidate in (output, *output.parents):
        if candidate.is_symlink() and not _is_trusted_system_tmp_alias(candidate):
            raise ProcessingError(
                "PX_OUTPUT_004",
                "publish",
                "output path and untrusted existing parents must not be symlinks",
                path=output.name,
            )


def _prepare_target(output: Path, force: bool) -> None:
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
    non_empty = next(output.iterdir(), None) is not None
    if not non_empty:
        return
    if not force:
        raise ProcessingError(
            "PX_OUTPUT_002",
            "publish",
            "non-empty output directory is rejected without --force",
            path=output.name,
        )
    if not _valid_owned_output(output):
        raise ProcessingError(
            "PX_OUTPUT_003",
            "publish",
            "--force may replace only a valid PixiPix-owned output",
            path=output.name,
        )


def _remove_temporary_tree(path: Path, parent: Path, prefix: str) -> bool:
    if path.parent != parent or not path.name.startswith(prefix) or path.is_symlink():
        return False
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return True


def publish_extraction(
    input_path: Path, loaded: LoadedConfig, output: Path, *, force: bool = False
) -> ExtractionResult:
    _prepare_target(output, force)
    run = extract_source(input_path, loaded)
    metadata = _stage_metadata(run, loaded)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    _prepare_target(output, force)
    build_prefix = f".{output.name}.pixipix-build-"
    backup_prefix = f".{output.name}.pixipix-backup-"
    build_root = Path(tempfile.mkdtemp(prefix=build_prefix, dir=parent))
    backup_root: Path | None = None
    previous: Path | None = None
    try:
        frames_root = build_root / "frames"
        frames_root.mkdir()
        marker = OutputMarker(schema_version=1, owner="pixipix", stage="extract")
        write_json(build_root / ".pixipix-output", marker)
        for frame in run.frame_images:
            frame_path = build_root.joinpath(*frame.metadata.relative_path.parts)
            write_png(frame_path, frame.pixels)
        pending_stage = build_root / ".stage.json.pending"
        write_json(pending_stage, metadata)
        _validate_staged_payload(build_root, metadata, pending_stage)
        pending_stage.replace(build_root / "stage.json")
        _validate_staged_output(build_root, metadata)
        # Revalidate after staging so a target created or changed during processing is
        # never moved aside under stale ownership assumptions.
        _prepare_target(output, force)

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
        if backup_root is not None and _remove_temporary_tree(backup_root, parent, backup_prefix):
            backup_root = None
        return run.result
    except ProcessingError:
        raise
    except OSError as error:
        raise ProcessingError(
            "PX_OUTPUT_005", "publish", "unable to write extraction output", path=output.name
        ) from error
    finally:
        if build_root.exists():
            _remove_temporary_tree(build_root, parent, build_prefix)
        if backup_root is not None and backup_root.exists():
            if previous is not None and previous.exists() and not output.exists():
                with suppress(OSError):
                    previous.replace(output)
            if backup_root.exists() and output.exists():
                _remove_temporary_tree(backup_root, parent, backup_prefix)
