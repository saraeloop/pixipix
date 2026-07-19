"""Deterministic source-space component extraction and publication."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from pixipix import __version__
from pixipix.config import ExtractConfig, LoadedConfig
from pixipix.errors import ProcessingError
from pixipix.imageio import generate_foreground_mask, load_source, write_png
from pixipix.models import (
    BackgroundSummary,
    BoolMask,
    Component,
    ExtractedFrame,
    ExtractionResult,
    ExtractionRun,
    FrameImage,
    InspectionResult,
    OutputMarker,
    Rect,
    RejectedComponent,
    RejectionReason,
    SourceImage,
    StageMetadata,
)
from pixipix.serialization import to_json_data, write_json

LabelMap = npt.NDArray[np.int32]

FOUR_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (0, -1),  # left
    (0, 1),  # right
    (1, 0),  # down
)
EIGHT_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, -1),  # up-left
    (-1, 0),  # up
    (-1, 1),  # up-right
    (0, -1),  # left
    (0, 1),  # right
    (1, -1),  # down-left
    (1, 0),  # down
    (1, 1),  # down-right
)


@dataclass(slots=True)
class ComponentMap:
    components: tuple[Component, ...]
    labels: LabelMap


@dataclass(slots=True)
class _Analysis:
    source: SourceImage
    mask: BoolMask
    component_map: ComponentMap
    accepted: tuple[Component, ...]
    rejected: tuple[RejectedComponent, ...]
    ordered: tuple[Component, ...]
    background: BackgroundSummary


def label_components(mask: BoolMask, connectivity: int, max_components: int) -> ComponentMap:
    """Label foreground using row-major discovery and ADR-locked neighbor order."""

    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    neighbors = FOUR_NEIGHBORS if connectivity == 4 else EIGHT_NEIGHBORS
    components: list[Component] = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column] != 0:
                continue
            discovery_index = len(components)
            if discovery_index >= max_components:
                raise ProcessingError(
                    "PX_EXTRACT_001",
                    "extract",
                    f"candidate component count exceeds configured limit {max_components}",
                    remediation="raise max_components explicitly or remove foreground noise",
                )
            label = discovery_index + 1
            pending: deque[tuple[int, int]] = deque([(row, column)])
            labels[row, column] = label
            area = 0
            left = column
            right = column
            top = row
            bottom = row
            while pending:
                current_row, current_column = pending.popleft()
                area += 1
                left = min(left, current_column)
                right = max(right, current_column)
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                for row_offset, column_offset in neighbors:
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and labels[next_row, next_column] == 0
                    ):
                        labels[next_row, next_column] = label
                        pending.append((next_row, next_column))
            components.append(
                Component(
                    discovery_index=discovery_index,
                    bounds=Rect(left=left, top=top, right=right + 1, bottom=bottom + 1),
                    area=area,
                )
            )
    return ComponentMap(tuple(components), labels)


def filter_components(
    components: tuple[Component, ...], config: ExtractConfig
) -> tuple[tuple[Component, ...], tuple[RejectedComponent, ...]]:
    accepted: list[Component] = []
    rejected: list[RejectedComponent] = []
    for component in components:
        reasons: list[RejectionReason] = []
        if component.area < config.minimum_area:
            reasons.append("below-minimum-area")
        if config.maximum_area is not None and component.area > config.maximum_area:
            reasons.append("above-maximum-area")
        if reasons:
            rejected.append(RejectedComponent(component, tuple(reasons)))
        else:
            accepted.append(component)
    return tuple(accepted), tuple(rejected)


def order_components(
    components: tuple[Component, ...], row_tolerance: int
) -> tuple[Component, ...]:
    """Group by row top, then apply the locked deterministic reading order."""

    by_top = sorted(
        components,
        key=lambda item: (
            item.bounds.top,
            item.bounds.left,
            item.area,
            item.discovery_index,
        ),
    )
    rows: list[tuple[int, list[Component]]] = []
    for component in by_top:
        matching_row = next(
            (row for row in rows if abs(component.bounds.top - row[0]) <= row_tolerance),
            None,
        )
        if matching_row is None:
            rows.append((component.bounds.top, [component]))
        else:
            matching_row[1].append(component)
    ordered: list[Component] = []
    for _, row in sorted(rows, key=lambda item: item[0]):
        ordered.extend(
            sorted(
                row,
                key=lambda item: (item.bounds.left, item.area, item.discovery_index),
            )
        )
    return tuple(ordered)


def _analyze(input_path: Path, loaded: LoadedConfig) -> _Analysis:
    config = loaded.config
    source = load_source(input_path, config.source)
    mask, background = generate_foreground_mask(source, config.background)
    component_map = label_components(
        mask, config.extract.connectivity, config.source.max_components
    )
    accepted, rejected = filter_components(component_map.components, config.extract)
    ordered = order_components(accepted, config.extract.row_tolerance)
    return _Analysis(source, mask, component_map, accepted, rejected, ordered, background)


def inspect_source(input_path: Path, loaded: LoadedConfig) -> InspectionResult:
    analysis = _analyze(input_path, loaded)
    assignments: tuple[str, ...] | None = None
    if len(analysis.ordered) == len(loaded.config.frames.names):
        assignments = loaded.config.frames.names
    return InspectionResult(
        source=analysis.source.metadata,
        background=analysis.background,
        candidates=analysis.component_map.components,
        accepted=analysis.accepted,
        rejected=analysis.rejected,
        ordered=analysis.ordered,
        frame_assignments=assignments,
        configured_source_cell_size=loaded.config.pixelize.source_cell_size,
    )


def _padded_bounds(bounds: Rect, padding: int, width: int, height: int) -> Rect:
    return Rect(
        left=max(0, bounds.left - padding),
        top=max(0, bounds.top - padding),
        right=min(width, bounds.right + padding),
        bottom=min(height, bounds.bottom + padding),
    )


def extract_source(input_path: Path, loaded: LoadedConfig) -> ExtractionRun:
    analysis = _analyze(input_path, loaded)
    config = loaded.config
    accepted_count = len(analysis.ordered)
    expected = config.source.expected_components
    if expected is not None and accepted_count != expected:
        raise ProcessingError(
            "PX_EXTRACT_002",
            "extract",
            f"accepted component count {accepted_count} does not match expected count {expected}",
            path=input_path.name,
            remediation="inspect component bounds and adjust extraction filters",
        )
    if accepted_count != len(config.frames.names):
        raise ProcessingError(
            "PX_EXTRACT_003",
            "name",
            f"accepted component count {accepted_count} does not match "
            f"frame-name count {len(config.frames.names)}",
            path=input_path.name,
            remediation="inspect component bounds and update thresholds or frame names",
        )

    metadata: list[ExtractedFrame] = []
    images: list[FrameImage] = []
    height, width = analysis.mask.shape
    for source_order, (component, name, filename) in enumerate(
        zip(analysis.ordered, config.frames.names, config.frames.filenames, strict=True)
    ):
        padded = _padded_bounds(component.bounds, config.extract.padding, width, height)
        crop = np.array(
            analysis.source.pixels[padded.top : padded.bottom, padded.left : padded.right],
            dtype=np.uint8,
            copy=True,
        )
        label_crop = analysis.component_map.labels[
            padded.top : padded.bottom, padded.left : padded.right
        ]
        crop[label_crop != component.discovery_index + 1] = 0
        frame = ExtractedFrame(
            name=name,
            relative_path=PurePosixPath("frames") / filename,
            source_order=source_order,
            discovery_index=component.discovery_index,
            component_area=component.area,
            original_bounds=component.bounds,
            padded_bounds=padded,
        )
        metadata.append(frame)
        images.append(FrameImage(frame, crop))
    result = ExtractionResult(
        source=analysis.source.metadata,
        background=analysis.background,
        candidates=analysis.component_map.components,
        accepted=analysis.accepted,
        rejected=analysis.rejected,
        ordered=analysis.ordered,
        frames=tuple(metadata),
    )
    return ExtractionRun(result=result, frame_images=tuple(images))


def _stage_metadata(run: ExtractionRun, loaded: LoadedConfig) -> StageMetadata:
    result = run.result
    return StageMetadata(
        schema_version=1,
        pixipix_version=__version__,
        stage="extract",
        status="successful",
        source_config_sha256=loaded.source_config_sha256,
        effective_config_sha256=loaded.effective_config_sha256,
        source=result.source,
        background=result.background,
        candidate_components=result.candidates,
        accepted_components=result.accepted,
        rejected_components=result.rejected,
        ordered_components=result.ordered,
        frames=result.frames,
        warnings=result.warnings,
    )


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
