from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pixipix import __version__
from pixipix.config import LoadedConfig, load_config
from pixipix.models import UInt8Image


def write_rgba(path: Path, pixels: UInt8Image) -> None:
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")


def write_rgb(path: Path, pixels: UInt8Image) -> None:
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def transparent_sheet() -> UInt8Image:
    pixels = np.zeros((8, 14, 4), dtype=np.uint8)
    pixels[1:4, 1:4] = (20, 40, 60, 255)
    pixels[1:3, 8:12] = (120, 80, 40, 200)
    pixels[6, 13] = (255, 0, 0, 255)  # deterministic rejected noise
    return pixels


def extraction_config(
    *,
    names: tuple[str, ...] = ("idle", "signal"),
    background: str = 'mode = "alpha"\nalpha_threshold = 8',
    minimum_area: int = 2,
    maximum_area: int | None = None,
    padding: int = 1,
    expected: int | None = 2,
    max_components: int = 16,
) -> str:
    expected_line = f"expected_components = {expected}\n" if expected is not None else ""
    maximum_line = f"maximum_area = {maximum_area}\n" if maximum_area is not None else ""
    quoted_names = ", ".join(f'"{name}"' for name in names)
    return (
        "[project]\n"
        'name = "test"\n'
        "strict = true\n\n"
        "[source]\n"
        f"{expected_line}"
        "max_width = 64\n"
        "max_height = 64\n"
        "max_pixels = 4096\n"
        f"max_components = {max_components}\n\n"
        "[background]\n"
        f"{background}\n\n"
        "[extract]\n"
        "connectivity = 8\n"
        f"minimum_area = {minimum_area}\n"
        f"{maximum_line}"
        f"padding = {padding}\n"
        "row_tolerance = 2\n\n"
        "[frames]\n"
        f"names = [{quoted_names}]\n"
    )


def write_config(path: Path, content: str | None = None) -> None:
    path.write_text(content or extraction_config(), encoding="utf-8")


def pipeline_config(
    *,
    names: tuple[str, ...] = ("idle", "signal"),
    scale: str = 'mode = "explicit-factor"\nfactor = 1.0',
    pixelize: str = (
        "source_cell_size = 2\n"
        'representative = "alpha-weighted-majority"\n'
        'alpha_policy = "binary"\n'
        "alpha_threshold = 128\n"
        'remainder_policy = "pad-transparent"'
    ),
    overrides: str = "",
    padding: int = 0,
    output: str = "",
    offsets: str = "",
) -> str:
    return (
        extraction_config(
            names=names,
            expected=len(names),
            padding=padding,
            max_components=max(16, len(names)),
        )
        + "\n[scale]\n"
        + scale
        + "\n\n[pixelize]\n"
        + pixelize
        + ("\n\n" + overrides if overrides else "")
        + ("\n\n[output]\n" + output if output else "")
        + ("\n\n" + offsets if offsets else "")
        + "\n"
    )


def alignment_config(
    *,
    names: tuple[str, ...] = ("idle", "signal"),
    width: int = 8,
    height: int = 8,
    anchor: str = "bottom-center",
    baseline_y: int | None = None,
    clip_policy: str | None = "error",
    offsets: str = "",
) -> str:
    baseline = f"\nbaseline_y = {baseline_y}" if baseline_y is not None else ""
    policy = f'\nclip_policy = "{clip_policy}"' if clip_policy is not None else ""
    output = (
        f'frame_width = {width}\nframe_height = {height}\nanchor = "{anchor}"{baseline}{policy}'
    )
    return pipeline_config(names=names, output=output, offsets=offsets)


def _write_declared_stage(
    root: Path,
    loaded: LoadedConfig,
    stage: str,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    frames_root = root / "frames"
    frames_root.mkdir(parents=True)
    marker = {"owner": "pixipix", "schemaVersion": 1, "stage": stage}
    (root / ".pixipix-output").write_text(json.dumps(marker), encoding="utf-8")
    for frame in frames:
        relative_path = frame["relativePath"]
        assert isinstance(relative_path, str)
        (root / relative_path).write_bytes(b"not a PNG")
    document = {
        "schemaVersion": 1,
        "pixipixVersion": __version__,
        "stage": stage,
        "status": "successful",
        "sourceConfigSha256": loaded.source_config_sha256,
        "effectiveConfigSha256": loaded.effective_config_sha256,
        "frames": frames,
        "warnings": [],
        **metadata,
    }
    (root / "stage.json").write_text(json.dumps(document), encoding="utf-8")


def write_declared_extract_stage(
    root: Path,
    loaded: LoadedConfig,
    dimensions: tuple[tuple[int, int], ...],
) -> None:
    """Write metadata-valid extract declarations backed by deliberately invalid PNG bytes."""

    assert len(dimensions) == len(loaded.config.frames.names)
    frames = [
        {
            "name": name,
            "relativePath": f"frames/{filename}",
            "sourceOrder": source_order,
            "paddedBounds": {
                "left": 0,
                "top": 0,
                "right": width,
                "bottom": height,
            },
        }
        for source_order, (name, filename, (width, height)) in enumerate(
            zip(
                loaded.config.frames.names,
                loaded.config.frames.filenames,
                dimensions,
                strict=True,
            )
        )
    ]
    _write_declared_stage(root, loaded, "extract", frames, {})


def write_declared_scale_stage(
    root: Path,
    loaded: LoadedConfig,
    input_dimensions: tuple[tuple[int, int], ...],
    output_dimensions: tuple[tuple[int, int], ...],
    *,
    factor: float,
) -> None:
    """Write structurally valid scale declarations backed by deliberately invalid PNG bytes."""

    assert len(input_dimensions) == len(output_dimensions)
    assert len(input_dimensions) == len(loaded.config.frames.names)
    frames = [
        {
            "name": name,
            "relativePath": f"frames/{filename}",
            "sourceOrder": source_order,
            "inputDimensions": {"width": input_width, "height": input_height},
            "outputDimensions": {"width": output_width, "height": output_height},
            "scaleMultiplier": 1.0,
            "effectiveFactor": factor,
        }
        for source_order, (
            name,
            filename,
            (input_width, input_height),
            (output_width, output_height),
        ) in enumerate(
            zip(
                loaded.config.frames.names,
                loaded.config.frames.filenames,
                input_dimensions,
                output_dimensions,
                strict=True,
            )
        )
    ]
    prior = {
        "stage": "extract",
        "schemaVersion": 1,
        "pixipixVersion": __version__,
        "effectiveConfigSha256": loaded.effective_config_sha256,
    }
    _write_declared_stage(
        root,
        loaded,
        "scale",
        frames,
        {
            "priorStage": prior,
            "scaleMode": "explicit-factor",
            "globalFactor": factor,
            "referenceFrame": None,
            "sourceReferenceMeasurement": None,
            "exactTargetSourceMeasurement": None,
            "logicalTargetSize": None,
            "sourceCellSize": loaded.config.pixelize.source_cell_size,
            "configuredFrameOverrides": [],
        },
    )


def write_declared_pixelize_stage(
    root: Path,
    loaded: LoadedConfig,
    logical_dimensions: tuple[tuple[int, int], ...],
) -> None:
    """Write metadata-valid pixelize declarations backed by invalid PNG bytes."""

    cell_size = loaded.config.pixelize.source_cell_size
    assert cell_size is not None
    assert len(logical_dimensions) == len(loaded.config.frames.names)
    frames = [
        {
            "name": name,
            "relativePath": f"frames/{filename}",
            "sourceOrder": source_order,
            "inputDimensions": {
                "width": logical_width * cell_size,
                "height": logical_height * cell_size,
            },
            "preparedDimensions": {
                "width": logical_width * cell_size,
                "height": logical_height * cell_size,
            },
            "topPadding": 0,
            "rightPadding": 0,
            "topCrop": 0,
            "rightCrop": 0,
            "logicalOutputDimensions": {
                "width": logical_width,
                "height": logical_height,
            },
        }
        for source_order, (name, filename, (logical_width, logical_height)) in enumerate(
            zip(
                loaded.config.frames.names,
                loaded.config.frames.filenames,
                logical_dimensions,
                strict=True,
            )
        )
    ]
    prior = {
        "stage": "scale",
        "schemaVersion": 1,
        "pixipixVersion": __version__,
        "effectiveConfigSha256": loaded.effective_config_sha256,
    }
    pixelize = loaded.config.pixelize
    _write_declared_stage(
        root,
        loaded,
        "pixelize",
        frames,
        {
            "priorStage": prior,
            "sourceCellSize": cell_size,
            "cellGridOrigin": "bottom-left",
            "representative": pixelize.representative,
            "alphaPolicy": pixelize.alpha_policy,
            "alphaThreshold": pixelize.alpha_threshold,
            "remainderPolicy": pixelize.remainder_policy,
        },
    )


def _resource_frame_names(count: int) -> tuple[str, ...]:
    return tuple(f"frame-{index:03d}" for index in range(count))


def resource_scenario_e(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "scenario-e.toml"
    write_config(
        config,
        pipeline_config(
            names=_resource_frame_names(4),
            scale='mode = "explicit-factor"\nfactor = 2.0',
        )
        + (
            "\n[resources]\n"
            "max_aggregate_input_pixels = 50000000\n"
            "max_aggregate_output_pixels = 80000000\n"
            "max_modeled_peak_live_bytes = 1500000000\n"
        ),
    )
    loaded = load_config(config)
    input_root = tmp_path / "scenario-e-scale"
    dimensions = ((2047, 2047),) * 4
    write_declared_scale_stage(
        input_root,
        loaded,
        dimensions,
        ((4094, 4094),) * 4,
        factor=2.0,
    )
    return config, input_root, tmp_path / "scenario-e-output"


def resource_scenario_f(tmp_path: Path) -> tuple[Path, Path, Path]:
    names = _resource_frame_names(61)
    config = tmp_path / "scenario-f.toml"
    write_config(
        config,
        pipeline_config(
            names=names,
            scale='mode = "explicit-factor"\nfactor = 2.0',
            overrides=(f"[frame_overrides.{names[-1]}]\nscale_multiplier = 0.5"),
        ),
    )
    loaded = load_config(config)
    input_root = tmp_path / "scenario-f-extract"
    write_declared_extract_stage(
        input_root,
        loaded,
        ((500, 500),) * 60 + ((1, 1),),
    )
    return config, input_root, tmp_path / "scenario-f-output"


def resource_scenario_g(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "scenario-g.toml"
    write_config(
        config,
        pipeline_config(
            names=("large",),
            scale='mode = "explicit-factor"\nfactor = 1.0',
        ),
    )
    loaded = load_config(config)
    input_root = tmp_path / "scenario-g-extract"
    write_declared_extract_stage(input_root, loaded, ((4096, 4096),))
    return config, input_root, tmp_path / "scenario-g-output"


def resource_scenario_h(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "scenario-h.toml"
    write_config(
        config,
        pipeline_config(
            names=_resource_frame_names(100),
            scale='mode = "explicit-factor"\nfactor = 1.2',
        )
        + (
            "\n[resources]\n"
            "max_aggregate_input_pixels = 50000000\n"
            "max_aggregate_output_pixels = 120000000\n"
            "max_modeled_peak_live_bytes = 1000000000\n"
        ),
    )
    loaded = load_config(config)
    input_root = tmp_path / "scenario-h-extract"
    write_declared_extract_stage(input_root, loaded, ((1000, 1000),) * 100)
    return config, input_root, tmp_path / "scenario-h-output"
