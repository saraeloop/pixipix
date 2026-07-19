"""Thin Typer adapter for PixiPix milestone commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from pixipix import __version__
from pixipix.config import load_config
from pixipix.errors import ExitCode, PixiPixError
from pixipix.models import Component, InspectionResult
from pixipix.stages.extract import inspect_source, publish_extraction

app = typer.Typer(
    name="pixipix",
    help="Tiny poses in. Tidy pixels out.",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def root(context: typer.Context) -> None:
    """Run PixiPix commands."""

    if context.invoked_subcommand is None:
        typer.echo(context.get_help())


def _call[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except PixiPixError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=int(error.exit_code)) from None
    except Exception:
        typer.echo(
            "PX_INTERNAL_001 [internal] unexpected internal error. "
            "Remediation: report the defect; tracebacks are intentionally disabled "
            "in this milestone.",
            err=True,
        )
        raise typer.Exit(code=int(ExitCode.INTERNAL_ERROR)) from None


def _component_line(index: int, component: Component) -> str:
    bounds = component.bounds
    return (
        f"  {index}: bounds=({bounds.left},{bounds.top})-({bounds.right},{bounds.bottom}) "
        f"area={component.area} discovery={component.discovery_index}"
    )


def _render_inspection(result: InspectionResult) -> str:
    source = result.source
    background = result.background
    lines = [
        f"source: {source.path}",
        f"dimensions: {source.width}x{source.height}",
        f"input mode: {source.input_mode}",
        f"normalized mode: {source.normalized_mode}",
        f"alpha present: {str(source.has_alpha).lower()}",
        f"background mode: {background.mode}",
        f"selected background: {background.selected_color or 'none'}",
        f"background tolerance: {background.tolerance}",
        f"pixels removed: {background.pixels_removed}",
        f"foreground touches boundary: {str(background.foreground_touches_boundary).lower()}",
    ]
    if background.foreground_bounds is None:
        lines.append("foreground bounds: none")
    else:
        bounds = background.foreground_bounds
        lines.append(
            f"foreground bounds: ({bounds.left},{bounds.top})-({bounds.right},{bounds.bottom})"
        )
    lines.extend(
        (
            f"candidate components: {len(result.candidates)}",
            f"accepted components: {len(result.accepted)}",
            f"rejected components: {len(result.rejected)}",
            "candidate component facts:",
        )
    )
    lines.extend(
        _component_line(component.discovery_index, component) for component in result.candidates
    )
    lines.append("rejected component facts:")
    lines.extend(
        f"{_component_line(item.component.discovery_index, item.component)} "
        f"reasons={','.join(item.reasons)}"
        for item in result.rejected
    )
    lines.append("deterministic accepted component order:")
    lines.extend(
        _component_line(index, component) for index, component in enumerate(result.ordered)
    )
    if result.frame_assignments is None:
        lines.append("frame assignments: unavailable (component/name count mismatch)")
    else:
        assignments = ", ".join(
            f"{index}={name}" for index, name in enumerate(result.frame_assignments)
        )
        lines.append(f"frame assignments: {assignments}")
    cell_size = result.configured_source_cell_size
    lines.append(
        f"configured source cell size: {cell_size if cell_size is not None else 'not configured'}"
    )
    return "\n".join(lines)


@app.command()
def version() -> None:
    """Print the PixiPix version."""

    typer.echo(f"PixiPix {__version__}")


@app.command("inspect")
def inspect_command(
    input_path: Annotated[Path, typer.Argument(exists=False, dir_okay=False, metavar="INPUT")],
    config_path: Annotated[Path, typer.Option("--config", dir_okay=False, metavar="CONFIG")],
) -> None:
    """Report deterministic source and extraction facts without writing output."""

    result = _call(lambda: inspect_source(input_path, load_config(config_path)))
    typer.echo(_render_inspection(result))


@app.command("extract")
def extract_command(
    input_path: Annotated[Path, typer.Argument(exists=False, dir_okay=False, metavar="INPUT")],
    config_path: Annotated[Path, typer.Option("--config", dir_okay=False, metavar="CONFIG")],
    output: Annotated[Path, typer.Option("--output", file_okay=False, metavar="OUTPUT")],
    force: Annotated[
        bool, typer.Option("--force", help="Replace only verified PixiPix-owned output.")
    ] = False,
) -> None:
    """Extract ordered RGBA frames and versioned stage metadata."""

    result = _call(
        lambda: publish_extraction(input_path, load_config(config_path), output, force=force)
    )
    typer.echo(f"extracted {len(result.frames)} frame(s) to {output}")


def main() -> None:
    app(prog_name="pixipix")
