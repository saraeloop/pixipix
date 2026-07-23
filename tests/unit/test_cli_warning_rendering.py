from __future__ import annotations

from typer.testing import CliRunner

from pixipix.cli import _render_warnings, _select_warnings
from pixipix.models import ProcessingWarning


def test_select_warnings_filters_by_structured_stage_without_mutating_or_deduplicating() -> None:
    scale = ProcessingWarning("PX_SCALE", "scale", "scale warning")
    unknown = ProcessingWarning("PX_FUTURE", "quantize", "future warning")
    align = ProcessingWarning("PX_ALIGN", "align", "align warning")
    warnings = (scale, align, unknown, align)

    selected = _select_warnings(warnings, command_stage="align", show_warnings=False)

    assert selected == (align, align)
    assert selected[0] is selected[1]
    assert warnings == (scale, align, unknown, align)


def test_select_warnings_returns_complete_tuple_in_stored_order() -> None:
    warnings = (
        ProcessingWarning("PX_SCALE", "scale", "scale warning"),
        ProcessingWarning("PX_FUTURE", "quantize", "future warning"),
        ProcessingWarning("PX_ALIGN", "align", "align warning"),
    )

    selected = _select_warnings(warnings, command_stage="align", show_warnings=True)

    assert selected is warnings


def _rendered_streams(
    warnings: tuple[ProcessingWarning, ...],
) -> tuple[bytes, bytes]:
    runner = CliRunner()
    with runner.isolation() as streams:
        _render_warnings(warnings)
        stdout, stderr, _combined = streams
        return stdout.getvalue(), stderr.getvalue()


def test_render_one_warning_writes_one_trailing_newline_to_stderr() -> None:
    warning = ProcessingWarning("PX_ONE", "scale", "one warning")

    stdout, stderr = _rendered_streams((warning,))

    expected = b"pixipix: warning [scale] PX_ONE: one warning\n"
    assert stdout == b""
    assert stderr == expected


def test_render_warnings_writes_exact_literal_stderr_bytes() -> None:
    warnings = (
        ProcessingWarning(
            "PX_ONE",
            "align",
            'frame "signal" uses [bold]literal[/bold] text',
        ),
        ProcessingWarning("PX_TWO", "quantize", "café pixel ✓"),
    )

    stdout, stderr = _rendered_streams(warnings)

    expected = (
        'pixipix: warning [align] PX_ONE: frame "signal" uses [bold]literal[/bold] text\n'
        "pixipix: warning [quantize] PX_TWO: café pixel ✓\n"
    ).encode()
    assert stdout == b""
    assert stderr == expected
    assert b"\x1b" not in stderr


def test_render_warnings_does_not_normalize_unvalidated_message_content() -> None:
    warning = ProcessingWarning(
        "PX_FUTURE",
        "quantize",
        "line one\nline two\r\ttab \x1b[31mliteral\x1b[0m",
    )

    stdout, stderr = _rendered_streams((warning,))

    assert stdout == b""
    assert stderr == (
        b"pixipix: warning [quantize] PX_FUTURE: line one\nline two\r\ttab \x1b[31mliteral\x1b[0m\n"
    )


def test_render_warnings_writes_zero_bytes_for_empty_tuple() -> None:
    stdout, stderr = _rendered_streams(())

    assert stdout == stderr == b""
