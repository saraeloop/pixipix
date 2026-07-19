"""Stable PixiPix domain errors and CLI exit-code mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    PROCESSING_FAILURE = 1
    CONFIGURATION_FAILURE = 2
    UNSUPPORTED_INPUT = 3
    INTERNAL_ERROR = 4


@dataclass(frozen=True, slots=True)
class PixiPixError(Exception):
    code: str
    stage: str
    message: str
    exit_code: ExitCode
    path: str | None = None
    frame: str | None = None
    remediation: str | None = None

    def __str__(self) -> str:
        context: list[str] = []
        if self.path is not None:
            context.append(f'path="{self.path}"')
        if self.frame is not None:
            context.append(f'frame="{self.frame}"')
        suffix = f" ({', '.join(context)})" if context else ""
        remedy = f" Remediation: {self.remediation}" if self.remediation else ""
        return f"{self.code} [{self.stage}] {self.message}{suffix}.{remedy}"


class ConfigurationError(PixiPixError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(
            code=code,
            stage="config",
            message=message,
            exit_code=ExitCode.CONFIGURATION_FAILURE,
            path=path,
            remediation=remediation,
        )


class UnsupportedInputError(PixiPixError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(
            code=code,
            stage="load",
            message=message,
            exit_code=ExitCode.UNSUPPORTED_INPUT,
            path=path,
            remediation=remediation,
        )


class ProcessingError(PixiPixError):
    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        *,
        path: str | None = None,
        frame: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(
            code=code,
            stage=stage,
            message=message,
            exit_code=ExitCode.PROCESSING_FAILURE,
            path=path,
            frame=frame,
            remediation=remediation,
        )
