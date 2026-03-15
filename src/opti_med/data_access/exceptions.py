"""Custom exceptions for data loading and validation."""

from __future__ import annotations

from pathlib import Path


class DataLoadError(Exception):
    """Raised when a source table cannot be loaded."""


class SchemaValidationError(DataLoadError):
    """Raised when a source table is missing required columns."""

    def __init__(self, table_name: str, path: Path, missing_columns: list[str]) -> None:
        message = (
            f"Table '{table_name}' at '{path}' is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        super().__init__(message)
