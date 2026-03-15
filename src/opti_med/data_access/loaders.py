"""CSV table loaders for the MIMIC-IV MVP foundations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError, SchemaValidationError
from opti_med.data_access.schemas import CORE_TABLE_SCHEMAS


@dataclass(frozen=True)
class LoadedTable:
    """A loaded table and its source metadata."""

    name: str
    path: Path
    dataframe: pd.DataFrame


class MimicCoreLoader:
    """Load and validate core MIMIC-IV hospital tables."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all configured core tables."""
        return {
            table_name: self.load_table(table_name).dataframe
            for table_name in CORE_TABLE_SCHEMAS
        }

    def load_table(self, table_name: str) -> LoadedTable:
        """Load a single table by name and validate required columns."""
        if table_name not in CORE_TABLE_SCHEMAS:
            known_tables = ", ".join(sorted(CORE_TABLE_SCHEMAS))
            raise DataLoadError(
                f"Unknown table '{table_name}'. Expected one of: {known_tables}"
            )

        path = self._build_table_path(table_name)
        if not path.exists():
            raise DataLoadError(
                f"Expected table '{table_name}' at '{path}', but the file does not exist."
            )

        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - passthrough for pandas IO errors
            raise DataLoadError(f"Failed to load table '{table_name}' from '{path}': {exc}") from exc

        self._validate_required_columns(table_name, path, dataframe)
        return LoadedTable(name=table_name, path=path, dataframe=dataframe)

    def _build_table_path(self, table_name: str) -> Path:
        return self.settings.hosp_root / f"{table_name}{self.settings.file_extension}"

    @staticmethod
    def _validate_required_columns(
        table_name: str, path: Path, dataframe: pd.DataFrame
    ) -> None:
        required_columns = set(CORE_TABLE_SCHEMAS[table_name])
        actual_columns = set(dataframe.columns)
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            raise SchemaValidationError(table_name, path, missing_columns)


def summarize_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Build simple row and column count summaries for loaded tables."""
    summaries: list[str] = []
    for table_name, dataframe in tables.items():
        summaries.append(
            f"{table_name}: rows={len(dataframe):,}, columns={dataframe.shape[1]}"
        )
    return summaries
