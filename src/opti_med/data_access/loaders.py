"""CSV table loaders for the MIMIC-IV clinical and ED demo datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError, SchemaValidationError
from opti_med.data_access.schemas import (
    CLINICAL_CORE_TABLE_SCHEMAS,
    CLINICAL_OPTIONAL_TABLE_SCHEMAS,
    ED_CORE_TABLE_SCHEMAS,
    ED_OPTIONAL_TABLE_SCHEMAS,
)


@dataclass(frozen=True)
class LoadedTable:
    """A loaded table and its source metadata."""

    name: str
    path: Path
    dataframe: pd.DataFrame
    dataset_name: str = "clinical"


class MimicCoreLoader:
    """Load and validate the current clinical demo tables used by the MVP."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all configured core tables."""
        return {
            table_name: self.load_table(table_name).dataframe
            for table_name in CLINICAL_CORE_TABLE_SCHEMAS
        }

    def load_table(self, table_name: str) -> LoadedTable:
        """Load a single table by name and validate required columns."""
        if table_name not in CLINICAL_CORE_TABLE_SCHEMAS:
            known_tables = ", ".join(sorted(CLINICAL_CORE_TABLE_SCHEMAS))
            raise DataLoadError(
                f"Unknown table '{table_name}'. Expected one of: {known_tables}"
            )

        return self._load_clinical_table(
            table_name,
            required_columns=CLINICAL_CORE_TABLE_SCHEMAS[table_name],
        )

    def load_optional_table(self, table_name: str) -> LoadedTable | None:
        """Load one optional clinical table when present, otherwise return None."""
        if table_name not in CLINICAL_OPTIONAL_TABLE_SCHEMAS:
            known_tables = ", ".join(sorted(CLINICAL_OPTIONAL_TABLE_SCHEMAS))
            raise DataLoadError(
                f"Unknown optional clinical table '{table_name}'. Expected one of: {known_tables}"
            )

        path = self._build_table_path(table_name)
        if not path.exists():
            return None

        return self._load_clinical_table(
            table_name,
            required_columns=CLINICAL_OPTIONAL_TABLE_SCHEMAS[table_name],
        )

    def _build_table_path(self, table_name: str) -> Path:
        return self.settings.hosp_root / f"{table_name}{self.settings.file_extension}"

    def _load_clinical_table(
        self,
        table_name: str,
        *,
        required_columns: list[str],
    ) -> LoadedTable:
        path = self._build_table_path(table_name)
        if not path.exists():
            raise DataLoadError(
                f"Expected table '{table_name}' at '{path}', but the file does not exist."
            )

        try:
            dataframe = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # pragma: no cover - passthrough for pandas IO errors
            raise DataLoadError(f"Failed to load table '{table_name}' from '{path}': {exc}") from exc

        self._validate_required_columns(
            table_name,
            path,
            dataframe,
            required_columns=required_columns,
        )
        return LoadedTable(
            name=table_name,
            path=path,
            dataframe=dataframe,
            dataset_name="clinical",
        )

    @staticmethod
    def _validate_required_columns(
        table_name: str,
        path: Path,
        dataframe: pd.DataFrame,
        *,
        required_columns: list[str],
    ) -> None:
        actual_columns = set(dataframe.columns)
        missing_columns = sorted(set(required_columns) - actual_columns)
        if missing_columns:
            raise SchemaValidationError(table_name, path, missing_columns)


class MimicDualDemoLoader:
    """Load the configured clinical and ED demos while keeping the current app intact."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clinical_loader = MimicCoreLoader(settings)

    def load_clinical_table(self, table_name: str) -> LoadedTable:
        """Load one required clinical table used by the current pipeline."""
        return self.clinical_loader.load_table(table_name)

    def load_ed_table(self, table_name: str, *, required: bool = True) -> LoadedTable:
        """Load one ED table from the ED demo tree."""
        schemas = ED_CORE_TABLE_SCHEMAS if required else ED_OPTIONAL_TABLE_SCHEMAS
        if table_name not in schemas:
            known_tables = ", ".join(sorted({*ED_CORE_TABLE_SCHEMAS, *ED_OPTIONAL_TABLE_SCHEMAS}))
            raise DataLoadError(
                f"Unknown ED table '{table_name}'. Expected one of: {known_tables}"
            )

        path = self.settings.ed_root / f"{table_name}{self.settings.file_extension}"
        if not path.exists():
            raise DataLoadError(
                f"Expected ED table '{table_name}' at '{path}', but the file does not exist."
            )

        try:
            dataframe = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # pragma: no cover - passthrough for pandas IO errors
            raise DataLoadError(f"Failed to load ED table '{table_name}' from '{path}': {exc}") from exc

        self.clinical_loader._validate_required_columns(
            table_name,
            path,
            dataframe,
            required_columns=schemas[table_name],
        )
        return LoadedTable(
            name=table_name,
            path=path,
            dataframe=dataframe,
            dataset_name="ed",
        )

    def load_optional_ed_table(self, table_name: str) -> LoadedTable | None:
        """Load one optional ED table when present, otherwise return None."""
        if table_name not in ED_OPTIONAL_TABLE_SCHEMAS:
            known_tables = ", ".join(sorted(ED_OPTIONAL_TABLE_SCHEMAS))
            raise DataLoadError(
                f"Unknown optional ED table '{table_name}'. Expected one of: {known_tables}"
            )

        path = self.settings.ed_root / f"{table_name}{self.settings.file_extension}"
        if not path.exists():
            return None
        return self.load_ed_table(table_name, required=False)

    def load_dual_demo_tables(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Load the core clinical tables plus the ED core table set."""
        return {
            "clinical": self.clinical_loader.load_all(),
            "ed": {
                table_name: self.load_ed_table(table_name).dataframe
                for table_name in ED_CORE_TABLE_SCHEMAS
            },
        }

def summarize_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Build simple row and column count summaries for loaded tables."""
    summaries: list[str] = []
    for table_name, dataframe in tables.items():
        summaries.append(
            f"{table_name}: rows={len(dataframe):,}, columns={dataframe.shape[1]}"
        )
    return summaries
