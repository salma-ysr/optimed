"""Helpers for loading standardized Parquet tables and analytical artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.standardized.specs import TableStandardizationSpec, get_table_spec


@dataclass(frozen=True, slots=True)
class StandardizedTableLoadResult:
    """Loaded standardized table plus manifest metadata when available."""

    dataset_name: str
    table_name: str
    path: Path
    dataframe: pd.DataFrame
    standardized_reference: dict[str, object] | None
    manifest_entry: dict[str, object] | None


class StandardizedParquetRepository:
    """Read standardized source tables and persisted analytical artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._latest_manifest_payload: dict[str, object] | None = None

    def load_source_table(
        self,
        dataset_name: str,
        table_name: str,
        *,
        columns: list[str] | None = None,
        required: bool = True,
    ) -> StandardizedTableLoadResult | None:
        """Load one standardized source table from Parquet."""
        spec = get_table_spec(dataset_name, table_name)
        path = self.source_table_path(spec)
        if not path.exists():
            if required:
                raise DataLoadError(
                    f"Expected standardized table '{dataset_name}.{table_name}' at '{path}', but it does not exist."
                )
            return None

        try:
            dataframe = pd.read_parquet(path, columns=columns)
        except Exception as exc:  # pragma: no cover - passthrough for parquet IO errors
            raise DataLoadError(
                f"Failed to load standardized table '{dataset_name}.{table_name}' from '{path}': {exc}"
            ) from exc

        manifest_entry = self._manifest_entry(dataset_name, table_name)
        standardized_reference = None
        if manifest_entry is not None:
            standardized_reference = manifest_entry.get("standardized_reference")
            if isinstance(standardized_reference, dict):
                standardized_reference = dict(standardized_reference)
        return StandardizedTableLoadResult(
            dataset_name=dataset_name,
            table_name=table_name,
            path=path,
            dataframe=dataframe,
            standardized_reference=standardized_reference,
            manifest_entry=manifest_entry,
        )

    def load_optional_source_table(
        self,
        dataset_name: str,
        table_name: str,
        *,
        columns: list[str] | None = None,
    ) -> StandardizedTableLoadResult | None:
        """Load one optional standardized source table when present."""
        return self.load_source_table(
            dataset_name,
            table_name,
            columns=columns,
            required=False,
        )

    def source_table_path(self, spec: TableStandardizationSpec) -> Path:
        """Return the standardized raw-table Parquet path for one spec."""
        return self.settings.standardized_root / spec.dataset_name / f"{spec.table_name}.parquet"

    def analytical_artifact_path(self, artifact_name: str) -> Path:
        """Return the standardized analytical artifact path."""
        return self.settings.standardized_root / f"{artifact_name}.parquet"

    def source_table_standardized_reference(
        self,
        dataset_name: str,
        table_name: str,
    ) -> dict[str, object] | None:
        """Return the standardized reference metadata for one source table when available."""
        manifest_entry = self._manifest_entry(dataset_name, table_name)
        if manifest_entry is None:
            return None
        standardized_reference = manifest_entry.get("standardized_reference")
        if isinstance(standardized_reference, dict):
            return dict(standardized_reference)
        return None

    def source_table_row_count(
        self,
        dataset_name: str,
        table_name: str,
    ) -> int | None:
        """Return the manifest row count for one standardized source table when available."""
        manifest_entry = self._manifest_entry(dataset_name, table_name)
        if manifest_entry is None:
            return None
        row_count = manifest_entry.get("row_count")
        return int(row_count) if row_count is not None else None

    def load_analytical_artifact(self, artifact_name: str) -> pd.DataFrame:
        """Load one persisted analytical artifact from Parquet."""
        path = self.analytical_artifact_path(artifact_name)
        if not path.exists():
            raise FileNotFoundError(f"Expected analytical artifact at '{path}', but it does not exist.")
        return pd.read_parquet(path)

    def _manifest_entry(self, dataset_name: str, table_name: str) -> dict[str, object] | None:
        payload = self._load_latest_manifest_payload()
        for entry in payload.get("tables", []):
            if entry.get("dataset_name") == dataset_name and entry.get("table_name") == table_name:
                return dict(entry)
        return None

    def _load_latest_manifest_payload(self) -> dict[str, object]:
        if self._latest_manifest_payload is not None:
            return self._latest_manifest_payload

        manifest_path = self.settings.standardized_manifest_path
        if not manifest_path.exists():
            self._latest_manifest_payload = {"tables": []}
            return self._latest_manifest_payload

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                self._latest_manifest_payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise DataLoadError(
                f"Failed to load standardized manifest at '{manifest_path}': {exc}"
            ) from exc
        return self._latest_manifest_payload
