"""Dataset discovery and manifest helpers for the clinical and ED demos."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.schemas import (
    CLINICAL_CORE_TABLE_SCHEMAS,
    CLINICAL_OPTIONAL_TABLE_SCHEMAS,
    ED_CORE_TABLE_SCHEMAS,
    ED_OPTIONAL_TABLE_SCHEMAS,
    TABLE_KEY_COLUMNS,
)


@dataclass(frozen=True)
class TableManifest:
    """Availability and lightweight metadata for one raw table."""

    dataset_name: str
    table_name: str
    path: Path
    present: bool
    required: bool
    row_count: int | None
    columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest for one demo dataset directory."""

    dataset_name: str
    root: Path
    table_group_dir: str
    tables: tuple[TableManifest, ...]

    @property
    def missing_required_tables(self) -> list[str]:
        return [table.table_name for table in self.tables if table.required and not table.present]

    @property
    def missing_optional_tables(self) -> list[str]:
        return [table.table_name for table in self.tables if not table.required and not table.present]


def build_dataset_manifests(settings: Settings) -> dict[str, DatasetManifest]:
    """Inspect the configured demo roots without changing the current pipeline behavior."""
    return {
        "clinical": _build_dataset_manifest(
            dataset_name="clinical",
            dataset_root=settings.clinical_data_root,
            table_group_dir=settings.hosp_dir_name,
            required_tables=CLINICAL_CORE_TABLE_SCHEMAS,
            optional_tables=CLINICAL_OPTIONAL_TABLE_SCHEMAS,
            file_extension=settings.file_extension,
        ),
        "ed": _build_dataset_manifest(
            dataset_name="ed",
            dataset_root=settings.ed_data_root,
            table_group_dir=settings.ed_dir_name,
            required_tables=ED_CORE_TABLE_SCHEMAS,
            optional_tables=ED_OPTIONAL_TABLE_SCHEMAS,
            file_extension=settings.file_extension,
        ),
    }


def summarize_dataset_manifests(manifests: dict[str, DatasetManifest]) -> list[str]:
    """Render compact human-readable manifest lines for CLI output."""
    summary_lines: list[str] = []
    for dataset_name, manifest in manifests.items():
        summary_lines.append(
            f"{dataset_name}: root={manifest.root}, tables_present="
            f"{sum(1 for table in manifest.tables if table.present)}/{len(manifest.tables)}"
        )
        for table in manifest.tables:
            row_text = "missing"
            if table.present and table.row_count is not None:
                row_text = f"{table.row_count:,} rows"
            key_text = ", ".join(table.key_columns) if table.key_columns else "n/a"
            summary_lines.append(
                f"  - {table.table_name}: {row_text}, key_columns={key_text}"
            )
        if manifest.missing_optional_tables:
            summary_lines.append(
                f"  - missing_optional_tables={', '.join(manifest.missing_optional_tables)}"
            )
    return summary_lines


def _build_dataset_manifest(
    *,
    dataset_name: str,
    dataset_root: Path,
    table_group_dir: str,
    required_tables: dict[str, list[str]],
    optional_tables: dict[str, list[str]],
    file_extension: str,
) -> DatasetManifest:
    tables: list[TableManifest] = []
    for table_name, required_columns in required_tables.items():
        tables.append(
            _inspect_table(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                table_group_dir=table_group_dir,
                table_name=table_name,
                required=True,
                required_columns=required_columns,
                file_extension=file_extension,
            )
        )
    for table_name, required_columns in optional_tables.items():
        tables.append(
            _inspect_table(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                table_group_dir=table_group_dir,
                table_name=table_name,
                required=False,
                required_columns=required_columns,
                file_extension=file_extension,
            )
        )
    return DatasetManifest(
        dataset_name=dataset_name,
        root=dataset_root,
        table_group_dir=table_group_dir,
        tables=tuple(tables),
    )


def _inspect_table(
    *,
    dataset_name: str,
    dataset_root: Path,
    table_group_dir: str,
    table_name: str,
    required: bool,
    required_columns: list[str],
    file_extension: str,
) -> TableManifest:
    path = dataset_root / table_group_dir / f"{table_name}{file_extension}"
    if not path.exists():
        return TableManifest(
            dataset_name=dataset_name,
            table_name=table_name,
            path=path,
            present=False,
            required=required,
            row_count=None,
            columns=tuple(),
            key_columns=tuple(TABLE_KEY_COLUMNS.get(table_name, [])),
            missing_required_columns=tuple(required_columns),
        )

    columns, row_count = _read_table_header_and_count(path)
    missing_required_columns = tuple(
        column for column in required_columns if column not in columns
    )
    return TableManifest(
        dataset_name=dataset_name,
        table_name=table_name,
        path=path,
        present=True,
        required=required,
        row_count=row_count,
        columns=columns,
        key_columns=tuple(TABLE_KEY_COLUMNS.get(table_name, [])),
        missing_required_columns=missing_required_columns,
    )


def _read_table_header_and_count(path: Path) -> tuple[tuple[str, ...], int]:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for _ in reader)
    return tuple(header), row_count
