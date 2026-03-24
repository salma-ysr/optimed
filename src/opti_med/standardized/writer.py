"""Durable persistence helpers for standardized analytical tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from opti_med.data_access.exceptions import DataLoadError


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataLoadError(
            "Parquet persistence requires the optional dependency 'pyarrow'. "
            "Install project dependencies before running full-data ingestion."
        ) from exc
    return pa, pq


@dataclass(frozen=True, slots=True)
class ParquetWriteResult:
    """Persisted Parquet artifact metadata."""

    output_path: Path
    row_count: int
    schema_summary: dict[str, str]


class ParquetStandardizedWriter:
    """Write standardized pandas chunks to one durable Parquet artifact."""

    def __init__(self, *, compression: str = "snappy") -> None:
        self.compression = compression

    def write_batches(
        self,
        batches: Iterable[pd.DataFrame],
        *,
        output_path: Path,
        empty_dataframe: pd.DataFrame,
    ) -> ParquetWriteResult:
        """Write one or more normalized batches to Parquet."""
        pa, pq = _require_pyarrow()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        if temp_path.exists():
            temp_path.unlink()

        writer = None
        arrow_schema = None
        row_count = 0
        schema_summary: dict[str, str] = {}
        try:
            for batch in batches:
                if batch.empty and writer is not None:
                    continue
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path,
                        table.schema,
                        compression=self.compression,
                    )
                    arrow_schema = table.schema
                    schema_summary = {
                        field.name: str(field.type)
                        for field in table.schema
                    }
                else:
                    table = pa.Table.from_pandas(
                        batch,
                        schema=arrow_schema,
                        preserve_index=False,
                    )
                writer.write_table(table)
                row_count += len(batch)

            if writer is None:
                empty_table = pa.Table.from_pandas(empty_dataframe, preserve_index=False)
                writer = pq.ParquetWriter(
                    temp_path,
                    empty_table.schema,
                    compression=self.compression,
                )
                arrow_schema = empty_table.schema
                writer.write_table(empty_table)
                schema_summary = {
                    field.name: str(field.type)
                    for field in empty_table.schema
                }
        finally:
            if writer is not None:
                writer.close()

        temp_path.replace(output_path)
        return ParquetWriteResult(
            output_path=output_path,
            row_count=row_count,
            schema_summary=schema_summary,
        )
