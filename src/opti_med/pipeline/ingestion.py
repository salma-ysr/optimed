"""Full-data raw discovery and standardized ingestion pipeline."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.contracts.pipeline import RawTableReference, StandardizedTableReference
from opti_med.data_access.exceptions import DataLoadError
from opti_med.standardized.specs import (
    GLOBAL_FLOAT_COLUMNS,
    GLOBAL_INTEGER_COLUMNS,
    SOURCE_MANIFEST_METADATA_COLUMNS,
    TableStandardizationSpec,
    iter_table_specs,
)
from opti_med.standardized.writer import ParquetStandardizedWriter


SUPPORTED_SOURCE_EXTENSIONS: tuple[str, ...] = (".csv.gz", ".csv")
VALID_INGESTION_BEHAVIORS: tuple[str, ...] = ("overwrite", "incremental")


@dataclass(frozen=True, slots=True)
class RawTableDiscovery:
    """Raw-table inspection metadata captured before ingestion."""

    dataset_name: str
    table_name: str
    required: bool
    present: bool
    source_path: Path
    detected_extension: str | None
    source_size_bytes: int | None
    source_last_modified_at: str | None
    source_signature: str | None
    raw_columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    duplicate_source_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngestedTableManifestEntry:
    """One table entry in the persisted ingestion manifest."""

    dataset_name: str
    table_name: str
    required: bool
    status: str
    source_path: str
    detected_extension: str | None
    row_count: int | None
    schema_summary: dict[str, str]
    raw_columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    duplicate_source_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    output_path: str | None
    source_size_bytes: int | None
    source_last_modified_at: str | None
    source_signature: str | None
    raw_reference: dict[str, object] | None
    standardized_reference: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class IngestionManifest:
    """Persisted summary for one full-data ingestion run."""

    run_id: str
    started_at: str
    completed_at: str
    ingestion_behavior: str
    file_extension_hint: str
    clinical_root: str
    ed_root: str
    standardized_root: str
    tables: tuple[IngestedTableManifestEntry, ...]
    manifest_path: Path


class FullDataIngestionPipeline:
    """Discover, validate, standardize, and persist the required full-data tables."""

    def __init__(
        self,
        settings: Settings,
        *,
        clinical_root: Path | None = None,
        ed_root: Path | None = None,
        standardized_root: Path | None = None,
        ingestion_behavior: str | None = None,
    ) -> None:
        self.settings = settings
        self.clinical_root = clinical_root or settings.raw_clinical_data_root
        self.ed_root = ed_root or settings.raw_ed_data_root
        self.standardized_root = standardized_root or settings.standardized_root
        self.ingestion_behavior = ingestion_behavior or settings.ingestion_behavior
        if self.clinical_root is None:
            raise DataLoadError(
                "Full-data ingestion requires an explicit clinical raw root. "
                "Pass --clinical-root or set OPTI_MED_RAW_CLINICAL_DATA_ROOT."
            )
        if self.ed_root is None:
            raise DataLoadError(
                "Full-data ingestion requires an explicit ED raw root. "
                "Pass --ed-root or set OPTI_MED_RAW_ED_DATA_ROOT."
            )
        if self.ingestion_behavior not in VALID_INGESTION_BEHAVIORS:
            valid = ", ".join(VALID_INGESTION_BEHAVIORS)
            raise DataLoadError(
                f"Unsupported ingestion behavior '{self.ingestion_behavior}'. Expected one of: {valid}"
            )
        self.writer = ParquetStandardizedWriter()

    def discover_sources(self) -> tuple[RawTableDiscovery, ...]:
        """Inspect all configured source tables without ingesting them."""
        discoveries = tuple(self._discover_one(spec) for spec in iter_table_specs())
        missing_required_tables = [
            f"{entry.dataset_name}.{entry.table_name}"
            for entry in discoveries
            if entry.required and not entry.present
        ]
        invalid_tables = [
            f"{entry.dataset_name}.{entry.table_name} missing columns: {', '.join(entry.missing_required_columns)}"
            for entry in discoveries
            if entry.present and entry.missing_required_columns
        ]
        if missing_required_tables or invalid_tables:
            details = []
            if missing_required_tables:
                details.append(
                    f"missing required tables: {', '.join(sorted(missing_required_tables))}"
                )
            if invalid_tables:
                details.append(
                    f"schema validation failures: {'; '.join(sorted(invalid_tables))}"
                )
            raise DataLoadError("Full-data ingestion discovery failed: " + " | ".join(details))
        return discoveries

    def run(self) -> IngestionManifest:
        """Execute one standardized ingestion run."""
        started_at = _utcnow_iso()
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        discoveries = self.discover_sources()
        previous_entries = self._load_previous_manifest_entries()
        manifest_entries: list[IngestedTableManifestEntry] = []

        for discovery in discoveries:
            if not discovery.present:
                manifest_entries.append(
                    IngestedTableManifestEntry(
                        dataset_name=discovery.dataset_name,
                        table_name=discovery.table_name,
                        required=discovery.required,
                        status="missing_optional",
                        source_path=str(discovery.source_path),
                        detected_extension=discovery.detected_extension,
                        row_count=None,
                        schema_summary={},
                        raw_columns=discovery.raw_columns,
                        normalized_columns=discovery.normalized_columns,
                        duplicate_source_columns=discovery.duplicate_source_columns,
                        missing_required_columns=discovery.missing_required_columns,
                        output_path=None,
                        source_size_bytes=discovery.source_size_bytes,
                        source_last_modified_at=discovery.source_last_modified_at,
                        source_signature=discovery.source_signature,
                        raw_reference=None,
                        standardized_reference=None,
                    )
                )
                continue

            previous = previous_entries.get((discovery.dataset_name, discovery.table_name))
            output_path = self._output_path_for(discovery)
            if self._should_skip_incremental(discovery, previous, output_path):
                manifest_entries.append(
                    IngestedTableManifestEntry(
                        dataset_name=discovery.dataset_name,
                        table_name=discovery.table_name,
                        required=discovery.required,
                        status="skipped_incremental",
                        source_path=str(discovery.source_path),
                        detected_extension=discovery.detected_extension,
                        row_count=previous.get("row_count"),
                        schema_summary=dict(previous.get("schema_summary", {})),
                        raw_columns=discovery.raw_columns,
                        normalized_columns=discovery.normalized_columns,
                        duplicate_source_columns=discovery.duplicate_source_columns,
                        missing_required_columns=discovery.missing_required_columns,
                        output_path=str(output_path),
                        source_size_bytes=discovery.source_size_bytes,
                        source_last_modified_at=discovery.source_last_modified_at,
                        source_signature=discovery.source_signature,
                        raw_reference=previous.get("raw_reference"),
                        standardized_reference=previous.get("standardized_reference"),
                    )
                )
                continue

            standardized_result = self._standardize_and_write(
                discovery,
                run_id=run_id,
                ingested_at=started_at,
                output_path=output_path,
            )
            raw_reference = self._build_raw_reference(
                discovery,
                run_id=run_id,
                row_count=standardized_result.row_count,
            )
            standardized_reference = self._build_standardized_reference(
                discovery,
                output_path=standardized_result.output_path,
                row_count=standardized_result.row_count,
                run_id=run_id,
                ingested_at=started_at,
                raw_reference=raw_reference,
            )
            manifest_entries.append(
                IngestedTableManifestEntry(
                    dataset_name=discovery.dataset_name,
                    table_name=discovery.table_name,
                    required=discovery.required,
                    status="ingested",
                    source_path=str(discovery.source_path),
                    detected_extension=discovery.detected_extension,
                    row_count=standardized_result.row_count,
                    schema_summary=standardized_result.schema_summary,
                    raw_columns=discovery.raw_columns,
                    normalized_columns=discovery.normalized_columns,
                    duplicate_source_columns=discovery.duplicate_source_columns,
                    missing_required_columns=discovery.missing_required_columns,
                    output_path=str(standardized_result.output_path),
                    source_size_bytes=discovery.source_size_bytes,
                    source_last_modified_at=discovery.source_last_modified_at,
                    source_signature=discovery.source_signature,
                    raw_reference=asdict(raw_reference),
                    standardized_reference=asdict(standardized_reference),
                )
            )

        completed_at = _utcnow_iso()
        manifest_dir = self.settings.manifest_root
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"run-{run_id}.json"
        manifest = IngestionManifest(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            ingestion_behavior=self.ingestion_behavior,
            file_extension_hint=self.settings.file_extension,
            clinical_root=str(self.clinical_root),
            ed_root=str(self.ed_root),
            standardized_root=str(self.standardized_root),
            tables=tuple(manifest_entries),
            manifest_path=manifest_path,
        )
        self._write_manifest(manifest)
        return manifest

    def _discover_one(self, spec: TableStandardizationSpec) -> RawTableDiscovery:
        root = self.clinical_root if spec.dataset_name == "clinical" else self.ed_root
        source_path, extension = self._resolve_source_path(root, spec)
        if source_path is None:
            fallback_path = root / spec.source_subdir / f"{spec.table_name}{self.settings.file_extension}"
            return RawTableDiscovery(
                dataset_name=spec.dataset_name,
                table_name=spec.table_name,
                required=spec.required,
                present=False,
                source_path=fallback_path,
                detected_extension=None,
                source_size_bytes=None,
                source_last_modified_at=None,
                source_signature=None,
                raw_columns=tuple(),
                normalized_columns=tuple(),
                duplicate_source_columns=tuple(),
                missing_required_columns=tuple(spec.required_columns),
            )

        stat = source_path.stat()
        raw_columns = _read_csv_header(source_path)
        normalized_columns, duplicate_source_columns = _normalize_and_deduplicate_columns(raw_columns)
        missing_required = tuple(
            column
            for column in spec.required_columns
            if column not in normalized_columns
        )
        return RawTableDiscovery(
            dataset_name=spec.dataset_name,
            table_name=spec.table_name,
            required=spec.required,
            present=True,
            source_path=source_path,
            detected_extension=extension,
            source_size_bytes=stat.st_size,
            source_last_modified_at=datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            source_signature=f"{stat.st_size}:{stat.st_mtime_ns}",
            raw_columns=raw_columns,
            normalized_columns=normalized_columns,
            duplicate_source_columns=duplicate_source_columns,
            missing_required_columns=missing_required,
        )

    def _resolve_source_path(
        self,
        root: Path,
        spec: TableStandardizationSpec,
    ) -> tuple[Path | None, str | None]:
        preferred_extensions = [self.settings.file_extension]
        preferred_extensions.extend(
            extension
            for extension in SUPPORTED_SOURCE_EXTENSIONS
            if extension != self.settings.file_extension
        )
        table_dir = root / spec.source_subdir
        for extension in preferred_extensions:
            candidate = table_dir / f"{spec.table_name}{extension}"
            if candidate.exists():
                return candidate, extension
        return None, None

    def _should_skip_incremental(
        self,
        discovery: RawTableDiscovery,
        previous: dict[str, object] | None,
        output_path: Path,
    ) -> bool:
        if self.ingestion_behavior != "incremental":
            return False
        if previous is None or not output_path.exists():
            return False
        return previous.get("source_signature") == discovery.source_signature

    def _standardize_and_write(
        self,
        discovery: RawTableDiscovery,
        *,
        run_id: str,
        ingested_at: str,
        output_path: Path,
    ):
        spec = _get_discovery_spec(discovery)
        batches = self._iter_standardized_batches(
            discovery,
            spec=spec,
            run_id=run_id,
            ingested_at=ingested_at,
        )
        empty_dataframe = self._build_empty_standardized_frame(
            discovery,
            spec=spec,
            run_id=run_id,
            ingested_at=ingested_at,
        )
        return self.writer.write_batches(
            batches,
            output_path=output_path,
            empty_dataframe=empty_dataframe,
        )

    def _iter_standardized_batches(
        self,
        discovery: RawTableDiscovery,
        *,
        spec: TableStandardizationSpec,
        run_id: str,
        ingested_at: str,
    ) -> Iterable[pd.DataFrame]:
        reader = pd.read_csv(
            discovery.source_path,
            chunksize=self.settings.ingestion_chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            yield self._normalize_chunk(
                chunk,
                discovery=discovery,
                spec=spec,
                run_id=run_id,
                ingested_at=ingested_at,
            )

    def _build_empty_standardized_frame(
        self,
        discovery: RawTableDiscovery,
        *,
        spec: TableStandardizationSpec,
        run_id: str,
        ingested_at: str,
    ) -> pd.DataFrame:
        empty = pd.DataFrame(columns=discovery.normalized_columns)
        return self._normalize_chunk(
            empty,
            discovery=discovery,
            spec=spec,
            run_id=run_id,
            ingested_at=ingested_at,
        )

    def _normalize_chunk(
        self,
        chunk: pd.DataFrame,
        *,
        discovery: RawTableDiscovery,
        spec: TableStandardizationSpec,
        run_id: str,
        ingested_at: str,
    ) -> pd.DataFrame:
        standardized = chunk.copy()
        if len(standardized.columns) != len(discovery.normalized_columns):
            raise DataLoadError(
                f"Unexpected column count while ingesting '{discovery.dataset_name}.{discovery.table_name}'. "
                "The source header and batch columns do not align."
            )
        standardized.columns = list(discovery.normalized_columns)

        for column_name in spec.expected_columns:
            if column_name not in standardized.columns:
                standardized[column_name] = pd.NA

        ordered_columns = list(spec.expected_columns)
        ordered_columns.extend(
            column_name
            for column_name in discovery.normalized_columns
            if column_name not in spec.expected_columns
        )
        standardized = standardized.reindex(columns=ordered_columns)

        for column_name in standardized.columns:
            if column_name in SOURCE_MANIFEST_METADATA_COLUMNS:
                continue
            try:
                standardized[column_name] = self._cast_series(
                    standardized[column_name],
                    column_name=column_name,
                    spec=spec,
                )
            except Exception as exc:
                raise DataLoadError(
                    "Failed to cast standardized column "
                    f"'{column_name}' while ingesting "
                    f"'{discovery.dataset_name}.{discovery.table_name}'."
                ) from exc

        standardized["_source_dataset"] = pd.Series(
            [discovery.dataset_name] * len(standardized),
            dtype="string",
        )
        standardized["_source_table"] = pd.Series(
            [discovery.table_name] * len(standardized),
            dtype="string",
        )
        standardized["_source_file"] = pd.Series(
            [str(discovery.source_path)] * len(standardized),
            dtype="string",
        )
        standardized["_source_file_size_bytes"] = pd.Series(
            [discovery.source_size_bytes] * len(standardized),
            dtype="Int64",
        )
        standardized["_source_last_modified_at"] = pd.Series(
            [discovery.source_last_modified_at] * len(standardized),
            dtype="string",
        )
        standardized["_ingestion_run_id"] = pd.Series(
            [run_id] * len(standardized),
            dtype="string",
        )
        standardized["_ingested_at"] = pd.Series(
            [ingested_at] * len(standardized),
            dtype="string",
        )
        return standardized

    def _cast_series(
        self,
        series: pd.Series,
        *,
        column_name: str,
        spec: TableStandardizationSpec,
    ) -> pd.Series:
        if _is_timestamp_column(column_name, spec):
            return pd.to_datetime(series, errors="coerce")
        if _is_integer_column(column_name, spec):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if _is_float_column(column_name, spec):
            return pd.to_numeric(series, errors="coerce").astype("Float64")
        return series.astype("string")

    def _build_raw_reference(
        self,
        discovery: RawTableDiscovery,
        run_id: str,
        *,
        row_count: int | None = None,
    ) -> RawTableReference:
        source_snapshot_id = f"{discovery.dataset_name}:{discovery.table_name}:{discovery.source_signature}"
        return RawTableReference(
            source_system=discovery.dataset_name,
            raw_table_name=discovery.table_name,
            source_snapshot_id=source_snapshot_id,
            artifact_uri=str(discovery.source_path),
            file_format=discovery.detected_extension,
            extract_run_id=run_id,
            provenance={
                "source_row_count": row_count,
                "source_last_modified_at": discovery.source_last_modified_at,
                "source_checksum": discovery.source_signature,
                "source_file_size_bytes": discovery.source_size_bytes,
            },
        )

    def _build_standardized_reference(
        self,
        discovery: RawTableDiscovery,
        *,
        output_path: Path,
        row_count: int,
        run_id: str,
        ingested_at: str,
        raw_reference: RawTableReference,
    ) -> StandardizedTableReference:
        return StandardizedTableReference(
            standardized_table_name=discovery.table_name,
            standardized_snapshot_id=f"{run_id}:{discovery.dataset_name}:{discovery.table_name}",
            artifact_uri=str(output_path),
            build_run_id=run_id,
            upstream_raw_table_ids=(raw_reference.source_snapshot_id,),
            schema_version="v1",
            provenance={
                "standardization_version": "v1",
                "record_count": row_count,
                "build_timestamp": ingested_at,
                "dataset_name": discovery.dataset_name,
            },
        )

    def _output_path_for(self, discovery: RawTableDiscovery) -> Path:
        dataset_dir = self.standardized_root / discovery.dataset_name
        return dataset_dir / f"{discovery.table_name}{self.settings.standardized_file_extension}"

    def _load_previous_manifest_entries(self) -> dict[tuple[str, str], dict[str, object]]:
        latest_manifest_path = self.settings.standardized_manifest_path
        if not latest_manifest_path.exists():
            return {}
        with latest_manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            (entry["dataset_name"], entry["table_name"]): entry
            for entry in payload.get("tables", [])
        }

    def _write_manifest(self, manifest: IngestionManifest) -> None:
        payload = {
            "run_id": manifest.run_id,
            "started_at": manifest.started_at,
            "completed_at": manifest.completed_at,
            "ingestion_behavior": manifest.ingestion_behavior,
            "file_extension_hint": manifest.file_extension_hint,
            "clinical_root": manifest.clinical_root,
            "ed_root": manifest.ed_root,
            "standardized_root": manifest.standardized_root,
            "tables": [asdict(entry) for entry in manifest.tables],
        }
        with manifest.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.settings.standardized_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.standardized_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def summarize_ingestion_manifest(manifest: IngestionManifest) -> list[str]:
    """Render concise CLI-friendly status lines from one ingestion manifest."""
    lines = [
        f"run_id={manifest.run_id}",
        f"clinical_root={manifest.clinical_root}",
        f"ed_root={manifest.ed_root}",
        f"standardized_root={manifest.standardized_root}",
    ]
    for entry in manifest.tables:
        row_count_text = "n/a" if entry.row_count is None else f"{entry.row_count:,}"
        lines.append(
            f"{entry.dataset_name}.{entry.table_name}: status={entry.status}, rows={row_count_text}"
        )
    return lines


def _read_csv_header(path: Path) -> tuple[str, ...]:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        return tuple(next(reader, []))


def _normalize_and_deduplicate_columns(columns: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    seen: Counter[str] = Counter()
    duplicates: list[str] = []
    normalized_columns: list[str] = []
    for raw_column in columns:
        normalized = _normalize_column_name(raw_column)
        seen[normalized] += 1
        if seen[normalized] > 1:
            duplicates.append(normalized)
            normalized = f"{normalized}__duplicate_{seen[normalized]}"
        normalized_columns.append(normalized)
    return tuple(normalized_columns), tuple(duplicates)


def _normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def _is_timestamp_column(column_name: str, spec: TableStandardizationSpec) -> bool:
    if column_name in spec.timestamp_columns:
        return True
    return column_name.endswith("time") or column_name.endswith("date") or column_name == "dod"


def _is_integer_column(column_name: str, spec: TableStandardizationSpec) -> bool:
    return column_name in spec.integer_columns or column_name in GLOBAL_INTEGER_COLUMNS


def _is_float_column(column_name: str, spec: TableStandardizationSpec) -> bool:
    return column_name in spec.float_columns or column_name in GLOBAL_FLOAT_COLUMNS


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_discovery_spec(discovery: RawTableDiscovery) -> TableStandardizationSpec:
    for spec in iter_table_specs():
        if spec.dataset_name == discovery.dataset_name and spec.table_name == discovery.table_name:
            return spec
    raise KeyError(f"Unknown discovery spec for '{discovery.dataset_name}.{discovery.table_name}'")
