"""QC report helpers for persisted analytical artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from opti_med.data_access.encounter_medication_state import (
    calculate_encounter_medication_state_qc_metrics,
)


def write_encounter_and_medication_events_qc_report(
    *,
    encounter_index: pd.DataFrame,
    medication_events: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write one Markdown QC report for the two analytical artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_encounter_and_medication_events_qc_report(
            encounter_index=encounter_index,
            medication_events=medication_events,
        ),
        encoding="utf-8",
    )
    return output_path


def build_encounter_and_medication_events_qc_report(
    *,
    encounter_index: pd.DataFrame,
    medication_events: pd.DataFrame,
) -> str:
    """Render one Markdown QC report for the two analytical artifacts."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Encounter And Medication Events QC",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Row Counts",
        "",
        f"- encounter_index rows: {len(encounter_index):,}",
        f"- encounter_index unique subjects: {encounter_index['subject_id'].nunique():,}" if not encounter_index.empty else "- encounter_index unique subjects: 0",
        f"- encounter_index unique encounter_ids: {encounter_index['encounter_id'].nunique():,}" if not encounter_index.empty else "- encounter_index unique encounter_ids: 0",
        f"- medication_events rows: {len(medication_events):,}",
        f"- medication_events unique subjects: {medication_events['subject_id'].nunique():,}" if not medication_events.empty else "- medication_events unique subjects: 0",
        f"- medication_events unique medication_event_ids: {medication_events['medication_event_id'].nunique():,}" if not medication_events.empty else "- medication_events unique medication_event_ids: 0",
        "",
        "## Null Rates",
        "",
        "### Encounter Index",
        "",
    ]
    lines.extend(_null_rate_lines(encounter_index, ["subject_id", "encounter_id", "hadm_id", "stay_id", "encounter_start", "encounter_end"]))
    lines.extend(
        [
            "",
            "### Medication Events",
            "",
        ]
    )
    lines.extend(
        _null_rate_lines(
            medication_events,
            [
                "subject_id",
                "encounter_id",
                "medication_event_id",
                "medication_normalized",
                "medication_prestandardized_text",
                "event_time",
                "event_source_category",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Source Mix",
            "",
            "### Encounter Index",
            "",
        ]
    )
    lines.extend(_value_count_lines(encounter_index, "encounter_source"))
    lines.extend(
        [
            "",
            "### Medication Events",
            "",
        ]
    )
    lines.extend(_value_count_lines(medication_events, "event_source_category"))
    if not medication_events.empty and "order_enrichment_applied_flag" in medication_events:
        lines.extend(
            [
                "",
                f"- order_enrichment_applied_flag=1 rows: {int(medication_events['order_enrichment_applied_flag'].sum()):,}",
            ]
        )
    lines.extend(
        [
            "",
            "## Key Collisions",
            "",
            f"- encounter_index duplicate encounter_id rows: {_duplicate_count(encounter_index, ['encounter_id']):,}",
            f"- medication_events duplicate medication_event_id rows: {_duplicate_count(medication_events, ['medication_event_id']):,}",
            (
                "- medication_events duplicate composite rows "
                f"(subject_id, encounter_id, medication_event_type, medication_normalized, event_time): "
                f"{_duplicate_count(medication_events, ['subject_id', 'encounter_id', 'medication_event_type', 'medication_normalized', 'event_time']):,}"
            ),
            "",
            "## Build Metadata",
            "",
            f"- encounter_index build run ids: {_distinct_values(encounter_index, 'encounter_index_build_run_id')}",
            f"- medication_events build run ids: {_distinct_values(medication_events, 'medication_event_build_run_id')}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_encounter_medication_state_qc_report(
    *,
    encounter_medication_state: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write one Markdown QC report for the encounter-medication-state artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_encounter_medication_state_qc_report(
            encounter_medication_state=encounter_medication_state,
        ),
        encoding="utf-8",
    )
    return output_path


def build_encounter_medication_state_qc_report(
    *,
    encounter_medication_state: pd.DataFrame,
) -> str:
    """Render one Markdown QC report for the encounter-medication-state artifact."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics = calculate_encounter_medication_state_qc_metrics(encounter_medication_state)
    status_counts = metrics["counts_by_medication_status_at_review"]
    lines: list[str] = [
        "# Encounter Medication State QC",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Row Counts",
        "",
        f"- encounter_medication_state rows: {metrics['row_count']:,}",
        (
            "- encounter_medication_state unique subjects: "
            f"{encounter_medication_state['subject_id'].nunique():,}"
            if not encounter_medication_state.empty
            else "- encounter_medication_state unique subjects: 0"
        ),
        (
            "- encounter_medication_state unique encounters: "
            f"{encounter_medication_state['encounter_id'].nunique():,}"
            if not encounter_medication_state.empty
            else "- encounter_medication_state unique encounters: 0"
        ),
        "",
        "## Review-Time Guardrails",
        "",
        (
            "- rows with review_timestamp > discharge_boundary when discharge exists: "
            f"{metrics['review_timestamp_after_discharge_count']:,}"
        ),
        (
            "- rows capped to discharge boundary: "
            f"{int(encounter_medication_state.get('review_time_capped_to_discharge_flag', pd.Series(dtype=int)).sum()):,}"
            if not encounter_medication_state.empty
            else "- rows capped to discharge boundary: 0"
        ),
        (
            "- review_time_validated_flag=1 rows: "
            f"{int(encounter_medication_state.get('review_time_validated_flag', pd.Series(dtype=int)).sum()):,}"
            if not encounter_medication_state.empty
            else "- review_time_validated_flag=1 rows: 0"
        ),
        "",
        "## Key Collisions",
        "",
        f"- duplicate primary-key rows: {metrics['duplicate_key_count']:,}",
        "",
        "## Medication Status Mix",
        "",
    ]
    if status_counts:
        lines.extend(
            [
                f"- medication_status_at_review={status}: {count:,}"
                for status, count in status_counts.items()
            ]
        )
    else:
        lines.append("- medication_status_at_review: no rows")
    lines.extend(
        [
            "",
            "## Uncertain States",
            "",
            f"- activity_uncertain_at_review_time rows: {metrics['uncertain_state_count']:,}",
        ]
    )
    return "\n".join(lines) + "\n"


def _null_rate_lines(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    if dataframe.empty:
        return [f"- {column_name}: rows=0, null_rate=n/a" for column_name in columns]
    lines: list[str] = []
    row_count = len(dataframe)
    for column_name in columns:
        null_count = int(dataframe[column_name].isna().sum()) if column_name in dataframe else row_count
        lines.append(
            f"- {column_name}: null_rows={null_count:,} ({(null_count / row_count):.2%})"
        )
    return lines


def _value_count_lines(dataframe: pd.DataFrame, column_name: str) -> list[str]:
    if dataframe.empty:
        return [f"- {column_name}: no rows"]
    counts = dataframe[column_name].fillna("null").astype(str).value_counts(dropna=False)
    return [f"- {column_name}={value}: {int(count):,}" for value, count in counts.items()]


def _duplicate_count(dataframe: pd.DataFrame, columns: list[str]) -> int:
    if dataframe.empty:
        return 0
    return int(dataframe.duplicated(subset=columns, keep=False).sum())


def _distinct_values(dataframe: pd.DataFrame, column_name: str) -> str:
    if dataframe.empty or column_name not in dataframe:
        return "[]"
    values = sorted({str(value) for value in dataframe[column_name].dropna().astype(str).tolist()})
    return "[" + ", ".join(values) + "]"
