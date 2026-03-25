"""QC report helpers for persisted analytical artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_semantics_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.encounter_medication_semantics import (
    calculate_first_scope_qc_metrics,
    calculate_first_scope_subset_metrics,
)
from opti_med.data_access.encounter_medication_state import (
    calculate_encounter_medication_state_qc_metrics,
)
from opti_med.data_access.medication_rxnorm_mapping import (
    extract_medication_rxnorm_candidate_rows_from_encounter_medication_state,
)
from opti_med.data_access.medication_events import MedicationEventsQcMetrics


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


def write_medication_rxnorm_mapping_qc_report(
    *,
    medication_rxnorm_mapping: pd.DataFrame,
    output_path: Path,
    encounter_medication_state: pd.DataFrame | None = None,
) -> Path:
    """Write one Markdown QC report for the 65+ RxNorm mapping artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_medication_rxnorm_mapping_qc_report(
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_medication_state=encounter_medication_state,
        ),
        encoding="utf-8",
    )
    return output_path


def write_first_scope_semantics_and_burden_qc_report(
    *,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
    encounter_medication_first_scope: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write one Markdown QC report for the 65+ semantics, burden, and first-scope artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_first_scope_semantics_and_burden_qc_report(
            encounter_medication_semantics=encounter_medication_semantics,
            encounter_medication_burden=encounter_medication_burden,
            encounter_medication_first_scope=encounter_medication_first_scope,
        ),
        encoding="utf-8",
    )
    return output_path


def write_encounter_and_medication_events_qc_report_from_metrics(
    *,
    encounter_index: pd.DataFrame,
    medication_events_metrics: MedicationEventsQcMetrics,
    output_path: Path,
) -> Path:
    """Write one Markdown QC report using streaming medication-events metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_encounter_and_medication_events_qc_report_from_metrics(
            encounter_index=encounter_index,
            medication_events_metrics=medication_events_metrics,
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


def build_medication_rxnorm_mapping_qc_report(
    *,
    medication_rxnorm_mapping: pd.DataFrame,
    encounter_medication_state: pd.DataFrame | None = None,
) -> str:
    """Render one Markdown QC report for the 65+ RxNorm mapping artifact."""
    metrics = calculate_medication_rxnorm_mapping_qc_metrics(
        medication_rxnorm_mapping=medication_rxnorm_mapping,
        encounter_medication_state=encounter_medication_state,
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Medication RxNorm Mapping 65plus QC",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Coverage",
        "",
        f"- mapping rows: {metrics['row_count']:,}",
        f"- matched rows: {metrics['matched_rows']:,} ({metrics['match_rate']:.2%})",
        f"- unresolved rows: {metrics['unresolved_rows']:,} ({metrics['unresolved_rate']:.2%})",
        f"- ambiguity rows: {metrics['ambiguity_rows']:,} ({metrics['ambiguity_rate']:.2%})",
        "",
        "## Lookup Status Mix",
        "",
    ]
    if metrics["lookup_status_counts"]:
        for status, count in sorted(metrics["lookup_status_counts"].items()):
            lines.append(f"- {status}: {count:,}")
    else:
        lines.append("- no rows")
    lines.extend(
        [
            "",
            "## API Call Mix",
            "",
        ]
    )
    if metrics["api_called_flag_counts"]:
        for flag_value, count in sorted(metrics["api_called_flag_counts"].items()):
            lines.append(f"- api_called_flag={flag_value}: {count:,}")
    else:
        lines.append("- no rows")
    lines.extend(
        [
            "",
            "## Top Unresolved Medication Strings",
            "",
        ]
    )
    if metrics["top_unresolved_medication_strings"]:
        for medication_string, count in metrics["top_unresolved_medication_strings"]:
            lines.append(f"- {medication_string}: {count:,}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_first_scope_semantics_and_burden_qc_report(
    *,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
    encounter_medication_first_scope: pd.DataFrame,
) -> str:
    """Render one Markdown QC report for the 65+ semantics, burden, and first-scope artifacts."""
    validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
    validate_encounter_medication_burden_artifact(encounter_medication_burden)
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    metrics = calculate_first_scope_qc_metrics(
        encounter_medication_semantics=encounter_medication_semantics,
        encounter_medication_burden=encounter_medication_burden,
    )
    first_scope_metrics = calculate_first_scope_subset_metrics(
        encounter_medication_first_scope=encounter_medication_first_scope,
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# First-Scope Semantics And Burden 65plus QC",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Artifact Rows",
        "",
        f"- encounter_medication_semantics_65plus rows: {metrics['row_count']:,}",
        f"- encounter_medication_burden_65plus rows: {len(encounter_medication_burden):,}",
        f"- encounter_medication_first_scope_65plus rows: {first_scope_metrics['row_count']:,}",
        (
            "- encounter_medication_first_scope_65plus active rows: "
            f"{first_scope_metrics['active_row_count']:,}"
        ),
        (
            "- encounter_medication_first_scope_65plus unique subjects: "
            f"{first_scope_metrics['unique_subject_count']:,}"
        ),
        (
            "- encounter_medication_first_scope_65plus unique encounters: "
            f"{first_scope_metrics['unique_encounter_count']:,}"
        ),
        "",
        "## Class Coverage",
        "",
    ]
    if metrics["class_counts_active"]:
        for class_name, count in sorted(metrics["class_counts_active"].items()):
            lines.append(f"- active {class_name}: {count:,}")
    else:
        lines.append("- no active rows")
    lines.extend(
        [
            "",
            "## Unresolved Class Rate",
            "",
            (
                "- unresolved rows in encounter_medication_semantics_65plus: "
                f"{metrics['unresolved_class_rows']:,}/{metrics['row_count']:,} "
                f"({metrics['unresolved_class_rate']:.2%})"
            ),
            "",
            "## Heuristic Comparison",
            "",
        ]
    )
    for class_name, comparison in metrics["heuristic_comparison_by_class"].items():
        lines.append(
            f"- {class_name}: standardized={comparison['standardized_positive']:,}, "
            f"heuristic={comparison['heuristic_positive']:,}, "
            f"overlap={comparison['overlap']:,}, "
            f"standardized_only={comparison['standardized_only']:,}, "
            f"heuristic_only={comparison['heuristic_only']:,}"
        )
    lines.extend(
        [
            "",
            "## First-Scope Rows By Class",
            "",
        ]
    )
    if first_scope_metrics["row_counts_by_class"]:
        for class_name, count in sorted(first_scope_metrics["row_counts_by_class"].items()):
            lines.append(f"- {class_name}: {count:,}")
    else:
        lines.append("- no rows")
    lines.extend(
        [
            "",
            "## Active First-Scope Rows By Class",
            "",
        ]
    )
    if first_scope_metrics["active_row_counts_by_class"]:
        for class_name, count in sorted(first_scope_metrics["active_row_counts_by_class"].items()):
            lines.append(f"- {class_name}: {count:,}")
    else:
        lines.append("- no active rows")
    lines.extend(
        [
            "",
            "## Duplicate Therapy Signals",
            "",
            (
                "- exact same-class duplicate-therapy signal rows: "
                f"{metrics['exact_duplicate_therapy_signal_rows']:,}"
            ),
            (
                "- exact same-class duplicate-therapy signal groups: "
                f"{metrics['exact_duplicate_therapy_signal_groups']:,}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def calculate_medication_rxnorm_mapping_qc_metrics(
    *,
    medication_rxnorm_mapping: pd.DataFrame,
    encounter_medication_state: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Calculate compact QC metrics for the 65+ RxNorm mapping artifact."""
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    if encounter_medication_state is not None:
        validate_encounter_medication_state_artifact(encounter_medication_state)

    row_count = int(len(medication_rxnorm_mapping))
    lookup_status = medication_rxnorm_mapping["lookup_status"].fillna("").astype(str)
    matched_rows = int(
        lookup_status.isin({"resolved", "resolved_term_only_no_ingredient"}).sum()
    )
    unresolved_rows = int(lookup_status.str.startswith("unresolved").sum())
    ambiguity_rows = int(
        (
            pd.to_numeric(
                medication_rxnorm_mapping["ambiguous_match_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        ).sum()
    )
    unresolved_keys = set(
        medication_rxnorm_mapping.loc[
            lookup_status.str.startswith("unresolved"),
            "medication_query_key",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )
    if encounter_medication_state is not None and unresolved_keys:
        candidate_rows = extract_medication_rxnorm_candidate_rows_from_encounter_medication_state(
            encounter_medication_state
        )
        unresolved_source = candidate_rows.loc[
            candidate_rows["medication_query_key"].astype(str).isin(unresolved_keys)
        ].copy()
        display_series = unresolved_source["medication_raw"].fillna(
            unresolved_source["medication_normalized"]
        )
        top_unresolved = [
            (str(label), int(count))
            for label, count in display_series.value_counts(dropna=False).head(20).items()
        ]
    else:
        display_series = medication_rxnorm_mapping.loc[
            lookup_status.str.startswith("unresolved"),
            "medication_raw",
        ].fillna(medication_rxnorm_mapping["medication_normalized"])
        top_unresolved = [
            (str(label), int(count))
            for label, count in display_series.value_counts(dropna=False).head(20).items()
        ]
    return {
        "row_count": row_count,
        "matched_rows": matched_rows,
        "match_rate": (matched_rows / row_count) if row_count else 0.0,
        "unresolved_rows": unresolved_rows,
        "unresolved_rate": (unresolved_rows / row_count) if row_count else 0.0,
        "ambiguity_rows": ambiguity_rows,
        "ambiguity_rate": (ambiguity_rows / row_count) if row_count else 0.0,
        "lookup_status_counts": lookup_status.value_counts(dropna=False).to_dict(),
        "api_called_flag_counts": (
            medication_rxnorm_mapping["api_called_flag"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        ),
        "top_unresolved_medication_strings": top_unresolved,
    }


def build_encounter_and_medication_events_qc_report_from_metrics(
    *,
    encounter_index: pd.DataFrame,
    medication_events_metrics: MedicationEventsQcMetrics,
) -> str:
    """Render one Markdown QC report using streaming medication-events metrics."""
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
        f"- medication_events rows: {medication_events_metrics.row_count:,}",
        f"- medication_events unique subjects: {medication_events_metrics.unique_subject_count:,}",
        f"- medication_events unique medication_event_ids: {medication_events_metrics.unique_medication_event_id_count:,}",
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
    medication_row_count = medication_events_metrics.row_count
    for column_name in [
        "subject_id",
        "encounter_id",
        "medication_event_id",
        "medication_normalized",
        "medication_prestandardized_text",
        "event_time",
        "event_source_category",
    ]:
        if medication_row_count == 0:
            lines.append(f"- {column_name}: rows=0, null_rate=n/a")
            continue
        null_count = medication_events_metrics.null_counts.get(column_name, medication_row_count)
        lines.append(f"- {column_name}: null_rows={null_count:,} ({(null_count / medication_row_count):.2%})")
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
    if medication_events_metrics.event_source_category_counts:
        for value, count in sorted(medication_events_metrics.event_source_category_counts.items()):
            lines.append(f"- event_source_category={value}: {count:,}")
    else:
        lines.append("- event_source_category: no rows")
    lines.extend(
        [
            "",
            f"- order_enrichment_applied_flag=1 rows: {medication_events_metrics.order_enrichment_applied_count:,}",
            "",
            "## Key Collisions",
            "",
            f"- encounter_index duplicate encounter_id rows: {_duplicate_count(encounter_index, ['encounter_id']):,}",
            f"- medication_events duplicate medication_event_id rows: {medication_events_metrics.duplicate_medication_event_id_count:,}",
            (
                "- medication_events duplicate composite rows "
                "(subject_id, encounter_id, medication_event_type, medication_normalized, event_time): "
                f"{medication_events_metrics.duplicate_composite_count:,}"
            ),
            "",
            "## Build Metadata",
            "",
            f"- encounter_index build run ids: {_distinct_values(encounter_index, 'encounter_index_build_run_id')}",
            f"- medication_events build run ids: [{', '.join(medication_events_metrics.build_run_ids)}]",
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
